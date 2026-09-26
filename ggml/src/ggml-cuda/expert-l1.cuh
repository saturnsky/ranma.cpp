#pragma once

// VRAM tier of the expert cache: one fixed-address arena per size class and kind, packed by slot
// with the tensor's per-expert stride, plus one device slot table per layer (expert id -> slot or
// -1). The kernels read the table and the arena base through ggml_cuda_expert_lookup. Arena and
// table addresses never change after allocate(), because captured graphs hold them; only the slot
// contents and the tables are rewritten, and only by install() while nothing computes.
//
// With enable_vmm() (a redraw condition set, HIP only) an arena is a reserved virtual address
// range backed by physical handles instead of one allocation: resize() maps or unmaps handles at the
// top of a class's slot range, so the class split can be redrawn (expert-redraw.h) while every base
// address, and with it every captured graph, stays valid.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-host.cuh"
#include "expert-l2-ledger.h"
#include "expert-plan.h"
#include "expert-redraw.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Frozen kernel-facing lookup. host_* are used by exclusive mode and the host tier; they stay null
// while only the inclusive VRAM tier exists.
struct ggml_cuda_expert_lookup {
    const void    * data           = nullptr; // arena base of this kind, slot-major with the tensor's nb[2] stride
    const int32_t * slots          = nullptr; // device table of the layer, shared by all kinds
    const void    * host_data      = nullptr;
    const int32_t * host_slots     = nullptr;
    const uint64_t * host_addresses = nullptr;
};

namespace ggml_cuda_expert {

struct l1_install_stats {
    size_t retained = 0;
    size_t copied   = 0;   // slices (layer, expert) copied, all kinds
    size_t bytes    = 0;   // bytes written into the VRAM arena
    size_t d2h_bytes = 0;  // exclusive only: bytes written back to the host arena
    double ms       = 0.0;
};

// One install as the three steps of design section 7. stage() is pure: given the current tables and
// whether the mover may retain, it decides the slot of every selected expert and the ledger, and
// touches no device memory. execute() runs the mover copies, publish() makes the tables visible.
struct l1_transaction {
    bool valid = false;
    install_transaction install;
    expert_slot_table selected;
};

// Bytes of the zeroed tail every per-class, per-kind arena carries past its last slot. MMQ loads
// whole MMQ_ITER_K (256-element) K tiles of src0, so for a matrix whose row length is not a multiple
// of MATRIX_ROW_PADDING it reads past the last row of the last expert. Inside the arena that read
// lands in the next slot, which holds finite quantized data and is harmless because the matching
// src1 columns are zero; past the last slot it must land in zeros. The host buffer type pads every
// quantized tensor by exactly ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING) and
// zeroes it (ggml_backend_cuda_host_buffer_type_get_alloc_size in ggml-cuda.cu); the arena uses the
// same rule, with a 512-byte floor so that a zero-capacity class arena is
// still a real allocation.
size_t arena_tail_bytes(const geometry & geo, int cls, int kind);
size_t arena_tail_total(const geometry & geo);

class l1_arena {
public:
    explicit l1_arena(const geometry & geo);
    ~l1_arena();

    l1_arena(const l1_arena &) = delete;
    l1_arena & operator=(const l1_arena &) = delete;

    // Exactly once. capacities[class] slots per class, plus `spare_slots` free slots per class that
    // the exclusive exchange rotates through (0 for inclusive mode); the tables start empty (-1).
    // `stage_slots[class]` more slots follow the spares: the VRAM staging of early-route layers
    // (expert-hash-stage.cuh). No plan, install or slot count below ever includes them.
    bool allocate(const std::vector<int> & capacities, int device, int spare_slots = 0,
                  const std::vector<int> * stage_slots = nullptr);
    // The first staging slot of a class (after the static capacity and the spares). A VMM arena keeps
    // its staging slots at a fixed index above the largest slot range the class can grow to.
    int stage_base(int cls) const {
        if (!vmm_stage_base_.empty()) { return vmm_stage_base_[(size_t) cls]; }
        return capacities_[(size_t) cls] + spare_slots_;
    }

    // Before allocate(): back the arenas with HIP virtual memory, so that resize() can move capacity
    // between classes. `max_static[class]` bounds the static capacity a class may grow to (the address
    // range reserved), `handle_bytes` is the size of the physical handles. allocate() falls back to
    // plain allocations when the device or the build has no virtual memory management; vmm() tells.
    void enable_vmm(const std::vector<int> & max_static, size_t handle_bytes);
    bool vmm() const;
    const std::string & vmm_reason() const { return vmm_reason_; }
    size_t vmm_reserved_bytes() const;

    // Size-class redraw (expert-redraw.h), while nothing computes:
    //   - set_static_capacities: the static capacity the next plans and installs are bound to;
    //   - set_retire: the slot count per class the next install leaves (slots at and above it retire;
    //     empty = none), set_unequal: an exclusive mover runs the unequal exchange (expert-plan.h);
    //   - evacuate: moves kept residents down (and swaps them with dropped ones), all kinds, VRAM to VRAM;
    //   - resize: the new slot count per class; shrinking classes first, their top handles are
    //     unmapped and mapped again at the top of the growing classes; the new slots and the tails are
    //     zeroed. Every static home and spare of a shrinking class must be below its new count.
    struct resize_stats {
        size_t handles_moved = 0, handles_created = 0, handles_released = 0, units_replaced = 0;
        size_t copied_bytes = 0, zeroed_bytes = 0, mapped_before = 0, mapped_after = 0;
        double ms = 0.0;
    };
    void set_static_capacities(const std::vector<int> & capacities);
    void set_retire(const std::vector<int> & limit) { retire_ = limit; }
    void set_unequal(bool on) { unequal_ = on; }
    bool evacuate(const std::vector<evac_op> & ops, double & ms);
    bool resize(const std::vector<int> & slots, resize_stats & stats, std::string & why);
    struct vmm_state;   // expert-l1.cu
    bool allocated() const { return !layer_slots_.empty(); }

    // Exclusive mode and every finite tier: this arena and `host` together own the routed expert
    // bytes. Must be called after allocate() and before the first assignment.
    void attach_host(host_arena * host, int spare_slots, bool host_master = false);
    bool owns_host_storage() const { return host_ != nullptr; }

    // With a finite host tier the file owns the experts that are in neither table, so this
    // adopts the two tables as they are instead of demanding that every expert have an arena home.
    bool assign_tier(const std::vector<std::vector<int32_t>> & gpu_slots,
                     const expert_locations & homes, const install_layout & layout,
                     const std::vector<std::vector<int32_t>> & selected,
                     const std::vector<std::vector<int>> & gpu_spares);
    const std::vector<std::vector<int>> & gpu_spares() const { return gpu_spares_; }
    const expert_locations & locations() const { return homes_; }
    const install_layout & layout() const { return layout_; }
    // (layer, kind, location) -> payload address of a host or lent slice
    using location_resolver = std::function<void *(int, int, expert_location)>;
    void attach_locations(location_resolver resolver) { resolve_ = std::move(resolver); }
    void * host_address(int layer, int kind, int expert) const;

    // What the mover needs from the SSD tier to move a slice whose source is the file. Empty
    // without a tier, and then a transaction may not contain such a move.
    struct tier_reader {
        // may rewrite the slots of the reads; `address` takes the rewritten reads
        std::function<bool(std::vector<l2_read> &, std::string &)> read;
        std::function<const void *(const l2_read &)> address;
        int slots = 0;   // ring slots one batch may use
        // the tier reads a slice promoted to a host slot straight into that slot
        bool direct_host = false;
    };
    void attach_tier_reader(tier_reader reader) { tier_ = std::move(reader); }

    // Three-tier mode: the per (layer, kind) device address table of the SSD tier, which the kernels
    // take before the host slot table. Null without a tier.
    using address_source = std::function<const uint64_t * (int layer, int kind)>;
    void attach_addresses(address_source addresses) { addresses_ = std::move(addresses); }

    bool write_gpu_slice(int cls, int kind, int slot, const void * src, bool src_is_device);
    bool read_gpu_slice(int cls, int kind, int slot, void * dst);
    // Only the GPU fixture reads an arena slot back directly.
    char * gpu_slice_address(int cls, int kind, int slot) const { return gpu_slice(cls, kind, slot); }
    bool   sync_copies();

    // Exclusive mode, once: gives every routed expert a home. `selected` gets the VRAM slots in
    // (layer, expert) order, everything else gets a host slot in the same order, and the slots past
    // the capacities become the spares. Publishes both tables.
    bool assign_exclusive(const std::vector<std::vector<int32_t>> & selected);

    // Exclusive mode: reads or writes `size` bytes at `offset` of the routed expert tensor
    // (layer, kind) through whichever home owns each expert slice it touches. This is what the
    // exclusive buffer type's set_tensor and get_tensor are.
    bool logical_io(int layer, int kind, void * data, size_t offset, size_t size, bool write);

    size_t device_bytes() const { return allocated_bytes_; }
    const std::vector<int> & capacities() const { return capacities_; }
    // Arena slots per class: the static capacity and the exchange spares (after a redraw: what
    // resize() made it). Every slot index of the class is below this.
    std::vector<int> slot_counts() const;
    // Mirror of the device table that maps an expert to its VRAM slot, or -1.
    const std::vector<std::vector<int32_t>> & host_slots() const { return host_slots_; }
    // Exclusive mode: mirror of the device table that maps an expert to its host arena slot.
    const std::vector<std::vector<int32_t>> & arena_slots() const { return arena_slots_; }
    const std::vector<std::vector<int32_t>> & selected() const { return selected_; }

    // Slot-table publish helpers used by install(); public for the tests and for the transaction
    // code of later stages.
    ggml_cuda_expert_lookup lookup(int layer, int kind) const noexcept;

    // Byte size of one (layer, expert) slice per size class, all three kinds together.

    // Pure: no device memory is read or written, no state changes.
    void * slice_address(int layer, int cls, int kind, expert_location at, int expert) const;
    size_t move_from_file(const install_transaction & tx, size_t first);
    l1_transaction stage(const std::vector<std::vector<int32_t>> & selected, bool retain) const;
    // The one mover: invalidates the tables when this arena has no host master, then moves every
    // slice the transaction lists, whatever tier each end of a move is in.
    bool execute(const install_transaction & tx);
    // Makes the staged tables visible and adopts them as the current ones.
    bool publish(const l1_transaction & tx);

    // Makes `selected` resident: stage, execute, publish. With retain, experts that keep their slot
    // are not copied again. Synchronous on the copy stream; the caller has drained the compute streams.
    bool install(const std::vector<std::vector<int32_t>> & selected, bool retain, l1_install_stats & stats);

    // Reads one cached slice back and compares it with the host tensor. False means the arena holds
    // wrong bytes, which is a fatal stride/source assumption failure. Inclusive mode only: exclusive
    // mode has no host master to compare against, it uses the load-time digests below.
    bool verify_slice(int layer, int kind, int expert) const;
    // Verifies up to `max_slices` resident slices (all when 0). Returns the number checked, or -1 on a mismatch.
    long verify_resident(size_t max_slices) const;

    // Debug invariants of the current assignment, shared by both movers.
    // With the SSD tier an expert may live in neither arena, so the "exactly one home" half of the
    // check is dropped; the "never two homes, never a slot twice" half stays.
    bool verify_current_assignment(std::string & reason) const;

    // Reads one VRAM slot back into `dst`, which must hold the class/kind stride. Verification only.
    bool read_slice(int layer, int kind, int slot, void * dst) const;

    // Reads every device table back ([layer][expert]; -1 rows for layers outside the cache). The device
    // tables can differ from the host mirror: the VRAM staging of early-route layers points at its own slots.
    bool read_device_tables(std::vector<std::vector<int32_t>> & out) const;

private:
    bool publish_tables(const std::vector<std::vector<int32_t>> & slots);
    char * gpu_slice(int cls, int kind, int slot) const {
        return static_cast<char *>(class_data_[cls][kind]) + size_t(slot)*geo_.class_bytes[cls][kind];
    }

    const geometry & geo_;
    int device_ = -1;
    cudaStream_t copy_stream_ = nullptr;
    std::vector<int> capacities_;
    // size-class redraw
    std::unique_ptr<vmm_state> vmm_;
    std::string vmm_reason_;
    std::vector<int> vmm_stage_base_;                  // [class] fixed first staging slot of a VMM arena
    std::vector<int> slots_;                           // [class] slot counts once a redraw owns them
    std::vector<int> retire_;                          // [class] slot count the next install leaves
    bool unequal_ = false;
    bool allocate_vmm(const std::vector<int> & capacities, int spare_slots, const std::vector<int> * stage_slots);
    void free_vmm();
    std::vector<std::array<void *, 3>> class_data_;    // [class][kind]
    std::vector<int32_t *> layer_slots_;               // [layer] device tables
    std::vector<std::vector<int32_t>> host_slots_;     // [layer][expert] mirror of the device tables
    std::vector<std::vector<int32_t>> selected_;       // [layer] sorted resident expert ids
    size_t allocated_bytes_ = 0;

    // exclusive mode
    host_arena * host_ = nullptr;
    address_source addresses_;
    tier_reader    tier_;
    int          spare_slots_ = 0;
    std::vector<std::vector<int32_t>> arena_slots_;    // [layer][expert] host arena slot or -1
    std::vector<int>                  host_capacities_;
    std::vector<std::vector<int>>     gpu_spares_;     // [class] free VRAM slots
    expert_locations homes_;
    install_layout layout_;
    bool host_master_ = true;
    location_resolver resolve_;
};

} // namespace ggml_cuda_expert
