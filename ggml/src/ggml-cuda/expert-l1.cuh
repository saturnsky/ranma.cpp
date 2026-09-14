#pragma once

// VRAM tier of the expert cache: one fixed-address arena per size class and kind, packed by slot
// with the tensor's per-expert stride, plus one device slot table per layer (expert id -> slot or
// -1). The kernels read the table and the arena base through ggml_cuda_expert_lookup. Arena and
// table addresses never change after allocate(), because captured graphs hold them; only the slot
// contents and the tables are rewritten, and only by install() while nothing computes.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-host.cuh"
#include "expert-plan.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Frozen kernel-facing lookup. host_* are used by exclusive mode; they stay null while only the
// inclusive VRAM tier exists.
struct ggml_cuda_expert_lookup {
    const void    * data       = nullptr; // arena base of this kind, slot-major with the tensor's nb[2] stride
    const int32_t * slots      = nullptr; // device table of the layer, shared by all kinds
    const void    * host_data  = nullptr;
    const int32_t * host_slots = nullptr;
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
    bool allocate(const std::vector<int> & capacities, int device, int spare_slots = 0);
    bool allocated() const { return !layer_slots_.empty(); }

    // Exclusive mode and every finite tier: this arena and `host` together own the routed expert
    // bytes. Must be called after allocate() and before the first assignment.
    void attach_host(host_arena * host, int spare_slots, bool host_master = false);
    bool owns_host_storage() const { return host_ != nullptr; }

    const std::vector<std::vector<int>> & gpu_spares() const { return gpu_spares_; }
    const expert_locations & locations() const { return homes_; }
    const install_layout & layout() const { return layout_; }
    using location_resolver = std::function<void *(int, int, expert_location)>;
    void attach_locations(location_resolver resolver) { resolve_ = std::move(resolver); }
    void * host_address(int layer, int kind, int expert) const;

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
    bool verify_current_assignment(std::string & reason) const;

    // Reads one VRAM slot back into `dst`, which must hold the class/kind stride. Verification only.
    bool read_slice(int layer, int kind, int slot, void * dst) const;

private:
    bool publish_tables(const std::vector<std::vector<int32_t>> & slots);
    char * gpu_slice(int cls, int kind, int slot) const {
        return static_cast<char *>(class_data_[cls][kind]) + size_t(slot)*geo_.class_bytes[cls][kind];
    }

    const geometry & geo_;
    int device_ = -1;
    cudaStream_t copy_stream_ = nullptr;
    std::vector<int> capacities_;
    std::vector<std::array<void *, 3>> class_data_;    // [class][kind]
    std::vector<int32_t *> layer_slots_;               // [layer] device tables
    std::vector<std::vector<int32_t>> host_slots_;     // [layer][expert] mirror of the device tables
    std::vector<std::vector<int32_t>> selected_;       // [layer] sorted resident expert ids
    size_t allocated_bytes_ = 0;

    // exclusive mode
    host_arena * host_ = nullptr;
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
