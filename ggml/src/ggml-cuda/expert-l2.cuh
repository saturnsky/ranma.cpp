#pragma once

// Finite host storage with demand reads from the original GGUF files.
// A GPU bitmap publishes the selected experts. The worker fills only misses and answers
// through a coherent address table. A done counter protects the slots until their last use.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-l2-class-ledger.h"
#include "expert-l2-ledger.h"
#include "expert-l2-relabel.h"
#include "expert-location.h"
#include "expert-os.h"
#include "expert-storage.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ggml_cuda_expert {

class l1_arena;

struct l2_config {
    size_t prefill_ring_bytes = 0;   // 0 = automatic
    size_t decode_ring_bytes  = 0;   // 0 = automatic
    int    queue_depth        = 4;
    int    worker_cpu         = -1;
    int    experts_used       = 1;
    uint64_t prefill_rows     = 512, decode_rows = 1;
    int64_t read_wait_ms      = 30000; // Existing CPU I/O queue deadline, never a GPU wait limit.
    bool   phase_rings        = false;   // a smaller ring during generation, lending its tail slots
    bool   verify             = false;
    // The class layout: the staging ring lives in the per storage class host arenas, next to the
    // residents and at their pitch, and every host resident is served through the address table.
    // False is the separate uniform-pitch ring with lent tail slots. The controller always sets it;
    // the default here keeps direct users (the fixtures) on the ring layout.
    bool   class_layout       = false;
    // Class layout: each storage class ring is this many times its floor, as far as the budget
    // allows (RANMA_EXPERT_L2_RING_FACTOR). 1 = the floors.
    double ring_factor        = 1.0;
    uint32_t log_mask         = 0;
    // Staged service: ubatches of at least this many rows publish their file reads one kind at a
    // time, in the order the graph multiplies them, so the kernels of a kind wait only for that
    // kind's slices while the next kind is still being read. 0 = one wait per layer for everything.
    int64_t staged_min_rows   = 0;
    bool    staged_drain      = false;   // complete every read of a kind before issuing the next kind
};

// Whether the staged service is on when RANMA_EXPERT_L2_STAGED is not set. On, it covers the
// ubatches with more rows than the decode bound, i.e. prompt processing; generation keeps the
// single wait. It reads the same bytes into the same slots and runs the same kernels.
static constexpr bool l2_staged_default = true;

// What the tier reports every so often under GGML_EXPERT_LOG_L2.
struct l2_counters {
    uint64_t repartitions = 0;
    uint64_t generations   = 0;
    uint64_t ssd_bytes     = 0;
    uint64_t ssd_reads     = 0;
    uint64_t ring_hits     = 0;
    uint64_t install_ssd_bytes = 0, install_ssd_reads = 0;
    uint64_t install_relabel_bytes = 0, install_ring_copy_bytes = 0;   // install moves from the file served by the ring
    uint64_t wait_ticks = 0, steady_ticks = 0, measured_layers = 0;
    uint64_t distinct = 0, vram_bytes = 0, host_bytes = 0, file_bytes = 0;
    uint64_t prompt_ubatches = 0, decode_ubatches = 0, decode_tokens = 0;
    uint64_t prompt_wait_ticks = 0, decode_wait_ticks = 0;
    double   service_ms    = 0.0;
    uint64_t verify_fills  = 0;
    uint64_t verify_bytes  = 0;
    uint64_t verify_bad    = 0;
    uint64_t owner_checks  = 0;
    uint64_t staged_generations = 0, staged_wait_ticks = 0;
};

// One relabel install (class layout). Slices are (layer, expert) pairs, bytes all three kinds.
struct l2_install_stats {
    size_t relabel_slices = 0, relabel_bytes = 0;       // host promotions that took the ring slot holding them
    size_t ring_copy_slices = 0, ring_copy_bytes = 0;   // VRAM promotions copied from a ring occupant
    size_t read_slices = 0, read_bytes = 0;             // moves from the file that were read
    size_t demoted = 0;                                 // residents that stayed in their slot as ring occupants
    size_t ring_before = 0, ring_kept = 0, ring_after = 0;   // ring occupants before, of those still there, after
    int    batches = 0;
};

// Class arena addresses. Lent residents use the common location map. With the class layout the
// index is the storage class and the address is that of slot 0 of the arena.
struct l2_host_geometry {
    std::vector<std::array<void *, 3>> device_base;  // [class][kind] device alias of the arena
    std::vector<std::array<void *, 3>> host_base;    // [class][kind] host address of the arena
};

class l2_tier {
public:
    l2_tier(const geometry & geo, const l2_config & cfg);
    ~l2_tier();

    l2_tier(const l2_tier &) = delete;
    l2_tier & operator=(const l2_tier &) = delete;

    // ---- sizing, before anything is allocated -------------------------------------------------
    size_t slot_pitch(int kind) const { return ring_pitch_[kind]; }
    size_t ring_stride() const { return ring_pitch_[0] + ring_pitch_[1] + ring_pitch_[2]; }
    int    prompt_slots() const { return prompt_slots_; }
    int    decode_slots() const { return decode_slots_; }
    size_t ring_bytes(int slots) const { return size_t(slots)*ring_stride(); }
    size_t metadata_bytes() const;
    // Bytes the budget must hold before a single expert can be resident.
    size_t fixed_bytes() const { return ring_total_bytes() + metadata_bytes(); }
    // The ring as allocated: the prompt ring, or with the class layout every class ring.
    size_t ring_total_bytes() const { return class_layout() ? storage_.ring_bytes() : ring_bytes(prompt_slots_); }
    bool   sized() const { return sized_; }
    const std::string & size_note() const { return size_note_; }

    // ---- the class layout ------------------------------------------------------------------------
    bool class_layout() const { return cfg_.class_layout; }
    double ring_factor() const { return cfg_.ring_factor; }
    // The storage classes, their pitches, resident ranges, rings and floors. Before allocate().
    void set_storage(const class_storage & storage);
    const class_storage & storage() const { return storage_; }
    const l2_class_ledger & class_ledger() const { return cledger_; }

    // ---- allocation -----------------------------------------------------------------------------
    bool allocate(int device);
    bool map();
    size_t host_bytes() const { return host_bytes_; }

    // ---- the loader ------------------------------------------------------------------------------
    void set_backing(int layer, int kind, int file_index, const char * path, uint64_t offset);
    // Opens the files not opened yet and checks the backing of the routed layers, or of the layers
    // flagged in `layers` only (a joint cache whose later models are not loaded yet).
    bool open_files(std::string & reason, const std::vector<uint8_t> * layers = nullptr);
    // False for an expert whose bytes the loader must skip.
    bool wanted(int layer, int expert) const;

    // ---- plans -----------------------------------------------------------------------------------
    // `keep_ring`: the install transaction already relabeled the slots (class layout); the ledger
    // must match `host` and the rings keep their contents. Otherwise the rings start empty.
    void set_homes(const std::vector<std::vector<int32_t>> & vram,
                   const expert_locations & host, const l2_host_geometry & host_geo, bool keep_ring = false);
    // The ring class that travels with an installed plan. Returns true when the size changed.
    bool set_prompt_ring(bool prompt);
    int  ring_count() const { return class_layout() ? cledger_.ring_total() : ledger_.ring_count(); }
    // Whether the current plan leaves any expert in the file. Recomputed with every plan; the
    // mailbox kernels run either way and cost nothing when no routed expert needs the worker.
    bool any_ssd() const { return any_ssd_; }
    const expert_locations & locations() const { return homes_; }
    // The payload address of a slice of `layer` at a host or lent location.
    void * host_address(int layer, int kind, int expert) const;
    void * location_address(int layer, int kind, expert_location at, bool device = false) const;
    // Clears the bytes after a slice written to `at`, which MMQ reads as the tail of the slice.
    void finish_write(int layer, int kind, expert_location at);
    // Publishes the address table again, e.g. after a joined model set the sector shifts of its
    // tensors. The worker must be stopped.
    void refresh_addresses();
    bool verify_resident(int layer, int kind, int expert, const void * data, std::string & reason) {
        return verify_slice(layer, kind, expert, data, reason);
    }
    // ---- the install transaction ------------------------------------------------------------------
    // Reads one slice out of the file into `dst`, through a bounce slot when `dst` is not sector
    // aligned (a host arena slot never is). Synchronous, used only while nothing computes.
    bool read_slice(int layer, int kind, int expert, void * dst, std::string & reason);

    // The common active prefix is never a lent source or destination. Install discards its tags.
    // With the class layout a batch takes at most one ring slot per move from each class ring.
    int install_read_slots() const;
    // Class layout: installs run through install_relabel.
    bool relabel_installs() const { return class_layout(); }
    // Runs the moves of `tx` (host locations `before` -> tx.host) by relabeling slots: reads only the
    // file promotions the rings do not hold, straight into their final slot, copies to and from VRAM
    // through `l1`. The worker must be stopped and the device idle. Publish with set_homes(..., true).
    bool install_relabel(const install_transaction & tx, const expert_locations & before, l1_arena & l1,
                         l2_install_stats & stats, std::string & reason);
    // Reads a batch of install slices. With the class layout a `direct` read goes straight into its
    // host resident slot, every other read into a ring slot of its storage class that this picks,
    // and `slot` comes back as the storage slot read into.
    bool read_install(std::vector<l2_read> & reads, std::string & reason);
    const void * read_address(const l2_read & read) const {
        return static_cast<char *>(read_slot(read)) + backing_[read.layer][read.kind].shift;
    }

    // ---- the hot path ------------------------------------------------------------------------------
    const uint64_t * addresses(int layer, int kind) const;
    void publish_and_wait(int layer, const ggml_tensor * ids, cudaStream_t stream);
    // Staged service only: orders the kernel that reads `kind` of `layer` after that kind's reads.
    // A no-op for the first kind, which the route waits for, and for a layer routed unstaged.
    bool staged() const { return cfg_.staged_min_rows > 0; }
    void wait_kind(int layer, int kind, cudaStream_t stream);
    void mark_done(int layer, cudaStream_t stream);

    void start_worker();
    void stop_worker();
    bool worker_running() const { return running_.load(std::memory_order_relaxed); }

    const l2_counters & counters() const { return counters_; }
    void report(const char * what);
    void set_phase(bool prompt) { phase_.store(prompt ? 0 : 1); }
    double wait_ms() const { return steady_khz_ > 0 ? double(counters_.steady_ticks)/steady_khz_ : 0; }

private:
    struct backing {
        int      file   = -1;
        uint64_t offset = 0;
        size_t   shift  = 0;   // offset % io_alignment, constant for the whole tensor
    };

    size_t bitmap_words() const { return size_t((geo_.n_experts + 31)/32); }
    size_t bitmap_bytes() const { return (size_t) geo_.n_layers*bitmap_words()*sizeof(uint32_t); }
    std::vector<int> demanded_ids(int layer) const;
    void   account_layer(int layer, uint32_t seq, const std::vector<int> & ids);
    void   size_ring();
    void   publish_tables();
    void   publish_layer(int layer);
    void   publish_layer_kind(int layer, int kind);
    void   verify_addresses(int demanded_layer, const std::vector<int> & ids);
    uint64_t address_of(int layer, int kind, int expert) const;
    void * ring_host(int kind, int slot) const {
        return static_cast<char *>(ring_host_base_[kind]) + size_t(slot)*ring_pitch_[kind];
    }
    // Class layout: slot `slot` of the storage class of `layer` (slot start, not payload).
    void * storage_slot(int layer, int kind, int slot, bool device = false) const {
        const int s = storage_.storage_of[geo_.layer_class[layer]];
        const auto & bases = device ? host_geo_.device_base : host_geo_.host_base;
        return static_cast<char *>(bases[s][kind]) + size_t(slot)*storage_.pitch[s][kind];
    }
    // Where a read of either layout lands, and the pitch of that slot.
    void * read_slot(const l2_read & read) const {
        return class_layout() ? storage_slot(read.layer, read.kind, read.slot) : ring_host(read.kind, read.slot);
    }
    size_t read_pitch(const l2_read & read) const {
        return class_layout() ? storage_.pitch[storage_.storage_of[geo_.layer_class[read.layer]]][read.kind] :
            ring_pitch_[read.kind];
    }
    bool   verify_slice(int layer, int kind, int expert, const void * data, std::string & reason);
    bool   read_raw(int layer, int kind, int expert, void * dst, std::string & reason);
    void   service_layer(int layer, uint32_t seq, const std::vector<int> & ids);
    void   serve_staged(int layer, uint32_t seq, const std::vector<l2_read> & reads);
    // `progress(n)` runs once the first n reads are in place (checked, tails cleared) while the
    // rest are still in flight.
    bool   run_reads(const std::vector<l2_read> & reads, std::string & reason, bool install = false,
                     const std::function<bool(size_t)> & progress = nullptr);
    void   worker_loop(std::vector<uint32_t> seen);
    void   collect_wait(int layer);
    void   collect_stage(int layer);
    void   report_counters(const char * what);
    struct sample {
        uint32_t seq = 0, rows = 0;
        uint64_t ubatch = 0;
        int layer = 0;
        bool prompt = false;
        uint64_t distinct = 0, ssd_bytes = 0, wait_ticks = 0, steady_ticks = 0;
    };

    const geometry & geo_;
    l2_config cfg_;
    int    device_ = -1;
    bool   sized_  = false;
    bool   mapped_ = false;
    std::string size_note_;

    std::array<size_t, 3> ring_pitch_ = {0, 0, 0};
    int    prompt_slots_ = 0;
    int    decode_slots_ = 0;
    bool   prompt_ring_  = true;
    size_t host_bytes_   = 0;

    std::array<expert_os::reservation, 3> ring_res_ = {};
    std::array<void *, 3> ring_host_base_   = {nullptr, nullptr, nullptr};
    std::array<void *, 3> ring_device_base_ = {nullptr, nullptr, nullptr};
    std::array<bool,   3> ring_registered_  = {false, false, false};

    void *   mail_host_     = nullptr;   // l2_mailbox[n_layers]
    void *   mail_device_   = nullptr;
    uint64_t * maps_host_   = nullptr;   // [(layer*3 + kind)*n_experts + expert]
    uint64_t * maps_device_ = nullptr;
    uint32_t * demand_host_ = nullptr;   // [layer*bitmap_words + word], written by the GPU
    uint32_t * demand_device_ = nullptr;
    uint32_t * serve_host_  = nullptr;   // same layout, written by the CPU when a plan changes
    uint32_t * serve_device_ = nullptr;

    std::vector<std::string>          paths_;
    std::vector<expert_os::file_handle> files_;
    std::vector<std::array<backing, 3>> backing_;   // [layer][kind]

    std::vector<std::vector<int32_t>> vram_;
    l2_host_geometry host_geo_;
    expert_locations homes_;
    bool any_lent_ = false;
    bool mailbox_active_ = false; // true once the mailbox exists; graphs capture its nodes

    l2_ledger ledger_;
    class_storage   storage_;   // class layout only
    l2_class_ledger cledger_;   // class layout only
    std::vector<std::vector<int>> home_slot_;   // class layout: [size class][host slot] storage slot
    std::mutex io_mutex_;
    std::unique_ptr<expert_os::read_queue> queue_;
    std::vector<std::unique_ptr<std::ifstream>> verify_files_;
    std::vector<char> verify_buffer_;
    void * bounce_aligned_ = nullptr;

    std::thread worker_;
    std::atomic<bool> stop_   { false };
    std::atomic<bool> running_{ false };
    bool verify_     = false;
    bool any_ssd_    = false;

    l2_counters counters_;
    int clock_khz_ = 0, steady_khz_ = 0, first_layer_ = -1;
    uint64_t ubatch_ = 0;
    bool inferred_prompt_ = true;
    std::atomic<int> phase_{-1}; // Without a profile, classify by the configured decode row bound.
    std::vector<sample> pending_, samples_;
    std::vector<uint32_t> wait_seen_;
    std::vector<uint8_t>  staged_layer_;   // compute thread: the last route of the layer was staged
    std::vector<uint32_t> stage_seen_;     // worker: the stage wait ticks of each layer already counted
};

} // namespace ggml_cuda_expert
