#pragma once

// Finite host storage with demand reads from the original GGUF files.
// A GPU bitmap publishes the selected experts. The worker fills only misses and answers
// through a coherent address table. A done counter protects the slots until their last use.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-l2-batch.h"
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
    // Early reads (RANMA_EXPERT_HASH_EARLY ssd|both): the demand of a layer whose route is known before
    // its generation (post_early) is read into the ring by the worker as soon as it is posted.
    bool    early             = false;
    // Batched service (expert-l2-batch.h, class layout): a prompt ubatch multiplies the experts that
    // need no read first and the file residents in batches through half the class ring each, so the
    // rings no longer have to hold the most one ubatch can demand. False is the prompt floor ring.
    bool    batched           = false;
    // The most rows a ubatch multiplies with the vector kernels; above it MMQ runs. A batched ubatch
    // of fewer rows runs MMQ as well when its demand can exceed the ring.
    int64_t vec_rows          = 8;
    // Where the fills and hits of a prompt ubatch go in the ring's least recently used list.
    l2_prompt_fill prompt_fill = l2_prompt_fill::mru;
};

// Whether the staged service is on when RANMA_EXPERT_L2_STAGED is not set. On, it covers the
// ubatches with more rows than the decode bound, i.e. prompt processing; generation keeps the
// single wait. It reads the same bytes into the same slots and runs the same kernels.
static constexpr bool l2_staged_default = true;

// Whether the batched service is on when RANMA_EXPERT_L2_BATCHED is not set, and where it puts the
// fills and hits of a prompt ubatch when RANMA_EXPERT_L2_PROMPT_FILL is not set.
static constexpr bool l2_batched_default = true;
static constexpr l2_prompt_fill l2_prompt_fill_default = l2_prompt_fill::lru;
// The ring factor of the batched service when RANMA_EXPERT_L2_RING_FACTOR is not set: each class ring
// is this share of its prompt floor (as far as the budget allows, never below the minimum).
static constexpr double l2_batched_ring_factor = 0.7;

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
    // Early reads (post_early). Hinted layers served early / skipped because the layer's generation was
    // already published; slices read and their bytes (also in ssd_reads / ssd_bytes); hinted experts
    // already in the ring; experts read early that the layer's next service found in the ring (not in
    // ring_hits), that it did not demand, or that were evicted before it; generations served while an
    // early batch was still reading; the decode wait of the declared early layers (in every mode).
    uint64_t early_jobs = 0, early_layers = 0, early_late = 0, early_reads = 0, early_bytes = 0;
    uint64_t early_already = 0, early_hits = 0, early_unused = 0, early_lost = 0, early_inline = 0;
    uint64_t early_layer_wait_ticks = 0, early_layer_samples = 0;
    // Batched service: generations served in batches, their batches, ring hits read again to make
    // room for two buffers, and the wall ticks the batch launches waited for their reads.
    uint64_t batched_generations = 0, batches = 0, batch_demoted = 0, batch_wait_ticks = 0;
};

// One posted early demand: per layer the hinted expert ids and the layer's published counter when
// the hint was made (a layer that published since then is already being served).
struct l2_early_layer {
    int layer = -1;
    uint32_t published = 0;
    std::vector<int> ids;
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
    // The batch launches per kind of a ubatch of `rows` rows of `layer` (0 = one launch, as without
    // batching). `mmq` says whether every kind of the layer runs MMQ at that row count; a ubatch that
    // cannot run MMQ is never batched. The same for the route and for every MUL_MAT_ID of the layer.
    int  batch_launches(int layer, int64_t rows, bool mmq) const;
    bool batched() const { return cfg_.batched && class_layout(); }
    // `launches`: batch_launches of this ubatch. Publishes the demand and orders the first launch of
    // the first kind after what it needs; with batch launches, also the batch tables of the layer.
    void publish_and_wait(int layer, const ggml_tensor * ids, cudaStream_t stream, int launches = 0);
    // Batch launches of `layer`: the device tables the MMQ launches read (expert -> batch, and
    // [batch][position] -> expert, -1 when empty) and the stride of the second.
    struct batch_tables { const int32_t * batch_of = nullptr; const int32_t * lists = nullptr; int capacity = 0; };
    batch_tables batch_device(int layer) const;
    // The batch launches the last route of `layer` published (0 after the layer is done): what the
    // MUL_MAT_IDs of the layer must launch, since the worker serves the generation by it.
    int  routed_launches(int layer) const {
        return layer >= 0 && (size_t) layer < route_launches_.size() ? route_launches_[(size_t) layer] : 0;
    }
    // Orders batch launch `batch` (1-based) of `kind` after that batch's reads, and reports it done.
    void batch_wait(int layer, int kind, int batch, cudaStream_t stream);
    void batch_done(int layer, int kind, int batch, cudaStream_t stream);
    // Staged service only: orders the kernel that reads `kind` of `layer` after that kind's reads.
    // A no-op for the first kind, which the route waits for, and for a layer routed unstaged.
    bool staged() const { return cfg_.staged_min_rows > 0; }
    // Whether the MUL_MAT_IDs of a layer may wait for their own kind (staged or batched service).
    bool kind_waits() const { return staged() || batched(); }
    void wait_kind(int layer, int kind, cudaStream_t stream);
    void mark_done(int layer, cudaStream_t stream);

    // ---- early reads -------------------------------------------------------------------------------
    // The layers whose route the model declares known before the graph (their decode waits are
    // counted separately in every mode).
    void set_early_layers(const std::vector<uint8_t> & layers);
    bool early() const { return cfg_.early; }
    // Posts the demand of hinted layers; the worker reads their file-tier experts into the ring before
    // it would have to, and serves them as hits when the generations arrive. Replaces a posted demand
    // the worker has not taken yet, or joins it when it is for the same generations. Never blocks on I/O.
    void post_early(std::vector<l2_early_layer> layers);
    // The published counter of a layer as the host sees it now.
    uint32_t published(int layer) const;

    void start_worker();
    void stop_worker();
    bool worker_running() const { return running_.load(std::memory_order_relaxed); }

    const l2_counters & counters() const { return counters_; }
    // A consistent copy while the worker runs (taken between two services).
    l2_counters counters_snapshot() {
        std::lock_guard<std::mutex> lock(io_mutex_);
        return counters_;
    }
    double steady_khz() const { return steady_khz_; }
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
    // The batched service of one generation with `launches` batch launches per kind.
    void   serve_batched(int layer, uint32_t seq, const std::vector<int> & ids, int launches);
    int    batch_capacity(int layer) const;
    void   serve_staged(int layer, uint32_t seq, const std::vector<l2_read> & reads);
    // `progress(n)` runs once the first n reads are in place (checked, tails cleared) while the
    // rest are still in flight.
    bool   run_reads(const std::vector<l2_read> & reads, std::string & reason, bool install = false,
                     const std::function<bool(size_t)> & progress = nullptr);
    void   worker_loop();
    // The worker's handling of a published generation of `layer` (under io_mutex_).
    void   handle_publish(int layer, uint32_t seq);
    void   serve_early(std::vector<l2_early_layer> job);
    // While an early batch reads: serves the published generations that need no read, except those
    // of layers whose early reads are still in flight (`busy`).
    void   serve_ready(const std::vector<uint8_t> & busy);
    bool   early_marked(int layer, int expert) const {
        return !early_mark_.empty() && early_mark_[(size_t) layer*(size_t) geo_.n_experts + (size_t) expert] != 0;
    }
    void   early_evicted(int layer, int expert);
    void   early_clear_all();
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
    // batched service
    void *    batch_mail_host_ = nullptr;     // l2_batch_mail[n_layers]
    void *    batch_mail_device_ = nullptr;
    int32_t * batch_tab_host_  = nullptr;     // [layer][3*n_experts]: batch of each expert, then the lists
    int32_t * batch_tab_device_ = nullptr;
    int32_t * batch_dev_       = nullptr;     // device copy the MMQ launches read, same layout
    size_t batch_tab_words() const { return 3*(size_t) geo_.n_experts; }

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
    std::vector<int>      route_launches_; // compute thread: the batch launches of the layer's last route
    std::vector<uint32_t> stage_seen_;     // worker: the stage wait ticks of each layer already counted
    std::vector<uint32_t> batch_seen_;     // worker: the batch wait ticks of each layer already counted
    std::vector<uint32_t> seen_;           // worker: the last published generation handled per layer
    // early reads
    std::vector<uint8_t> early_layer_;     // [layer] declared early layer
    std::vector<uint8_t> early_mark_;      // [layer*n_experts + expert] read early, not yet served (io_mutex_)
    std::vector<std::vector<int>> early_list_;   // [layer] the marked experts (io_mutex_)
    std::mutex early_mutex_;               // guards early_job_
    std::vector<l2_early_layer> early_job_;
    std::atomic<bool> early_pending_{ false };
    bool early_warned_ = false;
};

} // namespace ggml_cuda_expert
