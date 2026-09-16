#pragma once

// Finite host storage with demand reads from the original GGUF files.
// A GPU bitmap publishes the selected experts. The worker fills only misses and answers
// through a coherent address table. A done counter protects the slots until their last use.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-l2-ledger.h"
#include "expert-location.h"
#include "expert-os.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ggml_cuda_expert {

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
    uint32_t log_mask         = 0;
};

// What the tier reports every so often under GGML_EXPERT_LOG_L2.
struct l2_counters {
    uint64_t repartitions = 0;
    uint64_t generations   = 0;
    uint64_t ssd_bytes     = 0;
    uint64_t ssd_reads     = 0;
    uint64_t ring_hits     = 0;
    uint64_t install_ssd_bytes = 0, install_ssd_reads = 0;
    uint64_t wait_ticks = 0, steady_ticks = 0, measured_layers = 0;
    uint64_t distinct = 0, vram_bytes = 0, host_bytes = 0, file_bytes = 0;
    uint64_t prompt_ubatches = 0, decode_ubatches = 0, decode_tokens = 0;
    uint64_t prompt_wait_ticks = 0, decode_wait_ticks = 0;
    double   service_ms    = 0.0;
    uint64_t verify_fills  = 0;
    uint64_t verify_bytes  = 0;
    uint64_t verify_bad    = 0;
    uint64_t owner_checks  = 0;
};

// Class arena addresses. Lent residents use the common location map.
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
    size_t fixed_bytes() const { return ring_bytes(prompt_slots_) + metadata_bytes(); }
    bool   sized() const { return sized_; }
    const std::string & size_note() const { return size_note_; }

    // ---- allocation -----------------------------------------------------------------------------
    bool allocate(int device);
    bool map();
    size_t host_bytes() const { return host_bytes_; }

    // ---- the loader ------------------------------------------------------------------------------
    void set_backing(int layer, int kind, int file_index, const char * path, uint64_t offset);
    bool open_files(std::string & reason);
    // False for an expert whose bytes the loader must skip.
    bool wanted(int layer, int expert) const;

    // ---- plans -----------------------------------------------------------------------------------
    void set_homes(const std::vector<std::vector<int32_t>> & vram,
                   const expert_locations & host, const l2_host_geometry & host_geo);
    // The ring class that travels with an installed plan. Returns true when the size changed.
    bool set_prompt_ring(bool prompt);
    int  ring_count() const { return ledger_.ring_count(); }
    // Whether the current plan leaves any expert in the file. Recomputed with every plan; the
    // mailbox kernels run either way and cost nothing when no routed expert needs the worker.
    bool any_ssd() const { return any_ssd_; }
    const expert_locations & locations() const { return homes_; }
    void * host_address(int layer, int kind, int expert) const;
    void * location_address(int cls, int kind, expert_location at, bool device = false) const;
    void finish_write(int cls, int kind, expert_location at);
    bool verify_resident(int layer, int kind, int expert, const void * data, std::string & reason) {
        return verify_slice(layer, kind, expert, data, reason);
    }
    // ---- the install transaction ------------------------------------------------------------------
    // Reads one slice out of the file into `dst`, through a bounce slot when `dst` is not sector
    // aligned (a host arena slot never is). Synchronous, used only while nothing computes.
    bool read_slice(int layer, int kind, int expert, void * dst, std::string & reason);

    // The common active prefix is never a lent source or destination. Install discards its tags.
    int install_read_slots() const { return std::min({cfg_.queue_depth, ring_count(), decode_slots_}); }
    bool read_install(const std::vector<l2_read> & reads, std::string & reason);
    const void * read_address(const l2_read & read) const {
        return static_cast<char *>(ring_host(read.kind, read.slot)) + backing_[read.layer][read.kind].shift;
    }

    // ---- the hot path ------------------------------------------------------------------------------
    const uint64_t * addresses(int layer, int kind) const;
    void publish_and_wait(int layer, const ggml_tensor * ids, cudaStream_t stream);
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
    void   verify_addresses(int demanded_layer, const std::vector<int> & ids);
    uint64_t address_of(int layer, int kind, int expert) const;
    void * ring_host(int kind, int slot) const {
        return static_cast<char *>(ring_host_base_[kind]) + size_t(slot)*ring_pitch_[kind];
    }
    bool   verify_slice(int layer, int kind, int expert, const void * data, std::string & reason);
    bool   read_raw(int layer, int kind, int expert, void * dst, std::string & reason);
    void   service_layer(int layer, uint32_t seq, const std::vector<int> & ids);
    bool   run_reads(const std::vector<l2_read> & reads, std::string & reason, bool install = false);
    void   worker_loop(std::vector<uint32_t> seen);
    void   collect_wait(int layer);
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
};

} // namespace ggml_cuda_expert
