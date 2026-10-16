#pragma once

// Decode pool of the expert cache: extra inclusive VRAM slots for the experts a
// layer missed most recently, in the part of a compute buffer that the decode graphs leave unused.
//
// A compute buffer is reserved for the largest graph (a prompt ubatch at the full context); a graph
// of a few rows uses only its lower part. While the graphs stay below a boundary, the controller
// lends the rest to this pool (lend), and takes it back, in stream order, before a graph that
// reaches past the boundary (reclaim). Nothing is moved either way: a slot only ever holds a copy of
// an expert whose home is host memory.
//
// The kernels find a slot through the 64-bit address table they already take before the host slot
// table (ggml_cuda_expert_cache_select): a pooled expert's entry is the address of its slot; its
// previous value (the home address, or 0) is kept and written back when the slot is evicted or the
// pool is reclaimed. No matmul kernel changes. With the SSD tier the table is the tier's, and only
// host residents (serve bit clear) are admitted, so the worker never writes an entry the pool owns.
//
// Per routed layer, r positions (the same r for every layer, at most r_cap), least recently used.
// For every launch of up to `rows` rows, on the compute stream right after the router, the step
// kernel stamps the positions this launch hits and hands each admissible distinct miss a position
// (empty first, then the oldest; never one this launch uses; later rows first). It publishes the
// slot at once and lists it; the copy kernel, next in stream order, copies the three kinds from
// the home into the slot. The matmuls of the layer come after both and read the slot as a VRAM
// address. Every array has a fixed address from allocate(), so captured graphs stay valid; lend and
// reclaim only change contents.
//
// Admission filter (ghost:8): the recent pool's ghost term
// (RANMA_EXPERT_RECENT_FILTER of the 2026-09 L1 recent pool): a miss is admitted only when the same
// expert also missed in this layer within the last K steps of the layer; the misses of a step are
// stamped after its decisions, so a miss never admits itself. Steps count only the launches that step
// a lent pool, and the stamps survive reclaims, so the window spans lends.

#include "common.cuh"
#include "expert-profiler.cuh"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

struct ub_pool_config {
    bool   enabled   = false;
    size_t min_bytes = size_t(256) << 20; // smallest region worth lending
    size_t margin    = size_t(64) << 20;  // headroom above the largest decode extent seen
    int    r_cap     = 64;                // positions per layer allocated
    int    rows      = 0;                 // launches of at most this many rows step the pool; 0 = the decode rows
    bool   log       = false;
    int    ghost     = 8;                 // admission filter ghost:K; 0 = admit every admissible miss
};

// RANMA_EXPERT_UB_POOL_LOG=0|1. Read once.
ub_pool_config ub_pool_config_from_env();

static constexpr int ub_pool_max_experts = 1024;
static constexpr int ub_pool_max_rcap    = 256;
static constexpr int ub_pool_threads     = 256;
static constexpr int ub_pool_copy_blocks = 64;
static constexpr unsigned long long ub_counters_sentinel = 0x5E5E5E5E5E5E5E5Eull;

enum ub_pool_counter : int {
    ub_pool_hits = 0,      // selections served by a position
    ub_pool_fills,         // positions filled (copies)
    ub_pool_fill_bytes,
    ub_pool_over,          // admissible distinct misses beyond the free positions
    ub_pool_evictions,     // filled positions whose previous owner left
    ub_pool_steps,         // step launches with r > 0
    ub_pool_rejected,      // admissible misses the admission filter turned away
    ub_pool_n_counters,
};

// Where a miss of one kind of a layer reads its expert, as the matmul would without the pool, and
// the table the pool publishes into.
struct ub_pool_kind {
    const char *    tensor     = nullptr; // host arena base or mapped tensor alias
    const int32_t * host_slots = nullptr; // host arena slot of an expert; null = the expert index
    uint64_t *      addresses  = nullptr; // the address table of the lookup
    size_t          stride     = 0;       // nb[2]
};

struct ub_pool_layer {
    ub_pool_kind     kind[3];
    const int32_t *  slots = nullptr;     // VRAM slot table of the layer (L1 residents)
    const uint32_t * serve = nullptr;     // SSD tier: experts only the worker places; null = none
    int32_t          enabled = 0;
};

struct ub_pool_meta {
    int32_t  offset; // first position of the layer
    int32_t  r;      // active positions, 0 = the pool is not lent
    uint32_t step;   // launches that stepped a lent pool; kept across lends (stamps, ghost window)
    int32_t  pad;
    unsigned long long n_hits, n_fills; // of the layer, cumulative: the values the step kernel writes to ub_pool_live
};

// Per layer, in mapped host memory: the layer's cumulative counts, written by the step kernel of that layer when they
// change, so that the host reads them without waiting for the device.
struct ub_pool_live {
    unsigned long long hits, fills;
};

struct ub_pool_copy {
    int32_t expert;
    int32_t pos;     // absolute position index
};

class ub_pool {
public:
    ub_pool(int n_layers, int n_experts, int r_cap);
    ~ub_pool();

    ub_pool(const ub_pool &) = delete;
    ub_pool & operator=(const ub_pool &) = delete;

    // Once, before any graph capture. `layers[l].enabled` marks the layers that get positions; the
    // bytes of one position of layer l are the three kinds of `kind_bytes[l]` with their tails.
    // `ghost` > 0 turns on the admission filter ghost:K.
    bool allocate(int device, const std::vector<ub_pool_layer> & layers,
                  const std::vector<std::array<size_t, 3>> & kind_bytes,
                  const std::vector<std::array<size_t, 3>> & kind_tails, int ghost = 0);
    bool allocated() const { return blob_ != nullptr; }
    size_t device_bytes() const { return bytes_; }

    // Lends [base, base + bytes) on `stream`: lays out the positions (the same r for every enabled
    // layer), empty, and zeroes their tails. Returns r (0 = too small, nothing lent).
    int  lend(void * base, size_t bytes, cudaStream_t stream);
    // Writes every pooled entry back and sets r = 0 on `stream`. The region may be overwritten by any
    // work queued on `stream` after this.
    void reclaim(cudaStream_t stream);
    int  r_cap() const { return r_cap_; }
    bool lent() const { return r_ > 0; }
    int  r() const { return r_; }
    void * base() const { return base_; }
    size_t lent_bytes() const { return lent_bytes_; }
    const void * state_begin() const { return blob_; }
    const void * state_end() const { return (const char *) blob_ + bytes_; }

    // The step and copy of `layer` for a launch of n_rows rows, on the compute stream.
    // `prof` (a profiler's counts of the layer, with its bank stride, bank count and selection) makes the step also add the
    // top-k ids of the selected rows, what launch_profile_ids does, before it returns for an unlent layer.
    void launch_step(int layer, const int32_t * ids, int n_rows, int row_stride, int n_used, bool admit, cudaStream_t stream,
        uint32_t * prof_counts = nullptr, size_t prof_bank_stride = 0, uint32_t prof_n_banks = 0,
        const profile_selection * prof_selection = nullptr);
    void launch_copy(int layer, cudaStream_t stream);

    // Device must be idle (or `stream` synchronized by the caller). Cumulative since allocate.
    bool read_counters(std::array<uint64_t, ub_pool_n_counters> & out) const;
    // The same counts from the per-layer values in mapped host memory: never waits for the device, and misses
    // the launches that have not finished. Cumulative since allocate. False without the host memory.
    bool read_live(uint64_t & hits, uint64_t & fills, uint64_t & fill_bytes) const;
    // Device idle: every position of the lent pool holds its owner's home bytes, and every owner's
    // entries point at its position. Returns the positions checked, -1 on a mismatch.
    long verify(std::string & reason) const;

private:
    const int n_layers_;
    const int n_experts_;
    const int r_cap_;
    int    device_ = -1;
    void * blob_   = nullptr;
    size_t bytes_  = 0;
    int    r_      = 0;
    int    ghost_  = 0;
    void * base_   = nullptr;
    size_t lent_bytes_ = 0;

    std::vector<ub_pool_layer>          layers_;      // host copy of the descriptors
    std::vector<std::array<size_t, 3>>  kind_bytes_;
    std::vector<std::array<size_t, 3>>  kind_tails_;
    std::vector<size_t>                 pos_bytes_;   // [layer]

    ub_pool_layer * d_layers_ = nullptr;  // [L]
    ub_pool_meta *  d_meta_   = nullptr;  // [L]
    uint64_t *      d_addr_   = nullptr;  // [P][3] slot address of each kind
    uint64_t *      d_home_   = nullptr;  // [P][3] the owner's previous table entries
    int32_t *       d_owner_  = nullptr;  // [P]
    uint32_t *      d_stamp_  = nullptr;  // [P]
    int16_t *       d_pos_of_ = nullptr;  // [L][E] position of the layer holding the expert, -1
    ub_pool_copy *  d_copy_   = nullptr;  // [P]
    int32_t *       d_count_  = nullptr;  // [L]
    unsigned long long * d_counters_ = nullptr; // [ub_pool_n_counters]
    size_t *        d_sizes_  = nullptr;  // [L][3] slice bytes, then [L][3] tail bytes
    uint32_t *      d_last_miss_ = nullptr; // [L][E] step of the expert's last miss (ghost filter), ~0 = never

    // pinned staging of a lend
    uint64_t *      h_addr_ = nullptr;
    unsigned long long * h_counters_ = nullptr;   // read_counters
    ub_pool_live *  h_live_ = nullptr;   // [L] mapped host memory, null when it could not be allocated
    ub_pool_live *  d_live_ = nullptr;   // its device address
};

} // namespace ggml_cuda_expert
