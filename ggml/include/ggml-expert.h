#pragma once

// Optional backend-managed cache of MoE routed-expert weights (ranma).
//
// A backend that implements the cache answers the proc address GGML_EXPERT_IFACE_PROC_NAME with a
// ggml_backend_expert_iface_t; calling it returns one versioned function table. Everything the
// llama layer and the server need goes through that table, so a backend built without the cache
// simply does not answer the name and every caller treats the feature as absent.
//
// Layering rule: the backend knows nothing about servers, phases or request types. It sees opaque
// profile banks (histograms of router selections), plans derived from a bank, and an installed
// plan. The names of the banks and the moments at which they are marked, committed and installed
// belong to the caller.

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXPERT_ABI_VERSION      18
#define GGML_EXPERT_IFACE_PROC_NAME  "ggml_backend_expert_iface"

#define GGML_EXPERT_BANK_NONE        0xFFFFFFFFu
#define GGML_EXPERT_LEND_MAX_PIECES  48
#define GGML_EXPERT_PLAN_NONE        0xFFFFFFFFu

typedef uint32_t ggml_expert_bank_id;
typedef uint32_t ggml_expert_plan_id;

enum ggml_expert_mode {
    GGML_EXPERT_MODE_INCLUSIVE = 0, // cached experts are copies, the host tensor stays complete
    GGML_EXPERT_MODE_EXCLUSIVE = 1, // one copy per expert, either in VRAM or in host memory
};

enum ggml_expert_policy {
    GGML_EXPERT_POLICY_ADAPTIVE = 0,
    GGML_EXPERT_POLICY_STATIC = 1, // seeded placement, profiling without installs
    GGML_EXPERT_POLICY_OFF = 2,    // no L1 payload, seeded fixed L2, no profiling
};

enum ggml_expert_log_flags {
    GGML_EXPERT_LOG_INSTALL = 1u << 0,
    GGML_EXPERT_LOG_PROFILE = 1u << 1,
    GGML_EXPERT_LOG_PREFILL = 1u << 2,
    GGML_EXPERT_LOG_L2      = 1u << 3,
};

// Passed once, before the model's weight buffers are allocated. Strings must outlive the call only.
struct ggml_expert_config {
    uint32_t abi_version;             // GGML_EXPERT_ABI_VERSION

    size_t   l1_bytes;                // VRAM budget for cached expert slices and their tables; 0 = cache off, except for POLICY_OFF (L2 only)
    enum ggml_expert_mode mode;
    enum ggml_expert_policy policy;
    uint32_t random_seed;             // STATIC/OFF: deterministic per-layer expert order

    // Host tier budget. 0 (the option's -1): the budget that holds every expert outside L1 (exclusive) or every expert
    // (inclusive), the rings, the tables and the spares, and (exclusive) ring slots for the L1 experts the prefill lend
    // evicts (the slots a lend of l2_lend_pieces or of the op's lend_scratch blocks evicts, packed as a lend packs them;
    // at most the L1 slots); the backend sizes it from the geometry.
    size_t   l2_bytes;
    size_t   l2_lend_bytes;           // l2_bytes 0: the most VRAM a prefill lend takes from L1 (at most L1), for the log
    size_t   l2_lend_pieces[GGML_EXPERT_LEND_MAX_PIECES]; // l2_bytes 0: the pieces of the widest layer-major pass
    int32_t  l2_lend_n_pieces;
    size_t   l2_prefill_ring_bytes;
    size_t   l2_decode_ring_bytes;
    uint64_t l2_prefill_rows, l2_decode_rows; // Worst-case rows, before any expert allocation.
    int32_t  l2_experts_used;          // Filled from model metadata before configure.
    int32_t  l2_worker_cpu;           // -1 selects the last active logical CPU
    bool     l2_phase_rings;          // the ring shrinks during generation and grows at the request end;
                                      // false keeps one ring, sized for prompt processing

    bool     delta_install;           // keep slices that stay selected in place when a new plan is installed
    bool     freeze;                  // profile only: commits still score and plan, installs are refused
    bool     profile_archive;         // move records that leave the score window to archive/ instead of deleting them
    bool     profile_reset;           // discard every stored record of every bank at startup
    int32_t  spare_slots;             // exclusive: free slots rotated per exchange batch

    const char * profile_dir;         // root of the profile banks; NULL or "" = no profiling, no plans
    const char * initial_bank;        // bank labels separated by commas, most wanted first: the first one
                                      // with stored records seeds the plan installed at model load.
                                      // NULL or "" = start with empty arenas
    uint32_t log_mask;                // ggml_expert_log_flags

    // Joint cache: every model of the process that has routed experts can share the one cache,
    // with one L1 and one L2 budget. The first model's config names the models that join later,
    // by their first GGUF file; the backend reads their tensor layout from the file so that the
    // load-time plan and the arenas cover all of them. A later model's config names its own file
    // and is accepted as that member; its other fields are ignored.
    const char * model_path;          // first GGUF file of the model this config is passed with; NULL = unnamed
    float        model_weight;        // multiplier on this model's host read cost in the joint plan; <= 0 = 1
    uint32_t     n_join;              // number of models that join after this one (first config only)
    const char * const * join_paths;  // [n_join] first GGUF file of each; splits follow split.count
    const float * join_weights;       // [n_join] multipliers, NULL = 1 each


    // Layers of this model whose router selection is a pure function of the input token id, so the
    // caller can name the selected experts before the graph is computed (route_hint). Declared by
    // the model loader; only the first model's declaration is used. The environment variable
    // RANMA_EXPERT_HASH_EARLY (0, ssd, vram or both) decides what the cache does with the hints.
    const int32_t * early_route_layers; // [n_early_route_layers] the model's own layer indices
    uint32_t n_early_route_layers;

    // The prefill lend (ggml_expert_lend_*). `lend` turns it on: the prompt buffers come out of L1, and the
    // decode restores the evicted experts as it reads them. lend_scratch: the three device blocks of the expert-major
    // op (MOE_FFN_HAD) of one prompt ubatch (moe-ffn-had-layout.h), which a graph with the op borrows before it runs;
    // 0 = the op of the ubatch-major prompt borrows nothing.
    bool     lend;
    size_t   lend_scratch[3];

    // ABI 17: the share of the L1 budget (0..100 %) that is not planned: per routed layer, least recently used copies of
    // the experts the layer missed (the decode pool's positions in L1 slots, docs/ranma/expert-cache-ub-pool.md); 0 = none.
    uint32_t l1_lru_pct;

    // ABI 18: the three blocks of lend_scratch for a prompt call of n_tokens (called without a lock; never larger than
    // lend_scratch is used). A graph borrows only these, not the blocks of the configured ubatch. NULL = lend_scratch always.
    void   (*lend_scratch_of)(void * user, int64_t n_tokens, size_t out[3]);
    void *   lend_scratch_user;
};

// ---- the prefill lend -------------------------------------------------------------------------------------------
//
// A prompt borrows its VRAM buffers from the L1 budget: the pieces evict the lowest ranked L1 experts (by the installed
// plan's profile) and take the top slots of each size class. Nothing is lent from host memory: a lend that does not fit
// L1 is refused and the caller narrows its width (more passes). The evicted experts go where an install sends them (the
// L2, which holds only experts). The release does not refill: a decode launch that reads an evicted expert copies it
// back into a lent slot first (lazy restore).


// What a piece is for: a device piece of a role other than BUFFER is also handed to the expert-major op of the
// holder's graphs (in place of its pool or block allocations).
enum ggml_expert_lend_role {
    GGML_EXPERT_LEND_BUFFER   = 0,
    GGML_EXPERT_LEND_SCRATCH0 = 1, // the op's three scratch blocks (moe-ffn-had-layout.h)
    GGML_EXPERT_LEND_SCRATCH1 = 2,
    GGML_EXPERT_LEND_SCRATCH2 = 3,
    GGML_EXPERT_LEND_STAGE    = 4, // the first of the two staging buffers of the PCIe-once op
    GGML_EXPERT_LEND_T        = 5, // the down rows (T) of the exact PCIe-once op
    GGML_EXPERT_LEND_STAGE1   = 6, // the second staging buffer (a piece of its own: two smaller pieces pack better)
};

struct ggml_expert_lend_piece {
    size_t  bytes;
    int32_t role;  // ggml_expert_lend_role
};

struct ggml_expert_lend_request {
    int32_t n_pieces;
    struct ggml_expert_lend_piece pieces[GGML_EXPERT_LEND_MAX_PIECES];
};

// buffers[i] wraps the VRAM of piece i (a device buffer of the backend); the lend owns them: they stay valid until
// lend_release. Null for a piece of 0 bytes.
struct ggml_expert_lend_grant {
    ggml_backend_buffer_t buffers[GGML_EXPERT_LEND_MAX_PIECES];
};

// Cumulative since the cache was configured.
struct ggml_expert_lend_counts {
    uint64_t acquires;      // lends of new slots
    uint64_t retakes;       // lends of a restore space that held enough slots (no eviction)
    uint64_t releases;
    uint64_t evicted;       // L1 experts the lends evicted
    uint64_t evicted_bytes;
    uint64_t restores;      // experts the decode copied back into their lent slot
    uint64_t restore_bytes;
    int32_t  held_slots;    // L1 slots of the current lease or restore space
    int32_t  state;         // 0 none, 1 held, 2 restore
    uint64_t lendable_bytes; // VRAM the whole L1 could lend now (every resident evicted), in its contiguous ranges
};

// The experts one layer of the next graph will select, known before the graph is built. `ids`
// holds n_ids expert ids (all rows of the ubatch, duplicates allowed); `layer` is the model's own
// layer index. `rows` is the row count of the ubatch the hint is for.
struct ggml_expert_route_hint {
    int32_t         layer;
    int32_t         rows;
    int32_t         n_ids;
    const int32_t * ids;
};

// Numbers only; the caller attaches them to the record a bank commit writes.
struct ggml_expert_record {
    uint64_t request_count;
    uint64_t input_tokens;
    uint64_t output_tokens;
    uint64_t bank_tokens;
};

struct ggml_expert_status {
    bool     configured;
    bool     registered;              // a routed-expert weight context is known
    bool     installed;               // arenas exist, lookups can hit
    bool     disabled;                // a recoverable failure turned the cache off for this model
    const char * disabled_reason;     // static string, valid while the backend is loaded
    size_t   device_bytes;            // VRAM held by the cache
    size_t   host_bytes;              // host memory held by the cache (exclusive/L2)
    uint32_t n_banks;
};

// The decode pool, cumulative since the pool was allocated. Positions are counted, bytes are
// the bytes copied for them. The step kernels write the per-layer counts into host memory as they finish, so
// reading them never waits for the device; the work of graphs still running may be missing.
struct ggml_expert_pool_counts {
    uint64_t lends;       // the pool was lent (a lend starts it empty)
    uint64_t reclaims;    // the pool was taken back (a graph reached into it, e.g. a prompt ubatch)
    uint64_t hits;        // selections served by a position
    uint64_t fills;       // positions filled by the step of a launch (copied right before the layer's matmuls)
    uint64_t fill_bytes;
};

struct ggml_expert_iface {
    uint32_t abi_version;

    // Lifecycle. configure -> alloc_context -> finalize -> ... -> release.
    bool (*configure)(const struct ggml_expert_config * config);
    // The cache owns the weight buffer of the routed-expert context (VRAM, the L2 host arenas, the file). Returns NULL
    // when the ordinary allocation must be used (cache off, or the context holds no experts).
    ggml_backend_buffer_t (*alloc_context)(struct ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity);
    // Model load end: the weights are written and the host buffers are registered. Plans from the
    // stored profile, allocates the arenas exactly once and installs. Returns false when the cache
    // is off or was disabled (see status); the model still works without it.
    bool (*finalize)(void);
    void (*release)(struct ggml_context * ctx);
    bool (*status)(struct ggml_expert_status * out);

    // Profile banks and plans.
    // `prompt_bank` says the bank counts prompt processing. The backend knows no phase names.
    bool (*bank_open)(const char * label, bool prompt_bank, ggml_expert_bank_id * out_bank);
    bool (*bank_mark)(ggml_expert_bank_id bank);      // begin an interval: the bank histogram is zeroed
    bool (*bank_commit)(ggml_expert_bank_id bank, const struct ggml_expert_record * record, ggml_expert_plan_id * out_plan);
                                                       // store the interval, rescore, plan; nothing is installed
    bool (*bank_discard)(ggml_expert_bank_id bank);   // zero the bank without storing
    bool (*plan_install)(ggml_expert_plan_id plan);   // must run with no compute in flight on the device
    // Rows [row_begin, row_end) of the next graph computed on `backend` feed `bank`; an empty range
    // or GGML_EXPERT_BANK_NONE records nothing. Cheap when unchanged.
    bool (*profile_select)(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank);

    // The SSD tier, called by the model loader for routed expert tensors it is about to fill.
    // load_wanted is false for an expert whose bytes stay in the file, so the loader skips them.
    // set_backing says where the tensor's bytes live; false means the tier is off for this tensor
    // and the loader must read it whole, as before.
    bool (*load_wanted)(const struct ggml_tensor * tensor, int32_t expert);
    bool (*set_backing)(const struct ggml_tensor * tensor, int32_t file_index, const char * path, uint64_t file_offset);

    // Memory accounting for buffers the cache owns (exclusive mode).
    bool (*memory)(ggml_backend_buffer_t buffer, size_t * host_bytes, size_t * device_bytes);

    // A compute backend that runs the model whose routed-expert context is `weights`. The router
    // selections of its graphs are counted for that model only; selections of a backend that was
    // never attached are not profiled. Call once per context, before its first compute.
    bool (*attach_backend)(ggml_backend_t backend, struct ggml_context * weights);

    // The experts of layers declared in early_route_layers that the next graph computed on
    // `backend` will select. Called before the graph is built or launched, and only then; it never
    // waits for the device. Returns false when the cache does not use hints for this model (the
    // caller may stop computing them); true otherwise, whether or not this hint was used.
    bool (*route_hint)(ggml_backend_t backend, const struct ggml_expert_route_hint * hints, int32_t n_hints);

    // Before every graph computed on `backend`: [base, base + size) is its compute buffer and the graph
    // uses it up to `extent` (ggml_backend_sched_get_buffer_extent); `n_tokens` is the rows of the batch,
    // <= 0 for a graph that is not one. The cache may keep data in the unused part between graphs and takes
    // it back, in stream order, before a graph that reaches into it. Returns false
    // when it does not use compute buffers.
    bool (*compute_spare)(ggml_backend_t backend, void * base, size_t size, size_t extent, int32_t n_tokens);
    // Graphs of at most this many rows should be planned on their own (ggml_backend_sched_set_alloc_compact), so that
    // they use only the start of the compute buffer; 0 = no preference.
    int32_t (*compact_rows)(void);
    // The counts of the decode pool; false when it is not allocated. Never waits for the device.
    bool (*pool_counts)(struct ggml_expert_pool_counts * out);
    // The expert prefetch (RANMA_EXPERT_PREFETCH): the routed layers whose experts a graph of `n_rows` rows should
    // predict (bit l of mask[l/64]), for the first model of the cache. Returns a version that changes whenever a
    // mask changes (a graph built with another version must be rebuilt), 0 when nothing is predicted.
    uint32_t (*prefetch_layers)(int32_t n_rows, uint64_t * mask, int32_t n_words);

    // The prefill lend (above). lend_fits: whether `request` could be lent now (pure: nothing moves). lend_acquire:
    // with no compute in flight on `backend`; evicts what the pieces need and fills `out`. lend_release: the holder's
    // graphs that used the pieces were synchronized; the slots become the restore space. All false when the lend is
    // off. A lend taken here is ended only by lend_release (or the cache's release).
    bool (*lend_fits)(ggml_backend_t backend, const struct ggml_expert_lend_request * request);
    bool (*lend_acquire)(ggml_backend_t backend, const struct ggml_expert_lend_request * request, struct ggml_expert_lend_grant * out);
    bool (*lend_release)(ggml_backend_t backend);
    bool (*lend_counts)(struct ggml_expert_lend_counts * out);
};

typedef const struct ggml_expert_iface * (*ggml_backend_expert_iface_t)(void);

// The cache reads its experts through the backend's host-direct path. A backend that has one answers
// GGML_EXPERT_HOST_DIRECT_PROC_NAME: the call turns that path on for the expert cache before the
// model loads, so that the cache works without extra settings. It leaves alone what the backend's own
// switches set explicitly, and raises the token count up to which a weight is read in place to at
// least max_batch (the ubatch size). Returns whether the path is on.
#define GGML_EXPERT_HOST_DIRECT_PROC_NAME "ggml_backend_expert_host_direct"
typedef bool (*ggml_backend_expert_host_direct_t)(int64_t max_batch);

#ifdef __cplusplus
}
#endif
