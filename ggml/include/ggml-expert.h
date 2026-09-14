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

#define GGML_EXPERT_ABI_VERSION      2
#define GGML_EXPERT_IFACE_PROC_NAME  "ggml_backend_expert_iface"

#define GGML_EXPERT_BANK_NONE        0xFFFFFFFFu
#define GGML_EXPERT_PLAN_NONE        0xFFFFFFFFu

typedef uint32_t ggml_expert_bank_id;
typedef uint32_t ggml_expert_plan_id;

enum ggml_expert_policy {
    GGML_EXPERT_POLICY_ADAPTIVE = 0,
    GGML_EXPERT_POLICY_STATIC = 1, // seeded placement, profiling without installs
};

enum ggml_expert_log_flags {
    GGML_EXPERT_LOG_INSTALL = 1u << 0,
    GGML_EXPERT_LOG_PROFILE = 1u << 1,
    GGML_EXPERT_LOG_PREFILL = 1u << 2,
};

// Passed once, before the model's weight buffers are allocated. Strings must outlive the call only.
struct ggml_expert_config {
    uint32_t abi_version;             // GGML_EXPERT_ABI_VERSION

    size_t   l1_bytes;                // VRAM budget for cached expert slices and their tables; 0 = cache off
    enum ggml_expert_policy policy;
    uint32_t random_seed;             // STATIC: deterministic per-layer expert order

    bool     delta_install;           // keep slices that stay selected in place when a new plan is installed
    bool     freeze;                  // profile only: commits still score and plan, installs are refused
    bool     profile_archive;         // move records that leave the score window to archive/ instead of deleting them
    bool     profile_reset;           // discard every stored record of every bank at startup

    const char * profile_dir;         // root of the profile banks; NULL or "" = no profiling, no plans
    const char * initial_bank;        // the bank whose stored records seed the plan installed at model
                                      // load; NULL or "" = start with empty arenas
    uint32_t log_mask;                // ggml_expert_log_flags
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
    uint32_t n_banks;
};

struct ggml_expert_iface {
    uint32_t abi_version;

    // Lifecycle. configure -> register_context -> finalize -> ... -> release.
    bool (*configure)(const struct ggml_expert_config * config);
    // Called after the ordinary allocation of every host weight context. Returns false when the
    // context holds no routed experts; that is not an error.
    bool (*register_context)(struct ggml_context * ctx, ggml_backend_buffer_t buffer, const char * identity);
    // Model load end: the weights are written and the host buffers are registered. Plans from the
    // stored profile, allocates the arenas exactly once and installs. Returns false when the cache
    // is off or was disabled (see status); the model still works without it.
    bool (*finalize)(void);
    void (*release)(struct ggml_context * ctx);
    bool (*status)(struct ggml_expert_status * out);

    // Profile banks and plans.
    bool (*bank_open)(const char * label, ggml_expert_bank_id * out_bank);
    bool (*bank_mark)(ggml_expert_bank_id bank);      // begin an interval: the bank histogram is zeroed
    bool (*bank_commit)(ggml_expert_bank_id bank, const struct ggml_expert_record * record, ggml_expert_plan_id * out_plan);
                                                       // store the interval, rescore, plan; nothing is installed
    bool (*bank_discard)(ggml_expert_bank_id bank);   // zero the bank without storing
    bool (*plan_install)(ggml_expert_plan_id plan);   // must run with no compute in flight on the device
    // Rows [row_begin, row_end) of the next graph computed on `backend` feed `bank`; an empty range
    // or GGML_EXPERT_BANK_NONE records nothing. Cheap when unchanged.
    bool (*profile_select)(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank);
};

typedef const struct ggml_expert_iface * (*ggml_backend_expert_iface_t)(void);

#ifdef __cplusplus
}
#endif
