#pragma once

// ranma: turn the expert cache command line options into one backend config, and hold every rule
// about which combinations of those options are accepted.

#include "common.h"
#include "ggml-expert.h"

#include <string>
#include <vector>

struct expert_validation {
    bool        ok = true;
    std::string reason;
    // combinations that are accepted but not verified; the caller prints each once
    std::vector<std::string> warnings;
};

// One place for every rule about which option combinations are accepted. Called before the model loads.
expert_validation validate_expert_params(const common_params & params);

// True when a separate draft model is loaded and shares the target's expert cache (joint cache).
bool expert_draft_joins(const common_params & params);

// The config handed to the draft model's load in a joint cache: it names the draft's file, which
// the target's config declared, and nothing else.
ggml_expert_config expert_draft_config(const common_params & params);

// Builds the backend config from the params. The returned struct points into `params` strings, so
// `params` must outlive the model load. Also applies the debug override RANMA_EXPERT_TRACE=<mask>
// (decimal bit mask of ggml_expert_log_flags) to log_mask and prints one line when it does.
ggml_expert_config expert_config_from_params(const common_params & params);

// The cache reads its experts through the backend's host-direct path: turns that path on for every
// backend that has one, unless its own switch is set, with an in-place token limit of at least
// n_ubatch (see GGML_EXPERT_HOST_DIRECT_PROC_NAME). Call it before the model loads. Returns whether
// some backend has the path on.
bool expert_host_direct_auto(int32_t n_ubatch);

// True when the expert cache is on: an L1 budget (a size or auto) or a finite L2.
bool expert_cache_on(const common_params & params);

// --expert-l1-mib auto: the L1 budget in bytes for a device with `free` bytes free that the other weights (`model`), the
// KV cache (`context`) and the compute buffers (`compute`) will take from, less `margin` (--fit-target of the device);
// 0 when nothing is left. Whole MiB.
size_t expert_l1_auto_bytes(int64_t free, size_t model, size_t context, size_t compute, size_t margin);

// Resolves --expert-l1-mib auto: measures the model and the context as `mparams` / `cparams` will load them (a no_alloc
// load, the fit's own measurement) on the first GPU and sets params.expert_l1_mib from expert_l1_auto_bytes with
// --fit-target of that device (whether -fit is on or off). Nothing to do when auto was not given or the layer mode
// resolved it already (expert_l1_mib > 0). Throws when the measurement fails or leaves no L1.
void common_expert_l1_auto(common_params & params, const llama_model_params & mparams, const llama_context_params & cparams);
