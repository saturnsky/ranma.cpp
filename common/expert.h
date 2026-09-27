#pragma once

// ranma: turn the expert cache command line options into one backend config, and hold every rule
// about which combinations of those options are accepted.

#include "common.h"
#include "ggml-expert.h"

#include <string>

struct expert_validation {
    bool        ok = true;
    std::string reason;
};

// One place for every rule about which option combinations are accepted. Called before the model loads.
expert_validation validate_expert_params(const common_params & params);

// Builds the backend config from the params. The returned struct points into `params` strings, so
// `params` must outlive the model load. Also applies the debug override RANMA_EXPERT_TRACE=<mask>
// (decimal bit mask of ggml_expert_log_flags) to log_mask and prints one line when it does.
ggml_expert_config expert_config_from_params(const common_params & params);
