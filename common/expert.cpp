#include "expert.h"

#include "log.h"

#include "../ggml/src/ggml-cuda/expert-os.h"

#include <algorithm>
#include <cstdlib>
#include <cerrno>

static expert_validation expert_reject(const std::string & reason) {
    expert_validation res;
    res.ok     = false;
    res.reason = reason;
    return res;
}

expert_validation validate_expert_params(const common_params & params) {
    if (params.expert_l1_mib < 0) { return expert_reject("--expert-l1-mib must be 0 or more"); }
    if (params.expert_cache_mode != "inclusive" && params.expert_cache_mode != "exclusive") {
        return expert_reject("--expert-cache-mode must be 'inclusive' or 'exclusive', got '" + params.expert_cache_mode + "'");
    }
    if (params.expert_seed < 0) { return expert_reject("--expert-seed must be 0 or more"); }
    const bool active = params.expert_l1_mib > 0;
    if (active && (params.cpu_moe_explicit || params.n_cpu_moe_explicit)) {
        return expert_reject("expert budgets decide placement; do not combine them with --n-cpu-moe/--cpu-moe");
    }
    if (active && params.expert_cache_mode == "exclusive") {
        if (!ggml_cuda_expert::expert_os::supported()) { return expert_reject("exclusive storage needs Windows address reservation"); }
        if (params.fit_params && params.fit_params_explicit) { return expert_reject("--expert-cache-mode exclusive cannot be combined with -fit on"); }
    }
    if (!active && (!params.expert_profile_dir.empty() || params.expert_freeze || params.expert_profile_archive || params.expert_profile_reset)) {
        return expert_reject("expert profile options need an expert cache budget");
    }
    if (params.expert_profile_dir.empty() && (params.expert_profile_archive || params.expert_profile_reset)) {
        return expert_reject("expert profile archive/reset needs --expert-profile-dir");
    }
    return {};
}

ggml_expert_config expert_config_from_params(const common_params & params) {
    ggml_expert_config cfg = {};

    cfg.abi_version = GGML_EXPERT_ABI_VERSION;

    cfg.l1_bytes = (size_t) params.expert_l1_mib * 1024 * 1024;
    cfg.mode     = params.expert_cache_mode == "exclusive" ? GGML_EXPERT_MODE_EXCLUSIVE : GGML_EXPERT_MODE_INCLUSIVE;

    cfg.delta_install       = true;
    cfg.freeze              = params.expert_freeze;
    cfg.profile_archive     = params.expert_profile_archive;
    cfg.profile_reset       = params.expert_profile_reset;
    cfg.policy = params.expert_profile_dir.empty() ? GGML_EXPERT_POLICY_STATIC : GGML_EXPERT_POLICY_ADAPTIVE;
    cfg.random_seed = uint32_t(params.expert_seed);
    cfg.freeze = cfg.freeze || params.expert_profile_dir.empty();
    cfg.spare_slots = cfg.mode == GGML_EXPERT_MODE_EXCLUSIVE && cfg.policy == GGML_EXPERT_POLICY_ADAPTIVE && cfg.l1_bytes > 0 ? 8 : 0;

    cfg.profile_dir = params.expert_profile_dir.c_str();
    cfg.initial_bank = "decode";
    cfg.log_mask     = 0;

    const char * trace = getenv("RANMA_EXPERT_TRACE");
    if (trace != nullptr && trace[0] != '\0') {
        char * end = nullptr;
        errno = 0;
        const unsigned long mask = strtoul(trace, &end, 10);
        if (errno == 0 && end != nullptr && *end == '\0') {
            cfg.log_mask = (uint32_t) mask;
            COM_INF("expert cache: RANMA_EXPERT_TRACE sets log mask to %u\n", cfg.log_mask);
        } else {
            COM_WRN("expert cache: ignoring RANMA_EXPERT_TRACE='%s', expected a decimal bit mask\n", trace);
        }
    }

    return cfg;
}
