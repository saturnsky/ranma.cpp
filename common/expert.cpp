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
    if (params.expert_l2_worker_cpu < -1) { return expert_reject("--expert-l2-worker-cpu must be -1 or a logical CPU index"); }
    if (params.expert_l2_mib < -1 || params.expert_l2_staging_mib < 0 ||
            params.expert_l2_prefill_ring_mib < -1 || params.expert_l2_decode_ring_mib < -1) {
        return expert_reject("the --expert-l2-* sizes must be 0 or more");
    }
    // Follows --cache-ram: -1 is no limit, but an empty host tier has no meaning here because the
    // experts that are not in VRAM have nowhere else to live except the file.
    if (params.expert_l2_mib == 0) {
        return expert_reject("--expert-l2-mib 0 is not valid: the host tier cannot be empty; use -1 for unlimited");
    }
    const bool active = params.expert_l1_mib > 0 || params.expert_l2_mib > 0;
    if (active && (params.cpu_moe_explicit || params.n_cpu_moe_explicit)) {
        return expert_reject("expert budgets decide placement; do not combine them with --n-cpu-moe/--cpu-moe");
    }
    if (active && params.expert_cache_mode == "exclusive") {
        if (!ggml_cuda_expert::expert_os::supported()) { return expert_reject("exclusive storage needs Windows address reservation"); }
        if (params.fit_params && params.fit_params_explicit) { return expert_reject("--expert-cache-mode exclusive cannot be combined with -fit on"); }
    }
    const bool speculative = params.speculative.has_dft() || params.speculative.has_synth() ||
        std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) { return t != COMMON_SPECULATIVE_TYPE_NONE; });
    if (params.expert_l2_mib > 0) {
        if (!ggml_cuda_expert::expert_os::supported()) { return expert_reject("finite L2 needs Windows unbuffered reads"); }
        if (params.n_parallel <= 0 || params.n_batch <= 0 || params.n_ubatch < 0) { return expert_reject("finite L2 needs positive batch/parallel bounds and a nonnegative ubatch"); }
        if (!params.mmproj.path.empty()) { return expert_reject("finite L2 with mmproj has no gate yet"); }
        if (speculative) { return expert_reject("finite L2 with speculative decoding has no gate yet"); }
        const int p = params.expert_l2_prefill_ring_mib >= 0 ? params.expert_l2_prefill_ring_mib : params.expert_l2_staging_mib;
        const int d = params.expert_l2_decode_ring_mib >= 0 ? params.expert_l2_decode_ring_mib : params.expert_l2_staging_mib;
        if (std::max(p, params.expert_prefill_swap ? d : p) > params.expert_l2_mib) { return expert_reject("configured L2 ring exceeds --expert-l2-mib; metadata and residents also need space"); }
        // The exact minimum, including the L1 payload upper bound for inclusive, needs model geometry.
    } else if (params.expert_l2_staging_mib != 0 || params.expert_l2_prefill_ring_mib >= 0 || params.expert_l2_decode_ring_mib >= 0) {
        return expert_reject("the --expert-l2 ring sizes have no effect without --expert-l2-mib");
    }
    if (params.expert_prefill_swap) {
        if (!active) { return expert_reject("--expert-prefill-swap needs an expert cache budget"); }
        if (params.expert_profile_dir.empty()) { return expert_reject("--expert-prefill-swap needs --expert-profile-dir"); }
        if (params.n_parallel != 1) { return expert_reject("--expert-prefill-swap needs -np 1"); }
        if (!params.mmproj.path.empty() || speculative) { return expert_reject("--expert-prefill-swap with mmproj/speculative decoding has no gate yet"); }
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

    const size_t mib = 1024*1024;
    const int32_t prefill_mib = params.expert_l2_prefill_ring_mib >= 0 ?
        params.expert_l2_prefill_ring_mib : params.expert_l2_staging_mib;
    int32_t decode_mib = params.expert_l2_decode_ring_mib >= 0 ?
        params.expert_l2_decode_ring_mib : params.expert_l2_staging_mib;
    // Without the prefill swap there is one ring, prompt sized: nothing ever installs a plan that
    // would want the small one.
    if (!params.expert_prefill_swap) {
        decode_mib = prefill_mib;
    }
    // The backend spells an unlimited host arena 0; the option spells it -1.
    cfg.l2_bytes              = params.expert_l2_mib > 0 ? (size_t) params.expert_l2_mib*mib : 0;
    cfg.l2_prefill_ring_bytes = (size_t) prefill_mib*mib;
    cfg.l2_decode_ring_bytes  = (size_t) decode_mib*mib;
    cfg.l2_worker_cpu         = params.expert_l2_worker_cpu;
    cfg.l2_prefill_rows       = uint64_t(params.n_ubatch > 0 ? params.n_ubatch : std::max(1, params.n_batch));
    const bool speculative = params.speculative.has_dft() || params.speculative.has_synth() ||
        std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) { return t != COMMON_SPECULATIVE_TYPE_NONE; });
    cfg.l2_decode_rows = uint64_t(std::max(1, params.n_parallel)) *
        (speculative ? uint64_t(std::max(0, params.speculative.draft.n_max)) + 1 : 1);
    cfg.l2_phase_rings        = params.expert_prefill_swap;

    cfg.delta_install       = true;
    cfg.freeze              = params.expert_freeze;
    cfg.profile_archive     = params.expert_profile_archive;
    cfg.profile_reset       = params.expert_profile_reset;
    cfg.policy = params.expert_profile_dir.empty() ?
        (params.expert_l1_mib == 0 ? GGML_EXPERT_POLICY_OFF : GGML_EXPERT_POLICY_STATIC) : GGML_EXPERT_POLICY_ADAPTIVE;
    cfg.random_seed = uint32_t(params.expert_seed);
    cfg.freeze = cfg.freeze || params.expert_profile_dir.empty();
    cfg.spare_slots = cfg.mode == GGML_EXPERT_MODE_EXCLUSIVE && cfg.policy == GGML_EXPERT_POLICY_ADAPTIVE && cfg.l1_bytes > 0 ? 8 : 0;

    cfg.profile_dir = params.expert_profile_dir.c_str();
    // The plan installed at load is the plan of the phase the next compute will be in: with the
    // swap on that is prompt processing, otherwise generation. The other bank is the fallback for
    // a profile directory that has only ever seen one of them.
    cfg.initial_bank = params.expert_prefill_swap ? "prefill,decode" : "decode";
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
