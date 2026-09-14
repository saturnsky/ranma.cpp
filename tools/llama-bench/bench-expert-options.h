#pragma once

#include "common.h"
#include "expert.h"

#include <limits>
#include <stdexcept>
#include <string>

struct bench_expert_options {
    std::string mode = "off";
    std::string storage = "inclusive";
    std::string profile;
    int l1_mib = 0;
    int seed = 1;
    bool archive = false;
    bool reset = false;
    bool supplied = false, keep = false, restore_each = false;

    bool enabled() const { return supplied; }
    bool warm() const { return mode == "warm"; }

    bool parse(const std::string & arg, int argc, char ** argv, int & i) {
        if (arg.compare(0, 9, "--expert-") != 0) { return false; }
        supplied = true;
        if (arg == "--expert-profile-archive") { archive = true; return true; }
        if (arg == "--expert-profile-reset") { reset = true; return true; }
        if (arg == "--expert-profile-keep") { keep = true; return true; }
        if (arg == "--expert-profile-restore-each") { restore_each = true; return true; }
        if (++i >= argc) { throw std::invalid_argument("missing expert option value"); }
        const std::string value = argv[i];
        if (arg == "--expert-cache") { mode = value; }
        else if (arg == "--expert-cache-mode") { storage = value; }
        else if (arg == "--expert-profile-dir") { profile = value; }
        else {
            size_t end = 0;
            const long long n = std::stoll(value, &end);
            if (end != value.size() || n < 0 || n > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("expert sizes and seed must be nonnegative integers");
            }
            if (arg == "--expert-l1-mib") { l1_mib = (int) n; }
            else if (arg == "--expert-seed") { seed = (int) n; }
            else { throw std::invalid_argument("unknown expert option: " + arg); }
        }
        return true;
    }

    ggml_expert_config config(int batch = 2048, int ubatch = 512) const {
        if (mode != "off" && mode != "cold" && mode != "warm") {
            throw std::invalid_argument("--expert-cache must be off, cold or warm");
        }
        if (!enabled()) { return {}; }
        if (mode == "off" && l1_mib != 0) { throw std::invalid_argument("off has no L1; --expert-l1-mib must be 0"); }
        if (mode != "off" && (profile.empty() || l1_mib == 0)) {
            throw std::invalid_argument("cold/warm need --expert-profile-dir and --expert-l1-mib > 0");
        }
        if (mode == "off" && (!profile.empty() || archive || reset)) { throw std::invalid_argument("off has no profile"); }
        if (!warm() && (keep || restore_each)) { throw std::invalid_argument("profile keep/restore-each requires warm"); }
        if (reset) { throw std::invalid_argument("cold requires an empty directory; profile reset is not supported by bench"); }
        common_params p;
        p.expert_l1_mib = l1_mib;
        p.expert_cache_mode = storage;
        p.expert_profile_dir = profile;
        p.expert_profile_archive = archive;
        p.expert_profile_reset = reset;
        p.n_parallel = 1; p.n_batch = batch; p.n_ubatch = ubatch;
        p.expert_seed = seed;
        p.fit_params = false;
        const auto valid = validate_expert_params(p);
        if (!valid.ok) { throw std::invalid_argument(valid.reason); }
        auto cfg = expert_config_from_params(p);
        cfg.profile_dir = mode == "off" ? "" : profile.c_str();
        cfg.l1_bytes = size_t(l1_mib)*1024*1024;
        cfg.policy = warm() ? GGML_EXPERT_POLICY_ADAPTIVE : GGML_EXPERT_POLICY_STATIC;
        cfg.random_seed = (uint32_t) seed;
        cfg.freeze = !warm();
        if (!warm()) { cfg.spare_slots = 0; }
        if (mode == "off") { cfg.initial_bank = ""; }
        cfg.log_mask |= GGML_EXPERT_LOG_PROFILE | GGML_EXPERT_LOG_INSTALL;
        return cfg;
    }
};
