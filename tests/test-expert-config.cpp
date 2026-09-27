// Expert option contracts before model geometry is available.
#include "common.h"
#include "expert.h"

#include <cstdio>
#include <fstream>
#include "../tools/llama-bench/bench-expert-profiles.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    common_params p;
    CHECK(validate_expert_params(p).ok);
    p.expert_l1_mib = -1; CHECK(!validate_expert_params(p).ok); p.expert_l1_mib = 0;
    p.expert_seed = -1; CHECK(!validate_expert_params(p).ok); p.expert_seed = 1;
    p.expert_profile_dir = "profile"; CHECK(!validate_expert_params(p).ok); p.expert_profile_dir.clear();
    for (int l1 : {0, 3070}) for (bool profile : {false, true}) {
        p = common_params(); p.fit_params = false;
        p.expert_l1_mib = l1; p.n_parallel = 4;
        p.expert_profile_dir = profile ? "profile" : "";
        CHECK(validate_expert_params(p).ok == (l1 > 0 || !profile));
        const auto cfg = expert_config_from_params(p);
        CHECK(cfg.l1_bytes == size_t(l1)*1024*1024);
        CHECK(cfg.random_seed == 1);
        CHECK(profile ? cfg.policy == GGML_EXPERT_POLICY_ADAPTIVE : cfg.freeze);
    }
    p = common_params(); p.fit_params = false;
    p.expert_l1_mib = 3070;
    CHECK(validate_expert_params(p).ok); // The profile directory is optional.
    p.expert_l1_mib = 20000;
    CHECK(validate_expert_params(p).ok);
    p.n_cpu_moe_explicit = true; CHECK(!validate_expert_params(p).ok); p.n_cpu_moe_explicit = false;
    p.expert_seed = 73;
    CHECK(expert_config_from_params(p).random_seed == 73);
    bench_expert_options o; o.supplied = true;
    auto off = o.config();
    CHECK(off.l1_bytes == 0 && off.profile_dir[0] == 0);
    o.mode = "cold"; o.l1_mib = 3070; o.profile = "fixture";
    auto cold = o.config(1024, 128);
    CHECK(cold.policy == GGML_EXPERT_POLICY_STATIC && cold.freeze);
    o.mode = "warm";
    { auto warm = o.config(); CHECK(warm.policy == GGML_EXPERT_POLICY_ADAPTIVE && !warm.freeze); }
    const auto temporary = std::filesystem::temp_directory_path() /
        ("bench-profile-fixture-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    CHECK(std::filesystem::create_directory(temporary));
    const auto source = temporary / "source";
    CHECK(std::filesystem::create_directory(source));
    { std::ofstream file(source / "record"); file << "seed"; }
    o.profile = source.string(); o.restore_each = true;
    std::filesystem::path owned;
    {
        bench_expert_profiles work(o); owned = work.root;
        CHECK(std::filesystem::exists(work.active / "record"));
        { std::ofstream file(work.active / "new-record"); file << "updated"; }
        work.restore();
        CHECK(!std::filesystem::exists(work.active / "new-record"));
        CHECK(std::filesystem::exists(source / "record") && !std::filesystem::exists(source / "new-record"));
    }
    CHECK(!std::filesystem::exists(owned));
    o.mode = "cold"; o.restore_each = false;
    bool rejected = false;
    try { bench_expert_profiles work(o); } catch (const std::exception &) { rejected = true; }
    CHECK(rejected);
    o.profile = (temporary / "cold").string();
    { bench_expert_profiles work(o); CHECK(work.root.empty() && work.active == o.profile); }
    CHECK(std::filesystem::exists(source / "record"));
    std::filesystem::remove_all(temporary);
    printf("PASS: bench modes, source-preserving process snapshot, restore and owned cleanup\n");
    printf("PASS: cache modes, zero L1 and the optional profile directory\n");
    return 0;
}
