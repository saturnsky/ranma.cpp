// Expert option contracts before model geometry is available.
#include "common.h"
#include "expert.h"
#include "../ggml/src/ggml-cuda/expert-os.h"

#include <cstdio>
#include <fstream>
#include "../tools/llama-bench/bench-expert-profiles.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    common_params p;
    CHECK(validate_expert_params(p).ok);
    p.expert_l1_mib = -1; CHECK(!validate_expert_params(p).ok); p.expert_l1_mib = 0;
    p.expert_cache_mode = "invalid"; CHECK(!validate_expert_params(p).ok); p.expert_cache_mode = "inclusive";
    p.expert_seed = -1; CHECK(!validate_expert_params(p).ok); p.expert_seed = 1;
    p.expert_l2_worker_cpu = -2; CHECK(!validate_expert_params(p).ok); p.expert_l2_worker_cpu = -1;
    p.expert_profile_dir = "profile"; CHECK(!validate_expert_params(p).ok); p.expert_profile_dir.clear();
    p.expert_l2_staging_mib = 1; CHECK(!validate_expert_params(p).ok); p.expert_l2_staging_mib = 0;
    // -1 is the unlimited host arena; 0 would be an empty one, which has no meaning.
    CHECK(p.expert_l2_mib == -1 && validate_expert_params(p).ok);
    p.expert_l2_mib = 0; CHECK(!validate_expert_params(p).ok);
    p.expert_l2_mib = -2; CHECK(!validate_expert_params(p).ok); p.expert_l2_mib = -1;
    CHECK(expert_config_from_params(p).l2_bytes == 0);
    const bool os = ggml_cuda_expert::expert_os::supported();
    for (const char * mode : {"inclusive", "exclusive"}) for (int l1 : {0, 3070}) for (bool profile : {false, true}) {
        p = common_params(); p.fit_params = false;
        p.expert_cache_mode = mode; p.expert_l1_mib = l1; p.expert_l2_mib = 8000; p.n_parallel = 4;
        p.expert_profile_dir = profile ? "profile" : "";
        CHECK(validate_expert_params(p).ok == os);
        const auto cfg = expert_config_from_params(p);
        CHECK(cfg.l1_bytes == size_t(l1)*1024*1024 && cfg.l2_bytes == size_t(8000)*1024*1024);
        CHECK(cfg.l2_decode_rows == 4 && cfg.l2_prefill_rows == 512);
        CHECK(cfg.random_seed == 1 && cfg.l2_worker_cpu == -1);
        CHECK(profile ? cfg.policy == GGML_EXPERT_POLICY_ADAPTIVE : cfg.freeze);
        CHECK(l1 && profile && std::string(mode) == "exclusive" ? cfg.spare_slots == 8 : cfg.spare_slots == 0);
        p.expert_prefill_swap = true;
        CHECK(!validate_expert_params(p).ok);
        p.n_parallel = 1; CHECK(validate_expert_params(p).ok == (os && profile));
        p.expert_prefill_swap = false;
        p.mmproj.path = "fixture.gguf";
        CHECK(!validate_expert_params(p).ok);
        if (os) { CHECK(validate_expert_params(p).reason.find("no gate yet") != std::string::npos); }
        p.mmproj.path.clear(); p.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
        CHECK(!validate_expert_params(p).ok);
        p.speculative.draft.n_max = 7; p.n_parallel = 3;
        CHECK(expert_config_from_params(p).l2_decode_rows == 24);
    }
    p = common_params(); p.fit_params = false;
    p.expert_l1_mib = 3070;
    CHECK(validate_expert_params(p).ok); // Profile is optional even without finite L2.
    p.expert_l2_mib = 1;
    CHECK(validate_expert_params(p).ok == os); // Exact ring floor belongs to geometry validation.
    p.expert_l2_staging_mib = 2; CHECK(!validate_expert_params(p).ok);
    p.expert_l2_mib = 8000; p.expert_l2_staging_mib = 0;
    p.expert_l1_mib = 20000;
    CHECK(validate_expert_params(p).ok == os); // P_l1 cannot be inferred from the CLI L1 budget.
    p.n_cpu_moe_explicit = true; CHECK(!validate_expert_params(p).ok); p.n_cpu_moe_explicit = false;
    p.expert_seed = 73; p.expert_l2_worker_cpu = 2; p.n_ubatch = 128;
    auto cfg = expert_config_from_params(p);
    CHECK(cfg.random_seed == 73 && cfg.l2_worker_cpu == 2 && cfg.l2_prefill_rows == 128);
    p.n_ubatch = 0; p.n_batch = 1024;
    CHECK(expert_config_from_params(p).l2_prefill_rows == 1024);
    bench_expert_options o; o.supplied = true;
    auto off = o.config();
    CHECK(off.l1_bytes == 0 && off.policy == GGML_EXPERT_POLICY_OFF && off.profile_dir[0] == 0 && off.spare_slots == 0);
    CHECK(o.l2_mib == -1 && off.l2_bytes == 0);
    o.mode = "cold"; o.l1_mib = 3070; o.profile = "fixture";
    auto cold = o.config(1024, 128);
    CHECK(cold.policy == GGML_EXPERT_POLICY_STATIC && cold.freeze && cold.l2_prefill_rows == 128);
    o.mode = "warm"; o.storage = "exclusive";
    if (os) { auto warm = o.config(); CHECK(warm.policy == GGML_EXPERT_POLICY_ADAPTIVE && !warm.freeze && warm.spare_slots == 8); }
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
    printf("PASS: finite modes, zero L1, optional profile, phase restrictions, seed/worker and batch bounds\n");
    return 0;
}
