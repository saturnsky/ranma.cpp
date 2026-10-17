// The verification cost of the target shared by every drafter (common/speculative-verify-cost.h) on the CPU: a fixed
// curve followed, a change common to all widths, sparse widths and the blend with a prior table, variance and decay,
// the store (round trip, another key or width, damaged files), the widths 1, 9 and 17 of a 9-row learner, bad values,
// the graph builds taken out, the calls after a lend of the decode pool, an incomplete call, a duplicate call, the rounds
// (the time besides the target call, a drafter that adds its own work, a replay, calls of other sequences, the drafter
// of a round), the --spec-smart controller on the shared cost (the same decisions as on its own cost, its calibration
// learnt from its own rounds only) and the hazard model of ngram-varlen with a cost table given.
#include "common.h"
#include "ngram-varlen.h"
#include "speculative-smart.h"
#include "speculative-verify-cost.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string tmp(const char * name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

// a call of one sequence that generates (a verification of r rows)
static common_spec_verify_sample call(int64_t id, int32_t r, double us, double graph_us = 0.0) {
    common_spec_verify_sample s;
    s.call_id   = id;
    s.r         = r;
    s.target_us = us;
    s.graph_us  = graph_us;
    s.seq       = 0;
    return s;
}

// DS4.1F-like curve of the target call (us) per row count
static double curve(int32_t r) {
    static const double C[9] = { 39000, 56000, 70000, 83000, 96000, 108000, 122000, 137000, 160000 };
    return C[std::clamp(r, 1, 9) - 1];
}

// every width once (the first call of a width is not observed), then n calls of random widths with noise
static void feed(common_spec_verify_cost & vc, int64_t & id, int n, double scale, std::mt19937 & rng, double noise = 0.03) {
    std::uniform_int_distribution<int> w(1, vc.cfg.r_max);
    std::normal_distribution<double> e(0.0, noise);
    for (int i = 0; i < n; ++i) {
        const int r = w(rng);
        vc.observe(call(id++, r, curve(r) * scale * (1.0 + e(rng))));
    }
}

static int test_curve_and_level() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    std::mt19937 rng(1);
    int64_t id = 0;
    // the first call of every width builds its graphs: not observed, the clock still counts it
    for (int r = 1; r <= 9; ++r) {
        CHECK(std::string(vc.observe(call(id++, r, 1e9))) == "untimed");
    }
    CHECK(vc.level() == 0.0 && vc.clock == 9 && vc.n_core == 0);
    feed(vc, id, 3000, 1.0, rng);
    const auto t = vc.core_table_m(9);
    for (int r = 1; r <= 9; ++r) {
        CHECK(std::fabs(t[r - 1] / curve(r) - 1.0) < 0.03); // follows the curve within the noise
        const auto e = vc.estimate(r);
        CHECK(e.origin == COMMON_SPEC_COST_OBSERVED && e.shape_n > 20.0 && e.age >= 0 && e.aux_us == 0.0);
        CHECK(e.variance > 0.0 && e.variance < 0.01); // the ratio's spread of 3 % noise
    }
    // a change common to all widths (twice as slow): the level follows within a few of its half-lives, the shape stays
    feed(vc, id, 300, 2.0, rng);
    const auto t2 = vc.core_table_m(9);
    for (int r = 1; r <= 9; ++r) {
        CHECK(std::fabs(t2[r - 1] / (2.0 * curve(r)) - 1.0) < 0.06);
    }
    fprintf(stderr, "curve: r=9 %.1f ms (true %.1f), after x2 %.1f ms; %s\n", t[8] / 1000.0, curve(9) / 1000.0,
            t2[8] / 1000.0, vc.summary().c_str());
    return 0;
}

static int test_decay_and_variance() {
    common_spec_verify_cost_config c;
    c.r_max = 3;
    c.half_life = 100.0;
    common_spec_verify_cost vc(c);
    int64_t id = 0;
    vc.observe(call(id++, 2, 1.0)); // untimed
    vc.observe(call(id++, 2, 1000.0));
    vc.observe(call(id++, 2, 3000.0));
    // the level: the first sets it (1000), the second moves it (no confident shape: every width is a source) with the
    // weight of the first decayed by one call; the shape of width 2 is 1, then 3000 / 1000 (the level before) = 3
    const double wl = vc.level_decay;
    CHECK(std::fabs(vc.level() - (wl * 1000.0 + 3000.0) / (wl + 1.0)) < 1e-9);
    const double ws = vc.decay;
    const double sm = 1.0 + 2.0 / (ws + 1.0);
    CHECK(std::fabs(vc.shape_cells[1].mean - sm) < 1e-12);
    CHECK(std::fabs(vc.shape_cells[1].var - 2.0 * (3.0 - sm) / (ws + 1.0)) < 1e-12);
    const double mean0 = vc.shape_cells[1].mean;
    const double var0  = vc.shape_cells[1].var;
    const double n0 = vc.conf(vc.shape_cells[1]);
    // 100 calls of another width: the confidence of width 2 halves, its mean and variance stay
    for (int i = 0; i < 100; ++i) {
        vc.observe(call(id++, 1, 500.0));
    }
    CHECK(std::fabs(vc.conf(vc.shape_cells[1]) - n0 * 0.5) < 1e-9);
    CHECK(vc.shape_cells[1].mean == mean0 && vc.shape_cells[1].var == var0);
    return 0;
}

static int test_bad_wide_graph_dup() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    int64_t id = 0;
    for (int r : { 1, 9 }) {
        vc.observe(call(id++, r, 1.0)); // untimed
    }
    // r = 1 and r = 9 are learnt; r = 17 is wider than any piece: counted, not learnt
    CHECK(vc.observe(call(id++, 1, 40000.0)) == nullptr);
    CHECK(vc.observe(call(id++, 9, 160000.0)) == nullptr);
    CHECK(std::string(vc.observe(call(id++, 17, 300000.0))) == "wide");
    CHECK(vc.cost_n_run[0] == 1 && vc.cost_n_run[8] == 1 && vc.level() == 40000.0);
    // NaN, negative, infinite: not learnt
    const int64_t clock0 = vc.clock;
    CHECK(std::string(vc.observe(call(id++, 2, NAN))) == "bad");
    CHECK(std::string(vc.observe(call(id++, 2, -5.0))) == "bad");
    CHECK(std::string(vc.observe(call(id++, 2, INFINITY))) == "bad");
    CHECK(std::string(vc.observe(call(id++, 2, 50000.0, NAN))) == "bad");
    CHECK(vc.clock == clock0 + 4 && vc.shape_cells[1].n == 0.0 && !vc.used_run[1]);
    // the host time of graph builds and captures in the call is taken out
    vc.observe(call(id++, 3, 1.0)); // untimed
    common_spec_verify_obs o;
    CHECK(vc.observe(call(id++, 3, 70000.0 + 12000.0, 12000.0), &o) == nullptr);
    CHECK(o.obs_us == 70000.0 && std::fabs(vc.shape_cells[2].mean - 70000.0 / 40000.0) < 1e-12);
    // a duplicate call id: nothing happens, the clock stays
    const int64_t clock1 = vc.clock;
    CHECK(std::string(vc.observe(call(id - 1, 3, 1e6))) == "dup");
    CHECK(vc.clock == clock1 && vc.cost_n_run[2] == 1);
    // an incomplete call (an error, no synchronisation): no time, no clock
    auto f = call(id++, 3, 1e6);
    f.ok = false;
    CHECK(std::string(vc.observe(f)) == "failed" && vc.clock == clock1);
    // a prompt call: not a verification, no clock; a call with prompt tokens and generated ones, several sequences or
    // a replay: counted (clock), not learnt
    auto p = call(id++, 0, 1e6);
    p.n_prompt = 512;
    CHECK(std::string(vc.observe(p)) == "prompt" && vc.clock == clock1);
    auto mx = call(id++, 2, 1e6);
    mx.n_prompt = 100;
    CHECK(std::string(vc.observe(mx)) == "mixed");
    auto ms = call(id++, 2, 1e6);
    ms.n_seq = 2;
    CHECK(std::string(vc.observe(ms)) == "multi");
    auto rp = call(id++, 2, 1e6);
    rp.replay = true;
    CHECK(std::string(vc.observe(rp)) == "replay");
    CHECK(vc.clock == clock1 + 3 && vc.shape_cells[1].n == 0.0 && vc.cost_n_run[2] == 1);
    return 0;
}

static int test_lend() {
    common_spec_verify_cost_config c;
    c.r_max = 2;
    common_spec_verify_cost vc(c);
    int64_t id = 0;
    auto lc = [&](uint64_t lends) {
        auto s = call(id++, 1, 40000.0);
        s.pool_valid = true;
        s.lends = lends;
        return s;
    };
    vc.observe(lc(0)); // untimed
    CHECK(vc.observe(lc(0)) == nullptr);
    // a lend (in the prefill before): the call that sees it and the next 4 are not learnt, the 6th is
    for (int i = 0; i < 5; ++i) {
        CHECK(std::string(vc.observe(lc(1))) == "lend");
    }
    CHECK(vc.observe(lc(1)) == nullptr);
    CHECK(vc.cost_n_run[0] == 2);
    // a prompt call between does not count in the window
    auto p = call(id++, 0, 1e6);
    p.n_prompt = 100;
    p.pool_valid = true;
    p.lends = 2;
    vc.observe(p);
    CHECK(std::string(vc.observe(lc(2))) == "lend" && vc.pool_since_lend == 0);
    return 0;
}

// A plain reference of the smoothed lookup (O(R^2), the isotonic fit by the min-max formula): the contract of
// blend_ms() / smooth_us(). segment border: after the 8th row.
static std::vector<double> ref_smooth(const common_spec_verify_cost & vc, const std::vector<double> & prior, double unit) {
    const int R = vc.cfg.r_max;
    const int SEG = 8;
    const double k = vc.cfg.prior_n;
    const std::vector<double> eff = vc.eff_table_m(R);
    std::vector<double> n(R);
    for (int i = 0; i < R; ++i) {
        n[i] = vc.conf(vc.shape_cells[i]);
    }
    bool use_prior = !prior.empty();
    for (int r = 1; r <= R; ++r) {
        const double p = common_spec_verify_cost::prior_at(prior, r);
        use_prior = use_prior && std::isfinite(p) && p > 0.0;
    }
    double s = 1.0;
    if (use_prior && n[0] > 0.0) {
        const double w1 = n[0] / (n[0] + k);
        s = w1 * eff[0] / unit / prior[0] + (1.0 - w1);
    }
    std::vector<double> B(R), W(R);
    for (int lo = 0; lo < R; lo = (lo < SEG ? std::min(SEG, R) : R)) {
        const int hi = lo < SEG ? std::min(SEG, R) : R;
        bool any = false;
        for (int i = lo; i < hi; ++i) {
            any = any || n[i] > 0.0;
        }
        for (int i = lo; i < hi; ++i) {
            const double E = eff[i] / unit;
            if (!use_prior) {
                B[i] = E;
                W[i] = std::max(n[i], 1e-6);
                continue;
            }
            const double P = common_spec_verify_cost::prior_at(prior, i + 1);
            int l = -1, g = -1;
            for (int j = i - 1; j >= lo && l < 0; --j) {
                if (n[j] > 0.0) { l = j; }
            }
            for (int j = i + 1; j < hi && g < 0; ++j) {
                if (n[j] > 0.0) { g = j; }
            }
            double b = 1.0;
            if (!any) {
                b = s;
            } else {
                double lw = 0.0, gw = 0.0;
                if (l >= 0 && g >= 0) { lw = (double) (g - i) / (g - l); gw = (double) (i - l) / (g - l); }
                else if (l >= 0)      { lw = 1.0; }
                else if (g >= 0)      { gw = 1.0; }
                double num = k, den = k;
                if (l >= 0) { num += lw * n[l] * eff[l] / unit / prior[l]; den += lw * n[l]; }
                if (g >= 0) { num += gw * n[g] * eff[g] / unit / prior[g]; den += gw * n[g]; }
                b = den > 0.0 ? num / den : 1.0;
            }
            B[i] = (n[i] * E + k * P * b) / (n[i] + k);
            W[i] = std::max(n[i] + k, 1e-6);
        }
        std::vector<double> C(hi - lo);
        for (int i = lo; i < hi; ++i) { // iso(i) = max over s <= i of min over t >= i of the weighted mean of s..t
            double best = -1e300;
            for (int a = lo; a <= i; ++a) {
                double lowest = 1e300;
                for (int t = i; t < hi; ++t) {
                    double sw = 0.0, sv = 0.0;
                    for (int j = a; j <= t; ++j) { sw += W[j]; sv += W[j] * B[j]; }
                    lowest = std::min(lowest, sv / sw);
                }
                best = std::max(best, lowest);
            }
            C[i - lo] = best;
        }
        for (int i = lo; i < hi; ++i) {
            B[i] = C[i - lo];
        }
    }
    return std::vector<double>(B.begin(), B.end());
}

// width r (1 ..) observed `count` times at `us` after its untimed first call
static void observe_width(common_spec_verify_cost & vc, int64_t & id, int r, int count, double us) {
    vc.observe(call(id++, r, 1.0)); // the first call of a width in a process is not observed
    for (int i = 0; i < count; ++i) {
        vc.observe(call(id++, r, us));
    }
}

static const std::vector<double> SMOOTH_PRIOR = { 39.2, 56.9, 72.4, 84.8, 97.3, 109.7, 124.4, 139.0, 162.4 };

static int test_sparse_blend() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    const std::vector<double> prior = { 39.2, 56.9, 72.4, 84.8, 97.3, 109.7, 124.4, 139.0, 162.4 };
    // nothing observed: the prior exactly (also beyond the table: its last step)
    auto b0 = vc.blend_ms(prior, 11);
    for (int r = 1; r <= 9; ++r) {
        CHECK(b0[r - 1] == prior[r - 1]);
    }
    CHECK(std::fabs(b0[9] - (162.4 + (162.4 - 139.0))) < 1e-9);
    // only widths 1..3 observed, all 20 % faster than the prior: the rest is the prior at the level of r = 1
    int64_t id = 0;
    for (int k = 0; k < 2; ++k) {
        for (int r = 1; r <= 3; ++r) {
            for (int i = 0; i < (k == 0 ? 1 : 200); ++i) {
                vc.observe(call(id++, r, 0.8 * prior[r - 1] * 1000.0));
            }
        }
    }
    const auto b = vc.blend_ms(prior, 9);
    for (int r = 1; r <= 3; ++r) {
        CHECK(std::fabs(b[r - 1] / (0.8 * prior[r - 1]) - 1.0) < 0.01); // observed: w close to 1
    }
    for (int r = 4; r <= 9; ++r) {
        CHECK(std::fabs(b[r - 1] / (0.8 * prior[r - 1]) - 1.0) < 0.01); // not observed: the prior ratio
        CHECK(vc.estimate(r).origin == COMMON_SPEC_COST_INTERP && vc.estimate(r).shape_n == 0.0);
    }
    // a width observed twice: weight 2 / (2 + 2) between its observation and the neighbours' ratio (the nearest observed
    // width on its left, width 3; nothing observed on its right), then the rise with the width
    vc.observe(call(id++, 7, 1.0)); // untimed
    vc.observe(call(id++, 7, 150000.0));
    vc.observe(call(id++, 7, 150000.0));
    const auto b7 = vc.blend_ms(prior, 9);
    const double n7 = vc.conf(vc.shape_cells[6]);
    const double w7 = n7 / (n7 + 2.0);
    const double n3 = vc.conf(vc.shape_cells[2]);
    const double r3 = vc.eff_table_m(9)[2] / 1000.0 / prior[2];
    const double bn = (2.0 + n3 * r3) / (2.0 + n3);
    CHECK(std::fabs(b7[6] - (n7 * vc.eff_table_m(9)[6] / 1000.0 + 2.0 * prior[6] * bn) / (n7 + 2.0)) < 1e-9);
    CHECK(w7 > 0.4 && w7 < 0.5);
    const auto ref7 = ref_smooth(vc, prior, 1000.0);
    for (int r = 1; r <= 9; ++r) {
        CHECK(std::fabs(b7[r - 1] - ref7[r - 1]) < 1e-9);
        CHECK(r == 1 || r == 9 || b7[r - 1] >= b7[r - 2]);
    }
    fprintf(stderr, "blend: r=7 %.1f ms (observed %.1f, prior x neighbours %.1f), r=9 %.1f ms\n", b7[6],
            vc.eff_table_m(9)[6] / 1000.0, prior[6] * bn, b7[8]);
    return 0;
}

static int test_store() {
    const std::string path  = tmp("test-spec-verify-cost.txt");
    const std::string path2 = tmp("test-spec-verify-cost-2.txt");
    std::mt19937 rng(3);
    common_spec_verify_cost_config c;
    c.r_max = 9;
    c.store_key = "target=a.gguf@1:2 build=b1 l1=20480 env=[GGML_X=1]";
    {
        common_spec_verify_cost w(c);
        int64_t id = 0;
        feed(w, id, 9, 1.0, rng);
        feed(w, id, 500, 1.0, rng);
        w.observe_aux(3, 7000.0, nullptr); // untimed
        w.observe_aux(3, 7000.0, nullptr);
        w.observe_aux(3, 9000.0, nullptr);
        CHECK(w.save(path));

        // exact restore when the load caps are out of the way, and save -> load -> save byte-identical
        auto c2 = c;
        c2.cap_load = 1e300;
        c2.level_cap_load = 1e300;
        c2.store_path = path;
        {
            common_spec_verify_cost r(c2);
            CHECK(r.store_loaded && r.clock == 0);
            CHECK(r.level_cell.mean == w.level_cell.mean && r.level_cell.n == w.level_conf());
            for (int m = 0; m < 9; ++m) {
                CHECK(r.shape_cells[m].mean == w.shape_cells[m].mean && r.shape_cells[m].var == w.shape_cells[m].var);
                CHECK(r.shape_cells[m].n == w.conf(w.shape_cells[m]) && r.cost_n_run[m] == 0 && !r.used_run[m]);
            }
            CHECK(r.aux_cells[2].mean == 8000.0 && r.aux_cells[2].n == w.conf(w.aux_cells[2]));
            r.cfg.store_path.clear();
            CHECK(r.save(path2));
            CHECK(read_file(path) == read_file(path2));
        }
        // the normal load: confidence capped at 32, the level at 2, the wait at 16
        auto c3 = c;
        c3.store_path = path;
        {
            common_spec_verify_cost s(c3);
            CHECK(s.store_loaded && s.level_cell.n == 2.0);
            for (int m = 0; m < 9; ++m) {
                CHECK(s.shape_cells[m].n == std::min(w.conf(w.shape_cells[m]), 32.0));
                CHECK(s.wait_conf(m) == std::min(w.conf(w.shape_cells[m]), 16.0));
            }
            // the costs restore exactly at the level of the store
            const auto ts = s.core_table_m(9);
            const auto tw = w.core_table_m(9);
            for (int m = 0; m < 9; ++m) {
                CHECK(std::fabs(ts[m] - tw[m]) < 1e-6 * tw[m]);
            }
            s.cfg.store_path.clear();
        }
        // another key: cold
        auto c4 = c3;
        c4.store_key = "target=a.gguf@1:2 build=b1 l1=16384 env=[GGML_X=1]";
        {
            common_spec_verify_cost k(c4);
            CHECK(!k.store_loaded && k.level() == 0.0);
            k.cfg.store_path.clear();
        }
        // a wider learner reads the widths both have; a narrower one its own
        auto c5 = c3;
        c5.r_max = 12;
        {
            common_spec_verify_cost m(c5);
            CHECK(m.store_loaded && m.shape_cells[8].n > 0.0 && m.shape_cells[9].n == 0.0 && m.aux_cells[2].mean == 8000.0);
            CHECK(m.estimate(12).origin == COMMON_SPEC_COST_INTERP);
            m.cfg.store_path.clear();
        }
        auto c6 = c3;
        c6.r_max = 3;
        {
            common_spec_verify_cost n(c6);
            CHECK(n.store_loaded && n.shape_cells.size() == 3 && n.shape_cells[2].mean == w.shape_cells[2].mean);
            n.cfg.store_path.clear();
        }
        // a truncated file, a damaged line, a store of the --spec-smart controller: not used as a whole
        {
            std::string t = read_file(path);
            std::ofstream(path2, std::ios::binary | std::ios::trunc) << t.substr(0, t.size() / 2);
            auto c7 = c;
            c7.store_path = path2;
            common_spec_verify_cost tr(c7);
            CHECK(!tr.store_loaded && tr.level() == 0.0);
            tr.cfg.store_path.clear();
        }
        {
            std::string t = read_file(path);
            t.replace(t.find("shape 4 "), 8, "shape 4 x");
            std::ofstream(path2, std::ios::binary | std::ios::trunc) << t;
            auto c7 = c;
            c7.store_path = path2;
            std::string why;
            common_spec_verify_cost d(c);
            CHECK(!d.load(path2, &why) && why == "bad shape line" && !d.store_loaded);
        }
        {
            std::ofstream(path2, std::ios::binary | std::ios::trunc)
                << "ranma-spec-smart 6\nkey " << c.store_key << "\nshape 3 20 3\nlevel 40000 10\nratio 0 1 10 0\nend\n";
            std::string why;
            common_spec_verify_cost d(c);
            CHECK(!d.load(path2, &why) && why == "unknown format");
        }
    }
    // a periodic save after save_every ticks, and at destruction
    std::filesystem::remove(path);
    {
        auto cp = c;
        cp.store_path = path;
        cp.save_every = 5;
        common_spec_verify_cost p(cp);
        int64_t id = 0;
        for (int i = 0; i < 4; ++i) {
            p.observe(call(id++, 1, 40000.0));
        }
        CHECK(!std::filesystem::exists(path));
        p.observe(call(id++, 1, 40000.0));
        CHECK(std::filesystem::exists(path) && p.clock_saved == 5);
    }
    std::filesystem::remove(path);
    std::filesystem::remove(path2);
    return 0;
}

//
// rounds
//

// one round of seq 0 on the time line t: draft (draft_us), the target call of m + 1 rows, the rest (rest_us)
struct sim {
    common_spec_verify_cost   & vc;
    common_spec_verify_rounds & rd;
    int64_t t  = 1000000;
    int64_t id = 0;

    const char * round(int32_t owner, int32_t m, double draft_us, double target_us, double rest_us, int32_t seq = 0,
                       bool replay_after = false, double replay_us = 0.0) {
        rd.draft_begin(seq, t, 0.0);
        t += (int64_t) draft_us;
        rd.drafted(seq, t, m, m > 0 ? owner : -1, 1u << std::max(owner, 0), 0.0);
        auto s = call(id++, m + 1, target_us);
        s.seq = seq;
        const char * why = vc.observe(s);
        rd.call(s, why);
        t += (int64_t) target_us;
        if (replay_after) {
            auto r = call(id++, 2, replay_us);
            r.seq = seq;
            r.replay = true;
            rd.call(r, vc.observe(r));
            t += (int64_t) replay_us;
        }
        t += (int64_t) rest_us;
        return why;
    }
};

enum { T_NGRAM = 11, T_DSPARK = 5, T_MTP = 3 }; // the values of common_speculative_type do not matter here

static int test_rounds_split() {
    // the same target curve under three drafters: ngram (no extra work), DSpark (+7 ms of feature injection in every
    // round, its draft forward), MTP (its catch-up in the next draft step, not in the round). The target cost is the
    // same; only the rest of the round differs by the drafters' contexts loaded, not by the drafter of the round
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc_a(c), vc_b(c);
    common_spec_verify_rounds ra(&vc_a, 1), rb(&vc_b, 1);
    sim a{ vc_a, ra }, b{ vc_b, rb };
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> w(0, 8);
    std::uniform_int_distribution<int> who(0, 2);
    for (int i = 0; i < 4000; ++i) {
        const int m = w(rng);
        const int o = who(rng);
        const int32_t owner = o == 0 ? T_NGRAM : o == 1 ? T_DSPARK : T_MTP;
        const double rest = 2000.0 + 100.0 * m; // sampling, rollback, output
        // a: no draft model; b: a draft model loaded, whose context adds 7 ms to every round (whoever drafted), the
        // draft steps differ per drafter and a rollback adds 1 ms when the draft is not accepted whole
        a.round(owner, m, 0.0, curve(m + 1), rest);
        const double draft = owner == T_NGRAM ? 20.0 : owner == T_DSPARK ? 15000.0 : 4000.0 * m;
        const double rollback = m > 0 && (i % 3) == 0 ? 1000.0 : 0.0;
        b.round(owner, m, draft, curve(m + 1), rest + 7000.0 + rollback);
    }
    const auto ta = vc_a.core_table_m(9);
    const auto tb = vc_b.core_table_m(9);
    const auto xa = vc_a.aux_table_m(9);
    const auto xb = vc_b.aux_table_m(9);
    for (int m = 0; m < 9; ++m) {
        CHECK(std::fabs(ta[m] - curve(m + 1)) < 1e-6 * curve(m + 1)); // the core: the target call alone
        CHECK(std::fabs(tb[m] - ta[m]) < 1e-6 * ta[m]);                // the same with a draft model's work around it
        CHECK(std::fabs(xa[m] - (2000.0 + 100.0 * m)) < 1.0);           // the rest of the round
        const double extra = m > 0 ? 7000.0 + 1000.0 / 3.0 : 7000.0;    // + 7 ms, + the mean rollback
        CHECK(std::fabs(xb[m] - (2000.0 + 100.0 * m + extra)) < 400.0); // the rollbacks: a decayed sample mean
    }
    fprintf(stderr, "rounds: core r=5 %.1f/%.1f ms, aux r=5 %.2f/%.2f ms (no draft model / with one), eff r=5 %.1f\n",
            ta[4] / 1000.0, tb[4] / 1000.0, xa[4] / 1000.0, xb[4] / 1000.0, vc_b.eff_table_m(9)[4] / 1000.0);
    return 0;
}

static int test_rounds_edges() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    common_spec_verify_rounds rd(&vc, 2);
    sim s{ vc, rd };
    // widths seen once (untimed), then a normal round of 3 rows
    for (int m = 0; m < 9; ++m) {
        s.round(T_NGRAM, m, 0.0, curve(m + 1), 3000.0);
        s.round(T_NGRAM, m, 0.0, curve(m + 1), 3000.0);
    }
    const int64_t n_aux0 = vc.n_aux;
    // a replay after a checkpoint restore: the replay call is a cost of the round (aux), not of the width's target call
    s.round(T_MTP, 2, 100.0, curve(3), 3000.0, 0, true, 40000.0);
    s.round(T_NGRAM, 0, 0.0, curve(1), 3000.0); // closes it
    CHECK(vc.n_aux == n_aux0 + 2); // the last round of the loop, closed by it, and the round with the replay
    CHECK(vc.aux_cells[2].mean > 3000.0 + 40000.0 / 10.0); // the replay is in the mean of width 3
    // a call of another sequence inside the round: the round is not learnt
    rd.draft_begin(0, s.t, 0.0);
    const int64_t n1 = vc.n_aux;
    rd.drafted(0, s.t, 1, T_NGRAM, 1u << T_NGRAM, 0.0);
    auto o = call(s.id++, 1, 40000.0);
    o.seq = 1;
    rd.call(o, vc.observe(o));
    auto v = call(s.id++, 2, curve(2));
    rd.call(v, vc.observe(v));
    s.t += 100000;
    rd.draft_begin(0, s.t, 0.0);
    CHECK(vc.n_aux == n1);
    // the last round of a request: its target call is learnt, its rest is not (no next draft); a round whose draft was
    // never verified (the slot stopped) neither
    const int64_t nc = vc.n_core;
    rd.drafted(0, s.t, 1, T_NGRAM, 1u << T_NGRAM, 0.0);
    auto l = call(s.id++, 2, curve(2));
    rd.call(l, vc.observe(l));
    CHECK(vc.n_core == nc + 1);
    rd.request_end(0);
    CHECK(rd.owner(0) == -1);
    rd.draft_begin(0, s.t + 1000000, 0.0);
    CHECK(vc.n_aux == n1);
    rd.drafted(0, s.t, 2, T_DSPARK, 1u << T_DSPARK, 0.0);
    CHECK(rd.owner(0) == T_DSPARK && rd.m(0) == 2);
    rd.draft_begin(0, s.t + 50000, 0.0);
    CHECK(vc.n_aux == n1 && rd.owner(0) == -1);
    // the graph builds of the draft context inside the round are taken out of its rest
    rd.drafted(0, s.t, 0, -1, 1u << T_DSPARK, 1000.0);
    auto g = call(s.id++, 1, curve(1));
    rd.call(g, vc.observe(g));
    const double a0 = vc.aux_cells[0].mean;
    const double n0 = vc.aux_cells[0].n_at(vc.clock, vc.decay);
    rd.draft_begin(0, s.t + (int64_t) curve(1) + 3000 + 20000, 1000.0 + 20000.0);
    CHECK(std::fabs(vc.aux_cells[0].mean - (a0 * n0 + 3000.0) / (n0 + 1.0)) < 1e-6);
    return 0;
}

//
// the --spec-smart controller on the shared cost
//

static common_spec_smart_config smart_cfg(int n_max) {
    common_spec_smart_config c;
    c.n_max = n_max;
    c.lookahead_mix = 0.0;
    return c;
}

static int test_smart_shared_same() {
    // the controller on its own cost and on a shared one wider than it, with the same target times: the same table and
    // the same decisions (no rest of the round measured)
    common_spec_smart own(smart_cfg(3), 1);
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    common_spec_smart sh(smart_cfg(3), 1, &vc);
    CHECK(!own.shared() && sh.shared() && own.vc->cfg.r_max == 4);
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> w(0, 3);
    for (int i = 0; i < 500; ++i) {
        const int k = w(rng);
        const double us = curve(k + 1) * (1.0 + 0.05 * std::sin((double) i));
        own.observe_cost(k, us);
        own.n_closed++;
        vc.observe_core(k, us);
        vc.clock++;
    }
    const auto a = own.cost_table();
    const auto b = sh.cost_table();
    CHECK(a.size() == 4 && b.size() == 4);
    for (int k = 0; k < 4; ++k) {
        CHECK(a[k] == b[k]);
        CHECK(own.wait_conf(k) == sh.wait_conf(k) && own.waiting(k) == sh.waiting(k));
    }
    CHECK(own.frontier() == sh.frontier());
    for (auto * s : { &own, &sh }) {
        for (auto & x : s->cal_cells) { x = { 0.8, 1e9 }; }
        for (auto & x : s->qbar_cells) { x = { 0.8, 1e9 }; }
        s->add_draft_step(1000.0);
    }
    for (float p : { 0.1f, 0.5f, 0.9f, 0.99f }) {
        const float pp[3] = { p, p, p };
        for (int n = 0; n <= 3; ++n) {
            CHECK(own.decide(pp, n, 3) == sh.decide(pp, n, 3));
        }
    }
    // the controller's store with a shared cost keeps the calibration only (7); its own keeps the cost (6)
    const std::string p6 = tmp("test-spec-verify-cost-smart6.txt");
    const std::string p7 = tmp("test-spec-verify-cost-smart7.txt");
    CHECK(own.save(p6) && sh.save(p7));
    CHECK(read_file(p6).rfind("ranma-spec-smart 6\n", 0) == 0 && read_file(p6).find("\nratio 2 ") != std::string::npos);
    CHECK(read_file(p7).rfind("ranma-spec-smart 7\n", 0) == 0 && read_file(p7).find("\nratio ") == std::string::npos);
    // a 6 store read with a shared cost: the calibration is taken, the cost lines are not (another time)
    common_spec_verify_cost vc2(c);
    common_spec_smart sh2(smart_cfg(3), 1, &vc2);
    CHECK(sh2.load(p6) && vc2.level() == 0.0 && sh2.cal_cells[5].mean == 0.8);
    // a 7 store read on its own cost: the calibration, the cost stays cold
    common_spec_smart own2(smart_cfg(3), 1);
    CHECK(own2.load(p7) && own2.level() == 0.0 && own2.cal_cells[5].mean == 0.8);
    std::filesystem::remove(p6);
    std::filesystem::remove(p7);
    return 0;
}

static int test_smoothing_contract() {
    // random sparse states, a fall and a rise across the border included: the lookup is the contract above, finite,
    // non-negative, and rises with the width inside each segment (rows 1..8, and 9 on); a longer prefix is the same
    // table (the whole width is computed once)
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int it = 0; it < 300; ++it) {
        common_spec_verify_cost_config c;
        c.r_max = 9 + (it % 3) * 4; // 9, 13, 17 (the scratch of the usual width and the one beyond it)
        common_spec_verify_cost vc(c);
        std::vector<double> prior;
        for (int r = 1; r <= 9; ++r) {
            prior.push_back(SMOOTH_PRIOR[r - 1]);
        }
        const double step = SMOOTH_PRIOR[8] - SMOOTH_PRIOR[7];
        while ((int) prior.size() < c.r_max) {
            prior.push_back(prior.back() + step);
        }
        int64_t id = 0;
        for (int r = 1; r <= c.r_max; ++r) {
            if (u(rng) < 0.5) {
                const double f = 0.5 + 1.5 * u(rng); // some widths far from the prior, up or down
                observe_width(vc, id, r, 1 + (int) (u(rng) * 40), prior[r - 1] * 1000.0 * f);
            }
        }
        for (const auto & pr : { prior, std::vector<double>() }) {
            const int rows = c.r_max + 3;
            const auto b = vc.blend_ms(pr, rows);
            CHECK((int) b.size() == rows);
            if (pr.empty()) {
                const auto ref = ref_smooth(vc, pr, 1000.0);
                for (int r = 1; r <= c.r_max; ++r) {
                    CHECK(std::fabs(b[r - 1] - ref[r - 1]) <= 1e-9 * (1.0 + std::fabs(ref[r - 1])));
                }
            } else if (vc.level() > 0.0) {
                const auto ref = ref_smooth(vc, pr, 1000.0);
                for (int r = 1; r <= c.r_max; ++r) {
                    CHECK(std::fabs(b[r - 1] - ref[r - 1]) <= 1e-9 * (1.0 + std::fabs(ref[r - 1])));
                }
            }
            for (int r = 1; r <= rows; ++r) {
                CHECK(std::isfinite(b[r - 1]) && b[r - 1] >= 0.0);
                if (r >= 2 && r != 9) { // r = 9 starts the second segment: free to fall or rise
                    CHECK(b[r - 1] >= b[r - 2] - 1e-12);
                }
            }
            const auto s = vc.blend_ms(pr, 4); // the prefix of the same table
            for (int r = 1; r <= 4; ++r) {
                CHECK(s[r - 1] == b[r - 1]);
            }
        }
        // the unit: the table in us is the same values (no prior: nothing but the unit differs)
        const auto us = vc.smooth_us(c.r_max + 2);
        const auto ms = vc.blend_ms({}, c.r_max + 2);
        for (size_t i = 0; i < us.size(); ++i) {
            CHECK(std::fabs(us[i] - 1000.0 * ms[i]) <= 1e-9 * (1.0 + std::fabs(us[i])));
        }
    }
    return 0;
}

static int test_weighted_pav() {
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(0.0, 100.0);
    std::uniform_real_distribution<double> uw(0.01, 5.0);
    for (int it = 0; it < 1000; ++it) {
        const int n = 1 + it % 12;
        std::vector<double> v(n), w(n);
        for (int i = 0; i < n; ++i) {
            v[i] = u(rng);
            w[i] = uw(rng);
        }
        double sv = 0.0;
        for (int i = 0; i < n; ++i) { sv += w[i] * v[i]; }
        std::vector<double> p = v;
        common_spec_verify_cost::weighted_pav(p, w);
        double pv = 0.0;
        for (int i = 0; i < n; ++i) {
            pv += w[i] * p[i];
            CHECK(i == 0 || p[i] >= p[i - 1]);
        }
        CHECK(std::fabs(pv - sv) <= 1e-9 * sv);        // the weighted mean is kept
        std::vector<double> q = p;
        common_spec_verify_cost::weighted_pav(q, w);   // idempotent
        for (int i = 0; i < n; ++i) {
            CHECK(std::fabs(q[i] - p[i]) <= 1e-12 * (1.0 + p[i]));
        }
        std::vector<double> m = v; // an already rising input is not changed
        std::sort(m.begin(), m.end());
        std::vector<double> m2 = m;
        common_spec_verify_cost::weighted_pav(m2, w);
        CHECK(m2 == m);
    }
    return 0;
}

static int test_smoothing_segments() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    c.half_life = 1e9; // no decay: the confidence of a width is its number of observations
    int64_t id = 0;
    // a fall across the border stays (r = 9 cheaper than r = 8), the second segment is its own
    {
        common_spec_verify_cost vc(c);
        for (int r = 1; r <= 9; ++r) {
            const double ms = r <= 8 ? SMOOTH_PRIOR[r - 1] : 100.0; // the 9th row on another kernel: cheaper
            observe_width(vc, id, r, 60, ms * 1000.0);
        }
        const auto b = vc.blend_ms(SMOOTH_PRIOR, 9);
        CHECK(b[8] < b[7] - 20.0);
        CHECK(std::fabs(b[8] - 100.0) < 5.0 && std::fabs(b[7] - SMOOTH_PRIOR[7]) < 5.0);
    }
    // a rise across the border stays (a step of 60 ms), and a rising step inside a segment (r = 5) is kept
    {
        common_spec_verify_cost vc(c);
        for (int r = 1; r <= 9; ++r) {
            double ms = SMOOTH_PRIOR[r - 1];
            if (r >= 5) { ms += 30.0; }
            if (r == 9) { ms += 60.0; }
            observe_width(vc, id, r, 60, ms * 1000.0);
        }
        const auto b = vc.blend_ms(SMOOTH_PRIOR, 9);
        CHECK((b[4] - b[3]) > 0.9 * (SMOOTH_PRIOR[4] - SMOOTH_PRIOR[3] + 30.0));
        CHECK((b[8] - b[7]) > 0.9 * (SMOOTH_PRIOR[8] - SMOOTH_PRIOR[7] + 60.0));
    }
    // a fall inside a segment is not kept (r = 6 cheaper than r = 5 by measurement): pooled, non-decreasing
    {
        common_spec_verify_cost vc(c);
        for (int r = 1; r <= 8; ++r) {
            observe_width(vc, id, r, 60, (r == 6 ? SMOOTH_PRIOR[3] : SMOOTH_PRIOR[r - 1]) * 1000.0);
        }
        const auto b = vc.blend_ms(SMOOTH_PRIOR, 9);
        CHECK(b[5] >= b[4] - 1e-12);
        CHECK(b[5] > SMOOTH_PRIOR[3] + 1.0); // lifted by the pool, not the measurement alone
    }
    // an unobserved segment keeps the prior step at the level of r = 1: the ratio of r = 1 (0.8) carries over to r = 9,
    // while the neighbours of the first segment do not reach it
    {
        common_spec_verify_cost vc(c);
        for (int r = 1; r <= 8; ++r) {
            observe_width(vc, id, r, 200, 0.8 * SMOOTH_PRIOR[r - 1] * 1000.0);
        }
        const auto b = vc.blend_ms(SMOOTH_PRIOR, 9);
        const double n1 = vc.conf(vc.shape_cells[0]);
        const double w1 = n1 / (n1 + 2.0);
        const double s  = w1 * vc.eff_table_m(9)[0] / 1000.0 / SMOOTH_PRIOR[0] + (1.0 - w1);
        CHECK(std::fabs(b[8] - SMOOTH_PRIOR[8] * s) <= 1e-9 * b[8]);
        CHECK(s > 0.79 && s < 0.81 && b[8] - b[7] > 0.9 * (SMOOTH_PRIOR[8] - SMOOTH_PRIOR[7]) * 0.8); // the step is kept
    }
    return 0;
}

static int test_smoothing_sparse() {
    common_spec_verify_cost_config c;
    c.r_max = 9;
    c.half_life = 1e9;
    common_spec_verify_cost vc(c);
    int64_t id = 0;
    // r = 1 never observed, widths 2..4 are 30 % above the prior: r = 1 follows them (a cold r = 1), r = 6 also
    for (int r = 2; r <= 4; ++r) {
        observe_width(vc, id, r, 150, 1.3 * SMOOTH_PRIOR[r - 1] * 1000.0);
    }
    CHECK(vc.estimate(1).shape_n == 0.0);
    const auto b = vc.blend_ms(SMOOTH_PRIOR, 9);
    CHECK(std::fabs(b[0] / SMOOTH_PRIOR[0] - 1.3) < 0.04);   // converged to the neighbours' ratio
    CHECK(b[5] / SMOOTH_PRIOR[5] > 1.25);                     // r = 6 follows the nearest observed width
    CHECK(b[8] == SMOOTH_PRIOR[8]);                           // the other segment: the prior (r = 1 has no estimate)
    // the borrowed ratio adds nothing to the confidence
    CHECK(vc.estimate(1).shape_n == 0.0 && vc.estimate(6).shape_n == 0.0 && !vc.known(0) && !vc.known(5));
    // the cold learner: only the prior, any prefix of it, and a prior that is not usable only projects the estimates
    common_spec_verify_cost cold(c);
    const auto p = cold.blend_ms(SMOOTH_PRIOR, 9);
    for (int r = 1; r <= 9; ++r) {
        CHECK(p[r - 1] == SMOOTH_PRIOR[r - 1]);
    }
    std::vector<double> bad = SMOOTH_PRIOR;
    bad[4] = 0.0;
    const auto q = vc.blend_ms(bad, 9);
    const auto raw = vc.blend_ms({}, 9);
    CHECK(q == raw);
    return 0;
}

static int test_smoothing_is_a_lookup() {
    // the lookup changes neither the learned state nor the store
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    std::mt19937 rng(3);
    int64_t id = 0;
    for (int r = 1; r <= 9; ++r) {
        vc.observe(call(id++, r, 1.0));
    }
    feed(vc, id, 400, 1.0, rng, 0.1);
    const std::string p0 = tmp("test-spec-verify-cost-lookup-0.txt");
    const std::string p1 = tmp("test-spec-verify-cost-lookup-1.txt");
    CHECK(vc.save(p0));
    const auto cells = vc.shape_cells;
    const auto aux = vc.aux_cells;
    const int64_t clock = vc.clock;
    const auto eff = vc.eff_table_m(9);
    for (int i = 0; i < 20; ++i) {
        (void) vc.blend_ms(SMOOTH_PRIOR, 12);
        (void) vc.smooth_us(10);
    }
    CHECK(vc.save(p1));
    CHECK(read_file(p0) == read_file(p1));
    CHECK(read_file(p0).rfind("ranma-spec-verify-cost 1\n", 0) == 0);
    CHECK(vc.clock == clock && vc.eff_table_m(9) == eff);
    for (int i = 0; i < 9; ++i) {
        CHECK(vc.shape_cells[i].mean == cells[i].mean && vc.shape_cells[i].n == cells[i].n &&
              vc.shape_cells[i].t == cells[i].t && vc.shape_cells[i].var == cells[i].var);
        CHECK(vc.aux_cells[i].mean == aux[i].mean && vc.aux_cells[i].n == aux[i].n);
    }
    std::filesystem::remove(p0);
    std::filesystem::remove(p1);
    return 0;
}

static int test_smoothing_smart_same() {
    // --spec-smart and the hazard model get the same smoothed values (the unit apart), and the smart frontier, the
    // known and wait inputs do not change with a lookup
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    common_spec_smart sh(smart_cfg(3), 1, &vc);
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> w(0, 3);
    for (int i = 0; i < 300; ++i) {
        const int k = w(rng);
        // the 4th row (k = 3) is cheaper than the 3rd: a fall inside the segment
        const double us = k == 3 ? 0.9 * curve(3) : curve(k + 1) * (1.0 + 0.05 * std::sin((double) i));
        vc.observe_core(k, us);
        vc.clock++;
    }
    const auto t = sh.cost_table();
    const auto h = vc.blend_ms({}, 4);
    CHECK(t.size() == 4 && h.size() == 4);
    for (int k = 0; k < 4; ++k) {
        CHECK(std::fabs(t[k] - 1000.0 * h[k]) <= 1e-9 * (1.0 + t[k]));
        CHECK(k == 0 || t[k] >= t[k - 1]);
    }
    CHECK(t[3] == t[2] && t[3] > vc.eff_table_m(4)[3] + 100.0); // raw: 0.9 x the 3rd row, smoothed: pooled with it
    // the inputs of the frontier and of the wait are the raw observations
    std::vector<int> known, waiting;
    std::vector<double> wc;
    for (int m = 0; m < 9; ++m) {
        known.push_back(vc.known(m));
        waiting.push_back(vc.waiting(m));
        wc.push_back(vc.wait_conf(m));
    }
    const int32_t fr = sh.frontier();
    for (int i = 0; i < 10; ++i) {
        (void) sh.cost_table();
        (void) vc.blend_ms(SMOOTH_PRIOR, 9);
    }
    CHECK(sh.frontier() == fr);
    for (int m = 0; m < 9; ++m) {
        CHECK(vc.known(m) == (bool) known[m] && vc.waiting(m) == (bool) waiting[m] && vc.wait_conf(m) == wc[m]);
    }
    // an unobserved width is not known because the lookup interpolates it
    CHECK(!vc.known(6));
    return 0;
}

static int test_owner_alternation() {
    // rounds of ngram, DSpark (the --spec-smart drafter), none, MTP on one sequence. The central protocol: every round
    // starts by closing the controller's round; the controller drafts only when ngram did not; accepted() reaches it
    // only for its own rounds. Its calibration learns from its own rounds only, its rounds never span another
    // drafter's, and the shared cost learns every round
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    common_spec_verify_rounds rd(&vc, 1);
    common_spec_smart_config sc = smart_cfg(4);
    sc.block = true;
    sc.n_groups = 4;
    sc.log_path = tmp("test-spec-verify-cost-smart.log");
    std::filesystem::remove(sc.log_path);
    {
        common_spec_smart sm(sc, 1, &vc);
        int64_t t = 1000000;
        int64_t id = 0;
        int64_t smart_rounds = 0;
        const int32_t seq[8] = { T_NGRAM, T_DSPARK, -1, T_MTP, T_NGRAM, T_NGRAM, T_DSPARK, T_DSPARK };
        for (int i = 0; i < 64; ++i) {
            const int32_t who = seq[i % 8];
            // the round starts: the shared rounds and the controller close the previous one
            rd.draft_begin(0, t, 0.0);
            sm.close_round(0, t);
            int32_t m = 0;
            if (who == T_DSPARK) {
                if (sm.begin_block(0, 4)) {
                    const float p[4] = { 0.9f, 0.9f, 0.9f, 0.9f };
                    m = sm.block_drafted(0, p, 4);
                    sm.add_draft_step(15000.0);
                    t += 15000;
                }
                sm.drafted(0, m, t);
                smart_rounds++;
            } else {
                // another drafter (or none): the controller has no round open, none spans this one
                CHECK(!sm.seqs[0].active);
                m = who >= 0 ? 3 : 0;
            }
            rd.drafted(0, t, m, m > 0 ? who : -1, who >= 0 ? 1u << who : 0u, 0.0);
            auto s = call(id++, m + 1, curve(m + 1));
            rd.call(s, vc.observe(s));
            t += (int64_t) curve(m + 1) + 3000;
            // accepted: the gate of common_speculative_smart_accepted
            const auto cal0 = sm.cal_cells;
            if (rd.owner(0) == T_DSPARK) {
                sm.accepted(0, 1);
            }
            if (rd.owner(0) != T_DSPARK) {
                for (size_t j = 0; j < cal0.size(); ++j) {
                    CHECK(sm.cal_cells[j].mean == cal0[j].mean && sm.cal_cells[j].n == cal0[j].n);
                }
            }
        }
        rd.draft_begin(0, t, 0.0);
        sm.close_round(0, t);
        CHECK(sm.n_closed == smart_rounds); // its clock: its own closed rounds
        CHECK(vc.clock == 64);              // the shared clock: every target call
        CHECK(vc.n_core > 0 && vc.n_aux > 0);
        sm.log_flush();
    }
    // the controller's rounds end where the next round starts: a verify_us of its round is its own target call and
    // rest (at most the widest call + 3 ms), never several rounds
    std::ifstream f(sc.log_path);
    int n_closed = 0;
    for (std::string l; std::getline(f, l);) {
        if (l.find("\"ev\":\"round\"") == std::string::npos || l.find("\"end\":\"closed\"") == std::string::npos) {
            continue;
        }
        const size_t at = l.find("\"verify_us\":");
        CHECK(at != std::string::npos);
        const double v = atof(l.c_str() + at + 12);
        CHECK(v <= curve(5) + 3000.0 + 1.0);
        CHECK(l.find("\"obs_skip\":\"shared\"") != std::string::npos);
        n_closed++;
    }
    CHECK(n_closed > 0);
    f.close();
    std::filesystem::remove(sc.log_path);
    return 0;
}

//
// the hazard model of ngram-varlen with a cost table given
//

static int test_hazard_table() {
    common_ngram_varlen_hazard_params hp;
    common_ngram_varlen_hazard hz(hp);
    // a candidate of 8 tokens of one long match, confident
    common_ngram_varlen_cand cand;
    cand.n = 12;
    cand.n_samples = 1;
    for (int j = 0; j < 8; ++j) {
        cand.tokens.push_back(100 + j);
        cand.raw.push_back(0.99);
        cand.support.push_back(1);
    }
    const std::vector<common_ngram_varlen_cand> alt = { cand };
    size_t m0 = 0, m1 = 0, m2 = 0;
    hz.select(alt, 0, m0);
    // the table of the parameters given explicitly, and a blend before any observation (the prior): the same choice
    common_spec_verify_cost_config c;
    c.r_max = 9;
    common_spec_verify_cost vc(c);
    const auto prior = vc.blend_ms(hp.cost, 9);
    hz.select(alt, 0, m1, &hp.cost);
    hz.select(alt, 0, m2, &prior, hp.width);
    CHECK(m0 == m1 && m1 == m2);
    // a flat cost: the longest; a steep one: none
    const std::vector<double> flat  = { 40, 40.1, 40.2, 40.3, 40.4, 40.5, 40.6, 40.7, 40.8 };
    const std::vector<double> steep = { 40, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200 };
    size_t mf = 0, ms = 9;
    hz.select(alt, 0, mf, &flat);
    hz.select(alt, 0, ms, &steep);
    CHECK(mf >= 6 && ms == 0); // flat: a long draft (the 8th token adds less than its row costs), steep: none
    // a narrower piece (a request near its end): its own pieces
    size_t mw = 0;
    hz.select(alt, 0, mw, &flat, 2);
    CHECK(mw >= 1);
    fprintf(stderr, "hazard: m table %zu, flat %zu, steep %zu, width 2 %zu\n", m0, mf, ms, mw);
    return 0;
}

int main() {
    struct { const char * name; int (*fn)(); } tests[] = {
        { "curve_and_level",  test_curve_and_level },
        { "decay_and_variance", test_decay_and_variance },
        { "bad_wide_graph_dup", test_bad_wide_graph_dup },
        { "lend",             test_lend },
        { "sparse_blend",     test_sparse_blend },
        { "smoothing_contract", test_smoothing_contract },
        { "weighted_pav",     test_weighted_pav },
        { "smoothing_segments", test_smoothing_segments },
        { "smoothing_sparse", test_smoothing_sparse },
        { "smoothing_lookup", test_smoothing_is_a_lookup },
        { "smoothing_smart_same", test_smoothing_smart_same },
        { "store",            test_store },
        { "rounds_split",     test_rounds_split },
        { "rounds_edges",     test_rounds_edges },
        { "smart_shared_same", test_smart_shared_same },
        { "owner_alternation", test_owner_alternation },
        { "hazard_table",     test_hazard_table },
    };
    for (const auto & t : tests) {
        if (t.fn() != 0) {
            fprintf(stderr, "test-spec-verify-cost: %s FAILED\n", t.name);
            return 1;
        }
        fprintf(stderr, "test-spec-verify-cost: %s ok\n", t.name);
    }
    fprintf(stderr, "test-spec-verify-cost: OK\n");
    return 0;
}
