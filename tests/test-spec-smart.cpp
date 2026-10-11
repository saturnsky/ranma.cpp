// --spec-smart controller (common/speculative-smart.h) on the CPU: decisions from synthetic costs and acceptance,
// store round trip, the verification time as level x shape (level half-life), request-boundary shrink, round
// protocol, decay over rounds, the extension of a round by one width to verify a width of low shape confidence (no
// cold sweep, no forced round), the first use of a width not timed, the mean acceptance per position and the
// lookahead mix, the round log (--spec-smart-log), the cap of the shape confidence that decides the wait, the cut of
// the old mean when a waiting width is observed, the defaults and the option rules (common_speculative_smart_resolve).
// The fixed constants of the controller (level half-life, extension threshold and interval, wait confidence cap,
// lookahead mix) are changed here through common_spec_smart_config.
#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "speculative.h"
#include "speculative-smart.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// q = p exactly (a very confident identity calibration) and a given lookahead acceptance
static void set_identity(common_spec_smart & s, double qb) {
    for (int g = 0; g < common_spec_smart::N_GROUPS; ++g) {
        for (int b = 0; b < s.cfg.n_bins; ++b) {
            s.cal_cells[(size_t) g * s.cfg.n_bins + b] = { (b + 0.5) / s.cfg.n_bins, 1e9 };
        }
    }
    for (auto & x : s.qbar_cells) {
        x = { qb, 1e9 };
    }
}

static void set_costs(common_spec_smart & s, const std::vector<double> & ms, double draft_ms) {
    for (size_t k = 0; k < ms.size(); ++k) {
        s.observe_cost((int) k, ms[k] * 1000.0);
    }
    s.add_draft_step(draft_ms * 1000.0);
}

// the tests from before the lookahead mix use the lookahead of the position mean alone (lookahead_mix 0), which must
// decide as before. They were written for the half-life 1000 of the previous default (now 250) and the extension
// interval 64 (now 16) and keep them, so that their expectations stay those of the previous values
static const double OLD_HALF_LIFE = 1000.0;
static const int32_t OLD_REPROBE_EVERY = 64;

static common_spec_smart_config base_cfg(int n_max) {
    common_spec_smart_config c;
    c.n_max = n_max;
    c.half_life = OLD_HALF_LIFE;
    c.lookahead_mix = 0.0;
    c.reprobe_every = OLD_REPROBE_EVERY;
    return c;
}

// the cost table from the shape means (level x shape mean), as the controller computes it
static std::vector<double> ref_cost_table_mean(const common_spec_smart & s) {
    const int N = s.cfg.n_max + 1;
    std::vector<double> val(N, 0.0);
    std::vector<int> known(N, 0);
    for (int k = 0; k < N; ++k) {
        if (s.shape_cells[k].n > 0.0) {
            val[k] = s.shape_cells[k].mean; known[k] = 1;
        }
    }
    std::vector<double> out = val;
    for (int k = 0; k < N; ++k) {
        if (known[k]) {
            continue;
        }
        int j[2];
        int nj = 0;
        for (int d = 1; d < N && nj < 2; ++d) {
            if (k - d >= 0 && known[k - d]) { j[nj++] = k - d; }
            if (nj < 2 && k + d < N && known[k + d]) { j[nj++] = k + d; }
        }
        out[k] = nj == 2 ? val[j[0]] + (val[j[1]] - val[j[0]]) * (double) (k - j[0]) / (double) (j[1] - j[0])
               : nj == 1 ? val[j[0]] : 0.0;
    }
    int last_run = -1;
    for (int k = 0; k < N; ++k) {
        if (s.cost_n_run[k] > 0) {
            last_run = k;
        } else if (last_run >= 0) {
            out[k] = std::max(out[k], out[last_run]);
        }
        out[k] = std::max(out[k], 0.0);
    }
    for (auto & x : out) {
        x *= std::max(s.level_cell.mean, 0.0);
    }
    return out;
}

// forget all verification and draft step times
static void reset_costs(common_spec_smart & s) {
    for (auto & c : s.shape_cells) { c = {}; }
    std::fill(s.shape_over.begin(), s.shape_over.end(), 0.0);
    s.level_cell = {};
    std::fill(s.cost_n_run.begin(), s.cost_n_run.end(), 0);
    s.draft_cell = {};
}

static int test_decisions() {
    // cheap wide verification, confident drafts: draft to the cap and keep everything
    {
        common_spec_smart s(base_cfg(4), 1);
        set_identity(s, 0.9);
        set_costs(s, {50, 52, 54, 56, 58}, 1.0);
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.975f) == -1);
        CHECK(s.after_step(0, 0.975f) == -1);
        CHECK(s.after_step(0, 0.975f) == -1);
        CHECK(s.after_step(0, 0.975f) == 4); // at the cap: stop, keep all
        s.drafted(0, 4, 0);
        CHECK(s.seqs[0].k == 4);
    }
    // an unlikely first token is dropped (keep 0) even though drafting was worth starting
    {
        common_spec_smart s(base_cfg(4), 1);
        set_identity(s, 0.9);
        set_costs(s, {50, 60, 62, 64, 66}, 1.0);
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.025f) == 0);
    }
    // expensive widths: no draft at all (k = 0 is a candidate before the first step)
    {
        common_spec_smart s(base_cfg(4), 1);
        set_identity(s, 0.6);
        set_costs(s, {50, 100, 150, 200, 250}, 1.0);
        CHECK(!s.begin_round(0, 4));
        const float p0 = 0.6f;
        CHECK(s.decide(&p0, 1, 4) == 0);
    }
    // the continue value looks past one step: width 1 alone is not worth it, width 2 is (concave cost)
    {
        common_spec_smart s(base_cfg(2), 1);
        set_identity(s, 0.9);
        set_costs(s, {50, 110, 111}, 0.5);
        // tokens per ms: stop at 0: 1/50 = 0.0200; width 1: 1.9/110.5 = 0.0172; width 2: 2.71/112 = 0.0242
        CHECK(s.begin_round(0, 2));
        CHECK(s.after_step(0, 0.925f) == -1);
    }
    // hand check of the stop value with a tail drop: p = (0.975, 0.025), costs flat 50/60/61
    {
        common_spec_smart s(base_cfg(2), 1);
        set_identity(s, 0.5);
        set_costs(s, {50, 60, 61}, 1.0);
        const float p[2] = { 0.975f, 0.025f };
        // D = 2 ms. k0: 1/52, k1: 1.975/62 = 0.03185, k2: (1.975 + 0.024375)/63 = 0.03174 -> keep 1
        CHECK(s.decide(p, 2, 2) == 1);
    }
    // a per-request cap below n_max bounds the draft
    {
        common_spec_smart s(base_cfg(6), 1);
        set_identity(s, 0.95);
        set_costs(s, {50, 51, 52, 53, 54, 55, 56}, 0.1);
        CHECK(s.begin_round(0, 2));
        CHECK(s.after_step(0, 0.975f) == -1);
        CHECK(s.after_step(0, 0.975f) == 2);
    }
    return 0;
}

static int test_cost_prediction() {
    // unobserved widths: linear through the two nearest measured widths
    {
        common_spec_smart s(base_cfg(4), 1);
        s.observe_cost(0, 100.0);
        s.observe_cost(1, 110.0);
        const auto C = s.cost_table();
        CHECK(std::fabs(C[3] - 130.0) < 1e-9);
        CHECK(std::fabs(C[4] - 140.0) < 1e-9);
    }
    // never cheaper than the nearest narrower measured width
    {
        common_spec_smart s(base_cfg(4), 1);
        s.observe_cost(0, 100.0);
        s.observe_cost(3, 50.0);
        const auto C = s.cost_table();
        CHECK(C[1] == 100.0 && C[2] == 100.0); // interpolation 83.3 / 66.7 raised to width 0
        CHECK(C[3] == 50.0);                   // measured widths are taken as measured
        CHECK(C[4] == 50.0);                   // extrapolation 33.3 raised to width 3
    }
    // the first observation sets the level; a width without a shape gets time / level
    {
        common_spec_smart s(base_cfg(4), 1);
        CHECK(s.level() == 0.0 && s.cost(2) == 0.0); // nothing measured: no value
        s.observe_cost(1, 40000.0);
        CHECK(s.level() == 40000.0 && s.shape_cells[1].mean == 1.0 && s.level_cell.n == 1.0);
        s.observe_cost(3, 60000.0); // no shape yet: the level stays
        CHECK(s.level() == 40000.0 && s.shape_cells[3].mean == 1.5);
        CHECK(s.cost(3) == 60000.0 && s.cost(0) == 30000.0 && s.cost(2) == 50000.0 && s.cost(4) == 70000.0); // the line
        // no shape with a confidence of reprobe_n (4) or more: every width with a shape moves the level
        s.observe_cost(3, 90000.0);
        CHECK(s.level() == (40000.0 + 60000.0) / 2.0);          // 90000 / 1.5 against 40000
        CHECK(std::fabs(s.shape_cells[3].mean - 1.875) < 1e-12); // (1.5 + 90000 / 40000) / 2, the level before
    }
    // a stored level and shape: the level of this process scales every width, faster or slower (no asdown)
    const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-level.txt").string();
    {
        common_spec_smart_config c = base_cfg(4);
        common_spec_smart w(c, 1);
        w.level_cell = { 100.0, 50.0 };
        for (int k = 0; k <= 4; ++k) {
            w.shape_cells[k] = { 1.0 + 0.1 * k, 300.0 };
        }
        CHECK(w.save(path));
    }
    {
        common_spec_smart_config c = base_cfg(4);
        c.store_path = path;
        common_spec_smart s(c, 1);
        CHECK(s.store_loaded);
        CHECK(s.level_cell.n == 2.0 && s.shape_cells[3].n == 32.0 && s.wait_conf(3) == 16.0); // the wait: the running cap
        CHECK(std::fabs(s.cost(3) - 130.0) < 1e-9);
        s.observe_cost(0, 50.0);                            // faster now
        CHECK(std::fabs(s.level() - (2.0 * 100.0 + 50.0) / 3.0) < 1e-9);
        CHECK(std::fabs(s.cost(3) - s.level() * 1.3) < 1e-9 && s.cost(3) < 130.0); // down
        CHECK(s.cost(3) >= s.cost(0));
    }
    {
        common_spec_smart_config c = base_cfg(4);
        c.store_path = path;
        common_spec_smart s(c, 1);
        s.observe_cost(0, 400.0);                           // slower now
        CHECK(std::fabs(s.level() - 200.0) < 1e-9);
        CHECK(std::fabs(s.cost(3) - 260.0) < 1e-9);         // up as well (asdown applied only a ratio below 1)
        CHECK(s.cost(1) >= s.cost(0));                      // floor at the measured width 0
    }
    std::filesystem::remove(path);
    return 0;
}

static int test_store() {
    const auto dir  = std::filesystem::temp_directory_path();
    const std::string path  = (dir / "test-spec-smart-store.txt").string();
    const std::string path2 = (dir / "test-spec-smart-store2.txt").string();

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    common_spec_smart_config c = base_cfg(5);
    c.store_key = "target=a.gguf@1:2 build=b1 l1=16384";
    {
        common_spec_smart w(c, 1);
        for (auto & x : w.shape_cells) { x = { 0.5 + 2.0 * u(rng), 1.0 + 999.0 * u(rng), 0, 0.1 * u(rng) }; }
        w.level_cell = { 1e4 + 1e5 * u(rng), 1.0 + 99.0 * u(rng) };
        for (auto & x : w.cal_cells)  { x = { u(rng), 1000.0 * u(rng) }; }
        for (auto & x : w.qbar_cells) { x = { u(rng), 5000.0 * u(rng) }; }
        CHECK(w.save(path));

        // exact restore: means and confidence bit-exact when the load cap is out of the way
        common_spec_smart_config c2 = c;
        c2.cap_load = 1e300;
        c2.level_cap_load = 1e300;
        c2.store_path = path;
        common_spec_smart r(c2, 1);
        CHECK(r.store_loaded);
        CHECK(r.level_cell.mean == w.level_cell.mean && r.level_cell.n == w.level_cell.n);
        for (size_t i = 0; i < w.shape_cells.size(); ++i) {
            CHECK(r.shape_cells[i].mean == w.shape_cells[i].mean && r.shape_cells[i].n == w.shape_cells[i].n);
            CHECK(r.shape_cells[i].var == w.shape_cells[i].var);
        }
        for (size_t i = 0; i < w.cal_cells.size(); ++i) {
            CHECK(r.cal_cells[i].mean == w.cal_cells[i].mean && r.cal_cells[i].n == w.cal_cells[i].n);
        }
        for (size_t i = 0; i < w.qbar_cells.size(); ++i) {
            CHECK(r.qbar_cells[i].mean == w.qbar_cells[i].mean && r.qbar_cells[i].n == w.qbar_cells[i].n);
        }
        r.cfg.store_path.clear(); // no save at destruction
        CHECK(r.save(path2));
        CHECK(read_file(path) == read_file(path2)); // save -> load -> save is byte-identical

        // the normal load: values kept exactly, confidence capped at 32 (never lost to 0), the level at 2, the wait
        // confidence of the shape at the running cap 16
        common_spec_smart_config c3 = c;
        c3.store_path = path;
        common_spec_smart s(c3, 1);
        CHECK(s.store_loaded);
        CHECK(s.level_cell.mean == w.level_cell.mean && s.level_cell.n == std::min(w.level_cell.n, 2.0));
        for (size_t i = 0; i < w.shape_cells.size(); ++i) {
            CHECK(s.shape_cells[i].mean == w.shape_cells[i].mean && s.shape_cells[i].var == w.shape_cells[i].var);
            CHECK(s.shape_cells[i].n == std::min(w.shape_cells[i].n, 32.0));
            CHECK(s.wait_conf((int) i) == std::min(w.shape_cells[i].n, 16.0));
            CHECK(s.shape_cells[i].n > 0.0 && s.cost_n_run[i] == 0);
        }
        for (size_t i = 0; i < w.cal_cells.size(); ++i) {
            CHECK(s.cal_cells[i].mean == w.cal_cells[i].mean);
            CHECK(s.cal_cells[i].n == std::min(w.cal_cells[i].n, 32.0));
        }
        s.cfg.store_path.clear();

        // another key (another model, build or cache setting): not used, cold start
        common_spec_smart_config c4 = c;
        c4.store_path = path;
        c4.store_key  = "target=b.gguf@1:2 build=b1 l1=16384";
        common_spec_smart k(c4, 1);
        CHECK(!k.store_loaded && k.n_extend == 0);
        for (auto & x : k.shape_cells) { CHECK(x.n == 0.0); }
        CHECK(k.level() == 0.0);
        k.cfg.store_path.clear();

        // a larger n_max reads the stored widths and leaves the others unmeasured
        common_spec_smart_config c5 = c;
        c5.store_path = path;
        c5.n_max = 7;
        common_spec_smart m(c5, 1);
        CHECK(m.store_loaded && m.shape_cells[5].n > 0.0 && m.shape_cells[6].n == 0.0);
        CHECK(m.level_cell.mean == w.level_cell.mean);
        CHECK(m.qbar_cells.size() == 8 && m.qbar_cells[5].mean == w.qbar_cells[5].mean && m.qbar_cells[6].n == 0.0);
        m.cfg.store_path.clear();

        // a smaller n_max reads its positions only
        common_spec_smart_config c7 = c;
        c7.store_path = path;
        c7.n_max = 2;
        common_spec_smart sm(c7, 1);
        CHECK(sm.store_loaded && sm.qbar_cells.size() == 3 && sm.qbar_cells[2].mean == w.qbar_cells[2].mean);
        sm.cfg.store_path.clear();

        // a truncated file is rejected as a whole
        {
            std::string t = read_file(path);
            t.resize(t.size() / 2);
            std::ofstream f(path2, std::ios::binary | std::ios::trunc);
            f << t;
        }
        common_spec_smart_config c6 = c;
        c6.store_path = path2;
        common_spec_smart tr(c6, 1);
        CHECK(!tr.store_loaded);
        tr.cfg.store_path.clear();
    }
    std::filesystem::remove(path);
    std::filesystem::remove(path2);
    return 0;
}

static int test_request_shrink() {
    common_spec_smart s(base_cfg(3), 1);
    s.cal_cells[5]  = { 0.7, 500.0 };
    s.cal_cells[25] = { 0.3, 4.0 };
    s.qbar_cells[1] = { 0.8, 900.0 };
    s.shape_cells[2] = { 1.234, 700.0 };
    s.level_cell     = { 1000.0, 90.0 };
    s.request_begin(0);
    CHECK(s.cal_cells[5].mean == 0.7 && s.cal_cells[5].n == 16.0);
    CHECK(s.cal_cells[25].mean == 0.3 && s.cal_cells[25].n == 4.0);
    CHECK(s.qbar_cells[1].mean == 0.8 && s.qbar_cells[1].n == 16.0);
    CHECK(s.shape_cells[2].mean == 1.234 && s.shape_cells[2].n == 700.0); // costs are not shrunk
    CHECK(s.level_cell.mean == 1000.0 && s.level_cell.n == 90.0);
    return 0;
}

static int test_rounds() {
    common_spec_smart_config c;
    c.n_max = 2;
    c.half_life = OLD_HALF_LIFE;
    common_spec_smart s(c, 2);
    CHECK(!s.store_loaded && s.n_extend == 0);

    int64_t t = 0;
    s.request_begin(0);
    // widths 0, 1, 2, 0, 1, 2 (set after the decision, as the extra step of an extension is): the first closed round
    // of every width in this process is not timed (it builds the graphs), the second is
    for (int r = 0; r < 6; ++r) {
        s.close_round(0, t);
        if (r >= 1) {
            CHECK(s.seqs[0].timed == (r >= 4)); // the round closed just now
        }
        const int w = r % 3;
        s.begin_round(0, 8);
        s.seqs[0].forced = w;
        for (int i = 0; i < w; ++i) {
            CHECK(s.after_step(0, 0.5f) == (i + 1 == w ? w : -1));
            s.add_draft_step(1000.0);
        }
        s.drafted(0, w, t);
        if (w > 0) {
            s.accepted(0, 1);
        }
        t += 10000 * (w + 1);
    }
    s.close_round(0, t);
    CHECK(s.cost_n_run[0] == 1 && s.cost_n_run[1] == 1 && s.cost_n_run[2] == 1); // first use not timed
    CHECK(s.n_extend == 0 && s.used_run[0] && s.used_run[1] && s.used_run[2]);
    CHECK(s.level() == 10000.0 && s.shape_cells[2].mean == 3.0); // width 0 first: level 10 ms, width 2 ratio 3
    CHECK(s.cost(0) == 10000.0 && s.cost(2) == 30000.0);
    // acceptance: width 1 twice (pos 0 accepted), width 2 twice (pos 0 accepted, pos 1 rejected)
    const auto & c0 = s.cal_cells[(size_t) 0 * s.cfg.n_bins + s.bin(0.5f)];
    const auto & c1 = s.cal_cells[(size_t) 1 * s.cfg.n_bins + s.bin(0.5f)];
    CHECK(c0.n > 3.99 && c0.n < 4.0 + 1e-9 && c0.mean == 1.0);
    CHECK(c1.n > 1.99 && c1.n < 2.0 + 1e-9 && c1.mean == 0.0);
    const std::string sum = s.request_summary(0);
    CHECK(sum.find("rounds 6") != std::string::npos && sum.find("mean k 1.00") != std::string::npos);
    CHECK(sum.find("acceptance 0.667 (4/6)") != std::string::npos);
    fprintf(stderr, "summary: %s\n", sum.c_str());

    // a draft that is never verified is not a round; the last round of a request is dropped at the next request
    const int64_t n0 = s.cost_n_run[1];
    s.begin_round(1, 1);
    s.seqs[1].forced = 1;
    CHECK(s.after_step(1, 0.9f) == 1);
    s.drafted(1, 1, t);
    s.close_round(1, t + 5000); // no accepted(): discarded
    CHECK(s.cost_n_run[1] == n0);
    s.begin_round(1, 1);
    s.seqs[1].forced = 1;
    s.after_step(1, 0.9f);
    s.drafted(1, 1, t);
    s.accepted(1, 1);
    s.request_begin(1);         // new request: the pending round ends without a time
    s.close_round(1, t + 1000000000);
    CHECK(s.cost_n_run[1] == n0);

    // periodic save
    const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-periodic.txt").string();
    std::filesystem::remove(path);
    common_spec_smart_config cp = base_cfg(1);
    cp.store_path = path;
    cp.save_every = 3;
    {
        common_spec_smart p(cp, 1);
        for (int r = 0; r < 3; ++r) {
            p.close_round(0, 100 * r); // closes the rounds 1 and 2
            p.begin_round(0, 1);
            p.drafted(0, 0, 100 * r);
            CHECK(!std::filesystem::exists(path));
        }
        p.close_round(0, 300);
        CHECK(std::filesystem::exists(path)); // after 3 closed rounds
        CHECK(p.n_closed == 3 && p.n_closed_saved == 3);
    }
    std::filesystem::remove(path);
    return 0;
}

// one round of the protocol with a fixed draft probability; returns the draft length. cost_ms(k) gives the time
// of the verification that follows.
template <typename F>
static int run_round(common_spec_smart & s, int64_t & t, float p, int n_max_seq, F cost_ms) {
    s.close_round(0, t);
    int k = 0;
    if (s.begin_round(0, n_max_seq)) {
        for (;;) {
            s.add_draft_step(1000.0);
            t += 1000;
            const int r = s.after_step(0, p);
            if (r >= 0) {
                k = r;
                break;
            }
        }
    }
    s.drafted(0, k, t);
    if (k > 0) {
        s.accepted(0, k);
    }
    t += (int64_t) (cost_ms(k) * 1000.0);
    return k;
}

static int test_time_decay() {
    // cells that get no observation lose confidence with the rounds of the controller; their means stay
    common_spec_smart_config c = base_cfg(2);
    c.half_life = 1000.0;
    c.reprobe_every = 0;
    common_spec_smart s(c, 1);
    s.shape_cells[1] = { 2.0, 8.0 };
    s.shape_cells[2] = { 2.5, 8.0 };
    s.cal_cells[(size_t) 1 * s.cfg.n_bins + 10] = { 0.9, 8.0 };
    s.qbar_cells[2] = { 0.8, 8.0 };
    s.draft_cell    = { 1500.0, 8.0 };
    const double q0 = s.q(1, 0.52f);

    int64_t t = 0;
    for (int r = 0; r < 1000; ++r) {
        s.close_round(0, t);
        CHECK(!s.begin_round(0, 0)); // k = 0: only the width 0 cell is observed
        s.drafted(0, 0, t);
        t += 20000;
    }
    s.close_round(0, t);
    CHECK(s.n_closed == 1000);
    CHECK(std::fabs(s.conf(s.shape_cells[1]) - 4.0) < 1e-9);
    CHECK(std::fabs(s.conf(s.shape_cells[2]) - 4.0) < 1e-9);
    CHECK(std::fabs(s.conf(s.cal_cells[(size_t) 1 * s.cfg.n_bins + 10]) - 4.0) < 1e-9);
    CHECK(std::fabs(s.conf(s.qbar_cells[2]) - 4.0) < 1e-9);
    CHECK(std::fabs(s.conf(s.draft_cell) - 4.0) < 1e-9);
    CHECK(s.shape_cells[1].mean == 2.0 && s.cal_cells[(size_t) 1 * s.cfg.n_bins + 10].mean == 0.9);
    CHECK(s.shape_cells[2].mean == 2.5 && s.qbar_cells[2].mean == 0.8 && s.draft_cell.mean == 1500.0);
    // the level: set by the first round of width 0, the same value since; its confidence saturates near
    // 1 / (1 - decay) with the half-life of 64 rounds
    CHECK(s.level() == 20000.0 && s.cost(2) == 50000.0);
    CHECK(s.level_conf() > 90.0 && s.level_conf() < 1.0 / (1.0 - s.level_decay));
    // the acceptance goes to the prior (bin centre 0.525) as the confidence falls
    const double q1 = s.q(1, 0.52f);
    CHECK(q1 < q0 && q1 > 0.525);
    CHECK(std::fabs(q1 - (4.0 * 0.9 + 2.0 * 0.525) / 6.0) < 1e-9);
    // the observed width 0 keeps its confidence up: n = sum of decay^i over its 999 timed rounds
    CHECK(s.conf(s.shape_cells[0]) > 700.0);

    // one more half-life: a quarter
    s.n_closed += 1000;
    CHECK(std::fabs(s.conf(s.shape_cells[2]) - 2.0) < 1e-9);

    // an observation starts from the decayed confidence (the level does not move: width 2 is not confident)
    s.observe_cost(2, 60000.0);
    CHECK(std::fabs(s.shape_cells[2].n - 3.0) < 1e-9 && s.shape_cells[2].t == s.n_closed);
    CHECK(std::fabs(s.shape_cells[2].mean - (2.5 + 0.5 / 3.0)) < 1e-12 && s.level() == 20000.0);
    return 0;
}

// DS case without the cold sweep: width 2 was timed once too slow (60 ms, its true time is 34 ms) while width 1
// (30 ms) is the choice. The decision stops at width 1, so width 2 (confidence below 4) is verified again by an
// extension of such a round (one more draft step) and is then taken. Without extensions (reprobe_every 0) it keeps the
// slow value. noise: relative sd of the true times (0: exact). The cost table and the draft step must be those of the
// means in every round.
static const double DS_TRUE_MS[3] = { 28.0, 30.0, 34.0 };

static int run_ds_case(int reprobe_every, int n_rounds, int & k2_last, double & k2_cost, int64_t & n_ext,
                       double noise = 0.0, int * first_k2_out = nullptr) {
    common_spec_smart_config c;
    c.n_max = 2;
    c.half_life = OLD_HALF_LIFE;
    c.reprobe_every = reprobe_every;
    c.lookahead_mix = 0.0; // the rounds counted below are those of the position mean alone
    common_spec_smart s(c, 1);
    set_identity(s, 0.9);
    // one timed observation per width after the first use of each: 28 / 30 / 60 ms
    s.observe_cost(0, 28000.0);
    s.observe_cost(1, 30000.0);
    s.observe_cost(2, 60000.0);
    s.used_run.assign(3, 1);
    std::mt19937 rng(11);
    std::normal_distribution<double> nd(0.0, 1.0);

    int64_t t = 0;
    int64_t last_ext = -1000000;
    int     first_k2 = -1; // first round that the decision (not an extension) takes width 2
    k2_last = 0;
    for (int round = 0; round < n_rounds; ++round) {
        const int64_t e0 = s.n_extend;
        const int k = run_round(s, t, 0.925f, 2, [&](int w) {
            const double x = DS_TRUE_MS[w];
            return noise > 0.0 ? std::max(1.0, x * (1.0 + noise * nd(rng))) : x;
        });
        if (s.cost_table() != ref_cost_table_mean(s) || s.draft_step() != s.draft_cell.mean) {
            return 2; // not the decision of the means
        }
        if (s.n_extend > e0) {
            if (s.n_closed - last_ext < reprobe_every || k != 2 || s.seqs[0].k_nat != 1 || !s.seqs[0].ext_step) {
                return 1; // spacing, or not the width after the decision
            }
            last_ext = s.n_closed;
        } else if (k == 2 && first_k2 < 0) {
            first_k2 = round;
        }
        if (round >= n_rounds - 500 && k == 2) {
            k2_last++;
        }
    }
    s.close_round(0, t);
    k2_cost = s.cost(2) / 1000.0;
    n_ext   = s.n_extend;
    if (first_k2_out) {
        *first_k2_out = first_k2;
    }
    fprintf(stderr, "ds case reprobe_every %d noise %.2f: width 2 first chosen in round %d, extensions %lld, k2 in the last 500 rounds %d; %s\n",
            reprobe_every, noise, first_k2, (long long) n_ext, k2_last, s.request_summary(0).c_str());
    return 0;
}

static int test_reprobe() {
    int     k2_last = 0;
    double  k2_cost = 0.0;
    int64_t n_ext   = 0;

    // control, no extension: width 2 stays at its slow value and is never chosen
    CHECK(run_ds_case(0, 3000, k2_last, k2_cost, n_ext) == 0);
    CHECK(n_ext == 0);
    CHECK(std::fabs(k2_cost - 60.0) < 1e-6);
    CHECK(k2_last == 0);

    // extensions (defaults: below 4 observations, at most every 64 rounds): width 2 is measured again and taken
    int first = -1;
    CHECK(run_ds_case(64, 3000, k2_last, k2_cost, n_ext, 0.0, &first) == 0);
    CHECK(n_ext > 0);
    CHECK(k2_cost < 36.0);
    CHECK(k2_last >= 490);
    CHECK(first > 0 && first < 400);
    return 0;
}

// a round of the extension unit tests: level 10 ms, the shapes of `ms`, confidence `n` per width (below 4: waiting),
// draft step 0.1 ms, q = p, qbar `qb`; every width used and timed once by this process; the clock at `now`
static void ext_state(common_spec_smart & s, const std::vector<double> & ms, const std::vector<double> & n, double qb,
                      int64_t now, double draft_us = 100.0) {
    set_identity(s, qb);
    s.level_cell = { 10000.0, 1e6 };
    for (size_t k = 0; k < ms.size(); ++k) {
        s.shape_cells[k] = { ms[k] / 10.0, n[k] };
        s.cost_n_run[k]  = 1;
        s.used_run[k]    = 1;
    }
    s.draft_cell = { draft_us, 1e6 };
    s.n_closed   = now;
}

static int test_extend() {
    const std::vector<double> MS = { 10, 11, 30, 31, 32 };
    // the decision alone (every width confident): one token at p = 0.925, stop at width 1
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, MS, { 100, 100, 100, 100, 100 }, 0.9, 64);
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.925f) == 1);
        CHECK(s.n_extend == 0 && s.seqs[0].k_nat == -1);
    }
    // width 2 waits (confidence 3): the round drafts one more token and verifies width 2
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, MS, { 100, 100, 3, 100, 100 }, 0.9, 64);
        CHECK(s.waiting(2) && !s.waiting(1) && s.can_extend(2, 4) && !s.can_extend(3, 4));
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.925f) == -1);
        CHECK(s.seqs[0].k_nat == 1 && s.seqs[0].ext_step && s.seqs[0].forced == 2);
        CHECK(s.n_extend == 1 && s.last_extend == 64 && s.seqs[0].n_extend == 1);
        CHECK(s.after_step(0, 0.1f) == 2); // the extra step is kept whatever its probability
        s.drafted(0, 2, 0);
        CHECK(s.seqs[0].k == 2);
        // the interval: not again before 64 closed rounds
        s.n_closed = 127;
        CHECK(!s.can_extend(2, 4));
        s.n_closed = 128;
        CHECK(s.can_extend(2, 4));
    }
    // two or more widths above the decision: no extension (width 3 waits, the decision stops at 1)
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, MS, { 100, 100, 100, 3, 3 }, 0.9, 64);
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.925f) == 1);
        CHECK(s.n_extend == 0);
    }
    // several widths wait (2, 3, 4): only the one after the decision, one width
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, MS, { 100, 100, 3, 0, 0 }, 0.9, 64);
        CHECK(s.begin_round(0, 4));
        CHECK(s.after_step(0, 0.925f) == -1);
        CHECK(s.after_step(0, 0.925f) == 2);
        CHECK(s.n_extend == 1);
    }
    // before the interval (the clock at 63, the last extension at 0), with extensions off, and at the request limit
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, MS, { 100, 100, 3, 100, 100 }, 0.9, 63);
        CHECK(s.begin_round(0, 4) && s.after_step(0, 0.925f) == 1 && s.n_extend == 0);
        common_spec_smart_config c0 = base_cfg(4);
        c0.reprobe_every = 0;
        common_spec_smart o(c0, 1);
        ext_state(o, MS, { 100, 100, 3, 100, 100 }, 0.9, 1000);
        CHECK(o.begin_round(0, 4) && o.after_step(0, 0.925f) == 1 && o.n_extend == 0);
        common_spec_smart l(base_cfg(4), 1);
        ext_state(l, MS, { 100, 100, 3, 100, 100 }, 0.9, 64);
        CHECK(l.begin_round(0, 1) && l.after_step(0, 0.925f) == 1 && l.n_extend == 0); // width 2 above the limit
    }
    // the decision drafts nothing and width 1 waits: one token (width 1)
    {
        common_spec_smart s(base_cfg(4), 1);
        ext_state(s, { 10, 50, 60, 70, 80 }, { 100, 3, 100, 100, 100 }, 0.9, 64);
        CHECK(s.decide(nullptr, 0, 4) == 0);
        CHECK(s.begin_round(0, 4));
        CHECK(s.seqs[0].k_nat == 0 && s.seqs[0].ext_step && s.seqs[0].forced == 1);
        CHECK(s.after_step(0, 0.3f) == 1);
        // width 1 confident: no draft
        common_spec_smart o(base_cfg(4), 1);
        ext_state(o, { 10, 50, 60, 70, 80 }, { 100, 100, 3, 100, 100 }, 0.9, 64);
        CHECK(!o.begin_round(0, 4) && o.n_extend == 0);
    }
    // the decision drops a drafted token (keeps 1 of 2) and width 2 waits: the dropped token is kept, no extra step
    {
        common_spec_smart s(base_cfg(2), 1);
        ext_state(s, { 50, 60, 61 }, { 100, 100, 3 }, 0.5, 64, 1000.0);
        const float p[2] = { 0.975f, 0.025f };
        CHECK(s.decide(p, 2, 2) == 1);
        CHECK(s.begin_round(0, 2));
        CHECK(s.after_step(0, p[0]) == -1);
        CHECK(s.after_step(0, p[1]) == 2);
        CHECK(s.seqs[0].k_nat == 1 && !s.seqs[0].ext_step && s.n_extend == 1);
        common_spec_smart o(base_cfg(2), 1);
        ext_state(o, { 50, 60, 61 }, { 100, 100, 100 }, 0.5, 64, 1000.0);
        CHECK(o.begin_round(0, 2) && o.after_step(0, p[0]) == -1 && o.after_step(0, p[1]) == 1);
    }
    fprintf(stderr, "extension: only width k + 1 after a stop at k, one width, at most every 64 rounds, inside the limit\n");
    return 0;
}

static int test_store_decay() {
    // the store keeps the decayed confidence; a load starts its clock at 0
    const std::string path  = (std::filesystem::temp_directory_path() / "test-spec-smart-decay.txt").string();
    const std::string path2 = (std::filesystem::temp_directory_path() / "test-spec-smart-decay2.txt").string();
    common_spec_smart_config c = base_cfg(3);
    c.store_key = "k";
    common_spec_smart w(c, 1);
    for (int k = 0; k <= 3; ++k) {
        w.shape_cells[k] = { 1.0 * (k + 1), 20.0 * (k + 1), 0, 0.01 * (k + 1) };
    }
    w.level_cell = { 1000.0, 80.0, 2000 - 64 }; // one level half-life before the save
    w.cal_cells[7]   = { 0.6, 40.0 };
    w.qbar_cells[0]  = { 0.7, 64.0 };
    w.shape_cells[3].t = 1000; // last observed a half-life later than the others
    w.n_closed = 2000;
    CHECK(w.save(path));

    common_spec_smart_config c2 = c;
    c2.cap_load   = 1e300;
    c2.level_cap_load = 1e300;
    c2.store_path = path;
    common_spec_smart r(c2, 1);
    CHECK(r.store_loaded && r.n_closed == 0);
    for (int k = 0; k < 3; ++k) {
        CHECK(r.shape_cells[k].mean == w.shape_cells[k].mean);
        CHECK(r.shape_cells[k].n == w.conf(w.shape_cells[k]) && std::fabs(r.shape_cells[k].n - 5.0 * (k + 1)) < 1e-9);
        CHECK(r.shape_cells[k].t == 0 && r.shape_cells[k].var == w.shape_cells[k].var);
    }
    CHECK(std::fabs(r.shape_cells[3].n - 40.0) < 1e-9);
    CHECK(r.level_cell.mean == 1000.0 && std::fabs(r.level_cell.n - 40.0) < 1e-9 && r.level_cell.t == 0);
    CHECK(std::fabs(r.cal_cells[7].n - 10.0) < 1e-9 && r.cal_cells[7].mean == 0.6);
    CHECK(std::fabs(r.qbar_cells[0].n - 16.0) < 1e-9 && r.qbar_cells[0].mean == 0.7);
    r.cfg.store_path.clear();
    CHECK(r.save(path2));
    CHECK(read_file(path2) == read_file(path)); // at round 0 nothing more decays: byte-identical

    // the normal load caps at 32, the wait confidence of the shape at the running cap 16
    common_spec_smart_config c3 = c;
    c3.store_path = path;
    common_spec_smart l(c3, 1);
    CHECK(l.store_loaded && l.shape_cells[3].n == 32.0 && l.shape_cells[3].mean == 4.0 && l.wait_conf(3) == 16.0);
    CHECK(std::fabs(l.wait_conf(1) - 10.0) < 1e-9 && l.wait_conf(1) == l.conf(l.shape_cells[1])); // below the cap: as stored
    CHECK(l.level_cell.n == 2.0 && l.level_cell.mean == 1000.0 && std::fabs(l.cost(3) - 4000.0) < 1e-9);
    l.cfg.store_path.clear();

    // the formats before the time variance, before the mean acceptance per position and before the level and
    // shape are refused: cold start
    CHECK(read_file(path).rfind("ranma-spec-smart 6\n", 0) == 0);
    std::string why;
    for (const char * old : { "ranma-spec-smart 2", "ranma-spec-smart 4", "ranma-spec-smart 5" }) {
        {
            std::string t = read_file(path);
            t.replace(0, t.find('\n'), old);
            std::ofstream f(path2, std::ios::binary | std::ios::trunc);
            f << t;
        }
        common_spec_smart_config c4 = c;
        c4.store_path   = path2;
        common_spec_smart o(c4, 1);
        CHECK(!o.store_loaded);
        CHECK(!o.load(path2, &why) && why == "unknown format");
        o.cfg.store_path.clear();
    }
    common_spec_smart o(c, 1);
    {
        // ratio lines without the variance are rejected as a whole
        std::string t = read_file(path);
        size_t pos = 0;
        while ((pos = t.find("\nratio ", pos)) != std::string::npos) {
            const size_t e  = t.find('\n', pos + 1);
            const size_t sp = t.rfind(' ', e);
            t.erase(sp, e - sp);
            pos = sp;
        }
        std::ofstream f(path2, std::ios::binary | std::ios::trunc);
        f << t;
    }
    CHECK(!o.load(path2, &why) && why == "unknown line" && !o.store_loaded);
    {
        // a file without the level line is incomplete
        std::string t = read_file(path);
        const size_t a = t.find("\nlevel ");
        t.erase(a, t.find('\n', a + 1) - a);
        std::ofstream f(path2, std::ios::binary | std::ios::trunc);
        f << t;
    }
    CHECK(!o.load(path2, &why) && why == "truncated" && !o.store_loaded);

    std::filesystem::remove(path);
    std::filesystem::remove(path2);
    return 0;
}

// the DS case (width 2: true 34 ms, timed once at 60 ms) with 5 % noise on every time
static int test_ds_noise() {
    int     k2_last = 0;
    double  k2_cost = 0.0;
    int64_t n_ext   = 0;
    int     first   = -1;

    CHECK(run_ds_case(64, 3000, k2_last, k2_cost, n_ext, 0.05, &first) == 0);
    CHECK(first >= 0 && k2_last >= 450);
    return 0;
}

// synthetic acceptance given the accepted prefix: 0.75 / 0.35 / 0.15 / 0.1 at positions 0..3 (a steep fall with depth)
static const double SYN_ACC[4] = { 0.75, 0.35, 0.15, 0.1 };

static int syn_accepted(std::mt19937 & rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int n_acc = 0;
    while (n_acc < 4 && u(rng) < SYN_ACC[n_acc]) {
        n_acc++;
    }
    return n_acc;
}

// the final draft length of one round with a fixed draft probability (decisions only, nothing observed)
static int draft_length(const common_spec_smart & s, float p, int n_max_seq) {
    if (s.decide(nullptr, 0, n_max_seq) >= 0) {
        return 0;
    }
    std::vector<float> pv;
    for (;;) {
        pv.push_back(p);
        const int r = s.decide(pv.data(), (int) pv.size(), n_max_seq);
        if (r >= 0) {
            return r;
        }
    }
}

static int test_qbar_positions() {
    // one mean per position, learned apart (drafts of 4 at p = 0.625)
    common_spec_smart s(base_cfg(4), 1);
    CHECK(s.qbar_cells.size() == 5);
    const float pv[4] = { 0.625f, 0.625f, 0.625f, 0.625f };
    {
        std::mt19937 rng(5);
        for (int r = 0; r < 40000; ++r) {
            s.observe_acceptance(pv, 4, syn_accepted(rng));
        }
    }
    fprintf(stderr, "qbar per position: %.3f %.3f %.3f %.3f (n %.0f %.0f %.0f %.0f), cell 4 n %.0f\n", s.qbar(0),
            s.qbar(1), s.qbar(2), s.qbar(3), s.qbar_cells[0].n, s.qbar_cells[1].n, s.qbar_cells[2].n,
            s.qbar_cells[3].n, s.qbar_cells[4].n);
    for (int j = 0; j < 4; ++j) {
        CHECK(std::fabs(s.qbar(j) - SYN_ACC[j]) < (j < 3 ? 0.02 : 0.03));
    }
    CHECK(s.qbar_cells[4].n == 0.0); // a draft of 4 has no position 4
    CHECK(s.qbar(9) == s.qbar(4));   // beyond the last cell: the last cell

    // the controller before: the same observations, positions 2 and later pooled in one cell; the calibration cells
    // (position groups 0, 1, 2+) are the same in both
    common_spec_smart o(base_cfg(4), 1);
    o.cal_cells = s.cal_cells;
    common_spec_smart_cell pooled;
    {
        std::mt19937 rng(5);
        for (int r = 0; r < 40000; ++r) {
            const int n_acc = syn_accepted(rng);
            for (int j = 2; j <= std::min(n_acc, 3); ++j) {
                pooled.add(j < n_acc ? 1.0 : 0.0, o.decay, 0);
            }
        }
    }
    o.qbar_cells[0] = s.qbar_cells[0];
    o.qbar_cells[1] = s.qbar_cells[1];
    for (int j = 2; j <= 4; ++j) {
        o.qbar_cells[j] = pooled;
    }
    fprintf(stderr, "pooled positions 2+: %.3f\n", o.qbar(2));
    CHECK(o.qbar(3) > s.qbar(3) + 0.03);

    // over a range of cost slopes and draft probabilities: less often to width 4 and a lower total length
    int sum_new = 0, sum_old = 0, w4_new = 0, w4_old = 0, deeper = 0, deeper4 = 0, n_cases = 0;
    for (int i = 0; i <= 80; ++i) {
        const double slope = 0.25 * i; // ms per width
        std::vector<double> ms;
        for (int k = 0; k <= 4; ++k) {
            ms.push_back(50.0 + slope * k);
        }
        for (auto * x : { &s, &o }) {
            reset_costs(*x);
            set_costs(*x, ms, 1.0);
        }
        for (float p : { 0.525f, 0.625f, 0.925f }) {
            const int kn = draft_length(s, p, 4);
            const int ko = draft_length(o, p, 4);
            sum_new += kn;
            sum_old += ko;
            w4_new += kn == 4;
            w4_old += ko == 4;
            deeper += kn > ko;
            deeper4 += kn > ko && kn == 4;
            n_cases++;
        }
    }
    fprintf(stderr, "per position vs pooled 2+ over %d cases: sum of lengths %d vs %d, width 4 %d vs %d, deeper %d "
            "(to width 4: %d)\n", n_cases, sum_new, sum_old, w4_new, w4_old, deeper, deeper4);
    // position 2 alone (0.15) is a little above the pool of 2 and 3 (0.14), so a few rounds may go to width 3 more
    // often; never to width 4, where the pool overstated the acceptance (0.14 against 0.1)
    CHECK(sum_new < sum_old && w4_new < w4_old);
    CHECK(deeper4 == 0);
    return 0;
}

static int test_lookahead_mix() {
    // n_max 2, costs 50 / 50 / 58 ms, draft step 1 ms, q = p. After one step: keep 1 = (1 + p) / 51,
    // one more = (1 + p + p * qj) / 60 with qj = a * p + (1 - a) * qbar.
    // high p = 0.975, qbar = 0.2: a = 0: qj 0.2 -> keep 1; a = 0.5: qj 0.5875 -> one more
    // low  p = 0.325, qbar = 0.9: a = 0: qj 0.9 -> one more; a = 0.5: qj 0.6125 -> keep 1
    for (double a : { 0.0, 0.5 }) {
        common_spec_smart_config c = base_cfg(2);
        c.lookahead_mix = a;
        {
            common_spec_smart s(c, 1);
            set_identity(s, 0.2);
            set_costs(s, {50, 50, 58}, 1.0);
            const float p = 0.975f;
            CHECK(s.decide(&p, 1, 2) == (a > 0.0 ? -1 : 1));
            CHECK(std::fabs(s.lookahead(1, &p, 1) - (a * s.q(0, p) + (1.0 - a) * s.qbar(1))) < 1e-12);
            CHECK(s.lookahead(1, nullptr, 0) == s.qbar(1)); // before the first step: qbar only
        }
        {
            common_spec_smart s(c, 1);
            set_identity(s, 0.9);
            set_costs(s, {50, 50, 58}, 1.0);
            const float p = 0.325f;
            CHECK(s.decide(&p, 1, 2) == (a > 0.0 ? 1 : -1));
        }
    }
    // before the first step the mix does not matter
    {
        std::mt19937 rng(21);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int it = 0; it < 500; ++it) {
            common_spec_smart_config c0 = base_cfg(4);
            common_spec_smart_config c1 = c0;
            c1.lookahead_mix = 0.5;
            common_spec_smart s0(c0, 1);
            common_spec_smart s1(c1, 1);
            std::vector<double> ms;
            for (int k = 0; k <= 4; ++k) {
                ms.push_back(20.0 + 100.0 * u(rng));
            }
            const double d = 5.0 * u(rng);
            set_costs(s0, ms, d);
            set_costs(s1, ms, d);
            for (size_t j = 0; j < s0.qbar_cells.size(); ++j) {
                s0.qbar_cells[j] = s1.qbar_cells[j] = { u(rng), 100.0 * u(rng) };
            }
            CHECK(s0.decide(nullptr, 0, 4) == s1.decide(nullptr, 0, 4));
        }
    }
    // whole rounds with a fixed draft probability: with a > 0 a confident draft goes at least as deep as with a = 0
    // and an unsure one at most as deep, over a range of cost slopes, and strictly in some
    {
        int more_hi = 0, less_lo = 0, wrong = 0;
        for (int i = 0; i <= 40; ++i) {
            const double slope = 0.5 * i;
            common_spec_smart_config c = base_cfg(4);
            common_spec_smart s0(c, 1);
            c.lookahead_mix = 0.5;
            common_spec_smart s1(c, 1);
            for (auto * x : { &s0, &s1 }) {
                set_identity(*x, 0.6);
                set_costs(*x, {50, 50 + slope, 50 + 2 * slope, 50 + 3 * slope, 50 + 4 * slope}, 1.0);
            }
            const int hi0 = draft_length(s0, 0.975f, 4), hi1 = draft_length(s1, 0.975f, 4);
            const int lo0 = draft_length(s0, 0.425f, 4), lo1 = draft_length(s1, 0.425f, 4);
            more_hi += hi1 > hi0;
            less_lo += lo1 < lo0;
            wrong   += hi1 < hi0 || lo1 > lo0;
        }
        fprintf(stderr, "lookahead mix 0.5 vs 0 over 41 cost slopes: p 0.975 deeper in %d, p 0.425 shallower in %d, "
                "the other way in %d\n", more_hi, less_lo, wrong);
        CHECK(wrong == 0 && more_hi > 0 && less_lo > 0);
    }
    return 0;
}

// a = 0 with per-position cells equal to the old groups (positions 2+ the same cell): the decision of the old code
static int ref_decide_grouped(const common_spec_smart & s, const float * p, int n_drafted, int n_max_seq) {
    n_max_seq = std::min(n_max_seq, s.cfg.n_max);
    const std::vector<double> C = s.cost_table();
    const double d = s.draft_step();
    const double D = n_drafted * d;
    double best = -1.0, A = 0.0, S = 1.0;
    int best_k = 0;
    for (int k = 0; k <= n_drafted; ++k) {
        if (k > 0) {
            S *= s.q(k - 1, p[k - 1]);
            A += S;
        }
        const double T = (A + 1.0) / std::max(D + C[k], 1.0);
        if (T > best) {
            best   = T;
            best_k = k;
        }
    }
    double cont = -1.0, E = A, Sl = S;
    for (int m = 1; n_drafted + m <= n_max_seq; ++m) {
        Sl *= s.qbar(std::min(n_drafted + m - 1, 2)); // groups 0, 1, 2+
        E  += Sl;
        cont = std::max(cont, (E + 1.0) / std::max(D + m * d + C[n_drafted + m], 1.0));
    }
    return cont > best ? -1 : best_k;
}

static int test_mix0_old() {
    std::mt19937 rng(33);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    int n_cont = 0, n_stop = 0;
    for (int it = 0; it < 3000; ++it) {
        common_spec_smart s(base_cfg(6), 1);
        for (auto & x : s.cal_cells) {
            x = { u(rng), 50.0 * u(rng) };
        }
        s.qbar_cells[0] = { u(rng), 50.0 * u(rng) };
        s.qbar_cells[1] = { u(rng), 50.0 * u(rng) };
        const common_spec_smart_cell g2 = { u(rng), 50.0 * u(rng) };
        for (size_t j = 2; j < s.qbar_cells.size(); ++j) {
            s.qbar_cells[j] = g2;
        }
        std::vector<double> ms;
        double base = 20.0 + 30.0 * u(rng);
        for (int k = 0; k <= 6; ++k) {
            ms.push_back(base);
            base += 15.0 * u(rng);
        }
        set_costs(s, ms, 3.0 * u(rng));
        std::vector<float> p;
        const int n_drafted = (int) (u(rng) * 6.0);
        for (int i = 0; i < n_drafted; ++i) {
            p.push_back((float) u(rng));
        }
        const int n_max_seq = n_drafted + (int) (u(rng) * (7 - n_drafted));
        const int r = s.decide(p.data(), n_drafted, n_max_seq);
        CHECK(r == ref_decide_grouped(s, p.data(), n_drafted, n_max_seq));
        if (r < 0) {
            n_cont++;
        } else {
            n_stop++;
        }
    }
    fprintf(stderr, "mix 0 against the grouped decision: 3000 equal (%d one more, %d stop)\n", n_cont, n_stop);
    CHECK(n_cont > 100 && n_stop > 100);
    return 0;
}

// synthetic verification times at level 1 (ms): with q = p = 0.925, qbar 0.9 and a draft step of 1 ms the decision
// takes width 2; the other widths get only what the decisions of a cold start and the extensions give them
static const double SHAPE_MS[5] = { 30.0, 36.0, 40.0, 60.0, 80.0 };

struct level_env_result {
    std::vector<int> k_count;     // rounds per width after the change
    std::vector<int> ext_count;   // extended rounds per width after the change (verified width)
    std::vector<int> ext_before;  // the same before the change
    std::vector<int64_t> n_run0;  // timed observations per width of this process at the change
    std::vector<int> within10;    // rounds after the change until the prediction stays within 10 % of the truth
    std::vector<int> within5;     // the same for 5 % (-1: not within the phase)
    std::vector<double> max_err;  // largest relative error per width after `settle` rounds of the phase
};

// runs n0 rounds with truth(k, 0), then n1 rounds with truth(k, 1) (a request boundary between). The rounds
// counted are those of the second phase.
template <typename F>
static int run_level_env(common_spec_smart & s, int n0, int n1, int settle, F truth, level_env_result & res) {
    const int N = s.cfg.n_max + 1;
    int64_t t = 0;
    s.request_begin(0);
    res.ext_before.assign(N, 0);
    for (int r = 0; r < n0; ++r) {
        const int k = run_round(s, t, 0.925f, N - 1, [&](int w) { return truth(w, 0); });
        if (s.seqs[0].k_nat >= 0) {
            res.ext_before[k]++;
        }
    }
    res.n_run0 = s.cost_n_run;
    res.ext_count.assign(N, 0);
    res.k_count.assign(N, 0);
    res.within10.assign(N, -1);
    res.within5.assign(N, -1);
    res.max_err.assign(N, 0.0);
    std::vector<int> last_out10(N, 0), last_out5(N, 0);
    s.request_begin(0);
    for (int r = 1; r <= n1; ++r) {
        const int k = run_round(s, t, 0.925f, N - 1, [&](int w) { return truth(w, 1); });
        res.k_count[k]++;
        if (s.seqs[0].k_nat >= 0) {
            res.ext_count[k]++;
        }
        const auto C = s.cost_table();
        for (int w = 0; w < N; ++w) {
            const double err = std::fabs(C[w] / (1000.0 * truth(w, 1)) - 1.0);
            if (err >= 0.10) { last_out10[w] = r; }
            if (err >= 0.05) { last_out5[w] = r; }
            if (r > settle) {
                res.max_err[w] = std::max(res.max_err[w], err);
            }
        }
    }
    for (int w = 0; w < N; ++w) {
        res.within10[w] = last_out10[w] < n1 ? last_out10[w] : -1;
        res.within5[w]  = last_out5[w] < n1 ? last_out5[w] : -1;
    }
    return 0;
}

static void print_env(const char * name, const common_spec_smart & s, const level_env_result & r) {
    fprintf(stderr, "%s: rounds to stay within 10 %% / 5 %% per width:", name);
    for (size_t w = 0; w < r.within10.size(); ++w) {
        fprintf(stderr, " k%zu %d/%d", w, r.within10[w], r.within5[w]);
    }
    fprintf(stderr, "; rounds per width:");
    for (size_t w = 0; w < r.k_count.size(); ++w) {
        fprintf(stderr, " %d", r.k_count[w]);
    }
    fprintf(stderr, "; extensions per width before / after the change:");
    for (size_t w = 0; w < r.ext_count.size(); ++w) {
        fprintf(stderr, " %d/%d", r.ext_before[w], r.ext_count[w]);
    }
    fprintf(stderr, "; timed observations per width at the change:");
    for (size_t w = 0; w < r.n_run0.size(); ++w) {
        fprintf(stderr, " %lld", (long long) r.n_run0[w]);
    }
    fprintf(stderr, "; largest error after settling:");
    for (size_t w = 0; w < r.max_err.size(); ++w) {
        fprintf(stderr, " %.3f", r.max_err[w]);
    }
    fprintf(stderr, "; level %.2f ms, shape", s.level() / 1000.0);
    for (const auto & c : s.shape_cells) {
        fprintf(stderr, " %.3f", c.mean);
    }
    fprintf(stderr, "\n");
}

// fixed shape; the level rises 1.3x at a request boundary and falls back later: the widths that are not used follow
// through the level alone. A cold start (no sweep): a width that no decision and no extension verified is predicted
// from its neighbours (width 1 here: the line through 30 and 40 gives 35, true 36).
static int test_level_follow() {
    int worst10 = 0, worst5 = 0;
    for (double hl : { 1000.0, 250.0 }) {
        common_spec_smart_config c = base_cfg(4);
        c.half_life = hl;
        for (double f : { 1.3, 1.0 / 1.3 }) {
            common_spec_smart s(c, 1);
            set_identity(s, 0.9);
            level_env_result r;
            const double from = f > 1.0 ? 1.0 : 1.3;
            CHECK(run_level_env(s, 2000, 1500, 400, [&](int w, int ph) { return SHAPE_MS[w] * (ph ? from * f : from); }, r) == 0);
            char name[64];
            snprintf(name, sizeof(name), "level %s, half-life %.0f", f > 1.0 ? "x1.3" : "x1/1.3", hl);
            print_env(name, s, r);
            CHECK(r.k_count[2] > 1400); // width 2 in use
            // the width after it is verified again by extensions after the change of level
            CHECK(r.ext_count[3] >= 1 && r.ext_count[3] == r.ext_count[0] + r.ext_count[1] + r.ext_count[2] + r.ext_count[3] + r.ext_count[4]);
            // the width in use and the one after it (extended) within 5 %; the others (never verified again: width 0
            // and 1 below, width 4 never verified, its shape from the line through its neighbours) within 10 %
            for (int w = 0; w <= 4; ++w) {
                CHECK(r.within10[w] >= 0);
                CHECK(r.max_err[w] < 0.10);
                worst10 = std::max(worst10, r.within10[w]);
                if (w == 2 || w == 3) {
                    CHECK(r.within5[w] >= 0 && r.max_err[w] < 0.05);
                    worst5 = std::max(worst5, r.within5[w]);
                }
            }
            // the shape of the width in use moves little: the level takes the change (half-life 1000; with 250 the
            // ratio to width 0, which is not verified again, drifts a little more: printed)
            const double ratio_err = std::fabs(s.shape_cells[2].mean / s.shape_cells[0].mean / (SHAPE_MS[2] / SHAPE_MS[0]) - 1.0);
            fprintf(stderr, "  ratio of width 2 to width 0: error %.3f\n", ratio_err);
            if (hl == 1000.0) {
                CHECK(ratio_err < 0.05);
            }
        }
    }
    fprintf(stderr, "level follow: every width within 10 %% after %d rounds, widths 2 and 3 within 5 %% after %d rounds\n", worst10, worst5);
    CHECK(worst10 <= 200 && worst5 <= 300);
    return 0;
}

// the shape changes: width 3 (not used, the width after the one in use) becomes 1.3x slower at a request boundary.
// Its prediction follows through the extensions and the half-life of its shape; the other widths do not move.
static int test_shape_change() {
    common_spec_smart_config c = base_cfg(4);
    for (double hl : { 1000.0, 250.0 }) {
        c.half_life = hl;
        common_spec_smart s(c, 1);
        set_identity(s, 0.9);
        level_env_result r;
        CHECK(run_level_env(s, 2000, 8000, 0, [&](int w, int ph) { return SHAPE_MS[w] * (ph && w == 3 ? 1.3 : 1.0); }, r) == 0);
        char name[64];
        snprintf(name, sizeof(name), "width 3 x1.3, half-life %.0f", hl);
        print_env(name, s, r);
        CHECK(r.within10[3] >= 0 && r.within5[3] >= 0 && r.ext_count[3] > 0);
        for (int w : { 0, 1, 2 }) {
            CHECK(r.max_err[w] < 0.05); // the others do not move from the start of the phase
        }
        // width 4 is never verified in this run (the frontier stays at 3): its prediction is the line through widths
        // 2 and 3 and follows the change of width 3 (printed, not checked)
        CHECK(r.n_run0[4] == 0 && s.cost_n_run[4] == 0);
    }
    // the width in use changes: the level first takes it (all widths move); the extensions move width 3 back, the
    // others follow only when a decision takes them (printed)
    {
        c.half_life = 1000.0;
        common_spec_smart s(c, 1);
        set_identity(s, 0.9);
        level_env_result r;
        CHECK(run_level_env(s, 2000, 8000, 4000, [&](int w, int ph) { return SHAPE_MS[w] * (ph && w == 2 ? 1.3 : 1.0); }, r) == 0);
        print_env("width 2 (in use) x1.3, half-life 1000", s, r);
    }
    return 0;
}

//
// cold and warm starts without the sweep, and the discovery of wide widths (n_max 15)
//

// a synthetic drafter and target: the true verification time per width, the draft step, the extra time of the first
// verification of a width in the process (graph builds), and the draft top probability per position. A draft token is
// accepted with its top probability (the true calibration is q = p).
struct syn_model {
    const char *        name;
    std::vector<double> ms;       // true verification time per width (ms)
    double              draft_ms;
    double              graph_ms; // extra time of the first verification of a width
    int                 kind;     // 0: high acceptance at every depth (Qwen / Gemma), 1: acceptance falls with depth (DS),
                                  // 2: code (position 0 near 0.83, falling slowly with depth)
    double              level = 1.0;
    double              rel_sd = 0.0; // relative sd of the verification time (normal, 0: exact)
};

static float syn_p(const syn_model & m, int pos, std::mt19937 & rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (m.kind == 0) {
        return (float) (u(rng) < 0.85 ? 0.95 + 0.049 * u(rng) : 0.3 + 0.6 * u(rng));
    }
    if (m.kind == 2) {
        static const double code[5] = { 0.83, 0.75, 0.68, 0.62, 0.57 };
        return (float) std::clamp(code[std::min(pos, 4)] + 0.3 * (u(rng) - 0.5), 0.02, 0.99);
    }
    static const double base[5] = { 0.75, 0.5, 0.35, 0.25, 0.2 };
    return (float) std::clamp(base[std::min(pos, 4)] + 0.3 * (u(rng) - 0.5), 0.02, 0.99);
}

struct syn_result {
    std::vector<int> first;      // first round that verified the width (-1: never)
    std::vector<int> first_ext;  // 1: that round was an extension
    std::vector<int> n_k;        // rounds per width
    std::vector<int> n_ext;      // extended rounds per verified width
    int    rounds  = 0;
    double tokens  = 0.0;
    double time_ms = 0.0;
    int    bad     = 0;          // extensions that did not verify k_nat + 1
};

static void syn_run(common_spec_smart & s, const syn_model & m, int n_rounds, std::mt19937 & rng, int64_t & t,
                    std::vector<uint8_t> & graph_built, syn_result & res, int req_every = 300) {
    const int N = s.cfg.n_max + 1;
    if (res.first.empty()) {
        res.first.assign(N, -1);
        res.first_ext.assign(N, 0);
        res.n_k.assign(N, 0);
        res.n_ext.assign(N, 0);
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int r = 0; r < n_rounds; ++r) {
        if (res.rounds % req_every == 0) {
            s.request_begin(0);
        }
        s.close_round(0, t);
        const int64_t t0 = t;
        int k = 0;
        std::vector<float> ps;
        if (s.begin_round(0, N - 1)) {
            for (;;) {
                const float p = syn_p(m, (int) ps.size(), rng);
                ps.push_back(p);
                s.add_draft_step(m.draft_ms * 1000.0);
                t += (int64_t) (m.draft_ms * 1000.0);
                const int x = s.after_step(0, p);
                if (x >= 0) {
                    k = x;
                    break;
                }
            }
        }
        const bool ext = s.seqs[0].k_nat >= 0;
        if (ext && k != s.seqs[0].k_nat + 1) {
            res.bad++;
        }
        s.drafted(0, k, t);
        int a = 0;
        while (a < k && u(rng) < ps[a]) {
            a++;
        }
        if (k > 0) {
            s.accepted(0, a);
        }
        double v = m.ms[k] * m.level;
        if (m.rel_sd > 0.0) {
            std::normal_distribution<double> nd(0.0, 1.0);
            v *= std::max(0.2, 1.0 + m.rel_sd * nd(rng));
        }
        if (!graph_built[k]) {
            graph_built[k] = 1;
            v += m.graph_ms;
        }
        t += (int64_t) (v * 1000.0);
        if (res.first[k] < 0) {
            res.first[k]     = res.rounds;
            res.first_ext[k] = ext;
        }
        res.n_k[k]++;
        res.n_ext[k] += ext;
        res.tokens  += a + 1;
        res.time_ms += (t - t0) / 1000.0;
        res.rounds++;
    }
}

static void syn_print(const char * name, const common_spec_smart & s, const syn_result & r) {
    fprintf(stderr, "%s: %d rounds, %.3f tokens / ms, extensions %lld (%.2f %% of the rounds); first round per width (x: by an extension):",
            name, r.rounds, r.tokens / r.time_ms, (long long) s.n_extend, 100.0 * s.n_extend / r.rounds);
    for (size_t w = 0; w < r.first.size(); ++w) {
        fprintf(stderr, " k%zu %d%s", w, r.first[w], r.first_ext[w] ? "x" : "");
    }
    fprintf(stderr, "; rounds / extensions per width:");
    for (size_t w = 0; w < r.n_k.size(); ++w) {
        fprintf(stderr, " %d/%d", r.n_k[w], r.n_ext[w]);
    }
    fprintf(stderr, "\n");
}

static void syn_print_costs(const common_spec_smart & s, const syn_model & m) {
    const auto C = s.cost_table();
    fprintf(stderr, "  %s predicted / true ms per width (n: timed observations):", m.name);
    for (size_t w = 0; w < C.size(); ++w) {
        fprintf(stderr, " k%zu %.1f/%.1f(%lld)", w, C[w] / 1000.0, m.ms[w] * m.level, (long long) s.cost_n_run[w]);
    }
    fprintf(stderr, "\n");
}

static syn_model syn_qwen() {
    syn_model m{ "qwen-like", {}, 1.5, 150.0, 0 };
    for (int k = 0; k <= 15; ++k) {
        m.ms.push_back(20.0 + 0.8 * k); // nearly flat
    }
    return m;
}

static syn_model syn_ds() {
    syn_model m{ "ds-like", {}, 3.0, 150.0, 1 };
    for (int k = 0; k <= 15; ++k) {
        m.ms.push_back(28.0 + 6.0 * k); // steep (experts read from host memory)
    }
    return m;
}

static int test_cold_discovery() {
    // a cold start: no forced round; every extension verifies the width after the decision
    for (const syn_model & m : { syn_qwen(), syn_ds() }) {
        common_spec_smart_config c;
        c.n_max = 15;
        c.half_life = 250.0; // the queue value
        common_spec_smart s(c, 1);
        CHECK(!s.store_loaded);
        std::mt19937 rng(m.kind == 0 ? 101 : 202);
        int64_t t = 0;
        std::vector<uint8_t> graph(16, 0);
        syn_result r;
        syn_run(s, m, 3000, rng, t, graph, r);
        s.close_round(0, t);
        syn_print(m.name, s, r);
        syn_print_costs(s, m);
        CHECK(r.bad == 0);
        CHECK(s.n_extend <= 3000 / 64 + 1);   // the interval
        // the first time of every width is not observed: no prediction carries the 150 ms of a graph build
        const auto C = s.cost_table();
        for (int w = 0; w <= 15; ++w) {
            CHECK(C[w] < 1000.0 * (m.ms[w] + 50.0));
            if (r.first[w] >= 0) {
                CHECK(s.used_run[w]);
            }
        }
        if (m.kind == 0) {
            CHECK(r.first[7] >= 0 && r.first[15] >= 0); // the wide widths are found
            fprintf(stderr, "  qwen-like: width 7 first verified in round %d%s, width 15 in round %d%s\n", r.first[7],
                    r.first_ext[7] ? " (extension)" : "", r.first[15], r.first_ext[15] ? " (extension)" : "");
        } else {
            int wide = 0;
            for (int w = 4; w <= 15; ++w) {
                wide += r.n_ext[w];
            }
            fprintf(stderr, "  ds-like: extensions to width 4 or more: %d of %d rounds\n", wide, r.rounds);
            CHECK(wide <= 10);
        }
    }
    return 0;
}

// the level rises 1.3x after 2000 rounds of a cold start, on the Qwen-like target (width 15 in use) and on a
// Gemma-like one with a steeper time (20 + 2.5 k ms: a width near 10 in use). The widths next to the one in use are
// verified again by extensions (their shapes decay below the threshold); the others follow through the level.
static int test_level_wide() {
    for (int model = 0; model < 2; ++model) {
        for (double hl : { 250.0, 1000.0 }) {
            syn_model m = syn_qwen();
            if (model == 1) {
                m.name = "gemma-like";
                for (int k = 0; k <= 15; ++k) {
                    m.ms[k] = 20.0 + 2.5 * k;
                }
            }
            common_spec_smart_config c;
            c.n_max = 15;
            c.half_life = hl;
            c.reprobe_every = OLD_REPROBE_EVERY; // written for the interval of before
            common_spec_smart s(c, 1);
            std::mt19937 rng(303);
            int64_t t = 0;
            std::vector<uint8_t> graph(16, 0);
            syn_result r0, r1;
            syn_run(s, m, 2000, rng, t, graph, r0);
            m.level = 1.3;
            syn_run(s, m, 2000, rng, t, graph, r1);
            s.close_round(0, t);
            char name[128];
            snprintf(name, sizeof(name), "%s level x1.3 after 2000 rounds, half-life %.0f (the 2000 rounds after the change)", m.name, hl);
            syn_print(name, s, r1);
            syn_print_costs(s, m);
            // the width used most and the extensions after the change
            int main_w = 0;
            for (int w = 0; w <= 15; ++w) {
                if (r1.n_k[w] > r1.n_k[main_w]) {
                    main_w = w;
                }
            }
            int ext_near = 0, ext_far = 0;
            for (int w = 0; w <= 15; ++w) {
                (std::abs(w - main_w) <= 2 ? ext_near : ext_far) += r1.n_ext[w];
            }
            fprintf(stderr, "  main width %d, extensions after the change within 2 of it %d, farther %d\n", main_w, ext_near, ext_far);
            CHECK(r1.bad == 0);
            const auto C = s.cost_table();
            CHECK(std::fabs(C[main_w] / (1000.0 * m.ms[main_w] * m.level) - 1.0) < 0.05);
            if (model == 1) {
                CHECK(main_w < 15 && ext_near > 0);
                for (int w = std::max(0, main_w - 2); w <= std::min(15, main_w + 1); ++w) {
                    CHECK(std::fabs(C[w] / (1000.0 * m.ms[w] * m.level) - 1.0) < 0.05);
                }
            }
        }
    }
    return 0;
}

// a warm start: the store is used, no forced round, the first verification of every width is not observed
static int test_warm_start() {
    const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-warm.txt").string();
    std::filesystem::remove(path);
    const syn_model m = syn_qwen();
    common_spec_smart_config c;
    c.n_max = 15;
    c.half_life = 250.0;
    c.store_path = path;
    syn_result r0;
    {
        common_spec_smart s(c, 1);
        std::mt19937 rng(404);
        int64_t t = 0;
        std::vector<uint8_t> graph(16, 0);
        syn_run(s, m, 1500, rng, t, graph, r0);
        s.close_round(0, t);
        syn_print("qwen-like cold (store written)", s, r0);
    }
    CHECK(std::filesystem::exists(path));
    {
        common_spec_smart s(c, 1);
        CHECK(s.store_loaded && s.n_extend == 0);
        for (int w = 0; w <= 15; ++w) {
            CHECK(!s.used_run[w] && s.cost_n_run[w] == 0);
        }
        std::mt19937 rng(505);
        int64_t t = 0;
        std::vector<uint8_t> graph(16, 0); // a new process builds its graphs again
        syn_result r;
        syn_run(s, m, 1500, rng, t, graph, r);
        s.close_round(0, t);
        syn_print("qwen-like warm", s, r);
        syn_print_costs(s, m);
        CHECK(r.bad == 0);
        const auto C = s.cost_table();
        for (int w = 0; w <= 15; ++w) {
            CHECK(C[w] < 1000.0 * (m.ms[w] + 50.0));
            if (r.first[w] >= 0) {
                CHECK(s.used_run[w]);
                // the first closed round of every verified width was not timed
                CHECK(s.cost_n_run[w] <= r.n_k[w] - 1);
            }
        }
        s.cfg.store_path.clear();
    }
    std::filesystem::remove(path);
    return 0;
}

// a width with few observations and much noise does not move the level while a confident width exists
static int test_level_noise() {
    common_spec_smart_config c = base_cfg(4);
    c.reprobe_every = 0;
    common_spec_smart s(c, 1);
    for (int k = 0; k <= 4; ++k) {
        s.observe_cost(k, 1000.0 * SHAPE_MS[k]);
    }
    s.n_closed++;
    for (int r = 0; r < 300; ++r) {
        s.observe_cost(2, 1000.0 * SHAPE_MS[2]);
        s.n_closed++;
    }
    const auto   C0 = s.cost_table();
    const double L0 = s.level();
    for (double f : { 3.0, 0.3, 2.5, 0.2 }) {
        s.observe_cost(4, f * 1000.0 * SHAPE_MS[4]);
        s.n_closed++;
    }
    const auto C1 = s.cost_table();
    fprintf(stderr, "level noise: level %.3f -> %.3f ms, width 4 %.2f -> %.2f ms (confidence %.1f)\n", L0 / 1000.0,
            s.level() / 1000.0, C0[4] / 1000.0, C1[4] / 1000.0, s.conf(s.shape_cells[4]));
    CHECK(s.level() == L0);
    for (int k = 0; k <= 3; ++k) {
        CHECK(C1[k] == C0[k]);
    }
    CHECK(C1[4] != C0[4]);

    // without a confident width (cold), every observation moves the level
    common_spec_smart z(c, 1);
    z.observe_cost(0, 30000.0);
    z.observe_cost(4, 80000.0);
    const double Lz = z.level();
    z.observe_cost(4, 240000.0);
    CHECK(z.level() > Lz);
    return 0;
}

//
// the shape confidence cap (conf_cap) and the cut of the old mean of a waiting width
//

// three controllers compared below: the default (cap 16, an observation of a waiting width counts the old mean with the
// wait confidence), the cap alone (the old mean with its full confidence) and neither
struct cap_variant {
    const char * name;
    double       cap;
    bool         wait_reset;
};
static const cap_variant CAP_VARIANTS[3] = {
    { "cap 16 + cut", 16.0, true },
    { "cap 16",       16.0, false },
    { "no cap",        0.0, false },
};

// the cap itself: the confidence that decides the wait never exceeds the cap, decays from it, and a width left long
// enough waits again. Its next observation counts the old mean with the wait confidence (the cut); without the cut the
// old mean keeps its full confidence
static int test_conf_cap() {
    for (const cap_variant & v : CAP_VARIANTS) {
        common_spec_smart_config c = base_cfg(2);
        c.half_life = 250.0;
        c.reprobe_every = 0;
        c.conf_cap = v.cap;
        c.wait_reset = v.wait_reset;
        common_spec_smart s(c, 1);
        for (int r = 0; r < 300; ++r) {
            s.observe_cost(0, 30000.0);
            s.observe_cost(1, 36000.0 + (r % 2 ? 2000.0 : -2000.0));
            s.n_closed++;
        }
        CHECK(s.n_wait_reset == 0); // a width in use never waits
        const double n1 = s.conf(s.shape_cells[1]);
        const double w1 = s.wait_conf(1);
        fprintf(stderr, "%s: width 1 after 300 rounds: confidence %.1f, wait confidence %.2f, mean %.4f\n", v.name,
                n1, w1, s.shape_cells[1].mean);
        CHECK(n1 > 150.0); // the confidence of the mean is not capped
        if (v.cap > 0.0) {
            CHECK(w1 <= 16.0 && w1 > 15.9);
        } else {
            CHECK(w1 == n1);
        }
        CHECK(!s.waiting(1));
        // left for log2(16 / 4) = 2 half-lives: waits again with the cap, not without it
        s.n_closed += 490;
        CHECK(s.waiting(1) == false);
        s.n_closed += 20;
        CHECK(s.waiting(1) == (v.cap > 0.0));
        // one observation: the wait confidence goes to about 5 (waits again after log2(5 / 4) half-lives)
        const double n_before = s.conf(s.shape_cells[1]);
        const double w_before = s.wait_conf(1);
        const double L_before = s.level();
        const double m_before = s.shape_cells[1].mean;
        const double var_before = s.shape_cells[1].var;
        s.observe_cost(1, 30000.0);
        const double x = 30000.0 / L_before;
        if (v.cap > 0.0) {
            CHECK(s.wait_conf(1) > 4.5 && s.wait_conf(1) <= 5.0 && !s.waiting(1));
        }
        if (v.wait_reset && v.cap > 0.0) {
            // the cut: the old mean weighs the wait confidence (about 3.9), the observation 1 / 4.9
            CHECK(s.n_wait_reset == 1 && s.wr_last && s.wr_n_before == n_before && s.wr_n_after == w_before);
            CHECK(std::fabs(s.conf(s.shape_cells[1]) - (w_before + 1.0)) < 1e-9);
            CHECK(std::fabs(s.wait_conf(1) - s.conf(s.shape_cells[1])) < 1e-9);
            CHECK(std::fabs(s.shape_cells[1].mean - (m_before + (x - m_before) / (w_before + 1.0))) < 1e-9);
            const double d = x - m_before;
            CHECK(std::fabs(s.shape_cells[1].var - (w_before * var_before + d * (x - s.shape_cells[1].mean)) / (w_before + 1.0)) < 1e-12);
        } else {
            CHECK(s.n_wait_reset == 0 && !s.wr_last);
            CHECK(std::fabs(s.conf(s.shape_cells[1]) - (n_before + 1.0)) < 1e-9);
            CHECK(std::fabs(s.shape_cells[1].mean - (m_before + (x - m_before) / (n_before + 1.0))) < 1e-9);
        }
    }
    // the cap does not change the mean, the variance or any decision input other than the wait while no width waits:
    // the same observations give the same cost table with and without the cap (width 3 is observed every 7 rounds and
    // keeps a wait confidence above 4; width 2 is never observed)
    {
        common_spec_smart_config c = base_cfg(3);
        c.reprobe_every = 0;
        common_spec_smart_config c0 = c;
        c0.conf_cap = 0.0;
        common_spec_smart a(c, 1), b(c0, 1);
        std::mt19937 rng(9);
        std::normal_distribution<double> nd(1.0, 0.1);
        for (int r = 0; r < 2000; ++r) {
            const int k = r % 7 == 0 ? 3 : (r % 3 == 0 ? 0 : 1);
            const double x = 1000.0 * (30.0 + 6.0 * k) * nd(rng);
            a.observe_cost(k, x);
            b.observe_cost(k, x);
            a.n_closed++;
            b.n_closed++;
        }
        CHECK(a.n_wait_reset == 0 && b.n_wait_reset == 0);
        const auto Ca = a.cost_table();
        const auto Cb = b.cost_table();
        for (int k = 0; k <= 3; ++k) {
            CHECK(Ca[k] == Cb[k]);
            CHECK(a.shape_cells[k].n == b.shape_cells[k].n && a.shape_cells[k].var == b.shape_cells[k].var);
        }
        CHECK(a.level() == b.level() && a.level_cell.n == b.level_cell.n);
    }
    // a width that waits without the cap (few observations, confidence below 4): the wait confidence is that of the
    // mean, nothing to cut
    {
        common_spec_smart_config c = base_cfg(2);
        c.conf_cap = 0.0;
        common_spec_smart s(c, 1);
        s.observe_cost(0, 30000.0);
        s.observe_cost(2, 40000.0);
        CHECK(s.waiting(2));
        s.observe_cost(2, 40000.0);
        CHECK(s.n_wait_reset == 0);
    }
    return 0;
}

// a width used, then slow for a while, then as fast as before. How many rounds until it is taken again.
struct recover_result {
    int    k2_ph1      = 0;  // rounds of width 2 or more while it is slow
    int    first_ext   = -1; // rounds after it is fast again: first extension to width 2
    int    first_nat   = -1; // the same: first round that a decision (not an extension) takes width 2 or more
    int    back        = -1; // the same: first round from which 50 of the last 100 rounds take width 2 or more
    int    ext_ph2     = 0;  // extensions in the fast phase (3000 rounds)
    int    wr_ph2      = 0;  // cuts of the old mean in the fast phase
    double tps_ph2     = 0.0;
    double wait_at_ph2 = 0.0; // wait confidence of width 2 when it is fast again
    double conf_at_ph2 = 0.0; // confidence of its mean
    double pred_at_ph2 = 0.0; // its prediction (ms)
};

// the code part of the Qwen mix: width 2 in use (45 ms), slow for 800 rounds (63 ms: the decision leaves it for width
// 1), fast again
static syn_model syn_code(double k2_ms) {
    syn_model m{ "code-like", {}, 1.5, 150.0, 2 };
    m.ms = { 33.0, 37.0, k2_ms };
    for (int k = 3; k <= 7; ++k) {
        m.ms.push_back(k2_ms + 8.0 * (k - 2));
    }
    return m;
}

// the rp-en part: width 1 28 ms, width 2 34 ms (in use), wider widths expensive (48 ms and 8 ms more per width). One
// request of 200 rounds is slow as a whole (x1.18: width 1 33 ms) and width 2 more (46 ms): the shape of width 2 rises
static syn_model syn_rp(bool slow) {
    syn_model m{ "rp-like", {}, 1.5, 150.0, 2 };
    m.ms = { 25.0, 28.0, 34.0 };
    for (int k = 3; k <= 7; ++k) {
        m.ms.push_back(48.0 + 8.0 * (k - 3));
    }
    if (slow) {
        for (auto & x : m.ms) {
            x *= 33.0 / 28.0;
        }
        m.ms[2] = 46.0;
    }
    return m;
}

// scenario 0: code (600 fast, 800 slow, 3000 fast); 1: rp (1000, 200 slow, 3000)
static int run_recover(const cap_variant & v, double hl, uint32_t seed, int scenario, recover_result & out) {
    common_spec_smart_config c;
    c.n_max = 7;
    c.half_life = hl;
    c.conf_cap = v.cap;
    c.wait_reset = v.wait_reset;
    common_spec_smart s(c, 1);
    std::mt19937 rng(seed);
    int64_t t = 0;
    std::vector<uint8_t> graph(8, 0);
    syn_result r;
    const syn_model fast = scenario == 0 ? syn_code(45.0) : syn_rp(false);
    const syn_model slow = scenario == 0 ? syn_code(63.0) : syn_rp(true);
    syn_run(s, fast, scenario == 0 ? 600 : 1000, rng, t, graph, r);
    syn_result r1;
    const int n_slow = scenario == 0 ? 800 : 200;
    for (int i = 0; i < n_slow; ++i) {
        syn_run(s, slow, 1, rng, t, graph, r1);
    }
    for (int w = 2; w <= 7; ++w) {
        out.k2_ph1 += r1.n_k[w];
    }
    out.wait_at_ph2 = s.wait_conf(2);
    out.conf_at_ph2 = s.conf(s.shape_cells[2]);
    out.pred_at_ph2 = s.cost(2) / 1000.0;
    syn_result r2;
    std::vector<int> wide;
    const int64_t ext0 = s.n_extend;
    const int64_t wr0  = s.n_wait_reset;
    double tok = 0.0, ms = 0.0;
    for (int i = 0; i < 3000; ++i) {
        const double tok0 = r2.tokens, ms0 = r2.time_ms;
        const int64_t e0 = s.n_extend;
        syn_run(s, fast, 1, rng, t, graph, r2);
        tok += r2.tokens - tok0;
        ms  += r2.time_ms - ms0;
        const bool ext = s.n_extend > e0;
        const int  k   = s.seqs[0].k;
        wide.push_back(k >= 2 ? 1 : 0);
        if (ext && k == 2 && out.first_ext < 0) {
            out.first_ext = i;
        }
        if (!ext && k >= 2 && out.first_nat < 0) {
            out.first_nat = i;
        }
        if (out.back < 0 && i >= 99) {
            int n = 0;
            for (int j = i - 99; j <= i; ++j) {
                n += wide[j];
            }
            if (n >= 50) {
                out.back = i - 99;
            }
        }
    }
    out.ext_ph2 = (int) (s.n_extend - ext0);
    out.wr_ph2  = (int) (s.n_wait_reset - wr0);
    out.tps_ph2 = tok / ms;
    return 0;
}

static int test_conf_cap_recover() {
    for (int scenario : { 1, 0 }) {
        fprintf(stderr, "%s (n_max 7, 5 seeds each):\n", scenario == 0 ?
                "code: width 2 45 ms, slow (63 ms) for 800 rounds, then fast again" :
                "rp: width 1 28 ms, width 2 34 ms; one request of 200 rounds x1.18 with width 2 at 46 ms; then as before");
        for (double hl : { 250.0, 1000.0 }) {
            int latest[3] = { 0, 0, 0 };
            for (int vi = 0; vi < 3; ++vi) {
                const cap_variant & v = CAP_VARIANTS[vi];
                fprintf(stderr, "  half-life %.0f, %s:", hl, v.name);
                int worst_back = 0, never = 0;
                double sum_tps = 0.0;
                for (uint32_t seed = 1; seed <= 5; ++seed) {
                    recover_result r;
                    CHECK(run_recover(v, hl, (scenario == 0 ? 700 : 750) + seed, scenario, r) == 0);
                    fprintf(stderr, "\n    [slow k>=2 %d; at the change: wait %.1f conf %.1f pred %.1f ms; first extension %d, "
                            "first decision %d, back %d, extensions %d, cuts %d, %.4f tok/ms]", r.k2_ph1, r.wait_at_ph2,
                            r.conf_at_ph2, r.pred_at_ph2, r.first_ext, r.first_nat, r.back, r.ext_ph2, r.wr_ph2, r.tps_ph2);
                    if (r.back < 0) { never++; } else { worst_back = std::max(worst_back, r.back); }
                    sum_tps += r.tps_ph2;
                    if (!v.wait_reset || v.cap == 0.0) {
                        CHECK(r.wr_ph2 == 0);
                    }
                }
                latest[vi] = never > 0 ? 1 << 30 : worst_back;
                fprintf(stderr, "\n  -> latest return %d, no return within 3000 rounds in %d of 5 seeds, mean %.4f tok/ms\n",
                        worst_back, never, sum_tps / 5.0);
                if (hl == 250.0 && vi < 2) {
                    CHECK(never == 0);
                }
            }
            // the cut returns no later than the cap alone
            if (hl == 250.0) {
                CHECK(latest[0] <= latest[1]);
            }
        }
    }
    return 0;
}

// a stable time (no change): the extensions of the three variants, Qwen-like and DS-like, n_max 7 and 15, half-life 250
static int test_conf_cap_ext_rate() {
    for (int n_max : { 7, 15 }) {
        for (const syn_model & m0 : { syn_qwen(), syn_ds() }) {
            int64_t n_ext[3] = { 0, 0, 0 }, n_ext_late[3] = { 0, 0, 0 }, n_wr[3] = { 0, 0, 0 };
            double  tps[3] = { 0, 0, 0 };
            for (int i = 0; i < 3; ++i) {
                syn_model m = m0;
                m.ms.resize(n_max + 1);
                common_spec_smart_config c;
                c.n_max = n_max;
                c.half_life = 250.0;
                c.conf_cap = CAP_VARIANTS[i].cap;
                c.wait_reset = CAP_VARIANTS[i].wait_reset;
                common_spec_smart s(c, 1);
                std::mt19937 rng(m.kind == 0 ? 101 : 202);
                int64_t t = 0;
                std::vector<uint8_t> graph(n_max + 1, 0);
                syn_result r;
                syn_run(s, m, 2000, rng, t, graph, r);
                const int64_t e0 = s.n_extend;
                syn_run(s, m, 4000, rng, t, graph, r);
                n_ext[i]      = s.n_extend;
                n_ext_late[i] = s.n_extend - e0;
                n_wr[i]       = s.n_wait_reset;
                tps[i]        = r.tokens / r.time_ms;
                char name[96];
                snprintf(name, sizeof(name), "%s n_max %d %s", m.name, n_max, CAP_VARIANTS[i].name);
                syn_print(name, s, r);
                CHECK(r.bad == 0);
            }
            fprintf(stderr, "  %s n_max %d, extensions in 6000 rounds (last 4000) / cuts / tok/ms:", m0.name, n_max);
            for (int i = 0; i < 3; ++i) {
                fprintf(stderr, " %s %lld (%lld) / %lld / %.4f;", CAP_VARIANTS[i].name, (long long) n_ext[i],
                        (long long) n_ext_late[i], (long long) n_wr[i], tps[i]);
            }
            fprintf(stderr, "\n");
            CHECK(n_ext[0] <= 6000 / 64 + 1 && n_ext[1] >= n_ext[2]);
            CHECK(tps[0] > 0.98 * tps[1]);
        }
    }
    return 0;
}

// noise on the verification time (relative sd 10 % and 20 %), a stable time: the prediction of the width taken, the
// share of the width used most, the throughput, the extensions and cuts of the three variants
static int test_conf_cap_noise() {
    for (double rel : { 0.1, 0.2 }) {
        for (int model = 0; model < 3; ++model) {
            double sum_sd[3] = { 0, 0, 0 }, sum_tps[3] = { 0, 0, 0 }, sum_ext[3] = { 0, 0, 0 }, sum_wr[3] = { 0, 0, 0 };
            double sum_main[3] = { 0, 0, 0 }, sum_switch[3] = { 0, 0, 0 };
            for (int i = 0; i < 3; ++i) {
                for (uint32_t seed = 1; seed <= 5; ++seed) {
                    syn_model m = model == 0 ? syn_code(45.0) : model == 1 ? syn_qwen() : syn_ds();
                    m.ms.resize(8);
                    m.rel_sd = rel;
                    common_spec_smart_config c;
                    c.n_max = 7;
                    c.half_life = 250.0;
                    c.conf_cap = CAP_VARIANTS[i].cap;
                    c.wait_reset = CAP_VARIANTS[i].wait_reset;
                    common_spec_smart s(c, 1);
                    std::mt19937 rng(900 + seed);
                    int64_t t = 0;
                    std::vector<uint8_t> graph(8, 0);
                    syn_result r0, r;
                    syn_run(s, m, 1000, rng, t, graph, r0);
                    // the relative error of the prediction of the width taken, over the rounds; the width with the
                    // lowest predicted time per expected token at a typical draft (the "preferred" width) and how often
                    // it changes
                    double se = 0.0;
                    int    ne = 0, n_switch = 0, last_best = -1;
                    for (int j = 0; j < 3000; ++j) {
                        syn_run(s, m, 1, rng, t, graph, r);
                        const int k = s.seqs[0].k;
                        const double e = s.cost(k) / (1000.0 * m.ms[k]) - 1.0;
                        se += e * e;
                        ne++;
                        const auto C = s.cost_table();
                        int best = 0;
                        double best_v = 1e300;
                        double S = 1.0, A = 0.0;
                        for (int w = 0; w <= 7; ++w) {
                            if (w > 0) {
                                S *= s.qbar(w - 1);
                                A += S;
                            }
                            const double val = (C[w] + w * s.draft_step()) / (A + 1.0);
                            if (val < best_v) {
                                best_v = val;
                                best = w;
                            }
                        }
                        n_switch += last_best >= 0 && best != last_best;
                        last_best = best;
                    }
                    int main_n = 0;
                    for (int w = 0; w <= 7; ++w) {
                        main_n = std::max(main_n, r.n_k[w]);
                    }
                    sum_sd[i]     += std::sqrt(se / ne);
                    sum_tps[i]    += r.tokens / r.time_ms;
                    sum_ext[i]    += (double) s.n_extend;
                    sum_wr[i]     += (double) s.n_wait_reset;
                    sum_main[i]   += (double) main_n / r.rounds;
                    sum_switch[i] += n_switch;
                }
            }
            fprintf(stderr, "noise %.0f %%, %s (mean of 5 seeds, 4000 rounds): rms error of the prediction of the width "
                    "taken / share of the width used most / changes of the preferred width / tok/ms / extensions / cuts:",
                    100.0 * rel, model == 0 ? "code-like" : model == 1 ? "qwen-like" : "ds-like");
            for (int i = 0; i < 3; ++i) {
                fprintf(stderr, " %s %.2f %% / %.3f / %.1f / %.4f / %.1f / %.1f;", CAP_VARIANTS[i].name,
                        100.0 * sum_sd[i] / 5, sum_main[i] / 5, sum_switch[i] / 5, sum_tps[i] / 5, sum_ext[i] / 5,
                        sum_wr[i] / 5);
            }
            fprintf(stderr, "\n");
            CHECK(sum_tps[0] > 0.98 * sum_tps[1] && sum_tps[1] > 0.98 * sum_tps[2]);
        }
    }
    return 0;
}

//
// round log (--spec-smart-log)
//

// one JSON object: balanced braces and brackets outside strings, no stray text after the object, no nan / inf
static bool json_line_ok(const std::string & l) {
    if (l.size() < 2 || l.front() != '{' || l.back() != '}') {
        return false;
    }
    std::string stack;
    bool in_str = false;
    for (size_t i = 0; i < l.size(); ++i) {
        const char c = l[i];
        if (in_str) {
            if (c == '\\') {
                i++;
            } else if (c == '"') {
                in_str = false;
            }
            continue;
        }
        if (c == '"') {
            in_str = true;
        } else if (c == '{' || c == '[') {
            stack.push_back(c);
        } else if (c == '}' || c == ']') {
            if (stack.empty() || stack.back() != (c == '}' ? '{' : '[')) {
                return false;
            }
            stack.pop_back();
            if (stack.empty() && i + 1 != l.size()) {
                return false;
            }
        } else if (c == 'n' && l.compare(i, 3, "nan") == 0) {
            return false;
        } else if (c == 'i' && l.compare(i, 3, "inf") == 0) {
            return false;
        }
    }
    return !in_str && stack.empty();
}

// the number after "key": (NAN for null or a missing key)
static double json_num(const std::string & l, const char * key) {
    const std::string k = std::string("\"") + key + "\":";
    const size_t pos = l.find(k);
    if (pos == std::string::npos || l.compare(pos + k.size(), 4, "null") == 0) {
        return NAN;
    }
    return std::strtod(l.c_str() + pos + k.size(), nullptr);
}

static std::string json_str(const std::string & l, const char * key) {
    const std::string k = std::string("\"") + key + "\":\"";
    const size_t pos = l.find(k);
    if (pos == std::string::npos) {
        return "";
    }
    const size_t e = l.find('"', pos + k.size());
    return l.substr(pos + k.size(), e - pos - k.size());
}

struct log_round_ref {
    int    k = 0;
    int    k_nat = -1;   // extension: the width the decision stopped at
    bool   ext_step = false;
    int    steps = 0;
    int    acc = -1;     // -1: not verified
    double draft_us = 0.0;
    double verify_us = -1.0;
    int    req = 0;
};

static int test_round_log() {
    const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-rounds.jsonl").string();
    std::filesystem::remove(path);

    // the same inputs into a controller without and one with the round log: every decision, every cell and the cost
    // table must be the same; cold start, extensions (short half-life) and lookahead on
    common_spec_smart_config c;
    c.n_max         = 4;
    c.half_life     = 40.0;
    c.reprobe_every = 16;
    c.lookahead_mix = 0.5;
    common_spec_smart_config cl = c;
    cl.log_path = path;

    std::vector<log_round_ref> ref;
    int64_t n_extend = 0;
    int64_t n_wr     = 0;
    int     n_decisions = 0;
    {
        common_spec_smart a(c, 1);
        common_spec_smart b(cl, 1);
        CHECK(a.log == nullptr && b.log != nullptr);

        std::mt19937 rng(1234);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        int64_t t = 0;
        bool unverified_done = false;
        const int N_REQ = 4;
        for (int rq = 0; rq < N_REQ; ++rq) {
            a.request_begin(0);
            b.request_begin(0);
            const double level = rq % 2 == 0 ? 1.0 : 1.25; // a change of level at the request boundary
            for (int r = 0; r < 150; ++r) {
                a.close_round(0, t);
                b.close_round(0, t);
                const bool ga = a.begin_round(0, 4);
                const bool gb = b.begin_round(0, 4);
                CHECK(ga == gb);
                n_decisions++;
                log_round_ref rr;
                rr.req = rq + 1;
                std::vector<float> ps;
                int k = 0;
                if (ga) {
                    for (;;) {
                        const float p = (float) (0.2 + 0.79 * u(rng));
                        ps.push_back(p);
                        const int ka = a.after_step(0, p);
                        const int kb = b.after_step(0, p);
                        CHECK(ka == kb);
                        n_decisions++;
                        const double us = 900.0 + 200.0 * u(rng);
                        a.add_draft_step(us);
                        b.add_draft_step(us);
                        rr.draft_us += us;
                        t += (int64_t) us;
                        if (ka >= 0) {
                            k = ka;
                            break;
                        }
                    }
                }
                rr.steps = (int) ps.size();
                rr.k     = k;
                rr.k_nat = a.seqs[0].k_nat;
                rr.ext_step = a.seqs[0].ext_step;
                a.drafted(0, k, t);
                b.drafted(0, k, t);
                // one draft of the third request is never verified (the slot stopped)
                const bool unverified = rq == 2 && r >= 100 && k > 0 && !unverified_done;
                unverified_done = unverified_done || unverified;
                if (k > 0 && !unverified) {
                    int acc = 0;
                    while (acc < k && u(rng) < 0.9 * ps[acc]) {
                        acc++;
                    }
                    a.accepted(0, acc);
                    b.accepted(0, acc);
                    rr.acc = acc;
                } else if (k == 0) {
                    rr.acc = 0;
                }
                const int64_t v = (int64_t) (level * (30000.0 + 7000.0 * k + 2000.0 * u(rng)));
                if (r + 1 < 150) {
                    rr.verify_us = unverified ? -1.0 : (double) v;
                }
                t += v;
                ref.push_back(rr);
            }
        }

        // all state the decisions read is the same
        CHECK(a.cost_table() == b.cost_table());
        CHECK(a.draft_step() == b.draft_step());
        CHECK(a.level() == b.level() && a.n_extend == b.n_extend && a.n_closed == b.n_closed);
        for (size_t i = 0; i < a.cal_cells.size(); ++i) {
            CHECK(a.cal_cells[i].mean == b.cal_cells[i].mean && a.cal_cells[i].n == b.cal_cells[i].n);
        }
        for (size_t i = 0; i < a.qbar_cells.size(); ++i) {
            CHECK(a.qbar_cells[i].mean == b.qbar_cells[i].mean && a.qbar_cells[i].n == b.qbar_cells[i].n);
        }
        for (size_t i = 0; i < a.shape_cells.size(); ++i) {
            CHECK(a.shape_cells[i].mean == b.shape_cells[i].mean && a.shape_cells[i].var == b.shape_cells[i].var);
        }
        CHECK(a.n_extend > 0);
        n_extend = a.n_extend;
        n_wr     = a.n_wait_reset;
        CHECK(n_wr > 0 && b.n_wait_reset == n_wr);

        // decide() with and without the record: the same result (random states of b)
        for (int i = 0; i < 2000; ++i) {
            const int n = (int) (u(rng) * 5.0);
            std::vector<float> p(n);
            for (auto & x : p) {
                x = (float) u(rng);
            }
            common_spec_smart_diag dg;
            const int nm = 1 + (int) (u(rng) * 4.0);
            const int r0 = b.decide(p.data(), n, std::max(n, nm));
            const int r1 = b.decide(p.data(), n, std::max(n, nm), &dg);
            CHECK(r0 == r1 && dg.valid && dg.k == r1 && (int) dg.q.size() == n);
        }

        // the end of a request writes the buffer (only the last, still open round is missing)
        b.log_flush();
        const std::string mid = read_file(path);
        CHECK(std::count(mid.begin(), mid.end(), '\n') == (long) (1 + N_REQ + ref.size() - 1));
    }
    fprintf(stderr, "round log: %zu rounds, %d decisions the same with and without the log, %lld extensions\n",
            ref.size(), n_decisions, (long long) n_extend);

    // the file: a start line, one line per request, one line per round in order
    std::ifstream f(path);
    std::vector<std::string> rounds;
    int n_start = 0, n_req = 0, n_extend_log = 0, n_other = 0, n_dropped = 0, n_unverified = 0, n_wr_log = 0;
    std::vector<int> seen_closed(5, 0); // the first closed round of every width is not timed
    for (std::string l; std::getline(f, l);) {
        CHECK(json_line_ok(l));
        const std::string ev = json_str(l, "ev");
        if (ev == "start") {
            n_start++;
            CHECK(json_num(l, "n_max") == 4 && l.find("sweep") == std::string::npos);
            CHECK(json_num(l, "conf_cap") == 16.0);
        } else if (ev == "req") {
            n_req++;
            CHECK(json_num(l, "req") == n_req);
        } else {
            CHECK(ev == "round");
            rounds.push_back(l);
        }
    }
    CHECK(n_start == 1 && n_req == 4 && rounds.size() == ref.size());
    for (size_t i = 0; i < rounds.size(); ++i) {
        const std::string & l  = rounds[i];
        const log_round_ref & r = ref[i];
        CHECK(json_num(l, "round") == (double) (i + 1) && json_num(l, "req") == r.req);
        CHECK(json_num(l, "k") == r.k && json_num(l, "steps") == r.steps);
        CHECK(std::fabs(json_num(l, "draft_us") - r.draft_us) < 0.5);
        const std::string forced = json_str(l, "forced");
        const std::string end    = json_str(l, "end");
        n_extend_log  += forced == "extend";
        n_other       += forced != "extend" && forced != "none";
        n_dropped     += end == "dropped";
        n_unverified  += end == "unverified";
        if (r.acc >= 0) {
            CHECK(json_num(l, "accepted") == r.acc);
        } else {
            CHECK(std::isnan(json_num(l, "accepted")));
        }
        if (r.verify_us >= 0.0) {
            CHECK(end == "closed" && json_num(l, "verify_us") == r.verify_us);
        } else {
            CHECK(end != "closed" && std::isnan(json_num(l, "verify_us")));
        }
        CHECK(l.find("\"cost_table\":[") != std::string::npos && !std::isnan(json_num(l, "cost_pred")));
        CHECK(!std::isnan(json_num(l, "exp_acc")));
        // every round has a decision: the lookahead of the next position is recorded before every step that the
        // decision took (not the extra step of an extension)
        CHECK(l.find("\"la\":{\"stop\":") != std::string::npos);
        {
            const size_t pos = l.find("\"la_next\":[");
            const size_t e   = l.find(']', pos);
            const std::string arr = l.substr(pos + 11, e - pos - 11);
            const int n_la = arr.empty() ? 0 : 1 + (int) std::count(arr.begin(), arr.end(), ',');
            CHECK(n_la == r.steps - (r.ext_step ? 1 : 0));
        }
        if (forced == "extend") {
            CHECK(r.k_nat >= 0 && r.k == r.k_nat + 1 && json_num(l, "k_nat") == r.k_nat);
            CHECK(l.find(r.ext_step ? "\"ext_step\":true" : "\"ext_step\":false") != std::string::npos);
        } else {
            CHECK(forced == "none" && r.k_nat < 0 && json_num(l, "k_nat") == r.k && l.find("\"ext_step\":false") != std::string::npos);
        }
        if (end == "closed") {
            CHECK(l.find(seen_closed[r.k] ? "\"timed\":true" : "\"timed\":false") != std::string::npos);
            seen_closed[r.k] = 1;
        }
        // an observation of a waiting width: the confidence of its old mean before and after the cut (below 4)
        if (l.find("\"wait_reset\":true") != std::string::npos) {
            n_wr_log++;
            CHECK(end == "closed" && l.find("\"timed\":true") != std::string::npos);
            const size_t pos = l.find("\"wr_n\":[");
            CHECK(pos != std::string::npos);
            double n0 = 0.0, n1 = 0.0;
            CHECK(sscanf(l.c_str() + pos + 8, "%lf,%lf", &n0, &n1) == 2);
            CHECK(n1 < n0 && n1 < 4.0 && n1 >= 0.0);
        } else {
            CHECK(l.find("\"wait_reset\":false") != std::string::npos && l.find("\"wr_n\"") == std::string::npos);
        }
    }
    CHECK(n_other == 0 && n_extend_log == n_extend && n_wr_log == n_wr);
    CHECK(n_dropped == 4 && n_unverified == 1);
    fprintf(stderr, "round log: %zu lines checked (extend %d, wait reset %d, dropped %d, unverified %d): %s\n",
            rounds.size() + 5, n_extend_log, n_wr_log, n_dropped, n_unverified, path.c_str());

    // off: no file
    const std::string path_off = (std::filesystem::temp_directory_path() / "test-spec-smart-rounds-off.jsonl").string();
    std::filesystem::remove(path_off);
    {
        common_spec_smart s(c, 1);
        s.request_begin(0);
        s.begin_round(0, 4);
        s.drafted(0, 0, 0);
        s.log_flush();
        CHECK(s.log == nullptr);
    }
    CHECK(!std::filesystem::exists(path_off));
    common_params p;
    CHECK(p.speculative.draft.smart_log.empty());
    return 0;
}

// the decode pool counts in the round log: the draft part (round start -> drafted) and the verification part (drafted ->
// close) of every closed round, the rounds since the last lend, nothing when a count is missing, and no effect on the
// decisions or the cells
// the cost decomposition: the graph builds and captures of the target and the draft context are taken out of the
// verification time, those of a draft step out of the step time, and the rounds of a decode pool lend (the lend round
// and the next lend_skip - 1) are not observed; their acceptance is. Without counts, or with both switched off, the
// observations are the measured times
static int test_cost_decomp() {
    struct result {
        std::vector<double> level, draft;
        int64_t n_obs = 0;
        double  acc_lend = 0.0; // acceptance observations made in the lend rounds
    };
    auto run = [](bool on, bool counts, result & res) {
        common_spec_smart_config c;
        c.n_max = 2;
        c.reprobe_every = 0;
        if (!on) {
            c.cost_graph = false;
            c.lend_skip  = 0;
        }
        common_spec_smart s(c, 1);
        set_identity(s, 0.95);
        set_costs(s, { 40.0, 40.0, 40.0 }, 3.0); // every width costs the same: the decision drafts 2
        common_spec_smart_pool P;
        P.valid = true;
        common_spec_smart_graphs G;
        G.tgt.valid = G.dft.valid = true;
        int64_t t = 0;
        s.request_begin(0);
        double n_acc0 = 0.0;
        for (int r = 0; r < 40; ++r) {
            s.close_round(0, t, counts ? &P : nullptr, counts ? &G : nullptr);
            res.level.push_back(s.level());
            if (r == 20) {
                n_acc0 = s.conf(s.qbar_cells[0]);
            }
            if (r == 25) {
                res.acc_lend = s.conf(s.qbar_cells[0]) - n_acc0;
            }
            if (s.begin_round(0, 2)) {
                for (int step = 0;; ++step) {
                    const int32_t k = s.after_step(0, 0.95f);
                    // a draft step of 3 ms; the first one of round 7 builds a graph of the draft context for 2 ms
                    const double g_step = r == 7 && step == 0 ? 2000.0 : 0.0;
                    s.add_draft_step(3000.0 + g_step, counts ? g_step : 0.0);
                    t += (int64_t) (3000.0 + g_step);
                    if (k >= 0) {
                        break;
                    }
                }
            }
            res.draft.push_back(s.draft_step());
            if (r == 20) {
                P.lends++; // a lend in the draft steps of round 20
            }
            s.drafted(0, (int32_t) s.seqs[0].p.size(), t, counts ? &P : nullptr, counts ? &G : nullptr);
            if (s.seqs[0].k > 0) {
                s.accepted(0, 1);
            }
            // a verification of 40 ms; in rounds 10 and 30 a build (8 ms) and a capture (15 ms) of the target add 23 ms,
            // in the lend round and the next four the fills add 30 ms
            double v = 40000.0;
            if (r == 10 || r == 30) {
                G.tgt.builds++;
                G.tgt.build_us   += 8000;
                G.tgt.captures++;
                G.tgt.capture_us += 15000;
                v += 23000.0;
            }
            if (r >= 20 && r < 25) {
                v += 30000.0;
            }
            t += (int64_t) v;
        }
        s.close_round(0, t, counts ? &P : nullptr, counts ? &G : nullptr);
        res.level.push_back(s.level());
        for (const auto x : s.cost_n_run) {
            res.n_obs += x;
        }
    };
    result on, raw, nc;
    run(true,  true,  on);
    run(false, true,  raw);
    run(true,  false, nc);
    // with the decomposition every observed verification is 40 ms: the level stays there
    for (const double x : on.level) {
        CHECK(std::fabs(x - 40000.0) < 1e-6);
    }
    // the five lend rounds are not observed, their acceptance is
    CHECK(on.n_obs == raw.n_obs - 5);
    CHECK(on.acc_lend > 3.0);
    // the step that built a graph counts 3 ms with the decomposition, 5 ms without it
    CHECK(std::fabs(on.draft.back() - 3000.0) < 1e-6);
    CHECK(raw.draft.back() > 3000.0 + 1.0);
    // the measured times move the level (the builds and the lends)
    double max_dev = 0.0;
    for (const double x : raw.level) {
        max_dev = std::max(max_dev, std::fabs(x - 40000.0));
    }
    CHECK(max_dev > 1000.0);
    // without counts the switches change nothing: the same as switched off
    CHECK(nc.level == raw.level && nc.draft == raw.draft && nc.n_obs == raw.n_obs);
    return 0;
}

static int test_round_log_pool() {
    const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-pool.jsonl").string();
    std::filesystem::remove(path);
    common_spec_smart_config c;
    c.n_max = 2;
    c.cost_graph = false; // the counts only feed the log here (the cost decomposition is test_cost_decomp)
    c.lend_skip  = 0;
    common_spec_smart_config cl = c;
    cl.log_path = path;
    {
        common_spec_smart a(c, 1);
        common_spec_smart b(cl, 1);
        common_spec_smart_pool P;
        P.valid = true;
        common_spec_smart_graphs G; // the target builds a graph of 2 rows in the verification of round 1, captured
        G.tgt.valid = G.dft.valid = true;
        G.tgt.backend = true;
        int64_t t = 0;
        a.request_begin(0);
        b.request_begin(0);
        for (int r = 0; r < 5; ++r) {
            a.close_round(0, t);
            b.close_round(0, t, &P, &G);
            CHECK(a.begin_round(0, 2) == b.begin_round(0, 2));
            t += 1000;
            if (r == 0) {
                P.lends = 1; // a lend in the draft steps, and its first fills
                P.fills = 2; P.fill_bytes = 2 << 20;
            }
            a.drafted(0, 0, t);
            b.drafted(0, 0, t, r == 2 ? nullptr : &P, &G); // round 2: the pool count at drafted is missing
            G.dft.calls++;
            G.tgt.calls++;
            if (r == 1) {
                G.tgt.builds++; G.tgt.build_us += 1500; G.tgt.last_nt = 2; G.tgt.captures++; G.tgt.capture_us += 2500;
            }
            const int64_t v = 40000 + 1000 * r;
            if (r == 0) {
                P.fills += 7; P.fill_bytes += 7 << 20; P.hits += 3;
            } else {
                P.fills += 1; P.fill_bytes += 1 << 20; P.hits += 10;
            }
            t += v;
        }
        a.close_round(0, t);
        b.close_round(0, t, &P, &G);
        CHECK(a.cost_table() == b.cost_table() && a.level() == b.level() && a.n_closed == b.n_closed);
    }
    std::vector<std::string> lines;
    {
        std::istringstream in(read_file(path));
        for (std::string l; std::getline(in, l);) {
            if (l.find("\"ev\":\"round\"") != std::string::npos && l.find("\"end\":\"closed\"") != std::string::npos) {
                lines.push_back(l);
            }
        }
    }
    CHECK(lines.size() == 5);
    CHECK(lines[0].find("\"pool\":{\"draft\":{\"fills\":2,\"fill_mib\":2.0,\"hits\":0,\"lends\":1,\"reclaims\":0},"
                        "\"verify\":{\"fills\":7,\"fill_mib\":7.0,\"hits\":3,\"lends\":0,\"reclaims\":0},"
                        "\"since_lend\":0,\"lend\":1}") != std::string::npos);
    CHECK(lines[1].find("\"verify\":{\"fills\":1,\"fill_mib\":1.0,\"hits\":10,") != std::string::npos);
    CHECK(lines[1].find("\"since_lend\":1,") != std::string::npos);
    CHECK(lines[2].find("\"pool\"") == std::string::npos); // the count at drafted was missing
    CHECK(lines[3].find("\"draft\":{\"fills\":0,") != std::string::npos);
    CHECK(lines[3].find("\"since_lend\":3,") != std::string::npos);
    CHECK(lines[4].find("\"since_lend\":4,") != std::string::npos);
    // graphs: the draft part saw none, the verification of round 1 a build of 2 rows and a capture
    CHECK(lines[0].find("\"graph\":{\"draft\":{\"tgt\":{\"calls\":0,\"builds\":0,\"build_ms\":0.000,\"nt\":null,"
                        "\"captures\":0,\"capture_ms\":0.000,\"direct\":0,\"direct_ms\":0.000},\"dft\":{\"calls\":0,"
                        "\"builds\":0,\"build_ms\":0.000,\"nt\":null}},\"verify\":{\"tgt\":{\"calls\":1,") != std::string::npos);
    CHECK(lines[1].find("\"verify\":{\"tgt\":{\"calls\":1,\"builds\":1,\"build_ms\":1.500,\"nt\":2,\"extent_mib\":0.00,\"pos\":-1,\"captures\":1,"
                        "\"capture_ms\":2.500,") != std::string::npos);
    CHECK(lines[2].find("\"graph\"") != std::string::npos); // the graph counts do not depend on the pool counts
    std::filesystem::remove(path);
    return 0;
}

static int test_off() {
    // the defaults; an instance without draft-mtp never gets a controller, and every hook is a no-op
    common_params p;
    CHECK(p.speculative.draft.smart && !p.speculative.draft.smart_on && p.speculative.draft.smart_store.empty());
    CHECK(p.speculative.draft.smart_half_life == 250.0f && p.speculative.draft.smart_n_max == 7);
    CHECK(p.speculative.draft.n_max == 3); // the single-argument default is unchanged
    {
        common_spec_smart_config c;
        CHECK(c.half_life == 250.0 && c.reprobe_n == 4.0 && c.reprobe_every == 16);
        CHECK(c.cost_graph && c.lend_skip == 5);
        CHECK(c.level_half_life == 64.0 && c.level_cap_load == 2.0 && c.cap_load == 32.0 && c.conf_cap == 16.0);
        CHECK(c.lookahead_mix == 0.5);
    }
    CHECK(common_speculative_smart_summary(nullptr, 0).empty());
    common_speculative_smart_accepted(nullptr, 0, 3);

    for (bool smart : { false, true }) {
        common_params_speculative sp;
        sp.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
        sp.draft.smart = smart;
        common_speculative_smart_resolve(sp);
        CHECK(!sp.draft.smart_on && sp.draft.n_max == 3);
        sp.draft.smart_on = smart; // even when forced, no draft-mtp instance: no controller
        common_speculative * spec = common_speculative_init(sp, 1);
        CHECK(spec != nullptr);
        common_speculative_begin(spec, 0, { 1, 2, 3 });
        common_speculative_smart_accepted(spec, 0, 2);
        CHECK(common_speculative_smart_summary(spec, 0).empty());
        common_speculative_free(spec);
    }
    return 0;
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

// parse a llama-server command line (no model is loaded) and resolve --spec-smart as the server does
static bool parse_resolve(std::vector<std::string> args, common_params & p) {
    args.insert(args.begin(), "llama-server");
    std::vector<char *> argv;
    for (auto & a : args) {
        argv.push_back(a.data());
    }
    p = common_params();
    if (!common_params_parse((int) argv.size(), argv.data(), p, LLAMA_EXAMPLE_SERVER)) {
        return false;
    }
    common_speculative_smart_resolve(p.speculative);
    return true;
}

static int test_resolve() {
    // the option rules: on for draft-mtp by default with the bound 7; --spec-draft-n-max is the bound when given; a
    // threshold given on the command line or in the environment turns it off (with a warning), also against an
    // explicit --spec-smart; --no-spec-smart turns it off
    common_params p;
    const std::vector<std::string> mtp = { "--spec-type", "draft-mtp" };
    auto with = [&](std::vector<std::string> extra) {
        std::vector<std::string> a = mtp;
        a.insert(a.end(), extra.begin(), extra.end());
        return a;
    };

    CHECK(parse_resolve(with({}), p));
    CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 7 && !p.speculative.draft.n_max_explicit);
    CHECK(p.speculative.need_n_rs_seq() == 7 && common_speculative_n_max(&p.speculative) == 7);

    CHECK(parse_resolve(with({ "--spec-draft-n-max", "4" }), p));
    CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 4);
    CHECK(parse_resolve(with({ "--spec-draft-n-max", "15", "--spec-smart-half-life", "1000" }), p));
    CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 15 && p.speculative.draft.smart_half_life == 1000.0f);

    CHECK(parse_resolve(with({ "--no-spec-smart" }), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.n_max == 3);
    CHECK(parse_resolve(with({ "--no-spec-smart", "--spec-draft-n-max", "2" }), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.n_max == 2);

    CHECK(parse_resolve(with({ "--spec-draft-p-min", "0" }), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.p_min_explicit && p.speculative.draft.n_max == 3);
    CHECK(parse_resolve(with({ "--spec-draft-p-continue", "0.9", "--spec-draft-n-max", "15" }), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.n_max == 15);
    CHECK(parse_resolve(with({ "--spec-smart", "--spec-draft-p-min", "0.33,0.6" }), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.smart_explicit);

    // the environment counts as given
    set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN", "0.5");
    CHECK(parse_resolve(with({}), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.p_min_explicit);
    set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN", nullptr);
    set_env("LLAMA_ARG_SPEC_DRAFT_P_CONTINUE", "0.9");
    CHECK(parse_resolve(with({}), p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.p_continue_explicit);
    set_env("LLAMA_ARG_SPEC_DRAFT_P_CONTINUE", nullptr);
    set_env("LLAMA_ARG_SPEC_SMART", "0");
    CHECK(parse_resolve(with({}), p));
    CHECK(!p.speculative.draft.smart_on);
    set_env("LLAMA_ARG_SPEC_SMART", nullptr);
    set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX", "5");
    CHECK(parse_resolve(with({}), p));
    CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 5);
    set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX", nullptr);

    // another speculative type: off, n-max untouched
    CHECK(parse_resolve({ "--spec-type", "ngram-simple" }, p));
    CHECK(!p.speculative.draft.smart_on && p.speculative.draft.n_max == 3);

    // the removed options are rejected
    for (const char * o : { "--spec-smart-cost-lcb", "--spec-smart-level-half-life", "--spec-smart-reprobe-n",
                            "--spec-smart-reprobe-every", "--spec-smart-conf-cap", "--spec-smart-lookahead-mix" }) {
        CHECK(!parse_resolve(with({ o, "1" }), p));
    }
    return 0;
}

//
// trigram confidence
//

// the trigram counts of a stream, counted directly
static void ref_counts(const std::vector<int32_t> & s, int32_t a, int32_t b, int32_t tok, int32_t & cnt, int32_t & N) {
    cnt = 0;
    N   = 0;
    for (size_t i = 2; i < s.size(); ++i) {
        if (s[i - 2] == a && s[i - 1] == b) {
            N++;
            cnt += s[i] == tok;
        }
    }
}

// a top-1 calibrator written from its definition: logistic of the drafter top probability and the trigram counts,
// learnt by AdaGrad (its sum of squared gradients halving every 4096 updates) from the acceptance of the drafted token
// alone
struct ref_top1 {
    static constexpr int D = common_spec_tconf::DIM;
    double b[D], G[D];
    static double lg(double x) {
        x = std::min(std::max(x, 1e-4), 1.0 - 1e-4);
        return std::log(x / (1.0 - x));
    }
    // the request's counts of the other roles (RU): no role markers and no prompt length given
    void x_of(double p, int cnt, int N, bool mid, bool sw, double * x) const {
        const double g = N > 0 ? N / (N + 3.0) : 0.0;
        for (int d = 0; d < D; ++d) {
            x[d] = 0.0;
        }
        x[0]  = 1.0;
        x[1]  = lg(p);
        x[4]  = N > 0 ? g * lg((cnt + 0.5) / (N + 1.0)) : 0.0;
        x[5]  = g;
        x[12] = mid;
        x[13] = sw;
    }
    double q(const double * x) const {
        double z = 0.0;
        for (int d = 0; d < D; ++d) {
            z += b[d] * x[d];
        }
        return 1.0 / (1.0 + std::exp(-std::min(std::max(z, -30.0), 30.0)));
    }
    void learn(const double * x, double y) {
        const double r = y - q(x);
        for (int d = 0; d < D; ++d) {
            const double g = x[d] * r;
            G[d] = G[d] * std::exp2(-1.0 / 4096.0) + g * g;
            b[d] += 0.15 * g / std::sqrt(G[d]);
        }
    }
};

// the features of a request's counts: RA and RU
static void tc_features(double p, int cA, int NA, int cU, int NU, bool mid, bool sw, double * x) {
    const double n[common_spec_tconf::N_SRC] = { (double) cA, (double) cU, 0, 0, 0 };
    const double N[common_spec_tconf::N_SRC] = { (double) NA, (double) NU, 0, 0, 0 };
    common_spec_tconf::features(p, n, N, mid, sw, x);
}

static int test_tconf() {
    // 1. the features and the static coefficients against the offline fit (tconf_fit2.py reference points, k = 1)
    {
        struct pt { double p; int cA, NA, cU, NU, mid, sw; double q; };
        const pt R[] = {
            { 0.4, 4, 10, 0, 0, 0, 1, 0.2305283658606285 },  { 0.3, 0, 0, 9, 10, 0, 1, 0.5416871315347813 },
            { 0.9, 1, 5, 0, 20, 1, 0, 0.6573225453247713 },  { 0.05, 0, 0, 0, 0, 0, 0, 0.04846946976832208 },
            { 0.99, 3, 3, 2, 2, 1, 1, 0.992394076381946 },
        };
        for (const auto & r : R) {
            common_spec_tconf m;
            m.reset(1);
            double x[common_spec_tconf::DIM];
            tc_features(r.p, r.cA, r.NA, r.cU, r.NU, r.mid != 0, r.sw != 0, x);
            CHECK(std::fabs(m.q(x) - r.q) < 1e-5); // the table keeps 6 decimals
        }
        // the profile columns start at 0 for every k
        for (int k = 1; k <= common_spec_tconf::K_MAX; ++k) {
            common_spec_tconf m;
            m.reset(k);
            for (int d = 6; d < 12; ++d) {
                CHECK(m.b[d] == 0.0);
            }
        }
    }

    // 2. the trigram table: incremental insertion = counting the whole stream; a stream that does not continue it
    //    is inserted anew
    {
        std::mt19937 rng(7);
        std::vector<int32_t> s;
        common_spec_trigram tri;
        for (int step = 0; step < 300; ++step) {
            const int n_new = 1 + (int) (rng() % 4);
            for (int i = 0; i < n_new; ++i) {
                s.push_back((int32_t) (rng() % 6));
            }
            tri.sync(s.data(), s.size() - 1, s.back()); // context + id_last
            CHECK(tri.size() == s.size());
            if (step % 37 == 0) {
                for (int32_t a = 0; a < 6; ++a) {
                    for (int32_t b = 0; b < 6; ++b) {
                        int32_t toks[6] = { 0, 1, 2, 3, 4, 5 };
                        int32_t cnt[12], N[2];
                        tri.lookup(a, b, toks, 6, cnt, N);
                        for (int32_t t = 0; t < 6; ++t) {
                            int32_t rc, rN;
                            ref_counts(s, a, b, t, rc, rN);
                            CHECK(rc == cnt[2 * t] && rN == N[0] && cnt[2 * t + 1] == 0 && N[1] == 0); // no roles: other
                        }
                    }
                }
            }
        }
        // a rollback: the stream is cut and continues differently
        s.resize(s.size() - 5);
        s.push_back(5);
        s.push_back(5);
        tri.sync(s.data(), s.size() - 1, s.back());
        int32_t toks[1] = { 5 };
        int32_t cnt[2], N[2], rc, rN;
        tri.lookup(5, 5, toks, 1, cnt, N);
        ref_counts(s, 5, 5, 5, rc, rN);
        CHECK(cnt[0] == rc && N[0] == rN);
    }

    // 2b. roles: the markers (100 opens the assistant, 103 104 and 101 another role; the control token 102 another
    //     role too) and the generated tokens; incremental insertion counts by the roles of the whole stream
    {
        common_spec_roles roles;
        roles.markers.push_back({ { 100 }, common_spec_roles::ASSIST });
        roles.markers.push_back({ { 103, 104 }, common_spec_roles::OTHER }); // a two-token marker
        roles.markers.push_back({ { 101 }, common_spec_roles::OTHER });
        roles.ctrl.assign(110, 0);
        roles.ctrl[100] = roles.ctrl[101] = roles.ctrl[102] = roles.ctrl[103] = 1;
        const std::vector<int32_t> s = { 101, 1, 2, 3, 100, 1, 2, 4, 102, 1, 2, 5, 100, 1, 2, 6, 103, 104, 1, 2, 7, 100, 1, 2 };
        const size_t gen_from = 22; // the last two tokens are generated
        const auto r = roles.of(s.data(), s.size(), gen_from);
        const std::vector<uint8_t> want = { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1 };
        CHECK(r == want);
        common_spec_trigram tri;
        for (size_t n = 3; n <= s.size(); n += 5) {
            tri.sync(s.data(), n - 1, s[n - 1], &roles, gen_from);
        }
        tri.sync(s.data(), s.size() - 1, s.back(), &roles, gen_from);
        const int32_t toks[5] = { 3, 4, 5, 6, 7 };
        int32_t cnt[10], N[2];
        tri.lookup(1, 2, toks, 5, cnt, N);
        CHECK(N[common_spec_roles::ASSIST] == 2 && N[common_spec_roles::OTHER] == 3); // after (1, 2): 4, 6 / 3, 5, 7
        CHECK(cnt[2 * 1 + common_spec_roles::ASSIST] == 1 && cnt[2 * 2 + common_spec_roles::OTHER] == 1 &&
              cnt[2 * 4 + common_spec_roles::OTHER] == 1 && cnt[2 * 3 + common_spec_roles::ASSIST] == 1 &&
              cnt[2 * 0 + common_spec_roles::OTHER] == 1 && cnt[2 * 1 + common_spec_roles::OTHER] == 0);
        // without markers: the prompt is other, the generated tokens the assistant
        common_spec_roles none;
        std::vector<uint8_t> want2(s.size(), 0);
        want2[22] = want2[23] = 1;
        CHECK(none.of(s.data(), s.size(), gen_from) == want2);
    }

    // a vocabulary of 40 tokens: token t has the byte 'a' + t (t < 26, inside a word; t % 5 == 0 starts a word),
    // 30 = the first two bytes of a three-byte UTF-8 sequence, 31 = its last byte
    std::vector<uint32_t> tail(40, 0);
    std::vector<uint8_t>  wf(40, 0);
    for (int t = 0; t < 26; ++t) {
        tail[t] = (1u << 24) | (uint32_t) ('a' + t);
        wf[t]   = t % 5 == 0;
    }
    tail[30] = (2u << 24) | (0xEAu << 8) | 0xB0u; // last byte 0xB0, before it 0xEA
    tail[31] = (1u << 24) | 0x80u;

    // 3. the UTF-8 state of a context
    {
        common_spec_smart_config c = base_cfg(3);
        c.tconf = true;
        common_spec_smart s(c, 1);
        s.tconf_vocab(tail, wf);
        CHECK(!s.tconf_mid({ 1, 2 }));
        CHECK(s.tconf_mid({ 1, 30 }));      // 0xEA 0xB0: one byte missing
        CHECK(!s.tconf_mid({ 1, 30, 31 })); // complete
        CHECK(!s.tconf_mid({}));
    }

    // 4. k = 1 is the top-1 calibrator: the same decisions and coefficients as ref_top1, whether the drafter gives one
    //    candidate or ten (the others are neither scored nor learnt)
    {
        int64_t updates[2] = { 0, 0 };
        std::vector<double> qs[2];
        double b_end[2][common_spec_tconf::DIM];
        for (int variant = 0; variant < 2; ++variant) {
            common_spec_smart_config c = base_cfg(3);
            c.tconf   = true;
            c.tconf_k = 1;
            common_spec_smart s(c, 1);
            s.tconf_vocab(tail, wf);
            set_identity(s, 0.6);
            set_costs(s, { 30, 33, 36, 39 }, 2.0);

            ref_top1 ref;
            for (int d = 0; d < common_spec_tconf::DIM; ++d) {
                ref.b[d] = s.tconf_model.b[d];
                ref.G[d] = 1e-3;
            }

            std::mt19937 rng(11);
            std::vector<int32_t> ctx = { 1, 2, 3, 4 };
            s.request_begin(0);
            int64_t t = 0;
            for (int r = 0; r < 200; ++r) {
                const int32_t id_last = (int32_t) (rng() % 26);
                const int32_t pos0    = (int32_t) ctx.size();
                s.close_round(0, t);
                s.tconf_begin(0, ctx.data(), ctx.size(), id_last, pos0);
                std::vector<int32_t> full = ctx;
                full.push_back(id_last);
                const std::vector<int32_t> known = full; // the trigram table: the context and id_last, not the draft
                std::vector<double> rx; // reference rows of the drafted positions
                int k = 0;
                if (s.begin_round(0, 3)) {
                    for (;;) {
                        // ten candidates, top one first; the target token is one of them or none
                        int32_t ids[10];
                        float   p[10];
                        float   rest = 1.0f;
                        for (int i = 0; i < 10; ++i) {
                            ids[i] = (int32_t) ((rng() % 26 + i * 3) % 26);
                            p[i]   = i < 9 ? rest * (0.3f + 0.6f * (float) (rng() % 100) / 100.0f) : rest;
                            rest  -= i < 9 ? p[i] : 0.0f;
                        }
                        std::sort(p, p + 10, [](float a, float b) { return a > b; });
                        double q = -1.0;
                        const int32_t i_c = s.tconf_pick(0, ids, p, variant == 0 ? 1 : 10, -1, &q);
                        CHECK(i_c == 0);
                        int32_t cnt, N;
                        ref_counts(known, full[full.size() - 2], full.back(), ids[0], cnt, N);
                        const bool mid = s.tconf_mid(std::vector<int32_t>(full.end() - std::min<size_t>(6, full.size()), full.end()));
                        double x[common_spec_tconf::DIM];
                        ref.x_of(p[0], cnt, N, mid, wf[ids[0]] != 0, x);
                        CHECK(std::fabs(ref.q(x) - q) < 1e-12);
                        rx.insert(rx.end(), x, x + common_spec_tconf::DIM);
                        qs[variant].push_back(q);
                        full.push_back(ids[0]);
                        s.add_draft_step(2000.0);
                        const int32_t n = s.after_step(0, p[0], q);
                        if (n >= 0) {
                            k = n;
                            break;
                        }
                    }
                }
                s.drafted(0, k, t);
                const int a = k > 0 ? (int) (rng() % (k + 1)) : 0;
                if (k > 0) {
                    s.accepted(0, a);
                    for (int j = 0; j <= std::min(a, k - 1); ++j) {
                        ref.learn(rx.data() + (size_t) j * common_spec_tconf::DIM, j < a ? 1.0 : 0.0);
                    }
                }
                for (int d = 0; d < common_spec_tconf::DIM; ++d) {
                    CHECK(std::fabs(ref.b[d] - s.tconf_model.b[d]) < 1e-12); // learnt at once, no pending position
                }
                CHECK(!s.seqs[0].tc_pend_valid);
                // the context grows by id_last and the accepted tokens; the target token follows them
                ctx.push_back(id_last);
                for (int j = 0; j < a; ++j) {
                    ctx.push_back(full[ctx.size()]);
                }
                t += 40000;
            }
            updates[variant] = s.tc_updates;
            std::copy(s.tconf_model.b, s.tconf_model.b + common_spec_tconf::DIM, b_end[variant]);
            CHECK(s.tc_invert == 0);
            CHECK(s.request_summary(0).find("trigram confidence k 1") != std::string::npos);
        }
        CHECK(updates[0] == updates[1] && updates[0] > 50);
        CHECK(qs[0] == qs[1]);
        for (int d = 0; d < common_spec_tconf::DIM; ++d) {
            CHECK(b_end[0][d] == b_end[1][d]);
        }
    }

    // 5. k = 3: the candidate the trigram counts back is drafted; the rejected position is learnt at the next draft
    //    with its target token, and dropped when the next draft does not continue right after it
    {
        common_spec_smart_config c = base_cfg(3);
        c.tconf   = true;
        c.tconf_k = 3;
        common_spec_smart s(c, 1);
        s.tconf_vocab(tail, wf);
        set_identity(s, 0.6);
        set_costs(s, { 30, 31, 32, 33 }, 0.5);
        // the context repeats (7, 8) -> 9 ten times
        std::vector<int32_t> ctx;
        for (int i = 0; i < 10; ++i) {
            ctx.insert(ctx.end(), { 7, 8, 9 });
        }
        ctx.push_back(7);
        s.request_begin(0);
        s.close_round(0, 0);
        s.tconf_begin(0, ctx.data(), ctx.size(), 8, (int32_t) ctx.size());
        CHECK(s.begin_round(0, 3));
        const int32_t ids[3] = { 1, 2, 9 };
        const float   p[3]   = { 0.45f, 0.30f, 0.25f };
        double q = -1.0;
        CHECK(s.tconf_pick(0, ids, p, 3, -1, &q) == 2); // 9 after (7, 8): every time
        CHECK(q > 0.6 && s.tc_invert == 1 && s.seqs[0].tc_pos[0].N == 10);
        CHECK(s.seqs[0].tc_ctx.back() == 9);
        // a fixed (sampled) token stays
        double q2 = -1.0;
        CHECK(s.tconf_pick(0, ids, p, 3, 0, &q2) == 0);
        s.after_step(0, p[2], q);
        s.drafted(0, 1, 1000);
        s.accepted(0, 0); // rejected: its labels wait for the target token
        CHECK(s.tc_updates == 0 && s.seqs[0].tc_pend_valid && s.seqs[0].tc_pend_at == (int32_t) ctx.size() + 1);
        double b0[common_spec_tconf::DIM];
        std::copy(s.tconf_model.b, s.tconf_model.b + common_spec_tconf::DIM, b0);
        // the next draft continues right after the rejected position: the target token was candidate 1 (2)
        std::vector<int32_t> ctx2 = ctx;
        ctx2.push_back(8);
        s.close_round(0, 40000);
        s.tconf_begin(0, ctx2.data(), ctx2.size(), 2, (int32_t) ctx2.size());
        CHECK(s.tc_updates == 1 && !s.seqs[0].tc_pend_valid);
        {
            // the same step by hand: three rows, labels 0 1 0, gradient / sqrt(3)
            common_spec_tconf m;
            m.reset(3);
            const auto & ps = s.seqs[0].tc_pos; // cleared by tconf_begin
            CHECK(ps.empty());
            constexpr int D = common_spec_tconf::DIM;
            double X[3 * D];
            tc_features(p[0], 0, 0, 0, 10, false, wf[1] != 0, X);
            tc_features(p[1], 0, 0, 0, 10, false, wf[2] != 0, X + D);
            tc_features(p[2], 0, 0, 10, 10, false, wf[9] != 0, X + 2 * D);
            const double lab[3] = { 0, 1, 0 };
            m.update(X, lab, 3);
            for (int d = 0; d < D; ++d) {
                // the columns that are 0 in every row (RA, the profile, mid, sw) keep their value
                const bool moved = d < 2 || d == 4 || d == 5;
                CHECK(std::fabs(m.b[d] - s.tconf_model.b[d]) < 1e-12 && (moved ? m.b[d] != b0[d] : m.b[d] == b0[d]));
            }
        }
        // a rejected position whose next draft starts elsewhere (a new request, a reused draft) is not learnt
        CHECK(s.begin_round(0, 3));
        s.tconf_pick(0, ids, p, 3, -1, &q);
        s.after_step(0, p[2], q);
        s.drafted(0, 1, 50000);
        s.accepted(0, 0);
        CHECK(s.seqs[0].tc_pend_valid);
        s.close_round(0, 90000);
        s.tconf_begin(0, ctx2.data(), ctx2.size(), 2, (int32_t) ctx2.size() + 5);
        CHECK(s.tc_updates == 1 && !s.seqs[0].tc_pend_valid);
    }

    // 6. off: the hooks do nothing and the decision reads the calibration cells
    {
        common_spec_smart_config c = base_cfg(3);
        CHECK(!c.tconf && c.tconf_k == 1);
        common_spec_smart s(c, 1);
        const int32_t ctx[4] = { 1, 2, 3, 4 };
        s.tconf_begin(0, ctx, 4, 5, 4);
        CHECK(s.seqs[0].tri.size() == 0 && s.request_summary(0).empty());
    }

    // 7. host cost: a 100k-token prompt inserted once, then one draft step (k = 1, 3, 10 candidates)
    {
        common_spec_smart_config c = base_cfg(3);
        c.tconf = true;
        std::mt19937 rng(3);
        std::vector<int32_t> ctx(100000);
        for (auto & x : ctx) {
            x = (int32_t) (rng() % 2000); // a small vocabulary: long lists after every context
        }
        for (int k : { 1, 3, 10 }) {
            c.tconf_k = k;
            common_spec_smart s(c, 1);
            s.tconf_vocab(std::vector<uint32_t>(2000, (1u << 24) | 'a'), std::vector<uint8_t>(2000, 0));
            const int64_t t0 = ggml_time_us();
            s.tconf_begin(0, ctx.data(), ctx.size(), 7, (int32_t) ctx.size());
            const int64_t t1 = ggml_time_us();
            int32_t ids[10];
            float   p[10];
            for (int i = 0; i < 10; ++i) {
                ids[i] = (int32_t) (rng() % 2000);
                p[i]   = 0.1f;
            }
            const int n_it = 20000;
            double acc = 0.0;
            for (int it = 0; it < n_it; ++it) {
                double q;
                s.seqs[0].tc_ctx.resize(8); // the same context every time
                s.seqs[0].tc_pos.clear();
                acc += s.tconf_pick(0, ids, p, 10, -1, &q) + q;
            }
            const int64_t t2 = ggml_time_us();
            fprintf(stderr, "tconf k %2d: insert 100k tokens %.1f ms, one step %.2f us (%g)\n", k, (t1 - t0) / 1000.0,
                    (double) (t2 - t1) / n_it, acc);
        }
    }
    return 0;
}

// the trigram profile (common_spec_tprof): a toy vocabulary of words (tokens 0..NW-1, " w<t>") and control tokens
namespace tpt {
constexpr int32_t GMASK = 200, SYS = 201, USER = 202, AST = 203, THINK = 204, ETHINK = 205, EOS = 206, NV = 210;

static common_spec_tprof_vocab vocab(int32_t n_words = 200) {
    common_spec_tprof_vocab v;
    const int32_t nv = std::max(NV, n_words);
    v.piece.resize(nv);
    v.special.assign(nv, 0);
    for (int32_t t = 0; t < nv; ++t) {
        if (t >= 200 && t < NV) {
            v.piece[t]   = "<s" + std::to_string(t) + ">";
            v.special[t] = 1;
        } else {
            v.piece[t] = " w" + std::to_string(t);
        }
    }
    return v;
}

static common_spec_roles roles() {
    common_spec_roles r;
    r.markers.push_back({ { SYS }, common_spec_roles::OTHER });
    r.markers.push_back({ { USER }, common_spec_roles::OTHER });
    r.markers.push_back({ { AST }, common_spec_roles::ASSIST });
    r.ctrl.assign(NV, 0);
    r.ctrl[GMASK] = r.ctrl[SYS] = r.ctrl[USER] = r.ctrl[AST] = r.ctrl[EOS] = 1;
    return r;
}

struct msg {
    int                  role; // 0 system, 1 user, 2 assistant
    std::vector<int32_t> toks;
};

// the template: [gMASK] then per message its marker (an assistant message also <think></think>) and its tokens, then
// the generation prompt <|assistant|><think>
static std::vector<int32_t> render(const std::vector<msg> & ms) {
    std::vector<int32_t> out = { GMASK };
    for (const auto & m : ms) {
        out.push_back(m.role == 0 ? SYS : m.role == 1 ? USER : AST);
        if (m.role == 2) {
            out.push_back(THINK);
            out.push_back(ETHINK);
        }
        out.insert(out.end(), m.toks.begin(), m.toks.end());
    }
    out.push_back(AST);
    out.push_back(THINK);
    return out;
}

static std::vector<int32_t> words(std::mt19937 & rng, int n, int32_t nw = 200) {
    std::vector<int32_t> w(n);
    for (auto & x : w) {
        x = (int32_t) (rng() % nw);
    }
    return w;
}

// the generated tokens of an answer: </think>, the answer, the end token
static std::vector<int32_t> gen(const std::vector<int32_t> & a) {
    std::vector<int32_t> g = { ETHINK };
    g.insert(g.end(), a.begin(), a.end());
    g.push_back(EOS);
    return g;
}

static std::string tmp(const char * name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

static void rm(const std::string & p) {
    std::error_code ec;
    for (const char * suf : { "", ".bak", ".tmp", ".lock" }) {
        std::filesystem::remove(p + suf, ec);
    }
}
} // namespace tpt

static int test_tprof() {
    using namespace tpt;
    const auto R = roles();

    // 1. content-defined chunks: white space does not change them, an insertion changes only the chunks around it
    {
        common_spec_tprof_config c;
        c.path = tmp("test-spec-tprof-chunks.bin");
        common_spec_tprof p(c, vocab());
        std::mt19937 rng(1);
        const auto a = words(rng, 300);
        const auto ia = p.chunk_ids(a.data(), a.size());
        CHECK(ia.size() >= 10);
        // an insertion of 5 words at 150: the chunks before it and all but a few after it are the same
        auto b = a;
        b.insert(b.begin() + 150, { 1, 2, 3, 4, 5 });
        const auto ib = p.chunk_ids(b.data(), b.size());
        size_t pre = 0;
        while (pre < ia.size() && pre < ib.size() && ia[pre] == ib[pre]) {
            pre++;
        }
        size_t suf = 0;
        while (suf < ia.size() && suf < ib.size() && ia[ia.size() - 1 - suf] == ib[ib.size() - 1 - suf]) {
            suf++;
        }
        CHECK(pre + suf + 3 >= ia.size() && pre >= 3 && suf >= 3);
        // the same text with other white space: the pieces of a vocabulary where word t is "w<t>  \n"
        auto v2 = vocab();
        for (int32_t t = 0; t < 200; ++t) {
            v2.piece[t] = "w" + std::to_string(t) + ((t % 3) ? "  " : "\n");
        }
        common_spec_tprof p2(c, v2);
        CHECK(p2.chunk_ids(a.data(), a.size()) == ia);
    }

    // 2. a conversation: what is counted, once, and from which source
    {
        common_spec_tprof_config c;
        c.path      = tmp("test-spec-tprof-conv.bin");
        c.half_life = 131072.0;
        rm(c.path);
        std::mt19937 rng(2);
        const auto S  = words(rng, 120);
        const auto U1 = words(rng, 80);
        const auto A1 = words(rng, 200);
        const auto U2 = words(rng, 70);
        const auto A2 = words(rng, 180);
        const auto A2b = words(rng, 160);
        const auto U3 = words(rng, 60);
        const auto A3 = words(rng, 150);
        const auto U4 = words(rng, 50);
        const auto A4 = words(rng, 150);
        const auto U5 = words(rng, 40);
        const auto L  = words(rng, 90);
        const auto U6 = words(rng, 30);
        const auto U7 = words(rng, 35);
        {
            common_spec_tprof p(c, vocab());
            std::string why;
            CHECK(p.open(&why) && !p.read_only());

            // request 1: everything is new
            auto pr = render({ { 0, S }, { 1, U1 } });
            auto g  = gen(A1);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            auto st = p.stats();
            // ([gMASK] opens the prompt: no content, no trigram of its own)
            CHECK(st.tok_u == (double) pr.size() - 1 && st.tok_x == 0 && st.tok_known == 1 && st.tok_m == (double) g.size());
            CHECK(st.regen_m == 1 && st.weight == 1.0);
            // the clock (generated tokens) moved by the answer after it was counted
            double clk = (double) g.size();
            auto dec = [&](double t_add) { return std::exp2(-(clk - t_add) / c.half_life); };
            CHECK(std::fabs(p.weight(A1[10], A1[11], A1[12], common_spec_tprof::SRC_M) - dec(0)) < 1e-6);
            CHECK(std::fabs(p.weight(U1[10], U1[11], U1[12], common_spec_tprof::SRC_U) - dec(0)) < 1e-6);

            // request 2: the history is known (the answer as the template renders it); only the new user turn counts
            pr = render({ { 0, S }, { 1, U1 }, { 2, A1 }, { 1, U2 } });
            g  = gen(A2);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_u == (double) U2.size() + 3 && st.tok_x == 0 && st.tok_absorbed == 0);
            CHECK(st.tok_known == (double) pr.size() - st.tok_u);
            // the counts of request 1 decayed by its generated tokens; nothing was counted twice
            clk += (double) g.size();
            CHECK(std::fabs(p.weight(A1[10], A1[11], A1[12], common_spec_tprof::SRC_M) - dec(0)) < 1e-5);
            CHECK(p.weight(A1[10], A1[11], A1[12], common_spec_tprof::SRC_X) == 0 && p.weight(A1[10], A1[11], A1[12], common_spec_tprof::SRC_U) == 0);
            CHECK(std::fabs(p.weight(U1[10], U1[11], U1[12], common_spec_tprof::SRC_U) - dec(0)) < 1e-5);

            // request 3: a swipe of request 2: nothing of the prompt counts, the answer weighs 1 / (1 + 0.5)
            g = gen(A2b);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_u == 0 && st.tok_x == 0 && st.tok_known == (double) pr.size());
            CHECK(st.regen_m == 2 && std::fabs(st.weight - 1.0 / 1.5) < 1e-12);
            const double clk3 = clk;
            clk += (double) g.size();
            CHECK(std::fabs(p.weight(A2b[10], A2b[11], A2b[12], common_spec_tprof::SRC_M) - dec(clk3) / 1.5) < 1e-5);

            // request 4: the user edited the middle of A1: only the chunks around the edit count, as U
            auto A1e = A1;
            for (int i = 100; i < 106; ++i) {
                A1e[i] = (int32_t) (rng() % 200);
            }
            pr = render({ { 0, S }, { 1, U1 }, { 2, A1e }, { 1, U2 }, { 2, A2 }, { 1, U3 } });
            g  = gen(A3);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_x == 0 && st.tok_u > (double) U3.size() + 3 + 5 && st.tok_u < (double) U3.size() + 3 + 6 + 120);
            CHECK(p.weight(A1e[101], A1e[102], A1e[103], common_spec_tprof::SRC_U) > 0.99);

            // request 5: the front end cut the last word of A1 (an edge): it is found in a recent answer, not counted
            auto A1t = A1;
            A1t.pop_back();
            pr = render({ { 0, S }, { 1, U1 }, { 2, A1t }, { 1, U2 }, { 2, A2 }, { 1, U3 }, { 2, A3 }, { 1, U4 } });
            g  = gen(A4);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_x == 0 && st.tok_u == (double) U4.size() + 3);

            // request 6: the oldest messages were cut off: nothing new but the user turn
            pr = render({ { 0, S }, { 1, U3 }, { 2, A3 }, { 1, U4 }, { 2, A4 }, { 1, U5 } });
            g  = gen(words(rng, 100));
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_x == 0 && st.tok_u == (double) U5.size() + 3);

            // request 7: a lorebook entry comes in (a system message in the middle) and counts once; request 8 moves it
            const auto A6 = words(rng, 100);
            pr = render({ { 0, S }, { 0, L }, { 1, U3 }, { 2, A3 }, { 1, U4 }, { 2, A4 }, { 1, U6 } });
            g  = gen(A6);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_x == 0 && st.tok_u == (double) L.size() + 1 + U6.size() + 3);
            pr = render({ { 0, S }, { 1, U3 }, { 2, A3 }, { 0, L }, { 1, U4 }, { 2, A4 }, { 1, U6 }, { 2, A6 }, { 1, U7 } });
            g  = gen(words(rng, 100));
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(st.tok_x == 0 && st.tok_u == (double) U7.size() + 3);

            // a request whose answer loops: a trigram counts at most 4 times
            std::vector<int32_t> loop;
            for (int i = 0; i < 50; ++i) {
                loop.insert(loop.end(), { 5, 6, 7 });
            }
            const double w0 = p.weight(5, 6, 7, common_spec_tprof::SRC_M);
            pr = render({ { 1, words(rng, 40) } });
            g  = gen(loop);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            st = p.stats();
            CHECK(p.weight(5, 6, 7, common_spec_tprof::SRC_M) - w0 <= 4.0 + 1e-5 && st.sat_cut > 100);
        }
        // a conversation from elsewhere (a new profile): its assistant messages are X
        rm(c.path);
        {
            common_spec_tprof p(c, vocab());
            CHECK(p.open(nullptr));
            const auto X1 = words(rng, 120);
            const auto X2 = words(rng, 110);
            const auto pr = render({ { 0, S }, { 1, U1 }, { 2, X1 }, { 1, U2 }, { 2, X2 }, { 1, U3 } });
            const auto g  = gen(words(rng, 50));
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            const auto & st = p.stats();
            CHECK(st.tok_x == (double) X1.size() + 3 + X2.size() + 3);
            CHECK(st.tok_u == (double) pr.size() - st.tok_x - 1 && st.tok_known == 1);
            CHECK(p.weight(X1[10], X1[11], X1[12], common_spec_tprof::SRC_X) > 0.99);
        }
        rm(c.path);
    }

    // 3. the decay: lazy (at the time of a read or an update) = halving every half_life generated tokens. The
    //    trigram (207, 208, 209) of control tokens is not in the random words
    {
        common_spec_tprof_config c;
        c.path      = tmp("test-spec-tprof-decay.bin");
        c.half_life = 1000.0;
        rm(c.path);
        common_spec_tprof p(c, vocab());
        CHECK(p.open(nullptr));
        std::mt19937 rng(3);
        auto pr = render({ { 1, words(rng, 30) } });
        auto g  = gen({ 207, 208, 209 });
        p.commit(pr.data(), pr.size(), g.data(), g.size(), R); // at clock 0, then clock 5
        CHECK(std::fabs(p.weight(207, 208, 209, common_spec_tprof::SRC_M) - std::exp2(-5.0 / 1000.0)) < 1e-5);
        pr = render({ { 1, words(rng, 30) } });
        g  = gen(words(rng, 995));
        p.commit(pr.data(), pr.size(), g.data(), g.size(), R); // clock 1002
        const double w1 = std::exp2(-1002.0 / 1000.0);
        CHECK(std::fabs(p.weight(207, 208, 209, common_spec_tprof::SRC_M) - w1) < 1e-5);
        const double ctx0 = p.context_weight(207, 208, common_spec_tprof::SRC_M);
        CHECK(std::fabs(ctx0 - w1) < 1e-5);
        // an update decays the old count first, then adds; read 5 generated tokens later
        pr = render({ { 1, words(rng, 30) } });
        g  = gen({ 207, 208, 209 });
        p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
        const double f = std::exp2(-5.0 / 1000.0);
        CHECK(std::fabs(p.weight(207, 208, 209, common_spec_tprof::SRC_M) - (w1 + 1.0) * f) < 1e-5);
        CHECK(std::fabs(p.context_weight(207, 208, common_spec_tprof::SRC_M) - (ctx0 + 1.0) * f) < 1e-5);
        rm(c.path);
    }

    // 4. the file: a round trip, the lock, another tokenizer and a damaged file (errors: off, the file not touched),
    //    the previous save when the main file is damaged
    {
        common_spec_tprof_config c;
        c.path    = tmp("test-spec-tprof-file.bin");
        c.tok_key = "toy 1";
        rm(c.path);
        std::mt19937 rng(4);
        const auto U = words(rng, 100);
        const auto A = words(rng, 200);
        auto pr = render({ { 1, U } });
        auto g  = gen(A);
        std::vector<uint64_t> ids;
        {
            common_spec_tprof p(c, vocab());
            CHECK(p.open(nullptr));
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            common_spec_tprof_coef cf;
            cf.key = "pair k1";
            cf.b.assign(14, 0.5);
            cf.G.assign(14, 2.0);
            cf.updates = 77;
            p.put_coef(cf);
            ids = p.chunk_ids(A.data(), A.size());
            // a second process: the lock makes it read-only
            common_spec_tprof q(c, vocab());
            CHECK(q.open(nullptr) && q.read_only());
        } // saved here
        CHECK(std::filesystem::exists(c.path) && !std::filesystem::exists(c.path + ".lock"));
        {
            common_spec_tprof p(c, vocab());
            CHECK(p.open(nullptr) && !p.read_only());
            CHECK(std::fabs(p.weight(A[5], A[6], A[7], common_spec_tprof::SRC_M) - std::exp2(-(double) g.size() / c.half_life)) < 1e-5);
            CHECK(std::fabs(p.weight(U[5], U[6], U[7], common_spec_tprof::SRC_U) - std::exp2(-(double) g.size() / c.half_life)) < 1e-5);
            for (uint64_t id : ids) {
                CHECK(p.known(id));
            }
            common_spec_tprof_coef cf;
            CHECK(p.get_coef("pair k1", cf) && cf.b[3] == 0.5 && cf.G[13] == 2.0 && cf.updates == 77);
            CHECK(!p.get_coef("pair k3", cf));
            // the history of request 1 is known after the restart
            const auto pr2 = render({ { 1, U }, { 2, A }, { 1, words(rng, 20) } });
            const auto g2  = gen(words(rng, 30));
            p.commit(pr2.data(), pr2.size(), g2.data(), g2.size(), R);
            CHECK(p.stats().tok_u == 20 + 3 && p.stats().tok_x == 0);
        } // saved: the first save is the .bak now
        CHECK(std::filesystem::exists(c.path + ".bak"));
        const std::string good = read_file(c.path);
        // another tokenizer: an error, off, the file not touched
        {
            common_spec_tprof_config c2 = c;
            c2.tok_key = "toy 2";
            common_spec_tprof p(c2, vocab());
            std::string why;
            CHECK(!p.open(&why) && !p.on() && why.find("another tokenizer") != std::string::npos);
            double cnt[3], N[3];
            const int32_t t = A[7];
            p.lookup(A[5], A[6], &t, 1, cnt, N);
            CHECK(N[0] == 0 && cnt[0] == 0);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R); // nothing
            CHECK(p.stats().req == 0);
        }
        CHECK(read_file(c.path) == good && !std::filesystem::exists(c.path + ".lock"));
        // a damaged main file: the previous save is used
        {
            std::string bad = good;
            bad[bad.size() / 2] ^= 0x5a;
            std::ofstream(c.path, std::ios::binary | std::ios::trunc).write(bad.data(), (std::streamsize) bad.size());
            common_spec_tprof p(c, vocab());
            CHECK(p.open(nullptr) && p.on());
            CHECK(p.weight(A[5], A[6], A[7], common_spec_tprof::SRC_M) > 0.99);
        }
        // damaged and no previous save: an error, off, the file not touched
        {
            std::error_code ec;
            std::filesystem::remove(c.path + ".bak", ec);
            std::string bad = good;
            bad[bad.size() / 3] ^= 0x5a;
            std::ofstream(c.path, std::ios::binary | std::ios::trunc).write(bad.data(), (std::streamsize) bad.size());
            {
                common_spec_tprof p(c, vocab());
                std::string why;
                CHECK(!p.open(&why) && why.find("damaged") != std::string::npos);
            }
            CHECK(read_file(c.path) == bad);
        }
        rm(c.path);
    }

    // 5. the memory limit: the lightest trigrams go, the others keep their exact weights (the deletions of the open
    //    addressing table keep every entry reachable)
    {
        common_spec_tprof_config c;
        c.path      = tmp("test-spec-tprof-cap.bin");
        c.cap_mib   = 8.0;
        c.half_life = 1e9;
        rm(c.path);
        const int32_t NW = 5000;
        common_spec_tprof p(c, vocab(NW));
        CHECK(p.open(nullptr));
        std::mt19937 rng(5);
        std::map<std::array<int32_t, 3>, double> ref; // M counts (saturated per request, no decay to speak of)
        size_t max_nodes = 0;
        for (int r = 0; r < 40; ++r) {
            std::vector<int32_t> a = words(rng, 1500, NW);
            // a few heavy trigrams that must survive: (1, 2, 3) in every request
            a.insert(a.end(), { 1, 2, 3, 1, 2, 3 });
            for (auto & x : a) {
                x = x >= 200 && x < NV ? x + NV : x; // no control tokens
            }
            const auto pr = render({ { 1, { (int32_t) (r + 300) } } });
            const auto g  = gen(a);
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            std::map<std::array<int32_t, 3>, int> occ;
            std::vector<int32_t> s = pr;
            s.insert(s.end(), g.begin(), g.end());
            for (size_t i = pr.size(); i < s.size(); ++i) {
                occ[{ s[i - 2], s[i - 1], s[i] }]++;
            }
            for (const auto & [k, o] : occ) {
                ref[k] += std::min(o, 4);
            }
            max_nodes = std::max(max_nodes, p.n_nodes());
        }
        CHECK(p.stats().mib <= 8.0);
        size_t present = 0;
        for (const auto & [k, w] : ref) {
            const double x = p.weight(k[0], k[1], k[2], common_spec_tprof::SRC_M);
            if (x > 0.0) {
                present++;
                CHECK(std::fabs(x - w) < 1e-3 * w);
            }
        }
        // the prompts' trigrams (U) are in the table too
        for (int r = 0; r < 40; ++r) {
            const auto pr = render({ { 1, { (int32_t) (r + 300) } } });
            for (size_t i = 2; i < pr.size(); ++i) {
                if (p.weight(pr[i - 2], pr[i - 1], pr[i], common_spec_tprof::SRC_U) > 0.0 &&
                    p.weight(pr[i - 2], pr[i - 1], pr[i], common_spec_tprof::SRC_M) == 0.0) {
                    present++;
                }
            }
        }
        CHECK(present == p.n_nodes());       // every entry is reachable
        CHECK(ref.size() > 2 * p.n_nodes()); // the limit was reached
        CHECK(p.weight(1, 2, 3, common_spec_tprof::SRC_M) > 50.0);
        fprintf(stderr, "tprof cap 8 MiB: %zu trigrams kept of %zu, at most %zu, %.2f MiB\n", p.n_nodes(), ref.size(),
                max_nodes, p.stats().mib);
        rm(c.path);
    }

    // 6. the trigram confidence with a profile: the counts of an earlier request reach the features (PM)
    {
        const std::string path = tmp("test-spec-tprof-smart.bin");
        rm(path);
        common_spec_smart_config c = base_cfg(3);
        c.tconf      = true;
        c.tprof_path = path;
        {
            common_spec_smart s(c, 1);
            s.tconf_vocab(std::vector<uint32_t>(NV, (1u << 24) | 'w'), std::vector<uint8_t>(NV, 0));
            s.tconf_roles(roles());
            s.tconf_profile(vocab(), "toy");
            CHECK(s.tprof_on());
            set_identity(s, 0.6);
            set_costs(s, { 30, 33, 36, 39 }, 2.0);
            // request 1 generates (7, 8) -> 9 three times
            const auto pr = render({ { 1, { 50, 51, 52, 53 } } });
            s.request_begin(0);
            s.tconf_request(0, pr.data(), pr.size());
            const auto g = gen({ 7, 8, 9, 60, 7, 8, 9, 61, 7, 8, 9 });
            s.tconf_end(0, g.data(), g.size());
            s.tconf_end(0, g.data(), g.size()); // a second end of the same request: nothing
            CHECK(s.tprof->stats().req == 1);
            // request 2: the context ends with (7, 8)
            const auto pr2 = render({ { 1, { 70, 71, 7, 8 } } });
            s.request_begin(0);
            s.tconf_request(0, pr2.data(), pr2.size() - 1);
            s.close_round(0, 0);
            s.tconf_begin(0, pr2.data(), pr2.size() - 1, pr2.back(), (int32_t) pr2.size());
            // the stream is ... 7 8 <|assistant|> <think>; the context for the pick is the last two tokens
            s.seqs[0].tc_ctx = { 70, 71, 7, 8 };
            s.begin_round(0, 3);
            const int32_t ids[3] = { 9, 1, 2 };
            const float   p[3]   = { 0.5f, 0.3f, 0.2f };
            double q = -1.0;
            s.tconf_pick(0, ids, p, 3, -1, &q);
            const auto & ps = s.seqs[0].tc_pos[0];
            const double d = std::exp2(-(double) g.size() / c.tprof_half_life);
            CHECK(std::fabs(ps.src[4] - 3.0 * d) < 1e-4 && std::fabs(ps.src[5] - 3.0 * d) < 1e-4); // PM: n 3, N 3
            CHECK(ps.src[6] == 0 && ps.src[8] == 0);                                               // PX, PU
            CHECK(s.tprof->stats().req == 1 && s.tprof->stats().tok_m == (double) g.size());
        }
        CHECK(std::filesystem::exists(path));
        // a profile of another tokenizer: an error, the profile off, the confidence on
        {
            common_spec_smart s(c, 1);
            s.tconf_profile(vocab(), "another");
            CHECK(!s.tprof_on() && s.tconf_on());
        }
        rm(path);
    }

    // 6b. the coefficients of the trigram confidence are kept in the profile per model pair and k
    {
        const std::string path = tmp("test-spec-tprof-coef.bin");
        rm(path);
        common_spec_smart_config c = base_cfg(3);
        c.tconf          = true;
        c.tprof_path     = path;
        c.tprof_coef_key = "target=a draft=b";
        auto make = [&](const common_spec_smart_config & cc) {
            auto s = std::make_unique<common_spec_smart>(cc, 1);
            s->tconf_profile(vocab(), "toy");
            return s;
        };
        common_spec_tconf st;
        st.reset(1);
        {
            auto s = make(c);
            CHECK(s->tprof_on() && s->tconf_model.b[3] == st.b[3]); // nothing kept: static
            s->tconf_model.b[3] = 0.777;
            s->tconf_model.G[3] = 12.5;
            s->tc_updates       = 5;
        } // kept at exit
        {
            auto s = make(c);
            CHECK(s->tconf_model.b[3] == 0.777 && s->tconf_model.G[3] == 12.5 && s->tc_updates_kept == 5);
            s->tc_updates = 2;
        }
        {
            auto s = make(c);
            CHECK(s->tc_updates_kept == 7);
        }
        // another model pair or another k: the static coefficients
        common_spec_smart_config c2 = c;
        c2.tprof_coef_key = "target=a draft=c";
        CHECK(make(c2)->tconf_model.b[3] == st.b[3]);
        common_spec_smart_config c3 = c;
        c3.tconf_k = 3;
        common_spec_tconf st3;
        st3.reset(3);
        CHECK(make(c3)->tconf_model.b[3] == st3.b[3]);
        rm(path);
    }

    // 6d. the profile file: <--spec-smart-store>.trigram, on by default; --no-spec-trigram-profile or the environment
    //     turns it off; no store, no profile
    {
        common_params p;
        CHECK(parse_resolve({ "--spec-type", "draft-mtp" }, p) && p.speculative.draft.tprof);
        CHECK(common_spec_tprof_path(p.speculative.draft.smart_store, p.speculative.draft.tprof).empty());
        CHECK(parse_resolve({ "--spec-type", "draft-mtp", "--spec-smart-store", "J:/s/glm.txt" }, p));
        CHECK(common_spec_tprof_path(p.speculative.draft.smart_store, p.speculative.draft.tprof) == "J:/s/glm.txt.trigram");
        CHECK(parse_resolve({ "--spec-type", "draft-mtp", "--spec-smart-store", "J:/s/glm.txt", "--no-spec-trigram-profile" }, p));
        CHECK(!p.speculative.draft.tprof &&
              common_spec_tprof_path(p.speculative.draft.smart_store, p.speculative.draft.tprof).empty());
        set_env("LLAMA_ARG_SPEC_TRIGRAM_PROFILE", "0");
        CHECK(parse_resolve({ "--spec-type", "draft-mtp", "--spec-smart-store", "J:/s/glm.txt" }, p) && !p.speculative.draft.tprof);
        set_env("LLAMA_ARG_SPEC_TRIGRAM_PROFILE", "1");
        CHECK(parse_resolve({ "--spec-type", "draft-mtp", "--spec-smart-store", "J:/s/glm.txt" }, p) && p.speculative.draft.tprof);
        set_env("LLAMA_ARG_SPEC_TRIGRAM_PROFILE", nullptr);
        // the old form with a path is gone
        CHECK(!parse_resolve({ "--spec-type", "draft-mtp", "--spec-trigram-profile", "J:/s/x.bin" }, p));
    }

    // 6c. the decayed AdaGrad: the sum of squared gradients stays near 1 / (1 - 2^(-1/4096)) under a steady gradient,
    //     so the step does not go to 0
    {
        common_spec_tconf m;
        m.reset(1);
        double x[common_spec_tconf::DIM] = {};
        x[0] = 1.0;
        for (int it = 0; it < 100000; ++it) {
            m.b[0] = -30.0; // q ~ 0: gradient 1 for the label 1
            const double lab = 1.0;
            m.update(x, &lab, 1);
        }
        const double lim = 1.0 / (1.0 - std::exp2(-1.0 / 4096.0));
        CHECK(std::fabs(m.G[0] - lim) < 0.01 * lim);
    }

    // 7. host cost: a commit of a 100k-token prompt and a 1000-token answer, a lookup in a full profile
    {
        common_spec_tprof_config c;
        c.path = tmp("test-spec-tprof-cost.bin");
        rm(c.path);
        const int32_t NW = 30000;
        common_spec_tprof p(c, vocab(NW));
        CHECK(p.open(nullptr));
        std::mt19937 rng(7);
        auto fix = [&](std::vector<int32_t> v) {
            for (auto & x : v) {
                x = x >= 200 && x < NV ? x + NV : x;
            }
            return v;
        };
        double ms_first = 0.0;
        for (int r = 0; r < 8; ++r) {
            const auto pr = render({ { 0, fix(words(rng, 100000, NW)) } });
            const auto g  = gen(fix(words(rng, 1000, NW)));
            const int64_t t0 = ggml_time_us();
            p.commit(pr.data(), pr.size(), g.data(), g.size(), R);
            if (r == 0) {
                ms_first = (ggml_time_us() - t0) / 1000.0;
            }
        }
        const int64_t t1 = ggml_time_us();
        const int n_it = 200000;
        double acc = 0.0;
        for (int it = 0; it < n_it; ++it) {
            int32_t ids[10];
            for (int i = 0; i < 10; ++i) {
                ids[i] = (int32_t) (rng() % NW);
            }
            double cnt[30], N[3];
            p.lookup((int32_t) (rng() % NW), (int32_t) (rng() % NW), ids, it % 2 ? 10 : 1, cnt, N);
            acc += N[2] + cnt[2];
        }
        const int64_t t2 = ggml_time_us();
        fprintf(stderr, "tprof: commit of a 100k-token prompt %.1f ms; %zu trigrams %.1f MiB; lookup (1 or 10 candidates) "
                "%.3f us (%g)\n", ms_first, p.n_nodes(), p.stats().mib, (double) (t2 - t1) / n_it, acc);
        rm(c.path);
    }
    return 0;
}

//
// block drafter (draft-dspark): the width of a drafted block, the decision to draft before the forward, the position
// groups of the calibration, the store, the trigram confidence per position and the option rules
//

static common_spec_smart_config block_cfg() {
    common_spec_smart_config c;
    c.n_max    = 4;
    c.block    = true;
    c.n_groups = 4;
    return c;
}

// q = p exactly in all four position groups
static void set_identity_groups(common_spec_smart & s) {
    for (int g = 0; g < s.n_groups; ++g) {
        for (int b = 0; b < s.cfg.n_bins; ++b) {
            s.cal_cells[(size_t) g * s.cfg.n_bins + b] = { (b + 0.5) / s.cfg.n_bins, 1e9 };
        }
    }
}

// the best width by enumeration: (expected accepted + 1) / (D + C[k]), the last verified position with ql
static int ref_block_k(const std::vector<double> & q, const std::vector<double> & ql, int n_lim, double D,
                       const std::vector<double> & C) {
    int best_k = 0;
    double best = -1.0;
    for (int k = 0; k <= std::min<int>((int) q.size(), n_lim); ++k) {
        double A = 0.0;
        for (int m = 1; m <= k; ++m) {
            double P = 1.0;
            for (int j = 0; j < m; ++j) {
                P *= (j == k - 1) ? ql[j] : q[j];
            }
            A += P;
        }
        const double T = (A + 1.0) / std::max(D + C[k], 1.0);
        if (T > best) {
            best = T;
            best_k = k;
        }
    }
    return best_k;
}

static int test_block() {
    // 1. the width of a drafted block against the enumeration, with and without the last-position q
    {
        common_spec_smart s(block_cfg(), 1);
        CHECK(s.n_groups == 4 && s.cal_cells.size() == (size_t) 4 * s.cfg.n_bins);
        set_identity_groups(s);
        std::mt19937 rng(7);
        std::uniform_real_distribution<double> U(0.02, 0.98);
        for (int it = 0; it < 2000; ++it) {
            std::vector<double> C(5);
            C[0] = 30000.0 + 30000.0 * U(rng);
            for (int k = 1; k <= 4; ++k) {
                C[k] = C[k - 1] + 40000.0 * U(rng);
            }
            const double D = (it % 2) ? 0.0 : 30000.0 * U(rng);
            std::vector<double> qd(4), ql(4);
            for (int j = 0; j < 4; ++j) {
                qd[j] = U(rng);
                ql[j] = std::max(qd[j], U(rng));
            }
            const int n_lim = it % 5;
            CHECK(s.decide_block(nullptr, 4, n_lim, D, C, nullptr, qd.data(), nullptr) == ref_block_k(qd, qd, n_lim, D, C));
            CHECK(s.decide_block(nullptr, 4, n_lim, D, C, nullptr, qd.data(), ql.data()) == ref_block_k(qd, ql, n_lim, D, C));
            // without the trigram confidence: q(j, p) of the cells (identity: the bin centre)
            float p[4];
            std::vector<double> qc(4);
            for (int j = 0; j < 4; ++j) {
                p[j]  = (float) qd[j];
                qc[j] = s.q(j, p[j]);
            }
            CHECK(s.decide_block(p, 4, n_lim, D, C) == ref_block_k(qc, qc, n_lim, D, C));
        }
        // the measured DS4F costs (ms) and a confident block: the sunk draft makes the widest width pay
        std::vector<double> C = { 45400, 73300, 97100, 116600, 141300 };
        const double q4[4] = { 0.95, 0.95, 0.95, 0.95 };
        CHECK(s.decide_block(nullptr, 4, 4, 0.0, C, nullptr, q4) == 4);
        const double q0[4] = { 0.3, 0.9, 0.9, 0.9 };
        CHECK(s.decide_block(nullptr, 4, 4, 0.0, C, nullptr, q0) == 0);
    }

    // 2. the decision to draft before the forward: weak blocks skip it, a probe after skip_probe skipped rounds, strong
    //    blocks draft every round
    {
        auto cfg = block_cfg();
        cfg.skip_probe = 4;
        cfg.blk_keep = 8;
        common_spec_smart s(cfg, 1);
        set_identity_groups(s);
        // costs: verification 45 / 73 / 97 / 117 / 141 ms, draft step 30 ms (the joint expert cache)
        for (int k = 0; k <= 4; ++k) {
            const double ms[] = { 45.4, 73.3, 97.1, 116.6, 141.3 };
            s.observe_cost(k, ms[k] * 1000.0);
            s.used_run[k] = 1;
            s.cost_n_run[k] = 1;
        }
        s.add_draft_step(29800.0);
        int64_t t = 0;
        int32_t k_last = 0;
        auto round = [&](const float * p, bool & drew) {
            // the verification of the last round takes the time of its width (the costs stay as set)
            const double ms[] = { 45.4, 73.3, 97.1, 116.6, 141.3 };
            s.close_round(0, t += (int64_t) (ms[k_last] * 1000.0));
            drew = s.begin_block(0, 4);
            int32_t k = 0;
            if (drew) {
                k = s.block_drafted(0, p, 4);
                s.add_draft_step(29800.0);
            }
            k_last = k;
            s.drafted(0, k, t += 1000);
            if (k > 0) {
                s.accepted(0, 0);
            }
            return k;
        };
        const float weak[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        bool drew = false;
        round(weak, drew);
        CHECK(drew); // nothing kept yet: draft
        int n_skip = 0;
        for (int r = 0; r < 4; ++r) {
            round(weak, drew);
            n_skip += !drew;
        }
        CHECK(n_skip == 4);
        round(weak, drew);
        CHECK(drew && s.seqs[0].blk_probe); // the probe after 4 skipped rounds
        // strong blocks: the probe keeps them, and once they fill the kept blocks every round drafts
        const float strong[4] = { 0.99f, 0.99f, 0.99f, 0.99f };
        int n_draw = 0;
        for (int r = 0; r < 60; ++r) {
            round(strong, drew);
            n_draw += drew;
        }
        CHECK(n_draw >= 40);
        for (int r = 0; r < 10; ++r) {
            round(strong, drew);
            CHECK(drew && !s.seqs[0].blk_probe);
        }
        // with a cheap draft step (the draft resident in VRAM) the medium blocks draft
        common_spec_smart s2(cfg, 1);
        set_identity_groups(s2);
        for (int k = 0; k <= 4; ++k) {
            const double ms[] = { 45.0, 76.2, 100.6, 120.7, 146.3 };
            s2.observe_cost(k, ms[k] * 1000.0);
            s2.used_run[k] = 1;
            s2.cost_n_run[k] = 1;
        }
        s2.add_draft_step(4900.0);
        const float mid[4] = { 0.85f, 0.8f, 0.8f, 0.8f };
        s2.close_round(0, 1);
        CHECK(s2.begin_block(0, 4));
        CHECK(s2.block_drafted(0, mid, 4) >= 1);
        s2.drafted(0, 1, 2);
        s2.close_round(0, 50000);
        CHECK(s2.begin_block(0, 4));
        CHECK(s2.block_value(0, 4) > 1.0 / s2.cost(0));
    }

    // 3. the calibration of the four position groups takes the verified positions with the prefix accepted only
    {
        common_spec_smart s(block_cfg(), 1);
        s.close_round(0, 1);
        CHECK(s.begin_block(0, 4)); // nothing known: draft
        const float p[4] = { 0.92f, 0.92f, 0.92f, 0.92f };
        const int32_t k = s.block_drafted(0, p, 4);
        CHECK(k == 0); // cold: frontier 0 (the extension needs reprobe_every closed rounds)
        s.drafted(0, 3, 10); // the caller verifies 3 (as an extension would)
        s.accepted(0, 1);
        const int b = s.bin(0.92f);
        auto cell = [&](int g) { return s.cal_cells[(size_t) g * s.cfg.n_bins + b]; };
        CHECK(cell(0).n == 1.0 && cell(0).mean == 1.0); // accepted
        CHECK(cell(1).n == 1.0 && cell(1).mean == 0.0); // the first rejected one
        CHECK(cell(2).n == 0.0 && cell(3).n == 0.0);    // after a rejection, and not verified: not observed
        s.close_round(0, 20);
        CHECK(s.begin_block(0, 4) || true);
        s.block_drafted(0, p, 4);
        s.drafted(0, 4, 30);
        s.accepted(0, 4);
        CHECK(cell(3).n == 1.0 && cell(3).mean == 1.0); // position 3 has its own group
    }

    // 4. the store keeps the four groups; a store of draft-mtp (three groups) is not read
    {
        const std::string path = (std::filesystem::temp_directory_path() / "test-spec-smart-block.store").string();
        auto cfg = block_cfg();
        cfg.store_path = path;
        cfg.store_key  = "k drafter=dspark";
        {
            common_spec_smart s(cfg, 1);
            s.cal_cells[(size_t) 3 * s.cfg.n_bins + 5] = { 0.42, 7.0 };
            s.observe_cost(0, 45000.0);
            CHECK(s.save(path));
        }
        {
            common_spec_smart s(cfg, 1);
            CHECK(s.store_loaded);
            CHECK(s.cal_cells[(size_t) 3 * s.cfg.n_bins + 5].mean == 0.42);
        }
        {
            common_spec_smart_config m;
            m.n_max = 4;
            m.store_key = cfg.store_key;
            common_spec_smart s(m, 1);
            std::string why;
            CHECK(!s.load(path, &why) && why == "another calibration layout");
        }
        std::filesystem::remove(path);
    }

    // 5. the trigram confidence: the last verified position takes the candidate with the highest q; one coefficient
    //    set per position group with LLAMA_SPEC_DSPARK_TCONF_POS=per; the block drafter starts at its own fit
    {
        auto cfg = block_cfg();
        cfg.tconf = true;
        cfg.tconf_k = 2;
        cfg.tconf_per_pos = true;
        common_spec_smart s(cfg, 1);
        CHECK(s.tc_n_models() == 4);
        CHECK(std::fabs(s.tconf_model.b[1] - 1.014855) < 1e-9 && std::fabs(s.tc_model(3).b[1] - 1.014855) < 1e-9);
        const std::vector<int32_t> toks = { 1, 2, 3, 4, 5 };
        s.request_begin(0);
        s.tconf_request(0, toks.data(), toks.size());
        s.close_round(0, 1);
        s.tconf_begin(0, toks.data(), toks.size(), 6, 5);
        CHECK(s.begin_block(0, 4));
        const int32_t ids0[2] = { 10, 11 }, ids1[2] = { 20, 21 };
        const float   p0[2] = { 0.9f, 0.05f }, p1[2] = { 0.3f, 0.6f };
        double q = 0.0, qa[2];
        CHECK(s.tconf_pick(0, ids0, p0, 2, 0, &q, qa) == 0);
        CHECK(s.tconf_pick(0, ids1, p1, 2, 0, &q, qa) == 0 && qa[1] > qa[0]); // the drafted (top) token
        CHECK(s.seqs[0].tc_pos.size() == 2 && s.seqs[0].tc_pos[1].m == 1 && s.tc_invert == 0);
        CHECK(s.tconf_repick(0, 1, ids1, p1, 2, &q) == 1 && std::fabs(q - qa[1]) < 1e-12);
        CHECK(s.seqs[0].tc_pos.size() == 2 && s.seqs[0].tc_pos[1].chosen == 1 && s.tc_invert == 1);
        CHECK(s.seqs[0].tc_ctx.back() == 21 && s.seqs[0].tc_ctx.size() == s.seqs[0].tc_ctx0 + 2);
        const float pb[2] = { 0.9f, 0.3f };
        const double qdb[2] = { 0.8, 0.4 };
        s.block_drafted(0, pb, 2, qdb);
        s.block_last(0, 1, 0.6f, qa[1]);
        CHECK(s.seqs[0].p[1] == 0.6f && std::fabs(s.seqs[0].qd[1] - std::clamp(qa[1], 0.01, 0.99)) < 1e-12);
        // learning goes to the set of the position: position 1 was rejected with the other candidate drafted
        const double b0 = s.tc_model(0).b[0], b1 = s.tc_model(1).b[0];
        s.drafted(0, 2, 10);
        s.accepted(0, 1);
        CHECK(s.tc_model(0).b[0] != b0 && s.tc_model(1).b[0] == b1); // position 0 learnt now, 1 waits for its target
        s.tconf_begin(0, toks.data(), toks.size(), 20, 7);
        CHECK(s.tc_model(1).b[0] != b1);
    }

    // 6. the option rules: draft-dspark on by default as for draft-mtp; --no-spec-smart or a threshold turns it off
    {
        common_params p;
        CHECK(parse_resolve({ "--spec-type", "draft-dspark" }, p));
        CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 7); // no draft GGUF to read the block from
        CHECK(parse_resolve({ "--spec-type", "draft-dspark", "--no-spec-smart" }, p));
        CHECK(!p.speculative.draft.smart_on && p.speculative.draft.n_max == 3);
        CHECK(parse_resolve({ "--spec-type", "draft-dspark", "--spec-smart" }, p));
        CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 7); // no draft GGUF to read the block from
        CHECK(parse_resolve({ "--spec-type", "draft-dspark", "--spec-smart", "--spec-draft-n-max", "4" }, p));
        CHECK(p.speculative.draft.smart_on && p.speculative.draft.n_max == 4);
        CHECK(parse_resolve({ "--spec-type", "draft-dspark", "--spec-smart", "--spec-draft-p-min", "0" }, p));
        CHECK(!p.speculative.draft.smart_on);
        CHECK(parse_resolve({ "--spec-type", "draft-dflash", "--spec-smart" }, p));
        CHECK(!p.speculative.draft.smart_on);
    }
    return 0;
}

int main() {
    ggml_time_init(); // common_time_meas in the speculative hooks

    if (test_decisions())       { return 1; }
    if (test_cost_prediction()) { return 1; }
    if (test_store())           { return 1; }
    if (test_request_shrink())  { return 1; }
    if (test_rounds())          { return 1; }
    if (test_time_decay())      { return 1; }
    if (test_reprobe())         { return 1; }
    if (test_extend())          { return 1; }
    if (test_store_decay())     { return 1; }
    if (test_ds_noise())        { return 1; }
    if (test_qbar_positions())  { return 1; }
    if (test_lookahead_mix())   { return 1; }
    if (test_mix0_old())        { return 1; }
    if (test_level_follow())    { return 1; }
    if (test_shape_change())    { return 1; }
    if (test_cold_discovery())  { return 1; }
    if (test_level_wide())      { return 1; }
    if (test_warm_start())      { return 1; }
    if (test_level_noise())     { return 1; }
    if (test_conf_cap())        { return 1; }
    if (test_conf_cap_recover()) { return 1; }
    if (test_conf_cap_ext_rate()) { return 1; }
    if (test_conf_cap_noise())  { return 1; }
    if (test_round_log())       { return 1; }
    if (test_round_log_pool())  { return 1; }
    if (test_cost_decomp())     { return 1; }
    if (test_off())             { return 1; }
    if (test_resolve())         { return 1; }
    if (test_tconf())           { return 1; }
    if (test_tprof())           { return 1; }
    if (test_block())           { return 1; }
    fprintf(stderr, "test-spec-smart: OK\n");
    return 0;
}
