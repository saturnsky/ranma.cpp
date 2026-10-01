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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
// decide as before. They were written for the half-life 1000 of the previous default (now 250) and keep it, so that
// their expectations stay those of the previous values
static const double OLD_HALF_LIFE = 1000.0;

static common_spec_smart_config base_cfg(int n_max) {
    common_spec_smart_config c;
    c.n_max = n_max;
    c.half_life = OLD_HALF_LIFE;
    c.lookahead_mix = 0.0;
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

static int test_off() {
    // the defaults; an instance without draft-mtp never gets a controller, and every hook is a no-op
    common_params p;
    CHECK(p.speculative.draft.smart && !p.speculative.draft.smart_on && p.speculative.draft.smart_store.empty());
    CHECK(p.speculative.draft.smart_half_life == 250.0f && p.speculative.draft.smart_n_max == 7);
    CHECK(p.speculative.draft.n_max == 3); // the single-argument default is unchanged
    {
        common_spec_smart_config c;
        CHECK(c.half_life == 250.0 && c.reprobe_n == 4.0 && c.reprobe_every == 64);
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
    if (test_off())             { return 1; }
    if (test_resolve())         { return 1; }
    fprintf(stderr, "test-spec-smart: OK\n");
    return 0;
}
