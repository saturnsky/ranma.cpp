#include "speculative-verify-cost.h"

#include "log.h"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <stdlib.h> // _environ
#else
extern char ** environ;
#endif

#define VC_INF(fmt, ...) LOG_INF("spec-verify-cost: " fmt, __VA_ARGS__)
#define VC_WRN(fmt, ...) LOG_WRN("spec-verify-cost: " fmt, __VA_ARGS__)

// the meaning of the stored times: core = the target call without graph builds and captures, aux = the rest of a round
static const char * VC_STORE_MAGIC = "ranma-spec-verify-cost 1";

static const char * VC_SKIP[] = { "dup", "failed", "prompt", "mixed", "multi", "replay", "bad", "wide", "untimed", "lend",
                                  "calls", "open" };
static const int32_t VC_N_SKIP = (int32_t) (sizeof(VC_SKIP) / sizeof(VC_SKIP[0]));

int32_t common_spec_verify_cost::skip_index(const char * why) {
    for (int32_t i = 0; why != nullptr && i < VC_N_SKIP; ++i) {
        if (strcmp(why, VC_SKIP[i]) == 0) {
            return i;
        }
    }
    return -1;
}

static void vc_num(std::string & out, double v, const char * fmt = "%.1f") {
    if (!std::isfinite(v)) {
        out += "null";
        return;
    }
    char b[64];
    snprintf(b, sizeof(b), fmt, v);
    out += b;
}

common_spec_verify_cost::common_spec_verify_cost(const common_spec_verify_cost_config & cfg_in) : cfg(cfg_in) {
    cfg.r_max     = std::max<int32_t>(1, cfg.r_max);
    cfg.conf_cap  = cfg.conf_cap > 0.0 ? std::max(cfg.conf_cap, 1.0) : 0.0;
    cfg.prior_n   = std::max(cfg.prior_n, 0.0);
    decay       = std::pow(0.5, 1.0 / std::max(1.0, cfg.half_life));
    level_decay = std::pow(0.5, 1.0 / std::max(1.0, cfg.level_half_life));

    shape_cells.assign(cfg.r_max, {});
    shape_over.assign(cfg.r_max, 0.0);
    aux_cells.assign(cfg.r_max, {});
    cost_n_run.assign(cfg.r_max, 0);
    used_run.assign(cfg.r_max, 0);
    aux_used.assign(cfg.r_max, 0);
    n_skip.assign(VC_N_SKIP, 0);

    if (!cfg.store_path.empty()) {
        std::string why;
        if (load(cfg.store_path, &why)) {
            VC_INF("loaded %s (confidence capped at %g observations, the level at %g)\n", cfg.store_path.c_str(),
                   cfg.cap_load, cfg.level_cap_load);
        } else {
            VC_INF("store %s not used: %s\n", cfg.store_path.c_str(), why.c_str());
        }
    }
    if (!cfg.log_path.empty()) {
        log_f = fopen(cfg.log_path.c_str(), "ab");
        if (log_f == nullptr) {
            VC_WRN("cannot open the log %s, no log\n", cfg.log_path.c_str());
        } else {
            std::string l = "{\"ev\":\"start\",\"r_max\":" + std::to_string(cfg.r_max) + ",\"warm\":" +
                            (store_loaded ? "true" : "false") + ",\"half_life\":";
            vc_num(l, cfg.half_life, "%g");
            l += ",\"level_half_life\":";
            vc_num(l, cfg.level_half_life, "%g");
            l += ",\"lend_skip\":" + std::to_string(cfg.lend_skip) + ",\"cost_graph\":" + (cfg.cost_graph ? "true" : "false") + "}";
            log_put(l);
            log_flush();
        }
    }
}

common_spec_verify_cost::~common_spec_verify_cost() {
    if (!cfg.store_path.empty() && clock > clock_saved) {
        if (save(cfg.store_path)) {
            VC_INF("saved %s\n", cfg.store_path.c_str());
        }
    }
    if (log_f) {
        log_flush();
        fclose(log_f);
    }
}

void common_spec_verify_cost::log_put(const std::string & line) {
    if (!log_f) {
        return;
    }
    log_buf += line;
    log_buf += '\n';
    if (log_buf.size() >= (1u << 20)) {
        log_flush();
    }
}

void common_spec_verify_cost::log_flush() {
    if (log_f && !log_buf.empty()) {
        fwrite(log_buf.data(), 1, log_buf.size(), log_f);
        fflush(log_f);
    }
    log_buf.clear();
}

//
// the parts (moved from common_spec_smart, where the clock was its closed rounds)
//

bool common_spec_verify_cost::level_source(int32_t m) const {
    // widths with a confident shape only; when there is none, every width with a shape
    if (conf(shape_cells[m]) >= cfg.reprobe_n) {
        return true;
    }
    for (const auto & c : shape_cells) {
        if (conf(c) >= cfg.reprobe_n) {
            return false;
        }
    }
    return true;
}

void common_spec_verify_cost::observe_core(int32_t m, double us) {
    if (m < 0 || m >= cfg.r_max || !(us >= 0.0)) {
        return;
    }
    const double x = std::max(us, 1.0); // the ratios need a time above 0
    auto & S = shape_cells[m];
    const bool has_shape = S.n > 0.0 && S.mean > 0.0;

    // the level: the first observation sets it; later x / the shape of the width, both before this observation
    double L = level_cell.mean;
    if (!(L > 0.0)) {
        level_cell = {};
        level_cell.add(has_shape ? x / S.mean : x, level_decay, clock);
        L = level_cell.mean;
    } else if (has_shape && level_source(m)) {
        level_cell.add(x / S.mean, level_decay, clock);
    }

    // the shape of the width: x / the level before this observation. The wait confidence takes the same observation
    // but stays at most conf_cap, so that a width used much and then left waits for an extension again after a bounded
    // time; the mean and its confidence do not see the cap
    const double wc = wait_conf(m);
    // a waiting width: its old mean counts only with the wait confidence (below reprobe_n), so that a few extensions
    // move a mean that stayed high long after the width was left. The weighted variance does not change when all old
    // weights shrink by the same factor; only the confidence is cut. The level above took this observation as before.
    wr_last = false;
    if (cfg.wait_reset && waiting(m)) {
        const double n0 = S.n_at(clock, decay);
        if (wc < n0) {
            S.n = wc;
            S.t = std::max(S.t, clock);
            wr_last     = true;
            wr_n_before = n0;
            wr_n_after  = wc;
            n_wait_reset++;
        }
    }
    S.add(x / L, decay, clock);
    shape_over[m] = cfg.conf_cap > 0.0 ? std::max(0.0, S.n - std::min(wc + 1.0, cfg.conf_cap)) : 0.0;
    cost_n_run[m]++;
}

double common_spec_verify_cost::wait_conf(int32_t m) const {
    common_spec_cost_cell c = shape_cells[m];
    if (cfg.conf_cap > 0.0) {
        c.n = std::max(0.0, c.n - shape_over[m]);
    }
    return conf(c);
}

bool common_spec_verify_cost::waiting(int32_t m) const {
    return m >= 0 && m < cfg.r_max && wait_conf(m) < cfg.reprobe_n;
}

bool common_spec_verify_cost::known(int32_t m) const {
    return m >= 0 && m < cfg.r_max && (used_run[m] || shape_cells[m].n > 0.0);
}

void common_spec_verify_cost::lend_update(bool pool_valid, uint64_t lends, bool counted) {
    if (!pool_valid) {
        return;
    }
    if (lends != pool_lends) {
        pool_since_lend = 0;
    } else if (pool_since_lend >= 0 && counted) {
        pool_since_lend++;
    }
    pool_lends = lends;
}

bool common_spec_verify_cost::lend_round(bool pool_valid) const {
    // the call of a lend of the decode pool and the next ones fill its positions: not a time of the width
    return cfg.lend_skip > 0 && pool_valid && pool_since_lend >= 0 && pool_since_lend < cfg.lend_skip;
}

bool common_spec_verify_cost::take_first(int32_t m) {
    if (m < 0 || m >= cfg.r_max) {
        return false;
    }
    const bool timed = used_run[m] != 0;
    used_run[m] = 1;
    return timed;
}

void common_spec_verify_cost::tick() {
    clock++;
    if (!cfg.store_path.empty() && cfg.save_every > 0 && clock - clock_saved >= cfg.save_every) {
        if (save(cfg.store_path)) {
            clock_saved = clock;
        } else if (!save_failed) {
            save_failed = true;
            VC_WRN("cannot write %s\n", cfg.store_path.c_str());
        }
    }
}

//
// observations
//

const char * common_spec_verify_cost::observe(const common_spec_verify_sample & s, common_spec_verify_obs * out) {
    if (out) {
        *out = {};
        out->clock = clock;
    }
    if (s.call_id >= 0) {
        if (s.call_id <= last_call) {
            n_skip[skip_index("dup")]++;
            return "dup";
        }
        last_call = s.call_id;
    }
    n_calls++;

    auto skip = [&](const char * why) {
        n_skip[skip_index(why)]++;
        return why;
    };
    if (!s.ok) {
        return skip("failed"); // incomplete: no time, no clock
    }
    if (s.n_prompt > 0 && s.r <= 0) {
        return skip("prompt"); // not a verification
    }

    // a generation call: the lend window and the clock count it
    lend_update(s.pool_valid, s.lends, true);

    const char * why = nullptr;
    if (s.n_prompt > 0) {
        why = "mixed";
    } else if (s.n_seq != 1) {
        why = "multi";
    } else if (s.replay) {
        why = "replay";
    } else if (!(s.target_us >= 0.0) || !std::isfinite(s.target_us) || !std::isfinite(s.graph_us) || s.r < 1) {
        why = "bad";
    } else if (s.r > cfg.r_max) {
        why = "wide";
    }
    if (why == nullptr) {
        const int32_t m = s.r - 1;
        if (out && log_f) {
            const auto ct = core_table_m(cfg.r_max);
            const auto at = aux_table_m(cfg.r_max);
            out->pred_core = level_cell.mean > 0.0 ? ct[m] : -1.0;
            out->pred_eff  = level_cell.mean > 0.0 ? ct[m] + at[m] : -1.0;
            if (!log_prior.empty()) {
                out->pred_blend_ms = blend_ms(log_prior, s.r)[m];
            }
        }
        // the first call of a width in this process builds its graphs: not observed
        if (!take_first(m)) {
            why = "untimed";
        } else if (lend_round(s.pool_valid)) {
            why = "lend";
        } else {
            const double g   = cfg.cost_graph ? std::max(s.graph_us, 0.0) : 0.0;
            const double obs = std::max(1.0, s.target_us - g);
            observe_core(m, obs);
            n_core++;
            if (out) {
                out->obs_us = obs;
            }
        }
    }
    if (why != nullptr) {
        n_skip[skip_index(why)]++;
    }
    tick();
    return why;
}

const char * common_spec_verify_cost::observe_aux(int32_t r, double us, const char * why_core) {
    const char * why = nullptr;
    if (why_core != nullptr && strcmp(why_core, "untimed") != 0) {
        why = why_core; // a call that is not a single verification, or a lend: not a round of the width either
    } else if (!(us >= 0.0) || !std::isfinite(us)) {
        why = "bad";
    } else if (r < 1 || r > cfg.r_max) {
        why = "wide";
    } else if (!aux_used[r - 1]) {
        aux_used[r - 1] = 1; // the first round of a width builds the graphs of the drafters' contexts
        why = "untimed";
    }
    if (why != nullptr) {
        return why;
    }
    aux_cells[r - 1].add(us, decay, clock);
    n_aux++;
    return nullptr;
}

//
// estimates
//

std::vector<double> common_spec_verify_cost::core_table_m(int32_t n) const {
    const int32_t N = cfg.r_max;

    // shape values; a width without one takes the line through the two nearest widths with one (the narrower one
    // first on a tie)
    std::vector<double>  val(N, 0.0);
    std::vector<uint8_t> has(N, 0);
    for (int32_t k = 0; k < N; ++k) {
        if (shape_cells[k].n > 0.0) {
            val[k] = shape_cells[k].mean;
            has[k] = 1;
        }
    }
    std::vector<double> out = val;
    for (int32_t k = 0; k < N; ++k) {
        if (has[k]) {
            continue;
        }
        int32_t j[2];
        int32_t nj = 0;
        for (int32_t d = 1; d < N && nj < 2; ++d) {
            if (k - d >= 0 && has[k - d]) {
                j[nj++] = k - d;
            }
            if (nj < 2 && k + d < N && has[k + d]) {
                j[nj++] = k + d;
            }
        }
        if (nj == 2) {
            out[k] = val[j[0]] + (val[j[1]] - val[j[0]]) * (double) (k - j[0]) / (double) (j[1] - j[0]);
        } else if (nj == 1) {
            out[k] = val[j[0]];
        } else {
            out[k] = 0.0;
        }
    }

    // a width not measured by this process is never cheaper than the nearest narrower width that was
    int32_t last_run = -1;
    for (int32_t k = 0; k < N; ++k) {
        if (cost_n_run[k] > 0) {
            last_run = k;
        } else if (last_run >= 0) {
            out[k] = std::max(out[k], out[last_run]);
        }
        out[k] = std::max(out[k], 0.0);
    }

    // time = level x shape (no level yet: no value)
    const double L = std::max(level_cell.mean, 0.0);
    for (auto & x : out) {
        x *= L;
    }
    out.resize(std::max<int32_t>(n, 0), out.empty() ? 0.0 : out.back());
    return out;
}

std::vector<double> common_spec_verify_cost::aux_table_m(int32_t n) const {
    const int32_t N = cfg.r_max;
    std::vector<double> out(N, 0.0);
    std::vector<int32_t> kn;
    for (int32_t k = 0; k < N; ++k) {
        if (aux_cells[k].n > 0.0) {
            kn.push_back(k);
        }
    }
    for (int32_t k = 0; k < N && !kn.empty(); ++k) {
        if (aux_cells[k].n > 0.0) {
            out[k] = aux_cells[k].mean;
            continue;
        }
        // between two observed widths: the line; outside: the nearest
        const auto hi = std::lower_bound(kn.begin(), kn.end(), k);
        if (hi == kn.begin()) {
            out[k] = aux_cells[kn.front()].mean;
        } else if (hi == kn.end()) {
            out[k] = aux_cells[kn.back()].mean;
        } else {
            const int32_t a = *(hi - 1);
            const int32_t b = *hi;
            out[k] = aux_cells[a].mean + (aux_cells[b].mean - aux_cells[a].mean) * (double) (k - a) / (double) (b - a);
        }
    }
    out.resize(std::max<int32_t>(n, 0), out.empty() ? 0.0 : out.back());
    return out;
}

std::vector<double> common_spec_verify_cost::eff_table_m(int32_t n) const {
    std::vector<double> c = core_table_m(n);
    const std::vector<double> a = aux_table_m(n);
    for (size_t k = 0; k < c.size(); ++k) {
        c[k] += a[k];
    }
    return c;
}

common_spec_cost_estimate common_spec_verify_cost::estimate(int32_t r) const {
    common_spec_cost_estimate e;
    const int32_t m = std::clamp<int32_t>(r - 1, 0, cfg.r_max - 1);
    e.core_us  = core_table_m(cfg.r_max)[m];
    e.aux_us   = aux_table_m(cfg.r_max)[m];
    e.shape_n  = conf(shape_cells[m]);
    e.level_n  = level_conf();
    e.variance = shape_cells[m].var;
    e.age      = shape_cells[m].n > 0.0 ? clock - shape_cells[m].t : -1;
    e.origin   = cost_n_run[m] > 0 ? COMMON_SPEC_COST_OBSERVED :
                 shape_cells[m].n > 0.0 ? COMMON_SPEC_COST_STORED :
                 level_cell.mean > 0.0 ? COMMON_SPEC_COST_INTERP : COMMON_SPEC_COST_NONE;
    return e;
}

double common_spec_verify_cost::prior_at(const std::vector<double> & c, int32_t rows) {
    if (c.empty()) {
        return 0.0;
    }
    if (rows >= 1 && rows <= (int32_t) c.size()) {
        return c[rows - 1];
    }
    if (rows < 1) {
        return c.front();
    }
    const double step = c.size() > 1 ? c.back() - c[c.size() - 2] : 0.0;
    return c.back() + step * (double) (rows - (int32_t) c.size());
}

// The widths are smoothed in segments. A segment is a run of widths whose cost is expected to rise with the width:
// neither the neighbour ratios nor the monotone projection cross the border, so a real step or a fall across it
// stays. The border is after the 8th row: the EXL3 GEMV of the target (ggml-cuda/mul-mat-had.cu, MMH_GEMV_TOKENS)
// takes at most 8 rows and the 9th row goes to another kernel. A constant of the learner, not a backend contract.
static constexpr int32_t VC_SEG_ROWS = 8;
static constexpr double  VC_MIN_W    = 1e-6; // weight of a width without a confidence (projection only)

// the weighted isotonic (non-decreasing) fit of v[0..n) by w, in place (pool adjacent violators). The three scratch
// arrays hold n values each.
static void vc_pav(double * v, const double * w, int32_t n, double * bv, double * bw, int32_t * bc) {
    int32_t nb = 0;
    for (int32_t i = 0; i < n; ++i) {
        bv[nb] = v[i];
        bw[nb] = std::max(w[i], 1e-12);
        bc[nb] = 1;
        ++nb;
        while (nb > 1 && bv[nb - 2] > bv[nb - 1]) {
            const double sw = bw[nb - 2] + bw[nb - 1];
            bv[nb - 2] = (bv[nb - 2] * bw[nb - 2] + bv[nb - 1] * bw[nb - 1]) / sw;
            bw[nb - 2] = sw;
            bc[nb - 2] += bc[nb - 1];
            --nb;
        }
    }
    int32_t i = 0;
    for (int32_t b = 0; b < nb; ++b) {
        for (int32_t c = 0; c < bc[b]; ++c) {
            v[i++] = bv[b];
        }
    }
}

void common_spec_verify_cost::weighted_pav(std::vector<double> & v, const std::vector<double> & w) {
    const int32_t n = (int32_t) std::min(v.size(), w.size());
    std::vector<double>  bv(n), bw(n);
    std::vector<int32_t> bc(n);
    vc_pav(v.data(), w.data(), n, bv.data(), bw.data(), bc.data());
}

std::vector<double> common_spec_verify_cost::smooth_table(const std::vector<double> & prior, double unit,
                                                          int32_t n_rows) const {
    const int32_t R = cfg.r_max;
    std::vector<double> out(std::max<int32_t>(n_rows, 0));
    if (!prior.empty() && !(level_cell.mean > 0.0)) {
        for (int32_t r = 1; r <= n_rows; ++r) {
            out[r - 1] = prior_at(prior, r); // nothing observed: the prior
        }
        return out;
    }
    // a usable prior is positive and finite over the whole learner width; else the estimates are only projected
    bool use_prior = !prior.empty();
    for (int32_t r = 1; r <= R && use_prior; ++r) {
        const double p = prior_at(prior, r);
        use_prior = std::isfinite(p) && p > 0.0;
    }
    const std::vector<double> eff = eff_table_m(R);
    const double k = std::max(cfg.prior_n, 0.0);

    // scratch of the whole width (no allocation for the usual widths)
    double  stack_d[6 * 16];
    int32_t stack_i[2 * 16];
    std::vector<double>  heap_d;
    std::vector<int32_t> heap_i;
    double  * d  = stack_d;
    int32_t * ix = stack_i;
    if (R > 16) {
        heap_d.resize(6 * (size_t) R);
        heap_i.resize(2 * (size_t) R);
        d  = heap_d.data();
        ix = heap_i.data();
    }
    double * n  = d;         // confidence of the shape
    double * B  = d + R;     // the table before the projection
    double * W  = d + 2 * R; // weight of the projection
    double * s0 = d + 3 * R; // projection scratch
    double * s1 = d + 4 * R;
    int32_t * lft = ix;      // nearest width with a confidence on the left / right inside the segment, -1: none
    int32_t * rgt = ix + R;

    for (int32_t i = 0; i < R; ++i) {
        n[i] = conf(shape_cells[i]);
    }

    // a segment without any observation keeps the prior curve at the level of r = 1 (by the weight of that estimate)
    double s = 1.0;
    if (use_prior && n[0] > 0.0) {
        const double w1 = n[0] / (n[0] + cfg.prior_n);
        s = w1 * (eff[0] / unit) / prior_at(prior, 1) + (1.0 - w1);
    }

    for (int32_t lo = 0; lo < R;) {
        const int32_t hi = lo < VC_SEG_ROWS ? std::min(VC_SEG_ROWS, R) : R;
        int32_t last = -1;
        bool any = false;
        for (int32_t i = lo; i < hi; ++i) {
            lft[i] = last;
            if (n[i] > 0.0) {
                last = i;
                any = true;
            }
        }
        last = -1;
        for (int32_t i = hi - 1; i >= lo; --i) {
            rgt[i] = last;
            if (n[i] > 0.0) {
                last = i;
            }
        }
        for (int32_t i = lo; i < hi; ++i) {
            const double E = eff[i] / unit;
            if (!use_prior) {
                B[i] = E;
                W[i] = std::max(n[i], VC_MIN_W);
                continue;
            }
            const double P = prior_at(prior, i + 1);
            // the ratio observed / prior of the nearest observed widths of the segment (self excluded), weighted by
            // the distance (linear) and the confidence; what is borrowed adds nothing to the confidence of i
            double b = 1.0;
            if (!any) {
                b = s;
            } else {
                const int32_t l = lft[i];
                const int32_t g = rgt[i];
                double lw = 0.0, gw = 0.0;
                if (l >= 0 && g >= 0) {
                    lw = (double) (g - i) / (double) (g - l);
                    gw = (double) (i - l) / (double) (g - l);
                } else if (l >= 0) {
                    lw = 1.0;
                } else if (g >= 0) {
                    gw = 1.0;
                }
                double num = k, den = k;
                if (lw > 0.0) {
                    num += lw * n[l] * (eff[l] / unit) / prior_at(prior, l + 1);
                    den += lw * n[l];
                }
                if (gw > 0.0) {
                    num += gw * n[g] * (eff[g] / unit) / prior_at(prior, g + 1);
                    den += gw * n[g];
                }
                b = den > 0.0 ? num / den : 1.0;
            }
            B[i] = n[i] + k > 0.0 ? (n[i] * E + k * P * b) / (n[i] + k) : P * b;
            W[i] = std::max(n[i] + k, VC_MIN_W);
        }
        // the right neighbours are dead here: their storage holds the block counts of the projection
        vc_pav(B + lo, W + lo, hi - lo, s0, s1, rgt + lo);
        lo = hi;
    }

    for (int32_t r = 1; r <= n_rows; ++r) {
        if (r <= R) {
            out[r - 1] = B[r - 1];
        } else if (use_prior) {
            // beyond the learner: the prior goes on at the ratio of the last width, never falling
            out[r - 1] = std::max(out[r - 2], B[R - 1] * prior_at(prior, r) / prior_at(prior, R));
        } else {
            out[r - 1] = B[R - 1]; // as eff_table_m()
        }
    }
    return out;
}

std::vector<double> common_spec_verify_cost::blend_ms(const std::vector<double> & prior_ms, int32_t n_rows) const {
    return smooth_table(prior_ms, 1000.0, n_rows);
}

std::vector<double> common_spec_verify_cost::smooth_us(int32_t n_rows) const {
    return smooth_table({}, 1.0, n_rows);
}

std::string common_spec_verify_cost::summary() const {
    std::string s;
    char buf[160];
    snprintf(buf, sizeof(buf), "clock %lld, calls %lld, core %lld, aux %lld, level %.2f ms n %.1f, r: shape (n) aux ms",
             (long long) clock, (long long) n_calls, (long long) n_core, (long long) n_aux, level_cell.mean / 1000.0,
             level_conf());
    s += buf;
    for (int32_t m = 0; m < cfg.r_max; ++m) {
        if (shape_cells[m].n <= 0.0 && aux_cells[m].n <= 0.0) {
            continue;
        }
        snprintf(buf, sizeof(buf), " r%d %.3f (%.1f) %.2f", m + 1, shape_cells[m].mean, conf(shape_cells[m]),
                 aux_cells[m].mean / 1000.0);
        s += buf;
    }
    s += ", skipped:";
    for (int32_t i = 0; i < VC_N_SKIP; ++i) {
        if (n_skip[i] > 0) {
            snprintf(buf, sizeof(buf), " %s %lld", VC_SKIP[i], (long long) n_skip[i]);
            s += buf;
        }
    }
    return s;
}

//
// store
//

static void vc_put(std::string & out, double v) {
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v); // shortest form that reads back to the same double
    out.append(buf, res.ptr);
}

static bool vc_get(const std::string & tok, double & v) {
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    return res.ec == std::errc() && res.ptr == tok.data() + tok.size();
}

static bool vc_get(const std::string & tok, int32_t & v) {
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    return res.ec == std::errc() && res.ptr == tok.data() + tok.size();
}

static std::string vc_key_line(const std::string & key) {
    std::string k = key;
    std::replace(k.begin(), k.end(), '\n', ' ');
    std::replace(k.begin(), k.end(), '\r', ' ');
    return k;
}

bool common_spec_verify_cost::save(const std::string & path) const {
    std::string out;
    out += VC_STORE_MAGIC;
    out += "\nkey ";
    out += vc_key_line(cfg.store_key);
    out += "\nrows " + std::to_string(cfg.r_max) + "\n";
    out += "level ";
    vc_put(out, level_cell.mean); out += " ";
    vc_put(out, level_conf()); out += "\n";
    for (int32_t m = 0; m < cfg.r_max; ++m) {
        out += "shape " + std::to_string(m + 1) + " ";
        vc_put(out, shape_cells[m].mean); out += " ";
        vc_put(out, conf(shape_cells[m])); out += " ";
        vc_put(out, shape_cells[m].var); out += "\n";
    }
    for (int32_t m = 0; m < cfg.r_max; ++m) {
        out += "aux " + std::to_string(m + 1) + " ";
        vc_put(out, aux_cells[m].mean); out += " ";
        vc_put(out, conf(aux_cells[m])); out += " ";
        vc_put(out, aux_cells[m].var); out += "\n";
    }
    out += "end\n";

    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        f.write(out.data(), (std::streamsize) out.size());
        if (!f) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(tmp, path, ec);
    }
    return !ec;
}

bool common_spec_verify_cost::load(const std::string & path, std::string * why) {
    auto fail = [&](const std::string & w) {
        if (why) {
            *why = w;
        }
        return false;
    };

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return fail("no file (cold start)");
    }
    std::string line;
    if (!std::getline(f, line) || line != VC_STORE_MAGIC) {
        return fail("unknown format");
    }
    if (!std::getline(f, line) || line.rfind("key ", 0) != 0) {
        return fail("no key");
    }
    if (line.substr(4) != vc_key_line(cfg.store_key)) {
        return fail("another target or run condition (key differs)");
    }

    // parse everything first; the state changes only when the whole file is valid. A store of another width keeps
    // the widths both have; the others stay unmeasured
    std::vector<common_spec_cost_cell> shape_new(cfg.r_max);
    std::vector<common_spec_cost_cell> aux_new(cfg.r_max);
    common_spec_cost_cell level_new;
    bool rows_ok  = false;
    bool level_ok = false;
    bool end_ok   = false;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::vector<std::string> t;
        for (std::string w; ls >> w;) {
            t.push_back(w);
        }
        if (t.empty()) {
            continue;
        }
        if (t[0] == "end") {
            end_ok = true;
            break;
        }
        common_spec_cost_cell c;
        if (t[0] == "rows" && t.size() == 2) {
            int32_t r = 0;
            if (!vc_get(t[1], r) || r < 1) {
                return fail("bad rows line");
            }
            rows_ok = true;
        } else if (t[0] == "level" && t.size() == 3) {
            if (!vc_get(t[1], c.mean) || !vc_get(t[2], c.n) || !(c.mean >= 0.0) || !std::isfinite(c.mean)) {
                return fail("bad level line");
            }
            level_new = c;
            level_ok  = true;
        } else if ((t[0] == "shape" || t[0] == "aux") && t.size() == 5) {
            int32_t r = 0;
            if (!vc_get(t[1], r) || !vc_get(t[2], c.mean) || !vc_get(t[3], c.n) || !vc_get(t[4], c.var) || r < 1 ||
                !(c.var >= 0.0) || !std::isfinite(c.mean) || !(c.mean >= 0.0)) {
                return fail("bad " + t[0] + " line");
            }
            if (r <= cfg.r_max) {
                (t[0] == "shape" ? shape_new : aux_new)[r - 1] = c;
            }
        } else {
            return fail("unknown line");
        }
    }
    if (!rows_ok || !level_ok || !end_ok) {
        return fail("truncated");
    }

    // the stored means and variances are restored exactly; only their confidence (decayed when saved) is capped
    auto cap = [&](common_spec_cost_cell & c, double cap_n) {
        if (!(c.n >= 0.0)) {
            c.n = 0.0;
        }
        c.n = std::min(c.n, cap_n);
        c.t = clock;
    };
    cap(level_new, cfg.level_cap_load); // a small confidence: the first observations of this process move it
    level_cell = level_new;
    for (int32_t m = 0; m < cfg.r_max; ++m) {
        cap(shape_new[m], cfg.cap_load);
        shape_cells[m] = shape_new[m];
        // the wait confidence at the running cap as well (the store keeps only the confidence of the mean)
        shape_over[m]  = cfg.conf_cap > 0.0 ? std::max(0.0, shape_cells[m].n - cfg.conf_cap) : 0.0;
        cost_n_run[m]  = 0;
        cap(aux_new[m], cfg.cap_load);
        aux_cells[m] = aux_new[m];
    }
    store_loaded = true;
    return true;
}

//
// rounds
//

common_spec_verify_rounds::common_spec_verify_rounds(common_spec_verify_cost * vc, int32_t n_seq) : vc(vc) {
    seqs.resize(std::max<int32_t>(1, n_seq));
}

static std::string vc_type_name(const std::vector<std::string> & names, int32_t t) {
    if (t < 0) {
        return "none";
    }
    return t < (int32_t) names.size() && !names[t].empty() ? names[t] : std::to_string(t);
}

void common_spec_verify_rounds::draft_begin(int32_t seq, int64_t t_us, double dft_graph_us) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    if (!st.open) {
        return;
    }
    st.open = false;

    const char * why = nullptr;
    double wall = (double) (t_us - st.t_drafted);
    if (vc->cfg.cost_graph) {
        wall -= std::max(0.0, dft_graph_us - st.g_drafted);
    }
    const double aux = wall - st.target_us;
    if (st.call_id < 0) {
        why = "open"; // no call verified it (the slot stopped)
    } else if (!st.calls_ok) {
        why = "calls";
    } else {
        why = vc->observe_aux(st.m + 1, aux, st.why);
    }
    if (vc->logging()) {
        std::string l = "{\"ev\":\"round\",\"seq\":" + std::to_string(seq) + ",\"owner\":\"" +
                        vc_type_name(type_names, st.owner) + "\",\"attempted\":[";
        bool first = true;
        for (int32_t t = 0; t < 32; ++t) {
            if (st.attempted & (1u << t)) {
                l += (first ? "\"" : ",\"") + vc_type_name(type_names, t) + "\"";
                first = false;
            }
        }
        l += "],\"m\":" + std::to_string(st.m) + ",\"call\":" + std::to_string(st.call_id) + ",\"replays\":" +
             std::to_string(st.n_replay) + ",\"wall_us\":";
        vc_num(l, wall);
        l += ",\"target_us\":";
        vc_num(l, st.call_id >= 0 ? st.target_us : NAN);
        l += ",\"aux_us\":";
        vc_num(l, st.call_id >= 0 ? aux : NAN);
        l += ",\"pred_aux\":";
        vc_num(l, st.pred_aux);
        l += ",\"pred_eff\":";
        vc_num(l, st.pred_eff);
        l += ",\"skip\":";
        l += why != nullptr ? std::string("\"") + why + "\"" : std::string("null");
        l += "}";
        vc->log_put(l);
    }
}

void common_spec_verify_rounds::drafted(int32_t seq, int64_t t_us, int32_t m, int32_t owner, uint32_t attempted,
                                        double dft_graph_us) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    st = {};
    st.open      = true;
    st.t_drafted = t_us;
    st.g_drafted = dft_graph_us;
    st.m         = std::max<int32_t>(0, m);
    st.owner     = owner;
    st.attempted = attempted;
    if (vc->logging() && vc->level() > 0.0) {
        const int32_t i = std::min<int32_t>(st.m, vc->cfg.r_max - 1);
        st.pred_aux = vc->aux_table_m(vc->cfg.r_max)[i];
        st.pred_eff = vc->eff_table_m(vc->cfg.r_max)[i];
    } else {
        st.pred_aux = NAN;
        st.pred_eff = NAN;
    }
}

void common_spec_verify_rounds::call(const common_spec_verify_sample & s, const char * why,
                                     const common_spec_verify_obs * obs) {
    if (why != nullptr && strcmp(why, "dup") == 0) {
        return;
    }
    // a call of other sequences inside a round: its time is in the round's but not its own (several sequences are not
    // learnt yet)
    for (int32_t q = 0; q < (int32_t) seqs.size(); ++q) {
        if (q != s.seq && seqs[q].open) {
            seqs[q].calls_ok = false;
        }
    }
    if (s.seq >= 0 && s.seq < (int32_t) seqs.size()) {
        auto & st = seqs[s.seq];
        if (st.open) {
            if (st.call_id < 0) {
                st.call_id   = s.call_id;
                st.target_us = s.target_us;
                st.why       = why;
                if (s.r != st.m + 1 || s.replay || s.n_seq != 1) {
                    st.calls_ok = false; // not the verification of this draft
                }
            } else if (s.replay && s.n_seq == 1 && s.n_prompt == 0) {
                st.n_replay++; // a checkpoint restore replays the accepted tokens: a cost of the rollback (aux)
            } else {
                st.calls_ok = false;
            }
        }
    }
    if (vc->logging()) {
        const bool own = s.seq >= 0 && s.seq < (int32_t) seqs.size() && seqs[s.seq].open;
        std::string l = "{\"ev\":\"call\",\"call\":" + std::to_string(s.call_id) + ",\"seq\":" + std::to_string(s.seq) +
                        ",\"r\":" + std::to_string(s.r) + ",\"n_seq\":" + std::to_string(s.n_seq) + ",\"n_prompt\":" +
                        std::to_string(s.n_prompt) + ",\"replay\":" + (s.replay ? "true" : "false") + ",\"owner\":\"" +
                        (own ? vc_type_name(type_names, seqs[s.seq].owner) : std::string("-")) + "\",\"pos\":" +
                        std::to_string(s.pos) + ",\"target_us\":";
        vc_num(l, s.target_us);
        l += ",\"graph_us\":";
        vc_num(l, s.graph_us);
        l += ",\"obs_us\":";
        vc_num(l, obs && obs->obs_us >= 0.0 ? obs->obs_us : NAN);
        l += ",\"pred_core\":";
        vc_num(l, obs && obs->pred_core >= 0.0 ? obs->pred_core : NAN);
        l += ",\"pred_eff\":";
        vc_num(l, obs && obs->pred_eff >= 0.0 ? obs->pred_eff : NAN);
        if (!vc->log_prior.empty() && s.r >= 1) {
            // what a prior-blended table (the hazard model with the shared cost) would have used, before the call
            l += ",\"pred_blend_ms\":";
            vc_num(l, obs ? obs->pred_blend_ms : NAN, "%.3f");
            l += ",\"prior_ms\":";
            vc_num(l, common_spec_verify_cost::prior_at(vc->log_prior, s.r), "%.3f");
        }
        l += ",\"clock\":" + std::to_string(obs ? obs->clock : vc->clock) + ",\"since_lend\":" +
             std::to_string(vc->pool_since_lend) + ",\"skip\":";
        l += why != nullptr ? std::string("\"") + why + "\"" : std::string("null");
        l += "}";
        vc->log_put(l);
    }
}

void common_spec_verify_rounds::request_end(int32_t seq) {
    if (seq >= 0 && seq < (int32_t) seqs.size()) {
        seqs[seq].open = false;
    }
}

int32_t common_spec_verify_rounds::owner(int32_t seq) const {
    return seq >= 0 && seq < (int32_t) seqs.size() && seqs[seq].open ? seqs[seq].owner : -1;
}

uint32_t common_spec_verify_rounds::attempted(int32_t seq) const {
    return seq >= 0 && seq < (int32_t) seqs.size() && seqs[seq].open ? seqs[seq].attempted : 0;
}

int32_t common_spec_verify_rounds::m(int32_t seq) const {
    return seq >= 0 && seq < (int32_t) seqs.size() && seqs[seq].open ? seqs[seq].m : -1;
}

//
// store key
//

std::string common_spec_verify_cost_env_key() {
#ifdef _WIN32
    char ** env = _environ;
#else
    char ** env = environ;
#endif
    std::vector<std::string> parts;
    for (char ** e = env; e != nullptr && *e != nullptr; ++e) {
        const std::string kv = *e;
        auto starts = [&](const char * p) { return kv.rfind(p, 0) == 0; };
        const bool take = starts("GGML_") || starts("RANMA_") ||
                          (starts("LLAMA_") && !starts("LLAMA_SPEC_") && !starts("LLAMA_SERVER_") &&
                           !starts("LLAMA_ARG_") && !starts("LLAMA_LOG") && !starts("LLAMA_CACHE"));
        if (take) {
            parts.push_back(kv);
        }
    }
    std::sort(parts.begin(), parts.end());
    std::string out;
    for (const auto & p : parts) {
        if (!out.empty()) {
            out += ' ';
        }
        out += p;
    }
    return out;
}
