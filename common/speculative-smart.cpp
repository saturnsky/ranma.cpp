#include "speculative-smart.h"

#include "log.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define SMT_INF(fmt, ...) LOG_INF("spec-smart: " fmt, __VA_ARGS__)
#define SMT_WRN(fmt, ...) LOG_WRN("spec-smart: " fmt, __VA_ARGS__)
#define SMT_ERR(fmt, ...) LOG_ERR("spec-smart: " fmt, __VA_ARGS__)

static const char * SMART_STORE_MAGIC = "ranma-spec-smart 6";

//
// round log (--spec-smart-log): one JSON object per line, buffered, written at the end of a request and at exit
//

struct common_spec_smart_log {
    FILE *      f = nullptr;
    std::string buf;
    int64_t     n_round = 0; // rounds of this process
    int64_t     n_req   = 0; // requests of this process

    ~common_spec_smart_log() {
        flush();
        if (f) {
            fclose(f);
        }
    }

    void put(const std::string & line) {
        buf += line;
        buf += '\n';
        if (buf.size() >= (1u << 20)) {
            flush();
        }
    }

    void flush() {
        if (f && !buf.empty()) {
            fwrite(buf.data(), 1, buf.size(), f);
            fflush(f);
        }
        buf.clear();
    }
};

static void smart_json_num(std::string & out, double v, const char * fmt = "%.6g") {
    if (!std::isfinite(v)) {
        out += "null";
        return;
    }
    char b[64];
    snprintf(b, sizeof(b), fmt, v);
    out += b;
}

template <typename T>
static void smart_json_arr(std::string & out, const std::vector<T> & v, const char * fmt = "%.6g") {
    out += '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        smart_json_num(out, (double) v[i], fmt);
    }
    out += ']';
}

common_spec_smart::common_spec_smart(const common_spec_smart_config & cfg_in, int32_t n_seq) : cfg(cfg_in) {
    cfg.n_max  = std::max<int32_t>(0, cfg.n_max);
    cfg.n_bins = std::max<int32_t>(1, cfg.n_bins);
    decay       = std::pow(0.5, 1.0 / std::max(1.0, cfg.half_life));
    level_decay = std::pow(0.5, 1.0 / std::max(1.0, cfg.level_half_life));

    shape_cells.assign(cfg.n_max + 1, {});
    shape_over.assign(cfg.n_max + 1, 0.0);
    cal_cells.assign((size_t) N_GROUPS * cfg.n_bins, {});
    qbar_cells.assign(std::min<int32_t>(cfg.n_max + 1, N_QBAR_MAX), {});
    cfg.lookahead_mix = std::clamp(cfg.lookahead_mix, 0.0, 1.0);
    cfg.conf_cap = cfg.conf_cap > 0.0 ? std::max(cfg.conf_cap, 1.0) : 0.0;
    cost_n_run.assign(cfg.n_max + 1, 0);
    used_run.assign(cfg.n_max + 1, 0);
    cfg.tconf_k = std::clamp<int32_t>(cfg.tconf_k, 1, common_spec_tconf::K_MAX);
    tconf_model.reset(cfg.tconf_k);

    seqs.resize(std::max<int32_t>(1, n_seq));
    for (auto & st : seqs) {
        st.n_k.assign(cfg.n_max + 1, 0);
        st.obs_sum_k.assign(cfg.n_max + 1, 0.0);
        st.obs_n_k.assign(cfg.n_max + 1, 0);
    }

    if (!cfg.store_path.empty()) {
        std::string why;
        if (load(cfg.store_path, &why)) {
            SMT_INF("loaded %s (confidence capped at %.0f observations)\n", cfg.store_path.c_str(), cfg.cap_load);
        } else {
            SMT_INF("store %s not used: %s\n", cfg.store_path.c_str(), why.c_str());
        }
    }

    if (!cfg.log_path.empty()) {
        FILE * f = fopen(cfg.log_path.c_str(), "ab");
        if (f == nullptr) {
            SMT_WRN("cannot open the round log %s, no round log\n", cfg.log_path.c_str());
        } else {
            log = std::make_unique<common_spec_smart_log>();
            log->f = f;
            std::string l = "{\"ev\":\"start\",\"n_max\":" + std::to_string(cfg.n_max) +
                            ",\"warm\":" + (store_loaded ? "true" : "false") + ",\"half_life\":";
            smart_json_num(l, cfg.half_life);
            l += ",\"level_half_life\":";
            smart_json_num(l, cfg.level_half_life);
            l += ",\"reprobe_n\":";
            smart_json_num(l, cfg.reprobe_n);
            l += ",\"reprobe_every\":" + std::to_string(cfg.reprobe_every) + ",\"cost_graph\":" +
                 (cfg.cost_graph ? "true" : "false") + ",\"lend_skip\":" + std::to_string(cfg.lend_skip) + ",\"conf_cap\":";
            smart_json_num(l, cfg.conf_cap);
            l += ",\"lookahead_mix\":";
            smart_json_num(l, cfg.lookahead_mix);
            l += ",\"tconf\":";
            l += cfg.tconf ? std::to_string(cfg.tconf_k) : "0";
            l += "}";
            log->put(l);
            log->flush();
            SMT_INF("round log %s\n", cfg.log_path.c_str());
        }
    }

    SMT_INF("n_max = %d, %s, half-life %.0f rounds, level half-life %.0f rounds, extension to a width below %.1f "
            "observations at most every %d rounds, wait confidence cap %.0f (0: none), lookahead %.2f q(last) + %.2f "
            "qbar(position), graph builds and captures %s, rounds of a decode pool lend not observed: %d\n",
            cfg.n_max, store_loaded ? "warm" : "cold", cfg.half_life, cfg.level_half_life,
            cfg.reprobe_n, cfg.reprobe_every, cfg.conf_cap, cfg.lookahead_mix, 1.0 - cfg.lookahead_mix,
            cfg.cost_graph ? "taken out of the times" : "counted", cfg.lend_skip);
    if (cfg.tconf) {
        SMT_INF("trigram confidence: the acceptance of a draft token from its drafter probability and the request's "
                "trigram counts, %d candidate%s per step%s\n", cfg.tconf_k, cfg.tconf_k > 1 ? "s" : "",
                cfg.tconf_k > 1 ? " (the most likely accepted one is drafted)" : "");
    }
}

common_spec_smart::~common_spec_smart() {
    if (log) {
        for (int32_t s = 0; s < (int32_t) seqs.size(); ++s) {
            if (seqs[s].active) {
                log_round(s, "dropped", -1.0); // the last round of the last request: no verification time
            }
        }
        log.reset();
    }
    if (!cfg.store_path.empty() && n_closed > n_closed_saved) {
        if (save(cfg.store_path)) {
            SMT_INF("saved %s\n", cfg.store_path.c_str());
        }
    }
    tprof_put_coef();
    tprof.reset(); // waits for a running save of the profile and saves it
}

int32_t common_spec_smart::bin(float p) const {
    const int32_t b = (int32_t) std::floor((double) p * cfg.n_bins);
    return std::clamp<int32_t>(b, 0, cfg.n_bins - 1);
}

double common_spec_smart::q(int32_t i, float p) const {
    const int32_t g = std::min<int32_t>(std::max<int32_t>(i, 0), N_GROUPS - 1);
    const int32_t b = bin(p);
    const auto &  c = cal_cells[(size_t) g * cfg.n_bins + b];
    const double mid = (b + 0.5) / cfg.n_bins;
    const double n   = conf(c);
    return (n * c.mean + cfg.prior_n * mid) / (n + cfg.prior_n);
}

double common_spec_smart::qbar(int32_t i) const {
    const int32_t j = std::min<int32_t>(std::max<int32_t>(i, 0), (int32_t) qbar_cells.size() - 1);
    const auto &  c = qbar_cells[j];
    const double  n = conf(c);
    return (n * c.mean + cfg.prior_n * 0.5) / (n + cfg.prior_n);
}

double common_spec_smart::lookahead(int32_t j, const float * p, int32_t n_drafted, const double * qd) const {
    if (n_drafted <= 0 || cfg.lookahead_mix == 0.0) {
        return qbar(j);
    }
    // the acceptance of the last drafted position predicts the next ones; the position mean keeps the fall with depth
    const double a = cfg.lookahead_mix;
    const double ql = qd != nullptr ? qd[n_drafted - 1] : q(n_drafted - 1, p[n_drafted - 1]);
    return a * ql + (1.0 - a) * qbar(j);
}

double common_spec_smart::draft_step() const {
    return draft_cell.mean;
}

std::vector<double> common_spec_smart::cost_table() const {
    const int32_t N = cfg.n_max + 1;

    // shape values; a width without one takes the line through the two nearest widths with one (the narrower one
    // first on a tie)
    std::vector<double>  val(N, 0.0);
    std::vector<uint8_t> known(N, 0);
    for (int32_t k = 0; k < N; ++k) {
        if (shape_cells[k].n > 0.0) {
            val[k]   = shape_cells[k].mean;
            known[k] = 1;
        }
    }
    std::vector<double> out = val;
    for (int32_t k = 0; k < N; ++k) {
        if (known[k]) {
            continue;
        }
        int32_t j[2];
        int32_t nj = 0;
        for (int32_t d = 1; d < N && nj < 2; ++d) {
            if (k - d >= 0 && known[k - d]) {
                j[nj++] = k - d;
            }
            if (nj < 2 && k + d < N && known[k + d]) {
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
    return out;
}

int32_t common_spec_smart::decide(const float * p, int32_t n_drafted, int32_t n_max_seq,
                                  common_spec_smart_diag * diag, const double * qd) const {
    n_max_seq = std::min(n_max_seq, cfg.n_max);
    const std::vector<double> C = cost_table();
    const double d = draft_step();
    const double D = n_drafted * d;

    if (diag) {
        diag->q.clear();
        diag->la.clear();
    }

    // stop now, keeping k = 0..n_drafted tokens
    double  best   = -1.0;
    int32_t best_k = 0;
    double  best_A = 0.0;
    double  A = 0.0; // expected accepted tokens of the kept prefix
    double  S = 1.0; // probability that the whole prefix is accepted
    for (int32_t k = 0; k <= n_drafted; ++k) {
        if (k > 0) {
            const double qk = qd != nullptr ? qd[k - 1] : q(k - 1, p[k - 1]);
            S *= qk;
            A += S;
            if (diag) {
                diag->q.push_back(qk);
            }
        }
        const double T = (A + 1.0) / std::max(D + C[k], 1.0);
        if (T > best) {
            best   = T;
            best_k = k;
            best_A = A;
        }
    }

    // one more step, then the best stopping depth with the predicted acceptance of the positions not drafted yet
    double cont = -1.0;
    double E  = A;
    double Sl = S;
    for (int32_t m = 1; n_drafted + m <= n_max_seq; ++m) {
        const double la = lookahead(n_drafted + m - 1, p, n_drafted, qd);
        Sl *= la;
        E  += Sl;
        const double T = (E + 1.0) / std::max(D + m * d + C[n_drafted + m], 1.0);
        cont = std::max(cont, T);
        if (diag) {
            diag->la.push_back(la);
        }
    }

    const int32_t res = cont > best ? -1 : best_k;
    if (diag) {
        diag->valid      = true;
        diag->k          = res;
        diag->exp_acc    = best_A;
        diag->t_stop     = best;
        diag->t_cont     = cont;
        diag->draft_pred = d;
        diag->cost       = C;
    }
    return res;
}

bool common_spec_smart::level_source(int32_t k) const {
    // widths with a confident shape only; when there is none, every width with a shape
    if (conf(shape_cells[k]) >= cfg.reprobe_n) {
        return true;
    }
    for (const auto & c : shape_cells) {
        if (conf(c) >= cfg.reprobe_n) {
            return false;
        }
    }
    return true;
}

void common_spec_smart::observe_cost(int32_t k, double us) {
    if (k < 0 || k > cfg.n_max || !(us >= 0.0)) {
        return;
    }
    const double x = std::max(us, 1.0); // the ratios need a time above 0
    auto & S = shape_cells[k];
    const bool has_shape = S.n > 0.0 && S.mean > 0.0;

    // the level: the first observation sets it; later x / the shape of the width, both before this observation
    double L = level_cell.mean;
    if (!(L > 0.0)) {
        level_cell = {};
        level_cell.add(has_shape ? x / S.mean : x, level_decay, n_closed);
        L = level_cell.mean;
    } else if (has_shape && level_source(k)) {
        level_cell.add(x / S.mean, level_decay, n_closed);
    }

    // the shape of the width: x / the level before this observation. The wait confidence takes the same observation
    // but stays at most conf_cap, so that a width used much and then left waits for an extension again after a bounded
    // time; the mean and its confidence do not see the cap
    const double wc = wait_conf(k);
    // a waiting width: its old mean counts only with the wait confidence (below reprobe_n), so that a few extensions
    // move a mean that stayed high long after the width was left. The weighted variance does not change when all old
    // weights shrink by the same factor; only the confidence is cut. The level above took this observation as before.
    wr_last = false;
    if (cfg.wait_reset && waiting(k)) {
        const double n0 = S.n_at(n_closed, decay);
        if (wc < n0) {
            S.n = wc;
            S.t = std::max(S.t, n_closed);
            wr_last     = true;
            wr_n_before = n0;
            wr_n_after  = wc;
            n_wait_reset++;
        }
    }
    S.add(x / L, decay, n_closed);
    shape_over[k] = cfg.conf_cap > 0.0 ? std::max(0.0, S.n - std::min(wc + 1.0, cfg.conf_cap)) : 0.0;
    cost_n_run[k]++;
}

void common_spec_smart::observe_acceptance(const float * p, int32_t k, int32_t n_accepted) {
    // verified positions only: 0..n_accepted-1 accepted, n_accepted rejected (when inside the draft)
    const int32_t last = std::min(n_accepted, k - 1);
    for (int32_t j = 0; j <= last; ++j) {
        const double  y = j < n_accepted ? 1.0 : 0.0;
        const int32_t g = std::min<int32_t>(j, N_GROUPS - 1);
        cal_cells[(size_t) g * cfg.n_bins + bin(p[j])].add(y, decay, n_closed);
        qbar_cells[std::min<int32_t>(j, (int32_t) qbar_cells.size() - 1)].add(y, decay, n_closed);
    }
}

void common_spec_smart::shrink_calibration(double cap) {
    for (auto & c : cal_cells) {
        c.cap(cap, decay, n_closed);
    }
    for (auto & c : qbar_cells) {
        c.cap(cap, decay, n_closed);
    }
}

void common_spec_smart::add_draft_step(double us, double graph_us) {
    if (us >= 0.0) {
        // a graph built or captured inside the step is a cost of the graph runtime, not of the draft step
        const double x = cfg.cost_graph ? std::max(us - std::max(graph_us, 0.0), 0.0) : us;
        draft_cell.add(x, decay, n_closed);
    }
    if (log) {
        // the step belongs to every sequence that drafted in it
        for (auto & st : seqs) {
            if (st.drafting && (int32_t) st.p.size() > st.log_steps_timed) {
                st.log_steps_timed = (int32_t) st.p.size();
                st.log_draft_us += std::max(us, 0.0);
            }
        }
    }
}

void common_spec_smart::request_begin(int32_t seq) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    if (log) {
        if (st.active) {
            log_round(seq, "dropped", -1.0); // the last round of the previous request: no verification time
        }
        st.log_req = ++log->n_req;
        log->put("{\"ev\":\"req\",\"req\":" + std::to_string(st.log_req) + ",\"seq\":" + std::to_string(seq) +
                 ",\"round\":" + std::to_string(log->n_round) + ",\"n_closed\":" + std::to_string(n_closed) + "}");
        log->flush();
    }
    st.drafting = false;
    st.active   = false; // the last round of the previous request ended with it; its time includes the idle gap
    if (cfg.tconf) {
        st.tri.clear(); // a table per request (its prompt and generated tokens)
        st.tc_pos.clear();
        st.tc_pend_valid = false;
    }
    st.n_rounds = 0;
    st.n_draft_tok = 0;
    st.n_acc_tok = 0;
    st.n_extend = 0;
    std::fill(st.n_k.begin(), st.n_k.end(), 0);
    std::fill(st.obs_sum_k.begin(), st.obs_sum_k.end(), 0.0);
    std::fill(st.obs_n_k.begin(), st.obs_n_k.end(), 0);

    shrink_calibration(cfg.cap_request);
}

void common_spec_smart::close_round(int32_t seq, int64_t t_us, const common_spec_smart_pool * pool,
                                    const common_spec_smart_graphs * graphs) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    // round log: the counts here end the verification of the pending round and start the draft of the next one
    const common_spec_smart_pool now = pool != nullptr ? *pool : common_spec_smart_pool();
    st.pool_start = st.pool_next;
    st.pool_end   = now;
    st.pool_end.valid = now.valid && st.pool_start.valid && st.pool_drafted.valid;
    st.pool_next  = now;
    st.graph_start = st.graph_next;
    st.graph_end   = graphs != nullptr ? *graphs : common_spec_smart_graphs();
    st.graph_next  = st.graph_end;
    if (now.valid) {
        if (now.lends != pool_lends) {
            pool_since_lend = 0;
        } else if (pool_since_lend >= 0 && st.active) {
            pool_since_lend++;
        }
        pool_lends = now.lends;
    }
    if (!st.active) {
        return;
    }
    st.active = false;

    // a draft that was never verified (the slot stopped) is not a round
    if (st.k > 0 && !st.has_acc) {
        if (log) {
            log_round(seq, "unverified", -1.0);
        }
        return;
    }

    // the first closed round of a width in this process builds its graphs: not observed
    st.timed = used_run[st.k] != 0;
    used_run[st.k] = 1;

    const double us = (double) (t_us - st.t_drafted);
    wr_last = false;
    // the verification time without the graph builds and captures of the target and the draft context in it
    double obs = us;
    if (cfg.cost_graph && graphs != nullptr) {
        auto graph_us = [](const common_spec_smart_graph & a, const common_spec_smart_graph & b) {
            return a.valid && b.valid ? (double) ((b.build_us - a.build_us) + (b.capture_us - a.capture_us)) : 0.0;
        };
        obs = std::max(1.0, us - graph_us(st.graph_drafted.tgt, graphs->tgt) - graph_us(st.graph_drafted.dft, graphs->dft));
    }
    // the round of a lend of the decode pool and the next ones fill its positions: not a time of the width
    const bool lend_round = cfg.lend_skip > 0 && pool != nullptr && pool->valid && pool_since_lend >= 0 &&
                            pool_since_lend < cfg.lend_skip;
    st.log_obs_us = -1.0;
    st.log_skip   = !st.timed ? "untimed" : lend_round ? "lend" : nullptr;
    if (st.timed && !lend_round) {
        observe_cost(st.k, obs);
        st.log_obs_us = obs;
        st.obs_sum_k[st.k] += us;
        st.obs_n_k[st.k]++;
    }
    if (log) {
        log_round(seq, "closed", us); // after the observation: it records a cut of the old mean (wait_reset)
    }

    n_closed++;
    if (!cfg.store_path.empty() && cfg.save_every > 0 && n_closed - n_closed_saved >= cfg.save_every) {
        if (save(cfg.store_path)) {
            n_closed_saved = n_closed;
        } else if (!save_failed) {
            save_failed = true;
            SMT_WRN("cannot write %s\n", cfg.store_path.c_str());
        }
    }
}

bool common_spec_smart::begin_round(int32_t seq, int32_t n_max_seq) {
    auto & st = seqs[seq];
    st.drafting  = true;
    st.active    = false;
    st.has_acc   = false;
    st.p.clear();
    st.qd.clear();
    st.n_max_seq = std::max<int32_t>(0, std::min(n_max_seq, cfg.n_max));
    st.forced    = -1;
    st.timed     = true;
    st.k_nat     = -1;
    st.ext_step  = false;
    if (log) {
        st.log_forced      = 0;
        st.log_steps       = 0;
        st.log_acc         = -1;
        st.log_steps_timed = 0;
        st.log_draft_us    = 0.0;
        st.log_p.clear();
        st.log_la_next.clear();
        st.log_diag = {};
    }

    st.n_lim = std::min(st.n_max_seq, frontier());
    if (st.n_max_seq == 0) {
        return false;
    }

    // the decision does not draft beyond the widest width found so far; the width after it is found by an extension
    const int32_t r = decide(nullptr, 0, st.n_lim, log ? &st.log_diag : nullptr);
    if (log && r < 0 && !st.log_diag.la.empty()) {
        st.log_la_next.push_back(st.log_diag.la[0]);
    }
    if (r == 0 && can_extend(1, st.n_max_seq)) {
        start_extend(seq, 0, true); // no draft wanted and width 1 waits: draft one token
        return true;
    }
    return r < 0;
}

int32_t common_spec_smart::frontier() const {
    int32_t f = 0;
    for (int32_t k = 1; k <= cfg.n_max; ++k) {
        if (!used_run[k] && !(shape_cells[k].n > 0.0)) {
            break;
        }
        f = k;
    }
    return f;
}

double common_spec_smart::wait_conf(int32_t w) const {
    common_spec_smart_cell c = shape_cells[w];
    if (cfg.conf_cap > 0.0) {
        c.n = std::max(0.0, c.n - shape_over[w]);
    }
    return conf(c);
}

bool common_spec_smart::waiting(int32_t w) const {
    return w >= 0 && w <= cfg.n_max && wait_conf(w) < cfg.reprobe_n;
}

bool common_spec_smart::can_extend(int32_t w, int32_t n_max_seq) const {
    // only the width right after the one the decision stopped at, inside the limit of the request, at most once every
    // reprobe_every closed rounds
    return cfg.reprobe_every > 0 && w >= 1 && w <= std::min(n_max_seq, cfg.n_max) &&
           n_closed - last_extend >= cfg.reprobe_every && waiting(w);
}

void common_spec_smart::start_extend(int32_t seq, int32_t k_nat, bool ext_step) {
    auto & st = seqs[seq];
    st.k_nat      = k_nat;
    st.ext_step   = ext_step;
    st.forced     = ext_step ? k_nat + 1 : -1;
    st.log_forced = 3;
    last_extend   = n_closed;
    n_extend++;
    st.n_extend++;
}

int32_t common_spec_smart::after_step(int32_t seq, float p_top, double q_direct) {
    auto & st = seqs[seq];
    st.p.push_back(p_top);
    if (q_direct >= 0.0) {
        st.qd.push_back(std::clamp(q_direct, 0.01, 0.99));
    }
    const int32_t i = (int32_t) st.p.size();
    const double * qd = (int32_t) st.qd.size() == i ? st.qd.data() : nullptr;

    if (st.forced >= 0) {
        return i >= st.forced || i >= st.n_max_seq ? i : -1; // the extra step of an extension
    }

    const int32_t r = decide(st.p.data(), i, st.n_lim, log ? &st.log_diag : nullptr, qd);
    if (log && r < 0 && !st.log_diag.la.empty()) {
        st.log_la_next.push_back(st.log_diag.la[0]);
    }
    if (r >= 0 && can_extend(r + 1, st.n_max_seq)) {
        // the decision stops at r and width r + 1 waits: verify r + 1, keeping a token the decision dropped or
        // drafting one more
        const bool step = r + 1 > i;
        start_extend(seq, r, step);
        return step ? -1 : r + 1;
    }
    return r;
}

void common_spec_smart::drafted(int32_t seq, int32_t k, int64_t t_us, const common_spec_smart_pool * pool,
                                const common_spec_smart_graphs * graphs) {
    auto & st = seqs[seq];
    st.pool_drafted  = pool != nullptr ? *pool : common_spec_smart_pool();
    st.graph_drafted = graphs != nullptr ? *graphs : common_spec_smart_graphs();
    k = std::clamp<int32_t>(k, 0, std::min<int32_t>(cfg.n_max, (int32_t) st.p.size()));
    st.drafting  = false;
    st.active    = true;
    st.has_acc   = false;
    st.k         = k;
    st.t_drafted = t_us;
    if (log) {
        st.log_steps = (int32_t) st.p.size();
        st.log_p     = st.p;
        st.log_round = ++log->n_round;
        if (st.log_forced == 3 && st.log_diag.valid) {
            // extension: the last decision (stop at k_nat) is kept; q and exp_acc are those of the verified prefix k
            auto & dg = st.log_diag;
            dg.q.clear();
            dg.exp_acc = 0.0;
            double S = 1.0;
            const bool has_qd = st.qd.size() == st.p.size();
            for (int32_t j = 0; j < (int32_t) st.p.size(); ++j) {
                dg.q.push_back(has_qd ? st.qd[j] : q(j, st.p[j]));
                if (j < k) {
                    S *= dg.q.back();
                    dg.exp_acc += S;
                }
            }
        }
    }
    st.p.resize(k);
    if (st.qd.size() > (size_t) k) {
        st.qd.resize(k);
    }
    if (st.tc_pos.size() > (size_t) k) {
        st.tc_pos.resize(k); // the dropped tail is not verified
    }

    st.n_rounds++;
    st.n_draft_tok += k;
    st.n_k[k]++;
}

void common_spec_smart::accepted(int32_t seq, int32_t n_accepted) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    if (!st.active || st.has_acc || st.k == 0) {
        return;
    }
    n_accepted = std::clamp<int32_t>(n_accepted, 0, st.k);
    if (log) {
        st.log_acc = n_accepted;
    }
    observe_acceptance(st.p.data(), st.k, n_accepted);
    tconf_accepted(seq, n_accepted);
    st.has_acc = true;
    st.n_acc_tok += n_accepted;
}

void common_spec_smart::log_flush() const {
    if (log) {
        log->flush();
    }
}

void common_spec_smart::log_round(int32_t seq, const char * end, double verify_us) {
    if (!log) {
        return;
    }
    const auto & st = seqs[seq];
    const auto & dg = st.log_diag;

    std::string l;
    l.reserve(1024);
    l += "{\"ev\":\"round\",\"req\":" + std::to_string(st.log_req) + ",\"seq\":" + std::to_string(seq) +
         ",\"round\":" + std::to_string(st.log_round) + ",\"forced\":\"";
    l += st.log_forced == 3 ? "extend" : "none";
    l += "\",\"timed\":";
    l += st.timed ? "true" : "false";
    l += ",\"n_max_seq\":" + std::to_string(st.n_max_seq) + ",\"n_lim\":" + std::to_string(st.n_lim) +
         ",\"steps\":" + std::to_string(st.log_steps) +
         ",\"k\":" + std::to_string(st.k) + ",\"k_nat\":" + std::to_string(st.log_forced == 3 ? st.k_nat : st.k) +
         ",\"ext_step\":" + (st.log_forced == 3 && st.ext_step ? "true" : "false") + ",\"p\":";
    smart_json_arr(l, st.log_p);
    if (cfg.tconf) {
        // trigram confidence: the drafter rank of the drafted candidate and the trigram observations of every
        // verified draft step
        std::vector<int32_t> ci, cn;
        for (const auto & ps : st.tc_pos) {
            ci.push_back(ps.chosen);
            cn.push_back(ps.N);
        }
        l += ",\"tc_i\":";
        smart_json_arr(l, ci, "%g");
        l += ",\"tc_n\":";
        smart_json_arr(l, cn, "%g");
        // per verified draft step and source (RA, RU, PM, PX, PU): the count of the drafted token and the observations
        // of the context
        l += ",\"tc_src\":[";
        for (size_t j = 0; j < st.tc_pos.size(); ++j) {
            l += j > 0 ? ",[" : "[";
            for (int32_t d = 0; d < 2 * common_spec_tconf::N_SRC; ++d) {
                if (d > 0) {
                    l += ',';
                }
                smart_json_num(l, st.tc_pos[j].src[d], "%.4g");
            }
            l += ']';
        }
        l += ']';
    }
    l += ",\"q\":";
    smart_json_arr(l, dg.q);
    l += ",\"exp_acc\":";
    smart_json_num(l, dg.valid ? dg.exp_acc : NAN);
    l += ",\"la\":";
    if (dg.valid) {
        l += "{\"stop\":";
        smart_json_num(l, dg.t_stop * 1e6); // tokens / s
        l += ",\"cont\":";
        smart_json_num(l, dg.t_cont >= 0.0 ? dg.t_cont * 1e6 : NAN);
        l += ",\"q\":";
        smart_json_arr(l, dg.la);
        l += "}";
    } else {
        l += "null";
    }
    l += ",\"la_next\":";
    smart_json_arr(l, st.log_la_next);
    l += ",\"cost_pred\":";
    smart_json_num(l, dg.valid && st.k < (int32_t) dg.cost.size() ? dg.cost[st.k] : NAN, "%.1f");
    l += ",\"draft_pred\":";
    smart_json_num(l, dg.valid ? dg.draft_pred : NAN, "%.1f");
    l += ",\"cost_table\":";
    smart_json_arr(l, dg.cost, "%.1f");
    l += ",\"draft_us\":";
    smart_json_num(l, st.log_draft_us, "%.1f");
    l += ",\"verify_us\":";
    smart_json_num(l, verify_us >= 0.0 ? verify_us : NAN, "%.1f");
    if (std::string(end) == "closed") {
        // the time the cost model took (without graph builds and captures), or why it took none
        l += ",\"obs_us\":";
        smart_json_num(l, st.log_obs_us >= 0.0 ? st.log_obs_us : NAN, "%.1f");
        l += ",\"obs_skip\":";
        l += st.log_skip != nullptr ? std::string("\"") + st.log_skip + "\"" : std::string("null");
    }
    l += ",\"accepted\":";
    if (st.k == 0) {
        l += "0";
    } else if (st.log_acc >= 0) {
        l += std::to_string(st.log_acc);
    } else {
        l += "null";
    }
    // the decode pool in the draft steps (round start -> drafted) and in the verification (drafted -> close)
    if (std::string(end) == "closed" && st.pool_end.valid) {
        const auto & a = st.pool_start;
        const auto & b = st.pool_drafted;
        const auto & c = st.pool_end;
        auto part = [&](const common_spec_smart_pool & x, const common_spec_smart_pool & y) {
            l += "{\"fills\":" + std::to_string(y.fills - x.fills) + ",\"fill_mib\":";
            smart_json_num(l, (double) (y.fill_bytes - x.fill_bytes) / 1048576.0, "%.1f");
            l += ",\"hits\":" + std::to_string(y.hits - x.hits) + ",\"lends\":" + std::to_string(y.lends - x.lends) +
                 ",\"reclaims\":" + std::to_string(y.reclaims - x.reclaims) + "}";
        };
        l += ",\"pool\":{\"draft\":";
        part(a, b);
        l += ",\"verify\":";
        part(b, c);
        l += ",\"since_lend\":" + std::to_string(pool_since_lend) + ",\"lend\":" + std::to_string(c.lends) + "}";
    }
    // the graphs of the target and the draft context in the draft steps and in the verification
    if (std::string(end) == "closed") {
        auto gpart = [&](const common_spec_smart_graph & x, const common_spec_smart_graph & y) {
            if (!x.valid || !y.valid) {
                l += "null";
                return;
            }
            l += "{\"calls\":" + std::to_string(y.calls - x.calls) + ",\"builds\":" + std::to_string(y.builds - x.builds) +
                 ",\"build_ms\":";
            smart_json_num(l, (double) (y.build_us - x.build_us) / 1000.0, "%.3f");
            l += ",\"nt\":";
            l += y.builds > x.builds ? std::to_string(y.last_nt) : std::string("null");
            if (y.calls > x.calls) {
                // the last graph of the part: how far it reached into the compute buffer, and its highest position
                l += ",\"extent_mib\":";
                smart_json_num(l, (double) y.last_extent / 1048576.0, "%.2f");
                l += ",\"pos\":" + std::to_string(y.last_pos);
            }
            if (x.backend && y.backend) {
                l += ",\"captures\":" + std::to_string(y.captures - x.captures) + ",\"capture_ms\":";
                smart_json_num(l, (double) (y.capture_us - x.capture_us) / 1000.0, "%.3f");
                l += ",\"direct\":" + std::to_string(y.direct - x.direct) + ",\"direct_ms\":";
                smart_json_num(l, (double) (y.direct_us - x.direct_us) / 1000.0, "%.3f");
            }
            l += "}";
        };
        const auto & a = st.graph_start;
        const auto & b = st.graph_drafted;
        const auto & c = st.graph_end;
        if (a.tgt.valid && b.tgt.valid && c.tgt.valid) {
            l += ",\"graph\":{\"draft\":{\"tgt\":";
            gpart(a.tgt, b.tgt);
            l += ",\"dft\":";
            gpart(a.dft, b.dft);
            l += "},\"verify\":{\"tgt\":";
            gpart(b.tgt, c.tgt);
            l += ",\"dft\":";
            gpart(b.dft, c.dft);
            l += "}}";
        }
    }
    l += ",\"wait_reset\":";
    const bool wr = std::string(end) == "closed" && wr_last;
    l += wr ? "true" : "false";
    if (wr) {
        l += ",\"wr_n\":[";
        smart_json_num(l, wr_n_before, "%.2f");
        l += ',';
        smart_json_num(l, wr_n_after, "%.2f");
        l += ']';
    }
    l += ",\"end\":\"";
    l += end;
    l += "\"}";
    log->put(l);
}

std::string common_spec_smart::request_summary(int32_t seq) const {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return "";
    }
    const auto & st = seqs[seq];
    if (st.n_rounds == 0) {
        return "";
    }

    const std::vector<double> C = cost_table();

    int64_t sum_k = 0;
    for (int32_t k = 0; k <= cfg.n_max; ++k) {
        sum_k += (int64_t) k * st.n_k[k];
    }

    char buf[256];
    std::string s;
    snprintf(buf, sizeof(buf), "rounds %lld, mean k %.2f, acceptance %.3f (%lld/%lld), draft step %.2f ms, extensions %lld (total %lld), width share / predicted / observed ms:",
             (long long) st.n_rounds, (double) sum_k / (double) st.n_rounds,
             st.n_draft_tok > 0 ? (double) st.n_acc_tok / (double) st.n_draft_tok : 0.0,
             (long long) st.n_acc_tok, (long long) st.n_draft_tok, draft_step() / 1000.0,
             (long long) st.n_extend, (long long) n_extend);
    s += buf;
    for (int32_t k = 0; k <= cfg.n_max; ++k) {
        if (st.n_k[k] == 0) {
            continue;
        }
        if (st.obs_n_k[k] > 0) {
            snprintf(buf, sizeof(buf), " k%d %.1f%% %.2f/%.2f", k, 100.0 * st.n_k[k] / st.n_rounds, C[k] / 1000.0,
                     st.obs_sum_k[k] / st.obs_n_k[k] / 1000.0);
        } else {
            snprintf(buf, sizeof(buf), " k%d %.1f%% %.2f/-", k, 100.0 * st.n_k[k] / st.n_rounds, C[k] / 1000.0);
        }
        s += buf;
    }
    // the level (ms, confidence) and the shape of every width (mean ratio, confidence)
    snprintf(buf, sizeof(buf), ", level %.2f ms n %.1f, shape (confidence):", level_cell.mean / 1000.0, level_conf());
    s += buf;
    for (int32_t k = 0; k <= cfg.n_max; ++k) {
        snprintf(buf, sizeof(buf), " k%d %.3f (%.1f)", k, shape_cells[k].mean, conf(shape_cells[k]));
        s += buf;
    }
    snprintf(buf, sizeof(buf), ", draft step mean %.3f sd %.3f n %.1f", draft_cell.mean / 1000.0,
             std::sqrt(draft_cell.var) / 1000.0, conf(draft_cell));
    s += buf;
    // mean acceptance per position (with the prior, as the lookahead uses it) and its confidence, the first 8
    snprintf(buf, sizeof(buf), ", lookahead mix %.2f, qbar:", cfg.lookahead_mix);
    s += buf;
    for (int32_t j = 0; j < std::min<int32_t>(8, (int32_t) qbar_cells.size()); ++j) {
        snprintf(buf, sizeof(buf), " p%d %.3f n %.1f", j, qbar(j), conf(qbar_cells[j]));
        s += buf;
    }
    if (cfg.tconf) {
        snprintf(buf, sizeof(buf), ", trigram confidence k %d, updates %lld, other candidate drafted %lld, b",
                 cfg.tconf_k, (long long) tc_updates, (long long) tc_invert);
        s += buf;
        for (int32_t d = 0; d < common_spec_tconf::DIM; ++d) {
            snprintf(buf, sizeof(buf), " %.3f", tconf_model.b[d]);
            s += buf;
        }
    }
    return s;
}

//
// trigram confidence
//

void common_spec_trigram::clear() {
    map.clear();
    nodes.clear();
    n_ins     = 0;
    tok_prev  = -1;
    tok_last  = -1;
    role_last = common_spec_roles::OTHER;
}

static uint64_t tri_key(int32_t a, int32_t b) {
    return ((uint64_t) (uint32_t) a << 32) | (uint32_t) b;
}

void common_spec_trigram::add(int32_t a, int32_t b, int32_t c, uint8_t r) {
    auto it = map.find(tri_key(a, b));
    if (it == map.end()) {
        it = map.emplace(tri_key(a, b), head{ -1, { 0, 0 } }).first;
    }
    head & h = it->second;
    h.n[r]++;
    for (int32_t i = h.first; i >= 0; i = nodes[i].next) {
        if (nodes[i].tok == c) {
            nodes[i].cnt[r]++;
            return;
        }
    }
    node nd{ c, { 0, 0 }, h.first };
    nd.cnt[r] = 1;
    nodes.push_back(nd);
    h.first = (int32_t) nodes.size() - 1;
}

void common_spec_trigram::sync(const int32_t * toks, size_t n, int32_t last, const common_spec_roles * roles,
                               size_t gen_from) {
    const size_t len = n + 1;
    auto at = [&](size_t i) { return i < n ? toks[i] : last; };
    // the stream continues the inserted one when its tokens at the last two inserted positions are the same
    if (n_ins > len || (n_ins >= 1 && at(n_ins - 1) != tok_last) || (n_ins >= 2 && at(n_ins - 2) != tok_prev)) {
        clear();
    }
    if (n_ins == 0 && map.empty()) {
        map.reserve(len);
        nodes.reserve(len);
    }
    const bool markers = roles != nullptr && roles->on();
    uint8_t    role    = role_last;
    for (size_t i = n_ins; i < len; ++i) {
        if (markers) {
            role = roles->step(role, at, i);
        }
        const uint8_t r = i >= gen_from ? common_spec_roles::ASSIST : role;
        if (i >= 2) {
            add(at(i - 2), at(i - 1), at(i), r);
        }
    }
    role_last = role;
    n_ins     = len;
    tok_prev  = len >= 2 ? at(len - 2) : -1;
    tok_last  = at(len - 1);
}

void common_spec_trigram::lookup(int32_t a, int32_t b, const int32_t * toks, int32_t n, int32_t * cnt, int32_t * N) const {
    for (int32_t i = 0; i < 2 * n; ++i) {
        cnt[i] = 0;
    }
    N[0] = N[1] = 0;
    const auto it = map.find(tri_key(a, b));
    if (it == map.end()) {
        return;
    }
    N[0] = it->second.n[0];
    N[1] = it->second.n[1];
    for (int32_t j = it->second.first; j >= 0; j = nodes[j].next) {
        for (int32_t i = 0; i < n; ++i) {
            if (toks[i] == nodes[j].tok) {
                cnt[2 * i + 0] = nodes[j].cnt[0];
                cnt[2 * i + 1] = nodes[j].cnt[1];
            }
        }
    }
}

// static coefficients per k (layout of common_spec_tconf::features, the profile columns 0): IRLS (ridge 0.1) on the
// top-k drafter candidates of every logged chain position of the GLM-5.3-Flash EXL3 3.05bpw MTP top-k log main-v3
// (2026-10-10; 170 requests at fixed width 3, temperature 0), trigram source = the request's own prompt and generated
// tokens split by the role of the next token (after <|assistant|> and the generated tokens: RA, else RU), presets
// rp-ja, rp-en, rp-zh and code (rp-ko left out, so that the roleplay-ko measurements are not fit on). On the rp-ko
// questions >= 10: log-loss of the top-1 positions 0.4960 (k = 1) against 0.4962 without the role split, 0.5085 for the
// calibration cells and 0.591 for the raw probability. The profile columns start at 0 (no profile in the logs): an
// empty profile leaves the acceptance as without it, and the coefficients are learnt online.
static const double TCONF_B[common_spec_tconf::K_MAX][common_spec_tconf::DIM] = {
    { -0.861109, 0.718653, 0.464978, 0.222404, 0.612030, 1.118706, 0, 0, 0, 0, 0, 0, 1.117296, -0.092387 }, // k = 1, 43120 rows
    { -0.784430, 0.645351, 0.535515, 0.538600, 0.721376, 1.154944, 0, 0, 0, 0, 0, 0, 0.714918, -0.025989 }, // k = 2, 86240 rows
    { -0.823403, 0.651749, 0.593044, 0.652753, 0.675563, 1.116444, 0, 0, 0, 0, 0, 0, 0.578160, 0.013571 }, // k = 3, 129360 rows
    { -0.847199, 0.660359, 0.591215, 0.685364, 0.683916, 1.091001, 0, 0, 0, 0, 0, 0, 0.507744, 0.024719 }, // k = 4, 172480 rows
    { -0.860949, 0.667706, 0.612484, 0.725772, 0.690778, 1.130625, 0, 0, 0, 0, 0, 0, 0.448082, 0.028378 }, // k = 5, 215600 rows
    { -0.873029, 0.674453, 0.627535, 0.760495, 0.646373, 1.088258, 0, 0, 0, 0, 0, 0, 0.423837, 0.030459 }, // k = 6, 258720 rows
    { -0.881355, 0.680273, 0.647700, 0.774526, 0.670326, 1.116544, 0, 0, 0, 0, 0, 0, 0.376617, 0.033549 }, // k = 7, 301840 rows
    { -0.888891, 0.686119, 0.647795, 0.780649, 0.674477, 1.111934, 0, 0, 0, 0, 0, 0, 0.355944, 0.036448 }, // k = 8, 344960 rows
    { -0.895308, 0.692392, 0.659313, 0.787064, 0.673222, 1.113728, 0, 0, 0, 0, 0, 0, 0.339553, 0.039674 }, // k = 9, 388080 rows
    { -0.900759, 0.697504, 0.666567, 0.803820, 0.680551, 1.099522, 0, 0, 0, 0, 0, 0, 0.322382, 0.043712 }, // k = 10, 431200 rows
};

static double tconf_lg(double x) {
    x = std::clamp(x, 1e-4, 1.0 - 1e-4);
    return std::log(x / (1.0 - x));
}

void common_spec_tconf::reset(int32_t k) {
    k = std::clamp<int32_t>(k, 1, K_MAX);
    for (int32_t d = 0; d < DIM; ++d) {
        b[d] = TCONF_B[k - 1][d];
        G[d] = 1e-3;
    }
}

void common_spec_tconf::features(double p, const double * n, const double * N, bool mid, bool sw, double * x) {
    x[0] = 1.0;
    x[1] = tconf_lg(p);
    for (int32_t s = 0; s < N_SRC; ++s) {
        double g  = 0.0;
        double f1 = 0.0;
        if (N[s] > 0.0) {
            g  = N[s] / (N[s] + 3.0);
            f1 = tconf_lg((n[s] + 0.5) / (N[s] + 1.0));
        }
        x[2 + 2 * s]     = g * f1;
        x[2 + 2 * s + 1] = g;
    }
    x[X_MID] = mid ? 1.0 : 0.0;
    x[X_SW]  = sw ? 1.0 : 0.0;
}

double common_spec_tconf::q(const double * x) const {
    double z = 0.0;
    for (int32_t d = 0; d < DIM; ++d) {
        z += b[d] * x[d];
    }
    z = std::clamp(z, -30.0, 30.0);
    return 1.0 / (1.0 + std::exp(-z));
}

static const double G_DECAY = std::exp2(-1.0 / common_spec_tconf::G_HALF_LIFE);

void common_spec_tconf::update(const double * X, const double * lab, int32_t n) {
    // AdaGrad on the log-likelihood of the n candidates of one position, the gradient scaled by 1 / sqrt(n)
    double gr[DIM] = {};
    for (int32_t i = 0; i < n; ++i) {
        const double r = lab[i] - q(X + (size_t) i * DIM);
        for (int32_t d = 0; d < DIM; ++d) {
            gr[d] += X[(size_t) i * DIM + d] * r;
        }
    }
    const double sc = 1.0 / std::max(1.0, std::sqrt((double) n));
    for (int32_t d = 0; d < DIM; ++d) {
        gr[d] *= sc;
        G[d]   = G[d] * G_DECAY + gr[d] * gr[d];
        b[d]  += lr * gr[d] / std::sqrt(G[d]);
    }
}

void common_spec_smart::tconf_vocab(std::vector<uint32_t> tail, std::vector<uint8_t> word_first) {
    tc_tail = std::move(tail);
    tc_wf   = std::move(word_first);
}

bool common_spec_smart::tconf_mid(const std::vector<int32_t> & ctx) const {
    // the last (up to) 3 bytes of the context, from its last 6 tokens
    uint8_t by[3];
    int32_t nb = 0;
    for (int32_t t = (int32_t) ctx.size() - 1; t >= 0 && t >= (int32_t) ctx.size() - 6 && nb < 3; --t) {
        const int32_t id = ctx[t];
        if (id < 0 || (size_t) id >= tc_tail.size()) {
            continue;
        }
        const uint32_t v = tc_tail[id];
        const int32_t  c = (int32_t) ((v >> 24) & 3);
        for (int32_t j = 0; j < c && nb < 3; ++j) {
            by[nb++] = (uint8_t) (v >> (8 * j)); // by[0] = the last byte
        }
    }
    for (int32_t k = 1; k <= 3; ++k) {
        if (nb < k) {
            return false;
        }
        const uint8_t c = by[k - 1];
        if (c >= 0xC0) {
            const int32_t need = c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
            return need > k;
        }
        if (c < 0x80) {
            return false;
        }
    }
    return false;
}

void common_spec_smart::tconf_roles(common_spec_roles r) {
    tc_roles = std::move(r);
    if (!cfg.tconf) {
        return;
    }
    if (tc_roles.on()) {
        int32_t n_a = 0;
        for (const auto & m : tc_roles.markers) {
            n_a += m.role == common_spec_roles::ASSIST;
        }
        SMT_INF("trigram confidence: %d role markers from the chat template (%d of the assistant)\n",
                (int32_t) tc_roles.markers.size(), n_a);
    } else {
        SMT_INF("%s", "trigram confidence: no role markers (the prompt counts as other roles, the generated tokens as "
                "the assistant)\n");
    }
}

void common_spec_smart::tconf_profile(common_spec_tprof_vocab voc, const std::string & tok_key) {
    if (!cfg.tconf || cfg.tprof_path.empty()) {
        return;
    }
    common_spec_tprof_config pc;
    pc.path      = cfg.tprof_path;
    pc.tok_key   = tok_key;
    pc.cap_mib   = cfg.tprof_mib;
    pc.half_life = cfg.tprof_half_life;
    tprof = std::make_unique<common_spec_tprof>(pc, std::move(voc));
    std::string why;
    if (!tprof->open(&why)) {
        // an abnormal input: the profile is off, the file is not touched; the server and the trigram confidence of the
        // request go on
        SMT_ERR("trigram profile %s not used: %s; the trigram profile is off (the file is kept as it is)\n",
                cfg.tprof_path.c_str(), why.c_str());
        tprof.reset();
        return;
    }
    // the coefficients this model pair and k learnt before
    tprof_coef_key = cfg.tprof_coef_key + " k=" + std::to_string(cfg.tconf_k) + " dim=" +
                     std::to_string(common_spec_tconf::DIM);
    common_spec_tprof_coef cf;
    if (tprof->get_coef(tprof_coef_key, cf) && (int32_t) cf.b.size() == common_spec_tconf::DIM) {
        std::copy(cf.b.begin(), cf.b.end(), tconf_model.b);
        std::copy(cf.G.begin(), cf.G.end(), tconf_model.G);
        tc_updates_kept = cf.updates;
        SMT_INF("trigram profile: the coefficients of %lld updates are used\n", (long long) cf.updates);
    } else {
        SMT_INF("%s", "trigram profile: no coefficients of this model pair and k, the static ones are used\n");
    }
}

void common_spec_smart::tprof_put_coef() {
    if (!tprof) {
        return;
    }
    common_spec_tprof_coef cf;
    cf.key = tprof_coef_key;
    cf.b.assign(tconf_model.b, tconf_model.b + common_spec_tconf::DIM);
    cf.G.assign(tconf_model.G, tconf_model.G + common_spec_tconf::DIM);
    cf.updates = tc_updates_kept + tc_updates;
    tprof->put_coef(cf);
}

void common_spec_smart::tconf_request(int32_t seq, const int32_t * prompt, size_t n) {
    if (!cfg.tconf || seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    st.tc_n_prompt = n;
    if (tprof) {
        st.tc_prompt.assign(prompt, prompt + n);
        st.tc_open = true;
    }
}

void common_spec_smart::tconf_end(int32_t seq, const int32_t * gen, size_t n) {
    if (!cfg.tconf || !tprof || seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    if (!st.tc_open) {
        return; // no request began on the sequence since the last one was counted
    }
    st.tc_open = false;
    tprof->commit(st.tc_prompt.data(), st.tc_prompt.size(), gen, n, tc_roles);
    st.tc_prompt.clear();
    st.tc_prompt.shrink_to_fit();
    tprof_put_coef();
    tprof->maybe_save(false);
    {
        const auto & ps = tprof->stats();
        SMT_INF("trigram profile%s: commit %lld, counted M %.0f X %.0f U %.0f tokens (known %.0f, absorbed %.0f, "
                "saturated %.0f), answer %d to its prompt (weight %.2f), %zu trigrams %zu contexts %zu chunks %.1f MiB, "
                "clock %.0f, %.1f ms\n", tprof->read_only() ? " (read-only)" : "", (long long) ps.req, ps.tok_m, ps.tok_x,
                ps.tok_u, ps.tok_known, ps.tok_absorbed, ps.sat_cut, ps.regen_m, ps.weight, ps.nodes, ps.heads,
                ps.chunks, ps.mib, ps.clock, ps.commit_ms);
    }
    if (log) {
        const auto & ps = tprof->stats();
        std::string l = "{\"ev\":\"tprof\",\"req\":" + std::to_string(st.log_req) + ",\"seq\":" + std::to_string(seq) +
                        ",\"commit\":" + std::to_string(ps.req) + ",\"m\":";
        smart_json_num(l, ps.tok_m, "%.0f");
        l += ",\"x\":";
        smart_json_num(l, ps.tok_x, "%.0f");
        l += ",\"u\":";
        smart_json_num(l, ps.tok_u, "%.0f");
        l += ",\"known\":";
        smart_json_num(l, ps.tok_known, "%.0f");
        l += ",\"absorbed\":";
        smart_json_num(l, ps.tok_absorbed, "%.0f");
        l += ",\"sat_cut\":";
        smart_json_num(l, ps.sat_cut, "%.0f");
        l += ",\"answer\":" + std::to_string(ps.regen_m) + ",\"weight\":";
        smart_json_num(l, ps.weight, "%.4g");
        l += ",\"nodes\":" + std::to_string(ps.nodes) + ",\"heads\":" + std::to_string(ps.heads) +
             ",\"chunks\":" + std::to_string(ps.chunks) + ",\"mib\":";
        smart_json_num(l, ps.mib, "%.2f");
        l += ",\"clock\":";
        smart_json_num(l, ps.clock, "%.0f");
        l += ",\"commit_ms\":";
        smart_json_num(l, ps.commit_ms, "%.2f");
        l += ",\"compact_ms\":";
        smart_json_num(l, ps.compact_ms, "%.2f");
        l += ",\"save_ms\":";
        smart_json_num(l, ps.save_ms, "%.2f");
        l += ",\"b\":";
        smart_json_arr(l, std::vector<double>(tconf_model.b, tconf_model.b + common_spec_tconf::DIM));
        l += "}";
        log->put(l);
    }
}

void common_spec_smart::tconf_begin(int32_t seq, const int32_t * toks, size_t n, int32_t id_last, int32_t pos0) {
    if (!cfg.tconf || seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
    // the rejected position of the last round: its target token is id_last when this draft continues right after it
    if (st.tc_pend_valid && pos0 == st.tc_pend_at) {
        tconf_learn(st.tc_pend, id_last);
    }
    st.tc_pend_valid = false;

    st.tri.sync(toks, n, id_last, &tc_roles, st.tc_n_prompt);

    st.tc_ctx.clear();
    for (size_t i = n > 7 ? n - 7 : 0; i < n; ++i) {
        st.tc_ctx.push_back(toks[i]);
    }
    st.tc_ctx.push_back(id_last);
    st.tc_pos.clear();
    st.tc_pos0 = pos0;
}

int32_t common_spec_smart::tconf_pick(int32_t seq, const int32_t * ids, const float * p, int32_t n, int32_t fixed,
                                      double * q) {
    constexpr int32_t NS = common_spec_tconf::N_SRC;
    auto & st = seqs[seq];
    const int32_t kk = std::min<int32_t>(cfg.tconf_k, n);
    const int32_t nc = std::min<int32_t>(std::max<int32_t>(kk, fixed + 1), common_spec_tconf::K_MAX); // a fixed token
                                                                              // outside the k is scored, not learnt
    const size_t  m  = st.tc_ctx.size();

    // per candidate and source: the count of the candidate and the observations of the context
    double cn[common_spec_tconf::K_MAX][NS] = {};
    double cN[NS] = {};
    int32_t N = 0;
    if (m >= 2) {
        int32_t rc[2 * common_spec_tconf::K_MAX];
        int32_t rN[2];
        st.tri.lookup(st.tc_ctx[m - 2], st.tc_ctx[m - 1], ids, nc, rc, rN);
        cN[0] = rN[common_spec_roles::ASSIST];
        cN[1] = rN[common_spec_roles::OTHER];
        N     = rN[0] + rN[1];
        for (int32_t i = 0; i < nc; ++i) {
            cn[i][0] = rc[2 * i + common_spec_roles::ASSIST];
            cn[i][1] = rc[2 * i + common_spec_roles::OTHER];
        }
        if (tprof) {
            constexpr int32_t NP = common_spec_tprof::N_SRC;
            double pc[NP * common_spec_tconf::K_MAX];
            double pN[NP];
            tprof->lookup(st.tc_ctx[m - 2], st.tc_ctx[m - 1], ids, nc, pc, pN);
            for (int32_t s = 0; s < NP; ++s) {
                cN[2 + s] = pN[s];
                for (int32_t i = 0; i < nc; ++i) {
                    cn[i][2 + s] = pc[NP * i + s];
                }
            }
        }
    }
    const bool mid = tconf_mid(st.tc_ctx);

    tconf_pos ps;
    ps.n = kk;
    ps.N = N;
    ps.ids.assign(ids, ids + kk);
    ps.X.assign((size_t) kk * common_spec_tconf::DIM, 0.0);

    int32_t best   = 0;
    double  best_q = -1.0;
    double  x[common_spec_tconf::DIM];
    for (int32_t i = 0; i < nc; ++i) {
        const bool sw = ids[i] >= 0 && (size_t) ids[i] < tc_wf.size() && tc_wf[ids[i]] != 0;
        common_spec_tconf::features(p[i], cn[i], cN, mid, sw, x);
        const double qi = tconf_model.q(x);
        if (i < kk) {
            std::copy(x, x + common_spec_tconf::DIM, ps.X.begin() + (size_t) i * common_spec_tconf::DIM);
        }
        if (fixed >= 0 ? i == fixed : (i < kk && qi > best_q)) {
            best   = i;
            best_q = qi;
        }
    }
    ps.chosen = best;
    ps.id     = ids[best];
    for (int32_t s = 0; s < NS; ++s) {
        ps.src[2 * s]     = (float) cn[best][s];
        ps.src[2 * s + 1] = (float) cN[s];
    }
    if (best != 0) {
        tc_invert++;
    }
    st.tc_pos.push_back(std::move(ps));
    st.tc_ctx.push_back(ids[best]);
    *q = best_q;
    return best;
}

void common_spec_smart::tconf_learn(const tconf_pos & ps, int32_t target) {
    if (ps.n <= 0) {
        return;
    }
    double lab[common_spec_tconf::K_MAX];
    for (int32_t i = 0; i < ps.n; ++i) {
        lab[i] = ps.ids[i] == target ? 1.0 : 0.0;
    }
    tconf_model.update(ps.X.data(), lab, ps.n);
    tc_updates++;
}

void common_spec_smart::tconf_accepted(int32_t seq, int32_t n_accepted) {
    if (!cfg.tconf) {
        return;
    }
    auto & st = seqs[seq];
    // verified positions: the accepted ones (target = the drafted token) and the first rejected one, whose target
    // token is known only from the next draft unless the drafted token was the only candidate
    const int32_t k = std::min<int32_t>(st.k, (int32_t) st.tc_pos.size());
    for (int32_t j = 0; j < std::min(n_accepted, k); ++j) {
        tconf_learn(st.tc_pos[j], st.tc_pos[j].id);
    }
    if (n_accepted < k) {
        const auto & ps = st.tc_pos[n_accepted];
        if (ps.n == 1 && ps.chosen == 0) {
            tconf_learn(ps, -1); // the only candidate was rejected
        } else {
            st.tc_pend       = ps;
            st.tc_pend_valid = true;
            st.tc_pend_at    = st.tc_pos0 + 1 + n_accepted;
        }
    }
}

//
// store
//

static void smart_put(std::string & out, double v) {
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), v); // shortest form that reads back to the same double
    out.append(buf, res.ptr);
}

static bool smart_get(const std::string & tok, double & v) {
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    return res.ec == std::errc() && res.ptr == tok.data() + tok.size();
}

static bool smart_get(const std::string & tok, int32_t & v) {
    const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
    return res.ec == std::errc() && res.ptr == tok.data() + tok.size();
}

static std::string smart_key_line(const std::string & key) {
    std::string k = key;
    std::replace(k.begin(), k.end(), '\n', ' ');
    std::replace(k.begin(), k.end(), '\r', ' ');
    return k;
}

bool common_spec_smart::save(const std::string & path) const {
    std::string out;
    out += SMART_STORE_MAGIC;
    out += "\nkey ";
    out += smart_key_line(cfg.store_key);
    out += "\nshape " + std::to_string(cfg.n_max) + " " + std::to_string(cfg.n_bins) + " " + std::to_string(N_GROUPS) + "\n";
    out += "level ";
    smart_put(out, level_cell.mean); out += " ";
    smart_put(out, level_conf()); out += "\n";
    for (int32_t k = 0; k <= cfg.n_max; ++k) {
        out += "ratio " + std::to_string(k) + " ";
        smart_put(out, shape_cells[k].mean); out += " ";
        smart_put(out, conf(shape_cells[k])); out += " ";
        smart_put(out, shape_cells[k].var); out += "\n";
    }
    for (int32_t g = 0; g < N_GROUPS; ++g) {
        for (int32_t b = 0; b < cfg.n_bins; ++b) {
            const auto & c = cal_cells[(size_t) g * cfg.n_bins + b];
            out += "cal " + std::to_string(g) + " " + std::to_string(b) + " ";
            smart_put(out, c.mean); out += " ";
            smart_put(out, conf(c)); out += "\n";
        }
    }
    for (size_t j = 0; j < qbar_cells.size(); ++j) {
        out += "qbar " + std::to_string(j) + " ";
        smart_put(out, qbar_cells[j].mean); out += " ";
        smart_put(out, conf(qbar_cells[j])); out += "\n";
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

bool common_spec_smart::load(const std::string & path, std::string * why) {
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
    if (!std::getline(f, line) || line != SMART_STORE_MAGIC) {
        return fail("unknown format");
    }
    if (!std::getline(f, line) || line.rfind("key ", 0) != 0) {
        return fail("no key");
    }
    if (line.substr(4) != smart_key_line(cfg.store_key)) {
        return fail("another model file, build or cache setting (key differs)");
    }

    // parse everything first; the state changes only when the whole file is valid
    std::vector<common_spec_smart_cell> shape_new(cfg.n_max + 1);
    common_spec_smart_cell              level_new;
    bool level_ok = false;
    std::vector<common_spec_smart_cell> cal_new(cal_cells.size());
    std::vector<common_spec_smart_cell> qbar_new(qbar_cells.size());
    bool shape_ok = false;
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
        if (t[0] == "shape") {
            int32_t n_max_f = 0, n_bins_f = 0, n_groups_f = 0;
            if (t.size() != 4 || !smart_get(t[1], n_max_f) || !smart_get(t[2], n_bins_f) || !smart_get(t[3], n_groups_f)) {
                return fail("bad shape line");
            }
            if (n_bins_f != cfg.n_bins || n_groups_f != N_GROUPS) {
                return fail("another calibration layout");
            }
            shape_ok = true;
            continue;
        }
        common_spec_smart_cell c;
        if (t[0] == "level" && t.size() == 3) {
            if (!smart_get(t[1], c.mean) || !smart_get(t[2], c.n) || !(c.mean >= 0.0)) {
                return fail("bad level line");
            }
            level_new = c;
            level_ok  = true;
        } else if (t[0] == "ratio" && t.size() == 5) {
            int32_t k = 0;
            if (!smart_get(t[1], k) || !smart_get(t[2], c.mean) || !smart_get(t[3], c.n) || !smart_get(t[4], c.var) ||
                k < 0 || !(c.var >= 0.0)) {
                return fail("bad ratio line");
            }
            if (k <= cfg.n_max) {
                shape_new[k] = c;
            }
        } else if (t[0] == "cal" && t.size() == 5) {
            int32_t g = 0, b = 0;
            if (!smart_get(t[1], g) || !smart_get(t[2], b) || !smart_get(t[3], c.mean) || !smart_get(t[4], c.n) ||
                g < 0 || g >= N_GROUPS || b < 0 || b >= cfg.n_bins) {
                return fail("bad cal line");
            }
            cal_new[(size_t) g * cfg.n_bins + b] = c;
        } else if (t[0] == "qbar" && t.size() == 4) {
            int32_t j = 0;
            if (!smart_get(t[1], j) || !smart_get(t[2], c.mean) || !smart_get(t[3], c.n) || j < 0 || j >= N_QBAR_MAX) {
                return fail("bad qbar line");
            }
            if (j < (int32_t) qbar_new.size()) { // a store of a larger n_max: only the positions of this one
                qbar_new[j] = c;
            }
        } else {
            return fail("unknown line");
        }
    }
    if (!shape_ok || !level_ok || !end_ok) {
        return fail("truncated");
    }

    // the stored means and variances are restored exactly; only their confidence (decayed when saved) is capped, at
    // the current round
    auto cap = [&](common_spec_smart_cell & c, double cap_n) {
        if (!(c.n >= 0.0)) {
            c.n = 0.0;
        }
        c.n = std::min(c.n, cap_n);
        c.t = n_closed;
    };
    // the level with a small confidence: the first observations of this process move it
    cap(level_new, cfg.level_cap_load);
    level_cell = level_new;
    for (int32_t k = 0; k <= cfg.n_max; ++k) {
        cap(shape_new[k], cfg.cap_load);
        shape_cells[k] = shape_new[k];
        // the wait confidence at the running cap as well (the store keeps only the confidence of the mean)
        shape_over[k]  = cfg.conf_cap > 0.0 ? std::max(0.0, shape_cells[k].n - cfg.conf_cap) : 0.0;
        cost_n_run[k]  = 0;
    }
    for (size_t i = 0; i < cal_new.size(); ++i) {
        cap(cal_new[i], cfg.cap_load);
        cal_cells[i] = cal_new[i];
    }
    for (size_t i = 0; i < qbar_new.size(); ++i) {
        cap(qbar_new[i], cfg.cap_load);
        qbar_cells[i] = qbar_new[i];
    }
    store_loaded = true;
    return true;
}

//
// store key helpers
//

std::string common_spec_smart_file_id(const std::string & path) {
    if (path.empty()) {
        return "-";
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return "missing";
    }
    const auto mtime = std::filesystem::last_write_time(path, ec);
    return std::to_string((unsigned long long) size) + ":" +
           std::to_string((long long) (ec ? 0 : mtime.time_since_epoch().count()));
}

std::string common_spec_smart_build_id(const std::string & build_info) {
    std::string id = build_info;

    std::filesystem::path exe;
#ifdef _WIN32
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof(buf) / sizeof(buf[0])));
    if (n > 0 && n < sizeof(buf) / sizeof(buf[0])) {
        exe = std::filesystem::path(std::wstring(buf, n));
    }
#else
    std::error_code ec_exe;
    exe = std::filesystem::read_symlink("/proc/self/exe", ec_exe);
#endif
    if (exe.empty()) {
        return id;
    }

    std::vector<std::string> parts;
    std::error_code ec;
    for (const auto & e : std::filesystem::directory_iterator(exe.parent_path(), ec)) {
        if (!e.is_regular_file(ec)) {
            continue;
        }
        const std::string name = e.path().filename().string();
        const std::string ext  = e.path().extension().string();
        const bool lib = (name.rfind("llama", 0) == 0 || name.rfind("ggml", 0) == 0 ||
                          name.rfind("libllama", 0) == 0 || name.rfind("libggml", 0) == 0) &&
                         (ext == ".dll" || ext == ".so" || ext == ".dylib");
        if (lib || e.path() == exe) {
            parts.push_back(name + "=" + common_spec_smart_file_id(e.path().string()));
        }
    }
    std::sort(parts.begin(), parts.end());
    for (const auto & p : parts) {
        id += " " + p;
    }
    return id;
}
