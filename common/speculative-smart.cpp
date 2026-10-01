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
            l += ",\"reprobe_every\":" + std::to_string(cfg.reprobe_every) + ",\"conf_cap\":";
            smart_json_num(l, cfg.conf_cap);
            l += ",\"lookahead_mix\":";
            smart_json_num(l, cfg.lookahead_mix);
            l += "}";
            log->put(l);
            log->flush();
            SMT_INF("round log %s\n", cfg.log_path.c_str());
        }
    }

    SMT_INF("n_max = %d, %s, half-life %.0f rounds, level half-life %.0f rounds, extension to a width below %.1f "
            "observations at most every %d rounds, wait confidence cap %.0f (0: none), lookahead %.2f q(last) + %.2f "
            "qbar(position)\n",
            cfg.n_max, store_loaded ? "warm" : "cold", cfg.half_life, cfg.level_half_life,
            cfg.reprobe_n, cfg.reprobe_every, cfg.conf_cap, cfg.lookahead_mix, 1.0 - cfg.lookahead_mix);
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

double common_spec_smart::lookahead(int32_t j, const float * p, int32_t n_drafted) const {
    if (n_drafted <= 0 || cfg.lookahead_mix == 0.0) {
        return qbar(j);
    }
    // the acceptance of the last drafted position predicts the next ones; the position mean keeps the fall with depth
    const double a = cfg.lookahead_mix;
    return a * q(n_drafted - 1, p[n_drafted - 1]) + (1.0 - a) * qbar(j);
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
                                  common_spec_smart_diag * diag) const {
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
            const double qk = q(k - 1, p[k - 1]);
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
        const double la = lookahead(n_drafted + m - 1, p, n_drafted);
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

void common_spec_smart::add_draft_step(double us) {
    if (us >= 0.0) {
        draft_cell.add(us, decay, n_closed);
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
    st.n_rounds = 0;
    st.n_draft_tok = 0;
    st.n_acc_tok = 0;
    st.n_extend = 0;
    std::fill(st.n_k.begin(), st.n_k.end(), 0);
    std::fill(st.obs_sum_k.begin(), st.obs_sum_k.end(), 0.0);
    std::fill(st.obs_n_k.begin(), st.obs_n_k.end(), 0);

    shrink_calibration(cfg.cap_request);
}

void common_spec_smart::close_round(int32_t seq, int64_t t_us) {
    if (seq < 0 || seq >= (int32_t) seqs.size()) {
        return;
    }
    auto & st = seqs[seq];
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
    if (st.timed) {
        observe_cost(st.k, us);
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

int32_t common_spec_smart::after_step(int32_t seq, float p_top) {
    auto & st = seqs[seq];
    st.p.push_back(p_top);
    const int32_t i = (int32_t) st.p.size();

    if (st.forced >= 0) {
        return i >= st.forced || i >= st.n_max_seq ? i : -1; // the extra step of an extension
    }

    const int32_t r = decide(st.p.data(), i, st.n_lim, log ? &st.log_diag : nullptr);
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

void common_spec_smart::drafted(int32_t seq, int32_t k, int64_t t_us) {
    auto & st = seqs[seq];
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
            for (int32_t j = 0; j < (int32_t) st.p.size(); ++j) {
                dg.q.push_back(q(j, st.p[j]));
                if (j < k) {
                    S *= dg.q.back();
                    dg.exp_acc += S;
                }
            }
        }
    }
    st.p.resize(k);

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
    l += ",\"accepted\":";
    if (st.k == 0) {
        l += "0";
    } else if (st.log_acc >= 0) {
        l += std::to_string(st.log_acc);
    } else {
        l += "null";
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
    return s;
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
