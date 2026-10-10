#pragma once

// ranma: --spec-smart, the draft length of a one-token-per-step drafter (draft-mtp) chosen per step from the
// measured verification cost per width and a calibrated acceptance of the draft top probability.
// See docs/ranma/spec-smart.md. This file has no llama dependency, so that the controller can be tested on the CPU.

#include "speculative-tprof.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct common_spec_smart_config {
    int32_t     n_max = 3;     // widest round: n_max draft tokens (verification width n_max + 1)
    std::string store_path;    // empty: nothing is loaded or saved
    std::string store_key;     // model files + build + cache settings; a store with another key is not used

    // only n_max, store_path, store_key, half_life and log_path come from options; the other values are fixed
    // constants of the controller (the tests change them here)
    int32_t n_bins      = 20;     // calibration bins over the draft top probability
    double  half_life   = 250.0;  // confidence of every cell halves every half_life closed rounds (all rounds)
    double  level_half_life = 64.0; // the same for the level of the verification time (common to all widths)
    double  prior_n     = 2.0;    // pseudo-observations: q = bin centre (calibration), 0.5 (lookahead)
    double  lookahead_mix = 0.5;  // position j not drafted yet: mix * q(last drafted) + (1 - mix) * qbar(j) (0: qbar)
    double  cap_request = 16.0;   // calibration confidence cap at the start of every request
    double  cap_load    = 32.0;   // confidence cap of every stored cell when loaded
    double  conf_cap    = 16.0;   // cap of the shape confidence that decides the wait for an extension (running and
                                  // loaded): a width not observed for log2(conf_cap / reprobe_n) half-lives waits
                                  // again. The mean keeps its own confidence (0: no cap, the wait uses that one)
    double  level_cap_load = 2.0; // confidence cap of the stored level when loaded (the first observations move it)
    bool    wait_reset  = true;   // an observation of a waiting width counts the old mean of its shape with the wait
                                  // confidence (no option: false is the rule before, for the tests)
    int32_t save_every  = 256;    // closed rounds between periodic saves
    double  reprobe_n     = 4.0;  // a width whose shape confidence is below this waits for an extension: when the
                                  // decision stops at k and width k + 1 waits, the round verifies k + 1 ...
    int32_t reprobe_every = 16;   // ... at most once per this many closed rounds (0: never)
    // the cost of a round without what the graph runtime and the expert cache add to it now and then (the counts
    // come from the caller, close_round / drafted / add_draft_step; without them nothing changes):
    bool    cost_graph  = true;   // the host time of graph builds and captures is taken out of the verification time
                                  // and of the draft step time
    int32_t lend_skip   = 5;      // the verification time of the round in which the decode pool was lent and of the
                                  // next lend_skip - 1 rounds is not observed (0: every round is)
    std::string log_path;         // round log (JSON lines, appended); empty: none. The decisions do not depend on it
    // trigram confidence (LLAMA_SPEC_TRIGRAM_CONF, see common_spec_tconf): the acceptance of a draft token is a logistic
    // of its drafter probability and of the request's own trigram counts instead of the calibration cells. The server
    // turns it on (LLAMA_SPEC_TRIGRAM_CONF=0: off); here it is off, so that the tests of the cells stay as they are
    bool    tconf       = false;
    int32_t tconf_k     = 1;      // drafter candidates per step it scores and learns from (1..10); k >= 2 drafts the
                                  // candidate with the highest acceptance (greedy drafting only)
    // the trigram profile (--spec-trigram-profile, with the trigram confidence only): the counts of every request kept
    // across requests and restarts (see common_spec_tprof) in <--spec-smart-store>.trigram; empty path: none
    std::string tprof_path;
    double      tprof_mib       = 64.0;
    double      tprof_half_life = 131072.0; // generated tokens (LLAMA_SPEC_TRIGRAM_PROFILE_HALF_LIFE)
    std::string tprof_coef_key;             // the target and draft model files: the coefficients of the trigram
                                            // confidence kept in the profile for another pair (or k) are not used
};

// what a decision saw (round log only): filled by decide() when asked, never read by the controller
struct common_spec_smart_diag {
    bool    valid   = false;
    int32_t k       = -1;           // the result of decide(): -1 = one more step
    double  exp_acc = 0.0;          // expected accepted tokens of the best prefix
    double  t_stop  = 0.0;          // expected throughput (tokens / us) of stopping with the best prefix
    double  t_cont  = -1.0;         // the same for one more step and the best depth after it, -1: no step left
    double  draft_pred = 0.0;       // draft step time used (us)
    std::vector<double> q;          // calibrated acceptance of every drafted position
    std::vector<double> la;         // predicted acceptance of the positions not drafted yet (n_drafted, ...)
    std::vector<double> cost;       // predicted verification time per width (us)
};

struct common_spec_smart_log; // round log writer (speculative-smart.cpp)

// round log only: the cumulative counts of the expert cache's decode pool (ggml_expert_pool_counts), read by the caller
// at the end of the draft steps and at the end of the verification. The decisions never read them.
struct common_spec_smart_pool {
    bool     valid      = false;
    uint64_t lends      = 0;
    uint64_t reclaims   = 0;
    uint64_t hits       = 0;
    uint64_t fills      = 0;
    uint64_t fill_bytes = 0;
};

// round log only: the cumulative graph counts of one context (llama_get_graph_counts), read with the pool counts
struct common_spec_smart_graph {
    bool    valid      = false;
    int64_t calls      = 0;  // ubatch graphs computed
    int64_t builds     = 0;  // of them, built and allocated anew
    int64_t build_us   = 0;
    int32_t last_nt    = 0;  // rows of the last graph built
    size_t  last_extent = 0; // the last graph computed: its compute buffer extent and highest position
    int32_t last_pos   = -1;
    bool    backend    = false; // the GPU backend counts its graph calls
    int64_t captures   = 0;  // calls that captured a HIP graph
    int64_t capture_us = 0;
    int64_t direct     = 0;  // calls run without a graph (or a warmup call)
    int64_t direct_us  = 0;
};

// the target and the draft context
struct common_spec_smart_graphs {
    common_spec_smart_graph tgt;
    common_spec_smart_graph dft;
};

// decayed mean over time: the weight of an observation halves every half_life closed rounds of the controller,
// whether or not the cell is observed. The decay is lazy: n is the confidence at round t and is read as
// n * decay^(now - t). Time lowers only the confidence; the mean and the variance change only with a new observation.
// The variance is the weighted (population) variance of the same decayed weights, kept by weighted Welford. Since all
// weights decay by the same factor, the variance itself does not change with time.
struct common_spec_smart_cell {
    double  mean = 0.0;
    double  n    = 0.0; // virtual observations (confidence) at round t
    int64_t t    = 0;   // round of the last update
    double  var  = 0.0; // weighted variance of the observations

    double n_at(int64_t now, double decay) const {
        return now > t ? n * std::pow(decay, (double) (now - t)) : n;
    }

    void add(double x, double decay, int64_t now) {
        const double w0 = n_at(now, decay);
        n = w0 + 1.0;
        t = std::max(t, now);
        const double d = x - mean;
        mean += d / n;
        var = std::max(0.0, (w0 * var + d * (x - mean)) / n);
    }

    void cap(double c, double decay, int64_t now) {
        n = std::min(n_at(now, decay), c);
        t = std::max(t, now);
    }
};

// trigram counts of one request: the next tokens after every two-token context of its prompt and generated tokens,
// counted apart by the role of the next token (common_spec_roles: ASSIST, OTHER)
struct common_spec_trigram {
    // makes the table hold every trigram of the stream toks[0..n) + last; a stream that does not continue the one
    // inserted so far (another request, a rollback) is inserted anew. roles (may be null: every token OTHER) and
    // gen_from (the first generated token: ASSIST) give the role of every token
    void sync(const int32_t * toks, size_t n, int32_t last, const common_spec_roles * roles = nullptr,
              size_t gen_from = SIZE_MAX);
    void clear();
    // N[r] = observations after (a, b) of role r, cnt[2 i + r] = those of toks[i]
    void lookup(int32_t a, int32_t b, const int32_t * toks, int32_t n, int32_t * cnt, int32_t * N) const;
    size_t size() const { return n_ins; }

  private:
    void add(int32_t a, int32_t b, int32_t c, uint8_t r);
    struct node { int32_t tok, cnt[2], next; };
    struct head { int32_t first, n[2]; };
    std::unordered_map<uint64_t, head> map;
    std::vector<node> nodes;
    size_t  n_ins     = 0;  // stream tokens inserted
    int32_t tok_prev  = -1; // the last two of them
    int32_t tok_last  = -1;
    uint8_t role_last = common_spec_roles::OTHER; // the role of the last one (the markers)
};

// trigram confidence: per drafter candidate i, q_i = sigmoid(b . x_i) with
//   x_i = [1, lg p_i, f_RA, g_RA, f_RU, g_RU, f_PM, g_PM, f_PX, g_PX, f_PU, g_PU, mid, sw_i]
//   g_s = N_s / (N_s + 3), f_s = g_s lg((n_s + 0.5) / (N_s + 1)) (both 0 at N_s = 0), lg = logit
// p_i the drafter probability of the candidate (softmax over its top 10); per source s the count n_s of the candidate
// after the last two tokens and the observations N_s of that context: RA / RU the request's own tokens of the assistant
// / of the other roles, PM / PX / PU the trigram profile (--spec-trigram-profile; 0 without it). mid: the context ends
// inside an unfinished UTF-8 sequence, sw_i: the candidate starts a word. The coefficients start at the static fit of k
// and follow the verified positions by AdaGrad whose sum of squared gradients decays (half-life G_HALF_LIFE updates),
// so that the step stays near lr / sqrt(G_HALF_LIFE / ln 2) instead of going to 0; only the k candidates scored are learnt.
struct common_spec_tconf {
    static constexpr int32_t DIM   = 14;
    static constexpr int32_t K_MAX = 10;
    static constexpr int32_t N_SRC = 5;   // RA, RU, PM, PX, PU
    static constexpr int32_t X_MID = 12;  // index of mid in x
    static constexpr int32_t X_SW  = 13;  // index of sw in x
    static constexpr double  G_HALF_LIFE = 4096.0; // updates

    double b[DIM] = {};
    double G[DIM] = {};
    double lr = 0.15;

    void   reset(int32_t k);  // the static coefficients of k
    double q(const double * x) const;
    void   update(const double * X, const double * lab, int32_t n); // n rows of DIM
    // n[s], N[s]: the count of the candidate and the observations of the context per source
    static void features(double p, const double * n, const double * N, bool mid, bool sw, double * x);
};

// one draft step of the trigram confidence
struct common_spec_tconf_pos {
    int32_t n      = 0;  // candidates scored
    int32_t chosen = 0;  // index of the drafted one among the drafter candidates (may be >= n for a fixed one)
    int32_t id     = -1; // the drafted token
    int32_t N      = 0;  // trigram observations of the context in the request (both roles)
    float   src[2 * common_spec_tconf::N_SRC] = {}; // round log: n_s, N_s of the drafted candidate per source
    std::vector<int32_t> ids; // the n candidates scored
    std::vector<double>  X;   // n rows of common_spec_tconf::DIM
};

struct common_spec_smart {
    static constexpr int32_t N_GROUPS = 3;  // draft positions 0, 1 and 2+ share calibration cells
    static constexpr int32_t N_QBAR_MAX = 16; // mean acceptance per position 0..n_max, the last cell for all beyond
    using tconf_pos = common_spec_tconf_pos;

    common_spec_smart(const common_spec_smart_config & cfg, int32_t n_seq);
    ~common_spec_smart();

    common_spec_smart(const common_spec_smart &) = delete;
    common_spec_smart & operator=(const common_spec_smart &) = delete;

    // round protocol, per sequence, all times in microseconds:
    //   request_begin   at the start of a generation (drops an unfinished round, shrinks the calibration confidence)
    //   close_round     when the sequence drafts again: the time since drafted() is the verification of the last round
    //   begin_round     before the first draft step; false: draft nothing (k = 0)
    //   after_step      after each draft step with its top probability; -1: one more step, else keep that many tokens
    //   drafted         with the final draft length
    //   accepted        after the target verified the draft (k > 0 only)
    //   pool, graphs    (round log only, may be null) the decode pool counts and the graph counts of the target and
    //                   draft contexts at that moment, for close_round and drafted
    void    request_begin(int32_t seq);
    void    close_round(int32_t seq, int64_t t_us, const common_spec_smart_pool * pool = nullptr,
                        const common_spec_smart_graphs * graphs = nullptr);
    bool    begin_round(int32_t seq, int32_t n_max_seq);
    int32_t after_step(int32_t seq, float p_top, double q_direct = -1.0); // q_direct >= 0: the trigram confidence
    void    drafted(int32_t seq, int32_t k, int64_t t_us, const common_spec_smart_pool * pool = nullptr,
                    const common_spec_smart_graphs * graphs = nullptr);
    void    accepted(int32_t seq, int32_t n_accepted);
    void    add_draft_step(double us, double graph_us = 0.0); // graph_us: graph builds and captures inside the step

    // trigram confidence (cfg.tconf; every call returns at once without it):
    //   tconf_vocab  once: per token its last bytes (up to 3: the count in bits 24..25, the last byte in bits 0..7)
    //                and whether it starts a word
    //   tconf_roles  once (the server, from the chat template): the role markers and the control tokens
    //   tconf_request at the start of a request with its prompt: the generated tokens start after it
    //   tconf_begin  at every draft of the sequence: the context (prompt and generated tokens before id_last) and
    //                id_last at position pos0. Inserts the new tokens into the request's trigram table and learns the
    //                rejected position of the last round, whose target token is id_last
    //   tconf_pick   at every draft step with the drafter candidates (sorted, n of them): scores the first min(k, n)
    //                and returns the one to draft (fixed >= 0: that one, e.g. a sampled token), q its acceptance
    bool    tconf_on() const { return cfg.tconf; }
    void    tconf_vocab(std::vector<uint32_t> tail, std::vector<uint8_t> word_first);
    void    tconf_roles(common_spec_roles r);
    //   tconf_profile once (cfg.tprof_path): the token pieces and the tokenizer identity; opens the profile (an
    //                unusable one, of another tokenizer or damaged, is an error: the profile is off, the file is kept)
    //   tconf_end    at the end of a request with its generated tokens: counted into the profile
    void    tconf_profile(common_spec_tprof_vocab voc, const std::string & tok_key);
    void    tconf_request(int32_t seq, const int32_t * prompt, size_t n);
    void    tconf_end(int32_t seq, const int32_t * gen, size_t n);
    bool    tprof_on() const { return tprof != nullptr; }
    void    tconf_begin(int32_t seq, const int32_t * toks, size_t n, int32_t id_last, int32_t pos0);
    int32_t tconf_pick(int32_t seq, const int32_t * ids, const float * p, int32_t n, int32_t fixed, double * q);
    bool    tconf_mid(const std::vector<int32_t> & ctx) const; // the context ends inside an unfinished UTF-8 sequence

    // one line per request: mean k, share per width, predicted / observed cost per width, acceptance
    std::string request_summary(int32_t seq) const;

    // decision after n_drafted steps with top probabilities p[0..n_drafted): -1 = draft one more, else the number of
    // tokens to keep (0..n_drafted). Expected throughput = (expected accepted + 1) / (draft time so far + cost(k)).
    // diag (round log only): what the decision saw; nullptr: nothing is recorded (the result is the same)
    // qd (trigram confidence): the acceptance of every drafted position, used instead of q(i, p)
    int32_t decide(const float * p, int32_t n_drafted, int32_t n_max_seq, common_spec_smart_diag * diag = nullptr,
                   const double * qd = nullptr) const;

    std::vector<double> cost_table() const; // predicted verification time per width 0..n_max: level x shape
    double cost(int32_t k) const { return cost_table()[k]; }
    double q(int32_t i, float p) const;     // calibrated acceptance of a draft token at position i with top probability p
    double qbar(int32_t i) const;           // mean acceptance at position i (lookahead for steps not drafted yet)
    double lookahead(int32_t j, const float * p, int32_t n_drafted, const double * qd = nullptr) const; // predicted acceptance at position j >= n_drafted
    double draft_step() const;              // time of one draft step (its mean)
    double conf(const common_spec_smart_cell & c) const { return c.n_at(n_closed, decay); } // confidence now
    double level_conf() const { return level_cell.n_at(n_closed, level_decay); }            // confidence of the level
    double level() const { return level_cell.mean; }                                         // 0: not set yet
    bool   level_source(int32_t k) const;   // an observation of width k updates the level
    double wait_conf(int32_t w) const;      // shape confidence of width w capped at conf_cap: decides the wait only
    bool   waiting(int32_t w) const;        // width w has a wait confidence below reprobe_n (waits for an extension)
    bool   can_extend(int32_t w, int32_t n_max_seq) const; // a decision that stops at w - 1 verifies w instead now
    int32_t frontier() const;               // widest w such that every width 1..w was verified by this process or has
                                            // a shape (stored or observed); a decision does not draft beyond it

    void observe_cost(int32_t k, double us);
    void observe_acceptance(const float * p, int32_t k, int32_t n_accepted);
    void shrink_calibration(double cap);

    // round log: write the buffered lines to the file (at the end of a request); no effect without --spec-smart-log
    void log_flush() const;
    bool logging() const { return log != nullptr; } // the round log is on
    // the caller reads the pool and graph counts for close_round / drafted / add_draft_step
    bool wants_counts() const { return log != nullptr || cfg.cost_graph || cfg.lend_skip > 0; }

    bool save(const std::string & path) const;
    bool load(const std::string & path, std::string * why = nullptr);

    int32_t bin(float p) const;

    common_spec_smart_config cfg;
    double decay;       // per closed round
    double level_decay; // per closed round, for the level

    // verification time of width k = level x shape[k]. The level follows fast the change common to all widths, the
    // shape (ratio of the widths) slowly.
    common_spec_smart_cell              level_cell;  // us; mean 0: not set (the first observation sets it)
    std::vector<common_spec_smart_cell> shape_cells; // [n_max + 1], time / level per width; n 0: no value
    std::vector<double>                 shape_over;  // [n_max + 1], confidence of the shape above the wait confidence
                                                     // at the round of the cell (wait = n - over, decays alike)
    std::vector<common_spec_smart_cell> cal_cells;  // [N_GROUPS * n_bins], acceptance per position group and bin
    std::vector<common_spec_smart_cell> qbar_cells; // [min(n_max + 1, N_QBAR_MAX)], acceptance per position
    common_spec_smart_cell              draft_cell; // time of one draft step; never stored

    std::vector<int64_t> cost_n_run;     // verification times observed by this process per width
    std::vector<uint8_t> used_run;       // width verified (closed) once by this process: its first time is not observed

    bool    store_loaded = false;
    int64_t n_closed     = 0; // the clock of the decay: closed rounds of this process (0 when the store is loaded)
    int64_t n_extend     = 0; // rounds extended by one width to verify a waiting width
    int64_t n_wait_reset = 0; // observations of a waiting width: the old mean counted with the wait confidence
    bool    wr_last      = false; // the last observe_cost() was one (round log)
    double  wr_n_before  = 0.0;   // ... the confidence of the mean before it and after the cut (before the observation)
    double  wr_n_after   = 0.0;
    int64_t last_extend  = 0; // n_closed at the last one
    uint64_t pool_lends  = 0;  // the lend count at the last close_round
    int64_t pool_since_lend = -1; // closed rounds since the last lend (0: a lend in this round), -1: none seen
    int64_t n_closed_saved = 0;
    bool    save_failed  = false;

    struct seq_state {
        // pending round
        bool    drafting  = false; // between begin_round and drafted
        bool    active    = false; // between drafted and close_round
        bool    has_acc   = false;
        bool    timed     = true;  // false: the first closed round of its width in this process (not observed)
        int32_t forced    = -1;    // extension with one more draft step: the width of this round
        int32_t k_nat     = -1;    // extension: the width the decision stopped at (the round verifies k_nat + 1)
        bool    ext_step  = false; // extension: one more draft step was taken for it (else a dropped token was kept)
        int32_t n_max_seq = 0;
        int32_t n_lim     = 0;     // min(n_max_seq, frontier()) at the start of the round: the decision's limit
        int32_t k         = 0;
        int64_t t_drafted = 0;
        std::vector<float> p;
        std::vector<double> qd;          // trigram confidence of every draft step (empty without it)

        // trigram confidence
        common_spec_trigram     tri;
        std::vector<int32_t>    tc_ctx;  // the last tokens of the context, id_last and the drafted ones
        std::vector<tconf_pos>  tc_pos;  // per draft step
        tconf_pos               tc_pend; // the rejected position of the last round, waiting for its target token
        bool                    tc_pend_valid = false;
        int32_t                 tc_pend_at = -1; // ... the position of that token (the next round's pos0)
        int32_t                 tc_pos0 = -1;    // pos0 of the current round
        size_t                  tc_n_prompt = SIZE_MAX; // prompt tokens of the request (the generated ones follow)
        bool                    tc_open = false;        // a request began and was not counted into the profile yet
        std::vector<int32_t>    tc_prompt;              // its prompt (with a profile)

        // round log only (not read by the decisions)
        int64_t log_req   = 0;     // request id of this sequence
        int64_t log_round = 0;     // round id of the pending round
        int32_t log_forced = 0;    // 0 none, 3 extension
        int32_t log_steps = 0;     // draft steps of the round, dropped ones included
        int32_t log_acc   = -1;    // accepted tokens of the round, -1: not verified
        int32_t log_steps_timed = 0;
        double  log_obs_us = -1.0;  // the verification time observed for the round, -1: not observed
        const char * log_skip = nullptr; // why the round's time was not observed, nullptr: it was
        double  log_draft_us = 0.0; // time of the draft steps of the round
        std::vector<float>  log_p;       // top probability of every draft step
        std::vector<double> log_la_next; // before draft step i: the predicted acceptance of position i
        common_spec_smart_diag log_diag; // the last decision of the round
        common_spec_smart_pool pool_next;    // decode pool counts at the last close_round: the start of the next round
        common_spec_smart_pool pool_start;   // ... when the draft of the closed round started
        common_spec_smart_pool pool_drafted; // ... when the draft ended (drafted)
        common_spec_smart_pool pool_end;     // ... when the verification ended (close_round); valid: all three were read
        common_spec_smart_graphs graph_next, graph_start, graph_drafted, graph_end; // the same for the graph counts

        // request statistics
        int64_t n_rounds = 0;
        int64_t n_draft_tok = 0;
        int64_t n_acc_tok = 0;
        int64_t n_extend = 0;
        std::vector<int64_t> n_k;
        std::vector<double>  obs_sum_k;
        std::vector<int64_t> obs_n_k;
    };
    std::vector<seq_state> seqs;

    // trigram confidence
    common_spec_tconf     tconf_model;
    std::vector<uint32_t> tc_tail;
    std::vector<uint8_t>  tc_wf;
    common_spec_roles     tc_roles;
    std::unique_ptr<common_spec_tprof> tprof; // the trigram profile, nullptr: none (or off)
    std::string           tprof_coef_key;     // the coefficient block of this process: model pair, k, layout
    int64_t               tc_updates_kept = 0; // AdaGrad steps of the coefficients before this process (from the profile)
    void tprof_put_coef();
    int64_t               tc_updates = 0; // AdaGrad steps of the process
    int64_t               tc_invert  = 0; // draft steps that drafted another candidate than the drafter's top one
    void tconf_learn(const tconf_pos & ps, int32_t target);
    void tconf_accepted(int32_t seq, int32_t n_accepted);

    std::unique_ptr<common_spec_smart_log> log; // --spec-smart-log, nullptr when off
    void log_round(int32_t seq, const char * end, double verify_us);
    void start_extend(int32_t seq, int32_t k_nat, bool ext_step); // the round verifies k_nat + 1
};

// identity of a file for the store key: "<size>:<mtime>" (empty path: "-")
std::string common_spec_smart_file_id(const std::string & path);

// identity of the running build for the store key: build number and commit, plus the size and time of the
// executable and of the llama / ggml libraries next to it
std::string common_spec_smart_build_id(const std::string & build_info);
