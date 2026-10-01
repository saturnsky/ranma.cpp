#pragma once

// ranma: --spec-smart, the draft length of a one-token-per-step drafter (draft-mtp) chosen per step from the
// measured verification cost per width and a calibrated acceptance of the draft top probability.
// See docs/ranma/spec-smart.md. This file has no llama dependency, so that the controller can be tested on the CPU.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
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
    int32_t reprobe_every = 64;   // ... at most once per this many closed rounds (0: never)
    std::string log_path;         // round log (JSON lines, appended); empty: none. The decisions do not depend on it
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

struct common_spec_smart {
    static constexpr int32_t N_GROUPS = 3;  // draft positions 0, 1 and 2+ share calibration cells
    static constexpr int32_t N_QBAR_MAX = 16; // mean acceptance per position 0..n_max, the last cell for all beyond

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
    void    request_begin(int32_t seq);
    void    close_round(int32_t seq, int64_t t_us);
    bool    begin_round(int32_t seq, int32_t n_max_seq);
    int32_t after_step(int32_t seq, float p_top);
    void    drafted(int32_t seq, int32_t k, int64_t t_us);
    void    accepted(int32_t seq, int32_t n_accepted);
    void    add_draft_step(double us);

    // one line per request: mean k, share per width, predicted / observed cost per width, acceptance
    std::string request_summary(int32_t seq) const;

    // decision after n_drafted steps with top probabilities p[0..n_drafted): -1 = draft one more, else the number of
    // tokens to keep (0..n_drafted). Expected throughput = (expected accepted + 1) / (draft time so far + cost(k)).
    // diag (round log only): what the decision saw; nullptr: nothing is recorded (the result is the same)
    int32_t decide(const float * p, int32_t n_drafted, int32_t n_max_seq, common_spec_smart_diag * diag = nullptr) const;

    std::vector<double> cost_table() const; // predicted verification time per width 0..n_max: level x shape
    double cost(int32_t k) const { return cost_table()[k]; }
    double q(int32_t i, float p) const;     // calibrated acceptance of a draft token at position i with top probability p
    double qbar(int32_t i) const;           // mean acceptance at position i (lookahead for steps not drafted yet)
    double lookahead(int32_t j, const float * p, int32_t n_drafted) const; // predicted acceptance at position j >= n_drafted
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

        // round log only (not read by the decisions)
        int64_t log_req   = 0;     // request id of this sequence
        int64_t log_round = 0;     // round id of the pending round
        int32_t log_forced = 0;    // 0 none, 3 extension
        int32_t log_steps = 0;     // draft steps of the round, dropped ones included
        int32_t log_acc   = -1;    // accepted tokens of the round, -1: not verified
        int32_t log_steps_timed = 0;
        double  log_draft_us = 0.0; // time of the draft steps of the round
        std::vector<float>  log_p;       // top probability of every draft step
        std::vector<double> log_la_next; // before draft step i: the predicted acceptance of position i
        common_spec_smart_diag log_diag; // the last decision of the round

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

    std::unique_ptr<common_spec_smart_log> log; // --spec-smart-log, nullptr when off
    void log_round(int32_t seq, const char * end, double verify_us);
    void start_extend(int32_t seq, int32_t k_nat, bool ext_step); // the round verifies k_nat + 1
};

// identity of a file for the store key: "<size>:<mtime>" (empty path: "-")
std::string common_spec_smart_file_id(const std::string & path);

// identity of the running build for the store key: build number and commit, plus the size and time of the
// executable and of the llama / ggml libraries next to it
std::string common_spec_smart_build_id(const std::string & build_info);
