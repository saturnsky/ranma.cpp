#pragma once

// ranma: the verification cost of the target, shared by every drafter (docs/ranma/spec-smart.md, "Shared verification
// cost"). A round verifies m draft tokens in one target call of r = m + 1 rows. The time of that call depends on the
// target and how it runs (model, cache tiers, batch, contexts loaded next to it), not on which drafter made the draft, so
// one table per target and run condition serves ngram-varlen, draft-dspark and draft-mtp alike. What a drafter adds
// (its draft steps, the catch-up of its draft context) and the acceptance stay with the drafter.
//
// Two parts, both measured by the caller and learnt here:
//   core(r) the target call: from the call to its synchronisation, without the host time of graph builds and captures.
//           level x shape: the level follows a change common to all widths fast, the shape (time / level per width)
//           slowly.
//   aux(r)  the rest of a round of width r: from the end of the draft to the start of the next one, without the target
//           call (the processing of the batch by the drafters' contexts, the sampling and acceptance, the rollback, a
//           checkpoint replay, the output). A plain decayed mean per width over the rounds observed.
// The cost of a round is core(r) + aux(r).
//
// This file has no llama dependency, so that it can be tested on the CPU.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

// decayed mean over time: the weight of an observation halves every half_life ticks of a clock, whether or not the cell
// is observed. The decay is lazy: n is the confidence at tick t and is read as n * decay^(now - t). Time lowers only the
// confidence; the mean and the variance change only with a new observation. The variance is the weighted (population)
// variance of the same decayed weights, kept by weighted Welford. Since all weights decay by the same factor, the
// variance itself does not change with time.
struct common_spec_cost_cell {
    double  mean = 0.0;
    double  n    = 0.0; // virtual observations (confidence) at tick t
    int64_t t    = 0;   // tick of the last update
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

struct common_spec_verify_cost_config {
    int32_t r_max = 9;              // widest call kept (rows): the widest piece any drafter verifies
    double  half_life = 250.0;      // the confidence of the shape and aux cells halves every half_life ticks
    double  level_half_life = 64.0; // the same for the level
    double  reprobe_n = 4.0;        // a shape below this confidence is not a level source, and waits for an extension
    double  conf_cap  = 16.0;       // cap of the shape confidence that decides the wait (0: no cap)
    bool    wait_reset = true;      // an observation of a waiting width counts the old mean with the wait confidence
    double  cap_load = 32.0;        // confidence cap of every stored cell when loaded
    double  level_cap_load = 2.0;   // ... of the stored level
    int32_t lend_skip = 5;          // the call that sees a lend of the decode pool and the next lend_skip - 1 are not
                                    // observed (0: every call is)
    bool    cost_graph = true;      // the host time of graph builds and captures is taken out of the times
    double  prior_n = 2.0;          // blend with a prior table: an estimate of confidence n weighs n / (n + prior_n)
    int32_t save_every = 256;       // ticks between periodic saves
    std::string store_path;         // empty: nothing is loaded or saved
    std::string store_key;          // the target and its run condition; a store with another key is not used
    std::string log_path;           // one JSON line per call and per round (appended); empty: none
};

// a call of the target, as the caller measured it
struct common_spec_verify_sample {
    int64_t  call_id   = -1;    // increasing per call; a call id seen before is ignored
    int32_t  r         = 0;     // rows of the sequence in the call (the verification width)
    double   target_us = -1.0;  // the call, from its start to its synchronisation
    double   graph_us  = 0.0;   // host time of graph builds and captures of the target inside it
    bool     ok        = true;  // the call completed (no error, synchronised)
    bool     replay    = false; // the call replays the accepted tokens after a checkpoint restore
    int32_t  n_seq     = 1;     // sequences with tokens in the call
    int32_t  n_prompt  = 0;     // prompt tokens in the call
    bool     pool_valid = false; // the lend count of the decode pool was read
    uint64_t lends     = 0;
    // round log only
    int32_t  seq = -1;
    int32_t  pos = -1;          // position of the sequence (the context length)
};

// what an estimate rests on
enum common_spec_cost_origin {
    COMMON_SPEC_COST_NONE     = 0, // nothing known
    COMMON_SPEC_COST_OBSERVED = 1, // the width was observed by this process
    COMMON_SPEC_COST_STORED   = 2, // the width has a stored shape only
    COMMON_SPEC_COST_INTERP   = 3, // from the neighbouring widths
    COMMON_SPEC_COST_PRIOR    = 4, // from the prior table (blend only)
};

struct common_spec_cost_estimate {
    double  core_us  = 0.0;  // the target call
    double  aux_us   = 0.0;  // the rest of the round
    double  shape_n  = 0.0;  // confidence of the shape of the width (decayed observations, not a confidence interval)
    double  level_n  = 0.0;  // confidence of the level
    double  variance = 0.0;  // weighted variance of the shape (ratio^2)
    int64_t age      = -1;   // ticks since the shape of the width was observed, -1: never
    common_spec_cost_origin origin = COMMON_SPEC_COST_NONE;
};

// what observe() did with a call (round log)
struct common_spec_verify_obs {
    double  obs_us    = -1.0; // the time observed, -1: none
    double  pred_core = -1.0; // core(r) before the observation, -1: no value
    double  pred_eff  = -1.0; // core(r) + aux(r) before the observation
    double  pred_blend_ms = -1.0; // blend_ms() with the log prior at r before the observation (ms)
    int64_t clock     = 0;    // the clock before the call
};

struct common_spec_verify_cost {
    explicit common_spec_verify_cost(const common_spec_verify_cost_config & cfg);
    ~common_spec_verify_cost();

    common_spec_verify_cost(const common_spec_verify_cost &) = delete;
    common_spec_verify_cost & operator=(const common_spec_verify_cost &) = delete;

    // a call of the target: nullptr when observed, else the reason (a static string: "dup", "failed", "prompt",
    // "mixed", "multi", "replay", "bad", "wide", "untimed" (the first call of its width in this process), "lend").
    // Every completed call but a pure prompt call advances the clock once; a duplicate or failed one does nothing.
    const char * observe(const common_spec_verify_sample & s, common_spec_verify_obs * out = nullptr);

    // the rest of a round of width r (us) whose target call was observed with reason why_core (nullptr: observed);
    // nullptr when observed, else the reason
    const char * observe_aux(int32_t r, double us, const char * why_core);

    // estimates (us); r = 1 .. r_max, outside: the nearest
    common_spec_cost_estimate estimate(int32_t r) const;
    std::vector<double> core_table_m(int32_t n) const; // core by draft length m = 0 .. n - 1 (r = m + 1)
    std::vector<double> aux_table_m(int32_t n) const;  // aux by m, 0 where nothing is known
    std::vector<double> eff_table_m(int32_t n) const;  // core + aux by m

    // The looked-up cost of the target call, smoothed over the widths (a lookup only: the cells, the confidences, the
    // observations and the store are not changed). The widths are smoothed in segments (the border after the 8th row,
    // VC_SEG_ROWS in the .cpp); nothing is borrowed or projected across it. Inside a segment:
    //  - a prior table (ms per row count, index rows - 1; the last step goes on beyond it) is blended with the estimates:
    //    B(r) = (n E(r) + k P(r) b(r)) / (n + k), E = core + aux, n the shape confidence of r, k = prior_n, and b(r) the
    //    ratio estimate / prior of the nearest observed width on each side of r (self excluded; linear distance weights,
    //    confidence weights, k votes of 1); 1 where there is no neighbour. A segment without any observation keeps the
    //    prior at the level of r = 1 (by its weight). What is borrowed does not add to n.
    //  - a non-decreasing fit of B, weighted by n + k (pool adjacent violators), makes the table rise with the width.
    // Without a prior (or one that is not positive and finite over r_max) B = E, weighted by n. Before any observation
    // with a prior: the prior. The whole width r_max is computed once and rows 1 .. n_rows are its prefix (beyond r_max:
    // the prior at the ratio of the last width, never falling; without a prior the last value). Rows 1 .. n_rows, ms
    std::vector<double> blend_ms(const std::vector<double> & prior_ms, int32_t n_rows) const;
    // the same function without a prior, in us (--spec-smart): only the unit differs
    std::vector<double> smooth_us(int32_t n_rows) const;
    std::vector<double> smooth_table(const std::vector<double> & prior, double unit, int32_t n_rows) const; // unit: us per table unit
    // the weighted non-decreasing fit in place (pool adjacent violators): the weighted mean is kept
    static void weighted_pav(std::vector<double> & v, const std::vector<double> & w);
    static double prior_at(const std::vector<double> & prior_ms, int32_t rows);
    void set_log_prior(const std::vector<double> & prior_ms) { log_prior = prior_ms; } // round log only

    // the parts of observe(), for a caller that keeps its own round protocol (common_spec_smart without a shared
    // learner): m = r - 1 throughout
    double conf(const common_spec_cost_cell & c) const { return c.n_at(clock, decay); }
    double level_conf() const { return level_cell.n_at(clock, level_decay); }
    double level() const { return level_cell.mean; }
    bool   level_source(int32_t m) const;   // an observation of width m updates the level
    void   observe_core(int32_t m, double us);
    double wait_conf(int32_t m) const;      // shape confidence capped at conf_cap: decides the wait only
    bool   waiting(int32_t m) const;        // a wait confidence below reprobe_n
    bool   known(int32_t m) const;          // observed by this process or with a stored shape
    // lend count of the decode pool: counted is a call / round that the lend window counts
    void   lend_update(bool pool_valid, uint64_t lends, bool counted);
    bool   lend_round(bool pool_valid) const;
    bool   take_first(int32_t m);           // true: width m was observed (closed) before in this process; marks it
    void   tick();                          // clock + 1 and the periodic save

    bool save(const std::string & path) const;
    bool load(const std::string & path, std::string * why = nullptr);

    std::string summary() const; // one line: level, shape, aux per width

    common_spec_verify_cost_config cfg;
    double decay;       // per tick (shape, aux)
    double level_decay; // per tick (level)

    common_spec_cost_cell              level_cell;  // us; mean 0: not set (the first observation sets it)
    std::vector<common_spec_cost_cell> shape_cells; // [r_max] by m, time / level; n 0: no value
    std::vector<double>                shape_over;  // [r_max] confidence above the wait confidence (wait = n - over)
    std::vector<common_spec_cost_cell> aux_cells;   // [r_max] by m, us
    std::vector<int64_t>               cost_n_run;  // core observations of this process per width
    std::vector<uint8_t>               used_run;    // width seen once by this process: its first call is not observed
    std::vector<uint8_t>               aux_used;    // the same for the rounds

    bool    store_loaded = false;
    int64_t clock        = 0; // the clock of the decay: calls (0 when the store is loaded)
    int64_t n_wait_reset = 0; // observations of a waiting width: the old mean counted with the wait confidence
    bool    wr_last      = false; // the last observe_core() was one
    double  wr_n_before  = 0.0;
    double  wr_n_after   = 0.0;
    uint64_t pool_lends  = 0;  // the lend count at the last counted call
    int64_t pool_since_lend = -1; // counted calls since the last lend (0: a lend seen by this call), -1: none seen
    int64_t clock_saved  = 0;
    bool    save_failed  = false;
    int64_t last_call    = -1;

    // statistics (summary, tests)
    int64_t n_calls  = 0;       // calls passed to observe() (duplicates excluded)
    int64_t n_core   = 0;       // core observations
    int64_t n_aux    = 0;       // aux observations
    std::vector<int64_t> n_skip; // per reason index (see skip_index)
    static int32_t skip_index(const char * why);

    // log (cfg.log_path): one line per call / round; the caller adds its own fields with log_put
    bool logging() const { return log_f != nullptr; }
    void log_put(const std::string & line);
    void log_flush();
    std::vector<double> log_prior;

  private:
    FILE *      log_f = nullptr;
    std::string log_buf;
};

// the rounds of the sequences: who drafted, how wide, and the time from the end of a draft to the start of the next one,
// which gives aux(r) of the shared learner. A round of a sequence ends where its next draft starts, so that a round of
// another drafter never extends one of its own (the --spec-smart controller is closed at the same point).
struct common_spec_verify_rounds {
    common_spec_verify_rounds(common_spec_verify_cost * vc, int32_t n_seq);

    // the drafters start a round of seq at t_us (closes the previous one). dft_graph_us: the host time of the graph
    // builds and captures of the draft context so far (cumulative), taken out of the round's time
    void draft_begin(int32_t seq, int64_t t_us, double dft_graph_us);
    // the drafters are done: m tokens drafted by owner (-1: none), attempted: the drafters that were asked (bit per type)
    void drafted(int32_t seq, int64_t t_us, int32_t m, int32_t owner, uint32_t attempted, double dft_graph_us);
    // a target call ended (after learner->observe(), whose result is why and obs; writes the call line of the log)
    void call(const common_spec_verify_sample & s, const char * why, const common_spec_verify_obs * obs = nullptr);
    // the request of seq ended or a new one began: the open round ends without a time
    void request_end(int32_t seq);

    int32_t owner(int32_t seq) const;        // the drafter of the open round, -1: none (or no round)
    uint32_t attempted(int32_t seq) const;
    int32_t m(int32_t seq) const;

    // names of the drafter types for the round log (index = type); empty: numbers
    std::vector<std::string> type_names;

    struct seq_state {
        bool     open      = false; // drafted, not closed yet
        int64_t  t_drafted = 0;
        double   g_drafted = 0.0;
        int32_t  m         = 0;
        int32_t  owner     = -1;
        uint32_t attempted = 0;
        int64_t  call_id   = -1;    // the first call after the draft
        double   target_us = 0.0;   // ... its time (raw)
        const char * why   = nullptr; // ... why its core was not observed (nullptr: observed)
        bool     calls_ok  = true;  // false: a call that is not this round's verification or replay came in between
        int32_t  n_replay  = 0;
        double   pred_eff  = 0.0;   // eff(r) when drafted (round log)
        double   pred_aux  = 0.0;
    };
    std::vector<seq_state> seqs;

    common_spec_verify_cost * vc;
};

// the store key part of the environment: the switches that change how the target runs (GGML_*, RANMA_*, and LLAMA_*
// except the logs and options of the speculative drafters and the server), sorted, "name=value" joined by spaces
std::string common_spec_verify_cost_env_key();
