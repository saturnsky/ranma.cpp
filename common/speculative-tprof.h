#pragma once

// ranma: the trigram confidence of --spec-smart (docs/ranma/spec-smart.md, "Trigram confidence"):
// - the role of every token of a request (assistant or other), from the role markers of the chat template;
// - the trigram profile (--spec-trigram-profile): the trigram counts of the requests, kept across restarts.
// This file has no llama dependency, so that it can be tested on the CPU.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// the role of every token of a stream. A marker is the token sequence that opens a message of a role (from the chat
// template, e.g. <|assistant|>); a control token that ends no marker (an end-of-turn token, a tool response marker)
// opens a message of another role. Without markers the prompt is OTHER and the generated tokens are ASSIST.
struct common_spec_roles {
    static constexpr uint8_t OTHER  = 0;
    static constexpr uint8_t ASSIST = 1;

    struct marker {
        std::vector<int32_t> toks;
        uint8_t              role = OTHER;
    };
    std::vector<marker>  markers;
    std::vector<uint8_t> ctrl; // per token: a control token that is not a reasoning tag

    bool on() const { return !markers.empty(); }

    // the role of token i of the stream at(0..i], given the role of token i - 1
    template <typename F>
    uint8_t step(uint8_t role, const F & at, size_t i) const {
        const int32_t t = at(i);
        for (const auto & m : markers) {
            const size_t L = m.toks.size();
            if (L == 0 || L > i + 1 || m.toks[L - 1] != t) {
                continue;
            }
            bool eq = true;
            for (size_t j = 0; j + 1 < L && eq; ++j) {
                eq = at(i + 1 - L + j) == m.toks[j];
            }
            if (eq) {
                return m.role;
            }
        }
        if (t >= 0 && (size_t) t < ctrl.size() && ctrl[t]) {
            return OTHER;
        }
        return role;
    }

    // the role of every token of toks[0..n): with markers from the markers, the tokens from gen_from on ASSIST
    std::vector<uint8_t> of(const int32_t * toks, size_t n, size_t gen_from) const {
        std::vector<uint8_t> r(n, OTHER);
        uint8_t cur = OTHER;
        auto at = [&](size_t i) { return toks[i]; };
        for (size_t i = 0; i < n; ++i) {
            if (on()) {
                cur = step(cur, at, i);
            }
            r[i] = i >= gen_from ? ASSIST : cur;
        }
        return r;
    }
};

// the trigram profile (--spec-trigram-profile, the file <--spec-smart-store>.trigram): the trigram counts of every
// request, kept across requests and restarts in a file, by source: M the tokens this server generated, X assistant
// messages of a prompt that this server did not generate (a conversation from before the profile or from elsewhere),
// U the other new text of a prompt (user,
// system, tool messages, and the edited parts of an assistant message). A text counts once: the prompt is cut into
// content-defined chunks of its normalised bytes (no white space, no control tokens), and a chunk that was counted
// before (or generated here) is known and not counted again. The counts decay with the generated tokens:
// their weight halves every half_life generated tokens.
struct common_spec_tprof_config {
    std::string path;                 // the profile file
    std::string tok_key;              // the identity of the tokenizer: a profile of another tokenizer is not used
    double      cap_mib    = 64.0;    // memory of the tables (the node, context and chunk tables together)
    double      half_life  = 131072.0; // generated tokens
    double      eps        = 0.25;    // a trigram whose decayed weight is below it is dropped at a compaction
    int32_t     sat        = 4;       // a request counts a trigram at most this many times per source
    double      rho        = 0.5;     // the m-th answer to the same prompt weighs 1 / (1 + (m - 1) rho)
    int32_t     ring_n     = 64;      // generated answers kept to recognise their edges in a later prompt
    int32_t     save_every_req = 8;   // a save at the end of a request after this many requests ...
    double      save_every_s   = 120.0; // ... or this many seconds since the last one
};

// the profile file of a --spec-smart-store file: <store>.trigram in the same directory (its own key, the tokenizer: the
// store keeps the build key). No store, or the profile off: none (empty)
inline std::string common_spec_tprof_path(const std::string & store, bool on) {
    return on && !store.empty() ? store + ".trigram" : std::string();
}

// per token of the vocabulary: its bytes and whether it is a control token or a reasoning tag (no content, a cut)
struct common_spec_tprof_vocab {
    std::vector<std::string> piece;
    std::vector<uint8_t>     special;
};

// the counts of one request's commit and the state after it (round log, request summary)
struct common_spec_tprof_stats {
    int64_t req       = 0;  // commits of the profile (all processes)
    double  tok_m     = 0;  // generated tokens counted (before the weight)
    double  tok_x     = 0;  // prompt tokens counted as X
    double  tok_u     = 0;  // prompt tokens counted as U
    double  tok_known = 0;  // prompt tokens in known chunks
    double  tok_absorbed = 0; // prompt tokens of assistant chunks found in a recent generated answer
    double  sat_cut   = 0;  // counts cut by the saturation
    int32_t regen_m   = 1;  // the answer's number for its prompt
    double  weight    = 1;  // its weight
    double  commit_ms = 0;
    double  compact_ms = 0; // the compaction of this commit (0: none)
    double  save_ms   = 0;  // the last snapshot taken for a save (0: none in this commit)
    size_t  nodes = 0, heads = 0, chunks = 0;
    double  mib   = 0;      // allocated by the tables
    double  clock = 0;      // generated tokens of the profile
};

// a coefficient block of the trigram confidence kept in the profile: the coefficients of one target / draft model
// pair and k, with the optimizer state
struct common_spec_tprof_coef {
    std::string         key;
    std::vector<double> b, G;
    int64_t             updates = 0;
};

struct common_spec_tprof_impl;

struct common_spec_tprof {
    static constexpr int32_t SRC_M = 0;
    static constexpr int32_t SRC_X = 1;
    static constexpr int32_t SRC_U = 2;
    static constexpr int32_t N_SRC = 3;

    common_spec_tprof(const common_spec_tprof_config & cfg, common_spec_tprof_vocab vocab);
    ~common_spec_tprof(); // waits for a running save and saves (unless off or read-only)

    common_spec_tprof(const common_spec_tprof &) = delete;
    common_spec_tprof & operator=(const common_spec_tprof &) = delete;

    // opens the file (a new profile when there is none) and takes the lock; false: the profile is off, why says why
    // (another tokenizer, a damaged file). A lock held by another process makes it read-only (no saves).
    bool open(std::string * why);
    bool on() const;
    bool read_only() const;

    // per candidate i and source s: cnt[N_SRC * i + s] the decayed count of toks[i] after (a, b), N[s] that of the
    // context
    void lookup(int32_t a, int32_t b, const int32_t * toks, int32_t n, double * cnt, double * N) const;

    // the end of a request: its prompt and generated tokens; roles gives the messages of the prompt
    void commit(const int32_t * prompt, size_t n_prompt, const int32_t * gen, size_t n_gen,
                const common_spec_roles & roles);
    // saves now when due (force: always), through a snapshot written by a background thread
    void maybe_save(bool force);

    // coefficient blocks: get copies the block of key (false: none), put replaces or adds it
    bool get_coef(const std::string & key, common_spec_tprof_coef & c) const;
    void put_coef(const common_spec_tprof_coef & c);

    const common_spec_tprof_stats & stats() const;
    const common_spec_tprof_config & config() const;

    // tests: the trigram weight now (a, b, c, s) and of the context, the chunk ids of a token stream (cut at the
    // given message starts), the known flag of a chunk id
    double weight(int32_t a, int32_t b, int32_t c, int32_t s) const;
    double context_weight(int32_t a, int32_t b, int32_t s) const;
    std::vector<uint64_t> chunk_ids(const int32_t * toks, size_t n) const;
    bool known(uint64_t id) const;
    size_t n_nodes() const;

  private:
    std::unique_ptr<common_spec_tprof_impl> impl;
};
