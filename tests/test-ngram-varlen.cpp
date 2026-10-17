// ranma: the variable-length n-gram drafter (common/ngram-varlen.h) on the CPU.
//
// Without arguments: small fixed cases (cohort majority, ladder, support, prompt weight, new-text test, expiry).
// With --trace FILE: replays a token trace and writes the draft of every step, in the format of the reference script
// of the CPU replay, to compare the two.
//   trace: per request a line "P <prompt ids>" and a line "G <generated ids>"
//   output: per generated token but the last "<request> <step> <n> <samples> <length>: <ids>"
#include "ngram-varlen.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

using tokens = std::vector<llama_token>;

static int test_cases() {
    // two earlier runs of 10 11 12 continue with 1 2 3 and 1 2 30: the draft stops where they split
    {
        common_ngram_varlen ng({}, 1);
        ng.begin(0, { 10, 11, 12, 1, 2, 3, 20, 10, 11, 12, 1, 2, 30, 10, 11, 12 });
        auto d = ng.draft(0);
        CHECK((d.tokens == tokens{ 1, 2 }));
        CHECK(d.n == 3 && d.n_samples == 2);

        ng.add(0, 1);
        d = ng.draft(0);
        CHECK((d.tokens == tokens{ 2 }));
    }

    // one sample is no evidence
    {
        common_ngram_varlen ng({}, 1);
        ng.begin(0, { 10, 11, 12, 1, 2, 3, 20, 10, 11, 12 });
        CHECK(ng.draft(0).tokens.empty());
    }

    // ladder: two samples allow 4 tokens, three allow 8
    {
        const tokens run = { 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 40, 41 };
        tokens p;
        for (int k = 0; k < 2; ++k) {
            p.insert(p.end(), run.begin(), run.end());
            p.push_back(90 + k);
        }
        p.insert(p.end(), { 10, 11, 12 });

        common_ngram_varlen ng({}, 1);
        ng.begin(0, p);
        CHECK((ng.draft(0).tokens == tokens{ 1, 2, 3, 4 }));

        tokens q;
        for (int k = 0; k < 3; ++k) {
            q.insert(q.end(), run.begin(), run.end());
            q.push_back(90 + k);
        }
        q.insert(q.end(), { 10, 11, 12 });

        common_ngram_varlen ng3({}, 1);
        ng3.begin(0, q);
        CHECK((ng3.draft(0).tokens == tokens{ 1, 2, 3, 4, 5, 6, 7, 8 }));
    }

    // the longest matched context wins: 5 matched tokens pick the continuation 7, 3 matched tokens are split
    {
        common_ngram_varlen ng({}, 1);
        ng.begin(0, { 50, 51, 10, 11, 12, 7, 99, 50, 51, 10, 11, 12, 7, 98, 60, 61, 10, 11, 12, 8, 97, 62, 63, 10, 11, 12, 8, 96,
                      50, 51, 10, 11, 12 });
        auto d = ng.draft(0);
        CHECK((d.tokens == tokens{ 7 }));
        CHECK(d.n == 5 && d.n_samples == 2);
    }

    // a generated sample weighs 4 times a prompt sample: 1 generated + 2 prompt samples is 4/6 < 0.9, no draft;
    // with 2 generated samples against 1 prompt sample: 8/9 < 0.9, still none; 3 against 1: 12/13 > 0.9
    {
        common_ngram_varlen_params params;
        params.half_life = 1e30; // no decay

        auto run = [&](int n_gen, int n_prompt) {
            common_ngram_varlen ng(params, 1);
            tokens p;
            for (int k = 0; k < n_prompt; ++k) {
                p.insert(p.end(), { 10, 11, 12, 2, 2, (llama_token) (100 + k) });
            }
            ng.begin(0, p);
            ng.end(0);
            for (int k = 0; k < n_gen; ++k) {
                ng.begin(0, { (llama_token) (200 + k) });
                for (llama_token t : { 10, 11, 12, 1, 1, 300 + k }) {
                    ng.add(0, (llama_token) t);
                }
                ng.end(0);
            }
            ng.begin(0, { 400, 10, 11, 12 });
            return ng.draft(0).tokens;
        };
        CHECK(run(1, 2).empty());
        CHECK(run(2, 1).empty());
        CHECK((run(3, 1) == tokens{ 1, 1 }));
    }

    // the new-text test: a prompt that repeats an earlier request adds only its new text
    {
        common_ngram_varlen ng({}, 1);
        tokens p1;
        for (int i = 0; i < 40; ++i) {
            p1.push_back(1000 + i);
        }
        ng.begin(0, p1);
        ng.add(0, 5);
        ng.add(0, 6);
        ng.end(0);
        const size_t n1 = ng.get_n_samples();
        CHECK(n1 == 38);
        CHECK(ng.get_clock() == 42);

        tokens p2 = p1;
        p2.insert(p2.end(), { 5, 6, 7, 8, 9 });
        ng.begin(0, p2);
        CHECK(ng.get_clock() == 42 + 3); // 7 8 9 are new
        CHECK(ng.get_n_samples() == n1 + 1); // one 3-gram inside the new run 7 8 9
        ng.end(0);

        // a run of 16 tokens seen before, elsewhere in the prompt, is not new either
        tokens p3 = { 7000, 7001 };
        p3.insert(p3.end(), p1.begin() + 10, p1.begin() + 30);
        p3.push_back(7002);
        const int64_t c0 = ng.get_clock();
        ng.begin(0, p3);
        CHECK(ng.get_clock() == c0 + 3);
    }

    // expiry at the end of a request
    {
        common_ngram_varlen_params params;
        params.window = 50;
        common_ngram_varlen ng(params, 1);
        tokens p;
        for (int i = 0; i < 40; ++i) {
            p.push_back(2000 + i);
        }
        ng.begin(0, p);
        ng.end(0);
        CHECK(ng.get_n_samples() == 38);
        tokens q;
        for (int i = 0; i < 40; ++i) {
            q.push_back(3000 + i);
        }
        ng.begin(0, q);
        ng.end(0);
        // clock 80: the samples of the first prompt at clock 3..30 are 50 or more tokens old, 10 are left
        CHECK(ng.get_n_samples() == 38 + 10);
    }

    // the hazard supplement: a match of one sample over 12 tokens is no base draft, and a candidate of 8 tokens
    {
        tokens p;
        for (int i = 0; i < 12; ++i) {
            p.push_back(100 + i);
        }
        for (int i = 0; i < 16; ++i) {
            p.push_back(200 + i);
        }
        p.push_back(900);
        for (int i = 0; i < 12; ++i) {
            p.push_back(100 + i);
        }

        common_ngram_varlen ng({}, 1);
        ng.begin(0, p);
        CHECK(ng.draft(0).tokens.empty());

        common_ngram_varlen_draft_opts opts;
        opts.alt = true;
        auto d = ng.draft(0, &opts);
        CHECK(d.tokens.empty() && d.alt.size() == 1);
        CHECK((d.alt[0].tokens == tokens{ 200, 201, 202, 203, 204, 205, 206, 207 }));
        CHECK(d.alt[0].n == 12 && d.alt[0].n_samples == 1 && d.alt[0].support[3] == 1 && d.alt[0].raw[3] == 1.0);

        // before any observation q = 0.5: one token pays (56.9/1.5 < 39.2), two do not (72.4/1.75)
        common_ngram_varlen_hazard hz({});
        size_t m = 0;
        CHECK(hz.select(d.alt, ng.get_clock(), m) == 0 && m == 1);

        // a rejection is seen at the first position: it does not pay any more
        common_ngram_varlen_hazard::track t;
        hz.start(t, { { 12, 1.0, 1 } }, ng.get_clock());
        hz.piece(t, 1, 0);
        CHECK(!t.used);
        hz.select(d.alt, ng.get_clock() + 1, m);
        CHECK(m == 0);

        // a draft in parts: 3 tokens in pieces of 2, the first piece whole, the sampled token is not the next one
        common_ngram_varlen_hazard_params hp;
        hp.width = 2;
        common_ngram_varlen_hazard hz2(hp);
        hz2.start(t, { { 12, 1.0, 1 }, { 12, 1.0, 1 }, { 12, 1.0, 1 } }, 0);
        hz2.piece(t, 2, 2);
        CHECK(t.used && t.live && t.offset == 2);
        hz2.gate(t, false);
        CHECK(!t.used && hz2.n_cells() == 3);
    }

    return 0;
}

static bool read_ids(const std::string & line, tokens & out) {
    std::istringstream ss(line.substr(1));
    out.clear();
    long long v;
    while (ss >> v) {
        out.push_back((llama_token) v);
    }
    return true;
}

static int run_trace(const char * path, const char * out_path, const common_ngram_varlen_params & params) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    FILE * out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }

    common_ngram_varlen ng(params, 1);

    std::string line;
    tokens prompt;
    tokens gen;
    int req = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == 'P') {
            read_ids(line, prompt);
            continue;
        }
        if (line[0] != 'G') {
            continue;
        }
        read_ids(line, gen);

        ng.begin(0, prompt);
        for (size_t i = 0; i < gen.size(); ++i) {
            ng.add(0, gen[i]);
            if (i + 1 == gen.size()) {
                break;
            }
            const auto d = ng.draft(0);
            fprintf(out, "%d %zu %d %d %zu:", req, i, d.n, d.n_samples, d.tokens.size());
            for (llama_token t : d.tokens) {
                fprintf(out, " %d", t);
            }
            fprintf(out, "\n");
        }
        ng.end(0);
        req++;
    }

    if (out != stdout) {
        fclose(out);
    }
    fprintf(stderr, "requests %d, clock %lld, samples %zu\n", req, (long long) ng.get_clock(), ng.get_n_samples());
    return 0;
}

// With --hazard: the decisions of the hazard supplement on a trace, in the order of the CPU replay. Per decision point
// "<request> <i> <n> <m> <fb>": the matched context, the drafted length and whether the supplement drafted it.
// The acceptance is simulated on the generated tokens as the replay does (pieces of width tokens, a bonus token that
// differs from the next draft token ends the draft; a draft that reaches the end of the request counts nothing).
static bool g_shadow = false;

static int run_hazard(const char * path, const char * out_path, const common_ngram_varlen_params & params,
        const common_ngram_varlen_hazard_params & hp, bool hazard) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    FILE * out = out_path ? fopen(out_path, "wb") : stdout;
    if (!out) {
        fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }

    using clk = std::chrono::steady_clock;
    auto ns = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::nano>(b - a).count(); };

    common_ngram_varlen ng(params, 1);
    common_ngram_varlen_hazard hz(hp);

    common_ngram_varlen_draft_opts opts;
    opts.alt    = hazard;
    opts.shadow = hazard && g_shadow;

    common_ngram_varlen_hazard::shadow_list shadows;
    double t_shadow = 0;
    uint64_t n_labels = 0, n_pend_max = 0;

    double t_lookup = 0, t_select = 0, t_feed = 0;
    uint64_t n_lookup = 0, n_select = 0, n_feed = 0, n_fb = 0, n_prop = 0, n_draft = 0, n_acc = 0, n_alt_calls = 0;
    uint64_t n_out = 0;

    std::string line;
    tokens prompt;
    tokens gen;
    int req = 0;
    const size_t width = (size_t) hp.width;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        if (line[0] == 'P') {
            read_ids(line, prompt);
            continue;
        }
        if (line[0] != 'G') {
            continue;
        }
        read_ids(line, gen);

        ng.begin(0, prompt);
        size_t next_i = 0;
        bool stop = false;
        for (size_t i = 0; i < gen.size() && !stop; ++i) {
            ng.add(0, gen[i]);
            if (i + 1 == gen.size()) {
                break;
            }
            if (i != next_i) {
                continue;
            }

            if (opts.shadow) {
                const auto ts = clk::now();
                hz.shadow_step(shadows, ng.history(0), false);
                n_labels += hz.shadow_flush(shadows, ng.get_clock());
                t_shadow += ns(ts, clk::now());
            }

            auto t0 = clk::now();
            auto d = ng.draft(0, hazard ? &opts : nullptr);
            auto t1 = clk::now();
            t_lookup += ns(t0, t1);
            n_lookup++;

            tokens path;
            int n = 0;
            bool fb = false;
            std::vector<common_ngram_varlen_hazard::feat> feats;
            if (!d.tokens.empty()) {
                path = d.tokens;
                n    = d.n;
                for (size_t j = 0; hazard && j < path.size(); ++j) {
                    feats.push_back({ d.n, d.raw[j], d.support[j] });
                }
            } else if (hazard && !d.alt.empty()) {
                n_alt_calls++;
                size_t m = 0;
                t0 = clk::now();
                const size_t ci = hz.select(d.alt, ng.get_clock(), m);
                t_select += ns(t0, clk::now());
                n_select++;
                if (m > 0) {
                    const auto & c = d.alt[ci];
                    path.assign(c.tokens.begin(), c.tokens.begin() + m);
                    n  = c.n;
                    fb = true;
                    n_fb++;
                    for (size_t j = 0; j < m; ++j) {
                        feats.push_back({ c.n, c.raw[j], c.support[j] });
                    }
                }
            }

            if (opts.shadow) {
                const auto ts = clk::now();
                const int dn = n;
                if (std::any_of(d.shadow.begin(), d.shadow.end(), [&](const auto & c) { return c.n == dn; })) {
                    feats.clear();
                }
                hz.shadow_add(shadows, d.shadow, ng.history(0).size(), ng.get_clock());
                n_pend_max = std::max<uint64_t>(n_pend_max, shadows.size());
                t_shadow += ns(ts, clk::now());
            }

            common_ngram_varlen_hazard::track track;
            if (hazard && !feats.empty()) {
                hz.start(track, feats, ng.get_clock());
            }
            fprintf(out, "%d %zu %d %zu %d\n", req, i, path.empty() ? 0 : n, path.size(), fb ? 1 : 0);

            const size_t length = path.size();
            size_t offset  = 0;
            size_t outputs = 0;
            if (length) {
                n_prop++;
                n_draft += length;
            }
            while (true) {
                const size_t m = length ? std::min(width, length - offset) : 0;
                size_t a = 0;
                while (a < m && i + 1 + outputs + a < gen.size() && path[offset + a] == gen[i + 1 + outputs + a]) {
                    ++a;
                }
                if (i + 1 + outputs + a >= gen.size()) {
                    stop = true; // the request ends in this round: nothing is observed
                    break;
                }
                outputs += a + 1;
                n_acc   += a;
                if (m > 0) {
                    t0 = clk::now();
                    hz.piece(track, m, a);
                    t_feed += ns(t0, clk::now());
                    n_feed++;
                }
                if (a < m || offset + m >= length) {
                    break;
                }
                const bool match = gen[i + outputs] == path[offset + m];
                t0 = clk::now();
                hz.gate(track, match);
                t_feed += ns(t0, clk::now());
                n_feed++;
                if (!match) {
                    break;
                }
                offset += m + 1;
                if (offset >= length) {
                    break;
                }
            }
            common_ngram_varlen_hazard::drop(track);
            if (stop) {
                break;
            }
            n_out += outputs;
            next_i = i + outputs;
        }
        for (size_t k = ng.n_generated(0); k < gen.size(); ++k) {
            ng.add(0, gen[k]);
        }
        if (opts.shadow) {
            hz.shadow_step(shadows, ng.history(0), true);
        }
        ng.end(0);
        req++;
    }

    if (out != stdout) {
        fclose(out);
    }
    if (opts.shadow) {
        fprintf(stderr, "shadow: labels %llu, most anchors waiting %llu, %.3f us per lookup\n", (unsigned long long) n_labels,
                (unsigned long long) n_pend_max, n_lookup ? t_shadow/1e3/(double) n_lookup : 0.0);
    }
    fprintf(stderr, "requests %d, lookups %llu, proposals %llu (supplement %llu of %llu calls with candidates), drafted %llu, accepted %llu, outputs %llu, cells %zu\n",
            req, (unsigned long long) n_lookup, (unsigned long long) n_prop, (unsigned long long) n_fb, (unsigned long long) n_alt_calls,
            (unsigned long long) n_draft, (unsigned long long) n_acc, (unsigned long long) n_out, hz.n_cells());
    fprintf(stderr, "us/call: lookup %.3f (n=%llu), select %.3f (n=%llu), feedback %.3f (n=%llu)\n",
            n_lookup ? t_lookup/1e3/(double) n_lookup : 0.0, (unsigned long long) n_lookup,
            n_select ? t_select/1e3/(double) n_select : 0.0, (unsigned long long) n_select,
            n_feed ? t_feed/1e3/(double) n_feed : 0.0, (unsigned long long) n_feed);
    return 0;
}

int main(int argc, char ** argv) {
    const char * hazard_trace = nullptr;
    bool hazard = true;
    common_ngram_varlen_hazard_params hp;
    const char * trace = nullptr;
    const char * out   = nullptr;
    common_ngram_varlen_params params;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            trace = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else if (!strcmp(argv[i], "--hazard") && i + 1 < argc) {
            hazard_trace = argv[++i];
        } else if (!strcmp(argv[i], "--shadow")) {
            g_shadow = true;
        } else if (!strcmp(argv[i], "--no-hazard")) {
            hazard = false;
        } else if (!strcmp(argv[i], "--cost") && i + 1 < argc) {
            hp.cost.clear();
            std::istringstream cs(argv[++i]);
            std::string part;
            while (std::getline(cs, part, ',')) {
                hp.cost.push_back(atof(part.c_str()));
            }
        } else if (!strcmp(argv[i], "--ladder") && i + 1 < argc) {
            params.ladder = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--p-min") && i + 1 < argc) {
            params.p_min = atof(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--trace FILE | --hazard FILE [--no-hazard] [--shadow] [--cost MS,...]] [--out FILE] [--ladder N] [--p-min P]\n", argv[0]);
            return 1;
        }
    }

    if (trace) {
        return run_trace(trace, out, params);
    }
    if (hazard_trace) {
        return run_hazard(hazard_trace, out, params, hp, hazard);
    }

    const int res = test_cases();
    if (res == 0) {
        printf("test-ngram-varlen: OK\n");
    }
    return res;
}
