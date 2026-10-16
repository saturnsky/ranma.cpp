// ranma: the variable-length n-gram drafter (common/ngram-varlen.h) on the CPU.
//
// Without arguments: small fixed cases (cohort majority, ladder, support, prompt weight, new-text test, expiry).
// With --trace FILE: replays a token trace and writes the draft of every step, in the format of the reference script
// of the CPU replay, to compare the two.
//   trace: per request a line "P <prompt ids>" and a line "G <generated ids>"
//   output: per generated token but the last "<request> <step> <n> <samples> <length>: <ids>"
#include "ngram-varlen.h"

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

int main(int argc, char ** argv) {
    const char * trace = nullptr;
    const char * out   = nullptr;
    common_ngram_varlen_params params;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--trace") && i + 1 < argc) {
            trace = argv[++i];
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out = argv[++i];
        } else if (!strcmp(argv[i], "--ladder") && i + 1 < argc) {
            params.ladder = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--p-min") && i + 1 < argc) {
            params.p_min = atof(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [--trace FILE [--out FILE] [--ladder N] [--p-min P]]\n", argv[0]);
            return 1;
        }
    }

    if (trace) {
        return run_trace(trace, out, params);
    }

    const int res = test_cases();
    if (res == 0) {
        printf("test-ngram-varlen: OK\n");
    }
    return res;
}
