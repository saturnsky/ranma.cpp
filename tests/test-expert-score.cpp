#include "expert-score.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

using namespace ggml_cuda_expert;

namespace {

uint64_t sum(const std::vector<uint64_t> & values) {
    uint64_t total = 0;
    for (uint64_t v : values) { total += v; }
    return total;
}

} // namespace

int main() {
    // crc32 known vectors.
    CHECK(crc32("", 0) == 0);
    CHECK(crc32("123456789", 9) == 0xCBF43926ULL);
    CHECK(crc32("a", 1) == 0xE8B7BE43ULL);
    {
        // Chaining through the seed must equal one pass over the whole buffer.
        const char * text = "the quick brown fox";
        const uint64_t whole = crc32(text, strlen(text));
        const uint64_t split = crc32(text + 4, strlen(text) - 4, crc32(text, 4));
        CHECK(whole == split);
    }

    const scoring_params params;

    // Empty input.
    CHECK(score_records({}, params).empty());

    // All-zero history scores zero.
    {
        const std::vector<uint64_t> zero(6, 0);
        const std::vector<uint64_t> scores = score_records({zero, zero}, params);
        CHECK(scores.size() == 6 && sum(scores) == 0);
    }

    // A single record scores proportionally to its counts.
    {
        const std::vector<uint64_t> record{10, 0, 30, 60};
        const std::vector<uint64_t> scores = score_records({record}, params);
        CHECK(scores.size() == 4);
        CHECK(scores[1] == 0);
        const uint64_t total = sum(scores);
        CHECK(total >= score_scale - 4 && total <= score_scale + 4);
        // Fixed point, so allow the last few units of rounding slack.
        CHECK(scores[0] > score_scale/10 - 8 && scores[0] < score_scale/10 + 8);
        CHECK(scores[2] > 3*(score_scale/10) - 8 && scores[2] < 3*(score_scale/10) + 8);
        CHECK(scores[3] > 6*(score_scale/10) - 8 && scores[3] < 6*(score_scale/10) + 8);
    }

    // Window truncation: only the newest 10 of 12 records are used.
    {
        std::vector<std::vector<uint64_t>> records;
        for (int i = 0; i < 2; ++i) {
            records.push_back({1000000, 0, 0, 0});  // ancient and huge, must be ignored
        }
        for (int i = 0; i < 10; ++i) {
            records.push_back({0, 0, 5, 5});
        }
        const std::vector<uint64_t> scores = score_records(records, params);
        CHECK(scores[0] == 0 && scores[1] == 0);
        CHECK(scores[2] == scores[3]);
        const std::vector<std::vector<uint64_t>> newest(records.begin() + 2, records.end());
        CHECK(score_records(newest, params) == scores);
        // With a larger window the old records do show up.
        scoring_params wide = params;
        wide.window = 12;
        CHECK(score_records(records, wide)[0] > 0);
    }

    // Half-life: a one-turn-older record is weighted by exp2(-1/3).
    {
        const std::vector<std::vector<uint64_t>> records{{100, 0}, {0, 100}};
        const std::vector<uint64_t> scores = score_records(records, params);
        CHECK(scores[1] > scores[0]);
        const double ratio = (double) scores[1]/(double) scores[0];
        CHECK(std::fabs(ratio - std::exp2(1.0/3.0)) < 1e-6);
        // A longer half life flattens the ratio.
        scoring_params slow = params;
        slow.half_life_turns = 100.0;
        const std::vector<uint64_t> flat = score_records(records, slow);
        CHECK((double) flat[1]/(double) flat[0] < ratio);
    }

    // A zero-count record contributes weight but no balanced mass.
    {
        const std::vector<std::vector<uint64_t>> records{{0, 0}, {3, 1}};
        const std::vector<uint64_t> scores = score_records(records, params);
        CHECK(scores[0] > scores[1] && sum(scores) >= score_scale - 4);
    }

    printf("PASS: crc32 vectors, proportionality, window truncation and half-life weighting\n");

    return 0;
}
