#pragma once

// Turn a history of per-request expert counts into fixed-point scores.
// Arithmetic is kept bit-identical to the reference implementation
// (ggml-cuda/expert-profile-store.cu: weighted_request_sum + normalize +
// llround), so a stored profile reproduces the scores it was written with.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace ggml_cuda_expert {

struct scoring_params {
    int    window           = 10;
    double half_life_turns  = 3.0;
    double raw_weight       = 0.75;
    double balanced_weight  = 0.25;
};

static constexpr uint64_t score_scale = 1000000000000ULL;

// records are oldest first and all n_counts long.
inline std::vector<uint64_t> score_records(const std::vector<std::vector<uint64_t>> & records,
        const scoring_params & params) {
    if (records.empty() || records.front().empty()) {
        return std::vector<uint64_t>();
    }
    const size_t n_counts = records.front().size();
    const size_t window   = params.window > 0 ? (size_t) params.window : records.size();
    const size_t used     = std::min(window, records.size());
    const size_t first    = records.size() - used;

    std::vector<double> raw(n_counts, 0.0);
    std::vector<double> balanced(n_counts, 0.0);
    double total_weight = 0.0;
    for (size_t index = 0; index < used; ++index) {
        const std::vector<uint64_t> & record = records[first + index];
        const double age_turns = (double) (used - 1 - index);
        const double weight = std::exp2(-age_turns/params.half_life_turns);
        const uint64_t record_total = std::accumulate(record.begin(), record.end(), uint64_t(0));
        for (size_t i = 0; i < raw.size(); ++i) {
            raw[i] += weight*(double) record[i];
            if (record_total != 0) {
                balanced[i] += weight*(double) record[i]/(double) record_total;
            }
        }
        total_weight += weight;
    }
    const double raw_total = std::accumulate(raw.begin(), raw.end(), 0.0);
    std::vector<uint64_t> scores(n_counts, 0);
    if (raw_total == 0.0 || total_weight == 0.0) {
        return scores;
    }
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = params.raw_weight*raw[i]/raw_total + params.balanced_weight*balanced[i]/total_weight;
    }
    // The normalize step of the reference. The sum is ~1 already, but dividing
    // by it is what the stored scores were produced with.
    std::vector<double> final_score(n_counts, 0.0);
    const double total = std::accumulate(raw.begin(), raw.end(), 0.0);
    if (total > 0.0) {
        for (size_t i = 0; i < raw.size(); ++i) {
            final_score[i] += 1.0*raw[i]/total;
        }
    }
    for (size_t i = 0; i < n_counts; ++i) {
        scores[i] = (uint64_t) llround(final_score[i]*(double) score_scale);
    }
    return scores;
}

// Standard CRC-32 (IEEE 802.3, reflected, polynomial 0xEDB88320).
inline uint64_t crc32(const void * data, size_t bytes, uint64_t seed = 0) {
    static const uint32_t * table = [] {
        static uint32_t values[256];
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            values[i] = c;
        }
        return values;
    }();
    uint32_t crc = ~(uint32_t) seed;
    const unsigned char * bytes_in = (const unsigned char *) data;
    for (size_t i = 0; i < bytes; ++i) {
        crc = table[(crc ^ bytes_in[i]) & 0xFFu] ^ (crc >> 8);
    }
    return (uint64_t) (~crc);
}

} // namespace ggml_cuda_expert
