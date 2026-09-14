#pragma once

// Expert cache placement policy. Pure CPU code: no CUDA/HIP headers, no I/O.
// The selection semantics are those of the reference implementation
// (ggml-cuda/expert-cache.cu: sorted_candidates, plan_from_history,
// allocate_cold_capacities, make_fixed_capacity_plan).

#include "expert-geometry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ggml_cuda_expert {

// Fisher-Yates with a specified generator, independent of standard-library shuffle details.
inline std::vector<uint64_t> seeded_expert_scores(const geometry & geo, uint32_t seed) {
    uint64_t state = seed;
    auto next = [&]() {
        uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
        z = (z ^ (z >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27))*UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    };
    std::vector<uint64_t> scores(geo.n_counts(), 0);
    std::vector<int> ids(geo.n_experts);
    for (int l = 0; l < geo.n_layers; ++l) {
        if (geo.layer_class[l] < 0) { continue; }
        std::iota(ids.begin(), ids.end(), 0);
        for (int i = geo.n_experts - 1; i > 0; --i) { std::swap(ids[i], ids[next() % (i + 1)]); }
        for (int i = 0; i < geo.n_experts; ++i) { scores[size_t(l)*geo.n_experts + ids[i]] = geo.n_experts - i; }
    }
    return scores;
}

struct placement_inputs {
    const geometry * geo = nullptr;
    const uint64_t * counts = nullptr;  // n_counts scores; nullptr or all-zero total = cold start
    size_t budget_bytes = 0;            // bytes available for expert slices (overhead already subtracted)
    bool   exclusive = false;           // fill every slot even for zero-count experts
    const std::vector<int> * fixed_capacities = nullptr; // re-plan with frozen per-class capacities
};

struct placement_stats {
    uint64_t total = 0, selected_hits = 0, total_bytes = 0, selected_bytes = 0;
    int per_layer_min = 0, per_layer_max = 0, choices = 0;

    // Fractions in [0, 1]; the log line multiplies these by 100.
    double selection_hit() const {
        return total == 0 ? 0.0 : (double) selected_hits/(double) total;
    }
    double byte_hit() const {
        return total_bytes == 0 ? 0.0 : (double) selected_bytes/(double) total_bytes;
    }
};

struct placement {
    std::vector<std::vector<int32_t>> selected; // [layer] sorted ascending expert ids
    std::vector<int>                  capacities; // [class] slot count
    placement_stats                   stats;
};

namespace detail {

struct plan_candidate {
    int      layer;
    int      expert;
    int      cls;
    uint64_t count;
};

inline std::vector<plan_candidate> sorted_candidates(const geometry & geo, const uint64_t * counts) {
    std::vector<plan_candidate> candidates;
    candidates.reserve(geo.n_counts());
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        if (geo.layer_class[layer] < 0) {
            continue;
        }
        for (int expert = 0; expert < geo.n_experts; ++expert) {
            const size_t index = (size_t) layer*(size_t) geo.n_experts + (size_t) expert;
            candidates.push_back({layer, expert, geo.layer_class[layer], counts ? counts[index] : uint64_t(0)});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const plan_candidate & a, const plan_candidate & b) {
        if (a.count != b.count) {
            return a.count > b.count;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    });
    return candidates;
}

// Cold start: split the budget between size classes in proportion to their
// layer counts so a later refresh has stable capacity to fill. The
// proportional divisor is the full layer span, so a model
// with leading dense layers under-allocates in the first pass and recovers in
// the greedy loop below.
inline std::vector<int> allocate_cold_capacities(const geometry & geo, size_t remaining) {
    const int n_classes = (int) geo.class_bytes.size();
    std::vector<int> capacity(n_classes, 0);
    if (n_classes == 0 || geo.n_layers == 0) {
        return capacity;
    }
    size_t used = 0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const size_t bytes = geo.class_total_bytes(cls);
        const size_t share = remaining*(size_t) geo.class_layers[cls]/(size_t) geo.n_layers;
        capacity[cls] = std::min<int>(geo.class_layers[cls]*geo.n_experts, bytes == 0 ? 0 : int(share/bytes));
        used += (size_t) capacity[cls]*bytes;
    }
    remaining -= std::min(remaining, used);
    while (true) {
        int best = -1;
        for (int cls = 0; cls < n_classes; ++cls) {
            const size_t bytes = geo.class_total_bytes(cls);
            if (capacity[cls] < geo.class_layers[cls]*geo.n_experts && bytes <= remaining &&
                    (best < 0 || bytes < geo.class_total_bytes(best))) {
                best = cls;
            }
        }
        if (best < 0) {
            break;
        }
        ++capacity[best];
        remaining -= geo.class_total_bytes(best);
    }
    return capacity;
}

inline void finish_stats(const geometry & geo, const std::vector<std::vector<int32_t>> & selected,
        placement_stats & stats) {
    // The reference scans every layer, including dense ones, so per_layer_min is 0 for
    // any model with a non-routed layer. Replicated on purpose.
    int total = 0;
    int min_count = geo.n_experts;
    int max_count = 0;
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        const int count = (int) selected[layer].size();
        total += count;
        min_count = std::min(min_count, count);
        max_count = std::max(max_count, count);
    }
    stats.choices       = total;
    stats.per_layer_min = geo.n_layers == 0 ? 0 : min_count;
    stats.per_layer_max = max_count;
}

} // namespace detail

inline placement plan_placement(const placement_inputs & in) {
    placement result;
    if (in.geo == nullptr || in.geo->n_layers == 0 || in.geo->n_experts == 0) {
        return result;
    }
    const geometry & geo = *in.geo;
    result.selected.assign(geo.n_layers, std::vector<int32_t>());
    result.capacities.assign(geo.class_bytes.size(), 0);

    uint64_t total = 0;
    if (in.counts != nullptr) {
        const size_t n = geo.n_counts();
        for (size_t i = 0; i < n; ++i) {
            total += in.counts[i];
        }
    }
    result.stats.total = total;

    const std::vector<detail::plan_candidate> candidates = detail::sorted_candidates(geo, in.counts);

    if (in.fixed_capacities != nullptr) {
        result.capacities = *in.fixed_capacities;
        result.capacities.resize(geo.class_bytes.size(), 0);
        std::vector<int> used(result.capacities.size(), 0);
        for (const detail::plan_candidate & c : candidates) {
            const size_t bytes = geo.class_total_bytes(c.cls);
            result.stats.total_bytes += c.count*bytes;
            if ((!in.exclusive && c.count == 0) || used[c.cls] >= result.capacities[c.cls]) {
                continue;
            }
            result.selected[c.layer].push_back(c.expert);
            ++used[c.cls];
            result.stats.selected_hits  += c.count;
            result.stats.selected_bytes += c.count*bytes;
        }
        for (int layer = 0; layer < geo.n_layers; ++layer) {
            std::sort(result.selected[layer].begin(), result.selected[layer].end());
        }
        detail::finish_stats(geo, result.selected, result.stats);
        return result;
    }

    if (total != 0) {
        // Warm start: greedy global top-N by score within the byte budget. The
        // resulting per-class slot counts become the fixed class capacities.
        size_t remaining = in.budget_bytes;
        for (const detail::plan_candidate & c : candidates) {
            const size_t bytes = geo.class_total_bytes(c.cls);
            result.stats.total_bytes += c.count*bytes;
            if ((c.count != 0 || in.exclusive) && bytes <= remaining) {
                result.selected[c.layer].push_back(c.expert);
                ++result.capacities[c.cls];
                remaining -= bytes;
                result.stats.selected_hits  += c.count;
                result.stats.selected_bytes += c.count*bytes;
            }
        }
        for (int layer = 0; layer < geo.n_layers; ++layer) {
            std::sort(result.selected[layer].begin(), result.selected[layer].end());
        }
        detail::finish_stats(geo, result.selected, result.stats);
        return result;
    }

    result.capacities = detail::allocate_cold_capacities(geo, in.budget_bytes);
    if (in.exclusive) {
        // Cold and exclusive: every slot must be occupied, so fill the just
        // allocated capacities in deterministic layer/expert order.
        placement_inputs fixed = in;
        fixed.fixed_capacities = &result.capacities;
        placement filled = plan_placement(fixed);
        result.selected = std::move(filled.selected);
        result.stats.total_bytes    = filled.stats.total_bytes;
        result.stats.selected_hits  = filled.stats.selected_hits;
        result.stats.selected_bytes = filled.stats.selected_bytes;
    }
    detail::finish_stats(geo, result.selected, result.stats);
    return result;
}

// ---- three tiers: VRAM, host, SSD (stage 6) ---------------------------------------------------
//
// With a finite host budget the routed experts no longer fit in VRAM plus host memory, and the
// remainder stays in the GGUF file. The cut is the same greedy the VRAM tier uses, one level down:
// every expert that the VRAM plan did not take is a candidate, the score is the selection count
// times the bytes the choice would save reading, and the per-class slot count grows while the
// budget allows. This is the reference plan_l2_slots without its rule that forces
// layers 0 and 1 to be resident; that rule was the lead time of a speculative prefetch that was
// measured and rejected, so ranma has no protected layers (design 11 and 13-1).

} // namespace ggml_cuda_expert
