#pragma once

// Expert cache placement policy. Pure CPU code: no CUDA/HIP headers, no I/O.
// The selection semantics are those of the reference implementation
// (ggml-cuda/expert-cache.cu: sorted_candidates, plan_from_history,
// allocate_cold_capacities, make_fixed_capacity_plan).

#include "expert-geometry.h"
#include "expert-location.h"

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

struct mover_capability {
    bool has_host_master = true;
    int spare_slots = 0;
};

inline bool verify_assignment(const expert_slot_table & gpu, const expert_locations & host,
        const std::vector<int> & layer_class, const install_layout & layout,
        const std::vector<std::vector<int>> & spares, int experts, mover_capability mover,
        std::string & reason) {
    auto fail = [&](const char * text) { reason = text; return false; };
    reason.clear();
    const size_t classes = layout.gpu.size();
    if (experts <= 0 || gpu.size() != host.size() || gpu.size() != layer_class.size() ||
            layout.host.size() != classes || spares.size() != classes) { return fail("assignment dimensions"); }
    std::vector<std::vector<int>> go(classes), ho(classes);
    for (size_t c = 0; c < classes; ++c) {
        if (layout.gpu[c] < 0 || layout.host[c] < 0) { return fail("negative capacity"); }
        go[c].assign(layout.gpu[c], -1); ho[c].assign(layout.host[c], -1);
    }
    for (size_t l = 0; l < gpu.size(); ++l) {
        if (gpu[l].size() != size_t(experts) || host[l].size() != size_t(experts)) { return fail("expert table dimensions"); }
        const int c = layer_class[l];
        if (c < -1 || c >= int(classes)) { return fail("invalid size class"); }
        for (int e = 0; e < experts; ++e) {
            const int g = gpu[l][e];
            const expert_location h = host[l][e];
            if (c < 0) {
                if (g != -1 || h != expert_location{}) { return fail("dense layer has an expert home"); }
                continue;
            }
            if (g < -1 || (h.storage == expert_storage::none && h.slot != -1) ||
                    (h.storage != expert_storage::none && !h.resident())) { return fail("invalid location"); }
            if (g >= 0) {
                if (g >= layout.gpu[c] || go[c][g] != -1) { return fail("VRAM slot claimed twice or out of range"); }
                go[c][g] = int(l)*experts + e;
            }
            if (h.resident()) {
                if (!layout.contains(c, h)) { return fail("host location out of range"); }
                int & owner = ho[c][h.slot];
                if (owner != -1) { return fail("host location claimed twice"); }
                owner = int(l)*experts + e;
            }
            if (mover.has_host_master && g >= 0 && !h.resident()) { return fail("inclusive VRAM resident has no host master"); }
            if (!mover.has_host_master && g >= 0 && h.resident()) { return fail("exclusive expert has two homes"); }
            if (g < 0 && !h.resident()) { return fail("expert has no readable home"); }
        }
    }
    for (size_t c = 0; c < classes; ++c) {
        for (int g : spares[c]) {
            if (g < 0 || g >= layout.gpu[c] || go[c][g] != -1) { return fail("VRAM spare is not free"); }
            go[c][g] = -2;
        }
    }
    return true;
}

struct install_move {
    int layer, expert, cls, batch;
    expert_location from, to;
};

struct install_transaction {
    bool valid = false;
    std::string reason;
    expert_slot_table gpu_slots;
    expert_locations host;
    std::vector<std::vector<int>> gpu_spares;
    std::vector<install_move> moves;
    size_t retained_gpu = 0, retained_host = 0;
    size_t h2d_slices = 0, h2d_bytes = 0, d2h_slices = 0, d2h_bytes = 0;
};

// One ordered transaction for both movers. No device allocation or I/O occurs here.
inline install_transaction plan_install(const geometry & geo,
        const expert_slot_table & gpu_selected, const expert_slot_table & host_selected,
        const expert_slot_table & previous_gpu, const expert_locations & previous_host,
        const std::vector<int> & capacities, const install_layout & before, const install_layout & after,
        const std::vector<std::vector<int>> & gpu_spares, mover_capability mover,
        bool retain = true) {
    install_transaction out;
    auto fail = [&](const char * text) { out.valid = false; out.reason = text; return out; };
    if (!verify_assignment(previous_gpu, previous_host, geo.layer_class, before, gpu_spares,
            geo.n_experts, mover, out.reason)) { return out; }
    const size_t classes = capacities.size(), layers = geo.n_layers;
    if (classes != before.gpu.size() || classes != after.gpu.size() || classes != geo.class_bytes.size() ||
            after.host.size() != classes || gpu_selected.size() != layers || host_selected.size() != layers || previous_gpu.size() != layers ||
            mover.spare_slots < 0) { return fail("install dimensions"); }
    out.gpu_slots.assign(layers, std::vector<int32_t>(geo.n_experts, -1));
    out.host.assign(layers, std::vector<expert_location>(geo.n_experts));
    out.gpu_spares = gpu_spares;
    std::vector<std::vector<bool>> want_g(layers, std::vector<bool>(geo.n_experts)), want_h = want_g;
    std::vector<int> count_g(classes), count_h(classes);
    for (size_t l = 0; l < layers; ++l) {
        const int c = geo.layer_class[l];
        if (c < 0 && (!gpu_selected[l].empty() || !host_selected[l].empty())) { return fail("dense layer selected"); }
        for (int e : gpu_selected[l]) {
            if (e < 0 || e >= geo.n_experts || want_g[l][e]) { return fail("invalid GPU selection"); }
            want_g[l][e] = true; ++count_g[c];
        }
        for (int e : host_selected[l]) {
            if (e < 0 || e >= geo.n_experts || want_h[l][e]) { return fail("invalid host selection"); }
            want_h[l][e] = true; ++count_h[c];
        }
        for (int e = 0; e < geo.n_experts; ++e) {
            if (want_g[l][e] && want_h[l][e] != mover.has_host_master) { return fail("selection violates mover inclusion"); }
            if (c >= 0 && !want_g[l][e] && !want_h[l][e]) { return fail("selection loses an expert"); }
        }
    }
    std::vector<std::vector<expert_location>> pool(classes);
    std::vector<std::vector<bool>> occupied(classes);
    for (size_t c = 0; c < classes; ++c) {
        if (capacities[c] < 0 || after.gpu[c] != before.gpu[c] || capacities[c] > after.gpu[c] ||
                after.host[c] < 0 || count_g[c] > capacities[c] ||
                (!mover.has_host_master && count_g[c] != capacities[c]) ||
                count_h[c] > after.host[c]) { return fail("selection exceeds capacity"); }
        for (int s = 0; s < after.host[c]; ++s) { pool[c].push_back({expert_storage::host, s}); }
        occupied[c].assign(pool[c].size(), false);
    }
    auto index = [&](int c, expert_location at) -> int {
        if (!after.contains(c, at)) { return -1; }
        return at.slot;
    };
    auto emit = [&](int l, int e, int batch, expert_location from, expert_location to) {
        const int c = geo.layer_class[l];
        out.moves.push_back({l, e, c, batch, from, to});
        const size_t bytes = geo.class_total_bytes(c);
        if (to.storage == expert_storage::vram) { ++out.h2d_slices; out.h2d_bytes += bytes; }
        if (from.storage == expert_storage::vram) { ++out.d2h_slices; out.d2h_bytes += bytes; }
    };
    for (size_t l = 0; l < layers; ++l) {
        const int c = geo.layer_class[l];
        if (c < 0) { continue; }
        for (int e = 0; e < geo.n_experts; ++e) {
            const expert_location old = previous_host[l][e];
            const int i = index(c, old);
            if (i >= 0 && (want_h[l][e] || want_g[l][e])) { occupied[c][i] = true; }
            if (i >= 0 && want_h[l][e]) { out.host[l][e] = old; ++out.retained_host; }
        }
    }
    auto release_source = [&](int l, int e) {
        const int c = geo.layer_class[l], i = index(c, previous_host[l][e]);
        if (i >= 0 && out.host[l][e] != previous_host[l][e]) { occupied[c][i] = false; }
    };
    auto fill_host = [&](int l, int e, int batch, expert_location source) {
        const int c = geo.layer_class[l];
        for (size_t i = 0; i < pool[c].size(); ++i) {
            if (occupied[c][i]) { continue; }
            emit(l, e, batch, source, pool[c][i]);
            out.host[l][e] = pool[c][i]; occupied[c][i] = true;
            release_source(l, e);
            return true;
        }
        return false;
    };
    // Retained destinations keep their address, so moves out of a retiring range cannot form a cycle.
    // With a host master every VRAM slice is promoted from its host home, so the host table must
    // be complete before the promotions below read it. The same loop runs again at the end for the
    // exclusive path, where demotions free the slots this pre-pass could not.
    if (mover.has_host_master) {
        for (size_t l = 0; l < layers; ++l) for (int e : host_selected[l]) {
            if (!out.host[l][e].resident() && !fill_host(int(l), e, 0, previous_host[l][e])) {
                return fail("host master cannot be installed");
            }
        }
    }
    for (size_t c = 0; c < classes; ++c) {
        std::vector<std::pair<int, int>> incoming, outgoing;
        std::vector<bool> gpu_taken(after.gpu[c], false);
        for (size_t l = 0; l < layers; ++l) {
            if (geo.layer_class[l] != int(c)) { continue; }
            for (int e = 0; e < geo.n_experts; ++e) {
                const int g = previous_gpu[l][e];
                if (want_g[l][e] && g >= 0 && (retain || !mover.has_host_master)) {
                    out.gpu_slots[l][e] = g; gpu_taken[g] = true; ++out.retained_gpu;
                } else {
                    if (want_g[l][e]) { incoming.emplace_back(int(l), e); }
                    if (g >= 0) { outgoing.emplace_back(int(l), e); }
                }
            }
        }
        if (mover.has_host_master) {
            size_t slot = 0;
            for (auto item : incoming) {
                while (slot < gpu_taken.size() && gpu_taken[slot]) { ++slot; }
                if (slot == gpu_taken.size()) { return fail("no free VRAM slot"); }
                emit(item.first, item.second, 0, out.host[item.first][item.second], {expert_storage::vram, int(slot)});
                out.gpu_slots[item.first][item.second] = int(slot); gpu_taken[slot++] = true;
            }
        } else {
            if (incoming.size() != outgoing.size() || (!incoming.empty() &&
                    (mover.spare_slots == 0 || gpu_spares[c].size() < size_t(mover.spare_slots)))) {
                return fail("exclusive rotation lacks pairs or spares");
            }
            int batch = 0;
            for (size_t base = 0; base < incoming.size(); base += size_t(mover.spare_slots), ++batch) {
                const size_t n = std::min(size_t(mover.spare_slots), incoming.size() - base);
                for (size_t j = 0; j < n; ++j) {
                    const auto item = incoming[base + j];
                    const auto source = previous_host[item.first][item.second];
                    if (!source.resident()) { return fail("promotion has no source"); }
                    emit(item.first, item.second, batch, source, {expert_storage::vram, out.gpu_spares[c][j]});
                    out.gpu_slots[item.first][item.second] = out.gpu_spares[c][j];
                    release_source(item.first, item.second);
                }
                for (size_t j = 0; j < n; ++j) {
                    const auto item = outgoing[base + j];
                    const int g = previous_gpu[item.first][item.second];
                    if (want_h[item.first][item.second] && !fill_host(item.first, item.second, batch, {expert_storage::vram, g})) {
                        return fail("demotion has no destination");
                    }
                    out.gpu_spares[c][j] = g;
                }
            }
        }
    }
    for (size_t l = 0; l < layers; ++l) for (int e : host_selected[l]) {
        if (!out.host[l][e].resident() && !fill_host(int(l), e, 0, previous_host[l][e])) {
            return fail("host resident cannot be installed");
        }
    }
    out.valid = verify_assignment(out.gpu_slots, out.host, geo.layer_class, after, out.gpu_spares,
        geo.n_experts, mover, out.reason);
    return out;
}

} // namespace ggml_cuda_expert
