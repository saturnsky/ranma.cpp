#include "expert-plan.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

using namespace ggml_cuda_expert;

namespace {

// ---------------------------------------------------------------------------
// Independent reference: a direct transcription of the planner this one was
// derived from (sorted_candidates / allocate_cold_capacities /
// plan_from_history / make_fixed_capacity_plan), written against the same
// inputs but sharing no code with expert-plan.h.
// ---------------------------------------------------------------------------

struct reference_result {
    std::vector<std::vector<int32_t>> selected;
    std::vector<int> capacities;
};

struct reference_candidate {
    int layer;
    int expert;
    int cls;
    uint64_t count;
};

// The tie order written the direct way: repeatedly take, among the remaining items with the best
// key, the one whose layer has the fewest items taken so far, then the lower layer, then the lower
// expert. Quadratic; shares no code with sort_balancing_ties.
template <typename T, typename Before>
std::vector<T> reference_balanced_order(std::vector<T> items, int n_layers, Before before) {
    std::vector<T> out;
    std::vector<int> taken(n_layers, 0);
    while (!items.empty()) {
        size_t best = 0;
        for (size_t i = 1; i < items.size(); ++i) {
            const T & a = items[i];
            const T & b = items[best];
            if (before(b, a)) { continue; }
            if (before(a, b)) { best = i; continue; }
            if (taken[a.layer] != taken[b.layer]) { best = taken[a.layer] < taken[b.layer] ? i : best; continue; }
            if (a.layer != b.layer) { best = a.layer < b.layer ? i : best; continue; }
            best = a.expert < b.expert ? i : best;
        }
        out.push_back(items[best]);
        ++taken[items[best].layer];
        items.erase(items.begin() + best);
    }
    return out;
}

size_t reference_class_bytes(const geometry & geo, int cls) {
    return geo.class_bytes[cls][0] + geo.class_bytes[cls][1] + geo.class_bytes[cls][2];
}

std::vector<reference_candidate> reference_sorted_candidates(const geometry & geo, const uint64_t * counts) {
    std::vector<reference_candidate> candidates;
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        if (geo.layer_class[layer] < 0) { continue; }
        for (int expert = 0; expert < geo.n_experts; ++expert) {
            const uint64_t count = counts ? counts[(size_t) layer*geo.n_experts + expert] : uint64_t(0);
            candidates.push_back({layer, expert, geo.layer_class[layer], count});
        }
    }
    return reference_balanced_order(candidates, geo.n_layers,
        [](const reference_candidate & a, const reference_candidate & b) { return a.count > b.count; });
}

std::vector<int> reference_cold_capacities(const geometry & geo, size_t remaining) {
    const int n_classes = (int) geo.class_bytes.size();
    std::vector<int> capacity(n_classes, 0);
    // proportional to the routed layers only: dense and empty layer numbers do not count
    size_t routed = 0;
    for (int layer = 0; layer < geo.n_layers; ++layer) { routed += geo.layer_class[layer] >= 0 ? 1 : 0; }
    if (routed == 0) { return capacity; }
    size_t used = 0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const size_t bytes = reference_class_bytes(geo, cls);
        const size_t share = remaining*(size_t) geo.class_layers[cls]/routed;
        capacity[cls] = std::min<int>(geo.class_layers[cls]*geo.n_experts, int(share/bytes));
        used += (size_t) capacity[cls]*bytes;
    }
    remaining -= std::min(remaining, used);
    while (true) {
        int best = -1;
        for (int cls = 0; cls < n_classes; ++cls) {
            const size_t bytes = reference_class_bytes(geo, cls);
            if (capacity[cls] < geo.class_layers[cls]*geo.n_experts && bytes <= remaining &&
                    (best < 0 || bytes < reference_class_bytes(geo, best))) {
                best = cls;
            }
        }
        if (best < 0) { break; }
        ++capacity[best];
        remaining -= reference_class_bytes(geo, best);
    }
    return capacity;
}

reference_result reference_fixed_capacity_plan(const geometry & geo, const uint64_t * counts,
        const std::vector<int> & capacities, bool exclusive) {
    reference_result out;
    out.selected.assign(geo.n_layers, std::vector<int32_t>());
    out.capacities = capacities;
    std::vector<int> used(capacities.size(), 0);
    for (const reference_candidate & c : reference_sorted_candidates(geo, counts)) {
        if ((!exclusive && c.count == 0) || used[c.cls] >= capacities[c.cls]) { continue; }
        out.selected[c.layer].push_back(c.expert);
        ++used[c.cls];
    }
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        std::sort(out.selected[layer].begin(), out.selected[layer].end());
    }
    return out;
}

reference_result reference_plan(const geometry & geo, const uint64_t * counts, size_t budget,
        bool exclusive, const std::vector<int> * fixed) {
    if (fixed != nullptr) {
        return reference_fixed_capacity_plan(geo, counts, *fixed, exclusive);
    }
    uint64_t total = 0;
    if (counts != nullptr) {
        for (size_t i = 0; i < geo.n_counts(); ++i) { total += counts[i]; }
    }
    reference_result out;
    out.selected.assign(geo.n_layers, std::vector<int32_t>());
    out.capacities.assign(geo.class_bytes.size(), 0);
    if (total != 0) {
        size_t remaining = budget;
        for (const reference_candidate & c : reference_sorted_candidates(geo, counts)) {
            const size_t bytes = reference_class_bytes(geo, c.cls);
            if ((c.count != 0 || exclusive) && bytes <= remaining) {
                out.selected[c.layer].push_back(c.expert);
                ++out.capacities[c.cls];
                remaining -= bytes;
            }
        }
        for (int layer = 0; layer < geo.n_layers; ++layer) {
            std::sort(out.selected[layer].begin(), out.selected[layer].end());
        }
        return out;
    }
    out.capacities = reference_cold_capacities(geo, budget);
    if (exclusive) {
        out.selected = reference_fixed_capacity_plan(geo, counts, out.capacities, true).selected;
    }
    return out;
}

// ---------------------------------------------------------------------------

geometry make_geometry(const std::vector<int> & layer_class, int experts,
        const std::vector<std::array<size_t, 3>> & class_bytes) {
    geometry geo;
    geo.n_layers    = (int) layer_class.size();
    geo.n_experts   = experts;
    geo.layer_class = layer_class;
    geo.class_bytes = class_bytes;
    geo.class_layers.assign(class_bytes.size(), 0);
    geo.nb2.assign(geo.n_layers, {0, 0, 0});
    geo.tensors.assign(geo.n_layers, {nullptr, nullptr, nullptr});
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        const int cls = layer_class[layer];
        if (cls >= 0) {
            ++geo.class_layers[cls];
            geo.nb2[layer] = class_bytes[cls];
        }
    }
    return geo;
}

uint64_t lcg_state = 88172645463325252ULL;
uint64_t lcg() {
    lcg_state = lcg_state*6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state >> 17;
}

int total_selected(const std::vector<std::vector<int32_t>> & selected) {
    int total = 0;
    for (const auto & layer : selected) { total += (int) layer.size(); }
    return total;
}

} // namespace

int main() {
    // --- small hand-checked cases -----------------------------------------
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}};
        geometry geo = make_geometry({0, 0, 0, 0}, 4, bytes);
        std::vector<uint64_t> counts(geo.n_counts(), 0);
        counts[0*4 + 2] = 50;  // layer 0, expert 2
        counts[1*4 + 1] = 70;  // layer 1, expert 1
        counts[3*4 + 0] = 70;  // layer 3, expert 0

        placement_inputs in;
        in.geo = &geo;
        in.counts = counts.data();
        in.budget_bytes = 3*300;
        placement plan = plan_placement(in);
        CHECK(plan.stats.total == 190);
        CHECK(plan.capacities.size() == 1 && plan.capacities[0] == 3);
        CHECK((plan.selected[0] == std::vector<int32_t>{2}));
        CHECK((plan.selected[1] == std::vector<int32_t>{1}));
        CHECK(plan.selected[2].empty());
        CHECK((plan.selected[3] == std::vector<int32_t>{0}));
        CHECK(plan.stats.choices == 3);
        CHECK(plan.stats.per_layer_min == 0 && plan.stats.per_layer_max == 1);
        CHECK(plan.stats.selected_hits == 190 && plan.stats.total_bytes == 190*300);
        CHECK(plan.stats.selection_hit() == 1.0 && plan.stats.byte_hit() == 1.0);

        // Determinism.
        placement again = plan_placement(in);
        CHECK(again.selected == plan.selected && again.capacities == plan.capacities);

        // Budget boundary: less than one slice selects nothing.
        in.budget_bytes = 299;
        placement tight = plan_placement(in);
        CHECK(total_selected(tight.selected) == 0);
        CHECK(tight.capacities[0] == 0);
        CHECK(tight.stats.selected_hits == 0 && tight.stats.byte_hit() == 0.0);

        // Only two slices fit: the two highest counts win.
        in.budget_bytes = 2*300 + 299;
        placement two = plan_placement(in);
        CHECK(total_selected(two.selected) == 2);
        CHECK((two.selected[1] == std::vector<int32_t>{1}));
        CHECK((two.selected[3] == std::vector<int32_t>{0}));

        // Exclusive warm start fills the remaining budget with zero-count experts.
        in.budget_bytes = 5*300;
        in.exclusive = true;
        placement exclusive = plan_placement(in);
        CHECK(total_selected(exclusive.selected) == 5);
        CHECK(exclusive.capacities[0] == 5);
        in.exclusive = false;
    }

    // Cold start, inclusive: capacities only, no selection.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}, {50, 50, 50}};
        geometry geo = make_geometry({0, 0, 1, 1}, 8, bytes);
        placement_inputs in;
        in.geo = &geo;
        in.budget_bytes = 3000;
        placement cold = plan_placement(in);
        CHECK(total_selected(cold.selected) == 0);
        CHECK(cold.stats.total == 0);
        // proportional: 1500/300 = 5 and 1500/150 = 10, then the leftover 0
        CHECK(cold.capacities[0] == 5 && cold.capacities[1] == 10);

        // Cold and exclusive: deterministic fill of those capacities, a round robin over the layers
        // of each class from the front (a layer-first order would give layers 1 and 3 {} and {0, 1}).
        in.exclusive = true;
        placement filled = plan_placement(in);
        CHECK(filled.capacities == cold.capacities);
        CHECK(total_selected(filled.selected) == 15);
        CHECK((filled.selected[0] == std::vector<int32_t>{0, 1, 2}));
        CHECK((filled.selected[1] == std::vector<int32_t>{0, 1}));
        CHECK((filled.selected[2] == std::vector<int32_t>{0, 1, 2, 3, 4}));
        CHECK((filled.selected[3] == std::vector<int32_t>{0, 1, 2, 3, 4}));
    }

    // Cold start with dense layers between and after the routed ones: the split divides by the four
    // routed layers, not the span of eight. Dividing by the span gave {2, 16}: half the
    // proportional share each, and the greedy remainder went to the smaller class.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}, {50, 50, 50}};
        geometry geo = make_geometry({-1, -1, 0, 0, -1, -1, 1, 1}, 8, bytes);
        CHECK(geo.n_routed_layers() == 4);
        placement_inputs in;
        in.geo = &geo;
        in.budget_bytes = 3000;
        placement cold = plan_placement(in);
        CHECK(cold.capacities[0] == 5 && cold.capacities[1] == 10);
        // a remainder that fits neither share whole still goes to the greedy loop
        in.budget_bytes = 3000 + 299;
        cold = plan_placement(in);
        CHECK(cold.capacities[0] == 5 && cold.capacities[1] == 11);
        // nothing routed: no capacity
        geometry dense = make_geometry({-1, -1}, 8, bytes);
        in.geo = &dense;
        cold = plan_placement(in);
        CHECK(cold.capacities[0] == 0 && cold.capacities[1] == 0);
        printf("PASS: cold class split divides by the routed layers (dense and empty layer numbers excluded)\n");
    }

    // Cold fill across layers: a class whose last layer (layer 4) follows two dense layers. The
    // round robin gives the last layer its share; a layer-first order would give {0, 1, 2, 3},
    // {0}, {}.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}};
        geometry geo = make_geometry({0, 0, -1, -1, 0}, 4, bytes);
        placement_inputs in;
        in.geo = &geo;
        in.budget_bytes = 5*300;
        in.exclusive = true;
        const placement filled = plan_placement(in);
        CHECK(filled.capacities[0] == 5);
        CHECK((filled.selected[0] == std::vector<int32_t>{0, 1}));
        CHECK((filled.selected[1] == std::vector<int32_t>{0, 1}));
        CHECK(filled.selected[2].empty() && filled.selected[3].empty());
        CHECK((filled.selected[4] == std::vector<int32_t>{0}));
        // a scored plan keeps its order; only exact ties spread: of the two zero-count fills after
        // the two scored experts, the first goes to layer 1 (no expert yet), the second to layer 0
        std::vector<uint64_t> counts(geo.n_counts(), 0);
        counts[0*4 + 1] = 9;
        counts[4*4 + 3] = 7;
        in.counts = counts.data();
        in.budget_bytes = 4*300;
        const placement warm = plan_placement(in);
        CHECK((warm.selected[0] == std::vector<int32_t>{0, 1}));
        CHECK((warm.selected[1] == std::vector<int32_t>{0}));
        CHECK((warm.selected[4] == std::vector<int32_t>{3}));
        printf("PASS: cold L1 fills spread over the layers of a class in a round robin\n");
    }

    // Two classes: each class gets its own round robin; the rounding slot of class 0 goes to its
    // front layer.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}, {100, 100, 100}};
        geometry geo = make_geometry({0, 1, 0, 1, 0}, 6, bytes);
        placement_inputs in;
        in.geo = &geo;
        in.budget_bytes = 11*300;
        in.exclusive = true;
        const placement filled = plan_placement(in);
        CHECK(filled.capacities[0] == 7 && filled.capacities[1] == 4);
        CHECK((filled.selected[0] == std::vector<int32_t>{0, 1, 2}));
        CHECK((filled.selected[2] == std::vector<int32_t>{0, 1}));
        CHECK((filled.selected[4] == std::vector<int32_t>{0, 1}));
        CHECK((filled.selected[1] == std::vector<int32_t>{0, 1}));
        CHECK((filled.selected[3] == std::vector<int32_t>{0, 1}));
        printf("PASS: cold fill is a round robin over the layers of each class\n");
    }

    // Scored plan whose layers tie on different expert sets: each next tied pick goes to the layer
    // with the fewest picks so far, so the per-layer counts stay even. Ordering ties by expert index
    // then layer would give {4, 0, 2, 2}: layer 0 holds the low indices, layer 1 the high ones.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}};
        geometry geo = make_geometry({0, 0, 0, 0}, 8, bytes);
        const std::vector<std::vector<int>> tied{{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 6, 7}, {2, 3, 4, 5}};
        std::vector<uint64_t> counts(geo.n_counts(), 0);
        for (int l = 0; l < 4; ++l) {
            for (int e : tied[l]) { counts[l*8 + e] = 5; }
        }
        auto per_layer = [](const placement & p) {
            std::vector<size_t> n;
            for (const auto & row : p.selected) { n.push_back(row.size()); }
            return n;
        };
        std::vector<int> caps{8};
        placement_inputs in;
        in.geo = &geo;
        in.counts = counts.data();
        in.fixed_capacities = &caps;
        placement plan = plan_placement(in);
        CHECK((per_layer(plan) == std::vector<size_t>{2, 2, 2, 2}));
        CHECK((plan.selected[1] == std::vector<int32_t>{4, 5}));
        CHECK((plan.selected[2] == std::vector<int32_t>{0, 1}));

        // the order that ranks ties by expert index, then layer, for comparison
        std::vector<std::pair<int, int>> by_expert;   // (expert, layer)
        for (int l = 0; l < 4; ++l) {
            for (int e : tied[l]) { by_expert.emplace_back(e, l); }
        }
        std::sort(by_expert.begin(), by_expert.end());
        std::vector<size_t> skewed(4, 0);
        for (size_t i = 0; i < 8; ++i) { ++skewed[by_expert[i].second]; }
        CHECK((skewed == std::vector<size_t>{4, 0, 2, 2}));

        // a higher-scored expert counts: layer 0 already has one, so the ties go to layers 1-3 first
        counts[0*8 + 7] = 9;
        caps[0] = 8;
        plan = plan_placement(in);
        CHECK((per_layer(plan) == std::vector<size_t>{2, 2, 2, 2}));
        CHECK((plan.selected[0] == std::vector<int32_t>{0, 7}));
        caps[0] = 9;
        plan = plan_placement(in);
        CHECK((per_layer(plan) == std::vector<size_t>{3, 2, 2, 2}));
        printf("PASS: equal scores on different expert sets per layer keep the per-layer counts even\n");
    }

    // Fixed capacities re-plan.
    {
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 100}};
        geometry geo = make_geometry({-1, 0, 0}, 4, bytes);
        std::vector<uint64_t> counts(geo.n_counts(), 0);
        counts[1*4 + 3] = 9;
        counts[2*4 + 0] = 5;
        counts[2*4 + 1] = 1;
        std::vector<int> capacities{2};
        placement_inputs in;
        in.geo = &geo;
        in.counts = counts.data();
        in.budget_bytes = 0;  // ignored on the fixed-capacity path
        in.fixed_capacities = &capacities;
        placement plan = plan_placement(in);
        CHECK(total_selected(plan.selected) == 2);
        CHECK(plan.selected[0].empty());
        CHECK((plan.selected[1] == std::vector<int32_t>{3}));
        CHECK((plan.selected[2] == std::vector<int32_t>{0}));
        CHECK(plan.capacities == capacities);
    }

    // --- randomized cross-check against the reference ---------------------
    {
        int cold_cases = 0, warm_cases = 0, fixed_cases = 0, exclusive_cases = 0;
        for (int iteration = 0; iteration < 200; ++iteration) {
            const int n_classes = 1 + int(lcg() % 3);
            const int n_layers  = 4 + int(lcg() % 9);
            const int n_experts = 8 + int(lcg() % 57);
            std::vector<std::array<size_t, 3>> class_bytes(n_classes);
            for (int cls = 0; cls < n_classes; ++cls) {
                for (int kind = 0; kind < 3; ++kind) {
                    class_bytes[cls][kind] = 64 + (size_t) (lcg() % 4096);
                }
            }
            std::vector<int> layer_class(n_layers, -1);
            for (int cls = 0; cls < n_classes; ++cls) {
                layer_class[cls] = cls;  // every class owns at least one layer
            }
            for (int layer = n_classes; layer < n_layers; ++layer) {
                const uint64_t roll = lcg() % 8;
                layer_class[layer] = roll == 0 ? -1 : int(lcg() % (uint64_t) n_classes);
            }
            geometry geo = make_geometry(layer_class, n_experts, class_bytes);

            std::vector<uint64_t> counts(geo.n_counts(), 0);
            const bool cold = (iteration % 5) == 0;
            if (!cold) {
                for (size_t i = 0; i < counts.size(); ++i) {
                    // odd iterations: few distinct values, so scored plans also have ties above zero
                    counts[i] = (lcg() % 4) == 0 ? ((iteration & 1) ? 1 + lcg() % 3 : lcg() % 1000) : 0;
                }
            }
            uint64_t total = 0;
            for (uint64_t c : counts) { total += c; }

            const size_t max_bytes = (size_t) n_layers*(size_t) n_experts*3*4096;
            const size_t budget = (size_t) (lcg() % (max_bytes + 1));
            const bool exclusive = (lcg() % 2) == 0;
            const bool use_fixed = (iteration % 3) == 0;

            std::vector<int> fixed(n_classes, 0);
            for (int cls = 0; cls < n_classes; ++cls) {
                fixed[cls] = int(lcg() % (uint64_t) (geo.class_layers[cls]*n_experts + 1));
            }

            placement_inputs in;
            in.geo = &geo;
            in.counts = counts.data();
            in.budget_bytes = budget;
            in.exclusive = exclusive;
            in.fixed_capacities = use_fixed ? &fixed : nullptr;

            const placement plan = plan_placement(in);
            const reference_result expected =
                reference_plan(geo, counts.data(), budget, exclusive, use_fixed ? &fixed : nullptr);
            CHECK(plan.selected == expected.selected);
            CHECK(plan.capacities == expected.capacities);
            CHECK(plan.stats.total == total);

            // A plan must never exceed its own capacities or the budget.
            std::vector<int> used(n_classes, 0);
            size_t used_bytes = 0;
            for (int layer = 0; layer < n_layers; ++layer) {
                const int cls = layer_class[layer];
                if (cls < 0) { CHECK(plan.selected[layer].empty()); continue; }
                used[cls] += (int) plan.selected[layer].size();
                used_bytes += plan.selected[layer].size()*geo.class_total_bytes(cls);
                CHECK((int) plan.selected[layer].size() <= n_experts);
                CHECK(std::is_sorted(plan.selected[layer].begin(), plan.selected[layer].end()));
                CHECK(std::adjacent_find(plan.selected[layer].begin(), plan.selected[layer].end()) ==
                    plan.selected[layer].end());
            }
            for (int cls = 0; cls < n_classes; ++cls) {
                CHECK(used[cls] <= plan.capacities[cls]);
            }
            if (!use_fixed && total != 0) {
                CHECK(used_bytes <= budget);
            }

            cold_cases      += cold ? 1 : 0;
            warm_cases      += cold ? 0 : 1;
            fixed_cases     += use_fixed ? 1 : 0;
            exclusive_cases += exclusive ? 1 : 0;
        }
        CHECK(cold_cases > 0 && warm_cases > 0 && fixed_cases > 0 && exclusive_cases > 0);
        printf("PASS: 200 randomized plans match the independent reference (cold=%d warm=%d fixed=%d exclusive=%d)\n",
            cold_cases, warm_cases, fixed_cases, exclusive_cases);
    }

    return 0;
}
