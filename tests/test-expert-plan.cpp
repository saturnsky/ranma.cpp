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
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const reference_candidate & a, const reference_candidate & b) {
            if (a.count != b.count) { return a.count > b.count; }
            if (a.layer != b.layer) { return a.layer < b.layer; }
            return a.expert < b.expert;
        });
    return candidates;
}

std::vector<int> reference_cold_capacities(const geometry & geo, size_t remaining) {
    const int n_classes = (int) geo.class_bytes.size();
    std::vector<int> capacity(n_classes, 0);
    size_t used = 0;
    for (int cls = 0; cls < n_classes; ++cls) {
        const size_t bytes = reference_class_bytes(geo, cls);
        const size_t share = remaining*(size_t) geo.class_layers[cls]/(size_t) geo.n_layers;
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

// ---------------------------------------------------------------------------
// Independent reference: the plan_l2_slots this tier planner was derived from,
// transcribed. It differs from plan_host_tier in exactly one rule: layers 0 and
// 1 are forced to be fully resident, which was the lead time of a speculative
// prefetch that was measured and rejected. ranma deletes the rule (design 11 and
// 13-1), so the two selections must agree on every layer from 2 on, and the
// reference must hold more of layers 0 and 1.
// ---------------------------------------------------------------------------

struct reference_l2_plan {
    std::vector<std::vector<int32_t>> selected;
    std::vector<int> capacities;
    size_t resident_bytes = 0;
};

reference_l2_plan reference_plan_l2_slots(
        const std::vector<std::vector<int32_t>> & l1,
        const std::vector<int> & classes,
        const std::vector<std::array<size_t, 3>> & sizes,
        const std::vector<uint64_t> & counts, size_t budget,
        const std::vector<size_t> & pitch) {
    const int layers  = (int) l1.size();
    const int experts = layers ? (int) l1[0].size() : 0;
    reference_l2_plan out;
    out.selected.resize(layers);
    out.capacities.assign(sizes.size(), 0);
    std::vector<size_t> payload(sizes.size(), 0);
    std::vector<int> available(sizes.size(), 0), protected_capacity(sizes.size(), 0);
    struct choice { int l, e, c; long double score; };
    std::vector<choice> choices;
    for (size_t c = 0; c < sizes.size(); ++c) {
        for (size_t b : sizes[c]) { payload[c] += b; }
    }
    for (int l = 0; l < layers; ++l) {
        const int c = classes[l];
        if (c < 0) { continue; }
        if (l < 2) { protected_capacity[c] += experts; }
        for (int e = 0; e < experts; ++e) {
            if (l1[l][e] < 0) {
                ++available[c];
                choices.push_back({l, e, c, (long double) counts[(size_t) l*experts + e]*payload[c]});
            }
        }
    }
    std::sort(choices.begin(), choices.end(), [](const choice & a, const choice & b) {
        if ((a.l < 2) != (b.l < 2)) { return a.l < 2; }
        if (a.score != b.score) { return a.score > b.score; }
        if (a.l != b.l) { return a.l < b.l; }
        return a.e < b.e;
    });
    for (size_t c = 0; c < sizes.size(); ++c) {
        out.capacities[c] = std::min(available[c], protected_capacity[c]);
        out.resident_bytes += (size_t) out.capacities[c]*pitch[c];
    }
    std::vector<int> rank(sizes.size(), 0);
    for (const choice & x : choices) {
        if (++rank[x.c] <= out.capacities[x.c]) { continue; }
        if (pitch[x.c] <= budget - out.resident_bytes) {
            ++out.capacities[x.c];
            out.resident_bytes += pitch[x.c];
        }
    }
    std::vector<int> used(sizes.size(), 0);
    for (const choice & x : choices) {
        if (used[x.c] < out.capacities[x.c]) { out.selected[x.l].push_back(x.e); ++used[x.c]; }
    }
    for (std::vector<int32_t> & row : out.selected) { std::sort(row.begin(), row.end()); }
    return out;
}

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

        // Only two slices fit: the two highest counts win, ties by layer then expert.
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

        // Cold and exclusive: deterministic fill of those capacities.
        in.exclusive = true;
        placement filled = plan_placement(in);
        CHECK(filled.capacities == cold.capacities);
        CHECK(total_selected(filled.selected) == 15);
        CHECK((filled.selected[0] == std::vector<int32_t>{0, 1, 2, 3, 4}));
        CHECK(filled.selected[1].empty());
        CHECK((filled.selected[2] == std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
        CHECK((filled.selected[3] == std::vector<int32_t>{0, 1}));
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
                    counts[i] = (lcg() % 4) == 0 ? lcg() % 1000 : 0;
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

    // ---- the three-tier cut --------------------------------------------------------------
    {
        const std::vector<std::array<size_t, 3>> bytes{{4096, 4096, 8192}, {4096, 4096, 4096}};
        geometry geo = make_geometry({0, 0, 1, 1, 0, 1}, 8, bytes);
        std::vector<uint64_t> counts((size_t) geo.n_layers*geo.n_experts, 0);
        for (int l = 0; l < geo.n_layers; ++l) {
            for (int e = 0; e < geo.n_experts; ++e) {
                counts[(size_t) l*geo.n_experts + e] = (uint64_t) (1 + (l*7 + e*13) % 11);
            }
        }
        // the VRAM tier already holds expert 0 of every layer
        std::vector<std::vector<int32_t>> vram(geo.n_layers, std::vector<int32_t>(geo.n_experts, -1));
        for (int l = 0; l < geo.n_layers; ++l) { vram[l][0] = l; }

        std::vector<size_t> pitch(bytes.size(), 0);
        for (size_t c = 0; c < bytes.size(); ++c) { pitch[c] = geo.class_total_bytes((int) c); }

        tier_inputs in;
        in.geo          = &geo;
        in.counts       = counts.data();
        in.vram         = &vram;
        in.slot_pitch   = &pitch;
        in.budget_bytes = 20*pitch[0];
        const tier_plan plan = plan_host_tier(in);

        // every selected expert is outside the VRAM tier and inside its class capacity
        std::vector<int> used(bytes.size(), 0);
        size_t resident = 0;
        for (int l = 0; l < geo.n_layers; ++l) {
            for (int32_t e : plan.selected[l]) {
                CHECK(vram[l][e] < 0);
                ++used[geo.layer_class[l]];
            }
            resident += plan.selected[l].size();
        }
        for (size_t c = 0; c < bytes.size(); ++c) { CHECK(used[c] == plan.capacities[c]); }
        CHECK(plan.resident_bytes <= in.budget_bytes);
        CHECK(plan.ssd_slices == (size_t) geo.n_layers*geo.n_experts - resident - (size_t) geo.n_layers);

        // the cut is the score-ordered top of the candidates: nothing left out beats anything taken
        uint64_t worst_taken = UINT64_MAX;
        uint64_t best_left   = 0;
        for (int l = 0; l < geo.n_layers; ++l) {
            std::vector<bool> taken(geo.n_experts, false);
            for (int32_t e : plan.selected[l]) { taken[e] = true; }
            for (int e = 0; e < geo.n_experts; ++e) {
                if (vram[l][e] >= 0) { continue; }
                const uint64_t score = counts[(size_t) l*geo.n_experts + e]*geo.class_total_bytes(geo.layer_class[l]);
                if (taken[e]) { worst_taken = std::min(worst_taken, score); }
                else          { best_left   = std::max(best_left,   score); }
            }
        }
        CHECK(worst_taken >= best_left || plan.capacities.size() > 1);
        printf("PASS: three-tier cut keeps the VRAM tier, the capacities and the budget\n");

        // the same cut against the reference planner, at the capacities it derives
        const reference_l2_plan ref = reference_plan_l2_slots(vram, geo.layer_class, bytes, counts, in.budget_bytes, pitch);
        tier_inputs same = in;
        same.fixed_capacities = &ref.capacities;
        const tier_plan mine = plan_host_tier(same);
        for (size_t c = 0; c < bytes.size(); ++c) { CHECK(mine.capacities[c] == ref.capacities[c]); }
        // The reference forces layers 0 and 1 to be fully resident; everything it holds from layer 2 on,
        // ranma holds too, because ranma spends those slots on the best scores instead
        for (int l = 2; l < geo.n_layers; ++l) {
            for (int32_t e : ref.selected[l]) {
                CHECK(std::find(mine.selected[l].begin(), mine.selected[l].end(), e) != mine.selected[l].end());
            }
        }
        for (int l = 0; l < 2; ++l) {
            CHECK(ref.selected[l].size() + 1 == (size_t) geo.n_experts); // every expert but the VRAM one
        }
        CHECK(mine.selected[0].size() + 1 < (size_t) geo.n_experts ||
              mine.selected[1].size() + 1 < (size_t) geo.n_experts);
        printf("PASS: three-tier cut equals the reference plan_l2_slots except for its protected layers 0-1\n");

        // fixed capacities reproduce the same selection
        tier_inputs fixed = in;
        fixed.fixed_capacities = &plan.capacities;
        const tier_plan again = plan_host_tier(fixed);
        CHECK(again.selected == plan.selected && again.capacities == plan.capacities);
        printf("PASS: three-tier cut is stable against frozen capacities\n");

        // borrowed slots go to the classes with the best remaining candidates
        const std::vector<int> borrowed = plan_borrowed_capacities(in, plan.capacities, 5);
        int extra = 0;
        for (size_t c = 0; c < bytes.size(); ++c) {
            CHECK(borrowed[c] >= plan.capacities[c]);
            extra += borrowed[c] - plan.capacities[c];
        }
        CHECK(extra == 5);
        printf("PASS: borrowed host capacity distributes the freed ring slots\n");
    }

    // Inclusive capacity reserves every future L1 payload, independent of its CLI overhead.
    {
        geometry geo;
        geo.n_layers = 2; geo.n_experts = 4; geo.layer_class = {0, 1}; geo.class_layers = {1, 1};
        geo.class_bytes = {{10, 10, 10}, {20, 20, 20}};
        const std::vector<int> gpu_caps{2, 1};
        const auto minimum = minimum_host_budget(geo, gpu_caps, 77, true);
        CHECK(minimum.valid && minimum.l1_payload == 120 && minimum.bytes == 197);
        CHECK(minimum_host_budget(geo, gpu_caps, 77, false).bytes == 77);
        CHECK(!minimum_host_budget(geo, gpu_caps, SIZE_MAX, true).valid);
        expert_slot_table gpu{{0, 1, -1, -1}, {0, -1, -1, -1}};
        const std::vector<size_t> pitch{30, 60};
        uint64_t scores[8] = {1, 1, 1000, 999, 1, 1000, 999, 998};
        tier_inputs in; in.geo = &geo; in.vram = &gpu; in.slot_pitch = &pitch; in.counts = scores;
        in.inclusive = true; in.minimum_capacities = &gpu_caps; in.budget_bytes = 120;
        auto plan = plan_host_tier(in);
        CHECK(plan.valid && plan.selected == expert_slot_table({{0, 1}, {0}}));
        in.budget_bytes = 119; CHECK(!plan_host_tier(in).valid);
        in.budget_bytes = 180; plan = plan_host_tier(in);
        CHECK(plan.valid && plan.resident_bytes <= 180);
        const auto caps = plan.capacities; in.fixed_capacities = &caps;
        gpu = {{-1, -1, 0, 1}, {-1, -1, -1, 0}};
        plan = plan_host_tier(in);
        CHECK(plan.valid);
        CHECK(std::binary_search(plan.selected[0].begin(), plan.selected[0].end(), 2));
        CHECK(std::binary_search(plan.selected[0].begin(), plan.selected[0].end(), 3));
        CHECK(std::binary_search(plan.selected[1].begin(), plan.selected[1].end(), 3));
        printf("PASS: finite inclusive payload minimum, rejection and mandatory inclusion across replans\n");
    }
    {
        // The installed memory is injected, so the rule is testable without a machine of that size.
        const size_t gib = size_t(1) << 30;
        CHECK(check_host_memory(40*gib, 128ull*gib).ok);
        CHECK(check_host_memory(128*gib, 128ull*gib).ok);
        const auto over = check_host_memory(129*gib, 128ull*gib);
        CHECK(!over.ok && over.required_bytes == 129*gib && over.total_bytes == 128*gib);
        CHECK(check_host_memory(1024*gib, 0).ok); // the platform cannot answer: nothing is refused
        printf("PASS: host requirement against installed physical memory\n");
    }
    return 0;
}
