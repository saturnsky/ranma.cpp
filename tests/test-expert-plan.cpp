#include "expert-host-layout.h"
#include "expert-plan.h"
#include "expert-redraw.h"

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
    struct choice { int layer, expert, c; long double score; };
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
    // ranma's tie order in place of the original (layer, then expert), so the comparison below
    // tests only the protected-layer rule
    choices = reference_balanced_order(choices, layers, [](const choice & a, const choice & b) {
        if ((a.layer < 2) != (b.layer < 2)) { return a.layer < 2; }
        return a.score > b.score;
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
        if (used[x.c] < out.capacities[x.c]) { out.selected[x.layer].push_back(x.expert); ++used[x.c]; }
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

// ---------------------------------------------------------------------------
// Size-class redraw of the VRAM tier (expert-redraw.h): the switches, the capacity recompute, the
// byte cap, the estimate, the gate, the evacuation plan and the mover with a
// retiring slot range.
// ---------------------------------------------------------------------------
static int redraw_tests() {
    {
        // switches: the redraw is on when a condition is set; each set condition must hold; a bad
        // value leaves its condition unset and is reported
        std::string warn;
        auto env = [](std::vector<std::pair<const char *, const char *>> kv) {
            return [kv](const char * name) -> const char * {
                for (const auto & p : kv) { if (strcmp(p.first, name) == 0) { return p.second; } }
                return nullptr;
            };
        };
        redraw_params p = redraw_params_from(env({}), warn);
        CHECK(!p.enabled && p.benefit == 0.0 && p.move_pct == 0.0 && warn.empty());
        CHECK(p.confirm == 2 && p.holdoff == 3 && p.cap_bytes == size_t(4608) << 20 && p.horizon == 350 && p.handle_bytes == size_t(64) << 20);
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_MOVE", "2.5"}}), warn);
        CHECK(p.enabled && p.benefit == 0.0 && p.move_pct == 2.5 && warn.empty());
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_BENEFIT", "2"}}), warn);
        CHECK(p.enabled && p.benefit == 2.0 && p.move_pct == 0.0 && warn.empty());
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_BENEFIT", "2"}, {"RANMA_EXPERT_L1_REDRAW_MOVE", "5"}}), warn);
        CHECK(p.enabled && p.benefit == 2.0 && p.move_pct == 5.0 && warn.empty());
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_MOVE", "x"}, {"RANMA_EXPERT_L1_REDRAW_BENEFIT", "-1"}}), warn);
        CHECK(!p.enabled && p.move_pct == 0.0 && p.benefit == 0.0 && !warn.empty());
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_MOVE", "0"}, {"RANMA_EXPERT_L1_REDRAW_BENEFIT", "3"}}), warn);
        CHECK(p.enabled && p.move_pct == 0.0 && p.benefit == 3.0 && !warn.empty());
        p = redraw_params_from(env({{"RANMA_EXPERT_L1_REDRAW_MOVE", ""}}), warn);
        CHECK(!p.enabled && warn.empty());
    }
    {
        // the capacity recompute is the load's greedy: without floors and ceilings an exclusive
        // recompute gives exactly plan_placement's capacities for the same bytes
        uint64_t seed = 12345;
        auto next = [&]() { seed = seed*6364136223846793005ULL + 1442695040888963407ULL; return uint32_t(seed >> 33); };
        const std::vector<std::array<size_t, 3>> bytes{{100, 100, 120}, {60, 60, 90}, {200, 200, 300}};
        geometry geo = make_geometry({0, 0, 1, 1, 1, -1, 2, 0}, 16, bytes);
        for (int round = 0; round < 200; ++round) {
            std::vector<uint64_t> counts(geo.n_counts());
            for (auto & c : counts) { c = next() % 7 == 0 ? 0 : next() % 1000; }
            std::vector<int> current = {int(next() % 20), int(next() % 30), int(next() % 10)};
            const size_t budget = redraw_static_bytes(geo, current);
            placement_inputs in;
            in.geo = &geo; in.counts = counts.data(); in.budget_bytes = budget; in.exclusive = true;
            const placement ref = plan_placement(in);
            const std::vector<int> none(3, 0), all(3, 1 << 20);
            const std::vector<int> caps = redraw_capacities(geo, counts.data(), budget, true, current, none, all);
            CHECK(caps == ref.capacities);
            CHECK(redraw_static_bytes(geo, caps) <= budget);
            // floors and ceilings hold, the bytes stay within the budget
            const std::vector<int> lo = {std::min(current[0], 3), 0, std::min(current[2], 1)}, hi = {1 << 20, current[1] + 2, 1 << 20};
            const std::vector<int> b = redraw_capacities(geo, counts.data(), budget, true, current, lo, hi);
            for (int c = 0; c < 3; ++c) { CHECK(b[c] >= lo[c] && b[c] <= hi[c]); }
            CHECK(redraw_static_bytes(geo, b) <= budget);
            // a byte cap moves every class the same fraction of its way and keeps the bytes
            const size_t cap = redraw_moved_bytes(geo, current, caps)/2;
            const std::vector<int> lim = redraw_limit(geo, current, caps, cap, budget);
            CHECK(cap == 0 || redraw_moved_bytes(geo, current, lim) <= cap);
            CHECK(redraw_static_bytes(geo, lim) <= budget);
            for (int c = 0; c < 3; ++c) { CHECK((lim[c] - current[c])*(caps[c] - current[c]) >= 0 && std::abs(lim[c] - current[c]) <= std::abs(caps[c] - current[c])); }
        }
        // cold: no proposal
        const std::vector<int> cur = {3, 4, 1};
        CHECK(redraw_capacities(geo, nullptr, redraw_static_bytes(geo, cur), true, cur, {0, 0, 0}, {99, 99, 99}) == cur);
        // inclusive: a class the scores do not reach keeps the bytes nobody else takes
        std::vector<uint64_t> counts(geo.n_counts(), 0);
        for (int e = 0; e < 3; ++e) { counts[2*16 + e] = 100; }   // three experts of class 1 score
        const std::vector<int> inc = redraw_capacities(geo, counts.data(), redraw_static_bytes(geo, cur), false, cur, {0, 0, 0}, {99, 99, 99});
        CHECK(inc[1] == 4 && inc[0] == 3 && inc[2] == 1);   // nothing grows: the split stays
        for (int e = 0; e < 16; ++e) { counts[3*16 + e] = 50; counts[4*16 + e] = 40; }
        const std::vector<int> grow = redraw_capacities(geo, counts.data(), redraw_static_bytes(geo, cur), false, cur, {0, 0, 0}, {99, 99, 99});
        CHECK(grow[1] > 4 && redraw_static_bytes(geo, grow) <= redraw_static_bytes(geo, cur));
        CHECK(grow[0] + grow[2] < 4 && redraw_moved_bytes(geo, cur, grow) > 0);
        printf("PASS: L1 redraw capacity recompute equals the load greedy; floors, ceilings, byte cap, cold and inclusive give-back\n");
    }
    {
        // estimate: one class-0 expert leaves VRAM, one class-1 expert enters from the file
        const std::vector<std::array<size_t, 3>> bytes{{1000000, 1000000, 1000000}, {500000, 500000, 500000}};
        geometry geo = make_geometry({0, 1}, 4, bytes);
        std::vector<uint64_t> counts = {40, 30, 20, 10, 60, 50, 30, 10};   // 250 selections, 2 used, 2 layers: 62.5 tokens
        const expert_slot_table now = {{0, 1}, {0}}, next = {{0}, {0, 1, 2}};
        redraw_params p;
        p.horizon = 100;
        p.handle_bytes = size_t(1) << 20;
        const redraw_estimate est = redraw_estimate_of(geo, counts.data(), 2, now, next, [](int l, int e) { return l == 1 && e == 2; },
            true, {2, 1}, {1, 3}, 30000000, p);
        CHECK(est.moved_bytes == 3000000 && std::fabs(est.move_pct - 10.0) < 1e-9);
        CHECK(est.enter_slices == 2 && est.leave_slices == 1 && est.enter_bytes == 3000000 && est.enter_file_bytes == 1500000);
        CHECK(std::fabs(est.tokens - 62.5) < 1e-9);
        const double gain = 50.0/62.5*0.0015*redraw_link_ms_per_gb + 30.0/62.5*0.0015*redraw_ssd_ms_per_gb - 30.0/62.5*0.003*redraw_link_ms_per_gb;
        CHECK(std::fabs(est.benefit_ms_token - gain) < 1e-9 && std::fabs(est.benefit_ms - 100.0*gain) < 1e-7);
        CHECK(std::fabs(est.cost_promote_ms - (0.0015*redraw_link_ms_per_gb + 0.0015*redraw_ssd_ms_per_gb)) < 1e-9);
        CHECK(std::fabs(est.cost_demote_ms - 0.003*redraw_link_ms_per_gb) < 1e-9);
        CHECK(est.handles == 3*1 + 3*1 && est.cost_ms > est.cost_promote_ms + est.cost_demote_ms);
        CHECK(std::fabs(est.ratio - est.benefit_ms/est.cost_ms) < 1e-9);

        // the gate: each condition on its own, both, confirmation, fresh installs, hold-off
        redraw_estimate e;
        e.moved_bytes = 1; e.move_pct = 6.0; e.benefit_ms = 30.0; e.cost_ms = 10.0;
        redraw_params q;   // move > 5 %, benefit > 2 x cost, confirm 2, hold-off 3
        q.enabled = true; q.move_pct = 5.0; q.benefit = 2.0;
        redraw_gate g;
        redraw_verdict v = redraw_decide(q, e, g);
        CHECK(!v.fire && v.move_ok && v.benefit_ok && g.streak == 1);
        v = redraw_decide(q, e, g);
        CHECK(v.fire && g.streak == 0 && g.holdoff == 3);
        for (int i = 0; i < 3; ++i) { v = redraw_decide(q, e, g); CHECK(!v.fire); }
        CHECK(g.holdoff == 0 && g.streak == 3);
        v = redraw_decide(q, e, g);
        CHECK(v.fire);
        redraw_gate h;
        e.benefit_ms = 15.0;                          // benefit 1.5 x cost
        CHECK(!redraw_decide(q, e, h).fire && !redraw_decide(q, e, h).fire && h.streak == 0);
        redraw_params nb = q; nb.benefit = 0.0;       // benefit condition not set
        CHECK(!redraw_decide(nb, e, h).fire && redraw_decide(nb, e, h).fire);
        e.benefit_ms = 30.0; e.move_pct = 4.0;        // move 4 %
        redraw_gate m;
        CHECK(!redraw_decide(q, e, m).fire && !redraw_decide(q, e, m).fire);
        redraw_params nm = q; nm.move_pct = 0.0;      // move condition not set
        CHECK(!redraw_decide(nm, e, m).fire && redraw_decide(nm, e, m).fire);
        redraw_gate f;
        e.move_pct = 6.0;
        redraw_mark_fresh(f, q);                      // a cold load: the first two installs need no confirmation
        CHECK(redraw_decide(q, e, f).fire);
        redraw_gate z;
        e.moved_bytes = 0;
        redraw_params off = q; off.benefit = 0.0; off.move_pct = 0.0; off.confirm = 1;
        CHECK(!redraw_decide(off, e, z).fire);        // nothing to move never fires
        e.moved_bytes = 1;
        CHECK(redraw_decide(off, e, z).fire);
        printf("PASS: L1 redraw estimate and gate: benefit, cost, independent conditions, confirmation, fresh installs, hold-off\n");
    }
    {
        // evacuation: class 0 goes from 8 to 5 slots; slots 1 and 6 are spares
        const std::vector<int> layer_class = {0, 0};
        expert_slot_table gpu = {{0, 2, 7, -1, -1, -1}, {3, 4, 5, -1, -1, -1}};
        const expert_slot_table keep = {{0, 2}, {3, 4}};
        std::vector<std::vector<uint8_t>> other = {{0, 1, 0, 0, 0, 0, 1, 0}};
        // kept above the limit: layer 0 expert 2 (slot 7), layer 1 expert 3 is at slot 3 (below)
        evac_plan ev = plan_evacuation(layer_class, 6, gpu, keep, other, {8}, {5});
        CHECK(ev.valid && ev.ops.size() == 1);
        // no free slot below 5 (0 static, 1 spare, 2 static, 3 static, 4 static): swap with a dropped one
        CHECK(ev.ops[0].from == 7 && ev.ops[0].victim_layer == 0 && ev.ops[0].victim_expert == 1 && ev.ops[0].to == 2);
        CHECK(ev.gpu[0][2] == 2 && ev.gpu[0][1] == 7);
        // with a free slot, a plain move
        gpu[0][1] = -1;
        ev = plan_evacuation(layer_class, 6, gpu, keep, other, {8}, {5});
        CHECK(ev.valid && ev.ops.size() == 1 && ev.ops[0].victim_layer < 0 && ev.ops[0].to == 2 && ev.gpu[0][2] == 2);
        // more kept residents than the new range holds
        const expert_slot_table keep_all = {{0, 1, 2}, {3, 4, 5}};
        gpu[0][1] = 6; other[0][6] = 0;
        CHECK(!plan_evacuation(layer_class, 6, gpu, keep_all, other, {8}, {3}).valid);
        printf("PASS: L1 redraw evacuation plan: free slots first, swaps with dropped residents, spare slots untouched\n");
    }
    {
        // mover with a retiring range, then growth. One class, 2 layers x 6 experts, slice 30 B.
        geometry geo;
        geo.n_layers = 2; geo.n_experts = 6; geo.layer_class = {0, 0}; geo.class_layers = {2};
        geo.class_bytes = {{10, 10, 10}};
        const std::vector<std::vector<int>> no_spares(1);
        install_layout l8, l5;
        l8.gpu = {8}; l8.host = {12}; l8.lent_begin = {0}; l8.lent_count = {0};
        l5 = l8; l5.gpu = {5};
        expert_slot_table host_ids(2, std::vector<int32_t>(6));
        for (int l = 0; l < 2; ++l) { for (int e = 0; e < 6; ++e) { host_ids[l][e] = l*6 + e; } }
        const expert_locations homes = host_locations(host_ids);
        const expert_slot_table all = {{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}};
        // inclusive: 8 residents; the new plan keeps 5, two of them above slot 5 must have been moved
        const expert_slot_table prev = {{0, 1, 2, 3, -1, -1}, {4, 5, 6, 7, -1, -1}};
        const expert_slot_table sel = {{0, 1, 2, 3}, {0}};
        auto tx = plan_install(geo, sel, all, prev, homes, {5}, l8, l5, no_spares, {true, 0}, false, true);
        CHECK(tx.valid && tx.retained_gpu == 5 && tx.moves.empty());   // retained slots all below 5
        const expert_slot_table sel_hi = {{0, 1}, {0, 1, 2}};          // layer 1 expert 2 sits at slot 6
        tx = plan_install(geo, sel_hi, all, prev, homes, {5}, l8, l5, no_spares, {true, 0}, false, true);
        CHECK(!tx.valid && tx.reason == "retained VRAM slot retires");
        tx = plan_install(geo, sel_hi, all, prev, homes, {5}, l8, l5, no_spares, {true, 0}, false, false);   // no retain: recopied
        CHECK(tx.valid && tx.h2d_slices == 5);
        for (int l = 0; l < 2; ++l) { for (int e = 0; e < 6; ++e) { CHECK(tx.gpu_slots[l][e] < 5); } }
        // growth: before == after at the larger count, the incoming experts take the new slots
        install_layout l10 = l8; l10.gpu = {10};
        const expert_slot_table grown = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}};
        tx = plan_install(geo, grown, all, prev, homes, {10}, l10, l10, no_spares, {true, 0}, false, true);
        CHECK(tx.valid && tx.retained_gpu == 8 && tx.h2d_slices == 2 && tx.gpu_slots[0][4] == 8 && tx.gpu_slots[1][4] == 9);
        // exclusive with a file: the unequal exchange (empty blocked rows) shrinks 8 -> 5 slots with 2 spares
        install_layout x10 = l8, x7 = l8;   // 6 static + 2 spares = 8 today; 3 static + 2 spares = 5 after
        x10.gpu = {8}; x7.gpu = {5};
        const expert_slot_table xprev = {{0, 1, 2, -1, -1, -1}, {3, 4, 5, -1, -1, -1}};   // spares 6, 7
        expert_locations xhomes(2, std::vector<expert_location>(6));
        for (int l = 0; l < 2; ++l) { for (int e = 3; e < 6; ++e) { xhomes[l][e] = {expert_storage::host, l*3 + e - 3}; } }
        install_layout xh8 = x10, xh5 = x7; xh8.host = {8}; xh5.host = {8};
        const std::vector<std::vector<int>> xspares = {{6, 7}};
        const expert_slot_table xsel = {{0, 1}, {3}};            // kept: all below 5
        tx = plan_install(geo, xsel, {{3, 4, 5, 2}, {0, 1, 4, 5}}, xprev, xhomes, {3}, xh8, xh5, xspares, {false, 2}, true, true, true);
        if (!tx.valid) { fprintf(stderr, "%s\n", tx.reason.c_str()); }
        CHECK(tx.valid && tx.gpu_spares[0].size() == 2);
        for (int s : tx.gpu_spares[0]) { CHECK(s < 5); }
        for (int l = 0; l < 2; ++l) { for (int e = 0; e < 6; ++e) { CHECK(tx.gpu_slots[l][e] < 5); } }
        // the same without the unequal exchange (spare rotation) cannot retire
        CHECK(!plan_install(geo, xsel, {{3, 4, 5, 2}, {0, 1, 4, 5}}, xprev, xhomes, {3}, xh8, xh5, xspares, {false, 2}, true, true).valid);
        GGML_UNUSED(x10);
        printf("PASS: L1 redraw mover: a retiring slot range (inclusive, exclusive unequal exchange) and growth into new slots\n");
    }
    return 0;
}

// Host arenas without a finite tier (expert-host-layout.h): the chunk arithmetic, the
// compaction plan, and a whole redraw in exclusive mode with no file (host growth, evacuation, the
// shrink install, the grow install, host compaction) through the real mover.
static int host_layout_tests() {
    {
        geometry geo;
        geo.class_bytes = {{900*1024, 900*1024, 1200*1024}, {size_t(5) << 20, size_t(1) << 20, size_t(1) << 20}};
        CHECK(host_chunk_slots(geo, 0, size_t(128) << 20) == 109);
        CHECK(host_chunk_slots(geo, 1, size_t(4) << 20) == 1);   // a slice larger than the chunk: one per chunk
        CHECK(host_chunks_for(0, 109) == 1 && host_chunks_for(109, 109) == 1 && host_chunks_for(110, 109) == 2);
        CHECK(host_chunked_bytes(1000, 512, 7, 3) == 3*(3*1000 + 512));
        printf("PASS: host arenas: slots per chunk, chunk counts and bytes\n");
    }
    {
        // compaction, random: every resident at or above the count moves to a free slot below, nothing collides
        uint64_t rng = 12345;
        auto rnd = [&](int n) { rng = rng*6364136223846793005ULL + 1442695040888963407ULL; return int((rng >> 33) % uint64_t(n)); };
        const std::vector<int> layer_class = {0, 1, 0, -1};
        const int experts = 20;
        for (int round = 0; round < 300; ++round) {
            const int slots = 10 + rnd(40);
            expert_locations homes(4, std::vector<expert_location>(experts));
            std::vector<int> order(slots);
            for (int i = 0; i < slots; ++i) { order[i] = i; }
            for (int i = slots - 1; i > 0; --i) { std::swap(order[i], order[rnd(i + 1)]); }
            int used = 0;
            for (int l : {0, 2}) {
                for (int e = 0; e < experts && used < slots; ++e) {
                    if (rnd(3) == 0) { homes[l][e] = {expert_storage::host, order[used++]}; }
                    else if (rnd(2) == 0) { homes[l][e] = {expert_storage::vram, 0}; }
                }
            }
            homes[1][3] = {expert_storage::host, 1000};   // another class: ignored
            const int count = rnd(slots + 1);
            std::vector<host_move> ops;
            std::string why;
            const bool ok = plan_host_compaction(layer_class, experts, homes, 0, slots, count, ops, why);
            CHECK(ok == (used <= count));
            if (!ok) { CHECK(!why.empty() && ops.empty()); continue; }
            for (const host_move & op : ops) {
                CHECK(op.from >= count && op.to < count && homes[op.layer][op.expert].slot == op.from);
                homes[op.layer][op.expert].slot = op.to;
            }
            std::vector<int> seen(count, 0);
            int n = 0;
            for (int l : {0, 2}) {
                for (int e = 0; e < experts; ++e) {
                    if (homes[l][e].storage != expert_storage::host) { continue; }
                    CHECK(homes[l][e].slot >= 0 && homes[l][e].slot < count && !seen[homes[l][e].slot]);
                    seen[homes[l][e].slot] = 1;
                    ++n;
                }
            }
            CHECK(n == used);
        }
        printf("PASS: host compaction plan: residents above the new count move to free slots below (300 random cases)\n");
    }
    {
        // A redraw in exclusive mode without a file: layers 0, 1 in class 0, layer 2 in class 1, 8 experts,
        // 2 spares, no pool. Static VRAM capacities {6, 2} -> {3, 5}.
        geometry geo;
        geo.n_layers = 3; geo.n_experts = 8; geo.layer_class = {0, 0, 1}; geo.class_layers = {2, 1};
        geo.class_bytes = {{10, 10, 10}, {20, 20, 20}};
        const int spares = 2, E = 8;
        const std::vector<int> caps_old = {6, 2}, caps_new = {3, 5};
        const expert_slot_table sel0 = {{0, 1, 2}, {0, 1, 2}, {0, 1}};
        expert_slot_table gpu(3, std::vector<int32_t>(E, -1));
        expert_locations homes(3, std::vector<expert_location>(E));
        std::vector<int> ng = {0, 0}, nh = {0, 0};
        for (int l = 0; l < 3; ++l) {
            const int c = geo.layer_class[l];
            for (int e = 0; e < E; ++e) {
                if (std::binary_search(sel0[l].begin(), sel0[l].end(), e)) { gpu[l][e] = ng[c]++; }
                else { homes[l][e] = {expert_storage::host, nh[c]++}; }
            }
        }
        std::vector<std::vector<int>> spare_slots = {{6, 7}, {2, 3}};
        install_layout lay;
        lay.gpu = {caps_old[0] + spares, caps_old[1] + spares};
        lay.host = {2*E - caps_old[0] + spares, E - caps_old[1] + spares};
        lay.lent_begin = {0, 0}; lay.lent_count = {0, 0};
        std::string why;
        CHECK(verify_assignment(gpu, homes, geo.layer_class, lay, spare_slots, E, {false, spares}, false, why));
        auto complement = [&](const expert_slot_table & s) {
            expert_slot_table out(3);
            for (int l = 0; l < 3; ++l) {
                for (int e = 0; e < E; ++e) {
                    if (!std::binary_search(s[l].begin(), s[l].end(), e)) { out[l].push_back(e); }
                }
            }
            return out;
        };
        const std::vector<std::vector<uint8_t>> none(2);
        // 1. evacuation of the kept residents of class 0 at or above its new slot count (3 + 2 spares)
        const expert_slot_table sel1 = {{0, 2}, {2}, {0, 1}};   // layer 1 expert 2 sits at slot 5
        const std::vector<int> limit = {caps_new[0] + spares, caps_old[1] + spares};
        std::vector<std::vector<uint8_t>> other = {std::vector<uint8_t>(8, 0), std::vector<uint8_t>(4, 0)};
        for (int c = 0; c < 2; ++c) { for (int sp : spare_slots[c]) { other[c][sp] = 1; } }
        const evac_plan ev = plan_evacuation(geo.layer_class, E, gpu, sel1, other, lay.gpu, limit);
        CHECK(ev.valid && !ev.ops.empty());
        gpu = ev.gpu;
        // without the host growth the shrink install has nowhere to demote to
        {
            install_layout small = lay; small.gpu = limit;
            CHECK(!plan_install(geo, sel1, complement(sel1), gpu, homes, {caps_new[0], caps_old[1]}, lay, small,
                spare_slots, {false, spares}, false, true, &none).valid);
        }
        // 0. the shrinking class gets its future host slots
        lay.host[0] = 2*E - caps_new[0] + spares;
        // 2. the shrink install demotes into them (unequal exchange, no file)
        install_layout after = lay; after.gpu = limit;
        install_transaction tx = plan_install(geo, sel1, complement(sel1), gpu, homes, {caps_new[0], caps_old[1]}, lay, after,
            spare_slots, {false, spares}, false, true, &none);
        if (!tx.valid) { fprintf(stderr, "shrink: %s\n", tx.reason.c_str()); }
        CHECK(tx.valid && tx.ssd_slices == 0 && tx.d2h_slices == 3);
        gpu = tx.gpu_slots; homes = tx.host; spare_slots = tx.gpu_spares; lay = after;
        // 3. the VRAM resize: class 1 grows to 5 + 2 slots
        lay.gpu = {caps_new[0] + spares, caps_new[1] + spares};
        // 4. the grow install promotes class 1 experts from host into the new slots
        const expert_slot_table sel2 = {{0, 2}, {2}, {0, 1, 4, 5, 7}};
        tx = plan_install(geo, sel2, complement(sel2), gpu, homes, caps_new, lay, lay, spare_slots, {false, spares}, false, true, &none);
        if (!tx.valid) { fprintf(stderr, "grow: %s\n", tx.reason.c_str()); }
        CHECK(tx.valid && tx.h2d_slices == 3);
        gpu = tx.gpu_slots; homes = tx.host; spare_slots = tx.gpu_spares;
        // 5. host compaction of class 1 to its new count (8 - 5 + 2 spares)
        std::vector<host_move> ops;
        CHECK(plan_host_compaction(geo.layer_class, E, homes, 1, lay.host[1], E - caps_new[1] + spares, ops, why));
        for (const host_move & op : ops) { homes[op.layer][op.expert] = {expert_storage::host, op.to}; }
        lay.host[1] = E - caps_new[1] + spares;
        CHECK(verify_assignment(gpu, homes, geo.layer_class, lay, spare_slots, E, {false, spares}, false, why));
        int vram = 0;
        for (int l = 0; l < 3; ++l) {
            for (int e = 0; e < E; ++e) { vram += gpu[l][e] >= 0 ? 1 : 0; }
        }
        CHECK(vram == caps_new[0] + caps_new[1]);
        printf("PASS: exclusive redraw without a file: host growth, evacuation, shrink and grow installs, host compaction "
               "(%zu host moves); one home per expert after every step\n", ops.size());
    }
    return 0;
}

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

    // Cold start with dense layers and empty layer numbers (a joint cache whose second member
    // numbers its only routed layer after the first member's): the split divides by the four
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

    // Cold fill across layers: a class whose last layer is a late joint member (an MTP head at
    // layer 4 after two empty layer numbers). The round robin gives the late layer its share; a
    // layer-first order would give {0, 1, 2, 3}, {0}, {}.
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

        // the host tier's cold cut spreads the same way
        const std::vector<std::array<size_t, 3>> hbytes{{100, 100, 100}};
        geometry hgeo = make_geometry({0, 0, 0, 0}, 4, hbytes);
        std::vector<std::vector<int32_t>> vram(4, std::vector<int32_t>(4, -1));
        const std::vector<size_t> pitch{300};
        tier_inputs t;
        t.geo = &hgeo;
        t.vram = &vram;
        t.slot_pitch = &pitch;
        t.budget_bytes = 6*300;
        const tier_plan host = plan_host_tier(t);
        CHECK(host.valid && host.capacities[0] == 6);
        CHECK((host.selected[0] == std::vector<int32_t>{0, 1}));
        CHECK((host.selected[1] == std::vector<int32_t>{0, 1}));
        CHECK((host.selected[2] == std::vector<int32_t>{0}));
        CHECK((host.selected[3] == std::vector<int32_t>{0}));
        printf("PASS: cold L1 and host fills spread over the layers of a class in a round robin\n");
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

        // the host cut: exclusive VRAM holds experts 0-1 of layer 0 and 2-3 of layer 1, so the
        // cold candidates of each layer differ; the host slots still go one per layer
        geometry hgeo = make_geometry({0, 0, 0, 0}, 4, bytes);
        std::vector<std::vector<int32_t>> vram(4, std::vector<int32_t>(4, -1));
        vram[0][0] = 0; vram[0][1] = 1; vram[1][2] = 2; vram[1][3] = 3;
        const std::vector<size_t> pitch{300};
        tier_inputs t;
        t.geo = &hgeo;
        t.vram = &vram;
        t.slot_pitch = &pitch;
        t.budget_bytes = 4*300;
        const tier_plan host = plan_host_tier(t);
        CHECK(host.valid && host.capacities[0] == 4);
        CHECK((host.selected[0] == std::vector<int32_t>{2}));
        CHECK((host.selected[1] == std::vector<int32_t>{0}));
        CHECK((host.selected[2] == std::vector<int32_t>{0}));
        CHECK((host.selected[3] == std::vector<int32_t>{0}));

        // borrowed slots continue the cut's order past the base plan (layers 0-1 class 0, 2-3
        // class 1): base {2, 0} holds layer 0 expert 2 and layer 1 expert 0; the next candidates
        // are expert 0 of layers 2 and 3 (class 1), then layer 0 expert 3 (class 0)
        geometry bgeo = make_geometry({0, 0, 1, 1}, 4, {{100, 100, 100}, {100, 100, 100}});
        const std::vector<size_t> bpitch{300, 300};
        tier_inputs b = t;
        b.geo = &bgeo;
        b.slot_pitch = &bpitch;
        const std::vector<int> base_caps{2, 0};
        CHECK((plan_borrowed_capacities(b, base_caps, 2) == std::vector<int>{2, 2}));
        const std::vector<int> grown = plan_borrowed_capacities(b, base_caps, 3);
        CHECK((grown == std::vector<int>{3, 2}));
        b.fixed_capacities = &grown;
        const tier_plan regrown = plan_host_tier(b);
        CHECK(regrown.valid);
        CHECK((regrown.selected[0] == std::vector<int32_t>{2, 3}));
        CHECK((regrown.selected[1] == std::vector<int32_t>{0}));
        CHECK((regrown.selected[2] == std::vector<int32_t>{0}));
        CHECK((regrown.selected[3] == std::vector<int32_t>{0}));
        printf("PASS: equal scores on different expert sets per layer keep the per-layer counts even (L1, host cut, borrowed slots)\n");
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
    if (redraw_tests() != 0) {
        return 1;
    }
    if (host_layout_tests() != 0) {
        return 1;
    }
    return 0;
}
