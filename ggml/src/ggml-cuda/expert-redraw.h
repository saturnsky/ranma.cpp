#pragma once

// The evacuation before a shrink install of the VRAM (L1) arena: the prefill lend retires the top slots of
// each size class (expert-controller.cu, lend_evict_locked), so the kept residents above the new slot count
// move down first. Pure CPU code: no CUDA/HIP headers, no I/O.

#include "expert-location.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml_cuda_expert {

// Every static resident a shrinking class keeps must sit below the class's new slot count before
// the install that shrinks it. A kept resident above it moves to a free slot below (no static home,
// no pool position, no spare), else it swaps with a resident below that the new plan drops (that one
// is then demoted from above the limit by the install, which still reads the old range).
struct evac_op {
    int cls, layer, expert, from, to;
    int victim_layer = -1, victim_expert = -1;   // -1: `to` was free
};

struct evac_plan {
    bool valid = false;
    std::string reason;
    std::vector<evac_op> ops;
    expert_slot_table gpu;   // the static table after the ops
};

// `other[class][slot]`: slots held by pool positions and spares. `slots[class]`: the current slot
// count; `limit[class]`: the count after the shrink (>= slots means the class does not shrink).
inline evac_plan plan_evacuation(const std::vector<int> & layer_class, int n_experts, const expert_slot_table & gpu,
        const expert_slot_table & keep, const std::vector<std::vector<uint8_t>> & other, const std::vector<int> & slots,
        const std::vector<int> & limit) {
    evac_plan out;
    auto fail = [&](const std::string & text) { out.valid = false; out.reason = text; return out; };
    const size_t classes = slots.size(), layers = layer_class.size();
    if (limit.size() != classes || gpu.size() != layers || keep.size() != layers) { return fail("evacuation dimensions"); }
    out.gpu = gpu;
    for (size_t c = 0; c < classes; ++c) {
        if (limit[c] >= slots[c]) { continue; }
        if (limit[c] < 0) { return fail("negative slot limit"); }
        std::vector<int> owner_l(size_t(slots[c]), -1), owner_e(size_t(slots[c]), -1);
        std::vector<uint8_t> held(size_t(slots[c]), 0);
        for (size_t s = 0; c < other.size() && s < other[c].size() && s < held.size(); ++s) { held[s] = other[c][s]; }
        std::vector<std::pair<int, int>> above;   // kept residents at or above the limit, by slot
        std::vector<int> above_slot;
        for (size_t l = 0; l < layers; ++l) {
            if (layer_class[l] != int(c)) { continue; }
            if (gpu[l].size() != size_t(n_experts)) { return fail("evacuation table dimensions"); }
            for (int e = 0; e < n_experts; ++e) {
                const int g = gpu[l][e];
                if (g < 0) { continue; }
                if (g >= slots[c] || owner_l[size_t(g)] >= 0 || held[size_t(g)]) { return fail("static slot out of range or shared"); }
                owner_l[size_t(g)] = int(l); owner_e[size_t(g)] = e;
            }
        }
        auto kept = [&](int l, int e) { return std::binary_search(keep[size_t(l)].begin(), keep[size_t(l)].end(), e); };
        for (int s = limit[c]; s < slots[c]; ++s) {
            if (owner_l[size_t(s)] >= 0 && kept(owner_l[size_t(s)], owner_e[size_t(s)])) {
                above.emplace_back(owner_l[size_t(s)], owner_e[size_t(s)]);
                above_slot.push_back(s);
            }
        }
        std::vector<int> free_slots, victims;
        for (int s = 0; s < limit[c]; ++s) {
            if (held[size_t(s)]) { continue; }
            if (owner_l[size_t(s)] < 0) { free_slots.push_back(s); }
            else if (!kept(owner_l[size_t(s)], owner_e[size_t(s)])) { victims.push_back(s); }
        }
        size_t fi = 0, vi = 0;
        for (size_t i = 0; i < above.size(); ++i) {
            const int l = above[i].first, e = above[i].second, from = above_slot[i];
            if (fi < free_slots.size()) {
                const int to = free_slots[fi++];
                out.ops.push_back({int(c), l, e, from, to});
                out.gpu[size_t(l)][size_t(e)] = to;
            } else if (vi < victims.size()) {
                const int to = victims[vi++];
                const int vl = owner_l[size_t(to)], ve = owner_e[size_t(to)];
                out.ops.push_back({int(c), l, e, from, to, vl, ve});
                out.gpu[size_t(l)][size_t(e)] = to;
                out.gpu[size_t(vl)][size_t(ve)] = from;
            } else {
                return fail("class " + std::to_string(c) + " keeps more residents than its new slot range holds");
            }
        }
    }
    out.valid = true;
    return out;
}

} // namespace ggml_cuda_expert
