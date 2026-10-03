#pragma once

// Redraw of the VRAM (L1) size-class partition at a plan install. Pure CPU code: no CUDA/HIP
// headers, no I/O except the one environment reader below.
//
// The load freezes one static VRAM capacity per size class (expert-plan.h, plan_placement). A cache
// shared by several models, or a cold load, can leave that split far from what the profile asks for
// later. With a redraw condition set (RANMA_EXPERT_L1_REDRAW_BENEFIT and/or RANMA_EXPERT_L1_REDRAW_MOVE)
// the class arenas are HIP virtual address ranges backed by physical handles (expert-l1.cu), and an
// install may move capacity between classes:
//   1. the proposal: the load's global greedy on the current scores within the current static VRAM
//      bytes, held inside per-class floors and ceilings (redraw_capacities), with the bytes moved per
//      install capped (redraw_limit);
//   2. the gate: every condition that is set must hold. RANMA_EXPERT_L1_REDRAW_BENEFIT = k: the
//      estimated benefit over the horizon must exceed k times the estimated cost
//      (redraw_estimate_of); RANMA_EXPERT_L1_REDRAW_MOVE = p: the proposal must move more than p
//      percent of the L1 budget. A proposal must pass on redraw_confirm consecutive installs, except
//      in the first installs after a cold load or a member join; a redraw holds the next
//      redraw_holdoff installs off (redraw_decide);
//   3. the move: the shrinking classes install their new plan inside the smaller slot range first
//      (the kept residents above it are moved down beforehand, plan_evacuation), then the vacated
//      handles are unmapped and mapped at the top of the growing classes, then the full plan is
//      installed in the new capacities.

#include "expert-geometry.h"
#include "expert-location.h"
#include "expert-plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// Fixed parts of the gate and the move.
static constexpr int    redraw_confirm    = 2;     // consecutive installs a proposal must pass
static constexpr int    redraw_holdoff    = 3;     // installs held off after a redraw
static constexpr size_t redraw_cap_mib    = 4608;  // bytes that change class per install
static constexpr int    redraw_horizon    = 350;   // tokens a redraw pays over
static constexpr size_t redraw_handle_mib = 64;    // physical handle size

// Cost model of the gate, measured on the R9700 / PCIe 5.0 x16 / Gen5 x4 NVMe target.
static constexpr double redraw_link_ms_per_gb   = 20.6;  // host -> VRAM over PCIe
static constexpr double redraw_ssd_ms_per_gb    = 81.8;  // file -> host -> VRAM
static constexpr double redraw_remap_ms         = 0.45;  // unmap + map + access of one handle
static constexpr double redraw_create_ms_64mib  = 2.4;   // hipMemCreate of 64 MiB
static constexpr double redraw_d2d_ms_per_gib   = 3.6;   // VRAM to VRAM copy

struct redraw_params {
    bool   enabled   = false;  // at least one condition is set
    double benefit   = 0.0;    // k: benefit > k x cost; 0 = condition not set
    double move_pct  = 0.0;    // p: moved share of the L1 budget > p %; 0 = condition not set
    int    confirm   = redraw_confirm;
    int    holdoff   = redraw_holdoff;
    size_t cap_bytes = redraw_cap_mib << 20;
    int    horizon   = redraw_horizon;
    size_t handle_bytes = redraw_handle_mib << 20;
};

// Reads RANMA_EXPERT_L1_REDRAW_BENEFIT and RANMA_EXPERT_L1_REDRAW_MOVE through `get` (getenv in the
// controller). The redraw is on when at least one of them holds a valid positive number; an invalid
// value leaves its condition unset and is reported in `warnings`.
inline redraw_params redraw_params_from(const std::function<const char *(const char *)> & get, std::string & warnings) {
    redraw_params p;
    warnings.clear();
    auto number = [&](const char * name, double & out, double hi) {
        const char * v = get(name);
        if (v == nullptr || v[0] == '\0') { return; }
        char * end = nullptr;
        const double x = strtod(v, &end);
        if (end == v || *end != '\0' || !std::isfinite(x) || x <= 0.0 || x > hi) {
            warnings += std::string(warnings.empty() ? "" : "; ") + name + "=" + v + " ignored";
            return;
        }
        out = x;
    };
    number("RANMA_EXPERT_L1_REDRAW_BENEFIT", p.benefit, 1e9);
    number("RANMA_EXPERT_L1_REDRAW_MOVE", p.move_pct, 100.0);
    p.enabled = p.benefit > 0.0 || p.move_pct > 0.0;
    return p;
}

// ---- the proposal -------------------------------------------------------------------------------

inline size_t redraw_static_bytes(const geometry & geo, const std::vector<int> & capacities) {
    size_t bytes = 0;
    for (size_t c = 0; c < capacities.size() && c < geo.class_bytes.size(); ++c) {
        bytes += size_t(std::max(capacities[c], 0))*geo.class_vram_bytes(int(c));
    }
    return bytes;
}

// Bytes that change class: what the shrinking classes give up.
inline size_t redraw_moved_bytes(const geometry & geo, const std::vector<int> & current, const std::vector<int> & next) {
    size_t bytes = 0;
    for (size_t c = 0; c < current.size() && c < next.size(); ++c) {
        if (next[c] < current[c]) { bytes += size_t(current[c] - next[c])*geo.class_vram_bytes(int(c)); }
    }
    return bytes;
}

// The static capacities the load's greedy (plan_placement, warm start) gives the scores within
// `budget` bytes, held to [floor, ceiling] per class. A class's first `floor` candidates are taken
// out of the budget up front. Budget the proposal does not use (an inclusive plan takes no
// zero-count expert) goes back to the classes that shrank, so a redraw only moves what another
// class takes. Without scores the current capacities come back.
inline std::vector<int> redraw_capacities(const geometry & geo, const uint64_t * counts, size_t budget, bool exclusive,
        const std::vector<int> & current, const std::vector<int> & floor, const std::vector<int> & ceiling) {
    const size_t classes = geo.class_bytes.size();
    if (counts == nullptr || current.size() != classes || floor.size() != classes || ceiling.size() != classes) {
        return current;
    }
    uint64_t total = 0;
    for (size_t i = 0; i < geo.n_counts(); ++i) { total += counts[i]; }
    if (total == 0) {
        return current;
    }
    std::vector<int> caps(classes, 0), lo(classes, 0), hi(classes, 0);
    size_t reserved = 0;
    for (size_t c = 0; c < classes; ++c) {
        hi[c] = std::max(0, std::min(ceiling[c], geo.class_layers[c]*geo.n_experts));
        lo[c] = std::max(0, std::min(floor[c], hi[c]));
        caps[c] = lo[c];
        reserved += size_t(lo[c])*geo.class_vram_bytes(int(c));
    }
    if (reserved > budget) {
        return current;
    }
    size_t remaining = budget - reserved;
    const std::vector<detail::plan_candidate> candidates = detail::sorted_candidates(geo, counts);
    std::vector<int> rank(classes, 0);
    for (const detail::plan_candidate & cand : candidates) {
        const size_t bytes = geo.class_vram_bytes(cand.cls);
        if (++rank[cand.cls] <= lo[cand.cls]) {
            continue;
        }
        if ((cand.count != 0 || exclusive) && caps[cand.cls] < hi[cand.cls] && bytes <= remaining) {
            ++caps[cand.cls];
            remaining -= bytes;
        }
    }
    for (size_t c = 0; c < classes; ++c) {
        const size_t bytes = geo.class_vram_bytes(int(c));
        if (caps[c] < current[c] && bytes != 0) {
            const int back = (int) std::min<size_t>(size_t(std::min(current[c], hi[c]) - caps[c]), remaining/bytes);
            caps[c] += std::max(back, 0);
            remaining -= size_t(std::max(back, 0))*bytes;
        }
    }
    return caps;
}

// At most `cap_bytes` change class in one install: every class moves the same fraction of its way,
// and the growing classes give slots back until the static bytes fit `budget` again.
inline std::vector<int> redraw_limit(const geometry & geo, const std::vector<int> & current, const std::vector<int> & proposed,
        size_t cap_bytes, size_t budget) {
    const size_t moved = redraw_moved_bytes(geo, current, proposed);
    if (cap_bytes == 0 || moved <= cap_bytes || current.size() != proposed.size()) {
        return proposed;
    }
    const double f = double(cap_bytes)/double(moved);
    std::vector<int> out = current;
    for (size_t c = 0; c < current.size(); ++c) {
        out[c] = current[c] + (int) std::trunc(double(proposed[c] - current[c])*f);
    }
    while (redraw_static_bytes(geo, out) > budget) {
        int best = -1;
        for (size_t c = 0; c < out.size(); ++c) {
            if (out[c] > current[c] && (best < 0 || out[c] - current[c] > out[best] - current[best])) { best = int(c); }
        }
        if (best < 0) { return current; }
        --out[best];
    }
    return out;
}

// ---- the gate -----------------------------------------------------------------------------------

struct redraw_estimate {
    size_t budget_bytes = 0;     // the L1 budget the move percentage refers to
    size_t moved_bytes  = 0;     // bytes that change class
    double move_pct     = 0.0;
    size_t enter_slices = 0, enter_bytes = 0, enter_file_bytes = 0;  // experts the redraw brings into VRAM
    size_t leave_slices = 0, leave_bytes = 0;                        // experts it takes out
    size_t handles      = 0;     // physical handles to unmap and map
    double tokens       = 0.0;   // tokens the scores stand for
    double benefit_ms_token = 0.0;
    double benefit_ms   = 0.0;   // over the horizon
    double cost_promote_ms = 0.0, cost_demote_ms = 0.0, cost_remap_ms = 0.0, cost_evacuate_ms = 0.0;
    double cost_ms      = 0.0;
    double ratio        = 0.0;   // benefit / cost; infinity at zero cost
};

// Benefit and cost of installing `next` instead of `now` (both [layer] selections from the same
// scores; `now` with the current capacities, `next` with the proposed ones).
//   - scores become selections per token: tokens = sum of the scores / (experts used x routed layers);
//   - benefit per token: every expert that enters VRAM saves its selections per token x its bytes x
//     the price of the tier it is read from now (the file when `in_file` says so, else the host
//     link); every expert that leaves costs the same at the host link price (it keeps a host home in
//     the plan's cut or is among the weakest);
//   - cost: the bytes that enter VRAM at their tier's price, with an exclusive cache (`demote`) the
//     bytes that leave at the link price, the handle remaps, and at most one swap (three VRAM
//     copies) of every byte that changes class.
inline redraw_estimate redraw_estimate_of(const geometry & geo, const uint64_t * counts, int experts_used,
        const expert_slot_table & now, const expert_slot_table & next, const std::function<bool(int, int)> & in_file,
        bool demote, const std::vector<int> & current, const std::vector<int> & proposed, size_t budget_bytes,
        const redraw_params & p) {
    redraw_estimate out;
    out.budget_bytes = budget_bytes;
    out.moved_bytes  = redraw_moved_bytes(geo, current, proposed);
    out.move_pct     = budget_bytes == 0 ? 0.0 : 100.0*double(out.moved_bytes)/double(budget_bytes);
    uint64_t total = 0;
    for (size_t i = 0; counts != nullptr && i < geo.n_counts(); ++i) { total += counts[i]; }
    const int routed = geo.n_routed_layers();
    out.tokens = routed > 0 ? double(total)/(double(std::max(experts_used, 1))*double(routed)) : 0.0;
    const double gb = 1e9;
    double gain = 0.0;
    for (int l = 0; l < geo.n_layers && size_t(l) < now.size() && size_t(l) < next.size(); ++l) {
        const int c = geo.layer_class[l];
        if (c < 0) { continue; }
        const size_t bytes = geo.class_vram_bytes(c);
        auto has = [](const std::vector<int32_t> & v, int e) { return std::binary_search(v.begin(), v.end(), e); };
        for (int e = 0; e < geo.n_experts; ++e) {
            const bool a = has(now[l], e), b = has(next[l], e);
            if (a == b) { continue; }
            const double rate = out.tokens > 0.0 && counts != nullptr ? double(counts[size_t(l)*geo.n_experts + e])/out.tokens : 0.0;
            if (b) {
                const bool file = in_file && in_file(l, e);
                const double price = file ? redraw_ssd_ms_per_gb : redraw_link_ms_per_gb;
                ++out.enter_slices; out.enter_bytes += bytes; out.enter_file_bytes += file ? bytes : 0;
                gain += rate*double(bytes)/gb*price;
                out.cost_promote_ms += double(bytes)/gb*price;
            } else {
                ++out.leave_slices; out.leave_bytes += bytes;
                gain -= rate*double(bytes)/gb*redraw_link_ms_per_gb;
                out.cost_demote_ms += demote ? double(bytes)/gb*redraw_link_ms_per_gb : 0.0;
            }
        }
    }
    const size_t handle = std::max<size_t>(p.handle_bytes, 1);
    size_t partial = 0;
    for (size_t c = 0; c < current.size() && c < proposed.size() && c < geo.class_bytes.size(); ++c) {
        if (proposed[c] == current[c]) { continue; }
        const size_t slots = size_t(std::abs(proposed[c] - current[c]));
        for (int k = 0; k < geometry::n_kinds; ++k) {
            out.handles += (slots*geo.class_bytes[c][k] + handle - 1)/handle;
            partial += 2;   // the old and the new partial handle at the top of the range
        }
    }
    out.cost_remap_ms = double(out.handles)*redraw_remap_ms*2.0 +
        double(partial)*redraw_create_ms_64mib*double(handle)/double(size_t(64) << 20);
    out.cost_evacuate_ms = 3.0*double(out.moved_bytes)/double(size_t(1) << 30)*redraw_d2d_ms_per_gib;
    out.cost_ms = out.cost_promote_ms + out.cost_demote_ms + out.cost_remap_ms + out.cost_evacuate_ms;
    out.benefit_ms_token = gain;
    out.benefit_ms = gain*double(p.horizon);
    out.ratio = out.cost_ms > 0.0 ? out.benefit_ms/out.cost_ms : std::numeric_limits<double>::infinity();
    return out;
}

struct redraw_gate {
    int streak  = 0;   // consecutive installs whose proposal passed both conditions
    int holdoff = 0;   // installs still held off after a redraw
    int fresh   = 0;   // installs after a cold load or a member join that need no confirmation
};

struct redraw_verdict {
    bool fire = false;
    bool move_ok = false, benefit_ok = false;
    std::string why;
};

// A cold load or a member join: the next `confirm` installs may fire on their first proposal.
inline void redraw_mark_fresh(redraw_gate & g, const redraw_params & p) {
    g.fresh  = std::max(p.confirm, 1);
    g.streak = 0;
}

inline redraw_verdict redraw_decide(const redraw_params & p, const redraw_estimate & est, redraw_gate & g) {
    redraw_verdict v;
    v.move_ok    = p.move_pct <= 0.0 || est.move_pct > p.move_pct;
    v.benefit_ok = p.benefit <= 0.0 || est.benefit_ms > p.benefit*est.cost_ms;
    const bool fresh = g.fresh > 0;
    if (g.fresh > 0) { --g.fresh; }
    if (est.moved_bytes == 0) {
        g.streak = 0;
        v.why = "the split does not change";
    } else if (!v.move_ok || !v.benefit_ok) {
        g.streak = 0;
        v.why = !v.move_ok && !v.benefit_ok ? "move and benefit below threshold" : !v.move_ok ? "move below threshold" :
            "benefit below threshold";
    } else {
        ++g.streak;
        if (g.holdoff > 0) {
            v.why = "hold-off (" + std::to_string(g.holdoff) + " more installs)";
        } else if (!fresh && g.streak < p.confirm) {
            v.why = "needs " + std::to_string(p.confirm) + " consecutive proposals";
        } else {
            v.fire = true;
            v.why  = fresh ? "fired (first installs after a cold load or member join)" : "fired";
        }
    }
    if (v.fire) {
        g.streak  = 0;
        g.holdoff = p.holdoff;
    } else if (g.holdoff > 0) {
        --g.holdoff;
    }
    return v;
}

// ---- the move -----------------------------------------------------------------------------------

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
