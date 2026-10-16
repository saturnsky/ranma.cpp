#pragma once

// ranma: the LRU part of L1 (--expert-l1-lru-pct). Pure C++, shared by the controller (expert-controller.cu) and its CPU
// test (tests/test-expert-lend.cpp).
//
// N % of each size class's L1 slots are not planned: they sit above the class's static slots and spares, and every routed
// layer owns some of them as positions of its decode pool (expert-ub-pool.cuh), next to the positions the pool borrows from
// the compute buffer: one least recently used set per layer, with the pool's admission filter (ghost:K). The slots of a
// layer hold copies (inclusive); the original stays in the L2 or the file, so a copy is dropped without a write back.
//   - Shares (l1_lru_shares): at each plan install, the slots of a class are split over its layers in proportion to the
//     installed plan's profile score of the experts outside the static plan (the selections the static part misses),
//     rounded down, the leftover slots one each to the largest remainders (ties: the lower layer); equal shares without a
//     score. At most `cap` per layer, the excess goes to the others. A layer without any profile (a joined draft model)
//     counts as an average layer of its class (l1_lru_fill_mass).
//   - Classes (l1_lru_proportional): a class without static slots (no profile for its experts) takes the share of the
//     LRU part's bytes that its experts' bytes make of all classes', from the classes with the largest LRU parts. The
//     layers of a joined model hold only these fixed positions; the decode pool lends them no region.
//   - A prefill lend takes the top slots of a class, the LRU part first (it sits at the top). The copies that stay are
//     the most recently used ones whatever their slot (l1_lru_shrink): the ones above the lent top move down into the
//     slots below it whose copies are dropped (or that are empty). When the lend ends, the returned slots go, empty, to the
//     layers that have fewer than their share (l1_lru_revive).
//   - The copies stay through a plan install. A decode plan's install gives new shares (l1_lru_reshare): each layer keeps
//     its most recently used copies up to its new share in their slots, the others leave, and the free slots go, empty, to
//     the layers below their share. Nothing moves: a copy holds an expert of its own layer.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace ggml_cuda_expert {

// [class] the slots of the LRU part: floor(N % of the class's static capacity).
inline std::vector<int> l1_lru_class_slots(const std::vector<int> & capacities, int pct) {
    std::vector<int> out(capacities.size(), 0);
    if (pct <= 0) {
        return out;
    }
    pct = std::min(pct, 100);
    for (size_t c = 0; c < capacities.size(); ++c) {
        out[c] = (int) ((int64_t) std::max(capacities[c], 0)*pct/100);
    }
    return out;
}

// A class without static slots (its experts have no profile, e.g. a draft model that joined without one) would get no slots
// of the LRU part. Such a class takes a share of the LRU part's bytes in proportion to its capacity, the bytes of all
// its experts against those of all classes (`class_slices` experts of `slot_bytes` each), from the classes that have
// slots, the largest LRU part first. `x` holds the slots from l1_lru_class_slots; on return the needy classes have slots
// and the donors fewer, and the total bytes of the LRU part are unchanged (or less, when the donors cannot pay).
inline void l1_lru_proportional(std::vector<int> & x, const std::vector<int> & capacities, const std::vector<int64_t> & class_slices,
        const std::vector<size_t> & slot_bytes) {
    const size_t nc = x.size();
    if (capacities.size() != nc || class_slices.size() != nc || slot_bytes.size() != nc) {
        return;
    }
    double lru = 0.0, all = 0.0;
    std::vector<uint8_t> needy(nc, 0);
    bool any = false;
    for (size_t c = 0; c < nc; ++c) {
        lru += (double) x[c]*(double) slot_bytes[c];
        all += (double) class_slices[c]*(double) slot_bytes[c];
        needy[c] = capacities[c] <= 0 && x[c] == 0 && class_slices[c] > 0 && slot_bytes[c] > 0 ? 1 : 0;
        any = any || needy[c];
    }
    if (!any || lru <= 0.0 || all <= 0.0) {
        return;
    }
    double owed = 0.0;
    std::vector<int> want(nc, 0);
    for (size_t c = 0; c < nc; ++c) {
        if (needy[c]) {
            want[c] = (int) std::min<double>((double) class_slices[c], lru*(double) class_slices[c]/all);
            owed += (double) want[c]*(double) slot_bytes[c];
        }
    }
    std::vector<size_t> donors;
    for (size_t c = 0; c < nc; ++c) { if (!needy[c] && x[c] > 0) { donors.push_back(c); } }
    std::stable_sort(donors.begin(), donors.end(), [&](size_t a, size_t b) {
        return (double) x[a]*(double) slot_bytes[a] > (double) x[b]*(double) slot_bytes[b];
    });
    for (size_t c : donors) {
        if (owed <= 0.0) { break; }
        const int d = (int) std::min<double>((double) x[c], std::ceil(owed/(double) slot_bytes[c]));
        x[c] -= d;
        owed -= (double) d*(double) slot_bytes[c];
    }
    // the donors could not pay in full: the needy classes give back what is still owed
    for (size_t c = 0; c < nc && owed > 0.0; ++c) {
        if (needy[c] && want[c] > 0) {
            const int d = (int) std::min<double>((double) want[c], std::ceil(owed/(double) slot_bytes[c]));
            want[c] -= d;
            owed -= (double) d*(double) slot_bytes[c];
        }
    }
    for (size_t c = 0; c < nc; ++c) { x[c] += needy[c] ? want[c] : 0; }
}

// A layer without a profile has no mass, which would leave it no slots next to profiled layers of its class. Its mass
// becomes the mean of the profiled layers' (a capacity-proportional share: an average layer). `profiled[i]` 0 marks it;
// layers with mass < 0 (no positions) stay out. Without any profiled layer nothing changes (equal shares follow).
inline void l1_lru_fill_mass(std::vector<double> & mass, const std::vector<uint8_t> & profiled) {
    if (profiled.size() != mass.size()) {
        return;
    }
    double s = 0.0;
    size_t k = 0;
    for (size_t i = 0; i < mass.size(); ++i) {
        if (mass[i] >= 0.0 && profiled[i]) { s += mass[i]; ++k; }
    }
    if (k == 0 || !(s > 0.0)) {
        return;
    }
    for (size_t i = 0; i < mass.size(); ++i) {
        if (mass[i] >= 0.0 && !profiled[i]) { mass[i] = s/(double) k; }
    }
}

// `total` slots over the layers of `mass` (one entry per layer of the class; a layer with mass < 0 gets none), in
// proportion to the mass, at most `cap` each. Rounded down, the leftover slots one each to the largest remainders (ties:
// the lower index). Layers at the cap leave the rest to the others; open layers without mass share equally. Slots beyond
// n * cap stay unused.
inline std::vector<int> l1_lru_shares(const std::vector<double> & mass, int total, int cap) {
    const size_t n = mass.size();
    std::vector<int> out(n, 0);
    std::vector<uint8_t> open(n, 0);
    size_t n_open = 0;
    for (size_t i = 0; i < n; ++i) {
        open[i] = mass[i] >= 0.0 ? 1 : 0;
        n_open += open[i];
    }
    if (total <= 0 || cap <= 0 || n_open == 0) {
        return out;
    }
    int left = (int) std::min<int64_t>(total, (int64_t) n_open*cap);
    // water filling: a layer whose proportional share reaches the cap is fixed at the cap, the rest is shared again
    for (;;) {
        // equal shares when the open layers have no mass (none at all, or the others are at the cap)
        double s = 0.0;
        size_t k = 0;
        for (size_t i = 0; i < n; ++i) {
            if (open[i]) { s += mass[i]; ++k; }
        }
        const bool uniform = !(s > 0.0);
        if (uniform) {
            s = (double) k;
        }
        if (k == 0 || left <= 0) {
            break;
        }
        bool fixed = false;
        for (size_t i = 0; i < n; ++i) {
            if (!open[i]) { continue; }
            const double w = uniform ? 1.0 : mass[i];
            const double want = s > 0.0 ? (double) left*w/s : 0.0;
            if (want >= (double) cap) {
                out[i] = cap;
                open[i] = 0;
                left -= cap;
                fixed = true;
            }
        }
        if (fixed) {
            continue;
        }
        // proportional, rounded down, then the largest remainders
        std::vector<double> rem(n, -1.0);
        int given = 0;
        for (size_t i = 0; i < n; ++i) {
            if (!open[i]) { continue; }
            const double w = uniform ? 1.0 : mass[i];
            const double want = s > 0.0 ? (double) left*w/s : 0.0;
            const int f = std::min(cap, (int) want);
            out[i] = f;
            rem[i] = want - (double) f;
            given += f;
        }
        int extra = left - given;
        std::vector<size_t> order;
        for (size_t i = 0; i < n; ++i) { if (open[i] && out[i] < cap) { order.push_back(i); } }
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rem[a] > rem[b]; });
        for (size_t j = 0; extra > 0 && j < order.size(); ++j) {
            ++out[order[j]];
            --extra;
        }
        break;
    }
    return out;
}

// The slots of the LRU part of one class laid out over its layers: share[l] slots each, from `first` up, one round at
// a time (round j: one slot to every layer with share > j, lower layers first). [layer] its slots.
inline std::vector<std::vector<int>> l1_lru_layout(const std::vector<int> & share, int first) {
    std::vector<std::vector<int>> out(share.size());
    int most = 0;
    for (int s : share) { most = std::max(most, s); }
    int slot = first;
    for (int j = 0; j < most; ++j) {
        for (size_t l = 0; l < share.size(); ++l) {
            if (share[l] > j) { out[l].push_back(slot++); }
        }
    }
    return out;
}

// One position of the LRU part of a class, for l1_lru_shrink.
struct l1_lru_pos {
    int      layer = -1;
    int      slot  = -1;
    int      owner = -1;   // the expert, -1 empty
    uint32_t age   = 0;    // steps of its layer since it was last used (owned)
};

struct l1_lru_shrink_plan {
    std::vector<int>  keep;                  // [i] of the input: the copy stays (at `slot_of[i]`)
    std::vector<int>  drop;                  // [i] of the input: the copy leaves (its table entries go back)
    std::vector<std::array<int, 2>> moves;   // (from, to) slots: a kept copy above the top moves below it
    std::vector<int>  slot_of;               // [i]: the slot of a kept copy after the moves, else -1
    std::vector<std::vector<int>> layer_slots;   // [layer] its slots below the top after the lend (its kept copies first)
};

// A lend takes the slots at and above `top` of the class (`pos`: every position of the LRU part of the class, n_layers
// the layer count of the cache). The copies that stay are the most recently used ones (the lowest age, then the lower
// layer, then the lower slot), as many as there are slots below the top. A kept copy above the top moves into a slot
// below it whose copy leaves or that is empty, the lowest first. Every slot below the top stays a position: of the layer
// whose copy is in it, or of its old layer when it is empty.
inline l1_lru_shrink_plan l1_lru_shrink(const std::vector<l1_lru_pos> & pos, int top, int n_layers) {
    l1_lru_shrink_plan p;
    const size_t n = pos.size();
    p.slot_of.assign(n, -1);
    p.layer_slots.assign((size_t) std::max(n_layers, 0), {});
    size_t below = 0;
    std::vector<size_t> owned;
    for (size_t i = 0; i < n; ++i) {
        below += pos[i].slot < top ? 1 : 0;
        if (pos[i].owner >= 0) { owned.push_back(i); }
    }
    std::stable_sort(owned.begin(), owned.end(), [&](size_t a, size_t b) {
        if (pos[a].age != pos[b].age) { return pos[a].age < pos[b].age; }
        if (pos[a].layer != pos[b].layer) { return pos[a].layer < pos[b].layer; }
        return pos[a].slot < pos[b].slot;
    });
    std::vector<uint8_t> kept(n, 0);
    for (size_t j = 0; j < owned.size(); ++j) {
        if (j < below) {
            kept[owned[j]] = 1;
            p.keep.push_back((int) owned[j]);
        } else {
            p.drop.push_back((int) owned[j]);
        }
    }
    std::sort(p.keep.begin(), p.keep.end());
    std::sort(p.drop.begin(), p.drop.end());
    // the free slots below the top: empty, or their copy leaves
    std::vector<size_t> free_below;
    for (size_t i = 0; i < n; ++i) {
        if (pos[i].slot < top && !kept[i]) { free_below.push_back(i); }
    }
    std::sort(free_below.begin(), free_below.end(), [&](size_t a, size_t b) { return pos[a].slot < pos[b].slot; });
    std::vector<int> layer_of_slot_below(n, -1);   // [i] of a free slot: the layer that takes it (-1 = its own)
    size_t next = 0;
    for (int i : p.keep) {
        if (pos[(size_t) i].slot < top) {
            p.slot_of[(size_t) i] = pos[(size_t) i].slot;
            continue;
        }
        const size_t f = free_below[next++];   // below >= kept, so a free slot exists
        p.moves.push_back({ pos[(size_t) i].slot, pos[f].slot });
        p.slot_of[(size_t) i] = pos[f].slot;
        layer_of_slot_below[f] = pos[(size_t) i].layer;
    }
    for (int i : p.keep) {
        const int l = pos[(size_t) i].layer;
        if (l >= 0 && l < n_layers) { p.layer_slots[(size_t) l].push_back(p.slot_of[(size_t) i]); }
    }
    for (size_t j = next; j < free_below.size(); ++j) {
        const size_t f = free_below[j];
        const int l = pos[f].layer;
        if (l >= 0 && l < n_layers) { p.layer_slots[(size_t) l].push_back(pos[f].slot); }
    }
    return p;
}

// A lend ended: the returned slots (empty) go to the layers that have fewer slots than their share, lower layers first,
// the lowest slots first. `slots` [layer] the slots each layer has now; `share` [layer] its share. Layers at their share
// keep theirs; a returned slot no layer lacks stays with no layer.
inline void l1_lru_revive(std::vector<std::vector<int>> & slots, const std::vector<int> & share, std::vector<int> returned) {
    std::sort(returned.begin(), returned.end());
    size_t next = 0;
    for (size_t l = 0; l < slots.size() && l < share.size(); ++l) {
        while ((int) slots[l].size() < share[l] && next < returned.size()) {
            slots[l].push_back(returned[next++]);
        }
    }
}

struct l1_lru_reshare_plan {
    std::vector<int> keep;                      // [i] of the input: the copy stays in its slot
    std::vector<int> drop;                      // [i] of the input: the copy leaves (owned, not kept)
    std::vector<std::vector<int>> layer_slots;  // [layer] its slots (its kept copies first, then empty ones)
};

// An install: the copies of the LRU part of a class (`pos`, every position of the class; `keepable[i]` = 0 when the copy
// must leave: its expert is static now, or it has no admissible home) over the new shares `share` [layer]. Each layer
// keeps its most recently used keepable copies (lowest age, then lower slot), up to its share, in their slots; the other
// copies leave. The free slots (empty, or their copy left) go, empty, to the layers below their share, lower layers
// first, the lowest slots first. Nothing moves. Slots no layer takes stay with no layer.
inline l1_lru_reshare_plan l1_lru_reshare(const std::vector<l1_lru_pos> & pos, const std::vector<uint8_t> & keepable,
        const std::vector<int> & share, int n_layers) {
    l1_lru_reshare_plan p;
    p.layer_slots.assign((size_t) std::max(n_layers, 0), {});
    const size_t n = pos.size();
    std::vector<std::vector<size_t>> by_layer((size_t) std::max(n_layers, 0));
    for (size_t i = 0; i < n; ++i) {
        const int l = pos[i].layer;
        if (pos[i].owner >= 0 && i < keepable.size() && keepable[i] && l >= 0 && l < n_layers) {
            by_layer[(size_t) l].push_back(i);
        }
    }
    std::vector<uint8_t> kept(n, 0);
    for (size_t l = 0; l < by_layer.size(); ++l) {
        auto & v = by_layer[l];
        std::stable_sort(v.begin(), v.end(), [&](size_t a, size_t b) {
            if (pos[a].age != pos[b].age) { return pos[a].age < pos[b].age; }
            return pos[a].slot < pos[b].slot;
        });
        const size_t s = l < share.size() ? (size_t) std::max(share[l], 0) : 0;
        for (size_t j = 0; j < v.size() && j < s; ++j) {
            kept[v[j]] = 1;
            p.layer_slots[l].push_back(pos[v[j]].slot);
        }
    }
    std::vector<int> free;
    for (size_t i = 0; i < n; ++i) {
        if (kept[i]) {
            p.keep.push_back((int) i);
        } else {
            if (pos[i].owner >= 0) { p.drop.push_back((int) i); }
            free.push_back(pos[i].slot);
        }
    }
    l1_lru_revive(p.layer_slots, share, free);
    return p;
}

} // namespace ggml_cuda_expert
