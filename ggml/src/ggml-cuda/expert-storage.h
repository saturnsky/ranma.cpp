#pragma once

// Storage classes of the finite host tier. Pure CPU code: no CUDA/HIP headers, no I/O.
//
// With a finite host budget every routed expert slice that is not in VRAM lives in a slot of a
// per (storage class, kind) host arena: the host residents of the plan and the staging ring that
// the SSD tier reads file residents into share one arena and one slot pitch. A slot can therefore
// hold a resident or a ring occupant without moving its bytes.
//
// A storage class is one geometry (size) class, or several merged into one: a class of one or two
// layers would otherwise need a ring of its own that holds its worst layer's file residents at once,
// which costs far more than the padding its residents waste in the slots of a class of larger
// pitch.
//
// The slot pitch of a kind leaves room for the sector shift of the file offset (GGUF tensor starts
// are not sector aligned), so an unbuffered read of the covering sectors fits in the slot and the
// payload starts at slot + shift, plus the zeroed tail MMQ reads past the last row.

#include "expert-geometry.h"
#include "expert-location.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

inline size_t storage_align_up(size_t value, size_t align) {
    return (value + align - 1)/align*align;
}

// One slot of a kind: the slice, the largest sector shift and the MMQ tail, in whole sectors.
inline size_t storage_slot_pitch(size_t slice, size_t tail, size_t align) {
    return storage_align_up(slice + align - 1 + tail, align);
}

struct class_storage {
    std::vector<int> storage_of;                    // [geometry class] storage class
    std::vector<std::vector<int>> members;          // [storage class] geometry classes, ascending
    std::vector<std::array<size_t, 3>> pitch;       // [storage class][kind] slot pitch
    std::vector<std::array<size_t, 3>> tail;        // [storage class][kind] zeroed bytes past the last slot
    std::vector<int> resident_base, resident_slots; // [geometry class] its resident slot range, spares included
    std::vector<int> ring_base, ring_slots, floor;  // [storage class] the ring range and its prompt floor
    double factor = 1.0;                            // the ring factor the rings got, 0 = an explicit total

    int storages() const { return (int) members.size(); }
    bool valid() const { return !members.empty() && ring_slots.size() == members.size(); }
    size_t stride(int s) const { return pitch[s][0] + pitch[s][1] + pitch[s][2]; }
    int slots(int s) const { return ring_base[s] + ring_slots[s]; }
    int ring_total() const {
        int total = 0;
        for (int r : ring_slots) { total += r; }
        return total;
    }
    size_t ring_bytes() const {
        size_t total = 0;
        for (int s = 0; s < storages(); ++s) { total += size_t(ring_slots[s])*stride(s); }
        return total;
    }
    size_t tail_bytes() const {
        size_t total = 0;
        for (const auto & t : tail) { total += t[0] + t[1] + t[2]; }
        return total;
    }
    // Bytes one host slot of a geometry class costs, all kinds.
    size_t class_pitch(int cls) const { return stride(storage_of[cls]); }
    // The storage slot of resident slot `slot` of a geometry class.
    int resident_slot(int cls, int slot) const { return resident_base[cls] + slot; }
    // Every slot and tail of every arena.
    size_t arena_bytes() const {
        size_t total = tail_bytes();
        for (int s = 0; s < storages(); ++s) { total += size_t(slots(s))*stride(s); }
        return total;
    }

    // Lays out the slots: per storage class the resident ranges of its members in class order,
    // then the ring. `residents` counts spares; `ring` is per storage class.
    bool place(const std::vector<int> & residents, const std::vector<int> & ring) {
        if (residents.size() != storage_of.size() || ring.size() != members.size()) { return false; }
        resident_base.assign(storage_of.size(), 0);
        resident_slots = residents;
        ring_base.assign(members.size(), 0);
        ring_slots = ring;
        for (int s = 0; s < storages(); ++s) {
            int next = 0;
            for (int c : members[s]) {
                if (residents[c] < 0) { return false; }
                resident_base[c] = next;
                next += residents[c];
            }
            if (ring[s] < 0) { return false; }
            ring_base[s] = next;
        }
        return true;
    }
};

// Pitches of the geometry classes and the merge rule. `tails` is [class][kind] (arena_tail_bytes).
// `floor_slots` is the prompt floor a separate ring of a class would need; a class of at most two
// layers is merged into the nearest storage class of at least its stride when the floor bytes the
// merge saves exceed the padding every expert of the affected classes would waste as a resident.
// Classes are visited in ascending stride, so a chain of small classes ends in the largest.
inline class_storage plan_storage_classes(const geometry & geo, const std::vector<std::array<size_t, 3>> & tails,
        size_t align, int floor_slots, bool merge = true) {
    class_storage out;
    const int classes = (int) geo.class_bytes.size();
    if (classes == 0 || tails.size() != (size_t) classes || align == 0) { return out; }
    out.storage_of.resize(classes);
    for (int c = 0; c < classes; ++c) {
        out.storage_of[c] = c;
        out.members.push_back({c});
        std::array<size_t, 3> p = {0, 0, 0};
        for (int k = 0; k < geometry::n_kinds; ++k) { p[k] = storage_slot_pitch(geo.class_bytes[c][k], tails[c][k], align); }
        out.pitch.push_back(p);
        out.tail.push_back(tails[c]);
    }
    if (merge && floor_slots > 0) {
        // ascending stride, ties in class order (an insertion sort: std::stable_sort instantiates a
        // std::min that clashes with the HIP wrapper's in device translation units on Windows)
        std::vector<int> order;
        for (int c = 0; c < classes; ++c) {
            size_t at = order.size();
            while (at > 0 && out.stride(order[at - 1]) > out.stride(c)) { --at; }
            order.insert(order.begin() + (std::ptrdiff_t) at, c);
        }
        const uint64_t experts = (uint64_t) std::max(geo.n_experts, 0);
        for (int c : order) {
            const int own = out.storage_of[c];
            if (geo.class_layers[c] > 2 || out.members[own].size() != 1) { continue; }
            int target = -1;
            for (int s = 0; s < (int) out.members.size(); ++s) {
                if (s == own || out.members[s].empty() || out.stride(s) < out.stride(own)) { continue; }
                if (target < 0 || out.stride(s) < out.stride(target)) { target = s; }
            }
            if (target < 0) { continue; }
            std::array<size_t, 3> merged = out.pitch[target];
            for (int k = 0; k < geometry::n_kinds; ++k) { merged[k] = std::max(merged[k], out.pitch[own][k]); }
            const size_t merged_stride = merged[0] + merged[1] + merged[2];
            uint64_t waste = 0;
            for (int m : out.members[target]) { waste += (uint64_t) geo.class_layers[m]*experts*(merged_stride - out.stride(target)); }
            waste += (uint64_t) geo.class_layers[c]*experts*(merged_stride - out.stride(own));
            const uint64_t before = (uint64_t) floor_slots*(out.stride(own) + out.stride(target));
            const uint64_t after  = (uint64_t) floor_slots*merged_stride;
            if (before <= after || before - after <= waste) { continue; }
            out.pitch[target] = merged;
            for (int k = 0; k < geometry::n_kinds; ++k) { out.tail[target][k] = std::max(out.tail[target][k], out.tail[own][k]); }
            out.members[target].push_back(c);
            std::sort(out.members[target].begin(), out.members[target].end());
            out.members[own].clear();
            out.storage_of[c] = target;
        }
        // Renumber the storage classes that are left, in the order of their first member.
        std::vector<int> renumber(out.members.size(), -1);
        class_storage packed;
        packed.storage_of.resize(classes);
        for (int c = 0; c < classes; ++c) {
            const int s = out.storage_of[c];
            if (renumber[s] < 0) {
                renumber[s] = (int) packed.members.size();
                packed.members.push_back(out.members[s]);
                packed.pitch.push_back(out.pitch[s]);
                packed.tail.push_back(out.tail[s]);
            }
            packed.storage_of[c] = renumber[s];
        }
        out = packed;
    }
    out.resident_base.assign(classes, 0);
    out.resident_slots.assign(classes, 0);
    out.ring_base.assign(out.members.size(), 0);
    out.ring_slots.assign(out.members.size(), 0);
    out.floor.assign(out.members.size(), 0);
    return out;
}

// The most distinct experts one ubatch of `rows` rows can demand from one layer.
inline int storage_demand_bound(int experts, int used, uint64_t rows) {
    if (experts <= 0 || used <= 0 || rows == 0) { return 0; }
    return (int) std::min<uint64_t>((uint64_t) experts, (uint64_t) used*std::min<uint64_t>(rows, (uint64_t) experts));
}

// The ring of a storage class must hold every file resident of any one of its layers at once: a
// prompt ubatch can demand them all, and they stay leased until that layer is done. Per storage
// class the largest file resident count of one of its layers under the plan (`vram` [layer][expert]
// slot or -1, `host` [layer] host-resident ids), bounded by `bound`. Without a host plan every class
// gets the bound.
inline std::vector<int> storage_floors(const geometry & geo, const class_storage & st,
        const expert_slot_table * vram, const expert_slot_table * host, int bound) {
    std::vector<int> out((size_t) st.storages(), 0);
    for (int l = 0; l < geo.n_layers; ++l) {
        const int c = geo.layer_class[l];
        if (c < 0) { continue; }
        const int s = st.storage_of[c];
        if (vram == nullptr || host == nullptr || (size_t) l >= vram->size() || (size_t) l >= host->size()) {
            out[s] = bound;
            continue;
        }
        std::vector<bool> resident((size_t) geo.n_experts, false);
        for (int e = 0; e < geo.n_experts && (size_t) e < (*vram)[l].size(); ++e) { resident[e] = (*vram)[l][e] >= 0; }
        for (int e : (*host)[l]) { if (e >= 0 && e < geo.n_experts) { resident[e] = true; } }
        int file = 0;
        for (int e = 0; e < geo.n_experts; ++e) { file += resident[e] ? 0 : 1; }
        out[s] = std::max(out[s], std::min(file, bound));
    }
    return out;
}

// The default ring of each storage class: `factor` times its floor (rounded up), at most `limit`
// slots and never below the floor. A class ring is shared by every layer of the class during
// decode, so above the floor it needs room for each layer's reusable experts to survive until that
// layer comes round again.
inline std::vector<int> factor_rings(const std::vector<int> & floors, double factor, const std::vector<int> & limit) {
    std::vector<int> out = floors;
    if (limit.size() != floors.size() || !(factor >= 1.0)) { return out; }
    for (size_t s = 0; s < floors.size(); ++s) {
        const double want = std::ceil(double(floors[s])*factor);
        out[s] = std::max(floors[s], int(std::min(want, double(std::max(limit[s], floors[s])))));
    }
    return out;
}

// When the budget cannot hold the rings `want` above the floors, every class keeps its floor and
// gets the same fraction of what it wanted above it, so that the extra bytes fit in `available`.
inline std::vector<int> shrink_rings(const class_storage & st, const std::vector<int> & floors,
        const std::vector<int> & want, size_t available) {
    const int n = st.storages();
    if ((int) floors.size() != n || (int) want.size() != n) { return floors; }
    size_t extra = 0;
    for (int s = 0; s < n; ++s) { extra += size_t(std::max(want[s] - floors[s], 0))*st.stride(s); }
    if (extra <= available) { return want; }
    const double fraction = double(available)/double(extra);
    std::vector<int> out = floors;
    for (int s = 0; s < n; ++s) {
        out[s] += int(std::floor(double(std::max(want[s] - floors[s], 0))*fraction));
    }
    return out;
}

// An explicit total ring size split over the storage classes: every class gets its floor, and the
// bytes above the floors go to the classes in proportion to `demand` (bytes each class leaves in the
// file), never more slots than `limit`. A total below the floors gives the floors.
inline std::vector<int> split_ring(const class_storage & st, const std::vector<int> & floors,
        const std::vector<uint64_t> & demand, const std::vector<int> & limit, size_t total_bytes) {
    const int n = st.storages();
    std::vector<int> out = floors;
    if ((int) floors.size() != n || (int) demand.size() != n || (int) limit.size() != n) { return out; }
    size_t used = 0;
    for (int s = 0; s < n; ++s) { used += size_t(floors[s])*st.stride(s); }
    if (total_bytes <= used) { return out; }
    size_t extra = total_bytes - used;
    uint64_t weight = 0;
    for (int s = 0; s < n; ++s) { weight += out[s] < limit[s] ? demand[s] : 0; }
    if (weight != 0) {
        for (int s = 0; s < n; ++s) {
            if (out[s] >= limit[s] || demand[s] == 0) { continue; }
            const double share = double(extra)*double(demand[s])/double(weight);
            const int add = std::min(limit[s] - out[s], int(share/double(st.stride(s))));
            out[s] += add;
            used += size_t(add)*st.stride(s);
        }
    }
    // What is left goes one slot at a time to the class with the most demand per slot it holds.
    extra = total_bytes - used;
    while (true) {
        int best = -1;
        for (int s = 0; s < n; ++s) {
            if (out[s] >= limit[s] || st.stride(s) > extra) { continue; }
            if (best < 0 || double(demand[s])/double(out[s] + 1) > double(demand[best])/double(out[best] + 1)) { best = s; }
        }
        if (best < 0) { break; }
        ++out[best];
        extra -= st.stride(best);
    }
    return out;
}

} // namespace ggml_cuda_expert
