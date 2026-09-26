#pragma once

// Host arenas of the exclusive expert cache without a finite host tier (exclusive mode with an
// unlimited host budget). Pure CPU code: no CUDA/HIP headers, no I/O.
//
// Without a finite tier the host arena of a size class and kind is one registered allocation, and a
// kernel reads a host resident at host_data + host_slots[expert] * nb[2]. A redraw of the VRAM
// size-class split (expert-redraw.h) needs the host capacity of a class to follow its VRAM share: an
// expert that leaves VRAM must find a host slot of its class. With the redraw on in this mode:
//   - the host arena also publishes, per (layer, kind), the device address of every host resident
//     slice (0 for an expert in VRAM), and the lookup hands it to the kernels as host_addresses, which
//     they take before the slot table (ggml_cuda_expert_cache_select), the same table the finite tier
//     publishes;
//   - each (class, kind) arena is a list of chunks of about host_chunk_mib, each allocated and
//     registered on its own. The device aliases of separate registrations are not contiguous, which
//     the table makes irrelevant; in exchange a class's host capacity can grow (new chunks) and shrink
//     (the residents above the new count move down, the top chunks are released).

#include "expert-geometry.h"
#include "expert-location.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

static constexpr size_t host_chunk_mib = 128;   // chunk size of a chunked host arena

// Slots per chunk of a class: the most whole slices of the class's largest kind that fit in
// `chunk_bytes`, at least one. Every kind of a class uses the same count, so a slot index is in the
// same chunk in every kind.
inline int host_chunk_slots(const geometry & geo, int cls, size_t chunk_bytes) {
    size_t largest = 0;
    for (int k = 0; k < geometry::n_kinds; ++k) { largest = std::max(largest, geo.class_bytes[(size_t) cls][(size_t) k]); }
    if (largest == 0) { return 1; }
    return (int) std::max<size_t>(1, std::min<size_t>(chunk_bytes/largest, size_t(1) << 30));
}

// Chunks per (class, kind) for `slots` slots; at least one, so that the arena base of a class never
// goes away (a captured graph holds it as the lookup's host_data).
inline int host_chunks_for(int slots, int per_chunk) {
    return std::max(1, (std::max(slots, 0) + per_chunk - 1)/std::max(per_chunk, 1));
}

// Committed bytes of the chunks of one (class, kind), each with its own zeroed tail.
inline size_t host_chunked_bytes(size_t slice, size_t tail, int slots, int per_chunk) {
    return size_t(host_chunks_for(slots, per_chunk))*(size_t(per_chunk)*slice + tail);
}

struct host_move {
    int layer, expert, from, to;
};

// Compaction of the host slots of class `cls` below `count` (the new slot count, spares included):
// every host resident at or above it moves to the lowest free slot below it. `slots` is the current
// slot count. False, with the reason, when the class has more host residents than `count`.
inline bool plan_host_compaction(const std::vector<int> & layer_class, int n_experts, const expert_locations & homes,
        int cls, int slots, int count, std::vector<host_move> & out, std::string & why) {
    out.clear();
    why.clear();
    if (count < 0 || slots < 0) { why = "negative host slot count"; return false; }
    if (homes.size() != layer_class.size()) { why = "host table dimensions"; return false; }
    std::vector<uint8_t> used(size_t(std::max(slots, count)), 0);
    std::vector<std::pair<int, int>> above;
    std::vector<int> above_slot;
    for (size_t l = 0; l < layer_class.size(); ++l) {
        if (layer_class[l] != cls) { continue; }
        if (homes[l].size() != size_t(n_experts)) { why = "host table dimensions"; return false; }
        for (int e = 0; e < n_experts; ++e) {
            const expert_location & h = homes[l][(size_t) e];
            if (h.storage != expert_storage::host) { continue; }
            if (h.slot < 0 || h.slot >= slots || used[(size_t) h.slot]) { why = "host slot out of range or shared"; return false; }
            used[(size_t) h.slot] = 1;
            if (h.slot >= count) { above.emplace_back(int(l), e); above_slot.push_back(h.slot); }
        }
    }
    int next = 0;
    for (size_t i = 0; i < above.size(); ++i) {
        while (next < count && used[(size_t) next]) { ++next; }
        if (next >= count) {
            why = "class " + std::to_string(cls) + " has more host residents than " + std::to_string(count) + " slots";
            out.clear();
            return false;
        }
        used[(size_t) next] = 1;
        out.push_back({above[i].first, above[i].second, above_slot[i], next});
    }
    return true;
}

} // namespace ggml_cuda_expert
