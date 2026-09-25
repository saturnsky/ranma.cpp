#pragma once

// Early routes of the expert cache: layers whose router selection is a pure function of the input
// token id (the model declares them in ggml_expert_config::early_route_layers). The caller names the
// experts of those layers before the graph is built (ggml_expert_iface::route_hint), and the cache
// can start moving them while the graph is still being launched:
//
//   ssd   the SSD tier reads the file-tier experts into its staging ring before the layer publishes
//         its demand (expert-l2.cuh, post_early), so the layer's wait finds them in place;
//   vram  host-resident experts are copied into a few VRAM staging slots per layer on a side
//         stream, and the layer's slot table points at them, so the layer's matmuls read VRAM
//         instead of host memory over the link (expert-hash-stage.cuh).
//
// RANMA_EXPERT_HASH_EARLY = 0 (default) | ssd | vram | both. 0 keeps every path exactly as before.
// This header is the pure host part: the parsing of the switch, the distinct ids of a hint and the
// slot plan of the VRAM staging. No device code.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace ggml_cuda_expert {

struct hash_early_mode {
    bool ssd  = false;
    bool vram = false;
    bool any() const { return ssd || vram; }
};

// "0", "off", "" -> off; "ssd", "vram", "both". False for anything else (the caller keeps off).
inline bool parse_hash_early(const char * text, hash_early_mode & out) {
    out = hash_early_mode();
    if (text == nullptr || text[0] == '\0' || strcmp(text, "0") == 0 || strcmp(text, "off") == 0) { return true; }
    if (strcmp(text, "ssd") == 0)  { out.ssd = true; return true; }
    if (strcmp(text, "vram") == 0) { out.vram = true; return true; }
    if (strcmp(text, "both") == 0) { out.ssd = out.vram = true; return true; }
    return false;
}

inline const char * hash_early_name(const hash_early_mode & m) {
    return m.ssd && m.vram ? "both" : m.ssd ? "ssd" : m.vram ? "vram" : "0";
}

// The distinct valid expert ids of a hint, in first-seen order (row order, then the router's order).
inline std::vector<int> hint_distinct_ids(const int32_t * ids, int n_ids, int n_experts) {
    std::vector<int> out;
    std::vector<bool> seen((size_t) (n_experts > 0 ? n_experts : 0), false);
    for (int i = 0; ids != nullptr && i < n_ids; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= n_experts || seen[(size_t) e]) { continue; }
        seen[(size_t) e] = true;
        out.push_back(e);
    }
    return out;
}

// One layer's VRAM staging change. `owners[k]` is the expert staging slot k holds (and the layer's
// slot table points at), -1 for none. Wanted experts that already own a slot keep it; the others
// take a slot with no owner first, then a slot whose owner is no longer wanted. At most
// owners.size() experts are staged; the rest stay where they are (`over`). Duplicates in `wanted`
// count once.
struct stage_plan {
    std::vector<std::pair<int, int>> clears;   // (expert, slot): the table entry goes before the slot is rewritten
    std::vector<std::pair<int, int>> copies;   // (expert, slot): copy the slices, then point the table at the slot
    std::vector<std::pair<int, int>> keeps;    // (expert, slot): already staged; the entry is written again
    int over = 0;
    bool changed() const { return !clears.empty() || !copies.empty(); }
};

inline stage_plan plan_stage(std::vector<int32_t> & owners, const std::vector<int> & wanted) {
    stage_plan out;
    const int slots = (int) owners.size();
    std::vector<bool> kept((size_t) slots, false);
    std::vector<int> pending;
    for (size_t i = 0; i < wanted.size(); ++i) {
        const int e = wanted[i];
        if (std::find(wanted.begin(), wanted.begin() + (std::ptrdiff_t) i, e) != wanted.begin() + (std::ptrdiff_t) i) {
            continue;   // a duplicate: one slot per expert
        }
        int held = -1;
        for (int k = 0; k < slots && held < 0; ++k) {
            if (owners[(size_t) k] == e) { held = k; }
        }
        if (held >= 0) {
            kept[(size_t) held] = true;
            out.keeps.push_back({e, held});
        } else {
            pending.push_back(e);
        }
    }
    // free slots first, then the slots of experts that are no longer wanted
    std::vector<int> order;
    for (int k = 0; k < slots; ++k) { if (!kept[(size_t) k] && owners[(size_t) k] < 0) { order.push_back(k); } }
    for (int k = 0; k < slots; ++k) { if (!kept[(size_t) k] && owners[(size_t) k] >= 0) { order.push_back(k); } }
    size_t next = 0;
    for (int e : pending) {
        if (next >= order.size()) { ++out.over; continue; }
        const int k = order[next++];
        if (owners[(size_t) k] >= 0) { out.clears.push_back({owners[(size_t) k], k}); }
        owners[(size_t) k] = e;
        out.copies.push_back({e, k});
    }
    return out;
}

// An install republishes the slot tables from its plan: every staging entry is gone, and a staged
// expert may now have a VRAM or file home. The staging then owns nothing; the next hint copies again.
// (Keeping the owners would leave owners that the next hint does not name without a table entry.)
inline void stage_forget(std::vector<int32_t> & owners) {
    for (int32_t & e : owners) { e = -1; }
}

// The invariant between a layer's staging owners and its slot table ([expert] -> arena slot, -1):
// each owner is mapped to its slot, and no other expert is mapped into the staging range
// [first, first + owners.size()). False with the first violation in `why`.
inline bool stage_table_check(const std::vector<int32_t> & owners, const std::vector<int32_t> & table, int first,
        std::string & why) {
    const int slots = (int) owners.size();
    for (int k = 0; k < slots; ++k) {
        const int e = owners[(size_t) k];
        if (e < 0) { continue; }
        if (e >= (int) table.size() || table[(size_t) e] != first + k) {
            why = "the table maps expert " + std::to_string(e) + " to " +
                (e < (int) table.size() ? std::to_string(table[(size_t) e]) : std::string("nothing")) +
                ", not to its staging slot " + std::to_string(first + k);
            return false;
        }
    }
    for (size_t e = 0; e < table.size(); ++e) {
        const int k = table[e] - first;
        if (k >= 0 && k < slots && owners[(size_t) k] != (int32_t) e) {
            why = "expert " + std::to_string(e) + " points at staging slot " + std::to_string(table[e]) +
                " that holds expert " + std::to_string(owners[(size_t) k]);
            return false;
        }
    }
    return true;
}

} // namespace ggml_cuda_expert
