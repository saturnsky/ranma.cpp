#pragma once

// Batched service of the SSD tier for prompt ubatches. Pure CPU code: no CUDA/HIP headers, no I/O.
//
// A prompt ubatch can demand every file resident of a layer at once. Without batching, the ring of
// the layer's storage class must hold all of them until the layer is done, which ties the ring to
// the ubatch and the plan (the prompt floor). With batching, the MUL_MAT_ID of each kind runs as
// one launch over the experts that need no read (VRAM and host residents, ring hits) followed by
// launches over batches of the file residents. The batches go through two buffers of ring slots:
// while the GPU multiplies batch i, the worker reads batch i + 1, and batch i + 2 reuses the slots
// of batch i once the GPU reports batch i done. Every slice is read once, the three kinds of an
// expert share one slot, and each expert is multiplied by exactly one launch with the same tiles as
// the single launch, so the output does not change.
//
// The graph holds a fixed number of batch launches per kind and layer (`l2_batch_launches`); a
// generation that needs fewer leaves the rest empty. The number follows from the experts one ubatch
// can demand and the batch capacity, half the storage class ring.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// Where the fills and hits of a prompt ubatch go in the least recently used list of a ring. Decode
// always fills and hits at the most recently used end.
//  mru:  as decode.
//  lru:  fills at the least recently used end (they go first), hits move to the most recently used end.
//  scan: fills at the least recently used end, hits stay where they are (a prompt is a scan).
enum class l2_prompt_fill : uint8_t { mru, lru, scan };

inline const char * l2_prompt_fill_name(l2_prompt_fill fill) {
    switch (fill) {
        case l2_prompt_fill::mru:  return "mru";
        case l2_prompt_fill::lru:  return "lru";
        case l2_prompt_fill::scan: return "scan";
    }
    return "?";
}

struct l2_insert {
    bool cold_fill = false;   // a filled slot goes to the least recently used end
    bool touch_hits = true;   // a hit moves to the most recently used end
};

inline l2_insert l2_insert_for(l2_prompt_fill fill, bool prompt) {
    l2_insert out;
    if (prompt && fill != l2_prompt_fill::mru) {
        out.cold_fill  = true;
        out.touch_hits = fill == l2_prompt_fill::lru;
    }
    return out;
}

// Experts one batch launch covers when the storage class ring has `ring_slots` slots: half the ring,
// so that two batches are in the ring at once.
inline int l2_batch_capacity(int ring_slots) {
    return std::max(ring_slots/2, 1);
}

// The batch launches per kind that a ubatch whose worst demand is `demand` experts needs with
// `capacity` experts per batch. 0 when there is nothing to batch.
inline int l2_batch_launches(int demand, int capacity) {
    if (demand <= 0 || capacity <= 0) { return 0; }
    return (demand + capacity - 1)/capacity;
}

// One generation's batches, as the ledger planned them.
struct l2_batch_plan {
    bool ok = true;
    std::string reason;
    int hits = 0, early_hits = 0;
    int misses = 0;      // experts read, in the batches
    int demoted = 0;     // ring hits read again so that the batches have room (counted in misses, not hits)
    int buffers = 0;     // 1 or 2: batch b uses buffer b % buffers
    int capacity = 0;    // the most experts of one batch
    std::vector<std::vector<int>> batches;   // [batch] experts, ascending
    std::vector<std::vector<int>> slots;     // [buffer][position] storage slot of the layer's class
    std::vector<int> hit_experts;            // the ring hits kept, served by the first launch
    struct eviction { int layer, expert, slot; };
    std::vector<eviction> evicted;

    int count() const { return (int) batches.size(); }
    int slot(int batch, int position) const { return slots[(size_t) (batch % buffers)][(size_t) position]; }
    // The expert the slot of (batch, position) held for the batch before it in the same buffer, -1 if none.
    int previous(int batch, int position) const {
        if (batch < buffers) { return -1; }
        const std::vector<int> & before = batches[(size_t) (batch - buffers)];
        return position < (int) before.size() ? before[(size_t) position] : -1;
    }
    // Whether batch `batch` may be read once the GPU has finished `done` batches of the kind.
    bool readable(int batch, int done) const { return batch < buffers || done >= batch - buffers + 1; }
};

// Splits `experts` (ascending) into the fewest batches of at most `capacity`, all of nearly equal size.
inline std::vector<std::vector<int>> l2_split_batches(const std::vector<int> & experts, int capacity) {
    std::vector<std::vector<int>> out;
    const int m = (int) experts.size();
    if (m == 0 || capacity <= 0) { return out; }
    const int k = (m + capacity - 1)/capacity;
    const int base = m/k, extra = m%k;   // the first `extra` batches take one more
    size_t at = 0;
    for (int b = 0; b < k; ++b) {
        const int size = base + (b < extra ? 1 : 0);
        out.emplace_back(experts.begin() + (std::ptrdiff_t) at, experts.begin() + (std::ptrdiff_t) (at + (size_t) size));
        at += (size_t) size;
    }
    return out;
}

// The order in which the worker reads the batches of a generation: up, gate, down, each kind in
// batch order; a batch that waits for an earlier batch of its kind to finish on the GPU does not
// hold back the batches of the other kinds. `done[kind]` is the number of batches the GPU has
// finished; returns the (kind, batch) pairs that may be read now, in priority order, and advances
// `next`.
inline std::vector<std::pair<int, int>> l2_batch_readable(const l2_batch_plan & plan, std::vector<int> & next,
        const std::vector<int> & done) {
    std::vector<std::pair<int, int>> out;
    for (int kind = 0; kind < (int) next.size(); ++kind) {
        while (next[(size_t) kind] < plan.count() && plan.readable(next[(size_t) kind], done[(size_t) kind])) {
            out.push_back({kind, next[(size_t) kind]});
            ++next[(size_t) kind];
        }
    }
    return out;
}

} // namespace ggml_cuda_expert
