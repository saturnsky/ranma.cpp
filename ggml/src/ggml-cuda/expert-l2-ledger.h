#pragma once

// Demand-only staging ring. Slots stay pinned until their consuming layer completes.
// The inactive ring tail can hold residents between phase changes.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

struct l2_ring_plan {
    bool valid = false;
    int prompt_slots = 0, decode_slots = 0;
    int prompt_floor = 0, decode_floor = 0;
    size_t reserve_bytes = 0;
    std::string note;
};

inline l2_ring_plan plan_l2_ring(int experts, int used, uint64_t prompt_rows, uint64_t decode_rows,
        size_t stride, size_t prompt_bytes, size_t decode_bytes, bool phases) {
    l2_ring_plan out;
    if (experts <= 0 || used <= 0 || used > experts || !prompt_rows || !decode_rows || !stride) { return out; }
    auto demand = [&](uint64_t rows) { return int(std::min(uint64_t(experts), uint64_t(used)*std::min(rows, uint64_t(experts)))); };
    out.prompt_floor = demand(prompt_rows);
    out.decode_floor = demand(decode_rows);
    const size_t requested_p = prompt_bytes/stride, requested_d = decode_bytes/stride;
    if (requested_p > INT32_MAX || requested_d > INT32_MAX || size_t(experts) > SIZE_MAX/stride) { return out; }
    out.prompt_slots = std::max(out.prompt_floor, int(requested_p));
    if (prompt_bytes && requested_p < size_t(out.prompt_floor)) { out.note = "prefill ring raised to its demand floor"; }
    if (decode_bytes && phases) {
        out.decode_slots = std::max(out.decode_floor, int(requested_d));
        if (requested_d < size_t(out.decode_floor)) { out.note += (out.note.empty() ? "" : "; ") + std::string("decode ring raised to its demand floor"); }
    } else {
        const size_t reserve_slots = std::min(size_t(512)*1024*1024/stride,
            size_t(std::max(0, out.prompt_slots - out.decode_floor)));
        out.decode_slots = out.decode_floor + int(reserve_slots);
    }
    if (out.decode_slots > out.prompt_slots) {
        out.prompt_slots = out.decode_slots;
        out.note += (out.note.empty() ? "" : "; ") + std::string("prefill allocation raised to fit the decode ring");
    }
    if (!phases) { out.decode_slots = out.prompt_slots; }
    out.reserve_bytes = size_t(out.decode_slots - out.decode_floor)*stride;
    out.valid = size_t(out.prompt_slots) <= SIZE_MAX/stride;
    return out;
}

// One slice to read from the file into a ring slot or a host slot.
struct l2_read {
    int layer  = 0;
    int kind   = 0;
    int expert = 0;
    int slot   = 0;   // ring slot
};

struct l2_eviction {
    int layer, expert, slot;
};

struct l2_service {
    bool ok = true;
    std::string reason;
    std::vector<l2_read> reads;
    std::vector<l2_eviction> evicted;
    int hits   = 0;   // experts already in a ring slot
    int misses = 0;   // experts that needed a read
};

class l2_ledger {
public:
    void reset(int layers, int experts, int max_slots, int ring_count, int kinds = 3) {
        layers_     = layers;
        experts_    = experts;
        max_slots_  = max_slots;
        ring_count_ = ring_count;
        kinds_      = kinds;
        owners_.assign((size_t) max_slots, owner());
        done_.assign((size_t) layers, 0u);
        map_.assign((size_t) layers, std::vector<int32_t>((size_t) experts, -1));
        cursor_ = 0;
    }

    int  ring_count() const { return ring_count_; }
    int  max_slots()  const { return max_slots_; }
    int  slot_of(int layer, int expert) const { return map_[(size_t) layer][(size_t) expert]; }
    bool owns(int slot, int layer, int expert) const {
        if (slot < 0 || slot >= ring_count_) {
            return false;
        }
        const owner & o = owners_[(size_t) slot];
        return o.layer == layer && o.expert == expert;
    }
    // Slots [ring_count, max_slots) are lent to the host tier while the ring is small.
    int  borrowed_slots() const { return max_slots_ - ring_count_; }

    // The mailbox `done` counter of a layer, as the worker last read it.
    void set_done(int layer, uint32_t done) { done_[(size_t) layer] = done; }
    uint32_t done(int layer) const { return done_[(size_t) layer]; }

    void discard() {
        for (std::vector<int32_t> & row : map_) {
            std::fill(row.begin(), row.end(), int32_t(-1));
        }
        owners_.assign((size_t) max_slots_, owner());
        cursor_ = 0;
    }

    // Changes the active ring size and empties the ring. False when the size is out of range.
    bool set_ring_count(int count) {
        if (count <= 0 || count > max_slots_) {
            return false;
        }
        ring_count_ = count;
        discard();
        return true;
    }

    // `homed(layer, expert)` is true when the expert already lives in the VRAM or host
    // arena, so it needs no ring slot. Experts repeat across the top-k of several rows; the caller
    // may pass duplicates.
    template <typename Homed>
    l2_service service(int layer, const std::vector<int> & ids, uint32_t seq, Homed homed) {
        l2_service out;
        std::vector<bool> asked((size_t) experts_, false);
        std::vector<int>  wanted;
        for (int expert : ids) {
            if (expert < 0 || expert >= experts_) {
                out.ok = false;
                out.reason = "router id " + std::to_string(expert) + " is out of range";
                return out;
            }
            if (!asked[(size_t) expert]) {
                asked[(size_t) expert] = true;
                wanted.push_back(expert);
            }
        }
        // Everything already in the ring is pinned before any slot is handed out, or a top-k
        // larger than the free part of the ring would evict its own earlier experts.
        std::vector<bool> pinned((size_t) max_slots_, false);
        for (int expert : wanted) {
            const int slot = map_[(size_t) layer][(size_t) expert];
            if (slot >= 0) {
                pinned[(size_t) slot] = true;
            }
        }
        for (int expert : wanted) {
            if (homed(layer, expert)) {
                continue;
            }
            int slot = map_[(size_t) layer][(size_t) expert];
            if (slot >= 0) {
                ++out.hits;
                owners_[(size_t) slot].seq = seq;   // extend the lease to this generation
                continue;
            }
            slot = acquire(layer, expert, seq, pinned, out.evicted);
            if (slot < 0) {
                out.ok = false;
                out.reason = "no reusable ring slot for layer " + std::to_string(layer) +
                    " expert " + std::to_string(expert);
                return out;
            }
            pinned[(size_t) slot] = true;
            ++out.misses;
            for (int kind = 0; kind < kinds_; ++kind) {
                out.reads.push_back({layer, kind, expert, slot});
            }
        }
        return out;
    }

private:
    struct owner {
        int      layer   = -1;
        int      expert  = -1;
        uint32_t seq     = 0;
    };

    // Wrap safe: a slot is free when the GPU has finished the generation that claimed it.
    bool reusable(const owner & o) const {
        if (o.layer < 0) {
            return true;
        }
        return int32_t(done_[(size_t) o.layer] - o.seq) >= 0;
    }

    int acquire(int layer, int expert, uint32_t seq, const std::vector<bool> & pinned,
            std::vector<l2_eviction> & evicted) {
        for (int i = 0; i < ring_count_; ++i) {
            const int slot = cursor_;
            cursor_ = (cursor_ + 1)%ring_count_;
            if (pinned[(size_t) slot] || !reusable(owners_[(size_t) slot])) {
                continue;
            }
            owner & o = owners_[(size_t) slot];
            if (o.layer >= 0) {
                evicted.push_back({o.layer, o.expert, slot});
                map_[(size_t) o.layer][(size_t) o.expert] = -1;
            }
            o = owner{layer, expert, seq};
            map_[(size_t) layer][(size_t) expert] = slot;
            return slot;
        }
        return -1;
    }

    int layers_     = 0;
    int experts_    = 0;
    int max_slots_  = 0;
    int ring_count_ = 0;
    int kinds_      = 3;
    int cursor_     = 0;
    std::vector<owner>   owners_;
    std::vector<uint32_t> done_;
    std::vector<std::vector<int32_t>> map_;
};

} // namespace ggml_cuda_expert
