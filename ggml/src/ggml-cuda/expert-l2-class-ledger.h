#pragma once

// Slot ledger of the class layout of the SSD tier: one arena per (storage class, kind) holds both
// the host residents of the installed plan and the staging ring of that class.
//
// Every slot has one role: a host resident, a slot of the class's staging ring, or free. The ring of
// each class is its own least recently used list: a hit or a fill moves the slot to the most
// recently used end, and a miss takes the least recently used ring slot that is neither pinned by
// the current demand nor leased by a generation the GPU has not finished. Slot ids are per storage
// class; the storage class of a layer is fixed.
//
// Roles change only while nothing computes (the install transaction): a resident can become a ring
// occupant of the same expert in place (the ring grows, nothing is read), and a ring slot can become
// a resident (the ring shrinks; if the slot does not already hold that expert, the caller reads it).
// A ring never shrinks below its floor.

#include "expert-l2-ledger.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

enum class slot_role : uint8_t { free, resident, ring };

struct l2_relabel {
    bool ok   = false;
    bool read = false;   // the slot does not hold the expert yet: the caller reads it in
    std::string reason;
    std::vector<l2_eviction> evicted;
};

class l2_class_ledger {
public:
    // `layer_storage` [layer] is the storage class of a routed layer or -1; `slots` [storage class]
    // the arena slot count. Every slot starts free.
    void reset(int layers, int experts, const std::vector<int> & layer_storage, const std::vector<int> & slots,
            int kinds = 3) {
        layers_  = layers;
        experts_ = experts;
        kinds_   = kinds;
        layer_storage_ = layer_storage;
        layer_storage_.resize((size_t) std::max(layers, 0), -1);
        const size_t n = slots.size();
        owners_.assign(n, {});
        prev_.assign(n, {});
        next_.assign(n, {});
        head_.assign(n, -1);
        tail_.assign(n, -1);
        ring_count_.assign(n, 0);
        floor_.assign(n, 0);
        for (size_t s = 0; s < n; ++s) {
            owners_[s].assign((size_t) std::max(slots[s], 0), owner());
            prev_[s].assign(owners_[s].size(), -1);
            next_[s].assign(owners_[s].size(), -1);
        }
        done_.assign((size_t) std::max(layers, 0), 0u);
        map_.assign((size_t) std::max(layers, 0), std::vector<int32_t>((size_t) std::max(experts, 0), -1));
    }

    int storages() const { return (int) owners_.size(); }
    int slots(int s) const { return (int) owners_[(size_t) s].size(); }
    int storage_of(int layer) const { return layer_storage_[(size_t) layer]; }
    int ring_count(int s) const { return ring_count_[(size_t) s]; }
    int ring_total() const {
        int total = 0;
        for (int r : ring_count_) { total += r; }
        return total;
    }
    int  floor(int s) const { return floor_[(size_t) s]; }
    void set_floor(int s, int slots) { floor_[(size_t) s] = std::max(slots, 0); }
    slot_role role_of(int s, int slot) const { return owners_[(size_t) s][(size_t) slot].role; }
    // The expert a slot holds, as a resident or as a ring occupant; false when it holds none.
    bool occupant(int s, int slot, int & layer, int & expert) const {
        const owner & o = owners_[(size_t) s][(size_t) slot];
        layer = o.layer; expert = o.expert;
        return o.layer >= 0;
    }
    // The ring slot of an expert in its layer's storage class, or -1.
    int  slot_of(int layer, int expert) const { return map_[(size_t) layer][(size_t) expert]; }
    // Whether ring slot `slot` of the storage class of `layer` holds (layer, expert).
    bool owns(int slot, int layer, int expert) const {
        const int s = layer >= 0 && layer < layers_ ? storage_of(layer) : -1;
        if (s < 0 || slot < 0 || slot >= slots(s)) { return false; }
        const owner & o = owners_[(size_t) s][(size_t) slot];
        return o.role == slot_role::ring && o.layer == layer && o.expert == expert;
    }

    void set_done(int layer, uint32_t done) { done_[(size_t) layer] = done; }
    uint32_t done(int layer) const { return done_[(size_t) layer]; }

    // The ring slots of a class from least to most recently used.
    std::vector<int> order(int s) const {
        std::vector<int> out;
        for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) { out.push_back(slot); }
        return out;
    }

    // A free slot joins the ring, empty, at the most recently used end.
    bool make_ring(int s, int slot) {
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role != slot_role::free) { return false; }
        owners_[(size_t) s][(size_t) slot] = owner();
        owners_[(size_t) s][(size_t) slot].role = slot_role::ring;
        link_tail(s, slot);
        ++ring_count_[(size_t) s];
        return true;
    }

    // A free or resident slot is the home of the resident (layer, expert) of the plan; layer -1
    // keeps it a resident slot without an expert (a spare).
    bool set_resident(int s, int slot, int layer, int expert) {
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role == slot_role::ring) { return false; }
        if (layer >= 0 && (layer >= layers_ || storage_of(layer) != s || expert < 0 || expert >= experts_)) { return false; }
        owner & o = owners_[(size_t) s][(size_t) slot];
        o = owner();
        o.role = slot_role::resident;
        if (layer >= 0) { o.layer = layer; o.expert = expert; }
        return true;
    }

    // Every resident slot keeps its role and forgets its expert (a plan is about to be mirrored).
    void clear_residents() {
        for (auto & row : owners_) {
            for (owner & o : row) {
                if (o.role == slot_role::resident) { o.layer = -1; o.expert = -1; o.seq = 0; }
            }
        }
    }

    // Grow: a resident slot joins the ring in place. Its expert, if any, becomes a ring occupant
    // without a read; `cold` puts the slot at the least recently used end, so it goes first.
    // Without `keep` the slot joins empty (its expert now lives elsewhere).
    bool relabel_to_ring(int s, int slot, bool cold, bool keep = true) {
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role != slot_role::resident) { return false; }
        owner & o = owners_[(size_t) s][(size_t) slot];
        o.role = slot_role::ring;
        if (!keep) { o.layer = -1; o.expert = -1; o.seq = 0; }
        if (o.layer >= 0) {
            // a resident is never also a ring occupant; drop a stale copy all the same
            const int other = map_[(size_t) o.layer][(size_t) o.expert];
            if (other >= 0 && other != slot) { empty(s, other); }
            map_[(size_t) o.layer][(size_t) o.expert] = slot;
            o.seq = done_[(size_t) o.layer];   // reusable at once: nothing computes during an install
        }
        if (cold) { link_head(s, slot); } else { link_tail(s, slot); }
        ++ring_count_[(size_t) s];
        return true;
    }

    // Shrink: a ring slot becomes the home of the resident (layer, expert). When it already holds
    // that expert nothing is read; otherwise its occupant is evicted and `read` asks the caller to
    // read the expert in. Refused for a slot a pending generation still reads, and below the floor.
    l2_relabel relabel_to_resident(int s, int slot, int layer, int expert) {
        l2_relabel out;
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role != slot_role::ring ||
                layer < 0 || layer >= layers_ || storage_of(layer) != s || expert < 0 || expert >= experts_) {
            out.reason = "not a ring slot of the layer's storage class";
            return out;
        }
        owner & o = owners_[(size_t) s][(size_t) slot];
        if (!reusable(o)) {
            out.reason = "ring slot " + std::to_string(slot) + " is still leased";
            return out;
        }
        if (ring_count_[(size_t) s] - 1 < floor_[(size_t) s]) {
            out.reason = "the ring of storage class " + std::to_string(s) + " would fall below its floor of " +
                std::to_string(floor_[(size_t) s]) + " slots";
            return out;
        }
        const bool holds = o.layer == layer && o.expert == expert;
        if (!holds) {
            if (o.layer >= 0) {
                out.evicted.push_back({o.layer, o.expert, slot});
                map_[(size_t) o.layer][(size_t) o.expert] = -1;
            }
            const int other = map_[(size_t) layer][(size_t) expert];
            if (other >= 0) {
                // the expert sits in another ring slot: that copy is no longer needed
                out.evicted.push_back({layer, expert, other});
                empty(s, other);
            }
        } else {
            map_[(size_t) layer][(size_t) expert] = -1;
        }
        unlink(s, slot);
        --ring_count_[(size_t) s];
        o = owner();
        o.role = slot_role::resident;
        o.layer = layer; o.expert = expert;
        out.ok = true;
        out.read = !holds;
        return out;
    }

    // A ring slot that becomes a spare resident (a resident slot without an expert); its occupant
    // is evicted. Refused like relabel_to_resident.
    l2_relabel relabel_to_spare(int s, int slot) {
        l2_relabel out;
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role != slot_role::ring) {
            out.reason = "not a ring slot";
            return out;
        }
        owner & o = owners_[(size_t) s][(size_t) slot];
        if (!reusable(o)) {
            out.reason = "ring slot " + std::to_string(slot) + " is still leased";
            return out;
        }
        if (ring_count_[(size_t) s] - 1 < floor_[(size_t) s]) {
            out.reason = "the ring of storage class " + std::to_string(s) + " would fall below its floor of " +
                std::to_string(floor_[(size_t) s]) + " slots";
            return out;
        }
        if (o.layer >= 0) {
            out.evicted.push_back({o.layer, o.expert, slot});
            map_[(size_t) o.layer][(size_t) o.expert] = -1;
        }
        unlink(s, slot);
        --ring_count_[(size_t) s];
        o = owner();
        o.role = slot_role::resident;
        out.ok = true;
        return out;
    }

    // A ring slot for a transient read (a file promotion on its way to VRAM): the least recently
    // used one that no generation leases, emptied and moved to the most recently used end, so the
    // slots taken one after another are distinct. Slots flagged in `busy` are skipped too. -1 when
    // no ring slot is left.
    int take_free(int s, std::vector<l2_eviction> & evicted, const std::vector<uint8_t> * busy = nullptr) {
        for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) {
            owner & o = owners_[(size_t) s][(size_t) slot];
            if (!reusable(o) || (busy != nullptr && (*busy)[(size_t) slot])) { continue; }
            if (o.layer >= 0) {
                evicted.push_back({o.layer, o.expert, slot});
                map_[(size_t) o.layer][(size_t) o.expert] = -1;
                o.layer = -1; o.expert = -1; o.seq = 0;
            }
            touch(s, slot);
            return slot;
        }
        return -1;
    }

    // Moves every empty ring slot to the least recently used end, in list order, so the next
    // slots taken evict nothing.
    void sink_empty() {
        for (int s = 0; s < storages(); ++s) {
            std::vector<int> empty_slots;
            for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) {
                if (owners_[(size_t) s][(size_t) slot].layer < 0) { empty_slots.push_back(slot); }
            }
            for (size_t i = empty_slots.size(); i-- > 0;) {
                unlink(s, empty_slots[i]);
                link_head(s, empty_slots[i]);
            }
        }
    }

    // Ring slots of a class that hold an expert.
    int ring_occupants(int s) const {
        int n = 0;
        for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) {
            n += owners_[(size_t) s][(size_t) slot].layer >= 0 ? 1 : 0;
        }
        return n;
    }

    // Empties every ring and lists each ring's slots in index order again.
    void discard() {
        for (std::vector<int32_t> & row : map_) {
            std::fill(row.begin(), row.end(), int32_t(-1));
        }
        for (size_t s = 0; s < owners_.size(); ++s) {
            head_[s] = tail_[s] = -1;
            for (size_t slot = 0; slot < owners_[s].size(); ++slot) {
                prev_[s][slot] = next_[s][slot] = -1;
                owner & o = owners_[s][slot];
                if (o.role != slot_role::ring) { continue; }
                o = owner();
                o.role = slot_role::ring;
                link_tail(int(s), int(slot));
            }
        }
    }

    // The demand of one layer, served from the ring of its storage class. `homed(layer, expert)` is
    // true for an expert with a VRAM or host home. The caller may pass duplicates.
    template <typename Homed>
    l2_service service(int layer, const std::vector<int> & ids, uint32_t seq, Homed homed) {
        l2_service out;
        const int s = layer >= 0 && layer < layers_ ? storage_of(layer) : -1;
        if (s < 0) {
            out.ok = false;
            out.reason = "layer " + std::to_string(layer) + " has no storage class";
            return out;
        }
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
        // Everything already in the ring is pinned before any slot is handed out, or a demand
        // larger than the free part of the ring would evict its own earlier experts.
        std::vector<bool> pinned((size_t) slots(s), false);
        for (int expert : wanted) {
            const int slot = map_[(size_t) layer][(size_t) expert];
            if (slot >= 0) { pinned[(size_t) slot] = true; }
        }
        for (int expert : wanted) {
            if (homed(layer, expert)) {
                continue;
            }
            int slot = map_[(size_t) layer][(size_t) expert];
            if (slot >= 0) {
                ++out.hits;
                owners_[(size_t) s][(size_t) slot].seq = seq;   // extend the lease to this generation
                touch(s, slot);
                continue;
            }
            slot = acquire(s, pinned);
            if (slot < 0) {
                out.ok = false;
                out.reason = "no reusable ring slot in storage class " + std::to_string(s) + " for layer " +
                    std::to_string(layer) + " expert " + std::to_string(expert);
                return out;
            }
            owner & o = owners_[(size_t) s][(size_t) slot];
            if (o.layer >= 0) {
                out.evicted.push_back({o.layer, o.expert, slot});
                map_[(size_t) o.layer][(size_t) o.expert] = -1;
            }
            o.layer = layer; o.expert = expert; o.seq = seq;
            map_[(size_t) layer][(size_t) expert] = slot;
            touch(s, slot);
            pinned[(size_t) slot] = true;
            ++out.misses;
            for (int kind = 0; kind < kinds_; ++kind) {
                out.reads.push_back({layer, kind, expert, slot});
            }
        }
        return out;
    }

    // Every invariant of the ledger; the reason names the first one broken.
    bool check(std::string & reason) const {
        for (size_t s = 0; s < owners_.size(); ++s) {
            int ring = 0, listed = 0;
            for (size_t slot = 0; slot < owners_[s].size(); ++slot) {
                const owner & o = owners_[s][slot];
                ring += o.role == slot_role::ring ? 1 : 0;
                if (o.role == slot_role::free && o.layer >= 0) { reason = "a free slot has an occupant"; return false; }
                if (o.layer >= 0 && (o.layer >= layers_ || storage_of(o.layer) != int(s))) {
                    reason = "an occupant of another storage class"; return false;
                }
                if (o.role == slot_role::ring && o.layer >= 0 && map_[(size_t) o.layer][(size_t) o.expert] != int(slot)) {
                    reason = "a ring occupant is missing from the map"; return false;
                }
            }
            int last = -1;
            for (int slot = head_[s]; slot >= 0; slot = next_[s][(size_t) slot]) {
                if (owners_[s][(size_t) slot].role != slot_role::ring || prev_[s][(size_t) slot] != last || ++listed > ring) {
                    reason = "the ring list is broken"; return false;
                }
                last = slot;
            }
            if (last != tail_[s] || listed != ring || ring != ring_count_[s]) { reason = "the ring count or the list end is wrong"; return false; }
        }
        for (int l = 0; l < layers_; ++l) {
            for (int e = 0; e < experts_; ++e) {
                const int slot = map_[(size_t) l][(size_t) e];
                if (slot >= 0 && !owns(slot, l, e)) { reason = "a map entry has no ring slot"; return false; }
            }
        }
        return true;
    }

private:
    struct owner {
        int       layer  = -1;
        int       expert = -1;
        uint32_t  seq    = 0;
        slot_role role   = slot_role::free;
    };

    bool valid_slot(int s, int slot) const {
        return s >= 0 && s < storages() && slot >= 0 && slot < slots(s);
    }

    // Wrap safe: a slot is free when the GPU has finished the generation that claimed it.
    bool reusable(const owner & o) const {
        return o.layer < 0 || int32_t(done_[(size_t) o.layer] - o.seq) >= 0;
    }

    int acquire(int s, const std::vector<bool> & pinned) const {
        for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) {
            if (!pinned[(size_t) slot] && reusable(owners_[(size_t) s][(size_t) slot])) {
                return slot;
            }
        }
        return -1;
    }

    // A ring slot loses its occupant and stays where it is in the list.
    void empty(int s, int slot) {
        owner & o = owners_[(size_t) s][(size_t) slot];
        if (o.layer >= 0 && map_[(size_t) o.layer][(size_t) o.expert] == slot) {
            map_[(size_t) o.layer][(size_t) o.expert] = -1;
        }
        o.layer = -1; o.expert = -1; o.seq = 0;
    }

    void unlink(int s, int slot) {
        auto & prev = prev_[(size_t) s];
        auto & next = next_[(size_t) s];
        const int p = prev[(size_t) slot], n = next[(size_t) slot];
        if (p >= 0) { next[(size_t) p] = n; } else { head_[(size_t) s] = n; }
        if (n >= 0) { prev[(size_t) n] = p; } else { tail_[(size_t) s] = p; }
        prev[(size_t) slot] = next[(size_t) slot] = -1;
    }

    void link_tail(int s, int slot) {
        auto & prev = prev_[(size_t) s];
        auto & next = next_[(size_t) s];
        const int t = tail_[(size_t) s];
        prev[(size_t) slot] = t;
        next[(size_t) slot] = -1;
        if (t >= 0) { next[(size_t) t] = slot; } else { head_[(size_t) s] = slot; }
        tail_[(size_t) s] = slot;
    }

    void link_head(int s, int slot) {
        auto & prev = prev_[(size_t) s];
        auto & next = next_[(size_t) s];
        const int h = head_[(size_t) s];
        prev[(size_t) slot] = -1;
        next[(size_t) slot] = h;
        if (h >= 0) { prev[(size_t) h] = slot; } else { tail_[(size_t) s] = slot; }
        head_[(size_t) s] = slot;
    }

    // Moves a ring slot to the most recently used end.
    void touch(int s, int slot) {
        if (slot != tail_[(size_t) s]) {
            unlink(s, slot);
            link_tail(s, slot);
        }
    }

    int layers_  = 0;
    int experts_ = 0;
    int kinds_   = 3;
    std::vector<int> layer_storage_;
    std::vector<std::vector<owner>> owners_;   // [storage class][slot]
    std::vector<std::vector<int>> prev_, next_;
    std::vector<int> head_, tail_;             // least / most recently used ring slot
    std::vector<int> ring_count_, floor_;
    std::vector<uint32_t> done_;
    std::vector<std::vector<int32_t>> map_;    // [layer][expert] ring slot or -1
};

} // namespace ggml_cuda_expert
