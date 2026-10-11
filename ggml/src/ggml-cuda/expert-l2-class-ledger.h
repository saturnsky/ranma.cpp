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
//
// The unified policy (set_unified, off by default): every slot of a class is a ring slot once the
// residents of the seed plan are dissolved into the rings (dissolve), and each class list is split
// into a probation part at the least recently used end, the larger of a fixed share of the class's
// ring slots (empty slots included) and a per-class minimum, and a main part above it. A hit moves its slot to the most recently used
// end. A miss of an expert that also missed in its layer within the last K decode generations (the
// ghost) is filled at the most recently used end, any other miss at the bottom of the main part, so
// that exactly the probation share of the list lies below it; a prompt generation fills at the least
// recently used end and moves no hit, unless the policy says prompts follow the decode rules. Each
// operation keeps the split with O(1) moves. Off, none of this state exists and every path is the
// one above.

#include "expert-l2-ledger.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ggml_cuda_expert {

enum class slot_role : uint8_t { free, resident, ring };

// Where a slot goes in its class list: the most recently used end, the probation point (the bottom of
// the main part) or the least recently used end.
enum class l2_place : uint8_t { mru, probation, lru };

// The unified host policy (RANMA_EXPERT_L2_UNIFIED).
struct l2_unified_policy {
    bool     on         = false;
    double   probation  = 0.10;   // share of each class list below the probation point, 0..0.5
    uint32_t ghost      = 32;     // a miss within this many decode generations of its layer passes the ghost
    bool     prompt_lru = true;   // prompt generations follow the decode rules (default); false: misses at the LRU end, hits stay
};

// Whether the decode pool admits the SSD tier's ring occupants (RANMA_EXPERT_UB_POOL_RING, `ring_env`
// -1 when not set). Under the unified policy every host expert is a ring occupant, so the pool takes
// host experts only with ring admission: it is on unless the switch says 0.
inline bool l2_unified_pool_ring(int ring_env, bool unified) {
    return ring_env < 0 ? unified : ring_env != 0;
}

// Counts of the unified policy. Probation fills that were evicted before any hit are the immediate
// evictions; probation hits are the probation fills that hit before their eviction.
struct l2_unified_stats {
    uint64_t hits = 0, fills = 0, mru_fills = 0, probation_fills = 0, prompt_fills = 0;
    uint64_t probation_hits = 0, probation_evicted_unhit = 0, evictions = 0;
    uint64_t sunk = 0, adopted = 0, dissolved = 0;
};

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
        if (uni_.on) { unified_init(); }
    }

    // ---- the unified policy ----------------------------------------------------------------------
    // Turns the policy on (or off) for the lists as they are: every listed slot starts in the main part,
    // then each class's probation share is taken from its least recently used end.
    void set_unified(const l2_unified_policy & policy) {
        uni_ = policy;
        uni_.probation = std::min(std::max(uni_.probation, 0.0), 0.5);
        if (!uni_.on) {
            in_p_.clear(); unhit_.clear(); pcount_.clear(); pmin_.clear(); bnd_.clear(); last_miss_.clear(); step_.clear();
            return;
        }
        unified_init();
    }
    bool unified() const { return uni_.on; }
    const l2_unified_policy & unified_policy() const { return uni_; }
    const l2_unified_stats & unified_stats() const { return ustats_; }
    // The phase of the generations served from now on (unified policy only).
    void set_prompt(bool prompt) { prompt_ = prompt; }
    // Slots in the probation part of a class (unified policy only).
    int probation_count(int s) const { return uni_.on ? pcount_[(size_t) s] : 0; }
    // The least number of slots in the probation part of a class (unified policy only; the share
    // applies when it is larger, the class's ring slots when they are fewer).
    void set_probation_min(int s, int slots) {
        if (!uni_.on || s < 0 || s >= storages()) { return; }
        pmin_[(size_t) s] = std::max(slots, 0);
        rebalance(s);
    }

    // Every resident slot joins the ring of its class in place: one with an expert at `where`, an
    // empty one (a spare) at the least recently used end. Only while nothing computes. Returns the
    // residents with an expert that joined. `only` (per storage class, null = all) restricts it.
    int residents_to_ring(l2_place where, const std::vector<uint8_t> * only = nullptr) {
        int joined = 0;
        for (int s = 0; s < storages(); ++s) {
            if (only != nullptr && !(*only)[(size_t) s]) { continue; }
            for (int slot = 0; slot < slots(s); ++slot) {
                owner & o = owners_[(size_t) s][(size_t) slot];
                if (o.role != slot_role::resident) { continue; }
                o.role = slot_role::ring;
                ++ring_count_[(size_t) s];
                link_head(s, slot);
                if (o.layer >= 0) {
                    ++joined;
                    const int other = map_[(size_t) o.layer][(size_t) o.expert];
                    if (other >= 0 && other != slot) { empty(s, other); }
                    map_[(size_t) o.layer][(size_t) o.expert] = slot;
                    o.seq = done_[(size_t) o.layer];   // nothing computes
                    place(s, slot, where);
                }
                if (uni_.on) { rebalance(s); }
            }
        }
        ustats_.dissolved += (uint64_t) joined;
        return joined;
    }

    // Every resident slot joins the ring of its class in place, with its expert if it has one, and
    // each class list is laid out again: empty slots at the least recently used end, then the
    // occupants from the lowest to the highest `score(layer, expert)`, ties in slot order. Only while
    // nothing computes (every occupant is reusable at once). Returns the residents that joined.
    // `only` (per storage class, null = all) restricts it to those classes; the others keep their lists.
    template <typename Score>
    int dissolve(Score score, const std::vector<uint8_t> * only = nullptr) {
        int joined = 0;
        for (int s = 0; s < storages(); ++s) {
            if (only != nullptr && !(*only)[(size_t) s]) { continue; }
            std::vector<owner> & row = owners_[(size_t) s];
            for (size_t slot = 0; slot < row.size(); ++slot) {
                owner & o = row[slot];
                if (o.role != slot_role::resident) { continue; }
                o.role = slot_role::ring;
                ++ring_count_[(size_t) s];
                ++joined;
                if (o.layer >= 0) {
                    const int other = map_[(size_t) o.layer][(size_t) o.expert];
                    if (other >= 0 && other != int(slot)) { empty(s, other); }
                    map_[(size_t) o.layer][(size_t) o.expert] = int(slot);
                }
            }
            std::vector<int> empties;
            std::vector<std::pair<double, int>> held;
            for (size_t slot = 0; slot < row.size(); ++slot) {
                owner & o = row[slot];
                if (o.role != slot_role::ring) { continue; }
                if (o.layer < 0) {
                    empties.push_back(int(slot));
                } else {
                    o.seq = done_[(size_t) o.layer];   // nothing computes
                    held.push_back({double(score(o.layer, o.expert)), int(slot)});
                }
            }
            std::stable_sort(held.begin(), held.end(),
                [](const std::pair<double, int> & a, const std::pair<double, int> & b) { return a.first < b.first; });
            head_[(size_t) s] = tail_[(size_t) s] = -1;
            for (size_t slot = 0; slot < row.size(); ++slot) { prev_[(size_t) s][slot] = next_[(size_t) s][slot] = -1; }
            if (uni_.on) { unified_clear_lists(s); }
            for (int slot : empties) { link_tail(s, slot); }
            for (const auto & item : held) { link_tail(s, item.second); }
            if (uni_.on) { rebalance(s); }
        }
        ustats_.dissolved += (uint64_t) joined;
        return joined;
    }

    // A ring slot that loses its occupant (e.g. promoted to VRAM by an install) goes to the least
    // recently used end. Refused for a slot a pending generation still reads.
    bool sink(int s, int slot) {
        if (!valid_slot(s, slot) || owners_[(size_t) s][(size_t) slot].role != slot_role::ring ||
                !reusable(owners_[(size_t) s][(size_t) slot])) {
            return false;
        }
        empty(s, slot);
        unlink(s, slot);
        link_head(s, slot);
        if (uni_.on) { rebalance(s); }
        ++ustats_.sunk;
        return true;
    }

    // An empty ring slot (e.g. from take_free) takes (layer, expert), whose bytes the caller put there
    // without a read (a demotion from VRAM), at `where` in its class list. Refused when the slot is not an
    // empty ring slot of the layer's class or the expert already has a ring slot.
    bool adopt(int s, int slot, int layer, int expert, l2_place where) {
        if (!valid_slot(s, slot) || layer < 0 || layer >= layers_ || storage_of(layer) != s || expert < 0 ||
                expert >= experts_ || map_[(size_t) layer][(size_t) expert] >= 0) {
            return false;
        }
        owner & o = owners_[(size_t) s][(size_t) slot];
        if (o.role != slot_role::ring || o.layer >= 0) { return false; }
        o.layer = layer; o.expert = expert; o.seq = done_[(size_t) layer];
        map_[(size_t) layer][(size_t) expert] = slot;
        place(s, slot, where);
        ++ustats_.adopted;
        return true;
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
        if (uni_.on) { rebalance(s); }
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
        if (uni_.on) { rebalance(s); }
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
        if (uni_.on) { unhit_[(size_t) s][(size_t) slot] = 0; rebalance(s); }
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
        if (uni_.on) { unhit_[(size_t) s][(size_t) slot] = 0; rebalance(s); }
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
            if (uni_.on) { unhit_[(size_t) s][(size_t) slot] = 0; rebalance(s); }
            return slot;
        }
        return -1;
    }

    // Moves every empty ring slot to the least recently used end, in list order, so the next
    // slots taken evict nothing.
    // `only` (per storage class, null = all) restricts it to those classes.
    void sink_empty(const std::vector<uint8_t> * only = nullptr) {
        for (int s = 0; s < storages(); ++s) {
            if (only != nullptr && !(*only)[(size_t) s]) { continue; }
            std::vector<int> empty_slots;
            for (int slot = head_[(size_t) s]; slot >= 0; slot = next_[(size_t) s][(size_t) slot]) {
                if (owners_[(size_t) s][(size_t) slot].layer < 0) { empty_slots.push_back(slot); }
            }
            for (size_t i = empty_slots.size(); i-- > 0;) {
                unlink(s, empty_slots[i]);
                link_head(s, empty_slots[i]);
            }
            if (uni_.on) { rebalance(s); }
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
            if (uni_.on) { unified_clear_lists(int(s)); }
            for (size_t slot = 0; slot < owners_[s].size(); ++slot) {
                prev_[s][slot] = next_[s][slot] = -1;
                owner & o = owners_[s][slot];
                if (o.role != slot_role::ring) { continue; }
                o = owner();
                o.role = slot_role::ring;
                link_tail(int(s), int(slot));
            }
            if (uni_.on) { rebalance(int(s)); }
        }
    }

    // The demand of one layer, served from the ring of its storage class. `homed(layer, expert)` is
    // true for an expert with a VRAM or host home. The caller may pass duplicates.
    // `quiet(layer, expert)`: a hit on that expert was filled by an early read for this demand; it
    // counts in early_hits instead of hits. `early`: the early service (expert-l2-ledger.h,
    // l2_no_quiet): hits keep their lease, misses take `seq`, the layer's done counter.
    template <typename Homed, typename Quiet = l2_no_quiet>
    l2_service service(int layer, const std::vector<int> & ids, uint32_t seq, Homed homed, Quiet quiet = Quiet(),
            bool early = false) {
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
        // The unified policy: a prompt generation under the ring rule moves no hit and fills at the least
        // recently used end; a decode generation (not an early service) is one ghost step of its layer.
        const bool uni = uni_.on;
        const bool ring_rule = uni && prompt_ && !uni_.prompt_lru;
        const bool ghost_step = uni && !ring_rule && !early;
        const uint32_t step = ghost_step ? ++step_[(size_t) layer] : (uni ? step_[(size_t) layer] : 0u);
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
                ++(quiet(layer, expert) ? out.early_hits : out.hits);
                if (!early) {
                    owners_[(size_t) s][(size_t) slot].seq = seq;   // extend the lease to this generation
                }
                if (uni) {
                    ++ustats_.hits;
                    uint8_t & unhit = unhit_[(size_t) s][(size_t) slot];
                    if (unhit && !ring_rule) { ++ustats_.probation_hits; unhit = 0; }
                    if (!ring_rule) { touch(s, slot); }
                } else {
                    touch(s, slot);
                }
                continue;
            }
            slot = acquire(s, pinned);
            if (slot < 0) {
                if (uni) { rebalance(s); }
                out.ok = false;
                out.reason = "no reusable ring slot in storage class " + std::to_string(s) + " for layer " +
                    std::to_string(layer) + " expert " + std::to_string(expert);
                return out;
            }
            owner & o = owners_[(size_t) s][(size_t) slot];
            if (o.layer >= 0) {
                out.evicted.push_back({o.layer, o.expert, slot});
                map_[(size_t) o.layer][(size_t) o.expert] = -1;
                if (uni) {
                    ++ustats_.evictions;
                    ustats_.probation_evicted_unhit += unhit_[(size_t) s][(size_t) slot];
                }
            }
            o.layer = layer; o.expert = expert; o.seq = seq;
            map_[(size_t) layer][(size_t) expert] = slot;
            if (uni) {
                const l2_place where = ring_rule ? l2_place::lru : ghost_ok(layer, expert, step) ? l2_place::mru : l2_place::probation;
                ++ustats_.fills;
                ++(where == l2_place::lru ? ustats_.prompt_fills : where == l2_place::mru ? ustats_.mru_fills : ustats_.probation_fills);
                place(s, slot, where);
                unhit_[(size_t) s][(size_t) slot] = where == l2_place::probation ? 1 : 0;
            } else {
                touch(s, slot);
            }
            pinned[(size_t) slot] = true;
            ++out.misses;
            for (int kind = 0; kind < kinds_; ++kind) {
                out.reads.push_back({layer, kind, expert, slot});
            }
        }
        if (uni) {
            // stamped after every decision of the step, so that a miss never passes its own ghost
            if (ghost_step) {
                for (int expert : wanted) {
                    if (!homed(layer, expert)) { last_miss_[(size_t) layer*(size_t) experts_ + (size_t) expert] = step; }
                }
            }
            rebalance(s);
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
            if (uni_.on) {
                // the probation part is the least recently used end of the list, of the policy's share
                int in_p = 0, first_main = -1;
                bool main_seen = false;
                for (int slot = head_[s]; slot >= 0; slot = next_[s][(size_t) slot]) {
                    if (in_p_[s][(size_t) slot]) {
                        if (main_seen) { reason = "a probation slot above the main part"; return false; }
                        ++in_p;
                    } else if (!main_seen) {
                        main_seen = true;
                        first_main = slot;
                    }
                }
                for (size_t slot = 0; slot < owners_[s].size(); ++slot) {
                    if (owners_[s][slot].role != slot_role::ring && in_p_[s][slot]) { reason = "a probation mark off the ring"; return false; }
                    if (owners_[s][slot].layer < 0 && unhit_[s][slot]) { reason = "an empty slot marked as a probation fill"; return false; }
                }
                if (in_p != pcount_[s] || first_main != bnd_[s] || in_p != probation_target(int(s))) {
                    reason = "the probation part does not match its share"; return false;
                }
            }
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
    // ---- the unified policy --------------------------------------------------------------------------
    int probation_target(int s) const {
        const int n = ring_count_[(size_t) s];
        return std::min(n, std::max(int(uni_.probation*double(n) + 1e-9), pmin_[(size_t) s]));
    }

    bool ghost_ok(int layer, int expert, uint32_t step) const {
        const uint32_t last = last_miss_[(size_t) layer*(size_t) experts_ + (size_t) expert];
        const int32_t age = int32_t(step - last);
        return last != 0 && age > 0 && uint32_t(age) <= uni_.ghost;
    }

    void unified_clear_lists(int s) {
        std::fill(in_p_[(size_t) s].begin(), in_p_[(size_t) s].end(), uint8_t(0));
        std::fill(unhit_[(size_t) s].begin(), unhit_[(size_t) s].end(), uint8_t(0));
        pcount_[(size_t) s] = 0;
        bnd_[(size_t) s] = -1;
    }

    // Sizes the policy state for the current lists and splits each of them.
    void unified_init() {
        const size_t n = owners_.size();
        in_p_.assign(n, {});
        unhit_.assign(n, {});
        pcount_.assign(n, 0);
        pmin_.assign(n, 0);
        bnd_.assign(n, -1);
        for (size_t s = 0; s < n; ++s) {
            in_p_[s].assign(owners_[s].size(), 0);
            unhit_[s].assign(owners_[s].size(), 0);
            bnd_[s] = head_[s];
            rebalance(int(s));
        }
        last_miss_.assign((size_t) std::max(layers_, 0)*(size_t) std::max(experts_, 0), 0u);
        step_.assign((size_t) std::max(layers_, 0), 0u);
    }

    // Restores the probation share of a class: the bottom of the main part moves down, or the top of
    // the probation part moves up, one slot at a time (an operation changes the share by at most two).
    void rebalance(int s) {
        const int target = probation_target(s);
        int & count = pcount_[(size_t) s];
        int & bnd   = bnd_[(size_t) s];
        while (count < target && bnd >= 0) {
            in_p_[(size_t) s][(size_t) bnd] = 1;
            ++count;
            bnd = next_[(size_t) s][(size_t) bnd];
        }
        while (count > target) {
            const int last = bnd >= 0 ? prev_[(size_t) s][(size_t) bnd] : tail_[(size_t) s];
            in_p_[(size_t) s][(size_t) last] = 0;
            --count;
            bnd = last;
        }
    }

    // Moves a listed slot to `where`. Off, the probation point is the most recently used end.
    void place(int s, int slot, l2_place where) {
        unlink(s, slot);
        if (where == l2_place::lru) {
            link_head(s, slot);
        } else if (where == l2_place::probation && uni_.on) {
            rebalance(s);   // exactly the probation share lies below the slot
            const int bnd = bnd_[(size_t) s];
            if (bnd < 0) {
                link_tail(s, slot);
            } else {
                auto & prev = prev_[(size_t) s];
                auto & next = next_[(size_t) s];
                const int p = prev[(size_t) bnd];
                prev[(size_t) slot] = p;
                next[(size_t) slot] = bnd;
                prev[(size_t) bnd] = slot;
                if (p >= 0) { next[(size_t) p] = slot; } else { head_[(size_t) s] = slot; }
                in_p_[(size_t) s][(size_t) slot] = 0;
                bnd_[(size_t) s] = slot;
            }
        } else {
            link_tail(s, slot);
        }
        if (uni_.on) { rebalance(s); }
    }

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
        if (uni_.on) { unhit_[(size_t) s][(size_t) slot] = 0; }
    }

    void unlink(int s, int slot) {
        auto & prev = prev_[(size_t) s];
        auto & next = next_[(size_t) s];
        const int p = prev[(size_t) slot], n = next[(size_t) slot];
        if (uni_.on) {
            if (in_p_[(size_t) s][(size_t) slot]) {
                in_p_[(size_t) s][(size_t) slot] = 0;
                --pcount_[(size_t) s];
            } else if (bnd_[(size_t) s] == slot) {
                bnd_[(size_t) s] = n;
            }
        }
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
        if (uni_.on) {
            in_p_[(size_t) s][(size_t) slot] = 0;   // the most recently used end is in the main part
            if (bnd_[(size_t) s] < 0) { bnd_[(size_t) s] = slot; }
        }
    }

    void link_head(int s, int slot) {
        auto & prev = prev_[(size_t) s];
        auto & next = next_[(size_t) s];
        const int h = head_[(size_t) s];
        prev[(size_t) slot] = -1;
        next[(size_t) slot] = h;
        if (h >= 0) { prev[(size_t) h] = slot; } else { tail_[(size_t) s] = slot; }
        head_[(size_t) s] = slot;
        if (uni_.on) {
            in_p_[(size_t) s][(size_t) slot] = 1;   // the least recently used end is in the probation part
            ++pcount_[(size_t) s];
        }
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
    // the unified policy (empty while it is off)
    l2_unified_policy uni_;
    l2_unified_stats  ustats_;
    bool prompt_ = false;
    std::vector<std::vector<uint8_t>> in_p_;    // [storage class][slot] in the probation part
    std::vector<std::vector<uint8_t>> unhit_;   // [storage class][slot] a probation fill not hit yet
    std::vector<int> pcount_, bnd_;             // probation slots; the least recently used slot of the main part or -1
    std::vector<int> pmin_;                     // [storage class] the least number of probation slots
    std::vector<uint32_t> last_miss_;           // [layer*experts + expert] ghost step of the last miss, 0 = none
    std::vector<uint32_t> step_;                // [layer] decode generations served
};

} // namespace ggml_cuda_expert
