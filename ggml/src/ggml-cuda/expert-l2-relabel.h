#pragma once

// The install transaction of the class layout of the SSD tier. Pure CPU code: no CUDA/HIP headers,
// no I/O.
//
// A host resident of the plan lives in a logical slot of its size class (expert_location::slot); the
// tier maps every logical slot to one slot of the storage class arena (`home_slot`). An install keeps
// that map a bijection onto the resident slots of the arena and changes it instead of moving bytes:
// - a resident that leaves for the file stays in its slot, which joins the ring with the expert as a
//   cold occupant (nothing is copied, a later demand or promotion finds it there);
// - a file promotion that the ring already holds takes that ring slot as its home (no read);
// - any other promotion takes an empty ring slot, or else the least recently used one, and is read
//   straight into it; a promotion to VRAM is copied from a ring occupant when there is one.
// Every logical slot that changes gives its old storage slot to the ring and takes one from it, so
// each ring keeps its size. Slots a pending generation leases are never taken or relabeled.
//
// Moves from the file run in batches: all reads of a batch go out together, then the batch's copies to
// VRAM run, and the next batch starts after they complete. No slot is read into, or copied from, twice
// in one batch.
//
// The unified policy (l2_unified_policy): the plan has no host residents but the demotions from VRAM
// of this install, every slot is a ring slot, and a logical host slot without a home stays without
// one. After the moves, settle_unified turns the demotions into ring occupants at the probation point
// and empties the ring copies of the VRAM promotions; plan_unified_reseed then brings each class's
// ring to the best experts of a profile and lays it out in score order.
//
// A dropped demotion (l2_demotion_policy, unified classes only): a demotion from VRAM that is not copied at all. The
// expert keeps no ring slot and no home; its source is the file, which the tier reads when a demand needs it.
//
// The one-pass demotion (l2_demotion_policy::one_pass, unified classes only): a demotion goes straight to its final
// place. It keeps its copy (one D2H) in an empty ring slot, else in the ring slot of an expert an earlier move of this
// transaction promoted to VRAM (that copy is no longer needed and its read was queued first), else in the slot of the
// occupant of the lowest value, which it evicts. When the ring has neither an empty nor a released slot and the
// demoted expert is worth less than every occupant, it is dropped. An occupant is never evicted while an empty or
// released slot is left. `keep_all` (an L2 of -1, sized to hold every expert) never drops.
//
// A storage class can stay out of the policy for a while (`active`, per storage class, null = all): a
// joint cache whose later model has not bound yet keeps that model's classes as residents and rings, and
// they join the policy when it has (l2_tier::unify_storages). Every function below leaves the inactive
// classes exactly as they are.

#include "expert-l2-class-ledger.h"
#include "expert-plan.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// What the mover does for one move of the transaction.
struct l2_install_step {
    int  src     = -1;     // storage slot a copy reads from: the home of a host source, or a ring slot for a promotion to VRAM
    int  dst     = -1;     // storage slot that becomes the home of a host destination
    bool read    = false;  // read the file first: into `dst` for a host destination, into `src` for a VRAM one
    bool relabel = false;  // host destination whose ring slot already held the expert: nothing moves
    bool drop    = false;  // a demotion from VRAM that is not copied: the expert stays in the file only
    int  batch   = -1;     // moves from the file: the read batch
};

// What happens to the demotions from VRAM of a unified class.
struct l2_demotion_policy {
    std::function<bool(int, int)> force_drop;    // (layer, expert): drop this demotion (diagnostic)
    bool one_pass = false;                        // the one-pass place (above); off: the least recently used ring slot
    bool keep_all = false;                        // never drop (an L2 of -1)
    std::function<double(int, int)> value;       // (layer, expert): the profile value of keeping it; null: all equal
};

struct l2_install_plan {
    bool ok = false;
    std::string reason;
    std::vector<l2_install_step> steps;   // one per move, in move order
    std::vector<l2_eviction> evicted;     // ring occupants that lost their slot
    size_t relabeled   = 0;   // host promotions from a ring occupant
    size_t ring_copies = 0;   // VRAM promotions copied from a ring occupant
    size_t reads       = 0;   // moves read from the file
    size_t demoted     = 0;   // residents that stayed in their slot as ring occupants
    int    batches     = 0;
    size_t dropped     = 0;   // demotions from VRAM not copied (l2_install_step::drop)
    // one-pass demotions kept in an empty slot, in a released slot, over an evicted occupant of a lower value; and
    // evictions made while an empty or released slot was left (0 by the rule)
    size_t kept_empty = 0, kept_released = 0, kept_evicting = 0, needless = 0;
};

// `home_slot` [size class][logical slot] is the storage slot of every logical host slot; `before` and
// `after` are the host locations of the transaction's two plans and `moves` its moves. Updates the
// ledger and `home_slot`. Nothing may compute meanwhile.
inline l2_install_plan plan_relabel_install(l2_class_ledger & ledger, std::vector<std::vector<int>> & home_slot,
        const std::vector<int> & layer_class, const std::vector<int> & storage_of,
        const expert_locations & before, const expert_locations & after, const std::vector<install_move> & moves,
        bool unified = false, const std::vector<uint8_t> * active = nullptr, const l2_demotion_policy * demote = nullptr) {
    l2_install_plan out;
    // whether the unified policy applies to a size class / a storage class
    auto uni_s = [&](int s) { return unified && (active == nullptr || (*active)[(size_t) s] != 0); };
    auto uni_c = [&](int c) { return uni_s(storage_of[(size_t) c]); };
    auto fail = [&](const std::string & why) { out.ok = false; out.reason = why; return out; };
    const int classes = (int) home_slot.size();
    const int layers  = (int) layer_class.size();
    if ((int) storage_of.size() != classes || (int) before.size() != layers || (int) after.size() != layers) {
        return fail("relabel install dimensions");
    }
    const int experts = layers > 0 ? (int) before[0].size() : 0;
    std::vector<int> ring_before((size_t) ledger.storages());
    for (int s = 0; s < ledger.storages(); ++s) { ring_before[(size_t) s] = ledger.ring_count(s); }

    // the expert (layer*experts + expert) of every logical slot before and after
    std::vector<std::vector<int>> old_of((size_t) classes), new_of((size_t) classes);
    for (int c = 0; c < classes; ++c) {
        old_of[(size_t) c].assign(home_slot[(size_t) c].size(), -1);
        new_of[(size_t) c].assign(home_slot[(size_t) c].size(), -1);
    }
    for (int l = 0; l < layers; ++l) {
        const int c = layer_class[(size_t) l];
        if ((int) before[(size_t) l].size() != experts || (int) after[(size_t) l].size() != experts) { return fail("relabel install dimensions"); }
        for (int e = 0; e < experts; ++e) {
            for (int side = 0; side < 2; ++side) {
                const expert_location at = (side == 0 ? before : after)[(size_t) l][(size_t) e];
                if (at.storage == expert_storage::lent) { return fail("the class layout has no lent slots"); }
                if (at.storage != expert_storage::host) { continue; }
                if (c < 0 || c >= classes || at.slot < 0 || at.slot >= (int) home_slot[(size_t) c].size()) {
                    return fail("host location out of range");
                }
                (side == 0 ? old_of : new_of)[(size_t) c][(size_t) at.slot] = l*experts + e;
            }
        }
    }
    for (const install_move & m : moves) {
        if (m.from.storage == expert_storage::lent || m.to.storage == expert_storage::lent ||
                (m.from.storage == expert_storage::host && m.to.storage != expert_storage::vram) ||
                (m.from.storage != expert_storage::host && m.to.storage == m.from.storage)) {
            return fail("a move the class layout does not make");
        }
        if (m.cls < 0 || m.cls >= classes || m.layer < 0 || m.layer >= layers || layer_class[(size_t) m.layer] != m.cls) {
            return fail("a move of an unknown size class");
        }
    }
    // The old expert of the slot leaves through a move (to VRAM). A host master that stays is a source
    // too, possibly of a slot it only got in this transaction.
    std::vector<std::vector<uint8_t>> leaves((size_t) classes);
    for (int c = 0; c < classes; ++c) { leaves[(size_t) c].assign(home_slot[(size_t) c].size(), 0); }
    for (const install_move & m : moves) {
        if (m.from.storage == expert_storage::host) {
            const int key = m.layer*experts + m.expert;
            if (m.from.slot < 0 || m.from.slot >= (int) home_slot[(size_t) m.cls].size() ||
                    (old_of[(size_t) m.cls][(size_t) m.from.slot] != key && new_of[(size_t) m.cls][(size_t) m.from.slot] != key)) {
                return fail("a host source that does not hold the moved expert");
            }
            if (new_of[(size_t) m.cls][(size_t) m.from.slot] != key) { leaves[(size_t) m.cls][(size_t) m.from.slot] = 1; }
        }
    }

    // Residents that leave for the file become cold ring occupants in place; empty host slots join
    // the ring empty.
    for (int pass = 0; pass < 2; ++pass) {
        for (int c = 0; c < classes; ++c) {
            const int s = storage_of[(size_t) c];
            for (size_t i = 0; i < home_slot[(size_t) c].size(); ++i) {
                const int was = old_of[(size_t) c][i];
                if (was == new_of[(size_t) c][i] || leaves[(size_t) c][i] || (pass == 0) != (was >= 0)) { continue; }
                if (uni_c(c) && home_slot[(size_t) c][i] < 0) { continue; }   // a logical slot without a home: nothing joins
                if (was >= 0) {
                    const expert_location to = after[(size_t) (was/experts)][(size_t) (was%experts)];
                    if (to.storage != expert_storage::file) { return fail("a resident changes its host slot without a move"); }
                }
                if (!ledger.relabel_to_ring(s, home_slot[(size_t) c][i], true, was >= 0)) {
                    return fail("a home slot is not a resident slot of its storage class");
                }
                out.demoted += was >= 0 ? 1 : 0;
                home_slot[(size_t) c][i] = -1;
            }
        }
    }

    // Empty ring slots go first: taking them evicts nothing.
    ledger.sink_empty();

    std::vector<std::vector<uint8_t>> busy((size_t) ledger.storages());
    for (int s = 0; s < ledger.storages(); ++s) { busy[(size_t) s].assign((size_t) ledger.slots(s), 0); }
    int batch = -1;
    bool in_run = false;
    auto new_batch = [&]() {
        ++batch;
        for (auto & row : busy) { std::fill(row.begin(), row.end(), uint8_t(0)); }
    };
    // a ring slot for a read or a D2H copy; inside a batch never one the batch already uses
    auto take = [&](int s) {
        int slot = ledger.take_free(s, out.evicted, in_run ? &busy[(size_t) s] : nullptr);
        if (slot < 0 && in_run) {
            new_batch();
            slot = ledger.take_free(s, out.evicted, &busy[(size_t) s]);
        }
        return slot;
    };
    auto home = [&](int s, int slot, int layer, int expert) {
        const l2_relabel r = ledger.relabel_to_resident(s, slot, layer, expert);
        out.evicted.insert(out.evicted.end(), r.evicted.begin(), r.evicted.end());
        return r;
    };
    std::vector<std::vector<uint8_t>> dropped((size_t) classes);
    for (int c = 0; c < classes; ++c) { dropped[(size_t) c].assign(home_slot[(size_t) c].size(), 0); }
    // the ring slots whose copy an earlier move promoted to VRAM: [storage class] (slot, layer, expert)
    std::vector<std::vector<std::array<int, 3>>> released((size_t) ledger.storages());
    auto value = [&](int l, int e) { return demote != nullptr && demote->value ? demote->value(l, e) : 0.0; };
    // the one-pass place of a demotion of (layer, expert) in storage class s: a ring slot, or -1 to drop it
    auto place_demotion = [&](int s, int layer, int expert) -> int {
        int empty = -1, low = -1;
        double low_v = 0.0;
        for (int slot : ledger.order(s)) {
            int l = -1, e = -1;
            if (!ledger.reusable_slot(s, slot)) { continue; }
            if (!ledger.occupant(s, slot, l, e)) {
                if (empty < 0) { empty = slot; }
                continue;
            }
            const double v = value(l, e);
            if (low < 0 || v < low_v) { low = slot; low_v = v; }
        }
        if (empty >= 0) {
            ++out.kept_empty;
            return empty;
        }
        auto & rel = released[(size_t) s];
        while (!rel.empty()) {
            const std::array<int, 3> r = rel.back();
            rel.pop_back();
            if (ledger.owns(r[0], r[1], r[2]) && ledger.reusable_slot(s, r[0])) {
                ++out.kept_released;
                return r[0];
            }
        }
        if (low < 0 || (!demote->keep_all && value(layer, expert) < low_v)) {
            return -1;
        }
        ++out.kept_evicting;
        out.needless += empty >= 0 || !rel.empty() ? 1 : 0;
        return low;
    };
    out.steps.resize(moves.size());
    for (size_t k = 0; k < moves.size(); ++k) {
        const install_move & m = moves[k];
        l2_install_step & step = out.steps[k];
        const int c = m.cls, s = storage_of[(size_t) c];
        const int key = m.layer*experts + m.expert;
        const bool from_file = m.from.storage == expert_storage::file;
        if (from_file && !in_run) { in_run = true; new_batch(); }
        if (!from_file) { in_run = false; }
        if (m.to.storage == expert_storage::host) {
            const int i = m.to.slot;
            if (i < 0 || i >= (int) home_slot[(size_t) c].size() || new_of[(size_t) c][(size_t) i] != key ||
                    home_slot[(size_t) c][(size_t) i] >= 0) {
                return fail("a host destination that is not free for the moved expert");
            }
        }
        if (m.from.storage == expert_storage::host) {
            // to VRAM; the home stays when the expert keeps it (a host master)
            const int i = m.from.slot;
            step.src = home_slot[(size_t) c][(size_t) i];
            if (step.src < 0) { return fail("a host source without a home"); }
            if (new_of[(size_t) c][(size_t) i] != key) {
                // A later write into the slot runs after this copy: on the same stream, or after the
                // synchronization before a read batch.
                if (!ledger.relabel_to_ring(s, step.src, true, false)) { return fail("a host source is not a resident slot"); }
                home_slot[(size_t) c][(size_t) i] = -1;
            }
        } else if (m.from.storage == expert_storage::vram && uni_c(c) && demote != nullptr && demote->force_drop &&
                demote->force_drop(m.layer, m.expert)) {
            // dropped: no copy, no home; the file serves it
            step.drop = true;
            dropped[(size_t) c][(size_t) m.to.slot] = 1;
            ++out.dropped;
        } else if (m.from.storage == expert_storage::vram && uni_c(c) && demote != nullptr && demote->one_pass) {
            // to host, in one pass: its final ring slot, or the file
            const int t = place_demotion(s, m.layer, m.expert);
            if (t < 0) {
                if (demote->keep_all) {
                    return fail("no ring slot of storage class " + std::to_string(s) + " for a demotion an L2 of -1 keeps");
                }
                step.drop = true;
                dropped[(size_t) c][(size_t) m.to.slot] = 1;
                ++out.dropped;
                continue;
            }
            const l2_relabel r = home(s, t, m.layer, m.expert);
            if (!r.ok) { return fail(r.reason); }
            step.dst = t;
            home_slot[(size_t) c][(size_t) m.to.slot] = t;
        } else if (m.from.storage == expert_storage::vram) {
            // to host: copied back into a ring slot that becomes the home
            const int t = take(s);
            if (t < 0) { return fail("no ring slot of storage class " + std::to_string(s) + " for a demotion from VRAM"); }
            const l2_relabel r = home(s, t, m.layer, m.expert);
            if (!r.ok) { return fail(r.reason); }
            step.dst = t;
            home_slot[(size_t) c][(size_t) m.to.slot] = t;
        } else {
            step.batch = batch;
            const int q = ledger.slot_of(m.layer, m.expert);
            const bool held = q >= 0 && ledger.owns(q, m.layer, m.expert);
            if (m.to.storage == expert_storage::host) {
                if (held) {
                    const l2_relabel r = home(s, q, m.layer, m.expert);
                    if (!r.ok) { return fail(r.reason); }
                    step.dst = q;
                    step.relabel = true;
                    ++out.relabeled;
                } else {
                    const int t = take(s);
                    if (t < 0) { return fail("no ring slot of storage class " + std::to_string(s) + " for a promotion from the file"); }
                    const l2_relabel r = home(s, t, m.layer, m.expert);
                    if (!r.ok) { return fail(r.reason); }
                    step.dst = t;
                    step.read = true;
                    busy[(size_t) s][(size_t) t] = 1;
                    ++out.reads;
                }
                step.batch = batch;   // take() may have started a new batch
                home_slot[(size_t) c][(size_t) m.to.slot] = step.dst;
            } else {
                // to VRAM through a ring slot; a ring occupant keeps its slot and its expert
                if (held && !busy[(size_t) s][(size_t) q]) {
                    step.src = q;
                    ++out.ring_copies;
                    if (uni_c(c)) { released[(size_t) s].push_back({q, m.layer, m.expert}); }
                } else {
                    step.src = take(s);
                    if (step.src < 0) { return fail("no ring slot of storage class " + std::to_string(s) + " for a promotion to VRAM"); }
                    step.read = true;
                    ++out.reads;
                }
                step.batch = batch;
                busy[(size_t) s][(size_t) step.src] = 1;
            }
        }
    }
    out.batches = batch + 1;

    // Logical slots left without a home end the plan empty (spares): they take an empty ring slot
    // when there is one. Under the unified policy they stay without one.
    ledger.sink_empty();
    for (int c = 0; c < classes; ++c) {
        if (uni_c(c)) { continue; }
        const int s = storage_of[(size_t) c];
        for (size_t i = 0; i < home_slot[(size_t) c].size(); ++i) {
            if (home_slot[(size_t) c][i] >= 0) { continue; }
            if (new_of[(size_t) c][i] >= 0) { return fail("a host resident of the new plan has no move"); }
            const int t = ledger.take_free(s, out.evicted);
            const l2_relabel r = t >= 0 ? ledger.relabel_to_spare(s, t) : l2_relabel();
            if (!r.ok) { return fail(t < 0 ? "no ring slot for a spare host slot" : r.reason); }
            out.evicted.insert(out.evicted.end(), r.evicted.begin(), r.evicted.end());
            home_slot[(size_t) c][i] = t;
        }
    }
    ledger.sink_empty();

    // the rings kept their size, and every home is one resident slot holding the plan's expert
    for (int s = 0; s < ledger.storages(); ++s) {
        if (uni_s(s)) { continue; }
        if (ledger.ring_count(s) != ring_before[(size_t) s]) {
            return fail("the ring of storage class " + std::to_string(s) + " changed its size");
        }
    }
    std::vector<std::vector<uint8_t>> taken((size_t) ledger.storages());
    for (int s = 0; s < ledger.storages(); ++s) { taken[(size_t) s].assign((size_t) ledger.slots(s), 0); }
    for (int c = 0; c < classes; ++c) {
        const int s = storage_of[(size_t) c];
        for (size_t i = 0; i < home_slot[(size_t) c].size(); ++i) {
            const int slot = home_slot[(size_t) c][i];
            const int want = new_of[(size_t) c][i];
            if (uni_c(c) && slot < 0) {
                if (want >= 0 && !dropped[(size_t) c][i]) { return fail("a host resident of the new plan has no home"); }
                continue;
            }
            int l = -1, e = -1;
            const bool holds = ledger.occupant(s, slot, l, e);
            if (taken[(size_t) s][(size_t) slot] || ledger.role_of(s, slot) != slot_role::resident ||
                    (want < 0 ? holds : (!holds || l*experts + e != want))) {
                return fail("the homes do not match the new plan");
            }
            taken[(size_t) s][(size_t) slot] = 1;
        }
    }
    std::string why;
    if (!ledger.check(why)) { return fail(why); }
    out.ok = true;
    return out;
}

// The unified policy after the moves of an install: the ring copy of every VRAM resident (`vram`
// [layer][expert] slot or -1) is emptied, the demotions homed by this install join their rings at the
// probation point, empty slots go to the least recently used end, and no logical host slot keeps a home.
// Nothing may compute meanwhile. Returns the demotions that joined. With `active`, the classes outside
// it keep their ring copies, residents and homes (`storage_of` [size class] is then needed for the homes).
inline int settle_unified(l2_class_ledger & ledger, std::vector<std::vector<int>> & home_slot,
        const std::vector<std::vector<int32_t>> & vram, const std::vector<uint8_t> * active = nullptr,
        const std::vector<int> * storage_of = nullptr) {
    for (int l = 0; l < (int) vram.size(); ++l) {
        const int s = ledger.storage_of(l);
        if (s < 0 || (active != nullptr && !(*active)[(size_t) s])) { continue; }
        for (int e = 0; e < (int) vram[(size_t) l].size(); ++e) {
            if (vram[(size_t) l][(size_t) e] < 0) { continue; }
            const int slot = ledger.slot_of(l, e);
            if (slot >= 0 && ledger.owns(slot, l, e)) { ledger.sink(s, slot); }
        }
    }
    const int joined = ledger.residents_to_ring(l2_place::probation, active);
    ledger.sink_empty(active);
    for (size_t c = 0; c < home_slot.size(); ++c) {
        if (active != nullptr && storage_of != nullptr && !(*active)[(size_t) (*storage_of)[c]]) { continue; }
        std::fill(home_slot[c].begin(), home_slot[c].end(), -1);
    }
    return joined;
}

// One read of a reseed: (layer, expert) into ring slot `slot` of its storage class.
struct l2_reseed_read {
    int layer = -1, expert = -1, slot = -1;
};

// The unified policy at an install: each class's ring is brought to the best experts of a profile.
// `score(layer, expert)` > 0 for an expert the profile selected; `experts_of` [layer] the experts of a
// layer; `vram` [layer][expert] slot or -1 (VRAM residents are not candidates). With `fill`, the
// candidates the ring of a class can hold (best first) that it does not hold yet take, in score order,
// its empty slots first and then the slots of the lowest scored occupants outside that set; the caller
// reads them in. Either way every class list is laid out again in score order (dissolve). Nothing may
// compute meanwhile. `scope` (per storage class, null = all) limits it to those classes.
template <typename Score>
inline std::vector<l2_reseed_read> plan_unified_reseed(l2_class_ledger & ledger, const std::vector<int> & experts_of,
        const std::vector<std::vector<int32_t>> & vram, Score score, bool fill, const std::vector<uint8_t> * scope = nullptr) {
    std::vector<l2_reseed_read> out;
    const int layers = (int) experts_of.size();
    if (fill) {
        for (int s = 0; s < ledger.storages(); ++s) {
            if (scope != nullptr && !(*scope)[(size_t) s]) { continue; }
            struct cand { double score; int layer, expert; };
            std::vector<cand> cands;
            int experts = 0;
            for (int l = 0; l < layers; ++l) {
                if (ledger.storage_of(l) != s) { continue; }
                experts = std::max(experts, experts_of[(size_t) l]);
                for (int e = 0; e < experts_of[(size_t) l]; ++e) {
                    if (vram[(size_t) l][(size_t) e] >= 0) { continue; }
                    const double v = double(score(l, e));
                    if (v > 0.0) { cands.push_back({v, l, e}); }
                }
            }
            std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.score > b.score; });
            const size_t keep = std::min(cands.size(), (size_t) std::max(ledger.ring_count(s), 0));
            std::vector<uint8_t> wanted((size_t) layers*(size_t) std::max(experts, 1), 0);
            std::vector<cand> missing;
            for (size_t i = 0; i < keep; ++i) {
                wanted[(size_t) cands[i].layer*(size_t) experts + (size_t) cands[i].expert] = 1;
                if (ledger.slot_of(cands[i].layer, cands[i].expert) < 0) { missing.push_back(cands[i]); }
            }
            if (missing.empty()) { continue; }
            // victims: empty slots in list order, then occupants outside the wanted set, lowest score first
            std::vector<int> victims;
            std::vector<std::pair<double, int>> held;
            for (int slot : ledger.order(s)) {
                int l = -1, e = -1;
                if (!ledger.occupant(s, slot, l, e)) {
                    victims.push_back(slot);
                } else if (!wanted[(size_t) l*(size_t) experts + (size_t) e]) {
                    held.push_back({double(score(l, e)), slot});
                }
            }
            std::stable_sort(held.begin(), held.end(),
                [](const std::pair<double, int> & a, const std::pair<double, int> & b) { return a.first < b.first; });
            for (const auto & item : held) { victims.push_back(item.second); }
            size_t v = 0;
            for (const cand & c : missing) {
                bool placed = false;
                for (; v < victims.size() && !placed; ++v) {
                    const int slot = victims[v];
                    placed = ledger.sink(s, slot) && ledger.adopt(s, slot, c.layer, c.expert, l2_place::lru);
                    if (placed) { out.push_back({c.layer, c.expert, slot}); }
                }
                if (!placed) { break; }
            }
        }
    }
    ledger.dissolve(score, scope);
    return out;
}

} // namespace ggml_cuda_expert
