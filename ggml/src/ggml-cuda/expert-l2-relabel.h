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

#include "expert-l2-class-ledger.h"
#include "expert-plan.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// What the mover does for one move of the transaction.
struct l2_install_step {
    int  src     = -1;     // storage slot a copy reads from: the home of a host source, or a ring slot for a promotion to VRAM
    int  dst     = -1;     // storage slot that becomes the home of a host destination
    bool read    = false;  // read the file first: into `dst` for a host destination, into `src` for a VRAM one
    bool relabel = false;  // host destination whose ring slot already held the expert: nothing moves
    int  batch   = -1;     // moves from the file: the read batch
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
};

// `home_slot` [size class][logical slot] is the storage slot of every logical host slot; `before` and
// `after` are the host locations of the transaction's two plans and `moves` its moves. Updates the
// ledger and `home_slot`. Nothing may compute meanwhile.
inline l2_install_plan plan_relabel_install(l2_class_ledger & ledger, std::vector<std::vector<int>> & home_slot,
        const std::vector<int> & layer_class, const std::vector<int> & storage_of,
        const expert_locations & before, const expert_locations & after, const std::vector<install_move> & moves) {
    l2_install_plan out;
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
    // when there is one.
    ledger.sink_empty();
    for (int c = 0; c < classes; ++c) {
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
            int l = -1, e = -1;
            const bool holds = ledger.occupant(s, slot, l, e);
            const int want = new_of[(size_t) c][i];
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

} // namespace ggml_cuda_expert
