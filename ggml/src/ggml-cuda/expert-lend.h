#pragma once

// ranma: the prefill lend of the expert cache. Pure C++ (the restore list below is also a device function), shared by
// the controller (expert-controller.cu) and its CPU test (tests/test-expert-lend.cpp).
//
//   2. Host: a piece the GPU reads through a copy only (a graph input) comes from the L2: with a finite L2 the least
//      recently used ring slots of a storage class (the ring is the L2's own order), taken as one range at the top of
//      the arena; with an unlimited L2 the lend allocates host memory, which is what that budget is.
// A prompt borrows its buffers (the layer-major edges, carried tensors and MoE output, the expert-major op's scratch,
// its staging and its T) from the L1 budget instead of allocating them on its own:
//   1. The lowest ranked L1 residents leave (exclusive: to the L2 or the file; inclusive: they are copies), the kept
//      residents above the new slot count move down, and the top slots of each size class, all kinds, become the lent
//      regions (one contiguous region per class and kind). The pieces are packed into them. The first bytes of each
//      region stay zero: MMQ reads past the last row of the slot below.
//   2. Nothing is lent from host memory, and what does not fit is not allocated elsewhere: the caller narrows its
//      width (fewer rows per pass) and runs more passes. The L2 holds experts only.
// The lend ends without a refill: the regions become the restore space. Every lent L1 slot gets one of the evicted
// experts the installed decode plan keeps in L1 (the highest ranked first; none on a cold cache), and a decode launch that selects such an expert copies it into its slot before
// the layer's matmuls and publishes it there (lend_restore_list is the decision both the kernel and the test run).
// The next lend takes the restore space back: the tables return to the mirror of the installed plan.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LEND_HD __host__ __device__
#else
#define LEND_HD
#endif

namespace ggml_cuda_expert {

// ---- the rank of a resident ---------------------------------------------------------------------------------------

// One L1 resident a lend may evict.
struct lend_resident {
    int      layer  = -1;
    int      expert = -1;
    int      slot   = -1;
    uint64_t score  = 0;    // the installed plan's profile score; 0 for every expert of a cold cache
};

// The one rank function of the lend: lower first. Score ascending, then the later layer, then the higher expert id
// (plan_placement keeps lower layers and ids first on a tie, so this evicts in the reverse of its fill order).
inline bool lend_rank_less(const lend_resident & a, const lend_resident & b) {
    if (a.score != b.score) { return a.score < b.score; }
    if (a.layer != b.layer) { return a.layer > b.layer; }
    return a.expert > b.expert;
}

inline void lend_rank_order(std::vector<lend_resident> & r) {
    std::stable_sort(r.begin(), r.end(), lend_rank_less);
}

// ---- packing -----------------------------------------------------------------------------------------------------

struct lend_fit {
    int    region = -1;
    size_t offset = 0;      // from the region's first usable byte
};

static inline size_t lend_align_up(size_t v, size_t a) {
    return a > 1 ? (v + a - 1)/a*a : v;
}

// Largest piece first, each into the first region that holds it (first fit), or, when that leaves a piece out, each into
// the region with the most room left (worst fit: it spreads the pieces over the regions of a class, which first fit
// fills one after the other). A piece of 0 bytes takes no region (region -1, offset 0). False when neither fits.
inline bool lend_pack(const std::vector<size_t> & pieces, const std::vector<size_t> & regions, size_t align,
                      std::vector<lend_fit> & out) {
    std::vector<size_t> order(pieces.size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pieces[a] > pieces[b]; });
    for (int worst = 0; worst < 2; ++worst) {
        out.assign(pieces.size(), lend_fit());
        std::vector<size_t> used(regions.size(), 0);
        bool ok = true;
        for (size_t i : order) {
            if (pieces[i] == 0) { continue; }
            int pick = -1;
            size_t room = 0;
            for (size_t r = 0; r < regions.size(); ++r) {
                const size_t at = lend_align_up(used[r], align);
                if (at > regions[r] || regions[r] - at < pieces[i]) { continue; }
                if (pick < 0 || (worst && regions[r] - at > room)) {
                    pick = (int) r;
                    room = regions[r] - at;
                    if (!worst) { break; }
                }
            }
            if (pick < 0) {
                ok = false;
                break;
            }
            out[i].region = pick;
            out[i].offset = lend_align_up(used[(size_t) pick], align);
            used[(size_t) pick] = out[i].offset + pieces[i];
        }
        if (ok) {
            return true;
        }
    }
    return false;
}

// ---- the VRAM (L1) plan -----------------------------------------------------------------------------------------

struct lend_class {
    std::vector<size_t> kind_bytes;          // [kind] slot stride of the class's arena of that kind
    std::vector<size_t> guard;               // [kind] zero bytes kept at the start of a region
    int slots  = 0;                          // the class's slot count now (static capacity and spares)
    std::vector<lend_resident> residents;    // its L1 residents, any order
};

struct lend_region {
    int    cls  = -1;
    int    kind = -1;
    size_t begin = 0;     // first usable byte, from the base of the (class, kind) arena
    size_t bytes = 0;
};

struct lend_l1_plan {
    bool ok = false;
    std::string why;
    std::vector<int> evict;                  // [class] slots taken from the top
    std::vector<int> limit;                  // [class] slot count left: slots - evict
    std::vector<lend_resident> victims;      // lowest rank first, over all classes
    std::vector<lend_region> regions;
    std::vector<lend_fit> fits;              // [piece]
    int untied = 0;                          // victims of the last tie given back (their class was not needed)
};

inline std::vector<lend_region> lend_regions_of(const std::vector<lend_class> & cls, const std::vector<int> & evict) {
    std::vector<lend_region> out;
    for (size_t c = 0; c < cls.size(); ++c) {
        if (evict[c] <= 0) { continue; }
        const int limit = cls[c].slots - evict[c];
        for (size_t k = 0; k < cls[c].kind_bytes.size(); ++k) {
            const size_t b = cls[c].kind_bytes[k];
            const size_t g = k < cls[c].guard.size() ? cls[c].guard[k] : 0;
            const size_t span = (size_t) evict[c]*b;
            if (b == 0 || span <= g) { continue; }
            lend_region r;
            r.cls = (int) c;
            r.kind = (int) k;
            r.begin = (size_t) limit*b + g;
            r.bytes = span - g;
            out.push_back(r);
        }
    }
    return out;
}

// [region] the bytes the packed pieces reach in each region (their last end past its first usable byte), for the log of
// a lend: what the region holds beyond that is lent but unused.
inline std::vector<size_t> lend_region_used(const std::vector<lend_region> & regions, const std::vector<lend_fit> & fits,
                                            const std::vector<size_t> & pieces) {
    std::vector<size_t> used(regions.size(), 0);
    for (size_t i = 0; i < fits.size() && i < pieces.size(); ++i) {
        const lend_fit & f = fits[i];
        if (pieces[i] == 0 || f.region < 0 || (size_t) f.region >= used.size()) { continue; }
        used[(size_t) f.region] = std::max(used[(size_t) f.region], f.offset + pieces[i]);
    }
    return used;
}

// The fewest residents, lowest rank first over every class, whose slots taken from the top of their class hold every
// piece; of the last tie (score) only those of the classes the pieces need (below). `held[c]`: slots of class c the lend holds already (they hold no resident); the plan only adds to them, and its
// victims are the residents it adds.
inline lend_l1_plan plan_l1_lend(const std::vector<lend_class> & cls, const std::vector<size_t> & pieces, size_t align,
                                 const std::vector<int> * held = nullptr) {
    lend_l1_plan p;
    const size_t n_cls = cls.size();
    std::vector<lend_resident> all;
    std::vector<int> cls_of;
    for (size_t c = 0; c < n_cls; ++c) {
        for (const lend_resident & r : cls[c].residents) {
            all.push_back(r);
            cls_of.push_back((int) c);
        }
    }
    std::vector<size_t> order(all.size());
    for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lend_rank_less(all[a], all[b]); });
    std::vector<int> given(n_cls, 0);
    for (size_t c = 0; c < n_cls; ++c) {
        given[c] = held != nullptr && c < held->size() ? (*held)[c] : 0;
        if (given[c] < 0 || given[c] + (int) cls[c].residents.size() > cls[c].slots) {
            p.why = "the held slots do not match the class";
            return p;
        }
    }
    auto fits_now = [&]() {
        std::vector<lend_region> regs = lend_regions_of(cls, given);
        std::vector<size_t> bytes;
        for (const auto & r : regs) { bytes.push_back(r.bytes); }
        std::vector<lend_fit> f;
        if (lend_pack(pieces, bytes, align, f)) {
            p.regions = regs;
            p.fits = f;
            return true;
        }
        return false;
    };
    bool ok = fits_now();
    std::vector<int> victim_cls;
    for (size_t next = 0; !ok && next < order.size(); ++next) {
        const size_t i = order[next];
        ++given[(size_t) cls_of[i]];
        p.victims.push_back(all[i]);
        victim_cls.push_back(cls_of[i]);
        ok = fits_now();
    }
    if (!ok) {
        p.why = "the pieces do not fit in the whole L1";
        p.victims.clear();
        return p;
    }
    // The last tie: the victims of the score the pieces stopped in rank alike (the later layer and the higher id
    // only order them), so the plan gives back those of a class the pieces turn out not to need. One class at a time, the
    // one of the largest experts first, its victims of the tie leave the plan from the highest ranked down while the pieces
    // still fit, until no class gives one back. Nothing of a lower score stays while one of a higher leaves, and no
    // class loses more slots than before (e.g. a cold joint cache: the draft's class ranks first on the later layers, but
    // the target's regions hold every piece).
    if (!p.victims.empty()) {
        const lend_resident last = p.victims.back();
        size_t tie = p.victims.size();
        while (tie > 0 && p.victims[tie - 1].score == last.score) { --tie; }
        std::vector<size_t> by(n_cls);
        std::vector<size_t> cls_bytes(n_cls, 0);
        for (size_t c = 0; c < n_cls; ++c) {
            by[c] = c;
            for (size_t b : cls[c].kind_bytes) { cls_bytes[c] += b; }
        }
        std::stable_sort(by.begin(), by.end(), [&](size_t a, size_t b) { return cls_bytes[a] > cls_bytes[b]; });
        std::vector<uint8_t> back(p.victims.size(), 0);
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t c : by) {
                for (size_t i = p.victims.size(); i-- > tie;) {
                    if (back[i] || victim_cls[i] != (int) c) { continue; }
                    --given[c];
                    if (!fits_now()) {
                        ++given[c];   // the regions and fits stay those of the last plan that held them
                        break;
                    }
                    back[i] = 1;
                    ++p.untied;
                    changed = true;
                }
            }
        }
        size_t k = 0;
        for (size_t i = 0; i < p.victims.size(); ++i) {
            if (!back[i]) { p.victims[k++] = p.victims[i]; }
        }
        p.victims.resize(k);
    }
    p.ok = true;
    p.evict = given;
    p.limit.assign(n_cls, 0);
    for (size_t c = 0; c < n_cls; ++c) { p.limit[c] = cls[c].slots - given[c]; }
    return p;
}

// The largest single piece and the total the whole L1 could lend (every resident evicted), for the width search.
inline void lend_l1_capacity(const std::vector<lend_class> & cls, const std::vector<int> & held, size_t & total, size_t & largest) {
    total = 0;
    largest = 0;
    std::vector<int> all(cls.size(), 0);
    for (size_t c = 0; c < cls.size(); ++c) { all[c] = (int) cls[c].residents.size() + (c < held.size() ? held[c] : 0); }
    for (const lend_region & r : lend_regions_of(cls, all)) {
        total += r.bytes;
        largest = std::max(largest, r.bytes);
    }
}

// ---- the width of a prompt's passes (the layer-major prefill) ---------------------------------------------------

// The passes of a prompt of N tokens (more than one ubatch of C_ctx) on an L1 budget: P = ceil(N / W_max) passes, each of
// W = ceil(N / P) tokens rounded up to `unit`, in g = ceil(W / C_ctx) ubatches of C = ceil(W / g) tokens rounded up to
// `unit` and at most C_ctx (equal shares: the fewest passes, each as narrow as they allow). W_max is the widest multiple of
// `unit` that `fits` (a pass of that many tokens can be lent now), or `forced` (> 0) in its place. One pass when all N
// fit. When the rounding pushes the pass of P over what fits, P grows by one. P is 0 when no pass fits or the passes
// would not be fewer than the ubatches of C_ctx (the ubatch loop loads every expert as often).
struct lend_width {
    int64_t P = 0;      // passes
    int64_t W = 0;      // tokens of a pass (the last may be narrower)
    int64_t C = 0;      // tokens of a ubatch of a pass
    int64_t g = 0;      // ubatches of a pass
    int64_t W_max = 0;  // the widest pass that fits
};

template <typename F>
inline lend_width lend_width_rule(int64_t N, int64_t C_ctx, int64_t unit, int64_t forced, F && fits) {
    lend_width r;
    if (N <= 0 || C_ctx <= 0 || unit <= 0) { return r; }
    const int64_t K = (N + C_ctx - 1)/C_ctx;
    if (forced > 0 ? forced >= N : fits(N)) {
        r.P = 1; r.W = N; r.C = C_ctx; r.g = K; r.W_max = N;
        return r;
    }
    if (forced > 0) {
        r.W_max = std::max(unit, forced/unit*unit);
    } else {
        int64_t lo = 0, hi = (N + unit - 1)/unit;   // in units: lo fits (0: none), hi does not
        while (hi - lo > 1) {
            const int64_t mid = lo + (hi - lo)/2;
            (fits(mid*unit) ? lo : hi) = mid;
        }
        r.W_max = lo*unit;
    }
    if (r.W_max <= 0) { return r; }
    for (int64_t P = (N + r.W_max - 1)/r.W_max; P < K; ++P) {
        const int64_t w0 = (N + P - 1)/P;
        const int64_t g  = (w0 + C_ctx - 1)/C_ctx;
        const int64_t c  = std::min<int64_t>(C_ctx, ((w0 + g - 1)/g + unit - 1)/unit*unit);
        const int64_t w  = g*c;
        if (forced > 0 || fits(w)) {
            r.P = ((N + c - 1)/c + g - 1)/g;
            r.W = w;
            r.C = c;
            r.g = g;
            return r;
        }
    }
    return r;
}

// ---- the restore space ------------------------------------------------------------------------------------------

// Whether the installed decode plan keeps (layer, expert) in L1: `want` [layer*n_experts + expert], empty = no plan.
inline bool lend_wanted(const std::vector<uint8_t> & want, int n_experts, int layer, int expert) {
    const size_t i = (size_t) layer*(size_t) n_experts + (size_t) expert;
    return layer >= 0 && expert >= 0 && expert < n_experts && i < want.size() && want[i] != 0;
}

struct lend_restore {
    int layer = -1, expert = -1, cls = -1, slot = -1;
};

// The evicted experts get the lent slots of their class, the highest ranked first, from the bottom of the range up.
// `layer_class[layer]` gives the class of a victim. `keep`: experts the decode restored already, at their slots; they
// stay there and are not in the result. More victims than slots: the lowest ranked get none. `want` (null = every
// victim, see lend_wanted): only the victims the decode plan keeps in L1 get a slot; an empty vector gives none.
inline std::vector<lend_restore> plan_restore(const std::vector<lend_resident> & victims_low_first,
        const std::vector<int> & layer_class, const std::vector<int> & limit, const std::vector<int> & slots,
        const std::vector<lend_restore> * keep = nullptr, const std::vector<uint8_t> * want = nullptr, int n_experts = 0) {
    std::vector<lend_restore> out;
    std::vector<std::vector<uint8_t>> used(limit.size());
    for (size_t c = 0; c < limit.size(); ++c) {
        used[c].assign((size_t) std::max(0, slots[c] - limit[c]), 0);
    }
    auto kept = [&](int layer, int expert) {
        if (keep == nullptr) { return false; }
        for (const lend_restore & k : *keep) {
            if (k.layer == layer && k.expert == expert) { return true; }
        }
        return false;
    };
    if (keep != nullptr) {
        for (const lend_restore & k : *keep) {
            if (k.cls >= 0 && (size_t) k.cls < used.size() && k.slot >= limit[(size_t) k.cls] && k.slot < slots[(size_t) k.cls]) {
                used[(size_t) k.cls][(size_t) (k.slot - limit[(size_t) k.cls])] = 1;
            }
        }
    }
    std::vector<size_t> next(limit.size(), 0);
    for (size_t i = victims_low_first.size(); i-- > 0;) {
        const lend_resident & v = victims_low_first[i];
        if (v.layer < 0 || (size_t) v.layer >= layer_class.size() || kept(v.layer, v.expert)) { continue; }
        if (want != nullptr && !lend_wanted(*want, n_experts, v.layer, v.expert)) { continue; }
        const int c = layer_class[(size_t) v.layer];
        if (c < 0 || (size_t) c >= used.size()) { continue; }
        size_t & j = next[(size_t) c];
        while (j < used[(size_t) c].size() && used[(size_t) c][j]) { ++j; }
        if (j >= used[(size_t) c].size()) { continue; }
        used[(size_t) c][j] = 1;
        lend_restore r;
        r.layer = v.layer;
        r.expert = v.expert;
        r.cls = c;
        r.slot = limit[(size_t) c] + (int) j;
        out.push_back(r);
    }
    return out;
}

// The decode-time decision (the restore kernel, expert-controller.cu, runs exactly this in each block): the experts
// of one launch of a layer that have a restore slot and no VRAM slot, each once, in the order the rows name them.
// `ids`: rows x used ids, `stride` apart per row. Returns the count written to out_e (at most `max`).
LEND_HD inline int lend_restore_list(const int32_t * ids, int rows, int used, int stride, int n_experts,
                                     const int32_t * restore, const int32_t * slots, int32_t * out_e, int max) {
    int n = 0;
    for (int r = 0; r < rows; ++r) {
        for (int u = 0; u < used; ++u) {
            const int32_t e = ids[r*stride + u];
            if (e < 0 || e >= n_experts || restore[e] < 0 || slots[e] >= 0) { continue; }
            bool seen = false;
            for (int j = 0; j < n && !seen; ++j) { seen = out_e[j] == e; }
            if (!seen && n < max) { out_e[n++] = e; }
        }
    }
    return n;
}

// ---- the lease ------------------------------------------------------------------------------------------------------

enum class lend_state : uint8_t {
    none,       // the slots are the plan's
    held,       // the top slots hold the pieces of a prompt; no table names them
    restore,    // the top slots are the restore space of the evicted experts
};

// The host-side record of a lease, without device memory: the controller keeps it beside the tables. Each transition
// checks its precondition; a refused transition changes nothing.
struct lend_book {
    lend_state state = lend_state::none;
    uint64_t epoch = 0;               // every acquire of new regions; a restore entry of another epoch is stale
    const void * owner = nullptr;     // the compute context that holds it
    bool explicit_ = false;           // taken by lend_acquire (layer-major): only lend_release ends it
    std::vector<int> slots;           // [class] the slot counts before the lend
    std::vector<int> limit;           // [class] the slot counts the plan keeps (slots while none)
    std::vector<lend_resident> victims;
    std::vector<lend_restore> restore;
    uint64_t restored = 0;            // entries the decode published since the restore space opened

    bool lent(int cls) const { return cls >= 0 && (size_t) cls < limit.size() && limit[(size_t) cls] < slots[(size_t) cls]; }

    // A lease: `new_limit` are the slot counts the plan keeps (the evictions already ran) and `more` the residents
    // this lend evicted (the earlier victims stay victims: they are out of L1 until the decode restores them).
    bool hold(const std::vector<int> & new_limit, const std::vector<lend_resident> & more, const void * who, bool expl) {
        if (state == lend_state::held || new_limit.size() != slots.size()) { return false; }
        for (size_t c = 0; c < slots.size(); ++c) {
            if (new_limit[c] > limit[c] || new_limit[c] < 0) { return false; }
        }
        limit = new_limit;
        victims.insert(victims.end(), more.begin(), more.end());
        restore.clear();
        restored = 0;
        owner = who;
        explicit_ = expl;
        state = lend_state::held;
        ++epoch;
        return true;
    }
    // The prompt ended: the regions become the restore space of the victims the decode plan keeps in L1 (`want`, see
    // plan_restore).
    bool open_restore(const std::vector<int> & layer_class, const std::vector<uint8_t> * want = nullptr, int n_experts = 0) {
        if (state != lend_state::held) { return false; }
        lend_rank_order(victims);
        restore = plan_restore(victims, layer_class, limit, slots, nullptr, want, n_experts);
        state = lend_state::restore;
        owner = nullptr;
        explicit_ = false;
        return true;
    }
    // The plan's slot counts again (an install or a model release): nothing lent, nothing to restore.
    void clear() {
        state = lend_state::none;
        owner = nullptr;
        explicit_ = false;
        limit = slots;
        victims.clear();
        restore.clear();
        restored = 0;
    }
};

} // namespace ggml_cuda_expert
