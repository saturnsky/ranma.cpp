#pragma once

// ranma: the prefill lend of the expert cache. Pure C++ (the restore space's decisions are also device functions), shared by
// the controller (expert-controller.cu) and its CPU test (tests/test-expert-lend.cpp).
//
// A prompt borrows its buffers (the layer-major edges, carried tensors and MoE output, the expert-major op's scratch,
// its staging and its T) from the L1 budget instead of allocating them on its own:
//   1. The lowest ranked L1 residents leave (exclusive: to the L2 or the file; inclusive: they are copies), the kept
//      residents above the new slot count move down, and the top slots of each size class, all kinds, become the lent
//      regions (one contiguous region per class and kind). The pieces are packed into them. The first bytes of each
//      region stay zero: MMQ reads past the last row of the slot below.
//   2. Nothing is lent from host memory, and what does not fit is not allocated elsewhere: the caller narrows its
//      width (fewer rows per pass) and runs more passes. The L2 holds experts only.
// The lend ends without a refill: the regions become the restore space (below). A decode launch that reads an evicted
// expert the installed decode plan keeps in L1 copies it there and pins it; other recent misses use the free part as a
// least recently used cache. The next lend takes the restore space back: the tables return to the mirror of the plan.
// A joint cache (a draft model in the cache) lends over the layers of every model (see "a joint cache" below).

#include <algorithm>
#include <array>
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
    uint8_t  tier   = 3;    // evicted in tier order: 0 an empty slot, 1 an LRU copy, 2 a pinned copy (of a restore space), 3 a resident
};

// The one rank function of the lend: lower first. Tier ascending, then score ascending, then the later layer, then the
// higher expert id (plan_placement keeps lower layers and ids first on a tie, so this evicts in the reverse of its fill
// order).
inline bool lend_rank_less(const lend_resident & a, const lend_resident & b) {
    if (a.tier != b.tier) { return a.tier < b.tier; }
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
// piece; of the last tie (tier and score) only those of the classes the pieces need (below). `held[c]`: slots of class c the lend holds already (they hold no resident); the plan only adds to them, and its
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
    // The last tie: the victims of the tier and score the pieces stopped in rank alike (the later layer and the higher id
    // only order them), so the plan gives back those of a class the pieces turn out not to need. One class at a time, the
    // one of the largest experts first, its victims of the tie leave the plan from the highest ranked down while the pieces
    // still fit, until no class gives one back. Nothing of a lower tier or score stays while one of a higher leaves, and no
    // class loses more slots than before (e.g. a cold joint cache: the draft's class ranks first on the later layers, but
    // the target's regions hold every piece).
    if (!p.victims.empty()) {
        const lend_resident last = p.victims.back();
        size_t tie = p.victims.size();
        while (tie > 0 && p.victims[tie - 1].tier == last.tier && p.victims[tie - 1].score == last.score) { --tie; }
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

// The most slots of each class a lend of `pieces` evicts: the fewest top slots of that class alone whose regions hold
// every piece (packed as plan_l1_lend packs them), or all its residents when even they do not. plan_l1_lend evicts the
// lowest ranked residents over every class and stops as soon as the pieces fit, so it never takes more of a class than
// that class alone needs: an upper bound per class whatever the ranks, the exact count with one class. Only
// kind_bytes, guard, slots and the number of residents of `cls` are read (an L2 of -1 is sized before any resident).
inline std::vector<int> lend_evict_bound(const std::vector<lend_class> & cls, const std::vector<size_t> & pieces, size_t align) {
    std::vector<int> out(cls.size(), 0);
    bool any = false;
    for (size_t p : pieces) { any = any || p > 0; }
    if (!any) { return out; }
    for (size_t c = 0; c < cls.size(); ++c) {
        const int n_res = (int) std::min<size_t>(cls[c].residents.size(), (size_t) std::max(0, cls[c].slots));
        std::vector<int> given(cls.size(), 0);
        auto fits = [&](int n) {
            given[c] = n;
            std::vector<size_t> bytes;
            for (const auto & r : lend_regions_of(cls, given)) { bytes.push_back(r.bytes); }
            std::vector<lend_fit> f;
            return lend_pack(pieces, bytes, align, f);
        };
        // the first n that holds them, counted up one slot at a time as plan_l1_lend does, else every resident
        int n = 0;
        while (n < n_res && !fits(n)) { ++n; }
        out[c] = n;
    }
    return out;
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
//
// Temporary policy (2026-10-15, not final): the decisions are lend_space_admit and lend_space_key, nothing else.
// The lent slots of a class are one space shared by every layer of the class. A decode launch that reads an evicted
// expert the installed decode plan keeps in L1 (a "wanted" expert) copies it into the space and pins it there. Any
// other expert the launch misses may take an unpinned slot as a least recently used copy, when it passes the ghost
// filter of the decode pool (ghost:K). A slot goes to the best key: an empty slot first,
// else the least recently used unpinned slot that this launch does not read. Each pinned expert takes one slot from
// the LRU part, so with every wanted expert back the LRU part is empty.

// Whether the installed decode plan keeps (layer, expert) in L1: `want` [layer*n_experts + expert], empty = no plan.
inline bool lend_wanted(const std::vector<uint8_t> & want, int n_experts, int layer, int expert) {
    const size_t i = (size_t) layer*(size_t) n_experts + (size_t) expert;
    return layer >= 0 && expert >= 0 && expert < n_experts && i < want.size() && want[i] != 0;
}

struct lend_restore {
    int layer = -1, expert = -1, cls = -1, slot = -1;   // slot: the lent slot it holds (a kept copy), -1 = none yet
    int pin = 1;                                        // a kept copy: 1 pinned, 0 an LRU copy
};

// A new lend over an open restore space (the classes' spaces are [limit, slots)): the pieces take only as many top
// slots of the space as they need. The slots come from the empty slots first, then the LRU copies, then the pinned
// copies of the lowest rank (lend_rank_less over the tiers); the copies that stay and sit in the taken top move down into
// the freed slots below it. ok = false when the pieces need more than the spaces hold: the caller then drops the spaces
// and evicts L1 residents as before (every copy ranks below every resident, so nothing of a space would stay anyway).
struct lend_space_take {
    bool ok = false;
    lend_l1_plan plan;                     // regions and fits; plan.limit [class] = the first slot of the pieces (`top`)
    std::vector<lend_restore> kept;        // the copies that stay, at their final slots
    std::vector<lend_restore> dropped;     // the copies the pieces take (they leave the tables)
    std::vector<std::array<int, 3>> moves; // (class, from, to): a kept copy above the top moves down
    std::vector<int> empty_taken;          // [class] empty slots of the space the pieces take
};

inline lend_space_take plan_space_lend(const std::vector<lend_class> & kinds, const std::vector<int> & limit,
        const std::vector<lend_restore> & copies, const std::vector<uint64_t> & score, int n_experts,
        const std::vector<size_t> & pieces, size_t align) {
    lend_space_take out;
    const size_t nc = kinds.size();
    std::vector<lend_class> cls = kinds;
    std::vector<std::vector<uint8_t>> taken(nc);
    for (size_t c = 0; c < nc; ++c) {
        cls[c].residents.clear();
        taken[c].assign((size_t) std::max(cls[c].slots, 0), 0);
    }
    for (const lend_restore & r : copies) {
        if (r.cls < 0 || (size_t) r.cls >= nc || r.slot < limit[(size_t) r.cls] || r.slot >= cls[(size_t) r.cls].slots) {
            return out;
        }
        lend_resident x;
        x.layer  = r.layer;
        x.expert = r.expert;
        x.slot   = r.slot;
        x.score  = score.empty() ? 0 : score[(size_t) r.layer*(size_t) n_experts + (size_t) r.expert];
        x.tier   = r.pin ? 2 : 1;
        if (taken[(size_t) r.cls][(size_t) r.slot]) { return out; }
        taken[(size_t) r.cls][(size_t) r.slot] = 1;
        cls[(size_t) r.cls].residents.push_back(x);
    }
    // an empty slot of a space is a resident of tier 0 (layer -1 - class): the first a lend takes, and nothing leaves for it
    for (size_t c = 0; c < nc; ++c) {
        for (int slot = std::max(limit[c], 0); slot < cls[c].slots; ++slot) {
            if (taken[c][(size_t) slot]) { continue; }
            lend_resident x;
            x.layer  = -1 - (int) c;   // the class of an empty slot
            x.expert = slot;
            x.slot   = slot;
            x.tier   = 0;
            cls[c].residents.push_back(x);
        }
    }
    out.plan = plan_l1_lend(cls, pieces, align);
    if (!out.plan.ok) {
        return out;
    }
    out.empty_taken.assign(nc, 0);
    std::vector<std::vector<uint8_t>> gone(nc), used(nc);
    for (size_t c = 0; c < nc; ++c) {
        gone[c].assign((size_t) std::max(cls[c].slots, 0), 0);
        used[c].assign((size_t) std::max(cls[c].slots, 0), 0);
    }
    for (const lend_resident & v : out.plan.victims) {
        if (v.layer < 0) {
            ++out.empty_taken[(size_t) (-1 - v.layer)];
            continue;
        }
        for (const lend_restore & r : copies) {
            if (r.layer == v.layer && r.expert == v.expert) {
                out.dropped.push_back(r);
                gone[(size_t) r.cls][(size_t) r.slot] = 1;
            }
        }
    }
    for (const lend_restore & r : copies) {
        if (!gone[(size_t) r.cls][(size_t) r.slot]) { used[(size_t) r.cls][(size_t) r.slot] = 1; }
    }
    // the kept copies above the top move into the free slots below it, lowest first
    std::vector<int> next(limit);
    for (const lend_restore & r : copies) {
        if (gone[(size_t) r.cls][(size_t) r.slot]) { continue; }
        lend_restore k = r;
        const size_t c = (size_t) r.cls;
        if (r.slot >= out.plan.limit[c]) {
            while (next[c] < out.plan.limit[c] && used[c][(size_t) next[c]]) { ++next[c]; }
            if (next[c] >= out.plan.limit[c]) { out = lend_space_take(); return out; }
            used[c][(size_t) next[c]] = 1;
            out.moves.push_back({ r.cls, r.slot, next[c] });
            k.slot = next[c];
        }
        out.kept.push_back(k);
    }
    out.ok = true;
    return out;
}

// The wanted victims: the evicted experts the decode plan `want` keeps in L1 (null = every victim; an empty vector:
// none, a cold cache), highest rank first, without the ones in `keep` (back in their slot already).
inline std::vector<lend_restore> plan_restore(const std::vector<lend_resident> & victims_low_first,
        const std::vector<int> & layer_class, const std::vector<lend_restore> * keep = nullptr,
        const std::vector<uint8_t> * want = nullptr, int n_experts = 0) {
    std::vector<lend_restore> out;
    for (size_t i = victims_low_first.size(); i-- > 0;) {
        const lend_resident & v = victims_low_first[i];
        if (v.layer < 0 || (size_t) v.layer >= layer_class.size() || layer_class[(size_t) v.layer] < 0) { continue; }
        if (want != nullptr && !lend_wanted(*want, n_experts, v.layer, v.expert)) { continue; }
        bool skip = false;
        for (size_t k = 0; keep != nullptr && k < keep->size() && !skip; ++k) {
            skip = (*keep)[k].layer == v.layer && (*keep)[k].expert == v.expert;
        }
        for (size_t k = 0; k < out.size() && !skip; ++k) {
            skip = out[k].layer == v.layer && out[k].expert == v.expert;
        }
        if (skip) { continue; }
        lend_restore r;
        r.layer = v.layer;
        r.expert = v.expert;
        r.cls = layer_class[(size_t) v.layer];
        out.push_back(r);
    }
    return out;
}

// The wanted list of an eager restore by layer (expert-controller.cu feeds each layer's list to the restore kernels as a
// route, lend_space_max_ids at a time), each expert once.
inline std::vector<std::vector<int32_t>> lend_eager_batches(const std::vector<lend_restore> & wanted, int n_layers) {
    std::vector<std::vector<int32_t>> by((size_t) std::max(n_layers, 0));
    for (const lend_restore & r : wanted) {
        if (r.layer < 0 || r.layer >= n_layers) { continue; }
        auto & row = by[(size_t) r.layer];
        if (std::find(row.begin(), row.end(), r.expert) == row.end()) { row.push_back(r.expert); }
    }
    return by;
}

static constexpr uint32_t lend_never = 0xFFFFFFFFu;   // last miss of an expert that never missed

// Admission: a wanted expert always; with the LRU part off (`lru` false, RANMA_EXPERT_LEND_LRU=0) nothing else; any
// other miss only when it missed in its layer within the last `ghost` steps of the layer before this one (`step`);
// ghost 0 admits every miss. An expert that `n_rows` >= 2 rows of this launch selected has had its second access in this
// very step, so it passes the ghost filter without a recorded earlier miss.
LEND_HD inline bool lend_space_admit(bool wanted, uint32_t last_miss, uint32_t step, uint32_t ghost, bool lru = true,
        int n_rows = 1) {
    if (wanted) {
        return true;
    }
    if (!lru) {
        return false;
    }
    if (ghost == 0) {
        return true;
    }
    if (n_rows >= 2) {
        return true;
    }
    return last_miss != lend_never && last_miss != step && step - last_miss <= ghost;
}

// The rows of a launch that selected `e`.
LEND_HD inline int lend_rows_with(const int32_t * ids, int rows, int used, int stride, int32_t e) {
    int n = 0;
    for (int r = 0; r < rows; ++r) {
        for (int u = 0; u < used; ++u) {
            if (ids[r*stride + u] == e) { ++n; break; }
        }
    }
    return n;
}

// The victim order: the slot with the lowest key gets the next copy. An empty slot first (lowest index), then the
// unpinned slot used longest ago (`t`: the class clock of this launch); a pinned slot or one this launch reads
// (stamp == t) is never taken (the largest key).
static constexpr uint64_t lend_space_none = ~(uint64_t) 0;
LEND_HD inline uint64_t lend_space_key(int32_t own, int32_t pin, uint32_t stamp, uint32_t t, int j) {
    if (own < 0) {
        return (uint64_t) j;
    }
    if (pin != 0 || stamp == t) {
        return lend_space_none;
    }
    return (uint64_t(1) << 62) | ((uint64_t) (0xFFFFFFFFu - (t - stamp)) << 24) | (uint64_t) (j & 0xFFFFFF);
}

// One copy the restore space makes: `expert` of the launch's layer into lent slot first + j, replacing the
// occupant `evicted` (layer*n_experts + expert, or -1); `wanted` 1 pins it.
struct lend_space_copy {
    int32_t expert = -1, j = -1, evicted = -1, wanted = 0;
};

// Whether the copies of a launch may read their sources: every kind of the generation `seq` is in place. A staged SSD
// service publishes the kinds one at a time (kind_ready) before the whole generation (ready); the controller therefore
// waits for kinds 1 and 2 too, after the route's wait for kind 0, before the restore space copies.
LEND_HD inline bool lend_space_sources_ready(uint32_t ready, const uint32_t kind_ready[3], uint32_t seq) {
    return ready == seq || (kind_ready[0] == seq && kind_ready[1] == seq && kind_ready[2] == seq);
}

static constexpr int lend_space_max_ids = 64;   // distinct experts one launch considers

// One decode launch of a layer against the restore space of its class: the CPU reference of the restore kernels
// (expert-controller.cu), which take the same steps with lend_space_admit and lend_space_key.
//   slots [expert]: the layer's VRAM table (lent slot j is first + j); want [expert] >= 0: a wanted expert not back
//   yet; readable(e): it has a source to copy from; last_miss [expert], *step: the layer's ghost state; *clock: the
//   class's clock; own/pin/stamp [n]: the space of the class.
// Writes the copies (wanted ones first), updates own/pin/stamp, want, last_miss, the clocks and *hits. The caller
// copies, publishes slots[e] = first + j, and clears the table entry of each evicted occupant in its own layer.
template <typename Readable>
inline int lend_space_step(const int32_t * ids, int rows, int used, int stride, int n_experts, int layer,
        const int32_t * slots, int32_t * want, uint32_t * last_miss, uint32_t * step, uint32_t ghost,
        int first, int n, int32_t * own, int32_t * pin, uint32_t * stamp, uint32_t * clock,
        Readable readable, lend_space_copy * out, int max, int * hits = nullptr, bool lru = true) {
    const uint32_t t  = ++*clock;
    const uint32_t st = ++*step;
    int32_t cand[2][lend_space_max_ids];
    int nc[2] = {0, 0};
    int32_t miss[lend_space_max_ids];
    int nm = 0;
    for (int r = 0; r < rows; ++r) {
        for (int u = 0; u < used; ++u) {
            const int32_t e = ids[r*stride + u];
            if (e < 0 || e >= n_experts) { continue; }
            const int32_t g = slots[e];
            if (g >= first && g < first + n && own[g - first] == layer*n_experts + e) {
                stamp[g - first] = t;
                if (hits != nullptr) { ++*hits; }
                continue;
            }
            bool seen = false;
            for (int i = 0; i < nm && !seen; ++i) { seen = miss[i] == e; }
            if (g >= 0 || seen || nm >= lend_space_max_ids) { continue; }
            miss[nm++] = e;
            const int w = want[e] >= 0 ? 0 : 1;
            if (readable(e) && lend_space_admit(w == 0, last_miss[e], st, ghost, lru,
                    ghost != 0 && w != 0 ? lend_rows_with(ids, rows, used, stride, e) : 1)) {
                cand[w][nc[w]++] = e;
            }
        }
    }
    int n_out = 0;
    for (int w = 0; w < 2; ++w) {
        for (int i = 0; i < nc[w] && n_out < max; ++i) {
            uint64_t best = lend_space_none;
            int bj = -1;
            for (int j = 0; j < n; ++j) {
                const uint64_t k = lend_space_key(own[j], pin[j], stamp[j], t, j);
                if (k < best) { best = k; bj = j; }
            }
            if (bj < 0) { break; }
            const int32_t e = cand[w][i];
            out[n_out].expert  = e;
            out[n_out].j       = bj;
            out[n_out].evicted = own[bj];
            out[n_out].wanted  = w == 0 ? 1 : 0;
            ++n_out;
            own[bj]   = layer*n_experts + e;
            pin[bj]   = w == 0 ? 1 : 0;
            stamp[bj] = t;
            if (w == 0) { want[e] = -1; }
        }
    }
    for (int i = 0; i < nm; ++i) { last_miss[miss[i]] = st; }
    return n_out;
}

// ---- a joint cache (a draft model in the cache) ------------------------------------------------------------------------
//
// The lend of a joint cache is the one of a single model over the joint layers: the planner, the restore space and the
// rank see every member's layers (the joint layer offsets and the joint expert stride). What a joint cache adds:
//   - a member must be ready before its experts may be evicted or restored: bound (its weights context arrived), loaded
//     (its slices are in their homes), its aux rows final and its files open in the SSD tier. A dropped member's
//     layers keep their slots, unused: they are never restored. A lend waits until every member that is not dropped is
//     ready (a draft joins after the target's load; nothing computes before that);
//   - the restore kernels of every member write the one restore space (own/pin/stamp, the launch's copy list), so two
//     members' graphs must not run at once. They do not with a drafting context that computes in its target's compute
//     buffer (llama_context::share_compute_wait); the controller drains the previous context's stream at every change
//     of the context besides (lend_order_switch), which costs nothing when its outputs were read already.

struct lend_member {
    int  offset = 0, n_layers = 0;   // its joint layers
    bool bound = false, loaded = false, dropped = false, tier_ready = false;
};

// Whether a lend may evict any expert: every member that is not dropped is bound, loaded and (with an SSD tier) open in it.
inline bool lend_members_ready(const std::vector<lend_member> & members, bool tier, std::string * why = nullptr) {
    for (size_t m = 0; m < members.size(); ++m) {
        const lend_member & mb = members[m];
        if (mb.dropped) {
            continue;
        }
        const char * w = !mb.bound ? "is not bound" : !mb.loaded ? "is not loaded" : tier && !mb.tier_ready ? "is not open in the SSD tier" : nullptr;
        if (w != nullptr) {
            if (why != nullptr) { *why = "model " + std::to_string(m) + " of the joint cache " + w; }
            return false;
        }
    }
    return true;
}

// [joint layer] 1: the restore space may copy into that layer's table (its member is ready and not dropped).
inline std::vector<uint8_t> lend_layers_ready(const std::vector<lend_member> & members, int n_layers, bool tier) {
    std::vector<uint8_t> out((size_t) std::max(n_layers, 0), 0);
    for (const lend_member & mb : members) {
        const bool ok = !mb.dropped && mb.bound && mb.loaded && (!tier || mb.tier_ready);
        for (int l = mb.offset; ok && l < mb.offset + mb.n_layers && l < n_layers; ++l) {
            if (l >= 0) { out[(size_t) l] = 1; }
        }
    }
    return out;
}

// The order of the contexts that run restore kernels: before a graph of `ctx` the stream of the previous context is
// drained when the context changed. Returns that stream (null: nothing to drain) and records `ctx`, `stream`.
struct lend_order {
    const void * ctx    = nullptr;
    void       * stream = nullptr;
    uint64_t     drains = 0;
};

inline void * lend_order_switch(lend_order & o, const void * ctx, void * stream) {
    void * drain = o.ctx != nullptr && o.ctx != ctx ? o.stream : nullptr;
    o.drains += drain != nullptr ? 1 : 0;
    o.ctx    = ctx;
    o.stream = stream;
    return drain;
}

// A context freed: its stream is no longer drained.
inline void lend_order_forget(lend_order & o, const void * ctx) {
    if (o.ctx == ctx) {
        o.ctx    = nullptr;
        o.stream = nullptr;
    }
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
    std::vector<int> top;             // [class] while held: the first slot of the pieces; [limit, top) stays a restore space
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
        top = new_limit;
        victims.insert(victims.end(), more.begin(), more.end());
        restore.clear();
        restored = 0;
        owner = who;
        explicit_ = expl;
        state = lend_state::held;
        ++epoch;
        return true;
    }
    // A lease of the top of an open restore space (plan_space_lend): the pieces start at `new_top`, the plan keeps its
    // slots and victims, and [limit, new_top) stays a restore space with its copies.
    bool hold_space(const std::vector<int> & new_top, const void * who, bool expl) {
        if (state != lend_state::restore || new_top.size() != slots.size()) { return false; }
        for (size_t c = 0; c < slots.size(); ++c) {
            if (new_top[c] < limit[c] || new_top[c] > slots[c]) { return false; }
        }
        top = new_top;
        owner = who;
        explicit_ = expl;
        state = lend_state::held;
        ++epoch;
        return true;
    }
    // The prompt ended: the regions become the restore space of the victims the decode plan `want` keeps in L1
    // (lend_wanted; null = every victim).
    bool open_restore(const std::vector<int> & layer_class, const std::vector<uint8_t> * want = nullptr, int n_experts = 0) {
        if (state != lend_state::held) { return false; }
        lend_rank_order(victims);
        restore = plan_restore(victims, layer_class, nullptr, want, n_experts);
        top = slots;
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
        top = slots;
        victims.clear();
        restore.clear();
        restored = 0;
    }
};

} // namespace ggml_cuda_expert
