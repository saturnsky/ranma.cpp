// The prefill lend of the expert cache (ggml/src/ggml-cuda/expert-lend.h) against a simulated arena:
//   P1: the pieces evict the lowest ranked L1 residents first and take contiguous ranges at the top of each class;
//       nothing is lent from host memory;
//   P2: the width rule: the fewest passes whose pieces L1 holds (all of them in L1), in equal shares;
//   P3: the release does not refill; the lent slots are a restore space: a decode launch copies a wanted evicted expert
//       (the decode plan's) in when it reads it and pins it, other misses use the rest as an LRU cache (empty slots
//       first, then the least recently used unpinned slot); a later lend takes the space back.

#include "expert-l2-class-ledger.h"
#include "expert-lend.h"
#include "expert-plan.h"
#include "expert-redraw.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ggml_cuda_expert;

static int g_fail = 0;
static bool g_quiet = false;
// the paths a run took (each must be taken at least once)
static int n_l1_plans = 0, n_l1_regrow = 0, n_multi_pass = 0, n_restores = 0, n_sticky = 0, n_bounds = 0, n_wanted = 0;
static int n_lru_evictions = 0, n_lru_full = 0, n_space_short = 0, n_space_pinned_kept = 0;
static int n_untied = 0;   // plans that gave victims of their last tie back
#define CHECK(x) do { if (!(x)) { if (!g_quiet) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } ++g_fail; return false; } } while (0)

namespace {

// ---- a simulated L1: classes of slots, residents with scores ----
struct sim_l1 {
    int layers = 0, experts = 0;
    std::vector<int> layer_class;
    std::vector<int> slots;                       // [class]
    std::vector<std::vector<size_t>> kind_bytes;  // [class][kind]
    std::vector<std::vector<int32_t>> table;      // [layer][expert] slot or -1
    std::vector<uint64_t> score;                  // [layer*experts + expert]

    std::vector<lend_class> classes(const std::vector<int> & limit) const {
        std::vector<lend_class> out(slots.size());
        for (size_t c = 0; c < slots.size(); ++c) {
            out[c].kind_bytes = kind_bytes[c];
            out[c].guard = { 512, 512, 512 };
            out[c].slots = slots[c];
        }
        for (int l = 0; l < layers; ++l) {
            for (int e = 0; e < experts; ++e) {
                const int g = table[(size_t) l][(size_t) e];
                if (g >= 0 && g < limit[(size_t) layer_class[(size_t) l]]) {
                    lend_resident r;
                    r.layer = l; r.expert = e; r.slot = g; r.score = score[(size_t) l*experts + e];
                    out[(size_t) layer_class[(size_t) l]].residents.push_back(r);
                }
            }
        }
        return out;
    }
};

sim_l1 make_l1(std::mt19937 & rng, int n_cls, int layers, int experts, bool ties) {
    sim_l1 s;
    s.layers = layers;
    s.experts = experts;
    s.layer_class.resize((size_t) layers);
    for (int l = 0; l < layers; ++l) { s.layer_class[(size_t) l] = l % n_cls; }
    s.slots.resize((size_t) n_cls);
    s.kind_bytes.resize((size_t) n_cls);
    for (int c = 0; c < n_cls; ++c) {
        s.slots[(size_t) c] = 8 + (int) (rng() % 40);
        s.kind_bytes[(size_t) c] = { (size_t) (1024 + rng() % 4096)*16, (size_t) (1024 + rng() % 4096)*16, (size_t) (2048 + rng() % 4096)*16 };
    }
    s.table.assign((size_t) layers, std::vector<int32_t>((size_t) experts, -1));
    s.score.assign((size_t) layers*experts, 0);
    for (auto & v : s.score) { v = ties ? (uint64_t) (rng() % 3) : (uint64_t) (rng() % 100000); }
    // fill each class's slots with random experts of its layers, a few slots left free (spares)
    for (int c = 0; c < n_cls; ++c) {
        std::vector<std::pair<int, int>> cand;
        for (int l = 0; l < layers; ++l) {
            if (s.layer_class[(size_t) l] != c) { continue; }
            for (int e = 0; e < experts; ++e) { cand.push_back({l, e}); }
        }
        std::shuffle(cand.begin(), cand.end(), rng);
        std::vector<int> free_slots;
        for (int sl = 0; sl < s.slots[(size_t) c]; ++sl) { free_slots.push_back(sl); }
        std::shuffle(free_slots.begin(), free_slots.end(), rng);
        const int spares = (int) (rng() % 3);
        for (size_t i = 0; i + spares < free_slots.size() && i < cand.size(); ++i) {
            s.table[(size_t) cand[i].first][(size_t) cand[i].second] = free_slots[i];
        }
    }
    return s;
}

// P1: the victims are the lowest ranked residents, over every class, and nothing ranked lower stays (by tier and score: in
// the last tie the plan keeps the residents of a class the pieces do not need, whatever their layer and id)
static bool rank_below(const lend_resident & a, const lend_resident & b) {
    return a.tier != b.tier ? a.tier < b.tier : a.score < b.score;
}

bool check_rank(const std::vector<lend_class> & cls, const lend_l1_plan & p) {
    std::vector<lend_resident> all;
    for (const auto & c : cls) { all.insert(all.end(), c.residents.begin(), c.residents.end()); }
    std::set<std::pair<int, int>> gone;
    for (const auto & v : p.victims) { gone.insert({v.layer, v.expert}); }
    CHECK(gone.size() == p.victims.size());
    for (const auto & v : p.victims) {
        for (const auto & r : all) {
            if (gone.count({r.layer, r.expert}) == 0) {
                CHECK(!rank_below(r, v));   // a kept resident never ranks below an evicted one
            }
        }
    }
    for (size_t i = 1; i < p.victims.size(); ++i) {
        CHECK(!lend_rank_less(p.victims[i], p.victims[i - 1]));   // lowest first
    }
    return true;
}

// P1: the pieces lie in the regions, each region at the top of its class past the zero guard, apart from each other
bool check_regions(const std::vector<lend_class> & cls, const lend_l1_plan & p, const std::vector<size_t> & pieces, size_t align) {
    CHECK(p.fits.size() == pieces.size());
    for (const auto & r : p.regions) {
        const auto & c = cls[(size_t) r.cls];
        const size_t b = c.kind_bytes[(size_t) r.kind];
        CHECK(r.begin == (size_t) p.limit[(size_t) r.cls]*b + c.guard[(size_t) r.kind]);
        CHECK(r.begin + r.bytes == (size_t) c.slots*b);
    }
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (pieces[i] == 0) { continue; }
        const lend_fit & f = p.fits[i];
        CHECK(f.region >= 0 && (size_t) f.region < p.regions.size());
        CHECK(f.offset % align == 0);
        CHECK(f.offset + pieces[i] <= p.regions[(size_t) f.region].bytes);
        for (size_t j = 0; j < i; ++j) {
            if (pieces[j] == 0 || p.fits[j].region != f.region) { continue; }
            const size_t a0 = f.offset, a1 = f.offset + pieces[i], b0 = p.fits[j].offset, b1 = b0 + pieces[j];
            CHECK(a1 <= b0 || b1 <= a0);
        }
    }
    // the log's use of each region: the last end of its pieces, inside the region
    const std::vector<size_t> used = lend_region_used(p.regions, p.fits, pieces);
    CHECK(used.size() == p.regions.size());
    for (size_t r = 0; r < used.size(); ++r) {
        size_t end = 0;
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (pieces[i] != 0 && p.fits[i].region == (int) r) { end = std::max(end, p.fits[i].offset + pieces[i]); }
        }
        CHECK(used[r] == end && used[r] <= p.regions[r].bytes);
    }
    return true;
}

// P1: after the evacuation (the controller's step 1) no kept resident is at or above the new slot count, and no slot
// holds two experts
bool apply_evacuation(sim_l1 & s, const lend_l1_plan & p) {
    std::set<std::pair<int, int>> gone;
    for (const auto & v : p.victims) { gone.insert({v.layer, v.expert}); }
    expert_slot_table keep((size_t) s.layers);
    for (int l = 0; l < s.layers; ++l) {
        for (int e = 0; e < s.experts; ++e) {
            if (s.table[(size_t) l][(size_t) e] >= 0 && gone.count({l, e}) == 0) { keep[(size_t) l].push_back(e); }
        }
    }
    std::vector<std::vector<uint8_t>> other(s.slots.size());
    for (size_t c = 0; c < s.slots.size(); ++c) { other[c].assign((size_t) s.slots[c], 0); }
    const evac_plan ev = plan_evacuation(s.layer_class, s.experts, s.table, keep, other, s.slots, p.limit);
    CHECK(ev.valid);
    for (const evac_op & op : ev.ops) {
        s.table[(size_t) op.layer][(size_t) op.expert] = op.to;
        if (op.victim_layer >= 0) { s.table[(size_t) op.victim_layer][(size_t) op.victim_expert] = op.from; }
    }
    // the victims leave L1
    for (const auto & v : p.victims) { s.table[(size_t) v.layer][(size_t) v.expert] = -1; }
    std::vector<std::set<int>> used(s.slots.size());
    for (int l = 0; l < s.layers; ++l) {
        for (int e = 0; e < s.experts; ++e) {
            const int g = s.table[(size_t) l][(size_t) e];
            if (g < 0) { continue; }
            const int c = s.layer_class[(size_t) l];
            CHECK(g < p.limit[(size_t) c]);
            CHECK(used[(size_t) c].insert(g).second);
            CHECK(gone.count({l, e}) == 0);
        }
    }
    return true;
}

bool test_l1(uint32_t seed) {
    std::mt19937 rng(seed);
    const int n_cls = 1 + (int) (rng() % 3);
    sim_l1 s = make_l1(rng, n_cls, 6 + (int) (rng() % 10), 16 + (int) (rng() % 48), seed % 3 == 0);
    std::vector<size_t> pieces;
    const int n = 1 + (int) (rng() % 9);
    for (int i = 0; i < n; ++i) { pieces.push_back(rng() % 5 == 0 ? 0 : (size_t) (1 + rng() % 400000)); }
    std::vector<int> limit = s.slots;
    const auto cls = s.classes(limit);
    const lend_l1_plan p = plan_l1_lend(cls, pieces, 256);
    // the bound an L2 of -1 is sized with: never fewer slots of a class than the plan evicts, the same with one class,
    // every resident when nothing fits
    const std::vector<int> bound = lend_evict_bound(cls, pieces, 256);
    CHECK(bound.size() == cls.size());
    for (size_t c = 0; c < cls.size(); ++c) {
        CHECK(bound[c] <= (int) cls[c].residents.size());
        if (p.ok) {
            CHECK(p.evict[c] <= bound[c]);
            CHECK(cls.size() > 1 || p.evict[c] == bound[c]);
        } else {
            CHECK(bound[c] == (int) cls[c].residents.size());
        }
    }
    n_bounds += p.ok && cls.size() > 1 ? 1 : 0;
    if (!p.ok) {
        // the pieces do not fit even with every resident gone: every piece-fit with all slots must fail too
        std::vector<int> all(cls.size());
        for (size_t c = 0; c < cls.size(); ++c) { all[c] = (int) cls[c].residents.size(); }
        std::vector<size_t> bytes;
        for (const auto & r : lend_regions_of(cls, all)) { bytes.push_back(r.bytes); }
        std::vector<lend_fit> f;
        CHECK(!lend_pack(pieces, bytes, 256, f));
        return true;
    }
    if (!check_rank(cls, p) || !check_regions(cls, p, pieces, 256)) { return false; }
    n_l1_plans += p.victims.empty() ? 0 : 1;
    n_untied += p.untied > 0 ? 1 : 0;
    // the fewest: one victim less does not fit (unless none was needed)
    if (!p.victims.empty()) {
        std::vector<int> fewer = p.evict;
        const auto & last = p.victims.back();
        fewer[(size_t) s.layer_class[(size_t) last.layer]]--;
        std::vector<size_t> bytes;
        for (const auto & r : lend_regions_of(cls, fewer)) { bytes.push_back(r.bytes); }
        std::vector<lend_fit> f;
        CHECK(!lend_pack(pieces, bytes, 256, f));
    }
    if (!apply_evacuation(s, p)) { return false; }
    // a second lend over the held slots only adds the next lowest residents
    std::vector<size_t> more = pieces;
    more.push_back((size_t) (1 + rng() % 300000));
    const auto cls2 = s.classes(p.limit);
    std::vector<int> held(cls2.size());
    for (size_t c = 0; c < held.size(); ++c) { held[c] = s.slots[c] - p.limit[c]; }
    const lend_l1_plan p2 = plan_l1_lend(cls2, more, 256, &held);
    if (p2.ok) {
        n_l1_regrow++;
        for (size_t c = 0; c < held.size(); ++c) { CHECK(p2.evict[c] >= held[c]); }
        if (!check_rank(cls2, p2) || !check_regions(cls2, p2, more, 256)) { return false; }
    }
    return true;
}

// the checker is not blind: a victim set that keeps a lower ranked resident is caught
bool test_checker_sees_wrong_rank() {
    lend_class c;
    c.kind_bytes = { 4096, 4096, 4096 };
    c.guard = { 0, 0, 0 };
    c.slots = 4;
    for (int e = 0; e < 4; ++e) {
        lend_resident r; r.layer = 0; r.expert = e; r.slot = e; r.score = (uint64_t) (10 + e);
        c.residents.push_back(r);
    }
    lend_l1_plan wrong;
    wrong.victims = { c.residents[2] };   // score 12 out while 10 and 11 stay
    const int before = g_fail;
    g_quiet = true;
    const bool caught = !check_rank({ c }, wrong);
    g_quiet = false;
    g_fail = before;
    CHECK(caught);
    const lend_l1_plan right = plan_l1_lend({ c }, { 4096 }, 256);
    CHECK(right.ok && right.victims.size() == 1 && right.victims[0].expert == 0);
    return true;
}

// the eviction bound packs the pieces into the regions of each kind: more slots than the bytes over the slot stride
bool test_evict_bound_packing() {
    lend_class c;
    c.kind_bytes = { 1024, 1024, 2048 };   // a slot of 4096 bytes
    c.guard = { 0, 0, 0 };
    c.slots = 10;
    c.residents.resize(8);
    // 2560 bytes are less than one slot of 4096, but one piece needs a region of 2560: two slots of the 2048 kind
    CHECK(lend_evict_bound({ c }, { 2560 }, 256) == std::vector<int>({ 2 }));
    // three pieces of 1024: one slot holds one in each kind
    CHECK(lend_evict_bound({ c }, { 1024, 1024, 1024 }, 256) == std::vector<int>({ 1 }));
    // nothing to lend, and more than every resident can hold
    CHECK(lend_evict_bound({ c }, { 0 }, 256) == std::vector<int>({ 0 }));
    CHECK(lend_evict_bound({ c }, { 9*2048 }, 256) == std::vector<int>({ 8 }));
    return true;
}

// The last tie: a cold joint cache (every score 0). The draft's class ranks first (its layers are the later ones), but
// the target's regions hold every piece: the plan gives the draft's victims back and evicts the target's only.
bool test_untie_example() {
    // the target: 20 slots of 3 x 4 KiB on layer 0; the draft: 4 slots of 3 x 2 KiB on layer 1 (ranks first on a tie)
    auto make = [](size_t tb, int ts, int tl, size_t db, int ds, int dl, uint64_t tscore) {
        lend_class t, d;
        t.kind_bytes = { tb, tb, tb }; t.guard = { 0, 0, 0 }; t.slots = ts;
        d.kind_bytes = { db, db, db }; d.guard = { 0, 0, 0 }; d.slots = ds;
        for (int e = 0; e < ts; ++e) { lend_resident r; r.layer = tl; r.expert = e; r.slot = e; r.score = tscore; t.residents.push_back(r); }
        for (int e = 0; e < ds; ++e) { lend_resident r; r.layer = dl; r.expert = e; r.slot = e; d.residents.push_back(r); }
        return std::vector<lend_class>{ t, d };
    };
    // one piece of 3 target slots, larger than the whole draft region: the greedy rank takes the 4 draft slots first,
    // then 3 of the target; the tie gives the draft's back
    const lend_l1_plan p = plan_l1_lend(make(4096, 20, 0, 2048, 4, 1, 0), { 3*4096 }, 256);
    CHECK(p.ok && p.untied == 4 && p.evict == std::vector<int>({ 3, 0 }) && p.victims.size() == 3);
    for (const auto & v : p.victims) { CHECK(v.layer == 0); }
    // different scores are no tie: the draft (score 0) still leaves before the target (score 1)
    const lend_l1_plan q = plan_l1_lend(make(4096, 20, 0, 2048, 4, 1, 1), { 3*4096 }, 256);
    CHECK(q.ok && q.untied == 0 && q.evict == std::vector<int>({ 3, 4 }));
    // a piece only the draft's class holds (the target first on a tie now): the target's victims are given back
    const lend_l1_plan u = plan_l1_lend(make(2048, 4, 1, 8192, 4, 0, 0), { 3*4096 }, 256);
    CHECK(u.ok && u.untied == 4 && u.evict == std::vector<int>({ 0, 2 }));
    return true;
}

// P6: the lends of the 2026-10-15/17 logs, replayed with their pieces as reconstructed from the model shapes and the
// logged sums: DeepSeek-V4.1 Flash (12.65625 MiB experts, 4.21875 MiB per kind) and its DSpark draft
// in one joint cache. Every score 0 (cold); zero guards of 512 bytes. Prints the slots each lend takes.
static int replay_lend(const char * what, const std::vector<int> & slots, const std::vector<size_t> & kind,
                       const std::vector<int> & layer0, const std::vector<size_t> & pieces, int expect) {
    std::vector<lend_class> cls(slots.size());
    for (size_t c = 0; c < cls.size(); ++c) {
        cls[c].kind_bytes = { kind[c], kind[c], kind[c] };
        cls[c].guard = { 512, 512, 512 };
        cls[c].slots = slots[c];
        for (int s = 0; s < slots[c]; ++s) {
            lend_resident r; r.layer = layer0[c] + s/384; r.expert = s%384; r.slot = s;
            cls[c].residents.push_back(r);
        }
    }
    const lend_l1_plan p = plan_l1_lend(cls, pieces, 256);
    int n = 0;
    std::string per;
    for (size_t c = 0; c < cls.size(); ++c) { n += p.evict[c]; per += (c ? "/" : "") + std::to_string(p.evict[c]); }
    double a = 0, b = 0;
    for (size_t x : pieces) { a += x/1048576.0; }
    for (size_t c = 0; c < cls.size(); ++c) { b += p.evict[c]*3.0*kind[c]/1048576.0; }
    printf("test-expert-lend: replay %s: %zu pieces %.1f MiB, %d slots (%s per class) %.1f MiB, %d given back\n", what,
           pieces.size(), a, n, per.c_str(), b, p.untied);
    CHECK(p.ok && (expect < 0 || n == expect));
    return n;
}

static const size_t K41 = 4423680;   // DeepSeek-V4.1 Flash, one kind of an expert
static const std::vector<size_t> P_SINGLE = { 4905779200, 1226444800, 122644480, 1226444800, 509619712, 905969664, 1006632960, 1698693376 };
static const std::vector<size_t> P_JOINT  = { 598753280, 149688320, 14968832, 149688320, 126672384, 226492416, 251658240, 1698693376, 898129920 };

bool test_replay() {
    // single model, 59885 tokens, partial: 1109 slots (the logs of d64-e and t1)
    replay_lend("DS4.1F 59885 tokens", { 2000 }, { K41 }, { 0 }, P_SINGLE, 1109);
    // the HC state (80 KiB per token) lent per ubatch of 8192 (7 of 8192, one of 2541 tokens): 883
    auto split_hc = [](const std::vector<size_t> & p, int64_t n, int64_t ub) {
        std::vector<size_t> out;
        for (int64_t r0 = 0; r0 < n; r0 += ub) { out.push_back((size_t) std::min(ub, n - r0)*80*1024); }
        out.insert(out.end(), p.begin() + 1, p.end());
        return out;
    };
    auto split_stage = [](std::vector<size_t> p) {
        for (size_t i = 0; i < p.size(); ++i) {
            if (p[i] == (size_t) 128*3*K41 + 256) {
                p.erase(p.begin() + (ptrdiff_t) i);
                p.insert(p.begin() + (ptrdiff_t) i, { (size_t) 64*3*K41 + 256, (size_t) 64*3*K41 + 256 });
                break;
            }
        }
        return p;
    };
    replay_lend("DS4.1F 59885 tokens, HC state per ubatch", { 2000 }, { K41 }, { 0 }, split_hc(P_SINGLE, 59885, 8192), 883);
    replay_lend("DS4.1F 59885 tokens, HC state per ubatch, two staging pieces", { 2000 }, { K41 }, { 0 },
                split_stage(split_hc(P_SINGLE, 59885, 8192)), -1);
    // joint (J-on), 7309 tokens, exact: the log took 385 target + 30 draft slots; the tie gives the 30 back
    replay_lend("joint 7309 tokens", { 1427, 306, 30 }, { K41, 2949120, 5898240 }, { 100, 0, 200 }, P_JOINT, 385);
    // the two staging buffers as two pieces of 64 experts + 256 bytes each (in place of one of 128): 57 target slots less
    std::vector<size_t> joint2 = P_JOINT;
    joint2.erase(joint2.begin() + 7);
    joint2.insert(joint2.begin() + 7, { (size_t) 64*3*K41 + 256, (size_t) 64*3*K41 + 256 });
    replay_lend("joint 7309 tokens, two staging pieces", { 1427, 306, 30 }, { K41, 2949120, 5898240 }, { 100, 0, 200 }, joint2, 328);
    replay_lend("joint 7309 tokens, two staging pieces, HC state per ubatch of 2048", { 1427, 306, 30 },
                { K41, 2949120, 5898240 }, { 100, 0, 200 }, split_hc(joint2, 7309, 2048), -1);
    return true;
}

// ---- P2: the width rule: the fewest passes the L1 holds, equal shares, every piece in L1 ----
bool test_width(uint32_t seed) {
    std::mt19937 rng(seed);
    sim_l1 s = make_l1(rng, 1 + (int) (rng() % 2), 8, 64, false);
    const std::vector<int> limit = s.slots;
    const auto cls = s.classes(limit);
    const int64_t C = rng() % 3 == 0 ? 64 + (int64_t) (rng() % 4)*32 : 512 + (int64_t) (rng() % 4)*512;   // the context's ubatch
    const int64_t N = C + 1 + (int64_t) (rng() % (uint32_t) (30*C));                                    // more than one ubatch
    const int64_t unit = std::min<int64_t>(C, 256);
    const size_t per_row[4] = { 32 + rng() % 64, 32 + rng() % 64, 16 + rng() % 32, 64 + rng() % 128 };   // edges, y, T
    const size_t fixed = rng() % 2 ? (size_t) (rng() % 200000) : 0;                                      // the op's scratch
    auto request = [&](int64_t w) {
        std::vector<size_t> pc;
        for (size_t b : per_row) { pc.push_back(b*(size_t) w); }
        pc.push_back(fixed);
        return pc;
    };
    int n_fits = 0;
    auto fits = [&](int64_t w) { n_fits++; return plan_l1_lend(cls, request(w), 256).ok; };
    const lend_width r = lend_width_rule(N, C, unit, 0, fits);
    const int64_t K = (N + C - 1)/C;
    CHECK(n_fits <= 64);   // a search, not a scan
    if (r.P == 0) {
        // nothing fits, or not fewer passes than the ubatches
        if (r.W_max > 0) { CHECK(plan_l1_lend(cls, request(r.W_max), 256).ok); }
        return true;
    }
    // every piece of a pass in L1 (nothing in host memory: the plan has L1 regions only)
    const lend_l1_plan p = plan_l1_lend(cls, request(r.W), 256);
    CHECK(p.ok);
    for (const auto & f : p.fits) { CHECK(f.region >= -1 && f.region < (int) p.regions.size()); }
    if (r.P == 1) {
        CHECK(r.W == N && r.C == C && r.W_max == N);
        return true;
    }
    n_multi_pass++;
    // W_max is the widest: a unit more does not fit
    CHECK(r.W_max % unit == 0 && plan_l1_lend(cls, request(r.W_max), 256).ok);
    CHECK(r.W_max + unit >= N || !plan_l1_lend(cls, request(r.W_max + unit), 256).ok);
    // the fewest passes: ceil(N / W_max), more only when the rounding pushed those passes over what fits
    const int64_t P0 = (N + r.W_max - 1)/r.W_max;
    CHECK(r.P >= P0 && r.P < K);
    for (int64_t P = P0; P < r.P; ++P) {
        const int64_t w0 = (N + P - 1)/P, g = (w0 + C - 1)/C;
        const int64_t c = std::min<int64_t>(C, ((w0 + g - 1)/g + unit - 1)/unit*unit);
        CHECK(!plan_l1_lend(cls, request(g*c), 256).ok);
    }
    // equal shares: ubatches of at most the context's, in units, a pass of g of them covers ceil(N / P)
    CHECK(r.C <= C && (r.C % unit == 0 || r.C == C) && r.W == r.g*r.C);
    CHECK(r.W <= r.W_max || r.P > P0);
    CHECK(r.P*r.W >= N && r.W < (N + r.P - 1)/r.P + r.g*unit);
    // the ubatches of C cover N in exactly P passes of g
    const int64_t K2 = (N + r.C - 1)/r.C;
    CHECK((K2 + r.g - 1)/r.g == r.P);
    // a narrower width never needs more L1
    const lend_l1_plan narrow = plan_l1_lend(cls, request(std::max<int64_t>(1, r.W - unit)), 256);
    CHECK(narrow.ok && narrow.victims.size() <= p.victims.size());
    return true;
}

// the examples of the rule's decision (2026-10-15): N 32445, W_max 27648 -> two passes of 16223 (16384 in units of
// 256); W_max 10000 -> four passes of 8112 (8192)
bool test_width_examples() {
    for (const int64_t cap : { (int64_t) 27648, (int64_t) 10000 }) {
        const lend_width r = lend_width_rule(32445, 4096, 256, 0, [&](int64_t w) { return w <= cap; });
        if (cap == 27648) {
            CHECK(r.W_max == 27648 && r.P == 2 && r.g == 4 && r.C == 4096 && r.W == 16384);
        } else {
            CHECK(r.W_max == 9984 && r.P == 4 && r.g == 2 && r.C == 4096 && r.W == 8192);
        }
    }
    // a wide ubatch is split again: 59885 tokens in ubatches of 24576, at most 30000 per pass -> two passes of two
    // ubatches of 15104
    const lend_width r = lend_width_rule(59885, 24576, 256, 0, [&](int64_t w) { return w <= 30208; });
    CHECK(r.P == 2 && r.g == 2 && r.C == 15104 && r.W == 30208);
    // the rounding pushes the pass over the cap: one pass more (three passes are not fewer than three ubatches)
    const lend_width q = lend_width_rule(59885, 24576, 256, 0, [&](int64_t w) { return w <= 30000; });
    CHECK(q.P == 0 && q.W_max == 29952);
    // all fit: one pass in the context's ubatches; forced widths replace the search
    const lend_width one = lend_width_rule(1000, 128, 128, 0, [&](int64_t) { return true; });
    CHECK(one.P == 1 && one.C == 128 && one.W == 1000);
    const lend_width forced = lend_width_rule(1000, 128, 128, 256, [&](int64_t) { return false; });
    CHECK(forced.P == 4 && forced.g == 2 && forced.C == 128);
    return true;
}

// ---- P3: the restore space ----

// The space of one class as the kernels hold it, and the oracle checks of one launch.
struct sim_space {
    int first = 0, n = 0;
    std::vector<int32_t> own, pin;
    std::vector<uint32_t> stamp;
    uint32_t clock = 0;
    void open(int f, int count) { first = f; n = count; own.assign((size_t) n, -1); pin.assign((size_t) n, 0); stamp.assign((size_t) n, 0); clock = 0; }
    int pinned() const { int k = 0; for (int j = 0; j < n; ++j) { k += own[(size_t) j] >= 0 && pin[(size_t) j] ? 1 : 0; } return k; }
};

// One launch through lend_space_step, checked against what the policy promises: wanted experts first; an empty slot
// before any occupant is replaced; a replaced occupant is the least recently used unpinned slot this launch does not
// read; a pinned slot is never replaced; an LRU copy only after a miss within the ghost window. Applies the publish.
bool space_launch(sim_space & sp, int layer, int E, const std::vector<int32_t> & ids, int rows, int used,
                  std::vector<std::vector<int32_t>> & dev, std::vector<std::vector<int32_t>> & want,
                  std::vector<std::vector<uint32_t>> & last_miss, std::vector<uint32_t> & step, uint32_t ghost,
                  const std::function<bool(int)> & readable, int & n_copies, int & n_evict, int & n_wanted_copies) {
    std::vector<int32_t> own0 = sp.own, pin0 = sp.pin;
    std::vector<uint32_t> stamp0 = sp.stamp;
    std::vector<uint32_t> lm0 = last_miss[(size_t) layer];
    const uint32_t st = step[(size_t) layer] + 1, t = sp.clock + 1;
    // the slots this launch reads (hits) are not victims
    std::set<int> hit_j;
    for (int32_t e : ids) {
        const int32_t g = dev[(size_t) layer][(size_t) e];
        if (g >= sp.first && g < sp.first + sp.n && own0[(size_t) (g - sp.first)] == layer*E + e) { hit_j.insert(g - sp.first); }
    }
    lend_space_copy out[lend_space_max_ids];
    const int n = lend_space_step(ids.data(), rows, used, used, E, layer, dev[(size_t) layer].data(), want[(size_t) layer].data(),
        last_miss[(size_t) layer].data(), &step[(size_t) layer], ghost, sp.first, sp.n, sp.own.data(), sp.pin.data(),
        sp.stamp.data(), &sp.clock, [&](int e) { return readable(e); }, out, lend_space_max_ids);
    CHECK(sp.clock == t && step[(size_t) layer] == st);
    int empties = 0;
    for (int j = 0; j < sp.n; ++j) { empties += own0[(size_t) j] < 0 ? 1 : 0; }
    bool lru_seen = false;
    std::set<int> taken;
    for (int i = 0; i < n; ++i) {
        const lend_space_copy & c = out[i];
        CHECK(std::find(ids.begin(), ids.end(), c.expert) != ids.end());
        CHECK(dev[(size_t) layer][(size_t) c.expert] < 0 && readable(c.expert));
        CHECK(taken.insert(c.j).second);
        if (!c.wanted) {
            lru_seen = true;
            CHECK(lend_space_admit(false, lm0[(size_t) c.expert], st, ghost, true, lend_rows_with(ids.data(), rows, used, used, c.expert)));
        } else {
            CHECK(!lru_seen);   // the wanted first
        }
        // an empty slot first, in index order
        if (i < empties) {
            CHECK(c.evicted < 0 && own0[(size_t) c.j] < 0);
        } else {
            CHECK(c.evicted >= 0 && !pin0[(size_t) c.j] && hit_j.count(c.j) == 0);
            // the least recently used of the unpinned slots not read now and not taken before it
            for (int j = 0; j < sp.n; ++j) {
                if (own0[(size_t) j] >= 0 && !pin0[(size_t) j] && hit_j.count(j) == 0 && taken.count(j) == 0) {
                    CHECK(stamp0[(size_t) c.j] <= stamp0[(size_t) j]);
                }
            }
            ++n_evict;
            const int ol = c.evicted/E, oe = c.evicted%E;
            CHECK(dev[(size_t) ol][(size_t) oe] == sp.first + c.j);
            dev[(size_t) ol][(size_t) oe] = -1;
        }
        n_wanted_copies += c.wanted;
        ++n_copies;
    }
    // a wanted readable miss that got no slot: every slot was pinned or read now
    for (int32_t e : ids) {
        if (dev[(size_t) layer][(size_t) e] < 0 && want[(size_t) layer][(size_t) e] >= 0 && readable(e)) {
            bool placed = false;
            for (int i = 0; i < n; ++i) { placed = placed || out[i].expert == e; }
            if (!placed) {
                for (int j = 0; j < sp.n; ++j) { CHECK(sp.own[(size_t) j] >= 0 && (sp.pin[(size_t) j] || sp.stamp[(size_t) j] == t)); }
            }
        }
    }
    // the publish of the copy kernel's last block
    for (int i = 0; i < n; ++i) { dev[(size_t) layer][(size_t) out[i].expert] = sp.first + out[i].j; }
    // the tables and the space agree; a slot holds one expert
    for (int j = 0; j < sp.n; ++j) {
        const int32_t o = sp.own[(size_t) j];
        if (o >= 0) { CHECK(dev[(size_t) (o/E)][(size_t) (o%E)] == sp.first + j); }
    }
    return true;
}

bool test_restore(uint32_t seed) {
    std::mt19937 rng(seed);
    sim_l1 s = make_l1(rng, 1, 6, 40, seed % 2 == 0);
    lend_book book;
    book.slots = s.slots;
    book.limit = s.slots;
    std::vector<size_t> pieces = { (size_t) (1 + rng() % 200000), (size_t) (1 + rng() % 200000), (size_t) (rng() % 100000) };
    const lend_l1_plan p = plan_l1_lend(s.classes(book.limit), pieces, 256);
    if (!p.ok || p.victims.empty()) {
        return true;
    }
    if (!apply_evacuation(s, p)) { return false; }
    CHECK(book.hold(p.limit, p.victims, &book, true));
    CHECK(!book.hold(p.limit, {}, &s, false));                     // one lease at a time
    const uint64_t epoch = book.epoch;
    const int E = s.experts;
    std::vector<uint8_t> want_plan((size_t) s.layers*E, 0);
    for (const auto & v : p.victims) {
        if (rng() % 3) { want_plan[(size_t) v.layer*E + v.expert] = 1; }
    }
    CHECK(book.open_restore(s.layer_class, &want_plan, E));
    CHECK(book.state == lend_state::restore && book.owner == nullptr);
    sim_space sp;
    sp.open(book.limit[0], book.slots[0] - book.limit[0]);
    std::vector<std::vector<int32_t>> dev = s.table;
    std::vector<std::vector<int32_t>> want((size_t) s.layers, std::vector<int32_t>((size_t) E, -1));
    for (const auto & r : book.restore) { want[(size_t) r.layer][(size_t) r.expert] = 1; }
    std::vector<std::vector<uint32_t>> last_miss((size_t) s.layers, std::vector<uint32_t>((size_t) E, lend_never));
    std::vector<uint32_t> step((size_t) s.layers, 0);
    const uint32_t ghost = seed % 4 == 0 ? 0 : 8;
    // an expert without a source (no address, no host slot) is never copied
    const int unreadable = (int) (rng() % E);
    auto readable = [&](int e) { return e != unreadable; };
    // no refill: no victim is in a table before a decode launch reads it
    for (const auto & v : p.victims) { CHECK(dev[(size_t) v.layer][(size_t) v.expert] < 0); }
    int n_copies = 0, n_evict = 0, n_wanted = 0;
    for (int tok = 0; tok < 300; ++tok) {
        const int used = 4, rows = 1 + (int) (rng() % 2);
        for (int l = 0; l < s.layers; ++l) {
            std::vector<int32_t> ids((size_t) rows*used);
            for (auto & id : ids) { id = (int32_t) (rng() % (uint32_t) E); }
            if (rows*used >= 2) { ids[1] = ids[0]; }   // a duplicate in a launch is copied once
            if (!space_launch(sp, l, E, ids, rows, used, dev, want, last_miss, step, ghost, readable, n_copies, n_evict, n_wanted)) {
                return false;
            }
        }
    }
    // the LRU part shrinks by one slot per pinned expert; pinned ones are exactly the wanted copies still there
    int wanted_back = 0;
    for (const auto & r : book.restore) {
        const int32_t g = dev[(size_t) r.layer][(size_t) r.expert];
        if (g >= sp.first) { ++wanted_back; CHECK(sp.pin[(size_t) (g - sp.first)] == 1); }
    }
    CHECK(sp.pinned() == wanted_back && sp.pinned() == n_wanted);
    n_restores += n_wanted > 0 ? 1 : 0;
    n_lru_evictions += n_evict > 0 ? 1 : 0;
    n_lru_full += sp.pinned() == sp.n ? 1 : 0;
    // a later lend takes the space back: the copies leave the tables (the mirror), the lease is new
    std::vector<int> held(book.slots.size());
    for (size_t c = 0; c < held.size(); ++c) { held[c] = book.slots[c] - book.limit[c]; }
    const lend_l1_plan again = plan_l1_lend(s.classes(book.limit), pieces, 256, &held);
    CHECK(again.ok && again.victims.empty());                     // the same pieces need no new eviction
    CHECK(book.hold(again.limit, again.victims, &s, false));
    CHECK(book.epoch == epoch + 1 && book.restore.empty() && !book.explicit_);
    // an install under the lease keeps the wanted copies at their slots; the rest of the wanted set waits
    CHECK(book.open_restore(s.layer_class, &want_plan, E));
    std::vector<lend_restore> kept;
    for (const auto & r : book.restore) {
        if (rng() % 2) { lend_restore k = r; k.slot = sp.first + (int) kept.size(); kept.push_back(k); }
    }
    const auto again_restore = plan_restore(book.victims, s.layer_class, &kept, &want_plan, E);
    for (const auto & r : again_restore) {
        for (const auto & k : kept) { CHECK(!(k.layer == r.layer && k.expert == r.expert)); }
        CHECK(want_plan[(size_t) r.layer*E + r.expert] == 1);
    }
    CHECK(again_restore.size() + kept.size() == book.restore.size());
    n_sticky += kept.empty() ? 0 : 1;
    book.clear();
    CHECK(book.state == lend_state::none && book.limit == book.slots);
    return true;
}

// The LRU part grows and shrinks: four lent slots, wanted A and B. Misses C, D, F fill the empty slots as LRU copies
// (ghost 0); A takes the last empty slot; B finds none and replaces the least recently used copy; the LRU width is
// 4 - pinned; with every slot pinned a miss gets no slot.
bool test_space_example() {
    const int E = 8, L = 1;
    sim_space sp;
    sp.open(10, 4);
    std::vector<std::vector<int32_t>> dev((size_t) L, std::vector<int32_t>((size_t) E, -1));
    std::vector<std::vector<int32_t>> want((size_t) L, std::vector<int32_t>((size_t) E, -1));
    const int A = 0, B = 1, C = 2, D = 3, F = 4, G = 5, H = 6;
    want[0][A] = want[0][B] = 1;
    std::vector<std::vector<uint32_t>> lm((size_t) L, std::vector<uint32_t>((size_t) E, lend_never));
    std::vector<uint32_t> step((size_t) L, 0);
    auto all = [](int) { return true; };
    int nc = 0, ne = 0, nw = 0;
    auto run = [&](std::vector<int32_t> ids) { return space_launch(sp, 0, E, ids, 1, (int) ids.size(), dev, want, lm, step, 0, all, nc, ne, nw); };
    CHECK(run({ C }) && run({ D }) && run({ F }));
    CHECK(nc == 3 && ne == 0 && sp.pinned() == 0);              // LRU width 4
    CHECK(run({ A }));                                          // the empty slot, nothing replaced
    CHECK(ne == 0 && sp.pinned() == 1 && dev[0][A] == 13);
    CHECK(run({ D }));                                          // a hit: D is the most recent copy now
    CHECK(run({ B }));                                          // no empty slot: C, the least recently used, goes
    CHECK(ne == 1 && dev[0][C] < 0 && dev[0][B] == 10 && sp.pinned() == 2);   // LRU width 2
    CHECK(run({ G, D }));                                       // D is read now: F goes, not D
    CHECK(dev[0][F] < 0 && dev[0][D] >= 0 && dev[0][G] >= 0);
    // the ghost filter: with ghost 2, H is admitted on its second miss only
    nc = 0;
    CHECK(space_launch(sp, 0, E, { H }, 1, 1, dev, want, lm, step, 2, all, nc, ne, nw) && nc == 0 && dev[0][H] < 0);
    CHECK(space_launch(sp, 0, E, { H }, 1, 1, dev, want, lm, step, 2, all, nc, ne, nw) && nc == 1 && dev[0][H] >= 0);
    // two rows of one launch selecting an expert are its second access in that step: admitted at the first miss
    nc = 0;
    CHECK(dev[0][7] < 0 && lm[0][7] == lend_never);
    CHECK(space_launch(sp, 0, E, { 7, 7 }, 2, 1, dev, want, lm, step, 2, all, nc, ne, nw) && nc == 1 && dev[0][7] >= 0);
    CHECK(lend_space_admit(false, lend_never, 5, 8, true, 2) && !lend_space_admit(false, lend_never, 5, 8, true, 1));
    CHECK(!lend_space_admit(false, lend_never, 5, 8, false, 2));   // LRU off: still only a wanted expert
    // every slot pinned: nothing more is copied, the LRU width is 0
    want[0][C] = want[0][F] = 1;
    CHECK(run({ C }) && run({ F }));
    CHECK(sp.pinned() == 4);
    const int before = nc;
    CHECK(run({ D, G, H }) && nc == before);
    return true;
}

// A cold cache (no decode plan) wants nothing: the evicted experts come back only as LRU copies, never pinned.
bool test_space_cold() {
    lend_book book;
    book.slots = { 6 };
    book.limit = { 6 };
    std::vector<lend_resident> v(3);
    for (int i = 0; i < 3; ++i) { v[(size_t) i].layer = 0; v[(size_t) i].expert = i; }
    CHECK(book.hold({ 3 }, v, &book, true));
    const std::vector<uint8_t> none;
    CHECK(book.open_restore({ 0 }, &none, 8) && book.restore.empty());
    // the LRU part off (RANMA_EXPERT_LEND_LRU=0): only a wanted expert is admitted
    CHECK(lend_space_admit(true, lend_never, 5, 8, false) && !lend_space_admit(false, 4, 5, 8, false) && lend_space_admit(false, 4, 5, 8, true));
    CHECK(!lend_space_admit(false, lend_never, 5, 0, false) && lend_space_admit(false, lend_never, 5, 0, true));
    return true;
}

// P3: only the victims the installed decode plan keeps in L1 get a lent slot; a cold cache (no plan) restores nothing
bool test_restore_want(uint32_t seed) {
    std::mt19937 rng(seed);
    sim_l1 s = make_l1(rng, 1 + (int) (rng() % 2), 6, 40, false);
    const lend_l1_plan p = plan_l1_lend(s.classes(s.slots), { (size_t) (1 + rng() % 300000) }, 256);
    if (!p.ok || p.victims.empty()) {
        return true;
    }
    std::vector<uint8_t> want((size_t) s.layers*s.experts, 0);
    std::set<std::pair<int, int>> wanted;
    for (const auto & v : p.victims) {
        if (rng() % 2) {
            want[(size_t) v.layer*s.experts + v.expert] = 1;
            wanted.insert({v.layer, v.expert});
        }
    }
    for (int pass = 0; pass < 3; ++pass) {
        lend_book book;
        book.slots = s.slots;
        book.limit = s.slots;
        CHECK(book.hold(p.limit, p.victims, &book, true));
        const std::vector<uint8_t> none;
        CHECK(book.open_restore(s.layer_class, pass == 0 ? &want : pass == 1 ? &none : nullptr, s.experts));
        if (pass == 0) {
            CHECK(book.restore.size() == wanted.size());
            for (const auto & r : book.restore) { CHECK(wanted.count({r.layer, r.expert}) == 1); }
        } else if (pass == 1) {
            CHECK(book.restore.empty());
        } else {
            CHECK(book.restore.size() == p.victims.size());
        }
    }
    n_wanted += wanted.size() < p.victims.size() && !wanted.empty() ? 1 : 0;
    return true;
}

// A long lend, the decode fills its restore space, then a short lend: it takes only the top slots it needs (empty slots,
// then LRU copies, then the lowest ranked pinned copies), the copies that stay move below that top, the space goes on
// under it and after the lend; a lend that needs more than the space is refused (the caller drops the space).
bool test_space_lend(uint32_t seed) {
    std::mt19937 rng(seed);
    sim_l1 s = make_l1(rng, 1, 6, 40, seed % 2 == 0);
    lend_book book;
    book.slots = s.slots;
    book.limit = s.slots;
    book.top = s.slots;
    const std::vector<size_t> big = { (size_t) (100000 + rng() % 200000), (size_t) (100000 + rng() % 200000) };
    const lend_l1_plan p = plan_l1_lend(s.classes(book.limit), big, 256);
    if (!p.ok || p.victims.size() < 4) {
        return true;
    }
    if (!apply_evacuation(s, p)) { return false; }
    CHECK(book.hold(p.limit, p.victims, &book, true));
    const int E = s.experts;
    std::vector<uint8_t> want_plan((size_t) s.layers*E, 0);
    for (const auto & v : p.victims) { want_plan[(size_t) v.layer*E + v.expert] = 1; }
    CHECK(book.open_restore(s.layer_class, &want_plan, E));
    sim_space sp;
    sp.open(book.limit[0], book.slots[0] - book.limit[0]);
    std::vector<std::vector<int32_t>> dev = s.table;
    std::vector<std::vector<int32_t>> want((size_t) s.layers, std::vector<int32_t>((size_t) E, -1));
    for (const auto & r : book.restore) { want[(size_t) r.layer][(size_t) r.expert] = 1; }
    std::vector<std::vector<uint32_t>> last_miss((size_t) s.layers, std::vector<uint32_t>((size_t) E, lend_never));
    std::vector<uint32_t> step((size_t) s.layers, 0);
    auto all = [](int) { return true; };
    int nc = 0, ne = 0, nw = 0;
    auto decode = [&](int tokens) {
        for (int tok = 0; tok < tokens; ++tok) {
            for (int l = 0; l < s.layers; ++l) {
                std::vector<int32_t> ids(4);
                for (auto & id : ids) { id = (int32_t) (rng() % (uint32_t) E); }
                if (!space_launch(sp, l, E, ids, 1, 4, dev, want, last_miss, step, 2, all, nc, ne, nw)) { return false; }
            }
        }
        return true;
    };
    if (!decode(1 + (int) (rng() % 40))) { return false; }
    // the copies of the space now
    std::vector<lend_restore> copies;
    for (int j = 0; j < sp.n; ++j) {
        if (sp.own[(size_t) j] < 0) { continue; }
        lend_restore r;
        r.layer = sp.own[(size_t) j]/E; r.expert = sp.own[(size_t) j]%E; r.cls = 0; r.slot = sp.first + j; r.pin = sp.pin[(size_t) j];
        copies.push_back(r);
    }
    const int pinned_before = sp.pinned();
    // a short lend: a piece of one to a few slots
    lend_class kinds = s.classes(book.limit)[0];
    kinds.slots = book.slots[0];
    const size_t slot_bytes = kinds.kind_bytes[0];
    const std::vector<size_t> small = { slot_bytes*(1 + rng() % 3) };
    const lend_space_take t = plan_space_lend({ kinds }, book.limit, copies, s.score, E, small, 256);
    if (!t.ok) {
        return true;   // the space is too small for even the short piece
    }
    const int top = t.plan.limit[0];
    CHECK(top >= book.limit[0] && top < book.slots[0]);
    // the fewest slots: one slot fewer does not hold the pieces
    {
        std::vector<int> fewer = { book.slots[0] - top - 1 };
        std::vector<size_t> bytes;
        for (const auto & r : lend_regions_of({ kinds }, fewer)) { bytes.push_back(r.bytes); }
        std::vector<lend_fit> f;
        CHECK(!lend_pack(small, bytes, 256, f));
    }
    // the slots it takes: the empty ones and the dropped copies, nothing more
    const int empties = sp.n - (int) copies.size();
    CHECK((book.slots[0] - top) == std::min(empties, book.slots[0] - top) + (int) t.dropped.size());
    CHECK(t.dropped.empty() || t.empty_taken[0] == empties);
    // drop order: every dropped copy ranks below every kept one (LRU before pinned, then the score)
    auto rank_of = [&](const lend_restore & r) {
        lend_resident x; x.layer = r.layer; x.expert = r.expert; x.score = s.score[(size_t) r.layer*E + r.expert]; x.tier = r.pin ? 1 : 0;
        return x;
    };
    for (const auto & d : t.dropped) {
        for (const auto & k : t.kept) { CHECK(!lend_rank_less(rank_of(k), rank_of(d))); }
    }
    CHECK(t.kept.size() + t.dropped.size() == copies.size());
    // the moves: from the taken top down into free slots of the space; every kept copy below the top, one per slot
    std::set<int> at;
    for (const auto & k : t.kept) {
        CHECK(k.slot >= book.limit[0] && k.slot < top);
        CHECK(at.insert(k.slot).second);
    }
    for (const auto & m : t.moves) { CHECK(m[1] >= top && m[2] < top); }
    // the pieces lie above the top
    for (const auto & r : t.plan.regions) { CHECK(r.begin >= (size_t) top*kinds.kind_bytes[(size_t) r.kind]); }
    int kept_pinned = 0;
    for (const auto & k : t.kept) { kept_pinned += k.pin; }
    int dropped_pinned = 0;
    for (const auto & d : t.dropped) { dropped_pinned += d.pin; }
    CHECK(kept_pinned + dropped_pinned == pinned_before);
    n_space_short += 1;
    n_space_pinned_kept += kept_pinned > 0 && dropped_pinned < pinned_before ? 1 : 0;
    // the lend over the space: the tables follow the moves and the drops
    CHECK(book.hold_space(t.plan.limit, &s, false));
    CHECK(book.state == lend_state::held && book.top[0] == top && book.limit[0] == sp.first);
    for (const auto & d : t.dropped) {
        dev[(size_t) d.layer][(size_t) d.expert] = -1;
        if (d.pin) { want[(size_t) d.layer][(size_t) d.expert] = 1; }
    }
    sim_space sp2;
    sp2.open(sp.first, sp.n);
    for (const auto & k : t.kept) {
        dev[(size_t) k.layer][(size_t) k.expert] = k.slot;
        sp2.own[(size_t) (k.slot - sp.first)] = k.layer*E + k.expert;
        sp2.pin[(size_t) (k.slot - sp.first)] = k.pin;
    }
    // the lend ends: the whole space again, the kept copies in place, the decode goes on
    book.state = lend_state::held;
    CHECK(book.open_restore(s.layer_class, &want_plan, E) && book.top[0] == book.slots[0]);
    sp = sp2;
    if (!decode(20)) { return false; }
    // a lend wider than the space is refused (the caller drops the space and evicts residents)
    const size_t widest = *std::max_element(kinds.kind_bytes.begin(), kinds.kind_bytes.end());
    std::vector<size_t> huge = { (size_t) (sp.n + 2)*widest };
    copies.clear();
    for (int j = 0; j < sp.n; ++j) {
        if (sp.own[(size_t) j] < 0) { continue; }
        lend_restore r;
        r.layer = sp.own[(size_t) j]/E; r.expert = sp.own[(size_t) j]%E; r.cls = 0; r.slot = sp.first + j; r.pin = sp.pin[(size_t) j];
        copies.push_back(r);
    }
    CHECK(!plan_space_lend({ kinds }, book.limit, copies, s.score, E, huge, 256).ok);
    return true;
}

// The staged service: with only kind 0 of the generation in place the restore space does not copy; with every kind (or
// the whole generation) it does.
bool test_space_staged() {
    const uint32_t seq = 7;
    const uint32_t only0[3] = { seq, seq - 1, seq - 1 }, two[3] = { seq, seq, seq - 1 }, all[3] = { seq, seq, seq };
    CHECK(!lend_space_sources_ready(seq - 1, only0, seq));
    CHECK(!lend_space_sources_ready(seq - 1, two, seq));
    CHECK(lend_space_sources_ready(seq - 1, all, seq));
    CHECK(lend_space_sources_ready(seq, only0, seq));
    return true;
}

// Two file-only evicted experts (no host slot, no address): A is wanted (the decode plan's), B is not. The SSD tier
// serves a launch of A, A, B from the file into its ring (each expert's three kinds read once), then the restore space
// copies A from the ring slot the tier just published and pins it, while B is read where the ring has it (host-direct).
// The next generation leaves A out of the demand (the restored bitmap); A's ring copy stays until its lease is done,
// and only then can a new read reuse that slot.
bool test_file_only_restore() {
    const int E = 4, A = 1, B = 2, C = 3;
    l2_class_ledger ring;
    ring.reset(1, E, { 0 }, { 2 });
    CHECK(ring.make_ring(0, 0) && ring.make_ring(0, 1));
    std::vector<std::vector<int32_t>> dev(1, std::vector<int32_t>((size_t) E, -1));
    std::vector<uint8_t> restored((size_t) E, 0);
    auto homed = [&](int, int e) { return dev[0][(size_t) e] >= 0; };
    // the publish kernel: the demand without the restored experts
    auto demand = [&](std::vector<int> ids) {
        std::vector<int> out;
        for (int e : ids) { if (!restored[(size_t) e]) { out.push_back(e); } }
        return out;
    };
    // generation 1: A, A, B
    const l2_service g1 = ring.service(0, demand({ A, A, B }), 1, homed);
    CHECK(g1.ok && g1.misses == 2 && g1.reads.size() == 6);
    int reads_a[3] = {0, 0, 0}, reads_b[3] = {0, 0, 0};
    for (const l2_read & r : g1.reads) { (r.expert == A ? reads_a : reads_b)[r.kind]++; }
    for (int k = 0; k < 3; ++k) { CHECK(reads_a[k] == 1 && reads_b[k] == 1); }
    const int slot_a = ring.slot_of(0, A), slot_b = ring.slot_of(0, B);
    CHECK(slot_a >= 0 && slot_b >= 0 && slot_a != slot_b);
    // the restore space: A from its ring slot (readable: the tier published its address), B not admitted
    sim_space sp;
    sp.open(8, 2);
    std::vector<int32_t> want((size_t) E, -1);
    want[(size_t) A] = 1;
    std::vector<uint32_t> last_miss((size_t) E, lend_never);
    uint32_t step = 0;
    auto readable = [&](int e) { return ring.slot_of(0, e) >= 0; };
    lend_space_copy out[lend_space_max_ids];
    const int32_t ids[3] = { A, A, B };
    const int n = lend_space_step(ids, 1, 3, 3, E, 0, dev[0].data(), want.data(), last_miss.data(), &step, 8, sp.first, sp.n,
        sp.own.data(), sp.pin.data(), sp.stamp.data(), &sp.clock, readable, out, lend_space_max_ids);
    CHECK(n == 1 && out[0].expert == A && out[0].wanted == 1);
    const int source = ring.slot_of(0, A);           // what the copy kernel reads: the current ring bytes of A
    CHECK(source == slot_a);
    dev[0][(size_t) A] = sp.first + out[0].j;
    restored[(size_t) A] = 1;
    CHECK(dev[0][(size_t) B] < 0 && ring.slot_of(0, B) == slot_b);   // B is read in the ring (host-direct)
    // generation 2: A is no longer demanded; B is a hit
    const l2_service g2 = ring.service(0, demand({ A, B }), 2, homed);
    CHECK(g2.ok && g2.misses == 0 && g2.hits == 1 && g2.reads.empty());
    CHECK(ring.slot_of(0, A) == slot_a);             // A's ring copy stays
    // C needs a slot: A's copy holds lease 1, B's lease 2; nothing is reusable before the GPU finished generation 1
    const l2_service g3 = ring.service(0, { C }, 3, homed);
    CHECK(!g3.ok);
    ring.set_done(0, 1);
    const l2_service g4 = ring.service(0, { C }, 3, homed);
    CHECK(g4.ok && g4.misses == 1 && ring.slot_of(0, C) == slot_a && ring.slot_of(0, A) < 0);
    CHECK(dev[0][(size_t) A] >= sp.first);            // A stays in L1
    return true;
}

bool test_pack_edges() {
    std::vector<lend_fit> f;
    CHECK(lend_pack({}, {}, 256, f));
    CHECK(lend_pack({ 0, 0 }, {}, 256, f) && f[0].region == -1);
    CHECK(lend_pack({ 1024 }, { 1024 }, 256, f) && f[0].region == 0 && f[0].offset == 0);
    CHECK(!lend_pack({ 1025 }, { 1024 }, 256, f));
    CHECK(lend_pack({ 100, 100 }, { 512 }, 256, f) && f[1].offset == 256);   // aligned
    CHECK(!lend_pack({ 300, 300 }, { 511 }, 256, f));
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 2000;
    test_pack_edges();
    test_checker_sees_wrong_rank();
    test_evict_bound_packing();
    test_untie_example();
    test_replay();
    test_width_examples();
    test_space_example();
    test_space_cold();
    test_space_staged();
    test_file_only_restore();
    int n = 0;
    for (int i = 0; i < seeds; ++i) {
        const uint32_t seed = 1000u + (uint32_t) i;
        n += test_l1(seed) ? 1 : 0;
        n += test_width(seed) ? 1 : 0;
        n += test_restore(seed) ? 1 : 0;
        n += test_restore_want(seed) ? 1 : 0;
        n += test_space_lend(seed) ? 1 : 0;
        if (g_fail > 0) {
            fprintf(stderr, "first failure at seed %u\n", seed);
            break;
        }
    }
    printf("test-expert-lend: %d checks over %d seeds, %d failures; paths: L1 evictions %d, L1 growth over held slots %d, "
        "widths of several passes %d, restore runs %d, sticky installs %d, eviction bounds over several classes %d, "
        "restores limited to the decode plan %d, LRU replacements %d, spaces filled by pinned experts %d, short lends over a space %d "
        "(keeping pinned copies %d)\n",
        n, seeds, g_fail, n_l1_plans, n_l1_regrow, n_multi_pass, n_restores, n_sticky, n_bounds, n_wanted, n_lru_evictions, n_lru_full,
        n_space_short, n_space_pinned_kept);
    printf("test-expert-lend: plans that gave victims of their last tie back (a class not needed) %d\n", n_untied);
    const bool covered = n_l1_plans > 0 && n_l1_regrow > 0 && n_multi_pass > 0 && n_restores > 0 && n_sticky > 0 && n_bounds > 0 &&
        n_wanted > 0 && n_lru_evictions > 0 && n_space_short > 0 && n_space_pinned_kept > 0;
    if (!covered) {
        printf("a path was never taken\n");
    }
    printf("%s\n", g_fail == 0 && covered ? "PASS" : "FAIL");
    return g_fail == 0 && covered ? 0 : 1;
}
