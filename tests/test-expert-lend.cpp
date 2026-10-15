// The prefill lend of the expert cache (ggml/src/ggml-cuda/expert-lend.h) against a simulated arena:
//   P1: the pieces evict the lowest ranked L1 residents first and take contiguous ranges at the top of each class;
//       nothing is lent from host memory;
//   P2: the width rule: the fewest passes whose pieces L1 holds (all of them in L1), in equal shares;
//   P3: the release does not refill; a decode launch restores an evicted expert only when it reads it, into its own
//       lent slot, and a later lend takes the restore space back.
//       lent slot, and a later lend takes the restore space back.

#include "expert-lend.h"
#include "expert-redraw.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ggml_cuda_expert;

static int g_fail = 0;
static bool g_quiet = false;
// the paths a run took (each must be taken at least once)
static int n_l1_plans = 0, n_l1_regrow = 0, n_multi_pass = 0, n_restores = 0, n_sticky = 0, n_wanted = 0;
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

// P1: the victims are the lowest ranked residents, over every class, and nothing ranked lower stays (by score: in
// the last tie the plan keeps the residents of a class the pieces do not need, whatever their layer and id)
static bool rank_below(const lend_resident & a, const lend_resident & b) {
    return a.score < b.score;
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
bool test_restore(uint32_t seed) {
    std::mt19937 rng(seed);
    sim_l1 s = make_l1(rng, 1 + (int) (rng() % 2), 6, 40, seed % 2 == 0);
    lend_book book;
    book.slots = s.slots;
    book.limit = s.slots;
    std::vector<size_t> pieces = { (size_t) (1 + rng() % 200000), (size_t) (1 + rng() % 200000), (size_t) (rng() % 100000) };
    const lend_l1_plan p = plan_l1_lend(s.classes(book.limit), pieces, 256);
    if (!p.ok) {
        return true;
    }
    const auto residents_before = s.table;
    if (!apply_evacuation(s, p)) { return false; }
    const void * ctx_a = &book, * ctx_b = &s;
    CHECK(book.hold(p.limit, p.victims, ctx_a, true));
    CHECK(!book.hold(p.limit, {}, ctx_b, false));                  // one lease at a time
    CHECK(book.state == lend_state::held && book.explicit_);
    const uint64_t epoch = book.epoch;
    // the release: the regions become the restore space, nothing is copied yet
    CHECK(book.open_restore(s.layer_class));
    CHECK(book.state == lend_state::restore && book.owner == nullptr);
    CHECK(book.restore.size() == p.victims.size());                // one victim per lent slot
    std::set<std::pair<int, int>> slot_used;
    for (const auto & r : book.restore) {
        CHECK(r.slot >= book.limit[(size_t) r.cls] && r.slot < book.slots[(size_t) r.cls]);
        CHECK(slot_used.insert({r.cls, r.slot}).second);
        CHECK(residents_before[(size_t) r.layer][(size_t) r.expert] >= 0);   // it was an L1 resident before the lend
    }
    // highest rank gets the lowest slot of its class
    for (size_t i = 1; i < book.restore.size(); ++i) {
        const auto & a = book.restore[i - 1], & b = book.restore[i];
        if (a.cls == b.cls) {
            CHECK(a.slot < b.slot);
        }
    }
    // the device tables of the decode: the mirror plus the restored copies
    const size_t E = (size_t) s.experts;
    std::vector<std::vector<int32_t>> dev = s.table;
    std::vector<std::vector<int32_t>> rest((size_t) s.layers, std::vector<int32_t>(E, -1));
    for (const auto & r : book.restore) { rest[(size_t) r.layer][(size_t) r.expert] = r.slot; }
    // no refill: no victim is in a table before a decode launch reads it
    for (const auto & v : p.victims) { CHECK(dev[(size_t) v.layer][(size_t) v.expert] < 0); }
    std::set<std::pair<int, int>> read, restored;
    for (int t = 0; t < 200; ++t) {
        const int used = 4, rows = 1 + (int) (rng() % 2);
        for (int l = 0; l < s.layers; ++l) {
            std::vector<int32_t> ids((size_t) rows*used);
            for (auto & id : ids) { id = (int32_t) (rng() % E); }
            // a duplicate in a launch is restored once
            if (rows*used >= 2) { ids[1] = ids[0]; }
            int32_t list[64];
            const int n = lend_restore_list(ids.data(), rows, used, used, (int) E, rest[(size_t) l].data(), dev[(size_t) l].data(), list, 64);
            std::set<int32_t> seen;
            for (int i = 0; i < n; ++i) {
                CHECK(seen.insert(list[i]).second);
                CHECK(std::find(ids.begin(), ids.end(), list[i]) != ids.end());   // only what the launch reads
                CHECK(rest[(size_t) l][(size_t) list[i]] >= 0 && dev[(size_t) l][(size_t) list[i]] < 0);
            }
            // the last block's publish
            for (int i = 0; i < n; ++i) {
                dev[(size_t) l][(size_t) list[i]] = rest[(size_t) l][(size_t) list[i]];
                rest[(size_t) l][(size_t) list[i]] = -1;
                restored.insert({l, list[i]});
            }
            for (int32_t id : ids) { read.insert({l, id}); }
        }
    }
    // exactly the victims that were read came back, each into its own slot
    std::set<std::pair<int, int>> want;
    for (const auto & r : book.restore) {
        if (read.count({r.layer, r.expert})) { want.insert({r.layer, r.expert}); }
    }
    CHECK(restored == want);
    n_restores += restored.empty() ? 0 : 1;
    for (const auto & r : book.restore) {
        if (restored.count({r.layer, r.expert})) { CHECK(dev[(size_t) r.layer][(size_t) r.expert] == r.slot); }
    }
    // a later lend takes the space back: the copies leave the tables (the mirror), the lease is new
    std::vector<int> held(book.slots.size());
    for (size_t c = 0; c < held.size(); ++c) { held[c] = book.slots[c] - book.limit[c]; }
    const lend_l1_plan again = plan_l1_lend(s.classes(book.limit), pieces, 256, &held);
    CHECK(again.ok && again.victims.empty());                     // the same pieces need no new eviction
    CHECK(book.hold(again.limit, again.victims, ctx_b, false));
    CHECK(book.epoch == epoch + 1 && book.restore.empty() && !book.explicit_);
    // sticky: an install under the lease keeps the restored copies the new restore set still has
    CHECK(book.open_restore(s.layer_class));
    std::vector<lend_restore> kept;
    for (const auto & r : book.restore) {
        if (rng() % 2) { kept.push_back(r); }
    }
    const auto again_restore = plan_restore(book.victims, s.layer_class, book.limit, book.slots, &kept);
    std::set<std::pair<int, int>> slots2;
    for (const auto & k : kept) { slots2.insert({k.cls, k.slot}); }
    for (const auto & r : again_restore) {
        for (const auto & k : kept) { CHECK(!(k.layer == r.layer && k.expert == r.expert)); }
        CHECK(slots2.insert({r.cls, r.slot}).second);
    }
    CHECK(again_restore.size() + kept.size() == book.restore.size());
    n_sticky += kept.empty() ? 0 : 1;
    book.clear();
    CHECK(book.state == lend_state::none && book.limit == book.slots);
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
    test_untie_example();
    test_replay();
    test_width_examples();
    int n = 0;
    for (int i = 0; i < seeds; ++i) {
        const uint32_t seed = 1000u + (uint32_t) i;
        n += test_l1(seed) ? 1 : 0;
        n += test_width(seed) ? 1 : 0;
        n += test_restore(seed) ? 1 : 0;
        n += test_restore_want(seed) ? 1 : 0;
        if (g_fail > 0) {
            fprintf(stderr, "first failure at seed %u\n", seed);
            break;
        }
    }
    printf("test-expert-lend: %d checks over %d seeds, %d failures; paths: L1 evictions %d, L1 growth over held slots %d, "
        "widths of several passes %d, restore runs %d, sticky installs %d\n",
        n, seeds, g_fail, n_l1_plans, n_l1_regrow, n_multi_pass, n_restores, n_sticky);
    const bool covered = n_l1_plans > 0 && n_l1_regrow > 0 && n_multi_pass > 0 && n_restores > 0 && n_sticky > 0;
    printf("test-expert-lend: plans that gave victims of their last tie back (a class not needed) %d\n", n_untied);
    if (!covered) {
        printf("a path was never taken\n");
    }
    printf("%s\n", g_fail == 0 && covered ? "PASS" : "FAIL");
    return g_fail == 0 && covered ? 0 : 1;
}
