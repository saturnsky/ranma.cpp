// ranma: the staging ring ledger of the expert cache SSD tier (expert-l2-ledger.h), and for the
// class layout the storage classes (expert-storage.h), the per-class slot ledger
// (expert-l2-class-ledger.h) with its unified policy, the host cut's per-layer file cap (expert-plan.h)
// and the relabel install transaction (expert-l2-relabel.h).
//
// Everything the ring decides is here: which expert gets which slot, when a slot may be reused,
// how repeated demands retain slots and what a ring resize does. A mock read queue stands in
// for the disk and a fake mailbox for the GPU, so this test needs neither.

#include "expert-hash-early.h"
#include "expert-l2-class-ledger.h"
#include "expert-l2-ledger.h"
#include "expert-l2-relabel.h"
#include "expert-plan.h"
#include "expert-storage.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

using namespace ggml_cuda_expert;

namespace {

// Stands in for the disk: records what was asked for and writes a recognizable value into the slot.
struct mock_queue {
    struct fill {
        int layer, kind, expert;
    };
    std::map<int, fill> slots;    // ring slot -> what it last received
    size_t reads = 0;

    void run(const l2_service & service) {
        for (const l2_read & read : service.reads) {
            slots[read.slot] = fill{read.layer, read.kind, read.expert};
            ++reads;
        }
    }
};

// Stands in for the mailbox: the generation the GPU has finished for each layer.
struct fake_mailbox {
    std::vector<uint32_t> done;
    explicit fake_mailbox(int layers) : done((size_t) layers, 0u) {}
    void publish(l2_ledger & ledger) {
        for (size_t layer = 0; layer < done.size(); ++layer) {
            ledger.set_done((int) layer, done[layer]);
        }
    }
};

auto nothing_homed = [](int, int) { return false; };

// A slow reference of the ring's replacement: the round-robin cursor as before LRU existed, or a
// use stamp per slot with the smallest reusable stamp as the victim (ties to the lower slot).
struct reference_ring {
    struct owner { int layer = -1, expert = -1; uint32_t seq = 0; uint64_t used = 0; };
    bool lru;
    int ring;
    int cursor = 0;
    uint64_t clock = 0;
    std::vector<owner> owners;
    std::vector<uint32_t> done;
    std::map<std::pair<int, int>, int> map;

    reference_ring(bool lru, int ring, int layers) : lru(lru), ring(ring), owners((size_t) ring), done((size_t) layers, 0u) {}

    bool reusable(const owner & o) const { return o.layer < 0 || int32_t(done[(size_t) o.layer] - o.seq) >= 0; }

    // The slots given to the misses in order; false when the demand cannot be served.
    bool service(int layer, const std::vector<int> & wanted, uint32_t seq, int & hits, std::vector<int> & out) {
        std::vector<bool> pinned((size_t) ring, false);
        for (int e : wanted) {
            auto it = map.find({layer, e});
            if (it != map.end()) { pinned[(size_t) it->second] = true; }
        }
        for (int e : wanted) {
            auto it = map.find({layer, e});
            if (it != map.end()) {
                ++hits;
                owners[(size_t) it->second].seq = seq;
                if (lru) { owners[(size_t) it->second].used = ++clock; }
                continue;
            }
            int slot = -1;
            if (lru) {
                for (int s = 0; s < ring; ++s) {
                    if (pinned[(size_t) s] || !reusable(owners[(size_t) s])) { continue; }
                    if (slot < 0 || owners[(size_t) s].used < owners[(size_t) slot].used) { slot = s; }
                }
            } else {
                for (int i = 0; i < ring; ++i) {
                    const int s = cursor;
                    cursor = (cursor + 1)%ring;
                    if (!pinned[(size_t) s] && reusable(owners[(size_t) s])) { slot = s; break; }
                }
            }
            if (slot < 0) { return false; }
            owner & o = owners[(size_t) slot];
            if (o.layer >= 0) { map.erase({o.layer, o.expert}); }
            o = owner{layer, e, seq, ++clock};
            map[{layer, e}] = slot;
            pinned[(size_t) slot] = true;
            out.push_back(slot);
        }
        return true;
    }
};

} // namespace

// ---- the class layout ---------------------------------------------------------------------------

// A geometry with the slice sizes of a real 43-layer model: two large classes and two single-layer
// classes, all tensors 4096 or 2048 wide (tail 512 bytes).
static geometry class_geometry(int experts) {
    geometry geo;
    geo.n_layers = 43; geo.n_experts = experts;
    geo.layer_class.assign(43, 0);
    for (int l = 1; l < 43; l += 3) { geo.layer_class[l] = 1; }
    geo.layer_class[26] = 2; geo.layer_class[42] = 3;
    geo.class_bytes = {{2424832, 2424832, 3211264}, {3211264, 3211264, 3211264},
                       {3604480, 3604480, 4456448}, {3211264, 3211264, 4456448}};
    geo.class_layers.assign(4, 0);
    for (int c : geo.layer_class) { ++geo.class_layers[c]; }
    return geo;
}

static std::vector<std::array<size_t, 3>> class_tails(size_t classes) {
    return std::vector<std::array<size_t, 3>>(classes, std::array<size_t, 3>{512, 512, 512});
}

static int storage_tests() {
    constexpr size_t sector = 4096;
    CHECK(storage_slot_pitch(2424832, 512, sector) == 2424832 + 8192);
    CHECK(storage_slot_pitch(4096, 0, sector) == 8192 && storage_slot_pitch(1, 0, sector) == 4096);
    const geometry geo = class_geometry(256);
    const auto tails = class_tails(4);
    const int bound = storage_demand_bound(256, 6, 512);
    CHECK(bound == 256 && storage_demand_bound(256, 6, 1) == 6 && storage_demand_bound(256, 6, 0) == 0);

    // the merge rule: the two single-layer classes share the arena of the larger one
    class_storage st = plan_storage_classes(geo, tails, sector, bound);
    CHECK(st.valid() && st.storages() == 3);
    CHECK((st.storage_of == std::vector<int>{0, 1, 2, 2}));
    CHECK((st.members[2] == std::vector<int>{2, 3}));
    for (int k = 0; k < 3; ++k) {
        CHECK(st.pitch[0][k] == geo.class_bytes[0][k] + 8192);
        CHECK(st.pitch[2][k] == geo.class_bytes[2][k] + 8192);   // the pitch of the larger member
    }
    // every sector shift of every member fits: the covering read, and the payload plus its tail
    for (int c = 0; c < 4; ++c) {
        const int s = st.storage_of[c];
        for (int k = 0; k < 3; ++k) {
            for (size_t shift : {size_t(0), size_t(32), size_t(1856), size_t(3904), size_t(4095)}) {
                CHECK(storage_align_up(geo.class_bytes[c][k] + shift, sector) <= st.pitch[s][k]);
                CHECK(shift + geo.class_bytes[c][k] + tails[c][k] <= st.pitch[s][k]);
            }
            CHECK(st.pitch[s][k] % sector == 0);
        }
    }
    // no merge without a ring to save, or when the padding outweighs the floor
    CHECK(plan_storage_classes(geo, tails, sector, 0).storages() == 4);
    CHECK(plan_storage_classes(geo, tails, sector, bound, false).storages() == 4);
    CHECK(plan_storage_classes(geo, tails, sector, 1).storages() == 4);
    // a class of more than two layers never moves
    geometry wide = geo;
    wide.layer_class[39] = wide.layer_class[40] = 3;
    wide.class_layers.assign(4, 0);
    for (int c : wide.layer_class) { ++wide.class_layers[c]; }
    const class_storage kept = plan_storage_classes(wide, tails, sector, bound);
    CHECK(kept.storages() == 4 && kept.storage_of[3] == 3 && kept.members[3].size() == 1);

    // placement: the resident ranges of the members, then the ring
    CHECK(st.place({10, 20, 3, 4}, {7, 8, 9}));
    CHECK(st.resident_base[2] == 0 && st.resident_base[3] == 3 && st.ring_base[2] == 7 && st.slots(2) == 16);
    CHECK(st.ring_base[0] == 10 && st.slots(0) == 17 && st.resident_slot(3, 2) == 5);
    CHECK(st.ring_bytes() == 7*st.stride(0) + 8*st.stride(1) + 9*st.stride(2));
    CHECK(st.arena_bytes() == 17*st.stride(0) + 28*st.stride(1) + 16*st.stride(2) + st.tail_bytes());
    CHECK(!st.place({1, 2, 3}, {1, 1, 1}) && !st.place({1, 2, 3, -1}, {1, 1, 1}));

    // floors: the most file residents of one layer, per storage class, within the bound
    {
        expert_slot_table vram(43, std::vector<int32_t>(256, -1)), host(43);
        for (int l = 0; l < 43; ++l) {
            for (int e = 0; e < 100; ++e) { host[l].push_back(e); }
            for (int e = 100; e < 140; ++e) { vram[l][e] = 0; }
        }
        host[5].clear();        // class 0: 216 file residents
        host[26].resize(10);    // storage class 2: 206
        host[42].resize(60);    // storage class 2: 156
        CHECK((storage_floors(geo, st, &vram, &host, bound) == std::vector<int>{216, 116, 206}));
        CHECK((storage_floors(geo, st, &vram, &host, 150) == std::vector<int>{150, 116, 150}));
        CHECK((storage_floors(geo, st, nullptr, nullptr, bound) == std::vector<int>{256, 256, 256}));
    }
    // an explicit total: the floors first, then the rest by the bytes each class leaves in the file
    {
        const std::vector<int> floors = {10, 20, 30}, limit = {1000, 1000, 1000};
        const std::vector<uint64_t> demand = {3, 1, 0};
        const size_t at_floors = 10*st.stride(0) + 20*st.stride(1) + 30*st.stride(2);
        CHECK(split_ring(st, floors, demand, limit, 0) == floors);
        CHECK(split_ring(st, floors, demand, limit, at_floors - 1) == floors);
        const size_t extra = 400*st.stride(0);
        const auto ring = split_ring(st, floors, demand, limit, at_floors + extra);
        size_t bytes = 0;
        for (int s = 0; s < 3; ++s) { CHECK(ring[s] >= floors[s]); bytes += size_t(ring[s])*st.stride(s); }
        CHECK(bytes <= at_floors + extra && at_floors + extra - bytes < st.stride(2));
        CHECK(ring[2] == 30 && ring[0] - 10 > 2*(ring[1] - 20));
        CHECK(split_ring(st, floors, demand, {12, 21, 30}, at_floors + extra) == (std::vector<int>{12, 21, 30}));
    }
    // the default rings: factor x floor, rounded up, within the limit, never below the floor
    {
        const std::vector<int> floors = {191, 188, 180}, limit = {25*256, 16*256, 2*256};
        CHECK((factor_rings(floors, 2.0, limit) == std::vector<int>{382, 376, 360}));
        CHECK(factor_rings(floors, 1.0, limit) == floors);
        CHECK((factor_rings(floors, 1.5, limit) == std::vector<int>{287, 282, 270}));
        CHECK((factor_rings(floors, 4.0, limit) == std::vector<int>{764, 752, 512}));
        CHECK((factor_rings({0, 10, 300}, 2.0, {100, 100, 200}) == std::vector<int>{0, 20, 300}));
        CHECK(factor_rings(floors, 0.5, limit) == floors && factor_rings(floors, 2.0, {1, 2}) == floors);
        // the budget holds the whole factor: unchanged
        const std::vector<int> want = factor_rings(floors, 2.0, limit);
        size_t extra = 0;
        for (int s = 0; s < 3; ++s) { extra += size_t(want[s] - floors[s])*st.stride(s); }
        CHECK(shrink_rings(st, floors, want, extra) == want);
        CHECK(shrink_rings(st, floors, want, extra + 12345) == want);
        // nothing above the minimum: the floors
        CHECK(shrink_rings(st, floors, want, 0) == floors);
        // half of it: every class gets half of its extra, rounded down, and the result fits
        const auto half = shrink_rings(st, floors, want, extra/2);
        size_t used = 0;
        for (int s = 0; s < 3; ++s) {
            CHECK(half[s] >= floors[s] && half[s] <= want[s]);
            CHECK(std::abs(2*(half[s] - floors[s]) - (want[s] - floors[s])) <= 2);
            used += size_t(half[s] - floors[s])*st.stride(s);
        }
        CHECK(used <= extra/2);
        CHECK(shrink_rings(st, floors, {1, 2}, 0) == floors);
    }
    printf("PASS: storage classes: pitch with sector room and tail, merge rule, placement, floors, ring factor, "
           "proportional shrink and ring split\n");
    return 0;
}

// The host cut keeps no layer above the ring of its storage class.
static int file_cap_tests() {
    geometry geo;
    geo.n_layers = 4; geo.n_experts = 8;
    geo.layer_class = {0, 0, 1, 1}; geo.class_layers = {2, 2};
    geo.class_bytes = {{100, 100, 100}, {200, 200, 200}};
    expert_slot_table vram(4, std::vector<int32_t>(8, -1));
    vram[0][0] = 0; vram[2][0] = 0;
    std::vector<uint64_t> scores(32, 0);
    for (int e = 0; e < 8; ++e) { scores[0*8 + e] = 100 + e; scores[1*8 + e] = 1 + e; scores[2*8 + e] = 50; scores[3*8 + e] = 40 - e; }
    const std::vector<size_t> pitch = {400, 700};
    const std::vector<int> caps = {6, 7};
    tier_inputs in; in.geo = &geo; in.vram = &vram; in.slot_pitch = &pitch; in.counts = scores.data();
    in.fixed_capacities = &caps;
    const tier_plan greedy = plan_host_tier(in);
    CHECK(greedy.valid);
    // greedy: class 0 takes all of layer 0 (7 non-VRAM) minus none -> layer 1 keeps 8 - 0 = 8 in the file
    auto file_of = [&](const tier_plan & p, int l) {
        int resident = 0;
        for (int e = 0; e < 8; ++e) {
            resident += vram[l][e] >= 0 || std::binary_search(p.selected[l].begin(), p.selected[l].end(), e) ? 1 : 0;
        }
        return 8 - resident;
    };
    CHECK(file_of(greedy, 0) == 1 && file_of(greedy, 1) == 8);
    const std::vector<int> cap = {5, 8};
    in.file_cap = &cap;
    const tier_plan capped = plan_host_tier(in);
    CHECK(capped.valid && capped.capacities == caps);
    CHECK(file_of(capped, 1) == 5 && file_of(capped, 0) == 4);
    // the best file experts of the capped layer came in, the weakest host residents of the other went
    CHECK((capped.selected[1] == std::vector<int32_t>{5, 6, 7}));
    CHECK((capped.selected[0] == std::vector<int32_t>{5, 6, 7}));
    CHECK(capped.ssd_slices == greedy.ssd_slices && capped.ssd_bytes == greedy.ssd_bytes);
    // the other class is untouched
    CHECK(capped.selected[2] == greedy.selected[2] && capped.selected[3] == greedy.selected[3]);
    // infeasible: 15 non-VRAM experts in class 0, six host slots: 9 file residents > 2 layers x 4
    const std::vector<int> short_cap = {4, 8};
    in.file_cap = &short_cap;
    CHECK(!plan_host_tier(in).valid);
    const std::vector<int> tight = {3, 8};
    in.file_cap = &tight;
    const tier_plan refused = plan_host_tier(in);
    CHECK(!refused.valid && refused.reason.find("ring slots") != std::string::npos);
    const std::vector<int> wrong = {3};
    in.file_cap = &wrong;
    CHECK(!plan_host_tier(in).valid);
    printf("PASS: the host cut keeps each layer within its storage class ring, counts unchanged\n");
    return 0;
}

// Two storage classes: class 0 = layers 0 and 2 with 3 resident slots then a ring of 4, class 1 =
// layers 1 and 3 with 2 resident slots then a ring of 3.
static l2_class_ledger make_class_ledger() {
    l2_class_ledger ledger;
    ledger.reset(4, 16, {0, 1, 0, 1}, {7, 5}, 3);
    for (int slot = 0; slot < 3; ++slot) { ledger.set_resident(0, slot, -1, -1); }
    for (int slot = 0; slot < 2; ++slot) { ledger.set_resident(1, slot, -1, -1); }
    for (int slot = 3; slot < 7; ++slot) { ledger.make_ring(0, slot); }
    for (int slot = 2; slot < 5; ++slot) { ledger.make_ring(1, slot); }
    return ledger;
}

static int class_ledger_tests() {
    auto none = [](int, int) { return false; };
    std::string why;
    // ---- roles -------------------------------------------------------------------------------
    {
        l2_class_ledger ledger = make_class_ledger();
        CHECK(ledger.check(why));
        CHECK(ledger.storages() == 2 && ledger.ring_count(0) == 4 && ledger.ring_count(1) == 3 && ledger.ring_total() == 7);
        CHECK(ledger.role_of(0, 2) == slot_role::resident && ledger.role_of(0, 3) == slot_role::ring);
        CHECK((ledger.order(0) == std::vector<int>{3, 4, 5, 6}) && (ledger.order(1) == std::vector<int>{2, 3, 4}));
        CHECK(!ledger.make_ring(0, 3) && !ledger.make_ring(0, 0) && !ledger.make_ring(2, 0) && !ledger.make_ring(0, 7));
        CHECK(ledger.set_resident(0, 1, 2, 9) && !ledger.set_resident(0, 3, 2, 9) && !ledger.set_resident(0, 1, 1, 9));
        int l = -1, e = -1;
        CHECK(ledger.occupant(0, 1, l, e) && l == 2 && e == 9 && !ledger.occupant(0, 0, l, e));
        ledger.clear_residents();
        CHECK(!ledger.occupant(0, 1, l, e) && ledger.role_of(0, 1) == slot_role::resident);
        l2_class_ledger fresh;
        fresh.reset(4, 16, {0, 1, 0, 1}, {2, 2}, 3);
        CHECK(fresh.role_of(0, 0) == slot_role::free && fresh.ring_total() == 0 && fresh.check(why));
        printf("PASS: class ledger roles: resident, ring and free slots, ring lists in index order\n");
    }
    // ---- per-class service, LRU, leases and pins ----------------------------------------------
    {
        l2_class_ledger ledger = make_class_ledger();
        // a layer of class 1 only ever uses class 1 ring slots
        auto first = ledger.service(1, {4, 5, 4}, 1, none);
        CHECK(first.ok && first.misses == 2 && first.reads.size() == 6);
        CHECK(ledger.slot_of(1, 4) == 2 && ledger.slot_of(1, 5) == 3);
        CHECK(ledger.owns(2, 1, 4) && !ledger.owns(2, 0, 4) && !ledger.owns(0, 1, 4));
        auto other = ledger.service(0, {4, 5}, 1, none);
        CHECK(other.ok && other.misses == 2 && ledger.slot_of(0, 4) == 3 && ledger.slot_of(0, 5) == 4);
        // homed experts never reach the ring
        auto homed = ledger.service(2, {7}, 1, [](int, int e) { return e == 7; });
        CHECK(homed.ok && homed.misses == 0 && homed.hits == 0 && ledger.slot_of(2, 7) < 0);
        // a hit refreshes and extends the lease; the miss after it evicts the least recently used
        ledger.set_done(1, 1);
        auto hit = ledger.service(1, {4}, 2, none);
        CHECK(hit.ok && hit.hits == 1 && hit.reads.empty());
        CHECK((ledger.order(1) == std::vector<int>{4, 3, 2}));
        auto fill = ledger.service(1, {6}, 2, none);
        CHECK(fill.ok && fill.evicted.empty() && ledger.slot_of(1, 6) == 4);
        CHECK((ledger.order(1) == std::vector<int>{3, 2, 4}));
        // slot 3 (expert 5, lease 1) is reusable, slots 2 and 4 are leased by generation 2
        auto evict = ledger.service(3, {1}, 1, none);
        CHECK(evict.ok && evict.evicted.size() == 1 && evict.evicted[0].layer == 1 && evict.evicted[0].expert == 5);
        CHECK(ledger.slot_of(3, 1) == 3 && ledger.slot_of(1, 5) < 0);
        // everything leased or pinned: refused, class 0 unaffected
        auto full = ledger.service(3, {1, 2}, 1, none);
        CHECK(!full.ok && full.reason.find("storage class 1") != std::string::npos);
        ledger.set_done(1, 2); ledger.set_done(3, 1);
        auto after = ledger.service(3, {1, 2}, 2, none);
        CHECK(after.ok && after.hits == 1 && after.misses == 1 && after.evicted.size() == 1 && after.evicted[0].expert == 4);
        CHECK(!ledger.service(0, {99}, 2, none).ok && !ledger.service(9, {1}, 2, none).ok);
        CHECK(ledger.check(why));
        printf("PASS: class ledger service: per-class rings, LRU hits refresh, leases and pins\n");
    }
    // ---- grow: resident -> ring occupant in place ---------------------------------------------
    {
        l2_class_ledger ledger = make_class_ledger();
        CHECK(ledger.set_resident(0, 1, 2, 9));
        CHECK(ledger.relabel_to_ring(0, 1, true));
        CHECK(ledger.ring_count(0) == 5 && ledger.role_of(0, 1) == slot_role::ring && ledger.slot_of(2, 9) == 1);
        CHECK((ledger.order(0) == std::vector<int>{1, 3, 4, 5, 6}));
        CHECK(!ledger.relabel_to_ring(0, 1, true) && !ledger.relabel_to_ring(0, 4, false));
        // the demoted expert is a ring hit, no read
        auto hit = ledger.service(2, {9}, 1, none);
        CHECK(hit.ok && hit.hits == 1 && hit.reads.empty());
        // an empty resident joins at the hot end when not cold
        CHECK(ledger.relabel_to_ring(0, 0, false));
        CHECK((ledger.order(0) == std::vector<int>{3, 4, 5, 6, 1, 0}));
        // a cold demoted expert goes first
        CHECK(ledger.set_resident(0, 2, 0, 3) && ledger.relabel_to_ring(0, 2, true));
        ledger.set_done(2, 1);
        auto miss = ledger.service(0, {11}, 1, none);
        CHECK(miss.ok && miss.evicted.size() == 1 && miss.evicted[0].layer == 0 && miss.evicted[0].expert == 3);
        CHECK(ledger.slot_of(0, 11) == 2 && ledger.check(why));
        printf("PASS: grow relabels a resident into the ring in place, cold demotions go first\n");
    }
    // ---- shrink: ring -> resident, with a pending read unless the slot holds the expert ---------
    {
        l2_class_ledger ledger = make_class_ledger();
        ledger.set_floor(0, 2);
        CHECK(ledger.service(0, {4, 5}, 1, none).ok);   // slots 3, 4
        // leased by generation 1 of layer 0: refused
        auto leased = ledger.relabel_to_resident(0, 3, 0, 4);
        CHECK(!leased.ok && leased.reason.find("leased") != std::string::npos);
        ledger.set_done(0, 1);
        // the slot holds the promoted expert: no read
        auto keep = ledger.relabel_to_resident(0, 3, 0, 4);
        CHECK(keep.ok && !keep.read && keep.evicted.empty());
        CHECK(ledger.role_of(0, 3) == slot_role::resident && ledger.slot_of(0, 4) < 0 && ledger.ring_count(0) == 3);
        int l = -1, e = -1;
        CHECK(ledger.occupant(0, 3, l, e) && l == 0 && e == 4);
        // another expert: its occupant is evicted, and the copy of the promoted one elsewhere too
        CHECK(ledger.service(2, {8}, 1, none).ok);   // slot 5
        ledger.set_done(2, 1);
        auto swap = ledger.relabel_to_resident(0, 4, 2, 8);
        CHECK(swap.ok && swap.read && swap.evicted.size() == 2);
        CHECK(swap.evicted[0].expert == 5 && swap.evicted[1].expert == 8 && swap.evicted[1].slot == 5);
        CHECK(ledger.slot_of(0, 5) < 0 && ledger.slot_of(2, 8) < 0 && ledger.ring_count(0) == 2);
        CHECK(ledger.role_of(0, 5) == slot_role::ring && !ledger.occupant(0, 5, l, e));
        // the floor
        auto low = ledger.relabel_to_resident(0, 5, 0, 1);
        CHECK(!low.ok && low.reason.find("floor") != std::string::npos);
        // not a ring slot, wrong class
        CHECK(!ledger.relabel_to_resident(0, 0, 0, 1).ok && !ledger.relabel_to_resident(0, 5, 1, 1).ok);
        CHECK(ledger.check(why));
        printf("PASS: shrink relabels a ring slot into a resident, reads only what it does not hold, keeps the floor\n");
    }
    // ---- transient install reads -----------------------------------------------------------------
    {
        l2_class_ledger ledger = make_class_ledger();
        CHECK(ledger.service(0, {1, 2}, 1, none).ok);   // slots 3, 4, leased
        std::vector<l2_eviction> evicted;
        const int a = ledger.take_free(0, evicted), b = ledger.take_free(0, evicted);
        CHECK(a == 5 && b == 6 && evicted.empty());
        // the leased slots are skipped; the transient ones come round again
        CHECK(ledger.take_free(0, evicted) == 5);
        l2_class_ledger all_leased = make_class_ledger();
        CHECK(all_leased.service(0, {1, 2, 3, 4}, 1, none).ok);
        CHECK(all_leased.take_free(0, evicted) < 0 && all_leased.take_free(1, evicted) == 2);
        evicted.clear();
        ledger.set_done(0, 1);
        const int c = ledger.take_free(0, evicted);
        CHECK(c == 3 && evicted.size() == 1 && evicted[0].expert == 1 && ledger.slot_of(0, 1) < 0);
        std::set<int> distinct = {a, b, c, ledger.take_free(0, evicted)};
        CHECK(distinct.size() == 4 && evicted.size() == 2 && ledger.check(why));
        // discard keeps the roles and restarts the order
        ledger.discard();
        CHECK((ledger.order(0) == std::vector<int>{3, 4, 5, 6}) && ledger.slot_of(0, 2) < 0 && ledger.ring_count(0) == 4);
        CHECK(ledger.role_of(0, 0) == slot_role::resident && ledger.check(why));
        printf("PASS: install reads take distinct unleased ring slots; discard keeps the roles\n");
    }
    // ---- an install as relabels: reads = promotions not already in the ring ---------------------
    {
        // one storage class, one layer, 8 experts: residents {0,1,2} in slots 0..2, ring slots 3..6
        l2_class_ledger ledger;
        ledger.reset(1, 8, {0}, {7}, 3);
        for (int s = 0; s < 3; ++s) { CHECK(ledger.set_resident(0, s, 0, s)); }
        for (int s = 3; s < 7; ++s) { CHECK(ledger.make_ring(0, s)); }
        ledger.set_floor(0, 4);
        CHECK(ledger.service(0, {5, 6, 7}, 1, [](int, int e) { return e < 3; }).ok);   // 5, 6, 7 in slots 3, 4, 5
        ledger.set_done(0, 1);
        // new plan: residents {0, 5, 3}; 1 and 2 demoted, 5 promoted from the ring, 3 from the file
        CHECK(ledger.relabel_to_ring(0, 1, true) && ledger.relabel_to_ring(0, 2, true));
        int reads = 0;
        auto p5 = ledger.relabel_to_resident(0, ledger.slot_of(0, 5), 0, 5);
        CHECK(p5.ok && !p5.read);
        std::vector<l2_eviction> evicted;
        const int free_slot = ledger.order(0)[0];
        auto p3 = ledger.relabel_to_resident(0, free_slot, 0, 3);
        CHECK(p3.ok && p3.read);
        reads += p3.read ? 1 : 0;
        CHECK(reads == 1 && ledger.ring_count(0) == 4 && ledger.check(why));
        // the last cold demotion went first: expert 2 left the ring, expert 1 is still a hit
        CHECK(free_slot == 2 && ledger.slot_of(0, 2) < 0 && ledger.slot_of(0, 1) == 1);
        int residents = 0;
        for (int s = 0; s < 7; ++s) { residents += ledger.role_of(0, s) == slot_role::resident ? 1 : 0; }
        CHECK(residents == 3);
        printf("PASS: an install as relabels reads only the promotions that are not in the ring\n");
    }
    // ---- random demands against the slow reference, per storage class ----------------------------
    {
        const int layers = 4, experts = 16;
        const std::vector<int> layer_storage = {0, 1, 0, 1};
        for (int ring : {1, 3, 8}) {
            l2_class_ledger ledger;
            ledger.reset(layers, experts, layer_storage, {2 + ring, 5 + ring}, 3);
            for (int s = 0; s < 2; ++s) {
                const int residents = s == 0 ? 2 : 5;
                for (int i = 0; i < residents; ++i) { CHECK(ledger.set_resident(s, i, -1, -1)); }
                for (int i = 0; i < ring; ++i) { CHECK(ledger.make_ring(s, residents + i)); }
            }
            std::vector<reference_ring> ref = {reference_ring(true, ring, layers), reference_ring(true, ring, layers)};
            const int base[2] = {2, 5};
            uint32_t rng = 777u + 13u*(uint32_t) ring;
            auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
            std::vector<uint32_t> seq((size_t) layers, 0u);
            int served = 0;
            for (int step = 0; step < 4000; ++step) {
                const int layer = int(next()%layers);
                const int s = layer_storage[(size_t) layer];
                for (int l = 0; l < layers; ++l) {
                    if (next()%3 == 0) {
                        ledger.set_done(l, seq[(size_t) l]);
                        ref[0].done[(size_t) l] = ref[1].done[(size_t) l] = seq[(size_t) l];
                    }
                }
                std::vector<int> ids, wanted;
                std::vector<bool> seen((size_t) experts, false);
                const int n = 1 + int(next()%(uint32_t) std::min(ring, 4));
                for (int i = 0; i < n; ++i) {
                    const int e = next()%4 == 0 ? int(next()%experts) : int(next()%5);
                    ids.push_back(e);
                    if (!seen[(size_t) e]) { seen[(size_t) e] = true; wanted.push_back(e); }
                }
                const uint32_t g = ++seq[(size_t) layer];
                int ref_hits = 0;
                std::vector<int> ref_slots;
                const bool ref_ok = ref[(size_t) s].service(layer, wanted, g, ref_hits, ref_slots);
                const auto got = ledger.service(layer, ids, g, none);
                if (!ref_ok) {
                    CHECK(!got.ok);
                    ledger.discard();
                    for (int k = 0; k < 2; ++k) {
                        const std::vector<uint32_t> done = ref[(size_t) k].done;
                        ref[(size_t) k] = reference_ring(true, ring, layers);
                        ref[(size_t) k].done = done;
                    }
                    continue;
                }
                CHECK(got.ok && got.hits == ref_hits && got.misses == (int) ref_slots.size());
                for (size_t i = 0; i < ref_slots.size(); ++i) { CHECK(got.reads[i*3].slot == base[s] + ref_slots[i]); }
                for (const auto & old : got.evicted) {
                    CHECK(layer_storage[(size_t) old.layer] == s && ledger.slot_of(old.layer, old.expert) < 0);
                }
                ++served;
            }
            CHECK(served > 1000 && ledger.check(why));
        }
        printf("PASS: random demands match the LRU reference in every storage class\n");
    }
    return 0;
}

// ---- the relabel install transaction (expert-l2-relabel.h) ------------------------------------------

namespace {

// Four layers in three size classes (layers 0 and 2, layer 1, layer 3); classes 1 and 2 share storage
// class 1. The arenas and the VRAM arena are modelled as "which expert's bytes are in this slot", so
// every step of a transaction can be replayed and checked: a copy reads the bytes its source holds,
// a relabel only renames bytes that are already there.
struct relabel_world {
    static constexpr int layers = 4, experts = 12;
    geometry geo;
    std::vector<int> storage_of = {0, 1, 1};
    std::vector<int> vram_caps, host_caps, rings;
    int spare = 1;
    bool inclusive = false;
    l2_class_ledger ledger;
    std::vector<std::vector<int>> home_slot;
    std::vector<std::vector<int>> content;   // [storage class][slot] expert key or -1
    std::vector<std::vector<int>> vram;      // [size class][slot] expert key or -1
    expert_slot_table gpu;
    expert_locations host;
    std::vector<std::vector<int>> spares;
    install_layout layout;
    std::vector<uint32_t> seq;
    std::string error;

    relabel_world(std::vector<int> ring, bool incl, std::vector<int> vcaps = {6, 3, 3}, std::vector<int> hcaps = {8, 4, 3},
            int vram_spares = 1)
            : vram_caps(vcaps), host_caps(hcaps), rings(ring), inclusive(incl) {
        geo.n_layers = layers; geo.n_experts = experts;
        geo.layer_class = {0, 1, 0, 2};
        geo.class_bytes = {{100, 100, 100}, {200, 200, 200}, {150, 150, 150}};
        geo.class_layers = {2, 1, 1};
        spare = incl ? 0 : vram_spares;
        std::vector<int> slots = {0, 0};
        home_slot.assign(3, {});
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < host_caps[c] + spare; ++i) { home_slot[c].push_back(slots[storage_of[c]]++); }
        }
        const std::vector<int> ring_base = slots;
        for (int s = 0; s < 2; ++s) { slots[s] += rings[s]; }
        ledger.reset(layers, experts, {0, 1, 0, 1}, slots, 3);
        for (int c = 0; c < 3; ++c) for (int slot : home_slot[c]) { ledger.set_resident(storage_of[c], slot, -1, -1); }
        for (int s = 0; s < 2; ++s) {
            for (int i = 0; i < rings[s]; ++i) { ledger.make_ring(s, ring_base[s] + i); }
        }
        content.assign(2, {});
        for (int s = 0; s < 2; ++s) { content[s].assign((size_t) slots[s], -1); }
        vram.assign(3, {});
        spares.assign(3, {});
        layout.gpu = vram_caps; layout.host = host_caps;
        for (int c = 0; c < 3; ++c) {
            layout.gpu[c] += spare; layout.host[c] += spare;
            vram[c].assign((size_t) layout.gpu[c], -1);
            for (int i = 0; i < spare; ++i) { spares[c].push_back(vram_caps[c] + i); }
        }
        layout.lent_begin.assign(3, 0); layout.lent_count.assign(3, 0);
        seq.assign(layers, 0u);
    }

    // The VRAM and host selections of a plan: per size class the best scores, VRAM first.
    void choose(const std::vector<uint64_t> & score, expert_slot_table & gs, expert_slot_table & hs) const {
        gs.assign(layers, {}); hs.assign(layers, {});
        for (int c = 0; c < 3; ++c) {
            std::vector<std::pair<uint64_t, int>> order;
            for (int l = 0; l < layers; ++l) {
                if (geo.layer_class[l] != c) { continue; }
                for (int e = 0; e < experts; ++e) { order.push_back({score[(size_t) (l*experts + e)], l*experts + e}); }
            }
            std::sort(order.begin(), order.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            for (size_t i = 0; i < order.size(); ++i) {
                const int l = order[i].second/experts, e = order[i].second%experts;
                if ((int) i < vram_caps[c]) {
                    gs[l].push_back(e);
                    if (inclusive) { hs[l].push_back(e); }
                } else if ((int) i < host_caps[c] + (inclusive ? 0 : vram_caps[c])) {
                    hs[l].push_back(e);
                }
            }
        }
        for (auto & row : gs) { std::sort(row.begin(), row.end()); }
        for (auto & row : hs) { std::sort(row.begin(), row.end()); }
    }

    // The load: every selected expert written to its first home, as the loader does.
    void load(const std::vector<uint64_t> & score) {
        expert_slot_table gs, hs;
        choose(score, gs, hs);
        gpu.assign(layers, std::vector<int32_t>(experts, -1));
        host.assign(layers, std::vector<expert_location>(experts));
        std::vector<int> ng(3, 0), nh(3, 0);
        for (int l = 0; l < layers; ++l) {
            const int c = geo.layer_class[l];
            for (int e : gs[l]) { gpu[l][e] = ng[c]; vram[c][(size_t) ng[c]++] = l*experts + e; }
            for (int e : hs[l]) {
                host[l][e] = {expert_storage::host, nh[c]};
                ledger.set_resident(storage_of[c], home_slot[c][(size_t) nh[c]], l, e);
                content[storage_of[c]][(size_t) home_slot[c][(size_t) nh[c]++]] = l*experts + e;
            }
        }
    }

    bool fail(const std::string & why) { error = why; return false; }

    // Reads the file experts of `ids` that have no home into the ring, like the worker; without
    // `done` the generation stays leased.
    bool serve(int layer, const std::vector<int> & ids, bool done = true) {
        const uint32_t g = ++seq[(size_t) layer];
        auto homed = [&](int l, int e) { return gpu[l][e] >= 0 || host[l][e].resident(); };
        const l2_service got = ledger.service(layer, ids, g, homed);
        if (!got.ok) { return fail(got.reason); }
        for (const l2_read & r : got.reads) { content[(size_t) ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*experts + r.expert; }
        if (done) { ledger.set_done(layer, g); }
        return true;
    }

    // One install: plan_install, then the relabel plan replayed on the byte model.
    bool install(const std::vector<uint64_t> & score, l2_install_plan & plan) {
        expert_slot_table gs, hs;
        choose(score, gs, hs);
        const install_transaction tx = plan_install(geo, gs, hs, gpu, host, vram_caps, layout, layout, spares,
            {inclusive, spare}, true, true);
        if (!tx.valid) { return fail("plan_install: " + tx.reason); }
        std::vector<int> ring_before = {ledger.ring_count(0), ledger.ring_count(1)};
        plan = plan_relabel_install(ledger, home_slot, geo.layer_class, storage_of, host, tx.host, tx.moves);
        if (!plan.ok) { return fail("relabel: " + plan.reason); }
        if (plan.relabeled + plan.ring_copies + plan.reads != tx.ssd_slices) { return fail("file moves not all accounted"); }
        // replay, and check that no slot is used twice inside a read batch
        std::set<std::pair<int, int>> batch_slots;
        int batch = -2;
        for (size_t k = 0; k < tx.moves.size(); ++k) {
            const install_move & m = tx.moves[k];
            const l2_install_step & st = plan.steps[k];
            const int s = storage_of[m.cls], key = m.layer*experts + m.expert;
            if (m.from.storage == expert_storage::file) {
                if (st.batch != batch) { batch = st.batch; batch_slots.clear(); }
                const int slot = m.to.storage == expert_storage::host ? st.dst : st.src;
                if (!batch_slots.insert({s, slot}).second) { return fail("a slot used twice in one read batch"); }
                if (st.read) { content[s][(size_t) slot] = key; }
                if (content[s][(size_t) slot] != key) { return fail("a relabel or ring copy of a slot that does not hold the expert"); }
                if (m.to.storage == expert_storage::vram) { vram[m.cls][(size_t) m.to.slot] = key; }
            } else {
                batch = -2;
                if (m.from.storage == expert_storage::host) {
                    if (content[s][(size_t) st.src] != key) { return fail("a copy to VRAM from a slot that does not hold the expert"); }
                    vram[m.cls][(size_t) m.to.slot] = key;
                } else {
                    if (vram[m.cls][(size_t) m.from.slot] != key) { return fail("VRAM model"); }
                    content[s][(size_t) st.dst] = key;
                }
            }
        }
        gpu = tx.gpu_slots; host = tx.host; spares = tx.gpu_spares;
        if (ledger.ring_count(0) != ring_before[0] || ledger.ring_count(1) != ring_before[1]) { return fail("ring size changed"); }
        return check();
    }

    // Every home and every ring occupant holds its own bytes; residents mirror the plan.
    bool check() {
        std::string why;
        if (!ledger.check(why)) { return fail(why); }
        for (int l = 0; l < layers; ++l) {
            const int c = geo.layer_class[l], s = storage_of[c];
            for (int e = 0; e < experts; ++e) {
                const int key = l*experts + e;
                if (gpu[l][e] >= 0 && vram[c][(size_t) gpu[l][e]] != key) { return fail("VRAM home"); }
                if (host[l][e].resident()) {
                    const int slot = home_slot[c][(size_t) host[l][e].slot];
                    int ol = -1, oe = -1;
                    if (content[s][(size_t) slot] != key || ledger.role_of(s, slot) != slot_role::resident ||
                            !ledger.occupant(s, slot, ol, oe) || ol != l || oe != e) { return fail("host home"); }
                }
                const int ring = ledger.slot_of(l, e);
                if (ring >= 0 && content[s][(size_t) ring] != key) { return fail("ring occupant bytes"); }
            }
        }
        for (int c = 0; c < 3; ++c) {
            std::set<int> distinct(home_slot[c].begin(), home_slot[c].end());
            if (distinct.size() != home_slot[c].size()) { return fail("two host slots share a storage slot"); }
        }
        return true;
    }
};

// Scores 1000 - 10*key, so each class ranks its experts by layer and id: class 0 puts layer 0
// experts 0..5 in VRAM (1000..950) and layer 0 experts 6..11 (940..890) and layer 2 experts 0, 1
// (760, 750) in the host tier; class 1 (layer 1) VRAM 0..2 (880..860), host 3..6 (850..820); class 2
// (layer 3) VRAM 0..2 (640..620), host 3..5 (610..590). `set` overrides single keys.
std::vector<uint64_t> scores(std::initializer_list<std::pair<int, int>> set) {
    std::vector<uint64_t> score((size_t) relabel_world::layers*relabel_world::experts);
    for (size_t i = 0; i < score.size(); ++i) { score[i] = 1000 - 10*i; }
    for (const auto & item : set) { score[(size_t) item.first] = (uint64_t) item.second; }
    return score;
}

} // namespace

static int relabel_install_tests() {
    const int E = relabel_world::experts;
    auto homed_in = [](relabel_world & w) { return [&w](int l, int e) { return w.gpu[l][e] >= 0 || w.host[l][e].resident(); }; };
    // ---- demotions stay in the ring, promotions from the ring are not read ------------------------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        l2_install_plan plan;
        // layer 2 file experts 10 and 11 are demanded, then a plan makes them host residents
        CHECK(w.serve(2, {10, 11}));
        const int before_10 = w.ledger.slot_of(2, 10);
        CHECK(before_10 >= 0 && w.ledger.slot_of(2, 11) >= 0);
        CHECK(w.install(scores({{2*E + 10, 945}, {2*E + 11, 944}}), plan));
        CHECK(plan.relabeled == 2 && plan.reads == 0 && plan.ring_copies == 0 && plan.demoted == 2);
        CHECK(w.host[2][10].resident() && w.home_slot[0][(size_t) w.host[2][10].slot] == before_10);
        // the two residents they displaced (layer 2 experts 0 and 1) stay in the ring, nothing moved
        CHECK(!w.host[2][0].resident() && w.ledger.slot_of(2, 0) >= 0 && w.ledger.slot_of(2, 1) >= 0);
        CHECK(w.ledger.ring_count(0) == 6 && w.ledger.ring_count(1) == 5);
        const uint32_t g = ++w.seq[2];
        auto hit = w.ledger.service(2, {0}, g, homed_in(w));
        CHECK(hit.ok && hit.hits == 1 && hit.misses == 0);
        w.ledger.set_done(2, g);
        // the old plan back: relabels only, 10 and 11 stay in the ring
        CHECK(w.install(scores({}), plan));
        CHECK(plan.relabeled == 2 && plan.reads == 0 && plan.demoted == 2);
        CHECK(w.ledger.slot_of(2, 10) >= 0 && w.ledger.slot_of(2, 11) >= 0);
        printf("PASS: relabel install: demoted residents stay in the ring, promotions from the ring are not read\n");
    }
    // ---- promotions not in the ring are read into a ring slot that becomes their home -------------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        l2_install_plan plan;
        CHECK(w.install(scores({{2*E + 10, 945}, {1*E + 11, 855}, {3*E + 11, 615}}), plan));
        CHECK(plan.relabeled == 0 && plan.reads == 3 && plan.demoted == 3 && plan.batches == 1);
        // the reads took empty ring slots: the demoted residents are all still in the ring
        CHECK(w.ledger.slot_of(2, 1) >= 0 && w.ledger.slot_of(1, 6) >= 0 && w.ledger.slot_of(3, 5) >= 0);
        CHECK(w.ledger.ring_count(0) == 6 && w.ledger.ring_count(1) == 5);
        printf("PASS: relabel install: other promotions are read straight into their new home, rings keep their size\n");
    }
    // ---- VRAM changes: host -> VRAM frees the home, VRAM -> host copies into a ring slot -----------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        l2_install_plan plan;
        CHECK(w.install(scores({{0*E + 6, 1001}}), plan));
        CHECK(w.gpu[0][6] >= 0 && w.gpu[0][5] < 0 && w.host[0][5].resident() && plan.reads == 0);
        // a file expert served into the ring and then promoted to VRAM is copied from its ring slot
        CHECK(w.serve(2, {11}));
        CHECK(w.install(scores({{2*E + 11, 1002}, {0*E + 6, 1001}}), plan));
        CHECK(w.gpu[2][11] >= 0 && plan.ring_copies == 1 && plan.reads == 0 && plan.demoted == 1);
        printf("PASS: relabel install: VRAM promotions and demotions through the ring, a ring occupant is copied, not read\n");
    }
    // ---- leases: a leased ring slot is never taken or relabeled -----------------------------------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        l2_install_plan plan;
        CHECK(w.serve(2, {10, 11}, false));   // generation not done: leased
        const int s10 = w.ledger.slot_of(2, 10), s11 = w.ledger.slot_of(2, 11);
        CHECK(w.install(scores({{2*E + 7, 945}, {2*E + 8, 944}, {2*E + 9, 943}}), plan));
        CHECK(plan.reads == 3);
        for (const auto & st : plan.steps) { CHECK(st.dst != s10 && st.dst != s11 && st.src != s10 && st.src != s11); }
        CHECK(w.ledger.slot_of(2, 10) == s10 && w.ledger.slot_of(2, 11) == s11);
        relabel_world v({6, 5}, false);
        v.load(scores({}));
        CHECK(v.serve(2, {10}, false));
        CHECK(!v.install(scores({{2*E + 10, 945}}), plan) && v.error.find("leased") != std::string::npos);
        printf("PASS: relabel install: leased ring slots are skipped, a leased occupant is never relabeled\n");
    }
    // ---- a small ring splits the file moves into read batches, no slot used twice in one ---------
    {
        relabel_world w({1, 1}, false, {6, 3, 3}, {8, 4, 3}, 4);
        w.load(scores({}));
        l2_install_plan plan;
        // four file experts of class 0 promoted to VRAM in one rotation batch, the four they replace
        // drop to the file: all four reads need the one ring slot
        CHECK(w.install(scores({{2*E + 8, 1010}, {2*E + 9, 1009}, {2*E + 10, 1008}, {2*E + 11, 1007},
                                {0*E + 2, 0}, {0*E + 3, 0}, {0*E + 4, 0}, {0*E + 5, 0}}), plan));
        CHECK(plan.reads == 4 && plan.batches == 4 && plan.demoted == 0);
        printf("PASS: relabel install: a one-slot ring reads in %d batches without reusing a slot inside one\n", plan.batches);
    }
    // ---- prompt/decode switch: two plans alternating, only relabels ------------------------------
    {
        relabel_world w({6, 5}, false);
        const auto prompt = scores({});
        const auto decode = scores({{2*E + 10, 945}, {2*E + 11, 944}, {1*E + 11, 855}, {3*E + 11, 615}});
        w.load(prompt);
        // generation demands the decode plan's host experts before that plan is installed
        CHECK(w.serve(2, {10, 11}) && w.serve(1, {11}) && w.serve(3, {11}));
        size_t reads = 0;
        for (int round = 0; round < 6; ++round) {
            l2_install_plan plan;
            CHECK(w.install(round % 2 == 0 ? decode : prompt, plan));
            reads += plan.reads;
            CHECK(plan.relabeled == 4 && plan.demoted == 4);
        }
        CHECK(reads == 0 && w.ledger.ring_count(0) == 6 && w.ledger.ring_count(1) == 5);
        printf("PASS: relabel install: a prompt/decode switch back and forth relabels only, nothing is read\n");
    }
    // ---- random plans and demands, both movers, several ring sizes ---------------------------------
    for (int inclusive = 0; inclusive < 2; ++inclusive) {
        for (int ring : {1, 3, 8}) {
            relabel_world w({ring, ring}, inclusive != 0, inclusive ? std::vector<int>{4, 2, 2} : std::vector<int>{6, 3, 3},
                inclusive ? std::vector<int>{10, 5, 4} : std::vector<int>{8, 4, 3}, 1 + ring%3);
            uint32_t rng = 99u + 7u*(uint32_t) ring + 1000u*(uint32_t) inclusive;
            auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
            std::vector<uint64_t> score((size_t) relabel_world::layers*E);
            for (auto & v : score) { v = next()%64; }
            w.load(score);
            size_t reads = 0, from_ring = 0, installs = 0;
            for (int step = 0; step < 400; ++step) {
                if (next()%4 == 0) {
                    // a new plan that keeps part of the old ranking
                    for (auto & v : score) { if (next()%3 == 0) { v = next()%64; } }
                    l2_install_plan plan;
                    if (!w.install(score, plan)) { fprintf(stderr, "step %d: %s\n", step, w.error.c_str()); CHECK(false); }
                    reads += plan.reads; from_ring += plan.relabeled + plan.ring_copies; ++installs;
                    continue;
                }
                const int layer = int(next()%relabel_world::layers);
                std::vector<int> ids;
                for (int e = 0; e < E && (int) ids.size() < ring; ++e) {
                    if (w.gpu[layer][e] < 0 && !w.host[layer][e].resident() && next()%3 == 0) { ids.push_back(e); }
                }
                if (!w.serve(layer, ids)) { fprintf(stderr, "serve: %s\n", w.error.c_str()); CHECK(false); }
                CHECK(w.check());
            }
            CHECK(installs > 50 && reads + from_ring > 0);
            printf("  %s ring %d: %zu installs, %zu file moves read, %zu served from the ring\n",
                inclusive ? "inclusive" : "exclusive", ring, installs, reads, from_ring);
        }
    }
    printf("PASS: relabel install: random plans and demands keep homes, ring occupants and ring sizes consistent\n");
    return 0;
}

// Early routes: the early service of both ledgers (a demand served before its generation is
// published), the hint helpers and the VRAM staging slot plan (expert-hash-early.h).
static int early_tests() {
    auto none = [](int, int) { return false; };
    std::string why;
    // ---- class ledger: early misses are read into reusable slots, the real service hits them ------
    {
        l2_class_ledger ledger = make_class_ledger();
        ledger.set_done(0, 4);
        // early: slot leased to the done counter, so it is reusable at once
        auto early = ledger.service(0, {1, 2, 1}, ledger.done(0), none, l2_no_quiet(), true);
        CHECK(early.ok && early.misses == 2 && early.hits == 0 && early.reads.size() == 6 && early.early_hits == 0);
        const int s1 = ledger.slot_of(0, 1), s2 = ledger.slot_of(0, 2);
        CHECK(s1 >= 0 && s2 >= 0 && s1 != s2);
        // the real generation 5: both are early hits, not ring hits
        std::set<int> marked = {1, 2};
        auto quiet = [&](int l, int e) { return l == 0 && marked.count(e) != 0; };
        auto real = ledger.service(0, {2, 1, 3}, 5, none, quiet);
        CHECK(real.ok && real.early_hits == 2 && real.hits == 0 && real.misses == 1 && real.reads.size() == 3);
        // now leased to generation 5: another layer of class 0 cannot take them while 5 runs
        // the early slots are leased to generation 5 now: layer 2 of the same class finds one free slot
        auto other = ledger.service(2, {7, 8}, 1, none);
        CHECK(!other.ok);
        CHECK(ledger.slot_of(0, 1) == s1 && ledger.slot_of(0, 2) == s2 && ledger.check(why));
        printf("PASS: class ledger early service: reusable early slots, early hits outside the ring hit count\n");
    }
    // ---- class ledger: an early hit never shortens the lease of a pending generation ------------
    {
        l2_class_ledger ledger = make_class_ledger();
        ledger.set_done(0, 4);
        ledger.set_done(2, 0);
        // generation 5 of layer 0 holds expert 3 (not finished: done 4)
        CHECK(ledger.service(0, {3}, 5, none).ok);
        const int held = ledger.slot_of(0, 3);
        // an early demand for the same expert only refreshes it
        auto early = ledger.service(0, {3}, ledger.done(0), none, l2_no_quiet(), true);
        CHECK(early.ok && early.hits == 1 && early.misses == 0 && early.reads.empty());
        // the three other ring slots go to layer 2; a fourth expert finds no reusable slot, because
        // the held slot is still leased to generation 5
        CHECK(ledger.service(2, {7, 8, 9}, 1, none).ok);
        auto full = ledger.service(2, {10}, 2, none);
        CHECK(!full.ok && ledger.slot_of(0, 3) == held);
        ledger.set_done(0, 5);
        auto after = ledger.service(2, {10}, 2, none);
        CHECK(after.ok && after.misses == 1 && ledger.check(why));
        printf("PASS: class ledger early hit keeps the lease of the generation that holds the slot\n");
    }
    // ---- ring ledger: the same rules -------------------------------------------------------------
    {
        l2_ledger ledger;
        ledger.reset(4, 16, 2, 2, 3);
        ledger.set_done(0, 7);
        CHECK(ledger.service(0, {1}, 8, [](int, int) { return false; }).ok);   // leased to 8
        auto early = ledger.service(0, {1, 2}, ledger.done(0), [](int, int) { return false; }, l2_no_quiet(), true);
        CHECK(early.ok && early.hits == 1 && early.misses == 1);
        // expert 2's slot is reusable at once, expert 1's is not
        auto next = ledger.service(1, {5}, 1, [](int, int) { return false; });
        CHECK(next.ok && next.evicted.size() == 1 && next.evicted[0].expert == 2);
        CHECK(!ledger.service(1, {6}, 1, [](int, int) { return false; }).ok);
        ledger.set_done(0, 8);
        auto quiet_hit = ledger.service(0, {1}, 9, [](int, int) { return false; }, [](int, int e) { return e == 1; });
        CHECK(quiet_hit.ok && quiet_hit.early_hits == 1 && quiet_hit.hits == 0);
        printf("PASS: ring ledger early service: no lease on early misses, early hits keep leases\n");
    }
    // ---- hint helpers and the staging plan ---------------------------------------------------------
    {
        hash_early_mode m;
        CHECK(parse_hash_early(nullptr, m) && !m.any());
        CHECK(parse_hash_early("0", m) && !m.any() && parse_hash_early("off", m) && !m.any());
        CHECK(parse_hash_early("ssd", m) && m.ssd && !m.vram && std::string(hash_early_name(m)) == "ssd");
        CHECK(parse_hash_early("vram", m) && !m.ssd && m.vram);
        CHECK(parse_hash_early("both", m) && m.ssd && m.vram && std::string(hash_early_name(m)) == "both");
        CHECK(!parse_hash_early("1", m) && !m.any() && !parse_hash_early("SSD", m));
        const int32_t ids[] = {5, 3, 5, -1, 16, 7, 3, 15};
        CHECK((hint_distinct_ids(ids, 8, 16) == std::vector<int>{5, 3, 7, 15}));
        CHECK(hint_distinct_ids(nullptr, 3, 16).empty() && hint_distinct_ids(ids, 0, 16).empty());

        std::vector<int32_t> owners(3, -1);
        stage_plan p = plan_stage(owners, {5, 6});
        CHECK(p.changed() && p.clears.empty() && p.keeps.empty() && p.over == 0);
        CHECK((p.copies == std::vector<std::pair<int, int>>{{5, 0}, {6, 1}}) && (owners == std::vector<int32_t>{5, 6, -1}));
        // unchanged: nothing to do but the entries
        p = plan_stage(owners, {6, 5});
        CHECK(!p.changed() && (p.keeps == std::vector<std::pair<int, int>>{{6, 1}, {5, 0}}));
        // a new expert takes the free slot before any slot of an expert no longer wanted
        p = plan_stage(owners, {6, 7});
        CHECK(p.clears.empty() && (p.copies == std::vector<std::pair<int, int>>{{7, 2}}) && p.keeps.size() == 1);
        CHECK((owners == std::vector<int32_t>{5, 6, 7}));
        // full: the unwanted owners' slots are reused, their entries cleared first; the rest is over
        p = plan_stage(owners, {8, 7, 9, 10});
        CHECK((p.keeps == std::vector<std::pair<int, int>>{{7, 2}}));
        CHECK((p.clears == std::vector<std::pair<int, int>>{{5, 0}, {6, 1}}));
        CHECK((p.copies == std::vector<std::pair<int, int>>{{8, 0}, {9, 1}}) && p.over == 1);
        CHECK((owners == std::vector<int32_t>{8, 9, 7}));
        // duplicates take one slot
        {
            std::vector<int32_t> dup(3, -1);
            const stage_plan d = plan_stage(dup, {4, 4, 5, 4});
            CHECK(d.copies.size() == 2 && (dup == std::vector<int32_t>{4, 5, -1}) && d.over == 0);
            const stage_plan k = plan_stage(dup, {5, 5});
            CHECK(!k.changed() && k.keeps.size() == 1);
        }
        // nothing wanted: nothing changes, the staged experts stay (their entries stay valid)
        p = plan_stage(owners, {});
        CHECK(!p.changed() && p.keeps.empty() && (owners == std::vector<int32_t>{8, 9, 7}));
        // random plans: every wanted expert (up to the slot count) ends staged, each slot once, and a
        // slot changes owner only through a copy (with a clear when it had one)
        uint32_t rng = 12345;
        auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
        std::vector<int32_t> own(6, -1);
        for (int round = 0; round < 5000; ++round) {
            std::vector<int32_t> before = own;
            const int n = int(next()%10);
            std::vector<int32_t> raw;
            for (int i = 0; i < n; ++i) { raw.push_back(int32_t(next()%12)); }
            const std::vector<int> wanted = hint_distinct_ids(raw.data(), (int) raw.size(), 12);
            const stage_plan q = plan_stage(own, wanted);
            CHECK(q.keeps.size() + q.copies.size() + (size_t) q.over == wanted.size());
            CHECK(q.keeps.size() + q.copies.size() <= own.size());
            std::set<int> slots;
            for (const auto & k : q.keeps)  { CHECK(before[(size_t) k.second] == k.first && slots.insert(k.second).second); }
            for (const auto & c : q.copies) { CHECK(own[(size_t) c.second] == c.first && slots.insert(c.second).second); }
            for (const auto & c : q.clears) {
                CHECK(before[(size_t) c.second] == c.first);
                bool reused = false;
                for (const auto & k : q.copies) { reused = reused || k.second == c.second; }
                CHECK(reused);
            }
            for (size_t k = 0; k < own.size(); ++k) {
                bool copied = false;
                for (const auto & c : q.copies) { copied = copied || c.second == (int) k; }
                if (!copied) { CHECK(own[k] == before[k]); }
                if (copied && before[k] >= 0) {
                    bool cleared = false;
                    for (const auto & c : q.clears) { cleared = cleared || c.second == (int) k; }
                    CHECK(cleared);
                }
            }
            std::set<int32_t> distinct;
            for (int32_t e : own) { CHECK(e < 0 || distinct.insert(e).second); }
        }
        printf("PASS: early-route switch parsing, distinct hint ids and the staging slot plan\n");
    }
    // ---- staging owners and the slot table across installs ----------------------------------------
    {
        // host model of one layer's table: a plan's clears (only where the table still names the slot),
        // then its keeps and copies; an install republishes the table from the plan (no staging entry)
        constexpr int experts = 12, first = 4, slots = 3;
        auto apply = [&](std::vector<int32_t> & table, const stage_plan & p) {
            for (const auto & c : p.clears) { if (table[(size_t) c.first] == first + c.second) { table[(size_t) c.first] = -1; } }
            for (const auto & c : p.keeps)  { table[(size_t) c.first] = first + c.second; }
            for (const auto & c : p.copies) { table[(size_t) c.first] = first + c.second; }
        };
        auto install = [&](std::vector<int32_t> & table, int static_expert) {
            table.assign(experts, -1);
            table[(size_t) static_expert] = 0;
        };
        std::string why;
        // the failure seen on the model: staged {5, 6, 7}, an install, then a hint naming only 5 and 8
        {
            std::vector<int32_t> table(experts, -1), owners(slots, -1);
            apply(table, plan_stage(owners, {5, 6, 7}));
            CHECK(stage_table_check(owners, table, first, why));
            install(table, 1);
            std::vector<int32_t> stale = owners;
            apply(table, plan_stage(stale, {5, 8}));   // without forgetting: 6 and 7 stay owners, unmapped
            CHECK(!stage_table_check(stale, table, first, why));
            install(table, 1);
            stage_forget(owners);
            apply(table, plan_stage(owners, {5, 8}));
            CHECK(stage_table_check(owners, table, first, why));
            CHECK(table[5] >= first && table[8] >= first && table[6] == -1 && table[7] == -1);
        }
        // an install that makes a staged expert static: after forgetting, the static entry stays
        {
            std::vector<int32_t> table(experts, -1), owners(slots, -1);
            apply(table, plan_stage(owners, {2, 3}));
            install(table, 3);
            stage_forget(owners);
            CHECK(stage_table_check(owners, table, first, why));
            apply(table, plan_stage(owners, {2}));     // the controller never stages a static expert
            CHECK(stage_table_check(owners, table, first, why) && table[3] == 0);
        }
        // random hints with installs in between keep the invariant
        uint32_t rng = 777;
        auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
        std::vector<int32_t> table(experts, -1), owners(slots, -1);
        for (int round = 0; round < 5000; ++round) {
            if (next()%7 == 0) {
                install(table, int(next()%experts));
                stage_forget(owners);
            }
            std::vector<int32_t> raw;
            for (int i = 0; i < 4; ++i) {
                const int e = int(next()%experts);
                if (table[(size_t) e] < 0 || table[(size_t) e] >= first) { raw.push_back(e); }   // host residents only
            }
            apply(table, plan_stage(owners, hint_distinct_ids(raw.data(), (int) raw.size(), experts)));
            CHECK(stage_table_check(owners, table, first, why));
        }
        printf("PASS: staging owners and slot table stay consistent across installs (an install forgets the staging)\n");
    }
    return 0;
}

// The unified policy of the class ledger (l2_unified_policy): the probation split, the insertion points,
// the ghost, the prompt rule, dissolve, sink and adopt, against a slow reference that keeps each class
// list as a plain vector.
struct reference_unified {
    struct slot_state {
        int layer = -1, expert = -1;
        uint32_t seq = 0;
        bool unhit = false;
    };
    double q = 0.1;
    uint32_t ghost = 32;
    bool prompt_lru = false;
    int pmin = 0;
    int experts = 0;
    std::vector<int> layer_storage;
    std::vector<std::vector<int>> order;          // [class] LRU end first
    std::vector<std::vector<slot_state>> slots;   // [class][slot]
    std::vector<uint32_t> done, step, last;
    l2_unified_stats stats;

    reference_unified(const std::vector<int> & storage, const std::vector<int> & n, int experts, double q, uint32_t ghost,
            bool prompt_lru, int pmin = 0) : q(q), ghost(ghost), prompt_lru(prompt_lru), pmin(pmin), experts(experts), layer_storage(storage) {
        order.resize(n.size());
        slots.resize(n.size());
        for (size_t s = 0; s < n.size(); ++s) {
            slots[s].assign((size_t) n[s], {});
            for (int i = 0; i < n[s]; ++i) { order[s].push_back(i); }
        }
        done.assign(storage.size(), 0u);
        step.assign(storage.size(), 0u);
        last.assign(storage.size()*(size_t) experts, 0u);
    }
    int slot_of(int layer, int expert) const {
        const auto & row = slots[(size_t) layer_storage[(size_t) layer]];
        for (size_t i = 0; i < row.size(); ++i) {
            if (row[i].layer == layer && row[i].expert == expert) { return int(i); }
        }
        return -1;
    }
    bool reusable(const slot_state & st) const { return st.layer < 0 || int32_t(done[(size_t) st.layer] - st.seq) >= 0; }
    void move(int s, int slot, l2_place where) {
        auto & o = order[(size_t) s];
        o.erase(std::find(o.begin(), o.end(), slot));
        if (where == l2_place::lru) {
            o.insert(o.begin(), slot);
        } else if (where == l2_place::mru) {
            o.push_back(slot);
        } else {
            const int n = (int) slots[(size_t) s].size();
            const size_t p = (size_t) std::min(n, std::max(int(q*double(n) + 1e-9), pmin));
            o.insert(o.begin() + (std::ptrdiff_t) std::min(p, o.size()), slot);
        }
    }
    bool service(int layer, const std::vector<int> & wanted, uint32_t seq, bool prompt) {
        const int s = layer_storage[(size_t) layer];
        const bool ring_rule = prompt && !prompt_lru;
        const uint32_t now = ring_rule ? step[(size_t) layer] : ++step[(size_t) layer];
        std::set<int> pinned;
        for (int e : wanted) {
            const int slot = slot_of(layer, e);
            if (slot >= 0) { pinned.insert(slot); }
        }
        for (int e : wanted) {
            int slot = slot_of(layer, e);
            auto & row = slots[(size_t) s];
            if (slot >= 0) {
                row[(size_t) slot].seq = seq;
                ++stats.hits;
                if (row[(size_t) slot].unhit && !ring_rule) { ++stats.probation_hits; row[(size_t) slot].unhit = false; }
                if (!ring_rule) { move(s, slot, l2_place::mru); }
                continue;
            }
            slot = -1;
            for (int cand : order[(size_t) s]) {
                if (!pinned.count(cand) && reusable(row[(size_t) cand])) { slot = cand; break; }
            }
            if (slot < 0) { return false; }
            slot_state & st = row[(size_t) slot];
            if (st.layer >= 0) {
                ++stats.evictions;
                stats.probation_evicted_unhit += st.unhit ? 1 : 0;
            }
            const uint32_t prev = last[(size_t) layer*(size_t) experts + (size_t) e];
            const bool pass = !ring_rule && prev != 0 && int32_t(now - prev) > 0 && uint32_t(now - prev) <= ghost;
            const l2_place where = ring_rule ? l2_place::lru : pass ? l2_place::mru : l2_place::probation;
            st = {layer, e, seq, where == l2_place::probation};
            ++stats.fills;
            ++(where == l2_place::lru ? stats.prompt_fills : where == l2_place::mru ? stats.mru_fills : stats.probation_fills);
            move(s, slot, where);
            pinned.insert(slot);
        }
        if (!ring_rule) {
            for (int e : wanted) { last[(size_t) layer*(size_t) experts + (size_t) e] = now; }
        }
        return true;
    }
    void sink(int s, int slot) {
        slots[(size_t) s][(size_t) slot] = {};
        move(s, slot, l2_place::lru);
        ++stats.sunk;
    }
    void discard() {
        for (size_t s = 0; s < slots.size(); ++s) {
            for (auto & st : slots[s]) { st = {}; }
            order[s].clear();
            for (int i = 0; i < (int) slots[s].size(); ++i) { order[s].push_back(i); }
        }
    }
};

static bool same_stats(const l2_unified_stats & a, const l2_unified_stats & b) {
    return a.hits == b.hits && a.fills == b.fills && a.mru_fills == b.mru_fills && a.probation_fills == b.probation_fills &&
        a.prompt_fills == b.prompt_fills && a.probation_hits == b.probation_hits &&
        a.probation_evicted_unhit == b.probation_evicted_unhit && a.evictions == b.evictions && a.sunk == b.sunk;
}

static int unified_tests() {
    auto none = [](int, int) { return false; };
    std::string why;
    auto all_ring = [](int layers, int experts, const std::vector<int> & storage, const std::vector<int> & n) {
        l2_class_ledger ledger;
        ledger.reset(layers, experts, storage, n, 3);
        for (size_t s = 0; s < n.size(); ++s) {
            for (int i = 0; i < n[s]; ++i) { ledger.make_ring(int(s), i); }
        }
        return ledger;
    };
    l2_unified_policy policy;
    policy.on = true; policy.probation = 0.10; policy.ghost = 4;
    // ---- the decode pool's ring admission: on by default under the policy, an explicit switch wins ----
    CHECK(l2_unified_pool_ring(-1, true) && !l2_unified_pool_ring(-1, false));
    CHECK(!l2_unified_pool_ring(0, true) && !l2_unified_pool_ring(0, false));
    CHECK(l2_unified_pool_ring(1, true) && l2_unified_pool_ring(1, false));
    printf("PASS: unified policy: the decode pool's ring admission defaults on under it, an explicit 0 or 1 wins\n");
    // ---- insertion points, ghost, hits ---------------------------------------------------------------
    {
        l2_class_ledger ledger = all_ring(2, 32, {0, 0}, {20});
        ledger.set_unified(policy);
        CHECK(ledger.unified() && ledger.probation_count(0) == 2 && ledger.check(why));
        uint32_t g = 0;
        auto serve = [&](std::vector<int> ids) {
            ++g;
            ledger.set_done(0, g - 1);
            return ledger.service(0, ids, g, none);
        };
        // a first miss never passes its own ghost: it lands with exactly the probation share below it
        auto first = serve({7});
        const int s7 = ledger.slot_of(0, 7);
        CHECK(first.ok && first.misses == 1 && s7 == 0 && ledger.order(0)[2] == s7 && ledger.check(why));
        CHECK(ledger.unified_stats().probation_fills == 1 && ledger.unified_stats().mru_fills == 0);
        // a hit goes to the most recently used end and counts as a probation hit once
        CHECK(serve({7}).hits == 1 && ledger.order(0).back() == s7 && ledger.unified_stats().probation_hits == 1);
        CHECK(serve({7}).hits == 1 && ledger.unified_stats().probation_hits == 1);
        // a miss within K decode generations of the layer's last miss passes the ghost
        CHECK(serve({9}).ok);
        const int s9 = ledger.slot_of(0, 9);
        CHECK(ledger.order(0)[2] == s9 && !ledger.sink(0, s9));   // leased until the layer is done
        ledger.set_done(0, g);
        CHECK(ledger.sink(0, s9) && ledger.order(0)[0] == s9 && ledger.slot_of(0, 9) < 0);
        CHECK(serve({9}).misses == 1 && ledger.order(0).back() == ledger.slot_of(0, 9));
        CHECK(ledger.unified_stats().mru_fills == 1 && ledger.check(why));
        // ... and one older than K does not
        ledger.set_done(0, g);
        CHECK(ledger.sink(0, ledger.slot_of(0, 9)));
        for (int i = 0; i < 5; ++i) { CHECK(serve({7}).ok); }
        CHECK(serve({9}).misses == 1 && ledger.order(0)[2] == ledger.slot_of(0, 9) && ledger.check(why));
        printf("PASS: unified policy: misses at the probation point, ghost passes at the MRU end, hits move up\n");
    }
    // ---- the probation minimum, capped at the class's slots -------------------------------------------
    {
        l2_class_ledger ledger = all_ring(2, 32, {0, 0}, {20});
        ledger.set_unified(policy);
        ledger.set_probation_min(0, 5);
        CHECK(ledger.probation_count(0) == 5 && ledger.check(why));
        CHECK(ledger.service(0, {7}, 1, none).ok && ledger.order(0)[5] == ledger.slot_of(0, 7) && ledger.check(why));
        ledger.set_probation_min(0, 1);
        CHECK(ledger.probation_count(0) == 2 && ledger.check(why));   // the share is larger
        ledger.set_probation_min(0, 50);
        CHECK(ledger.probation_count(0) == 20 && ledger.check(why));  // capped: the probation point is the MRU end
        ledger.set_done(0, 1);
        CHECK(ledger.service(0, {8}, 2, none).ok && ledger.order(0).back() == ledger.slot_of(0, 8) && ledger.check(why));
        // residents join their ring: an expert at the requested point, a spare at the LRU end
        l2_class_ledger mixed;
        mixed.reset(2, 16, {0, 0}, {10}, 3);
        CHECK(mixed.set_resident(0, 0, 1, 3) && mixed.set_resident(0, 1, -1, -1));
        for (int slot = 2; slot < 10; ++slot) { CHECK(mixed.make_ring(0, slot)); }
        mixed.set_unified(policy);
        mixed.set_probation_min(0, 3);
        CHECK(mixed.residents_to_ring(l2_place::probation) == 1 && mixed.ring_count(0) == 10 && mixed.check(why));
        // the spare joined below the expert after it, which moved it one slot up
        CHECK(mixed.order(0)[4] == 0 && mixed.order(0)[0] == 1 && mixed.slot_of(1, 3) == 0);
        printf("PASS: unified policy: the probation minimum per class, capped at its slots; residents join the ring\n");
    }
    // ---- prompt rule ---------------------------------------------------------------------------------
    for (bool prompt_lru : {false, true}) {
        l2_unified_policy p = policy;
        p.prompt_lru = prompt_lru;
        l2_class_ledger ledger = all_ring(2, 32, {0, 0}, {20});
        ledger.set_unified(p);
        uint32_t g = 0;
        auto serve = [&](std::vector<int> ids) {
            ++g;
            ledger.set_done(0, g - 1);
            return ledger.service(0, ids, g, none);
        };
        CHECK(serve({3}).ok);
        const std::vector<int> before = ledger.order(0);
        ledger.set_prompt(true);
        CHECK(serve({3, 11}).ok);
        const int s11 = ledger.slot_of(0, 11);
        if (!prompt_lru) {
            // the hit stays where it was, the miss is the next to go, and no ghost step was taken
            std::vector<int> expect = before;
            expect.erase(std::find(expect.begin(), expect.end(), s11));
            expect.insert(expect.begin(), s11);
            CHECK(ledger.order(0) == expect && ledger.unified_stats().prompt_fills == 1);
            ledger.set_prompt(false);
            ledger.set_done(0, g);
            CHECK(ledger.sink(0, s11) && serve({11}).misses == 1 && ledger.order(0)[2] == ledger.slot_of(0, 11));
        } else {
            CHECK(ledger.order(0).back() == ledger.slot_of(0, 3) && ledger.order(0)[2] == s11);
            CHECK(ledger.unified_stats().prompt_fills == 0 && ledger.unified_stats().probation_fills == 2);
        }
        CHECK(ledger.check(why));
    }
    printf("PASS: unified policy: a prompt generation fills at the LRU end and moves no hit (or follows decode)\n");
    // ---- dissolve, sink, adopt -----------------------------------------------------------------------
    {
        l2_class_ledger ledger;
        ledger.reset(2, 16, {0, 0}, {8}, 3);
        for (int e = 0; e < 4; ++e) { CHECK(ledger.set_resident(0, e, 0, e)); }
        CHECK(ledger.set_resident(0, 4, -1, -1));
        for (int slot = 5; slot < 8; ++slot) { CHECK(ledger.make_ring(0, slot)); }
        l2_unified_policy p = policy;
        p.probation = 0.25;
        ledger.set_unified(p);
        CHECK(ledger.probation_count(0) == 0 && ledger.check(why));
        CHECK(ledger.dissolve([](int, int e) { return 10.0 - e; }) == 5);
        CHECK((ledger.order(0) == std::vector<int>{4, 5, 6, 7, 3, 2, 1, 0}) && ledger.ring_count(0) == 8);
        CHECK(ledger.probation_count(0) == 2 && ledger.slot_of(0, 0) == 0 && ledger.check(why));
        CHECK(ledger.sink(0, 0) && ledger.order(0)[0] == 0 && ledger.slot_of(0, 0) < 0);
        CHECK(ledger.adopt(0, 0, 0, 9, l2_place::probation) && ledger.order(0)[2] == 0 && ledger.slot_of(0, 9) == 0);
        CHECK(!ledger.adopt(0, 0, 0, 10, l2_place::mru) && !ledger.adopt(0, 4, 0, 9, l2_place::mru) &&
              !ledger.adopt(0, 4, 2, 1, l2_place::mru));
        CHECK(ledger.adopt(0, 4, 1, 5, l2_place::mru) && ledger.order(0).back() == 4 && ledger.check(why));
        // a leased slot is not sunk
        CHECK(ledger.service(1, {5}, 1, none).hits == 1 && !ledger.sink(0, 4));
        ledger.set_done(1, 1);
        CHECK(ledger.sink(0, 4) && ledger.check(why));
        // after a dissolve with the policy off, the same order and no split
        l2_class_ledger off;
        off.reset(2, 16, {0, 0}, {8}, 3);
        for (int e = 0; e < 4; ++e) { CHECK(off.set_resident(0, e, 0, e)); }
        CHECK(off.set_resident(0, 4, -1, -1));
        for (int slot = 5; slot < 8; ++slot) { CHECK(off.make_ring(0, slot)); }
        CHECK(off.dissolve([](int, int e) { return 10.0 - e; }) == 5);
        CHECK((off.order(0) == std::vector<int>{4, 5, 6, 7, 3, 2, 1, 0}) && off.probation_count(0) == 0 && off.check(why));
        printf("PASS: unified policy: residents dissolve into the rings in score order, sink and adopt\n");
    }
    // ---- the policy off is the plain ring, also after it was on --------------------------------------
    {
        const std::vector<int> storage = {0, 1, 0, 1};
        l2_class_ledger plain = all_ring(4, 16, storage, {6, 7});
        l2_class_ledger toggled = all_ring(4, 16, storage, {6, 7});
        toggled.set_unified(policy);
        toggled.set_unified(l2_unified_policy());
        uint32_t rng = 4242u;
        auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
        std::vector<uint32_t> seq(4, 0u);
        for (int step = 0; step < 2000; ++step) {
            const int layer = int(next()%4);
            for (int l = 0; l < 4; ++l) {
                if (next()%2 == 0) { plain.set_done(l, seq[(size_t) l]); toggled.set_done(l, seq[(size_t) l]); }
            }
            std::vector<int> ids;
            for (int i = 0, n = 1 + int(next()%4); i < n; ++i) { ids.push_back(int(next()%9)); }
            const uint32_t g = ++seq[(size_t) layer];
            const auto a = plain.service(layer, ids, g, none);
            const auto b = toggled.service(layer, ids, g, none);
            CHECK(a.ok == b.ok && a.hits == b.hits && a.misses == b.misses && a.reads.size() == b.reads.size());
            for (size_t i = 0; i < a.reads.size(); ++i) { CHECK(a.reads[i].slot == b.reads[i].slot); }
            if (!a.ok) { plain.discard(); toggled.discard(); }
            CHECK(plain.order(0) == toggled.order(0) && plain.order(1) == toggled.order(1));
        }
        CHECK(!toggled.unified() && toggled.check(why));
        printf("PASS: unified policy off: the ledger serves exactly as the plain ring\n");
    }
    // ---- random demands against the slow reference ---------------------------------------------------
    {
        const std::vector<int> storage = {0, 1, 0, 1};
        const int experts = 24;
        int runs = 0;
        for (double q : {0.0, 0.1, 0.25}) for (uint32_t k : {1u, 4u, 32u}) for (bool prompt_lru : {false, true}) for (int pmin : {0, 4}) {
            const std::vector<int> n = {10, 23};
            l2_class_ledger ledger = all_ring(4, experts, storage, n);
            l2_unified_policy p;
            p.on = true; p.probation = q; p.ghost = k; p.prompt_lru = prompt_lru;
            ledger.set_unified(p);
            for (int c = 0; c < 2; ++c) { ledger.set_probation_min(c, pmin); }
            reference_unified ref(storage, n, experts, q, k, prompt_lru, pmin);
            uint32_t rng = 99u + 7u*(uint32_t) runs;
            auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
            std::vector<uint32_t> seq(4, 0u);
            int served = 0;
            for (int step = 0; step < 3000; ++step) {
                const int layer = int(next()%4);
                const int s = storage[(size_t) layer];
                for (int l = 0; l < 4; ++l) {
                    if (next()%3 != 0) { ledger.set_done(l, seq[(size_t) l]); ref.done[(size_t) l] = seq[(size_t) l]; }
                }
                if (next()%17 == 0) {
                    const int slot = int(next()%(uint32_t) n[(size_t) s]);
                    const bool can = ref.reusable(ref.slots[(size_t) s][(size_t) slot]);
                    CHECK(ledger.sink(s, slot) == can);
                    if (can) { ref.sink(s, slot); }
                }
                const bool prompt = next()%5 == 0;
                std::vector<int> ids, wanted;
                std::vector<bool> seen((size_t) experts, false);
                for (int i = 0, m = 1 + int(next()%(prompt ? 6u : 3u)); i < m; ++i) {
                    const int e = next()%3 == 0 ? int(next()%experts) : int(next()%8);
                    ids.push_back(e);
                    if (!seen[(size_t) e]) { seen[(size_t) e] = true; wanted.push_back(e); }
                }
                const uint32_t g = ++seq[(size_t) layer];
                ledger.set_prompt(prompt);
                const bool ref_ok = ref.service(layer, wanted, g, prompt);
                const auto got = ledger.service(layer, ids, g, none);
                CHECK(got.ok == ref_ok);
                if (!ref_ok) {
                    ledger.discard();
                    ref.discard();
                } else {
                    ++served;
                }
                CHECK(ledger.order(0) == ref.order[0] && ledger.order(1) == ref.order[1]);
                for (int e : wanted) { CHECK(ledger.slot_of(layer, e) == (ref_ok ? ref.slot_of(layer, e) : -1)); }
                if (step%97 == 0) { CHECK(ledger.check(why)); }
            }
            CHECK(served > 2000 && ledger.check(why) && same_stats(ledger.unified_stats(), ref.stats));
            CHECK(ledger.unified_stats().probation_fills + ledger.unified_stats().mru_fills + ledger.unified_stats().prompt_fills ==
                  ledger.unified_stats().fills);
            ++runs;
        }
        printf("PASS: unified policy: %d random runs (probation share and minimum, ghost K, prompt rule) match the slow reference\n", runs);
    }
    return 0;
}

// The unified policy's installs on the relabel byte model: no host residents but the demotions of the
// install, which settle_unified turns into ring occupants; plan_unified_reseed brings each class ring to
// the best experts of the plan's scores and lays it out in score order.
//
// With `active` (per storage class) the classes outside it are the ones of a model that has not bound yet:
// their layers keep their VRAM slices and host residents, and the whole install leaves their lists alone.
static bool unified_install(relabel_world & w, const std::vector<uint64_t> & score, int reseed, l2_install_plan & plan,
        size_t & reads, int & demoted, const std::vector<uint8_t> * active = nullptr) {
    const int E = relabel_world::experts, L = relabel_world::layers;
    auto in_policy = [&](int s) { return active == nullptr || (*active)[(size_t) s] != 0; };
    auto held = [&](int l) { return !in_policy(w.storage_of[(size_t) w.geo.layer_class[l]]); };
    expert_slot_table gs, hs;
    w.choose(score, gs, hs);
    hs.assign(L, {});
    std::vector<int> room = w.layout.host;
    for (int l = 0; l < L; ++l) {
        const int c = w.geo.layer_class[l];
        if (held(l)) {
            gs[l].clear();
            for (int e = 0; e < E; ++e) {
                if (w.gpu[l][e] >= 0) { gs[l].push_back(e); }
                if (w.host[l][e].resident()) { hs[l].push_back(e); }
            }
            continue;
        }
        for (int e = 0; e < E; ++e) {
            if (w.gpu[l][e] >= 0 && room[c] > 0 && std::find(gs[l].begin(), gs[l].end(), e) == gs[l].end()) {
                hs[l].push_back(e);
                --room[c];
            }
        }
    }
    const install_transaction tx = plan_install(w.geo, gs, hs, w.gpu, w.host, w.vram_caps, w.layout, w.layout, w.spares,
        {false, w.spare}, true, true);
    if (!tx.valid) { return w.fail("plan_install: " + tx.reason); }
    plan = plan_relabel_install(w.ledger, w.home_slot, w.geo.layer_class, w.storage_of, w.host, tx.host, tx.moves, true, active);
    if (!plan.ok) { return w.fail("relabel: " + plan.reason); }
    for (size_t k = 0; k < tx.moves.size(); ++k) {
        const install_move & m = tx.moves[k];
        const l2_install_step & st = plan.steps[k];
        const int s = w.storage_of[m.cls], key = m.layer*E + m.expert;
        if (m.from.storage == expert_storage::file) {
            const int slot = m.to.storage == expert_storage::host ? st.dst : st.src;
            if (st.read) { w.content[s][(size_t) slot] = key; }
            if (w.content[s][(size_t) slot] != key) { return w.fail("a ring copy of a slot that does not hold the expert"); }
            w.vram[m.cls][(size_t) m.to.slot] = key;
        } else if (m.from.storage == expert_storage::vram) {
            if (w.vram[m.cls][(size_t) m.from.slot] != key) { return w.fail("VRAM model"); }
            w.content[s][(size_t) st.dst] = key;
        } else {
            return w.fail("a host source under the unified policy");
        }
    }
    w.gpu = tx.gpu_slots; w.spares = tx.gpu_spares;
    demoted = settle_unified(w.ledger, w.home_slot, w.gpu, active, &w.storage_of);
    {
        const expert_locations kept = tx.host;
        w.host = host_locations(expert_slot_table(L, std::vector<int32_t>(E, -1)));
        for (int l = 0; l < L; ++l) { if (held(l)) { w.host[l] = kept[l]; } }
    }
    reads = 0;
    if (reseed > 0) {
        auto sc = [&](int l, int e) { return double(score[(size_t) (l*E + e)]); };
        const auto got = plan_unified_reseed(w.ledger, std::vector<int>(L, E), w.gpu, sc, reseed == 2, active);
        for (const auto & r : got) { w.content[(size_t) w.ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*E + r.expert; }
        reads = got.size();
    }
    // every slot is a ring slot, every occupant holds its bytes, no VRAM resident keeps a ring copy
    std::string why;
    if (!w.ledger.check(why)) { return w.fail(why); }
    for (int s = 0; s < 2; ++s) {
        for (int slot = 0; slot < w.ledger.slots(s); ++slot) {
            if (in_policy(s) && w.ledger.role_of(s, slot) != slot_role::ring) { return w.fail("a slot outside the ring"); }
            int l = -1, e = -1;
            if (!w.ledger.occupant(s, slot, l, e)) { continue; }
            if (w.content[s][(size_t) slot] != l*E + e) { return w.fail("ring occupant bytes"); }
            if (in_policy(s) && w.gpu[l][e] >= 0) { return w.fail("a VRAM resident in the ring"); }
        }
    }
    for (int l = 0; l < L; ++l) for (int e = 0; e < E; ++e) {
        const int c = w.geo.layer_class[l];
        if (w.gpu[l][e] >= 0 && w.vram[c][(size_t) w.gpu[l][e]] != l*E + e) { return w.fail("VRAM home"); }
    }
    if (reseed == 2) {
        // each class ring holds exactly the best non-VRAM experts it has room for, in score order
        for (int s = 0; s < 2; ++s) {
            if (!in_policy(s)) { continue; }
            std::vector<std::pair<uint64_t, int>> cands;
            for (int l = 0; l < L; ++l) {
                if (w.ledger.storage_of(l) != s) { continue; }
                for (int e = 0; e < E; ++e) {
                    if (w.gpu[l][e] < 0) { cands.push_back({score[(size_t) (l*E + e)], l*E + e}); }
                }
            }
            std::sort(cands.begin(), cands.end(), [](auto a, auto b) { return a.first > b.first; });
            const size_t keep = std::min(cands.size(), (size_t) w.ledger.ring_count(s));
            for (size_t i = 0; i < keep; ++i) {
                if (w.ledger.slot_of(cands[i].second/E, cands[i].second%E) < 0) { return w.fail("a best expert left out"); }
            }
            uint64_t last = 0;
            bool first = true;
            for (int slot : w.ledger.order(s)) {
                int l = -1, e = -1;
                if (!w.ledger.occupant(s, slot, l, e)) {
                    if (!first) { return w.fail("an empty slot above an occupant"); }
                    continue;
                }
                const uint64_t v = score[(size_t) (l*E + e)];
                if (!first && v < last) { return w.fail("the ring is not in score order"); }
                last = v; first = false;
            }
        }
    }
    return true;
}

static int unified_install_tests() {
    const int E = relabel_world::experts;
    l2_unified_policy policy;
    policy.on = true; policy.probation = 0.10; policy.ghost = 8;
    // ---- the start: residents dissolve, the empty ring slots take the next best experts ------------------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        w.ledger.set_unified(policy);
        for (int s = 0; s < 2; ++s) { w.ledger.set_probation_min(s, w.rings[s]); }
        const int slots0 = w.ledger.slots(0), slots1 = w.ledger.slots(1);
        l2_install_plan plan;
        size_t reads = 0;
        int demoted = 0;
        CHECK(unified_install(w, scores({}), 2, plan, reads, demoted));
        CHECK(plan.demoted > 0 && demoted == 0 && reads > 0);
        CHECK(w.ledger.ring_count(0) == slots0 && w.ledger.ring_count(1) == slots1);
        CHECK(w.ledger.probation_count(0) == std::max(int(0.1*slots0), 6) && w.ledger.probation_count(1) == std::max(int(0.1*slots1), 5));
        // the same plan again reads nothing
        CHECK(unified_install(w, scores({}), 2, plan, reads, demoted) && reads == 0 && plan.reads == 0);
        printf("PASS: unified install: residents dissolve into the rings, the seed fills the empty slots in score order\n");
    }
    // ---- VRAM changes: promotions leave no ring copy, demotions join at the probation point ----------
    {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        w.ledger.set_unified(policy);
        for (int s = 0; s < 2; ++s) { w.ledger.set_probation_min(s, 2); }
        l2_install_plan plan;
        size_t reads = 0;
        int demoted = 0;
        CHECK(unified_install(w, scores({}), 2, plan, reads, demoted));
        // layer 0 expert 6 (in the ring) goes to VRAM, expert 5 (in VRAM) comes down; no reseed
        CHECK(w.ledger.slot_of(0, 6) >= 0 && w.gpu[0][5] >= 0);
        if (!unified_install(w, scores({{0*E + 6, 1001}}), 0, plan, reads, demoted)) {
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, w.error.c_str());
            return 1;
        }
        CHECK(w.gpu[0][6] >= 0 && w.ledger.slot_of(0, 6) < 0 && plan.ring_copies == 1 && plan.reads == 0);
        CHECK(demoted == 1 && w.ledger.slot_of(0, 5) >= 0 && w.ledger.order(0)[(size_t) w.ledger.probation_count(0)] == w.ledger.slot_of(0, 5));
        // a reseed in order only keeps the contents and sorts them
        CHECK(unified_install(w, scores({{0*E + 6, 1001}}), 1, plan, reads, demoted) && reads == 0);
        printf("PASS: unified install: VRAM promotions leave no ring copy, demotions join the ring at the probation point\n");
    }
    // ---- random plans, demands and reseed modes ---------------------------------------------------
    {
        for (int round = 0; round < 3; ++round) {
            relabel_world w({6, 5}, false);
            w.load(scores({}));
            w.ledger.set_unified(policy);
            for (int s = 0; s < 2; ++s) { w.ledger.set_probation_min(s, round); }
            l2_install_plan plan;
            size_t reads = 0;
            int demoted = 0;
            CHECK(unified_install(w, scores({}), 2, plan, reads, demoted));
            uint32_t rng = 31u + 17u*(uint32_t) round;
            auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
            for (int step = 0; step < 300; ++step) {
                if (next()%10 == 0) {
                    std::vector<uint64_t> sc = scores({});
                    for (auto & v : sc) { v = 1 + next()%1000; }
                    CHECK(unified_install(w, sc, int(next()%3), plan, reads, demoted));
                    continue;
                }
                const int layer = int(next()%relabel_world::layers);
                std::vector<int> ids;
                for (int i = 0, n = 1 + int(next()%3); i < n; ++i) { ids.push_back(int(next()%E)); }
                const uint32_t g = ++w.seq[(size_t) layer];
                w.ledger.set_prompt(next()%4 == 0);
                auto homed = [&](int l, int e) { return w.gpu[l][e] >= 0; };
                const l2_service got = w.ledger.service(layer, ids, g, homed);
                CHECK(got.ok);
                for (const l2_read & r : got.reads) {
                    w.content[(size_t) w.ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*E + r.expert;
                }
                w.ledger.set_done(layer, g);
            }
            std::string why;
            CHECK(w.ledger.check(why));
        }
        printf("PASS: unified install: random plans, demands and reseed modes keep the rings and the byte model consistent\n");
    }
    return 0;
}

// A model that binds later: its storage class stays as residents and ring through the start and every install
// of the others, then joins the policy on its own, while the other class keeps its list untouched.
struct ledger_snapshot {
    std::vector<int> cells;
    std::vector<int> order;
    int ring = 0;
    bool operator==(const ledger_snapshot & o) const { return cells == o.cells && order == o.order && ring == o.ring; }
};

static ledger_snapshot snapshot_of(const l2_class_ledger & ledger, int s) {
    ledger_snapshot out;
    for (int slot = 0; slot < ledger.slots(s); ++slot) {
        int l = -1, e = -1;
        out.cells.push_back((int) ledger.role_of(s, slot));
        out.cells.push_back(ledger.occupant(s, slot, l, e) ? l*relabel_world::experts + e : -1);
    }
    out.order = ledger.order(s);
    out.ring = ledger.ring_count(s);
    return out;
}

static int unified_late_tests() {
    const int E = relabel_world::experts, L = relabel_world::layers;
    l2_unified_policy policy;
    policy.on = true; policy.probation = 0.10; policy.ghost = 8;
    for (int round = 0; round < 3; ++round) {
        relabel_world w({6, 5}, false);
        w.load(scores({}));
        w.ledger.set_unified(policy);
        for (int s = 0; s < 2; ++s) { w.ledger.set_probation_min(s, round); }
        // storage class 1 (layers 1 and 3) belongs to the model that binds later
        std::vector<uint8_t> active = {1, 0};
        const ledger_snapshot late_before = snapshot_of(w.ledger, 1);
        const int late_slots = w.ledger.slots(1);
        l2_install_plan plan;
        size_t reads = 0;
        int demoted = 0;
        if (!unified_install(w, scores({}), 2, plan, reads, demoted, &active)) {
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, w.error.c_str());
            return 1;
        }
        CHECK(reads > 0 && snapshot_of(w.ledger, 1) == late_before);
        CHECK(w.host[1][3].resident() && w.host[3][3].resident() && !w.host[0][6].resident());
        // installs and generations of the active class, with the late class untouched (even by a plan that
        // wants other VRAM experts in its layers)
        uint32_t rng = 77u + 13u*(uint32_t) round;
        auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
        for (int step = 0; step < 200; ++step) {
            if (next()%8 == 0) {
                std::vector<uint64_t> sc = scores({});
                for (auto & v : sc) { v = 1 + next()%1000; }
                if (!unified_install(w, sc, int(next()%3), plan, reads, demoted, &active)) {
                    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, w.error.c_str());
                    return 1;
                }
                CHECK(snapshot_of(w.ledger, 1) == late_before);
                continue;
            }
            const int layer = (next()%2) ? 0 : 2;
            std::vector<int> ids;
            for (int i = 0, n = 1 + int(next()%3); i < n; ++i) { ids.push_back(int(next()%E)); }
            const uint32_t g = ++w.seq[(size_t) layer];
            auto homed = [&](int l, int e) { return w.gpu[l][e] >= 0; };
            const l2_service got = w.ledger.service(layer, ids, g, homed);
            CHECK(got.ok);
            for (const l2_read & r : got.reads) { w.content[(size_t) w.ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*E + r.expert; }
            w.ledger.set_done(layer, g);
        }
        CHECK(snapshot_of(w.ledger, 1) == late_before);
        // the model binds: its class joins, the other class's list is not touched
        const ledger_snapshot run_before = snapshot_of(w.ledger, 0);
        const std::vector<uint8_t> scope = {0, 1};
        std::vector<uint64_t> sc = scores({});
        for (auto & v : sc) { v = 1 + next()%1000; }
        auto score = [&](int l, int e) { return double(sc[(size_t) (l*E + e)]); };
        const int joined = w.ledger.residents_to_ring(l2_place::probation, &scope);
        w.ledger.sink_empty(&scope);
        for (int c = 0; c < 3; ++c) {
            if (w.storage_of[(size_t) c] == 1) { std::fill(w.home_slot[(size_t) c].begin(), w.home_slot[(size_t) c].end(), -1); }
        }
        for (int l = 0; l < L; ++l) { if (w.storage_of[(size_t) w.geo.layer_class[l]] == 1) { w.host[l] = std::vector<expert_location>(E); } }
        active = {1, 1};
        CHECK(joined > 0);
        const auto got = plan_unified_reseed(w.ledger, std::vector<int>(L, E), w.gpu, score, true, &scope);
        for (const auto & r : got) { w.content[(size_t) w.ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*E + r.expert; }
        CHECK(snapshot_of(w.ledger, 0) == run_before);
        std::string why;
        CHECK(w.ledger.check(why));
        CHECK(w.ledger.ring_count(1) == late_slots);
        // every slot of the class is a ring slot, holds its bytes, and the best experts it has room for are there, in score order
        std::vector<std::pair<double, int>> cands;
        for (int l = 0; l < L; ++l) {
            if (w.ledger.storage_of(l) != 1) { continue; }
            for (int e = 0; e < E; ++e) { if (w.gpu[l][e] < 0) { cands.push_back({score(l, e), l*E + e}); } }
        }
        std::sort(cands.begin(), cands.end(), [](auto a, auto b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min(cands.size(), (size_t) late_slots); ++i) {
            CHECK(w.ledger.slot_of(cands[i].second/E, cands[i].second%E) >= 0);
        }
        double last = 0.0;
        bool first = true;
        for (int slot : w.ledger.order(1)) {
            CHECK(w.ledger.role_of(1, slot) == slot_role::ring);
            int l = -1, e = -1;
            if (!w.ledger.occupant(1, slot, l, e)) { CHECK(first); continue; }
            CHECK(w.content[1][(size_t) slot] == l*E + e && w.gpu[l][e] < 0);
            const double v = score(l, e);
            CHECK(first || v >= last);
            last = v; first = false;
        }
        // the whole cache runs under the policy from here on
        for (int step = 0; step < 150; ++step) {
            if (next()%8 == 0) {
                std::vector<uint64_t> s2 = scores({});
                for (auto & v : s2) { v = 1 + next()%1000; }
                if (!unified_install(w, s2, int(next()%3), plan, reads, demoted)) {
                    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, w.error.c_str());
                    return 1;
                }
                continue;
            }
            const int layer = int(next()%L);
            std::vector<int> ids;
            for (int i = 0, n = 1 + int(next()%3); i < n; ++i) { ids.push_back(int(next()%E)); }
            const uint32_t g = ++w.seq[(size_t) layer];
            auto homed = [&](int l, int e) { return w.gpu[l][e] >= 0; };
            const l2_service served = w.ledger.service(layer, ids, g, homed);
            CHECK(served.ok);
            for (const l2_read & r : served.reads) { w.content[(size_t) w.ledger.storage_of(r.layer)][(size_t) r.slot] = r.layer*E + r.expert; }
            w.ledger.set_done(layer, g);
        }
        CHECK(w.ledger.check(why));
    }
    printf("PASS: unified late bind: a model's class stays untouched through the start and the others' installs, then joins on its own\n");
    return 0;
}

int main() {
    constexpr size_t mib = 1024*1024, stride = 4*mib;
    auto plan = plan_l2_ring(512, 8, 512, 1, stride, 0, 0, true);
    CHECK(plan.valid && plan.prompt_slots == 512 && plan.decode_slots == 8 + 128);
    CHECK(plan.reserve_bytes == 512*mib);
    plan = plan_l2_ring(512, 8, 32, 4*5, stride, 0, 0, true);
    CHECK(plan.valid && plan.prompt_floor == 256 && plan.decode_floor == 160 && plan.decode_slots == 256);
    plan = plan_l2_ring(512, 8, 512, 1, stride, 4*mib, 4*mib, true);
    CHECK(plan.valid && plan.prompt_slots == 512 && plan.decode_slots == 8 && !plan.note.empty());
    plan = plan_l2_ring(512, 8, 16, 1, stride, 1024*mib, 256*mib, false);
    CHECK(plan.valid && plan.prompt_slots == 256 && plan.decode_slots == 256);
    plan = plan_l2_ring(1, 1, 1, 1, 4096, 0, 0, true);
    CHECK(plan.valid && plan.prompt_slots == 1 && plan.decode_slots == 1);
    CHECK(!plan_l2_ring(512, 0, 512, 1, stride, 0, 0, true).valid);
    CHECK(!plan_l2_ring(512, 8, 0, 1, stride, 0, 0, true).valid);
    CHECK(!plan_l2_ring(512, 8, 512, 1, SIZE_MAX, 0, 0, true).valid);
    printf("PASS: prompt/decode/speculative batch floors, reserve, explicit minimum and overflow\n");
    const int layers  = 4;
    const int experts = 16;
    const int kinds   = 3;

    // ---- decode: misses are read, hits are not ------------------------------------------------
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, /*max_slots =*/ 8, /*ring_count =*/ 8, kinds);
        fake_mailbox mail(layers);
        mock_queue queue;

        mail.publish(ledger);
        const l2_service first = ledger.service(0, {3, 5, 3}, 1, nothing_homed);
        CHECK(first.ok);
        CHECK(first.misses == 2 && first.hits == 0);
        CHECK(first.reads.size() == 2*kinds);
        queue.run(first);
        CHECK(queue.reads == 2*kinds);
        CHECK(ledger.slot_of(0, 3) >= 0 && ledger.slot_of(0, 5) >= 0);
        CHECK(ledger.slot_of(0, 3) != ledger.slot_of(0, 5));

        // the same generation asks again: both are ring hits and nothing is read
        const l2_service again = ledger.service(0, {3, 5}, 2, nothing_homed);
        CHECK(again.ok && again.hits == 2 && again.misses == 0 && again.reads.empty());

        // an expert that already has an arena home never reaches the ring
        const l2_service homed = ledger.service(1, {7},
            3, [](int, int expert) { return expert == 7; });
        CHECK(homed.ok && homed.hits == 0 && homed.misses == 0 && homed.reads.empty());
        CHECK(ledger.slot_of(1, 7) < 0);
        printf("PASS: decode reads a miss once and answers the repeat from the ring\n");
    }

    // ---- decode: a slot is not reused before the GPU is done with it ---------------------------
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, /*max_slots =*/ 2, /*ring_count =*/ 2, kinds);
        fake_mailbox mail(layers);
        mail.publish(ledger);

        const l2_service first = ledger.service(0, {0, 1}, 1, nothing_homed);
        CHECK(first.ok && first.misses == 2);
        const int slot0 = ledger.slot_of(0, 0);
        const int slot1 = ledger.slot_of(0, 1);

        // the GPU has not reported generation 1 yet, so the ring has nothing to give
        const l2_service blocked = ledger.service(1, {2}, 2, nothing_homed);
        CHECK(!blocked.ok);
        CHECK(blocked.reason.find("no reusable ring slot") != std::string::npos);

        // the layer completes, both slots become reusable
        mail.done[0] = 1;
        mail.publish(ledger);
        const l2_service after = ledger.service(1, {2}, 2, nothing_homed);
        CHECK(after.ok && after.misses == 1);
        CHECK(ledger.slot_of(1, 2) == slot0 || ledger.slot_of(1, 2) == slot1);
        CHECK(ledger.slot_of(0, 0) < 0 || ledger.slot_of(0, 1) < 0); // the evicted one is gone
        printf("PASS: a ring slot is reused only after the mailbox says the layer is done\n");
    }

    // ---- decode: a top-k larger than the free ring does not evict its own experts ---------------
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, /*max_slots =*/ 4, /*ring_count =*/ 4, kinds);
        fake_mailbox mail(layers);
        mail.publish(ledger);

        const l2_service service = ledger.service(0, {1, 2, 3, 4}, 1, nothing_homed);
        CHECK(service.ok && service.misses == 4);
        std::set<int> used;
        for (int expert : {1, 2, 3, 4}) {
            const int slot = ledger.slot_of(0, expert);
            CHECK(slot >= 0);
            CHECK(used.insert(slot).second);
        }
        // one more expert than the ring holds is a readable failure, not a silent eviction
        const l2_service overflow = ledger.service(0, {1, 2, 3, 4, 5}, 2, nothing_homed);
        CHECK(!overflow.ok);
        printf("PASS: decode pins what it already holds before it hands out a slot\n");
    }

    // ---- repartition between the two ring sizes -------------------------------------------------
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, /*max_slots =*/ 8, /*ring_count =*/ 8, kinds);
        fake_mailbox mail(layers);
        mail.publish(ledger);
        const l2_service filled = ledger.service(0, {1, 2}, 1, nothing_homed);
        CHECK(filled.ok);
        CHECK(ledger.borrowed_slots() == 0);

        // shrink to the decode ring: four slots are lent to the host tier and the ring is empty
        CHECK(ledger.set_ring_count(4));
        CHECK(ledger.ring_count() == 4 && ledger.borrowed_slots() == 4);
        CHECK(ledger.slot_of(0, 1) < 0 && ledger.slot_of(0, 2) < 0);

        // the small ring never hands out a borrowed slot
        mail.done[0] = 1;
        mail.publish(ledger);
        for (int expert = 0; expert < 4; ++expert) {
            const l2_service service = ledger.service(1, {expert}, 2, nothing_homed);
            CHECK(service.ok);
            CHECK(ledger.slot_of(1, expert) < 4);
        }

        // grow back to the prompt ring
        CHECK(ledger.set_ring_count(8));
        CHECK(ledger.borrowed_slots() == 0);
        CHECK(ledger.slot_of(1, 0) < 0);
        CHECK(!ledger.set_ring_count(9));
        CHECK(!ledger.set_ring_count(0));
        printf("PASS: a ring resize lends the tail slots and empties the ring\n");
    }

    // A demand can use the whole active ring. Unselected experts never get slots.
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, 8, 8, kinds);
        std::vector<int> ids;
        for (int i = 0; i < 512; ++i) { ids.push_back(i%8); }
        const auto first = ledger.service(0, ids, 1, nothing_homed);
        CHECK(first.ok && first.misses == 8 && first.reads.size() == 8*kinds);
        for (int e = 8; e < experts; ++e) { CHECK(ledger.slot_of(0, e) == -1); }
        const int slot = ledger.slot_of(0, 3);
        ledger.set_done(0, 1);
        const auto again = ledger.service(0, {3}, 2, nothing_homed);
        CHECK(again.ok && again.hits == 1 && again.reads.empty());
        CHECK(ledger.slot_of(0, 3) == slot && ledger.owns(slot, 0, 3));
        CHECK(!ledger.service(1, {8,9,10,11,12,13,14,15}, 1, nothing_homed).ok);
        ledger.set_done(0, 2);
        const auto next = ledger.service(1, {8,9,10,11,12,13,14,15}, 1, nothing_homed);
        CHECK(next.ok);
        CHECK(!ledger.owns(slot, 0, 3));
        printf("PASS: 512 rows deduplicate into demand-only slots, with retained leases\n");
    }
    // Generation zero after wrap is a real lease, not a free/prefetched owner.
    {
        l2_ledger ledger;
        ledger.reset(layers, experts, 1, 1, kinds);
        ledger.set_done(0, UINT32_MAX);
        CHECK(ledger.service(0, {1}, 0, nothing_homed).ok);
        CHECK(!ledger.service(1, {2}, 1, nothing_homed).ok);
        ledger.set_done(0, 0);
        const auto reused = ledger.service(1, {2}, 1, nothing_homed);
        CHECK(reused.ok && reused.evicted.size() == 1);
        CHECK(reused.evicted[0].layer == 0 && reused.evicted[0].expert == 1);
        printf("PASS: generation wrap preserves slot leases and eviction identity\n");
    }
    CHECK(storage_tests() == 0);
    CHECK(file_cap_tests() == 0);
    CHECK(class_ledger_tests() == 0);
    CHECK(relabel_install_tests() == 0);
    CHECK(early_tests() == 0);
    CHECK(unified_tests() == 0);
    CHECK(unified_install_tests() == 0);
    CHECK(unified_late_tests() == 0);
    printf("OK\n");
    return 0;
}
