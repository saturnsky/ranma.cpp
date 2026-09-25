// ranma: the staging ring ledger of the expert cache SSD tier (expert-l2-ledger.h), and for the
// class layout the storage classes (expert-storage.h), the per-class slot ledger
// (expert-l2-class-ledger.h), the host cut's per-layer file cap (expert-plan.h) and the relabel
// install transaction (expert-l2-relabel.h).
//
// Everything the ring decides is here: which expert gets which slot, when a slot may be reused,
// how repeated demands retain slots and what a ring resize does. A mock read queue stands in
// for the disk and a fake mailbox for the GPU, so this test needs neither.

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
    printf("OK\n");
    return 0;
}
