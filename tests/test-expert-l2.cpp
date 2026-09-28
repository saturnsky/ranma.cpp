// ranma: the staging ring ledger of the expert cache SSD tier (expert-l2-ledger.h).
//
// Everything the ring decides is here: which expert gets which slot, when a slot may be reused,
// how repeated demands retain slots and what a ring resize does. A mock read queue stands in
// for the disk and a fake mailbox for the GPU, so this test needs neither.

#include "expert-l2-ledger.h"

#include <cstdio>
#include <map>
#include <set>
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

} // namespace

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
    printf("OK\n");
    return 0;
}
