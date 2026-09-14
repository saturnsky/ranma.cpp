// Replay ordered moves against symbolic payloads, independently of the planner's bookkeeping.
#include "expert-plan.h"

#include <cstdio>
#include <map>
#include <tuple>

using namespace ggml_cuda_expert;
#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static int replay(bool inclusive) {
    geometry geo;
    geo.n_layers = 5; geo.n_experts = 8;
    geo.layer_class = {-1, 0, 1, 0, 1}; geo.class_layers = {2, 2};
    geo.class_bytes = {{10, 10, 10}, {20, 20, 20}};
    const std::vector<int> caps{4, 3};
    const mover_capability mover{inclusive, inclusive ? 0 : 2};
    install_layout layout;
    layout.gpu = {caps[0] + mover.spare_slots, caps[1] + mover.spare_slots};
    layout.host = inclusive ? std::vector<int>{16, 16} : std::vector<int>{14, 15};
    expert_slot_table gpu(5, std::vector<int32_t>(8, -1));
    expert_locations host(5, std::vector<expert_location>(8));
    std::vector<std::vector<int>> spares(2);
    using key = std::tuple<int, int, int>;
    std::map<key, int> payload;
    auto at = [](int c, expert_location p) { return key{c, int(p.storage), p.slot}; };
    for (int c = 0; c < 2; ++c) {
        int g = 0, h = 0;
        for (int l = 0; l < 5; ++l) if (geo.layer_class[l] == c) for (int e = 0; e < 8; ++e) {
            const bool in_g = g < caps[c];
            if (in_g) { gpu[l][e] = g; payload[at(c, {expert_storage::vram, g++})] = l*8 + e; }
            if ((inclusive || !in_g) && h < layout.host[c] - (inclusive ? 0 : mover.spare_slots)) {
                host[l][e] = {expert_storage::host, h++}; payload[at(c, host[l][e])] = l*8 + e;
            }
        }
        for (int s = 0; s < mover.spare_slots; ++s) { spares[c].push_back(caps[c] + s); }
    }
    size_t d2h = 0;
    for (int round = 0; round < 40; ++round) {
        install_layout next = layout;
        expert_slot_table gs(5), hs(5);
        for (int c = 0; c < 2; ++c) {
            int selected_h = 0;
            const int nh = 16 - (inclusive ? 0 : caps[c]);
            for (int i = 0; i < 16; ++i) {
                const int id = (i + round*5 + c*3)%16;
                const int l = 1 + c + 2*(id/8), e = id%8;
                if (i < caps[c]) { gs[l].push_back(e); }
                if ((inclusive || i >= caps[c]) && selected_h++ < nh) { hs[l].push_back(e); }
            }
        }
        const auto tx = plan_install(geo, gs, hs, gpu, host, caps, layout, next, spares, mover);
        if (!tx.valid) { fprintf(stderr, "round %d inclusive %d: %s\n", round, inclusive, tx.reason.c_str()); }
        CHECK(tx.valid);
        for (const auto & m : tx.moves) {
            const int expected = m.layer*8 + m.expert;
            CHECK(payload[at(m.cls, m.from)] == expected);
            if (m.from.storage == expert_storage::vram) { CHECK(!inclusive); ++d2h; }
            payload[at(m.cls, m.to)] = expected;
        }
        for (int l = 0; l < 5; ++l) for (int e = 0; e < 8; ++e) {
            const int c = geo.layer_class[l], expected = l*8 + e;
            if (tx.gpu_slots[l][e] >= 0) { CHECK(payload[at(c, {expert_storage::vram, tx.gpu_slots[l][e]})] == expected); }
            if (tx.host[l][e].resident()) { CHECK(payload[at(c, tx.host[l][e])] == expected); }
        }
        for (int l = 0; l < 5; ++l) {
            for (int e : gs[l]) { CHECK(tx.gpu_slots[l][e] >= 0); }
            for (int e : hs[l]) { CHECK(tx.host[l][e].resident()); }
        }
        std::string reason;
        CHECK(verify_assignment(tx.gpu_slots, tx.host, geo.layer_class, next, tx.gpu_spares, 8, mover, reason));
        // Corrupt one invariant at a time, independent of the move replay above.
        auto bad_g = tx.gpu_slots;
        int first_l = -1, first_e = -1;
        for (int l = 1; l < 5 && first_l < 0; ++l) for (int e = 0; e < 8; ++e) if (bad_g[l][e] >= 0) { first_l = l; first_e = e; break; }
        const int other_e = (first_e + 1)%8;
        bad_g[first_l][other_e] = bad_g[first_l][first_e];
        CHECK(!verify_assignment(bad_g, tx.host, geo.layer_class, next, tx.gpu_spares, 8, mover, reason));
        auto bad_h = tx.host;
        if (inclusive) { bad_h[first_l][first_e] = {}; }
        else { bad_h[first_l][first_e] = {expert_storage::host, 0}; }
        CHECK(!verify_assignment(tx.gpu_slots, bad_h, geo.layer_class, next, tx.gpu_spares, 8, mover, reason));
        auto bad_spares = tx.gpu_spares;
        bad_spares[geo.layer_class[first_l]].push_back(tx.gpu_slots[first_l][first_e]);
        CHECK(!verify_assignment(tx.gpu_slots, tx.host, geo.layer_class, next, bad_spares, 8, mover, reason));
        bad_h = tx.host;
        for (int l = 1; l < 5; ++l) {
            bool changed = false;
            for (int e = 0; e < 8; ++e) if (bad_h[l][e].resident()) {
                bad_h[l][(e + 1)%8] = bad_h[l][e]; changed = true; break;
            }
            if (changed) { break; }
        }
        CHECK(!verify_assignment(tx.gpu_slots, bad_h, geo.layer_class, next, tx.gpu_spares, 8, mover, reason));
        bad_g = tx.gpu_slots; bad_h = tx.host;
        bad_g[first_l][first_e] = -1; bad_h[first_l][first_e] = {};
        CHECK(!verify_assignment(bad_g, bad_h, geo.layer_class, next, tx.gpu_spares, 8, mover, reason));
        gpu = tx.gpu_slots; host = tx.host; spares = tx.gpu_spares; layout = next;
    }
    CHECK(inclusive ? d2h == 0 : d2h > 0);
    printf("PASS: %s two-tier, 40 ordered payload replays and rejected invariant violations\n", inclusive ? "inclusive" : "exclusive");
    return 0;
}

int main() {
    CHECK(replay(true) == 0);
    CHECK(replay(false) == 0);
    return 0;
}
