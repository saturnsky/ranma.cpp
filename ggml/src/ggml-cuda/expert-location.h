#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_cuda_expert {

enum class expert_storage { none, vram, host };

struct expert_location {
    expert_storage storage = expert_storage::none;
    int slot = -1;
    bool resident() const { return storage == expert_storage::host; }
    bool operator==(const expert_location & other) const { return storage == other.storage && slot == other.slot; }
    bool operator!=(const expert_location & other) const { return !(*this == other); }
};

using expert_locations = std::vector<std::vector<expert_location>>;
using expert_slot_table = std::vector<std::vector<int32_t>>;

struct install_layout {
    std::vector<int> gpu, host; // Physical slot counts, including spares.

    bool contains(int cls, expert_location at) const {
        if (cls < 0 || size_t(cls) >= host.size()) { return false; }
        return at.storage == expert_storage::host && at.slot >= 0 && at.slot < host[cls];
    }
};

inline expert_locations host_locations(const expert_slot_table & slots) {
    expert_locations out(slots.size());
    for (size_t l = 0; l < slots.size(); ++l) {
        out[l].resize(slots[l].size());
        for (size_t e = 0; e < slots[l].size(); ++e) {
            if (slots[l][e] >= 0) { out[l][e] = {expert_storage::host, slots[l][e]}; }
        }
    }
    return out;
}

inline expert_slot_table host_slot_table(const expert_locations & locations) {
    expert_slot_table out(locations.size());
    for (size_t l = 0; l < locations.size(); ++l) {
        out[l].assign(locations[l].size(), -1);
        for (size_t e = 0; e < locations[l].size(); ++e) {
            if (locations[l][e].storage == expert_storage::host) { out[l][e] = locations[l][e].slot; }
        }
    }
    return out;
}

} // namespace ggml_cuda_expert
