#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_cuda_expert {

enum class expert_storage { file, vram, host, lent };

struct expert_location {
    expert_storage storage = expert_storage::file;
    int slot = -1;
    bool resident() const { return storage == expert_storage::host || storage == expert_storage::lent; }
    bool operator==(const expert_location & other) const { return storage == other.storage && slot == other.slot; }
    bool operator!=(const expert_location & other) const { return !(*this == other); }
};

using expert_locations = std::vector<std::vector<expert_location>>;
using expert_slot_table = std::vector<std::vector<int32_t>>;

struct install_layout {
    std::vector<int> gpu, host; // Physical slot counts, including spares.
    std::vector<int> lent_begin, lent_count;

    bool contains(int cls, expert_location at) const {
        if (cls < 0 || size_t(cls) >= host.size()) { return false; }
        if (at.storage == expert_storage::host) { return at.slot >= 0 && at.slot < host[cls]; }
        return at.storage == expert_storage::lent && size_t(cls) < lent_count.size() &&
            size_t(cls) < lent_begin.size() && at.slot >= lent_begin[cls] && at.slot - lent_begin[cls] < lent_count[cls];
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
