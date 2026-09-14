#pragma once

// Host tier of the exclusive expert cache: the home of every routed expert that does not fit in
// the VRAM arena. One arena per size class and kind, with exactly the slot geometry of the VRAM
// arena (slot-major, the tensor's per-expert stride, the same zeroed tail), plus one device table
// per layer that maps an expert id to its host slot.
//
// The arena is ordinary private memory while the loader writes it, and is registered as
// coarse-grained mapped memory once, after the load, so the kernels can read it in place. That is
// the same two-step the HIP host buffer type uses (ggml_backend_cuda_finalize_host_buffer): the
// registration must see the final contents, and the device alias of a coarse buffer is not the
// host address.
//
// Addresses never change after allocate(); only slot contents and the tables do, and only while
// nothing computes.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-os.h"

#include <array>
#include <cstdint>
#include <vector>

namespace ggml_cuda_expert {

class host_arena {
public:
    explicit host_arena(const geometry & geo);
    ~host_arena();

    host_arena(const host_arena &) = delete;
    host_arena & operator=(const host_arena &) = delete;

    // Exactly once. capacities[class] is the number of experts of that class that live here, and
    // spare_slots more slots per class are kept free for the exchange rotation.
    bool allocate(const std::vector<int> & capacities, int spare_slots, int device);
    bool allocated() const { return !class_host_.empty(); }

    // Registers the arenas as coarse-grained mapped memory and resolves the device aliases. Called
    // once, after the loader has written every slice. Until then the kernels must not read here.
    bool map();
    bool mapped() const { return mapped_; }

    size_t host_bytes() const { return host_bytes_; }
    size_t device_bytes() const { return table_bytes_; }
    const std::vector<int> & capacities() const { return capacities_; }

    // Host address of one slice. Valid from allocate() on.
    void * slice(int cls, int kind, int slot) const;
    // Device alias of the arena base of a class and kind, or null before map().
    const void * device_data(int cls, int kind) const;
    // Device table of a layer: expert id -> host slot or -1.
    const int32_t * device_slots(int layer) const;

    // Copies the given tables to the device. The caller owns the host mirror.
    bool publish_tables(const std::vector<std::vector<int32_t>> & slots);

private:
    const geometry & geo_;
    int    device_      = -1;
    bool   mapped_      = false;
    size_t host_bytes_  = 0;
    size_t table_bytes_ = 0;
    int    spare_slots_ = 0;
    cudaStream_t copy_stream_ = nullptr;
    std::vector<int> capacities_;
    std::vector<std::array<expert_os::reservation, 3>> class_res_;    // [class][kind]
    std::vector<std::array<void *, 3>> class_host_;                   // [class][kind] host base
    std::vector<std::array<void *, 3>> class_device_;                 // [class][kind] device alias
    std::vector<std::array<bool,   3>> class_registered_;
    std::vector<int32_t *> layer_slots_;                              // [layer] device tables
};

} // namespace ggml_cuda_expert
