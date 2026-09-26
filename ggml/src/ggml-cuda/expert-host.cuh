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
//
// With a finite host tier in the class layout (expert-storage.h) there is one arena per storage
// class and kind instead: the resident slots of its size classes and the SSD tier's staging ring,
// all at the storage pitch. The payload of a slot starts at the sector shift of its tensor, which
// only the tier knows, so the tier serves every such resident through its address table and slice()
// returns the start of the slot.
//
// Without a finite tier (expert-host-layout.h): enable_addresses() makes publish_tables() also write
// the device address of every host resident slice, which the kernels take before the slot table; and
// enable_chunks() splits each (class, kind) arena into separately registered chunks whose count
// set_capacity() changes, so that a class's host capacity can follow a redraw of the VRAM split.
// Chunk 0 of every (class, kind) is never released, so device_data() stays valid.

#include "common.cuh"
#include "expert-geometry.h"
#include "expert-host-layout.h"
#include "expert-os.h"
#include "expert-storage.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

class host_arena {
public:
    explicit host_arena(const geometry & geo);
    ~host_arena();

    host_arena(const host_arena &) = delete;
    host_arena & operator=(const host_arena &) = delete;

    // Before allocate(), without a class layout only (see the header comment). enable_chunks needs
    // the address table, and allocate() refuses it otherwise.
    void enable_addresses() { addresses_on_ = true; }
    void enable_chunks(size_t chunk_bytes) { chunk_bytes_ = chunk_bytes; }
    bool addresses_enabled() const { return addresses_on_; }
    bool chunked() const { return !chunk_slots_.empty(); }
    int  chunk_slots(int cls) const { return chunked() ? chunk_slots_[(size_t) cls] : 0; }
    size_t chunk_count() const;
    // Committed bytes of chunk slots past each class's slot count (the chunk tails), all kinds.
    size_t slack_bytes() const;
    // The address table of a layer and kind (expert id -> device address of its host slice, 0 when
    // the expert has no host slot), or null without the table or before map().
    const uint64_t * device_addresses(int layer, int kind) const;
    // Device address of one slot, or 0 before map().
    uint64_t device_address(int cls, int kind, int slot) const;

    // Chunked arenas, while nothing computes: the class keeps `capacity` experts (plus the spares).
    // Growing adds zeroed chunks (registered at once after map()); shrinking releases the chunks past
    // the new slot count and zeroes the vacated slots of the top one. No slot at or above the new
    // count may hold a resident (the caller compacts first).
    struct resize_stats {
        size_t chunks_added = 0, chunks_released = 0, bytes_added = 0, bytes_released = 0;
        double ms = 0.0;
    };
    bool set_capacity(int cls, int capacity, resize_stats & stats, std::string & why);

    // Exactly once. capacities[class] is the number of experts of that class that live here, and
    // spare_slots more slots per class are kept free for the exchange rotation. With `storage` the
    // arenas follow the class layout; its resident ranges must be the capacities plus the spares.
    bool allocate(const std::vector<int> & capacities, int spare_slots, int device,
                  const class_storage * storage = nullptr);
    bool class_layout() const { return class_layout_; }
    bool allocated() const { return !class_host_.empty() || !chunks_.empty(); }

    // Registers the arenas as coarse-grained mapped memory and resolves the device aliases. Called
    // once, after the loader has written every slice. Until then the kernels must not read here.
    bool map();
    bool mapped() const { return mapped_; }

    size_t host_bytes() const { return host_bytes_; }
    size_t device_bytes() const { return table_bytes_; }
    const std::vector<int> & capacities() const { return capacities_; }

    // Host address of one slice (with the class layout the start of its slot). Valid from
    // allocate() on.
    void * slice(int cls, int kind, int slot) const;
    // Device alias of the arena base of a class and kind (with the class layout the first resident
    // slot of the class), or null before map().
    const void * device_data(int cls, int kind) const;
    // Class layout: slot 0 of the arena of a storage class and kind; the device alias is null
    // before map().
    void * storage_base(int storage, int kind, bool device) const;
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
    bool          class_layout_ = false;
    class_storage storage_;
    // [class][kind]; with the class layout [storage class][kind]
    std::vector<std::array<expert_os::reservation, 3>> class_res_;
    std::vector<std::array<void *, 3>> class_host_;                   // [class][kind] host base
    std::vector<std::array<void *, 3>> class_device_;                 // [class][kind] device alias
    std::vector<std::array<bool,   3>> class_registered_;
    std::vector<int32_t *> layer_slots_;                              // [layer] device tables

    // address table and chunks (expert-host-layout.h)
    struct chunk {
        expert_os::reservation res;
        void * host       = nullptr;
        void * device     = nullptr;
        bool   registered = false;
    };
    bool add_chunk(int cls, int kind);
    void drop_chunk(chunk & c);
    bool register_chunk(chunk & c);
    bool upload_addresses();
    bool   addresses_on_ = false;
    size_t chunk_bytes_  = 0;
    std::vector<int> chunk_slots_;                                    // [class] slots per chunk
    std::vector<std::array<std::vector<chunk>, 3>> chunks_;           // [class][kind]
    uint64_t * addr_dev_ = nullptr;                                   // [layer][kind][expert]
    std::vector<uint64_t> addr_host_;
    std::vector<std::vector<int32_t>> slots_mirror_;                  // the last published slot tables
};

} // namespace ggml_cuda_expert
