#include "expert-host.cuh"

#include "expert-l1.cuh"

#include <chrono>
#include <cstring>

namespace ggml_cuda_expert {

host_arena::host_arena(const geometry & geo) : geo_(geo) {
}

host_arena::~host_arena() {
    if (device_ >= 0) {
        ggml_cuda_set_device(device_);
        (void) cudaDeviceSynchronize();
    }
#if defined(GGML_USE_HIP)
    for (size_t cls = 0; cls < class_host_.size(); ++cls) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            if (class_registered_[cls][kind] && hipHostUnregister(class_host_[cls][kind]) != hipSuccess) {
                (void) hipGetLastError();
            }
        }
    }
#endif
    for (auto & cls : class_res_) {
        for (expert_os::reservation & res : cls) {
            expert_os::release(res);
        }
    }
    for (auto & cls : chunks_) {
        for (auto & list : cls) {
            for (chunk & c : list) {
                drop_chunk(c);
            }
        }
    }
    for (int32_t * table : layer_slots_) {
        if (table) {
            (void) cudaFree(table);
        }
    }
    if (addr_dev_) {
        (void) cudaFree(addr_dev_);
    }
    if (copy_stream_) {
        (void) cudaStreamDestroy(copy_stream_);
    }
}

bool host_arena::allocate(const std::vector<int> & capacities, int spare_slots, int device,
        const class_storage * storage) {
    if (allocated() || capacities.size() != geo_.class_bytes.size() || geo_.n_layers <= 0 || spare_slots < 0) {
        return false;
    }
    if (storage != nullptr) {
        if (!storage->valid() || storage->resident_slots.size() != capacities.size()) {
            return false;
        }
        for (size_t cls = 0; cls < capacities.size(); ++cls) {
            if (storage->resident_slots[cls] != capacities[cls] + spare_slots) {
                return false;
            }
        }
        class_layout_ = true;
        storage_      = *storage;
    }
    if ((addresses_on_ || chunk_bytes_ != 0) && class_layout_) {
        return false; // the tier's own address table serves the class layout
    }
    if (chunk_bytes_ != 0 && !addresses_on_) {
        return false; // chunk aliases are not contiguous: the kernels need the address table
    }
    device_      = device;
    spare_slots_ = spare_slots;
    capacities_  = capacities;
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    if (chunk_bytes_ != 0) {
        chunk_slots_.assign(capacities.size(), 1);
        chunks_.assign(capacities.size(), {});
        for (size_t cls = 0; cls < capacities.size(); ++cls) {
            chunk_slots_[cls] = host_chunk_slots(geo_, (int) cls, chunk_bytes_);
            const int n = host_chunks_for(capacities[cls] + spare_slots, chunk_slots_[cls]);
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                for (int i = 0; i < n; ++i) {
                    if (!add_chunk((int) cls, kind)) {
                        return false;
                    }
                }
            }
        }
    }

    const size_t arenas = chunked() ? 0 : class_layout_ ? (size_t) storage_.storages() : capacities.size();
    class_res_.assign(arenas, {});
    class_host_.assign(arenas, {nullptr, nullptr, nullptr});
    class_device_.assign(arenas, {nullptr, nullptr, nullptr});
    class_registered_.assign(arenas, {false, false, false});
    for (size_t cls = 0; cls < arenas; ++cls) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const size_t bytes = class_layout_ ?
                size_t(storage_.slots((int) cls))*storage_.pitch[cls][kind] + storage_.tail[cls][kind] :
                size_t(capacities[cls] + spare_slots)*geo_.class_bytes[cls][kind] + arena_tail_bytes(geo_, (int) cls, kind);
            std::optional<expert_os::reservation> res = expert_os::reserve(bytes);
            if (!res || !expert_os::commit(*res, 0, bytes)) {
                if (res) {
                    expert_os::release(*res);
                }
                return false;
            }
            // The tail and the spare slots must read as zeros until something owns them: MMQ reads
            // past the last row of the last slot, see arena_tail_bytes in expert-l1.cuh. With the
            // class layout every slot also reads as zeros after its payload.
            memset(res->base, 0, bytes);
            class_res_[cls][kind]  = *res;
            class_host_[cls][kind] = res->base;
            host_bytes_ += bytes;
        }
    }

    layer_slots_.assign(geo_.n_layers, nullptr);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        if (geo_.layer_class[layer] < 0) {
            continue;
        }
        const size_t bytes = size_t(geo_.n_experts)*sizeof(int32_t);
        if (cudaMalloc((void **) &layer_slots_[layer], bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            layer_slots_[layer] = nullptr;
            return false;
        }
        CUDA_CHECK(cudaMemsetAsync(layer_slots_[layer], 0xff, bytes, copy_stream_));
        table_bytes_ += bytes;
    }
    if (addresses_on_) {
        const size_t n = size_t(geo_.n_layers)*geometry::n_kinds*size_t(geo_.n_experts);
        if (cudaMalloc((void **) &addr_dev_, n*sizeof(uint64_t)) != cudaSuccess) {
            (void) cudaGetLastError();
            addr_dev_ = nullptr;
            return false;
        }
        CUDA_CHECK(cudaMemsetAsync(addr_dev_, 0, n*sizeof(uint64_t), copy_stream_));
        addr_host_.assign(n, 0);
        table_bytes_ += n*sizeof(uint64_t);
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

bool host_arena::add_chunk(int cls, int kind) {
    const size_t bytes = size_t(chunk_slots_[(size_t) cls])*geo_.class_bytes[(size_t) cls][(size_t) kind] +
        arena_tail_bytes(geo_, cls, kind);
    std::optional<expert_os::reservation> res = expert_os::reserve(bytes);
    if (!res || !expert_os::commit(*res, 0, bytes)) {
        if (res) {
            expert_os::release(*res);
        }
        return false;
    }
    // unowned slots and the tail read as zeros (MMQ reads past the last row, arena_tail_bytes)
    memset(res->base, 0, bytes);
    chunk c;
    c.res  = *res;
    c.host = res->base;
    if (mapped_ && !register_chunk(c)) {
        drop_chunk(c);
        return false;
    }
    chunks_[(size_t) cls][(size_t) kind].push_back(c);
    host_bytes_ += bytes;
    return true;
}

bool host_arena::register_chunk(chunk & c) {
#if defined(GGML_USE_HIP)
    const unsigned int flags = hipHostRegisterMapped | hipExtHostRegisterCoarseGrained;
    const hipError_t err = hipHostRegister(c.host, c.res.bytes, flags);
    if (err != hipSuccess) {
        (void) hipGetLastError();
        GGML_LOG_ERROR("expert cache: failed to register a %.2f MiB host chunk: %s\n", c.res.bytes/1024.0/1024.0,
            hipGetErrorString(err));
        return false;
    }
    c.registered = true;
    if (hipHostGetDevicePointer(&c.device, c.host, 0) != hipSuccess) {
        (void) hipGetLastError();
        GGML_LOG_ERROR("expert cache: failed to map a host chunk\n");
        return false;
    }
    return true;
#else
    GGML_UNUSED(c);
    GGML_LOG_ERROR("expert cache: the host arena needs HIP mapped host memory\n");
    return false;
#endif
}

void host_arena::drop_chunk(chunk & c) {
#if defined(GGML_USE_HIP)
    if (c.registered && hipHostUnregister(c.host) != hipSuccess) {
        (void) hipGetLastError();
    }
#endif
    c.registered = false;
    c.device     = nullptr;
    c.host       = nullptr;
    expert_os::release(c.res);
}

size_t host_arena::chunk_count() const {
    size_t n = 0;
    for (const auto & cls : chunks_) {
        for (const auto & list : cls) {
            n += list.size();
        }
    }
    return n;
}

size_t host_arena::slack_bytes() const {
    size_t bytes = 0;
    for (size_t cls = 0; cls < chunks_.size(); ++cls) {
        const size_t used = size_t(capacities_[cls] + spare_slots_);
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const size_t slots = chunks_[cls][(size_t) kind].size()*size_t(chunk_slots_[cls]);
            bytes += slots > used ? (slots - used)*geo_.class_bytes[cls][(size_t) kind] : 0;
        }
    }
    return bytes;
}

uint64_t host_arena::device_address(int cls, int kind, int slot) const {
    if (!mapped_ || cls < 0 || (size_t) cls >= capacities_.size() || kind < 0 || kind >= geometry::n_kinds || slot < 0) {
        return 0;
    }
    const size_t stride = geo_.class_bytes[(size_t) cls][(size_t) kind];
    if (chunked()) {
        const int per = chunk_slots_[(size_t) cls];
        const auto & list = chunks_[(size_t) cls][(size_t) kind];
        if ((size_t) (slot/per) >= list.size() || list[(size_t) (slot/per)].device == nullptr) {
            return 0;
        }
        return (uint64_t) (uintptr_t) list[(size_t) (slot/per)].device + uint64_t(slot % per)*stride;
    }
    if (class_layout_ || class_device_[(size_t) cls][(size_t) kind] == nullptr) {
        return 0;
    }
    return (uint64_t) (uintptr_t) class_device_[(size_t) cls][(size_t) kind] + uint64_t(slot)*stride;
}

const uint64_t * host_arena::device_addresses(int layer, int kind) const {
    if (addr_dev_ == nullptr || !mapped_ || layer < 0 || layer >= geo_.n_layers || kind < 0 || kind >= geometry::n_kinds) {
        return nullptr;
    }
    return addr_dev_ + (size_t(layer)*geometry::n_kinds + size_t(kind))*size_t(geo_.n_experts);
}

// Rebuilds the address table from the last published slot tables and copies it to the device.
// Nothing to do before map() (map() calls it again) or before the first publish.
bool host_arena::upload_addresses() {
    if (addr_dev_ == nullptr || !mapped_ || slots_mirror_.empty()) {
        return true;
    }
    if (slots_mirror_.size() != (size_t) geo_.n_layers) {
        return false;
    }
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        const int cls = geo_.layer_class[layer];
        const bool rows = cls >= 0 && slots_mirror_[layer].size() == (size_t) geo_.n_experts;
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            uint64_t * row = addr_host_.data() + (size_t(layer)*geometry::n_kinds + size_t(kind))*size_t(geo_.n_experts);
            for (int e = 0; e < geo_.n_experts; ++e) {
                const int32_t slot = rows ? slots_mirror_[layer][e] : -1;
                row[e] = slot >= 0 ? device_address(cls, kind, slot) : 0;
                if (slot >= 0 && row[e] == 0) {
                    return false;
                }
            }
        }
    }
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpyAsync(addr_dev_, addr_host_.data(), addr_host_.size()*sizeof(uint64_t), cudaMemcpyHostToDevice,
        copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

bool host_arena::set_capacity(int cls, int capacity, resize_stats & stats, std::string & why) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!chunked() || cls < 0 || (size_t) cls >= capacities_.size() || capacity < 0) {
        why = "no chunked host arena, or a wrong class or capacity";
        return false;
    }
    const int per    = chunk_slots_[(size_t) cls];
    const int before = capacities_[(size_t) cls] + spare_slots_;
    const int after  = capacity + spare_slots_;
    const int want   = host_chunks_for(after, per);
    ggml_cuda_set_device(device_);
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
        auto & list = chunks_[(size_t) cls][(size_t) kind];
        while ((int) list.size() < want) {
            if (!add_chunk(cls, kind)) {
                why = "a host chunk of class " + std::to_string(cls) + " could not be allocated or registered";
                return false;
            }
            stats.chunks_added++;
            stats.bytes_added += list.back().res.bytes;
        }
        while ((int) list.size() > want) {
            stats.chunks_released++;
            stats.bytes_released += list.back().res.bytes;
            host_bytes_ -= list.back().res.bytes;
            drop_chunk(list.back());
            list.pop_back();
        }
        // the vacated slots that stay committed read as zeros again
        const size_t stride = geo_.class_bytes[(size_t) cls][(size_t) kind];
        for (int slot = after; slot < before && slot < want*per; ++slot) {
            memset(static_cast<char *>(list[(size_t) (slot/per)].host) + size_t(slot % per)*stride, 0, stride);
        }
    }
    capacities_[(size_t) cls] = capacity;
    stats.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool host_arena::map() {
    if (!allocated() || mapped_) {
        return allocated() && mapped_;
    }
#if defined(GGML_USE_HIP)
    ggml_cuda_set_device(device_);
    for (auto & cls : chunks_) {
        for (auto & list : cls) {
            for (chunk & c : list) {
                if (!register_chunk(c)) {
                    return false;
                }
            }
        }
    }
    for (size_t cls = 0; cls < class_host_.size(); ++cls) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const size_t bytes = class_res_[cls][kind].bytes;
            const unsigned int flags = hipHostRegisterMapped | hipExtHostRegisterCoarseGrained;
            const hipError_t err = hipHostRegister(class_host_[cls][kind], bytes, flags);
            if (err != hipSuccess) {
                (void) hipGetLastError();
                GGML_LOG_ERROR("expert cache: failed to register %.2f MiB of host arena: %s\n",
                    bytes/1024.0/1024.0, hipGetErrorString(err));
                return false;
            }
            class_registered_[cls][kind] = true;
            if (hipHostGetDevicePointer(&class_device_[cls][kind], class_host_[cls][kind], 0) != hipSuccess) {
                (void) hipGetLastError();
                GGML_LOG_ERROR("expert cache: failed to map the host arena of class %d kind %d\n", (int) cls, kind);
                return false;
            }
        }
    }
    // the device must see every CPU write that came before the registration
    CUDA_CHECK(cudaDeviceSynchronize());
    mapped_ = true;
    if (!upload_addresses()) {
        GGML_LOG_ERROR("expert cache: the host address table could not be built\n");
        return false;
    }
    return true;
#else
    GGML_LOG_ERROR("expert cache: the host arena needs HIP mapped host memory\n");
    return false;
#endif
}

void * host_arena::slice(int cls, int kind, int slot) const {
    if (cls < 0 || (size_t) cls >= (chunked() ? capacities_.size() : class_host_.size()) || kind < 0 ||
            kind >= geometry::n_kinds || slot < 0) {
        return nullptr;
    }
    if (slot >= capacities_[cls] + spare_slots_) { return nullptr; }
    if (chunked()) {
        const int per = chunk_slots_[(size_t) cls];
        return static_cast<char *>(chunks_[(size_t) cls][(size_t) kind][(size_t) (slot/per)].host) +
            size_t(slot % per)*geo_.class_bytes[cls][kind];
    }
    if (class_layout_) {
        const int s = storage_.storage_of[cls];
        return static_cast<char *>(class_host_[s][kind]) + size_t(storage_.resident_slot(cls, slot))*storage_.pitch[s][kind];
    }
    return static_cast<char *>(class_host_[cls][kind]) + size_t(slot)*geo_.class_bytes[cls][kind];
}

const void * host_arena::device_data(int cls, int kind) const {
    if (!mapped_ || cls < 0 || (size_t) cls >= capacities_.size() || kind < 0 || kind >= geometry::n_kinds) {
        return nullptr;
    }
    if (class_layout_) {
        const int s = storage_.storage_of[cls];
        return static_cast<const char *>(class_device_[s][kind]) + size_t(storage_.resident_base[cls])*storage_.pitch[s][kind];
    }
    if (chunked()) {
        return chunks_[(size_t) cls][(size_t) kind].front().device;
    }
    return class_device_[cls][kind];
}

void * host_arena::storage_base(int storage, int kind, bool device) const {
    if (!class_layout_ || storage < 0 || storage >= storage_.storages() || kind < 0 || kind >= geometry::n_kinds) {
        return nullptr;
    }
    return device ? class_device_[storage][kind] : class_host_[storage][kind];
}

const int32_t * host_arena::device_slots(int layer) const {
    if (layer < 0 || (size_t) layer >= layer_slots_.size()) {
        return nullptr;
    }
    return layer_slots_[layer];
}

bool host_arena::publish_tables(const std::vector<std::vector<int32_t>> & slots) {
    if (!allocated() || slots.size() != (size_t) geo_.n_layers) {
        return false;
    }
    ggml_cuda_set_device(device_);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        if (layer_slots_[layer] == nullptr) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(layer_slots_[layer], slots[layer].data(),
            size_t(geo_.n_experts)*sizeof(int32_t), cudaMemcpyHostToDevice, copy_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    if (addresses_on_) {
        slots_mirror_ = slots;
        return upload_addresses();
    }
    return true;
}

} // namespace ggml_cuda_expert
