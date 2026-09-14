#include "expert-host.cuh"

#include "expert-l1.cuh"

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
    for (int32_t * table : layer_slots_) {
        if (table) {
            (void) cudaFree(table);
        }
    }
    if (copy_stream_) {
        (void) cudaStreamDestroy(copy_stream_);
    }
}

bool host_arena::allocate(const std::vector<int> & capacities, int spare_slots, int device) {
    if (allocated() || capacities.size() != geo_.class_bytes.size() || geo_.n_layers <= 0 || spare_slots < 0) {
        return false;
    }
    device_      = device;
    spare_slots_ = spare_slots;
    capacities_  = capacities;
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));

    class_res_.assign(capacities.size(), {});
    class_host_.assign(capacities.size(), {nullptr, nullptr, nullptr});
    class_device_.assign(capacities.size(), {nullptr, nullptr, nullptr});
    class_registered_.assign(capacities.size(), {false, false, false});
    for (size_t cls = 0; cls < capacities.size(); ++cls) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const size_t bytes = size_t(capacities[cls] + spare_slots)*geo_.class_bytes[cls][kind] +
                arena_tail_bytes(geo_, (int) cls, kind);
            std::optional<expert_os::reservation> res = expert_os::reserve(bytes);
            if (!res || !expert_os::commit(*res, 0, bytes)) {
                if (res) {
                    expert_os::release(*res);
                }
                return false;
            }
            // The tail and the spare slots must read as zeros until something owns them: MMQ reads
            // past the last row of the last slot, see arena_tail_bytes in expert-l1.cuh.
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
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

bool host_arena::map() {
    if (!allocated() || mapped_) {
        return allocated() && mapped_;
    }
#if defined(GGML_USE_HIP)
    ggml_cuda_set_device(device_);
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
    return true;
#else
    GGML_LOG_ERROR("expert cache: the host arena needs HIP mapped host memory\n");
    return false;
#endif
}

void * host_arena::slice(int cls, int kind, int slot) const {
    if (cls < 0 || (size_t) cls >= class_host_.size() || kind < 0 || kind >= geometry::n_kinds || slot < 0) {
        return nullptr;
    }
    if (slot >= capacities_[cls] + spare_slots_) { return nullptr; }
    return static_cast<char *>(class_host_[cls][kind]) + size_t(slot)*geo_.class_bytes[cls][kind];
}

const void * host_arena::device_data(int cls, int kind) const {
    if (!mapped_ || cls < 0 || (size_t) cls >= class_device_.size() || kind < 0 || kind >= geometry::n_kinds) {
        return nullptr;
    }
    return class_device_[cls][kind];
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
    return true;
}

} // namespace ggml_cuda_expert
