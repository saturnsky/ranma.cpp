#include "expert-l1.cuh"

#include <chrono>
#include <cstring>

namespace ggml_cuda_expert {

// See the comment on arena_tail_bytes in expert-l1.cuh for why the tail exists and why it is exactly
// the host buffer type's quantized row padding.
static constexpr size_t arena_tail_floor = 512;

static int class_reference_layer(const geometry & geo, int cls) {
    for (int layer = 0; layer < geo.n_layers; ++layer) {
        if (geo.layer_class[layer] == cls) {
            return layer;
        }
    }
    return -1;
}

size_t arena_tail_bytes(const geometry & geo, int cls, int kind) {
    const int layer = class_reference_layer(geo, cls);
    if (layer < 0 || kind < 0 || kind >= geometry::n_kinds || geo.tensors[layer][kind] == nullptr) {
        return arena_tail_floor;
    }
    const ggml_tensor * t = geo.tensors[layer][kind];
    const int64_t rest = t->ne[0] % MATRIX_ROW_PADDING;
    const size_t pad = rest == 0 ? 0 : ggml_row_size(t->type, MATRIX_ROW_PADDING - rest);
    return std::max(pad, arena_tail_floor);
}

size_t arena_tail_total(const geometry & geo) {
    size_t total = 0;
    for (int cls = 0; cls < (int) geo.class_bytes.size(); ++cls) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            total += arena_tail_bytes(geo, cls, kind);
        }
    }
    return total;
}

l1_arena::l1_arena(const geometry & geo) : geo_(geo) {
}

l1_arena::~l1_arena() {
    if (device_ < 0) {
        return;
    }
    ggml_cuda_set_device(device_);
    (void) cudaDeviceSynchronize();
    for (auto & cls : class_data_) {
        for (void * p : cls) {
            if (p) {
                (void) cudaFree(p);
            }
        }
    }
    for (int32_t * p : layer_slots_) {
        if (p) {
            (void) cudaFree(p);
        }
    }
    if (copy_stream_) {
        (void) cudaStreamDestroy(copy_stream_);
    }
}

bool l1_arena::allocate(const std::vector<int> & capacities, int device) {
    if (allocated() || capacities.size() != geo_.class_bytes.size() || geo_.n_layers <= 0) {
        return false;
    }
    device_ = device;
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    capacities_ = capacities;
    class_data_.assign(capacities.size(), {nullptr, nullptr, nullptr});
    for (size_t c = 0; c < capacities.size(); ++c) {
        if (capacities[c] == 0) { continue; }
        for (int k = 0; k < geometry::n_kinds; ++k) {
            const size_t bytes = size_t(capacities[c])*geo_.class_bytes[c][k] +
                arena_tail_bytes(geo_, (int) c, k);
            if (cudaMalloc(&class_data_[c][k], bytes) != cudaSuccess) {
                (void) cudaGetLastError();
                class_data_[c][k] = nullptr;
                return false;
            }
            CUDA_CHECK(cudaMemsetAsync(class_data_[c][k], 0, bytes, copy_stream_));
            allocated_bytes_ += bytes;
        }
    }
    layer_slots_.assign(geo_.n_layers, nullptr);
    host_slots_.assign(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
    selected_.assign(geo_.n_layers, {});
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (geo_.layer_class[l] < 0) {
            continue;
        }
        const size_t bytes = size_t(geo_.n_experts)*sizeof(int32_t);
        if (cudaMalloc((void **) &layer_slots_[l], bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            layer_slots_[l] = nullptr;
            return false;
        }
        CUDA_CHECK(cudaMemsetAsync(layer_slots_[l], 0xff, bytes, copy_stream_));
        allocated_bytes_ += bytes;
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

ggml_cuda_expert_lookup l1_arena::lookup(int layer, int kind) const noexcept {
    ggml_cuda_expert_lookup out;
    if (layer < 0 || layer >= geo_.n_layers || kind < 0 || kind >= geometry::n_kinds || layer_slots_.empty()) {
        return out;
    }
    const int cls = geo_.layer_class[layer];
    if (cls < 0 || layer_slots_[layer] == nullptr) {
        return out;
    }
    // With zero L1 slots all lookups miss; the slot table contains only -1 and every expert is
    // read from the mapped host tensor exactly as before.
    out.data  = class_data_[cls][kind];
    out.slots = layer_slots_[layer];
    return out;
}

// The host master of an expert is its slice of the mapped routed-expert tensor.
void * l1_arena::host_address(int layer, int kind, int expert) const {
    const int cls = geo_.layer_class[layer];
    return static_cast<char *>(geo_.tensors[layer][kind]->data) + size_t(expert)*geo_.class_bytes[cls][kind];
}

bool l1_arena::write_gpu_slice(int cls, int kind, int slot, const void * src, bool src_is_device) {
    if (!allocated() || cls < 0 || (size_t) cls >= capacities_.size() || slot < 0) {
        return false;
    }
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpyAsync(gpu_slice(cls, kind, slot), src, geo_.class_bytes[cls][kind],
        src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, copy_stream_));
    return true;
}

bool l1_arena::read_gpu_slice(int cls, int kind, int slot, void * dst) {
    if (!allocated() || cls < 0 || (size_t) cls >= capacities_.size() || slot < 0 || dst == nullptr) {
        return false;
    }
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpyAsync(dst, gpu_slice(cls, kind, slot), geo_.class_bytes[cls][kind],
        cudaMemcpyDeviceToHost, copy_stream_));
    return true;
}

bool l1_arena::sync_copies() {
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

bool l1_arena::read_slice(int layer, int kind, int slot, void * dst) const {
    if (!allocated() || layer < 0 || layer >= geo_.n_layers || kind < 0 || kind >= geometry::n_kinds || slot < 0) {
        return false;
    }
    const int cls = geo_.layer_class[layer];
    if (cls < 0) {
        return false;
    }
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpyAsync(dst, gpu_slice(cls, kind, slot), geo_.class_bytes[cls][kind],
        cudaMemcpyDeviceToHost, copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}

bool l1_arena::publish_tables(const std::vector<std::vector<int32_t>> & slots) {
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (layer_slots_[l] == nullptr) {
            continue;
        }
        CUDA_CHECK(cudaMemcpyAsync(layer_slots_[l], slots[l].data(), size_t(geo_.n_experts)*sizeof(int32_t),
            cudaMemcpyHostToDevice, copy_stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    return true;
}


// Fills the arena with `selected` and publishes the slot tables. The tables are blanked first, so
// that a kernel captured in a graph never reads a slot while the mover is rewriting it.
bool l1_arena::install(const std::vector<std::vector<int32_t>> & selected, l1_install_stats & stats) {
    if (!allocated() || selected.size() != size_t(geo_.n_layers)) { return false; }
    const auto t0 = std::chrono::steady_clock::now();
    ggml_cuda_set_device(device_);
    std::vector<std::vector<int32_t>> slots(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
    if (!publish_tables(slots)) { return false; }
    std::vector<int> next(capacities_.size(), 0);
    for (int l = 0; l < geo_.n_layers; ++l) {
        const int cls = geo_.layer_class[l];
        if (cls < 0) { continue; }
        for (int32_t e : selected[l]) {
            if (e < 0 || e >= geo_.n_experts || next[cls] >= capacities_[cls]) { return false; }
            const int slot = next[cls]++;
            for (int k = 0; k < geometry::n_kinds; ++k) {
                if (!write_gpu_slice(cls, k, slot, host_address(l, k, e), /*src_is_device =*/ false)) { return false; }
                stats.bytes += geo_.class_bytes[cls][k];
            }
            slots[l][e] = slot;
            ++stats.copied;
        }
    }
    if (!sync_copies() || !publish_tables(slots)) { return false; }
    host_slots_ = slots;
    selected_   = selected;
    stats.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool l1_arena::verify_slice(int layer, int kind, int expert) const {
    if (!allocated() || layer < 0 || layer >= geo_.n_layers || expert < 0 || expert >= geo_.n_experts) {
        return false;
    }
    const int cls  = geo_.layer_class[layer];
    const int slot = cls >= 0 ? host_slots_[layer][expert] : -1;
    if (slot < 0) {
        return false;
    }
    const size_t stride = geo_.class_bytes[cls][kind];
    std::vector<uint8_t> back(stride);
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpyAsync(back.data(), static_cast<const char *>(class_data_[cls][kind]) + size_t(slot)*stride,
        stride, cudaMemcpyDeviceToHost, copy_stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    const ggml_tensor * t = geo_.tensors[layer][kind];
    return memcmp(back.data(), static_cast<const char *>(t->data) + size_t(expert)*stride, stride) == 0;
}

long l1_arena::verify_resident(size_t max_slices) const {
    long checked = 0;
    for (int l = 0; l < geo_.n_layers; ++l) {
        for (int e : selected_[l]) {
            for (int k = 0; k < geometry::n_kinds; ++k) {
                if (!verify_slice(l, k, e)) {
                    return -1;
                }
            }
            if (++checked >= long(max_slices) && max_slices != 0) {
                return checked;
            }
        }
    }
    return checked;
}

} // namespace ggml_cuda_expert
