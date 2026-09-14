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

bool l1_arena::verify_current_assignment(std::string & reason) const {
    // The master of every expert stays in its original host tensor.
    const auto tx = stage(selected_, true);
    reason = tx.install.reason;
    return tx.valid;
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


l1_transaction l1_arena::stage(const expert_slot_table & selected, bool retain) const {
    l1_transaction tx;
    if (!allocated() || selected.size() != size_t(geo_.n_layers)) { return tx; }
    expert_slot_table hs(geo_.n_layers);
    // The host master of every expert is a slice of its mapped tensor; the transaction sees that as
    // one host slot per expert, in (layer, expert) order.
    expert_locations homes(geo_.n_layers, std::vector<expert_location>(geo_.n_experts));
    install_layout layout;
    layout.gpu = capacities_; layout.host.assign(capacities_.size(), 0);
    const std::vector<std::vector<int>> spares(capacities_.size());
    for (int l = 0; l < geo_.n_layers; ++l) {
        const int c = geo_.layer_class[l];
        if (c < 0) { continue; }
        for (int e = 0; e < geo_.n_experts; ++e) { homes[l][e] = {expert_storage::host, layout.host[c]++}; }
    }
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (geo_.layer_class[l] < 0) { continue; }
        for (int e = 0; e < geo_.n_experts; ++e) { hs[l].push_back(e); }
    }
    tx.install = plan_install(geo_, selected, hs, host_slots_, homes, capacities_, layout, layout, spares,
        {true, 0}, retain);
    tx.selected = selected;
    tx.valid = tx.install.valid;
    return tx;
}

// Where a slice of a move lives outside the VRAM arena: in the mapped tensor of its expert.
void * l1_arena::slice_address(int layer, int cls, int kind, expert_location at, int expert) const {
    GGML_UNUSED(cls); GGML_UNUSED(at);
    return host_address(layer, kind, expert);
}

bool l1_arena::execute(const install_transaction & tx) {
    if (!tx.valid || !allocated()) { return false; }
    ggml_cuda_set_device(device_);
    // Every slot the mover is about to rewrite must be invisible while it does.
    expert_slot_table dark(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
    if (!publish_tables(dark)) { return false; }
    for (size_t i = 0; i < tx.moves.size();) {
        const auto & move = tx.moves[i];
        const bool to_gpu   = move.to.storage   == expert_storage::vram;
        const bool from_gpu = move.from.storage == expert_storage::vram;
        for (int k = 0; k < geometry::n_kinds; ++k) {
            void * dst = to_gpu ? nullptr : slice_address(move.layer, move.cls, k, move.to, move.expert);
            const void * src = from_gpu ? nullptr : slice_address(move.layer, move.cls, k, move.from, move.expert);
            bool ok = true;
            if (from_gpu) {
                ok = read_gpu_slice(move.cls, k, move.from.slot, dst);
            } else if (to_gpu) {
                ok = write_gpu_slice(move.cls, k, move.to.slot, src, false);
            } else {
                if (dst == nullptr || src == nullptr || !sync_copies()) { return false; }
                memcpy(dst, src, geo_.class_bytes[move.cls][k]);
            }
            if (!ok) { return false; }
        }
        ++i;
    }
    return sync_copies();
}

bool l1_arena::publish(const l1_transaction & tx) {
    if (!tx.valid || !allocated() || !publish_tables(tx.install.gpu_slots)) { return false; }
    host_slots_ = tx.install.gpu_slots;
    selected_ = tx.selected; gpu_spares_ = tx.install.gpu_spares;
    return true;
}

bool l1_arena::install(const expert_slot_table & selected, bool retain, l1_install_stats & stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto tx = stage(selected, retain);
    if (!tx.valid || !execute(tx.install) || !publish(tx)) { return false; }
    stats.retained = tx.install.retained_gpu;
    stats.copied = tx.install.h2d_slices; stats.bytes = tx.install.h2d_bytes;
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
