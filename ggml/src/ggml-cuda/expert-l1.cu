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

bool l1_arena::allocate(const std::vector<int> & capacities, int device, int spare_slots) {
    if (allocated() || capacities.size() != geo_.class_bytes.size() || geo_.n_layers <= 0 || spare_slots < 0) {
        return false;
    }
    device_      = device;
    spare_slots_ = spare_slots;
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    capacities_ = capacities;
    class_data_.assign(capacities.size(), {nullptr, nullptr, nullptr});
    for (size_t c = 0; c < capacities.size(); ++c) {
        if (capacities[c] == 0 && spare_slots == 0) { continue; }
        for (int k = 0; k < geometry::n_kinds; ++k) {
            const size_t bytes = size_t(capacities[c] + spare_slots)*geo_.class_bytes[c][k] +
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
    // With zero L1 slots all lookups miss; the host pointer keeps the existing dispatch active.
    // It is never indexed as L1 because the slot table contains only -1.
    out.data  = class_data_[cls][kind] ? class_data_[cls][kind] :
        (host_ && host_->mapped() ? host_->device_data(cls, kind) : nullptr);
    out.slots = layer_slots_[layer];
    if (addresses_) {
        out.host_addresses = addresses_(layer, kind);
    }
    if (host_ != nullptr && host_->mapped()) {
        // Owned host storage: an expert that misses the VRAM arena is read from the host arena,
        // not from the tensor, which has no bytes of its own. The kernels take the host slot table
        // first; host_addresses, when a tier attached one, answers the experts it does not hold.
        out.host_data  = host_->device_data(cls, kind);
        out.host_slots = host_->device_slots(layer);
    }
    return out;
}

void l1_arena::attach_host(host_arena * host, int spare_slots, bool host_master) {
    host_master_ = host_master;
    host_        = host;
    spare_slots_ = spare_slots;
}

bool l1_arena::assign_exclusive(const std::vector<std::vector<int32_t>> & selected) {
    if (!allocated() || host_ == nullptr || !host_->allocated() || selected.size() != (size_t) geo_.n_layers) {
        return false;
    }
    const size_t classes = capacities_.size();
    host_capacities_ = host_->capacities();
    arena_slots_.assign(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
    std::vector<std::vector<int32_t>> gpu(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
    std::vector<int> next_gpu(classes, 0), next_host(classes, 0);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        const int cls = geo_.layer_class[layer];
        if (cls < 0) {
            continue;
        }
        std::vector<bool> wanted(geo_.n_experts, false);
        for (int expert : selected[layer]) {
            if (expert < 0 || expert >= geo_.n_experts || wanted[expert]) {
                return false;
            }
            wanted[expert] = true;
        }
        for (int expert = 0; expert < geo_.n_experts; ++expert) {
            if (wanted[expert]) {
                if (next_gpu[cls] >= capacities_[cls]) {
                    return false;
                }
                gpu[layer][expert] = next_gpu[cls]++;
            } else {
                if (next_host[cls] >= host_capacities_[cls]) {
                    return false;
                }
                arena_slots_[layer][expert] = next_host[cls]++;
            }
        }
    }
    for (size_t cls = 0; cls < classes; ++cls) {
        if (next_gpu[cls] != capacities_[cls] || next_host[cls] != host_capacities_[cls]) {
            return false; // exclusive leaves no empty slot
        }
    }
    gpu_spares_.assign(classes, {});
    for (size_t cls = 0; cls < classes; ++cls) {
        for (int i = 0; i < spare_slots_; ++i) {
            gpu_spares_[cls].push_back(capacities_[cls] + i);
        }
    }
    homes_ = host_locations(arena_slots_);
    layout_.gpu = capacities_; layout_.host = host_capacities_;
    layout_.lent_begin.assign(classes, 0); layout_.lent_count.assign(classes, 0);
    for (size_t c = 0; c < classes; ++c) { layout_.gpu[c] += spare_slots_; layout_.host[c] += spare_slots_; }
    host_slots_ = std::move(gpu);
    selected_   = selected;
    return publish_tables(host_slots_) && host_->publish_tables(arena_slots_);
}

bool l1_arena::assign_tier(const expert_slot_table & gpu_slots, const expert_locations & homes,
        const install_layout & layout, const expert_slot_table & selected,
        const std::vector<std::vector<int>> & spares) {
    if (!allocated() || !host_ || gpu_slots.size() != size_t(geo_.n_layers) || homes.size() != gpu_slots.size()) { return false; }
    host_slots_ = gpu_slots; homes_ = homes; layout_ = layout;
    arena_slots_ = host_slot_table(homes);
    host_capacities_ = layout.host;
    selected_ = selected; gpu_spares_ = spares;
    return publish_tables(host_slots_) && host_->publish_tables(arena_slots_);
}

void * l1_arena::host_address(int layer, int kind, int expert) const {
    const int cls = geo_.layer_class[layer];
    if (!host_) {
        return static_cast<char *>(geo_.tensors[layer][kind]->data) + size_t(expert)*geo_.class_bytes[cls][kind];
    }
    const expert_location home = homes_[layer][expert];
    if (resolve_) { return resolve_(cls, kind, home); }
    return home.storage == expert_storage::host ? host_->slice(cls, kind, home.slot) : nullptr;
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

bool l1_arena::logical_io(int layer, int kind, void * data, size_t offset, size_t size, bool write) {
    if (!allocated() || host_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            kind < 0 || kind >= geometry::n_kinds) {
        return false;
    }
    const int cls = geo_.layer_class[layer];
    if (cls < 0) {
        return false;
    }
    const size_t stride = geo_.class_bytes[cls][kind];
    if (stride == 0 || offset > stride*size_t(geo_.n_experts) || size > stride*size_t(geo_.n_experts) - offset) {
        return false;
    }
    ggml_cuda_set_device(device_);
    char * bytes = static_cast<char *>(data);
    while (size != 0) {
        const int    expert = int(offset/stride);
        const size_t within = offset%stride;
        const size_t chunk  = std::min(size, stride - within);
        const int    slot   = host_slots_[layer][expert];
        char * home = static_cast<char *>(host_address(layer, kind, expert));
        if (slot < 0 && home == nullptr) { return false; }
        if (slot >= 0 && (!home || write)) {
            char * arena = gpu_slice(cls, kind, slot) + within;
            CUDA_CHECK(cudaMemcpyAsync(write ? arena : bytes, write ? bytes : arena, chunk,
                write ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, copy_stream_));
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        }
        if (home) { memcpy(write ? home + within : bytes, write ? bytes : home + within, chunk); }
        bytes  += chunk;
        offset += chunk;
        size   -= chunk;
    }
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
    if (host_) {
        return verify_assignment(host_slots_, homes_, geo_.layer_class, layout_, gpu_spares_, geo_.n_experts,
            {host_master_, spare_slots_}, bool(addresses_), reason);
    }
    // The unlimited inclusive path keeps its master in each original host tensor.
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
    expert_locations homes = homes_;
    install_layout layout = layout_;
    std::vector<std::vector<int>> spares = gpu_spares_;
    if (!host_) {
        homes.assign(geo_.n_layers, std::vector<expert_location>(geo_.n_experts));
        layout.gpu = capacities_; layout.host.assign(capacities_.size(), 0);
        layout.lent_begin.assign(capacities_.size(), 0); layout.lent_count.assign(capacities_.size(), 0);
        spares.assign(capacities_.size(), {});
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int c = geo_.layer_class[l];
            if (c < 0) { continue; }
            for (int e = 0; e < geo_.n_experts; ++e) { homes[l][e] = {expert_storage::host, layout.host[c]++}; }
        }
    }
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (geo_.layer_class[l] < 0) { continue; }
        for (int e = 0; e < geo_.n_experts; ++e) {
            if (host_master_ || std::find(selected[l].begin(), selected[l].end(), e) == selected[l].end()) { hs[l].push_back(e); }
        }
    }
    tx.install = plan_install(geo_, selected, hs, host_slots_, homes, capacities_, layout, layout, spares,
        {host_master_, spare_slots_}, false, retain);
    tx.selected = selected;
    tx.valid = tx.install.valid;
    return tx;
}

// Where a slice of a move lives outside the VRAM arena. A tier resolves host and lent slots; a
// two-tier arena reads its own host arena; without a host arena the tensor still owns the bytes.
void * l1_arena::slice_address(int layer, int cls, int kind, expert_location at, int expert) const {
    if (resolve_) { return resolve_(cls, kind, at); }
    if (host_)    { return host_->slice(cls, kind, at.slot); }
    return host_address(layer, kind, expert);
}

// Reads a batch of file-resident experts into ring slots and copies them to their destinations.
// Returns the number of moves consumed, or 0 on a refusal.
size_t l1_arena::move_from_file(const install_transaction & tx, size_t first) {
    if (!tier_.read || !tier_.address || tier_.slots <= 0 || !sync_copies()) { return 0; }
    std::vector<l2_read> reads;
    size_t count = 0;
    while (first + count < tx.moves.size() && count < size_t(tier_.slots) &&
            tx.moves[first + count].from.storage == expert_storage::file) {
        const auto & item = tx.moves[first + count];
        for (int k = 0; k < geometry::n_kinds; ++k) { reads.push_back({item.layer, k, item.expert, int(count)}); }
        ++count;
    }
    std::string reason;
    if (!tier_.read(reads, reason)) {
        GGML_ABORT("expert cache: install read failed: %s", reason.c_str());
    }
    for (const l2_read & read : reads) {
        const auto & item = tx.moves[first + read.slot];
        const void * src = tier_.address(read);
        if (item.to.storage == expert_storage::vram) {
            if (!write_gpu_slice(item.cls, read.kind, item.to.slot, src, false)) { return 0; }
        } else {
            void * dst = slice_address(item.layer, item.cls, read.kind, item.to, item.expert);
            if (dst == nullptr) { return 0; }
            memcpy(dst, src, geo_.class_bytes[item.cls][read.kind]);
        }
    }
    // A ring slot may be refilled by the next batch, so every copy out of it completes first.
    return sync_copies() ? count : 0;
}

bool l1_arena::execute(const install_transaction & tx) {
    if (!tx.valid || !allocated()) { return false; }
    ggml_cuda_set_device(device_);
    if (!host_) {
        expert_slot_table dark(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
        if (!publish_tables(dark)) { return false; }
    }
    for (size_t i = 0; i < tx.moves.size();) {
        const auto & move = tx.moves[i];
        if (move.from.storage == expert_storage::file) {
            const size_t done = move_from_file(tx, i);
            if (done == 0) { return false; }
            i += done;
            continue;
        }
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
    host_slots_ = tx.install.gpu_slots; homes_ = tx.install.host;
    selected_ = tx.selected; gpu_spares_ = tx.install.gpu_spares;
    if (host_) { arena_slots_ = host_slot_table(homes_); return host_->publish_tables(arena_slots_); }
    return true;
}

bool l1_arena::install(const expert_slot_table & selected, bool retain, l1_install_stats & stats) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto tx = stage(selected, retain);
    if (!tx.valid || !execute(tx.install) || !publish(tx)) { return false; }
    stats.retained = tx.install.retained_gpu;
    stats.copied = tx.install.h2d_slices; stats.bytes = tx.install.h2d_bytes;
    stats.d2h_bytes = tx.install.d2h_bytes;
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
