#include "expert-l2.cuh"
#include "expert-l1.cuh"

#if defined(GGML_USE_HIP)

#include "ggml-expert.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64)
#define RANMA_CPU_PAUSE() __builtin_ia32_pause()
#else
#define RANMA_CPU_PAUSE() std::this_thread::yield()
#endif

namespace ggml_cuda_expert {

struct alignas(64) l2_mailbox {
    uint32_t published, ready, done, generation, invalid;
    uint32_t rows, wait_generation;
    uint32_t stage_ticks;     // staged service: wall ticks the later kinds waited, cumulative
    uint64_t wait_ticks, steady_ticks;
    uint32_t need;            // set by the publish kernel when a routed expert needs the worker
    uint32_t kind_ready[3];   // staged service: the generation whose slices of that kind are in place
};
static_assert(sizeof(l2_mailbox) == 64, "the mailbox layout is part of the host budget");

// Batched service, one per layer. `launches` follows the published generation (publish kernel); the
// worker writes the tables of a batched generation, then `gen`, and releases each batch of a kind
// through `ready`; the GPU reports each finished batch launch of a kind through `done`. A ready or
// done word is (generation << 32) | batches, so a word of an earlier generation never passes.
struct alignas(64) l2_batch_mail {
    uint32_t launches;     // batch launches per kind of the published generation
    uint32_t gen;          // the generation the tables belong to
    uint32_t count;        // batches of that generation
    uint32_t stage_ticks;  // wall ticks the batch launches waited, cumulative
    uint32_t pad0[12];
    uint64_t ready[3];
    uint64_t pad1[5];
    uint64_t done[3];
    uint64_t pad2[5];
};
static_assert(sizeof(l2_batch_mail) == 192, "the batch mailbox layout is part of the host budget");

static __device__ __forceinline__ uint64_t l2_batch_word(uint32_t seq, uint32_t batches) {
    return (uint64_t(seq) << 32) | batches;
}

// `serve` marks the experts only the worker can place: file residents and ring occupants. It follows
// a plan, not a generation. When no routed id needs the worker, this kernel answers its own wait, so
// a fully resident layer never reaches the CPU.
// The bitmap and the flags are built in shared memory and then stored: PCIe has no atomic OR, and on
// some hosts an atomicOr on the mapped mailbox is lost.
static __global__ void l2_publish_kernel(l2_mailbox * m, uint32_t * demand, const uint32_t * serve,
        const int32_t * ids, int rows, int used, int stride, int experts, l2_batch_mail * bm, int launches) {
    extern __shared__ uint32_t demand_shared[];
    __shared__ uint32_t need, invalid;
    const int words = (experts + 31)/32;
    for (int i = threadIdx.x; i < words; i += blockDim.x) { demand_shared[i] = 0; }
    if (threadIdx.x == 0) {
        need = 0;
        invalid = 0;
        m->rows = uint32_t(rows);
        __hip_atomic_store(&m->ready, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < int64_t(rows)*used; i += blockDim.x) {
        const int e = ids[(i/used)*stride + i%used];
        if ((unsigned) e < (unsigned) experts) {
            atomicOr(demand_shared + e/32, 1u << (e%32));
            if (serve[e/32] & (1u << (e%32))) { need = 1; }
        } else {
            invalid = 1;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < words; i += blockDim.x) { demand[i] = demand_shared[i]; }
    __syncthreads();
    if (threadIdx.x == 0) {
        m->need = need;
        m->invalid = invalid;
        bm->launches = uint32_t(launches);
        const uint32_t seq = ++m->generation;
        if (!need) {
            __hip_atomic_store(&m->ready, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        __threadfence_system();
        __hip_atomic_store(&m->published, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static __global__ void l2_wait_kernel(l2_mailbox * m) {
    const uint64_t begin = clock64(), steady_begin = wall_clock64();
    while (__hip_atomic_load(&m->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != m->generation) {
        __builtin_amdgcn_s_sleep(1);
    }
    m->wait_ticks = clock64() - begin;
    m->steady_ticks = wall_clock64() - steady_begin;
    __threadfence_system();
    __hip_atomic_store(&m->wait_generation, m->generation, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Staged service: waits until either the whole generation or the slices of `kind` are ready; a
// generation the publish kernel answered itself passes at once. The first kind records the
// per-generation sample exactly like l2_wait_kernel, the later kinds only add their wall time to
// stage_ticks.
static __global__ void l2_wait_kind_kernel(l2_mailbox * m, int kind, int record) {
    const uint64_t begin = clock64(), steady_begin = wall_clock64();
    const uint32_t seq = m->generation;
    while (__hip_atomic_load(&m->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq &&
           __hip_atomic_load(&m->kind_ready[kind], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq) {
        __builtin_amdgcn_s_sleep(1);
    }
    if (record) {
        m->wait_ticks = clock64() - begin;
        m->steady_ticks = wall_clock64() - steady_begin;
        __threadfence_system();
        __hip_atomic_store(&m->wait_generation, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    } else {
        m->stage_ticks += uint32_t(wall_clock64() - steady_begin);
        __threadfence_system();
    }
}

// Batched service: copies the batch tables of the generation to device memory for the MMQ launches,
// after the route wait, so the worker has written them if the generation is batched. Any other
// generation (one the publish kernel answered, one with nothing to batch) gets the tables of a single
// launch: every expert in the first launch, every list empty.
static __global__ void l2_batch_prep_kernel(const l2_mailbox * m, const l2_batch_mail * bm, const int32_t * host,
        int32_t * dev, int experts, int words) {
    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid = __hip_atomic_load(&bm->gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == m->generation;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < words; i += blockDim.x) {
        dev[i] = valid ? host[i] : (i < experts ? 0 : -1);
    }
}

// Batch `batch` (1-based) of `kind`: waits until its reads are in place, or the whole generation is.
static __global__ void l2_batch_wait_kernel(l2_mailbox * m, l2_batch_mail * bm, int kind, int batch) {
    const uint64_t steady_begin = wall_clock64();
    const uint32_t seq = m->generation;
    while (__hip_atomic_load(&m->ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != seq) {
        const uint64_t word = __hip_atomic_load(&bm->ready[kind], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        if (uint32_t(word >> 32) == seq && uint32_t(word) >= uint32_t(batch)) { break; }
        __builtin_amdgcn_s_sleep(1);
    }
    bm->stage_ticks += uint32_t(wall_clock64() - steady_begin);
    __threadfence_system();
}

// After batch launch `batch` of `kind`: the GPU no longer reads its slots.
static __global__ void l2_batch_done_kernel(const l2_mailbox * m, l2_batch_mail * bm, int kind, int batch) {
    if (threadIdx.x == 0) {
        __threadfence_system();
        __hip_atomic_store(&bm->done[kind], l2_batch_word(m->generation, uint32_t(batch)), __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static __global__ void l2_done_kernel(l2_mailbox * m) {
    if (threadIdx.x == 0) {
        __hip_atomic_store(&m->done, m->generation, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static bool coherent_alloc(void ** host, void ** device, size_t bytes) {
    if (hipHostMalloc(host, bytes, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    memset(*host, 0, bytes);
    if (hipHostGetDevicePointer(device, *host, 0) != hipSuccess) {
        (void) hipGetLastError();
        return false;
    }
    return true;
}

// ---- construction ----------------------------------------------------------------------------------

l2_tier::l2_tier(const geometry & geo, const l2_config & cfg) : geo_(geo), cfg_(cfg) {
    backing_.assign((size_t) geo_.n_layers, {backing(), backing(), backing()});
    size_ring();
    verify_ = cfg_.verify;
}

l2_tier::~l2_tier() {
    stop_worker();
    report("release");
    for (expert_os::file_handle & file : files_) {
        expert_os::close_file(file);
    }
    if (device_ >= 0) {
        ggml_cuda_set_device(device_);
        (void) cudaDeviceSynchronize();
    }
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
        if (ring_registered_[kind] && ring_host_base_[kind] != nullptr) {
            (void) hipHostUnregister(ring_host_base_[kind]);
        }
        expert_os::release(ring_res_[kind]);
    }
    if (mail_host_)   { (void) hipHostFree(mail_host_); }
    if (maps_host_)   { (void) hipHostFree(maps_host_); }
    if (demand_host_) { (void) hipHostFree(demand_host_); }
    if (serve_host_)  { (void) hipHostFree(serve_host_); }
    if (batch_mail_host_) { (void) hipHostFree(batch_mail_host_); }
    if (batch_tab_host_)  { (void) hipHostFree(batch_tab_host_); }
    if (batch_dev_)       { (void) cudaFree(batch_dev_); }
    expert_os::aligned_free(bounce_aligned_);
}

// The pitch of a ring slot: the whole slice plus room for the sector shift of the file offset, so
// an unbuffered read of the covering sector range fits and the payload starts at slot + shift. The
// pitch is the largest over the size classes, because any class may land in any slot.
void l2_tier::size_ring() {
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
        size_t pitch = 0;
        for (size_t cls = 0; cls < geo_.class_bytes.size(); ++cls) {
            const size_t tail = arena_tail_bytes(geo_, int(cls), kind);
            const size_t slice = geo_.class_bytes[cls][kind];
            if (tail > SIZE_MAX - 2*expert_os::io_alignment || slice > SIZE_MAX - tail - 2*expert_os::io_alignment) { return; }
            pitch = std::max(pitch, expert_os::align_up_io(slice + expert_os::io_alignment - 1 + tail));
        }
        ring_pitch_[kind] = pitch;
    }
    const size_t stride = ring_stride();
    if (stride == 0 || geo_.n_experts <= 0) {
        return;
    }
    const auto plan = plan_l2_ring(geo_.n_experts, cfg_.experts_used, cfg_.prefill_rows, cfg_.decode_rows,
        stride, cfg_.prefill_ring_bytes, cfg_.decode_ring_bytes, cfg_.phase_rings);
    prompt_slots_ = plan.prompt_slots; decode_slots_ = plan.decode_slots;
    size_note_ = plan.note;
    if (!size_note_.empty()) {
        size_note_ += ": prefill=" + std::to_string(prompt_slots_) + " slots (floor " + std::to_string(plan.prompt_floor) +
            "), decode=" + std::to_string(decode_slots_) + " (floor " + std::to_string(plan.decode_floor) + ")";
    }
    sized_ = plan.valid;
}

size_t l2_tier::metadata_bytes() const {
    const size_t mail   = (size_t) geo_.n_layers*sizeof(l2_mailbox);
    const size_t maps   = (size_t) geo_.n_layers*geometry::n_kinds*(size_t) geo_.n_experts*sizeof(uint64_t);
    const size_t demand = bitmap_bytes();   // the demand table and the serve table have one layout
    const size_t bounce = std::max({ring_pitch_[0], ring_pitch_[1], ring_pitch_[2]});
    const size_t batch = (size_t) geo_.n_layers*sizeof(l2_batch_mail);
    const size_t tables = (size_t) geo_.n_layers*batch_tab_words()*sizeof(int32_t);
    return expert_os::align_up_io(mail) + expert_os::align_up_io(maps) + 2*expert_os::align_up_io(demand) + bounce +
        expert_os::align_up_io(batch) + expert_os::align_up_io(tables);
}

void l2_tier::set_storage(const class_storage & storage) {
    GGML_ASSERT(class_layout() && !mailbox_active_);
    storage_ = storage;
    bool ok = storage_.valid() && storage_.storage_of.size() == geo_.class_bytes.size() &&
        storage_.floor.size() == storage_.members.size();
    for (int s = 0; ok && s < storage_.storages(); ++s) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            // a read of the covering sectors must fit, and every slot must stay sector aligned
            ok = ok && storage_.pitch[s][kind] % expert_os::io_alignment == 0;
            for (int c : storage_.members[s]) {
                ok = ok && storage_.pitch[s][kind] >= geo_.class_bytes[c][kind] + expert_os::io_alignment - 1 +
                    arena_tail_bytes(geo_, c, kind);
            }
        }
        ok = ok && storage_.ring_slots[s] >= storage_.floor[s];
    }
    sized_ = ok;
}

int l2_tier::install_read_slots() const {
    if (!class_layout()) {
        return std::min({cfg_.queue_depth, ring_count(), decode_slots_});
    }
    int slots = cfg_.queue_depth;
    for (int s = 0; s < storage_.storages(); ++s) {
        if (storage_.ring_slots[s] > 0) { slots = std::min(slots, storage_.ring_slots[s]); }
    }
    return std::max(slots, 1);
}

bool l2_tier::allocate(int device) {
    if (!sized_) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not size its ring\n");
        return false;
    }
    device_ = device;
    ggml_cuda_set_device(device_);
    CUDA_CHECK(hipDeviceGetAttribute(&clock_khz_, hipDeviceAttributeClockRate, device_));
    CUDA_CHECK(hipDeviceGetAttribute(&steady_khz_, hipDeviceAttributeWallClockRate, device_));
    GGML_ASSERT(clock_khz_ > 0 && steady_khz_ > 0);
    pending_.resize(geo_.n_layers); wait_seen_.assign(geo_.n_layers, 0);
    staged_layer_.assign(geo_.n_layers, 0); stage_seen_.assign(geo_.n_layers, 0); batch_seen_.assign(geo_.n_layers, 0);
    route_launches_.assign(geo_.n_layers, 0);
    seen_.assign(geo_.n_layers, 0);
    if (early_layer_.size() != (size_t) geo_.n_layers) { early_layer_.assign(geo_.n_layers, 0); }
    if (cfg_.early) {
        early_mark_.assign((size_t) geo_.n_layers*(size_t) geo_.n_experts, 0);
        early_list_.assign(geo_.n_layers, {});
    }
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (geo_.layer_class[l] >= 0) { first_layer_ = l; break; }
    }

    // The class layout keeps its ring in the host arenas, which the controller allocates.
    for (int kind = 0; kind < geometry::n_kinds && !class_layout(); ++kind) {
        const size_t bytes = (size_t) prompt_slots_*ring_pitch_[kind];
        std::optional<expert_os::reservation> res = expert_os::reserve(bytes);
        if (!res || !expert_os::commit(*res, 0, bytes)) {
            if (res) {
                expert_os::release(*res);
            }
            GGML_LOG_ERROR("expert cache: the SSD tier could not reserve %.2f MiB for its ring\n",
                bytes/1024.0/1024.0);
            return false;
        }
        memset(res->base, 0, bytes);
        ring_res_[kind]       = *res;
        ring_host_base_[kind] = res->base;
        host_bytes_ += bytes;
    }

    if (!coherent_alloc(&mail_host_, &mail_device_, (size_t) geo_.n_layers*sizeof(l2_mailbox))) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its mailboxes\n");
        return false;
    }
    void * maps = nullptr, * maps_device = nullptr;
    if (!coherent_alloc(&maps, &maps_device,
            (size_t) geo_.n_layers*geometry::n_kinds*(size_t) geo_.n_experts*sizeof(uint64_t))) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its address table\n");
        return false;
    }
    maps_host_   = static_cast<uint64_t *>(maps);
    maps_device_ = static_cast<uint64_t *>(maps_device);
    void * demand = nullptr, * demand_device = nullptr;
    if (!coherent_alloc(&demand, &demand_device, bitmap_bytes())) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its demand table\n");
        return false;
    }
    demand_host_   = static_cast<uint32_t *>(demand);
    demand_device_ = static_cast<uint32_t *>(demand_device);
    void * serve = nullptr, * serve_device = nullptr;
    if (!coherent_alloc(&serve, &serve_device, bitmap_bytes())) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its serve table\n");
        return false;
    }
    serve_host_   = static_cast<uint32_t *>(serve);
    serve_device_ = static_cast<uint32_t *>(serve_device);
    if (!coherent_alloc(&batch_mail_host_, &batch_mail_device_, (size_t) geo_.n_layers*sizeof(l2_batch_mail))) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its batch mailboxes\n");
        return false;
    }
    void * tab = nullptr, * tab_device = nullptr;
    if (!coherent_alloc(&tab, &tab_device, (size_t) geo_.n_layers*batch_tab_words()*sizeof(int32_t))) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its batch tables\n");
        return false;
    }
    batch_tab_host_   = static_cast<int32_t *>(tab);
    batch_tab_device_ = static_cast<int32_t *>(tab_device);
    if (batched() && cudaMalloc((void **) &batch_dev_, (size_t) geo_.n_layers*batch_tab_words()*sizeof(int32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        batch_dev_ = nullptr;
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its device batch tables\n");
        return false;
    }
    // Graphs capture the mailbox nodes, so they must exist from the first plan on. With an all-clear
    // serve table they cost two tiny self-answering kernels for each routed layer.
    mailbox_active_ = true;
    host_bytes_ += metadata_bytes();

    queue_.reset(new expert_os::read_queue(cfg_.queue_depth));
    if (!queue_->valid()) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not create its read queue\n");
        return false;
    }
    size_t bounce = 0;
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
        bounce = std::max(bounce, ring_pitch_[kind]);
    }
    bounce_aligned_ = expert_os::aligned_alloc(bounce, expert_os::io_alignment);
    if (bounce_aligned_ == nullptr) {
        GGML_LOG_ERROR("expert cache: the SSD tier could not allocate its bounce slot\n");
        return false;
    }

    prompt_ring_ = true;
    if (class_layout()) {
        std::vector<int> layer_storage((size_t) geo_.n_layers, -1), slots;
        for (int l = 0; l < geo_.n_layers; ++l) {
            if (geo_.layer_class[l] >= 0) { layer_storage[(size_t) l] = storage_.storage_of[geo_.layer_class[l]]; }
        }
        for (int s = 0; s < storage_.storages(); ++s) { slots.push_back(storage_.slots(s)); }
        cledger_.reset(geo_.n_layers, geo_.n_experts, layer_storage, slots, geometry::n_kinds);
        // Host slot i of a size class starts in its resident range; relabel installs remap it.
        home_slot_.assign(geo_.class_bytes.size(), {});
        for (int c = 0; c < (int) geo_.class_bytes.size(); ++c) {
            for (int i = 0; i < storage_.resident_slots[c]; ++i) {
                home_slot_[c].push_back(storage_.resident_slot(c, i));
                GGML_ASSERT(cledger_.set_resident(storage_.storage_of[c], home_slot_[c][i], -1, -1));
            }
        }
        for (int s = 0; s < storage_.storages(); ++s) {
            for (int i = 0; i < storage_.ring_slots[s]; ++i) { GGML_ASSERT(cledger_.make_ring(s, storage_.ring_base[s] + i)); }
            cledger_.set_floor(s, storage_.floor[s]);
            std::string members;
            for (int c : storage_.members[s]) { members += (members.empty() ? "" : ",") + std::to_string(c); }
            GGML_LOG_INFO("expert cache: SSD tier storage class %d (size classes %s): pitch %zu/%zu/%zu KiB, "
                          "%d resident slots, ring %d slots / %zu MiB (floor %d)\n",
                s, members.c_str(), storage_.pitch[s][0]/1024, storage_.pitch[s][1]/1024, storage_.pitch[s][2]/1024,
                storage_.ring_base[s], storage_.ring_slots[s],
                size_t(storage_.ring_slots[s])*storage_.stride(s)/(1024*1024), storage_.floor[s]);
        }
        char factor[48];
        if (storage_.factor > 0.0) { snprintf(factor, sizeof(factor), "ring factor %g", storage_.factor); }
        else { snprintf(factor, sizeof(factor), "explicit ring size"); }
        GGML_LOG_INFO("expert cache: SSD tier class layout: ring %d slots (%zu MiB) in the host arenas, least recently "
                      "used per storage class, %s, installs relabel slots and keep the rings, metadata and bounce %zu KiB, queue depth %d\n",
            cledger_.ring_total(), storage_.ring_bytes()/(1024*1024), factor,
            metadata_bytes()/1024, cfg_.queue_depth);
    } else {
        ledger_.reset(geo_.n_layers, geo_.n_experts, prompt_slots_, prompt_slots_, geometry::n_kinds);
        GGML_LOG_INFO("expert cache: SSD tier ring %d slots (%zu MiB, prompt) / %d slots (%zu MiB, decode), "
                      "metadata and bounce %zu KiB, queue depth %d\n",
            prompt_slots_, ring_bytes(prompt_slots_)/(1024*1024),
            decode_slots_, ring_bytes(decode_slots_)/(1024*1024),
            metadata_bytes()/1024, cfg_.queue_depth);
    }
    if (!size_note_.empty() && !class_layout()) {
        GGML_LOG_INFO("expert cache: %s\n", size_note_.c_str());
    }
    if (batched()) {
        std::string caps;
        for (int s = 0; s < storage_.storages(); ++s) {
            caps += (s ? "," : "") + std::to_string(std::min(geo_.n_experts, l2_batch_capacity(storage_.ring_slots[s])));
        }
        GGML_LOG_INFO("expert cache: SSD tier batched service for ubatches of more than %llu rows: the experts that need no "
                      "read first, then the file residents in batches of at most %s experts (per storage class) through two "
                      "buffers; prompt fills %s\n", (unsigned long long) cfg_.decode_rows, caps.c_str(),
            l2_prompt_fill_name(cfg_.prompt_fill));
    }
    if (staged()) {
        GGML_LOG_INFO("expert cache: SSD tier staged service for ubatches of at least %lld rows: up, gate and "
                      "down are released one kind at a time%s\n", (long long) cfg_.staged_min_rows,
            cfg_.staged_drain ? ", each kind read to completion before the next is issued" : "");
    }
    return true;
}

bool l2_tier::map() {
    if (mapped_) {
        return true;
    }
    ggml_cuda_set_device(device_);
    // The class layout's ring is part of the host arenas, which are registered with the residents.
    for (int kind = 0; kind < geometry::n_kinds && !class_layout(); ++kind) {
        const size_t bytes = ring_res_[kind].bytes;
        // Mapped, not coarse grained: the worker refills a slot while the GPU runs, and coarse
        // grained memory is only coherent at kernel boundaries. Phase C measures whether this
        // device would in fact keep a stale line.
        const hipError_t err = hipHostRegister(ring_host_base_[kind], bytes, hipHostRegisterMapped);
        if (err != hipSuccess) {
            (void) hipGetLastError();
            GGML_LOG_ERROR("expert cache: failed to register %.2f MiB of ring: %s\n",
                bytes/1024.0/1024.0, hipGetErrorString(err));
            return false;
        }
        ring_registered_[kind] = true;
        if (hipHostGetDevicePointer(&ring_device_base_[kind], ring_host_base_[kind], 0) != hipSuccess) {
            (void) hipGetLastError();
            GGML_LOG_ERROR("expert cache: failed to map the ring of kind %d\n", kind);
            return false;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    mapped_ = true;
    return true;
}

// ---- the loader ------------------------------------------------------------------------------------

void l2_tier::set_backing(int layer, int kind, int file_index, const char * path, uint64_t offset) {
    if (layer < 0 || layer >= geo_.n_layers || kind < 0 || kind >= geometry::n_kinds) {
        return;
    }
    // Files are numbered by the tier, by path: the loader's index counts the files of one model,
    // and the models of a joint cache each start at 0.
    GGML_UNUSED(file_index);
    const std::string name = path ? path : "";
    size_t index = 0;
    while (index < paths_.size() && paths_[index] != name) {
        ++index;
    }
    if (index == paths_.size()) {
        paths_.push_back(name);
    }
    backing_[(size_t) layer][kind] = backing{(int) index, offset, size_t(offset % expert_os::io_alignment)};
}

bool l2_tier::open_files(std::string & reason, const std::vector<uint8_t> * layers) {
    // Files opened earlier stay open: a model that joins later adds its own files.
    files_.resize(paths_.size(), nullptr);
    for (size_t i = 0; i < paths_.size(); ++i) {
        if (paths_[i].empty() || files_[i] != nullptr) {
            continue;
        }
        files_[i] = expert_os::open_unbuffered(paths_[i].c_str());
        if (files_[i] == nullptr) {
            reason = "cannot open '" + paths_[i] + "' for unbuffered reading";
            return false;
        }
    }
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        const int cls = geo_.layer_class[layer];
        if (cls < 0 || (layers != nullptr && ((size_t) layer >= layers->size() || !(*layers)[(size_t) layer]))) {
            continue;
        }
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const backing & b = backing_[(size_t) layer][kind];
            // The sector shift of a tensor is only constant across its experts when one expert is a
            // whole number of sectors. Everything below depends on that.
            if (geo_.class_bytes[cls][kind] % expert_os::io_alignment != 0) {
                reason = "layer " + std::to_string(layer) + " kind " + std::to_string(kind) +
                    " has an expert slice of " + std::to_string(geo_.class_bytes[cls][kind]) +
                    " bytes, which is not a whole number of sectors";
                return false;
            }
            if (b.file < 0 || (size_t) b.file >= files_.size() || files_[(size_t) b.file] == nullptr) {
                reason = "layer " + std::to_string(layer) + " kind " + std::to_string(kind) +
                    " has no file behind it";
                return false;
            }
            const uint64_t need = b.offset + uint64_t(geo_.n_experts)*geo_.class_bytes[cls][kind];
            const uint64_t have = expert_os::file_size(files_[(size_t) b.file]);
            if (need > have) {
                reason = "layer " + std::to_string(layer) + " kind " + std::to_string(kind) +
                    " runs past the end of '" + paths_[(size_t) b.file] + "'";
                return false;
            }
        }
    }
    return true;
}

bool l2_tier::wanted(int layer, int expert) const {
    if (layer < 0 || layer >= geo_.n_layers || (size_t) layer >= vram_.size()) {
        return true;
    }
    return vram_[layer][expert] >= 0 || homes_[layer][expert].resident();
}

// ---- addresses ---------------------------------------------------------------------------------------

uint64_t l2_tier::address_of(int layer, int kind, int expert) const {
    const int cls = geo_.layer_class[layer];
    if (cls < 0 || host_geo_.device_base.empty() || vram_[(size_t) layer][(size_t) expert] >= 0) {
        return 0;   // the VRAM slot table answers this one
    }
    const expert_location home = homes_[layer][expert];
    if (class_layout()) {
        if (!mapped_) {
            return 0;   // no device aliases yet; publish_tables publishes zeros until then
        }
        // Every host resident is served from here: its payload sits at slot + shift, which the
        // host slot table cannot express.
        if (home.storage == expert_storage::host) {
            return uint64_t(reinterpret_cast<uintptr_t>(location_address(layer, kind, home, true)));
        }
        const int slot = cledger_.slot_of(layer, expert);
        if (!cledger_.owns(slot, layer, expert)) {
            return 0;
        }
        return uint64_t(reinterpret_cast<uintptr_t>(
            static_cast<char *>(storage_slot(layer, kind, slot, true)) + backing_[layer][kind].shift));
    }
    if (home.storage == expert_storage::host) { return 0; } // The host slot table answers.
    if (home.storage == expert_storage::lent) {
        return uint64_t(reinterpret_cast<uintptr_t>(location_address(layer, kind, home, true)));
    }
    const int ring = ledger_.slot_of(layer, expert);
    if (!ledger_.owns(ring, layer, expert)) {
        return 0;
    }
    return uint64_t(reinterpret_cast<uintptr_t>(
        static_cast<char *>(ring_device_base_[kind]) + size_t(ring)*ring_pitch_[kind] + backing_[layer][kind].shift));
}

const uint64_t * l2_tier::addresses(int layer, int kind) const {
    // Nothing is in the file, so the host slot table answers every miss and the kernels take
    // exactly the path they took without a tier.
    if ((!mailbox_active_ && !any_lent_ && !class_layout()) || maps_device_ == nullptr || layer < 0 ||
            layer >= geo_.n_layers) {
        return nullptr;
    }
    return maps_device_ + ((size_t) layer*geometry::n_kinds + (size_t) kind)*(size_t) geo_.n_experts;
}

void l2_tier::publish_tables() {
    const bool addresses_ready = mapped_ && !host_geo_.device_base.empty();
    const size_t words = bitmap_words();
    any_ssd_ = false;
    any_lent_ = false;
    if (serve_host_ != nullptr) { memset(serve_host_, 0, bitmap_bytes()); }
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        const bool routed = geo_.layer_class[layer] >= 0;
        for (int expert = 0; expert < geo_.n_experts; ++expert) {
            const bool ssd = routed && !wanted(layer, expert);
            any_lent_ = any_lent_ || (routed && homes_[layer][expert].storage == expert_storage::lent);
            any_ssd_ = any_ssd_ || ssd;
            // A ring occupant is `ssd` as well: it must reach the worker to get its lease extended.
            if (ssd && serve_host_ != nullptr) {
                serve_host_[(size_t) layer*words + size_t(expert/32)] |= 1u << (expert%32);
            }
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                maps_host_[((size_t) layer*geometry::n_kinds + (size_t) kind)*(size_t) geo_.n_experts +
                    (size_t) expert] = routed && addresses_ready ? address_of(layer, kind, expert) : 0;
            }
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void l2_tier::set_homes(const std::vector<std::vector<int32_t>> & vram,
        const expert_locations & host, const l2_host_geometry & host_geo, bool keep_ring) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    vram_ = vram;
    homes_ = host;
    host_geo_ = host_geo;
    if (class_layout() && keep_ring) {
        // install_relabel already made the resident slots match the plan
        GGML_ASSERT(relabel_installs());
        for (int layer = 0; layer < geo_.n_layers && layer < (int) homes_.size(); ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0) { continue; }
            for (int expert = 0; expert < geo_.n_experts; ++expert) {
                const expert_location at = homes_[layer][expert];
                if (at.storage == expert_storage::file) { continue; }
                GGML_ASSERT(at.storage == expert_storage::host && at.slot >= 0 && at.slot < storage_.resident_slots[cls]);
                const int s = storage_.storage_of[cls], slot = home_slot_[cls][at.slot];
                int l = -1, e = -1;
                if (cledger_.role_of(s, slot) != slot_role::resident || !cledger_.occupant(s, slot, l, e) || l != layer || e != expert) {
                    GGML_ABORT("expert cache: the SSD tier's slot of layer %d expert %d does not hold it after the install", layer, expert);
                }
            }
        }
    } else if (class_layout()) {
        // The resident roles mirror the plan; the rings start empty, as with the ring layout.
        cledger_.clear_residents();
        for (int layer = 0; layer < geo_.n_layers && layer < (int) homes_.size(); ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0) { continue; }
            for (int expert = 0; expert < geo_.n_experts; ++expert) {
                const expert_location at = homes_[layer][expert];
                if (at.storage == expert_storage::file) { continue; }
                GGML_ASSERT(at.storage == expert_storage::host && at.slot >= 0 && at.slot < storage_.resident_slots[cls]);
                GGML_ASSERT(cledger_.set_resident(storage_.storage_of[cls], home_slot_[cls][at.slot], layer, expert));
            }
        }
        cledger_.discard();
    } else {
        ledger_.discard();
    }
    // an install may relabel or empty the ring slots an early read filled
    early_clear_all();
    publish_tables();
    if (verify_) { verify_addresses(-1, {}); }
}

void l2_tier::refresh_addresses() {
    GGML_ASSERT(!worker_running());
    std::lock_guard<std::mutex> lock(io_mutex_);
    publish_tables();
    if (verify_) { verify_addresses(-1, {}); }
}

void * l2_tier::location_address(int layer, int kind, expert_location at, bool device) const {
    const int cls = geo_.layer_class[layer];
    if (class_layout()) {
        if (at.storage != expert_storage::host) { return nullptr; }
        GGML_ASSERT(cls >= 0 && at.slot >= 0 && at.slot < storage_.resident_slots[cls]);
        return static_cast<char *>(storage_slot(layer, kind, home_slot_[cls][at.slot], device)) +
            backing_[(size_t) layer][kind].shift;
    }
    if (at.storage == expert_storage::host) {
        const auto & bases = device ? host_geo_.device_base : host_geo_.host_base;
        return static_cast<char *>(bases[cls][kind]) + size_t(at.slot)*geo_.class_bytes[cls][kind];
    }
    if (at.storage == expert_storage::lent) {
        GGML_ASSERT(at.slot >= 0 && at.slot < prompt_slots_);
        return static_cast<char *>(device ? ring_device_base_[kind] : ring_host_base_[kind]) + size_t(at.slot)*ring_pitch_[kind];
    }
    return nullptr;
}

void * l2_tier::host_address(int layer, int kind, int expert) const {
    return location_address(layer, kind, homes_[layer][expert]);
}

void l2_tier::finish_write(int layer, int kind, expert_location at) {
    const int cls = geo_.layer_class[layer];
    const size_t bytes = geo_.class_bytes[cls][kind];
    if (class_layout()) {
        // The slot's previous occupant may have sat at another sector shift, so its bytes can
        // reach past this payload.
        if (at.storage != expert_storage::host) { return; }
        const size_t shift = backing_[(size_t) layer][kind].shift;
        const size_t pitch = storage_.pitch[storage_.storage_of[cls]][kind];
        memset(static_cast<char *>(location_address(layer, kind, at)) + bytes, 0, pitch - shift - bytes);
        return;
    }
    if (at.storage != expert_storage::lent) { return; }
    memset(static_cast<char *>(location_address(layer, kind, at)) + bytes, 0, ring_pitch_[kind] - bytes);
}

bool l2_tier::set_prompt_ring(bool prompt) {
    prompt_ring_ = prompt;
    // The class layout keeps its rings at their allocated size in both phases and lends nothing.
    if (class_layout()) { return false; }
    const int slots = prompt ? prompt_slots_ : decode_slots_;
    if (slots == ledger_.ring_count()) { return false; }
    std::lock_guard<std::mutex> lock(io_mutex_);
    GGML_ASSERT(ledger_.set_ring_count(slots));
    ++counters_.repartitions;
    return true;
}

// ---- reads -------------------------------------------------------------------------------------------

bool l2_tier::read_install(std::vector<l2_read> & reads, std::string & reason) {
    GGML_ASSERT(!worker_running());
    if (!class_layout()) {
        for (const auto & read : reads) {
            GGML_ASSERT(!read.direct && read.slot >= 0 && read.slot < install_read_slots());
        }
        ledger_.discard();
        return run_reads(reads, reason, true);
    }
    // A slice for a host slot is read straight into it; one on its way to VRAM into a ring slot of
    // its class, one slot per (layer, expert) of the batch for all kinds.
    cledger_.discard();
    std::vector<l2_eviction> evicted;
    std::vector<std::array<int, 3>> scratch;   // layer, expert, storage slot
    for (l2_read & read : reads) {
        const int cls = geo_.layer_class[read.layer];
        GGML_ASSERT(cls >= 0);
        if (read.direct) {
            GGML_ASSERT(read.slot >= 0 && read.slot < storage_.resident_slots[cls]);
            read.slot = home_slot_[cls][read.slot];
            continue;
        }
        int slot = -1;
        for (const auto & item : scratch) {
            if (item[0] == read.layer && item[1] == read.expert) { slot = item[2]; }
        }
        if (slot < 0) {
            slot = cledger_.take_free(storage_.storage_of[cls], evicted);
            if (slot < 0) {
                reason = "no ring slot of storage class " + std::to_string(storage_.storage_of[cls]) +
                    " for an install read of layer " + std::to_string(read.layer);
                return false;
            }
            scratch.push_back({read.layer, read.expert, slot});
        }
        read.slot = slot;
    }
    return run_reads(reads, reason, true);
}

bool l2_tier::install_relabel(const install_transaction & tx, const expert_locations & before, l1_arena & l1,
        l2_install_stats & stats, std::string & reason) {
    GGML_ASSERT(relabel_installs() && !worker_running() && mapped_);
    std::lock_guard<std::mutex> lock(io_mutex_);
    stats = {};
    // The device is idle, so every generation it published is done and no ring slot is leased.
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        cledger_.set_done(layer, expert_os::load_acquire(&static_cast<const l2_mailbox *>(mail_host_)[layer].done));
    }
    std::vector<std::array<int, 2>> occupants;
    for (int s = 0; s < storage_.storages(); ++s) {
        for (int slot : cledger_.order(s)) {
            int l = -1, e = -1;
            if (cledger_.occupant(s, slot, l, e)) { occupants.push_back({l, e}); }
        }
    }
    stats.ring_before = occupants.size();
    const l2_install_plan plan = plan_relabel_install(cledger_, home_slot_, geo_.layer_class, storage_.storage_of,
        before, tx.host, tx.moves);
    if (!plan.ok) {
        reason = plan.reason;
        return false;
    }
    auto payload = [&](int layer, int kind, int slot) {
        return static_cast<char *>(storage_slot(layer, kind, slot)) + backing_[(size_t) layer][kind].shift;
    };
    std::vector<size_t> copied_back;   // moves from VRAM, their tails are cleared once the copies are done
    for (size_t k = 0; k < tx.moves.size();) {
        const install_move & m = tx.moves[k];
        if (m.from.storage == expert_storage::file) {
            // one read batch: every read goes out together, then the batch's copies to VRAM
            size_t end = k;
            std::vector<l2_read> reads;
            while (end < tx.moves.size() && tx.moves[end].from.storage == expert_storage::file &&
                    plan.steps[end].batch == plan.steps[k].batch) {
                const install_move & mv = tx.moves[end];
                const l2_install_step & step = plan.steps[end];
                const size_t bytes = geo_.class_total_bytes(mv.cls);
                if (step.read) {
                    const int slot = mv.to.storage == expert_storage::host ? step.dst : step.src;
                    for (int kind = 0; kind < geometry::n_kinds; ++kind) { reads.push_back({mv.layer, kind, mv.expert, slot}); }
                    ++stats.read_slices; stats.read_bytes += bytes;
                } else if (step.relabel) {
                    ++stats.relabel_slices; stats.relabel_bytes += bytes;
                } else {
                    ++stats.ring_copy_slices; stats.ring_copy_bytes += bytes;
                }
                ++end;
            }
            // copies out of these slots, queued by earlier moves, complete before a read lands
            if (!l1.sync_copies() || !run_reads(reads, reason, true)) {
                return false;
            }
            for (size_t j = k; j < end; ++j) {
                const install_move & mv = tx.moves[j];
                const l2_install_step & step = plan.steps[j];
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    const int slot = mv.to.storage == expert_storage::host ? step.dst : step.src;
                    if (verify_ && !step.read && !verify_slice(mv.layer, kind, mv.expert, payload(mv.layer, kind, slot), reason)) {
                        return false;
                    }
                    if (mv.to.storage == expert_storage::vram &&
                            !l1.write_gpu_slice(mv.cls, kind, mv.to.slot, payload(mv.layer, kind, step.src), false)) {
                        reason = "a copy to VRAM was refused";
                        return false;
                    }
                }
            }
            k = end;
            continue;
        }
        const l2_install_step & step = plan.steps[k];
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const bool ok = m.from.storage == expert_storage::host ?
                l1.write_gpu_slice(m.cls, kind, m.to.slot, payload(m.layer, kind, step.src), false) :
                l1.read_gpu_slice(m.cls, kind, m.from.slot, payload(m.layer, kind, step.dst));
            if (!ok) {
                reason = "a copy between VRAM and the host was refused";
                return false;
            }
        }
        if (m.from.storage == expert_storage::vram) { copied_back.push_back(k); }
        ++k;
    }
    if (!l1.sync_copies()) {
        reason = "the copy stream failed";
        return false;
    }
    for (size_t k : copied_back) {
        const install_move & m = tx.moves[k];
        const int s = storage_.storage_of[m.cls];
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            // the slot's previous occupant may have sat at another sector shift
            const size_t shift = backing_[(size_t) m.layer][kind].shift, bytes = geo_.class_bytes[m.cls][kind];
            char * data = payload(m.layer, kind, plan.steps[k].dst);
            memset(data + bytes, 0, storage_.pitch[s][kind] - shift - bytes);
            if (verify_ && !verify_slice(m.layer, kind, m.expert, data, reason)) {
                return false;
            }
        }
    }
    counters_.install_relabel_bytes   += stats.relabel_bytes;
    counters_.install_ring_copy_bytes += stats.ring_copy_bytes;
    stats.demoted = plan.demoted;
    stats.batches = plan.batches;
    for (const auto & item : occupants) {
        const int slot = cledger_.slot_of(item[0], item[1]);
        stats.ring_kept += slot >= 0 && cledger_.owns(slot, item[0], item[1]) ? 1 : 0;
    }
    for (int s = 0; s < storage_.storages(); ++s) { stats.ring_after += (size_t) cledger_.ring_occupants(s); }
    return true;
}

bool l2_tier::run_reads(const std::vector<l2_read> & reads, std::string & reason, bool install,
        const std::function<bool(size_t)> & progress) {
    if (reads.empty()) {
        return true;
    }
    std::vector<expert_os::read_op> ops;
    ops.reserve(reads.size());
    for (const l2_read & read : reads) {
        const int cls = geo_.layer_class[read.layer];
        const backing & b = backing_[(size_t) read.layer][read.kind];
        const size_t slice = geo_.class_bytes[cls][read.kind];
        const uint64_t offset = b.offset + uint64_t(read.expert)*slice;
        if (size_t(offset % expert_os::io_alignment) != b.shift) {
            reason = "the file offset of layer " + std::to_string(read.layer) + " expert " +
                std::to_string(read.expert) + " is not at the sector shift of its tensor";
            return false;
        }
        expert_os::read_op op;
        op.file   = files_[(size_t) b.file];
        op.offset = offset - b.shift;
        op.dst    = read_slot(read);
        op.bytes  = expert_os::align_up_io(slice + b.shift);
        GGML_ASSERT(op.bytes <= read_pitch(read));
        ops.push_back(op);
    }
    if (!queue_->submit(ops.data(), ops.size())) {
        reason = "the read queue refused a descriptor";
        return false;
    }
    auto finish = [&](size_t i) {
        const int cls = geo_.layer_class[reads[i].layer];
        const size_t shift = backing_[(size_t) reads[i].layer][reads[i].kind].shift;
        const size_t need = geo_.class_bytes[cls][reads[i].kind] + shift;
        if (queue_->results()[i].got < need) {
            reason = "short read for layer " + std::to_string(reads[i].layer) + " expert " +
                std::to_string(reads[i].expert) + " kind " + std::to_string(reads[i].kind);
            return false;
        }
        // MMQ loads complete K tiles past the final row, starting at the tensor's sector shift.
        memset(static_cast<char *>(read_slot(reads[i])) + need, 0, read_pitch(reads[i]) - need);
        (install ? counters_.install_ssd_bytes : counters_.ssd_bytes) += geo_.class_bytes[cls][reads[i].kind];
        ++(install ? counters_.install_ssd_reads : counters_.ssd_reads);
        return true;
    };
    auto check = [&](size_t i) {
        const l2_read & read = reads[i];
        return !verify_ || verify_slice(read.layer, read.kind, read.expert,
            static_cast<char *>(read_slot(read)) + backing_[read.layer][read.kind].shift, reason);
    };
    if (progress) {
        // Each completed prefix is finished and checked before its reader hears of it; the queue
        // keeps the later reads in flight meanwhile.
        size_t finished = 0;
        return queue_->wait_all(cfg_.read_wait_ms, &reason, [&](size_t count) {
            for (; finished < count; ++finished) {
                if (!finish(finished) || !check(finished)) { return false; }
            }
            return progress(count);
        });
    }
    if (!queue_->wait_all(cfg_.read_wait_ms, &reason)) {
        return false;
    }
    for (size_t i = 0; i < reads.size(); ++i) {
        if (!finish(i)) { return false; }
    }
    for (size_t i = 0; i < reads.size(); ++i) {
        if (!check(i)) { return false; }
    }
    return true;
}

// One slice into any destination. The host arena slots are not sector aligned, so the read lands in
// the bounce slot and is copied from there. Not thread safe: the install path and the verification
// path both run with the worker stopped or under the io mutex.
bool l2_tier::read_slice(int layer, int kind, int expert, void * dst, std::string & reason) {
    if (!read_raw(layer, kind, expert, dst, reason)) {
        return false;
    }
    if (verify_ && !verify_slice(layer, kind, expert, dst, reason)) {
        return false;
    }
    counters_.install_ssd_bytes += geo_.class_bytes[geo_.layer_class[layer]][kind];
    ++counters_.install_ssd_reads;
    return true;
}

bool l2_tier::verify_slice(int layer, int kind, int expert, const void * data, std::string & reason) {
    const backing & b = backing_[(size_t) layer][kind];
    if (b.file < 0 || (size_t) b.file >= paths_.size()) {
        reason = "layer " + std::to_string(layer) + " kind " + std::to_string(kind) + " has no file behind it";
        return false;
    }
    const size_t bytes = geo_.class_bytes[geo_.layer_class[layer]][kind];
    const uint64_t offset = b.offset + uint64_t(expert)*bytes;
    verify_buffer_.resize(bytes);
    verify_files_.resize(paths_.size());
    auto & handle = verify_files_[(size_t) b.file];
    if (!handle) { handle.reset(new std::ifstream(paths_[(size_t) b.file], std::ios::binary)); }
    auto & file = *handle;
    file.seekg((std::streamoff) offset);
    file.read(verify_buffer_.data(), (std::streamsize) bytes);
    if (!file || memcmp(data, verify_buffer_.data(), bytes) != 0) {
        ++counters_.verify_bad;
        reason = "verification failed for layer " + std::to_string(layer) + " expert " +
            std::to_string(expert) + " kind " + std::to_string(kind) + " file '" +
            paths_[(size_t) b.file] + "' offset " + std::to_string(offset);
        return false;
    }
    ++counters_.verify_fills;
    counters_.verify_bytes += bytes;
    return true;
}

bool l2_tier::read_raw(int layer, int kind, int expert, void * dst, std::string & reason) {
    const int cls = geo_.layer_class[layer];
    if (cls < 0 || dst == nullptr) {
        reason = "layer " + std::to_string(layer) + " is not routed";
        return false;
    }
    const backing & b = backing_[(size_t) layer][kind];
    if (b.file < 0 || (size_t) b.file >= files_.size() || files_[(size_t) b.file] == nullptr) {
        reason = "layer " + std::to_string(layer) + " kind " + std::to_string(kind) + " has no open file behind it";
        return false;
    }
    const size_t slice = geo_.class_bytes[cls][kind];
    expert_os::read_op op;
    op.file   = files_[(size_t) b.file];
    op.offset = b.offset + uint64_t(expert)*slice - b.shift;
    op.dst    = bounce_aligned_;
    op.bytes  = expert_os::align_up_io(slice + b.shift);
    if (!queue_->submit(&op, 1) || !queue_->wait_all(cfg_.read_wait_ms, &reason)) {
        if (reason.empty()) {
            reason = "the read queue refused a descriptor";
        }
        return false;
    }
    if (queue_->results()[0].got < slice + b.shift) {
        reason = "short read for layer " + std::to_string(layer) + " expert " + std::to_string(expert);
        return false;
    }
    memcpy(dst, static_cast<char *>(bounce_aligned_) + b.shift, slice);
    return true;
}

// ---- demand service -----------------------------------------------------------------------------

std::vector<int> l2_tier::demanded_ids(int layer) const {
    std::vector<int> ids;
    const uint32_t * bits = demand_host_ + size_t(layer)*bitmap_words();
    for (int expert = 0; expert < geo_.n_experts; ++expert) {
        if (bits[expert/32] & (1u << (expert%32))) { ids.push_back(expert); }
    }
    return ids;
}

// The per-ubatch record. A generation the GPU answered itself runs only this part: it reads and
// leases nothing, but it still counts its rows, distinct experts and bytes by tier.
void l2_tier::account_layer(int layer, uint32_t seq, const std::vector<int> & ids) {
    auto & item = pending_[layer];
    item = {}; item.layer = layer; item.seq = seq;
    item.rows = static_cast<l2_mailbox *>(mail_host_)[layer].rows;
    if (layer == first_layer_) { ++ubatch_; inferred_prompt_ = item.rows > cfg_.decode_rows; }
    item.ubatch = ubatch_;
    const int phase = phase_.load();
    item.prompt = phase < 0 ? inferred_prompt_ : phase == 0;
    item.distinct = ids.size(); counters_.distinct += ids.size();
    const uint64_t bytes = geo_.class_total_bytes(geo_.layer_class[layer]);
    for (int e : ids) {
        if (vram_[layer][e] >= 0) { counters_.vram_bytes += bytes; }
        else if (homes_[layer][e].resident()) { counters_.host_bytes += bytes; }
        else { counters_.file_bytes += bytes; }
    }
    if (layer == first_layer_) {
        if (item.prompt) { ++counters_.prompt_ubatches; }
        else { ++counters_.decode_ubatches; counters_.decode_tokens += item.rows; }
    }
}

void l2_tier::service_layer(int layer, uint32_t seq, const std::vector<int> & ids) {
    const auto start = std::chrono::steady_clock::now();
    account_layer(layer, seq, ids);
    auto & item = pending_[layer];
    const auto before_bytes = counters_.ssd_bytes;
    for (int other = 0; other < geo_.n_layers; ++other) {
        const uint32_t done = expert_os::load_acquire(&static_cast<l2_mailbox *>(mail_host_)[other].done);
        if (class_layout()) { cledger_.set_done(other, done); } else { ledger_.set_done(other, done); }
    }
    const int launches = batched() ?
        int(expert_os::load_acquire(&static_cast<const l2_batch_mail *>(batch_mail_host_)[layer].launches)) : 0;
    if (launches > 0) {
        serve_batched(layer, seq, ids, launches);
        item.ssd_bytes = counters_.ssd_bytes - before_bytes;
        std::atomic_thread_fence(std::memory_order_release);
        counters_.service_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return;
    }
    auto homed = [this](int l, int e) { return wanted(l, e); };
    // a prompt ubatch's fills and hits go where RANMA_EXPERT_L2_PROMPT_FILL says, decode's to the most recently used end
    const l2_insert insert = l2_insert_for(cfg_.prompt_fill, item.rows > cfg_.decode_rows);
    l2_service service;
    if (cfg_.early) {
        // a slot an early read filled for this demand is not a ring hit: its read was counted
        auto quiet = [this](int l, int e) { return early_marked(l, e); };
        service = class_layout() ? cledger_.service(layer, ids, seq, homed, quiet, false, insert) :
            ledger_.service(layer, ids, seq, homed, quiet);
    } else {
        service = class_layout() ? cledger_.service(layer, ids, seq, homed, l2_no_quiet(), false, insert) :
            ledger_.service(layer, ids, seq, homed);
    }
    if (!service.ok) {
        GGML_ABORT("expert cache: the SSD tier cannot serve layer %d: %s", layer, service.reason.c_str());
    }
    for (const l2_eviction & old : service.evicted) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            maps_host_[((size_t) old.layer*geometry::n_kinds + kind)*geo_.n_experts + old.expert] = 0;
        }
        early_evicted(old.layer, old.expert);
    }
    counters_.ring_hits += (uint64_t) service.hits;
    if (cfg_.early && !early_list_[(size_t) layer].empty()) {
        // this service consumed the layer's early reads: the demanded ones were hits, the rest wasted
        counters_.early_hits += (uint64_t) service.early_hits;
        std::vector<bool> asked((size_t) geo_.n_experts, false);
        for (int e : ids) {
            if (e >= 0 && e < geo_.n_experts) { asked[(size_t) e] = true; }
        }
        for (int e : early_list_[(size_t) layer]) {
            uint8_t & mark = early_mark_[(size_t) layer*(size_t) geo_.n_experts + (size_t) e];
            if (mark && !asked[(size_t) e]) { ++counters_.early_unused; }
            mark = 0;
        }
        early_list_[(size_t) layer].clear();
    }
    std::string reason;
    if (staged() && item.rows >= uint64_t(cfg_.staged_min_rows)) {
        serve_staged(layer, seq, service.reads);
    } else if (!run_reads(service.reads, reason)) {
        GGML_ABORT("expert cache: the SSD tier failed to read layer %d: %s", layer, reason.c_str());
    }
    item.ssd_bytes = counters_.ssd_bytes - before_bytes;
    publish_layer(layer);
    if (verify_) { verify_addresses(layer, ids); }
    std::atomic_thread_fence(std::memory_order_release);
    counters_.service_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The batched service of one prompt generation (expert-l2-batch.h). The experts that need no read
// (residents and ring hits) are released at once for the first launch of every kind. The file
// experts go in batches through one or two buffers of ring slots: the worker reads every batch that
// may be read (the first two of each kind at once, batch b once the GPU has finished batch b - 2 of
// that kind) in the order up, gate, down, publishes the addresses of a batch of a kind when its
// reads are in place and then the batch's readiness. Before batch b of a kind is read into a buffer,
// the addresses of that kind of the batch the buffer held go, so the address table ends as the
// ledger does: each slot holds the last batch expert that used it.
void l2_tier::serve_batched(int layer, uint32_t seq, const std::vector<int> & ids, int launches) {
    l2_mailbox & mail = static_cast<l2_mailbox *>(mail_host_)[layer];
    l2_batch_mail & bm = static_cast<l2_batch_mail *>(batch_mail_host_)[layer];
    auto homed = [this](int l, int e) { return wanted(l, e); };
    const l2_insert insert = l2_insert_for(cfg_.prompt_fill, true);
    const int capacity = batch_capacity(layer);
    l2_batch_plan plan;
    if (cfg_.early) {
        auto quiet = [this](int l, int e) { return early_marked(l, e); };
        plan = cledger_.service_batched(layer, ids, seq, homed, launches, capacity, quiet, insert);
    } else {
        plan = cledger_.service_batched(layer, ids, seq, homed, launches, capacity, l2_no_quiet(), insert);
    }
    if (!plan.ok) {
        GGML_ABORT("expert cache: the SSD tier cannot serve layer %d in batches: %s", layer, plan.reason.c_str());
    }
    const size_t n_experts = (size_t) geo_.n_experts;
    auto entry = [&](int l, int kind, int e) -> uint64_t & {
        return maps_host_[((size_t) l*geometry::n_kinds + (size_t) kind)*n_experts + (size_t) e];
    };
    for (const auto & old : plan.evicted) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) { entry(old.layer, kind, old.expert) = 0; }
        early_evicted(old.layer, old.expert);
    }
    for (const std::vector<int> & batch : plan.batches) {
        for (int e : batch) {
            // a demoted hit still names its old slot, which a batch may take
            for (int kind = 0; kind < geometry::n_kinds; ++kind) { entry(layer, kind, e) = 0; }
        }
    }
    counters_.ring_hits += (uint64_t) plan.hits;
    counters_.batch_demoted += (uint64_t) plan.demoted;
    if (cfg_.early && !early_list_[(size_t) layer].empty()) {
        counters_.early_hits += (uint64_t) plan.early_hits;
        std::vector<bool> asked(n_experts, false);
        for (int e : ids) {
            if (e >= 0 && e < geo_.n_experts) { asked[(size_t) e] = true; }
        }
        for (int e : early_list_[(size_t) layer]) {
            uint8_t & mark = early_mark_[(size_t) layer*n_experts + (size_t) e];
            if (mark && !asked[(size_t) e]) { ++counters_.early_unused; }
            mark = 0;
        }
        early_list_[(size_t) layer].clear();
    }
    const int batches = plan.count();
    if (batches > 0) {
        int32_t * tab = batch_tab_host_ + (size_t) layer*batch_tab_words();
        for (size_t i = 0; i < n_experts; ++i) { tab[i] = 0; }
        for (size_t i = n_experts; i < batch_tab_words(); ++i) { tab[i] = -1; }
        GGML_ASSERT(batches <= launches && plan.capacity <= capacity);
        for (int b = 0; b < batches; ++b) {
            for (size_t p = 0; p < plan.batches[(size_t) b].size(); ++p) {
                const int e = plan.batches[(size_t) b][p];
                tab[(size_t) e] = b + 1;
                tab[n_experts + (size_t) b*(size_t) capacity + p] = e;
            }
        }
        bm.count = (uint32_t) batches;
        std::atomic_thread_fence(std::memory_order_release);
        expert_os::store_release(&bm.gen, seq);
    }
    std::atomic_thread_fence(std::memory_order_release);
    // the first launch of every kind reads residents and hits only
    for (int kind = 0; kind < geometry::n_kinds; ++kind) { expert_os::store_release(&mail.kind_ready[kind], seq); }
    if (batches == 0) {
        return;
    }
    std::vector<int> next(geometry::n_kinds, 0), done(geometry::n_kinds, 0);
    auto waited = std::chrono::steady_clock::now();   // since the last read wave
    std::string reason;
    while (next[0] < batches || next[1] < batches || next[2] < batches) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            const uint64_t word = expert_os::load_acquire(&bm.done[kind]);
            done[(size_t) kind] = uint32_t(word >> 32) == seq ? int(uint32_t(word)) : 0;
        }
        const std::vector<std::pair<int, int>> groups = l2_batch_readable(plan, next, done);
        if (groups.empty()) {
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waited).count();
            if (ms > double(cfg_.read_wait_ms)) {
                GGML_ABORT("expert cache: layer %d: the GPU did not finish a batch within %lld ms (done %d/%d/%d of %d)",
                    layer, (long long) cfg_.read_wait_ms, done[0], done[1], done[2], batches);
            }
            RANMA_CPU_PAUSE();
            continue;
        }
        std::vector<l2_read> reads;
        std::vector<size_t> ends;
        for (const auto & group : groups) {
            const int kind = group.first, b = group.second;
            const std::vector<int> & batch = plan.batches[(size_t) b];
            for (size_t p = 0; p < batch.size(); ++p) {
                const int before = plan.previous(b, (int) p);
                if (before >= 0) { entry(layer, kind, before) = 0; }   // the GPU has finished it
                reads.push_back({layer, kind, batch[p], plan.slot(b, (int) p)});
            }
            ends.push_back(reads.size());
        }
        std::atomic_thread_fence(std::memory_order_release);
        size_t released = 0;
        auto progress = [&](size_t count) {
            for (; released < groups.size() && ends[released] <= count; ++released) {
                const int kind = groups[released].first, b = groups[released].second;
                const std::vector<int> & batch = plan.batches[(size_t) b];
                for (size_t p = 0; p < batch.size(); ++p) {
                    entry(layer, kind, batch[p]) = uint64_t(reinterpret_cast<uintptr_t>(
                        static_cast<char *>(storage_slot(layer, kind, plan.slot(b, (int) p), true)) +
                        backing_[(size_t) layer][kind].shift));
                }
                std::atomic_thread_fence(std::memory_order_release);
                // the last batch of a kind also releases the launches the generation leaves empty
                expert_os::store_release(&bm.ready[kind], (uint64_t(seq) << 32) | uint64_t(b + 1 == batches ? launches : b + 1));
            }
            return true;
        };
        if (!run_reads(reads, reason, false, progress)) {
            GGML_ABORT("expert cache: the SSD tier failed to read layer %d: %s", layer, reason.c_str());
        }
        GGML_ASSERT(released == groups.size());
        waited = std::chrono::steady_clock::now();
    }
    ++counters_.batched_generations;
    counters_.batches += (uint64_t) batches;
    if (verify_) {
        std::vector<int> owned = plan.hit_experts;
        for (const std::vector<int> & batch : plan.batches) {
            for (int e : batch) {
                if (cledger_.slot_of(layer, e) >= 0) { owned.push_back(e); }
            }
        }
        verify_addresses(layer, owned);
    }
}


// The reads of one generation, ordered by kind in the order the graph multiplies them (up, gate,
// down). Each kind's addresses and readiness are published as soon as its last read is in place, so
// the kernels of a kind run while the next kind is still being read. The slots, pins and evictions
// are those the ledger chose for the whole generation; only the order of the reads and the moment
// each kind is published differ from the single service.
void l2_tier::serve_staged(int layer, uint32_t seq, const std::vector<l2_read> & reads) {
    std::vector<l2_read> ordered;
    ordered.reserve(reads.size());
    std::array<size_t, geometry::n_kinds> end = {};
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
        for (const l2_read & read : reads) {
            if (read.kind == kind) { ordered.push_back(read); }
        }
        end[kind] = ordered.size();
    }
    l2_mailbox & mail = static_cast<l2_mailbox *>(mail_host_)[layer];
    int released = 0;
    auto release = [&](size_t count) {
        for (; released < geometry::n_kinds && end[released] <= count; ++released) {
            publish_layer_kind(layer, released);
            expert_os::store_release(&mail.kind_ready[released], seq);
        }
        return true;
    };
    release(0);   // the kinds with nothing to read
    std::string reason;
    bool ok = true;
    if (cfg_.staged_drain) {
        for (int kind = 0; kind < geometry::n_kinds && ok; ++kind) {
            const size_t begin = kind == 0 ? 0 : end[kind - 1];
            ok = run_reads(std::vector<l2_read>(ordered.begin() + begin, ordered.begin() + end[kind]), reason) &&
                release(end[kind]);
        }
    } else {
        ok = run_reads(ordered, reason, false, release);
    }
    if (!ok) {
        GGML_ABORT("expert cache: the SSD tier failed to read layer %d: %s", layer, reason.c_str());
    }
    GGML_ASSERT(released == geometry::n_kinds);
    ++counters_.staged_generations;
}

void l2_tier::publish_layer(int layer) {
    for (int expert = 0; expert < geo_.n_experts; ++expert) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            maps_host_[((size_t) layer*geometry::n_kinds + kind)*geo_.n_experts + expert] =
                address_of(layer, kind, expert);
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void l2_tier::publish_layer_kind(int layer, int kind) {
    for (int expert = 0; expert < geo_.n_experts; ++expert) {
        maps_host_[((size_t) layer*geometry::n_kinds + kind)*geo_.n_experts + expert] = address_of(layer, kind, expert);
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void l2_tier::verify_addresses(int demanded_layer, const std::vector<int> & ids) {
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        for (int expert = 0; expert < geo_.n_experts; ++expert) {
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                const uint64_t actual = maps_host_[((size_t) layer*geometry::n_kinds + kind)*geo_.n_experts + expert];
                const uint64_t expected = address_of(layer, kind, expert);
                if (actual != expected) {
                    ++counters_.verify_bad;
                    GGML_ABORT("expert cache: address ownership mismatch at layer %d expert %d kind %d: "
                               "actual %llu expected %llu", layer, expert, kind,
                        (unsigned long long) actual, (unsigned long long) expected);
                }
                ++counters_.owner_checks;
            }
        }
    }
    if (demanded_layer < 0) {
        return;
    }
    for (int expert : ids) {
        if (expert < 0 || expert >= geo_.n_experts) {
            GGML_ABORT("expert cache: invalid demanded expert %d in layer %d", expert, demanded_layer);
        }
        if (wanted(demanded_layer, expert)) {
            continue;
        }
        const int slot = class_layout() ? cledger_.slot_of(demanded_layer, expert) : ledger_.slot_of(demanded_layer, expert);
        if (!(class_layout() ? cledger_.owns(slot, demanded_layer, expert) : ledger_.owns(slot, demanded_layer, expert))) {
            ++counters_.verify_bad;
            GGML_ABORT("expert cache: demanded layer %d expert %d has no owned ring slot",
                demanded_layer, expert);
        }
    }
}

void l2_tier::handle_publish(int layer, uint32_t seq) {
    l2_mailbox * mail = static_cast<l2_mailbox *>(mail_host_);
    collect_wait(layer);   // the previous sample, before this one overwrites it
    collect_stage(layer);
    if (mail[layer].invalid) {
        GGML_ABORT("expert cache: invalid router id at layer %d", layer);
    }
    if (expert_os::load_acquire(&mail[layer].ready) != seq) {
        service_layer(layer, seq, demanded_ids(layer));
        expert_os::store_release(&mail[layer].ready, seq);
        ++counters_.generations;
    } else {
        account_layer(layer, seq, demanded_ids(layer));
    }
    seen_[(size_t) layer] = seq;
}

void l2_tier::worker_loop() {
    std::string pin_error;
    if (!expert_os::pin_current_thread(cfg_.worker_cpu, pin_error)) {
        GGML_LOG_WARN("expert cache: SSD worker pin failed: %s\n", pin_error.c_str());
    }
    l2_mailbox * mail = static_cast<l2_mailbox *>(mail_host_);
    running_.store(true, std::memory_order_relaxed);
    GGML_LOG_DEBUG("expert cache: the SSD tier worker is running (%s)\n",
        any_ssd_ ? "serving the file" : "nothing is in the file, it will idle");
    while (!stop_.load(std::memory_order_relaxed)) {
        bool work = false;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            if (geo_.layer_class[layer] < 0) {
                continue;
            }
            const uint32_t seq = expert_os::load_acquire(&mail[layer].published);
            if (seq != seen_[(size_t) layer]) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                handle_publish(layer, seq);
                work = true;
            }
            if (expert_os::load_acquire(&mail[layer].wait_generation) != wait_seen_[layer]) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                collect_wait(layer);
            }
        }
        // Early demands come after every published generation: a generation never waits for a
        // read it does not need.
        if (!work && early_pending_.load(std::memory_order_acquire)) {
            std::vector<l2_early_layer> job;
            {
                std::lock_guard<std::mutex> lock(early_mutex_);
                job.swap(early_job_);
                early_pending_.store(false, std::memory_order_release);
            }
            if (!job.empty()) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                serve_early(std::move(job));
                work = true;
            }
        }
        if (!work) {
            RANMA_CPU_PAUSE();
        }
    }
    running_.store(false, std::memory_order_relaxed);
}

void l2_tier::start_worker() {
    // The worker also runs for a plan with nothing in the file: the layers then answer themselves,
    // and the worker only records their samples.
    if (!mailbox_active_ || worker_.joinable()) {
        return;
    }
    // Capture before the new thread can miss the first publish while it is being scheduled.
    const l2_mailbox * mail = static_cast<const l2_mailbox *>(mail_host_);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        seen_[(size_t) layer] = expert_os::load_acquire(&mail[layer].published);
    }
    {
        // a demand posted before the stop belongs to a plan that may be gone
        std::lock_guard<std::mutex> lock(early_mutex_);
        early_job_.clear();
        early_pending_.store(false, std::memory_order_release);
    }
    stop_.store(false);
    worker_ = std::thread([this]() { worker_loop(); });
}

void l2_tier::stop_worker() {
    stop_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
    running_.store(false, std::memory_order_relaxed);
    if (mail_host_) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        for (int l = 0; l < geo_.n_layers; ++l) { collect_wait(l); collect_stage(l); }
    }
}

// ---- early reads ------------------------------------------------------------------------------------

void l2_tier::set_early_layers(const std::vector<uint8_t> & layers) {
    early_layer_.assign(geo_.n_layers, 0);
    for (int l = 0; l < geo_.n_layers && l < (int) layers.size(); ++l) {
        early_layer_[(size_t) l] = layers[(size_t) l] && geo_.layer_class[l] >= 0 ? 1 : 0;
    }
}

uint32_t l2_tier::published(int layer) const {
    if (mail_host_ == nullptr || layer < 0 || layer >= geo_.n_layers) { return 0; }
    return expert_os::load_acquire(&static_cast<const l2_mailbox *>(mail_host_)[layer].published);
}

void l2_tier::post_early(std::vector<l2_early_layer> layers) {
    if (!cfg_.early || !mailbox_active_ || !any_ssd_ || layers.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(early_mutex_);
    for (l2_early_layer & in : layers) {
        if (in.layer < 0 || in.layer >= geo_.n_layers || geo_.layer_class[in.layer] < 0 || in.ids.empty()) { continue; }
        l2_early_layer * same = nullptr;
        for (l2_early_layer & have : early_job_) { same = have.layer == in.layer ? &have : same; }
        if (same == nullptr) {
            early_job_.push_back(std::move(in));
        } else if (same->published == in.published) {
            same->ids.insert(same->ids.end(), in.ids.begin(), in.ids.end());   // for the same generation
        } else {
            *same = std::move(in);
        }
    }
    std::sort(early_job_.begin(), early_job_.end(),
        [](const l2_early_layer & a, const l2_early_layer & b) { return a.layer < b.layer; });
    early_pending_.store(!early_job_.empty(), std::memory_order_release);
}

void l2_tier::early_evicted(int layer, int expert) {
    if (!early_marked(layer, expert)) { return; }
    early_mark_[(size_t) layer*(size_t) geo_.n_experts + (size_t) expert] = 0;
    ++counters_.early_lost;
}

void l2_tier::early_clear_all() {
    for (int l = 0; l < (int) early_list_.size(); ++l) {
        for (int e : early_list_[(size_t) l]) {
            uint8_t & mark = early_mark_[(size_t) l*(size_t) geo_.n_experts + (size_t) e];
            counters_.early_lost += mark ? 1 : 0;
            mark = 0;
        }
        early_list_[(size_t) l].clear();
    }
}

// Serves posted early demands, lowest layer first, as one read batch. Each layer's addresses are
// published as soon as its last read is in place, and while the later layers are still being read,
// every published generation that needs no read (typically a layer served early) is answered at once.
// The reads are ordinary demand reads that happen sooner: counted in ssd_reads/ssd_bytes, checked
// under RANMA_EXPERT_L2_VERIFY; the generation that later hits them does not count them as ring hits
// again.
void l2_tier::serve_early(std::vector<l2_early_layer> job) {
    ++counters_.early_jobs;
    for (int other = 0; other < geo_.n_layers; ++other) {
        const uint32_t done = expert_os::load_acquire(&static_cast<l2_mailbox *>(mail_host_)[other].done);
        if (class_layout()) { cledger_.set_done(other, done); } else { ledger_.set_done(other, done); }
    }
    auto homed = [this](int l, int e) { return wanted(l, e); };
    struct span { int layer; size_t end; };
    std::vector<span> spans;
    std::vector<l2_read> reads;
    for (const l2_early_layer & in : job) {
        const int layer = in.layer;
        if (published(layer) != in.published) {
            ++counters_.early_late;   // its generation is out; the ordinary service reads it
            continue;
        }
        const uint32_t now = class_layout() ? cledger_.done(layer) : ledger_.done(layer);
        const l2_service service = class_layout() ? cledger_.service(layer, in.ids, now, homed, l2_no_quiet(), true) :
            ledger_.service(layer, in.ids, now, homed, l2_no_quiet(), true);
        if (!service.ok) {
            // no reusable ring slot right now: the ordinary service will read it
            if (!early_warned_) {
                GGML_LOG_WARN("expert cache: an early read of layer %d was skipped: %s\n", layer, service.reason.c_str());
                early_warned_ = true;
            }
            ++counters_.early_late;
            continue;
        }
        for (const l2_eviction & old : service.evicted) {
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                maps_host_[((size_t) old.layer*geometry::n_kinds + kind)*geo_.n_experts + old.expert] = 0;
            }
            early_evicted(old.layer, old.expert);
        }
        counters_.early_already += (uint64_t) service.hits;
        for (const l2_read & r : service.reads) {
            if (r.kind == 0) {
                uint8_t & mark = early_mark_[(size_t) layer*(size_t) geo_.n_experts + (size_t) r.expert];
                if (!mark) { early_list_[(size_t) layer].push_back(r.expert); }
                mark = 1;
            }
            counters_.early_bytes += geo_.class_bytes[geo_.layer_class[layer]][r.kind];
            ++counters_.early_reads;
            reads.push_back(r);
        }
        ++counters_.early_layers;
        spans.push_back({layer, reads.size()});
    }
    // The addresses go out before the reads complete: no kernel reads them before the worker has
    // served the layer's generation, and that waits for the layer's reads (`busy`). A slot that was
    // evicted for them was reusable, so no running generation reads it either.
    std::vector<uint8_t> busy((size_t) geo_.n_layers, 0);
    for (const span & sp : spans) {
        publish_layer(sp.layer);
        busy[(size_t) sp.layer] = 1;
    }
    std::atomic_thread_fence(std::memory_order_release);
    if (verify_) { verify_addresses(-1, {}); }
    size_t completed_spans = 0;
    auto progress = [&](size_t count) {
        bool any = false;
        for (; completed_spans < spans.size() && spans[completed_spans].end <= count; ++completed_spans) {
            busy[(size_t) spans[completed_spans].layer] = 0;
            any = true;
        }
        if (any && completed_spans < spans.size()) { serve_ready(busy); }
        return true;
    };
    std::string reason;
    if (!reads.empty() && !run_reads(reads, reason, false, progress)) {
        GGML_ABORT("expert cache: the SSD tier failed an early read: %s", reason.c_str());
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void l2_tier::serve_ready(const std::vector<uint8_t> & busy) {
    const l2_mailbox * mail = static_cast<const l2_mailbox *>(mail_host_);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        if (geo_.layer_class[layer] < 0 || busy[(size_t) layer]) { continue; }
        const uint32_t seq = expert_os::load_acquire(&mail[layer].published);
        if (seq == seen_[(size_t) layer]) { continue; }
        bool resident = true;
        for (int e : demanded_ids(layer)) {
            resident = resident && (wanted(layer, e) ||
                (class_layout() ? cledger_.slot_of(layer, e) : ledger_.slot_of(layer, e)) >= 0);
        }
        if (!resident) { continue; }   // it needs a read: after this batch
        handle_publish(layer, seq);
        ++counters_.early_inline;
    }
}

// ---- the hot path ---------------------------------------------------------------------------------------

int l2_tier::batch_capacity(int layer) const {
    const int s = storage_.storage_of[geo_.layer_class[layer]];
    return std::min(geo_.n_experts, l2_batch_capacity(storage_.ring_slots[s]));
}

int l2_tier::batch_launches(int layer, int64_t rows, bool mmq) const {
    if (!batched() || !mmq || !mailbox_active_ || layer < 0 || layer >= geo_.n_layers || geo_.layer_class[layer] < 0 ||
            rows <= int64_t(cfg_.decode_rows)) {
        return 0;
    }
    const int s = storage_.storage_of[geo_.layer_class[layer]];
    const int demand = storage_demand_bound(geo_.n_experts, cfg_.experts_used, uint64_t(rows));
    // a ubatch the vector kernels multiply keeps them while the ring holds its worst demand
    if (rows <= cfg_.vec_rows && demand <= storage_.ring_slots[s]) {
        return 0;
    }
    return l2_batch_launches(demand, batch_capacity(layer));
}

l2_tier::batch_tables l2_tier::batch_device(int layer) const {
    batch_tables out;
    if (batch_dev_ == nullptr || layer < 0 || layer >= geo_.n_layers || geo_.layer_class[layer] < 0) { return out; }
    out.batch_of = batch_dev_ + (size_t) layer*batch_tab_words();
    out.lists    = out.batch_of + geo_.n_experts;
    out.capacity = batch_capacity(layer);
    return out;
}

void l2_tier::batch_wait(int layer, int kind, int batch, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    ggml_cuda_kernel_launch(l2_batch_wait_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer,
        static_cast<l2_batch_mail *>(batch_mail_device_) + layer, kind, batch);
}

void l2_tier::batch_done(int layer, int kind, int batch, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    ggml_cuda_kernel_launch(l2_batch_done_kernel, launch, static_cast<const l2_mailbox *>(mail_device_) + layer,
        static_cast<l2_batch_mail *>(batch_mail_device_) + layer, kind, batch);
}

void l2_tier::publish_and_wait(int layer, const ggml_tensor * ids, cudaStream_t stream, int launches) {
    if (!mailbox_active_ || mail_device_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            geo_.layer_class[layer] < 0) {
        return;
    }
    if (ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t)) {
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    const ggml_cuda_kernel_launch_params publish_launch(dim3(1), dim3(128), bitmap_words()*sizeof(uint32_t), stream);
    ggml_cuda_kernel_launch(l2_publish_kernel, publish_launch,
        static_cast<l2_mailbox *>(mail_device_) + layer,
        demand_device_ + size_t(layer)*bitmap_words(), serve_device_ + size_t(layer)*bitmap_words(),
        (const int32_t *) ids->data, (int) ids->ne[1], (int) ids->ne[0],
        (int) (ids->nb[1]/sizeof(int32_t)), geo_.n_experts,
        static_cast<l2_batch_mail *>(batch_mail_device_) + layer, launches);
    route_launches_[(size_t) layer] = launches;
    if (launches > 0) {
        // The first launch of every kind needs no read; the batches wait for their own reads.
        staged_layer_[(size_t) layer] = 1;
        ggml_cuda_kernel_launch(l2_wait_kind_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer, 0, 1);
        const ggml_cuda_kernel_launch_params prep(dim3(1), dim3(256), 0, stream);
        ggml_cuda_kernel_launch(l2_batch_prep_kernel, prep, static_cast<const l2_mailbox *>(mail_device_) + layer,
            static_cast<const l2_batch_mail *>(batch_mail_device_) + layer,
            (const int32_t *) (batch_tab_device_ + (size_t) layer*batch_tab_words()),
            batch_dev_ + (size_t) layer*batch_tab_words(), geo_.n_experts, (int) batch_tab_words());
        return;
    }
    if (staged()) {
        // The worker decides from the same row count. A mismatch would still be safe: every kind
        // wait also passes on the whole generation, and the worker publishes that last either way.
        staged_layer_[(size_t) layer] = any_ssd_ && ids->ne[1] >= cfg_.staged_min_rows;
        if (staged_layer_[(size_t) layer]) {
            ggml_cuda_kernel_launch(l2_wait_kind_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer, 0, 1);
            return;
        }
    }
    ggml_cuda_kernel_launch(l2_wait_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer);
}

void l2_tier::wait_kind(int layer, int kind, cudaStream_t stream) {
    if (!kind_waits() || !mailbox_active_ || mail_device_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            kind <= 0 || kind >= geometry::n_kinds || !staged_layer_[(size_t) layer]) {
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    ggml_cuda_kernel_launch(l2_wait_kind_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer, kind, 0);
}

void l2_tier::mark_done(int layer, cudaStream_t stream) {
    if (!mailbox_active_ || mail_device_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            geo_.layer_class[layer] < 0) {
        return;
    }
    route_launches_[(size_t) layer] = 0;
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    ggml_cuda_kernel_launch(l2_done_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer);
}

void l2_tier::collect_wait(int layer) {
    const auto & mail = static_cast<const l2_mailbox *>(mail_host_)[layer];
    const uint32_t seq = expert_os::load_acquire(&mail.wait_generation);
    if (seq == wait_seen_[layer]) { return; }
    auto & item = pending_[layer];
    // A generation the GPU answered itself stores wait_generation without the worker, so the sample
    // can be one sweep behind. Leave it for the next sweep.
    if (item.seq != seq) { return; }
    item.wait_ticks = expert_os::load_acquire(&mail.wait_ticks);
    item.steady_ticks = expert_os::load_acquire(&mail.steady_ticks);
    wait_seen_[layer] = seq;
    counters_.wait_ticks += item.wait_ticks;
    counters_.steady_ticks += item.steady_ticks;
    (item.prompt ? counters_.prompt_wait_ticks : counters_.decode_wait_ticks) += item.steady_ticks;
    if (!item.prompt && layer < (int) early_layer_.size() && early_layer_[(size_t) layer]) {
        counters_.early_layer_wait_ticks += item.steady_ticks;
        ++counters_.early_layer_samples;
    }
    ++counters_.measured_layers;
    if (cfg_.log_mask & GGML_EXPERT_LOG_L2) { samples_.push_back(item); }
    if (counters_.measured_layers % 64 == 0) { report_counters("periodic"); }
}

// The later kind waits of a staged layer add up in its mailbox; a generation's share arrives with the
// next generation of the layer or when the worker stops.
void l2_tier::collect_stage(int layer) {
    if (!kind_waits()) { return; }
    const uint32_t ticks = expert_os::load_acquire(&static_cast<const l2_mailbox *>(mail_host_)[layer].stage_ticks);
    counters_.staged_wait_ticks += uint32_t(ticks - stage_seen_[(size_t) layer]);
    stage_seen_[(size_t) layer] = ticks;
    if (batched()) {
        const uint32_t bt = expert_os::load_acquire(&static_cast<const l2_batch_mail *>(batch_mail_host_)[layer].stage_ticks);
        counters_.batch_wait_ticks += uint32_t(bt - batch_seen_[(size_t) layer]);
        batch_seen_[(size_t) layer] = bt;
    }
}

void l2_tier::report(const char * what) { report_counters(what); }

void l2_tier::report_counters(const char * what) {
    if ((cfg_.log_mask & GGML_EXPERT_LOG_L2) == 0) { return; }
    std::ostringstream out;
    out << "{\"kind\":\"l2\",\"event\":\"" << what << "\",\"generations\":" << counters_.generations
        << ",\"measured_layers\":" << counters_.measured_layers << ",\"clock_khz\":" << clock_khz_
        << ",\"clock64_ticks\":" << counters_.wait_ticks << ",\"clock64_nominal_ms\":" << double(counters_.wait_ticks)/clock_khz_
        << ",\"steady_khz\":" << steady_khz_ << ",\"gpu_wait_ms\":" << wait_ms()
        << ",\"prompt_ubatches\":" << counters_.prompt_ubatches << ",\"decode_ubatches\":" << counters_.decode_ubatches
        << ",\"decode_tokens\":" << counters_.decode_tokens
        << ",\"prompt_wait_ms\":" << (steady_khz_ > 0 ? double(counters_.prompt_wait_ticks)/steady_khz_ : 0)
        << ",\"decode_wait_ms\":" << (steady_khz_ > 0 ? double(counters_.decode_wait_ticks)/steady_khz_ : 0)
        << ",\"distinct\":" << counters_.distinct << ",\"vram_bytes\":" << counters_.vram_bytes
        << ",\"host_bytes\":" << counters_.host_bytes << ",\"file_bytes\":" << counters_.file_bytes
        << ",\"ssd_bytes\":" << counters_.ssd_bytes << ",\"reads\":" << counters_.ssd_reads
        << ",\"ring_hits\":" << counters_.ring_hits << ",\"service_ms\":" << counters_.service_ms
        << ",\"install_ssd_bytes\":" << counters_.install_ssd_bytes << ",\"install_ssd_reads\":" << counters_.install_ssd_reads
        << ",\"install_relabel_bytes\":" << counters_.install_relabel_bytes
        << ",\"install_ring_copy_bytes\":" << counters_.install_ring_copy_bytes
        << ",\"verify_fills\":" << counters_.verify_fills << ",\"verify_bytes\":" << counters_.verify_bytes
        << ",\"verify_bad\":" << counters_.verify_bad << ",\"owner_checks\":" << counters_.owner_checks
        << ",\"staged_generations\":" << counters_.staged_generations
        << ",\"staged_wait_ms\":" << (steady_khz_ > 0 ? double(counters_.staged_wait_ticks)/steady_khz_ : 0)
        << ",\"batched_generations\":" << counters_.batched_generations << ",\"batches\":" << counters_.batches
        << ",\"batch_demoted\":" << counters_.batch_demoted
        << ",\"batch_wait_ms\":" << (steady_khz_ > 0 ? double(counters_.batch_wait_ticks)/steady_khz_ : 0)
        << ",\"repartitions\":" << counters_.repartitions << ",\"ring_slots\":" << ring_count()
        << ",\"layout\":\"" << (class_layout() ? "class" : "ring") << '"';
    if (std::find(early_layer_.begin(), early_layer_.end(), uint8_t(1)) != early_layer_.end()) {
        out << ",\"early\":" << (cfg_.early ? 1 : 0)
            << ",\"early_layer_decode_wait_ms\":" << (steady_khz_ > 0 ? double(counters_.early_layer_wait_ticks)/steady_khz_ : 0)
            << ",\"early_layer_samples\":" << counters_.early_layer_samples
            << ",\"early_jobs\":" << counters_.early_jobs << ",\"early_layers\":" << counters_.early_layers
            << ",\"early_late\":" << counters_.early_late << ",\"early_reads\":" << counters_.early_reads
            << ",\"early_bytes\":" << counters_.early_bytes << ",\"early_already\":" << counters_.early_already
            << ",\"early_hits\":" << counters_.early_hits << ",\"early_unused\":" << counters_.early_unused
            << ",\"early_lost\":" << counters_.early_lost << ",\"early_inline\":" << counters_.early_inline;
    }
    if (class_layout()) {
        out << ",\"class_ring_slots\":[";
        for (int s = 0; s < storage_.storages(); ++s) { out << (s ? "," : "") << storage_.ring_slots[s]; }
        out << ']';
    }
    out
        << ",\"samples\":[";
    bool comma = false;
    for (const auto & item : samples_) {
        if (comma) { out << ','; } comma = true;
        out << "{\"layer\":" << item.layer << ",\"ubatch\":" << item.ubatch << ",\"rows\":" << item.rows
            << ",\"phase\":\"" << (item.prompt ? "prefill" : "decode") << "\",\"distinct\":" << item.distinct
            << ",\"ssd_bytes\":" << item.ssd_bytes << ",\"gpu_wait_ms\":" << double(item.steady_ticks)/steady_khz_ << '}';
    }
    out << "]}";
    GGML_LOG_INFO("expert_metrics %s\n", out.str().c_str());
    samples_.clear();
}

} // namespace ggml_cuda_expert

#endif // GGML_USE_HIP
