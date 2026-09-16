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
    uint32_t rows, wait_generation, reserved0;
    uint64_t wait_ticks, steady_ticks;
    uint32_t need;            // set by the publish kernel when a routed expert needs the worker
    uint32_t reserved[3];
};

// `serve` marks the experts only the worker can place: file residents and ring occupants. It follows
// a plan, not a generation. When no routed id needs the worker, this kernel answers its own wait, so
// a fully resident layer never reaches the CPU.
static __global__ void l2_publish_kernel(l2_mailbox * m, uint32_t * demand, const uint32_t * serve,
        const int32_t * ids, int rows, int used, int stride, int experts) {
    const int words = (experts + 31)/32;
    for (int i = threadIdx.x; i < words; i += blockDim.x) { demand[i] = 0; }
    if (threadIdx.x == 0) {
        m->invalid = 0;
        m->need = 0;
        m->rows = uint32_t(rows);
        __hip_atomic_store(&m->ready, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < int64_t(rows)*used; i += blockDim.x) {
        const int e = ids[(i/used)*stride + i%used];
        if ((unsigned) e < (unsigned) experts) {
            atomicOr(demand + e/32, 1u << (e%32));
            if (serve[e/32] & (1u << (e%32))) { atomicOr(&m->need, 1u); }
        } else {
            atomicOr(&m->invalid, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t seq = ++m->generation;
        if (!m->need) {
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
    return expert_os::align_up_io(mail) + expert_os::align_up_io(maps) + 2*expert_os::align_up_io(demand) + bounce;
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
    for (int l = 0; l < geo_.n_layers; ++l) {
        if (geo_.layer_class[l] >= 0) { first_layer_ = l; break; }
    }

    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
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

    ledger_.reset(geo_.n_layers, geo_.n_experts, prompt_slots_, prompt_slots_, geometry::n_kinds);
    prompt_ring_ = true;
    GGML_LOG_INFO("expert cache: SSD tier ring %d slots (%zu MiB, prompt) / %d slots (%zu MiB, decode), "
                  "metadata and bounce %zu KiB, queue depth %d\n",
        prompt_slots_, ring_bytes(prompt_slots_)/(1024*1024),
        decode_slots_, ring_bytes(decode_slots_)/(1024*1024),
        metadata_bytes()/1024, cfg_.queue_depth);
    if (!size_note_.empty()) {
        GGML_LOG_INFO("expert cache: %s\n", size_note_.c_str());
    }
    return true;
}

bool l2_tier::map() {
    if (mapped_) {
        return true;
    }
    ggml_cuda_set_device(device_);
    for (int kind = 0; kind < geometry::n_kinds; ++kind) {
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
    if ((size_t) file_index >= paths_.size()) {
        paths_.resize((size_t) file_index + 1);
    }
    paths_[(size_t) file_index] = path ? path : "";
    backing_[(size_t) layer][kind] = backing{file_index, offset, size_t(offset % expert_os::io_alignment)};
}

bool l2_tier::open_files(std::string & reason) {
    files_.assign(paths_.size(), nullptr);
    for (size_t i = 0; i < paths_.size(); ++i) {
        if (paths_[i].empty()) {
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
        if (cls < 0) {
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
    if (home.storage == expert_storage::host) { return 0; } // The host slot table answers.
    if (home.storage == expert_storage::lent) {
        return uint64_t(reinterpret_cast<uintptr_t>(location_address(cls, kind, home, true)));
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
    if ((!mailbox_active_ && !any_lent_) || maps_device_ == nullptr || layer < 0 || layer >= geo_.n_layers) {
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
        const expert_locations & host, const l2_host_geometry & host_geo) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    vram_ = vram;
    homes_ = host;
    host_geo_ = host_geo;
    ledger_.discard();
    publish_tables();
    if (verify_) { verify_addresses(-1, {}); }
}

void * l2_tier::location_address(int cls, int kind, expert_location at, bool device) const {
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
    return location_address(geo_.layer_class[layer], kind, homes_[layer][expert]);
}

void l2_tier::finish_write(int cls, int kind, expert_location at) {
    if (at.storage != expert_storage::lent) { return; }
    const size_t bytes = geo_.class_bytes[cls][kind];
    memset(static_cast<char *>(location_address(cls, kind, at)) + bytes, 0, ring_pitch_[kind] - bytes);
}

bool l2_tier::set_prompt_ring(bool prompt) {
    const int slots = prompt ? prompt_slots_ : decode_slots_;
    prompt_ring_ = prompt;
    if (slots == ledger_.ring_count()) { return false; }
    std::lock_guard<std::mutex> lock(io_mutex_);
    GGML_ASSERT(ledger_.set_ring_count(slots));
    ++counters_.repartitions;
    return true;
}

// ---- reads -------------------------------------------------------------------------------------------

bool l2_tier::read_install(const std::vector<l2_read> & reads, std::string & reason) {
    GGML_ASSERT(!worker_running());
    for (const auto & read : reads) {
        GGML_ASSERT(read.slot >= 0 && read.slot < install_read_slots());
    }
    ledger_.discard();
    return run_reads(reads, reason, true);
}

bool l2_tier::run_reads(const std::vector<l2_read> & reads, std::string & reason, bool install) {
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
        op.dst    = ring_host(read.kind, read.slot);
        op.bytes  = expert_os::align_up_io(slice + b.shift);
        ops.push_back(op);
    }
    if (!queue_->submit(ops.data(), ops.size())) {
        reason = "the read queue refused a descriptor";
        return false;
    }
    if (!queue_->wait_all(cfg_.read_wait_ms, &reason)) {
        return false;
    }
    for (size_t i = 0; i < reads.size(); ++i) {
        const int cls = geo_.layer_class[reads[i].layer];
        const size_t shift = backing_[(size_t) reads[i].layer][reads[i].kind].shift;
        const size_t need = geo_.class_bytes[cls][reads[i].kind] + shift;
        if (queue_->results()[i].got < need) {
            reason = "short read for layer " + std::to_string(reads[i].layer) + " expert " +
                std::to_string(reads[i].expert) + " kind " + std::to_string(reads[i].kind);
            return false;
        }
        // MMQ loads complete K tiles past the final row, starting at the tensor's sector shift.
        memset(static_cast<char *>(ring_host(reads[i].kind, reads[i].slot)) + need,
            0, ring_pitch_[reads[i].kind] - need);
        (install ? counters_.install_ssd_bytes : counters_.ssd_bytes) += geo_.class_bytes[cls][reads[i].kind];
        ++(install ? counters_.install_ssd_reads : counters_.ssd_reads);
    }
    if (verify_) {
        for (const l2_read & read : reads) {
            if (!verify_slice(read.layer, read.kind, read.expert,
                    static_cast<char *>(ring_host(read.kind, read.slot)) + backing_[read.layer][read.kind].shift, reason)) {
                return false;
            }
        }
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
        ledger_.set_done(other, expert_os::load_acquire(&static_cast<l2_mailbox *>(mail_host_)[other].done));
    }
    const l2_service service = ledger_.service(layer, ids, seq, [this](int l, int e) { return wanted(l, e); });
    if (!service.ok) {
        GGML_ABORT("expert cache: the SSD tier cannot serve layer %d: %s", layer, service.reason.c_str());
    }
    for (const l2_eviction & old : service.evicted) {
        for (int kind = 0; kind < geometry::n_kinds; ++kind) {
            maps_host_[((size_t) old.layer*geometry::n_kinds + kind)*geo_.n_experts + old.expert] = 0;
        }
    }
    counters_.ring_hits += (uint64_t) service.hits;
    std::string reason;
    if (!run_reads(service.reads, reason)) {
        GGML_ABORT("expert cache: the SSD tier failed to read layer %d: %s", layer, reason.c_str());
    }
    item.ssd_bytes = counters_.ssd_bytes - before_bytes;
    publish_layer(layer);
    if (verify_) { verify_addresses(layer, ids); }
    std::atomic_thread_fence(std::memory_order_release);
    counters_.service_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
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
        const int slot = ledger_.slot_of(demanded_layer, expert);
        if (!ledger_.owns(slot, demanded_layer, expert)) {
            ++counters_.verify_bad;
            GGML_ABORT("expert cache: demanded layer %d expert %d has no owned ring slot",
                demanded_layer, expert);
        }
    }
}

void l2_tier::worker_loop(std::vector<uint32_t> seen) {
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
            if (seq != seen[(size_t) layer]) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                collect_wait(layer);   // the previous sample, before this one overwrites it
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
                seen[(size_t) layer] = seq;
                work = true;
            }
            if (expert_os::load_acquire(&mail[layer].wait_generation) != wait_seen_[layer]) {
                std::lock_guard<std::mutex> lock(io_mutex_);
                collect_wait(layer);
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
    std::vector<uint32_t> seen((size_t) geo_.n_layers);
    for (int layer = 0; layer < geo_.n_layers; ++layer) {
        seen[(size_t) layer] = expert_os::load_acquire(&mail[layer].published);
    }
    stop_.store(false);
    worker_ = std::thread([this, seen = std::move(seen)]() mutable { worker_loop(std::move(seen)); });
}

void l2_tier::stop_worker() {
    stop_.store(true);
    if (worker_.joinable()) {
        worker_.join();
    }
    running_.store(false, std::memory_order_relaxed);
    if (mail_host_) {
        std::lock_guard<std::mutex> lock(io_mutex_);
        for (int l = 0; l < geo_.n_layers; ++l) { collect_wait(l); }
    }
}

// ---- the hot path ---------------------------------------------------------------------------------------

void l2_tier::publish_and_wait(int layer, const ggml_tensor * ids, cudaStream_t stream) {
    if (!mailbox_active_ || mail_device_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            geo_.layer_class[layer] < 0) {
        return;
    }
    if (ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t)) {
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(1), 0, stream);
    const ggml_cuda_kernel_launch_params publish_launch(dim3(1), dim3(128), 0, stream);
    ggml_cuda_kernel_launch(l2_publish_kernel, publish_launch,
        static_cast<l2_mailbox *>(mail_device_) + layer,
        demand_device_ + size_t(layer)*bitmap_words(), serve_device_ + size_t(layer)*bitmap_words(),
        (const int32_t *) ids->data, (int) ids->ne[1], (int) ids->ne[0],
        (int) (ids->nb[1]/sizeof(int32_t)), geo_.n_experts);
    ggml_cuda_kernel_launch(l2_wait_kernel, launch, static_cast<l2_mailbox *>(mail_device_) + layer);
}

void l2_tier::mark_done(int layer, cudaStream_t stream) {
    if (!mailbox_active_ || mail_device_ == nullptr || layer < 0 || layer >= geo_.n_layers ||
            geo_.layer_class[layer] < 0) {
        return;
    }
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
    ++counters_.measured_layers;
    if (cfg_.log_mask & GGML_EXPERT_LOG_L2) { samples_.push_back(item); }
    if (counters_.measured_layers % 64 == 0) { report_counters("periodic"); }
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
        << ",\"verify_fills\":" << counters_.verify_fills << ",\"verify_bytes\":" << counters_.verify_bytes
        << ",\"verify_bad\":" << counters_.verify_bad << ",\"owner_checks\":" << counters_.owner_checks
        << ",\"repartitions\":" << counters_.repartitions << ",\"ring_slots\":" << ledger_.ring_count()
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
