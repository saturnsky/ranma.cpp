// Standalone HIP fixture: real mailbox publication, ring reads and MMQ padding, and the VRAM
// size-class redraw on HIP virtual memory.
#include "expert-l2.cu"
#include "expert-os-win32.cpp"
#include "expert-l1.cu"
#include "expert-host.cu"
#include "expert-hash-early.h"
#include "expert-hash-stage.cuh"
#include "mmq.cuh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>

void ggml_cuda_set_device(int d) { CUDA_CHECK(hipSetDevice(d)); }
int ggml_cuda_get_device() { int d = 0; CUDA_CHECK(hipGetDevice(&d)); return d; }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "%s: %s at %s:%d in %s\n", stmt, msg, file, line, func);
    abort();
}
using namespace ggml_cuda_expert;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

// A batch mailbox for the publish kernels the fixtures launch by hand (no batch launches).
static l2_batch_mail * test_batch_mail() {
    static l2_batch_mail * device = nullptr;
    if (device == nullptr) {
        void * host = nullptr;
        if (!coherent_alloc(&host, (void **) &device, sizeof(l2_batch_mail))) { abort(); }
    }
    return device;
}

static __global__ void readback(const uint64_t * table, int expert, char * out, size_t n) {
    const char * src = (const char *) (uintptr_t) table[expert];
    for (size_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += blockDim.x*gridDim.x) { out[i] = src[i]; }
}

static int bitmap_test() {
    constexpr int experts = 73, words = (experts + 31)/32, used = 8, stride = 11;
    l2_mailbox * host = nullptr, * device = nullptr;
    uint32_t * bits = nullptr, * device_bits = nullptr, * serve = nullptr, * device_serve = nullptr;
    CUDA_CHECK(hipHostMalloc(&host, sizeof(*host), hipHostMallocMapped | hipHostMallocCoherent));
    CUDA_CHECK(hipHostGetDevicePointer((void **) &device, host, 0));
    CUDA_CHECK(hipHostMalloc(&bits, words*sizeof(uint32_t), hipHostMallocMapped | hipHostMallocCoherent));
    CUDA_CHECK(hipHostGetDevicePointer((void **) &device_bits, bits, 0));
    CUDA_CHECK(hipHostMalloc(&serve, words*sizeof(uint32_t), hipHostMallocMapped | hipHostMallocCoherent));
    CUDA_CHECK(hipHostGetDevicePointer((void **) &device_serve, serve, 0));
    memset(serve, 0xff, words*sizeof(uint32_t));
    memset(host, 0, sizeof(*host));
    int32_t * ids = nullptr;
    CUDA_CHECK(hipMalloc(&ids, 512*stride*sizeof(int32_t)));
    for (int pattern = 0; pattern < 4; ++pattern) {
        const int rows = pattern == 0 ? 1 : 512;
        std::vector<int32_t> input(rows*stride, -1);
        uint32_t expected[words] = {};
        for (int r = 0; r < rows; ++r) {
            for (int k = 0; k < used; ++k) {
                int e = pattern < 2 ? 40 : (r*used + k)%experts;
                if (pattern == 3 && r == 0 && k == 0) { e = -1; }
                input[r*stride + k] = e;
                if (e >= 0) { expected[e/32] |= 1u << (e%32); }
            }
        }
        CUDA_CHECK(hipMemcpy(ids, input.data(), input.size()*sizeof(int32_t), hipMemcpyHostToDevice));
        l2_publish_kernel<<<1, 128, words*sizeof(uint32_t)>>>(device, device_bits, device_serve, ids, rows, used, stride, experts, test_batch_mail(), 0);
        CUDA_CHECK(hipDeviceSynchronize());
        CHECK(host->published == uint32_t(pattern + 1));
        CHECK(bool(host->invalid) == (pattern == 3));
        CHECK(memcmp(bits, expected, sizeof(expected)) == 0);
    }
    CUDA_CHECK(hipFree(ids));
    CUDA_CHECK(hipHostFree(serve));
    CUDA_CHECK(hipHostFree(bits));
    CUDA_CHECK(hipHostFree(host));
    printf("PASS: bitmap duplicate/distinct/512-row/strided/invalid-id publication\n");
    return 0;
}

// The serve bitmap decides whether a generation needs the worker at all.
static int serve_test() {
    constexpr int experts = 73, words = (experts + 31)/32, used = 8, stride = 11, rows = 4;
    l2_mailbox * host = nullptr, * device = nullptr;
    uint32_t * bits = nullptr, * device_bits = nullptr, * serve = nullptr, * device_serve = nullptr;
    CHECK(coherent_alloc((void **) &host, (void **) &device, sizeof(*host)));
    CHECK(coherent_alloc((void **) &bits, (void **) &device_bits, words*sizeof(uint32_t)));
    CHECK(coherent_alloc((void **) &serve, (void **) &device_serve, words*sizeof(uint32_t)));
    int32_t * ids = nullptr;
    CUDA_CHECK(hipMalloc(&ids, rows*stride*sizeof(int32_t)));
    std::vector<int32_t> input(rows*stride, -1);
    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < used; ++k) { input[r*stride + k] = (r*used + k)%experts; }
    }
    const int routed = input[0], absent = experts - 1;   // rows*used = 32 < absent
    CUDA_CHECK(hipMemcpy(ids, input.data(), input.size()*sizeof(int32_t), hipMemcpyHostToDevice));

    l2_publish_kernel<<<1, 128, words*sizeof(uint32_t)>>>(device, device_bits, device_serve, ids, rows, used, stride, experts, test_batch_mail(), 0);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(host->published == 1 && host->need == 0 && host->ready == host->generation);

    serve[routed/32] |= 1u << (routed%32);
    l2_publish_kernel<<<1, 128, words*sizeof(uint32_t)>>>(device, device_bits, device_serve, ids, rows, used, stride, experts, test_batch_mail(), 0);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(host->published == 2 && host->need == 1 && host->ready == 0);

    serve[routed/32] &= ~(1u << (routed%32));
    serve[absent/32] |= 1u << (absent%32);
    l2_publish_kernel<<<1, 128, words*sizeof(uint32_t)>>>(device, device_bits, device_serve, ids, rows, used, stride, experts, test_batch_mail(), 0);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(host->published == 3 && host->need == 0 && host->ready == host->generation);

    CUDA_CHECK(hipFree(ids));
    CUDA_CHECK(hipHostFree(serve));
    CUDA_CHECK(hipHostFree(bits));
    CUDA_CHECK(hipHostFree(host));
    printf("PASS: a resident plan answers its own wait; one served expert still calls the worker\n");
    return 0;
}

static __global__ void wait_started(l2_mailbox * m) {
    __hip_atomic_store(&m->need, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static int wait_clock_test() {
    l2_mailbox * host = nullptr, * device = nullptr;
    CHECK(coherent_alloc((void **) &host, (void **) &device, sizeof(*host)));
    host->generation = 1;
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    std::chrono::steady_clock::time_point begin;
    std::thread responder([&] {
        while (!expert_os::load_acquire(&host->need)) { std::this_thread::yield(); }
        begin = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        expert_os::store_release(&host->ready, 1);
    });
    wait_started<<<1, 1, 0, stream>>>(device);
    l2_wait_kernel<<<1, 1, 0, stream>>>(device);
    CUDA_CHECK(hipStreamSynchronize(stream));
    responder.join();
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    int khz = 0; CUDA_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, 0));
    int steady_khz = 0; CUDA_CHECK(hipDeviceGetAttribute(&steady_khz, hipDeviceAttributeWallClockRate, 0));
    const double nominal_ms = double(host->wait_ticks)/khz;
    const double gpu_ms = double(host->steady_ticks)/steady_khz;
    printf("CLOCK_CROSSCHECK clock_khz=%d ticks=%llu nominal_ms=%.6f steady_khz=%d gpu_ms=%.6f wall_ms=%.6f ratio=%.6f\n",
        khz, (unsigned long long) host->wait_ticks, nominal_ms, steady_khz, gpu_ms, wall_ms, gpu_ms/wall_ms);
    CHECK(host->wait_generation == 1 && gpu_ms > 0.9*wall_ms && gpu_ms < 1.1*wall_ms);
    CUDA_CHECK(hipStreamDestroy(stream)); CUDA_CHECK(hipHostFree(host));
    return 0;
}

static int mover_test(bool inclusive) {
    constexpr int layers = 2, experts = 4, slice = 4096;
    geometry geo;
    geo.n_layers = layers; geo.n_experts = experts;
    geo.layer_class = {0, 0}; geo.class_layers = {2}; geo.class_bytes = {{slice, slice, slice}};
    geo.tensors.resize(layers);
    ggml_tensor tensors[layers][3] = {};
    std::vector<char> values[layers][3];
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        values[l][k].resize(experts*slice);
        for (int e = 0; e < experts; ++e) { memset(values[l][k].data() + e*slice, 1 + l*12 + k*4 + e, slice); }
        auto & t = tensors[l][k]; t.type = GGML_TYPE_F32; t.ne[0] = 32; t.ne[1] = 32; t.ne[2] = experts; t.ne[3] = 1;
        t.data = values[l][k].data(); geo.tensors[l][k] = &t;
    }
    host_arena host(geo);
    l1_arena gpu(geo);
    CHECK(gpu.allocate({3}, 0, inclusive ? 0 : 2));
    if (!inclusive) {
        CHECK(host.allocate({5}, 2, 0)); gpu.attach_host(&host, 2);
        CHECK(gpu.assign_exclusive({{0, 1}, {0}}));
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
            CHECK(gpu.logical_io(l, k, values[l][k].data(), 0, experts*slice, true));
        }
        CHECK(host.map());
    }
    const std::vector<expert_slot_table> plans{{{0, 1}, {0}}, {{2}, {1, 3}}, {{0, 3}, {2}}};
    for (const auto & selected : plans) {
        l1_install_stats stats;
        CHECK(gpu.install(selected, true, stats));
        std::string reason;
        CHECK(gpu.verify_current_assignment(reason));
        if (inclusive) { CHECK(gpu.verify_resident(0) == 3); CHECK(stats.d2h_bytes == 0); }
        else {
            std::vector<char> actual(experts*slice);
            for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
                CHECK(gpu.logical_io(l, k, actual.data(), 0, actual.size(), false));
                CHECK(actual == values[l][k]);
            }
        }
    }
    printf("PASS: %s real L1 install, readback and assignment invariants\n", inclusive ? "inclusive" : "exclusive");
    return 0;
}

static int finite_inclusive_test() {
    constexpr int slice = 4096, experts = 4;
    geometry geo;
    geo.n_layers = 1; geo.n_experts = experts;
    geo.layer_class = {0}; geo.class_layers = {1}; geo.class_bytes = {{slice, slice, slice}};
    geo.tensors.resize(1);
    ggml_tensor tensor = {}; tensor.type = GGML_TYPE_F32;
    tensor.ne[0] = 32; tensor.ne[1] = 32; tensor.ne[2] = experts; tensor.ne[3] = 1;
    geo.tensors[0] = {&tensor, &tensor, &tensor};
    host_arena host(geo); l1_arena gpu(geo);
    CHECK(host.allocate({2}, 0, 0)); CHECK(gpu.allocate({1}, 0, 0));
    gpu.attach_host(&host, 0, true);
    gpu.attach_addresses([](int, int) -> const uint64_t * { return nullptr; });
    install_layout layout; layout.gpu = {1}; layout.host = {2}; layout.lent_begin = {0}; layout.lent_count = {0};
    const expert_slot_table gs{{0, -1, -1, -1}}, hs{{0, 1, -1, -1}};
    CHECK(gpu.assign_tier(gs, host_locations(hs), layout, {{0}}, {{}}));
    std::vector<char> input(2*slice, char(71)), output(2*slice);
    for (int k = 0; k < 3; ++k) {
        CHECK(gpu.logical_io(0, k, input.data(), 0, input.size(), true));
        CHECK(memcmp(host.slice(0, k, 0), input.data(), slice) == 0);
        std::vector<char> back(slice);
        CUDA_CHECK(hipMemcpy(back.data(), gpu.gpu_slice_address(0, k, 0), slice, hipMemcpyDeviceToHost));
        CHECK(memcmp(back.data(), input.data(), slice) == 0);
        // Cross an expert boundary and update both homes of the GPU resident.
        char patch[64]; memset(patch, 39, sizeof(patch));
        CHECK(gpu.logical_io(0, k, patch, slice - 32, sizeof(patch), true));
        CHECK(gpu.logical_io(0, k, output.data(), 0, output.size(), false));
        CHECK(memcmp(output.data() + slice - 32, patch, sizeof(patch)) == 0);
        CUDA_CHECK(hipMemcpy(back.data(), gpu.gpu_slice_address(0, k, 0), slice, hipMemcpyDeviceToHost));
        CHECK(memcmp(back.data(), output.data(), slice) == 0);
        CHECK(!gpu.logical_io(0, k, output.data(), 2*slice, slice, false));
    }
    std::string reason; CHECK(gpu.verify_current_assignment(reason));
    printf("PASS: finite inclusive host master, GPU duplicate writes and cross-expert I/O\n");
    return 0;
}

// ---- early routes: early SSD reads through the real tier worker ------------------------------------

static bool wait_until(const std::function<bool()> & done, int seconds = 10) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done()) {
        if (std::chrono::steady_clock::now() > deadline) { return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// Two layers, every expert in the file, a ring of 8 slots, one-row generations. A hinted expert is
// read before its layer publishes; the generation then waits for nothing but the worker's answer and
// reads the same bytes; the counters keep early reads out of the ring hits and count them once.
static int early_ssd_test(const std::filesystem::path & dir) {
    constexpr int K = 256, M = 64, experts = 9;
    constexpr size_t slice = size_t(K/32)*24*M, shift = 512;
    const auto path = dir / "early-weights.bin";
    std::vector<char> file(shift + 2*experts*slice + 4096);
    for (size_t i = 0; i < file.size(); ++i) { file[i] = char((i*131 + i/977) & 0xff); }
    { std::ofstream out(path, std::ios::binary); out.write(file.data(), file.size()); CHECK(bool(out)); }
    geometry geo;
    geo.n_layers = 2; geo.n_experts = experts;
    geo.layer_class = {0, 0}; geo.class_layers = {2};
    geo.class_bytes = {{slice, slice, slice}};
    ggml_tensor weights = {};
    weights.type = GGML_TYPE_Q5_1; weights.ne[0] = K; weights.ne[1] = M; weights.ne[2] = experts; weights.ne[3] = 1;
    geo.tensors = {{{&weights, &weights, &weights}}, {{&weights, &weights, &weights}}};
    l2_config cfg; cfg.verify = true; cfg.early = true; cfg.decode_rows = 1; cfg.prefill_rows = 8;
    cfg.prefill_ring_bytes = 1; cfg.decode_ring_bytes = 1;
    l2_tier tier(geo, cfg);
    CHECK(tier.allocate(0));
    l2_host_geometry host_geo; host_geo.device_base.resize(1);
    std::vector<std::vector<int32_t>> homes(2, std::vector<int32_t>(experts, -1));
    tier.set_homes(homes, host_locations(homes), host_geo);
    const std::string filename = path.string();
    // layer l's tensor starts at shift + l*experts*slice; each kind reads the same bytes
    for (int l = 0; l < 2; ++l) for (int k = 0; k < 3; ++k) { tier.set_backing(l, k, 0, filename.c_str(), shift + l*experts*slice); }
    std::string reason;
    CHECK(tier.open_files(reason));
    CHECK(tier.map());
    tier.set_early_layers({1, 1});
    auto expect = [&](int layer, int expert) { return file.data() + shift + (layer*experts + expert)*slice; };
    int32_t * dev_ids = nullptr; char * copy = nullptr;
    CUDA_CHECK(hipMalloc(&dev_ids, 4*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&copy, slice));
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    ggml_tensor ids = {}; ids.type = GGML_TYPE_I32; ids.data = dev_ids;
    ids.ne[0] = 1; ids.ne[1] = 1; ids.ne[2] = ids.ne[3] = 1; ids.nb[0] = ids.nb[1] = 4;
    std::vector<char> back(slice);
    // one generation of `layer` selecting `expert`: its kind-2 slice comes back through the table
    auto generation = [&](int layer, int expert) {
        CUDA_CHECK(hipMemcpyAsync(dev_ids, &expert, sizeof(int32_t), hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        tier.publish_and_wait(layer, &ids, stream);
        readback<<<32, 128, 0, stream>>>(tier.addresses(layer, 2), expert, copy, slice);
        tier.mark_done(layer, stream);
        CUDA_CHECK(hipMemcpyAsync(back.data(), copy, slice, hipMemcpyDeviceToHost, stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        return memcmp(back.data(), expect(layer, expert), slice) == 0;
    };
    auto post = [&](int layer, std::vector<int> e, int stale = 0) {
        l2_early_layer item; item.layer = layer; item.published = tier.published(layer) - (uint32_t) stale; item.ids = std::move(e);
        tier.post_early({item});
    };
    tier.start_worker();
    // 1. a hinted expert is read early and hit by its generation
    post(0, {4, 4});
    CHECK(wait_until([&] { return tier.counters().early_reads == 3; }));
    CHECK(tier.counters().ssd_reads == 3 && tier.counters().early_layers == 1 && tier.counters().early_jobs == 1);
    CHECK(generation(0, 4));
    CHECK(wait_until([&] { return tier.counters().generations == 1; }));
    CHECK(tier.counters().early_hits == 1 && tier.counters().ring_hits == 0 && tier.counters().ssd_reads == 3);
    // 2. a hint whose generation was already published is skipped
    post(1, {2}, 1);
    CHECK(wait_until([&] { return tier.counters().early_late == 1; }));
    CHECK(tier.counters().early_reads == 3);
    // 3. a hinted expert that the generation does not select is counted as unused; the selected one is read
    post(0, {5});
    CHECK(wait_until([&] { return tier.counters().early_reads == 6; }));
    CHECK(generation(0, 6));
    CHECK(wait_until([&] { return tier.counters().generations == 2; }));
    CHECK(tier.counters().early_unused == 1 && tier.counters().ssd_reads == 9 && tier.counters().early_hits == 1);
    // 4. an expert already in the ring is not read again (early_already), and its generation is an ordinary hit
    post(0, {6});
    CHECK(wait_until([&] { return tier.counters().early_already == 1; }));
    CHECK(generation(0, 6));
    CHECK(wait_until([&] { return tier.counters().generations == 3; }));
    CHECK(tier.counters().ring_hits == 1 && tier.counters().ssd_reads == 9);
    // 5. hints and generations of both layers back to back: whatever the timing, each new expert is
    //    read exactly once and every generation reads the file's bytes
    for (int round = 0; round < 40; ++round) {
        const int a = round%experts, b = (round*5 + 3)%experts;
        post(0, {a});
        post(1, {b});
        CHECK(generation(0, a));
        CHECK(generation(1, b));
    }
    tier.stop_worker();
    const l2_counters c = tier.counters();
    CHECK(c.verify_bad == 0 && c.verify_fills == c.ssd_reads);
    CHECK(c.early_reads <= c.ssd_reads && c.early_hits + c.early_unused + c.early_lost <= c.early_reads/3);
    printf("PASS: early SSD reads: %llu early of %llu reads, %llu early hits, %llu late, %llu inline, %llu unused\n",
        (unsigned long long) c.early_reads, (unsigned long long) c.ssd_reads, (unsigned long long) c.early_hits,
        (unsigned long long) c.early_late, (unsigned long long) c.early_inline, (unsigned long long) c.early_unused);
    CUDA_CHECK(hipStreamDestroy(stream)); CUDA_CHECK(hipFree(copy)); CUDA_CHECK(hipFree(dev_ids));
    return 0;
}

// ---- early routes: VRAM staging kernels and their protocol (expert-hash-stage.cuh) ----------------

// Resolves each id like the matmuls: a table slot reads the arena, -1 reads the host copy; writes a
// checksum of the slice per id.
static __global__ void stage_reader(const int32_t * table, const int32_t * ids, int n, const char * arena, const char * host,
        size_t slice, uint32_t * out) {
    const int i = threadIdx.x;
    if (i < n) {
        const int e = ids[i];
        const int slot = table[e];
        const char * src = slot >= 0 ? arena + size_t(slot)*slice : host + size_t(e)*slice;
        uint32_t sum = 0;
        for (size_t j = 0; j < slice; j += 4) { sum = sum*31u + *(const uint32_t *) (src + j); }
        out[i] = sum;
    }
}

// Keeps a stream busy for about `ticks` wall-clock ticks (a slow copy stand-in).
static __global__ void stage_spin(uint64_t ticks) {
    const uint64_t begin = wall_clock64();
    while (wall_clock64() - begin < ticks) { __builtin_amdgcn_s_sleep(8); }
}

// The controller's hint protocol for one layer (stage_hint_locked), driven by hand: the target on the
// compute stream, the side stream after an event of the compute stream, conditional clears, copies,
// the publish, and the trailing event record that submits the side stream's work (without a submit
// point the queued side-stream commands can sit on the host while the compute stream waits for them).
static int stage_test() {
    constexpr int experts = 16, slots = 4, first = 2;   // arena slots 0, 1 static; 2..5 staging
    constexpr size_t slice = 4096;
    char * host = nullptr, * arena = nullptr;
    int32_t * table = nullptr, * ids = nullptr;
    uint32_t * out = nullptr;
    stage_state * state = nullptr;
    CUDA_CHECK(hipHostMalloc(&host, experts*slice));
    for (size_t i = 0; i < experts*slice; ++i) { host[i] = char((i*7 + i/slice*13) & 0xff); }
    CUDA_CHECK(hipMalloc(&arena, (first + slots)*slice));
    CUDA_CHECK(hipMemset(arena, 0, (first + slots)*slice));
    // expert 9 is a static VRAM resident in slot 0
    CUDA_CHECK(hipMemcpy(arena, host + 9*slice, slice, hipMemcpyHostToDevice));
    std::vector<int32_t> tab(experts, -1); tab[9] = 0;
    std::vector<int32_t> stat = tab;   // the static VRAM residents
    CUDA_CHECK(hipMalloc(&table, experts*sizeof(int32_t)));
    CUDA_CHECK(hipMemcpy(table, tab.data(), experts*sizeof(int32_t), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMalloc(&ids, 2*8*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&out, 2*8*sizeof(uint32_t)));
    int32_t * pinned_ids = nullptr; uint32_t * pinned_out = nullptr;   // two launches may be queued at once
    CUDA_CHECK(hipHostMalloc(&pinned_ids, 2*8*sizeof(int32_t)));
    CUDA_CHECK(hipHostMalloc(&pinned_out, 2*8*sizeof(uint32_t)));
    CUDA_CHECK(hipMalloc(&state, sizeof(stage_state)));
    CUDA_CHECK(hipMemset(state, 0, sizeof(stage_state)));
    CUDA_CHECK(hipDeviceSynchronize());   // the null-stream memsets before the non-blocking streams start
    hipStream_t cs, side;
    hipEvent_t ev_start, ev_done;
    CUDA_CHECK(hipStreamCreateWithFlags(&cs, hipStreamNonBlocking));
    CUDA_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
    CUDA_CHECK(hipEventCreateWithFlags(&ev_start, hipEventDisableTiming));
    CUDA_CHECK(hipEventCreateWithFlags(&ev_done, hipEventDisableTiming));
    auto sum_of = [&](int e) {
        uint32_t sum = 0;
        for (size_t j = 0; j < slice; j += 4) { uint32_t w; memcpy(&w, host + e*slice + j, 4); sum = sum*31u + w; }
        return sum;
    };
    std::vector<int32_t> owners(slots, -1);
    uint32_t gen = 0;
    double issue_us = 0.0;   // host time of the hints that issued work
    size_t issued = 0, issued_copies = 0;
    // `delay` spins the side stream first (a slow copy)
    auto hint = [&](const std::vector<int> & wanted_in, bool delay) {
        std::vector<int> wanted;
        for (int e : wanted_in) { if (stat[e] < 0) { wanted.push_back(e); } }   // host residents only
        const stage_plan plan = plan_stage(owners, wanted);
        if (!plan.changed()) { return plan; }
        const auto t0 = std::chrono::steady_clock::now();
        stage_targets t; t.n = 1; t.index[0] = 0; t.value[0] = ++gen;
        stage_target_kernel<<<1, stage_max_list, 0, cs>>>(state, t);
        CUDA_CHECK(hipEventRecord(ev_start, cs));
        CUDA_CHECK(hipStreamWaitEvent(side, ev_start, 0));
        if (delay) { stage_spin<<<1, 1, 0, side>>>(2000000); }
        stage_list clears, sets;
        for (const auto & c : plan.clears) { clears.expert[clears.n] = c.first; clears.slot[clears.n] = first + c.second; ++clears.n; }
        if (clears.n) { stage_clear_kernel<<<1, stage_max_list, 0, side>>>(table, clears); }
        for (const auto & c : plan.copies) {
            CUDA_CHECK(hipMemcpyAsync(arena + (first + c.second)*slice, host + c.first*slice, slice, hipMemcpyHostToDevice, side));
        }
        for (const auto & c : plan.keeps)  { sets.expert[sets.n] = c.first; sets.slot[sets.n] = first + c.second; ++sets.n; }
        for (const auto & c : plan.copies) { sets.expert[sets.n] = c.first; sets.slot[sets.n] = first + c.second; ++sets.n; }
        stage_publish_kernel<<<1, stage_max_list, 0, side>>>(table, sets, state, gen);
        CUDA_CHECK(hipEventRecord(ev_done, side));
        if (!delay) {
            issue_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            ++issued;
            issued_copies += plan.copies.size();
        }
        return plan;
    };
    // a "graph": the wait, then the reads of the ids; `buf` 0 or 1 (two may be queued); the sums land
    // in pinned_out[buf*8 + i] after the stream is synchronized
    auto launch = [&](const std::vector<int32_t> & sel, int buf) {
        memcpy(pinned_ids + buf*8, sel.data(), sel.size()*sizeof(int32_t));
        CUDA_CHECK(hipMemcpyAsync(ids + buf*8, pinned_ids + buf*8, sel.size()*sizeof(int32_t), hipMemcpyHostToDevice, cs));
        stage_wait_kernel<<<1, 64, 0, cs>>>(state, ids + buf*8, (int) sel.size(), 1, 1, experts, table, first, first + slots);
        stage_reader<<<1, 64, 0, cs>>>(table, ids + buf*8, (int) sel.size(), arena, host, slice, out + buf*8);
        CUDA_CHECK(hipMemcpyAsync(pinned_out + buf*8, out + buf*8, sel.size()*sizeof(uint32_t), hipMemcpyDeviceToHost, cs));
    };
    const uint32_t * r1 = pinned_out, * r2 = pinned_out + 8;
    // 1. two host experts staged, a static one untouched
    hint({3, 4, 9}, true);
    launch({3, 4, 9, 5}, 0);
    CUDA_CHECK(hipStreamSynchronize(cs));
    CHECK(r1[0] == sum_of(3) && r1[1] == sum_of(4) && r1[2] == sum_of(9) && r1[3] == sum_of(5));
    stage_state st;
    CUDA_CHECK(hipMemcpy(&st, state, sizeof(st), hipMemcpyDeviceToHost));
    CHECK(st.hits == 2 && st.host == 1 && st.launches == 1 && st.target == 1 && st.done == 1);
    CUDA_CHECK(hipMemcpy(tab.data(), table, experts*sizeof(int32_t), hipMemcpyDeviceToHost));
    CHECK(tab[3] == first + 0 && tab[4] == first + 1 && tab[9] == 0);
    // 2. a hint while the previous graph has not run yet: the graph sees its own staging, not the
    //    next one, and nothing deadlocks; the next graph sees the new one
    CHECK(hint({3, 4}, false).keeps.size() == 2);   // unchanged: nothing issued
    hint({5, 6, 7, 8}, false);
    launch({5, 6, 7, 8}, 0);
    hint({10, 11, 12, 13}, true);        // reuses every slot while the graph above may not have started
    launch({10, 11, 12, 13}, 1);
    CUDA_CHECK(hipStreamSynchronize(cs));
    CUDA_CHECK(hipStreamSynchronize(side));
    for (int i = 0; i < 4; ++i) { CHECK(r1[i] == sum_of(5 + i) && r2[i] == sum_of(10 + i)); }
    CUDA_CHECK(hipMemcpy(tab.data(), table, experts*sizeof(int32_t), hipMemcpyDeviceToHost));
    for (int e = 3; e <= 8; ++e) { CHECK(tab[e] == -1); }
    // 3. an entry an install rewrote is not cleared by a later reuse of its old staging slot
    CUDA_CHECK(hipMemcpy(arena + 1*slice, host + 10*slice, slice, hipMemcpyHostToDevice));
    tab.assign(experts, -1); tab[9] = 0; tab[10] = 1;   // the install made expert 10 static in slot 1
    stat = tab;
    CUDA_CHECK(hipMemcpy(table, tab.data(), experts*sizeof(int32_t), hipMemcpyHostToDevice));
    hint({2, 3, 4, 5}, false);           // every staging slot is reused, one of them held expert 10
    launch({10, 2, 3, 4, 5, 11}, 0);
    CUDA_CHECK(hipStreamSynchronize(cs));
    CHECK(r1[0] == sum_of(10) && r1[1] == sum_of(2) && r1[4] == sum_of(5) && r1[5] == sum_of(11));
    CUDA_CHECK(hipMemcpy(tab.data(), table, experts*sizeof(int32_t), hipMemcpyDeviceToHost));
    CHECK(tab[10] == 1 && tab[11] == -1 && tab[2] >= first);
    // 4. random hints, one launch each, sometimes two hints before a launch, sometimes a launch queued
    //    behind the previous one: every read matches
    uint32_t rng = 99;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng >> 8; };
    std::vector<int32_t> prev_sel;
    for (int round = 0; round < 300; ++round) {
        std::vector<int> want;
        std::vector<int32_t> sel;
        for (int i = 0; i < 6; ++i) { const int e = int(next()%experts); want.push_back(e); sel.push_back(e); }
        hint(want, next()%3 == 0);
        if (next()%4 == 0) {
            std::vector<int> other = {int(next()%experts), int(next()%experts)};
            hint(other, false);
        }
        launch(sel, round%2);
        const bool queued = !prev_sel.empty();
        if (!queued && next()%2 == 0) {
            prev_sel = sel;   // leave it queued; the next hint and launch go in behind it
            continue;
        }
        CUDA_CHECK(hipStreamSynchronize(cs));
        const uint32_t * r = pinned_out + (round%2)*8;
        for (size_t i = 0; i < sel.size(); ++i) { CHECK(r[i] == sum_of(sel[i])); }
        if (queued) {
            const uint32_t * q = pinned_out + ((round + 1)%2)*8;
            for (size_t i = 0; i < prev_sel.size(); ++i) { CHECK(q[i] == sum_of(prev_sel[i])); }
        }
        prev_sel.clear();
    }
    // 5. an install between a hint and the next ones (the model failure): the install drains the
    //    device, republishes the table from its plan (no staging entry) and the staging forgets its
    //    owners; a hint naming only some of the former owners then leaves a consistent table, and a
    //    graph queued after the hint reads the right bytes
    for (int round = 0; round < 50; ++round) {
        std::vector<int> want;
        for (int i = 0; i < 4; ++i) { want.push_back(int(next()%experts)); }
        hint(want, false);
        CUDA_CHECK(hipDeviceSynchronize());
        const int now_static = int(next()%experts);
        CUDA_CHECK(hipMemcpy(arena + 1*slice, host + now_static*slice, slice, hipMemcpyHostToDevice));
        tab.assign(experts, -1); tab[9] = 0;
        if (now_static != 9) { tab[now_static] = 1; }
        stat = tab;
        CUDA_CHECK(hipMemcpy(table, tab.data(), experts*sizeof(int32_t), hipMemcpyHostToDevice));
        stage_forget(owners);
        std::vector<int> subset = {want[0], int(next()%experts)};
        hint(subset, next()%2 == 0);
        std::vector<int32_t> sel = {want[0], want[1], want[2], subset[1], now_static, 9};
        launch(sel, 0);
        CUDA_CHECK(hipStreamSynchronize(cs));
        CUDA_CHECK(hipStreamSynchronize(side));
        for (size_t i = 0; i < sel.size(); ++i) { CHECK(pinned_out[i] == sum_of(sel[i])); }
        CUDA_CHECK(hipMemcpy(tab.data(), table, experts*sizeof(int32_t), hipMemcpyDeviceToHost));
        std::string why;
        if (!stage_table_check(owners, tab, first, why)) {
            fprintf(stderr, "round %d: %s; want %d %d %d %d subset %d %d static %d owners %d %d %d %d table", round, why.c_str(),
                want[0], want[1], want[2], want[3], subset[0], subset[1], now_static, owners[0], owners[1], owners[2], owners[3]);
            for (int e = 0; e < experts; ++e) { fprintf(stderr, " %d", tab[e]); }
            fprintf(stderr, "\n");
        }
        CHECK(stage_table_check(owners, tab, first, why));
        CHECK(now_static == 9 || tab[now_static] == 1);
    }
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemcpy(&st, state, sizeof(st), hipMemcpyDeviceToHost));
    CHECK(st.done == gen && st.target == gen);
    printf("PASS: VRAM staging: stream-ordered targets, conditional clears, reuse while a graph is queued, installs between hints "
           "(%llu staged reads, %llu host reads, %llu waits of %llu launches; host issue time %.1f us per hint "
           "with %.2f copies)\n", (unsigned long long) st.hits,
        (unsigned long long) st.host, (unsigned long long) st.waits, (unsigned long long) st.launches,
        issued ? issue_us/double(issued) : 0.0, issued ? double(issued_copies)/double(issued) : 0.0);
    CUDA_CHECK(hipEventDestroy(ev_start)); CUDA_CHECK(hipEventDestroy(ev_done));
    CUDA_CHECK(hipStreamDestroy(side)); CUDA_CHECK(hipStreamDestroy(cs));
    CUDA_CHECK(hipFree(state)); CUDA_CHECK(hipFree(out)); CUDA_CHECK(hipFree(ids)); CUDA_CHECK(hipFree(table));
    CUDA_CHECK(hipFree(arena)); CUDA_CHECK(hipHostFree(host));
    CUDA_CHECK(hipHostFree(pinned_ids)); CUDA_CHECK(hipHostFree(pinned_out));
    return 0;
}

// ---- size-class redraw on HIP virtual memory (expert-redraw.h, l1_arena::resize) ------------------
//
// Two size classes with odd slice sizes (slots straddle the 256 KiB handles) and staging slots per
// class with their own bytes. A sequence of redraws runs
// the controller's steps at the arena level (evacuation, the install with a retiring range, resize,
// the install in the new capacities) through the real mover. After every redraw: every VRAM
// resident (inclusive) or every expert through its one home (exclusive) equals its source, a graph
// captured before the first redraw replays and reads the same bytes through the same base
// addresses, the staging slots are untouched, and the assignment invariants hold. One
// redraw takes a class to zero static slots (its floor), and the classes swap capacity repeatedly.

static __global__ void redraw_gather(const char * arena, const int32_t * slots, const char * host, const int32_t * host_slots,
        size_t stride, char * out) {
    const int e = blockIdx.x;
    const int32_t s = slots[e];
    const char * src = s >= 0 ? arena + size_t(s)*stride : (host != nullptr && host_slots != nullptr && host_slots[e] >= 0 ?
        host + size_t(host_slots[e])*stride : nullptr);
    for (size_t i = threadIdx.x; i < stride; i += blockDim.x) { out[size_t(e)*stride + i] = src ? src[i] : char(0); }
}

static char redraw_byte(int layer, int kind, int expert, size_t i) {
    return char(((layer*37 + kind*11 + expert*5 + int(i % 241)) & 0x7f) | 1);
}

static int redraw_test(bool exclusive) {
    constexpr int layers = 3, experts = 12;
    geometry geo;
    geo.n_layers = layers; geo.n_experts = experts;
    geo.layer_class = {0, 0, 1}; geo.class_layers = {2, 1};
    geo.class_bytes = {{70*1024 + 64, 70*1024 + 64, 90*1024 + 128}, {50*1024, 50*1024, 110*1024 + 256}};
    geo.nb2.resize(layers); geo.tensors.resize(layers);
    ggml_tensor tensors[layers][3] = {};
    std::vector<char> values[layers][3];
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
        values[l][k].resize(experts*stride);
        for (int e = 0; e < experts; ++e) for (size_t i = 0; i < stride; ++i) { values[l][k][e*stride + i] = redraw_byte(l, k, e, i); }
        auto & t = tensors[l][k]; t.type = GGML_TYPE_F32; t.ne[0] = 32; t.ne[1] = 1; t.ne[2] = experts; t.ne[3] = 1;
        t.data = values[l][k].data(); geo.tensors[l][k] = &t; geo.nb2[l][k] = stride;
    }
    const int spares = exclusive ? 2 : 0;
    const std::vector<int> stage = {2, 2};
    host_arena host(geo);
    l1_arena gpu(geo);
    gpu.enable_vmm({24, 12}, 256*1024);
    CHECK(gpu.allocate({10, 6}, 0, spares, &stage));
    if (!gpu.vmm()) { printf("FAIL: no VMM arena: %s\n", gpu.vmm_reason().c_str()); return 1; }
    auto pattern = [](int cls, int kind, int slot, size_t i) { return char(0x80 | ((cls*29 + kind*3 + slot*7 + int(i % 199)) & 0x7f)); };
    auto fill_slot = [&](int cls, int slot) {
        for (int k = 0; k < 3; ++k) {
            std::vector<char> b(geo.class_bytes[cls][k]);
            for (size_t i = 0; i < b.size(); ++i) { b[i] = pattern(cls, k, slot, i); }
            if (!gpu.write_gpu_slice(cls, k, slot, b.data(), false) || !gpu.sync_copies()) { return false; }
        }
        return true;
    };
    auto slot_intact = [&](int cls, int slot) {
        for (int k = 0; k < 3; ++k) {
            std::vector<char> b(geo.class_bytes[cls][k]);
            if (!gpu.read_gpu_slice(cls, k, slot, b.data()) || !gpu.sync_copies()) { return false; }
            for (size_t i = 0; i < b.size(); ++i) { if (b[i] != pattern(cls, k, slot, i)) { return false; } }
        }
        return true;
    };
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < stage[c]; ++i) { CHECK(fill_slot(c, gpu.stage_base(c) + i)); }
    }
    // the initial plan
    expert_slot_table sel = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};
    auto complement = [&](const expert_slot_table & s) {
        expert_slot_table out(layers);
        for (int l = 0; l < layers; ++l) for (int e = 0; e < experts; ++e) {
            if (!std::binary_search(s[l].begin(), s[l].end(), e)) { out[l].push_back(e); }
        }
        return out;
    };
    const std::vector<int> host_caps = {2*experts, experts};
    if (exclusive) {
        CHECK(host.allocate(host_caps, 0, 0));
        gpu.attach_host(&host, spares);
        expert_slot_table g(layers, std::vector<int32_t>(experts, -1)), h = g;
        std::vector<int> ng = {0, 0}, nh = {0, 0};
        for (int l = 0; l < layers; ++l) {
            const int c = geo.layer_class[l];
            for (int e = 0; e < experts; ++e) {
                if (std::binary_search(sel[l].begin(), sel[l].end(), e)) { g[l][e] = ng[c]++; } else { h[l][e] = nh[c]++; }
            }
        }
        install_layout lay; lay.gpu = gpu.slot_counts(); lay.host = host_caps; lay.lent_begin = {0, 0}; lay.lent_count = {0, 0};
        CHECK(gpu.assign_tier(g, host_locations(h), lay, sel, {{10, 11}, {6, 7}}));
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
            CHECK(gpu.logical_io(l, k, values[l][k].data(), 0, values[l][k].size(), true));
        }
        CHECK(host.map());
    } else {
        l1_install_stats st;
        CHECK(gpu.install(sel, true, st));
    }
    // one graph over every (layer, kind), captured before any redraw
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    size_t out_bytes = 0;
    std::vector<size_t> out_off;
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        out_off.push_back(out_bytes); out_bytes += experts*geo.class_bytes[geo.layer_class[l]][k];
    }
    char * out = nullptr; CUDA_CHECK(hipMalloc(&out, out_bytes));
    hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        const ggml_cuda_expert_lookup lk = gpu.lookup(l, k);
        redraw_gather<<<experts, 256, 0, stream>>>((const char *) lk.data, lk.slots, (const char *) lk.host_data, lk.host_slots,
            geo.class_bytes[geo.layer_class[l]][k], out + out_off[l*3 + k]);
    }
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::vector<char> back(out_bytes);
    auto check_all = [&](const char * when) {
        std::string reason;
        if (!gpu.verify_current_assignment(reason)) { printf("FAIL %s: assignment: %s\n", when, reason.c_str()); return false; }
        CUDA_CHECK(hipGraphLaunch(exec, stream)); CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(back.data(), out, out_bytes, hipMemcpyDeviceToHost));
        size_t resident = 0;
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
            const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
            for (int e = 0; e < experts; ++e) {
                const char * got = back.data() + out_off[l*3 + k] + e*stride;
                const bool in_vram = gpu.host_slots()[l][e] >= 0;
                resident += in_vram && k == 0;
                if (in_vram || exclusive) {
                    if (memcmp(got, values[l][k].data() + e*stride, stride) != 0) {
                        printf("FAIL %s: layer %d kind %d expert %d (slot %d) differs from its source\n", when, l, k, e, gpu.host_slots()[l][e]);
                        return false;
                    }
                } else {
                    for (size_t i = 0; i < stride; ++i) { if (got[i] != 0) { printf("FAIL %s: miss not zero\n", when); return false; } }
                }
            }
            if (exclusive) {
                std::vector<char> io(values[l][k].size());
                if (!gpu.logical_io(l, k, io.data(), 0, io.size(), false) || io != values[l][k]) {
                    printf("FAIL %s: logical read of layer %d kind %d\n", when, l, k); return false;
                }
            }
        }
        size_t want = 0;
        for (const auto & v : sel) { want += v.size(); }
        if (resident != want) { printf("FAIL %s: %zu VRAM residents, the plan has %zu\n", when, resident, want); return false; }
        if (!exclusive && gpu.verify_resident(0) != long(want)) { printf("FAIL %s: verify_resident\n", when); return false; }
        for (int c = 0; c < 2; ++c) {
            for (int i = 0; i < stage[c]; ++i) {
                if (!slot_intact(c, gpu.stage_base(c) + i)) { printf("FAIL %s: staging slot %d of class %d\n", when, i, c); return false; }
            }
            // the zeroed tail past the top slot, where MMQ reads past the last row
            for (int k = 0; k < 3; ++k) {
                std::vector<char> tail(arena_tail_bytes(geo, c, k), char(1));
                CUDA_CHECK(hipMemcpy(tail.data(), gpu.gpu_slice_address(c, k, gpu.slot_counts()[c]), tail.size(), hipMemcpyDeviceToHost));
                for (char x : tail) { if (x != 0) { printf("FAIL %s: tail of class %d kind %d not zero\n", when, c, k); return false; } }
            }
        }
        return true;
    };
    CHECK(check_all("before the first redraw"));
    const size_t mapped0 = gpu.device_bytes();
    uint64_t rng = exclusive ? 99 : 7;
    auto rnd = [&](int n) { rng = rng*6364136223846793005ULL + 1442695040888963407ULL; return int((rng >> 33) % uint64_t(n)); };
    // a selection of `count` experts per class that keeps a random part of the current one
    auto choose = [&](int cls, int count) {
        std::vector<int> ls;
        for (int l = 0; l < layers; ++l) { if (geo.layer_class[l] == cls) { ls.push_back(l); } }
        expert_slot_table part(layers);
        for (size_t i = 0; i < ls.size(); ++i) {
            const int n = count/int(ls.size()) + (int(i) < count % int(ls.size()) ? 1 : 0);
            std::vector<int> order(experts);
            std::iota(order.begin(), order.end(), 0);
            for (int j = experts - 1; j > 0; --j) { std::swap(order[j], order[rnd(j + 1)]); }
            // current residents first half of the time, so that some kept ones sit high
            std::stable_partition(order.begin(), order.end(), [&](int e) {
                return rnd(2) == 0 && std::binary_search(sel[ls[i]].begin(), sel[ls[i]].end(), e); });
            part[ls[i]].assign(order.begin(), order.begin() + n);
            std::sort(part[ls[i]].begin(), part[ls[i]].end());
        }
        return part;
    };
    const std::vector<std::vector<int>> splits = {{6, 10}, {12, 0}, {8, 8}, {4, 12}, {16, 2}, {10, 6}, {3, 11}, {14, 4}};
    size_t swaps = 0, moved = 0, created = 0, replaced = 0;
    for (size_t r = 0; r < splits.size(); ++r) {
        const std::vector<int> caps_old = gpu.capacities(), slots_old = gpu.slot_counts(), caps_new = splits[r];
        std::vector<int> caps1(2), limit(2), slots_new(2);
        for (int c = 0; c < 2; ++c) {
            caps1[c] = std::min(caps_old[c], caps_new[c]);
            limit[c] = caps1[c] + slots_old[c] - caps_old[c];
            slots_new[c] = caps_new[c] + slots_old[c] - caps_old[c];
        }
        expert_slot_table next(layers);
        for (int c = 0; c < 2; ++c) {
            const expert_slot_table part = choose(c, caps_new[c]);
            for (int l = 0; l < layers; ++l) { if (geo.layer_class[l] == c) { next[l] = part[l]; } }
        }
        expert_slot_table step1 = sel;
        for (int l = 0; l < layers; ++l) { if (caps_new[geo.layer_class[l]] < caps_old[geo.layer_class[l]]) { step1[l] = next[l]; } }
        // 1. evacuation
        std::vector<std::vector<uint8_t>> other(2);
        for (int c = 0; c < 2; ++c) {
            other[c].assign(slots_old[c], 0);
            if (exclusive) { for (int s : gpu.gpu_spares()[c]) { other[c][s] = 1; } }
        }
        const evac_plan ev = plan_evacuation(geo.layer_class, experts, gpu.host_slots(), step1, other, slots_old, limit);
        if (!ev.valid) { printf("FAIL redraw %zu: %s\n", r, ev.reason.c_str()); return 1; }
        double ms = 0.0;
        CHECK(gpu.evacuate(ev.ops, ms));
        for (const auto & op : ev.ops) { swaps += op.victim_layer >= 0; }
        // 2. the install with the retiring range
        gpu.set_static_capacities(caps1);
        gpu.set_retire(limit);
        gpu.set_unequal(exclusive);
        auto install = [&](const expert_slot_table & s, const std::vector<int> & slots_after) {
            if (exclusive) {
                install_layout before = gpu.layout(), after = before;
                after.gpu = slots_after;
                const install_transaction tx = plan_install(geo, s, complement(s), gpu.host_slots(), gpu.locations(),
                    gpu.capacities(), before, after, gpu.gpu_spares(), {false, spares}, false, true, /*unequal =*/ true);
                if (!tx.valid) { printf("FAIL redraw %zu: %s\n", r, tx.reason.c_str()); return false; }
                return gpu.execute(tx) && gpu.assign_tier(tx.gpu_slots, tx.host, after, s, tx.gpu_spares);
            }
            GGML_UNUSED(slots_after);
            l1_install_stats st;
            return gpu.install(s, true, st);
        };
        CHECK(install(step1, limit));
        gpu.set_retire({});
        sel = step1;
        CHECK(check_all("after the shrink install"));
        // 3. the handles
        l1_arena::resize_stats rs;
        std::string why;
        if (!gpu.resize(slots_new, rs, why)) { printf("FAIL redraw %zu: resize: %s\n", r, why.c_str()); return 1; }
        moved += rs.handles_moved; created += rs.handles_created; replaced += rs.units_replaced;
        gpu.set_static_capacities(caps_new);
        CHECK(gpu.slot_counts() == slots_new);
        CHECK(check_all("after the resize"));
        // 4. the install in the new capacities
        CHECK(install(next, slots_new));
        gpu.set_unequal(false);
        sel = next;
        CHECK(check_all("after the redraw"));
    }
    // the backed bytes track the split within a granule per (class, kind)
    CHECK(gpu.device_bytes() != 0 && mapped0 != 0);
    CUDA_CHECK(hipGraphExecDestroy(exec)); CUDA_CHECK(hipGraphDestroy(graph));
    CUDA_CHECK(hipFree(out)); CUDA_CHECK(hipStreamDestroy(stream));
    printf("PASS: %s L1 redraw on VMM: %zu redraws (one to a zero static class), %zu swaps, %zu handles moved, %zu created, "
           "%zu partial units replaced; every slice equals its source, the graph captured before replays across them, "
           "staging slots intact\n", exclusive ? "exclusive" : "inclusive", splits.size(), swaps, moved, created, replaced);
    return 0;
}

// ---- host arenas without a finite tier (expert-host-layout.h) -------------------------------------
//
// Exclusive mode with an unlimited host tier. host_table_test: the address table on the contiguous
// arenas publishes host_data + slot * stride for every host resident, 0 for VRAM residents, follows
// an install, and the kernels' select reads the same bytes with and without it. host_redraw_test:
// chunked host arenas (small chunks, so slots straddle several chunks per class) under a sequence of
// VRAM size-class redraws in both directions, run as the controller does (host growth, evacuation,
// the shrink install, resize, the grow install, host compaction) through the real l1_arena::install.
// After every step every expert equals its source through its one home (graph captured before the
// first redraw, reading through ggml_cuda_expert_cache_select), the logical reads agree, the host
// capacities and chunk counts follow the split. host_chunk_cost_test times chunk growth and release
// at a Qwen class-0 slice geometry with 128 MiB chunks.

static __global__ void select_gather(const char * tensor, const char * arena, const int32_t * slots, const int32_t * host_slots,
        const uint64_t * host_addresses, size_t stride, char * out) {
    const int e = blockIdx.x;
    const ggml_cuda_expert_source src = ggml_cuda_expert_cache_select(tensor, arena, slots, (uint32_t) e, host_slots, host_addresses);
    const char * p = (const char *) src.data + size_t(src.channel)*stride;
    for (size_t i = threadIdx.x; i < stride; i += blockDim.x) { out[size_t(e)*stride + i] = p[i]; }
}

struct host_fixture_geo {
    static constexpr int layers = 3, experts = 12;
    geometry geo;
    ggml_tensor tensors[layers][3] = {};
    std::vector<char> values[layers][3];
    host_fixture_geo() {
        geo.n_layers = layers; geo.n_experts = experts;
        geo.layer_class = {0, 0, 1}; geo.class_layers = {2, 1};
        geo.class_bytes = {{70*1024 + 64, 70*1024 + 64, 90*1024 + 128}, {50*1024, 50*1024, 110*1024 + 256}};
        geo.nb2.resize(layers); geo.tensors.resize(layers);
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
            const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
            values[l][k].resize(experts*stride);
            for (int e = 0; e < experts; ++e) for (size_t i = 0; i < stride; ++i) { values[l][k][e*stride + i] = redraw_byte(l, k, e, i); }
            auto & t = tensors[l][k]; t.type = GGML_TYPE_F32; t.ne[0] = 32; t.ne[1] = 1; t.ne[2] = experts; t.ne[3] = 1;
            t.data = values[l][k].data(); geo.tensors[l][k] = &t; geo.nb2[l][k] = stride;
        }
    }
};

// Every expert through the select rule, with the lookup's pointers (or with the address table
// dropped), compared with its source.
static bool select_all_equal(const host_fixture_geo & f, l1_arena & gpu, bool use_table, const char * when) {
    const geometry & geo = f.geo;
    for (int l = 0; l < f.layers; ++l) for (int k = 0; k < 3; ++k) {
        const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
        const ggml_cuda_expert_lookup lk = gpu.lookup(l, k);
        char * out = nullptr;
        CUDA_CHECK(hipMalloc(&out, f.experts*stride));
        select_gather<<<f.experts, 256>>>((const char *) lk.host_data, (const char *) lk.data, lk.slots, lk.host_slots,
            use_table ? lk.host_addresses : nullptr, stride, out);
        std::vector<char> back(f.experts*stride);
        CUDA_CHECK(hipMemcpy(back.data(), out, back.size(), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipFree(out));
        if (back != f.values[l][k]) { printf("FAIL %s: layer %d kind %d through the select (table %d)\n", when, l, k, use_table); return false; }
    }
    return true;
}

static int host_table_test() {
    host_fixture_geo f;
    const geometry & geo = f.geo;
    constexpr int E = host_fixture_geo::experts;
    const int spares = 2;
    const std::vector<int> caps = {10, 6}, host_caps = {2*E - 10, E - 6};
    for (int table = 0; table < 2; ++table) {
        host_arena host(geo);
        if (table) { host.enable_addresses(); }
        l1_arena gpu(geo);
        CHECK(gpu.allocate(caps, 0, spares));
        CHECK(host.allocate(host_caps, spares, 0));
        gpu.attach_host(&host, spares);
        expert_slot_table sel = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};
        CHECK(gpu.assign_exclusive(sel));
        for (int l = 0; l < f.layers; ++l) for (int k = 0; k < 3; ++k) {
            CHECK(gpu.logical_io(l, k, f.values[l][k].data(), 0, f.values[l][k].size(), true));
        }
        CHECK(host.map());
        auto check_table = [&](const char * when) {
            for (int l = 0; l < f.layers; ++l) for (int k = 0; k < 3; ++k) {
                const ggml_cuda_expert_lookup lk = gpu.lookup(l, k);
                if (!table) {
                    if (lk.host_addresses != nullptr) { printf("FAIL %s: a table without the switch\n", when); return false; }
                    continue;
                }
                if (lk.host_addresses == nullptr) { printf("FAIL %s: no table\n", when); return false; }
                std::vector<uint64_t> t(E);
                CUDA_CHECK(hipMemcpy(t.data(), lk.host_addresses, E*sizeof(uint64_t), hipMemcpyDeviceToHost));
                const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
                for (int e = 0; e < E; ++e) {
                    const int hs = gpu.arena_slots()[l][e];
                    const uint64_t want = hs >= 0 ? (uint64_t) (uintptr_t) lk.host_data + uint64_t(hs)*stride : 0;
                    if (t[e] != want || (hs >= 0) == (gpu.host_slots()[l][e] >= 0)) {
                        printf("FAIL %s: layer %d kind %d expert %d: table %llx, want %llx\n", when, l, k, e,
                            (unsigned long long) t[e], (unsigned long long) want);
                        return false;
                    }
                }
            }
            return select_all_equal(f, gpu, true, when) && select_all_equal(f, gpu, false, when);
        };
        CHECK(check_table("after the load"));
        // an ordinary exclusive install (spare rotation): the table follows the host slots
        for (int round = 0; round < 3; ++round) {
            sel = round == 0 ? expert_slot_table{{3, 5, 7, 9, 11}, {0, 2, 4, 6, 8}, {1, 3, 5, 7, 9, 11}} :
                  round == 1 ? expert_slot_table{{0, 1, 2, 10, 11}, {5, 6, 7, 8, 9}, {0, 2, 4, 6, 8, 10}} :
                               expert_slot_table{{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};
            l1_install_stats st;
            CHECK(gpu.install(sel, true, st));
            CHECK(check_table("after an install"));
        }
    }
    printf("PASS: host address table (exclusive, unlimited host): off = no table; on = host_data + slot x stride per host "
           "resident, 0 in VRAM, follows installs; the select reads the same bytes with and without it\n");
    return 0;
}

static int host_redraw_test() {
    host_fixture_geo f;
    const geometry & geo = f.geo;
    constexpr int layers = host_fixture_geo::layers, E = host_fixture_geo::experts;
    const int spares = 2;
    const std::vector<int> stage = {2, 2};
    // 300 KiB chunks: 3 slots per chunk in class 0 (largest kind 90 KiB), 2 in class 1 (110 KiB)
    const size_t chunk = 300*1024;
    host_arena host(geo);
    host.enable_addresses();
    host.enable_chunks(chunk);
    l1_arena gpu(geo);
    gpu.enable_vmm({24, 12}, 256*1024);
    std::vector<int> caps = {10, 6};
    CHECK(gpu.allocate(caps, 0, spares, &stage));
    if (!gpu.vmm()) { printf("FAIL: no VMM arena: %s\n", gpu.vmm_reason().c_str()); return 1; }
    auto host_caps_of = [&](const std::vector<int> & c) { return std::vector<int>{2*E - c[0], E - c[1]}; };
    CHECK(host.allocate(host_caps_of(caps), spares, 0));
    CHECK(host.chunked() && host.chunk_slots(0) == 3 && host.chunk_slots(1) == 2);
    gpu.attach_host(&host, spares);
    expert_slot_table sel = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};
    CHECK(gpu.assign_exclusive(sel));
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        CHECK(gpu.logical_io(l, k, f.values[l][k].data(), 0, f.values[l][k].size(), true));
    }
    CHECK(host.map());
    // one graph over every (layer, kind), captured before any redraw, through the kernels' select
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    size_t out_bytes = 0;
    std::vector<size_t> out_off;
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        out_off.push_back(out_bytes); out_bytes += E*geo.class_bytes[geo.layer_class[l]][k];
    }
    char * out = nullptr; CUDA_CHECK(hipMalloc(&out, out_bytes));
    hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
        const ggml_cuda_expert_lookup lk = gpu.lookup(l, k);
        select_gather<<<E, 256, 0, stream>>>((const char *) lk.host_data, (const char *) lk.data, lk.slots, lk.host_slots,
            lk.host_addresses, geo.class_bytes[geo.layer_class[l]][k], out + out_off[l*3 + k]);
    }
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    std::vector<char> back(out_bytes);
    auto check_all = [&](const char * when) {
        std::string reason;
        if (!gpu.verify_current_assignment(reason)) { printf("FAIL %s: assignment: %s\n", when, reason.c_str()); return false; }
        CUDA_CHECK(hipGraphLaunch(exec, stream)); CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(back.data(), out, out_bytes, hipMemcpyDeviceToHost));
        for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
            const size_t stride = geo.class_bytes[geo.layer_class[l]][k];
            for (int e = 0; e < E; ++e) {
                const bool vram = gpu.host_slots()[l][e] >= 0, home = gpu.arena_slots()[l][e] >= 0;
                if (vram == home) { printf("FAIL %s: layer %d expert %d has %d homes\n", when, l, e, vram ? 2 : 0); return false; }
                if (memcmp(back.data() + out_off[l*3 + k] + e*stride, f.values[l][k].data() + e*stride, stride) != 0) {
                    printf("FAIL %s: layer %d kind %d expert %d (VRAM slot %d, host slot %d) differs from its source\n", when, l, k, e,
                        gpu.host_slots()[l][e], gpu.arena_slots()[l][e]);
                    return false;
                }
            }
            std::vector<char> io(f.values[l][k].size());
            if (!gpu.logical_io(l, k, io.data(), 0, io.size(), false) || io != f.values[l][k]) {
                printf("FAIL %s: logical read of layer %d kind %d\n", when, l, k); return false;
            }
        }
        size_t chunks = 0;
        for (int c = 0; c < 2; ++c) { chunks += 3*size_t(host_chunks_for(host.capacities()[c] + spares, host.chunk_slots(c))); }
        if (chunks != host.chunk_count()) { printf("FAIL %s: %zu chunks, want %zu\n", when, host.chunk_count(), chunks); return false; }
        return select_all_equal(f, gpu, true, when);
    };
    CHECK(check_all("before the first redraw"));
    uint64_t rng = 4242;
    auto rnd = [&](int n) { rng = rng*6364136223846793005ULL + 1442695040888963407ULL; return int((rng >> 33) % uint64_t(n)); };
    auto choose = [&](int cls, int count) {
        std::vector<int> ls;
        for (int l = 0; l < layers; ++l) { if (geo.layer_class[l] == cls) { ls.push_back(l); } }
        expert_slot_table part(layers);
        for (size_t i = 0; i < ls.size(); ++i) {
            const int n = count/int(ls.size()) + (int(i) < count % int(ls.size()) ? 1 : 0);
            std::vector<int> order(E);
            std::iota(order.begin(), order.end(), 0);
            for (int j = E - 1; j > 0; --j) { std::swap(order[j], order[rnd(j + 1)]); }
            std::stable_partition(order.begin(), order.end(), [&](int e) {
                return rnd(2) == 0 && std::binary_search(sel[ls[i]].begin(), sel[ls[i]].end(), e); });
            part[ls[i]].assign(order.begin(), order.begin() + n);
            std::sort(part[ls[i]].begin(), part[ls[i]].end());
        }
        return part;
    };
    const std::vector<std::vector<int>> splits = {{6, 10}, {16, 2}, {8, 8}, {3, 12}, {20, 0}, {10, 6}, {4, 11}, {14, 4}};
    size_t moved = 0, added = 0, released = 0;
    for (size_t r = 0; r < splits.size(); ++r) {
        const std::vector<int> caps_old = gpu.capacities(), slots_old = gpu.slot_counts(), caps_new = splits[r];
        std::vector<int> caps1(2), limit(2), slots_new(2), host_up(2);
        const std::vector<int> host_final = host_caps_of(caps_new);
        for (int c = 0; c < 2; ++c) {
            caps1[c] = std::min(caps_old[c], caps_new[c]);
            limit[c] = caps1[c] + slots_old[c] - caps_old[c];
            slots_new[c] = caps_new[c] + slots_old[c] - caps_old[c];
            host_up[c] = std::max(host.capacities()[c], host_final[c]);
        }
        expert_slot_table next(layers);
        for (int c = 0; c < 2; ++c) {
            const expert_slot_table part = choose(c, caps_new[c]);
            for (int l = 0; l < layers; ++l) { if (geo.layer_class[l] == c) { next[l] = part[l]; } }
        }
        expert_slot_table step1 = sel;
        for (int l = 0; l < layers; ++l) { if (caps_new[geo.layer_class[l]] < caps_old[geo.layer_class[l]]) { step1[l] = next[l]; } }
        // 0. host growth of the shrinking classes
        l1_arena::host_resize_stats hs;
        std::string why;
        if (!gpu.resize_host(host_up, hs, why)) { printf("FAIL redraw %zu: host growth: %s\n", r, why.c_str()); return 1; }
        added += hs.chunks_added;
        CHECK(check_all("after the host growth"));
        // 1. evacuation
        std::vector<std::vector<uint8_t>> other(2);
        for (int c = 0; c < 2; ++c) {
            other[c].assign(slots_old[c], 0);
            for (int s : gpu.gpu_spares()[c]) { other[c][s] = 1; }
        }
        const evac_plan ev = plan_evacuation(geo.layer_class, E, gpu.host_slots(), step1, other, slots_old, limit);
        if (!ev.valid) { printf("FAIL redraw %zu: %s\n", r, ev.reason.c_str()); return 1; }
        double ms = 0.0;
        CHECK(gpu.evacuate(ev.ops, ms));
        // 2. the shrink install (the controller's install_plain path: unequal exchange, retiring range)
        gpu.set_static_capacities(caps1);
        gpu.set_retire(limit);
        gpu.set_unequal(true);
        l1_install_stats st;
        if (!gpu.install(step1, true, st)) { printf("FAIL redraw %zu: shrink install\n", r); return 1; }
        gpu.set_retire({});
        sel = step1;
        CHECK(check_all("after the shrink install"));
        // 3. the handles
        l1_arena::resize_stats rs;
        if (!gpu.resize(slots_new, rs, why)) { printf("FAIL redraw %zu: resize: %s\n", r, why.c_str()); return 1; }
        gpu.set_static_capacities(caps_new);
        CHECK(check_all("after the resize"));
        // 4. the grow install
        if (!gpu.install(next, true, st)) { printf("FAIL redraw %zu: grow install\n", r); return 1; }
        gpu.set_unequal(false);
        sel = next;
        CHECK(check_all("after the grow install"));
        // 5. host compaction and chunk release of the growing classes
        if (!gpu.resize_host(host_final, hs, why)) { printf("FAIL redraw %zu: host shrink: %s\n", r, why.c_str()); return 1; }
        moved += hs.moved; released += hs.chunks_released;
        CHECK(host.capacities() == host_final);
        CHECK(check_all("after the host compaction"));
    }
    CHECK(moved > 0 && added > 0 && released > 0);
    // an ordinary install after the redraws still rotates through the spares
    l1_install_stats st;
    expert_slot_table again(layers);
    for (int c = 0; c < 2; ++c) {
        const expert_slot_table part = choose(c, gpu.capacities()[c]);
        for (int l = 0; l < layers; ++l) { if (geo.layer_class[l] == c) { again[l] = part[l]; } }
    }
    CHECK(gpu.install(again, true, st));
    CHECK(check_all("after a plain install"));
    CUDA_CHECK(hipGraphExecDestroy(exec)); CUDA_CHECK(hipGraphDestroy(graph));
    CUDA_CHECK(hipFree(out)); CUDA_CHECK(hipStreamDestroy(stream));
    printf("PASS: exclusive L1 redraw with chunked host arenas (no finite tier): %zu redraws both ways (one to a zero static "
           "class), %zu chunks added, %zu released, %zu host residents compacted; every expert equals its source through its "
           "one home after every step, the graph captured before replays across them, slack %zu bytes\n",
        splits.size(), added, released, moved, host.slack_bytes());
    return 0;
}

static int host_chunk_cost_test() {
    // Qwen class 0 slices (900/900/1200 KiB), 128 MiB chunks: 109 slots per chunk
    geometry geo;
    geo.n_layers = 1; geo.n_experts = 512; geo.layer_class = {0}; geo.class_layers = {1};
    geo.class_bytes = {{900*1024, 900*1024, 1200*1024}};
    geo.nb2.resize(1); geo.tensors.resize(1);
    ggml_tensor t[3] = {};
    for (int k = 0; k < 3; ++k) {
        t[k].type = GGML_TYPE_F32; t[k].ne[0] = 32; t[k].ne[1] = 1; t[k].ne[2] = 512; t[k].ne[3] = 1;
        geo.tensors[0][k] = &t[k]; geo.nb2[0][k] = geo.class_bytes[0][k];
    }
    host_arena host(geo);
    host.enable_addresses();
    host.enable_chunks(size_t(128) << 20);
    CHECK(host.allocate({0}, 0, 0));
    CHECK(host.map());
    const int per = host.chunk_slots(0);
    host_arena::resize_stats grow, shrink;
    std::string why;
    CHECK(host.set_capacity(0, 4*per, grow, why));        // 1 -> 4 chunks per kind
    CHECK(host.chunk_count() == 12 && grow.chunks_added == 9);
    CHECK(host.set_capacity(0, 0, shrink, why));
    CHECK(host.chunk_count() == 3 && shrink.chunks_released == 9);
    printf("PASS: host chunk cost (128 MiB, %d slots of 900/900/1200 KiB): %zu chunks (%.1f MiB) added in %.1f ms = %.2f ms per "
           "chunk (commit, zero, register), released in %.1f ms\n", per, grow.chunks_added,
        double(grow.bytes_added)/double(1 << 20), grow.ms, grow.ms/double(grow.chunks_added), shrink.ms);
    return 0;
}

// ---- batched service end to end: worker, file reads, batch launches of MMQ_ID --------------------------
//
// A class-layout tier over a file of Q8_0 experts, a third of them host residents. Each prompt ubatch
// runs through publish_and_wait, the kind waits, the first launch over the experts that need no read
// and the batch launches with their waits and done reports, with the worker reading the batches. The
// output must be bit-identical to one MMQ_ID launch over a contiguous copy of all experts. Ring sizes
// from the smallest (two slots, batches of one) to one that needs a single batch; with RANMA's
// verification on, every read and the address table are checked after each service.

static void batch_mmq_launch(hipStream_t stream, const char * x, const int * y, const int32_t * ids_dst, const int32_t * bounds,
        float * dst, const char * x_cache, const int32_t * slots, const uint64_t * addresses, const int32_t * map, int key,
        int channels, int K, int M, int cols, int tokens, size_t channel_bytes) {
    constexpr int J = 16;
    const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, false, cc);
    const int ntx = (tokens + J - 1)/J, nty = (M + config.I - 1)/config.I;
    const dim3 grid(nty, ntx, channels), block(32, config.nthreads/32, 1);
    const auto one = init_fastdiv_values(1);
    mul_mat_q<GGML_TYPE_Q8_0, J, false><<<grid, block, mmq_get_nbytes_shared(config, cc), stream>>>(
        x, y, ids_dst, bounds, dst, nullptr, nullptr, x_cache, slots, slots, addresses, map, key,
        init_fastdiv_values(K/32), M, cols, K/32, cols, M, one, init_fastdiv_values(channels), (int64_t) channel_bytes, 0, 0,
        one, one, 0, 0, 0, init_fastdiv_values(ntx));
    CUDA_CHECK(hipGetLastError());
}

static int batched_service_test(const std::filesystem::path & dir) {
    constexpr int K = 512, M = 256, experts = 24, used = 4, tokens = 32, layers = 2, J = 16;
    constexpr size_t block = 34, slice = size_t(K/32)*block*M, shift = 4064;
    static_assert(slice % 4096 == 0, "a slice must be whole sectors");
    const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
    if (ggml_cuda_mmq_get_stream_k(GGML_TYPE_Q8_0, J, false, cc)) { printf("SKIP: batched service test needs a tiling MMQ config\n"); return 0; }
    // the file: layer, kind, expert; Q8_0 blocks with a small scale and random quants
    std::vector<char> file(shift + size_t(layers*3*experts)*slice);
    uint32_t rng = 77;
    auto next = [&]() { rng = rng*1664525u + 1013904223u; return rng; };
    for (size_t b = 0; b < (file.size() - shift)/block; ++b) {
        char * p = file.data() + shift + b*block;
        const uint16_t d = 0x2000 + uint16_t(next() % 64);   // about 0.008
        memcpy(p, &d, 2);
        for (size_t i = 2; i < block; ++i) { p[i] = char(next() % 31) - 15; }
    }
    const auto path = dir / "batch-weights.bin";
    { std::ofstream out(path, std::ios::binary); out.write(file.data(), (std::streamsize) file.size()); CHECK(bool(out)); }
    auto file_slice = [&](int l, int k, int e) { return file.data() + shift + (size_t((l*3 + k)*experts + e))*slice; };

    ggml_tensor weights = {};
    weights.type = GGML_TYPE_Q8_0; weights.ne[0] = K; weights.ne[1] = M; weights.ne[2] = experts; weights.ne[3] = 1;
    geometry geo;
    geo.n_layers = layers; geo.n_experts = experts;
    geo.layer_class = {0, 0}; geo.class_layers = {layers};
    geo.class_bytes = {{slice, slice, slice}};
    geo.tensors = {{{&weights, &weights, &weights}}, {{&weights, &weights, &weights}}};

    // y (activations in the MMQ layout, finite scales), ids, and the MUL_MAT_ID bookkeeping on the host
    const int cols = tokens*used;
    const size_t y_bytes = (size_t(cols)*(K/128) + J)*sizeof(block_q8_1_mmq);
    std::vector<char> y_host(y_bytes);
    for (size_t b = 0; b < y_bytes/sizeof(block_q8_1_mmq); ++b) {
        char * p = y_host.data() + b*sizeof(block_q8_1_mmq);
        for (int w = 0; w < 4; ++w) { const uint32_t v = 0x3C003C00u; memcpy(p + 4*w, &v, 4); }
        for (int i = 16; i < (int) sizeof(block_q8_1_mmq); ++i) { p[i] = char(next() % 21) - 10; }
    }
    int * y = nullptr; float * dst = nullptr, * ref = nullptr; int32_t * dev_ids = nullptr, * ids_dst = nullptr, * bounds = nullptr;
    int32_t * no_slots = nullptr; char * dummy = nullptr, * contiguous = nullptr;
    CUDA_CHECK(hipMalloc(&y, y_bytes)); CUDA_CHECK(hipMemcpy(y, y_host.data(), y_bytes, hipMemcpyHostToDevice));
    CUDA_CHECK(hipMalloc(&dst, size_t(M)*cols*sizeof(float))); CUDA_CHECK(hipMalloc(&ref, size_t(M)*cols*sizeof(float)));
    CUDA_CHECK(hipMalloc(&dev_ids, size_t(cols)*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&ids_dst, size_t(cols)*sizeof(int32_t))); CUDA_CHECK(hipMalloc(&bounds, (experts + 1)*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&no_slots, experts*sizeof(int32_t))); CUDA_CHECK(hipMemset(no_slots, 0xff, experts*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&dummy, 4096));
    CUDA_CHECK(hipMalloc(&contiguous, size_t(layers*3*experts)*slice));
    CUDA_CHECK(hipMemcpy(contiguous, file.data() + shift, size_t(layers*3*experts)*slice, hipMemcpyHostToDevice));
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));

    int total_batches = 0;
    for (const int ring : {2, 3, 5, 9, 40}) {
        for (const l2_prompt_fill fill : {l2_prompt_fill::lru, l2_prompt_fill::scan, l2_prompt_fill::mru}) {
            l2_config cfg;
            cfg.class_layout = true; cfg.batched = true; cfg.verify = true; cfg.log_mask = GGML_EXPERT_LOG_L2;
            cfg.experts_used = used; cfg.prefill_rows = tokens; cfg.decode_rows = 1; cfg.staged_min_rows = 2;
            cfg.prompt_fill = fill;
            l2_tier tier(geo, cfg);
            // residents: every third expert of each layer
            expert_slot_table host(layers, std::vector<int32_t>(experts, -1));
            int residents = 0;
            for (int l = 0; l < layers; ++l) for (int e = 0; e < experts; e += 3) { host[l][e] = residents++; }
            std::vector<std::array<size_t, 3>> tails(1);
            for (int k = 0; k < 3; ++k) { tails[0][k] = arena_tail_bytes(geo, 0, k); }
            class_storage st = plan_storage_classes(geo, tails, expert_os::io_alignment, 0, false);
            CHECK(st.place({residents}, {ring}));
            st.floor = {2};
            tier.set_storage(st);
            CHECK(tier.sized());
            CHECK(tier.allocate(0));
            l2_host_geometry host_geo;
            host_geo.device_base.resize(1); host_geo.host_base.resize(1);
            std::vector<void *> arenas;
            for (int k = 0; k < 3; ++k) {
                void * h = nullptr; void * d = nullptr;
                const size_t bytes = size_t(st.slots(0))*st.pitch[0][k] + st.tail[0][k];
                CUDA_CHECK(hipHostMalloc(&h, bytes, hipHostMallocMapped | hipHostMallocCoherent));
                memset(h, 0, bytes);
                CUDA_CHECK(hipHostGetDevicePointer(&d, h, 0));
                host_geo.host_base[0][k] = h; host_geo.device_base[0][k] = d;
                arenas.push_back(h);
            }
            const std::string filename = path.string();
            for (int l = 0; l < layers; ++l) for (int k = 0; k < 3; ++k) {
                tier.set_backing(l, k, 0, filename.c_str(), shift + uint64_t((l*3 + k)*experts)*slice);
            }
            std::string reason;
            CHECK(tier.open_files(reason));
            CHECK(tier.map());
            const std::vector<std::vector<int32_t>> vram(layers, std::vector<int32_t>(experts, -1));
            const expert_locations homes = host_locations(host);
            // the residents' bytes: written as the loader would, at slot + shift, tail cleared
            tier.set_homes(vram, homes, host_geo);
            for (int l = 0; l < layers; ++l) for (int e = 0; e < experts; ++e) {
                if (host[l][e] < 0) { continue; }
                for (int k = 0; k < 3; ++k) {
                    memcpy(tier.location_address(l, k, homes[l][e]), file_slice(l, k, e), slice);
                    tier.finish_write(l, k, homes[l][e]);
                }
            }
            tier.refresh_addresses();
            tier.start_worker();
            for (int ub = 0; ub < 6; ++ub) {
                const int layer = ub % layers;
                // routing: `used` distinct experts per token, more of them as the ubatches go on
                std::vector<int32_t> ids(cols);
                const int spread = ub < 2 ? 6 : experts;
                for (int t = 0; t < tokens; ++t) {
                    for (int u = 0; u < used; ++u) {
                        int e;
                        bool again;
                        do {
                            e = int(next() % uint32_t(spread));
                            again = false;
                            for (int v = 0; v < u; ++v) { again = again || ids[t*used + v] == e; }
                        } while (again);
                        ids[t*used + u] = e;
                    }
                }
                std::vector<int32_t> order, start(experts + 1, 0);
                for (int e = 0; e < experts; ++e) {
                    start[e] = (int) order.size();
                    for (int c = 0; c < cols; ++c) { if (ids[c] == e) { order.push_back(c); } }
                }
                start[experts] = (int) order.size();
                CUDA_CHECK(hipMemcpy(dev_ids, ids.data(), cols*sizeof(int32_t), hipMemcpyHostToDevice));
                CUDA_CHECK(hipMemcpy(ids_dst, order.data(), cols*sizeof(int32_t), hipMemcpyHostToDevice));
                CUDA_CHECK(hipMemcpy(bounds, start.data(), (experts + 1)*sizeof(int32_t), hipMemcpyHostToDevice));
                ggml_tensor tids = {}; tids.type = GGML_TYPE_I32; tids.data = dev_ids;
                tids.ne[0] = used; tids.ne[1] = tokens; tids.ne[2] = tids.ne[3] = 1;
                tids.nb[0] = 4; tids.nb[1] = 4*used;
                const int launches = tier.batch_launches(layer, tokens, true);
                CHECK(launches == l2_batch_launches(std::min(experts, used*tokens), std::min(experts, l2_batch_capacity(ring))));
                tier.publish_and_wait(layer, &tids, stream, launches);
                const l2_tier::batch_tables tables = tier.batch_device(layer);
                CHECK(tables.batch_of != nullptr && tables.capacity == std::min(experts, l2_batch_capacity(ring)));
                for (int k = 0; k < 3; ++k) {
                    tier.wait_kind(layer, k, stream);
                    CUDA_CHECK(hipMemsetAsync(dst, 0xff, size_t(M)*cols*sizeof(float), stream));
                    batch_mmq_launch(stream, dummy, y, ids_dst, bounds, dst, dummy, no_slots, tier.addresses(layer, k),
                        tables.batch_of, 0, experts, K, M, cols, tokens, slice);
                    for (int b = 1; b <= launches; ++b) {
                        tier.batch_wait(layer, k, b, stream);
                        batch_mmq_launch(stream, dummy, y, ids_dst, bounds, dst, dummy, no_slots, tier.addresses(layer, k),
                            tables.lists + size_t(b - 1)*tables.capacity, -1, tables.capacity, K, M, cols, tokens, slice);
                        tier.batch_done(layer, k, b, stream);
                    }
                    // the reference: one launch over a contiguous copy of every expert
                    batch_mmq_launch(stream, contiguous + size_t((layer*3 + k)*experts)*slice, y, ids_dst, bounds, ref, nullptr,
                        nullptr, nullptr, nullptr, 0, experts, K, M, cols, tokens, slice);
                    std::vector<float> a(size_t(M)*cols), r(size_t(M)*cols);
                    CUDA_CHECK(hipMemcpyAsync(a.data(), dst, a.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
                    CUDA_CHECK(hipMemcpyAsync(r.data(), ref, r.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
                    CUDA_CHECK(hipStreamSynchronize(stream));
                    bool nonzero = false;
                    for (size_t i = 0; i < a.size(); ++i) { CHECK(std::isfinite(r[i])); nonzero = nonzero || r[i] != 0.0f; }
                    CHECK(nonzero);
                    if (memcmp(a.data(), r.data(), a.size()*sizeof(float)) != 0) {
                        fprintf(stderr, "ring %d ub %d kind %d: the batched launches differ from the single launch\n", ring, ub, k);
                        return 1;
                    }
                }
                tier.mark_done(layer, stream);
                CUDA_CHECK(hipStreamSynchronize(stream));
            }
            tier.stop_worker();
            const l2_counters c = tier.counters();
            CHECK(c.verify_bad == 0 && c.batched_generations > 0 && c.batches >= c.batched_generations);
            CHECK(c.ssd_reads > 0 && c.ssd_reads % 3 == 0);
            total_batches += (int) c.batches;
            printf("PASS: batched service, ring %d, prompt fill %s: %llu generations in %llu batches, %llu reads, %llu ring hits, "
                   "%llu demoted, bit-identical to one launch\n", ring, l2_prompt_fill_name(fill),
                (unsigned long long) c.batched_generations, (unsigned long long) c.batches, (unsigned long long) c.ssd_reads,
                (unsigned long long) c.ring_hits, (unsigned long long) c.batch_demoted);
            for (void * h : arenas) { CUDA_CHECK(hipHostFree(h)); }
        }
    }
    CHECK(total_batches > 0);
    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipFree(contiguous)); CUDA_CHECK(hipFree(dummy)); CUDA_CHECK(hipFree(no_slots));
    CUDA_CHECK(hipFree(bounds)); CUDA_CHECK(hipFree(ids_dst)); CUDA_CHECK(hipFree(dev_ids));
    CUDA_CHECK(hipFree(ref)); CUDA_CHECK(hipFree(dst)); CUDA_CHECK(hipFree(y));
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_cuda_set_device(0);
    CHECK(bitmap_test() == 0);
    CHECK(serve_test() == 0);
    CHECK(wait_clock_test() == 0);
    CHECK(mover_test(true) == 0);
    CHECK(mover_test(false) == 0);
    CHECK(finite_inclusive_test() == 0);
    CHECK(stage_test() == 0);
    CHECK(redraw_test(false) == 0);
    CHECK(redraw_test(true) == 0);
    CHECK(host_table_test() == 0);
    CHECK(host_redraw_test() == 0);
    CHECK(host_chunk_cost_test() == 0);
    CHECK(early_ssd_test(argv[1]) == 0);
    CHECK(batched_service_test(argv[1]) == 0);
    constexpr int K = 640, M = 2560, N = 14, J = 16, experts = 17, selected = 4;
    constexpr size_t slice = size_t(K/32)*24*M, shift = 4064, tail = 288;
    const auto path = std::filesystem::path(argv[1]) / "ring-weights.bin";
    std::vector<char> file(shift + experts*slice + 4096, char(0xff));
    for (int e = 0; e < experts; ++e) {
        memset(file.data() + shift + e*slice, 0, slice);
        file[shift + e*slice] = char(e + 1);
    }
    { std::ofstream out(path, std::ios::binary); out.write(file.data(), file.size()); CHECK(bool(out)); }
    geometry geo;
    geo.n_layers = 2; geo.n_experts = experts;
    geo.layer_class = {0, 0}; geo.class_layers = {2};
    geo.class_bytes = {{slice, slice, slice}};
    ggml_tensor weights = {};
    weights.type = GGML_TYPE_Q5_1; weights.ne[0] = K; weights.ne[1] = M; weights.ne[2] = experts; weights.ne[3] = 1;
    geo.tensors = {{{&weights, &weights, &weights}}, {{&weights, &weights, &weights}}};
    l2_config cfg; cfg.verify = true; cfg.log_mask = GGML_EXPERT_LOG_L2;
    l2_tier tier(geo, cfg);
    CHECK(tier.allocate(0));
    l2_host_geometry host_geo; host_geo.device_base.resize(1);
    std::vector<std::vector<int32_t>> homes(2, std::vector<int32_t>(experts, -1));
    tier.set_homes(homes, host_locations(homes), host_geo);
    const std::string filename = path.string();
    for (int l = 0; l < 2; ++l) for (int k = 0; k < 3; ++k) { tier.set_backing(l, k, 0, filename.c_str(), shift); }
    std::string reason;
    CHECK(tier.open_files(reason));
    CHECK(tier.map());
    int * y = nullptr; float * dst = nullptr; int32_t * slots = nullptr, * dev_ids = nullptr; char * copy = nullptr;
    CUDA_CHECK(hipMalloc(&y, 1024*1024)); CUDA_CHECK(hipMemset(y, 0, 1024*1024));
    CUDA_CHECK(hipMalloc(&dst, M*N*sizeof(float)));
    CUDA_CHECK(hipMalloc(&slots, experts*sizeof(int32_t))); CUDA_CHECK(hipMemset(slots, 0xff, experts*sizeof(int32_t)));
    CUDA_CHECK(hipMalloc(&dev_ids, N*sizeof(int32_t)));
    std::vector<int32_t> host_ids(N, selected);
    CUDA_CHECK(hipMemcpy(dev_ids, host_ids.data(), N*sizeof(int32_t), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMalloc(&copy, slice + tail));
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    ggml_tensor ids = {}; ids.type = GGML_TYPE_I32; ids.data = dev_ids;
    ids.ne[0] = ids.ne[2] = ids.ne[3] = 1; ids.ne[1] = N; ids.nb[0] = ids.nb[1] = 4;
    const int cc = GGML_CUDA_CC_OFFSET_AMD + 0x1201;
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q5_1, J, false, cc);
    CHECK(config.type != GGML_TYPE_COUNT);
    const int ntx = (N + J - 1)/J, nty = (M + config.I - 1)/config.I;
    const bool stream_k = ggml_cuda_mmq_get_stream_k(GGML_TYPE_Q5_1, J, false, cc);
    const dim3 grid = stream_k ? dim3(ntx*nty, 1, 1) : dim3(nty, ntx, 1), block(32, config.nthreads/32, 1);
    const auto one = init_fastdiv_values(1);
    std::vector<float> result(M*N);
    std::vector<char> before(slice + tail);
    hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
    tier.start_worker();
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    tier.publish_and_wait(0, &ids, stream);
    readback<<<32, 128, 0, stream>>>(tier.addresses(0, 2), selected, copy, slice + tail);
    CUDA_CHECK(hipMemcpyAsync(before.data(), copy, before.size(), hipMemcpyDeviceToHost, stream));
    // Channel zero resolves directly to the selected expert in mapped ring memory.
    mul_mat_q<GGML_TYPE_Q5_1, J, false><<<grid, block, mmq_get_nbytes_shared(config, cc), stream>>>(
        nullptr, y, nullptr, nullptr, dst, nullptr, nullptr, copy, slots, slots, tier.addresses(0, 2) + selected, nullptr, 0,
        init_fastdiv_values(K/32), M, N, K/32, N, M, one, one, slice, 0, 0, one, one, 0, 0, 0, init_fastdiv_values(ntx));
    CUDA_CHECK(hipGetLastError());
    tier.mark_done(0, stream);
    CUDA_CHECK(hipMemcpyAsync(result.data(), dst, result.size()*sizeof(float), hipMemcpyDeviceToHost, stream));
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int pass = 0; pass < 3; ++pass) {
        CUDA_CHECK(hipGraphLaunch(executable, stream)); CUDA_CHECK(hipStreamSynchronize(stream));
        CHECK(memcmp(before.data(), file.data() + shift + selected*slice, slice) == 0);
        for (size_t i = slice; i < before.size(); ++i) { CHECK(before[i] == 0); }
        for (float value : result) { CHECK(std::isfinite(value) && value == 0.0f); }
    }
    tier.stop_worker();
    CHECK(tier.counters().ssd_reads == 3);
    CHECK(tier.counters().ring_hits == 2);
    CHECK(tier.counters().verify_bad == 0);
    CHECK(tier.counters().measured_layers == 3 && tier.counters().distinct == 3);
    CHECK(tier.counters().ssd_bytes == 3*slice && tier.counters().file_bytes == 9*slice);
    CHECK(tier.counters().prompt_ubatches == 3 && tier.counters().decode_ubatches == 0);
    CHECK(tier.counters().install_ssd_bytes == 0 && tier.wait_ms() > 0);
    const int batch = tier.install_read_slots();
    CHECK(batch == cfg.queue_depth);
    std::vector<l2_read> reads;
    for (int i = 0; i < batch; ++i) for (int k = 0; k < 3; ++k) { reads.push_back({i%2, k, selected + i, i}); }
    for (int pass = 0; pass < 3; ++pass) {
        CHECK(tier.read_install(reads, reason));
        for (const auto & read : reads) {
            const void * src = tier.read_address(read);
            CUDA_CHECK(hipMemcpy(copy, src, slice + tail, hipMemcpyHostToDevice));
            CUDA_CHECK(hipMemcpy(before.data(), copy, before.size(), hipMemcpyDeviceToHost));
            CHECK(memcmp(before.data(), file.data() + shift + read.expert*slice, slice) == 0);
            for (size_t j = slice; j < before.size(); ++j) { CHECK(before[j] == 0); }
        }
    }
    CHECK(tier.counters().install_ssd_bytes == 3*reads.size()*slice);
    CHECK(tier.counters().ssd_reads == 3 && tier.counters().verify_bad == 0);
    // Reusing the scratch prefix must not leave a stale demand hit.
    tier.set_homes(homes, host_locations(homes), host_geo);
    tier.start_worker();
    CUDA_CHECK(hipGraphLaunch(executable, stream)); CUDA_CHECK(hipStreamSynchronize(stream));
    tier.stop_worker();
    CHECK(memcmp(before.data(), file.data() + shift + selected*slice, slice) == 0);
    CHECK(tier.counters().ssd_reads == 6 && tier.counters().ring_hits == 2);
    printf("PASS: batched install uses shifted ring data, exact H2D and zero tails; demand refills after scratch use\n");
    printf("PASS: 14-row demand reads 1 of 17 experts, skips next layer, replays exact payload and zero MMQ padding\n");
    // A plan that keeps every routed expert out of the file must not reach the worker at all.
    std::vector<std::vector<int32_t>> vram(2, std::vector<int32_t>(experts, -1));
    for (int l = 0; l < 2; ++l) { vram[l][selected] = 0; }
    tier.set_homes(vram, host_locations(homes), host_geo);
    const l2_counters before_resident = tier.counters();
    tier.start_worker();
    tier.publish_and_wait(0, &ids, stream);
    tier.mark_done(0, stream);
    CUDA_CHECK(hipStreamSynchronize(stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (tier.counters().measured_layers == before_resident.measured_layers &&
            std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    tier.stop_worker();
    CHECK(tier.counters().generations == before_resident.generations);
    CHECK(tier.counters().ssd_reads == before_resident.ssd_reads);
    CHECK(tier.counters().ssd_bytes == before_resident.ssd_bytes);
    CHECK(tier.counters().distinct == before_resident.distinct + 1);
    CHECK(tier.counters().vram_bytes > before_resident.vram_bytes);
    CHECK(tier.counters().measured_layers == before_resident.measured_layers + 1);
    printf("PASS: a fully resident layer records its sample with no worker service and no read\n");
    CUDA_CHECK(hipGraphExecDestroy(executable)); CUDA_CHECK(hipGraphDestroy(graph));
    CUDA_CHECK(hipStreamDestroy(stream)); CUDA_CHECK(hipFree(copy)); CUDA_CHECK(hipFree(dev_ids));
    CUDA_CHECK(hipFree(slots)); CUDA_CHECK(hipFree(dst)); CUDA_CHECK(hipFree(y));
    return 0;
}
