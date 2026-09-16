// Standalone HIP fixture: real mailbox publication, ring reads and MMQ padding.
#include "expert-l2.cu"
#include "expert-os-win32.cpp"
#include "expert-l1.cu"
#include "expert-host.cu"
#include "mmq.cuh"

#include <cmath>
#include <filesystem>
#include <fstream>

void ggml_cuda_set_device(int d) { CUDA_CHECK(hipSetDevice(d)); }
int ggml_cuda_get_device() { int d = 0; CUDA_CHECK(hipGetDevice(&d)); return d; }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "%s: %s at %s:%d in %s\n", stmt, msg, file, line, func);
    abort();
}
using namespace ggml_cuda_expert;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

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
        l2_publish_kernel<<<1, 128>>>(device, device_bits, device_serve, ids, rows, used, stride, experts);
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

    l2_publish_kernel<<<1, 128>>>(device, device_bits, device_serve, ids, rows, used, stride, experts);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(host->published == 1 && host->need == 0 && host->ready == host->generation);

    serve[routed/32] |= 1u << (routed%32);
    l2_publish_kernel<<<1, 128>>>(device, device_bits, device_serve, ids, rows, used, stride, experts);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(host->published == 2 && host->need == 1 && host->ready == 0);

    serve[routed/32] &= ~(1u << (routed%32));
    serve[absent/32] |= 1u << (absent%32);
    l2_publish_kernel<<<1, 128>>>(device, device_bits, device_serve, ids, rows, used, stride, experts);
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
    __hip_atomic_store(&m->reserved0, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

static int wait_clock_test() {
    l2_mailbox * host = nullptr, * device = nullptr;
    CHECK(coherent_alloc((void **) &host, (void **) &device, sizeof(*host)));
    host->generation = 1;
    hipStream_t stream; CUDA_CHECK(hipStreamCreate(&stream));
    std::chrono::steady_clock::time_point begin;
    std::thread responder([&] {
        while (!expert_os::load_acquire(&host->reserved0)) { std::this_thread::yield(); }
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
        nullptr, y, nullptr, nullptr, dst, nullptr, nullptr, copy, slots, slots, tier.addresses(0, 2) + selected,
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
