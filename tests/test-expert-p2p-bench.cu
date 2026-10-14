// Model-free bandwidth / limits / contention probe of the peer-to-peer read engine (expert-p2p.h)
// against the host path (unbuffered overlapped read into pinned host memory, then hipMemcpyAsync).
// Reads raw byte ranges of the given GGUF files; loads no model, writes nothing to disk.
//
// usage: test-expert-p2p-bench --files a.gguf,b.gguf [--total-gib N] [--rounds N] [--only limits|ring|bw|cont]
#include "../ggml/src/ggml-cuda/expert-p2p-win32.cpp"

#include <hip/hip_runtime.h>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace ggml_cuda_expert;

#define HIPCHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "FAIL line %d: %s: %s\n", __LINE__, #x, hipGetErrorName(e_)); fflush(stderr); exit(1); } } while (0)

namespace {

uint64_t rng_state = 0xB3A5C0FFEEull;
uint64_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }
double ms_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

double cpu_ms() {
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto v = [](FILETIME f) { return double((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 1e4; };
    return v(k) + v(u);
}
double private_gib() {
    PROCESS_MEMORY_COUNTERS_EX m = {};
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &m, sizeof(m));
    return double(m.PrivateUsage) / double(1ull << 30);
}
double ws_gib() {
    PROCESS_MEMORY_COUNTERS_EX m = {};
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &m, sizeof(m));
    return double(m.WorkingSetSize) / double(1ull << 30);
}
void sys_mem(double & avail_gib, double & commit_gib) {
    MEMORYSTATUSEX s = {}; s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    avail_gib = double(s.ullAvailPhys) / double(1ull << 30);
    commit_gib = double(s.ullTotalPageFile - s.ullAvailPageFile) / double(1ull << 30);
}
double hip_free_gib() { size_t f = 0, t = 0; (void) hipMemGetInfo(&f, &t); return double(f) / double(1ull << 30); }
void stamp(const char * what) {
    double a, c; sys_mem(a, c);
    printf("  [mem %-22s] sys avail %.1f GiB, sys commit %.1f GiB, proc private %.2f GiB, ws %.2f GiB, hip free %.2f GiB\n",
        what, a, c, private_gib(), ws_gib(), hip_free_gib());
    fflush(stdout);
}
void gpu_proc_counter(const char * what) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "powershell -NoProfile -Command \"(Get-Counter '\\GPU Process Memory(pid_%lu*)\\Dedicated Usage','\\GPU Process Memory(pid_%lu*)\\Shared Usage','\\GPU Process Memory(pid_%lu*)\\Total Committed' -ErrorAction SilentlyContinue).CounterSamples | ForEach-Object { '{0} = {1:N0} MiB' -f $_.Path.Substring($_.Path.LastIndexOf('\\')+1), ($_.CookedValue/1MB) }\"",
        (unsigned long) GetCurrentProcessId(), (unsigned long) GetCurrentProcessId(), (unsigned long) GetCurrentProcessId());
    printf("  [gpu process counters: %s]\n", what); fflush(stdout);
    system(cmd);
}

struct host_file { std::string path; HANDLE h = INVALID_HANDLE_VALUE; uint64_t size = 0; };

// the read lists (the same for both paths)
std::vector<p2p_read> make_reads(const std::vector<host_file> & files, const std::vector<int> & eng, size_t region_bytes,
        bool slices, size_t large_bytes) {
    std::vector<p2p_read> out;
    size_t dst = 0;
    for (;;) {
        p2p_read r;
        const size_t fi = rnd() % files.size();
        r.file = eng[fi];
        if (slices) { static const size_t sz[3] = {2433024, 3219456, 2826240}; r.bytes = sz[rnd() % 3]; }
        else { r.bytes = large_bytes; }
        r.zero_tail = 0;
        const uint64_t span = files[fi].size - (uint64_t(1) << 30) - r.bytes - 8192;
        r.offset = (uint64_t(1) << 30) + (rnd() % span) / 32 * 32;
        dst = (dst + 255) & ~size_t(255);
        if (dst + r.bytes > region_bytes) { break; }
        r.region = 0; r.dst = dst;
        dst += r.bytes;
        out.push_back(r);
    }
    return out;
}

// host path: unbuffered overlapped read -> pinned -> hipMemcpyAsync
struct host_pipe {
    struct slot { unsigned char * buf = nullptr; OVERLAPPED ov = {}; hipEvent_t ev = nullptr; size_t delta = 0; };
    std::vector<slot> slots;
    size_t cap = 0;
    hipStream_t stream = nullptr;
    host_pipe(int depth, size_t cap_bytes) : slots(depth), cap(cap_bytes) {
        HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (auto & s : slots) {
            HIPCHECK(hipHostMalloc((void **) &s.buf, cap, hipHostMallocDefault));
            HIPCHECK(hipEventCreateWithFlags(&s.ev, hipEventDisableTiming));
            s.ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        }
    }
    ~host_pipe() {
        for (auto & s : slots) { (void) hipHostFree(s.buf); (void) hipEventDestroy(s.ev); CloseHandle(s.ov.hEvent); }
        (void) hipStreamDestroy(stream);
    }
    bool issue(slot & s, const host_file & f, const p2p_read & r) {
        const uint64_t a = r.offset & ~uint64_t(4095), e = (r.offset + r.bytes + 4095) & ~uint64_t(4095);
        if (size_t(e - a) > cap) { return false; }
        if (size_t(e - a) > cap) { return false; }
        s.delta = size_t(r.offset - a);
        ResetEvent(s.ov.hEvent);
        s.ov.Offset = DWORD(a); s.ov.OffsetHigh = DWORD(a >> 32);
        DWORD got = 0;
        const BOOL ok = ReadFile(f.h, s.buf, DWORD(e - a), &got, &s.ov);
        if (!ok && GetLastError() != ERROR_IO_PENDING) { return false; }
        return true;
    }
    bool run(const std::vector<host_file> & files, const std::vector<int> & eng, const std::vector<p2p_read> & reads, char * dst_base) {
        auto file_of = [&](int e) -> const host_file & { for (size_t i = 0; i < files.size(); ++i) { if (eng[i] == e) { return files[i]; } } return files[0]; };
        const size_t n = reads.size(), S = slots.size();
        for (size_t i = 0; i < std::min(n, S); ++i) { if (!issue(slots[i], file_of(reads[i].file), reads[i])) { return false; } }
        for (size_t i = 0; i < n; ++i) {
            slot & s = slots[i % S];
            DWORD got = 0;
            if (!GetOverlappedResult(file_of(reads[i].file).h, &s.ov, &got, TRUE)) { return false; }
            HIPCHECK(hipMemcpyAsync(dst_base + reads[i].dst, s.buf + s.delta, reads[i].bytes, hipMemcpyHostToDevice, stream));
            HIPCHECK(hipEventRecord(s.ev, stream));
            if (i + S < n) {
                HIPCHECK(hipEventSynchronize(s.ev));
                if (!issue(s, file_of(reads[i + S].file), reads[i + S])) { return false; }
            }
        }
        HIPCHECK(hipStreamSynchronize(stream));
        return true;
    }
};

// background loads
__global__ void k_read_host(const uint4 * p, size_t n, unsigned long long * out) {
    unsigned long long s = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint4 v = p[i];
        s += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x123456789abcdefull) { out[0] = s; }
}

struct load {
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> bytes{0};
    double t0 = 0, t1 = 0;
    std::thread th;
    virtual ~load() {}
    virtual void body() = 0;
    void start() { stop = false; bytes = 0; t0 = ms_now(); th = std::thread([this]() { HIPCHECK(hipSetDevice(0)); body(); }); }
    double finish() { stop = true; th.join(); t1 = ms_now(); return double(bytes.load()) / ((t1 - t0) * 1e6); }
};

struct load_h2d : load {   // pinned host -> VRAM through hipMemcpyAsync
    void body() override {
        const size_t sz = size_t(128) << 20;
        void * h = nullptr; void * d = nullptr; hipStream_t s;
        HIPCHECK(hipHostMalloc(&h, sz, hipHostMallocDefault)); memset(h, 1, sz);
        HIPCHECK(hipMalloc(&d, sz));
        HIPCHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        while (!stop) { HIPCHECK(hipMemcpyAsync(d, h, sz, hipMemcpyHostToDevice, s)); HIPCHECK(hipStreamSynchronize(s)); bytes += sz; }
        (void) hipStreamDestroy(s); (void) hipFree(d); (void) hipHostFree(h);
    }
};
struct load_d2d : load {   // device-to-device copies (the install path: ring region -> arena slot)
    void body() override {
        const size_t sz = size_t(256) << 20;
        void * a = nullptr; void * b = nullptr; hipStream_t s;
        HIPCHECK(hipMalloc(&a, sz)); HIPCHECK(hipMalloc(&b, sz)); HIPCHECK(hipMemset(a, 1, sz));
        HIPCHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        while (!stop) { HIPCHECK(hipMemcpyAsync(b, a, sz, hipMemcpyDeviceToDevice, s)); HIPCHECK(hipStreamSynchronize(s)); bytes += sz; }
        (void) hipStreamDestroy(s); (void) hipFree(a); (void) hipFree(b);
    }
};
struct load_shader : load {   // a kernel reads pinned host memory over the link
    void body() override {
        const size_t sz = size_t(512) << 20;
        void * h = nullptr; unsigned long long * o = nullptr; hipStream_t s;
        HIPCHECK(hipHostMalloc(&h, sz, hipHostMallocMapped)); memset(h, 1, sz);
        void * hd = nullptr; HIPCHECK(hipHostGetDevicePointer(&hd, h, 0));
        HIPCHECK(hipMalloc((void **) &o, 8));
        HIPCHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        while (!stop) {
            hipLaunchKernelGGL(k_read_host, dim3(1024), dim3(256), 0, s, (const uint4 *) hd, sz / 16, o);
            HIPCHECK(hipStreamSynchronize(s)); bytes += sz;
        }
        (void) hipStreamDestroy(s); (void) hipFree(o); (void) hipHostFree(h);
    }
};
struct load_cpu : load {   // 4 threads streaming reads over 256 MiB buffers (host DRAM bandwidth probe)
    void body() override {
        const int T = 4; const size_t sz = size_t(256) << 20;
        std::vector<std::thread> ts;
        std::atomic<uint64_t> tot{0};
        for (int t = 0; t < T; ++t) {
            ts.emplace_back([&, t]() {
                unsigned char * b = (unsigned char *) VirtualAlloc(nullptr, sz, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
                memset(b, t + 1, sz);
                volatile uint64_t sink = 0;
                while (!stop) {
                    uint64_t s = 0; const uint64_t * p = (const uint64_t *) b;
                    for (size_t i = 0; i < sz / 8; ++i) { s += p[i]; }
                    sink = sink + s; tot += sz;
                }
                VirtualFree(b, 0, MEM_RELEASE);
            });
        }
        for (auto & t : ts) { t.join(); }
        bytes = tot.load();
    }
};

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> paths;
    double total_gib = 12.0;
    int rounds = 2;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--files" && i + 1 < argc) {
            std::string s = argv[++i]; size_t p = 0;
            while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) { q = s.size(); } paths.push_back(s.substr(p, q - p)); p = q + 1; }
        } else if (a == "--total-gib" && i + 1 < argc) { total_gib = atof(argv[++i]);
        } else if (a == "--rounds" && i + 1 < argc) { rounds = atoi(argv[++i]);
        } else if (a == "--only" && i + 1 < argc) { only = argv[++i];
        } else { fprintf(stderr, "usage: %s --files a,b [--total-gib N] [--rounds N] [--only limits|ring|bw|cont]\n", argv[0]); return 2; }
    }
    if (paths.empty()) { fprintf(stderr, "--files required\n"); return 2; }
    hipDeviceProp_t prop = {};
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device %s %s\n", prop.name, prop.gcnArchName);
    if (strncmp(prop.gcnArchName, "gfx1201", 7) != 0) { printf("expected gfx1201\n"); return 3; }
    HIPCHECK(hipSetDevice(0));
    HIPCHECK(hipFree(0));
    stamp("start");

    std::vector<host_file> files;
    for (auto & p : paths) {
        host_file f; f.path = p;
        f.h = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (f.h == INVALID_HANDLE_VALUE) { printf("cannot open %s\n", p.c_str()); return 1; }
        LARGE_INTEGER sz; GetFileSizeEx(f.h, &sz); f.size = uint64_t(sz.QuadPart);
        files.push_back(f);
    }

    std::string why;
    // ================= limits =================
    if (only.empty() || only == "limits") {
        printf("== limits ==\n");
        for (size_t staging : {size_t(128) << 20, size_t(1) << 30, size_t(2) << 30}) {
            p2p_config cfg; cfg.staging_bytes = staging;
            const double t = ms_now(); const double c = private_gib();
            auto e = p2p_engine::create(0, cfg, why);
            const double ms = ms_now() - t;
            if (!e) { printf("staging %zu MiB: refused (%s) after %.0f ms\n", staging >> 20, why.c_str(), ms); continue; }
            for (auto & p : paths) { e->add_file(p, why); }
            printf("staging %zu MiB: created in %.0f ms, engine VRAM %zu MiB, proc private %+.2f GiB\n", staging >> 20, ms,
                e->staging_bytes() >> 20, private_gib() - c);
            const double r0 = ms_now(), pr0 = private_gib(), hf0 = hip_free_gib();
            void * reg = nullptr; std::string w;
            const int rid = e->alloc_region(size_t(1) << 30, &reg, w);
            printf("   +1 GiB region: alloc %.0f ms, rid %d, private %+.2f GiB, hip free %+.2f GiB\n",
                ms_now() - r0, rid, private_gib() - pr0, hip_free_gib() - hf0);
            if (rid >= 0) {
                std::vector<p2p_read> rd(1); rd[0].file = 0; rd[0].offset = (uint64_t(1) << 30); rd[0].bytes = size_t(64) << 20; rd[0].region = 0; rd[0].dst = 0;
                const double q0 = ms_now();
                const bool ok = e->run(rd.data(), 1, w);
                printf("   first run (64 MiB, includes ring/file registration): %s in %.0f ms\n", ok ? "ok" : w.c_str(), ms_now() - q0);
            }
        }
        {
            p2p_config cfg; cfg.staging_bytes = size_t(128) << 20;
            auto e = p2p_engine::create(0, cfg, why);
            for (auto & p : paths) { e->add_file(p, why); }
            stamp("before regions");
            gpu_proc_counter("before regions");
            for (int gib : {1, 2, 3}) {
                const double pr0 = private_gib(), hf0 = hip_free_gib(), t = ms_now();
                void * reg = nullptr; std::string w;
                const int rid = e->alloc_region(size_t(gib) << 30, &reg, w);
                const double ms = ms_now() - t;
                printf("region %d GiB: rid %d (%s), %.0f ms (%.2f s/GiB), private %+.2f GiB, hip free %+.2f GiB\n", gib, rid, w.c_str(), ms,
                    ms / 1000.0 / gib, private_gib() - pr0, hip_free_gib() - hf0);
                if (rid >= 0) {
                    unsigned long long * o = nullptr; HIPCHECK(hipMalloc((void **) &o, 32)); HIPCHECK(hipMemset(o, 0, 32));
                    HIPCHECK(hipMemset(reg, 0x5A, size_t(gib) << 30));
                    hipLaunchKernelGGL(k_read_host, dim3(1024), dim3(256), 0, 0, (const uint4 *) reg, (size_t(gib) << 30) / 16, o);
                    const hipError_t er = hipDeviceSynchronize();
                    printf("   memset + kernel over the whole %d GiB region: %s\n", gib, hipGetErrorName(er));
                    (void) hipFree(o);
                }
                stamp("after region");
            }
            gpu_proc_counter("after 1+2+3 GiB regions (6 GiB imported)");
            std::string w2; void * big = nullptr;
            const int rid4 = e->alloc_region(size_t(4) << 30, &big, w2);
            printf("4 GiB region request: %s\n", rid4 < 0 ? ("refused by the engine: " + w2).c_str() : "created");
        }
        stamp("end limits");
    }


    // ================= ring (staging) size sweep =================
    if (only.empty() || only == "ring") {
        printf("== staging ring size sweep (expert slices, chunk 1 MiB; lists = command lists = fence signals) ==\n");
        std::vector<int> eng;
        for (size_t mib : {16, 32, 64, 256}) {
            for (int qd : {2, 4, 8, 16, 32}) {
                p2p_config cfg; cfg.staging_bytes = mib << 20; cfg.queue_depth = qd;
                auto e = p2p_engine::create(0, cfg, why);
                if (!e) { printf("staging %zu MiB: %s\n", mib, why.c_str()); continue; }
                eng.clear();
                for (auto & p : paths) { eng.push_back(e->add_file(p, why)); }
                void * reg = nullptr;
                if (e->alloc_region(size_t(1) << 30, &reg, why) != 0) { printf("region: %s\n", why.c_str()); continue; }
                { auto rd = make_reads(files, eng, size_t(1) << 30, true, 0); e->run(rd.data(), rd.size(), why); }
                for (int rep = 0; rep < rounds; ++rep) {
                    size_t bytes = 0; const p2p_stats s0 = e->stats(); const double c0 = cpu_ms(), t0 = ms_now();
                    while (double(bytes) < total_gib * double(1ull << 30) / 2) {
                        auto rd = make_reads(files, eng, size_t(1) << 30, true, 0);
                        size_t b = 0; for (auto & r : rd) { b += r.bytes; }
                        if (!e->run(rd.data(), rd.size(), why)) { printf("run failed %s\n", why.c_str()); return 1; }
                        bytes += b;
                    }
                    const double wall = ms_now() - t0, cpu = cpu_ms() - c0; const p2p_stats s1 = e->stats();
                    const double gb = double(bytes) / 1e9;
                    printf("staging %3zu MiB (QD %d): %6.2f GB/s, cpu %.3f s/GB, fence signals %.1f per GB (%.1f MiB per signal), io wait %.0f ms, fence wait %.0f ms\n",
                        mib, qd, gb / (wall / 1e3), (cpu / 1e3) / gb, double(s1.lists - s0.lists) / gb, 1000.0 / 1.048576 / (double(s1.lists - s0.lists) / gb),
                        s1.io_wait_ms - s0.io_wait_ms, s1.fence_wait_ms - s0.fence_wait_ms);
                    fflush(stdout);
                }
            }
        }
    }

    // ================= bandwidth and contention =================
    if (only.empty() || only == "bw" || only == "cont") {
        std::vector<int> eng;
        p2p_config cfg; cfg.staging_bytes = size_t(128) << 20;
        auto engine = p2p_engine::create(0, cfg, why);
        if (!engine) { printf("engine: %s\n", why.c_str()); return 1; }
        for (auto & p : paths) { eng.push_back(engine->add_file(p, why)); }
        const size_t region_bytes = size_t(1) << 30;
        void * region = nullptr;
        if (engine->alloc_region(region_bytes, &region, why) != 0) { printf("region: %s\n", why.c_str()); return 1; }
        char * dev = nullptr; HIPCHECK(hipMalloc((void **) &dev, region_bytes));
        host_pipe pipe_small(16, (size_t(4) << 20) + 8192), pipe_large(4, (size_t(256) << 20) + 8192);
        host_pipe * pipe_ptr = &pipe_small;
        { auto rd = make_reads(files, eng, region_bytes, true, 0); engine->run(rd.data(), rd.size(), why); pipe_ptr->run(files, eng, rd, dev); }

        struct kind { const char * name; bool slices; size_t large; };
        const kind kinds[2] = {{"expert slices 2.4-3.2 MiB", true, 0}, {"large reads 256 MiB", false, size_t(256) << 20}};
        if (only.empty() || only == "bw") {
            printf("== bandwidth (raw GGUF ranges of %zu file(s), unbuffered; fresh random offsets each run) ==\n", files.size());
            for (const kind & k : kinds) {
                pipe_ptr = k.slices ? &pipe_small : &pipe_large;
                for (int rd_i = 0; rd_i < rounds; ++rd_i) {
                    for (int path = 0; path < 2; ++path) {   // 0 = P2P, 1 = host
                        size_t bytes = 0, runs = 0; double a_min = 1e9, c_max = 0;
                        stamp(path == 0 ? "bw pre P2P" : "bw pre host");
                        const double c0 = cpu_ms(), t0 = ms_now();
                        const p2p_stats s0 = engine->stats();
                        while (double(bytes) < total_gib * double(1ull << 30)) {
                            auto rd = make_reads(files, eng, region_bytes, k.slices, k.large);
                            size_t b = 0; for (auto & r : rd) { b += r.bytes; }
                            const bool ok = path == 0 ? engine->run(rd.data(), rd.size(), why) : pipe_ptr->run(files, eng, rd, dev);
                            if (!ok) { printf("run failed %s\n", why.c_str()); return 1; }
                            bytes += b; ++runs;
                            double a, c; sys_mem(a, c); a_min = std::min(a_min, a); c_max = std::max(c_max, c);
                        }
                        const double wall = ms_now() - t0, cpu = cpu_ms() - c0;
                        const p2p_stats s1 = engine->stats();
                        printf("%-26s %-4s round %d: %6.2f GB/s (%.1f GiB in %.0f ms, %zu runs), cpu %.0f ms = %.3f cpu-s/GB, min avail %.1f GiB, max commit %.1f GiB\n",
                            k.name, path == 0 ? "P2P" : "host", rd_i, double(bytes) / (wall * 1e6), double(bytes) / double(1ull << 30), wall, runs, cpu,
                            (cpu / 1e3) / (double(bytes) / 1e9), a_min, c_max);
                        if (path == 0) { printf("      engine: io wait %.0f ms, fence wait %.0f ms, pieces %llu\n", s1.io_wait_ms - s0.io_wait_ms, s1.fence_wait_ms - s0.fence_wait_ms, (unsigned long long) (s1.pieces - s0.pieces)); }
                        fflush(stdout);
                    }
                }
            }
            stamp("after bw");
        }

        if (only.empty() || only == "cont") {
            const double phase_ms = 3000.0;
            printf("== contention (expert slices; each phase %.0f s) ==\n", phase_ms / 1000.0);
            auto p2p_for = [&](double ms) {
                size_t bytes = 0; const double t0 = ms_now();
                while (ms_now() - t0 < ms) {
                    auto rd = make_reads(files, eng, region_bytes, true, 0);
                    size_t b = 0; for (auto & r : rd) { b += r.bytes; }
                    if (!engine->run(rd.data(), rd.size(), why)) { printf("run failed %s\n", why.c_str()); exit(1); }
                    bytes += b;
                }
                return double(bytes) / ((ms_now() - t0) * 1e6);
            };
            auto host_for = [&](double ms) {
                size_t bytes = 0; const double t0 = ms_now();
                while (ms_now() - t0 < ms) {
                    auto rd = make_reads(files, eng, region_bytes, true, 0);
                    size_t b = 0; for (auto & r : rd) { b += r.bytes; }
                    if (!pipe_ptr->run(files, eng, rd, dev)) { printf("host run failed\n"); exit(1); }
                    bytes += b;
                }
                return double(bytes) / ((ms_now() - t0) * 1e6);
            };
            const double p2p_alone = p2p_for(phase_ms);
            const double host_alone = host_for(phase_ms);
            printf("alone: P2P %.2f GB/s, host path %.2f GB/s\n", p2p_alone, host_alone);
            load_h2d h2d; load_shader sh; load_cpu cpu; load_d2d d2d;
            struct ld { const char * name; load * l; };
            ld loads[4] = {{"D2D copy 256 MiB (install path)", &d2d}, {"H2D hipMemcpyAsync pinned->VRAM", &h2d}, {"shader reading pinned host mem", &sh}, {"4-thread CPU DRAM stream", &cpu}};
            for (auto & L : loads) {
                L.l->start(); Sleep(1000); (void) L.l->finish();
                L.l->start(); Sleep(DWORD(phase_ms)); const double alone = L.l->finish();
                L.l->start(); const double bp = p2p_for(phase_ms); const double with_p2p = L.l->finish();
                L.l->start(); const double bh = host_for(phase_ms); const double with_host = L.l->finish();
                printf("%-32s alone %6.2f GB/s | with P2P: load %6.2f (%3.0f%%), P2P %6.2f (%3.0f%% of alone) | with host path: load %6.2f (%3.0f%%), host %6.2f (%3.0f%% of alone)\n",
                    L.name, alone, with_p2p, 100 * with_p2p / alone, bp, 100 * bp / p2p_alone, with_host, 100 * with_host / alone, bh, 100 * bh / host_alone);
                fflush(stdout);
            }
            stamp("after cont");
        }
    }
    printf("DONE\n");
    return 0;
}
