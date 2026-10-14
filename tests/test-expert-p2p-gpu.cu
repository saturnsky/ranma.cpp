// Standalone HIP fixture for the peer-to-peer read engine of the expert cache (expert-p2p.h): real
// GGUF ranges are read through the engine (IoRing -> GPU_UPLOAD staging -> D3D12 copy queue ->
// imported VRAM region) and compared byte for byte with an unbuffered host read of the same range.
//
// Cases: one large unaligned read; many small reads at random unaligned offsets with zero tails;
// requests larger than the whole staging with small chunks (every staging slot is reused while the
// run is in flight); the progress prefix; the tier's consumption pattern (a kernel reads the slot,
// the slot is rewritten through the engine, a kernel that was enqueued behind a host-mailbox wait
// reads it again, many times: no stale device cache lines); refusals (a bad device, a missing file,
// a read out of range). Read-only on the files; writes nothing to disk.
//
// usage: test-expert-p2p-gpu [--files a.gguf,b.gguf,...] [--seed N] [--quick]
#include "../ggml/src/ggml-cuda/expert-p2p-win32.cpp"

#include <hip/hip_runtime.h>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace ggml_cuda_expert;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); fflush(stderr); return 1; } } while (0)
#define HIPCHECK(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "FAIL line %d: %s: %s\n", __LINE__, #x, hipGetErrorName(e_)); return 1; } } while (0)

namespace {

const char * default_files[] = { nullptr };  // pass GGUF files (>3 GiB each) with --files

struct host_file {
    std::string path;
    HANDLE h = INVALID_HANDLE_VALUE;
    uint64_t size = 0;
};

uint64_t rng_state = 0x5EED0F2F0ull;
uint64_t rnd() { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17; return rng_state; }

// the reference: an unbuffered synchronous read of the covering sectors, then the payload
bool host_read(host_file & f, uint64_t offset, size_t bytes, std::vector<unsigned char> & out) {
    const uint64_t a = offset & ~uint64_t(4095), e = (offset + bytes + 4095) & ~uint64_t(4095);
    const size_t n = size_t(e - a);
    unsigned char * buf = (unsigned char *) VirtualAlloc(nullptr, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { return false; }
    OVERLAPPED ov = {};
    ov.Offset = DWORD(a); ov.OffsetHigh = DWORD(a >> 32);
    ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    DWORD got = 0;
    bool ok = ReadFile(f.h, buf, DWORD(n), &got, &ov) != 0;
    if (!ok && GetLastError() == ERROR_IO_PENDING) { ok = GetOverlappedResult(f.h, &ov, &got, TRUE) != 0; }
    CloseHandle(ov.hEvent);
    ok = ok && got >= size_t(offset - a) + bytes;
    if (ok) { out.assign(buf + (offset - a), buf + (offset - a) + bytes); }
    VirtualFree(buf, 0, MEM_RELEASE);
    return ok;
}

struct fixture {
    std::vector<host_file> files;
    std::vector<int> engine_file;   // engine index of files[i]
};

// every read's payload equals the file, every zero tail is zero
int check_reads(fixture & fx, const void * region_dev, size_t region_bytes, const std::vector<p2p_read> & reads, const char * what) {
    std::vector<unsigned char> back(region_bytes), ref;
    HIPCHECK(hipMemcpy(back.data(), region_dev, region_bytes, hipMemcpyDeviceToHost));
    size_t bad = 0, bad_tail = 0;
    for (const p2p_read & r : reads) {
        size_t fi = 0;
        while (fx.engine_file[fi] != r.file) { ++fi; }
        CHECK(host_read(fx.files[fi], r.offset, r.bytes, ref));
        if (memcmp(back.data() + r.dst, ref.data(), r.bytes) != 0) { ++bad; }
        for (size_t z = 0; z < r.zero_tail; ++z) {
            if (back[r.dst + r.bytes + z] != 0) { ++bad_tail; break; }
        }
    }
    printf("  %s: %zu reads checked against the file: %zu payload mismatches, %zu tails not zero\n", what, reads.size(), bad, bad_tail);
    CHECK(bad == 0 && bad_tail == 0);
    return 0;
}

// random reads packed into [0, region_bytes): sizes in [lo, hi], unaligned offsets, tails in [0, tail_hi]
std::vector<p2p_read> random_reads(fixture & fx, size_t count, size_t lo, size_t hi, size_t tail_hi, size_t region_bytes) {
    std::vector<p2p_read> out;
    size_t dst = 0;
    for (size_t i = 0; i < count; ++i) {
        p2p_read r;
        const size_t fi = rnd() % fx.files.size();
        r.file = fx.engine_file[fi];
        r.bytes = lo + size_t(rnd() % (hi - lo + 1));
        r.bytes = std::max<size_t>(32, r.bytes & ~size_t(31));
        const uint64_t span = fx.files[fi].size - (uint64_t(1) << 30) - r.bytes - 8192;
        r.offset = (uint64_t(1) << 30) + (rnd() % span) / 32 * 32;   // the GGUF data alignment, not a sector
        r.zero_tail = tail_hi ? size_t(rnd() % (tail_hi + 1)) : 0;
        r.region = 0;
        dst = (dst + 255) & ~size_t(255);
        if (dst + r.bytes + r.zero_tail > region_bytes) { break; }
        r.dst = dst;
        dst += r.bytes + r.zero_tail;
        out.push_back(r);
    }
    return out;
}

__global__ void k_sum(const uint64_t * p, size_t n, unsigned long long * out) {
    unsigned long long s = 0, x = 0;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        const uint64_t v = p[i];
        s += v * (2*i + 1);
        x ^= v;
    }
    atomicAdd(&out[0], s);
    atomicXor(&out[1], x);
}

// the tier's wait: spin on a host-coherent word until it equals `want`
__global__ void k_wait(uint32_t * mailbox, uint32_t want) {
    while (__hip_atomic_load(mailbox, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
        __builtin_amdgcn_s_sleep(1);
    }
}

void cpu_sum(const unsigned char * b, size_t bytes, unsigned long long & s, unsigned long long & x) {
    const uint64_t * p = (const uint64_t *) b;
    s = 0; x = 0;
    for (size_t i = 0; i < bytes/8; ++i) { s += p[i] * (2*i + 1); x ^= p[i]; }
}

double ms_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> paths;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--files" && i + 1 < argc) {
            paths.clear();
            std::string s = argv[++i];
            size_t p = 0;
            while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) { q = s.size(); } paths.push_back(s.substr(p, q - p)); p = q + 1; }
        } else if (a == "--seed" && i + 1 < argc) {
            rng_state = strtoull(argv[++i], nullptr, 10) | 1;
        } else if (a == "--quick") {
            quick = true;
        } else {
            fprintf(stderr, "usage: %s [--files a,b,...] [--seed N] [--quick]\n", argv[0]);
            return 2;
        }
    }
    if (paths.empty()) { fprintf(stderr, "usage: %s --files a.gguf,b.gguf,... [--seed N] [--quick]\n", argv[0]); return 2; }
    hipDeviceProp_t prop = {};
    HIPCHECK(hipGetDeviceProperties(&prop, 0));
    printf("device %s %s\n", prop.name, prop.gcnArchName);
    if (strncmp(prop.gcnArchName, "gfx1201", 7) != 0) { printf("expected gfx1201 (HIP_VISIBLE_DEVICES=1)\n"); return 3; }
    HIPCHECK(hipSetDevice(0));

    // ---- refusals first: nothing must stay allocated ----
    {
        std::string why;
        p2p_config cfg;
        auto bad = p2p_engine::create(7, cfg, why);
        printf("create on device 7: %s (%s)\n", bad ? "created" : "refused", why.c_str());
        (void) hipGetLastError();
        CHECK(!bad && !why.empty());
    }

    fixture fx;
    std::string why;
    p2p_config cfg;
    cfg.staging_bytes = size_t(64) << 20;
    cfg.chunk_bytes = size_t(1) << 20;
    cfg.queue_depth = 32;
    double t = ms_now();
    auto engine = p2p_engine::create(0, cfg, why);
    if (!engine) { printf("FAIL: engine creation: %s\n", why.c_str()); return 1; }
    printf("engine: %s (created in %.0f ms)\n", engine->description().c_str(), ms_now() - t);
    for (const std::string & p : paths) {
        host_file f;
        f.path = p;
        f.h = CreateFileA(p.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (f.h == INVALID_HANDLE_VALUE) { printf("FAIL: cannot open %s\n", p.c_str()); return 1; }
        LARGE_INTEGER sz; GetFileSizeEx(f.h, &sz); f.size = uint64_t(sz.QuadPart);
        CHECK(f.size > (uint64_t(3) << 30));
        const int idx = engine->add_file(p, why);
        CHECK(idx >= 0);
        CHECK(engine->add_file(p, why) == idx);   // idempotent
        fx.files.push_back(f);
        fx.engine_file.push_back(idx);
    }
    CHECK(engine->add_file("no-such-file.gguf", why) < 0);
    printf("files: %zu registered; a missing file is refused (%s)\n", fx.files.size(), why.c_str());

    const size_t region_bytes = size_t(160) << 20;
    void * region = nullptr;
    const int rid = engine->alloc_region(region_bytes, &region, why);
    if (rid < 0) { printf("FAIL: region: %s\n", why.c_str()); return 1; }
    CHECK(rid == 0 && region != nullptr && engine->region_device(0) == region);
    printf("region: %zu MiB imported at %p; engine VRAM: staging %zu MiB + regions %zu MiB\n", region_bytes >> 20, region,
        engine->staging_bytes() >> 20, engine->region_bytes() >> 20);

    // ---- 1: one large unaligned read ----
    {
        HIPCHECK(hipMemset(region, 0xCD, region_bytes));
        HIPCHECK(hipDeviceSynchronize());
        p2p_read r;
        r.file = fx.engine_file[0]; r.offset = (uint64_t(1) << 30) + 12345*32; r.bytes = (size_t(96) << 20) + 4000; r.region = 0;
        r.dst = 4096 + 64; r.zero_tail = 9000;
        t = ms_now();
        CHECK(engine->run(&r, 1, why) || (printf("run: %s\n", why.c_str()), false));
        const double ms = ms_now() - t;
        printf("case 1: one read of %.1f MiB (unaligned offset and destination) in %.1f ms = %.2f GB/s\n", r.bytes/1048576.0, ms, r.bytes/(ms*1e6));
        if (check_reads(fx, region, region_bytes, {r}, "case 1")) { return 1; }
    }

    // ---- 2: many small reads with tails, in several runs ----
    {
        size_t total = 0;
        const int runs = quick ? 2 : 6;
        for (int k = 0; k < runs; ++k) {
            HIPCHECK(hipMemset(region, 0xCD, region_bytes));
            HIPCHECK(hipDeviceSynchronize());
            std::vector<p2p_read> reads = random_reads(fx, 400, 4096, size_t(3) << 20, 8192, region_bytes);
            size_t bytes = 0;
            for (auto & r : reads) { bytes += r.bytes; }
            std::vector<size_t> seen;
            t = ms_now();
            CHECK(engine->run(reads.data(), reads.size(), why, [&](size_t n) { seen.push_back(n); return true; }) || (printf("run: %s\n", why.c_str()), false));
            const double ms = ms_now() - t;
            CHECK(!seen.empty() && seen.back() == reads.size());
            for (size_t i = 1; i < seen.size(); ++i) { CHECK(seen[i] > seen[i - 1]); }
            printf("case 2 run %d: %zu reads, %.1f MiB in %.1f ms = %.2f GB/s, %zu progress calls\n", k, reads.size(), bytes/1048576.0, ms, bytes/(ms*1e6), seen.size());
            if (check_reads(fx, region, region_bytes, reads, "case 2")) { return 1; }
            total += reads.size();
        }
        CHECK(total > 0);
    }

    // ---- 3: small chunks, requests larger than the staging (every slot reused within a run) ----
    {
        p2p_config tiny = cfg;
        tiny.staging_bytes = size_t(16) << 20;
        tiny.chunk_bytes = size_t(64) << 10;
        tiny.queue_depth = 8;
        auto e2 = p2p_engine::create(0, tiny, why);
        if (!e2) { printf("FAIL: second engine: %s\n", why.c_str()); return 1; }
        for (size_t i = 0; i < paths.size(); ++i) { CHECK(e2->add_file(paths[i], why) == fx.engine_file[i]); }
        void * r2 = nullptr;
        const size_t r2_bytes = size_t(80) << 20;
        CHECK(e2->alloc_region(r2_bytes, &r2, why) == 0);
        HIPCHECK(hipMemset(r2, 0xCD, r2_bytes));
        HIPCHECK(hipDeviceSynchronize());
        std::vector<p2p_read> reads;
        size_t dst = 0;
        for (int i = 0; i < 3; ++i) {
            p2p_read r;
            r.file = fx.engine_file[(size_t) i % fx.files.size()];
            r.offset = (uint64_t(2) << 30) + uint64_t(i)*77777*32;
            r.bytes = (size_t(20) << 20) + size_t(i)*4096 + 96;
            r.region = 0; r.dst = dst; r.zero_tail = 512;
            dst += (r.bytes + r.zero_tail + 255) & ~size_t(255);
            reads.push_back(r);
        }
        t = ms_now();
        CHECK(e2->run(reads.data(), reads.size(), why) || (printf("run: %s\n", why.c_str()), false));
        const double ms = ms_now() - t;
        const p2p_stats & st = e2->stats();
        printf("case 3: %zu reads of ~20 MiB through a 16 MiB staging in 64 KiB chunks: %llu pieces, %llu command lists, %.1f ms\n",
            reads.size(), (unsigned long long) st.pieces, (unsigned long long) st.lists, ms);
        CHECK(st.pieces >= 3*320);
        if (check_reads(fx, r2, r2_bytes, reads, "case 3")) { return 1; }
        e2.reset();
    }

    // ---- 4: the tier's pattern: read, rewrite through the engine, read behind a mailbox wait ----
    {
        const int iters = quick ? 100 : 400;
        uint32_t * mailbox = nullptr;
        HIPCHECK(hipHostMalloc((void **) &mailbox, 64, hipHostMallocMapped | hipHostMallocCoherent));
        uint32_t * mailbox_dev = nullptr;
        HIPCHECK(hipHostGetDevicePointer((void **) &mailbox_dev, mailbox, 0));
        *mailbox = 0;
        unsigned long long * sums = nullptr;
        HIPCHECK(hipMalloc(&sums, 32));
        hipStream_t s;
        HIPCHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        hipEvent_t first_read;
        HIPCHECK(hipEventCreateWithFlags(&first_read, hipEventDisableTiming));
        std::vector<unsigned char> ref;
        int stale = 0, bad = 0;
        size_t slot_bytes_sum = 0;
        for (int it = 0; it < iters; ++it) {
            // slot sizes that stay in the device L2 (64 KiB .. 2 MiB)
            const size_t slot_bytes = (size_t(64) << 10) << (it % 6);
            slot_bytes_sum += slot_bytes;
            const size_t dst = size_t(it % 4) * (size_t(4) << 20);
            HIPCHECK(hipMemsetAsync(sums, 0, 32, s));
            // K1 pulls the old bytes into the device caches, then the stream waits for the mailbox
            hipLaunchKernelGGL(k_sum, dim3(64), dim3(256), 0, s, (const uint64_t *) ((char *) region + dst), slot_bytes/8, sums);
            HIPCHECK(hipEventRecord(first_read, s));
            hipLaunchKernelGGL(k_wait, dim3(1), dim3(1), 0, s, mailbox_dev, uint32_t(it + 1));
            hipLaunchKernelGGL(k_sum, dim3(64), dim3(256), 0, s, (const uint64_t *) ((char *) region + dst), slot_bytes/8, sums + 2);
            HIPCHECK(hipGetLastError());
            // host: after K1, rewrite the slot with new bytes, then release the wait
            while (hipEventQuery(first_read) == hipErrorNotReady) { YieldProcessor(); }
            p2p_read r;
            const size_t fi = rnd() % fx.files.size();
            r.file = fx.engine_file[fi]; r.bytes = slot_bytes; r.region = 0; r.dst = dst;
            r.offset = (uint64_t(1) << 30) + (rnd() % (fx.files[fi].size - (uint64_t(2) << 30))) / 32 * 32;
            CHECK(engine->run(&r, 1, why) || (printf("run: %s\n", why.c_str()), false));
            __atomic_store_n(mailbox, uint32_t(it + 1), __ATOMIC_RELEASE);
            HIPCHECK(hipStreamSynchronize(s));
            unsigned long long h[4];
            HIPCHECK(hipMemcpy(h, sums, 32, hipMemcpyDeviceToHost));
            CHECK(host_read(fx.files[fi], r.offset, r.bytes, ref));
            unsigned long long cs, cx;
            cpu_sum(ref.data(), ref.size(), cs, cx);
            if (h[2] != cs || h[3] != cx) {
                ++bad;
                // the old bytes again: a stale line
                if (h[2] == h[0] && h[3] == h[1]) { ++stale; }
            }
        }
        printf("case 4: %d rewrites of a slot the device just read (64 KiB .. 2 MiB, %.1f MiB total), read again behind a "
               "host-mailbox wait: %d mismatches (%d equal to the old bytes)\n", iters, slot_bytes_sum/1048576.0, bad, stale);
        CHECK(bad == 0);
        HIPCHECK(hipStreamDestroy(s));
        HIPCHECK(hipEventDestroy(first_read));
        HIPCHECK(hipFree(sums));
        HIPCHECK(hipHostFree(mailbox));
    }

    // ---- 6: latency and the bridge (last read completion -> copies signalled), copies submitted as the reads
    //         complete vs queued before the reads behind a CPU-signalled gate; 1 slice, one expert (3 slices),
    //         a layer of 6 experts (18 slices); decode-sized runs, idle machine ----
    {
        const int iters = quick ? 20 : 60;
        p2p_config gated_cfg = cfg;
        gated_cfg.pregate = true;
        auto gated = p2p_engine::create(0, gated_cfg, why);
        if (!gated) { printf("FAIL: gated engine: %s\n", why.c_str()); return 1; }
        for (size_t i = 0; i < paths.size(); ++i) { CHECK(gated->add_file(paths[i], why) == fx.engine_file[i]); }
        void * g_region = nullptr;
        CHECK(gated->alloc_region(region_bytes, &g_region, why) == 0);
        struct mode { const char * name; p2p_engine * e; void * region; };
        const mode modes[2] = {{"as completed", engine.get(), region}, {"pre-gated", gated.get(), g_region}};
        for (int slices : {1, 3, 18}) {
            for (int mi = 0; mi < 2; ++mi) {
                p2p_engine & e = *modes[mi].e;
                std::vector<double> ms, tail;
                std::vector<p2p_read> last;
                for (int it = 0; it < iters; ++it) {
                    std::vector<p2p_read> reads;
                    for (int k = 0; k < slices; ++k) {
                        p2p_read r;
                        const size_t fi = rnd() % fx.files.size();
                        r.file = fx.engine_file[fi]; r.bytes = k % 3 == 2 ? 3219456 : 2433024; r.region = 0;
                        r.dst = size_t(k) * (size_t(4) << 20); r.zero_tail = 512;
                        r.offset = (uint64_t(1) << 30) + (rnd() % (fx.files[fi].size - (uint64_t(2) << 30))) / 32 * 32;
                        reads.push_back(r);
                    }
                    const p2p_stats before = e.stats();
                    const double t0 = ms_now();
                    CHECK(e.run(reads.data(), reads.size(), why) || (printf("run: %s\n", why.c_str()), false));
                    ms.push_back(ms_now() - t0);
                    const p2p_stats & after = e.stats();
                    if (after.tail_runs > before.tail_runs) { tail.push_back(after.tail_ms - before.tail_ms); }
                    last = reads;
                }
                if (check_reads(fx, modes[mi].region, region_bytes, last, "case 6 last run")) { return 1; }
                std::sort(ms.begin(), ms.end());
                std::sort(tail.begin(), tail.end());
                printf("case 6: %2d slice(s), copies %-12s: run median %.3f ms (p90 %.3f, min %.3f); bridge median %.3f ms "
                       "(p90 %.3f, min %.3f), %zu runs, %llu gated\n", slices, modes[mi].name, ms[ms.size()/2], ms[ms.size()*9/10], ms[0],
                    tail.empty() ? -1.0 : tail[tail.size()/2], tail.empty() ? -1.0 : tail[tail.size()*9/10], tail.empty() ? -1.0 : tail[0],
                    ms.size(), (unsigned long long) e.stats().gated_runs);
            }
        }
        gated.reset();
    }

    // ---- 5: refusals of a live engine ----
    {
        p2p_read r;
        r.file = fx.engine_file[0]; r.offset = uint64_t(1) << 30; r.bytes = 4096; r.region = 0; r.dst = region_bytes - 100;
        std::string w2;
        CHECK(!engine->run(&r, 1, w2));
        printf("case 5: a read past the region is refused (%s)\n", w2.c_str());
        // the engine stays refused after a failure: the caller must fall back
        r.dst = 0;
        CHECK(!engine->run(&r, 1, w2));
        printf("case 5: the engine refuses further runs after a failure (%s)\n", w2.c_str());
        void * big = nullptr;
        CHECK(engine->alloc_region(size_t(4) << 30, &big, w2) < 0);
        printf("case 5: a region of 4 GiB is refused (%s)\n", w2.c_str());
    }

    const p2p_stats & st = engine->stats();
    printf("stats: runs %llu, reads %llu, %.1f MiB, pieces %llu, lists %llu, run %.1f ms, io wait %.1f ms, fence wait %.1f ms\n",
        (unsigned long long) st.runs, (unsigned long long) st.reads, st.bytes/1048576.0, (unsigned long long) st.pieces,
        (unsigned long long) st.lists, st.run_ms, st.io_wait_ms, st.fence_wait_ms);
    engine.reset();
    for (auto & f : fx.files) { CloseHandle(f.h); }
    printf("PASS\n");
    return 0;
}
