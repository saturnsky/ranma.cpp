// Micro-benchmark of the decode pool's step kernel: the LRU victim rule of ggml/src/ggml-cuda/expert-ub-pool.cu
// against a decaying-LFU variant (replay: per-request expert counts, halved every 256 layer calls, ties by LRU,
// admission ghost:8). A reduced replica of ub_step_kernel (hit test, ghost filter, victim ranking by counting, fill);
// no prefetch/ring/tables/copy kernel. Real shape: DS4.1F, 40 layers, 384 experts, top-6, 10 positions per layer,
// 1-row launches, 256 threads, one block. The input is a trace file (convert_trace.py of the issue folder).
// build: clang++ -x hip --offload-arch=gfx1201 -O3 pool_lfu_bench.cpp -o pool_lfu_bench.exe
// run:   pool_lfu_bench trace.bin [chunk_tokens=500] [event_tokens=40] [dump_tokens=0] [dump_file]
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); exit(1); } } while (0)

static constexpr int NT = 256, MAXE = 1024, NLAY = 40, NEXP = 384, NUSED = 6, RPOS = 10, GHOST = 8, DECAY = 256;
static constexpr int LOGW = 2 + 2*NUSED;
static constexpr uint32_t NEVER = 0xFFFFFFFFu;

struct State {
    int32_t * owner;       // [L*R]
    uint32_t * stamp;      // [L*R]
    int16_t * pos_of;      // [L][E]
    uint32_t * last_miss;  // [L][E]
    uint32_t * freq;       // [L][E] (LFU only)
    uint32_t * step;       // [L]
    unsigned long long * counters;  // hits, fills, over, evict
    int32_t * log;         // [n][LOGW] debug: hits, fills, (new, old) per rank; may be null
};

template <bool LFU>
static __device__ __forceinline__ bool precedes(const State & s, int off, uint32_t t, int x, int y, int layer) {
    const bool ex = s.owner[off + x] < 0, ey = s.owner[off + y] < 0;
    if (ex != ey) return ex;
    if (!ex) {
        if (LFU) {
            const uint32_t fx = s.freq[layer*NEXP + s.owner[off + x]], fy = s.freq[layer*NEXP + s.owner[off + y]];
            if (fx != fy) return fx < fy;
        }
        const uint32_t ax = t - s.stamp[off + x], ay = t - s.stamp[off + y];
        if (ax != ay) return ax > ay;
    }
    return x < y;
}

static __device__ __forceinline__ bool eligible(const State & s, int off, uint32_t t, int p) {
    return s.owner[off + p] < 0 || s.stamp[off + p] != t;
}

template <bool LFU>
static __global__ void __launch_bounds__(NT) step_kernel(const State s, const int32_t * __restrict__ ids, const int layer, const int logidx) {
    __shared__ uint32_t s_t;
    __shared__ int s_n_miss, s_n_victims;
    __shared__ unsigned int s_hits, s_over, s_fills, s_evict;
    __shared__ int32_t s_key[MAXE], s_miss[MAXE], s_miss_key[MAXE], s_victim[256];
    __shared__ uint8_t s_missed[MAXE];
    const int tid = threadIdx.x, E = NEXP, r = RPOS, off = layer*RPOS;
    if (tid == 0) {
        s.step[layer] = s.step[layer] + 1;
        s_t = s.step[layer];
        s_n_miss = s_n_victims = 0;
        s_hits = s_over = s_fills = s_evict = 0;
    }
    __syncthreads();
    const uint32_t t = s_t;
    int16_t * pos_of = s.pos_of + size_t(layer)*E;
    uint32_t * fr = s.freq + size_t(layer)*E;
    for (int e = tid; e < E; e += NT) {
        s_key[e] = -1;
        s_missed[e] = 0;
        if (LFU && t % DECAY == 0) fr[e] >>= 1;
    }
    __syncthreads();
    unsigned int hits = 0;
    for (int i = tid; i < NUSED; i += NT) {
        const int id = ids[i];
        if ((unsigned) id >= (unsigned) E) continue;
        if (LFU) atomicAdd(&fr[id], 1u);
        const int p = pos_of[id];
        if (p >= 0 && p < r) { s.stamp[off + p] = t; hits++; continue; }
        // the ghost filter: only an expert that missed within the last GHOST steps (not this one) may be admitted
        const uint32_t last = s.last_miss[size_t(layer)*E + id];
        s_missed[id] = 1;
        if (!(last != NEVER && last != t && t - last <= (uint32_t) GHOST)) { s_missed[id] = 2; continue; }
        atomicMax(&s_key[id], NUSED - 1 - i);
    }
    if (hits) atomicAdd(&s_hits, hits);
    __syncthreads();
    for (int e = tid; e < E; e += NT) {
        if (s_key[e] >= 0) {
            const int j = atomicAdd(&s_n_miss, 1);
            s_miss[j] = e; s_miss_key[j] = s_key[e];
        }
    }
    __syncthreads();
    const int n_miss = s_n_miss;
    if (n_miss > 0) {
        for (int p = tid; p < r; p += NT) {
            if (!eligible(s, off, t, p)) continue;
            int rank = 0;
            for (int q = 0; q < r; ++q) rank += q != p && eligible(s, off, t, q) && precedes<LFU>(s, off, t, q, p, layer);
            s_victim[rank] = p;
            atomicAdd(&s_n_victims, 1);
        }
        __syncthreads();
        const int nfill = min(n_miss, s_n_victims);
        unsigned int fills = 0, over = 0, evict = 0;
        for (int j = tid; j < n_miss; j += NT) {
            const int key = s_miss_key[j];
            int rank = 0;
            for (int i = 0; i < n_miss; ++i) rank += s_miss_key[i] > key;
            if (rank >= nfill) { over++; continue; }
            const int p = s_victim[rank], q = off + p, e = s_miss[j], old = s.owner[q];
            if (old >= 0) { pos_of[old] = -1; evict++; }
            s.owner[q] = e;
            s.stamp[q] = t;
            pos_of[e] = (int16_t) p;
            if (s.log && logidx >= 0) { s.log[logidx*LOGW + 2 + 2*rank] = e; s.log[logidx*LOGW + 3 + 2*rank] = old; }
            fills++;
        }
        if (fills) atomicAdd(&s_fills, fills);
        if (over) atomicAdd(&s_over, over);
        if (evict) atomicAdd(&s_evict, evict);
    }
    __threadfence_system();
    __syncthreads();
    for (int e = tid; e < E; e += NT) if (s_missed[e] != 0) s.last_miss[size_t(layer)*E + e] = t;
    if (tid == 0) {
        s.counters[0] += s_hits; s.counters[1] += s_fills; s.counters[2] += s_over; s.counters[3] += s_evict;
        if (s.log && logidx >= 0) { s.log[logidx*LOGW] = s_hits; s.log[logidx*LOGW + 1] = s_fills; }
    }
}

static __global__ void empty_kernel(const State s, const int32_t * __restrict__ ids, const int layer, const int logidx) {
    if (threadIdx.x == 0 && ids[0] == -12345) s.counters[3]++;
}

static __global__ void reset_kernel(const State s) {
    for (int i = threadIdx.x; i < NLAY*RPOS; i += NT) { s.owner[i] = -1; s.stamp[i] = 0; }
    for (int i = threadIdx.x; i < NLAY*NEXP; i += NT) { s.pos_of[i] = -1; s.freq[i] = 0; }   // last_miss persists (replay: lastv)
}

struct Var {
    State s;
    int mode;   // 0 lru, 1 lfu, 2 empty
    std::vector<double> gpu_us, host_us, ev_us, wall_us;
};

static void alloc(State & s, bool log, size_t nlog) {
    CK(hipMalloc(&s.owner, NLAY*RPOS*4)); CK(hipMalloc(&s.stamp, NLAY*RPOS*4));
    CK(hipMalloc(&s.pos_of, NLAY*NEXP*2)); CK(hipMalloc(&s.last_miss, NLAY*NEXP*4));
    CK(hipMalloc(&s.freq, NLAY*NEXP*4)); CK(hipMalloc(&s.step, NLAY*4)); CK(hipMalloc(&s.counters, 32));
    std::vector<uint32_t> nv(NLAY*NEXP, NEVER);
    CK(hipMemcpy(s.last_miss, nv.data(), nv.size()*4, hipMemcpyHostToDevice));
    CK(hipMemset(s.step, 0, NLAY*4)); CK(hipMemset(s.counters, 0, 32));
    reset_kernel<<<1, NT>>>(s);
    s.log = nullptr;
    if (log) { CK(hipMalloc(&s.log, nlog*LOGW*4)); CK(hipMemset(s.log, 0xFF, nlog*LOGW*4)); }
}

static void launch(const Var & v, const int32_t * ids, int layer, int logidx) {
    if (v.mode == 0) step_kernel<false><<<1, NT>>>(v.s, ids, layer, logidx);
    else if (v.mode == 1) step_kernel<true><<<1, NT>>>(v.s, ids, layer, logidx);
    else empty_kernel<<<1, NT>>>(v.s, ids, layer, logidx);
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size()/2]; }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[size_t(p*(v.size() - 1))]; }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: pool_lfu_bench trace.bin [chunk_tokens=500] [event_tokens=40] [dump_tokens=0] [dump_file]\n"); return 2; }
    const int chunk = argc > 2 ? atoi(argv[2]) : 500, evtok = argc > 3 ? atoi(argv[3]) : 40, dumptok = argc > 4 ? atoi(argv[4]) : 0;
    FILE * f = fopen(argv[1], "rb");
    int hdr[4]; if (!f || fread(hdr, 4, 4, f) != 4) { fprintf(stderr, "trace\n"); return 2; }
    const int L = hdr[0], K = hdr[2], T = hdr[3];
    if (L != NLAY || K != NUSED) { fprintf(stderr, "shape %d %d\n", L, K); return 2; }
    std::vector<int32_t> req(T); if (fread(req.data(), 4, T, f) != (size_t) T) return 2;
    std::vector<uint16_t> raw(size_t(T)*L*K); if (fread(raw.data(), 2, raw.size(), f) != raw.size()) return 2; fclose(f);
    std::vector<int32_t> idv(raw.begin(), raw.end());
    int32_t * d_ids; CK(hipMalloc(&d_ids, idv.size()*4)); CK(hipMemcpy(d_ids, idv.data(), idv.size()*4, hipMemcpyHostToDevice));
    // chunks: within one request, at most `chunk` tokens
    std::vector<std::pair<int, int>> chunks;
    for (int t = 0; t < T;) { int e = t + 1; while (e < T && e - t < chunk && req[e] == req[t]) e++; chunks.push_back({t, e}); t = e; }
    Var v[3]; v[0].mode = 0; v[1].mode = 1; v[2].mode = 2;
    for (int i = 0; i < 3; ++i) alloc(v[i].s, dumptok > 0 && i < 2, size_t(dumptok)*L);
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    const char * names[3] = {"LRU", "LFU", "empty"};
    const int WARM = 2;   // warm-up chunks, discarded
    int prev_req = -1;
    const int order3[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};   // the position of each variant rotates across rounds
    for (size_t c = 0; c < chunks.size(); ++c) {
        const int t0 = chunks[c].first, t1 = chunks[c].second;
        const bool newreq = req[t0] != prev_req; prev_req = req[t0];
        for (int oi = 0; oi < 3; ++oi) {
            Var & x = v[order3[c % 3][oi]];
            if (newreq && x.mode < 2) reset_kernel<<<1, NT>>>(x.s);
            CK(hipDeviceSynchronize());
            // batch: host submission time and GPU time of the whole chunk (kernels back to back)
            const int ntok = t1 - t0;
            auto h0 = std::chrono::steady_clock::now();
            CK(hipEventRecord(e0));
            for (int t = t0; t < t1; ++t) for (int l = 0; l < L; ++l) launch(x, d_ids + (size_t(t)*L + l)*K, l, t < dumptok ? t*L + l : -1);
            auto h1 = std::chrono::steady_clock::now();
            CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1));
            auto h2 = std::chrono::steady_clock::now();
            if ((int) c >= WARM) {
                x.wall_us.push_back(std::chrono::duration<double, std::micro>(h2 - h0).count()/(ntok*L));
                x.gpu_us.push_back(ms*1000.0/(ntok*L));
                x.host_us.push_back(std::chrono::duration<double, std::micro>(h1 - h0).count()/(ntok*L));
            }
        }
    }
    unsigned long long cnt[2][4];
    for (int i = 0; i < 2; ++i) CK(hipMemcpy(cnt[i], v[i].s.counters, 32, hipMemcpyDeviceToHost));
    if (dumptok > 0 && argc > 5) {
        FILE * o = fopen(argv[5], "wb");
        for (int i = 0; i < 2; ++i) {
            std::vector<int32_t> lg(size_t(dumptok)*L*LOGW);
            CK(hipMemcpy(lg.data(), v[i].s.log, lg.size()*4, hipMemcpyDeviceToHost));
            fwrite(lg.data(), 4, lg.size(), o);
        }
        fclose(o);
    }
    // per-launch event pairs (kernel time incl. the event floor), interleaved per sample, on evolving state
    for (int rep = 0; rep < 30; ++rep) {
        const int tt = 2000 + (rep % 7)*500;
        for (int oi = 0; oi < 3; ++oi) {
            Var & x = v[order3[rep % 3][oi]];
            for (int t = tt; t < tt + evtok; ++t) for (int l = 0; l < L; ++l) {
                CK(hipEventRecord(e0));
                launch(x, d_ids + (size_t(t)*L + l)*K, l, -1);
                CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
                float ms; CK(hipEventElapsedTime(&ms, e0, e1));
                if (rep >= 2) x.ev_us.push_back(ms*1000.0);
            }
        }
    }
    printf("chunks %zu (<=%d tokens), warm %d, tokens %d\n", chunks.size(), chunk, WARM, T);
    for (int i = 0; i < 3; ++i) {
        printf("%-5s batch GPU us/launch: median %.3f p10 %.3f p90 %.3f | host submit us/launch: median %.3f p10 %.3f p90 %.3f | event-pair us: median %.3f p10 %.3f p90 %.3f (n=%zu)\n",
               names[i], median(v[i].gpu_us), pct(v[i].gpu_us, .1), pct(v[i].gpu_us, .9),
               median(v[i].host_us), pct(v[i].host_us, .1), pct(v[i].host_us, .9),
               median(v[i].ev_us), pct(v[i].ev_us, .1), pct(v[i].ev_us, .9), v[i].ev_us.size());
    }
    for (int i = 0; i < 3; ++i) printf("%-5s wall (submit+sync) us/launch: median %.3f p10 %.3f p90 %.3f\n", names[i], median(v[i].wall_us), pct(v[i].wall_us, .1), pct(v[i].wall_us, .9));
    { std::vector<double> dw; for (size_t k = 0; k < v[0].wall_us.size(); ++k) dw.push_back(v[1].wall_us[k] - v[0].wall_us[k]); printf("LFU-LRU paired wall us/launch: median %.3f p10 %.3f p90 %.3f\n", median(dw), pct(dw, .1), pct(dw, .9)); }
    std::vector<double> dg, dh;
    for (size_t k = 0; k < v[0].gpu_us.size(); ++k) { dg.push_back(v[1].gpu_us[k] - v[0].gpu_us[k]); dh.push_back(v[1].host_us[k] - v[0].host_us[k]); }
    printf("LFU-LRU paired-chunk batch GPU us/launch: median %.3f p10 %.3f p90 %.3f | host median %.3f\n", median(dg), pct(dg, .1), pct(dg, .9), median(dh));
    printf("LFU-LRU event-pair median diff %.3f us\n", median(v[1].ev_us) - median(v[0].ev_us));
    for (int i = 0; i < 2; ++i) printf("%s (pre-event phase) hits %llu fills %llu over %llu evict %llu\n", names[i], cnt[i][0], cnt[i][1], cnt[i][2], cnt[i][3]);
    return 0;
}
