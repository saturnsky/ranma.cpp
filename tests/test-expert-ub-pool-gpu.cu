// Standalone HIP fixture of the decode pool (expert-ub-pool.cuh): lend, step, copy first, eviction,
// over-r, reclaim and the address tables, with a lookup that resolves every selected expert the way
// ggml_cuda_expert_cache_select does and compares the bytes it reaches with the expert's home.
#include "expert-ub-pool.cu"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

void ggml_cuda_set_device(int d) { CUDA_CHECK(hipSetDevice(d)); }
int ggml_cuda_get_device() { int d = 0; CUDA_CHECK(hipGetDevice(&d)); return d; }
[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "%s: %s at %s:%d in %s\n", stmt, msg, file, line, func);
    abort();
}
using namespace ggml_cuda_expert;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

static constexpr int L = 3, E = 40, USED = 4, STRIDE_IDS = 5;
static const size_t KIND_BYTES[3] = { 4096 + 96, 4096 + 96, 2048 + 32 };

// What a matmul of the launch reads for every selected expert: a checksum per (row, k, kind).
static __global__ void resolve_kernel(const int32_t * ids, int n_rows, const int32_t * slots, const uint64_t * const * tables,
        const char * const * tensors, const size_t * strides, uint32_t * out) {
    const int i = blockIdx.x;   // row*USED + k
    const int kind = blockIdx.y;
    const int row = i/USED, k = i%USED;
    if (row >= n_rows) { return; }
    const int e = ids[row*STRIDE_IDS + k];
    const ggml_cuda_expert_source src = ggml_cuda_expert_cache_select(tensors[kind], (const void *) 0x1, slots, (uint32_t) e,
        nullptr, tables[kind]);
    const char * p = src.slot >= 0 ? nullptr : (const char *) src.data + size_t(src.channel)*strides[kind];
    __shared__ uint32_t acc;
    if (threadIdx.x == 0) { acc = 0; }
    __syncthreads();
    uint32_t s = 0;
    if (p != nullptr) {
        for (size_t b = threadIdx.x; b < strides[kind]; b += blockDim.x) { s += (uint32_t) (uint8_t) p[b]*(uint32_t) (b + 1); }
    }
    atomicAdd(&acc, s);
    __syncthreads();
    if (threadIdx.x == 0) { out[i*3 + kind] = acc; }
}

struct fixture {
    bool tier = false;                     // coherent table with home addresses and a serve bitmap
    std::vector<char *> host;              // [kind] host tensor (mapped), E slices
    std::vector<char *> host_dev;          // device alias
    std::vector<uint64_t *> tables;        // [layer*3 + kind] address table (device pointer)
    std::vector<uint64_t *> tables_host;   // coherent host pointer (tier) or null
    std::vector<uint32_t *> serve_dev;     // [layer]
    std::vector<uint32_t *> serve_host;    // [layer] (tier)
    std::vector<int32_t *> slots;          // [layer]
    std::vector<std::vector<uint64_t>> initial; // [layer*3 + kind]
    std::vector<uint32_t> ref;             // [kind][e] checksum of the home
    char * region = nullptr;
    size_t region_bytes = 0;
};

static uint32_t checksum(const char * p, size_t n) {
    uint32_t s = 0;
    for (size_t b = 0; b < n; ++b) { s += (uint32_t) (uint8_t) p[b]*(uint32_t) (b + 1); }
    return s;
}

static void table_read(const fixture & f, int layer, int kind, std::vector<uint64_t> & out) {
    out.resize(E);
    CUDA_CHECK(hipMemcpy(out.data(), f.tables[layer*3 + kind], E*sizeof(uint64_t), hipMemcpyDefault));
}

static int build(fixture & f, bool tier, std::vector<ub_pool_layer> & layers, std::vector<std::array<size_t, 3>> & bytes,
        std::vector<std::array<size_t, 3>> & tails) {
    f.tier = tier;
    std::mt19937 rng(tier ? 7 : 3);
    f.ref.assign(3*E, 0);
    for (int k = 0; k < 3; ++k) {
        char * h = nullptr;
        CUDA_CHECK(hipHostMalloc((void **) &h, E*KIND_BYTES[k], hipHostMallocMapped));
        for (size_t b = 0; b < E*KIND_BYTES[k]; ++b) { h[b] = (char) rng(); }
        void * d = nullptr;
        CUDA_CHECK(hipHostGetDevicePointer(&d, h, 0));
        f.host.push_back(h);
        f.host_dev.push_back((char *) d);
        for (int e = 0; e < E; ++e) { f.ref[k*E + e] = checksum(h + e*KIND_BYTES[k], KIND_BYTES[k]); }
    }
    layers.assign(L, ub_pool_layer());
    bytes.assign(L, {0, 0, 0});
    tails.assign(L, {0, 0, 0});
    f.initial.assign(L*3, {});
    for (int l = 0; l < L; ++l) {
        // experts 0..3 are L1 residents (slot >= 0), never admitted
        std::vector<int32_t> s(E, -1);
        for (int e = 0; e < 4; ++e) { s[e] = e; }
        int32_t * ds = nullptr;
        CUDA_CHECK(hipMalloc(&ds, E*sizeof(int32_t)));
        CUDA_CHECK(hipMemcpy(ds, s.data(), E*sizeof(int32_t), hipMemcpyHostToDevice));
        f.slots.push_back(ds);
        uint32_t * serve = nullptr;
        if (tier) {
            // experts 36..39 live in the file: served by the worker, not admitted, address 0
            uint32_t * sh = nullptr;
            CUDA_CHECK(hipHostMalloc((void **) &sh, 2*sizeof(uint32_t), hipHostMallocMapped | hipHostMallocCoherent));
            sh[0] = 0; sh[1] = 0;
            for (int e = 36; e < 40; ++e) { sh[e/32] |= 1u << (e%32); }
            CUDA_CHECK(hipHostGetDevicePointer((void **) &serve, sh, 0));
            f.serve_host.push_back(sh);
        }
        f.serve_dev.push_back(serve);
        layers[l].slots   = ds;
        layers[l].serve   = serve;
        layers[l].enabled = l != 1;   // layer 1 gets no positions
        for (int k = 0; k < 3; ++k) {
            std::vector<uint64_t> init(E, 0);
            uint64_t * dev = nullptr;
            uint64_t * hostp = nullptr;
            if (tier) {
                CUDA_CHECK(hipHostMalloc((void **) &hostp, E*sizeof(uint64_t), hipHostMallocMapped | hipHostMallocCoherent));
                for (int e = 0; e < E; ++e) {
                    init[e] = (e >= 4 && e < 36) ? (uint64_t) (uintptr_t) (f.host_dev[k] + e*KIND_BYTES[k]) : 0;
                    hostp[e] = init[e];
                }
                CUDA_CHECK(hipHostGetDevicePointer((void **) &dev, hostp, 0));
            } else {
                CUDA_CHECK(hipMalloc(&dev, E*sizeof(uint64_t)));
                CUDA_CHECK(hipMemset(dev, 0, E*sizeof(uint64_t)));
            }
            f.tables.push_back(dev);
            f.tables_host.push_back(hostp);
            f.initial[l*3 + k] = init;
            layers[l].kind[k].tensor     = f.host_dev[k];
            layers[l].kind[k].host_slots = nullptr;
            layers[l].kind[k].addresses  = dev;
            layers[l].kind[k].stride     = KIND_BYTES[k];
            bytes[l][k] = KIND_BYTES[k];
            tails[l][k] = 512;
        }
    }
    return 0;
}

// Runs one launch of layer `layer` and checks every selected expert's bytes as a matmul reads them.
static int launch(fixture & f, ub_pool & pool, int layer, const std::vector<std::vector<int>> & rows) {
    const int n_rows = (int) rows.size();
    std::vector<int32_t> ids(n_rows*STRIDE_IDS, -1);
    for (int r = 0; r < n_rows; ++r) { for (int k = 0; k < USED; ++k) { ids[r*STRIDE_IDS + k] = rows[r][k]; } }
    int32_t * d_ids = nullptr;
    CUDA_CHECK(hipMalloc(&d_ids, ids.size()*sizeof(int32_t)));
    CUDA_CHECK(hipMemcpy(d_ids, ids.data(), ids.size()*sizeof(int32_t), hipMemcpyHostToDevice));
    pool.launch_step(layer, d_ids, n_rows, STRIDE_IDS, USED, true, 0);
    pool.launch_copy(layer, 0);
    const uint64_t * tabs[3] = { f.tables[layer*3 + 0], f.tables[layer*3 + 1], f.tables[layer*3 + 2] };
    const char * tens[3] = { f.host_dev[0], f.host_dev[1], f.host_dev[2] };
    const uint64_t ** d_tabs = nullptr;
    const char ** d_tens = nullptr;
    size_t * d_str = nullptr;
    uint32_t * d_out = nullptr;
    CUDA_CHECK(hipMalloc(&d_tabs, sizeof(tabs)));
    CUDA_CHECK(hipMalloc(&d_tens, sizeof(tens)));
    CUDA_CHECK(hipMalloc(&d_str, sizeof(KIND_BYTES)));
    CUDA_CHECK(hipMalloc(&d_out, n_rows*USED*3*sizeof(uint32_t)));
    CUDA_CHECK(hipMemcpy(d_tabs, tabs, sizeof(tabs), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_tens, tens, sizeof(tens), hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(d_str, KIND_BYTES, sizeof(KIND_BYTES), hipMemcpyHostToDevice));
    resolve_kernel<<<dim3(n_rows*USED, 3), 256>>>(d_ids, n_rows, f.slots[layer], d_tabs, d_tens, d_str, d_out);
    CUDA_CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> out(n_rows*USED*3);
    CUDA_CHECK(hipMemcpy(out.data(), d_out, out.size()*sizeof(uint32_t), hipMemcpyDeviceToHost));
    for (int r = 0; r < n_rows; ++r) {
        for (int k = 0; k < USED; ++k) {
            const int e = rows[r][k];
            if (e < 4) { continue; }   // an L1 resident, no arena in this fixture
            for (int kind = 0; kind < 3; ++kind) {
                if (f.tier && e >= 36) { continue; }   // file experts: the worker's business
                if (out[(r*USED + k)*3 + kind] != f.ref[kind*E + e]) {
                    fprintf(stderr, "layer %d row %d expert %d kind %d: bytes differ\n", layer, r, e, kind);
                    return 1;
                }
            }
        }
    }
    CUDA_CHECK(hipFree(d_ids)); CUDA_CHECK(hipFree(d_tabs)); CUDA_CHECK(hipFree(d_tens));
    CUDA_CHECK(hipFree(d_str)); CUDA_CHECK(hipFree(d_out));
    return 0;
}

static bool pooled(const fixture & f, const ub_pool & pool, int layer, int e) {
    std::vector<uint64_t> t;
    table_read(f, layer, 0, t);
    const uintptr_t a = (uintptr_t) t[e];
    return a >= (uintptr_t) f.region && a < (uintptr_t) f.region + f.region_bytes && pool.lent();
}


// The counts in mapped host memory (read_live) equal the device counters once the device is idle.
static bool live_matches(const ub_pool & pool) {
    CUDA_CHECK(hipDeviceSynchronize());
    std::array<uint64_t, ub_pool_n_counters> c{};
    uint64_t hits = 0, fills = 0, fill_bytes = 0;
    if (!pool.read_counters(c) || !pool.read_live(hits, fills, fill_bytes)) {
        return false;
    }
    const bool ok = hits == c[ub_pool_hits] && fills == c[ub_pool_fills] && fill_bytes == c[ub_pool_fill_bytes];
    if (!ok) {
        fprintf(stderr, "live counts hits %llu fills %llu (%llu B), counters %llu %llu (%llu B)\n",
            (unsigned long long) hits, (unsigned long long) fills, (unsigned long long) fill_bytes,
            (unsigned long long) c[ub_pool_hits], (unsigned long long) c[ub_pool_fills], (unsigned long long) c[ub_pool_fill_bytes]);
    }
    return ok;
}
static int run(bool tier) {
    fixture f;
    std::vector<ub_pool_layer> layers;
    std::vector<std::array<size_t, 3>> bytes, tails;
    if (build(f, tier, layers, bytes, tails)) { return 1; }
    ub_pool pool(L, E, 8);
    CHECK(pool.allocate(0, layers, bytes, tails));
    // a region that holds 2 positions per enabled layer (2 layers), filled with garbage
    size_t pos = 0;
    for (int k = 0; k < 3; ++k) { pos += (KIND_BYTES[k] + 512 + 255) & ~size_t(255); }
    f.region_bytes = 2*2*pos + 300;
    CUDA_CHECK(hipMalloc((void **) &f.region, f.region_bytes));
    CUDA_CHECK(hipMemset(f.region, 0xCD, f.region_bytes));
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    CHECK(pool.r() == 2);

    // 1 row: misses 10, 11 (admitted), 2 (L1), 12 -> over-r (later rows first: by router slot order here)
    if (launch(f, pool, 0, {{10, 11, 2, 12}})) { return 1; }
    std::array<uint64_t, ub_pool_n_counters> c{};
    CHECK(pool.read_counters(c));
    CHECK(c[ub_pool_fills] == 2 && c[ub_pool_over] == 1 && c[ub_pool_hits] == 0);
    CHECK(pooled(f, pool, 0, 10) && pooled(f, pool, 0, 11) && !pooled(f, pool, 0, 12));
    // layer 1 has no positions: nothing changes there
    if (launch(f, pool, 1, {{10, 11, 12, 13}})) { return 1; }
    CHECK(!pooled(f, pool, 1, 10));
    // hit on 10 (stamp), miss 13 evicts the oldest not used now: 11
    if (launch(f, pool, 0, {{10, 13, 3, 0}})) { return 1; }
    CHECK(pool.read_counters(c));
    CHECK(c[ub_pool_hits] == 1 && c[ub_pool_evictions] == 1);
    CHECK(pooled(f, pool, 0, 10) && pooled(f, pool, 0, 13) && !pooled(f, pool, 0, 11));
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 2, t);
        CHECK(t[11] == f.initial[0*3 + 2][11]);   // evicted entry back at its home
    }
    // 4 rows: 10 and 13 used by this launch must stay; 20, 21 are misses beyond the free positions
    if (launch(f, pool, 0, {{20, 10, 1, 2}, {13, 21, 1, 2}, {10, 13, 1, 2}, {13, 10, 1, 2}})) { return 1; }
    CHECK(pooled(f, pool, 0, 10) && pooled(f, pool, 0, 13) && !pooled(f, pool, 0, 20) && !pooled(f, pool, 0, 21));
    // 2 rows: later row first -> 31 (row 1) gets a slot before 30 (row 0) when only one is free
    if (launch(f, pool, 0, {{10, 30, 1, 2}, {31, 1, 2, 3}})) { return 1; }
    CHECK(pooled(f, pool, 0, 10) && pooled(f, pool, 0, 31) && !pooled(f, pool, 0, 30));
    if (tier) {
        // file experts are never admitted
        if (launch(f, pool, 2, {{36, 37, 38, 39}})) { return 1; }
        CHECK(!pooled(f, pool, 2, 36));
    }
    if (launch(f, pool, 2, {{5, 6, 7, 8}})) { return 1; }
    CHECK(pooled(f, pool, 2, 5) && pooled(f, pool, 2, 6));
    std::string why;
    const long checked = pool.verify(why);
    if (checked < 0) { fprintf(stderr, "verify: %s\n", why.c_str()); }
    CHECK(checked == 4);

    // reclaim: every table as before the lend, the region can be overwritten
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemset(f.region, 0xEE, f.region_bytes));
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < 3; ++k) {
            std::vector<uint64_t> t;
            table_read(f, l, k, t);
            CHECK(t == f.initial[l*3 + k]);
        }
    }
    CHECK(!pool.lent());
    // reclaimed: a launch steps nothing and reads the homes
    if (launch(f, pool, 0, {{10, 11, 12, 13}})) { return 1; }
    CHECK(!pooled(f, pool, 0, 10));
    // lend again: empty positions
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    if (launch(f, pool, 0, {{10, 22, 23, 1}})) { return 1; }
    CHECK(pool.read_counters(c));
    CHECK(live_matches(pool));
    CHECK(pooled(f, pool, 0, 22) || pooled(f, pool, 0, 10));
    CHECK(pool.verify(why) == 2);
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    for (int k = 0; k < 3; ++k) {
        std::vector<uint64_t> t;
        table_read(f, 0, k, t);
        CHECK(t == f.initial[k]);
    }
    printf("PASS: decode pool %s: lend, hits, fills, eviction, launch protection, over-r, later rows first, reclaim, re-lend, host counts\n",
        tier ? "on a coherent tier table with a serve bitmap" : "on its own address table");
    return 0;
}

static uint64_t counter(const ub_pool & pool, int which) {
    std::array<uint64_t, ub_pool_n_counters> c{};
    if (!pool.read_counters(c)) { return ~uint64_t(0); }
    return c[(size_t) which];
}

static int tables_at_home(const fixture & f) {
    for (int l = 0; l < L; ++l) {
        for (int k = 0; k < 3; ++k) {
            std::vector<uint64_t> t;
            table_read(f, l, k, t);
            if (t != f.initial[l*3 + k]) { fprintf(stderr, "layer %d kind %d: table not at home\n", l, k); return 1; }
        }
    }
    return 0;
}

static int alloc_region(fixture & f) {
    size_t pos = 0;
    for (int k = 0; k < 3; ++k) { pos += (KIND_BYTES[k] + 512 + 255) & ~size_t(255); }
    f.region_bytes = 2*2*pos + 300;   // 2 positions per enabled layer
    CUDA_CHECK(hipMalloc((void **) &f.region, f.region_bytes));
    CUDA_CHECK(hipMemset(f.region, 0xCD, f.region_bytes));
    return 0;
}

// Admission filter ghost:K: a miss enters only when the expert also missed in the layer within the last K
// steps; a step's misses are stamped after its decisions; the stamps and the step count survive a reclaim.
static int run_ghost(bool tier) {
    fixture f;
    std::vector<ub_pool_layer> layers;
    std::vector<std::array<size_t, 3>> bytes, tails;
    if (build(f, tier, layers, bytes, tails)) { return 1; }
    ub_pool pool(L, E, 8);
    CHECK(pool.allocate(0, layers, bytes, tails, /*ghost =*/ 2));
    if (alloc_region(f)) { return 1; }
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    // t1: first miss of 10, rejected
    if (launch(f, pool, 0, {{10, 1, 2, 3}})) { return 1; }
    CHECK(counter(pool, ub_pool_rejected) == 1 && counter(pool, ub_pool_fills) == 0 && !pooled(f, pool, 0, 10));
    // t2: second miss within 2 steps, admitted
    if (launch(f, pool, 0, {{10, 1, 2, 3}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == 1 && pooled(f, pool, 0, 10));
    // t3 miss 11, t4 t5 other, t6 miss 11 (3 steps later): rejected; t7 miss 11: admitted
    if (launch(f, pool, 0, {{11, 1, 2, 3}})) { return 1; }
    if (launch(f, pool, 0, {{10, 1, 2, 3}})) { return 1; }
    if (launch(f, pool, 0, {{10, 1, 2, 3}})) { return 1; }
    if (launch(f, pool, 0, {{11, 1, 2, 3}})) { return 1; }
    CHECK(!pooled(f, pool, 0, 11) && counter(pool, ub_pool_rejected) == 3);
    if (launch(f, pool, 0, {{11, 1, 2, 3}})) { return 1; }
    CHECK(pooled(f, pool, 0, 11) && counter(pool, ub_pool_fills) == 2);
    // t8: two rows miss 12 in one launch: a miss never admits itself
    if (launch(f, pool, 0, {{12, 1, 2, 3}, {12, 10, 1, 2}})) { return 1; }
    CHECK(!pooled(f, pool, 0, 12) && counter(pool, ub_pool_rejected) == 4);
    if (tier) {
        // file experts are misses (stamped) but never admitted, and not counted as rejected
        if (launch(f, pool, 2, {{36, 37, 1, 2}})) { return 1; }
        if (launch(f, pool, 2, {{36, 37, 1, 2}})) { return 1; }
        CHECK(!pooled(f, pool, 2, 36) && counter(pool, ub_pool_rejected) == 4);
    }
    // t9: miss 13; reclaim and lend again; t10: miss 13 is admitted (the window spans the lends)
    if (launch(f, pool, 0, {{13, 1, 2, 3}})) { return 1; }
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    if (tables_at_home(f)) { return 1; }
    if (launch(f, pool, 0, {{13, 1, 2, 3}})) { return 1; }     // not lent: no step
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    if (launch(f, pool, 0, {{13, 1, 2, 3}})) { return 1; }
    CHECK(pooled(f, pool, 0, 13));
    std::string why;
    CHECK(pool.verify(why) == 1);
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    if (tables_at_home(f)) { return 1; }
    printf("PASS: decode pool %s: admission filter ghost:2 (first miss rejected, second within the window admitted, "
           "outside rejected, no self-admission in a launch, window across a reclaim)\n", tier ? "on a coherent tier table" : "on its own address table");
    return 0;
}

// Ring admission: experts 30..35 are ring occupants (served, their entries name ring slots
// that hold their bytes), 36..39 file residents (served, entries 0).
static int run_ring() {
    fixture f;
    std::vector<ub_pool_layer> layers;
    std::vector<std::array<size_t, 3>> bytes, tails;
    if (build(f, true, layers, bytes, tails)) { return 1; }
    constexpr int R0 = 30, R1 = 36;
    std::vector<char *> ring(3), ring_dev(3);
    for (int k = 0; k < 3; ++k) {
        CUDA_CHECK(hipHostMalloc((void **) &ring[k], (R1 - R0)*KIND_BYTES[k], hipHostMallocMapped));
        CUDA_CHECK(hipHostGetDevicePointer((void **) &ring_dev[k], ring[k], 0));
        for (int e = R0; e < R1; ++e) { memcpy(ring[k] + (e - R0)*KIND_BYTES[k], f.host[k] + e*KIND_BYTES[k], KIND_BYTES[k]); }
    }
    auto ring_addr = [&](int k, int e) { return (uint64_t) (uintptr_t) (ring_dev[k] + (e - R0)*KIND_BYTES[k]); };
    auto publish_ring = [&](int l, int e) {   // what the worker publishes for a ring occupant
        for (int k = 0; k < 3; ++k) { f.tables_host[l*3 + k][e] = ring_addr(k, e); }
    };
    for (int l = 0; l < L; ++l) {
        for (int e = R0; e < R1; ++e) {
            f.serve_host[l][e/32] |= 1u << (e%32);
            publish_ring(l, e);
        }
    }
    ub_pool pool(L, E, 8);
    CHECK(pool.allocate(0, layers, bytes, tails));
    CHECK(pool.enable_ring() && pool.ring() && pool.ring_words() == 2);
    if (alloc_region(f)) { return 1; }
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    auto bit = [&](int l, int e) {
        CUDA_CHECK(hipDeviceSynchronize());
        uint32_t dev[2];
        CUDA_CHECK(hipMemcpy(dev, pool.ring_bits_device() + l*2, sizeof(dev), hipMemcpyDeviceToHost));
        const bool h = (pool.ring_bits_host()[l*2 + e/32] & (1u << (e%32))) != 0;
        const bool d = (dev[e/32] & (1u << (e%32))) != 0;
        return h == d ? (h ? 1 : 0) : -1;   // -1: the two copies disagree
    };
    auto entries = [&](int l, int e, bool zero) {   // all three kinds 0, or all at the ring slot
        for (int k = 0; k < 3; ++k) {
            std::vector<uint64_t> t;
            table_read(f, l, k, t);
            if (t[e] != (zero ? 0 : ring_addr(k, e))) { return false; }
        }
        return true;
    };
    // ring occupants are admitted (their copy reads the ring slot), a file resident is not
    if (launch(f, pool, 0, {{30, 31, 36, 1}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == 2 && counter(pool, ub_pool_ring_fills) == 2);
    CHECK(pooled(f, pool, 0, 30) && pooled(f, pool, 0, 31) && !pooled(f, pool, 0, 36));
    CHECK(bit(0, 30) == 1 && bit(0, 31) == 1 && bit(0, 36) == 0 && bit(2, 30) == 0);
    // the ring reuses the slot of 30 meanwhile: the pooled copy is unaffected
    for (int k = 0; k < 3; ++k) { memset(ring[k] + (30 - R0)*KIND_BYTES[k], 0x77, KIND_BYTES[k]); }
    if (launch(f, pool, 0, {{30, 1, 2, 3}})) { return 1; }
    CHECK(counter(pool, ub_pool_hits) == 1);
    // evicted ring owners get 0 (not the saved slot), and leave the bitmap
    if (launch(f, pool, 0, {{32, 33, 1, 2}})) { return 1; }
    CHECK(entries(0, 30, true) && entries(0, 31, true) && pooled(f, pool, 0, 32) && pooled(f, pool, 0, 33));
    CHECK(bit(0, 30) == 0 && bit(0, 31) == 0 && bit(0, 32) == 1 && bit(0, 33) == 1);
    CHECK(counter(pool, ub_pool_ring_fills) == 4);
    // host residents take the positions: the marks go, their eviction restores the home
    if (launch(f, pool, 0, {{10, 11, 1, 2}})) { return 1; }
    CHECK(entries(0, 32, true) && entries(0, 33, true) && bit(0, 32) == 0 && bit(0, 33) == 0);
    CHECK(counter(pool, ub_pool_ring_fills) == 4);
    if (launch(f, pool, 0, {{34, 10, 1, 2}})) { return 1; }
    CHECK(pooled(f, pool, 0, 34) && pooled(f, pool, 0, 10) && !pooled(f, pool, 0, 11) && bit(0, 34) == 1);
    CHECK(counter(pool, ub_pool_ring_fills) == 5);
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 1, t);
        CHECK(t[11] == f.initial[0*3 + 1][11]);
    }
    std::string why;
    const long checked = pool.verify(why);
    if (checked < 0) { fprintf(stderr, "verify: %s\n", why.c_str()); }
    CHECK(checked == 2);
    // reclaim: the ring owner gets 0, the host resident its home, the bitmap is empty
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(entries(0, 34, true) && bit(0, 34) == 0);
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        CHECK(t[10] == f.initial[0*3 + 0][10]);
    }
    // the worker published 34 again; with the ring admission off a lend admits host residents only (the old rule)
    publish_ring(0, 34);
    pool.set_ring(false);
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    const uint64_t fills = counter(pool, ub_pool_fills);
    if (launch(f, pool, 0, {{34, 35, 12, 1}})) { return 1; }
    CHECK(!pooled(f, pool, 0, 34) && !pooled(f, pool, 0, 35) && pooled(f, pool, 0, 12));
    CHECK(counter(pool, ub_pool_fills) == fills + 1 && counter(pool, ub_pool_ring_fills) == 5 && bit(0, 34) == 0);
    CHECK(entries(0, 34, false));
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    // on again: a ring occupant whose entries are 0 (the ring dropped it) is not admitted
    pool.set_ring(true);
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2);
    for (int k = 0; k < 3; ++k) { f.tables_host[0*3 + k][35] = 0; }
    if (launch(f, pool, 0, {{35, 34, 1, 2}})) { return 1; }
    CHECK(!pooled(f, pool, 0, 35) && pooled(f, pool, 0, 34) && counter(pool, ub_pool_ring_fills) == 6);
    CHECK(live_matches(pool));
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    CHECK(entries(0, 34, true) && bit(0, 34) == 0);
    for (int k = 0; k < 3; ++k) { CUDA_CHECK(hipHostFree(ring[k])); }
    printf("PASS: decode pool ring admission: ring occupants admitted and copied from the ring, file residents not, "
           "the bitmap follows fills, evictions and the reclaim, ring owners go back as 0, off = host residents only\n");
    return 0;
}

// Fixed positions (the LRU part of L1, --expert-l1-lru-pct): positions in "L1 slots" before those of a lent region. Experts
// 30..35 are ring occupants as in run_ring.
static int run_fixed() {
    fixture f;
    std::vector<ub_pool_layer> layers;
    std::vector<std::array<size_t, 3>> bytes, tails;
    if (build(f, true, layers, bytes, tails)) { return 1; }
    constexpr int R0 = 30, R1 = 36, SLOTS = 6;
    std::vector<char *> ring(3), ring_dev(3), arena(3);
    for (int k = 0; k < 3; ++k) {
        CUDA_CHECK(hipHostMalloc((void **) &ring[k], (R1 - R0)*KIND_BYTES[k], hipHostMallocMapped));
        CUDA_CHECK(hipHostGetDevicePointer((void **) &ring_dev[k], ring[k], 0));
        for (int e = R0; e < R1; ++e) { memcpy(ring[k] + (e - R0)*KIND_BYTES[k], f.host[k] + e*KIND_BYTES[k], KIND_BYTES[k]); }
        // the "L1 arena" of a kind: slot-major, no tail per slot
        CUDA_CHECK(hipMalloc((void **) &arena[k], SLOTS*KIND_BYTES[k] + 512));
        CUDA_CHECK(hipMemset(arena[k], 0xAB, SLOTS*KIND_BYTES[k] + 512));
    }
    auto ring_addr = [&](int k, int e) { return (uint64_t) (uintptr_t) (ring_dev[k] + (e - R0)*KIND_BYTES[k]); };
    for (int l = 0; l < L; ++l) {
        for (int e = R0; e < R1; ++e) {
            f.serve_host[l][e/32] |= 1u << (e%32);
            for (int k = 0; k < 3; ++k) { f.tables_host[l*3 + k][e] = ring_addr(k, e); }
        }
    }
    auto slot_pos = [&](int slot) {
        ub_fixed_pos x;
        for (int k = 0; k < 3; ++k) { x.addr[k] = (uint64_t) (uintptr_t) (arena[k] + slot*KIND_BYTES[k]); }
        return x;
    };
    auto in_arena = [&](int l, int e) {
        std::vector<uint64_t> t;
        table_read(f, l, 0, t);
        const uintptr_t a = (uintptr_t) t[e];
        return a >= (uintptr_t) arena[0] && a < (uintptr_t) arena[0] + SLOTS*KIND_BYTES[0];
    };
    auto bit = [&](ub_pool & pool, int l, int e) {
        CUDA_CHECK(hipDeviceSynchronize());
        uint32_t dev[2];
        CUDA_CHECK(hipMemcpy(dev, pool.ring_bits_device() + l*2, sizeof(dev), hipMemcpyDeviceToHost));
        const bool h = (pool.ring_bits_host()[l*2 + e/32] & (1u << (e%32))) != 0;
        const bool d = (dev[e/32] & (1u << (e%32))) != 0;
        return h == d ? (h ? 1 : 0) : -1;
    };
    // r_cap 8, of which 4 for a region: at most 4 fixed positions per layer
    ub_pool pool(L, E, 8, 4);
    CHECK(pool.allocate(0, layers, bytes, tails));
    CHECK(pool.enable_ring());
    {
        std::vector<std::vector<ub_fixed_pos>> pos(L);
        pos[0] = { slot_pos(0), slot_pos(1), slot_pos(2) };
        pos[2] = { slot_pos(3), slot_pos(4) };
        CHECK(pool.set_fixed(pos, {}));
        std::vector<std::vector<ub_fixed_pos>> bad(L);
        bad[1] = { slot_pos(5) };   // layer 1 takes no positions
        CHECK(!pool.set_fixed(bad, {}));
        bad[1].clear();
        bad[0] = { slot_pos(0), slot_pos(1), slot_pos(2), slot_pos(3), slot_pos(4) };   // more than r_cap - r_base
        CHECK(!pool.set_fixed(bad, {}));
    }
    CHECK(pool.fixed(0) == 3 && pool.fixed(2) == 2 && pool.fixed_max() == 3 && !pool.lent());
    // not lent: the fixed positions take misses (a ring occupant and a host resident), the copies are read
    if (launch(f, pool, 0, {{30, 10, 2, 1}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == 2 && counter(pool, ub_pool_ring_fills) == 1 && in_arena(0, 30) && in_arena(0, 10));
    CHECK(bit(pool, 0, 30) == 1);
    // the ring reuses the slot of 30 (the L2 evicted it): the copy stays and is read (launch checks the bytes)
    for (int k = 0; k < 3; ++k) { memset(ring[k] + (30 - R0)*KIND_BYTES[k], 0x77, KIND_BYTES[k]); }
    if (launch(f, pool, 0, {{30, 10, 1, 2}})) { return 1; }
    CHECK(counter(pool, ub_pool_hits) == 2 && counter(pool, ub_pool_fixed_hits) == 2);
    // a third miss fills the last fixed position; a fourth evicts the oldest (10 is used by every launch)
    if (launch(f, pool, 0, {{11, 10, 1, 2}})) { return 1; }
    if (launch(f, pool, 0, {{12, 10, 1, 2}})) { return 1; }
    CHECK(in_arena(0, 12) && in_arena(0, 11) && in_arena(0, 10) && !in_arena(0, 30));
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        CHECK(t[30] == 0 && bit(pool, 0, 30) == 0);   // a ring owner leaves as 0 (its old ring slot holds something else)
    }
    // a region after the fixed positions: an expert in a fixed position is a hit, never a second copy
    if (alloc_region(f)) { return 1; }
    CHECK(pool.lend(f.region, f.region_bytes, 0) == 2 && pool.r() == 2);
    const uint64_t fills0 = counter(pool, ub_pool_fills);
    if (launch(f, pool, 0, {{10, 11, 12, 31}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == fills0 + 1 && pooled(f, pool, 0, 31) && in_arena(0, 10));
    if (launch(f, pool, 0, {{13, 14, 10, 31}})) { return 1; }   // 13, 14 evict 11, 12 (fixed) or the region's empty slot
    std::string why;
    long checked = pool.verify(why);
    if (checked < 0) { fprintf(stderr, "verify: %s\n", why.c_str()); }
    CHECK(checked == 5);
    // layer 2 too: its region positions follow its 2 fixed ones
    if (launch(f, pool, 2, {{5, 6, 7, 32}})) { return 1; }
    CHECK(in_arena(2, 5) && in_arena(2, 6));
    // reclaim: the region's owners go home, the fixed ones stay (and stay marked)
    std::vector<int> fixed_owned;
    for (int e = 4; e < 36; ++e) { if (in_arena(0, e)) { fixed_owned.push_back(e); } }
    CHECK(fixed_owned.size() == 3);
    pool.reclaim(0);
    CUDA_CHECK(hipDeviceSynchronize());
    CUDA_CHECK(hipMemset(f.region, 0xEE, f.region_bytes));
    for (int e : fixed_owned) { CHECK(in_arena(0, e)); }
    for (int e = 4; e < 36; ++e) {
        if (std::find(fixed_owned.begin(), fixed_owned.end(), e) != fixed_owned.end() || in_arena(2, e)) { continue; }
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        CHECK(t[e] == f.initial[0][e] || (e >= R0 && (t[e] == 0 || t[e] == ring_addr(0, e))));
    }
    for (int e : fixed_owned) { if (e >= R0) { CHECK(bit(pool, 0, e) == 1); } }
    if (launch(f, pool, 0, {{fixed_owned[0], fixed_owned[1], 1, 2}})) { return 1; }
    checked = pool.verify(why);
    CHECK(checked == 5);   // 3 + 2 fixed copies
    // a lend moves a fixed copy to another slot (read_fixed, D2D, set_fixed with the new address) and drops another
    std::vector<std::vector<ub_fixed_pos>> pos;
    std::vector<uint32_t> steps;
    CHECK(pool.read_fixed(pos, steps) && pos[0].size() == 3 && steps[0] > 0);
    std::vector<std::pair<int, ub_fixed_pos>> drop = { { 0, pos[0][2] } };
    ub_fixed_pos moved = pos[0][0];
    for (int k = 0; k < 3; ++k) {
        CUDA_CHECK(hipMemcpy(arena[k] + 5*KIND_BYTES[k], arena[k] + 0*KIND_BYTES[k], KIND_BYTES[k], hipMemcpyDeviceToDevice));
        moved.addr[k] = (uint64_t) (uintptr_t) (arena[k] + 5*KIND_BYTES[k]);
    }
    std::vector<std::vector<ub_fixed_pos>> next(L);
    next[0] = { moved, pos[0][1] };
    next[2] = pos[2];
    CHECK(pool.set_fixed(next, drop));
    CHECK(pool.fixed(0) == 2 && in_arena(0, moved.owner) && !in_arena(0, pos[0][2].owner));
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        CHECK(t[moved.owner] == moved.addr[0]);
        const int d = pos[0][2].owner;
        CHECK(t[d] == (pos[0][2].ring ? 0 : f.initial[0][d]));
    }
    CUDA_CHECK(hipMemset(arena[0], 0x11, KIND_BYTES[0]));   // the old slot is the lend's now
    if (launch(f, pool, 0, {{moved.owner, pos[0][1].owner, 1, 2}})) { return 1; }
    CHECK(pool.verify(why) == 4);
    // drop everything: every table at home (ring owners 0), the bitmap empty
    CHECK(pool.read_fixed(pos, steps));
    drop.clear();
    for (int l = 0; l < L; ++l) {
        for (ub_fixed_pos & x : pos[(size_t) l]) {
            if (x.owner >= 0) { drop.push_back({ l, x }); x.owner = -1; x.ring = 0; }
        }
    }
    CHECK(pool.set_fixed(pos, drop));
    for (int l = 0; l < L; ++l) {
        for (int e = 4; e < 36; ++e) {
            std::vector<uint64_t> t;
            table_read(f, l, 0, t);
            CHECK(!in_arena(l, e) && (t[e] == f.initial[(size_t) l*3][(size_t) e] || (e >= R0 && (t[e] == 0 || t[e] == ring_addr(0, e)))));
            if (e >= R0) { CHECK(bit(pool, l, e) == 0); }
        }
    }
    CHECK(live_matches(pool));
    for (int k = 0; k < 3; ++k) { CUDA_CHECK(hipHostFree(ring[k])); CUDA_CHECK(hipFree(arena[k])); }
    printf("PASS: decode pool fixed positions: misses fill them while nothing is lent, a ring copy survives its ring slot's reuse, "
           "LRU eviction, a region after them without duplicates, the reclaim keeps them, a move and a drop through "
           "read_fixed/set_fixed, a full drop restores every table\n");
    return 0;
}

// The copies of the fixed positions through an install (the controller's x_release_locked / x_restore_locked): set aside
// (entries home), the tables published again (an expert becomes static, a host home moves, a ring slot is reused), then
// check_fixed decides and set_fixed takes the copies back with their new homes.
static int run_keep() {
    fixture f;
    std::vector<ub_pool_layer> layers;
    std::vector<std::array<size_t, 3>> bytes, tails;
    if (build(f, true, layers, bytes, tails)) { return 1; }
    constexpr int R0 = 30, R1 = 36, SLOTS = 4;
    std::vector<char *> ring(3), ring_dev(3), arena(3), moved(3), moved_dev(3);
    for (int k = 0; k < 3; ++k) {
        CUDA_CHECK(hipHostMalloc((void **) &ring[k], (R1 - R0)*KIND_BYTES[k], hipHostMallocMapped));
        CUDA_CHECK(hipHostGetDevicePointer((void **) &ring_dev[k], ring[k], 0));
        for (int e = R0; e < R1; ++e) { memcpy(ring[k] + (e - R0)*KIND_BYTES[k], f.host[k] + e*KIND_BYTES[k], KIND_BYTES[k]); }
        CUDA_CHECK(hipMalloc((void **) &arena[k], SLOTS*KIND_BYTES[k] + 512));
        CUDA_CHECK(hipMemset(arena[k], 0xAB, SLOTS*KIND_BYTES[k] + 512));
        // the new host home of expert 11 after the install
        CUDA_CHECK(hipHostMalloc((void **) &moved[k], KIND_BYTES[k], hipHostMallocMapped));
        memcpy(moved[k], f.host[k] + 11*KIND_BYTES[k], KIND_BYTES[k]);
        CUDA_CHECK(hipHostGetDevicePointer((void **) &moved_dev[k], moved[k], 0));
    }
    for (int l = 0; l < L; ++l) {
        for (int e = R0; e < R1; ++e) {
            f.serve_host[l][e/32] |= 1u << (e%32);
            for (int k = 0; k < 3; ++k) { f.tables_host[l*3 + k][e] = (uint64_t) (uintptr_t) (ring_dev[k] + (e - R0)*KIND_BYTES[k]); }
        }
    }
    auto slot_pos = [&](int slot) {
        ub_fixed_pos x;
        for (int k = 0; k < 3; ++k) { x.addr[k] = (uint64_t) (uintptr_t) (arena[k] + slot*KIND_BYTES[k]); }
        return x;
    };
    ub_pool pool(L, E, 8, 4);
    CHECK(pool.allocate(0, layers, bytes, tails));
    CHECK(pool.enable_ring());
    {
        std::vector<std::vector<ub_fixed_pos>> pos(L);
        pos[0] = { slot_pos(0), slot_pos(1), slot_pos(2), slot_pos(3) };
        CHECK(pool.set_fixed(pos, {}));
    }
    // copies of 10, 11 (host homes), 30, 31 (ring occupants)
    if (launch(f, pool, 0, {{10, 11, 30, 31}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == 4 && counter(pool, ub_pool_ring_fills) == 2);
    // set aside: the entries go home, the positions stay empty
    std::vector<std::vector<ub_fixed_pos>> saved, pos;
    std::vector<uint32_t> steps;
    CHECK(pool.read_fixed(saved, steps));
    pos = saved;
    std::vector<std::pair<int, ub_fixed_pos>> drop;
    for (ub_fixed_pos & x : pos[0]) { if (x.owner >= 0) { drop.push_back({ 0, x }); x.owner = -1; x.ring = 0; } }
    CHECK(drop.size() == 4 && pool.set_fixed(pos, drop));
    // the install: 10 becomes an L1 resident, 11's host home moves, 31's ring slot is reused (entries 0)
    {
        const int32_t s10 = 5;
        CUDA_CHECK(hipMemcpy(f.slots[0] + 10, &s10, sizeof(s10), hipMemcpyHostToDevice));
        for (int k = 0; k < 3; ++k) {
            f.tables_host[k][11] = (uint64_t) (uintptr_t) moved_dev[k];
            f.tables_host[k][31] = 0;
            memset(ring[k] + (31 - R0)*KIND_BYTES[k], 0x55, KIND_BYTES[k]);
        }
    }
    std::vector<std::pair<int, int>> q;
    for (const ub_fixed_pos & x : saved[0]) { q.push_back({ 0, x.owner }); }
    std::vector<int> state;
    std::vector<std::array<uint64_t, 3>> home;
    CHECK(pool.check_fixed(q, state, home));
    for (size_t i = 0; i < q.size(); ++i) {
        const int e = q[i].second;
        if (e == 10) { CHECK(state[i] == ub_fixed_static); }
        if (e == 11) { CHECK(state[i] == ub_fixed_home && home[i][0] == (uint64_t) (uintptr_t) moved_dev[0] &&
                             home[i][2] == (uint64_t) (uintptr_t) moved_dev[2]); }
        if (e == 30 || e == 31) { CHECK(state[i] == ub_fixed_ring && home[i][0] == 0); }
    }
    // back: the stamps kept, 10 leaves (its slot stays, empty)
    std::vector<std::vector<ub_fixed_pos>> next(L);
    for (size_t i = 0; i < q.size(); ++i) {
        ub_fixed_pos x = saved[0][i];
        if (state[i] == ub_fixed_static || state[i] == ub_fixed_none) {
            x.owner = -1;
            x.ring  = 0;
        } else {
            x.ring = state[i] == ub_fixed_ring ? 1 : 0;
            for (int k = 0; k < 3; ++k) { x.home[k] = home[i][(size_t) k]; }
        }
        next[0].push_back(x);
    }
    CHECK(pool.set_fixed(next, {}));
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        const uintptr_t a0 = (uintptr_t) arena[0], a1 = a0 + SLOTS*KIND_BYTES[0];
        for (int e : { 11, 30, 31 }) { CHECK((uintptr_t) t[e] >= a0 && (uintptr_t) t[e] < a1); }
        CHECK((uintptr_t) t[10] < a0 || (uintptr_t) t[10] >= a1);
    }
    CHECK(pool.read_fixed(pos, steps));
    for (size_t i = 0; i < pos[0].size(); ++i) { if (pos[0][i].owner >= 0) { CHECK(pos[0][i].stamp == saved[0][i].stamp); } }
    // 31's ring slot holds something else now: the kept copy is read (launch checks the bytes); hits, no fills
    const uint64_t fills0 = counter(pool, ub_pool_fills), hits0 = counter(pool, ub_pool_fixed_hits);
    if (launch(f, pool, 0, {{11, 30, 31, 1}})) { return 1; }
    CHECK(counter(pool, ub_pool_fills) == fills0 && counter(pool, ub_pool_fixed_hits) == hits0 + 3);
    std::string why;
    CHECK(pool.verify(why) == 3);
    // dropped later: 11 goes to its new home, the ring copies to 0
    CHECK(pool.read_fixed(pos, steps));
    drop.clear();
    for (ub_fixed_pos & x : pos[0]) { if (x.owner >= 0) { drop.push_back({ 0, x }); x.owner = -1; x.ring = 0; } }
    CHECK(pool.set_fixed(pos, drop));
    {
        std::vector<uint64_t> t;
        table_read(f, 0, 0, t);
        CHECK(t[11] == (uint64_t) (uintptr_t) moved_dev[0] && t[30] == 0 && t[31] == 0);
    }
    // without the ring admission a served expert has no place in a fixed position
    ub_pool plain(L, E, 8, 4);
    CHECK(plain.allocate(0, layers, bytes, tails));
    CHECK(plain.check_fixed({ { 0, 31 }, { 0, 12 }, { 0, 2 } }, state, home));
    CHECK(state[0] == ub_fixed_none && state[1] == ub_fixed_home && state[2] == ub_fixed_static);
    for (int k = 0; k < 3; ++k) { CUDA_CHECK(hipHostFree(ring[k])); CUDA_CHECK(hipFree(arena[k])); CUDA_CHECK(hipHostFree(moved[k])); }
    printf("PASS: decode pool fixed positions through an install: set aside, a static expert leaves, a moved host home is "
           "taken over, ring copies stay after their ring slot's reuse, stamps kept, hits after, drops to the new homes\n");
    return 0;
}

// `bench`: the cost of a decode launch's step and copy kernels against the number of positions per layer (the LRU part
// of L1 widens the pool from about 10 to tens of positions). The real kernels on a DS4.1F-like shape (40 layers, 384
// experts, top-6, one row, ghost:8), with positions of 256-byte kinds so that the copies cost little; the ids follow a
// Zipf law per layer. Each position count is measured against an idle pool (no position: the step returns at once) in
// alternating chunks inside one process. Prints the median of the chunk pairs, in us per (step + copy) launch.
static int bench() {
    constexpr int BL = 40, BE = 384, BU = 6, TOKENS = 400, CHUNKS = 24;
    const size_t KB = 256;
    std::mt19937 rng(11);
    std::vector<int32_t> ids((size_t) TOKENS*BL*BU);
    std::vector<double> w(BE);
    for (int e = 0; e < BE; ++e) { w[e] = 1.0/std::pow((double) (e + 1), 1.1); }
    for (int t = 0; t < TOKENS; ++t) {
        for (int l = 0; l < BL; ++l) {
            std::vector<double> ww = w;
            std::shuffle(ww.begin(), ww.end(), std::mt19937(1000 + l));   // a different hot set per layer
            std::discrete_distribution<int> d(ww.begin(), ww.end());
            std::vector<int> row;
            while ((int) row.size() < BU) { const int e = d(rng); if (std::find(row.begin(), row.end(), e) == row.end()) { row.push_back(e); } }
            for (int k = 0; k < BU; ++k) { ids[((size_t) t*BL + l)*BU + k] = row[k]; }
        }
    }
    int32_t * d_ids = nullptr;
    CUDA_CHECK(hipMalloc(&d_ids, ids.size()*sizeof(int32_t)));
    CUDA_CHECK(hipMemcpy(d_ids, ids.data(), ids.size()*sizeof(int32_t), hipMemcpyHostToDevice));
    char * host = nullptr, * host_dev = nullptr;
    CUDA_CHECK(hipHostMalloc((void **) &host, BE*KB, hipHostMallocMapped));
    CUDA_CHECK(hipHostGetDevicePointer((void **) &host_dev, host, 0));
    int32_t * slots = nullptr;
    CUDA_CHECK(hipMalloc(&slots, BE*sizeof(int32_t)));
    CUDA_CHECK(hipMemset(slots, 0xFF, BE*sizeof(int32_t)));
    std::vector<ub_pool_layer> layers(BL);
    std::vector<std::array<size_t, 3>> bytes(BL, { KB, KB, KB }), tails(BL, { 0, 0, 0 });
    for (int l = 0; l < BL; ++l) {
        for (int k = 0; k < 3; ++k) {
            uint64_t * t = nullptr;
            CUDA_CHECK(hipMalloc(&t, BE*sizeof(uint64_t)));
            CUDA_CHECK(hipMemset(t, 0, BE*sizeof(uint64_t)));
            layers[l].kind[k].tensor = host_dev;
            layers[l].kind[k].addresses = t;
            layers[l].kind[k].stride = KB;
        }
        layers[l].slots = slots;
        layers[l].enabled = 1;
    }
    const int counts[] = { 10, 40, 100, 190 };
    char * arena = nullptr;
    CUDA_CHECK(hipMalloc(&arena, (size_t) BL*190*3*KB));
    hipStream_t st;
    CUDA_CHECK(hipStreamCreate(&st));
    printf("bench: step + copy per launch (40 layers x %d tokens per chunk, 1 row, top-%d, ghost:8), median of %d chunk pairs\n",
        TOKENS, BU, CHUNKS);
    for (int r : counts) {
        ub_pool pool(BL, BE, 64 + 190, 64);
        if (!pool.allocate(0, layers, bytes, tails, 8)) { return 1; }
        std::vector<std::vector<ub_fixed_pos>> on(BL), off(BL);
        for (int l = 0; l < BL; ++l) {
            for (int p = 0; p < r; ++p) {
                ub_fixed_pos x;
                for (int k = 0; k < 3; ++k) { x.addr[k] = (uint64_t) (uintptr_t) (arena + (((size_t) l*190 + p)*3 + k)*KB); }
                on[l].push_back(x);
            }
        }
        auto run_chunk = [&]() {
            const auto t0 = std::chrono::steady_clock::now();
            for (int t = 0; t < TOKENS; ++t) {
                for (int l = 0; l < BL; ++l) {
                    pool.launch_step(l, d_ids + ((size_t) t*BL + l)*BU, 1, BU, BU, true, st);
                    pool.launch_copy(l, st, 1, BU);
                }
            }
            CUDA_CHECK(hipStreamSynchronize(st));
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count()/(TOKENS*BL);
        };
        std::vector<double> d_on, d_off, diff;
        for (int c = 0; c < CHUNKS + 2; ++c) {
            if (!pool.set_fixed(on, {})) { return 1; }   // the positions start empty each chunk: fills and hits
            const double a = run_chunk();
            std::vector<std::vector<ub_fixed_pos>> now;
            std::vector<uint32_t> steps;
            pool.read_fixed(now, steps);
            std::vector<std::pair<int, ub_fixed_pos>> drop;
            for (int l = 0; l < BL; ++l) { for (auto & x : now[l]) { if (x.owner >= 0) { drop.push_back({ l, x }); } } }
            if (!pool.set_fixed(off, drop)) { return 1; }
            const double b = run_chunk();
            if (c >= 2) { d_on.push_back(a); d_off.push_back(b); diff.push_back(a - b); }
        }
        std::array<uint64_t, ub_pool_n_counters> cn{};
        pool.read_counters(cn);
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
        printf("bench: %3d positions: %.2f us per launch, idle pool %.2f us, difference %.2f us [%.2f .. %.2f]; hits %.1f %%, "
               "fills %llu, rejected %llu\n", r, med(d_on), med(d_off), med(diff), *std::min_element(diff.begin(), diff.end()),
            *std::max_element(diff.begin(), diff.end()),
            100.0*(double) cn[ub_pool_hits]/(double) ((size_t) (CHUNKS + 2)*TOKENS*BL*BU), (unsigned long long) cn[ub_pool_fills],
            (unsigned long long) cn[ub_pool_rejected]);
    }
    return 0;
}

int main(int argc, char ** argv) {
    CUDA_CHECK(hipSetDevice(0));
    if (argc > 1 && std::string(argv[1]) == "bench") { return bench(); }
    if (run(false)) { return 1; }
    if (run(true)) { return 1; }
    if (run_ghost(false)) { return 1; }
    if (run_ghost(true)) { return 1; }
    if (run_ring()) { return 1; }
    if (run_fixed()) { return 1; }
    if (run_keep()) { return 1; }
    printf("ALL PASS\n");
    return 0;
}
