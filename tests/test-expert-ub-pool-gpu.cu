// Standalone HIP fixture of the decode pool (expert-ub-pool.cuh): lend, step, copy first, eviction,
// over-r, reclaim and the address tables, with a lookup that resolves every selected expert the way
// ggml_cuda_expert_cache_select does and compares the bytes it reaches with the expert's home.
#include "expert-ub-pool.cu"

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

int main() {
    CUDA_CHECK(hipSetDevice(0));
    if (run(false)) { return 1; }
    if (run(true)) { return 1; }
    if (run_ghost(false)) { return 1; }
    if (run_ghost(true)) { return 1; }
    printf("ALL PASS\n");
    return 0;
}
