#include "expert-ub-pool.cuh"

#if defined(GGML_USE_HIP)

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ggml_cuda_expert {

static int ub_env_int(const char * name, int fallback) {
    const char * v = getenv(name);
    return v == nullptr || *v == '\0' ? fallback : atoi(v);
}

ub_pool_config ub_pool_config_from_env() {
    ub_pool_config c;
    c.enabled   = true;   // regions of at least 256 MiB, 64 MiB of margin, 64 positions per layer, the decode rows, ghost:8
    c.log       = ub_env_int("RANMA_EXPERT_UB_POOL_LOG", 0) != 0;
    return c;
}

// ---- kernels ---------------------------------------------------------------------------------------

static constexpr uint32_t ub_never = 0xFFFFFFFFu;   // last miss of an expert that never missed

struct ub_step_args {
    const int32_t * ids;
    int n_rows, row_stride, n_used, n_experts, layer, r_cap, admit, ghost;
    const ub_pool_layer * layers;
    ub_pool_meta * meta;
    const uint64_t * addr;   // [P][3]
    uint64_t * home;         // [P][3]
    int32_t * owner;
    uint32_t * stamp;
    int16_t * pos_of;        // [L][E]
    ub_pool_copy * copy;
    int32_t * count;
    unsigned long long * counters;
    uint32_t * last_miss;    // [E] of the layer; used only when ghost > 0
    ub_pool_live * live;     // [L] mapped host memory, or null
    uint64_t pos_bytes;
    ub_pf_dev pf;            // prefetch staging (pf_on)
    int pf_on;
    int32_t * ringpos;       // [P] ring admission (enable_ring), else null
    uint32_t * ringbits;     // [L][words] device copy
    uint32_t * ringbits_host;   // [L][words] mapped host memory (device address)
    int words;
    uint32_t * prof_counts;                       // the layer's profile counts, or null: no aggregation here
    size_t prof_bank_stride;
    uint32_t prof_n_banks;
    const profile_selection * prof_selection;
};

// ring admission: the layer's bitmap of positions admitted from the ring, into both copies (after the layer's
// positions changed; every thread of the block calls it, after a barrier)
static __device__ void ub_ring_bits(const ub_step_args & a, const int16_t * pos_of, const int off, const int r) {
    for (int w = threadIdx.x; w < a.words; w += blockDim.x) {
        uint32_t m = 0;
        for (int b = 0; b < 32; ++b) {
            const int e = w*32 + b;
            const int p = e < a.n_experts ? pos_of[e] : -1;
            m |= (p >= 0 && p < r && a.ringpos[off + p] != 0) ? 1u << b : 0u;
        }
        a.ringbits[size_t(a.layer)*a.words + w] = m;
        a.ringbits_host[size_t(a.layer)*a.words + w] = m;
    }
    __threadfence_system();
}

// The admission rule of the step kernel, for an expert not in a position: not an L1
// resident, not one the SSD worker serves, and with a host home.
static __device__ __forceinline__ bool ub_admissible(const ub_pool_layer & L, const int id) {
    if (L.slots[id] >= 0) {
        return false;
    }
    if (L.serve != nullptr && (L.serve[id/32] & (1u << (id%32))) != 0) {
        return false;   // a file resident or ring occupant: only the SSD worker places it
    }
    if (L.kind[0].host_slots != nullptr && L.kind[0].addresses != nullptr && L.kind[0].addresses[id] == 0 &&
            L.kind[0].host_slots[id] < 0) {
        return false;   // no host home
    }
    return true;
}

// Ring admission: a miss the SSD worker serves whose three table entries name a ring slot (not 0). An L1 resident
// or a file resident (entries 0) is not admitted.
static __device__ __forceinline__ bool ub_ring_admissible(const ub_pool_layer & L, const int id) {
    if (L.slots[id] >= 0 || L.serve == nullptr || (L.serve[id/32] & (1u << (id%32))) == 0) {
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if (L.kind[k].addresses == nullptr || L.kind[k].addresses[id] == 0) {
            return false;
        }
    }
    return true;
}

// The recent pool's ghost term (RANMA_EXPERT_RECENT_FILTER, 2026-09): `last` = the step of the expert's
// last miss, `t` = the step now.
static __device__ __forceinline__ bool ub_ghost_ok(const uint32_t last, const uint32_t t, const uint32_t k) {
    return last != ub_never && last != t && t - last <= k;
}

static __device__ __forceinline__ bool ub_eligible(const ub_step_args & a, const int off, const uint32_t t, const int p) {
    const int32_t o = a.owner[off + p];
    if (o == ub_pending || (o >= 0 && a.pf_on && a.pf.inplace && a.pf.pdown[off + p] != 0)) {
        return false;   // a copy may still write it
    }
    return o < 0 || a.stamp[off + p] != t;
}

// inplace prefetch: one thread settles the pending positions of a layer (see expert-ub-pool.cuh).
static __device__ void ub_pf_settle(const ub_pool_layer & L, const ub_pf_dev & pf, const int off, const int r, const uint32_t t,
        int32_t * owner, uint32_t * stamp, int16_t * pos_of, uint64_t * home, const uint64_t * addr) {
    for (int p = 0; p < r; ++p) {
        const int q = off + p;
        if (owner[q] != ub_pending) {
            continue;
        }
        const uint32_t g = pf.pend_gen[q];
        if (__hip_atomic_load(&pf.done[size_t(q)*2], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != g) {
            continue;   // up/gate not finished (or never issued)
        }
        const bool down = __hip_atomic_load(&pf.done[size_t(q)*2 + 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == g;
        const int e = pf.pend_e[q];
        if (pos_of[e] >= 0 || !ub_admissible(L, e)) {
            if (down) {
                owner[q] = -1;   // complete and not needed; until the down copy finished it stays pending
            }
            continue;
        }
        for (int kind = 0; kind < 3; ++kind) {
            uint64_t * table = L.kind[kind].addresses;
            home[size_t(q)*3 + kind] = table != nullptr ? table[e] : 0;
            if (table != nullptr && (kind < 2 || down)) {
                table[e] = addr[size_t(q)*3 + kind];
            }
        }
        pf.pdown[q] = down ? 0u : g;
        owner[q]  = e;
        stamp[q]  = t - 1;
        pos_of[e] = (int16_t) p;
    }
    // owned positions whose down copy finished since
    for (int p = 0; p < r; ++p) {
        const int q = off + p;
        const uint32_t g = pf.pdown[q];
        if (g == 0) {
            continue;
        }
        const int e = owner[q];
        if (e < 0) {
            pf.pdown[q] = 0;
            continue;
        }
        if (__hip_atomic_load(&pf.done[size_t(q)*2 + 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == g) {
            if (L.kind[2].addresses != nullptr) {
                L.kind[2].addresses[e] = addr[size_t(q)*3 + 2];
            }
            pf.pdown[q] = 0;
        }
    }
}

// inplace: before the down matmul of a layer (its own launch, after the up/gate matmuls)
static __global__ void ub_settle_down_kernel(const ub_pool_layer * layers, const ub_pool_meta * meta, const ub_pf_dev pf,
        int32_t * owner, const uint64_t * addr, const int layer) {
    if (threadIdx.x != 0) {
        return;
    }
    const int r = meta[layer].r, off = meta[layer].offset;
    const ub_pool_layer & L = layers[layer];
    for (int p = 0; p < r; ++p) {
        const int q = off + p;
        const uint32_t g = pf.pdown[q];
        if (g == 0 || owner[q] < 0) {
            continue;
        }
        if (__hip_atomic_load(&pf.done[size_t(q)*2 + 1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == g) {
            if (L.kind[2].addresses != nullptr) {
                L.kind[2].addresses[owner[q]] = addr[size_t(q)*3 + 2];
            }
            pf.pdown[q] = 0;
        }
    }
    __threadfence_system();
}

// empty positions first, then the oldest stamp, then the lower position
static __device__ __forceinline__ bool ub_precedes(const ub_step_args & a, const int off, const uint32_t t, const int x, const int y) {
    const bool ex = a.owner[off + x] < 0;
    const bool ey = a.owner[off + y] < 0;
    if (ex != ey) {
        return ex;
    }
    if (!ex) {
        const uint32_t ax = t - a.stamp[off + x];
        const uint32_t ay = t - a.stamp[off + y];
        if (ax != ay) {
            return ax > ay;
        }
    }
    return x < y;
}

// One block per launch of a routed layer. Everything of the layer is written by this block, or by
// the copy kernel that follows it on the same stream.
static __global__ void __launch_bounds__(ub_pool_threads) ub_step_kernel(const ub_step_args a) {
    __shared__ uint32_t s_t;
    __shared__ int s_r, s_off, s_n_miss, s_n_victims;
    __shared__ unsigned int s_hits, s_over, s_fills, s_evict, s_rejected;
    __shared__ int32_t s_key[ub_pool_max_experts];
    __shared__ int32_t s_miss[ub_pool_max_experts];
    __shared__ int32_t s_miss_key[ub_pool_max_experts];
    __shared__ int32_t s_victim[ub_pool_max_rcap];
    __shared__ unsigned int s_rows_w[ub_pool_max_experts];   // ghost filter: the selections (= rows) of the expert in this launch
    __shared__ uint8_t s_missed[ub_pool_max_experts];   // ghost filter: 1 a miss, 2 an admissible miss it rejected
    __shared__ int8_t  s_pfidx[ub_pool_max_experts];    // prefetch: the arrived staging slot holding the expert, -1
    __shared__ int32_t s_pf_owner[ub_pf_max_slots];
    __shared__ uint32_t s_pf_expect, s_pf_arrived, s_pf_taken;
    __shared__ int s_pf_set;
    __shared__ int s_ring_on;                            // ring admission in this lend
    __shared__ uint8_t s_viaring[ub_pool_max_experts];  // ring admission: the miss is a ring occupant
    __shared__ unsigned int s_ring_fills;

    const int tid = threadIdx.x;
    const int E   = a.n_experts;
    const ub_pool_layer & L = a.layers[a.layer];
    if (a.prof_counts != nullptr) {
        // profile_ids_kernel (expert-profiler.cu), here so that the layer needs no launch of its own; an unlent layer counts too
        const profile_selection sel = *a.prof_selection;
        if (sel.bank < a.prof_n_banks) {
            for (int i = tid; i < a.n_rows*a.n_used; i += blockDim.x) {
                const int row = i/a.n_used;
                if (row < sel.row_begin || row >= sel.row_end) {
                    continue;
                }
                const int id = a.ids[row*a.row_stride + i - row*a.n_used];
                if ((unsigned) id < (unsigned) E) {
                    atomicAdd(&a.prof_counts[size_t(sel.bank)*a.prof_bank_stride + id], 1U);
                }
            }
        }
    }
    if (tid == 0) {
        ub_pool_meta & m = a.meta[a.layer];
        s_r   = m.r;
        s_off = m.offset;
        // the copy kernel of this launch reads only the list of this step
        *a.count = 0;
        if (s_r > 0) {
            m.step = m.step + 1;
        }
        s_t = m.step;
        s_n_miss = 0;
        s_n_victims = 0;
        s_hits = s_over = s_fills = s_evict = s_rejected = 0;
        s_ring_fills = 0;
        s_ring_on = a.ringpos != nullptr && m.ring != 0 && s_r > 0;
        s_pf_arrived = s_pf_taken = 0;
        s_pf_set = 0;
        s_pf_expect = 0;
        if (a.pf_on && a.pf.inplace && s_r > 0) {
            // the copies that finished become owned positions before anything of this step looks at the layer
            ub_pf_settle(L, a.pf, s_off, s_r, s_t, a.owner, a.stamp, a.pos_of + size_t(a.layer)*a.n_experts, a.home, a.addr);
            s_pf_expect = __hip_atomic_load(&a.pf.expect[a.layer], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            const int n = a.pf.res_n[a.layer];
            for (int j = 0; j < n; ++j) {
                const int q = a.pf.res_q[a.layer*a.pf.slots + j];
                if (a.owner[q] == a.pf.res_e[a.layer*a.pf.slots + j]) {
                    s_pf_arrived |= 1u << j;
                }
            }
        } else if (a.pf_on && s_r > 0) {
            // a slot whose generation is the layer's expected one holds the complete bytes of its owner
            const uint32_t ex = __hip_atomic_load(&a.pf.expect[a.layer], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            const int set = a.layer % a.pf.ring;
            s_pf_expect = ex;
            s_pf_set = set;
            for (int j = 0; j < a.pf.slots; ++j) {
                const int idx = set*a.pf.slots + j;
                const uint32_t g = __hip_atomic_load(&a.pf.gen[idx], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
                s_pf_owner[j] = -1;
                if (ex != 0 && g == ex) {
                    s_pf_owner[j] = __hip_atomic_load(&a.pf.owner[idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
                    s_pf_arrived |= 1u << j;
                }
            }
        }
    }
    __syncthreads();
    const int r = s_r;
    if (r <= 0) {
        return;
    }
    const uint32_t t = s_t;
    const int off = s_off;
    int16_t * pos_of = a.pos_of + size_t(a.layer)*E;
    const bool ghost = a.ghost > 0;

    for (int e = tid; e < E; e += blockDim.x) {
        s_key[e] = -1;
        s_missed[e] = 0;
        s_rows_w[e] = 0;
        s_pfidx[e] = -1;
        s_viaring[e] = 0;
    }
    const bool ring_on = s_ring_on != 0;
    __syncthreads();
    const bool pf_on = a.pf_on && !a.pf.inplace && s_pf_arrived != 0;
    if (pf_on) {
        if (tid < a.pf.slots) {
            const int o = s_pf_owner[tid];
            if ((unsigned) o < (unsigned) E) {
                s_pfidx[o] = (int8_t) tid;
            }
        }
        __syncthreads();
    }

    const int n = a.n_rows*a.n_used;
    unsigned int hits = 0;
    if (ghost) {
        // an expert two rows of this launch selected has its second access in this step: it passes the filter
        // (the router picks distinct experts within a row, so each selection is one row)
        for (int i = tid; i < n; i += blockDim.x) {
            const int row = i/a.n_used;
            const int id  = a.ids[row*a.row_stride + i - row*a.n_used];
            if ((unsigned) id < (unsigned) E) {
                atomicAdd(&s_rows_w[id], 1U);
            }
        }
        __syncthreads();
    }
    for (int i = tid; i < n; i += blockDim.x) {
        const int row = i/a.n_used;
        const int k   = i - row*a.n_used;
        const int id  = a.ids[row*a.row_stride + k];
        if ((unsigned) id >= (unsigned) E) {
            continue;
        }
        const int p = pos_of[id];
        if (p >= 0 && p < r) {
            a.stamp[off + p] = t;
            hits++;
            continue;
        }
        if (L.slots[id] >= 0) {
            continue;   // static: not a miss
        }
        bool via_ring = false;
        if (!a.admit || !ub_admissible(L, id)) {
            if (!(ring_on && a.admit && ub_ring_admissible(L, id))) {
                if (ghost) { s_missed[id] = 1; }
                continue;
            }
            via_ring = true;
        }
        // a staged expert is admitted ahead of the other misses and without the admission filter
        const bool staged = pf_on && s_pfidx[id] >= 0;
        if (ghost) {
            // every selection of one expert in this launch reads the same stamp and row count, so they agree
            if (!staged && s_rows_w[id] < 2 && !ub_ghost_ok(a.last_miss[id], t, (uint32_t) a.ghost)) {
                s_missed[id] = 2;
                continue;
            }
            s_missed[id] = 1;
        }
        if (via_ring) {
            s_viaring[id] = 1;   // every selection of the expert in this launch agrees
        }
        // later rows first, then the first router slot of that row
        atomicMax(&s_key[id], (staged ? 0x40000000 : 0) + row*a.n_used + (a.n_used - 1 - k));
    }
    if (hits != 0) {
        atomicAdd(&s_hits, hits);
    }
    __syncthreads();

    unsigned int rejected = 0;
    for (int e = tid; e < E; e += blockDim.x) {
        if (s_key[e] >= 0) {
            const int j = atomicAdd(&s_n_miss, 1);
            s_miss[j]     = e;
            s_miss_key[j] = s_key[e];
        }
        rejected += s_missed[e] == 2;
    }
    if (rejected != 0) {
        atomicAdd(&s_rejected, rejected);
    }
    __syncthreads();
    const int n_miss = s_n_miss;

    if (n_miss > 0) {
        // ranked by counting, so the order is deterministic
        for (int p = tid; p < r; p += blockDim.x) {
            if (!ub_eligible(a, off, t, p)) {
                continue;
            }
            int rank = 0;
            for (int q = 0; q < r; ++q) {
                rank += q != p && ub_eligible(a, off, t, q) && ub_precedes(a, off, t, q, p);
            }
            s_victim[rank] = p;
            atomicAdd(&s_n_victims, 1);
        }
        __syncthreads();
        const int n_fill = min(n_miss, s_n_victims);
        unsigned int fills = 0, over = 0, evict = 0, ring_fills = 0;
        for (int j = tid; j < n_miss; j += blockDim.x) {
            const int key = s_miss_key[j];
            int rank = 0;
            for (int i = 0; i < n_miss; ++i) {
                rank += s_miss_key[i] > key;
            }
            if (rank >= n_fill) {
                over++;
                continue;
            }
            const int p = s_victim[rank];
            const int q = off + p;
            const int e = s_miss[j];
            const int old = a.owner[q];
            if (old >= 0) {
                // the old owner is not used by this launch, and the previous launch is complete; an owner from the
                // ring gets 0 (its saved slot may hold another expert by now), and the worker publishes it again
                const bool old_ring = ring_on && a.ringpos[q] != 0;
                for (int kind = 0; kind < 3; ++kind) {
                    if (L.kind[kind].addresses != nullptr) {
                        L.kind[kind].addresses[old] = old_ring ? 0 : a.home[size_t(q)*3 + kind];
                    }
                }
                pos_of[old] = -1;
                evict++;
            }
            for (int kind = 0; kind < 3; ++kind) {
                uint64_t * table = L.kind[kind].addresses;
                a.home[size_t(q)*3 + kind] = table != nullptr ? table[e] : 0;
                if (table != nullptr) {
                    table[e] = a.addr[size_t(q)*3 + kind];
                }
            }
            a.owner[q] = e;
            a.stamp[q] = t;
            pos_of[e]  = (int16_t) p;
            if (a.pf_on && a.pf.inplace) {
                a.pf.pdown[q] = 0;
            }
            if (ring_on) {
                a.ringpos[q] = s_viaring[e];
                ring_fills += s_viaring[e];
            }
            const int pj = pf_on ? (int) s_pfidx[e] : -1;
            a.copy[off + rank] = { e, q, pj >= 0 ? s_pf_set*a.pf.slots + pj : -1 };
            if (pj >= 0) {
                atomicOr(&s_pf_taken, 1u << pj);
            }
            fills++;
        }
        if (tid == 0) {
            *a.count = n_fill;
        }
        if (fills) { atomicAdd(&s_fills, fills); }
        if (over)  { atomicAdd(&s_over, over); }
        if (evict) { atomicAdd(&s_evict, evict); }
        if (ring_fills) { atomicAdd(&s_ring_fills, ring_fills); }
    }
    // the tables may be mapped host memory, which the next kernels read
    __threadfence_system();
    __syncthreads();
    if (ring_on && s_fills != 0) {
        // the positions changed: the tier's publish kernel (next on the stream) and the worker see the new set
        ub_ring_bits(a, pos_of, off, r);
    }
    if (ghost) {
        // the misses of this step, stamped after its decisions, so that a miss never admits itself
        for (int e = tid; e < E; e += blockDim.x) {
            if (s_missed[e] != 0) {
                a.last_miss[e] = t;
            }
        }
    }
    if (a.pf_on) {
        // the host's budget record: the experts the layer's positions hold now, then the step's clock and slots
        ub_pf_stat & st = a.pf.stat[a.layer];
        if (a.pf.inplace && tid == 0) {
            const int n = a.pf.res_n[a.layer];
            for (int j = 0; j < n; ++j) {
                const int q = a.pf.res_q[a.layer*a.pf.slots + j];
                if (a.owner[q] == a.pf.res_e[a.layer*a.pf.slots + j] && a.stamp[q] == t) {
                    s_pf_taken |= 1u << j;
                }
            }
        }
        for (int w = tid; w < (E + 31)/32; w += blockDim.x) {
            uint32_t m = 0;
            for (int b = 0; b < 32; ++b) {
                const int e = w*32 + b;
                const int p = e < E ? pos_of[e] : -1;
                m |= (p >= 0 && p < r) ? 1u << b : 0u;
            }
            st.pooled[w] = m;
        }
        __syncthreads();
        if (tid == 0) {
            st.arrived = s_pf_arrived;
            st.taken   = s_pf_taken;
            st.n_rows  = (uint32_t) a.n_rows;
            st.t_step  = (uint64_t) wall_clock64();
            __threadfence_system();
            __hip_atomic_store(&st.gen, s_pf_expect, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            if (s_pf_taken)   { atomicAdd(&a.counters[ub_pool_pf_fills], (unsigned long long) __popc(s_pf_taken)); }   // inplace: hits
            if (s_pf_arrived) { atomicAdd(&a.counters[ub_pool_pf_arrived], (unsigned long long) __popc(s_pf_arrived)); }
        }
    }
    if (tid == 0) {
        atomicAdd(&a.counters[ub_pool_steps], 1ull);
        if (s_hits)  { atomicAdd(&a.counters[ub_pool_hits], (unsigned long long) s_hits); }
        if (s_fills) {
            atomicAdd(&a.counters[ub_pool_fills], (unsigned long long) s_fills);
            atomicAdd(&a.counters[ub_pool_fill_bytes], (unsigned long long) s_fills*a.pos_bytes);
        }
        if (s_over)  { atomicAdd(&a.counters[ub_pool_over], (unsigned long long) s_over); }
        if (s_evict) { atomicAdd(&a.counters[ub_pool_evictions], (unsigned long long) s_evict); }
        if (s_rejected) { atomicAdd(&a.counters[ub_pool_rejected], (unsigned long long) s_rejected); }
        if (s_ring_fills) { atomicAdd(&a.counters[ub_pool_ring_fills], (unsigned long long) s_ring_fills); }
        if ((s_hits || s_fills) && a.live != nullptr) {
            // the layer's own totals: only this layer's launches write them, in stream order
            ub_pool_meta & m = a.meta[a.layer];
            m.n_hits  += s_hits;
            m.n_fills += s_fills;
            a.live[a.layer].hits  = m.n_hits;
            a.live[a.layer].fills = m.n_fills;
            __threadfence_system();
        }
    }
}

typedef int32_t ub_i4 __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ void ub_copy_bytes(const char * src, char * dst, const size_t bytes, const int lane, const int n_lanes) {
    const uintptr_t al = (uintptr_t) src | (uintptr_t) dst | (uintptr_t) bytes;
    if (al % 16 == 0) {
        const ub_i4 * s = (const ub_i4 *) src;
        ub_i4 * d = (ub_i4 *) dst;
        for (size_t i = lane; i < bytes/16; i += n_lanes) {
            d[i] = s[i];
        }
        return;
    }
    for (size_t i = lane; i < bytes; i += n_lanes) {
        dst[i] = src[i];
    }
}

// This lane's share of the three kinds of one listed expert, from its home, or from a prefetch staging slot
// laid out as a position (`koff` = the kind offsets of the layer).
static __device__ __forceinline__ void ub_copy_entry(const ub_pool_layer & L, const ub_pool_copy en, const uint64_t * addr,
        const uint64_t * home, const int lane, const int n_lanes, const char * staging, const uint64_t slot_bytes, const uint64_t * koff) {
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const ub_pool_kind & c = L.kind[k];
        const uint64_t h = home[size_t(en.pos)*3 + k];
        const char * src;
        if (en.pf >= 0) {
            src = staging + size_t(en.pf)*slot_bytes + koff[k];
        } else if (h != 0) {
            src = (const char *) (uintptr_t) h;
        } else {
            const int32_t channel = c.host_slots != nullptr ? c.host_slots[en.expert] : en.expert;
            src = c.tensor + size_t(channel)*c.stride;
        }
        ub_copy_bytes(src, (char *) (uintptr_t) addr[size_t(en.pos)*3 + k], c.stride, lane, n_lanes);
    }
}

// Block (j, y) copies its share of the j-th expert the step listed.
static __global__ void __launch_bounds__(ub_pool_threads) ub_copy_kernel(const ub_pool_layer * layers, const int layer,
        const ub_pool_copy * __restrict__ list, const int32_t * __restrict__ count, const uint64_t * __restrict__ addr,
        const uint64_t * __restrict__ home, const char * staging, const uint64_t slot_bytes, const size_t * sizes, const int n_layers) {
    const int j = blockIdx.x;
    if (j >= *count) {
        return;
    }
    const ub_pool_copy en = list[j];
    uint64_t koff[3] = { 0, 0, 0 };
    if (en.pf >= 0) {
        // the copy engine wrote the slot: drop any line of it the GPU caches still hold from an earlier use
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
        uint64_t o = 0;
        for (int k = 0; k < 3; ++k) {
            koff[k] = o;
            o += (sizes[size_t(layer)*3 + k] + sizes[size_t(n_layers)*3 + size_t(layer)*3 + k] + 255) & ~uint64_t(255);
        }
    }
    ub_copy_entry(layers[layer], en, addr, home, blockIdx.y*blockDim.x + threadIdx.x, gridDim.y*blockDim.x, staging, slot_bytes, koff);
}

// inplace publish of layer `a.layer`: settle, choose the experts and the tail positions, reserve them, publish.
static __global__ void __launch_bounds__(ub_pool_threads) ub_publish_inplace_kernel(const ub_step_args a, const int k) {
    __shared__ int32_t s_score[ub_pool_max_experts];
    __shared__ uint8_t s_pend[ub_pool_max_experts];
    __shared__ int32_t s_victim[ub_pool_max_rcap];
    __shared__ int32_t s_pick[ub_pf_max_slots];
    __shared__ unsigned long long s_best;
    __shared__ int s_r, s_off, s_n, s_b;
    __shared__ int s_ring_changed;   // ring admission: a reservation evicted an owner from the ring
    __shared__ uint32_t s_t;
    const int tid = threadIdx.x;
    const int E = a.n_experts;
    const ub_pool_layer & L = a.layers[a.layer];
    const ub_pf_dev & pf = a.pf;
    int16_t * pos_of = a.pos_of + size_t(a.layer)*E;
    const int rows = min(a.n_rows, ub_pf_max_rows);
    const int kk   = min(k, ub_pf_max_k);
    if (tid == 0) {
        const ub_pool_meta & m = a.meta[a.layer];
        s_r = m.r;
        s_off = m.offset;
        s_t = m.step;
        s_n = 0;
        s_ring_changed = 0;
        const int bucket = rows <= 1 ? 0 : rows <= 3 ? 1 : 2;
        int b = __hip_atomic_load(&pf.budget[a.layer*3 + bucket], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        s_b = s_r > 0 ? max(0, min(b, pf.slots)) : 0;
        if (s_r > 0) {
            ub_pf_settle(L, pf, s_off, s_r, s_t, a.owner, a.stamp, pos_of, a.home, a.addr);
        }
    }
    for (int e = tid; e < E; e += blockDim.x) {
        s_score[e] = 0;
        s_pend[e] = 0;
    }
    __syncthreads();
    const int r = s_r, off = s_off;
    const uint32_t t = s_t;
    for (int p = tid; p < r; p += blockDim.x) {
        if (a.owner[off + p] == ub_pending) {
            const int e = pf.pend_e[off + p];
            if ((unsigned) e < (unsigned) E) {
                s_pend[e] = 1;
            }
        }
    }
    for (int i = tid; i < rows*kk; i += blockDim.x) {
        const int row = i/kk;
        const int c = i - row*kk;
        const int e = a.ids[row*a.row_stride + c];
        pf.mail[a.layer].ids[row*ub_pf_max_k + c] = e;
        if ((unsigned) e < (unsigned) E) {
            atomicAdd(&s_score[e], kk - c);
        }
    }
    __syncthreads();
    // the b best predicted experts that a reservation can take (highest summed rank score, then the lower id)
    for (int j = 0; j < s_b; ++j) {
        if (tid == 0) {
            s_best = 0;
        }
        __syncthreads();
        unsigned long long best = 0;
        for (int e = tid; e < E; e += blockDim.x) {
            const int sc = s_score[e];
            if (sc > 0 && !s_pend[e] && pos_of[e] < 0 && ub_admissible(L, e)) {
                const unsigned long long key = ((unsigned long long) sc << 32) | (unsigned long long) (0xFFFFFFFFu - (uint32_t) e);
                best = key > best ? key : best;
            }
        }
        if (best != 0) {
            atomicMax(&s_best, best);
        }
        __syncthreads();
        if (s_best == 0) {
            break;
        }
        if (tid == 0) {
            const int e = (int) (0xFFFFFFFFu - (uint32_t) (s_best & 0xFFFFFFFFull));
            s_pick[s_n++] = e;
            s_score[e] = 0;
        }
        __syncthreads();
    }
    const int n = s_n;
    // the positions: unfinished reservations first, then empty, then the oldest stamp, then the lower index
    if (n > 0) {
        for (int p = tid; p < r; p += blockDim.x) {
            const int op = a.owner[off + p];
            const int cp = op == ub_pending ? 0 : op < 0 ? 1 : pf.pdown[off + p] != 0 ? 3 : 2;
            int rank = 0;
            for (int q = 0; q < r; ++q) {
                if (q == p) {
                    continue;
                }
                const int oq = a.owner[off + q];
                const int cq = oq == ub_pending ? 0 : oq < 0 ? 1 : pf.pdown[off + q] != 0 ? 3 : 2;
                bool before = cq < cp;
                if (cq == cp) {
                    const uint32_t aq = t - a.stamp[off + q], ap = t - a.stamp[off + p];
                    before = cq == 2 && aq != ap ? aq > ap : q < p;
                }
                rank += before;
            }
            if (rank < n && cp != 3) {
                s_victim[rank] = p;
            } else if (rank < n) {
                s_victim[rank] = -1;   // only positions with a down copy due are left: not taken
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t seq = pf.seq[a.layer] + 1;
        pf.seq[a.layer] = seq;
        const uint32_t gen = (seq << 7) | (uint32_t) (a.layer & 127);
        ub_pf_mail & m = pf.mail[a.layer];
        int nn = min(n, r);
        for (int j = 0; j < nn; ++j) {
            if (s_victim[j] < 0) {
                nn = j;
                break;
            }
        }
        for (int j = 0; j < nn; ++j) {
            const int q = off + s_victim[j];
            const int old = a.owner[q];
            const bool old_ring = a.ringpos != nullptr && old >= 0 && a.ringpos[q] != 0;
            if (old >= 0) {
                // the old owner is not used before this layer's step, and its last launch is complete (one from the
                // ring gets 0, see the step kernel)
                for (int kind = 0; kind < 3; ++kind) {
                    if (L.kind[kind].addresses != nullptr) {
                        L.kind[kind].addresses[old] = old_ring ? 0 : a.home[size_t(q)*3 + kind];
                    }
                }
                pos_of[old] = -1;
            }
            if (a.ringpos != nullptr) {
                s_ring_changed |= old_ring ? 1 : 0;
                a.ringpos[q] = 0;
            }
            a.owner[q] = ub_pending;
            pf.pend_e[q] = s_pick[j];
            pf.pend_gen[q] = gen;
            pf.res_q[a.layer*pf.slots + j] = q;
            pf.res_e[a.layer*pf.slots + j] = s_pick[j];
            m.res_q[j] = q;
            m.res_e[j] = s_pick[j];
        }
        pf.res_n[a.layer] = nn;
        pf.expect[a.layer] = gen;
        m.n_res  = nn;
        m.n_rows = rows;
        m.k      = kk;
        m.t_pub  = (uint64_t) wall_clock64();
        __threadfence_system();
        __hip_atomic_store(&m.gen, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (a.ringpos != nullptr) {
        __syncthreads();
        if (s_ring_changed) {
            ub_ring_bits(a, pos_of, off, r);
        }
    }
}

// The publish of a prediction for layer `layer`: the ids into the mapped mailbox, a new generation
// (seq << 7 | layer) as the layer's expected one, then the mailbox's generation, last.
static __global__ void ub_publish_kernel(const ub_pf_dev pf, const int layer, const int32_t * __restrict__ ids, const int n_rows,
        const int row_stride, const int k) {
    ub_pf_mail & m = pf.mail[layer];
    const int rows = min(n_rows, ub_pf_max_rows);
    const int kk   = min(k, ub_pf_max_k);
    for (int i = threadIdx.x; i < rows*kk; i += blockDim.x) {
        const int r = i/kk;
        const int c = i - r*kk;
        m.ids[r*ub_pf_max_k + c] = ids[r*row_stride + c];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t seq = pf.seq[layer] + 1;
        pf.seq[layer] = seq;
        const uint32_t gen = (seq << 7) | (uint32_t) (layer & 127);
        pf.expect[layer] = gen;
        m.n_rows = rows;
        m.k      = kk;
        m.t_pub  = (uint64_t) wall_clock64();
        __threadfence_system();
        __hip_atomic_store(&m.gen, gen, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

struct ub_lend_args {
    const ub_pool_layer * layers;
    ub_pool_meta * meta;
    const uint64_t * addr;
    uint64_t * home;
    int32_t * owner;
    uint32_t * stamp;
    int32_t * count;
    const size_t * strides;   // [L][3]
    const size_t * tails;     // [L][3]
    int r;
    int ring;                 // ring admission in this lend
    int32_t * ringpos;        // [P] or null
};

// lend: the r positions of every enabled layer, empty, their tails zeroed. The step count is kept.
static __global__ void __launch_bounds__(ub_pool_threads) ub_lend_kernel(const ub_lend_args a) {
    const int layer = blockIdx.x;
    const int tid = threadIdx.x;
    const ub_pool_layer & L = a.layers[layer];
    ub_pool_meta & m = a.meta[layer];
    if (!L.enabled) {
        if (tid == 0) {
            m.r = 0;
            a.count[layer] = 0;
        }
        return;
    }
    const int r = a.r;
    const int off = m.offset;
    for (int p = 0; p < r; ++p) {
        const int q = off + p;
        if (tid == 0) {
            a.owner[q] = -1;
            a.stamp[q] = 0;
            if (a.ringpos != nullptr) {
                a.ringpos[q] = 0;
            }
        }
        for (int k = 0; k < 3; ++k) {
            char * tail = (char *) (uintptr_t) a.addr[size_t(q)*3 + k] + a.strides[size_t(layer)*3 + k];
            const size_t n = a.tails[size_t(layer)*3 + k];
            for (size_t i = tid; i < n; i += blockDim.x) {
                tail[i] = 0;
            }
        }
    }
    if (tid == 0) {
        a.count[layer] = 0;
        m.r = r;
        m.ring = a.ring;
    }
    // the tables may be mapped host memory, which the next kernels read
    __threadfence_system();
}

// reclaim: the entries of every owner back to their homes, then r = 0
static __global__ void ub_reclaim_kernel(const ub_pool_layer * layers, ub_pool_meta * meta, const uint64_t * home,
        int32_t * owner, int16_t * pos_of, int n_experts, uint32_t * pdown, int32_t * ringpos, uint32_t * ringbits,
        uint32_t * ringbits_host, int words) {
    const int layer = blockIdx.x;
    ub_pool_meta & m = meta[layer];
    const int r = m.r;
    const ub_pool_layer & L = layers[layer];
    for (int p = threadIdx.x; p < r; p += blockDim.x) {
        const int q = m.offset + p;
        const int e = owner[q];
        if (pdown != nullptr) {
            pdown[q] = 0;
        }
        // ring admission: an owner from the ring gets 0, not its saved slot (see the step kernel)
        const bool from_ring = ringpos != nullptr && ringpos[q] != 0;
        if (ringpos != nullptr) {
            ringpos[q] = 0;
        }
        if (e < 0) {
            owner[q] = -1;   // empty, or a pending prefetch (its copy finished before the reclaim)
            continue;
        }
        for (int k = 0; k < 3; ++k) {
            if (L.kind[k].addresses != nullptr) {
                L.kind[k].addresses[e] = from_ring ? 0 : home[size_t(q)*3 + k];
            }
        }
        pos_of[size_t(layer)*n_experts + e] = -1;
        owner[q] = -1;
    }
    if (ringbits != nullptr) {
        for (int w = threadIdx.x; w < words; w += blockDim.x) {
            ringbits[size_t(layer)*words + w] = 0;
            ringbits_host[size_t(layer)*words + w] = 0;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        m.r = 0;
    }
}

// ---- host side --------------------------------------------------------------------------------------

ub_pool::ub_pool(int n_layers, int n_experts, int r_cap) : n_layers_(n_layers), n_experts_(n_experts), r_cap_(r_cap) {}

ub_pool::~ub_pool() {
    if (blob_ != nullptr) {
        ggml_cuda_set_device(device_);
        (void) cudaFree(blob_);
    }
    if (h_addr_ != nullptr) { (void) cudaFreeHost(h_addr_); }
    if (h_counters_ != nullptr) { (void) cudaFreeHost(h_counters_); }
    if (h_live_ != nullptr) { (void) hipHostFree(h_live_); }
    if (pf_alloc_ != nullptr) { (void) cudaFree(pf_alloc_); }
    if (pf_staging_ != nullptr) { (void) cudaFree(pf_staging_); }
    if (h_pf_stat_ != nullptr) { (void) hipHostFree(h_pf_stat_); }
    if (h_pf_mail_ != nullptr) { (void) hipHostFree(h_pf_mail_); }
    if (h_pf_budget_ != nullptr) { (void) hipHostFree(h_pf_budget_); }
    if (pf_in_alloc_ != nullptr) { (void) cudaFree(pf_in_alloc_); }
    if (d_ringpos_ != nullptr) { (void) cudaFree(d_ringpos_); }
    if (d_ringbits_ != nullptr) { (void) cudaFree(d_ringbits_); }
    if (h_ringbits_ != nullptr) { (void) hipHostFree(h_ringbits_); }
}

bool ub_pool::allocate(int device, const std::vector<ub_pool_layer> & layers,
        const std::vector<std::array<size_t, 3>> & kind_bytes, const std::vector<std::array<size_t, 3>> & kind_tails, int ghost) {
    GGML_ASSERT(blob_ == nullptr);
    if ((int) layers.size() != n_layers_ || n_experts_ > ub_pool_max_experts || r_cap_ > ub_pool_max_rcap) {
        return false;
    }
    device_     = device;
    ghost_      = std::max(ghost, 0);
    layers_     = layers;
    kind_bytes_ = kind_bytes;
    kind_tails_ = kind_tails;
    pos_bytes_.assign((size_t) n_layers_, 0);
    for (int l = 0; l < n_layers_; ++l) {
        if (!layers_[l].enabled) {
            continue;
        }
        size_t b = 0;
        for (int k = 0; k < 3; ++k) {
            b += (kind_bytes_[l][k] + kind_tails_[l][k] + 255) & ~size_t(255);
        }
        pos_bytes_[l] = b;
    }

    const size_t L = (size_t) n_layers_, E = (size_t) n_experts_, P = L*(size_t) r_cap_;
    size_t at = 0;
    auto take = [&at](size_t bytes) { const size_t here = at; at += (bytes + 255) & ~size_t(255); return here; };
    const size_t o_layers = take(L*sizeof(ub_pool_layer));
    const size_t o_meta   = take(L*sizeof(ub_pool_meta));
    const size_t o_addr   = take(P*3*sizeof(uint64_t));
    const size_t o_home   = take(P*3*sizeof(uint64_t));
    const size_t o_owner  = take(P*sizeof(int32_t));
    const size_t o_stamp  = take(P*sizeof(uint32_t));
    const size_t o_pos_of = take(L*E*sizeof(int16_t));
    const size_t o_copy   = take(P*sizeof(ub_pool_copy));
    const size_t o_count  = take(L*sizeof(int32_t));
    const size_t o_cnt    = take(ub_pool_n_counters*sizeof(unsigned long long));
    const size_t o_sizes  = take(L*3*sizeof(size_t)*2);
    const size_t o_last   = take(L*E*sizeof(uint32_t));
    bytes_ = at;

    ggml_cuda_set_device(device_);
    if (cudaMalloc(&blob_, bytes_) != cudaSuccess) {
        (void) cudaGetLastError();
        blob_ = nullptr;
        return false;
    }
    char * b = (char *) blob_;
    d_layers_   = (ub_pool_layer *) (b + o_layers);
    d_meta_     = (ub_pool_meta *) (b + o_meta);
    d_addr_     = (uint64_t *) (b + o_addr);
    d_home_     = (uint64_t *) (b + o_home);
    d_owner_    = (int32_t *) (b + o_owner);
    d_stamp_    = (uint32_t *) (b + o_stamp);
    d_pos_of_   = (int16_t *) (b + o_pos_of);
    d_copy_     = (ub_pool_copy *) (b + o_copy);
    d_count_    = (int32_t *) (b + o_count);
    d_counters_ = (unsigned long long *) (b + o_cnt);
    d_sizes_    = (size_t *) (b + o_sizes);
    d_last_miss_ = (uint32_t *) (b + o_last);
    size_t * d_sizes = d_sizes_;

    CUDA_CHECK(cudaMemset(blob_, 0, bytes_));
    CUDA_CHECK(cudaMemset(d_owner_, 0xFF, P*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_pos_of_, 0xFF, L*E*sizeof(int16_t)));
    CUDA_CHECK(cudaMemset(d_last_miss_, 0xFF, L*E*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_layers_, layers_.data(), L*sizeof(ub_pool_layer), cudaMemcpyHostToDevice));
    std::vector<size_t> sizes(L*6, 0);
    for (size_t l = 0; l < L; ++l) {
        for (int k = 0; k < 3; ++k) {
            sizes[l*3 + k]         = kind_bytes_[l][k];
            sizes[L*3 + l*3 + k]   = kind_tails_[l][k];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_sizes, sizes.data(), sizes.size()*sizeof(size_t), cudaMemcpyHostToDevice));
    std::vector<ub_pool_meta> meta(L);
    for (size_t l = 0; l < L; ++l) {
        meta[l] = { int32_t(l*(size_t) r_cap_), 0, 0, 0, 0, 0 };
    }
    CUDA_CHECK(cudaMemcpy(d_meta_, meta.data(), L*sizeof(ub_pool_meta), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMallocHost((void **) &h_addr_, P*3*sizeof(uint64_t)));
    CUDA_CHECK(cudaMallocHost((void **) &h_counters_, ub_pool_n_counters*sizeof(unsigned long long)));
    // the per-layer counts the host reads without a synchronize (read_live); without them only read_live fails
    if (hipHostMalloc((void **) &h_live_, L*sizeof(ub_pool_live), hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess) {
        memset(h_live_, 0, L*sizeof(ub_pool_live));
        if (hipHostGetDevicePointer((void **) &d_live_, h_live_, 0) != hipSuccess) {
            (void) hipGetLastError();
            (void) hipHostFree(h_live_);
            h_live_ = nullptr;
            d_live_ = nullptr;
        }
    } else {
        (void) hipGetLastError();
        h_live_ = nullptr;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

int ub_pool::lend(void * base, size_t bytes, cudaStream_t stream, void * base2, size_t bytes2) {
    GGML_ASSERT(r_ == 0);
    size_t per_round = 0;
    for (int l = 0; l < n_layers_; ++l) {
        per_round += pos_bytes_[l];
    }
    // each region starts 256-aligned and holds whole rounds (one position of every layer)
    auto rounds = [per_round](void * b, size_t n) -> size_t {
        if (b == nullptr || per_round == 0) {
            return 0;
        }
        const uintptr_t start = ((uintptr_t) b + 255) & ~uintptr_t(255);
        return n > size_t(start - (uintptr_t) b) ? (n - size_t(start - (uintptr_t) b))/per_round : 0;
    };
    const int r1 = (int) std::min<size_t>(rounds(base, bytes), (size_t) r_cap_);
    const int r2 = (int) std::min<size_t>(rounds(base2, bytes2), (size_t) (r_cap_ - r1));
    const int r  = r1 + r2;
    if (r <= 0) {
        return 0;
    }
    // positions [0, r1) of every layer in the first region, [r1, r) in the second, each region layer-major
    auto lay = [&](void * b, size_t n, int p0, int p1) {
        uintptr_t at = ((uintptr_t) b + 255) & ~uintptr_t(255);
        for (int l = 0; l < n_layers_; ++l) {
            if (!layers_[l].enabled) {
                continue;
            }
            for (int p = p0; p < p1; ++p) {
                const size_t q = size_t(l)*r_cap_ + p;
                for (int k = 0; k < 3; ++k) {
                    h_addr_[q*3 + k] = (uint64_t) at;
                    at += (kind_bytes_[l][k] + kind_tails_[l][k] + 255) & ~size_t(255);
                }
            }
        }
        GGML_ASSERT(at <= (uintptr_t) b + n);
    };
    if (r1 > 0) {
        lay(base, bytes, 0, r1);
    }
    if (r2 > 0) {
        lay(base2, bytes2, r1, r);
    }
    ggml_cuda_set_device(device_);
    const size_t P = size_t(n_layers_)*r_cap_;
    CUDA_CHECK(cudaMemcpyAsync(d_addr_, h_addr_, P*3*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    ub_lend_args a;
    a.layers     = d_layers_;
    a.meta       = d_meta_;
    a.addr       = d_addr_;
    a.owner      = d_owner_;
    a.stamp      = d_stamp_;
    a.count      = d_count_;
    a.strides    = d_sizes_;
    a.tails      = d_sizes_ + size_t(n_layers_)*3;
    a.r          = r;
    a.ring       = ring_ ? 1 : 0;
    a.ringpos    = d_ringpos_;
    {
        const ggml_cuda_kernel_launch_params launch(dim3(n_layers_), dim3(ub_pool_threads), 0, stream);
        ggml_cuda_kernel_launch(ub_lend_kernel, launch, a);
    }
    // the staging must stay untouched until the uploads ran (the lend kernel itself is short)
    CUDA_CHECK(cudaStreamSynchronize(stream));
    r_ = r;
    r2_ = r2;
    base_ = r1 > 0 ? base : base2;
    lent_bytes_ = (r1 > 0 ? bytes : 0) + (r2 > 0 ? bytes2 : 0);
    return r;
}

long ub_pool::owned_in_second() const {
    if (r_ == 0 || r2_ == 0) {
        return 0;
    }
    const size_t P = size_t(n_layers_)*r_cap_;
    std::vector<int32_t> owner(P);
    ggml_cuda_set_device(device_);
    CUDA_CHECK(cudaMemcpy(owner.data(), d_owner_, P*sizeof(int32_t), cudaMemcpyDeviceToHost));
    long n = 0;
    for (int l = 0; l < n_layers_; ++l) {
        if (!layers_[l].enabled) {
            continue;
        }
        for (int p = r_ - r2_; p < r_; ++p) {
            n += owner[size_t(l)*r_cap_ + p] >= 0 ? 1 : 0;
        }
    }
    return n;
}

void ub_pool::reclaim(cudaStream_t stream) {
    if (r_ == 0) {
        return;
    }
    ggml_cuda_set_device(device_);
    const ggml_cuda_kernel_launch_params launch(dim3(n_layers_), dim3(256), 0, stream);
    ggml_cuda_kernel_launch(ub_reclaim_kernel, launch, (const ub_pool_layer *) d_layers_, d_meta_, (const uint64_t *) d_home_,
        d_owner_, d_pos_of_, n_experts_, pf_.inplace ? pf_.pdown : (uint32_t *) nullptr, d_ringpos_, d_ringbits_,
        d_ringbits_host_, (int) ring_words());
    r_ = 0;
    r2_ = 0;
    base_ = nullptr;
    lent_bytes_ = 0;
}

void ub_pool::launch_step(int layer, const int32_t * ids, int n_rows, int row_stride, int n_used, bool admit, cudaStream_t stream,
        uint32_t * prof_counts, size_t prof_bank_stride, uint32_t prof_n_banks, const profile_selection * prof_selection) {
    ub_step_args a;
    a.ids        = ids;
    a.n_rows     = n_rows;
    a.row_stride = row_stride;
    a.n_used     = n_used;
    a.n_experts  = n_experts_;
    a.layer      = layer;
    a.r_cap      = r_cap_;
    a.admit      = admit ? 1 : 0;
    a.ghost      = ghost_;
    a.layers     = d_layers_;
    a.meta       = d_meta_;
    a.addr       = d_addr_;
    a.home       = d_home_;
    a.owner      = d_owner_;
    a.stamp      = d_stamp_;
    a.pos_of     = d_pos_of_;
    a.copy       = d_copy_;
    a.count      = d_count_ + layer;
    a.counters   = d_counters_;
    a.last_miss  = d_last_miss_ + size_t(layer)*n_experts_;
    a.live       = d_live_;
    a.pos_bytes  = pos_bytes_[(size_t) layer];
    a.pf         = pf_;
    a.pf_on      = pf_.staging != nullptr ? 1 : 0;
    a.ringpos       = d_ringpos_;
    a.ringbits      = d_ringbits_;
    a.ringbits_host = d_ringbits_host_;
    a.words         = (int) ring_words();
    a.prof_counts      = n_rows*n_used > 0 ? prof_counts : nullptr;
    a.prof_bank_stride = prof_bank_stride;
    a.prof_n_banks     = prof_n_banks;
    a.prof_selection   = prof_selection;
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(ub_pool_threads), 0, stream);
    ggml_cuda_kernel_launch(ub_step_kernel, launch, a);
}

bool ub_pool::enable_ring() {
    if (blob_ == nullptr || ring_enabled()) {
        return false;
    }
    const size_t P = size_t(n_layers_)*size_t(r_cap_), B = size_t(n_layers_)*ring_words()*sizeof(uint32_t);
    ggml_cuda_set_device(device_);
    int32_t * pos = nullptr;
    uint32_t * bits = nullptr, * host = nullptr, * host_dev = nullptr;
    bool ok = cudaMalloc((void **) &pos, P*sizeof(int32_t)) == cudaSuccess && cudaMalloc((void **) &bits, B) == cudaSuccess &&
        hipHostMalloc((void **) &host, B, hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess &&
        hipHostGetDevicePointer((void **) &host_dev, host, 0) == hipSuccess;
    if (!ok) {
        (void) hipGetLastError();
        if (pos != nullptr) { (void) cudaFree(pos); }
        if (bits != nullptr) { (void) cudaFree(bits); }
        if (host != nullptr) { (void) hipHostFree(host); }
        return false;
    }
    CUDA_CHECK(cudaMemset(pos, 0, P*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(bits, 0, B));
    memset(host, 0, B);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_ringpos_       = pos;
    d_ringbits_      = bits;
    h_ringbits_      = host;
    d_ringbits_host_ = host_dev;
    ring_            = true;
    return true;
}

void ub_pool::launch_copy(int layer, cudaStream_t stream) {
    const ggml_cuda_kernel_launch_params launch(dim3(r_cap_, ub_pool_copy_blocks), dim3(ub_pool_threads), 0, stream);
    ggml_cuda_kernel_launch(ub_copy_kernel, launch, (const ub_pool_layer *) d_layers_, layer,
        (const ub_pool_copy *) d_copy_ + size_t(layer)*r_cap_, (const int32_t *) d_count_ + layer,
        (const uint64_t *) d_addr_, (const uint64_t *) d_home_, (const char *) pf_.staging, pf_.slot_bytes,
        (const size_t *) d_sizes_, n_layers_);
}

bool ub_pool::enable_prefetch(int ring, int slots, bool inplace) {
    if (blob_ == nullptr || prefetch_on() || ring < 1 || slots < 1 || slots > ub_pf_max_slots || n_layers_ > 128) {
        return false;
    }
    if (inplace) {
        ring = 1;   // the fill staging is not used; one small set keeps the shared code paths valid
    }
    size_t slot_bytes = 0;
    for (int l = 0; l < n_layers_; ++l) {
        slot_bytes = std::max(slot_bytes, pos_bytes_[(size_t) l]);
    }
    if (slot_bytes == 0) {
        return false;
    }
    const size_t L = (size_t) n_layers_, S = size_t(ring)*size_t(slots);
    ggml_cuda_set_device(device_);
    const size_t o_gen = 0, o_owner = o_gen + S*sizeof(uint32_t), o_expect = o_owner + S*sizeof(int32_t),
                 o_seq = o_expect + L*sizeof(uint32_t), state = o_seq + L*sizeof(uint32_t);
    if (cudaMalloc(&pf_alloc_, state) != cudaSuccess || cudaMalloc((void **) &pf_staging_, S*slot_bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        if (pf_alloc_ != nullptr) { (void) cudaFree(pf_alloc_); pf_alloc_ = nullptr; }
        pf_staging_ = nullptr;
        return false;
    }
    CUDA_CHECK(cudaMemset(pf_alloc_, 0, state));
    CUDA_CHECK(cudaMemset((char *) pf_alloc_ + o_owner, 0xFF, S*sizeof(int32_t)));
    ub_pf_stat * d_stat = nullptr;
    ub_pf_mail * d_mail = nullptr;
    if (hipHostMalloc((void **) &h_pf_stat_, L*sizeof(ub_pf_stat), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess ||
        hipHostMalloc((void **) &h_pf_mail_, L*sizeof(ub_pf_mail), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess ||
        hipHostGetDevicePointer((void **) &d_stat, h_pf_stat_, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &d_mail, h_pf_mail_, 0) != hipSuccess) {
        (void) hipGetLastError();
        if (h_pf_stat_ != nullptr) { (void) hipHostFree(h_pf_stat_); h_pf_stat_ = nullptr; }
        if (h_pf_mail_ != nullptr) { (void) hipHostFree(h_pf_mail_); h_pf_mail_ = nullptr; }
        (void) cudaFree(pf_alloc_);
        (void) cudaFree(pf_staging_);
        pf_alloc_ = nullptr;
        pf_staging_ = nullptr;
        return false;
    }
    memset(h_pf_stat_, 0, L*sizeof(ub_pf_stat));
    memset(h_pf_mail_, 0, L*sizeof(ub_pf_mail));
    char * b = (char *) pf_alloc_;
    pf_.gen        = (const uint32_t *) (b + o_gen);
    pf_.owner      = (const int32_t *) (b + o_owner);
    pf_.expect     = (uint32_t *) (b + o_expect);
    pf_.seq        = (uint32_t *) (b + o_seq);
    pf_.staging    = pf_staging_;
    pf_.stat       = d_stat;
    pf_.mail       = d_mail;
    pf_.slot_bytes = slot_bytes;
    pf_.ring       = ring;
    pf_.slots      = slots;
    pf_.inplace    = inplace ? 1 : 0;
    if (inplace) {
        const size_t P = L*(size_t) r_cap_;
        const size_t o_pe = 0, o_pg = o_pe + P*sizeof(int32_t), o_dn = o_pg + P*sizeof(uint32_t),
                     o_pd = o_dn + 2*P*sizeof(uint32_t),
                     o_rq = o_pd + P*sizeof(uint32_t), o_re = o_rq + L*size_t(slots)*sizeof(int32_t),
                     o_rn = o_re + L*size_t(slots)*sizeof(int32_t), in_bytes = o_rn + L*sizeof(int32_t);
        int32_t * d_budget = nullptr;
        if (cudaMalloc(&pf_in_alloc_, in_bytes) != cudaSuccess ||
            hipHostMalloc((void **) &h_pf_budget_, L*3*sizeof(int32_t), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess ||
            hipHostGetDevicePointer((void **) &d_budget, h_pf_budget_, 0) != hipSuccess) {
            (void) hipGetLastError();
            return false;   // the destructor frees what was allocated; prefetch_on() stays false below
        }
        CUDA_CHECK(cudaMemset(pf_in_alloc_, 0, in_bytes));
        CUDA_CHECK(cudaMemset(pf_in_alloc_, 0xFF, P*sizeof(int32_t)));
        memset(h_pf_budget_, 0, L*3*sizeof(int32_t));
        char * c = (char *) pf_in_alloc_;
        pf_.pend_e   = (int32_t *) (c + o_pe);
        pf_.pend_gen = (uint32_t *) (c + o_pg);
        pf_.done     = (const uint32_t *) (c + o_dn);
        pf_.pdown    = (uint32_t *) (c + o_pd);
        pf_.res_q    = (int32_t *) (c + o_rq);
        pf_.res_e    = (int32_t *) (c + o_re);
        pf_.res_n    = (int32_t *) (c + o_rn);
        pf_.budget   = d_budget;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return true;
}

void ub_pool::launch_publish(int layer, const int32_t * ids, int n_rows, int row_stride, int k, cudaStream_t stream) {
    if (!prefetch_on()) {
        return;
    }
    if (!pf_.inplace) {
        const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(64), 0, stream);
        ggml_cuda_kernel_launch(ub_publish_kernel, launch, pf_, layer, ids, n_rows, row_stride, k);
        return;
    }
    ub_step_args a{};
    a.ids        = ids;
    a.n_rows     = n_rows;
    a.row_stride = row_stride;
    a.n_used     = k;
    a.n_experts  = n_experts_;
    a.layer      = layer;
    a.r_cap      = r_cap_;
    a.layers     = d_layers_;
    a.meta       = d_meta_;
    a.addr       = d_addr_;
    a.home       = d_home_;
    a.owner      = d_owner_;
    a.stamp      = d_stamp_;
    a.pos_of     = d_pos_of_;
    a.pf         = pf_;
    a.pf_on      = 1;
    a.ringpos       = d_ringpos_;
    a.ringbits      = d_ringbits_;
    a.ringbits_host = d_ringbits_host_;
    a.words         = (int) ring_words();
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(ub_pool_threads), 0, stream);
    ggml_cuda_kernel_launch(ub_publish_inplace_kernel, launch, a, k);
}

void ub_pool::launch_settle_down(int layer, cudaStream_t stream) {
    if (!prefetch_on() || !pf_.inplace) {
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3(1), dim3(32), 0, stream);
    ggml_cuda_kernel_launch(ub_settle_down_kernel, launch, (const ub_pool_layer *) d_layers_, (const ub_pool_meta *) d_meta_, pf_,
        d_owner_, (const uint64_t *) d_addr_, layer);
}

char * ub_pool::pf_kind_address(int layer, int set, int j, int kind) const {
    size_t o = 0;
    for (int k = 0; k < kind; ++k) {
        o += (kind_bytes_[(size_t) layer][(size_t) k] + kind_tails_[(size_t) layer][(size_t) k] + 255) & ~size_t(255);
    }
    return pf_staging_ + (size_t(set)*(size_t) pf_.slots + (size_t) j)*pf_.slot_bytes + o;
}

bool ub_pool::read_counters(std::array<uint64_t, ub_pool_n_counters> & out) const {
    if (blob_ == nullptr) {
        return false;
    }
    ggml_cuda_set_device(device_);
    // Through the pinned buffer, filled with a value no counter reaches first: some runs logged stack contents
    // as counters (a read that left a pageable stack array alone, never reproduced with this check in place).
    unsigned long long * v = h_counters_;
    for (int i = 0; i < ub_pool_n_counters; ++i) {
        v[i] = ub_counters_sentinel;
    }
    if (cudaMemcpy(v, d_counters_, ub_pool_n_counters*sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    for (int i = 0; i < ub_pool_n_counters; ++i) {
        if (v[i] == ub_counters_sentinel) {
            GGML_LOG_WARN("expert cache: decode pool: the counter read left counter %d unwritten; not logged\n", i);
            return false;
        }
        out[(size_t) i] = v[i];
    }
    return true;
}

bool ub_pool::read_live(uint64_t & hits, uint64_t & fills, uint64_t & fill_bytes) const {
    hits = fills = fill_bytes = 0;
    if (h_live_ == nullptr) {
        return false;
    }
    const volatile ub_pool_live * v = h_live_;
    for (int l = 0; l < n_layers_; ++l) {
        const uint64_t h = v[l].hits, f = v[l].fills;
        hits       += h;
        fills      += f;
        fill_bytes += f*pos_bytes_[(size_t) l];
    }
    return true;
}

long ub_pool::verify(std::string & reason) const {
    if (blob_ == nullptr || r_ == 0) {
        return 0;
    }
    ggml_cuda_set_device(device_);
    const size_t L = (size_t) n_layers_, E = (size_t) n_experts_, P = L*(size_t) r_cap_;
    std::vector<int32_t> owner(P);
    std::vector<uint64_t> addr(P*3), home(P*3);
    std::vector<int16_t> pos_of(L*E);
    CUDA_CHECK(cudaMemcpy(owner.data(), d_owner_, P*sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(addr.data(), d_addr_, P*3*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(home.data(), d_home_, P*3*sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pos_of.data(), d_pos_of_, L*E*sizeof(int16_t), cudaMemcpyDeviceToHost));
    std::vector<int32_t> ringpos(P, 0);
    if (d_ringpos_ != nullptr) {
        CUDA_CHECK(cudaMemcpy(ringpos.data(), d_ringpos_, P*sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
    long checked = 0;
    std::vector<char> a, b;
    for (size_t l = 0; l < L; ++l) {
        if (!layers_[l].enabled) {
            continue;
        }
        for (int p = 0; p < r_; ++p) {
            const size_t q = l*r_cap_ + p;
            const int e = owner[q];
            if (e < 0) {
                continue;
            }
            if (pos_of[l*E + e] != p) {
                reason = "layer " + std::to_string(l) + " position " + std::to_string(p) + ": the expert map disagrees";
                return -1;
            }
            for (int k = 0; k < 3; ++k) {
                const ub_pool_kind & c = layers_[l].kind[k];
                if (c.addresses != nullptr) {
                    uint64_t entry = 0;
                    CUDA_CHECK(cudaMemcpy(&entry, c.addresses + e, sizeof(entry), hipMemcpyDefault));
                    if (entry != addr[q*3 + k]) {
                        reason = "layer " + std::to_string(l) + " expert " + std::to_string(e) + " kind " + std::to_string(k) +
                            ": the address table does not point at its position";
                        return -1;
                    }
                }
                if (ringpos[q] != 0) {
                    continue;   // admitted from the ring: the slot it was copied from may hold another expert by now
                }
                const char * src = nullptr;
                if (home[q*3 + k] != 0) {
                    src = (const char *) (uintptr_t) home[q*3 + k];
                } else {
                    int32_t channel = e;
                    if (c.host_slots != nullptr) {
                        CUDA_CHECK(cudaMemcpy(&channel, c.host_slots + e, sizeof(channel), hipMemcpyDefault));
                    }
                    src = c.tensor + size_t(channel)*c.stride;
                }
                a.resize(c.stride);
                b.resize(c.stride);
                CUDA_CHECK(cudaMemcpy(a.data(), src, c.stride, hipMemcpyDefault));
                CUDA_CHECK(cudaMemcpy(b.data(), (const void *) (uintptr_t) addr[q*3 + k], c.stride, cudaMemcpyDeviceToHost));
                if (memcmp(a.data(), b.data(), c.stride) != 0) {
                    reason = "layer " + std::to_string(l) + " expert " + std::to_string(e) + " kind " + std::to_string(k) +
                        ": the slot does not hold the home bytes";
                    return -1;
                }
            }
            ++checked;
        }
    }
    return checked;
}

} // namespace ggml_cuda_expert

#endif // GGML_USE_HIP
