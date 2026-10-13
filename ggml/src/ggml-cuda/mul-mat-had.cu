// GGML_OP_MUL_MAT_HAD: y = rot_out * H(W^T H(rot_in * x)), see ggml_mul_mat_had()
// EXL3 trellis format and codebooks (mul1, mcg, 3inst): ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
//
// 1. prologue: one block per output row (token, or token x expert slot with ids) writes H(rot_in * x) once: F32 for
//    the GEMV, or for the WMMA GEMM in fp16, scaled by a power of 2 that keeps the values in the fp16 normal range.
// 2. GEMV (few rows per weight) or WMMA GEMM (RDNA4): a block owns one 128-row group of one weight matrix, the
//    H128 output block. Each 16x16 trellis tile is decoded once per block and used for all rows of the block.
// 3. epilogue in the same block (or in the last block of a k split): output Hadamard, rot_out and the row scale.
// Invalid expert ids are clamped to a valid expert (wrong rows, never a read outside the weight); there are no
// device-side checks, the op and its shapes are checked when the graph is built.
//
// Input precision, from the src1 precision of the op (ggml_prec_set_src(op, prec, 1)):
// - GGML_PREC_F16 or lower (the F16 mode): the WMMA GEMM takes one fp16 value of the scaled row, the precision class
//   of the official ExLlamaV3 kernels, which take fp16 activations.
// - GGML_PREC_F32: no GEMM; every row runs the GEMV with F32 input (after the routing pass in items of 8 rows), so
//   the product keeps the fp32 grade.
// - not set: MMH_PREC_DEFAULT; GGML_CUDA_MUL_MAT_HAD_PREC = f32 | f16 sets it for the process (to test the F32 path).
// The GEMV with EXL3 weights takes fp16 inputs in the F16 mode (v_dot2; for mul1 the codebook's affine map is applied
// once to the sums), except a dense op of one row; otherwise F32 inputs.
// The CPU backend always computes the fp32 grade.

#include "mul-mat-had.cuh"
#include "mapped-host.cuh"
#include "expert-controller.cuh"

#include <cfloat>
#include <map>
#include <mutex>

#define MMH_NG          128 // output columns per block, one H128 output block
#define MMH_GEMV_TOKENS   8 // GEMV without a routing pass up to this many tokens
#define MMH_GEMV_LIST   512 // most rows one GEMV block collects without a routing pass
#define MMH_MAX_EXPERTS 4096 // weight matrices the routing pass counts in LDS
#define MMH_FUSED_LDS   49152 // LDS of a GEMV block that stages its own input (fused)
#define MMH_FUSED_MAX_BLOCKS 512 // fused GEMV up to this many blocks with work (each block transforms its own input)
#define MMH_GEMV_WAVES  1280 // GEMV waves the k split aims at
#define MMH_GEMV_MIN_KSLICES 16 // fewest 16-value k slices per GEMV wave from the k split
#define MMH_GEMV_MAX_SPLIT 64 // largest k split
// rows of one expert per GEMV pass with ids and several tokens: 2 for 2 tokens, 4 for 3 or more (the 4-row instance decodes a
// tile once for up to 4 rows; for 2 tokens it measured 3-4% slower in VRAM, from 3 tokens it is 6-9% faster with the verify
// batch sharing of GLM-5.3 and 0.5-5% faster without sharing, in VRAM and host memory; 8 rows are slower in VRAM)
#define MMH_GEMV_IDS_ROWS_2TOK 2
#define MMH_GEMV_IDS_ROWS      4

// src1 precision of an op that does not set one (GGML_PREC_F32 = F32 GEMV only, GGML_PREC_F16 = fp16 GEMM)
#define MMH_PREC_DEFAULT GGML_PREC_F16

struct mmh_params {
    int64_t k;       // weight ne0
    int64_t n_exp;   // weight ne2
    int64_t nb01;
    int64_t nb02;

    int64_t n_rows;  // ne1*ne2*ne3 of dst, < 2^30
    int     ne1;
    int     ne2;
    int64_t nb1;
    int64_t nb2;
    int64_t nb3;

    int     ne11;
    int     bcast;   // without ids: rows of b per weight matrix along dim 2
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    const int32_t * ids;
    int64_t ids_nb0;
    int64_t ids_nb1;

    const float * x;
    const char  * rot_in;
    int64_t rin_nb1;
    const char  * rot_out;
    int64_t rout_nb0;
    int64_t rout_nb1;
    float * dst;

    int rin_f16;
    int rout_f16;
    int had;
    float hscale;    // 1/sqrt(had)

    // expert cache (expert-l1.cuh), as for MUL_MAT_ID: the weight matrix of expert e is read from its arena slot, in
    // exclusive mode a miss from the host slot of e; null without the cache. rot_in / rot_out of a resident expert are
    // read from the cache's rotation arenas at the same slot (rin_cache / rout_cache, null when the cache has none),
    // of any other expert from the tensor by e.
    const void     * w_cache;
    const int32_t  * w_slots;
    const int32_t  * w_host_slots;
    const uint64_t * w_host_addresses;
    bool w_remote;   // routed weights come over the host link (mapped host buffer or expert cache), not from VRAM
    const char     * rin_cache;
    const char     * rout_cache;
};

static __device__ __forceinline__ int mmh_expert(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    int e;
    if (p.ids) {
        e = *(const int32_t *) ((const char *) p.ids + i1*p.ids_nb0 + i2*p.ids_nb1);
    } else {
        e = i2 / p.bcast;
    }
    // invalid ids give wrong rows, never an access outside the weight
    return e < 0 ? 0 : (e >= p.n_exp ? (int) p.n_exp - 1 : e);
}

static __device__ __forceinline__ const float * mmh_src_row(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    const int i3 = r / (p.ne1*p.ne2);
    if (p.ids) {
        return (const float *) ((const char *) p.x + (i1 % p.ne11)*p.nb11 + i2*p.nb12);
    }
    return (const float *) ((const char *) p.x + i1*p.nb11 + i2*p.nb12 + i3*p.nb13);
}

static __device__ __forceinline__ float * mmh_dst_row(const mmh_params & p, const int r) {
    const int i1 = r % p.ne1;
    const int i2 = (r / p.ne1) % p.ne2;
    const int i3 = r / (p.ne1*p.ne2);
    return (float *) ((char *) p.dst + i1*p.nb1 + i2*p.nb2 + i3*p.nb3);
}

static __device__ __forceinline__ float mmh_rot(const char * base, const int f16, const int64_t i) {
    return f16 ? __half2float(((const half *) base)[i]) : ((const float *) base)[i];
}

// a wave-uniform value in a scalar register, so that addresses built from it stay scalar
static __device__ __forceinline__ int mmh_uniform(const int v) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_readfirstlane(v);
#else
    return v;
#endif // defined(GGML_USE_HIP)
}

// the weight matrix of expert e (wave-uniform): the tensor, or the expert cache slot that holds it
static __device__ __forceinline__ const char * mmh_weight_matrix(const mmh_params & p, const char * w, const int e) {
    const ggml_cuda_expert_source src =
        ggml_cuda_expert_cache_select(w, p.w_cache, p.w_slots, e, p.w_host_slots, p.w_host_addresses);
    const uint64_t a = (uint64_t) (uintptr_t) src.data + (uint64_t) src.channel*p.nb02;
#if defined(GGML_USE_HIP)
    return (const char *) (uintptr_t) (((uint64_t) __builtin_amdgcn_readfirstlane((uint32_t) (a >> 32)) << 32) |
        (uint32_t) __builtin_amdgcn_readfirstlane((uint32_t) a));
#else
    return (const char *) (uintptr_t) a;
#endif // defined(GGML_USE_HIP)
}

// the rot_in / rot_out row of expert e (wave-uniform): the rotation arena slot of a resident expert, else the tensor row
static __device__ __forceinline__ const char * mmh_rot_row(const mmh_params & p, const char * rot, const char * cache,
        const int64_t nb1, const int e) {
    if (cache != nullptr) {
        const int slot = mmh_uniform(p.w_slots[e]);
        if (slot >= 0) {
            return cache + slot*nb1;
        }
    }
    return rot + e*nb1;
}

static __device__ __forceinline__ half2 mmh_as_half2(const uint32_t v) {
    half2 h;
    memcpy(&h, &v, sizeof(h));
    return h;
}

static __device__ __forceinline__ uint32_t mmh_as_u32(const half2 h) {
    uint32_t v;
    memcpy(&v, &h, sizeof(v));
    return v;
}

// c + w.x*x0 + w.y*x1 for a half2 w, fp32 FMAs (v_fma_mix_f32)
static __device__ __forceinline__ float mmh_fma2(const uint32_t w, const float x0, const float x1, const float c) {
    const float2 f = __half22float2(mmh_as_half2(w));
    return fmaf(f.y, x1, fmaf(f.x, x0, c));
}

// c + w.x*x.x + w.y*x.y for half2 w and x, fp32 sum (v_dot2_f32_f16)
static __device__ __forceinline__ float mmh_dot2(const uint32_t w, const uint32_t x, const float c) {
#if defined(V_DOT2_F32_F16_AVAILABLE)
    typedef _Float16 h2_t __attribute__((ext_vector_type(2)));
    h2_t a;
    h2_t b;
    memcpy(&a, &w, sizeof(a));
    memcpy(&b, &x, sizeof(b));
    return __builtin_amdgcn_fdot2(a, b, c, false);
#else
    const float2 f = __half22float2(mmh_as_half2(x));
    return mmh_fma2(w, f.x, f.y, c);
#endif // defined(V_DOT2_F32_F16_AVAILABLE)
}

// in-place normalized Hadamard of 128 values, value c = 32*j + lane; blocks of `had` (32, 64 or 128)
static __device__ __forceinline__ void mmh_fwht(float v[4], const int lane, const int had, const float scale) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] *= scale;
    }
#pragma unroll
    for (int h = 1; h < 32; h *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xFFFFFFFF, v[j], h, 32);
            v[j] = (lane & h) ? o - v[j] : v[j] + o;
        }
    }
    if (had >= 64) {
        float a = v[0], b = v[1];
        v[0] = a + b;
        v[1] = a - b;
        a = v[2]; b = v[3];
        v[2] = a + b;
        v[3] = a - b;
    }
    if (had >= 128) {
        float a = v[0], b = v[2];
        v[0] = a + b;
        v[2] = a - b;
        a = v[1]; b = v[3];
        v[1] = a + b;
        v[3] = a - b;
    }
}

#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))
#define MMH_DPP_SUMS
#endif // defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4))

#if defined(MMH_DPP_SUMS)
// v + v of lane ^ m inside each row of 16 lanes (DPP quad_perm for m = 1, 2, row_xmask for 4, 8)
template <int m>
static __device__ __forceinline__ float mmh_add_xor(const float v) {
    constexpr int ctrl = m == 1 ? 0xB1 : m == 2 ? 0x4E : 0x160 + m;
    return v + __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), ctrl, 0xF, 0xF, false));
}
#endif // defined(MMH_DPP_SUMS)

// the sum over the 16 lanes of each half of the wave (xor butterfly, the same value in every lane)
static __device__ __forceinline__ float mmh_sum16(float v) {
#if defined(MMH_DPP_SUMS)
    v = mmh_add_xor<1>(v);
    v = mmh_add_xor<2>(v);
    v = mmh_add_xor<4>(v);
    return mmh_add_xor<8>(v);
#else
#pragma unroll
    for (int m = 1; m < 16; m *= 2) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, m, 32);
    }
    return v;
#endif // defined(MMH_DPP_SUMS)
}

// the sum over the 32 lanes of the wave, the same value in every lane
static __device__ __forceinline__ float mmh_sum32(float v) {
    v = mmh_sum16(v);
#if defined(MMH_DPP_SUMS)
    return v + __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xFEDCBA98u, false, false));
#else
    return v + __shfl_xor_sync(0xFFFFFFFF, v, 16, 32);
#endif // defined(MMH_DPP_SUMS)
}

// fp16 GEMV input: one 128-value block v (value c = 32*j + lane) rounded to fp16 (saturated to the finite range, NaN
// kept) into the frag layout of halves at xd, and with sums the F32 sums of the rounded values of its 8 slices of 16
// (k = 16*i .. 16*i + 15) at xs[i]
template <bool sums>
static __device__ __forceinline__ void mmh_store_h16(const float v[4], const int lane, half * xd, float * xs) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = 32*j + lane;
        const half h = __float2half_rn(fabsf(v[j]) > 65504.0f ? copysignf(65504.0f, v[j]) : v[j]);
        xd[16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = h;
        if constexpr (sums) {
            const float s = mmh_sum16(__half2float(h));
            if (lane % 16 == 0) {
                xs[2*j + lane/16] = s;
            }
        }
    }
}

// max over the threads of the block, any block size up to 1024
static __device__ __forceinline__ float mmh_block_max(float v, float * s_red) {
#pragma unroll
    for (int h = 16; h > 0; h >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, h, 32));
    }
    const int wave = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        s_red[wave] = v;
    }
    __syncthreads();
    float m = 0.0f;
    for (int i = 0; i < (int) blockDim.x/32; ++i) {
        m = fmaxf(m, s_red[i]);
    }
    return m;
}

// layouts of H(rot_in * x):
// - frag (GEMV): F32, per row and 16 values of k, for g = 0..3: k = 2g, 2g + 1, 2g + 8, 2g + 9; with f16 the same
//   order in fp16 (mmh_store_h16, not scaled), and with xsum also the F32 sums of the rounded values of each row over
//   the n_ranges k ranges of kper 16-value slices of the GEMV waves (xsum[n_ranges*row + range], from the slice sums in
//   dynamic LDS)
// - plain (GEMM, f16): fp16 values of the row scaled by a power of 2, K values per row; rscale holds the inverse scale
// The prologue also clears the k-split tickets of the GEMV that follows it (zero, n_zero).
template <bool frag, bool f16>
static __global__ void __launch_bounds__(256) mmh_prologue(const mmh_params p, half * __restrict__ xh, float * __restrict__ rscale,
        int * __restrict__ zero, const int n_zero, float * __restrict__ xsum, const int n_ranges, const int kper) {
    __shared__ float s_red[8];
    extern __shared__ float s_slice[]; // frag f16 with xsum: the sum of each 16 values of k

    const int r    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;

    for (int i = r*blockDim.x + threadIdx.x; i < n_zero; i += gridDim.x*blockDim.x) {
        zero[i] = 0;
    }

    const int e = mmh_expert(p, r);
    const float * src = mmh_src_row(p, r);
    const char  * rin = mmh_rot_row(p, p.rot_in, p.rin_cache, p.rin_nb1, e);

    float s = 1.0f;
    if constexpr (!frag) {
        float amax = 0.0f;
        for (int64_t i = threadIdx.x; i < p.k; i += blockDim.x) {
            amax = fmaxf(amax, fabsf(src[i]*mmh_rot(rin, p.rin_f16, i)));
        }
        amax = mmh_block_max(amax, s_red);

        // |H v| <= sqrt(had) * max|v|: with the scale, every value stays below 2^14
        const float bound = amax*sqrtf((float) p.had);
        if (bound > 0.0f && bound <= FLT_MAX) {
            int ex;
            frexpf(bound, &ex);
            s = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
        }
    }
    if (threadIdx.x == 0) {
        rscale[r] = 1.0f/s;
    }

    for (int64_t c0 = 128*wave; c0 < p.k; c0 += 128*(blockDim.x/32)) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c = c0 + 32*j + lane;
            v[j] = src[c]*mmh_rot(rin, p.rin_f16, c)*s;
        }
        mmh_fwht(v, lane, p.had, p.hscale);
        if constexpr (frag && f16) {
            if (xsum) {
                mmh_store_h16<true>(v, lane, xh + p.k*r + c0, s_slice + c0/16);
            } else {
                mmh_store_h16<false>(v, lane, xh + p.k*r + c0, nullptr);
            }
            continue;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t c  = c0 + 32*j + lane;
            if constexpr (frag) {
                ((float *) xh)[p.k*r + 16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = v[j];
            } else {
                xh[p.k*r + c] = __float2half_rn(v[j]);
            }
        }
    }

    if constexpr (frag && f16) {
        if (xsum) {
            __syncthreads();
            const int kslices = p.k/16;
            for (int t = wave; t < n_ranges; t += blockDim.x/32) {
                float sum = 0.0f;
                for (int i = t*kper + lane; i < min(kslices, (t + 1)*kper); i += 32) {
                    sum += s_slice[i];
                }
                sum = mmh_sum32(sum);
                if (lane == 0) {
                    xsum[(int64_t) n_ranges*r + t] = sum;
                }
            }
        }
    }
}

// rows grouped by weight matrix: rows_sorted, and items (expert, first row, row count <= mt)
// one block, counts in LDS by atomics, so the order inside one expert is arbitrary; each row is computed alone
// With items_r, the GEMM items go to two lists: the experts read from VRAM (an expert cache slot, tile mt) and the
// others (tile mt_r); without it all go to items.
static __global__ void __launch_bounds__(1024) mmh_route(const mmh_params p, const int mt, const int mt_r,
        int * __restrict__ eid, int * __restrict__ rank, int * __restrict__ rows_sorted, int * __restrict__ items,
        int * __restrict__ n_items, int * __restrict__ items_r, int * __restrict__ n_items_r) {
    __shared__ int s_count[MMH_MAX_EXPERTS];
    __shared__ int s_offs[MMH_MAX_EXPERTS];
    __shared__ int s_rows[1024];
    __shared__ int s_items[1024];
    __shared__ int s_items_r[1024];

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int64_t e = tid; e < p.n_exp; e += nth) {
        s_count[e] = 0;
    }
    __syncthreads();

    // eid and rank of row r are written and read by the same thread
    for (int r = tid; r < p.n_rows; r += nth) {
        const int e = mmh_expert(p, r);
        eid[r]  = e;
        rank[r] = atomicAdd(&s_count[e], 1);
    }
    __syncthreads();

    // each thread scans a contiguous range of experts
    const int64_t per = (p.n_exp + nth - 1)/nth;
    const int64_t e0  = std::min<int64_t>(p.n_exp, tid*per);
    const int64_t e1  = std::min<int64_t>(p.n_exp, e0 + per);
    auto remote = [&](const int64_t e) {
        return items_r != nullptr && (p.w_slots == nullptr || p.w_slots[e] < 0);
    };
    int sum_rows    = 0;
    int sum_items   = 0;
    int sum_items_r = 0;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        sum_rows += c;
        if (remote(e)) {
            sum_items_r += (c + mt_r - 1)/mt_r;
        } else {
            sum_items   += (c + mt - 1)/mt;
        }
    }
    s_rows[tid]    = sum_rows;
    s_items[tid]   = sum_items;
    s_items_r[tid] = sum_items_r;
    __syncthreads();
    for (int h = 1; h < nth; h *= 2) {
        const int a = tid >= h ? s_rows[tid - h]    : 0;
        const int b = tid >= h ? s_items[tid - h]   : 0;
        const int d = tid >= h ? s_items_r[tid - h] : 0;
        __syncthreads();
        s_rows[tid]    += a;
        s_items[tid]   += b;
        s_items_r[tid] += d;
        __syncthreads();
    }
    int row    = s_rows[tid]    - sum_rows;
    int item   = s_items[tid]   - sum_items;
    int item_r = s_items_r[tid] - sum_items_r;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        s_offs[e] = row;
        const bool r_list = remote(e);
        const int  m      = r_list ? mt_r : mt;
        int * list = r_list ? items_r : items;
        int & k    = r_list ? item_r  : item;
        for (int i = 0; i < c; i += m) {
            list[3*k + 0] = (int) e;
            list[3*k + 1] = row + i;
            list[3*k + 2] = min(m, c - i);
            ++k;
        }
        row += c;
    }
    if (tid == nth - 1) {
        *n_items = s_items[tid];
        if (items_r != nullptr) {
            *n_items_r = s_items_r[tid];
        }
    }
    __syncthreads();

    for (int r = tid; r < p.n_rows; r += nth) {
        rows_sorted[s_offs[eid[r]] + rank[r]] = r;
    }
}

// EXL3: 16x16 tiles of 256 positions, position p of a tile ends at bit ((p + 1)*b2)/2 of a ring of 32-bit words read
// MSB first, its state is the 16-bit window that ends there (b2 = 2*bits). Lane l decodes positions 8l .. 8l + 7:
// column c0 = l/4 and c0 + 8, k = 2*(l%4) + {0, 1, 8, 9}, so its 8 windows sit in 3 consecutive words of the ring.
template <int b2>
struct mmh_exl3 {
    static constexpr int words      = 4*b2;
    static constexpr int tile_bytes = 16*b2;

    int i0, i1, i2; // word index of the 3 words of this lane
    int o;          // bit offset of the first window in word i0

    __device__ __forceinline__ explicit mmh_exl3(const int lane) {
        const int s0 = (((8*lane + 1)*b2) >> 1) - 16 + 128*b2;
        const int wb = s0/32;
        i0 = wb % words;
        i1 = (wb + 1) % words;
        i2 = (wb + 2) % words;
        o  = s0 % 32;
    }

    // end of window j relative to the start of window 0 of the lane
    static constexpr int end(const int j) {
        return (((j + 1)*b2) >> 1) - (b2 >> 1) + 16;
    }

    // whether the 8 windows of every lane end within its first 2 words (then the third word is not read)
    static constexpr bool fits_two_words() {
        for (int lane = 0; lane < 32; ++lane) {
            if (((((8*lane + 1)*b2) >> 1) - 16 + 128*b2) % 32 + end(7) > 64) {
                return false;
            }
        }
        return true;
    }
    static constexpr bool two_words = fits_two_words();
};

// the low 32 bits of (hi:lo) >> s, 0 <= s < 32
static __device__ __forceinline__ uint32_t mmh_funnel(const uint32_t hi, const uint32_t lo, const int s) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_alignbit(hi, lo, s);
#else
    return __funnelshift_r(lo, hi, s);
#endif // defined(GGML_USE_HIP)
}

#if defined(GGML_USE_HIP)
typedef unsigned short mmh_u16x2 __attribute__((ext_vector_type(2)));
#endif // defined(GGML_USE_HIP)

// the trellis hash x*c + a mod 2^32 of the 16-bit state x (the low 16 bits of s, the bits above are ignored).
// HIP: x*(c mod 2^16) + (a mod 2^16) + ((x*(c >> 16) + (a >> 16)) mod 2^16) << 16, where the high half is one packed
// 16-bit multiply (add) whose low lane is zero (or a mod 2^16), and the rest one 24-bit multiply-add: two full-rate
// instructions instead of v_mul_lo_u32 (a quarter of the rate) or three 24-bit ones. Plain C, so that the compiler
// knows their latencies.
template <uint32_t c, uint32_t a>
static __device__ __forceinline__ uint32_t mmh_trellis_hash(const uint32_t s) {
#if defined(GGML_USE_HIP)
    const mmh_u16x2 xv = { (unsigned short) s, (unsigned short) s };
    const mmh_u16x2 cv = { 0, (unsigned short) (c >> 16) };
    const mmh_u16x2 av = { (unsigned short) (a & 0xFFFFu), (unsigned short) (a >> 16) };
    const mmh_u16x2 tv = xv*cv + av;
    uint32_t t;
    memcpy(&t, &tv, sizeof(t));
    return (s & 0xFFFFu)*(c & 0xFFFFu) + t;
#else
    return (s & 0xFFFFu)*c + a;
#endif // defined(GGML_USE_HIP)
}

// mul1 codebook: value = fp16(u*k_inv + k_bias) of u = byte_sum(state*0x83DCD12D) + 1024, an integer in [1024, 2044]
#define MMH_MUL1_KINV 0.00676727294921875f // fp16 0x1EEE
#define MMH_MUL1_BIAS (-10.3828125f)       // fp16 0xC931

// two values u of the mul1 codebook as half2 (exact: 1024 + byte sum, the fp16 bit pattern 0x6400 + byte sum)
static __device__ __forceinline__ uint32_t mmh_mul1_u_pair(const uint32_t s0, const uint32_t s1) {
    const uint32_t p0 = mmh_trellis_hash<0x83DCD12Du, 0>(s0);
    const uint32_t p1 = mmh_trellis_hash<0x83DCD12Du, 0>(s1);
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_sad_hi_u8(p1, 0u, __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u));
#else
    return __dp4a(p0, 0x01010101u, 0x6400u) | (__dp4a(p1, 0x01010101u, 0x6400u) << 16);
#endif
}

// two mul1 codebook values as half2: fp16(u*k_inv + k_bias), one fp16 rounding
static __device__ __forceinline__ uint32_t mmh_mul1_pair(const uint32_t s0, const uint32_t s1) {
    return mmh_as_u32(__hfma2(mmh_as_half2(mmh_mul1_u_pair(s0, s1)), mmh_as_half2(0x1EEE1EEEu), mmh_as_half2(0xC931C931u)));
}

// two mcg or 3inst codebook values as half2: each hash keeps its sign and low bits as two fp16 halves of exponent
// 12..15, which are added in fp16
static __device__ __forceinline__ uint32_t mmh_half_sum_pair(uint32_t p0, uint32_t p1) {
    p0 = (p0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    p1 = (p1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
#if defined(GGML_USE_HIP)
    const uint32_t lo = __builtin_amdgcn_perm(p1, p0, 0x05040100u);
    const uint32_t hi = __builtin_amdgcn_perm(p1, p0, 0x07060302u);
#else
    const uint32_t lo = __byte_perm(p0, p1, 0x5410);
    const uint32_t hi = __byte_perm(p0, p1, 0x7632);
#endif // defined(GGML_USE_HIP)
    return mmh_as_u32(__hadd2(mmh_as_half2(lo), mmh_as_half2(hi)));
}

// EXL3 types: twice the bits (0 for other types) and the codebook (0 = mul1, 1 = mcg, 2 = 3inst)
static constexpr bool mmh_is_exl3(const ggml_type type) {
    return (type >= GGML_TYPE_EXL3_M1 && type <= GGML_TYPE_EXL3_M3H) || (type >= GGML_TYPE_EXL3_G1 && type <= GGML_TYPE_EXL3_G8) ||
           (type >= GGML_TYPE_EXL3_T1 && type <= GGML_TYPE_EXL3_T8);
}

static constexpr int mmh_exl3_b2(const ggml_type type) {
    return !mmh_is_exl3(type) ? 0 : (type - GGML_TYPE_EXL3_M1) % 16 < 8 ? 2*((type - GGML_TYPE_EXL3_M1) % 16 + 1) :
        2*((type - GGML_TYPE_EXL3_M1) % 16 - 7) + 1;
}

static constexpr int mmh_exl3_codebook(const ggml_type type) {
    return mmh_is_exl3(type) ? (type - GGML_TYPE_EXL3_M1)/16 : -1;
}

// one lane's share of the 16x16 tile (kt, nt) of a 128-row group, as half2 over k, k + 1:
// out[0] = (c0, r0), out[1] = (c0, r0 + 8), out[2] = (c0 + 8, r0), out[3] = (c0 + 8, r0 + 8), r0 = 2*(lane%4)
// F32 weights give k in out and k + 1 in lo, as F32. fetch() only loads, so the loads can be issued ahead.
// coal (EXL3): the wave loads each tile once, one word per lane, and decode() takes the 3 words of a lane from their
// lanes (ds_bpermute) instead of 3 loads per lane of the whole tile: fewer memory instructions, more LDS-unit work.
// The words and so the results are the same.
static __device__ __forceinline__ uint32_t mmh_lane_word(const uint32_t v, const int src) {
#if defined(GGML_USE_HIP)
    return (uint32_t) __builtin_amdgcn_ds_bpermute(4*src, (int) v);
#else
    return __shfl_sync(0xFFFFFFFF, v, src, 32);
#endif // defined(GGML_USE_HIP)
}

// EXL3 tiles are read through a buffer resource of the group (scalar base, scalar tile offset, the lane's word offset in
// a vector register), so that a tile load needs no vector address arithmetic
#if defined(GGML_USE_HIP) && defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_make_buffer_rsrc) &&     __has_builtin(__builtin_amdgcn_raw_buffer_load_b32) && defined(RDNA4)
#define MMH_BUFFER_LOADS
#endif // ...

template <ggml_type type, bool coal = false>
struct mmh_weight {
    static constexpr int  b2        = mmh_exl3_b2(type);
    static constexpr int  codebook  = mmh_exl3_codebook(type);
    static constexpr bool is_exl3   = b2 > 0;
    static constexpr int  words     = is_exl3 ? 4*b2 : 0; // 32-bit words of a tile
    static constexpr bool coalesced = is_exl3 && coal;
    static constexpr bool two      = is_exl3 && mmh_exl3<is_exl3 ? b2 : 2>::two_words; // the lane reads 2 words of 3
    static constexpr int  n_raw     = coalesced ? (words > 32 ? 2 : 1) : is_exl3 ? (two ? 2 : 3) : type == GGML_TYPE_F16 ? 4 : 8;

    struct raw {
        uint32_t v[n_raw];
    };

    const char * group; // wave-uniform base of the 128-row group
#if defined(MMH_BUFFER_LOADS)
    __amdgpu_buffer_rsrc_t rsrc; // EXL3: the group as a raw buffer (0x31004000: the RDNA4 raw 32-bit buffer config)
#endif // defined(MMH_BUFFER_LOADS)
    int64_t  nb01;
    uint32_t off[3];    // EXL3: byte offsets of the 3 words of this lane in a tile (coal: word indices); F16/F32: offset of (c0, r0)
    int      o;         // EXL3: bit offset of the first window in the first word
    int      lane;

    __device__ __forceinline__ mmh_weight(const char * group, const int64_t nb01, const int lane) : group(group), nb01(nb01), lane(lane) {
#if defined(MMH_BUFFER_LOADS)
        rsrc = __builtin_amdgcn_make_buffer_rsrc((void *) group, 0, 0x7FFFFFFF, 0x31004000);
#endif // defined(MMH_BUFFER_LOADS)
        if constexpr (coalesced) {
            const mmh_exl3<b2> c(lane);
            off[0] = c.i0; off[1] = c.i1; off[2] = c.i2; o = c.o;
        } else if constexpr (is_exl3) {
            const mmh_exl3<b2> c(lane);
            off[0] = 4*c.i0; off[1] = 4*c.i1; off[2] = 4*c.i2; o = c.o;
        } else {
            off[0] = (lane/4)*(uint32_t) nb01 + 2*(lane % 4)*(type == GGML_TYPE_F32 ? 4 : 2);
            off[1] = off[2] = 0;
            o = 0;
        }
    }

    __device__ __forceinline__ raw fetch(const int kt, const int nt) const {
        raw r;
        if constexpr (coalesced) {
#if defined(MMH_BUFFER_LOADS)
            const int so = (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
            r.v[0] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, 4*(words >= 32 ? lane : lane % words), so, 0);
            if constexpr (words > 32) {
                r.v[1] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, 4*(32 + lane % (words - 32)), so, 0);
            }
#else
            const uint32_t * t = (const uint32_t *) (group + (int64_t) (8*kt + nt)*mmh_exl3<b2>::tile_bytes);
            r.v[0] = t[words >= 32 ? lane : lane % words];
            if constexpr (words > 32) {
                r.v[1] = t[32 + lane % (words - 32)];
            }
#endif // defined(MMH_BUFFER_LOADS)
        } else if constexpr (is_exl3) {
#if defined(MMH_BUFFER_LOADS)
            const int so = (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
#pragma unroll
            for (int i = 0; i < n_raw; ++i) {
                r.v[i] = __builtin_amdgcn_raw_buffer_load_b32(rsrc, off[i], so, 0);
            }
#else
            const char * t = group + (int64_t) (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
#pragma unroll
            for (int i = 0; i < n_raw; ++i) {
                r.v[i] = *(const uint32_t *) (t + off[i]);
            }
#endif // defined(MMH_BUFFER_LOADS)
        } else if constexpr (type == GGML_TYPE_F16) {
            const char * a = group + 16*nt*nb01 + 32*kt + off[0];
            const char * b = a + 8*nb01;
            r.v[0] = *(const uint32_t *) a;
            r.v[1] = *(const uint32_t *) (a + 16);
            r.v[2] = *(const uint32_t *) b;
            r.v[3] = *(const uint32_t *) (b + 16);
        } else {
            static_assert(type == GGML_TYPE_F32, "unsupported weight type");
            const char * a = group + 16*nt*nb01 + 64*kt + off[0];
            const char * b = a + 8*nb01;
            const float2 f[4] = { *(const float2 *) a, *(const float2 *) (a + 32), *(const float2 *) b, *(const float2 *) (b + 32) };
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                r.v[2*i + 0] = __float_as_uint(f[i].x);
                r.v[2*i + 1] = __float_as_uint(f[i].y);
            }
        }
        return r;
    }

    // coal: word i of the tile
    __device__ __forceinline__ uint32_t word(const raw & r, const int i) const {
        if constexpr (words > 32) {
            const uint32_t a = mmh_lane_word(r.v[0], i & 31);
            const uint32_t b = mmh_lane_word(r.v[1 % n_raw], (i - 32) & 31);
            return i < 32 ? a : b;
        } else {
            return mmh_lane_word(r.v[0], i);
        }
    }

    // u: mul1 gives the values u of the codebook (the affine map is applied by the caller), other types their values
    template <bool u = false>
    __device__ __forceinline__ void decode(const raw & r, uint32_t out[4], uint32_t lo[4]) const {
        if constexpr (is_exl3) {
            uint32_t w0, w1, w2;
            if constexpr (coalesced) {
                w0 = word(r, off[0]);
                w1 = word(r, off[1]);
                w2 = two ? 0 : word(r, off[2]);
            } else {
                w0 = r.v[0];
                w1 = r.v[1 % n_raw];
                w2 = two ? 0 : r.v[2 % n_raw];
            }
            // the 96 bits of the 3 words from the first window of the lane on; windows only need their low 16 bits right.
            // o is never 0 for the EXL3 bit widths, so 32 - o is a valid alignbit shift (no 64-bit shifts)
            const uint32_t v0 = mmh_funnel(w0, w1, 32 - o);
            const uint32_t v1 = mmh_funnel(w1, w2, 32 - o);
            const uint32_t v2 = w2 << o;
            uint32_t st[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = mmh_exl3<b2>::end(j);
                st[j] = e <= 32 ? v0 >> (32 - e) : e < 64 ? mmh_funnel(v0, v1, 64 - e) : e == 64 ? v1 :
                    e < 96 ? mmh_funnel(v1, v2, 96 - e) : v2;
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (codebook == 0) {
                    out[i] = u ? mmh_mul1_u_pair(st[2*i], st[2*i + 1]) : mmh_mul1_pair(st[2*i], st[2*i + 1]);
                } else if constexpr (codebook == 1) {
                    // mcg: x*0xCBAC1FED
                    out[i] = mmh_half_sum_pair(mmh_trellis_hash<0xCBAC1FEDu, 0>(st[2*i]), mmh_trellis_hash<0xCBAC1FEDu, 0>(st[2*i + 1]));
                } else {
                    // 3inst: x*89226354 + 64248484
                    out[i] = mmh_half_sum_pair(mmh_trellis_hash<89226354u, 64248484u>(st[2*i]),
                        mmh_trellis_hash<89226354u, 64248484u>(st[2*i + 1]));
                }
            }
            GGML_UNUSED(lo);
        } else if constexpr (type == GGML_TYPE_F16) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = r.v[i];
            }
            GGML_UNUSED(lo);
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = r.v[2*i + 0];
                lo[i]  = r.v[2*i + 1];
            }
        }
    }
};

// one pass of the GEMV over the k range [kb, ke) of a wave for the rows rowp[0 .. nc - 1] (nc <= rows): stages the input
// (fused), decodes each tile once for all rows and leaves the row sums of the wave's 16 columns in s_part[ksp].
// h16: fp16 input (8 bytes per lane and 16 k) and v_dot2 products; mul1 then multiplies the codebook values u and
// applies the affine map once to the sums: k_inv*sum(u*x) + k_bias*sum(x), sum(x) over [kb, ke) from the staged slice
// sums (fused) or the range sums of the prologue (xsum: this wave's range of row 0, rows n_ranges apart).
template <ggml_type type, int rows, bool fused, bool coal, int srows, bool h16>
static __device__ __forceinline__ void mmh_gemv_pass(const mmh_params & p, const mmh_weight<type, coal> & wl,
        const void * __restrict__ xh, const float * __restrict__ xsum, const int n_ranges, float4 * s_xf,
        float (*s_part)[srows][MMH_NG], const int * rowp, const int nc, const int e, const int kb, const int ke,
        const int kslices, const int koff, const int nks, const int lane, const int wave, const int nt, const int ksp) {
    typedef mmh_weight<type, coal> wtype;
    typedef typename std::conditional<h16, uint2, float4>::type xtype;
    constexpr bool defer = h16 && wtype::codebook == 0;
    constexpr int pf = rows <= 2 ? 4 : 2;
    // the first weight loads go out before the input (the fused staging does not wait for them). Tile indices past
    // the range are clamped to its last tile, so that the k loop below loads without conditions (its ring of pf
    // loads stays in fixed registers and each step waits only for its own load)
    typename wtype::raw wr[pf];
    if (kb < ke) {
#pragma unroll
        for (int d = 0; d < pf; ++d) {
            wr[d] = wl.fetch(min(kb + d, ke - 1), nt);
        }
    }

    // input row bases are wave-uniform, the lane adds its 16-byte (h16: 8-byte) group of each 16 k
    const xtype * xr[rows];
    float sx[rows]; // defer, prologue: the range sum of row i
    if constexpr (fused) {
        // as mmh_prologue<true, h16>: one 128-value block of one row per wave and pass; the waves of the block
        // read the previous rows only before the barrier after their k loop. The slice sums follow the rows.
        float * s_xs = (float *) ((char *) s_xf + (size_t) srows*nks*4*sizeof(xtype));
        const char * rin = mmh_rot_row(p, p.rot_in, p.rin_cache, p.rin_nb1, e);
        const int nch = nks/8;
        for (int t = wave; t < nc*nch; t += blockDim.x/32) {
            const int i  = t/nch;
            const int ch = t % nch;
            const float * src = mmh_src_row(p, rowp[i]) + 16*koff + 128*ch;
            const char  * rc  = rin + (16*koff + 128*ch)*(p.rin_f16 ? 2 : 4);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                v[j] = src[c]*mmh_rot(rc, p.rin_f16, c);
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            if constexpr (h16) {
                mmh_store_h16<defer>(v, lane, (half *) s_xf + (16*nks*i + 128*ch), s_xs + (nks*i + 8*ch));
            } else {
                float * xd = (float *) s_xf + (16*nks*i + 128*ch);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = 32*j + lane;
                    xd[16*(c/16) + 4*((c % 8)/2) + 2*((c % 16)/8) + c % 2] = v[j];
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            xr[i] = (const xtype *) s_xf + 4*nks*(i < nc ? i : 0) + lane % 4;
        }
    } else {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            const int64_t r = mmh_uniform(rowp[i < nc ? i : 0]);
            xr[i] = (const xtype *) xh + r*(4*kslices) + lane % 4;
            if constexpr (defer) {
                sx[i] = xsum[r*n_ranges];
            }
        }
    }

    float acc[rows][2];
#pragma unroll
    for (int i = 0; i < rows; ++i) {
        acc[i][0] = 0.0f;
        acc[i][1] = 0.0f;
    }

    // the input of the next xpf steps (slot d % xpf holds step kt + d): staged in LDS (fused) for several rows only one
    // step ahead, from global memory (prologue) two steps ahead; fewer registers than the ring of pf weight loads
    constexpr int xpf = fused ? (rows > 1 ? 1 : pf) : 2;
    xtype xv[xpf][rows];
    if (kb < ke) {
#pragma unroll
        for (int d = 0; d < xpf; ++d) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    xv[d][i] = xr[i][4*(min(kb + d, ke - 1) - koff)];
                }
            }
        }
    }

    // one k step: decode the tile and add its products to the rows
    auto step = [&](const typename wtype::raw & cw, const xtype (&cx)[rows]) {
        uint32_t wf[4];
        uint32_t wlo[4];
        wl.template decode<defer>(cw, wf, wlo);
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            if (rows == 1 || i < nc) {
                const xtype x = cx[i];
                float a0 = acc[i][0];
                float a1 = acc[i][1];
                if constexpr (h16) {
                    a0 = mmh_dot2(wf[0], x.x, a0);
                    a0 = mmh_dot2(wf[1], x.y, a0);
                    a1 = mmh_dot2(wf[2], x.x, a1);
                    a1 = mmh_dot2(wf[3], x.y, a1);
                } else if constexpr (type == GGML_TYPE_F32) {
                    a0 = fmaf(__uint_as_float(wf[0]),  x.x, a0);
                    a0 = fmaf(__uint_as_float(wlo[0]), x.y, a0);
                    a0 = fmaf(__uint_as_float(wf[1]),  x.z, a0);
                    a0 = fmaf(__uint_as_float(wlo[1]), x.w, a0);
                    a1 = fmaf(__uint_as_float(wf[2]),  x.x, a1);
                    a1 = fmaf(__uint_as_float(wlo[2]), x.y, a1);
                    a1 = fmaf(__uint_as_float(wf[3]),  x.z, a1);
                    a1 = fmaf(__uint_as_float(wlo[3]), x.w, a1);
                } else {
                    a0 = mmh_fma2(wf[0], x.x, x.y, a0);
                    a0 = mmh_fma2(wf[1], x.z, x.w, a0);
                    a1 = mmh_fma2(wf[2], x.x, x.y, a1);
                    a1 = mmh_fma2(wf[3], x.z, x.w, a1);
                }
                acc[i][0] = a0;
                acc[i][1] = a1;
            }
        }
    };

    // whole groups of pf steps: slot d holds step kt + d and is refilled with step kt + d + pf (clamped)
    int kt = kb;
    for (; kt + pf <= ke; kt += pf) {
#pragma unroll
        for (int d = 0; d < pf; ++d) {
            const typename wtype::raw cw = wr[d];
            xtype cx[rows];
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                cx[i] = xv[d % xpf][i];
            }
            const int kn = min(kt + d + pf, ke - 1);
            wr[d] = wl.fetch(kn, nt);
            const int kx = min(kt + d + xpf, ke - 1);
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    xv[d % xpf][i] = xr[i][4*(kx - koff)];
                }
            }
            step(cw, cx);
        }
    }
    // the last ke - kt < pf steps, already loaded in slots 0 ..
#pragma unroll
    for (int d = 0; d < pf - 1; ++d) {
        if (kt + d < ke) {
            xtype cx[rows];
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                if (rows == 1 || i < nc) {
                    cx[i] = d < xpf ? xv[d % xpf][i] : xr[i][4*(kt + d - koff)];
                }
            }
            step(wr[d], cx);
        }
    }

#pragma unroll
    for (int i = 0; i < rows; ++i) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 1, 32);
            acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 2, 32);
        }
        if constexpr (defer) {
            if (rows == 1 || i < nc) {
                if constexpr (fused) {
                    // the staged slice sums of row i (after the staged rows)
                    const float * xs = (const float *) ((const char *) s_xf + (size_t) srows*nks*4*sizeof(xtype)) + nks*i - koff;
                    float sum = 0.0f;
                    for (int t = kb + lane; t < ke; t += 32) {
                        sum += xs[t];
                    }
                    sx[i] = mmh_sum32(sum);
                }
                const float b = MMH_MUL1_BIAS*sx[i];
                acc[i][0] = fmaf(MMH_MUL1_KINV, acc[i][0], b);
                acc[i][1] = fmaf(MMH_MUL1_KINV, acc[i][1], b);
            }
        }
        if (i < nc && lane % 4 == 0) {
            s_part[ksp][i][16*nt + lane/4]     = acc[i][0];
            s_part[ksp][i][16*nt + lane/4 + 8] = acc[i][1];
        }
    }
}

// GEMV: a block of 8*ksplit waves owns one 128-row group g (blockIdx.x) of one expert, for the k range blockIdx.z of
// gridDim.z; wave = column tile nt + 8*k split. Its rows come from the routing pass (items) or are collected here: with
// ids, block y is row y and runs only if y is the first row of its expert, then for every row of that expert. The
// weight and input loads of the next pf k steps are in flight while a step is decoded. With a k split over blocks,
// each block stores its partial sums and the last block of the group adds them in k order, then runs the epilogue;
// it also resets its ticket, so that tickets that start at zero stay zero between ops.
// fused: no prologue, the block writes H(rot_in * x) of its rows for the 128-value blocks of its k range to LDS (the
// same values as the prologue, row scale 1), and its tickets must be zero when it starts. coal: see mmh_weight.
// h16: fp16 input, see mmh_gemv_pass (xsum: the range sums of the prologue, mul1).
template <ggml_type type, int rows, bool fused, bool coal, bool h16 = false>
static __global__ void __launch_bounds__(1024) mmh_gemv(const mmh_params p, const char * __restrict__ w,
        const void * __restrict__ xh, const float * __restrict__ xsum, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, const int ksplit,
        float * __restrict__ part, int * __restrict__ tickets, const int max_chunks) {
    typedef mmh_weight<type, coal> wtype;
    extern __shared__ float4 s_xf[]; // fused: rows x staged k slices x 4 float4 (h16: 4 uint2, then the slice sums)

    __shared__ int   s_rows[MMH_GEMV_LIST];
    __shared__ int   s_eid[MMH_GEMV_LIST];
    __shared__ int   s_wcount[32];
    __shared__ int   s_last;
    __shared__ float s_part[4][rows][MMH_NG];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int nt   = wave % 8;
    const int ksp  = wave / 8;
    const int g    = blockIdx.x;

    int e;
    int nr;
    const int * rowp;
    if (items) {
        if ((int) blockIdx.y >= *n_items) {
            return;
        }
        e    = items[3*blockIdx.y + 0];
        rowp = rows_sorted + items[3*blockIdx.y + 1];
        nr   = items[3*blockIdx.y + 2];
    } else {
        // n_rows <= MMH_GEMV_LIST here: the expert of every row, once
        for (int r = tid; r < p.n_rows; r += blockDim.x) {
            s_eid[r] = mmh_expert(p, r);
        }
        __syncthreads();
        const int y = blockIdx.y;
        e = p.ids ? s_eid[y] : y;
        if (p.ids) {
            bool dup = false;
            for (int r = tid; r < y; r += blockDim.x) {
                dup = dup || s_eid[r] == e;
            }
            if (__syncthreads_or(dup)) {
                return;
            }
        }
        // ordered compaction of the rows of expert e
        int n = 0;
        for (int r0 = p.ids ? y : 0; r0 < p.n_rows; r0 += blockDim.x) {
            const int r = r0 + tid;
            const bool hit = r < p.n_rows && s_eid[r] == e;
#if defined(GGML_USE_HIP)
            const uint64_t mask = __ballot(hit);
            const int before = __popcll(mask & ((1ull << lane) - 1));
            const int count  = __popcll(mask);
#else
            const uint32_t mask = __ballot_sync(0xFFFFFFFF, hit);
            const int before = __popc(mask & ((1u << lane) - 1));
            const int count  = __popc(mask);
#endif
            if (lane == 0) {
                s_wcount[wave] = count;
            }
            __syncthreads();
            int offs = n;
            for (int i = 0; i < wave; ++i) {
                offs += s_wcount[i];
            }
            if (hit && offs + before < MMH_GEMV_LIST) {
                s_rows[offs + before] = r;
            }
            for (int i = 0; i < (int) blockDim.x/32; ++i) {
                n += s_wcount[i];
            }
            __syncthreads();
        }
        nr   = min(n, MMH_GEMV_LIST);
        rowp = s_rows;
    }
    e = mmh_uniform(e);

    const int kz      = blockIdx.z;
    const int kblocks = gridDim.z;
    const int kslices = p.k/16;
    const int per     = (kslices + kblocks*ksplit - 1)/(kblocks*ksplit);
    const int kb      = (kz*ksplit + ksp)*per;
    const int ke      = min(kslices, kb + per);

    // fused: the k slices of the 128-value blocks that hold the k range of the block, from koff
    const int kb_blk = min(kslices, kz*ksplit*per);
    const int koff   = fused ? kb_blk/8*8 : 0;
    const int nks    = fused ? (min(kslices, kb_blk + ksplit*per) + 7)/8*8 - koff : 0;

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    // h16 with the prologue: the range sums of this wave (range kz*ksplit + ksp of kblocks*ksplit)
    int           n_ranges = 0;
    const float * xsr      = nullptr;
    if constexpr (h16 && !fused && wtype::codebook == 0) {
        n_ranges = kblocks*ksplit;
        xsr      = xsum + (kz*ksplit + ksp);
    }
    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_cache, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;

    for (int i0 = 0; i0 < nr; i0 += rows) {
        const int nc = min(rows, nr - i0);

        // a pass decodes each tile once for all its rows; a single row (an expert of one token) takes the one-row pass,
        // whose registers allow the deeper prefetch
        if (rows > 1 && nc == 1) {
            mmh_gemv_pass<type, 1, fused, coal, rows, h16>(p, wl, xh, xsr, n_ranges, s_xf, s_part, rowp + i0, nc, e, kb, ke,
                kslices, koff, nks, lane, wave, nt, ksp);
        } else {
            mmh_gemv_pass<type, rows, fused, coal, rows, h16>(p, wl, xh, xsr, n_ranges, s_xf, s_part, rowp + i0, nc, e, kb, ke,
                kslices, koff, nks, lane, wave, nt, ksp);
        }
        __syncthreads();

        if (kblocks > 1) {
            for (int i = wave; i < nc; i += blockDim.x/32) {
                float * dp = part + (((int64_t) rowp[i0 + i]*gridDim.x + g)*kblocks + kz)*MMH_NG;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.0f;
                    for (int s = 0; s < ksplit; ++s) {
                        sum += s_part[s][i][32*j + lane];
                    }
                    dp[32*j + lane] = sum;
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                const int64_t t = ((int64_t) blockIdx.y*gridDim.x + g)*max_chunks + i0/rows;
                const bool last = atomicAdd(&tickets[t], 1) == kblocks - 1;
                if (last) {
                    tickets[t] = 0;
                }
                s_last = last;
            }
            __syncthreads();
            if (!s_last) {
                continue;
            }
            __threadfence();
        }

        for (int i = wave; i < nc; i += blockDim.x/32) {
            const int r = rowp[i0 + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float sum = 0.0f;
                if (kblocks > 1) {
                    const float * dp = part + ((int64_t) r*gridDim.x + g)*kblocks*MMH_NG + 32*j + lane;
                    for (int z = 0; z < kblocks; ++z) {
                        sum += dp[z*MMH_NG];
                    }
                } else {
                    for (int s = 0; s < ksplit; ++s) {
                        sum += s_part[s][i][32*j + lane];
                    }
                }
                v[j] = sum;
            }
            mmh_fwht(v, lane, p.had, p.hscale);
            float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
            const float rs = fused ? 1.0f : rscale[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = 32*j + lane;
                out[c] = v[j]*rs*mmh_rot(rout, p.rout_f16, c*(p.rout_nb0/(p.rout_f16 ? 2 : 4)));
            }
        }
        __syncthreads();
    }
}

// WMMA GEMM (RDNA4), fp16 input: 8 waves, wave w owns column tile w of the 128-row group; mt rows per block. Per step
// of 32 k: the block stages the fp16 rows in LDS (two buffers, one barrier per step), each wave decodes its 2 tiles
// into a wave-private LDS area (the decode lane order is not the WMMA lane order), then one WMMA per 16 k and m tile,
// accumulating in the WMMA accumulator (its rounding is far below the fp16 input rounding). The loads of the next step
// are issued before the decode. The accumulators go to the epilogue through LDS in passes of up to 32 rows. A k split
// over blocks (gridDim.z) works as in the GEMV.
template <ggml_type type, int mt>
static __global__ void __launch_bounds__(256, 2) mmh_gemm(const mmh_params p, const char * __restrict__ w,
        const half * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, float * __restrict__ part,
        int * __restrict__ tickets) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    typedef _Float16 h8_t __attribute__((ext_vector_type(8)));
    typedef float    f8_t __attribute__((ext_vector_type(8)));
    typedef mmh_weight<type> wtype;

    constexpr int ks  = 32;
    constexpr int xst = ks + 8;      // halves per staged row, with padding
    constexpr int wst = ks + 8;      // halves per decoded weight row
    constexpr int yst = MMH_NG + 4;  // floats per output row
    constexpr int nm  = mt/16;
    constexpr int ec  = mt > 32 ? 32 : mt; // output rows per pass through LDS
    constexpr int main_bytes = (2*mt*xst + 8*16*wst)*(int) sizeof(half);
    constexpr int epi_bytes  = ec*yst*(int) sizeof(float);
    __shared__ __align__(16) char smem[main_bytes > epi_bytes ? main_bytes : epi_bytes];
    __shared__ int s_last;

    if ((int) blockIdx.y >= *n_items) {
        return;
    }
    const int e     = mmh_uniform(items[3*blockIdx.y + 0]);
    const int start = items[3*blockIdx.y + 1];
    const int cnt   = items[3*blockIdx.y + 2];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = mmh_uniform(tid / 32);
    const int g    = blockIdx.x;

    half  * xs = (half *) smem;
    half  * ws = xs + 2*mt*xst + wave*16*wst;
    float * ys = (float *) smem;

    // the 16-byte chunks of the staged rows this thread loads: mt rows x 4
    constexpr int cpr    = 4;
    constexpr int nchunk = cpr*mt;
    constexpr int cpt    = (nchunk + 255)/256;
    const half * xsrc[cpt];
    int  xdst[cpt];
    bool xval[cpt];
#pragma unroll
    for (int t = 0; t < cpt; ++t) {
        const int c    = tid + 256*t;
        const int i    = c/cpr;
        const int part = c % cpr;
        xval[t] = c < nchunk && i < cnt;
        const int64_t r = xval[t] ? rows_sorted[start + i] : 0;
        xsrc[t] = xh + r*p.k + 8*part;
        xdst[t] = i*xst + 8*part;
    }

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const int c0 = lane/4;
    const int r0 = 2*(lane % 4);

    f8_t acc[nm];
#pragma unroll
    for (int m = 0; m < nm; ++m) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            acc[m][l] = 0.0f;
        }
    }

    auto fetch_x = [&](const int k0, uint4 * xv) {
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            xv[t] = make_uint4(0, 0, 0, 0);
            if (xval[t]) {
                xv[t] = *(const uint4 *) (xsrc[t] + k0);
            }
        }
    };

    const int kz      = blockIdx.z;
    const int kblocks = gridDim.z;
    const int per     = (p.k/ks + kblocks - 1)/kblocks;
    const int s0      = kz*per;
    const int nsteps  = min(p.k/ks, s0 + per);

    uint4 xv[cpt];
    typename wtype::raw wr[2];
    if (s0 < nsteps) {
        fetch_x(ks*s0, xv);
        wr[0] = wl.fetch(2*s0 + 0, wave);
        wr[1] = wl.fetch(2*s0 + 1, wave);
    }

    for (int step = s0; step < nsteps; ++step) {
        half * xb = xs + (step % 2)*mt*xst;
        uint4 cxv[cpt];
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            cxv[t] = xv[t];
        }
        const typename wtype::raw cw0 = wr[0];
        const typename wtype::raw cw1 = wr[1];
        if (step + 1 < nsteps) {
            fetch_x(ks*(step + 1), xv);
            wr[0] = wl.fetch(2*(step + 1) + 0, wave);
            wr[1] = wl.fetch(2*(step + 1) + 1, wave);
        }
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {
            uint32_t wf[4];
            uint32_t wlo[4];
            wl.decode(kt == 0 ? cw0 : cw1, wf, wlo);
            *(uint32_t *) (ws + c0*wst       + 16*kt + r0)     = wf[0];
            *(uint32_t *) (ws + c0*wst       + 16*kt + r0 + 8) = wf[1];
            *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0)     = wf[2];
            *(uint32_t *) (ws + (c0 + 8)*wst + 16*kt + r0 + 8) = wf[3];
        }
#pragma unroll
        for (int t = 0; t < cpt; ++t) {
            if (tid + 256*t < nchunk) {
                *(uint4 *) (xb + xdst[t]) = cxv[t];
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const h8_t b = *(const h8_t *) (ws + (lane % 16)*wst + 16*kk + 8*(lane/16));
#pragma unroll
            for (int m = 0; m < nm; ++m) {
                const half * xrow = xb + (16*m + lane % 16)*xst + 16*kk + 8*(lane/16);
                const h8_t a = *(const h8_t *) xrow;
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[m]);
            }
        }
    }

    const char * rout = mmh_rot_row(p, p.rot_out, p.rout_cache, p.rout_nb1, e) + (int64_t) g*MMH_NG*p.rout_nb0;
    // output Hadamard, rot_out and the row scale of row i of the item
    auto finish = [&](const int i, float v[4]) {
        const int r = rows_sorted[start + i];
        mmh_fwht(v, lane, p.had, p.hscale);
        float * out = mmh_dst_row(p, r) + (int64_t) g*MMH_NG;
        const float rs = rscale[r];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = 32*j + lane;
            out[c] = v[j]*rs*mmh_rot(rout, p.rout_f16, c*(p.rout_nb0/(p.rout_f16 ? 2 : 4)));
        }
    };

    // the accumulators go through LDS ec rows at a time: to the epilogue, or with a k split to the partial sums
#pragma unroll
    for (int e0 = 0; e0 < mt; e0 += ec) {
        __syncthreads();
#pragma unroll
        for (int m = e0/16; m < (e0 + ec)/16; ++m) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                ys[(16*m - e0 + 8*(lane/16) + l)*yst + 16*wave + lane % 16] = acc[m][l];
            }
        }
        __syncthreads();
        for (int i = e0 + wave; i < min(cnt, e0 + ec); i += 8) {
            if (kblocks > 1) {
                float * dp = part + (((int64_t) rows_sorted[start + i]*gridDim.x + g)*kblocks + kz)*MMH_NG;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    dp[32*j + lane] = ys[(i - e0)*yst + 32*j + lane];
                }
            } else {
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] = ys[(i - e0)*yst + 32*j + lane];
                }
                finish(i, v);
            }
        }
    }

    if (kblocks > 1) {
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            s_last = atomicAdd(&tickets[(int64_t) blockIdx.y*gridDim.x + g], 1) == kblocks - 1;
        }
        __syncthreads();
        if (!s_last) {
            return;
        }
        __threadfence();
        for (int i = wave; i < cnt; i += 8) {
            const int r = rows_sorted[start + i];
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float * dp = part + ((int64_t) r*gridDim.x + g)*kblocks*MMH_NG + 32*j + lane;
                float sum = 0.0f;
                for (int z = 0; z < kblocks; ++z) {
                    sum += dp[z*MMH_NG];
                }
                v[j] = sum;
            }
            finish(i, v);
        }
    }
#else
    GGML_UNUSED_VARS(p, w, xh, rscale, items, n_items, rows_sorted, part, tickets);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

bool ggml_cuda_mul_mat_had_supported(const int device, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * b = op->src[1];
    const int had = ggml_get_op_params_i32(op, 0);

    if (ggml_cuda_info().devices[device].warp_size != 32) {
        return false;
    }
    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16 && !ggml_is_exl3(w->type)) {
        return false;
    }
    if (b->type != GGML_TYPE_F32 || b->nb[0] != sizeof(float) || op->type != GGML_TYPE_F32) {
        return false;
    }
    if ((had != 32 && had != 64 && had != 128) || w->ne[0] % MMH_NG != 0 || w->ne[1] % MMH_NG != 0) {
        return false;
    }
    if (w->nb[1] % 8 != 0 || w->nb[2] % 8 != 0 || w->ne[2] > MMH_MAX_EXPERTS || ggml_nrows(op) > INT_MAX/2) {
        return false;
    }
    return true;
}

// src1 precision of the op: true for the F16 mode (GGML_PREC_F16 or lower), false for GGML_PREC_F32
static bool mmh_input_f16(const ggml_tensor * dst) {
    static const ggml_prec env = [] {
        const char * v = getenv("GGML_CUDA_MUL_MAT_HAD_PREC");
        if (v == nullptr) {
            return GGML_PREC_UNDEFINED;
        }
        const ggml_prec prec = strcmp(v, "f16") == 0 ? GGML_PREC_F16 : strcmp(v, "f32") == 0 ? GGML_PREC_F32 : GGML_PREC_UNDEFINED;
        if (prec == GGML_PREC_UNDEFINED) {
            GGML_LOG_WARN("%s: unknown GGML_CUDA_MUL_MAT_HAD_PREC = '%s', available: f32, f16\n", __func__, v);
        } else {
            GGML_LOG_INFO("%s: GGML_CUDA_MUL_MAT_HAD_PREC = %s\n", __func__, v);
        }
        return prec;
    }();
    ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    if (prec == GGML_PREC_UNDEFINED) {
        prec = env != GGML_PREC_UNDEFINED ? env : MMH_PREC_DEFAULT;
    }
    return prec >= GGML_PREC_F16;
}

// with WMMA, whether a dense op of `tokens` tokens runs the GEMV instead of the GEMM: the GEMV decodes every weight
// once for all its rows, but its cost per weight grows with the rows, while the GEMM tile costs about the same up to
// 16 rows; with few 128-row groups the GEMM also has few blocks (measured crossovers)
static bool mmh_dense_gemv(const int64_t tokens, const int64_t groups, const int64_t weights) {
    if (tokens <= 1) {
        return true;
    }
    if (tokens <= 2) {
        return groups <= 64;
    }
    if (tokens <= 3) {
        return groups <= 48;
    }
    if (tokens <= 4) {
        return groups <= 32;
    }
    return groups <= 16 || weights <= (1 << 21);
}

// k-split tickets of the fused GEMV, one buffer per device and stream: all zero between ops, as the last block of each
// group resets its ticket. A new buffer is cleared on a stream of its own and waited for, so that it can also be made
// while the op's stream is being captured into a CUDA graph (nullptr if n is too large).
static int * mmh_fused_tickets(const int device, cudaStream_t stream, const int64_t n) {
    constexpr int64_t cap = 16384;
    if (n > cap) {
        return nullptr;
    }
    static std::mutex mutex;
    static std::map<std::pair<int, cudaStream_t>, int *> buffers;
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = buffers.find({device, stream});
    if (it != buffers.end()) {
        return it->second;
    }
    int * t = nullptr;
    cudaStream_t clear;
    CUDA_CHECK(cudaMalloc((void **) &t, cap*sizeof(int)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&clear, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemsetAsync(t, 0, cap*sizeof(int), clear));
    CUDA_CHECK(cudaStreamSynchronize(clear));
    CUDA_CHECK(cudaStreamDestroy(clear));
    buffers[{device, stream}] = t;
    return t;
}

// the inline GEMV; coal only with one row per pass (expert ops), where it measured faster
template <ggml_type type, int rows, bool h16>
static void mmh_launch_gemv(const bool fused, const bool coal, const dim3 grid, const int ksplit, const size_t lds,
        cudaStream_t stream, const mmh_params & p, const char * w, const void * xh, const float * xsum, const float * rscale,
        float * part, int * tickets, const int max_chunks) {
    if (coal && fused) {
        mmh_gemv<type, rows, true, true, h16><<<grid, 256*ksplit, lds, stream>>>(p, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else if (coal) {
        mmh_gemv<type, rows, false, true, h16><<<grid, 256*ksplit, 0, stream>>>(p, w, xh, xsum, rscale, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else if (fused) {
        mmh_gemv<type, rows, true, false, h16><<<grid, 256*ksplit, lds, stream>>>(p, w, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    } else {
        mmh_gemv<type, rows, false, false, h16><<<grid, 256*ksplit, 0, stream>>>(p, w, xh, xsum, rscale, nullptr, nullptr, nullptr, ksplit, part, tickets, max_chunks);
    }
}

// the inline GEMV with F32 or (h16) fp16 input
template <ggml_type type, bool h16>
static void mmh_launch_gemv_rows(const int rows_t, const bool fused, const bool coal, const dim3 grid, const int ksplit,
        const size_t lds, cudaStream_t stream, const mmh_params & p, const char * w, const void * xh, const float * xsum,
        const float * rscale, float * part, int * tickets, const int max_chunks) {
    switch (rows_t) {
        case 1:  mmh_launch_gemv<type, 1, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        case 2:  mmh_launch_gemv<type, 2, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        case 4:  mmh_launch_gemv<type, 4, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
        default: mmh_launch_gemv<type, 8, h16>(fused, coal, grid, ksplit, lds, stream, p, w, xh, xsum, rscale, part, tickets, max_chunks); break;
    }
}

template <ggml_type type, bool f16>
static void mmh_launch(ggml_backend_cuda_context & ctx, const mmh_params & p, const char * w, const int64_t n,
        const bool inline_rows) {
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t n_groups = n/MMH_NG;

    // the WMMA GEMM takes fp16 input: in the F32 mode the rows run the F32 GEMV
    const bool wmma = !inline_rows && f16 && type != GGML_TYPE_F32 && GGML_CUDA_CC_IS_RDNA4(cc);

    ggml_cuda_pool_alloc<half>  xh(ctx.pool());
    ggml_cuda_pool_alloc<float> rscale(ctx.pool());

    // rows per weight matrix, for the tile size
    const int64_t used    = p.ids ? std::min(p.n_rows, p.n_exp) : p.n_exp;
    const int64_t per_exp = (p.n_rows + used - 1)/used;

    if (inline_rows) {
        const int64_t n_items = p.ids ? p.n_rows : p.n_exp;
        // with ids, an expert of several tokens is decoded once for up to MMH_GEMV_IDS_ROWS of its rows per pass (an
        // expert of one token takes the one-row pass of the same kernel)
        const int     rows    = p.ids && p.ne2 > 1 ? (p.ne2 <= 2 ? MMH_GEMV_IDS_ROWS_2TOK : MMH_GEMV_IDS_ROWS) :
                                (int) std::min<int64_t>(MMH_GEMV_TOKENS, per_exp);

        // k split S = ksplit*kblocks: the power of 2 nearest (in log scale) to the split that gives about
        // MMH_GEMV_WAVES waves, with at least MMH_GEMV_MIN_KSLICES k slices per wave; S >= 2 splits k over 2 waves of
        // a block and S/2 blocks (small blocks fill the CUs better than blocks of 4 k splits)
        const int64_t kslices = p.k/16;
        const double  want    = (double) MMH_GEMV_WAVES/(8.0*(double) (n_groups*used));
        int split = 1;
        while (split < MMH_GEMV_MAX_SPLIT && want >= 1.41421356*split && kslices >= MMH_GEMV_MIN_KSLICES*2*split) {
            split *= 2;
        }
        // with ids and more than one token an expert can have several rows, which a block runs one after the other:
        // two waves per k range keep that chain short
        if (p.ids && p.ne2 > 1 && split < 2 && kslices >= 4*MMH_GEMV_MIN_KSLICES) {
            split = 2;
        }
        const int ksplit  = split >= 2 ? 2 : 1;
        const int kblocks = split/ksplit;

        ggml_cuda_pool_alloc<float> part(ctx.pool());
        ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
        const int max_chunks = (int) ((std::min<int64_t>(p.n_rows, MMH_GEMV_LIST) + rows - 1)/rows);
        const int n_tickets  = kblocks > 1 ? (int) (n_items*n_groups*max_chunks) : 0;
        const int rows_t     = rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : 8;

        // fused: the k slices one block stages (at most one 128-value block more at each end of its range), LDS of
        // the staged rows beside the static LDS of the GEMV
        const int64_t per    = (kslices + kblocks*ksplit - 1)/(kblocks*ksplit);
        const int64_t nks    = std::min<int64_t>(kslices, (per*ksplit + 7)/8*8 + 8);
        const size_t  lds    = (size_t) rows_t*nks*16*sizeof(float);
        const size_t  lds_static = (size_t) (2*MMH_GEMV_LIST + 33)*sizeof(int) + (size_t) 4*rows_t*MMH_NG*sizeof(float);
        bool fused = lds + lds_static <= MMH_FUSED_LDS && n_groups*used*kblocks <= MMH_FUSED_MAX_BLOCKS;
        // the fp16 input: 2 bytes per value, and for mul1 the F32 sums of each 16 values (the fused choice is the same, but see below).
        // Not for a dense op of one row, where it measured slower for the larger types
        const bool    h16    = f16 && mmh_is_exl3(type) && (p.ids || per_exp > 1) && kslices*sizeof(float) <= MMH_FUSED_LDS;
        const bool    sums   = h16 && mmh_exl3_codebook(type) == 0;
        const size_t  lds16  = (size_t) rows_t*nks*(16*sizeof(half) + (sums ? sizeof(float) : 0));
        // routed weights over the host link: the fp16 input also decides the fused choice (2 rows of the gate/up of
        // DeepSeek V4.1 stage 28.8 KB instead of 49.3 KB and skip the prologue: 0.95 ms per call of 40 layers). In VRAM
        // it measured 0.3 ms slower, so the F32 size decides there
        if (h16 && p.w_remote && !fused) {
            fused = lds16 + lds_static <= MMH_FUSED_LDS && n_groups*used*kblocks <= MMH_FUSED_MAX_BLOCKS;
        }
        int * fused_tickets = nullptr;
        if (fused && kblocks > 1) {
            fused_tickets = mmh_fused_tickets(ctx.device, stream, n_tickets);
            fused = fused_tickets != nullptr;
        }

        if (kblocks > 1) {
            part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
        }
        ggml_cuda_pool_alloc<float> xsum(ctx.pool());
        if (!fused) {
            xh.alloc((h16 ? 1 : 2)*p.n_rows*p.k);
            rscale.alloc(p.n_rows);
            if (sums) {
                xsum.alloc(p.n_rows*split);
            }
            if (kblocks > 1) {
                tickets.alloc(n_tickets);
            }
            if (h16) {
                mmh_prologue<true, true><<<p.n_rows, 256, sums ? kslices*sizeof(float) : 0, stream>>>(p, xh.get(), rscale.get(),
                    tickets.ptr, n_tickets, xsum.ptr, split, (int) per);
            } else {
                mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets, nullptr, 0, 0);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        int * tk = fused ? fused_tickets : tickets.ptr;

        const bool coal = p.ids != nullptr;
        const dim3 grid(n_groups, n_items, kblocks);
        if constexpr (f16 && mmh_is_exl3(type)) {
            if (h16) {
                mmh_launch_gemv_rows<type, true>(rows_t, fused, coal, grid, ksplit, lds16, stream, p, w, xh.ptr, xsum.ptr, rscale.ptr,
                    part.ptr, tk, max_chunks);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        mmh_launch_gemv_rows<type, false>(rows_t, fused, coal, grid, ksplit, lds, stream, p, w, xh.ptr, nullptr, rscale.ptr, part.ptr,
            tk, max_chunks);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    xh.alloc((wmma ? 1 : 2)*p.n_rows*p.k);
    rscale.alloc(p.n_rows);

    // Popular experts get several times the mean number of rows, and an expert with more rows than the tile width reads
    // its weights once per tile (over the link when they sit in host memory). Like MMQ, size the tile against a multiple
    // of the mean for the experts read from host memory. Experts read from VRAM (weights in VRAM, or an expert cache
    // slot) are not re-read over a link and the wider tile is slower there: they keep the tile of the mean, and with the
    // expert cache the two kinds of experts run as two GEMM launches. The multiple is 3.
    const int64_t mt_scale = GGML_CUDA_CC_IS_RDNA4(cc) && p.w_remote ? 3 : 1;
    auto tile = [&](const int64_t scale) {
        const int64_t rows_opt = std::min<int64_t>(p.n_rows, per_exp*scale);
        return !wmma ? 8 : rows_opt <= 16 ? 16 : rows_opt <= 32 ? 32 : rows_opt <= 64 ? 64 : 128;
    };
    const int  mt_r  = tile(mt_scale);
    // without cache slots every routed expert is read from host memory: one list of the wide tile
    const bool split = p.ids && p.w_slots != nullptr && tile(1) != mt_r;
    const int  mt    = split ? tile(1) : mt_r;
    const int64_t max_items   = used + p.n_rows/mt + 1;
    const int64_t max_items_r = split ? used + p.n_rows/mt_r + 1 : 0;

    GGML_ASSERT(max_items <= 65535);

    // WMMA: blocks also split k while there are few blocks and each keeps >= 8 steps
    const int64_t items_est = p.ids ? used + p.n_rows/mt : p.n_exp*((per_exp + mt - 1)/mt);
    int kblocks = 1;
    while (wmma && kblocks < 8 && n_groups*items_est*kblocks < 256 && p.k/32 >= 8*kblocks*2) {
        kblocks *= 2;
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
    // the two launches count their k splits in tickets of their own
    const int n_tickets = kblocks > 1 ? (int) ((max_items + max_items_r)*n_groups) : 0;
    if (kblocks > 1) {
        part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
        tickets.alloc(n_tickets);
    }

    if (wmma) {
        mmh_prologue<false, true><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets, nullptr, 0, 0);
    } else {
        mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), nullptr, 0, nullptr, 0, 0);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int> route(ctx.pool(), 3*p.n_rows + 3*(max_items + max_items_r) + 2);
    int * eid         = route.get();
    int * rank        = eid + p.n_rows;
    int * rows_sorted = rank + p.n_rows;
    int * items       = rows_sorted + p.n_rows;
    int * n_items     = items + 3*max_items;
    int * items_r     = split ? n_items + 1 : nullptr;
    int * n_items_r   = split ? items_r + 3*max_items_r : nullptr;

    mmh_route<<<1, 1024, 0, stream>>>(p, mt, mt_r, eid, rank, rows_sorted, items, n_items, items_r, n_items_r);
    CUDA_CHECK(cudaGetLastError());

    if (wmma) {
        if constexpr (type != GGML_TYPE_F32) {
            auto gemm = [&](const int m, const int64_t n_max, const int * it, const int * n_it, int * tk) {
                const dim3 grid(n_groups, n_max, kblocks);
                if (m == 16) {
                    mmh_gemm<type, 16><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), it, n_it, rows_sorted, part.ptr, tk);
                } else if (m == 32) {
                    mmh_gemm<type, 32><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), it, n_it, rows_sorted, part.ptr, tk);
                } else if (m == 64) {
                    mmh_gemm<type, 64><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), it, n_it, rows_sorted, part.ptr, tk);
                } else {
                    mmh_gemm<type, 128><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), it, n_it, rows_sorted, part.ptr, tk);
                }
                CUDA_CHECK(cudaGetLastError());
            };
            gemm(mt, max_items, items, n_items, tickets.ptr);
            if (split) {
                gemm(mt_r, max_items_r, items_r, n_items_r, tickets.ptr ? tickets.ptr + max_items*n_groups : nullptr);
            }
        }
    } else {
        // the waves split k also for precision: a lane then sums at most a quarter of k
        const dim3 grid(n_groups, max_items, kblocks);
        const int ksplit = p.k/16 >= 64 ? 4 : p.k/16 >= 32 ? 2 : 1;
        mmh_gemv<type, 8, false, false><<<grid, 256*ksplit, 0, stream>>>(p, w, xh.get(), nullptr, rscale.get(), items, n_items, rows_sorted, ksplit, nullptr, nullptr, 1);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type type>
static void mmh_launch_prec(ggml_backend_cuda_context & ctx, const mmh_params & p, const char * w, const int64_t n,
        const bool inline_rows, const bool f16) {
    if (f16) {
        mmh_launch<type, true>(ctx, p, w, n, inline_rows);
    } else {
        mmh_launch<type, false>(ctx, p, w, n, inline_rows);
    }
}

// a weight or scale in a mapped host buffer is read through its device alias (host-direct)
static const char * mmh_data(const ggml_tensor * t) {
#if defined(GGML_USE_HIP)
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        return (const char *) ggml_hip_mapped_host_device_alias(t);
    }
#endif // defined(GGML_USE_HIP)
    return (const char *) t->data;
}

#if defined(GGML_USE_HIP)
// the rotation arena of the expert cache for the rows `rot` (base, nb[1]) of the op, when it mirrors that tensor
static const char * mmh_rot_cache(const ggml_cuda_expert_lookup & cached, const int which, const char * rot, const int64_t nb1) {
    const ggml_tensor * src = cached.aux_src[which];
    return src != nullptr && mmh_data(src) == rot && (int64_t) src->nb[1] == nb1 ? (const char *) cached.aux[which] : nullptr;
}
#endif // defined(GGML_USE_HIP)

// the weight base for the kernels; with ids, a routed expert weight of the expert cache also fills the cache fields
// of p (MUL_MAT_ID rule: a weight in a mapped host buffer or in the exclusive buffer, whose tensor has logical
// addresses only and is read through the lookup alone)
// With defer_wait the staged SSD tier's wait for this kind is not enqueued here; the caller enqueues it with
// ggml_cuda_expert_before_read before the first kernel that reads the kind's addresses (the table pointers are
// final, only their contents fill in as the kind is published).
static const char * mmh_weight_data(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * ids,
        mmh_params & p, bool defer_wait = false) {
    p.w_cache          = nullptr;
    p.w_slots          = nullptr;
    p.w_host_slots     = nullptr;
    p.w_host_addresses = nullptr;
    p.w_remote         = false;
    p.rin_cache        = nullptr;
    p.rout_cache       = nullptr;
#if defined(GGML_USE_HIP)
    const bool exclusive = w->buffer && ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_get_type(w->buffer));
    GGML_ASSERT(!exclusive || ids);
    const char * data = mmh_data(w);
    if (ids && (exclusive || (w->buffer && ggml_backend_buffer_is_host(w->buffer)))) {
        const ggml_cuda_expert_lookup cached = ggml_cuda_expert_lookup_tensor(w);
        if (!defer_wait) {
            ggml_cuda_expert_before_read(ctx, w);
        }
        p.w_remote         = true;
        p.w_cache          = cached.data;
        p.w_slots          = cached.slots;
        p.w_host_slots     = cached.host_slots;
        p.w_host_addresses = cached.host_addresses;
        if (cached.host_data != nullptr) {
            data = (const char *) cached.host_data;
        }
        // the rotations move with the expert (D8 stage 2) when the op reads the tensors the cache mirrors
        p.rin_cache  = mmh_rot_cache(cached, 0, p.rot_in,  p.rin_nb1);
        p.rout_cache = mmh_rot_cache(cached, 1, p.rot_out, p.rout_nb1);
        GGML_ASSERT(!exclusive || p.w_host_slots != nullptr);
    }
    return data;
#else
    GGML_UNUSED_VARS(ctx, ids, defer_wait);
    return mmh_data(w);
#endif // defined(GGML_USE_HIP)
}

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * w       = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * rot_in  = dst->src[3];
    const ggml_tensor * rot_out = dst->src[4];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));
    GGML_ASSERT(!ids || ids->type == GGML_TYPE_I32);

    const int had = ggml_get_op_params_i32(dst, 0);

#if defined(GGML_USE_HIP)
    // expert cache: the SSD tier may reuse the layer's ring slots once its last kind is read (as MUL_MAT_ID)
    struct done_guard {
        ggml_backend_cuda_context & ctx;
        const ggml_tensor * w;
        ~done_guard() { ggml_cuda_expert_layer_done(ctx, w); }
    } done{ctx, w};
#endif // defined(GGML_USE_HIP)

    mmh_params p;
    p.k        = w->ne[0];
    p.n_exp    = w->ne[2];
    p.nb01     = w->nb[1];
    p.nb02     = w->nb[2];
    p.n_rows   = ggml_nrows(dst);
    p.ne1      = (int) dst->ne[1];
    p.ne2      = (int) dst->ne[2];
    p.nb1      = dst->nb[1];
    p.nb2      = dst->nb[2];
    p.nb3      = dst->nb[3];
    p.ne11     = (int) x->ne[1];
    p.nb11     = x->nb[1];
    p.nb12     = x->nb[2];
    p.nb13     = x->nb[3];
    p.bcast    = ids ? 1 : (int) (x->ne[2]/w->ne[2]);
    p.ids      = ids ? (const int32_t *) ids->data : nullptr;
    p.ids_nb0  = ids ? ids->nb[0] : 0;
    p.ids_nb1  = ids ? ids->nb[1] : 0;
    p.x        = (const float *) x->data;
    p.rot_in   = mmh_data(rot_in);
    p.rin_nb1  = rot_in->nb[1];
    p.rot_out  = mmh_data(rot_out);
    p.rout_nb0 = rot_out->nb[0];
    p.rout_nb1 = rot_out->nb[1];
    p.dst      = (float *) dst->data;
    p.rin_f16  = rot_in->type == GGML_TYPE_F16;
    p.rout_f16 = rot_out->type == GGML_TYPE_F16;
    p.had      = had;
    p.hscale   = 1.0f/sqrtf((float) had);

    if (p.n_rows == 0) {
        return;
    }

    const char * wd = mmh_weight_data(ctx, w, ids, p);
    const int64_t n = w->ne[1];

    // without a routing pass: few tokens (ids) or few rows per weight matrix; with WMMA a dense matrix of more than
    // one row goes to the GEMM, which stages the rows once per block, unless the GEMV is faster (mmh_dense_gemv)
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = w->type != GGML_TYPE_F32 && GGML_CUDA_CC_IS_RDNA4(cc);
    const int64_t tokens = ids ? dst->ne[2] : (p.n_rows + p.n_exp - 1)/p.n_exp;
    const bool inline_rows = p.n_rows <= MMH_GEMV_LIST && tokens <= MMH_GEMV_TOKENS &&
        (ids || !wmma || mmh_dense_gemv(tokens, n/MMH_NG, p.k*n));

    // F32 weights run only the GEMV, which has no F16 mode
    const bool f16 = w->type != GGML_TYPE_F32 && mmh_input_f16(dst);

    switch (w->type) {
        case GGML_TYPE_F32:      mmh_launch<GGML_TYPE_F32, false>(ctx, p, wd, n, inline_rows); break;
        case GGML_TYPE_F16:      mmh_launch_prec<GGML_TYPE_F16>     (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M1:  mmh_launch_prec<GGML_TYPE_EXL3_M1> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M2:  mmh_launch_prec<GGML_TYPE_EXL3_M2> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M3:  mmh_launch_prec<GGML_TYPE_EXL3_M3> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M4:  mmh_launch_prec<GGML_TYPE_EXL3_M4> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M5:  mmh_launch_prec<GGML_TYPE_EXL3_M5> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M6:  mmh_launch_prec<GGML_TYPE_EXL3_M6> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M7:  mmh_launch_prec<GGML_TYPE_EXL3_M7> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M8:  mmh_launch_prec<GGML_TYPE_EXL3_M8> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M1H: mmh_launch_prec<GGML_TYPE_EXL3_M1H>(ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M2H: mmh_launch_prec<GGML_TYPE_EXL3_M2H>(ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_M3H: mmh_launch_prec<GGML_TYPE_EXL3_M3H>(ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G1:  mmh_launch_prec<GGML_TYPE_EXL3_G1> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G2:  mmh_launch_prec<GGML_TYPE_EXL3_G2> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G3:  mmh_launch_prec<GGML_TYPE_EXL3_G3> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G4:  mmh_launch_prec<GGML_TYPE_EXL3_G4> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G5:  mmh_launch_prec<GGML_TYPE_EXL3_G5> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G6:  mmh_launch_prec<GGML_TYPE_EXL3_G6> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G7:  mmh_launch_prec<GGML_TYPE_EXL3_G7> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_G8:  mmh_launch_prec<GGML_TYPE_EXL3_G8> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T1:  mmh_launch_prec<GGML_TYPE_EXL3_T1> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T2:  mmh_launch_prec<GGML_TYPE_EXL3_T2> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T3:  mmh_launch_prec<GGML_TYPE_EXL3_T3> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T4:  mmh_launch_prec<GGML_TYPE_EXL3_T4> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T5:  mmh_launch_prec<GGML_TYPE_EXL3_T5> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T6:  mmh_launch_prec<GGML_TYPE_EXL3_T6> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T7:  mmh_launch_prec<GGML_TYPE_EXL3_T7> (ctx, p, wd, n, inline_rows, f16); break;
        case GGML_TYPE_EXL3_T8:  mmh_launch_prec<GGML_TYPE_EXL3_T8> (ctx, p, wd, n, inline_rows, f16); break;
        default:
            GGML_ABORT("unsupported MUL_MAT_HAD weight type %s", ggml_type_name(w->type));
    }
}

// ---- GGML_OP_MOE_FFN_HAD: the routed experts of one layer, expert by expert (ggml_moe_ffn_had) ----------------------
//
// The rows (token t, slot k) of the op are sorted by expert and cut into chunks of at most R rows at expert boundaries;
// an expert of more than R rows takes chunks of its own. Per chunk: up and gate (a prologue and the WMMA GEMM of
// MUL_MAT_HAD, with the same expert cache lookup), the GLU in place, down, then every token adds its weighted down rows
// of the chunk in slot order. The intermediates scale with R instead of the token count. A token's rows are summed in
// chunk order (the expert order) and inside a chunk in slot order; which rows of a split expert land in which of its
// chunks does not change that order, so the result does not depend on the order the routing pass ranks rows in.
//
// Rows per GEMM item (mt): an expert whose weights are in VRAM (the VRAM arena, a VRAM weight, a staged copy) takes the
// tile of the mean rows per expert, as MUL_MAT_HAD; an expert read from host memory takes three times that (fewer reads
// of the weight over the link, the rule of MMQ and of MUL_MAT_HAD for remote weights). Each chunk runs one GEMM per
// kind for each of the two tiles.
//
// Staging (RANMA_EXPERT_MAJOR_STAGE_MIB > 0): the experts of a chunk with at least `stage_min` rows that are not in VRAM
// are copied into one of two VRAM buffers (chunk c uses buffer c % 2) by a resident copy kernel on a side stream while
// the previous chunk computes (docs/hip-copy-api-overheads.ko.md 2.13: the resident side-stream copy kernel with device
// flags, launched once and flushed). Signals are device flags: ready[c] (copier -> compute) and done = the number of
// chunks whose GEMMs finished (compute -> copier, which reuses a buffer only for chunk c + 2). Before the GEMMs of chunk c
// a one-block wait kernel points the entries of its staged experts in the op's address tables at the buffer; after the
// down GEMM a release kernel restores them and counts the chunk done.
// RANMA_EXPERT_MAJOR_STAGE_ENGINE=sdma (the default) copies with the copy engine instead: after the routing pass the host
// reads the plan (one synchronize per op), issues one hipMemcpyAsync per expert and kind on the side stream in chunk
// order, and events order the copies of chunk c after the GEMMs of chunk c - 2 and the GEMMs of chunk c after its copies
// (the copy kernel slowed the GEMMs it ran beside, as in docs/hip-copy-api-overheads.ko.md 2.13 4b). The tables take the
// place of the SSD tier's address
// table (ggml_cuda_expert_cache_select), so the GEMM kernels are unchanged. Every wait is bounded only to detect a hang:
// a wait over MMH_MOE_STAGE_TIMEOUT (or a staged source that is not 16-byte aligned) records an error in mapped host
// memory and stops the copier and the waits; a host function after the op reports it and aborts. The legitimate waits are
// bounded by one chunk of compute and one chunk of copies, far below the limit.

#define MMH_MOE_ROWS 4096 // default rows per chunk
#define MMH_MOE_MAX_USED 64 // most experts per token
#define MMH_MOE_STAGE_MAX_CHUNKS 1024
#define MMH_MOE_STAGE_MAX_SLOTS 64
#define MMH_MOE_STAGE_TIMEOUT 500000000ull // 5 s of the 100 MHz wall clock

struct mmh_moe_route {
    int * rows_sorted; // N: row g = k + n_used*t, sorted by expert
    int * pos;         // N: index of row g in rows_sorted
    int * eid;         // N: expert of row g
    int * offs;        // n_exp: first index of the expert in rows_sorted
    int * cnt;         // n_exp: rows of the expert
    int * chunks;      // 2*c_max: first index, row count
    int * n_chunks;
    int * stage_list;  // c_max*stage_cap: the staged experts of each chunk
    int * stage_n;     // c_max
};

struct mmh_moe_stage_flags {
    int ready[MMH_MOE_STAGE_MAX_CHUNKS];
    int arrive[MMH_MOE_STAGE_MAX_CHUNKS];
    int done;
    int abort;
    int staged;
};

// the first error of an op (mapped host memory, written by the device with system scope)
struct mmh_moe_stage_error {
    int      code;   // 1 copier waited for a free buffer, 2 compute waited for a copy, 3 unaligned source
    int      layer;
    int      chunk;
    int      value;  // done (1), ready (2), expert (3)
    uint64_t ticks;  // wall clock ticks (100 MHz) waited
};

// the copies of an op for the copy engine (mapped host memory, written by the device after the routing pass)
struct mmh_moe_stage_plan {
    int      n_chunks;
    int      stage_n[MMH_MOE_STAGE_MAX_CHUNKS];
    uint64_t src[MMH_MOE_STAGE_MAX_CHUNKS*MMH_MOE_STAGE_MAX_SLOTS*3];
};

static __device__ void mmh_moe_stage_fail(mmh_moe_stage_flags * f, mmh_moe_stage_error * err, const int code, const int layer,
        const int chunk, const int value, const uint64_t ticks) {
    if (atomicCAS(&f->abort, 0, 1) == 0) {
        err->layer = layer;
        err->chunk = chunk;
        err->value = value;
        err->ticks = ticks;
        __threadfence_system();
        __hip_atomic_store(&err->code, code, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

// where the copier reads an expert: the cache's selection of the op (VRAM slots never reach it)
struct mmh_moe_stage_src {
    const char     * data[3];
    const void     * arena[3];
    const int32_t  * slots[3];
    const int32_t  * host_slots[3];
    const uint64_t * host_addresses[3];
    int64_t          nb02[3];
    int64_t          koff[4]; // byte offset of each kind in a staging slot, koff[3] = slot bytes
};

static __device__ __forceinline__ int mmh_moe_expert(const int32_t * ids, const int64_t nb0, const int64_t nb1,
        const int n_used, const int g, const int n_exp) {
    const int e = *(const int32_t *) ((const char *) ids + (g % n_used)*nb0 + (g / n_used)*nb1);
    return e < 0 ? 0 : (e >= n_exp ? n_exp - 1 : e);
}

// one block: rows sorted by expert, the chunk table and the staged experts of each chunk. The experts are ordered by
// class, then by id: those read from VRAM first (they do not use the link, so the copies of the staged experts fill it
// meanwhile), then the others by id, so that the in-place reads of host experts are spread between the staged ones
// (RANMA_EXPERT_MAJOR_ORDER=last: the staged experts first, the in-place ones last).
static __global__ void __launch_bounds__(1024) mmh_moe_route_kernel(const int32_t * __restrict__ ids, const int64_t ids_nb0,
        const int64_t ids_nb1, const int n_used, const int n_rows, const int n_exp, const int chunk_rows, const int c_max,
        const int32_t * __restrict__ slots, const bool all_local, const int stage_cap, const int stage_min, const bool host_last,
        const mmh_moe_route rt) {
    __shared__ int  s_count[MMH_MAX_EXPERTS];
    __shared__ int  s_offs[MMH_MAX_EXPERTS];
    __shared__ int  s_scan[1024];
    __shared__ char s_class[MMH_MAX_EXPERTS];

    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    for (int e = tid; e < n_exp; e += nth) {
        s_count[e] = 0;
    }
    __syncthreads();
    for (int g = tid; g < n_rows; g += nth) {
        const int e = mmh_moe_expert(ids, ids_nb0, ids_nb1, n_used, g, n_exp);
        rt.eid[g] = e;
        rt.pos[g] = atomicAdd(&s_count[e], 1);
    }
    __syncthreads();

    auto stageable = [&](const int e, const int r) {
        return stage_cap > 0 && r >= stage_min && !all_local && (slots == nullptr || slots[e] < 0);
    };
    const int per = (n_exp + nth - 1)/nth;
    const int e0  = min(n_exp, tid*per);
    const int e1  = min(n_exp, e0 + per);
    for (int e = e0; e < e1; ++e) {
        const int c = s_count[e];
        s_class[e] = all_local || (slots != nullptr && slots[e] >= 0) ? 0 : stageable(e, min(c, chunk_rows)) || !host_last ? 1 : 2;
    }
    int base = 0;
    for (int cls = 0; cls < 3; ++cls) {
        int sum = 0;
        for (int e = e0; e < e1; ++e) {
            sum += s_class[e] == cls ? s_count[e] : 0;
        }
        __syncthreads();
        s_scan[tid] = sum;
        __syncthreads();
        for (int h = 1; h < nth; h *= 2) {
            const int a = tid >= h ? s_scan[tid - h] : 0;
            __syncthreads();
            s_scan[tid] += a;
            __syncthreads();
        }
        int row = base + s_scan[tid] - sum;
        for (int e = e0; e < e1; ++e) {
            if (s_class[e] == cls) {
                s_offs[e]  = row;
                rt.offs[e] = row;
                rt.cnt[e]  = s_count[e];
                row += s_count[e];
            }
        }
        base += s_scan[nth - 1];
    }
    __syncthreads();
    for (int g = tid; g < n_rows; g += nth) {
        const int p = s_offs[rt.eid[g]] + rt.pos[g];
        rt.pos[g] = p;
        rt.rows_sorted[p] = g;
    }

    // chunks: whole experts packed in order up to chunk_rows rows and stage_cap staged experts; an expert of more rows
    // takes chunks of its own
    if (tid == 0) {
        int nc    = 0;
        int cur_b = 0;
        int cur_n = 0;
        int cur_s = 0;
        auto emit = [&](const int b, const int n, const int s) {
            if (n > 0 && nc < c_max) {
                rt.chunks[2*nc + 0] = b;
                rt.chunks[2*nc + 1] = n;
                rt.stage_n[nc] = s;
                ++nc;
            }
        };
        for (int i = 0; i < 3*n_exp; ++i) {
            const int e = i % n_exp;
            const int c = s_count[e];
            if (c == 0 || s_class[e] != i / n_exp) {
                continue;
            }
            if (c > chunk_rows) {
                emit(cur_b, cur_n, cur_s);
                cur_n = 0;
                cur_s = 0;
                for (int i = 0; i < c; i += chunk_rows) {
                    const int r = min(chunk_rows, c - i);
                    const int s = stageable(e, r) && nc < c_max;
                    if (s) {
                        rt.stage_list[nc*stage_cap] = e;
                    }
                    emit(s_offs[e] + i, r, s);
                }
                continue;
            }
            const bool st = stageable(e, c);
            if (cur_n + c > chunk_rows || (st && cur_s == stage_cap)) {
                emit(cur_b, cur_n, cur_s);
                cur_n = 0;
                cur_s = 0;
            }
            if (cur_n == 0) {
                cur_b = s_offs[e];
            }
            if (st && nc < c_max) {
                rt.stage_list[nc*stage_cap + cur_s++] = e;
            }
            cur_n += c;
        }
        emit(cur_b, cur_n, cur_s);
        *rt.n_chunks = nc;
    }
}

// one block: the GEMM items (expert, first local row, rows <= mt) of chunk c, in two lists: the experts read from VRAM
// (tile mt_v) and the others (tile mt_r)
static __global__ void __launch_bounds__(1024) mmh_moe_items_kernel(const mmh_moe_route rt, const int c, const int n_exp,
        const int32_t * __restrict__ slots, const bool all_local, const int stage_cap, const int mt_v, const int mt_r,
        int * __restrict__ items_v, int * __restrict__ n_items_v, int * __restrict__ items_r, int * __restrict__ n_items_r) {
    __shared__ int s_scan[2][1024];

    const int tid  = threadIdx.x;
    const int nth  = blockDim.x;
    const bool live = c < *rt.n_chunks;
    const int b    = live ? rt.chunks[2*c + 0] : 0;
    const int nr   = live ? rt.chunks[2*c + 1] : 0;
    const int ns   = live && stage_cap > 0 ? rt.stage_n[c] : 0;

    auto local = [&](const int e) {
        if (all_local || (slots != nullptr && slots[e] >= 0)) {
            return true;
        }
        for (int j = 0; j < ns; ++j) {
            if (rt.stage_list[c*stage_cap + j] == e) {
                return true;
            }
        }
        return false;
    };

    const int per = (n_exp + nth - 1)/nth;
    const int e0  = min(n_exp, tid*per);
    const int e1  = min(n_exp, e0 + per);
    int sum[2] = {0, 0};
    for (int e = e0; e < e1; ++e) {
        const int lo = max(rt.offs[e], b);
        const int hi = min(rt.offs[e] + rt.cnt[e], b + nr);
        if (hi > lo) {
            const int l  = local(e) ? 0 : 1;
            const int mt = l == 0 ? mt_v : mt_r;
            sum[l] += (hi - lo + mt - 1)/mt;
        }
    }
    s_scan[0][tid] = sum[0];
    s_scan[1][tid] = sum[1];
    __syncthreads();
    for (int h = 1; h < nth; h *= 2) {
        const int a0 = tid >= h ? s_scan[0][tid - h] : 0;
        const int a1 = tid >= h ? s_scan[1][tid - h] : 0;
        __syncthreads();
        s_scan[0][tid] += a0;
        s_scan[1][tid] += a1;
        __syncthreads();
    }
    int item[2] = { s_scan[0][tid] - sum[0], s_scan[1][tid] - sum[1] };
    for (int e = e0; e < e1; ++e) {
        const int lo = max(rt.offs[e], b);
        const int hi = min(rt.offs[e] + rt.cnt[e], b + nr);
        if (hi <= lo) {
            continue;
        }
        const int l  = local(e) ? 0 : 1;
        const int mt = l == 0 ? mt_v : mt_r;
        int * items = l == 0 ? items_v : items_r;
        for (int i = lo; i < hi; i += mt) {
            items[3*item[l] + 0] = e;
            items[3*item[l] + 1] = i - b;
            items[3*item[l] + 2] = min(mt, hi - i);
            ++item[l];
        }
    }
    if (tid == nth - 1) {
        *n_items_v = s_scan[0][tid];
        *n_items_r = s_scan[1][tid];
    }
}

static __global__ void mmh_moe_iota(int * __restrict__ v, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = i;
    }
}

// the plain fp16 prologue of the GEMM (as mmh_prologue<false, true>) for the local rows of chunk c: the input row is
// the row of the token in x (local = false) or the local row of x (local = true)
static __global__ void __launch_bounds__(256) mmh_moe_prologue(const mmh_params p, const mmh_moe_route rt, const int c,
        const int32_t * __restrict__ ids, const int64_t ids_nb0, const int64_t ids_nb1, const int n_used,
        const float * __restrict__ x, const int64_t x_nb1, const bool local, half * __restrict__ xh,
        float * __restrict__ rscale) {
    __shared__ float s_red[8];

    const int r = blockIdx.x;
    if (c >= *rt.n_chunks || r >= rt.chunks[2*c + 1]) {
        return;
    }
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;

    const int g = rt.rows_sorted[rt.chunks[2*c + 0] + r];
    const int e = mmh_moe_expert(ids, ids_nb0, ids_nb1, n_used, g, (int) p.n_exp);
    const float * src = (const float *) ((const char *) x + (int64_t) (local ? r : g / n_used)*x_nb1);
    const char  * rin = mmh_rot_row(p, p.rot_in, p.rin_cache, p.rin_nb1, e);

    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < p.k; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(src[i]*mmh_rot(rin, p.rin_f16, i)));
    }
    amax = mmh_block_max(amax, s_red);
    float s = 1.0f;
    const float bound = amax*sqrtf((float) p.had);
    if (bound > 0.0f && bound <= FLT_MAX) {
        int ex;
        frexpf(bound, &ex);
        s = ldexpf(1.0f, max(-100, min(100, 14 - ex)));
    }
    if (threadIdx.x == 0) {
        rscale[r] = 1.0f/s;
    }
    for (int64_t c0 = 128*wave; c0 < p.k; c0 += 128*(blockDim.x/32)) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t col = c0 + 32*j + lane;
            v[j] = src[col]*mmh_rot(rin, p.rin_f16, col)*s;
        }
        mmh_fwht(v, lane, p.had, p.hscale);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            xh[p.k*r + c0 + 32*j + lane] = __float2half_rn(v[j]);
        }
    }
}

// up = glu(gate, up) for the local rows of chunk c
static __global__ void mmh_moe_glu(const mmh_moe_route rt, const int c, float * __restrict__ up,
        const float * __restrict__ gate, const int64_t n_ff, const bool clamp, const float limit) {
    if (c >= *rt.n_chunks) {
        return;
    }
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= rt.chunks[2*c + 1]*n_ff) {
        return;
    }
    float g = gate[i];
    float u = up[i];
    if (clamp) {
        g = fminf(g, limit);
        u = fmaxf(fminf(u, limit), -limit);
    }
    up[i] = g/(1.0f + expf(-g))*u;
}

// out[:, t] += the sum, in slot order, of weights[k, t] * y[row] over the slots k of t whose row is in chunk c; one block
// per token, which leaves at once when none of its rows is in the chunk
static __global__ void __launch_bounds__(256) mmh_moe_scatter(const mmh_moe_route rt, const int c, const float * __restrict__ y,
        const float * __restrict__ weights, const int64_t w_nb1, const int64_t w_nb2, const int n_used,
        float * __restrict__ out, const int64_t out_nb1, const int n_embd) {
    if (c >= *rt.n_chunks) {
        return;
    }
    __shared__ int   s_p[MMH_MOE_MAX_USED];
    __shared__ float s_w[MMH_MOE_MAX_USED];
    __shared__ int   s_any;
    const int t  = blockIdx.x;
    const int b  = rt.chunks[2*c + 0];
    const int nr = rt.chunks[2*c + 1];
    if (threadIdx.x == 0) {
        s_any = 0;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n_used; k += blockDim.x) {
        const int p = rt.pos[k + n_used*t] - b;
        const bool in = p >= 0 && p < nr;
        s_p[k] = in ? p : -1;
        s_w[k] = in ? *(const float *) ((const char *) weights + k*w_nb1 + t*w_nb2) : 0.0f;
        if (in) {
            s_any = 1;
        }
    }
    __syncthreads();
    if (!s_any) {
        return;
    }
    float * o = (float *) ((char *) out + t*out_nb1);
    for (int col = threadIdx.x; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < n_used; ++k) {
            if (s_p[k] >= 0) {
                acc += s_w[k]*y[(int64_t) s_p[k]*n_embd + col];
            }
        }
        o[col] += acc;
    }
}

static __global__ void mmh_moe_table_init(uint64_t * __restrict__ tab, const uint64_t * __restrict__ src, const int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        tab[i] = src ? src[i] : 0;
    }
}

static __device__ __forceinline__ uint64_t mmh_moe_src_address(const mmh_moe_stage_src & s, const int kind, const int e) {
    const ggml_cuda_expert_source src = ggml_cuda_expert_cache_select(s.data[kind], s.arena[kind], s.slots[kind], e,
        s.host_slots[kind], s.host_addresses[kind]);
    return (uint64_t) (uintptr_t) src.data + (uint64_t) src.channel*s.nb02[kind];
}

// the resident copier (side stream): the staged experts of every chunk, in chunk order, into buffer c % 2
static __global__ void __launch_bounds__(256) mmh_moe_stage_copy(const mmh_moe_route rt, const int stage_cap,
        const mmh_moe_stage_src s, char * __restrict__ buf0, char * __restrict__ buf1, mmh_moe_stage_flags * f,
        mmh_moe_stage_error * err, const int layer) {
    const int n_chunks = *rt.n_chunks;
    const int64_t units_slot = s.koff[3]/16;
    for (int c = 0; c < n_chunks; ++c) {
        if (c >= 2) {
            const uint64_t t0 = wall_clock64();
            int d;
            while ((d = __hip_atomic_load(&f->done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) < c - 1) {
                if (__hip_atomic_load(&f->abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
                    return;
                }
                if (wall_clock64() - t0 > MMH_MOE_STAGE_TIMEOUT) {
                    mmh_moe_stage_fail(f, err, 1, layer, c, d, wall_clock64() - t0);
                    return;
                }
                __builtin_amdgcn_s_sleep(2);
            }
        }
        char * buf = (c & 1) ? buf1 : buf0;
        const int n = rt.stage_n[c];
        for (int64_t u = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; u < n*units_slot; u += (int64_t) gridDim.x*blockDim.x) {
            const int     j    = (int) (u / units_slot);
            const int64_t off  = (u % units_slot)*16;
            const int     kind = off < s.koff[1] ? 0 : off < s.koff[2] ? 1 : 2;
            if (off - s.koff[kind] >= s.nb02[kind]) {
                continue; // padding between kinds
            }
            const int      e = rt.stage_list[c*stage_cap + j];
            const uint64_t a = mmh_moe_src_address(s, kind, e);
            if (a % 16 != 0) {
                mmh_moe_stage_fail(f, err, 3, layer, c, e, 0);
                return;
            }
            const uint4 v = *(const uint4 *) (uintptr_t) (a + (off - s.koff[kind]));
            *(uint4 *) (buf + j*s.koff[3] + off) = v;
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            if (atomicAdd(&f->arrive[c], 1) == (int) gridDim.x - 1) {
                __threadfence();
                __hip_atomic_store(&f->ready[c], 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
}

// compute stream, before the GEMMs of chunk c: wait for its copies, then point its staged experts at the buffer
static __global__ void mmh_moe_stage_wait(const mmh_moe_route rt, const int c, const int stage_cap, const mmh_moe_stage_src s,
        char * buf0, char * buf1, uint64_t * __restrict__ tab0, uint64_t * __restrict__ tab1, uint64_t * __restrict__ tab2,
        mmh_moe_stage_flags * f, mmh_moe_stage_error * err, const int layer, const bool spin) {
    if (c >= *rt.n_chunks || rt.stage_n[c] == 0) {
        return;
    }
    __shared__ int s_ready;
    if (threadIdx.x == 0 && !spin) {
        s_ready = 1;
    } else if (threadIdx.x == 0) {
        const uint64_t t0 = wall_clock64();
        int r = 0;
        while (!(r = __hip_atomic_load(&f->ready[c], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT))) {
            if (__hip_atomic_load(&f->abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
                break;
            }
            if (wall_clock64() - t0 > MMH_MOE_STAGE_TIMEOUT) {
                mmh_moe_stage_fail(f, err, 2, layer, c, r, wall_clock64() - t0);
                break;
            }
            __builtin_amdgcn_s_sleep(2);
        }
        s_ready = r;
    }
    __syncthreads();
    if (!s_ready) {
        return;
    }
    char * buf = (c & 1) ? buf1 : buf0;
    for (int j = threadIdx.x; j < rt.stage_n[c]; j += blockDim.x) {
        const int e = rt.stage_list[c*stage_cap + j];
        const uint64_t base = (uint64_t) (uintptr_t) (buf + j*s.koff[3]);
        tab0[e] = base + s.koff[0];
        tab1[e] = base + s.koff[1];
        tab2[e] = base + s.koff[2];
        atomicAdd(&f->staged, 1);
    }
}

// one block after the routing pass: the chunk count, the staged experts per chunk and their sources, for the host
static __global__ void mmh_moe_stage_plan_kernel(const mmh_moe_route rt, const int stage_cap, const mmh_moe_stage_src s,
        mmh_moe_stage_plan * plan) {
    const int n = *rt.n_chunks;
    if (threadIdx.x == 0) {
        plan->n_chunks = n;
    }
    for (int c = 0; c < n; ++c) {
        const int ns = rt.stage_n[c];
        if (threadIdx.x == 0) {
            plan->stage_n[c] = ns;
        }
        for (int i = threadIdx.x; i < 3*ns; i += blockDim.x) {
            plan->src[3*(c*stage_cap) + i] = mmh_moe_src_address(s, i % 3, rt.stage_list[c*stage_cap + i/3]);
        }
    }
    __threadfence_system();
}

// compute stream, after the down GEMM of chunk c: restore the entries of its staged experts, then count the chunk done
static __global__ void mmh_moe_stage_release(const mmh_moe_route rt, const int c, const int stage_cap, const mmh_moe_stage_src s,
        uint64_t * __restrict__ tab0, uint64_t * __restrict__ tab1, uint64_t * __restrict__ tab2, mmh_moe_stage_flags * f) {
    if (c >= *rt.n_chunks) {
        return;
    }
    for (int j = threadIdx.x; j < rt.stage_n[c]; j += blockDim.x) {
        const int e = rt.stage_list[c*stage_cap + j];
        tab0[e] = s.host_addresses[0] ? s.host_addresses[0][e] : 0;
        tab1[e] = s.host_addresses[1] ? s.host_addresses[1][e] : 0;
        tab2[e] = s.host_addresses[2] ? s.host_addresses[2][e] : 0;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(&f->done, c + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template <ggml_type type>
static void mmh_moe_gemm_t(const int mt, const dim3 grid, cudaStream_t stream, const mmh_params & p, const char * w,
        const half * xh, const float * rscale, const int * items, const int * n_items, const int * iota) {
    switch (mt) {
        case 16:  mmh_gemm<type,  16><<<grid, 256, 0, stream>>>(p, w, xh, rscale, items, n_items, iota, nullptr, nullptr); break;
        case 32:  mmh_gemm<type,  32><<<grid, 256, 0, stream>>>(p, w, xh, rscale, items, n_items, iota, nullptr, nullptr); break;
        case 64:  mmh_gemm<type,  64><<<grid, 256, 0, stream>>>(p, w, xh, rscale, items, n_items, iota, nullptr, nullptr); break;
        default:  mmh_gemm<type, 128><<<grid, 256, 0, stream>>>(p, w, xh, rscale, items, n_items, iota, nullptr, nullptr); break;
    }
}

static void mmh_moe_gemm(const ggml_type type, const int mt, const dim3 grid, cudaStream_t stream, const mmh_params & p,
        const char * w, const half * xh, const float * rscale, const int * items, const int * n_items, const int * iota) {
    switch (type) {
#define MMH_MOE_CASE(T) case T: mmh_moe_gemm_t<T>(mt, grid, stream, p, w, xh, rscale, items, n_items, iota); break;
        MMH_MOE_CASE(GGML_TYPE_EXL3_M1)  MMH_MOE_CASE(GGML_TYPE_EXL3_M2)  MMH_MOE_CASE(GGML_TYPE_EXL3_M3)
        MMH_MOE_CASE(GGML_TYPE_EXL3_M4)  MMH_MOE_CASE(GGML_TYPE_EXL3_M5)  MMH_MOE_CASE(GGML_TYPE_EXL3_M6)
        MMH_MOE_CASE(GGML_TYPE_EXL3_M7)  MMH_MOE_CASE(GGML_TYPE_EXL3_M8)  MMH_MOE_CASE(GGML_TYPE_EXL3_M1H)
        MMH_MOE_CASE(GGML_TYPE_EXL3_M2H) MMH_MOE_CASE(GGML_TYPE_EXL3_M3H) MMH_MOE_CASE(GGML_TYPE_EXL3_G1)
        MMH_MOE_CASE(GGML_TYPE_EXL3_G2)  MMH_MOE_CASE(GGML_TYPE_EXL3_G3)  MMH_MOE_CASE(GGML_TYPE_EXL3_G4)
        MMH_MOE_CASE(GGML_TYPE_EXL3_G5)  MMH_MOE_CASE(GGML_TYPE_EXL3_G6)  MMH_MOE_CASE(GGML_TYPE_EXL3_G7)
        MMH_MOE_CASE(GGML_TYPE_EXL3_G8)  MMH_MOE_CASE(GGML_TYPE_EXL3_T1)  MMH_MOE_CASE(GGML_TYPE_EXL3_T2)
        MMH_MOE_CASE(GGML_TYPE_EXL3_T3)  MMH_MOE_CASE(GGML_TYPE_EXL3_T4)  MMH_MOE_CASE(GGML_TYPE_EXL3_T5)
        MMH_MOE_CASE(GGML_TYPE_EXL3_T6)  MMH_MOE_CASE(GGML_TYPE_EXL3_T7)  MMH_MOE_CASE(GGML_TYPE_EXL3_T8)
#undef MMH_MOE_CASE
        default:
            GGML_ABORT("unsupported MOE_FFN_HAD weight type %s", ggml_type_name(type));
    }
}

static int mmh_moe_env_int(const char * name, const int def) {
    const char * v = getenv(name);
    return v ? atoi(v) : def;
}

// the rows of a pass come from the op (the graph builder passes the layer mode's value after its no_alloc measurements,
// 0 before), else the default
static int mmh_moe_chunk_rows(const ggml_tensor * op) {
    const int r = ggml_get_op_params_i32(op, 2);
    return r > 0 ? r : MMH_MOE_ROWS;
}

// rows per GEMM item from the mean rows of an expert: the tile that holds them (VRAM weights, as MUL_MAT_HAD), or for
// host weights the one that holds three times as many; RANMA_EXPERT_MAJOR_MT forces one tile for both
static int mmh_moe_mt(const int64_t n_rows, const int64_t n_exp, const bool remote) {
    static const int env = mmh_moe_env_int("RANMA_EXPERT_MAJOR_MT", 0);
    if (env == 16 || env == 32 || env == 64 || env == 128) {
        return env;
    }
    const int64_t used = std::max<int64_t>(1, std::min(n_rows, n_exp));
    const int64_t want = (remote ? 3 : 1)*((n_rows + used - 1)/used);
    return want <= 16 ? 16 : want <= 32 ? 32 : want <= 64 ? 64 : 128;
}

static int64_t mmh_moe_max_items(const int64_t chunk_rows, const int64_t n_exp, const int mt) {
    return std::min(n_exp, chunk_rows) + chunk_rows/mt + 1;
}

// the staging state of a device: two buffers, the flags and the op's three address tables (kept, a captured graph never
// holds them: graphs with this op are not captured while staging is on)
struct mmh_moe_stage_dev {
    char                * buf         = nullptr;
    size_t                bytes       = 0;
    mmh_moe_stage_flags * flags       = nullptr;
    uint64_t            * tabs        = nullptr; // 3*MMH_MAX_EXPERTS
    int32_t             * no_slots    = nullptr; // MMH_MAX_EXPERTS of -1
    cudaStream_t          side        = nullptr;
    cudaEvent_t           ev_plan     = nullptr;
    cudaEvent_t           ev_done     = nullptr;
    mmh_moe_stage_error * err_host    = nullptr; // mapped
    mmh_moe_stage_error * err_dev     = nullptr;
    mmh_moe_stage_plan  * plan_host   = nullptr; // mapped
    mmh_moe_stage_plan  * plan_dev    = nullptr;
    cudaEvent_t           ev_ready[MMH_MOE_STAGE_MAX_CHUNKS] = {};
    cudaEvent_t           ev_cdone[MMH_MOE_STAGE_MAX_CHUNKS] = {};
};

// host function after a staged op: report and abort on a recorded error
static void mmh_moe_stage_check(void * data) {
    const mmh_moe_stage_error * e = (const mmh_moe_stage_error *) data;
    const int code = __atomic_load_n(&e->code, __ATOMIC_ACQUIRE);
    if (code != 0) {
        static const char * what[] = { "", "the copier waited for a free buffer (done)", "a chunk waited for its copies (ready)",
            "a staged source is not 16-byte aligned (expert)" };
        GGML_LOG_ERROR("expert-major staging failed: %s; layer %d, chunk %d, value %d, waited %.3f s\n",
            code >= 1 && code <= 3 ? what[code] : "?", e->layer, e->chunk, e->value, e->ticks/1e8);
        GGML_ABORT("expert-major staging failed");
    }
}

static size_t mmh_moe_stage_bytes() {
    const size_t mib = (size_t) std::max(0, mmh_moe_env_int("RANMA_EXPERT_MAJOR_STAGE_MIB", 0)); // per call, see mmh_moe_chunk_rows
    return mib << 20;
}

bool ggml_cuda_moe_ffn_had_graph_ok(void) {
    return mmh_moe_stage_bytes() == 0;
}

static mmh_moe_stage_dev * mmh_moe_stage_get(const int device) {
    static std::mutex mutex;
    static std::map<int, mmh_moe_stage_dev> devs;
    std::lock_guard<std::mutex> lock(mutex);
    mmh_moe_stage_dev & d = devs[device];
    if (d.buf == nullptr) {
        d.bytes = mmh_moe_stage_bytes();
        CUDA_CHECK(cudaMalloc((void **) &d.buf, d.bytes));
        CUDA_CHECK(cudaMalloc((void **) &d.flags, sizeof(mmh_moe_stage_flags)));
        CUDA_CHECK(cudaMalloc((void **) &d.tabs, 3*MMH_MAX_EXPERTS*sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc((void **) &d.no_slots, MMH_MAX_EXPERTS*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(d.no_slots, 0xff, MMH_MAX_EXPERTS*sizeof(int32_t)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&d.side, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&d.ev_plan, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&d.ev_done, cudaEventDisableTiming));
        CUDA_CHECK(cudaHostAlloc((void **) &d.err_host, sizeof(mmh_moe_stage_error), cudaHostAllocMapped));
        memset(d.err_host, 0, sizeof(mmh_moe_stage_error));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &d.err_dev, d.err_host, 0));
        CUDA_CHECK(cudaHostAlloc((void **) &d.plan_host, sizeof(mmh_moe_stage_plan), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &d.plan_dev, d.plan_host, 0));
        for (int i = 0; i < MMH_MOE_STAGE_MAX_CHUNKS; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&d.ev_ready[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&d.ev_cdone[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        GGML_LOG_INFO("%s: expert-major staging: %.0f MiB of VRAM on device %d\n", __func__, d.bytes/1048576.0, device);
    }
    return &d;
}

bool ggml_cuda_moe_ffn_had_supported(const int device, const ggml_tensor * op) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || ggml_cuda_info().devices[device].warp_size != 32) {
        return false;
    }
    const ggml_tensor * x       = op->src[0];
    const ggml_tensor * ids     = op->src[1];
    const ggml_tensor * weights = op->src[2];
    if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || ids->type != GGML_TYPE_I32 ||
            weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    const int64_t n_rows = ggml_nelements(ids);
    if (n_rows >= (1 << 30) || ids->ne[0] > MMH_MOE_MAX_USED) {
        return false;
    }
    for (int i = 3; i <= 5; ++i) {
        const ggml_tensor * d = op->src[i];
        const ggml_tensor * w = d->src[0];
        if (!ggml_is_exl3(w->type) || ggml_get_op_params_i32(d, 0) != MMH_NG || !ggml_cuda_mul_mat_had_supported(device, d) ||
                !mmh_input_f16(d) || w->view_src != nullptr) {
            return false;
        }
    }
    const int64_t n_exp = op->src[3]->src[0]->ne[2];
    const int64_t R     = std::min<int64_t>(mmh_moe_chunk_rows(op), n_rows);
    return mmh_moe_max_items(R, n_exp, mmh_moe_mt(n_rows, n_exp, false)) <= 65535;
}

void ggml_cuda_moe_ffn_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * ids     = dst->src[1];
    const ggml_tensor * weights = dst->src[2];
    const ggml_tensor * desc[3] = { dst->src[3], dst->src[4], dst->src[5] }; // up, gate, down

    const ggml_glu_op glu   = (ggml_glu_op) ggml_get_op_params_i32(dst, 0);
    const float       limit = ggml_get_op_params_f32(dst, 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int     n_used   = (int) ids->ne[0];
    const int64_t n_rows   = (int64_t) n_used*n_tokens;
    const int64_t n_ff     = desc[0]->src[0]->ne[1];
    const int64_t n_exp    = desc[0]->src[0]->ne[2];

    cudaStream_t stream = ctx.stream();

    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));

#if defined(GGML_USE_HIP)
    // expert cache: the SSD tier may reuse the ring slots of the layer once every chunk has read them
    struct done_guard {
        ggml_backend_cuda_context & ctx;
        const ggml_tensor * const * desc;
        ~done_guard() {
            for (int i = 0; i < 3; ++i) {
                ggml_cuda_expert_layer_done(ctx, desc[i]->src[0]);
            }
        }
    } done{ctx, desc};
#endif // defined(GGML_USE_HIP)

    if (n_rows == 0) {
        return;
    }

    // the three GEMMs: local row r of a chunk is output row r of an [n, 1, R] buffer
    // RANMA_EXPERT_MAJOR_KIND_WAIT: lazy (default) puts the staged SSD tier's wait for the gate and down kinds right
    // before the first GEMM of that kind, so the file reads of a later kind overlap the earlier kinds' GEMMs (as the
    // per-kind ops do); "all" waits for the three kinds before the first chunk
    static const bool kind_lazy = [] {
        const char * v = getenv("RANMA_EXPERT_MAJOR_KIND_WAIT");
        return v == nullptr || strcmp(v, "all") != 0;
    }();
    mmh_params   p[3];
    const char * wd[3];
    bool remote = false; // some weight is read from host memory (a mapped host buffer, the exclusive buffer)
    for (int i = 0; i < 3; ++i) {
        const ggml_tensor * w       = desc[i]->src[0];
        const ggml_tensor * rot_in  = desc[i]->src[3];
        const ggml_tensor * rot_out = desc[i]->src[4];
        mmh_params & q = p[i];
        memset(&q, 0, sizeof(q));
        q.k        = w->ne[0];
        q.n_exp    = w->ne[2];
        q.nb01     = w->nb[1];
        q.nb02     = w->nb[2];
        q.n_rows   = n_rows;
        q.ne1      = 1;
        q.ne2      = (int) std::min<int64_t>(n_rows, INT_MAX);
        q.nb1      = w->ne[1]*sizeof(float);
        q.nb2      = w->ne[1]*sizeof(float);
        q.nb3      = 0;
        q.rot_in   = mmh_data(rot_in);
        q.rin_nb1  = rot_in->nb[1];
        q.rot_out  = mmh_data(rot_out);
        q.rout_nb0 = rot_out->nb[0];
        q.rout_nb1 = rot_out->nb[1];
        q.rin_f16  = rot_in->type == GGML_TYPE_F16;
        q.rout_f16 = rot_out->type == GGML_TYPE_F16;
        q.had      = MMH_NG;
        q.hscale   = 1.0f/sqrtf((float) MMH_NG);
        // the expert cache lookup; the staged SSD tier's wait for the gate and down kinds is placed below (up was waited
        // for by the top-k publish)
        wd[i] = mmh_weight_data(ctx, w, ids, q, kind_lazy && i > 0);
        remote = remote || (w->buffer && (ggml_backend_buffer_is_host(w->buffer) ||
            ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_get_type(w->buffer))));
    }
    const bool all_local = !remote;

    const int64_t R     = std::min<int64_t>(mmh_moe_chunk_rows(dst), n_rows);
    const int     mt_v  = mmh_moe_mt(n_rows, n_exp, false);
    const int     mt_r  = mmh_moe_mt(n_rows, n_exp, true);
    const int64_t max_items_v = mmh_moe_max_items(R, n_exp, mt_v);
    const int64_t max_items_r = mmh_moe_max_items(R, n_exp, mt_r);
    GGML_ASSERT(max_items_v <= 65535 && max_items_r <= 65535);

    // staging: slots of whole experts (all three kinds) in each of the two buffers
    static const int stage_min_env = mmh_moe_env_int("RANMA_EXPERT_MAJOR_STAGE_MIN", 0);
    static const int stage_blocks  = std::max(32, mmh_moe_env_int("RANMA_EXPERT_MAJOR_STAGE_BLOCKS", 64));
    static const bool stage_log    = mmh_moe_env_int("RANMA_EXPERT_MAJOR_STAGE_LOG", 0) != 0;
    static const bool host_last    = [] {
        const char * v = getenv("RANMA_EXPERT_MAJOR_ORDER");
        return v != nullptr && strcmp(v, "last") == 0;
    }();
    static const bool stage_sdma   = [] {
        const char * v = getenv("RANMA_EXPERT_MAJOR_STAGE_ENGINE");
        return v == nullptr || strcmp(v, "kernel") != 0;
    }();
    mmh_moe_stage_src ss;
    memset(&ss, 0, sizeof(ss));
    int64_t koff = 0;
    for (int i = 0; i < 3; ++i) {
        ss.koff[i] = koff;
        koff += GGML_PAD(p[i].nb02, 256);
    }
    ss.koff[3] = koff;
    const int stage_min = stage_min_env > 0 ? stage_min_env : mt_r + 1;
    int stage_cap = 0;
    int64_t c_max = 3*((n_rows + R - 1)/R) + 1;
    if (remote && mmh_moe_stage_bytes() > 0) {
        stage_cap = (int) std::min<int64_t>(MMH_MOE_STAGE_MAX_SLOTS, (int64_t) (mmh_moe_stage_bytes()/2)/ss.koff[3]);
        if (stage_cap > 0) {
            c_max += n_rows/((int64_t) stage_cap*std::max(1, stage_min)) + 1;
        }
        if (c_max > MMH_MOE_STAGE_MAX_CHUNKS) {
            stage_cap = 0;
        }
    }
    if (stage_cap == 0) {
        c_max = 3*((n_rows + R - 1)/R) + 1;
    }

    ggml_cuda_pool_alloc<int>   route(ctx.pool(), 3*n_rows + 2*n_exp + 2*c_max + 1 + c_max*(stage_cap + 1));
    ggml_cuda_pool_alloc<int>   items(ctx.pool(), 3*(max_items_v + max_items_r) + 2);
    ggml_cuda_pool_alloc<int>   iota(ctx.pool(), R);
    ggml_cuda_pool_alloc<half>  xh(ctx.pool(), R*std::max(n_embd, n_ff));
    ggml_cuda_pool_alloc<float> rscale(ctx.pool(), R);
    ggml_cuda_pool_alloc<float> vu(ctx.pool(), R*n_ff);
    ggml_cuda_pool_alloc<float> vg(ctx.pool(), R*n_ff);
    ggml_cuda_pool_alloc<float> vy(ctx.pool(), R*n_embd);

    mmh_moe_route rt;
    rt.rows_sorted = route.get();
    rt.pos         = rt.rows_sorted + n_rows;
    rt.eid         = rt.pos + n_rows;
    rt.offs        = rt.eid + n_rows;
    rt.cnt         = rt.offs + n_exp;
    rt.chunks      = rt.cnt + n_exp;
    rt.n_chunks    = rt.chunks + 2*c_max;
    rt.stage_n     = rt.n_chunks + 1;
    rt.stage_list  = rt.stage_n + c_max;
    int * items_v   = items.get();
    int * items_r   = items_v + 3*max_items_v;
    int * n_items_v = items_r + 3*max_items_r;
    int * n_items_r = n_items_v + 1;

    const int32_t * ids_d   = (const int32_t *) ids->data;
    const int64_t   ids_nb0 = ids->nb[0];
    const int64_t   ids_nb1 = ids->nb[1];
    const int32_t * slots   = p[0].w_slots; // one table per layer, for all kinds

    mmh_moe_stage_dev * sd = stage_cap > 0 ? mmh_moe_stage_get(ctx.device) : nullptr;
    // gate and down waits still to enqueue (lazy mode). The staging plan and table init read the host addresses of
    // every kind before the first chunk, so with staging they are enqueued now, as before.
    bool kind_wait[3] = { false, kind_lazy, kind_lazy };
#if defined(GGML_USE_HIP)
    auto enqueue_kind_wait = [&](int i) {
        if (kind_wait[i]) {
            kind_wait[i] = false;
            const ggml_tensor * w = desc[i]->src[0];
            if (w->buffer && (ggml_backend_buffer_is_host(w->buffer) ||
                    ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_get_type(w->buffer)))) {
                ggml_cuda_expert_before_read(ctx, w);
            }
        }
    };
    if (sd) {
        enqueue_kind_wait(1);
        enqueue_kind_wait(2);
    }
#else
    GGML_UNUSED(kind_wait);
    auto enqueue_kind_wait = [&](int) {};
#endif // defined(GGML_USE_HIP)
    const int layer = atoi(desc[0]->src[0]->name + (strncmp(desc[0]->src[0]->name, "blk.", 4) == 0 ? 4 : 0));
    uint64_t * tab[3] = { nullptr, nullptr, nullptr };
    if (sd) {
        for (int i = 0; i < 3; ++i) {
            ss.data[i]           = wd[i];
            ss.arena[i]          = p[i].w_cache;
            ss.slots[i]          = p[i].w_slots;
            ss.host_slots[i]     = p[i].w_host_slots;
            ss.host_addresses[i] = p[i].w_host_addresses;
            ss.nb02[i]           = p[i].nb02;
            tab[i] = sd->tabs + (size_t) i*MMH_MAX_EXPERTS;
            mmh_moe_table_init<<<(unsigned) ((n_exp + 255)/256), 256, 0, stream>>>(tab[i], p[i].w_host_addresses, (int) n_exp);
            CUDA_CHECK(cudaGetLastError());
            // the GEMMs take the op's table where the tier's would be; without the cache a table of no VRAM slots
            if (p[i].w_cache == nullptr || p[i].w_slots == nullptr) {
                p[i].w_cache = sd->buf;
                p[i].w_slots = sd->no_slots;
            }
            p[i].w_host_addresses = tab[i];
        }
        CUDA_CHECK(cudaMemsetAsync(sd->flags, 0, sizeof(mmh_moe_stage_flags), stream));
    }

    mmh_moe_route_kernel<<<1, 1024, 0, stream>>>(ids_d, ids_nb0, ids_nb1, n_used, (int) n_rows, (int) n_exp, (int) R,
        (int) c_max, slots, all_local, stage_cap, stage_min, host_last, rt);
    CUDA_CHECK(cudaGetLastError());
    mmh_moe_iota<<<(unsigned) ((R + 255)/256), 256, 0, stream>>>(iota.get(), (int) R);
    CUDA_CHECK(cudaGetLastError());

    char * buf0 = sd ? sd->buf : nullptr;
    char * buf1 = sd ? sd->buf + (sd->bytes/2)/256*256 : nullptr;
    int n_loop = (int) c_max;
    auto flush_side = [&]() {
#if defined(GGML_USE_HIP)
        (void) hipStreamQuery(sd->side);
#else
        (void) cudaStreamQuery(sd->side);
#endif // defined(GGML_USE_HIP)
    };
    // copy engine: the copies of chunk c on the side stream, after the GEMMs of chunk c - 2 (which used the same buffer)
    auto enqueue_copies = [&](const int c) {
        if (c >= n_loop) {
            return;
        }
        if (c >= 2) {
            CUDA_CHECK(cudaStreamWaitEvent(sd->side, sd->ev_cdone[c - 2], 0));
        }
        char * buf = (c & 1) ? buf1 : buf0;
        for (int j = 0; j < sd->plan_host->stage_n[c]; ++j) {
            for (int k = 0; k < 3; ++k) {
                const void * src = (const void *) (uintptr_t) sd->plan_host->src[3*(c*stage_cap + j) + k];
                CUDA_CHECK(cudaMemcpyAsync(buf + j*ss.koff[3] + ss.koff[k], src, ss.nb02[k], cudaMemcpyDeviceToDevice, sd->side));
            }
        }
        CUDA_CHECK(cudaEventRecord(sd->ev_ready[c], sd->side));
        flush_side();
    };
    if (sd && stage_sdma) {
        CUDA_CHECK(cudaEventRecord(sd->ev_plan, stream));
        CUDA_CHECK(cudaStreamWaitEvent(sd->side, sd->ev_plan, 0));
        mmh_moe_stage_plan_kernel<<<1, 256, 0, stream>>>(rt, stage_cap, ss, sd->plan_dev);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        n_loop = sd->plan_host->n_chunks;
        enqueue_copies(0);
        enqueue_copies(1);
    } else if (sd) {
        // the copier starts after the routing pass; flush the side stream so that it is resident before the GEMMs
        CUDA_CHECK(cudaEventRecord(sd->ev_plan, stream));
        CUDA_CHECK(cudaStreamWaitEvent(sd->side, sd->ev_plan, 0));
        mmh_moe_stage_copy<<<stage_blocks, 256, 0, sd->side>>>(rt, stage_cap, ss, buf0, buf1, sd->flags, sd->err_dev, layer);
        CUDA_CHECK(cudaGetLastError());
        flush_side();
    }

    float * outs[3] = { vu.get(), vg.get(), vy.get() };
    const dim3 grid_glu((unsigned) ((R*n_ff + 255)/256));
    const dim3 grid_scatter((unsigned) n_tokens);

    for (int c = 0; c < n_loop; ++c) {
        mmh_moe_items_kernel<<<1, 1024, 0, stream>>>(rt, c, (int) n_exp, slots, all_local, stage_cap, mt_v, mt_r,
            items_v, n_items_v, items_r, n_items_r);
        CUDA_CHECK(cudaGetLastError());
        if (sd) {
            if (stage_sdma) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, sd->ev_ready[c], 0));
            }
            mmh_moe_stage_wait<<<1, 64, 0, stream>>>(rt, c, stage_cap, ss, buf0, buf1, tab[0], tab[1], tab[2], sd->flags,
                sd->err_dev, layer, !stage_sdma);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int i = 0; i < 3; ++i) {
            const bool local = i == 2;
            mmh_moe_prologue<<<(unsigned) R, 256, 0, stream>>>(p[i], rt, c, ids_d, ids_nb0, ids_nb1, n_used,
                local ? vu.get() : (const float *) x->data, local ? n_ff*(int64_t) sizeof(float) : (int64_t) x->nb[1], local,
                xh.get(), rscale.get());
            CUDA_CHECK(cudaGetLastError());
            mmh_params q = p[i];
            q.dst = outs[i];
            const unsigned groups = (unsigned) (desc[i]->src[0]->ne[1]/MMH_NG);
            mmh_moe_gemm(desc[i]->src[0]->type, mt_v, dim3(groups, (unsigned) max_items_v, 1), stream, q, wd[i], xh.get(),
                rscale.get(), items_v, n_items_v, iota.get());
            enqueue_kind_wait(i); // first chunk only; later chunks run after it in stream order
            CUDA_CHECK(cudaGetLastError());
            if (!all_local) {
                mmh_moe_gemm(desc[i]->src[0]->type, mt_r, dim3(groups, (unsigned) max_items_r, 1), stream, q, wd[i], xh.get(),
                    rscale.get(), items_r, n_items_r, iota.get());
                CUDA_CHECK(cudaGetLastError());
            }
            if (i == 1) {
                mmh_moe_glu<<<grid_glu, 256, 0, stream>>>(rt, c, vu.get(), vg.get(), n_ff,
                    glu == GGML_GLU_OP_SWIGLU_CLAMP, limit);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (sd) {
            mmh_moe_stage_release<<<1, 64, 0, stream>>>(rt, c, stage_cap, ss, tab[0], tab[1], tab[2], sd->flags);
            CUDA_CHECK(cudaGetLastError());
            if (stage_sdma) {
                CUDA_CHECK(cudaEventRecord(sd->ev_cdone[c], stream));
                enqueue_copies(c + 2);
            }
        }
        mmh_moe_scatter<<<grid_scatter, 256, 0, stream>>>(rt, c, vy.get(), (const float *) weights->data, weights->nb[1],
            weights->nb[2], n_used, (float *) dst->data, dst->nb[1], (int) n_embd);
        CUDA_CHECK(cudaGetLastError());
    }

    if (sd) {
        // the next op reuses the buffers, tables and flags
        CUDA_CHECK(cudaEventRecord(sd->ev_done, sd->side));
        CUDA_CHECK(cudaStreamWaitEvent(stream, sd->ev_done, 0));
        CUDA_CHECK(cudaLaunchHostFunc(stream, mmh_moe_stage_check, sd->err_host));
        if (stage_log) {
            mmh_moe_stage_flags f;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(&f, sd->flags, sizeof(f), cudaMemcpyDeviceToHost));
            int n_chunks = 0;
            CUDA_CHECK(cudaMemcpy(&n_chunks, rt.n_chunks, sizeof(int), cudaMemcpyDeviceToHost));
            // link bytes: the staged copies, and the in-place reads of the other host experts (one per item of mt_r rows)
            std::vector<int> cnt(n_exp), sl(n_exp, -1), sn(n_chunks), list((size_t) n_chunks*stage_cap);
            CUDA_CHECK(cudaMemcpy(cnt.data(), rt.cnt, n_exp*sizeof(int), cudaMemcpyDeviceToHost));
            if (slots) {
                CUDA_CHECK(cudaMemcpy(sl.data(), slots, n_exp*sizeof(int), cudaMemcpyDeviceToHost));
            }
            CUDA_CHECK(cudaMemcpy(sn.data(), rt.stage_n, n_chunks*sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(list.data(), rt.stage_list, list.size()*sizeof(int), cudaMemcpyDeviceToHost));
            std::vector<char> staged(n_exp, 0);
            int64_t n_staged = 0;
            for (int c = 0; c < n_chunks; ++c) {
                for (int j = 0; j < sn[c]; ++j) {
                    staged[list[(size_t) c*stage_cap + j]] = 1;
                    ++n_staged;
                }
            }
            const int64_t eb = p[0].nb02 + p[1].nb02 + p[2].nb02;
            int64_t reads = 0;
            for (int e = 0; e < n_exp; ++e) {
                if (cnt[e] > 0 && !all_local && sl[e] < 0 && !staged[e]) {
                    reads += (cnt[e] + mt_r - 1)/mt_r;
                }
            }
            GGML_LOG_INFO("%s: %s: %d chunks, %d experts staged, abort %d (cap %d, min %d, %s); link %.1f MiB staged + %.1f MiB read in place\n",
                __func__, desc[0]->src[0]->name, n_chunks, f.staged, f.abort, stage_cap, stage_min, stage_sdma ? "sdma" : "kernel",
                n_staged*eb/1048576.0, reads*eb/1048576.0);
        }
    }
}
