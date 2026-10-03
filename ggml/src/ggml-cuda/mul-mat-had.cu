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
// The GEMV keeps its F32 input in both modes (fp16 inputs did not make it faster), and the CPU backend always
// computes the fp32 grade.

#include "mul-mat-had.cuh"
#include "mapped-host.cuh"
#include "expert-controller.cuh"

#include <cfloat>

#define MMH_NG          128 // output columns per block, one H128 output block
#define MMH_GEMV_TOKENS   8 // GEMV without a routing pass up to this many tokens
#define MMH_GEMV_LIST   512 // most rows one GEMV block collects without a routing pass
#define MMH_MAX_EXPERTS 4096 // weight matrices the routing pass counts in LDS

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
    // exclusive mode a miss from the host slot of e; null without the cache. rot_in / rot_out stay indexed by e.
    const void     * w_cache;
    const int32_t  * w_slots;
    const int32_t  * w_host_slots;
    const uint64_t * w_host_addresses;
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
// - frag (GEMV): F32, per row and 16 values of k, for g = 0..3: k = 2g, 2g + 1, 2g + 8, 2g + 9
// - plain (GEMM, f16): fp16 values of the row scaled by a power of 2, K values per row; rscale holds the inverse scale
// The prologue also clears the k-split tickets of the GEMV that follows it (zero, n_zero).
template <bool frag, bool f16>
static __global__ void __launch_bounds__(256) mmh_prologue(const mmh_params p, half * __restrict__ xh, float * __restrict__ rscale,
        int * __restrict__ zero, const int n_zero) {
    static_assert(frag != f16, "the GEMV input is F32, the GEMM input fp16");
    __shared__ float s_red[8];

    const int r    = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int wave = threadIdx.x / 32;

    for (int i = r*blockDim.x + threadIdx.x; i < n_zero; i += gridDim.x*blockDim.x) {
        zero[i] = 0;
    }

    const int e = mmh_expert(p, r);
    const float * src = mmh_src_row(p, r);
    const char  * rin = p.rot_in + e*p.rin_nb1;

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
}

// rows grouped by weight matrix: rows_sorted, and items (expert, first row, row count <= mt)
// one block, counts in LDS by atomics, so the order inside one expert is arbitrary; each row is computed alone
static __global__ void __launch_bounds__(1024) mmh_route(const mmh_params p, const int mt, int * __restrict__ eid,
        int * __restrict__ rank, int * __restrict__ rows_sorted, int * __restrict__ items, int * __restrict__ n_items) {
    __shared__ int s_count[MMH_MAX_EXPERTS];
    __shared__ int s_offs[MMH_MAX_EXPERTS];
    __shared__ int s_rows[1024];
    __shared__ int s_items[1024];

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
    int sum_rows  = 0;
    int sum_items = 0;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        sum_rows  += c;
        sum_items += (c + mt - 1)/mt;
    }
    s_rows[tid]  = sum_rows;
    s_items[tid] = sum_items;
    __syncthreads();
    for (int h = 1; h < nth; h *= 2) {
        const int a = tid >= h ? s_rows[tid - h]  : 0;
        const int b = tid >= h ? s_items[tid - h] : 0;
        __syncthreads();
        s_rows[tid]  += a;
        s_items[tid] += b;
        __syncthreads();
    }
    int row  = s_rows[tid]  - sum_rows;
    int item = s_items[tid] - sum_items;
    for (int64_t e = e0; e < e1; ++e) {
        const int c = s_count[e];
        s_offs[e] = row;
        for (int i = 0; i < c; i += mt) {
            items[3*item + 0] = (int) e;
            items[3*item + 1] = row + i;
            items[3*item + 2] = min(mt, c - i);
            ++item;
        }
        row += c;
    }
    if (tid == nth - 1) {
        *n_items = s_items[tid];
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
};

static __device__ __forceinline__ uint32_t mmh_funnel(const uint32_t hi, const uint32_t lo, const int s) {
    return (uint32_t) (((((uint64_t) hi) << 32) | lo) >> s);
}

// two mul1 codebook values as half2: fp16(byte_sum(state*0x83DCD12D) + 1024)*k_inv + k_bias, one fp16 rounding
// x*0x83DCD12D mod 2^32 for x < 2^16 with 24-bit multiplies (v_mul_lo_u32 runs at a quarter of the rate);
// in asm, as the compiler folds the 24-bit products back into v_mul_lo_u32
static __device__ __forceinline__ uint32_t mmh_mul1_hash(const uint32_t x) {
#if defined(GGML_USE_HIP)
    uint32_t lo;
    uint32_t hi;
    asm("v_mul_u32_u24 %0, 0xdcd12d, %1" : "=v"(lo) : "v"(x));
    asm("v_mul_u32_u24 %0, 0x83, %1" : "=v"(hi) : "v"(x));
    return lo + (hi << 24);
#else
    return x*0x83DCD12Du;
#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ uint32_t mmh_mul1_pair(const uint32_t s0, const uint32_t s1) {
    const uint32_t p0 = mmh_mul1_hash(s0 & 0xFFFFu);
    const uint32_t p1 = mmh_mul1_hash(s1 & 0xFFFFu);
#if defined(GGML_USE_HIP)
    const uint32_t h = __builtin_amdgcn_sad_hi_u8(p1, 0u, __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u));
#else
    const uint32_t h = __dp4a(p0, 0x01010101u, 0x6400u) | (__dp4a(p1, 0x01010101u, 0x6400u) << 16);
#endif
    return mmh_as_u32(__hfma2(mmh_as_half2(h), mmh_as_half2(0x1EEE1EEEu), mmh_as_half2(0xC931C931u)));
}

// mcg and 3inst hashes of x < 2^16: x*0xCBAC1FED and x*89226354 + 64248484 mod 2^32, 24-bit multiplies as above
static __device__ __forceinline__ uint32_t mmh_mcg_hash(const uint32_t x) {
#if defined(GGML_USE_HIP)
    uint32_t lo;
    uint32_t hi;
    asm("v_mul_u32_u24 %0, 0xac1fed, %1" : "=v"(lo) : "v"(x));
    asm("v_mul_u32_u24 %0, 0xcb, %1" : "=v"(hi) : "v"(x));
    return lo + (hi << 24);
#else
    return x*0xCBAC1FEDu;
#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ uint32_t mmh_3inst_hash(const uint32_t x) {
#if defined(GGML_USE_HIP)
    uint32_t lo;
    uint32_t hi;
    asm("v_mul_u32_u24 %0, 0x517c72, %1" : "=v"(lo) : "v"(x));
    asm("v_mul_u32_u24 %0, 0x5, %1" : "=v"(hi) : "v"(x));
    return lo + (hi << 24) + 64248484u;
#else
    return x*89226354u + 64248484u;
#endif // defined(GGML_USE_HIP)
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
template <ggml_type type>
struct mmh_weight {
    static constexpr int  b2       = mmh_exl3_b2(type);
    static constexpr int  codebook = mmh_exl3_codebook(type);
    static constexpr bool is_exl3  = b2 > 0;
    static constexpr int  n_raw   = is_exl3 ? 3 : type == GGML_TYPE_F16 ? 4 : 8;

    struct raw {
        uint32_t v[n_raw];
    };

    const char * group; // wave-uniform base of the 128-row group
    int64_t  nb01;
    uint32_t off[3];    // EXL3: byte offsets of the 3 words of this lane in a tile; F16/F32: offset of (c0, r0)
    int      o;         // EXL3: bit offset of the first window in the first word

    __device__ __forceinline__ mmh_weight(const char * group, const int64_t nb01, const int lane) : group(group), nb01(nb01) {
        if constexpr (is_exl3) {
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
        if constexpr (is_exl3) {
            const char * t = group + (int64_t) (8*kt + nt)*mmh_exl3<b2>::tile_bytes;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                r.v[i] = *(const uint32_t *) (t + off[i]);
            }
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

    __device__ __forceinline__ void decode(const raw & r, uint32_t out[4], uint32_t lo[4]) const {
        if constexpr (is_exl3) {
            const uint32_t w0 = r.v[0];
            const uint32_t w1 = r.v[1];
            const uint32_t w2 = r.v[2];
            const uint32_t v0 = (uint32_t) (((((uint64_t) w0) << 32) | w1) >> (32 - o));
            const uint32_t v1 = (uint32_t) (((((uint64_t) w1) << 32) | w2) >> (32 - o));
            const uint32_t v2 = w2 << o;
            uint32_t st[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = mmh_exl3<b2>::end(j);
                st[j] = e <= 32 ? v0 >> (32 - e) : e <= 64 ? mmh_funnel(v0, v1, 64 - e) : mmh_funnel(v1, v2, 96 - e);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if constexpr (codebook == 0) {
                    out[i] = mmh_mul1_pair(st[2*i], st[2*i + 1]);
                } else if constexpr (codebook == 1) {
                    out[i] = mmh_half_sum_pair(mmh_mcg_hash(st[2*i] & 0xFFFFu), mmh_mcg_hash(st[2*i + 1] & 0xFFFFu));
                } else {
                    out[i] = mmh_half_sum_pair(mmh_3inst_hash(st[2*i] & 0xFFFFu), mmh_3inst_hash(st[2*i + 1] & 0xFFFFu));
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

// GEMV: a block of 8*ksplit waves owns one 128-row group g (blockIdx.x) of one expert, for the k range blockIdx.z of
// gridDim.z; wave = column tile nt + 8*k split. Its rows come from the routing pass (items) or are collected here: with
// ids, block y is row y and runs only if y is the first row of its expert, then for every row of that expert. The
// weight and input loads of the next pf k steps are in flight while a step is decoded. With a k split over blocks,
// each block stores its partial sums and the last block of the group adds them in k order, then runs the epilogue.
template <ggml_type type, int rows>
static __global__ void __launch_bounds__(1024) mmh_gemv(const mmh_params p, const char * __restrict__ w,
        const float4 * __restrict__ xh, const float * __restrict__ rscale, const int * __restrict__ items,
        const int * __restrict__ n_items, const int * __restrict__ rows_sorted, const int ksplit,
        float * __restrict__ part, int * __restrict__ tickets, const int max_chunks) {
    typedef mmh_weight<type> wtype;
    constexpr int pf = rows <= 2 ? 4 : 2;

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

    const wtype wl(mmh_weight_matrix(p, w, e) + (int64_t) g*MMH_NG*p.nb01, p.nb01, lane);
    const char * rout = p.rot_out + e*p.rout_nb1 + (int64_t) g*MMH_NG*p.rout_nb0;

    for (int i0 = 0; i0 < nr; i0 += rows) {
        const int nc = min(rows, nr - i0);

        // input row bases are wave-uniform, the lane adds its 16-byte group of each 16 k
        const float4 * xr[rows];
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            xr[i] = xh + (int64_t) mmh_uniform(rowp[i0 + (i < nc ? i : 0)])*(4*kslices) + lane % 4;
        }

        float acc[rows][2];
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            acc[i][0] = 0.0f;
            acc[i][1] = 0.0f;
        }

        typename wtype::raw wr[pf];
        float4 xv[pf][rows];
#pragma unroll
        for (int d = 0; d < pf; ++d) {
            if (kb + d < ke) {
                wr[d] = wl.fetch(kb + d, nt);
#pragma unroll
                for (int i = 0; i < rows; ++i) {
                    if (i < nc) {
                        xv[d][i] = xr[i][4*(kb + d)];
                    }
                }
            }
        }

        for (int kt = kb; kt < ke; kt += pf) {
#pragma unroll
            for (int d = 0; d < pf; ++d) {
                if (kt + d < ke) {
                    const typename wtype::raw cw = wr[d];
                    float4 cx[rows];
#pragma unroll
                    for (int i = 0; i < rows; ++i) {
                        cx[i] = xv[d][i];
                    }
                    if (kt + d + pf < ke) {
                        wr[d] = wl.fetch(kt + d + pf, nt);
#pragma unroll
                        for (int i = 0; i < rows; ++i) {
                            if (i < nc) {
                                xv[d][i] = xr[i][4*(kt + d + pf)];
                            }
                        }
                    }
                    uint32_t wf[4];
                    uint32_t wlo[4];
                    wl.decode(cw, wf, wlo);
#pragma unroll
                    for (int i = 0; i < rows; ++i) {
                        if (i < nc) {
                            const float4 x = cx[i];
                            float a0 = acc[i][0];
                            float a1 = acc[i][1];
                            if constexpr (type == GGML_TYPE_F32) {
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
                }
            }
        }

#pragma unroll
        for (int i = 0; i < rows; ++i) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 1, 32);
                acc[i][c] += __shfl_xor_sync(0xFFFFFFFF, acc[i][c], 2, 32);
            }
            if (i < nc && lane % 4 == 0) {
                s_part[ksp][i][16*nt + lane/4]     = acc[i][0];
                s_part[ksp][i][16*nt + lane/4 + 8] = acc[i][1];
            }
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
                s_last = atomicAdd(&tickets[t], 1) == kblocks - 1;
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
            const float rs = rscale[r];
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

    const char * rout = p.rot_out + e*p.rout_nb1 + (int64_t) g*MMH_NG*p.rout_nb0;
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

template <ggml_type type, bool f16>
static void mmh_launch(ggml_backend_cuda_context & ctx, const mmh_params & p, const char * w, const int64_t n,
        const bool inline_rows) {
    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t n_groups = n/MMH_NG;

    // the WMMA GEMM takes fp16 input: in the F32 mode the rows run the F32 GEMV
    const bool wmma = !inline_rows && f16 && type != GGML_TYPE_F32 && GGML_CUDA_CC_IS_RDNA4(cc);

    ggml_cuda_pool_alloc<half>  xh(ctx.pool(), (wmma ? 1 : 2)*p.n_rows*p.k);
    ggml_cuda_pool_alloc<float> rscale(ctx.pool(), p.n_rows);

    // rows per weight matrix, for the tile size
    const int64_t used    = p.ids ? std::min(p.n_rows, p.n_exp) : p.n_exp;
    const int64_t per_exp = (p.n_rows + used - 1)/used;

    if (inline_rows) {
        const int64_t n_items = p.ids ? p.n_rows : p.n_exp;
        const int     rows    = std::min<int64_t>(MMH_GEMV_TOKENS, per_exp);
        const float4 * x4 = (const float4 *) xh.get();

        // waves split k inside a block until there are enough waves; with very few blocks, blocks split k too
        const int64_t kslices = p.k/16;
        int ksplit  = 1;
        int kblocks = 1;
        while (ksplit < 4 && n_groups*used*8*ksplit < 4096 && kslices >= 8*ksplit*2) {
            ksplit *= 2;
        }
        while (kblocks < 8 && n_groups*used*kblocks < 32 && kslices >= 4*ksplit*kblocks*2) {
            kblocks *= 2;
        }

        ggml_cuda_pool_alloc<float> part(ctx.pool());
        ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
        const int max_chunks = (int) ((std::min<int64_t>(p.n_rows, MMH_GEMV_LIST) + rows - 1)/rows);
        const int n_tickets  = kblocks > 1 ? (int) (n_items*n_groups*max_chunks) : 0;
        if (kblocks > 1) {
            part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
            tickets.alloc(n_tickets);
        }
        mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets);
        CUDA_CHECK(cudaGetLastError());

        const dim3 grid(n_groups, n_items, kblocks);
        if (rows <= 1) {
            mmh_gemv<type, 1><<<grid, 256*ksplit, 0, stream>>>(p, w, x4, rscale.get(), nullptr, nullptr, nullptr, ksplit, part.ptr, tickets.ptr, max_chunks);
        } else if (rows <= 2) {
            mmh_gemv<type, 2><<<grid, 256*ksplit, 0, stream>>>(p, w, x4, rscale.get(), nullptr, nullptr, nullptr, ksplit, part.ptr, tickets.ptr, max_chunks);
        } else if (rows <= 4) {
            mmh_gemv<type, 4><<<grid, 256*ksplit, 0, stream>>>(p, w, x4, rscale.get(), nullptr, nullptr, nullptr, ksplit, part.ptr, tickets.ptr, max_chunks);
        } else {
            mmh_gemv<type, 8><<<grid, 256*ksplit, 0, stream>>>(p, w, x4, rscale.get(), nullptr, nullptr, nullptr, ksplit, part.ptr, tickets.ptr, max_chunks);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int mt = !wmma ? 8 : per_exp <= 16 ? 16 : per_exp <= 32 ? 32 : per_exp <= 64 ? 64 : 128;
    const int64_t max_items = used + p.n_rows/mt + 1;

    GGML_ASSERT(max_items <= 65535);

    // WMMA: blocks also split k while there are few blocks and each keeps >= 8 steps
    const int64_t items_est = p.ids ? used + p.n_rows/mt : p.n_exp*((per_exp + mt - 1)/mt);
    int kblocks = 1;
    while (wmma && kblocks < 8 && n_groups*items_est*kblocks < 256 && p.k/32 >= 8*kblocks*2) {
        kblocks *= 2;
    }
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    ggml_cuda_pool_alloc<int>   tickets(ctx.pool());
    const int n_tickets = kblocks > 1 ? (int) (max_items*n_groups) : 0;
    if (kblocks > 1) {
        part.alloc(p.n_rows*n_groups*kblocks*MMH_NG);
        tickets.alloc(n_tickets);
    }

    if (wmma) {
        mmh_prologue<false, true><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), tickets.ptr, n_tickets);
    } else {
        mmh_prologue<true, false><<<p.n_rows, 256, 0, stream>>>(p, xh.get(), rscale.get(), nullptr, 0);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<int> route(ctx.pool(), 3*p.n_rows + 3*max_items + 1);
    int * eid         = route.get();
    int * rank        = eid + p.n_rows;
    int * rows_sorted = rank + p.n_rows;
    int * items       = rows_sorted + p.n_rows;
    int * n_items     = items + 3*max_items;

    mmh_route<<<1, 1024, 0, stream>>>(p, mt, eid, rank, rows_sorted, items, n_items);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(n_groups, max_items, kblocks);
    if (wmma) {
        if constexpr (type != GGML_TYPE_F32) {
            if (mt == 16) {
                mmh_gemm<type, 16><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else if (mt == 32) {
                mmh_gemm<type, 32><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else if (mt == 64) {
                mmh_gemm<type, 64><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            } else {
                mmh_gemm<type, 128><<<grid, 256, 0, stream>>>(p, w, xh.get(), rscale.get(), items, n_items, rows_sorted, part.ptr, tickets.ptr);
            }
        }
    } else {
        // the waves split k also for precision: a lane then sums at most a quarter of k
        const int ksplit = p.k/16 >= 64 ? 4 : p.k/16 >= 32 ? 2 : 1;
        mmh_gemv<type, 8><<<grid, 256*ksplit, 0, stream>>>(p, w, (const float4 *) xh.get(), rscale.get(), items, n_items, rows_sorted, ksplit, nullptr, nullptr, 1);
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

// the weight base for the kernels; with ids, a routed expert weight of the expert cache also fills the cache fields
// of p (MUL_MAT_ID rule: a weight in a mapped host buffer or in the exclusive buffer, whose tensor has logical
// addresses only and is read through the lookup alone)
static const char * mmh_weight_data(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * ids,
        mmh_params & p) {
    p.w_cache          = nullptr;
    p.w_slots          = nullptr;
    p.w_host_slots     = nullptr;
    p.w_host_addresses = nullptr;
#if defined(GGML_USE_HIP)
    const bool exclusive = w->buffer && ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_get_type(w->buffer));
    GGML_ASSERT(!exclusive || ids);
    const char * data = mmh_data(w);
    if (ids && (exclusive || (w->buffer && ggml_backend_buffer_is_host(w->buffer)))) {
        const ggml_cuda_expert_lookup cached = ggml_cuda_expert_lookup_tensor(w);
        ggml_cuda_expert_before_read(ctx, w);
        p.w_cache          = cached.data;
        p.w_slots          = cached.slots;
        p.w_host_slots     = cached.host_slots;
        p.w_host_addresses = cached.host_addresses;
        if (cached.host_data != nullptr) {
            data = (const char *) cached.host_data;
        }
        GGML_ASSERT(!exclusive || p.w_host_slots != nullptr);
    }
    return data;
#else
    GGML_UNUSED_VARS(ctx, ids);
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
    // one row goes to the GEMM, which stages the rows once per block
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool wmma = w->type != GGML_TYPE_F32 && GGML_CUDA_CC_IS_RDNA4(cc);
    const int64_t tokens = ids ? dst->ne[2] : (p.n_rows + p.n_exp - 1)/p.n_exp;
    const bool inline_rows = p.n_rows <= MMH_GEMV_LIST && tokens <= (ids || !wmma ? MMH_GEMV_TOKENS : 1);

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
