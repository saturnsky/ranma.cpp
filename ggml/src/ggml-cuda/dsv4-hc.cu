#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// the mul and the add are separate ops in the unfused graph, so they must not contract into an FMA
static __device__ __forceinline__ float dsv4_hc_affine_sigmoid(float x, float scale, float base) {
    const float v = __fadd_rn(__fmul_rn(x, scale), base);
    return 1.0f / (1.0f + expf(-v));
}

static __global__ void dsv4_hc_coef_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_pre  = scale[0*ss0];
    const float scale_post = scale[1*ss0];
    const float scale_comb = scale[2*ss0];

    // pre: ggml_sigmoid then ggml_scale_bias(1.0f, eps)
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        const float s = dsv4_hc_affine_sigmoid(mixes[ih*sm0 + it*sm1], scale_pre, base[ih*sb0]);
        dst[ih*sd0 + it*sd1] = fmaf(1.0f, s, eps);
    }

    // post: ggml_sigmoid then ggml_scale(2.0f)
    for (int ih = 0; ih < DSV4_HC; ++ih) {
        const int idx = DSV4_HC + ih;
        const float s = dsv4_hc_affine_sigmoid(mixes[idx*sm0 + it*sm1], scale_post, base[idx*sb0]);
        dst[idx*sd0 + it*sd1] = fmaf(2.0f, s, 0.0f);
    }

    // comb: identical arithmetic to dsv4_hc_comb_f32
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int idx = 0; idx < DSV4_HC*DSV4_HC; ++idx) {
        dst[(comb_offset + idx)*sd0 + it*sd1] = comb[idx];
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_coef(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[1] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_coef_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// DeepSeek V4 KV compressor: fused gather + per-feature softmax + weighted sum.
// One block handles one compressor block; one thread handles one output feature.

template <bool overlap>
static __global__ void dsv4_compress_f32(
        const float   * __restrict__ kv,
        const float   * __restrict__ score,
        const int32_t * __restrict__ idxs,
        float         * __restrict__ dst,
        int64_t n_embd_head,
        int64_t n_rows,
        int64_t n_read,
        int     ratio,
        int64_t sk0,
        int64_t sk1,
        int64_t ss0,
        int64_t ss1,
        int64_t sd0,
        int64_t sd2) {
    ggml_cuda_pdl_lc();

    extern __shared__ int32_t s_idx[];

    const int64_t ib    = blockIdx.y;
    const int     n_per = overlap ? 2*ratio : ratio;

    ggml_cuda_pdl_sync();

    for (int k = threadIdx.x; k < n_per; k += blockDim.x) {
        const bool cur = overlap && k >= ratio;
        s_idx[k] = cur ? idxs[n_read + ib*ratio + (k - ratio)] : idxs[ib*ratio + k];
    }
    __syncthreads();

    const int64_t f = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= n_embd_head) {
        return;
    }

    // pass 1: max of the gathered scores for this feature
    float m = -INFINITY;
    for (int k = 0; k < n_per; ++k) {
        const int64_t row = s_idx[k];
        if ((uint64_t) row >= (uint64_t) n_rows) {
            continue; // missing segment: score is -INFINITY
        }
        const int64_t col = (overlap && k >= ratio) ? f + n_embd_head : f;
        m = fmaxf(m, score[col*ss0 + row*ss1]);
    }

    // pass 2: unnormalized softmax weights, weighted sum
    float sum = 0.0f;
    float acc = 0.0f;
    for (int k = 0; k < n_per; ++k) {
        const int64_t row = s_idx[k];
        if ((uint64_t) row >= (uint64_t) n_rows) {
            continue; // missing segment: weight 0, value 0
        }
        const int64_t col = (overlap && k >= ratio) ? f + n_embd_head : f;

        const float w = expf(score[col*ss0 + row*ss1] - m);
        sum += w;
        acc += w*kv[col*sk0 + row*sk1];
    }

    dst[f*sd0 + ib*sd2] = sum > 0.0f ? acc/sum : 0.0f;
}

void ggml_cuda_op_dsv4_compress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kv    = dst->src[0];
    const ggml_tensor * score = dst->src[1];
    const ggml_tensor * idxs  = dst->src[2];

    GGML_ASSERT(kv->type    == GGML_TYPE_F32);
    GGML_ASSERT(score->type == GGML_TYPE_F32);
    GGML_ASSERT(idxs->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(idxs));

    GGML_TENSOR_LOCALS(size_t, nbk, kv,    nb);
    GGML_TENSOR_LOCALS(size_t, nbs, score, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int     ratio   = ggml_get_op_params_i32(dst, 0);
    const bool    overlap = ggml_get_op_params_i32(dst, 1) != 0;

    const int64_t n_embd_head = dst->ne[0];
    const int64_t n_blocks    = dst->ne[2];
    const int64_t n_rows      = kv->ne[1];
    const int64_t n_read      = (int64_t) ratio*n_blocks;
    const int64_t n_per_block = overlap ? 2*ratio : ratio;

    GGML_ASSERT(dst->ne[1] == 1);
    GGML_ASSERT(idxs->ne[0] == n_per_block*n_blocks);

    // small blocks: the feature dimension is the only parallelism, and decode
    // calls this with a single compressor block
    const int block_size = 64;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd_head + block_size - 1)/block_size, n_blocks, 1);
    const size_t shmem = n_per_block*sizeof(int32_t);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, shmem, ctx.stream());

    auto kernel = overlap ? dsv4_compress_f32<true> : dsv4_compress_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) kv->data, (const float *) score->data, (const int32_t *) idxs->data, (float *) dst->data,
            n_embd_head, n_rows, n_read, ratio,
            (int64_t) (nbk0 / sizeof(float)), (int64_t) (nbk1 / sizeof(float)),
            (int64_t) (nbs0 / sizeof(float)), (int64_t) (nbs1 / sizeof(float)),
            (int64_t) (nbd0 / sizeof(float)), (int64_t) (nbd2 / sizeof(float)));
}

// ADD -> DSV4_HC_POST in one launch: x is the sum of the two ADD operands, formed in the operand
// order of the ADD node; the rest is the arithmetic of dsv4_hc_post_f32.
template <bool has_comb>
static __global__ void dsv4_hc_post_add_f32(
        const float * xa,
        const float * xb,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sxa0,
        int64_t sxa1,
        int64_t sxb0,
        int64_t sxb1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float x = xa[i0*sxa0 + it*sxa1] + xb[i0*sxb0 + it*sxb1];

    float sum = x * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_post_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst) {
    const ggml_tensor * xa       = add->src[0];
    const ggml_tensor * xb       = add->src[1];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(dst->src[0] == add);
    GGML_ASSERT(xa->type == GGML_TYPE_F32);
    GGML_ASSERT(xb->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(xa, add) && ggml_are_same_shape(xb, add));
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nba, xa,       nb);
    GGML_TENSOR_LOCALS(size_t, nbb, xb,       nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = add->ne[0];
    const int64_t n_tokens = add->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_add_f32<true> : dsv4_hc_post_add_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) xa->data, (const float *) xb->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nba0 / sizeof(float), nba1 / sizeof(float),
            nbb0 / sizeof(float), nbb1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// DSV4_COMPRESS reading its kv and score sources as [a | b] row concatenations, which removes the
// two CONCAT launches in front of it. Row r < n_rows_a comes from a, the others from b at r - n_rows_a;
// the per-feature loops are those of dsv4_compress_f32, only the load addresses differ.
struct dsv4_compress_src {
    const float * a;
    const float * b;
    int64_t sa0;
    int64_t sa1;
    int64_t sb0;
    int64_t sb1;
};

static __device__ __forceinline__ float dsv4_compress_src_load(
        const dsv4_compress_src & s, const int64_t n_rows_a, const int64_t col, const int64_t row) {
    return row < n_rows_a ? s.a[col*s.sa0 + row*s.sa1] : s.b[col*s.sb0 + (row - n_rows_a)*s.sb1];
}

template <bool overlap>
static __global__ void dsv4_compress_concat_f32(
        const dsv4_compress_src kv,
        const dsv4_compress_src score,
        const int32_t * __restrict__ idxs,
        float         * __restrict__ dst,
        int64_t n_embd_head,
        int64_t n_rows_a,
        int64_t n_rows,
        int64_t n_read,
        int     ratio,
        int64_t sd0,
        int64_t sd2) {
    ggml_cuda_pdl_lc();

    extern __shared__ int32_t s_idx[];

    const int64_t ib    = blockIdx.y;
    const int     n_per = overlap ? 2*ratio : ratio;

    ggml_cuda_pdl_sync();

    for (int k = threadIdx.x; k < n_per; k += blockDim.x) {
        const bool cur = overlap && k >= ratio;
        s_idx[k] = cur ? idxs[n_read + ib*ratio + (k - ratio)] : idxs[ib*ratio + k];
    }
    __syncthreads();

    const int64_t f = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= n_embd_head) {
        return;
    }

    // pass 1: max of the gathered scores for this feature
    float m = -INFINITY;
    for (int k = 0; k < n_per; ++k) {
        const int64_t row = s_idx[k];
        if ((uint64_t) row >= (uint64_t) n_rows) {
            continue; // missing segment: score is -INFINITY
        }
        const int64_t col = (overlap && k >= ratio) ? f + n_embd_head : f;
        m = fmaxf(m, dsv4_compress_src_load(score, n_rows_a, col, row));
    }

    // pass 2: unnormalized softmax weights, weighted sum
    float sum = 0.0f;
    float acc = 0.0f;
    for (int k = 0; k < n_per; ++k) {
        const int64_t row = s_idx[k];
        if ((uint64_t) row >= (uint64_t) n_rows) {
            continue; // missing segment: weight 0, value 0
        }
        const int64_t col = (overlap && k >= ratio) ? f + n_embd_head : f;

        const float w = expf(dsv4_compress_src_load(score, n_rows_a, col, row) - m);
        sum += w;
        acc += w*dsv4_compress_src_load(kv, n_rows_a, col, row);
    }

    dst[f*sd0 + ib*sd2] = sum > 0.0f ? acc/sum : 0.0f;
}

void ggml_cuda_op_dsv4_compress_concat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * kv_cat, const ggml_tensor * score_cat, ggml_tensor * dst) {
    const ggml_tensor * idxs = dst->src[2];

    GGML_ASSERT(dst->src[0] == kv_cat && dst->src[1] == score_cat);
    GGML_ASSERT(kv_cat->op == GGML_OP_CONCAT && score_cat->op == GGML_OP_CONCAT);
    GGML_ASSERT(idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(idxs));

    const ggml_tensor * kv_a    = kv_cat->src[0];
    const ggml_tensor * kv_b    = kv_cat->src[1];
    const ggml_tensor * score_a = score_cat->src[0];
    const ggml_tensor * score_b = score_cat->src[1];

    GGML_ASSERT(kv_a->ne[1] == score_a->ne[1]);
    GGML_ASSERT(kv_cat->ne[1] == score_cat->ne[1]);

    const auto make_src = [](const ggml_tensor * a, const ggml_tensor * b) {
        GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32);
        return dsv4_compress_src {
            (const float *) a->data, (const float *) b->data,
            (int64_t) (a->nb[0] / sizeof(float)), (int64_t) (a->nb[1] / sizeof(float)),
            (int64_t) (b->nb[0] / sizeof(float)), (int64_t) (b->nb[1] / sizeof(float)),
        };
    };

    GGML_TENSOR_LOCALS(size_t, nbd, dst, nb);

    const int     ratio   = ggml_get_op_params_i32(dst, 0);
    const bool    overlap = ggml_get_op_params_i32(dst, 1) != 0;

    const int64_t n_embd_head = dst->ne[0];
    const int64_t n_blocks    = dst->ne[2];
    const int64_t n_rows_a    = kv_a->ne[1];
    const int64_t n_rows      = kv_cat->ne[1];
    const int64_t n_read      = (int64_t) ratio*n_blocks;
    const int64_t n_per_block = overlap ? 2*ratio : ratio;

    GGML_ASSERT(dst->ne[1] == 1);
    GGML_ASSERT(idxs->ne[0] == n_per_block*n_blocks);

    // the launch geometry of ggml_cuda_op_dsv4_compress
    const int block_size = 64;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd_head + block_size - 1)/block_size, n_blocks, 1);
    const size_t shmem = n_per_block*sizeof(int32_t);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, shmem, ctx.stream());

    auto kernel = overlap ? dsv4_compress_concat_f32<true> : dsv4_compress_concat_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            make_src(kv_a, kv_b), make_src(score_a, score_b), (const int32_t *) idxs->data, (float *) dst->data,
            n_embd_head, n_rows_a, n_rows, n_read, ratio,
            (int64_t) (nbd0 / sizeof(float)), (int64_t) (nbd2 / sizeof(float)));
}

// GET_ROWS -> SET_ROWS with the gathered rows going straight to their destination rows, for one or
// two row pairs that share both index vectors (the kv and the score state of a compressor).
// dst[didx[i]] = src[sidx[i]] is a plain F32 copy.
struct dsv4_row_copy_pair {
    const float * src;
    float       * dst;
    int64_t       s1;  // source row stride, in floats
    int64_t       d1;  // destination row stride, in floats
};

struct dsv4_row_copy_args {
    dsv4_row_copy_pair pair[2];
};

static __device__ __forceinline__ int64_t dsv4_row_index(const void * idx, const bool is_i64, const int64_t i) {
    return is_i64 ? ((const int64_t *) idx)[i] : (int64_t) ((const int32_t *) idx)[i];
}

static __global__ void dsv4_row_copy_f32(
        const dsv4_row_copy_args args,
        const void * sidx,
        const void * didx,
        const bool   sidx_i64,
        const bool   didx_i64,
        const int64_t ncols) {
    ggml_cuda_pdl_lc();

    const int64_t i = blockIdx.x;
    const dsv4_row_copy_pair p = blockIdx.y == 0 ? args.pair[0] : args.pair[1];

    ggml_cuda_pdl_sync();

    const int64_t row_src = dsv4_row_index(sidx, sidx_i64, i);
    const int64_t row_dst = dsv4_row_index(didx, didx_i64, i);

    const float * src = p.src + row_src*p.s1;
    float       * dst = p.dst + row_dst*p.d1;

    for (int64_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[col] = src[col];
    }
}

void ggml_cuda_op_dsv4_row_copy(ggml_backend_cuda_context & ctx,
        const ggml_tensor * const * get_rows, const ggml_tensor * const * set_rows, const int n_pairs) {
    GGML_ASSERT(n_pairs == 1 || n_pairs == 2);

    const ggml_tensor * sidx = get_rows[0]->src[1];
    const ggml_tensor * didx = set_rows[0]->src[1]; // ggml_set_rows: src[0] rows, src[1] indices, src[2] destination
    const int64_t n     = get_rows[0]->ne[1];
    const int64_t ncols = get_rows[0]->ne[0];

    dsv4_row_copy_args args = {};
    for (int p = 0; p < n_pairs; ++p) {
        const ggml_tensor * gr = get_rows[p];
        const ggml_tensor * sr = set_rows[p];

        GGML_ASSERT(gr->src[1] == sidx && sr->src[1] == didx && sr->src[0] == gr);
        GGML_ASSERT(gr->ne[0] == ncols && gr->ne[1] == n);
        GGML_ASSERT(gr->src[0]->type == GGML_TYPE_F32 && sr->type == GGML_TYPE_F32);

        args.pair[p].src = (const float *) gr->src[0]->data;
        args.pair[p].dst = (float *) sr->data;
        args.pair[p].s1  = (int64_t) (gr->src[0]->nb[1] / sizeof(float));
        args.pair[p].d1  = (int64_t) (sr->nb[1] / sizeof(float));
    }

    const int block_size = ncols >= 256 ? 256 : 64;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims(n, n_pairs, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_row_copy_f32, launch_params,
            args, sidx->data, didx->data, sidx->type == GGML_TYPE_I64, didx->type == GGML_TYPE_I64, ncols);
}

// Low-rank gated residual of Qwen3.8 hyper-connections (src/models/qwen4exp.cpp build_hc_mix) for decode
// batches, matched in ggml-cuda.cu:
//   pre:  MUL_MAT(down, xn) -> SCALE -> SILU -> MUL_MAT(up, .) -> RESHAPE -> DSV4_HC_PRE (gated)
// The chain is two launches: the down projection with its activation, then the up projection with
// the gated stream mean. Each thread issues all 16-byte loads of its part of a
// weight row before the first use, so a whole row is in flight per block. The dot products accumulate
// in FP32; the elementwise arithmetic is that of scale_f32, the silu op and dsv4_hc_pre_f32.

#define HC_GATED_THREADS 256
#define HC_GATED_NCH     5   // 16-byte chunks of a row per thread and segment (k = 10240 down rows are one segment)

// 8 consecutive weights of a row as one load unit
template <typename T> struct hc_gated_chunk;

template <> struct hc_gated_chunk<half> {
    int4 v = make_int4(0, 0, 0, 0);

    static __device__ __forceinline__ hc_gated_chunk load(const half * row, const int c) {
        hc_gated_chunk r;
        r.v = ((const int4 *) row)[c];
        return r;
    }
    __device__ __forceinline__ void to_float(float * f) const {
        const half2 * h = (const half2 *) &v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 t = __half22float2(h[i]);
            f[2*i + 0] = t.x;
            f[2*i + 1] = t.y;
        }
    }
};

// acc += <f, x[0..7]> in order
static __device__ __forceinline__ float hc_gated_dot8(const float * f, const float4 xa, const float4 xb, float acc) {
    acc = fmaf(f[0], xa.x, acc);
    acc = fmaf(f[1], xa.y, acc);
    acc = fmaf(f[2], xa.z, acc);
    acc = fmaf(f[3], xa.w, acc);
    acc = fmaf(f[4], xb.x, acc);
    acc = fmaf(f[5], xb.y, acc);
    acc = fmaf(f[6], xb.z, acc);
    acc = fmaf(f[7], xb.w, acc);
    return acc;
}

// Sums n values over the width lanes of a group and spreads the n sums over the lanes: the n values halve at every
// xor step while n > 1, the remaining steps sum one value. Lane l ends with sum n_index(l) in v[0].
template <int width, int n>
static __device__ __forceinline__ float hc_gated_reduce_scatter(float * v, const int lane) {
    int m = n;
#pragma unroll
    for (int o = width/2; o > 0; o >>= 1) {
        if (m > 1) {
            m /= 2;
            const bool upper = lane & o;
#pragma unroll
            for (int j = 0; j < m; ++j) {
                const float send = upper ? v[j] : v[j + m];
                const float keep = upper ? v[j + m] : v[j];
                v[j] = keep + __shfl_xor_sync(0xffffffff, send, o, width);
            }
        } else {
            v[0] += __shfl_xor_sync(0xffffffff, v[0], o, width);
        }
    }
    return v[0];
}

template <int width, int n>
static __device__ __forceinline__ int hc_gated_scatter_index(const int lane) {
    int idx = 0;
    int m = n;
#pragma unroll
    for (int o = width/2; o > 0 && m > 1; o >>= 1) {
        m /= 2;
        if (lane & o) {
            idx += m;
        }
    }
    return idx;
}

// the lanes that hold a sum once (the lowest lane of each group of equal sums)
template <int width, int n>
static __device__ __forceinline__ bool hc_gated_scatter_owner(const int lane) {
    return (lane & (width/n - 1)) == 0;
}

static constexpr int hc_gated_pow2(const int n) {
    return n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : 8;
}

// t[it, j] = silu(scale*<w[j, :], x[:, it]> + bias) for J rows j per block
template <typename T, int ncols, int J>
static __global__ void __launch_bounds__(HC_GATED_THREADS) hc_gated_down_silu(
        const T * __restrict__ w, const int64_t stride_w,
        const float * __restrict__ x, const int64_t stride_x,
        float * __restrict__ t,
        const int k, const int m, const float scale, const float bias) {
    constexpr int np = hc_gated_pow2(ncols);
    static_assert(J*np <= 32, "too many sums per block");

    const int tid = threadIdx.x;
    const int j0  = blockIdx.x*J;
    const int k8  = k/8;

    float acc[J*np];
#pragma unroll
    for (int i = 0; i < J*np; ++i) {
        acc[i] = 0.0f;
    }

    for (int c0 = 0; c0 < k8; c0 += HC_GATED_NCH*HC_GATED_THREADS) {
        hc_gated_chunk<T> wc[J][HC_GATED_NCH];
#pragma unroll
        for (int jj = 0; jj < J; ++jj) {
            const T * row = w + (int64_t) min(j0 + jj, m - 1)*stride_w;
#pragma unroll
            for (int ic = 0; ic < HC_GATED_NCH; ++ic) {
                const int c = c0 + ic*HC_GATED_THREADS + tid;
                if (c < k8) {
                    wc[jj][ic] = hc_gated_chunk<T>::load(row, c);
                }
            }
        }
        if (c0 == 0) {
            ggml_cuda_pdl_sync(); // the weights do not depend on the previous kernel
        }
#pragma unroll
        for (int ic = 0; ic < HC_GATED_NCH; ++ic) {
            const int c = c0 + ic*HC_GATED_THREADS + tid;
            if (c >= k8) {
                break;
            }
            float f[J][8];
#pragma unroll
            for (int jj = 0; jj < J; ++jj) {
                wc[jj][ic].to_float(f[jj]);
            }
#pragma unroll
            for (int it = 0; it < ncols; ++it) {
                const float4 * xc = (const float4 *) (x + it*stride_x) + 2*c;
                const float4 xa = xc[0];
                const float4 xb = xc[1];
#pragma unroll
                for (int jj = 0; jj < J; ++jj) {
                    acc[jj*np + it] = hc_gated_dot8(f[jj], xa, xb, acc[jj*np + it]);
                }
            }
        }
    }

    __shared__ float red[J*np][HC_GATED_THREADS/32];
    const int lane = tid % 32;
    const int warp = tid / 32;
    const float s = hc_gated_reduce_scatter<32, J*np>(acc, lane);
    if (hc_gated_scatter_owner<32, J*np>(lane)) {
        red[hc_gated_scatter_index<32, J*np>(lane)][warp] = s;
    }
    __syncthreads();
    ggml_cuda_pdl_lc();

    if (tid < J*np) {
        const int jj = tid / np;
        const int it = tid % np;
        if (it < ncols && j0 + jj < m) {
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < HC_GATED_THREADS/32; ++i) {
                sum += red[tid][i];
            }
            const float v = scale*sum + bias;
            t[it*m + j0 + jj] = v / (1.0f + expf(-v));
        }
    }
}

// dst[col, it] = scale_pre*sum_h x[col, h, it]*sigmoid(<u[h*n_embd + col, :], t[it, :]>); a group of 32 lanes
// handles ncpg columns with 8 lanes per stream h
template <typename T, int ncols, int ncpg>
static __global__ void __launch_bounds__(HC_GATED_THREADS) hc_gated_up_mix(
        const T * __restrict__ u, const int64_t stride_u,
        const float * __restrict__ t,
        const float * __restrict__ x, const int64_t sx0, const int64_t sx1, const int64_t sx2,
        float * __restrict__ dst, const int64_t sd0, const int64_t sd1,
        const int k, const int n_embd, const float scale) {
    extern __shared__ float hc_gated_t[]; // [ncols][k]
    constexpr int np = hc_gated_pow2(ncols);

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int h    = lane / 8;
    const int l8   = lane % 8;
    const int col0 = (blockIdx.x*(HC_GATED_THREADS/32) + tid/32)*ncpg;
    const int k8   = k/8;

    const T * rows[ncpg];
#pragma unroll
    for (int cc = 0; cc < ncpg; ++cc) {
        rows[cc] = u + (int64_t) (h*n_embd + min(col0 + cc, n_embd - 1))*stride_u;
    }

    float acc[ncpg][np];
#pragma unroll
    for (int cc = 0; cc < ncpg; ++cc) {
#pragma unroll
        for (int it = 0; it < np; ++it) {
            acc[cc][it] = 0.0f;
        }
    }

    for (int c0 = 0; c0 < k8; c0 += 8*HC_GATED_NCH) {
        hc_gated_chunk<T> uc[ncpg][HC_GATED_NCH];
#pragma unroll
        for (int cc = 0; cc < ncpg; ++cc) {
#pragma unroll
            for (int ic = 0; ic < HC_GATED_NCH; ++ic) {
                const int c = c0 + 8*ic + l8;
                if (c < k8) {
                    uc[cc][ic] = hc_gated_chunk<T>::load(rows[cc], c);
                }
            }
        }
        if (c0 == 0) {
            ggml_cuda_pdl_sync();
            for (int i = tid; i < ncols*k; i += HC_GATED_THREADS) {
                hc_gated_t[i] = t[i];
            }
            __syncthreads();
        }
#pragma unroll
        for (int ic = 0; ic < HC_GATED_NCH; ++ic) {
            const int c = c0 + 8*ic + l8;
            if (c >= k8) {
                break;
            }
            float f[ncpg][8];
#pragma unroll
            for (int cc = 0; cc < ncpg; ++cc) {
                uc[cc][ic].to_float(f[cc]);
            }
#pragma unroll
            for (int it = 0; it < ncols; ++it) {
                const float4 * tc = (const float4 *) (hc_gated_t + it*k) + 2*c;
                const float4 ta = tc[0];
                const float4 tb = tc[1];
#pragma unroll
                for (int cc = 0; cc < ncpg; ++cc) {
                    acc[cc][it] = hc_gated_dot8(f[cc], ta, tb, acc[cc][it]);
                }
            }
        }
    }
    ggml_cuda_pdl_lc();

    const int  it    = hc_gated_scatter_index<8, np>(l8);
    const bool owner = hc_gated_scatter_owner<8, np>(l8) && it < ncols;
#pragma unroll
    for (int cc = 0; cc < ncpg; ++cc) {
        const int col = col0 + cc;
        const float g = hc_gated_reduce_scatter<8, np>(acc[cc], l8);
        float xv = 0.0f;
        float wv = 0.0f;
        if (owner && col < n_embd) {
            xv = x[col*sx0 + h*sx1 + it*sx2];
            wv = 1.0f / (1.0f + expf(-g));
        }
        float sum = 0.0f;
#pragma unroll
        for (int ih = 0; ih < DSV4_HC; ++ih) {
            sum += __shfl_sync(0xffffffff, xv, 8*ih + l8, 32) * __shfl_sync(0xffffffff, wv, 8*ih + l8, 32);
        }
        if (owner && h == 0 && col < n_embd) {
            dst[col*sd0 + it*sd1] = scale * sum;
        }
    }
}

static bool hc_gated_weight_ok(const ggml_tensor * w) {
    if (w->type != GGML_TYPE_F16) {
        return false;
    }
    return w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == ggml_type_size(w->type) && w->ne[0] % 8 == 0 &&
        w->nb[1] % 16 == 0 && (uintptr_t) w->data % 16 == 0;
}

// a column-major activation [k, ncols] read with 16-byte loads
static bool hc_gated_cols_ok(const ggml_tensor * x, const int64_t k) {
    return x->type == GGML_TYPE_F32 && x->ne[0] == k && x->ne[1] >= 1 && x->ne[1] <= GGML_CUDA_HC_GATED_MAX_TOKENS &&
        x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 && (uintptr_t) x->data % 16 == 0;
}

static bool hc_gated_scale_ok(const ggml_tensor * s, const ggml_tensor * src) {
    return s->op == GGML_OP_SCALE && s->src[0] == src && s->type == GGML_TYPE_F32 && ggml_are_same_shape(s, src) &&
        ggml_is_contiguous(s) && ggml_is_contiguous(src);
}

bool ggml_cuda_hc_gated_pre_supported(const ggml_tensor * mm_down, const ggml_tensor * scale, const ggml_tensor * act,
        const ggml_tensor * mm_up, const ggml_tensor * gate, const ggml_tensor * pre) {
    const ggml_tensor * w  = mm_down->src[0];
    const ggml_tensor * xn = mm_down->src[1];
    const ggml_tensor * u  = mm_up->src[0];
    const ggml_tensor * x  = pre->src[0];

    if (!hc_gated_weight_ok(w) || !hc_gated_weight_ok(u)) {
        return false;
    }
    const int64_t k     = w->ne[0];
    const int64_t m     = w->ne[1];
    const int64_t ncols = xn->ne[1];
    if (!hc_gated_cols_ok(xn, k) || mm_down->type != GGML_TYPE_F32 || mm_down->ne[0] != m || mm_down->ne[1] != ncols ||
            !ggml_is_contiguous(mm_down)) {
        return false;
    }
    if (!hc_gated_scale_ok(scale, mm_down) || act->op != GGML_OP_UNARY || ggml_get_unary_op(act) != GGML_UNARY_OP_SILU ||
            act->src[0] != scale || !ggml_are_same_shape(act, scale) || !ggml_is_contiguous(act) || act->type != GGML_TYPE_F32) {
        return false;
    }
    // the activation stays in shared memory: [ncols][m]
    if (mm_up->src[1] != act || u->ne[0] != m || m % 8 != 0 || ncols*m*(int64_t) sizeof(float) > 32*1024) {
        return false;
    }
    if (pre->op != GGML_OP_DSV4_HC_PRE || ggml_get_op_params_i32(pre, 1) == 0 || pre->src[1] != gate ||
            gate->op != GGML_OP_RESHAPE || gate->src[0] != mm_up) {
        return false;
    }
    const int64_t n_embd = x->ne[0];
    if (x->type != GGML_TYPE_F32 || x->ne[1] != DSV4_HC || x->ne[2] != ncols || x->ne[3] != 1 ||
            u->ne[1] != n_embd*DSV4_HC || mm_up->ne[0] != n_embd*DSV4_HC || mm_up->ne[1] != ncols ||
            !ggml_is_contiguous(mm_up) || !ggml_is_contiguous(gate) || gate->ne[0] != n_embd || gate->ne[1] != DSV4_HC ||
            gate->ne[2] != ncols || pre->type != GGML_TYPE_F32 || pre->ne[0] != n_embd || pre->ne[1] != ncols) {
        return false;
    }
    return true;
}

template <typename T, int ncols>
static void hc_gated_pre_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_down, const ggml_tensor * scale,
        const ggml_tensor * mm_up, ggml_tensor * pre) {
    const ggml_tensor * w  = mm_down->src[0];
    const ggml_tensor * xn = mm_down->src[1];
    const ggml_tensor * u  = mm_up->src[0];
    const ggml_tensor * x  = pre->src[0];

    const int k      = (int) w->ne[0];
    const int m      = (int) w->ne[1];
    const int n_embd = (int) x->ne[0];

    float scale_v;
    float bias_v;
    memcpy(&scale_v, (const float *) scale->op_params + 0, sizeof(float));
    memcpy(&bias_v,  (const float *) scale->op_params + 1, sizeof(float));
    const float scale_pre = ggml_get_op_params_f32(pre, 0);

    ggml_cuda_pool_alloc<float> t(ctx.pool(), (size_t) ncols*m);

    // more tokens: two rows per block of the down projection (fewer reads of the activation) and two columns per
    // lane group of the up projection (fewer reads of t)
    constexpr int J    = ncols <= 2 ? 1 : 2;
    constexpr int ncpg = ncols <= 2 ? 1 : 2;
    const dim3 block(HC_GATED_THREADS, 1, 1);
    {
        const dim3 grid((m + J - 1)/J, 1, 1);
        ggml_cuda_kernel_launch(hc_gated_down_silu<T, ncols, J>, ggml_cuda_kernel_launch_params(grid, block, 0, ctx.stream()),
            (const T *) w->data, (int64_t) (w->nb[1]/sizeof(T)), (const float *) xn->data, (int64_t) (xn->nb[1]/sizeof(float)),
            t.get(), k, m, scale_v, bias_v);
    }
    {
        constexpr int cols_per_block = ncpg*(HC_GATED_THREADS/32);
        const dim3 grid((n_embd + cols_per_block - 1)/cols_per_block, 1, 1);
        const size_t shmem = (size_t) ncols*m*sizeof(float);
        ggml_cuda_kernel_launch(hc_gated_up_mix<T, ncols, ncpg>, ggml_cuda_kernel_launch_params(grid, block, shmem, ctx.stream()),
            (const T *) u->data, (int64_t) (u->nb[1]/sizeof(T)), (const float *) t.get(),
            (const float *) x->data, (int64_t) (x->nb[0]/sizeof(float)), (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (x->nb[2]/sizeof(float)),
            (float *) pre->data, (int64_t) (pre->nb[0]/sizeof(float)), (int64_t) (pre->nb[1]/sizeof(float)),
            m, n_embd, scale_pre);
    }
}

template <typename T>
static void hc_gated_pre_switch_ncols(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_down, const ggml_tensor * scale,
        const ggml_tensor * mm_up, ggml_tensor * pre) {
    switch (mm_down->src[1]->ne[1]) {
        case 1: hc_gated_pre_launch<T, 1>(ctx, mm_down, scale, mm_up, pre); break;
        case 2: hc_gated_pre_launch<T, 2>(ctx, mm_down, scale, mm_up, pre); break;
        case 3: hc_gated_pre_launch<T, 3>(ctx, mm_down, scale, mm_up, pre); break;
        case 4: hc_gated_pre_launch<T, 4>(ctx, mm_down, scale, mm_up, pre); break;
        case 5: hc_gated_pre_launch<T, 5>(ctx, mm_down, scale, mm_up, pre); break;
        case 6: hc_gated_pre_launch<T, 6>(ctx, mm_down, scale, mm_up, pre); break;
        case 7: hc_gated_pre_launch<T, 7>(ctx, mm_down, scale, mm_up, pre); break;
        case 8: hc_gated_pre_launch<T, 8>(ctx, mm_down, scale, mm_up, pre); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_hc_gated_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_down, const ggml_tensor * scale,
        const ggml_tensor * mm_up, ggml_tensor * pre) {
    switch (mm_down->src[0]->type) {
        case GGML_TYPE_F16: hc_gated_pre_switch_ncols<half>(ctx, mm_down, scale, mm_up, pre); break;
        default: GGML_ABORT("unsupported type: %s", ggml_type_name(mm_down->src[0]->type));
    }
}
