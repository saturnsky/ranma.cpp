// GGML_OP_MUL_MAT_HAD: y = rot_out * H(W^T H(rot_in * x)), see ggml_mul_mat_had()

#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "vec.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// in-place normalized Sylvester Hadamard of n values (n a power of 2)
static void mmh_hadamard(float * x, int64_t n) {
    const float scale = 1.0f / sqrtf((float) n);
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= scale;
    }
    for (int64_t len = 1; len < n; len <<= 1) {
        for (int64_t i = 0; i < n; i += 2*len) {
            for (int64_t j = i; j < i + len; ++j) {
                const float u = x[j];
                const float v = x[j + len];
                x[j]       = u + v;
                x[j + len] = u - v;
            }
        }
    }
}

static float mmh_load(const ggml_tensor * t, const char * p) {
    return t->type == GGML_TYPE_F16 ? GGML_CPU_FP16_TO_FP32(*(const ggml_fp16_t *) p) : *(const float *) p;
}

// expert / batch index of each dst row, and where its input row is
struct mmh_geometry {
    const ggml_tensor * dst;
    const ggml_tensor * w;
    const ggml_tensor * x;
    const ggml_tensor * ids;

    void decode(int64_t r, int64_t & i1, int64_t & i2, int64_t & i3) const {
        i1 = r % dst->ne[1];
        i2 = (r / dst->ne[1]) % dst->ne[2];
        i3 = r / (dst->ne[1]*dst->ne[2]);
    }

    int64_t expert(int64_t r) const {
        int64_t i1, i2, i3;
        decode(r, i1, i2, i3);
        if (ids) {
            return *(const int32_t *) ((const char *) ids->data + i1*ids->nb[0] + i2*ids->nb[1]);
        }
        return i2 / (x->ne[2] / w->ne[2]);
    }

    const float * input(int64_t r) const {
        int64_t i1, i2, i3;
        decode(r, i1, i2, i3);
        if (ids) {
            return (const float *) ((const char *) x->data + (i1 % x->ne[1])*x->nb[1] + i2*x->nb[2]);
        }
        return (const float *) ((const char *) x->data + i1*x->nb[1] + i2*x->nb[2] + i3*x->nb[3]);
    }

    float * output(int64_t r) const {
        int64_t i1, i2, i3;
        decode(r, i1, i2, i3);
        return (float *) ((char *) dst->data + i1*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]);
    }
};

// work buffer: expert row offsets | rows sorted by expert | H(rot_in * x) per row | per-thread weight block
static size_t mmh_thread_size(int64_t had, int64_t k) {
    return GGML_PAD(had*k*sizeof(float) + had*k*sizeof(ggml_fp16_t) + had*sizeof(float), CACHE_LINE_SIZE);
}

size_t ggml_cpu_mul_mat_had_wsize(const struct ggml_tensor * node, int n_tasks) {
    const int64_t had    = ggml_get_op_params_i32(node, 0);
    const int64_t k      = node->src[0]->ne[0];
    const int64_t n_rows = ggml_nrows(node);
    const int64_t n_exp  = node->src[0]->ne[2];
    return GGML_PAD((n_exp + 1 + n_rows)*sizeof(int64_t), CACHE_LINE_SIZE) + GGML_PAD(n_rows*k*sizeof(float), CACHE_LINE_SIZE)
         + n_tasks*mmh_thread_size(had, k) + CACHE_LINE_SIZE;
}

void ggml_compute_forward_mul_mat_had(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * w       = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * rot_in  = dst->src[3];
    const ggml_tensor * rot_out = dst->src[4];

    const int64_t had    = ggml_get_op_params_i32(dst, 0);
    const int64_t k      = w->ne[0];
    const int64_t n      = w->ne[1];
    const int64_t n_exp  = w->ne[2];
    const int64_t n_rows = ggml_nrows(dst);
    const int64_t n_blk  = n / had;

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float));

    const mmh_geometry geo = { dst, w, x, ids };

    char * wdata = (char *) params->wdata;
    int64_t * offsets = (int64_t *) wdata;
    int64_t * rows    = offsets + n_exp + 1;
    float   * xh      = (float *) (wdata + GGML_PAD((n_exp + 1 + n_rows)*sizeof(int64_t), CACHE_LINE_SIZE));
    char    * scratch = (char *) xh + GGML_PAD(n_rows*k*sizeof(float), CACHE_LINE_SIZE) + params->ith*mmh_thread_size(had, k);
    GGML_ASSERT(params->wsize >= ggml_cpu_mul_mat_had_wsize(dst, params->nth) - CACHE_LINE_SIZE);

    // rows sorted by expert, in row order
    if (params->ith == 0) {
        memset(offsets, 0, (n_exp + 1)*sizeof(int64_t));
        for (int64_t r = 0; r < n_rows; ++r) {
            const int64_t e = geo.expert(r);
            GGML_ASSERT(e >= 0 && e < n_exp);
            offsets[e + 1]++;
        }
        for (int64_t e = 0; e < n_exp; ++e) {
            offsets[e + 1] += offsets[e];
        }
        for (int64_t r = 0; r < n_rows; ++r) {
            rows[offsets[geo.expert(r)]++] = r;
        }
        for (int64_t e = n_exp; e > 0; --e) {
            offsets[e] = offsets[e - 1];
        }
        offsets[0] = 0;
    }

    // H(rot_in * x) once per row
    for (int64_t r = params->ith; r < n_rows; r += params->nth) {
        const int64_t e   = geo.expert(r);
        const float * src = geo.input(r);
        const char  * rin = (const char *) rot_in->data + e*rot_in->nb[1];
        float * dsth = xh + r*k;
        for (int64_t i = 0; i < k; ++i) {
            dsth[i] = src[i] * mmh_load(rot_in, rin + i*rot_in->nb[0]);
        }
        for (int64_t i = 0; i < k; i += had) {
            mmh_hadamard(dsth + i, had);
        }
    }

    ggml_barrier(params->threadpool);

    float       * wblk = (float *) scratch;
    ggml_fp16_t * wraw = (ggml_fp16_t *) (wblk + had*k);
    float       * acc  = (float *) (wraw + had*k);

    // one task = one block of `had` output rows of one expert, for every row routed to it
    for (int64_t task = params->ith; task < n_exp*n_blk; task += params->nth) {
        const int64_t e   = task / n_blk;
        const int64_t blk = task % n_blk;
        if (offsets[e] == offsets[e + 1]) {
            continue;
        }

        const char * wsrc = (const char *) w->data + e*w->nb[2] + blk*had*w->nb[1];
        if (ggml_is_exl3(w->type)) {
            ggml_exl3_decode_group(w->type, wsrc, k, wraw);
            ggml_cpu_fp16_to_fp32(wraw, wblk, had*k);
        } else {
            for (int64_t j = 0; j < had; ++j) {
                const char * row = wsrc + j*w->nb[1];
                if (w->type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) row, wblk + j*k, k);
                } else {
                    memcpy(wblk + j*k, row, k*sizeof(float));
                }
            }
        }

        const char * rout = (const char *) rot_out->data + e*rot_out->nb[1] + blk*had*rot_out->nb[0];
        for (int64_t ri = offsets[e]; ri < offsets[e + 1]; ++ri) {
            const int64_t r = rows[ri];
            for (int64_t j = 0; j < had; ++j) {
                ggml_vec_dot_f32((int) k, acc + j, 0, wblk + j*k, 0, xh + r*k, 0, 1);
            }
            mmh_hadamard(acc, had);
            float * out = geo.output(r) + blk*had;
            for (int64_t j = 0; j < had; ++j) {
                out[j] = acc[j] * mmh_load(rot_out, rout + j*rot_out->nb[0]);
            }
        }
    }
}

// GGML_OP_MOE_FFN_HAD: reference implementation, one token at a time (see ggml_moe_ffn_had() and ggml_moe_ffn_lora())

// y = rot_out * H(W_e^T H(rot_in * x)) for one row, with the weight and rotations of the MUL_MAT_HAD node d
static void mmh_moe_row(const ggml_tensor * d, int64_t e, const float * x, float * y, float * xh, float * wblk,
        ggml_fp16_t * wraw, float * acc) {
    const ggml_tensor * w       = d->src[0];
    const ggml_tensor * rot_in  = d->src[3];
    const ggml_tensor * rot_out = d->src[4];
    const int64_t had = ggml_get_op_params_i32(d, 0);
    const int64_t k   = w->ne[0];
    const int64_t n   = w->ne[1];

    const char * rin = (const char *) rot_in->data + e*rot_in->nb[1];
    for (int64_t i = 0; i < k; ++i) {
        xh[i] = x[i] * mmh_load(rot_in, rin + i*rot_in->nb[0]);
    }
    for (int64_t i = 0; i < k; i += had) {
        mmh_hadamard(xh + i, had);
    }
    const char * rout = (const char *) rot_out->data + e*rot_out->nb[1];
    for (int64_t blk = 0; blk < n/had; ++blk) {
        const char * wsrc = (const char *) w->data + e*w->nb[2] + blk*had*w->nb[1];
        if (ggml_is_exl3(w->type)) {
            ggml_exl3_decode_group(w->type, wsrc, k, wraw);
            ggml_cpu_fp16_to_fp32(wraw, wblk, had*k);
        } else {
            for (int64_t j = 0; j < had; ++j) {
                const char * row = wsrc + j*w->nb[1];
                if (w->type == GGML_TYPE_F16) {
                    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) row, wblk + j*k, k);
                } else {
                    memcpy(wblk + j*k, row, k*sizeof(float));
                }
            }
        }
        for (int64_t j = 0; j < had; ++j) {
            ggml_vec_dot_f32((int) k, acc + j, 0, wblk + j*k, 0, xh, 0, 1);
        }
        mmh_hadamard(acc, had);
        for (int64_t j = 0; j < had; ++j) {
            y[blk*had + j] = acc[j] * mmh_load(rot_out, rout + (blk*had + j)*rot_out->nb[0]);
        }
    }
}

// y += scale * B_e^T (A_e^T v) for each LoRA delta of the MUL_MAT_HAD node d (its src[5] chain), in chain order;
// z holds the largest rank
static void mmh_moe_lora(const ggml_tensor * d, int64_t e, const float * v, float * y, float * z) {
    for (const ggml_tensor * l = d->src[5]; l != nullptr; l = l->src[2]) {
        const float scale = ggml_get_op_params_f32(l, 0);
        if (scale == 0.0f) {
            continue;
        }
        const ggml_tensor * a = l->src[0];
        const ggml_tensor * b = l->src[1];
        const int64_t k = a->ne[0];
        const int64_t r = a->ne[1];
        const int64_t n = b->ne[1];
        for (int64_t j = 0; j < r; ++j) {
            const char * col = (const char *) a->data + j*a->nb[1] + e*a->nb[2];
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += mmh_load(a, col + i*a->nb[0]) * v[i];
            }
            z[j] = sum;
        }
        for (int64_t i = 0; i < n; ++i) {
            const char * row = (const char *) b->data + i*b->nb[1] + e*b->nb[2];
            float sum = 0.0f;
            for (int64_t j = 0; j < r; ++j) {
                sum += mmh_load(b, row + j*b->nb[0]) * z[j];
            }
            y[i] += scale*sum;
        }
    }
}

static int64_t mmh_moe_lora_max_rank(const ggml_tensor * node) {
    int64_t r = 1;
    for (int i = 3; i <= 5; ++i) {
        for (const ggml_tensor * l = node->src[i]->src[5]; l != nullptr; l = l->src[2]) {
            r = std::max(r, l->src[0]->ne[1]);
        }
    }
    return r;
}

// per thread: xh, the decoded weight block (F32 and raw F16), acc, up, gate, down output, the LoRA rank values
static size_t mmh_moe_thread_size(const ggml_tensor * node) {
    const int64_t n_embd = node->ne[0];
    const int64_t n_ff   = node->src[3]->src[0]->ne[1];
    const int64_t kmax   = std::max(n_embd, n_ff);
    int64_t had = 0;
    for (int i = 3; i <= 5; ++i) {
        had = std::max<int64_t>(had, ggml_get_op_params_i32(node->src[i], 0));
    }
    return GGML_PAD((kmax + had*kmax + had + 2*n_ff + n_embd + mmh_moe_lora_max_rank(node))*sizeof(float) +
        had*kmax*sizeof(ggml_fp16_t), CACHE_LINE_SIZE);
}

size_t ggml_cpu_moe_ffn_had_wsize(const struct ggml_tensor * node, int n_tasks) {
    return n_tasks*mmh_moe_thread_size(node) + CACHE_LINE_SIZE;
}

void ggml_compute_forward_moe_ffn_had(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * ids     = dst->src[1];
    const ggml_tensor * weights = dst->src[2];
    const ggml_tensor * up      = dst->src[3];
    const ggml_tensor * gate    = dst->src[4];
    const ggml_tensor * down    = dst->src[5];

    const ggml_glu_op glu   = (ggml_glu_op) ggml_get_op_params_i32(dst, 0);
    const float       limit = ggml_get_op_params_f32(dst, 1);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_ff     = up->src[0]->ne[1];
    const int64_t n_exp    = up->src[0]->ne[2];
    const int64_t kmax     = std::max(n_embd, n_ff);
    int64_t had = 0;
    for (int i = 3; i <= 5; ++i) {
        had = std::max<int64_t>(had, ggml_get_op_params_i32(dst->src[i], 0));
    }

    GGML_ASSERT(params->wsize >= ggml_cpu_moe_ffn_had_wsize(dst, params->nth) - CACHE_LINE_SIZE);
    char  * scratch = (char *) params->wdata + params->ith*mmh_moe_thread_size(dst);
    float * xh   = (float *) scratch;
    float * wblk = xh + kmax;
    float * acc  = wblk + had*kmax;
    float * vu   = acc + had;
    float * vg   = vu + n_ff;
    float * vy   = vg + n_ff;
    float * lz   = vy + n_embd;
    ggml_fp16_t * wraw = (ggml_fp16_t *) (lz + mmh_moe_lora_max_rank(dst));

    for (int64_t t = params->ith; t < n_tokens; t += params->nth) {
        const float * xt  = (const float *) ((const char *) x->data + t*x->nb[1]);
        float       * out = (float *) ((char *) dst->data + t*dst->nb[1]);
        for (int64_t i = 0; i < n_embd; ++i) {
            out[i] = 0.0f;
        }
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + k*ids->nb[0] + t*ids->nb[1]);
            GGML_ASSERT(e >= 0 && e < n_exp);
            const float w = *(const float *) ((const char *) weights->data + k*weights->nb[1] + t*weights->nb[2]);
            mmh_moe_row(up,   e, xt, vu, xh, wblk, wraw, acc);
            mmh_moe_lora(up,  e, xt, vu, lz);
            mmh_moe_row(gate, e, xt, vg, xh, wblk, wraw, acc);
            mmh_moe_lora(gate, e, xt, vg, lz);
            for (int64_t i = 0; i < n_ff; ++i) {
                float g = vg[i];
                float u = vu[i];
                if (glu == GGML_GLU_OP_SWIGLU_CLAMP) {
                    g = std::min(g, limit);
                    u = std::max(std::min(u, limit), -limit);
                }
                vu[i] = g/(1.0f + expf(-g)) * u;
            }
            mmh_moe_row(down, e, vu, vy, xh, wblk, wraw, acc);
            mmh_moe_lora(down, e, vu, vy, lz);
            for (int64_t i = 0; i < n_embd; ++i) {
                out[i] += w*vy[i];
            }
        }
    }
}
