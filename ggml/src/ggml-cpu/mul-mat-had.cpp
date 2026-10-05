// GGML_OP_MUL_MAT_HAD: y = rot_out * H(W^T H(rot_in * x)), see ggml_mul_mat_had()

#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "vec.h"

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
