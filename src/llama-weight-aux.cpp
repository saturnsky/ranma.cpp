#include "llama-weight-aux.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include "gguf.h"

#include <algorithm>
#include <stdexcept>

ggml_tensor * llama_weight_aux_mul_mat(
        ggml_context               * ctx,
        const llama_weight_aux_map * waux,
        ggml_tensor                * w,
        ggml_tensor                * cur,
        ggml_tensor                * ids) {
    if (!ggml_is_exl3(w->type)) {
        return nullptr;
    }

    const ggml_tensor * src = w->view_src ? w->view_src : w;
    const auto it = waux ? waux->find(src) : llama_weight_aux_map::const_iterator();
    if (waux == nullptr || it == waux->end()) {
        GGML_ABORT("EXL3 weight '%s' has no rot_in / rot_out, it must be a weight created by the model loader", w->name);
    }
    const llama_weight_aux & aux = it->second;

    ggml_tensor * rot_in  = aux.rot_in;
    ggml_tensor * rot_out = aux.rot_out;
    int64_t       n       = aux.n;

    if (w->view_src) {
        // whole 128-row groups of the weight, for one expert or a range of experts
        const size_t  offs = w->view_offs;
        const int64_t e0   = offs / src->nb[2];
        const int64_t r0   = (offs % src->nb[2]) / src->nb[1];
        if (w->ne[0] != src->ne[0] || w->nb[1] != src->nb[1] || (w->ne[2] > 1 && w->nb[2] != src->nb[2]) ||
                (offs % src->nb[2]) % (128*src->nb[1]) != 0 || r0 + w->ne[1] > src->ne[1] || e0 + w->ne[2] > src->ne[2]) {
            GGML_ABORT("view '%s' of EXL3 weight '%s' does not take whole 128-row groups", w->name, src->name);
        }
        rot_in  = ggml_view_2d(ctx, rot_in,  rot_in->ne[0], w->ne[2], rot_in->nb[1],  e0*rot_in->nb[1]);
        rot_out = ggml_view_2d(ctx, rot_out, w->ne[1],      w->ne[2], rot_out->nb[1], e0*rot_out->nb[1] + r0*rot_out->nb[0]);
        n = std::min<int64_t>(w->ne[1], aux.n - r0);
        GGML_ASSERT(n > 0);
    }

    // EXL3 stores dims padded to 128 with zero weights: pad the input with zeros and drop the padded outputs
    if (cur->ne[0] != w->ne[0]) {
        GGML_ASSERT(cur->ne[0] == aux.k);
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx, cur);
        }
        cur = ggml_pad(ctx, cur, (int) (w->ne[0] - aux.k), 0, 0, 0);
    } else if (cur->nb[0] != ggml_element_size(cur)) {
        cur = ggml_cont(ctx, cur);
    }

    ggml_tensor * res = ggml_mul_mat_had(ctx, w, cur, rot_in, rot_out, ids, 128);

    if (n < w->ne[1]) {
        res = ggml_view_4d(ctx, res, n, res->ne[1], res->ne[2], res->ne[3], res->nb[1], res->nb[2], res->nb[3], 0);
        res = ggml_cont(ctx, res);
    }

    return res;
}

ggml_tensor * llama_weight_aux_get_rows(
        ggml_context * ctx,
        ggml_tensor  * table,
        ggml_tensor  * bias,
        ggml_tensor  * rows,
        int64_t        n_heads) {
    ggml_tensor * res = ggml_get_rows(ctx, table, rows);
    if (!ggml_is_exl3_row(table->type)) {
        GGML_ASSERT(bias == nullptr);
        return res;
    }

    const int64_t dim = table->ne[0];
    GGML_ASSERT(bias != nullptr && bias->ne[0] == dim && bias->ne[1] == n_heads && ggml_nrows(bias) == n_heads);
    GGML_ASSERT(ggml_nelements(rows) % n_heads == 0);
    const int64_t n = ggml_nelements(rows) / n_heads;

    // ExLlamaV3 ngram_dequant: fp16(codebook * scale + bias[head]), the sum in f32
    // get_rows gives codebook * scale in f32 (exact), F16 -> F32 of the bias is exact, the add rounds once in f32
    res = ggml_reshape_3d(ctx, res, dim, n_heads, n);
    res = ggml_add(ctx, res, bias->type == GGML_TYPE_F32 ? bias : ggml_cast(ctx, bias, GGML_TYPE_F32));
    res = ggml_cast(ctx, ggml_cast(ctx, res, GGML_TYPE_F16), GGML_TYPE_F32);

    return ggml_reshape_2d(ctx, res, dim, n_heads * n);
}

ggml_tensor * llama_weight_aux_test_op(ggml_context * ctx, ggml_tensor * w, int n_ids) {
    ggml_tensor * rot_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, w->ne[0], w->ne[2]);
    ggml_tensor * rot_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, w->ne[1], w->ne[2]);
    if (n_ids > 0) {
        ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], n_ids, 512);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_ids, 512);
        return ggml_mul_mat_had(ctx, w, b, rot_in, rot_out, ids, 128);
    }
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]);
    return ggml_mul_mat_had(ctx, w, b, rot_in, rot_out, nullptr, 128);
}

static std::string llama_weight_aux_base(const std::string & name) {
    static const std::string suffix = ".weight";
    if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error(format("EXL3 tensor '%s' is not a .weight tensor", name.c_str()));
    }
    return name.substr(0, name.size() - suffix.size());
}

void llama_model_loader::create_weight_aux(
        ggml_context * ctx, ggml_tensor * w, const std::string & name, const std::initializer_list<int64_t> & ne, bool duplicated) {
    const std::string base = llama_weight_aux_base(name);

    const ggml_tensor * meta = get_tensor_meta(name.c_str());
    if (meta && (meta->ne[0] != w->ne[0] || meta->ne[1] != w->ne[1])) {
        throw std::runtime_error(format("EXL3 tensor '%s' cannot be reshaped from %s to %s",
                name.c_str(), llama_format_tensor_shape(meta).c_str(), llama_format_tensor_shape(w).c_str()));
    }
    check_weight_aux(w);

    llama_weight_aux aux;
    aux.k = ne.size() > 0 ? ne.begin()[0] : w->ne[0];
    aux.n = ne.size() > 1 ? ne.begin()[1] : w->ne[1];

    const struct {
        const char  *  suffix;
        int64_t        len;
        ggml_tensor ** dst;
    } rots[2] = {
        { "rot_in",  w->ne[0], &aux.rot_in  },
        { "rot_out", w->ne[1], &aux.rot_out },
    };
    for (const auto & r : rots) {
        const std::string rname = base + "." + r.suffix;
        const int64_t len = r.len;

        ggml_tensor * t = nullptr;
        if (files.empty()) {
            const int64_t tid = gguf_find_tensor(metadata, rname.c_str());
            t = ggml_new_tensor_2d(ctx, tid >= 0 ? gguf_get_tensor_type(metadata, tid) : GGML_TYPE_F16, len, w->ne[2]);
        } else {
            const ggml_tensor * rmeta = get_tensor_meta(rname.c_str());
            if (rmeta == nullptr) {
                throw std::runtime_error(format("EXL3 weight '%s' needs tensor '%s', which is not in the model", name.c_str(), rname.c_str()));
            }
            if ((rmeta->type != GGML_TYPE_F16 && rmeta->type != GGML_TYPE_F32) ||
                    rmeta->ne[0] != len || rmeta->ne[1] != w->ne[2] || rmeta->ne[2] != 1 || rmeta->ne[3] != 1) {
                const std::vector<int64_t> want = w->ne[2] > 1 ? std::vector<int64_t>{ len, w->ne[2] } : std::vector<int64_t>{ len };
                throw std::runtime_error(format("EXL3 tensor '%s' is %s %s, expected f16 or f32 %s", rname.c_str(),
                        ggml_type_name(rmeta->type), llama_format_tensor_shape(rmeta).c_str(), llama_format_tensor_shape(want).c_str()));
            }
            if (duplicated) {
                t = ggml_get_tensor(ctx, rname.c_str());
            }
            if (t == nullptr) {
                t = ggml_dup_tensor(ctx, rmeta);
                if (duplicated) {
                    size_data += ggml_nbytes(rmeta);
                } else {
                    n_created++;
                }
            }
        }
        ggml_set_name(t, rname.c_str());
        *r.dst = t;
    }

    waux[w] = aux;
}

void llama_model_loader::check_weight_aux(const ggml_tensor * w) const {
    if (ggml_is_exl3(w->type) && (w->ne[0] % 128 != 0 || w->ne[1] % 128 != 0 || w->ne[3] != 1)) {
        throw std::runtime_error(format("EXL3 tensor '%s' has shape %s, the first two dims must be multiples of 128",
                ggml_get_name(w), llama_format_tensor_shape(w).c_str()));
    }
}

void llama_model_loader::skip_weight_aux(const ggml_tensor * w, const std::string & name) {
    if (!ggml_is_exl3(w->type)) {
        return;
    }
    const std::string base = llama_weight_aux_base(name);
    for (const char * suffix : { "rot_in", "rot_out" }) {
        const ggml_tensor * rmeta = get_tensor_meta((base + "." + suffix).c_str());
        if (rmeta) {
            size_data -= ggml_nbytes(rmeta);
            n_created++;
        }
    }
}
