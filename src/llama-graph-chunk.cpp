#include "llama-graph-chunk.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-ext.h"
#include "llama-kv-cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// the layer mode's decisions for the graphs of the process (llama-ext.h)
static llama_layer_mode_config g_layer_mode_config;

llama_layer_mode_config llama_layer_mode_get_config() {
    return g_layer_mode_config;
}

void llama_layer_mode_set_config(const llama_layer_mode_config & cfg) {
    g_layer_mode_config = cfg;
}

// read at every graph build: the layer mode (common/layer-mode.cpp) sets it between the no_alloc measurements of its
// candidate chunks, before the model it serves loads; it does not change afterwards
int64_t llm_graph_chunk_tokens() {
    const long long v = llama_layer_mode_get_config().attn_chunk;
    return v <= 0 ? (int64_t) 0 : std::max<int64_t>(v, LLM_GRAPH_CHUNK_MIN);
}

int64_t llm_graph_chunk_tokens_dsv4_raw(const llama_hparams & hparams) {
    const long long v = 4*(long long) hparams.indexer_top_k;
    return v <= 0 ? (int64_t) 0 : std::max<int64_t>(v, LLM_GRAPH_CHUNK_MIN);
}

int64_t llm_graph_chunk_tokens_min(int64_t a, int64_t b) {
    if (a <= 0) {
        return std::max<int64_t>(b, 0);
    }
    return b <= 0 ? a : std::min(a, b);
}

llm_graph_chunks::llm_graph_chunks(int64_t n_tokens, int64_t c_req) : n_tokens(n_tokens), c(n_tokens), n(1) {
    const int64_t cc = c_req;
    if (cc > 0 && cc < n_tokens) {
        c = cc;
        n = (int) ((n_tokens + cc - 1)/cc);
    }
}

ggml_tensor * llm_graph_rows(ggml_context * ctx, ggml_tensor * t, int dim, int64_t i0, int64_t n) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS);
    GGML_ASSERT(i0 >= 0 && n > 0 && i0 + n <= t->ne[dim]);
    if (i0 == 0 && n == t->ne[dim]) {
        return t;
    }
    int64_t ne[GGML_MAX_DIMS] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    ne[dim] = n;
    return ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], i0*t->nb[dim]);
}

ggml_tensor * llm_graph_rows_cache::get(ggml_context * ctx, ggml_tensor * t, int dim, int64_t i0, int64_t n, bool flat) {
    if (i0 == 0 && n == t->ne[dim] && !flat) {
        return t;
    }
    for (const auto & e : entries) {
        if (e.t == t && e.dim == dim && e.i0 == i0 && e.n == n && e.flat == flat) {
            return e.view;
        }
    }
    ggml_tensor * v = llm_graph_rows(ctx, t, dim, i0, n);
    if (flat) {
        GGML_ASSERT(ggml_nelements(v) == v->ne[0]*n);
        v = ggml_reshape_2d(ctx, v, v->ne[0], n);
    }
    entries.push_back({ t, dim, i0, n, flat, v });
    return v;
}

llm_graph_join_acc::llm_graph_join_acc(ggml_context * ctx, ggml_cgraph * gf, int dim, int64_t n_total) :
        ctx(ctx), gf(gf), dim(dim), n_total(n_total) {
    GGML_ASSERT(dim >= 0 && dim < GGML_MAX_DIMS && n_total > 0);
}

void llm_graph_join_acc::add(ggml_tensor * part, int64_t i0) {
    const int64_t n = part->ne[dim];
    GGML_ASSERT(i0 == n_added && n > 0 && i0 + n <= n_total && "chunks are joined in row order");
    for (int d = dim + 1; d < GGML_MAX_DIMS; ++d) {
        GGML_ASSERT(part->ne[d] == 1);
    }
    n_added += n;

    if (acc == nullptr) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            ne[d] = part->ne[d];
        }
        ne[dim] = n_total;
    } else {
        for (int d = 0; d < dim; ++d) {
            GGML_ASSERT(part->ne[d] == ne[d]);
        }
    }

    // one chunk holds every row
    if (n == n_total) {
        acc = part;
        return;
    }

    if (!ggml_is_contiguous(part)) {
        part = ggml_cont(ctx, part);
    }

    if (part->type == GGML_TYPE_F32 || part->type == GGML_TYPE_F16) {
        const int64_t row = ggml_nelements(part)/n;
        ggml_tensor * rows = ggml_reshape_2d(ctx, part, row, n);
        if (acc == nullptr && part->type == GGML_TYPE_F32) {
            // the first chunk padded to every row; the padding is written over by the later chunks
            acc = ggml_pad(ctx, rows, 0, (int) (n_total - n), 0, 0);
        } else {
            if (acc == nullptr) {
                // a tensor of every row whose values the chunks write over
                acc = ggml_reshape_2d(ctx, ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float) (row*n_total), 1.0f), part->type),
                        row, n_total);
            }
            ggml_tensor * idx = ggml_cast(ctx, ggml_arange(ctx, (float) i0, (float) (i0 + n), 1.0f), GGML_TYPE_I32);
            acc = ggml_set_rows(ctx, acc, rows, idx);
        }
    } else {
        GGML_ASSERT(part->type == GGML_TYPE_I32);
        if (acc == nullptr) {
            // a tensor of the whole shape whose values the chunks write over
            ggml_tensor * blank = ggml_cast(ctx, ggml_arange(ctx, 0.0f, (float) (ggml_nelements(part)/n*n_total), 1.0f),
                    GGML_TYPE_I32);
            acc = ggml_reshape_4d(ctx, blank, ne[0], ne[1], ne[2], ne[3]);
        }
        const size_t offs = (size_t) i0*acc->nb[dim];
        GGML_ASSERT(offs <= (size_t) INT32_MAX && "ggml_set takes a 32-bit offset");
        acc = ggml_set_inplace(ctx, acc, part, acc->nb[1], acc->nb[2], acc->nb[3], offs);
    }

    // the copy right after the chunk, so its result is free before the next chunk
    ggml_build_forward_expand(gf, acc);
}

ggml_tensor * llm_graph_join_acc::get() const {
    GGML_ASSERT(acc != nullptr && n_added == n_total && "every row joined");
    if (acc->ne[0] == ne[0] && acc->ne[1] == ne[1] && acc->ne[2] == ne[2] && acc->ne[3] == ne[3]) {
        return acc;
    }
    return ggml_reshape_4d(ctx, acc, ne[0], ne[1], ne[2], ne[3]);
}

bool llm_graph_chunk_mask_enabled() {
    static const bool on = [] {
        const char * env = getenv("RANMA_CHUNK_MASK");
        return env == nullptr || atoi(env) != 0;
    }();
    return on;
}

bool llm_graph_chunk_mask_usable(const llama_cparams & cparams, const llama_hparams & hparams, const llama_ubatch & ubatch,
                                 bool swa_ok, int64_t c_req) {
    const bool swa = hparams.swa_type == LLAMA_SWA_TYPE_NONE || (swa_ok && hparams.swa_type == LLAMA_SWA_TYPE_STANDARD);
    if (c_req < 0) {
        c_req = llm_graph_chunk_tokens();
    }
    return llm_graph_chunk_mask_enabled() && llm_graph_chunks(ubatch.n_tokens, c_req).n > 1 &&
        cparams.causal_attn && !hparams.use_alibi && swa &&
        (!ubatch.is_pos_2d() || ubatch.embd == nullptr) && (cparams.kv_unified || ubatch.n_seqs_unq == 1);
}

void llm_graph_chunk_pos::build(ggml_context * ctx, int64_t n_tokens) {
    tok_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_tokens);
    tok_seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(tok_pos);
    ggml_set_input(tok_seq);
    ggml_set_name(tok_pos, "chunk_tok_pos");
    ggml_set_name(tok_seq, "chunk_tok_seq");
}

void llm_graph_chunk_pos::set_input(const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(tok_pos->buffer) && ggml_backend_buffer_is_host(tok_seq->buffer));
    GGML_ASSERT(tok_pos->ne[1] == (int64_t) ubatch->n_tokens && tok_seq->ne[0] == (int64_t) ubatch->n_tokens);

    float   * pos = (float   *) tok_pos->data;
    int32_t * seq = (int32_t *) tok_seq->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        // positions are exact in F32 below 2^24
        GGML_ASSERT(ubatch->pos[i] >= 0 && ubatch->pos[i] < (1 << 24));
        pos[i] = -((float) ubatch->pos[i] + 0.5f);
        seq[i] = ubatch->seq_idx[ubatch->seq_id[i][0]];
        GGML_ASSERT(seq[i] >= 0 && seq[i] < (int32_t) ubatch->n_seqs_unq);
    }
}

ggml_tensor * llm_graph_chunk_mask(ggml_context * ctx, llm_graph_rows_cache & rows, const llm_graph_chunk_pos & cp,
                                   ggml_tensor * tab, int64_t i0, int64_t n, ggml_type type, uint32_t n_swa) {
    return llm_graph_chunk_mask(ctx, rows, tab, cp.tok_pos, cp.tok_seq, i0, n, type, n_swa);
}

ggml_tensor * llm_graph_chunk_mask(ggml_context * ctx, llm_graph_rows_cache & rows, ggml_tensor * tab, ggml_tensor * tok_key,
                                   ggml_tensor * tok_seq, int64_t i0, int64_t n, ggml_type type, uint32_t n_swa) {
    GGML_ASSERT(tab->type == GGML_TYPE_F32 && tok_key->type == GGML_TYPE_F32 && tok_key->ne[0] == 1);

    // [n_keys, n]: -key of each key for the token's sequence, then + key(token) + 0.5, positive where the key is visible
    ggml_tensor * m = ggml_get_rows(ctx, tab, rows.get(ctx, tok_seq, 0, i0, n, false));
    m = ggml_sub_inplace(ctx, m, rows.get(ctx, tok_key, 1, i0, n, false));
    // 1 or 0, then log: 0 or -inf, exactly the values of the host mask
    if (n_swa > 0) {
        // also n_swa - (distance + 0.5) > 0, i.e. the key is less than n_swa positions behind
        ggml_tensor * w = ggml_step_inplace(ctx, ggml_scale_bias(ctx, m, -1.0f, (float) n_swa));
        m = ggml_mul_inplace(ctx, ggml_step(ctx, m), w);
    } else {
        m = ggml_step_inplace(ctx, m);
    }
    m = ggml_log_inplace(ctx, m);
    if (type != GGML_TYPE_F32) {
        m = ggml_cast(ctx, m, type);
    }
    ggml_set_name(m, "chunk_mask");
    return m;
}

bool llm_graph_chunk_mask_check_enabled() {
    static const bool on = [] {
        const char * env = getenv("RANMA_CHUNK_MASK_CHECK");
        return env != nullptr && atoi(env) != 0;
    }();
    return on;
}

llm_graph_host_tensor::llm_graph_host_tensor(ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
    ctx = ggml_init(params);
    t   = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);
    buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(cpu));
    GGML_ASSERT(buf != nullptr);
}

llm_graph_host_tensor::~llm_graph_host_tensor() {
    ggml_backend_buffer_free((ggml_backend_buffer_t) buf);
    ggml_free(ctx);
}

void llm_graph_chunk_mask_compare(const char * name, const ggml_tensor * host, const ggml_tensor * tab,
                                  const ggml_tensor * tok_key, const ggml_tensor * tok_seq, uint32_t n_swa) {
    const int64_t n_keys   = host->ne[0];
    const int64_t n_tokens = ggml_nelements(host)/n_keys;
    GGML_ASSERT(tab->ne[0] == n_keys && tok_key->ne[1] == n_tokens && tok_seq->ne[0] == n_tokens);

    const float   * tb  = (const float   *) tab->data;
    const float   * key = (const float   *) tok_key->data;
    const int32_t * seq = (const int32_t *) tok_seq->data;

    int64_t n_bad = 0;
    int64_t n_vis = 0;
    for (int64_t i = 0; i < n_tokens; ++i) {
        for (int64_t j = 0; j < n_keys; ++j) {
            const float h = host->type == GGML_TYPE_F16 ?
                ggml_fp16_to_fp32(((const ggml_fp16_t *) host->data)[i*n_keys + j]) : ((const float *) host->data)[i*n_keys + j];
            // the device ops: get_rows, sub, step, log
            const float d = tb[seq[i]*n_keys + j] - key[i];
            const bool vis = d > 0.0f && (n_swa == 0 || (float) n_swa - d > 0.0f);
            n_vis += vis;
            const bool ok = vis ? (h == 0.0f && !std::signbit(h)) : (std::isinf(h) && h < 0.0f);
            if (!ok && n_bad++ < 4) {
                fprintf(stderr, "%s: chunk mask check %s: token %lld key %lld host %f device %s\n", __func__, name,
                        (long long) i, (long long) j, h, vis ? "0" : "-inf");
            }
        }
    }
    fprintf(stderr, "%s: chunk mask check %s: %lld x %lld, %lld visible, %lld mismatches\n", __func__, name,
            (long long) n_keys, (long long) n_tokens, (long long) n_vis, (long long) n_bad);
    GGML_ASSERT(n_bad == 0 && "device chunk mask differs from the host mask");
}

void llm_graph_input_chunk_mask::window(int64_t i0, int64_t n, int64_t & j0, int64_t & nj) const {
    j0 = 0;
    nj = kv_pos->ne[0];
    for (const auto & w : windows) {
        if (w.i0 == i0) {
            GGML_ASSERT(w.n == n && w.j0 >= 0 && w.nj > 0 && w.j0 + w.nj <= kv_pos->ne[0]);
            j0 = w.j0;
            nj = w.nj;
            return;
        }
    }
}

std::vector<llm_graph_chunk_window> llm_graph_input_chunk_mask::make_windows(int64_t n_tokens) const {
    std::vector<llm_graph_chunk_window> res;
    if (ops.window == nullptr) {
        return res;
    }

    const int64_t n_kv = ops.n_kv(mctx);
    const llm_graph_chunks ch = chunks(n_tokens);
    for (int k = 0; k < ch.n; ++k) {
        llm_graph_chunk_window w = { ch.i0(k), ch.nk(k), 0, n_kv };
        ops.window(mctx, w.i0, w.n, w.j0, w.nj);
        if (w.j0 < 0 || w.nj <= 0 || w.j0 + w.nj > n_kv) {
            w.j0 = 0;
            w.nj = n_kv;
        }
        res.push_back(w);
    }

    return res;
}

void llm_graph_input_chunk_mask::set_input(const llama_ubatch * ubatch) {
    ops.set_pos(mctx, kv_pos, ubatch);
    pos.set_input(ubatch);

    if (ops.check != nullptr && !windows.empty()) {
        ops.check(mctx, ubatch, windows);
    }

    if (llm_graph_chunk_mask_check_enabled()) {
        llm_graph_host_tensor kq(GGML_TYPE_F32, kv_pos->ne[0], ubatch->n_tokens);
        ops.set_mask(mctx, kq.t, ubatch);
        llm_graph_chunk_mask_compare("kq_mask", kq.t, kv_pos, pos.tok_pos, pos.tok_seq, n_swa);
    }
}

bool llm_graph_input_chunk_mask::can_reuse(const llm_graph_params & params) {
    mctx = params.mctx;

    bool res = llm_graph_chunk_mask_usable(params.cparams, params.hparams, params.ubatch, n_swa > 0, c_req);

    res &= kv_pos->ne[0] == ops.n_kv(mctx);
    res &= kv_pos->ne[1] == params.ubatch.n_seqs_unq;
    res &= pos.tok_pos->ne[1] == params.ubatch.n_tokens;

    // the key views of every chunk are baked into the graph
    if (res && ops.window != nullptr) {
        res &= make_windows(params.ubatch.n_tokens) == windows;
    }

    return res;
}

llm_graph_input_chunk_mask * llm_graph_build_inp_chunk_mask(llm_graph_result * res, ggml_context * ctx, ggml_cgraph * gf,
        const llama_cparams & cparams, const llama_hparams & hparams, const llama_ubatch & ubatch,
        const llama_memory_context_i * mctx, llm_graph_chunk_kv_ops ops, uint32_t n_swa, int64_t c_req) {
    if (c_req < 0) {
        c_req = llm_graph_chunk_tokens();
    }
    if (!llm_graph_chunk_mask_usable(cparams, hparams, ubatch, n_swa > 0, c_req)) {
        return nullptr;
    }

    auto inp = std::make_unique<llm_graph_input_chunk_mask>(mctx, ops, n_swa, c_req);

    inp->kv_pos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ops.n_kv(mctx), ubatch.n_seqs_unq);
    ggml_set_input(inp->kv_pos);
    ggml_set_name(inp->kv_pos, "chunk_kv_pos");
    inp->pos.build(ctx, ubatch.n_tokens);
    inp->windows = inp->make_windows(ubatch.n_tokens);

    // set_input fills them, so they stay allocated in every graph
    ggml_build_forward_expand(gf, inp->kv_pos);
    ggml_build_forward_expand(gf, inp->pos.tok_pos);
    ggml_build_forward_expand(gf, inp->pos.tok_seq);

    return (llm_graph_input_chunk_mask *) res->add_input(std::move(inp));
}
