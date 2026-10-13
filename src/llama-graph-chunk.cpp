#include "llama-graph-chunk.h"

#include "ggml.h"
#include "llama-ext.h"

#include <algorithm>
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

llm_graph_chunks::llm_graph_chunks(int64_t n_tokens) : n_tokens(n_tokens), c(n_tokens), n(1) {
    const int64_t cc = llm_graph_chunk_tokens();
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

static ggml_tensor * llm_graph_join_range(ggml_context * ctx, const std::vector<ggml_tensor *> & parts, int dim,
                                          size_t b, size_t e) {
    if (e - b == 1) {
        return parts[b];
    }
    const size_t m = b + (e - b)/2;
    return ggml_concat(ctx, llm_graph_join_range(ctx, parts, dim, b, m), llm_graph_join_range(ctx, parts, dim, m, e), dim);
}

ggml_tensor * llm_graph_join(ggml_context * ctx, const std::vector<ggml_tensor *> & parts, int dim) {
    GGML_ASSERT(!parts.empty());
    return llm_graph_join_range(ctx, parts, dim, 0, parts.size());
}
