#pragma once

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// Token chunks of a wide graph: the attention of a ubatch of n_tokens is computed for at most c query rows at a
// time, so that the [n_kv, n_tokens] intermediates of a sparse selection (scores, masks) are only [n_kv, c] large,
// while the KV writes and the rest of the layer (MoE, recurrent layers) still run over the whole ubatch once.
//
// c comes from the attention chunk of llama_layer_mode_config (tokens, read at each graph build; set by the layer mode). Unset, 0 or c >= n_tokens gives one chunk, the
// unchanged graph. A c below LLM_GRAPH_CHUNK_MIN is raised to it: every chunk adds its own nodes per layer.
#define LLM_GRAPH_CHUNK_MIN 64

int64_t llm_graph_chunk_tokens();

struct llm_graph_chunks {
    explicit llm_graph_chunks(int64_t n_tokens);

    int64_t n_tokens;
    int64_t c;
    int     n;

    int64_t i0(int k) const { return k*c; }
    int64_t nk(int k) const { return k + 1 < n ? c : n_tokens - (int64_t) (n - 1)*c; }
};

// view of the rows [i0, i0 + n) of t along dim; t itself for the full range
ggml_tensor * llm_graph_rows(ggml_context * ctx, ggml_tensor * t, int dim, int64_t i0, int64_t n);

// Row views of host-resident graph inputs (masks, index lists), made once per graph and shared by all layers. The
// scheduler copies every distinct view of an input it reads from another backend as a split input of its own, and split
// inputs are allocated for the whole graph, so per-layer views of an [n_kv, n_tokens] mask would add one full mask per
// layer. This holds for the unchunked graph as well: its 2D reshape of a full input is shared too. A full-range request
// without the reshape gives t itself.
struct llm_graph_rows_cache {
    // flat: reshaped to [ne[0], n] after the view (the other dims of t must be 1)
    ggml_tensor * get(ggml_context * ctx, ggml_tensor * t, int dim, int64_t i0, int64_t n, bool flat);

    struct entry { const ggml_tensor * t; int dim; int64_t i0, n; bool flat; ggml_tensor * view; };
    std::vector<entry> entries;
};

// the chunk results joined along dim by a balanced concat tree; meant for the compact per-token results
// (attention output, selection indices), never for an [n_kv, n_tokens] mask
ggml_tensor * llm_graph_join(ggml_context * ctx, const std::vector<ggml_tensor *> & parts, int dim);
