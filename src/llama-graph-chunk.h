#pragma once

#include "ggml.h"
#include "llama-graph.h"

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

struct llama_ubatch;

// Masks of a chunk made on the device. An [n_keys, n_tokens] host mask grows with the square of the ubatch and every chunk
// view of it becomes a device copy that lives for the whole graph (see llm_graph_rows_cache), so a wide ubatch would still
// carry the full mask. Instead the graph uploads the token positions and a per-sequence table of key positions, both linear,
// and every layer expands the [n_keys, n] mask of the chunk it is computing; the allocator reuses it for the next chunk.
//
// The mask is causal over positions: key r is visible to token t when tab[r, seq(t)] is finite and pos(r) <= pos(t).
// It covers what a single stream without SWA, ALiBi or 2D positions puts into the host mask (0 or -inf), bit for bit.
// It is on whenever the graph is chunked.
struct llama_cparams;
struct llama_hparams;

// enabled, the graph of this ubatch is chunked, and its causal mask is what llama_kv_cache::set_input_kq_pos covers
// (causal, one stream, no SWA or ALiBi, 2D positions only for tokens); fixed per ubatch shape, so usable in can_reuse
// swa_ok: the cache applies a standard sliding window, which the mask then takes as n_swa
bool llm_graph_chunk_mask_usable(const llama_cparams & cparams, const llama_hparams & hparams, const llama_ubatch & ubatch,
                                 bool swa_ok = false);

struct llm_graph_chunk_pos {
    ggml_tensor * tok_pos = nullptr; // F32 [1, n_tokens] -(pos + 0.5)
    ggml_tensor * tok_seq = nullptr; // I32 [n_tokens]    index of the token's sequence among the ubatch's unique ones

    void build(ggml_context * ctx, int64_t n_tokens);
    void set_input(const llama_ubatch * ubatch) const;
};

// tab: F32 [n_keys, n_seqs_unq], -pos of each key per sequence of the ubatch, -inf where the key is not in the sequence.
// Returns the additive mask [n_keys, n] of type for the token rows [i0, i0 + n): 0 where visible, -inf elsewhere.
// n_swa > 0 also hides the keys n_swa or more positions behind the token (LLAMA_SWA_TYPE_STANDARD).
ggml_tensor * llm_graph_chunk_mask(ggml_context * ctx, llm_graph_rows_cache & rows, const llm_graph_chunk_pos & cp,
                                   ggml_tensor * tab, int64_t i0, int64_t n, ggml_type type, uint32_t n_swa = 0);

// the same with another token key than the position (F32 [1, n_tokens], -(key + 0.5)), e.g. the rank of a token in its
// sequence for pools in cache order
ggml_tensor * llm_graph_chunk_mask(ggml_context * ctx, llm_graph_rows_cache & rows, ggml_tensor * tab, ggml_tensor * tok_key,
                                   ggml_tensor * tok_seq, int64_t i0, int64_t n, ggml_type type, uint32_t n_swa = 0);

// RANMA_CHUNK_MASK_CHECK=1: the inputs check themselves against the host masks they replace at every set_input (slow,
// for validation). A mismatch aborts.
bool llm_graph_chunk_mask_check_enabled();

// a host-memory scratch tensor for those checks (F16 or F32 [ne0, ne1])
struct llm_graph_host_tensor {
    llm_graph_host_tensor(ggml_type type, int64_t ne0, int64_t ne1);
    ~llm_graph_host_tensor();

    ggml_context * ctx = nullptr;
    void         * buf = nullptr; // ggml_backend_buffer_t
    ggml_tensor  * t   = nullptr;
};

// aborts unless host (F16/F32 [n_keys, n_tokens], 0 or -inf) is what llm_graph_chunk_mask makes from tab/tok_key/tok_seq
void llm_graph_chunk_mask_compare(const char * name, const ggml_tensor * host, const ggml_tensor * tab,
                                  const ggml_tensor * tok_key, const ggml_tensor * tok_seq, uint32_t n_swa = 0);

// How a chunk mask input reaches its KV cache from the graph's memory context (rebound by can_reuse): the cell count, the
// position table (set_input_kq_pos) and, for RANMA_CHUNK_MASK_CHECK, the host mask it replaces (set_input_kq_mask).
struct llm_graph_chunk_kv_ops {
    uint32_t (*n_kv)    (const llama_memory_context_i * mctx);
    void     (*set_pos) (const llama_memory_context_i * mctx, ggml_tensor * dst, const llama_ubatch * ubatch);
    void     (*set_mask)(const llama_memory_context_i * mctx, ggml_tensor * dst, const llama_ubatch * ubatch);
};

// the ops of a KV cache context type T that get picks out of the memory context
template <typename T, const T * (*get)(const llama_memory_context_i *)>
llm_graph_chunk_kv_ops llm_graph_chunk_kv_ops_of() {
    return {
        [](const llama_memory_context_i * m) -> uint32_t { return get(m)->get_n_kv(); },
        [](const llama_memory_context_i * m, ggml_tensor * dst, const llama_ubatch * ub) { get(m)->set_input_kq_pos(dst, ub); },
        [](const llama_memory_context_i * m, ggml_tensor * dst, const llama_ubatch * ub) { get(m)->set_input_kq_mask(dst, ub, true); },
    };
}

// The device-side causal mask of one KV cache for a chunked graph, in place of that attention input's kq_mask (which the
// graph then leaves out).
class llm_graph_input_chunk_mask : public llm_graph_input_i {
public:
    llm_graph_input_chunk_mask(const llama_memory_context_i * mctx, llm_graph_chunk_kv_ops ops, uint32_t n_swa) :
        mctx(mctx), ops(ops), n_swa(n_swa) {}
    virtual ~llm_graph_input_chunk_mask() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    // the [n_kv, n] causal mask of the token rows [i0, i0 + n)
    ggml_tensor * mask(ggml_context * ctx, int64_t i0, int64_t n, ggml_type type) {
        return llm_graph_chunk_mask(ctx, rows, pos, kv_pos, i0, n, type, n_swa);
    }

    const llama_memory_context_i * mctx;
    const llm_graph_chunk_kv_ops ops;
    const uint32_t n_swa; // 0, or the standard window of an SWA cache

    ggml_tensor * kv_pos = nullptr; // F32 [n_kv, n_seqs_unq], see llama_kv_cache::set_input_kq_pos
    llm_graph_chunk_pos  pos;
    llm_graph_rows_cache rows;
};

// creates the input (when llm_graph_chunk_mask_usable) and adds it to the graph result; nullptr otherwise
llm_graph_input_chunk_mask * llm_graph_build_inp_chunk_mask(llm_graph_result * res, ggml_context * ctx, ggml_cgraph * gf,
        const llama_cparams & cparams, const llama_hparams & hparams, const llama_ubatch & ubatch,
        const llama_memory_context_i * mctx, llm_graph_chunk_kv_ops ops, uint32_t n_swa = 0);
