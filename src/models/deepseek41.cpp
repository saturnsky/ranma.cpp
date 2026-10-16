#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-graph-chunk.h"
#include "llama-layer-exec.h"
#include "llama-prefetch.h"

#include "ggml-account.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// DeepSeek-V4.1.
//
// Shares almost everything with DeepSeek-V4: the hyper-connection stream, the MoE, the latent
// attention and the compressed KV cache. Three things differ.
//
// 1. The hyper-connection coefficients lag by one sublayer. A sublayer computes the mix that the
//    NEXT one consumes, so attention uses what the previous layer's FFN produced. V4 computes and
//    consumes in the same sublayer.
// 2. There is no learned hyper-connection head. Nothing is left to collapse the copies with in V4,
//    which is why it needs one; here the last layer's FFN mix is still unused and does the job.
//    These two facts explain each other, and the file ships no output_hc_* tensors.
// 3. The engram tables: n-gram keyed lookups added into the stream at a few layers.
//
// The sparse attention also differs. V4 compresses KV on every layer at one of two fixed ratios;
// V4.1 compresses on a few source layers and the layers after each source read the same rows, and
// it derives index keys from that shared latent rather than from a second compressor. See
// build_attention_v41().
//
// Two level top-k. The candidate source layer scores blocks of candidate_block_size compressed
// positions by their best index score and keeps the best candidate_topk_blocks, always including
// the block with the query's newest position. The index sources after it rank positions only
// inside those blocks. Until the compressed length passes candidate_topk_blocks *
// candidate_block_size (16384 tokens on the ratio 1 stream) every block is kept, so nothing is
// built there. The carry holds one entry per block, not per position. See build_candidate_keep().

// ranma: the tail prefill (RANMA_DSV41_TAIL_DIAG). When a ubatch of one sequence asks for the logits of its last row
// alone, the layers from il_first on compute only the rows that row still depends on through the sliding window. With
// a window of W rows and the last layer L, the query rows of layer l are O_L = 1 and O_l = K_(l+1), and its input and
// raw KV rows are K_l = O_l + W - 1 = W + (L - l)(W - 1). Rows are tails of the ubatch, counted from its end. il_first
// is the last layer that has to see every row (the last KV source, whose compressor projects every row, or the last
// engram layer); it reads all rows for that and computes the tail like the layers after it. The layers before it run
// over the whole ubatch, so that nothing a later ubatch or a later layer reads is missing, except the raw KV of the
// tail layers outside their K rows, which the cache records (see dsv41_tail_plan_make).
//
// The output need not be in the ubatch. When the batch has one output, its last token (llama_kv_cache_dsv4_context::
// get_batch_out), every ubatch of it computes the rows of the tail that fall into it: with the output d rows past the
// end of the ubatch, layer l has K_l - d input rows and O_l - d query rows, clamped to [0, n]. The rows before them that
// an earlier ubatch of the batch computed are in the raw cache already. A layer whose query rows all lie past the ubatch
// only writes the raw KV of its rows (and the KV source on il_first), the layers after it are not built, and a ubatch
// without the output has no logits.
//
// A batch without outputs plans the same rows, as if its last token were the output, and builds no head: its call end
// is the state the next call continues from, which needs the raw KV of the last W rows of every layer and nothing else
// of the tail. The last layer still computes its one query row, which nothing reads.
//
// mode EXACT: a tail layer makes and writes the raw KV of its K rows; the query, the indexer, the attention, the
// hyper-connections and the FFN run over its O rows. The math is that of the whole graph for the output row and the raw
// KV of the last W rows of every layer, but the matrix products see other row counts, so the result is not bit exact.
//
// mode DIAG, the falsifier of the design: the attention of a tail layer runs over its K rows and only the hyper-
// connection post mix and the FFN are narrowed to the O rows. The queries of the first W - 1 of the K rows read raw
// cells this ubatch did not write at this layer; those rows are dropped before the FFN, so the O rows do not see them.
struct dsv41_tail_rows {
    int64_t n_k = 0; // input and raw KV rows
    int64_t n_o = 0; // rows handed to the next layer
};

// ranma: how the graph chunks its attention (the attention chunk, the raw chunk): the chunk size of its device-side
// masks, 0 for an unchunked graph, and the ops of the raw cache that give the raw window of a chunk of query rows
struct dsv41_tail_chunking {
    int64_t c = 0;
    llm_graph_chunk_kv_ops ops = {};
};

struct dsv41_tail_plan {
    enum mode_t : int32_t { OFF = 0, DIAG = 1, EXACT = 2 };

    int32_t mode      = OFF;
    bool    requested = false;   // the env asked for a tail mode
    int32_t il_first  = -1;
    int64_t n_tokens  = 0;
    int64_t n_past    = 0;       // the output is this many rows past the last row of the ubatch (0: the last row)
    bool    out       = true;    // the batch asks for the output; false: the tail ends a call without outputs
    int64_t c         = 0;       // the query chunk size of a chunked graph, 0 unchunked
    std::vector<dsv41_tail_rows> rows; // [n_layer], 0 before il_first
    // chunked graph: per layer from il_first, the chunks of its query rows in ubatch rows [i0, i0 + n) with the raw
    // columns [j0, j0 + nj) they read (all raw cells without the raw chunk); baked into the graph's views
    std::vector<std::vector<llm_graph_chunk_window>> q_chunks;
    const char * reason = nullptr;     // why the plan is off
    // RANMA_DSV41_TAIL_POISON: the raw cells a tail layer leaves unwritten in this ubatch are written with this value
    bool  poison_on = false;
    float poison    = 0.0f;

    bool active() const { return mode != OFF; }

    // the ubatch computes the output row (the last row of the tail, with or without logits)
    bool has_output() const { return n_past == 0; }

    // the graph has a head and logits: the ubatch computes the output row and the batch asks for it
    bool has_logits() const { return n_past == 0 && out; }

    // the query rows of layer il: the attention of DIAG runs over all its input rows
    int64_t n_q(int il) const { return mode == EXACT ? rows[il].n_o : rows[il].n_k; }

    bool same_graph(const dsv41_tail_plan & o) const {
        if (!active() || !o.active()) {
            return active() == o.active();
        }
        if (mode != o.mode || il_first != o.il_first || n_tokens != o.n_tokens || has_output() != o.has_output() ||
                out != o.out || c != o.c || rows.size() != o.rows.size()) {
            return false;
        }
        if (poison_on != o.poison_on || (poison_on && !(poison == o.poison || (std::isnan(poison) && std::isnan(o.poison))))) {
            return false;
        }
        for (size_t il = 0; il < rows.size(); ++il) {
            if (rows[il].n_k != o.rows[il].n_k || rows[il].n_o != o.rows[il].n_o) {
                return false;
            }
        }
        return q_chunks == o.q_chunks;
    }
};

// env, read at each graph build: RANMA_DSV41_TAIL_DIAG (default 0).
// The tail is always on: against the whole graph the output differs by rounding only. Every layer before il_first is bit
// equal, and the first tensors that differ are the query and raw KV projections of il_first, whose inputs are bit equal
// and whose matrix products see fewer rows (ds41-exact-tail/RESULT.ko.md).
static int32_t dsv41_tail_mode_env() {
    const char * diag = getenv("RANMA_DSV41_TAIL_DIAG");
    if (diag != nullptr && atoi(diag) != 0) {
        return dsv41_tail_plan::DIAG;
    }
    return dsv41_tail_plan::EXACT;
}

// env: RANMA_DSV41_TAIL_POISON (diagnostic, default 0): 1 or nan writes NaN, any other number that value, into the raw
// cells of the ubatch rows a tail layer leaves unwritten. A read of such a cell that its mask does not hide then shows
// in the output. A masked read can show too where a kernel multiplies a masked score by the value (0 * NaN), so a NaN
// result is checked again with a finite value.
static bool dsv41_tail_poison_env(float & value) {
    const char * v = getenv("RANMA_DSV41_TAIL_POISON");
    if (v == nullptr || *v == '\0' || strcmp(v, "0") == 0) {
        return false;
    }
    if (strcmp(v, "1") == 0 || strcmp(v, "nan") == 0 || strcmp(v, "NAN") == 0) {
        value = NAN;
    } else {
        value = strtof(v, nullptr);
    }
    return true;
}

// The plan of one ubatch. Anything outside what the tail covers runs the whole graph:
//   - one sequence of token rows at consecutive positions, with one position per row;
//   - no embeddings or layer inputs extracted, and a standard sliding window on every tail layer;
//   - EXACT in a batch whose one output is its last token, or that has no outputs (llama_kv_cache_dsv4_context::
//     get_batch_out): the output is in this ubatch or later, a ubatch without logits has no backend sampler, and the
//     tail leaves some row of the ubatch out;
//   - otherwise one output, the last row of the ubatch, and a ubatch longer than K_(il_first) (DIAG always).
// A chunked graph (ch.c > 0) chunks the query rows of every tail layer by the same size, the first chunk at the first
// query row; each chunk keeps its rows of the ubatch, so the masks, positions, compressed keys and raw windows are those
// of the same rows in the whole graph. mctx gives the raw windows and the cell count.
// The ubatches of the batch before the first tail ubatch, and the batches before it, run in full.
//
// The raw KV of a tail layer is written for its K rows only. Every layer writes at least its last W rows before the
// output, so decoding on from the output reads written cells. llm_graph_input_dsv41_tail::set_input records the rows
// left out in the cache, which refuses to continue, remove back or save a state where that would read them.
static dsv41_tail_plan dsv41_tail_plan_make(
        const llama_model_deepseek41 & pmodel,
        const llama_cparams & cparams,
        const llama_ubatch & ubatch,
        int64_t n_outputs,
        bool has_samplers,
        const llama_memory_context_i * mctx,
        const dsv41_tail_chunking & ch) {
    const auto & hparams = pmodel.hparams;

    dsv41_tail_plan plan;
    plan.n_tokens = ubatch.n_tokens;

    const int32_t mode = dsv41_tail_mode_env();
    plan.requested = mode != dsv41_tail_plan::OFF;

    auto off = [&](const char * reason) {
        plan.mode     = dsv41_tail_plan::OFF;
        plan.il_first = -1;
        plan.c        = 0;
        plan.reason   = reason;
        plan.rows.clear();
        plan.q_chunks.clear();
        return plan;
    };

    if (mode == dsv41_tail_plan::OFF) {
        return off("not requested");
    }
    if (cparams.embeddings || cparams.embeddings_nextn) {
        return off("embeddings");
    }
    for (bool b : cparams.embeddings_layer_inp) {
        if (b) {
            return off("layer inputs extracted");
        }
    }

    const int64_t n = ubatch.n_tokens;

    if (ubatch.token == nullptr || ubatch.embd != nullptr || ubatch.embd_state != nullptr || ubatch.is_mixed()) {
        return off("not a token batch");
    }
    if (ubatch.n_pos != 1 || ubatch.pos == nullptr) {
        return off("not one position per row");
    }
    if (ubatch.n_seqs_unq != 1 || ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr) {
        return off("not one sequence");
    }
    for (int64_t i = 0; i < n; ++i) {
        if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i][0] != ubatch.seq_id_unq[0]) {
            return off("not one sequence");
        }
        if (ubatch.pos[i] != ubatch.pos[0] + (llama_pos) i) {
            return off("positions not consecutive");
        }
    }
    // EXACT: the one output of the batch, its last token, may lie past this ubatch; a batch without outputs ends at its
    // last token all the same
    llama_seq_id out_seq = -1;
    llama_pos    out_pos = -1;
    bool         out_has = false;
    const bool batch_out = mode == dsv41_tail_plan::EXACT && mctx != nullptr &&
        static_cast<const llama_kv_cache_dsv4_context *>(mctx)->get_batch_out(out_seq, out_pos, out_has) &&
        out_seq == ubatch.seq_id_unq[0] && out_pos >= ubatch.pos[n - 1];

    plan.n_past = batch_out ? out_pos - ubatch.pos[n - 1] : 0;
    plan.out    = batch_out ? out_has : true;

    if (plan.n_past == 0 && plan.out) {
        if (n_outputs != 1 || ubatch.output == nullptr || !ubatch.output[n - 1]) {
            return off("output not the last row alone");
        }
    } else if (plan.n_past == 0) {
        if (n_outputs != 0) {
            return off("outputs in a batch that asked for none");
        }
        if (has_samplers) {
            return off("backend samplers on a ubatch without the output");
        }
    } else {
        if (n_outputs != 0) {
            return off("output not the last token of the batch");
        }
        if (has_samplers) {
            return off("backend samplers on a ubatch without the output");
        }
    }

    const int32_t n_layer = (int32_t) hparams.n_layer();
    const int64_t n_swa   = hparams.n_swa;

    if (hparams.swa_type != LLAMA_SWA_TYPE_STANDARD || n_swa <= 0) {
        return off("no standard sliding window");
    }

    // the last layer that reads every row
    int32_t il_first = 0;
    for (int32_t il = 0; il < n_layer; ++il) {
        if (hparams.dsv41_is_kv_source(il) || pmodel.engram_index(il) >= 0) {
            il_first = il;
        }
    }
    for (int32_t il = il_first; il < n_layer; ++il) {
        if (!hparams.is_swa(il)) {
            return off("a tail layer has no sliding window");
        }
    }

    plan.rows.assign(n_layer, dsv41_tail_rows{});
    for (int32_t il = n_layer - 1; il >= il_first; --il) {
        auto & r = plan.rows[il];
        r.n_o = il == n_layer - 1 ? 1 : plan.rows[il + 1].n_k;
        r.n_k = r.n_o + n_swa - 1;
    }

    if (!batch_out) {
        // a ubatch no longer than the tail gains nothing at its first layer; it runs in full, explicitly
        if (n <= plan.rows[il_first].n_k) {
            return off("ubatch not longer than the tail");
        }
    } else {
        // the rows of the tail in this ubatch, the output n_past rows past its end
        bool narrowed = false;
        for (int32_t il = il_first; il < n_layer; ++il) {
            auto & r = plan.rows[il];
            r.n_k = std::clamp<int64_t>(r.n_k - plan.n_past, 0, n);
            r.n_o = std::clamp<int64_t>(r.n_o - plan.n_past, 0, n);
            narrowed = narrowed || r.n_k < n || r.n_o < n;
        }
        if (!narrowed) {
            return off("the tail covers the ubatch");
        }
    }

    plan.mode      = mode;
    plan.il_first  = il_first;
    plan.reason    = nullptr;
    plan.poison_on = dsv41_tail_poison_env(plan.poison);

    if (ch.c > 0) {
        GGML_ASSERT(mctx != nullptr);

        const int64_t n_kv = ch.ops.n_kv(mctx);

        plan.c = ch.c;
        plan.q_chunks.assign(n_layer, {});
        for (int32_t il = il_first; il < n_layer; ++il) {
            const int64_t n_q = plan.n_q(il);
            const int64_t q0  = n - n_q;
            if (n_q == 0) {
                break;
            }

            // one chunk where the query rows fit in one (always on the last layer)
            const llm_graph_chunks chunks(n_q, ch.c);
            for (int k = 0; k < chunks.n; ++k) {
                llm_graph_chunk_window w = { q0 + chunks.i0(k), chunks.nk(k), 0, n_kv };
                if (ch.ops.window != nullptr) {
                    ch.ops.window(mctx, w.i0, w.n, w.j0, w.nj);
                    // as llm_graph_input_chunk_mask::make_windows
                    if (w.j0 < 0 || w.nj <= 0 || w.j0 + w.nj > n_kv) {
                        w.j0 = 0;
                        w.nj = n_kv;
                    }
                }
                plan.q_chunks[il].push_back(w);
            }
        }
    }

    return plan;
}

static void dsv41_tail_log(const dsv41_tail_plan & plan) {
    if (!plan.requested) {
        return;
    }

    std::string msg;
    if (plan.active()) {
        int64_t sum_k = 0;
        int64_t sum_o = 0;
        for (const auto & r : plan.rows) {
            sum_k += r.n_k;
            sum_o += r.n_o;
        }
        msg = format("%s: %" PRId64 " rows, tail from layer %d (K %" PRId64 ", O %" PRId64 "), sum K %" PRId64 ", sum O %" PRId64,
                plan.mode == dsv41_tail_plan::DIAG ? "diag" : "exact", plan.n_tokens, plan.il_first,
                plan.rows[plan.il_first].n_k, plan.rows[plan.il_first].n_o, sum_k, sum_o);
        if (!plan.has_output()) {
            msg += format(", output %" PRId64 " rows on", plan.n_past);
        }
        if (!plan.out) {
            msg += ", no output";
        }
        if (plan.poison_on) {
            msg += format(", poison %g", plan.poison);
        }
        if (plan.c > 0) {
            size_t n_chunks = 0;
            for (const auto & qc : plan.q_chunks) {
                n_chunks += qc.size();
            }
            msg += format(", query chunks of %" PRId64 " (%zu in all)", plan.c, n_chunks);
        }
    } else {
        msg = format("off for %" PRId64 " rows: %s", plan.n_tokens, plan.reason);
    }

    // once per change, a graph is rebuilt for every ubatch that cannot reuse the last one
    static std::string last;
    if (msg != last) {
        LLAMA_LOG_INFO("%s: %s\n", __func__, msg.c_str());
        last = msg;
    }
}

// ranma: keeps a graph of one tail plan from being reused for a ubatch that plans differently (another output row,
// another length class, other query chunks or raw windows); every tensor the plan narrows is a view of an input that
// set_input refreshes
class llm_graph_input_dsv41_tail : public llm_graph_input_i {
public:
    llm_graph_input_dsv41_tail(const llama_model_deepseek41 & pmodel, dsv41_tail_plan plan,
                               const llama_memory_context_i * mctx, dsv41_tail_chunking chunking) :
        pmodel(pmodel), plan(std::move(plan)), mctx(mctx), chunking(chunking) {}
    virtual ~llm_graph_input_dsv41_tail() = default;

    // the raw rows the graph leaves unwritten go to the cache record, before the graph runs: an interrupted ubatch
    // counts as written by the plan, which only leaves more rows out than a whole one would.
    // RANMA_DSV4_RAW_NKV_CHECK: the raw windows of the query chunks against the host mask, as the chunk mask input
    // checks those of the ubatch chunks
    void set_input(const llama_ubatch * ubatch) override {
        if (!plan.active()) {
            return;
        }

        const auto * mctx_dsv4 = static_cast<const llama_kv_cache_dsv4_context *>(mctx);
        const int64_t   n     = ubatch->n_tokens;
        const llama_pos p_end = ubatch->pos[n - 1];
        for (int32_t il = plan.il_first; il < (int32_t) plan.rows.size(); ++il) {
            const int64_t n_k = plan.rows[il].n_k;
            if (n_k < n) {
                mctx_dsv4->set_raw_written(ubatch->seq_id_unq[0], il, p_end + 1 - (llama_pos) n_k);
            }
        }

        if (chunking.ops.window == nullptr || chunking.ops.check == nullptr) {
            return;
        }
        std::vector<llm_graph_chunk_window> windows;
        for (const auto & qc : plan.q_chunks) {
            windows.insert(windows.end(), qc.begin(), qc.end());
        }
        chunking.ops.check(mctx, ubatch, windows);
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = params.mctx;
        return dsv41_tail_plan_make(pmodel, params.cparams, params.ubatch, params.n_outputs, !params.samplers.empty(), mctx,
                chunking).same_graph(plan);
    }

    const llama_model_deepseek41 & pmodel;
    const dsv41_tail_plan plan;
    const llama_memory_context_i * mctx;
    const dsv41_tail_chunking chunking;
};

// ranma: the carried top-k (and candidate keep) narrowed to the last n of the query rows it holds; the query rows of a
// tail layer are a tail of those of every layer before it
static dsv41_top_k_carry dsv41_carry_tail(ggml_context * ctx, const dsv41_top_k_carry & c, int64_t n) {
    dsv41_top_k_carry res = c;
    if (res.top_k != nullptr) {
        GGML_ASSERT(res.top_k->ne[3] == 1 && n <= res.top_k->ne[1]);
        res.top_k = llm_graph_rows(ctx, res.top_k, 1, res.top_k->ne[1] - n, n);
    }
    if (res.cand_keep != nullptr) {
        GGML_ASSERT(res.cand_keep->ne[3] == 1 && n <= res.cand_keep->ne[2]);
        res.cand_keep = llm_graph_rows(ctx, res.cand_keep, 2, res.cand_keep->ne[2] - n, n);
    }
    return res;
}

// mean over the hyper-connection copies; deepseek4.cpp keeps its own copy of this
static ggml_tensor * dsv41_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx, acc, ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx, acc, 1.0f/hc);
}

int llama_model_deepseek41::engram_index(int il) const {
    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] == (uint32_t) il) {
            return (int) e;
        }
    }
    return -1;
}

// ranma: the layer-major prefill (llama-layer-exec.h) runs the layers before the first tail layer layer by layer: the
// tail prefill never narrows their rows, whatever ubatch of the batch the output lies in
int llama_dsv41_layer_exec_layers(const llama_model & model) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);
    const int32_t n_layer = (int32_t) pmodel.hparams.n_layer();
    if (dsv41_tail_mode_env() == dsv41_tail_plan::OFF) {
        return n_layer;
    }
    // as dsv41_tail_plan_make: the last layer that reads every row
    int32_t il_first = 0;
    for (int32_t il = 0; il < n_layer; ++il) {
        if (pmodel.hparams.dsv41_is_kv_source(il) || pmodel.engram_index(il) >= 0) {
            il_first = il;
        }
    }
    return il_first;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    ml.get_arr_n(LLM_KV_ENGRAM_LAYER_IDS, engram_n_layer);
    if (engram_n_layer == 0 || engram_n_layer > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("DeepSeek-V4.1 engram layer count %u is out of range", engram_n_layer));
    }
    ml.get_arr(LLM_KV_ENGRAM_LAYER_IDS, hparams.engram_layer_ids);

    // A file without the candidate keys runs without the two level mask. That matches the
    // reference while the index sources see at most candidate_topk_blocks * candidate_block_size
    // compressed rows. Layers 20 and up compress at ratio 1, so a row is a token and the limit is
    // 16384 tokens for the released model, and only there.
    uint32_t candidate_source = 0;
    if (ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, candidate_source, false)) {
        hparams.candidate_source_layer = (int32_t) candidate_source;
    }
    if (hparams.candidate_source_layer >= 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOPK_BLOCKS, hparams.candidate_topk_blocks);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,  hparams.candidate_block_size);

        if (hparams.candidate_topk_blocks == 0 || hparams.candidate_block_size == 0) {
            throw std::runtime_error("DeepSeek-V4.1 candidate mask needs a nonzero block count and block size");
        }
        if ((uint32_t) hparams.candidate_source_layer >= hparams.n_layer()) {
            throw std::runtime_error(format("candidate source layer %d is out of range", hparams.candidate_source_layer));
        }
    } else {
        LLAMA_LOG_WARN("%s: no candidate mask keys in the file, so the two level top-k is off; "
                "results can differ from the reference beyond 16384 tokens of context (ratio 1 stream)\n", __func__);
    }

    ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     hparams.engram_n_head);
    ml.get_key(LLM_KV_ENGRAM_KEY_LENGTH,     hparams.engram_key_length);
    ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, hparams.engram_max_ngram_size);
    ml.get_key(LLM_KV_ENGRAM_PAD_ID,         engram_pad_id);

    if (hparams.engram_n_head == 0 || hparams.engram_max_ngram_size < 2) {
        throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head and a 2-gram");
    }

    for (uint32_t e = 0; e < engram_n_layer; ++e) {
        if (hparams.engram_layer_ids[e] >= hparams.n_layer()) {
            throw std::runtime_error(format("engram layer %u is out of range", hparams.engram_layer_ids[e]));
        }
    }

    ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS, engram_multipliers);
    ml.get_arr(LLM_KV_ENGRAM_PRIMES,      engram_primes);
    ml.get_arr(LLM_KV_ENGRAM_OFFSETS,     engram_offsets);
    ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,   engram_token_map);

    // the hash indexes straight into these, so a short array would read past the end
    const size_t n_bucket = (size_t) (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;

    if (engram_multipliers.size() != (size_t) engram_n_layer * hparams.engram_max_ngram_size) {
        throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (engram_primes.size() != (size_t) engram_n_layer * n_bucket ||
        engram_offsets.size() != engram_primes.size()) {
        throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (uint64_t p : engram_primes) {
        if (p == 0) {
            throw std::runtime_error("engram prime of zero would divide by zero in the hash");
        }
    }
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head   = hparams.n_embd_head_k();
    const int64_t o_groups      = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank   = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult       = hparams.dsv4_hc_mult;
    const int64_t hc_dim        = hc_mult * n_embd;
    const int64_t hc_mix_dim    = (2 + hc_mult) * hc_mult;
    const int64_t n_embd_indexer = hparams.indexer_head_size;

    if ((size_t) n_vocab > engram_token_map.size()) {
        throw std::runtime_error(format("engram token map has %zu entries, too few for %" PRId64 " tokens",
                                        engram_token_map.size(), n_vocab));
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups),
        // so reshape at load and keep the graph free of it
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // Only the KV source layers carry a compressor, and only those with a ratio above 1 pool
        // with a gate, so both are optional rather than keyed off the ratio the way V4 does it.
        layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, TENSOR_NOT_REQUIRED);
        layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);

        // An index source scores queries against shared index keys. Only a layer that also
        // compresses its own KV builds those keys; the rest read what an earlier layer published.
        layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, TENSOR_NOT_REQUIRED);
        layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", i), {n_embd_head, n_embd_indexer}, TENSOR_NOT_REQUIRED);
        layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_embd_indexer}, TENSOR_NOT_REQUIRED);

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);

        const int eg = engram_index(i);
        if (eg >= 0) {
            const int64_t n_cols = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
            const int64_t key_len = hparams.engram_key_length;

            // The table has hundreds of millions of rows and is far too large to hold in memory,
            // but each token only touches n_cols of them, so read those rows on demand.
            // It is one tensor of the model, or the pieces of the table set of the model
            const std::string embd_name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i).str();
            const llama_table_layout layout = llama_table_layout_get(ml, tn(LLM_TENSOR_ENGRAM_EMBD, i).str());
            const bool pieces = layout.parts > 0;
            int64_t n_rows    = 0;
            int64_t part_rows = 0;
            if (pieces) {
                n_rows    = layout.rows;
                part_rows = layout.part_rows;
            } else {
                const auto * embd_w = ml.get_weight(embd_name.c_str());
                if (embd_w == nullptr) {
                    throw std::runtime_error(format("%s is missing", embd_name.c_str()));
                }
                n_rows = embd_w->tensor->ne[1];
            }

            // a row index is a bucket offset plus a hash, so the last bucket has to end inside
            uint64_t max_row = 0;
            for (int64_t b = 0; b < n_cols; ++b) {
                const size_t k = (size_t) eg*n_cols + b;
                max_row = std::max(max_row, engram_offsets[k] + engram_primes[k]);
            }
            if ((int64_t) max_row > n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the engram buckets (%" PRIu64 ")",
                                                embd_name.c_str(), n_rows, max_row));
            }

            layer.engram_table = llama_table_pieces_create(ml, format("engram layer %d table", i),
                    layout.parts, n_rows, part_rows, key_len,
                    [&](uint32_t k) { return tn(LLM_TENSOR_ENGRAM_EMBD, ("p" + std::to_string(k) + ".weight").c_str(), i).str(); },
                    [&](uint32_t k, int64_t rows, int64_t dim) {
                        const std::string sfx = pieces ? "p" + std::to_string(k) + ".weight" : "weight";
                        return create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, sfx.c_str(), i), {dim, rows}, TENSOR_READ_LAZY);
                    });
            layer.engram_wkv  = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV,  "weight", i), {n_cols * key_len, n_embd * (hc_mult + 1)}, 0);
            layer.engram_q    = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,    "weight", i), {n_embd, hc_mult}, 0);
            layer.engram_k    = create_tensor(tn(LLM_TENSOR_ENGRAM_K,    "weight", i), {n_embd, hc_mult}, 0);
        }

    }

    // Work out which layer publishes the stream each layer reads. Only a source carries a
    // compressor, only an index key owner carries indexer_attn_k, and only an index source
    // carries indexer_attn_q_b, so the file itself says which layer plays which role.
    hparams.dsv41_kv_source.fill(-1);
    hparams.dsv41_index_key_source.fill(-1);
    hparams.dsv41_topk_source.fill(-1);

    int32_t last_kv_source    = -1;
    int32_t last_key_owner    = -1;
    int32_t last_index_source = -1;

    for (int i = 0; i < n_layer; ++i) {
        const auto & layer = layers[i];

        if (layer.attn_comp_wkv)    { last_kv_source    = i; }
        if (layer.indexer_attn_k)   { last_key_owner    = i; }
        if (layer.indexer_attn_q_b) { last_index_source = i; }

        if (hparams.dsv4_compress_ratios[i] == 0) {
            // pure sliding window, no compressed stream to read
            continue;
        }

        if (last_kv_source < 0 || last_key_owner < 0 || last_index_source < 0) {
            throw std::runtime_error(format("layer %d reads a compressed stream before any layer publishes one", i));
        }

        // the row layout of a stream follows the ratio it was compressed at, so a reader that
        // disagrees with its source would index into rows that stand for different positions
        if (hparams.dsv4_compress_ratios[i] != hparams.dsv4_compress_ratios[last_kv_source]) {
            throw std::runtime_error(format("layer %d compresses at ratio %u but reads layer %d, compressed at %u",
                                            i, hparams.dsv4_compress_ratios[i],
                                            last_kv_source, hparams.dsv4_compress_ratios[last_kv_source]));
        }

        hparams.dsv41_kv_source[i]        = last_kv_source;
        hparams.dsv41_index_key_source[i] = last_key_owner;
        hparams.dsv41_topk_source[i]      = last_index_source;
    }

    // the mask is built from an index source's own score
    if (hparams.candidate_source_layer >= 0 && !hparams.dsv41_is_index_source(hparams.candidate_source_layer)) {
        throw std::runtime_error(format("candidate source layer %d is not an index source", hparams.candidate_source_layer));
    }

    // a compressor with no gate only makes sense where there is nothing to pool
    for (int i = 0; i < n_layer; ++i) {
        if (hparams.dsv41_is_kv_source(i) && !layers[i].attn_comp_wgate && hparams.dsv4_compress_ratios[i] != 1) {
            throw std::runtime_error(format("layer %d compresses %u tokens per row but has no pooling gate",
                                            i, hparams.dsv4_compress_ratios[i]));
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash: each token gathers n_cols rows of this layer's table.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// The hash runs host-side because ggml has no 64 bit integers and no xor. Look-back stops at the
// start of the sequence, and the compressed token map folds case and accents together first.
class llm_graph_input_engram : public llm_graph_input_i {
public:
    llm_graph_input_engram(const llama_model_deepseek41 & pmodel,
                           const llama_kv_cache_dsv4_raw_context * mctx,
                           int eg,
                           const llama_table_pieces * table,
                           enum llama_ple_prefetch prefetch) : pmodel(pmodel), mctx(mctx), eg(eg), table(table), prefetch(prefetch) {}
    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx)->get_raw();
        const int64_t n_cols = (pmodel.hparams.engram_max_ngram_size - 1) * pmodel.hparams.engram_n_head;
        return rows->ne[0] == (int64_t) table->size() * n_cols * params.ubatch.n_tokens;
    }

    // per piece of the table, the rows it holds as local row numbers (the others read its row 0), see llama_table_pieces_split
    ggml_tensor * rows = nullptr;   // I32 [n_pieces * n_cols * n_tokens]
    ggml_tensor * mask = nullptr;   // F32 [1, n_pieces * n_cols * n_tokens], 1 where the piece holds the row; null for one piece

    const llama_model_deepseek41 & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_dsv4_raw_context * mctx;

    // which engram layer this is, so the right multipliers and buckets are used
    const int eg;

    // this layer's lazily mapped table and the --ple-prefetch mode: the rows are handed to the OS before the gather
    const llama_table_pieces * table;
    const enum llama_ple_prefetch prefetch;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

void llm_graph_input_engram::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const bool account = ggml_account_enabled();
    const bool timing  = llama_prefetch_timing_enabled();
    const int64_t t_start = account || timing ? ggml_time_us() : 0;

    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_gram   = hp.engram_max_ngram_size;
    const int64_t n_heads  = hp.engram_n_head;
    const int64_t n_cols   = (n_gram - 1) * n_heads;
    const int64_t n_prev   = n_gram - 1;

    const uint64_t * mult = pmodel.engram_multipliers.data() + (size_t) eg*n_gram;
    const uint64_t * prime = pmodel.engram_primes.data()  + (size_t) eg*n_cols;
    const uint64_t * offset = pmodel.engram_offsets.data() + (size_t) eg*n_cols;

    // an image arrives as an embd batch, so ubatch->token is null; the reference gives those
    // positions no engram contribution at all, which the padding token stands in for here
    const int32_t pad = (int32_t) pmodel.engram_pad_id;
    auto map_of = [&](llama_token t) -> uint64_t {
        if (t < 0 || (size_t) t >= pmodel.engram_token_map.size()) {
            return (uint64_t) pad;
        }
        return (uint64_t) pmodel.engram_token_map[t];
    };

    std::vector<int32_t> idx(n_cols * n_tokens);

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "engram n-gram lookups do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch
    mctx->get_prev_tokens(*ubatch, n_prev, prev);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // look-back stops at the start of the sequence; everything from there on reads as padding
        std::vector<uint64_t> ctx(n_gram);
        ctx[0] = ubatch->token ? map_of(ubatch->token[i]) : (uint64_t) pad;
        bool blocked = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = blocked ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            blocked = blocked || t < 0;
            ctx[s] = blocked ? (uint64_t) pad : map_of(t);
        }

        // compressed ids stay under 2^17 and the multipliers under 2^37, so no product overflows
        uint64_t rolling = ctx[0] * mult[0];
        for (int64_t s = 1; s < n_gram; ++s) {
            rolling ^= ctx[s] * mult[s];

            for (int64_t h = 0; h < n_heads; ++h) {
                const int64_t b = (s - 1)*n_heads + h;
                idx[i*n_cols + b] = (int32_t) (rolling % prime[b] + offset[b]);
            }
        }
    }

    const int64_t t_hashed = account || timing ? ggml_time_us() : 0;
    llama_prefetch_stats stats;   // LLAMA_PREFETCH_TIMING
    if (n_tokens <= 1 && pmodel.can_prefetch.count(table->parts[0])) {
        // handed to the resident worker (a copy of the rows); the scheduler waits for it before the first CPU split
        llama_prefetch_decode_submit(table->parts, table->part_rows, idx, prefetch, (int) pmodel.hparams.engram_layer_ids[eg], timing);
    }
    if (account) {
        // logical rows: one per index, whatever the number of pieces
        ggml_account_add(GGML_ACCOUNT_ENGRAM_HASH_MS, (t_hashed - t_start)/1000.0);
        if (n_tokens <= 1) {
            ggml_account_add(GGML_ACCOUNT_ENGRAM_PREFETCH_MS, (ggml_time_us() - t_hashed)/1000.0);
        }
        ggml_account_add(GGML_ACCOUNT_ENGRAM_LOGICAL_BYTES,
                (double) idx.size() * (double) ggml_row_size(table->parts[0]->type, table->parts[0]->ne[0]));
    }
    const int64_t t_input = timing ? ggml_time_us() : 0;

    // global row numbers -> per piece local row numbers and the mask that keeps the piece which holds the row
    std::vector<int32_t> piece_idx;
    std::vector<float>   piece_mask;
    llama_table_pieces_split(*table, idx, piece_idx, piece_mask);

    ggml_backend_tensor_set(rows, piece_idx.data(), 0, piece_idx.size()*ggml_element_size(rows));
    if (mask) {
        ggml_backend_tensor_set(mask, piece_mask.data(), 0, piece_mask.size()*ggml_element_size(mask));
    }

    if (timing) {
        stats.t_hash_us  = t_hashed - t_start;
        stats.t_input_us = ggml_time_us() - t_input;
        llama_prefetch_stats_log("engram", (int) pmodel.hparams.engram_layer_ids[eg], n_tokens, stats);
    }
}

ggml_tensor * llama_model_deepseek41::graph::build_inp_engram(
        const llama_model & model,
        int il) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    const int64_t n_cols  = (hparams.engram_max_ngram_size - 1) * hparams.engram_n_head;
    const int64_t key_len = hparams.engram_key_length;

    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    const llama_table_pieces & table = model.layers[il].engram_table;
    auto inp = std::make_unique<llm_graph_input_engram>(pmodel, mctx_cur->get_raw(), pmodel.engram_index(il),
                                                        &table, cparams.ple_prefetch);

    // the shapes depend on the number of pieces of the table, never on the rows: every piece gathers n_cols * n_tokens rows
    // and a mask keeps the piece that holds each one
    const int64_t n_gather = n_cols * n_tokens;
    inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_gather * (int64_t) table.size());
    ggml_set_input(inp->rows);
    ggml_tensor * rows = inp->rows;
    ggml_tensor * mask = nullptr;
    if (table.size() > 1) {
        inp->mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_gather * (int64_t) table.size());
        ggml_set_input(inp->mask);
        mask = inp->mask;
    }
    res->add_input(std::move(inp));

    // gather then flatten, laying the buckets out slowest, as the reference does
    // (the table is MXFP8 rows read in place from the mapping; the gather dequantizes to F32 on the CPU)
    ggml_tensor * emb = llama_weight_aux_get_rows_pieces(ctx0, table, nullptr, rows, mask, 1);

    if (ggml_account_enabled()) {
        // tag the gather nodes so that the CPU backend can time them (RANMA_PREFILL_ACCOUNT)
        std::vector<ggml_tensor *> todo = { emb };
        for (size_t i = 0; i < todo.size() && i < 64; ++i) {
            ggml_tensor * t = todo[i];
            if (t->op == GGML_OP_GET_ROWS) {
                ggml_set_name(t, "engram_gather");
            } else if (t->op == GGML_OP_ADD) {
                todo.push_back(t->src[0]);
                todo.push_back(t->src[1]);
            } else if (t->op == GGML_OP_MUL) {
                todo.push_back(t->src[0]);
            }
        }
    }
    emb = ggml_reshape_2d(ctx0, emb, key_len * n_cols, n_tokens);
    cb(emb, "engram_embd", il);

    return emb;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model & model,
        ggml_tensor * x,
        ggml_tensor * emb,
        int il) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // one projection makes a key per hc copy plus one value they all share
    ggml_tensor * kv = build_lora_mm(model.layers[il].engram_wkv, emb);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, hc_dim, nt, kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_2d(ctx0, kv, n_embd, nt, kv->nb[1], hc_dim*kv->nb[0]));

    // The gate scales reach ggml_mul, which takes only f32, and a file quantized before
    // llama-quant.cpp learned to skip them carries them quantized. get_rows dequantizes.
    auto as_f32 = [&](ggml_tensor * w) {
        if (w->type == GGML_TYPE_F32) {
            return w;
        }
        ggml_tensor * ids = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) w->ne[1], 1.0f), GGML_TYPE_I32);
        return ggml_get_rows(ctx0, w, ids);
    };

    // normalized per (token, hc copy) over n_embd, not jointly over the copies. The reference
    // keeps engram_q and engram_k apart but only ever uses their product, so applying one to each
    // side of the dot product gives the same result.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
        t = ggml_rms_norm(ctx0, t, norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, nt);
        t = ggml_mul(ctx0, t, ggml_reshape_2d(ctx0, w, hc_dim, 1));
        return ggml_reshape_3d(ctx0, t, n_embd, hc, nt);
    };

    ggml_tensor * k = grouped_norm(key, as_f32(model.layers[il].engram_k));
    ggml_tensor * q = grouped_norm(x,   as_f32(model.layers[il].engram_q));

    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, k, q));
    s = ggml_scale(ctx0, s, 1.0f/sqrtf((float) n_embd));

    // signed square root before the sigmoid, matching the training kernel.
    // The reference uses copysign, which treats +0 as positive where ggml_sgn gives 0, so a dot
    // product of exactly zero gates 0.5 here against 0.50025 there. qwen4exp's PLE gate is built
    // the same way.
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "engram_gate", il);

    // the value is shared across the copies, only the gate differs
    ggml_tensor * v = ggml_reshape_3d(ctx0, value, n_embd, 1, nt);
    v = ggml_repeat_4d(ctx0, v, n_embd, hc, nt, 1);

    return ggml_add(ctx0, x, ggml_mul(ctx0, v, gate));
}

// Rope settings for one layer. A layer that reads a compressed stream rotates with the
// compressor's base and YaRN; a plain sliding window layer rotates with the model's.
struct dsv41_rope_cfg {
    float   base;
    float   scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
    int32_t n_ctx_orig;
};

dsv41_rope_cfg llama_model_deepseek41::graph::rope_cfg(int il) const {
    if (hparams.dsv4_compress_ratios[il] == 0) {
        return { freq_base, 1.0f, 0.0f, dsv4_rope_attn_factor(1.0f, 0.0f), 0.0f, 0.0f, 0 };
    }

    return {
        hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
        dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow, n_ctx_orig,
    };
}

// Undo the rotation the query carried into attention, then the grouped output projection. wo_a is
// block diagonal over groups, each projecting only its own heads, hence a batched mul_mat.
ggml_tensor * llama_model_deepseek41::graph::build_attention_tail(
        const llama_model & model,
        ggml_tensor * out,
        ggml_tensor * inp_pos,
        int64_t nt,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = (n_head/n_groups)*n_embd_head;

    const dsv41_rope_cfg rc = rope_cfg(il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);

    ggml_tensor * oa = build_mm(layer.wo_a, out);
    cb(oa, "attn_wo_a", il);

    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

// Score this layer's queries against the shared index keys and keep the best compressed positions.
// The keys were published by an earlier layer, so this only builds the query side.
ggml_tensor * llama_model_deepseek41::graph::build_indexer_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        dsv41_top_k_carry & top_k_carry,
        int il,
        int64_t i0,
        int64_t n,
        ggml_tensor * comp_mask,
        ggml_tensor ** cand_keep_out) const {
    const auto & layer = model.layers[il];

    ggml_tensor * mask      = inp_comp.kq_mask;
    ggml_tensor * cand_keep = top_k_carry.cand_keep;

    // ranma: a chunk reads its rows of the whole-ubatch tensors (inp_pos through the shared view cache, it is
    // a graph input), the causal mask of those rows and the same rows of the carried candidate keep
    if (n >= 0) {
        GGML_ASSERT(inp_cm != nullptr && comp_mask != nullptr && cand_keep_out != nullptr);
        qr      = llm_graph_rows(ctx0, qr,  1, i0, n);
        cur     = llm_graph_rows(ctx0, cur, 1, i0, n);
        inp_pos = inp_cm->rows.get(ctx0, inp_pos, 0, i0, n, false);
        mask    = comp_mask;
        if (cand_keep) {
            cand_keep = llm_graph_rows(ctx0, cand_keep, 2, i0, n);
        }
    }

    const int64_t n_idx_head      = hparams.indexer_n_head;
    const int64_t n_idx_head_dim  = hparams.indexer_head_size;
    const int64_t n_idx_head_rope = hparams.n_rot();
    const int64_t n_idx_head_nope = n_idx_head_dim - n_idx_head_rope;
    const int64_t nt              = cur->ne[1];

    GGML_ASSERT(inp_comp.kq_mask);
    GGML_ASSERT(n_idx_head_dim >= n_idx_head_rope);

    ggml_tensor * idx_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    idx_q = ggml_reshape_3d(ctx0, idx_q, n_idx_head_dim, n_idx_head, nt);
    idx_q = ggml_rope_ext(ctx0, idx_q, inp_pos, nullptr, n_idx_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    idx_q = ggml_rope_set_offset(idx_q, n_idx_head_nope);
    cb(idx_q, "idx_q", il);

    ggml_tensor * idx_k_rot = inp_dsv4->get_lid().k_rot;
    if (idx_k_rot) {
        idx_q = llama_mul_mat_hadamard(ctx0, idx_q, idx_k_rot);
        cb(idx_q, "idx_q_rot", il);
    }

    // one weight per head, scaled so the score matches the reference's
    // softmax_scale * n_heads**-0.5
    ggml_tensor * idx_w = build_lora_mm(layer.indexer_proj, cur);
    idx_w = ggml_scale(ctx0, idx_w, 1.0f/sqrtf(float(n_idx_head_dim*n_idx_head)));
    cb(idx_w, "idx_weights", il);

    ggml_tensor * idx_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);

    const int64_t n_comp = mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= idx_k->ne[2]);

    idx_k = ggml_view_4d(ctx0, idx_k,
            idx_k->ne[0], idx_k->ne[1], n_comp, idx_k->ne[3],
            idx_k->nb[1], idx_k->nb[2], idx_k->nb[3], 0);
    cb(idx_k, "idx_k", il);

    const int64_t n_stream = idx_k->ne[3];
    idx_q = ggml_view_4d(ctx0, idx_q,
            idx_q->ne[0], idx_q->ne[1], idx_q->ne[2]/n_stream, n_stream,
            idx_q->nb[1], idx_q->nb[2], idx_q->nb[3]/n_stream, 0);
    idx_w = ggml_view_4d(ctx0, idx_w,
            idx_w->ne[0], idx_w->ne[1]/n_stream, idx_w->ne[2], n_stream,
            idx_w->nb[1], idx_w->nb[2]/n_stream, idx_w->nb[3]/n_stream, 0);

    // Level two: a source after the candidate source ranks only the positions inside its blocks.
    // The candidate source itself ranks everything, so its mask never touches its own top-k.
    const bool use_cand = hparams.candidate_source_layer >= 0 && il > hparams.candidate_source_layer && cand_keep;
    bool cand_in_score  = false;

    // the attention mask is F16 when flash attention is on and F32 otherwise. it only ever holds
    // 0 or -inf, so converting it either way is exact.
    ggml_tensor * score = nullptr;
    if (cparams.fused_lid) {
        ggml_tensor * mask_f16 = mask->type == GGML_TYPE_F16 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F16);
        if (use_cand) {
            // the op takes the keep of each block, a backend then skips the keys of a dropped block and scores it -inf
            score = ggml_lightning_indexer_cand(ctx0, idx_q, idx_k, idx_w, mask_f16, cand_keep, (int32_t) hparams.candidate_block_size);
            cand_in_score = true;
        } else {
            score = ggml_lightning_indexer(ctx0, idx_q, idx_k, idx_w, mask_f16);
        }
        cb(score, "idx_score", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
    } else {
        idx_q = ggml_permute(ctx0, idx_q, 0, 2, 1, 3);
        idx_k = ggml_permute(ctx0, idx_k, 0, 2, 1, 3);

        score = ggml_mul_mat(ctx0, idx_k, idx_q);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        score = ggml_relu(ctx0, score);
        score = ggml_mul(ctx0, score, idx_w);
        score = ggml_sum_rows(ctx0, score);
        score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));

        score = ggml_add(ctx0, score, mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
        cb(score, "idx_score", il);
    }

    ggml_tensor * ranked = score;
    if (use_cand && !cand_in_score) {
        const int64_t bs = hparams.candidate_block_size;
        ggml_tensor * keep = cand_keep;
        GGML_ASSERT(keep->ne[0] == 1 && keep->ne[1]*bs == score->ne[0] && keep->ne[2] == score->ne[1] && keep->ne[3] == score->ne[3]);

        // each block's entry broadcasts over its positions. Nothing else reads this score, so it
        // is masked in place; the score is contiguous, and so is the reshape.
        ranked = ggml_reshape_4d(ctx0, score, bs, keep->ne[1], score->ne[1], score->ne[3]);
        ranked = ggml_add_inplace(ctx0, ranked, keep);
        ranked = ggml_reshape_4d(ctx0, ranked, score->ne[0], score->ne[1], 1, score->ne[3]);
        cb(ranked, "idx_score_cand", il);
    }

    const uint32_t n_top_k = score->ne[0] < hparams.indexer_top_k ? score->ne[0] : hparams.indexer_top_k;

    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, ranked, n_top_k));
    cb(top_k, "idx_top_k", il);

    if (il == hparams.candidate_source_layer) {
        ggml_tensor * keep = build_candidate_keep(score,
                mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32), il);
        if (cand_keep_out) {
            *cand_keep_out = keep;
        } else {
            top_k_carry.cand_keep = keep;
        }
    }

    return top_k;
}

// Level one of the two level top-k, as select_candidate_blocks() in the reference. Blocks of
// candidate_block_size consecutive positions are scored by their best position, the block holding
// the query's newest position is pinned in, and the best candidate_topk_blocks blocks stay.
// Returns 0 / -inf per block, F32 [1, n_blocks, nt, n_stream], to be broadcast over the block's
// positions by the caller, or null when every block would stay anyway.
//
//   score    F32 [n_comp, nt, 1, n_stream]  index score, -inf on the positions the query cannot reach
//   kq_mask  same shape and type            0 on the reachable prefix, -inf after it
ggml_tensor * llama_model_deepseek41::graph::build_candidate_keep(
        ggml_tensor * score,
        ggml_tensor * kq_mask,
        int il) const {
    const int64_t bs     = hparams.candidate_block_size;
    const int64_t n_comp = score->ne[0];
    const int64_t nt     = score->ne[1];
    const int64_t ns     = score->ne[3];

    GGML_ASSERT(score->type == GGML_TYPE_F32 && kq_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(score->ne[2] == 1 && ggml_are_same_shape(score, kq_mask));
    // the compressed length is padded to 256 rows, so blocks never straddle the end
    GGML_ASSERT(n_comp % bs == 0);

    const int64_t n_blocks = n_comp / bs;
    const int64_t n_top    = std::min<int64_t>(hparams.candidate_topk_blocks, n_blocks);

    // Taking every block is the same as taking none away: a block the query cannot reach has no
    // finite position, and the attention mask hides those anyway.
    if (n_blocks <= (int64_t) hparams.candidate_topk_blocks) {
        return nullptr;
    }

    // block score = best position in it. A block with no reachable position comes out as
    // -FLT_MAX rather than -inf, which still ranks below every real score (they are >= 0).
    ggml_tensor * blk = ggml_pool_2d(ctx0, score, GGML_OP_POOL_MAX, bs, 1, bs, 1, 0, 0);
    cb(blk, "cand_block_score", il);

    // The newest block is the last one with a reachable position. Pooling the mask marks each
    // block 0 (reachable) or -FLT_MAX; clamping to -1 turns the sum over blocks into minus the
    // number of unreachable ones, so the last reachable index is n_blocks - 1 minus that.
    ggml_tensor * dark = ggml_clamp(ctx0, ggml_pool_2d(ctx0, kq_mask, GGML_OP_POOL_MAX, bs, 1, bs, 1, 0, 0), -1.0f, 0.0f);
    ggml_tensor * last = ggml_scale_bias(ctx0, ggml_sum_rows(ctx0, dark), 1.0f, (float) (n_blocks - 1));
    // a query that sees nothing yet has no newest block; pin block 0, which the mask hides
    last = ggml_clamp(ctx0, last, 0.0f, (float) (n_blocks - 1));
    ggml_tensor * last_idx = ggml_reshape_3d(ctx0, ggml_cast(ctx0, last, GGML_TYPE_I32), 1, nt, ns);
    cb(last_idx, "cand_last_block", il);

    // pin it: write +inf over that block's score, one row of one element per query
    ggml_tensor * rows   = ggml_reshape_4d(ctx0, blk, 1, n_blocks, nt, ns);
    ggml_tensor * pinned = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, 1, nt, ns), INFINITY);
    rows = ggml_set_rows(ctx0, rows, pinned, last_idx);

    ggml_tensor * top = ggml_cont(ctx0, ggml_top_k(ctx0, ggml_reshape_4d(ctx0, rows, n_blocks, nt, 1, ns), n_top));
    cb(top, "cand_top_blocks", il);

    // chosen blocks -> 0 in a -inf row per query
    ggml_tensor * keep  = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blocks, nt, ns), -INFINITY);
    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_top,    nt, ns), 0.0f);
    keep = ggml_set_rows(ctx0, keep, zeros, ggml_reshape_3d(ctx0, top, n_top, nt, ns));
    cb(keep, "cand_keep", il);

    return keep;
}

// The compressor of a KV source layer, over the whole ubatch: the rows it completes go to the
// compressed cache, their index keys to the indexer cache where this layer owns them, and what did
// not complete a row is carried into the next ubatch.
void llama_model_deepseek41::graph::build_kv_source(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        const llama_dsv4_comp_state * comp_state,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t ratio       = hparams.dsv4_compress_ratios[il];

    ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
    cb(state_kv, "comp_state_kv", il);

    // At ratio 1 there is nothing to pool and the file carries no gate. The softmax below
    // then runs over a single element and returns 1.0 whatever the score holds, so the
    // values reach the cache unweighted, which is what a plain projection means.
    ggml_tensor * state_score = layer.attn_comp_wgate
        ? build_lora_mm(layer.attn_comp_wgate, cur)
        : state_kv;
    cb(state_score, "comp_state_score", il);

    const dsv4_state_tensors restored = dsv4_build_state_restore(ctx0, inp_comp, comp_state, il);

    ggml_tensor * base_kv = dsv4_view_2d(
            ctx0, restored.kv, restored.kv->ne[0], comp_state->get_n_rows(), 0);
    ggml_tensor * base_score = dsv4_view_2d(
            ctx0, restored.score, restored.score->ne[0], comp_state->get_n_rows(), 0);

    ggml_tensor * source_kv    = ggml_concat(ctx0, base_kv,    state_kv,    1);
    ggml_tensor * source_score = ggml_concat(ctx0, base_score, state_score, 1);

    // the indexer reads the latent before it is rotated, so ask for both forms at once
    ggml_tensor * latent_pre = nullptr;

    ggml_tensor * latent = build_hca_compressed_kv_from_state(
            source_kv,
            source_score,
            inp_comp.state_read_idxs,
            inp_comp.state_write_pos,
            layer.attn_comp_norm,
            ratio,
            n_embd_head,
            "comp_kv",
            il,
            &latent_pre);

    if (hparams.dsv41_owns_index_k(il)) {
        const int64_t n_idx_head_dim  = hparams.indexer_head_size;
        const int64_t n_idx_head_rope = hparams.n_rot();

        ggml_tensor * idx_k = build_lora_mm(layer.indexer_attn_k, latent_pre);
        idx_k = build_norm(idx_k, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
        idx_k = ggml_rope_ext(ctx0, idx_k, inp_comp.state_write_pos, nullptr, n_idx_head_rope,
                rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
                dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
        idx_k = ggml_rope_set_offset(idx_k, n_idx_head_dim - n_idx_head_rope);
        cb(idx_k, "idx_k_new", il);

        if (inp_dsv4->get_lid().k_rot) {
            idx_k = llama_mul_mat_hadamard(ctx0, idx_k, inp_dsv4->get_lid().k_rot);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(
                    ctx0, idx_k, inp_comp.state_write_idxs, il));
    }

    if (inp_dsv4->get_csa().k_rot) {
        latent = llama_mul_mat_hadamard(ctx0, latent, inp_dsv4->get_csa().k_rot);
        cb(latent, "comp_kv_rot", il);
    }

    ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(
                ctx0, latent, inp_comp.state_write_idxs, il));

    // carry whatever did not complete a row into the next ubatch
    ggml_tensor * snapshot_kv    = ggml_concat(ctx0, restored.kv,    state_kv,    1);
    ggml_tensor * snapshot_score = ggml_concat(ctx0, restored.score, state_score, 1);

    const dsv4_state_tensors snapshot = dsv4_build_state_snapshot(
            ctx0, inp_comp, comp_state, snapshot_kv, snapshot_score, il);
    if (snapshot.kv != nullptr) {
        ggml_build_forward_expand(gf, snapshot.kv);
    }
    if (snapshot.score != nullptr) {
        ggml_build_forward_expand(gf, snapshot.score);
    }

    ggml_tensor * persist_kv = ggml_get_rows(ctx0, state_kv, inp_comp.state_persist_src_idxs);
    ggml_tensor * persist_score = ggml_get_rows(ctx0, state_score, inp_comp.state_persist_src_idxs);

    ggml_build_forward_expand(gf, comp_state->cpy_kv(
                ctx0, persist_kv, inp_comp.state_persist_dst_idxs, il));
    ggml_build_forward_expand(gf, comp_state->cpy_score(
                ctx0, persist_score, inp_comp.state_persist_dst_idxs, il));
}

// DeepSeek-V4.1 attention: a sliding window of raw KV, plus, where the layer uses one, the
// compressed positions the indexer picked, concatenated into a single masked attention.
//
// Only a source layer compresses. The layers after it read the same rows, which the KV cache
// hands them through the reuse callback, so a reader builds no compressor at all.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        dsv41_top_k_carry & top_k_carry,
        int il) const {
    // ranma: a chunked graph, see build_attention_v41_chunked
    if (inp_cm != nullptr) {
        return build_attention_v41_chunked(model, inp_dsv4, cur, inp_pos, top_k_carry, il);
    }

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    // Query. V4 normalizes again after wq_b; V4.1 normalizes only the low rank part.
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    // the sliding window KV, which every layer keeps for itself
    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        // no compressed stream, so this layer sees only its own window
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);

        return build_attention_tail(model, out, inp_pos, nt, il);
    }

    // The plan slot follows the ratio, since a plan encodes how many tokens make a row. The rows
    // themselves always live in the CSA cache, and the index keys in the indexer cache, whichever
    // plan produced them.
    const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

    const auto & inp_comp = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();

    const llama_dsv4_comp_state * comp_state = use_csa
        ? inp_dsv4->mctx->get_csa_state()
        : inp_dsv4->mctx->get_hca_state();

    GGML_ASSERT(inp_comp.kq_mask && "a compressed layer needs a plan for its ratio");

    if (hparams.dsv41_is_kv_source(il) && inp_comp.state_pos) {
        build_kv_source(model, inp_dsv4, inp_comp, comp_state, cur, il);
    }

    // an index source picks the positions; the layers in between reuse what it picked. Without the
    // carry a reusing layer would fall back to the full causal mask and attend to every compressed
    // row, which matches the reference only while a source has <= index_topk rows
    if (hparams.dsv41_is_index_source(il)) {
        top_k_carry.top_k = build_indexer_top_k(model, inp_dsv4, inp_comp, qr, cur, inp_pos, top_k_carry, il);
        top_k_carry.il    = (int32_t) il;
    }
    GGML_ASSERT(top_k_carry.top_k != nullptr && top_k_carry.il == hparams.dsv41_topk_source[il] &&
            "a compressed layer needs the top-k of its index source");
    ggml_tensor * top_k = top_k_carry.top_k;

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

    const int64_t n_comp = inp_comp.kq_mask->ne[0];
    GGML_ASSERT(n_comp > 0);
    GGML_ASSERT(n_comp <= comp_k->ne[2]);

    comp_k = ggml_view_4d(ctx0, comp_k,
            comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
            comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
    cb(comp_k, "comp_k", il);

    // ranma: [raw | compressed] without copying the compressed cells: the raw cells go into the staging area in front of
    // the compressed cells of the KV source (llama_kv_cache_dsv4::init_kall_v41), and attention reads a view of both.
    // Only the window of raw cells the ubatch can see is copied; the staging cells outside it hold rows of an earlier
    // layer, which the mask drops for every token just as it drops the raw cells there. Without the staging area
    // (several sequences, LLAMA_DSV4_KALL_VIEW=0) the graph concatenates as before, and so does a graph of the
    // layer-major prefill: the layers that read one KV source would write its staging area in several segments.
    ggml_tensor * k_all = nullptr;
    if (!layer_exec) {
        const auto *   comp_ctx = inp_dsv4->mctx->get_csa();
        const uint32_t n_stage  = comp_ctx->get_k_joint_prefix(il);
        const uint32_t n_raw    = (uint32_t) raw_k->ne[2];

        if (raw_k->ne[3] == 1 && n_stage >= n_raw && n_raw > 0) {
            ggml_tensor * stage = comp_ctx->get_k_joint(ctx0, il, n_stage - n_raw, n_raw, 0);
            k_all = comp_ctx->get_k_joint(ctx0, il, n_stage - n_raw, n_raw, (uint32_t) n_comp);
            GGML_ASSERT(stage != nullptr && k_all != nullptr);
            GGML_ASSERT(stage->ne[0] == raw_k->ne[0] && stage->ne[1] == raw_k->ne[1]);

            uint32_t w_lo = 0;
            uint32_t w_n  = 0;
            mctx_raw->get_chunk_window(0, (uint32_t) nt, w_lo, w_n);
            inp_attn->stage_lo = w_lo;
            inp_attn->stage_n  = w_n;

            const int64_t j0 = (int64_t) w_lo - (int64_t) mctx_raw->get_k_lo();
            GGML_ASSERT(j0 >= 0 && j0 + (int64_t) w_n <= (int64_t) n_raw);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0,
                    llm_graph_rows(ctx0, raw_k, 2, j0, w_n),
                    llm_graph_rows(ctx0, stage, 2, j0, w_n)));
        }
    }
    if (k_all == nullptr) {
        k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
    }
    cb(k_all, "k_all", il);

    ggml_tensor * raw_mask  = inp_attn->get_kq_mask();
    ggml_tensor * comp_mask = top_k
        ? build_top_k_mask(inp_comp.kq_mask, top_k, "comp_top_k_mask", il)
        : inp_comp.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
    cb(kq_mask, "kq_mask", il);

    const int64_t n_kv_max = top_k
        ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0]
        : 0;

    out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
            nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
}

// ranma: the attention mask of one chunk, [n_raw + n_comp, n] of the type of ninf: the raw window rows, then the
// compressed rows, where only the top-k positions take their causal value and every other one is -inf. These are
// the values of build_top_k_mask over the causal mask concatenated behind the raw one, but the [n_comp, n] causal
// mask itself is never made: the compressed causal mask is a prefix, position j visible to token t while
// j < n_visible(t) (dsv4_set_kq_mask), so the value of a selected j follows from j and the token key alone.
//
//   raw      F32 [n_raw, n]      the raw window mask of the chunk, 0 or -inf
//   top_k    I32 [n_top_k, n]    the selected compressed positions, distinct per token
//   key      F32 [1, n]          -(n_visible - 0.5) per token (build_comp_chunk_key)
//   raw_idx  I32 [n_raw]         0 .. n_raw - 1
//   ninf     [1, 1, 1, 1]        -inf in the type of the mask
ggml_tensor * llama_model_deepseek41::graph::build_chunk_kq_mask(
        ggml_tensor * raw,
        ggml_tensor * top_k,
        ggml_tensor * key,
        ggml_tensor * raw_idx,
        ggml_tensor * ninf,
        int64_t n_comp,
        int il) const {
    const int64_t n_raw = raw->ne[0];
    const int64_t n     = raw->ne[1];
    const int64_t n_top = top_k->ne[0];

    GGML_ASSERT(raw->type == GGML_TYPE_F32 && ggml_nelements(raw) == n_raw*n);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && top_k->ne[1] == n && ggml_nelements(top_k) == n_top*n);
    GGML_ASSERT(key->type == GGML_TYPE_F32 && key->ne[0] == 1 && key->ne[1] == n);
    GGML_ASSERT(raw_idx->type == GGML_TYPE_I32 && raw_idx->ne[0] == n_raw);
    GGML_ASSERT(n_top <= n_comp);

    // one row of one element per key, as in build_top_k_mask
    ggml_tensor * mask = ggml_repeat_4d(ctx0, ninf, 1, n_raw + n_comp, n, 1);

    // selected j: n_visible - 0.5 - j = -(j + key) > 0 when visible; step gives 1 or 0, log 0 or -inf. All exact in F32.
    ggml_tensor * pos = ggml_cast(ctx0, top_k, GGML_TYPE_F32);
    ggml_tensor * vis = ggml_add(ctx0, pos, key);
    vis = ggml_scale_inplace(ctx0, vis, -1.0f);
    vis = ggml_step_inplace(ctx0, vis);
    vis = ggml_log_inplace(ctx0, vis);

    // behind the raw rows
    ggml_tensor * idx = ggml_cast(ctx0, ggml_scale_bias(ctx0, pos, 1.0f, (float) n_raw), GGML_TYPE_I32);

    mask = ggml_set_rows(ctx0, mask, ggml_reshape_3d(ctx0, vis, 1, n_top, n), idx);

    // the raw rows, the same row indices for every token
    mask = ggml_set_rows(ctx0, mask, ggml_reshape_3d(ctx0, raw, 1, n_raw, n), ggml_reshape_2d(ctx0, raw_idx, n_raw, 1));

    mask = ggml_reshape_2d(ctx0, mask, n_raw + n_comp, n);
    cb(mask, "kq_mask", il);

    return mask;
}

// ranma: build_attention_v41 of a chunked graph (the attention chunk; one stream, one sequence, device-side masks,
// see build_inp_dsv4_chunk). The whole ubatch computes the query and the raw KV, runs the compressor of a source
// layer and writes the raw window, as the unchunked graph does. Then every chunk of query rows makes its raw window
// mask, its index scores and top-k where this layer is an index source, its attention mask and its flash attention,
// so that every [n_comp, n] intermediate is one chunk wide. The outputs are joined for the attention tail, which
// runs over the whole ubatch once. An index source carries the top-k of all rows (the candidate source also its
// block keep) to the layers after it: compact per-token tensors, never an [n_comp, n_tokens] mask.
ggml_tensor * llama_model_deepseek41::graph::build_attention_v41_chunked(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        dsv41_top_k_carry & top_k_carry,
        int il) const {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t nt               = cur->ne[1];
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    GGML_ASSERT(inp_cm != nullptr && inp_dch != nullptr);
    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % hparams.dsv4_o_group_count == 0);

    const dsv41_rope_cfg rc = rope_cfg(il);

    // the query and the raw KV of the whole ubatch, as in build_attention_v41
    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, rc.n_ctx_orig,
            rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
    kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
    cb(kv, "kv", il);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // the compressed stream this layer reads, and the compressor of a source over the whole ubatch
    const llm_graph_input_dsv4::comp_input * inp_comp = nullptr;
    if (ratio != 0) {
        const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

        inp_comp = use_csa ? &inp_dsv4->get_csa() : &inp_dsv4->get_hca();

        const llama_dsv4_comp_state * comp_state = use_csa
            ? inp_dsv4->mctx->get_csa_state()
            : inp_dsv4->mctx->get_hca_state();

        GGML_ASSERT(inp_comp->kq_mask && "a compressed layer needs a plan for its ratio");

        if (hparams.dsv41_is_kv_source(il) && inp_comp->state_pos) {
            build_kv_source(model, inp_dsv4, *inp_comp, comp_state, cur, il);
        }
    }

    const bool index_source = inp_comp != nullptr && hparams.dsv41_is_index_source(il);
    const bool cand_source  = index_source && il == hparams.candidate_source_layer;

    // the carried top-k holds the last of the ubatch rows (all of them, or the query rows of a tail layer)
    if (inp_comp != nullptr && !index_source) {
        GGML_ASSERT(top_k_carry.top_k != nullptr && top_k_carry.il == hparams.dsv41_topk_source[il] &&
                top_k_carry.top_k->ne[1] >= nt && "a compressed layer needs the top-k of its index source");
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    // the whole ubatch writes the raw window, then every chunk reads it
    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "raw_k", il);

    // the host masks stay out of a chunked graph; only their shapes and types are read
    const int64_t   n_raw     = inp_attn->self_kq_mask->ne[0];
    const ggml_type type_mask = inp_attn->self_kq_mask->type;

    // raw chunk: every chunk reads only its raw window [j0, j0 + nj) of the raw cells (inp_cm->window), so
    // its K is concatenated per chunk. The compressed width stays that of the ubatch: the top-k of an index source is
    // carried to its reader layers at one width. The reserve graph has whole windows: it is the widest.
    const bool win = !inp_cm->windows.empty();

    ggml_tensor * k_all   = raw_k;
    ggml_tensor * comp_k  = nullptr;
    ggml_tensor * raw_idx = nullptr;
    ggml_tensor * ninf    = nullptr;
    int64_t       n_comp  = 0;

    if (inp_comp != nullptr) {
        comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

        n_comp = inp_comp->kq_mask->ne[0];
        GGML_ASSERT(n_comp > 0);
        GGML_ASSERT(n_comp <= comp_k->ne[2]);

        comp_k = ggml_view_4d(ctx0, comp_k,
                comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
                comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
        cb(comp_k, "comp_k", il);

        // once for the whole ubatch
        if (!win) {
            k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
            cb(k_all, "k_all", il);
            ggml_build_forward_expand(gf, k_all);
        }

        raw_idx = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) n_raw, 1.0f), GGML_TYPE_I32);
        ninf    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, type_mask, 1, 1, 1, 1), -INFINITY);
    }

    // the causal mask the indexer of a chunk reads: F16 for the fused op, F32 where the score is added to it or the
    // candidate blocks pool it (build_indexer_top_k converts what it is handed)
    const ggml_type type_idx = cparams.fused_lid && !cand_source ? GGML_TYPE_F16 : GGML_TYPE_F32;

    const llm_graph_chunks chunks = inp_cm->chunks(nt);
    GGML_ASSERT(chunks.n > 1 || (inp_cm->one && inp_cm->windows.empty()));

    llm_graph_join_acc outs(ctx0, gf, 1, nt);
    llm_graph_join_acc top_ks(ctx0, gf, 1, nt);
    llm_graph_join_acc keeps(ctx0, gf, 2, nt);

    for (int ic = 0; ic < chunks.n; ++ic) {
        const int64_t i0 = chunks.i0(ic);
        const int64_t n  = chunks.nk(ic);

        ggml_tensor * mask     = nullptr;
        int64_t       n_kv_max = 0;

        // the raw window of the chunk (all raw cells without the raw chunk)
        int64_t j0 = 0;
        int64_t nj = 0;
        inp_cm->window(i0, n, j0, nj);

        ggml_tensor * k_cur = k_all;
        if (win) {
            k_cur = llm_graph_rows(ctx0, raw_k, 2, j0, nj);
            if (inp_comp != nullptr) {
                k_cur = ggml_concat(ctx0, k_cur, comp_k, 2);
                cb(k_cur, "k_all", il);
            }
        }

        if (inp_comp == nullptr) {
            // no compressed stream, so this layer sees only its own window
            GGML_ASSERT(hparams.is_swa(il));
            mask = inp_cm->mask(ctx0, i0, n, type_mask);
        } else {
            ggml_tensor * top_k = nullptr;
            if (index_source) {
                ggml_tensor * comp_mask = build_comp_chunk_mask(inp_dsv4, *inp_comp, i0, n, type_idx);
                ggml_tensor * keep      = nullptr;

                top_k = build_indexer_top_k(model, inp_dsv4, *inp_comp, qr, cur, inp_pos, top_k_carry, il,
                        i0, n, comp_mask, &keep);

                top_ks.add(top_k, i0);
                if (keep != nullptr) {
                    keeps.add(keep, i0);
                }
            } else {
                top_k = llm_graph_rows(ctx0, top_k_carry.top_k, 1, top_k_carry.top_k->ne[1] - nt + i0, n);
            }

            mask = build_chunk_kq_mask(inp_cm->mask(ctx0, i0, n, GGML_TYPE_F32), top_k,
                    build_comp_chunk_key(inp_dsv4, *inp_comp, i0, n), llm_graph_rows(ctx0, raw_idx, 0, 0, nj), ninf, n_comp, il);

            n_kv_max = std::min<int64_t>(nj, hparams.n_swa) + top_k->ne[0];
        }

        ggml_tensor * out = build_attn_mha(llm_graph_rows(ctx0, q, 2, i0, n), k_cur, k_cur, nullptr, mask,
                layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);

        // finish this chunk before the next one starts, so the allocator reuses its intermediates
        ggml_build_forward_expand(gf, out);
        outs.add(out, i0);
    }

    if (index_source) {
        top_k_carry.top_k = top_ks.get();
        top_k_carry.il    = (int32_t) il;
        ggml_build_forward_expand(gf, top_k_carry.top_k);

        if (cand_source) {
            // build_candidate_keep gives every chunk a keep or none at all, by the compressed length only
            top_k_carry.cand_keep = keeps.empty() ? nullptr : keeps.get();
            if (top_k_carry.cand_keep != nullptr) {
                ggml_build_forward_expand(gf, top_k_carry.cand_keep);
            }
        }
    }

    ggml_tensor * out = outs.get();
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_out_raw", il);

    return build_attention_tail(model, out, inp_pos, nt, il);
}

// ranma: the attention of a tail layer (see dsv41_tail_plan) over the input cur of n_in rows, the last n_in of the
// ubatch: the whole ubatch on il_first, whose KV source compresses every row, and K rows after it. The raw KV is made
// and written for the last K rows, through the write indices of the same tokens; the query, the indexer, the masks and
// the output for the last n_q. Positions stay those of the ubatch rows. Returns [n_embd, n_q].
struct dsv41_tail_builder {
    const llama_model_deepseek41::graph & g;
    const dsv41_tail_plan & plan;

    // the host masks narrowed once to the widest query rows of the tail, as one copy that every layer views: a view of a
    // host input per layer would be a device copy of its own for the whole graph (see llm_graph_rows_cache)
    std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> masks;

    ggml_tensor * mask(ggml_tensor * m, int64_t n) {
        GGML_ASSERT(m->ne[1] == g.n_tokens && m->ne[2] == 1 && m->ne[3] == 1);

        ggml_tensor * t = nullptr;
        for (const auto & e : masks) {
            if (e.first == m) {
                t = e.second;
            }
        }
        if (t == nullptr) {
            const int64_t n_max = plan.n_q(plan.il_first);
            t = ggml_cont(g.ctx0, llm_graph_rows(g.ctx0, m, 1, g.n_tokens - n_max, n_max));
            masks.emplace_back(m, t);
        }

        GGML_ASSERT(n <= t->ne[1]);
        return llm_graph_rows(g.ctx0, t, 1, t->ne[1] - n, n);
    }

    // a chunked graph: the device-side raw mask [nj, n] of the ubatch rows [i0, i0 + n) over the raw columns [j0, j0 + nj)
    // of w, made as llm_graph_input_chunk_mask::mask makes it for a chunk of the ubatch (which looks its window up by i0
    // among the ubatch chunks, where a query chunk need not be)
    ggml_tensor * raw_chunk_mask(const llm_graph_chunk_window & w, ggml_type type) const {
        llm_graph_input_chunk_mask * cm = g.inp_cm;

        ggml_tensor * tab = cm->rows.get(g.ctx0, cm->kv_pos, 0, w.j0, w.nj, false);
        return llm_graph_chunk_mask(g.ctx0, cm->rows, tab, cm->pos.tok_pos, cm->pos.tok_seq, w.i0, w.n, type, cm->n_swa);
    }

    // a chunked graph: the attention of the n_q query rows of a tail layer, after its raw KV is written, as
    // build_attention_v41_chunked computes it for the whole ubatch. Every chunk of plan.q_chunks[il] reads the device
    // masks, the compressed token keys and its raw window by its ubatch rows, and q, qr, cur_q, pos_q and the selection
    // by its rows among the query rows. The compressed width stays that of the ubatch. An index source carries the top-k
    // (and the candidate keep) of the query rows. Returns the joined [n_embd_head*n_head, n_q], before k_rot is undone.
    ggml_tensor * attention_chunked(
            const llama_model & model,
            llm_graph_input_dsv4 * inp_dsv4,
            const llm_graph_input_dsv4::comp_input * inp_comp,
            ggml_tensor * qr,
            ggml_tensor * cur_q,
            ggml_tensor * pos_q,
            ggml_tensor * q,
            ggml_tensor * raw_k,
            dsv41_top_k_carry & top_k_carry,
            float kq_scale,
            int il) {
        ggml_context * ctx0 = g.ctx0;
        const auto & hparams = g.hparams;

        const auto & layer = model.layers[il];
        llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
        llm_graph_input_chunk_mask * cm = g.inp_cm;

        const int64_t n_q = plan.n_q(il);
        const int64_t q0  = g.n_tokens - n_q;

        const std::vector<llm_graph_chunk_window> & qc = plan.q_chunks[il];

        GGML_ASSERT(cm != nullptr && g.inp_dch != nullptr && !qc.empty());
        GGML_ASSERT(q->ne[2] == n_q && cur_q->ne[1] == n_q && pos_q->ne[0] == n_q);

        // the host masks stay out of a chunked graph; only their shapes and types are read
        const int64_t   n_raw     = inp_attn->self_kq_mask->ne[0];
        const ggml_type type_mask = inp_attn->self_kq_mask->type;

        // raw chunk: every chunk reads only its raw window
        const bool win = cm->ops.window != nullptr;

        const bool index_source = inp_comp != nullptr && hparams.dsv41_is_index_source(il);
        const bool cand_source  = index_source && il == hparams.candidate_source_layer;

        // the selection of the query rows: the carry holds the last rows of the ubatch, at least the query rows
        dsv41_top_k_carry carry_q;
        if (inp_comp != nullptr) {
            if (!index_source) {
                GGML_ASSERT(top_k_carry.top_k != nullptr && top_k_carry.il == hparams.dsv41_topk_source[il] &&
                        "a compressed layer needs the top-k of its index source");
            }
            carry_q = dsv41_carry_tail(ctx0, top_k_carry, n_q);
        }

        ggml_tensor * k_all   = raw_k;
        ggml_tensor * comp_k  = nullptr;
        ggml_tensor * raw_idx = nullptr;
        ggml_tensor * ninf    = nullptr;
        int64_t       n_comp  = 0;

        if (inp_comp != nullptr) {
            comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

            n_comp = inp_comp->kq_mask->ne[0];
            GGML_ASSERT(n_comp > 0);
            GGML_ASSERT(n_comp <= comp_k->ne[2]);

            comp_k = ggml_view_4d(ctx0, comp_k,
                    comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
                    comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
            g.cb(comp_k, "comp_k", il);

            // once for the layer
            if (!win) {
                k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
                g.cb(k_all, "k_all", il);
                ggml_build_forward_expand(g.gf, k_all);
            }

            raw_idx = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) n_raw, 1.0f), GGML_TYPE_I32);
            ninf    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, type_mask, 1, 1, 1, 1), -INFINITY);
        }

        // the causal mask the indexer of a chunk reads, as in build_attention_v41_chunked
        const ggml_type type_idx = g.cparams.fused_lid && !cand_source ? GGML_TYPE_F16 : GGML_TYPE_F32;

        llm_graph_join_acc outs(ctx0, g.gf, 1, n_q);
        llm_graph_join_acc top_ks(ctx0, g.gf, 1, n_q);
        llm_graph_join_acc keeps(ctx0, g.gf, 2, n_q);

        int64_t i_next = q0;
        for (const llm_graph_chunk_window & w : qc) {
            GGML_ASSERT(w.i0 == i_next && w.n > 0);
            i_next += w.n;

            // the rows of the chunk among the query rows
            const int64_t i0 = w.i0 - q0;

            ggml_tensor * k_cur = k_all;
            if (win) {
                k_cur = llm_graph_rows(ctx0, raw_k, 2, w.j0, w.nj);
                if (inp_comp != nullptr) {
                    k_cur = ggml_concat(ctx0, k_cur, comp_k, 2);
                    g.cb(k_cur, "k_all", il);
                }
            }

            ggml_tensor * mask     = nullptr;
            int64_t       n_kv_max = 0;

            if (inp_comp == nullptr) {
                // no compressed stream, so this layer sees only its own window
                mask = raw_chunk_mask(w, type_mask);
            } else {
                ggml_tensor * top_k = nullptr;
                if (index_source) {
                    ggml_tensor * comp_mask = g.build_comp_chunk_mask(inp_dsv4, *inp_comp, w.i0, w.n, type_idx);
                    ggml_tensor * keep      = nullptr;

                    top_k = g.build_indexer_top_k(model, inp_dsv4, *inp_comp, qr, cur_q, pos_q, carry_q, il,
                            i0, w.n, comp_mask, &keep);

                    top_ks.add(top_k, i0);
                    if (keep != nullptr) {
                        keeps.add(keep, i0);
                    }
                } else {
                    top_k = llm_graph_rows(ctx0, carry_q.top_k, 1, i0, w.n);
                }

                mask = g.build_chunk_kq_mask(raw_chunk_mask(w, GGML_TYPE_F32), top_k,
                        g.build_comp_chunk_key(inp_dsv4, *inp_comp, w.i0, w.n), llm_graph_rows(ctx0, raw_idx, 0, 0, w.nj),
                        ninf, n_comp, il);

                n_kv_max = std::min<int64_t>(w.nj, hparams.n_swa) + top_k->ne[0];
            }

            ggml_tensor * out = g.build_attn_mha(llm_graph_rows(ctx0, q, 2, i0, w.n), k_cur, k_cur, nullptr, mask,
                    layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);

            // finish this chunk before the next one starts, so the allocator reuses its intermediates
            ggml_build_forward_expand(g.gf, out);
            outs.add(out, i0);
        }
        GGML_ASSERT(i_next == g.n_tokens);

        if (index_source) {
            carry_q.top_k = top_ks.get();
            carry_q.il    = (int32_t) il;
            ggml_build_forward_expand(g.gf, carry_q.top_k);

            if (cand_source) {
                // build_candidate_keep gives every chunk a keep or none at all, by the compressed length only
                carry_q.cand_keep = keeps.empty() ? nullptr : keeps.get();
                if (carry_q.cand_keep != nullptr) {
                    ggml_build_forward_expand(g.gf, carry_q.cand_keep);
                }
            }

            top_k_carry = carry_q;
        }

        // joined over the query rows
        return outs.get();
    }

    // RANMA_DSV41_TAIL_POISON: the raw cells of the ubatch rows before the K rows of layer il, which the plan leaves
    // unwritten, get the poison value, through the write indices of their tokens
    void poison(llm_graph_input_dsv4_raw * inp_attn, int il) const {
        const int64_t n_skip = g.n_tokens - plan.rows[il].n_k;
        if (!plan.poison_on || n_skip <= 0) {
            return;
        }

        const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

        ggml_tensor * k_idxs = inp_attn->get_k_idxs();
        GGML_ASSERT(k_idxs->ne[0] == g.n_tokens && (int64_t) mctx_raw->get_n_write() == g.n_tokens);
        k_idxs = llm_graph_rows(g.ctx0, k_idxs, 0, 0, n_skip);

        ggml_tensor * p = ggml_fill(g.ctx0,
                ggml_new_tensor_3d(g.ctx0, GGML_TYPE_F32, g.hparams.n_embd_head_k(), 1, n_skip), plan.poison);
        g.cb(p, "tail_poison", il);

        ggml_build_forward_expand(g.gf, mctx_raw->cpy_k(g.ctx0, p, k_idxs, il));
    }

    // a tail layer none of whose query rows is in the ubatch (the output lies in a later one): the KV source over every
    // row on il_first, and the raw KV of its K rows, which the queries of the later ubatches read. Nothing else.
    void write_only(
            const llama_model & model,
            llm_graph_input_dsv4 * inp_dsv4,
            ggml_tensor * cur,
            ggml_tensor * inp_pos,
            int il) {
        ggml_context * ctx0 = g.ctx0;
        const auto & hparams = g.hparams;

        const auto & layer = model.layers[il];
        llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

        const int64_t n_embd_head      = hparams.n_embd_head_k();
        const int64_t n_embd_head_rope = hparams.n_rot();
        const int64_t ratio            = hparams.dsv4_compress_ratios[il];

        const int64_t n_tokens = g.n_tokens;
        const int64_t n_in     = cur->ne[1];
        const int64_t n_k      = plan.rows[il].n_k;

        GGML_ASSERT(hparams.is_swa(il));
        GGML_ASSERT(plan.n_q(il) == 0 && n_k <= n_in && n_in <= n_tokens);
        GGML_ASSERT(il == plan.il_first ? n_in == n_tokens : (n_in == n_k && n_k > 0));

        if (ratio != 0 && hparams.dsv41_is_kv_source(il)) {
            const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

            const llm_graph_input_dsv4::comp_input * inp_comp = use_csa ? &inp_dsv4->get_csa() : &inp_dsv4->get_hca();

            const llama_dsv4_comp_state * comp_state = use_csa
                ? inp_dsv4->mctx->get_csa_state()
                : inp_dsv4->mctx->get_hca_state();

            GGML_ASSERT(inp_comp->kq_mask && "a compressed layer needs a plan for its ratio");

            if (inp_comp->state_pos) {
                GGML_ASSERT(n_in == n_tokens && "a KV source of the tail compresses every row");
                g.build_kv_source(model, inp_dsv4, *inp_comp, comp_state, cur, il);
            }
        }

        poison(inp_attn, il);

        if (n_k == 0) {
            return;
        }

        const dsv41_rope_cfg rc = g.rope_cfg(il);

        ggml_tensor * cur_k = llm_graph_rows(ctx0, cur,     1, n_in - n_k,     n_k);
        ggml_tensor * pos_k = llm_graph_rows(ctx0, inp_pos, 0, n_tokens - n_k, n_k);

        ggml_tensor * kv = g.build_lora_mm(layer.wkv, cur_k);
        kv = g.build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_k);
        kv = ggml_rope_ext(ctx0, kv, pos_k, nullptr, n_embd_head_rope, g.rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
        g.cb(kv, "kv", il);

        if (inp_attn->self_k_rot) {
            kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot);
        }
        ggml_build_forward_expand(g.gf, kv);

        const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

        // as in attention(): row i of the write indices is token i
        ggml_tensor * k_idxs = inp_attn->get_k_idxs();
        GGML_ASSERT(k_idxs->ne[0] == n_tokens && (int64_t) mctx_raw->get_n_write() == n_tokens);
        k_idxs = llm_graph_rows(ctx0, k_idxs, 0, n_tokens - n_k, n_k);

        ggml_build_forward_expand(g.gf, mctx_raw->cpy_k(ctx0, kv, k_idxs, il));
    }

    ggml_tensor * attention(
            const llama_model & model,
            llm_graph_input_dsv4 * inp_dsv4,
            ggml_tensor * cur,
            ggml_tensor * inp_pos,
            dsv41_top_k_carry & top_k_carry,
            int il) {
        ggml_context * ctx0 = g.ctx0;
        const auto & hparams = g.hparams;

        const auto & layer = model.layers[il];
        llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

        const int64_t n_embd_head      = hparams.n_embd_head_k();
        const int64_t n_embd_head_rope = hparams.n_rot();
        const int64_t ratio            = hparams.dsv4_compress_ratios[il];

        const int64_t n_tokens = g.n_tokens;
        const int64_t n_in     = cur->ne[1];
        const int64_t n_k      = plan.rows[il].n_k;
        const int64_t n_q      = plan.n_q(il);

        GGML_ASSERT(n_embd_head == g.n_embd_head_v);
        GGML_ASSERT(g.n_head % hparams.dsv4_o_group_count == 0);
        GGML_ASSERT(hparams.is_swa(il));
        GGML_ASSERT(0 < n_q && n_q <= n_k && n_k <= n_in && n_in <= n_tokens);
        GGML_ASSERT(il == plan.il_first ? n_in == n_tokens : n_in == n_k);

        const dsv41_rope_cfg rc = g.rope_cfg(il);

        ggml_tensor * cur_k = llm_graph_rows(ctx0, cur,     1, n_in - n_k,     n_k);
        ggml_tensor * cur_q = llm_graph_rows(ctx0, cur,     1, n_in - n_q,     n_q);
        ggml_tensor * pos_k = llm_graph_rows(ctx0, inp_pos, 0, n_tokens - n_k, n_k);
        ggml_tensor * pos_q = llm_graph_rows(ctx0, inp_pos, 0, n_tokens - n_q, n_q);

        ggml_tensor * qr = g.build_lora_mm(layer.wq_a, cur_q);
        qr = g.build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        g.cb(qr, "qr", il);

        ggml_tensor * q = g.build_lora_mm(layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, g.n_head, n_q);
        q = ggml_rope_ext(ctx0, q, pos_q, nullptr, n_embd_head_rope, g.rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        q = ggml_rope_set_offset(q, n_embd_head - n_embd_head_rope);
        g.cb(q, "q", il);

        ggml_tensor * kv = g.build_lora_mm(layer.wkv, cur_k);
        kv = g.build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_k);
        kv = ggml_rope_ext(ctx0, kv, pos_k, nullptr, n_embd_head_rope, g.rope_type, rc.n_ctx_orig,
                rc.base, rc.scale, rc.ext_factor, rc.attn_factor, rc.beta_fast, rc.beta_slow);
        kv = ggml_rope_set_offset(kv, n_embd_head - n_embd_head_rope);
        g.cb(kv, "kv", il);

        const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

        // the compressed stream: the compressor of a source over every row, the selection for the query rows
        const llm_graph_input_dsv4::comp_input * inp_comp = nullptr;
        llm_graph_input_dsv4::comp_input comp_q;
        ggml_tensor * top_k = nullptr;

        // a chunked graph has no host masks; the source and the raw KV are the same, the rest is per query chunk
        const bool chunked = plan.c > 0;
        GGML_ASSERT(chunked == (g.inp_cm != nullptr));

        if (ratio != 0) {
            const bool use_csa = (uint32_t) ratio == inp_dsv4->mctx->get_csa_state()->get_ratio();

            inp_comp = use_csa ? &inp_dsv4->get_csa() : &inp_dsv4->get_hca();

            const llama_dsv4_comp_state * comp_state = use_csa
                ? inp_dsv4->mctx->get_csa_state()
                : inp_dsv4->mctx->get_hca_state();

            GGML_ASSERT(inp_comp->kq_mask && "a compressed layer needs a plan for its ratio");

            if (hparams.dsv41_is_kv_source(il) && inp_comp->state_pos) {
                GGML_ASSERT(n_in == n_tokens && "a KV source of the tail compresses every row");
                g.build_kv_source(model, inp_dsv4, *inp_comp, comp_state, cur, il);
            }
        }

        if (inp_comp != nullptr && !chunked) {
            // the plan of the query rows: the causal mask narrowed to them, everything else as is
            comp_q = *inp_comp;
            comp_q.kq_mask = mask(inp_comp->kq_mask, n_q);

            if (hparams.dsv41_is_index_source(il)) {
                // the candidate keep of an earlier source is narrowed with the queries; a new one has their rows
                dsv41_top_k_carry carry_q = dsv41_carry_tail(ctx0, top_k_carry, n_q);
                carry_q.top_k = g.build_indexer_top_k(model, inp_dsv4, comp_q, qr, cur_q, pos_q, carry_q, il);
                carry_q.il    = (int32_t) il;
                top_k_carry   = carry_q;
                top_k = top_k_carry.top_k;
            } else {
                GGML_ASSERT(top_k_carry.top_k != nullptr && top_k_carry.il == hparams.dsv41_topk_source[il] &&
                        "a compressed layer needs the top-k of its index source");
                top_k = dsv41_carry_tail(ctx0, top_k_carry, n_q).top_k;
            }
            GGML_ASSERT(top_k->ne[1] == n_q);
        }

        ggml_tensor * k_rot = inp_attn->self_k_rot;
        if (k_rot) {
            q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }

        ggml_build_forward_expand(g.gf, q);
        ggml_build_forward_expand(g.gf, kv);

        const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

        // the write indices of the K rows: one stream, and the write ubatch is the ubatch itself (one sequence, no
        // shared tokens), so row i of the indices is token i
        ggml_tensor * k_idxs = inp_attn->get_k_idxs();
        GGML_ASSERT(k_idxs->ne[0] == n_tokens && (int64_t) mctx_raw->get_n_write() == n_tokens);
        k_idxs = llm_graph_rows(ctx0, k_idxs, 0, n_tokens - n_k, n_k);

        ggml_build_forward_expand(g.gf, mctx_raw->cpy_k(ctx0, kv, k_idxs, il));
        poison(inp_attn, il);

        ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
        g.cb(raw_k, "raw_k", il);

        if (chunked) {
            ggml_tensor * out = attention_chunked(model, inp_dsv4, inp_comp, qr, cur_q, pos_q, q, raw_k,
                    top_k_carry, kq_scale, il);
            if (k_rot) {
                out = llama_mul_mat_hadamard(ctx0, out, k_rot);
            }
            g.cb(out, "attn_out_raw", il);

            return g.build_attention_tail(model, out, pos_q, n_q, il);
        }

        ggml_tensor * raw_mask = mask(inp_attn->get_kq_mask(), n_q);

        ggml_tensor * out = nullptr;

        if (inp_comp == nullptr) {
            out = g.build_attn_mha(q, raw_k, raw_k, nullptr, raw_mask, layer.attn_sinks, nullptr, 0, kq_scale, il);
        } else {
            ggml_tensor * comp_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);

            const int64_t n_comp = inp_comp->kq_mask->ne[0];
            GGML_ASSERT(n_comp > 0);
            GGML_ASSERT(n_comp <= comp_k->ne[2]);

            comp_k = ggml_view_4d(ctx0, comp_k,
                    comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
                    comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);
            g.cb(comp_k, "comp_k", il);

            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
            g.cb(k_all, "k_all", il);

            ggml_tensor * comp_mask = g.build_top_k_mask(comp_q.kq_mask, top_k, "comp_top_k_mask", il);

            ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
            g.cb(kq_mask, "kq_mask", il);

            const int64_t n_kv_max = std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0];

            out = g.build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks,
                    nullptr, n_kv_max, kq_scale, il);
        }
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        g.cb(out, "attn_out_raw", il);

        return g.build_attention_tail(model, out, pos_q, n_q, il);
    }
};

llama_model_deepseek41::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & pmodel = static_cast<const llama_model_deepseek41 &>(model);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    // ranma: device-side chunk masks (the attention chunk), then the host masks stay out of the graph
    build_inp_dsv4_chunk(inp_dsv4);
    if (inp_cm == nullptr) {
        ggml_build_forward_expand(gf, inp_attn->self_kq_mask);
    }

    // ranma: the tail prefill (RANMA_DSV41_TAIL_DIAG). Its last layer computes the output row alone, so there is
    // nothing to gather and no output ids. A chunked graph chunks the query rows of its tail layers by its chunk size.
    dsv41_tail_chunking tail_chunking;
    if (inp_cm != nullptr) {
        tail_chunking.c   = inp_cm->c_req;
        tail_chunking.ops = inp_cm->ops;
        // a graph of one chunk (llm_graph_chunk_mask_one) reads every raw cell, like build_attention_v41_chunked; so do its tail layers
        if (inp_cm->windows.empty()) { tail_chunking.ops.window = nullptr; }
    }
    const dsv41_tail_plan tail = dsv41_tail_plan_make(pmodel, cparams, ubatch, n_outputs, !samplers.empty(), mctx, tail_chunking);
    dsv41_tail_log(tail);
    if (tail.requested) {
        res->add_input(std::make_unique<llm_graph_input_dsv41_tail>(pmodel, tail, mctx, tail_chunking));
    }
    dsv41_tail_builder tail_attn { *this, tail, {} };

    ggml_tensor * inp_out_ids = tail.active() ? nullptr : build_inp_out_ids();

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // Layer 0 has no previous sublayer to take a mix from, so the reference hands it a one-hot
    // that selects the first copy.
    ggml_tensor * pre_mix = ggml_concat(ctx0,
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1,      n_tokens), 1.0f),
            ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc - 1, n_tokens), 0.0f), 0);
    cb(pre_mix, "hc_pre_init", -1);

    // the positions picked by the most recent index source, reused by the layers that follow it
    dsv41_top_k_carry top_k_carry;

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv41_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        // the engram sits before the block and writes straight into the stream
        if (pmodel.engram_index(il) >= 0) {
            inpL = build_engram(model, inpL, build_inp_engram(model, il), il);
            cb(inpL, "engram_out", il);
        }

        // ranma: a tail layer reads n_in rows (every row on the first one) and carries on with the rows of its
        // attention output (O for EXACT, K for DIAG); the hyper-connections are per row, so they follow the rows: the
        // incoming mix collapses the n_in rows for the attention, the attention mixes are made for its output rows
        const bool in_tail = tail.active() && il >= tail.il_first;
        GGML_ASSERT(!in_tail || il == tail.il_first || pmodel.engram_index(il) < 0);

        // the output lies in a later ubatch and no query row of this layer is in this one: the KV source and the raw
        // KV of its rows, and the layers after it have no rows at all
        if (in_tail && tail.n_q(il) == 0) {
            cur = build_hc_pre(inpL, pre_mix, il);
            cb(cur, "hc_attn_pre", il);

            cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            tail_attn.write_only(model, inp_dsv4, cur, inp_pos, il);
            break;
        }

        const int64_t n_in  = inpL->ne[2];
        const int64_t n_att = in_tail ? tail.n_q(il) : n_in;

        ggml_tensor * inp_att  = llm_graph_rows(ctx0, inpL, 2, n_in - n_att, n_att);
        ggml_tensor * residual = inp_att;
        ggml_tensor * attn_pre = nullptr;
        ggml_tensor * post     = nullptr;
        ggml_tensor * comb     = nullptr;

        // this sublayer's mixes are for the next one, so the collapse uses the incoming mix
        build_hc_mixes(inp_att,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &attn_pre, &post, &comb, il);

        cur = build_hc_pre(inpL, pre_mix, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = in_tail
            ? tail_attn.attention(model, inp_dsv4, cur, inp_pos, top_k_carry, il)
            : build_attention_v41(model, inp_dsv4, cur, inp_pos, top_k_carry, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        if (in_tail && tail.rows[il].n_o < n_att) {
            // DIAG: the attention ran over every input row, the FFN takes the rows the next layer reads
            const int64_t n_o = tail.rows[il].n_o;
            inpL     = llm_graph_rows(ctx0, inpL,     2, n_att - n_o, n_o);
            attn_pre = llm_graph_rows(ctx0, attn_pre, 1, n_att - n_o, n_o);
        }

        residual = inpL;

        // the FFN mix is what the next layer's attention collapses with
        build_hc_mixes(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &pre_mix, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl if the input is from mtmd
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv41_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (tail.active() && !tail.has_logits()) {
        // a ubatch before the one with the output, or the end of a call without outputs: the caches are written, there
        // are no logits
        GGML_ASSERT(n_outputs == 0);
        return;
    }

    if (tail.active()) {
        // the last tail layer computed the output row and nothing else
        GGML_ASSERT(n_outputs == 1 && inpL->ne[2] == 1 && pre_mix->ne[1] == 1);
    } else if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
        pre_mix = ggml_get_rows(ctx0, pre_mix, inp_out_ids);
    }

    // The last layer's FFN mix is the one nothing has consumed, and it collapses the copies here.
    // This is what a learned hyper-connection head does in V4, which is why this model has none.
    cur = build_hc_pre(inpL, pre_mix, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
