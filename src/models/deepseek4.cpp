#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-graph-chunk.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    if (hparams.n_layer_nextn > 0) {
        const uint32_t n_layer_main = hparams.n_layer_all - hparams.n_layer_nextn;
        const std::string mtp_probe = "blk." + std::to_string(n_layer_main) + ".nextn.eh_proj.weight";
        if (ml.get_weight(mtp_probe.c_str()) == nullptr) {
            hparams.n_layer_nextn = 0;
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    // DeepSeek-V4.1 routes every layer through the gate and omits the key, leaving the count at 0
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count, false);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    // tokens of an image span attend bidirectionally to the whole span, the window only applies to older tokens
    // ref: get_window_topk_idxs_visible in the reference impl
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_FULL;
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, 0);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        // for wo_a, the shape in the file is (n_head * n_embd_head / o_groups, o_lora_rank*o_groups)
        // so we reshape here, to avoid reshaping the tensor in the graph
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        // vision variant only: routing bias for image tokens
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, flags | TENSOR_NOT_REQUIRED);
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd},             flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd},             flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab},    TENSOR_NOT_REQUIRED | flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd},             TENSOR_NOT_REQUIRED | flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;

// One ggml_dsv4_compress op replaces the gather + reorder + softmax + weighted-sum
// chain of every compressor call.

// Small-kernel fusions of the DeepSeek V4 decode graph, shared with the CUDA/HIP backend, always on. The graph
// side orders the APE row gather ahead of the score matmul and the compressor source concats next to the compressor.

// Expand the compressor inputs of one layer in the order the backend fusions expect:
// the APE gather first, so that the score matmul is directly followed by its APE add, and
// all state inputs before the concats, so that the two concats sit next to the compressor.
static void dsv4_fusion3_expand_state(
        ggml_cgraph * gf,
        ggml_tensor * state_kv,
        ggml_tensor * state_score,
        ggml_tensor * ape_rows) {
    ggml_build_forward_expand(gf, ape_rows);
    ggml_build_forward_expand(gf, state_kv);
    ggml_build_forward_expand(gf, state_score);
}

static void dsv4_fusion3_expand_restored(ggml_cgraph * gf, ggml_tensor * restored_kv, ggml_tensor * restored_score) {
    ggml_build_forward_expand(gf, restored_kv);
    ggml_build_forward_expand(gf, restored_score);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_mean(ggml_tensor * x) const {
    const int64_t hc = x->ne[1];

    ggml_tensor * acc = ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t s = 1; s < hc; ++s) {
        acc = ggml_add(ctx0, acc, ggml_view_2d(ctx0, x, x->ne[0], x->ne[2], x->nb[2], s*x->nb[1]));
    }
    return ggml_scale(ctx0, acc, 1.0f/hc);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

// derive the pre / post / comb coefficients from the stream, without collapsing it
// DeepSeek-V4 always collapses right away, V4.1 hands `pre` to the next sublayer instead
void llama_model_deepseek4::graph::build_hc_mixes(
        ggml_tensor *  x,
        ggml_tensor *  hc_fn,
        ggml_tensor *  hc_scale,
        ggml_tensor *  hc_base,
        ggml_tensor ** pre,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    if (cparams.fused_dsv4_hc_coef) {
        // one kernel produces pre, post and comb in the [(2 + hc)*hc, n_tokens] layout of mixes
        ggml_tensor * coef = ggml_dsv4_hc_coef(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COEF, coef, il});
        cb(coef, "hc_coef", il);

        *pre  = dsv4_view_2d(ctx0, coef, hc, nt, 0);
        *post = dsv4_view_2d(ctx0, coef, hc, nt, hc);
        *comb = ggml_view_3d(ctx0, coef, hc, hc, nt,
                dsv4_elem_offset(coef, hc), coef->nb[1], dsv4_elem_offset(coef, 2*hc));

        cb(*pre,  "hc_pre",  il);
        cb(*post, "hc_post", il);
        cb(*comb, "hc_comb", il);

        return;
    }

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    *pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    *pre = dsv4_hc_affine(ctx0, *pre, scale_pre, base_pre);
    *pre = ggml_sigmoid(ctx0, *pre);
    *pre = ggml_scale_bias(ctx0, *pre, 1.0f, hparams.dsv4_hc_eps);
    cb(*pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    ggml_tensor * pre = nullptr;
    build_hc_mixes(x, hc_fn, hc_scale, hc_base, &pre, post, comb, il);

    return build_hc_pre(x, pre, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_pre(x, pre, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il,
        ggml_tensor ** pre_rope) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == ratio*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    GGML_ASSERT(kv_state->ne[0] == n_embd_head);
    GGML_ASSERT(score_state->ne[0] == n_embd_head);

    ggml_tensor * comp = nullptr;

    if (cparams.fused_dsv4_compress) {
        comp = ggml_dsv4_compress(ctx0, kv_state, score_state, state_read_idxs, ratio, false);
        cb(comp, name, il);

        res->add_fused_node({LLM_FUSED_OP_DSV4_COMPRESS, comp, il});
    } else {
        ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, ratio, n_blocks);
        cb(kv, name, il);

        ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
        score = ggml_reshape_3d(ctx0, score, n_embd_head, ratio, n_blocks);
        cb(score, name, il);

        ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
        ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx0, scores);
        comp = ggml_mul(ctx0, values, weights);
        comp = ggml_sum_rows(ctx0, comp);
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
        cb(comp, name, il);
    }

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    // DeepSeek-V4.1 derives its index keys from the pooled latent before it is rotated, because
    // the rotated form is what goes into the cache.
    if (pre_rope) {
        *pre_rope = comp;
    }

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * comp = nullptr;

    if (cparams.fused_dsv4_compress) {
        // the zero / -INFINITY sentinel row for a missing previous segment is
        // implicit in the op: an index outside [0, kv_state->ne[1]) reads as 0 / -INFINITY
        comp = ggml_dsv4_compress(ctx0, kv_state, score_state, state_read_idxs, (int32_t) ratio, true);
        cb(comp, name, il);

        GGML_ASSERT(comp->ne[0] == n_embd_head);
        GGML_ASSERT(comp->ne[2] == n_blocks);

        res->add_fused_node({LLM_FUSED_OP_DSV4_COMPRESS, comp, il});
    } else {
        kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
        score_state = dsv4_append_zero_row(ctx0, score_state, true);

        const int64_t n_read = ratio*n_blocks;

        ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, state_read_idxs);
        ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

        ggml_tensor * kv_prev = ggml_cont(ctx0,
                ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
        kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
        cb(kv_prev, name, il);

        ggml_tensor * score_prev = ggml_cont(ctx0,
                ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
        score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
        cb(score_prev, name, il);

        ggml_tensor * kv_cur = ggml_cont(ctx0,
                ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                    n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
        kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

        ggml_tensor * score_cur = ggml_cont(ctx0,
                ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                    n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
        score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

        ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
        ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

        values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
        scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx0, scores);
        comp = ggml_mul(ctx0, values, weights);
        comp = ggml_sum_rows(ctx0, comp);
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
        cb(comp, name, il);
    }

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    comp = ggml_rope_ext(ctx0, comp, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    comp = ggml_rope_set_offset(comp, n_embd_head_nope);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il,
        int64_t i0,
        int64_t n,
        ggml_tensor * lid_mask) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();

    // a chunk: its rows of the whole-ubatch tensors (inp_pos through the shared view cache, it is a graph input)
    if (n >= 0) {
        GGML_ASSERT(inp_cm != nullptr && lid_mask != nullptr);
        qr      = llm_graph_rows(ctx0, qr,  1, i0, n);
        cur     = llm_graph_rows(ctx0, cur, 1, i0, n);
        inp_pos = inp_cm->rows.get(ctx0, inp_pos, 0, i0, n, false);
    }
    ggml_tensor * lid_kq_mask = lid_mask ? lid_mask : inp_lid.kq_mask;
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    indexer_q = ggml_rope_ext(ctx0, indexer_q, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    indexer_q = ggml_rope_set_offset(indexer_q, n_embd_indexer_head_nope);
    cb(indexer_q, "lid_q_rope", il);

    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = lid_kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid) {
        indexer_score = ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, lid_kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, lid_kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);

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
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    // when the raw and the compressed K of the layer share one allocation, [raw | compressed] is a
    // plain view of that allocation and the per-token copy of the whole compressed cache disappears
    ggml_tensor * k_all = raw_k->ne[3] == 1 ?
        inp_dsv4->mctx->get_csa()->get_k_joint(ctx0, il, (uint32_t) raw_k->ne[2], (uint32_t) n_csa) : nullptr;
    if (k_all == nullptr) {
        k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    }
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * csa_mask = build_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    cb(kq_mask, "csa_lid_kq_mask", il);

    const int64_t n_kv_max = std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + top_k->ne[0];
    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, n_kv_max, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

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
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = raw_k->ne[3] == 1 ?
        inp_dsv4->mctx->get_hca()->get_k_joint(ctx0, il, (uint32_t) raw_k->ne[2], (uint32_t) n_hca) : nullptr;
    if (k_all == nullptr) {
        k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    }
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, 0, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    // ranma: a chunked graph builds the query and the output projection per chunk (build_attention_chunked)
    const bool chunked = inp_dsv4 != nullptr && inp_cm != nullptr;

    // the query of the rows of qr_rows at the positions pos
    auto build_q = [&](ggml_tensor * qr_rows, ggml_tensor * pos) {
        ggml_tensor * q = build_lora_mm(layer.wq_b, qr_rows);
        q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, qr_rows->ne[1]);
        if (!hparams.dflash_dsv41) {
            // V4 normalizes each head again after wq_b; a V4.1 draft normalizes only the low-rank part
            q = ggml_rms_norm(ctx0, q, norm_rms_eps);
            cb(q, "q_norm", il);
        }

        q = ggml_rope_ext(ctx0, q, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        q = ggml_rope_set_offset(q, n_embd_head_nope);
        cb(q, "q", il);
        return q;
    };

    ggml_tensor * q = chunked ? nullptr : build_q(qr, inp_pos);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    kv = ggml_rope_set_offset(kv, n_embd_head_nope);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

        dsv4_fusion3_expand_state(gf, hca_state_kv, hca_state_score, ape_rows);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        dsv4_fusion3_expand_state(gf, csa_state_kv, csa_state_score, csa_ape_rows);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        dsv4_fusion3_expand_restored(gf, csa_restored.kv, csa_restored.score);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il));

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        dsv4_fusion3_expand_state(gf, lid_state_kv, lid_state_score, lid_ape_rows);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        dsv4_fusion3_expand_restored(gf, lid_restored.kv, lid_restored.score);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il));

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        dsv4_fusion3_expand_restored(gf, hca_restored.kv, hca_restored.score);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                DSV4_HCA_RATIO,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_dsv4->get_hca().k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, inp_dsv4->get_hca().k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_hca(), hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, inp_dsv4->get_hca().state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_dsv4->get_hca().state_persist_src_idxs);

        hca_state_kv = inp_dsv4->mctx->get_hca_state()->cpy_kv(ctx0,
                hca_persist_kv, inp_dsv4->get_hca().state_persist_dst_idxs, il);
        hca_state_score = inp_dsv4->mctx->get_hca_state()->cpy_score(ctx0,
                hca_persist_score, inp_dsv4->get_hca().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    // the output projection of the attention rows out at the positions pos
    auto build_tail = [&](ggml_tensor * out, ggml_tensor * pos) {
        const int64_t n = pos->ne[0];

        out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n);
        out = ggml_rope_ext_back(ctx0, out, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        out = ggml_rope_set_offset(out, n_embd_head_nope);
        cb(out, "attn_derope", il);

        out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n);
        out = ggml_permute(ctx0, out, 0, 2, 1, 3);
        ggml_tensor * oa = build_mm(layer.wo_a, out);
        cb(oa, "attn_wo_a", il);
        oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
        oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n);

        out = build_lora_mm(layer.wo_b, oa);
        cb(out, "attn_out", il);

        return out;
    };

    if (chunked) {
        return build_attention_chunked(model, inp_dsv4, inp_attn, build_q, build_tail, kv, qr, cur, inp_pos,
                layer.attn_sinks, 1.0f/sqrtf(float(n_embd_head)), il);
    }

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, kv,
                nullptr, layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    GGML_ASSERT(inp_pos->ne[0] == nt);
    return build_tail(out, inp_pos);
}

// ranma: the raw SWA cache of the DSV4 memory context, for llm_graph_input_chunk_mask
static const llama_kv_cache_dsv4_raw_context * dsv4_raw_ctx(const llama_memory_context_i * mctx) {
    return static_cast<const llama_kv_cache_dsv4_context *>(mctx)->get_raw();
}

// ranma: the compressed prefix masks of a chunked DSV4 graph. Compressed row j is visible to token t when j < n_visible(t)
// of its plan (dsv4_set_kq_mask), so a table of -j and the token key -(n_visible - 0.5) give the mask of a chunk through
// llm_graph_chunk_mask, with the sequence column 0 of the single sequence.
class llm_graph_input_dsv4_chunk : public llm_graph_input_i {
public:
    llm_graph_input_dsv4_chunk(const llama_kv_cache_dsv4_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_dsv4_chunk() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ggml_backend_buffer_is_host(iota->buffer));
        float * t = (float *) iota->data;
        for (int64_t j = 0; j < iota->ne[0]; ++j) {
            t[j] = -(float) j;
        }

        set_key(csa_key, mctx->get_csa_plan(*ubatch), ubatch);
        set_key(hca_key, mctx->get_hca_plan(*ubatch), ubatch);
        set_key(lid_key, mctx->get_lid_plan(*ubatch), ubatch);

        if (llm_graph_chunk_mask_check_enabled()) {
            check("csa", csa_key, mctx->get_csa_plan(*ubatch), ubatch);
            check("hca", hca_key, mctx->get_hca_plan(*ubatch), ubatch);
            check("lid", lid_key, mctx->get_lid_plan(*ubatch), ubatch);
        }
    }

    // the host prefix mask of dsv4_set_kq_mask against the device rule (the chunk mask input is set before this one)
    void check(const char * name, ggml_tensor * key, const llama_kv_cache_dsv4_context::comp_plan & plan,
               const llama_ubatch * ubatch) const {
        if (plan.n_kv <= 0) {
            return;
        }
        llm_graph_host_tensor host(GGML_TYPE_F32, plan.n_kv, ubatch->n_tokens);
        llm_graph_host_tensor tab (GGML_TYPE_F32, plan.n_kv, 1);
        float * h = (float *) host.t->data;
        float * t = (float *) tab.t->data;
        for (int64_t j = 0; j < plan.n_kv; ++j) {
            t[j] = -(float) j;
        }
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            for (int64_t j = 0; j < plan.n_kv; ++j) {
                h[i*plan.n_kv + j] = j < plan.n_visible[i] ? 0.0f : -INFINITY;
            }
        }
        llm_graph_chunk_mask_compare(name, host.t, tab.t, key, cm->pos.tok_seq);
    }

    // the LID and CSA prefix masks are the same tensor
    static bool same_prefix(const llama_kv_cache_dsv4_context * mctx, const llama_ubatch & ubatch, bool same_type) {
        const auto & csa = mctx->get_csa_plan(ubatch);
        const auto & lid = mctx->get_lid_plan(ubatch);
        return same_type && csa.n_kv == lid.n_kv && csa.n_visible == lid.n_visible;
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

        const int64_t n_max = std::max({ mctx->get_csa_plan(params.ubatch).n_kv, mctx->get_hca_plan(params.ubatch).n_kv,
                                         mctx->get_lid_plan(params.ubatch).n_kv, (int64_t) 1 });

        bool res = true;
        res &= iota->ne[0] == n_max;
        res &= csa_key->ne[1] == params.ubatch.n_tokens;
        res &= lid_as_csa == same_prefix(mctx, params.ubatch, same_type);

        return res;
    }

    static void set_key(ggml_tensor * key, const llama_kv_cache_dsv4_context::comp_plan & plan, const llama_ubatch * ubatch) {
        GGML_ASSERT(ggml_backend_buffer_is_host(key->buffer));
        GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens && key->ne[1] == (int64_t) ubatch->n_tokens);
        float * k = (float *) key->data;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            k[i] = -((float) plan.n_visible[i] - 0.5f);
        }
    }

    // the [n_rows, n] prefix mask of the token rows [i0, i0 + n)
    ggml_tensor * mask(ggml_context * ctx, llm_graph_input_chunk_mask & cm, ggml_tensor * key, int64_t n_rows,
                       int64_t i0, int64_t n, ggml_type type) {
        ggml_tensor * tab = cm.rows.get(ctx, iota, 0, 0, n_rows, false);
        return llm_graph_chunk_mask(ctx, cm.rows, tab, key, cm.pos.tok_seq, i0, n, type);
    }

    const llama_kv_cache_dsv4_context * mctx;

    llm_graph_input_chunk_mask * cm = nullptr; // its token sequence column, for the check

    bool same_type  = false; // the LID and CSA masks have one type
    bool lid_as_csa = false; // see same_prefix

    ggml_tensor * iota    = nullptr; // F32 [n_max, 1]     -j
    ggml_tensor * csa_key = nullptr; // F32 [1, n_tokens]  -(n_visible - 0.5)
    ggml_tensor * hca_key = nullptr;
    ggml_tensor * lid_key = nullptr;
};

void llama_model_deepseek4::graph::build_inp_dsv4_chunk(llm_graph_input_dsv4 * inp_dsv4) {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsv4_context *>(mctx);

    // the prefix masks hold one sequence in one stream
    if (mctx_cur->get_csa_plan(ubatch).n_stream != 1 || ubatch.n_seqs_unq != 1) {
        return;
    }

    inp_cm = llm_graph_build_inp_chunk_mask(res, ctx0, gf, cparams, hparams, ubatch, mctx,
            llm_graph_chunk_kv_ops_of<llama_kv_cache_dsv4_raw_context, dsv4_raw_ctx>(), hparams.n_swa);
    if (inp_cm == nullptr) {
        return;
    }

    auto inp = std::make_unique<llm_graph_input_dsv4_chunk>(mctx_cur);
    inp->cm = inp_cm;

    const auto & inp_csa = inp_dsv4->get_csa();
    const auto & inp_lid = inp_dsv4->get_lid();
    inp->same_type  = inp_csa.kq_mask && inp_lid.kq_mask && inp_csa.kq_mask->type == inp_lid.kq_mask->type;
    inp->lid_as_csa = llm_graph_input_dsv4_chunk::same_prefix(mctx_cur, ubatch, inp->same_type);

    const int64_t n_max = std::max({ mctx_cur->get_csa_plan(ubatch).n_kv, mctx_cur->get_hca_plan(ubatch).n_kv,
                                     mctx_cur->get_lid_plan(ubatch).n_kv, (int64_t) 1 });

    inp->iota    = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_max, 1);
    inp->csa_key = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    inp->hca_key = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    inp->lid_key = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    for (ggml_tensor * t : { inp->iota, inp->csa_key, inp->hca_key, inp->lid_key }) {
        ggml_set_input(t);
        // set_input fills them, so they stay allocated in every graph
        ggml_build_forward_expand(gf, t);
    }

    inp_dch = (llm_graph_input_dsv4_chunk *) res->add_input(std::move(inp));
}

// ranma: the token key of the plan of inp_comp, one of the three compressed inputs of inp_dsv4
static ggml_tensor * dsv4_chunk_plan_key(llm_graph_input_dsv4 * inp_dsv4, llm_graph_input_dsv4_chunk * inp_dch,
                                         const llm_graph_input_dsv4::comp_input & inp_comp) {
    if (&inp_comp == &inp_dsv4->get_csa()) {
        return inp_dch->csa_key;
    }
    if (&inp_comp == &inp_dsv4->get_hca()) {
        return inp_dch->hca_key;
    }
    GGML_ASSERT(&inp_comp == &inp_dsv4->get_lid() && "not a compressed input of this graph");
    return inp_dch->lid_key;
}

ggml_tensor * llama_model_deepseek4::graph::build_comp_chunk_mask(
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        int64_t i0,
        int64_t n,
        ggml_type type) const {
    GGML_ASSERT(inp_cm != nullptr && inp_dch != nullptr && inp_comp.kq_mask != nullptr);

    ggml_tensor * key = dsv4_chunk_plan_key(inp_dsv4, inp_dch, inp_comp);

    return inp_dch->mask(ctx0, *inp_cm, key, inp_comp.kq_mask->ne[0], i0, n, type);
}

ggml_tensor * llama_model_deepseek4::graph::build_comp_chunk_key(
        llm_graph_input_dsv4 * inp_dsv4,
        const llm_graph_input_dsv4::comp_input & inp_comp,
        int64_t i0,
        int64_t n) const {
    GGML_ASSERT(inp_cm != nullptr && inp_dch != nullptr);

    return inp_cm->rows.get(ctx0, dsv4_chunk_plan_key(inp_dsv4, inp_dch, inp_comp), 1, i0, n, false);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_chunked(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        const dsv4_rows_fn & build_q,
        const dsv4_rows_fn & build_tail,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(inp_cm != nullptr && inp_dch != nullptr);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    const auto & inp_csa = inp_dsv4->get_csa();
    const auto & inp_hca = inp_dsv4->get_hca();
    const auto & inp_lid = inp_dsv4->get_lid();

    // the same choice as build_attention_impl
    const bool csa = ratio == DSV4_CSA_RATIO && inp_csa.kq_mask && inp_lid.kq_mask && inp_lid.k_rot;
    const bool hca = !csa && ratio == DSV4_HCA_RATIO && inp_hca.kq_mask;
    if (!csa && !hca) {
        GGML_ASSERT(hparams.is_swa(il));
    }

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    // the whole ubatch writes the raw cache, then every chunk reads it
    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    ggml_tensor * k_all = raw_k;

    int64_t n_comp = 0;
    if (csa || hca) {
        const auto * comp_ctx = csa ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
        n_comp = (csa ? inp_csa : inp_hca).kq_mask->ne[0];

        ggml_tensor * comp_k = comp_ctx->get_k(ctx0, il);
        GGML_ASSERT(n_comp > 0 && n_comp <= comp_k->ne[2]);
        comp_k = ggml_view_4d(ctx0, comp_k,
                comp_k->ne[0], comp_k->ne[1], n_comp, comp_k->ne[3],
                comp_k->nb[1], comp_k->nb[2], comp_k->nb[3], 0);

        // once for the whole ubatch, as in build_csa_lid_attention / build_hca_attention
        k_all = raw_k->ne[3] == 1 ? comp_ctx->get_k_joint(ctx0, il, (uint32_t) raw_k->ne[2], (uint32_t) n_comp) : nullptr;
        if (k_all == nullptr) {
            k_all = ggml_concat(ctx0, raw_k, comp_k, 2);
        }
        cb(k_all, csa ? "csa_k_all" : "hca_k_all", il);
    }

    const ggml_type type_raw = inp_attn->self_kq_mask->type;
    const int64_t   n_raw    = inp_attn->self_kq_mask->ne[0];

    const llm_graph_chunks chunks(n_tokens);
    GGML_ASSERT(chunks.n > 1);

    std::vector<ggml_tensor *> outs;
    for (int ic = 0; ic < chunks.n; ++ic) {
        const int64_t i0 = chunks.i0(ic);
        const int64_t n  = chunks.nk(ic);

        // the positions of the chunk through the shared view cache, inp_pos is a graph input
        ggml_tensor * pos = inp_cm->rows.get(ctx0, inp_pos, 0, i0, n, false);

        ggml_tensor * q = build_q(llm_graph_rows(ctx0, qr, 1, i0, n), pos);
        if (k_rot) {
            q = llama_mul_mat_hadamard(ctx0, q, k_rot);
        }

        ggml_tensor * mask = ggml_reshape_4d(ctx0, inp_cm->mask(ctx0, i0, n, type_raw), n_raw, n, 1, 1);
        int64_t n_kv_max = 0;

        if (csa) {
            ggml_tensor * lid_mask = inp_dch->mask(ctx0, *inp_cm, inp_dch->lid_key, inp_lid.kq_mask->ne[0], i0, n,
                                                   inp_lid.kq_mask->type);
            ggml_tensor * top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il, i0, n, lid_mask);

            // the CSA and LID plans usually agree (same ratio): then one prefix mask serves both
            ggml_tensor * csa_mask = inp_dch->lid_as_csa ? lid_mask :
                inp_dch->mask(ctx0, *inp_cm, inp_dch->csa_key, n_comp, i0, n, inp_csa.kq_mask->type);
            csa_mask = build_top_k_mask(ggml_reshape_4d(ctx0, csa_mask, n_comp, n, 1, 1), top_k, "csa_top_k_mask", il);

            mask = ggml_concat(ctx0, mask, csa_mask, 0);
            cb(mask, "csa_lid_kq_mask", il);

            n_kv_max = std::min<int64_t>(n_raw, hparams.n_swa) + top_k->ne[0];
        } else if (hca) {
            ggml_tensor * hca_mask = inp_dch->mask(ctx0, *inp_cm, inp_dch->hca_key, n_comp, i0, n, inp_hca.kq_mask->type);

            mask = ggml_concat(ctx0, mask, ggml_reshape_4d(ctx0, hca_mask, n_comp, n, 1, 1), 0);
            cb(mask, "hca_kq_mask", il);
        }

        ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, mask, sinks, nullptr, n_kv_max, kq_scale, il);
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, csa ? "attn_csa_lid" : hca ? "attn_hca" : "attn_raw", il);

        out = build_tail(out, pos);

        // finish this chunk before the next one starts, so the allocator reuses its intermediates
        ggml_build_forward_expand(gf, out);
        outs.push_back(out);
    }

    return llm_graph_join(ctx0, outs, 1);
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    // ranma: device-side chunk masks, then the host masks stay out of the graph
    build_inp_dsv4_chunk(inp_dsv4);
    if (inp_cm == nullptr) {
        ggml_build_forward_expand(gf, inp_attn->self_kq_mask);
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;

        // may apply exp_probs_b_vl is input is from mtmd
        ASSERT_EMBD_OR_TOKEN(ubatch);
        const bool is_media = ubatch.embd != nullptr;
        if (is_media) {
            if (layer.ffn_exp_probs_b_vl) {
                exp_probs_b = layer.ffn_exp_probs_b_vl;
            }
        } else if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
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
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
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
        res->t_layer_inp[n_layer] = build_hc_mean(inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * flat_out = inp_out_ids ? ggml_get_rows(ctx0, flat, inp_out_ids) : flat;

    ggml_tensor * h_nextn = cparams.embeddings_nextn_masked ? flat_out : inpL;
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    if (inp_out_ids) {
        inpL = ggml_reshape_3d(ctx0, flat_out, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


llama_model_deepseek4::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "DEEPSEEK4 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DEEPSEEK4 MTP currently only supports a single MTP block");
    GGML_ASSERT(cparams.nextn_layer_offset >= 0 &&
            cparams.nextn_layer_offset < (int) hparams.n_layer_nextn &&
            "nextn_layer_offset out of range [0, n_layer_nextn)");
    ASSERT_EMBD_OR_TOKEN(ubatch);
    GGML_ASSERT(ubatch.token && "DEEPSEEK4 MTP requires token input");

    const int64_t hc = hparams.dsv4_hc_mult;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) (n_embd*hc) && "DEEPSEEK4 MTP hidden width mismatch");

    const int il = hparams.n_layer() + cparams.nextn_layer_offset;
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd_out());

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_out(), n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    ggml_tensor * h_norm = build_norm(h_state, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens);
    e_norm = ggml_repeat_4d(ctx0, e_norm, n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    cur = build_attention(model, inp_attn, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    GGML_ASSERT((uint32_t) il >= hparams.dsv4_hash_layer_count && "DEEPSEEK4 MTP does not support hash-routed MTP blocks");
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used(),
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    inpL = build_cvec(inpL, il);
    cb(inpL, "mtp_l_out", il);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    ggml_tensor * h_nextn = ggml_get_rows(ctx0, flat, inp_out_ids);
    cb(h_nextn, "h_nextn", -1);
    res->t_h_nextn = h_nextn;

    inpL = ggml_reshape_3d(ctx0, h_nextn, n_embd, hc, n_outputs);

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "mtp_hc_head", -1);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm ? layer.nextn.shared_head_norm : model.output_norm;
    GGML_ASSERT(head_norm_w && "DEEPSEEK4 MTP missing shared head norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DEEPSEEK4 MTP missing LM head");
    cur = build_mm(head_w, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
