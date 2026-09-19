#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

using llama_adapter_cvec_ptr = std::shared_ptr<llama_adapter_cvec>;

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // optional copy of `b` pre-multiplied by the effective LoRA scale, so that the graph does not
    // need a separate GGML_OP_SCALE node. built by llama_adapter_lora::ensure_scaled_b() when the
    // adapter is attached to a context, never while a graph is being built or run.
    ggml_tensor * b_scaled = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

struct llama_adapter_lora {
    llama_model * model = nullptr;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // pre-scaled `b` copies (see ensure_scaled_b): built at most once per adapter, for the first
    // adapter scale that actually needs them
    bool  scaled_b_done          = false;
    float scaled_b_adapter_scale = 0.0f;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    explicit llama_adapter_lora(llama_model * model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    // build `b_scaled = b*get_scale(alpha, adapter_scale)` for every dense weight whose effective
    // scale is not exactly 1 (a scale of 1 needs no copy: the graph simply drops the scale node).
    // called from llama_context::set_adapters_lora, i.e. never while a graph exists.
    // the copies are built at most once per adapter: if a different adapter scale is used later
    // the graph falls back to the ggml_scale node, so data a live graph may point at is never
    // rewritten and several contexts may share one adapter with different scales.
    void ensure_scaled_b(float adapter_scale);

    uint32_t get_n_nodes() const {
        // a, b, scale, add, 2 x mul_mat; a folded scale needs 5, so this may over-reserve
        return ab_map.size() * 6u;
    }
};

// LLAMA_LORA_FOLD_SCALE (default 1): fold the LoRA scale out of the compute graph.
// LLAMA_LORA_FOLD_SCALE=0 restores the previous graph exactly. read once per process.
bool llama_adapter_lora_fold_scale_enabled();

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
using llama_adapter_loras_ptr = std::unique_ptr<llama_adapter_loras>;
