#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_coef(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_compress(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// fused paths of the DeepSeek V4 decode graph, matched in ggml-cuda.cu
void ggml_cuda_op_dsv4_hc_post_add(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst);
void ggml_cuda_op_dsv4_compress_concat(ggml_backend_cuda_context & ctx,
        const ggml_tensor * kv_cat, const ggml_tensor * score_cat, ggml_tensor * dst);
void ggml_cuda_op_dsv4_row_copy(ggml_backend_cuda_context & ctx,
        const ggml_tensor * const * get_rows, const ggml_tensor * const * set_rows, int n_pairs);

// low-rank gated residual (Qwen3.8 hyper-connections) of up to GGML_CUDA_HC_GATED_MAX_TOKENS tokens, matched in ggml-cuda.cu
#define GGML_CUDA_HC_GATED_MAX_TOKENS 8

bool ggml_cuda_hc_gated_pre_supported(const ggml_tensor * mm_down, const ggml_tensor * scale, const ggml_tensor * act,
        const ggml_tensor * mm_up, const ggml_tensor * gate, const ggml_tensor * pre);
void ggml_cuda_op_hc_gated_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_down, const ggml_tensor * scale,
        const ggml_tensor * mm_up, ggml_tensor * pre);
