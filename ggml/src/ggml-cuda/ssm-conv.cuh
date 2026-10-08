#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// concat(conv state, x) -> ssm_conv -> silu -> cpy of the trailing windows back into the conv-state cache, as one kernel.
// The windows are the last d_conv-1 columns of the concatenated row, starting at column win_col[k].
#define GGML_CUDA_SSM_CONV_FUSED_MAX_WINDOWS 4
#define GGML_CUDA_SSM_CONV_FUSED_MAX_TOKENS  8

struct ggml_cuda_ssm_conv_state_fused {
    int     n_win;
    int     win_col[GGML_CUDA_SSM_CONV_FUSED_MAX_WINDOWS];
    float * win_dst[GGML_CUDA_SSM_CONV_FUSED_MAX_WINDOWS];
};

void ggml_cuda_op_ssm_conv_state_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * concat,
                                       const ggml_tensor * ssm_conv, const ggml_tensor * silu,
                                       const ggml_cuda_ssm_conv_state_fused & fused);
