#include "common.cuh"

#define MMVF_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVF kernels.

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_f(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t * src0_ne, const size_t * src0_nb, int64_t ne11);

// Skinny dense F32 x F32 products with more columns than MMVF_MAX_BATCH_SIZE (RDNA4, instead of hipBLAS).
bool ggml_cuda_should_use_mul_mat_f32_skinny(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_f32_skinny(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// GGML_OP_RELU_SUM_HEADS alone, and fused with the MMVF product mm that produces its src[0] (rsh = the op node).
void ggml_cuda_op_relu_sum_heads(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_mul_mat_vec_f_relu_sum_heads_supported(const ggml_tensor * mm, const ggml_tensor * rsh);
void ggml_cuda_mul_mat_vec_f_relu_sum_heads(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * rsh);
