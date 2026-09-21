#include "common.cuh"

void ggml_cuda_op_top_k_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
