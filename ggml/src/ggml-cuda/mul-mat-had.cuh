#pragma once

#include "common.cuh"

// GGML_OP_MUL_MAT_HAD for F32, F16 and EXL3 weights, had 32/64/128, K and N multiples of 128
bool ggml_cuda_mul_mat_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_MOE_FFN_HAD (RDNA4, EXL3 experts): the routed experts of one layer, expert by expert in chunks of rows
bool ggml_cuda_moe_ffn_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_moe_ffn_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
