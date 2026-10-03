#pragma once

#include "common.cuh"

// GGML_OP_MUL_MAT_HAD for F32, F16 and EXL3 weights, had 32/64/128, K and N multiples of 128
bool ggml_cuda_mul_mat_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
