#pragma once

#include "common.cuh"

// GGML_OP_MUL_MAT_HAD for F32, F16 and EXL3 weights, had 32/64/128, K and N multiples of 128
bool ggml_cuda_mul_mat_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_MOE_FFN_HAD (RDNA4, EXL3 experts): the routed experts of one layer, expert by expert in chunks of rows
bool ggml_cuda_moe_ffn_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_moe_ffn_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// false while the expert-major staging (RANMA_EXPERT_MAJOR_STAGE_MIB) is on: its side stream is not captured
bool ggml_cuda_moe_ffn_had_graph_ok(void);

// stage-falsify probe U (diagnostic): op 0 sets the experts per group of the one-chunk MOE_FFN_HAD (0 = off), op 1 turns the
// phase stamps on or off, op 2 reads the phase times (ms) of the last stamped op into out[12]
bool ggml_cuda_moe_group_diag(int op, int a, double * out);
