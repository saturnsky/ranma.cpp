#pragma once

#include "common.cuh"

// GGML_OP_MUL_MAT_HAD for F32, F16 and EXL3 weights, had 32/64/128, K and N multiples of 128
bool ggml_cuda_mul_mat_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_mul_mat_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_OP_MOE_FFN_HAD (RDNA4, EXL3 experts): the routed experts of one layer, expert by expert in chunks of rows
bool ggml_cuda_moe_ffn_had_supported(int device, const ggml_tensor * op);

void ggml_cuda_moe_ffn_had(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// false while the expert-major staging (RANMA_EXPERT_MAJOR_STAGE_MIB) is on, for a PCIe-once op (op_params[3]) and while
// the link bytes are counted: their side stream and plan readback are not captured
bool ggml_cuda_moe_ffn_had_graph_ok(const ggml_tensor * op);

// the link bytes of the expert-major op (logical): out[0] staged copies, out[1] in-place host reads, out[2] ops counted,
// out[3] PCIe-once ops, out[4] their chunks, out[5] PCIe-once ops that fell back to the in-place path. enable > 0 turns the
// counting on for every op with host experts, < 0 off, 0 leaves it. Cumulative; synchronizes the device.
bool ggml_cuda_moe_link_stats(int device, int enable, uint64_t * out);

// stage-falsify probe U (diagnostic): op 0 sets the experts per group of the one-chunk MOE_FFN_HAD (0 = off), op 1 turns the
// phase stamps on or off, op 2 reads the phase times (ms) of the last stamped op into out[20], op 3 sets the group of the
// streamed op (RANMA_EXPERT_MAJOR_STREAM; 0 = off, -1 = the environment), op 4 its ring slots (0 = the environment),
// op 5 poisons the ring of the next streamed op, op 6 the two staging buffers of the next PCIe-once op
bool ggml_cuda_moe_group_diag(int op, int a, double * out);
