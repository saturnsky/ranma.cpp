#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

// Shared q8_1 quantization of src1 between MMVQ nodes of one graph.
//
// Several MUL_MATs of a layer read the very same activation tensor, and each of them quantizes it
// again. The plan pass groups such nodes; the first node of a group keeps its q8_1 buffer alive so
// that the rest of the group can reuse it. `allow` is false when the caller cannot guarantee that
// all nodes run on the same stream. Must be paired with ggml_cuda_mmvq_share_q8_end().
void ggml_cuda_mmvq_share_q8_plan(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, bool allow);
void ggml_cuda_mmvq_share_q8_end(ggml_backend_cuda_context & ctx);

// Tell the plan that this MUL_MAT never reaches the backend, so that the group it belongs to keeps
// moving: the cursor passes its entry and the buffer of a finished group is handed back.
void ggml_cuda_mmvq_share_q8_skip(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);

// GGML_CUDA_MMVQ_ID_FOLD_SHARED (on by default, 0 disables) and its one-line report, GGML_CUDA_MMVQ_ID_FOLD_SHARED_LOG=1
bool ggml_cuda_mmvq_id_fold_shared_enabled();
bool ggml_cuda_mmvq_id_fold_shared_log();

// true when the one-token kernel of `routed_type` is compiled with a `shared_type` unit on this arch
bool ggml_cuda_mmvq_id_fold_shared_types(ggml_type routed_type, ggml_type shared_type, int cc);

// Offer the unit to the next launch of this context; the launch reports back whether it took it.
// The caller may only treat the folded nodes as computed after ggml_cuda_mmvq_shared_fold_taken()
// returned true. The unit is defined by ggml_cuda_mmvq_shared_fold in common.cuh.
void ggml_cuda_mmvq_shared_fold_arm(ggml_backend_cuda_context & ctx, const ggml_cuda_mmvq_shared_fold & fold);
void ggml_cuda_mmvq_shared_fold_disarm(ggml_backend_cuda_context & ctx);
bool ggml_cuda_mmvq_shared_fold_taken(const ggml_backend_cuda_context & ctx);
