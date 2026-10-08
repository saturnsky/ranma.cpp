#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// the input state is row ids[0] of a state cache, read in place instead of through a GET_ROWS copy (single sequence)
struct ggml_cuda_gated_delta_net_state_rows {
    const float *   base;       // first row of the cache
    const int32_t * ids;        // device pointer, row index in ids[0]
    int64_t         row_stride; // elements per row
};

// cache may be null (snapshots then go to the op result as usual)
void ggml_cuda_op_gated_delta_net_fused_state(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_state_rows rows,
                                              const ggml_cuda_gated_delta_net_fused_cache * cache);
