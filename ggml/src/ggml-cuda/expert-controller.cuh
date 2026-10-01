#pragma once

// Owner of the expert cache state machine for the one routed-expert model a process holds:
//
//   unconfigured --configure--> configured --register_context--> registered --finalize--> installed
//   any state --disable(reason)--> disabled (lookups miss, the model keeps running uncached)
//   installed --release--> unconfigured
//
// The controller composes the pure policy (geometry, plan, score, profile store) with the device
// mechanisms (profiler, L1 arena). Its C ABI is the ggml_expert_iface table returned by
// ggml_backend_cuda_expert_iface(); the two hot-path entry points below are called by the kernels'
// dispatch code and never take a lock.

#include "common.cuh"
#include "expert-l1.cuh"

#include "ggml-expert.h"

#if defined(GGML_USE_HIP)

// Arena and slot table for a routed-expert weight tensor, or an empty lookup when the cache is not
// installed, the tensor is not one of the registered experts, or the op runs on another device.
ggml_cuda_expert_lookup ggml_cuda_expert_lookup_tensor(const ggml_tensor * src0);

// Adds the router selections of `ids` ("ffn_moe_topk-<layer>", I32 [n_used, n_rows]) to the bank
// selected for this compute. No-op unless the cache is installed.
void ggml_cuda_expert_profile_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids);

// The GPU has read the last kind of a routed layer, so the SSD tier may reuse that layer's ring
// slots. No-op unless the tier is built.
void ggml_cuda_expert_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0);

// With the staged SSD tier service, orders the next kernel that reads `src0` after the file reads
// of that expert kind. Called next to every lookup that can hand a kernel ring addresses. No-op
// unless the finite tier is built, the plan leaves experts in the file and the layer was routed
// with enough rows to be staged.
void ggml_cuda_expert_before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0);

// Batched SSD tier service (expert-l2-batch.h): the batch launches per kind of the MUL_MAT_ID that
// reads `src0` for a ubatch of `rows` rows, 0 for one launch as without batching, and the device tables
// the launches read: `batch_of` [expert] the launch of each expert (0 = the first), `lists`
// [batch - 1][capacity] the experts of each batch launch, -1 when empty. A ubatch with batch launches
// runs MMQ whatever its row count.
struct ggml_cuda_expert_batches {
    int launches = 0;
    int capacity = 0;
    const int32_t * batch_of = nullptr;
    const int32_t * lists = nullptr;
};
ggml_cuda_expert_batches ggml_cuda_expert_batch_launches(const ggml_tensor * src0, int64_t rows);
// Before batch launch `batch` (1-based) of `src0`: waits for its reads. After it: reports it done.
void ggml_cuda_expert_batch_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int batch);
void ggml_cuda_expert_batch_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int batch);

// True for the buffer type that exclusive mode allocates the routed expert weights on. Its tensors
// carry logical addresses only: their bytes live in the VRAM arena or in the host arena, so every
// kernel that reads them must resolve them through ggml_cuda_expert_lookup_tensor.
bool ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_type_t buft);

// The versioned function table answered for GGML_EXPERT_IFACE_PROC_NAME.
const ggml_expert_iface * ggml_backend_cuda_expert_iface(void);

// Index of a CUDA/HIP device object, -1 for devices of other backends (defined in ggml-cuda.cu).
int ggml_backend_cuda_dev_index(ggml_backend_dev_t dev);

#else

// The cache is a HIP-only feature, but ggml-cuda.cu asks this question outside the HIP guards.
static inline bool ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

#endif // GGML_USE_HIP
