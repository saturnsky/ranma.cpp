#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"

#if !defined(GGML_USE_MUSA)
// Portable warp vote: HIP returns a 64-bit mask, CUDA a 32-bit one.
#if defined(GGML_USE_HIP)
typedef unsigned long long fattn_sparse_ballot_t;
static __device__ __forceinline__ fattn_sparse_ballot_t fattn_sparse_ballot(bool pred) {
    return __ballot(pred);
}
#else
typedef uint32_t fattn_sparse_ballot_t;
static __device__ __forceinline__ fattn_sparse_ballot_t fattn_sparse_ballot(bool pred) {
    return __ballot_sync(0xFFFFFFFF, pred);
}
#endif // defined(GGML_USE_HIP)

static __device__ __forceinline__ int fattn_sparse_popc(fattn_sparse_ballot_t mask) {
    return sizeof(fattn_sparse_ballot_t) == 4 ? __popc(uint32_t(mask)) : __popcll((unsigned long long) mask);
}

// Cells one block of the compaction kernels votes over. The serial kernel walks a list's cells in
// steps of this size, the parallel pair gives every step a block of its own.
static constexpr int fattn_sparse_block_size      = 256;
static constexpr int fattn_sparse_values_per_lane = 8;
static constexpr int fattn_sparse_chunk           = fattn_sparse_block_size*fattn_sparse_values_per_lane;

// Below this many chunks the serial kernel wins: it needs one launch where the parallel pair needs
// two, and a launch costs more than the chunks it saves.
static constexpr int fattn_sparse_parallel_min_chunks = 5;

// one list per group of ncols1 queries: a column is selected if any query of the group can see it
__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int ncols1, const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = fattn_sparse_values_per_lane;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[fattn_sparse_block_size/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30; i0 += blockDim.x*values_per_lane) {
        fattn_sparse_ballot_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            for (int q = 0; q < q1 - q0 && !selected; ++q) {
                selected = i < ne30 && isfinite(__half2float(mask[q*s31 + i]));
            }
            selected_warp[item] = fattn_sparse_ballot(selected);
            warp_count += fattn_sparse_popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < fattn_sparse_block_size/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const fattn_sparse_ballot_t lane_mask = lane == 0 ? 0 : ((fattn_sparse_ballot_t(1) << lane) - 1);
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + fattn_sparse_popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & (fattn_sparse_ballot_t(1) << lane)) && dst < n_kv_max) {
                indices[dst] = i;
            }
            warp_item_offset += fattn_sparse_popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}

// Parallel form of the kernel above: one pass counts the selected cells of every chunk of a list,
// the next writes a chunk's indices behind the counts of the chunks in front of it. Both passes vote
// over the same cells in the same order as the serial kernel, so the lists come out identical.
__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_count_sparse_indices(
        const half * mask_ptr, int32_t * counts_tmp_ptr, const int ne30, const int n_queries,
        const int ncols1, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = fattn_sparse_values_per_lane;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int group    = blockIdx.x;
    const int chunk    = blockIdx.y;
    const int sequence = blockIdx.z;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const half * mask = mask_ptr + sequence*s33 + q0*s31;
    const int i0 = chunk*fattn_sparse_chunk;

    __shared__ int warp_counts[fattn_sparse_block_size/warp_size];

    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
        const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
        bool selected = false;
        for (int q = 0; q < q1 - q0 && !selected; ++q) {
            selected = i < ne30 && isfinite(__half2float(mask[q*s31 + i]));
        }
        warp_count += fattn_sparse_popc(fattn_sparse_ballot(selected));
    }

    if (lane == 0) {
        warp_counts[warp] = warp_count;
    }
    __syncthreads();

    if (tid == 0) {
        int count = 0;
#pragma unroll
        for (int iw = 0; iw < fattn_sparse_block_size/warp_size; ++iw) {
            count += warp_counts[iw];
        }
        counts_tmp_ptr[(int64_t(sequence)*gridDim.x + group)*gridDim.y + chunk] = count;
    }
}

__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_write_sparse_indices(
        const half * mask_ptr, const int32_t * counts_tmp_ptr, int32_t * indices_ptr, int32_t * counts_ptr,
        const int ne30, const int n_queries, const int ncols1, const int n_kv_max,
        const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int values_per_lane = fattn_sparse_values_per_lane;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int group    = blockIdx.x;
    const int chunk    = blockIdx.y;
    const int sequence = blockIdx.z;
    const int n_chunks = gridDim.y;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    const int64_t list = int64_t(sequence)*gridDim.x + group;
    const half    * mask       = mask_ptr + sequence*s33 + q0*s31;
    const int32_t * counts_tmp = counts_tmp_ptr + list*n_chunks;
    int32_t       * indices    = indices_ptr + list*n_kv_max;
    const int i0 = chunk*fattn_sparse_chunk;

    __shared__ int warp_offsets[fattn_sparse_block_size/warp_size];
    __shared__ int chunk_count;
    __shared__ int chunk_base;

    // the running count the serial kernel would have carried into this chunk. A list has a few
    // hundred chunk counts at most and they are in L2 after the first pass, so one thread adds them up.
    if (tid == 0) {
        int base = 0;
        for (int c = 0; c < chunk; ++c) {
            base += counts_tmp[c];
        }
        chunk_base = base;
    }

    fattn_sparse_ballot_t selected_warp[values_per_lane];
    int warp_count = 0;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
        const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
        bool selected = false;
        for (int q = 0; q < q1 - q0 && !selected; ++q) {
            selected = i < ne30 && isfinite(__half2float(mask[q*s31 + i]));
        }
        selected_warp[item] = fattn_sparse_ballot(selected);
        warp_count += fattn_sparse_popc(selected_warp[item]);
    }

    if (lane == 0) {
        warp_offsets[warp] = warp_count;
    }
    __syncthreads();

    if (tid == 0) {
        int offset = 0;
#pragma unroll
        for (int iw = 0; iw < fattn_sparse_block_size/warp_size; ++iw) {
            const int count = warp_offsets[iw];
            warp_offsets[iw] = offset;
            offset += count;
        }
        chunk_count = offset;
    }
    __syncthreads();

    const fattn_sparse_ballot_t lane_mask = lane == 0 ? 0 : ((fattn_sparse_ballot_t(1) << lane) - 1);
    int warp_item_offset = 0;
#pragma unroll
    for (int item = 0; item < values_per_lane; ++item) {
        const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
        const int dst = chunk_base + warp_offsets[warp] + warp_item_offset + fattn_sparse_popc(selected_warp[item] & lane_mask);
        if ((selected_warp[item] & (fattn_sparse_ballot_t(1) << lane)) && dst < n_kv_max) {
            indices[dst] = i;
        }
        warp_item_offset += fattn_sparse_popc(selected_warp[item]);
    }

    // the last chunk knows the count of the whole list, and the padding it writes starts behind
    // every index of the list
    if (chunk == n_chunks - 1) {
        const int count = min(chunk_base + chunk_count, n_kv_max);
        for (int i = count + tid; i < n_kv_max; i += fattn_sparse_block_size) {
            indices[i] = -1;
        }
        if (tid == 0) {
            counts_ptr[list] = count;
        }
    }
    __syncthreads();

    // the dependent grid reads indices; it starts once every block of this grid has signalled
    ggml_cuda_pdl_lc();
}
#endif // !defined(GGML_USE_MUSA)

void ggml_cuda_flash_attn_ext_compact_mask(
        ggml_backend_cuda_context & ctx, const ggml_tensor * mask, int32_t * indices, int32_t * counts,
        int32_t n_queries, int32_t ncols1, int32_t n_kv_max) {
#if defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(ctx, mask, indices, counts, n_queries, ncols1, n_kv_max);
    GGML_ABORT("sparse flash attention is not supported on MUSA");
#else
    cudaStream_t stream = ctx.stream();

    const int64_t s31      = mask->nb[1] / sizeof(half);
    const int64_t s33      = mask->nb[3] / sizeof(half);
    const int     ne30     = int(mask->ne[0]);
    const int     ne33     = int(mask->ne[3]);
    const int     n_groups = (n_queries + ncols1 - 1)/ncols1;
    const int     n_chunks = (ne30 + fattn_sparse_chunk - 1)/fattn_sparse_chunk;
    const size_t  n_lists  = size_t(n_groups)*ne33;

    const dim3 block_dim(fattn_sparse_block_size, 1, 1);

    // GGML_CUDA_FATTN_COMPACT_PARALLEL=0 forces the serial kernel, =1 or unset picks by list length.
    static const bool parallel_enabled = [] {
        const char * value = getenv("GGML_CUDA_FATTN_COMPACT_PARALLEL");
        return !value || atoi(value) != 0;
    }();
    // GGML_CUDA_FATTN_COMPACT_VERIFY=1 runs both kernels and compares the lists on the host.
    static const bool verify_enabled = [] {
        const char * value = getenv("GGML_CUDA_FATTN_COMPACT_VERIFY");
        return value && atoi(value) != 0;
    }();

    const auto launch_serial = [&](int32_t * dst_indices, int32_t * dst_counts) {
        const dim3 blocks_num(n_groups, ne33, 1);
        const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);
        ggml_cuda_kernel_launch(flash_attn_mask_to_sparse_indices, launch_params,
            (const half *) mask->data, dst_indices, dst_counts, ne30, n_queries, ncols1, n_kv_max, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    };

    // The chunk counts of the first pass live in the pool for the two launches only; like every pool
    // buffer of this file they are handed back right away and can only be reused by a later kernel of
    // the same stream.
    const auto launch_parallel = [&](int32_t * dst_indices, int32_t * dst_counts) {
        ggml_cuda_pool_alloc<int32_t> counts_tmp(ctx.pool(), n_lists*n_chunks);

        const dim3 blocks_num(n_groups, n_chunks, ne33);
        const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);

        ggml_cuda_kernel_launch(flash_attn_mask_count_sparse_indices, launch_params,
            (const half *) mask->data, counts_tmp.ptr, ne30, n_queries, ncols1, s31, s33);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_kernel_launch(flash_attn_mask_write_sparse_indices, launch_params,
            (const half *) mask->data, counts_tmp.ptr, dst_indices, dst_counts,
            ne30, n_queries, ncols1, n_kv_max, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    };

    if (verify_enabled) {
        // The comparison reads the result back, which a stream that is capturing a graph cannot do.
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));

        if (capture_status != cudaStreamCaptureStatusNone) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                GGML_LOG_WARN("compact verify: the stream is capturing a graph, nothing is compared; "
                              "set GGML_CUDA_DISABLE_GRAPHS=1\n");
            }
        } else {
            const size_t n_indices = n_lists*n_kv_max;

            // The parallel result goes to a buffer of its own, indices and counts laid out as the
            // caller lays them out, so the serial kernel keeps feeding the model and a wrong parallel
            // kernel cannot change the output of the run that is verifying it.
            ggml_cuda_pool_alloc<int32_t> parallel(ctx.pool(), n_indices + n_lists);

            launch_serial(indices, counts);
            launch_parallel(parallel.ptr, parallel.ptr + n_indices);

            std::vector<int32_t> host_serial(n_indices + n_lists);
            std::vector<int32_t> host_parallel(n_indices + n_lists);
            CUDA_CHECK(cudaMemcpyAsync(host_serial.data(), indices,
                n_indices*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_serial.data() + n_indices, counts,
                n_lists*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(host_parallel.data(), parallel.ptr,
                (n_indices + n_lists)*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            static int64_t n_calls        = 0;
            static int64_t n_lists_total  = 0;
            static int     ncols1_logged  = 0;
            static int     n_chunks_logged = 0;
            n_calls++;

            for (int64_t sequence = 0; sequence < ne33; ++sequence) {
                for (int64_t group = 0; group < n_groups; ++group) {
                    const int64_t list = sequence*n_groups + group;
                    n_lists_total++;
                    for (int32_t i = 0; i < n_kv_max; ++i) {
                        const size_t pos = size_t(list)*n_kv_max + i;
                        if (host_serial[pos] != host_parallel[pos]) {
                            GGML_ABORT("compact verify: sequence %lld list %lld position %d: serial %d, parallel %d\n",
                                (long long) sequence, (long long) group, i,
                                host_serial[pos], host_parallel[pos]);
                        }
                    }
                    if (host_serial[n_indices + list] != host_parallel[n_indices + list]) {
                        GGML_ABORT("compact verify: sequence %lld list %lld count: serial %d, parallel %d\n",
                            (long long) sequence, (long long) group,
                            host_serial[n_indices + list], host_parallel[n_indices + list]);
                    }
                }
            }

            // one line whenever ncols1 or the chunk count changes, and every 1000 calls, so the log
            // names the list shapes that have been through both kernels
            if (ncols1 != ncols1_logged || n_chunks != n_chunks_logged || n_calls % 1000 == 0) {
                ncols1_logged   = ncols1;
                n_chunks_logged = n_chunks;
                GGML_LOG_INFO("compact verify: ncols1 %d, %d chunks, %d slots, %lld calls, %lld lists, all equal\n",
                    ncols1, n_chunks, n_kv_max, (long long) n_calls, (long long) n_lists_total);
            }
            return;
        }
    }

    if (parallel_enabled && n_chunks >= fattn_sparse_parallel_min_chunks) {
        launch_parallel(indices, counts);
    } else {
        launch_serial(indices, counts);
    }
#endif // !defined(GGML_USE_MUSA)
}

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1);
    return false;
#else
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    const int64_t n_gather = (ncols1 == 1 ? Q->ne[1] : ncols1) * (int64_t) n_kv_max;

    return GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) &&
        mask != nullptr && n_kv_max > 0 && max_bias == 0.0f && logit_softcap == 0.0f &&
        mask->ne[0] == K->ne[1] && mask->ne[1] >= Q->ne[1] && mask->ne[2] == 1 &&
        K->ne[1] >= std::max<int64_t>(4096, 2*n_gather);
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 1, ncols2)) {
        if (ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1)) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 1, ncols2>(ctx, dst);
            return;
        }
    }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    // On RDNA it is preferable to minimize wasted compute vs. duplicate I/O for the mask.
    if (amd_wmma_available(cc)) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 2 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
            return;
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K_case, type_V_case)                                                                                \
    if constexpr (GGML_CUDA_FA_##type_K_case##_##type_V_case) {                                                                    \
        const bool type_K_okay = type_K == GGML_TYPE_##type_K_case || (type_K == GGML_TYPE_F32 && GGML_TYPE_##type_K_case == GGML_TYPE_F16); \
        const bool type_V_okay = type_V == GGML_TYPE_##type_V_case || (type_V == GGML_TYPE_F32 && GGML_TYPE_##type_V_case == GGML_TYPE_F16); \
        if (head_size == (D) && type_K_okay && type_V_okay) {                                                                      \
            return ggml_cuda_flash_attn_ext_vec_case<D, GGML_TYPE_##type_K_case, GGML_TYPE_##type_V_case>;                         \
        }                                                                                                                          \
    }                                                                                                                              \

#define FATTN_VEC_CASES_ALL_D(type_K_case, type_V_case) \
    FATTN_VEC_CASE( 64, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(128, type_K_case, type_V_case)       \
    FATTN_VEC_CASE(256, type_K_case, type_V_case)       \

typedef void (* fattn_vec_case_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Vector kernel for the given head size and K/V types, nullptr if its template instance was not compiled:
static fattn_vec_case_t ggml_cuda_get_fattn_vec_case(const int64_t head_size, const ggml_type type_K, const ggml_type type_V) {
    FATTN_VEC_CASES_ALL_D(F16,  F16)
    FATTN_VEC_CASES_ALL_D(Q4_0, F16)
    FATTN_VEC_CASES_ALL_D(Q4_1, F16)
    FATTN_VEC_CASES_ALL_D(Q5_0, F16)
    FATTN_VEC_CASES_ALL_D(Q5_1, F16)
    FATTN_VEC_CASES_ALL_D(Q8_0, F16)
    FATTN_VEC_CASES_ALL_D(BF16, F16)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q4_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q4_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q4_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_0)

    FATTN_VEC_CASES_ALL_D(F16,  Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q5_1)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q5_1)
    FATTN_VEC_CASES_ALL_D(BF16, Q5_1)

    FATTN_VEC_CASES_ALL_D(F16,  Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q4_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q5_1, Q8_0)
    FATTN_VEC_CASES_ALL_D(Q8_0, Q8_0)
    FATTN_VEC_CASES_ALL_D(BF16, Q8_0)

    FATTN_VEC_CASES_ALL_D(F16,  BF16)
    FATTN_VEC_CASES_ALL_D(Q4_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q4_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_0, BF16)
    FATTN_VEC_CASES_ALL_D(Q5_1, BF16)
    FATTN_VEC_CASES_ALL_D(Q8_0, BF16)
    FATTN_VEC_CASES_ALL_D(BF16, BF16)

    return nullptr;
}

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    fattn_vec_case_t vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type);
    if (vec_case == nullptr) {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_WARN("%s: no FlashAttention vector kernel compiled for K/V types %s-%s, converting K and V to f16 instead (slow). "
                "Add \"%s-%s\" to GGML_CUDA_FA_QUANTS to compile it.\n",
                __func__, ggml_type_name(K->type), ggml_type_name(V->type), ggml_type_name(K->type), ggml_type_name(V->type));
            warned = true;
        }
        vec_case = ggml_cuda_get_fattn_vec_case(Q->ne[0], GGML_TYPE_F16, GGML_TYPE_F16);
    }
    GGML_ASSERT(vec_case != nullptr);
    vec_case(ctx, dst);
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE    =   0,
    BEST_FATTN_KERNEL_TILE    = 200,
    BEST_FATTN_KERNEL_VEC     = 100,
    BEST_FATTN_KERNEL_MMA_F16 = 400,
};

// K/V types for which there is a vector kernel template instance, other kernels convert these to f16:
static bool ggml_cuda_fattn_kv_type_supported(const ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                // the sparse gather exists only in the MMA kernel: (DKQ, DV, 1, 8) with GQA > 4
                const bool sparse_decode = gqa_opt_applies && gqa_ratio > 4 &&
                    ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(K->ne[0], V->ne[0], 1, 8) &&
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1);
                if (!sparse_decode && cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 &&
                        !(gqa_ratio > 4 && (Q->ne[0] >= 256 || K->ne[1] >= 8192))) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

#ifdef GGML_USE_HIP
    static const bool prefill_wmma = [] {
        const char * value = getenv("GGML_HIP_PREFILL_WMMA");
        return !value || atoi(value) != 0;
    }();
    // The 16-row bound is a conservative policy choice, not a kernel requirement: 2..15 rows were not measured.
    if (prefill_wmma && GGML_CUDA_CC_IS_RDNA4(cc) && amd_wmma_available(cc) &&
        Q->ne[0] == 512 && V->ne[0] == 512 && Q->ne[1] >= 16 &&
        gqa_ratio == 8 && gqa_opt_applies && K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }
#endif
    // AMD WMMA is faster than the tile kernel if the wide tiles with high arithmetic intensity can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72 &&
            Q->ne[1] * gqa_ratio_eff > (Q->ne[0] <= 128 ? 8 : 16)) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC: {
            const bool f16_fallback = ggml_cuda_get_fattn_vec_case(Q->ne[0], K->type, V->type) == nullptr;
            need_f16_K = K->type == GGML_TYPE_F32 || f16_fallback;
            need_f16_V = V->type == GGML_TYPE_F32 || f16_fallback;
        } break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
            ggml_cuda_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_cuda_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
