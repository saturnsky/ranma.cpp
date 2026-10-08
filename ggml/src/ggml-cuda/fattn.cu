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
template <int ncols1, bool oob>
__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_to_sparse_indices(
        const half * mask_ptr, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
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
            if (i < ne30) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
                }
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
// over the same cells in the same order as the serial kernel, with the same selection expression,
// so the lists come out identical.
template <int ncols1, bool oob>
__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_count_sparse_indices(
        const half * mask_ptr, int32_t * counts_tmp_ptr, const int ne30, const int n_queries,
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
        if (i < ne30) {
#pragma unroll
            for (int q = 0; q < ncols1; ++q) {
                selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
            }
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

template <int ncols1, bool oob>
__launch_bounds__(fattn_sparse_block_size, 1)
static __global__ void flash_attn_mask_write_sparse_indices(
        const half * mask_ptr, const int32_t * counts_tmp_ptr, int32_t * indices_ptr, int32_t * counts_ptr,
        const int ne30, const int n_queries, const int n_kv_max,
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
        if (i < ne30) {
#pragma unroll
            for (int q = 0; q < ncols1; ++q) {
                selected |= (!oob || q < q1 - q0) && isfinite(__half2float(mask[q*s31 + i]));
            }
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

typedef decltype(&flash_attn_mask_to_sparse_indices<1, false>)    fattn_sparse_serial_t;
typedef decltype(&flash_attn_mask_count_sparse_indices<1, false>) fattn_sparse_count_t;
typedef decltype(&flash_attn_mask_write_sparse_indices<1, false>) fattn_sparse_write_t;

// the serial kernel and the parallel pair of one tile width, so that both forms vote alike
template <int ncols1, bool oob>
static void fattn_sparse_pick_kernels(fattn_sparse_serial_t & serial, fattn_sparse_count_t & count, fattn_sparse_write_t & write) {
    serial = flash_attn_mask_to_sparse_indices<ncols1, oob>;
    count  = flash_attn_mask_count_sparse_indices<ncols1, oob>;
    write  = flash_attn_mask_write_sparse_indices<ncols1, oob>;
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

    // the last group of queries is partial only if ncols1 does not divide n_queries
    // ncols1: 1 and 8 for the CUDA MMA gather, 1, 2 and 4 for the HIP tile gather
    const bool oob = n_queries % ncols1 != 0;
    fattn_sparse_serial_t kernel_serial = nullptr;
    fattn_sparse_count_t  kernel_count  = nullptr;
    fattn_sparse_write_t  kernel_write  = nullptr;
    switch (ncols1) {
        case 1: fattn_sparse_pick_kernels<1, false>(kernel_serial, kernel_count, kernel_write); break;
        case 2: oob ? fattn_sparse_pick_kernels<2, true>(kernel_serial, kernel_count, kernel_write) :
                      fattn_sparse_pick_kernels<2, false>(kernel_serial, kernel_count, kernel_write); break;
        case 4: oob ? fattn_sparse_pick_kernels<4, true>(kernel_serial, kernel_count, kernel_write) :
                      fattn_sparse_pick_kernels<4, false>(kernel_serial, kernel_count, kernel_write); break;
        case 8: oob ? fattn_sparse_pick_kernels<8, true>(kernel_serial, kernel_count, kernel_write) :
                      fattn_sparse_pick_kernels<8, false>(kernel_serial, kernel_count, kernel_write); break;
        default: GGML_ABORT("sparse mask compaction not compiled for ncols1 = %d", ncols1);
    }

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
        ggml_cuda_kernel_launch(kernel_serial, launch_params,
            (const half *) mask->data, dst_indices, dst_counts, ne30, n_queries, n_kv_max, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    };

    // The chunk counts of the first pass live in the pool for the two launches only; like every pool
    // buffer of this file they are handed back right away and can only be reused by a later kernel of
    // the same stream.
    const auto launch_parallel = [&](int32_t * dst_indices, int32_t * dst_counts) {
        ggml_cuda_pool_alloc<int32_t> counts_tmp(ctx.pool(), n_lists*n_chunks);

        const dim3 blocks_num(n_groups, n_chunks, ne33);
        const ggml_cuda_kernel_launch_params launch_params(blocks_num, block_dim, 0, stream);

        ggml_cuda_kernel_launch(kernel_count, launch_params,
            (const half *) mask->data, counts_tmp.ptr, ne30, n_queries, s31, s33);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_kernel_launch(kernel_write, launch_params,
            (const half *) mask->data, counts_tmp.ptr, dst_indices, dst_counts,
            ne30, n_queries, n_kv_max, s31, s33);
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

bool ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(const int cc, const ggml_tensor * dst, const int ncols1, const int ncols2) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(cc, dst, ncols1, ncols2);
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

    // the dense kernel handles up to 64/ncols2 queries per K/V pass, the single-query gather has to beat that
    const int64_t n_gather = (ncols1 == 1 ? std::min<int64_t>(Q->ne[1], 64/ncols2) : ncols1) * (int64_t) n_kv_max;

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
        // a sparse variant at the full tile width gathers the union of its queries once, prefer it for large batches
        constexpr bool has_wide_sparse = ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse(DKQ, DV, 64/ncols2, ncols2);
        if (!(has_wide_sparse && Q->ne[1] > 32/ncols2) && ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, ncols2)) {
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
            if (gqa_ratio == 20 && GGML_CUDA_CC_IS_NVIDIA(cc)) { // GLM 4.7 Flash
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
                    ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse(cc, dst, 1, 8);
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
    while (max_bias == 0.0f && gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
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
    if (amd_mfma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if (Q->ne[0] > 256 && gqa_opt_applies && Q->ne[1] * gqa_ratio_eff > 128) {
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

// ---- MLA decode attention (RDNA4, WMMA) ----------------------------------------------------------------------------
// One query row over a latent cache with K == V (DeepSeek/GLM style MLA after absorbing the key projection): 64 heads of 512
// values share one K/V head. The tile kernel serializes 16 gather/compute stages per KV tile and needs about 100-250 us per
// layer for the 2048 selected cells of a sparse decode; here a block stages 16 cells at a time in LDS, runs the scores
// S^T[cell][head] = K Q^T and the update O[head][dv] += P V on the WMMA units and keeps the next chunk's loads in flight.
// A block owns 32 heads and a run of NCH chunks of 16 list entries, the partial results are merged by a second kernel.
#if defined(GGML_USE_HIP)

static constexpr int fattn_mla_dim      = 512;
static constexpr int fattn_mla_heads    = 64;
static constexpr int fattn_mla_chunk    = 16;
static constexpr int fattn_mla_pitch    = 520; // halves per LDS row (16 B of padding against bank conflicts)
static constexpr int fattn_mla_max_part = 128; // upper bound of partial results per head (scratch size)
static constexpr int fattn_mla_max_rows = 8;   // query rows (MTP verification), each one a block row of its own

template <int NCH>
static __global__ void __launch_bounds__(256) fattn_mla_decode_kernel(
        const float * __restrict__ Q, const int64_t q_head_stride, const int64_t q_row_stride, const half * __restrict__ K, const int64_t k_stride,
        const int32_t * __restrict__ indices_all, const int32_t * __restrict__ pcount_all, const int n_cells, const int64_t list_stride,
        const half * __restrict__ mask_all, const int64_t mask_row_stride, float * __restrict__ Opart_all, float2 * __restrict__ ML_all,
        const int n_part, const float scale) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    using halfx8_t  = __attribute__((ext_vector_type(8))) _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8))) float;

    __shared__ _Float16 Qs[32*fattn_mla_pitch];
    __shared__ _Float16 Ks[fattn_mla_chunk*fattn_mla_pitch];
    __shared__ float biasS[fattn_mla_chunk];

    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, ht = w & 1, dqq = w >> 1;
    const int g = blockIdx.x, hh = blockIdx.y, row = blockIdx.z;
    const int32_t * indices = indices_all ? indices_all + row*list_stride : nullptr;
    const half * mask = mask_all + row*mask_row_stride;
    float * Opart = Opart_all + (int64_t) row*n_part*fattn_mla_heads*fattn_mla_dim;
    float2 * ML = ML_all + (int64_t) row*n_part*fattn_mla_heads;
    const int cnt = indices ? min(pcount_all[row], n_cells) : n_cells;

    for (int e = tid; e < 32*fattn_mla_dim/4; e += 256) {
        const int h = e/(fattn_mla_dim/4), c4 = e % (fattn_mla_dim/4);
        const float4 f = *(const float4 *) (Q + row*q_row_stride + (int64_t) (hh*32 + h)*q_head_stride + c4*4);
        _Float16 * d = Qs + h*fattn_mla_pitch + c4*4;
        d[0] = (_Float16) (f.x*scale); d[1] = (_Float16) (f.y*scale); d[2] = (_Float16) (f.z*scale); d[3] = (_Float16) (f.w*scale);
    }

    floatx8_t O[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        O[i] = floatx8_t{0, 0, 0, 0, 0, 0, 0, 0};
    }
    float m_run = -INFINITY, l_run = 0.0f;

    const int r_ld = tid/16, s_ld = tid % 16; // row of the chunk and first 16 B segment of this thread (segments s_ld + 16 j)
    uint4 pre[4];
    float pre_bias = 0.0f;
    const auto issue = [&](const int c) {
        const int pos = (g*NCH + c)*fattn_mla_chunk + r_ld;
        int cid = -1;
        if (pos < cnt) {
            cid = indices ? indices[pos] : pos;
        }
        float bias = 0.0f;
        if (cid >= 0) {
            if (mask) {
                bias = __half2float(mask[cid]);
            }
            const uint4 * src = (const uint4 *) (K + (int64_t) cid*k_stride);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pre[j] = src[s_ld + 16*j];
            }
        }
        const bool valid = cid >= 0 && bias > -INFINITY;
        pre_bias = valid ? bias : -INFINITY;
        if (!valid) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pre[j] = make_uint4(0, 0, 0, 0);
            }
        }
    };
    issue(0);

    for (int c = 0; c < NCH; ++c) {
        {
            uint4 * dst = (uint4 *) (Ks + r_ld*fattn_mla_pitch);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[s_ld + 16*j] = pre[j];
            }
            if (s_ld == 0) {
                biasS[r_ld] = pre_bias;
            }
        }
        __syncthreads();
        if (c + 1 < NCH) {
            issue(c + 1);
        }

        floatx8_t s0 = floatx8_t{0, 0, 0, 0, 0, 0, 0, 0}, s1 = s0;
        const _Float16 * Ka = Ks + (lane & 15)*fattn_mla_pitch + 8*(lane >> 4);
        const _Float16 * Qb = Qs + (ht*16 + (lane & 15))*fattn_mla_pitch + 8*(lane >> 4);
#pragma unroll
        for (int ks = 0; ks < 32; ks += 2) {
            const halfx8_t a0 = *(const halfx8_t *) (Ka + ks*16), b0 = *(const halfx8_t *) (Qb + ks*16);
            const halfx8_t a1 = *(const halfx8_t *) (Ka + (ks + 1)*16), b1 = *(const halfx8_t *) (Qb + (ks + 1)*16);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, s0);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, s1);
        }
        float s[8];
        float mc = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s[i] = s0[i] + s1[i] + biasS[8*(lane >> 4) + i];
            mc = fmaxf(mc, s[i]);
        }
        mc = fmaxf(mc, __shfl_xor(mc, 16, 32));
        const float m_new = fmaxf(m_run, mc);
        float corr, ls = 0.0f;
        halfx8_t P;
        if (m_new == -INFINITY) { // nothing visible yet
            corr = 1.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                P[i] = (_Float16) 0.0f;
            }
        } else {
            corr = m_run == -INFINITY ? 0.0f : expf(m_run - m_new);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = expf(s[i] - m_new);
                ls += p;
                P[i] = (_Float16) p;
            }
            ls += __shfl_xor(ls, 16, 32);
        }
        l_run = l_run*corr + ls;
        m_run = m_new;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float f = __shfl(corr, 8*(lane >> 4) + i, 32);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                O[nt][i] *= f;
            }
        }
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            halfx8_t bv;
            const _Float16 * vp = Ks + (8*(lane >> 4))*fattn_mla_pitch + dqq*128 + nt*16 + (lane & 15);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                bv[i] = vp[i*fattn_mla_pitch];
            }
            O[nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(P, bv, O[nt]);
        }
        __syncthreads();
    }

    const int head0 = hh*32 + ht*16;
    float * op = Opart + ((int64_t) g*fattn_mla_heads + head0)*fattn_mla_dim;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            op[(int64_t) (8*(lane >> 4) + i)*fattn_mla_dim + dqq*128 + nt*16 + (lane & 15)] = O[nt][i];
        }
    }
    if (dqq == 0 && lane < 16) {
        ML[(int64_t) g*fattn_mla_heads + head0 + lane] = make_float2(m_run, l_run);
    }
#else
    GGML_UNUSED_VARS(Q, q_head_stride, q_row_stride, K, k_stride, indices_all, pcount_all, n_cells, list_stride, mask_all, mask_row_stride, Opart_all, ML_all, n_part, scale);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
}

static __global__ void __launch_bounds__(512) fattn_mla_combine_kernel(
        const float * __restrict__ Opart_all, const float2 * __restrict__ ML_all, const int n_part, float * __restrict__ dst_all) {
    const int h = blockIdx.x, t = threadIdx.x, row = blockIdx.y;
    const float * Opart = Opart_all + (int64_t) row*n_part*fattn_mla_heads*fattn_mla_dim;
    const float2 * ML = ML_all + (int64_t) row*n_part*fattn_mla_heads;
    float * dst = dst_all + (int64_t) row*fattn_mla_heads*fattn_mla_dim;
    float M = -INFINITY;
    for (int g = 0; g < n_part; ++g) {
        M = fmaxf(M, ML[(int64_t) g*fattn_mla_heads + h].x);
    }
    float L = 0.0f, acc = 0.0f;
    for (int g = 0; g < n_part; ++g) {
        const float2 ml = ML[(int64_t) g*fattn_mla_heads + h];
        const float f = ml.x == -INFINITY ? 0.0f : expf(ml.x - M);
        L += f*ml.y;
        acc += f*Opart[((int64_t) g*fattn_mla_heads + h)*fattn_mla_dim + t];
    }
    dst[h*fattn_mla_dim + t] = L > 0.0f ? acc/L : 0.0f;
}

// true when the MLA decode kernel computed `dst`; false leaves the op to the regular kernels
static bool ggml_cuda_flash_attn_ext_mla_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_DISABLE_FATTN_MLA_DECODE");
        return value == nullptr || atoi(value) == 0;
    }();
    if (!enabled) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA4(cc) || !amd_wmma_available(cc) || ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false;
    }
    if (Q->ne[0] != fattn_mla_dim || K->ne[0] != fattn_mla_dim || V->ne[0] != fattn_mla_dim || Q->ne[1] < 1 || Q->ne[1] > fattn_mla_max_rows || Q->ne[2] != fattn_mla_heads ||
            Q->ne[3] != 1 || K->ne[2] != 1 || K->ne[3] != 1 || V->ne[2] != 1 || V->ne[3] != 1 || mask == nullptr || sinks != nullptr) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // V is the K cache itself (the latent is both key and value)
    if (V->data != K->data || V->nb[1] != K->nb[1] || V->ne[1] != K->ne[1] || K->nb[0] != sizeof(half)) {
        return false;
    }
    if (mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1 || mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half) != 0) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f, scale = 1.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || Q->nb[2] % 16 != 0 || Q->nb[1] % 16 != 0 || ((uintptr_t) Q->data) % 16 != 0 || K->nb[1] % 16 != 0 || ((uintptr_t) K->data) % 16 != 0 ||
            !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_kv     = K->ne[1];
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    const bool    use_list = n_kv_max > 0 && n_kv >= 2*(int64_t) n_kv_max;
    const int64_t n_cells  = use_list ? n_kv_max : n_kv;
    // the dense form visits every cache row; far beyond a few thousand rows only the list form is worth it
    if (n_cells <= 0 || (!use_list && n_kv > 8192)) {
        return false;
    }

    const int nch    = n_cells <= 1024 ? 4 : 8;
    const int n_part = (int) ((n_cells + fattn_mla_chunk*nch - 1)/(fattn_mla_chunk*nch));
    if (n_part > fattn_mla_max_part) {
        return false;
    }

    const int n_rows = (int) Q->ne[1];
    cudaStream_t stream = ctx.stream();
    // one index list per query row, the live counts of all rows behind the lists
    ggml_cuda_pool_alloc<int32_t> list(ctx.pool(), use_list ? (size_t) n_rows*((size_t) n_kv_max + 1) : 0);
    ggml_cuda_pool_alloc<float>   part(ctx.pool(), (size_t) n_rows*n_part*fattn_mla_heads*fattn_mla_dim);
    ggml_cuda_pool_alloc<float2>  ml(ctx.pool(), (size_t) n_rows*n_part*fattn_mla_heads);
    if (use_list) {
        ggml_cuda_flash_attn_ext_compact_mask(ctx, mask, list.ptr, list.ptr + (size_t) n_rows*n_kv_max, n_rows, 1, n_kv_max);
    }

    const dim3 grid(n_part, 2, n_rows);
    const float * q = (const float *) Q->data;
    const int64_t q_head_stride = Q->nb[2]/sizeof(float);
    const int64_t q_row_stride  = Q->nb[1]/sizeof(float);
    const int64_t k_stride      = K->nb[1]/sizeof(half);
    const int32_t * indices = use_list ? list.ptr : nullptr;
    const int32_t * pcount  = use_list ? list.ptr + (size_t) n_rows*n_kv_max : nullptr;
    const int64_t mask_row_stride = mask->nb[1]/sizeof(half);
    if (nch == 4) {
        fattn_mla_decode_kernel<4><<<grid, 256, 0, stream>>>(q, q_head_stride, q_row_stride, (const half *) K->data, k_stride, indices, pcount,
            (int) n_cells, (int64_t) n_kv_max, (const half *) mask->data, mask_row_stride, part.ptr, ml.ptr, n_part, scale);
    } else {
        fattn_mla_decode_kernel<8><<<grid, 256, 0, stream>>>(q, q_head_stride, q_row_stride, (const half *) K->data, k_stride, indices, pcount,
            (int) n_cells, (int64_t) n_kv_max, (const half *) mask->data, mask_row_stride, part.ptr, ml.ptr, n_part, scale);
    }
    CUDA_CHECK(cudaGetLastError());
    fattn_mla_combine_kernel<<<dim3(fattn_mla_heads, n_rows), fattn_mla_dim, 0, stream>>>(part.ptr, ml.ptr, n_part, (float *) dst->data);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif // defined(GGML_USE_HIP)

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
#if defined(GGML_USE_HIP)
    if (ggml_cuda_flash_attn_ext_mla_decode(ctx, dst)) {
        return;
    }
#endif // defined(GGML_USE_HIP)
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
