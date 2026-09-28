#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;

    // the largest column a value that ties with prefix may have; every tied
    // column at or below it is kept, which makes the result reproducible
    uint32_t col_prefix;
    int col_rank;
};

// col_prefix is the largest column a value that ties with the threshold may have: UINT32_MAX
// accepts all of them, 0 is the start of the walk that finds the bound of the stable selection
static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k, uint32_t col_prefix) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {};
        states[row].rank       = k;
        states[row].col_prefix = col_prefix;
    }
}

// GGML_CUDA_TOP_K_STABLE_TIES=1: among the values that tie with the threshold the smallest
// columns are selected, so the selected set is the same in every run. It is meant for tests
// that compare selections or outputs between runs. Off by default: which of the tied values
// are kept is then left to the order the gather happens to run in.
static bool top_k_stable_ties() {
    static const bool stable = [] {
        const char * value = getenv("GGML_CUDA_TOP_K_STABLE_TIES");
        return value != nullptr && atoi(value) != 0;
    }();
    return stable;
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// The value selection above leaves `rank` values that tie exactly with `prefix`, and the
// budget cuts through them. Which of those a parallel gather keeps is otherwise decided by
// the order the atomics happen to run in, so the selection is not reproducible - and ties
// are common here, because rectified scores collapse to exactly zero. Select the `rank`
// smallest columns among them, by the same radix walk, taken from the low bin upwards.
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_col_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];

    // bits above this pass are already decided, bits below it are not looked at yet
    const uint32_t col_mask = shift + RADIX_BITS >= 32 ? 0u : ~((1u << (shift + RADIX_BITS)) - 1);

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key == state.prefix && (((uint32_t) col) & col_mask) == state.col_prefix) {
            atomicAdd(&histogram[(((uint32_t) col) >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_col_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift,
        int first) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int rank = first ? state.rank : state.col_rank;
        int bin = 0;
        while (bin < NBINS - 1 && histogram[bin] < rank) {
            rank -= histogram[bin++];
        }
        state.col_prefix |= (uint32_t) bin << shift;
        state.col_rank = rank;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix && (uint32_t) col <= state->col_prefix) {
            // exactly `rank` columns pass this test, so the set written here is fixed;
            // only the order within the reserved slots depends on the atomics
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    const bool stable_ties = top_k_stable_ties();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        states, nrows, k, stable_ties ? 0u : UINT32_MAX);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    // only the bits a column index can actually use
    int col_bits = 1;
    while (col_bits < 32 && ((uint32_t) 1 << col_bits) < (uint32_t) ncols) {
        ++col_bits;
    }
    const int col_passes = stable_ties ? (col_bits + RADIX_BITS - 1) / RADIX_BITS : 0;

    for (int pass = col_passes - 1; pass >= 0; --pass) {
        const int shift = pass * RADIX_BITS;
        top_k_radix_col_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_col_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(
                histograms, states, blocks_per_row, shift, pass == col_passes - 1);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

template<typename M>
static __device__ float top_k_block_mask(const M * mask, size_t i) {
    return (float) mask[i];
}

template<typename M>
static __global__ void top_k_block_count(const int * map, const M * mask, int * weights,
        int nb, int nc, int nq) {
    const int row = blockIdx.x;
    const int * cell_blocks = map + (size_t) (row/nq)*nc;
    int * count = weights + (size_t) row*(nb + 1);
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        const float m = top_k_block_mask(mask, (size_t) row*nc + c);
        const int b = cell_blocks[c];
        assert(b >= 0 && b < nb && (m == 0.0f || m == -INFINITY));
        atomicAdd(count + (m == 0.0f ? b : nb), 1);
    }
}

static __global__ void top_k_block_histogram(const float * scores, const int * weights,
        const top_k_radix_state * states, int * histograms, int nb, int blocks_per_row, int shift) {
    const int row  = blockIdx.x/blocks_per_row;
    const int part = blockIdx.x%blocks_per_row;
    const int tid  = threadIdx.x;
    __shared__ int hist[256];
    hist[tid] = 0;
    __syncthreads();
    const auto state = states[row];
    for (int b = part*256 + tid; b <= nb; b += blocks_per_row*256) {
        const int weight = weights[(size_t) row*(nb + 1) + b];
        if (weight == 0) {
            continue;
        }
        const float value = b == nb ? -INFINITY : scores[(size_t) row*nb + b] + 0.0f;
        assert(isfinite(value) || value == -INFINITY);
        const uint32_t key = top_k_float_to_ordered(value);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(hist + ((key >> shift) & 255), weight);
        }
    }
    __syncthreads();
    histograms[((size_t) row*blocks_per_row + part)*256 + tid] = hist[tid];
}

template<typename M>
static __device__ uint32_t top_k_block_cell_key(const float * scores, const int * map, const M * mask,
        int row, int c, int nb, int nc, int nq) {
    const int b = map[(size_t) (row/nq)*nc + c];
    return top_k_float_to_ordered(scores[(size_t) row*nb + b] + top_k_block_mask(mask, (size_t) row*nc + c));
}

// Stable selection: histogram of the original columns of the cells at the chosen score boundary.
template<typename M>
static __global__ void top_k_block_tie_histogram(const float * scores, const int * map, const M * mask,
        const top_k_radix_state * states, int * histograms, int nb, int nc, int nq, int parts, int shift) {
    const int row  = blockIdx.x/parts;
    const int part = blockIdx.x%parts;
    const int tid  = threadIdx.x;
    __shared__ int hist[256];
    hist[tid] = 0;
    __syncthreads();
    const auto state = states[row];
    const uint32_t col_mask = shift + 8 >= 32 ? 0u : ~((1u << (shift + 8)) - 1);
    for (int c = part*256 + tid; c < nc; c += parts*256) {
        if ((((uint32_t) c) & col_mask) == state.col_prefix &&
                top_k_block_cell_key(scores, map, mask, row, c, nb, nc, nq) == state.prefix) {
            atomicAdd(hist + (((uint32_t) c >> shift) & 255), 1);
        }
    }
    __syncthreads();
    histograms[((size_t) row*parts + part)*256 + tid] = hist[tid];
}

template<typename M>
static __global__ void top_k_block_gather(const float * scores, const int * map, const M * mask,
        top_k_radix_state * states, int * out, int nb, int nc, int nq, int k, int parts, bool preserve_ties) {
    const int row  = blockIdx.x/parts;
    const int part = blockIdx.x%parts;
    auto * state = states + row;
    for (int c = part*256 + threadIdx.x; c < nc; c += parts*256) {
        const uint32_t key = top_k_block_cell_key(scores, map, mask, row, c, nb, nc, nq);
        if (key > state->prefix) {
            // exactly k - rank cells pass this test unless the weights and the scores
            // disagree, and then the row is wrong anyway: do not write past the row
            const int pos = atomicAdd(&state->greater_count, 1);
            assert(pos < k - state->rank);
            out[(size_t) row*k + pos] = c;
        } else if (key == state->prefix && (!preserve_ties || (uint32_t) c <= state->col_prefix)) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                out[(size_t) row*k + k - state->rank + pos] = c;
            }
        }
    }
}

// Single-query decode path: one workgroup per row runs the whole selection, so the device
// work follows the block count and the selected cells instead of the cell count. A block's
// cells are read through the inverse map, and the per-block weights come from the host meta
// instead of a pass over every cell. A row whose meta drops the promise (see ggml.h) is
// selected by the general cell walk inside the same kernel, so the launch list never moves.
#define TOP_K_BLOCK_ROW_NT   1024
#define TOP_K_BLOCK_ROW_NREP 16
#define TOP_K_BLOCK_ROW_PAD  257

// per-bin totals, then the bin the budget falls into, by the rule the multi-block kernels
// use: descending over values, ascending over column indices
static __device__ int top_k_block_row_pick(const int * __restrict__ tot, int * __restrict__ scan,
        int * __restrict__ pick, int rank, bool descending) {
    const int tid = threadIdx.x;
    if (tid < 256) {
        scan[tid] = tot[tid];
    }
    if (tid == 0) {
        *pick = descending ? 0 : 255;
    }
    __syncthreads();

    // inclusive scan from the end the walk starts at
    for (int off = 1; off < 256; off *= 2) {
        int add = 0;
        if (tid < 256) {
            const int src = descending ? tid + off : tid - off;
            if (src >= 0 && src < 256) {
                add = scan[src];
            }
        }
        __syncthreads();
        if (tid < 256) {
            scan[tid] += add;
        }
        __syncthreads();
    }

    // the walk stops at the first bin whose inclusive sum reaches the budget
    if (tid < 256 && scan[tid] >= rank) {
        if (descending) {
            atomicMax(pick, tid);
        } else {
            atomicMin(pick, tid);
        }
    }
    __syncthreads();

    const int bin = *pick;
    const int res = rank - (scan[bin] - tot[bin]);
    __syncthreads();
    return res;
}

template<typename M>
static __global__ void __launch_bounds__(TOP_K_BLOCK_ROW_NT)
top_k_block_row(const float * __restrict__ scores, const int * __restrict__ map,
        const M * __restrict__ mask, const int * __restrict__ cells, const int * __restrict__ meta,
        int * __restrict__ weights, int * __restrict__ out, int nb, int nc, int nq, int r, int k, bool stable_ties) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int rep = tid/(TOP_K_BLOCK_ROW_NT/TOP_K_BLOCK_ROW_NREP);

    __shared__ int s_hist[TOP_K_BLOCK_ROW_NREP][TOP_K_BLOCK_ROW_PAD];
    __shared__ int s_tot[256];
    __shared__ int s_scan[256];
    __shared__ int s_pick;
    __shared__ int s_greater;
    __shared__ int s_equal;

    const float * sc  = scores  + (size_t) row*nb;
    const int *   cb  = map     + (size_t) (row/nq)*nc;
    const M *     mk  = mask    + (size_t) row*nc;
    const int *   bc  = cells == nullptr ? nullptr : cells + (size_t) (row/nq)*r*nb;
    int *         wr  = weights + (size_t) row*(nb + 1);
    int *         dst = out     + (size_t) row*k;

    const int * mrow   = meta == nullptr ? nullptr : meta + (size_t) row*GGML_TOP_K_BLOCK_META_N;
    const bool  fast   = mrow != nullptr && mrow[0] != 0;
    const int   n_bid  = fast ? mrow[1] : 0;
    const int   dead   = fast ? mrow[2] : -1;
    const int   n_tail = fast ? mrow[3] : 0;
    const int   n_msk  = fast ? mrow[4] : 0;
    const int * tail   = mrow == nullptr ? nullptr : mrow + GGML_TOP_K_BLOCK_META_HEAD;

    const uint32_t key_ninf = top_k_float_to_ordered(-INFINITY);

    // the cell path adds the mask's 0.0f to the block score, so add it here too
    auto key_of = [&](int b) -> uint32_t {
        return b == nb ? key_ninf : top_k_float_to_ordered(sc[b] + 0.0f);
    };
    auto cell_key_of = [&](int c) -> uint32_t {
        return top_k_float_to_ordered(sc[cb[c]] + top_k_block_mask(mk, (size_t) c));
    };
    // visible cells of a block: the r members of a full block, plus the cells the host listed
    // for the spare block, which are the ones no full block covers
    auto n_full = [&](int b, uint32_t key) -> int {
        return b < n_bid && key != key_ninf ? r : 0;
    };
    auto cell_of = [&](int b, int i, int nf) -> int {
        return i < nf ? bc[(size_t) b*r + i] : tail[i - nf];
    };
    auto weight_of = [&](int b, uint32_t key) -> int {
        if (!fast) {
            return wr[b];
        }
        return b == nb ? n_msk : n_full(b, key) + (b == dead ? n_tail : 0);
    };

    if (!fast) {
        // general fallback: count the visible cells of every block, as the multi-block path does
        for (int b = tid; b <= nb; b += TOP_K_BLOCK_ROW_NT) {
            wr[b] = 0;
        }
        __syncthreads();
        for (int c = tid; c < nc; c += TOP_K_BLOCK_ROW_NT) {
            const float m = top_k_block_mask(mk, (size_t) c);
            const int   b = cb[c];
            assert(b >= 0 && b < nb && (m == 0.0f || m == -INFINITY));
            atomicAdd(wr + (m == 0.0f ? b : nb), 1);
        }
        __syncthreads();
    }

    // value pass: weighted radix over the blocks, four bytes from the top
    uint32_t prefix = 0;
    uint32_t prefix_mask = 0;
    int rank = k;

    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < TOP_K_BLOCK_ROW_NREP*256; i += TOP_K_BLOCK_ROW_NT) {
            s_hist[i/256][i%256] = 0;
        }
        __syncthreads();

        for (int b = tid; b <= nb; b += TOP_K_BLOCK_ROW_NT) {
            const uint32_t key = key_of(b);
            const int      w   = weight_of(b, key);
            if (w != 0 && (key & prefix_mask) == prefix) {
                atomicAdd(&s_hist[rep][(key >> shift) & 255], w);
            }
        }
        __syncthreads();

        if (tid < 256) {
            int sum = 0;
            for (int i = 0; i < TOP_K_BLOCK_ROW_NREP; ++i) {
                sum += s_hist[i][tid];
            }
            s_tot[tid] = sum;
        }
        __syncthreads();

        rank = top_k_block_row_pick(s_tot, s_scan, &s_pick, rank, true);
        prefix      |= (uint32_t) s_pick << shift;
        prefix_mask |= (uint32_t) 255    << shift;
        __syncthreads();
    }

    // tie pass of the stable selection: the smallest cell indices among the cells that tie with
    // the threshold. Without it every tied cell is a candidate and the gather keeps the first
    // `rank` of them it happens to reach.
    uint32_t col_prefix = stable_ties ? 0u : UINT32_MAX;
    int col_rank = rank;

    int col_bits = 1;
    while (col_bits < 32 && ((uint32_t) 1 << col_bits) < (uint32_t) nc) {
        ++col_bits;
    }

    for (int pass = stable_ties ? (col_bits + 7)/8 - 1 : -1; pass >= 0; --pass) {
        const int      shift    = pass*8;
        const uint32_t col_mask = shift + 8 >= 32 ? 0u : ~((1u << (shift + 8)) - 1);

        for (int i = tid; i < TOP_K_BLOCK_ROW_NREP*256; i += TOP_K_BLOCK_ROW_NT) {
            s_hist[i/256][i%256] = 0;
        }
        __syncthreads();

        if (fast) {
            for (int b = tid; b < nb; b += TOP_K_BLOCK_ROW_NT) {
                const uint32_t key = key_of(b);
                if (key != prefix) {
                    continue;
                }
                const int nf = n_full(b, key);
                const int n  = nf + (b == dead ? n_tail : 0);
                for (int i = 0; i < n; ++i) {
                    const uint32_t c = (uint32_t) cell_of(b, i, nf);
                    if ((c & col_mask) == col_prefix) {
                        atomicAdd(&s_hist[rep][(c >> shift) & 255], 1);
                    }
                }
            }
        } else {
            for (int c = tid; c < nc; c += TOP_K_BLOCK_ROW_NT) {
                if ((((uint32_t) c) & col_mask) == col_prefix && cell_key_of(c) == prefix) {
                    atomicAdd(&s_hist[rep][(((uint32_t) c) >> shift) & 255], 1);
                }
            }
        }
        __syncthreads();

        if (tid < 256) {
            int sum = 0;
            for (int i = 0; i < TOP_K_BLOCK_ROW_NREP; ++i) {
                sum += s_hist[i][tid];
            }
            s_tot[tid] = sum;
        }
        __syncthreads();

        col_rank = top_k_block_row_pick(s_tot, s_scan, &s_pick, col_rank, false);
        col_prefix |= (uint32_t) s_pick << shift;
        __syncthreads();
    }

    // gather: every cell above the threshold, then the reserved tie slots
    if (tid == 0) {
        s_greater = 0;
        s_equal   = 0;
    }
    __syncthreads();

    if (fast) {
        for (int b = tid; b < nb; b += TOP_K_BLOCK_ROW_NT) {
            const uint32_t key = key_of(b);
            if (key < prefix) {
                continue;
            }
            const int nf = n_full(b, key);
            const int n  = nf + (b == dead ? n_tail : 0);
            for (int i = 0; i < n; ++i) {
                const uint32_t c = (uint32_t) cell_of(b, i, nf);
                if (key > prefix) {
                    // exactly k - rank cells pass this test unless the meta lied about a
                    // weight, and then the row is wrong anyway: do not write past the row
                    const int pos = atomicAdd(&s_greater, 1);
                    assert(pos < k - rank);
                    if (pos < k - rank) {
                        dst[pos] = c;
                    }
                } else if (c <= col_prefix) {
                    const int pos = atomicAdd(&s_equal, 1);
                    if (pos < rank) {
                        dst[k - rank + pos] = c;
                    }
                }
            }
        }
    } else {
        for (int c = tid; c < nc; c += TOP_K_BLOCK_ROW_NT) {
            const uint32_t key = cell_key_of(c);
            if (key > prefix) {
                const int pos = atomicAdd(&s_greater, 1);
                assert(pos < k - rank);
                dst[pos] = c;
            } else if (key == prefix && (uint32_t) c <= col_prefix) {
                const int pos = atomicAdd(&s_equal, 1);
                if (pos < rank) {
                    dst[k - rank + pos] = c;
                }
            }
        }
    }
}

static __global__ void top_k_block_all(int * out, int nc, int nr) {
    const size_t i = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < (size_t) nc*nr) { out[i] = i%nc; }
}

template<typename M>
static void top_k_block_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * scores = dst->src[0];
    const int nb = scores->ne[0];
    const int nq = scores->ne[1];
    const int nr = ggml_nrows(scores);
    const int nc = dst->src[1]->ne[0];
    const int k  = dst->ne[0];
    // the graph can ask for the stable selection, and the test switch asks for it everywhere
    const bool preserve_ties = ggml_get_op_params_i32(dst, 0) != 0 || top_k_stable_ties();
    const auto * sc = (const float *) scores->data;
    const auto * map = (const int *) dst->src[1]->data;
    const auto * mask = (const M *) dst->src[2]->data;
    const auto * cells = dst->src[3] ? (const int *) dst->src[3]->data : nullptr;
    const auto * meta = dst->src[4] ? (const int *) dst->src[4]->data : nullptr;
    auto * out = (int *) dst->data;
    auto stream = ctx.stream();
    if (k == nc) {
        top_k_block_all<<<((size_t) nc*nr + 255)/256, 256, 0, stream>>>(out, nc, nr);
        return;
    }
    // Scratch sizes must not follow the current cell count: the legacy pool keeps every
    // best-fit miss, so a request that grows with each prefill ubatch leaves one buffer
    // per size step behind. Round the weights to a power of two and give the histograms
    // their maximum width, so a run touches only a handful of distinct sizes.
    size_t nb_alloc = 1024;
    while (nb_alloc < (size_t) nb + 1) {
        nb_alloc *= 2;
    }
    // One query per stream is the decode case: one workgroup per row selects without a pass
    // over the cells. The choice follows the shapes and the presence of the two extra inputs
    // only, so the captured launch list of a graph cannot go stale.
    if (nq == 1 && cells != nullptr && meta != nullptr) {
        ggml_cuda_pool_alloc<int> weights(ctx.pool(), (size_t) nr*nb_alloc);
        top_k_block_row<<<nr, TOP_K_BLOCK_ROW_NT, 0, stream>>>(sc, map, mask, cells, meta,
                weights.get(), out, nb, nc, nq, (int) (dst->src[3]->ne[0]/nb), k, preserve_ties);
        return;
    }
    const int bp = std::min((nb + 1024)/1024, 64);
    const int cp = std::min((nc + 1023)/1024, 64);
    ggml_cuda_pool_alloc<int> weights(ctx.pool(), (size_t) nr*nb_alloc);
    ggml_cuda_pool_alloc<top_k_radix_state> states(ctx.pool(), nr);
    ggml_cuda_pool_alloc<int> hist(ctx.pool(), (size_t) nr*64*256);
    CUDA_CHECK(cudaMemsetAsync(weights.get(), 0, (size_t) nr*(nb + 1)*sizeof(int), stream));
    top_k_block_count<<<nr, 256, 0, stream>>>(map, mask, weights.get(), nb, nc, nq);
    top_k_radix_init<<<(nr + 255)/256, 256, 0, stream>>>(states.get(), nr, k, 0u);
    for (int shift = 24; shift >= 0; shift -= 8) {
        top_k_block_histogram<<<nr*bp, 256, 0, stream>>>(sc, weights.get(), states.get(), hist.get(), nb, bp, shift);
        top_k_radix_select<256, 8><<<nr, 256, 0, stream>>>(hist.get(), states.get(), bp, shift);
    }
    // the stable selection among the tied cells; the weighted score threshold above does not depend on it
    if (preserve_ties) {
        int bits = 1;
        while (bits < 32 && (1u << bits) < (uint32_t) nc) {
            ++bits;
        }
        const int passes = (bits + 7)/8;
        for (int pass = passes - 1; pass >= 0; --pass) {
            top_k_block_tie_histogram<<<nr*cp, 256, 0, stream>>>(sc, map, mask, states.get(), hist.get(), nb, nc, nq, cp, pass*8);
            top_k_radix_col_select<256, 8><<<nr, 256, 0, stream>>>(hist.get(), states.get(), cp, pass*8, pass == passes - 1);
        }
    }
    top_k_block_gather<<<nr*cp, 256, 0, stream>>>(sc, map, mask, states.get(), out, nb, nc, nq, k, cp, preserve_ties);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k_block(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)
    if (dst->src[2]->type == GGML_TYPE_F16) {
        top_k_block_cuda<half>(ctx, dst);
    } else {
        top_k_block_cuda<float>(ctx, dst);
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("block top-k requires HIP without CUB");
#endif
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}
