#pragma once

// Device side of the expert sidecar (ggml-expert-sidecar.h). Off unless a caller configures it: the router hook
// then launches one single-block kernel per routed layer that writes the selections into coherent mapped host
// memory. A call is opened by a one-thread kernel on the model's stream that sets the epoch and the bank of the
// model's records, so the record kernels (also those in captured graphs) find their place at run time and a
// later call can be opened before an earlier one was read.

#include "common.cuh"

#include "ggml-expert-sidecar.h"

#include <cstdint>

namespace ggml_cuda_expert {

// per member, in VRAM, written by sidecar_begin_kernel in stream order
struct sidecar_cur {
    uint32_t epoch; // 0: no call open
    uint32_t bank;
    uint32_t mode;  // ggml_expert_sidecar_mode
    uint32_t cap;   // rows kept per layer in this call (IDS)
};

// per bank and joint layer, in VRAM
struct sidecar_state {
    uint32_t epoch;
    int32_t  launches;
    int32_t  rows;
    int32_t  pad;
};

// per bank and joint layer, in mapped host memory; the payload (u16 ids or u32 counts) follows
struct sidecar_rec {
    uint32_t epoch; // written last
    int32_t  launches;
    int32_t  rows;
    int32_t  n_used;
};

static constexpr int sidecar_max_hist = 1024;

static __global__ void sidecar_begin_kernel(sidecar_cur * cur, const uint32_t epoch, const uint32_t bank,
        const uint32_t mode, const uint32_t cap) {
    cur->epoch = epoch;
    cur->bank  = bank;
    cur->mode  = mode;
    cur->cap   = cap;
}

static __global__ void sidecar_ids_kernel(
        const int32_t * __restrict__ ids, const int n_rows, const int row_stride, const int n_used, const int n_experts,
        const sidecar_cur * cur, sidecar_state * state, char * recs, const size_t rec_bytes, const int n_layers,
        const int layer, const uint32_t rec_cap) {
    __shared__ uint32_t s_epoch;
    __shared__ uint32_t s_mode;
    __shared__ uint32_t s_cap;
    __shared__ int      s_base;
    __shared__ int      s_fresh;
    __shared__ uint32_t s_rec;
    __shared__ uint32_t hist[sidecar_max_hist];

    if (threadIdx.x == 0) {
        const uint32_t epoch = cur->epoch;
        s_mode = epoch == 0 ? (uint32_t) GGML_EXPERT_SIDECAR_NONE : cur->mode;
        if (s_mode != (uint32_t) GGML_EXPERT_SIDECAR_NONE) {
            const uint32_t bank = cur->bank;
            const uint32_t idx  = bank*(uint32_t) n_layers + (uint32_t) layer;
            sidecar_state & st = state[idx];
            s_fresh = st.epoch != epoch;
            if (s_fresh) {
                st.epoch    = epoch;
                st.launches = 0;
                st.rows     = 0;
            }
            s_base = st.rows;
            st.launches += 1;
            st.rows     += n_rows;
            sidecar_rec * rec = (sidecar_rec *) (recs + idx*rec_bytes);
            rec->launches = st.launches;
            rec->rows     = st.rows;
            rec->n_used   = n_used;
            s_epoch = epoch;
            s_cap   = cur->cap;
            s_rec   = idx;
        }
    }
    __syncthreads();
    if (s_mode == (uint32_t) GGML_EXPERT_SIDECAR_NONE) {
        return;
    }
    sidecar_rec * rec = (sidecar_rec *) (recs + s_rec*rec_bytes);
    const int n = n_rows*n_used;
    if (s_mode == (uint32_t) GGML_EXPERT_SIDECAR_IDS) {
        uint16_t * dst = (uint16_t *) (rec + 1);
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int d = s_base*n_used + i;
            if (d < (int) (s_cap*(uint32_t) n_used) && d < (int) rec_cap) {
                const int row = i/n_used;
                const int e   = ids[row*row_stride + (i - row*n_used)];
                dst[d] = e < 0 || e >= n_experts ? (uint16_t) 0xFFFF : (uint16_t) e;
            }
        }
    } else if (n_experts <= sidecar_max_hist) {
        uint32_t * counts = (uint32_t *) (rec + 1);
        for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
            hist[e] = 0;
            if (s_fresh) {
                counts[e] = 0;
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const int row = i/n_used;
            const int e   = ids[row*row_stride + (i - row*n_used)];
            if (e >= 0 && e < n_experts) {
                atomicAdd(&hist[e], 1u);
            }
        }
        __syncthreads();
        // one thread per expert: the read-modify-write of the host counts does not race
        for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
            if (hist[e] != 0) {
                counts[e] += hist[e];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // the epoch last, so that a record with the call's epoch has its payload in place
        __threadfence_system();
        rec->epoch = s_epoch;
    }
}

} // namespace ggml_cuda_expert
