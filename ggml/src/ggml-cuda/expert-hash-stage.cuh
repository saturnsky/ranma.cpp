#pragma once

// VRAM staging of the experts of early-route layers (expert-hash-early.h, RANMA_EXPERT_HASH_EARLY
// vram|both). Each such layer owns a few extra slots at the end of its size class's VRAM arena,
// outside every plan. For a hint the host picks, per layer, the hinted experts that live in host
// memory, and then, before the graph is launched:
//
//   compute stream: stage_target_kernel stores the layer's new staging generation (stream order puts
//                   it before the graph the hint is for, and after every earlier graph);
//   an event on the compute stream, waited for by the side stream, so nothing below starts before
//   every earlier graph (which may still read the staging slots or the table) has finished;
//   side stream:    stage_clear_kernel removes the table entries of the experts whose slots are
//                   reused (only where the table still points at that slot), the slices are copied
//                   host -> slot with the copy engine, and stage_publish_kernel points the table at
//                   the slots and then stores the layer's done generation.
//
// In the graph, stage_wait_kernel runs right before the layer's routed matmuls (where the router ids
// are known) and waits until done has reached target. A graph without a hint finds them equal and
// passes at once. Because the target is written in stream order, a hint that arrives while an
// earlier graph still runs never makes that graph wait for it. A table entry always names a slot
// that holds exactly that expert's bytes: an entry is removed before its slot is rewritten, and an
// install that rewrites the table only removes entries.
//
// The wait kernel also counts, per layer, the selections served by a staging slot and those still
// read from host memory.

#include "common.cuh"

#include <cstdint>

namespace ggml_cuda_expert {

static constexpr int stage_max_list = 32;   // entries of one kernel argument list, and slots per layer

struct stage_list {
    int32_t n = 0;
    int32_t expert[stage_max_list];
    int32_t slot[stage_max_list];            // arena slot index of the layer's class
};

struct stage_targets {
    int32_t  n = 0;
    int32_t  index[8];
    uint32_t value[8];
};

// One early layer, in VRAM.
struct stage_state {
    uint32_t target;               // staging generation the next graph needs (compute stream)
    uint32_t done;                 // staging generation whose table is final (side stream)
    unsigned long long hits;       // selections read from a staging slot
    unsigned long long host;       // selections without a VRAM slot (read from host memory)
    unsigned long long waits;      // launches whose wait found the staging unfinished
    unsigned long long launches;
};

static __global__ void stage_target_kernel(stage_state * states, const stage_targets targets) {
    if (threadIdx.x < (unsigned) targets.n) {
        __hip_atomic_store(&states[targets.index[threadIdx.x]].target, targets.value[threadIdx.x], __ATOMIC_RELEASE,
            __HIP_MEMORY_SCOPE_AGENT);
    }
}

static __global__ void stage_clear_kernel(int32_t * table, const stage_list list) {
    if (threadIdx.x < (unsigned) list.n) {
        atomicCAS(table + list.expert[threadIdx.x], list.slot[threadIdx.x], -1);
    }
}

static __global__ void stage_publish_kernel(int32_t * table, const stage_list list, stage_state * state, const uint32_t value) {
    if (threadIdx.x < (unsigned) list.n) {
        table[list.expert[threadIdx.x]] = list.slot[threadIdx.x];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(&state->done, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// One block. `slot_begin`..`slot_end` are the layer's staging slots in the class arena.
static __global__ void stage_wait_kernel(stage_state * state, const int32_t * ids, const int rows, const int used,
        const int stride, const int experts, const int32_t * table, const int slot_begin, const int slot_end) {
    __shared__ unsigned int hits, host;
    if (threadIdx.x == 0) {
        const uint32_t want = __hip_atomic_load(&state->target, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        bool waited = false;
        while (int32_t(__hip_atomic_load(&state->done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) - want) < 0) {
            waited = true;
            __builtin_amdgcn_s_sleep(1);
        }
        hits = 0;
        host = 0;
        state->waits += waited ? 1 : 0;
        state->launches += 1;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows*used; i += blockDim.x) {
        const int e = ids[(i/used)*stride + i%used];
        if (e < 0 || e >= experts) { continue; }
        const int slot = __hip_atomic_load(table + e, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (slot >= slot_begin && slot < slot_end) {
            atomicAdd(&hits, 1u);
        } else if (slot < 0) {
            atomicAdd(&host, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        state->hits += hits;
        state->host += host;
    }
}

} // namespace ggml_cuda_expert
