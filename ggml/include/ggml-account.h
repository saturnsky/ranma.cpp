#pragma once

// Prefill ubatch time accounting (ranma). RANMA_PREFILL_ACCOUNT=1 turns it on, the default is off.
// One record per ubatch collects host phases, the engram gather and the SSD tier, and is printed as one JSON line
// on stderr: {"kind":"prefill_account","v":1,...}. A fact that was not measured prints as null.
// With the switch off every call returns at once: no clock, no allocation, no output.

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Additive fields (ms or bytes). A field that was never touched prints null.
enum ggml_account_field {
    GGML_ACCOUNT_BEGIN_MS,            // process_ubatch: hint, memory context, choice of the graph
    GGML_ACCOUNT_GRAPH_BUILD_MS,      // build and allocation of a graph that was not reused
    GGML_ACCOUNT_INPUT_MS,            // set_inputs, includes the engram hash and prefetch below
    GGML_ACCOUNT_SUBMIT_MS,           // graph_compute until it returns (asynchronous, not GPU time)
    GGML_ACCOUNT_ENGRAM_HASH_MS,
    GGML_ACCOUNT_ENGRAM_PREFETCH_MS,
    GGML_ACCOUNT_ENGRAM_LOGICAL_BYTES,
    GGML_ACCOUNT_HOST_SYNC_MS,        // existing stream synchronization of the expert-major op staging plan
    GGML_ACCOUNT_OP_STAGE_BYTES,      // expert-major op staging (side stream copies)
    GGML_ACCOUNT_CTRL_STAGE_BYTES,    // expert controller staging (host to device slot copies)
    // GPU stream time between the stamps of RANMA_PREFILL_GPU_STAMP (ggml-cuda/prefill-stamp.cuh): all of it, and its
    // attention, routed MoE (router and experts) and remaining sections
    GGML_ACCOUNT_GPU_STREAM_MS,
    GGML_ACCOUNT_GPU_ATTN_MS,
    GGML_ACCOUNT_GPU_MOE_MS,
    GGML_ACCOUNT_GPU_OTHER_MS,
    // SSD tier, three consecutive values per field: + 0 up, + 1 gate, + 2 down
    GGML_ACCOUNT_SSD_PAYLOAD_BYTES,   // slice bytes read from the file
    GGML_ACCOUNT_SSD_IO_BYTES        = GGML_ACCOUNT_SSD_PAYLOAD_BYTES + 3,   // sector aligned bytes of those reads
    GGML_ACCOUNT_SSD_WORKER_MS       = GGML_ACCOUNT_SSD_IO_BYTES + 3,        // worker waiting for the reads of the kind
    GGML_ACCOUNT_SSD_GPU_MS          = GGML_ACCOUNT_SSD_WORKER_MS + 3,       // compute stream waiting in the wait kernel
    GGML_ACCOUNT_FIELDS              = GGML_ACCOUNT_SSD_GPU_MS + 3,
};

GGML_API bool     ggml_account_enabled(void);

// Context id, monotone from 1. 0 when off.
GGML_API uint64_t ggml_account_next_ctx(void);

// Opens the record of a ubatch and makes it the current one. Returns its key, 0 when off.
GGML_API uint64_t ggml_account_begin(uint64_t ctx, uint64_t ub, int32_t tokens);
// Key of the ubatch being submitted, 0 outside one.
GGML_API uint64_t ggml_account_current(void);
// Tokens of record `key`, -1 when the record is gone.
GGML_API int32_t  ggml_account_tokens(uint64_t key);

// Add to a field of the current record or of record `key`. No effect when the record is gone.
GGML_API void ggml_account_add(enum ggml_account_field field, double value);
GGML_API void ggml_account_add_for(uint64_t key, enum ggml_account_field field, double value);

// A CPU node ran on one thread from t0_us to t1_us. The record keeps the earliest start and the latest end per node
// and prints the sum over nodes, so thread times are never added.
GGML_API void ggml_account_node_span(const void * node, int64_t t0_us, int64_t t1_us);

// A short annotation of the current record, copied. Digits only prints as a number, else as a string.
GGML_API void ggml_account_note(const char * name, const char * value);
GGML_API void ggml_account_note_for(uint64_t key, const char * name, const char * value);

// The device side of record `key` owes n more contributions. The line waits until they are settled.
// expect returns false when the record is gone.
GGML_API bool ggml_account_expect(uint64_t key, int n);
GGML_API void ggml_account_settle(uint64_t key, int n);

// The host side is finished (ok = submitted). The line is printed now, or when the last contribution is settled.
GGML_API void ggml_account_end(uint64_t key, bool ok);

// Prints the finished records that still wait for contributions, marked incomplete, except the newest keep_newest.
GGML_API void ggml_account_flush(int keep_newest);

// A cost that belongs to no ubatch (a server checkpoint): its own line, with the last ctx and ub.
GGML_API void ggml_account_boundary(const char * what, double ms, int64_t n_tokens, int64_t bytes);

#ifdef __cplusplus
}
#endif
