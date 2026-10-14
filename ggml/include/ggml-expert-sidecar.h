#pragma once

// The expert sidecar (ranma, RANMA_EXPERT_SIDECAR, docs/ranma/expert-sidecar.md): a backend whose expert cache
// sees the router selections of its models on the device answers GGML_EXPERT_SIDECAR_PROC_NAME with the table
// below. The selections reach host memory without a copy call, and a call is read after a synchronization the
// caller performs anyway.

#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_EXPERT_SIDECAR_PROC_NAME "ggml_backend_expert_sidecar"
#define GGML_EXPERT_SIDECAR_VERSION   1
#define GGML_EXPERT_SIDECAR_BANKS     4 // calls of one model that may be unread at a time

enum ggml_expert_sidecar_mode {
    GGML_EXPERT_SIDECAR_IDS    = 0, // the ids of every row, up to the row cap
    GGML_EXPERT_SIDECAR_COUNTS = 1, // per layer, how often each expert was selected
    GGML_EXPERT_SIDECAR_NONE   = 2, // nothing
};

// Layer block written by end(), the same bytes as in the sidecar file (little-endian, no padding):
//   u16 layer, u16 n_used (IDS) or n_expert (COUNTS), u32 rows, u32 stored, u32 launches,
//   IDS:    u16 ids[stored*n_used], 0 pad to 4 bytes, then with tier u8 tier[stored*n_used], 0 pad to 4 bytes
//   COUNTS: u32 counts[n_expert]
// layer is the model's own layer index (a joint cache offset is removed). tier: 0 VRAM arena, 1 host, 2 file.

struct ggml_expert_sidecar_iface {
    uint32_t version; // GGML_EXPERT_SIDECAR_VERSION

    // Before the first begin: the rows kept per layer and call in IDS mode for decode calls and for prefill
    // calls. The first call sizes the capture; false when the cache is not installed.
    bool (*configure)(int32_t rows_decode, int32_t rows_prefill);

    // A call of the model that computes on `backend`, ordered on its stream before the next compute. Returns a
    // ticket != 0, or 0 when that backend runs no model of the cache. At most GGML_EXPERT_SIDECAR_BANKS tickets
    // of one model may be unread: the caller synchronizes and reads before it begins another.
    uint32_t (*begin)(ggml_backend_t backend, enum ggml_expert_sidecar_mode mode, bool prefill);

    // After the caller synchronized the call of `ticket`: writes its layer blocks to buf and returns their bytes.
    // Writes nothing and returns the bytes needed when cap is smaller; can be called again.
    size_t (*end)(ggml_backend_t backend, uint32_t ticket, bool tier, void * buf, size_t cap,
                  int32_t * n_blocks, bool * truncated);
};

typedef const struct ggml_expert_sidecar_iface * (*ggml_backend_expert_sidecar_t)(void);

#ifdef __cplusplus
}
#endif
