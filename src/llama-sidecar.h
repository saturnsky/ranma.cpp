#pragma once

// ranma: the expert sidecar (RANMA_EXPERT_SIDECAR=<path>, docs/ranma/expert-sidecar.md). A low-cost binary record
// of every decode call: its tokens in batch order and, per routed layer, the experts the router selected for every
// row in top-k order, plus the server's request and speculation events. Off (and free) unless the variable is set.
// Staging API like llama-ext.h; the server includes this header directly.

#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>

enum llama_sidecar_phase {
    LLAMA_SIDECAR_PHASE_AUTO    = 0, // a call of one token is a decode, a longer one a prefill
    LLAMA_SIDECAR_PHASE_PREFILL = 1,
    LLAMA_SIDECAR_PHASE_DECODE  = 2,
    LLAMA_SIDECAR_PHASE_VERIFY  = 3, // a target call that checks draft tokens
    LLAMA_SIDECAR_PHASE_DRAFT   = 4, // a call of the draft model
    LLAMA_SIDECAR_PHASE_MIXED   = 5, // prompt and generated tokens in one call
};

enum llama_sidecar_event_kind {
    LLAMA_SIDECAR_EVENT_REQ_BEGIN = 1, // a: seq id, b: prompt tokens, c: reused (cached) tokens
    LLAMA_SIDECAR_EVENT_REQ_END   = 2, // a: generated tokens, b: drafted tokens, c: accepted tokens
    LLAMA_SIDECAR_EVENT_ACCEPT    = 3, // a: pos of the first verified token, b: drafted, c: accepted, d: call_seq of the verify call
};

// Whether RANMA_EXPERT_SIDECAR is set (read once).
LLAMA_API bool llama_sidecar_enabled(void);

// The request and phase of the following decode calls of ctx (sticky until changed). No-op when off.
LLAMA_API void llama_sidecar_set_phase(struct llama_context * ctx, int32_t req_id, enum llama_sidecar_phase phase);

// The call_seq of the last decode call of ctx, -1 when none or off.
LLAMA_API int64_t llama_sidecar_call_seq(const struct llama_context * ctx);

// The ctx_id of ctx in the sidecar file, -1 when off.
LLAMA_API int32_t llama_sidecar_ctx_id(const struct llama_context * ctx);

LLAMA_API void llama_sidecar_event(struct llama_context * ctx, enum llama_sidecar_event_kind kind, int32_t req_id,
                                   int64_t a, int64_t b, int64_t c, int64_t d);

// A key/value string for the file (build, settings).
LLAMA_API void llama_sidecar_note(const char * key, const char * value);

// Hands the open block to the writer and waits until everything is on disk.
LLAMA_API void llama_sidecar_flush(void);

// ---- internal (llama_context) ----

struct llama_batch_ext;
struct llama_sidecar_ctx;

struct llama_sidecar_ctx_deleter {
    void operator()(llama_sidecar_ctx * p) const;
};
using llama_sidecar_ctx_ptr = std::unique_ptr<llama_sidecar_ctx, llama_sidecar_ctx_deleter>;

// Null when the sidecar is off. Picks the source (device: the backend's expert cache; eval: the scheduler's eval
// callback, which it then returns in *cb / *cb_ud for the context to install; never for a context that computes on
// a GPU, where the callback would synchronize every split).
llama_sidecar_ctx_ptr llama_sidecar_ctx_init(struct llama_context * lctx, ggml_backend_t expert_backend, bool gpu_compute,
                                             ggml_backend_sched_eval_callback * cb, void ** cb_ud);

// At the start of a decode call; may synchronize `sched` when too many calls are unread.
void llama_sidecar_call_begin(llama_sidecar_ctx * sc, const llama_batch_ext & batch, ggml_backend_sched_t sched);
// At its end (any return path).
void llama_sidecar_call_end(llama_sidecar_ctx * sc);
// After the context synchronized: reads and writes the finished calls (not while a decode call is running).
void llama_sidecar_collect(llama_sidecar_ctx * sc);

// Brackets one decode call.
struct llama_sidecar_scope {
    llama_sidecar_ctx * sc;
    llama_sidecar_scope(llama_sidecar_ctx * sc, const llama_batch_ext & batch, ggml_backend_sched_t sched) : sc(sc) {
        if (sc) { llama_sidecar_call_begin(sc, batch, sched); }
    }
    ~llama_sidecar_scope() {
        if (sc) { llama_sidecar_call_end(sc); }
    }
};
