#pragma once

// Shared server and benchmark policy for the MoE expert cache (docs/ranma/expert-cache-banks.md).
//
// The backend owns histograms ("banks") of router selections and the plans derived from them. It
// knows bank ids and plan ids and nothing else. This policy names the moments: it profiles slot 0,
// keeps one bank for prompt processing and one for generation, commits one record per phase per
// request, and installs the plan of the phase the server is about to enter.
//
//   model load end                  the backend installs the seed plan (config initial_bank)
//   warm-up finished                discard both banks
//   prompt processing starts        mark prefill, profile slot 0 into it
//   generation starts               profile slot 0 into decode, commit prefill,
//                                   swap on: install the newest decode plan, then mark decode
//   request ends                    stop profiling, commit decode, install the plan of the next
//                                   phase (swap on: prefill, swap off: decode)
//   decode error / slot reset       discard whatever bank is open
//
// With the swap off the prefill bank is still marked and committed, so a prompt-processing history
// exists the moment the swap is turned on; no plan is ever installed from it, and the only install
// is the newest decode plan at the end of every request.
//
// An install that the backend refuses (compute in flight, only reachable with n_parallel > 1) is
// remembered and retried on the first update_slots pass in which every slot is idle. There is no
// timer anywhere in this file.
//
// The llama entry points are reached through a table of function pointers so that the policy can be
// driven against a fake backend without a GPU (tests/test-server-expert.cpp). Production code uses
// the default table and does not mention it.

#include "ggml-expert.h"
#include "llama.h"

#include "log.h"

#include <cinttypes>
#include <cstdint>

#define COMMON_EXPERT_INF(fmt, ...) LOG_INF("expert policy: " fmt, __VA_ARGS__)
#define COMMON_EXPERT_DBG(fmt, ...) LOG_DBG("expert policy: " fmt, __VA_ARGS__)

// The options this policy needs. The server fills it from common_params; keeping it separate is what
// lets the test build this header without the server's command line.
struct common_expert_params {
    int  l1_mib    = 0;
    bool prefill_swap = false;
    bool freeze       = false;
};

struct common_expert_backend {
    bool (*available)       (const llama_context *);
    bool (*bank_open)       (llama_context *, const char *, bool, ggml_expert_bank_id *);
    bool (*bank_mark)       (llama_context *, ggml_expert_bank_id);
    bool (*bank_commit)     (llama_context *, ggml_expert_bank_id, const ggml_expert_record *, ggml_expert_plan_id *);
    bool (*bank_discard)    (llama_context *, ggml_expert_bank_id);
    bool (*plan_install)    (llama_context *, ggml_expert_plan_id);
    void (*set_profiled_seq)(llama_context *, llama_seq_id, ggml_expert_bank_id);
};

inline const common_expert_backend & common_expert_llama_backend() {
    static const common_expert_backend backend = {
        llama_expert_available,
        llama_expert_bank_open,
        llama_expert_bank_mark,
        llama_expert_bank_commit,
        llama_expert_bank_discard,
        llama_expert_plan_install,
        llama_expert_set_profiled_seq,
    };
    return backend;
}

struct common_expert {
    // after the model is loaded; also reached when the server resumes from the sleeping state
    void init(llama_context * ctx_new, const common_expert_params & params,
              const common_expert_backend & backend = common_expert_llama_backend()) {
        release();
        be = &backend;

        if (ctx_new == nullptr || !be->available(ctx_new)) {
            if (params.l1_mib > 0) {
                COMMON_EXPERT_INF("%s", "not available for this model, running without it\n");
            }
            be = nullptr;
            return;
        }

        ggml_expert_bank_id decode  = GGML_EXPERT_BANK_NONE;
        ggml_expert_bank_id prefill = GGML_EXPERT_BANK_NONE;
        if (!be->bank_open(ctx_new, label_decode, false, &decode) || !be->bank_open(ctx_new, label_prefill, true, &prefill)) {
            COMMON_EXPERT_INF("%s", "cannot open the 'decode' and 'prefill' profile banks, running without it\n");
            be = nullptr;
            return;
        }

        ctx          = ctx_new;
        bank_decode  = decode;
        bank_prefill = prefill;
        swap         = params.prefill_swap;
        frozen       = params.freeze;

        // The load-time warm-up of this tree runs inside the model load, before these banks exist,
        // and nothing is profiled until a request selects a sequence; discarding both here is the
        // first moment at which the rule "no warm-up row enters a bank" can be enforced.
        on_warmup_done();

        COMMON_EXPERT_INF("active, budget = %d MiB, banks '%s' and '%s', prefill swap %s%s\n",
            params.l1_mib, label_decode, label_prefill, swap ? "on" : "off",
            frozen ? ", frozen (profile only)" : "");
    }

    // A benchmark can recreate the context while the model and its bank/plan ids remain live.
    void rebind(llama_context * context) { ctx = context; }
    bool ready() const { return active(); }

    // before the model is freed
    void release() {
        ctx            = nullptr;
        be             = nullptr;
        bank_decode    = GGML_EXPERT_BANK_NONE;
        bank_prefill   = GGML_EXPERT_BANK_NONE;
        plan_decode    = GGML_EXPERT_PLAN_NONE;
        plan_prefill   = GGML_EXPERT_PLAN_NONE;
        deferred_plan  = GGML_EXPERT_PLAN_NONE;
        deferred_label = nullptr;
        marked_decode  = false;
        marked_prefill = false;
        swap           = false;
        frozen         = false;
    }

    // the server finished whatever warm-up it does: those rows belong to no bank
    void on_warmup_done() {
        if (!active()) {
            return;
        }
        be->bank_discard(ctx, bank_decode);
        be->bank_discard(ctx, bank_prefill);
        marked_decode  = false;
        marked_prefill = false;
    }

    // the slot begins prompt processing (SLOT_STATE_STARTED, first prompt batch)
    void on_prompt_start(int slot_id) {
        if (!active() || slot_id != id_slot_profiled) {
            return;
        }
        be->bank_mark(ctx, bank_prefill);
        marked_prefill = true;
        be->set_profiled_seq(ctx, slot_id, bank_prefill);
    }

    // the prompt is processed and the slot begins generating
    void on_generation_start(int slot_id, uint64_t n_prompt_processed) {
        if (!active() || slot_id != id_slot_profiled) {
            return;
        }
        be->set_profiled_seq(ctx, slot_id, bank_decode);

        if (marked_prefill) {
            marked_prefill = false;
            ggml_expert_record record = {};
            record.request_count = 1;
            record.input_tokens  = n_prompt_processed;
            record.output_tokens = 0;
            record.bank_tokens   = n_prompt_processed;
            commit(bank_prefill, label_prefill, record, plan_prefill);
        }

        if (swap) {
            install(plan_decode, label_decode, "prompt processing -> generation");
        }

        be->bank_mark(ctx, bank_decode);
        marked_decode = true;
    }

    // the request ended, normally or because the client went away
    void on_request_end(int slot_id, uint64_t n_prompt_tokens, uint64_t n_generated, bool other_slots_busy, bool prompt_completed = false) {
        if (!active() || slot_id != id_slot_profiled) {
            return;
        }
        be->set_profiled_seq(ctx, -1, GGML_EXPERT_BANK_NONE);

        if (marked_prefill) {
            marked_prefill = false;
            if (prompt_completed) {
                ggml_expert_record record = {1, n_prompt_tokens, 0, n_prompt_tokens};
                commit(bank_prefill, label_prefill, record, plan_prefill);
            } else { be->bank_discard(ctx, bank_prefill); }
        }

        if (marked_decode) {
            marked_decode = false;
            ggml_expert_record record = {};
            record.request_count = 1;
            record.input_tokens  = n_prompt_tokens;
            record.output_tokens = n_generated;
            record.bank_tokens   = n_generated;
            commit(bank_decode, label_decode, record, plan_decode);
        }

        // the plan of the phase the server enters next: with the swap on the next thing this cache
        // sees is another prompt, without it another generation
        if (swap) {
            install(plan_prefill, label_prefill, "generation -> prompt processing", other_slots_busy);
        } else {
            install(plan_decode, label_decode, "request end", other_slots_busy);
        }
    }

    // a decode failed or a slot was reset before the phase it was in could be committed
    void on_interrupted(int slot_id) {
        if (!active() || slot_id != id_slot_profiled) {
            return;
        }
        be->set_profiled_seq(ctx, -1, GGML_EXPERT_BANK_NONE);
        if (marked_prefill) {
            marked_prefill = false;
            be->bank_discard(ctx, bank_prefill);
        }
        if (marked_decode) {
            marked_decode = false;
            be->bank_discard(ctx, bank_decode);
        }
        COMMON_EXPERT_DBG("%s", "the interval of the interrupted request was discarded\n");
    }

    // every update_slots pass in which all slots are idle
    void on_all_idle() {
        if (!active() || deferred_plan == GGML_EXPERT_PLAN_NONE) {
            return;
        }
        const ggml_expert_plan_id plan  = deferred_plan;
        const char *              label = deferred_label != nullptr ? deferred_label : label_decode;
        deferred_plan  = GGML_EXPERT_PLAN_NONE;
        deferred_label = nullptr;
        install(plan, label, "deferred, every slot is idle");
    }

private:
    bool active() const { return ctx != nullptr && be != nullptr && bank_decode != GGML_EXPERT_BANK_NONE; }

    // A bank whose interval saw no token is not a record: the prompt came out of the prompt cache,
    // or the request produced nothing. Storing it would tell the score window that every expert was
    // unwanted for one request.
    void commit(ggml_expert_bank_id bank, const char * label, const ggml_expert_record & record,
                ggml_expert_plan_id & newest) {
        if (record.bank_tokens == 0) {
            be->bank_discard(ctx, bank);
            COMMON_EXPERT_DBG("bank '%s' saw no token, the interval was discarded\n", label);
            return;
        }
        ggml_expert_plan_id plan = GGML_EXPERT_PLAN_NONE;
        if (!be->bank_commit(ctx, bank, &record, &plan) || plan == GGML_EXPERT_PLAN_NONE) {
            COMMON_EXPERT_DBG("bank '%s' commit produced no plan\n", label);
            return;
        }
        newest = plan;
        COMMON_EXPERT_DBG("bank '%s' committed %" PRIu64 " tokens, newest plan is %u\n",
            label, record.bank_tokens, (unsigned) plan);
    }

    // `reason` says why this plan is wanted now; the backend logs the slices, MiB and ms of the
    // install itself on the line that follows this one.
    void install(ggml_expert_plan_id plan, const char * label, const char * reason, bool other_slots_busy = false) {
        // --expert-freeze means no install at all, so the refusal is here and not only in the
        // backend: nothing is deferred and nothing is retried (docs/ranma/expert-cache-banks.md).
        if (frozen || plan == GGML_EXPERT_PLAN_NONE) { return; }
        if (other_slots_busy) {
            deferred_plan  = plan;
            deferred_label = label;
            COMMON_EXPERT_INF("plan %u of '%s' deferred at the %s boundary: another slot is computing\n",
                (unsigned) plan, label, reason);
            return;
        }
        COMMON_EXPERT_INF("installing plan %u of '%s' (%s)\n", (unsigned) plan, label, reason);
        if (!be->plan_install(ctx, plan)) {
            deferred_plan  = plan;
            deferred_label = label;
            COMMON_EXPERT_INF("plan %u of '%s' was not installed; it is retried once every slot is idle\n",
                (unsigned) plan, label);
        }
    }

    static constexpr const char * label_decode     = "decode";
    static constexpr const char * label_prefill    = "prefill";
    static constexpr int          id_slot_profiled = 0; // the server profiles slot 0 only

    llama_context *               ctx = nullptr;
    const common_expert_backend * be  = nullptr;

    ggml_expert_bank_id bank_decode  = GGML_EXPERT_BANK_NONE;
    ggml_expert_bank_id bank_prefill = GGML_EXPERT_BANK_NONE;

    ggml_expert_plan_id plan_decode   = GGML_EXPERT_PLAN_NONE; // newest plan of each bank
    ggml_expert_plan_id plan_prefill  = GGML_EXPERT_PLAN_NONE;
    ggml_expert_plan_id deferred_plan  = GGML_EXPERT_PLAN_NONE;
    const char *        deferred_label = nullptr;

    bool marked_decode  = false; // an interval of this bank is open
    bool marked_prefill = false;
    bool swap           = false;
    bool frozen         = false;
};
