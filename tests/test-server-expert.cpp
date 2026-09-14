// ranma: the server-side policy of the MoE expert cache (common/expert-policy.h).
//
// The policy is driven against a fake backend that records every call, so the whole test runs
// without a GPU and without a model. Each case states the call sequence it expects as data.

#include "expert-policy.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)

// ---------------------------------------------------------------------------------------------
// fake backend

struct fake_backend {
    std::vector<std::string> calls;

    std::map<std::string, ggml_expert_bank_id> bank_ids;
    std::map<ggml_expert_bank_id, std::string> bank_labels;
    std::map<ggml_expert_bank_id, ggml_expert_plan_id> newest;

    // stand-in for the selection set of a plan: two plans with the same content install once
    std::map<ggml_expert_plan_id, uint64_t> content;

    ggml_expert_plan_id next_plan = 100;
    ggml_expert_plan_id installed = GGML_EXPERT_PLAN_NONE;

    bool available_ = true;
    bool freeze     = false;
    bool refuse     = false; // the backend reports compute in flight

    void reset() {
        *this = fake_backend();
    }
};

static fake_backend g_fake;

static void record(const std::string & line) {
    g_fake.calls.push_back(line);
}

static bool fake_available(const llama_context * ctx) {
    return ctx != nullptr && g_fake.available_;
}

static bool fake_bank_open(llama_context *, const char * label, bool, ggml_expert_bank_id * out) {
    const auto it = g_fake.bank_ids.find(label);
    if (it != g_fake.bank_ids.end()) {
        *out = it->second;
        return true;
    }
    const ggml_expert_bank_id id = (ggml_expert_bank_id) g_fake.bank_ids.size();
    g_fake.bank_ids[label]  = id;
    g_fake.bank_labels[id]  = label;
    *out = id;
    record("open(" + std::string(label) + ")=" + std::to_string(id));
    return true;
}

static bool fake_bank_mark(llama_context *, ggml_expert_bank_id bank) {
    record("mark(" + g_fake.bank_labels[bank] + ")");
    return true;
}

static bool fake_bank_discard(llama_context *, ggml_expert_bank_id bank) {
    record("discard(" + g_fake.bank_labels[bank] + ")");
    return true;
}

static bool fake_bank_commit(llama_context *, ggml_expert_bank_id bank, const ggml_expert_record * rec,
                             ggml_expert_plan_id * out) {
    const ggml_expert_plan_id id = g_fake.next_plan++;
    // the content of a plan stands for its selection set: the same interval gives the same plan
    g_fake.content[id]  = rec->bank_tokens;
    g_fake.newest[bank] = id;
    *out = id;
    record("commit(" + g_fake.bank_labels[bank] +
           ",in=" + std::to_string(rec->input_tokens) +
           ",out=" + std::to_string(rec->output_tokens) +
           ",tok=" + std::to_string(rec->bank_tokens) + ")->" + std::to_string(id));
    return true;
}

static bool fake_plan_install(llama_context *, ggml_expert_plan_id plan) {
    if (g_fake.freeze) {
        record("install(" + std::to_string(plan) + ") refused: frozen");
        return false;
    }
    if (g_fake.refuse) {
        record("install(" + std::to_string(plan) + ") refused: busy");
        return false;
    }
    if (g_fake.installed != GGML_EXPERT_PLAN_NONE && g_fake.content[g_fake.installed] == g_fake.content[plan]) {
        g_fake.installed = plan;
        record("install(" + std::to_string(plan) + ") no-op");
        return true;
    }
    g_fake.installed = plan;
    record("install(" + std::to_string(plan) + ")");
    return true;
}

static void fake_set_profiled_seq(llama_context *, llama_seq_id seq, ggml_expert_bank_id bank) {
    const std::string bank_name = bank == GGML_EXPERT_BANK_NONE ? "none" : g_fake.bank_labels[bank];
    record("seq(" + std::to_string((int) seq) + "," + bank_name + ")");
}

static const common_expert_backend fake_table = {
    fake_available,
    fake_bank_open,
    fake_bank_mark,
    fake_bank_commit,
    fake_bank_discard,
    fake_plan_install,
    fake_set_profiled_seq,
};

// ---------------------------------------------------------------------------------------------
// helpers

static llama_context * const ctx = (llama_context *) (intptr_t) 1; // never dereferenced

// Every case starts here: the banks are opened and the warm-up rows are discarded.
static const std::vector<std::string> init_calls = {
    "open(decode)=0",
    "open(prefill)=1",
    "discard(decode)",
    "discard(prefill)",
};

static std::vector<std::string> with_init(const std::vector<std::string> & rest) {
    std::vector<std::string> out = init_calls;
    out.insert(out.end(), rest.begin(), rest.end());
    return out;
}

static bool same(const char * name, const std::vector<std::string> & expected) {
    if (g_fake.calls == expected) {
        return true;
    }
    fprintf(stderr, "case %s: call sequence differs\n", name);
    const size_t n = expected.size() > g_fake.calls.size() ? expected.size() : g_fake.calls.size();
    for (size_t i = 0; i < n; ++i) {
        const char * want = i < expected.size()      ? expected[i].c_str()      : "<end>";
        const char * got  = i < g_fake.calls.size()  ? g_fake.calls[i].c_str()  : "<end>";
        fprintf(stderr, "  %2zu  want %-52s got %s\n", i, want, got);
    }
    return false;
}

static common_expert start(bool swap, bool freeze) {
    g_fake.reset();
    g_fake.freeze = freeze;

    common_expert_params params;
    params.l1_mib    = 3072;
    params.prefill_swap = swap;
    params.freeze       = freeze;

    common_expert policy;
    policy.init(ctx, params, fake_table);
    return policy;
}

int main() {
    // (a) swap off, one request: profile both phases, install the newest decode plan at the end
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, /*other_slots_busy =*/ false);

        CHECK(same("a", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->101",
            "install(101)",
        })));
    }

    // (b) swap on, two consecutive requests. The first request has no decode plan yet, so its
    //     prompt-to-generation boundary installs nothing; from the second request on every request
    //     performs two installs.
    {
        common_expert policy = start(/*swap =*/ true, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 40);
        policy.on_request_end(0, 40, 16, false);

        CHECK(same("b", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->101",
            "install(100)",                                   // next phase is prompt processing
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=40,out=0,tok=40)->102",
            "install(101)",                                   // next phase is generation
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=40,out=16,tok=16)->103",
            "install(102)",
        })));
    }

    // (c) Frozen profiling commits records without attempting an install.
    {
        common_expert policy = start(/*swap =*/ true, /*freeze =*/ true);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, false);

        CHECK(same("c", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->101",
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->102",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->103",
        })));
        const auto calls = g_fake.calls.size();
        policy.on_all_idle();
        CHECK(g_fake.calls.size() == calls);
    }

    // (d) swap off, n_parallel = 2: slot 0 ends while slot 1 computes. The install never reaches
    //     the backend at the request end; it is performed on the first all-idle pass. Slot 1 is
    //     never profiled.
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(1);
        policy.on_generation_start(1, 77);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, /*other_slots_busy =*/ true);
        policy.on_request_end(1, 77, 20, false);              // slot 1 is not profiled
        policy.on_all_idle();                                 // the server saw every slot idle
        policy.on_all_idle();                                 // nothing is left to install

        CHECK(same("d", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->101",
            "install(101)",                                   // performed by the all-idle pass
        })));
    }

    // (e) the warm-up rows enter no bank: the only calls before the first request are the two
    //     bank_open calls and the two discards
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        CHECK(same("e", init_calls));
        policy.on_all_idle();
        CHECK(same("e idle", init_calls));
    }

    // (f) the prompt was fully served from the prompt cache: the prefill interval saw no token and
    //     is discarded, the decode interval is a record
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 0);
        policy.on_request_end(0, 0, 12, false);

        CHECK(same("f", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "discard(prefill)",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=0,out=12,tok=12)->100",
            "install(100)",
        })));
    }

    // (g) the client went away mid-generation: the tokens generated so far are a record. A request
    //     that produced nothing is a discard and installs nothing.
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 5, false);

        CHECK(same("g", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=5,tok=5)->101",
            "install(101)",
        })));
    }
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 0, false);

        CHECK(same("g zero", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "discard(decode)",
        })));
    }

    // (h) two identical requests produce the same plan: the second install is a no-op in the backend
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_request_end(0, 30, 12, false);

        CHECK(same("h", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->101",
            "install(101)",
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->102",
            "mark(decode)",
            "seq(-1,none)",
            "commit(decode,in=30,out=12,tok=12)->103",
            "install(103) no-op",
        })));
    }

    // an interrupted request discards whatever interval is open
    {
        common_expert policy = start(/*swap =*/ false, /*freeze =*/ false);
        policy.on_prompt_start(0);
        policy.on_generation_start(0, 30);
        policy.on_interrupted(0);
        policy.on_request_end(0, 30, 7, false);

        CHECK(same("interrupted", with_init({
            "mark(prefill)",
            "seq(0,prefill)",
            "seq(0,decode)",
            "commit(prefill,in=30,out=0,tok=30)->100",
            "mark(decode)",
            "seq(-1,none)",
            "discard(decode)",
            "seq(-1,none)",
        })));
    }

    // A completed PP-only benchmark records prefill. Interrupted server prompts still discard it.
    {
        common_expert policy = start(true, false);
        policy.on_prompt_start(0);
        policy.on_request_end(0, 512, 0, false, true);
        CHECK(same("completed PP", with_init({
            "mark(prefill)", "seq(0,prefill)", "seq(-1,none)",
            "commit(prefill,in=512,out=0,tok=512)->100", "install(100)",
        })));
    }

    printf("OK\n");
    return 0;
}
