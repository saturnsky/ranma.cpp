// Layer-major prefill on a tiny random DeepSeek-V4.1 model (CPU, no real model).
//
// A batch of several ubatches computed layer by layer must give the logits of the ubatch loop with the same ubatch size
// bit for bit. The model has the V4.1 roles that cross layers: a ratio-2 KV source (with the pooling gate) whose
// compressed rows and index keys the next two layers read and whose top-k they reuse (a tensor carried two segments
// on), a ratio-1 KV source that is also the candidate source, a later index source that reads its index keys and its
// candidate keep, and two engram layers. With the tail prefill on the layers
// before its first layer run layer by layer and the tail layers in the last segment of every ubatch; when it is off
// for a call (the layer inputs of every row are extracted) every layer runs layer by layer.
//
// Cases: prompts that cross the window, the compressor groups and the ubatch size at odd places; a second call on a
// prefix; the next tokens decoded after the prefill (the KV, compressor and tail records it left);
// groups of ubatches (layer_exec_group_tokens) and MoE bundles (layer_exec_moe_tokens); a partial checkpoint
// taken after a call without outputs, restored and continued as the server does; one attention chunk per ubatch (host
// masks); and with --long the shapes of 32K (ubatch 2048) and 128K (ubatch 3584) prompts.
//
// usage: test-layer-exec-dsv41 [-s SEED] [--keep model.gguf] [--quick] [--long] [--only N1 N2 UB]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "speculative.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"
#include "ggml-expert.h"
#include "../ggml/src/ggml-cuda/expert-lend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 64;
static const uint32_t N_HEAD  = 4;
static const uint32_t N_HD    = 128;
static const uint32_t N_QLORA = 32;
static const uint32_t N_OGRP  = 2;
static const uint32_t N_OLORA = 16;
static const uint32_t N_HC    = 4;
static const uint32_t N_EXP   = 8;
static const uint32_t N_FFEXP = 32;
static const uint32_t N_IDXH  = 4;
static const uint32_t N_IDXD  = 64;
static const uint32_t N_SWA   = 16;
static uint32_t N_CTX = 4096;
static const uint32_t N_LAYER = 8;
// sources: 2 (ratio 2, read by 3 and 4), 5 (ratio 1, read by 6 and 7; candidate source); index sources 2, 5, 7
static const uint32_t RATIOS[N_LAYER] = { 0, 0, 2, 2, 2, 1, 1, 1 };
static const uint32_t ENGRAM_LAYERS[2] = { 1, 3 };
static const uint32_t EG_HEADS = 2, EG_KEY = 16, EG_NGRAM = 3;

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) { setenv(name, value, 1); } else { unsetenv(name); }
#endif
}

// the layer mode's decisions for the process (llama_layer_mode_config), switched between the contexts of a case
static void set_layer_exec(bool on) {
    llama_layer_mode_config c = llama_layer_mode_get_config();
    c.layer_exec = on;
    llama_layer_mode_set_config(c);
}

struct gen_t {
    std::mt19937_64 rng;
    explicit gen_t(uint64_t s) : rng(s) {}
};

static void add_tensor(gguf_context * g, ggml_context * ctx, gen_t & gen, const std::string & name,
        std::vector<int64_t> ne, float stdev, float mean = 0.0f) {
    ggml_tensor * t = nullptr;
    switch (ne.size()) {
        case 1: t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]); break;
        case 2: t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]); break;
        default: t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2]); break;
    }
    ggml_set_name(t, name.c_str());
    std::normal_distribution<float> dis(mean, stdev);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        d[i] = dis(gen.rng);
    }
    gguf_add_tensor(g, t);
}

static bool write_model(const std::string & path, uint64_t seed) {
    gen_t gen(seed);
    gguf_context * g = gguf_init_empty();
    const std::string a = "deepseek41";
    auto key = [&](const char * k) { return a + "." + k; };
    auto u32 = [&](const std::string & k, uint32_t v) { gguf_set_val_u32(g, k.c_str(), v); };
    auto f32 = [&](const std::string & k, float v) { gguf_set_val_f32(g, k.c_str(), v); };
    auto arr_u32 = [&](const std::string & k, const std::vector<uint32_t> & v) {
        gguf_set_arr_data(g, k.c_str(), GGUF_TYPE_UINT32, v.data(), v.size());
    };
    auto arr_u64 = [&](const std::string & k, const std::vector<uint64_t> & v) {
        gguf_set_arr_data(g, k.c_str(), GGUF_TYPE_UINT64, v.data(), v.size());
    };

    gguf_set_val_str(g, "general.architecture", a.c_str());
    u32(key("vocab_size"), N_VOCAB);
    u32(key("context_length"), N_CTX);
    u32(key("embedding_length"), N_EMBD);
    u32(key("block_count"), N_LAYER);
    u32(key("feed_forward_length"), 64);
    u32(key("attention.head_count"), N_HEAD);
    u32(key("attention.head_count_kv"), 1);
    u32(key("attention.key_length"), N_HD);
    u32(key("attention.value_length"), N_HD);
    u32(key("rope.dimension_count"), N_HD / 2);
    f32(key("rope.freq_base"), 10000.0f);
    f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    u32(key("attention.sliding_window"), N_SWA);
    u32(key("attention.q_lora_rank"), N_QLORA);
    u32(key("expert_feed_forward_length"), N_FFEXP);
    u32(key("expert_shared_count"), 1);
    u32(key("expert_count"), N_EXP);
    u32(key("expert_used_count"), 2);
    u32(key("expert_gating_func"), 4);
    f32(key("expert_weights_scale"), 1.5f);
    gguf_set_val_bool(g, key("expert_weights_norm").c_str(), true);
    f32(key("swiglu_clamp_exp"), 10.0f);
    u32(key("attention.indexer.head_count"), N_IDXH);
    u32(key("attention.indexer.key_length"), N_IDXD);
    u32(key("attention.indexer.top_k"), 8);
    u32(key("attention.output_group_count"), N_OGRP);
    u32(key("attention.output_lora_rank"), N_OLORA);
    f32(key("attention.compress_rope_freq_base"), 160000.0f);
    u32(key("hyper_connection.count"), N_HC);
    u32(key("hyper_connection.sinkhorn_iterations"), 2);
    f32(key("hyper_connection.epsilon"), 1e-6f);
    arr_u32(key("attention.compress_ratios"), std::vector<uint32_t>(RATIOS, RATIOS + N_LAYER));
    u32(key("attention.indexer.candidate_source_layer"), 5);
    u32(key("attention.indexer.candidate_topk_blocks"), 4);
    u32(key("attention.indexer.candidate_block_size"), 8);

    const uint32_t n_cols = (EG_NGRAM - 1) * EG_HEADS;
    arr_u32(key("engram.layer_ids"), std::vector<uint32_t>(ENGRAM_LAYERS, ENGRAM_LAYERS + 2));
    u32(key("engram.head_count"), EG_HEADS);
    u32(key("engram.key_length"), EG_KEY);
    u32(key("engram.max_ngram_size"), EG_NGRAM);
    u32(key("engram.pad_id"), 2);
    std::vector<uint64_t> mult, prime, offs;
    const uint64_t primes_pool[4] = { 97, 89, 83, 79 };
    uint64_t rows_needed = 0;
    for (uint32_t e = 0; e < 2; e++) {
        for (uint32_t i = 0; i < EG_NGRAM; i++) {
            mult.push_back((gen.rng() | 1) & 0xffffffffULL);
        }
        uint64_t off = 0;
        for (uint32_t b = 0; b < n_cols; b++) {
            prime.push_back(primes_pool[b % 4]);
            offs.push_back(off);
            off += primes_pool[b % 4];
        }
        rows_needed = std::max(rows_needed, off);
    }
    arr_u64(key("engram.multipliers"), mult);
    arr_u64(key("engram.primes"), prime);
    arr_u64(key("engram.offsets"), offs);
    {
        std::vector<uint32_t> map(N_VOCAB);
        for (uint32_t i = 0; i < N_VOCAB; i++) { map[i] = i % 64; }
        arr_u32(key("engram.token_map"), map);
    }

    gguf_set_val_str(g, "tokenizer.ggml.model", "test");
    {
        std::vector<std::string> toks(N_VOCAB);
        std::vector<const char *> tp(N_VOCAB);
        std::vector<float> scores(N_VOCAB, 0.0f);
        for (uint32_t i = 0; i < N_VOCAB; i++) { toks[i] = "tok_" + std::to_string(i); tp[i] = toks[i].c_str(); }
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", tp.data(), N_VOCAB);
        gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), N_VOCAB);
    }

    ggml_init_params ip = { 64u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float W = 0.25f, NRM = 0.05f;
    const int64_t hc_dim = N_HC * N_EMBD, hc_mix = (2 + N_HC) * N_HC;
    add_tensor(g, ctx, gen, "token_embd.weight", { N_EMBD, N_VOCAB }, 1.0f);
    add_tensor(g, ctx, gen, "output_norm.weight", { N_EMBD }, NRM, 1.0f);
    add_tensor(g, ctx, gen, "output.weight", { N_EMBD, N_VOCAB }, W);
    for (uint32_t i = 0; i < N_LAYER; i++) {
        auto nm = [&](const char * s) { return "blk." + std::to_string(i) + "." + s; };
        add_tensor(g, ctx, gen, nm("attn_norm.weight"), { N_EMBD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_sinks.weight"), { N_HEAD }, 0.5f);
        add_tensor(g, ctx, gen, nm("attn_q_a.weight"), { N_EMBD, N_QLORA }, W);
        add_tensor(g, ctx, gen, nm("attn_q_a_norm.weight"), { N_QLORA }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_q_b.weight"), { N_QLORA, N_HEAD * N_HD }, W);
        add_tensor(g, ctx, gen, nm("attn_kv.weight"), { N_EMBD, N_HD }, W);
        add_tensor(g, ctx, gen, nm("attn_kv_a_norm.weight"), { N_HD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_output_a.weight"), { N_HEAD * N_HD / N_OGRP, N_OLORA, N_OGRP }, W);
        add_tensor(g, ctx, gen, nm("attn_output_b.weight"), { N_OGRP * N_OLORA, N_EMBD }, W);
        add_tensor(g, ctx, gen, nm("hc_attn_fn.weight"), { hc_dim, hc_mix }, 0.1f);
        add_tensor(g, ctx, gen, nm("hc_attn_base.weight"), { hc_mix }, 0.5f);
        add_tensor(g, ctx, gen, nm("hc_attn_scale.weight"), { 3 }, 0.5f, 1.0f);
        add_tensor(g, ctx, gen, nm("hc_ffn_fn.weight"), { hc_dim, hc_mix }, 0.1f);
        add_tensor(g, ctx, gen, nm("hc_ffn_base.weight"), { hc_mix }, 0.5f);
        add_tensor(g, ctx, gen, nm("hc_ffn_scale.weight"), { 3 }, 0.5f, 1.0f);
        const bool source = i == 2 || i == 5;
        if (source) {
            add_tensor(g, ctx, gen, nm("attn_compressor_kv.weight"), { N_EMBD, N_HD }, W);
            if (RATIOS[i] != 1) {
                add_tensor(g, ctx, gen, nm("attn_compressor_gate.weight"), { N_EMBD, N_HD }, W);
            }
            add_tensor(g, ctx, gen, nm("attn_compressor_norm.weight"), { N_HD }, NRM, 1.0f);
            add_tensor(g, ctx, gen, nm("indexer.attn_k.weight"), { N_HD, N_IDXD }, W);
            add_tensor(g, ctx, gen, nm("indexer.k_norm.weight"), { N_IDXD }, NRM, 1.0f);
        }
        if (source || i == 7) {
            add_tensor(g, ctx, gen, nm("indexer.attn_q_b.weight"), { N_QLORA, N_IDXH * N_IDXD }, W);
            add_tensor(g, ctx, gen, nm("indexer.proj.weight"), { N_EMBD, N_IDXH }, W);
        }
        add_tensor(g, ctx, gen, nm("ffn_gate_inp.weight"), { N_EMBD, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("exp_probs_b.bias"), { N_EXP }, 0.1f);
        add_tensor(g, ctx, gen, nm("ffn_norm.weight"), { N_EMBD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("ffn_gate_exps.weight"), { N_EMBD, N_FFEXP, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_down_exps.weight"), { N_FFEXP, N_EMBD, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_up_exps.weight"), { N_EMBD, N_FFEXP, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_gate_shexp.weight"), { N_EMBD, N_FFEXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_down_shexp.weight"), { N_FFEXP, N_EMBD }, W);
        add_tensor(g, ctx, gen, nm("ffn_up_shexp.weight"), { N_EMBD, N_FFEXP }, W);
        for (uint32_t e = 0; e < 2; e++) {
            if (ENGRAM_LAYERS[e] == i) {
                add_tensor(g, ctx, gen, nm("engram_embd.weight"), { EG_KEY, (int64_t) rows_needed + 8 }, 1.0f);
                add_tensor(g, ctx, gen, nm("engram_wkv.weight"), { (int64_t) n_cols * EG_KEY, (int64_t) N_EMBD * (N_HC + 1) }, 0.1f);
                add_tensor(g, ctx, gen, nm("engram_q.weight"), { N_EMBD, N_HC }, W);
                add_tensor(g, ctx, gen, nm("engram_k.weight"), { N_EMBD, N_HC }, W);
            }
        }
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    ggml_free(ctx);
    gguf_free(g);
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------

// the environment of a context; nullptr leaves a variable as the test's environment had it
struct cfg_t {
    uint32_t n_ubatch = 64;
    uint32_t n_batch  = 4096;
    uint32_t n_rs     = 0;            // n_rs_seq of the context (rollback planes, ngram-varlen)
    bool layer = false;               // layer_exec
    const char * group      = nullptr; // layer_exec_group_tokens
    const char * moe        = nullptr; // layer_exec_moe_tokens
    const char * expect_masks = "device"; // the layer_exec_masks mode of the layer-major prefills
};

static llama_model * g_model = nullptr;
static uint32_t g_vocab = 0;
static int g_layer_runs = 0;
static std::vector<std::string> g_mask_modes;
static int g_carried_max = 0; // the most carried tensors a layer-major prefill planned (layer_exec_plan)
static int g_groups_max = 0;
static int g_bundles = 0;     // layer-major prefills that ran the MoE in bundles
static int g_split   = 0;     // layer-major prefills whose width rule split the batch into narrower ubatches
static int g_passes_last = 0; // the passes of the last layer-major prefill (layer_exec_passes)
static int g_edges_cut = 0;   // layer-major prefills whose edge slots were lent in pieces of ubatches (layer_exec_lend)

static int g_tail_exact_n = 0;  // dsv41_tail_log lines of an exact tail
static int g_tail_inp_n   = 0;  // ... of a tail turned off for the layer inputs of every row

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "dsv41_tail_log: exact") != nullptr) {
        g_tail_exact_n++;
    }
    if (strstr(text, "dsv41_tail_log: off") != nullptr && strstr(text, "layer inputs extracted") != nullptr) {
        g_tail_inp_n++;
    }
    if (strstr(text, "layer_exec_decode: ") != nullptr && strstr(text, " tokens in ") != nullptr) {
        g_layer_runs++;
        if (const char * g = strstr(text, " group")) {
            const char * p = g;
            while (p > text && p[-1] >= '0' && p[-1] <= '9') { p--; }
            g_groups_max = std::max(g_groups_max, atoi(p));
        }
    }
    if (const char * m = strstr(text, "layer_exec_masks: ")) {
        m += strlen("layer_exec_masks: ");
        g_mask_modes.emplace_back(m, strcspn(m, ","));
    }
    if (const char * le = strstr(text, "layer_exec_lend: ")) {
        int n_pc = 0, n_cut = 0, n_slots = 0;
        size_t n_edge = 0;
        if (sscanf(le, "layer_exec_lend: %d pieces: %zu of the segment edges (%d of %d slot", &n_pc, &n_edge, &n_cut, &n_slots) == 4 &&
                n_cut > 0 && n_edge > (size_t) n_slots) {
            g_edges_cut++;
        }
    }
    if (const char * ps = strstr(text, "layer_exec_passes: ")) {
        if (const char * ti = strstr(ps, " tokens in ")) { g_passes_last = atoi(ti + strlen(" tokens in ")); }
    }
    if (strstr(text, "layer_exec_passes: ") != nullptr && strstr(text, "ubatch 768 of the context's 1024") != nullptr) {
        g_split++;
    }
    if (strstr(text, "layer_exec_plan: ") != nullptr) {
        if (const char * c = strstr(text, " carried)")) {
            const char * p = c;
            while (p > text && p[-1] >= '0' && p[-1] <= '9') { p--; }
            g_carried_max = std::max(g_carried_max, atoi(p));
        }
        if (strstr(text, "MoE bundles of") != nullptr) {
            g_bundles++;
        }
    }
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "layer_exec") != nullptr || strstr(text, "layer-major") != nullptr || strstr(text, "does not cut") != nullptr) {
        fputs(text, stderr);
    }
}

static void set_exec_widths(const char * group, const char * moe) {
    llama_layer_mode_config lm = llama_layer_mode_get_config();
    lm.layer_exec_group_tokens = group ? atoll(group) : 0;
    lm.layer_exec_moe_tokens   = moe ? atoll(moe) : 0;
    llama_layer_mode_set_config(lm);
}

static llama_context_ptr make_ctx(const cfg_t & c) {
    set_layer_exec(c.layer);
    set_exec_widths(c.group, c.moe);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N_CTX;
    cp.n_batch = c.n_batch;
    cp.n_ubatch = c.n_ubatch;
    cp.n_seq_max = 1;
    cp.n_rs_seq = c.n_rs;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = false;
    return llama_context_ptr(llama_init_from_model(g_model, cp));
}

using logits_t = std::map<int, std::vector<float>>;

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int p1,
        const std::vector<int> & out, logits_t * res, int key_offset = 0) {
    common_batch b(ctx);
    std::vector<int> idx_of;
    for (int p = p0; p < p1; p++) {
        const bool o = std::find(out.begin(), out.end(), p) != out.end();
        const int idx = b.add(toks[p], p, 0, o);
        idx_of.push_back(o ? idx : -1);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
        return false;
    }
    llama_synchronize(ctx);
    if (res) {
        for (int p = p0; p < p1; p++) {
            if (idx_of[p - p0] >= 0) {
                const float * l = llama_get_logits_ith(ctx, idx_of[p - p0]);
                if (!l) { return false; }
                (*res)[key_offset + p].assign(l, l + g_vocab);
            }
        }
    }
    return true;
}

struct diff_t { double max_abs = 0; double max_rel = 0; bool top1_same = true; size_t n = 0; };

static diff_t compare(const logits_t & a, const logits_t & b) {
    diff_t d;
    if (a.size() != b.size()) { d.max_abs = INFINITY; d.max_rel = INFINITY; return d; }
    for (auto & kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end()) { d.max_abs = INFINITY; d.max_rel = INFINITY; return d; }
        const auto & x = kv.second; const auto & y = it->second;
        double scale = 0, m = 0;
        for (float v : x) { scale = std::max(scale, (double) std::fabs(v)); }
        for (size_t i = 0; i < x.size(); i++) {
            double e = std::fabs((double) x[i] - y[i]);
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { e = INFINITY; }
            m = std::max(m, e);
        }
        d.max_abs = std::max(d.max_abs, m);
        d.max_rel = std::max(d.max_rel, scale > 0 ? m / scale : m);
        if (std::max_element(x.begin(), x.end()) - x.begin() != std::max_element(y.begin(), y.end()) - y.begin()) {
            d.top1_same = false;
        }
        d.n++;
    }
    return d;
}

// the masks of a call of n tokens: device masks for every ubatch, also for one attention chunk (raw chunks of 64 rows, the
// default of this model: 4 * top-k 8, at least 64), since the one-chunk graph of the layer-major prefill is
// built with llm_graph_chunk_mask_one
static std::string expect_masks(const cfg_t & /*c*/, int /*n*/) {
    return "device";
}

// checks that every call of more than one ubatch ran layer-major with the expected masks
static bool check_runs(const cfg_t & c, int runs0, size_t masks0, int want, const std::vector<int> & calls = {}) {
    if (!c.layer) { want = 0; }
    if (g_layer_runs - runs0 != want) {
        fprintf(stderr, "the layer-major prefill ran %d times, expected %d\n", g_layer_runs - runs0, want);
        return false;
    }
    if (g_mask_modes.size() - masks0 != (size_t) want) {
        fprintf(stderr, "%zu layer_exec_masks lines, expected %d\n", g_mask_modes.size() - masks0, want);
        return false;
    }
    for (size_t i = masks0; i < g_mask_modes.size(); i++) {
        const std::string want_m = i - masks0 < calls.size() ? expect_masks(c, calls[i - masks0]) : std::string(c.expect_masks);
        if (g_mask_modes[i] != want_m) {
            fprintf(stderr, "the layer-major prefill used %s masks, expected %s\n", g_mask_modes[i].c_str(), want_m.c_str());
            return false;
        }
    }
    return true;
}

// a prefill of [0, n1) (its last logits), optionally a second call [n1, n2), then n_next single tokens
static bool scenario(const cfg_t & c, const std::vector<llama_token> & toks, int n1, int n2, int n_next, logits_t & res,
                     double * ms = nullptr) {
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    const int runs0 = g_layer_runs;
    const size_t masks0 = g_mask_modes.size();
    const auto t0 = std::chrono::steady_clock::now();
    if (!decode_range(ctx.get(), toks, 0, n1, { n1 - 1 }, &res)) { return false; }
    int p = n1;
    if (n2 > n1) {
        if (!decode_range(ctx.get(), toks, n1, n2, { n2 - 1 }, &res)) { return false; }
        p = n2;
    }
    if (ms) { *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    std::vector<int> calls;
    for (int n : { n1, n2 - n1 }) {
        if (n > (int) c.n_ubatch) { calls.push_back(n); }
    }
    if (!check_runs(c, runs0, masks0, (int) calls.size(), calls)) { return false; }
    set_layer_exec(false);
    for (int i = 0; i < n_next; i++, p++) {
        if (!decode_range(ctx.get(), toks, p, p + 1, { p }, &res)) { return false; }
    }
    return true;
}

// the server's checkpoint flow: a call [0, E) without outputs (layer-major), a partial checkpoint at E, a call [E, N)
// with its last logits and 2 decoded tokens (r1); then the memory back to E, the checkpoint restored and the same again
// (r2, keys offset by 100000)
static bool scenario_ckpt(const cfg_t & c, const std::vector<llama_token> & toks, int E, int N, logits_t & res) {
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    const int runs0 = g_layer_runs;
    const size_t masks0 = g_mask_modes.size();
    if (!decode_range(ctx.get(), toks, 0, E, {}, nullptr)) { return false; }
    if (!check_runs(c, runs0, masks0, E > (int) c.n_ubatch, { E })) { return false; }
    set_layer_exec(false);
    common_prompt_checkpoint ck;
    ck.update_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (ck.data_tgt.empty()) { fprintf(stderr, "empty checkpoint\n"); return false; }
    for (int r = 0; r < 2; r++) {
        const int off = r*100000;
        if (r == 1) {
            llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, E, -1);
            if (!ck.load_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) { fprintf(stderr, "restore failed\n"); return false; }
        }
        if (!decode_range(ctx.get(), toks, E, N, { N - 1 }, &res, off)) { return false; }
        for (int p = N; p < N + 2; p++) {
            if (!decode_range(ctx.get(), toks, p, p + 1, { p }, &res, off)) { return false; }
        }
    }
    return true;
}

// the cut of a model-free graph of two layers, each with a compressor-like state of a base plane and two rollback
// planes, written as DeepSeek-V4 does: a restore (a plane copied onto the base, no layer output read, earliest segment
// 0), then a read of [base | rows of the ubatch], a snapshot into a plane and a persist into the base (earliest segment
// l). variant 0: every access to the state of layer l runs in segment l, in graph order. variant 1: the router input of
// layer 0 also reads the state of layer 1, which would then be read in segment 0 and written in segment 1 (across the
// MoE of layer 0): the cut must refuse it.
static int fixture_cut(ggml_backend_dev_t cpu) {
    int n_fail = 0;
    const int64_t D = 8, T = 4, RW = 4;
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(cpu);
    for (int variant = 0; variant < 2; variant++) {
        ggml_init_params ip = { 64*ggml_tensor_overhead(), nullptr, true };
        ggml_context * cw = ggml_init(ip);
        ggml_context * cs = ggml_init(ip);
        ggml_context * ci = ggml_init(ip);
        ggml_init_params gp = { 1024*ggml_tensor_overhead() + ggml_graph_overhead_custom(1024, false), nullptr, true };
        ggml_context * cg = ggml_init(gp);
        ggml_tensor * W[2];
        ggml_tensor * S[2];
        for (int l = 0; l < 2; l++) {
            W[l] = ggml_new_tensor_2d(cw, GGML_TYPE_F32, D, D);
            S[l] = ggml_new_tensor_2d(cs, GGML_TYPE_F32, D, 3*RW);
            ggml_format_name(S[l], "fx_state-%d", l);
        }
        auto in_i32 = [&](int64_t n, const char * name) {
            ggml_tensor * t = ggml_new_tensor_1d(ci, GGML_TYPE_I32, n);
            ggml_set_name(t, name);
            ggml_set_input(t);
            return t;
        };
        ggml_tensor * inp = ggml_new_tensor_2d(ci, GGML_TYPE_F32, D, T);
        ggml_set_name(inp, "fx_inp");
        ggml_set_input(inp);
        ggml_tensor * ridx  = in_i32(RW, "fx_restore_src");
        ggml_tensor * didx  = in_i32(RW, "fx_restore_dst");
        ggml_tensor * sidx  = in_i32(T,  "fx_read");
        ggml_tensor * snsrc = in_i32(RW, "fx_snapshot_src");
        ggml_tensor * sndst = in_i32(RW, "fx_snapshot_dst");
        ggml_tensor * psrc  = in_i32(T,  "fx_persist_src");
        ggml_tensor * pdst  = in_i32(T,  "fx_persist_dst");
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
        ggml_backend_buffer_t bs = ggml_backend_alloc_ctx_tensors_from_buft(cs, buft);
        ggml_backend_buffer_t bi = ggml_backend_alloc_ctx_tensors_from_buft(ci, buft);
        ggml_backend_buffer_set_usage(bw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_cgraph * gf = ggml_new_graph_custom(cg, 1024, false);
        ggml_tensor * h = ggml_scale(cg, inp, 1.0f);
        ggml_tensor * wr[2][3] = {}; // per layer: restore, snapshot, persist
        ggml_tensor * rd_plane[2] = {};
        for (int l = 0; l < 2; l++) {
            auto all = [&]() { return ggml_view_2d(cg, S[l], D, 3*RW, S[l]->nb[1], 0); };
            ggml_tensor * g = ggml_get_rows(cg, all(), ridx);
            rd_plane[l] = g;
            ggml_tensor * r = ggml_set_rows(cg, all(), g, didx);
            ggml_build_forward_expand(gf, r);
            ggml_tensor * base = ggml_view_2d(cg, r, D, RW, r->nb[1], 0);
            ggml_tensor * src = ggml_concat(cg, base, h, 1);
            ggml_tensor * rd = ggml_get_rows(cg, src, sidx);
            ggml_tensor * sn = ggml_set_rows(cg, all(), ggml_get_rows(cg, src, snsrc), sndst);
            ggml_build_forward_expand(gf, sn);
            ggml_tensor * ps = ggml_set_rows(cg, all(), ggml_get_rows(cg, h, psrc), pdst);
            ggml_build_forward_expand(gf, ps);
            wr[l][0] = r; wr[l][1] = sn; wr[l][2] = ps;
            ggml_tensor * x = ggml_add(cg, h, rd);
            if (variant == 1 && l == 0) {
                ggml_tensor * s1 = ggml_view_2d(cg, S[1], D, RW, S[1]->nb[1], 0);
                x = ggml_add(cg, x, ggml_get_rows(cg, s1, sidx));
            }
            ggml_format_name(x, "fx_moe_in-%d", l);
            ggml_tensor * y = ggml_mul_mat(cg, W[l], x);
            ggml_format_name(y, "fx_moe_out-%d", l);
            h = ggml_scale(cg, y, 0.5f);
        }
        ggml_build_forward_expand(gf, ggml_scale(cg, h, 1.0f));

        const int n = ggml_graph_n_nodes(gf);
        std::vector<int32_t> seg((size_t) n, -1);
        char err[512] = {};
        const bool ok = llama_layer_exec_analyze_segments(gf, 2, "fx_moe_in", "fx_moe_out", T, seg.data(), err, sizeof err);
        auto idx_of = [&](const ggml_tensor * t) {
            for (int i = 0; i < n; i++) {
                if (ggml_graph_node(gf, i) == t) { return i; }
            }
            return -1;
        };
        auto seg_of = [&](const ggml_tensor * t) {
            const int i = idx_of(t);
            return i < 0 ? -3 : seg[(size_t) i];
        };
        bool pass = false;
        std::string what;
        if (variant == 0) {
            pass = ok;
            what = ok ? "" : err;
            for (int l = 0; l < 2 && ok; l++) {
                for (int w = 0; w < 3; w++) {
                    if (seg_of(wr[l][w]) != l) {
                        pass = false;
                        what += " write " + std::to_string(w) + " of layer " + std::to_string(l) + " in segment " +
                            std::to_string(seg_of(wr[l][w])) + ";";
                    }
                }
                if (seg_of(rd_plane[l]) != l) {
                    pass = false;
                    what += " the plane read of layer " + std::to_string(l) + " in segment " +
                        std::to_string(seg_of(rd_plane[l])) + ";";
                }
                // the graph order: restore, snapshot, persist
                if (!(idx_of(wr[l][0]) < idx_of(wr[l][1]) && idx_of(wr[l][1]) < idx_of(wr[l][2]))) {
                    pass = false;
                    what += " order;";
                }
            }
            if (pass) {
                what = "restore, snapshot and persist of layer 1 in segment 1";
            }
        } else {
            pass = !ok;
            what = ok ? "cut, expected a refusal" : err;
        }
        printf("fixture cut %s: %s (%s)\n", variant == 0 ? "co-placed restore/snapshot/persist" : "state read across a MoE",
               pass ? "ok" : "FAIL <<<<<<", what.c_str());
        fflush(stdout);
        n_fail += pass ? 0 : 1;
        ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bs);
        ggml_backend_buffer_free(bi);
        ggml_free(cg);
        ggml_free(ci);
        ggml_free(cs);
        ggml_free(cw);
    }
    return n_fail;
}

// the compressor states (base and rollback planes of every layer) and the pending rollback and depth of sequence 0
struct rs_snap {
    std::vector<uint8_t> st;
    uint32_t idx = 0;
    uint32_t depth = 0;
};

static bool rs_take(llama_context * ctx, std::vector<rs_snap> & v) {
    rs_snap s;
    if (!llama_dsv4_rs_dump(ctx, 0, &s.st, &s.idx, &s.depth)) {
        fprintf(stderr, "rs: no DeepSeek-V4 cache to dump\n");
        return false;
    }
    v.push_back(std::move(s));
    return true;
}

// ngram-varlen on the target: R = n_rs_seq rollback planes. A prompt (llama_set_rs_prompt) of several ubatches, then
//  - a removal of its last two rows, refused (a prompt keeps one plane),
//  - the last prompt row taken back and decoded again as a call of one token (the prompt reuse of the server),
//  - a verification call of m + 1 rows (m drafts, all with logits: the planes of depth m),
//  - a partial removal after k accepted drafts (k = 0: the first position rejected), refused or not,
//  - n3 > 0: a prompt of n3 tokens (layer-major when the case is) that starts with that rollback pending,
//  - and 3 tokens on.
// The result of every seq_rm is recorded in ev, the states after every call and removal in snaps; the layer-major run has
// to give the ubatch loop's logits, results and states. pend: the rollback pending before the n3 prompt.
static bool scenario_rs(const cfg_t & c, const std::vector<llama_token> & toks, int n1, int m, int k, int n3, logits_t & res,
                        std::vector<int> & ev, std::vector<rs_snap> & snaps, uint32_t & pend) {
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    const int runs0 = g_layer_runs;
    const size_t masks0 = g_mask_modes.size();
    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_set_rs_prompt(ctx.get(), true);
    const bool ok0 = decode_range(ctx.get(), toks, 0, n1, { n1 - 1 }, &res);
    llama_set_rs_prompt(ctx.get(), false);
    if (!ok0) { fprintf(stderr, "rs: prompt failed\n"); return false; }
    if (!check_runs(c, runs0, masks0, n1 > (int) c.n_ubatch ? 1 : 0, { n1 })) { return false; }
    set_layer_exec(false);
    if (!rs_take(ctx.get(), snaps)) { return false; }
    // depth 2 after a prompt: refused
    ev.push_back(llama_memory_seq_rm(mem, 0, n1 - 2, -1) ? 1 : 0);
    // the prompt's last row reused
    ev.push_back(llama_memory_seq_rm(mem, 0, n1 - 1, -1) ? 1 : 0);
    if (ev.back()) {
        llama_set_rs_prompt(ctx.get(), true);
        const bool ok = decode_range(ctx.get(), toks, n1 - 1, n1, { n1 - 1 }, &res, 200000);
        llama_set_rs_prompt(ctx.get(), false);
        if (!ok) { fprintf(stderr, "rs: reuse decode failed\n"); return false; }
    }
    if (!rs_take(ctx.get(), snaps)) { return false; }
    // the verification call, m + 1 rows with logits (m = 0: a plain decode of one token, q = 0)
    std::vector<int> outs;
    for (int p = n1; p <= n1 + m; p++) { outs.push_back(p); }
    if (!decode_range(ctx.get(), toks, n1, n1 + m + 1, outs, &res, 300000)) {
        fprintf(stderr, "rs: verification failed (layer=%d)\n", (int) c.layer);
        return false;
    }
    if (!rs_take(ctx.get(), snaps)) { return false; }
    int p = n1 + m + 1;
    if (k < m) {
        p = n1 + k + 1;
        ev.push_back(llama_memory_seq_rm(mem, 0, p, -1) ? 1 : 0);
    } else {
        // everything accepted: the call itself taken back (m = 0: its single row)
        ev.push_back(llama_memory_seq_rm(mem, 0, n1 + m, -1) ? 1 : 0);
        p = n1 + m + (ev.back() ? 0 : 1);
    }
    if (!rs_take(ctx.get(), snaps)) { return false; }
    pend = snaps.back().idx;
    if (n3 > 0) {
        set_layer_exec(c.layer);
        const int runs1 = g_layer_runs;
        const size_t masks1 = g_mask_modes.size();
        llama_set_rs_prompt(ctx.get(), true);
        const bool ok = decode_range(ctx.get(), toks, p, p + n3, { p + n3 - 1 }, &res, 500000);
        llama_set_rs_prompt(ctx.get(), false);
        set_layer_exec(false);
        if (!ok) { fprintf(stderr, "rs: prompt after the rollback failed (layer=%d)\n", (int) c.layer); return false; }
        if (!check_runs(c, runs1, masks1, n3 > (int) c.n_ubatch ? 1 : 0, { n3 })) { return false; }
        p += n3;
        if (!rs_take(ctx.get(), snaps)) { return false; }
    }
    for (int i = 0; i < 3; i++, p++) {
        if (!decode_range(ctx.get(), toks, p, p + 1, { p }, &res, 400000)) {
            fprintf(stderr, "rs: next token failed at %d (layer=%d)\n", p, (int) c.layer);
            return false;
        }
        if (!rs_take(ctx.get(), snaps)) { return false; }
    }
    return true;
}

static bool rs_same(const std::vector<rs_snap> & a, const std::vector<rs_snap> & b, std::string & why) {
    if (a.size() != b.size()) { why = "a different number of states"; return false; }
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].idx != b[i].idx || a[i].depth != b[i].depth) {
            why = "state " + std::to_string(i) + ": pending " + std::to_string(a[i].idx) + "/" + std::to_string(b[i].idx) +
                ", depth " + std::to_string(a[i].depth) + "/" + std::to_string(b[i].depth);
            return false;
        }
        if (a[i].st.empty() || a[i].st.size() != b[i].st.size() || memcmp(a[i].st.data(), b[i].st.data(), a[i].st.size()) != 0) {
            why = "state " + std::to_string(i) + ": the compressor planes differ";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// DSpark feature window (ranma: llama_set_embeddings_layer_inp_last). A tiny DSV4 draft of the DFlash architecture (every
// stage a sliding window of D_SWA over uncompressed KV, the V4.1 flavor without an output head) reads the inputs of the
// target layers 6, 7 and the output of the last layer (8). Its drafter takes the features of the last D_SWA rows of a
// call only; the target's exact tail and the layer-major prefill then stay on. Checked against the drafter that takes
// every row: the target logits, the draft logits of every round (a block decoded after the prompt and after each
// verification) and the draft's newest position.

static const uint32_t D_LAYER = 2, D_SWA = 16, D_BLOCK = 4;
static const int32_t  D_TAPS[3] = { 6, 7, 8 };

static bool write_draft(const std::string & path, uint64_t seed) {
    gen_t gen(seed ^ 0x5eed);
    gguf_context * g = gguf_init_empty();
    const std::string a = "dflash";
    auto key = [&](const char * k) { return a + "." + k; };
    auto u32 = [&](const std::string & k, uint32_t v) { gguf_set_val_u32(g, k.c_str(), v); };
    auto f32 = [&](const std::string & k, float v) { gguf_set_val_f32(g, k.c_str(), v); };
    gguf_set_val_str(g, "general.architecture", a.c_str());
    u32(key("vocab_size"), N_VOCAB);
    u32(key("context_length"), N_CTX);
    u32(key("embedding_length"), N_EMBD);
    u32(key("block_count"), D_LAYER);
    u32(key("feed_forward_length"), 64);
    u32(key("attention.head_count"), N_HEAD);
    u32(key("attention.head_count_kv"), 1);
    u32(key("attention.key_length"), N_HD);
    u32(key("attention.value_length"), N_HD);
    u32(key("rope.dimension_count"), N_HD / 2);
    f32(key("rope.freq_base"), 10000.0f);
    f32(key("attention.layer_norm_rms_epsilon"), 1e-6f);
    u32(key("attention.sliding_window"), D_SWA);
    u32(key("attention.q_lora_rank"), N_QLORA);
    u32(key("expert_feed_forward_length"), N_FFEXP);
    u32(key("expert_shared_count"), 1);
    u32(key("expert_count"), N_EXP);
    u32(key("expert_used_count"), 2);
    u32(key("expert_gating_func"), 4);
    f32(key("expert_weights_scale"), 1.5f);
    gguf_set_val_bool(g, key("expert_weights_norm").c_str(), true);
    f32(key("swiglu_clamp_exp"), 10.0f);
    u32(key("attention.output_group_count"), N_OGRP);
    u32(key("attention.output_lora_rank"), N_OLORA);
    u32(key("hyper_connection.count"), N_HC);
    u32(key("hyper_connection.sinkhorn_iterations"), 2);
    f32(key("hyper_connection.epsilon"), 1e-6f);
    {
        std::vector<uint32_t> ratios(D_LAYER, 0);
        gguf_set_arr_data(g, key("attention.compress_ratios").c_str(), GGUF_TYPE_UINT32, ratios.data(), ratios.size());
    }
    gguf_set_arr_data(g, key("target_layers").c_str(), GGUF_TYPE_INT32, D_TAPS, 3);
    u32(key("block_size"), D_BLOCK);
    gguf_set_val_str(g, key("sample_from_anchor").c_str(), "false");

    gguf_set_val_str(g, "tokenizer.ggml.model", "test");
    {
        std::vector<std::string> toks(N_VOCAB);
        std::vector<const char *> tp(N_VOCAB);
        std::vector<float> scores(N_VOCAB, 0.0f);
        for (uint32_t i = 0; i < N_VOCAB; i++) { toks[i] = "tok_" + std::to_string(i); tp[i] = toks[i].c_str(); }
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", tp.data(), N_VOCAB);
        gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), N_VOCAB);
    }

    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    const float W = 0.25f, NRM = 0.05f;
    const int64_t hc_dim = N_HC * N_EMBD, hc_mix = (2 + N_HC) * N_HC;
    add_tensor(g, ctx, gen, "fc.weight", { 3 * N_EMBD, N_EMBD }, 0.1f);
    add_tensor(g, ctx, gen, "enc.output_norm.weight", { N_EMBD }, NRM, 1.0f);
    add_tensor(g, ctx, gen, "output_norm.weight", { N_EMBD }, NRM, 1.0f);
    for (uint32_t i = 0; i < D_LAYER; i++) {
        auto nm = [&](const char * s) { return "blk." + std::to_string(i) + "." + s; };
        add_tensor(g, ctx, gen, nm("attn_norm.weight"), { N_EMBD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_sinks.weight"), { N_HEAD }, 0.5f);
        add_tensor(g, ctx, gen, nm("attn_q_a.weight"), { N_EMBD, N_QLORA }, W);
        add_tensor(g, ctx, gen, nm("attn_q_a_norm.weight"), { N_QLORA }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_q_b.weight"), { N_QLORA, N_HEAD * N_HD }, W);
        add_tensor(g, ctx, gen, nm("attn_kv.weight"), { N_EMBD, N_HD }, W);
        add_tensor(g, ctx, gen, nm("attn_kv_a_norm.weight"), { N_HD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("attn_output_a.weight"), { N_HEAD * N_HD / N_OGRP, N_OLORA, N_OGRP }, W);
        add_tensor(g, ctx, gen, nm("attn_output_b.weight"), { N_OGRP * N_OLORA, N_EMBD }, W);
        add_tensor(g, ctx, gen, nm("hc_attn_fn.weight"), { hc_dim, hc_mix }, 0.1f);
        add_tensor(g, ctx, gen, nm("hc_attn_base.weight"), { hc_mix }, 0.5f);
        add_tensor(g, ctx, gen, nm("hc_attn_scale.weight"), { 3 }, 0.5f, 1.0f);
        add_tensor(g, ctx, gen, nm("hc_ffn_fn.weight"), { hc_dim, hc_mix }, 0.1f);
        add_tensor(g, ctx, gen, nm("hc_ffn_base.weight"), { hc_mix }, 0.5f);
        add_tensor(g, ctx, gen, nm("hc_ffn_scale.weight"), { 3 }, 0.5f, 1.0f);
        add_tensor(g, ctx, gen, nm("ffn_gate_inp.weight"), { N_EMBD, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("exp_probs_b.bias"), { N_EXP }, 0.1f);
        add_tensor(g, ctx, gen, nm("ffn_norm.weight"), { N_EMBD }, NRM, 1.0f);
        add_tensor(g, ctx, gen, nm("ffn_gate_exps.weight"), { N_EMBD, N_FFEXP, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_down_exps.weight"), { N_FFEXP, N_EMBD, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_up_exps.weight"), { N_EMBD, N_FFEXP, N_EXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_gate_shexp.weight"), { N_EMBD, N_FFEXP }, W);
        add_tensor(g, ctx, gen, nm("ffn_down_shexp.weight"), { N_FFEXP, N_EMBD }, W);
        add_tensor(g, ctx, gen, nm("ffn_up_shexp.weight"), { N_EMBD, N_FFEXP }, W);
    }
    const bool ok = gguf_write_to_file(g, path.c_str(), false);
    ggml_free(ctx);
    gguf_free(g);
    return ok;
}

static llama_model * g_draft = nullptr;

// how a DSpark case runs: the target's layer-major prefill and tail (the drafter reads the window of its features),
// how the prompt is cut into calls, and the draft context's ubatch
struct dcfg_t {
    bool layer  = false;
    int split = 0;               // 0 one call; 1 [0, N-4) without outputs, [N-4, N); 2 [0, N/2), [N/2, N);
                                 // 3 (E) [0, N-W) without layer inputs, then [N-W, N) as a call of its own
    uint32_t ub     = 128;
    uint32_t ub_dft = 64;
};

struct drec_t {
    logits_t tgt;   // the target's logits
    logits_t dft;   // the draft's block logits, round*16 + row
    logits_t feat;  // one-call prompts: the target features of the last D_SWA rows, tap*1000 + row
    llama_pos pos_max = -2; // the draft's newest position at the end
    int layer_runs = 0;
    int tail_exact = 0;
    int tail_layer_inp = 0;
};

// a call of the target over [p0, p1) with logits at the positions in out, then the drafter's process() of that batch
static bool d_call(llama_context * ctx, common_speculative * spec, const std::vector<llama_token> & toks, int p0, int p1,
                   const std::vector<int> & out, logits_t & res, int key_off, bool process = true) {
    common_batch b(ctx);
    std::vector<int> idx_of;
    for (int p = p0; p < p1; p++) {
        const bool o = std::find(out.begin(), out.end(), p) != out.end();
        idx_of.push_back(o ? b.add(toks[p], p, 0, o) : (b.add(toks[p], p, 0, o), -1));
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
        return false;
    }
    llama_synchronize(ctx);
    for (int p = p0; p < p1; p++) {
        if (idx_of[p - p0] >= 0) {
            const float * l = llama_get_logits_ith(ctx, idx_of[p - p0]);
            if (!l) { return false; }
            res[key_off + p].assign(l, l + g_vocab);
        }
    }
    return !process || common_speculative_process(spec, b);
}

// a block of the draft at pos (the anchor and D_BLOCK - 1 masks), its logits under key round*16, then taken back out of
// the draft memory (the drafter decodes it again from the verification)
static bool d_block(llama_context * ctx_dft, llama_token anchor, int pos, int round, logits_t & res) {
    common_batch b(ctx_dft);
    for (uint32_t i = 0; i < D_BLOCK; i++) {
        b.add(i == 0 ? anchor : 1, pos + (int) i, 0, true);
    }
    if (llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, b.get()) != 0) {
        return false;
    }
    llama_synchronize(ctx_dft);
    for (uint32_t i = 0; i < D_BLOCK; i++) {
        const float * l = llama_get_logits_ith(ctx_dft, (int32_t) i);
        if (!l) { return false; }
        res[round*16 + (int) i].assign(l, l + g_vocab);
    }
    return llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, pos, -1);
}

static llama_context_ptr make_draft_ctx(llama_context * ctx_tgt, uint32_t ub) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N_CTX;
    cp.n_batch = ub;
    cp.n_ubatch = ub;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = false;
    cp.ctx_other = ctx_tgt;
    return llama_context_ptr(llama_init_from_model(g_draft, cp));
}

// a prompt of n tokens (cut as dc.split), then three rounds: a draft block, the verification of 3 drafts (4 rows with
// logits) and the acceptance of 3, 1 and 0 of them (the target and the draft taken back past the accepted rows)
static bool scenario_dspark(const dcfg_t & dc, const std::vector<llama_token> & toks, int n, drec_t & rec) {
    cfg_t c; c.n_ubatch = dc.ub; c.n_batch = 4096; c.n_rs = 5; c.layer = dc.layer;
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    auto ctx_dft = make_draft_ctx(ctx.get(), dc.ub_dft);
    if (!ctx_dft) { fprintf(stderr, "dspark: no draft context\n"); return false; }
    common_params_speculative sp;
    sp.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    sp.draft.n_max = 3;
    sp.draft.backend_sampling = false;
    sp.draft.ctx_tgt = ctx.get();
    sp.draft.ctx_dft = ctx_dft.get();
    common_speculative * spec = common_speculative_init(sp, 1);
    if (spec == nullptr) { fprintf(stderr, "dspark: no drafter\n"); return false; }
    struct spec_free { common_speculative * s; ~spec_free() { common_speculative_free(s); } } sf { spec };

    const int runs0 = g_layer_runs;
    const int te0 = g_tail_exact_n, tl0 = g_tail_inp_n;
    bool ok = true;
    const int W = (int) D_SWA;
    switch (dc.split) {
        case 0: ok = d_call(ctx.get(), spec, toks, 0, n, { n - 1 }, rec.tgt, 0); break;
        case 1: ok = d_call(ctx.get(), spec, toks, 0, n - 4, {}, rec.tgt, 0) && d_call(ctx.get(), spec, toks, n - 4, n, { n - 1 }, rec.tgt, 0); break;
        case 2: ok = d_call(ctx.get(), spec, toks, 0, n/2, {}, rec.tgt, 0) && d_call(ctx.get(), spec, toks, n/2, n, { n - 1 }, rec.tgt, 0); break;
        case 3: {
            for (int32_t t : D_TAPS) { llama_set_embeddings_layer_inp(ctx.get(), (uint32_t) t, false); }
            ok = d_call(ctx.get(), spec, toks, 0, n - W, {}, rec.tgt, 0, /*process =*/ false);
            for (int32_t t : D_TAPS) { llama_set_embeddings_layer_inp(ctx.get(), (uint32_t) t, true); }
            ok = ok && d_call(ctx.get(), spec, toks, n - W, n, { n - 1 }, rec.tgt, 0);
        } break;
    }
    if (!ok) { fprintf(stderr, "dspark: prompt failed\n"); return false; }
    if (dc.split == 0) {
        llama_synchronize(ctx.get());
        for (int k = 0; k < 3; k++) {
            for (int i = n - W; i < n; i++) {
                const float * f = llama_get_embeddings_layer_inp_row(ctx.get(), (uint32_t) D_TAPS[k], i);
                if (!f) { fprintf(stderr, "dspark: window row %d of layer %d not made\n", i, D_TAPS[k]); return false; }
                rec.feat[k*1000 + i].assign(f, f + N_EMBD);
            }
        }
    }
    rec.layer_runs     = g_layer_runs - runs0;
    rec.tail_exact     = g_tail_exact_n - te0;
    rec.tail_layer_inp = g_tail_inp_n - tl0;
    set_layer_exec(false);

    llama_memory_t mem = llama_get_memory(ctx.get());
    llama_memory_t mem_dft = llama_get_memory(ctx_dft.get());
    int cur = n;
    const int accept[3] = { 3, 1, 0 };
    for (int r = 0; r < 3; r++) {
        if (!d_block(ctx_dft.get(), toks[cur], cur, r, rec.dft)) { fprintf(stderr, "dspark: block %d failed\n", r); return false; }
        std::vector<int> outs = { cur, cur + 1, cur + 2, cur + 3 };
        if (!d_call(ctx.get(), spec, toks, cur, cur + 4, outs, rec.tgt, 100000*(r + 1))) {
            fprintf(stderr, "dspark: verification %d failed\n", r);
            return false;
        }
        const int keep = cur + accept[r] + 1;
        if (keep < cur + 4) {
            if (!llama_memory_seq_rm(mem, 0, keep, -1) || !llama_memory_seq_rm(mem_dft, 0, keep, -1)) {
                fprintf(stderr, "dspark: rollback %d refused\n", r);
                return false;
            }
        }
        cur = keep;
    }
    if (!d_block(ctx_dft.get(), toks[cur], cur, 3, rec.dft)) { return false; }
    rec.pos_max = llama_memory_seq_pos_max(mem_dft, 0);
    return true;
}

// the falsifier of the window: the features of every row injected by hand, those outside the last W rows of the prompt
// replaced by noise (garble) or not; the draft's first block must not see the difference
static bool scenario_garble(const std::vector<llama_token> & toks, int n, int g_lo, int g_hi, logits_t & res) {
    cfg_t c; c.n_ubatch = 128; c.n_batch = 4096;
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    for (int32_t t : D_TAPS) { llama_set_embeddings_layer_inp(ctx.get(), (uint32_t) t, true); }
    auto ctx_dft = make_draft_ctx(ctx.get(), 64);
    if (!ctx_dft) { return false; }
    llama_set_embeddings_nextn(ctx_dft.get(), true, true);
    llama_set_causal_attn(ctx_dft.get(), false);
    logits_t tl;
    if (!d_call(ctx.get(), nullptr, toks, 0, n, { n - 1 }, tl, 0, false)) { return false; }
    const int n_enc = 3*(int) N_EMBD;
    std::vector<float> feat((size_t) n*n_enc);
    std::mt19937 rng(77);
    std::normal_distribution<float> dis(0.0f, 3.0f);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 3; k++) {
            const float * src = llama_get_embeddings_layer_inp_row(ctx.get(), (uint32_t) D_TAPS[k], i);
            if (!src) { fprintf(stderr, "garble: row %d not extracted\n", i); return false; }
            for (int j = 0; j < (int) N_EMBD; j++) {
                feat[(size_t) i*n_enc + k*N_EMBD + j] = i >= g_lo && i < g_hi ? dis(rng) : src[j];
            }
        }
    }
    common_batch bi(ctx_dft.get());
    for (int i0 = 0; i0 < n; i0 += 64) {
        bi.clear();
        for (int i = i0; i < std::min(n, i0 + 64); i++) {
            const llama_pos pa[4] = { i, i, i, 0 };
            bi.add_embd({ feat.data() + (size_t) i*n_enc, 1, (size_t) n_enc }, pa, 0, false);
        }
        if (llama_process(ctx_dft.get(), LLAMA_PROCESS_TYPE_DECODE, bi.get()) != 0) { return false; }
    }
    return d_block(ctx_dft.get(), toks[n], n, 0, res);
}

// A lease mock: the prefill lend of a joint expert cache (the target's class and the draft's), with the planner of the
// CUDA cache (expert-lend.h) over byte arenas, given to the layer-major prefill of the CPU fixture in place of its own
// host buffers (llama_layer_exec_set_test_lend). Every slot holds a resident with its own bytes or is empty; a lease
// evicts the lowest ranked residents of both classes, moves the kept ones above the new top down, writes zero heads and
// fills the rest of each lent region with a canary; the release checks that the residents, the zero heads and the
// canaries outside the pieces are untouched (a write out of a piece shows), and that it comes once per lease.
struct lmock_t {
    std::vector<ggml_cuda_expert::lend_class> shape;          // kind_bytes, guard, slots per class (no residents)
    std::vector<std::vector<std::vector<uint8_t>>> arena;     // [class][kind]
    std::vector<std::vector<int>> own;                        // [class][slot] resident id, -1 empty
    std::vector<uint64_t> score;                              // [id]
    std::vector<int> id_class;                                // [id]
    bool held = false;
    int acquires = 0, releases = 0, refused_held = 0, release_unheld = 0, fits_calls = 0, evicted = 0, moved = 0;
    bool intact = true;
    std::string why;
    std::vector<ggml_backend_buffer_t> bufs;
    std::vector<std::pair<size_t, size_t>> piece_at[2][3];    // [class][kind] (offset, bytes) of the held pieces
    std::vector<int> limit;

    static uint8_t pat(int id, int k, size_t i) { return (uint8_t) (((size_t) id*97u + (size_t) k*13u + i*5u) ^ 0xA5u); }
    void write(int c, int slot, int id) {
        for (int k = 0; k < 3; k++) {
            const size_t b = shape[(size_t) c].kind_bytes[(size_t) k];
            for (size_t i = 0; i < b; i++) { arena[(size_t) c][(size_t) k][(size_t) slot*b + i] = pat(id, k, i); }
        }
    }
    bool holds(int c, int slot, int id) const {
        for (int k = 0; k < 3; k++) {
            const size_t b = shape[(size_t) c].kind_bytes[(size_t) k];
            for (size_t i = 0; i < b; i++) {
                if (arena[(size_t) c][(size_t) k][(size_t) slot*b + i] != pat(id, k, i)) { return false; }
            }
        }
        return true;
    }
    // what lend_classes_locked gives the planner: the residents (layer = class, expert = id), the empty slots as tier 0
    std::vector<ggml_cuda_expert::lend_class> classes() const {
        std::vector<ggml_cuda_expert::lend_class> out = shape;
        for (size_t c = 0; c < out.size(); c++) {
            for (int s = 0; s < out[c].slots; s++) {
                ggml_cuda_expert::lend_resident r;
                r.slot = s;
                if (own[c][(size_t) s] < 0) {
                    r.layer = -1 - (int) c; r.expert = s; r.tier = 0;
                } else {
                    r.layer = (int) c; r.expert = own[c][(size_t) s]; r.score = score[(size_t) own[c][(size_t) s]];
                }
                out[c].residents.push_back(r);
            }
        }
        return out;
    }
    void init(uint64_t seed, size_t capacity) {
        *this = lmock_t();
        std::mt19937 rng((uint32_t) seed);
        // the target's class holds three quarters in about 160 slots, the draft's a quarter in about 120 (slots small next to
        // the pieces, so a lease keeps some residents above its top and moves them down)
        const size_t B0 = std::max<size_t>(1024, capacity*3/4/(4*160)/256*256), B1 = std::max<size_t>(512, capacity/4/(2*120)/256*256);
        const int s0 = (int) (capacity*3/4/(4*B0)) + 1, s1 = (int) (capacity/4/(2*B1)) + 1;
        shape.resize(2);
        shape[0].kind_bytes = { B0, B0, 2*B0 }; shape[0].guard = { 256, 256, 256 }; shape[0].slots = s0;
        shape[1].kind_bytes = { B1, B1, 2*B1 }; shape[1].guard = { 256, 256, 256 }; shape[1].slots = s1;
        arena.resize(2);
        own.resize(2);
        for (int c = 0; c < 2; c++) {
            for (int k = 0; k < 3; k++) { arena[(size_t) c].push_back(std::vector<uint8_t>((size_t) shape[(size_t) c].slots*shape[(size_t) c].kind_bytes[(size_t) k], 0)); }
            own[(size_t) c].assign((size_t) shape[(size_t) c].slots, -1);
            for (int s = 0; s + 1 < shape[(size_t) c].slots; s++) {   // the top slot of each class empty (a spare)
                const int id = (int) score.size();
                score.push_back(rng() % 1000);
                id_class.push_back(c);
                own[(size_t) c][(size_t) s] = id;
                write(c, s, id);
            }
        }
        limit = { s0, s1 };
    }
    void fail(const std::string & w) { if (intact) { why = w; } intact = false; }
    void check_residents() {
        for (int c = 0; c < 2; c++) {
            for (int s = 0; s < shape[(size_t) c].slots; s++) {
                const int id = own[(size_t) c][(size_t) s];
                if (id >= 0 && !holds(c, s, id)) { fail("resident " + std::to_string(id) + " lost its bytes"); }
            }
        }
    }
    bool fits(const ggml_expert_lend_request * rq) {
        fits_calls++;
        std::vector<size_t> pieces;
        for (int i = 0; i < rq->n_pieces; i++) { pieces.push_back(rq->pieces[i].bytes); }
        return ggml_cuda_expert::plan_l1_lend(classes(), pieces, 256).ok;
    }
    bool acquire(const ggml_expert_lend_request * rq, ggml_expert_lend_grant * out) {
        if (held) { refused_held++; return false; }
        std::vector<size_t> pieces;
        for (int i = 0; i < rq->n_pieces; i++) { pieces.push_back(rq->pieces[i].bytes); }
        const auto cls = classes();
        const ggml_cuda_expert::lend_l1_plan p = ggml_cuda_expert::plan_l1_lend(cls, pieces, 256);
        if (!p.ok) { return false; }
        // the victims leave; the kept residents at or above the new top move into the freed slots below it
        for (const auto & v : p.victims) {
            if (v.layer < 0) { continue; }
            own[(size_t) v.layer][(size_t) v.slot] = -1;
            evicted++;
        }
        for (int c = 0; c < 2; c++) {
            int next = 0;
            for (int s = p.limit[(size_t) c]; s < shape[(size_t) c].slots; s++) {
                const int id = own[(size_t) c][(size_t) s];
                if (id < 0) { continue; }
                while (next < p.limit[(size_t) c] && own[(size_t) c][(size_t) next] >= 0) { next++; }
                if (next >= p.limit[(size_t) c]) { fail("no free slot below the top"); return false; }
                write(c, next, id);
                own[(size_t) c][(size_t) next] = id;
                own[(size_t) c][(size_t) s] = -1;
                moved++;
            }
        }
        check_residents();
        // zero heads, the canary over the lent regions, then the pieces
        for (int c = 0; c < 2; c++) {
            for (int k = 0; k < 3; k++) {
                piece_at[c][k].clear();
                const size_t b = shape[(size_t) c].kind_bytes[(size_t) k];
                auto & a = arena[(size_t) c][(size_t) k];
                const size_t top = (size_t) p.limit[(size_t) c]*b;
                if (p.limit[(size_t) c] >= shape[(size_t) c].slots) { continue; }
                std::fill(a.begin() + (ptrdiff_t) top, a.begin() + (ptrdiff_t) (top + 256), 0);
                std::fill(a.begin() + (ptrdiff_t) (top + 256), a.end(), 0xEE);
            }
        }
        memset(out, 0, sizeof(*out));
        bufs.clear();
        for (int i = 0; i < rq->n_pieces; i++) {
            if (pieces[(size_t) i] == 0) { continue; }
            const auto & f = p.fits[(size_t) i];
            const auto & r = p.regions[(size_t) f.region];
            uint8_t * at = arena[(size_t) r.cls][(size_t) r.kind].data() + r.begin + f.offset;
            out->buffers[i] = ggml_backend_cpu_buffer_from_ptr(at, pieces[(size_t) i]);
            bufs.push_back(out->buffers[i]);
            piece_at[r.cls][r.kind].push_back({ r.begin + f.offset, pieces[(size_t) i] });
        }
        limit = p.limit;
        held = true;
        acquires++;
        return true;
    }
    bool release() {
        if (!held) { release_unheld++; return false; }
        for (ggml_backend_buffer_t b : bufs) { ggml_backend_buffer_free(b); }
        bufs.clear();
        check_residents();
        for (int c = 0; c < 2; c++) {
            for (int k = 0; k < 3; k++) {
                if (limit[(size_t) c] >= shape[(size_t) c].slots) { continue; }
                const size_t b = shape[(size_t) c].kind_bytes[(size_t) k];
                const auto & a = arena[(size_t) c][(size_t) k];
                const size_t top = (size_t) limit[(size_t) c]*b;
                for (size_t i = top; i < top + 256; i++) { if (a[i] != 0) { fail("a zero head was written"); } }
                for (size_t i = top + 256; i < a.size(); i++) {
                    bool in = false;
                    for (const auto & pa : piece_at[c][k]) { in = in || (i >= pa.first && i < pa.first + pa.second); }
                    if (!in && a[i] != 0xEE) { fail("a write outside the pieces"); break; }
                }
            }
        }
        held = false;
        releases++;
        return true;
    }
};

static lmock_t g_lmock;
static bool lm_fits(ggml_backend_t, const ggml_expert_lend_request * rq) { return g_lmock.fits(rq); }
static bool lm_acquire(ggml_backend_t, const ggml_expert_lend_request * rq, ggml_expert_lend_grant * out) { return g_lmock.acquire(rq, out); }
static bool lm_release(ggml_backend_t) { return g_lmock.release(); }
static bool lm_counts(ggml_expert_lend_counts * out) {
    memset(out, 0, sizeof(*out));
    size_t total = 0, largest = 0;
    ggml_cuda_expert::lend_l1_capacity(g_lmock.classes(), { 0, 0 }, total, largest);
    out->lendable_bytes = total;
    return true;
}
static ggml_expert_iface make_lmock_iface() {
    ggml_expert_iface f;
    memset(&f, 0, sizeof(f));
    f.abi_version  = GGML_EXPERT_ABI_VERSION;
    f.lend_fits    = lm_fits;
    f.lend_acquire = lm_acquire;
    f.lend_release = lm_release;
    f.lend_counts  = lm_counts;
    return f;
}

int main(int argc, char ** argv) {
    uint64_t seed = 1234;
    std::string keep;
    bool quick = false;
    bool longs = false;
    int only[3] = { 0, 0, 0 };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-s" && i + 1 < argc) { seed = strtoull(argv[++i], nullptr, 10); }
        else if (a == "--keep" && i + 1 < argc) { keep = argv[++i]; }
        else if (a == "--quick") { quick = true; }
        else if (a == "--long") { longs = true; }
        else if (a == "--only" && i + 3 < argc) { only[0] = atoi(argv[++i]); only[1] = atoi(argv[++i]); only[2] = atoi(argv[++i]); }
    }
    if (longs) { N_CTX = 131072 + 512; }

    ggml_backend_dev_t cpu = nullptr;
    {
        ggml_backend_reg_t r = ggml_backend_load("ggml-cpu.dll");
        if (!r) { r = ggml_backend_load("./ggml-cpu.dll"); }
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_CPU) { cpu = ggml_backend_dev_get(i); }
        }
        if (!cpu) { fprintf(stderr, "no CPU backend device\n"); return 2; }
    }
    llama_backend_init();
    llama_log_set(log_cb, nullptr);
    const int n_fail_fixture = fixture_cut(cpu);
    if (getenv("RANMA_LAYER_EXEC_LOG") == nullptr) { set_env("RANMA_LAYER_EXEC_LOG", "1"); }
    set_env("GGML_CUDA_TOP_K_STABLE_TIES", "1");

    const std::string path = !keep.empty() ? keep : std::string("test-layer-exec-dsv41-model.gguf");
    if (!write_model(path, seed)) { fprintf(stderr, "model write failed\n"); return 2; }

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { cpu, nullptr };
    mp.devices = devs;
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mp));
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    if (keep.empty()) { remove(path.c_str()); }
    g_model = model.get();
    g_vocab = llama_vocab_n_tokens(llama_model_get_vocab(g_model));

    std::mt19937 rng((uint32_t) seed);
    std::vector<llama_token> toks((size_t) N_CTX);
    for (auto & t : toks) { t = 3 + rng() % (g_vocab - 3); }

    int n_fail = n_fail_fixture;
    auto report = [&](const std::string & what, const diff_t & d, bool must_equal) {
        const bool bad = must_equal ? !(d.max_abs == 0.0) || !d.top1_same : !d.top1_same;
        if (bad && must_equal) { n_fail++; }
        printf("%-78s n=%-3zu max_abs=%.3e max_rel=%.3e top1=%s %s\n", what.c_str(), d.n, d.max_abs, d.max_rel,
               d.top1_same ? "same" : "DIFF", bad ? (must_equal ? "<<<<<< MISMATCH" : "(top-1 differs)") : "");
        fflush(stdout);
    };

    // the ubatch loop against the layer-major prefill, same ubatch size: bit for bit
    struct case_t {
        int n1, n2; uint32_t ub;
        const char * group = nullptr, * moe = nullptr, * masks = "device";
        bool ckpt = false;
    };
    std::vector<case_t> cases = {
        { 1000, 0, 128 }, { 517, 0, 96 }, { 300, 517, 128 }, { 130, 700, 64 },
        // groups of ubatches, MoE bundles (aligned and not), both
        { 1000, 0, 128, "256" }, { 1000, 0, 128, "384" },
        { 1000, 0, 128, nullptr, "256" }, { 1000, 0, 128, nullptr, "100" },
        { 1000, 0, 128, nullptr, "64" },
        { 1000, 0, 128, "512", "200" },
        // the server's checkpoint flow (with groups)
        { 900, 1000, 128, nullptr, nullptr, "device", true },
        { 700, 777, 96, "192", nullptr, "device", true },
    };
    if (quick) { cases = { { 1000, 0, 128 }, { 300, 517, 128 }, { 1000, 0, 128, "256", "200" } }; }
    if (only[0] > 0) { cases = { { only[0], only[1], (uint32_t) only[2] } }; }
    for (const auto & cs : cases) {
        const int n2 = std::max(cs.n2, cs.n1);
        cfg_t a; a.n_ubatch = cs.ub; a.n_batch = std::max<uint32_t>(4096, (uint32_t) n2 + 8);
        a.group = cs.group; a.moe = cs.moe; a.expect_masks = cs.masks;
        cfg_t b = a; b.layer = true;
        logits_t ra, rb;
        double ms_a = 0, ms_b = 0;
        const bool ok = cs.ckpt ? scenario_ckpt(a, toks, cs.n1, n2, ra) && scenario_ckpt(b, toks, cs.n1, n2, rb)
                                : scenario(a, toks, cs.n1, n2, 4, ra, &ms_a) && scenario(b, toks, cs.n1, n2, 4, rb, &ms_b);
        char buf[200];
        snprintf(buf, sizeof buf, "%s n=%d+%d ub=%u group=%s moe=%s (%.0f / %.0f ms)",
                 cs.ckpt ? "ckpt" : "layer-major vs ubatches:", cs.n1, n2 - cs.n1, cs.ub,
                 cs.group ? cs.group : "-", cs.moe ? cs.moe : "-", ms_b, ms_a);
        if (!ok) {
            printf("%s: decode failed\n", buf);
            n_fail++;
            continue;
        }
        report(buf, compare(ra, rb), true);
        if (cs.ckpt) {
            // r2 (restored) against r1 in each mode (info: the comparison above already holds the restored run of both modes
            // bit for bit; the ubatch loop itself continues a restored state at about 1e-3 from the continued one here)
            for (const logits_t * r : { &ra, &rb }) {
                logits_t r1, r2;
                for (auto & kv : *r) { (kv.first >= 100000 ? r2[kv.first - 100000] : r1[kv.first]) = kv.second; }
                report(std::string("  restored vs continued (info), ") + (r == &ra ? "ubatches" : "layer-major"), compare(r1, r2), false);
            }
        }
    }

    // ngram-varlen: rollback planes (R) with a layer-major prompt
    if (only[0] == 0) {
        struct rs_t {
            int n1; uint32_t ub; uint32_t R; int m; int k; const char * group;
            int n3; uint32_t ub_ref; // ub_ref: the ubatch of the ubatch loop (0: ub)
        };
        std::vector<rs_t> rs = {
            { 1000, 128, 32, 8, 8, nullptr, 0, 0 }, { 1000, 128, 32, 8, 4, nullptr, 0, 0 },
            { 1000, 128, 32, 8, 0, nullptr, 0, 0 }, { 1000, 128, 32, 1, 0, nullptr, 0, 0 },
            { 1000, 128, 32, 1, 1, nullptr, 0, 0 }, { 1000, 128, 32, 0, 0, nullptr, 0, 0 },
            { 517, 96, 94, 8, 3, nullptr, 0, 0 }, { 1000, 64, 62, 8, 7, nullptr, 0, 0 },
            { 1000, 64, 8, 8, 2, nullptr, 0, 0 }, { 1000, 128, 32, 8, 4, "256", 0, 0 },
            { 130, 64, 32, 8, 0, nullptr, 0, 0 },
            // the capacity of R = 5: m = 4, m = 5 (all accepted, none, part)
            { 1000, 128, 5, 4, 2, nullptr, 0, 0 }, { 1000, 128, 5, 5, 5, nullptr, 0, 0 },
            { 1000, 128, 5, 5, 0, nullptr, 0, 0 },
            // a prompt of several ubatches with a rollback pending: plain, groups, R = 5, odd sizes
            { 1000, 128, 32, 8, 4, nullptr, 700, 0 }, { 1000, 128, 32, 8, 0, "256", 900, 0 },
            { 1000, 128, 5, 5, 2, nullptr, 600, 0 },
            { 517, 96, 94, 8, 3, nullptr, 517, 0 },
            // ... split again by the width rule (passes of at most 1536 tokens: ubatches of 768, those of the ubatch loop)
            { 700, 1024, 32, 8, 4, "1536", 3000, 768 },
        };
        if (quick) { rs.resize(3); }
        int n_rm_ok = 0, n_rm_no = 0, n_pend = 0;
        for (const rs_t & r : rs) {
            cfg_t a; a.n_ubatch = r.ub_ref > 0 ? r.ub_ref : r.ub; a.n_batch = 4096; a.n_rs = r.R;
            a.group = r.ub_ref > 0 ? nullptr : r.group;
            cfg_t b = a; b.n_ubatch = r.ub; b.group = r.group; b.layer = true;
            logits_t ra, rb;
            std::vector<int> ea, eb;
            std::vector<rs_snap> sa, sb;
            uint32_t pa = 0, pb = 0;
            const int split0 = g_split;
            const bool ok = scenario_rs(a, toks, r.n1, r.m, r.k, r.n3, ra, ea, sa, pa) &&
                            scenario_rs(b, toks, r.n1, r.m, r.k, r.n3, rb, eb, sb, pb);
            char buf[240];
            snprintf(buf, sizeof buf, "rs R=%u n=%d ub=%u m=%d k=%d group=%s n3=%d rm=%d%d%d pend=%u states=%zu", r.R,
                     r.n1, r.ub, r.m, r.k, r.group ? r.group : "-", r.n3, ea.size() > 0 ? ea[0] : -1,
                     ea.size() > 1 ? ea[1] : -1, ea.size() > 2 ? ea[2] : -1, pa, sa.size());
            if (!ok) { printf("%s: decode failed\n", buf); n_fail++; continue; }
            bool bad = false;
            if (ea != eb) { printf("%s: seq_rm results differ between the ubatch loop and layer-major <<<<<< MISMATCH\n", buf); bad = true; }
            if (ea.size() < 3 || ea[0] != 0) { printf("%s: a removal of depth 2 after a prompt was accepted <<<<<< MISMATCH\n", buf); bad = true; }
            if (r.m == 0 && ea.size() >= 3 && ea[2] != 0) { printf("%s: a removal after a plain decode was accepted <<<<<< MISMATCH\n", buf); bad = true; }
            if (r.n3 > 0 && (pa == 0 || pb != pa)) { printf("%s: no rollback pending before the prompt (%u / %u) <<<<<< MISMATCH\n", buf, pa, pb); bad = true; }
            if (r.ub_ref > 0 && g_split == split0) { printf("%s: the batch was not split again <<<<<< MISMATCH\n", buf); bad = true; }
            std::string why;
            if (!rs_same(sa, sb, why)) { printf("%s: %s <<<<<< MISMATCH\n", buf, why.c_str()); bad = true; }
            n_fail += bad ? 1 : 0;
            for (int e : ea) { (e ? n_rm_ok : n_rm_no)++; }
            n_pend += r.n3 > 0 && pa > 0;
            report(buf, compare(ra, rb), true);
        }
        printf("rs coverage: seq_rm accepted %d, refused %d, prompts with a rollback pending %d\n", n_rm_ok, n_rm_no, n_pend);
        if (n_rm_ok < 1 || (!quick && n_pend < 5)) { printf("rs coverage missing\n"); n_fail++; }
    }

    // the width rule splits the batch again: 3000 tokens in ubatches of 1024, passes of at most 1536 tokens, are two passes
    // of two ubatches of 768; the ubatch loop in ubatches of 768 gives the same logits bit for bit
    if (only[0] == 0) {
        cfg_t a; a.n_ubatch = 768; a.n_batch = 4096; a.n_rs = 32;
        cfg_t b = a; b.n_ubatch = 1024; b.layer = true; b.group = "1536";
        logits_t ra, rb;
        const int split0 = g_split;
        const bool ok = scenario(a, toks, 3000, 3000, 4, ra) && scenario(b, toks, 3000, 3000, 4, rb);
        const char * what = "layer-major split again: n=3000 ub=1024 group=1536 vs ubatches of 768";
        if (!ok || g_split == split0) {
            printf("%s: %s\n", what, ok ? "the batch was not split again" : "decode failed");
            n_fail++;
        } else {
            report(what, compare(ra, rb), true);
        }
    }

    // the cases above must have exercised the V4.1 parts: a carried tensor, groups, bundles
    if (only[0] == 0) {
        printf("coverage: carried tensors %d, groups up to %d, bundled prefills %d, edge slots lent per ubatch %d\n", g_carried_max,
               g_groups_max, g_bundles, g_edges_cut);
        if (g_carried_max < 1 || g_groups_max < 2 || g_bundles < 1 || g_edges_cut < 1) {
            printf("coverage missing\n");
            n_fail++;
        }
    }

    // DSpark: the feature window of the drafter
    if (only[0] == 0) {
        const std::string dpath = keep.empty() ? std::string("test-layer-exec-dsv41-draft.gguf") : keep + ".draft.gguf";
        llama_model_ptr draft;
        if (!write_draft(dpath, seed)) {
            printf("dspark: draft write failed\n");
            n_fail++;
        } else {
            llama_model_params dmp = llama_model_default_params();
            dmp.devices = devs;
            dmp.n_gpu_layers = 0;
            draft.reset(llama_model_load_from_file(dpath.c_str(), dmp));
            if (keep.empty()) { remove(dpath.c_str()); }
            if (!draft || llama_model_dflash_feature_window(draft.get()) != (int32_t) D_SWA) {
                printf("dspark: the draft did not load or has no feature window (%d)\n",
                       draft ? llama_model_dflash_feature_window(draft.get()) : -1);
                n_fail++;
                draft.reset();
            }
        }
        g_draft = draft.get();
        auto drun = [&](const dcfg_t & dc, int n, drec_t & r, const char * name) {
            if (!scenario_dspark(dc, toks, n, r)) {
                printf("dspark %s n=%d: failed <<<<<< MISMATCH\n", name, n);
                n_fail++;
                return false;
            }
            return true;
        };
        // one comparison: target logits and draft logits; exact: bit for bit, else the top-1 of every row and a
        // relative difference of rounding size
        auto dcmp = [&](const std::string & what, const drec_t & a, const drec_t & b, bool exact) {
            const diff_t dt = compare(a.tgt, b.tgt);
            const diff_t dd = compare(a.dft, b.dft);
            const bool ok_t = exact ? dt.max_abs == 0.0 && dt.top1_same : dt.top1_same && dt.max_rel < 1e-2;
            const bool ok_d = exact ? dd.max_abs == 0.0 && dd.top1_same : dd.top1_same && dd.max_rel < 1e-2;
            const bool ok_p = a.pos_max == b.pos_max;
            printf("%-64s target n=%zu max_rel=%.3e top1=%s | draft n=%zu max_abs=%.3e max_rel=%.3e top1=%s | pos_max %d/%d %s\n",
                   what.c_str(), dt.n, dt.max_rel, dt.top1_same ? "same" : "DIFF", dd.n, dd.max_abs, dd.max_rel,
                   dd.top1_same ? "same" : "DIFF", (int) a.pos_max, (int) b.pos_max,
                   ok_t && ok_d && ok_p ? (exact ? "(bit equal)" : "(rounding)") : "<<<<<< MISMATCH");
            fflush(stdout);
            n_fail += ok_t && ok_d && ok_p ? 0 : 1;
        };
        auto expect = [&](const std::string & what, bool cond) {
            if (!cond) {
                printf("dspark %s <<<<<< MISMATCH\n", what.c_str());
                n_fail++;
            }
        };
        for (int n : { 900, 1000 }) {
            if (!g_draft) { break; }
            const int n_split_max = quick ? 1 : 3;
            dcfg_t w1;
            dcfg_t w2;   w2.layer = true;
            drec_t r1, r2;
            if (!drun(w1, n, r1, "w1") || !drun(w2, n, r2, "w2")) {
                continue;
            }
            const std::string tag = " n=" + std::to_string(n);
            dcmp("window layer-major vs ubatches" + tag, r2, r1, true);
            expect("the window keeps the exact tail on" + tag, r1.tail_exact > 0 && r1.tail_layer_inp == 0 && r2.tail_exact > 0);
            expect("the window prompt ran layer-major" + tag, r2.layer_runs == 1);
            printf("dspark%s: tail lines w1 %d w2 %d, layer-major runs w2 %d\n", tag.c_str(), r1.tail_exact,
                   r2.tail_exact, r2.layer_runs);
            // the lease mock of a joint cache: an L1 that holds a pass of about 2/3 of the prompt, so the width rule
            // narrows it to two passes; the logits stay those of the ubatch loop bit for bit
            if (n == 1000) {
                size_t cap = 0;
                {
                    cfg_t c; c.n_ubatch = w2.ub; c.n_batch = 4096; c.n_rs = 5; c.layer = true;
                    auto cx = make_ctx(c);
                    uint64_t out[2] = { 0, 0 };
                    if (cx && llama_layer_exec_lend_bytes(cx.get(), 448, 0, out, nullptr, nullptr)) { cap = (size_t) out[0]*3/2; }
                    set_layer_exec(false);
                }
                expect("lease mock: the size query of a pass" + tag, cap > 0);
                if (cap > 0) {
                    g_lmock.init(seed, cap);
                    const ggml_expert_iface mi = make_lmock_iface();
                    llama_layer_exec_set_test_lend(&mi);
                    drec_t rm;
                    g_passes_last = 0;
                    const int cut0 = g_edges_cut;
                    const bool okm = drun(w2, n, rm, "w2 lease mock");
                    llama_layer_exec_set_test_lend(nullptr);
                    if (okm) {
                        dcmp("window layer-major, joint lease mock vs ubatches" + tag, rm, r1, true);
                        expect("lease mock: the prompt ran layer-major" + tag, rm.layer_runs == 1);
                        expect("lease mock: the width rule narrowed it to several passes" + tag, g_passes_last > 1);
                        expect("lease mock: one release per lease, none held" + tag, g_lmock.acquires >= 1 &&
                               g_lmock.acquires == g_lmock.releases && !g_lmock.held && g_lmock.refused_held == 0 &&
                               g_lmock.release_unheld == 0);
                        expect("lease mock: residents, zero heads and canaries intact (" + g_lmock.why + ")" + tag, g_lmock.intact);
                        expect("lease mock: residents evicted" + tag, g_lmock.evicted > 0);
                        expect("lease mock: the edge slots lent per ubatch" + tag, g_edges_cut > cut0);
                    }
                    printf("dspark%s: lease mock: %d passes, %d leases, %d releases, %d fits queries, %d evicted, %d moved down, "
                           "slots kept %d/%d and %d/%d, intact %s\n", tag.c_str(), g_passes_last, g_lmock.acquires, g_lmock.releases,
                           g_lmock.fits_calls, g_lmock.evicted, g_lmock.moved, g_lmock.limit[0], g_lmock.shape[0].slots,
                           g_lmock.limit[1], g_lmock.shape[1].slots, g_lmock.intact ? "yes" : g_lmock.why.c_str());
                }
            }
            // prompts of two calls (the checkpoint cut of the server, two long calls) and E: a prefix without layer inputs,
            // then the window as a call of its own. The reference is the ubatch loop with the same cut: another cut moves
            // the ubatch edges, and the random fixture amplifies the rounding of that (a weight change of 1e-7 moves the
            // logits by 1e-3, one of 1e-6 by up to 0.1), so a cut against the one call proves nothing
            for (int sp = 1; sp <= n_split_max; sp++) {
                dcfg_t sw = w2;   sw.split = sp;
                dcfg_t sb = w1;   sb.split = sp;
                drec_t qw, qb;
                if (!drun(sw, n, qw, "split window") || !drun(sb, n, qb, "split ubatches")) {
                    continue;
                }
                const char * name = sp == 1 ? "checkpoint cut" : sp == 2 ? "two calls" : "E (prefix without features)";
                dcmp(std::string("window layer-major, ") + name + " vs ubatches" + tag, qw, qb, true);
                if (sp != 1) {
                    expect(std::string(name) + " ran layer-major" + tag, qw.layer_runs >= 1);
                }
            }
        }
        // the falsifier: the features outside the window replaced by noise
        if (g_draft) {
            // (and the control: noise on the oldest row a query of the first block reaches, at n - W + 1, changes the draft)
            const int nn = 900, w = (int) D_SWA;
            logits_t ga, gb, gc;
            const bool ok = scenario_garble(toks, nn, 0, 0, ga) && scenario_garble(toks, nn, 0, nn - w, gb) &&
                            scenario_garble(toks, nn, nn - w + 1, nn - w + 2, gc);
            if (!ok) {
                printf("dspark garble: failed <<<<<< MISMATCH\n");
                n_fail++;
            } else {
                const diff_t d = compare(ga, gb);
                const diff_t dc = compare(ga, gc);
                printf("dspark: features outside the window replaced by noise: max_abs=%.3e %s; the oldest row it reaches: "
                       "max_abs=%.3e %s\n", d.max_abs, d.max_abs == 0.0 ? "(no effect)" : "<<<<<< MISMATCH", dc.max_abs,
                       dc.max_abs > 0.0 ? "(changes the draft)" : "<<<<<< BLIND");
                n_fail += d.max_abs == 0.0 && dc.max_abs > 0.0 ? 0 : 1;
            }
        }
        g_draft = nullptr;
    }

    // sensitivity: a different first token must change the last logits
    {
        cfg_t b; b.n_ubatch = 128; b.layer = true;
        logits_t r1, r2;
        std::vector<llama_token> alt = toks; alt[0] = 3 + (toks[0] + 40) % (g_vocab - 3);
        scenario(b, toks, 1000, 1000, 0, r1);
        scenario(b, alt, 1000, 1000, 0, r2);
        const diff_t d = compare(r1, r2);
        printf("control (token 0 changed): max_rel=%.3e %s\n", d.max_rel, d.max_rel > 1e-4 ? "sensitive" : "BLIND");
        if (!(d.max_rel > 1e-4)) { n_fail++; }
    }

    // the shapes of the long prompts: 32K in ubatches of 2048 (the tail on, groups of 4), 128K in ubatches of 3584
    if (longs) {
        struct long_t { int n; uint32_t ub; const char * group; };
        for (const long_t & lc : { long_t{ 32000, 2048, nullptr }, long_t{ 32000, 2048, "8192" }, long_t{ 131000, 3584, nullptr } }) {
            cfg_t a; a.n_ubatch = lc.ub; a.n_batch = (uint32_t) lc.n + 8; a.group = lc.group;
            cfg_t b = a; b.layer = true;
            logits_t ra, rb;
            double ms_a = 0, ms_b = 0;
            if (!scenario(a, toks, lc.n, lc.n, 2, ra, &ms_a) || !scenario(b, toks, lc.n, lc.n, 2, rb, &ms_b)) {
                printf("long decode failed (n=%d)\n", lc.n);
                n_fail++;
                continue;
            }
            char buf[160];
            snprintf(buf, sizeof buf, "long: n=%d ub=%u group=%s (%.0f / %.0f ms)", lc.n, lc.ub, lc.group ? lc.group : "-", ms_b, ms_a);
            report(buf, compare(ra, rb), true);
        }
    }

    printf("%s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
