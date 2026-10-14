// Layer-major prefill on a tiny random DeepSeek-V4 model (CPU, no real model).
//
// A batch of several ubatches computed layer by layer (every ubatch through a layer, then the MoE of that layer once
// over all of them) must give the logits of the ubatch loop with the same ubatch size bit for bit: the segments are
// the same nodes over the same rows, and only the MoE runs over the whole width. The model has the real layer roles:
// hash-routed layers, sliding-window-only layers, CSA layers (ratio 4, with the indexer) and HCA layers (ratio 128),
// hyper-connections and a shared expert. The prompts cross the window, the compressor groups and the ubatch size at
// odd places; the next tokens decoded after the prefill check the KV and compressor state it left, and a second call
// on the same sequence checks a prefill on a prefix.
//
// The layer-major prefill makes every attention mask on the device, also for a ubatch of one attention chunk (its
// "layer_exec_masks: device" log line is checked).
// The ubatch sizes 64..128 are one raw chunk (the default raw chunk of this model is 64) or two.
//
// Also reported (not required to match): the difference to one ubatch of the whole width with attention chunks of the
// ubatch size (the layer mode reference), and a graph of many attention chunks (more than the allocator's free blocks
// could take while every chunk result lived until a join at the end).
//
// usage: test-layer-exec-dsv4 [-s SEED] [--keep model.gguf] [--quick]

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

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
static const uint32_t N_HD    = 128;  // head size, 64 of it roped
static const uint32_t N_QLORA = 32;
static const uint32_t N_OGRP  = 2;
static const uint32_t N_OLORA = 16;
static const uint32_t N_HC    = 4;
static const uint32_t N_EXP   = 8;
static const uint32_t N_USED  = 2;
static const uint32_t N_FFEXP = 32;
static const uint32_t N_IDXH  = 4;
static const uint32_t N_IDXD  = 64;
static const uint32_t N_TOPK  = 8;
static const uint32_t N_SWA   = 16;
static const uint32_t N_HASH  = 2;
static uint32_t N_CTX = 20480;
static const uint32_t N_LAYER = 6;
static const uint32_t RATIOS[N_LAYER] = { 0, 4, 128, 4, 0, 128 };

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
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
    const std::string a = "deepseek4";
    auto key = [&](const char * k) { return a + "." + k; };
    auto u32 = [&](const std::string & k, uint32_t v) { gguf_set_val_u32(g, k.c_str(), v); };
    auto f32 = [&](const std::string & k, float v) { gguf_set_val_f32(g, k.c_str(), v); };

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
    u32(key("expert_used_count"), N_USED);
    u32(key("expert_gating_func"), 4);
    f32(key("expert_weights_scale"), 1.5f);
    gguf_set_val_bool(g, key("expert_weights_norm").c_str(), true);
    f32(key("swiglu_clamp_exp"), 10.0f);
    u32(key("attention.indexer.head_count"), N_IDXH);
    u32(key("attention.indexer.key_length"), N_IDXD);
    u32(key("attention.indexer.top_k"), N_TOPK);
    u32(key("attention.output_group_count"), N_OGRP);
    u32(key("attention.output_lora_rank"), N_OLORA);
    f32(key("attention.compress_rope_freq_base"), 160000.0f);
    u32(key("hyper_connection.count"), N_HC);
    u32(key("hyper_connection.sinkhorn_iterations"), 2);
    f32(key("hyper_connection.epsilon"), 1e-6f);
    u32(key("hash_layer_count"), N_HASH);
    {
        std::vector<uint32_t> r(RATIOS, RATIOS + N_LAYER);
        gguf_set_arr_data(g, key("attention.compress_ratios").c_str(), GGUF_TYPE_UINT32, r.data(), r.size());
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
    add_tensor(g, ctx, gen, "output_hc_fn.weight", { hc_dim, N_HC }, 0.1f);
    add_tensor(g, ctx, gen, "output_hc_base.weight", { N_HC }, 0.5f);
    add_tensor(g, ctx, gen, "output_hc_scale.weight", { 1 }, 0.5f, 1.0f);
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
        const uint32_t ratio = RATIOS[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;
            add_tensor(g, ctx, gen, nm("attn_compressor_kv.weight"), { N_EMBD, coff * N_HD }, W);
            add_tensor(g, ctx, gen, nm("attn_compressor_gate.weight"), { N_EMBD, coff * N_HD }, W);
            add_tensor(g, ctx, gen, nm("attn_compressor_ape.weight"), { coff * N_HD, ratio }, 0.5f);
            add_tensor(g, ctx, gen, nm("attn_compressor_norm.weight"), { N_HD }, NRM, 1.0f);
            if (ratio == 4) {
                add_tensor(g, ctx, gen, nm("indexer.proj.weight"), { N_EMBD, N_IDXH }, W);
                add_tensor(g, ctx, gen, nm("indexer.attn_q_b.weight"), { N_QLORA, N_IDXH * N_IDXD }, W);
                add_tensor(g, ctx, gen, nm("indexer_compressor_kv.weight"), { N_EMBD, 2 * N_IDXD }, W);
                add_tensor(g, ctx, gen, nm("indexer_compressor_gate.weight"), { N_EMBD, 2 * N_IDXD }, W);
                add_tensor(g, ctx, gen, nm("indexer_compressor_ape.weight"), { 2 * N_IDXD, ratio }, 0.5f);
                add_tensor(g, ctx, gen, nm("indexer_compressor_norm.weight"), { N_IDXD }, NRM, 1.0f);
            }
        }
        add_tensor(g, ctx, gen, nm("ffn_gate_inp.weight"), { N_EMBD, N_EXP }, W);
        if (i < N_HASH) {
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, N_VOCAB);
            ggml_set_name(t, nm("ffn_gate_tid2eid.weight").c_str());
            int32_t * d = (int32_t *) t->data;
            for (uint32_t v = 0; v < N_VOCAB; v++) {
                const uint32_t e0 = gen.rng() % N_EXP;
                d[v*N_USED + 0] = (int32_t) e0;
                d[v*N_USED + 1] = (int32_t) ((e0 + 1 + gen.rng() % (N_EXP - 1)) % N_EXP);
            }
            gguf_add_tensor(g, t);
        } else {
            add_tensor(g, ctx, gen, nm("exp_probs_b.bias"), { N_EXP }, 0.1f);
        }
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

// ---------------------------------------------------------------------------------------------------------------

struct cfg_t {
    uint32_t n_ubatch  = 64;
    uint32_t n_batch   = 2048;
    bool     layer     = false; // layer_exec
    const char * attn_chunk = nullptr; // attn_chunk
};

static llama_model * g_model = nullptr;
static int g_layer_runs = 0; // prefills the layer-major path computed (its log line)
static std::vector<std::string> g_mask_modes; // per layer-major prefill: "device", "host" or "mixed" (layer_exec_masks)

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (strstr(text, "layer_exec_decode: ") != nullptr && strstr(text, " tokens in ") != nullptr) {
        g_layer_runs++;
    }
    if (const char * m = strstr(text, "layer_exec_masks: ")) {
        m += strlen("layer_exec_masks: ");
        g_mask_modes.emplace_back(m, strcspn(m, ","));
    }
    if (level >= GGML_LOG_LEVEL_WARN || strstr(text, "layer_exec_decode") != nullptr || strstr(text, "layer-major") != nullptr ||
            strstr(text, "layer_exec_masks") != nullptr) {
        fputs(text, stderr);
    }
}
static uint32_t g_vocab = 0;

static llama_context_ptr make_ctx(const cfg_t & c) {
    set_layer_exec(c.layer);
    {
        llama_layer_mode_config lm = llama_layer_mode_get_config();
        lm.attn_chunk = c.attn_chunk ? atoll(c.attn_chunk) : 0;
        llama_layer_mode_set_config(lm);
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N_CTX;
    cp.n_batch = c.n_batch;
    cp.n_ubatch = c.n_ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = false;
    return llama_context_ptr(llama_init_from_model(g_model, cp));
}

using logits_t = std::map<int, std::vector<float>>;

// decode tokens[p0, p1) of sequence 0; logits of the positions in out
static bool decode_range(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int p1,
        const std::vector<int> & out, logits_t * res) {
    if (getenv("TEST_LX_TRACE")) { fprintf(stderr, "decode [%d, %d) layer=%s\n", p0, p1, llama_layer_mode_get_config().layer_exec ? "1" : "0"); fflush(stderr); }
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
                (*res)[p].assign(l, l + g_vocab);
            }
        }
    }
    return true;
}

struct diff_t { double max_abs = 0; double max_rel = 0; bool top1_same = true; int worst_pos = -1; size_t n = 0; };

static diff_t compare(const logits_t & a, const logits_t & b) {
    diff_t d;
    for (auto & kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end()) { d.max_abs = INFINITY; d.worst_pos = kv.first; return d; }
        const auto & x = kv.second; const auto & y = it->second;
        double scale = 0, m = 0;
        for (float v : x) { scale = std::max(scale, (double) std::fabs(v)); }
        for (size_t i = 0; i < x.size(); i++) {
            double e = std::fabs((double) x[i] - y[i]);
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { e = INFINITY; }
            m = std::max(m, e);
        }
        if (m > d.max_abs || d.worst_pos < 0) { d.max_abs = std::max(d.max_abs, m); if (m >= d.max_abs) { d.worst_pos = kv.first; } }
        d.max_rel = std::max(d.max_rel, scale > 0 ? m / scale : m);
        if (std::max_element(x.begin(), x.end()) - x.begin() != std::max_element(y.begin(), y.end()) - y.begin()) {
            d.top1_same = false;
        }
        d.n++;
    }
    return d;
}

// one scenario: a prefill of [0, n1), then optionally a second call [n1, n2), then n_next single tokens; logits of
// every position of the calls and of each next token
static bool scenario(const cfg_t & c, const std::vector<llama_token> & toks, int n1, int n2, int n_next, logits_t & res,
                     double * ms = nullptr) {
    auto ctx = make_ctx(c);
    if (!ctx) { return false; }
    std::vector<int> out1, out2;
    for (int p = 0; p < n1; p++) { out1.push_back(p); }
    for (int p = n1; p < n2; p++) { out2.push_back(p); }
    const int runs0 = g_layer_runs;
    const size_t masks0 = g_mask_modes.size();
    const auto t0 = std::chrono::steady_clock::now();
    // the prefill outputs its last position only (exact-tail style), the others are compared through the next tokens
    if (!decode_range(ctx.get(), toks, 0, n1, { n1 - 1 }, &res)) { return false; }
    if (ms) { *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    int p = n1;
    if (n2 > n1) {
        if (!decode_range(ctx.get(), toks, n1, n2, { n2 - 1 }, &res)) { return false; }
        p = n2;
    }
    // the layer-major prefill must have run for every call of several ubatches
    const int want = c.layer ? (n1 > (int) c.n_ubatch) + (n2 - n1 > (int) c.n_ubatch) : 0;
    if (g_layer_runs - runs0 != want) {
        fprintf(stderr, "the layer-major prefill ran %d times, expected %d\n", g_layer_runs - runs0, want);
        return false;
    }
    // and made its masks on the device
    {
        const std::string expect = "device";
        if (g_mask_modes.size() - masks0 != (size_t) want) {
            fprintf(stderr, "%zu layer_exec_masks lines, expected %d\n", g_mask_modes.size() - masks0, want);
            return false;
        }
        for (size_t i = masks0; i < g_mask_modes.size(); i++) {
            if (g_mask_modes[i] != expect) {
                fprintf(stderr, "the layer-major prefill used %s masks, expected %s\n", g_mask_modes[i].c_str(), expect.c_str());
                return false;
            }
        }
    }
    set_layer_exec(false);
    for (int i = 0; i < n_next; i++, p++) {
        if (!decode_range(ctx.get(), toks, p, p + 1, { p }, &res)) { return false; }
    }
    return true;
}

int main(int argc, char ** argv) {
    uint64_t seed = 1234;
    std::string keep;
    bool quick = false;
    int only[3] = { 0, 0, 0 };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-s" && i + 1 < argc) { seed = strtoull(argv[++i], nullptr, 10); }
        else if (a == "--keep" && i + 1 < argc) { keep = argv[++i]; }
        else if (a == "--quick") { quick = true; }
        else if (a == "--ctx" && i + 1 < argc) { N_CTX = (uint32_t) atoi(argv[++i]); }
        else if (a == "--only" && i + 3 < argc) { only[0] = atoi(argv[++i]); only[1] = atoi(argv[++i]); only[2] = atoi(argv[++i]); }
    }

    // CPU backend only: no GPU is touched
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
    if (getenv("RANMA_LAYER_EXEC_LOG") == nullptr) { set_env("RANMA_LAYER_EXEC_LOG", "1"); }
    const std::string path = !keep.empty() ? keep : std::string("test-layer-exec-dsv4-model.gguf");
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
    std::vector<llama_token> toks(std::max<size_t>(17100, (size_t) N_CTX));
    for (auto & t : toks) { t = 3 + rng() % (g_vocab - 3); }

    int n_fail = 0;
    auto report = [&](const char * what, const diff_t & d, bool must_equal) {
        const bool bad = must_equal ? !(d.max_abs == 0.0) || !d.top1_same : !d.top1_same;
        if (bad && must_equal) { n_fail++; }
        printf("%-58s n=%-3zu max_abs=%.3e max_rel=%.3e top1=%s %s\n", what, d.n, d.max_abs, d.max_rel,
               d.top1_same ? "same" : "DIFF", bad ? (must_equal ? "<<<<<< MISMATCH" : "(top-1 differs)") : "");
    };

    // the ubatch loop against the layer-major prefill, same ubatch size: bit for bit
    struct case_t { int n1, n2; uint32_t ub; };
    std::vector<case_t> cases = {
        { 200, 0, 64 }, { 517, 0, 64 }, { 517, 0, 96 }, { 517, 0, 128 }, { 1000, 0, 100 },
        { 300, 517, 64 }, { 130, 700, 128 },
    };
    if (quick) { cases = { { 200, 0, 64 }, { 300, 517, 64 }, { 517, 0, 96 } }; }
    if (only[0] > 0) { cases = { { only[0], only[1], (uint32_t) only[2] } }; quick = true; }
    for (const auto & cs : cases) {
        const int n2 = std::max(cs.n2, cs.n1);
        cfg_t a; a.n_ubatch = cs.ub; a.n_batch = std::max<uint32_t>(2048, (uint32_t) n2 + 8);
        cfg_t b = a; b.layer = true;
        logits_t ra, rb;
        double ms_a = 0, ms_b = 0;
        if (!scenario(a, toks, cs.n1, n2, 4, ra, &ms_a) || !scenario(b, toks, cs.n1, n2, 4, rb, &ms_b)) {
            printf("decode failed (n1=%d n2=%d ub=%u)\n", cs.n1, n2, cs.ub);
            n_fail++;
            continue;
        }
        char buf[128];
        snprintf(buf, sizeof buf, "layer-major vs ubatches: n=%d+%d ub=%u (%.0f / %.0f ms)", cs.n1, n2 - cs.n1, cs.ub,
                 ms_b, ms_a);
        report(buf, compare(ra, rb), true);
    }

    // sensitivity: a different first token must change the last logits (the test is not blind)
    {
        cfg_t b; b.n_ubatch = 64; b.layer = true;
        logits_t r1, r2;
        std::vector<llama_token> alt = toks; alt[0] = 3 + (toks[0] + 40) % (g_vocab - 3);
        scenario(b, toks, 517, 517, 0, r1);
        scenario(b, alt, 517, 517, 0, r2);
        const diff_t d = compare(r1, r2);
        printf("control (token 0 changed): max_rel=%.3e %s\n", d.max_rel, d.max_rel > 1e-4 ? "sensitive" : "BLIND");
        if (!(d.max_rel > 1e-4)) { n_fail++; }
    }

    // the layer mode reference: one ubatch of the whole width with attention chunks of the chunk size (not required
    // to match: the attention plan and the matmul row counts differ)
    for (uint32_t c : { 64u, 128u }) {
        cfg_t a; a.n_ubatch = 2048; a.attn_chunk = c == 64 ? "64" : "128";
        cfg_t b; b.n_ubatch = c; b.layer = true;
        logits_t ra, rb;
        if (!scenario(a, toks, 517, 517, 4, ra) || !scenario(b, toks, 517, 517, 4, rb)) {
            printf("decode failed (layer mode reference, c=%u)\n", c);
            n_fail++;
            continue;
        }
        char buf[128];
        snprintf(buf, sizeof buf, "layer-major ub=%u vs one ubatch with chunks of %u (info)", c, c);
        report(buf, compare(ra, rb), false);
        cfg_t u; u.n_ubatch = c;
        logits_t ru;
        scenario(u, toks, 517, 517, 4, ru);
        snprintf(buf, sizeof buf, "ubatches ub=%u vs one ubatch with chunks of %u (info)", c, c);
        report(buf, compare(ra, ru), false);
    }

    // many attention chunks in one graph: 17000 tokens in chunks of 64 are 266 chunks, as many as the allocator has
    // free blocks per buffer chunk (the CPU buffers of this tiny model did not run out of them with the old join either,
    // so this checks that the joined chunks still compute, not the allocator limit)
    if (!quick) {
        cfg_t a; a.n_ubatch = 20480; a.n_batch = 20480; a.attn_chunk = "64";
        logits_t ra;
        const bool ok = scenario(a, toks, 17000, 17000, 1, ra);
        printf("one ubatch of 17000 tokens in 266 attention chunks: %s\n", ok ? "ok" : "FAILED");
        if (!ok) { n_fail++; }
    }

    printf("%s (%d failures)\n", n_fail == 0 ? "PASS" : "FAIL", n_fail);
    return n_fail == 0 ? 0 : 1;
}
