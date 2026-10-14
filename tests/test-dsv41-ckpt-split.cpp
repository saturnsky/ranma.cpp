// DeepSeek-V4.1 call splitting and partial checkpoint restore on a tiny random model (CPU, no real model).
//
// A prompt processed in one call must give the same logits as the same prompt split into several calls, and a
// partial checkpoint (LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY, what the server keeps) restored into the same or a fresh
// context must continue exactly like the context it was taken from. The tiny model has the real layer roles:
// sliding-window-only layers, a ratio-2 compressed source with an index source and key owner (+ readers), a ratio-1
// source with its own index source, the candidate mask, and engram layers (their history lives in the raw cells'
// ext.tok). Odd and even split points cross the unfinished group of the ratio-2 compressor.
//
// The model is written as a GGUF file with random F32 tensors; the cache is observed through the logits, so a wrong
// K row, mask, ext.tok, carry state or overwritten prefix shows as a logit difference far above the CPU noise floor
// measured by splitting into ubatches.
//
// usage: test-dsv41-ckpt-split [-n N_TOKENS] [-k N_OUT] [-s SEED] [--keep model.gguf] [--ratio2-only]

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 64;
static const uint32_t N_HEAD  = 4;
static const uint32_t N_HD    = 128;  // head size, 16 of it roped
static const uint32_t N_QLORA = 32;
static const uint32_t N_OGRP  = 2;
static const uint32_t N_OLORA = 16;
static const uint32_t N_HC    = 4;
static const uint32_t N_EXP   = 4;
static const uint32_t N_FFEXP = 32;
static const uint32_t N_IDXH  = 4;
static const uint32_t N_IDXD  = 64;
static const uint32_t N_SWA   = 16;
static uint32_t N_CTX = 256;
static const uint32_t N_LAYER = 6;
static const uint32_t RATIOS[N_LAYER] = { 0, 0, 2, 2, 1, 1 };
static const uint32_t ENGRAM_LAYERS[2] = { 1, 3 };
static const uint32_t EG_HEADS = 2, EG_KEY = 16, EG_NGRAM = 3;

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

static bool write_model(const std::string & path, uint64_t seed, bool ratio2_only) {
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
    f32(key("expert_weights_scale"), 1.0f);
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
    {
        std::vector<uint32_t> r(RATIOS, RATIOS + N_LAYER);
        arr_u32(key("attention.compress_ratios"), r);
    }
    if (!ratio2_only) {
        u32(key("attention.indexer.candidate_source_layer"), 4);
        u32(key("attention.indexer.candidate_topk_blocks"), 4);
        u32(key("attention.indexer.candidate_block_size"), 8);
    }

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
        const bool src2 = i == 2;
        const bool src1 = i == 4;
        if (src2 || src1) {
            add_tensor(g, ctx, gen, nm("attn_compressor_kv.weight"), { N_EMBD, N_HD }, W);
            if (src2) {
                add_tensor(g, ctx, gen, nm("attn_compressor_gate.weight"), { N_EMBD, N_HD }, W);
            }
            add_tensor(g, ctx, gen, nm("attn_compressor_norm.weight"), { N_HD }, NRM, 1.0f);
            add_tensor(g, ctx, gen, nm("indexer.attn_q_b.weight"), { N_QLORA, N_IDXH * N_IDXD }, W);
            add_tensor(g, ctx, gen, nm("indexer.proj.weight"), { N_EMBD, N_IDXH }, W);
            add_tensor(g, ctx, gen, nm("indexer.attn_k.weight"), { N_HD, N_IDXD }, W);
            add_tensor(g, ctx, gen, nm("indexer.k_norm.weight"), { N_IDXD }, NRM, 1.0f);
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

struct run_t {
    llama_model * model = nullptr;
    uint32_t n_ubatch = 16;
    uint32_t n_batch  = 512;

    llama_context_ptr make_ctx() const {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = N_CTX;
        cp.n_batch = n_batch;
        cp.n_ubatch = n_ubatch;
        cp.n_seq_max = 1;
        cp.n_threads = 4;
        cp.n_threads_batch = 4;
        cp.op_offload = false;
        return llama_context_ptr(llama_init_from_model(model, cp));
    }
};

// decode tokens[p0, p1) of sequence 0 at their positions; output flag for the positions in `out` (a set of positions).
static bool decode_range(llama_context * ctx, const std::vector<llama_token> & toks, int p0, int p1,
        const std::vector<int> & out, std::map<int, std::vector<float>> * logits_out, uint32_t n_vocab) {
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
    if (logits_out) {
        for (int p = p0; p < p1; p++) {
            if (idx_of[p - p0] >= 0) {
                const float * l = llama_get_logits_ith(ctx, idx_of[p - p0]);
                if (!l) { return false; }
                (*logits_out)[p].assign(l, l + n_vocab);
            }
        }
    }
    return true;
}

struct diff_t { double max_abs = 0; double max_rel = 0; bool top1_same = true; int worst_pos = -1; };

static diff_t compare(const std::map<int, std::vector<float>> & a, const std::map<int, std::vector<float>> & b) {
    diff_t d;
    for (auto & kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end()) { d.max_abs = INFINITY; d.worst_pos = kv.first; return d; }
        const auto & x = kv.second; const auto & y = it->second;
        double scale = 0;
        for (float v : x) { scale = std::max(scale, (double) std::fabs(v)); }
        double m = 0;
        for (size_t i = 0; i < x.size(); i++) {
            double e = std::fabs((double) x[i] - y[i]);
            if (!std::isfinite(x[i]) || !std::isfinite(y[i])) { e = INFINITY; }
            m = std::max(m, e);
        }
        if (m > d.max_abs) { d.max_abs = m; d.worst_pos = kv.first; }
        d.max_rel = std::max(d.max_rel, scale > 0 ? m / scale : m);
        if (std::max_element(x.begin(), x.end()) - x.begin() != std::max_element(y.begin(), y.end()) - y.begin()) {
            d.top1_same = false;
        }
    }
    return d;
}

static std::vector<int> last_k(int n, int k) {
    std::vector<int> v;
    for (int i = std::max(0, n - k); i < n; i++) { v.push_back(i); }
    return v;
}

int main(int argc, char ** argv) {
    int N = 48, K = 4;
    uint64_t seed = 1234;
    std::string keep;
    bool ratio2_only = false;
    uint32_t scan_ub = 0;
    int split_scan = 0;
    uint32_t ref_ub = 16;   // ubatches below 64 tokens share one numerical path (bit-identical); 64 and up and 1 differ at the 1e-3 level
    double tol = 2e-3;   // relative to the largest |logit| of the row
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-n" && i + 1 < argc) { N = atoi(argv[++i]); }
        else if (a == "-k" && i + 1 < argc) { K = atoi(argv[++i]); }
        else if (a == "-s" && i + 1 < argc) { seed = strtoull(argv[++i], nullptr, 10); }
        else if (a == "--keep" && i + 1 < argc) { keep = argv[++i]; }
        else if (a == "--ref-ub" && i + 1 < argc) { ref_ub = (uint32_t) atoi(argv[++i]); }
        else if (a == "--scan-ub" && i + 1 < argc) { scan_ub = (uint32_t) atoi(argv[++i]); }
        else if (a == "--ctx" && i + 1 < argc) { N_CTX = (uint32_t) atoi(argv[++i]); }
        else if (a == "--split-scan" && i + 1 < argc) { split_scan = atoi(argv[++i]); }
        else if (a == "--ratio2-only") { ratio2_only = true; }
        else if (a == "--tol" && i + 1 < argc) { tol = atof(argv[++i]); }
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

    const std::string path = !keep.empty() ? keep : std::string("test-dsv41-ckpt-split-model.gguf");
    if (!write_model(path, seed, ratio2_only)) { fprintf(stderr, "model write failed\n"); return 2; }

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { cpu, nullptr };
    mp.devices = devs;
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), mp));
    if (!model) { fprintf(stderr, "model load failed\n"); return 2; }
    if (keep.empty()) { remove(path.c_str()); }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    run_t R; R.model = model.get();

    std::mt19937 rng(seed);
    std::vector<llama_token> toks(N);
    for (auto & t : toks) { t = 3 + rng() % (n_vocab - 3); }

    if (scan_ub) {
        // per-position logit difference between a one-ubatch run and a run with ubatch scan_ub, all outputs
        std::vector<int> all; for (int i = 0; i < N; i++) { all.push_back(i); }
        std::map<int, std::vector<float>> A, B;
        run_t x = R; x.n_ubatch = 512; x.n_batch = 512; auto c1 = x.make_ctx(); decode_range(c1.get(), toks, 0, N, all, &A, n_vocab);
        run_t y = R; y.n_ubatch = scan_ub; y.n_batch = 512; auto c2 = y.make_ctx(); decode_range(c2.get(), toks, 0, N, all, &B, n_vocab);
        for (int i = 0; i < N; i++) {
            std::map<int, std::vector<float>> a1{{i, A[i]}}, b1{{i, B[i]}};
            const diff_t d = compare(a1, b1);
            printf("pos %3d max_abs=%.3e%s\n", i, d.max_abs, d.max_rel > 2e-3 ? " <<" : "");
        }
        return 0;
    }
    if (split_scan) {
        // per-position difference between one call and a split at p (both in ubatches of ref_ub), all outputs
        std::vector<int> all; for (int i = 0; i < N; i++) { all.push_back(i); }
        std::map<int, std::vector<float>> A, B;
        run_t x = R; x.n_ubatch = ref_ub; x.n_batch = 512; auto c1 = x.make_ctx(); decode_range(c1.get(), toks, 0, N, all, &A, n_vocab);
        auto c2 = x.make_ctx();
        decode_range(c2.get(), toks, 0, split_scan, all, &B, n_vocab);
        decode_range(c2.get(), toks, split_scan, N, all, &B, n_vocab);
        for (int i = 0; i < N; i++) {
            std::map<int, std::vector<float>> a1{{i, A[i]}}, b1{{i, B[i]}};
            const diff_t d = compare(a1, b1);
            if (d.max_abs > 0) { printf("pos %3d max_abs=%.3e max_rel=%.3e%s\n", i, d.max_abs, d.max_rel, d.max_rel > 2e-3 ? " <<" : ""); }
        }
        return 0;
    }
    int n_fail = 0;
    const auto report = [&](const char * what, int p, const diff_t & d, bool must_equal) {
        const bool bad = !(d.max_rel <= tol) || !d.top1_same;
        if (bad && must_equal) { n_fail++; }
        if (bad || p < 0) {
            printf("%-34s p=%-3d max_abs=%.3e max_rel=%.3e top1=%s worst_pos=%d %s\n", what, p, d.max_abs, d.max_rel,
                    d.top1_same ? "same" : "DIFF", d.worst_pos, bad ? "<<<<<< MISMATCH" : "");
        }
    };

    const std::vector<int> out = last_k(N, K);
    // a ubatch of exactly one token takes the decode-shaped numerical path of the CPU backend (about 1e-3 away from the
    // batched path on this model, which a top-k or expert choice can amplify), so a comparison that ends a call with a
    // one-token ubatch is informational only
    const auto one_tok = [&](int a, int b) { return a % 16 == 1 || b % 16 == 1; };

    // reference: one call (ubatch large enough for everything), last K outputs
    std::map<int, std::vector<float>> ref;
    {
        run_t big = R; big.n_ubatch = ref_ub; big.n_batch = 512;
        auto ctx = big.make_ctx();
        if (!ctx) { fprintf(stderr, "ctx failed\n"); return 2; }
        if (!decode_range(ctx.get(), toks, 0, N, out, &ref, n_vocab)) { fprintf(stderr, "ref decode failed\n"); return 2; }
    }
    // sanity: the logits must depend on the position (otherwise the test is blind)
    {
        double spread = 0;
        for (auto & kv : ref) { for (float v : kv.second) { spread = std::max(spread, (double) std::fabs(v)); } }
        printf("reference: N=%d outputs=%zu max|logit|=%.3f\n", N, ref.size(), spread);
        if (!(spread > 1e-3) || !std::isfinite(spread)) { printf("blind or non-finite reference\n"); return 3; }
    }
    // reference with all outputs (no exact-tail skipping): last K must equal the last-K-only reference
    {
        run_t big = R; big.n_ubatch = ref_ub; big.n_batch = 512;
        auto ctx = big.make_ctx();
        std::vector<int> all; for (int i = 0; i < N; i++) { all.push_back(i); }
        std::map<int, std::vector<float>> got;
        decode_range(ctx.get(), toks, 0, N, all, &got, n_vocab);
        std::map<int, std::vector<float>> got_k; for (int p : out) { got_k[p] = got[p]; }
        report("whole all-out vs whole last-k", -1, compare(ref, got_k), true);
    }
    // sensitivity control: a different token far outside the sliding window must change the logits (compressed prefix)
    {
        run_t big = R; big.n_ubatch = 512; big.n_batch = 512;
        auto ctx = big.make_ctx();
        std::vector<llama_token> alt = toks; alt[0] = 3 + (toks[0] + 40) % (n_vocab - 3);
        std::map<int, std::vector<float>> got;
        decode_range(ctx.get(), alt, 0, N, out, &got, n_vocab);
        const diff_t d = compare(ref, got);
        printf("control (token 0 changed): max_rel=%.3e %s\n", d.max_rel, d.max_rel > 1e-4 ? "sensitive" : "BLIND");
        if (!(d.max_rel > 1e-4)) { n_fail++; }
    }
    // noise floor and ubatch splitting inside one call (n_ubatch 16/8/5)
    for (uint32_t ub : { 512u, 64u, 16u, 5u, 1u }) {
        run_t r = R; r.n_ubatch = ub; r.n_batch = 512;
        auto ctx = r.make_ctx();
        std::map<int, std::vector<float>> got;
        decode_range(ctx.get(), toks, 0, N, out, &got, n_vocab);
        char buf[64]; snprintf(buf, sizeof buf, "one call, n_ubatch=%u", ub);
        report(buf, -1, compare(ref, got), ub != 512 && ub != 64 && ub != 1);
    }

    // split at every p: [0,p) without outputs, then [p,N) with the last K outputs (the server's call split)
    for (uint32_t ub : { 512u, 16u }) {
        for (int p = 1; p < N; p++) {
            run_t r = R; r.n_ubatch = ub; r.n_batch = 512;
            auto ctx = r.make_ctx();
            std::map<int, std::vector<float>> got;
            if (!decode_range(ctx.get(), toks, 0, p, {}, nullptr, n_vocab) ||
                !decode_range(ctx.get(), toks, p, N, out, &got, n_vocab)) {
                printf("split decode failed p=%d\n", p); n_fail++; continue;
            }
            char buf[64]; snprintf(buf, sizeof buf, "2 calls ub=%u", ub);
            report(buf, p, compare(ref, got), ub < 64 && !one_tok(p, N - p));
        }
    }
    // three calls like the server: [0,p) , [p,N-K) , [N-K,N)
    for (int p = 1; p < N - K; p++) {
        run_t r = R; r.n_ubatch = 16; r.n_batch = 512;
        auto ctx = r.make_ctx();
        std::map<int, std::vector<float>> got;
        if (!decode_range(ctx.get(), toks, 0, p, {}, nullptr, n_vocab) ||
            !decode_range(ctx.get(), toks, p, N - K, {}, nullptr, n_vocab) ||
            !decode_range(ctx.get(), toks, N - K, N, out, &got, n_vocab)) { n_fail++; continue; }
        report("3 calls ub=16", p, compare(ref, got), !one_tok(p, N - K - p) && K % 16 != 1);
    }

    // partial checkpoint at p (before the call [p,N)): r1 = continue, r2 = restore the same context and redo,
    // r3 = restore into a fresh context
    for (int p = 1; p < N; p++) {
        run_t r = R; r.n_ubatch = 16; r.n_batch = 512;
        auto ctx = r.make_ctx();
        std::map<int, std::vector<float>> r1, r2, r3;
        if (!decode_range(ctx.get(), toks, 0, p, {}, nullptr, n_vocab)) { n_fail++; continue; }
        common_prompt_checkpoint ck;
        ck.update_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (ck.data_tgt.empty()) { printf("empty checkpoint p=%d\n", p); n_fail++; continue; }
        if (!decode_range(ctx.get(), toks, p, N, out, &r1, n_vocab)) { n_fail++; continue; }
        // server flow: the memory is rolled back to the checkpoint state, then the same tokens are processed
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, p, -1);
        if (!ck.load_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) { printf("restore failed p=%d\n", p); n_fail++; continue; }
        if (!decode_range(ctx.get(), toks, p, N, out, &r2, n_vocab)) { printf("r2 decode failed p=%d\n", p); n_fail++; continue; }
        // r3: a second context that holds the prefix itself (a partial state restores only over a memory that holds it)
        auto ctx2 = r.make_ctx();
        if (!decode_range(ctx2.get(), toks, 0, p, {}, nullptr, n_vocab) ||
            !ck.load_tgt(ctx2.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) { printf("r3 restore failed p=%d\n", p); n_fail++; continue; }
        if (!decode_range(ctx2.get(), toks, p, N, out, &r3, n_vocab)) { printf("r3 decode failed p=%d\n", p); n_fail++; continue; }
        // r4: poison the future (other tokens at the same positions), roll back, restore, redo
        std::map<int, std::vector<float>> r4;
        {
            std::vector<llama_token> other = toks;
            for (int q = p; q < N; q++) { other[q] = 3 + (toks[q] * 7 + 11) % (n_vocab - 3); }
            if (!ck.load_tgt(ctx2.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) { printf("r4 restore0 failed p=%d\n", p); n_fail++; continue; }
            if (!decode_range(ctx2.get(), other, p, N, {N - 1}, nullptr, n_vocab)) { printf("r4 poison failed p=%d\n", p); n_fail++; continue; }
            if (!ck.load_tgt(ctx2.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                !decode_range(ctx2.get(), toks, p, N, out, &r4, n_vocab)) { printf("r4 failed p=%d\n", p); n_fail++; continue; }
        }
        report("ckpt: r4(poisoned future) vs r1", p, compare(r1, r4), true);
        const bool m1 = !one_tok(p, N - p);
        report("ckpt: r1 vs whole", p, compare(ref, r1), m1);
        report("ckpt: r2(restore same) vs r1", p, compare(r1, r2), true);
        report("ckpt: r3(restore, 2nd ctx) vs r1", p, compare(r1, r3), true);
    }

    // checkpoint at the last K positions with a call [0,E) in several ubatches, as the long prompt does
    {
        const int E = N - K;
        run_t r = R; r.n_ubatch = 16; r.n_batch = 512;
        auto ctx = r.make_ctx();
        std::map<int, std::vector<float>> r1, r2;
        decode_range(ctx.get(), toks, 0, E, {}, nullptr, n_vocab);
        common_prompt_checkpoint ck;
        ck.update_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        decode_range(ctx.get(), toks, E, N, out, &r1, n_vocab);
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, E, -1);
        ck.load_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        decode_range(ctx.get(), toks, E, N, out, &r2, n_vocab);
        report("ckpt at N-K: r1 vs whole", E, compare(ref, r1), !one_tok(E, N - E) && K % 16 != 1);
        report("ckpt at N-K: r2 vs r1", E, compare(r1, r2), true);
    }

    printf("RESULT: %s (%d failing comparisons, tol=%.1e rel)\n", n_fail ? "FAIL" : "PASS", n_fail, tol);
    llama_backend_free();
    return n_fail ? 1 : 0;
}
