// ranma diagnostic (may be dropped later): model-free op replay of the DeepSeek V4 Flash chunked attention
// (the attention chunk) against the unchunked graph, on the real DS4F shapes (ctx 16384, ubatch 2048, 7309-token
// prompt, n_swa 128, raw cache 2304 cells, CSA/LID ratio 4, HCA ratio 128, top-k 512, head 512 x 64, indexer
// 128 x 64). Every ubatch of the prompt is replayed with random Q/K/weights:
//   1. device chunk masks (raw SWA, LID/CSA prefix, HCA prefix; the op chain of llm_graph_chunk_mask) read back and
//      compared bit for bit with the host mask rule (llama_kv_cache SWA / dsv4_set_kq_mask)
//   2. fused lightning-indexer scores and top-k (512): whole ubatch once against c-row chunks
//   3. final CSA/HCA/raw masks and flash attention: whole ubatch (host masks) against chunks (device masks), plus FA on
//      the same final mask rows (FA only)
//   4. the projection GEMMs of the chunk (real layer-2 EXL3 indexer.attn_q_b / attn_q_b, F32 indexer.proj) on W
//      columns against c columns, and the top-k sets those GEMM outputs give
// usage: test-ds4-chunk-replay <weights dir> [c=512] [seed=1]
// Set GGML_CUDA_TOP_K_STABLE_TIES=1. Exit 0 always unless an op fails; the verdict lines say what differs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t N_PROMPT   = 7309;
static const int64_t UB         = 2048;
static const int64_t N_RAW      = 2304;
static const int     N_SWA      = 128;
static const int64_t DKQ        = 512;
static const int64_t N_HEAD     = 64;
static const int64_t D_IDX      = 128;
static const int64_t N_IDX_HEAD = 64;
static const int64_t TOP_K      = 512;
static const int64_t Q_LORA     = 1024;
static const int64_t N_EMBD     = 4096;

// ---- copies of the graph helpers at f5b0dc6e4 (src/llama-graph-chunk.cpp, src/models/deepseek4.cpp) ----

static ggml_tensor * rows_view(ggml_context * ctx, ggml_tensor * t, int dim, int64_t i0, int64_t n) {
    GGML_ASSERT(i0 >= 0 && n > 0 && i0 + n <= t->ne[dim]);
    if (i0 == 0 && n == t->ne[dim]) {
        return t;
    }
    int64_t ne[GGML_MAX_DIMS] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
    ne[dim] = n;
    return ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], i0*t->nb[dim]);
}

static ggml_tensor * chunk_mask(ggml_context * ctx, ggml_tensor * tab, ggml_tensor * tok_key, ggml_tensor * tok_seq,
                                int64_t i0, int64_t n, ggml_type type, uint32_t n_swa) {
    ggml_tensor * m = ggml_get_rows(ctx, tab, rows_view(ctx, tok_seq, 0, i0, n));
    m = ggml_sub_inplace(ctx, m, rows_view(ctx, tok_key, 1, i0, n));
    if (n_swa > 0) {
        ggml_tensor * w = ggml_step_inplace(ctx, ggml_scale_bias(ctx, m, -1.0f, (float) n_swa));
        m = ggml_mul_inplace(ctx, ggml_step(ctx, m), w);
    } else {
        m = ggml_step_inplace(ctx, m);
    }
    m = ggml_log_inplace(ctx, m);
    if (type != GGML_TYPE_F32) {
        m = ggml_cast(ctx, m, type);
    }
    return m;
}

static ggml_tensor * top_k_mask(ggml_context * ctx, ggml_tensor * kq_mask, ggml_tensor * top_k) {
    ggml_tensor * kq_mask_all = ggml_fill(ctx, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);
    ggml_tensor * top_k_3d = ggml_view_4d(ctx, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx, zeros, 0.0f);
    ggml_tensor * r = ggml_set_rows(ctx, kq_mask_all, zeros, top_k_3d);
    r = ggml_view_4d(ctx, r, r->ne[1], r->ne[2], 1, r->ne[3], r->nb[2], r->nb[3], r->nb[3], 0);
    return ggml_add(ctx, r, kq_mask);
}

// build_attn_mha with flash attention: q [D, n_head, n], k = v [D, 1, n_kv]
static ggml_tensor * fa(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * mask, ggml_tensor * sinks,
                        int64_t n_kv_max) {
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * cur = ggml_flash_attn_ext(ctx, q, k, k, mask, 1.0f/sqrtf((float) DKQ), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(cur, sinks);
    ggml_flash_attn_ext_set_n_kv_max(cur, (int32_t) n_kv_max);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    return ggml_reshape_2d(ctx, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
}

// ---- harness ----

static ggml_backend_t g_backend;

struct inputs {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    inputs() {
        ggml_init_params p = { 64*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(p);
    }
    ggml_tensor * add(ggml_type type, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        return ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    }
    void alloc() { buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend); GGML_ASSERT(buf); }
    ~inputs() { if (buf) ggml_backend_buffer_free(buf); ggml_free(ctx); }
};

struct graph {
    ggml_context * ctx;
    ggml_cgraph  * gf;
    ggml_gallocr_t ga = nullptr;
    graph() {
        ggml_init_params p = { (size_t) 65536*ggml_tensor_overhead() + ggml_graph_overhead_custom(65536, false), nullptr, true };
        ctx = ggml_init(p);
        gf  = ggml_new_graph_custom(ctx, 65536, false);
    }
    ggml_tensor * out(ggml_tensor * t) { ggml_set_output(t); ggml_build_forward_expand(gf, t); return t; }
    void run() {
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        GGML_ASSERT(ggml_gallocr_alloc_graph(ga, gf));
        GGML_ASSERT(ggml_backend_graph_compute(g_backend, gf) == GGML_STATUS_SUCCESS);
    }
    ~graph() { if (ga) ggml_gallocr_free(ga); ggml_free(ctx); }
};

template <typename T> static std::vector<T> get(const ggml_tensor * t) {
    GGML_ASSERT(ggml_is_contiguous(t));
    std::vector<T> v(ggml_nbytes(t)/sizeof(T));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return get<float>(t);
    GGML_ASSERT(t->type == GGML_TYPE_F16);
    std::vector<ggml_fp16_t> h = get<ggml_fp16_t>(t);
    std::vector<float> v(h.size());
    ggml_fp16_to_fp32_row(h.data(), v.data(), (int64_t) h.size());
    return v;
}

static void set_f32(ggml_tensor * t, const std::vector<float> & v) {
    GGML_ASSERT((int64_t) v.size() == ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F16);
        std::vector<ggml_fp16_t> h(v.size());
        ggml_fp32_to_fp16_row(v.data(), h.data(), (int64_t) v.size());
        ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
    }
}

static std::mt19937 g_rng;
static std::vector<float> randn(int64_t n, float s) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto & x : v) x = d(g_rng);
    return v;
}

static std::vector<uint8_t> read_file(const std::string & path, size_t n) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
    std::vector<uint8_t> v(n);
    GGML_ASSERT(fread(v.data(), 1, n, f) == n);
    fclose(f);
    return v;
}

struct diff {
    int64_t n = 0, n_diff = 0, n_nonfinite = 0;
    double max_abs = 0, max_ref = 0, sum_sq = 0;
    void add(float a, float b) { // a reference, b candidate
        ++n;
        const bool fa_ = std::isfinite(a), fb = std::isfinite(b);
        if (!fb && fa_) ++n_nonfinite;
        if (memcmp(&a, &b, sizeof(float)) != 0) ++n_diff;
        if (fa_ && fb) {
            const double d = fabs((double) a - b);
            max_abs = std::max(max_abs, d);
            max_ref = std::max(max_ref, fabs((double) a));
            sum_sq += d*d;
        } else if (fa_ != fb || (a != b)) {
            if (!(std::isinf(a) && std::isinf(b) && a == b)) max_abs = INFINITY;
        }
    }
    void print(const char * name) const {
        printf("  %-34s n %10lld  bit-diff %10lld  max_abs %.3e  max_abs/max|ref| %.3e  rms %.3e  nonfinite %lld\n", name,
               (long long) n, (long long) n_diff, max_abs, max_ref > 0 ? max_abs/max_ref : 0.0, n ? sqrt(sum_sq/n) : 0.0,
               (long long) n_nonfinite);
    }
};

// top-k sets (rows of k indices); returns rows whose sets differ, total missing entries, rows with a different order
struct topk_cmp { int64_t rows = 0, rows_set_diff = 0, swapped = 0, rows_order_diff = 0; };
static void cmp_topk(topk_cmp & r, const int32_t * a, const int32_t * b, int64_t k, int64_t n_rows) {
    std::vector<int32_t> sa(k), sb(k), x;
    for (int64_t i = 0; i < n_rows; ++i) {
        ++r.rows;
        if (memcmp(a + i*k, b + i*k, k*sizeof(int32_t)) != 0) ++r.rows_order_diff;
        sa.assign(a + i*k, a + (i + 1)*k);
        sb.assign(b + i*k, b + (i + 1)*k);
        std::sort(sa.begin(), sa.end());
        std::sort(sb.begin(), sb.end());
        if (sa != sb) {
            ++r.rows_set_diff;
            x.clear();
            std::set_difference(sa.begin(), sa.end(), sb.begin(), sb.end(), std::back_inserter(x));
            r.swapped += (int64_t) x.size();
        }
    }
}

static int64_t pad256(int64_t n) { return std::max<int64_t>((n + 255)/256*256, 256); }

static const uint16_t F16_ZERO = 0x0000, F16_NINF = 0xFC00;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <weights dir> [c=512] [seed=1]\n", argv[0]);
        return 2;
    }
    const std::string wdir = argv[1];
    const int64_t c    = argc > 2 ? atoll(argv[2]) : 512;
    const int     seed = argc > 3 ? atoi(argv[3]) : 1;
    g_rng.seed(seed);

    ggml_backend_load_all();
    g_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    GGML_ASSERT(g_backend);
    printf("backend %s (%s), c %lld, seed %d, STABLE_TIES %s\n", ggml_backend_name(g_backend),
           ggml_backend_dev_description(ggml_backend_get_device(g_backend)), (long long) c, seed,
           getenv("GGML_CUDA_TOP_K_STABLE_TIES") ? getenv("GGML_CUDA_TOP_K_STABLE_TIES") : "unset");

    // real layer-2 projection weights
    inputs wt;
    ggml_tensor * w_iqb  = wt.add(GGML_TYPE_EXL3_M5, Q_LORA, N_IDX_HEAD*D_IDX);
    ggml_tensor * w_iqbi = wt.add(GGML_TYPE_F16, Q_LORA);
    ggml_tensor * w_iqbo = wt.add(GGML_TYPE_F16, N_IDX_HEAD*D_IDX);
    ggml_tensor * w_qb   = wt.add(GGML_TYPE_EXL3_M5, Q_LORA, N_HEAD*DKQ);
    ggml_tensor * w_qbi  = wt.add(GGML_TYPE_F16, Q_LORA);
    ggml_tensor * w_qbo  = wt.add(GGML_TYPE_F16, N_HEAD*DKQ);
    ggml_tensor * w_proj = wt.add(GGML_TYPE_F32, N_EMBD, N_IDX_HEAD);
    wt.alloc();
    const std::pair<ggml_tensor *, const char *> wf[] = {
        { w_iqb, "blk.2.indexer.attn_q_b.weight" }, { w_iqbi, "blk.2.indexer.attn_q_b.rot_in" },
        { w_iqbo, "blk.2.indexer.attn_q_b.rot_out" }, { w_qb, "blk.2.attn_q_b.weight" },
        { w_qbi, "blk.2.attn_q_b.rot_in" }, { w_qbo, "blk.2.attn_q_b.rot_out" }, { w_proj, "blk.2.indexer.proj.weight" } };
    for (const auto & p : wf) {
        auto v = read_file(wdir + "/" + p.second + ".bin", ggml_nbytes(p.first));
        ggml_backend_tensor_set(p.first, v.data(), 0, v.size());
    }

    // totals
    int64_t mask_bad[3] = {0, 0, 0}, mask_n[3] = {0, 0, 0};
    bool vis_seen[6] = {false}; // CSA n_visible 0,1,511,512,513 and HCA n_visible 1
    diff d_score_all, d_fa_all[3], d_fa_iso_all[3], d_gemm_all[3];
    topk_cmp tk_all, tk_gemm_all;
    int64_t fmask_bad[3] = {0, 0, 0};

    const char * kind_name[3] = { "csa", "hca", "raw" };

    for (int64_t p0 = 0; p0 < N_PROMPT; p0 += UB) {
        const int64_t W = std::min(UB, N_PROMPT - p0);
        std::vector<std::pair<int64_t, int64_t>> chunks;
        for (int64_t i0 = 0; i0 < W; i0 += c) chunks.push_back({ i0, std::min(c, W - i0) });

        // host state of the caches after this ubatch's writes
        std::vector<int64_t> cellpos(N_RAW, -1);
        for (int64_t p = std::max<int64_t>(0, p0 + W - N_RAW); p < p0 + W; ++p) cellpos[p % N_RAW] = p;
        std::vector<float> tab_raw(N_RAW), tok_pos(W), csa_key(W), hca_key(W);
        std::vector<int32_t> tok_seq(W, 0), nv_csa(W), nv_hca(W);
        int64_t mx_csa = 0, mx_hca = 0;
        for (int64_t r = 0; r < N_RAW; ++r) tab_raw[r] = cellpos[r] < 0 ? -INFINITY : -(float) cellpos[r];
        for (int64_t i = 0; i < W; ++i) {
            const int64_t pos = p0 + i;
            tok_pos[i] = -((float) pos + 0.5f);
            nv_csa[i] = (int32_t) ((pos + 1)/4);
            nv_hca[i] = (int32_t) ((pos + 1)/128);
            mx_csa = std::max<int64_t>(mx_csa, nv_csa[i]);
            mx_hca = std::max<int64_t>(mx_hca, nv_hca[i]);
            csa_key[i] = -((float) nv_csa[i] - 0.5f);
            hca_key[i] = -((float) nv_hca[i] - 0.5f);
            const int32_t vs[5] = { 0, 1, 511, 512, 513 };
            for (int k = 0; k < 5; ++k) vis_seen[k] |= nv_csa[i] == vs[k];
            vis_seen[5] |= nv_hca[i] == 1;
        }
        const int64_t n_csa = pad256(mx_csa), n_hca = pad256(mx_hca), n_max = std::max(n_csa, n_hca);
        std::vector<float> iota(n_max);
        for (int64_t j = 0; j < n_max; ++j) iota[j] = -(float) j;

        // reference masks (integer rules), F16 bits
        std::vector<uint16_t> ref_raw(N_RAW*W), ref_csa(n_csa*W), ref_hca(n_hca*W);
        for (int64_t i = 0; i < W; ++i) {
            const int64_t pos = p0 + i;
            for (int64_t r = 0; r < N_RAW; ++r) {
                const int64_t kp = cellpos[r];
                ref_raw[i*N_RAW + r] = kp >= 0 && kp <= pos && pos - kp < N_SWA ? F16_ZERO : F16_NINF;
            }
            for (int64_t j = 0; j < n_csa; ++j) ref_csa[i*n_csa + j] = j < nv_csa[i] ? F16_ZERO : F16_NINF;
            for (int64_t j = 0; j < n_hca; ++j) ref_hca[i*n_hca + j] = j < nv_hca[i] ? F16_ZERO : F16_NINF;
        }

        const int64_t K = std::min(TOP_K, n_csa);
        printf("ubatch p0 %lld W %lld chunks %zu: n_raw %lld n_csa(lid) %lld n_hca %lld top_k %lld\n", (long long) p0,
               (long long) W, chunks.size(), (long long) N_RAW, (long long) n_csa, (long long) n_hca, (long long) K);

        // ---- inputs ----
        inputs in;
        ggml_tensor * t_tab   = in.add(GGML_TYPE_F32, N_RAW, 1);
        ggml_tensor * t_tpos  = in.add(GGML_TYPE_F32, 1, W);
        ggml_tensor * t_tseq  = in.add(GGML_TYPE_I32, W);
        ggml_tensor * t_iota  = in.add(GGML_TYPE_F32, n_max, 1);
        ggml_tensor * t_ckey  = in.add(GGML_TYPE_F32, 1, W);
        ggml_tensor * t_hkey  = in.add(GGML_TYPE_F32, 1, W);
        ggml_tensor * h_raw   = in.add(GGML_TYPE_F16, N_RAW, W, 1, 1);
        ggml_tensor * h_csa   = in.add(GGML_TYPE_F16, n_csa, W, 1, 1);
        ggml_tensor * h_hca   = in.add(GGML_TYPE_F16, n_hca, W, 1, 1);
        ggml_tensor * idx_q   = in.add(GGML_TYPE_F32, D_IDX, N_IDX_HEAD, W);
        ggml_tensor * idx_k   = in.add(GGML_TYPE_F16, D_IDX, 1, n_csa, 1);
        ggml_tensor * idx_w   = in.add(GGML_TYPE_F32, N_IDX_HEAD, W, 1, 1);
        ggml_tensor * q       = in.add(GGML_TYPE_F32, DKQ, N_HEAD, W);
        ggml_tensor * kbig    = in.add(GGML_TYPE_F16, DKQ, 1, N_RAW + n_max, 1);
        ggml_tensor * sinks   = in.add(GGML_TYPE_F32, N_HEAD);
        ggml_tensor * qr      = in.add(GGML_TYPE_F32, Q_LORA, W);
        ggml_tensor * xcur    = in.add(GGML_TYPE_F32, N_EMBD, W);
        ggml_tensor * fm[3];
        fm[0] = in.add(GGML_TYPE_F16, N_RAW + n_csa, W, 1, 1);
        fm[1] = in.add(GGML_TYPE_F16, N_RAW + n_hca, W, 1, 1);
        fm[2] = in.add(GGML_TYPE_F16, N_RAW, W, 1, 1);
        in.alloc();

        set_f32(t_tab, tab_raw);
        set_f32(t_tpos, tok_pos);
        ggml_backend_tensor_set(t_tseq, tok_seq.data(), 0, ggml_nbytes(t_tseq));
        set_f32(t_iota, iota);
        set_f32(t_ckey, csa_key);
        set_f32(t_hkey, hca_key);
        ggml_backend_tensor_set(h_raw, ref_raw.data(), 0, ggml_nbytes(h_raw));
        ggml_backend_tensor_set(h_csa, ref_csa.data(), 0, ggml_nbytes(h_csa));
        ggml_backend_tensor_set(h_hca, ref_hca.data(), 0, ggml_nbytes(h_hca));
        set_f32(idx_q, randn(ggml_nelements(idx_q), 1.0f));
        set_f32(idx_k, randn(ggml_nelements(idx_k), 1.0f));
        set_f32(idx_w, randn(ggml_nelements(idx_w), 1.0f/sqrtf((float) (D_IDX*N_IDX_HEAD))));
        set_f32(q, randn(ggml_nelements(q), 1.0f));
        set_f32(kbig, randn(ggml_nelements(kbig), 1.0f));
        set_f32(sinks, randn(ggml_nelements(sinks), 1.0f));
        set_f32(qr, randn(ggml_nelements(qr), 1.0f));
        set_f32(xcur, randn(ggml_nelements(xcur), 1.0f));

        auto kall = [&](ggml_context * ctx, int64_t n_comp) {
            return ggml_view_4d(ctx, kbig, DKQ, 1, N_RAW + n_comp, 1, kbig->nb[1], kbig->nb[2], kbig->nb[3], 0);
        };
        const int64_t n_kv_max_csa = std::min<int64_t>(N_RAW, N_SWA) + K;

        // ---- whole ubatch (the unchunked graph: host masks) ----
        graph G;
        ggml_tensor * sc_f = G.out(ggml_lightning_indexer(G.ctx, idx_q, idx_k, idx_w, h_csa));
        ggml_tensor * tk_f = G.out(ggml_cont(G.ctx, ggml_top_k(G.ctx, sc_f, (int) K)));
        ggml_tensor * fm_f[3], * fa_f[3];
        fm_f[0] = G.out(ggml_concat(G.ctx, h_raw, top_k_mask(G.ctx, h_csa, tk_f), 0));
        fm_f[1] = G.out(ggml_concat(G.ctx, h_raw, h_hca, 0));
        fm_f[2] = h_raw;
        fa_f[0] = G.out(fa(G.ctx, q, kall(G.ctx, n_csa), fm_f[0], sinks, n_kv_max_csa));
        fa_f[1] = G.out(fa(G.ctx, q, kall(G.ctx, n_hca), fm_f[1], sinks, 0));
        fa_f[2] = G.out(fa(G.ctx, q, kall(G.ctx, 0),     fm_f[2], sinks, 0));
        // GEMMs on W columns
        ggml_tensor * g_f[3];
        g_f[0] = G.out(ggml_mul_mat_had(G.ctx, w_iqb, qr, w_iqbi, w_iqbo, nullptr, 128));
        g_f[1] = G.out(ggml_mul_mat_had(G.ctx, w_qb,  qr, w_qbi,  w_qbo,  nullptr, 128));
        g_f[2] = G.out(ggml_scale(G.ctx, ggml_mul_mat(G.ctx, w_proj, xcur), 1.0f/sqrtf((float) (D_IDX*N_IDX_HEAD))));
        ggml_tensor * gsc_f = G.out(ggml_lightning_indexer(G.ctx, ggml_reshape_3d(G.ctx, g_f[0], D_IDX, N_IDX_HEAD, W),
                                                           idx_k, g_f[2], h_csa));
        ggml_tensor * gtk_f = G.out(ggml_cont(G.ctx, ggml_top_k(G.ctx, gsc_f, (int) K)));
        G.run();

        // the whole-ubatch final masks become the FA-only input of the chunks
        for (int k = 0; k < 3; ++k) {
            auto v = get<uint16_t>(fm_f[k]);
            ggml_backend_tensor_set(fm[k], v.data(), 0, ggml_nbytes(fm[k]));
        }

        // ---- chunks (device masks) ----
        graph C;
        struct per_chunk {
            ggml_tensor * m_raw, * m_csa, * m_hca, * sc, * tk, * fmk[3], * fak[3], * fai[3], * g[3], * gsc, * gtk;
        };
        std::vector<per_chunk> pc;
        for (const auto & ch : chunks) {
            const int64_t i0 = ch.first, n = ch.second;
            per_chunk r;
            r.m_raw = C.out(chunk_mask(C.ctx, t_tab, t_tpos, t_tseq, i0, n, GGML_TYPE_F16, N_SWA));
            r.m_csa = C.out(chunk_mask(C.ctx, rows_view(C.ctx, t_iota, 0, 0, n_csa), t_ckey, t_tseq, i0, n, GGML_TYPE_F16, 0));
            r.m_hca = C.out(chunk_mask(C.ctx, rows_view(C.ctx, t_iota, 0, 0, n_hca), t_hkey, t_tseq, i0, n, GGML_TYPE_F16, 0));
            ggml_tensor * mraw4 = ggml_reshape_4d(C.ctx, r.m_raw, N_RAW, n, 1, 1);
            ggml_tensor * qv    = rows_view(C.ctx, q, 2, i0, n);

            r.sc = C.out(ggml_lightning_indexer(C.ctx, rows_view(C.ctx, idx_q, 2, i0, n), idx_k,
                                                rows_view(C.ctx, idx_w, 1, i0, n), r.m_csa));
            r.tk = C.out(ggml_cont(C.ctx, ggml_top_k(C.ctx, r.sc, (int) K)));

            r.fmk[0] = C.out(ggml_concat(C.ctx, mraw4, top_k_mask(C.ctx, ggml_reshape_4d(C.ctx, r.m_csa, n_csa, n, 1, 1), r.tk), 0));
            r.fmk[1] = C.out(ggml_concat(C.ctx, mraw4, ggml_reshape_4d(C.ctx, r.m_hca, n_hca, n, 1, 1), 0));
            r.fmk[2] = mraw4;
            r.fak[0] = C.out(fa(C.ctx, qv, kall(C.ctx, n_csa), r.fmk[0], sinks, n_kv_max_csa));
            r.fak[1] = C.out(fa(C.ctx, qv, kall(C.ctx, n_hca), r.fmk[1], sinks, 0));
            r.fak[2] = C.out(fa(C.ctx, qv, kall(C.ctx, 0),     r.fmk[2], sinks, 0));
            // FA only: the same final mask rows as the whole ubatch
            r.fai[0] = C.out(fa(C.ctx, qv, kall(C.ctx, n_csa), rows_view(C.ctx, fm[0], 1, i0, n), sinks, n_kv_max_csa));
            r.fai[1] = C.out(fa(C.ctx, qv, kall(C.ctx, n_hca), rows_view(C.ctx, fm[1], 1, i0, n), sinks, 0));
            r.fai[2] = C.out(fa(C.ctx, qv, kall(C.ctx, 0),     rows_view(C.ctx, fm[2], 1, i0, n), sinks, 0));

            ggml_tensor * qrv = rows_view(C.ctx, qr, 1, i0, n);
            r.g[0] = C.out(ggml_mul_mat_had(C.ctx, w_iqb, qrv, w_iqbi, w_iqbo, nullptr, 128));
            r.g[1] = C.out(ggml_mul_mat_had(C.ctx, w_qb,  qrv, w_qbi,  w_qbo,  nullptr, 128));
            r.g[2] = C.out(ggml_scale(C.ctx, ggml_mul_mat(C.ctx, w_proj, rows_view(C.ctx, xcur, 1, i0, n)),
                                      1.0f/sqrtf((float) (D_IDX*N_IDX_HEAD))));
            r.gsc = C.out(ggml_lightning_indexer(C.ctx, ggml_reshape_3d(C.ctx, r.g[0], D_IDX, N_IDX_HEAD, n), idx_k, r.g[2],
                                                 r.m_csa));
            r.gtk = C.out(ggml_cont(C.ctx, ggml_top_k(C.ctx, r.gsc, (int) K)));
            pc.push_back(r);
        }
        C.run();

        // ---- compare ----
        const auto v_sc_f  = get_f32(sc_f);
        const auto v_tk_f  = get<int32_t>(tk_f);
        const auto v_gtk_f = get<int32_t>(gtk_f);
        std::vector<float> v_fa_f[3], v_g_f[3];
        std::vector<uint16_t> v_fm_f[2];
        for (int k = 0; k < 3; ++k) { v_fa_f[k] = get_f32(fa_f[k]); v_g_f[k] = get_f32(g_f[k]); }
        for (int k = 0; k < 2; ++k) v_fm_f[k] = get<uint16_t>(fm_f[k]);

        diff d_score, d_fa[3], d_fai[3], d_g[3];
        topk_cmp tk, tkg;
        int64_t ub_mask_bad[3] = {0, 0, 0}, ub_fm_bad[2] = {0, 0};
        int64_t first_bad[3] = {-1, -1, -1};
        for (size_t ic = 0; ic < chunks.size(); ++ic) {
            const int64_t i0 = chunks[ic].first, n = chunks[ic].second;
            const per_chunk & r = pc[ic];

            // 1. masks
            const std::vector<uint16_t> * refs[3] = { &ref_raw, &ref_csa, &ref_hca };
            ggml_tensor * dev[3] = { r.m_raw, r.m_csa, r.m_hca };
            const int64_t width[3] = { N_RAW, n_csa, n_hca };
            for (int k = 0; k < 3; ++k) {
                GGML_ASSERT(dev[k]->ne[0] == width[k] && dev[k]->ne[1] == n);
                const auto v = get<uint16_t>(dev[k]);
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < width[k]; ++j) {
                        mask_n[k]++;
                        if (v[i*width[k] + j] != (*refs[k])[(i0 + i)*width[k] + j]) {
                            if (ub_mask_bad[k]++ == 0) first_bad[k] = (i0 + i)*width[k] + j;
                        }
                    }
                }
            }

            // 2. LID scores and top-k
            const auto v_sc = get_f32(r.sc);
            for (int64_t i = 0; i < n*n_csa; ++i) d_score.add(v_sc_f[i0*n_csa + i], v_sc[i]);
            const auto v_tk = get<int32_t>(r.tk);
            cmp_topk(tk, v_tk_f.data() + i0*K, v_tk.data(), K, n);

            // 3. final masks and FA
            for (int k = 0; k < 2; ++k) {
                const auto v = get<uint16_t>(r.fmk[k]);
                const int64_t w = r.fmk[k]->ne[0];
                for (int64_t i = 0; i < n*w; ++i) ub_fm_bad[k] += v[i] != v_fm_f[k][i0*w + i];
            }
            for (int k = 0; k < 3; ++k) {
                const auto v  = get_f32(r.fak[k]);
                const auto vi = get_f32(r.fai[k]);
                const int64_t w = DKQ*N_HEAD;
                for (int64_t i = 0; i < n*w; ++i) { d_fa[k].add(v_fa_f[k][i0*w + i], v[i]); d_fai[k].add(v_fa_f[k][i0*w + i], vi[i]); }
            }

            // 4. GEMMs and the top-k they give
            for (int k = 0; k < 3; ++k) {
                const auto v = get_f32(r.g[k]);
                const int64_t w = r.g[k]->ne[0];
                for (int64_t i = 0; i < n*w; ++i) d_g[k].add(v_g_f[k][i0*w + i], v[i]);
            }
            const auto v_gtk = get<int32_t>(r.gtk);
            cmp_topk(tkg, v_gtk_f.data() + i0*K, v_gtk.data(), K, n);
        }

        printf("  masks: raw %lld bad (first flat %lld), lid/csa %lld bad (first %lld), hca %lld bad (first %lld)\n",
               (long long) ub_mask_bad[0], (long long) first_bad[0], (long long) ub_mask_bad[1], (long long) first_bad[1],
               (long long) ub_mask_bad[2], (long long) first_bad[2]);
        printf("  final masks: csa_lid_kq_mask %lld bad, hca_kq_mask %lld bad\n", (long long) ub_fm_bad[0], (long long) ub_fm_bad[1]);
        d_score.print("lid score (same inputs)");
        printf("  lid top-k (same inputs): rows %lld, set differs %lld (entries %lld), order differs %lld\n",
               (long long) tk.rows, (long long) tk.rows_set_diff, (long long) tk.swapped, (long long) tk.rows_order_diff);
        for (int k = 0; k < 3; ++k) {
            char nm[64];
            snprintf(nm, sizeof(nm), "fa %s (chunk masks)", kind_name[k]);   d_fa[k].print(nm);
            snprintf(nm, sizeof(nm), "fa %s (same final mask)", kind_name[k]); d_fai[k].print(nm);
        }
        const char * gname[3] = { "gemm indexer.attn_q_b (exl3)", "gemm attn_q_b (exl3)", "gemm indexer.proj (f32)" };
        for (int k = 0; k < 3; ++k) d_g[k].print(gname[k]);
        printf("  lid top-k from chunk GEMMs: rows %lld, set differs %lld (entries %lld), order differs %lld\n",
               (long long) tkg.rows, (long long) tkg.rows_set_diff, (long long) tkg.swapped, (long long) tkg.rows_order_diff);

        for (int k = 0; k < 3; ++k) mask_bad[k] += ub_mask_bad[k];
        for (int k = 0; k < 2; ++k) fmask_bad[k] += ub_fm_bad[k];
        auto merge = [](diff & a, const diff & b) {
            a.n += b.n; a.n_diff += b.n_diff; a.n_nonfinite += b.n_nonfinite; a.sum_sq += b.sum_sq;
            a.max_abs = std::max(a.max_abs, b.max_abs); a.max_ref = std::max(a.max_ref, b.max_ref);
        };
        merge(d_score_all, d_score);
        for (int k = 0; k < 3; ++k) { merge(d_fa_all[k], d_fa[k]); merge(d_fa_iso_all[k], d_fai[k]); merge(d_gemm_all[k], d_g[k]); }
        tk_all.rows += tk.rows; tk_all.rows_set_diff += tk.rows_set_diff; tk_all.swapped += tk.swapped; tk_all.rows_order_diff += tk.rows_order_diff;
        tk_gemm_all.rows += tkg.rows; tk_gemm_all.rows_set_diff += tkg.rows_set_diff; tk_gemm_all.swapped += tkg.swapped;
        tk_gemm_all.rows_order_diff += tkg.rows_order_diff;
    }

    printf("\nSUMMARY c %lld seed %d\n", (long long) c, seed);
    printf("  coverage: csa n_visible 0/1/511/512/513 %d%d%d%d%d, hca n_visible 1 %d\n", vis_seen[0], vis_seen[1],
           vis_seen[2], vis_seen[3], vis_seen[4], vis_seen[5]);
    printf("  device masks vs host rule: raw %lld/%lld, lid/csa %lld/%lld, hca %lld/%lld bad\n", (long long) mask_bad[0],
           (long long) mask_n[0], (long long) mask_bad[1], (long long) mask_n[1], (long long) mask_bad[2], (long long) mask_n[2]);
    printf("  final masks chunk vs whole: csa_lid %lld bad, hca %lld bad\n", (long long) fmask_bad[0], (long long) fmask_bad[1]);
    d_score_all.print("lid score (same inputs)");
    printf("  lid top-k (same inputs): rows %lld, set differs %lld (entries %lld), order differs %lld\n",
           (long long) tk_all.rows, (long long) tk_all.rows_set_diff, (long long) tk_all.swapped, (long long) tk_all.rows_order_diff);
    for (int k = 0; k < 3; ++k) {
        char nm[64];
        snprintf(nm, sizeof(nm), "fa %s (chunk masks)", kind_name[k]);     d_fa_all[k].print(nm);
        snprintf(nm, sizeof(nm), "fa %s (same final mask)", kind_name[k]); d_fa_iso_all[k].print(nm);
    }
    const char * gname[3] = { "gemm indexer.attn_q_b (exl3)", "gemm attn_q_b (exl3)", "gemm indexer.proj (f32)" };
    for (int k = 0; k < 3; ++k) d_gemm_all[k].print(gname[k]);
    printf("  lid top-k from chunk GEMMs: rows %lld, set differs %lld (entries %lld), order differs %lld\n",
           (long long) tk_gemm_all.rows, (long long) tk_gemm_all.rows_set_diff, (long long) tk_gemm_all.swapped,
           (long long) tk_gemm_all.rows_order_diff);

    const bool masks_ok = mask_bad[0] == 0 && mask_bad[1] == 0 && mask_bad[2] == 0 && fmask_bad[0] == 0 && fmask_bad[1] == 0;
    const bool sel_ok   = d_score_all.n_diff == 0 && tk_all.rows_set_diff == 0;
    bool finite = true;
    for (int k = 0; k < 3; ++k) finite &= d_fa_all[k].n_nonfinite == 0 && d_gemm_all[k].n_nonfinite == 0;
    printf("VERDICT masks %s, selection (same inputs) %s, finite %s\n", masks_ok ? "equal" : "DIFFER",
           sel_ok ? "equal" : "DIFFER", finite ? "yes" : "NO");

    ggml_backend_free(g_backend);
    return 0;
}
