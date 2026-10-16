#include "layer-mode.h"

#include "expert.h"
#include "fit.h"
#include "log.h"

#include "ggml-cpp.h"
#include "gguf.h"

#include "../src/llama-ext.h"

#include "../ggml/src/ggml-cuda/expert-lend.h"
#include "../ggml/src/ggml-cuda/moe-ffn-had-layout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

// Defaults of the automatic sizing; each can be changed through its environment variable (development use). The width
// is not derived from rates of this machine: it is the widest that the measured memory allows. The VRAM kept outside
// the plan is --fit-target of the device (upstream's margin of -fit, the same as --expert-l1-mib auto).
// - preempt: ms a prompt batch of a slot other than 0 may take while slot 0 has priority (the wait of a new slot-0 request).
#define LM_PREEMPT_MS   3000
#define LM_UB_MIN       1024
// the widest ubatch of the layer-major prefill (default): the per-token time of the segment ops (dense products, sparse
// attention, indexer) hardly changes above a few thousand tokens, the fixed cost of a segment graph is paid once per
// ubatch and layer, and the compute buffers grow with the width (DS4F: 0.30 MiB per token, 2.5 GiB at 8192)
#define LM_LX_UB        8192
#define LM_CHUNK_MIN    512
#define LM_ROWS         4096 // MOE_FFN_HAD rows per chunk (MMH_MOE_ROWS)
#define LM_STAGE_SLOTS  64   // MMH_MOE_STAGE_MAX_SLOTS
#define LM_MAX_ITEMS    65535 // MOE_FFN_HAD: GEMM items of one pass (grid y)

static const size_t MiB = 1024*1024;

static bool lm_env_set(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && v[0] != '\0';
}

static void lm_setenv(const char * name, const std::string & value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

// process private bytes now and the peak so far (-1: unknown on this platform)
static void lm_private(int64_t & now, int64_t & peak) {
    now  = -1;
    peak = -1;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc = {};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        now  = (int64_t) pmc.PrivateUsage;
        peak = (int64_t) pmc.PeakPagefileUsage;
    }
#endif
}

namespace {

// what the sizing needs from the GGUF metadata
struct lm_model {
    std::string arch;
    int64_t  n_embd       = 0;
    int64_t  n_ff_exp     = 0;
    int64_t  n_expert     = 0;
    int64_t  n_used       = 0;
    int64_t  n_ctx_train  = 0;
    int      n_moe_layers = 0;    // routed layers of the trunk (an MTP block of its own is not computed by a prompt step)
    uint64_t exps_bytes   = 0;    // the routed expert weights of those layers, three kinds
    uint64_t exps_all     = 0;    // every routed expert weight, MTP blocks included (the expert cache holds them all)
    uint64_t expert_bytes = 0;    // largest staging slot: one expert of the largest layer, three kinds, each padded to 256 B
    uint64_t kind_bytes[3] = { 0, 0, 0 }; // one expert of the largest layer per kind (up, gate, down): an L1 slot of that kind
    bool     exl3         = true; // every routed expert weight is EXL3
    bool     merged       = false; // gate and up merged (the expert-major op does not take it)
};

struct lm_eval {
    bool    ok       = false;
    int64_t l1       = 0; // the L1 budget of this width (--expert-l1-mib, or what auto leaves)
    int64_t dev_need = 0; // bytes the plan needs on the GPU, the measured part and the fixed parts
    int64_t dev_free = 0;
    int64_t host     = 0; // host compute buffer + graph metadata (information, not a limit)
};

} // namespace

static int64_t lm_kv_int(const gguf_context * g, const std::string & key) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        return 0;
    }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(g, id);
        case GGUF_TYPE_UINT64: return (int64_t) gguf_get_val_u64(g, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(g, id);
        default:               return 0;
    }
}

static bool lm_read_model(const std::string & path, lm_model & m, std::string & why) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr first(gguf_init_from_file(path.c_str(), gp));
    if (!first) {
        why = "cannot read the GGUF metadata of " + path;
        return false;
    }
    const int64_t arch_id = gguf_find_key(first.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(first.get(), arch_id) != GGUF_TYPE_STRING) {
        why = "the model has no architecture";
        return false;
    }
    m.arch        = gguf_get_val_str(first.get(), arch_id);
    m.n_embd      = lm_kv_int(first.get(), m.arch + ".embedding_length");
    m.n_ff_exp    = lm_kv_int(first.get(), m.arch + ".expert_feed_forward_length");
    m.n_expert    = lm_kv_int(first.get(), m.arch + ".expert_count");
    m.n_used      = lm_kv_int(first.get(), m.arch + ".expert_used_count");
    m.n_ctx_train = lm_kv_int(first.get(), m.arch + ".context_length");

    // the tensors of every split
    std::vector<std::string> paths = { path };
    const int n_split = (int) lm_kv_int(first.get(), "split.count");
    if (n_split > 1) {
        char prefix[4096] = {};
        if (llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, n_split) == 0) {
            why = "cannot name the splits of " + path;
            return false;
        }
        for (int i = 1; i < n_split; ++i) {
            char sp[4096] = {};
            llama_split_path(sp, sizeof(sp), prefix, i, n_split);
            paths.emplace_back(sp);
        }
    }
    std::map<int, uint64_t> layer_bytes;  // trunk and MTP layers: routed bytes
    std::map<int, uint64_t> layer_expert; // one expert of the layer, each kind padded to 256 B
    std::map<int, std::array<uint64_t, 3>> layer_kind; // one expert of the layer per kind
    std::set<int> mtp;                    // blocks with NextN tensors (an MTP head)
    for (size_t f = 0; f < paths.size(); ++f) {
        gguf_context_ptr other;
        const gguf_context * g = first.get();
        if (f > 0) {
            other.reset(gguf_init_from_file(paths[f].c_str(), gp));
            if (!other) {
                why = "cannot read the GGUF metadata of " + paths[f];
                return false;
            }
            g = other.get();
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const char * name = gguf_get_tensor_name(g, i);
            int  il = -1;
            char kind[16] = {};
            if (sscanf(name, "blk.%d.%15[a-z_]", &il, kind) == 2 && strcmp(kind, "nextn") == 0) {
                mtp.insert(il);
                continue;
            }
            if (sscanf(name, "blk.%d.ffn_%15[a-z_]", &il, kind) != 2) {
                continue;
            }
            const std::string k = kind;
            if (k == "gate_up_exps") {
                m.merged = true;
            }
            if (k != "up_exps" && k != "gate_exps" && k != "down_exps" && k != "gate_up_exps") {
                continue;
            }
            if (strstr(name, ".weight") == nullptr) {
                continue;
            }
            const size_t bytes = gguf_get_tensor_size(g, i);
            m.exl3 = m.exl3 && ggml_is_exl3(gguf_get_tensor_type(g, i));
            layer_bytes[il] += bytes;
            layer_expert[il] += (bytes/std::max<int64_t>(m.n_expert, 1) + 255)/256*256;
            if (k != "gate_up_exps") {
                const int ki = k == "up_exps" ? 0 : k == "gate_exps" ? 1 : 2;
                layer_kind[il][(size_t) ki] += bytes/std::max<int64_t>(m.n_expert, 1);
            }
        }
    }
    for (const auto & [il, bytes] : layer_bytes) {
        m.exps_all += bytes;
        if (mtp.count(il) > 0) {
            continue;
        }
        m.exps_bytes  += bytes;
        m.expert_bytes = std::max(m.expert_bytes, layer_expert[il]);
        for (int k = 0; k < 3; ++k) {
            m.kind_bytes[k] = std::max(m.kind_bytes[k], layer_kind[il][(size_t) k]);
        }
        m.n_moe_layers++;
    }
    if (m.n_moe_layers == 0 || m.n_expert <= 0 || m.n_used <= 0 || m.n_embd <= 0 || m.n_ff_exp <= 0) {
        why = "the model has no routed experts";
        return false;
    }
    return true;
}

// the largest rank of the routed expert deltas of a LoRA file (blk.N.ffn_{up,gate,down}_exps.weight.lora_a [k, rank, E])
// and how many of them the expert-major op cannot take (not F16/F32)
static bool lm_lora_rank(const std::string & path, int64_t & rank, int & n_other) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context_ptr g(gguf_init_from_file(path.c_str(), gp));
    if (!g || meta == nullptr) {
        return false;
    }
    ggml_context_ptr meta_ptr(meta);
    rank    = 0;
    n_other = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t != nullptr; t = ggml_get_next_tensor(meta, t)) {
        const std::string name = ggml_get_name(t);
        if (name.find("_exps.weight.lora_") == std::string::npos || name.rfind("blk.", 0) != 0) {
            continue;
        }
        if (t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_F32) {
            n_other++;
        }
        if (name.size() > 7 && name.compare(name.size() - 7, 7, ".lora_a") == 0) {
            rank = std::max<int64_t>(rank, t->ne[1]);
        }
    }
    return true;
}

// The prompt buffers of the layer-major prefill for one pass of n_tokens, from a no_alloc load of the
// model and the context as they will run (llama_layer_exec_lend_bytes); false when it would not run.
static bool lm_layer_major_lend(common_params & params, int64_t n_tokens, size_t l1, uint64_t out[2], std::vector<size_t> & pieces) {
    struct quiet_t {
        ggml_log_callback cb;
        void * ud;
    } q;
    llama_log_get(&q.cb, &q.ud);
    if (params.verbosity < LOG_LEVEL_DEBUG) {
        llama_log_set([](ggml_log_level level, const char * text, void * ud) {
            const quiet_t * q = (const quiet_t *) ud;
            q->cb(level >= GGML_LOG_LEVEL_ERROR ? level : GGML_LOG_LEVEL_DEBUG, text, q->ud);
        }, &q);
    }
    llama_model_params mp = common_model_params_to_llama(params);
    mp.no_alloc  = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_context_params cp = common_context_params_to_llama(params);
    bool ok = false;
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mp);
    if (model != nullptr) {
        llama_context * ctx = llama_init_from_model(model, cp);
        if (ctx != nullptr) {
            size_t  pc[GGML_EXPERT_LEND_MAX_PIECES] = {};
            int32_t n_pc = 0;
            ok = llama_layer_exec_lend_bytes(ctx, n_tokens, l1, out, pc, &n_pc);
            pieces.assign(pc, pc + n_pc);
            llama_free(ctx);
        }
        llama_model_free(model);
    }
    llama_log_set(q.cb, q.ud);
    return ok;
}

// DeepSeek-V4.1 with the layer mode off: its prompt processing is slow, and a small n_batch cuts the exact tail prefill short
static void lm_warn_off(const common_params & params) {
    if (params.model.path.empty()) {
        return;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context_ptr g(gguf_init_from_file(params.model.path.c_str(), gp));
    if (!g) {
        return;
    }
    const int64_t id = gguf_find_key(g.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(g.get(), id) != GGUF_TYPE_STRING || strcmp(gguf_get_val_str(g.get(), id), "deepseek41") != 0) {
        return;
    }
    LOG_WRN("layer mode is off: DeepSeek-V4.1 prompt processing is slow without it; --layer-mode on is recommended. "
            "The exact tail prefill applies only in part when n_batch (%d) is smaller than the tail "
            "it needs\n", (int) params.n_batch);
}

void common_layer_mode_apply(common_params & params) {
    if (params.layer_mode == 0) {
        lm_warn_off(params);
        return;
    }
    auto off = [&](const std::string & why) {
        LOG_WRN("layer mode: %s; prompt processing runs as with --layer-mode off\n", why.c_str());
        params.layer_mode = 0;
    };

    // ---- support ---------------------------------------------------------------------------------------------------
    const bool cache = expert_cache_on(params);
    if (!cache) {
        off("it needs the expert cache (--expert-l1-mib / --expert-l2-mib), whose experts are read from host memory");
        return;
    }
    lm_model m;
    std::string why;
    if (!lm_read_model(params.model.path, m, why)) {
        off(why);
        return;
    }
    // the expert-major MoE op takes EXL3 experts (build_moe_ffn of these models, separate gate/up, SwiGLU); the
    // attention chunks exist for each of them
    static const char * const archs[] = { "glm5-next", "qwen4exp", "deepseek41", "deepseek4" };
    if (std::none_of(std::begin(archs), std::end(archs), [&](const char * a) { return m.arch == a; })) {
        off("model architecture '" + m.arch + "' is not supported yet (glm5-next, qwen4exp, deepseek41, deepseek4)");
        return;
    }
    if (!m.exl3 || m.merged) {
        off(m.merged ? "merged gate/up experts are not supported" : "only EXL3 routed experts are supported");
        return;
    }
    // contexts the no_alloc measurement does not build (a draft or MTP context, a multimodal projector) are not planned
    // for yet: the mode stays off with them rather than guessing their memory. ngram-varlen has no context of its own and
    // reads no layer inputs; the rollback planes it asks of the target (n_rs_seq) are part of the KV cache that the
    // measurement builds from the same context parameters
    if (params.speculative.has_dft() || std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) { return t != COMMON_SPECULATIVE_TYPE_NONE && t != COMMON_SPECULATIVE_TYPE_NGRAM_VARLEN; })) {
        off("a draft model or MTP context is not covered by the memory check yet");
        return;
    }
    if (!params.mmproj.path.empty()) {
        off("a multimodal projector (--mmproj) is not covered by the memory check yet");
        return;
    }
    // several sequences: the measurement builds the context with every sequence, so its graph is in the plan; without a
    // unified KV cache a ubatch of several sequences is not chunked (llm_graph_chunk_mask_usable)
    if (params.n_parallel > 1 && !params.kv_unified) {
        LOG_WRN("layer mode: -np %d without a unified KV cache: the attention runs unchunked\n", params.n_parallel);
    }
    // LoRA on routed experts: the op adds F16/F32 deltas itself (its rank rows are in the scratch below); a delta of
    // another type keeps its layer on the per-kind MoE path, whose intermediates this plan does not count
    int64_t max_rank = 0;
    for (const common_adapter_lora_info & la : params.lora_adapters) {
        int64_t rank = 0;
        int n_other = 0;
        if (!lm_lora_rank(la.path, rank, n_other)) {
            LOG_WRN("layer mode: cannot read the LoRA %s; its expert deltas are not in the plan\n", la.path.c_str());
            continue;
        }
        max_rank = std::max(max_rank, rank);
        if (n_other > 0) {
            LOG_WRN("layer mode: LoRA %s: %d routed expert deltas are neither F16 nor F32: those layers keep the per-kind MoE "
                "path, whose intermediates the plan does not count\n", la.path.c_str(), n_other);
        }
    }
    if (params.n_ctx == 0) {
        LOG_WRN("%s", "layer mode: no -c given: sized for the training context; set -c for a tighter plan\n");
    }

    // ---- development overrides --------------------------------------------------------------------------------------
    const bool env_stage = lm_env_set("RANMA_EXPERT_MAJOR_STAGE_MIB");
    // the streamed op (RANMA_EXPERT_MAJOR_STREAM=1, mul-mat-had.cu): a ring of expert slots in the op's scratch block
    const bool    stream       = lm_env_set("RANMA_EXPERT_MAJOR_STREAM") && atoi(getenv("RANMA_EXPERT_MAJOR_STREAM")) != 0;
    const int64_t stream_group = stream ? std::max(1, lm_env_set("RANMA_EXPERT_MAJOR_STREAM_GROUP") ?
        atoi(getenv("RANMA_EXPERT_MAJOR_STREAM_GROUP")) : 16) : 0;
    const bool    env_ring     = lm_env_set("RANMA_EXPERT_MAJOR_STREAM_MIB");
    const int64_t stage_min = lm_env_set("RANMA_EXPERT_MAJOR_STAGE_MIN") ? std::max(1, atoi(getenv("RANMA_EXPERT_MAJOR_STAGE_MIN"))) : 129;

    // the VRAM kept outside the plan: --fit-target of the device. --expert-l1-mib auto: L1 is what the model, the KV
    // cache and the compute buffers of a width leave less that reserve, so each width has its own L1; the width's lent
    // buffers must fit that L1
    const bool    l1_auto = params.expert_l1_auto && params.expert_l1_mib <= 0;
    const int64_t reserve = params.fit_params_target.empty() ? 0 : (int64_t) params.fit_params_target[0];
    int64_t l1 = l1_auto ? 0 : (int64_t) params.expert_l1_mib*(int64_t) MiB;

    // ---- width: the widest that fits, at most the context of one sequence -------------------------------------------
    // No rate of the link or of the GEMMs enters: the measurement below decides. The upper end is the context of one
    // sequence and the widest pass the expert-major op takes (its GEMM items: min(E, rows) + rows/16 + 1 per pass).
    const int64_t n_ctx     = params.n_ctx > 0 ? params.n_ctx : m.n_ctx_train;
    const int64_t n_ctx_seq = params.n_parallel > 1 && !params.kv_unified ? n_ctx/params.n_parallel : n_ctx;
    int64_t w_op = 512;
    while ((w_op + 512)*m.n_used/16 + std::min<int64_t>(m.n_expert, (w_op + 512)*m.n_used) + 1 <= LM_MAX_ITEMS) {
        w_op += 512;
    }
    int64_t ub_top = std::max<int64_t>(std::min<int64_t>(n_ctx_seq, w_op)/512*512, 512);
    if (params.n_batch_explicit) {
        ub_top = std::min<int64_t>(ub_top, params.n_batch);
    }
    // The layer-major prefill reads each expert once per pass, and a pass is as wide as L1 holds its
    // buffers (the width rule, at prompt time). The ubatch is then only the width of its segment graphs: it sizes the
    // compute buffers (fixed, outside L1), not the expert reads, so it is at most LM_LX_UB instead of the widest that
    // fits. The ubatch loop without it keeps the widest (there a wider ubatch reads the experts fewer times).
    // (it exists for DeepSeek-V4 and V4.1; the layer mode turns it on for them, see below)
    const bool lx_on = m.arch == "deepseek4" || m.arch == "deepseek41";
    if (lx_on) {
        ub_top = std::min<int64_t>(ub_top, LM_LX_UB);
    }
    int64_t ub = params.n_ubatch_explicit ? params.n_ubatch : ub_top;
    const double host_layer = (double) std::max<int64_t>(0, (int64_t) m.exps_bytes - l1)/m.n_moe_layers;
    LOG_INF("layer mode: %s, %d MoE layers, %.0f MiB of experts per layer outside L1; width %s %lld (context of a sequence "
        "%lld, expert-major op up to %lld)\n", m.arch.c_str(), m.n_moe_layers, host_layer/MiB,
        params.n_ubatch_explicit ? "(-ub)" : "at most", (long long) ub, (long long) n_ctx_seq, (long long) w_op);

    // ---- buffers of the expert-major op, per width -------------------------------------------------------------------
    // rows per op chunk: all rows of the width (width * top_k), so that every expert is read once per layer; the op's
    // scratch then grows with the width (about 40 KiB per row on GLM-5.3), which the memory check below counts
    auto rows_of = [=](int64_t w) { return w*m.n_used; };
    // scratch: every device buffer of one launch of the width (moe-ffn-had-layout.h, the op's own layout), with the
    // rank rows of the LoRA deltas
    // the ring of the streamed op (one chunk, no LoRA): two groups of slots (or RANMA_EXPERT_MAJOR_STREAM_MIB), at most the
    // experts that can have more rows than the host tile the op picks for the width (rows / (tile + 1))
    auto ring_bytes = [=](int64_t w) -> int64_t {
        if (!stream || rows_of(w) < w*m.n_used || max_rank > 0) {
            return 0;
        }
        if (env_ring) {
            return (int64_t) std::max(0, atoi(getenv("RANMA_EXPERT_MAJOR_STREAM_MIB")))*(int64_t) MiB;
        }
        const int64_t rows = w*m.n_used;
        const int64_t used = std::max<int64_t>(1, std::min<int64_t>(rows, m.n_expert));
        const int64_t want = 3*((rows + used - 1)/used);
        const int64_t tile = want <= 16 ? 16 : want <= 32 ? 32 : want <= 64 ? 64 : 128;
        const int64_t slots = std::min<int64_t>({ 2*stream_group, (int64_t) m.n_expert, rows/(tile + 1) });
        return slots > 0 ? (slots*(int64_t) m.expert_bytes + MiB - 1)/MiB*MiB : 0;
    };
    // the op's scratch of one prompt ubatch, as the three blocks the prefill lend places in L1 (moe-ffn-had-layout.h). The
    // layer-major prefill borrows its own buffers for each call (src/llama-layer-exec.cpp).
    auto scratch_split = [=](int64_t w, size_t out[3]) {
        const int64_t ring = ring_bytes(w);
        ggml_moe_ffn_had_scratch_split(w, m.n_used, rows_of(w), m.n_expert, m.n_embd, m.n_ff_exp,
            rows_of(w) < w*m.n_used ? LM_STAGE_SLOTS : 0, max_rank, out, ring > 0 ? stream_group : 0, ring);
    };
    auto scratch_bytes = [&](int64_t w) -> int64_t {
        size_t b[3];
        scratch_split(w, b);
        return (int64_t) (b[0] + b[1] + b[2]);
    };
    // The prefill lend puts the blocks into the top slots of the L1 arena of each kind, after the lowest ranked experts
    // left. Estimated before the load with the largest layer's experts in one size class, the exclusive exchange spares kept.
    auto lend_fits = [&](int64_t w, int64_t l1w) {
        size_t b[3];
        scratch_split(w, b);
        const uint64_t per = m.kind_bytes[0] + m.kind_bytes[1] + m.kind_bytes[2];
        if (per == 0) {
            return false;
        }
        const int64_t slots = std::max<int64_t>(0, l1w/(int64_t) per - (params.expert_cache_mode == "exclusive" ? 8 : 0));
        std::vector<size_t> regions;
        for (int k = 0; k < 3; ++k) {
            const size_t span = (size_t) slots*m.kind_bytes[k];
            regions.push_back(span > 4096 ? span - 4096 : 0);
        }
        std::vector<ggml_cuda_expert::lend_fit> fit;
        return ggml_cuda_expert::lend_pack({ b[0], b[1], b[2] }, regions, 256, fit);
    };
    // staging (two buffers): only experts with at least stage_min rows of a chunk are staged, at most R/stage_min of them.
    // The streamed op of one chunk does not stage: its ring is in the scratch block.
    auto stage_bytes = [&](int64_t w) -> int64_t {
        if (stream && rows_of(w) >= w*m.n_used && max_rank == 0) {
            return 0;
        }
        if (env_stage) {
            return (int64_t) std::max(0, atoi(getenv("RANMA_EXPERT_MAJOR_STAGE_MIB")))*(int64_t) MiB;
        }
        if (w*m.n_used/m.n_expert < stage_min) {
            return 0; // a mean expert gets fewer rows than a tile: reading in place costs the same link bytes
        }
        if (rows_of(w) >= w*m.n_used) {
            return 0; // one chunk: no earlier chunk to hide the copies behind
        }
        const int64_t r     = std::min<int64_t>(rows_of(w), w*m.n_used);
        const int64_t slots = std::min<int64_t>(LM_STAGE_SLOTS, (r + stage_min - 1)/stage_min);
        return (2*slots*(int64_t) m.expert_bytes + 2*256 + MiB - 1)/MiB*MiB;
    };

    // ---- measurement: a no_alloc load of the model and the context at (width, chunk) ----------------------------------
    // what the layer mode decides for the graphs (llama_layer_mode_config): the expert-major MoE op and the
    // layer-major prefill from here on, the attention chunk per measurement; the values before are restored when the mode
    // stays off
    const llama_layer_mode_config lm_prev = llama_layer_mode_get_config();
    llama_layer_mode_config lm_cfg = lm_prev;
    lm_cfg.expert_major = true;
    lm_cfg.layer_exec   = lx_on;
    llama_layer_mode_set_config(lm_cfg);
    const bool    chunked   = !(params.n_parallel > 1 && !params.kv_unified);
    const int32_t ub_prev   = params.n_ubatch;
    const int32_t b_prev    = params.n_batch;

    auto set_width = [&](int64_t w) {
        params.n_ubatch = (int32_t) w;
        params.n_batch  = params.n_batch_explicit ? std::max<int32_t>(params.n_batch, (int32_t) w) : (int32_t) w;
    };
    int64_t meta_w = 0; // width and bytes of the last measurement that raised the process peak
    int64_t meta_b = 0;
    auto eval = [&](int64_t w, int64_t c) {
        set_width(w);
        lm_cfg.attn_chunk = c >= w ? 0 : c;
        llama_layer_mode_set_config(lm_cfg);
        llama_model_params   mp = common_model_params_to_llama(params);
        llama_context_params cp = common_context_params_to_llama(params);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0, hp_ctx = 0, hp_exp = 0;
        int64_t now0, peak0, now1, peak1;
        lm_private(now0, peak0);
        common_device_memory_data_vec dm;
        lm_eval e;
        try {
            dm = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, hp_ngl, hp_ctx, hp_exp,
                params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
        } catch (const std::exception & ex) {
            LOG_WRN("layer mode: measurement at width %lld failed: %s\n", (long long) w, ex.what());
            return e;
        }
        lm_private(now1, peak1);
        // the graph metadata of the wide graph: the growth of the process peak over the private bytes before the load.
        // The peak only rises, so a load that stays under an earlier peak is estimated from the last one that rose it,
        // in proportion to the width (the node count of a graph grows with its width)
        int64_t meta = 0;
        if (peak1 >= 0 && now0 >= 0) {
            if (peak1 > peak0 || meta_w == 0) {
                meta   = std::max<int64_t>(0, peak1 - now0);
                meta_w = w;
                meta_b = meta;
            } else {
                meta = (int64_t) ((double) meta_b*w/meta_w);
            }
        }
        if (dm.size() < 2) {
            LOG_WRN("%s", "layer mode: the measurement found no GPU\n");
            return e;
        }
        const common_device_memory_data & d = dm[0];
        // the op's buffers are lent from L1 (the prefill lend), not added to it
        const int64_t buffers = scratch_bytes(w);
        e.l1       = l1_auto ? (int64_t) expert_l1_auto_bytes(d.free, d.model, d.context, d.compute, (size_t) reserve) : l1;
        const bool    lent    = e.l1 > 0 && lend_fits(w, e.l1);
        e.dev_free = d.free;
        e.dev_need = (int64_t) (d.model + d.context + d.compute) + reserve + e.l1;
        e.host     = (int64_t) dm.back().compute + meta;
        e.ok       = (l1_auto ? e.l1 > 0 : e.dev_need <= e.dev_free) && lent;
        LOG_INF("layer mode: width %lld, chunk %lld: GPU model %zu + KV %zu + compute %zu + L1 %lld%s + reserve %lld = %lld of %lld "
            "MiB free; op buffers %lld MiB lent from L1: %s; host compute %zu + graph %lld MiB (not a limit) -> %s\n", (long long) w,
            (long long) std::min(c, w), d.model/MiB, d.context/MiB, d.compute/MiB, (long long) (e.l1/MiB), l1_auto ? " (auto)" : "",
            (long long) (reserve/MiB), (long long) (e.dev_need/MiB), (long long) (e.dev_free/MiB), (long long) (buffers/MiB),
            lent ? "fits" : "does not fit", dm.back().compute/MiB, (long long) (meta/MiB),
            e.ok ? "fits" : "does not fit");
        return e;
    };

    // the chunk: the given one, or the smallest (the most chunks: the least compute buffer, the most graph nodes)
    auto c_low = [&](int64_t w) { return chunked ? std::min<int64_t>(LM_CHUNK_MIN, w) : w; };

    // the unchunked graph first (chunks cost time: every chunk adds its own attention nodes per layer), then the widest
    // width with the smallest chunk: halved until it fits, then bisected between the fitting and the failing width
    const bool try_whole = chunked;
    lm_eval e_whole;
    if (try_whole) {
        e_whole = eval(ub, ub);
    }
    lm_eval e = e_whole;
    if (!e_whole.ok) {
        e = eval(ub, c_low(ub));
        if (!e.ok && !params.n_ubatch_explicit && e.dev_free > 0) {
            int64_t hi = ub; // does not fit
            int64_t lo = 0;  // fits
            lm_eval e_lo;
            while (lo == 0 && hi > LM_UB_MIN) {
                const int64_t w = std::max<int64_t>(LM_UB_MIN, hi/2/512*512);
                lm_eval ew = eval(w, c_low(w));
                if (ew.ok) {
                    lo   = w;
                    e_lo = ew;
                } else {
                    hi = w;
                }
            }
            // to within 1/16 of the width (at least 512)
            while (lo > 0 && hi - lo > std::max<int64_t>(512, lo/16)) {
                const int64_t w = (lo + (hi - lo)/2)/512*512;
                if (w <= lo) {
                    break;
                }
                lm_eval ew = eval(w, c_low(w));
                if (ew.ok) {
                    lo   = w;
                    e_lo = ew;
                } else {
                    hi = w;
                }
            }
            if (lo > 0) {
                ub = lo;
                e  = e_lo;
            } else {
                ub = hi;
            }
        }
    }
    if (!e.ok) {
        params.n_ubatch = ub_prev;
        params.n_batch  = b_prev;
        llama_layer_mode_set_config(lm_prev);
        off("no width of at least " + std::to_string(std::min<int64_t>(ub, LM_UB_MIN)) + " tokens fits the free memory left by the "
            "model, the KV cache and --expert-l1-mib");
        return;
    }

    // the largest chunk that fits: the unchunked graph, else between the two measured ends, linearly, checked
    int64_t c = e_whole.ok ? ub : c_low(ub);
    if (!e_whole.ok && chunked && c < ub) {
        const lm_eval hi = eval(ub, ub);
        if (hi.ok) {
            c = ub;
        } else {
            const double slope = (double) (hi.dev_need - e.dev_need)/(double) (ub - c);
            int64_t cc = slope > 0 ? c + (int64_t) ((e.dev_free - e.dev_need)/slope) : c;
            cc = std::clamp<int64_t>(cc/256*256, c, ub - 256);
            for (int tries = 0; cc > c && tries < 4; ++tries) {
                if (eval(ub, cc).ok) {
                    c = cc;
                    break;
                }
                cc = std::max<int64_t>(c, (cc*3/4)/256*256);
            }
        }
    }

    // --expert-l1-mib auto: the L1 the chosen width and chunk leave (one more measurement of exactly that setup)
    if (l1_auto) {
        const lm_eval fin = eval(ub, c);
        if (fin.l1 > 0) {
            l1 = fin.l1;
            params.expert_l1_mib = (int32_t) (l1/(int64_t) MiB);
            LOG_INF("layer mode: --expert-l1-mib auto is %d MiB at width %lld (the free VRAM less the model, the KV cache, "
                "the compute buffers and --fit-target %lld MiB)\n", params.expert_l1_mib, (long long) ub, (long long) (reserve/MiB));
        }
    }

    // ---- apply -------------------------------------------------------------------------------------------------------
    set_width(ub);
    // the logical batch: the context (the context clamps it), so that a prompt is one call of several ubatches and a
    // model that computes only what the output of a call needs (the exact tail of DeepSeek-V4.1) does that once, at the
    // end of the prompt. The measurement above ran with -b = -ub: n_batch sizes no buffer of these contexts (KV cache
    // and graphs follow n_ubatch, the outputs n_outputs_max), only the per-token target features a drafter reads, and a
    // drafter turns the layer mode off (above)
    if (!params.n_batch_explicit) {
        params.n_batch = (int32_t) std::max<int64_t>(ub, n_ctx);
    }
    lm_cfg.attn_chunk        = c >= ub ? 0 : c;
    lm_cfg.expert_major_rows = rows_of(ub);
    llama_layer_mode_set_config(lm_cfg);
    // the op's chunk staging is a VRAM block of its own (RANMA_EXPERT_MAJOR_STAGE_MIB): the layer mode does not set it,
    // its buffers come out of the prefill lend only
    const int64_t stage = env_stage ? stage_bytes(ub) : 0;
    if (stream && !env_ring) {
        // the op sizes its ring from this (0: no ring, the op keeps its other path)
        lm_setenv("RANMA_EXPERT_MAJOR_STREAM_MIB", std::to_string(ring_bytes(ub)/MiB));
    }
    // the op's scratch of a prompt ubatch is lent from L1 while the prompt runs (ggml_expert_config::lend_scratch)
    params.layer_mode_lend = true;
    {
        scratch_split(ub, params.layer_mode_lend_scratch);
        // a call borrows the blocks of its own width (copies only: the lambda outlives this function)
        params.layer_mode_lend_scratch_of = [scratch_split](int64_t w, size_t out[3]) { scratch_split(std::max<int64_t>(w, 1), out); };
    }
    if (params.layer_mode_preempt_ms <= 0) {
        params.layer_mode_preempt_ms = LM_PREEMPT_MS;
    }
    expert_host_direct_auto(params.n_ubatch);
    // An L2 of -1 also holds the L1 experts the prefill lend evicts (ggml_expert_config::l2_lend_pieces): a lend takes the
    // op's scratch of a ubatch-major prompt or the layer-major pieces of a pass (a new lend ends the one before it, so the
    // two never hold at once). The width rule keeps a pass inside L1, so it is the pieces of one pass over the context of
    // a sequence, at most all of L1. The backend packs both piece lists into the L1 slots as the lend does and keeps ring
    // slots for the larger eviction; layer_mode_lend_bytes is the larger sum, for its log.
    params.layer_mode_lend_bytes = params.layer_mode_lend_scratch[0] + params.layer_mode_lend_scratch[1] + params.layer_mode_lend_scratch[2];
    params.layer_mode_lend_pieces.clear();
    if (params.expert_l2_mib < 0 && lx_on && n_ctx_seq > ub) {
        uint64_t b[2] = { 0, 0 };
        if (lm_layer_major_lend(params, n_ctx_seq, (size_t) l1, b, params.layer_mode_lend_pieces)) {
            LOG_INF("layer mode: the layer-major prefill of %lld tokens in one pass borrows %llu MiB of VRAM (host inputs %llu "
                "MiB, not lent)\n", (long long) n_ctx_seq, (unsigned long long) (b[0] >> 20), (unsigned long long) (b[1] >> 20));
            params.layer_mode_lend_bytes = std::max<size_t>(params.layer_mode_lend_bytes, b[0]);
        } else {
            LOG_INF("%s", "layer mode: the layer-major prefill could not size its prompt buffers; the L2 of -1 holds the op scratch\n");
        }
    }
    params.layer_mode_lend_bytes = std::min<size_t>(params.layer_mode_lend_bytes, (size_t) std::max<int64_t>(0, l1));
    LOG_INF("layer mode: on: -ub %d -b %d, attention chunk %s, expert-major MoE %s (rows %lld), staging %lld MiB, scratch "
        "%lld MiB (stream ring %lld MiB) lent from L1 while a prompt runs (blocks %zu/%zu/%zu MiB), host-direct up to %d tokens, "
        "other slots' prompt batches about %d ms\n",
        params.n_ubatch, params.n_batch, c >= ub ? "off (whole width)" : std::to_string(c).c_str(),
        "on", (long long) std::min<int64_t>(rows_of(ub), ub*m.n_used),
        (long long) (stage/MiB), (long long) (scratch_bytes(ub)/MiB), (long long) (ring_bytes(ub)/MiB),
        params.layer_mode_lend_scratch[0] >> 20, params.layer_mode_lend_scratch[1] >> 20, params.layer_mode_lend_scratch[2] >> 20,
        params.n_ubatch, params.layer_mode_preempt_ms);
}
