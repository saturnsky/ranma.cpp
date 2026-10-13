#include "layer-mode.h"

#include "expert.h"
#include "fit.h"
#include "log.h"

#include "ggml-cpp.h"
#include "gguf.h"

#include "../src/llama-ext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
// is not derived from rates of this machine: it is the widest that the measured memory allows.
// - reserve: MiB of VRAM kept outside the plan (HIP pool, runtime, graph copies; 3.1 GiB measured on DS4F, plus room).
// - preempt: ms a prompt batch of a slot other than 0 may take while slot 0 has priority (the wait of a new slot-0 request).
#define LM_RESERVE_MIB  4096
#define LM_PREEMPT_MS   3000
#define LM_UB_MIN       1024
#define LM_CHUNK_MIN    512
#define LM_ROWS         4096 // MOE_FFN_HAD rows per chunk (MMH_MOE_ROWS)
#define LM_STAGE_SLOTS  64   // MMH_MOE_STAGE_MAX_SLOTS
#define LM_MAX_ITEMS    65535 // MOE_FFN_HAD: GEMM items of one pass (grid y)

static const size_t MiB = 1024*1024;

static double lm_env_f(const char * name, double def) {
    const char * v = getenv(name);
    return v != nullptr && v[0] != '\0' ? atof(v) : def;
}

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
    int      n_moe_layers = 0;
    uint64_t exps_bytes   = 0;    // every routed expert weight, three kinds
    uint64_t expert_bytes = 0;    // largest staging slot: one expert of a layer, three kinds, each padded to 256 B
    bool     exl3         = true; // every routed expert weight is EXL3
    bool     merged       = false; // gate and up merged (the expert-major op does not take it)
};

struct lm_eval {
    bool    ok       = false;
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
    std::set<int> layers;
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
            m.exps_bytes += bytes;
            m.exl3 = m.exl3 && ggml_is_exl3(gguf_get_tensor_type(g, i));
            layers.insert(il);
        }
    }
    m.n_moe_layers = (int) layers.size();
    if (m.n_moe_layers == 0 || m.n_expert <= 0 || m.n_used <= 0 || m.n_embd <= 0 || m.n_ff_exp <= 0) {
        why = "the model has no routed experts";
        return false;
    }
    // one expert of a layer, the three kinds each padded as the op pads them (the mean layer: EXL3 layers of one model
    // share the bits of each kind in the RANMA releases; a larger layer only lowers its staging slots)
    const uint64_t per_layer = m.exps_bytes/m.n_moe_layers;
    m.expert_bytes = (per_layer/m.n_expert + 3*255)/256*256;
    return true;
}

void common_layer_mode_apply(common_params & params) {
    if (params.layer_mode == 0) {
        return;
    }
    auto off = [&](const std::string & why) {
        LOG_WRN("layer mode: %s; prompt processing runs as with --layer-mode off\n", why.c_str());
        params.layer_mode = 0;
    };

    // ---- support ---------------------------------------------------------------------------------------------------
    const bool cache = params.expert_l1_mib > 0 || params.expert_l2_mib > 0;
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
    // the expert-major MoE op takes EXL3 experts; the attention chunks exist for GLM-5.3 (glm5-next)
    if (m.arch != "glm5-next") {
        off("model architecture '" + m.arch + "' is not supported yet (glm5-next only)");
        return;
    }
    if (!m.exl3 || m.merged) {
        off(m.merged ? "merged gate/up experts are not supported" : "only EXL3 routed experts are supported");
        return;
    }
    if (params.n_parallel > 1 && !params.kv_unified) {
        LOG_WRN("layer mode: -np %d without a unified KV cache: the attention runs unchunked (one stream only)\n", params.n_parallel);
    }
    if (!params.lora_adapters.empty()) {
        LOG_WRN("%s", "layer mode: a LoRA on expert weights keeps those layers on the per-kind MoE path, whose intermediates "
            "this sizing does not count\n");
    }
    if (params.speculative.has_dft() || std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
            [](common_speculative_type t) { return t != COMMON_SPECULATIVE_TYPE_NONE; })) {
        LOG_WRN("%s", "layer mode: the draft context is not included in the memory check\n");
    }
    if (params.n_ctx == 0) {
        LOG_WRN("%s", "layer mode: no -c given: sized for the training context; set -c for a tighter plan\n");
    }

    // ---- development overrides --------------------------------------------------------------------------------------
    const bool env_stage = lm_env_set("RANMA_EXPERT_MAJOR_STAGE_MIB");
    const int64_t stage_min = lm_env_set("RANMA_EXPERT_MAJOR_STAGE_MIN") ? std::max(1, atoi(getenv("RANMA_EXPERT_MAJOR_STAGE_MIN"))) : 129;

    const int64_t reserve = (int64_t) lm_env_f("RANMA_LAYER_MODE_RESERVE_MIB", LM_RESERVE_MIB)*(int64_t) MiB;
    const int64_t l1 = (int64_t) params.expert_l1_mib*(int64_t) MiB;
    // host: what is available now, less what the expert cache will take in host memory (the finite L2, or every expert
    // outside L1 when it is unlimited) and a reserve for the rest of the process and the other programs on the host
            std::max<int64_t>(0, (int64_t) m.exps_bytes - l1);

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
    int64_t ub = params.n_ubatch_explicit ? params.n_ubatch : ub_top;
    const double host_layer = (double) std::max<int64_t>(0, (int64_t) m.exps_bytes - l1)/m.n_moe_layers;
    LOG_INF("layer mode: %s, %d MoE layers, %.0f MiB of experts per layer outside L1; width %s %lld (context of a sequence "
        "%lld, expert-major op up to %lld)\n", m.arch.c_str(), m.n_moe_layers, host_layer/MiB,
        params.n_ubatch_explicit ? "(-ub)" : "at most", (long long) ub, (long long) n_ctx_seq, (long long) w_op);

    // ---- buffers of the expert-major op, per width -------------------------------------------------------------------
    // rows per op chunk: all rows of the width (width * top_k), so that every expert is read once per layer; the op's
    // scratch then grows with the width (about 40 KiB per row on GLM-5.3), which the memory check below counts
    auto rows_of = [=](int64_t w) { return w*m.n_used; };
    // scratch (HIP pool, op lifetime, kept by the pool): R rows of x (fp16), up, gate (f32), y (f32), plus routing tables
    auto scratch_bytes = [&](int64_t w) {
        const int64_t r = std::min<int64_t>(rows_of(w), w*m.n_used);
        return r*(2*std::max(m.n_embd, m.n_ff_exp) + 8*m.n_ff_exp + 4*m.n_embd + 8) + w*m.n_used*12;
    };
    // staging (two buffers): only experts with at least stage_min rows of a chunk are staged, at most R/stage_min of them
    auto stage_bytes = [&](int64_t w) -> int64_t {
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
    llama_layer_mode_set_config(lm_cfg);
    const bool    chunked   = !(params.n_parallel > 1 && !params.kv_unified);
    const int32_t ub_prev   = params.n_ubatch;
    const int32_t b_prev    = params.n_batch;
    // the buffers come out of the L1 budget when it is at least twice as large, else on top of it
    auto carve_bytes = [&](int64_t w) {
        const int64_t buffers = stage_bytes(w) + scratch_bytes(w);
        return l1 >= 2*buffers ? buffers : (int64_t) 0;
    };

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
        const int64_t buffers = stage_bytes(w) + scratch_bytes(w);
        const int64_t carve   = carve_bytes(w);
        e.dev_free = d.free;
        e.dev_need = (int64_t) (d.model + d.context + d.compute) + reserve + l1 + (buffers - carve);
        e.host     = (int64_t) dm.back().compute + meta;
        e.ok       = e.dev_need <= e.dev_free;
        LOG_INF("layer mode: width %lld, chunk %lld: GPU model %zu + KV %zu + compute %zu + L1 %lld + buffers %lld + reserve "
            "%lld = %lld of %lld MiB free; host compute %zu + graph %lld MiB (not a limit) -> %s\n", (long long) w,
            (long long) std::min(c, w), d.model/MiB, d.context/MiB, d.compute/MiB, (long long) (l1/MiB),
            (long long) (buffers/MiB), (long long) (reserve/MiB), (long long) (e.dev_need/MiB), (long long) (e.dev_free/MiB),
            dm.back().compute/MiB, (long long) (meta/MiB), e.ok ? "fits" : "does not fit");
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

    // ---- apply -------------------------------------------------------------------------------------------------------
    set_width(ub);
    lm_cfg.attn_chunk        = c >= ub ? 0 : c;
    lm_cfg.expert_major_rows = rows_of(ub);
    llama_layer_mode_set_config(lm_cfg);
    const int64_t stage = stage_bytes(ub);
    if (!env_stage && stage > 0) {
        lm_setenv("RANMA_EXPERT_MAJOR_STAGE_MIB", std::to_string(stage/MiB));
    }
    params.layer_mode_l1_carve_mib = (int32_t) ((carve_bytes(ub) + MiB - 1)/MiB);
    if (params.layer_mode_preempt_ms <= 0) {
        params.layer_mode_preempt_ms = LM_PREEMPT_MS;
    }
    expert_host_direct_auto(params.n_ubatch);

    LOG_INF("layer mode: on: -ub %d -b %d, attention chunk %s, expert-major MoE %s (rows %lld), staging %lld MiB, scratch "
        "%lld MiB, %d MiB of them from the L1 budget, host-direct up to %d tokens, other slots' prompt batches about %d ms\n",
        params.n_ubatch, params.n_batch, c >= ub ? "off (whole width)" : std::to_string(c).c_str(),
        "on", (long long) std::min<int64_t>(rows_of(ub), ub*m.n_used),
        (long long) (stage/MiB), (long long) (scratch_bytes(ub)/MiB), params.layer_mode_l1_carve_mib, params.n_ubatch,
        params.layer_mode_preempt_ms);
}
