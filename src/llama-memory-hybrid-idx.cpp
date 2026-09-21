#include "llama-memory-hybrid-idx.h"

#include "llama-kv-cache-dsv4.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-qsa-dump.h"


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>

// the indexer cache layout is part of the sequence state, so a file written by another
// layout must be refused instead of read as garbage
static constexpr uint32_t LLAMA_HYBRID_IDX_STATE_MAGIC          = 0x58444951; // QIDX
static constexpr uint32_t LLAMA_HYBRID_IDX_STATE_VERSION_RAW    = 1;   // one key per token
static constexpr uint32_t LLAMA_HYBRID_IDX_STATE_VERSION_POOLED = 2;   // one pooled key per block

static constexpr uint32_t LLAMA_HYBRID_IDX_STATE_VERSION_TRANSFORMED = 3;

// [TAG_QSA_POOLED] the pooled indexer is the default; the per-token cache stays available
static bool qwen_idx_pooled_enabled() {
    const char * env = std::getenv("LLAMA_QSA_LEGACY");
    return env == nullptr || std::atoi(env) == 0;
}

static bool qwen_idx_norm_rope_enabled() {
    const char * env = std::getenv("LLAMA_QSA_CACHE_NORM_ROPE");
    return env == nullptr || std::atoi(env) != 0;
}

// pooled keys the cache can be asked for, one per block of `ratio` tokens
static uint32_t qwen_idx_comp_rows(uint32_t kv_size, uint32_t ratio, uint32_t n_pad) {
    return GGML_PAD(std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio), n_pad);
}

static int64_t qwen_idx_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("Qwen pooled indexer sequence id out of stream range");
    }
    return (int64_t) seq_id*size;
}

static bool qwen_idx_token_has_seq(const llama_ubatch & ubatch, uint32_t i, llama_seq_id seq_id) {
    for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
        if (ubatch.seq_id[i][s] == seq_id) {
            return true;
        }
    }
    return false;
}

// Plans one ubatch: which raw keys stay as the open block's members, which blocks the
// ubatch completes, and where their pooled keys go. Decode completes 0 or 1 block.
//
// A reservation ubatch carries no real positions - every token sits at 0 - so it is planned
// as if its tokens were consecutive, and at the full block count, which is the worst case
// a later graph can ask for.
static llama_memory_hybrid_idx_context::idx_pool_plan qwen_idx_build_pool_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        uint32_t kv_size,
        uint32_t n_rows,
        uint32_t n_stream,
        bool reserve) {
    llama_memory_hybrid_idx_context::idx_pool_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = ubatch.n_seqs_unq;

    if (plan.n_stream <= 0) {
        plan.n_stream = 1;
    }

    const uint32_t n_tps = std::max<uint32_t>(1, ubatch.n_seq_tokens);

    std::vector<llama_pos> pos_of(ubatch.n_tokens);
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        pos_of[i] = reserve ? (llama_pos) (i%n_tps) : ubatch.pos[i];
    }

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    const int64_t state_rows = (int64_t) ratio*n_stream;
    std::vector<persist_row> persist_rows;
    std::map<std::pair<llama_seq_id, llama_pos>, int32_t> current;
    std::map<llama_seq_id, uint32_t> write_counts;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            current[{ ubatch.seq_id[i][s], pos_of[i] }] = (int32_t) i;
        }
    }

    // a member either arrives in this ubatch or was persisted by an earlier one
    const auto source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        const auto it = current.find({ seq_id, pos });
        if (it != current.end()) {
            return (int32_t) (state_rows + it->second);
        }
        return (int32_t) (qwen_idx_stream_offset(n_stream, seq_id, ratio) + pos%ratio);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = pos_of[i];
        if (pos < 0) {
            continue;
        }

        plan.n_visible[i] = (int32_t) ((pos + 1)/ratio);
        plan.n_kv = std::max<int64_t>(plan.n_kv, plan.n_visible[i]);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int32_t dst = (int32_t) (qwen_idx_stream_offset(n_stream, seq_id, ratio) + pos%ratio);
            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [dst](const persist_row & row) { return row.dst == dst; });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ dst, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1)%ratio != 0) {
                continue;
            }

            const llama_pos start = pos + 1 - ratio;
            const int64_t cache_off = qwen_idx_stream_offset(n_stream, seq_id, kv_size);
            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            for (uint32_t j = 0; j < ratio; ++j) {
                plan.state_read_idxs.push_back(source_idx(seq_id, start + j));
            }
            ++write_counts[seq_id];
        }
    }

    // Keep the graph shape stable for decode and equal-size ubatches. A step that does not
    // seal a real block writes a scratch row instead; the cache reserves its tail for that,
    // so the write can never land on a block the graph reads.
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        uint32_t n_tokens = 0;
        uint32_t i_first = ubatch.n_tokens;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (pos_of[i] >= 0 && qwen_idx_token_has_seq(ubatch, i, seq_id)) {
                ++n_tokens;
                i_first = std::min(i_first, i);
            }
        }
        if (n_tokens == 0) {
            continue;
        }

        const uint32_t n_writes = (n_tokens + ratio - 1)/ratio;
        if (write_counts[seq_id] < n_writes) {
            if (write_counts[seq_id] + 1 != n_writes || i_first >= ubatch.n_tokens) {
                throw std::runtime_error("Qwen pooled indexer positions are not contiguous");
            }
            const int64_t cache_off = qwen_idx_stream_offset(n_stream, seq_id, kv_size);
            plan.state_write_idxs.push_back(cache_off + kv_size - 1);
            const int32_t src = (int32_t) (state_rows + i_first);
            for (uint32_t j = 0; j < ratio; ++j) {
                plan.state_read_idxs.push_back(src);
            }
        }
    }

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) { return a.dst < b.dst; });
    for (const auto & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }

    plan.n_kv = reserve ? (int64_t) n_rows : std::max<int64_t>(256, GGML_PAD(plan.n_kv, 256));
    if (plan.n_kv > n_rows) {
        plan.n_kv = n_rows;
    }

    return plan;
}

// the pooled cache is addressed by block row, not by slot, so every stream reads from its own base
static llama_kv_cache::slot_info_vec_t qwen_idx_build_comp_sinfos(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_stream) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (const auto & ubatch : ubatches) {
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = n_stream;
        sinfo.s1 = 0;
        sinfo.resize(ubatch.n_seqs_unq);

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const uint32_t stream = (uint32_t) ubatch.seq_id_unq[s];
            if (stream >= n_stream) {
                throw std::runtime_error("Qwen pooled indexer stream out of range");
            }
            sinfo.s0 = std::min(sinfo.s0, stream);
            sinfo.s1 = std::max(sinfo.s1, stream);
            sinfo.strm[s] = stream;
            sinfo.idxs[s].resize(1, 0);
        }

        if (sinfo.s1 - sinfo.s0 + 1 != ubatch.n_seqs_unq) {
            throw std::runtime_error("Qwen pooled indexer streams are not contiguous");
        }
        sinfos.push_back(std::move(sinfo));
    }
    return sinfos;
}

static std::vector<llama_memory_hybrid_idx_context::idx_pool_plan> qwen_idx_build_pool_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        uint32_t kv_size,
        uint32_t n_rows,
        uint32_t n_stream) {
    std::vector<llama_memory_hybrid_idx_context::idx_pool_plan> plans;
    plans.reserve(ubatches.size());
    for (const auto & ubatch : ubatches) {
        plans.push_back(qwen_idx_build_pool_plan(ubatch, ratio, kv_size, n_rows, n_stream, false));
    }
    return plans;
}

static std::vector<uint32_t> qwen_idx_pool_ns(const std::vector<llama_ubatch> & ubatches) {
    std::vector<uint32_t> res;
    res.reserve(ubatches.size());
    for (const auto & ubatch : ubatches) {
        res.push_back(std::max<uint32_t>(1, ubatch.n_seqs_unq));
    }
    return res;
}

static void qwen_idx_state_stream_range(uint32_t n_stream, llama_seq_id seq_id, uint32_t & s0, uint32_t & ns) {
    if (seq_id < 0) {
        s0 = 0;
        ns = n_stream;
        return;
    }
    if ((uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("Qwen pooled indexer state sequence id out of range");
    }
    s0 = (uint32_t) seq_id;
    ns = 1;
}

// only the rows the attention positions can reach are worth saving
static void qwen_idx_state_write_cache(
        llama_io_write_i & io,
        const llama_kv_cache * kv,
        const llama_kv_cache * attn,
        uint32_t ratio,
        llama_seq_id seq_id) {
    const uint32_t kv_size = kv->get_size();
    const auto layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = layer_ids.size();
    uint32_t s0;
    uint32_t ns;
    qwen_idx_state_stream_range(kv->get_n_stream(), seq_id, s0, ns);

    std::vector<uint32_t> n_rows(ns);
    for (uint32_t s = 0; s < ns; ++s) {
        const llama_pos pos_max = attn->seq_pos_max((llama_seq_id) (s0 + s));
        n_rows[s] = pos_max < 0 ? 0 : (uint32_t) (pos_max + 1)/ratio;
        if (n_rows[s] > kv_size) {
            throw std::runtime_error("Qwen pooled indexer cache state row count exceeds cache size");
        }
    }

    io.write(&ns, sizeof(ns));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il : layer_ids) {
        io.write(&il, sizeof(il));
        ggml_tensor * k = kv->get_k_storage(il);
        for (uint32_t s = 0; s < ns; ++s) {
            io.write_tensor(k, (size_t) (s0 + s)*k->nb[2], (size_t) n_rows[s]*k->nb[1]);
        }
    }
}

static void qwen_idx_state_read_cache(
        llama_io_read_i & io,
        llama_kv_cache * kv,
        const llama_kv_cache * attn,
        uint32_t ratio,
        llama_seq_id seq_id) {
    uint32_t ns;
    uint32_t n_layer;
    io.read(&ns, sizeof(ns));
    io.read(&n_layer, sizeof(n_layer));
    const uint32_t kv_size = kv->get_size();

    uint32_t s0;
    uint32_t ns_expected;
    qwen_idx_state_stream_range(kv->get_n_stream(), seq_id, s0, ns_expected);
    if (ns != ns_expected || n_layer != kv->get_layer_ids().size()) {
        throw std::runtime_error("Qwen pooled indexer cache state stream/layer mismatch");
    }

    // the attention cells are restored first, so the row count is already known here
    std::vector<uint32_t> n_rows(ns);
    for (uint32_t s = 0; s < ns; ++s) {
        const llama_pos pos_max = attn->seq_pos_max((llama_seq_id) (s0 + s));
        n_rows[s] = pos_max < 0 ? 0 : (uint32_t) (pos_max + 1)/ratio;
        if (n_rows[s] > kv_size) {
            throw std::runtime_error("Qwen pooled indexer cache state row count exceeds cache size");
        }
    }

    for (uint32_t il : kv->get_layer_ids()) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != il) {
            throw std::runtime_error("Qwen pooled indexer cache state layer mismatch");
        }
        ggml_tensor * k = kv->get_k_storage(il);
        for (uint32_t s = 0; s < ns; ++s) {
            io.read_tensor(k, (size_t) (s0 + s)*k->nb[2], (size_t) n_rows[s]*k->nb[1]);
        }
    }
}

// every QSA layer of this architecture shares one compression ratio, and the cache is
// laid out for it, so a model that mixed ratios would need one cache per ratio
static uint32_t qwen_idx_ratio(const llama_model & model) {
    uint32_t ratio = 0;

    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        const uint32_t r = model.hparams.dsv4_compress_ratios[il];
        if (r == 0) {
            continue;
        }
        if (ratio == 0) {
            ratio = r;
        } else if (ratio != r) {
            throw std::runtime_error("the pooled Qwen indexer needs a single compression ratio");
        }
    }

    return ratio == 0 ? 4 : ratio;
}

//
// llama_memory_hybrid_idx
//

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    idx_pooled(filter_idx != nullptr && qwen_idx_pooled_enabled()),
    idx_norm_rope(idx_pooled && qwen_idx_norm_rope_enabled()),
    idx_ratio(qwen_idx_ratio(model)),
    idx_n_seq_max(n_seq_max),
    idx_raw_type(type_k),
    idx_n_rows(filter_idx != nullptr && qwen_idx_pooled_enabled() ?
            qwen_idx_comp_rows(kv_size, idx_ratio, n_pad) : kv_size),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        if (idx_pooled && unified && n_seq_max > 1) {
            throw std::runtime_error("the pooled Qwen indexer does not support a unified KV cache with multiple sequences; "
                    "run without --kv-unified");
        }

        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        // [TAG_QSA_POOLED] one row per completed block, plus a padded tail: a step that
        // completes no block still writes, and its scratch row must miss every live block
        const uint32_t idx_size = idx_pooled ? idx_n_rows + n_pad : kv_size;

        // f32 pooled keys reproduce the per-token path exactly, because the members that
        // reach the sum are the same f16-derived values in both.
        const ggml_type idx_type = idx_pooled ? GGML_TYPE_F32 : type_k;

        LLAMA_LOG_INFO("%s: creating indexer %scache, size = %u cells, type = %s\n", __func__,
                idx_norm_rope ? "norm+RoPE pooled K " : idx_pooled ? "pooled K " : "K", idx_size, ggml_type_name(idx_type));

        return new llama_kv_cache(
            model, hparams_idx, idx_type, type_v, v_trans, offload, idx_pooled ? false : unified,
            idx_size, n_seq_max, n_pad, idx_pooled ? 0 : n_swa, idx_pooled ? LLAMA_SWA_TYPE_NONE : swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()),
    // the members of the block no ubatch has completed yet: ratio rows per stream
    idx_state(!idx_pooled ? nullptr : new llama_dsv4_comp_state(
            model, offload, false, n_seq_max, idx_ratio, idx_ratio,
            model.hparams.indexer_head_size, n_rs_seq, "qwen_idx", filter_idx)) {}

llama_memory_hybrid_idx::~llama_memory_hybrid_idx() = default;

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // The raw indexer uses the attention cache's slot layout; a separate one can drift
        // from it. The pooled indexer is addressed by block row instead, so it takes no slots.
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx && !idx_pooled) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
}

void llama_memory_hybrid_idx::clear(bool data) {
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
    if (idx_state) {
        idx_state->clear(-1, data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // The pooled cache keeps whole blocks, so it can only follow a removal that ends a
    // sequence on a block boundary. Anything else is refused before a cache is touched,
    // which makes the caller reprocess instead of reading a half-pooled block.
    bool pooled_removes_existing = true;
    if (idx_pooled) {
        if (p1 >= 0) {
            return false;
        }
        if (p0 > 0) {
            if (seq_id < 0 || (uint32_t) seq_id >= idx_n_seq_max) {
                return false;
            }
            pooled_removes_existing = p0 <= get_mem_attn()->seq_pos_max(seq_id);
            if (pooled_removes_existing && p0%idx_ratio != 0) {
                return false;
            }
        }
    }

    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (mem_idx) {
        if (idx_pooled) {
            // A suffix removal is safe because a query only ever reads the blocks below its
            // own position. A range beyond pos_max removes nothing and must not throw away
            // the open block.
            if (p0 <= 0 || pooled_removes_existing) {
                idx_state->clear(seq_id, true);
            }
        } else {
            mem_idx->seq_rm(seq_id, p0, p1);
        }
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
        if (idx_state) {
            idx_state->seq_cp(seq_id_src, seq_id_dst);
        }
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
        if (idx_state) {
            for (llama_seq_id id = 0; id < (llama_seq_id) idx_n_seq_max; ++id) {
                if (id != seq_id) {
                    idx_state->clear(id, true);
                }
            }
        }
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        // the pooled keys are rotated at read time by the block's position, so a shift would
        // have to repool every block behind p0; context shift is refused instead
        GGML_ASSERT(!idx_pooled && "the pooled Qwen indexer does not support context shift");
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        GGML_ASSERT(!idx_pooled && "the pooled Qwen indexer does not support context division");
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    if (idx_state) {
        for (const auto & buft_size : idx_state->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    llama_memory_hybrid::state_write(io, seq_id, flags);

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            const uint32_t version = idx_norm_rope ? LLAMA_HYBRID_IDX_STATE_VERSION_TRANSFORMED : idx_pooled ?
                LLAMA_HYBRID_IDX_STATE_VERSION_POOLED : LLAMA_HYBRID_IDX_STATE_VERSION_RAW;

            io.write(&LLAMA_HYBRID_IDX_STATE_MAGIC, sizeof(LLAMA_HYBRID_IDX_STATE_MAGIC));
            io.write(&version, sizeof(version));
            if (idx_norm_rope) {
                GGML_ASSERT(!idx_transform.empty());
                io.write(idx_transform.data(), idx_transform.size());
            }

            if (idx_pooled) {
                qwen_idx_state_write_cache(io, mem_idx.get(), get_mem_attn(), idx_ratio, seq_id);
                idx_state->state_write(io, seq_id, flags, std::vector<uint32_t>(idx_n_seq_max, 0));
            } else {
                mem_idx->state_write(io, seq_id, flags);
            }
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                uint32_t magic;
                uint32_t version;
                io.read(&magic, sizeof(magic));
                io.read(&version, sizeof(version));

                const uint32_t version_expected = idx_norm_rope ? LLAMA_HYBRID_IDX_STATE_VERSION_TRANSFORMED : idx_pooled ?
                    LLAMA_HYBRID_IDX_STATE_VERSION_POOLED : LLAMA_HYBRID_IDX_STATE_VERSION_RAW;

                if (magic != LLAMA_HYBRID_IDX_STATE_MAGIC || version != version_expected) {
                    throw std::runtime_error("incompatible indexer cache state format");
                }

                if (idx_norm_rope) {
                    GGML_ASSERT(!idx_transform.empty());
                    std::vector<uint8_t> transform(idx_transform.size());
                    io.read(transform.data(), transform.size());
                    if (transform != idx_transform) {
                        throw std::runtime_error("incompatible Qwen indexer transform parameters");
                    }
                }
                if (idx_pooled) {
                    qwen_idx_state_read_cache(io, mem_idx.get(), get_mem_attn(), idx_ratio, seq_id);
                    idx_state->state_read(io, seq_id, flags);
                } else {
                    mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
                }
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    get_mem_recr()->seq_rm(seq_id, -1, -1);

    if (mem_idx) {
        if (idx_pooled) {
            idx_state->clear(seq_id, true);
        } else {
            mem_idx->seq_rm(seq_id, -1, -1);
        }
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

llama_dsv4_comp_state * llama_memory_hybrid_idx::get_idx_state() const {
    return idx_state.get();
}

bool llama_memory_hybrid_idx::get_idx_norm_rope() const {
    return idx_norm_rope;
}

void llama_memory_hybrid_idx::bind_idx_transform(const ggml_tensor * rope, float norm_eps) const {
    GGML_ASSERT(idx_norm_rope && rope->op == GGML_OP_ROPE);
    std::vector<uint8_t> transform(sizeof(rope->op_params) + sizeof(norm_eps));
    memcpy(transform.data(), rope->op_params, sizeof(rope->op_params));
    memcpy(transform.data() + sizeof(rope->op_params), &norm_eps, sizeof(norm_eps));
    if (!idx_transform.empty() && idx_transform != transform) {
        throw std::runtime_error("Qwen cached indexer transform parameters changed");
    }
    idx_transform = std::move(transform);
}

bool llama_memory_hybrid_idx::get_idx_pooled() const {
    return idx_pooled;
}

uint32_t llama_memory_hybrid_idx::get_idx_ratio() const {
    return idx_ratio;
}

uint32_t llama_memory_hybrid_idx::get_idx_n_seq_max() const {
    return idx_n_seq_max;
}

uint32_t llama_memory_hybrid_idx::get_idx_n_rows() const {
    return idx_n_rows;
}

ggml_type llama_memory_hybrid_idx::get_idx_raw_type() const {
    return idx_raw_type;
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * completed_pos,
        const std::vector<int64_t> * write_idxs,
        ggml_tensor * blk_meta,
        uint32_t width,
        bool allow_fast) const {
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = (n_kv + ratio - 1)/ratio;
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = (int32_t *) cell_blk->data;
    // with the norm+RoPE cache the block positions are no longer a graph input, but the scan
    // that fills them still feeds the completed-block bookkeeping below, so it keeps a buffer
    std::vector<int32_t> blk_pos_tmp;
    if (blk_pos == nullptr || blk_pos->data == nullptr) {
        blk_pos_tmp.resize(4*n_blocks*n_ns);
    }
    int32_t * dst_blk_pos = blk_pos_tmp.empty() ? (int32_t *) blk_pos->data : blk_pos_tmp.data();
    std::vector<llama_qsa_dump_block> dump_blocks;

    // the pooled indexer builds no gather, so blk_cells has no tensor to fill
    std::vector<int32_t> blk_cells_unused;
    if (blk_cells == nullptr || blk_cells->data == nullptr) {
        blk_cells_unused.resize((size_t) r*n_blocks*n_ns);
    }
    int32_t * dst_blk_cells = blk_cells_unused.empty() ?
        (int32_t *) blk_cells->data : blk_cells_unused.data();
    float   * dst_bias      = (float   *) bias->data;

    // [TAG_QSA_BLOCK_META] what the block top-k needs to select a single-query row without a
    // pass over the cells: the count of full blocks, the spare block and the visible cells no
    // full block covers. Every row starts at "not described"; the loops below promote a row
    // only once all the properties the op relies on have been checked for it.
    int32_t * dst_meta = blk_meta != nullptr && blk_meta->data != nullptr ?
        (int32_t *) blk_meta->data : nullptr;

    if (dst_meta != nullptr) {
        GGML_ASSERT(ggml_backend_buffer_is_host(blk_meta->buffer));
        GGML_ASSERT(blk_meta->ne[0] == GGML_TOP_K_BLOCK_META_N);
        std::fill(dst_meta, dst_meta + GGML_TOP_K_BLOCK_META_N*n_tps*n_ns, 0);
    }

    // the inverse map is only usable as an input if it was filled into a real tensor
    const bool have_cells = blk_cells != nullptr && blk_cells->data != nullptr;

    std::vector<int32_t> dead_cells;

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];

        // the pooled indexer keeps no per-token cells of its own, so the token layout is
        // read from the attention cache it mirrors
        const auto & cells = (idx_pooled ? get_mem_attn() : get_mem_idx())->get_cells(seq_of_stream);

        int32_t * cur_cell_blk  = dst_cell_blk  + s*n_kv;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = n_seq_present <= 1;

        // Pooled rows are addressed by pos/ratio, while the block ids below are handed out in
        // order over the blocks that are full. The two agree only while a stream holds one
        // sequence, which a per-sequence (non-unified) attention cache guarantees.
        GGML_ASSERT((!idx_pooled || one_seq) &&
                "the pooled Qwen indexer needs one sequence per stream: run without a unified KV cache");

        // a cell no block covers needs its own -inf, which a per-block bias cannot carry
        // every cache path keeps the position below the cell window, so this stays false
        bool oor = false;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            oor = false;
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j)) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                if (pb >= n_blocks) {
                    oor = true;
                    continue;
                }

                int32_t g = -1;

                for (int32_t c = grp_head[pb]; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(grp_head[pb]);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    grp_head[pb] = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j)) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        GGML_ASSERT((!blk_bias || !oor) && "qsa: cell position runs past the cell window");

        int32_t n_bid = 0;

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            for (int32_t g = grp_head[pb]; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            if (llama_qsa_dump_enabled()) {
                dump_blocks.push_back({(int32_t) (s*n_blocks + b), seq_of_stream, bid_idx[b],
                        {sec_pos[0], sec_pos[1], sec_pos[2], sec_pos[3]}});
            }
            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        dead_cells.clear();

        // cells that carry a token but sit in no full block: they all share the spare block,
        // so the block top-k cannot find them through the inverse map
        bool dead_listed = true;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            } else if (!cells.is_empty(j)) {
                if ((int64_t) dead_cells.size() < GGML_TOP_K_BLOCK_META_CELLS) {
                    dead_cells.push_back((int32_t) j);
                } else {
                    dead_listed = false;
                }
            }

            cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j];
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                // [TAG_QSA_BLOCK_META] a described row promises the op that every visible full
                // block contributes exactly `ratio` visible cells, which needs each of them to
                // end at or before the query: a block the query cuts through is not described.
                // A position shared by two cells puts more than `ratio` cells behind one block,
                // so only a stream without duplicated positions is described. qwen4exp uses
                // IMROPE, whose batches all report is_pos_2d(), text included, so that flag
                // cannot tell text from images here.
                bool    described = dst_meta != nullptr && allow_fast && have_cells &&
                    width > 0 && !ranked && one_seq && dead_listed && !dup;
                int64_t n_visible = 0;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? 1e9f : 0.0f;

                    n_visible += r;

                    if (bid_idx[b] + r - 1 > q) {
                        described = false;
                    }
                }

                if (described) {
                    int32_t * meta_row = dst_meta + i*GGML_TOP_K_BLOCK_META_N;
                    int32_t * meta_tail = meta_row + GGML_TOP_K_BLOCK_META_HEAD;
                    int32_t   n_tail = 0;

                    for (int32_t c : dead_cells) {
                        if (cells.seq_has((uint32_t) c, seq_id) && (int64_t) cells.pos_get(c) <= q) {
                            meta_tail[n_tail++] = c;
                        }
                    }

                    n_visible += n_tail;

                    // below the budget the boundary falls on the masked cells, which no block
                    // lists: that row keeps the general walk
                    if (n_visible >= (int64_t) width) {
                        meta_row[0] = 1;
                        meta_row[1] = n_bid;
                        meta_row[2] = dead_bid;
                        meta_row[3] = n_tail;
                        meta_row[4] = (int32_t) (n_kv - n_visible);
                    }
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }

    if (completed_pos) {
        GGML_ASSERT(write_idxs && completed_pos->ne[0] == (int64_t) (4*write_idxs->size()));
        std::vector<int32_t> positions(completed_pos->ne[0], 0);
        const int64_t cache_size = mem_idx->get_size();
        for (size_t w = 0; w < write_idxs->size(); ++w) {
            const int64_t row = (*write_idxs)[w]%cache_size;
            if (row >= idx_n_rows) {
                continue; // scratch never names a live block
            }
            const llama_seq_id seq = (llama_seq_id) ((*write_idxs)[w]/cache_size);
            int64_t s = 0;
            while (s < n_ns && ubatch->seq_id[s*n_tps][0] != seq) {
                ++s;
            }
            GGML_ASSERT(s < n_ns && row < n_blocks);
            for (int64_t sec = 0; sec < 4; ++sec) {
                positions[sec*write_idxs->size() + w] = dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + row];
            }
        }
        ggml_backend_tensor_set(completed_pos, positions.data(), 0, positions.size()*sizeof(int32_t));
    }

    if (llama_qsa_dump_enabled()) {
        llama_qsa_dump_set_blocks(dump_blocks);
        llama_qsa_dump_set_cell_blk(dst_cell_blk, n_kv, n_ns);
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr || mem->get_idx_pooled() ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())),
    // reservation walks ubatches this context never saw, so its plan is built on demand
    idx_pool_reserve(mem->get_idx_pooled()),
    ctx_idx_pooled(mem->get_idx_pooled() ?
        new llama_kv_cache_dsv4_comp_context(mem->get_mem_idx()) : nullptr) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)),
    ctx_idx_pooled(nullptr) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(mem->get_idx_pooled() ? qwen_idx_pool_ns(ubatches) : llama_memory_hybrid_idx_ns(sinfos_idx)),
    ctx_idx(mem->get_mem_idx() == nullptr || mem->get_idx_pooled() ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)),
    idx_pool_plans(mem->get_idx_pooled() ? qwen_idx_build_pool_plans(
            ubatches, mem->get_idx_ratio(),
            mem->get_mem_idx()->get_size(),
            mem->get_idx_n_rows(),
            mem->get_idx_n_seq_max()) : std::vector<idx_pool_plan>()),
    ctx_idx_pooled(mem->get_idx_pooled() ? new llama_kv_cache_dsv4_comp_context(
            // not moved: the order in which these two arguments are built is unspecified,
            // and the sinfos are read from the same vector
            mem->get_mem_idx(), qwen_idx_build_comp_sinfos(ubatches, mem->get_idx_n_seq_max()),
            ubatches) : nullptr) {}

llama_memory_hybrid_idx_context::~llama_memory_hybrid_idx_context() = default;

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }
    if (ctx_idx_pooled) {
        ctx_idx_pooled->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
    }

    // the open-block state has no graph of its own, so a pending seq_cp is applied here
    if (mem && mem->get_idx_pooled() && idx_pool_plans.empty() && mem->get_idx_state()) {
        auto * state = mem->get_idx_state();
        state->apply_copies(state->sc_info);
        state->sc_info = {};
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

const llama_kv_cache_dsv4_comp_context * llama_memory_hybrid_idx_context::get_idx_pooled_ctx() const {
    return ctx_idx_pooled.get();
}

const llama_dsv4_comp_state * llama_memory_hybrid_idx_context::get_idx_state() const {
    return mem ? mem->get_idx_state() : nullptr;
}

const llama_memory_hybrid_idx_context::idx_pool_plan & llama_memory_hybrid_idx_context::get_idx_pool_plan(
        const llama_ubatch & ubatch) const {
    GGML_ASSERT(mem && mem->get_idx_pooled());

    if (idx_pool_reserve) {
        idx_pool_reserve_plan = qwen_idx_build_pool_plan(
                ubatch, mem->get_idx_ratio(),
                mem->get_mem_idx()->get_size(),
                mem->get_idx_n_rows(),
                mem->get_idx_n_seq_max(), true);

        return idx_pool_reserve_plan;
    }

    GGML_ASSERT(i_cur < idx_pool_plans.size());

    return idx_pool_plans[i_cur];
}

bool llama_memory_hybrid_idx_context::get_idx_norm_rope() const {
    return mem && mem->get_idx_norm_rope();
}

void llama_memory_hybrid_idx_context::bind_idx_transform(const ggml_tensor * rope, float norm_eps) const {
    GGML_ASSERT(mem);
    mem->bind_idx_transform(rope, norm_eps);
}

bool llama_memory_hybrid_idx_context::get_idx_pooled() const {
    return mem && mem->get_idx_pooled();
}

ggml_type llama_memory_hybrid_idx_context::get_idx_raw_type() const {
    GGML_ASSERT(mem);

    return mem->get_idx_raw_type();
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias,
        ggml_tensor * completed_pos,
        ggml_tensor * blk_meta,
        uint32_t width,
        bool allow_fast) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias, completed_pos,
            completed_pos ? &get_idx_pool_plan(*ubatch).state_write_idxs : nullptr,
            blk_meta, width, allow_fast);
}

template<typename T>
static void qwen_idx_set_tensor(ggml_tensor * dst, const std::vector<T> & src) {
    if (dst == nullptr || dst->buffer == nullptr) {
        return;
    }

    GGML_ASSERT(dst->ne[0] == (int64_t) src.size());

    if (!src.empty()) {
        ggml_backend_tensor_set(dst, src.data(), 0, src.size()*sizeof(T));
    }
}

void llama_memory_hybrid_idx_context::set_input_idx_pool_plan(
        ggml_tensor * state_persist_src_idxs,
        ggml_tensor * state_persist_dst_idxs,
        ggml_tensor * state_read_idxs,
        ggml_tensor * state_write_idxs,
        const llama_ubatch * ubatch) const {
    const auto & plan = get_idx_pool_plan(*ubatch);

    qwen_idx_set_tensor(state_persist_src_idxs, plan.state_persist_src_idxs);
    qwen_idx_set_tensor(state_persist_dst_idxs, plan.state_persist_dst_idxs);
    qwen_idx_set_tensor(state_read_idxs,        plan.state_read_idxs);
    qwen_idx_set_tensor(state_write_idxs,       plan.state_write_idxs);
}
