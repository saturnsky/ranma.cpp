#include "llama-kv-cache-dsv4.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-graph-chunk.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;

static constexpr uint32_t DSV4_STATE_MAGIC         = 0x34565344; // DSV4
static constexpr uint32_t DSV4_STATE_VERSION       = 1;
static constexpr uint32_t DSV4_STATE_MODE_FULL     = 0;
static constexpr uint32_t DSV4_STATE_MODE_PARTIAL  = 1;
// ranma: or'ed into the mode, a block of tail prefill records follows it (llama_kv_cache_dsv4::tail_state_write). A
// state without it, which a cache without the tail prefill or a build before the records wrote, is read as before.
static constexpr uint32_t DSV4_STATE_MODE_TAIL     = 0x100;
static constexpr uint32_t DSV4_STATE_TAIL_VER      = 1;
static constexpr uint32_t DSV4_K_CACHE_STATE_VER   = 2;
static constexpr uint32_t DSV4_COMP_STATE_VER      = 1;

// ranma: the tail prefill is on (always for DeepSeek-V4.1, never for the other models of this cache) or its diagnostic
// is asked for (RANMA_DSV41_TAIL_DIAG); only then the cache keeps its records. Its per-call log lines are written at
// info level only with the diagnostic, at debug level otherwise
static bool dsv4_env_on(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && atoi(v) != 0;
}

static bool dsv4_tail_diag() {
    return dsv4_env_on("RANMA_DSV41_TAIL_DIAG");
}

static bool dsv4_exact_tail_on(bool v41) {
    return v41;
}

static bool dsv4_tail_requested(bool v41) {
    return dsv4_exact_tail_on(v41) || dsv4_env_on("RANMA_DSV41_TAIL_DIAG");
}

static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static void dsv4_clear_tensor_stream(ggml_tensor * tensor, uint32_t stream) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[3] == 1);
    GGML_ASSERT(stream < (uint32_t) tensor->ne[2]);

    const size_t stream_size = tensor->nb[2];
    ggml_backend_tensor_memset(tensor, 0, stream*stream_size, stream_size);
}

static uint32_t dsv4_state_n_used_k_rows(llama_pos pos_max, uint32_t ratio, uint32_t kv_size) {
    if (pos_max < 0) {
        return 0;
    }

    const uint64_t n_rows = ((uint64_t) pos_max + 1)/ratio;

    return (uint32_t) std::min<uint64_t>(kv_size, n_rows);
}

static int64_t dsv4_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("DSV4 sequence id out of stream range");
    }

    return (int64_t) seq_id*size;
}

static bool dsv4_ubatch_has_coupled(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static bool dsv4_token_has_seq(const llama_ubatch & ubatch, uint32_t i, llama_seq_id seq_id) {
    for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
        if (ubatch.seq_id[i][s] == seq_id) {
            return true;
        }
    }

    return false;
}

static llama_ubatch dsv4_build_raw_write_ubatch(const llama_ubatch & ubatch) {
    if (!dsv4_ubatch_has_coupled(ubatch)) {
        return ubatch;
    }
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        throw std::runtime_error("DSV4 coupled embedding ubatches are not supported");
    }

    std::vector<uint32_t> counts(ubatch.n_seqs_unq, 0);
    uint32_t n_tokens = 0;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                ++counts[s];
                ++n_tokens;
            }
        }
    }

    if (n_tokens == 0) {
        return ubatch;
    }

    const uint32_t n_seq_tokens = counts[0];
    for (uint32_t s = 1; s < counts.size(); ++s) {
        if (counts[s] != n_seq_tokens) {
            throw std::runtime_error("DSV4 coupled raw writes require equal sequence lengths");
        }
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize((size_t) n_tokens*ubatch.n_pos);
    data->n_seq_id.reserve(n_tokens);
    data->seq_id.reserve(n_tokens);
    data->seq_id_data.reserve(n_tokens);
    data->seq_id_unq.assign(ubatch.seq_id_unq, ubatch.seq_id_unq + ubatch.n_seqs_unq);
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->output.assign(n_tokens, 0);
    if (ubatch.token) {
        data->token.reserve(n_tokens);
    }

    for (uint32_t s = 0; s < data->seq_id_unq.size(); ++s) {
        data->seq_idx[data->seq_id_unq[s]] = s;
    }

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (!dsv4_token_has_seq(ubatch, i, seq_id)) {
                continue;
            }

            const uint32_t dst = data->n_seq_id.size();
            if (ubatch.token) {
                data->token.push_back(ubatch.token[i]);
            }
            for (uint32_t p = 0; p < ubatch.n_pos; ++p) {
                data->pos[(size_t) p*n_tokens + dst] = ubatch.pos[(size_t) p*ubatch.n_tokens + i];
            }
            data->n_seq_id.push_back(1);
            data->seq_id_data.push_back(seq_id);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->seq_id.push_back(&data->seq_id_data[i]);
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ ubatch.n_seqs_unq,
        /*.n_seqs_unq   =*/ ubatch.n_seqs_unq,
        /*.n_pos        =*/ ubatch.n_pos,
        /*.token        =*/ data->token.empty() ? nullptr : data->token.data(),
        /*.embd         =*/ nullptr,
        /*.embd_state   =*/ nullptr,
        /*.pos          =*/ data->pos.data(),
        /*.n_seq_id     =*/ data->n_seq_id.data(),
        /*.seq_id       =*/ data->seq_id.data(),
        /*.seq_id_unq   =*/ data->seq_id_unq.data(),
        /*.seq_idx      =*/ data->seq_idx.data(),
        /*.output       =*/ data->output.data(),
        /*.type         =*/ nullptr,
        /*.decision_order =*/ nullptr,
        /*.data         =*/ data,
    };

    return res;
}

static std::vector<llama_ubatch> dsv4_build_raw_write_ubatches(const std::vector<llama_ubatch> & ubatches) {
    std::vector<llama_ubatch> res;
    res.reserve(ubatches.size());
    for (const llama_ubatch & ubatch : ubatches) {
        res.push_back(dsv4_build_raw_write_ubatch(ubatch));
    }
    return res;
}

static bool dsv4_batch_has_coupled(const llama_batch & batch) {
    if (!batch.n_seq_id) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static int64_t dsv4_comp_graph_n_stream(const llama_ubatch & ubatch, uint32_t n_stream) {
    // Coupled sequence sets must stay in one graph stream because their
    // compressed state is shared. Independent per-seq state can fan out.
    if (n_stream <= 1 || ubatch.n_seqs_unq <= 1 || dsv4_ubatch_has_coupled(ubatch)) {
        return 1;
    }

    return ubatch.n_seqs_unq;
}

static void dsv4_state_src_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t     & s0,
        uint32_t     & ns) {
    if (seq_id >= 0 && n_stream > 1) {
        if ((uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = (uint32_t) seq_id;
        ns = 1;
        return;
    }

    s0 = 0;
    ns = seq_id >= 0 ? 1 : n_stream;
}

static void dsv4_state_dst_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t       ns,
        uint32_t     & s0) {
    if (seq_id >= 0) {
        if (ns != 1) {
            throw std::runtime_error("DSV4 sequence state stream count mismatch");
        }
        if (n_stream > 1 && (uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = n_stream > 1 ? (uint32_t) seq_id : 0;
        return;
    }

    if (ns != n_stream) {
        throw std::runtime_error("DSV4 full state stream count mismatch");
    }

    s0 = 0;
}

static void dsv4_state_write_tensor_streams(
        llama_io_write_i & io,
        ggml_tensor      * tensor,
        uint32_t           tensor_rows,
        uint32_t           n_rows,
        uint32_t           s0,
        uint32_t           ns,
        const std::vector<uint32_t> * stream_ids = nullptr) {
    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    io.write(&type_i,   sizeof(type_i));
    io.write(&ne0,      sizeof(ne0));
    io.write(&rows,     sizeof(rows));
    io.write(&row_size, sizeof(row_size));

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    if (stream_ids && stream_ids->size() != ns) {
        throw std::runtime_error("DSV4 state tensor stream map size mismatch");
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t stream = stream_ids ? (*stream_ids)[s] : s0 + s;
        if ((int64_t) stream >= tensor->ne[2]) {
            throw std::runtime_error("DSV4 state tensor stream out of range");
        }
        const size_t offset = (size_t) stream*stream_stride;
        io.write_tensor(tensor, offset, size);
    }
}

static void dsv4_state_read_tensor_streams(
        llama_io_read_i & io,
        ggml_tensor     * tensor,
        uint32_t          tensor_rows,
        uint32_t          n_rows,
        uint32_t          s0,
        uint32_t          ns) {
    int32_t  type_i_ref;
    uint64_t ne0_ref;
    uint64_t rows_ref;
    uint64_t row_size_ref;

    io.read(&type_i_ref,   sizeof(type_i_ref));
    io.read(&ne0_ref,      sizeof(ne0_ref));
    io.read(&rows_ref,     sizeof(rows_ref));
    io.read(&row_size_ref, sizeof(row_size_ref));

    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (type_i != type_i_ref || ne0 != ne0_ref || rows != rows_ref || row_size != row_size_ref) {
        throw std::runtime_error("DSV4 state tensor metadata mismatch");
    }
    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const size_t offset = (size_t) (s0 + s)*stream_stride;
        io.read_tensor(tensor, offset, size);
    }
}

static void dsv4_state_write_k_cache(
        llama_io_write_i    & io,
        const llama_kv_cache * kv,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags,
        uint32_t              n_rows) {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(kv->get_n_stream(), seq_id, s0, ns);

    const uint32_t version = DSV4_K_CACHE_STATE_VER;
    const uint32_t kv_size = kv->get_size();
    const auto layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = layer_ids.size();

    if (n_rows > kv_size) {
        throw std::runtime_error("DSV4 K-cache state row count exceeds cache size");
    }

    io.write(&version, sizeof(version));
    io.write(&n_rows,  sizeof(n_rows));
    io.write(&ns,      sizeof(ns));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il : layer_ids) {
        io.write(&il, sizeof(il));
        dsv4_state_write_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows, s0, ns);
    }
}

static void dsv4_state_read_k_cache(
        llama_io_read_i  & io,
        llama_kv_cache   * kv,
        llama_seq_id       seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t n_rows_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,     sizeof(version));
    io.read(&n_rows_ref,  sizeof(n_rows_ref));
    io.read(&ns,          sizeof(ns));
    io.read(&n_layer_ref, sizeof(n_layer_ref));

    if (version != 1 && version != DSV4_K_CACHE_STATE_VER) {
        throw std::runtime_error("DSV4 K-cache state version mismatch");
    }

    const uint32_t kv_size = kv->get_size();
    if (version == 1 && n_rows_ref != kv_size) {
        LLAMA_LOG_INFO("kv size ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }
    if (n_rows_ref > kv_size) {
        LLAMA_LOG_INFO("kv rows ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(kv->get_n_stream(), seq_id, ns, s0);

    const auto layer_ids = kv->get_layer_ids();
    if (n_layer_ref != layer_ids.size()) {
        throw std::runtime_error("DSV4 K-cache layer count mismatch");
    }

    for (uint32_t il : layer_ids) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != il) {
            throw std::runtime_error("DSV4 K-cache layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows_ref, s0, ns);
    }
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

// When no HCA block completes in a ubatch the compressor result is discarded,
// so the compress + cache-write subgraph can be dropped entirely instead of
// being run on a dummy block. Only HCA (ratio 128) is affected; the CSA/LID dummy
// block (ratio 4) is kept because its period is too short for graph reuse to pay.

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);

    // n_stream is the persistent cache/state layout; plan.n_stream is the
    // graph view for this ubatch and can be a subset of those streams.
    if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
        throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
    }

    const int64_t state_rows = (int64_t) state_size*n_stream;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
    // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
    // the "previous-window" gather indices for every block, followed by the
    // "current-window" indices for every block. Collect them separately here and
    // append cur after prev once the loop has visited all completed blocks
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;
    std::map<llama_seq_id, uint32_t> state_write_counts;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            curr_token_idx_map[std::make_pair(ubatch.seq_id[i][s], ubatch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return (int32_t) (state_rows + ubatch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
        return (int32_t) (stream_off + pos%state_size);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i];

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const int32_t state_idx = (int32_t) (stream_off + pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);

            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_pos.push_back((int32_t) source_start);
            ++state_write_counts[seq_id];

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    // The overlap compressor reads a fixed window per block, so every sequence has to write the
    // same number of blocks or the graph shape moves between ubatches. For DeepSeek-V4 this is
    // exactly the CSA and indexer streams.
    if (overlap && !plan.state_pos.empty()) {
        assert(kv_size > 0);

        // Pad each stream to the reserve plan's block count.
        const auto append_dummy_block = [&](llama_seq_id seq_id, uint32_t i) {
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
            const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

            plan.state_write_idxs.push_back(cache_off + kv_size - 1);
            plan.state_write_pos .push_back(0);

            if (overlap) {
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(source_idx);
                    overlap_cur_reads .push_back(source_idx);
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(source_idx);
                }
            }
        };

        if (dsv4_ubatch_has_coupled(ubatch)) {
            if (plan.state_write_idxs.empty()) {
                uint32_t i = 0;
                while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(ubatch.seq_id[i][0], i);
            }
        } else {
            const uint32_t n_blocks = (std::max<uint32_t>(1, ubatch.n_seq_tokens) + ratio - 1)/ratio;

            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id_unq[s];
                const uint32_t n_writes = state_write_counts[seq_id];
                if (n_writes >= n_blocks) {
                    continue;
                }
                if (n_writes + 1 != n_blocks) {
                    throw std::runtime_error("DSV4 CSA sequence positions are not contiguous");
                }

                uint32_t i = 0;
                while (i < ubatch.n_tokens && (ubatch.pos[i] < 0 || !dsv4_token_has_seq(ubatch, i, seq_id))) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(seq_id, i);
            }
        }
    }

    // A non-overlap stream completes a block only every `ratio` tokens, so a decode step often
    // completes none. For DeepSeek-V4 this is the HCA stream; V4.1 pools without overlap on both
    // of its streams and reaches this at ratio 2 on every other token.
    if (!overlap && !plan.state_pos.empty() && plan.state_write_idxs.empty() &&
            ratio != DSV4_HCA_RATIO) {
        assert(kv_size > 0);
        // the last slot must not be live, or the dummy write would corrupt it;
        // a full stream implies a completed block, which implies real writes
        assert(plan.n_kv < (int64_t) kv_size);

        // Keep the compress/write ops in the graph when no HCA block completes
        // in this ubatch. The dummy block writes to the last cache slot and is
        // masked out.
        uint32_t i = 0;
        while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
            ++i;
        }
        assert(i < ubatch.n_tokens);

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
        const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

        plan.state_write_idxs.push_back(cache_off + kv_size - 1);
        plan.state_write_pos .push_back(0);

        for (uint32_t j = 0; j < ratio; ++j) {
            plan.state_read_idxs.push_back(source_idx);
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    // Keep the mask (and with it the compressed-attention branch) present even
    // before the first block is visible, so the graph topology never changes.
    // Padded slots are masked out; comp cache buffers are zero-initialized.
    plan.n_kv = std::max<int64_t>(GGML_PAD(plan.n_kv, 256u), 256);

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) {
                return a.dst < b.dst;
            });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }

    if (n_rs_seq > 0) {
        // Emit restore/snapshot entries for all layout streams so that the
        // graph tensor sizes do not depend on the ubatch's sequence count.
        // Streams not present in the ubatch get no-op entries.
        for (uint32_t stream = 0; stream < n_stream; ++stream) {
            llama_seq_id seq_id = -1;
            if (n_stream == 1) {
                // a unified stream serves any single sequence
                seq_id = ubatch.n_seqs_unq > 0 ? ubatch.seq_id_unq[0] : -1;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    if (ubatch.seq_id_unq[s] == (llama_seq_id) stream) {
                        seq_id = ubatch.seq_id_unq[s];
                        break;
                    }
                }
            }

            const int64_t stream_off = (int64_t) stream*state_size;
            const uint32_t rollback = seq_id >= 0 && (uint32_t) seq_id < rs_idx.size() ? rs_idx[seq_id] : 0;
            // Keep the restore graph fixed-width when no rollback is pending.
            const int64_t src_plane = rollback > 0 && rollback <= n_rs_seq ? (int64_t) rollback*state_rows : 0;
            for (uint32_t r = 0; r < state_size; ++r) {
                plan.state_restore_src_idxs.push_back((int32_t) (src_plane + stream_off + r));
                plan.state_restore_dst_idxs.push_back((int32_t) (stream_off + r));
            }

            std::vector<uint32_t> token_idxs;
            token_idxs.reserve(ubatch.n_tokens);
            if (seq_id >= 0) {
                for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                    if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                        token_idxs.push_back(i);
                    }
                }
            }

            const uint32_t n_seq_tokens = (uint32_t) token_idxs.size();
            const int64_t scratch_off = (int64_t) state_rows*(1 + n_rs_seq);
            for (uint32_t d = 1; d <= n_rs_seq; ++d) {
                const int64_t dst_plane = (int64_t) d*state_rows;
                const uint32_t prefix = d <= n_seq_tokens ? n_seq_tokens - d : 0;

                for (uint32_t r = 0; r < state_size; ++r) {
                    int32_t src = (int32_t) (stream_off + r);

                    for (uint32_t j = 0; j < prefix; ++j) {
                        const uint32_t i_tok = token_idxs[j];
                        if (ubatch.pos[i_tok] >= 0 && (uint32_t) (ubatch.pos[i_tok]%state_size) == r) {
                            src = (int32_t) (scratch_off + i_tok);
                        }
                    }

                    if (n_seq_tokens == 0) {
                        // no-op: copy the snapshot plane onto itself
                        src = (int32_t) (dst_plane + stream_off + r);
                    }

                    plan.state_snapshot_src_idxs.push_back(src);
                    plan.state_snapshot_dst_idxs.push_back((int32_t) (dst_plane + stream_off + r));
                }
            }
        }
    }

    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    if (debug) {
        LLAMA_LOG_DEBUG("%s: ratio=%u, n_tokens=%u, n_seqs_unq=%u, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, ratio, ubatch.n_tokens, ubatch.n_seqs_unq,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            const uint32_t rollback = seq_id >= 0 && (uint32_t) seq_id < rs_idx.size() ? rs_idx[seq_id] : 0;
            LLAMA_LOG_DEBUG("%s:   seq %d pos [%d, %d] rollback=%u\n", __func__, seq_id,
                    ubatch.pos[0], ubatch.pos[ubatch.n_tokens - 1], rollback);
        }
    }

    return plan;
}

static std::vector<llama_kv_cache_dsv4_context::comp_plan> dsv4_build_comp_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx) {
    std::vector<llama_kv_cache_dsv4_context::comp_plan> plans;
    plans.reserve(ubatches.size());

    // the first ubatch touching a seq consumes its rollback restore
    std::vector<uint32_t> rs(rs_idx);
    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(ubatch, ratio, overlap, state_size, kv_size, n_stream, n_rs_seq, rs));

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            if (seq_id >= 0 && (size_t) seq_id < rs.size()) {
                rs[seq_id] = 0;
            }
        }
    }

    return plans;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_comp_sinfos(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_stream) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
            throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
        }

        const uint32_t ns = (uint32_t) dsv4_comp_graph_n_stream(ubatch, n_stream);
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = n_stream > 1 ? LLAMA_MAX_SEQ : 0;
        sinfo.s1 = 0;
        sinfo.resize(ns);

        for (uint32_t s = 0; s < ns; ++s) {
            const llama_seq_id seq_id = n_stream > 1 ? ubatch.seq_id_unq[s] : 0;
            const uint32_t strm = (uint32_t) dsv4_stream_offset(n_stream, seq_id, 1);

            sinfo.s0 = std::min(sinfo.s0, strm);
            sinfo.s1 = std::max(sinfo.s1, strm);
            sinfo.strm[s] = strm;
            sinfo.idxs[s].resize(1, 0);
        }

        if (n_stream > 1 && sinfo.s1 - sinfo.s0 + 1 != ns) {
            throw std::runtime_error("DSV4 compressed streams are not contiguous in ubatch");
        }

        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_raw_read_sinfos(
        const llama_kv_cache::slot_info_vec_t & sinfos_write,
        const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (size_t i = 0; i < ubatches.size(); ++i) {
        const llama_ubatch & ubatch = ubatches[i];
        const auto & sinfo_write = sinfos_write[i];

        if (!dsv4_ubatch_has_coupled(ubatch)) {
            sinfos.push_back(sinfo_write);
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[0][0];
        uint32_t i_stream = 0;
        for (; i_stream < sinfo_write.n_stream(); ++i_stream) {
            if (sinfo_write.strm[i_stream] == seq_id) {
                break;
            }
        }
        if (i_stream == sinfo_write.n_stream()) {
            throw std::runtime_error("DSV4 raw write stream not found for coupled read");
        }

        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = sinfo_write.strm[i_stream];
        sinfo.s1 = sinfo_write.strm[i_stream];
        sinfo.resize(1);
        sinfo.strm[0] = sinfo_write.strm[i_stream];
        sinfo.idxs[0] = sinfo_write.idxs[i_stream];
        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_reserve_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);
    plan.n_kv = kv_size;

    if (ubatch.n_tokens == 0) {
        return plan;
    }

    // worst case over every seq split: sum of per-seq ceil(tokens/ratio) is at
    // most floor(n_tokens/ratio) + n_seqs
    const uint32_t n_seqs = std::max<uint32_t>(1, ubatch.n_seqs);
    const size_t n_blocks = (size_t) ubatch.n_tokens/ratio + n_seqs;

    const uint64_t state_rows = (uint64_t) state_size*n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>(ubatch.n_tokens, state_rows);
    const size_t n_restore = n_rs_seq > 0 ? (size_t) state_size*n_stream : 0;
    const size_t n_snapshot = (size_t) n_rs_seq*state_size*n_stream;

    plan.state_pos .resize(ubatch.n_tokens);
    plan.state_persist_src_idxs.resize(n_persist);
    plan.state_persist_dst_idxs.resize(n_persist);
    plan.state_restore_src_idxs.resize(n_restore);
    plan.state_restore_dst_idxs.resize(n_restore);
    plan.state_snapshot_src_idxs.resize(n_snapshot);
    plan.state_snapshot_dst_idxs.resize(n_snapshot);
    plan.state_read_idxs .resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_pos .resize(n_blocks);

    return plan;
}

static void dsv4_make_k_only(llama_hparams & hparams) {
    // llama_kv_cache uses hparams.is_mla() to allocate K-only storage.
    hparams.n_embd_head_k_mla_impl = hparams.n_embd_head_k();
    hparams.n_embd_head_v_mla_impl = hparams.n_embd_head_k();
}

//
// llama_dsv4_comp_state
//

llama_dsv4_comp_state::llama_dsv4_comp_state(
        const llama_model & model,
                bool        offload,
                bool        unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
                bool        overlap,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            uint32_t        n_rs_seq,
        const char    * name,
        const llama_memory_i::layer_filter_cb & filter) :
    ratio(ratio),
    overlap(overlap),
    state_size(state_size),
    n_embd_state(n_embd_state),
    n_stream(unified ? 1 : n_seq_max),
    n_rs_seq(n_rs_seq) {
    const llama_hparams & hparams = model.hparams;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*hparams.n_layer()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DSV4 compressor state");
        }

        const uint32_t n_planes = n_stream*(1 + n_rs_seq);
        ggml_tensor * kv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);

        ggml_format_name(kv,    "dsv4_%s_state_kv_l%d",    name, il);
        ggml_format_name(score, "dsv4_%s_state_score_l%d", name, il);

        std::vector<ggml_tensor *> kv_stream;
        std::vector<ggml_tensor *> score_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            kv_stream.push_back(ggml_view_2d(ctx, kv, n_embd_state, state_size, kv->nb[1], s*kv->nb[2]));
            score_stream.push_back(ggml_view_2d(ctx, score, n_embd_state, state_size, score->nb[1], s*score->nb[2]));
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, kv, score, std::move(kv_stream), std::move(score_stream) });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for DSV4 compressor state");
        }

        ggml_backend_buffer_clear(buf, 0);

        LLAMA_LOG_INFO("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), name, ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n",
            __func__, name, ratio, state_size, n_embd_state, n_stream, n_rs_seq, layers.size(), total_size()/1024.0/1024.0);
}

void llama_dsv4_comp_state::clear(llama_seq_id seq_id, bool data) {
    if (!data) {
        return;
    }

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_stream);

        if (n_stream == 1) {
            // every plane belongs to this sequence: one clear per buffer instead of one
            // synchronous memset per layer, plane and tensor
            for (auto & [_, buf] : ctxs_bufs) {
                ggml_backend_buffer_clear(buf.get(), 0);
            }
            return;
        }

        for (const auto & layer : layers) {
            for (uint32_t d = 0; d <= n_rs_seq; ++d) {
                const uint32_t stream = d*n_stream + (uint32_t) seq_id;
                dsv4_clear_tensor_stream(layer.kv,    stream);
                dsv4_clear_tensor_stream(layer.score, stream);
            }
        }
        return;
    }

    for (auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

void llama_dsv4_comp_state::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < n_stream);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_stream);

    if (seq_id_src == seq_id_dst) {
        return;
    }

    clear(seq_id_dst, true);

    sc_info.ssrc.push_back((uint32_t) seq_id_src);
    sc_info.sdst.push_back((uint32_t) seq_id_dst);
}

void llama_dsv4_comp_state::apply_copies(const stream_copy_info & sc_info) const {
    for (size_t i = 0; i < sc_info.ssrc.size(); ++i) {
        const uint32_t ssrc = sc_info.ssrc[i];
        const uint32_t sdst = sc_info.sdst[i];

        for (const auto & layer : layers) {
            ggml_backend_tensor_copy(layer.kv_stream[ssrc], layer.kv_stream[sdst]);
            ggml_backend_tensor_copy(layer.score_stream[ssrc], layer.score_stream[sdst]);
        }
    }
}

uint32_t llama_dsv4_comp_state::get_ratio() const {
    return ratio;
}

bool llama_dsv4_comp_state::get_overlap() const {
    return overlap;
}

uint32_t llama_dsv4_comp_state::get_state_size() const {
    return state_size;
}

uint32_t llama_dsv4_comp_state::get_n_stream() const {
    return n_stream;
}

uint32_t llama_dsv4_comp_state::get_n_rs_seq() const {
    return n_rs_seq;
}

uint32_t llama_dsv4_comp_state::get_n_rows() const {
    return state_size*n_stream;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_dsv4_comp_state::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ret[buft] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

void llama_dsv4_comp_state::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        const std::vector<uint32_t> & rs_idx) const {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(n_stream, seq_id, s0, ns);

    std::vector<uint32_t> stream_ids(ns);
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t seq = seq_id >= 0 ? (uint32_t) seq_id : s0 + s;
        if (seq >= rs_idx.size() || rs_idx[seq] > n_rs_seq) {
            throw std::runtime_error("DSV4 recurrent state rollback index out of range");
        }
        stream_ids[s] = rs_idx[seq]*n_stream + s0 + s;
    }

    const uint32_t version      = DSV4_COMP_STATE_VER;
    const uint32_t n_layer      = layers.size();

    io.write(&version,      sizeof(version));
    io.write(&ratio,        sizeof(ratio));
    io.write(&state_size,   sizeof(state_size));
    io.write(&n_embd_state, sizeof(n_embd_state));
    io.write(&ns,           sizeof(ns));
    io.write(&n_layer,      sizeof(n_layer));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));

        dsv4_state_write_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns, &stream_ids);
        dsv4_state_write_tensor_streams(io, layer.score, state_size, state_size, s0, ns, &stream_ids);
    }
}

void llama_dsv4_comp_state::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t ratio_ref;
    uint32_t state_size_ref;
    uint32_t n_embd_state_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,          sizeof(version));
    io.read(&ratio_ref,        sizeof(ratio_ref));
    io.read(&state_size_ref,   sizeof(state_size_ref));
    io.read(&n_embd_state_ref, sizeof(n_embd_state_ref));
    io.read(&ns,               sizeof(ns));
    io.read(&n_layer_ref,      sizeof(n_layer_ref));

    if (version != DSV4_COMP_STATE_VER) {
        throw std::runtime_error("DSV4 compressor state version mismatch");
    }
    if (ratio_ref != ratio || state_size_ref != state_size || n_embd_state_ref != n_embd_state) {
        throw std::runtime_error("DSV4 compressor state metadata mismatch");
    }
    if (n_layer_ref != layers.size()) {
        throw std::runtime_error("DSV4 compressor state layer count mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(n_stream, seq_id, ns, s0);

    for (const auto & layer : layers) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != layer.il) {
            throw std::runtime_error("DSV4 compressor state layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns);
        dsv4_state_read_tensor_streams(io, layer.score, state_size, state_size, s0, ns);
    }
}

ggml_tensor * llama_dsv4_comp_state::get_kv_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].kv;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_score_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].score;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_kv(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_kv_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::get_score(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_score_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::cpy_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_kv_all(ctx, il), cur, idxs);
}

ggml_tensor * llama_dsv4_comp_state::cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_score_all(ctx, il), cur, idxs);
}

size_t llama_dsv4_comp_state::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

//
// llama_kv_cache_dsv4
//

// env: LLAMA_DSV4_KALL_VIEW (default 1)
// 1 = raw and compressed K of a layer live in one allocation and attention reads them as a single
//     view; 0 = the previous layout, attention concatenates the two caches every token
static bool dsv4_kall_view_enabled() {
    static const bool res = []() {
        const char * env = getenv("LLAMA_DSV4_KALL_VIEW");
        return env == nullptr || atoi(env) != 0;
    }();

    return res;
}

// With the joint layout, attention reads only the raw cells [lo, hi) that a token of the ubatch can see (the
// window behind the smallest position of the ubatch and the ubatch itself, lo and hi on 256 cells); the cells
// outside are masked for every token anyway, so the result is the same while the masks, the sparse lists and
// the attention shrink to the window instead of the whole raw cache (pad(n_swa + n_ubatch) cells).
//
// A ubatch of at least 64 tokens (prefill) reads the narrow window [lo, hi) and, when hi is before the end of
// the raw cache, concatenates the compressed cells behind it (a copy of hi - lo + n_comp rows per layer, small next
// to the attention of that many tokens). A smaller ubatch (decode, a draft batch) reads the whole raw cache as before,
// so its attention width and result stay the same.
static constexpr uint32_t DSV4_RAW_NKV_MIN_TOKENS = 64;

void llama_kv_cache_dsv4::init_kall(
        const llama_model & model,
                ggml_type   type_k,
                     bool   offload,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch) {
    raw_nkv_min_tokens = DSV4_RAW_NKV_MIN_TOKENS;

    if (!dsv4_kall_view_enabled()) {
        LLAMA_LOG_INFO("%s: DSV4 joint raw+compressed K layout disabled (LLAMA_DSV4_KALL_VIEW=0)\n", __func__);
        return;
    }

    // the joint layout places the raw cells of a layer directly in front of its compressed cells,
    // which only has a single well-defined stride with one stream
    if (n_seq_max != 1) {
        LLAMA_LOG_INFO("%s: DSV4 joint raw+compressed K layout disabled (n_seq_max = %u > 1)\n", __func__, n_seq_max);
        return;
    }

    if (model.hparams.no_alloc) {
        return;
    }

    // same sizing as llama_kv_cache_iswa - the DSV4 raw cache is never unified
    const uint32_t size_swa = swa_full ? kv_size : GGML_PAD(std::min(kv_size, hparams_raw.n_swa + n_ubatch), 256u);

    const uint32_t size_csa = GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u);
    const uint32_t size_hca = GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u);

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    const uint32_t n_layer = hparams_raw.n_layer_all;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(4u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    struct kall_plan {
        int32_t        il;
        uint32_t       n_cmp;
        ggml_context * ctx;
    };

    std::vector<kall_plan> plan;

    for (uint32_t il = 0; il < hparams_raw.n_layer(); ++il) {
        const uint32_t ratio = model.hparams.dsv4_compress_ratios[il];

        if (ratio != DSV4_CSA_RATIO && ratio != DSV4_HCA_RATIO) {
            continue;
        }

        if (!hparams_raw.has_kv(il)) {
            continue;
        }

        // the raw K of a DSV4 layer has to live in the SWA half of the raw cache
        if (!hparams_raw.is_swa(il)) {
            LLAMA_LOG_WARN("%s: DSV4 joint raw+compressed K layout disabled (layer %u is not SWA)\n", __func__, il);
            kall_layers.clear();
            ctx_map.clear();
            return;
        }

        const uint32_t n_embd_k_gqa = hparams_raw.n_embd_k_gqa(il);

        const llama_hparams & hparams_cmp = ratio == DSV4_CSA_RATIO ? hparams_csa : hparams_hca;

        // the concatenated layout requires identical rows in both caches, and get_k_joint shapes the
        // joint view with the compressed hparams, so the head split has to match as well
        GGML_ASSERT(n_embd_k_gqa                  == hparams_cmp.n_embd_k_gqa(il));
        GGML_ASSERT(hparams_raw.n_embd_head_k(il) == hparams_cmp.n_embd_head_k(il));
        GGML_ASSERT(hparams_raw.n_head_kv(il)     == hparams_cmp.n_head_kv(il));

        const uint32_t n_cmp = ratio == DSV4_CSA_RATIO ? size_csa : size_hca;

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for the DSV4 joint K storage");
        }

        ggml_tensor * joint = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, size_swa + n_cmp, 1);
        ggml_format_name(joint, "cache_kall_l%d", il);

        kall_layers[il].joint = joint;

        plan.push_back({ (int32_t) il, n_cmp, ctx });
    }

    if (kall_layers.empty()) {
        return;
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for the DSV4 joint K storage");
        }

        LLAMA_LOG_INFO("%s: %10s DSV4 joint K buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs_kall.emplace_back(std::move(ctx), buf);
    }

    // the sub-caches receive views of the joint tensors as their K storage
    for (const auto & p : plan) {
        const int32_t  il    = p.il;
        const uint32_t n_cmp = p.n_cmp;
        ggml_context * ctx   = p.ctx;

        auto & kl = kall_layers[il];

        ggml_tensor * joint = kl.joint;

        kl.view_raw = ggml_view_3d(ctx, joint, joint->ne[0], size_swa, 1,
                joint->nb[1], joint->nb[1]*size_swa, 0);
        ggml_format_name(kl.view_raw, "cache_kall_raw_l%d", il);

        kl.view_cmp = ggml_view_3d(ctx, joint, joint->ne[0], n_cmp, 1,
                joint->nb[1], joint->nb[1]*n_cmp, size_swa*joint->nb[1]);
        ggml_format_name(kl.view_cmp, "cache_kall_cmp_l%d", il);

        const ggml_status st_raw = ggml_backend_view_init(kl.view_raw);
        const ggml_status st_cmp = ggml_backend_view_init(kl.view_cmp);
        if (st_raw != GGML_STATUS_SUCCESS || st_cmp != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to initialize the DSV4 joint K views");
        }
    }

    kall_raw_size = size_swa;

    LLAMA_LOG_INFO("%s: DSV4 joint raw+compressed K layout enabled for %d layers, raw prefix = %u cells\n",
            __func__, (int) kall_layers.size(), kall_raw_size);
    LLAMA_LOG_INFO("%s: DSV4 raw attention width: the visible raw cells, copy from %u tokens\n",
            __func__, raw_nkv_min_tokens);
}

llama_kv_cache_dsv4::llama_kv_cache_dsv4(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    hparams_raw(model.hparams),
    hparams_csa(model.hparams),
    hparams_hca(model.hparams),
    hparams_lid(model.hparams),
    n_seq_max(n_seq_max),
    n_rs_seq(n_rs_seq),
    rs_idx(n_seq_max, 0) {

    tail_v41     = model.arch == LLM_ARCH_DEEPSEEK41;
    tail_n_swa   = model.hparams.n_swa;
    tail_n_layer = model.hparams.n_layer();
    tail_lo     .assign(n_seq_max, {});
    tail_bad    .assign(n_seq_max, 0);
    tail_end    .assign(n_seq_max, -1);
    tail_pending.assign(n_seq_max, 0);

    const layer_filter_cb filter_raw = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return true;
    };

    GGML_UNUSED(unified);

    // Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
    const bool unified_raw = false;

    hparams_raw.n_layer_nextn = 0;
    hparams_csa.n_layer_nextn = 0;
    hparams_hca.n_layer_nextn = 0;
    hparams_lid.n_layer_nextn = 0;

    dsv4_make_k_only(hparams_raw);
    dsv4_make_k_only(hparams_csa);
    dsv4_make_k_only(hparams_hca);

    // one allocation per layer for the raw and the compressed K, so that the attention graph can read
    // [raw | compressed] as a single view instead of copying both into a new tensor every token
    init_kall(model, type_k, offload, swa_full, kv_size, n_seq_max, n_ubatch);

    // the raw cache is built first: only a layer whose raw K went into the joint storage may hand the
    // joint storage to its compressed cache, otherwise the view would read raw cells that are not there
    const llama_kv_cache::layer_k_storage_cb kall_raw_cb =
            [this](int32_t il, uint32_t n_embd_k_gqa, uint32_t kv_size_l, uint32_t n_stream_l) -> llama_kv_cache::k_storage_info {
        auto it = kall_layers.find(il);
        if (it == kall_layers.end()) {
            return {};
        }

        ggml_tensor * view = it->second.view_raw;

        if (n_stream_l != 1 || kv_size_l != kall_raw_size || n_embd_k_gqa != (uint32_t) view->ne[0]) {
            throw std::runtime_error("DSV4 joint K storage: raw cache layout of layer " +
                    std::to_string(il) + " differs from the joint allocation");
        }

        it->second.raw_ok = true;

        return { view, it->second.joint, 0 };
    };

    const llama_kv_cache::layer_k_storage_cb kall_cmp_cb =
            [this](int32_t il, uint32_t n_embd_k_gqa, uint32_t kv_size_l, uint32_t n_stream_l) -> llama_kv_cache::k_storage_info {
        auto it = kall_layers.find(il);
        if (it == kall_layers.end() || !it->second.raw_ok) {
            return {};
        }

        ggml_tensor * view = it->second.view_cmp;

        if (n_stream_l != 1 || kv_size_l != (uint32_t) view->ne[1] || n_embd_k_gqa != (uint32_t) view->ne[0]) {
            throw std::runtime_error("DSV4 joint K storage: compressed cache layout of layer " +
                    std::to_string(il) + " differs from the joint allocation");
        }

        return { view, it->second.joint, kall_raw_size };
    };

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr, kall_raw_cb);

    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_k_swa  = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_swa  = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
    dsv4_make_k_only(hparams_lid);

    // DeepSeek-V4 compresses on every layer, at one of two fixed ratios, and each layer owns its
    // own rows. V4.1 compresses on a few source layers and shares each stream with the layers that
    // follow, still at two ratios. The two plan slots carry the two ratios in both cases; what
    // differs is which layers own storage and which borrow it through the reuse callback.
    const bool is_v41 = model.arch == LLM_ARCH_DEEPSEEK41;

    uint32_t ratio_a = DSV4_CSA_RATIO;
    uint32_t ratio_b = DSV4_HCA_RATIO;

    if (is_v41) {
        ratio_a = 0;
        ratio_b = 0;

        for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
            const uint32_t r = model.hparams.dsv4_compress_ratios[il];
            if (r == 0 || r == ratio_a || r == ratio_b) {
                continue;
            }
            if (ratio_a == 0) {
                ratio_a = r;
            } else if (ratio_b == 0) {
                ratio_b = r;
            } else {
                throw std::runtime_error("DeepSeek-V4.1 supports at most two compression ratios");
            }
        }

        // a file with no compressed layers is pure sliding window attention, which is valid.
        // every filter below then selects nothing, so the ratio only has to stay non-zero for
        // the size arithmetic.
        if (ratio_a == 0) {
            ratio_a = 1;
        }
        if (ratio_b == 0) {
            ratio_b = ratio_a;
        }
    }

    // V4.1 keeps every compressed row in one cache, sized for the finest ratio, with a slot per
    // source layer; a coarser source simply uses fewer of its rows. Readers alias onto their
    // source's slot, which is what the reuse callbacks below do.
    const uint32_t ratio_kv = is_v41 ? std::min(ratio_a, ratio_b) : DSV4_CSA_RATIO;

    const layer_filter_cb filter_csa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        if (is_v41) {
            return model.hparams.dsv41_is_kv_source(il);
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_hca = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        // V4.1 holds every compressed row in the CSA cache, so this one stays empty
        if (is_v41) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    // the pooling state is per ratio, since it carries a partial group across ubatches
    const layer_filter_cb filter_state_a = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        if (is_v41) {
            return model.hparams.dsv41_is_kv_source(il) && model.hparams.dsv4_compress_ratios[il] == ratio_a;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_state_b = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        if (is_v41) {
            return model.hparams.dsv41_is_kv_source(il) && model.hparams.dsv4_compress_ratios[il] == ratio_b;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    // V4.1 derives index keys from the already pooled latent, so they need storage but no pooling
    // state of their own; V4 runs a second compressor for them.
    const layer_filter_cb filter_lid = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        if (is_v41) {
            return model.hparams.dsv41_owns_index_k(il);
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_reuse_cb reuse_kv = is_v41
        ? layer_reuse_cb([&](int32_t il) { return model.hparams.dsv41_kv_source[il]; })
        : layer_reuse_cb(nullptr);

    const layer_reuse_cb reuse_lid = is_v41
        ? layer_reuse_cb([&](int32_t il) { return model.hparams.dsv41_index_key_source[il]; })
        : layer_reuse_cb(nullptr);

    const bool unified_compressed = false;

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressed KV cache, ratio = %u, size = %u cells\n",
            __func__, ratio_a, dsv4_comp_size(kv_size, ratio_kv));

    kv_csa = std::make_unique<llama_kv_cache>(
            model, hparams_csa, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, ratio_kv), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, reuse_kv, nullptr, "", kall_cmp_cb);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressed KV cache, ratio = %u, size = %u cells\n",
            __func__, ratio_b, dsv4_comp_size(kv_size, ratio_b));

    kv_hca = std::make_unique<llama_kv_cache>(
            model, hparams_hca, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, ratio_b), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_hca, nullptr, nullptr, "", kall_cmp_cb);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, ratio_kv));

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified_compressed, GGML_PAD(dsv4_comp_size(kv_size, ratio_kv), 256u), n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_lid, reuse_lid, nullptr);

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressor state\n", __func__);

    csa_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, ratio_a, !is_v41, is_v41 ? ratio_a : 2*ratio_a,
            (is_v41 ? 1 : 2)*model.hparams.n_embd_head_k(), n_rs_seq, "csa", filter_state_a);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressor state\n", __func__);

    hca_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, ratio_b, false, ratio_b,
            model.hparams.n_embd_head_k(), n_rs_seq, "hca", filter_state_b);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer compressor state\n", __func__);

    lid_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, unified_compressed, n_seq_max, ratio_a, !is_v41, is_v41 ? ratio_a : 2*ratio_a,
            (is_v41 ? 1 : 2)*model.hparams.indexer_head_size, n_rs_seq, "lid", filter_lid);

    // DSV4 attention reads compressed-K / compressor-state rows that the current
    // graph does not necessarily overwrite; uninitialized buffer contents would
    // otherwise leak in (instance-specific garbage) and corrupt recall. Zero all
    // compressed buffers up front so reads of un-written rows are deterministic.
    clear_compressed(-1, true);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    const bool raw_per_seq  = kv_raw->get_base()->get_n_stream() != 1;
    const bool comp_per_seq = csa_state->get_n_stream() > 1;
    const bool has_coupled = dsv4_batch_has_coupled(balloc.get_batch());

    // ranma: a batch that would read raw cells a tail prefill left unwritten fails, so that the caller starts over
    if (!tail_check_batch(balloc)) {
        return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_COMPUTE);
    }

    // ranma: the end contract of the tail prefill, a batch of one sequence at consecutive positions whose only output,
    // if any, is its last token. A batch without outputs ends there too: its state is what the next call continues from.
    llama_seq_id out_seq = -1;
    llama_pos    out_pos = -1;
    bool         out_has = false;
    {
        const llama_batch & batch = balloc.get_batch();
        const int32_t   n         = batch.n_tokens;
        const uint32_t  n_outputs = balloc.get_n_outputs();

        bool ok = n > 0 && batch.token != nullptr && batch.embd == nullptr &&
            batch.pos != nullptr && batch.n_seq_id != nullptr && batch.seq_id != nullptr && batch.logits != nullptr &&
            (n_outputs == 0 || (n_outputs == 1 && batch.logits[n - 1] != 0));
        for (int32_t i = 0; ok && i < n; ++i) {
            ok = batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == batch.seq_id[0][0] && batch.pos[i] == batch.pos[0] + i;
        }
        if (ok) {
            out_seq = batch.seq_id[0][0];
            out_pos = batch.pos[n - 1];
            out_has = n_outputs == 1;
        }
    }

    if (dsv4_tail_requested(tail_v41)) {
        tail_log_batch(balloc, out_seq, out_pos, out_has);
    }

    const auto make_context = [&](std::vector<llama_ubatch> ubatches) -> llama_memory_context_ptr {
        auto ubatches_raw = dsv4_build_raw_write_ubatches(ubatches);

        auto sinfos_raw_base_write = kv_raw->get_base()->prepare(ubatches_raw);
        if (sinfos_raw_base_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_write = kv_raw->get_swa()->prepare(ubatches_raw);
        if (sinfos_raw_swa_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_read = dsv4_build_raw_read_sinfos(sinfos_raw_swa_write, ubatches);

        auto ctx = std::make_unique<llama_kv_cache_dsv4_context>(
                this,
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                std::move(ubatches),
                std::move(ubatches_raw));
        ctx->set_batch_out(out_seq, out_pos, out_has);

        return ctx;
    };

    // Match llama_kv_cache_iswa splitting when DSV4 compressed state does not
    // require per-sequence graph layout.
    do {
        if (raw_per_seq || comp_per_seq) {
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    // When raw or compressed state is per-sequence, independent sequences can
    // share an equal-length ubatch. Coupled sequence sets still serialize until
    // DSV4 has explicit shared-state handling for compressed streams.
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (has_coupled) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                ubatch = balloc.split_equal(n_ubatch, raw_per_seq || comp_per_seq, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_full() {
    return std::make_unique<llama_kv_cache_dsv4_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv4_context>(
            this,
            lctx,
            optimize,
            std::move(csa_state->sc_info),
            std::move(hca_state->sc_info),
            std::move(lid_state->sc_info));
}

bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}

void llama_kv_cache_dsv4::clear(bool data) {
    tail_reset(-1);
    kv_raw->clear(data);
    clear_compressed(-1, true); // DSV4 compressed buffers must never expose stale/uninit rows
}

bool llama_kv_cache_dsv4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        return false;
    }

    if (p0 > 0) {
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
            return false;
        }

        const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);

        // ranma: a removal into the rows a tail prefill left unwritten is refused, as one past the compressed rollback
        // limit. A refusal leaves the memory as it was, which the caller can go on from, except while a call is pending:
        // that is the cleanup of an aborted or failed ubatch (llama_context::decode), whose positions then stay with
        // cells nobody wrote, so the sequence is marked and nothing continues from it until it is cleared or restored.
        const auto refuse = [&]() {
            if (!tail_lo[seq_id].empty() && tail_pending[seq_id]) {
                tail_bad[seq_id] = 1;
            }
            return false;
        };

        if (p0 <= pos_max && !tail_can_continue(seq_id, p0 - 1)) {
            LLAMA_LOG_WARN("%s: seq_id = %d, removal from pos %d refused: the tail prefill left raw cells before it unwritten\n",
                    __func__, seq_id, p0);
            return refuse();
        }

        if (p0 > pos_max) {
            bool res = true;

            res = res & kv_raw->seq_rm(seq_id, p0, -1);
            res = res & kv_csa->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);
            res = res & kv_hca->seq_rm(seq_id, p0/DSV4_HCA_RATIO, -1);
            res = res & kv_lid->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);

            return res;
        }

        if (!comp_can_rollback(seq_id, p0)) {
            return refuse();
        }

        const llama_pos rollback = pos_max - (p0 - 1);

        const bool res = kv_raw->seq_rm(seq_id, p0, p1);
        if (res) {
            rs_idx[seq_id] = (uint32_t) rollback;
        } else if (!tail_lo[seq_id].empty()) {
            // the raw cache may have removed part of it
            tail_bad[seq_id] = 1;
        }

        return res;
    }

    const bool res = kv_raw->seq_rm(seq_id, p0, p1);

    if (res) {
        clear_compressed(seq_id, true);
        tail_reset(seq_id);
    }

    return res;
}

void llama_kv_cache_dsv4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "DSV4 only supports full sequence copies");

    kv_raw->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_csa->seq_cp(seq_id_src, seq_id_dst, -1, -1);
    kv_hca->seq_cp(seq_id_src, seq_id_dst, -1, -1);
    kv_lid->seq_cp(seq_id_src, seq_id_dst, -1, -1);

    csa_state->seq_cp(seq_id_src, seq_id_dst);
    hca_state->seq_cp(seq_id_src, seq_id_dst);
    lid_state->seq_cp(seq_id_src, seq_id_dst);

    if (seq_id_src != seq_id_dst) {
        rs_idx[seq_id_dst] = 0;

        // ranma: the copy has the same cells, written or not
        if (seq_id_src >= 0 && (uint32_t) seq_id_src < n_seq_max && seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_seq_max) {
            tail_lo     [seq_id_dst] = tail_lo     [seq_id_src];
            tail_bad    [seq_id_dst] = tail_bad    [seq_id_src];
            tail_end    [seq_id_dst] = tail_end    [seq_id_src];
            tail_pending[seq_id_dst] = tail_pending[seq_id_src];
        }
    }
}

void llama_kv_cache_dsv4::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    kv_raw->seq_keep(seq_id);

    for (llama_seq_id id = 0; id < (llama_seq_id) n_seq_max; ++id) {
        if (id == seq_id) {
            continue;
        }

        kv_raw->seq_rm(id, -1, -1);
        clear_compressed(id, true);
        tail_reset(id);
    }
}

// ranma: moved positions no longer match those a tail prefill recorded; such a sequence is not continued
static void dsv4_tail_mark_moved(const std::vector<std::vector<llama_pos>> & tail_lo, std::vector<uint8_t> & tail_bad,
        llama_seq_id seq_id) {
    for (size_t s = 0; s < tail_lo.size(); ++s) {
        if ((seq_id < 0 || (size_t) seq_id == s) && !tail_lo[s].empty()) {
            tail_bad[s] = 1;
        }
    }
}

void llama_kv_cache_dsv4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_raw->seq_add(seq_id, p0, p1, shift);
    if (shift != 0) {
        dsv4_tail_mark_moved(tail_lo, tail_bad, seq_id);
    }
}

void llama_kv_cache_dsv4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_raw->seq_div(seq_id, p0, p1, d);
    if (d != 1) {
        dsv4_tail_mark_moved(tail_lo, tail_bad, seq_id);
    }
}

llama_pos llama_kv_cache_dsv4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    // The raw SWA cache may contain a wider window, but the compressed DSV4
    // state cannot be rolled back within that window. Report only the current
    // boundary so server-context uses checkpoints for rollback.
    return kv_raw->seq_pos_max(seq_id);
}

llama_pos llama_kv_cache_dsv4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return kv_raw->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv4::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_raw->memory_breakdown();
    for (const auto & buft_size : kv_csa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_hca->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : csa_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : hca_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : lid_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    // the joint raw/compressed K buffers are not owned by any of the sub-caches
    for (const auto & [_, buf] : ctxs_bufs_kall) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return mb;
}

void llama_kv_cache_dsv4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool partial_only = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    // ranma: a state is restored to continue from its last position; one that cannot be is not saved
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if ((seq_id < 0 || (uint32_t) seq_id == s) &&
                !tail_can_continue((llama_seq_id) s, kv_raw->seq_pos_max((llama_seq_id) s))) {
            throw std::runtime_error("DSV4 state: the tail prefill of this sequence did not complete");
        }
    }

    // ranma: the records of the tail prefill go with the state when it ran on one of its sequences or is asked for,
    // so that a restore knows which raw cells are written; otherwise the state is as upstream writes it
    bool with_tail = dsv4_tail_requested(tail_v41);
    for (uint32_t s = 0; s < n_seq_max && !with_tail; ++s) {
        with_tail = (seq_id < 0 || (uint32_t) seq_id == s) && !tail_lo[s].empty();
    }

    const uint32_t magic   = DSV4_STATE_MAGIC;
    const uint32_t version = DSV4_STATE_VERSION;
    const uint32_t mode    = (partial_only ? DSV4_STATE_MODE_PARTIAL : DSV4_STATE_MODE_FULL) |
        (with_tail ? DSV4_STATE_MODE_TAIL : 0);

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&mode,    sizeof(mode));

    if (with_tail) {
        tail_state_write(io, seq_id);
    }

    kv_raw->state_write(io, seq_id, flags);

    if (!partial_only) {
        const llama_pos pos_max = seq_id >= 0 ? kv_raw->seq_pos_max(seq_id) : -1;

        //FIXME : note that we conflate token positions with rows, which is not true for multi-modal case.
        const uint32_t n_rows_csa = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_csa->get_size()) : kv_csa->get_size();
        const uint32_t n_rows_hca = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_HCA_RATIO, kv_hca->get_size()) : kv_hca->get_size();
        const uint32_t n_rows_lid = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_lid->get_size()) : kv_lid->get_size();

        dsv4_state_write_k_cache(io, kv_csa.get(), seq_id, flags, n_rows_csa);
        dsv4_state_write_k_cache(io, kv_hca.get(), seq_id, flags, n_rows_hca);
        dsv4_state_write_k_cache(io, kv_lid.get(), seq_id, flags, n_rows_lid);
    }

    csa_state->state_write(io, seq_id, flags, rs_idx);
    hca_state->state_write(io, seq_id, flags, rs_idx);
    lid_state->state_write(io, seq_id, flags, rs_idx);
}

void llama_kv_cache_dsv4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    uint32_t magic;
    uint32_t version;
    uint32_t mode = DSV4_STATE_MODE_FULL;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));

    if (magic != DSV4_STATE_MAGIC) {
        throw std::runtime_error("DSV4 state magic mismatch");
    }
    if (version != DSV4_STATE_VERSION) {
        throw std::runtime_error("DSV4 state version mismatch");
    }

    io.read(&mode, sizeof(mode));

    const bool with_tail = mode & DSV4_STATE_MODE_TAIL;
    mode &= ~DSV4_STATE_MODE_TAIL;

    if (mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL) {
        throw std::runtime_error("DSV4 state mode mismatch");
    }

    const bool partial_only = mode == DSV4_STATE_MODE_PARTIAL;
    if (partial_only != !!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("DSV4 state flags mismatch");
    }

    // ranma: the tail records come before anything is restored, so that a mismatch leaves the memory as it was
    std::vector<tail_state_rec> recs;
    if (with_tail) {
        recs = tail_state_read(io);
    }
    std::vector<llama_pos> pos_max_before(n_seq_max, -1);
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        pos_max_before[s] = kv_raw->seq_pos_max((llama_seq_id) s);
    }

    kv_raw->state_read(io, seq_id, flags);

    if (!partial_only) {
        clear_compressed(seq_id, true);

        dsv4_state_read_k_cache(io, kv_csa.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_hca.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_lid.get(), seq_id, flags);
    }

    csa_state->state_read(io, seq_id, flags);
    hca_state->state_read(io, seq_id, flags);
    lid_state->state_read(io, seq_id, flags);

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }

    tail_after_restore(seq_id, with_tail ? &recs : nullptr, partial_only, pos_max_before);
}

llama_kv_cache_iswa * llama_kv_cache_dsv4::get_raw() const {
    return kv_raw.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_csa() const {
    return kv_csa.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_hca() const {
    return kv_hca.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_lid() const {
    return kv_lid.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_csa_state() const {
    return csa_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_hca_state() const {
    return hca_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_lid_state() const {
    return lid_state.get();
}

uint32_t llama_kv_cache_dsv4::get_n_rs_seq() const {
    return n_rs_seq;
}

uint32_t llama_kv_cache_dsv4::get_raw_nkv_min_tokens() const {
    return raw_nkv_min_tokens;
}

std::string llama_kv_cache_dsv4::tail_describe(llama_seq_id seq_id) const {
    if (tail_bad[seq_id]) {
        return "broken (a refused removal)";
    }
    const auto & lo = tail_lo[seq_id];
    if (lo.empty()) {
        return "every cell written";
    }
    // runs of layers whose written cells start at the same position; -1: every cell of the layer
    std::string res = "written from";
    for (size_t il = 0; il < lo.size(); ) {
        size_t il_end = il + 1;
        while (il_end < lo.size() && lo[il_end] == lo[il]) {
            ++il_end;
        }
        if (lo[il] >= 0) {
            res += il_end - il == 1 ? format(" l%zu:%d", il, lo[il]) : format(" l%zu-%zu:%d", il, il_end - 1, lo[il]);
        }
        il = il_end;
    }
    return res;
}

void llama_kv_cache_dsv4::tail_log_batch(const llama_batch_allocr & balloc, llama_seq_id out_seq, llama_pos out_pos, bool out_has) const {
    const uint32_t n_tokens = balloc.get_n_tokens();
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        const llama_pos p0 = balloc.seq_pos_min((llama_seq_id) s);
        if (p0 < 0) {
            continue;
        }
        const llama_pos p1 = balloc.seq_pos_max((llama_seq_id) s);

        const std::string end = (llama_seq_id) s == out_seq ? format("end %d, %s", out_pos + 1, out_has ? "output" : "no output") :
            std::string("none");
        const std::string msg = format("seq %u: batch [%d, %d) of %u tokens, %u outputs, contract %s; memory to %d, raw %s",
                s, p0, p1 + 1, n_tokens, balloc.get_n_outputs(), end.c_str(),
                kv_raw->seq_pos_max((llama_seq_id) s) + 1, tail_describe((llama_seq_id) s).c_str());
        // a decode of one token per sequence is logged at debug level only, as is any batch without the diagnostic
        if (n_tokens > 1 && dsv4_tail_diag()) {
            LLAMA_LOG_INFO("%s: %s\n", __func__, msg.c_str());
        } else {
            LLAMA_LOG_DEBUG("%s: %s\n", __func__, msg.c_str());
        }
    }
}

void llama_kv_cache_dsv4::tail_set_written(llama_seq_id seq_id, int32_t il, llama_pos p0) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max && il >= 0 && (uint32_t) il < tail_n_layer);

    auto & lo = tail_lo[seq_id];
    if (lo.empty()) {
        lo.assign(tail_n_layer, -1);
    }
    // the run of written cells now starts at p0, unless it already started later
    lo[il] = std::max(lo[il], p0);
    tail_seen = true;
}

bool llama_kv_cache_dsv4::comp_can_rollback(llama_seq_id seq_id, llama_pos p0) const {
    const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);
    if (p0 > pos_max) {
        return true;
    }
    if (n_rs_seq == 0) {
        return false;
    }

    const llama_pos rollback = pos_max - (p0 - 1);
    if (rollback < 1 || rollback > (llama_pos) n_rs_seq) {
        return false;
    }

    // pending rollback is single-use: stacked partial removals don't compose
    return rs_idx[seq_id] == 0;
}

bool llama_kv_cache_dsv4::seq_can_resume(llama_seq_id seq_id, llama_pos pos) const {
    // only with the tail prefill asked for; otherwise the caller keeps its generic rules
    if (!dsv4_exact_tail_on(tail_v41)) {
        return false;
    }
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max || pos <= 0) {
        return false;
    }
    // a refused cleanup, or a call that did not complete
    if (tail_bad[seq_id] || tail_pending[seq_id]) {
        return false;
    }

    const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);
    if (pos_max < 0 || pos > pos_max + 1) {
        return false;
    }

    return tail_can_continue(seq_id, pos - 1) && comp_can_rollback(seq_id, pos);
}

bool llama_kv_cache_dsv4::tail_can_continue(llama_seq_id seq_id, llama_pos p_end) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max || p_end < 0) {
        return true;
    }
    if (tail_bad[seq_id]) {
        return false;
    }
    // the next position reads the cells [p_end + 2 - n_swa, p_end] of every layer besides its own
    const llama_pos need = std::max<llama_pos>(0, p_end + 2 - (llama_pos) tail_n_swa);
    for (llama_pos p : tail_lo[seq_id]) {
        if (p > need) {
            return false;
        }
    }
    return true;
}

void llama_kv_cache_dsv4::tail_reset(llama_seq_id seq_id) {
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_id < 0 || (uint32_t) seq_id == s) {
            tail_lo[s].clear();
            tail_bad[s]     = 0;
            tail_end[s]     = -1;
            tail_pending[s] = 0;
        }
    }
}

// the sequences of the ubatches of one batch, each once
static std::vector<llama_seq_id> dsv4_ubatches_seqs(const std::vector<llama_ubatch> & ubatches, uint32_t n_seq_max) {
    std::vector<uint8_t> seen(n_seq_max, 0);
    std::vector<llama_seq_id> res;
    for (const llama_ubatch & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t k = 0; k < ubatch.n_seq_id[i]; ++k) {
                const llama_seq_id s = ubatch.seq_id[i][k];
                if (s >= 0 && (uint32_t) s < n_seq_max && !seen[s]) {
                    seen[s] = 1;
                    res.push_back(s);
                }
            }
        }
    }
    return res;
}

void llama_kv_cache_dsv4::tail_call_begin(const std::vector<llama_ubatch> & ubatches) {
    for (llama_seq_id s : dsv4_ubatches_seqs(ubatches, n_seq_max)) {
        tail_pending[s] = 1;
    }
}

void llama_kv_cache_dsv4::tail_call_end(const std::vector<llama_ubatch> & ubatches, const char * what) {
    const bool log = dsv4_tail_diag();
    for (llama_seq_id s : dsv4_ubatches_seqs(ubatches, n_seq_max)) {
        tail_pending[s] = 0;
        tail_end[s]     = kv_raw->seq_pos_max(s) + 1;
        if (log) {
            LLAMA_LOG_INFO("%s: seq %d: call to %d complete (%s), raw %s\n", __func__, s, tail_end[s],
                    what, tail_describe(s).c_str());
        }
    }
}

// the block: version, layer count, record count, then per record its sequence, its end, the count of its first written
// positions (0 or the layer count) and those positions
void llama_kv_cache_dsv4::tail_state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<tail_state_rec> recs;
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        const llama_pos pos_max = kv_raw->seq_pos_max((llama_seq_id) s);
        if ((seq_id >= 0 && (uint32_t) seq_id != s) || pos_max < 0) {
            continue;
        }
        recs.push_back({ (llama_seq_id) s, pos_max + 1, tail_lo[s] });
    }

    const uint32_t ver     = DSV4_STATE_TAIL_VER;
    const uint32_t n_layer = tail_n_layer;
    const uint32_t n_rec   = (uint32_t) recs.size();

    io.write(&ver,     sizeof(ver));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_rec,   sizeof(n_rec));
    for (const auto & r : recs) {
        const int32_t  s    = r.seq_id;
        const int32_t  end  = r.end;
        const uint32_t n_lo = (uint32_t) r.lo.size();

        io.write(&s,    sizeof(s));
        io.write(&end,  sizeof(end));
        io.write(&n_lo, sizeof(n_lo));
        for (llama_pos p : r.lo) {
            const int32_t v = p;
            io.write(&v, sizeof(v));
        }
    }
}

std::vector<llama_kv_cache_dsv4::tail_state_rec> llama_kv_cache_dsv4::tail_state_read(llama_io_read_i & io) const {
    uint32_t ver     = 0;
    uint32_t n_layer = 0;
    uint32_t n_rec   = 0;

    io.read(&ver,     sizeof(ver));
    io.read(&n_layer, sizeof(n_layer));
    io.read(&n_rec,   sizeof(n_rec));

    if (ver != DSV4_STATE_TAIL_VER) {
        throw std::runtime_error("DSV4 state: tail record version mismatch");
    }
    if (n_layer != tail_n_layer || n_rec > n_seq_max) {
        throw std::runtime_error("DSV4 state: tail records of another model or cache");
    }

    std::vector<tail_state_rec> recs(n_rec);
    for (auto & r : recs) {
        int32_t  s    = -1;
        int32_t  end  = 0;
        uint32_t n_lo = 0;

        io.read(&s,    sizeof(s));
        io.read(&end,  sizeof(end));
        io.read(&n_lo, sizeof(n_lo));
        if (s < 0 || (uint32_t) s >= n_seq_max || (n_lo != 0 && n_lo != n_layer)) {
            throw std::runtime_error("DSV4 state: tail record out of range");
        }

        r.seq_id = s;
        r.end    = end;
        r.lo.resize(n_lo);
        for (auto & p : r.lo) {
            int32_t v = 0;
            io.read(&v, sizeof(v));
            p = v;
        }
    }
    return recs;
}

void llama_kv_cache_dsv4::tail_after_restore(llama_seq_id seq_id, const std::vector<tail_state_rec> * recs, bool partial,
        const std::vector<llama_pos> & pos_max_before) {
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_id >= 0 && (uint32_t) seq_id != s) {
            continue;
        }
        tail_lo[s].clear();
        tail_bad[s]     = 0;
        tail_pending[s] = 0;

        const llama_pos pos_max = kv_raw->seq_pos_max((llama_seq_id) s);
        tail_end[s] = pos_max >= 0 ? pos_max + 1 : -1;
        if (pos_max < 0) {
            continue;
        }

        // the record of this sequence: the one saved for it, or the only one when it was restored into another id
        const tail_state_rec * rec = nullptr;
        if (recs != nullptr) {
            for (const auto & r : *recs) {
                if (r.seq_id == (llama_seq_id) s || (seq_id >= 0 && recs->size() == 1)) {
                    rec = &r;
                }
            }
        }

        if (rec != nullptr && rec->end == pos_max + 1) {
            // the state says which cells are written
            tail_lo[s] = rec->lo;
            tail_seen  = tail_seen || !rec->lo.empty();
        } else if (rec != nullptr || tail_seen || dsv4_tail_requested(tail_v41)) {
            // a state without records (or with one that does not fit) was continuable at its end (see state_write);
            // where a tail may have run, nothing is known further back, so only the window of that end counts as
            // written
            tail_lo[s].assign(tail_n_layer, std::max<llama_pos>(0, pos_max + 1 - (llama_pos) tail_n_swa));
        }

        // a partial state keeps the global KV of the memory, which held the prefix to its end only if it reached as far
        if (partial && rec != nullptr && pos_max_before[s] < pos_max) {
            LLAMA_LOG_ERROR("%s: seq_id = %u: a partial state to %d restored over a memory to %d, whose global KV it "
                    "lacks; the sequence cannot continue until it is cleared\n", __func__, s, pos_max + 1, pos_max_before[s] + 1);
            tail_bad[s] = 1;
        }

        if (dsv4_tail_diag()) {
            LLAMA_LOG_INFO("%s: seq_id = %u: %s state to %d restored (%s), raw %s\n", __func__, s,
                    partial ? "partial" : "full", pos_max + 1,
                    rec == nullptr ? "no records" : rec->end == pos_max + 1 ? "records" : "records of another end",
                    tail_describe((llama_seq_id) s).c_str());
        }
    }
}

bool llama_kv_cache_dsv4::tail_check_batch(const llama_batch_allocr & balloc) const {
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (tail_lo[s].empty() && !tail_bad[s]) {
            continue;
        }
        const llama_pos p0 = balloc.seq_pos_min((llama_seq_id) s);
        if (p0 < 0) {
            continue;
        }
        const llama_pos p_end = std::min(p0 - 1, kv_raw->seq_pos_max((llama_seq_id) s));
        if (!tail_can_continue((llama_seq_id) s, p_end)) {
            LLAMA_LOG_ERROR("%s: seq_id = %u cannot continue at pos %d: a tail prefill left raw cells it reads unwritten "
                    "(an interrupted or partly removed tail batch); clear the sequence and process it again\n",
                    __func__, s, p0);
            return false;
        }
    }
    return true;
}

const std::vector<uint32_t> & llama_kv_cache_dsv4::get_rs_idx() const {
    return rs_idx;
}

void llama_kv_cache_dsv4::reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches) {
    if (n_rs_seq == 0) {
        return;
    }

    for (const llama_ubatch & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id[i][s];
                if (seq_id >= 0 && (uint32_t) seq_id < n_seq_max) {
                    rs_idx[seq_id] = 0;
                }
            }
        }
    }
}

void llama_kv_cache_dsv4::clear_compressed(llama_seq_id seq_id, bool data) {
    if (seq_id < 0) {
        kv_csa->clear(data);
        kv_hca->clear(data);
        kv_lid->clear(data);
    } else {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);

        const auto clear_seq = [seq_id, data](llama_kv_cache * kv) {
            kv->seq_rm(seq_id, -1, -1);

            if (data) {
                //TODO: do not clear the kv-cache during `seq_rm`, ref: https://github.com/ggml-org/llama.cpp/pull/26490#discussion_r3798143663
                for (uint32_t il : kv->get_layer_ids()) {
                    dsv4_clear_tensor_stream(kv->get_k_storage(il), (uint32_t) seq_id);
                }
            }
        };

        clear_seq(kv_csa.get());
        clear_seq(kv_hca.get());
        clear_seq(kv_lid.get());
    }

    csa_state->clear(seq_id, data);
    hca_state->clear(seq_id, data);
    lid_state->clear(seq_id, data);

    if (seq_id >= 0) {
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }
}

//
// llama_kv_cache_dsv4_raw_context
//

static llama_kv_cache::slot_info dsv4_build_full_sinfo(const llama_kv_cache * kv) {
    const uint32_t n_stream = kv->get_n_stream();

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = n_stream - 1;
    sinfo.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfo.strm[s] = s;
        sinfo.idxs[s].resize(1, 0);
    }

    return sinfo;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(nullptr),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
    sinfos_read.push_back(dsv4_build_full_sinfo(kv_swa));
    sinfos_write = sinfos_read;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa_mem(kv->get_swa()->init_update(lctx, optimize)),
    n_kv(kv_swa->get_size()),
    status(llama_memory_status_combine(ctx_base_mem->get_status(), ctx_swa_mem->get_status())) {
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base_write,
        slot_info_vec_t sinfos_swa_write,
        slot_info_vec_t sinfos_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_write,
        uint32_t n_min_tokens_window) :
    kv_swa(kv->get_swa()),
    sinfos_write(std::move(sinfos_swa_write)),
    sinfos_read(std::move(sinfos_swa_read)),
    ubatches(std::move(ubatches)),
    ubatches_write(std::move(ubatches_write)),
    ctx_base_mem(std::make_unique<llama_kv_cache_context>(
                kv->get_base(), std::move(sinfos_base_write), this->ubatches_write)),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    n_min_tokens_window(n_min_tokens_window),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_kv_cache_dsv4_raw_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (ctx_base_mem) {
        ctx_base_mem->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_raw_context::apply() {
    bool res = true;

    if (ctx_base_mem) {
        res = res & ctx_base_mem->apply();
    }
    if (ctx_swa_mem) {
        res = res & ctx_swa_mem->apply();
    }
    if (!ubatches_write.empty()) {
        kv_swa->apply_ubatch(sinfos_write[i_next], ubatches_write[i_next]);

        k_lo = 0;

        if (!kv_swa->has_k_joint()) {
            n_kv = kv_swa->get_n_kv(sinfos_read[i_next]);
        } else {
            // with joint raw/compressed K storage the compressed cells follow the whole raw cache, so a view of
            // [raw | compressed] reads the raw cells up to the end of the raw cache
            const uint32_t size = kv_swa->get_size();

            n_kv = size;

            uint32_t lo = 0;
            uint32_t hi = 0;
            if (kv_swa->get_visible_range(sinfos_read[i_next], ubatches[i_next], lo, hi)) {
                // 256-cell steps like get_n_kv, so that a graph can be reused while the window stays in its bucket
                const uint32_t n_align = 256;

                if (ubatches[i_next].n_tokens < n_min_tokens_window) {
                    // decode and draft batches keep the whole raw cache, as before the narrow window: the width of
                    // the attention (and so its result) does not depend on where the window sits, and a graph is
                    // reused for as long as the cache does not change
                    lo = 0;
                    hi = size;
                }

                lo = lo/n_align*n_align;
                hi = std::min(size, GGML_PAD(hi, n_align));

                k_lo = lo;
                n_kv = hi - lo;
            }
        }
    }

    return res;
}

llama_memory_status llama_kv_cache_dsv4_raw_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_raw_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_dsv4_raw_context::get_k_lo() const {
    return k_lo;
}

void llama_kv_cache_dsv4_raw_context::get_chunk_window(uint32_t t0, uint32_t n_tok, uint32_t & lo, uint32_t & n) const {
    lo = k_lo;
    n  = n_kv;

    // only a context that applied a ubatch knows its cells (the reserve graph keeps the whole window, its upper bound)
    if (!has_ubatch() || i_next >= ubatches.size() || i_next >= sinfos_read.size()) {
        return;
    }

    uint32_t c_lo = 0;
    uint32_t c_hi = 0;
    if (!kv_swa->get_visible_range(sinfos_read[i_next], ubatches[i_next], c_lo, c_hi, t0, t0 + n_tok, true)) {
        return;
    }

    // the 256-cell steps of the window of the ubatch (apply), inside it
    const uint32_t n_align = 256;

    c_lo = std::max(k_lo, c_lo/n_align*n_align);
    c_hi = std::min(k_lo + n_kv, (uint32_t) GGML_PAD(c_hi, n_align));
    if (c_lo >= c_hi) {
        return;
    }

    lo = c_lo;
    n  = c_hi - c_lo;
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_write() const {
    if (ubatches_write.empty()) {
        return 0;
    }

    return ubatches_write[i_next].n_tokens;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::get_k(ggml_context * ctx, int32_t il) const {
    ggml_tensor * k = kv_swa->get_k(ctx, il, k_lo + n_kv, sinfos_read[i_next]);
    if (k_lo == 0) {
        return k;
    }

    return ggml_view_4d(ctx, k, k->ne[0], k->ne[1], n_kv, k->ne[3], k->nb[1], k->nb[2], k->nb[3], k->nb[2]*k_lo);
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const auto & sinfo = sinfos_write[i_next];

    if (k_cur->ne[2] == k_idxs->ne[0]) {
        return kv_swa->cpy_k(ctx, k_cur, k_idxs, il, sinfo);
    }

    // k_idxs may be expanded to one block per stream while k_cur is only
    // the token block. Keep zero deps on all copies so each write executes.
    const int64_t n_fanout = (int64_t) sinfo.size()*sinfo.n_stream();

    GGML_ASSERT(sinfo.n_stream() > 1);
    GGML_ASSERT(k_cur->ne[2] == (int64_t) sinfo.size());
    GGML_ASSERT(k_idxs->ne[0] == n_fanout);

    ggml_tensor * res = nullptr;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        ggml_tensor * k_idxs_s = ggml_view_1d(ctx, k_idxs, sinfo.size(), s*sinfo.size()*ggml_element_size(k_idxs));
        ggml_tensor * cur = kv_swa->cpy_k(ctx, k_cur, k_idxs_s, il, sinfo);
        if (res == nullptr) {
            res = cur;
        } else {
            res = ggml_add(ctx, res, ggml_sub(ctx, cur, cur));
        }
    }

    return res;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatches_write.empty() ? ubatch.n_tokens : ubatches_write[i_next].n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_rot(ggml_context * ctx) const {
    return kv_swa->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_idxs(ggml_tensor * dst) const {
    kv_swa->set_input_k_idxs(dst, &ubatches_write[i_next], sinfos_write[i_next]);
}

// env: RANMA_DSV4_RAW_NKV_CHECK (default 0)
// 1 = every raw mask of a window narrower than the raw cache is compared with the mask over the whole raw cache: the
//     columns of the window have to be equal and every column outside -inf, otherwise abort. Needs a host buffer of
//     n_tokens x raw cache size values per ubatch; for correctness runs only.
static bool dsv4_raw_nkv_check() {
    static const bool res = []() {
        const char * env = getenv("RANMA_DSV4_RAW_NKV_CHECK");
        return env != nullptr && atoi(env) != 0;
    }();

    return res;
}

static float dsv4_mask_value(const ggml_tensor * t, int64_t i) {
    return t->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) t->data)[i]) : ((const float *) t->data)[i];
}

static void dsv4_check_raw_window(const llama_kv_cache * kv, const ggml_tensor * dst, const llama_ubatch * ubatch,
        bool causal_attn, uint32_t k_lo) {
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1 && "the raw window is single-stream");
    GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    const int64_t n_cells = kv->get_size();
    const int64_t n_kv    = dst->ne[0];
    const int64_t n_rows  = dst->ne[1];

    llm_graph_host_tensor full(dst->type, n_cells, n_rows);
    kv->set_input_kq_mask(full.t, ubatch, causal_attn, 0);

    int64_t n_visible = 0;
    for (int64_t r = 0; r < n_rows; ++r) {
        for (int64_t c = 0; c < n_cells; ++c) {
            const float v_full = dsv4_mask_value(full.t, r*n_cells + c);

            if (c >= k_lo && c < k_lo + n_kv) {
                const float v_win = dsv4_mask_value(dst, r*n_kv + c - k_lo);
                if (!(v_win == v_full || (std::isnan(v_win) && std::isnan(v_full)))) {
                    GGML_ABORT("RANMA_DSV4_RAW_NKV_CHECK: row %" PRId64 " cell %" PRId64 " is %f in the window [%u, %" PRId64 ") but %f over the whole raw cache",
                            r, c, v_win, k_lo, k_lo + n_kv, v_full);
                }
                n_visible += v_full != -INFINITY;
            } else if (v_full != -INFINITY) {
                GGML_ABORT("RANMA_DSV4_RAW_NKV_CHECK: row %" PRId64 " sees cell %" PRId64 " (%f) outside the window [%u, %" PRId64 ") of %" PRId64 " cells",
                        r, c, v_full, k_lo, k_lo + n_kv, n_cells);
            }
        }
    }

    LLAMA_LOG_DEBUG("%s: window [%u, %" PRId64 ") of %" PRId64 " raw cells, %" PRId64 " tokens, %" PRId64 " visible entries: ok\n",
            __func__, k_lo, k_lo + n_kv, n_cells, n_rows, n_visible);
}

bool llama_kv_cache_dsv4_raw_context::check_enabled() {
    return dsv4_raw_nkv_check();
}

void llama_kv_cache_dsv4_raw_context::check_chunk_windows(const llama_ubatch * ubatch, const std::vector<chunk_window> & windows) const {
    if (!dsv4_raw_nkv_check() || windows.empty()) {
        return;
    }

    // the causal mask of the whole ubatch over the whole raw cache, once for all chunks (F16 holds 0 and -inf exactly)
    const int64_t n_cells = kv_swa->get_size();

    llm_graph_host_tensor full(GGML_TYPE_F16, n_cells, ubatch->n_tokens);
    kv_swa->set_input_kq_mask(full.t, ubatch, true, 0);

    for (const auto & w : windows) {
        GGML_ASSERT(w.t0 + w.n_tok <= ubatch->n_tokens);
        GGML_ASSERT(w.lo >= k_lo && w.lo + w.n <= k_lo + n_kv && "the chunk window lies outside the window of the ubatch");

        int64_t n_visible = 0;
        for (int64_t r = w.t0; r < (int64_t) (w.t0 + w.n_tok); ++r) {
            for (int64_t c = 0; c < n_cells; ++c) {
                const float v = dsv4_mask_value(full.t, r*n_cells + c);
                if (c >= w.lo && c < w.lo + w.n) {
                    n_visible += v != -INFINITY;
                } else if (v != -INFINITY) {
                    GGML_ABORT("RANMA_DSV4_RAW_NKV_CHECK: row %" PRId64 " of the chunk [%u, %u) sees cell %" PRId64 " (%f) outside its window [%u, %u) of %" PRId64 " cells",
                            r, w.t0, w.t0 + w.n_tok, c, v, w.lo, w.lo + w.n, n_cells);
                }
            }
        }

        LLAMA_LOG_DEBUG("%s: chunk [%u, %u): window [%u, %u) of [%u, %u), %" PRId64 " raw cells, %" PRId64 " visible entries: ok\n",
                __func__, w.t0, w.t0 + w.n_tok, w.lo, w.lo + w.n, k_lo, k_lo + n_kv, n_cells, n_visible);
    }
}

void llama_kv_cache_dsv4_raw_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv_swa->set_input_kq_mask(dst, ubatch, causal_attn, k_lo);

    if (dsv4_raw_nkv_check() && (k_lo != 0 || n_kv != kv_swa->get_size())) {
        dsv4_check_raw_window(kv_swa, dst, ubatch, causal_attn, k_lo);
    }
}

void llama_kv_cache_dsv4_raw_context::set_input_kq_pos(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv_swa->set_input_kq_pos(dst, ubatch, k_lo);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_rot(ggml_tensor * dst) const {
    kv_swa->set_input_k_rot(dst);
}

void llama_kv_cache_dsv4_raw_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv_swa->get_prev_tokens(ubatch, n, res);
}

//
// llama_kv_cache_dsv4_comp_context
//

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv) : kv(kv), n_kv(kv->get_size()) {
    const uint32_t n_stream = kv->get_n_stream();

    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(
        llama_kv_cache * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) :
    kv(kv),
    sinfos(std::move(sinfos)),
    ubatches(std::move(ubatches)),
    n_kv(kv->get_size()) {
}

bool llama_kv_cache_dsv4_comp_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

uint32_t llama_kv_cache_dsv4_comp_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k_joint(ggml_context * ctx, int32_t il, uint32_t raw_lo, uint32_t n_raw, uint32_t n_comp) const {
    // the compressed cells start right after the raw cache, so [raw_lo, raw_lo + n_raw) has to end there
    return kv->get_k_joint(ctx, il, raw_lo + n_raw, n_raw + n_comp, raw_lo);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_comp_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_context
//

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv) :
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw())),
    ctx_csa_mem(kv->get_csa()->init_full()),
    ctx_hca_mem(kv->get_hca()->init_full()),
    ctx_lid_mem(kv->get_lid()->init_full()),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_csa())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_hca())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_lid())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    reserve_plans(true),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        llama_context * lctx,
        bool optimize,
        stream_copy_info sc_info_csa,
        stream_copy_info sc_info_hca,
        stream_copy_info sc_info_lid) :
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw(), lctx, optimize)),
    ctx_csa_mem(kv->get_csa()->init_update(lctx, optimize)),
    ctx_hca_mem(kv->get_hca()->init_update(lctx, optimize)),
    ctx_lid_mem(kv->get_lid()->init_update(lctx, optimize)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    sc_info_csa(std::move(sc_info_csa)),
    sc_info_hca(std::move(sc_info_hca)),
    sc_info_lid(std::move(sc_info_lid)),
    status(llama_memory_status_combine(
                llama_memory_status_combine(
                    llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                    llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status())),
                this->sc_info_csa.empty() && this->sc_info_hca.empty() && this->sc_info_lid.empty() ?
                    LLAMA_MEMORY_STATUS_NO_UPDATE : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        slot_info_vec_t sinfos_raw_base_write,
        slot_info_vec_t sinfos_raw_swa_write,
        slot_info_vec_t sinfos_raw_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_raw) :
    ubatches(std::move(ubatches)),
    plans_csa(dsv4_build_comp_plans(this->ubatches, kv->get_csa_state()->get_ratio(), kv->get_csa_state()->get_overlap(),
                kv->get_csa_state()->get_state_size(), kv->get_csa()->get_size(), kv->get_csa_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    plans_hca(dsv4_build_comp_plans(this->ubatches, kv->get_hca_state()->get_ratio(), kv->get_hca_state()->get_overlap(),
                kv->get_hca_state()->get_state_size(), kv->get_hca()->get_size(), kv->get_hca_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    plans_lid(dsv4_build_comp_plans(this->ubatches, kv->get_lid_state()->get_ratio(), kv->get_lid_state()->get_overlap(),
                kv->get_lid_state()->get_state_size(), kv->get_lid()->get_size(), kv->get_lid_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_rs_idx())),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(
                kv->get_raw(),
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                this->ubatches,
                std::move(ubatches_raw),
                kv->get_raw_nkv_min_tokens())),
    ctx_csa_mem(nullptr),
    ctx_hca_mem(nullptr),
    ctx_lid_mem(nullptr),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_csa(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_csa()->get_n_stream()),
                this->ubatches)),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_hca(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_hca()->get_n_stream()),
                this->ubatches)),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_lid(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_lid()->get_n_stream()),
                this->ubatches)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(ctx_raw->get_status()) {
    this->kv = kv;
    kv->reset_rs_idx_for_ubatches(this->ubatches);
}

llama_kv_cache_dsv4_context::~llama_kv_cache_dsv4_context() = default;

bool llama_kv_cache_dsv4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_raw->next();
    ctx_csa->next();
    ctx_hca->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        // ranma: the last ubatch computed, so the call is complete (see tail_call_end)
        if (kv != nullptr && !ubatches.empty()) {
            kv->tail_call_end(ubatches, out_seq < 0 ? "no end contract" : out_has ? "output" : "no output");
        }
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_raw->apply();

    if (ctx_csa_mem) {
        res = res & ctx_csa_mem->apply();
        res = res & ctx_hca_mem->apply();
        res = res & ctx_lid_mem->apply();
    }

    if (ubatches.empty()) {
        csa_state->apply_copies(sc_info_csa);
        hca_state->apply_copies(sc_info_hca);
        lid_state->apply_copies(sc_info_lid);
    } else if (i_next == 0 && kv != nullptr) {
        // ranma: the call begins with its first ubatch
        kv->tail_call_begin(ubatches);
    }

    return res;
}

llama_memory_status llama_kv_cache_dsv4_context::get_status() const {
    return status;
}

bool llama_kv_cache_dsv4_context::get_batch_out(llama_seq_id & seq_id, llama_pos & pos, bool & has_output) const {
    seq_id     = out_seq;
    pos        = out_pos;
    has_output = out_has;
    return out_seq >= 0;
}

void llama_kv_cache_dsv4_context::set_batch_out(llama_seq_id seq_id, llama_pos pos, bool has_output) {
    out_seq = seq_id;
    out_pos = pos;
    out_has = has_output;
}

void llama_kv_cache_dsv4_context::set_raw_written(llama_seq_id seq_id, int32_t il, llama_pos p0) const {
    GGML_ASSERT(kv != nullptr);
    kv->tail_set_written(seq_id, il, p0);
}

const llama_ubatch & llama_kv_cache_dsv4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsv4_raw_context * llama_kv_cache_dsv4_context::get_raw() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_raw.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_csa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_csa.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_hca() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_hca.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_lid() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_lid.get();
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_csa_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return csa_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_hca_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return hca_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_lid_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return lid_state;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_csa.empty()) {
        return empty;
    }

    return plans_csa[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_hca.empty()) {
        return empty;
    }

    return plans_hca[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_lid.empty()) {
        return empty;
    }

    return plans_lid[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_csa_plan();
    }

    reserve_plan_csa = dsv4_build_reserve_comp_plan(
            ubatch, csa_state->get_ratio(), csa_state->get_overlap(),
            csa_state->get_state_size(), get_csa()->get_n_kv(), csa_state->get_n_stream(), csa_state->get_n_rs_seq());

    return reserve_plan_csa;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_hca_plan();
    }

    reserve_plan_hca = dsv4_build_reserve_comp_plan(
            ubatch, hca_state->get_ratio(), hca_state->get_overlap(),
            hca_state->get_state_size(), get_hca()->get_n_kv(), hca_state->get_n_stream(), hca_state->get_n_rs_seq());

    return reserve_plan_hca;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_lid_plan();
    }

    reserve_plan_lid = dsv4_build_reserve_comp_plan(
            ubatch, lid_state->get_ratio(), lid_state->get_overlap(),
            lid_state->get_state_size(), get_lid()->get_n_kv(), lid_state->get_n_stream(), lid_state->get_n_rs_seq());

    return reserve_plan_lid;
}
