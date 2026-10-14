#pragma once

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class llama_dsv4_comp_state {
public:
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    stream_copy_info sc_info;

    llama_dsv4_comp_state(
            const llama_model & model,
            bool            offload,
            bool            unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
                bool        overlap,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            uint32_t        n_rs_seq,
            const char    * name,
        const llama_memory_i::layer_filter_cb & filter);

    void clear(llama_seq_id seq_id, bool data);
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst);
    void apply_copies(const stream_copy_info & sc_info) const;

    uint32_t get_ratio()      const;
    bool     get_overlap()    const;
    uint32_t get_state_size() const;
    uint32_t get_n_stream()   const;
    uint32_t get_n_rs_seq()   const;
    uint32_t get_n_rows()     const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, const std::vector<uint32_t> & rs_idx) const;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    ggml_tensor * get_kv       (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score    (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_kv_all   (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score_all(ggml_context * ctx, int32_t il) const;

    ggml_tensor * cpy_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;

private:
    struct layer {
        uint32_t il;

        ggml_tensor * kv;
        ggml_tensor * score;

        std::vector<ggml_tensor *> kv_stream;
        std::vector<ggml_tensor *> score_stream;
    };

    const uint32_t ratio;
    // whether a pooled group overlaps the previous one; V4's CSA does, V4.1's compressor does not
    const bool     overlap;
    const uint32_t state_size;
    const uint32_t n_embd_state;
    const uint32_t n_stream;
    const uint32_t n_rs_seq;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    std::vector<layer> layers;

    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;
};

//
// llama_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by llama_kv_cache_dsv4_context / llm_graph_input_dsv4.
// FIXME: currently the cache only supports non-unified mode even if unified flag is passed
// FIXME: we currently conflate token_pos and buffer contents. See https://github.com/ggml-org/llama.cpp/pull/25521#discussion_r3558173819

class llama_kv_cache_dsv4 : public llama_memory_i {
public:
    llama_kv_cache_dsv4(
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
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_dsv4() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    // ranma: with the tail prefill (DeepSeek-V4.1), true when the raw cells the next position reads are written and the
    // compressed state can be rolled back to pos (no rollback at seq_pos_max + 1)
    bool seq_can_resume(llama_seq_id seq_id, llama_pos pos) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsv4 specific API
    //

    llama_kv_cache_iswa * get_raw() const;
    llama_kv_cache      * get_csa() const;
    llama_kv_cache      * get_hca() const;
    llama_kv_cache      * get_lid() const;
    llama_dsv4_comp_state * get_csa_state() const;
    llama_dsv4_comp_state * get_hca_state() const;
    llama_dsv4_comp_state * get_lid_state() const;

    uint32_t get_n_rs_seq() const;
    const std::vector<uint32_t> & get_rs_idx() const;
    void reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches);

    // ranma: the smallest ubatch that reads the narrow raw window; a smaller one reads the whole raw cache
    uint32_t get_raw_nkv_min_tokens() const;

    // ranma: the tail prefill of DeepSeek-V4.1 writes the raw KV of a decoder layer only for
    // the rows that the output of its batch depends on. Per sequence and layer, the first position of the run of
    // written raw cells that ends at the last position of the sequence; a sequence is continuable at p_end when every
    // layer has its sliding window before p_end + 1 written. A removal, a decode or a state save that would read an
    // unwritten cell is refused, and a refused removal marks the sequence broken until it is cleared or restored.
    void tail_set_written(llama_seq_id seq_id, int32_t il, llama_pos p0);
    bool tail_can_continue(llama_seq_id seq_id, llama_pos p_end) const;

    // the compressed state of the sequence can go back to p0 > 0: nothing to undo past its end, else a rollback of at
    // most n_rs_seq positions while none is pending. Independent of the raw cells (tail_can_continue).
    bool comp_can_rollback(llama_seq_id seq_id, llama_pos p0) const;

    // ranma: the calls of the sequences of a batch begin with the first ubatch applied and end once the last one
    // computed; a call that never ends (an aborted or failed ubatch) leaves its sequences pending
    void tail_call_begin(const std::vector<llama_ubatch> & ubatches);
    void tail_call_end(const std::vector<llama_ubatch> & ubatches, const char * what);

private:
    bool     tail_v41     = false;                // DeepSeek-V4.1, which has the tail prefill
    uint32_t tail_n_swa   = 0;
    uint32_t tail_n_layer = 0;
    bool     tail_seen    = false;                // a tail ran on this cache, so a restored state is not trusted further back
    std::vector<std::vector<llama_pos>> tail_lo;  // [n_seq_max][n_layer], empty: every cell written
    std::vector<uint8_t>                tail_bad; // [n_seq_max] a refused removal left unknown cells
    std::vector<llama_pos>              tail_end; // [n_seq_max] the end of the last call that completed, -1 none
    std::vector<uint8_t>                tail_pending; // [n_seq_max] a call began and has not completed

    void tail_reset(llama_seq_id seq_id);         // -1: every sequence

    // the record of a sequence in a saved state (DSV4_STATE_MODE_TAIL): its end, and the first written position of
    // each layer (empty: every cell written)
    struct tail_state_rec {
        llama_seq_id           seq_id = -1;
        llama_pos              end    = 0;
        std::vector<llama_pos> lo;
    };

    void tail_state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    std::vector<tail_state_rec> tail_state_read(llama_io_read_i & io) const;

    // -1: every sequence. recs: the records of the state, if it has them; partial: the state had no global payload, so
    // the memory had to hold the prefix up to its end before (pos_max_before, per sequence)
    void tail_after_restore(llama_seq_id seq_id, const std::vector<tail_state_rec> * recs, bool partial,
            const std::vector<llama_pos> & pos_max_before);
    bool tail_check_batch(const llama_batch_allocr & balloc) const;

    // diagnostics, logged when the tail prefill is asked for: the cells a sequence has written, and a batch with its
    // end contract and the record of its sequences
    std::string tail_describe(llama_seq_id seq_id) const;
    void tail_log_batch(const llama_batch_allocr & balloc, llama_seq_id out_seq, llama_pos out_pos, bool out_has) const;

    // joint raw + compressed K storage, allocated before the sub-caches so that they can view it
    struct kall_layer {
        ggml_tensor * joint    = nullptr;
        ggml_tensor * view_raw = nullptr;
        ggml_tensor * view_cmp = nullptr;

        // set once the raw cache has accepted the joint storage for this layer
        bool raw_ok = false;
    };

    uint32_t kall_raw_size = 0;

    uint32_t raw_nkv_min_tokens = 0;

    void init_kall(
            const llama_model & model,
                    ggml_type   type_k,
                         bool   offload,
                         bool   swa_full,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch);

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs_kall;
    std::unordered_map<int32_t, kall_layer> kall_layers;

    llama_hparams hparams_raw;
    llama_hparams hparams_csa;
    llama_hparams hparams_hca;
    llama_hparams hparams_lid;

    const uint32_t n_seq_max;
    const uint32_t n_rs_seq;

    std::vector<uint32_t> rs_idx;

    // positions per row of the compressed K caches, for the rows a sequence state saves: the CSA and HCA ratios on
    // DeepSeek-V4, the finest ratio of the shared caches on V4.1 (a coarser source uses fewer of its rows)
    uint32_t state_ratio_csa = 4;
    uint32_t state_ratio_hca = 128;
    uint32_t state_ratio_lid = 4;

    std::unique_ptr<llama_kv_cache_iswa> kv_raw;
    std::unique_ptr<llama_kv_cache>      kv_csa;
    std::unique_ptr<llama_kv_cache>      kv_hca;
    std::unique_ptr<llama_kv_cache>      kv_lid;
    std::unique_ptr<llama_dsv4_comp_state> csa_state;
    std::unique_ptr<llama_dsv4_comp_state> hca_state;
    std::unique_ptr<llama_dsv4_comp_state> lid_state;

    void clear_compressed(llama_seq_id seq_id, bool data);
};

// DSV4 raw attention only uses the SWA half of kv_raw. The base half is kept
// for generic ISWA bookkeeping, but it has no DSV4 layers to expose here.
class llama_kv_cache_dsv4_raw_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base_write,
            slot_info_vec_t sinfos_swa_write,
            slot_info_vec_t sinfos_swa_read,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_ubatch> ubatches_write,
            uint32_t n_min_tokens_window);

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // the raw cells attention reads are [get_k_lo(), get_k_lo() + get_n_kv()): get_k, the masks and the joint view
    uint32_t get_n_kv() const;
    uint32_t get_k_lo() const;
    uint32_t get_n_write() const;

    // ranma: the raw cells [lo, lo + n) of [get_k_lo(), get_k_lo() + get_n_kv()) that the tokens [t0, t0 + n_tok) of the
    // current ubatch can see under the causal mask, on 256 cells (a chunk of query rows of a chunked graph,
    // the raw chunk); every other cell is masked for each of those tokens. The whole window when this cannot be
    // told cheaply (several streams, no visible cell, a context without a ubatch).
    void get_chunk_window(uint32_t t0, uint32_t n_tok, uint32_t & lo, uint32_t & n) const;

    // ranma: the context applied a ubatch, so its cells and plans are those of a real ubatch (false for the reserve
    // context, whose graph has to be sized at the widest)
    bool has_ubatch() const { return !ubatches_write.empty(); }

    struct chunk_window {
        uint32_t t0, n_tok; // the token rows of the chunk
        uint32_t lo, n;     // its raw cells
    };

    // RANMA_DSV4_RAW_NKV_CHECK=1: aborts unless every cell outside [lo, lo + n) is -inf for the tokens [t0, t0 + n_tok)
    // of each chunk in the causal mask of the ubatch over the whole raw cache (a host buffer of n_tokens x raw cache
    // cells per ubatch; for correctness runs only)
    void check_chunk_windows(const llama_ubatch * ubatch, const std::vector<chunk_window> & windows) const;
    static bool check_enabled();

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_kq_pos (ggml_tensor * dst, const llama_ubatch * ubatch) const; // ranma: see llama_kv_cache::set_input_kq_pos
    void set_input_k_rot(ggml_tensor * dst) const;

    // see llama_kv_cache::get_prev_tokens(); the engram n-gram hash needs the preceding tokens
    void get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const;

private:
    size_t i_next = 0;

    llama_kv_cache * kv_swa = nullptr;

    slot_info_vec_t sinfos_write;
    slot_info_vec_t sinfos_read;
    std::vector<llama_ubatch> ubatches;
    std::vector<llama_ubatch> ubatches_write;

    const llama_memory_context_ptr ctx_base_mem;
    const llama_memory_context_ptr ctx_swa_mem;

    uint32_t n_kv = 0;
    uint32_t k_lo = 0; // ranma: first raw cell attention reads (the visible raw cells)

    const uint32_t n_min_tokens_window = 0;

    const llama_memory_status status;
};

// DSV4 compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class llama_kv_cache_dsv4_comp_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv);

    llama_kv_cache_dsv4_comp_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    bool next();

    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    // view of the raw cells [raw_lo, raw_lo + n_raw) followed by the first n_comp compressed cells of the joint
    // raw/compressed K storage of the layer. returns nullptr when the layer has no joint storage or when the raw cells
    // do not end where the compressed ones start (then the caller has to concat)
    ggml_tensor * get_k_joint(ggml_context * ctx, int32_t il, uint32_t raw_lo, uint32_t n_raw, uint32_t n_comp) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    llama_kv_cache * kv;

    size_t i_cur = 0;
    slot_info_vec_t sinfos;
    std::vector<llama_ubatch> ubatches;

    uint32_t n_kv;
};

class llama_kv_cache_dsv4_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    struct comp_plan {
        // Per-ubatch recipe for updating compressor state, committing completed
        // compressed rows, and masking the compressed attention source.

        // APE row ids, i.e. pos % ratio, for the compressor-state updates.
        std::vector<int32_t> state_pos;

        // Current-ubatch source row ids and unique persistent-state
        // destination row ids for deterministic ring-state updates.
        std::vector<int32_t> state_persist_src_idxs;
        std::vector<int32_t> state_persist_dst_idxs;

        // Device-side rollback restore copies snapshot planes back to the
        // current compressor-state plane before the graph reads it.
        std::vector<int32_t> state_restore_src_idxs;
        std::vector<int32_t> state_restore_dst_idxs;

        // Device-side rollback snapshots copy rows from the graph-local
        // [persistent_state | current_ubatch_scratch] tensor into rollback
        // planes after the graph has computed current-token compressor state.
        std::vector<int32_t> state_snapshot_src_idxs;
        std::vector<int32_t> state_snapshot_dst_idxs;

        // Flattened source row ids used for state-backed commits. Source rows
        // index the graph-local [persistent_state | current_ubatch_scratch]
        // tensor. For overlapped compression the first half is previous rows
        // and the second half is current rows; a final synthetic zero/-inf row
        // may be addressed for the first block's previous half.
        std::vector<int32_t> state_read_idxs;

        // Final compressed-cache row ids written by state-backed commits.
        // A non-boundary CSA/LID decode step can target a masked scratch row.
        std::vector<int64_t> state_write_idxs;

        // RoPE positions for state-backed commits.
        std::vector<int32_t> state_write_pos;

        // Number of completed compressed rows visible for each query token.
        std::vector<int32_t> n_visible;

        // Number of streams used by the attention graph for this ubatch.
        int64_t n_stream = 1;

        // Graph-width for compressed rows. This can be larger than n_visible
        // so masked padding rows do not force a new graph at every CSA block.
        int64_t n_kv = 0;
    };

    llama_kv_cache_dsv4_context(llama_memory_status status);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            llama_context * lctx,
            bool optimize,
            stream_copy_info sc_info_csa,
            stream_copy_info sc_info_hca,
            stream_copy_info sc_info_lid);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            slot_info_vec_t sinfos_raw_base_write,
            slot_info_vec_t sinfos_raw_swa_write,
            slot_info_vec_t sinfos_raw_swa_read,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_ubatch> ubatches_raw);

    virtual ~llama_kv_cache_dsv4_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_dsv4_context specific API
    //

    const llama_kv_cache_dsv4_raw_context * get_raw() const;
    const llama_kv_cache_dsv4_comp_context * get_csa() const;
    const llama_kv_cache_dsv4_comp_context * get_hca() const;
    const llama_kv_cache_dsv4_comp_context * get_lid() const;
    const llama_dsv4_comp_state       * get_csa_state() const;
    const llama_dsv4_comp_state       * get_hca_state() const;
    const llama_dsv4_comp_state       * get_lid_state() const;

    const comp_plan & get_csa_plan() const;
    const comp_plan & get_hca_plan() const;
    const comp_plan & get_lid_plan() const;

    const comp_plan & get_csa_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_hca_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_lid_plan(const llama_ubatch & ubatch) const;

    // ranma: the tail prefill. The end contract of a batch of one sequence of tokens at consecutive positions whose
    // only output, if any, is its last token: the sequence, the position of the last token and whether the batch asks
    // for its output, else false. The ubatches of such a batch can compute only what the state at its end depends on.
    bool get_batch_out(llama_seq_id & seq_id, llama_pos & pos, bool & has_output) const;
    void set_batch_out(llama_seq_id seq_id, llama_pos pos, bool has_output);

    // the graph of the current ubatch writes the raw KV of layer il for the positions from p0 on (see
    // llama_kv_cache_dsv4::tail_set_written)
    void set_raw_written(llama_seq_id seq_id, int32_t il, llama_pos p0) const;

private:
    size_t i_next = 0;

    llama_kv_cache_dsv4 * kv = nullptr;

    llama_seq_id out_seq = -1;
    llama_pos    out_pos = -1;
    bool         out_has = false;

    std::vector<llama_ubatch> ubatches;

    std::vector<comp_plan> plans_csa;
    std::vector<comp_plan> plans_hca;
    std::vector<comp_plan> plans_lid;

    const std::unique_ptr<llama_kv_cache_dsv4_raw_context> ctx_raw;
    const llama_memory_context_ptr ctx_csa_mem;
    const llama_memory_context_ptr ctx_hca_mem;
    const llama_memory_context_ptr ctx_lid_mem;

    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_csa;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_hca;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_lid;

    llama_dsv4_comp_state * csa_state = nullptr;
    llama_dsv4_comp_state * hca_state = nullptr;
    llama_dsv4_comp_state * lid_state = nullptr;

    stream_copy_info sc_info_csa;
    stream_copy_info sc_info_hca;
    stream_copy_info sc_info_lid;

    bool reserve_plans = false;
    mutable comp_plan reserve_plan_csa;
    mutable comp_plan reserve_plan_hca;
    mutable comp_plan reserve_plan_lid;

    const llama_memory_status status;
};
