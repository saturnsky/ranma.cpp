#pragma once

// ranma: the device buffers of one MOE_FFN_HAD launch (mul-mat-had.cu), as offsets into one block. The op takes them
// from this layout, and the layer mode (common/layer-mode.cpp) sizes the op's scratch with the same function, so the
// two cannot drift apart. Plain C++: no CUDA types.

#include <algorithm>
#include <cstddef>
#include <cstdint>

#define GGML_MOE_FFN_HAD_MAX_ITEMS 65535 // GEMM items of one pass (grid y)

// GEMM items of a pass of `rows` rows with tiles of `mt` rows: every expert that has rows, plus one per full tile
static inline int64_t ggml_moe_ffn_had_max_items(int64_t rows, int64_t n_exp, int mt) {
    return std::min(n_exp, rows) + rows/mt + 1;
}

struct ggml_moe_ffn_had_layout {
    // offsets in bytes, each 256-aligned
    size_t route, items, iota, xh, rscale, vu, vg, vy, lz;
    size_t bytes; // the whole block
};

// n_rows: tokens * top_k; R: rows per pass; c_max: bound of the passes (with the staging splits); stage_cap: staging
// slots of a buffer (0 = none); max_rank: largest LoRA rank of the op's weights (0 = none)
static inline ggml_moe_ffn_had_layout ggml_moe_ffn_had_layout_of(int64_t n_rows, int64_t R, int64_t n_exp, int64_t n_embd,
        int64_t n_ff, int64_t max_items_v, int64_t max_items_r, int64_t c_max, int64_t stage_cap, int64_t max_rank) {
    ggml_moe_ffn_had_layout l = {};
    size_t at = 0;
    auto take = [&at](int64_t bytes) {
        const size_t here = at;
        at += ((size_t) std::max<int64_t>(bytes, 1) + 255) & ~(size_t) 255;
        return here;
    };
    l.route  = take(4*(3*n_rows + 2*n_exp + 2*c_max + 1 + c_max*(stage_cap + 1)));
    l.items  = take(4*(3*(max_items_v + max_items_r) + 2));
    l.iota   = take(4*R);
    l.xh     = take(2*R*std::max(n_embd, n_ff));
    l.rscale = take(4*R);
    l.vu     = take(4*R*n_ff);
    l.vg     = take(4*R*n_ff);
    l.vy     = take(4*R*n_embd);
    l.lz     = take(max_rank > 0 ? 4*R*max_rank : 4);
    l.bytes  = at;
    return l;
}

// The largest block of a launch of `n_tokens` tokens in one pass (R = n_tokens * top_k, no staging), with the most
// GEMM items (tiles of 16 rows): what the layer mode reserves for the op.
static inline size_t ggml_moe_ffn_had_scratch_bytes(int64_t n_tokens, int64_t top_k, int64_t rows, int64_t n_exp,
        int64_t n_embd, int64_t n_ff, int64_t stage_cap, int64_t max_rank) {
    const int64_t n_rows = n_tokens*top_k;
    const int64_t R      = std::min(rows > 0 ? rows : n_rows, n_rows);
    const int64_t items  = ggml_moe_ffn_had_max_items(R, n_exp, 16);
    int64_t c_max = 3*((n_rows + R - 1)/R) + 1;
    if (stage_cap > 0) {
        c_max += n_rows/stage_cap + 1;
    }
    return ggml_moe_ffn_had_layout_of(n_rows, R, n_exp, n_embd, n_ff, items, items, c_max, stage_cap, max_rank).bytes;
}
