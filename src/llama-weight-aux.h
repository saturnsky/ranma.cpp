#pragma once

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model_loader;

// tensors that a weight needs beside its own data
// EXL3 weights (ggml_is_exl3) are used with ggml_mul_mat_had and need <base>.rot_in and <base>.rot_out
struct llama_weight_aux {
    ggml_tensor * rot_in  = nullptr; // [ne0(, ne2)]
    ggml_tensor * rot_out = nullptr; // [ne1(, ne2)]

    // logical in/out size: the weight can be stored padded to a multiple of 128 (ne0 >= k, ne1 >= n)
    int64_t k = 0;
    int64_t n = 0;
};

// key: the weight tensor of the model
using llama_weight_aux_map = std::unordered_map<const ggml_tensor *, llama_weight_aux>;

// stored size of an EXL3 dim of logical size n
static inline int64_t llama_weight_aux_pad(int64_t n) {
    return (n + 127) / 128 * 128;
}

// w * cur (ggml_mul_mat), or with ids the expert product (ggml_mul_mat_id), for a weight that needs aux tensors
// w can also be a view of whole 128-row groups of such a weight
// returns nullptr if w needs no aux tensors; aborts for an EXL3 weight that is not in waux
ggml_tensor * llama_weight_aux_mul_mat(
        ggml_context               * ctx,
        const llama_weight_aux_map * waux,
        ggml_tensor                * w,
        ggml_tensor                * cur,
        ggml_tensor                * ids);

// rows of an embedding table that holds n_heads row ranges: ggml_get_rows(table, rows), rows = I32 [n_heads * n_tokens]
// with the head as the fastest index; returns F32 [ne0, n_heads * n_tokens]
// an EXL3 row codec table (ggml_is_exl3_row) needs bias = [ne0, n_heads]: its rows are the official reconstruction
// fp16(row + bias[head]); any other table takes bias = nullptr and is gathered as is
ggml_tensor * llama_weight_aux_get_rows(
        ggml_context * ctx,
        ggml_tensor  * table,
        ggml_tensor  * bias,
        ggml_tensor  * rows,
        int64_t        n_heads);

// An engram table (DeepSeek V4.1 engram, Qwen3.8 PLE) stored as pieces of consecutive rows, one tensor per piece
// (<table>.p<i>.weight, from the table set of the model, docs/ranma/engram-tables.md). A table
// stored as one tensor of the model (<table>.weight) is a single piece (parts.size() == 1, part_rows == rows).
struct llama_table_pieces {
    std::vector<ggml_tensor *> parts; // piece i holds the rows [i*part_rows, min((i+1)*part_rows, rows))
    int64_t part_rows = 0;
    int64_t rows      = 0;

    size_t size() const { return parts.size(); }
    bool   empty() const { return parts.empty(); }
};

// The pieces of the engram table `name` (e.g. blk.1.engram_embd) from the engram_tables.* layout keys (those of the table set,
// which the loader copies into the model metadata); parts == 0 when the model has no table set (the table is the one tensor
// <name>.weight of the model). Throws when the table set does not list `name`.
struct llama_table_layout {
    uint32_t parts     = 0;
    int64_t  rows      = 0;
    int64_t  part_rows = 0;
};

llama_table_layout llama_table_layout_get(llama_model_loader & ml, const std::string & name);

// Create the pieces that the layout (llama_table_layout_get) describes, or the single tensor of a model without a table set
// (parts == 0: create(0, rows, dim) is called once). piece_name(i) is the name of piece i; the loader checks that every
// piece exists, has `dim` values per row, the row count of the layout and the same type as the others.
// create(i, rows_of_piece, dim) creates the tensor.
llama_table_pieces llama_table_pieces_create(
        llama_model_loader                                                     & ml,
        const std::string                                                      & what,
        uint32_t                                                                 parts,
        int64_t                                                                  rows,
        int64_t                                                                  part_rows,
        int64_t                                                                  dim,
        const std::function<std::string(uint32_t)>                            & piece_name,
        const std::function<ggml_tensor *(uint32_t, int64_t, int64_t)>         & create);

// host side of a gather over pieces: global row numbers -> per-piece local row numbers (piece-major, parts.size() * n
// entries) and the 0/1 mask that keeps, for each row, the piece that holds it. A row outside a piece reads that piece's
// local row 0 (always inside it) and is masked out, so the shapes of the graph do not depend on the rows. A table of one
// piece gets idx = rows and no mask.
void llama_table_pieces_split(const llama_table_pieces & t, const std::vector<int32_t> & rows,
        std::vector<int32_t> & idx, std::vector<float> & mask);

// llama_weight_aux_get_rows over a table in pieces: rows = I32 [parts.size() * n], mask = F32 [1, parts.size() * n] (both
// from llama_table_pieces_split; mask is null for one piece, then rows = I32 [n]); returns F32 [ne0, n]
ggml_tensor * llama_weight_aux_get_rows_pieces(
        ggml_context             * ctx,
        const llama_table_pieces & table,
        ggml_tensor              * bias,
        ggml_tensor              * rows,
        ggml_tensor              * mask,
        int64_t                    n_heads);

// the op that runs weight w, to select its buffer type: mul_mat (n_ids = 0) or mul_mat_id with n_ids experts per token
ggml_tensor * llama_weight_aux_test_op(ggml_context * ctx, ggml_tensor * w, int n_ids);
