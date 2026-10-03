#pragma once

#include "ggml.h"

#include <cstdint>
#include <unordered_map>

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

// the op that runs weight w, to select its buffer type: mul_mat (n_ids = 0) or mul_mat_id with n_ids experts per token
ggml_tensor * llama_weight_aux_test_op(ggml_context * ctx, ggml_tensor * w, int n_ids);
