#pragma once

#include <cstdint>
#include <cstddef>

struct ggml_tensor;

// Debug harness for the Qwen sparse-attention (QSA) indexer.
//
// With LLAMA_QSA_DUMP=<path> set, every QSA layer writes one line per graph
// evaluation to <path>:
//
//   <step> <layer> <n_kv> <n_selected> <sha1 cells> <sha1 blocks>
//
// The hashes make two indexer implementations comparable without keeping the
// whole selection around. Each top-k row is sorted before hashing, so an
// implementation is free to return the same set in another order.
//
// The budget is whole blocks plus a tail, and every cell of a block carries its
// block's score, so the cut usually falls inside a group of exactly tied cells.
// Which of them a parallel top-k keeps is reproducible only with
// GGML_CUDA_TOP_K_STABLE_TIES=1. The second hash folds the cells onto their
// blocks and stays the implementation-independent comparison: that is the
// selection the indexer actually decides.
//
// Tensors named "indexer_dbg_<tag>-<layer>" are dumped as well, as comment lines
//
//   # dbg <step> <layer> <name> <bytes> <sha1 of the raw bytes>
//
// They are what makes an implementation change provable: the selection itself is
// tie-sensitive, while the pooled block keys behind it are not.
//
// Enabling the dump is not free of side effects: it changes the graph, because a
// ggml_cont is inserted before every named debug tensor, and it installs its own
// eval callback when the caller has set none.

bool llama_qsa_dump_enabled();

// records the context width the layer was built with, so the dumped line can name it
void llama_qsa_dump_set_n_kv(int il, int64_t n_kv);

// records the cell -> block map of the current ubatch, for the block-level hash
void llama_qsa_dump_set_cell_blk(const int32_t * cell_blk, int64_t n_kv, int64_t n_stream);

// ggml_backend_sched_eval_callback; observes the tensors named by build_qsa_top_k
bool llama_qsa_dump_eval_callback(ggml_tensor * t, bool ask, void * user_data);
