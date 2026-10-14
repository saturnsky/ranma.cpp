#pragma once

// ranma: layer-major prefill (on for DeepSeek-V4 and V4.1 with the layer mode, llama_layer_mode_config).
//
// A logical batch of several ubatches normally runs every ubatch through the whole model, so the experts of every
// layer are read once per ubatch. Here every ubatch (a chunk) is applied to the memory and its graph built with its
// inputs set first, without computing it. Each chunk graph is then cut at the MoE of every layer (the ops from the
// router input to the MoE output, "the region"): segment s holds what runs after the MoE of layer s-1 and before the
// MoE of layer s. The layers run in order: segment l of every chunk in chunk order, then the MoE of layer l once over
// the rows of all chunks, taken from a graph of the whole width that the model builds with the reserve memory context.
// The tensors a segment hands to the next (the MoE input, the hyper-connection residual, ...) live in buffers owned
// here, one row range per chunk. The KV and compressor writes of a layer happen in chunk order, as with ubatches.
//
// Nothing in the model builders, ggml or the memory modules changes: the segments and the MoE are copies of the nodes
// of the built graphs, computed by the scheduler like any graph. Whatever the cut cannot classify falls back to the
// ubatch loop before the memory is touched.

#include "ggml.h"

#include <cstdint>
#include <set>
#include <string>
#include <vector>

struct ggml_cgraph;
struct ggml_tensor;
struct llama_model;

bool llama_layer_exec_enabled();

// RANMA_LAYER_EXEC_LOG=1: the plan and the time of every phase
int llama_layer_exec_log();

// DeepSeek-V4.1: the layers the layer-major prefill runs layer by layer, those before the first layer of the tail prefill
// (deepseek41.cpp), whose rows it never narrows; all of them when the tail is off. The layers from there on run in
// the last segment of every ubatch, ubatch by ubatch, as in the ubatch loop (deepseek41.cpp).
int llama_dsv41_layer_exec_layers(const llama_model & model);

// the names the graph callback gives the router input and the MoE output of layer il ("<name>-<il>")
struct llama_lx_names {
    const char * moe_in  = nullptr;
    const char * moe_out = nullptr;
    // states that a later segment may read after the segment that writes them: caches that never give a cell of a live
    // sequence to another position (DeepSeek-V4.1's compressed and index key caches, which the layers after a source read).
    // A later ubatch only adds cells for its own positions, which the masks of the earlier ubatches hide.
    const std::set<const ggml_tensor *> * appendable = nullptr;
};

// a tensor of segment s that a later segment (or the MoE of layer s) reads
struct llama_lx_edge {
    int           node = -1;  // index in the graph
    ggml_tensor * t    = nullptr;
    size_t        pt   = 0;   // bytes per token row
    int           s    = -1;  // the segment that writes it
    int           t_last = -1; // the last segment that reads it
    // the result of an in-place op (e.g. the last ggml_set of a chunk join) is a view of the node it writes into: that node
    // is placed in the buffer, and the writes into it fill it
    ggml_tensor * place_as = nullptr;
};

// the cut of a chunk graph
struct llama_lx_plan {
    int     n_layer  = 0;
    int64_t n_tokens = 0;

    std::vector<ggml_tensor *> x; // per layer: the router input
    std::vector<ggml_tensor *> y; // per layer: the MoE output

    // per segment (n_layer + 1): the nodes to copy, in graph order (its own, and the pure and view nodes they read)
    std::vector<std::vector<int>> nodes;
    // per segment: the tensors it stores for the next one, in graph order; the router input x[s] is one of them
    std::vector<std::vector<llama_lx_edge>> out;
    // the tensors read two or more segments after the one that writes them (e.g. the top-k an index source carries to
    // the layers after it), in graph order: each keeps its own buffer from its segment to the last one that reads it
    std::vector<llama_lx_edge> carry;
    // per segment: the index in out[s - 1] of each tensor it reads from the previous segment (by graph node)
    // true when every read of a segment's inputs precedes the writes of its outputs, so one buffer set serves all layers
    bool one_set = true;

    size_t input_bytes = 0;            // the inputs of the graph
    std::vector<size_t> seg_input_bytes; // per segment: the inputs it reads (a host input is copied for every segment)
    // the attention masks, per segment: the masks made on the device (nodes named "chunk_mask", see llm_graph_chunk_mask)
    // and the bytes of the host mask inputs it reads (inputs named "*kq_mask")
    std::vector<int>    seg_dev_masks;
    std::vector<size_t> seg_host_mask_bytes;
    size_t host_mask_bytes = 0;        // the host mask inputs any segment reads
    std::vector<const ggml_tensor *> inputs; // the inputs any node of the graph reads (what a scheduled graph allocates)
    std::string err;
};

// the cut of gf, whose MoE inputs and outputs are named by names; false (with plan.err) when it cannot be cut
bool llama_lx_analyze(ggml_cgraph * gf, int n_layer, const llama_lx_names & names, int64_t n_tokens, llama_lx_plan & plan);

// the MoE regions of a graph of the whole width
struct llama_lx_region {
    ggml_tensor *    x = nullptr;
    ggml_tensor *    y = nullptr;
    std::vector<int> nodes; // the nodes to copy, in graph order (the region and the pure nodes it reads)
};

bool llama_lx_regions(ggml_cgraph * gf, int n_layer, const llama_lx_names & names, std::vector<llama_lx_region> & out,
                      std::vector<ggml_tensor *> & inputs, std::string & err);
