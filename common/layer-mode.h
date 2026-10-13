#pragma once

// ranma: the layer compute mode (--layer-mode, docs/ranma/layer-mode.md). A prompt step is one wide ubatch computed
// layer by layer: the routed experts of a layer expert by expert over the whole width (the expert-major MoE op), the
// attention in chunks of query rows, so that the experts outside VRAM cross the link once per layer
// for many tokens. The mode sizes the width, the chunk and the buffers itself; off leaves everything as upstream.

#include "common.h"

// Decides the layer mode settings and applies them to params and the process-wide llama_layer_mode_config
// (src/llama-ext.h: the expert-major MoE op, the attention chunk and, for DeepSeek-V4 and V4.1, the layer-major
// prefill). Call before the model
// loads (common_init_result does). Sets n_ubatch and n_batch (unless -ub/-b were given), the
// host-direct token limit, the part of --expert-l1-mib its buffers take and the prompt batch time of the other slots.
// The width and the chunk are checked against the free memory with no_alloc loads of the model (the fit's own
// measurement). An unsupported model or setup, or a configuration where no width fits, logs one warning and leaves
// params and the config as they were (layer_mode becomes 0). Not thread safe: it sets the config and replaces the
// llama logger during the measurement.
void common_layer_mode_apply(common_params & params);
