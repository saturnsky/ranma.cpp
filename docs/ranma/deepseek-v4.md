# DeepSeek V4

DeepSeek V4 Flash keeps, per attention layer, a sliding window of raw KV cells
together with a much smaller cache of compressed cells, and a lightning indexer
selects a few of the compressed cells for every query. This page describes what
this fork changes in that path: how the attention reads the selected cells, how
the graph is built around the hyper-connections and the KV compressor, and how
the two caches of a layer are laid out. The attention section is a HIP kernel
path and applies to RDNA4 (gfx1201) only; the other sections are backend
independent and have CPU and CUDA/HIP implementations.

## Gathering the selected cells for head size 512

### What it is

A CSA layer attends over the concatenation `[raw window | compressed cells]`.
Its mask leaves the cells of the sliding window and the indexer-selected
compressed cells finite and sets everything else to `-INFINITY`, so only a small
part of each mask row is ever used. The model tells the flash attention how small
by annotating the node with `n_kv_max`, the bound on the number of finite entries
per mask row; for a CSA layer that is the sliding window plus the indexer budget.

The HIP tile flash attention can use that bound: it compacts the mask of every
query tile into the ascending list of the cells any row of the tile can see and
then gathers only those K and V rows instead of scanning the whole cache. The
gathering variant is shared with the Qwen sparse attention and exists for every
head shape the tile kernel serves; for DeepSeek V4 Flash that is the 512/512
head shape with GQA 64, eight columns per block. This fork allows it for
attention calls that carry sinks, which the attention of this model does.

### When it applies

The gather replaces the dense kernel for a call only if all of the following
hold:

- the backend is HIP and the device has 32-wide waves,
- the node is annotated with a positive `n_kv_max` and has a mask,
- the tile holds at most four query rows: one in generation, four when a
  prompt is processed in batches,
- K and V are F16, so that the per-row gather does not have to go through the
  on-the-fly conversion,
- the call has neither ALiBi nor a logit softcap,
- the cache is long enough: at least the minimum cell count of the gate, and at
  least `2 * rows per tile * n_kv_max`, because a tile gathers the union of the
  lists of its rows.

Attention sinks do not prevent the gather. The sink correction acts on the
accumulated maximum and sum after the KV loop and does not depend on which cells
the loop visited.

### Switches

The gate is shared with the Qwen sparse-attention path and is described in
[qwen-sparse-attention.md](qwen-sparse-attention.md); its switches are repeated
here for reference.

| Switch | Default | Effect |
|---|---|---|
| `GGML_CUDA_FATTN_SPARSE=0` | on | Always use the dense tile kernel, in the same binary. The reference path for equivalence checks. |
| `GGML_CUDA_FATTN_SPARSE_MIN_KV=<cells>` | 4096 | Override the minimum cache length at which the gather is used. |
| `GGML_CUDA_FATTN_SPARSE_LOG=1` | off | Log the first shape that takes the gather (head size, query rows, GQA, budget, cell count). |

### Limits and fallbacks

- Short contexts keep the dense kernel: below the gate's cell count the dense
  kernel is already as cheap as compacting the mask. With the model's budget of
  640 a prompt-processing tile of four rows gathers from 5120 cells on, one row
  of generation from the minimum cell count on.
- `n_kv_max` is a budget, not a mask. The gather keeps at most that many cells
  per row; the model has to annotate a bound that is not smaller than the number
  of finite entries its mask actually contains.

### How to verify

- `test-backend-ops -o FLASH_ATTN_EXT` contains 512/512 cases with sinks: the
  model budget, a budget that is not a multiple of the KV tile, one and two
  sequences, the boundary of the cell gate, and tiles of two and four query
  rows that gather the union of their rows' lists.
- `GGML_CUDA_FATTN_SPARSE_LOG=1` confirms that a run actually takes the gather.
- Running the same prompt with `GGML_CUDA_FATTN_SPARSE=0` gives the dense result
  to compare against. The two are not bit-identical: the gather visits the same
  cells but accumulates them in a different order, so compare distributions
  rather than sampled text.

### What was checked

On a Radeon AI PRO R9700 (PCIe 5.0 x16) with DeepSeek V4 Flash UD-IQ3_XXS,
base revision ec5a12b85 with this patch, a 20480 MiB exclusive expert cache and
the MoE tensors in host memory.

`test-backend-ops -o FLASH_ATTN_EXT` passes with the default and with
`GGML_CUDA_FATTN_SPARSE=0`, and the 512/512 cases with sinks were confirmed to
gather at one, two and four query rows per tile.

Generation, teacher-forced: `llama-perplexity -b 1 -ub 1 -c 6144 --chunks 1`,
3072 scored tokens, with `GGML_CUDA_FATTN_SPARSE_MIN_KV=1280` so that the gate
opens at this length, against the dense kernel:

| run | mean KLD | same top token |
| --- | --- | --- |
| dense against dense | 0.000000 | 100 % |
| dense at two tokens per batch against dense | 0.0040 | 98.21 % |
| gather against dense | 0.0047 | 98.21 % |

The dense kernel reproduces itself exactly at this batch size, so the gather is
compared with the difference that a second batch size already makes.

Prompt processing: `llama-perplexity -b 512 -ub 512 -c 24576 --chunks 1`, 12288
scored tokens, default gate, so the four-row tiles gather wherever a layer
holds at least 5120 cells. Two dense processes do not reproduce each other
here: they differ by mean KLD 0.0032 with the same top token 98.36 %. The
gather against the first of them gives mean KLD 0.0017 with the same top token
99.19 %; the perplexity is 2.4771 against 2.4737 and 2.4709 for the two dense
runs.

`llama-bench -p 512 -n 128 -r 1 -d 32768`, a warm cache from one cold pass of
the same benchmark, one process per setting with an idle gap of five minutes
between them:

| setting | pp512 | tg128 |
| --- | --- | --- |
| `GGML_CUDA_FATTN_SPARSE=0` | 155.72 | 27.04 |
| default | 196.44 | 27.97 |

Generation gains 3.4 % and prompt processing 26 % at a depth of 32768, one run
per row.

## Hyper-connection coefficients in one kernel

### What it is

Every hyper-connection block of the model derives three things from one small
mixing matmul whose result `mixes` has `(2 + hc)*hc` rows per token, with
`hc = 4` streams: the pre coefficients that fold the streams into the block
input, the post coefficients that scale the block output, and the `hc x hc`
combination matrix that mixes the streams, which is normalized by a fixed
number of Sinkhorn iterations.

The pre and post coefficients were an affine transform, a sigmoid and a scale
on 4-element tensors, one chain each, while the combination matrix was already
computed by a single op from the same `mixes`, scale and base tensors.
`GGML_OP_DSV4_HC_COEF` computes all three at once. It takes the same three
sources and writes pre, post and comb into one `[(2 + hc)*hc, n_tokens]` tensor
in the row layout of `mixes`: the pre coefficients in the first `hc` rows, the
post coefficients in the next `hc`, the combination matrix in the remaining
`hc*hc`. The graph takes three views of that tensor and the rest of the block
is unchanged.

### Switches

| Switch | Default | Effect |
|---|---|---|
| `LLAMA_DSV4_HC_COEF_FUSED=0` | on | Build the separate coefficient ops and the combination op instead. The reference path for equivalence checks. |

### How the fallback is chosen

The op takes part in the fused-op probe that the context runs once before the
first batch: a reserve graph is built and every fused node is checked against
the device its layer is assigned to. If a node did not land on that device,
usually because the backend does not implement the op, the fusion is disabled
and the graph is built from the unfused ops. The coefficient fusion is resolved
before the combination fusion, because with the coefficient fusion off the
graph falls back to the separate combination op, which then has to be probed on
a graph that contains it.

Backends other than CPU and CUDA/HIP do not implement the op, so the probe
turns the fusion off there.

### Limits

- The op is written for four hyper-connection streams, the configuration of
  the published model, and asserts that.
- The fused kernel does not produce bit-identical results: the coefficients and
  the Sinkhorn iterations are computed in one pass with a different order of
  operations. The difference is of the same size as the difference between two
  micro-batch sizes of the unfused path.

### How to verify

- `test-backend-ops -o DSV4_HC_COEF` covers one and many tokens and one and
  several Sinkhorn iterations.
- The load log reports `fused DeepSeek V4 HC coefficients enabled` or, if the
  probe rejected the op, that it is disabled.
- `LLAMA_DSV4_HC_COEF_FUSED=0` gives the unfused graph for a comparison run.

## Fused KV compressor

### What it is

The compressed caches are filled by a compressor: when a block of `ratio` raw
cells is complete, the compressor reads the value and the score state of those
cells, takes a softmax over the block independently per feature, and writes the
weighted sum as one compressed cell. The model uses two variants: ratio 128 for
the HCA layers, and an overlapping ratio 4 for the CSA layers and the indexer,
where each block reads the previous window as well as the current one and takes
the low half of the features from the previous and the high half from the
current window.

That was a chain of two `get_rows`, permutes with `cont`, `soft_max`, `mul` and
`sum_rows`, and for the overlapping variant additional zero-row appends,
strided copies and concatenations. `GGML_OP_DSV4_COMPRESS` does the gather, the
per-feature softmax over the gathered rows and the weighted sum in one kernel.
It reads the value state, the score state and the row indices of the compress
plan, and writes one column per block. The norm and rope tail of the compressor
is unchanged.

### The missing-segment convention

The first block of a sequence has no previous window. Instead of appending a
zero row to the state and pointing at it, the plan writes an index outside the
state. The op treats any index that is not inside the state as a missing row:
its value reads as 0 and its score as `-INFINITY`, so it contributes nothing to
the softmax. The comparison is made on the unsigned value of the index, so a
negative index is out of range as well and cannot address memory in front of
the state.

### Switches

| Switch | Default | Effect |
|---|---|---|
| `LLAMA_DSV4_COMPRESSOR_FUSED=0` | on | Build the previous op chain instead. The reference path for equivalence checks. |

### Limits and fallbacks

- The op takes part in the same fused-op probe as the hyper-connection
  coefficients, so on a backend that does not implement it the graph falls back
  to the op chain.
- The CUDA/HIP implementation is selected only for F32 value and score states
  with a unit first stride and contiguous `I32` indices; anything else is left
  to the CPU implementation.
- Results are not bit-identical to the op chain: the softmax and the weighted
  sum are reduced in one pass in a different order.

### How to verify

- `test-backend-ops -o DSV4_COMPRESS` covers the shapes of the model (head size
  512 and 128, ratio 4 with overlap and ratio 128, a single block and a full
  batch of blocks), a head size that is not a multiple of the thread block, and
  the missing-segment sentinel.
- The load log reports whether the fused compressor is enabled.
- `LLAMA_DSV4_COMPRESSOR_FUSED=0` gives the op chain for a comparison run.

## No dummy compression when no HCA block completes

### What it is

An HCA block covers 128 raw cells, so in decode a block completes on one step
out of 128. To keep the shape of the graph the same on every step, a ubatch in
which no block completed still ran the whole compressor on a dummy block and
wrote the result to the masked last slot of the cache, where it could not be
read. The compress plan no longer appends that dummy block; the graph builder
already handles a plan without one, and the state update for the new token is
unchanged.

The graph shape therefore changes when a block completes and changes back
afterwards, so the graph is rebuilt twice per 128 tokens. That is much less
work than running the compressor and the cache write on every step.

The dummy blocks of the ratio-4 compressors stay: a block completes every
fourth token there, which is too short a period for graph reuse to pay.

### Switches

| Switch | Default | Effect |
|---|---|---|
| `LLAMA_DSV4_HCA_SKIP_DUMMY=0` | on | Keep the dummy block, and with it the fixed graph shape. The reference path for equivalence checks. |

### Limits

- Only the ratio-128 compressor is affected.
- The dummy block was written to the last cache slot, which is only safe while
  that slot is not live; the code that did so also asserted this. Dropping the
  write removes that dependency.

### How to verify

The dummy block never contributed to any result, so output is expected to be
unchanged, not merely close: a greedy continuation with the switch on and off
has to agree. Rebuilds show up in the `graphs reused` counter of the
performance summary.

## Raw and compressed K of a layer in one tensor

### What it is

A CSA or HCA layer attends over `[raw window | compressed cells]`. The two
halves live in two different caches, so the graph built that K with a concat,
which copied the raw window and every compressed cell of the layer into a new
tensor on every token; the copy grows with the context.

The raw cache and the compressed cache of such a layer now share one
allocation, the raw cells first and the compressed cells directly behind them.
Each cache receives a view of that allocation as the K storage of the layer, so
the attention input is a view of the joint tensor and no copy is needed. The
allocation is made before the sub-caches are built, one buffer per buffer type,
and it is reported in the memory breakdown.

Because the compressed cells start behind the whole raw cache, the raw part of
the attention input has a fixed size instead of the number of cells currently
in use. The raw cache reports that fixed size for all of its sliding-window
layers as soon as one layer uses the joint storage, so those layers attend over
the full window; the cells that are not in use are masked out as before, so the
result is the same.

The sub-caches see ordinary K tensors, which happen to be views. Cache writes,
the RoPE shift of a context shift, and session save and restore go through the
same code as without the joint layout and are unaffected.

### When the layout is used

All of the following have to hold, otherwise the graph keeps the concat:

- a single sequence: the raw prefix of a layer has a single well-defined stride
  only with one stream,
- the layer is a compressed layer (ratio 4 or ratio 128) that has a KV cache,
- the layer is a sliding-window layer of the raw cache: its raw K has to live
  in the sliding-window half. A compressed layer that is not sliding-window
  disables the joint layout for the whole cache,
- the raw and the compressed hparams of the layer agree on the row size and on
  the split of that row into heads, because the joint view is shaped from the
  compressed hparams and compared against a raw K.

If a sub-cache then asks for a K storage whose size does not match the joint
allocation, that is a bug and not a configuration, so it is reported as an
error rather than silently ignored.

### Switches

| Switch | Default | Effect |
|---|---|---|
| `LLAMA_DSV4_KALL_VIEW=0` | on | Keep separate tensors for the raw and the compressed cache and concatenate them in the graph. Read once, when the cache is created. |

### Limits and fallbacks

- The concat path stays in the graph and is taken whenever the joint view is
  not available for a layer, including the multi-sequence case.
- The joint allocation sizes the raw half like the sliding-window cache it
  replaces, so the memory footprint is the same as before; it is one buffer per
  layer instead of two.
- The output is unchanged: the same cells are read in the same order, only from
  one tensor instead of a copy of two.

### How to verify

- The load log lists the joint K buffer sizes and the number of layers that use
  the joint layout, or the reason why it is disabled.
- A greedy continuation with `LLAMA_DSV4_KALL_VIEW=0` and with the default has
  to agree token for token.
- Saving and restoring a session, and a context shift, exercise the paths that
  read the joint tensor through the sub-caches.

## Small-kernel fusions in decode

Several short chains of small kernels of every layer run as one launch each:
the persisted compressor state rows, the FFN sum that feeds the
hyper-connection post, the q, kv and compressor norm and rope including the
store into the K cache, and the compressor source concatenations; the
compressor APE add is taken by its matmul. The CUDA/HIP backend matches the
unfused path; a KL-divergence comparison against it stays at the measurement
floor.

| Switch | Default | Effect |
|---|---|---|
| `GGML_DSV4_FUSION3=0` | on | Keep the original graph order and kernels. The reference path for equivalence checks. |
| `GGML_DSV4_FUSION3_MASK=<bits>` | `0x3f` | Enable a subset: `0x01` APE add, `0x02` state rows, `0x04` FFN sum, `0x08` norm and rope, `0x10` concatenations, `0x20` K cache store (with `0x08`). |
