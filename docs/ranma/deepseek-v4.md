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

The HIP tile flash attention can use that bound: it compacts every mask row into
the ascending list of the cells it selects and then gathers only those K and V
rows instead of scanning the whole cache. This fork instantiates that gathering
variant for the 512/512 head shape with GQA 64 that DeepSeek V4 Flash uses in
decode, and allows it for attention calls that carry sinks.

### When it applies

The gather replaces the dense kernel for a call only if all of the following
hold:

- the backend is HIP and the device has 32-wide waves,
- the node is annotated with a positive `n_kv_max` and has a mask,
- the head shape is one the switch instantiates a gathering variant for, which
  for this model is 512/512 with eight columns per block and one query row,
- K and V are F16, so that the per-row gather does not have to go through the
  on-the-fly conversion,
- the call has neither ALiBi nor a logit softcap,
- the cache is long enough: at least the minimum cell count of the gate, and at
  least twice `n_kv_max`.

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

- Prefill and short contexts keep the dense kernel: with more than one query row
  per block the shape does not qualify, and below the minimum cell count the
  dense kernel is already as cheap as compacting the mask.
- `n_kv_max` is a budget, not a mask. The gather keeps at most that many cells
  per row; the model has to annotate a bound that is not smaller than the number
  of finite entries its mask actually contains.
- Only head shapes listed in the predicate are instantiated, because every entry
  costs compile time and code size. Any other shape takes the dense kernel.

### How to verify

- `test-backend-ops -o FLASH_ATTN_EXT` contains 512/512 cases with sinks: the
  model budget, a budget that is not a multiple of the KV tile, one and two
  sequences, the boundary of the cell gate, and a two-row case that has to fall
  back to the dense path.
- `GGML_CUDA_FATTN_SPARSE_LOG=1` confirms that a run actually takes the gather.
- Running the same prompt with `GGML_CUDA_FATTN_SPARSE=0` gives the dense result
  to compare against. The two are not bit-identical: the gather visits the same
  cells but accumulates them in a different order, so compare distributions
  rather than sampled text.

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
