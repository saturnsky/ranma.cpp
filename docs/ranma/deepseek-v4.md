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
