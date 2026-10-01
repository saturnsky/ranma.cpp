# Qwen sparse attention

The Qwen sparse-attention layers attend to a subset of the KV cache. A small
indexer runs in front of each of them: the cells of a sequence are grouped, in
sequence order, into pools of `ratio` cells, and one key represents each
complete pool - the mean of the raw indexer keys of its cells, normalized and
rotated to the position of its first cell. The query scores every pool it can
see, a top-k over those scores picks `indexer_top_k / ratio` pools, and the
attention layer may read the cells of the picked pools and those of the query's
own incomplete pool (at most `ratio - 1`). Everything outside the selection is
masked away.

Two properties of that design shape everything on this page. The selection is
what the indexer decides, so an implementation is correct when it selects the
same pools; and a pool score is a sum of rectified head scores, so every pool
that no head scores positively ties at exactly zero, and the selection budget
can cut through such a group of tied values.

## Switches

All of them are read once per process. The default of every one of them is the
behaviour described in the sections below; the others select a reference path
for equivalence checks, or a diagnostic.

| name | default | effect |
| --- | --- | --- |
| `GGML_CUDA_TOP_K_STABLE_TIES` | off | `1` makes the top-k pick the smallest columns among tied values, so selections and outputs can be compared between runs. Option for reproducible runs. |
| `GGML_CUDA_FATTN_SPARSE` | on | `0` keeps the dense flash-attention kernel. Reference path. |
| `GGML_CUDA_FATTN_SPARSE_MIN_KV` | 4096 | the floor of the cell count at which the gather is used. |

## Tie-breaking in the top-k

Which of the tied values a top-k keeps is unspecified: `ggml_top_k` makes no
promise about it, and the radix implementation on CUDA and HIP picks a value
threshold and then gathers in whatever order its atomics happen to run. Ties
are common wherever scores are rectified, since those collapse to exactly zero,
as the pool scores here do. Two runs of the same build can therefore select
different pools, which is valid but makes selections and outputs hard to
compare.

`GGML_CUDA_TOP_K_STABLE_TIES=1` fixes the choice: a second radix walk, over
the column index and from the low bin upwards, finds the largest column index
that still fits in the budget, and only the tied columns at or below it are
kept. The selected set is then fully determined by the input; the order within
the reserved slots is not, and callers do not depend on it. The extra walk
covers only the bits a column index needs, so it is two passes for up to 64k
columns rather than four.

The switch is off by default: nothing about quality prefers one order among
equal scores, so normal use keeps the behaviour of `ggml_top_k` and does not
pay for the extra passes. It is an option for reproducible runs, and it
applies to every `ggml_top_k` that the backend runs through the radix
selection - the indexers of other sparse-attention models included - not only
to this graph. The expert
routing of a mixture-of-experts model uses an argsort and is not affected.
With the switch unset, the only permanent change is one extra comparison in
the gather.

`test-backend-ops -o TOP_K` covers both settings.

## Attending only to the selected cells

The selection leaves a bounded number of finite entries in each mask row, but
attention still read every cell of the KV cache and discarded the rest through
the mask, so its traffic followed the context depth instead of the budget.

### The index lists

The Qwen graph passes the width of its selection to the flash-attention node
as a per-row bound (`n_kv_max`): the cells of the picked pools plus the slots
of the incomplete pool, `ratio * min(n_pool, indexer_top_k / ratio) + ratio -
1`, where `n_pool` is the pool count of the graph, padded to a multiple of 64.
When the bound is set, the backend compacts the mask ahead of the attention
kernel and hands the kernel index lists instead of the per-tile cell bound it
otherwise gets.

A list belongs to a query tile, not to a query. A tile of `ncols1` query rows
gets one list, the union of the cells its rows can see, and a live count that
says how many of the `ncols1*bound` slots the union actually filled; the
counts of all lists sit packed behind the lists in the same buffer. Slots past
the count hold -1 and are dropped. The compaction kernel writes one list per
(sequence, tile) pair, and the attention kernel addresses it as
`(sequence % ne33)*n_tiles + tile`, so several sequences in one graph each read
their own lists.

The union is safe because the kernel still reads the mask at the physical cell
of each list entry, once per query column. A cell that only a sibling row of
the tile can see arrives masked for this row and contributes nothing, exactly
as in the dense kernel. The attended set and the arithmetic are those of the
dense masked kernel; only the memory traffic changes.

### Where the gather is taken

The tile kernel is the one RDNA4 uses for this model, and it takes the gather
when all of the following hold:

- the per-row bound is set;
- K and V are F16 - single rows are read, so they cannot go through the
  on-the-fly conversion;
- no ALiBi, no logit softcap, no attention sink;
- the mask covers the whole cache contiguously and has a single head;
- the wave is 32 lanes wide;
- the tile holds at most four query rows;
- the cache holds at least `max(4096, 2*ncols1*bound)` cells - below that the
  dense kernel is already as cheap as compacting the mask.

While the bound is not below the number of cells, as in a short context, the
last condition keeps the dense kernel.

The tile width is not a choice of this path: it is what the column switch
picks for the batch. One query row per tile is the GQA-12 generation block,
and two and four rows come out of the generic switch for the small batches a
speculative or multi-slot step produces. Wider tiles keep the dense kernel,
and head size 256 with eight or more query rows does not reach the tile kernel
at all on this device.

The same index lists are what the NVIDIA MMA kernel consumes upstream. The
WMMA kernel of this backend is left on the dense path: the gate above opens at
32k cells, and at that depth a measurement of the single-request Q8 WMMA shape
gave 0.85x of the dense kernel, with an improvement only from 48k on. Turning
it on would need its own threshold, which is not part of this change.

The compaction votes through a ballot that does not assume a wave width, so
the kernel builds for both backends.

| switch | default | effect |
| --- | --- | --- |
| `GGML_CUDA_FATTN_SPARSE` | on | `0` forces the dense kernel in the same binary |
| `GGML_CUDA_FATTN_SPARSE_MIN_KV` | 4096 | replaces the 4096 of the bound above |

### What was checked

`test-backend-ops -o FLASH_ATTN_EXT` carries the generation shape at one, two
and four query rows per tile, with one and with two sequences, at cell counts
where the gate opens and with a budget far below the cell count so that the
lists are short.

The measurements below were taken with this change as part of the series
released as ranma_20261001, then on upstream revision ec5a12b85. That series
used the Qwen indexer upstream had before pull request #29751, with this fork's
changes to it at the time, so the cells it selected differ from those of the
k-pool indexer described above; the attention kernel path is the same. They
have not been repeated with the current indexer.

On a Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL, `llama-bench
-ncmoe 35 -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -p 512 -n 128
-r 1 -d 0,8192,32768,65536`, MoE tensors placed in host memory, one run per row
with an idle gap between runs:

| cells | dense tg128 | gathering tg128 |
| --- | --- | --- |
| 0 | 25.20 / 26.22 | 26.24 |
| 8192 | 25.23 / 25.77 | 25.79 |
| 32768 | 24.44 / 24.48 | 24.98 |
| 65536 | 23.09 / 23.06 | 24.30 |

The dense column holds two runs, because the first run of a session reads the
host-resident tensors cold and is slower than every later one; the two shallow
rows are where that shows, and the gate is shut there anyway, so both paths run
the same code. Where the gate is open the gather is worth 2.0% at 32k cells and
5.3% at 64k. Prompt processing moved by a few percent in both directions at a
single repetition, including at depths where the gate is shut, so no prompt
figure is claimed from this run.

With four slots on one server and a split cache, the four query streams each
decode through a one-row tile. Four concurrent answers of 64 greedy tokens over
prompts of about 4000 tokens diverge from the dense build at token 6, 26, 58 and
63; two runs of the gathering build against each other diverge at 6, 56, 58 and
61, so the divergence is the run-to-run order of a shared batch, not the path.
A unified cache, which is what puts four query rows in one tile, was not
measured.
