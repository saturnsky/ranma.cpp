# Qwen sparse attention

The Qwen sparse-attention layers attend to a subset of the KV cache. A small
indexer runs in front of each of them: the keys of `ratio` consecutive tokens
form a block, one mean-pooled, normalized and rotated key represents the
block, the query scores every block, and a top-k over those scores picks the
cells the attention layer is allowed to read. Everything outside the selection
is masked away.

Two properties of that design shape everything on this page. The selection is
what the indexer decides, so an implementation is correct when it selects the
same cells; and every cell of a block carries the score of its block, so the
selection budget almost always cuts through a group of exactly tied values.

## Switches

All of them are read once per process. The default of every one of them is the
behaviour described in the sections below; the others select a reference path
for equivalence checks, or a diagnostic.

| name | default | effect |
| --- | --- | --- |
| `GGML_CUDA_TOP_K_STABLE_TIES` | off | `1` makes the top-k pick the smallest columns among tied values, so selections and outputs can be compared between runs. Test switch. |
| `LLAMA_QSA_DUMP` | unset | a path: write one line per indexer layer and graph evaluation there. Diagnostic; it changes the graph. |
| `LLAMA_QSA_LEGACY` | off | `1` selects the per-token indexer cache instead of pooled block keys. Reference path. |
| `LLAMA_QSA_CACHE_NORM_ROPE` | on | `0` stores the pooled keys untransformed and applies norm and rotation in every graph. Reference path. |
| `LLAMA_QSA_BLOCK_TOP_K` | `1` | `0` selects over the cells, `2` forces the general block kernels. Reference paths. |
| `LLAMA_QSA_RAW_PREFIX` | unset | a path prefix: the dump also writes raw output logits with their shape. Diagnostic. |
| `GGML_CUDA_FATTN_SPARSE` | on | `0` keeps the dense flash-attention kernel. Reference path. |
| `GGML_CUDA_FATTN_SPARSE_MIN_KV` | 4096 | the floor of the cell count at which the gather is used. |
| `GGML_CUDA_FATTN_SPARSE_LOG` | off | `1` logs the first shape that takes the gather. Diagnostic. |
| `GGML_CUDA_FATTN_COMPACT_PARALLEL` | on | `0` always builds an index list with the serial kernel. Reference path. |
| `GGML_CUDA_FATTN_COMPACT_VERIFY` | off | `1` compares both compaction kernels on the host and aborts on a difference. Diagnostic. |
| `LLAMA_QSA_ALL_CELLS_BYPASS` | on | `0` builds the selection at every context length, as upstream does. Reference path. |

## Tie-breaking in the top-k

Which of the tied values a top-k keeps is unspecified: `ggml_top_k` makes no
promise about it, and the radix implementation on CUDA and HIP picks a value
threshold and then gathers in whatever order its atomics happen to run. Ties
are common here, and also wherever scores are rectified, since those collapse
to exactly zero. Two runs of the same build can therefore select different
cells, which is valid but makes selections and outputs hard to compare.

`GGML_CUDA_TOP_K_STABLE_TIES=1` fixes the choice: a second radix walk, over
the column index and from the low bin upwards, finds the largest column index
that still fits in the budget, and only the tied columns at or below it are
kept. The selected set is then fully determined by the input; the order within
the reserved slots is not, and callers do not depend on it. The extra walk
covers only the bits a column index needs, so it is two passes for a context
of up to 64k rather than four.

The switch is off by default: nothing about quality prefers one order among
equal scores, so normal use keeps the behaviour of `ggml_top_k` and does not
pay for the extra passes. It is a test switch, and it applies to every
`ggml_top_k` that the backend runs through the radix selection - the indexers
of other sparse-attention models included - not only to this graph. The expert
routing of a mixture-of-experts model uses an argsort and is not affected.
With the switch unset, the only permanent change is one extra comparison in
the gather.

`test-backend-ops -o TOP_K` covers both settings.

## The indexer dump

Because the selection is what the indexer decides, a change to how it stores
or scores its keys has to be shown not to change that selection.
`LLAMA_QSA_DUMP=<path>` writes the evidence for it.

Every indexer layer appends one line per graph evaluation:

```
<step> <layer> <n_kv> <n_selected> <sha1 of the cells> <sha1 of the blocks>
```

Each selected row is sorted before it is hashed, so an implementation may
return the same set in another order. The cell hash is only comparable between
runs when `GGML_CUDA_TOP_K_STABLE_TIES=1` is set, because the budget cuts
through tied block scores. The block hash folds the selected cells onto their
blocks and is the implementation-independent comparison: it is the selection
the indexer actually makes.

Tensors whose name begins with `indexer_dbg_` are dumped as comment lines:

```
# dbg <step> <layer> <name> <bytes> <sha1 of the raw bytes>
```

These are what makes a change to the key pipeline provable, since the pooled
keys behind the selection are not tie-sensitive: two implementations that
produce the same bytes here cannot select differently for any reason of their
own.

An equivalence check is then: run the same prompt twice with the same build,
once with the path under test and once with the reference path, both with
`GGML_CUDA_TOP_K_STABLE_TIES=1` and both writing a dump, and compare the two
files. Equal `# dbg` hashes prove the keys are bit-identical; equal block
hashes prove the selections agree; equal cell hashes prove they agree down to
the tied cells.

Enabling the dump is not free of side effects. It inserts a `ggml_cont` in
front of every tensor it names, so the graph is not the graph a normal run
builds, and it installs its own evaluation callback when the caller has set
none. With the variable unset, nothing changes.

## Pooled block keys in the indexer cache

The indexer cache holds one pooled key per block. Since a step completes at
most one block, only that block has to be pooled; the rest of the context is
read from the cache as it stands.

Each ubatch is planned on the host before its graph is built. The plan says
which of the ubatch's raw keys belong to a block that is still open and must
be kept for a later step, which blocks the ubatch completes, and which cache
row each completed key is written to. The graph pools the completed blocks,
writes them into the indexer cache, and copies the members of the still-open
block into a small persistent state tensor, which the next ubatch reads back
as the earlier members of that block. A reservation ubatch carries no real
positions, so it is planned as if its tokens were consecutive and at the full
block count - the worst case any later graph can ask for.

`LLAMA_QSA_LEGACY=1` restores the previous per-token indexer cache, which
stores one raw key per token and pools the whole context on every graph. It is
kept as the reference the pooled path is compared against, not as a fallback.
Keys pooled on the per-token path pass through the cache type first, so the
pooled path applies the same rounding to its members before pooling them; both
produce the same block key bit for bit, which the indexer dump shows directly.

The pooled cache keeps one stream per sequence. A unified KV cache keeps a
single stream for all sequences, so with more than one sequence the two
shapes do not line up; that combination is refused when the context is
created, with a message to run without `--kv-unified`. A single sequence, a
split cache and `LLAMA_QSA_LEGACY=1` are not affected.

The layout of the indexer cache is part of the sequence state. Each layout
writes its own version into the state file, and a file written by another
layout is refused instead of being read as keys that do not match it.

## Cached norm and rotation

A block key is normalized and rotated before it is scored. Both are applied
once, when the block is completed, and the indexer cache holds the finished
key, so a graph reads it and scores it directly. Only the rope section
positions of the blocks a ubatch completes are uploaded; the positions of the
whole context are no longer a graph input. The host still scans the block
positions, because the bookkeeping of the completed blocks comes out of that
scan.

A stored key depends on the rope parameters and on the normalization epsilon,
so those are recorded when the first indexer graph is built. A later graph
that would use different ones is rejected, and the pair is written into the
state file under a state version of its own: a file produced with a different
transform is refused rather than scored against keys that do not match it.

`LLAMA_QSA_CACHE_NORM_ROPE=0` goes back to storing the pooled keys
untransformed and transforming the whole context in every graph. It is the
in-binary reference for "what the cache holds equals what a recompute
produces". It applies to the pooled cache only; with the per-token indexer
cache selected, the transform always happens in the graph.

The dump gains a line for the transformed cache:

```
# ktrans <step> <layer> <valid blocks> <sha1 of the block descriptions> <sha1 of the block keys>
```

It hashes the valid blocks together with their sequence, their first position
and their section positions, so a cached block can be compared with a
recomputed one even when the two paths hold it in different cache rows.

## Block top-k

The indexer scores blocks, but the selection used to run over cells: the score
of a block was expanded to each of its cells, the attention mask was added to
the expanded scores, and a top-k walked all cells of the context for every
query row. Both the temporary and the walk grew with the context, for a
quantity that has one value per block.

`GGML_OP_TOP_K_BLOCK` selects the same cells from the block scores directly.
It takes the scores, the cell-to-block map and the mask, and weights each
block with the number of its cells the mask leaves visible - which is exactly
what the expansion encoded - so the per-cell expansion disappears while the
result stays a list of cell indices. There is a CPU reference implementation;
the device implementation is built for HIP only, and other backends report the
op as unsupported.

The device side has two paths:

- a single-query row - the token-generation shape - whose inputs carry a
  description of the blocks is selected by one workgroup. It reads the cells
  of a block through the block-to-cells map and takes the per-block weights
  from the description, so it never walks the cells. The description is a
  promise about one row: every block below a given index holds `ratio` cells,
  all of them visible unless the block itself is masked; one spare block
  additionally holds a listed handful of visible cells that no full block
  covers; and a given number of cells is masked. A row that cannot make that
  promise sets the flag to zero and is selected by the general walk inside the
  same kernel, so the list of launches does not depend on the data.
- every other shape, prompt processing included, goes through the general
  weighted radix kernels: count the visible cells per block, walk histograms
  over the weighted block scores to find the threshold, then gather the cells.

A budget equal to the number of cells short-circuits to "every cell". Scratch
sizes are rounded - the weights to a power of two, the histograms to their
maximum width - so a prompt processed in ubatches of changing size does not
leave one pool buffer per size behind.

`LLAMA_QSA_BLOCK_TOP_K` selects the path:

| value | effect |
| --- | --- |
| `1` | default; every row is selected over the blocks |
| `0` | selection over the cells, the reference path |
| `2` | block selection forced through the general kernels, the reference for the single-row kernel |

The graph also falls back to the cell selection when the device does not
support the op, when the mask is not a plain contiguous causal mask over the
whole context, under SWA or ALiBi, and when the indexer cache does not hold
the transformed keys. The chosen path is recorded in the dump as a `# select`
comment in front of each selection line.

`test-backend-ops -o TOP_K_BLOCK` covers both device paths, one and two
sequences, one and two query rows, described and undescribed rows, an F16 and
an F32 mask, a budget equal to the cell count and a very small budget.

### Tie-breaking and run-to-run differences

The op takes a parameter that asks for the smallest cell indices among the
tied ones, for tests; the model does not set it and relies on
`GGML_CUDA_TOP_K_STABLE_TIES` instead. Two implementations may therefore
select different cells at the budget boundary, as `ggml_top_k` permits, and
that is visible in the output. The size of the effect was measured on a
Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL, `llama-perplexity
-c 4096` over 8 chunks of wikitext-2, 13 runs:

- with `GGML_CUDA_TOP_K_STABLE_TIES=1` the block path gives logits identical
  to the cell path (mean KLD 0, same top token 100 %), and 1740 of 1740
  selections of a 64k decode are identical;
- with the switch unset, two runs of the cell path in the same numerical mode
  agree (mean KLD 0, same top token 100 %), while the block path against the
  cell path in the same mode gives mean KLD 0.00015 and 0.00073 with the same
  top token 99.91 % and 99.62 %, and no difference in perplexity;
- independently of the selection path, a process start lands in one of several
  numerical modes of the prompt-processing kernels. Two runs in different
  modes differ by mean KLD 0.020 with the same top token 95.1 %, and do so
  identically for the cell path and the block path. The modes were later
  traced to small F32 products such as the MoE router, for which hipBLASLt
  chose its solution per process; on RDNA4 these products now run on
  fixed-order kernels ([rdna4-prefill.md](rdna4-prefill.md)).

When two builds or two paths have to be compared exactly, set the tie switch
and compare the dump.

## Attending only to the selected cells

The selection leaves a bounded number of finite entries in each mask row, but
attention still read every cell of the KV cache and discarded the rest through
the mask, so its traffic followed the context depth instead of the budget.

### The index lists

The flash-attention node carries the width of the indexer selection as a
per-row bound. When it is set, the backend compacts the mask ahead of the
attention kernel and hands the kernel index lists instead of the per-tile cell
bound it otherwise gets.

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
- no ALiBi and no logit softcap (a node with attention sinks may gather: the
  sink correction acts after the KV loop and does not depend on which cells
  the loop visited);
- the mask covers the whole cache contiguously and has a single head;
- the wave is 32 lanes wide;
- the tile holds at most four query rows;
- the cache holds at least `max(4096, 2*ncols1*bound)` cells - below that the
  dense kernel is already as cheap as compacting the mask.

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
| `GGML_CUDA_FATTN_SPARSE_LOG` | off | `1` logs the first shape that takes the gather |

### What was checked

On a Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL, base revision
ec5a12b85 with this patch, `llama-bench -ncmoe 35 -ngl 999 -t 16 -fa on -ctk
f16 -ctv f16 -b 512 -ub 512 -p 512 -n 128 -r 1 -d 0,8192,32768,65536`, MoE
tensors placed in host memory, one run per row with an idle gap between runs:

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
A unified cache, which is what puts four query rows in one tile, cannot be
measured here: with more than one sequence the pooled indexer refuses it (see
[Pooled block keys in the indexer cache](#pooled-block-keys-in-the-indexer-cache)).

`test-backend-ops -o FLASH_ATTN_EXT` carries the generation shape at one, two
and four query rows per tile, with one and with two sequences, at cell counts
where the gate opens and with a budget far below the cell count so that the
lists are short. For a comparison on a real model,
`LLAMA_QSA_RAW_PREFIX=<path>` makes the dump write the raw output logits with
their shape, so two runs can be compared on the distribution rather than on
sampled text.

## Compacting the mask

Turning the mask of a query tile into its index list is a scan of the mask. One
block of 256 threads walks the cells in chunks of 2048, votes each chunk over
the queries of the tile and carries the running count of the list from chunk to
chunk, so a long list becomes a chain of dependent steps - at a depth of 64k, 32
of them, for every sparse attention call.

A list of five chunks or more is built by two launches with one block per list
and chunk instead. The first counts the selected cells of every chunk; the
second adds up the counts of the chunks in front of its own and writes its
indices at that offset, and the block of the last chunk knows the count of the
whole list, so it writes the padding behind it and stores the count. Both passes
vote over the same cells in the same order as the serial kernel and only replace
its carried count by the sum of the chunk counts, so the list is the same: same
cells, ascending, same truncation at the budget, same padding, same count.
Shorter lists keep the serial kernel, which needs one launch instead of two. The
chunk counts are a few kilobytes and are taken from the pool of the context for
the duration of the two launches.

| switch | default | effect |
| --- | --- | --- |
| `GGML_CUDA_FATTN_COMPACT_PARALLEL` | on | `0` always uses the serial kernel |
| `GGML_CUDA_FATTN_COMPACT_VERIFY` | off | `1` runs both kernels on the real launch arguments, feeds the model from the serial one, compares every index and every count on the host and aborts on the first difference |

The verify mode is the equivalence gate between the two kernels. It reads the
result back, which a stream that is capturing a graph cannot do: in that case it
warns once and compares nothing, so it has to be run with
`GGML_CUDA_DISABLE_GRAPHS=1`. It logs a line whenever the tile width or the
number of chunks changes and every 1000 calls, with the number of calls and
lists checked so far.

### What was checked

`test-backend-ops -o FLASH_ATTN_EXT` carries the generation shape with lists of
8 and 32 chunks at one and four query rows per tile, with one and with two
sequences, and a four-row case at 10240 cells where the gate stays shut. On a
Radeon AI PRO R9700 the suite passes with the default, with
`GGML_CUDA_FATTN_COMPACT_PARALLEL=0` and in the verify mode; the verify mode
compared lists of 2, 4, 8 and 32 chunks at one, two and four query rows per
tile, all equal. With Qwen3.8-Flash-Next UD-Q4_K_XL, a prompt of 66886 tokens
and 64 generated tokens in the verify mode compared every list of 33 chunks
without a difference.

Base revision ec5a12b85 with this patch, the same model, `llama-bench -ncmoe 35
-ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -p 512 -n 128 -r 1 -d
32768,65536`, MoE tensors placed in host memory, one process per row with an
idle gap of five minutes between them:

| cells | serial tg128 | parallel tg128 |
| --- | --- | --- |
| 32768 | 25.01 | 25.12 |
| 65536 | 24.31 | 24.70 |

That is 0.4% at 32k cells and 1.6% at 64k, one run per row. Prompt processing
does not go through the gather at a batch of 512 and is not claimed.

## While the budget covers every cell

The indexer selects at most `indexer_top_k + ratio - 1` cells for a query.
While the cache holds no more cells than that, the selection is the whole
cache: the mask rebuilt from it equals the attention mask it started from, and
the scores decide nothing. Building the selection anyway costs the projection,
normalization and rotation of the indexer query, the scores against every
block, the top-k and the rebuilt mask, in every sparse-attention layer of
every step. That is the whole cost of the indexer, paid by a context that is
too short to gain anything from it.

A graph whose selection width equals its cell count therefore builds no
selection, and the layer attends through the dense path, as a layer without an
indexer does. The cell count of a graph is padded, so the dense path ends a
few tokens before the context reaches the budget.

What the later steps need is still done. The indexer keys are written as
before, the pooled block keys and the members of the open block included,
because the steps after the budget is exceeded select over them. The inputs of
the selection stay in the graph although nothing reads them: the host pass
that fills them is also the one that finds the blocks a ubatch completes.

The selection is kept when the mask does not span exactly the cells of the
graph, and under `LLAMA_QSA_DUMP`, which reads the selection back.
`LLAMA_QSA_ALL_CELLS_BYPASS=0` keeps it at every length, which is the upstream
behaviour and the reference path.

The two settings compute the same attention: with every cell selected the
rebuilt mask equals the attention mask value for value, and the attention node
is the one the dense path builds. The model output is still not bit-identical,
and why the bits differ was not traced for this revision. Its size was
measured on a Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL, base
revision ec5a12b85 with this patch, `llama-perplexity -c 4096 --chunks 8` over
wikitext-2 with `GGML_CUDA_TOP_K_STABLE_TIES=1`, where the first half of every
chunk is processed without a selection:

- two runs with the switch at `0` give the same statistics to the last digit
  (PPL 3.1707, mean KLD 0.000000, same top token 100 %);
- a run of the default against the switch at `0` gives mean KLD 0.0193 with the
  same top token 95.20 %, and PPL 3.1764 against 3.1707, a difference of
  +0.0057 with an error of 0.0059.

The default does not reproduce the bits of the reference path. Numbers of the
same size between two runs that land in different numerical modes are given in
[Tie-breaking and run-to-run differences](#tie-breaking-and-run-to-run-differences).

With `llama-bench -ncmoe 35 -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub
512 -p 512 -n 128 -r 1 -d 0,4096`, MoE tensors in host memory, one process per
setting with an idle gap of five minutes between them, generation of 128
tokens measured 26.09 t/s with the switch at `0` and 26.45 t/s with the
default at a depth of 0 (+1.4%). At 4096 cells the budget no longer covers the
cache and both settings build the selection, and the two runs still differ by
+0.7% (25.86 and 26.03 t/s), so a single run carries a spread of that size.
Prompt processing of 512 tokens at a depth of 0 measured 593.7 and 594.5 t/s.
