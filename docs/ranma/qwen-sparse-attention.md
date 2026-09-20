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
| `GGML_CUDA_FATTN_SPARSE_MIN_KV` | 4096 | lowest cell count at which the gather is used. |
| `GGML_CUDA_FATTN_SPARSE_LOG` | off | `1` logs the first shape that takes the gather. Diagnostic. |
| `GGML_CUDA_FATTN_COMPACT_PARALLEL` | on | `0` always compacts a mask row with the serial kernel. Reference path. |
| `GGML_CUDA_FATTN_COMPACT_VERIFY` | off | `1` compares both compaction kernels on the host and aborts on a difference. Diagnostic. |

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
  identically for the cell path and the block path. The cause of the modes is
  not identified.

When two builds or two paths have to be compared exactly, set the tie switch
and compare the dump.

## Attending only to the selected cells

The selection leaves a bounded number of finite entries in each mask row, but
attention still read every cell of the KV cache and discarded the rest through
the mask, so its traffic followed the context depth instead of the budget.

The tile flash-attention kernel that RDNA4 uses for the Qwen generation shape
- head size 256, GQA-12 groups, one query row per tile - can gather instead.
The mask row is compacted into the ascending list of the cells it leaves
finite, and the kernel loads its K and V rows through that list. The mask
value is read at the original cell coordinate, and a padding slot of the list
contributes nothing, so the attended set and the arithmetic are those of the
dense masked kernel; only the memory traffic changes. The compaction votes
through a ballot that does not assume a wave width, and only the mask rows
that some block of the launch actually reads are compacted.

The bound comes from the graph: the attention node carries the width of the
indexer selection, and the model sets it only when it is smaller than the
number of cells, so a short context builds exactly the node the dense path
builds.

The gather is taken on HIP only, and only when all of the following hold:

- the per-row bound is set;
- K and V are F16 - single rows are read, so they cannot go through the
  on-the-fly conversion;
- no ALiBi, no logit softcap, no attention sink;
- the mask covers the whole cache contiguously and has a single head;
- one query row per tile;
- the wave is 32 lanes wide;
- the cache holds at least `max(4096, 2*bound)` cells; below that the dense
  kernel is already as cheap as compacting the mask.

Every other shape, prompt processing included, keeps the dense kernel.

| switch | default | effect |
| --- | --- | --- |
| `GGML_CUDA_FATTN_SPARSE` | on | `0` forces the dense kernel in the same binary |
| `GGML_CUDA_FATTN_SPARSE_MIN_KV` | 4096 | replaces the 4096 of the bound above |
| `GGML_CUDA_FATTN_SPARSE_LOG` | off | `1` logs the first shape that takes the gather |

`test-backend-ops -o FLASH_ATTN_EXT` carries the Qwen generation shape with
one and with two sequences, at the cell count where the gate opens, and with a
short index list. For a comparison on a real model,
`LLAMA_QSA_RAW_PREFIX=<path>` makes the dump write the raw output logits with
their shape, so two runs can be compared on the distribution rather than on
sampled text.

## Compacting the mask

Turning a mask row into its list of selected cells is a scan of the row. One
block of 256 threads walks the row in chunks of 2048 cells and carries the
running count from chunk to chunk, so a long row becomes a chain of dependent
steps - at a depth of 64k, 32 of them, for every sparse attention call.

A row of five chunks or more is compacted by two launches with one block per
row and chunk instead. The first counts the selected cells of every chunk; the
second adds up the counts in front of its own chunk and writes its indices at
that offset, and the block of the last chunk writes the padding behind the
row. Both passes vote over the same cells in the same order as the serial
kernel and only replace the carried row count by the sum of the chunk counts,
so the list is the same: same cells, ascending, same truncation at the bound,
same padding. Shorter rows keep the serial kernel, which needs one launch
instead of two. The counts are a few kilobytes and are taken from the pool of
the context for the duration of the two launches.

| switch | default | effect |
| --- | --- | --- |
| `GGML_CUDA_FATTN_COMPACT_PARALLEL` | on | `0` always uses the serial kernel |
| `GGML_CUDA_FATTN_COMPACT_VERIFY` | off | `1` runs both kernels on the real launch arguments, feeds the model from the serial one, compares every entry on the host and aborts on the first difference |

The verify mode is the equivalence gate between the two kernels. It reads the
result back, which a stream that is capturing a graph cannot do: in that case
it warns once and compares nothing, so it has to be run with
`GGML_CUDA_DISABLE_GRAPHS=1`. It reports the number of calls and rows it has
checked as it goes.
