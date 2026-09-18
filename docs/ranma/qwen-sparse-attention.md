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
