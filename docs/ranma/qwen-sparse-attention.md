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
