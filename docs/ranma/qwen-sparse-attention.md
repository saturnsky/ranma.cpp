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
| `GGML_CUDA_TOP_K_STABLE_TIES` | off | `1` makes the top-k pick the smallest columns among tied values, so selections and outputs can be compared between runs. Option for reproducible runs. |
| `LLAMA_QSA_LEGACY` | off | `1` selects the per-token indexer cache instead of pooled block keys. Reference path. |
| `LLAMA_QSA_CACHE_NORM_ROPE` | on | `0` stores the pooled keys untransformed and applies norm and rotation in every graph. Reference path. |

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
pay for the extra passes. It is an option for reproducible runs, and it
applies to every `ggml_top_k` that the backend runs through the radix
selection - the indexers of other sparse-attention models included - not only
to this graph. The expert
routing of a mixture-of-experts model uses an argsort and is not affected.
With the switch unset, the only permanent change is one extra comparison in
the gather.

`test-backend-ops -o TOP_K` covers both settings.

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
produce the same block key bit for bit.

The pooled cache keeps one stream per sequence. A unified KV cache keeps a
single stream for all sequences, so with more than one sequence the two
shapes do not line up; that combination is refused when the context is
created, with a message to run without `--kv-unified`. A single sequence, a
split cache and `LLAMA_QSA_LEGACY=1` are not affected.

The layout of the indexer cache is part of the sequence state. Each layout
writes its own version into the state file, and a file written by another
layout is refused instead of being read as keys that do not match it.

### Suffix removal inside an open block

A suffix removal whose start is not a block boundary cuts a block that is
still open: the tokens of that block below the cut stay, and the next ubatch
needs their raw keys as the earlier members of the block. The pooled cache
therefore records which position each open-block row of the persistent state
tensor holds, and accepts such a removal when every row of the cut block below
the cut is still held at its position; otherwise the removal is refused before
any cache is touched, as before. The rows form a ring of `ratio + n_rs_seq - 1`
per stream (`ratio` without speculative decoding, so the graph is unchanged
there), so a speculative rollback of up to `n_rs_seq` tokens always finds the
rows it needs. A partial (checkpoint) state carries the open-block rows and
their positions too, so a server checkpoint taken at any position can be
restored and continued. A full state written without the positions still
loads; its rows then count as unknown and an unaligned removal is refused.

The server no longer aborts when a memory refuses a suffix removal. After a
checkpoint restore it falls back to the next older context checkpoint, as when
the newest one does not fit, and reprocesses the prompt from the start only
when none is left (one warning line per step); during a speculative rollback it
ends the request with an error and clears the slot. When the memory of a
standalone draft model refuses the removal of a draft, the server rebuilds the
draft sequence from the newest prompt checkpoint whose removal it accepts, else
from the start; a draft fed by target hidden states, a prompt with media or with
shifted positions ends the request instead.

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
