# Per-position draft thresholds

## What it is

Two speculative-decoding options that gate a draft token by its top-1 probability, with one value
per draft position instead of one value for the whole draft.

| Option | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--spec-draft-p-min P0,P1,...` | `LLAMA_ARG_SPEC_DRAFT_P_MIN` | `0.00` | Minimum probability for the token drafted at each position. Below it the token is dropped and drafting stops. |
| `--spec-draft-p-continue P0,P1,...` | `LLAMA_ARG_SPEC_DRAFT_P_CONTINUE` | off | Minimum probability for continuing after a kept token. Below it the token is kept in the draft, but no further draft step is run. |

Both take a comma-separated list of probabilities in [0, 1]. Positions are 0-based and use the same
numbering as the server's per-position acceptance metric: position 0 is the first drafted token. A
list shorter than the draft repeats its last value, so a single value applies to every position,
which is what the previous scalar form of `--spec-draft-p-min` did. Empty items, values outside
[0, 1] and trailing characters are rejected at argument parsing.

At draft position `i` the top-1 probability of the drafted token is compared against both gates:

- below `p_min[i]` the token is dropped and drafting stops, so it is never verified;
- at or above `p_min[i]` but below `p_continue[i]` the token is kept in the draft, but no further
  draft step is run after it.

`p_min` is honoured per position by every draft-model drafter (`draft-simple`, `draft-eagle3`,
`draft-dflash`, `draft-dflash2`, `draft-dspark`, `draft-mtp`). `p_continue` applies to the drafters
that produce one token per draft step (`draft-simple`, `draft-eagle3`, `draft-mtp`); block drafters
compute the whole block in a single decode, so there is no later draft step to skip - they log a
warning and ignore it.

The user-facing description of both options is in [docs/speculative.md](../speculative.md).

## Why it exists

Dropping a draft token and stopping the draft save two different things. A dropped token saves one
row of the target-model verification batch; a stopped draft saves one full decode of the draft
model.

The two costs are not comparable, and only one of them is constant. A draft step costs the same at
every position. A verification row costs what the kernel the batch lands in charges for it: while
the batch is still in the matrix-vector path, each row adds real time; once the batch is wide enough
to use the quantized GEMM path, extra rows are almost free until the next tile boundary, and the
first row past that boundary pays for a whole new tile. A single `p_min` cannot express "verify this
token but do not build on it", and a single value for all positions cannot follow a cost curve that
is steep for the first rows and flat afterwards.

The same tile boundary also bounds the useful draft length: the last draft length that keeps the
verification batch inside one tile is the last one that is free to verify.

## Choosing values

The thresholds describe a drafter's confidence calibration and a target's verification cost curve,
so they are tuned per pair, not universal. A profile that follows from the structure above looks
like this:

```
--spec-draft-p-min 0.33,0.6,0.6,0 --spec-draft-p-continue 0.9 --spec-draft-n-max 15
```

- `p_min` is non-zero only for the first few positions, because those are the rows that still cost
  something to verify; later positions are free to verify, so a token there is worth keeping even
  at a low probability.
- `p_continue` is a single high value: a draft step is expensive enough that it is only worth taking
  after a token the drafter is confident about.
- `n_max` sits at the last draft length that keeps the verification batch inside one tile of the
  target's GEMM path.

These values were tuned for one target model, one draft head and one GPU. On other hardware the
verification cost curve, and therefore the positions that deserve a threshold, are different; the
structure carries over, the numbers do not.

## Limits

- Runtime cost is two vector lookups per draft step.
- Output equivalence is not affected by the options themselves, but changing the draft length
  changes the verification batch width, and different widths can select different matmul kernels,
  which is already true of upstream speculative decoding at any batch width.
- The server's per-request parameter schema still exposes `speculative.p_min` as a single value in
  its (upstream-disabled) binding; the options are configured on the command line or through the
  environment.

## How to verify

- `--spec-draft-p-min 0.8` behaves exactly as the previous scalar option did, because a short list
  repeats its last value.
- The drafter logs both lists at start-up, `off` for an unset one, so a misparsed list is visible
  there.
- Setting `--spec-draft-p-continue` with a block drafter must produce the warning and change
  nothing else.
- The per-position acceptance metric of the server uses the same position numbering, so it can be
  read directly against the list index.
