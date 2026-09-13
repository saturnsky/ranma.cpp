# Per-position draft thresholds: `--spec-draft-p-min` list and `--spec-draft-p-continue`

## What it is

Two speculative-decoding options in `common/` (`common/arg.cpp`, `common/common.h`, `common/speculative.cpp`):

1. `--spec-draft-p-min P0,P1,...` now takes one probability per draft position (0-based, the same numbering as the
   server metric `spec_decode_num_accepted_tokens_per_pos_total`). A short list repeats its last value, so a single
   value behaves as before. A draft token whose top-1 probability is below `p_min[i]` is dropped and drafting stops.
2. `--spec-draft-p-continue P0,P1,...` is a second, per-position gate. A token that passes `p_min[i]` but is below
   `p_continue[i]` is kept in the draft, but no further draft step is run after it.

`p_min` is honoured per position by every draft-model drafter (`draft-simple`, `draft-eagle3`, `draft-dflash`,
`draft-dflash2`, `draft-dspark`, `draft-mtp`). `p_continue` applies to the drafters that produce one token per draft
step (`draft-simple`, `draft-eagle3`, `draft-mtp`); block drafters log a warning and ignore it. Defaults are
unchanged. The user-facing description is in [docs/speculative.md](../speculative.md).

## Why it exists

Dropping a draft token and stopping the draft save two different things. A dropped token saves one row of the
target-model verification batch; a stopped draft saves one draft-model decode (one MTP head step, about 2 ms on the
target below). The verification row is cheap or free once the batch is in the GEMM (MMQ) path and expensive while
it is still in the matrix-vector (MMVQ) path, so the value of a token depends on its position, while a head step
costs the same at every position. A single `p_min` cannot express "verify this token but do not build on it", and
a single value for all positions cannot follow a verification cost curve that is steep for the first rows and flat
afterwards.

Measured verification cost per call on the reference setup (median ms, rows = kept draft tokens + 1, measured
inside `llama-server` with a rotating verification width; `mmvq.cu` RDNA4 tables of `docs/ranma/rdna4-small-batch.md`):

| rows | 1 | 2 | 3 | 4 | 5..8 | 9..16 | 17..32 |
|---|---|---|---|---|---|---|---|
| ms | 34 | 38 | 41 | 49.5 | 49.1..49.7 | 50.7..52.4 | 61.0..63.3 |

Rows 1..4 are MMVQ (row 4 is the rows_per_block 4 corner), rows 5+ are MMQ: flat inside a 16-row tile, +1 ms at row
9, +8.6 ms at row 17. So the first three draft positions decide rows 2..4 where each row costs 3..8 ms, every
position after that costs nothing to verify until row 17, and a draft longer than 15 tokens pays for a second MMQ
tile on every round.

The MTP head's top-1 probability is bimodal at every position (64..69 % of steps at p >= 0.99, 3..5 % below 0.5),
but it is only a strong signal for the first positions: among steps with p >= 0.99 the target accepted 96 % at
position 0, 69 % at positions 1..3, 22 % at positions 4..14 and 1 % at positions 15..30. The chance that a draft
token is accepted given that all earlier ones were is 75..84 % at every position; the chance of reaching position
16 is 2.8 %.

## Measured effect

Setup: `llama-server` on a Radeon AI PRO R9700 (32 GiB, gfx1201, HIP, headless), gemma-4-31B-it Q4_K_M target with
the Gemma4 MTP head (`google-mtp-Q8_0.gguf`, `--spec-type draft-mtp`), `-c 8192 -np 1 -fa on`, greedy sampling,
256 output tokens per request, SPEED-Bench Qualitative (nvidia/SPEED-Bench) 88 prompts = 8 per category, with the
masked rows restored from their sources with NVIDIA's `specdec_bench` loader. Decode t/s = sum of generated tokens
over sum of generation time for the 88 requests; every case is one sequential run, and the run-to-run noise of the
session is about 1 t/s.

Revision: every case ran the same binary, built from the code of this commit (the one that adds this page) on
ranma `aa59cf92e` (upstream llama.cpp `43f3dda62` + fork README, host-direct MoE, GPU heartbeat, MMVQ clamp and
the RDNA4 small-batch commit whose kernels produce the verification cost curve above). Only documentation was
edited after the measurement. The measurement-only build that produced the capture (draft steps and verification
batches recorded to disk) is not part of `ranma`. The exact commits are kept on the dated branch `ranma_20260914`,
which is never rebased.

| policy | n_max 7 | n_max 8 | n_max 13 | n_max 15 | n_max 16 | n_max 31 |
|---|---:|---:|---:|---:|---:|---:|
| no speculation | 28.0 | | | | | |
| `p-min 0` | 69.2 | 69.5 | | 63.1 | | 42.0 |
| `p-min 0.9` | 68.7 | 69.9 | | 71.0 | | 68.1 |
| `p-min 0.8` (best single value in replay) | | | 72.6 | | | |
| `p-min 0.33,0.6,0.6,0 --spec-draft-p-continue 0.9` | 71.0 | 71.7 | | **74.1** | 72.3 | 71.4 |

(acceptance rate of the last row: 73 / 71 / 62 / 61 / 54 % at n_max 7 / 8 / 15 / 16 / 31.)

- The per-position profile at `n_max 15` is 2.65x no-speculation, +6.6 % over the best setting without thresholds
  (`p-min 0`, `n_max 8`) and +2.1 % over the best single value (`p-min 0.8`, `n_max 13`). It was the fastest setting
  in 9 of the 11 categories; the other two were within noise.
- `n_max`: without thresholds every extra position costs a head step that is almost always wasted (`n_max 31` is
  40 % slower than `n_max 8`). With thresholds the optimum is 15, the last draft length that keeps the verification
  batch inside one 16-row MMQ tile; 16 costs 2.4 % and 31 costs 3.6 % for 2.8 % of rounds that could use them.
- A per-position fit of all 30 values by coordinate descent on a replay of the same prompts reached 74.5 t/s, the
  same as the hand-set profile within noise, so the profile is not leaving anything on the table on this set.

Recommended profile for this model and GPU:

```
--spec-type draft-mtp -md google-mtp-Q8_0.gguf \
--spec-draft-p-min 0.33,0.6,0.6,0 --spec-draft-p-continue 0.9 --spec-draft-n-max 15
```

The reasoning behind the values: `p_continue 0.9` because a head step is 2 ms and a token at p >= 0.9 is accepted
often enough to be worth one more step; `p_min` only for the first three positions because those are the rows that
still cost something to verify; `n_max 15` because of the 16-row tile.

## What it costs

- Nothing at runtime: two `std::vector<float>` lookups per draft step.
- The values are tuned for one target, one draft head and one kernel cost curve. Another GPU (a different MMVQ/MMQ
  crossover), a different draft head (different confidence calibration) or a chat workload with much longer
  outputs may want different numbers; the structure (thresholds on the first rows, a continue gate, n_max at the
  tile boundary) is what carries over, not the values.
- The 88-prompt set is a throughput benchmark, not a quality one: greedy sampling, 256 tokens, mostly inside the
  model's thinking phase. Output equivalence was not established by this measurement (verification widths on either
  side of the MMVQ/MMQ crossover are not bit-identical), which is the same situation as upstream speculative
  decoding at any batch width.

## Method notes

The head-step cost, the verification cost per width and the per-position acceptance statistics were captured with a
measurement-only branch that records every draft step and verification batch of a `p-min 0`, `n_max 31` run and
then replays candidate policies offline (`harness/simulate.py` in the evidence directory). The replay reproduces the
measured ranking but has an absolute error of a few t/s, so every number in the table above is a real run.
