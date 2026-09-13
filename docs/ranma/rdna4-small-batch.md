# RDNA4 small-batch matmul: MMVQ rows_per_block 4 and the MMQ crossover (HIP)

## What it is

Two tuning entries in `ggml/src/ggml-cuda/mmvq.cu`, both keyed on the RDNA4 architecture:

1. `calc_rows_per_block` returns 4 instead of 1 for 3..8 activation columns. The MMVQ kernel then computes four
   weight rows per thread block instead of one.
2. `ggml_cuda_should_use_mmvq` gets an RDNA4 entry: dense `MUL_MAT` on Q4_K, Q5_K and Q6_K weights uses MMVQ up to
   4 rows and MMQ from 5 rows. Upstream uses MMVQ up to 8 rows on RDNA4.

Both are architecture tables of the same shape as the existing Ada Lovelace, Blackwell and CDNA entries in the same
two functions. There is no environment variable and no CLI flag.

## Why it exists

A speculative-decoding verification step, a multi-slot server (`-np`), or any decode call with 2..16 rows runs the
matmuls at exactly the widths where the two quantized kernels hand over. On RDNA4 that handover was in the wrong
place. Measured on the target model below, one decode call cost 35 ms with one row and then about +10 ms for every
extra row up to 8 rows, and a 9-row call cost 51 ms: a 9-row verification was 1.9x cheaper than an 8-row one. Any
policy that decides how many draft tokens to verify, or a server that batches 3..8 slots, pays for that bend.

The per-row cost came from the MMVQ kernel itself. With one weight row per block, every block re-reads all columns
of the activation matrix `y` from L2: about 130 MB per column for a 260 MB weight on the R9700, which matches the
measured +160 us per extra column on a single `ffn_up` matmul. Four rows per block cut that traffic four-fold. MMQ,
the quantized GEMM path, computes a 16-column tile regardless of the row count on RDNA4, so its cost is flat from
3 to 16 rows and it only loses below 3 rows to its fixed tile cost. Together the two entries make the cost per call
non-decreasing in the row count.

## Upstream context

- `calc_rows_per_block` already has per-table entries (generic/GCN/Turing/GB10 use 2 rows for 2..8 columns); the
  RDNA tables had none, so RDNA4 ran one row per block at every width.
- `ggml_cuda_should_use_mmvq` already lowers the MMVQ limit for K-quants on Ada Lovelace (<= 7), Blackwell (<= 5),
  Orin (<= 1), CDNA1 (Q4_K <= 2) and CDNA2 (<= 3), each tuned on one card and applied to the family. The RDNA4
  entry follows that convention.
- `ggml_cuda_should_use_mmq` returns true for every supported type on RDNA4 ("MMQ is consistently faster than
  dequantization + hipBLAS", upstream #18537), so in a normal build the 5..8-row calls go to MMQ. In a build with
  `GGML_CUDA_FORCE_CUBLAS` the MMQ selector returns false and those calls fall to hipBLAS instead, where upstream
  would still have used MMVQ; the Ada Lovelace, Blackwell and CDNA entries in the same function behave the same way.
- The tables are keyed on the architecture (`__GFX12__` on the device side, `GGML_CUDA_CC_IS_RDNA4` on the host
  side). They were tuned on gfx1201 (Navi 48). gfx1200 (Navi 44) is the same design at half the width and was not
  measured; the code base has no per-chip table, and the RDNA3 precedent splits a family only when the memory
  system differs (RDNA3.5 APUs).

## What it does not touch

- `MUL_MAT_ID` (MoE experts): multi-token expert matmuls go to the dedicated `mul_mat_vec_q_moe` kernel with its
  own fixed two rows per block before the column switch, and the MoE MMVQ/MMQ limit stays
  `get_mmvq_mmid_max_batch`.
- Fused gate/up MMVQ: fusion requires a single column for dense ops.
- Single-row decode and prompt processing: one row keeps rows_per_block 1, and 512-token batches were already MMQ.
  The standard curve below shows both unchanged.
- Other quantization types keep the upstream limit of 8 rows for MMVQ. Only Q4_K, Q5_K and Q6_K were measured.
- The last partial block: a weight with a row count that is not a multiple of 4 is handled by the preceding commit,
  which clamps the reads of the last block to the last row (upstream's rows_per_block 2 tables had the same unguarded
  read since #5434; only the store was guarded, whisper.cpp #2231).

## Numerics

rows_per_block 4 keeps the per-row accumulation order. Measured at matched routing (4 rows per call, MMVQ in both
builds), the KL divergence of the new logits against the old ones is zero at the tool's resolution: mean 0.000000,
median 0.000000, maximum 5.5e-5, which is the same maximum a build gets against its own saved logits, and 100 %
top-1 agreement over 4096 scored tokens, with and without flash attention. Any remaining difference is below the
f16 resolution of the stored logits.

The routing change at 5..8 rows is a real numerical change, and it is not only an accumulation-order effect. MMVQ and
MMQ quantize the activations to q8_1 separately and approximate the constant term of Q4_K and Q5_K differently: MMVQ
multiplies the block minimum by the sum of the quantized int8 activations times their scale
(`vec_dot_q4_K_q8_1_impl_vmmq`), MMQ's quantizer stores the sum of the original float activations and uses that
(`quantize_mmq_q8_1`, `ds4` layout). Q6_K has no minimum term, so there only the accumulation order differs. This
patch introduces no new approximation; it applies the existing MMQ one from 5 rows instead of from 9 rows, so outputs
at 5..8 rows now differ from upstream in the same way outputs at 9+ rows always differed from outputs at 8 rows. For
scale on this model: MMQ against MMVQ at 8 rows measures a median KL of 2.9e-3 per token (mean 0.27 with a heavy tail
on a high-perplexity corpus, top-1 agreement 90.7 %), and the upstream MMVQ kernel already disagrees with itself
between its 1-, 4- and 8-column instantiations by a median of about 1e-3 (top-1 93-95 %). Both builds pass
`test-backend-ops` (1297/1297 MUL_MAT cases, including Q4_K/Q5_K at 1..8 columns with a 1023-row weight).

End to end, greedy decoding of 22 SPEED-Bench prompts (256 tokens each) with the MTP drafter gives 21 of 22
byte-identical outputs against upstream at every draft depth tried (n-max 3: same routing; n-max 7: 8-row calls
move from MMVQ to MMQ; n-max 8: 9-row calls, MMQ in both). The one differing prompt is the same in every comparison
and is one on which upstream degenerates into 256 empty tokens. An upstream-against-upstream repeat is 22/22.

## Measured effect

Measured on the development machine, 2026-09-13: Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16,
Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. The GPU runs with a -75 mV voltage offset and a -30 %
power limit for every measurement here, and headless (no display attached; the CPU's integrated GPU drives the
display), so all 32 GiB are available to the model.

Model: gemma-4-31B-it Q4_K_M (dense, 17.05 GiB). A = upstream kernels, B = this patch, same build options; the
feature is compile-time, so A/B/A swaps the binaries and the two A passes bracket drift.

Revision: A is ranma `3d5d9b87c` (upstream llama.cpp `43f3dda62` + fork README + host-direct MoE; the GPU heartbeat
commit between it and this one touches only `llama-server`, not the kernels). B is A plus the preceding clamp
commit `f2dc2ee9b` and this commit; the `mmvq.cu` of the measured B binary is the one this commit carries. The exact
commits are kept on the dated branch `ranma_20260914`, which is never rebased.

### Verification cost curve: one decode call with N rows

`llama-bench -ngl 999 -fa on -t 16 -b 16 -ub 16 -n 0 -p 1..16 -d 0,8192 -r 5`, ms per call = N / (t/s) * 1000.

| N rows | A (d0) | B (d0) | A again (d0) | A (d8192) | B (d8192) | A again (d8192) | route in B |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 35.3 | 35.4 | 35.6 | 39.0 | 39.3 | 39.4 | MMVQ, unchanged |
| 2 | 39.9 | 39.3 | 39.8 | 41.3 | 42.1 | 41.7 | MMVQ, unchanged |
| 3 | 48.4 | 42.7 | 48.9 | 49.3 | 47.4 | 50.0 | MMVQ rows_per_block 4 |
| 4 | 57.2 | 48.3 | 57.2 | 56.7 | 51.7 | 57.3 | MMVQ rows_per_block 4 |
| 5 | 68.4 | 50.3 | 67.7 | 67.8 | 57.4 | 68.6 | MMQ |
| 6 | 78.5 | 50.3 | 78.3 | 76.9 | 57.0 | 77.1 | MMQ |
| 7 | 88.3 | 50.7 | 89.0 | 85.8 | 57.6 | 86.1 | MMQ |
| 8 | 98.8 | 50.7 | 99.1 | 95.5 | 58.0 | 96.0 | MMQ |
| 9 | 51.3 | 51.6 | 51.3 | 57.8 | 58.8 | 58.9 | MMQ, unchanged |
| 10 | 51.9 | 51.5 | 51.7 | 58.1 | 59.3 | 59.0 | MMQ, unchanged |
| 12 | 51.9 | 52.1 | 51.9 | 58.7 | 59.5 | 59.4 | MMQ, unchanged |
| 16 | 52.6 | 52.9 | 52.7 | 65.0 | 66.0 | 65.8 | MMQ, unchanged |

Drift between the two A passes is at most 1.1 ms; rows 1, 2 and 9..16 stay inside it. The crossover was chosen
from two extra runs of the same build with the routing forced: MMVQ with rows_per_block 4 costs 43.8 / 50.6 / 57.3
ms at 3 / 4 / 5 rows and MMQ costs 50.9 / 51.4 / 52.2 ms, so MMQ wins from 5 rows at both depths; at 2 rows MMQ
costs 50.7 ms against 40.3 ms, so the limit must not go lower.

### Multi-slot decode

Parallel sequences are the non-speculative way to reach 2..8 rows per decode call, so this is the patch in ordinary
server use. `llama-batched-bench -ngl 999 -fa on -t 16 -c 33792 -b 2048 -ub 512 -npp 512,4096 -ntg 128
-npl 1,2,4,8`; S_TG is the aggregate decode throughput over all sequences in tokens per second.

| prompt | sequences | S_TG A | S_TG B | S_TG A again | B vs A |
|---:|---:|---:|---:|---:|---|
| 512 | 1 | 27.97 | 27.90 | 27.86 | unchanged |
| 512 | 2 | 49.37 | 47.99 | 47.67 | unchanged (inside drift) |
| 512 | 4 | 65.31 | 76.95 | 62.66 | +17.8 % .. +22.8 % |
| 512 | 8 | 73.23 | 135.69 | 70.26 | +85 % .. +93 % |
| 4096 | 1 | 26.84 | 26.73 | 26.66 | unchanged |
| 4096 | 2 | 46.34 | 45.63 | 45.55 | unchanged (inside drift) |
| 4096 | 4 | 60.11 | 70.93 | 58.89 | +18.0 % .. +20.4 % |
| 4096 | 8 | 66.54 | 117.10 | 65.88 | +76 % .. +78 % |

Prompt processing (S_PP) is flat across A/B/A at both prompt lengths, as expected for an MMQ path.

### Speculative decoding on the server

The same 22 prompts as the hash check, greedy, `--spec-type draft-mtp` with the Gemma 4 MTP head, decode tokens per
second as reported by the server:

| `--spec-draft-n-max` | rows per verification | A | B |
|---:|---|---:|---:|
| 3 | up to 4 (MMVQ in both) | 48.2 | 57.5 (+19 %) |
| 7 | up to 8 (MMVQ in A, MMQ in B) | 37.7 | 63.7 (+69 %) |
| 8 | up to 9 (MMQ in both) | 68.9 | 66.5 (-3.5 %, within run-to-run) |

### Standard curve, single sequence

`llama-bench -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -p 512 -n 128 -r 3 -d 65536,0,8192,32768,65536`, first
65536 pass discarded as warm-up.

| depth | pp512 A | pp512 B | pp512 A again | tg128 A | tg128 B | tg128 A again |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 950.4 | 949.7 | 948.5 | 28.82 | 28.72 | 27.74 |
| 8192 | 658.2 | 658.9 | 657.4 | 26.68 | 26.59 | 26.54 |
| 32768 | 343.2 | 343.5 | 343.0 | 24.57 | 24.53 | 24.50 |
| 65536 | 206.1 | 206.1 | 206.0 | 22.20 | 22.15 | 22.14 |

B differs from the two A passes by at most 0.1 % on pp512 and sits inside the A/A range on tg128 at every depth;
neither path runs the changed code (prompt processing is MMQ, single-row decode keeps rows_per_block 1).

## Trade-offs and open questions

- Users who compare outputs across batch widths will see the MMVQ/MMQ boundary at 5 rows instead of 9. Upstream
  already accepts that boundary; a multi-slot server never guaranteed which kernel a request lands on.
- gfx1200 (RX 9060 XT) is untested. If it regresses, the fix is a sub-family entry in the same two tables.
- The MoE reference model of the ranma methodology is not measured here. Its benchmark rows are bound by PCIe reads
  of host-resident experts. The patch may still have an effect there (the dense projections of that model take
  the same 3..8-row path), but the input-dependent run-to-run noise of those rows is larger than the effect this
  patch can plausibly produce, so a table from them could not be trusted either way.
- Two columns keep rows_per_block 1: four rows per block measured as a loss there (456 -> 471 us on `ffn_up`).
- The remaining gap between 3 rows (42.7 ms) and 2 rows (39.3 ms) is the next thing to look at; a per-tensor
  rows_per_block or a two-column path that stages `y` once in LDS are the candidates.
- While measuring the numerics, the upstream MMVQ kernel turned out to disagree with itself by the same margin
  across its 1-, 4- and 8-column instantiations as it does with MMQ. That is upstream behaviour on every
  architecture and outside this patch, but it is the reason "same kernel family" does not imply bit-identical
  output.
