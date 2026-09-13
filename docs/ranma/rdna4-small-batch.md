# RDNA4 small-batch matmul (HIP)

## What it is

Two architecture entries in the quantized dense matrix multiplication dispatch of the HIP/CUDA
backend (`ggml/src/ggml-cuda/mmvq.cu`), both keyed on RDNA4:

1. `calc_rows_per_block` returns four weight rows per MMVQ thread block for 3..8 activation
   columns. One and two columns keep one row per block.
2. `ggml_cuda_should_use_mmvq` gets an RDNA4 arm that lowers the MMVQ/MMQ crossover per weight
   type.

| Weight type | First activation-column count routed to MMQ |
| --- | ---: |
| Q1_0 | 4 |
| Q4_K, Q5_K | 5 |
| NVFP4 | 7 |
| Q6_K | 8 |
| Other supported types | 9 (upstream default) |

There is no CLI option and no environment switch: both are static tables, selected on the device
side by the compiled architecture and on the host side by the compute capability, exactly like the
existing Ada Lovelace, Blackwell, Orin and CDNA entries in the same two functions.

## When it applies

A dense `MUL_MAT` with a quantized weight, on a HIP build running on an RDNA4 device, with an
activation matrix of 1..8 columns. In practice that is a speculative-decoding verification step, a
server batching a few slots, or any other decode call with more than one row. Single-column decode
and prompt batches of hundreds of columns are outside the range that changes.

## Why it exists

With one weight row per block, every block re-reads all columns of the activation matrix from L2.
The activation traffic therefore grows with the column count, and on RDNA4 the cost of one call
grew steadily from one to eight columns. MMQ, the quantized GEMM path, computes a 16-column tile
regardless of the column count, so its cost is flat across that range and it only loses below a few
columns to its fixed tile cost.

Upstream left both RDNA tables empty, so RDNA4 used one row per block at every width and stayed on
MMVQ up to eight columns. The result was a cost curve that fell when a call got wider: a nine-column
call was cheaper than an eight-column one. Any policy that decides how many draft tokens to verify,
and any server that batches a few slots, pays for that bend.

Four rows per block cut the repeated activation reads four-fold, which fixes the slope inside the
MMVQ range, and the lowered crossover hands the remaining widths to the flat MMQ path. Together the
two entries make the cost of a call non-decreasing in the column count. Two columns keep one row per
block because four rows per block measured as a loss there.

## What it does not touch

- `MUL_MAT_ID` (MoE experts): multi-token expert matmuls use their own kernel with its own fixed
  rows per block, and the MoE MMVQ/MMQ limit is a separate one.
- Fused gate/up MMVQ, which requires a single activation column.
- Single-row decode (one row per block) and prompt processing (already MMQ).
- Every non-RDNA4 device and every non-HIP build.

## Selection rule and limits

The thresholds come from paired MMVQ/MMQ timings over synthetic K/M shapes and both weight-reuse
conditions; a type moves to MMQ at the first column count from which every measured pass at and
above it was at least 3 % faster. The measurements were taken on gfx1201. gfx1200 is the same design
at half the width and was not measured; the code base has no per-chip table, and the RDNA3 precedent
splits a family only when the memory system differs.

In a build with `GGML_CUDA_FORCE_CUBLAS` the MMQ selector returns false, so the calls this table
hands to MMQ fall to hipBLAS instead, where upstream would have used MMVQ. The Ada Lovelace,
Blackwell and CDNA entries of the same function behave the same way.

## Numerics

Four rows per block keep the per-row accumulation order, so a call that stays on MMVQ produces the
same values as before.

Moving 5..8-column calls to MMQ is a real numerical change and not only an accumulation-order
effect: MMVQ and MMQ quantize the activations separately, and they approximate the constant term of
Q4_K and Q5_K differently - MMVQ multiplies the block minimum by the sum of the quantized activations
times their scale, while MMQ's quantizer stores the sum of the original float activations. Q6_K has
no minimum term, so there only the accumulation order differs. No new approximation is introduced:
the existing MMQ one now applies from five columns instead of from nine, so outputs at 5..8 columns
differ from the MMVQ ones in the same way outputs at nine columns and above always did.

## How to verify

- `test-backend-ops -o MUL_MAT` covers the affected widths, including K-quant weights at 1..8
  columns with a row count that is not a multiple of four.
- The cost curve over the column count is visible in
  `llama-bench -ngl 999 -fa on -b 16 -ub 16 -n 0 -p 1,2,3,4,5,6,7,8,9,16 -d 0,8192`: milliseconds
  per call is the column count divided by the reported tokens per second. With these tables the
  curve no longer falls at the crossover.
- Multi-slot decode throughput, the non-speculative way to reach 3..8 columns per call, is visible
  in `llama-batched-bench -npl 1,2,4,8`.
