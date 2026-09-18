# RDNA4 small-batch kernels (HIP)

Two related pieces of the single-token and few-token path on HIP: the dispatch of the quantized
dense matrix multiplication, and a tile flash-attention block shape for head groups of twelve.

## Environment switches

| Name | Default | Effect of the non-default value |
| --- | --- | --- |
| `GGML_HIP_FATTN_GQA12` | `1` (on) | `0` removes the 12-column tile attention dispatch below, so the selection falls back to the upstream one. Kept as the reference path for equivalence checks; read once per process. |

The matmul dispatch has no switch and no CLI option.

# Dense matmul dispatch

## What it is

Two architecture entries in the quantized dense matrix multiplication dispatch of the HIP/CUDA
backend (`ggml/src/ggml-cuda/mmvq.cu`), both keyed on RDNA4:

1. `calc_rows_per_block` returns four weight rows per MMVQ thread block for 3..8 activation
   columns. One and two columns keep one row per block.
2. `ggml_cuda_should_use_mmvq` gets an RDNA4 arm that lowers the MMVQ/MMQ crossover per weight
   type.
3. At one activation column, the eight-warp RDNA4 weight types take four rows per block as well,
   but only for a matrix that is wide enough; the host chooses the variant per call.

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
changes only through the one-column rule below; prompt batches of hundreds of columns are outside
the range that changes at all.

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

## One column: four rows for a wide matrix

At one output column the block computes a single weight row, so a large matrix is covered by as many
blocks as it has rows and each of them re-reads the one activation column. Four rows per block
amortize that read there too, but they also divide the number of blocks by four, and a narrow matrix
then no longer fills the compute units.

The rule is therefore conditional and is evaluated on the host for every call
(`mmvq_rdna4_alt_rows`):

- the RDNA4 parameter table is in use, and
- the call has exactly one activation column, and
- the weight type is one of the RDNA4 eight-warp types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q4_K,
  Q5_K, Q6_K, IQ4_NL, IQ4_XS), and
- the weight has at least `MMVQ_RDNA4_ALT_ROWS_MIN_NROWS` = 768 rows, the narrowest matrix for which
  the four-row block measured faster than the one-row block.

The decision is passed to the kernel as a template flag, so the row count stays a compile-time
constant. The flag is combined with a constant that is only true for HIP builds and with the
eight-warp condition, so no other backend and no other type compiles the extra kernel. The existing
small-k and halved-iteration one-column variants take precedence over it.

## What it does not touch

- `MUL_MAT_ID` (MoE experts): multi-token expert matmuls use their own kernel with its own fixed
  rows per block, and the MoE MMVQ/MMQ limit is a separate one.
- Fused gate/up MMVQ, which requires a single activation column.
- One-column calls on a weight with fewer than 768 rows, on a type that is not an eight-warp
  type, or on a non-HIP build: they keep one row per block.
- Prompt processing, which is already on MMQ.
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
same values as before, at one column as well as at 3..8 columns. A weight whose row count is not a
multiple of four is safe because the kernel clamps the row reads of the last, partial block to the
last valid row; the duplicated row contributes a partial sum that is never stored.

Moving 5..8-column calls to MMQ is a real numerical change and not only an accumulation-order
effect: MMVQ and MMQ quantize the activations separately, and they approximate the constant term of
Q4_K and Q5_K differently - MMVQ multiplies the block minimum by the sum of the quantized activations
times their scale, while MMQ's quantizer stores the sum of the original float activations. Q6_K has
no minimum term, so there only the accumulation order differs. No new approximation is introduced:
the existing MMQ one now applies from five columns instead of from nine, so outputs at 5..8 columns
differ from the MMVQ ones in the same way outputs at nine columns and above always did.

## How to verify

- `test-backend-ops -o MUL_MAT` covers the affected widths, including K-quant weights at 1..8
  columns with a row count that is neither a multiple of four nor below the 768-row bound.
- The one-column rule is visible as a decode-throughput difference (`llama-bench -n 128`) on a model
  whose projections have at least 768 rows and an eight-warp weight type.
- The cost curve over the column count is visible in
  `llama-bench -ngl 999 -fa on -b 16 -ub 16 -n 0 -p 1,2,3,4,5,6,7,8,9,16 -d 0,8192`: milliseconds
  per call is the column count divided by the reported tokens per second. With these tables the
  curve no longer falls at the crossover.
- Multi-slot decode throughput, the non-speculative way to reach 3..8 columns per call, is visible
  in `llama-batched-bench -npl 1,2,4,8`.

# 12-column tile attention for GQA-12 head groups

## What it is

The tile flash-attention dispatch (`ggml/src/ggml-cuda/fattn-tile.cuh`) only knows blocks of 8, 4
and 2 columns per K/V head, where a column is one head of the query group. A GQA ratio of twelve is
not a power of two, so it falls back to four columns and reads every K/V head three times.

On HIP, a single query row whose group size is a multiple of twelve now launches a 12-column block
for 256-wide heads: 192 threads, six warps of two columns each, which is the warp count that divides
twelve. Each K/V head is read once. The configuration tables of both AMD tile paths get the matching
256/256/12 entry; the kernel itself is the existing tile template instantiated with one query row
and twelve columns.

## When it applies

- a HIP build, and
- head sizes 256 for both K and V, and
- exactly one query row, which is token generation, and
- a GQA ratio that is a multiple of twelve, and
- the conditions under which the tile path already uses the GQA optimization: a mask is present,
  there is no ALiBi slope, and the KV length is a multiple of the kernel's KV stride.

Prompt processing, other head sizes and other GQA ratios keep the upstream dispatch.

## Limits

- The condition is a multiple of twelve, so ratios of 24, 36 and 48 also take the 12-column block
  and read each head 2, 3 or 4 times - still fewer than the fallback, but not the single read that
  a ratio of exactly twelve gets.
- The path is compiled and selected for every HIP target, not only for RDNA4. Non-RDNA AMD devices
  have a 64-lane physical warp while the tile path launches with a fixed 32-lane warp width, and
  that combination was not measured. `GGML_HIP_FATTN_GQA12=0` is the way to take it out of the
  selection there.
- The block shape exists for 256-wide heads only.

## How to verify

- `test-backend-ops -o FLASH_ATTN_EXT` contains two cases with 12:1 head groups and F16 K/V at KV
  512 and 8192; they exercise exactly this block shape.
- An A/B against `GGML_HIP_FATTN_GQA12=0` on a model with a GQA ratio of twelve shows the effect on
  the `FLASH_ATTN_EXT` operation time and on decode throughput at depth.
