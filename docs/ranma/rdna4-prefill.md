# RDNA4 prefill selection (HIP)

Prompt processing on an RDNA4 device runs kernels that are chosen for wide batches. This page
collects the fork's additions to that selection.

## Environment switches

| Name | Default | Effect of the non-default value |
| --- | --- | --- |
| `GGML_HIP_PREFILL_WMMA` | `1` (on) | `0` removes the D512 WMMA attention selection below, so 512-wide heads keep the upstream dispatch. Kept as the reference path for equivalence checks. |
| `GGML_HIP_PREFILL_BLAS` | `1` (on) | `0` removes the F16 BLAS matmul policy below, so those matmuls stay on MMQ. Kept as the reference path for equivalence checks. |
| `ROCBLAS_USE_HIPBLASLT` | unset | The F16 BLAS policy arms only when this is exactly `1`, because it selects the library the measurements were taken with. |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | unset | Setting it disarms the F16 BLAS policy, because it overrides the compute type the GEMM would use. |
| `GGML_CUDA_SKINNY_F32` | `1` (on) | `0` sends the skinny F32 products below to hipBLAS again. |

The variables are read once per process, so they have to be set before the process starts. A build
with `GGML_CUDA_FORCE_MMQ` compiles the BLAS policy out entirely; the skinny F32 kernels do not depend
on it.

## WMMA attention for 512-wide heads

### What it is

The F16 MMA flash-attention kernel (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`) is the wide-tile
attention path used on hardware with matrix cores. On AMD its device code refused head sizes above
256 on every target, so a model with 512-wide heads processed its prompt in the tile kernel even
though the wide-tile kernel exists for that shape.

The kernel now compiles its DKQ = DV = 512 instances when it is built for RDNA4. Every other AMD
target keeps the DKQ <= 256 limit, so nothing else changes what it compiles or selects. The HIP
kernel chooser (`ggml/src/ggml-cuda/fattn.cu`) selects the MMA kernel for that shape.

### When it applies

All of the following must hold:

- the device is RDNA4 and has WMMA, and
- K and V heads are 512 wide, and
- the batch has at least 16 query rows, and
- the GQA ratio is 8, and
- the GQA optimization already applies, and
- K and V are F16.

Everything else keeps the upstream selection: heads up to 256 follow the existing WMMA rule, fewer
than 16 query rows keep the tile kernels, and a quantized KV cache is never selected for this path.
Single-token generation therefore does not change.

### Limits

The selection is deliberately narrow: one head size, one GQA ratio, F16 KV and a row bound. The
kernel itself only requires that the product of its two column counts is at least 16, which is two
query rows at a GQA ratio of 8, so the 16-row bound is a conservative policy choice rather than a
kernel requirement; the 2..15-row range was not compared against the tile kernel. A speculative
verification batch of 16 rows or more with this head shape also takes the kernel.

### How to verify

- `test-backend-ops -o FLASH_ATTN_EXT` compares the selected kernel against the CPU reference for
  512-wide heads at several query-row counts, including rows just below and just above the bound.
- An A/B against `GGML_HIP_PREFILL_WMMA=0` on a model with 512-wide heads shows the effect on
  prompt throughput; a model with narrower heads is the control and must not move.

## F16 BLAS for wide dense Q2_K/Q6_K/IQ2 matmuls

### What it is

For a few quantized weight types, dequantizing both operands to F16 and calling hipBLASLt once is
faster than MMQ as soon as the activation matrix is wide, which is the prompt-processing case. A
dense `MUL_MAT` therefore converts both operands to F16 per operation and runs the GEMM when the
weight type is one of the measured ones and the activation matrix has at least the listed number of
columns:

| Weight type | Minimum activation columns (tokens) |
| --- | ---: |
| Q2_K | 64 |
| Q6_K | 256 |
| IQ2_S | 512 |
| IQ2_XS | 512 |

### When it applies

In addition to the type and column count:

- a HIP build, not compiled with `GGML_CUDA_FORCE_MMQ`, running on an RDNA4 device, and
- `ROCBLAS_USE_HIPBLASLT=1` with no `GGML_CUDA_CUBLAS_COMPUTE_TYPE` override, and
- default operation precision, and
- a single batch (all higher dimensions of both operands are 1), and
- contiguous weight, activation and destination, and
- the weight in an ordinary device buffer of the same device, and
- a conversion that fits the scratch cap below.

The test is on the operation being a `MUL_MAT`, so the per-expert fallback of `MUL_MAT_ID`, which
calls the same function, is excluded together with every MoE weight.

### Padded row pitch

Both F16 copies are written with a row pitch of K + 64 elements, and the GEMM is given that leading
dimension. hipBLASLt on RDNA4 slows down when the leading dimension in bytes is a multiple of
16 KiB, which for F16 operands is exactly K = 8192, 16384, 32768 and 65536; 64 extra elements break
that alignment and change nothing at other K.

This needs converters that write their rows at a destination pitch instead of contiguously, so
`convert.cu` carries a pitched variant of the K-quant block dequantizer and of the unary converter,
exported as `ggml_get_to_fp16_pitched_cuda`. It returns a converter for the four weight types above
and for F32 activations, and nothing for any other type, so a type without a pitched converter can
never reach the GEMM. The K-quant converters require K to be a multiple of the super-block size.
The padding columns are written by nobody and read by no GEMM - the leading dimension only strides
the rows, and the K extent passed to the GEMM is the real one - so they are left uninitialized.

### Scratch and memory

The conversion buffers come from the backend pool per operation and are returned to it afterwards;
no persistent F16 copy of a weight is kept, and the quantized weights stay where they are. For a
weight of K x M and an activation of K x N, the conversion needs `2*(K+64)*(M+N)` bytes, and the
operation only qualifies while that stays within 512 MiB. The cap bounds the conversion scratch
only, not the library workspace, and the pool can retain the VRAM it handed out.

### Limits

- The boundaries were selected on the padded path with a fixed rule: the first column count at which
  every measured count at and above it, under both weight-reuse conditions and in all passes, was at
  least 3 % faster than MMQ. IQ2_S met that rule slightly below its listed boundary and was rounded
  up; Q2_K was faster at every measured width, so its boundary is the smallest measured one.
- They come from one GPU, one library version, a fixed set of K/M shapes and at most 1024 columns.
  Other shapes, wider batches and other hardware carry no measured guarantee, and the IQ2_XS margin
  is the thinnest of the four.
- Every other quantized type stays on MMQ; for the large-block types BLAS was much slower at the
  same shapes.
- The F16 operands round differently from MMQ's Q8 activation quantization, so a qualifying
  operation produces slightly different values than it did on MMQ, in the same way any
  dequantize-and-GEMM path does.

### How to verify

- `test-backend-ops -o MUL_MAT` compares the selected path against the CPU reference for the four
  types above, at and above their boundaries, including a single-column case and a weight whose row
  count leaves a tail.
- An A/B against `GGML_HIP_PREFILL_BLAS=0` on a model that uses one of the four types shows the
  effect on prompt throughput. Running without `ROCBLAS_USE_HIPBLASLT=1` is the same as running
  with the policy off.

## Fixed-order kernels for skinny F32 matmuls

### What it is

Some models multiply a prompt batch with small F32 weight matrices: a handful to a few hundred weight
rows against every token of the ubatch. In Qwen3.8-Flash-Next these are the hyper-connection
injection (4 x n x 10240, weight rows x tokens x K), the alpha/beta projections of the SSM layers
(48 x n x 2560) and the MoE router (512 x n x 2560). On RDNA4 no ggml kernel took them: MMVF stops at
8 columns, MMF has no F32 path without F32 matrix cores, and MMQ is for quantized weights. They went
to hipBLAS.

`ggml/src/ggml-cuda/mmvf.cu` now carries two kernels for them:

- up to 8 weight rows, a streaming kernel: every wave reads a few token rows once and keeps all weight
  rows of the current K step in registers; K is split over the waves of a block and the partial sums
  are added in a fixed order;
- above that, a register-tiled SGEMM with both operands staged through shared memory. K is split
  within the block and, when the grid would be too small to fill the device, over several blocks
  whose partial results a second kernel adds in index order.

### Why it exists

hipBLAS, and hipBLASLt underneath it, chooses its solution for a GEMM once per process. For these
shapes the choice was not stable: two processes with the same model and input could run the same
product at very different speeds and with a different summation order, so both the prompt
throughput and the logits depended on the process start. The shapes are also far from what a library
GEMM is tuned for: the injection product is little more than one read of the activation, and the BLAS
path took many times that.

The new kernels have one summation order, fixed by the launch configuration and independent of the
process, and use no atomics, so a product gives the same bits in every run.

### When it applies

`ggml_cuda_mul_mat` tries the kernels last, just before the BLAS fallback, so every earlier selection
keeps its precedence. All of the following must hold:

- the device is RDNA4, and
- weight, activation and result are F32, and
- both operands are single 2D matrices (all higher dimensions 1), and
- rows have a unit element stride, K is a multiple of 4, and the row pitch and base address of both
  operands are 16-byte aligned, and
- the activation has more than 8 columns (above the MMVF limit) and the weight at most 512 rows, and
- the weight is not in host memory of a discrete device.

Single-token generation and small verification batches (8 columns or fewer) keep MMVF. Other element
types, including the BF16 indexer products of the same model, stay on BLAS.

### Numerics

The kernels compute the same sums as the BLAS path in another order, so the results differ at
rounding level. In a model whose MoE router reads such a product, a rounding difference can move an
expert across the router's top-k boundary for some tokens, so the end-to-end difference is larger
than the rounding itself. Against the BLAS path it is of the same size as the difference between two
BLAS processes that landed on different hipBLASLt solutions.

### Measured effect

Base revision: the series of ranma_20260922 (`e48103e1e`) rebased on upstream `ec5a12b85`, with this patch; Radeon AI PRO R9700 (PCIe 5.0 x16) with the setup of
[benchmark.md](benchmark.md) (-30 % power limit, 0 mV voltage offset, headless), Qwen3.8-Flash-Next
UD-Q4_K_XL with the exclusive expert cache of 20480 MiB (warm, `--expert-l2-mib -1`, host-direct with
`GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`). The reference is the same binary with `GGML_CUDA_SKINNY_F32=0`
and `ROCBLAS_USE_HIPBLASLT=0`: with hipBLASLt the reference itself moves between processes, while
rocBLAS alone gave the same speed and the same bits in every process.

`llama-perplexity -c 4096 --chunks 8` over wikitext-2, against the BLAS path:

| run | mean KLD | same top token | PPL |
| --- | ---: | ---: | ---: |
| BLAS path, second process | 0.000096 | 99.945 % | 3.1644 |
| this patch | 0.0212 | 95.14 % | 3.1688 |

Two processes of the BLAS path with hipBLASLt that land on different solutions differ by a mean KLD of
about 0.020 on the same measurement. With the patch the result no longer depends on the process or on
`ROCBLAS_USE_HIPBLASLT`: repeated `llama-perplexity -c 512 --chunks 32` processes, with and without
hipBLASLt, printed the same perplexity.

`llama-bench` with the protocol of [benchmark.md](benchmark.md) (one process per setting, a discarded
pass at depth 65536 first, five idle minutes between processes), PP512 and TG128 in t/s:

| setting | PP @0 | PP @4096 | PP @8192 | PP @32768 | PP @65536 | TG @0 | TG @4096 | TG @8192 | TG @32768 | TG @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `GGML_CUDA_SKINNY_F32=0` | 905.1 | 899.4 | 868.1 | 765.5 | 638.5 | 50.22 | 48.87 | 48.69 | 46.66 | 44.27 |
| default | 1066.6 | 1006.6 | 975.9 | 830.4 | 691.4 | 50.38 | 49.05 | 48.42 | 46.67 | 44.63 |

Prompt processing gains 18 % at depth 0 and 8 % at depth 65536. Generation does not use the path and
stays within the spread of single runs.

Per product, `test-backend-ops perf -o MUL_MAT` (one op per graph, so the activation stays in the
last-level cache), in microseconds; the hipBLASLt column is one process and moves with the solution
that process chose:

| product (rows x tokens x K) | this patch | rocBLAS | hipBLASLt |
| --- | ---: | ---: | ---: |
| 4 x 512 x 10240 | 13.7 | 492 | 583 |
| 48 x 512 x 2560 | 24.2 | 191 | 225 |
| 512 x 512 x 2560 | 111 | 155 | 164 |

In a standalone graph of 128 such products, each with its own activation so that it is read from
memory, the 4 x 512 x 10240 product took 37 us, close to one read of its 21 MB activation, against
534 us with rocBLAS and 595 to 606 us in three hipBLASLt processes. With uniform random inputs, the
maximum absolute error against a double-precision reference was 2 to 20 times smaller than
hipBLASLt's for the three products.

### Limits

- The kernel configurations were tuned for the three products above. Other shapes inside the bounds
  are computed correctly and take the nearest configuration, but carry no measured speed guarantee.
- Weights with more than 512 rows, batched products and non-F32 types keep their previous path.
- Only RDNA4 selects the kernels; their launch assumes its wave32 execution.

### How to verify

- `test-backend-ops -o MUL_MAT` compares the kernels against the CPU reference for the three model
  products at column counts on both sides of the MMVF limit (1 to 512), for odd row counts (2, 5, 13,
  100, 511) and for operands with a padded row pitch.
- `test-backend-ops perf -o MUL_MAT` includes the three products at 9, 64, 256 and 512 columns; run it
  with `GGML_CUDA_SKINNY_F32=0` for the BLAS side.
- Two `llama-perplexity` processes of the same model print the same numbers with the default, whatever
  `ROCBLAS_USE_HIPBLASLT` is set to.
