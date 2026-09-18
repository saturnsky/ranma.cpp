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

The variables are read once per process, so they have to be set before the process starts. A build
with `GGML_CUDA_FORCE_MMQ` compiles the BLAS policy out entirely.

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
