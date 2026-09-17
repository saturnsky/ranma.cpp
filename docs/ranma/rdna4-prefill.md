# RDNA4 prefill selection (HIP)

Prompt processing on an RDNA4 device runs kernels that are chosen for wide batches. This page
collects the fork's additions to that selection.

## Environment switches

| Name | Default | Effect of the non-default value |
| --- | --- | --- |
| `GGML_HIP_PREFILL_WMMA` | `1` (on) | `0` removes the D512 WMMA attention selection below, so 512-wide heads keep the upstream dispatch. Kept as the reference path for equivalence checks. |

The variable is read once per process, so it has to be set before the process starts.

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
