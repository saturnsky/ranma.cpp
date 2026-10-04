# EXL3 weights (HIP)

## What it is

GGUF files can hold the weights of an [ExLlamaV3](https://github.com/turboderp-org/exllamav3) EXL3
checkpoint unchanged: the trellis tiles of the mul1, mcg and 3inst codebooks keep their bytes, and the
input and output scales of each weight become `<name>.rot_in` / `<name>.rot_out` tensors. The matrix
product is the op `GGML_OP_MUL_MAT_HAD` (`ggml_mul_mat_had`), `rot_out * H(W^T H(rot_in * x))` with the
Hadamard transforms of EXL3. The loader gives every EXL3 weight its rotations, so the model graphs only
call `build_mm`.

- Types: `GGML_TYPE_EXL3_*`, IDs 256 + 16 * codebook + code (mul1 256..266, mcg 272..279, 3inst
  288..295), and the row codec types 304..311 for the n-gram embedding tables of Qwen3.8. A file with
  these types is read only if it has the EXL3 metadata the converter writes (`quantize.exl3.version`,
  `quantize.exl3.codebook`); a file of another program that uses the same IDs is refused at load time.
- Conversion: `convert_hf_to_gguf.py <EXL3 checkpoint directory>` converts a checkpoint whose
  `quantization_config.quant_method` is `exl3` through the existing model class of the architecture.
  MTP layers are not converted.
- Backends: the CPU backend computes every case; HIP has GEMV kernels for decode and, on RDNA4, a WMMA GEMM
  for prompt processing. Routed expert banks work with host-direct weights and the expert cache, whose
  slots also hold the rotations of their expert.

## Checked models

Converted and run on the target environment (Radeon AI PRO R9700, gfx1201): `llama` (Llama 3.2 1B),
`qwen3moe` (Qwen3-30B-A3B), `lfm2moe` (LFM2.5-8B-A1B), `qwen4exp` (Qwen3.8-Flash-Next, with the n-gram
tables) and `deepseek4` (DeepSeek-V4-Flash). Loading is not limited to these architectures; others are
untested.

## Precision

The GEMM of prompt processing takes the activations in fp16, the precision class of the official ExLlamaV3
kernels. An op that asks for `GGML_PREC_F32` with `ggml_prec_set_src(op, GGML_PREC_F32, 1)` runs the
F32-input GEMV for all of its rows instead (8 rows per pass, slower for long prompts); the CPU backend
always computes in F32. `GGML_CUDA_MUL_MAT_HAD_PREC=f32` sets this for every op of a process, to test the
F32 path of a model.

## RDNA3

Not tested on hardware. The HIP sources compile for gfx1100/1101/1102. The GEMV kernels use only RDNA3
instructions, so decode is expected to run on the GPU. The WMMA GEMM is RDNA4 only; on RDNA3 the rows of a
prompt run on the GEMV after a routing pass, which is slower.

## Qwen3.8 hyper-connection fusion

The gated residual mix and the combine of each qwen4exp hyper-connection site run as fused decode kernels
(up to 5 and 8 tokens) instead of small MUL_MATs and elementwise ops. The fused kernels accumulate in FP32,
so the results differ slightly from the unfused graph. `GGML_CUDA_HC_GATED_FUSION=0` turns the fusion off.
