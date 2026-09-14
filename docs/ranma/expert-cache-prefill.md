# Expert cache in prompt processing (MMQ arena read) and the delta install

## What it is

The expert cache (`expert-cache.md`) keeps the most valuable MoE routed experts in a fixed-address
VRAM arena and maps every expert id to a slot through a per-layer device table. The first version
of the cache was read by the MMVQ kernels only, so it was a decode-only feature: a prompt token
batch runs on MMQ, which read every expert from host memory over PCIe.

MMQ now picks the source of each expert matrix the same way MMVQ does. For the channel of expert
`e` in a layer whose table is `slots`:

```
slots[e] >= 0 ? arena + slots[e]*nb[2] : host_tensor + e*nb[2]
```

Both kernels call one device helper, `ggml_cuda_expert_cache_select` in
`ggml/src/ggml-cuda/common.cuh`. Prompt processing therefore hits the same cache the decode step
hits, with the same plan, the same slot tables and no extra copy.

The same commit turns an install into a delta transaction. `plan_install` (`expert-plan.h`)
decides, without touching device memory, which selected experts keep the slot they already have and
which have to be copied; `l1_arena::stage`, `execute` and `publish` are the three steps, and the
slot tables are blanked while the mover runs so that no captured graph reads a slot in flight.

## Why it exists

A resident expert costs nothing extra to read from the arena, so any kernel that reads a routed
expert should read it there. The gain is not the same as for decode. A decode step reads a whole
expert matrix for one token, so it is bound by the PCIe read and a hit removes the read entirely. A
prompt ubatch reads almost every expert of a layer once and reuses each for many tokens, so the read
is already amortized and only a high resident fraction shows up in the prefill rate.

The delta install exists because two consecutive plans of one conversation differ by little. A full
refill of a 20 GiB budget moves 20 GiB at every request boundary; a delta moves a few hundred MiB.

## How it is layered

- `ggml/src/ggml-cuda/common.cuh` holds `ggml_cuda_expert_source` and
  `ggml_cuda_expert_cache_select`, the only place that knows the "slot or expert id" rule.
- `ggml/src/ggml-cuda/mmq.cu` looks up the tensor with `ggml_cuda_expert_lookup_tensor(src0)` when
  src0 lives in the HIP host buffer type and the op is a MUL_MAT_ID, and puts the arena base and the
  slot table into `mmq_args` (both null otherwise, including for every dense matmul).
- `ggml/src/ggml-cuda/mmq.cuh` calls the helper at the three places where `mul_mat_q` turns a
  channel into an x pointer (the tiling branch, the stream-k loop and the stream-k last tile) and
  passes the resulting channel base to `mul_mat_q_process_tile`. The stream-k fixup kernel does not
  read src0 and is unchanged.
- `expert-plan.h::plan_install` is the single place that knows where every expert lives, which is
  what lets one mover serve both the promotion of a new resident and the retention of an old one.

Two details of the MMQ path are worth naming.

**64-bit channel base.** `mul_mat_q` used to fold the channel into `offset_x`, an `int` count of
quantization blocks. A 20 GiB arena does not fit that for every quantization, so the channel term is
now a 64-bit byte offset applied to the pointer (`x + (int64_t) channel*nb[2]`) and `mmq_args`
carries `stride_channel_x_bytes`. The row and sample terms stay in `offset_x`. This also removes the
same latent overflow on the plain host tensor path.

**Arena tail.** MMQ loads whole `MMQ_ITER_K` (256 element) K tiles, so for a matrix whose row length
is not a multiple of `MATRIX_ROW_PADDING` it reads past the last row. This model's `ffn_down_exps`
have k = 800. Inside the arena the read lands in the next slot, which holds finite quantized data
and cannot change a result because the matching src1 columns are zero; past the last slot it must
land in zeros. The host buffer type pads every quantized tensor by
`ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)` and zeroes it
(`docs/ranma/host-direct-moe.md`); each per-class, per-kind arena carries exactly that many zeroed
bytes past its last slot, with a 512-byte floor, charged to the budget like the slot tables. Stale
quantized bytes in that tail produce NaN even when multiplied by zero, so every storage the kernels
read (arena, host arena, ring slot) carries the same zeroed tail.

## Options

None. The arena read and the delta install follow the expert cache being on.

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. Model:
Qwen3.8-Flash-Next UD-Q4_K_XL, every routed expert in host memory, host-direct on with
`GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`, `--load-mode none`.

**This commit alone.** `llama-bench` PP512 / TG128, `-r 1`, depths `8192,0,8192` in one model load
with the first pair discarded, inclusive, 3072 MiB budget, Warm from the same profile:

| | PP512 @ 0 | TG128 @ 0 | PP512 @ 8192 | TG128 @ 8192 |
|---|---:|---:|---:|---:|
| MMVQ only (previous commit) | 535.87 | 30.16 | 534.63 | 29.17 |
| MMQ reads the arena too | 547.12 | 30.06 | 546.30 | 29.50 |

At a 3 GiB budget the resident fraction is 4 % of the expert bytes, and prompt processing moves by
2 %. The delta path is what the install log shows: a first install of the 3072 MiB budget writes
1027 slices in 70.3 ms, and the installs that follow retain 958 to 974 of those 1027 slices and copy
53 to 69 (159 to 207 MiB) in 10.8 to 12.3 ms.

**Correctness.** Cache off and on must produce the same logits. Eight prompts of 18 to 197 tokens,
so every prefill goes through MMQ, greedy, 48 tokens, top-3 logprobs: identical with the cache off
and on, on the inclusive 3072 MiB and 20000 MiB budgets of this commit and on every later
configuration of the cache (`expert-cache.md`). `test-backend-ops -b ROCm0 -o MUL_MAT_ID
--host-weights` passes with the host-direct environment set, and `llama-perplexity -c 128 -b 128
--chunks 6` at `MAX_BATCH=512` stays finite.

## What it costs

- One table read per expert channel in the MMQ prologue, next to the existing MMVQ one. No extra
  copy, no extra allocation on the hot path.
- The arena tail: `ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)` bytes per
  size class and kind, at least 512, inside the budget. For this model that is 9 x a few hundred
  bytes.
- The channel base is one 64-bit multiply instead of a 32-bit one, per tile.
- The delta install: one pass over the plan on the host before the device is drained; the copies
  are only those of the experts that changed.

## Limitations

- **Host-direct only.** The arena is read on the in-place MMQ path, that is for token counts up to
  `GGML_CUDA_HOST_DIRECT_MAX_BATCH`. A larger batch, or host-direct off, still copies the whole
  expert weight to VRAM through the scheduler and reads the copy; that path is untouched. Trimming
  that copy with the resident set is a possible later addition, not done here.
- **The plan is the decode plan by default.** Prompt processing is profiled into its own bank, but
  the plan installed while a prompt is processed is the one chosen for generation unless
  `--expert-prefill-swap` is on (`expert-cache-banks.md`).
- **The gain needs a large resident fraction.** At a 3 GiB budget the prefill rate does not move.
  The feature is free, but it is not a reason to raise the budget by itself; the decode gain is.
- **No fused gate/up on this path.** MMQ has no gate fusion, so the table-sharing rule the MMVQ hook
  needs does not apply here.

## Revision

Every number above comes from the binary built from this commit and the commits that follow it on
ranma `ccd8fd1e9` (upstream llama.cpp `093a2f86c`, plus the fork's earlier commits). The exact
commits are kept on the dated snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that
branch is never rebased.
