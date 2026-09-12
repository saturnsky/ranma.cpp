# Host-direct MoE weights (HIP)

## What it is

A discrete GPU normally cannot use a weight that sits in host memory: the scheduler either copies the weight to VRAM
for every op, or it leaves the op on the CPU backend. This feature adds a third option for one narrow case. A backend
can declare that a given op may read a given `src` in place from the device's own host buffer type, and the scheduler
then keeps the op on that backend and creates no copy.

The HIP backend declares this for `MUL_MAT_ID` with a quantized `src0`, up to a token count limit. The expert weights
stay in mapped pinned host memory and the `MUL_MAT_ID` kernels (MMVQ, or MMQ above the MMVQ batch limit) read them
over PCIe, one expert row block at a time, while every other tensor of the op stays in VRAM.

Because MMQ loads whole K tiles (`MMQ_ITER_K` elements) of `src0`, a quantized matrix whose row length is not a
multiple of the tile is read past its last row. A device buffer covers that read with zeroed row padding
(`MATRIX_ROW_PADDING`, see `[TAG_ALLOC_SIZE_EXPAND]` in `ggml-cuda.cu`); with host-direct enabled the host buffer
type allocates and clears the same padding, so a weight looks the same to the kernel wherever it lives. Without the
padding the last row of the last expert of such a tensor reads whatever follows it in the buffer, and a non-finite
scale there turns into NaN activations for the tokens routed to that expert.

## How the backend declares it

Through the proc-address extension point, the same way as `ggml_backend_register_host_buffer` and
`ggml_backend_get_features`: a backend's `get_proc_address` answers the name `"ggml_backend_host_direct_op"` with a
`ggml_backend_host_direct_op_t`. `ggml_backend_sched_new` resolves it once per backend and caches it;
`ggml_backend_sched_buft_supported` calls the cached pointer, or skips the whole rule when the backend returned NULL.
`ggml_backend_dev_host_direct_op` is a convenience wrapper that resolves the address on every call, for the few
callers outside the scheduler.

It is deliberately *not* a new member of `ggml_backend_device_i`. `ggml-base.dll` reads the iface struct that a
backend DLL defines, so adding a member in the middle of it silently changes the backend ABI while
`GGML_BACKEND_API_VERSION` stays 2: an unpatched `ggml-hip.dll` loaded by a patched `ggml-base.dll` passes the version
check and then has `event_new` called where the core expects `host_direct_op`. Swapping a single backend DLL between
builds is routine here, so the proc address is the safe shape. It also needs no version bump and no edit to any other
backend.

## Why it exists

A large MoE model does not fit in 16 GiB of VRAM, so its routed experts live in system RAM (`-ncmoe`, or an `-ot`
override that sends `ffn_*_exps` to the CPU). Each decode step then touches only the few experts that the router
selected. Computing that on the CPU backend is bound by CPU memory bandwidth and by the round trip through the CPU
backend; reading the same bytes straight from host memory into a GPU kernel is bound by PCIe bandwidth only, and the
GPU needs far fewer cycles per byte.

Prompt processing was expected to be the opposite case: a 512-token batch reads every selected expert, so a single
copy to VRAM looked cheaper than reading host memory from the kernel. The token count limit exists to separate the
two regimes. The measurements below show that once MMQ reads the weight in place, the limit is not needed on this
machine; see "Open questions".

## Upstream context

Reading weights in place from a host buffer is already upstream behaviour for *integrated* devices
(`ggml_backend_cuda_device_supports_buft` returns true for the host buffer type when `integrated` is set). That path
has a troubled history:

- **#15034**: output corruption on integrated devices. The cause was a race on *input* tensors: an input in a host
  buffer could be reallocated or rewritten while a kernel was still reading it.
- **#16308**: disabled the `integrated` host-buffer rule to stop the corruption.
- **#24233**: restored it for HIP.
- **#28604**: reverted that again.
- **#27311**: the pending proper fix, a UMA ring buffer for host-resident inputs.

This feature does not depend on that fix and cannot hit that race. It admits only tensors whose buffer has usage
`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, which are written once at load time and never reallocated or rewritten. It also
leaves `supports_buft` alone, so nothing else changes for integrated devices.

## Environment variables

The switch is environment-only in this patch, like `GGML_OP_OFFLOAD_MIN_BATCH`. There is no CLI flag.

| Variable | Default | Meaning |
|---|---|---|
| `GGML_CUDA_HOST_DIRECT` | unset | Unset or `0` disables the feature. Any other value enables it. No effect outside HIP. |
| `GGML_CUDA_HOST_DIRECT_MAX_BATCH` | `0` | Token count limit (`op->ne[2]`) up to which a weight may be read in place. `0` means the per-type MMVQ `MUL_MAT_ID` limit, `get_mmvq_mmid_max_batch(type, cc)`. Set it to at least the ubatch size (for example `512`) to read in place during prompt processing as well; see "Measured effect". |
| `GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB` | `1024` | Host buffers of at least this size take the coarse-grained path: `VirtualAlloc`, then `hipHostRegister(Mapped \| CoarseGrained)` after the model is loaded. `0` disables it, so every host buffer is a plain `hipHostMalloc(Mapped)`. Windows only. |

When the feature is enabled, the host buffer type allocates with `hipHostMallocMapped` instead of `cudaMallocHost`,
because a kernel can only dereference a host pointer that was mapped into the device address space.

A buffer on the coarse path is *not* readable by a kernel until it has been registered. `llama_model::load_tensors`
calls the backend's `ggml_backend_finalize_host_buffer` proc address once the weights are written, and until that
succeeds `host_direct_op` returns false, so the scheduler keeps using the copy or CPU path.

## Recommended profile

```
GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512
```

`MAX_BATCH` should be at least the ubatch size (`-ub`, default 512), so that prompt processing also reads the experts in
place. The defaults are deliberately left as they are: the feature stays off unless asked for, and the limit keeps its
conservative MMVQ meaning, because the measurements below come from one machine and one ubatch size. The profile is
what those measurements support.

## Measured effect

Measured 2026-09-13 on the development machine: Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16,
Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. The GPU runs with a -75 mV voltage offset and a -30 % power
limit for every measurement here, and headless (no display attached; the CPU's integrated GPU drives the display), so
all 32 GiB are available to the model. Host-resident weights are bound by the PCIe link, so a narrower or slower link
changes these numbers first.

Revision: "off" and "on" are the same binary, built from the code of this commit (the one that adds this page)
on upstream llama.cpp `43f3dda62` (ggml-org master, 2026-09-11) plus the fork README commit `d4e7123f1`.
The documentation was edited after the measurement, the code was not. The unpatched reference of the warm-up
note is `d4e7123f1` itself, whose tree is upstream `43f3dda62` plus README and CI files. The exact commits are kept
on the dated branch `ranma_20260914`, which is never rebased.

Model: Qwen3.8-Flash-Next UD-Q4_K_XL, 103.68 GiB, 48 MoE layers with 512 experts, 74.1 GiB of routed experts
(gate/up q4_K, down q5_1, five layers with a q8_0 down projection). `llama-bench -ngl 999 -t 16 -fa on -ctk f16
-ctv f16 -p 512 -n 128 -r 3 -lm none -d 65536,0,8192,32768,65536`, one process per configuration, same binary, only
the environment differs. The first depth entry is a warm-up pass and is discarded; see the warm-up note below for why.
"off" is the upstream behaviour: the selected experts are copied to VRAM per op for batches of 32 tokens and more, and
the CPU backend computes smaller batches. "on, limit 4" is `GGML_CUDA_HOST_DIRECT=1` with the default limit (the
per-type MMVQ limit, 4 for these types), so prompt processing still takes the copy path. "on, no limit" adds
`GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`, so prompt processing reads the experts in place through MMQ.

Two placements:

- **R9700**: expert layers 35..47 in VRAM, `-ncmoe 35`, 27.7 GiB of device buffers at depth 65536.
- **RX 9070 XT (emulated)**: this is not a measurement on that card. It is the same R9700 with the placement a
  16 GiB card would use: expert layers 45..47 in VRAM, `-ncmoe 45`, 13.1 GiB of device buffers at depth 65536.
  Both cards are the same Navi 48 die with 64 CUs and 640 GB/s GDDR6; the boost clock differs by under 2 %, which
  is below the run-to-run drift observed here. A 9070 XT with a display attached has 1-2 GB less VRAM, which for
  this model is about one expert layer, so it may need a higher `-ncmoe`.

### RX 9070 XT placement (`-ncmoe 45`), t/s

| depth | tg128 off | on, limit 4 | on, no limit | pp512 off | on, limit 4 | on, no limit |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 13.38 | 21.55 | 21.43 | 321.3 | 349.8 | 412.2 |
| 8192 | 13.24 | 20.93 | 20.82 | 314.9 | 339.6 | 412.3 |
| 32768 | 12.48 | 19.14 | 19.26 | 301.1 | 291.2 | 430.7 |
| 65536 | 11.60 | 17.45 | 17.49 | 295.5 | 286.7 | 393.8 |

### R9700 placement (`-ncmoe 35`), t/s

| depth | tg128 off | on, limit 4 | on, no limit | pp512 off | on, limit 4 | on, no limit |
|---|--:|--:|--:|--:|--:|--:|
| 0 | 16.15 | 24.15 | 24.42 | 535.0 | 454.4 | 696.3 |
| 8192 | 15.39 | 23.51 | 23.51 | 525.5 | 435.4 | 678.3 |
| 32768 | 14.50 | 21.46 | 21.52 | 464.1 | 400.1 | 700.7 |
| 65536 | 13.29 | 19.25 | 19.36 | 419.4 | 363.1 | 593.8 |

What the numbers say:

- **Decode.** 51-61 % faster on the 9070 XT placement and 45-53 % faster on the R9700 placement, at every depth,
  with either limit. The R9700 gain is smaller because 13 of the 48 expert layers already sit in VRAM and are
  untouched by the feature. In an earlier pass without the warm-up, two "off" runs agreed to within 0.3 t/s at every
  depth, so decode drift is far below these gains. The limit makes no difference to decode (within 0.3 t/s).
- **Prompt processing with the default limit** takes the copy path for every 512-token op, so it was expected to
  match "off". It does on the 9070 XT placement within a few percent, but on the R9700 placement it is 13-15 %
  slower than "off" at every depth. The cause is not established; the only difference for a copied op is that the
  source is the mapped, coarse-grained host allocation instead of `cudaMallocHost` memory.
- **Prompt processing with no limit** is faster than the copy path at every measured point: 28-43 % on the 9070 XT
  placement and 29-51 % on the R9700 placement. At 512 tokens per ubatch each selected expert receives only about
  top-k tokens, so MMQ reads a weight tile once, the same bytes the copy path moves, without the staging copy and its
  synchronization. A limit of 2 behaves like the limit of 4 (measured at depths 0 and 8192 without the warm-up pass).
- **This is why the recommended profile sets `MAX_BATCH` to the ubatch size.** No limit is never slower on decode
  and is the fastest prompt processing setting on both placements, while the default limit can cost prompt processing
  against "off". The default itself is unchanged; larger ubatch sizes (`-ub 2048`) and other GPUs have not been
  measured.
- **Warm-up note.** Prompt processing on this platform speeds up with the age of the process, not with depth: with
  the depth order reversed (`-d 65536,0`), depth 0 after the 65536 pass runs 1.2-1.6x faster than depth 0 in a fresh
  process, on the unpatched base commit (copy path) as well as with this feature, and on both placements. A single
  `llama-bench` warm-up run does not reach the steady state; decode moves by a few percent, prompt processing by up to
  a factor of two. It is not VRAM residency: sampled `GPU Process Memory` counters show shared usage constant to within
  0.2 GiB (the registered host experts) and dedicated usage rising only with the KV cache, and a placement 5 GiB below
  the budget shows the same rise. The cause is not identified. The tables above therefore discard a full depth-65536
  pass before the timed runs; numbers taken without it understate prompt processing at shallow depths in every
  configuration, including "off".

## Limitations

- HIP only. The device callback returns false for CUDA, so the code still builds but the feature does nothing.
- The coarse-grained path is Windows only. On other platforms every host buffer is a plain `hipHostMalloc(Mapped)`,
  whose device alias happens to be the host address, so `GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB` has no effect there.
- Quantized `src0` only. An F16 or F32 expert weight is never admitted.
- Only `MUL_MAT_ID`. A dense `MUL_MAT` weight is never admitted.
- The default token count limit is the per-type MMVQ `MUL_MAT_ID` limit (4 for Q4_K on RDNA4). Above it, MMQ reads
  the weight in place. The measurements above show that no limit is the fastest setting for prompt processing on
  both placements and costs nothing on decode, while the default limit can make prompt processing slower than the
  copy path. The default is kept conservative on purpose; use the recommended profile.
- With a limit in effect, token counts between the limit and `GGML_OP_OFFLOAD_MIN_BATCH` (default 32) run the MoE on
  the CPU backend, because the weight is neither admitted here nor large enough for the offload rule to copy it.
- `MUL_MAT_ID` weights are admitted per op, not per graph. This is equivalent to a per-graph decision because all
  `MUL_MAT_ID` nodes of one graph share `ne[2]`. In particular the reserve graph is built at the full batch size, so
  with a limit it takes the copy path and the compute buffer is sized for it.

## Testing notes

The `MUL_MAT_ID` `--host-weights` oracle below was run on gfx1201 on 2026-09-12 and 2026-09-13 (508/508 passed, with
and without the coarse path). That sweep only had matrices whose row length is a multiple of the MMQ K tile, so it
did not catch the missing host padding described under "What it is"; the 2026-09-14 revision adds a real expert
geometry with `k = 800` (q4_K/q5_K gate-up, q5_1/q8_0 down, 512 experts, 10 used, 33 and 128 tokens) to
`test-backend-ops`, which fails without the padding and passes with it.

- **Test the model's own row lengths.** A `k` that is a multiple of 256 never reads past a row, so a sweep made of
  such shapes cannot show a padding problem. The server symptom of the missing padding was NaN logits from the second
  request on, even when the second request repeated the first one: the first prompt was short enough that its
  tokens missed the affected row, and the most likely carrier of the state is a NaN left in a KV cache cell that a
  later request masks but still reads. `llama-perplexity` with `-c 128 -b 128` showed it on the first chunk and is
  the quicker reproduction.
- **The test does not cover the scheduler.** `test-backend-ops` calls `ggml_backend_graph_compute` directly and never
  builds a `ggml_backend_sched`, so `--host-weights` exercises the kernels, the alias resolution and the device hook,
  but not `ggml_backend_sched_buft_supported` and the splits around it. Only a model run reaches the scheduler path.
- **Force the non-identity alias when testing.** A `hipHostMalloc(Mapped)` buffer aliases to its own host address, so
  a test that only uses those cannot expose a bug in the register path or in the offset arithmetic. Lower the coarse
  threshold so that test-sized buffers take the `VirtualAlloc` plus `hipHostRegister` path:

  ```
  GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB=1 test-backend-ops -b ROCm0 -o MUL_MAT_ID --host-weights
  ```

  `--host-weights` puts every quantized weight in the host buffer, so it asks `ggml_backend_dev_host_direct_op` per op
  and reports a case the device would not admit as `not supported [<backend> host-direct]`. A run therefore reports OK
  and skip counts, and the skipped list is worth reading: a type or batch size that is skipped unexpectedly means the
  admission rule and the kernel dispatch have drifted apart.

- **`--host-weights` without `GGML_CUDA_HOST_DIRECT`.** The host buffer type then allocates with `cudaMallocHost`,
  which on HIP is `hipHostMalloc` with default flags, and whether `hipHostGetDevicePointer` returns a usable alias for
  such an allocation is not established here. The admission hook makes the question harmless, because it returns false
  for every op and the whole sweep is skipped, but the result is a run that reports zero tested cases for a
  non-obvious reason, so the flag also prints a warning when the variable is unset.

## Open questions

- Whether the recommended profile holds for larger ubatch sizes (`-ub 2048`) and on other GPUs. Only a 512-token
  ubatch on one machine has been measured.
- Why a copied op is 13-15 % slower on the R9700 placement when the feature is enabled with the default limit, i.e.
  whether a DMA copy out of the mapped coarse-grained allocation is slower than out of `cudaMallocHost` memory.
- The prompt processing warm-up: a single `llama-bench` warm-up run is not enough to reach the steady state on the
  in-place path, the steady state takes minutes to reach, and the cause is not identified. A timer-driven driver or
  OS activity (the step happens at the same elapsed time on both placements) is the shape to look for.
- Whether `hipHostMalloc(hipHostMallocMapped | hipHostMallocNonCoherent)` gives the same read bandwidth as the
  coarse-grained path. If it does, the whole `VirtualAlloc` plus after-load `hipHostRegister` machinery, and the
  finalize proc address with it, can be deleted.
