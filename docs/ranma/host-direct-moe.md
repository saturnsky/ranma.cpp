# Host-direct MoE weights (HIP)

## What it is

A discrete GPU normally cannot use a weight that sits in host memory: the scheduler either copies the weight to VRAM
for every op, or it leaves the op on the CPU backend. This feature adds a third option for one narrow case. A backend
can declare that a given op may read a given `src` in place from the device's own host buffer type, and the scheduler
then keeps the op on that backend and creates no copy.

The HIP backend declares this for `MUL_MAT_ID` with a quantized `src0`, up to a token count limit. The expert weights
stay in mapped pinned host memory and the `MUL_MAT_ID` kernels (MMVQ, or MMQ above the MMVQ batch limit) read them
over PCIe, one expert row block at a time, while every other tensor of the op stays in VRAM.

The feature is off by default and is turned on with an environment switch, `GGML_CUDA_HOST_DIRECT=1`. There is no CLI
flag.

## Why it exists

A large MoE model does not fit in the VRAM of one card, so its routed experts live in system RAM (`-ncmoe`, or an
`-ot` override that sends `ffn_*_exps` to the CPU). Each decode step then touches only the few experts that the
router selected. Computing that on the CPU backend is bound by CPU memory bandwidth and by the round trip through the
CPU backend; reading the same bytes straight from host memory into a GPU kernel is bound by PCIe bandwidth only, and
the GPU needs far fewer cycles per byte.

Prompt processing was expected to be the opposite case: a 512-token batch reads every selected expert, so a single
copy to VRAM looked cheaper than reading host memory from the kernel. The token count limit separates the two
regimes. In practice, once MMQ reads the weight in place, prompt processing is faster that way too on the machine
this was measured on, which is why the recommended setting raises the limit to the ubatch size while the default
stays conservative.

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

## How the backend declares it

Through the proc-address extension point, the same way as `ggml_backend_register_host_buffer` and
`ggml_backend_get_features`: a backend's `get_proc_address` answers the name `"ggml_backend_host_direct_op"` with a
`ggml_backend_host_direct_op_t`. `ggml_backend_sched_new` resolves it once per backend and caches it;
`ggml_backend_sched_buft_supported` calls the cached pointer, or skips the whole rule when the backend returned NULL.
`ggml_backend_dev_host_direct_op` is a convenience wrapper that resolves the address on every call, for the few
callers outside the scheduler.

It is deliberately *not* a new member of `ggml_backend_device_i`. `ggml-base` reads the iface struct that a backend
shared library defines, so adding a member in the middle of it silently changes the backend ABI while
`GGML_BACKEND_API_VERSION` stays 2: an unpatched backend library loaded by a patched core passes the version check
and then has `event_new` called where the core expects `host_direct_op`. Swapping a single backend library between
builds is routine here, so the proc address is the safe shape. It also needs no version bump and no edit to any other
backend.

## The admission predicate

A weight is admitted only when all of the following hold:

- `GGML_CUDA_HOST_DIRECT` is set,
- the op is `MUL_MAT_ID` and the tensor is its `src0`,
- the tensor type is quantized,
- the tensor's buffer is the HIP backend's host buffer type and is mapped (see the coarse path below),
- the token count `op->ne[2]` is at most the limit, and
- above the per-type MMVQ `MUL_MAT_ID` limit, `ggml_cuda_should_use_mmq` holds for the op.

The last condition mirrors the dispatch in `ggml_cuda_mul_mat_id`: MMVQ and MMQ are the only kernel families that
resolve a mapped host address, so an op that would reach MMF or the sorted fallback must not be admitted. `MMF`,
`MMVF` and the cuBLAS leaf assert that `src0` is device readable instead of launching on a host address.

## Row padding

Because MMQ loads whole K tiles (`MMQ_ITER_K` elements) of `src0`, a quantized matrix whose row length is not a
multiple of the tile is read past its last row. A device buffer covers that read with zeroed row padding
(`MATRIX_ROW_PADDING`, see `[TAG_ALLOC_SIZE_EXPAND]` in `ggml-cuda.cu`); with host-direct enabled the host buffer
type allocates and clears the same padding in `get_alloc_size` and `init_tensor`, so a weight looks the same to the
kernel wherever it lives. Without the padding the last row of the last expert of such a tensor reads whatever follows
it in the buffer, and a non-finite scale there turns into NaN activations for the tokens routed to that expert.

## Environment switches

| Variable | Default | Meaning |
|---|---|---|
| `GGML_CUDA_HOST_DIRECT` | unset | Unset or `0` disables the feature. Any other value enables it. No effect outside HIP. |
| `GGML_CUDA_HOST_DIRECT_MAX_BATCH` | `0` | Token count limit (`op->ne[2]`) up to which a weight may be read in place. `0` means the per-type MMVQ `MUL_MAT_ID` limit, `get_mmvq_mmid_max_batch(type, cc)`. Set it to at least the ubatch size (for example `512`) to read in place during prompt processing as well. |
| `GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB` | `1024` | Host buffers of at least this size take the coarse-grained path: `VirtualAlloc`, then `hipHostRegister(Mapped \| CoarseGrained)` after the model is loaded. `0` disables it, so every host buffer is a plain `hipHostMalloc(Mapped)`. Windows only. |

All three are read once, when the backend registers its devices.

When the feature is enabled, the host buffer type allocates with `hipHostMalloc(Mapped)` instead of
`cudaMallocHost`, because a kernel can only dereference a host pointer that was mapped into the device address space.
An allocation that fails falls back to a plain CPU buffer, with a warning, and host-direct then does not apply to it.

A buffer on the coarse path is *not* readable by a kernel until it has been registered. `llama_model::load_tensors`
calls the backend's `ggml_backend_finalize_host_buffer` proc address once the weights are written, and until that
succeeds `host_direct_op` returns false, so the scheduler keeps using the copy or CPU path. A registration that
cannot be completed - the register call, the device pointer lookup or the synchronization that publishes the CPU's
writes - is undone on the spot and the buffer stays in its plain state.

## Recommended setting

```
GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512
```

`MAX_BATCH` should be at least the ubatch size (`-ub`, default 512), so that prompt processing also reads the experts
in place. The defaults are deliberately left as they are: the feature stays off unless asked for, and the limit keeps
its conservative MMVQ meaning, because the measurements behind the recommendation come from one machine and one
ubatch size.

## Limits and fallbacks

- HIP only. The device callback returns false for CUDA, so the code still builds and the feature does nothing.
- The coarse-grained path is Windows only. On other platforms every host buffer is a plain `hipHostMalloc(Mapped)`,
  whose device alias happens to be the host address, so `GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB` has no effect there.
- Quantized `src0` only. An F16 or F32 expert weight is never admitted.
- Only `MUL_MAT_ID`. A dense `MUL_MAT` weight is never admitted.
- With a limit in effect, token counts between the limit and `GGML_OP_OFFLOAD_MIN_BATCH` (default 32) run the MoE on
  the CPU backend, because the weight is neither admitted here nor large enough for the offload rule to copy it.
- `MUL_MAT_ID` weights are admitted per op, not per graph. This is equivalent to a per-graph decision because all
  `MUL_MAT_ID` nodes of one graph share `ne[2]`. In particular the reserve graph is built at the full batch size, so
  with a limit it takes the copy path and the compute buffer is sized for it.
- Prompt processing with the default limit takes the copy path out of the mapped, coarse-grained allocation rather
  than out of `cudaMallocHost` memory, and that copy was measured to be slower than the unpatched one on one of the
  two expert placements tried. The cause is not established; raising the limit to the ubatch size avoids it.

## How to verify it

- **`test-backend-ops --host-weights`** puts every quantized weight of the case in the device's host buffer type.
  The test computes graphs directly and never builds a `ggml_backend_sched`, so it asks
  `ggml_backend_dev_host_direct_op` per op and reports a case the device would not admit as
  `not supported [<backend> host-direct]` instead of handing a host pointer to a kernel that cannot take one. A run
  therefore reports OK and skip counts, and the skipped list is worth reading: a type or batch size that is skipped
  unexpectedly means the admission rule and the kernel dispatch have drifted apart. The `MUL_MAT_ID` cases cover 1 to
  8 rows and one real expert geometry (512 experts, 10 used, k = 800), which fails without the row padding.
- **Force the non-identity alias.** A `hipHostMalloc(Mapped)` buffer aliases to its own host address, so a test that
  only uses those cannot expose a bug in the register path or in the offset arithmetic. Lower the coarse threshold so
  that test-sized buffers take the `VirtualAlloc` plus `hipHostRegister` path:

  ```
  GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_COARSE_MIN_MIB=1 test-backend-ops -b ROCm0 -o MUL_MAT_ID --host-weights
  ```

- **`--host-weights` without `GGML_CUDA_HOST_DIRECT`.** The host buffer type then allocates with `cudaMallocHost`,
  which on HIP is `hipHostMalloc` with default flags, and whether `hipHostGetDevicePointer` returns a usable alias
  for such an allocation is not established here. The admission hook makes the question harmless, because it returns
  false for every op and the whole sweep is skipped, but the result is a run that reports zero tested cases for a
  non-obvious reason, so the flag also prints a warning when the variable is unset.
- **Test the model's own row lengths.** A `k` that is a multiple of 256 never reads past a row, so a sweep made of
  such shapes cannot show a padding problem. The server symptom of missing padding is NaN logits from the second
  request on, even when the second request repeats the first one; `llama-perplexity -c 128 -b 128` shows it on the
  first chunk and is the quicker reproduction.
- **The test does not cover the scheduler.** Only a model run reaches `ggml_backend_sched_buft_supported` and the
  splits around it.
