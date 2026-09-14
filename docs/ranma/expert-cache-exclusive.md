# Expert cache: exclusive mode (`--expert-cache-mode exclusive`)

## What it is

Exclusive mode keeps one resident copy of each routed expert, in a VRAM arena slot or a host arena
slot. Inclusive mode, the default, keeps a host copy of every VRAM resident, so the VRAM budget is a
second copy of those experts. Exclusive mode takes that memory back: the budget costs that much less
host memory than inclusive mode, and less than running with no cache at all, because the experts
that live in VRAM are not in RAM as well.

## Why it exists

Inclusive mode is simple because the host tensor is complete. An install copies whatever the plan
wants from host to VRAM, and if anything goes wrong the cache can switch itself off and the model
keeps running from host memory. The price is the second copy: on a machine that also has to hold the
KV cache, the prompt cache and everything else the server keeps in RAM, a large budget is paid
twice.

Exclusive mode gives that memory back. What it costs is the simplicity: with one copy per expert an
install can no longer overwrite a slot, it has to exchange the slot's contents with the incoming
expert, and there is no fallback once the buffer owns the weights. Most of this page is about how
those two things are kept safe.

## How it works

Four pieces are added to the inclusive cache.

**A host arena for the host residents** (`ggml/src/ggml-cuda/expert-host.cu`). It has the same shape
as the VRAM arena: one arena per size class and kind, packed by slot, the same zeroed tail for the
MMQ read-ahead, and a device table per layer that maps an expert id to its host slot. It is plain
private memory while the loader writes it and is registered as mapped memory once, after the load,
exactly as the host buffer type registers its buffers.

**A buffer type without pages** (`expert-controller.cu`, `expert-os.h`). The routed expert weight
context is placed on a reserved address range (`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`), so every
tensor keeps a distinct, stable address but none of them has storage behind it. The buffer's
`set_tensor` and `get_tensor` translate a byte range of a tensor into the (layer, expert, kind)
slices it covers and read or write them at their real home (`l1_arena::logical_io`). The loader
therefore writes every slice straight to the arena it will live in, VRAM or host, and never into a
staging copy. `expert_os` is the OS wrapper for the reservation and for the commit of the host
arena; its POSIX side reports "unsupported" and the option validator refuses the mode there with a
message.

The weight context is not routed experts only. With a fully offloaded model the token embedding
table is put in the same host buffer type and lands in the same context. Those tensors are handed to
a separate buffer of the type the context was going to use, so they keep the storage and the
placement they would have had with the cache off; the exclusive buffer owns that buffer and frees
it. Giving them device memory instead spends VRAM and changes the graph splits, and with them the
logits.

**Kernel reads from either arena.** The kernel lookup structure gains `host_data` and `host_slots`.
MMVQ and MMQ resolve a VRAM slot or a host slot with the same selection code. The fused up/gate
kernel switches both matrices with the layer tables. The reserved tensor address is never
dereferenced: the `MUL_MAT_ID` kernels are the only readers, and `ggml_cuda_mul_mat_id` asserts
before any other path would touch the tensor. `ggml_backend_cuda_device_supports_buft` answers for
the exclusive buffer type only on the device the cache was registered on.

**One transaction with a single-copy mover** (`expert-plan.h::plan_install`, `l1_arena::execute`).
An install becomes an exchange: a batch of promotions into spare VRAM slots followed by the
demotions that free the slots the next batch uses. Incoming and outgoing slices rotate through a
fixed number of spare slots per size class, so the transaction never needs a scratch copy of a
slice, and a readable source is preserved until its last use. Tables are published only after the
copies finish. `verify_assignment` checks after every install that no expert has two homes and no
slot two owners. Inclusive mode uses the same transaction with a host-master mover that needs no
device-to-host copy.

Two things follow from the single copy:

- **The plan is made before the loader writes.** Inclusive mode can plan after the load. Exclusive
  mode reads the profile, plans and allocates both arenas inside the buffer allocation, because a
  slice written to the wrong arena would have to be moved with no master to read it from. A side
  effect is that there is no cold round: the plan of the stored profile is resident when the server
  starts answering. Without a profile the slots are selected with `--expert-seed` (default 1).
- **A failure after the buffer exists aborts.** Everywhere else the cache disables itself and the
  model keeps running from the host tensors. Once the exclusive buffer owns the weights there is no
  such state, so a later failure is an abort with its reason instead of a kernel reading reserved
  pages. Failures before that point (budget too small for the tables and spares, arena allocation,
  slot assignment) still disable the cache, and the model loads with the ordinary host allocation.

## How to use it

```
llama-server -m MODEL -ngl 999 -c 8192 --load-mode none \
    --expert-l1-mib 20000 --expert-cache-mode exclusive \
    --expert-profile-dir DIR
```

with `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512` in the environment; the
cache needs the host-direct path, which is off by default and has no CLI flag
(`host-direct-moe.md`). Everything else is the same as inclusive mode: the budget decides the
placement, so `--n-cpu-moe` and `--cpu-moe` are refused, and `--load-mode none` is required.

| Option | Default | Effect |
|---|---|---|
| `--expert-cache-mode MODE` | `inclusive` | `inclusive` keeps a host copy of every VRAM resident; `exclusive` gives every routed expert exactly one home. |

The option validator checks before the model loads:

- The operating system must support the address reservation. Elsewhere than Windows the option is
  refused with a message that says so; it never aborts.
- An explicit `-fit on` is refused, because the fitter sizes the context against free VRAM that the
  budget has already spent on slices that cannot move back to host memory. `-fit` is on by default,
  so when it was not given the CLI turns it off itself, logs one line, and the context size has to
  be set explicitly.

The newest plan is installed at the end of every request, and with `--expert-prefill-swap` also at
the boundary between prompt processing and generation (`expert-cache-banks.md`); the install is the
exchange above. `RANMA_EXPERT_VERIFY=1` checks every VRAM and every host slot after each install
(see "Design notes" for what it compares against).

## What it costs

- VRAM: the same as inclusive mode (budget, tables, profiler banks, arena tails). The spare slots
  are inside the budget.
- Host memory: the model minus the budget, plus the host tables and the spare host slots per class.
  The second copy of inclusive mode is gone.
- Per decode token: the same as inclusive mode. The kernels read one table and pick one of two base
  addresses.
- At install: every exchanged slice is written to VRAM and read back to host, on the copy stream,
  with the device drained. An exchange moves more bytes per slice than an inclusive copy.
- At model load: noticeably longer than inclusive mode, because the plan is made from the stored
  profile before the weights are written and every slice goes to its final home.

## Limits and fallbacks

- **Windows and HIP only.** The address reservation is `VirtualAlloc`; the POSIX side of `expert-os`
  reports "unsupported" and the validator refuses the mode. The POSIX file compiles to nothing on
  Windows and was checked by inspection only; no POSIX build was made.
- **Only MMVQ and MMQ may read the weights.** A routed expert tensor in the exclusive buffer has a
  reserved address, so any other `MUL_MAT_ID` path would fault. `ggml_cuda_mul_mat_id` asserts
  before its sorted fallback instead. On RDNA4 with a quantized MoE the dispatch always reaches MMVQ
  or MMQ; a device where `ggml_cuda_should_use_mmq` says no for large batches would hit the assert.
- **No fallback after the load.** See "How it works": a failure once the buffer owns the weights is
  an abort.
- **The cache is a process singleton** bound to the device it was registered on. A second device
  does not accept the exclusive buffer type.
- Everything inclusive mode is limited by still applies: one model per process, slot 0 is profiled,
  `--load-mode none` is required.

## How to verify it

- `test-expert-exclusive` replays assignments and installs against the two-tier layout on the CPU:
  unique physical ownership, the exchange through the spare slots, and deliberately invalid
  assignments that `verify_assignment` must reject.
- `test-expert-config` covers the option validator, including the refusal on a platform without the
  reservation and the `-fit` rule.
- `RANMA_EXPERT_VERIFY=1` on a real run compares every VRAM and host slot with the digest recorded
  while the loader wrote it, after every install.

## Design notes

- **Non-expert tensors of the routed context are kept, not refused.** Refusing them would mean
  exclusive mode never runs on a model whose token embedding shares the context. The delegate buffer
  has the context's own buffer type; because that buffer is not in the model's buffer list, the
  controller runs the backend's post-load step for host buffers on it itself.
- **The buffer type's `alloc_buffer` goes to device memory.** An allocation on this buffer type that
  is not one of the tensors placed above needs real storage: a LoRA adapter takes the buffer type of
  its base tensor (`src/llama-adapter.cpp`), and a base tensor in the exclusive buffer would
  otherwise hand it a null allocator.
- **`RANMA_EXPERT_VERIFY` compares against a digest of the loader's bytes.** Inclusive mode verifies
  an arena slice against the host master. Exclusive mode has no master and the backend does not know
  the model's file handles, so a 128-bit digest is recorded per (layer, kind, expert) while the
  loader writes, and every slot is compared against it after every install. A write that does not
  cover whole slices leaves experts without a digest and the log counts them.
- **No dark step in an exclusive install.** The inclusive mover publishes -1 for a slot before it
  overwrites it, so the slot is never visible as resident while it changes. In exclusive mode -1
  would mean "read the tensor", which has no bytes, so the tables stay valid throughout and the
  ordering on the copy stream is what protects a slot until it is reused.
- **The fused `MUL_MAT_ID` path** (models with a quantized expert down bias or scale) reads the
  arenas too.
