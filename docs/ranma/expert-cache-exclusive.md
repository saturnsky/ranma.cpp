# Expert cache: exclusive mode (`--expert-cache-mode exclusive`)

## What it is

Exclusive mode keeps one resident copy of each routed expert, in a VRAM arena slot or a host arena
slot. Inclusive mode, the default, keeps a host copy of every VRAM resident, so the VRAM budget is a
second copy of those experts. Exclusive mode takes that memory back: a 20 GiB budget costs 20 GiB
of host memory less than inclusive mode, and less than running with no cache at all, because the
experts that live in VRAM are not in RAM either.

With unlimited host memory every expert has one of the two homes. With a finite `--expert-l2-mib`
the experts outside both resident sets stay in the GGUF file and are read on demand
(`expert-cache-l2.md`). The file is never modified in either mode.

## Why it exists

Inclusive mode is simple because the host tensor is complete. An install copies whatever the plan
wants from host to VRAM, and if anything goes wrong the cache can switch itself off and the model
keeps running from host memory. The price is the second copy. On the reference model a 20000 MiB
budget takes process private memory from 81 GiB to 98 GiB, which on a 128 GiB machine leaves little
for the KV cache checkpoints, the prompt cache and everything else the server keeps in RAM.

Exclusive mode gives that memory back. What it costs is the simplicity: with one copy per expert an
install can no longer overwrite a slot, it has to exchange the slot's contents with the incoming
expert, and there is no fallback once the buffer owns the weights. Most of this page is about how
those two things are kept safe.

## How it works

Four pieces are added to the inclusive cache.

**A host arena for the host residents** (`ggml/src/ggml-cuda/expert-host.cu`). It has the same
shape as the VRAM arena: one arena per size class and kind, packed by slot, the same zeroed tail for
the MMQ read-ahead, and a device table per layer that maps an expert id to its host slot. It is plain
private memory while the loader writes it and is registered as mapped memory once, after the load,
exactly as the host buffer type registers its buffers.

**A buffer type without pages** (`expert-controller.cu`, `expert-os.h`). The routed expert weight
context is placed on a reserved address range (`VirtualAlloc(MEM_RESERVE, PAGE_NOACCESS)`), so every
tensor keeps a distinct, stable address but none of them has storage behind it. The buffer's
`set_tensor` and `get_tensor` translate a byte range of a tensor into the (layer, expert, kind)
slices it covers and read or write them at their real home (`l1_arena::logical_io`). The loader
therefore writes every slice straight to the arena it will live in, VRAM or host, and never into a
staging copy. `expert_os` is the OS wrapper for the reservation and for the commit of the host arena;
its POSIX side reports "unsupported" and the option validator refuses the mode there with a message.

The weight context is not routed experts only. With a fully offloaded model the token embedding
(644 MiB here) is put in the same host buffer type and lands in the same context. Those tensors are
handed to a separate buffer of the type the context was going to use, so they keep the storage and
the placement they would have had with the cache off; the exclusive buffer owns that buffer and
frees it. Giving them device memory instead was tried first: it spent 644 MiB of VRAM, changed the
graph from 2 splits to 3 and with it the logits.

**Kernel reads from either arena.** The kernel lookup structure gains `host_data` and `host_slots`.
MMVQ and MMQ resolve a VRAM slot, a host slot and (with a finite host tier) a ring address with the
same selection code. The fused up/gate kernel switches both matrices with the layer tables. The
reserved tensor address is never dereferenced: the MUL_MAT_ID kernels are the only readers, and
`ggml_cuda_mul_mat_id` asserts before any other path would touch the tensor.

**One transaction with a single-copy mover** (`expert-plan.h::plan_install`, `l1_arena::execute`).
An install becomes an exchange: a batch of promotions into spare VRAM slots followed by the demotions
that free the slots the next batch uses. Incoming and outgoing slices rotate through a fixed number
of spare slots per size class, so the transaction never needs a scratch copy of a slice, and a
readable source is preserved until its last use. Tables are published only after the copies finish.
`verify_assignment` checks after every install that no expert has two homes and no slot two owners.
Inclusive mode uses the same transaction with a host-master mover that needs no device-to-host copy.

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

with `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`, the recommended host-direct
profile. Everything else is the same as inclusive mode: the budget decides the placement, so
`--n-cpu-moe` and `--cpu-moe` are refused, `--load-mode none` is required, and the profile directory
holds the `decode` and `prefill` banks.

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

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. Model:
Qwen3.8-Flash-Next UD-Q4_K_XL (73450 MiB of routed expert weights, 24576 (layer, expert) slices).
Protocol of `benchmark.md`: `llama-bench` PP512 / TG128, one repetition, depths 0, 4096, 8192, 32768
and 65536 after a discarded 65536 pass, unlimited host memory, Warm from one Cold seed per device
row.

**Host memory.** Peak process private bytes of each run:

| device row | budget | no cache (base) | inclusive Warm | exclusive Warm |
|---|---:|---:|---:|---:|
| R9700 | 20000 MiB | 80.74 GiB | 97.59 GiB | 78.15 GiB |
| RX 9070 XT emulation | 3072 MiB | 80.74 GiB | 80.65 GiB | 77.82 GiB |

The difference between the modes is the VRAM arena: 19.4 GiB at the 20000 MiB budget, 2.8 GiB at
3072 MiB. Exclusive mode with a 20 GiB budget uses 2.6 GiB less host memory than running with no
cache at all.

**Throughput.** PP512 / TG128 in t/s at depths 0 / 4096 / 8192 / 32768 / 65536:

| device row | budget | mode | PP512 | TG128 |
|---|---:|---|---|---|
| R9700 | 20000 MiB | inclusive | 864.89, 863.78, 842.60, 733.35, 609.81 | 38.00, 37.71, 37.02, 32.04, 26.89 |
| R9700 | 20000 MiB | exclusive | 937.43, 927.21, 898.39, 771.49, 631.95 | 38.90, 38.52, 37.62, 32.01, 26.89 |
| RX 9070 XT emulation | 3072 MiB | inclusive | 662.87, 638.14, 611.07, 558.11, 481.22 | 29.89, 30.80, 30.45, 26.66, 22.95 |
| RX 9070 XT emulation | 3072 MiB | exclusive | 615.13, 602.39, 577.88, 529.72, 459.98 | 29.77, 30.73, 30.42, 26.59, 22.92 |

Decode is the same in both modes within the run-to-run drift of this protocol (up to 0.9 % on TG128
between two base runs on the R9700 row, 2.4 % on the other row), as it should be: the same bytes
are read by the same kernels from the same kind of memory. Prompt processing differs by up to 8 % in
either direction between the two rows; the base-to-base drift of prompt throughput is 4 to 7 % under
this protocol, so no direction is claimed.

**Install and exchange.** The load-time install has no timing of its own; it is the loader's writes.
At a request boundary an exchange moves every byte twice (the promoted slice up, the demoted slice
down), so it costs about 1.4x an inclusive copy per slice: in the same runs the control time of a
TG128 test at the 20000 MiB budget, which is one commit and one delta install, is 33 to 45 ms
inclusive and 54 to 74 ms exclusive on the R9700 row (the `ctl ms` column of `benchmark.md`).

**Correctness.** Eight prompts x 48 greedy tokens with top-3 logprobs, exclusive 3072 MiB with a
finite 8192 MiB host tier and the prompt swap on, against the cache off on the same binary: every
token and every logprob identical (`expert-cache.md`). Exclusive mode builds the same graph as a run
without the cache (2 splits, the same `ROCm0` model buffer and the same `ROCm_Host` compute buffer),
because the non-expert tensors of the routed context keep their buffer type.

**Model load.** About 1.6x the load time of inclusive mode or the cache off on a warm page cache
(59 to 67 s against 37 to 41 s here). The host arena is committed and touched, and every slice goes
through the buffer's `set_tensor` instead of one read into a host buffer.

## What it costs

- VRAM: the same as inclusive mode (budget, tables, profiler banks, arena tails). The spare slots
  are inside the budget.
- Host memory: the model minus the budget, plus the host tables and the spare host slots per class.
  The second copy of inclusive mode is gone.
- Per decode token: the same as inclusive mode. The kernels read one table and pick one of two base
  addresses.
- At install: every exchanged slice is written to VRAM and read back to host, on the copy stream,
  with the device drained.
- At model load: about 1.6x the load time, and the plan is made from the stored profile before the
  weights are written.

## Limitations

- **Windows and HIP only.** The address reservation is `VirtualAlloc`; the POSIX side of
  `expert-os` reports "unsupported" and the validator refuses the mode. The POSIX file compiles to
  nothing on Windows and was checked by inspection only; no POSIX build was made.
- **Only MMVQ and MMQ may read the weights.** A routed expert tensor in the exclusive buffer has a
  reserved address, so any other `MUL_MAT_ID` path would fault. `ggml_cuda_mul_mat_id` asserts
  before its sorted fallback instead. On RDNA4 with a quantized MoE the dispatch always reaches MMVQ
  or MMQ, which is why the assert has never fired; a device where `ggml_cuda_should_use_mmq` says no
  for large batches would hit it.
- **No fallback after the load.** See "How it works": a failure once the buffer owns the weights is
  an abort.
- **Model load is slower**, about 1.6x here, and an exchange is about 1.4x an inclusive copy per
  slice.
- Everything inclusive mode is limited by still applies: one model per process, slot 0 is profiled,
  `--load-mode none` is required.

## Design notes

- **`host_addresses` is null without a finite host tier.** A host slot is addressed by its slot
  index. The finite tier supplies an address table for streamed and lent ring slices; the kernels
  take a nonzero address before the host-slot path.
- **Non-expert tensors of the routed context are kept, not refused.** Refusing them would mean
  exclusive mode never runs on the reference model, whose token embedding shares the context. The
  delegate buffer has the context's own buffer type; because that buffer is not in the model's
  buffer list, the controller runs the backend's post-load step for host buffers on it itself.
- **The buffer type's `alloc_buffer` goes to device memory.** An allocation on this buffer type that
  is not one of the tensors placed above needs real storage: a LoRA adapter takes the buffer type of
  its base tensor (`src/llama-adapter.cpp`), and a base tensor in the exclusive buffer would
  otherwise hand it a null allocator.
- **`RANMA_EXPERT_VERIFY` compares against a digest of the loader's bytes.** Inclusive mode verifies
  an arena slice against the host master. Exclusive mode has no master and the backend does not
  know the model's file handles, so a 128-bit digest is recorded per (layer, kind, expert) while the
  loader writes, and every slot is compared against it after every install. A write that does not
  cover whole slices leaves experts without a digest and the log counts them. With a finite host
  tier a slice promoted from the file is compared with the file payload before a digest is retained
  for it.
- **No dark step in an exclusive install.** The inclusive mover publishes -1 for a slot before it
  overwrites it, so the slot is never visible as resident while it changes. In exclusive mode -1
  would mean "read the tensor", which has no bytes, so the tables stay valid throughout and the
  ordering on the copy stream is what protects a slot until it is reused.
- **The fused `MUL_MAT_ID` path** (models with a quantized expert down bias or scale) reads the
  arenas too. It notifies the host tier after the fused kernel has finished reading the down weights;
  the controller checks kind 2, so gate and up fusions do not release the layer's slots.

## Revision

Every number above comes from the binary built from the commits that add this feature on ranma
`ccd8fd1e9` (upstream llama.cpp `093a2f86c`, plus the fork's earlier commits), against the published
snapshot `ranma_20260914` (`771fa0bda`) as the base row. The exact commits are kept on the dated
snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that branch is never rebased.
