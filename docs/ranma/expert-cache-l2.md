# Expert cache: finite host tier and file backing (`--expert-l2-mib`)

## What it is

`--expert-l2-mib N` bounds the host memory the expert cache may hold. L1 is VRAM, L2 is host memory,
and the GGUF file on the SSD is the backing store. An expert that fits in neither resident tier stays
in the file and is read on demand into a fixed ring of host slots before its layer uses it; MMVQ and
MMQ read that ring through mapped memory over PCIe, exactly as they read a host resident.

Both `--expert-cache-mode inclusive` and `exclusive` support a finite tier. Inclusive means the host
set contains the VRAM set; exclusive means each resident expert has one home, in VRAM or in host
memory. Neither mode changes or deletes the GGUF. L2 is a cache over that file, and the unlimited
tier is the degenerate case in which the cache holds everything.

`--expert-l2-mib -1`, the default, is the unlimited tier; `0` is refused, because an empty host tier
leaves the non-VRAM experts nowhere to live but the file. `--expert-l1-mib 0` is valid with a finite
tier: no VRAM payload, every demand served from host memory or the file.

The finite tier needs Windows and HIP: its unbuffered read queue and its address reservation have no
implementation elsewhere, and the validator refuses the option there with a message. The unlimited
inclusive path stays OS independent.

## Why it exists

The experts left outside VRAM can exceed the system memory of the machine. A finite host budget
trades SSD reads and GPU waiting for lower resident memory, which is what lets the reference model
(73450 MiB of routed experts) run on a 64 GB or 32 GB machine at all. Weights are immutable, so
eviction needs no writeback.

## Demand path

1. After routing, one GPU block clears the layer's demand bitmap, synchronizes, and atomically sets
   one bit for every expert selected by every row. A block barrier and a system fence precede the
   generation's system-scope release store. Duplicate selections cannot lose bits.
2. The publish kernel also reads a CPU-published serve bitmap that marks the experts only the
   worker can place: file residents and ring occupants. When no routed id of the layer is marked,
   the kernel stores the readiness itself and the wait kernel passes on its first load, so a fully
   resident layer never reaches the CPU.
3. A GPU wait kernel waits for the CPU's matching readiness generation. It is a barrier on the
   stream, and there is no GPU timeout.
4. The CPU worker scans the bitmap, resolves VRAM and host hits, and reads only the remaining
   demanded slices into ring slots with unbuffered I/O (queue depth 4 by default). It publishes
   `slot_base + shift` addresses, then the readiness.
5. After the last down-weight read of the layer, a done kernel releases the layer's pins. The stream
   orders this before the next layer's demand.

One row and 512 rows use this same path. There is no whole-layer mode and no next-layer prefetch: a
large prompt can select every expert of a layer, but only its actual bitmap decides which slices are
read, and the next layer's selection is not known until its router has run.

The ring uses round-robin replacement with pinning, not LRU. A ring larger than the live set of one
layer can retain recently read slices; the worker invalidates an evicted address before overwriting
its slot, and a ring occupant stays marked in the serve bitmap so that the worker extends its lease
before a later layer can evict a slot the current layer still reads.

The mailbox exists from the moment the tier allocates, not from the first plan that leaves something
in the file: captured graphs hold its two kernels, so the alternative would be a plan-dependent
graph. With an all-clear serve bitmap the two kernels answer themselves and cost two small launches
per routed layer.

### Addresses and padding

For each kind (gate, up, down), the ring pitch includes the largest class slice, the maximum
4095-byte file-sector shift, the MMQ read-ahead padding, and rounding to 4096 bytes.
`shift = file_offset % 4096`; the GGUF's 32-byte alignment keeps the payload vector aligned. Reads
stay at that shifted address; there is no compaction.

Every byte from `shift + slice_bytes` to the end of the pitch is cleared after every read. MMQ reads
whole K tiles beyond the final row, so stale quantized bytes there produce NaN even when multiplied
by zero (`expert-cache-prefill.md`). Lent resident slots use shift zero and also have zeroed tails.
A GPU test (`test-expert-l2-gpu`) exercises a shifted ring through MMQ.

## Ring size and budget

For a layer with E experts, U selected experts per row, and R rows:

```
maximum distinct demand = min(E, U * R)
floor bytes             = maximum distinct demand * sum(maximum pitch of each kind)
```

The prompt bound uses `n_ubatch`; the decode bound uses `n_parallel * (draft_max + 1)`. Automatic
prompt ring storage is the prompt floor. Automatic decode ring storage is its floor plus a reserve
of `min(512 MiB, prompt ring - decode floor)`. With `--expert-prefill-swap` the ring tail beyond
the active decode storage holds additional host residents while tokens are generated (lent slots),
and the reserve is never lent. Without the swap there is one ring, sized for the prompt. An explicit
ring below its floor is raised with one log line. A larger ring is a performance knob; no optimal
default is claimed (see "Known limits").

The host budget pays for the ring, the mailbox, demand and address tables, the resident class
storage, spare slots, aligned read staging and padding. Geometry validation computes exact pitches
and capacities before allocation and refuses a budget below the minimum with both numbers; it never
falls back to an unlimited allocation.

Finite inclusive mode additionally needs host space for every possible VRAM resident:

```
P_l1        = sum(VRAM class capacity * class expert payload bytes)
L2 minimum >= P_l1 + ring + host metadata + padding + spare slots
```

`P_l1` is smaller than the L1 option value, which also pays for slot tables, histograms, padding and
spares. Inclusive 20000 MiB / 8192 MiB, for instance, is refused.

These are cache allocation budgets, not a cap on the process. Model metadata, non-expert weights,
KV, backend workspaces, the driver, profiles and diagnostic buffers are additional.

**Installed memory.** When the plan resolves its host requirement (the requested budget or, for the
unlimited tier, the byte count of the host-resident set), it is compared with the physical memory
installed in the machine. A requirement above that line can never be met, so the model load stops
with one error line that names both numbers in GiB and the option to lower, before the first expert
byte is copied. Available memory is not consulted: below the installed total, what else runs on the
machine is the user's business.

## Placement and the install transaction

The loader redirects routed weights to the cache's own storage for a finite tier in both modes. It
skips the payloads the plan leaves in the file and records their file ranges. The three-tier cut is
the same greedy the VRAM tier uses, one level down: every expert the VRAM plan did not take is a
candidate for host memory, ordered by score x bytes, until the host budget is full.

`expert-plan.h::plan_install` produces one transaction over four locations: VRAM slot, host slot,
lent ring slot and file. The unlimited case is the same function without file and lent locations.
One mover, `l1_arena::execute`, moves every slice of a transaction whatever tier its two ends are
in; for a source in the file a small reader gives it the batched read and the ring address. Before
an install the device drains and the worker stops; ordered copies preserve every source until its
last use; tables are published only after the copies finish. Growing the ring retires lent slots
through the same transaction before reusing their bytes.

`verify_assignment` and the CPU tests check unique physical ownership, valid demand resolution, no
duplicate VRAM/host home in exclusive mode, a host copy for every inclusive VRAM resident, ring
growth and shrinkage, and deliberately invalid assignments.

## Options

Use `--load-mode none`, `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`. Do not
combine cache placement with `--cpu-moe` or `--n-cpu-moe`.

| Option | Default | Meaning |
|---|---|---|
| `--expert-l2-mib N` | -1 | Host memory budget in MiB; -1 is unlimited and 0 is refused. |
| `--expert-l2-staging-mib N` | 0 | Ring size in MiB; 0 is automatic. Sets both rings unless one of the two below overrides it. |
| `--expert-l2-prefill-ring-mib N` | the staging value | Ring while a prompt is processed; 0 is automatic. |
| `--expert-l2-decode-ring-mib N` | the staging value | Ring while tokens are generated; 0 is automatic. Only differs from the prompt ring with `--expert-prefill-swap`. |
| `--expert-l2-worker-cpu N` | -1 | Logical CPU of the worker thread; -1 is the last active logical CPU. |
| `--expert-l1-mib N` | 0 | VRAM budget; zero is valid with a finite tier. |
| `--expert-cache-mode MODE` | inclusive | Relation of the VRAM set to the host set. |

The validator refuses a finite tier with a multimodal projector and with speculative decoding: those
combinations have no correctness check yet, not a structural problem. More than one server slot is
accepted by the tier; the prompt swap keeps its single-slot rule.

Three debug environment variables, all read in one controller function: `RANMA_EXPERT_L2_QD` (read
queue depth, default 4), `RANMA_EXPERT_L2_WAIT` (CPU I/O queue deadline in ms, default 30000) and
`RANMA_EXPERT_L2_VERIFY` (independent buffered payload and ownership checks). `RANMA_EXPERT_TRACE=8`
adds `expert_metrics` JSON lines with cumulative reads, ring hits, SSD payload bytes, CPU service
time, GPU waiting per phase and per-ubatch samples, plus the round lines and install lines described
in `benchmark.md`.

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10, the model file
on the local SSD. Model: Qwen3.8-Flash-Next UD-Q4_K_XL (73450 MiB of routed experts). `llama-bench`
PP512 / TG128, exclusive mode, host-direct on, `--load-mode none`.

**This commit alone.** 3072 MiB budget, `-r 1`, depths `8192,0,8192` with the first pair discarded,
Warm from a Cold seed of the same budget. Byte split = selected bytes served from VRAM / host / file:

| host tier | PP512 @ 0 | TG128 @ 0 | PP512 @ 8192 | TG128 @ 8192 | byte split |
|---|---:|---:|---:|---:|---|
| unlimited | 571.07 | 29.89 | 576.54 | 28.88 | - |
| 40960 MiB | 402.77 | 29.61 | 435.29 | 28.77 | 71.0 / 28.4 / 0.6 % |
| 24576 MiB | 268.87 | 28.50 | 287.23 | 27.61 | 71.0 / 27.6 / 1.4 % |

**Where the prompt-processing cost of a finite tier goes.** On this binary, R9700 row, 3072 MiB
budget, one 512-token prompt ubatch at depth 0 takes 883.5 ms with the unlimited tier and 1271.2 ms
with a 40960 MiB tier. The difference was decomposed with per-op timing:

| term | per prompt ubatch |
|---|---:|
| synchronous file service: the wait kernel blocks the stream while the worker reads the layer's file-resident experts (29 of 48 layers demanded at least one; 2.95 GB per ubatch at 10.3 GB/s) | 315 ms |
| demand-publish kernel, 48 launches of one block | 30 ms |
| other tier work on the GPU (serve-bitmap decode, address indirection) | 43 ms |
| rest of `llama_decode`, faster with the finite tier because it leaves more free RAM for the per-layer embedding faults | -58 ms |

A prompt ubatch touches 27.5 GB of distinct expert bytes; with 44 GiB resident (VRAM plus host) 2.95
GB of them, 10.7 %, are file residents and are read synchronously, layer by layer, while the GPU
waits. The ring does not help: it holds about 1.7 GB, less than the file set of one ubatch, and a
prompt reads the file set in layer order once per ubatch, so by the time a slice is wanted again it
has been rotated out (9 ring hits in 64632 reads). Matrix multiplication itself is unchanged between
the two rows.

**Correctness.** Eight prompts x 48 greedy tokens with top-3 logprobs, exclusive 3072 MiB, 8192 MiB
host tier, swap on, so every layer reads from all three tiers: identical to the cache off on the same
binary, and `RANMA_EXPERT_VERIFY` found no slice that differed from its source (`expert-cache.md`).

**Lazy file mappings.** A read-only mapping of a model shard slows unbuffered reads of the same file
on this Windows host: replaying one 9.95 GB install read sequence through the same queue took 1062
ms without GGUF mappings and 2595 ms with read-only mappings of all four shards; mapping 4 KiB per
file was enough to cause it. The loader therefore maps only the shards that hold lazy tensors when
ordinary mmap loading is off (a separate commit of this series). A shard that holds both the lazy
per-layer embedding table and routed experts still pays that cost.

## What it costs

- Host memory: the budget, of which the ring, tables and spares take their share before the
  residents.
- Per routed layer, always: two small kernel launches (publish and wait) that answer themselves
  when nothing is demanded from the file, about 0.6 ms per layer for the publish on this device.
- Per prompt ubatch with file residents: the synchronous file service above, proportional to the
  bytes the ubatch demands from the file.
- One worker thread, spinning on the mailbox, pinned to one logical CPU.
- At install: file reads for every slice promoted from the file, in addition to the copies.

## Known limits

- **Windows and HIP only**, for the unbuffered read queue and the address reservation.
- **Prompt processing with file residents is synchronous.** The wait kernel blocks the stream while
  the worker reads; nothing overlaps that read with matrix multiplication. Chunking the current
  layer's demand and overlapping its reads is a possible later change, not done here.
- **The ring is sized for the worst possible distinct demand of one ubatch**, so a large ubatch
  reserves a large ring, and yet the ring is smaller than the file set of one prompt ubatch on the
  reference model, so its reuse is close to zero. Sizing the prompt ring from the plan (the bytes a
  prompt ubatch will read from the file) and replacing round-robin by a hit-refreshing policy are
  the next steps; both change nothing until the ring is at least as large as that set.
- **The prompt swap's boundary install reads from the SSD** with a finite tier: 2.5 to 3.3 s per
  request on this machine at 40960 MiB (`expert-cache-banks.md`).
- **No correctness check yet** for a finite tier with a multimodal projector or speculative
  decoding; the validator refuses both.
- **`llama-server` of this tree can hang at shutdown** after every request has been answered, with
  and without the expert cache. The cause is open and is not in the cache.

## Revision

Every number above comes from the binary built from the commits that add this feature on ranma
`ccd8fd1e9` (upstream llama.cpp `093a2f86c`, plus the fork's earlier commits), against the published
snapshot `ranma_20260914` (`771fa0bda`) as the base row. The exact commits are kept on the dated
snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that branch is never rebased.
