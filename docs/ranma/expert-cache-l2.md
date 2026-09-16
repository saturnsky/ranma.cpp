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

The experts left outside VRAM can exceed the system memory of the machine, and then the model does
not load at all. A finite host budget trades SSD reads and GPU waiting for lower resident memory,
which is what lets a model whose routed experts are larger than RAM run. Weights are immutable, so
eviction needs no writeback.

## Demand path

1. After routing, one GPU block clears the layer's demand bitmap, synchronizes, and atomically sets
   one bit for every expert selected by every row. A block barrier and a system fence precede the
   generation's system-scope release store. Duplicate selections cannot lose bits.
2. The publish kernel also reads a CPU-published serve bitmap that marks the experts only the worker
   can place: file residents and ring occupants. When no routed id of the layer is marked, the
   kernel stores the readiness itself and the wait kernel passes on its first load, so a fully
   resident layer never reaches the CPU.
3. A GPU wait kernel waits for the CPU's matching readiness generation. It is a barrier on the
   stream, and there is no GPU timeout.
4. The CPU worker scans the bitmap, resolves VRAM and host hits, and reads only the remaining
   demanded slices into ring slots with unbuffered I/O. It publishes `slot_base + shift` addresses,
   then the readiness.
5. After the last down-weight read of the layer, a done kernel releases the layer's pins. The stream
   orders this before the next layer's demand.

One row and a full ubatch use this same path. There is no whole-layer mode and no next-layer
prefetch: a large prompt can select every expert of a layer, but only its actual bitmap decides which
slices are read, and the next layer's selection is not known until its router has run.

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

## Ring size and budget

For a layer with E experts, U selected experts per row, and R rows:

```
maximum distinct demand = min(E, U * R)
floor bytes             = maximum distinct demand * sum(maximum pitch of each kind)
```

The prompt bound uses `n_ubatch`; the decode bound uses `n_parallel * (draft_max + 1)`. Automatic
prompt ring storage is the prompt floor. Automatic decode ring storage is its floor plus a reserve
of `min(512 MiB, prompt ring - decode floor)`. With `--expert-prefill-swap` the ring tail beyond the
active decode storage holds additional host residents while tokens are generated (lent slots), and
the reserve is never lent. Without the swap there is one ring, sized for the prompt. An explicit ring
below its floor is raised with one log line. A larger ring is a performance knob; no optimal default
is claimed (see "Limits and fallbacks").

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
spares. A host budget well below the VRAM budget is therefore refused in inclusive mode.

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
lent ring slot and file. The unlimited case is the same function without file and lent locations. One
mover, `l1_arena::execute`, moves every slice of a transaction whatever tier its two ends are in; for
a source in the file a small reader gives it the batched read and the ring address. Before an install
the device drains and the worker stops; ordered copies preserve every source until its last use;
tables are published only after the copies finish. Growing the ring retires lent slots through the
same transaction before reusing their bytes.

## Options

Use `--load-mode none`, `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`: the
cache reads host and ring memory from the kernels, which is the host-direct path, off by default and
without a CLI flag (`host-direct-moe.md`). Do not combine cache placement with `--cpu-moe` or
`--n-cpu-moe`.

| Option | Default | Meaning |
|---|---|---|
| `--expert-l2-mib N` | -1 | Host memory budget in MiB; -1 is unlimited and 0 is refused. |
| `--expert-l2-staging-mib N` | 0 | Ring size in MiB; 0 is automatic. Sets both rings unless one of the two below overrides it. |
| `--expert-l2-prefill-ring-mib N` | the staging value | Ring while a prompt is processed; 0 is automatic. |
| `--expert-l2-decode-ring-mib N` | the staging value | Ring while tokens are generated; 0 is automatic. Only differs from the prompt ring with `--expert-prefill-swap`. |
| `--expert-l2-worker-cpu N` | -1 | Logical CPU of the worker thread; -1 is the last active logical CPU. |
| `--expert-l1-mib N` | 0 | VRAM budget; zero is valid with a finite tier. |
| `--expert-cache-mode MODE` | inclusive | Relation of the VRAM set to the host set. |

`llama-bench` accepts the same host tier options.

The validator refuses a finite tier with a multimodal projector and with speculative decoding: those
combinations have no correctness check yet, not a structural problem. More than one server slot is
accepted by the tier; the prompt swap keeps its single-slot rule.

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_L2_QD` | 4 | Read queue depth of the worker. |
| `RANMA_EXPERT_L2_WAIT` | 30000 | Deadline in ms for the CPU-side I/O queue. |
| `RANMA_EXPERT_L2_VERIFY` | the `RANMA_EXPERT_VERIFY` value | Independent buffered payload and ownership checks. |

All three are read in one controller function, through the same validating parser.
`RANMA_EXPERT_TRACE=8` adds `expert_metrics` JSON lines with cumulative reads, ring hits, SSD payload
bytes, CPU service time, GPU waiting per phase and per-ubatch samples, plus the round lines and
install lines of the profile banks.

## What it costs

- Host memory: the budget, of which the ring, tables and spares take their share before the
  residents.
- Per routed layer, always: two small kernel launches (publish and wait) that answer themselves when
  nothing is demanded from the file.
- Per prompt ubatch with file residents: the synchronous file service, proportional to the bytes the
  ubatch demands from the file.
- One worker thread, spinning on the mailbox, pinned to one logical CPU.
- At install: file reads for every slice promoted from the file, in addition to the copies.

## Limits and fallbacks

- **Windows and HIP only**, for the unbuffered read queue and the address reservation. The validator
  refuses the option elsewhere before the model loads.
- **Prompt processing with file residents is synchronous.** The wait kernel blocks the stream while
  the worker reads; nothing overlaps that read with matrix multiplication. Chunking the current
  layer's demand and overlapping its reads is a possible later change, not done here.
- **The ring is sized for the worst possible distinct demand of one ubatch**, so a large ubatch
  reserves a large ring. When the ring is smaller than what one prompt ubatch reads from the file,
  and a prompt reads that set in layer order once per ubatch, a slice has been rotated out by the
  time it is wanted again, so ring reuse during prompt processing is close to zero. Sizing the
  prompt ring from the plan and replacing round-robin by a hit-refreshing policy are the next steps;
  both change nothing until the ring is at least as large as that set.
- **The prompt swap's boundary install reads from the SSD** with a finite tier, so the swap costs
  noticeably more per request than it does with everything resident (`expert-cache-banks.md`).
- **No correctness check yet** for a finite tier with a multimodal projector or speculative decoding;
  the validator refuses both.
- **A lazily mapped model file slows the unbuffered reads.** A read-only mapping of a shard makes
  unbuffered reads of the same file markedly slower on this platform, which is why the loader maps
  only the shards that hold lazy tensors when ordinary mmap loading is off. A shard that holds both
  a lazy tensor and routed experts still pays that cost.

## How to verify it

- `test-expert-l2` replays the ledger on the CPU: demand resolution, pinning and lease extension,
  round-robin eviction, ring growth and shrinkage, and deliberately invalid assignments.
- `test-expert-os` covers the OS wrapper: the reservation, the commit of the host arena and the
  unbuffered read queue.
- `test-expert-plan` covers the four-location transaction, including promotions out of the file.
- `test-expert-l2-gpu` is an optional HIP target that drives a shifted ring slice through MMQ, which
  is the check that the shift and the cleared tail are right on the device.
- `RANMA_EXPERT_L2_VERIFY=1` re-reads every payload through a buffered path and checks ownership
  after each install; `RANMA_EXPERT_TRACE=8` reports what each interval read from each tier.
