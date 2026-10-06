# Expert cache: finite host tier and file backing (`--expert-l2-mib`)

## What it is

`--expert-l2-mib N` bounds the host memory the expert cache may hold. L1 is VRAM, L2 is host memory,
and the GGUF file on the SSD is the backing store. An expert that fits in neither resident tier stays
in the file and is read on demand into a fixed ring of host slots before its layer uses it; MMVQ and
MMQ read that ring through mapped memory over PCIe, exactly as they read a host resident.

The ring lives inside the host arenas of the residents, one ring per storage class ("Storage
classes" below).

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

Prompt ubatches take a staged form of steps 3 and 4 ("Staged service for prompt batches" below);
generation keeps the single wait.

One row and a full ubatch use this same path. There is no whole-layer mode and no next-layer
prefetch: a large prompt can select every expert of a layer, but only its actual bitmap decides which
slices are read, and the next layer's selection is not known until its router has run.

Each storage class ring uses least recently used replacement with pinning. Every hit and every fill
moves its slot to the most recently used end of the class's list, and a miss takes the least recently
used ring slot that is neither pinned by the current demand nor leased by a generation the GPU has
not finished. The worker invalidates an evicted address before overwriting its slot, and a ring
occupant stays marked in the serve bitmap so that the worker extends its lease before a later layer
can evict a slot the current layer still reads. The replacement decides which slices are read from
the file and where they land, never their contents.

### Staged service for prompt batches

A ubatch with more rows than the decode bound (`n_parallel * (draft_max + 1)`) is served one weight
kind at a time. The worker issues the layer's file reads grouped by kind, in the order the graph
multiplies them: up, gate, down. As soon as the last read of a kind is in place it publishes that
kind's addresses and a per-kind readiness generation, while the reads of the next kind stay in
flight in the same queue. On the GPU the route waits only for the up slices; the gate and down
launches are each preceded by one small wait kernel for their own kind. So the gate reads overlap
the up multiplication and the down reads overlap the gate multiplication.

The ledger service is unchanged: the same slices are read into the same slots with the same pins
and evictions, and the kernels and their arguments are the same, so the output is bit-identical to
the single wait. Only the order of the reads and the moment each kind becomes visible differ. The
up reads of a layer cannot overlap anything in that layer, because its routing is known only just
before the up multiplication. Every kind wait also passes on the layer's whole-generation readiness,
so a generation the GPU answered itself, or one the worker served in one piece, never blocks on a
per-kind signal.

The gain comes from the idle time the single wait leaves: with it, the GPU sits idle for all of a
layer's file reads, while those reads use only a small share of the host I/O bandwidth that the
GPU's own reads of host memory also need.

`RANMA_EXPERT_L2_STAGED=0` restores the single wait for every ubatch. A plan that leaves nothing in
the file launches no kind waits.

The mailbox exists from the moment the tier allocates, not from the first plan that leaves something
in the file: captured graphs hold its two kernels, so the alternative would be a plan-dependent
graph. With an all-clear serve bitmap the two kernels answer themselves and cost two small launches
per routed layer.

### Early routes (layers routed by token id)

Some models route their first layers by the input token id alone: DeepSeek V4 gathers the selected
experts of its hash layers from a `[n_expert_used, n_vocab]` table (`ffn_gate_tid2eid`). The loader
declares such layers in `ggml_expert_config::early_route_layers`; `llama_decode` passes their
experts to the cache (`ggml_expert_iface::route_hint`) for every token ubatch of at most the decode
row bound, before the graph is built, and the server does it right after sampling
(`llama_expert_route_hint`), before the rest of its loop and a draft model's decode. What the cache
does with a hint is chosen by `RANMA_EXPERT_HASH_EARLY` (default `0`: the hint is refused and
nothing changes):

- `ssd`: the tier worker serves the hinted demand at once, lowest layer first, as one read batch
  into the storage class ring, with the ordinary ledger rules except that nothing is leased to a
  generation (a slot read early is reusable at once and the most recently used; a hit on a slot a
  pending generation holds keeps that lease). The layer's generation then finds the experts in the
  ring. A hint whose layer published since the hint is dropped (`early_late`). While the batch reads
  the later layers, a published generation that needs no read is answered at once
  (`early_inline`). The reads count once in `reads`/`ssd_bytes`; the generation's hits on them
  count in `early_hits`, not in `ring_hits`.
- `vram`: each early layer takes `RANMA_EXPERT_HASH_STAGE_SLOTS` VRAM slots of its size class
  (default 6, taken from the static capacity at load and logged), placed after the exchange spares
  and outside every plan. For a hint, the hinted experts that are host residents are copied
  into them on a side stream with the copy engine, and the layer's slot table points at them, so the
  matmuls read VRAM. Order: a kernel on the compute stream stores the layer's staging generation;
  the side stream waits for an event of the compute stream (every earlier graph has finished with
  the slots), removes the table entries of the slots it reuses (only where the table still names
  that slot), copies, points the table at the slots and stores the generation as done. A kernel
  before the layer's matmuls waits until done reaches the generation. A table entry always names a
  slot that holds exactly that expert's bytes, and an install that rewrites the table only removes
  entries. Needs owned host storage: exclusive mode or a finite host tier.
- `both`: the two together.

`expert_metrics` lines: the `l2` line carries `early_*` counters and `early_layer_decode_wait_ms`
(the decode wait of the declared early layers, also with the switch off); a `hash_stage` line per
bank commit carries the staging counters (copies, bytes, kept, over the slot count, host time of the
hints) and per layer the selections read from a staging slot (`hits`), those without a VRAM slot
(`host`, host or file) and the launches whose wait found the staging unfinished (`waits`).
`RANMA_EXPERT_VERIFY` also compares every staged slice with its host bytes after each hint, and
`RANMA_EXPERT_L2_VERIFY` checks the early reads like every other read.

### Storage classes (the class layout)

A storage class is a size class (the experts of the layers whose three tensors have one type and
shape), or several of them merged. Each storage class has one host arena per kind; its slots are the
host residents of its size classes, at load one contiguous range per size class, followed by the
storage class's staging ring. Every slot has the same pitch, so a slot can hold a resident or a ring
occupant without moving its bytes, and the install transaction relabels slots between the two roles
("Placement and the install transaction"): after the first install a size class's host slots are
spread over the arena. The arenas are registered once, coarse grained, like the host arena without a
tier.

Merge rule: a size class of at most two layers is merged into the nearest storage class of at least
its stride when the prompt floor bytes it saves exceed the padding every expert of the affected
classes would waste as a resident. Such a class otherwise needs a ring that holds its worst layer's
file residents at once. A draft model's layers in a joint cache take part in the same rule.

The pitch of a kind is `align_up(max slice + 4095 + MMQ tail, 4096)` over the storage class's
members, 8 KiB more than the slice for 4096-wide or 2048-wide tensors. It leaves room for the sector
shift of the tensor's file offset (GGUF tensor starts are not sector aligned) and the zeroed tail.

Every payload, a host resident's as well as a ring occupant's, starts at `slot + shift` of its layer
and kind. The host slot table cannot express that, so the tier's address table serves every host
resident too: one 8-byte load per non-VRAM expert and kind, which ring occupants already pay. The
kernels are unchanged. The host slot table stays published for the paths without a tier.

The slot ledger (`expert-l2-class-ledger.h`) gives every slot one role: resident, ring or free. Each
ring is its own least recently used list with the pins and leases described above. Its
relabel operations are the install transaction's building blocks and are covered by the CPU tests:
a resident becomes a ring occupant of the same expert in place (the ring grows, no read), a ring slot
becomes a resident (the ring shrinks; a read only when the slot does not hold that expert), and no
ring shrinks below its floor. The install transaction (`expert-l2-relabel.h`) is built from them.

### Addresses and padding

For each kind (gate, up, down), the pitch of a storage class includes its largest slice, the maximum
4095-byte file-sector shift, the MMQ read-ahead padding, and rounding to 4096 bytes.
`shift = file_offset % 4096`; the GGUF's 32-byte alignment keeps the payload vector aligned. Reads
stay at that shifted address; there is no compaction.

Every byte from `shift + slice_bytes` to the end of the pitch is cleared after every read. MMQ reads
whole K tiles beyond the final row, so stale quantized bytes there produce NaN even when multiplied
by zero (`expert-cache-prefill.md`). A host resident written by an install move gets the same
clearing, because the slot's previous occupant may have had a smaller shift.

## Ring size and budget

For a layer with E experts, U selected experts per row, and R rows:

```
maximum distinct demand = min(E, U * R)
floor bytes             = maximum distinct demand * sum(maximum pitch of each kind)
```

The prompt bound uses `n_ubatch`; the decode bound uses `n_parallel * (draft_max + 1)`. The rings
are sized per storage class (below); a larger ring is a performance knob.

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

**Rings per storage class.** Each storage class has a ring of its own, and it must hold every file resident of any
one of its layers at once: a prompt ubatch can demand them all, and they stay leased until the layer
is done. That is the prompt floor of the class: the most file residents one of its layers keeps
under the plan that seeds the load, never more than `min(E, U * n_ubatch)`. The floors are found by
growing the rings from zero until the host cut under them keeps no layer above its ring. Without a
stored profile every storage class's floor is `min(E, U * n_ubatch)`.

The ring of a storage class seeded from a stored profile is `RANMA_EXPERT_L2_RING_FACTOR` times its
floor (default 1, the floors), rounded up and at most the class's layers x E slots. A factor above 1
gives each layer's reusable experts more room to survive in the shared class ring until that layer
comes round again, but every extra ring slot is a host resident less. Measured on DS4F (exclusive,
L1 16 GiB, no speculative decoding, replies identical): factor 2 helps only out of domain (roleplay
profile, coding requests, L2 24 GiB: 14.00 -> 14.23 t/s, +1.7 %; factor 3 +1.2 %) and costs in domain
(roleplay, L2 24 GiB: 20.12 -> 19.79 t/s, -1.6 %; L2 40 GiB: 25.15 -> 24.79, -1.4 %), where the host
residents it removes cost more SSD reads than the ring hits it adds (478863 -> 545988 reads at
24 GiB). The CPU replay had predicted a gain in domain as well. The minimum budget stays the floors: when the budget above that minimum cannot hold
the full factor, every class keeps its floor plus the same fraction of what it wanted above it (one
log line); the factor never refuses a budget. A larger ring leaves fewer host residents, so a class
whose layers then keep more experts in the file than its ring grows to that count (the floors are
used when that no longer fits). Without a stored profile the factor is not applied: the floors are
then already the most one layer can demand (`min(E, U * n_ubatch)`, 7.0 GiB of rings for DS4F;
factor 2 would take 14 GiB), and one log line says so.

`--expert-l2-staging-mib` overrides the factor. It is the total over the classes: the classes get their floors and the rest in
proportion to the bytes each leaves in the file; a total below the floors is raised with one log line.
Both phases use the same rings: the prefill swap lends no slots,
and there is no decode reserve. Every later plan is held to the rings: a layer whose greedy cut
would leave more experts in the file than its ring holds takes its best file experts back from the
weakest host residents of other layers of its class (`tier_inputs::file_cap`; the per-class counts do
not change). An install whose plan cannot be held to the rings stops the process with the numbers.
Every resident slot, every spare and every ring slot costs the storage pitch. The minimum budget is
the rings, metadata, spares and tails (plus `P_l1` for inclusive); a budget below it is refused with
the numbers.

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

`expert-plan.h::plan_install` produces one transaction over three locations: VRAM slot, host slot
and file. The unlimited case is the same function without file locations. One
mover, `l1_arena::execute`, moves every slice of a transaction whatever tier its two ends are in; for
a source in the file a small reader gives it the batched read and the ring address. Before an install
the device drains and the worker stops; ordered copies preserve every source until its last use;
tables are published only after the copies finish.

**Install by relabel.** The SSD tier runs the transaction's host and file moves itself (`l2_tier::install_relabel`, planned by the pure
`expert-l2-relabel.h`) and the ring contents survive the install. A host slot of the plan
(`expert_location::slot`) is a logical slot; the tier maps each one to a storage slot of its storage
class, and the install changes that map instead of moving bytes:

- A host resident that leaves for the file stays where it is: its slot joins the ring with the
  expert as a cold (least recently used) occupant. A later demand is a ring hit, a later promotion a
  relabel.
- A promotion from the file that the ring holds takes that ring slot as its home: no read, no copy.
  A promotion to VRAM of a ring occupant is copied from its ring slot without a read.
- Any other promotion from the file takes an empty ring slot, or else the least recently used one
  that no generation leases, and is read straight into it; for a host promotion that slot becomes the
  home, for a VRAM promotion the slice is then copied up. A demotion from VRAM to a host slot is
  copied back into such a ring slot, which becomes the home. The slot a host resident promoted to
  VRAM leaves joins the ring empty once its copy is queued.
- Each ring keeps its size: every host slot that changes gives its storage slot to the ring and
  takes one from it, and the number of host slots per size class is fixed. The transaction checks
  this and that every home holds the plan's expert.

Nothing computes during the install: the device is drained, the worker stopped, and the done
generation of every layer is refreshed first, so no ring slot is leased; a leased slot would be
skipped, and a leased occupant is never relabeled (the install stops instead). Moves from the file
run in read batches: all reads of a batch go out together into distinct slots, then the batch's
copies to VRAM, and the copy stream is synchronized before the next batch's reads; no slot is read
into or copied from twice in one batch. The address table, the serve bitmap and the VRAM tables are
published after every read and copy has completed. A prompt/decode plan switch of the prefill swap is
the same transaction: the class layout has no lent slots and one ring size, so nothing else changes
at a switch.

The install line reports the moves from the file split into `relabel` (host promotions from the
ring), `ring-copy` (VRAM promotions from the ring) and `read`, and the ring occupants kept across the
install; the `install` metric adds `read_bytes` (bytes actually read; `ssd_bytes` still counts every
move from the file), `relabel_bytes`, `ring_copy_bytes`, `ring_kept` and `demoted_to_ring`.

## Options

Use `--load-mode none`, `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`: the
cache reads host and ring memory from the kernels, which is the host-direct path, off by default and
without a CLI flag (`host-direct-moe.md`). Do not combine cache placement with `--cpu-moe` or
`--n-cpu-moe`.

| Option | Default | Meaning |
|---|---|---|
| `--expert-l2-mib N` | -1 | Host memory budget in MiB; -1 is unlimited and 0 is refused. |
| `--expert-l2-staging-mib N` | 0 | Total ring size over the storage classes in MiB; 0 is automatic (the floors times `RANMA_EXPERT_L2_RING_FACTOR`). |
| `--expert-l2-prefill-ring-mib N` | the staging value | Overrides the staging value; 0 is automatic. |
| `--expert-l2-worker-cpu N` | -1 | Logical CPU of the worker thread; -1 is the last active logical CPU. |
| `--expert-l1-mib N` | 0 | VRAM budget; zero is valid with a finite tier. |
| `--expert-cache-mode MODE` | inclusive | Relation of the VRAM set to the host set. |

`llama-bench` accepts the same host tier options.

The validator refuses a finite tier with a multimodal projector and with speculative decoding: those
combinations have no correctness check yet, not a structural problem. A draft model in the joint
cache (`expert-cache-joint.md`), a block drafter or an MTP head loaded with `-md`, is the exception, and
so is an MTP context on the target's own weights (`--spec-type draft-mtp` without `-md`). More than one server slot is
accepted by the tier; the prompt swap keeps its single-slot rule. The slot count has to be given: without
`-np` llama-server leaves it to be chosen later (four slots), and the validator refuses the tier with
`finite L2 needs positive batch/parallel bounds`. Give `-np 1` for one user (`expert-cache.md`).

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_L2_QD` | 4 | Read queue depth of the worker. |
| `RANMA_EXPERT_L2_WAIT` | 30000 | Deadline in ms for the CPU-side I/O queue. |
| `RANMA_EXPERT_L2_VERIFY` | the `RANMA_EXPERT_VERIFY` value | Independent buffered payload and ownership checks. |
| `RANMA_EXPERT_L2_STAGED` | 1 | Staged service: 0 is off, 1 covers the ubatches with more rows than the decode bound, N > 1 the ubatches of at least N rows. |
| `RANMA_EXPERT_L2_STAGED_DRAIN` | 0 | 1 completes every read of a kind before the next kind is issued (a diagnostic; the queue then drains at each kind). |
| `RANMA_EXPERT_L2_RING_FACTOR` | 1 | Each storage class ring is this many times its prompt floor, as far as the budget allows; a real number of at least 1, 1 = the floors. Not applied without a stored profile. `--expert-l2-staging-mib` overrides it. |
| `RANMA_EXPERT_HASH_EARLY` | 0 | Layers routed by token id ("Early routes"): `ssd` reads their file-tier experts when the tokens are known, `vram` stages their host-resident experts in VRAM slots, `both`; 0 changes nothing. |
| `RANMA_EXPERT_HASH_STAGE_SLOTS` | 6 | `vram`/`both`: staging slots per early layer (1..32), taken from the static VRAM capacity. |

All of them are read in one controller function, through the same validating parser.
`RANMA_EXPERT_TRACE=8` adds `expert_metrics` JSON lines with cumulative reads, ring hits, SSD payload
bytes, CPU service time, GPU waiting per phase and per-ubatch samples, plus the round lines and
install lines of the profile banks. The `l2` line names the ring slots of each storage class; the `budget` line adds the storage classes (members, pitch, ring,
floor) and `slot_padding`, the resident slot bytes beyond the payload. With the staged service, `prompt_wait_ms` and the per-layer
samples count the wait for the up slices only; `staged_wait_ms` adds up the gate and down waits, and
`staged_generations` counts the generations the worker served in stages.

## What it costs

- Host memory: the budget, of which the ring, tables and spares take their share before the
  residents.
- Per routed layer, always: two small kernel launches (publish and wait) that answer themselves when
  nothing is demanded from the file.
- Per prompt ubatch with file residents: the file service, proportional to the bytes the ubatch
  demands from the file. The up reads of each layer stall the stream; the gate and down reads stall
  it only for the part the preceding multiplication does not cover. Two more small wait kernels per
  routed layer.
- The SSD reads share the host I/O bandwidth with the GPU's reads of host memory, so the reads that
  overlap a multiplication slow it somewhat.
- One worker thread, spinning on the mailbox, pinned to one logical CPU.
- At install: file reads for every slice promoted from the file, in addition to the copies. With the
  class layout's relabel install only for the promotions that the rings do not hold.

## Limits and fallbacks

- **Several models.** In a joint cache (`expert-cache-joint.md`) each model's file-resident experts
  are read from its own GGUF; the tier numbers files by path. A model that joins later opens its
  files and has its backing checked when its load ends.
- **Windows and HIP only**, for the unbuffered read queue and the address reservation. The validator
  refuses the option elsewhere before the model loads.
- **The up reads of a layer are not overlapped.** The staged service hides the gate and down reads
  behind the up and gate multiplications, but a layer's routing is known only just before its up
  multiplication, so its up reads always stall the stream. There is no next-layer prefetch, and a
  multiplication never starts on part of a kind's slices.
- **Generation waits for all three kinds at once.** A decode ubatch reads few slices per layer, and
  it keeps the single wait.
- **Rings per storage class cost more than one shared ring.** Each storage class holds its own worst
  layer's file residents, so the floors together are larger than one ring for the whole model (DS4F:
  about 4.0..5.2 GiB at L2 40..24 GiB against 2.8 GiB, est.), a factor above 1 multiplies that as far
  as the budget allows, and without a stored profile each floor is `min(E, U * n_ubatch)` slots. The
  ring sizes are fixed at load: there is no sizing by measured reuse and no resizing at install.
- **The ring is sized for the worst possible distinct demand of one ubatch**, so a large ubatch
  reserves a large ring. When the ring is smaller than what one prompt ubatch reads from the file,
  and a prompt reads that set in layer order once per ubatch, a slice has been rotated out by the
  time it is wanted again, so ring reuse during prompt processing is close to zero.
- **The prompt swap's boundary install reads from the SSD** with a finite tier, so the swap costs
  noticeably more per request than it does with everything resident (`expert-cache-banks.md`). The
  class layout reads only the promotions its rings do not hold.
- **Demoted residents go first.** A resident that the install moves to the file joins its ring as the
  least recently used occupant, so the reads of the same install evict it before older occupants
  (after the empty slots). Whether another order keeps more useful experts is not measured.
- **No correctness check yet** for a finite tier with a multimodal projector or speculative decoding
  other than a draft model in the joint cache; the validator refuses those.
- **A lazily mapped model file slows the unbuffered reads.** A read-only mapping of a shard makes
  unbuffered reads of the same file markedly slower on this platform, which is why the loader maps
  only the shards that hold lazy tensors when ordinary mmap loading is off. A shard that holds both
  a lazy tensor and routed experts still pays that cost.

## How to verify it

- `test-expert-l2` replays the ledger on the CPU: demand resolution, pinning and lease extension,
  round-robin eviction of the fixture ring, ring growth and shrinkage, and deliberately invalid
  assignments. For the class layout it covers the
  pitch and shift arithmetic, the merge rule, slot placement, floors and the ring split, the host
  cut's per-layer file cap, and the class ledger: roles, per-class service against a slow least recently used reference,
  both relabels with leases and floors, transient install slots, and an install as relabels. The
  relabel install transaction is replayed on a byte model of the arenas and VRAM: demotions kept in
  the ring, promotions from the ring without a read, VRAM promotions and demotions, leased slots
  skipped and never relabeled, read batches that never reuse a slot, a prompt/decode switch back and
  forth without reads, and random plans and demands for both modes and several ring sizes.
- `test-expert-os` covers the OS wrapper: the reservation, the commit of the host arena and the
  unbuffered read queue.
- `test-expert-plan` covers the four-location transaction, including promotions out of the file.
- `test-expert-l2-gpu` is an optional HIP target that drives a shifted ring slice through MMQ, which
  is the check that the shift and the cleared tail are right on the device.
- `RANMA_EXPERT_L2_VERIFY=1` re-reads every payload through a buffered path and checks ownership
  after each install; `RANMA_EXPERT_TRACE=8` reports what each interval read from each tier.
- The staged service changes no byte a kernel reads: `llama-perplexity` with `RANMA_EXPERT_L2_STAGED=0`
  and with the default must print identical values, and `staged_generations` in the trace shows that
  the staged path ran.
