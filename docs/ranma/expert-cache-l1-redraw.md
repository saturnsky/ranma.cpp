# Expert cache: redraw of the VRAM size-class split

## What it is

The load freezes one static VRAM (L1) capacity per size class: the greedy of the load-time plan
decides how many experts of each class fit into the L1 budget. Later plans only move experts inside
those class capacities. A cold load (no stored profile) or a model that joins the cache later (a
draft model or an MTP head, `expert-cache-joint.md`) can leave that split far from what the profile
asks for once it has been measured.

The redraw lets a plan install move VRAM capacity between size classes. The class arenas are then HIP
virtual address ranges backed by physical handles: shrinking a class unmaps handles at the top of its
range, growing a class maps handles at the top of its range. Base addresses never change, so slot
tables and captured graphs stay valid.

It is off by default. It turns on when at least one of its two conditions is set:

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_L1_REDRAW_BENEFIT` | not set | k > 0: a redraw fires only when its estimated benefit over the horizon exceeds k times its estimated cost. |
| `RANMA_EXPERT_L1_REDRAW_MOVE` | not set | p > 0: a redraw fires only when the proposed split moves more than p % of the L1 budget. |

When both are set, both must hold. An invalid value (not a positive number) leaves that condition
unset and prints one warning; with neither condition set the arenas are allocated as before and no
install evaluates a redraw. The measured row below used `BENEFIT=2` and `MOVE=5`.

## How an install decides

Only generation-bank plans are considered, inside the install's quiescence (device drained, cache
mutex held):

1. **Proposal.** The load's greedy runs again on the plan's scores within the current static VRAM
   bytes, held inside per-class floors and ceilings: a class never exceeds its address range; with an
   inclusive finite host tier the host cut must still hold the class's VRAM residents; with an
   exclusive finite host tier a class keeps enough VRAM slots that no layer leaves more experts in the
   file than its storage class ring holds (`expert-cache-l2.md`). At most 4.5 GiB change class per
   redraw; every class moves the same fraction of its way.
2. **Estimate.** The benefit per token is the read time the entering experts save (host link or SSD,
   by their current home) minus what the leaving ones cost, at each expert's selection rate; it is
   counted over 350 tokens. The cost is the promotion and demotion traffic, the handle remaps and the
   worst-case evacuation copies, from a cost model measured on the R9700 / PCIe 5.0 x16 / Gen5 NVMe
   target.
3. **Gate.** Every set condition must hold on two consecutive installs, except in the first two
   installs after a cold load or a member join. A redraw holds the next three installs off.

Each decision is one log line: `expert cache: L1 redraw check, plan N of bank 'B': static split ... ->
... (floors ..., ceilings ...), moves X MiB = Y % of the L1 budget (...); benefit ... ms, cost ... ms
(...), ratio R (...); streak s, hold-off h: fired | needs 2 consecutive proposals | hold-off | ...
below threshold | the split does not change`. With the L2 log bit (`RANMA_EXPERT_TRACE=8`) an
`expert_metrics {"kind":"l1_redraw_check",...}` line carries the same numbers.

## How a redraw moves

1. The kept residents of a shrinking class that sit at or above its new slot count move down to a
   free slot (VRAM to VRAM), or swap with a resident the new plan drops.
2. A shrink install: the plan with the shrinking classes at their new capacity and the growing ones
   at the old. Nothing is retained or placed in the retiring slots; exclusive mode runs the unequal
   exchange of the install transaction, so promotions and demotions need not pair up.
3. The arena moves the handles: the shrinking classes' top handles are unmapped and mapped at the top
   of the growing classes; new slots and the zeroed tail past the top slot are cleared.
4. A grow install of the plan in the new capacities.

Every step keeps one home per expert in exclusive mode, and the file stays the backing of what an
install drops. Plans committed before a redraw are placed again from their scores at install. Host
capacities stay as loaded. The VRAM staging slots of token-routed layers (`expert-cache-l2.md`,
"Early routes") sit at a fixed index above the largest range a class can reach and never move.

## Modes and fallbacks

- Inclusive mode, unlimited or finite host tier: supported.
- Exclusive mode with a finite host tier: supported.
- Exclusive mode with the unlimited host tier: supported through the host arena changes below.
- A frozen cache or a policy other than adaptive has no installs: refused at load with one warning.
- A device or driver without virtual memory management, or a failing VMM call at allocation: plain
  arenas and no redraw, with one warning. CUDA builds have no redraw.
- Handles are 64 MiB; the address range of each class covers the largest static capacity the class
  can reach within the current static bytes.

## Exclusive mode without a finite host tier

In exclusive mode with the unlimited host tier (`--expert-l2-mib -1`) each size class keeps one host
arena for the experts that are not in VRAM, sized by the load's split, and the kernels read a host
resident by its slot index. When the redraw is on in this mode, the host arenas change at load:

- **Address table.** The host arena publishes the device address of every host-resident slice per
  (layer, kind, expert), 0 for an expert in VRAM, and the lookup hands it to the kernels as the host
  address table they already read first with a finite tier. The kernels are unchanged and read the
  same bytes. The table is one small device buffer (DS4F/Qwen size: well under 1 MiB) rewritten at
  every install.
- **Chunks.** Each (class, kind) arena is a list of chunks of about 128 MiB (a whole number of slots of
  the class's largest slice; every kind of a class uses the same count), each allocated and
  registered on its own. A redraw first grows the host capacity of the classes that shrink in VRAM
  (new chunks) before the shrink install demotes their experts there. After the grow install the
  classes that grew in VRAM move their host residents down and release the freed top chunks.

Load logs one line with the table size, the chunk count, the slots per chunk and the slack (unused
slots of the top chunks, about half a chunk per class and kind on average). A redraw logs the host
capacities before the shrink install and after the grow install, with the chunks added or released and
the residents moved. The host part of a redraw (chunk growth about 10 ms per chunk, compaction copies,
release about 5 ms per chunk) is not priced in the gate, and host memory peaks at the moved bytes
above the final size during a redraw.

## Measured

DS4F with its MTP head in the joint cache, exclusive, L1 16 GiB, L2 60 GiB, both models cold, roleplay,
`BENEFIT=2`, `MOVE=5` (development build of this change): one redraw fired (25.8 % of the L1 budget
moved), decode 25.16 -> 27.34 t/s (+8.7 %, against an earlier row of a different session). A forced
redraw at every install under `RANMA_EXPERT_VERIFY=2` and `RANMA_EXPERT_L2_VERIFY=1` found no bad slice,
and the replies were identical. DS4F alone, roleplay, L2 24 GiB, cold: no redraw fired, -0.6 %
(within noise). Qwen exclusive with the unlimited host tier, the address table alone (a development switch,
no redraw): decode unchanged (47.24 t/s both ways, replies identical); forced redraws under `RANMA_EXPERT_VERIFY` (three
redraws, host chunks +6/-3) found no bad slice and the replies were identical.

## Limits

- Two installs per redraw: host residents the shrink install drops and the grow install wants back
  are read again. Bounded by the moved bytes; not priced in the gate.
- The gate's token normalisation mixes the layers of all cached models; it is an estimate.
- Whether WDDM evicts idle VMM handles like other device allocations is not verified; the GPU
  keep-alive covers ordinary allocations.

## How to verify it

- `test-expert-plan` covers the switches, the capacity recompute (identical to the load greedy
  without bounds), floors, ceilings and the byte cap, the estimate, the gate, the evacuation plan and
  the install with a retiring slot range.
- `test-expert-l2-gpu` runs eight redraws on a VMM arena in both modes through the real mover and
  checks every slice, the staging slots, the zeroed tails and a graph captured before the first
  redraw after every step. It also checks the host address table against the slot tables across
  installs, and runs redraws in exclusive mode without a finite tier with chunked host arenas (host
  growth, the two installs, compaction and chunk release).
- `RANMA_EXPERT_VERIFY` checks every install of a redraw like any other install.
