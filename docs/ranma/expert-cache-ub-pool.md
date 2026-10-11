# Expert cache: the decode pool in the compute buffer

## What it is

A compute buffer is reserved for the largest graph a context computes: a prompt ubatch at the full
context. A decode graph (one token, or the few rows of a speculative verification batch) uses only
its lower part. The decode pool lends the rest of that buffer, while the graphs stay small, to extra
VRAM positions of the expert cache: per routed layer, the same number r of least recently used
positions, each holding a copy of one expert whose home is host memory (inclusive). Before a graph
that reaches into the lent region (a prompt ubatch, a large catch-up), the pool takes the region
back in stream order, and the next small graph lends it again.

`RANMA_EXPERT_PREFETCH` (experimental, GLM-5.3 Flash) fills pool positions with predicted experts before their
layer runs: [expert-prefetch.md](expert-prefetch.md).

It is on by default with the expert cache (`RANMA_EXPERT_UB_POOL=0` turns it off). Nothing moves
either way: an L1 resident stays where it is, the static plan does not change, and an expert in a
position is still in its host home. The pool changes only which address a matmul reads, so the
output is the same with the pool on or off.

## How a graph finds a position

The kernels already take a 64-bit address table before the host slot table
(`ggml_cuda_expert_cache_select`): a pooled expert's entry is the address of its position. The
previous value (its home address, or 0) is kept and written back when the position is evicted or
the pool is taken back. No matmul kernel changes. With the SSD tier the table is the tier's, and only
host residents are admitted (the experts the SSD worker serves are left to it, except "Ring admission" below); without a table from
the tier or a host arena the pool brings its own, zero by default.

For every launch of up to `rows` rows (the decode rows: `n_parallel x (draft_max + 1)`), a step kernel
runs on the compute stream right after the router and before the layer's matmuls. It stamps the
positions the launch hits and gives each admissible distinct miss a position: an empty one first,
then the oldest, never one this launch uses, later rows first. A copy kernel next in stream order
copies the three kinds of every new entry from its home. Both kernels run whether or not the pool
is lent (r = 0 makes them return at once), so a captured graph stays valid either way.

## Where the region comes from

- The graph allocator places a graph of at most `rows` rows on its own (`ggml_backend_sched_set_alloc_compact`),
  so that it uses only the start of the buffer; `ggml_backend_sched_get_buffer_extent` reports how far
  it reaches, also for a resumed graph.
- Before every graph, llama passes the buffer, its size, the graph's extent and its rows to the cache
  (`compute_spare` of the expert interface, ABI 8). The cache keeps the largest extent of the decode
  graphs per buffer; the region starts at that mark plus a margin (`max(64 MiB, mark / 4)`, 2 MiB
  aligned) and must be at least 256 MiB.
- Only the largest compute buffer lends: the target's, not the smaller one of a draft context. With
  `LLAMA_DRAFT_SHARE_COMPUTE=1` (default off) a drafting context, an MTP context or a draft model's context
  (`-md`: DSpark, DFlash, EAGLE3 and the plain draft model), computes in views of the target's buffers, and
  each context waits for the other before its graphs. On DS4.1F with DSpark at ubatch 2048 the draft's own
  compute buffer is 2220 MiB, which sharing should save (not measured yet). A draft outside the target's
  expert cache is not supported while the target has one (it would not take back what the pool lends): it
  keeps its own buffers, with one warning line. Following up with that case would mean letting such a draft
  call the target's cache hooks.
- Every allocation from a shared buffer type logs one line: `a new shared block of N MiB`, or `a N MiB buffer
  is a view of the shared M MiB block`. The scheduler's `compute buffer size` line prints the size the context
  asked for in either case, so only these lines tell whether a draft's buffer took VRAM of its own.
- `LLAMA_MTP_SHARE_COMPUTE` is an alias of `LLAMA_DRAFT_SHARE_COMPUTE` (one info line when it turned sharing on);
  when both are set the canonical name applies, with one warning line if their values differ.
- The decode high-water mark of a buffer counts the small graphs (at most `rows` rows) of every context
  that computes in it, so with a shared buffer the region starts above the draft's decode graphs as well
  (a DSpark block forward and the injection of the accepted tokens), and a draft forward does not take
  the region back. Only the cached model's own contexts lend. A larger draft graph (the injection of a
  prompt) reaches into the region and takes it back, like the target's prompt ubatch at the same request
  boundary; the next small graph of the target lends again.
- An install, a member join or a scheduler reserve takes the region back synchronously.

## Measured

GLM-5.3 3.05 bpw, server with smart MTP, code preset of 12 requests, ub 2048, L1 16896, L2 86016, pool and
`LLAMA_DRAFT_SHARE_COMPUTE` on (then set through its alias); one run per row, each after a GPU rest:

| row | filter | decode t/s | from request 4 | pool hits | fills |
|---|---|---|---|---|---|
| e | none | 18.93 | 19.77 | 219745 | 566198 |
| h | ghost:8 | 19.95 | 20.56 | 194404 | 260931 |

A lend starts empty. A warm lend (filling the positions at the lend with the experts of the last
reclaim or the profile's best host residents) was tried and removed (2026-10-10): it gave no gain
(19.96 t/s with ghost:8 on this preset; on a roleplay preset 24.40 against 24.76 t/s without it),
and its copy of 90-250 ms per lend landed in one decode step.

## Admission filter

`RANMA_EXPERT_UB_POOL_FILTER=ghost:K` admits a miss only when the same expert also missed in the
layer within the last K steps of the layer. It is the recent pool's ghost term
(`RANMA_EXPERT_RECENT_FILTER`, 2026-09): a step's misses are stamped after its decisions, so a miss
never admits itself, and a step is one launch (a verification batch of several rows is one step).
Steps count only the launches that step a lent pool, and the stamps survive a reclaim, so the window
spans lends. On by default with K = 8 (`off` turns it off): in the table above it cut the fills by
54 % and lifted decode by 5 % (h against e).

## Ring admission (experimental)

`RANMA_EXPERT_UB_POOL_RING=1` (default 0) also admits the experts that the SSD worker serves from its host
ring (ring occupants), under the same admission filter. A CPU replay of DS4F traces put the gain at r = 9 (the
positions a 2048-row ubatch leaves on DS4.1F) at 1.5 to 2.6 ms per token over host residents only with a 24 GiB
host tier, and 0.04 to 1.0 ms with 40.8 GiB; it is not measured on the model yet. File residents (in neither
ring nor host memory) are still not admitted.

- **Which.** A miss the worker serves is a ring occupant when its three table entries are not 0: the worker
  zeroes an entry before it reuses the slot, and an expert in the file has entry 0.
- **Copy.** The copy kernel reads the ring slot as it reads any home, on the compute stream, before the tier's
  publish of the same layer. Nothing else protects the slot: the worker serves only while the stream waits at a
  layer's readiness, so no read can land in that slot during the copy. Because of this the ring admission stays
  off with `RANMA_EXPERT_HASH_EARLY=ssd|both` (early reads run while the graphs run). Every context computes
  on its own stream: before a graph of another context than the last one, the controller waits for the last
  one's stream (normally idle already, since its outputs were read). A joint cache's draft model is judged
  at the first graph of its context with a known buffer: at the base of the target's compute buffer it
  computes in a view of it, which exists only with `LLAMA_DRAFT_SHARE_COMPUTE=1`, whose contexts synchronize
  each other's backend before every graph, and the ring admission stays on; anywhere else (no sharing) it
  turns off for good, after the ring positions went back. Either outcome is logged in one line.
- **While pooled.** The pool marks the expert in a bitmap per layer, written by the kernel that changed the
  layer's positions into device memory and into mapped host memory. The tier's publish kernel leaves a marked
  expert out of the demand, so a pool hit never reaches the worker. The worker neither publishes nor
  invalidates a marked expert's entries. The expert is invisible to the ring meanwhile: no lease, no least
  recently used bump, so the ring ages it and may reuse its slot (the CPU replay modelled it the same way).
  After a staged service the worker skips its final repeat of the layer's publish (every kind was published
  before its readiness), since the GPU may be past the layer by then.
- **Leaving.** An eviction, a prefetch reservation or a reclaim writes 0 into a ring owner's entries instead of
  the saved slot address, which may hold another expert by now, and clears its mark. The next demand of the
  expert reaches the worker again: a ring hit republishes the slot, otherwise it is read from the file.
- **Diagnostics.** The counter line adds `ring_fills` (fills of ring occupants, part of `fills`), and the
  pool's on line says `ring occupants admitted`. `RANMA_EXPERT_VERIFY` checks a ring owner's table entries but
  not its bytes, whose source slot may be gone. `RANMA_EXPERT_L2_VERIFY` accepts a 0 entry of a served expert
  in the layers other than the one just served (an entry the pool let go, published again at the layer's next
  service); the layer just served and every layer after an install are still checked exactly.

With the switch at 0 nothing is allocated, the step kernel's ring branch is never taken, the publish kernel
gets a null bitmap, and the worker's checks see no bitmap.

## Switches

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_UB_POOL` | 1 | 0 turns the decode pool off. |
| `RANMA_EXPERT_UB_POOL_FILTER` | `ghost:8` | `ghost:K` (1..65535) or `off`: the admission filter above. |
| `RANMA_EXPERT_UB_POOL_MIN_MIB` | 256 | Smallest region worth lending. |
| `RANMA_EXPERT_UB_POOL_MARGIN_MIB` | 64 | Least headroom above the decode graphs' largest extent. |
| `RANMA_EXPERT_UB_POOL_RCAP` | 64 | Positions per layer allocated (the most a lend can use). |
| `RANMA_EXPERT_UB_POOL_ROWS` | decode rows | Launches of at most this many rows step the pool and are planned compactly. |
| `RANMA_EXPERT_UB_POOL_RING` | 0 | 1: also admit ring occupants of the SSD tier (experimental, above). |
| `RANMA_EXPERT_UB_POOL_LOG` | 0 | 1: lends, reclaims and counters. |

All are read once at startup. `RANMA_EXPERT_VERIFY` compares every position with its home before every
reclaim.

## Counts without a synchronize

Besides the device counters (logged with `RANMA_EXPERT_UB_POOL_LOG`, read after a synchronize), every
layer's hits and fills are written by its step kernel into a small mapped host array (16 bytes per
layer) when they change. `ggml_expert_iface::pool_counts` sums them with the lend and reclaim counts;
it never waits for the device, and the launches that have not finished are missing.
The `--spec-smart-log` round log uses it to record each round's pool work (`docs/ranma/spec-smart.md`).

## What it costs

- VRAM: about 0.3 MiB of state (geometry 46 x 288, r_cap 64); the positions live in the compute buffer.
- Per decode launch and pooled layer: one step kernel (one block) and one copy kernel launch.
- Per lend: an upload of the position addresses and a lend kernel.
- Per step that hits or fills: two 8-byte writes to mapped host memory.

## Not done

- Experts the SSD worker serves are not admitted (ring occupants are with `RANMA_EXPERT_UB_POOL_RING=1`; file
  residents never).
- Early-route (hash) layers and the layers of a joint cache's draft member get no positions.
- A compute buffer of more than one chunk reports no extent, so the pool never lends from it.
