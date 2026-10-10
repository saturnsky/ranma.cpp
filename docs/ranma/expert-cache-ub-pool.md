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

It is always on with the expert cache. Nothing moves
either way: an L1 resident stays where it is, the static plan does not change, and an expert in a
position is still in its host home. The pool changes only which address a matmul reads, so the
output is the same with the pool on or off.

## How a graph finds a position

The kernels already take a 64-bit address table before the host slot table
(`ggml_cuda_expert_cache_select`): a pooled expert's entry is the address of its position. The
previous value (its home address, or 0) is kept and written back when the position is evicted or
the pool is taken back. No matmul kernel changes. With the SSD tier the table is the tier's, and only
host residents are admitted (the experts the SSD worker serves are left to it); without a table from
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
  `LLAMA_MTP_SHARE_COMPUTE=1` an MTP context computes in views of the target's buffers, and each context
  waits for the other before its graphs.
- An install, a member join or a scheduler reserve takes the region back synchronously.

## Measured

GLM-5.3 3.05 bpw, server with smart MTP, code preset of 12 requests, ub 2048, L1 16896, L2 86016, pool and
`LLAMA_MTP_SHARE_COMPUTE` on; one run per row, each after a GPU rest:

| row | filter | decode t/s | from request 4 | pool hits | fills |
|---|---|---|---|---|---|
| e | none | 18.93 | 19.77 | 219745 | 566198 |
| h | ghost:8 | 19.95 | 20.56 | 194404 | 260931 |

A lend starts empty. A warm lend (filling the positions at the lend with the experts of the last
reclaim or the profile's best host residents) was tried and removed (2026-10-10): it gave no gain
(19.96 t/s with ghost:8 on this preset; on a roleplay preset 24.40 against 24.76 t/s without it),
and its copy of 90-250 ms per lend landed in one decode step.

## Admission filter

The filter admits a miss only when the same expert also missed in the
layer within the last K = 8 steps of the layer. It is the recent pool's ghost term
(`RANMA_EXPERT_RECENT_FILTER`, 2026-09): a step's misses are stamped after its decisions, so a miss
never admits itself, and a step is one launch (a verification batch of several rows is one step).
Steps count only the launches that step a lent pool, and the stamps survive a reclaim, so the window
spans lends. In the table above it cut the fills by 54 % and lifted decode by 5 % (h against e).

## Settings

The pool has no switches for its size or admission: a region is lent from 256 MiB up with 64 MiB of headroom
above the decode graphs' largest extent, a layer is allocated 64 positions (the most a lend can use), launches
of at most the decode rows step the pool and are planned compactly, the admission filter is ghost:8, and the
ring admission follows the unified host policy.

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_UB_POOL` | 1 | 0 turns the decode pool off. |
| `RANMA_EXPERT_UB_POOL_LOG` | 0 | 1: lends, reclaims and counters. |

It is read once at startup. `RANMA_EXPERT_VERIFY` compares every position with its home before every
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

- Experts the SSD worker serves are not admitted.
- Early-route (hash) layers and the layers of a joint cache's draft member get no positions.
- A compute buffer of more than one chunk reports no extent, so the pool never lends from it.
