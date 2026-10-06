# Graph runtime

Changes to the machinery that turns a graph into work for a backend: the compute
buffers the graph allocator keeps, the way the scheduler gets the inputs of a
graph onto the device, the nodes a LoRA adapter adds to a graph, and the graphs
that llama and the HIP backend keep per batch shape.

## Compute buffer regrowth headroom

### What it is

When the graph allocator has to enlarge a compute buffer that it already
allocated once, it allocates the new buffer with headroom instead of allocating
exactly the planned size. The headroom is one `GGML_ALLOC_REGROW_HEADROOM_DIV`-th
of the planned size (the constant is 8 in `ggml/src/ggml-alloc.c`, i.e. one
eighth), clamped to the maximum allocation size of the buffer type. It applies
per chunk of a buffer.

The first allocation of a buffer is unchanged, and the plans are unchanged: only
the size of a buffer that grows after its first allocation differs.

### When it applies

`llama_context` reserves its compute buffers with a worst-case graph, and
`ggml_gallocr_reserve_n` only replaces a buffer when the plan of a later graph is
larger than what is allocated. That is correct only if the plan of a graph were
monotonic in the sizes of its tensors, and it is not: the best-fit planner can
need more room for a graph whose every node is no larger than the same node of
the reserved graph, because a large tensor that finds a free block inside the
plan of one graph may not find one in the plan of another.

A long context makes this visible. A prompt graph is planned again at every step
as the context grows; once a context-independent tensor stops fitting into a hole
of the plan, the plan grows by a small amount at every further step. Without
headroom the allocator then frees the buffer and allocates one that is slightly
larger at every step. A device allocator that cannot place the larger block in
the hole left by the smaller one keeps both, so the memory the process holds
grows by a whole compute buffer per step even though the reported free memory of
the device does not move. A server that alternates prompt processing and
generation at a long context walks into the same pattern.

With headroom, the step that grows the buffer pays once and the steps behind it
fit inside what was allocated.

### Options

None. The behaviour has no environment switch; the headroom divisor is a
compile-time constant in `ggml-alloc.c`.

### Limits

- The headroom is not free: a compute buffer that has regrown is up to one eighth
  larger than the plan it was grown for, and it keeps that size for the lifetime
  of the allocator.
- The clamp to `ggml_backend_buft_get_max_size()` means that a buffer type with a
  small maximum allocation size gets no headroom, and the old behaviour returns
  for it.
- Nothing is done about the first cause: the plan of a graph can still exceed the
  reserved plan. The headroom only keeps the repeated reallocations that follow
  from being one reallocation.

### How to verify it

Build without `NDEBUG` and watch the allocator's reallocation log
(`reallocating <buffer type> buffer from size A to B`). The line prints the size
that is really allocated, headroom included, not the planned size. On a run that
previously logged a reallocation per step at a long context there should now be
one line, with a size above the plan, and no further lines.


## Asynchronous graph input uploads

### What it is

The scheduler uploads the inputs of a split through a ring of pinned host slots
on the stream of the split's backend, instead of copying every input with a
blocking tensor copy. The host no longer synchronizes once per input.

### When it applies

`ggml_backend_sched_compute_splits()` has to copy the inputs a caller provided
(the tensors flagged `GGML_TENSOR_FLAG_INPUT`) into the buffers of the backend
that runs the split. A blocking copy has to wait for the backend before it can
touch the destination, so each input costs one host synchronization. That is
paid between the end of one token and the launch of the next graph, which makes
it a per-token cost of single-token decoding: the more inputs a model's graph
has, the larger it is. Prompt processing pays it too, but there it disappears
next to the compute.

The ring removes the wait: the input is copied into pinned host memory the
backend can read asynchronously, and the upload is queued on the same stream as
the graph. Being on that stream is what makes it safe - the upload is ordered
after the previous graph, which may still be reading the destination, and before
the graph that is queued next.

### How it works

Each backend of the scheduler owns one ring, allocated from the host buffer type
of the backend's device:

- `GGML_SCHED_INPUT_STAGING_SLOTS` slots (4), each at most
  `GGML_SCHED_INPUT_STAGING_MAX` bytes (4 MiB). Both are compile-time constants
  in `ggml/src/ggml-backend.cpp` and can be overridden by the build.
- A slot is sized from the bytes the previous graph compute asked for: twice that
  high-water mark, clamped to `GGML_SCHED_INPUT_STAGING_MAX`, never below 64 KiB.
  The ring is only reallocated when that is larger than the current slot size, so
  it settles after the first few computes and grows again when the input sizes
  grow with the context.
- Inside a compute, one slot is current and is filled by bumping an offset; each
  input takes its byte size padded to 256 bytes.
- At the end of a split, if anything was staged, an event is recorded on the
  backend. The slot carries that event until the ring comes round to it again,
  and only there does the host wait. With four slots, three graph computes may be
  in flight before a wait is possible.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `LLAMA_INPUT_UPLOAD_ASYNC` | env | `1` | `0` restores the blocking copy of every graph input. Read once per process. |
| `GGML_SCHED_INPUT_STAGING_SLOTS` | compile-time | `4` | Number of slots in the ring. |
| `GGML_SCHED_INPUT_STAGING_MAX` | compile-time | `4*1024*1024` | Upper bound of one slot; a larger input takes the blocking path. |

### Limits and fallbacks

The blocking copy is kept, per input, for everything the ring cannot serve:

- the backend has no `set_tensor_async`;
- the source is not in host memory;
- the destination is a view, or does not live in the backend's default buffer
  type (which is what `set_tensor_async` expects);
- the input does not fit in what is left of the current slot;
- the device offers no pinned host buffer type, or no events, or the pinned
  buffer could not be allocated. The ring is then disabled for that backend for
  the rest of the run and every input of it takes the blocking path.

The whole path is disabled for a scheduler created with more than one copy, i.e.
with pipeline parallelism, where the input copies are already double buffered and
guarded by events.

Teardown is the one place where the ordering is an assumption rather than a
check. `ggml_backend_sched_free()` frees the slot events and the pinned buffer
without waiting for uploads that may still be in flight, because a caller such as
`llama_context` frees its backends before the scheduler and waiting there would
touch a stream whose backend no longer exists. As with the copy events the
scheduler frees in the same loop, the caller is expected to have synchronized the
backends before it frees the scheduler.

### How to verify it

- For an equivalence check, run the same prompt with `LLAMA_INPUT_UPLOAD_ASYNC=1`
  and `=0`: the path is a pure transport change and the logits must not move.


## LoRA scale folding

### What it is

The LoRA delta of a target is `scale * B * (A * x)`, and the graph builder used
to express the `scale` factor as a `GGML_OP_SCALE` node between the `B` matmul
and the `ggml_add` that folds the delta into the base output. That node is now
emitted only when the effective scale is not exactly 1; where it is not 1, the
scale is folded into a pre-scaled copy of the dense `B` matrices that is built
when the adapter is attached to a context.

The effective scale of a target is `alpha/rank` times the adapter scale the
caller passed, so an adapter with `alpha == rank` used at scale 1 has an
effective scale of exactly 1 and needs no copy at all.

### When it applies

The scale node costs a kernel launch per target and per token, which is
significant for a small adapter attached to many targets during single-token
decoding. It also sits between the `B` matmul and the add, so the CUDA/HIP
backend's `mul_mat` + `add` fusion does not see the two nodes as adjacent and
cannot fuse them. Removing the node gives back both.

### How it works

- `llama_adapter_lora::ensure_scaled_b()` is called from
  `llama_context::set_adapters_lora()`, i.e. at a point where no graph exists.
- It considers the dense weights only (`ne[2] == 1`), contiguous, F16 or F32, and
  only those whose effective scale differs from 1. For each of them it allocates
  a copy in the same buffer type as the original `B` - so a `B` that lives in
  host memory keeps living in host memory - and fills it by reading, scaling on
  the host and writing back.
- The copies are built at most once per adapter, for the first adapter scale that
  needs them. The adapter records that scale; a later attach with a different
  scale keeps the scale node instead of rewriting the copies, which is what makes
  it safe for a graph to point at them and for several contexts to attach the
  same adapter one after another.
- `build_lora_mm()` uses the pre-scaled copy when it exists and was built for the
  scale of this attachment, and emits the scale node otherwise. It drops the node
  outright when the effective scale is 1.
- `build_lora_mm_id()` (routed targets) only drops a scale of exactly 1. A routed
  target holds one `B` per expert and is applied with `ggml_mul_mat_id`, whose
  result the backend cannot fuse with the following add anyway, so no copy is
  kept for it.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `LLAMA_LORA_FOLD_SCALE` | env | `1` | `0` restores the previous graph exactly: every target keeps its scale node and no copies are built. Read once per process. |
| `--lora <fname>` | llama-bench | - | Apply a LoRA adapter with scale 1.0 to every context the tool creates. Repeatable. |
| `--lora-scaled <fname> <scale>` | llama-bench | - | The same with an explicit adapter scale. Repeatable. |

The adapters loaded by llama-bench belong to the loaded model and are freed with
it when the tool moves on to the next model.

### Limits

- Only dense targets get a pre-scaled copy, and only F16 and F32 ones. Anything
  else keeps the scale node unless its effective scale is 1.
- The copies cost memory: one copy of every dense `B` of the adapter, in the
  buffer type of the original. For a low-rank adapter that is small, but it grows
  with the rank and the number of targets.
- One scale per adapter. If the same adapter is attached with another scale
  later, that attachment runs with the scale node.
- If a context or a buffer for the copies cannot be allocated, the adapter stays
  in the "not built" state and the graph keeps the scale node, so a later attach
  can try again. Nothing fails.
- `ensure_scaled_b()` mutates state that belongs to the adapter, not to the
  context, so the adapter must not be attached from two contexts at the same
  time.

### How to verify it

- `test-backend-ops -o MUL_MAT_VEC_FUSION` includes rank-2 F16 `B` shapes: one
  token, which is the case the backend fuses, and two and four tokens, which keep
  the same matmul but take the unfused path.
- Run the same prompt with `LLAMA_LORA_FOLD_SCALE=1` and `=0` and compare the
  logits; the pre-scaled copy is a different rounding of the same product only
  where the copy is F16, so an adapter with an effective scale of 1 must match
  exactly.
- A kernel trace with an adapter attached shows the scale launches disappear, and
  for a single-token decode the `B` matmul and the add appear as one kernel.


## LoRA on the GLM5-Next attention output

`glm5-next` multiplied its KDA and MLA attention output projections (`attn_output`) with `ggml_mul_mat`, so
an adapter that targets `attn_output`, such as the GLM-5.3 heretic adapter, was loaded but not applied to
them. They now go through `build_lora_mm` like the other projections, and the adapter changes the output.
Without an adapter `build_lora_mm` adds no node here (the precision policy applies only to tensors listed in
the GGUF precision metadata, the BF16 accumulation only to NVFP4 weights), so the graph is unchanged.


## One HIP graph per batch shape

### What it is

The CUDA/HIP backend keeps the graphs it captured per context and used to key
them by the first node of the ggml graph. llama builds the graph of another
batch width in the same metadata buffer, so every width shared one stored
graph: a speculative verification whose width changes between rounds ran each
changed call without a graph and captured again on the next call. With
per-shape keying the key also holds a shape signature - the node count and the
shape and op of the first and the last node - so a width that returns launches
its instantiated graph directly.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_GRAPH_PER_SHAPE` | env | `1` in HIP builds, `0` in CUDA builds | `0` keys the stored graphs by the first node only, the upstream keying; `1` keys them by first node and batch shape with at most 32 graphs per context; `N > 1` allows at most `N` (capped at 256). Read once per process. |
| `GGML_CUDA_GRAPH_EVICT_SECONDS` | env | `60` | A per-shape graph that was not used for this many seconds is dropped by the sweep that runs every 5 s; `0` drops per-shape graphs only at the per-context bound. Graphs keyed by the first node only (per-shape keying off) keep the upstream 10 s, since nothing else bounds their number. Read once per process. |
| `GGML_CUDA_GRAPH_QUICK_CAPTURE` | env | `1` | `1` captures a changed graph at once, without the direct warmup call, when a graph of the same family was captured and replayed in the same context before (see below). Needs the per-shape keying, so it only acts where that is on (HIP builds by default). `0` always warms a changed graph up with a direct call. Read once per process. |
| `GGML_CUDA_GRAPH_DEFER_FREE` | env | `1` | `1` destroys a graph dropped at the per-context bound after a later graph compute of its context, once an event recorded at the drop shows that the work queued before it has finished; graphs dropped unused by the sweep are destroyed there too. At most one graph is destroyed per compute. `0` destroys them where they are dropped, at the bound after a synchronization of the stream. Read once per process. |

The default bound was 16 until the llama graph per batch shape (next section)
was turned on by default. Each llama graph it keeps is a graph of its own for
the backend, so with 24 stored llama graphs and 16 backend graphs the backend
dropped and captured graphs again at its bound; 32 leaves room for them.

### Limits

- The graphs of one context are bounded. When the bound is reached, the least
  recently used graph is dropped. Its instance may still be queued, so it is
  destroyed after a later graph compute of the context, once an event recorded
  at the drop has completed (`GGML_CUDA_GRAPH_DEFER_FREE=1`). The drop happens
  while the scheduler splits the new graph, because the graph optimize step
  looks the graph up; destroying it there after a stream synchronization
  (`GGML_CUDA_GRAPH_DEFER_FREE=0`, the previous behaviour) made the split of
  that call wait and pay the destruction before the new graph was submitted.
- Destroying a graph takes host time: about 3-4 ms for a verification graph of
  Qwen3.8 Flash Next on HIP, under 0.1 ms for its drafter graph. After the
  compute the host waits for the GPU anyway, so at most one graph is destroyed
  per compute; a sweep that drops several graphs at once is spread over the
  following computes.
- A graph captured before the memory pool returned memory to the driver is
  captured again.
- An unused graph used to be dropped after 10 s. A speculative verification
  uses its rarer widths less often than that, so each such use ran the graph
  directly and the next one captured it again. The per-shape graphs are bounded
  by the per-context limit, so they are now kept for 60 s by default; a longer
  time keeps more instances alive, at most the bound (host and driver memory of
  the executable graphs; they have no compute buffers of their own).

A graph whose properties changed is normally run directly once and captured
on the next call that finds it unchanged. When the KV view of a verification
width grows by a padding step, llama builds the graph of that width again, so
each width paid a direct call and then a capture after every step. With
`GGML_CUDA_GRAPH_QUICK_CAPTURE=1` such a graph is captured on the call that
changed, if its family is ready:

- The family is the shape signature of the key (which holds the width) and the
  op and type of every node, so graphs of one family launch the same kernels in
  the same order. It is computed only when a graph is captured or a changed
  graph is about to be warmed up.
- A family is ready once one of its graphs was captured and then replayed in
  the same backend context, and no capture of the family issued a BLAS call (a
  BLAS library may prepare kernels or workspaces on the first call of a new
  problem size).
- A capture that issues BLAS calls marks its family as a BLAS family for the
  rest of the context: the family is no longer ready and never becomes ready
  again, so every later change of its graphs keeps the warmup. Whether a
  capture issues BLAS calls is only known after it, so when a family that was
  ready without BLAS calls (for example below a KV size at which a matrix
  multiplication goes to BLAS) first issues them, that one capture is still
  taken at once.
- A width used for the first time, a context whose memory pool released memory
  since the ready capture, and a family whose last quick capture changed again
  before it was replayed keep the direct warmup call. The last case makes the
  family wait for a regular warmup and replay again, so a graph that changes on
  every call is not captured on every call.
- The kernels and their order are the same; only the call that captures moves.
  The capture and the instantiation are still paid.
- Measured on Qwen3.8 Flash Next with an MTP drafter, with the series released
  as ranma_20261001, whose Qwen indexer built no selection while its budget
  covered every cell (the current one builds it at every length): the replies
  of a fixed width 2 verification were identical with the switch on and off.
  Of the graphs rebuilt at a padding step about two thirds were captured at
  once; the rest belong to the graphs of the sparse attention selection above
  2048 cells, whose captures issue BLAS calls and keep the warmup. The host
  time of the backend graph management (compare, direct runs, capture,
  instantiate) of a mixed run fell by about 20 %. These numbers predate the
  BLAS family marking: before it, one verification family that became ready
  below 2048 cells was still captured at once about 15 times per run after its
  captures began to issue BLAS calls, with identical replies. The marking
  sends those back to the warmup (about 5 ms each).

### How to verify it

- The backend logs `per-shape graphs on, at most N per context` once per
  process when the keying is on, and `per-shape graphs off` when the variable
  turns it off.
- Measured on DeepSeek V4 Flash with a DSpark drafter (verification widths 1 to
  4): the extra time of the second call after a width change dropped from 9..13
  ms to 0..3 ms per round, and the replies were identical with the switch on and
  off.
- The backend context logs `per-shape graphs: ... evicted at the limit` when it
  is destroyed; a count that grows with the run means the bound is too small for
  the widths in use. The same line counts the graphs `dropped unused` by the
  sweep. With the quick capture a second line counts the changed graphs
  `captured without a direct call` and those that changed again before a
  replay.


## One llama graph per batch shape

### What it is

llama keeps one built graph per output class and builds, splits and allocates a
new one whenever the batch shape changes. A speculative verification changes
its width from round to round (1 to n-max + 1 tokens), so the call after every
width change paid the build, the scheduler split and the allocation again, and
the HIP backend compared the properties of the whole graph against the one it
had stored. With the reuse on, a context keeps the built graph of each batch
shape in a graph slot of the scheduler, together with its split and its
allocation. A width seen before is made current again: no build, no split, no
allocation, and the backend finds the same split and replays its stored graph.

A stored graph stays valid until the compute buffers are planned again (a
reserve, or a buffer that has to grow). Then the slots are dropped and each
shape is built again the next time it is used. When all slots are in use, the
least recently used shape is evicted.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `LLAMA_GRAPH_REUSE_SHAPES` | env | `24` when not set | `0` keeps one graph per output class, the behaviour before this change; `1` keeps up to 8 batch shapes per context; `N > 1` keeps up to `N` (capped at 64). Read once per process. |

24 covers the up to 16 verification widths of a speculative server and leaves
room for a second graph variant of a width (DeepSeek V4 Flash alternates two
graph plans at some widths).

### Limits

- It switches itself off, and logs the reason, when graph reuse is disabled
  (`LLAMA_GRAPH_REUSE_DISABLE`), for a model loaded with `no_alloc`, with
  `GGML_CUDA_GRAPH_OPT=1` (the backend keeps the stream plan of the last
  optimized graph only) and with pipeline parallelism (more than one scheduler
  copy).
- Host memory: every slot keeps the split and the graph copy of its shape. In
  the server rows below the peak private bytes rose by 0.85 to 0.97 GiB with
  both defaults (24 llama graphs, 32 backend graphs); VRAM did not change,
  since all slots share the compute buffers.
- The graphs are still rebuilt when the compute buffers are planned again, for
  example when the KV cache view of a graph crosses a padding step of 256
  cells. That cost is paid once per step and per width in use, not per width
  change.

### How to verify it

- The context logs `graph reuse across batch shapes on, up to N graphs` at
  every reserve, or `off` with the reason. When it is destroyed it logs how many
  graphs were built, made current again, rebuilt after a buffer change and
  evicted.
- Measured on 2026-09-29 (the build
  also carried a per-round speculative log, used for the switch cost below, and
  diagnostics that were off), Radeon AI PRO R9700, 128 GB host memory, English roleplay (17
  requests), MTP `--spec-draft-n-max 7 --spec-draft-p-min 0.7` with the joint
  cache, expert cache exclusive 20480 MiB and an unlimited host tier, warm
  profile, one run per row with a 300 s rest. A = `LLAMA_GRAPH_REUSE_SHAPES=0`
  and `GGML_CUDA_GRAPH_PER_SHAPE=16` (the previous defaults), B = 24 and 32
  (the defaults now):

  | model | A decode t/s | B decode t/s | B against A | replies | acceptance |
  | --- | ---: | ---: | ---: | --- | ---: |
  | Qwen3.8-Flash-Next UD-Q4_K_XL (prefill swap on) | 46.52 | 51.67 | +11.1 % | 17/17 identical | 77.9 % both |
  | DeepSeek V4 Flash UD-IQ3_XXS | 28.97 | 31.45 | +8.6 % | 17/17 identical | 68.8 % both |

  The extra time of the call after a width change and of the call after it
  (from the per-round log) was +3.6 to +9.8 ms in A and about 0 (within
  +-0.8 ms) in B.
