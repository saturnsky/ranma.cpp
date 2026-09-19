# Graph runtime

Changes to the machinery that turns a graph into work for a backend: the compute
buffers the graph allocator keeps, the way the scheduler gets the inputs of a
graph onto the device, and the nodes a LoRA adapter adds to a graph.

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
| `LLAMA_DECODE_HOST_TIMING` | env | `0` | `N > 0` logs one line of host-side phase timing per `N` single-token decodes (see below). |
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

- The scheduler counts its uploads. `ggml_backend_sched_get_input_upload_stats()`
  (declared in `ggml/include/ggml-backend.h`) returns the cumulative number of
  asynchronous and of blocking uploads, the byte total and the accumulated upload
  time; any output pointer may be `NULL`. The counts and the bytes are always
  accumulated. The time is only measured while
  `ggml_backend_sched_set_input_upload_timing(sched, true)` is in force, so that
  a normal run does not pay for timestamps.
- `LLAMA_DECODE_HOST_TIMING=N` prints, every `N` single-token decodes, the
  averages over those decodes in microseconds per token: the whole decode, then
  the batch, memory, graph build, input setting, compute, output copy and
  synchronization phases, with the upload time, the asynchronous and blocking
  upload counts and the uploaded KiB reported inside the compute phase. On a
  model with a busy input set, the upload count should be the input count with
  zero blocking uploads, and the upload time should be a small fraction of what
  `LLAMA_INPUT_UPLOAD_ASYNC=0` shows.
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
