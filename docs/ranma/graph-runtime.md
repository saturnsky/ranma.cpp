# Graph runtime

Changes to the machinery that turns a graph into work for a backend: the compute
buffers the graph allocator keeps.

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
