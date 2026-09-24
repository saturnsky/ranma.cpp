# MoE decode on the HIP backend

Changes to the kernels and to the host-side graph passes that drive one-token
decoding of a mixture-of-experts model on RDNA4.

## Shared q8_1 quantization of an MMVQ input

### What it is

An MMVQ (`mul_mat_vec_q`) call quantizes its `src1` to q8_1 before it runs. When
several `GGML_OP_MUL_MAT` nodes of one graph read the same activation tensor,
each of them used to allocate a buffer and launch `quantize_q8_1` over identical
bytes. A pass over the graph now groups such nodes: the first node of a group
quantizes and keeps its buffer, the rest of the group reuse it.

### When it applies

It is not specific to MoE, but that is where it pays. The attention block of a
model that projects one normed activation into several matrices - a query
projection, a key/value projection, a compressor - issues one MMVQ per
projection, and at one token those launches are short enough that their
quantization is a visible share of the layer. Prompt-sized inputs are excluded:
above `MMVQ_MAX_BATCH_SIZE` columns the backend takes MMQ instead, so the pass
never looks at the prompt graphs.

### How it works

The pass runs once per graph evaluation, before the node loop:

- A node is a candidate when it is a `GGML_OP_MUL_MAT` that is going to be
  computed, whose `src1` is contiguous F32 with at most `MMVQ_MAX_BATCH_SIZE`
  columns.
- Candidates are grouped by identical `src1` bytes: same data pointer, same type,
  same shape and same strides.
- A group ends at the first node in between that writes into the byte range of
  that tensor. The range is computed from the strides, so a strided view is
  covered as well, and any node that is not a view or a no-op is considered a
  writer of its own destination.
- Each member is marked `reuse` (an earlier member has quantized these bytes) and
  `hold` (a later member still needs the buffer). The first member allocates and
  holds; the last member hands the buffer back to the pool before it allocates
  anything else of its own, because the pool frees strictly in reverse allocation
  order.
- At run time an MMVQ call finds its entry through a cursor that walks the plan
  in graph order with a bounded forward search, so candidates that never reach
  the MMVQ path only cost a few comparisons. No match means no sharing, which is
  always safe: the call quantizes for itself.
- Before the shared buffer is used, the call checks it against the tensor it has
  in hand (pointer, size, shape and strides); a mismatch falls back to its own
  quantization.

The plan lives in the backend context that evaluates the graph, so the pool
allocation it holds and the device the kernels run on always belong together. It
is rebuilt from the graph whenever the nodes are really evaluated - which
includes a CUDA-graph capture and excludes a replay - so a captured graph
produces the same sequence of pool allocations every time it is captured.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_MMVQ_SHARE_Q8` | env | `1` | `0` restores one quantize call per `MUL_MAT`. Reference path for equivalence checks. |

### Limits

- The pass is disabled for a graph evaluation that launches concurrent events:
  the members of a group would then run on different streams, and a pool buffer
  is only ordered within one stream.
- Groups never cross a write into the input, and a group of one member shares
  nothing.
- `MUL_MAT_ID` calls do not take part; they look up no plan entry.
- The forward search that matches a call to its plan entry covers a window of
  plan entries. A graph in which more than that many candidates in a row skip the
  MMVQ path simply stops sharing from there on.

### How to verify it

- `test-backend-ops -o MUL_MAT` covers the path, and a run with
  `GGML_CUDA_MMVQ_SHARE_Q8=0` is the reference for comparing logits.


## Expert-first launch grid for MUL_MAT_ID

### What it is

The one launch that computes the routed experts of a `MUL_MAT_ID` node uses the
grid `(expert, row block)` instead of `(row block, expert)`, so that consecutive
blocks of the dispatch belong to different experts.

### When it applies

With the expert cache, the experts a token routes to are read partly from VRAM
and partly from mapped host memory, inside one launch. A device works a dispatch
through in block order, and one expert already holds enough blocks to fill the
device, so with the expert-major grid the blocks of an expert that waits on its
host reads stand in front of the blocks of the experts that are resident. The
launch then takes the sum of the two parts. With the experts alternating in
dispatch order, the resident blocks flow through the execution slots the waiting
blocks leave free, and the launch takes about the longer of the two parts.

### How it works

The two grid dimensions are swapped at launch time and the kernel is told about
it through a flag in its fusion arguments: it reads the row block from the
dimension the swap moved it to and the channel from the other. Every thread
computes exactly what it computed before - only the mapping from a block index to
an (expert, row block) pair changes - so the result is bit-identical.

The swap is applied when all of the following hold:

- the build is HIP;
- the launch has an `ids` tensor and more than one channel, i.e. it is a routed
  launch;
- the expert cache supplied the address table the launch reads its weights
  through;
- the row-block count fits the 65535 limit of the grid dimension it moves to,
  which is checked before the swap.

Nothing restricts it to a single token: a multi-token `MUL_MAT_ID` launch that
meets these conditions gets the alternating grid as well. It is the one-token
launch of a decode that is bound by the host reads, so that is where the
difference shows.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_MMVQ_ID_EXPERTS_FIRST` | env | `1` | `0` keeps the expert-major grid. Reference path for equivalence checks. |

### Limits

- HIP only; other backends are not compiled with the swap.
- Without the expert cache there is no address table, so the grid is unchanged:
  a launch whose weights are all in VRAM has nothing to overlap.

### How to verify it

The output must be bit-identical with the switch on and off - the same summands
in the same order, only dispatched differently - so a greedy continuation with
`GGML_CUDA_MMVQ_ID_EXPERTS_FIRST=0` and `=1` should reproduce token for token.


## Shared expert unit in the routed launch

### What it is

Upstream fuses the dense shared expert of a layer into the one-token routed
gate/up launch when the shared matrices have the routed type: the launch gets
one more channel, and that channel runs the routed code on the shared matrices
(`ggml_cuda_match_shared_expert()`, upstream pull request #29184). This change
extends the same channel to the cases upstream does not cover:

- a shared expert whose type differs from the routed type, which is how the
  usual mixed quantizations store it (routed IQ2/IQ3 with a Q6_K shared expert,
  routed Q4_K/Q5_K/Q5_1 with a Q8_0 shared expert);
- the shared down matrix, which rides in the routed down launch of the layer.

Such a channel is computed by a **shared unit** of the one-token kernel instead
of the routed code. The launches the shared expert would need of its own - the
fused gate/up matmul, the down matmul and their q8_1 quantizations - disappear.

### When it applies

Every one-token decode step of a model whose shared expert pairs with its
routed experts as in the table below. With the expert cache the routed launch
alternates its experts (previous section), and the shared channel is the last
expert of that alternation, so its rows run inside the wait for the experts read
over the link; that is where it pays most. Without the cache it still removes
the launches of the shared expert.

### How it works

**The unit.** Which shared type a routed type can carry is a property of the
routed type alone (`ggml_cuda_mmvq_shared_unit_type()`); every pair is another
compiled kernel instance, which is why the list is short.

| Routed type | Shared type | Layout of the unit |
| --- | --- | --- |
| IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | Q6_K | replayed |
| Q4_K, Q5_K, Q5_1, Q8_0 | Q8_0 | mirrored |

The two layouts follow from the RDNA4 parameter table. Where the routed kernel
runs one warp and one row per block and the standalone shared kernel runs eight
warps and four rows, the unit **replays**: a row of the result does not depend on
how the rows are spread over blocks, so one lane walks all eight warp strides,
keeps one partial sum per warp, adds them in warp order and finishes with the
same lane reduction - the same summands in the same order, hence the same rows.
Where both types run the same number of warps and the rows per block follow the
row count, which is the same for both, the unit **mirrors** the standalone
one-column kernel instead: every thread of a shared block does what the same
thread of a standalone launch does, including the reduction over the warps
through shared memory, the clamp of the last block and the rule that lane i
writes row i. The kernel picks the layout at compile time from the warp and row
counts of the two types, and refuses to compile a unit for which neither fits.

The gate/up unit reads the q8_1 input the routed launch already made (it is the
same tensor and the same bytes); the down unit quantizes the shared GLU result
exactly as a launch of its own would. A gate unit only rides in the fused variant
of the routed kernel, which is the one that has the shared memory for it.

**The match.** The gate/up pair is upstream's pattern
(`ggml_cuda_match_shared_expert()`): the routed `MUL_MAT_ID` gate and up and
their GLU, and a shared `MUL_MAT` gate and up and their GLU behind them, which
the graph-optimize pass moves right behind the routed nodes. The extension only
widens the type check to the pairs of the table (one token, shared matrices of
the routed shape in this device's memory, SwiGLU or clamped SwiGLU without
swapped operands) and also accepts a shared input that is the routed input seen
through another chain of views.

The down pair is a pattern of its own (`ggml_cuda_match_shared_expert_down()`):
a one-token routed `MUL_MAT_ID` whose input is a GLU result, and a shared
`MUL_MAT` of the routed shape and a pair type whose input is the shared GLU
result, computed before the routed launch. The graph-optimize pass moves the
shared down matmul right behind the routed down launch and keeps the routed
input and ids allocated until the shared result is placed, so that the launch
can write it while it still reads them; at evaluation the usual fusion checks
(`ggml_can_fuse_subgraph_ext()`, `ggml_cuda_check_fusion_memory_ranges()`)
decide.

**The launch.** The fusion arguments carry the shared matrices
(`shared_up`, `shared_gate`, null for the down matrix), the shared result and,
for the down matrix, the shared input. The launch adds one channel to its grid
as upstream does; a channel of another type or without a gate is flagged as a
shared unit, gets its row stride and its q8_1 input, and runs in the kernel
instance of the routed type that has the unit compiled in. The matmuls the unit
takes over are reported to the q8_1 sharing plan of the graph, so that their
group keeps moving; the same is done for upstream's same-type channel.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_MMVQ_ID_FOLD_SHARED` | env | `1` | `0` disables the shared unit: a shared expert of another type, and every shared down matrix, keep their own launches. Upstream's same-type channel is not affected (`GGML_CUDA_DISABLE_FUSION=1` turns off all fusions). Reference path for equivalence checks. |

### Limits

- HIP with the RDNA4 parameter table only, and only for the one-token kernel of
  a routed type that has a shared type in the table above. The small-k and
  halved-iteration variants of a routed kernel carry no unit; the
  alternating-rows variant does, because the mirrored unit follows the rows per
  block of the launch it rides in.
- A model that pairs its types differently (for example routed IQ3_XXS with a
  Q8_0 shared expert) keeps the shared launches; so does a launch with more than
  one token, where only upstream's same-type channel applies.
- Only SwiGLU and clamped SwiGLU shared activations, without swapped operands.

### How to verify it

- Compare logits or a greedy continuation against a run with
  `GGML_CUDA_MMVQ_ID_FOLD_SHARED=0`. The unit computes the same sums in the same
  order as a standalone launch, so the result is expected to be unchanged.
- `test-backend-ops -o MUL_MAT_ID_SHARED_UNIT` compares the gate/up and the down
  pairs of the table against the CPU backend, at one token (unit) and two (no
  unit).
- With `--log-verbosity 5` (debug) the graph-optimize pass logs how many routed gate/up
  and down launches got a shared expert matched.


## Reading each distinct expert once in a multi-token launch

### What it is

A `MUL_MAT_ID` launch with two to four tokens - a speculative-decoding
verification batch, or several server slots decoding together - stays on MMVQ
below the MMQ crossover of the weight type. `mul_mat_vec_q_moe` gives every
(token, expert slot) pair its own warp, so tokens that route to the same expert
read that expert's weights once per token. With the expert read from host
memory, every repeated read is another transfer over the PCIe link.

### When it applies

- HIP builds, `MUL_MAT_ID` through MMVQ with two or more tokens.
- By default only when the weights are read from host memory: host-direct
  mapped weights or the expert cache. Weights in VRAM keep `mul_mat_vec_q_moe`,
  whose repeated reads are served by the GPU caches.
- The MMVQ/MMQ crossover is unchanged; five or more tokens still take MMQ for
  the types that switch there.

### How it works

The pairs are grouped by expert on the device, from the ids alone, so there is
no host readback and a captured graph stays valid. The first pair of a group
loads each weight block once and dots it with the inputs of every pair of the
group; the other pairs of the group exit. Each output keeps the lane layout, the
K order, the warp reduction and the epilogue of `mul_mat_vec_q_moe`, so the
result is bit-identical. The fused gate/up/GLU, biases and scale tensors are
supported, and with the expert cache the grid alternates host and VRAM experts
as the one-token launch does.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_MMVQ_ID_DEDUP` | env | `1` | `1` uses the grouped kernel for weights read from host memory, `2` also for weights in VRAM, `0` keeps `mul_mat_vec_q_moe`. |

### Limits

- HIP only.
- In VRAM the grouped kernel pays off only when the tokens share experts; with
  mostly distinct experts it can be slower, which is why `2` is not the default.
- A fused gate launch with five or more tokens keeps the old kernel.

### How to verify it

`llama-perplexity -b B -ub B` for B = 2..4 with `GGML_CUDA_MMVQ_ID_DEDUP=0` and
unset must print identical values; `test-backend-ops -o MUL_MAT_ID` covers the
routing cases (same, disjoint, partial and repeated experts).
