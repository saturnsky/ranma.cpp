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
| `GGML_CUDA_MMVQ_SHARE_Q8_STATS` | env | `0` | `1` logs the total and the shared quantize calls of the first evaluated graph. |

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

- `GGML_CUDA_MMVQ_SHARE_Q8_STATS=1` prints the total and shared quantize call
  counts of the first graph; the difference is the number of launches removed.
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


## Folding the dense shared expert into the routed launch

### What it is

In a layer whose routed launch already alternates its experts, the dense shared
expert of the same layer is computed as one more unit of that grid instead of by
launches of its own: the shared gate and up matrices ride along with the routed
gate/up launch and produce the shared GLU, the shared down matrix rides along
with the routed down launch. The launches the shared expert would need - the
fused gate/up matmul, the down matmul and their q8_1 quantizations - disappear.

### When it applies

A routed launch that reads at least one expert over the link is bound by that
read, and the blocks of the resident experts run inside the wait. The shared
expert is small compared to that wait, so its rows can be computed inside it for
close to nothing. The unit has to alternate with the routed blocks to get that:
appended behind them or placed in front of them, a synthetic kernel shows only
part of the cost being hidden.

The fold is tried per layer, and only where the graph has exactly the shape the
unit expects.

### How it works

**The unit.** Which shared type a routed type can carry is a property of the
routed type alone, so the matcher derives it and no dispatch or launch argument
is needed. A model that pairs them differently does not fold.

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

**The match.** A host-side matcher recognizes the layer structurally, without
looking at any address, so the same matcher can run in the graph-optimize pass
before the tensors are allocated:

- two routed `MUL_MAT_ID` matmuls of one token and more than one expert,
  followed by their GLU;
- a shared GLU within a window of 64 nodes behind them, whose two `MUL_MAT`
  nodes are the two nodes directly in front of it, are read by nothing but the
  GLU, and are not graph outputs;
- shared gate/up matrices of the shared type that belongs to the routed type, in
  this device's memory, of the same shape and stride as each other and of the
  routed shape;
- a shared input that is the input of the routed launch seen through reshapes;
- shared results that are plain contiguous F32 row vectors.

The shared down matmul is folded in addition when it reads the shared GLU, fits
the same rules, and a routed down launch of the same layer follows it. If it does
not, the gate/up half is still folded on its own.

**Writing early.** A folded result is written at the routed launch, which comes
before the node that would have produced it, so those bytes must belong to it
from that point on. `ggml_backend_cuda_graph_optimize()` asks the allocator for
that by adding an allocation dependency that pulls both results in front of the
routed launch. The evaluation then verifies the addresses it actually got: a fold
is refused when the target bytes touch any tensor read or written by a node of
the window, or when the input of the shared expert is written in between. An
address that cannot be compared counts as a conflict, which is the safe
direction.

**The handshake.** For each candidate node, the evaluation loop offers the unit
to the launch that is about to run; the launch takes it only if it really is the
alternating routed launch of a type that carries a unit, and reports back. A
taken unit marks the folded nodes as computed, so the node loop skips them before
any fusion matching, and tells the q8_1 sharing plan about the matmuls that never
run so that their groups keep moving. A layer that does not fold runs exactly as
before, including the fusion of the shared down matmul with the final add of the
layer.

The decision is made in the node loop, which runs on a plain evaluation and on a
CUDA-graph capture but not on a replay, and it depends only on the graph and on
the addresses the allocator handed out - both fixed for a captured graph - so a
capture and each of its replays compute the same thing. The offered unit and the
bookkeeping of a graph's fold decisions live in the backend context that
evaluates the graph.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `GGML_CUDA_MMVQ_ID_FOLD_SHARED` | env | `1` | `0` disables the fold; the shared expert keeps its own launches and the evaluation is the previous one. Reference path for equivalence checks, and the fallback for a model shape that should not fold. |
| `GGML_CUDA_MMVQ_ID_FOLD_SHARED_LOG` | env | `0` | `1` reports, for the first three evaluations, how many gate/up and down units were folded, how many candidates were rejected and why. |

### Limits

- HIP with the RDNA4 parameter table only, and only for the one-token kernel of
  a routed type that has a shared type in the table above. Every pair is another
  compiled kernel instance, which is why the lists are short. The small-k and
  halved-iteration variants of a routed kernel carry no unit; the
  alternating-rows variant does, because the mirrored unit follows the rows per
  block of the launch it rides in.
- Only when the expert cache supplies the address table and the grid alternates
  the experts; without the wait there is nothing to hide the shared rows in.
- Only SwiGLU and clamped SwiGLU shared activations, without swapped operands.
- A layer whose shared matrices have another type, another shape or another
  buffer is simply not folded, and so is a layer whose results do not get
  private memory. Rejections are per layer; the rest of the model still folds.
- The rejection reasons are counted in fixed-size arrays of the fold state
  whether or not the log switch is on. Only the report itself is built under the
  switch.

### How to verify it

- `GGML_CUDA_MMVQ_ID_FOLD_SHARED_LOG=1` prints the fold counts per evaluation:
  on a model that folds, the gate/up count should be the number of layers whose
  shared matrices match, and the down count the same minus the layers without a
  routed down launch to carry it.
- Compare logits or a greedy continuation against a run with
  `GGML_CUDA_MMVQ_ID_FOLD_SHARED=0`. The unit computes the same sums in the same
  order as a standalone launch, so the result is expected to be unchanged.
- A kernel trace of one token should show the shared expert's own matmul and
  quantize launches gone for every folded layer.
