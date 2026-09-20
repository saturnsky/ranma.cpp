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
