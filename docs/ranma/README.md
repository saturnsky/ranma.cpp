# Changes over upstream

Each user-visible change gets a line here and a page under `docs/ranma/` that describes what it is, when it
applies, how to switch it, and its limits. The main features are summarized in the [README](../../README.md#main-features);
measured effects are collected in [benchmarks](benchmarks/README.md), and starting settings by model and hardware are in
[recipes.md](recipes.md).

### RDNA4 kernels (HIP)

- **Small-batch matmul dispatch** - four weight rows per MMVQ block for 3..8 activation columns and a
  per-type MMVQ/MMQ crossover, so the cost of a decode call no longer rises and then falls with the number of rows
  in it. This is the range of a speculative verification step and of a server that batches a few slots.
  [rdna4-small-batch.md](rdna4-small-batch.md)
- **Four rows per block at one column** - single-token decode of a wide matrix reads the activation once per
  four weight rows; chosen per call from the matrix size. Same page.
- **12-column tile attention for GQA-12 groups** - token generation on a model whose head group is a multiple
  of twelve reads each K/V head once instead of three times. Same page.
- **WMMA attention for 512-wide heads** - prompt processing of a 512/512 head at a GQA ratio that is a multiple of 8 with an F16 KV cache
  takes the wide-tile MMA kernel.
  [rdna4-prefill.md](rdna4-prefill.md)
- **Padded F16 BLAS for wide dense Q2_K/Q6_K/IQ2 matmuls** - wide prompt matmuls of those types convert both
  operands to F16 with a padded row pitch and run hipBLASLt instead of MMQ. Needs `ROCBLAS_USE_HIPBLASLT=1`.
  Same page.
- **Fixed-order kernels for skinny F32 matmuls** - F32 products of up to 512 weight rows with more than eight
  activation columns (router, SSM and hyper-connection projections of a prompt batch) run on dedicated kernels
  instead of hipBLAS, whose per-process solution choice changed their speed and summation order from one process
  to the next. Same page.
- **Wider MoE column tiles** - the MMQ tile width of a `MUL_MAT_ID` is sized against three times the mean
  column count per expert, so popular experts re-read their weights less often. Bit-identical. Same page.
- **Short f16 mat-vec rows** - one-column f16 mat-vecs with rows of 256 or 512 elements (the per-head
  projections of absorbed MLA attention) read whole rows with 16-byte loads. Bit-identical.
  [rdna4-small-batch.md](rdna4-small-batch.md#short-f16-mat-vec-rows)

### Server and common tools

- **GPU heartbeat (llama-server)** - `--gpu-heartbeat-seconds 5` records one GPU event per interval while the
  server is idle and while the model is freed, so that Windows does not evict the VRAM of the process between
  turns. Off by default. [gpu-heartbeat.md](gpu-heartbeat.md)
- **Per-position draft thresholds** - `--spec-draft-p-min` takes one probability per draft position, and
  `--spec-draft-p-continue` keeps a token in the draft but stops drafting after it. Defaults unchanged; with
  `draft-mtp`, giving either turns the smart draft length off.
  [spec-draft-thresholds.md](spec-draft-thresholds.md)
- **Shared compute buffers of a drafting context** - an MTP context or a draft model's context computes in views
  of its target's compute buffers whenever it can (not with pipeline parallelism, nor for a draft model outside
  the target's expert cache). No switch. [expert-cache-ub-pool.md](expert-cache-ub-pool.md)
- **Trigram confidence and profile** - the smart draft length takes the acceptance of a draft token from the
  request's own trigram counts (always on), and with a smart store the
  counts and the learnt coefficients are kept in `<store>.trigram` across requests and restarts.
  [spec-smart.md](spec-smart.md)
- **Smart draft length for draft-mtp (llama-server)** - `draft-mtp` chooses the draft length at every step from
  the measured verification time per width and a calibrated acceptance of the draft probabilities, up to
  `--spec-draft-n-max` (7 when not given); `--spec-smart-store PATH` keeps the estimates across restarts of the same
  model, build and cache settings. On by default with `draft-mtp`; `--no-spec-smart`, or giving
  `--spec-draft-p-min` or `--spec-draft-p-continue`, returns to the single-argument rule.
  [spec-smart.md](spec-smart.md)
- **Slot 0 priority (llama-server)** - with several slots, the prompts of the other slots wait while slot 0 works
  on a request; their generation still shares slot 0's batches (`--slot0-priority strict` holds it too while slot 0
  generates). On by default, `--slot0-priority off` restores the upstream batching. [slot0-priority.md](slot0-priority.md)
- **n-gram drafts of variable length** - `--spec-type ngram-varlen` drafts from the n-gram history of the session,
  verifies long drafts in parts and rolls back as deep as needed; a hazard supplement drafts a long match where nothing
  else drafts, and the verification cost of the target is shared by all drafters. [spec-smart.md](spec-smart.md)
- **Layer compute mode** - `--layer-mode on` computes a prompt step as one wide ubatch layer by layer (routed experts
  expert by expert, attention in chunks) and sizes the width, the attention chunk and the staging buffers from the
  model and the free memory. GLM-5.3, Qwen3.8 and DeepSeek V4 / V4.1 Flash with EXL3 experts and the expert cache;
  off by default. For DeepSeek V4 and V4.1 it also runs the prompt layer-major, reading each expert once per group of
  ubatches. [layer-mode.md](layer-mode.md)
- **Reuse of a just-restored context checkpoint (llama-server)** - when the first prompt batch after a checkpoint
  restore starts at that checkpoint, the server keeps the restored entry instead of serializing the unchanged
  state again.
  [server-checkpoints.md](server-checkpoints.md)
- **Per-layer embedding prefetch** - `--ple-prefetch {off,prefill,always}` (default `always`) hands the rows of
  a lazily mapped per-layer embedding table to the operating system before the gather, and the gather of that
  tensor runs on the threadpool. [ple-prefetch.md](ple-prefetch.md)
- **Only the shards with lazy tensors are mapped** - with mmap loading off, the loader no longer maps model
  files that nothing reads through the mapping. Same page.
- **Engram tables as a GGUF set of their own** - a split conversion of DeepSeek V4.1 (engram) or Qwen3.8 (PLE)
  writes the tables as their own set `<model>-engram…`, which the model index lists by table name, rows and row width;
  one set serves every quantization (and an F16 or EXL3 variant of the tables fits too). The loader takes the one
  `<model>-engram*.gguf` next to the model or the set given by `--engram-tables PATH`, and `--engram-tables` also
  replaces the tables that a model holds itself.
  [engram-tables.md](engram-tables.md)

### Host-resident MoE experts

- **Host-direct MoE weights (HIP)** - `MUL_MAT_ID` kernels read host-resident expert weights in place over PCIe
  instead of copying them per op or computing them on the CPU. Off by default; the recommended profile for a
  model whose experts live in system RAM is `GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`
  (the second value at least the ubatch size); the expert cache applies it by itself.
  [host-direct-moe.md](host-direct-moe.md)
- **Expert cache** - `--expert-l1-mib N --expert-profile-dir DIR` gives the routed experts a VRAM budget: the
  server profiles which experts the router selects, plans the most valuable set for the budget and installs it
  at a request boundary. The budget decides the expert placement, so it replaces `--n-cpu-moe`. It turns
  host-direct on and loads with `--load-mode none` by itself. `--expert-l1-mib auto` sizes the budget from the free
  VRAM, and `--expert-l1-lru-pct N` keeps part of it as per-layer LRU copies. [expert-cache.md](expert-cache.md)
- **The cache serves prompt processing, and installs are deltas** - MMQ reads the cache arena through the same
  slot tables as MMVQ, and an install moves only what changed between two plans.
  [expert-cache-prefill.md](expert-cache-prefill.md)
- **Exclusive mode (Windows)** - `--expert-cache-mode exclusive` gives every routed expert exactly one home, a
  VRAM slot or a host slot, so the budget is not a second copy of experts that also sit in RAM.
  [expert-cache-exclusive.md](expert-cache-exclusive.md)
- **Profile banks and the prefill swap** - prompt processing and generation are profiled into separate banks;
  `--expert-prefill-swap` additionally installs the prompt plan while a prompt is processed.
  [expert-cache-banks.md](expert-cache-banks.md)
- **Finite host tier with file backing (Windows)** - `--expert-l2-mib N` bounds the host memory of the cache;
  what fits in neither VRAM nor that budget stays in the GGUF file and is read on demand into a ring of host
  slots that the kernels address directly. [expert-cache-l2.md](expert-cache-l2.md)
- **`llama-perplexity` takes the expert cache options**, with the cache frozen, so a model whose routed experts
  do not fit in VRAM can be scored without `--n-cpu-moe`.
- **Unified host policy of the SSD tier** - every host slot of the finite tier is a ring slot under one LRU policy
  with a probation part; a plan install lays the rings out in score order without reading, and a prompt's reads are
  cached like decode reads. Always on where its gate allows it. [expert-cache-l2.md](expert-cache-l2.md)
- **Decode pool** - lends the part of the compute buffer that decode graphs do not use to extra expert positions;
  always on with the expert cache. Its ring admission takes the SSD
  tier's ring occupants (on under the unified policy). [expert-cache-ub-pool.md](expert-cache-ub-pool.md)
- **Expert prefetch (experimental, GLM-5.3 Flash)** - `RANMA_EXPERT_PREFETCH=1` copies the predicted experts of the
  next layer into the decode pool with the copy engine. [expert-prefetch.md](expert-prefetch.md)
- **Joint cache of models with different expert counts** - a draft model with another expert count (the DSpark
  drafter of DeepSeek V4.1 Flash) joins the target's cache. [expert-cache-joint.md](expert-cache-joint.md)
- **Expert sidecar (diagnostics)** - `RANMA_EXPERT_SIDECAR=<file>` records the router selections of every decode call
  in token order, with the server's request and draft/accept events, as a binary file for offline cache-policy
  replay. Off by default. [expert-sidecar.md](expert-sidecar.md)

### Graph runtime

- **Compute buffers regrow with headroom** - a compute buffer that has to grow after its first allocation is
  allocated one eighth larger, so a long prompt does not reallocate a slightly larger buffer at every step.
  [graph-runtime.md](graph-runtime.md)
- **Graph inputs are uploaded through a pinned staging ring** - asynchronously on the stream of the backend
  instead of one blocking copy per input. Same page.
- **Every small input of a split goes through the staging ring** - also the views of user inputs and the results
  of a CPU split, which took a stream synchronize and a blocking copy each. [graph-runtime.md](graph-runtime.md#inputs-that-are-not-marked-as-graph-inputs)
- **One upload for the small inputs of a split** - the staged inputs of at most 64 KiB are written by one kernel
  instead of one copy each. Bit-identical. Same page.
- **LoRA scale folding** - a LoRA scale of exactly 1 produces no graph node, and other scales are folded into
  a pre-scaled copy of the dense B matrices at attach time. `llama-bench` gains `--lora` and `--lora-scaled`.
  Same page.
- **One HIP graph per batch shape** - the backend keys the graphs it captured by the batch shape as well as by
  the first node, so a speculative verification whose width changes between rounds launches the graph it already
  captured for that width. HIP builds only, at most 96 graphs per context. Same page.
- **Graph metadata given back** - stored graph slots and graph metadata buffers keep only what their graph uses
  (the worst case is not kept). [graph-runtime.md](graph-runtime.md#graph-metadata-given-back)
- **One llama graph per batch shape** - a context keeps the built graph of up to 64 batch shapes together with
  its scheduler split and allocation, so a verification width seen before is made current again instead of
  being built, split and allocated anew. With both
  defaults, decode with MTP (`--spec-draft-n-max 7`) in English roleplay is +11.1 % (Qwen3.8-Flash-Next) and
  +8.6 % (DeepSeek V4 Flash) with identical replies. Same page.

### MoE decode kernels (HIP)

- **One q8_1 quantization per shared input** - `MUL_MAT` nodes of one graph that read the same activation
  share its quantization. [moe-decode.md](moe-decode.md)
- **Expert-first launch grid** - with the expert cache, the routed `MUL_MAT_ID` launch alternates its blocks
  between the experts, so the experts in VRAM compute inside the wait for the experts read over the link.
  Same page.
- **Shared expert unit in the routed launch** - on RDNA4 a dense shared expert whose type differs from the
  routed type (Q6_K with IQ2/IQ3 routed experts, Q8_0 with Q4_K, Q5_K, Q5_1 or Q8_0), and the shared down
  matrix, run as one more channel of the routed launches, on top of upstream's same-type shared expert
  fusion; its own launches disappear. Same page.

### Qwen sparse attention

- **Stable top-k tie selection for reproducible runs** - `GGML_CUDA_TOP_K_STABLE_TIES=1` makes the radix top-k
  select the smallest columns among exactly tied values, so selections and outputs can be compared between runs.
  Off by default. [qwen-sparse-attention.md](qwen-sparse-attention.md)
- **Attention reads only the selected cells** - on HIP the tile flash attention gathers the selected K/V rows
  through the compacted index list of its query tile instead of scanning the KV cache. It applies to tiles of up
  to four query rows, from a cache length that depends on the selection budget. Same page.
- **Parallel mask compaction** for long index lists, with `GGML_CUDA_FATTN_COMPACT_VERIFY=1` as the equivalence
  gate. Same page.

### DeepSeek V4

- **Selected-cell attention for 512-wide heads**, including attention calls with sinks.
  [deepseek-v4.md](deepseek-v4.md)
- **Hyper-connection coefficients in one op** with a backend capability probe and the upstream ops as the
  fallback. Same page.
- **Hyper-connection mixes in one kernel** - the norm, the mixing matmul and the coefficients of a site run as
  one HIP kernel for up to four tokens (decode and speculative verification), also in `glm5-next`. Same page.
- **Fused KV compressor** - gather, per-feature softmax and weighted sum in one op. Same page.
- **No dummy HCA compression** on the decode steps in which no block completes. Same page.
- **Raw and compressed K of a layer in one tensor**, so the attention reads a view instead of a concatenated
  copy on every token. Same page.

### DeepSeek V4.1 Flash

- **DeepSeek V4.1 Flash** - upstream pull request
  [ggml-org/llama.cpp#28696](https://github.com/ggml-org/llama.cpp/pull/28696) (not merged), with the top-k reuse
  fix from its review, the two level candidate mask, EXL3 conversions, MXFP8 engram tables and the DSpark drafter
  with `--spec-smart`. [deepseek-v41.md](deepseek-v41.md), [engram-mxfp8.md](engram-mxfp8.md)

### GLM-5.3 Flash

- **LoRA on the attention output** - the KDA and MLA attention output projections of `glm5-next` apply a LoRA
  adapter that targets `attn_output`; upstream multiplies them without the adapter. Without an adapter the graph
  is unchanged. [graph-runtime.md](graph-runtime.md#lora-on-the-glm5-next-attention-output)
- **MTP (NextN) draft** - upstream pull request
  [ggml-org/llama.cpp#29928](https://github.com/ggml-org/llama.cpp/pull/29928) (GLM5-Next MTP) is merged upstream
  (`b9acf138a`), so the series carries no commits of its own for it. `--spec-type draft-mtp` runs the MTP layer of the model file
  without `-md`, and the expert cache accepts it with a finite host tier and profiles its NextN layer.
  [exl3.md](exl3.md#checked-models), [expert-cache-joint.md](expert-cache-joint.md)
- **Hyper-connection coefficients in one node** - `glm5-next` emits the `DSV4_HC_COEF` node of DeepSeek V4
  instead of a dozen small ops per site.
  [glm5-next.md](glm5-next.md)
- **KDA conv step and recurrent state** - the conv step with its state copies runs as one HIP kernel and the scan
  reads its state row in place. Bit-identical. Same page.
- **MLA decode attention on WMMA (RDNA4)** - one to eight query rows of the MLA layers attend over the latent
  cache with WMMA, from the compacted cell list or the dense rows. Same page.
- **MTP catch-up in the first draft batch** - the rows up to the last accepted position are decoded together with
  the first draft row, not as their own call. On for the `glm` architectures; `RANMA_MTP_FUSE_CATCHUP=0` turns it
  off. Same page.

### EXL3 weights and Qwen3.8 decode (HIP)

- **EXL3 weights** - GGUF files hold the weights of ExLlamaV3 EXL3 checkpoints unchanged (mul1, mcg and 3inst
  codebooks), converted by `convert_hf_to_gguf.py`, and run as `MUL_MAT_HAD` on the CPU and on HIP (GEMV for
  decode, WMMA GEMM on RDNA4 for prompts), with host-direct expert banks and the expert cache. Checked with
  `llama`, `qwen3moe`, `lfm2moe`, `qwen4exp`, `deepseek4` and `glm5-next` models; RDNA3 compiles but is
  untested.
  [exl3.md](exl3.md)
- **Fused Qwen3.8 hyper-connections** - the gated residual mix and the combine of each hyper-connection site run
  as fused decode kernels. Same page.
