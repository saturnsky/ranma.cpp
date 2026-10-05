# Changes over upstream

Each user-visible change gets a line here and a page under `docs/ranma/` that describes what it is, when it
applies, how to switch it, and its limits. The main features are summarized in the [README](../../README.md#main-features).

### RDNA4 kernels (HIP)

- **Small-batch matmul dispatch** - four weight rows per MMVQ block for 3..8 activation columns and a
  per-type MMVQ/MMQ crossover, so the cost of a decode call no longer rises and then falls with the number of rows
  in it. This is the range of a speculative verification step and of a server that batches a few slots.
  [rdna4-small-batch.md](rdna4-small-batch.md)
- **Four rows per block at one column** - single-token decode of a wide matrix reads the activation once per
  four weight rows; chosen per call from the matrix size. Same page.
- **12-column tile attention for GQA-12 groups** - token generation on a model whose head group is a multiple
  of twelve reads each K/V head once instead of three times. `GGML_HIP_FATTN_GQA12=0` restores the upstream
  dispatch. Same page.
- **WMMA attention for 512-wide heads** - prompt processing of a 512/512 head at GQA 8 with an F16 KV cache
  takes the wide-tile MMA kernel. `GGML_HIP_PREFILL_WMMA=0` restores the upstream dispatch.
  [rdna4-prefill.md](rdna4-prefill.md)
- **Padded F16 BLAS for wide dense Q2_K/Q6_K/IQ2 matmuls** - wide prompt matmuls of those types convert both
  operands to F16 with a padded row pitch and run hipBLASLt instead of MMQ. Needs `ROCBLAS_USE_HIPBLASLT=1`;
  `GGML_HIP_PREFILL_BLAS=0` turns it off. Same page.
- **Fixed-order kernels for skinny F32 matmuls** - F32 products of up to 512 weight rows with more than eight
  activation columns (router, SSM and hyper-connection projections of a prompt batch) run on dedicated kernels
  instead of hipBLAS, whose per-process solution choice changed their speed and summation order from one process
  to the next. `GGML_CUDA_SKINNY_F32=0` restores hipBLAS. Same page.
- **Wider MoE column tiles** - the MMQ tile width of a `MUL_MAT_ID` is sized against three times the mean
  column count per expert, so popular experts re-read their weights less often. Bit-identical;
  `GGML_CUDA_MMQ_ID_NCOLS_OPT_SCALE=1` restores the upstream width. Same page.

### Server and common tools

- **GPU heartbeat (llama-server)** - `--gpu-heartbeat-seconds 5` records one GPU event per interval while the
  server is idle and while the model is freed, so that Windows does not evict the VRAM of the process between
  turns. Off by default. [gpu-heartbeat.md](gpu-heartbeat.md)
- **Per-position draft thresholds** - `--spec-draft-p-min` takes one probability per draft position, and
  `--spec-draft-p-continue` keeps a token in the draft but stops drafting after it. Defaults unchanged.
  [spec-draft-thresholds.md](spec-draft-thresholds.md)
- **Smart draft length for draft-mtp (llama-server)** - `--spec-smart` chooses the draft length at every step from
  the measured verification time per width and a calibrated acceptance of the draft probabilities, up to
  `--spec-draft-n-max`; `--spec-smart-store PATH` keeps the estimates across restarts of the same model, build and
  cache settings. Off by default. [spec-smart.md](spec-smart.md)
- **Reuse of a just-restored context checkpoint (llama-server)** - when the first prompt batch after a checkpoint
  restore starts at that checkpoint, the server keeps the restored entry instead of serializing the unchanged
  state again. `LLAMA_SERVER_CKPT_REUSE=0` restores the upstream behaviour.
  [server-checkpoints.md](server-checkpoints.md)
- **Per-layer embedding prefetch** - `--ple-prefetch {off,prefill,always}` (default `always`) hands the rows of
  a lazily mapped per-layer embedding table to the operating system before the gather, and the gather of that
  tensor runs on the threadpool. [ple-prefetch.md](ple-prefetch.md)
- **Only the shards with lazy tensors are mapped** - with mmap loading off, the loader no longer maps model
  files that nothing reads through the mapping. Same page.

### Host-resident MoE experts

- **Host-direct MoE weights (HIP)** - `MUL_MAT_ID` kernels read host-resident expert weights in place over PCIe
  instead of copying them per op or computing them on the CPU. Off by default; the recommended profile for a
  model whose experts live in system RAM is `GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`
  (the second value at least the ubatch size). [host-direct-moe.md](host-direct-moe.md)
- **Expert cache** - `--expert-l1-mib N --expert-profile-dir DIR` gives the routed experts a VRAM budget: the
  server profiles which experts the router selects, plans the most valuable set for the budget and installs it
  at a request boundary. The budget decides the expert placement, so it replaces `--n-cpu-moe`. Needs host-direct
  and `--load-mode none`. [expert-cache.md](expert-cache.md)
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

### Graph runtime

- **Compute buffers regrow with headroom** - a compute buffer that has to grow after its first allocation is
  allocated one eighth larger, so a long prompt does not reallocate a slightly larger buffer at every step.
  [graph-runtime.md](graph-runtime.md)
- **Graph inputs are uploaded through a pinned staging ring** - asynchronously on the stream of the backend
  instead of one blocking copy per input. `LLAMA_INPUT_UPLOAD_ASYNC=0` restores the blocking path. Same page.
- **LoRA scale folding** - a LoRA scale of exactly 1 produces no graph node, and other scales are folded into
  a pre-scaled copy of the dense B matrices at attach time. `llama-bench` gains `--lora` and `--lora-scaled`.
  Same page.
- **One HIP graph per batch shape** - the backend keys the graphs it captured by the batch shape as well as by
  the first node, so a speculative verification whose width changes between rounds launches the graph it already
  captured for that width. On by default in HIP builds; `GGML_CUDA_GRAPH_PER_SHAPE=0` restores the keying by the
  first node. Same page.

### MoE decode kernels (HIP)

- **One q8_1 quantization per shared input** - `MUL_MAT` nodes of one graph that read the same activation
  share its quantization. [moe-decode.md](moe-decode.md)
- **Expert-first launch grid** - with the expert cache, the routed `MUL_MAT_ID` launch alternates its blocks
  between the experts, so the experts in VRAM compute inside the wait for the experts read over the link.
  Same page.
- **Shared expert folded into the routed launch** - on RDNA4 the dense shared expert of a layer is computed as
  one more unit of that grid, and its own launches disappear. Same page.
- **The same fold for Q8_0 shared experts** of models whose routed experts are Q4_K, Q5_K, Q5_1 or Q8_0. Same
  page.

### Qwen sparse attention

- **Stable top-k tie selection for reproducible runs** - `GGML_CUDA_TOP_K_STABLE_TIES=1` makes the radix top-k
  select the smallest columns among exactly tied values, so selections and outputs can be compared between runs.
  Off by default. [qwen-sparse-attention.md](qwen-sparse-attention.md)
- **Attention reads only the selected cells** - on HIP the tile flash attention gathers the selected K/V rows
  through the compacted index list of its query tile instead of scanning the KV cache. It applies to tiles of up
  to four query rows, from a cache length that depends on the selection budget. `GGML_CUDA_FATTN_SPARSE=0` keeps
  the dense kernel. Same page.
- **Parallel mask compaction** for long index lists, with `GGML_CUDA_FATTN_COMPACT_VERIFY=1` as the equivalence
  gate. Same page.

### DeepSeek V4

- **Selected-cell attention for 512-wide heads**, including attention calls with sinks.
  [deepseek-v4.md](deepseek-v4.md)
- **Hyper-connection coefficients in one op** with a backend capability probe and the upstream ops as the
  fallback. Same page.
- **Fused KV compressor** - gather, per-feature softmax and weighted sum in one op. Same page.
- **No dummy HCA compression** on the decode steps in which no block completes. Same page.
- **Raw and compressed K of a layer in one tensor**, so the attention reads a view instead of a concatenated
  copy on every token. Same page.

### EXL3 weights and Qwen3.8 decode (HIP)

- **EXL3 weights** - GGUF files hold the weights of ExLlamaV3 EXL3 checkpoints unchanged (mul1, mcg and 3inst
  codebooks), converted by `convert_hf_to_gguf.py`, and run as `MUL_MAT_HAD` on the CPU and on HIP (GEMV for
  decode, WMMA GEMM on RDNA4 for prompts), with host-direct expert banks and the expert cache. Checked with
  `llama`, `qwen3moe`, `lfm2moe`, `qwen4exp` and `deepseek4` models; RDNA3 compiles but is untested.
  [exl3.md](exl3.md)
- **Fused Qwen3.8 hyper-connections** - the gated residual mix and the combine of each hyper-connection site run
  as fused decode kernels. `GGML_CUDA_HC_GATED_FUSION=0` turns them off. Same page.
