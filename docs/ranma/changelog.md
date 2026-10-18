# Changelog

This file lists what each RANMA.cpp release adds, changes, fixes and removes compared with the release before it.
Entries are written from the point of view of a user of the fork; the commit-level history is in the git log.

The series is rebuilt for every release and follow-up fixes are folded into the commit they fix, so the history of a
release does not show a fix as a separate change. This file is where the fixes between releases are recorded.
The refs and the release cycle are described in [releases.md](releases.md).

## ranma_20261018

Upstream base: `8f574b0ad`, tagged 2026-10-18

### Added

- Layer compute mode, `--layer-mode on` ([layer-mode.md](layer-mode.md)): a prompt step is one wide ubatch computed
  layer by layer, the routed experts of a layer expert by expert (a new `MOE_FFN_HAD` op for EXL3 experts, also with
  LoRA deltas) and the attention in chunks whose masks are made on the device. The width, the attention chunk and the
  staging buffers are sized from the model and the free memory before the load. It runs for GLM-5.3, Qwen3.8,
  DeepSeek V4 Flash and DeepSeek V4.1 Flash with the expert cache; any other setup runs as `off`. Off by default.
- Layer-major prefill for DeepSeek V4 and V4.1 (part of `--layer-mode`): the ubatches of a prompt run in groups and
  read each expert of a layer from the host once per group instead of once per ubatch. Its buffers are borrowed from
  L1 (the prefill lend), the evicted experts return in decode, and a draft model in the joint expert cache takes part.
- DeepSeek V4.1 Flash: exact tail prefill (on by default: a prompt computes only the rows its output needs in the last
  layers, and the cache can resume at a position), the indexer on the fused lightning indexer op with candidate blocks
  skipped in the HIP decode kernel, attention over `[raw | compressed]` K without concatenating every compressed
  cell, attention sinks in the RDNA4 MLA decode kernel, the fused top-k router for 384 experts, and a DSpark or DFlash
  draft that reads the target features of its sliding window only, which keeps the layer mode on with it
  ([deepseek-v41.md](deepseek-v41.md), [layer-mode.md](layer-mode.md)).
- Slot 0 priority (`--slot0-priority {off,prefill,strict}`, default `prefill`): with several slots, the prompts of the
  other slots wait while slot 0 works on a request ([slot0-priority.md](slot0-priority.md)).
- `--spec-type ngram-varlen`: drafts of variable length from the n-gram history of the session, verified in parts of
  `--spec-ngram-varlen-n-max` tokens with a rollback depth of `--spec-ngram-varlen-rs`, a hazard supplement that
  drafts a long match where nothing else drafts, and its shadow learning and online cost (all on by default;
  `--no-spec-ngram-varlen-hazard`, `--no-spec-ngram-varlen-hazard-shadow` and `--no-spec-ngram-varlen-hazard-online`
  turn them off). The verification cost of the target is shared by all drafters (`--spec-verify-cost-store`,
  `--spec-verify-cost-log`), so `--spec-smart` and the hazard model price a round with the same measured numbers
  ([spec-smart.md](spec-smart.md)).
- `--expert-l1-mib auto` sizes L1 from the free VRAM less `--fit-target`; `--expert-l1-lru-pct N` keeps N % of L1 as
  per-layer LRU copies of recent misses ([expert-cache.md](expert-cache.md),
  [expert-cache-ub-pool.md](expert-cache-ub-pool.md)).
- An L2 of `-1` (the default) is sized at load to hold every expert, the prompt buffers and the experts a lend evicts;
  a demotion from VRAM can be dropped to the file instead of copied into the L2
  ([expert-cache-l2.md](expert-cache-l2.md)).
- The decode pool lends the expert-major op's scratch block between prompt graphs, and the lent slots are a restore
  space: pinned plan experts and an LRU cache of recent misses (`RANMA_EXPERT_LEND_RESTORE=eager` restores the decode
  plan's experts when the prompt ends). The SSD tier's rings order experts of equal score by the prompt profile.
- Expert sidecar (`RANMA_EXPERT_SIDECAR=<file>`, diagnostic): a binary record of the router selections of every decode
  call with the server's request and draft events, for offline replay of cache policies
  ([expert-sidecar.md](expert-sidecar.md)).
- A read engine from NVMe straight into VRAM (D3D12 and IoRing, Windows) with two test programs; the expert cache does not
  use it yet ([expert-cache-p2p.md](expert-cache-p2p.md)).
- RDNA4 kernels: the 512-wide WMMA attention reads 4-row index lists, key tiles are shared across tokens in the
  64 and 32 head lightning indexer prefill, and `HC_PRE`, `RMS_NORM` and the gamma multiply run in one kernel.
- Engram row hints of single-token decode run on a resident worker, and the lazy table prefetch of prompts is gated
  before CPU gathers ([engram-tables.md](engram-tables.md), [ple-prefetch.md](ple-prefetch.md)).
- Diagnostics, off by default: `RANMA_PREFILL_ACCOUNT`, `LLAMA_PREFETCH_TIMING`, `RANMA_EXPERT_L2_TIMELINE`,
  `RANMA_PREFILL_GPU_STAMP`, `RANMA_TENSOR_DUMP` and `RANMA_UB_DUMP`.
- Release table of this release in `docs/ranma/benchmarks/`: DeepSeek V4.1 Flash in EXL3 with layer-major prefill and
  n-gram drafting on the R9700 with 128 and 192 GB, server scenarios and prompts of 512 to 32768 tokens.

### Changed

- The expert cache accepts a multimodal projector and speculative decoding setups that no run has checked with a
  startup warning ("use at your own risk") instead of refusing them.
- The unified host policy of the SSD tier, the decode pool and its ring admission are always on where their gates
  allow them; their tuning values are fixed (probation part 0.10 of each class list, a ghost window of 32 generations,
  the rings laid out in score order at a plan install).
- A drafting context (MTP, DSpark, DFlash, EAGLE3, a plain draft model) computes in views of its target's compute
  buffers whenever that is possible, and only the largest compute buffer lends to the decode pool.
- The layer mode has no host memory budget: the host compute buffer is held for the life of the context, as
  upstream does, and is only logged. With the layer-major prefill the ubatch is at most 8192, and `-b` is the context size.
- DeepSeek V4.1 Flash scores the indexer with the fused op: the compute buffer at ubatch 2048 shrinks from 18189 to
  1805 MiB (32768 context), and it reads only the visible raw cells of its joint K layout and of each attention chunk.
- DeepSeek V4 snapshots the rollback state of its compressors only as deep as the call can be rolled back.

### Removed

The environment switches below are gone and a variable still set in a script is ignored. Each one's former
default is the behavior now.

- Optimizations that were on by default: `GGML_HIP_FATTN_GQA12`, `GGML_HIP_PREFILL_WMMA`, `GGML_HIP_PREFILL_BLAS`,
  `GGML_CUDA_SKINNY_F32`, `GGML_CUDA_MMQ_ID_NCOLS_OPT_SCALE`, `GGML_CUDA_DISABLE_MMVF_SHORT_ROWS`,
  `GGML_CUDA_MMVQ_ID_DEDUP`, `GGML_CUDA_MMVQ_ID_EXPERTS_FIRST`, `GGML_CUDA_MMVQ_ID_FOLD_SHARED`,
  `GGML_CUDA_MMVQ_SHARE_Q8`, `GGML_CUDA_HC_MIXES_FUSION`, `GGML_CUDA_HC_MIXES_MAX_ROWS`, `GGML_CUDA_HC_GATED_FUSION`,
  `GGML_CUDA_DISABLE_SSM_CONV_STATE_FUSION`, `GGML_CUDA_DISABLE_GDN_STATE_ROWS`, `GGML_CUDA_DISABLE_FATTN_MLA_DECODE`,
  `GGML_CUDA_FATTN_SPARSE`, `GGML_CUDA_FATTN_SPARSE_MIN_KV` (the gather opens from 4096 cells),
  `GGML_CUDA_FATTN_COMPACT_PARALLEL`, `GGML_DSV4_FUSION3` with its `GGML_CUDA_DSV4_FUSION3_*` parts,
  `LLAMA_DSV4_HC_COEF_FUSED`, `LLAMA_DSV4_COMPRESSOR_FUSED`, `LLAMA_DSV4_HCA_SKIP_DUMMY`, `LLAMA_LORA_FOLD_SCALE`.
- Graph runtime and server: `LLAMA_INPUT_UPLOAD_ASYNC`, `GGML_SCHED_STAGE_ALL_INPUTS`, `GGML_SCHED_STAGED_GATHER`,
  `GGML_CUDA_GRAPH_PER_SHAPE`, `GGML_CUDA_GRAPH_QUICK_CAPTURE`, `GGML_CUDA_GRAPH_DEFER_FREE`,
  `GGML_CUDA_GRAPH_EVICT_SECONDS` (per-shape graphs are dropped only at the per-context bound),
  `LLAMA_GRAPH_REUSE_SHAPES` (a context keeps 64 shapes), `LLAMA_GRAPH_META_TRIM`, `LLAMA_SERVER_CKPT_REUSE`.
- Speculative decoding: `LLAMA_SPEC_TRIGRAM_CONF` (the trigram confidence is always on), `LLAMA_SPEC_TRIGRAM_TOPK`
  (one candidate), `LLAMA_SPEC_DSPARK_TCONF_POS` (one coefficient set for all positions) and `RANMA_MTP_EXPERT_JOIN`
  (the NextN layer always joins the profile).
- Expert cache: `RANMA_EXPERT_UB_POOL` with its `_FILTER`, `_RING`, `_ROWS`, `_RCAP`, `_MIN_MIB` and `_MARGIN_MIB`
  tuning (the decode pool is always on with the expert cache), and `RANMA_EXPERT_L2_UNIFIED` with its `_RESEED`,
  `_PROMPT`, `_PROBATION`, `_PROBATION_MIN` and `_GHOST` tuning (the unified host policy is always on where its gate
  allows it).
- `LLAMA_DRAFT_SHARE_COMPUTE` (and its alias `LLAMA_MTP_SHARE_COMPUTE`): a drafting context always shares its
  target's compute buffers when it can ([expert-cache-ub-pool.md](expert-cache-ub-pool.md)).
- The redraw of the VRAM size-class split at plan installs (`RANMA_EXPERT_L1_REDRAW_BENEFIT`,
  `RANMA_EXPERT_L1_REDRAW_MOVE`, off unless set) and its HIP virtual memory arenas: the size-class split stays
  as the load sets it. The host address table and the chunked host arenas that exclusive mode with an unlimited
  host tier had for it go as well.

## ranma_20261013

Upstream base: `3d65c90d0`, tagged 2026-10-13

### Added

- DeepSeek V4.1 Flash from the upstream pull request #28696, with the top-k reuse fix from its review, the two level
  candidate mask, EXL3 conversions and the DSpark drafter ([deepseek-v41.md](deepseek-v41.md)).
- `GGML_TYPE_MXFP8` rows for the V4.1 engram tables (gather only), read from the original FP8 checkpoint.
- Engram tables (V4.1 engram, Qwen3.8 PLE) as a GGUF set of their own (`<model>-engram…`) that every quantization
  shares, found next to the model or by `--engram-tables`, which also replaces tables a model holds itself;
  `convert_hf_to_gguf.py --resume` / `--resume-check` for split conversions.
- The expert cache joins models with different expert counts (the 128-expert DSpark drafter of a 384-expert target).
- Decode pool (on by default with the expert cache, `RANMA_EXPERT_UB_POOL=0` turns it off): the unused part of the
  compute buffer holds extra expert positions during decode; its ring admission takes the SSD tier's ring occupants.
- `LLAMA_DRAFT_SHARE_COMPUTE=1` (off by default): a drafting context (MTP or a draft model) computes in its target's
  compute buffers; `LLAMA_MTP_SHARE_COMPUTE` is an alias.
- `--spec-smart` for `draft-dspark` (on by default as for `draft-mtp`), with a block drafter and the DSpark round log
  (`LLAMA_SPEC_DSPARK_LOG`).
- Trigram confidence of a draft token from the request's own trigram counts, and a trigram profile next to the
  smart store that keeps the counts and the learnt coefficients across requests and restarts.
- Expert prefetch into the decode pool for GLM-5.3 Flash (`RANMA_EXPERT_PREFETCH=1`, experimental, off by default;
  [expert-prefetch.md](expert-prefetch.md)).
- Settings by model and hardware for the published DeepSeek V4.1 Flash files: context, expert cache budgets and the
  draft model, with the memory measured ([recipes.md](recipes.md)).
- Release table of this release in `docs/ranma/benchmarks/`: the server scenarios of Qwen3.8-Flash-Next, DeepSeek V4
  Flash, GLM-5.3 Flash and DeepSeek V4.1 Flash in EXL3, DeepSeek V4.1 Flash also with 224 GB and on the RX 9070 XT
  emulation with 64 GB.

### Changed

- The expert cache needs no environment switch and no `--load-mode`: an expert cache budget turns the host-direct
  path on when `GGML_CUDA_HOST_DIRECT` is not set, raises its in-place token limit to the ubatch size when
  `GGML_CUDA_HOST_DIRECT_MAX_BATCH` is not set, and loads the model with `--load-mode none` unless `--load-mode` is
  given. Values that are set are kept; without the expert cache nothing changes.
- The SSD tier's unified host policy is on by default (`RANMA_EXPERT_L2_UNIFIED=0` turns it off); with it the
  decode pool's ring admission is on, a plan install lays the rings out in score order without reading
  (`RANMA_EXPERT_L2_UNIFIED_RESEED=order`), and a prompt's reads are cached like decode reads
  (`RANMA_EXPERT_L2_UNIFIED_PROMPT=lru`).
- Trigram confidence is on by default (`LLAMA_SPEC_TRIGRAM_CONF=0` turns it off); the trigram profile is on by
  default with a smart store (`--no-spec-trigram-profile`).
- Graph reuse keeps 64 llama graphs and 96 HIP graphs per context, without the timed sweep
  (`GGML_CUDA_GRAPH_EVICT_SECONDS` defaults to 0).
- Graph slots and the graph metadata buffer give back what their graph does not use (`LLAMA_GRAPH_META_TRIM=0`
  keeps the worst case).
- The smart draft length leaves graph builds, captures and decode pool lends out of the round cost.

### Fixed

- `--gpu-heartbeat-seconds` starts the heartbeat before the model load, so a long expert cache setup does not let
  Windows evict the VRAM of the process.

## ranma_20261009

Upstream base: `462524043`, tagged 2026-10-09

### Added

- GLM-5.3 Flash (glm5-next) with the NextN MTP head: the speculative catch-up rows of a verification batch are now
  decoded in the batch of the first draft step, which saves one launch per round and the decode of rejected rows.
  The held-back rows are saved with a prompt checkpoint. On by default for glm architectures;
  `RANMA_MTP_FUSE_CATCHUP=0` turns it off.
- Small host inputs of a graph split that are not marked as graph inputs now use the pinned staging ring, and the
  small staged inputs are uploaded with one kernel (`GGML_SCHED_STAGE_ALL_INPUTS=0` and `GGML_SCHED_STAGED_GATHER=0`
  restore the old paths). Measured on GLM-5.3 Flash decode.
- Release table of this release in `docs/ranma/benchmarks/` (Qwen3.8-Flash-Next, DeepSeek V4 Flash and GLM-5.3 Flash,
  each in EXL3 and in a general GGUF, against upstream), and a GLM-5.3 Flash page for the decode kernels.

### Changed

- GLM-5.3 Flash decode on RDNA4: MLA decode attention on the WMMA units, one-column f16 mat-vecs with short rows read
  with 16 byte loads, the hyper-connection coefficients fused into one `DSV4_HC_COEF` node
  (`LLAMA_DSV4_HC_COEF_FUSED=0` keeps the separate ops), `RMS_NORM` + `MUL_MAT` + `DSV4_HC_COEF` fused for the
  hyper-connection mixes of up to four tokens, and the KDA conv step fused with the GDN state read in place.

### Fixed

- GLM-5.3 Flash with the NextN MTP head on the expert cache: the context is now also joined to the cache profile.
  Joined only as a separate draft model, the NextN block counted no selection and got no VRAM slice, so every draft
  step read its experts from the host tier. `RANMA_MTP_EXPERT_JOIN=0` leaves it out of the profile.

### Removed

- The fast path for the gathered sparse attention of glm5-next is no longer carried; upstream removed that gather path
  (pull request #30042).

## ranma_20261007

Upstream base: `8e1642198`, tagged 2026-10-07

### Added

- GLM-5.3 Flash (glm5-next) NextN MTP draft head, carried unchanged from upstream pull request #29928
  (`--spec-type draft-mtp`).
- GLM-5.3 Flash in EXL3: the converter splits the fused KDA qkv and conv1d projections and keeps the MTP layer, and the
  graph runs the EXL3 weights of the KDA, MLA, indexer and LM head projections through `build_mm`.
- The expert cache accepts an MTP context that runs on the target's own weights (the NextN block of glm5-next) together
  with a finite L2 and the prefill swap.
- `ranma-release.yml`: a GitHub workflow that builds the Windows HIP executables of a `ranma_*` tag and attaches them as
  a zip to a draft GitHub Release. It never publishes a release.

### Changed

- The one-token routed MMVQ launch can carry a shared expert of another type than the routed experts, and a Q8_0 shared
  expert of eight-warp routed types, on top of upstream's shared-expert fusion (pull request #29184).
- Converting an EXL3 checkpoint now follows the model class for MTP layers (default, `--no-mtp`, `--mtp`). A model
  whose MTP layers cannot be stored as EXL3, such as Qwen3.8-Flash-Next, is converted without them after an error
  message.

### Fixed

- GLM-5.3 Flash: a LoRA adapter that targets the attention output (`attn_output`), such as the Heretic adapter, was
  loaded but not applied. The attention output projections now go through `build_lora_mm`.

### Removed

- The fork's own folds of the dense shared expert into the one-token MUL_MAT_ID MMVQ launch are replaced by upstream's
  shared-expert fusion (see Changed).
- The `relu_sum_heads` op for the Qwen3.8 indexer score is no longer carried.

## ranma_20261005

Upstream base: `42d958167`, tagged 2026-10-05

### Added

- EXL3 weights (ExLlamaV3 trellis quantizations) in GGUF, on CPU and HIP: the `mul1`, `mcg` and `3inst` codebooks at
  1 to 8 bits (1.5, 2.5 and 3.5 bits for `mul1`), run through a new `MUL_MAT_HAD` op (matrix product with Hadamard
  rotations). A loader check accepts the EXL3 type IDs only from a model with EXL3 metadata.
- `convert_hf_to_gguf.py` converts EXL3 checkpoints, including the n-gram embedding tables of Qwen3.8 (as the
  `EXL3R_M<bits>` row codec types) and the routed experts and grouped `wo_a` of DeepSeek V4 Flash. Rows converted from
  an EXL3 source are kept in source order and marked with `<arch>.rope.style = "neox"`.
- The expert cache holds EXL3 expert banks, moving the rotation rows with their expert.
- Decode fusions for Qwen3.8 on RDNA4: the low-rank gated residual mix and the hyper-connection combine.
- `docs/ranma/benchmarks/` and a README restructured around the main features; the list of changes moves to
  `docs/ranma/README.md`.

### Changed

- The `MUL_MAT_HAD` GEMV has a lower fixed cost, fewer instructions per weight in the trellis decode, and a
  fp16-input mode.
- A whole-sequence clear of a single-stream DeepSeek V4 compressor state uses one buffer clear instead of one tensor
  clear per layer, rollback plane and tensor.
- The Qwen3.8 sparse-attention indexer work of the fork is rebuilt on the indexer of upstream pull request #29751; the
  sparse gather, mask compaction and server recovery commits stay.

### Fixed

- Expert cache on hosts without PCIe AtomicOps (seen on RDNA3): the demand bitmap and mailbox flags were set with
  `atomicOr` on mapped host memory, and lost updates could make the CPU worker miss demanded experts. They are now
  built in shared memory and stored with plain writes.
- Expert cache: a routed bank with `.bias` or `.scale` companions (gpt-oss biases, NVFP4 scales) turned the cache off
  with a false "not quantized" error. The geometry now takes only the `_exps.weight` tensors of a bank.

### Removed

- The Qwen3.8-Flash-Next MTP draft head of upstream pull request #28243, which upstream has merged (pull request
  #29761).
- The pooled indexer key cache, the cache of keys after norm and RoPE, the weighted block top-k, the skipped selection
  while the budget covers every cell, and the incremental block layout, which the indexer of upstream pull request
  #29751 replaces.

## ranma_20261001

Upstream base: `ed7ac35e1`, tagged 2026-10-01

### Added

- `--spec-smart` (on by default in `llama-server`): `draft-mtp` chooses the draft length of every round from the
  measured acceptance and the predicted verification time, instead of fixed per-position thresholds.
- Qwen3.8-Flash-Next MTP draft head from upstream pull request #28243, now part of the main release (before, it was
  only in `ranma_20260928_qwen_mtp`).

### Changed

- One llama graph per batch shape is kept by default (24 shapes per context, `LLAMA_GRAPH_REUSE_SHAPES=0` turns it
  off). Unused per-shape CUDA graphs live for 60 s (`GGML_CUDA_GRAPH_EVICT_SECONDS`), and a changed graph of a replayed
  family is captured at once (`GGML_CUDA_GRAPH_QUICK_CAPTURE`).
- On RDNA4, `MUL_MAT_ID` runs on MMVQ up to 8 tokens for experts read from host memory (host-direct or the expert
  cache) for IQ2_XS, IQ3_S, IQ3_XXS, MXFP4, Q4_K, Q5_K, Q5_1 and Q8_0.
- A rollback of the Qwen indexer to a block boundary no longer clears the open block, which removes a host round trip
  per rollback with MTP.
- Pull requests are accepted on conditions, against `ranma_upstream`; the release cycle is in `docs/ranma/releases.md`.

## ranma_20260928

Upstream base: `53ed051ce`, tagged 2026-09-28

### Added

- One expert cache for every MoE model of a process: a draft model with routed experts joins the target's cache by
  default (`--expert-cache-draft off` keeps it out; `--expert-cache-weight target=W,draft=W` weights the plan). The
  finite L2 and the prefill swap work with a draft in the joint cache, and also with an MTP head loaded from its own
  file.
- Early fetch of the experts of token-routed layers (DeepSeek V4 routes its first layers by token id), so the SSD tier
  can read them before the layer runs.
- Optional redraw of the VRAM size-class partition of the expert cache at a plan install.
- `GGML_CUDA_GRAPH_PER_SHAPE`: one CUDA graph per batch shape, on by default for HIP builds.
- `LLAMA_GRAPH_REUSE_SHAPES=N`: one llama graph per batch shape, off by default in this release.

### Changed

- The SSD tier staging ring lives inside the per-class host arenas; installs relabel host slots and keep the rings
  instead of copying. Exclusive mode with an unlimited host tier gets a host address table and chunked host arenas.
- The expert cache orders equal scores with a balanced tie order, so a cold fill is a round robin over the layers of a
  class.
- The stable tie selection of the radix top-k is an optional feature (`GGML_CUDA_TOP_K_STABLE_TIES=1`), no longer a
  test switch.

### Fixed

- `llama-server` aborted with "failed to remove sequence" whenever a memory refused to drop a suffix (the pooled Qwen
  indexer does when a removal cuts a block). After a checkpoint restore it now drops the checkpoints above the kept
  tokens and restores an older one, and a refusal during speculative rollback ends the request instead of the server.

### ranma_20260928_qwen_mtp

Upstream base: `53ed051ce`, tagged 2026-09-28. `ranma_20260928` plus upstream pull request #28243.

- Added: Qwen3.8-Flash-Next MTP draft head from upstream pull request #28243, applied as one commit, and a benchmark
  of MTP n1 in English roleplay.

## ranma_20260924

Upstream base: `ec5a12b85`, tagged 2026-09-24

### Added

- `llama-completion` accepts the `--expert-*` options and drives the expert cache like the server.

### Changed

- The SSD tier reads of prompt batches overlap with the expert matmuls, one weight kind at a time.
- A multi-token `MUL_MAT_ID` MMVQ kernel on HIP reads each distinct expert once.
- On RDNA4 the `MMQ_ID` column tile is sized against three times the mean column count per expert.
- `llama-server` keeps a just-restored context checkpoint instead of creating it again
  (`LLAMA_SERVER_CKPT_REUSE=0` restores the old behaviour).
- The Qwen indexer block layout is updated incrementally instead of rescanning every cell (`LLAMA_QSA_LAYOUT_CACHE=0`),
  and the indexer score uses a fused `relu_sum_heads` op (`LLAMA_QSA_SCORE_FUSE=0` keeps the old graph).
- Small DeepSeek V4 decode kernels are fused.

### Fixed

- Qwen3.8-Flash-Next and other models on RDNA4: small F32 matmuls (hyper-connection injection, SSM alpha/beta, MoE
  router) went to hipBLAS, which chooses its solution per process, so prompt speed and the summation order of the logits
  differed between processes. They now run on fixed-order kernels.

## ranma_20260922

Upstream base: `aa39d7a3e`, tagged 2026-09-22

### Added

- Qwen3.8 sparse-attention (QSA) indexer: a cache of pooled keys per block, a cache of keys after norm and RoPE, a
  weighted block top-k (`GGML_OP_TOP_K_BLOCK`), no selection while the budget covers every cell, and a gather of only
  the selected cells in the HIP tile flash attention (head sizes 256 and 512) with a faster mask compaction.
- DeepSeek V4 Flash: the hyper-connection coefficients in the comb kernel, a fused KV compressor, no dummy HCA
  compression when no block completes, and raw and compressed K in one tensor.
- `llama-perplexity` accepts the `--expert-*` options.
- Graph inputs of a split are uploaded through a pinned staging ring (`LLAMA_INPUT_UPLOAD_ASYNC=0` restores the
  blocking copy).
- RDNA4: WMMA attention for 512-wide heads in prompt processing, 12-column tile attention for one-row GQA-12 groups,
  four rows per block for one-column MMVQ, and a padded F16 BLAS path for wide dense Q2_K, Q6_K and IQ2 matmuls.
- MMVQ: one q8_1 quantization per graph for a shared input (`GGML_CUDA_MMVQ_SHARE_Q8=0` restores one per `MUL_MAT`),
  alternating experts in the `MUL_MAT_ID` launch grid, and the shared expert folded into the one-token routed launch.
- `GGML_CUDA_TOP_K_STABLE_TIES=1`: a stable tie selection in the radix top-k.

### Changed

- The RDNA4 dense MMVQ and MMQ dispatch tables are tuned (one entry set for 2 to 16 columns).
- The LoRA scale node is skipped when the effective scale is 1.

### Fixed

- Long contexts on DeepSeek V4 Flash (HIP, Windows): the compute buffer was freed and allocated again, 0.5 MiB larger,
  at every step of a prompt, and dedicated GPU memory and commit grew by about 4 GiB over 65536 tokens. A buffer that
  has to grow is now allocated with 1/8 headroom.
- `llama-perplexity` and its KL mode sized the logits buffers for the whole chunk instead of the scored half, which
  committed 6.4 GB instead of 3.2 GB at `-c 8192` with a 129k vocabulary.

## ranma_20260917

Upstream base: `093a2f86c`, tagged 2026-09-17

### Added

- Profiled static expert cache for host-resident MoE experts (`--expert-l1-mib`, `--expert-profile-dir`,
  `--expert-seed`, `--expert-freeze`, profile archive and reset).
- The expert cache serves prompt processing through MMQ, and an install is a delta transaction instead of a full
  refill.
- `--expert-cache-mode exclusive`: every routed expert lives in exactly one place, a VRAM or a host arena slot.
- Profile banks for prompt processing and generation, and `--expert-prefill-swap`.
- Finite host tier with file backing (`--expert-l2-mib`): what fits in neither VRAM nor the budget is read on demand
  from the GGUF file.
- `--ple-prefetch {off,prefill,always}`: prefetch of the lazy per-layer embedding rows, with a parallel gather.

### Changed

- With ordinary mmap loading off, only the model shards that hold lazy tensors are mapped.

## ranma_20260914

Upstream base: `43f3dda62`, tagged 2026-09-14

### Added

- Fork README, contribution policy and trimmed CI.
- Host-direct `MUL_MAT_ID` on mapped host weights on HIP: the GPU reads quantized MoE expert weights in place from
  pinned host memory (off by default).
- `--gpu-heartbeat-seconds`: `llama-server` records a GPU event on every model device while idle, so that Windows
  keeps VRAM resident (off by default, 5 s recommended).
- `--spec-draft-p-min` takes a list indexed by draft position, and `--spec-draft-p-continue` adds a second gate.

### Changed

- MMVQ weight row reads of the last partial block are clamped to the tensor.
- RDNA4: four rows per block for MMVQ with 3 to 8 columns, and a dense K-quant MMQ crossover.
