# Layer compute mode (`--layer-mode`)

For MoE models whose experts do not fit in VRAM (the expert cache with its host tier), a prompt step reads every
expert of a layer that is not in VRAM once per graph, whatever the number of tokens in the graph. A wider ubatch
spreads that read over more tokens. `--layer-mode on` makes a prompt step one wide ubatch and computes it layer by
layer:

- the routed experts of a layer expert by expert over the whole width (`GGML_OP_MOE_FFN_HAD`), so the MoE
  intermediates are a few thousand rows large instead of `[n_ff, top_k, width]`; experts with many rows are copied to
  VRAM by the copy engine behind the previous chunk (staging), the rest are read in place (host-direct);
- the attention in chunks of query rows, so the `[n_kv, width]` selection intermediates are `[n_kv, chunk]` large.

`--layer-mode off` (the default) runs the upstream graphs with the given `-ub`. Env: `LLAMA_ARG_LAYER_MODE`.

## What `on` decides

Before the model loads, from the GGUF metadata and the free memory:

| setting | rule |
|---|---|
| width (`-ub`, `-b`) | the widest that fits the measured memory (below), at most the context of one sequence and the widest pass of the expert-major op (`min(E, width * top_k) + width * top_k / 16 + 1 <= 65535` GEMM items). No rate of the link or of the GEMMs is assumed: the width is halved from the top until it fits, then bisected to within 1/16 (at least 512). A prompt shorter than the width runs as one graph of its own length. |
| attention chunk | the largest that fits: the whole width (no chunks) is measured first; only when it does not fit, a chunk between 512 and the width (linear estimate from both ends, then checked) |
| expert-major rows per pass | the whole width (`width * top_k`), so every expert of a layer is read once per graph; the op's scratch grows with it (about 40 KiB per row on GLM-5.3, 2.5 GiB at width 8192) and is counted in the check |
| staging buffer | none in one pass (no earlier pass to hide the copies behind); with the expert-major rows below the width, two buffers of `min(64, rows/129)` experts when a mean expert gets at least 129 rows |
| host-direct token limit | the width |
| L1 budget | the staging buffer and the op's scratch come out of `--expert-l1-mib` when it is at least twice as large |
| op scratch block | the op's scratch is one fixed VRAM block of the expert cache (`ggml_expert_config::op_scratch_bytes`); between prompt graphs the decode pool borrows it (below) |
| other slots' prompt batches | see below |

The plan fits when, on the GPU, model + KV + compute buffer (measured with a no_alloc load of the model and the
context, as `-fit` measures) + L1 budget + the op's buffers + a reserve of 4 GiB is at most the free memory, and on
of `-fit` and of `--expert-l1-mib auto`) is at most the free memory. The host compute buffer is a compute buffer
the scheduler holds for the life of the context, so it has no budget: the host compute buffer and the growth of the
process peak during that load (graph metadata) are logged for information only. Each measurement logs one `layer mode: width ..., chunk ...` line; the decision logs `layer mode: on: ...`.

Given values win: `-ub` fixes the width (the chunk is still chosen), `-b` bounds it, and the development switches
`RANMA_EXPERT_MAJOR_MOE`, `RANMA_ATTN_CHUNK`, `RANMA_EXPERT_MAJOR_STAGE_MIB`, `RANMA_EXPERT_MAJOR_ROWS` and
the setting is restored when the mode stays off. A prompt batch
`GGML_CUDA_HOST_DIRECT_MAX_BATCH` keep their values when set. The reserves can be changed for development:
`RANMA_LAYER_MODE_RESERVE_MIB` (4096), `RANMA_LAYER_MODE_HOST_MIB` (from the available memory),

## The op's scratch during generation

The scratch is only used by graphs that run the expert-major op, so the decode pool
([expert-cache-ub-pool.md](expert-cache-ub-pool.md)) lends it as extra positions between them, next to the spare part
of the compute buffer. The positions hold copies of host experts only, never an L1 resident, so the L1 plan is not
touched; taking the block back copies nothing (the pooled experts' table entries return to their homes).

- A graph that contains `MOE_FFN_HAD` takes the block back before it is launched or captured (in stream order: the
  reclaim kernel runs on that graph's stream after the earlier decode kernels; another context's stream is synchronized
  first; the inplace prefetch's copies are waited for). A prompt too short for the op (under 64 tokens) does not take it back.
- The next graph of decode rows lends it again, empty. With `-np 2` that is one take-back and one lend per wide prompt
  batch of either slot, the same cycle the compute buffer's part of the pool has.
- A plan install (also the prefill/decode profile swap) reclaims the pool while nothing computes, as before.
- The log: one line at the first lend that includes the block, the first graph of each op count and width
  (`expert-major MoE: N MOE_FFN_HAD nodes in a graph of T tokens`), and with `RANMA_EXPERT_UB_POOL_LOG=1` one line per
  take-back with the pooled experts it dropped and the running counts of lends, take-backs and dropped experts.

## Slot 0 priority

With `-np 2` or more and `--slot0-priority` on, a request on slot 0 waits for at most one batch of another slot. A
wide batch would make that wait long, so in layer mode a prompt batch of a slot other than 0 takes about
3000 ms of compute: its token count is that time times the rate measured on the
previous such batches (500 tokens/s before the first one), in steps of 256. Slot 0's own prompt batches keep the
full width.
The 3000 ms is the target that sizes the batch, not a limit: a batch is not stopped when the time is over, so with a slow SSD, a cold
cache or the first batch (the 500 tokens/s assumption) the wait of slot 0 can exceed 3 s.

## Supported setups

The expert cache must be on (`--expert-l1-mib` or `--expert-l2-mib`), the model must be GLM-5.3 (`glm5-next`) with
EXL3 routed experts (separate gate and up). Anything else logs `layer mode: ...; prompt processing runs as with
--layer-mode off` and runs as `off`. So does a setup with a context the memory check does not build yet: a draft
model or an MTP context (`-md`, `--spec-type`), or a multimodal projector (`--mmproj`).

- MTP blocks of their own (blocks with `nextn` tensors) are not counted as routed layers of a prompt step; the
  staging slot is sized for the largest routed layer.
- LoRA on routed experts: F16/F32 deltas are added by the expert-major op itself; the largest rank of the adapters'
  routed deltas is read from the files and its rows are part of the op's scratch. Deltas of another type keep their
  layer on the per-kind MoE path (a warning names the file), whose intermediates the plan does not count.
- `-np` > 1: the measurement builds the context with every sequence. Without a unified KV cache a ubatch of several
  sequences runs its attention unchunked (a warning says so).
- The op's scratch is sized by the same layout function the op uses (`ggml/src/ggml-cuda/moe-ffn-had-layout.h`).
