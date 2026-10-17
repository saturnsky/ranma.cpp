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
| width (`-ub`) | the widest that fits the measured memory (below), at most the context of one sequence and the widest pass of the expert-major op (`min(E, width * top_k) + width * top_k / 16 + 1 <= 65535` GEMM items). No rate of the link or of the GEMMs is assumed: the width is halved from the top until it fits, then bisected to within 1/16 (at least 512). A prompt shorter than the width runs as one graph of its own length. |
| width with the layer-major prefill (on for DeepSeek-V4 and V4.1) | at most 8192 (then the same check): the passes of a prompt read each expert once and get their width from L1 when the prompt runs, so the ubatch only sets the width of the segment graphs and their compute buffers (outside L1). Their per-token time hardly changes above a few thousand tokens while the buffers grow with the width. Without the layer-major prefill the widest is kept: there a wider ubatch reads the experts fewer times. |
| logical batch (`-b`) | the context: a prompt is one call (`llama_decode`) of several ubatches of the width (below) |
| attention chunk | the largest that fits: the whole width (no chunks) is measured first; only when it does not fit, a chunk between 512 and the width (linear estimate from both ends, then checked) |
| expert-major rows per pass | the whole width (`width * top_k`), so every expert of a layer is read once per graph; the op's scratch grows with it (about 40 KiB per row on GLM-5.3, 2.5 GiB at width 8192) and is counted in the check |
| staging buffer | none in one pass (no earlier pass to hide the copies behind); with the expert-major rows below the width, two buffers of `min(64, rows/129)` experts when a mean expert gets at least 129 rows |
| host-direct token limit | the width |
| L1 budget | the staging buffer and the op's scratch come out of `--expert-l1-mib` when it is at least twice as large |
| op scratch block | the op's scratch is one fixed VRAM block of the expert cache (`ggml_expert_config::op_scratch_bytes`); between prompt graphs the decode pool borrows it (below) |
| other slots' prompt batches | see below |

The plan fits when, on the GPU, model + KV + compute buffer (measured with a no_alloc load of the model and the
context, as `-fit` measures) + L1 budget + the op's buffers + `--fit-target` (per device, default 1024 MiB, the margin
of `-fit` and of `--expert-l1-mib auto`) is at most the free memory. The host compute buffer is a compute buffer
the scheduler holds for the life of the context, so it has no budget: the host compute buffer and the growth of the
process peak during that load (graph metadata) are logged for information only. Each measurement logs one `layer mode: width ..., chunk ...` line; the decision logs `layer mode: on: ...`.

Given values win: `-ub` fixes the width (the chunk is still chosen), `-b` bounds it and keeps its value, and the
development switches `RANMA_EXPERT_MAJOR_STAGE_MIB` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH` keep their values when set.
The VRAM reserve is `--fit-target`. The mode passes the expert-major MoE op, the attention chunk, the rows per pass and
the layer-major prefill to the graphs through a process-wide setting (`llama_layer_mode_config`, `src/llama-ext.h`), not
through environment variables; the layer-major prefill is switched on for DeepSeek-V4 and V4.1 (the models it cuts) and
the setting is restored when the mode stays off. A prompt batch
of another slot takes 3000 ms of compute (below).

## The logical batch

`-b` is not tied to the width. The memory check runs with `-b` equal to the width, and the context is then built with
`-b` equal to the context size: no buffer of the context grows with it (the KV cache and the graphs follow `-ub`, the
output buffer the number of outputs), only the per-token target features a drafter reads, and the one drafter the
mode keeps reads those of the last rows of a call only (below). The ubatches of a long prompt are the same as with `-b` equal to the width; what changes is the end of
the call:

- DeepSeek-V4.1's exact tail prefill computes the rows the output of a call needs. With one
  call per width every call ended in a tail of 20 layers and read their experts again; with one call per prompt only
  the ubatches near its end compute tail rows, the earlier ones write the raw KV of the tail layers and stop.
- The other models compute every row of every ubatch either way: the same ubatches give the same results.
- `llama-server` still ends a prompt call where it creates a context checkpoint (at a user message, `4 + n_ubatch`
  and 4 tokens before the end of the prompt, `--ctx-checkpoints`), so with checkpoints a long prompt is three or four
  calls instead of one per width.
- In `llama-server` a prompt fills a batch up to the context only when the batch is empty. A prompt that joins a batch
  that already holds tokens (the generated tokens of other slots, another slot's prompt) fills it to one ubatch, as
  before: those tokens would otherwise wait for the whole prompt, and a call of several sequences has no tail end.
- A call is not interrupted: a request cancelled during its prompt, and the progress reports of a streamed prompt,
  wait for the end of the call, now the prompt (or the part up to the next checkpoint) instead of one width.

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

## The pieces of the prefill lend

The layer-major prefill borrows its VRAM buffers from L1 (`expert-lend.h`): the lowest ranked
L1 experts leave and the top slots of their size class hold the buffers. A class lends its gate, up and down arenas
together (one slot number holds one expert in each), so each lent region is one kind of those slots and the largest
indivisible piece sets how many slots a lend takes; what the other regions do not use is lent but empty. The pieces
are therefore kept small:

- The segment edges are lent per ubatch: the segments read and write the rows of one ubatch at a time, so an edge
  slot is cut into pieces of whole ubatches (as many pieces as a lend takes, 48 in all). The slot of the router input
  stays one piece (the MoE of a layer reads it over every row of the pass), and so do the carried tensors and the MoE
  output. DeepSeek-V4.1 Flash: the hyper-connection state (80 KiB per token, 4.6 GiB at 59885 tokens) becomes pieces
  of 640 MiB.
- The two staging buffers of the PCIe-once op are two pieces (`GGML_EXPERT_LEND_STAGE`, `GGML_EXPERT_LEND_STAGE1`);
  the op takes both from the lend or both from its pool.
- Of the residents that tie at the rank the lend stopped in (same tier and score, e.g. every resident of a cold
  cache), only those of the classes the pieces need leave: the others are given back, the class of the largest experts
  first. Residents of a lower tier or score than the tie still leave first.

Each lend logs one `prefill lend N packing:` line after its summary: every piece (role, MiB, class.kind), the use and
capacity of every lent region, the evicted residents per class and the tied victims given back. The pieces are
packed largest first by first fit, and by worst fit (into the region with the most room) when first fit leaves one
out. Replayed with the pieces of the 2026-10-15/17 logs (`tests/test-expert-lend.cpp`, P6): DeepSeek-V4.1 Flash at
59885 tokens 1109 -> 888 slots (an exact packing would take 875), the joint cache with its DSpark draft at 7309
tokens 415 -> 304.

## Slot 0 priority

With `-np 2` or more and `--slot0-priority` on, a request on slot 0 waits for at most one batch of another slot. A
wide batch would make that wait long, so in layer mode a prompt batch of a slot other than 0 takes about
3000 ms of compute: its token count is that time times the rate measured on the
previous such batches (500 tokens/s before the first one), in steps of 256, and may span several ubatches. Slot 0's
own prompt batches are not held to it (see the logical batch above).
The 3000 ms is the target that sizes the batch, not a limit: a batch is not stopped when the time is over, so with a slow SSD, a cold
cache or the first batch (the 500 tokens/s assumption) the wait of slot 0 can exceed 3 s.

## Supported setups

The expert cache must be on (`--expert-l1-mib` or `--expert-l2-mib`), the model must be GLM-5.3 (`glm5-next`) , Qwen3.8 (`qwen4exp`), DeepSeek-V4.1-Flash (`deepseek41`) or DeepSeek-V4-Flash
(`deepseek4`) with
EXL3 routed experts (separate gate and up). Anything else logs `layer mode: ...; prompt processing runs as with
--layer-mode off` and runs as `off`. So does a setup with a context the memory check does not build yet: a draft
model or an MTP context (`-md`, `--spec-type` other than `ngram-varlen`, `draft-dflash` or `draft-dspark`), or a
multimodal projector (`--mmproj`). `ngram-varlen` alone has no context of its own, so the mode stays on with it.

A DFlash/DSpark drafter (`--spec-type draft-dflash` or `draft-dspark`) keeps the mode on when its draft is a DSV4
draft: every stage attends a standard sliding window over uncompressed KV (the GGUF has `dflash.hyper_connection.count`,
`dflash.attention.sliding_window` and no compressed stage). Such a draft reads back the target features of the last
`n_swa` positions only: the injection of a target row writes the KV of that row alone, and a later query reaches no
position older than its window. Its drafter therefore asks the target for the features of the last `n_swa` rows of
every call (`llama_set_embeddings_layer_inp_last`), which keeps the exact tail of DeepSeek-V4.1 (its last layer then
computes those rows instead of one) and the layer-major prefill (the features come out of the segment that makes
them) on, and holds `n_swa` rows per sequence in the output buffer instead of `n_batch`. The draft context gets its
own ubatch, as wide as the window and the blocks of every sequence with their verification, and its memory is
measured once with a no_alloc load of the draft context on the target's: its weights and KV on the GPU, its compute as
far as it exceeds the target's compute buffer it shares, and in host memory its compute, its routed experts and the
window outputs. A measurement that fails keeps the mode off. A draft without a feature window, or one whose window is
narrower than a verification of its block, needs the features of every row and keeps the mode off.

With the draft in the joint expert cache (`expert-cache-joint.md`) the prefill lend works over the layers of both
models: a prompt evicts the lowest ranked L1 residents of every class, whichever model they belong to, and the decode
of both models brings the experts of the decode plan back into the restore space as it reads them. A lend waits until
the draft has joined (its weights loaded, its aux rows final, its files open in the SSD tier). The restore kernels of
the two models share one restore space, so their graphs never run at once: the draft computes in the target's compute
buffer, and the cache also drains the other context's stream whenever the computing context changes. The target
features of the window are copied to the host output buffer inside the layer-major call, so nothing of them lives in
lent memory after it.

- MTP blocks of their own (blocks with `nextn` tensors) are not counted as routed layers of a prompt step; the
  staging slot is sized for the largest routed layer.
- LoRA on routed experts: F16/F32 deltas are added by the expert-major op itself; the largest rank of the adapters'
  routed deltas is read from the files and its rows are part of the op's scratch. Deltas of another type keep their
  layer on the per-kind MoE path (a warning names the file), whose intermediates the plan does not count.
- `-np` > 1: the measurement builds the context with every sequence. Without a unified KV cache a ubatch of several
  sequences runs its attention unchunked (a warning says so).
- The op's scratch is sized by the same layout function the op uses (`ggml/src/ggml-cuda/moe-ffn-had-layout.h`).
