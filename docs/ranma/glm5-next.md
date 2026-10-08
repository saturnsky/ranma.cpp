# GLM-5.3 Flash (`glm5-next`) decode

Changes to the decode path of `glm5-next` models (GLM-5.3 Flash): its graph and the HIP kernels that only its
layers reach. The LoRA on the attention output is described in
[graph-runtime.md](graph-runtime.md#lora-on-the-glm5-next-attention-output), the MTP layer in the expert cache in
[expert-cache-joint.md](expert-cache-joint.md). The hyper-connection mixes kernel that `glm5-next` shares with
DeepSeek V4 is in [deepseek-v4.md](deepseek-v4.md#hyper-connection-mixes-in-one-kernel), the short f16 mat-vec rows
of the MLA projections in [rdna4-small-batch.md](rdna4-small-batch.md#short-f16-mat-vec-rows).

## Switches

| Switch | Default | Effect of the non-default value |
|---|---|---|
| `LLAMA_DSV4_HC_COEF_FUSED=0` | on | The hyper-connection coefficients are built from the separate ops again (also for DeepSeek V4). |
| `GGML_CUDA_DISABLE_SSM_CONV_STATE_FUSION=1` | off | The KDA conv step runs as separate concat, conv, silu and copy kernels. |
| `GGML_CUDA_DISABLE_GDN_STATE_ROWS=1` | off | The recurrent state row is copied out before the scan again. |
| `GGML_CUDA_DISABLE_FATTN_MLA_DECODE=1` | off | MLA decode attention takes the regular flash-attention kernels. |
| `RANMA_MTP_FUSE_CATCHUP` | unset | `0` decodes the catch-up rows as their own batch; `1` also joins them for other single-head MTP models. Unset, it is on for the `glm` architectures only. |

All are read once per process.

## Hyper-connection coefficients in one node

`glm5-next` computed the pre, post and combination coefficients of each hyper-connection site with a chain of
about a dozen small elementwise and Sinkhorn ops. It now emits the single `GGML_OP_DSV4_HC_COEF` node that DeepSeek
V4 uses ([deepseek-v4.md](deepseek-v4.md#hyper-connection-coefficients-in-one-kernel)) and takes pre, post and comb
as views of its result. The same fused-op probe decides at load whether the backend runs it; otherwise the separate
ops are built. Not bit-identical to the op chain: the kernel agrees with the CPU reference in `test-backend-ops`,
and a KL-divergence check of GLM-5.3 Flash with this node and the mixes kernel stayed within the difference
between two micro-batch sizes of the unfused path.

## KDA conv step and recurrent state

A decode step of a linear-attention (KDA) layer ran concat, `ssm_conv`, silu and one state copy per rollback window
as separate nodes, and a `GET_ROWS` copied the whole recurrent state only to hand it to `gated_delta_net`.

- The graph puts silu directly after `ssm_conv` (the existing `ssm_conv` + silu fusion never matched because of a
  reshape between them), writes the conv states back after the conv chain, gathers the recurrent state right before
  the scan and drops two copies of data that was already contiguous.
- HIP runs concat + `ssm_conv` + silu + the state copies of up to four rollback windows (up to eight tokens) as one
  kernel. It does not fuse when a later node still reads the concat or conv result.
- A `GET_ROWS` of one state row followed by `gated_delta_net` lets the op read the row straight from the cache,
  together with the existing snapshot copy fusion.

Both are bit-identical to the separate nodes. `test-backend-ops -o SSM_CONV_STATE_FUSION,GATED_DELTA_NET_STATE_ROWS`
covers them.

## MLA decode attention on WMMA (RDNA4)

The MLA layers attend with 64 heads over a latent cache whose K and V are the same 512 values per cell (absorbed
MLA, V a view of K). For one to eight query rows the tile kernel needed 100-250 us per layer in a sparse decode,
because it serializes sixteen gather and compute stages per KV tile. A block now stages 16 cells in LDS, computes
the scores and the P*V update with WMMA while the loads of the next chunk are in flight, and a second kernel merges
the partial results. The cells come from the existing mask compaction when the cache holds at least twice the index
budget; otherwise the dense cache rows are visited, up to 8192.

- Rounding differs from the tile kernel (f16 probabilities, f32 accumulation); a KL-divergence check of GLM-5.3
  Flash at an 8192 context stayed within the difference between two micro-batch sizes of the tile kernel.
- Sinks, ALiBi, softcap, other head counts and a V that is not a view of K take the regular kernels. DeepSeek V4
  attends with sinks and is therefore not affected.
- `test-backend-ops -o FLASH_ATTN_EXT` covers the dense and list forms, several query rows and the fallbacks.

## MTP catch-up rows in the first draft batch

With a single-head MTP draft, the catch-up decode of a verification batch ran as its own call, including the rows
of the rejected positions. The draft now keeps the rows of a short batch (up to 16) and runs the rows up to the last
accepted position together with the first draft row in one batch, with an output only on the draft row. That saves
one launch per round and the decode of the rejected rows. A kept batch is dropped when its positions do not continue
the draft memory (request ended, rollback) and is decoded separately when it cannot join the draft batch. Chained
heads and shared-memory drafts are unchanged.
