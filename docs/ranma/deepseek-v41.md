# DeepSeek V4.1 Flash

DeepSeek V4.1 Flash (`deepseek41`) runs from the upstream pull request
[#28696](https://github.com/ggml-org/llama.cpp/pull/28696) (head `2076412`, not merged yet), carried at the start
of the series with the authors' commits, plus the changes of this fork listed here. If upstream merges a different
implementation, this fork follows upstream, and GGUF files made for this one may need to be converted again.

## The pull request

The converter (`DeepseekV41ForCausalLM`), the architecture, the engram tables, the hyper-connections whose
coefficients lag by one sublayer, the shared compressed streams with their layer roles, the sparse attention, the
DSpark draft model (`dflash`) and the chat template's DSML tag names. One commit of it is adapted to the newer
upstream base: the guard against the endless K rotation search of a cache with no layers sits in the
`llama_kv_cache` constructor, where upstream moved that search (the commit says so in a `[ranma: ...]` line).

## Changes of this fork

- **Top-k of the index source for the layers that reuse it.** A compressed layer that is not an index source used
  the full causal mask over the compressed rows instead of the positions its source picked; this matches the
  reference only while a source holds at most `index_topk` (512) rows. The fix is the one posted by nicholasshirley
  in the review of the pull request
  ([comment](https://github.com/ggml-org/llama.cpp/pull/28696#issuecomment-5873400013)), in a commit of its own
  under his name. On top of it the carry names the source layer and is checked against `dsv41_topk_source`.
- **Two level candidate mask.** As in the reference, the candidate source layer (20) scores blocks of 8 compressed
  positions, keeps the best 2048 blocks (with the block of the newest position), and the index sources after it
  (24, 28, 32, 36) rank positions only inside them. Layers 20 and up compress at ratio 1, so the mask acts from
  16384 tokens of context on. The keep row is per block (64 MiB at a 64K context and a 2048 ubatch). The layer and
  the block sizes are GGUF keys (`attention.indexer.candidate_source_layer`, `candidate_topk_blocks`,
  `candidate_block_size`) that the converter writes; a file without them runs without the mask and warns at load.
- **EXL3 weights.** `wo_a`, the output head and the DSpark confidence head go through `build_mm`, so EXL3
  conversions run ([exl3.md](exl3.md)); the DSpark head on an EXL3 target finds the rotations of the target's
  shared weights.
- **Engram tables as MXFP8**, read from the original FP8 checkpoint without changing a value, with a prefetch of
  the gathered rows: [engram-mxfp8.md](engram-mxfp8.md). A split conversion writes them as a table set of their
  own that every quantization shares: [engram-tables.md](engram-tables.md).
- **DSpark draft model.** `convert_hf_to_gguf.py` converts the DSpark drafter from FP8 and EXL3 sources. Its 128
  experts per layer join the expert cache of the 384-expert target ([expert-cache-joint.md](expert-cache-joint.md)),
  a GPU-resident drafter also runs outside the joint cache with a finite host tier, and `--spec-smart` chooses per
  round whether to draft and how many tokens to verify ([spec-smart.md](spec-smart.md)).
- The expert cache's unified host policy of the SSD tier and its prompt rule were measured on V4.1
  ([expert-cache-l2.md](expert-cache-l2.md)).

The V4 kernels of this fork (fused hyper-connection coefficients, compressor and the rest of
[deepseek-v4.md](deepseek-v4.md)) take the compression ratio of the layer, so they serve the V4.1 streams too.

## Limits

- Text only: the checkpoint's vision part is not converted.
- The MTP layers of the checkpoint are not mapped; speculative decoding uses the DSpark drafter.
- The candidate mask and the top-k carry are checked to compile and run on the real weights; the per-layer
  comparison with the reference (`inference/model.py`) was done on a small random-weight model by the reviewer for
  the carry, not yet for the candidate mask.
