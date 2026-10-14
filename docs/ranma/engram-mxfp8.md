# DeepSeek V4.1 engram tables as MXFP8

## What it is

The two engram tables of DeepSeek V4.1 Flash (layers 1 and 14, about 384 million rows of 256 values each)
are stored in the GGUF exactly as the original checkpoint holds them: FP8 E4M3 values with one E8M0 scale
per 32 values. Nothing is requantized, so the table is lossless and about 101.5 GB in total per table
(264 bytes per row).

`GGML_TYPE_MXFP8` (ranma id 320) is the type. Its block is the layout of upstream PR ggml-org/llama.cpp#26157:
256 E4M3 values, then 8 E8M0 scales (264 bytes). A value is `e4m3(q) * 2^(e - 127)`, the dequantization of
`ParallelEngramEmbedding.forward` in the reference `model.py`. The only difference from the PR is the type id: it
stays 320 here and moves to upstream's when the PR merges. The E4M3 NaN code reads as 0, as in the PR.

An engram row is exactly one block, and the checkpoint stores a row's 256 value bytes in `.weight` and its 8
scale bytes in `.scale`, so converting is concatenating the two rows. No value changes.

## Type rules

- Rows are only gathered: `to_float` and CPU `GET_ROWS`. `mul_mat`, `mul_mat_id` and the CPU `supports_op` refuse
  the type, as for the EXL3 row codecs. `ggml_is_row_gather_only()` names both groups.
- The CUDA backend does not support `GET_ROWS` on it, so the scheduler gathers on the CPU from the mapped file;
  `ggml_is_row_gather_only()` also keeps the CUDA buffer types from padding its rows.
- The gather returns F32, which enters the GPU graph as before.

## Converter

`convert_hf_to_gguf.py --engram-source <folder>` reads the tables from the original FP8 checkpoint folder
(`model.safetensors.index.json` and the shards that hold `layers.N.engram.embed.weight` / `.scale`). An EXL3
conversion folder has no tables, so the option is needed there; without the option the model folder is searched.
A missing table, an unexpected dtype (not F8_E4M3 values with F8_E8M0 scales), a shape other than
`[rows, 256]` / `[rows, 8]`, or a row count that differs from `engram_num_embeddings` stops the conversion.
The dense engram tensors (q, k) and the EXL3 `wkv` still come from the model folder.

With `--split-max-size` the tables are a GGUF set of their own (`<model>-engram…`, one piece of at most the split size per
file), which the model set lists and does not contain; one table set serves every quantization of the model. See
`engram-tables.md`.

The table is written by streaming: `GGUFWriter` calls the table object's `tofile()` when it writes the data, and
the object reads the two shard ranges in chunks of 2M rows, joins them and writes them. There is no staging file;
the peak memory is a few chunk buffers (about 1.2 GB). The output file grows by about 101.5 GB per table.

## Runtime

`blk.N.engram_embd.weight` (or its pieces `blk.N.engram_embd.p<i>.weight` in the table set) is a lazily mapped tensor. After graph inputs are prepared, the common `llama_prefetch_batch` path discovers CPU `GET_ROWS` nodes with lazy host weights and host input row IDs.
It hands sorted, merged row ranges to the operating system with the helper of
the per-layer embedding prefetch (`ple-prefetch.md`). It follows `--ple-prefetch` and its thresholds
(256 rows for `prefill`, 16 for `always`; a decode token gathers 24 rows per engram layer). A CPU `GET_ROWS` of
an engram table with 256 rows or more also runs on the threadpool.

In a prompt ubatch, a gather of at least 256 rows with preceding device computation starts its hint asynchronously, including the first engram. Smaller gathers stay synchronous. The CPU scheduler waits for the hints of the tables in its split after input copies and backend transition waits, before computing the split. This uses a dedicated readiness callback, not the user's eval callback, and does not split or synchronize the graph further. A late hint produces an exposed wait instead of an overlapping hint invocation and gather.

`LLAMA_ENGRAM_PREFETCH_ASYNC=0` keeps hints synchronous. The default is on; the switch now applies to the common lazy-table path, not only engram. Gathers at the front of the graph (including shared Qwen PLE) stay synchronous because there is no preceding device computation to overlap. Single-token input preparation keeps the existing synchronous prefetch calls.

`LLAMA_PREFETCH_PIECES_PARALLEL=1` enables concurrent piece hints. The default is 0 pending measurement. The worker count is bounded by the queued piece count and hardware concurrency; the default uses one worker. Each piece has independent completion state. Parallel hints also apply to synchronous prompt gathers, which wait before submission. Workers own row IDs and are joined before the ubatch submission scope returns, including abort and error returns. Thread creation failure drains unclaimed tasks synchronously; refused OS hints leave ordinary demand paging intact.

`LLAMA_PREFETCH_TIMING=1` logs `table_prefetch` records with `what=lazy_gather` and a `table_prefetch_wait` record with `us.wait`. Accounting records include `table_prefetch_wait_ms`. Piece timings are per-piece work, not a summed parallel wall time. The common path includes the placeholder row 0 that a masked piece gather really reads. Completion means the OS hint call returned, not guaranteed page residency or completion of all OS background I/O (particularly `MADV_WILLNEED`). No pinning or residency promise is added.

The loader marks all lazy tensors with `GGML_TENSOR_FLAG_LAZY` (ranma extension, bit 15). A CPU gather uses the threadpool from 256 rows when its source is a marked host weight, without tensor-name checks. A new model with host input IDs and lazy host tables needs no model-specific prefetch code. Computed or device-side IDs are skipped because they are not yet available during input preparation. Models without lazy gather tables keep their existing execution path. Performance of this gated implementation is not yet measured; the previous ungated later-layer implementation measured 12.79 s -> 11.91 s on a 7309-token DS4.1F prompt.

Gemma4 keeps its existing synchronous PLE hints and may receive a duplicate hint from the common path. These pages have already been hinted, but submitting another hint still has a cost. This implementation accepts that cost to keep the upstream input helper and Gemma4 model source unchanged; Gemma4 performance is not part of this work's validation.

## Tests

`test-backend-ops -o GET_ROWS` has `test_get_rows_mxfp8` cases. The numpy dequantization is in
`gguf-py/gguf/quants.py` (`MXFP8`); `gguf-py/gguf/mxfp8.py` holds the repack.
