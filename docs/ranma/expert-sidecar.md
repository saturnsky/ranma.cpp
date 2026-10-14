# Expert sidecar (RANMA_EXPERT_SIDECAR)

A low-cost binary record of the routed-expert selections, for offline replay of expert-cache policies and for
drafter analysis. For every decode call it keeps the tokens in batch order and, per routed layer, the experts the
router selected for every row in top-k order; the server adds request and speculation events (drafted / accepted).
Off unless the variable is set; then nothing in the decode path changes.

## Settings (read once per process)

| variable | values | default |
|---|---|---|
| `RANMA_EXPERT_SIDECAR` | output file; `{pid}` is replaced by the process id; an existing file is overwritten | off |
| `RANMA_EXPERT_SIDECAR_PREFILL` | `off` (tokens only), `counts` (per layer and expert), `ids` (per row, up to the row cap) | `off` |
| `RANMA_EXPERT_SIDECAR_ROWS` | rows kept per layer in a decode, verify or draft call | 64 |
| `RANMA_EXPERT_SIDECAR_PREFILL_ROWS` | rows kept per layer in a prefill call with `ids` | 4096 |
| `RANMA_EXPERT_SIDECAR_TIER` | `1`: one tier byte per selection (home at read time, see below) | 0 |
| `RANMA_EXPERT_SIDECAR_SOURCE` | `auto`, `device` (HIP expert cache), `eval` (scheduler eval callback, CPU contexts) | `auto` |

A call is a prefill call when the server tags it PREFILL or MIXED, or when it is not a VERIFY call and has more
tokens than `RANMA_EXPERT_SIDECAR_ROWS` (the prompt catch-up of a draft model).

## How the selections are taken

- device (HIP): the router hook of the expert cache (`ggml_cuda_expert_profile_ids`, the place the profiler, the
  decode pool and the SSD tier already use) launches one single-block kernel per routed layer that writes the ids into
  coherent mapped host memory; no copy call. Each decode call opens a bank with a one-thread kernel on the model's
  stream (epoch, bank, mode), so the record kernels, also those inside captured graphs, find their place at run time.
  Four banks per model: a call is read after the synchronization that the caller performs anyway (sampling); only a
  fifth unread call forces a synchronization (flag bit 2). Only models in the cache are recorded (a draft model outside
  a joint cache has no ids).
- eval: the scheduler's eval callback reads the `ffn_moe_topk-<il>` tensors of host buffers. Used for CPU contexts
  only (on a GPU the callback synchronizes every split); never installed over a user callback.
- A writer thread writes whole blocks (about 256 KiB, or 1 s, or at a request end, or when a context is freed) with
  `fwrite` + `fflush`; an abnormal exit loses at most the open block. More than 64 MiB of queued blocks are dropped
  and counted (DROPPED event).

## File format, version 1

Little-endian, packed, arrays padded with zeros to 4 bytes (`pad4`).

File header: `char magic[8] = "RNMSIDE1"`, `u32 version = 1`, `u32 json_bytes`, JSON (process: `pid`, `t0_unix_us`,
`exe`, `cmdline`, `env` = variables starting with RANMA_/LLAMA_/GGML_/LLM_/HIP_, `settings`), zeros to 8 bytes.

Block: `u32 magic = 0x4B4C4253 ("SBLK")`, `u32 seq`, `u32 payload_bytes`, `u32 n_records`, `u32 crc32 (zlib) of the
payload`, `u32 0`, payload. A reader stops at the first block whose magic, length or crc does not match.

Record: `u16 type, u16 flags, u32 bytes` (whole record, multiple of 4), then the body.

- 1 CTX: `u32 ctx_id, u32 json_bytes, json` with `arch, desc, n_layer, n_layer_all, n_expert, n_expert_used, n_ctx,
  n_batch, n_ubatch, n_seq_max, source, expert_cache`. Layer numbers are the model's own (a joint cache offset is
  removed) and below `n_layer_all`: an MTP context (DeepSeek V4 Flash) records its head at layer `n_layer` and up.
- 2 CALL (one decode call): `u32 ctx_id, u32 call_seq, i32 req_id, u8 phase (1 prefill, 2 decode, 3 verify,
  4 draft, 5 mixed, 0 unknown), u8 mode (0 ids, 1 counts, 2 tokens), u8 source (0 none, 1 device, 2 eval), u8 0,
  u64 t_begin_us, u32 t_collect_us, u32 n_tokens, u32 n_layer_blocks, u32 0, i32 token[n], i32 pos[n], i32 seq[n],
  u8 output[n] pad4`, then the layer blocks in ascending layer order. flags: bit 0 a layer was cut at the row cap,
  bit 1 tier bytes present, bit 2 a synchronization was added for this call, bit 3 no expert source.
  - ids layer block: `u16 layer, u16 n_used, u32 rows, u32 stored, u32 launches, u16 ids[stored*n_used] pad4`, then
    with bit 1 `u8 tier[stored*n_used] pad4`. ids[r*k + j] is rank j (0 = highest router score) of row r; 0xFFFF is an
    invalid id.
  - counts layer block: `u16 layer, u16 n_expert, u32 rows, u32 0, u32 launches, u32 counts[n_expert]`.
  - Rows follow the ubatches. With one sequence (`-np 1`) row r is token r. A layer with fewer rows than tokens
    computed only some of them: the output tokens when rows equals their number (the last layer of a prompt, the
    DeepSeek-V4.1 tail), else the last `rows` tokens.
- 3 EVENT: `u32 ctx_id, u16 kind, u16 0, i32 req_id, u64 t_us, i64 a, b, c, d`. kind 1 REQ_BEGIN (slot, prompt
  tokens, reused tokens), 2 REQ_END (generated, drafted, accepted), 3 ACCEPT (pos of the first verified token,
  drafted, accepted, call_seq of the verify call), 4 DROPPED (blocks, bytes).
- 4 NOTE: `u32 key_bytes, u32 value_bytes, key, value pad4`. The server writes `build`, `ctx_role.<ctx_id>`,
  `spec_type`, `expert_profile_dir`, `expert_l1_mib`, `expert_l2_mib`, `expert_cache_draft`.

Tier byte: 0 VRAM arena, 1 host, 2 file (SSD tier), 255 unknown; the home when the call is read, so an expert read from
the file during the call may show as host, and decode pool copies are not distinguished.
