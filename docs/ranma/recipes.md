# Recipes: settings by model and hardware

Starting settings for the EXL3 GGUF files published for this fork
([Hugging Face, SaturnHeaven](https://huggingface.co/SaturnHeaven)): context, expert cache budgets and draft model for a few
hardware configurations, each with the memory it was measured to use. They are examples, not tuned optima: choose the context, the
split between L1 and L2 and the draft model for your own use and system, and size them as described below.

Every row was measured with `ranma_20261013` on a Radeon AI PRO R9700 (32 GB, RDNA4) in a machine with 224 GB of host memory. The
128 GB and 64 GB host rows were reproduced on that machine by occupying memory, and the 16 GB card row by limiting the device
memory the server uses ([benchmark record](benchmarks/2026-10-13-release.md)).

## Common settings

The commands below share these options:

- **Pass the first shard** (`-00001-of-0000N.gguf`) to `-m`; the other shards are found by name.
- **Host-direct and `--load-mode none` are automatic** with an expert cache budget ([expert-cache.md](expert-cache.md#requirements));
  no environment switch is needed. The decode pool is on by default ([expert-cache-ub-pool.md](expert-cache-ub-pool.md)).
- **`-np 1`** for one user. Without it the server runs four slots, the per-slot state grows, and a finite L2 is refused
  (`finite L2 needs positive batch/parallel bounds`).
- **`--expert-cache-mode exclusive`** keeps one copy of each expert, in VRAM or in RAM
  ([expert-cache-exclusive.md](expert-cache-exclusive.md)).
- **`--expert-profile-dir`**: any writable folder, one per model and workload works best. Without it the placement stays a fixed random
  one; with it the placement adapts at request boundaries, and about ten requests rebuild a profile.
- No `--n-cpu-moe` / `--cpu-moe`: the budget decides the placement.

## Sizing L1 and L2

- **L1 (VRAM).** L1 = VRAM - (memory the cache does not cover) - margin. The cache does not cover the weights outside it, the KV
  cache and the compute buffers; the load log prints the model, KV and compute buffer sizes. The KV cache and the compute buffers grow
  with the context, so a shorter context leaves more VRAM to L1. Keep a margin of a few GiB, more if the GPU also drives a display.
  The L1 budget includes the cache's own tables, so it is not exactly the expert bytes. A draft model needs VRAM of its own.
- **L2 (host RAM).** `-1` keeps every expert that is not in L1 in RAM. A finite L2 bounds the RAM the experts use, and the experts that
  fit in neither tier are read from the GGUF files on the SSD ([expert-cache-l2.md](expert-cache-l2.md)). In exclusive mode, L1 + L2
  at or above the routed expert size reads nothing from the SSD. Set L2 to the RAM you can spare minus what your other programs use;
  the process uses some RAM on top of L2 (given per model below). The server refuses a host requirement above the physical memory
  installed; it does not look at how much is free.

## DeepSeek-V4.1-Flash

Files: [SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-EXL3-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-EXL3-GGUF)
(3.0 bpw, Mia-AiLab, and its DSpark draft model),
[SaturnHeaven/DeepSeek-V4.1-Flash-UNCENSORED-RANMA-EXL3-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-UNCENSORED-RANMA-EXL3-GGUF)
(2.9 bpw, dealignai, and its DSpark draft model), and the engram tables both use,
[SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF](https://huggingface.co/SaturnHeaven/DeepSeek-V4.1-Flash-RANMA-Engram-GGUF).
Give `--engram-tables` the first file of the table set ([engram-tables.md](engram-tables.md)): a folder is searched for files named
after the model, which the shared set is not. The tables are read from their files on demand.

At `-c 32768` with the default ubatch, about 10.4 GiB of device memory sits outside the expert cache; the compute buffers grow with the
context (about 1.3 GiB at 8192, 4.4 GiB at 32768). Peak private bytes of the process: 120.6 GiB with L2 92160, 45.2 GiB with L2 32768,
180.7 GiB with L2 153600 (they include the backing of the VRAM allocations, [method](benchmarks/method.md#memory-columns)).

| Hardware | `-c` | L1 / L2 (MiB) | Draft | Dedicated VRAM peak |
|---|---:|---|---|---|
| 32 GB + 128 GB | 32768 | 17727 / 92160 | none | 27.7 GiB |
| 16 GB + 64 GB | 32768 | 4096 / 32768 | none | 14.2 GiB |
| 32 GB + 224 GB | 32768 | 17727 / 153600 | none | 27.8 GiB |
| 32 GB + 224 GB | 32768 | 16384 / 153600 | DSpark | 27.9 GiB |

```
llama-server -m EXL3-3.0bpw-h6/DeepSeek-V4.1-Flash-EXL3-3.0bpw-h6-00001-of-00006.gguf \
    --engram-tables engram-MXFP8/DeepSeek-V4.1-Flash-engram-MXFP8-00001-of-00007.gguf \
    -c 32768 -ngl 999 -fa on -np 1 \
    --expert-cache-mode exclusive --expert-l1-mib 17727 --expert-l2-mib 92160 \
    --expert-profile-dir profiles/ds41f-30
```

The rows were measured with the 3.0 bpw files. The 2.9 bpw files take the same settings
(`-m EXL3-2.9bpw-h6/DeepSeek-V4.1-Flash-UNCENSORED-EXL3-2.9bpw-h6-00001-of-00006.gguf`, same table set).

DSpark draft model (its experts join the expert cache, [expert-cache-joint.md](expert-cache-joint.md)): add
`-md DSpark-EXL3-4bpw/DeepSeek-V4.1-Flash-DSpark-EXL3-4bpw.gguf -ngld 999 --spec-type draft-dspark` and lower L1 to 16384 as in the
last row. The draft length is chosen by `--spec-smart`, on by default ([spec-smart.md](spec-smart.md)).

## Related documents

[expert-cache.md](expert-cache.md) (options), [expert-cache-exclusive.md](expert-cache-exclusive.md),
[expert-cache-l2.md](expert-cache-l2.md) (SSD tier), [expert-cache-joint.md](expert-cache-joint.md) (draft models in the cache),
[expert-cache-ub-pool.md](expert-cache-ub-pool.md), [spec-smart.md](spec-smart.md), [engram-tables.md](engram-tables.md),
[exl3.md](exl3.md), [gpu-heartbeat.md](gpu-heartbeat.md).
