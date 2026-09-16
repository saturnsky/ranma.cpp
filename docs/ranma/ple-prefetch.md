# Per-layer embedding prefetch and gather (`--ple-prefetch`)

## What it is

`--ple-prefetch {off,prefill,always}` hands the rows that a per-layer-embedding (PLE) gather is about
to read to the operating system before the gather runs. The default is `always`. The option is a
common option, so `llama-server`, `llama-cli` and the other common tools take it, and `llama-bench`
parses the same three values and applies one value to the whole run; `LLAMA_ARG_PLE_PREFETCH` sets
it in the environment. It does nothing for a model without a per-layer embedding table, and nothing
when that table is not in host memory.

Independently of the option, a CPU `GET_ROWS` on a tensor named exactly `per_layer_token_embd.weight`
runs on the threadpool when it gathers 256 rows or more. Every other `GET_ROWS` keeps the
single-thread path.

## Why it exists

The per-layer embedding table of Qwen3.8-Flash-Next is a 32.78 GiB Q5_0 tensor inside a 44.6 GB
shard. The loader maps it lazily and never reads it, so each gathered row is a demand page fault.
The model gathers 16 rows of 110 bytes per token, so a 512-token prompt ubatch is 8192 scattered
faults, 32 MiB of page-granular I/O for 880 KiB of wanted payload.

The cost of a fault depends only on whether the page is already in the page cache. With plenty of
free RAM it is about 2 us and the gather takes about 17 ms per ubatch. When another consumer of RAM
leaves about 26 GiB free, the same 8192 faults cost about 105 us each and the gather takes about
875 ms per ubatch, most of a prompt ubatch. The fault count does not change; only its price does.

Upstream keeps `GET_ROWS` on one thread because extra threads cost more than they save for the
small gathers of a GPU-offloaded graph. That holds for every other gather, but here one thread means
one outstanding fault at a time. Two changes address it: the big gather is spread over the
threadpool so the faults are concurrent, and the merged row ranges are handed to the operating system
before the gather runs so it can read them together. The second is `PrefetchVirtualMemory` on
Windows and `madvise(MADV_WILLNEED)` on POSIX; where neither exists the call does nothing.

Both are hints about page residency, not about arithmetic. The gathered rows are disjoint per thread,
so the output bytes are the same as on the single-thread path.

## Thresholds

| value | prefetch applies to | effect |
|---|---|---|
| `off` | nothing | the gather is still parallel from 256 rows on |
| `prefill` | gathers of 256 rows or more, 16 tokens or more of this model | prompt ubatches only |
| `always` (default) | gathers of 16 rows or more, one decoded token | prompt and decode |

256 rows is where the prompt measurement settled. 16 rows is one decoded token; it is the default
because it wins on decode and costs nothing on prompt processing.

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. Model:
Qwen3.8-Flash-Next UD-Q4_K_XL, every routed expert in host memory (`-ncmoe 999`) with host-direct on
(`GGML_CUDA_HOST_DIRECT=1`, `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`), `--load-mode none`. `llama-bench
-p 512 -n 128 -r 1 -b 512 -ub 512 -fa on -t 16`, depths `8192,0,8192` in one model load with the
first pair discarded as warm-up:

| `--ple-prefetch` | PP512 @ 0 | TG128 @ 0 | PP512 @ 8192 | TG128 @ 8192 |
|---|---:|---:|---:|---:|
| `off` | 504.13 | 19.36 | 470.86 | 18.83 |
| `always` | 538.95 | 19.99 | 538.55 | 19.46 |

In that configuration free RAM is ample and the page cache absorbs most of the fault cost; the
parallel gather, which both rows have, is what keeps the gather short there. The prefetch is worth
+7 % of prompt throughput and +3 % of decode at depth 0, and it removes the prompt slowdown at
depth 8192. A configuration that leaves little free RAM, where the fault price is 50x higher, was
not measured with `llama-bench`; the 875 ms per ubatch above is what the gather cost there before
this change.

## Status

The Win32 call is resolved with `GetProcAddress`, so the link line is unchanged and an older host
simply gets no prefetching. The POSIX path uses `madvise(MADV_WILLNEED)` per range and is not
measured.

## Revision

Every number above comes from the binary built from this commit on ranma `ccd8fd1e9` (upstream
llama.cpp `093a2f86c`, ggml-org master of 2026-09-14, plus the fork's earlier commits). The exact
commits are kept on the dated snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that
branch is never rebased.
