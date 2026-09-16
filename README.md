# RANMA.cpp

**RA**deon **N**arrative lla**MA**.cpp — a Radeon-optimized [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for roleplay and narrative inference.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Upstream](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-lightgrey.svg)](https://github.com/ggml-org/llama.cpp)

> This is a hobby project maintained by a single developer.
> **Issues are welcome. Pull requests are not accepted.** See [Project policy](#project-policy).

## What this fork is

RANMA.cpp is llama.cpp with a small, curated set of changes aimed at one workload:
**interactive roleplay and narrative generation** on consumer AMD Radeon GPUs, running on Windows.

That workload looks different from the general-purpose serving that upstream optimizes for:

- One user (or a handful of slots), not dozens of concurrent requests.
- Long, growing multi-turn contexts where most of the prompt was already seen last turn.
- Decode throughput at long context matters more than batch prefill throughput.
- Sessions stay open for hours with idle gaps between turns.
- Large MoE models that do not fit in VRAM, or even in RAM.

Some of the optimizations that pay off here trade away multi-request throughput or
generality, so they are not appropriate for upstream. Others are general improvements
and may be submitted upstream. This fork is where both kinds live together in a tested state.

It is not a general replacement for llama.cpp. If you serve many users, use upstream.

## Roadmap

The first phase is porting. The features come from a private experimental fork that the
maintainer has been running for personal use; each one is cleaned up, reshaped to fit
upstream's code layout, and measured again before it lands here. Once that backlog is
ported, the fork keeps going in the same direction: changes for an individual user
running llama.cpp on Windows with a Radeon GPU, measured on that setup.

## Target environment

| | Primary target |
|---|---|
| OS | Windows 11 |
| GPU | AMD Radeon RDNA4, `gfx1201` (e.g. Radeon RX 9070 series) |
| Backend | HIP / ROCm (`GGML_HIP=ON`) |
| Use case | `llama-server` driving a roleplay client, single or few slots |

All changes are developed and tested only on this environment. Some rely on Windows APIs
or on gfx1201-specific behavior; on other platforms they may have no effect or may
misbehave. Other platforms and backends are not tested and not supported by this fork.

## Principles

1. **Follow upstream closely.** The `master` branch is a pristine mirror of
   `ggml-org/llama.cpp`. The `ranma` branch is `master` plus the curated patch set,
   and is rebased onto upstream as often as practical. That means the history of
   `ranma` is rewritten regularly: commit hashes change on every rebase, and a commit
   whose reason has disappeared (upstream landed an equivalent, or the feature stopped
   paying for itself) is dropped rather than carried along. Do not build long-lived
   work on `ranma` hashes. Dated snapshot branches (`ranma_YYYYMMDD`) are never
   rebased and keep the exact commits that the `docs/ranma/` pages cite.
2. **Curate, do not accumulate.** Every change must be explainable in a few sentences,
   measurable on the target workload, and small enough to carry across upstream updates.
   Features that stop paying for themselves are removed.
3. **Be honest about trade-offs.** Each feature documents what it costs: concurrency,
   VRAM, generality, or complexity.
4. **Upstream what belongs upstream.** Changes that are general improvements may be
   submitted to llama.cpp rather than kept only here.

## Changes over upstream

- **Host-direct MoE weights (HIP)** — `MUL_MAT_ID` kernels read host-resident expert weights in place over PCIe
  instead of copying them per op or computing them on the CPU. Off by default. Recommended profile for a MoE model
  whose experts live in system RAM (`-ncmoe`):

  ```
  GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512
  ```

  `MAX_BATCH` should be at least the ubatch size (`-ub`, default 512). Measured on a Radeon AI PRO R9700 with
  Qwen3.8-Flash-Next UD-Q4_K_XL: decode 45-61 % faster and prompt processing 28-51 % faster than the upstream copy
  path, on both a 16 GiB and a 32 GiB expert placement. Details, limits and the full tables:
  [docs/ranma/host-direct-moe.md](docs/ranma/host-direct-moe.md).

- **GPU heartbeat (llama-server)** — while the server is idle, and while the model is being torn down, a worker
  thread records one GPU event on every device that holds model buffers, once per interval. On Windows the video
  memory manager evicts a process' whole VRAM residency after about 10 s without a submission, which costs several
  seconds of paging on the next request and parks the evicted copy in system RAM until then. Off by default.
  Recommended value:

  ```
  --gpu-heartbeat-seconds 5
  ```

  Measured on a Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL and 26 GiB resident in VRAM: the first token
  after a 15-60 s idle gap costs 3-16 s without the heartbeat and 0.2 s with it, available system RAM no longer drops
  by 25 GiB while idle, and a shutdown after an idle gap takes 5.7 s instead of 11.8 s. Details and limits:
  [docs/ranma/gpu-heartbeat.md](docs/ranma/gpu-heartbeat.md).

- **RDNA4 small-batch matmul (HIP)** — MMVQ computes four weight rows per block for 3..8 activation columns, and
  dense Q4_K/Q5_K/Q6_K matmuls switch to MMQ from 5 rows instead of 9. On a Radeon AI PRO R9700 with gemma-4-31B
  Q4_K_M one decode call with 3 / 4 / 5 / 8 rows went from 48 / 57 / 68 / 99 ms to 43 / 48 / 50 / 51 ms, so the
  cost per call no longer drops at 9 rows; multi-slot decode with 4 and 8 sequences is 18-23 % and 76-93 % faster.
  Single-row decode, prompt processing and MoE expert matmuls are unchanged. No switch: the entries are
  architecture tables like upstream's Ada/Blackwell/CDNA ones. Details and the numerics discussion:
  [docs/ranma/rdna4-small-batch.md](docs/ranma/rdna4-small-batch.md).

- **Per-position draft thresholds** — `--spec-draft-p-min` takes one probability per draft position and a new
  `--spec-draft-p-continue` keeps a token in the draft but stops drafting after it. Dropping a token saves one
  verification row, which only costs something while the batch is still in the MMVQ path; stopping saves one
  draft-model step, which costs the same at every position. Defaults unchanged. Recommended profile for
  gemma-4-31B with the Gemma4 MTP head on a Radeon AI PRO R9700:

  ```
  --spec-draft-p-min 0.33,0.6,0.6,0 --spec-draft-p-continue 0.9 --spec-draft-n-max 15
  ```

  Measured on 88 SPEED-Bench prompts: 74.1 t/s decode against 28.0 without speculation, 69.5 with no thresholds
  and 72.6 with the best single `p-min`. Details, the cost curve and the reasoning behind the values:
  [docs/ranma/spec-draft-thresholds.md](docs/ranma/spec-draft-thresholds.md).

- **Per-layer embedding prefetch and gather** - `--ple-prefetch {off,prefill,always}` (default `always`) hands the
  rows a per-layer-embedding gather is about to read to the operating system before the gather runs, and the gather
  of that one tensor runs on the threadpool. The table is mapped lazily, so without this every gathered row is a
  page fault on one thread. Measured on a Radeon AI PRO R9700 with Qwen3.8-Flash-Next UD-Q4_K_XL, experts in host
  memory: prompt processing 504 -> 539 t/s and decode 19.4 -> 20.0 t/s at depth 0, with ample free RAM; the fault
  price grows 50x when RAM is scarce. Details: [docs/ranma/ple-prefetch.md](docs/ranma/ple-prefetch.md).

Each feature that lands gets a line here and a page under `docs/ranma/` describing its
rationale, measured effect, and trade-offs.

## Building

RANMA.cpp builds exactly like upstream. For the primary target, follow the HIP section of
[docs/build.md](docs/build.md) on Windows: install the ROCm SDK, open an
*x64 Native Tools Command Prompt for VS*, then:

```bat
set PATH=%HIP_PATH%\bin;%PATH%
cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Prebuilt Windows binaries may be published on the releases page once the first
curated features land.

## Project policy

- **Issues**: welcome. Bug reports on the target environment and reproducible
  performance regressions are most useful. Feature requests are read but not promised.
- **Pull requests**: not accepted. This is a one-person hobby project and there is no
  review bandwidth. If you have a fix, open an issue describing it; if it is general,
  consider sending it to upstream llama.cpp instead.
- **Sponsorship**: may be enabled in the future.
- **Upstream contributions**: selected features from this fork may be submitted to
  llama.cpp under the upstream contribution rules.

## Branches

| Branch | Purpose |
|---|---|
| `master` | Unmodified mirror of upstream `ggml-org/llama.cpp` `master`. Never commit here. |
| `ranma` | Curated fork line: `master` + accepted features. This is the default branch. |
| `feature/*` | One curated feature being prepared for `ranma`. |
| `ranma_YYYYMMDD` | Snapshot of `ranma` on that date. Never rebased or force-pushed, so the revisions cited in `docs/ranma/` still resolve after `ranma` has moved on. |

Upstream is tracked as the git remote `upstream`. When `ranma` is rebased, upstream
changes to this README and other fork-owned files are reviewed by hand and applied or
dropped as appropriate.

## Upstream documentation

Everything not covered above is unchanged from llama.cpp:

- [Build guide](docs/build.md) · [Install](docs/install.md) · [Docker](docs/docker.md)
- [Supported models](docs/models.md) · [Multimodal](docs/multimodal.md)
- [llama-server](tools/server/README.md) · [Function calling](docs/function-calling.md)
- [Speculative decoding](docs/speculative.md) · [Multi-GPU](docs/multi-gpu.md)

## License and credits

RANMA.cpp is distributed under the [MIT License](LICENSE), the same license as llama.cpp.

All of the heavy lifting is the work of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
and its contributors. This fork exists only to carry a narrow set of workload-specific
changes on top of it.
