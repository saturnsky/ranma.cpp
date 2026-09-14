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
| GPU | AMD Radeon RDNA4, `gfx1201`; developed and measured on a Radeon AI PRO R9700 |
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

Each user-visible change gets a line here and a page under `docs/ranma/` that describes what it is, when it
applies, how to switch it, and its limits.

### RDNA4 kernels (HIP)

- **Small-batch matmul dispatch** - four weight rows per MMVQ block for 3..8 activation columns and a
  per-type MMVQ/MMQ crossover, so the cost of a decode call no longer rises and then falls with the number of rows
  in it. This is the range of a speculative verification step and of a server that batches a few slots.
  [docs/ranma/rdna4-small-batch.md](docs/ranma/rdna4-small-batch.md)
- **Four rows per block at one column** - single-token decode of a wide matrix reads the activation once per
  four weight rows; chosen per call from the matrix size. Same page.
- **12-column tile attention for GQA-12 groups** - token generation on a model whose head group is a multiple
  of twelve reads each K/V head once instead of three times. `GGML_HIP_FATTN_GQA12=0` restores the upstream
  dispatch. Same page.
- **WMMA attention for 512-wide heads** - prompt processing of a 512/512 head at GQA 8 with an F16 KV cache
  takes the wide-tile MMA kernel. `GGML_HIP_PREFILL_WMMA=0` restores the upstream dispatch.
  [docs/ranma/rdna4-prefill.md](docs/ranma/rdna4-prefill.md)
- **Padded F16 BLAS for wide dense Q2_K/Q6_K/IQ2 matmuls** - wide prompt matmuls of those types convert both
  operands to F16 with a padded row pitch and run hipBLASLt instead of MMQ. Needs `ROCBLAS_USE_HIPBLASLT=1`;
  `GGML_HIP_PREFILL_BLAS=0` turns it off. Same page.
- **Fixed-order kernels for skinny F32 matmuls** - F32 products of up to 512 weight rows with more than eight
  activation columns (router, SSM and hyper-connection projections of a prompt batch) run on dedicated kernels
  instead of hipBLAS, whose per-process solution choice changed their speed and summation order from one process
  to the next. `GGML_CUDA_SKINNY_F32=0` restores hipBLAS. Same page.
- **Wider MoE column tiles** - the MMQ tile width of a `MUL_MAT_ID` is sized against three times the mean
  column count per expert, so popular experts re-read their weights less often. Bit-identical;
  `GGML_CUDA_MMQ_ID_NCOLS_OPT_SCALE=1` restores the upstream width. Same page.

### Server and common tools

- **GPU heartbeat (llama-server)** - `--gpu-heartbeat-seconds 5` records one GPU event per interval while the
  server is idle and while the model is freed, so that Windows does not evict the VRAM of the process between
  turns. Off by default. [docs/ranma/gpu-heartbeat.md](docs/ranma/gpu-heartbeat.md)
- **Per-position draft thresholds** - `--spec-draft-p-min` takes one probability per draft position, and
  `--spec-draft-p-continue` keeps a token in the draft but stops drafting after it. Defaults unchanged.
  [docs/ranma/spec-draft-thresholds.md](docs/ranma/spec-draft-thresholds.md)
- **Per-layer embedding prefetch** - `--ple-prefetch {off,prefill,always}` (default `always`) hands the rows of
  a lazily mapped per-layer embedding table to the operating system before the gather, and the gather of that
  tensor runs on the threadpool. [docs/ranma/ple-prefetch.md](docs/ranma/ple-prefetch.md)
- **Only the shards with lazy tensors are mapped** - with mmap loading off, the loader no longer maps model
  files that nothing reads through the mapping. Same page.

### Host-resident MoE experts

- **Host-direct MoE weights (HIP)** - `MUL_MAT_ID` kernels read host-resident expert weights in place over PCIe
  instead of copying them per op or computing them on the CPU. Off by default; the recommended profile for a
  model whose experts live in system RAM is `GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`
  (the second value at least the ubatch size). [docs/ranma/host-direct-moe.md](docs/ranma/host-direct-moe.md)
- **Expert cache** - `--expert-l1-mib N --expert-profile-dir DIR` gives the routed experts a VRAM budget: the
  server profiles which experts the router selects, plans the most valuable set for the budget and installs it
  at a request boundary. The budget decides the expert placement, so it replaces `--n-cpu-moe`. Needs host-direct
  and `--load-mode none`. [docs/ranma/expert-cache.md](docs/ranma/expert-cache.md)
- **The cache serves prompt processing, and installs are deltas** - MMQ reads the cache arena through the same
  slot tables as MMVQ, and an install moves only what changed between two plans.
  [docs/ranma/expert-cache-prefill.md](docs/ranma/expert-cache-prefill.md)
- **Exclusive mode (Windows)** - `--expert-cache-mode exclusive` gives every routed expert exactly one home, a
  VRAM slot or a host slot, so the budget is not a second copy of experts that also sit in RAM.
  [docs/ranma/expert-cache-exclusive.md](docs/ranma/expert-cache-exclusive.md)
- **Profile banks and the prefill swap** - prompt processing and generation are profiled into separate banks;
  `--expert-prefill-swap` additionally installs the prompt plan while a prompt is processed.
  [docs/ranma/expert-cache-banks.md](docs/ranma/expert-cache-banks.md)

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
- **Sponsorship**: [GitHub Sponsors](https://github.com/sponsors/saturnsky). It does not buy review, features, or support.
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
