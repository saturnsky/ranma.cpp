# RANMA.cpp

**RA**deon **N**arrative lla**MA**.cpp — a Radeon-optimized [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for roleplay and narrative inference.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Upstream](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-lightgrey.svg)](https://github.com/ggml-org/llama.cpp)

> This is a hobby project maintained by a single developer.
> **Issues are welcome. Pull requests are accepted on the terms of [CONTRIBUTING.md](CONTRIBUTING.md).**
> See [Project policy](#project-policy).

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

It can also serve one person's own development work in a single session, such as a coding assistant
on the same machine.

It is not a general replacement for llama.cpp. The expert cache does not change its placement while a
request is processed: it installs a new plan only when every slot is idle, so when several sessions send
requests in turn it rarely gets the chance. If you serve many users, use upstream.

## Main features

- **Expert cache** - for a MoE model whose routed experts do not fit in VRAM, the server profiles which
  experts the router selects and keeps the most valuable set in a VRAM budget (`--expert-l1-mib`); the
  other experts are read from host memory over PCIe, and with `--expert-l2-mib` from the GGUF file when
  they do not fit in RAM either. Needs host-direct and `--load-mode none`.
  [docs/ranma/expert-cache.md](docs/ranma/expert-cache.md),
  [docs/ranma/expert-cache-l2.md](docs/ranma/expert-cache-l2.md)
  - Give the slot count with `-np` (`-np 1` for one user). Without it the server runs four slots: the
    state kept per slot grows with them (for Qwen3.8-Flash-Next at a 256K context the recurrent state
    and the compute buffers grow by several GiB), which can overflow VRAM, and a finite host tier is
    refused with `finite L2 needs positive batch/parallel bounds`.
- **Smart MTP draft length (`--spec-smart`)** - with `--spec-type draft-mtp`, llama-server chooses the
  draft length at every step from the measured verification time per width and a calibrated acceptance
  of the draft probabilities, without per-model thresholds. On by default with `draft-mtp`;
  `--spec-smart-store PATH` keeps the estimates across restarts.
  [docs/ranma/spec-smart.md](docs/ranma/spec-smart.md)
- **EXL3 weights** - ExLlamaV3 EXL3 quantizations run from GGUF files, on the CPU and on HIP (GEMV for
  decode, WMMA GEMM on RDNA4 for prompts), with host-direct expert banks and the expert cache. They need
  GGUF files in this fork's format: an ordinary GGUF or an EXL3 safetensors checkpoint is not read
  directly. Take the converted files from Hugging Face
  ([Qwen3.8-Flash-Next](https://huggingface.co/SaturnHeaven/Qwen3.8-Flash-Next-RANMA-EXL3-GGUF),
  [DeepSeek-V4-Flash-0731](https://huggingface.co/SaturnHeaven/DeepSeek-V4-Flash-0731-RANMA-EXL3-GGUF)).
  For another EXL3 model, convert its EXL3 checkpoint with `convert_hf_to_gguf.py`. Several
  architectures have been converted and run ([checked models](docs/ranma/exl3.md#checked-models)), but
  not every model is guaranteed to work yet. These files use tensor types of this fork and do not open
  in upstream llama.cpp, LM Studio, Ollama or other llama.cpp-based tools.
  [docs/ranma/exl3.md](docs/ranma/exl3.md)

Every other change, one line each with its page: [docs/ranma/README.md](docs/ranma/README.md).

## Benchmarks

The benchmark of a release is the last commit of its series
([docs/ranma/releases.md](docs/ranma/releases.md#release-cycle)).

## Roadmap

Planned, not done yet. Nothing here is promised or scheduled.

- **DeepSeek V4 Flash**: try further performance work. It may not pay off.
- **DeepSeek V4.1 Flash**: take the upstream pull request for this model while it is not merged yet,
  then add EXL3 support on top. If upstream merges a different implementation, this fork follows
  upstream, and GGUF files made for the earlier one, EXL3 GGUF files in particular, may stop loading and
  need to be converted again.
- **Experimental RDNA3 and NVIDIA (CUDA) builds**: open the paths that do not depend on RDNA4
  instructions and check only that they compile. Whether they work is not guaranteed; reports and pull
  requests are welcome.

## Target environment

| | Primary target |
|---|---|
| OS | Windows 11 |
| GPU | AMD Radeon RDNA4, `gfx1201`; developed and measured on a Radeon AI PRO R9700 |
| Backend | HIP / ROCm (`GGML_HIP=ON`) |
| Use case | `llama-server` driving a roleplay client, single or few slots |

The maintainer develops and tests every change only on this environment. Some changes rely on Windows
APIs or on gfx1201-specific behavior; on other platforms they may have no effect or may misbehave. Other
GPUs, backends and operating systems may work, but nothing is guaranteed there: releases are made
without testing them. Reports and pull requests are welcome
([CONTRIBUTING.md](CONTRIBUTING.md#other-architectures)).

### Features by GPU

| feature | RDNA4 (gfx1201) | RDNA3 (gfx1100/1101/1102) |
|---|:---:|:---:|
| Host-direct MoE weights | ✔ | ◇ ranma_20261005 |
| Expert cache, VRAM tier | ✔ | ◇ ranma_20261005 |
| Expert cache, finite host tier with file backing | ✔ | ◇ ranma_20261005 |
| Smart MTP draft length | ✔ | ? |
| Qwen sparse attention, selected cells only | ✔ | ? |
| DeepSeek V4 selected-cell attention | ✔ | ? |
| DeepSeek V4 graph ops (hyper-connection coefficients, KV compressor) | ✔ | ? |
| EXL3 decode (GEMV) | ✔ | ○ |
| EXL3 prompt processing (WMMA GEMM) | ✔ | ✕ (prompts run on the GEMV ○) |
| Qwen3.8 hyper-connection fusion | ✔ | ○ |

✔ tested by the maintainer at the current release. ◇ reported by a user who tested it on their own machine at
the release named; not tested by the maintainer. ○ the HIP device code compiles for gfx1100 (the EXL3 kernels
also for gfx1101 and gfx1102); not run. ✕ not implemented for that GPU yet; it takes a slower general path.
? not confirmed. Optimizations that only make sense with RDNA4 instructions or RDNA4 tuning, such as the
RDNA4 matmul selection and the shared expert unit, are not listed.

## Principles

1. **Follow upstream closely.** The `master` branch is the unmodified upstream
   `ggml-org/llama.cpp` commit that the latest release is based on. The `ranma` branch
   is the latest release: `master` plus the curated patch set, rebuilt for every release
   on a recent upstream. That means the history of `ranma` is rewritten regularly: commit
   hashes change from release to release, and a commit whose reason has disappeared
   (upstream landed an equivalent, or the feature stopped paying for itself) is dropped
   rather than carried along. Do not build long-lived work on `ranma` hashes. Every
   release is a tag (`ranma_YYYYMMDD`) that never moves and keeps the exact commits that
   the `docs/ranma/` pages cite. See [docs/ranma/releases.md](docs/ranma/releases.md).
2. **Curate, do not accumulate.** Every change must be explainable in a few sentences,
   measurable on the target workload, and small enough to carry across upstream updates.
   Features that stop paying for themselves are removed.
3. **Be honest about trade-offs.** Each feature documents what it costs: concurrency,
   VRAM, generality, or complexity.
4. **Upstream what belongs upstream.** Changes that are general improvements may be
   submitted to llama.cpp rather than kept only here.

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
- **Pull requests**: accepted on conditions, against `ranma_upstream`. There is no promise of
  review, merge or response time; the pull request carries its own verification and test
  environment; an accepted change may be squashed, reordered, rewritten or dropped in a later
  release. See [CONTRIBUTING.md](CONTRIBUTING.md#pull-requests).
- **Sponsorship**: [GitHub Sponsors](https://github.com/sponsors/saturnsky). It does not buy review, features, or support.
- **Upstream contributions**: selected features from this fork may be submitted to
  llama.cpp under the upstream contribution rules.

## Branches

| Ref | Purpose |
|---|---|
| `ranma` | The latest release. This is the default branch. |
| `ranma_YYYYMMDD` | A release, as a tag that never moves, so the revisions cited in `docs/ranma/` still resolve after `ranma` has moved on. The three latest releases are also kept as branches of the same name. |
| `master` | The unmodified upstream `ggml-org/llama.cpp` `master` commit that the latest release is based on. Never commit here. |
| `ranma_upstream` | The latest release rebased onto a newer upstream commit with its overlaps with upstream resolved: the baseline of further work, the starting point of the next release and the target of pull requests. Not benchmarked again. |
| `features/*` | Changes prepared for upstream pull requests. |

How a release is made and how far each ref is verified: [docs/ranma/releases.md](docs/ranma/releases.md).

Upstream is tracked as the git remote `upstream`. When the series is rebased, upstream
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

The EXL3 format support follows [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT License,
turboderp); see [NOTICE](NOTICE).
