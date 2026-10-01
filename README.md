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

## Changes over upstream

Each user-visible change gets a line here and a page under `docs/ranma/` that describes what it is, when it
applies, how to switch it, and its limits.

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
| `ranma_upstream` | The latest release rebased onto a newer upstream commit, the starting point of the next release and the target of pull requests. Build-checked only. |
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
