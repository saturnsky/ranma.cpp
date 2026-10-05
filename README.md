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

The release ranma_20261005 is based on upstream `42d958167`. Its benchmark records keep the release each number was
measured at: for ranma_20261005 Qwen3.8-Flash-Next in EXL3 was measured against UD-Q4_K_XL; the MTP draft length is
from ranma_20261001, the ratios against upstream from ranma_20260928 and the full `llama-bench` set from
ranma_20260922. Everything was measured on the Radeon AI PRO R9700;
[docs/ranma/benchmarks](docs/ranma/benchmarks/README.md) has every number with its release, and
[the method](docs/ranma/benchmarks/method.md).

EXL3 against UD-Q4_K_XL, measured for ranma_20261005: Qwen3.8-Flash-Next EXL3 3.05 bpw and 4.05 bpw on two systems,
the R9700 with 128 GB (exclusive expert cache of 20480 MiB, unlimited host tier) and the RX 9070 XT emulation with
64 GB (3072 MiB, host tier of 40960 MiB). `llama-bench` TG128 at depth 0, warm with the prefill swap, and server
decode in the five scenarios with MTP and `--spec-smart`, warm, against UD-Q4_K_XL on the same system
([record](docs/ranma/benchmarks/2026-10-05-exl3.md), with the 64 GB rows, the prompt rates, cold and memory).

| model | system | TG128 @0 t/s | TG128 @0 against UD-Q4_K_XL | server decode with MTP against UD-Q4_K_XL |
|---|---|---:|---:|---:|
| EXL3 3.05 bpw | R9700, 128 GB | 53.5 | +8.6 % | +17.2 to +23.1 % |
| EXL3 4.05 bpw | R9700, 128 GB | 49.6 | +0.7 % | +3.9 to +10.0 % |
| UD-Q4_K_XL | R9700, 128 GB | 49.3 | | |
| EXL3 3.05 bpw | RX 9070 XT emulation, 64 GB | 43.2 | +20.5 % | +50.6 to +85.9 % |
| EXL3 4.05 bpw | RX 9070 XT emulation, 64 GB | 37.0 | +3.3 % | +15.4 to +28.1 % |
| UD-Q4_K_XL | RX 9070 XT emulation, 64 GB | 35.8 | | |

Decode against upstream, measured at ranma_20260928 against its upstream base: the server scenarios that the project
runs (roleplay in four languages and coding, one slot, greedy decoding), release warm against upstream. Upstream keeps
the routed experts of the first 35 layers in host memory (`-ncmoe 35`; `-ncmoe 36` with the DeepSeek MTP head on the
GPU); the release runs the exclusive expert cache with a VRAM budget of 20480 MiB (18432 MiB with the DeepSeek MTP
head outside the cache) and an unlimited host tier
([record](docs/ranma/benchmarks/2026-09-28-release.md), with the t/s, the cold and 64 GB rows and the prompt times).

| model | English roleplay | coding | Korean roleplay | Japanese roleplay | Chinese roleplay |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash UD-IQ3_XXS, MTP n1 | x2.79 | x2.52 | x2.89 | x2.95 | x2.92 |
| DeepSeek V4 Flash UD-IQ3_XXS, no MTP | x2.97 | x2.70 | x3.07 | x3.11 | x3.09 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | x2.78 | x2.57 | x2.78 | x2.79 | x2.79 |

MTP draft length, measured for ranma_20261001: decode with no MTP, with the single-argument values tuned on this
machine for these two models (Qwen3.8-Flash-Next `--spec-draft-n-max 2 --spec-draft-p-min 0`, DeepSeek V4 Flash
`--spec-draft-n-max 1 --spec-draft-p-min 0`), and with `--spec-smart`, the default of `draft-mtp` now, from a cold
start and from a warm store. Six scenarios per model (coding, roleplay in four languages, a mixed scenario), the MTP
head on the GPU in the joint expert cache ([record](docs/ranma/benchmarks/2026-10-01-mtp-smart.md)).

| model | English roleplay t/s: no MTP / single / smart cold / smart warm | single against no MTP | smart cold against single | smart warm against single |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next UD-Q4_K_XL | 46.10 / 57.13 / 56.99 / 57.54 | +19.7 to +29.5 % | -4.3 to +0.6 % | -1.9 to +2.6 % |
| DeepSeek V4 Flash UD-IQ3_XXS | 31.17 / 35.00 / 34.07 / 34.55 | +9.8 to +14.6 % | -3.8 to +1.0 % | -3.2 to +1.5 % |

Smart needs no per-model values and stays within a few percent of the tuned values; on DeepSeek V4 Flash it is 1.3 to
3.8 % slower in every roleplay scenario. `--no-spec-smart` or the thresholds give the single-argument rule back.

`llama-bench` PP512 / TG128 in t/s at depth 0, 8192 and 65536, measured at ranma_20260922 against its upstream base on
two systems: the R9700 (32 GiB) with 128 GiB of host memory, and an RX 9070 XT (16 GiB) with 64 GiB, emulated on the
R9700 by the placement and the host-tier budget. Upstream runs `-ncmoe 35`; ranma_20260922 runs the exclusive expert
cache (20480 MiB on the R9700, 3072 MiB for Qwen and 4096 MiB for DeepSeek on the emulated RX 9070 XT) with prefill
swap where that is faster. The last column is the peak VRAM / host commit of the process in GiB
([record](docs/ranma/benchmarks/2026-09-22-full-set.md), with the 32 GB rows and Gemma 4 31B).

| model | system | @0 | @8192 | @65536 | VRAM / host |
|---|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next UD-Q4_K_XL | upstream, R9700, 128 GB | 335 / 15.9 | 316 / 15.2 | 307 / 13.2 | 29.9 / 80.7 |
| | ranma_20260922, R9700, 128 GB | 1014 / 49.7 | 949 / 47.9 | 692 / 44.8 | 29.1 / 78.2 |
| | ranma_20260922, RX 9070 XT emulation, 64 GB | 625 / 36.7 | 596 / 38.2 | 480 / 35.2 | 12.0 / 48.8 |
| DeepSeek V4 Flash UD-IQ3_XXS | upstream, R9700, 128 GB | 248 / 9.8 | 209 / 9.7 | (measured to 8192) | 27.8 / 98.7 |
| | ranma_20260922, R9700, 128 GB | 310 / 31.8 | 255 / 30.9 | 108 / 28.1 | 27.9 / 90.3 |
| | ranma_20260922, RX 9070 XT emulation, 64 GB | 129 / 18.8 | 128 / 20.3 | 78 / 18.9 | 11.8 / 49.0 |

## Roadmap

Planned, not done yet. Nothing here is promised or scheduled.

- **DeepSeek V4 Flash**: try further performance work. It may not pay off.
- **GLM 5.3 and DeepSeek V4.1 Flash**: take the upstream pull requests for these models that are not
  merged yet, then add EXL3 support on top. If upstream merges a different implementation, this fork
  follows upstream, and GGUF files made for the earlier one, EXL3 GGUF files in particular, may stop
  loading and need to be converted again.
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
RDNA4 matmul selection and the shared-expert fold, are not listed.

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

The EXL3 format support follows [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT License,
turboderp); see [NOTICE](NOTICE).
