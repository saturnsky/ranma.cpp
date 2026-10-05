# Contributing to RANMA.cpp

RANMA.cpp is a hobby fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
maintained by a single developer. The contribution policy is deliberately narrow.

## Issues

Issues are welcome. The most useful reports are:

- Bugs or crashes on the target environment (Windows 11, AMD Radeon gfx1201, HIP backend).
- Reproducible performance regressions on that environment, ideally with `llama-bench`
  numbers before and after.
- Incorrect behavior that is specific to this fork and does not reproduce on upstream
  llama.cpp at the same base commit.

Feature requests are read but not promised. Problems that also reproduce on upstream
should be reported to upstream instead.

When reporting, please include the fork commit (`git rev-parse HEAD`), the upstream base
commit it was rebased onto, and the environment as listed under
[Your test environment](#your-test-environment) below (at least the SDK and driver versions).

## Pull requests

Pull requests are welcome. RANMA.cpp is maintained by one person with limited time, and
the way this fork is rebuilt for every release (see [releases](docs/ranma/releases.md))
means that some things cannot be promised. The terms below say what can and cannot be.
Please read them before you start; opening a pull request means you accept them.

### No promise of review or merge

- There is no response time. A pull request may wait a long time, and it may be closed
  without review or without a detailed reason.
- For anything larger than a small fix, open an issue first and describe the change.
  This avoids work on something that cannot be taken.
- If the change is a general improvement to llama.cpp, please submit it to
  [upstream](https://github.com/ggml-org/llama.cpp) as well (or instead). Changes that
  land upstream reach this fork on the next rebase.

### Target branch

Open pull requests against `ranma_upstream`, the latest release rebased onto a recent
upstream commit. Release snapshots are tags and are never changed.

### History is not preserved

Accepted improvements are meant to stay, and the maintainer will try to carry them forward.
This fork, however, is rebuilt for every release: commits are squashed, split, reordered and
rewritten, and the release is published as a new tag. By opening a pull request you agree
that, if your change is accepted:

- it may be squashed with other commits, split, reordered or rewritten, so your commits
  and their hashes may not survive as separate commits;
- it may be changed later, or removed in any later release, for example when it is found to
  cause a regression or to conflict with other work;
- credit is kept on a best-effort basis, through a `Co-authored-by:` line or the release
  notes, without a guarantee of either.

### Verification is your job

The maintainer cannot reproduce most environments, so a pull request has to carry its own
evidence. Every pull request must include:

- **Correctness**
  - `test-backend-ops` results for every operation the change touches, on your hardware.
  - Whether the generated output changes. Compare the generated text, and if possible the
    token probabilities, before and after with a fixed seed and a short prompt. If the
    output changes, explain why (for example a different reduction order).
- **Performance**, if the change claims a speedup or may affect speed:
  - `llama-bench` (or `llama-server` request timings) before and after, built from the same
    base commit, with the number of repetitions and the run order.
  - The full command lines and the raw output. Summaries alone are not enough.
- **Memory**: whether VRAM or host memory use changes, and by how much.
- **Models**: the model file names, quantization types and context sizes you used.

### Your test environment

Describe the environment of every result. All of the following are required:

- OS and build number (for example Windows 11 26200, or the distribution and kernel version)
- CPU model
- RAM: capacity, speed and channel configuration
- GPU model and VRAM size; overclocking, undervolting or power limits if any
- PCIe: the generation and lane count the GPU actually negotiated (not the slot rating),
  and whether Resizable BAR is on
- GPU driver version
- SDK and compiler versions (for example ROCm/HIP SDK, CUDA toolkit), and the CMake options
- Storage type, if the change touches the expert cache SSD tier
- The fork commit and the upstream commit it is based on

### Other architectures

Pull requests that add or improve support for other GPUs, backends or operating systems are
welcome. The maintainer will try to keep such support working when it does not get in the
way of the target environment, but cannot promise it:

- the change must not alter the behavior or speed of the target environment (Windows 11,
  Radeon gfx1201, HIP). Keep it behind a compile-time or runtime check of the architecture
  where possible;
- the maintainer tests only the target environment, and releases are made without testing
  any other one. CI is trimmed and does not cover other platforms either;
- support added this way may therefore break, or be removed, in a later release without
  notice. Fixes for such breakage are welcome in turn.

### AI-assisted contributions

Disclose whether you used AI tools to write the change. Whatever the tooling, you must
understand the change and have run the verification above yourself; results you did not
run are not accepted as evidence.

### License

Contributions are accepted under the license of this repository (MIT).

## Sponsorship

The fork can be sponsored through [GitHub Sponsors](https://github.com/sponsors/saturnsky).
Sponsorship does not buy review, features, or support.

## Coding conventions

Code in this fork follows the upstream llama.cpp
[coding and naming guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
so that selected changes can be submitted upstream without rework.

## AI usage

The maintainer uses AI-assisted tooling during development. Every line carried by this
fork is reviewed and tested by the maintainer on the target environment, and the
maintainer takes full responsibility for it. Changes submitted from this fork to upstream
follow upstream's AI usage policy and disclosure requirements.
