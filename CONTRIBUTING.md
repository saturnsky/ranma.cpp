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
commit it was rebased onto, the ROCm SDK version, and the driver version.

## Pull requests

Pull requests are **not accepted** and will be closed without review. There is no review
bandwidth in a one-person project, and every change carried by this fork has to be small
enough to rebase onto upstream repeatedly, which is a judgment call the maintainer keeps.

If you have a fix, open an issue describing it. If the change is a general improvement
to llama.cpp, submit it to upstream; if it lands there, this fork picks it up on the next
rebase.

## Sponsorship

Sponsorship may be enabled in the future. It does not buy review, features, or support.

## Coding conventions

Code in this fork follows the upstream llama.cpp
[coding and naming guidelines](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md)
so that selected changes can be submitted upstream without rework.

## AI usage

The maintainer uses AI-assisted tooling during development. Every line carried by this
fork is reviewed and tested by the maintainer on the target environment, and the
maintainer takes full responsibility for it. Changes submitted from this fork to upstream
follow upstream's AI usage policy and disclosure requirements.
