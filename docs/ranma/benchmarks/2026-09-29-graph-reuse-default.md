# Graph reuse per batch shape turned on by default

- Release: `ranma_20261001` (measured on a development build of its series, before the release)
- Upstream base: `ed7ac35e1`
- Measured: 2026-09-29
- Method: [method.md](method.md); server settings of [2026-09-28-release.md](2026-09-28-release.md).

Measured on 2026-09-29, after the rows of [2026-09-28-release.md](2026-09-28-release.md) (the build also carried a
per-round speculative log and diagnostics that were off). ranma_20261001 turns on
`LLAMA_GRAPH_REUSE_SHAPES` (24) and raises
the HIP bound of `GGML_CUDA_GRAPH_PER_SHAPE` from 16 to 32
([graph-runtime.md](../graph-runtime.md)); the rows set both by environment on the
same build. English roleplay, MTP
`--spec-draft-n-max 7 --spec-draft-p-min 0.7`, joint cache on, expert cache
exclusive 20480 MiB, unlimited host tier, warm profile, one run per row with a
300 s rest:

| model | previous defaults (off, 16) | new defaults (24, 32) | change | peak private GiB | replies |
|---|---:|---:|---:|---:|---|
| Qwen3.8-Flash-Next UD-Q4_K_XL, prefill swap on | 46.52 | 51.67 | +11.1 % | 85.43 -> 86.40 | 17/17 identical |
| DeepSeek V4 Flash UD-IQ3_XXS | 28.97 | 31.45 | +8.6 % | 88.35 -> 89.20 | 17/17 identical |

The acceptance is the same in both columns (Qwen 77.9 %, DeepSeek 68.8 %) and
the peak dedicated VRAM too (28.4 and 28.7 GiB). The rows of the main tables of
[2026-09-28-release.md](2026-09-28-release.md) ran with the previous defaults.
