# Benchmarks

The numbers of this fork against upstream. The first part holds this release: the server scenarios that the
project runs (roleplay in four languages and coding) for DeepSeek V4 Flash with and without MTP and for
Qwen3.8-Flash-Next, the `llama-bench` depth curves of both, Gemma 4 31B as the dense reference, and the options
of this release that are off by default. The second part is the full set, measured once at ranma_20260922 and
not repeated for every release. The feature pages describe what each change does and how to switch it; this page
holds the throughput. The expert cache that most rows use is described in [expert-cache.md](expert-cache.md).

## This release

Measured on 2026-09-27 and 2026-09-28 on the Radeon AI PRO R9700 with 128 GB of host memory. Three builds:

- **upstream** `53ed051ce`, the base of this release;
- **this release** `2b68c6055` (snapshot branch `ranma_20260928`) on upstream `53ed051ce`;
- **the previous release**, ranma_20260924 (`2bb543bdb` on upstream `ec5a12b85`), measured as its series rebased
  onto `53ed051ce` (`a60d2cc2e`), so that the difference between the two releases holds only the changes of this
  fork. It appears in the English roleplay rows with MTP, in one host tier row and in the Gemma rows.

Decode in t/s of the server scenarios, release warm against upstream:

| model | English roleplay | coding | Korean roleplay | Japanese roleplay | Chinese roleplay |
|---|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash UD-IQ3_XXS, MTP n1 | x2.79 | x2.52 | x2.89 | x2.95 | x2.92 |
| DeepSeek V4 Flash UD-IQ3_XXS, no MTP | x2.97 | x2.70 | x3.07 | x3.11 | x3.09 |
| Qwen3.8-Flash-Next UD-Q4_K_XL | x2.78 | x2.57 | x2.78 | x2.79 | x2.79 |

The tables below give the t/s behind these ratios, the cold rows (a process that starts with no profile), the
64 GB rows and the prompt times.

### Benchmark conditions

The state of every ranma switch in the main tables (the scenario tables, the `llama-bench` curves and the
Gemma rows). The joint cache, the L1 redraw and the hash early read were not enabled in the main tables; their
measured effects are in "Optional features" below (at 128 GB with MTP, and at 64 GB without MTP).

| switch | state in the main tables | default of this release |
|---|---|---|
| joint cache, `--expert-cache-draft` | off, passed explicitly in the DeepSeek MTP rows: the head stays on the GPU outside the cache (`-ngld 999`), the placement of the previous release; L1 18432 MiB in those rows, 20480 MiB in every other expert cache row | on |
| L1 redraw, `RANMA_EXPERT_L1_REDRAW_BENEFIT` / `_MOVE` | not set (off) | off |
| hash early read, `RANMA_EXPERT_HASH_EARLY` | `0` (off) | `0` |
| L2 staging ring size, `RANMA_EXPERT_L2_RING_FACTOR` | 1.0 (applies to the 64 GB rows) | 1.0 |
| prefill swap, `--expert-prefill-swap` | per row: on in the Qwen roleplay rows and the Qwen warm `llama-bench` row, off elsewhere | off |
| expert cache mode and budget | `--expert-cache-mode exclusive`, `--expert-l1-mib` as above, `--expert-l2-mib -1` (128 GB) or `40960` (64 GB) | - |
| host-direct, `GGML_CUDA_HOST_DIRECT` / `_MAX_BATCH` | `1` / `512` (required by the expert cache) | off |
| MTP with a finite host tier (the head outside the cache passes the finite-L2 gate) | used by the 64 GB MTP rows | on |
| one HIP graph per batch shape, `GGML_CUDA_GRAPH_PER_SHAPE` | on | on for HIP builds |
| one llama graph per batch shape, `LLAMA_GRAPH_REUSE_SHAPES` | not set (off) | off |
| pinned staging ring for graph inputs, `LLAMA_INPUT_UPLOAD_ASYNC` | on | on |
| MoE decode kernels (`GGML_CUDA_MMVQ_ID_DEDUP`, `_EXPERTS_FIRST`, `_FOLD_SHARED`, `GGML_CUDA_MMVQ_SHARE_Q8`) | defaults (on) | on |
| stable top-k ties, `GGML_CUDA_TOP_K_STABLE_TIES` (option for reproducible runs) | off | off |
| expert trace, `RANMA_EXPERT_TRACE` | not set | not set |
| server: `--ple-prefetch always`, `--gpu-heartbeat-seconds 5` | set, as in the project's launchers | - |

### Protocol

This release was measured on the R9700 with two host tiers: unlimited (a 128 GB system) and 40 GiB (a 64 GB
system). The RX 9070 XT emulation, the 32 GB system and the small-card placements are in the full set below; it
is the reference for estimating such a machine, with this section giving the change since.

Same machine as the full set below: Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16, Ryzen 9
7950X3D, 128 GiB DDR5-5600, Windows 11, `amdhip64_7.dll` 10.0.3581.0 shipped next to the binaries, -30 % power
limit and 0 mV voltage offset on the GPU, headless. The three builds were made with the same toolchain and
options, **all with OpenMP** (the `system_info` line of every server row reports `OPENMP = 1`). Earlier
measurements of this release from a build without OpenMP are not used on this page.

**Server rows.** Every row is one fresh `llama-server` process that replays one preset through the
OpenAI-compatible chat endpoint and exits: one slot, `--ctx-size 65536 -b 512 -ub 512 -fa on -ctk f16 -ctv f16
-t 16 --poll 0 --load-mode none --cache-ram 8192 --ctx-checkpoints 4 --jinja`, greedy sampling (temperature 0,
top-k 1, seed 1), at most 384 generated tokens per request, prompt caching on. Every model runs with its own GGUF
chat template, reasoning at the server's default and no LoRA. The environment of every row sets
`GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`; the ranma rows also pass
`--ple-prefetch always` and `--gpu-heartbeat-seconds 5`, as the project's launchers do. **Decode t/s** is the
sum of the generated tokens over the sum of the generation times of all requests of the preset; prompt time is
the sum of the prompt processing times.

**Presets.** The roleplay and coding presets come from the `roleplay` and `coding` categories of the
qualitative split of [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) (NVIDIA Evaluation
Dataset License). The copy used has its masked rows restored from their original sources. Per category the rows
are ordered by `sha256("ranma-mtp-method-v1:" + question_id)`. The first 10 form the warm-up preset and the next
10 the evaluation preset.
- English roleplay: 10 conversations with 17 requests, 7 from RoleBench (rolebench-eng) and 3 from CoSER.
- Coding: 10 conversations with 12 requests, 8 from HumanEvalPack and 2 from Spec-Bench.
- Korean, Japanese and Chinese (simplified) roleplay: machine translations of the same English rows made by an
  LLM (Claude Opus 5.5), with the same conversations and turns. A glossary pass fixed the repeated templates and
  names. Each language had a translation pass, a mechanical check (turn count, format marks) and an independent
  full review pass, all by the same model.
- The warm-up presets have other conversations of the same kind (20 roleplay requests, 10 coding requests). They
  are used only to record the warm profiles.

The dataset license does not allow redistribution, so the presets and the translations are not published. Taking
the same rows with the rule above and translating them yourself gives an approximate reproduction.

**Placement.** Upstream keeps the routed experts of the first 35 layers in host memory (`-ncmoe 35`, the
placement of the full set below); with the DeepSeek MTP head on the GPU it keeps one layer more (`-ncmoe 36`)
to stay under 30 GiB. The release rows run the exclusive expert cache (`--expert-cache-mode exclusive`) with a VRAM budget of
`--expert-l1-mib 20480`, or 18432 when the DeepSeek MTP head sits on the GPU outside the cache
(`-ngld 999 --expert-cache-draft off`), the arrangement the previous release also uses. The prefill swap is on in
the Qwen roleplay rows and off in the Qwen coding and all DeepSeek rows, as in the project's launchers (the
previous release refuses the swap together with speculative decoding). MTP is `--spec-type draft-mtp` with the
separate DeepSeek V4 Flash MTP head GGUF, `--spec-draft-n-max 1 --spec-draft-p-min 0`, on every build.

**128 GB and 64 GB.** As in the full set below, "128 GB" is the unlimited host tier (`--expert-l2-mib -1`) and
"64 GB" bounds the host tier to 40960 MiB, the emulated RAM minus 24 GB; the remaining experts are read from the
model file ([expert-cache-l2.md](expert-cache-l2.md)). The 64 GB rows exist for English roleplay and coding, warm
only. The page cache of the 128 GB machine still holds the model file, so their file tier cost is a lower bound.

**Cold and warm.** A cold row starts with an empty profile directory: the process loads the seeded placement and
warms itself from its own records, installing a new plan after the requests as a server does. A warm row starts
from its own copy of a profile that the warm-up preset of the same scenario recorded in a separate process (seed
rows, not in the tables). The DeepSeek warm profiles were recorded once per scenario with MTP and are shared by
the MTP and the no-MTP rows; the records are routing counts of the target model, which do not depend on the
build, the budget or the host tier. The English roleplay profile of DeepSeek was recorded earlier on 2026-09-27
by a build of this series without OpenMP.

**Rests.** Before every row a temperature gate waited until the GPU hotspot, VRAM and edge sensors had stopped
cooling (at most 1 C over the last 60 s) and were back near the idle baseline (hotspot +2 C, VRAM +1 C, edge
+2 C, CPU SoC +3 C), for at least 60 and at most 300 s. After the long server rows the sensors did not come back
inside those margins within the cap, so 71 of the 72 gates (main pass, Gemma rows and the 64 GB option rows
below) ended at 300 s. The rows with the previous release in English roleplay and the 128 GB option rows ran
earlier the same evening with a fixed 300 s rest. Scenarios alternate the order of their rows (upstream, cold, warm, 64 GB, then the reverse in the
next scenario).

### DeepSeek V4 Flash UD-IQ3_XXS, MTP n1

Decode t/s, and the draft acceptance (accepted over drafted tokens):

| scenario | upstream `-ncmoe 36` | release cold | release warm | release warm, 64 GB | cold / upstream | warm / upstream | MTP acceptance (upstream / release) |
|---|---:|---:|---:|---:|---:|---:|---:|
| English roleplay | 12.64 | 33.64 | 35.29 | 30.58 | x2.66 | x2.79 | 73.1 % / 72.5 % |
| Coding | 12.71 | 30.19 | 32.01 | 26.48 | x2.38 | x2.52 | 77.4 % / 79.5 % |
| Korean roleplay | 12.99 | 35.82 | 37.56 | - | x2.76 | x2.89 | 75.9 % / 75.5 % |
| Japanese roleplay | 12.72 | 36.03 | 37.57 | - | x2.83 | x2.95 | 72.5 % / 72.4 % |
| Chinese roleplay | 12.23 | 34.03 | 35.72 | - | x2.78 | x2.92 | 66.7 % / 67.6 % |

Prompt time in s (sum over the preset) and decode of the first request in t/s:

| scenario | prompt s: upstream | cold | warm | warm, 64 GB | first request: upstream | cold | warm | warm, 64 GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English roleplay | 52.8 | 35.5 | 35.2 | 51.2 | 12.6 | 20.7 | 36.8 | 32.0 |
| Coding | 16.7 | 11.0 | 10.9 | 16.8 | 13.2 | 22.9 | 36.3 | 31.3 |
| Korean roleplay | 81.6 | 57.8 | 57.5 | - | 13.5 | 21.3 | 36.9 | - |
| Japanese roleplay | 77.7 | 54.6 | 54.4 | - | 12.7 | 22.0 | 39.1 | - |
| Chinese roleplay | 53.3 | 36.0 | 35.7 | - | 11.6 | 20.1 | 35.6 | - |

**Against the previous release** (English roleplay, MTP n1, 128 GB). These rows ran in one block after a
reboot, before the main pass: four cold rows and one warm row of the previous release, and a cold and a warm row
of this release's code without the change that spreads equally scored experts across the layers of a class
in the plan. The release rows of the main pass above ran later the same evening, so the comparison
crosses two passes.

| row | decode t/s | first request | prompt s |
|---|---:|---:|---:|
| previous release, cold (4 rows) | 33.16 / 33.19 / 33.20 / 33.28 | 20.4 to 20.6 | 35.7 to 36.1 |
| previous release, warm | 35.24 | 36.8 | 35.2 |
| this release without the equal-score spread, cold | 33.77 | 21.0 | 35.5 |
| this release without the equal-score spread, warm | 35.18 | 36.7 | 35.3 |
| this release (main pass), cold | 33.64 | 20.7 | 35.5 |
| this release (main pass), warm | 35.29 | 36.8 | 35.2 |

Cold, this release decodes 1.5 % faster than the mean of the four previous-release rows (33.70 against 33.21 t/s,
the mean of the two release rows; +1.7 % and +1.3 % each). Warm, the two are equal (35.24 against 35.24 t/s;
-0.2 % and +0.1 %). All nine rows produce the same replies. The four previous-release cold rows lie within
0.3 % of each other, which bounds the repeat noise of this pass.

### DeepSeek V4 Flash UD-IQ3_XXS, no MTP

| scenario | upstream `-ncmoe 35` | release cold | release warm | release warm, 64 GB | cold / upstream | warm / upstream |
|---|---:|---:|---:|---:|---:|---:|
| English roleplay | 10.74 | 30.52 | 31.85 | 28.80 | x2.84 | x2.97 |
| Coding | 10.72 | 27.14 | 28.90 | 25.09 | x2.53 | x2.70 |
| Korean roleplay | 10.73 | 31.50 | 32.98 | - | x2.94 | x3.07 |
| Japanese roleplay | 10.66 | 31.64 | 33.16 | - | x2.97 | x3.11 |
| Chinese roleplay | 10.73 | 31.71 | 33.19 | - | x2.96 | x3.09 |

| scenario | prompt s: upstream | cold | warm | warm, 64 GB | first request: upstream | cold | warm | warm, 64 GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English roleplay | 50.1 | 32.6 | 32.3 | 46.2 | 10.7 | 19.5 | 32.2 | 28.9 |
| Coding | 16.0 | 10.3 | 10.1 | 15.5 | 10.7 | 19.3 | 31.6 | 28.6 |
| Korean roleplay | 78.7 | 55.1 | 54.9 | - | 10.6 | 19.4 | 33.6 | - |
| Japanese roleplay | 74.8 | 51.1 | 50.8 | - | 10.6 | 19.6 | 35.0 | - |
| Chinese roleplay | 51.5 | 33.6 | 33.3 | - | 10.7 | 19.5 | 34.1 | - |

In English roleplay at 64 GB the previous release decodes 28.38 t/s, this release 28.80 (+1.5 %; warm, no MTP,
the row of "Optional features" below).

### Qwen3.8-Flash-Next UD-Q4_K_XL

No MTP rows: MTP for Qwen is not part of this fork. Prefill swap on in the roleplay rows.

| scenario | upstream `-ncmoe 35` | release cold | release warm | release warm, 64 GB | cold / upstream | warm / upstream |
|---|---:|---:|---:|---:|---:|---:|
| English roleplay | 16.75 | 45.07 | 46.54 | 44.26 | x2.69 | x2.78 |
| Coding | 16.74 | 40.69 | 43.03 | 40.65 | x2.43 | x2.57 |
| Korean roleplay | 16.68 | 44.81 | 46.42 | - | x2.69 | x2.78 |
| Japanese roleplay | 16.64 | 44.67 | 46.36 | - | x2.68 | x2.79 |
| Chinese roleplay | 16.71 | 44.51 | 46.59 | - | x2.66 | x2.79 |

| scenario | prompt s: upstream | cold | warm | warm, 64 GB | first request: upstream | cold | warm | warm, 64 GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| English roleplay | 39.5 | 20.8 | 20.3 | 34.1 | 16.7 | 27.9 | 44.3 | 41.0 |
| Coding | 11.9 | 6.5 | 6.3 | 7.6 | 16.6 | 27.8 | 45.4 | 43.6 |
| Korean roleplay | 45.4 | 24.3 | 24.1 | - | 16.6 | 27.7 | 40.2 | - |
| Japanese roleplay | 44.5 | 24.2 | 23.8 | - | 16.5 | 27.9 | 40.8 | - |
| Chinese roleplay | 38.3 | 20.4 | 19.9 | - | 16.7 | 27.7 | 43.5 | - |

### `llama-bench` curves

The protocol of the full set below (`-d 65536,0,4096,8192,32768,65536`, the first pair discarded; upstream
DeepSeek `-d 8192,0,4096,8192`), one process per row, the same temperature gate as the server rows. The
release rows run the exclusive expert cache with 20480 MiB and an unlimited host tier. In `llama-bench`,
`--expert-cache cold` runs the whole process from the seeded placement and writes a profile; `warm` runs from
that profile ([expert-cache.md](expert-cache.md)). A cold `llama-bench` row therefore never installs a plan,
unlike a cold server row, and shows the seeded placement alone. The last column is the peak VRAM / private
bytes of the process in GiB (see "Memory" below).

Qwen3.8-Flash-Next UD-Q4_K_XL:

| | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 | VRAM / host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `53ed051ce`, `-ncmoe 35` | 351 | 327 | 347 | 427 * | 438 * | 16.6 | 16.3 | 16.1 | 15.4 | 14.0 | 29.9 / 80.7 |
| release cold | 823 | 777 | 749 | 680 | 594 | 28.6 | 28.2 | 28.2 | 28.0 | 27.5 | 29.1 / 78.1 |
| release warm + prefill swap | 1130 | 1123 | 1064 | 947 | 767 | 51.5 | 50.5 | 50.7 | 49.6 | 48.1 | 29.1 / 78.2 |
| cold against upstream | x2.34 | x2.38 | x2.16 | x1.59 | x1.36 | x1.72 | x1.73 | x1.75 | x1.82 | x1.97 |  |
| warm + prefill swap against upstream | x3.22 | x3.43 | x3.06 | x2.22 | x1.75 | x3.10 | x3.09 | x3.14 | x3.23 | x3.44 |  |

\* The upstream prompt throughput of this row rises with depth, from 351 t/s at depth 0 to 438 at 65536
(+24.6 %). Prompt processing does not get faster with a longer context; the earlier upstream rows on this page
fall with depth (335 to 307 at `aa39d7a3e`). The row was measured once and the cause was not diagnosed, so the
PP512 ratios at 32768 and 65536 are the ratios to this measurement, not to a verified upstream curve.

DeepSeek V4 Flash UD-IQ3_XXS (no prefill swap, the 128 GB setting of the full set below):

| | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 | VRAM / host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `53ed051ce`, `-ncmoe 35` | 268 | 243 | 221 | - | - | 10.2 | 10.0 | 10.0 | - | - | 26.6 / 98.7 |
| release cold | 331 | 292 | 267 | 207 | 165 | 19.8 | 19.1 | 19.1 | 18.6 | 18.1 | 28.0 / 90.0 |
| release warm | 354 | 312 | 283 | 216 | 171 | 34.4 | 33.7 | 33.7 | 32.1 | 30.2 | 28.0 / 90.5 |
| cold against upstream | x1.24 | x1.20 | x1.21 | - | - | x1.94 | x1.91 | x1.91 | - | - |  |
| warm against upstream | x1.32 | x1.28 | x1.28 | - | - | x3.37 | x3.37 | x3.37 | - | - |  |

### Gemma 4 31B Q4_K_M

Dense, the whole model in VRAM, `ROCBLAS_USE_HIPBLASLT=1` for every build. `llama-bench` with
`-d 32768,0,4096,8192,32768`, the first pair discarded:

| | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | VRAM / host |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `53ed051ce` | 895 | 735 | 628 | 327 | 28.7 | 27.0 | 26.5 | 24.4 | 23.8 / 21.0 |
| previous release | 1002 | 865 | 759 | 441 | 29.5 | 27.5 | 27.0 | 24.8 | 23.3 / 21.0 |
| release | 1002 | 865 | 760 | 441 | 29.4 | 27.5 | 27.1 | 24.8 | 23.3 / 21.0 |
| release against upstream | x1.12 | x1.18 | x1.21 | x1.35 | x1.03 | x1.02 | x1.02 | x1.02 |  |

English roleplay in the server, with the same server settings as above except `--ctx-checkpoints 8` and
**`--reasoning off`**: with reasoning at the server default the server turns Gemma's thinking on and most of a
384-token reply is thinking text, so these rows, like the project's Gemma launcher, run with reasoning off. MTP
uses the Gemma 4 31B MTP assistant model (Q8_0, on the GPU, f16 draft cache). The release rows use the per-position
draft thresholds of [spec-draft-thresholds.md](spec-draft-thresholds.md) (`--spec-draft-n-max 15
--spec-draft-p-min 0.33,0.6,0.6,0 --spec-draft-p-continue 0.9`); upstream has only a scalar `p_min` and no
`p_continue`, so its row uses the best scalar setting of the earlier study (`--spec-draft-n-max 13
--spec-draft-p-min 0.8`). A release row with that scalar setting separates the gain of the kernels from the gain
of the thresholds.

| configuration | decode t/s | first request | prompt s | acceptance | against upstream |
|---|---:|---:|---:|---:|---:|
| upstream, no MTP | 27.71 | 28.5 | 14.2 | - |  |
| previous release, no MTP | 28.29 | 29.0 | 13.5 | - | +2.1 % |
| release, no MTP | 28.30 | 29.0 | 13.5 | - | +2.1 % |
| upstream, MTP, scalar `n_max 13, p_min 0.8` | 35.82 | 39.2 | 15.0 | 55.9 % |  |
| release, MTP, the same scalar setting | 40.94 | 44.8 | 14.4 | 55.7 % | +14.3 % |
| previous release, MTP, per-position thresholds | 42.29 | 48.3 | 14.3 | 53.7 % | +18.1 % |
| release, MTP, per-position thresholds | 42.52 | 48.1 | 14.3 | 53.7 % | +18.7 % |

Of the +18.7 % of the release over upstream with MTP, +14.3 % comes from the kernels (the same scalar setting on
both builds) and +3.8 % from the per-position thresholds (release per-position against release scalar). MTP adds
+29.3 % to upstream and +50.2 % to the release. The previous release and this release produce the same replies
and are equal (+0.5 % with MTP, +0.0 % without).

### Optional features

Three features of this release are not used by any row above: the joint cache, which is on by default when a
draft model with routed experts is loaded and was switched off explicitly in the main tables, and the L1 redraw
and the hash early read, which are off by default. Turning one on changes the numbers by the amounts in this
section.

- **Joint cache** (`--expert-cache-draft on`, [expert-cache-joint.md](expert-cache-joint.md)): the DeepSeek MTP
  head joins the target's expert cache. The rows with it use the budget of 20480 MiB, since the head's experts
  then live inside the cache.
- **L1 redraw** (`RANMA_EXPERT_L1_REDRAW_BENEFIT=2`, `RANMA_EXPERT_L1_REDRAW_MOVE=5`,
  [expert-cache-l1-redraw.md](expert-cache-l1-redraw.md)).
- **Hash early read** (`RANMA_EXPERT_HASH_EARLY=vram|both`, [expert-cache-l2.md](expert-cache-l2.md)): as soon
  as the tokens are known, the host-resident experts of the three layers that DeepSeek routes by token id are
  staged in VRAM (`vram`); `both` also reads their file-tier experts early, which needs a finite host tier.

English roleplay, DeepSeek with MTP n1, 128 GB. R = redraw on; D = joint cache; H = hash early read `both`
(with the unlimited host tier only its VRAM staging acts). These
rows ran earlier the same evening with a fixed 300 s rest; R is their reference, because the redraw never fired
(below), and R lies within 0.4 % of the release rows of the main pass.

| configuration | cold | warm | cold against R | warm against R |
|---|---:|---:|---:|---:|
| release, main pass | 33.64 | 35.29 |  |  |
| R | 33.74 | 35.16 |  |  |
| R + D | 33.46 | 35.32 | -0.9 % | +0.5 % |
| R + H | 33.59 | 34.96 | -0.4 % | -0.6 % |
| R + D + H | 33.22 | 35.05 | -1.6 % | -0.3 % |

All rows produce the same replies. At 128 GB none of the three moves decode beyond about 1 %, and R + D + H cold
is the only cell below that (-1.6 %). The redraw evaluated a proposal at every plan install (17 per row) and never
fired: the benefit over the horizon never exceeded twice the cost (largest ratio 0.61 in a cold row with the
joint cache, 0.20 without it, at most 0.035 warm). The cold rows proposed moves of up to 6.5 % of the budget,
the warm rows of at most 1.7 %. At the tested settings the redraw is inactive.

English roleplay, DeepSeek without MTP, 64 GB, warm, the hash early read: the previous release, the release of
the main pass, and three rows of the release in one block with the expert trace switch on
(`RANMA_EXPERT_TRACE=10`) for the counters; one more `both` row ran without the trace.

| configuration | decode t/s | prompt s | against off (trace block) |
|---|---:|---:|---:|
| previous release | 28.38 | 47.3 |  |
| release, main pass (off) | 28.80 | 46.2 |  |
| release, `both`, no trace | 28.72 | 46.7 |  |
| release, off (trace block) | 28.69 | 46.2 |  |
| release, `vram` (trace block) | 28.89 | 46.4 | +0.7 % |
| release, `both` (trace block) | 28.69 | 46.7 | +0.0 % |

All six rows produce the same replies. `vram` is 0.7 % faster than off and `both` equal, both inside what
single rows resolve (see "Equal rows" below). The counters of the off row say why: over the 6206 generated
tokens, decode waited 18.5 s for experts from the file, and 43 ms of that in the three hash-routed layers. At
40 GiB these layers rarely wait on the SSD, so an early file read has little to hide (in the `both` row 69 early
reads, 23 of them used), and the staging copy into VRAM travels over the same host link that the misses of the
other layers use.

### Memory

Maxima in GiB. For the server rows the client sampled the server process after load, after the first and after
the last request, so these are the largest of three samples; for the `llama-bench` rows the runner sampled the
process ten times a second. The columns are those of "Memory of each row" below: private bytes include the backing
of the VRAM allocations.

| row | VRAM dedicated | GPU shared | private bytes |
|---|---:|---:|---:|
| DeepSeek, MTP n1, upstream `-ncmoe 36`, server | 27.5 | 76.6 | 103.7 |
| DeepSeek, MTP n1, release 18432 MiB, 128 GB, server | 29.4 | 73.7 | 90.2 |
| DeepSeek, MTP n1, release 18432 MiB, 64 GB, server | 29.3 | 40.9 | 57.3 |
| DeepSeek, MTP n1, release joint cache 20480 MiB, 128 GB, server | 28.7 | 75.5 | 85.0 |
| DeepSeek, upstream `-ncmoe 35`, server | 25.9 | 74.3 | 99.6 |
| DeepSeek, release 20480 MiB, 128 GB, server | 27.8 | 71.3 | 90.9 |
| DeepSeek, release 20480 MiB, 64 GB, server | 27.8 | 40.6 | 60.0 |
| Qwen, upstream `-ncmoe 35`, server | 28.2 | 53.0 | 83.7 |
| Qwen, release 20480 MiB, 128 GB, server | 27.2 | 52.8 | 81.9 |
| Qwen, release 20480 MiB, 64 GB, server | 27.1 | 40.9 | 70.0 |
| Gemma, upstream, server | 23.9 | 0.9 | 31.1 |
| Gemma, release, server | 24.2 | 1.0 | 31.3 |
| Gemma, upstream, MTP, server | 24.5 | 1.3 | 31.7 |
| Gemma, release, MTP, server | 24.8 | 1.4 | 32.7 |
| Qwen, upstream `-ncmoe 35`, `llama-bench` | 29.9 | 53.1 | 80.7 |
| Qwen, release warm + prefill swap, `llama-bench` | 29.1 | 52.8 | 78.2 |
| DeepSeek, upstream `-ncmoe 35`, `llama-bench` | 26.6 | 74.3 | 98.7 |
| DeepSeek, release warm, `llama-bench` | 28.0 | 71.4 | 90.5 |
| Gemma, upstream, `llama-bench` | 23.8 | 0.2 | 21.0 |
| Gemma, release, `llama-bench` | 23.3 | 0.2 | 21.0 |

The upstream DeepSeek server with the MTP head commits 104 GiB; the release rows commit 90 GiB at 128 GB and 57
to 60 GiB at 64 GB.

### Reading the tables

- **Against upstream.** Warm, the release decodes 2.5 to 3.1 times as fast as upstream in every scenario of both
  MoE models, and processes the prompts of a preset in 29 to 37 % less time for DeepSeek and 46 to 49 % less for
  Qwen. Coding gains least in every model (x2.52 / x2.70 / x2.57 warm); the three translated roleplay presets
  gain as much as English or more.
- **Cold and warm.** Over a whole preset a cold process decodes 3 to 6 % below a warm one. Its first request
  decodes at 56 to 69 % of the warm first request, because the process starts on the seeded placement and
  installs the plans of its own records as the preset proceeds. Prompt time is the same cold and warm (within
  3 %).
- **MTP.** With the release warm, MTP n1 adds 7.6 to 13.9 % to DeepSeek decode (+10.8 % English roleplay,
  +10.8 % coding, +7.6 % Chinese roleplay); with upstream it adds 13.9 to 21.1 %. The two rows of a scenario
  generate different text, and the MTP rows of the release have 2 GiB less expert budget (18432 against
  20480 MiB). Acceptance differs between the builds by at most 2.1 points.
- **64 GB.** A host tier of 40 GiB costs DeepSeek 13 to 17 % of warm decode with MTP and 10 to 13 % without it,
  and 43 to 54 % more prompt time; Qwen loses 5 % of decode. With MTP, DeepSeek at 64 GB still decodes 5 to 6 %
  faster than without it.
- **`llama-bench` and the server.** Warm English roleplay in the server decodes 7.5 % (DeepSeek, 31.85 against
  34.4 t/s) and 9.7 % (Qwen, 46.54 against 51.5) below warm TG128 at depth 0. They are different workloads (a
  server request carries its conversation and reasoning, the plan follows the scenario's profile), and this page
  does not split the difference. A cold `llama-bench` row is the seeded placement alone: 1.7 to 2.0 times
  upstream in decode, 55 to 60 % of the warm row.
- **Gemma.** A dense model in VRAM, so only the kernel changes apply: prompt processing +12 % at depth 0 to +35 %
  at 32768, decode +2 to +3 %, the same as the previous release. With MTP the kernels and the per-position
  thresholds together give +18.7 %.
- **Equal rows.** Differences within about 1 % are inside what one pass of single rows can resolve on this
  machine and are read as equal: the release and the previous release warm (English roleplay, MTP), the release
  and the previous release on Gemma, the option rows at 128 GB except R + D + H cold, and the hash early read at
  64 GB.

### Method notes

- **OpenMP.** All three builds use OpenMP. Earlier rows of this release came from a build without it, while the
  previous release had been built with it; the graphs of these models keep one small CPU split, whose thread
  handoff differs between the two. None of those rows is used here.
- **One pass.** Every configuration is one row, measured once; no row was repeated to pick the better run. The
  only repeats are the four previous-release cold rows (spread 0.3 %).
- **Temperature gate.** See "Rests" above. Most gates ran to the 300 s cap, so in practice the rest was the
  fixed 300 s of the full set below with a check that the card had cooled.
- **Background activity.** Before the final measurements, three previous-release cold rows ran 2.1 % slower
  than the same rows after a reboot (32.52 against 33.21 t/s); the slow episode was traced to Windows background
  services. Every row on this page ran after that reboot, with a host sampler (CPU load, top processes, disk,
  available memory) attached to the whole pass.
- **Replies.** Greedy decoding gives the same replies for cold, warm and 64 GB rows of a scenario, and for the
  previous release and this release. The Qwen roleplay rows with the prefill swap are the exception in two
  scenarios: one request of the English and one of the Korean preset ends differently cold and warm, with the
  same token count. Upstream generates different replies from the release, and MTP from no MTP.

## The full set of ranma_20260922

Everything from here on was measured once, on 2026-09-22, at ranma_20260922 (`e48103e1e`), and is not repeated
for every release. It holds the device rows (the R9700 and the RX 9070 XT emulation), the host memory systems and
the placements that the release section above does not repeat.

### Protocol of the full set

Measured on the development machine, 2026-09-22: Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16,
Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10 SDK with HIP 7.15.26333-6b0e43f341
(`amdhip64_7.dll` 10.0.3581.0, the runtime DLLs shipped next to the binaries). The GPU runs with a -30 % power limit and a 0 mV voltage offset for every measurement here, and headless (no display attached; the CPU's integrated GPU drives
the display), so all 32 GiB are available to the model. Host-resident weights are bound by the PCIe link, so a
narrower or slower link changes these numbers first.

In the sections below (measured 2026-09-22), upstream is `aa39d7a3e`, the revision ranma_20260922 was based on,
built in its own build directory with the same toolchain and options, and ranma_20260922 is the last commit of
that snapshot (`e48103e1e`).

Every row is one `llama-bench` process:

```
llama-bench -m MODEL -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -lm none --poll 0 \
    -p 512 -n 128 -r 1 -d 65536,0,4096,8192,32768,65536 <placement options of the row>
```

with `HIP_VISIBLE_DEVICES` selecting the R9700. The first pair of PP512 and TG128, at the deepest depth, is a
discarded warm-up: on this platform the prompt throughput at shallow depths rises with the elapsed time of the
process, so every process starts with the same deepest pass and the table holds the remaining depths in the
order they ran. The rows of a process share one model load. One GPU process ran at a time, and the GPU idled
for five minutes between two processes, so that no row starts on a card the row before it has heated. No row
was repeated to pick the better run.

**Placement.** The rows with `-ncmoe N` keep the routed experts of the first N layers in host memory, the way
upstream places a model that does not fit. `-ncmoe 35` is the lowest value at which the device buffers of
Qwen3.8-Flash-Next stay under 30 GiB at depth 65536; the same value holds the DeepSeek device buffers under
28 GiB. The expert cache rows have no `-ncmoe`: the cache owns the placement and `--expert-l1-mib` is its VRAM
budget. They run `--expert-cache warm --expert-cache-mode exclusive` from a profile that a Cold run of the
same protocol recorded ([expert-cache.md](expert-cache.md)); every expert cache row starts from its own copy
of that profile.

**Switches.** The rows called host-direct and the expert cache rows run with `GGML_CUDA_HOST_DIRECT=1` and
`GGML_CUDA_HOST_DIRECT_MAX_BATCH=512` ([host-direct-moe.md](host-direct-moe.md)). The row called "no switch
set" is ranma_20260922 as it behaves with no environment variable and no new option. Everything else is at its
default.

**The RX 9070 XT rows** are not measurements on that card. They are the same R9700 with the expert placement a
16 GiB card would use (device buffers under 15 GiB at the deepest depth: `-ncmoe 45`, or an expert cache
budget of 3072 MiB for Qwen and 4096 MiB for DeepSeek, whose device buffers outside the cache are smaller;
both use about 12 GiB of the card). Both cards are the same Navi 48 die with 64 CUs and 640 GB/s GDDR6; the
boost clock differs by under 2 %. A 9070 XT with a display attached has 1-2 GB less VRAM, which for these MoE
models is about one expert layer, so it may need a higher `-ncmoe` or a smaller budget.

**The RAM rows** emulate a machine with less than 128 GB by bounding the host tier of the expert cache to the
emulated RAM minus 24 GB: 128 GB is `--expert-l2-mib -1`, 64 GB is `40960`, 32 GB is `8192`
([expert-cache-l2.md](expert-cache-l2.md)). This emulates the capacity only. The page cache of the 128 GB
machine still holds the model file, so the file-tier cost of the 64 GB and 32 GB rows is a lower bound.
Upstream and the `-ncmoe` rows need every expert outside VRAM in host memory, so they exist for 128 GB only.

### Qwen3.8-Flash-Next UD-Q4_K_XL

PP512 and TG128 in t/s at the depth after the `@`.

#### R9700

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `aa39d7a3e`, `-ncmoe 35` | 335 | 330 | 316 | 316 | 307 | 15.9 | 15.8 | 15.2 | 14.4 | 13.2 |
| ranma_20260922, `-ncmoe 35`, no switch set | 590 | 564 | 542 | 497 | 447 | 17.3 | 16.9 | 17.0 | 16.3 | 16.2 |
| ranma_20260922, `-ncmoe 35`, host-direct | 650 | 621 | 598 | 546 | 481 | 26.8 | 26.2 | 26.1 | 25.5 | 24.9 |
| ranma_20260922, expert cache 20480 MiB | 916 | 886 | 861 | 762 | 626 | 50.4 | 49.2 | 48.8 | 46.7 | 44.7 |

#### RX 9070 XT emulation

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `aa39d7a3e`, `-ncmoe 45` | 310 | 307 | 301 | 297 | 290 | 13.7 | 13.5 | 13.3 | 12.5 | 11.6 |
| ranma_20260922, `-ncmoe 45`, host-direct | 649 | 620 | 594 | 551 | 484 | 23.6 | 23.1 | 23.0 | 22.6 | 22.2 |
| ranma_20260922, expert cache 3072 MiB | 622 | 612 | 588 | 544 | 480 | 37.6 | 39.0 | 39.2 | 37.5 | 36.0 |

#### By system

The best configuration of each system: the exclusive expert cache with the budget of the device row, with or
without `--expert-prefill-swap` ([expert-cache-prefill.md](expert-cache-prefill.md)), whichever has the higher
mean TG128; inside 1 % the mean PP512 decides.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R9700, 128 GB (exclusive + prefill swap) | 1014 | 989 | 949 | 844 | 692 | 49.7 | 48.4 | 47.9 | 46.9 | 44.8 |
| R9700, 64 GB (exclusive + prefill swap) | 851 | 880 | 844 | 755 | 632 | 49.3 | 48.0 | 47.8 | 45.6 | 44.0 |
| RX 9070 XT emulation, 64 GB (exclusive + prefill swap) | 625 | 613 | 596 | 555 | 480 | 36.7 | 38.0 | 38.2 | 36.7 | 35.2 |
| RX 9070 XT emulation, 32 GB (exclusive + prefill swap) | 152 | 138 | 133 | 137 | 131 | 18.3 | 21.0 | 22.3 | 20.6 | 19.9 |

Both settings of every system:

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R9700, 128 GB, exclusive | 916 | 886 | 861 | 762 | 626 | 50.4 | 49.2 | 48.8 | 46.7 | 44.7 |
| R9700, 64 GB, exclusive | 774 | 804 | 794 | 729 | 617 | 49.0 | 47.8 | 47.6 | 45.4 | 43.8 |
| RX 9070 XT emulation, 64 GB, exclusive | 536 | 510 | 489 | 501 | 423 | 36.8 | 38.1 | 38.3 | 36.8 | 35.3 |
| RX 9070 XT emulation, 32 GB, exclusive | 154 | 135 | 133 | 137 | 128 | 18.4 | 20.5 | 22.4 | 20.8 | 19.4 |
| R9700, 128 GB, exclusive + prefill swap | 1014 | 989 | 949 | 844 | 692 | 49.7 | 48.4 | 47.9 | 46.9 | 44.8 |
| R9700, 64 GB, exclusive + prefill swap | 851 | 880 | 844 | 755 | 632 | 49.3 | 48.0 | 47.8 | 45.6 | 44.0 |
| RX 9070 XT emulation, 64 GB, exclusive + prefill swap | 625 | 613 | 596 | 555 | 480 | 36.7 | 38.0 | 38.2 | 36.7 | 35.2 |
| RX 9070 XT emulation, 32 GB, exclusive + prefill swap | 152 | 138 | 133 | 137 | 131 | 18.3 | 21.0 | 22.3 | 20.6 | 19.9 |

### DeepSeek V4 Flash UD-IQ3_XXS

#### R9700

The rows with `-ncmoe 35` stop at depth 8192 (`-d 8192,0,4096,8192`, first pair discarded): with the experts
of 35 layers behind the PCIe link, the fill of 65536 tokens takes longer than the watchdog of the measurement
lets a process run without output. These rows hold about 99 GiB of host memory, so there is no upstream row
for the RX 9070 XT placement, which would hold more.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | TG128 @0 | TG128 @4096 | TG128 @8192 |
|---|---:|---:|---:|---:|---:|---:|
| upstream `aa39d7a3e`, `-ncmoe 35` | 248 | 228 | 209 | 9.8 | 9.8 | 9.7 |
| ranma_20260922, `-ncmoe 35`, host-direct | 265 | 244 | 219 | 17.5 | 17.3 | 17.2 |

With the expert cache the full curve fits:

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ranma_20260922, expert cache 20480 MiB | 310 | 276 | 255 | 161 | 108 | 31.8 | 30.8 | 30.9 | 29.7 | 28.1 |

#### By system

The same four systems as for Qwen, the better of exclusive and exclusive + prefill swap by the same rule. The
32 GB rows stop at depth 32768 (`-d 32768,0,4096,8192,32768`, first pair discarded): with an 8 GiB host tier
the fill of 65536 tokens reads most experts from the file and exceeds the same watchdog.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R9700, 128 GB (exclusive) | 310 | 276 | 255 | 161 | 108 | 31.8 | 30.8 | 30.9 | 29.7 | 28.1 |
| R9700, 64 GB (exclusive + prefill swap) | 199 | 204 | 196 | 138 | 97 | 28.1 | 28.4 | 28.6 | 27.8 | 26.2 |
| RX 9070 XT emulation, 64 GB (exclusive + prefill swap) | 129 | 129 | 128 | 103 | 78 | 18.8 | 19.7 | 20.3 | 20.0 | 18.9 |
| RX 9070 XT emulation, 32 GB (exclusive + prefill swap) | 79 | 77 | 75 | 66 | - | 9.0 | 9.4 | 9.6 | 9.5 | - |

Both settings of every system:

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R9700, 128 GB, exclusive | 310 | 276 | 255 | 161 | 108 | 31.8 | 30.8 | 30.9 | 29.7 | 28.1 |
| R9700, 64 GB, exclusive | 197 | 195 | 189 | 135 | 96 | 28.3 | 28.3 | 28.5 | 27.7 | 26.1 |
| RX 9070 XT emulation, 64 GB, exclusive | 130 | 129 | 127 | 103 | 78 | 18.9 | 19.8 | 20.3 | 19.9 | 19.0 |
| RX 9070 XT emulation, 32 GB, exclusive | 77 | 75 | 75 | 64 | - | 9.0 | 9.4 | 9.5 | 9.5 | - |
| R9700, 128 GB, exclusive + prefill swap | 296 | 268 | 245 | 158 | 107 | 31.7 | 31.1 | 31.0 | 29.8 | 28.1 |
| R9700, 64 GB, exclusive + prefill swap | 199 | 204 | 196 | 138 | 97 | 28.1 | 28.4 | 28.6 | 27.8 | 26.2 |
| RX 9070 XT emulation, 64 GB, exclusive + prefill swap | 129 | 129 | 128 | 103 | 78 | 18.8 | 19.7 | 20.3 | 20.0 | 18.9 |
| RX 9070 XT emulation, 32 GB, exclusive + prefill swap | 79 | 77 | 75 | 66 | - | 9.0 | 9.4 | 9.6 | 9.5 | - |

### Gemma 4 31B Q4_K_M

R9700, dense, the whole model in VRAM, `-d 32768,0,4096,8192,32768` with the first pair discarded, and
`ROCBLAS_USE_HIPBLASLT=1` for both builds.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream `aa39d7a3e` | 897 | 735 | 628 | 328 | 28.3 | 26.7 | 26.3 | 24.2 |
| ranma_20260922 | 1001 | 868 | 762 | 440 | 29.3 | 27.3 | 26.9 | 24.7 |

### Memory of each row

The runner that started every process sampled it ten times a second: private bytes and working set from
Windows, dedicated and shared GPU memory of the process from the `GPU Process Memory` performance counters.
The table holds the maximum of each over the life of the process, in GiB. Every column is the process alone.

- **VRAM dedicated** is the memory on the card. It matches the placement rule above: under 30 GiB on the
  R9700 rows, about 12 GiB on the RX 9070 XT rows.
- **GPU shared** is host memory the GPU addresses: the pinned host buffers, which hold the experts of the
  `-ncmoe` rows and the host tier of the expert cache. It is part of the private bytes, not in addition.
- **Private bytes** is the commit charge of the process. On Windows it also carries the backing of the VRAM
  allocations: a process with the whole model in VRAM (Gemma) commits 17 GiB before its working set reaches
  5 GiB. So this column bounds the host memory a row needs from above, by up to the VRAM column.
- **Working set** is what was resident, the mapped pages of the model file included. It is neither a floor
  nor a ceiling of what a row needs: file pages are page cache the system reclaims, and committed memory
  the process has not touched is not in it.

| configuration | VRAM dedicated | GPU shared | private bytes | working set |
|---|---:|---:|---:|---:|
| Qwen, upstream `aa39d7a3e`, `-ncmoe 35` | 29.9 | 53.1 | 80.7 | 64.8 |
| Qwen, `-ncmoe 35`, no switch set | 29.7 | 53.2 | 80.6 | 64.7 |
| Qwen, `-ncmoe 35`, host-direct | 28.6 | 53.1 | 80.5 | 64.7 |
| Qwen, expert cache 20480 MiB, 128 GB, exclusive | 29.1 | 52.8 | 78.0 | 64.4 |
| Qwen, expert cache 20480 MiB, 128 GB, exclusive + prefill swap | 29.1 | 52.8 | 78.2 | 64.4 |
| Qwen, expert cache 20480 MiB, 64 GB, exclusive | 29.0 | 40.9 | 66.1 | 52.5 |
| Qwen, expert cache 20480 MiB, 64 GB, exclusive + prefill swap | 29.0 | 40.9 | 66.3 | 52.5 |
| Qwen, upstream `aa39d7a3e`, `-ncmoe 45` | 15.3 | 67.8 | 80.7 | 79.4 |
| Qwen, `-ncmoe 45`, host-direct | 13.9 | 67.8 | 80.5 | 79.3 |
| Qwen, expert cache 3072 MiB, 128 GB, exclusive | 12.1 | 69.8 | 77.7 | 81.4 |
| Qwen, expert cache 3072 MiB, 64 GB, exclusive | 12.0 | 40.9 | 48.8 | 52.5 |
| Qwen, expert cache 3072 MiB, 64 GB, exclusive + prefill swap | 12.0 | 40.9 | 48.8 | 52.5 |
| Qwen, expert cache 3072 MiB, 32 GB, exclusive | 11.9 | 8.9 | 16.7 | 20.5 |
| Qwen, expert cache 3072 MiB, 32 GB, exclusive + prefill swap | 11.9 | 8.9 | 16.7 | 20.5 |
| DeepSeek, upstream `aa39d7a3e`, `-ncmoe 35` | 27.8 | 74.3 | 98.7 | 74.8 |
| DeepSeek, `-ncmoe 35`, host-direct | 23.9 | 74.3 | 98.8 | 74.8 |
| DeepSeek, expert cache 20480 MiB, 128 GB, exclusive | 27.9 | 71.4 | 90.3 | 72.4 |
| DeepSeek, expert cache 20480 MiB, 128 GB, exclusive + prefill swap | 27.9 | 71.4 | 90.5 | 72.4 |
| DeepSeek, expert cache 20480 MiB, 64 GB, exclusive | 27.9 | 40.6 | 59.5 | 41.6 |
| DeepSeek, expert cache 20480 MiB, 64 GB, exclusive + prefill swap | 27.9 | 40.6 | 59.7 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 64 GB, exclusive | 11.8 | 40.6 | 49.0 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 64 GB, exclusive + prefill swap | 11.8 | 40.6 | 49.0 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 32 GB, exclusive | 11.9 | 8.6 | 16.7 | 9.3 |
| DeepSeek, expert cache 4096 MiB, 32 GB, exclusive + prefill swap | 11.9 | 8.5 | 16.7 | 9.3 |
| Gemma, upstream `aa39d7a3e` | 23.8 | 0.2 | 21.0 | 20.7 |
| Gemma, ranma_20260922 | 23.3 | 0.2 | 21.0 | 20.8 |

Two things follow. The RAM rows hold what they were set to: the 32 GB rows commit 17 GiB, of which 9 GiB is
pinned host memory (the host tier), and the 64 GB rows on the 16 GiB placement commit 49 GiB with 41 GiB
pinned. The 64 GB rows on the R9700 placement commit 66 GiB because the 20 GiB expert cache in VRAM is
charged too; their pinned host memory is the same 41 GiB. The upstream and `-ncmoe` rows of DeepSeek commit
99 GiB, the reason those rows exist for 128 GB only.

The expert cache budgets were chosen for the card, and chosen conservatively: 20480 MiB for the R9700 and
3072 / 4096 MiB for a 16 GiB card leave 2 to 4 GiB of the 32 GiB and about 4 GiB of the 16 GiB unused (1 to
2 GB of which a display would take). A larger `--expert-l1-mib` fills that with experts, so a machine with
either card can allocate more than these rows did. What that gains was not measured.

### Reading the tables

- **Qwen, R9700.** Against upstream at the same `-ncmoe 35` placement, ranma_20260922 with no switch set decodes
  9 % faster at depth 0 and 23 % faster at 65536, and processes prompts 76 % faster at depth 0; that is the
  changes of this series that are on by default. The host-direct switch raises decode to +69 % / +89 % (depth
  0 / 65536). The expert cache with the same 20 GiB of VRAM decodes 3.2x / 3.4x upstream and processes prompts
  2.7x / 2.0x.
- **Qwen, RX 9070 XT emulation.** With 3 expert layers in VRAM, host-direct decodes +72 % / +91 % over
  upstream; the expert cache with 3 GiB decodes 2.7x / 3.1x. Prompt processing of the cache row is within 4 %
  of the host-direct row: a prompt ubatch reads almost every expert once, and 3 GiB holds few of them
  ([expert-cache-prefill.md](expert-cache-prefill.md)).
- **Qwen by system.** The 64 GB rows decode within 1 to 3 % of the 128 GB rows on the R9700 placement; on the
  3 GiB placement the difference is 2 to 3 %. The 32 GB rows are a different regime: half the decode
  throughput of 64 GB and a quarter of the prompt throughput, with the model file in the loop. The prefill
  swap is worth +10 to +12 % of prompt throughput at 128 GB, +2 to +10 % at 64 GB on the R9700 placement and
  +11 to +22 % on the 3 GiB placement, and nothing at 32 GB; on decode it stays within 2 % of the exclusive
  row, below it by 1 to 2 % at 128 GB.
- **DeepSeek.** Host-direct decodes +78 % over upstream at the same placement; the expert cache decodes 3.2x
  at depth 0 and holds 28.1 t/s at 65536. Prompt processing with the cache falls with depth faster than for
  Qwen (310 to 108 t/s). The 64 GB rows keep about 90 % of the 128 GB decode; the RX 9070 XT emulation at
  4 GiB keeps 60 to 70 %, and 32 GB with that placement decodes at 9 to 10 t/s with an 8 GiB host tier.
- **Gemma.** A dense model in VRAM, so only the kernel changes of this series apply: decode +2 to +4 %, prompt
  processing +12 % at depth 0 and +34 % at 32768.
- **The shape of the curves.** In the rows on the R9700 placement decode falls with depth, or is flat within
  2 % from depth 0 to 8192. On the 3 or 4 GiB placement, and in the 32 GB rows, decode rises from depth 0 to
  8192 before it falls, by 4 to 22 %. This was observed on every such row and was not diagnosed. The depth 0
  cell of the Qwen 3072 MiB row was measured again alone in a new process (`-d 65536,0`) and gave the same
  value to two decimals, so the table keeps the values as measured. No other cell was repeated.
- **Prompt processing at depth 0 falls into one of two levels per process on this platform**, a few percent
  apart, which the discarded warm-up pass does not remove. Compare PP512 columns across rows with that in
  mind; TG128 does not show it.
