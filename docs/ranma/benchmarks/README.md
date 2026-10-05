# Benchmarks

The throughput of this fork against upstream llama.cpp. The numbers are kept as records: each record was measured at
one release, against the upstream commit that release is based on, and is not measured again for every later
release. Every number below therefore names the release tag it was measured at. The feature pages describe what each
change does and how to switch it; the conditions that the records share are in [method.md](method.md).

Which release each kind of number comes from:

- **ranma_20261005** (upstream `42d958167`): Qwen3.8-Flash-Next in EXL3 3.05 and 4.05 bpw against UD-Q4_K_XL, the
  `llama-bench` curves and the server scenarios with MTP and `--spec-smart`, on the R9700 with 128 and 64 GB and the
  RX 9070 XT emulation with 64 GB.
- **ranma_20261001** (upstream `ed7ac35e1`): the only comparison measured again for that release is the MTP draft
  length, `--spec-smart` against the single-argument values and no MTP. The rows of 2026-09-29 measured one of its
  changes, graph reuse per batch shape on by default. Both ran on development builds of its series.
- **ranma_20260928** (upstream `53ed051ce`): the decode ratios against upstream in the server scenarios, the
  `llama-bench` curves of that release, Gemma 4 31B and the memory of those rows. The Qwen3.8-Flash-Next MTP n1 rows
  were measured at `ranma_20260928_qwen_mtp`, ranma_20260928 with the MTP head of upstream pull request #28243.
- **ranma_20260922** (upstream `aa39d7a3e`): the full set of `llama-bench` curves, with the RX 9070 XT emulation and
  the 64 and 32 GB systems.

## Records

| record | release | upstream base | measured | what |
|---|---|---|---|---|
| [2026-10-05-exl3.md](2026-10-05-exl3.md) | ranma_20261005 | `42d958167` | 2026-10-05 | Qwen3.8-Flash-Next EXL3 3.05 and 4.05 bpw against UD-Q4_K_XL: `llama-bench` curves cold and warm, server scenarios with MTP, three systems, memory |
| [2026-10-01-mtp-smart.md](2026-10-01-mtp-smart.md) | ranma_20261001 (development build) | `ed7ac35e1` | 2026-10-01 | MTP draft length: no MTP, single argument, `--spec-smart` cold and warm; six scenarios, two models |
| [2026-09-29-graph-reuse-default.md](2026-09-29-graph-reuse-default.md) | ranma_20261001 (development build) | `ed7ac35e1` | 2026-09-29 | graph reuse per batch shape turned on by default |
| [2026-09-28-qwen-mtp-n1.md](2026-09-28-qwen-mtp-n1.md) | ranma_20260928_qwen_mtp | `53ed051ce` | 2026-09-28 | Qwen3.8-Flash-Next MTP n1 in English roleplay |
| [2026-09-28-release.md](2026-09-28-release.md) | ranma_20260928 | `53ed051ce` | 2026-09-27 and 2026-09-28 | server scenarios against upstream and the previous release, `llama-bench` curves, Gemma 4 31B, optional features, memory |
| [2026-09-22-full-set.md](2026-09-22-full-set.md) | ranma_20260922 | `aa39d7a3e` | 2026-09-22 | `llama-bench` curves on the R9700 and the RX 9070 XT emulation, 128, 64 and 32 GB, Gemma 4 31B, memory |

## Qwen3.8-Flash-Next EXL3

| what | numbers | release | record |
|---|---|---|---|
| `llama-bench` PP512 / TG128 at depth 0, warm + prefill swap, R9700 with 128 GB: EXL3 3.05 / 4.05 bpw / UD-Q4_K_XL | 1255 / 53.5, 1246 / 49.6, 1138 / 49.3 | ranma_20261005 | [EXL3](2026-10-05-exl3.md#r9700-128-gb) |
| `llama-bench` TG128 against UD-Q4_K_XL, warm, depth 0 and 65536 | 3.05 bpw +5.9 to +9.5 % on the R9700, +17.7 to +20.5 % on the RX 9070 XT emulation; 4.05 bpw within 1 % on the R9700 | ranma_20261005 | [EXL3](2026-10-05-exl3.md#exl3-against-ud-q4_k_xl-warm-tg128--pp512-ratio-at-depth-0-and-the-deepest-depth) |
| Server decode with MTP and `--spec-smart`, warm, five scenarios, against UD-Q4_K_XL | 3.05 bpw +17.2 to +23.1 % (R9700, 128 GB), +50.6 to +85.9 % (RX 9070 XT emulation, 64 GB); 4.05 bpw +3.9 to +10.0 % and +15.4 to +28.1 % | ranma_20261005 | [EXL3](2026-10-05-exl3.md#mtp-smart-server-rows-warm-decode-ts) |

## Qwen3.8-Flash-Next UD-Q4_K_XL

| what | numbers | release | record |
|---|---|---|---|
| Server decode, release warm against upstream (`-ncmoe 35`), five scenarios | x2.57 (coding) to x2.79; English roleplay 46.54 against 16.75 t/s | ranma_20260928 | [server scenarios](2026-09-28-release.md#qwen38-flash-next-ud-q4_k_xl) |
| Server decode at 64 GB (host tier of 40 GiB), warm | 5 % below 128 GB | ranma_20260928 | [reading the tables](2026-09-28-release.md#reading-the-tables) |
| MTP, single argument (`--spec-draft-n-max 2 --spec-draft-p-min 0`) against no MTP, six scenarios | +19.7 to +29.5 % | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| `--spec-smart` against the single argument | cold -4.3 to +0.6 %, warm -1.9 to +2.6 % | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| English roleplay decode: no MTP / single / smart cold / smart warm | 46.10 / 57.13 / 56.99 / 57.54 t/s | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| Graph reuse per batch shape on by default, English roleplay with MTP | +11.1 % (46.52 to 51.67 t/s) | ranma_20261001 | [graph reuse](2026-09-29-graph-reuse-default.md) |
| MTP n1 with the joint cache, English roleplay, warm | 55.62 t/s: +20.0 % over no MTP, x3.32 upstream without MTP | ranma_20260928_qwen_mtp | [Qwen MTP n1](2026-09-28-qwen-mtp-n1.md) |
| `llama-bench` PP512 / TG128 at depth 0, release warm + prefill swap against upstream | 1130 / 51.5 against 351 / 16.6 | ranma_20260928 | [`llama-bench` curves](2026-09-28-release.md#llama-bench-curves) |
| `llama-bench` PP512 / TG128 on the R9700 with 128 GB, depth 0 and 65536, against upstream | 1014 / 49.7 and 692 / 44.8 against 335 / 15.9 and 307 / 13.2 | ranma_20260922 | [full set](2026-09-22-full-set.md#qwen38-flash-next-ud-q4_k_xl) |
| `llama-bench` PP512 / TG128 on the RX 9070 XT emulation with 64 GB, depth 0 | 625 / 36.7 | ranma_20260922 | [full set](2026-09-22-full-set.md#by-system) |

## DeepSeek V4 Flash UD-IQ3_XXS

| what | numbers | release | record |
|---|---|---|---|
| Server decode with MTP n1, release warm against upstream (`-ncmoe 36`) | x2.52 (coding) to x2.95; English roleplay 35.29 against 12.64 t/s | ranma_20260928 | [MTP n1](2026-09-28-release.md#deepseek-v4-flash-ud-iq3_xxs-mtp-n1) |
| Server decode without MTP, release warm against upstream (`-ncmoe 35`) | x2.70 (coding) to x3.11; English roleplay 31.85 against 10.74 t/s | ranma_20260928 | [no MTP](2026-09-28-release.md#deepseek-v4-flash-ud-iq3_xxs-no-mtp) |
| Server decode at 64 GB (host tier of 40 GiB), warm | 13 to 17 % below 128 GB with MTP, 10 to 13 % without | ranma_20260928 | [reading the tables](2026-09-28-release.md#reading-the-tables) |
| MTP, single argument (`--spec-draft-n-max 1 --spec-draft-p-min 0`) against no MTP, six scenarios | +9.8 to +14.6 % | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| `--spec-smart` against the single argument | coding +1.0 % cold, +1.5 % warm; every roleplay scenario 1.5 to 3.8 % slower cold, 1.3 to 3.2 % slower warm | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| English roleplay decode: no MTP / single / smart cold / smart warm | 31.17 / 35.00 / 34.07 / 34.55 t/s | ranma_20261001 | [MTP draft length](2026-10-01-mtp-smart.md) |
| Graph reuse per batch shape on by default, English roleplay with MTP | +8.6 % (28.97 to 31.45 t/s) | ranma_20261001 | [graph reuse](2026-09-29-graph-reuse-default.md) |
| `llama-bench` PP512 / TG128 at depth 0, release warm against upstream | 354 / 34.4 against 268 / 10.2 | ranma_20260928 | [`llama-bench` curves](2026-09-28-release.md#llama-bench-curves) |
| `llama-bench` PP512 / TG128 on the R9700 with 128 GB, depth 0 and 65536 | 310 / 31.8 and 108 / 28.1; upstream 248 / 9.8 at depth 0 | ranma_20260922 | [full set](2026-09-22-full-set.md#deepseek-v4-flash-ud-iq3_xxs) |

## Gemma 4 31B Q4_K_M

Dense, the whole model in VRAM, so only the kernel changes apply.

| what | numbers | release | record |
|---|---|---|---|
| `llama-bench` PP512 / TG128 at depth 0 and 32768, against upstream | 1002 / 29.4 and 441 / 24.8 against 895 / 28.7 and 327 / 24.4 (prompt x1.12 to x1.35, decode x1.02 to x1.03) | ranma_20260928 | [Gemma](2026-09-28-release.md#gemma-4-31b-q4_k_m) |
| Server, English roleplay, MTP with per-position thresholds against upstream MTP with its best scalar setting | 42.52 against 35.82 t/s (+18.7 %); without MTP 28.30 against 27.71 (+2.1 %) | ranma_20260928 | [Gemma](2026-09-28-release.md#gemma-4-31b-q4_k_m) |
| `llama-bench` PP512 / TG128 at depth 0, against upstream | 1001 / 29.3 against 897 / 28.3 | ranma_20260922 | [full set](2026-09-22-full-set.md#gemma-4-31b-q4_k_m) |
