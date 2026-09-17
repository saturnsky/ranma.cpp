# Benchmarking the expert cache

`llama-bench` and `llama-server` drive the same policy object (`common/expert-policy.h`) for profile
collection, commits and installs, so a `llama-bench` row measures the same cache the server runs.
This page describes the `llama-bench` modes, the protocol behind the published numbers, and the
numbers themselves.

## Modes

| `--expert-cache` | L1 | host tier | profiles and placement |
|---|---|---|---|
| `off` | none; `--expert-l1-mib` must be 0 | unlimited or finite, seeded and fixed | no profiler bank, no records |
| `cold` | positive budget, seeded random residents | unlimited or finite, seeded and fixed | records the `prefill` and `decode` banks separately, installs nothing; the profile directory must be absent or empty |
| `warm` | positive budget, inclusive or exclusive | unlimited or finite | starts from a copy of the supplied profile and adapts with the server policy |

Cold and Warm need `--expert-profile-dir DIR`. Warm copies the source once before the process starts
its tests and never updates the source; the temporary copy is deleted at normal exit unless
`--expert-profile-keep` is set. A nonempty Cold destination is refused, and `llama-bench` refuses
`--expert-profile-reset` so that it cannot delete an input.

"Cold" means a seeded fixed placement, not cold hardware: the page cache, the profile and the SSD are
not flushed between processes. The Cold row of a device row is the seed of its Warm rows.

The independent repetition is one process. `-r 1` is the protocol; additional repetitions evolve
the same profile and placement inside one loaded model. `--expert-profile-restore-each` is an
optional Warm control that restores both the profile copy and the placement before each repetition;
it reloads the model and costs more.

The server's expert options are accepted, except `--expert-freeze` (Off and Cold never install,
Warm always may) and `--expert-profile-reset`: `--expert-cache-mode`, `--expert-l1-mib`,
`--expert-l2-mib` (-1 unlimited, 0 refused), the three ring options, `--expert-seed` (default 1),
`--expert-l2-worker-cpu`, `--expert-profile-archive`, and `--expert-prefill-swap` as a Warm option
(Off and Cold have no phase installs). `--ple-prefetch` applies to the whole run
(`ple-prefetch.md`).

## Records and timing

The warm-up decode is excluded from records. Depth fill is prompt processing and enters the
`prefill` record; a PP test commits its `prefill` record when it completes, a TG test its `decode`
record. The benchmark drives prompt start before depth fill and PP, generation start before TG,
request end after each test, and all-idle between repetitions, exactly the server's moments
(`expert-cache-banks.md`). The model-load seed serves until the policy has produced the required
plan, so the first test of a process may not run on the same placement as the later ones.

Markdown output keeps the normal `llama-bench` table and adds the cache mode column and a `ctl ms`
column. With the cache on, `t/s` is compute-only time; the policy and install cost sits in
`expert_control_ns`, and `ctl ms` is its mean per test, so a quoted row can be compared with a plain
one. JSON and JSONL include the expert settings. `RANMA_EXPERT_TRACE=8` adds `expert_metrics` JSON
lines to stderr even in the quiet mode: round lines with the selected bytes served from VRAM, host
memory and the file (the byte split quoted on the other pages), install lines with the install time
and the bytes moved per direction, and tier lines with reads, ring hits, SSD bytes and GPU waiting.
Install and profile-save time are reported separately and must not be presented as steady
throughput. With a finite host tier the control time includes SSD reads and can be seconds per test
(`expert-cache-banks.md`).

## Protocol of the published numbers

Machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit -30 %,
voltage offset 0 mV; Ryzen 9 7950X3D (16 cores), 128 GiB DDR5-5600, Windows 11, ROCm 10, the model
file on the local SSD. Model: Qwen3.8-Flash-Next UD-Q4_K_XL (48 routed layers x 512 experts, 10
used, 73450 MiB of routed expert weights, 4 shards).

Every row is one `llama-bench` process:

```
llama-bench -m MODEL -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -lm none \
    -p 512 -n 128 -r 1 -d 65536,0,4096,8192,32768,65536 -o md -oe jsonl \
    <placement options of the row>
```

with `HIP_VISIBLE_DEVICES=1`, `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`.
The first (65536) pair of PP and TG is a discarded warm-up: on this platform shallow-depth prompt
throughput rises with process elapsed time, so every process starts with the same deepest pass and
the reported rows are the remaining five, identified by execution order. The rows of a process share
one model load. The base revision is `ranma_20260914` (`771fa0bda`), the published snapshot before
this series; its rows carry no expert option and place experts with `-ncmoe`.

**Device rows.** The R9700 row uses 30 GiB of its 32 GiB. The RX 9070 XT row is not a measurement on
that card: it is the same R9700 with the expert placement a 16 GiB card would use (device buffers
under 15 GiB at depth 65536). The same silicon, so only the capacity is emulated; a card with a
display attached has 1 to 2 GB less VRAM, which for this model is about one expert layer. The
placement of each row is the largest one whose `ROCm0` buffers (model + KV + recurrent state +
compute + the expert cache arena) stay under the cap at depth 65536, proven by a `-v` load:

| device row | cap | base placement | base `ROCm0` at 65536 | cache budget | cache `ROCm0` at 65536 |
|---|---:|---|---:|---:|---:|
| R9700 | 30 GiB | `-ncmoe 35` (13 expert layers in VRAM) | 26.38 GiB | `--expert-l1-mib 20000` | 26.38 GiB |
| RX 9070 XT emulation | 15 GiB | `-ncmoe 45` (3 layers) | 11.73 GiB | `--expert-l1-mib 3072` | 9.85 GiB |

**RAM rows.** A machine with less than 128 GB is emulated by bounding the host tier to the emulated
RAM minus 24 GB: 128 GB = `--expert-l2-mib -1`, 64 GB = `40960`, 32 GB = `8192`. This emulates the
capacity only. The page-cache behaviour of a smaller machine is not emulated, so the file-tier cost
of the 64 GB and 32 GB rows is a lower bound. The base revision needs the experts outside VRAM in
host memory: 56.7 GB (`-ncmoe 35`) or 72.5 GB (`-ncmoe 45`) plus the 24 GB, so it fits the 128 GB row
only and the other rows say "does not fit".

**Rows and order.** Per device row: base, Cold exclusive (writes the seed), inclusive Warm,
exclusive Warm, inclusive + swap Warm, exclusive + swap Warm, base again (drift), all at 128 GB; then
exclusive Warm, exclusive + swap Warm, exclusive Warm again at 64 GB; then the winner of the 64 GB
comparison twice at 32 GB. The 64 GB winner is the higher mean TG128 over the five depths; when the
difference is inside the drift between the two exclusive rows, mean PP512 decides. 24 processes, the
longest 1425 s.

## Results

PP512 and TG128 in t/s at depths 0, 4096, 8192, 32768, 65536. The 64 GB exclusive row and the 32 GB
rows are the mean of their two repeats.

### R9700, 20000 MiB budget, base at `-ncmoe 35`

| RAM | mode | PP512 | TG128 |
|---|---|---|---|
| 128 GB | base, no cache | 349.89, 343.86, 338.34, 331.30, 322.30 | 23.90, 23.43, 23.10, 21.14, 18.95 |
| 128 GB | Cold, exclusive | 636.56, 656.79, 634.96, 573.21, 493.20 | 24.37, 23.96, 23.57, 21.46, 19.10 |
| 128 GB | Warm, inclusive | 864.89, 863.78, 842.60, 733.35, 609.81 | 38.00, 37.71, 37.02, 32.04, 26.89 |
| 128 GB | Warm, exclusive | 937.43, 927.21, 898.39, 771.49, 631.95 | 38.90, 38.52, 37.62, 32.01, 26.89 |
| 128 GB | Warm, inclusive + prefill swap | 942.85, 943.39, 897.43, 762.83, 616.78 | 38.49, 38.16, 37.46, 31.75, 26.71 |
| 128 GB | Warm, exclusive + prefill swap | 947.95, 953.39, 916.02, 786.23, 643.71 | 38.20, 37.78, 37.36, 31.90, 26.75 |
| 128 GB | base again | 338.67, 332.25, 326.49, 318.83, 311.98 | 23.80, 23.33, 22.90, 21.05, 18.88 |
| 64 GB | Warm, exclusive | 756.18, 772.05, 759.96, 692.15, 576.16 | 37.92, 37.63, 36.76, 31.44, 26.44 |
| 64 GB | Warm, exclusive + prefill swap | 916.86, 915.62, 878.40, 765.02, 626.96 | 37.85, 37.58, 36.77, 31.39, 26.50 |
| 64 GB | base | does not fit (needs 80.7 GB) | |
| 32 GB | Warm, exclusive + prefill swap | 488.33, 479.72, 458.96, 448.33, 386.25 | 32.00, 34.33, 34.99, 29.50, 25.25 |
| 32 GB | base | does not fit (needs 80.7 GB) | |

Base-to-base drift: PP512 up to 12.5 t/s (3.9 %), TG128 up to 0.20 t/s (0.9 %). The two 64 GB
exclusive rows differ by up to 0.16 t/s on TG128; the swap row is inside that on TG128 and 15.4 %
above on mean PP512, so it is the 32 GB representative. The two 32 GB repeats differ by up to 9.7 %
on PP512 and 2.8 % on TG128.

### RX 9070 XT emulation, 3072 MiB budget, base at `-ncmoe 45`

| RAM | mode | PP512 | TG128 |
|---|---|---|---|
| 128 GB | base, no cache | 337.97, 325.27, 325.18, 321.26, 314.78 | 21.36, 21.00, 20.74, 19.12, 17.37 |
| 128 GB | Cold, exclusive | 621.97, 607.97, 587.09, 536.59, 465.16 | 21.60, 21.23, 20.93, 19.25, 17.40 |
| 128 GB | Warm, inclusive | 662.87, 638.14, 611.07, 558.11, 481.22 | 29.89, 30.80, 30.45, 26.66, 22.95 |
| 128 GB | Warm, exclusive | 615.13, 602.39, 577.88, 529.72, 459.98 | 29.77, 30.73, 30.42, 26.59, 22.92 |
| 128 GB | Warm, inclusive + prefill swap | 643.43, 630.11, 613.28, 559.30, 481.76 | 29.13, 30.42, 30.43, 26.64, 22.95 |
| 128 GB | Warm, exclusive + prefill swap | 659.75, 635.08, 612.71, 557.87, 482.24 | 29.74, 30.76, 30.27, 26.54, 22.91 |
| 128 GB | base again | 327.63, 322.93, 304.07, 313.91, 304.78 | 21.25, 20.91, 20.25, 19.17, 17.36 |
| 64 GB | Warm, exclusive | 517.19, 501.77, 486.27, 484.53, 401.05 | 29.20, 30.10, 29.75, 25.84, 22.40 |
| 64 GB | Warm, exclusive + prefill swap | 589.51, 570.30, 564.62, 524.99, 453.37 | 29.12, 29.40, 29.19, 26.09, 22.51 |
| 64 GB | base | does not fit (needs 96.4 GB) | |
| 32 GB | Warm, exclusive + prefill swap | 152.76, 140.23, 133.92, 136.15, 127.58 | 16.22, 17.79, 19.24, 17.00, 14.11 |
| 32 GB | base | does not fit (needs 96.4 GB) | |

Base-to-base drift: PP512 up to 21.1 t/s (6.9 %), TG128 up to 0.49 t/s (2.4 %). The 64 GB swap row
is inside the exclusive drift (0.40 t/s) on TG128 and 13.1 % above on mean PP512. The two 32 GB
repeats differ by up to 11 % on TG128; those are the slowest rows of the set (about 1400 s each) with
8.3 % of the selected bytes read from the file.

### Reading the tables

- **Decode.** The cache is worth +63 % / +42 % (depth 0 / 65536) with 20 GiB of VRAM and +40 % /
  +32 % with 3 GiB over the same VRAM used for whole expert layers. The Cold row shows that a random
  placement is worth nothing: the gain is the profile.
- **Prompt processing.** Base to Cold is not the cache; it is the per-layer embedding prefetch and
  gather of this series (`ple-prefetch.md`), which the base lacks. Cold to Warm is the cache: +47 %
  with 20 GiB, nothing with 3 GiB, because a prompt ubatch reads almost every expert once and only the
  resident fraction helps it (`expert-cache-prefill.md`).
- **Inclusive against exclusive.** Same decode within drift; exclusive uses 19.4 GiB less host
  memory at 20000 MiB (`expert-cache-exclusive.md`).
- **The prefill swap.** A few percent with unlimited host memory, +13 to +15 % of prompt throughput
  with a 40 GiB host tier, at a boundary install of 2.5 to 3.3 s per request there
  (`expert-cache-banks.md`).
- **Finite host tier.** Decode holds up until the host tier is a small fraction of the model; prompt
  processing pays earlier (`expert-cache-l2.md`).
- **The 32 GB rows.** At the R9700 placement the machine still decodes at 25 to 35 t/s with 8 GiB of
  host tier; at the 3 GiB placement the SSD carries 8 % of the selected bytes and the row is a
  different regime (14 to 19 t/s, 130 to 150 t/s prompt). Their decode throughput is not monotone in
  depth (it rises from depth 0 to 8192 before it falls); this was observed on both device rows and
  not diagnosed.

## Evidence

Every process ran under a memory and GPU watchdog that recorded the PID, the runtime module hashes,
the command line, the memory before and during the run and the recovery after exit; one GPU process
at a time; failed attempts kept as separate records. The base revision was built in its own build
directory from a detached worktree of `771fa0bda`. The GPU tuning of the machine was not changed for
this work.
