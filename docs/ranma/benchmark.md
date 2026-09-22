# Benchmarks

The numbers of this fork against upstream, measured once for the whole series. The feature pages describe what
each change does and how to switch it; this page holds the throughput. The `llama-bench` modes of the expert
cache that the rows below use are described in [expert-cache.md](expert-cache.md).

## Protocol

Measured on the development machine, 2026-09-22: Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16,
Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. The GPU runs with a -30 % power limit and a 0 mV
voltage offset for every measurement here, and headless (no display attached; the CPU's integrated GPU drives
the display), so all 32 GiB are available to the model. Host-resident weights are bound by the PCIe link, so a
narrower or slower link changes these numbers first.

Upstream is `aa39d7a3e`, the revision this series is based on, built in its own build directory with the same
toolchain and options. "This fork" is the last commit of the series.

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
set" is this fork as it behaves with no environment variable and no new option. Everything else is at its
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

## Qwen3.8-Flash-Next UD-Q4_K_XL

PP512 and TG128 in t/s at the depth after the `@`.

### R9700

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream, `-ncmoe 35` | 335 | 330 | 316 | 316 | 307 | 15.9 | 15.8 | 15.2 | 14.4 | 13.2 |
| this fork, `-ncmoe 35`, no switch set | 590 | 564 | 542 | 497 | 447 | 17.3 | 16.9 | 17.0 | 16.3 | 16.2 |
| this fork, `-ncmoe 35`, host-direct | 650 | 621 | 598 | 546 | 481 | 26.8 | 26.2 | 26.1 | 25.5 | 24.9 |
| this fork, expert cache 20480 MiB | 916 | 886 | 861 | 762 | 626 | 50.4 | 49.2 | 48.8 | 46.7 | 44.7 |

### RX 9070 XT emulation

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream, `-ncmoe 45` | 310 | 307 | 301 | 297 | 290 | 13.7 | 13.5 | 13.3 | 12.5 | 11.6 |
| this fork, `-ncmoe 45`, host-direct | 649 | 620 | 594 | 551 | 484 | 23.6 | 23.1 | 23.0 | 22.6 | 22.2 |
| this fork, expert cache 3072 MiB | 622 | 612 | 588 | 544 | 480 | 37.6 | 39.0 | 39.2 | 37.5 | 36.0 |

### By system

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

## DeepSeek V4 Flash UD-IQ3_XXS

### R9700

The rows with `-ncmoe 35` stop at depth 8192 (`-d 8192,0,4096,8192`, first pair discarded): with the experts
of 35 layers behind the PCIe link, the fill of 65536 tokens takes longer than the watchdog of the measurement
lets a process run without output. These rows hold about 99 GiB of host memory, so there is no upstream row
for the RX 9070 XT placement, which would hold more.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | TG128 @0 | TG128 @4096 | TG128 @8192 |
|---|---:|---:|---:|---:|---:|---:|
| upstream, `-ncmoe 35` | 248 | 228 | 209 | 9.8 | 9.8 | 9.7 |
| this fork, `-ncmoe 35`, host-direct | 265 | 244 | 219 | 17.5 | 17.3 | 17.2 |

With the expert cache the full curve fits:

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | PP512 @65536 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 | TG128 @65536 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| this fork, expert cache 20480 MiB | 310 | 276 | 255 | 161 | 108 | 31.8 | 30.8 | 30.9 | 29.7 | 28.1 |

### By system

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

## Gemma 4 31B Q4_K_M

R9700, dense, the whole model in VRAM, `-d 32768,0,4096,8192,32768` with the first pair discarded, and
`ROCBLAS_USE_HIPBLASLT=1` for both builds.

| configuration | PP512 @0 | PP512 @4096 | PP512 @8192 | PP512 @32768 | TG128 @0 | TG128 @4096 | TG128 @8192 | TG128 @32768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| upstream | 897 | 735 | 628 | 328 | 28.3 | 26.7 | 26.3 | 24.2 |
| this fork | 1001 | 868 | 762 | 440 | 29.3 | 27.3 | 26.9 | 24.7 |

## Memory of each row

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
| Qwen, upstream, `-ncmoe 35` | 29.9 | 53.1 | 80.7 | 64.8 |
| Qwen, `-ncmoe 35`, no switch set | 29.7 | 53.2 | 80.6 | 64.7 |
| Qwen, `-ncmoe 35`, host-direct | 28.6 | 53.1 | 80.5 | 64.7 |
| Qwen, expert cache 20480 MiB, 128 GB, exclusive | 29.1 | 52.8 | 78.0 | 64.4 |
| Qwen, expert cache 20480 MiB, 128 GB, exclusive + prefill swap | 29.1 | 52.8 | 78.2 | 64.4 |
| Qwen, expert cache 20480 MiB, 64 GB, exclusive | 29.0 | 40.9 | 66.1 | 52.5 |
| Qwen, expert cache 20480 MiB, 64 GB, exclusive + prefill swap | 29.0 | 40.9 | 66.3 | 52.5 |
| Qwen, upstream, `-ncmoe 45` | 15.3 | 67.8 | 80.7 | 79.4 |
| Qwen, `-ncmoe 45`, host-direct | 13.9 | 67.8 | 80.5 | 79.3 |
| Qwen, expert cache 3072 MiB, 128 GB, exclusive | 12.1 | 69.8 | 77.7 | 81.4 |
| Qwen, expert cache 3072 MiB, 64 GB, exclusive | 12.0 | 40.9 | 48.8 | 52.5 |
| Qwen, expert cache 3072 MiB, 64 GB, exclusive + prefill swap | 12.0 | 40.9 | 48.8 | 52.5 |
| Qwen, expert cache 3072 MiB, 32 GB, exclusive | 11.9 | 8.9 | 16.7 | 20.5 |
| Qwen, expert cache 3072 MiB, 32 GB, exclusive + prefill swap | 11.9 | 8.9 | 16.7 | 20.5 |
| DeepSeek, upstream, `-ncmoe 35` | 27.8 | 74.3 | 98.7 | 74.8 |
| DeepSeek, `-ncmoe 35`, host-direct | 23.9 | 74.3 | 98.8 | 74.8 |
| DeepSeek, expert cache 20480 MiB, 128 GB, exclusive | 27.9 | 71.4 | 90.3 | 72.4 |
| DeepSeek, expert cache 20480 MiB, 128 GB, exclusive + prefill swap | 27.9 | 71.4 | 90.5 | 72.4 |
| DeepSeek, expert cache 20480 MiB, 64 GB, exclusive | 27.9 | 40.6 | 59.5 | 41.6 |
| DeepSeek, expert cache 20480 MiB, 64 GB, exclusive + prefill swap | 27.9 | 40.6 | 59.7 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 64 GB, exclusive | 11.8 | 40.6 | 49.0 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 64 GB, exclusive + prefill swap | 11.8 | 40.6 | 49.0 | 41.6 |
| DeepSeek, expert cache 4096 MiB, 32 GB, exclusive | 11.9 | 8.6 | 16.7 | 9.3 |
| DeepSeek, expert cache 4096 MiB, 32 GB, exclusive + prefill swap | 11.9 | 8.5 | 16.7 | 9.3 |
| Gemma, upstream | 23.8 | 0.2 | 21.0 | 20.7 |
| Gemma, this fork | 23.3 | 0.2 | 21.0 | 20.8 |

Two things follow. The RAM rows hold what they were set to: the 32 GB rows commit 17 GiB, of which 9 GiB is
pinned host memory (the host tier), and the 64 GB rows on the 16 GiB placement commit 49 GiB with 41 GiB
pinned. The 64 GB rows on the R9700 placement commit 66 GiB because the 20 GiB expert cache in VRAM is
charged too; their pinned host memory is the same 41 GiB. The upstream and `-ncmoe` rows of DeepSeek commit
99 GiB, the reason those rows exist for 128 GB only.

The expert cache budgets were chosen for the card, and chosen conservatively: 20480 MiB for the R9700 and
3072 / 4096 MiB for a 16 GiB card leave 2 to 4 GiB of the 32 GiB and about 4 GiB of the 16 GiB unused (1 to
2 GB of which a display would take). A larger `--expert-l1-mib` fills that with experts, so a machine with
either card can allocate more than these rows did. What that gains was not measured.

## Reading the tables

- **Qwen, R9700.** Against upstream at the same `-ncmoe 35` placement, this fork with no switch set decodes
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
