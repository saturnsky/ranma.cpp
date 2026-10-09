# Benchmark method

The conditions that the records in this folder share. Every record names at its top the release it measured, the
upstream commit that release is based on and the date, and states the settings of its own rows. What is said here
holds for every record unless the record says otherwise. The expert cache that most rows use is described in
[expert-cache.md](../expert-cache.md).

## Machine

Radeon AI PRO R9700 32 GiB (gfx1201, 64 CU) on PCIe 5.0 x16, Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11,
`amdhip64_7.dll` 10.0.3581.0 shipped next to the binaries. The GPU runs with a -30 % power limit and a 0 mV voltage
offset for every measurement, and headless (no display attached; the CPU's integrated GPU drives the display), so all
32 GiB are available to the model. Host-resident weights are bound by the PCIe link, so a narrower or slower link
changes these numbers first. The builds compared in one record are made with the same toolchain and options.

## Server rows

Every server row is one fresh `llama-server` process that replays one preset through the
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

## `llama-bench` rows

Every `llama-bench` row is one process:

```
llama-bench -m MODEL -ngl 999 -t 16 -fa on -ctk f16 -ctv f16 -b 512 -ub 512 -lm none --poll 0 \
    -p 512 -n 128 -r 1 -d 65536,0,4096,8192,32768,65536 <placement options of the row>
```

with `HIP_VISIBLE_DEVICES` selecting the R9700. The first pair of PP512 and TG128, at the deepest depth, is a
discarded warm-up: on this platform the prompt throughput at shallow depths rises with the elapsed time of the
process, so every process starts with the same deepest pass and the table holds the remaining depths in the
order they ran. The rows of a process share one model load. One GPU process runs at a time.

## Placement

The rows with `-ncmoe N` keep the routed experts of the first N layers in host memory, the way upstream places a
model that does not fit. The expert cache rows have no `-ncmoe`: the cache owns the placement and
`--expert-l1-mib` is its VRAM budget. They need host-direct, `GGML_CUDA_HOST_DIRECT=1` and
`GGML_CUDA_HOST_DIRECT_MAX_BATCH=512` ([host-direct-moe.md](../host-direct-moe.md)), in the environment of the row.

## Host memory: 128, 64 and 32 GB

The RAM rows emulate a machine with less than 128 GB by bounding the host tier of the expert cache to the
emulated RAM minus 24 GB: 128 GB is `--expert-l2-mib -1`, 64 GB is `40960`, 32 GB is `8192`
([expert-cache-l2.md](../expert-cache-l2.md)). Up to the 2026-10-05 records this emulated the capacity only. The
page cache of the 128 GB machine still held the model file, so the file-tier cost of those 64 GB and 32 GB rows is
a lower bound.

From the 2026-10-09 record on, the 64 and 32 GB rows also take the physical memory away. Before the process starts,
a separate process allocates locked physical pages with `AllocateUserPhysicalPages`, so that the available memory is
the emulated RAM minus 12 GiB (52 and 20 GiB). Those pages are not paged out and the page cache cannot use them, so
file-tier reads go to the SSD as on the smaller machine. The pages are released after the row. With mmap, the device
weights stay mapped as clean file pages, which Windows can drop at no cost, so the run's memory floor counts them as
available.

Upstream and the `-ncmoe` rows need every expert outside VRAM in host memory, so they exist for 128 GB only.

## Cold and warm

**Server.** A cold row starts with an empty profile directory: the process loads the seeded placement and
warms itself from its own records, installing a new plan after the requests as a server does. A warm row starts
from its own copy of a profile that the warm-up preset of the same scenario recorded in a separate process (seed
rows, not in the tables).

**`llama-bench`.** `--expert-cache cold` runs the whole process from the seeded placement and writes a profile;
`warm` runs from that profile ([expert-cache.md](../expert-cache.md)). A cold `llama-bench` row therefore never
installs a plan, unlike a cold server row, and shows the seeded placement alone.

## Rests

No row starts on a card that the row before it has heated: between two GPU processes the card rests for 300 s, as
a fixed rest or as a temperature gate capped at 300 s (600 s from the 2026-10-09 record on, with the card kept awake so that its fan runs). Each record says which. Scenarios of the server records
alternate the order of their rows where a record says so.

## Memory columns

The memory tables give the maximum over the life of a process, in GiB, of the process alone. The records say how
often the process was sampled.

- **VRAM dedicated** is the memory on the card. It matches the placement rule of the record: under 30 GiB on
  the R9700 rows, about 12 GiB on the RX 9070 XT rows of the full set.
- **GPU shared** is host memory the GPU addresses: the pinned host buffers, which hold the experts of the
  `-ncmoe` rows and the host tier of the expert cache. It is part of the private bytes, not in addition.
- **Private bytes** is the commit charge of the process. On Windows it also carries the backing of the VRAM
  allocations: a process with the whole model in VRAM (Gemma) commits 17 GiB before its working set reaches
  5 GiB. So this column bounds the host memory a row needs from above, by up to the VRAM column.
- **Working set** is what was resident, the mapped pages of the model file included. It is neither a floor
  nor a ceiling of what a row needs: file pages are page cache the system reclaims, and committed memory
  the process has not touched is not in it.

## Single rows

- **One pass.** Every configuration is one row, measured once; no row was repeated to pick the better run. The
  records name the few cells that were repeated.
- **Equal rows.** Differences within about 1 % are inside what one pass of single rows can resolve on this
  machine and are read as equal.
- **Replies.** Rows use greedy decoding, so the replies of two rows can be compared. Upstream generates different
  replies from a release, and MTP from no MTP.
