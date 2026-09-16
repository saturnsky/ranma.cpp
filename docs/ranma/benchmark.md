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

All expert options of the server are accepted: `--expert-cache-mode`, `--expert-l1-mib`,
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
display attached has 1 to 2 GB less VRAM, which for this model is about one expert layer. **RAM
rows.** A machine with less than 128 GB is emulated by bounding the host tier to the emulated RAM
minus 24 GB: 128 GB = `--expert-l2-mib -1`, 64 GB = `40960`, 32 GB = `8192`. This emulates the
capacity only. The page-cache behaviour of a smaller machine is not emulated, so the file-tier cost
of the 64 GB and 32 GB rows is a lower bound. The base revision needs the experts outside VRAM in
host memory: 56.7 GB (`-ncmoe 35`) or 72.5 GB (`-ncmoe 45`) plus the 24 GB, so it fits the 128 GB
row only and the other rows say "does not fit".
