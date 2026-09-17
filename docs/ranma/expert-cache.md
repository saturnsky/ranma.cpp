# Expert cache: profiled static placement of MoE experts (`--expert-l1-mib`)

## What it is

A VRAM cache of MoE routed-expert weights for a model whose experts live in system RAM (host-direct
MoE, `docs/ranma/host-direct-moe.md`). The server profiles which experts the router selects while it
generates, the backend plans which experts are worth keeping in VRAM under a byte budget, and the
plan is installed at a request boundary. A decode step then reads a resident expert from VRAM and
every other expert from host memory over PCIe, through the same kernels.

Three parts, in the order the data flows:

1. **Profiler** (`ggml/src/ggml-cuda/expert-profiler.cu`). A small kernel runs after the top-k
   kernel of every routed layer and adds the selected expert ids of the profiled rows to a histogram
   in VRAM. Nothing is read back on the hot path. The rows and the histogram ("bank") are chosen by
   the caller before each decode; the server profiles the generation phase of slot 0 into a bank
   called `decode` and its prompt phase into a bank called `prefill` (`expert-cache-banks.md`).
2. **Profile store and plan** (`expert-profile-store.cpp`, `expert-score.h`, `expert-plan.h`). When
   a request ends the server commits the bank: the histogram since the last commit is stored as one
   record under `<profile-dir>/<bank>/records/`, the newest ten records are scored with a half-life
   of three requests, and a greedy plan by score x bytes fills the budget. The plan is a set of
   `(layer, expert)` pairs; it is not installed yet.
3. **Arena** (`expert-l1.cu`). One fixed-address arena per size class and kind (up, gate, down),
   packed by slot, plus one device table per layer that maps an expert id to its slot or -1. At the
   end of every request the newest plan is installed: experts that keep their slot stay, the rest are
   copied in, and the tables are published. The arena is allocated once at model load and never
   moves, because captured HIP graphs hold its address; only slot contents and tables change, and
   only while nothing computes.

The cache is inclusive by default: each VRAM resident also keeps its host copy, and the host tensor
stays complete. `--expert-cache-mode exclusive` gives every routed expert exactly one home instead
and takes the budget back out of host memory (`expert-cache-exclusive.md`). `--expert-l2-mib`
bounds the host memory the cache may use in either mode and leaves the rest of the experts in the
GGUF file, read on demand (`expert-cache-l2.md`).

This is a fork feature of the HIP build. It is not compiled into the CUDA backend; the options are
accepted there and do nothing.

## Why it exists

A decode step of a large MoE model on this setup is bound by PCIe reads of the selected experts.
Which experts get selected is far from uniform over a conversation, so a modest VRAM budget holds
the experts that carry most of the traffic, and every hit is a slice that does not cross the link.

Two alternatives were measured before this design was chosen: a hot cache filled on demand during
inference lost throughput at every budget, and an LRU stayed well below a static profiled placement.
The reasons are structural. A cache that fills during inference either stalls the decode on the copy
or needs a host round trip per token, and both cost more than the PCIe read they replace; a static
plan installed at a request boundary costs nothing per token. A recency-weighted profile of the last
few requests came within a few points of a placement that knew the next request in advance.

Two rules follow from HIP graphs: the arena and the tables are allocated exactly once and at fixed
addresses (a replayed graph does not revisit the host dispatch, so a lazily filled slot would never
be seen), and the plan is installed only while no compute is in flight.

The plan comes from the generation phase only, never from a histogram that mixes prompt and
generation selections. Prompt processing and generation select different experts, and a prompt
ubatch of hundreds of tokens outweighs single-token generation in any mixed count, so a mixed plan
would serve generation badly. Generation throughput is what this cache is for; how prompt
processing is served is the subject of `expert-cache-banks.md`.

## How it is layered

- `ggml/include/ggml-expert.h` is the whole contract: a config struct passed once at model load, and
  one versioned function table (`ggml_expert_iface`) that the backend answers for the proc address
  `ggml_backend_expert_iface`. The backend knows profile banks, plans and installs. It does not know
  requests, phases, slots or names.
- `llama.h` adds the `llama_expert_*` functions that forward to that table and turn a sequence id
  into a row range of every ubatch (`llama_expert_set_profiled_seq`). `llama_model_params` carries
  the config pointer; the model configures the backend before its buffers exist, registers the
  routed-expert context after allocation and finalizes (plan, allocate, install) after the weights
  are written.
- `common/expert.cpp` owns every rule about option combinations (`validate_expert_params`) and the
  translation from `common_params` to the config. `common/expert-policy.h`, driven by the server and
  by `llama-bench`, owns the policy: which slot is profiled, when a bank is committed, when a plan is
  installed.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--expert-l1-mib N` | 0 (off) | VRAM budget in MiB for expert payload and cache overhead. The budget also decides the expert placement (every routed expert goes to host memory), so `--n-cpu-moe`/`--cpu-moe` are refused together with it. `--expert-cache-mib` is accepted as an alias. Zero is valid together with a finite `--expert-l2-mib`. |
| `--expert-profile-dir DIR` | none | Root of the profile banks, `DIR/decode/` and `DIR/prefill/`. Without it the placement is seeded and fixed: no records, no installs. |
| `--expert-cache-mode MODE` | `inclusive` | `inclusive` keeps a host copy of each VRAM resident; `exclusive` keeps one home per expert (`expert-cache-exclusive.md`). |
| `--expert-l2-mib N` | -1 (unlimited) | Host memory budget in MiB; what fits in neither tier stays in the file (`expert-cache-l2.md`). 0 is refused. |
| `--expert-prefill-swap` | off | Hold the prompt-processing plan while a prompt is processed (`expert-cache-banks.md`). |
| `--expert-seed N` | 1 | Seed of the fixed random placement used when no profile is available. |
| `--expert-freeze` | off | Profile and plan, never change the cache contents (for collecting a profile without disturbing a measurement). |
| `--expert-profile-archive` | off | Records that leave the ten-record score window move to `DIR/<bank>/archive/` instead of being deleted. |
| `--expert-profile-reset` | off | Delete the stored records of every bank at startup. Also the way to replace a profile that belongs to another model or quantization. |

Most options are also environment variables of the usual form (`LLAMA_ARG_EXPERT_L1_MIB` and so
on). `llama-bench` takes the same options plus its own `--expert-cache off|cold|warm`
(`benchmark.md`).

Debug environment variables, read once at startup: `RANMA_EXPERT_TRACE=<mask>` turns on log lines
(1 install, 2 profile, 4 prompt-processing, 8 host tier and per-commit round lines);
`RANMA_EXPERT_VERIFY=1` reads every resident slice back after each install and compares it with its
source (slow; used by the correctness checks).

The profile directory is tied to the model: a `manifest.json` per bank records the routed-expert
geometry (layer count, expert count, per-kind bytes and types), and a store whose manifest does not
match the loaded model is ignored with a warning, not overwritten. Ten requests rebuild a profile
from cold. The store format is version 2; there is no reader for older layouts.

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. Model:
Qwen3.8-Flash-Next UD-Q4_K_XL (48 routed layers x 512 experts, 10 used, 73450 MiB of routed expert
weights). Protocol of `benchmark.md`: `llama-bench` PP512 / TG128, one repetition, depths 0, 4096,
8192, 32768 and 65536 in one model load after a discarded 65536 pass, host-direct on, `--load-mode
none`, unlimited host memory.

**Correctness.** The cache only changes where a resident expert's bytes are read from, so the same
binary must produce the same logits with the cache off and on. Eight prompts x 48 greedy tokens with
their top-3 logprobs were compared between the cache off and the cache on (exclusive, 3072 MiB, a
finite 8192 MiB host tier, prompt swap on, so every tier and both kernels are exercised): every token
and every logprob identical, and `RANMA_EXPERT_VERIFY` found no slice that differed from its source.
This machine produces one of two deterministic results per process for this prompt set with the
cache off, for a reason that is not the cache (the two diverge on one of the eight prompts from its
first token on); the cache run reproduces one of the two exactly.

**Decode.** TG128 in t/s at depth 0 / 4096 / 8192 / 32768 / 65536, Warm rows from a profile that a
Cold row of the same budget wrote:

| device row | budget | no cache (`-ncmoe`, base) | Cold, seeded random placement | Warm, exclusive |
|---|---:|---|---|---|
| R9700 | 20000 MiB | 23.90, 23.43, 23.10, 21.14, 18.95 | 24.37, 23.96, 23.57, 21.46, 19.10 | 38.90, 38.52, 37.62, 32.01, 26.89 |
| RX 9070 XT emulation | 3072 MiB | 21.36, 21.00, 20.74, 19.12, 17.37 | 21.60, 21.23, 20.93, 19.25, 17.40 | 29.77, 30.73, 30.42, 26.59, 22.92 |

The "no cache" row is the published base revision with `-ncmoe 35` (13 expert layers in VRAM, the
same VRAM as the 20000 MiB budget) and `-ncmoe 45` (3 layers, close to the 3072 MiB budget). The
RX 9070 XT row is not a measurement on that card: it is the same R9700 with the placement a 16 GiB
card would use, device buffers under 15 GiB at depth 65536.

Three things the table says. The same VRAM is worth far more as a profiled budget than as whole
layers: +63 % at depth 0 and +42 % at depth 65536 with 20 GiB, +39 % and +32 % with 3 GiB. A random
placement of the same budget (the Cold row) is worth nothing on decode, so the gain is the profile,
not the VRAM. And the gain persists at long context, where attention takes a growing share of the
step: the cache removes expert reads, and those are a smaller part of a 64K-context step.

**Prompt processing.** PP512 at the same depths:

| device row | budget | no cache (base) | Cold | Warm, exclusive |
|---|---:|---|---|---|
| R9700 | 20000 MiB | 349.89, 343.86, 338.34, 331.30, 322.30 | 636.56, 656.79, 634.96, 573.21, 493.20 | 937.43, 927.21, 898.39, 771.49, 631.95 |
| RX 9070 XT emulation | 3072 MiB | 337.97, 325.27, 325.18, 321.26, 314.78 | 621.97, 607.97, 587.09, 536.59, 465.16 | 615.13, 602.39, 577.88, 529.72, 459.98 |

Here the attribution matters. The difference between the base and the Cold row is not the cache:
the base revision predates the per-layer embedding prefetch and parallel gather (`ple-prefetch.md`),
and with 50 to 70 GB of experts in host memory its embedding gather pays the expensive page faults
that page describes. The cache's own effect on prompt processing is Cold against Warm: +47 % with a
20 GiB budget, nothing with a 3 GiB budget. A prompt ubatch reads almost every expert of a layer
once, so what helps it is the resident fraction, not the hit rate on generation
(`expert-cache-prefill.md`).

**Install cost.** From the install log of an inclusive 3072 MiB run of the same binary: a first
install writes 1027 slices in about 70 ms, and the installs that follow retain 958 to 974 of those slices and copy
53 to 69 (159 to 207 MiB) in 11 to 12 ms. A first install of a 20000 MiB budget is a few hundred
milliseconds. With `llama-bench` this time is reported in its own column (`ctl ms`) and is not part
of the throughput above; on the server it lands between two requests.

## What it costs

- VRAM: the budget, plus 4 bytes per (layer, expert) for the tables, 16 bytes per (layer, expert)
  for the profiler banks and a zeroed tail per size class and kind for the MMQ read-ahead
  (`ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)`, at least 512 bytes), all
  inside the budget.
- Host memory: inclusive mode keeps every expert on the host, so the VRAM residents are a second
  copy. Process private memory peaks in the runs above: 80.7 GiB with no cache, 97.6 GiB inclusive at
  20000 MiB, 78.2 GiB exclusive at 20000 MiB.
- Per decode token: one 128-thread kernel per routed layer (the histogram add) and one table read
  per expert in the kernel prologue. Nothing is synchronized.
- At request end: one device synchronize, a 200 KiB device-to-host copy per bank, one record write.
- At install: the copies of the experts that changed, on a separate stream, with the device drained.

## Limitations

- **One model per process.** A second routed model loaded in the same process runs uncached with a
  warning. This matches the single-user server the fork targets.
- **Slot 0 is profiled.** With `--parallel N > 1` the other slots use the cache but do not feed the
  profile.
- **HIP only, host-direct required.** The experts must sit in the HIP host buffer type.
  `--expert-l1-mib` puts every routed expert there by itself and logs one line saying so, which is
  why it cannot be combined with `--n-cpu-moe`/`--cpu-moe`. A manual `-ot` on `_exps` tensors is left
  alone and is the user's responsibility; if the experts end up in VRAM anyway there is nothing to
  cache and the option is ignored with a log line.
- **`--load-mode none` is required.** With the default mmap load the experts are mapped into
  `CPU_Mapped` buffers, host-direct never engages and the cache finds no routed context (it logs
  "not available for this model"). The loader already warns about that combination.
- **The arena is read on the in-place kernels only.** MMVQ (decode) and MMQ (prompt processing up
  to `GGML_CUDA_HOST_DIRECT_MAX_BATCH` tokens) read it. A larger batch, or host-direct off, copies the
  whole expert weight to VRAM through the scheduler and reads the copy; that path is untouched.
- **Startup without a profile is a random placement.** The server seeds a fixed placement from
  `--expert-seed` and evolves it at request boundaries once a profile directory is given; the first
  requests run at about the no-cache speed.
- **Prompt processing and the copy path do not produce bit-identical logits with each other.** The
  copy path splits the graph around every expert weight, which changes what the backend fuses. A
  cache check compares off and on under one `GGML_CUDA_HOST_DIRECT_MAX_BATCH` setting.

## Revision

Every number above comes from the binary built from the commits that add this feature on ranma
`ccd8fd1e9` (upstream llama.cpp `093a2f86c`, ggml-org master of 2026-09-14, plus the fork's earlier
commits), against the published snapshot `ranma_20260914` (`771fa0bda`) as the base row. The exact
commits are kept on the dated snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that
branch is never rebased.
