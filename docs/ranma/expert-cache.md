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
   in VRAM. Nothing is read back on the hot path.
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
and takes the budget back out of host memory (`expert-cache-exclusive.md`).

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
would serve generation badly.

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
| `--expert-l1-mib N` | 0 (off) | VRAM budget in MiB for expert payload and cache overhead. The budget also decides the expert placement (every routed expert goes to host memory), so `--n-cpu-moe`/`--cpu-moe` are refused together with it. `--expert-cache-mib` is accepted as an alias. |
| `--expert-profile-dir DIR` | none | Without it the placement is seeded and fixed: no records, no installs. |
| `--expert-cache-mode MODE` | `inclusive` | `inclusive` keeps a host copy of each VRAM resident; `exclusive` keeps one home per expert (`expert-cache-exclusive.md`). |
| `--expert-seed N` | 1 | Seed of the fixed random placement used when no profile is available. |
| `--expert-freeze` | off | Profile and plan, never change the cache contents (for collecting a profile without disturbing a measurement). |
| `--expert-profile-archive` | off | Records that leave the ten-record score window move to `DIR/<bank>/archive/` instead of being deleted. |
| `--expert-profile-reset` | off | Delete the stored records of every bank at startup. Also the way to replace a profile that belongs to another model or quantization. |

Most options are also environment variables of the usual form (`LLAMA_ARG_EXPERT_L1_MIB` and so
on). `llama-bench` takes the same options plus its own `--expert-cache off|cold|warm`
(`benchmark.md`).

Debug environment variables, read once at startup: `RANMA_EXPERT_TRACE=<mask>` turns on log lines (1
install, 2 profile, 4 prompt-processing); `RANMA_EXPERT_VERIFY=1` reads every resident slice back
after each install and compares it with its source (slow; used by the correctness checks).

The profile directory is tied to the model: a `manifest.json` per bank records the routed-expert
geometry (layer count, expert count, per-kind bytes and types), and a store whose manifest does not
match the loaded model is ignored with a warning, not overwritten. Ten requests rebuild a profile
from cold. The store format is version 2; there is no reader for older layouts.

## What it costs

- VRAM: the budget, plus 4 bytes per (layer, expert) for the tables, 16 bytes per (layer, expert)
  for the profiler banks and a zeroed tail per size class and kind for the MMQ read-ahead
  (`ggml_row_size(type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING)`, at least 512 bytes), all
  inside the budget.
- Host memory: inclusive mode keeps every expert on the host, so the VRAM residents are a second
  copy.
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
