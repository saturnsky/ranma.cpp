# Expert cache: one cache for every routed model of a process (`--expert-cache-draft`)

## What it is

When the server loads a separate draft model that has routed experts (for example a block drafter
with MoE stages) and the expert cache is on, the draft joins the target's cache: one VRAM arena
(`--expert-l1-mib`) and one host budget (`--expert-l2-mib`) serve both models, and one greedy plan
decides which `(model, layer, expert)` slices live in VRAM. Without the joint cache the draft keeps
its own placement flags (`-ngld`, `-otd`) and its whole expert weight either takes VRAM away from the
target's arena or is read from host memory on every draft call.

`--expert-cache-draft off` keeps the old behaviour: the draft is loaded as before and never profiled.

## How it works

- **Declaration.** The target's config names the draft's GGUF file. When the target's routed
  context is registered, the backend reads the draft's tensor layout from the file metadata (no
  data, split files follow `split.count`) and lays the geometries end to end: the target's layers
  are joint layers `0..n-1`, the draft's layers follow. The size classes are recomputed over the
  union, so the draft's MXFP4 slices form their own class next to the target's classes. The
  load-time plan and the arenas therefore cover both models before the first expert byte is written.
- **Binding.** The draft's config names its own file and is accepted as that member. Its routed
  weight context must have the layout that was read from the file; then the joint geometry takes its
  tensors. A draft whose routed experts do not all land in the host buffer type, or whose layout
  differs, is not cached (log line), and in exclusive mode its slots stay unused.
- **Load order.** Exclusive mode needs the arenas before the loader writes, so they are built when
  the target loads and the target is served (warm-up included) before the draft is loaded; the
  draft's slices are written straight into their homes when it loads. Inclusive mode builds the
  arenas once every declared model has loaded; until then lookups miss and the host tensors are read.
- **Placement.** Each model keeps its own half-life score per bank. The joint score of a slice is
  the model's score times its weight (`--expert-cache-weight`, default 1 for every model), and the
  existing greedy ranks every slice of every model by that score per byte, with no per-model floor
  or cap. With weight 1 this is the plain frequency x bytes greedy. The per-class capacities are
  frozen at load as before, over the union.
- **Profiling.** Every compute context attaches to the model it runs (`attach_backend`), and the
  router selections of a context count only into its own model's layers. A context that runs no
  cached model (a draft outside the cache) is not profiled. Each model has its own row selection, so
  the draft's context counts every row of its block decodes into the `decode` bank while the server
  is generating, and nothing outside generation. The draft only routes in block decodes, so its
  experts appear only in the decode plan.
- **Profiles.** `<profile-dir>/<model key>/<bank>/records/` per model, where the model key is the
  architecture name plus a 64-bit hash of the routed-expert geometry signature (layers, types,
  shapes, bytes per expert). The manifest keeps the full signature, so a profile of another model or
  geometry is ignored with a log line. A cache of one model uses the same layout and the same key,
  so a model finds its records whether it runs alone or with a draft in the joint cache.
- **Modes.** Inclusive and exclusive both work; the mode is one for the whole cache. The prefill swap
  and a finite L2 are refused together with speculative decoding, as before (see limits).

## Options

| Option | Default | Meaning |
|---|---|---|
| `--expert-cache-draft on\|off` | on | A draft model with routed experts shares the cache. With `on`, a draft override that places its routed experts (`-otd ...exps...`) is refused, and so are `--cpu-moe-draft`/`--n-cpu-moe-draft` (as for the target). Env `LLAMA_ARG_EXPERT_CACHE_DRAFT`. |
| `--expert-cache-weight MODEL=W[,...]` | `target=1,draft=1` | Multiplier on the host read cost of one model's experts in the joint greedy. A draft block decode is short and serial before verification, so one host byte of the draft may cost more time than one of the target; the weight expresses that. Env `LLAMA_ARG_EXPERT_CACHE_WEIGHT`. |

## Log lines

- `expert cache: model <key> (<path>) is declared to join: ...` at configure.
- `expert cache: model <i> <key>: joint layers a..b, weight w` and one `size class` line per class.
- `expert cache: model <key> (<path>) joins the cache as model <i>` and `model <i> <key> bound`.
- `expert cache: plan budget=... models=<key>*<w>,...` and `installed ... models=...` (inclusive).
- `expert cache: model <i> <key> at model load|after a model joined|after install: L1 n experts / MiB, host n experts / MiB`.
- `expert cache: compute context <n> runs model <i> (<key>)`.

## Limits and fallbacks

- **One expert count.** All models of a joint cache must route the same number of experts per layer;
  a model that does not is not cached.
- **Finite L2.** Not implemented for a joint cache: the SSD tier numbers the files of one model. The
  server refuses a finite L2 with speculative decoding anyway; a backend config that asks for both
  caches the target only.
- **Prefill swap.** Refused with speculative decoding, as before. A plan that swaps only the draft's
  share is not implemented.
- **One device.** The draft's routed experts must be on the cache's device.
- **Frozen classes.** The share of each size class is fixed at load from the stored profiles (or the
  cold split); later plans move slices inside the classes only.
