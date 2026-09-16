# Per-layer embedding prefetch and gather (`--ple-prefetch`)

## What it is

`--ple-prefetch {off,prefill,always}` hands the rows that a per-layer-embedding (PLE) gather is
about to read to the operating system before the gather runs.

| Option | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--ple-prefetch off` | `LLAMA_ARG_PLE_PREFETCH` | | Never prefetch. |
| `--ple-prefetch prefill` | `LLAMA_ARG_PLE_PREFETCH` | | Prefetch gathers of 256 rows or more, which is a prompt ubatch. |
| `--ple-prefetch always` | `LLAMA_ARG_PLE_PREFETCH` | default | Also prefetch the 16-row gather of a single decoded token. |

It is a common option, so `llama-server`, `llama-cli` and the other common tools take it;
`llama-bench` parses the same three values and applies one value to the whole run.

Independently of the option, a CPU `GET_ROWS` on a tensor named exactly
`per_layer_token_embd.weight` runs on the threadpool when it gathers 256 rows or more. Every other
`GET_ROWS` keeps the single-thread path.

The option does nothing for a model without a per-layer embedding table, and nothing when that table
is not in host memory.

## Why it exists

The per-layer embedding table of a model that has one is a multi-GiB tensor that the loader maps
lazily and never reads, so every gathered row is a demand page fault. The model gathers one small
row per layer per token, so a prompt ubatch is thousands of scattered faults, and each fault reads a
whole page for a row of about a hundred bytes.

The price of a fault depends only on whether the page is already in the page cache. With plenty of
free RAM it is a few microseconds and the gather is short. When another consumer of RAM leaves
little free, the same number of faults costs two orders of magnitude more each, and the gather can
take most of a prompt ubatch. The fault count does not change; only its price does.

Upstream keeps `GET_ROWS` on one thread because extra threads cost more than they save for the
small gathers of a GPU-offloaded graph. That holds for every other gather, but here one thread means
one outstanding fault at a time. Two changes address that:

- the big gather is spread over the threadpool, so the faults are concurrent. The gathered rows are
  disjoint per thread, so the output bytes are the same as on the single-thread path;
- the rows the gather is about to read are handed to the operating system first, so it can read them
  together. They are sorted, deduplicated and merged into ranges - neighbouring rows become one
  range - and the whole set is submitted in a single call.

Both are hints about page residency, not about arithmetic. The gather still reads the same bytes.

## How it works

`src/llama-prefetch.{h,cpp}` is the OS wrapper: `PrefetchVirtualMemory` on Windows, resolved
dynamically so that the link line does not change and an older host simply gets no prefetching;
`madvise(MADV_WILLNEED)` on POSIX; nothing elsewhere. The call is best-effort: it cannot fail in a
way the caller has to handle, and its result is ignored.

The prefetch is issued by the graph input that builds the gather index list, before that list is
uploaded, so the rows are already being read while the rest of the ubatch is set up. A table whose
buffer is not host memory is skipped, because a device pointer is not page-faulted by the CPU.

The thread-count decision lives in the CPU backend's task-count function and is made by tensor name.
A name is what that function has at hand; a tensor flag or an op parameter would keep the backend
independent of model tensor names but would need a format or graph-builder change. The cost of the
current choice is that the CPU backend knows one model tensor name.

## Thresholds

| value | prefetch applies to |
| --- | --- |
| `off` | nothing; the gather is still parallel from 256 rows on |
| `prefill` | gathers of 256 rows or more, i.e. prompt ubatches |
| `always` (default) | gathers of 16 rows or more, i.e. prompt ubatches and single decoded tokens |

256 rows is where the prompt measurement settled. 16 rows is one decoded token; it is the default
because it helps decode and costs nothing on prompt processing.

## Limits

- Measured on Windows only; the POSIX path is written but not measured.
- The gather parallelization is gated on one tensor name, so a model without a per-layer embedding
  table sees no change at all.
- A table that is not in host memory is skipped.
- The benefit depends on how much free RAM the machine has: with an ample page cache the faults are
  cheap and there is little to save.

## How to verify

- `--ple-prefetch off` against the default on a model with a per-layer embedding table, with the
  table lazily mapped, shows the effect on prompt and decode throughput.
- Outputs must be identical between the three values: the changes affect page residency and thread
  count only.
