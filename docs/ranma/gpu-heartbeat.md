# GPU heartbeat (llama-server)

## What it is

A worker thread in `llama-server` that records one GPU event on every device holding model buffers, once per
interval, for as long as the server has nothing else to submit. It runs in two phases:

- **idle** - every slot is idle, so nothing else touches the GPU.
- **teardown** - the model and its contexts are being freed.

The worker owns no model data. It creates one extra `ggml_backend_t` per device (`ggml_backend_dev_init(dev, nullptr)`)
and one `ggml_backend_event_t` per device, records the events once per interval, and completes the previous pulse
before recording the next, so at most one pulse is ever outstanding. A pulse costs about 50 us of CPU per device and
submits no work beyond the event itself.

Two rules keep it out of the way of real work:

- `set_idle(false)`, called when a slot becomes busy and before any state-changing task (slot save / restore / erase,
  cache clear, LoRA set), blocks until no pulse is in flight. After it returns, the caller owns the model's GPU state.
- `begin_shutdown()`, the first statement of the server's `destroy()`, records and completes a pulse immediately and
  then keeps pulsing at the interval until `release()`, which joins the worker after the last model buffer is freed.

Implementation: `tools/server/server-gpu-heartbeat.h`, hooked from `tools/server/server-context.cpp`.

## Why it exists

On Windows, the video memory manager (VidMm) evicts a process' entire VRAM residency to system RAM after a short
period without a GPU submission from that process (about 10 s on the development machine). The eviction is per
process and is not triggered by memory pressure, so an idle `llama-server` that holds its weights and KV cache in
VRAM loses its residency during any ordinary gap between turns in a conversation. Three costs follow:

- **Next-request latency.** The resident set is paged back over PCIe ahead of the first kernel, so the first decode
  after an idle gap stalls for as long as that page-back takes; the larger the resident set and the slower the
  link, the longer the stall.
- **Host RAM while idle.** The evicted VRAM lands in system RAM (the process' working set grows by the size of its
  resident set) and stays there until the next request pages it back. On a machine whose RAM is already mostly
  pinned expert weights, an idle server that grows by the size of its VRAM is a stability problem, not just a
  latency one.
- **Slower teardown after an idle gap.** Freeing allocations that were evicted first pages them back into VRAM, so a
  shutdown that follows an idle gap takes longer than one that follows a request.

Recording a single event on the device is enough to reset the policy timer; no real work has to be submitted. How
much each of the three costs amounts to depends on the resident set, the PCIe link and the driver; the numbers for
one machine are under [Measured effect](#measured-effect).

## Multi-device rationale

Residency is managed per adapter, so a pulse on one device does nothing for another. The worker therefore creates a
backend and an event for *every* device that holds model buffers (the devices the target model was placed on, plus
the draft model's when speculative decoding is enabled, as reported by `llama_model_get_device`) and pulses all of them in one pass per interval: record on each, then
synchronize each at the next pulse. The cost is one event record per device per interval, which stays negligible for
any realistic device count.

A device is skipped with a warning if its backend cannot be initialized or if it does not support events
(`ggml_backend_event_new` returns `NULL`). If no device is left, the heartbeat stays disabled. Only devices of type
`GGML_BACKEND_DEVICE_TYPE_GPU` and `GGML_BACKEND_DEVICE_TYPE_IGPU` are considered.

## Option

| Option | Env | Default | Meaning |
|---|---|--:|---|
| `--gpu-heartbeat-seconds SECONDS` | `LLAMA_ARG_GPU_HEARTBEAT_SECONDS` | `0` | Interval in seconds between pulses; fractions are accepted (`0.5`). `0` disables the feature. Values outside 0-3600 or non-numeric strings are rejected. |

The default is `0` so that upstream semantics are unchanged unless the option is asked for: without it the server
submits nothing while idle, exactly as before.

The recommended value is **5 seconds** - comfortably inside the eviction window observed here (about 10 s), with enough margin that a
scheduling delay on a loaded machine does not let the window expire, and far too cheap to measure against a server
that is otherwise doing nothing.

```
llama-server -m model.gguf --gpu-heartbeat-seconds 5
```

## Measured effect

Measured 2026-09-13 on a Radeon AI PRO R9700 (32 GiB, PCIe 5.0 x16, tuned -75 mV / power limit -30 %, headless with
the display on the iGPU), Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10, with Qwen3.8-Flash-Next
UD-Q4_K_XL (103.68 GiB) and the host-direct profile `GGML_CUDA_HOST_DIRECT=1 GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`.
Two expert placements, as in [host-direct-moe.md](host-direct-moe.md): `-ncmoe 35` (26.4 GiB resident in VRAM, the
R9700 row) and `-ncmoe 45` (11.4 GiB resident, the RX 9070 XT emulation row; same silicon, only the 16 GiB VRAM
budget is reproduced, so this is not a measurement on that card). Server: `-c 8192 -np 1 -fa on`, one process per
configuration, A/B/A order off / on / off. Each process answered one warm-up request, then one 64-token request after
an idle gap of 5, 15, 30 and 60 s (same prompt, `temperature 0`; the prompt cache leaves 4 tokens to re-process, so time to
first token is one 4-token prefill step plus the first sampled token, i.e. the first GPU work after the gap), then received Ctrl+C. A sampler read the process' GPU dedicated memory and the
system's available RAM every 250 ms. Harness and raw logs: `C:/LLM/logs/ranma-gpu-heartbeat-20260913`.

Revision: both arms ran the same binary, built from this commit (the one that adds this page) on ranma `3d5d9b87c`
(upstream llama.cpp `43f3dda62` + fork README + host-direct MoE). The matrix was taken before two review changes
were folded into the commit: the option now parses its value as a double instead of an int, and the worker
synchronizes its own backend after each event completes. Both were checked functionally on the rebuilt binary; the
timing matrix was not repeated, and neither change touches what the tables measure. The exact commits are kept on
the dated branch `ranma_20260914`, which is never rebased.

### First token after an idle gap (seconds)

R9700 row, 26.4 GiB resident:

| idle gap | off | on, 5 s | off again |
|---:|--:|--:|--:|
| 5 s | 0.19 | 0.18 | 0.17 |
| 15 s | 15.94 | 0.19 | 7.22 |
| 30 s | 3.34 | 0.20 | 2.93 |
| 60 s | 3.26 | 0.18 | 2.98 |

RX 9070 XT row (emulated), 11.4 GiB resident:

| idle gap | off | on, 5 s | off again |
|---:|--:|--:|--:|
| 5 s | 0.21 | 0.19 | 0.18 |
| 15 s | 1.97 | 0.20 | 1.80 |
| 30 s | 1.53 | 0.18 | 1.73 |
| 60 s | 1.52 | 0.21 | 1.51 |

With the heartbeat off, the process' GPU dedicated memory starts falling about 10 s after the last request and
reaches zero within 8-10 s. A request that lands while that eviction is still running pays the most (the 15 s rows:
7-16 s on the larger placement); after the eviction has finished, the first token costs the page-back of the whole
resident set, about 3 s for 26 GiB and 1.5 s for 11 GiB. With the heartbeat on, dedicated memory never moves and
every gap costs the same 0.15-0.21 s as a 5 s gap.

### System RAM while idle

The eviction is not free on the host side either. On the R9700 row, available system RAM fell from 50 GiB to 25 GiB
while the 26 GiB were being evicted and stayed there until the next request (the process' working set grew by the
same amount). With the heartbeat on, available RAM stayed flat at 50 GiB. On a machine that is already close to full
because the experts live in host memory, this is the memory stability aspect: an idle server should not grow by the
size of its VRAM.

### Teardown after Ctrl+C (seconds, R9700 row)

| shutdown sent | off | on, 5 s |
|---|--:|--:|
| right after a request (matrix runs) | 7.98 / 6.97 | 6.31 |
| after 8 s idle | 5.74 | 5.74 |
| after 30 s idle (already evicted) | 11.81 | 5.69 |

The server frees its VRAM within 0.5 s of Ctrl+C and spends the remaining 5-6 s releasing pinned host memory, so
the eviction window can never expire during teardown here and the two arms are equal when the shutdown follows a
request. The difference appears when the shutdown follows an idle gap: the driver pages the evicted 24 GiB back into
VRAM before it frees it (dedicated memory rose from 3 MiB to 23.7 GiB in the first 3 s after Ctrl+C), which doubles
the teardown. The heartbeat avoids that because nothing was evicted. No host RAM spike was seen during teardown in
any arm; the host cost of eviction is paid while idle, as described above.

### Decode speed while busy

The worker sleeps while any slot is busy, so there is no mechanism for a difference, and none was measured: a
separate off / on / off run on the R9700 row with a longer answer (about 90 tokens before the model stopped) decoded
at 16.02 / 16.16 / 16.09 t/s. The 64-token requests in the tables above are too short to compare decode speed and are
not used for that.

## Limitations

- **Only verified on Windows.** The idle eviction described above was observed with the Windows VidMm; whether other
  platforms or drivers evict an idle process' VRAM has not been checked. Where nothing is evicted, the option is
  harmless - one event per interval per device - but buys nothing. It is not compiled out, because nothing in the
  mechanism is platform-specific.
- **Backends without events are skipped.** A backend whose device interface has no `event_new` cannot be pulsed.
  After each event completes, the worker also synchronizes its own backend instance, because Metal keeps one command
  buffer per recorded event until the backend is synchronized. Only HIP has been exercised.
- **mmproj devices are not covered.** The device list comes from the target and draft `llama_model` objects
  (`llama_model_get_device`, a staging API in `src/llama-ext.h`). A multimodal projector loaded on a device outside
  that set is not pulsed.
- **Idle only.** Nothing is submitted while the server is busy; the heartbeat cannot help a request that is already
  slow for another reason.
- **Not a residency guarantee.** It resets the idle timer. Memory pressure from another process can still evict the
  model, and nothing here prevents that.

## Testing notes

- `llama-server --help` must list `--gpu-heartbeat-seconds`; the option is server-only
  (`set_examples({LLAMA_EXAMPLE_SERVER})`).
- With `--gpu-heartbeat-seconds 0` (the default) the worker thread is never created, so the server must be
  bit-identical in behaviour to a build without the feature.
- The interval is parsed from the whole string, so `0.5` is half a second and `1x` is an error, not `1`.
- Per-pulse lines are logged at `DBG` level only (`-lv 1` or `LLAMA_LOG_VERBOSITY`). At default verbosity there are
  two lines: one at start with the interval and the device names, one at release with the pulse counts and the maximum
  wait.
- The interesting race is a pulse overlapping a model-state change. To exercise it, run with a short interval (1 s)
  and drive `/slots/{id}?action=save`, `/slots/{id}?action=restore`, `/slots/{id}?action=erase` and
  `POST /lora-adapters` in a loop between completions; `set_idle(false)` must be reached before each of them.
- The teardown path is reached by `--sleep-idle-seconds` (the server enters the sleeping state, destroys the model,
  and reloads it on the next request), which also checks that the heartbeat re-initializes after a resume.

## Open questions

- Whether the event record alone resets the VidMm timer in every driver version, or whether some drivers require an
  actual kernel dispatch. Only the event path has been used here.
- Whether a longer interval (for example half the eviction window per device count) is safe on a machine where the
  worker thread can be descheduled for seconds.
- Whether the teardown phase also needs a pulse inside the backend's buffer-free path, for a model large enough that
  a single `free` call exceeds the eviction window on its own.
