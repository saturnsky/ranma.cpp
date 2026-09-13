# GPU heartbeat (llama-server)

## What it is

A worker thread in `llama-server` that records one GPU event on every device holding model buffers,
once per interval, for as long as the server has nothing else to submit. It runs in two phases:

- **idle** - every slot is idle, so nothing else touches the GPU;
- **teardown** - the model and its contexts are being freed.

The worker owns no model data. It creates one extra backend per device and one event per device,
records the events once per interval, and completes the previous pulse before recording the next,
so at most one pulse is ever outstanding. A pulse submits no work beyond the event itself.

Two rules keep it out of the way of real work:

- becoming busy, and every state-changing task (slot save, restore and erase, cache clear, LoRA
  change), blocks until no pulse is in flight; after that call returns, the caller owns the model's
  GPU state;
- the first statement of the server's teardown switches the worker to the teardown phase, which
  records and completes a pulse immediately and then keeps pulsing at the interval until the last
  model buffer is freed, at which point the worker is joined.

Implementation: `tools/server/server-gpu-heartbeat.h`, hooked from `tools/server/server-context.cpp`.

## Option

| Option | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--gpu-heartbeat-seconds SECONDS` | `LLAMA_ARG_GPU_HEARTBEAT_SECONDS` | `0` | Interval in seconds between pulses; fractions are accepted (`0.5`). `0` disables the feature. Values outside 0-3600 and strings that are not a number in full are rejected. |

The default is `0`, so behaviour is unchanged unless the option is asked for: the worker thread is
then never created and the server submits nothing while idle, exactly as before. The option is
server-only.

A reasonable value is **5 seconds**: comfortably inside the eviction window observed on the target
platform, with enough margin that a scheduling delay on a loaded machine does not let the window
expire, and far too cheap to matter against a server that is otherwise doing nothing.

```
llama-server -m model.gguf --gpu-heartbeat-seconds 5
```

## Why it exists

On Windows, the video memory manager evicts a process' entire VRAM residency to system RAM after a
short period without a GPU submission from that process. The eviction is per process and is not
triggered by memory pressure, so an idle `llama-server` that holds its weights and KV cache in VRAM
loses its residency during any ordinary gap between turns in a conversation. Three costs follow:

- **Next-request latency.** The resident set is paged back over PCIe ahead of the first kernel, so
  the first decode after an idle gap stalls for as long as that page-back takes; the larger the
  resident set and the slower the link, the longer the stall.
- **Host RAM while idle.** The evicted VRAM lands in system RAM - the process' working set grows by
  the size of its resident set - and stays there until the next request pages it back. On a machine
  whose RAM is already mostly taken by model weights, an idle server that grows by the size of its
  VRAM is a stability problem, not only a latency one.
- **Slower teardown after an idle gap.** Freeing allocations that were evicted first pages them back
  into VRAM, so a shutdown that follows an idle gap takes longer than one that follows a request.

Recording a single event on the device is enough to reset the eviction timer; no real work has to be
submitted. How much each of the three costs amounts to depends on the resident set, the link and the
driver.

## Multiple devices

Residency is managed per adapter, so a pulse on one device does nothing for another. The worker
therefore creates a backend and an event for *every* device that holds model buffers - the devices
the target model was placed on, plus the draft model's when speculative decoding is enabled - and
pulses all of them in one pass per interval: record on each, then synchronize each at the next
pulse. The cost is one event record per device per interval.

A device is skipped with a warning if its backend cannot be initialized or if it does not support
events. If no device is left, the heartbeat stays disabled. Only GPU and integrated-GPU devices are
considered.

## Limits

- **Only the Windows behaviour was observed.** Whether other platforms or drivers evict an idle
  process' VRAM has not been checked. Where nothing is evicted, the option is harmless - one event
  per interval per device - but buys nothing. It is not compiled out, because nothing in the
  mechanism is platform-specific.
- **Backends without events are skipped.** A backend whose device interface cannot create an event
  cannot be pulsed. After each event completes, the worker also synchronizes its own backend,
  because some backends keep one command buffer per recorded event until the backend is
  synchronized.
- **Projector devices are not covered.** The device list comes from the target and draft models, so
  a multimodal projector loaded on a device outside that set is not pulsed.
- **Idle only.** Nothing is submitted while the server is busy; the heartbeat cannot help a request
  that is already slow for another reason.
- **Not a residency guarantee.** It resets the idle timer. Memory pressure from another process can
  still evict the model.

## How to verify

- `llama-server --help` lists `--gpu-heartbeat-seconds`.
- With the default `0` the worker thread is never created, so the server behaves like a build
  without the feature.
- The interval is parsed from the whole string: `0.5` is half a second and `1x` is an error, not
  `1`.
- Per-pulse lines are logged at debug verbosity only. At default verbosity there are two lines: one
  at start with the interval and the device names, one at release with the pulse counts and the
  longest wait.
- The interesting race is a pulse overlapping a model-state change. To exercise it, run with a
  one-second interval and drive the slot save, restore and erase endpoints and a LoRA change in a
  loop between completions.
- The teardown phase is reached through the idle-sleep option, which destroys the model and reloads
  it on the next request; that also checks that the heartbeat re-initializes after a resume.
- The residency effect itself is visible by watching the process' dedicated GPU memory and the
  system's available RAM across an idle gap, with the option off and on.
