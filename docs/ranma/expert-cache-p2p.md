# Peer-to-peer reads from the SSD into VRAM (Windows, engine and tests)

## What it is

A read engine that moves byte ranges of GGUF files straight into VRAM, without passing through host memory:

    unbuffered IoRing read -> staging buffer on a D3D12 GPU_UPLOAD heap (VRAM behind the resizable BAR)
      -> D3D12 copy queue -> a D3D12 DEFAULT buffer created SHARED and imported into HIP

The engine (`ggml/src/ggml-cuda/expert-p2p.h`, `expert-p2p-win32.cpp`) and its tests are in this release. **The expert
cache does not use it yet**: the SSD tier ([expert-cache-l2.md](expert-cache-l2.md)) still reads into its host ring
slots, and no option or environment switch turns the engine on in `llama-server`. The page records what the engine
does and how to check it on a machine.

## Why it exists

The host tier's reads cost host memory bandwidth twice (the SSD's DMA into the ring, then the GPU's read of the ring
over the link), and the GPU's host reads share that bandwidth with the SSD's DMA. A peer-to-peer read uses almost no
host memory bandwidth (model-free measurement on the target with `test-expert-p2p-bench`: 0.01 GB of CPU memory
bandwidth per GB read, against 0.84 for a read into host memory), and a slice in VRAM is read by the kernels at VRAM
speed.

## How it works

- A D3D12 device on the adapter whose LUID is the HIP device's; a copy queue with a fence; staging buffers of at most
  1 GiB (a larger mapped allocation failed on the target) cut into chunk slots and registered once with one IoRing;
  the GGUF files registered with the same ring.
- A read is split into sector-aligned chunks; each chunk is one IoRing read into a free staging slot and one copy of
  its payload into the destination (plus a copy from a zeroed buffer for a requested tail). Every sweep of completions
  becomes one command list and one fence signal; a staging slot is reused once the fence has passed its copy. The
  caller learns about completed reads in submit order after the fence passed them.
- The destination is a **region**: an imported VRAM buffer of at most 3 GiB (an imported buffer of 4 GiB faulted in the
  first kernel on the target). Kernels read a region like any HIP allocation.
- D3D12, DXGI and the IoRing API are resolved at run time; no library is linked.
- The completion is waited for on the CPU (a fence imported into HIP as an external semaphore would carry a fixed
  wait value into a captured graph).

## Settings

The engine reads these environment variables when it is created (`p2p_config_from_env`); an invalid value is ignored
with a note.

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_P2P_STAGING_MIB` | 128 | GPU_UPLOAD staging (16 to 4096). |
| `RANMA_EXPERT_P2P_CHUNK_KIB` | 1024 | Read and copy piece size (64 to 65536, rounded up to a multiple of 64 KiB, at most the staging). |
| `RANMA_EXPERT_P2P_QD` | 32 | Reads in flight (1 to 256). |
| `RANMA_EXPERT_P2P_WAIT_MS` | 10000 | A run that takes longer fails, with the engine state in the message (100 to 600000). |
| `RANMA_EXPERT_P2P_PREGATE` | 0 | 1 queues a run's copies before its reads behind a fence the CPU signals (measured slower). |

## What it costs

- VRAM: the staging, a zeroed source buffer and the regions. VidMm counts an imported buffer twice (the D3D12
  allocation and its HIP mapping; measured, the physical memory is used once), and the process commit charge grows by
  the imported size.
- Per run, after the last read: about 0.1 ms until the copies are signalled (measured with 1 MiB chunks, one to
  eighteen slices per run, idle).

## Limits

- Windows, HIP, a GPU with GPU upload heaps (resizable BAR). A failure to create the engine is reported with a reason.
- A failing read at run time fails the run with the engine state in the message.

## How to verify it

Build the targets (they are not part of the default build; Windows HIP only) and run them with `HIP_VISIBLE_DEVICES`
selecting the discrete GPU. Both read raw byte ranges of GGUF files (each larger than 3 GiB for the full test) and load
no model.

- `test-expert-p2p-gpu --files a.gguf,b.gguf [--seed N] [--quick]` reads real ranges through the engine and compares
  them byte for byte with host reads (one large unaligned read, many small reads with tails, requests larger than the
  staging with small chunks), checks that a region rewritten through the engine is read again by a kernel without stale
  device cache lines, checks the refusals, and measures the latency and the bridge of both copy modes.
- `test-expert-p2p-bench --files a.gguf,b.gguf [--total-gib N] [--rounds N] [--only limits|ring|bw|cont]` measures the
  limits of the imported regions, bandwidth against the host path (an unbuffered read into pinned host memory, then
  `hipMemcpyAsync`) and the contention with a running kernel.
