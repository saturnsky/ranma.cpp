# Expert cache: peer-to-peer reads from the SSD into VRAM (Windows)

## What it is

An optional path of the SSD tier (`expert-cache-l2.md`) that reads expert slices from the GGUF files
straight into VRAM, without passing through host memory:

    unbuffered IoRing read -> staging buffer on a D3D12 GPU_UPLOAD heap (VRAM behind the resizable BAR)
      -> D3D12 copy queue -> a D3D12 DEFAULT buffer created SHARED and imported into HIP

The imported buffers form a **VRAM ring** of the tier. Its slots have one pitch per kind (the largest
slice of that kind over all size classes plus its arena tail), so any layer's expert fits any slot.
The kernels read a slice in a VRAM ring slot through the tier's address table, exactly as they read a
host ring slot; nothing changes in the kernels. Everything is off unless `RANMA_EXPERT_P2P=1`.

Four uses, each with its own switch:

- **Prompt ubatches** (more rows than the decode bound): the file experts a layer needs are read into
  the VRAM ring, and the MMQ kernels read them from VRAM instead of over the link.
- **Generation ubatches** (optional): the same for decode misses.
- **Installs**: a promotion from the file to VRAM is read into the VRAM ring and copied device to
  device into its arena slot; it takes no host ring slot and no host memory pass.
- **Pins**: during generation the ring holds the best experts outside the arena (by the plan's
  selection counts, file experts weighted higher), copied in while the worker is idle; a prompt
  ubatch releases them, so the ring serves as the prompt's stream buffer and, in generation, as
  VRAM capacity.

## Why it exists

The host tier's reads cost host memory bandwidth twice (the SSD's DMA into the ring, then the GPU's
read of the ring over the link), and the GPU's host reads share that bandwidth with the SSD's DMA. A
peer-to-peer read uses almost no host memory bandwidth (model-free measurement on the target: 0.01 GB
of CPU memory bandwidth per GB read, against 0.84 for a read into host memory), and a slice in VRAM is
read by the kernels at VRAM speed.

## How it works

- **Engine** (`expert-p2p.h`, `expert-p2p-win32.cpp`): a D3D12 device on the adapter whose LUID is
  the HIP device's; a copy queue with a fence; staging buffers of at most 1 GiB (a larger mapped
  allocation failed on the target) cut into chunk slots and registered once with one IoRing; the GGUF
  files registered with the same ring. A read is split into sector-aligned chunks; each chunk is one
  IoRing read into a free staging slot and one copy of its payload into the destination (plus a copy
  from a zeroed buffer for the arena tail). Every sweep of completions becomes one command list and
  one fence signal; a staging slot is reused once the fence has passed its copy. The caller learns
  about completed reads in submit order after the fence passed them. D3D12, DXGI and the IoRing API
  are resolved at run time; no library is linked.
- **Demand service** (worker thread): for each file expert of a generation, a VRAM ring hit is served
  there; an expert a host ring slot already holds stays on the host path (a host hit costs no SSD
  read); otherwise, in a phase with peer-to-peer reads, the expert gets a reusable VRAM slot and is
  read peer to peer, or, when no VRAM slot is reusable, takes the host ring as before. The
  peer-to-peer reads are issued first, the host reads run, then the peer-to-peer reads complete and
  the layer is published. With the staged prompt service a kind is released once both its host reads
  and its peer-to-peer reads are in place.
- **Completion**: the worker waits for the copy fence on the CPU and then answers the layer's mailbox,
  as it does after a host read; the GPU side is unchanged, so captured graphs need nothing new. (A
  fence imported into HIP as an external semaphore would carry a fixed wait value into a captured
  graph.)
- **Leases**: VRAM ring slots are leased by generation like host ring slots; a pin is a slot whose
  lease does not end until the pin is released.
- **Budget**: the staging and the VRAM ring are subtracted from the VRAM (L1) budget before the
  placement, so a run compares at equal total VRAM. With `RANMA_EXPERT_P2P_HOST_RING=release` each
  host storage class ring shrinks by the VRAM ring's slot count (to at least 4 slots) and a layer may
  keep that many more experts in the file; the freed host bytes become host residents.

## Options

| Environment switch | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_P2P` | 0 | 1 turns the path on (finite host tier only). 0: no engine, no ring, the host path exactly. |
| `RANMA_EXPERT_P2P_PROMPT` | 1 | File experts of prompt ubatches are read into the VRAM ring. |
| `RANMA_EXPERT_P2P_DECODE` | 0 | The same for generation ubatches. |
| `RANMA_EXPERT_P2P_INSTALL` | 1 | Relabel installs read promotions to VRAM through the VRAM ring (device copy into the arena). |
| `RANMA_EXPERT_P2P_PIN` | 1 | Pins during generation. |
| `RANMA_EXPERT_P2P_PIN_FILE_WEIGHT` | 4 | A file expert's selection count against a host resident's when pins are ranked. |
| `RANMA_EXPERT_P2P_RING_MIB` | 2048 | VRAM ring size. |
| `RANMA_EXPERT_P2P_DECODE_RING_MIB` | 512 | With generation reads on: the part of the ring pins leave to them. |
| `RANMA_EXPERT_P2P_HOST_RING` | keep | `keep` or `release` (see Budget). Not used with install-time ring sizing. |
| `RANMA_EXPERT_P2P_STAGING_MIB` | 128 | GPU_UPLOAD staging. |
| `RANMA_EXPERT_P2P_PIN_KEEP` | 0 | The best this many pins stay through prompt ubatches. |
| `RANMA_EXPERT_P2P_WAIT_MS` | 10000 | A peer-to-peer run that takes longer fails, with the engine state in the message. |
| `RANMA_EXPERT_P2P_CHUNK_KIB` | 1024 | Read and copy piece size. |
| `RANMA_EXPERT_P2P_QD` | 32 | Reads in flight. |
| `RANMA_EXPERT_P2P_PREGATE` | 0 | 1 queues a run's copies before its reads behind a fence the CPU signals (measured slower). |

The load line `expert cache: P2P (NVMe -> VRAM) on: ...` names the adapter, staging, ring, the parts
that are on and the VRAM taken from the L1 budget; a failure logs `expert cache: P2P off, the SSD tier
keeps its host path: <reason>`. Each bank commit logs `expert cache: P2P bank ... since the last
commit: ...` with the demand reads, VRAM ring hits, experts that found no VRAM slot, engine time,
install reads and pins. `RANMA_EXPERT_TRACE=8` adds `p2p_*`, `install_p2p_*` and `pin*` fields to
the `l2` metrics line.

## What it costs

- VRAM: staging + ring, out of the L1 budget. VidMm counts an imported buffer twice (the D3D12
  allocation and its HIP mapping; measured, the physical memory is used once), and the process commit
  charge grows by the imported size.
- Per peer-to-peer run, after the last read: about 0.1 ms until the copies are signalled (measured
  with 1 MiB chunks, one to eighteen slices per run, idle).
- The worker thread spins on the read completions and the copy fence while a run is in flight.
- Pins: after each prompt, copies of up to the ring's pin slots (from host memory, or peer to peer for
  file experts) while the worker is idle; a layer that is published during a copy waits for it.

## Limits and fallbacks

- **Windows, HIP, a GPU with GPU upload heaps** (resizable BAR). Any failure at load (engine, ring,
  opening a file) leaves the path off with one log line. A failing read at run time ends the process,
  as a failing host read does.
- An imported buffer of 4 GiB faulted in the first kernel on the target; the ring is made of pieces of
  at most 3 GiB.
- One pitch for all layers: small experts waste part of their slot (DS4F: about 40 % for the smallest
  class).
- The mover path (`RANMA_EXPERT_L2_INSTALL_RELABEL=0` or the `ring` layout) installs without it.
- The recent pool (`expert-cache-recent.md`) fills from whatever address the tier publishes; a slice
  in the VRAM ring is copied device to device. There is no separate peer-to-peer pool fill.
- A staged prompt layer that has host reads and peer-to-peer reads releases its first kind only after
  all its host reads.

## How to verify it

- `test-expert-p2p-gpu` (standalone, Windows HIP): reads real GGUF ranges through the engine and
  compares them byte for byte with host reads (one large unaligned read, many small reads with tails,
  requests larger than the staging with small chunks), checks the tier's consumption pattern (a slot
  read by a kernel, rewritten through the engine, read again by a kernel enqueued behind a host-mailbox
  wait: no stale device cache lines), the refusals, and measures the latency and the bridge of both
  copy modes. Usage: `test-expert-p2p-gpu [--files a.gguf,b.gguf] [--seed N] [--quick]`, with
  `HIP_VISIBLE_DEVICES` selecting the discrete GPU.
- `test-expert-l2`: the ring ledger's partial service (misses without a reusable slot handed back),
  lease expiry, and the relabel install with promotions read elsewhere.
- `RANMA_EXPERT_L2_VERIFY=1` copies every slice read into the VRAM ring back and compares it with the
  file (and its arena tail with zeros), including pins and install reads.
