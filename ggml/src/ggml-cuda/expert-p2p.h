#pragma once

// Peer-to-peer reads from the GGUF files into VRAM for the expert cache (Windows only).
//
// HIP device memory has no CPU address, so an unbuffered file read cannot target it. A D3D12 buffer
// on a GPU_UPLOAD heap is VRAM that the CPU maps through the resizable BAR, and the NVMe controller
// can write into it directly; it cannot be shared with HIP, though. The chain is therefore
//
//   IoRing read (unbuffered, into a staging buffer on a GPU_UPLOAD heap, registered once)
//   -> D3D12 copy queue: CopyBufferRegion into a committed DEFAULT buffer created SHARED
//   -> that buffer imported into HIP (hipImportExternalMemory, D3D12 resource) = a region
//
// and the bytes never pass through host memory. Reads are split into chunks and pipelined: while
// the copy of one chunk runs, the next chunks are being read. A fence on the copy queue orders the
// copies; the caller learns about completed reads from `finish` (in submit order), after the copy
// queue has signalled them.
//
// Measured on the target machine (model-free probes): 11.7-12.2 GB/s from one thread at 2-15 ms
// CPU per GB with the staging buffer registered; the copy runs at 38-80 GB/s. A mapped allocation
// over 1 GiB failed, so the staging is made of buffers of at most 1 GiB. An imported region of
// 4 GiB faulted in the first kernel, 3 GiB worked, so a region stays below 4 GiB. VidMm counts an
// imported region twice (the D3D12 allocation and the HIP mapping of it), and the process commit
// charge grows by its size; the physical VRAM is used once.
//
// Everything is created by `create`, which fails with a reason (and nothing allocated) on another
// platform or when any step is unsupported, so the caller keeps its old path.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace ggml_cuda_expert {

struct p2p_config {
    size_t  staging_bytes = size_t(128) << 20;   // GPU_UPLOAD staging, buffers of at most 1 GiB
    size_t  chunk_bytes   = size_t(1) << 20;     // one IoRing read and one copy; a multiple of 64 KiB
    int     queue_depth   = 32;                  // reads in flight
    int64_t wait_ms       = 10000;               // a run that takes longer fails (with the engine's state)
    // A run whose reads all fit in flight at once records its copies before the reads complete and
    // queues them behind a wait on a second fence, which the CPU signals after the last completion:
    // the copies start without an ExecuteCommandLists after the reads.
    bool    pregate       = false;
};

// One slice: `bytes` at file offset `offset` (any alignment) land at offset `dst` of a region, and
// the `zero_tail` bytes after them are cleared.
struct p2p_read {
    int      file      = -1;
    uint64_t offset    = 0;
    size_t   bytes     = 0;
    int      region    = -1;
    size_t   dst       = 0;
    size_t   zero_tail = 0;
};

struct p2p_stats {
    uint64_t runs = 0, reads = 0, bytes = 0, pieces = 0, lists = 0, zero_bytes = 0;
    double   run_ms = 0.0;
    double   io_wait_ms = 0.0;      // waiting for a read completion with nothing else to do
    double   fence_wait_ms = 0.0;   // every read done, waiting for the copy queue
    double   tail_ms = 0.0;         // last read completion seen -> the run reported complete (the bridge)
    uint64_t tail_runs = 0, gated_runs = 0;
};

class p2p_engine {
public:
    // An imported region of 4 GiB faulted on the target; this is the largest one handed out.
    static constexpr size_t max_region_bytes = size_t(3) << 30;

    virtual ~p2p_engine() = default;

    // D3D12 device on the adapter of HIP device `device` (matched by LUID), the copy queue and its
    // fence, the staging buffers and the IoRing with the staging registered. Null and a reason on
    // failure; nothing stays allocated then.
    static std::unique_ptr<p2p_engine> create(int device, const p2p_config & cfg, std::string & reason);

    // Opens `path` for unbuffered reading (idempotent by path) and returns its index. The files are
    // registered with the ring at the next run; call before a run, never during one.
    virtual int add_file(const std::string & path, std::string & reason) = 0;

    // A committed DEFAULT + SHARED buffer imported into HIP; returns its index and the HIP pointer.
    // `bytes` <= max_region_bytes.
    virtual int alloc_region(size_t bytes, void ** device, std::string & reason) = 0;
    virtual void * region_device(int region) const = 0;

    // Starts a run: the first reads go out, `begin` returns without waiting. The destinations must
    // not be read or written by anyone else until `finish` has reported them.
    virtual bool begin(const p2p_read * reads, size_t n, std::string & reason) = 0;
    // Completes the run started by `begin`. `progress(k)`, when given, is called whenever the first k
    // reads of the run (in submit order) have been copied and the copy queue has signalled them;
    // returning false fails the run. A read reported complete is visible to HIP kernels launched (or
    // released from a wait) afterwards.
    virtual bool finish(std::string & reason, const std::function<bool(size_t)> & progress = nullptr) = 0;
    bool run(const p2p_read * reads, size_t n, std::string & reason, const std::function<bool(size_t)> & progress = nullptr) {
        return begin(reads, n, reason) && finish(reason, progress);
    }

    // Copies region bytes back into host memory through the copy queue and a readback buffer
    // (verification). Allowed inside a `progress` callback of a run; no HIP call is made, so a HIP
    // stream that waits on the caller cannot block it.
    virtual bool read_back(int region, size_t offset, void * dst, size_t bytes, std::string & reason) = 0;
    // Several spans at once: one command list and one fence per readback buffer's worth.
    struct span { int region; size_t offset; void * dst; size_t bytes; };
    virtual bool read_back_many(const span * spans, size_t n, std::string & reason) = 0;
    // Writes host memory into a region through the staging buffers and the copy queue, then clears
    // `zero_tail` bytes after it; returns once the copy queue has signalled it. Not during a run.
    virtual bool write(int region, size_t offset, const void * src, size_t bytes, size_t zero_tail, std::string & reason) = 0;

    virtual const p2p_stats & stats() const = 0;
    // adapter, staging, chunk, queue depth
    virtual std::string description() const = 0;
    virtual size_t staging_bytes() const = 0;   // VRAM of the staging buffers and the zero source
    virtual size_t region_bytes() const = 0;    // VRAM of the regions
};

// The configuration from the environment (RANMA_EXPERT_P2P_STAGING_MIB, _CHUNK_KIB, _QD); `notes`
// collects what was ignored.
p2p_config p2p_config_from_env(std::string & notes);

} // namespace ggml_cuda_expert
