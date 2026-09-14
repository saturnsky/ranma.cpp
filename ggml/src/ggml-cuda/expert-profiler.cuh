#pragma once

// Router selection profiler: one uint32 histogram per bank in VRAM, filled by a small kernel that
// runs right after the top-k kernel on the compute stream. Nothing is read back on the hot path;
// the caller reads a bank when it commits it. The device-side selection tells the kernel which
// rows of the current batch to count and into which bank, so the same captured graph can profile
// a different row or bank on every replay.

#include "common.cuh"
#include "expert-geometry.h"

#include <cstdint>
#include <vector>

namespace ggml_cuda_expert {

struct profile_selection {
    int32_t  row_begin; // rows [row_begin, row_end) of the batch are counted
    int32_t  row_end;
    uint32_t bank;      // histogram index
    uint32_t reserved;
};

class profiler {
public:
    profiler(const geometry & geo, uint32_t n_banks);
    ~profiler();

    profiler(const profiler &) = delete;
    profiler & operator=(const profiler &) = delete;

    // Allocates and zeroes every bank and the selection. Call before any graph capture.
    bool allocate(int device);
    bool allocated() const { return counts_ != nullptr; }

    uint32_t n_banks()  const { return n_banks_; }
    size_t   n_counts() const { return n_counts_; }
    size_t   device_bytes() const;

    // Kernel-facing pointers, fixed after allocate().
    uint32_t *                counts_base() const { return counts_; }
    const profile_selection * selection()   const { return selection_; }

    // Publishes a new selection on `stream` (ordered before the next kernels on it). No launch when
    // the selection is unchanged.
    bool select(int32_t row_begin, int32_t row_end, uint32_t bank, cudaStream_t stream);

    // Synchronize the device, copy a bank to the host and optionally zero it.
    bool read_bank(uint32_t bank, std::vector<uint64_t> & out, bool zero_after);
    bool zero_bank(uint32_t bank);

private:
    const uint32_t   n_banks_;
    const size_t     n_counts_;
    int              device_ = -1;
    uint32_t *          counts_    = nullptr;
    profile_selection * selection_ = nullptr;
    profile_selection   last_selection_ = { -1, -1, 0xFFFFFFFFu, 0 };
};

// Adds the top-k ids of the selected rows to counts[bank*bank_stride + id]. ids is [n_used, n_rows]
// with a row stride of `row_stride` int32 elements.
void launch_profile_ids(
        const int32_t * ids, int n_rows, int row_stride, int n_used, int n_experts,
        uint32_t * counts, size_t bank_stride, uint32_t n_banks, const profile_selection * selection, cudaStream_t stream);

} // namespace ggml_cuda_expert
