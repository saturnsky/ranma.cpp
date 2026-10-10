#include "llama-prefetch.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

// Resolved at run time, exactly as llama_mmap does for the load-time prefetch, so that the link
// line does not change and an older host simply gets no prefetching.
typedef BOOL (WINAPI * llama_prefetch_fn)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);

static llama_prefetch_fn llama_prefetch_resolve() {
    static const llama_prefetch_fn fn = reinterpret_cast<llama_prefetch_fn>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory")));
    return fn;
}

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    if (ranges == nullptr || n_ranges == 0) {
        return;
    }

    const llama_prefetch_fn prefetch = llama_prefetch_resolve();
    if (prefetch == nullptr) {
        return;
    }

    std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
    entries.reserve(n_ranges);
    for (size_t i = 0; i < n_ranges; ++i) {
        if (ranges[i].addr == nullptr || ranges[i].size == 0) {
            continue;
        }
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = const_cast<void *>(ranges[i].addr);
        entry.NumberOfBytes  = (SIZE_T) ranges[i].size;
        entries.push_back(entry);
    }
    if (entries.empty()) {
        return;
    }

    // The return value is deliberately ignored: a refused hint is not an error for the caller.
    prefetch(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0);
}

#elif defined(__unix__) || defined(__APPLE__)

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    if (ranges == nullptr || n_ranges == 0) {
        return;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return;
    }
    const size_t mask = (size_t) page - 1;

    for (size_t i = 0; i < n_ranges; ++i) {
        if (ranges[i].addr == nullptr || ranges[i].size == 0) {
            continue;
        }
        // madvise wants a page-aligned start; round down and extend the length by the same amount.
        char * const  base    = (char *) const_cast<void *>(ranges[i].addr);
        const size_t  skew    = (size_t) (uintptr_t) base & mask;
        // Errors are ignored: MADV_WILLNEED is advisory and the caller has no fallback.
        (void) madvise(base - skew, ranges[i].size + skew, MADV_WILLNEED);
    }
}

#else

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    (void) ranges;
    (void) n_ranges;
}

#endif

// Batched prefetch of the rows of one ubatch gather from a lazily mapped table (see llama-prefetch.h).
//
// A per-layer embedding or engram table is a multi-GiB table that the loader maps lazily, so every gathered row
// is a demand page fault. A 512-token prompt ubatch gathers thousands of ~110 byte rows: that is
// thousands of serialized 4 KiB reads when the pages are not in the page cache. One call with the
// merged ranges lets the OS read them together. It is a hint; the gather reads the same bytes.
//
// Thresholds are the measured ones, see docs/ranma/ple-prefetch.md.
static const size_t PLE_PREFETCH_MIN_ROWS_PREFILL = 256;
static const size_t PLE_PREFETCH_MIN_ROWS_DECODE  = 16;

void llama_prefetch_table_rows(const ggml_tensor * table, const std::vector<int32_t> & rows,
                               enum llama_ple_prefetch mode) {
    if (mode == LLAMA_PLE_PREFETCH_OFF || table == nullptr || table->data == nullptr) {
        return;
    }

    const size_t min_rows = mode == LLAMA_PLE_PREFETCH_ALWAYS
        ? PLE_PREFETCH_MIN_ROWS_DECODE
        : PLE_PREFETCH_MIN_ROWS_PREFILL;
    if (rows.size() < min_rows) {
        return;
    }

    // A table that is not in host memory is not page-faulted by the CPU; prefetching a device
    // pointer would be meaningless. This is also what keeps the option inert for other placements.
    if (table->buffer == nullptr || !ggml_backend_buffer_is_host(table->buffer)) {
        return;
    }

    std::vector<int32_t> sorted = rows;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    // Merge consecutive rows so that neighbouring tokens become one range rather than many.
    std::vector<llama_prefetch_range> ranges;
    for (size_t first = 0; first < sorted.size();) {
        size_t last = first + 1;
        while (last < sorted.size() && sorted[last] == sorted[last - 1] + 1) {
            ++last;
        }
        llama_prefetch_range range;
        range.addr = (const char *) table->data + (size_t) sorted[first] * table->nb[1];
        range.size = (size_t) (sorted[last - 1] - sorted[first] + 1) * table->nb[1];
        ranges.push_back(range);
        first = last;
    }

    llama_prefetch_ranges(ranges.data(), ranges.size());
}
