#include "llama-prefetch.h"

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
