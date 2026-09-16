#pragma once

#include <cstddef>

// Best-effort hint that a set of virtual ranges is about to be read.
//
// A lazily mapped tensor is faulted in one page at a time by the thread that touches it, which on
// Windows is a queue-depth-1 random read pattern. Handing the whole set of ranges to the operating
// system in one call lets it issue a few large concurrent reads instead. This is a hint only: it
// never changes what the process reads, it cannot fail in a way the caller must handle, and it does
// nothing at all on platforms without such a call.
//
// The Win32 declarations stay in the .cpp so that windows.h does not leak into a header.

struct llama_prefetch_range {
    const void * addr;
    size_t       size;
};

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges);
