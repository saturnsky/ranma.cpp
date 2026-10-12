#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;

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

// Batched prefetch of the rows of a lazily mapped table that a GET_ROWS of this ubatch is about to read
// (per-layer embeddings, DeepSeek V4.1 engram tables). The rows are sorted, deduplicated and merged into
// ranges; the thresholds are the measured ones of docs/ranma/ple-prefetch.md. A hint: it does nothing for
// mode off, for a gather below the threshold, or for a table that is not in host memory.
void llama_prefetch_table_rows(const struct ggml_tensor * table, const std::vector<int32_t> & rows,
                               enum llama_ple_prefetch mode);

// The same for a table stored as pieces of part_rows consecutive rows (a table of one piece is the call above): rows are the
// GLOBAL row numbers of the gather, the threshold counts all of them, and each piece is hinted with its own rows.
void llama_prefetch_table_pieces_rows(const std::vector<struct ggml_tensor *> & parts, int64_t part_rows,
                                      const std::vector<int32_t> & rows, enum llama_ple_prefetch mode);
