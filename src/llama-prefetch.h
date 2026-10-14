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

// LLAMA_PREFETCH_TIMING=1 (diagnostic, default off): the callers of the table prefetch below time its steps, count the
// process page faults around it and describe, per piece of the table, how its row ranges fall onto OS pages, and print one
// "table_prefetch {json}" log line per gather. With the switch off nothing is measured and nothing is printed; the
// prefetch itself is the same either way.
bool llama_prefetch_timing_enabled();

// The page faults of the whole process so far (all threads), -1 where the OS does not tell.
int64_t llama_process_page_faults();

// One piece of a table in one prefetch: the rows asked for and how the merged row ranges map onto OS pages.
struct llama_prefetch_piece_stats {
    int64_t rows         = 0;   // row numbers asked for, repeats included
    int64_t unique_rows  = 0;
    int64_t row_ranges   = 0;   // ranges of consecutive rows: what is handed to the OS now
    int64_t row_bytes    = 0;
    int64_t page_bytes   = 0;   // the same ranges, each rounded out to whole pages (pages shared by two ranges count twice)
    int64_t page_ranges  = 0;   // the rounded ranges after merging the ones that share a page
    int64_t page_runs    = 0;   // ... and also the ones that touch (consecutive pages)
    int64_t unique_pages = 0;
};

// The steps of one gather's prefetch (us; -1 = not measured) and the faults around the OS calls.
struct llama_prefetch_stats {
    int64_t t_hash_us    = -1;  // caller: the row numbers (n-gram hash)
    int64_t t_split_us   = 0;   // copy of the row numbers and their distribution to the pieces
    int64_t t_sort_us    = 0;   // sort and deduplicate
    int64_t t_ranges_us  = 0;   // merge into ranges and build the OS range array
    int64_t t_os_us      = 0;   // the OS call itself
    int64_t t_pages_us   = 0;   // this diagnostic's page arithmetic (not part of the prefetch)
    int64_t t_input_us   = -1;  // caller: per-piece row numbers and mask, and their copy into the input tensors
    int64_t faults_before = -1, faults_after = -1;   // process page faults right before and after the prefetch
    int64_t page_size    = 0;
    bool    skipped      = true; // nothing was handed to the OS (mode off, too few rows, table not in host memory)
    std::vector<llama_prefetch_piece_stats> pieces;
};

// Prints the "table_prefetch" line of one gather (what: the table, layer: its layer or -1).
void llama_prefetch_stats_log(const char * what, int layer, int64_t n_tokens, const llama_prefetch_stats & st);

// Batched prefetch of the rows of a lazily mapped table that a GET_ROWS of this ubatch is about to read
// (per-layer embeddings, DeepSeek V4.1 engram tables). The rows are sorted, deduplicated and merged into
// ranges; the thresholds are the measured ones of docs/ranma/ple-prefetch.md. A hint: it does nothing for
// mode off, for a gather below the threshold, or for a table that is not in host memory.
// With stats (LLAMA_PREFETCH_TIMING) the steps are timed and the pages of each piece described.
void llama_prefetch_table_rows(const struct ggml_tensor * table, const std::vector<int32_t> & rows,
                               enum llama_ple_prefetch mode, llama_prefetch_stats * stats = nullptr);

// The same for a table stored as pieces of part_rows consecutive rows (a table of one piece is the call above): rows are the
// GLOBAL row numbers of the gather, the threshold counts all of them, and each piece is hinted with its own rows.
void llama_prefetch_table_pieces_rows(const std::vector<struct ggml_tensor *> & parts, int64_t part_rows,
                                      const std::vector<int32_t> & rows, enum llama_ple_prefetch mode,
                                      llama_prefetch_stats * stats = nullptr);
