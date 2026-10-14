#include "llama-prefetch.h"

#include "llama-impl.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

bool llama_prefetch_timing_enabled() {
    static const bool on = [] {
        const char * env = getenv("LLAMA_PREFETCH_TIMING");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return on;
}

#if defined(_WIN32)

// Resolved at run time, exactly as llama_mmap does for the load-time prefetch, so that the link
// line does not change and an older host simply gets no prefetching.
typedef BOOL (WINAPI * llama_prefetch_fn)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);

static llama_prefetch_fn llama_prefetch_resolve() {
    static const llama_prefetch_fn fn = reinterpret_cast<llama_prefetch_fn>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory")));
    return fn;
}

// The OS range array and the call; with stats the two are timed apart.
static void prefetch_ranges_impl(const llama_prefetch_range * ranges, size_t n_ranges, llama_prefetch_stats * stats) {
    if (ranges == nullptr || n_ranges == 0) {
        return;
    }

    const llama_prefetch_fn prefetch = llama_prefetch_resolve();
    if (prefetch == nullptr) {
        return;
    }

    const int64_t t0 = stats ? ggml_time_us() : 0;
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

    const int64_t t1 = stats ? ggml_time_us() : 0;
    // The return value is deliberately ignored: a refused hint is not an error for the caller.
    prefetch(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0);
    if (stats) {
        stats->t_ranges_us += t1 - t0;
        stats->t_os_us     += ggml_time_us() - t1;
        stats->skipped      = false;
    }
}

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    prefetch_ranges_impl(ranges, n_ranges, nullptr);
}

// PROCESS_MEMORY_COUNTERS of psapi.h, resolved at run time like the prefetch call so that the link line does not change.
struct llama_process_counters {
    DWORD  cb;
    DWORD  PageFaultCount;
    SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
           QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage;
};
typedef BOOL (WINAPI * llama_counters_fn)(HANDLE, llama_process_counters *, DWORD);

int64_t llama_process_page_faults() {
    static const llama_counters_fn fn = reinterpret_cast<llama_counters_fn>(
        reinterpret_cast<void *>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32GetProcessMemoryInfo")));
    llama_process_counters pmc = {};
    pmc.cb = sizeof(pmc);
    if (fn == nullptr || !fn(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return -1;
    }
    return (int64_t) pmc.PageFaultCount;   // 32 bits, wraps: callers take differences modulo 2^32
}

static int64_t llama_page_size() {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int64_t) si.dwPageSize;
}

#elif defined(__unix__) || defined(__APPLE__)

static void prefetch_ranges_impl(const llama_prefetch_range * ranges, size_t n_ranges, llama_prefetch_stats * stats) {
    if (ranges == nullptr || n_ranges == 0) {
        return;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return;
    }
    const size_t mask = (size_t) page - 1;

    const int64_t t0 = stats ? ggml_time_us() : 0;
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
    if (stats) {
        stats->t_os_us += ggml_time_us() - t0;
        stats->skipped  = false;
    }
}

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    prefetch_ranges_impl(ranges, n_ranges, nullptr);
}

int64_t llama_process_page_faults() {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return -1;
    }
    return (int64_t) ru.ru_minflt + (int64_t) ru.ru_majflt;
}

static int64_t llama_page_size() {
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (int64_t) page : 0;
}

#else

static void prefetch_ranges_impl(const llama_prefetch_range * ranges, size_t n_ranges, llama_prefetch_stats * stats) {
    (void) ranges;
    (void) n_ranges;
    (void) stats;
}

void llama_prefetch_ranges(const llama_prefetch_range * ranges, size_t n_ranges) {
    (void) ranges;
    (void) n_ranges;
}

int64_t llama_process_page_faults() {
    return -1;
}

static int64_t llama_page_size() {
    return 0;
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

static bool prefetch_enough_rows(size_t n_rows, enum llama_ple_prefetch mode) {
    if (mode == LLAMA_PLE_PREFETCH_OFF) {
        return false;
    }
    const size_t min_rows = mode == LLAMA_PLE_PREFETCH_ALWAYS
        ? PLE_PREFETCH_MIN_ROWS_DECODE
        : PLE_PREFETCH_MIN_ROWS_PREFILL;
    return n_rows >= min_rows;
}

// How the merged row ranges of one piece fall onto OS pages (LLAMA_PREFETCH_TIMING). The ranges ascend and do not overlap,
// so a page shared by two ranges is always shared with the previous one.
static void prefetch_page_stats(const std::vector<llama_prefetch_range> & ranges, int64_t page,
                                llama_prefetch_piece_stats & ps) {
    ps.row_ranges = (int64_t) ranges.size();
    int64_t last = -2;   // highest page of the ranges so far
    for (const llama_prefetch_range & r : ranges) {
        ps.row_bytes += (int64_t) r.size;
        if (page <= 0) {
            continue;
        }
        const int64_t p0 = (int64_t) ((uintptr_t) r.addr / (uintptr_t) page);
        const int64_t p1 = (int64_t) (((uintptr_t) r.addr + r.size - 1) / (uintptr_t) page);
        ps.page_bytes   += (p1 - p0 + 1)*page;
        ps.page_ranges  += p0 <= last     ? 0 : 1;
        ps.page_runs    += p0 <= last + 1 ? 0 : 1;
        ps.unique_pages += p1 - std::max(p0, last + 1) + 1;
        last = std::max(last, p1);
    }
}

// the rows (any order, repeats allowed) of one table or one piece of it
static void prefetch_table_rows_unchecked(const ggml_tensor * table, std::vector<int32_t> sorted, llama_prefetch_stats * stats) {
    if (table == nullptr || table->data == nullptr) {
        return;
    }

    // A table that is not in host memory is not page-faulted by the CPU; prefetching a device
    // pointer would be meaningless. This is also what keeps the option inert for other placements.
    if (table->buffer == nullptr || !ggml_backend_buffer_is_host(table->buffer)) {
        return;
    }

    const int64_t t0 = stats ? ggml_time_us() : 0;
    const int64_t n_rows = (int64_t) sorted.size();
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    const int64_t t1 = stats ? ggml_time_us() : 0;

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

    if (stats) {
        stats->t_sort_us   += t1 - t0;
        stats->t_ranges_us += ggml_time_us() - t1;
    }

    prefetch_ranges_impl(ranges.data(), ranges.size(), stats);

    if (stats) {
        const int64_t t2 = ggml_time_us();
        llama_prefetch_piece_stats ps;
        ps.rows        = n_rows;
        ps.unique_rows = (int64_t) sorted.size();
        prefetch_page_stats(ranges, stats->page_size, ps);
        stats->pieces.push_back(ps);
        stats->t_pages_us += ggml_time_us() - t2;
    }
}

static void prefetch_stats_begin(llama_prefetch_stats * stats) {
    if (stats) {
        stats->page_size     = llama_page_size();
        stats->faults_before = llama_process_page_faults();
    }
}

static void prefetch_stats_end(llama_prefetch_stats * stats) {
    if (stats) {
        stats->faults_after = llama_process_page_faults();
    }
}

void llama_prefetch_table_rows(const ggml_tensor * table, const std::vector<int32_t> & rows,
                               enum llama_ple_prefetch mode, llama_prefetch_stats * stats) {
    if (!prefetch_enough_rows(rows.size(), mode)) {
        return;
    }
    prefetch_stats_begin(stats);
    const int64_t t0 = stats ? ggml_time_us() : 0;
    std::vector<int32_t> copy = rows;
    if (stats) {
        stats->t_split_us += ggml_time_us() - t0;
    }
    prefetch_table_rows_unchecked(table, std::move(copy), stats);
    prefetch_stats_end(stats);
}

void llama_prefetch_table_pieces_rows(const std::vector<ggml_tensor *> & parts, int64_t part_rows,
                                      const std::vector<int32_t> & rows, enum llama_ple_prefetch mode,
                                      llama_prefetch_stats * stats) {
    if (parts.empty() || !prefetch_enough_rows(rows.size(), mode)) {
        return;
    }
    prefetch_stats_begin(stats);
    const int64_t t0 = stats ? ggml_time_us() : 0;
    if (parts.size() == 1) {
        std::vector<int32_t> copy = rows;
        if (stats) {
            stats->t_split_us += ggml_time_us() - t0;
        }
        prefetch_table_rows_unchecked(parts[0], std::move(copy), stats);
        prefetch_stats_end(stats);
        return;
    }

    // the threshold counts the rows of the whole gather, each piece then gets its own (local) rows
    std::vector<std::vector<int32_t>> local(parts.size());
    for (const int32_t g : rows) {
        const size_t k = (size_t) ((int64_t) g / part_rows);
        if (g >= 0 && k < parts.size()) {
            local[k].push_back((int32_t) ((int64_t) g - (int64_t) k*part_rows));
        }
    }
    if (stats) {
        stats->t_split_us += ggml_time_us() - t0;
    }
    for (size_t k = 0; k < parts.size(); ++k) {
        if (!local[k].empty()) {
            prefetch_table_rows_unchecked(parts[k], std::move(local[k]), stats);
        }
    }
    prefetch_stats_end(stats);
}

void llama_prefetch_stats_log(const char * what, int layer, int64_t n_tokens, const llama_prefetch_stats & st) {
    // the fault counter is 32 bits on Windows: the difference is taken modulo 2^32
    const int64_t faults = st.faults_before >= 0 && st.faults_after >= 0
        ? (int64_t) (uint32_t) (st.faults_after - st.faults_before) : -1;
    std::string out;
    char buf[320];
    snprintf(buf, sizeof(buf),
        "{\"kind\":\"table_prefetch\",\"what\":\"%s\",\"layer\":%d,\"tokens\":%" PRId64 ",\"skipped\":%s,"
        "\"page_size\":%" PRId64 ",\"faults\":%" PRId64 ",\"us\":{\"hash\":%" PRId64 ",\"split\":%" PRId64
        ",\"sort\":%" PRId64 ",\"ranges\":%" PRId64 ",\"os\":%" PRId64 ",\"input\":%" PRId64 ",\"pages_diag\":%" PRId64 "}",
        what, layer, n_tokens, st.skipped ? "true" : "false", st.page_size, faults,
        st.t_hash_us, st.t_split_us, st.t_sort_us, st.t_ranges_us, st.t_os_us, st.t_input_us, st.t_pages_us);
    out += buf;
    // columns of "pieces" and "sum": rows, unique_rows, row_ranges, row_bytes, page_bytes, page_ranges, page_runs, unique_pages
    llama_prefetch_piece_stats sum;
    out += ",\"pieces\":[";
    for (size_t i = 0; i < st.pieces.size(); ++i) {
        const llama_prefetch_piece_stats & p = st.pieces[i];
        snprintf(buf, sizeof(buf), "%s[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
            i ? "," : "", p.rows, p.unique_rows, p.row_ranges, p.row_bytes, p.page_bytes, p.page_ranges, p.page_runs, p.unique_pages);
        out += buf;
        sum.rows        += p.rows;        sum.unique_rows  += p.unique_rows;
        sum.row_ranges  += p.row_ranges;  sum.row_bytes    += p.row_bytes;
        sum.page_bytes  += p.page_bytes;  sum.page_ranges  += p.page_ranges;
        sum.page_runs   += p.page_runs;   sum.unique_pages += p.unique_pages;
    }
    snprintf(buf, sizeof(buf), "],\"sum\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]}",
        sum.rows, sum.unique_rows, sum.row_ranges, sum.row_bytes, sum.page_bytes, sum.page_ranges, sum.page_runs, sum.unique_pages);
    out += buf;
    LLAMA_LOG_INFO("table_prefetch %s\n", out.c_str());
}
