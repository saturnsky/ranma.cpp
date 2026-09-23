// Windows implementation of the expert cache OS abstraction. See expert-os.h.
// The whole file is empty on other platforms so that the build lists both implementations
// unconditionally.

#if defined(_WIN32)

#include "expert-os.h"

#include "ggml-impl.h"

#include <cstdio>
#include <chrono>
#include <malloc.h>
#include <system_error>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace ggml_cuda_expert {
namespace expert_os {

void * aligned_alloc(size_t bytes, size_t alignment) { return _aligned_malloc(bytes, alignment); }
void aligned_free(void * p) { _aligned_free(p); }

uint32_t load_acquire(const uint32_t * p) {
    return uint32_t(_InterlockedCompareExchange(reinterpret_cast<volatile long *>(const_cast<uint32_t *>(p)), 0, 0));
}
uint64_t load_acquire(const uint64_t * p) {
    return uint64_t(_InterlockedCompareExchange64(reinterpret_cast<volatile long long *>(const_cast<uint64_t *>(p)), 0, 0));
}
void store_release(uint32_t * p, uint32_t value) { _InterlockedExchange(reinterpret_cast<volatile long *>(p), long(value)); }

uint64_t total_physical_bytes() {
    MEMORYSTATUSEX status = {};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? uint64_t(status.ullTotalPhys) : 0;
}

bool pin_current_thread(int cpu, std::string & reason) {
    const DWORD total = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (cpu < 0) { cpu = int(total) - 1; }
    if (cpu < 0 || DWORD(cpu) >= total) { reason = "CPU index is outside the active processors"; return false; }
    int local = cpu;
    const WORD groups = GetActiveProcessorGroupCount();
    for (WORD group = 0; group < groups; ++group) {
        const DWORD count = GetActiveProcessorCount(group);
        if (local >= int(count)) { local -= int(count); continue; }
        GROUP_AFFINITY affinity = {};
        affinity.Group = group; affinity.Mask = KAFFINITY(1) << local;
        if (SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr)) { return true; }
        reason = "SetThreadGroupAffinity error " + std::to_string(GetLastError());
        return false;
    }
    reason = "CPU topology changed during pinning";
    return false;
}

bool atomic_replace(const std::filesystem::path & temporary, const std::filesystem::path & destination, std::string & error) {
    if (MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) { return true; }
    const std::error_code ec(int(GetLastError()), std::system_category());
    error = "failed to publish " + destination.string() + ": " + ec.message();
    return false;
}

std::optional<reservation> reserve(size_t bytes) {
    if (bytes == 0) {
        return std::nullopt;
    }
    void * base = VirtualAlloc(nullptr, bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (base == nullptr) {
        GGML_LOG_ERROR("expert cache: reserving %.2f MiB of address space failed with error %lu\n",
            bytes/1024.0/1024.0, (unsigned long) GetLastError());
        return std::nullopt;
    }
    reservation res;
    res.base  = base;
    res.bytes = bytes;
    return res;
}

bool commit(const reservation & res, size_t offset, size_t bytes) {
    if (res.base == nullptr || offset > res.bytes || bytes > res.bytes - offset) {
        return false;
    }
    if (bytes == 0) {
        return true;
    }
    void * page = VirtualAlloc(static_cast<char *>(res.base) + offset, bytes, MEM_COMMIT, PAGE_READWRITE);
    if (page == nullptr) {
        GGML_LOG_ERROR("expert cache: committing %.2f MiB failed with error %lu\n",
            bytes/1024.0/1024.0, (unsigned long) GetLastError());
        return false;
    }
    return true;
}

void release(reservation & res) {
    if (res.base != nullptr && !VirtualFree(res.base, 0, MEM_RELEASE)) {
        GGML_LOG_ERROR("expert cache: VirtualFree failed with error %lu\n", (unsigned long) GetLastError());
    }
    res.base  = nullptr;
    res.bytes = 0;
}

// ---- unbuffered reads --------------------------------------------------------------------------

file_handle open_unbuffered(const char * path) {
    if (path == nullptr || path[0] == '\0') {
        return nullptr;
    }
    // FILE_SHARE_READ only: another process may read the model while the server serves from it, but
    // nothing may write it. That is what makes the file offsets recorded at load time stay true.
    HANDLE handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        GGML_LOG_ERROR("expert cache: cannot open '%s' for unbuffered reading: error %lu\n",
            path, (unsigned long) GetLastError());
        return nullptr;
    }
    return (file_handle) handle;
}

void close_file(file_handle & file) {
    if (file != nullptr) {
        CloseHandle((HANDLE) file);
        file = nullptr;
    }
}

uint64_t file_size(file_handle file) {
    LARGE_INTEGER size;
    if (file == nullptr || !GetFileSizeEx((HANDLE) file, &size)) {
        return 0;
    }
    return (uint64_t) size.QuadPart;
}

struct read_queue::slot {
    OVERLAPPED ov = {};
    size_t     op = 0;
    bool       busy = false;
};

read_queue::read_queue(int depth) : depth_(depth < 1 ? 1 : (depth > max_queue_depth ? max_queue_depth : depth)) {
    slots_.reserve((size_t) depth_);
    for (int i = 0; i < depth_; ++i) {
        slot * s = new slot();
        s->ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
        if (s->ov.hEvent == nullptr) {
            delete s;
            GGML_LOG_ERROR("expert cache: read queue could not create an event: error %lu\n",
                (unsigned long) GetLastError());
            return;
        }
        slots_.push_back(s);
    }
    valid_ = true;
}

void read_queue::cancel_pending() {
    for (slot * s : slots_) {
        if (s->busy) {
            // Cancellation is only a request; the OVERLAPPED and destination remain owned by the OS.
            (void) CancelIoEx((HANDLE) ops_[s->op].file, &s->ov);
        }
    }
    for (slot * s : slots_) {
        if (!s->busy) {
            continue;
        }
        const DWORD wait = WaitForSingleObject(s->ov.hEvent, 5000);
        if (wait != WAIT_OBJECT_0 || !HasOverlappedIoCompleted(&s->ov)) {
            const read_op & op = ops_[s->op];
            GGML_ABORT("expert cache: cancellation did not complete for read of %zu bytes at offset %llu "
                       "(wait %lu); refusing to release pending I/O storage", op.bytes,
                (unsigned long long) op.offset, (unsigned long) wait);
        }
        ops_[s->op].ok = false;
        s->busy = false;
    }
}

read_queue::~read_queue() {
    cancel_pending();
    for (slot * s : slots_) {
        if (s->ov.hEvent != nullptr) {
            CloseHandle(s->ov.hEvent);
        }
        delete s;
    }
}

bool read_queue::submit(const read_op * ops, size_t n) {
    if (!valid_ || (ops == nullptr && n != 0)) {
        return false;
    }
    if (completed_) {
        ops_.clear();
        completed_ = false;
    }
    for (size_t i = 0; i < n; ++i) {
        const read_op & op = ops[i];
        if (op.file == nullptr || op.dst == nullptr ||
                op.offset % io_alignment != 0 || op.bytes % io_alignment != 0 ||
                ((uintptr_t) op.dst) % io_alignment != 0) {
            return false;
        }
        ops_.push_back(op);
    }
    return true;
}

bool read_queue::wait_all(int64_t deadline_ms, std::string * reason, const std::function<bool(size_t)> & progress) {
    if (!valid_) {
        return false;
    }
    completed_ = true;
    const ULONGLONG start = GetTickCount64();
    size_t next = 0;
    size_t done = 0;
    bool   ok   = true;
    char   message[256];

    // With a progress callback: which reads completed without error, and how long the completed
    // prefix in submit order was at the last report.
    std::vector<uint8_t> finished(progress ? ops_.size() : 0, 0);
    size_t reported = 0;
    auto finish = [&](size_t index) {
        if (progress && ops_[index].ok) { finished[index] = 1; }
    };
    auto report = [&]() {
        size_t prefix = reported;
        while (prefix < finished.size() && finished[prefix]) { ++prefix; }
        if (prefix == reported) {
            return true;
        }
        reported = prefix;
        if (progress(prefix)) {
            return true;
        }
        if (reason != nullptr && reason->empty()) {
            *reason = "the reader of the completed reads gave up";
        }
        return false;
    };

    auto remaining_ms = [&]() -> DWORD {
        if (deadline_ms < 0) {
            return INFINITE;
        }
        const ULONGLONG spent = GetTickCount64() - start;
        return spent >= (ULONGLONG) deadline_ms ? 0 : (DWORD) ((ULONGLONG) deadline_ms - spent);
    };

    while (done < ops_.size()) {
        // fill the queue
        while (next < ops_.size()) {
            slot * free_slot = nullptr;
            for (slot * s : slots_) {
                if (!s->busy) {
                    free_slot = s;
                    break;
                }
            }
            if (free_slot == nullptr) {
                break;
            }
            read_op & op = ops_[next];
            ResetEvent(free_slot->ov.hEvent);
            free_slot->ov.Offset     = (DWORD) (op.offset & 0xFFFFFFFFull);
            free_slot->ov.OffsetHigh = (DWORD) (op.offset >> 32);
            free_slot->op   = next;
            free_slot->busy = true;
            DWORD got = 0;
            const BOOL issued = ReadFile((HANDLE) op.file, op.dst, (DWORD) op.bytes, &got, &free_slot->ov);
            const DWORD issue_error = issued ? ERROR_SUCCESS : GetLastError();
            if (issued) {
                op.got  = got;
                op.ok   = true;
                free_slot->busy = false;
                finish(next);
                ++done;
            } else {
                const DWORD err = issue_error;
                if (err == ERROR_IO_PENDING) {
                    // stays in flight
                } else if (err == ERROR_HANDLE_EOF) {
                    op.got = 0;
                    op.ok  = true;
                    free_slot->busy = false;
                    finish(next);
                    ++done;
                } else {
                    snprintf(message, sizeof(message), "read of %zu bytes at offset %llu failed with error %lu",
                        op.bytes, (unsigned long long) op.offset, (unsigned long) err);
                    if (reason != nullptr && reason->empty()) {
                        *reason = message;
                    }
                    op.ok = false;
                    ok    = false;
                    free_slot->busy = false;
                    ++done;
                }
            }
            ++next;
        }

        // a longer completed prefix is reported only after the queue has been refilled
        if (progress && !report()) {
            cancel_pending();
            return false;
        }

        // wait for one of the in-flight reads
        HANDLE events[64];
        slot * busy[64];
        DWORD  count = 0;
        for (slot * s : slots_) {
            if (s->busy && count < 64) {
                events[count] = s->ov.hEvent;
                busy[count]   = s;
                ++count;
            }
        }
        if (count == 0) {
            continue; // everything issued so far completed inline
        }
        const DWORD wait = WaitForMultipleObjects(count, events, FALSE, remaining_ms());
        const DWORD wait_error = wait == WAIT_FAILED ? GetLastError() : ERROR_SUCCESS;
        if (wait == WAIT_TIMEOUT) {
            if (reason != nullptr && reason->empty()) {
                snprintf(message, sizeof(message), "%u reads still outstanding after %lld ms",
                    (unsigned) count, (long long) deadline_ms);
                *reason = message;
            }
            cancel_pending();
            return false;
        }
        if (wait < WAIT_OBJECT_0 || wait >= WAIT_OBJECT_0 + count) {
            if (reason != nullptr && reason->empty()) {
                snprintf(message, sizeof(message), "waiting for a read failed with error %lu",
                    (unsigned long) wait_error);
                *reason = message;
            }
            cancel_pending();
            return false;
        }
        slot *    s  = busy[wait - WAIT_OBJECT_0];
        read_op & op = ops_[s->op];
        DWORD     got = 0;
        if (GetOverlappedResult((HANDLE) op.file, &s->ov, &got, FALSE)) {
            op.got = got;
            op.ok  = true;
        } else {
            const DWORD err = GetLastError();
            if (err == ERROR_HANDLE_EOF) {
                op.got = 0;
                op.ok  = true;
            } else {
                snprintf(message, sizeof(message), "read of %zu bytes at offset %llu failed with error %lu",
                    op.bytes, (unsigned long long) op.offset, (unsigned long) err);
                if (reason != nullptr && reason->empty()) {
                    *reason = message;
                }
                op.ok = false;
                ok    = false;
            }
        }
        s->busy = false;
        finish(s->op);
        ++done;
    }
    // the completions of the last pass
    if (progress && ok && !report()) {
        return false;
    }

    return ok;
}

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // _WIN32
