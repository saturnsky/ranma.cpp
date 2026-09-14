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

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace ggml_cuda_expert {
namespace expert_os {

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

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // _WIN32
