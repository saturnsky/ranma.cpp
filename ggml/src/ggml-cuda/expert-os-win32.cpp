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

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // _WIN32
