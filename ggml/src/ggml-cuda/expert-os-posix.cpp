// POSIX stub of the expert cache OS abstraction. See expert-os.h.
//
// There is no implementation yet: exclusive mode and the host tier are Windows only for now, and
// validate_expert_params refuses them when supported() returns false, so nothing here is ever
// reached through an abort. The interface is shaped so that mmap(PROT_NONE) plus mprotect is a
// drop-in replacement later. The whole file is empty on Windows.

#if !defined(_WIN32)

#include "expert-os.h"

#include "ggml-impl.h"
#include <cstdlib>

namespace ggml_cuda_expert {
namespace expert_os {

namespace {
void warn_once() {
    static bool warned = false;
    if (!warned) {
        warned = true;
        GGML_LOG_WARN("expert cache: address reservation is not implemented on this platform; "
                      "exclusive mode is unavailable\n");
    }
}
} // namespace

bool atomic_replace(const std::filesystem::path & temporary, const std::filesystem::path & destination, std::string & error) {
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (!ec) { return true; }
    error = "failed to publish " + destination.string() + ": " + ec.message();
    return false;
}

std::optional<reservation> reserve(size_t bytes) {
    GGML_UNUSED(bytes);
    warn_once();
    return std::nullopt;
}

bool commit(const reservation & res, size_t offset, size_t bytes) {
    GGML_UNUSED(res);
    GGML_UNUSED(offset);
    GGML_UNUSED(bytes);
    warn_once();
    return false;
}

void release(reservation & res) {
    res.base  = nullptr;
    res.bytes = 0;
}

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // !_WIN32
