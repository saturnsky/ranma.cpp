// POSIX side of the expert cache OS abstraction. See expert-os.h.
// The whole file is empty on Windows.

#if !defined(_WIN32)

#include "expert-os.h"

#include "ggml-impl.h"
#include <cstdlib>

namespace ggml_cuda_expert {
namespace expert_os {

bool atomic_replace(const std::filesystem::path & temporary, const std::filesystem::path & destination, std::string & error) {
    std::error_code ec;
    std::filesystem::rename(temporary, destination, ec);
    if (!ec) { return true; }
    error = "failed to publish " + destination.string() + ": " + ec.message();
    return false;
}

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // !_WIN32
