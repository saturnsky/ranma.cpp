#pragma once

// Narrow OS abstraction for the expert cache: the operations that have no portable equivalent.
// Today that is the atomic publish of a profile record file.

#include <filesystem>
#include <string>

namespace ggml_cuda_expert {
namespace expert_os {

bool atomic_replace(const std::filesystem::path & temporary, const std::filesystem::path & destination, std::string & error);

} // namespace expert_os
} // namespace ggml_cuda_expert
