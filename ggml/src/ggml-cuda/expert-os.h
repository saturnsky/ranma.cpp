#pragma once

// Narrow OS abstraction for the expert cache (design section 8): the address-space half that
// exclusive mode needs.
//
// Exclusive mode needs two things from the operating system. First, a reservation of virtual
// addresses with no pages behind it: the routed expert tensors are placed on it so that every
// (layer, kind) slice has a distinct, stable logical handle while its bytes live in the VRAM arena
// or in the host arena. Second, a way to commit the pages of the host arena, which is ordinary
// private memory that is later registered as coarse-grained mapped memory.
//
// Unlimited inclusive storage is OS independent.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <filesystem>
#include <string>
#include <vector>

namespace ggml_cuda_expert {
namespace expert_os {

bool atomic_replace(const std::filesystem::path & temporary, const std::filesystem::path & destination, std::string & error);

struct reservation {
    void * base  = nullptr;
    size_t bytes = 0;
};

// True when reserve/commit/release are implemented on this platform. Inline and platform-only so
// that the option validator in common/ can ask without linking against the backend.
inline bool supported() {
#if defined(_WIN32)
    return true;
#else
    return false;
#endif
}

// Address-only reservation, no pages and no access. Returns nullopt on failure or when the
// platform has no implementation.
std::optional<reservation> reserve(size_t bytes);

// Backs [offset, offset + bytes) of a reservation with readable and writable pages. The range is
// rounded out to whole pages by the operating system.
bool commit(const reservation & res, size_t offset, size_t bytes);

// Frees the whole reservation, committed or not, and clears `res`.
void release(reservation & res);

} // namespace expert_os
} // namespace ggml_cuda_expert
