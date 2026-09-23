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
#include <unistd.h>

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

void * aligned_alloc(size_t bytes, size_t alignment) {
    void * p = nullptr;
    return posix_memalign(&p, alignment, bytes) == 0 ? p : nullptr;
}
void aligned_free(void * p) { free(p); }
uint32_t load_acquire(const uint32_t * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
uint64_t load_acquire(const uint64_t * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void store_release(uint32_t * p, uint32_t value) { __atomic_store_n(p, value, __ATOMIC_RELEASE); }
uint64_t total_physical_bytes() {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page  = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? uint64_t(pages)*uint64_t(page) : 0;
#else
    return 0;
#endif
}

bool pin_current_thread(int cpu, std::string & reason) {
    GGML_UNUSED(cpu); reason = "worker pinning is not implemented on this platform"; return false;
}
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

file_handle open_unbuffered(const char * path) {
    GGML_UNUSED(path);
    warn_once();
    return nullptr;
}

void close_file(file_handle & file) {
    file = nullptr;
}

uint64_t file_size(file_handle file) {
    GGML_UNUSED(file);
    return 0;
}

read_queue::read_queue(int depth) : depth_(depth > 0 ? depth : 1) {
    warn_once();
}

read_queue::~read_queue() {
}

bool read_queue::submit(const read_op * ops, size_t n) {
    GGML_UNUSED(ops);
    GGML_UNUSED(n);
    return false;
}

bool read_queue::wait_all(int64_t deadline_ms, std::string * reason, const std::function<bool(size_t)> & progress) {
    GGML_UNUSED(deadline_ms);
    GGML_UNUSED(progress);
    if (reason != nullptr && reason->empty()) {
        *reason = "unbuffered reads are not implemented on this platform";
    }
    return false;
}

} // namespace expert_os
} // namespace ggml_cuda_expert

#endif // !_WIN32
