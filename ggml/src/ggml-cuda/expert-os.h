#pragma once

// Narrow OS abstraction for the expert cache (design section 8): the address-space half that
// exclusive mode needs, and the unbuffered read queue that the SSD tier needs.
//
// Exclusive mode needs two things from the operating system. First, a reservation of virtual
// addresses with no pages behind it: the routed expert tensors are placed on it so that every
// (layer, kind) slice has a distinct, stable logical handle while its bytes live in the VRAM arena
// or in the host arena. Second, a way to commit the pages of the host arena, which is ordinary
// private memory that is later registered as coarse-grained mapped memory.
//
// The SSD tier adds a third: read the GGUF files of the model without the page cache, so that the
// bytes of an expert land straight in a ring slot and the operating system keeps no second copy of
// a file that is larger than the machine's memory.
//
// Unlimited inclusive storage is OS independent. Finite host storage uses this interface in both modes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace ggml_cuda_expert {
namespace expert_os {

void * aligned_alloc(size_t bytes, size_t alignment);
void aligned_free(void * p);
uint32_t load_acquire(const uint32_t * p);
uint64_t load_acquire(const uint64_t * p);
void store_release(uint32_t * p, uint32_t value);
bool pin_current_thread(int cpu, std::string & reason); // -1 selects the last active logical CPU.
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

// Installed physical memory of the machine, or 0 when the platform cannot answer. Free memory is
// never asked: what the cache must not do is plan a host tier the machine cannot hold at all.
uint64_t total_physical_bytes();

// ---- unbuffered reads (SSD tier) -------------------------------------------------------------
//
// Every offset, destination and length of an unbuffered read must be a multiple of the sector
// size. The tier keeps its ring slots at this alignment and reads the whole sector range that
// covers a slice, so the slice starts at `slot + (file_offset % io_alignment)`.

static constexpr size_t io_alignment = 4096;

inline uint64_t align_down_io(uint64_t value) { return value & ~(uint64_t) (io_alignment - 1); }
inline size_t   align_up_io(size_t value)     { return (value + io_alignment - 1) & ~(size_t) (io_alignment - 1); }

// Opaque file handle, null when the file is not open.
using file_handle = void *;

// Opens `path` for unbuffered overlapped reading, shared for reading only, so that no other
// process can write the file while the tier serves from it. Returns null and logs on failure.
file_handle open_unbuffered(const char * path);
void        close_file(file_handle & file);
// Byte size of an open file, or 0 when it cannot be asked.
uint64_t    file_size(file_handle file);

struct read_op {
    file_handle file   = nullptr;
    uint64_t    offset = 0;   // multiple of io_alignment
    void *      dst    = nullptr; // aligned to io_alignment
    size_t      bytes  = 0;   // multiple of io_alignment
    size_t      got    = 0;   // bytes the operating system delivered; short at end of file
    bool        ok     = false;
};

// One read queue per tier. submit() only records; wait_all() runs the pump and keeps at most
// `depth` reads in flight, which is what sets the queue depth the drive sees. At most
// max_queue_depth reads are in flight at once (the Win32 wait limit).
static constexpr int max_queue_depth = 64;
class read_queue {
public:
    explicit read_queue(int depth);
    ~read_queue();

    read_queue(const read_queue &) = delete;
    read_queue & operator=(const read_queue &) = delete;

    bool valid() const { return valid_; }
    int  depth() const { return depth_; }

    // Copies the descriptors; the caller may reuse its array. The destinations must stay alive
    // until wait_all() returns.
    bool submit(const read_op * ops, size_t n);
    // Issues and completes everything submitted since the last wait_all(). False on a failed read,
    // or when `deadline_ms` passes with reads still outstanding; `reason` then says which.
    // `progress`, when given, is called with n whenever the first n reads in submit order have all
    // completed without error, while the later ones stay in flight; the queue is refilled before
    // the call. Returning false cancels the outstanding reads and fails the wait.
    bool wait_all(int64_t deadline_ms, std::string * reason = nullptr,
                  const std::function<bool(size_t)> & progress = nullptr);

    // Result of the ops of the last wait_all(), in submit order.
    const std::vector<read_op> & results() const { return ops_; }

private:
    struct slot;
#if defined(_WIN32)
    void cancel_pending();
#endif

    bool  valid_     = false;
    bool  completed_ = true;   // the next submit() starts a new batch
    int   depth_     = 0;
    std::vector<read_op> ops_;
    std::vector<slot *>  slots_;
};

} // namespace expert_os
} // namespace ggml_cuda_expert
