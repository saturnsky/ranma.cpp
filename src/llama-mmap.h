#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define LLAMA_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct llama_file;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    // list of [first, last) byte ranges within a file
    using ranges = std::vector<std::pair<size_t, size_t>>;

    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false,
               const ranges & lazy_ranges = {});
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_memory_range {
    const void * addr;
    size_t size;
};

using llama_memory_ranges = std::vector<llama_memory_range>;

// host memory reserved as address space whose pages can be given back to the system and taken again, for buffers
// sized for the worst case of which a smaller part is in use (the graph metadata of llm_graph_result)
// the interface is the part of std::vector<uint8_t> that such a buffer needs, the contents are not initialized
struct llama_vmem {
    llama_vmem() = default;
    llama_vmem(const llama_vmem &) = delete;
    llama_vmem & operator=(const llama_vmem &) = delete;
    ~llama_vmem();

    // make all `size` bytes usable; the contents are kept only when the size does not change
    void resize(size_t size);

    // give back the pages past the first `keep` bytes; they must not be accessed until the next resize()
    void release_tail(size_t keep);

    uint8_t * data() const { return ptr; }
    size_t    size() const { return n; }

private:
    uint8_t * ptr = nullptr;
    size_t    n   = 0;
    size_t    n_committed = 0; // the first n_committed bytes are usable
    bool      paged = false;   // false: a plain heap allocation, release_tail() does nothing
};

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);

size_t llama_path_max();
