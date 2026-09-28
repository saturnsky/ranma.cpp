#pragma once

#include "common.cuh"

#include <vector>

// The matrix multiplication kernels dereference src0 in the device address space. Only an integrated device, or a
// kernel that resolves the mapped alias first, can do that for a tensor that lives in a host buffer.
static inline void ggml_cuda_assert_src0_is_device_readable(const ggml_tensor * src0) {
    GGML_ASSERT(src0->buffer == nullptr || !ggml_backend_buffer_is_host(src0->buffer) ||
                ggml_cuda_info().devices[ggml_cuda_get_device()].integrated);
}

// Device alias of the base of a host buffer that was registered as coarse-grained mapped memory.
// Returns nullptr if the buffer is not one of those, in which case the alias is the host address.
const void * ggml_backend_cuda_host_buffer_device_base(ggml_backend_buffer_t buffer);

#if defined(GGML_USE_HIP)

// Resolve a tensor in a mapped host buffer to the address that HIP kernels accept. Post-load coarse registration
// does not give the host and device virtual addresses the same value, so keep the byte offset of the tensor inside
// its buffer and add it to the base alias.
static inline const void * ggml_hip_mapped_host_device_alias(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_host(tensor->buffer));

    const char * base  = static_cast<const char *>(ggml_backend_buffer_get_base(tensor->buffer));
    const char * data  = static_cast<const char *>(tensor->data);
    const size_t bytes = ggml_backend_buffer_get_size(tensor->buffer);

    GGML_ASSERT(data >= base && (size_t) (data - base) < bytes);

    // A coarse buffer owns its alias, so there is nothing to cache and nothing that can go stale when a buffer is
    // freed and another one is allocated at the same address.
    const void * coarse_base = ggml_backend_cuda_host_buffer_device_base(tensor->buffer);
    if (coarse_base != nullptr) {
        return static_cast<const char *>(coarse_base) + (data - base);
    }

    // hipHostMalloc(Mapped) buffers alias to the host address, so a stale cache hit is harmless here.
    struct alias_entry {
        ggml_backend_buffer_t buffer;
        const char * host_base;
        const char * device_base;
        size_t size;
    };
    static thread_local std::vector<alias_entry> entries;

    for (const alias_entry & entry : entries) {
        if (entry.buffer == tensor->buffer && entry.host_base == base && entry.size == bytes) {
            return entry.device_base + (data - base);
        }
    }

    void * device_base = nullptr;
    CUDA_CHECK(hipHostGetDevicePointer(&device_base, const_cast<char *>(base), 0));
    entries.push_back({ tensor->buffer, base, static_cast<const char *>(device_base), bytes });
    return static_cast<const char *>(device_base) + (data - base);
}

#endif // GGML_USE_HIP
