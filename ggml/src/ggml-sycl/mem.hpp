#ifndef GGML_SYCL_MEM_HPP
#define GGML_SYCL_MEM_HPP

#include <cstddef>
#include <sycl/sycl.hpp>

enum MemoryAPIType {
    MEMORY_API_TYPE_LEVEL_ZERO = 0,
    MEMORY_API_TYPE_SYCL = 1,
};

const char * mem_api_int2str(int mem_api);

/*
 * Returns true only when free_bytes and total_bytes were successfully
 * obtained from the requested API or the SYCL fallback.
 *
 * On failure, returns false, sets free_bytes to zero, and attempts to retain
 * total_bytes if the device capacity can still be queried. Callers must treat
 * free memory as unknown when false is returned.
 */
bool get_memory_size(
    const sycl::device & dev,
    size_t & free_bytes,
    size_t & total_bytes,
    MemoryAPIType api_type);

#endif // GGML_SYCL_MEM_HPP
