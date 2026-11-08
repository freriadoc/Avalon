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

    // Linux: the whole pages inside [first, last) (offsets in the file) replaced by anonymous memory at the same
    // addresses (MAP_FIXED), filled from the file with O_DIRECT by n_threads readers in 64 MiB pieces, then read-only.
    // The same bytes at the same addresses, no longer read through the page cache. A piece whose read fails is mapped
    // from the file again. Returns the bytes now in anonymous memory (0 elsewhere)
    size_t make_resident(const struct llama_file * file, size_t first, size_t last, int n_threads);

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

// Prefetch the host pages covering these memory ranges.
void llama_prefetch(llama_memory_ranges mr);

size_t llama_path_max();

// the host memory this process can still take without swapping: the smaller of MemAvailable and, under every cgroup v2
// limit above it, the limit less what cannot be reclaimed (anonymous, shared, kernel); SIZE_MAX if unknown (not Linux).
// The rule of the expert arena (ggml-cuda moe_cache_room_bytes)
size_t llama_host_room_bytes();
