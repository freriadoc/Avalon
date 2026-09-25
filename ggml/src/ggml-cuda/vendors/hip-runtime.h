#pragma once

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Pinned ring for pageable host -> device copies (GGML_HIP_COPY_RING=N,MIB, default 8,16;
// GGML_HIP_COPY_RING=0 or GGML_CUDA_NO_PINNED: direct copies, as in upstream).
// The source is copied into the next of N pinned buffers of MIB MiB, whose DMA
// is queued on the stream; a buffer is reused once the event of its previous DMA has completed,
// so the call returns without waiting for the whole transfer and the source may change as soon
// as it returns. ROCm never sees the pageable range (no locking of mmap pages), and the pinned
// memory is bounded: N * MIB per thread. GGML_HIP_COPY_NT=0 copies with memcpy instead of
// non-temporal stores (below glibc's non-temporal threshold, memcpy writes through the cache and
// the throughput then depends on whether the ring fits in L3).
// Device -> host, device -> device, pinned sources, small copies and stream capture stay direct.
// Default measured on the qwen4exp benchmark (88 GiB mmap model, RX 7900 XT, PCIe 3.0 x16):
// 2 GiB copy bench 13.8-14.0 GB/s against 14.1 direct; prefill 757 t/s against 761 direct at
// 19k tokens, 518 against 513 at 81k; smaller rings lose 1 to 3 %.
struct ggml_hip_copy_ring_config {
    size_t n     = 8;
    size_t bytes = 16u << 20;
    bool   nt    = true;
    ggml_hip_copy_ring_config() {
        const char * env = std::getenv("GGML_HIP_COPY_RING");
        if (env) {
            char * end = nullptr;
            n = std::strtoul(env, &end, 10);
            bytes = end && *end == ',' ? (size_t) (std::strtod(end + 1, nullptr) * 1048576.0) : 0;
            bytes &= ~(size_t) 4095;
        }
        if (n == 0 || bytes == 0 || std::getenv("GGML_CUDA_NO_PINNED") != nullptr) {
            n = 0;
        }
        const char * nt_env = std::getenv("GGML_HIP_COPY_NT");
        nt = !nt_env || std::atoi(nt_env) != 0;
    }
};

inline const ggml_hip_copy_ring_config & ggml_hip_copy_ring_get_config() {
    static const ggml_hip_copy_ring_config config;
    return config;
}

#include <immintrin.h>

// non-temporal copy: the destination bypasses the cache; the fence makes the stores visible
// before the DMA is queued. dst must be 32-byte aligned (ring buffers are page aligned).
__attribute__((target("avx2")))
inline void ggml_hip_copy_nt(void * dst, const void * src, size_t n) {
    char * d = static_cast<char *>(dst);
    const char * s = static_cast<const char *>(src);
    size_t i = 0;
    for (; i + 128 <= n; i += 128) {
        const __m256i a = _mm256_loadu_si256((const __m256i *) (s + i));
        const __m256i b = _mm256_loadu_si256((const __m256i *) (s + i + 32));
        const __m256i c = _mm256_loadu_si256((const __m256i *) (s + i + 64));
        const __m256i e = _mm256_loadu_si256((const __m256i *) (s + i + 96));
        _mm256_stream_si256((__m256i *) (d + i), a);
        _mm256_stream_si256((__m256i *) (d + i + 32), b);
        _mm256_stream_si256((__m256i *) (d + i + 64), c);
        _mm256_stream_si256((__m256i *) (d + i + 96), e);
    }
    std::memcpy(d + i, s + i, n - i);
    _mm_sfence();
}

struct ggml_hip_copy_ring {
    int          device = -1;
    size_t       next   = 0;
    void *       buf[64] = {};
    hipEvent_t   ev[64]  = {};
    bool         ok     = false;

    explicit ggml_hip_copy_ring(int dev) : device(dev) {
        const ggml_hip_copy_ring_config & cfg = ggml_hip_copy_ring_get_config();
        if (cfg.n > 64) {
            return;
        }
        for (size_t i = 0; i < cfg.n; i++) {
            if (hipHostMalloc(&buf[i], cfg.bytes, hipHostMallocPortable) != hipSuccess ||
                hipEventCreateWithFlags(&ev[i], hipEventDisableTiming) != hipSuccess) {
                (void) hipGetLastError();
                return;
            }
        }
        ok = true;
        std::fprintf(stderr, "ggml_hip_copy_ring: device %d: %zu x %zu MiB pinned, %s copies\n",
                dev, cfg.n, cfg.bytes >> 20, cfg.nt ? "non-temporal" : "memcpy");
    }
    ~ggml_hip_copy_ring() {
        for (size_t i = 0; i < 64; i++) {
            if (ev[i])  { (void) hipEventDestroy(ev[i]); }
            if (buf[i]) { (void) hipHostFree(buf[i]); }
        }
    }
};

// nullptr: the ring does not apply, copy directly
inline ggml_hip_copy_ring * ggml_hip_copy_ring_for(const void * src, size_t size, hipMemcpyKind kind,
                                                   hipStream_t stream) {
    const ggml_hip_copy_ring_config & cfg = ggml_hip_copy_ring_get_config();
    if (cfg.n == 0 || kind != hipMemcpyHostToDevice || size <= (1u << 20) || !src) {
        return nullptr;
    }
    hipPointerAttribute_t attributes{};
    const hipError_t status = hipPointerGetAttributes(&attributes, src);
    if (status != hipSuccess) {
        (void) hipGetLastError(); // older HIP versions reject unregistered host pointers
    } else if (attributes.type != hipMemoryTypeUnregistered) {
        return nullptr; // pinned or device memory: the direct path is already asynchronous
    }
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture) != hipSuccess || capture != hipStreamCaptureStatusNone) {
        (void) hipGetLastError();
        return nullptr;
    }
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess || device < 0 || device >= 16) {
        (void) hipGetLastError();
        return nullptr;
    }
    // one ring per thread and device: its events are recorded on that device's streams
    thread_local ggml_hip_copy_ring * rings[16] = {};
    if (!rings[device]) {
        // never freed: at process exit, hipHostFree could run after the runtime shutdown
        rings[device] = new ggml_hip_copy_ring(device);
    }
    return rings[device]->ok ? rings[device] : nullptr;
}

inline hipError_t ggml_hip_copy_async(void * dst, const void * src, size_t size, hipMemcpyKind kind,
                                      hipStream_t stream) {
    ggml_hip_copy_ring * ring = ggml_hip_copy_ring_for(src, size, kind, stream);
    if (!ring) {
        return hipMemcpyAsync(dst, src, size, kind, stream);
    }
    const ggml_hip_copy_ring_config & cfg = ggml_hip_copy_ring_get_config();
    for (size_t offset = 0; offset < size; offset += cfg.bytes) {
        const size_t bytes = std::min(cfg.bytes, size - offset);
        const size_t i = ring->next;
        // an event never recorded counts as complete
        hipError_t status = hipEventSynchronize(ring->ev[i]);
        if (status != hipSuccess) {
            return status;
        }
        if (cfg.nt) {
            ggml_hip_copy_nt(ring->buf[i], static_cast<const char *>(src) + offset, bytes);
        } else {
            std::memcpy(ring->buf[i], static_cast<const char *>(src) + offset, bytes);
        }
        status = hipMemcpyAsync(static_cast<char *>(dst) + offset, ring->buf[i], bytes, kind, stream);
        if (status == hipSuccess) {
            status = hipEventRecord(ring->ev[i], stream);
        }
        if (status != hipSuccess) {
            return status;
        }
        ring->next = (i + 1) % cfg.n;
    }
    return hipSuccess;
}

inline hipError_t ggml_hip_copy(void * dst, const void * src, size_t size, hipMemcpyKind kind) {
    if (!ggml_hip_copy_ring_for(src, size, kind, nullptr)) {
        return hipMemcpy(dst, src, size, kind);
    }
    const hipError_t status = ggml_hip_copy_async(dst, src, size, kind, nullptr);
    return status == hipSuccess ? hipStreamSynchronize(nullptr) : status;
}
