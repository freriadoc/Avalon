#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/moe-cache.cuh"
#include "../ggml-backend-moe-cache.h"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/conv3d.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/qwen4-hc.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/qsa-select.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <utility>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
// ROCm gives a process 4 hardware queues per device by default and maps further streams onto
// them round-robin: two streams on one queue run one after the other. With a second context
// (an MTP sidecar), the expert prefetch stream landed on the queue of the target's compute
// stream and the uploads stopped overlapping the compute (45% -> 5% of the copy time under a
// kernel, prefill -20%). One process drives up to 8 streams at once (target compute and
// prefetch, draft compute, MoE cache fills and zone moves, VBR transcodes, null stream,
// auxiliaries): 8 queues give each its own. Set at library load, before the HIP runtime reads
// its settings on its first call; an explicit GPU_MAX_HW_QUEUES is kept.
//
// rocBLAS (behind hipBLAS) gives each handle 32 MiB of device memory at its first GEMM, which it
// offers to Tensile for the solutions that use a workspace; ggml makes one handle per backend
// context, after the MoE cache was sized from the free memory (target and MTP draft: 64 MiB on top
// of the peak). The GEMMs ggml sends run without it: with 64 KiB (0 would mean the managed 32 MiB)
// the prefill is unchanged (956 vs 955 t/s, 11k prompt) and nothing fails. An explicit
// ROCBLAS_DEVICE_MEMORY_SIZE is kept.
__attribute__((constructor)) static void ggml_hip_set_hw_queues() {
    setenv("GPU_MAX_HW_QUEUES", "8", /*overwrite =*/ 0);
    setenv("ROCBLAS_DEVICE_MEMORY_SIZE", "65536", /*overwrite =*/ 0);
}
#endif

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        info.devices[id].total_vram   = device_vram; // the MoE cache's automatic reserve is bounded by it
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaErrorMemoryAllocation) {
                // last resort: surrender MoE cache storage before aborting
                (void)cudaGetLastError();
                if (ggml_moe_cache_trim(device) > 0) {
                    err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
                }
            }
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            // on OOM, surrender MoE cache storage and retry once; each vendor returns its own
            // error enum from cuMemCreate (CUresult on CUDA, hipError_t on HIP, MUresult on MUSA)
            auto create_result = cuMemCreate(&handle, reserve_size, &prop, 0);
            const bool out_of_memory =
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
                create_result == cudaErrorMemoryAllocation;
#else
                create_result == CUDA_ERROR_OUT_OF_MEMORY;
#endif
            if (out_of_memory && ggml_moe_cache_trim(device) > 0) {
                create_result = cuMemCreate(&handle, reserve_size, &prop, 0);
            }
            CU_CHECK(create_result);

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

static std::mutex g_cuda_contexts_mutex;
static std::vector<ggml_backend_cuda_context *> g_cuda_contexts;

void ggml_cuda_register_context(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(g_cuda_contexts_mutex);
    g_cuda_contexts.push_back(ctx);
}

void ggml_cuda_unregister_context(ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(g_cuda_contexts_mutex);
    g_cuda_contexts.erase(std::remove(g_cuda_contexts.begin(), g_cuda_contexts.end(), ctx), g_cuda_contexts.end());
}

static __global__ void k_queue_prime() {}

// N bytes of private memory per thread (the compiler adds a little): the volatile array stays in scratch
template <int N>
static __global__ void k_scratch_prime(int * out, int sel) {
    volatile int buf[N/4];
    for (int i = 0; i < N/4; ++i) {
        buf[i] = i*sel + (int) threadIdx.x;
    }
    if (sel == 0x7fffffff) {
        out[threadIdx.x] = buf[(sel + threadIdx.x) % (N/4)];
    }
}

typedef void (*scratch_prime_t)(int *, int);
template <size_t... I>
static std::array<scratch_prime_t, sizeof...(I)> ggml_cuda_scratch_primes(std::index_sequence<I...>) {
    return {{ k_scratch_prime<16*(I + 1)>... }};
}

// the prime kernel with the least private memory that covers `bytes` (steps of 16 B up to 2 KiB), or nullptr
static scratch_prime_t ggml_cuda_pick_scratch_prime(size_t bytes, size_t & got) {
    static const auto kernels = ggml_cuda_scratch_primes(std::make_index_sequence<128>{});
    static std::vector<size_t> sizes;
    if (sizes.empty()) {
        for (scratch_prime_t k : kernels) {
            cudaFuncAttributes attr = {};
            sizes.push_back(cudaFuncGetAttributes(&attr, (const void *) k) == cudaSuccess ? attr.localSizeBytes : 0);
        }
    }
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (sizes[i] >= bytes) {
            got = sizes[i];
            return kernels[i];
        }
    }
    return nullptr;
}

void ggml_cuda_prime_queues(int device) {
    std::lock_guard<std::mutex> lock(g_cuda_contexts_mutex);
    for (ggml_backend_cuda_context * ctx : g_cuda_contexts) {
        if (ctx->device != device) {
            continue;
        }
        cudaStream_t stream = ctx->stream();
        if (!ctx->primed) {
            k_queue_prime<<<1, 1, 0, stream>>>();
            ctx->primed = true;
        }
        if (ctx->fattn_private > ctx->primed_private) {
            size_t got = 0;
            scratch_prime_t k = ggml_cuda_pick_scratch_prime(ctx->fattn_private, got);
            if (k) {
                k<<<1, WARP_SIZE, 0, stream>>>(nullptr, 0);
                GGML_LOG_INFO("%s: %s: queue scratch for %zu B of private memory per thread (prime kernel %zu B)\n",
                        __func__, ctx->name.c_str(), ctx->fattn_private, got);
                ctx->primed_private = got;
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    ggml_cuda_unregister_context(this);
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (q8_1_reuse.buf != nullptr) {
        CUDA_CHECK(cudaFree(q8_1_reuse.buf));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    // false when the memory is not ours to free (borrowed from the MoE cache)
    bool owned = true;

    // set when the memory was borrowed from the MoE cache instead of allocated
    void *      moe_cache_session = nullptr;
    ggml_backend_t moe_cache_backend = nullptr;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr, bool owned = true) :
        device(device), dev_ptr(dev_ptr), owned(owned),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        if (moe_cache_session && ggml_moe_cache.give_back) {
            ggml_moe_cache.give_back(moe_cache_session, moe_cache_backend, dev_ptr);
        } else if (owned) {
            CUDA_CHECK(cudaFree(dev_ptr));
        }
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + original_size, 0, padded_size - original_size, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Compute buffers that borrow from the MoE cache.
//
// During a prefill the cache is bypassed, so its memory can host the compute buffer of
// the graph instead of being idle; the memory goes back to the cache when the buffer is
// freed. The buffer type keeps the name of the normal CUDA type, so every
// ggml_backend_buft_is_cuda check still holds, and only allocation differs. If the cache
// cannot lend, the normal allocation is used.
struct ggml_backend_cuda_borrowed_buffer_type_context {
    int device;
    void * session;
    ggml_backend_t backend;
    std::string name;
};

static size_t ggml_backend_cuda_borrowed_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // the cache lends contiguous runs of one pool: ask ggml-alloc for chunks it can serve
    static const size_t max_chunk = [] () -> size_t {
        const char * text = getenv("GGML_SCHED_BORROW_CHUNK_MB");
        const long long value = text && text[0] ? atoll(text) : 0;
        // 1 GiB: the loan plan seats every chunk list it can (see moe_cache_plan_borrows), so
        // the limit only trades fragmentation against how finely chunks spread over the
        // pools. Measured on the qwen4exp graph (7.2 GiB in one piece): 512 MiB chunks lose
        // 654 MiB to fragmentation, 1 GiB 292 MiB at the same most-lent pool (71 %), 2 GiB
        // 237 MiB but 77 %. Measured, not derived: the best value moves with the graph.
        return (value > 0 ? (size_t)value : 1024) << 20;
    }();
    return max_chunk;
}

static ggml_backend_buffer_t ggml_backend_cuda_borrowed_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_borrowed_buffer_type_context * ctx =
        (ggml_backend_cuda_borrowed_buffer_type_context *)buft->context;

    void * ptr = ggml_moe_cache.borrow ? ggml_moe_cache.borrow(ctx->session, ctx->backend, size) : nullptr;
    if (!ptr) {
        return ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), size);
    }

    ggml_backend_cuda_buffer_context * buf_ctx =
        new ggml_backend_cuda_buffer_context(ctx->device, ptr, /*owned =*/ false);
    buf_ctx->moe_cache_session = ctx->session;
    buf_ctx->moe_cache_backend = ctx->backend;
    return ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(ctx->device),
            ggml_backend_cuda_buffer_interface, buf_ctx, size);
}

ggml_backend_buffer_type_t ggml_backend_cuda_borrowed_buffer_type(
        int device, void * session, ggml_backend_t backend) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type *> types;
    for (ggml_backend_buffer_type * type : types) {
        const auto * ctx = (const ggml_backend_cuda_borrowed_buffer_type_context *)type->context;
        if (ctx->device == device && ctx->session == session && ctx->backend == backend) {
            return type;
        }
    }

    ggml_backend_buffer_type_i iface = ggml_backend_cuda_buffer_type_interface;
    iface.alloc_buffer = ggml_backend_cuda_borrowed_buffer_type_alloc_buffer;
    iface.get_max_size = ggml_backend_cuda_borrowed_buffer_type_get_max_size;

    ggml_backend_buffer_type * type = new ggml_backend_buffer_type{
        /* .iface   = */ iface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ new ggml_backend_cuda_borrowed_buffer_type_context{
            device, session, backend, GGML_CUDA_NAME + std::to_string(device)},
    };
    types.push_back(type);
    return type;
}

bool ggml_backend_cuda_borrowed_buffer_type_source(
        ggml_backend_buffer_type_t buft, void ** session, ggml_backend_t * backend) {
    if (!buft || buft->iface.alloc_buffer != ggml_backend_cuda_borrowed_buffer_type_alloc_buffer) {
        return false;
    }
    const auto * ctx = (const ggml_backend_cuda_borrowed_buffer_type_context *)buft->context;
    *session = ctx->session;
    *backend = ctx->backend;
    return true;
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name            = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer        = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

// What ggml_cuda_mul_mat_cublas_impl converts, in a scratch (GGML_SCHED_SCRATCH_SRC) when the scheduler
// gave one, else in pool memory: src0 and src1 in the compute type, and the output when it is not F32.
struct cublas_scratch_layout {
    size_t src0 = 0;
    size_t src1 = 0;
    size_t dst  = 0;
    size_t size = 0;
};

static bool ggml_cuda_cublas_prefer_f32_output(ggml_type compute_type, int cc) {
    if (compute_type == GGML_TYPE_F16) {
        return cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    }
    if (compute_type == GGML_TYPE_BF16) {
        return !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }
    return false;
}

static cublas_scratch_layout ggml_cuda_cublas_scratch_get_layout(const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * dst, ggml_type compute_type, int cc) {
    const size_t ts = ggml_type_size(compute_type);
    cublas_scratch_layout l;
    if (src0->type != compute_type) {
        l.src0 = l.size; l.size += GGML_PAD(ggml_nelements(src0)*ts, 256);
    }
    if (src1->type != compute_type) {
        l.src1 = l.size; l.size += GGML_PAD(ggml_nelements(src1)*ts, 256);
    }
    if (compute_type != GGML_TYPE_F32 && !ggml_cuda_cublas_prefer_f32_output(compute_type, cc)) {
        l.dst = l.size; l.size += GGML_PAD(ggml_nelements(dst)*ts, 256);
    }
    return l;
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const cublas_scratch_layout scratch_l = ggml_cuda_cublas_scratch_get_layout(src0, src1, dst, compute_type, cc);
    const ggml_tensor * scratch_t = dst->src[GGML_SCHED_SCRATCH_SRC];
    char * scratch = scratch_l.size > 0 && scratch_t && scratch_t->data && ggml_nbytes(scratch_t) >= scratch_l.size ?
        (char *) scratch_t->data : nullptr;
    const auto conv_buf = [&](ggml_cuda_pool_alloc<cuda_t> & pool_alloc, size_t off, int64_t n) -> cuda_t * {
        return scratch ? (cuda_t *) (scratch + off) : pool_alloc.alloc(n);
    };

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        cuda_t * src0_conv = conv_buf(src0_alloc, scratch_l.src0, ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_conv, ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_conv, ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_conv;
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        cuda_t * src1_conv = conv_buf(src1_alloc, scratch_l.src1, ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_conv, ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_conv, ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_conv;
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const bool prefer_f32_output = ggml_cuda_cublas_prefer_f32_output(compute_type, cc);

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) conv_buf(dst_temp, scratch_l.dst, ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda((const cuda_t *) dst_ptr, dst_ddf, ne_dst, main_stream);
    }
}

// the type ggml_cuda_mul_mat_cublas computes in
static ggml_type ggml_cuda_cublas_compute_type(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    const ggml_prec prec = (ggml_prec) ggml_get_op_params_i32(dst, 0);
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    // F16 is the only compute type that can not satisfy a request for BF16
    if (prec == GGML_PREC_BF16 && compute_type == GGML_TYPE_F16) {
        compute_type = fast_bf16_hardware_available(cc) ? GGML_TYPE_BF16 : GGML_TYPE_F32;
    } else if (prec == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }
    // src1 that may not be converted below F32
    if ((ggml_prec) ggml_get_op_params_i32(dst, 3) == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    return compute_type;
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_type compute_type = ggml_cuda_cublas_compute_type(src0, src1, dst, cc);

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // up to MMVQ_MAX_BATCH_SIZE columns (the kernel fuses the gate and the GLU for each), as many as one matrix of
    // a few tokens; not batched products
    if (tensor->op == GGML_OP_MUL_MAT && (dst->ne[1] > MMVQ_MAX_BATCH_SIZE || dst->ne[2] != 1 || dst->ne[3] != 1)) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_match_shared_expert(const ggml_cgraph * graph, int routed_idx, int shared_idx) {
    if (routed_idx + 2 >= graph->n_nodes || shared_idx + 2 >= graph->n_nodes || shared_idx < routed_idx + 3) {
        return false;
    }
    const int nodes[] = { routed_idx, routed_idx + 1, routed_idx + 2, shared_idx, shared_idx + 1, shared_idx + 2 };
    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU,
                           GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
    const int outputs[] = { routed_idx + 2, shared_idx + 2 };
    if (!ggml_can_fuse_subgraph_ext(graph, nodes, 6, ops, outputs, 2)) {
        return false;
    }

    const ggml_tensor * routed = graph->nodes[routed_idx + 2];
    const ggml_tensor * shared = graph->nodes[shared_idx + 2];
    const ggml_tensor * gate = routed->src[0];
    const ggml_tensor * up = routed->src[1];
    const ggml_tensor * shared_gate = shared->src[0];
    const ggml_tensor * shared_up = shared->src[1];
    const auto is_pair = [&](const ggml_tensor * a, const ggml_tensor * b, int idx) {
        return (a == graph->nodes[idx] && b == graph->nodes[idx + 1]) ||
               (b == graph->nodes[idx] && a == graph->nodes[idx + 1]);
    };
    // only batch-size independent checks here: graph_optimize must produce the same graph topology for every ubatch
    // size, otherwise ggml-alloc has to re-reserve (and the scheduler to synchronize) at runtime.
    // the MMVQ batch size check is done in ggml_cuda_try_fuse
    if (!is_pair(gate, up, routed_idx) || !is_pair(shared_gate, shared_up, shared_idx) ||
            !ggml_cuda_should_fuse_mul_mat(up, gate, routed) ||
            !ggml_cuda_should_fuse_mul_mat(shared_up, shared_gate, shared) ||
            !up->src[0]->buffer ||
            !ggml_is_quantized(up->src[0]->type)) {
        return false;
    }
    const ggml_tensor * input = up->src[1];
    const ggml_tensor * weight = up->src[0];
    const ggml_tensor * shared_weight = shared_up->src[0];
    if (input->op != GGML_OP_RESHAPE || input->src[0] != shared_up->src[1] ||
            input->ne[1] != 1 || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(shared_up->src[1]) || !ggml_is_matrix(shared_up->src[1]) ||
            weight->type != shared_weight->type || weight->ne[0] != shared_weight->ne[0] ||
            weight->ne[1] != shared_weight->ne[1] || weight->nb[1] != shared_weight->nb[1] || weight->ne[3] != 1 ||
            !ggml_is_matrix(shared_weight) || !ggml_is_contiguous(shared_weight) ||
            !ggml_is_contiguous(shared_gate->src[0]) || !ggml_is_contiguous(routed) || !ggml_is_contiguous(shared)) {
        return false;
    }
    if (shared_weight->op != GGML_OP_NONE || shared_gate->src[0]->op != GGML_OP_NONE ||
            ggml_get_glu_op(routed) != ggml_get_glu_op(shared) ||
            ggml_get_op_params_f32(routed, 3) != ggml_get_op_params_f32(shared, 3)) {
        return false;
    }
    return true;
}

// the kernel ggml_cuda_mul_mat computes dst with: the same answer before the graph is allocated (for the
// scheduler's scratch) and when it runs; a src0 without a buffer yet (an input copied from another backend,
// a result) will be in the compute buffer
enum ggml_cuda_mul_mat_path {
    GGML_CUDA_MM_FWHT,           // a Hadamard src0: the FWHT kernel, or the path without it if it declines
    GGML_CUDA_MM_MMVF,
    GGML_CUDA_MM_MMVF_T,         // a transposed vector: MMVF with src0 and src1 swapped
    GGML_CUDA_MM_MMF,
    GGML_CUDA_MM_MMVQ,
    GGML_CUDA_MM_MMQ,
    GGML_CUDA_MM_CUBLAS,
    GGML_CUDA_MM_Q8_0_F32,       // src1 that may not be converted below F32 (ggml_prec_set_src), small batch
    GGML_CUDA_MM_Q8_0_F16,       // src1 that may not be converted below F16, small batch
};

static ggml_cuda_mul_mat_path ggml_cuda_mul_mat_get_path(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int device,
        bool fwht = true) {
    GGML_TENSOR_BINARY_OP_LOCALS

    if (fwht && ggml_cuda_op_mul_mat_use_fwht(dst)) {
        return GGML_CUDA_MM_FWHT;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = src0->buffer ?
        ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src :
        ggml_nbytes(src0) != ggml_backend_buft_get_alloc_size(ggml_backend_cuda_buffer_type(device), src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return GGML_CUDA_MM_CUBLAS;
    }

    // src1 may not be converted below F32 or F16 (ggml_prec_set_src): MMVQ and MMQ would quantize it, cuBLAS
    // computes in F32 (asked) or F16
    const ggml_prec prec_src1 = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    if (ggml_is_quantized(src0->type) && prec_src1 == GGML_PREC_F32) {
        return ggml_cuda_mul_mat_q8_0_f32_supported(src0, src1, dst) ? GGML_CUDA_MM_Q8_0_F32 : GGML_CUDA_MM_CUBLAS;
    }
    if (ggml_is_quantized(src0->type) && prec_src1 == GGML_PREC_F16) {
        return ggml_cuda_mul_mat_q8_0_f16_supported(src0, src1, dst) ? GGML_CUDA_MM_Q8_0_F16 : GGML_CUDA_MM_CUBLAS;
    }

    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        return GGML_CUDA_MM_MMVF;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, warp_size, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        return GGML_CUDA_MM_MMVF_T;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        return GGML_CUDA_MM_MMF;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        return GGML_CUDA_MM_MMVQ;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return GGML_CUDA_MM_MMQ;
    }
    return GGML_CUDA_MM_CUBLAS;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_cuda_mul_mat_path path = ggml_cuda_mul_mat_get_path(src0, src1, dst, ctx.device);
    if (path == GGML_CUDA_MM_FWHT) {
        if (ggml_cuda_op_fwht(ctx, src1, dst)) {
            return;
        }
        path = ggml_cuda_mul_mat_get_path(src0, src1, dst, ctx.device, /*fwht =*/ false);
    }

    switch (path) {
        case GGML_CUDA_MM_FWHT:
            GGML_ABORT("fatal error");
        case GGML_CUDA_MM_MMVF:
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
            return;
        case GGML_CUDA_MM_MMVF_T: {
            ggml_tensor dst_vec = *dst;
            dst_vec.ne[0] = ne11;
            dst_vec.ne[1] = 1;
            dst_vec.nb[1] = dst_vec.nb[0]*ne11;
            dst_vec.nb[2] = dst_vec.nb[1];
            dst_vec.nb[3] = dst_vec.nb[1];
            ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
            return;
        }
        case GGML_CUDA_MM_MMF:
            ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
            return;
        case GGML_CUDA_MM_MMVQ:
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
            return;
        case GGML_CUDA_MM_MMQ:
            ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
            return;
        case GGML_CUDA_MM_CUBLAS:
            ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
            return;
        case GGML_CUDA_MM_Q8_0_F32:
            ggml_cuda_mul_mat_q8_0_f32(ctx, src0, src1, dst);
            return;
        case GGML_CUDA_MM_Q8_0_F16:
            ggml_cuda_mul_mat_q8_0_f16(ctx, src0, src1, dst);
            return;
    }
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        memcpy(dst_slice.op_params, dst->op_params, sizeof(dst_slice.op_params));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_3D:
            ggml_cuda_op_conv3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        case GGML_OP_QSA_SELECT:
            ggml_cuda_op_qsa_select(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

#if defined(GGML_USE_HIP)
// While a graph that waits for a fill of the MoE cache is queued on a compute stream (k_moe_split_wait: the hosted MoE
// dispatcher moves an expert to the GPU and the graph waits for its copy, ggml_backend_cuda_context::split_wait_queued),
// the stream's asynchronous copies to and from pinned host memory are made by a kernel on that stream, not by an SDMA
// engine. ROCr has one command ring per engine for the whole process (amd_gpu_agent.cpp) and a copy that depends on
// earlier work waits in it (a POLL_REGMEM packet, amd_blit_sdma.cpp), holding back every copy behind it. The outputs of
// a graph (logits, MTP hidden states, sampled tokens) are queued right after it, so they wait for its end; the fill it
// waits for could queue behind them in the same ring: a deadlock (seen with the logits, 970 KiB a token, ahead of a fill
// that found no free engine and fell back to engine 0). The kernel runs alone on its stream; reading or writing host
// memory from a kernel slows the kernels of other queues running meanwhile (tests/tlbbench: 3.6 -> 9.3 us), which is
// why the prefill's copies (MTP hidden states, 320 MiB a batch, while the draft computes) keep the engines: made by a
// kernel, the prefill with MTP fell from 1066 to 830 t/s. Up to 16 KiB ROCm uses a kernel already
// (GPU_FORCE_BLIT_COPY_SIZE).
static constexpr size_t ggml_cuda_host_copy_kernel_min = 16 << 10;
static constexpr int    ggml_cuda_host_copy_kernel_blocks = 16;
static __global__ void k_ggml_cuda_host_copy(const char * __restrict__ src, char * __restrict__ dst, size_t n) {
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    const size_t t = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    size_t done = 0;
    if ((((uintptr_t) src | (uintptr_t) dst) & 15) == 0) {
        for (size_t i = t; i < n / 16; i += stride) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        }
        done = n / 16 * 16;
    }
    for (size_t i = done + t; i < n; i += stride) {
        dst[i] = src[i];
    }
}
// the device address of pinned host memory, or nullptr (pageable memory: left to cudaMemcpyAsync)
static void * ggml_cuda_pinned_device_ptr(const void * host) {
    hipPointerAttribute_t attr;
    if (hipPointerGetAttributes(&attr, host) != hipSuccess) {
        (void) hipGetLastError();
        return nullptr;
    }
    if (attr.type != hipMemoryTypeHost || !attr.devicePointer) {
        return nullptr;
    }
    return (char *) attr.devicePointer + ((const char *) host - (const char *) attr.hostPointer);
}
// true when the copy was queued; to_host: dst is the host side, else src
static bool ggml_cuda_host_copy_by_kernel(void * dst, const void * src, size_t size, bool to_host, cudaStream_t stream) {
    if (size <= ggml_cuda_host_copy_kernel_min) {
        return false;
    }
    void * dev = ggml_cuda_pinned_device_ptr(to_host ? dst : src);
    if (!dev) {
        return false;
    }
    k_ggml_cuda_host_copy<<<ggml_cuda_host_copy_kernel_blocks, 256, 0, stream>>>(
        to_host ? (const char *) src : (const char *) dev, to_host ? (char *) dev : (char *) dst, size);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
#endif

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

#if defined(GGML_USE_HIP)
    if (cuda_ctx->split_wait_queued &&
        ggml_cuda_host_copy_by_kernel((char *) tensor->data + offset, data, size, false, cuda_ctx->stream())) {
        return;
    }
#endif
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

#if defined(GGML_USE_HIP)
    if (cuda_ctx->split_wait_queued &&
        ggml_cuda_host_copy_by_kernel(data, (const char *) tensor->data + offset, size, true, cuda_ctx->stream())) {
        return;
    }
#endif
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    // pinned host -> device: queue the DMA on the destination stream instead of a synchronous
    // set_tensor. The scheduler only lets a host backend write into its buffer again after it
    // has synchronized with this device (a split fed by the device waits for its outputs, and a
    // split without inputs waits for the previous split), so the DMA has completed by then.
    if (!ggml_backend_is_cuda(backend_src) && ggml_backend_is_cuda(backend_dst) &&
        buf_src && ggml_backend_buft_is_cuda_host(buf_src->buft) && ggml_backend_buffer_is_cuda(buf_dst) &&
        ggml_is_contiguous(src) && ggml_is_contiguous(dst) && ggml_nbytes(src) == ggml_nbytes(dst)) {
        ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;
        ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;
        if (cuda_ctx_dst->device != buf_ctx_dst->device) {
            return false;
        }
        ggml_cuda_set_device(cuda_ctx_dst->device);
#if defined(GGML_USE_HIP)
        if (!cuda_ctx_dst->split_wait_queued ||
            !ggml_cuda_host_copy_by_kernel(dst->data, src->data, ggml_nbytes(dst), false, cuda_ctx_dst->stream()))
#endif
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyHostToDevice, cuda_ctx_dst->stream()));
        return true;
    }

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    // a hosted MoE block may be waiting for this thread to compute its missed experts
    if (!ggml_cuda_moe_hosted_wait((void *) cuda_ctx->stream(), nullptr)) {
        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
    }
    cuda_ctx->split_wait_queued = false;

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// one graph per shape: the first node alone gave the verification batches of a speculative decoder (2, 3 or 4
// tokens, built in the same memory by llama.cpp) one shared entry, each change of draft length reset it (a step
// run kernel by kernel, then a new capture); the node count and the first node's shape tell them apart, and a
// collision only costs a capture (the properties of every node are compared before a replay)
static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    const ggml_tensor * n0 = cgraph->nodes[0];
    uintptr_t key = (uintptr_t) n0;
    key ^= (uintptr_t) cgraph->n_nodes * 0x9E3779B97F4A7C15ull;
    key ^= (uintptr_t) (n0->ne[1] * 0xC2B2AE3D27D4EB4Full + n0->ne[2] * 0x165667B19E3779F9ull);
    return (const void *) key;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm, neox and M-RoPE (text: not vision, no channel offset) modes
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_MROPE &&
        mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }
    if (((const int32_t *) rope->op_params)[15] != 0) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.

// Whether a node writes into [lo0, hi0) or [lo1, hi1): its result, or the temporary memory the scheduler gave it
// (src[GGML_SCHED_SCRATCH_SRC]), which ggml-alloc may also place over a tensor whose last reader in the graph has run.
static bool ggml_cuda_node_writes_into(const ggml_tensor * n, const char * lo0, const char * hi0, const char * lo1, const char * hi1) {
    const ggml_tensor * written[2] = { n, n->src[GGML_SCHED_SCRATCH_SRC] };
    for (const ggml_tensor * t : written) {
        if (t == nullptr || t->data == nullptr || ggml_nbytes(t) == 0) {
            continue;
        }
        const char * lo = (const char *) t->data;
        const char * hi = lo + ggml_nbytes(t);
        if ((lo < hi0 && lo0 < hi) || (lo < hi1 && lo1 < hi)) {
            return true;
        }
    }
    return false;
}

// A recurrent state gathered by GET_ROWS (one cache row per sequence) that only a gated delta net
// reads, through reshapes: the kernel can read the cache rows itself (one copy of the state, 3 MiB
// per layer in qwen4exp, and a kernel less). Each block reads its columns of the state before it
// writes them, so the kernel may write the new state over the row it reads; no node in between may
// write the cache.
static bool ggml_cuda_gdn_absorbs_gather(const ggml_cgraph * cgraph, int idx) {
    const ggml_tensor * gr = cgraph->nodes[idx];
    const ggml_tensor * src = gr->src[0];
    const ggml_tensor * ids = gr->src[1];
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(gr) || src->nb[0] != sizeof(float) || !ggml_is_contiguous(ids) ||
            gr->ne[2] != 1 || gr->ne[3] != 1 || src->ne[2] != 1 || src->ne[3] != 1 || ids->ne[0] != gr->ne[1] ||
            (gr->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    // the cache and the row indices must still hold their values when the GDN runs: ggml-alloc
    // takes the GET_ROWS for the last reader of the indices and may give their memory to a node in
    // between (seen in a prefill graph: the kernel then read past the cache and hung the queue)
    const char * src_lo = (const char *) src->data;
    const char * src_hi = src_lo + ggml_nbytes(src);
    const char * ids_lo = (const char *) ids->data;
    const char * ids_hi = ids_lo + ggml_nbytes(ids);
    const auto root = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->view_offs == 0) {
            t = t->src[0];
        }
        return t;
    };
    int gdn_idx = -1;
    for (int k = idx + 1; k < cgraph->n_nodes; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op == GGML_OP_GATED_DELTA_NET && root(n->src[5]) == gr) {
            const ggml_tensor * v = n->src[2];
            if (gdn_idx >= 0 || gr->ne[0] != v->ne[0] * v->ne[0] * v->ne[1] || gr->ne[1] != v->ne[3] ||
                    !ggml_is_contiguous(n->src[5])) {
                return false;
            }
            // ggml-alloc frees the indices after the GET_ROWS: the GDN's own result may be placed over them, and
            // its blocks that finish first would overwrite indices others have not read yet (seen in a prefill:
            // a block then read its state far past the cache and hung the queue)
            if (ggml_cuda_node_writes_into(n, ids_lo, ids_hi, ids_lo, ids_hi)) {
                return false;
            }
            gdn_idx = k;
            continue;
        }
        // any other reader of the gathered rows (a view of them is fine only on the way to the GDN)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * s = n->src[j];
            if (s && (s == gr || s->view_src == gr) && !(ggml_cuda_is_view_or_noop(n) && root(n) == gr)) {
                return false;
            }
        }
        // nothing may write the cache or the indices before the GDN has read them
        if (gdn_idx < 0 && !ggml_cuda_is_view_or_noop(n) && ggml_cuda_node_writes_into(n, src_lo, src_hi, ids_lo, ids_hi)) {
            return false;
        }
    }
    return gdn_idx >= 0;
}


// The pooled keys of a lightning indexer gathered by GET_ROWS from the cache rows (qwen4exp's QSA: one key every
// kpool cells, all of them gathered at every step) that only the indexer of a small batch reads, through reshapes:
// the kernel reads each key in its cache row through the row indices (ggml_cuda_gathered_rows_of), with the
// dequantization it already does for a quantized K. The gathered f32 copy is never made. No node in between may
// write the cache or the indices.
static bool ggml_cuda_indexer_absorbs_gather(const ggml_cgraph * cgraph, int idx) {
    const ggml_tensor * gr = cgraph->nodes[idx];
    const ggml_tensor * src = gr->src[0];
    const ggml_tensor * ids = gr->src[1];
    switch (src->type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16: case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }
    if (gr->op != GGML_OP_GET_ROWS || gr->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(gr) || src->nb[0] != ggml_type_size(src->type) || !ggml_is_contiguous(ids) ||
            gr->ne[2] != 1 || gr->ne[3] != 1 || src->ne[2] != 1 || src->ne[3] != 1 || ids->ne[0] != gr->ne[1] ||
            (gr->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const char * src_lo = (const char *) src->data;
    const char * src_hi = src_lo + ggml_nbytes(src);
    const char * ids_lo = (const char *) ids->data;
    const char * ids_hi = ids_lo + ggml_nbytes(ids);
    const auto root = [](const ggml_tensor * t) {
        while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->view_offs == 0) {
            t = t->src[0];
        }
        return t;
    };
    int li_idx = -1;
    for (int k = idx + 1; k < cgraph->n_nodes; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op == GGML_OP_LIGHTNING_INDEXER && root(n->src[1]) == gr) {
            const ggml_tensor * kt = n->src[1];
            if (li_idx >= 0 || !ggml_is_contiguous(kt) || kt->ne[0] != gr->ne[0] || kt->ne[0]*kt->ne[1] != gr->ne[0] ||
                    kt->ne[2] != gr->ne[1] || kt->ne[3] != 1) {
                return false;
            }
            // small batches only (decoding, MTP verification): a prefill reads every key once per token, so its
            // gather is amortized, and reading in place dequantizes each key again for every token (-0.5 % at 19 k)
            if (n->src[0]->ne[2] > 8) {
                return false;
            }
            // ggml-alloc frees the indices after the GET_ROWS: the scores may be placed over them (seen in a prefill)
            if (ggml_cuda_node_writes_into(n, src_lo, src_hi, ids_lo, ids_hi)) {
                return false;
            }
            li_idx = k;
            continue;
        }
        // any other reader of the gathered rows (a view of them is fine only on the way to the indexer)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * s = n->src[j];
            if (s && (s == gr || s->view_src == gr) && !(ggml_cuda_is_view_or_noop(n) && root(n) == gr)) {
                return false;
            }
        }
        // nothing may write the cache or the indices before the indexer has read them
        if (li_idx < 0 && !ggml_cuda_is_view_or_noop(n) && ggml_cuda_node_writes_into(n, src_lo, src_hi, ids_lo, ids_hi)) {
            return false;
        }
    }
    return li_idx >= 0;
}

// concat(conv state, x) -> rollback-slot copies of its tail -> ssm_conv (-> silu), as qwen4exp's linear
// attention builds it (build_conv_state_at): one kernel, ggml_cuda_op_ssm_conv_cached
struct ggml_cuda_conv_cached_match {
    const ggml_tensor *       root   = nullptr;   // the state's source (a GET_ROWS of the cache, or any contiguous tensor)
    const ggml_tensor *       x      = nullptr;
    const ggml_tensor *       w      = nullptr;
    ggml_tensor *             out    = nullptr;   // ssm_conv, or the silu after it
    bool                      silu   = false;
    int                       last   = 0;
    ggml_cuda_ssm_conv_slots  slots;
};

static const ggml_tensor * ggml_cuda_view_root(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->view_offs == 0) {
        t = t->src[0];
    }
    return t;
}

// The gated delta net's l2 norm of q and k (rms_norm over heads of 128 of a view of the conv output's first
// channels, then scale) right after the cached conv that produces them (views in between): the conv computes
// it (ggml_cuda_op_ssm_conv_cached, qk_out). Returns the scale node's index, -1 if not.
static int ggml_cuda_match_conv_l2_norm(const ggml_cgraph * cgraph, int last, const ggml_tensor * conv_out) {
    int j = last + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
        j++;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return -1;
    }
    const ggml_tensor * rms = cgraph->nodes[j];
    const ggml_tensor * sc  = cgraph->nodes[j + 1];
    if (rms->op != GGML_OP_RMS_NORM || sc->op != GGML_OP_SCALE || sc->src[0] != rms || ggml_get_op_params_f32(sc, 1) != 0.0f) {
        return -1;
    }
    const ggml_tensor * v = rms->src[0];
    if (v->view_src != conv_out || v->view_offs != 0 || v->type != GGML_TYPE_F32 || v->ne[0] != 128 ||
            v->nb[0] != sizeof(float) || v->nb[1] != 128 * sizeof(float) || v->nb[2] != conv_out->nb[1] ||
            v->ne[2] != conv_out->ne[1] || v->ne[3] != conv_out->ne[2] || (int64_t) v->ne[1] * 128 > conv_out->ne[0] ||
            !ggml_is_contiguous(rms) || !ggml_is_contiguous(sc) || ggml_nelements(sc) != ggml_nelements(v)) {
        return -1;
    }
    return j + 1;
}


// root_absorbed: the state's GET_ROWS is left to the kernel (ggml_cuda_conv_absorbs_gather), which then reads the
// cache rows and never the gathered copy: that copy's memory, dead once the window is built, may hold the output
static bool ggml_cuda_match_conv_cached(const ggml_cgraph * cgraph, int ci, ggml_cuda_conv_cached_match & m,
        bool root_absorbed) {
    const ggml_tensor * cc = cgraph->nodes[ci];
    if (cc->op != GGML_OP_CONCAT || cc->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cc, 0) != 0) {
        return false;
    }
    const ggml_tensor * state = cc->src[0];
    const ggml_tensor * x     = cc->src[1];
    const int64_t n_cols = state->ne[0];
    const int64_t n_ch   = state->ne[1];
    const int64_t n_s    = state->ne[2];
    const int64_t n_t    = x->ne[0];
    const ggml_tensor * root = ggml_cuda_view_root(state);
    if (n_cols != 3 || state->type != GGML_TYPE_F32 || !ggml_is_contiguous(state) || state->ne[3] != 1 ||
            x->type != GGML_TYPE_F32 || x->ne[1] != n_ch || x->ne[2] != n_s || x->ne[3] != 1 ||
            n_t < 1 || n_t > GGML_CUDA_SSM_CONV_CACHED_MAX_T || !root || !ggml_is_contiguous(root) ||
            ggml_nelements(root) != ggml_nelements(state)) {
        return false;
    }
    int nodes[32] = { ci };
    int n_nodes = 1;
    int outs[8];
    int n_outs = 0;
    m = {};
    for (int k = ci + 1; k < cgraph->n_nodes && n_nodes < 30; k++) {
        ggml_tensor * n = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(n)) {
            if (n->view_src == cc) {
                nodes[n_nodes++] = k;
            }
            continue;
        }
        if (n->op == GGML_OP_CPY && !m.w) {
            const ggml_tensor * tail = n->src[0];
            const ggml_tensor * dst  = n->src[1];
            if (tail->view_src != cc || tail->ne[0] != 3 || tail->ne[1] != n_ch || tail->ne[2] != n_s ||
                    tail->nb[1] != cc->nb[1] || tail->nb[2] != cc->nb[2] || tail->view_offs % sizeof(float) != 0 ||
                    (int64_t) (tail->view_offs / sizeof(float)) + 3 > 3 + n_t ||
                    dst->type != GGML_TYPE_F32 || dst->ne[0] != 3 * n_ch || dst->ne[1] != n_s || dst->nb[0] != sizeof(float) ||
                    m.slots.n >= GGML_CUDA_SSM_CONV_CACHED_MAX_SLOTS) {
                return false;
            }
            m.slots.dst[m.slots.n]      = (char *) dst->data;
            m.slots.nb1[m.slots.n]      = dst->nb[1];
            m.slots.tail_col[m.slots.n] = (int) (tail->view_offs / sizeof(float));
            m.slots.n++;
            nodes[n_nodes++] = k;
            outs[n_outs++]   = k;
            continue;
        }
        if (n->op == GGML_OP_SSM_CONV && !m.w && n->src[0] == cc) {
            const ggml_tensor * w = n->src[1];
            if (w->type != GGML_TYPE_F32 || w->ne[0] != 4 || w->ne[1] != n_ch || w->nb[0] != sizeof(float) ||
                    n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n)) {
                return false;
            }
            m.w   = w;
            m.out = n;
            m.last = k;
            nodes[n_nodes++] = k;
            continue;
        }
        if (m.w && n->op == GGML_OP_UNARY && ggml_get_unary_op(n) == GGML_UNARY_OP_SILU && n->src[0] == m.out &&
                ggml_is_contiguous(n)) {
            m.out  = n;
            m.silu = true;
            m.last = k;
            nodes[n_nodes++] = k;
        }
        break;
    }
    if (!m.w || m.slots.n == 0) {
        return false;
    }
    outs[n_outs++] = m.last;
    enum ggml_op ops[32];
    for (int j = 0; j < n_nodes; j++) {
        ops[j] = cgraph->nodes[nodes[j]]->op;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, nodes, n_nodes, ops, outs, n_outs)) {
        return false;
    }
    // the output overlaps nothing the kernel reads
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * al = (const char *) a->data, * ah = al + ggml_nbytes(a);
        const char * bl = (const char *) b->data, * bh = bl + ggml_nbytes(b);
        return al < bh && bl < ah;
    };
    const ggml_tensor * reads[4] = { root, root->op == GGML_OP_GET_ROWS ? root->src[0] : root, x, m.w };
    // ggml-alloc may give the output the memory of x (dead once the window is built): a thread loads its
    // channel of x for every token before writing, and element (t, c) of the output sits where element
    // (t, c) of x does when x is the transposed [n_ch, n_t] tensor, so that exact alias is safe
    const bool x_alias_ok = m.out->data == x->data && x->nb[1] == sizeof(float) &&
                            x->nb[0] == (size_t) n_ch * sizeof(float) && x->nb[2] == (size_t) (n_ch * n_t) * sizeof(float);
    for (const ggml_tensor * r : reads) {
        if (r == root && root_absorbed && root->op == GGML_OP_GET_ROWS) {
            continue;
        }
        if (overlap(m.out, r) && !(r == x && x_alias_ok)) {
            return false;
        }
    }
    m.root = root;
    m.x    = x;
    return true;
}

// The conv state's GET_ROWS (one cache row per sequence) left to the cached ssm_conv that reads it: one
// sequence only (a thread may then write the row it read), nothing in between writes the cache or the
// row indices (see ggml_cuda_gdn_absorbs_gather), the gathered rows read by nothing else.
static bool ggml_cuda_conv_absorbs_gather(const ggml_cgraph * cgraph, int idx) {
    const ggml_tensor * gr = cgraph->nodes[idx];
    if (gr->op != GGML_OP_GET_ROWS || gr->ne[1] != 1 || gr->src[0]->type != GGML_TYPE_F32 ||
            gr->src[1]->type != GGML_TYPE_I32 || gr->src[0]->nb[0] != sizeof(float) || (gr->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * src = gr->src[0];
    const ggml_tensor * ids = gr->src[1];
    const char * slo = (const char *) src->data, * shi = slo + ggml_nbytes(src);
    const char * ilo = (const char *) ids->data, * ihi = ilo + ggml_nbytes(ids);
    for (int k = idx + 1; k < cgraph->n_nodes; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op == GGML_OP_CONCAT && ggml_cuda_view_root(n->src[0]) == gr) {
            ggml_cuda_conv_cached_match m;
            if (!ggml_cuda_match_conv_cached(cgraph, k, m, true)) {
                return false;
            }
            // nothing after reads the gathered rows (the concat's views are inside the fused group)
            for (int j = m.last + 1; j < cgraph->n_nodes; j++) {
                for (int q = 0; q < GGML_MAX_SRC; q++) {
                    const ggml_tensor * sq = cgraph->nodes[j]->src[q];
                    if (sq && ggml_cuda_view_root(sq) == gr) {
                        return false;
                    }
                }
            }
            return true;
        }
        if (!ggml_cuda_is_view_or_noop(n)) {
            for (int q = 0; q < GGML_MAX_SRC; q++) {
                const ggml_tensor * sq = n->src[q];
                if (sq && ggml_cuda_view_root(sq) == gr) {
                    return false;   // another reader before the concat
                }
            }
            if (n->data && ggml_nbytes(n) > 0) {
                const char * lo = (const char *) n->data, * hi = lo + ggml_nbytes(n);
                if ((lo < shi && slo < hi) || (lo < ihi && ilo < hi)) {
                    return false;
                }
            }
        }
    }
    return false;
}

static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

// match ssm_scan + the strided cpy that scatters its state snapshots into the cache, so the kernel writes them and skips the cpy
static int ggml_cuda_try_ssm_scan_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_ssm_scan_fused_cache & fused_state_cpy) {
    const ggml_tensor * ssm = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the scan output must not be a graph output
    if (ssm->op != GGML_OP_SSM_SCAN || ssm->type != GGML_TYPE_F32 || (ssm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const int64_t K = ggml_get_op_params_i32(ssm, 0); // snapshot slot count

    const ggml_tensor * s = ssm->src[0];
    const ggml_tensor * x = ssm->src[1];
    const ggml_tensor * A = ssm->src[3];

    const int64_t d_state = s->ne[0];
    const int64_t D       = d_state * s->ne[1] * x->ne[1]; // d_state * head_dim * n_head
    const int64_t n_tok   = x->ne[2];
    const int64_t n_seqs  = x->ne[3];

    // only the mamba-2 kernels (group scan and SSD) write to the cache; mamba-1 still uses the cpy
    if (A->nb[1] != sizeof(float) || (d_state != 96 && d_state != 128 && d_state != 256)) {
        return 0;
    }

    // the scan reads its input rows from the cache (picked by ids), so with more than one seq a seq can read a row that another seq writes in the same launch
    if (n_seqs != 1) {
        return 0;
    }

    const int64_t n_written = std::min<int64_t>(n_tok, K);
    const size_t  tail_off  = ggml_row_size(GGML_TYPE_F32, ggml_nelements(x));

    // snapshot cpy is the first real node after the scan (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the scan snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this scan's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != ssm || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D, the per-seq stride the kernel takes from src0->nb[3]
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback slot 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        float bias;
        memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));

        return bias == 0.0f && scale->type == GGML_TYPE_F32;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16 && unary->type != GGML_TYPE_BF16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        // the fused softcap kernel is F32 only
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
static bool ggml_cuda_fusion_disabled() {
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    return disable_fusion;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // the mean of groups of gathered rows, as qwen4exp's indexer pools a block's raw keys (build_qsa_sel): GET_ROWS,
    // RESHAPE [n, n_group, n_out], CONT of the first VIEW [n, n_out], an ADD per following VIEW, SCALE; one kernel
    // (the same order of additions) instead of six
    if (node->op == GGML_OP_GET_ROWS && node->type == GGML_TYPE_F32 && i + 4 < cgraph->n_nodes &&
            (node->src[0]->type == GGML_TYPE_Q8_0 || node->src[0]->type == GGML_TYPE_Q4_0) &&
            node->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(node->src[1]) && ggml_is_contiguous(node) &&
            node->ne[2] == 1 && node->ne[3] == 1 && node->src[0]->ne[2] == 1 && node->src[0]->ne[3] == 1) {
        const ggml_tensor * rs = cgraph->nodes[i + 1];
        const int n_group = rs->op == GGML_OP_RESHAPE && rs->src[0] == node && rs->ne[3] == 1 ? (int) rs->ne[1] : 0;
        const int count = 2 + 2*n_group + 1;   // GET_ROWS, RESHAPE, (VIEW, CONT|ADD) per member, SCALE
        if (n_group >= 2 && n_group <= 8 && i + count <= cgraph->n_nodes && node->ne[0] % 2 == 0) {
            bool ok = true;
            const ggml_tensor * prev = nullptr;
            for (int k = 0; k < n_group && ok; ++k) {
                const ggml_tensor * view = cgraph->nodes[i + 2 + 2*k];
                const ggml_tensor * op   = cgraph->nodes[i + 3 + 2*k];
                ok = view->op == GGML_OP_VIEW && view->src[0] == rs && view->view_offs == (size_t) k*rs->nb[1] &&
                     view->ne[0] == rs->ne[0] && view->ne[1] == rs->ne[2] && view->ne[2] == 1 && view->ne[3] == 1 &&
                     view->nb[1] == rs->nb[2] && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
                     ggml_are_same_shape(op, view) &&
                     (k == 0 ? op->op == GGML_OP_CONT && op->src[0] == view
                             : op->op == GGML_OP_ADD && op->src[0] == prev && op->src[1] == view);
                prev = op;
            }
            ggml_tensor * scale = cgraph->nodes[i + count - 1];
            if (ok && scale->op == GGML_OP_SCALE && scale->src[0] == prev && ggml_is_contiguous(scale) &&
                    scale->type == GGML_TYPE_F32) {
                enum ggml_op ops[2 + 2*8 + 1];
                for (int k = 0; k < count; ++k) {
                    ops[k] = cgraph->nodes[i + k]->op;
                }
                const int out_idx = i + count - 1;
                // the output must not cover the rows or the indices the kernel reads
                const auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
                    const char * a0 = (const char *) a->data; const char * b0 = (const char *) b->data;
                    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
                };
                if (ggml_can_fuse_subgraph(cgraph, i, count, ops, &out_idx, 1) &&
                        !overlaps(scale, node->src[0]) && !overlaps(scale, node->src[1])) {
                    float sc, bias;
                    memcpy(&sc,   (const float *) scale->op_params + 0, sizeof(float));
                    memcpy(&bias, (const float *) scale->op_params + 1, sizeof(float));
                    if (ggml_cuda_get_rows_mean(*cuda_ctx, node, (float *) scale->data, n_group, sc, bias)) {
                        return count - 1;
                    }
                }
            }
        }
    }

    // rows gathered to f32, then converted to f16 (the K/V cells a small batch attends, qwen4exp's QSA gather): the
    // rows are written to f16 at once (get_rows dequantizes to half as well), the f32 copy is never made
    if (node->op == GGML_OP_GET_ROWS && node->type == GGML_TYPE_F32 && i + 1 < cgraph->n_nodes &&
            ggml_can_fuse(cgraph, i, { GGML_OP_GET_ROWS, GGML_OP_CPY })) {
        ggml_tensor * cpy = cgraph->nodes[i + 1];
        // ggml_cast makes the copy its own src[1]: the generic overlap check would always refuse, so check what the
        // fused kernel reads (rows and indices) against what it writes
        auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
            const char * a0 = (const char *) a->data; const char * b0 = (const char *) b->data;
            return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
        };
        if (cpy->src[0] == node && cpy->type == GGML_TYPE_F16 && ggml_are_same_shape(node, cpy) &&
                ggml_is_contiguous(node) && ggml_is_contiguous(cpy) &&
                !overlaps(cpy, node->src[0]) && !overlaps(cpy, node->src[1])) {
            ggml_tensor rows_f16 = *node;
            rows_f16.type  = GGML_TYPE_F16;
            rows_f16.data  = cpy->data;
            rows_f16.nb[0] = ggml_type_size(GGML_TYPE_F16);
            for (int d = 1; d < GGML_MAX_DIMS; d++) {
                rows_f16.nb[d] = rows_f16.nb[d - 1] * rows_f16.ne[d - 1];
            }
            ggml_cuda_op_get_rows(*cuda_ctx, &rows_f16);
            return 1;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID && cuda_ctx->stream_context().concurrent_events.empty() &&
            ggml_cuda_match_shared_expert(cgraph, i, i + 3) &&
            ggml_cuda_should_fuse_mul_mat_vec_q(cgraph->nodes[i + 2]->src[1])) {
        const int outputs[] = { i + 2, i + 5 };
        if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, 6, outputs, 2)) {
            ggml_tensor * routed = cgraph->nodes[i + 2];
            ggml_tensor * shared = cgraph->nodes[i + 5];
            const ggml_tensor * up = routed->src[1];
            ggml_cuda_mm_fusion_args_host fusion{};
            fusion.gate = routed->src[0]->src[0];
            fusion.glu_op = ggml_get_glu_op(routed);
            fusion.glu_limit = ggml_get_op_params_f32(routed, 3);
            fusion.shared_up = shared->src[1]->src[0];
            fusion.shared_gate = shared->src[0]->src[0];
            fusion.shared_dst = shared;
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, up->src[0], up->src[1], up->src[2], routed, &fusion);
            return 5;
        }
    }

    if (node->op == GGML_OP_SCALE || node->op == GGML_OP_MUL_MAT) {
        const int skip = ggml_cuda_try_fuse_qwen4_hc(*cuda_ctx, cgraph, i,
            [](const ggml_cgraph * g, int idx, int count, const int * outs, int n_outs) {
                return ggml_cuda_check_fusion_memory_ranges(g, idx, count, outs, n_outs);
            });
        if (skip > 0) {
            return skip;
        }
    }

    // sigmoid of the shared expert's gate -> gated shared expert, the weighted reduction of the routed
    // experts, then their sum: one kernel (qwen-style MoE with a gated shared expert). The gate's
    // product stays a kernel of its own: the output may reuse the memory of its input.
    if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID && i + 2 < cgraph->n_nodes) {
        const ggml_tensor * sig   = node;
        const ggml_tensor * logit = sig->src[0];
        const ggml_tensor * gated = cgraph->nodes[i + 1];
        const ggml_tensor * sh    = gated->op == GGML_OP_MUL ? (gated->src[0] == sig ? gated->src[1] : gated->src[0]) : nullptr;
        ggml_cuda_moe_weighted_reduction_match match;
        if (logit->type == GGML_TYPE_F32 && ggml_is_contiguous(logit) && logit->ne[0] == 1 && logit->ne[2] == 1 && logit->ne[3] == 1 &&
                sh && (gated->src[0] == sig || gated->src[1] == sig) && sh->type == GGML_TYPE_F32 && ggml_is_contiguous(sh) &&
                ggml_is_contiguous(gated) && sh->ne[1] == logit->ne[1] &&
                ggml_cuda_match_moe_weighted_reduction(cgraph, i + 2, match) && i + 2 + match.node_count < cgraph->n_nodes) {
            const int add_idx = i + 2 + match.node_count;
            const ggml_tensor * add = cgraph->nodes[add_idx];
            const int count = match.node_count + 3;
            if (add->op == GGML_OP_ADD && ggml_is_contiguous(add) && add->type == GGML_TYPE_F32 && count < 32 &&
                    ((add->src[0] == match.dst && add->src[1] == gated) || (add->src[0] == gated && add->src[1] == match.dst)) &&
                    ggml_are_same_shape(add, match.dst) && ggml_are_same_shape(sh, match.dst)) {
                enum ggml_op ops[32];
                for (int k = 0; k < count; k++) {
                    ops[k] = cgraph->nodes[i + k]->op;
                }
                if (ggml_can_fuse_subgraph(cgraph, i, count, ops, &add_idx, 1) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, count, &add_idx, 1)) {
                    ggml_cuda_op_moe_weighted_reduction(*cuda_ctx, match.experts, match.expert_scale, match.weights,
                                                        cgraph->nodes[add_idx], sh, logit);
                    return count - 1;
                }
            }
        }
    }

    // the conv state's window, its rollback copies and the convolution: one kernel
    if (node->op == GGML_OP_CONCAT) {
        ggml_cuda_conv_cached_match m;
        const bool gathered = cuda_ctx->gdn_gathers.count(ggml_cuda_view_root(node->src[0])) > 0;
        if (ggml_cuda_match_conv_cached(cgraph, i, m, gathered)) {
            const ggml_tensor * state = node->src[0];
            const float *   st  = gathered ? (const float *) m.root->src[0]->data : (const float *) m.root->data;
            const int32_t * ids = gathered ? (const int32_t *) m.root->src[1]->data : nullptr;
            const int64_t   row = gathered ? (int64_t) (m.root->src[0]->nb[1] / sizeof(float)) : state->ne[0] * state->ne[1];
            // the l2 norm of q and k after it, when its result overlaps nothing the conv reads or writes
            const int sc_idx = ggml_cuda_match_conv_l2_norm(cgraph, m.last, m.out);
            if (sc_idx > 0) {
                ggml_tensor * sc = cgraph->nodes[sc_idx];
                const ggml_tensor * rms = cgraph->nodes[sc_idx - 1];
                const auto overlaps = [sc](const ggml_tensor * t) {
                    if (t == nullptr || t->data == nullptr) {
                        return false;
                    }
                    const char * a = (const char *) sc->data;
                    const char * b = (const char *) t->data;
                    return a < b + ggml_nbytes(t) && b < a + ggml_nbytes(sc);
                };
                // the state comes from the cache rows when the gather is skipped (its own memory is never used)
                bool ok = !overlaps(m.x) && !overlaps(m.w) && !overlaps(m.out) &&
                          (gathered ? !overlaps(m.root->src[0]) && !overlaps(m.root->src[1]) : !overlaps(m.root));
                for (int k = 0; k < m.slots.n && ok; k++) {
                    const char * a = (const char *) sc->data;
                    ok = !(a < m.slots.dst[k] + m.slots.nb1[k] * state->ne[2] && m.slots.dst[k] < a + ggml_nbytes(sc));
                }
                if (ok) {
                    float eps;
                    memcpy(&eps, rms->op_params, sizeof(float));
                    ggml_cuda_op_ssm_conv_cached(*cuda_ctx, st, ids, row, m.x, m.w, (float *) m.out->data,
                                                 state->ne[1], m.x->ne[0], state->ne[2], m.silu, m.slots,
                                                 (float *) sc->data, rms->src[0]->ne[0] * rms->src[0]->ne[1], eps,
                                                 ggml_get_op_params_f32(sc, 0));
                    return sc_idx - i;
                }
            }
            ggml_cuda_op_ssm_conv_cached(*cuda_ctx, st, ids, row, m.x, m.w, (float *) m.out->data,
                                         state->ne[1], m.x->ne[0], state->ne[2], m.silu, m.slots);
            return m.last - i;
        }
    }

    // gated delta net gates: add(alpha, dt) -> softplus -> mul(A), then sigmoid(b), next to each other
    // (views in between): one kernel
    if (node->op == GGML_OP_ADD && i + 3 < cgraph->n_nodes) {
        int idx[4] = { i, -1, -1, -1 };   // add, softplus, mul, sigmoid
        int k = i + 1;
        const auto next_real = [&](int from) {
            while (from < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[from])) {
                from++;
            }
            return from;
        };
        k = next_real(k);       idx[1] = k < cgraph->n_nodes ? k : -1;
        k = next_real(k + 1);   idx[2] = idx[1] >= 0 && k < cgraph->n_nodes ? k : -1;
        k = next_real(k + 1);   idx[3] = idx[2] >= 0 && k < cgraph->n_nodes ? k : -1;
        if (idx[3] >= 0) {
            const ggml_tensor * add = node;
            const ggml_tensor * sp  = cgraph->nodes[idx[1]];
            ggml_tensor *       mul = cgraph->nodes[idx[2]];
            ggml_tensor *       sig = cgraph->nodes[idx[3]];
            const auto vec = [](const ggml_tensor * t, int64_t n) {
                return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nelements(t) == n;
            };
            const int64_t H = add->ne[0];
            const int64_t n = ggml_nelements(add);
            const ggml_tensor * dt    = add->src[1];
            const ggml_tensor * alpha = add->src[0];
            const ggml_tensor * a     = mul->src[0] == sp ? mul->src[1] : mul->src[0];
            const ggml_tensor * b     = sig->src[0];
            if (sp->op == GGML_OP_UNARY && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS && sp->src[0] == add &&
                    mul->op == GGML_OP_MUL && (mul->src[0] == sp || mul->src[1] == sp) &&
                    sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
                    vec(add, n) && vec(alpha, n) && vec(dt, H) && dt->ne[0] == H && vec(a, H) && a->ne[0] == H &&
                    vec(sp, n) && vec(mul, n) && vec(sig, n) && vec(b, n) && ggml_are_same_shape(add, alpha)) {
                const int count = idx[3] - i + 1;
                enum ggml_op ops[32];
                bool fits = count < 32;
                for (int j = 0; fits && j < count; j++) {
                    ops[j] = cgraph->nodes[i + j]->op;
                }
                // the outputs, and the views in between (the GDN reads g through a reshape)
                int outs[32];
                int n_outs = 0;
                for (int j = i; fits && j <= idx[3]; j++) {
                    if (j == idx[2] || j == idx[3] || ggml_cuda_is_view_or_noop(cgraph->nodes[j])) {
                        outs[n_outs++] = j;
                    }
                }
                // each thread reads alpha[i] and b[i] before writing g[i] and beta[i]: an output may alias
                // alpha or b exactly (ggml-alloc gives beta the memory of alpha once the add has read
                // it), not dt or A (read by every token)
                const auto no_overlap = [&]() {
                    const ggml_tensor * ins[4] = { alpha, b, dt, a };
                    const ggml_tensor * os[2]  = { mul, sig };
                    for (const ggml_tensor * o : os) {
                        const char * lo = (const char *) o->data, * hi = lo + ggml_nbytes(o);
                        for (int j = 0; j < 4; j++) {
                            const char * il = (const char *) ins[j]->data, * ih = il + ggml_nbytes(ins[j]);
                            const bool exact = j < 2 && il == lo && ih == hi;
                            if (!exact && il < hi && lo < ih) {
                                return false;
                            }
                        }
                    }
                    const char * m0 = (const char *) mul->data, * s0 = (const char *) sig->data;
                    return !(m0 < s0 + ggml_nbytes(sig) && s0 < m0 + ggml_nbytes(mul));
                };
                if (fits && ggml_can_fuse_subgraph(cgraph, i, count, ops, outs, n_outs) && no_overlap()) {
                    ggml_cuda_op_gdn_gates(*cuda_ctx, alpha, dt, a, b, mul, sig);
                    return count - 1;
                }
            }
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    // ssm_scan -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_SSM_SCAN) {
        ggml_cuda_ssm_scan_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_ssm_scan_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused ssm_scan snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_ssm_scan_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;
        ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    // the ids read next by a MoE block whose weights are in host memory (a hosted block): its entry
                    // kernel computes the top-k, a kernel less
                    bool hosted_reader = false;
                    for (int k = i + (int) ops.size(); k < cgraph->n_nodes; k++) {
                        const ggml_tensor * n = cgraph->nodes[k];
                        if (n->op == GGML_OP_MUL_MAT_ID && n->src[2] == ids) {
                            hosted_reader = n->src[0]->buffer && !ggml_backend_buft_is_cuda(n->src[0]->buffer->buft);
                            break;
                        }
                    }
                    if (!(hosted_reader && ggml_cuda_topk_moe_defer(*cuda_ctx, logits, weights, ids, clamp, scale, args))) {
                        ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    }
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // gated norm: rms_norm(x) * w * sigmoid(gate), w one vector
    if (node->op == GGML_OP_RMS_NORM && i + 3 < cgraph->n_nodes) {
        // the next three computing nodes (views in between, such as the gate's reshape)
        int idx[3];
        int k = i;
        for (int j = 0; j < 3; j++) {
            k++;
            while (k < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
                k++;
            }
            idx[j] = k < cgraph->n_nodes ? k : cgraph->n_nodes - 1;
        }
        const ggml_tensor * mul_w = cgraph->nodes[idx[0]];
        const ggml_tensor * sig   = cgraph->nodes[idx[1]];
        ggml_tensor *       out   = cgraph->nodes[idx[2]];
        const ggml_tensor * x     = node->src[0];
        const ggml_tensor * w     = mul_w->op == GGML_OP_MUL ? (mul_w->src[0] == node ? mul_w->src[1] : mul_w->src[0]) : nullptr;
        const ggml_tensor * gate  = sig->src[0];
        if (w && (mul_w->src[0] == node || mul_w->src[1] == node) &&
                sig->op == GGML_OP_UNARY && ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID &&
                out->op == GGML_OP_MUL && ((out->src[0] == mul_w && out->src[1] == sig) || (out->src[0] == sig && out->src[1] == mul_w)) &&
                x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && ggml_is_contiguous(out) && out->type == GGML_TYPE_F32 &&
                w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == x->ne[0] &&
                gate->type == GGML_TYPE_F32 && gate->nb[0] == sizeof(float) && ggml_are_same_shape(gate, x) && ggml_are_same_shape(out, x)) {
            // the computing nodes only: the gate's reshape in between is a view of a tensor computed
            // before the group
            const int nodes[4] = { i, idx[0], idx[1], idx[2] };
            const enum ggml_op ops[4] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL };
            const int out_idx = idx[2];
            const int count = idx[2] - i + 1;
            // a row of x and of gate is read before it is written: out may alias either exactly, not w
            const auto aliasing_ok = [&]() {
                const char * lo = (const char *) out->data, * hi = lo + ggml_nbytes(out);
                const ggml_tensor * ins[3] = { x, gate, w };
                for (int j = 0; j < 3; j++) {
                    const char * il = (const char *) ins[j]->data, * ih = il + ggml_nbytes(ins[j]);
                    const bool exact = j < 2 && il == lo && ih == hi && ggml_is_contiguous(ins[j]);
                    if (!exact && il < hi && lo < ih) {
                        return false;
                    }
                }
                return true;
            };
            if (ggml_can_fuse_subgraph_ext(cgraph, nodes, 4, ops, &out_idx, 1) && aliasing_ok()) {
                // the q8_1 of the output too when the next product reads it (whole warps per q8_1 block)
                void * q8_1 = x->ne[0] % QK8_1 == 0 ? ggml_cuda_q8_1_produce(*cuda_ctx, cgraph, out_idx) : nullptr;
                ggml_cuda_op_rms_norm_mul_sigmoid_gate(*cuda_ctx, node, w, gate, out, q8_1);
                return count - 1;
            }
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
        ggml_cuda_op_rms_norm_scale_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    // consecutive mmvq products of the same activation by matrices of one type and width (qkv / z /
    // alpha / beta, Q / K / V): one launch. Last among the fusions, so that the others keep their nodes.
    if (node->op == GGML_OP_MUL_MAT) {
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        const ggml_tensor * src1 = node->src[1];
        const auto member = [&](const ggml_tensor * mm) {
            const ggml_tensor * w = mm->src[0];
            const ggml_tensor * a = mm->src[1];
            return mm->op == GGML_OP_MUL_MAT && w->type == node->src[0]->type && ggml_is_quantized(w->type) &&
                   w->ne[0] == node->src[0]->ne[0] && w->ne[2] == 1 && w->ne[3] == 1 && w->nb[0] == ggml_type_size(w->type) &&
                   w->buffer && ggml_backend_buft_is_cuda(w->buffer->buft) &&
                   // a compute-buffer matrix would need its padding cleared first, as ggml_cuda_mul_mat_vec_q does
                   ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                   (a == src1 || (a->data == src1->data && ggml_are_same_shape(a, src1) && ggml_are_same_stride(a, src1))) &&
                   a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) && a->ne[2] == 1 && a->ne[3] == 1 &&
                   mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm) &&
                   !ggml_cuda_op_mul_mat_use_fwht(mm) &&
                   !ggml_cuda_should_use_mmvf(w->type, cc, ggml_cuda_info().devices[cuda_ctx->device].warp_size, w->ne, w->nb, a->ne[1]) &&
                   ggml_cuda_should_use_mmvq(w->type, cc, a->ne[1]);
        };
        // a member whose result an existing fusion takes next (bias add, glu) is left alone
        const auto taken_by_fusion = [&](int idx) {
            for (int k = idx + 1; k < std::min(idx + 4, cgraph->n_nodes); k++) {
                const ggml_tensor * n = cgraph->nodes[k];
                if ((n->op == GGML_OP_GLU || n->op == GGML_OP_ADD) && (n->src[0] == cgraph->nodes[idx] || n->src[1] == cgraph->nodes[idx])) {
                    return true;
                }
            }
            return false;
        };
        if (member(node) && !taken_by_fusion(i)) {
            int idx[GGML_CUDA_MMVQ_MULTI_MAX] = { i };
            int n = 1;
            for (int k = i + 1; k < cgraph->n_nodes && n < GGML_CUDA_MMVQ_MULTI_MAX; k++) {
                const ggml_tensor * nk = cgraph->nodes[k];
                if (ggml_cuda_is_view_or_noop(nk)) {
                    continue;
                }
                if (!member(nk) || taken_by_fusion(k) || nk->src[0]->ne[1] % 8 != 0 ||
                        cgraph->nodes[idx[n - 1]]->src[0]->ne[1] % 8 != 0) {
                    break;
                }
                idx[n++] = k;
            }
            if (n >= 2) {
                const ggml_tensor * w[GGML_CUDA_MMVQ_MULTI_MAX];
                ggml_tensor *       d[GGML_CUDA_MMVQ_MULTI_MAX];
                bool ok = true;
                for (int m = 0; m < n; m++) {
                    w[m] = cgraph->nodes[idx[m]]->src[0];
                    d[m] = cgraph->nodes[idx[m]];
                    // the outputs overlap neither the activation nor each other
                    const char * lo = (const char *) d[m]->data, * hi = lo + ggml_nbytes(d[m]);
                    const char * al = (const char *) src1->data, * ah = al + ggml_nbytes(src1);
                    ok = ok && !(lo < ah && al < hi);
                    for (int q = 0; q < m && ok; q++) {
                        const char * ql = (const char *) d[q]->data, * qh = ql + ggml_nbytes(d[q]);
                        ok = !(lo < qh && ql < hi);
                    }
                }
                if (ok) {
                    ggml_cuda_mul_mat_vec_q_multi(*cuda_ctx, w, d, n, src1);
                    return idx[n - 1] - i;
                }
            }
        }
    }

    return 0;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;
    cuda_ctx->split_wait_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            cuda_ctx->q8_1_reuse.valid = false;
            cuda_ctx->mmq_scratch_last = nullptr;
            cuda_ctx->gdn_gathers.clear();
            // a node (or a fused group, a hosted MoE block) computed nodes [first, last]: drop the
            // reused activation quantization if one of them wrote over it
            const auto written = [&](int first, int last) {
                // a group that produced the quantization of its output wrote that output only (its other nodes
                // are not computed, though their addresses may cover it)
                for (int k = first; k <= last && k < cgraph->n_nodes && !cuda_ctx->q8_1_reuse.producer; k++) {
                    if (!ggml_cuda_is_view_or_noop(cgraph->nodes[k])) {
                        ggml_cuda_q8_1_reuse_written(*cuda_ctx, cgraph->nodes[k]);
                    }
                }
                cuda_ctx->q8_1_reuse.producer = nullptr;
            };
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // a deferred top-k, before anything but its hosted block reads its ids or weights
                if (cuda_ctx->topk_pending.logits) {
                    const auto & tp = cuda_ctx->topk_pending;
                    const bool hosted = node->op == GGML_OP_MUL_MAT_ID && node->src[2] == tp.ids_t;
                    const auto root = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
                    bool reads = false;
                    for (int j = 0; j < GGML_MAX_SRC && !reads && !hosted; j++) {
                        const ggml_tensor * src = node->src[j];
                        reads = src && (root(src) == root(tp.ids_t) || root(src) == root(tp.weights_t));
                    }
                    if (reads) {
                        ggml_cuda_topk_moe_pending_flush(*cuda_ctx);
                    }
                }

                // the CPU rows of a hosted MoE block, before a node (or a fusion starting here)
                // reads the block's output. Only the weighting reads it; the fusions that contain it
                // start at it or, with the shared expert's gate, two nodes earlier (sigmoid, gated
                // product, weighting): a window of 3. A wider one would also hold a node that runs
                // before such a fusion (the gate's product) behind the CPU rows.
                constexpr int hosted_read_window = 3;
                for (int k = i; k < std::min(i + hosted_read_window, cgraph->n_nodes); k++) {
                    ggml_cuda_moe_hosted_flush(*cuda_ctx, cgraph->nodes[k]);
                }

                // weights in host memory: the scheduler left a MoE block of the cache on the GPU
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0]->buffer &&
                        !ggml_backend_buft_is_cuda(node->src[0]->buffer->buft)) {
                    const int covered = ggml_cuda_moe_hosted_block(*cuda_ctx, cgraph, i);
                    if (covered <= 0) {
                        GGML_ABORT("%s: MUL_MAT_ID with host weights outside a hosted MoE block (%s)", __func__, node->name);
                    }
                    written(i, i + covered);
                    i += covered;
                    continue;
                }

                // a recurrent state's gather left to the gated delta net or the cached ssm_conv that reads it, the
                // pooled keys' gather left to the lightning indexer
                // (only with fusions on: the cached ssm_conv that reads the conv state's rows is a fusion)
                if (node->op == GGML_OP_GET_ROWS && !ggml_cuda_fusion_disabled() &&
                        (ggml_cuda_gdn_absorbs_gather(cgraph, i) || ggml_cuda_conv_absorbs_gather(cgraph, i) ||
                         ggml_cuda_indexer_absorbs_gather(cgraph, i))) {
                    cuda_ctx->gdn_gathers.insert(node);
                    continue;
                }

                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    written(i, i + nodes_to_skip);
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                written(i, i);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
            ggml_cuda_moe_hosted_flush(*cuda_ctx, nullptr);
            ggml_cuda_topk_moe_pending_flush(*cuda_ctx);
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph->split_wait = cuda_ctx->split_wait_captured;
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        cuda_ctx->split_wait_queued |= graph->split_wait;
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // the q8_1 reuse buffer (ggml_cuda_q8_1_reuse): allocated here, never inside a graph capture;
    // 1 MiB holds the activations of up to 64 rows of 10240 (larger ones are not reused)
    if (!cuda_ctx->q8_1_reuse.buf) {
        const size_t size = GGML_CUDA_Q8_1_REUSE_BYTES;
        if (cudaMalloc((void **) &cuda_ctx->q8_1_reuse.buf, size) == cudaSuccess) {
            cuda_ctx->q8_1_reuse.size = size;
        } else {
            (void) cudaGetLastError();
            cuda_ctx->q8_1_reuse.buf = nullptr;
        }
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

// the scratch ggml_cuda_mul_mat(_id) takes for dst instead of pool memory, 0 if none; key: what it then
// holds (equal keys: nodes that can share the content, mmq's quantized activation)
static size_t ggml_cuda_mul_mat_scratch_size(const ggml_tensor * dst, int device, int64_t key[GGML_CUDA_MMQ_SCRATCH_KEY]) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int cc = ggml_cuda_info().devices[device].cc;
    if (dst->op == GGML_OP_MUL_MAT_ID) {
        if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
            return 0;
        }
        if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE &&
                (ggml_is_quantized(src0->type) ? dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc) : GGML_CUDA_CC_IS_AMD(cc))) {
            return 0;
        }
        if (!ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/ src0->ne[2])) {
            return 0;
        }
        return ggml_cuda_mmq_scratch_size(dst, cc, key);
    }
    switch (ggml_cuda_mul_mat_get_path(src0, src1, dst, device)) {
        case GGML_CUDA_MM_MMQ:
            return ggml_cuda_mmq_scratch_size(dst, cc, key);
        case GGML_CUDA_MM_MMVQ: {
            // the quantized activation, when the q8_1 reuse buffer cannot hold it (prompt batches of the
            // MTP draft's 3D products); nothing to share
            const size_t need = ggml_nrows(src1) * GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING) * sizeof(block_q8_1)/QK8_1;
            if (need <= GGML_CUDA_Q8_1_REUSE_BYTES) {
                return 0;
            }
            for (int i = 0; i < GGML_CUDA_MMQ_SCRATCH_KEY; i++) {
                key[i] = 0;
            }
            key[0] = -(int64_t) (intptr_t) dst;
            return need;
        }
        case GGML_CUDA_MM_CUBLAS: {
            // nothing to share: a key no other node has
            for (int i = 0; i < GGML_CUDA_MMQ_SCRATCH_KEY; i++) {
                key[i] = 0;
            }
            key[0] = -(int64_t) (intptr_t) dst;
            return ggml_cuda_cublas_scratch_get_layout(src0, src1, dst, ggml_cuda_cublas_compute_type(src0, src1, dst, cc), cc).size;
        }
        default:
            return 0;
    }
}

// The temporary memory of the matrix products (mmq's and mmvq's quantized activation, mmq's expert ids,
// cuBLAS's conversions), of the radix top-k and of flash attention comes from the compute buffer: the scheduler counts it in the reservation, where the pool
// kept the largest request of each size for good (~380 MiB at 8192-token ubatches). Consecutive mmq
// nodes that quantize the same activation the same way (q/k/v, gate and up, hc down and inject) share
// one scratch, filled by the first: as long as no node in between writes over that activation.
static void ggml_cuda_add_mmq_scratch(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
        ggml_backend_graph_optimize_params * params) {
    if (params->add_scratch == nullptr) {
        return;
    }
    const auto root = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };

    std::vector<ggml_tensor *> group;
    int64_t group_key[GGML_CUDA_MMQ_SCRATCH_KEY];
    size_t  group_size = 0;
    const ggml_tensor * group_src1 = nullptr;
    const auto flush = [&]() {
        if (!group.empty()) {
            params->add_scratch(params->user_data, group.data(), (int) group.size(), group_size);
            group.clear();
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        int64_t key[GGML_CUDA_MMQ_SCRATCH_KEY];
        size_t  need = 0;
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[GGML_SCHED_SCRATCH_SRC] == nullptr) {
            need = ggml_cuda_mul_mat_scratch_size(node, cuda_ctx->device, key);
        } else if (node->op == GGML_OP_FLASH_ATTN_EXT && node->src[GGML_SCHED_SCRATCH_SRC] == nullptr) {
            ggml_cuda_set_device(cuda_ctx->device);
            need = ggml_cuda_flash_attn_ext_scratch_size(*cuda_ctx, node);
            for (int k = 0; k < GGML_CUDA_MMQ_SCRATCH_KEY; k++) {
                key[k] = 0;
            }
            key[0] = -(int64_t) (intptr_t) node;
        } else if (node->op == GGML_OP_TOP_K && node->src[GGML_SCHED_SCRATCH_SRC] == nullptr) {
            need = ggml_cuda_top_k_scratch_size(node);
            for (int k = 0; k < GGML_CUDA_MMQ_SCRATCH_KEY; k++) {
                key[k] = 0;
            }
            key[0] = -(int64_t) (intptr_t) node;
        }
        if (need > 0) {
            if (!group.empty() && memcmp(key, group_key, sizeof(key)) == 0) {
                group.push_back(node);
                group_size = std::max(group_size, need);
                continue;
            }
            flush();
            group.push_back(node);
            memcpy(group_key, key, sizeof(key));
            group_size = need;
            group_src1 = node->src[1] ? root(node->src[1]) : nullptr;
            continue;
        }
        // a node that writes into the activation's memory (an op whose result is a view of it) ends the sharing
        if (!group.empty() && group_src1 && !ggml_cuda_is_view_or_noop(node) && root(node) == group_src1) {
            flush();
        }
    }
    flush();
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));

    // not with GGML_CUDA_GRAPH_OPT: it reorders the nodes onto concurrent streams below
    static const bool graph_opt = getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) == 1;
    static const bool no_mmq_scratch = getenv("GGML_CUDA_SCRATCH") != nullptr && atoi(getenv("GGML_CUDA_SCRATCH")) == 0;

    auto add_alloc_deps = [&](size_t start, size_t last_node) {

        for (size_t i = start; i < last_node; ++i) {
            params->add_alloc_dep(params->user_data, cgraph->nodes[i], cgraph->nodes[last_node]);

            for (int j = 0; j < GGML_MAX_SRC; ++j) {
                if (cgraph->nodes[i]->src[j]) {
                    params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[j], cgraph->nodes[last_node]);
                }
            }
        }
    };

    if (!disable_fusion) {
        // add alloc deps for performance positive fusions. This may increase the overall compute buffer size.
        // TODO: consolidate fusion paths in graph_optimize and graph_compute
        ggml_cuda_set_device(cuda_ctx->device);
        for (int i = 0; i + 5 < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID) {
                continue;
            }
            for (int j = i + 3; j + 2 < cgraph->n_nodes; ++j) {
                if (cgraph->nodes[j]->op == GGML_OP_MUL_MAT_ID && cgraph->nodes[j + 1]->op == GGML_OP_MUL_MAT_ID) {
                    break;
                }
                if (cgraph->nodes[j]->op != GGML_OP_MUL_MAT || !ggml_cuda_match_shared_expert(cgraph, i, j)) {
                    continue;
                }
                // Group both outputs before allocation so the shared result cannot alias intervening nodes.
                std::rotate(cgraph->nodes + i + 3, cgraph->nodes + j, cgraph->nodes + j + 3);
                ggml_tensor * up = cgraph->nodes[i + 2]->src[1];
                params->add_alloc_dep(params->user_data, up->src[1], cgraph->nodes[i + 5]);
                params->add_alloc_dep(params->user_data, up->src[2], cgraph->nodes[i + 5]);
                i += 5;
                break;
            }
        }
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // the l2 norm of q and k right after the cached conv (computed by the conv's kernel): the conv's inputs
            // (window, new tokens) live until the norm's last node, which is never placed over them
            if (cgraph->nodes[i]->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
                const ggml_tensor * v = cgraph->nodes[i]->src[0];
                const ggml_tensor * conv_out = v->view_src;
                const ggml_tensor * ssm = conv_out && conv_out->op == GGML_OP_UNARY ? conv_out->src[0] : conv_out;
                if (ssm && ssm->op == GGML_OP_SSM_CONV && cgraph->nodes[i + 1]->op == GGML_OP_SCALE &&
                        cgraph->nodes[i + 1]->src[0] == cgraph->nodes[i]) {
                    ggml_tensor * sc = cgraph->nodes[i + 1];
                    for (int a = 0; a < GGML_MAX_SRC; a++) {
                        ggml_tensor * in = ssm->src[a];
                        if (!in) {
                            continue;
                        }
                        params->add_alloc_dep(params->user_data, in, sc);
                        for (int b = 0; b < GGML_MAX_SRC; b++) {
                            if (in->src[b]) {
                                params->add_alloc_dep(params->user_data, in->src[b], sc);
                            }
                        }
                    }
                }
            }
            // the hyper-connection combine + norm (K1): its inputs live until its last node, so that its outputs are
            // never placed over them and its fusion never depends on the layout (an unrelated change of the layout
            // once left a layer's second boundary unfused: 6 kernels instead of 1)
            if (ggml_cuda_qwen4_hc_k1_at(cgraph, i)) {
                ggml_tensor * post = cgraph->nodes[i + 3];
                ggml_tensor * mul  = cgraph->nodes[i + 5];
                for (ggml_tensor * in : { cgraph->nodes[i]->src[0], post->src[0], post->src[1], mul->src[1] }) {
                    params->add_alloc_dep(params->user_data, in, mul);
                }
            }
            ggml_cuda_moe_weighted_reduction_match match;
            if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
                if (match.expert_scale != nullptr) {
                    params->add_alloc_dep(
                        params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
                }
                i += match.node_count - 1;
            }

            if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
                    cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
                std::vector<ggml_op>    ops;
                ops.reserve(13);  // max ops; avoids gcc -Wstringop-overflow false positive

                const ggml_tensor * node = cgraph->nodes[i];

                if (can_fuse) {
                    const ggml_tensor * logits  = node->src[0];
                    ggml_tensor *       weights = nullptr;
                    ggml_tensor *       ids     = nullptr;

                    if (!args.delayed_softmax) {
                        int out_nodes[2];  // nodes which can't be elided

                        if (args.sigmoid) {
                            ops.insert(ops.end(), { GGML_OP_UNARY });
                        } else if (args.sqrt_softplus) {
                            ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                        } else {
                            ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                        }
                        const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                        if (args.prob_bias) {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                                    GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 4;
                        } else {
                            ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                            out_nodes[0] = i_probs + 3;
                        }
                        ids = cgraph->nodes[out_nodes[0]];

                        if (args.norm) {
                            ops.insert(ops.end(),
                                       { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                        }
                        if (args.scale) {
                            ops.insert(ops.end(), { GGML_OP_SCALE });
                        }

                        weights      = cgraph->nodes[i + ops.size() - 1];
                        out_nodes[1] = i + ops.size() - 1;

                        if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                                ggml_cuda_should_use_topk_moe(node, logits, weights, ids)) {

                            add_alloc_deps(i, i + ops.size());
                            i += ops.size() - 1;
                        }
                    }
                }
            }
        }
    }

    if (!graph_opt && !no_mmq_scratch) {
        ggml_cuda_add_mmq_scratch(cuda_ctx, cgraph, params);
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    if (op->src[0]->type == GGML_TYPE_BF16 && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
                        return false;
                    }
                    return op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    if (op->src[0]->type == GGML_TYPE_BF16 &&
                            (ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_OAI || ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU_CLAMP)) {
                        return false;
                    }
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_cuda_op_mul_mat_use_fwht(op)) {
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT_ID && ggml_get_op_params_i32(op, 3) == GGML_PREC_F32) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_K:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_K:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if ((src0_type == GGML_TYPE_Q8_0 || src0_type == GGML_TYPE_Q4_0) && src1_type == GGML_TYPE_F16 &&
                        ggml_is_contiguous(op->src[1]) && op->src[0]->nb[0] == ggml_type_size(src0_type)) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_SCALE:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_BF16) && op->type == op->src[0]->type;
        case GGML_OP_CLAMP:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            if (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16 || op->type == GGML_TYPE_BF16) {
                return op->src[0]->type == GGML_TYPE_BF16 && op->type == GGML_TYPE_BF16 &&
                    (op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_F32);
            }
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 96 || d_state == 128 || d_state == 256) && d_head % 16 == 0)
                const int64_t d_state = op->src[0]->ne[0];
                return (d_state == 96 || d_state == 128 || d_state == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_3D:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op);
        case GGML_OP_CONV_2D_DW:
            return (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            {
                // bitonic path: the padded row must fit in shared memory
                int64_t ncols_pad = 1;
                while (ncols_pad < op->src[0]->ne[0]) {
                    ncols_pad *= 2;
                }
                return ncols_pad * sizeof(int) <= ggml_cuda_info().devices[dev_ctx->device].smpb;
            }
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return true;
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);
        case GGML_OP_QSA_SELECT:
            return true;

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    if (!ggml_cuda_moe_hosted_wait(nullptr, event->context)) {
        CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
    }
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// see ggml_backend_cuda_context::nt_weights
static void ggml_backend_cuda_set_weights_read_once(ggml_backend_t backend, bool read_once) {
    if (backend && ggml_backend_is_cuda(backend)) {
        ((ggml_backend_cuda_context *) backend->context)->nt_weights = read_once;
    }
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    if (strcmp(name, "ggml_backend_set_weights_read_once") == 0) {
        return (void *)ggml_backend_cuda_set_weights_read_once;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
#if !defined(GGML_USE_MUSA)
        ggml_moe_cache_register(&reg);
#endif
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
