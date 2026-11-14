#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>
#endif

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "base.hpp"
#include "mem.hpp"

const char * mem_api_int2str(int mem_api) {
    switch (mem_api) {
        case MEMORY_API_TYPE_SYCL:
            return "SYCL API";
        case MEMORY_API_TYPE_LEVEL_ZERO:
            return "Level Zero API";
        default:
            return "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API

/*
 * Sysman memory-module queries require zesInit() to have been called.
 * Initialise it once, in a thread-safe manner, before using the Sysman API.
 */
static bool ggml_sycl_init_level_zero_sysman() {
    static const ze_result_t init_result = zesInit(0);
    if (init_result != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG(
            "[SYCL] zesInit failed (Level Zero result: %d).\n",
            static_cast<int>(init_result));
        return false;
    }

    return true;
}

/*
 * Query aggregate memory-module capacity and free memory through Level Zero
 * Sysman. This is an aggregate across the device's reported memory modules;
 * it should not be interpreted as a guarantee that every byte is equally
 * usable by every allocation type.
 *
 * Returns true only when every enumerated module was queried successfully and
 * the resulting totals are internally consistent.
 */
static bool query_free_memory_by_ze(
    const sycl::device & dev,
    size_t & free_bytes,
    size_t & total_bytes) {

    GGML_SYCL_DEBUG(
        "[SYCL] call %s: querying free memory using Level Zero Sysman.\n",
        __func__);

    free_bytes = 0;
    total_bytes = 0;

    if (!ggml_sycl_init_level_zero_sysman()) {
        return false;
    }

#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend =
        sycl::backend::ext_oneapi_level_zero;
#else
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    try {
        if (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG(
                "[SYCL] Device backend is not Level Zero; "
                "cannot query its native Level Zero handle.\n");
            return false;
        }

        const ze_device_handle_t ze_dev =
            sycl::get_native<kL0Backend>(dev);

        if (ze_dev == nullptr) {
            GGML_SYCL_DEBUG(
                "[SYCL] Level Zero device handle is null.\n");
            return false;
        }

        uint32_t module_count = 0;
        ze_result_t result =
            zesDeviceEnumMemoryModules(ze_dev, &module_count, nullptr);

        if (result != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG(
                "[SYCL] Failed to enumerate Level Zero memory modules "
                "(result: %d, count: %u).\n",
                static_cast<int>(result), module_count);
            return false;
        }

        std::vector<zes_mem_handle_t> modules(module_count);

        result = zesDeviceEnumMemoryModules(
            ze_dev, &module_count, modules.data());

        if (result != ZE_RESULT_SUCCESS ||
            module_count == 0 ||
            module_count > modules.size()) {
            GGML_SYCL_DEBUG(
                "[SYCL] Failed to retrieve Level Zero memory modules "
                "(result: %d, count: %u).\n",
                static_cast<int>(result), module_count);
            return false;
        }

        bool all_modules_valid = true;

        for (uint32_t i = 0; i < module_count; ++i) {
            zes_mem_state_t state = {};
            state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext = nullptr;

            result = zesMemoryGetState(modules[i], &state);
            if (result != ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG(
                    "[SYCL] Failed to query memory module %u "
                    "(result: %d).\n",
                    i, static_cast<int>(result));
                all_modules_valid = false;
                continue;
            }

            free_bytes += static_cast<size_t>(state.free);
            total_bytes += static_cast<size_t>(state.size);
        }

        if (!all_modules_valid || total_bytes == 0 ||
            free_bytes > total_bytes) {
            GGML_SYCL_DEBUG(
                "[SYCL] Level Zero memory query returned incomplete "
                "or inconsistent results (free: %zu, total: %zu).\n",
                free_bytes, total_bytes);
            free_bytes = 0;
            total_bytes = 0;
            return false;
        }

        GGML_SYCL_DEBUG(
            "[SYCL] Level Zero memory: free=%zu MiB, total=%zu MiB.\n",
            free_bytes / (1024 * 1024),
            total_bytes / (1024 * 1024));

        return true;
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG(
            "[SYCL] Level Zero memory query failed: %s\n", e.what());
        free_bytes = 0;
        total_bytes = 0;
        return false;
    }
}

#endif // GGML_SYCL_SUPPORT_LEVEL_ZERO_API

/*
 * Query memory through the Intel SYCL free-memory extension.
 *
 * The total capacity is obtained from the standard SYCL device information
 * query. Free memory is available only when the device/runtime exposes the
 * Intel extension.
 */
static bool get_memory_size_by_sycl_api(
    const sycl::device & dev,
    size_t & free_bytes,
    size_t & total_bytes) {

    GGML_SYCL_DEBUG(
        "[SYCL] call %s: querying free memory using SYCL API.\n",
        __func__);

    free_bytes = 0;
    total_bytes = 0;

    try {
        total_bytes =
            dev.get_info<sycl::info::device::global_mem_size>();

#if defined(__SYCL_COMPILER_VERSION) && \
    __SYCL_COMPILER_VERSION >= 20221105
        if (dev.has(sycl::aspect::ext_intel_free_memory)) {
            try {
                free_bytes =
                    dev.get_info<sycl::ext::intel::info::device::free_memory>();

                if (total_bytes == 0 || free_bytes > total_bytes) {
                    GGML_SYCL_DEBUG(
                        "[SYCL] Free-memory extension returned "
                        "inconsistent values (free: %zu, total: %zu).\n",
                        free_bytes, total_bytes);
                    free_bytes = 0;
                    return false;
                }

                GGML_SYCL_DEBUG(
                    "[SYCL] SYCL memory: free=%zu MiB, total=%zu MiB.\n",
                    free_bytes / (1024 * 1024),
                    total_bytes / (1024 * 1024));

                return true;
            } catch (const sycl::exception & e) {
                GGML_SYCL_DEBUG(
                    "[SYCL] Failed to query Intel free-memory extension: %s\n",
                    e.what());
                free_bytes = 0;
                return false;
            }
        }

        GGML_SYCL_DEBUG(
            "[SYCL] Device does not support "
            "sycl::aspect::ext_intel_free_memory.\n");
#else
        GGML_SYCL_DEBUG(
            "[SYCL] Compiler version does not expose the Intel "
            "free-memory extension used by this implementation.\n");
#endif
    } catch (const sycl::exception & e) {
        GGML_SYCL_DEBUG(
            "[SYCL] Failed to query SYCL device memory information: %s\n",
            e.what());
        free_bytes = 0;
        total_bytes = 0;
        return false;
    }

    return false;
}

/*
 * Returns true only if free memory was successfully queried.
 *
 * If the query fails, total_bytes is populated when the standard SYCL device
 * capacity query succeeds, free_bytes is set to zero, and false is returned.
 * Callers must treat false as "free memory unknown", not as zero free memory
 * and certainly not as total memory being free.
 */
bool get_memory_size(
    const sycl::device & dev,
    size_t & free_bytes,
    size_t & total_bytes,
    MemoryAPIType api_type) {

    free_bytes = 0;
    total_bytes = 0;

    if (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        if (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            return true;
        }

        GGML_SYCL_DEBUG(
            "[SYCL] Level Zero memory query failed; falling back "
            "to the SYCL API.\n");
#else
        GGML_SYCL_DEBUG(
            "[SYCL] Level Zero API support is not compiled in; "
            "falling back to the SYCL API.\n");
#endif
    }

    if (get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)) {
        return true;
    }

    /*
     * Preserve total capacity when possible, but do not invent a free-memory
     * value. A caller that needs a memory budget should use a conservative
     * policy when this function returns false.
     */
    if (total_bytes == 0) {
        try {
            total_bytes =
                dev.get_info<sycl::info::device::global_mem_size>();
        } catch (const sycl::exception & e) {
            GGML_SYCL_DEBUG(
                "[SYCL] Unable to query total device memory: %s\n",
                e.what());
        }
    }

    free_bytes = 0;

    GGML_SYCL_DEBUG(
        "[SYCL] Unable to determine free device memory "
        "(total capacity: %zu bytes).\n",
        total_bytes);

    return false;
}
