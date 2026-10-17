// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-backend-moe-cache.h"

// Optional MoE expert cache function table, populated by a supporting backend.
struct ggml_moe_cache_api ggml_moe_cache = {};

void ggml_moe_cache_unregister(const void * owner) {
    if (ggml_moe_cache.owner == owner) {
        ggml_moe_cache = {};
    }
}

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

// shared planning logic for allocating a list of tensors into one or more buffers of the given type
struct ggml_backend_buft_alloc_buffer_n_plan_item {
    size_t size;  // total bytes for this buffer
    int    first; // first tensor index (inclusive)
    int    last;  // last tensor index (exclusive)
};

using ggml_backend_buft_alloc_buffer_n_plan_t = std::vector<ggml_backend_buft_alloc_buffer_n_plan_item>;

static ggml_backend_buft_alloc_buffer_n_plan_t ggml_backend_buft_alloc_buffer_n_plan(
        ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    ggml_backend_buft_alloc_buffer_n_plan_t plan;

    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    const size_t max_size  = ggml_backend_buft_get_max_size(buft);

    size_t cur_buf_size = 0;
    int    first        = 0;

    for (int i = 0; i < n_tensors; i++) {
        size_t this_size = 0;
        struct ggml_tensor * t = tensors[i];
        if (t->data == NULL && t->view_src == NULL) {
            this_size = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), alignment);
        }

        // flush the current buffer if adding this tensor would exceed max_size
        if (cur_buf_size > 0 && (cur_buf_size + this_size) > max_size) {
            plan.push_back({ cur_buf_size, first, i });
            cur_buf_size = this_size;
            first        = i;
        } else {
            cur_buf_size += this_size;
        }
    }

    if (cur_buf_size > 0) {
        plan.push_back({ cur_buf_size, first, n_tensors });
    }

    return plan;
}

// default implementation of alloc_buffer_n
// allocates tensors from a list into one or more buffers of the given type
static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    std::vector<ggml_backend_buffer_t> buffers;
    buffers.reserve(plan.size());

    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, item.size);
        if (buffer == NULL) {
            GGML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), item.size);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            return NULL;
        }

        struct ggml_tallocr tallocr = ggml_tallocr_new(buffer);

        // allocate tensors in the current buffer
        struct ggml_tensor * t_failed = NULL;
        for (int j = item.first; j < item.last; j++) {
            struct ggml_tensor * t = tensors[j];
            if (t->data == NULL) {
                if (t->view_src == NULL) {
                    if (ggml_tallocr_alloc(&tallocr, t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                } else if (t->buffer == NULL) {
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            } else {
                if (t->view_src != NULL && t->buffer == NULL) {
                    // view of a pre-allocated tensor
                    if (ggml_backend_view_init(t) != GGML_STATUS_SUCCESS) {
                        t_failed = t;
                        break;
                    }
                }
            }
        }
        if (t_failed != NULL) {
            GGML_LOG_ERROR("%s: failed to initialize tensor %s\n", __func__, t_failed->name);
            for (ggml_backend_buffer_t b : buffers) {
                ggml_backend_buffer_free(b);
            }
            ggml_backend_buffer_free(buffer);
            return NULL;
        }

        buffers.push_back(buffer);
    }

    if (buffers.empty()) {
        return NULL;
    }

    if (buffers.size() == 1) {
        return buffers[0];
    }

    return ggml_backend_multi_buffer_alloc_buffer(buffers.data(), buffers.size());
}

// default implementation of get_alloc_size_n
// returns the total size that alloc_buffer_n_default would allocate for the given tensors
static size_t ggml_backend_buft_get_alloc_size_n_default(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    const ggml_backend_buft_alloc_buffer_n_plan_t plan = ggml_backend_buft_alloc_buffer_n_plan(buft, tensors, n_tensors);

    size_t total = 0;
    for (const ggml_backend_buft_alloc_buffer_n_plan_item & item : plan) {
        total += item.size;
    }
    return total;
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.alloc_buffer_n) {
        return buft->iface.alloc_buffer_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_alloc_buffer_n_default(buft, tensors, n_tensors);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));

        // [TAG_ALLOC_SIZE_EXPAND]
        // if you hit this assert, update ggml_backend_op_alloc_size_may_expand() accordingly
        GGML_ASSERT(size <= ggml_nbytes(tensor) ||
                    ggml_op_is_empty(tensor->op) ||
                    ggml_is_quantized(tensor->type) || // [TAG_ALLOC_SIZE_EXPAND]
                    ggml_op_alloc_size_may_expand(tensor->op));

        return size;
    }
    return ggml_nbytes(tensor);
}

size_t ggml_backend_buft_get_alloc_size_n(ggml_backend_buffer_type_t buft, struct ggml_tensor ** tensors, int n_tensors) {
    GGML_ASSERT(buft);
    if (buft->iface.get_alloc_size_n) {
        return buft->iface.get_alloc_size_n(buft, tensors, n_tensors);
    }
    return ggml_backend_buft_get_alloc_size_n_default(buft, tensors, n_tensors);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

static void ggml_backend_moe_cache_invalidate_buffer(
        ggml_backend_buffer_t buffer, const void * address, size_t size) {
    if (!ggml_moe_cache.invalidate || !buffer || !address || size == 0 ||
        buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        !ggml_backend_buffer_is_host(buffer)) {
        return;
    }
    ggml_moe_cache.invalidate(address, size);
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    // Host weight buffers can back queued cache-fill jobs.
    if (buffer->iface.get_base) {
        void * base = ggml_backend_buffer_get_base(buffer);
        if (base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, base, ggml_backend_buffer_get_size(buffer));
        }
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    if (buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    if (usage != buffer->usage && buffer->iface.get_base) {
        ggml_backend_moe_cache_invalidate_buffer(
                buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
    }
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    } else if (ggml_backend_buffer_is_meta(buffer)) {
        ggml_backend_meta_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        if (buffer->iface.get_base) {
            ggml_backend_moe_cache_invalidate_buffer(
                    buffer, ggml_backend_buffer_get_base(buffer), buffer->size);
        }
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                buf, (const char *) tensor->data + offset, size);
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.get_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset,
            (n_copies - 1)*stride_tensor + size);
    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.get_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    ggml_backend_moe_cache_invalidate_buffer(
            buf, (const char *) tensor->data + offset, size);
    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (ggml_backend_buffer_is_host(src_buf)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst_buf)) {
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_backend_moe_cache_invalidate_buffer(
                dst_buf, dst->data, ggml_nbytes(dst));
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph, struct ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph, params);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

// the step from one expert to the next in the copy of an expert tensor of a packed layer (its experts an entry apart
// in host memory, nb[2] above its bytes): the expert's bytes and what MMQ reads past its last row (the MoE cache's
// moe_cache_role_pad before its rounding to 256 bytes); 0 for any other tensor. A packed entry stores that pad (the
// next expert's first bytes) right after the expert's bytes: a run of experts is one 2D copy, nothing past it to copy.
static size_t ggml_sched_packed_expert_step(const struct ggml_tensor * t) {
    if (t->ne[2] <= 1 || t->ne[3] != 1 || !ggml_is_quantized(t->type) || t->nb[1] != ggml_row_size(t->type, t->ne[0]) ||
            t->nb[2] <= t->nb[1]*t->ne[1] || !t->buffer || t->view_src ||
            ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return 0;
    }
    // whole blocks (MMQ counts the step in blocks), within the entry's pad (that, rounded up to 256 bytes)
    const size_t over = (size_t) ((GGML_PAD(t->ne[0], 512) - t->ne[0]) / ggml_blck_size(t->type)) * ggml_type_size(t->type);
    const size_t step = t->nb[1]*t->ne[1] + over;
    GGML_ASSERT(step <= t->nb[2]);
    return step;
}

// experts e0 .. e1 - 1 of an expert tensor of a packed layer into its copy (at their step), each with its pad, one 2D
// copy; the last expert of the tensor without its pad (past ggml_nbytes: the copy's padding, zeroed by the caller)
static void ggml_sched_copy_packed_experts(ggml_backend_t backend, struct ggml_tensor * dst, const struct ggml_tensor * src,
        int64_t e0, int64_t e1, size_t step) {
    const int64_t n_expert = src->ne[2];
    const int64_t full = e1 == n_expert ? e1 - 1 : e1;   // the rows copied with their pad
    if (full > e0) {
        ggml_backend_tensor_set_2d_async(backend, dst, (const uint8_t *) src->data + e0*src->nb[2], e0*step, step,
            full - e0, step, src->nb[2]);
    }
    if (full < e1) {
        ggml_backend_tensor_set_async(backend, dst, (const uint8_t *) src->data + full*src->nb[2], full*step,
            src->nb[1]*src->ne[1]);
    }
}

// a copy of the tensor with the same memory layout; for an expert tensor of a packed layer, its experts at their
// step (ggml_sched_packed_expert_step) instead of an entry apart
static struct ggml_tensor * ggml_dup_tensor_copy_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor_layout(ctx, tensor);
    const size_t step = ggml_sched_packed_expert_step(tensor);
    if (step) {
        dup->nb[2] = step;
        dup->nb[3] = step*dup->ne[2];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

#ifndef GGML_SCHED_MAX_PREFETCH_SLOTS
#define GGML_SCHED_MAX_PREFETCH_SLOTS 8
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor ** inputs;
    int n_inputs;
    int inputs_capacity;
    // graph view of this split
    struct ggml_cgraph graph;
};

// a graph's split and allocation kept for later runs (ggml_backend_sched_plan_*): the scheduler's per-graph state,
// swapped with it
struct ggml_backend_sched_plan {
    const struct ggml_cgraph * key = nullptr;   // null: free
    int      key_n_nodes = 0;
    uint64_t gen         = 0;                   // the allocator's generation when it was allocated
    uint64_t used        = 0;
    bool     is_alloc    = false;
    bool     is_reset    = false;
    bool     arrays      = false;               // the arrays below exist

    struct ggml_hash_set  hash_set = {};
    int                 * hv_tensor_backend_ids = nullptr;
    struct ggml_tensor ** hv_tensor_copies      = nullptr;
    int * node_backend_ids      = nullptr;
    int * leaf_backend_ids      = nullptr;
    int * prev_node_backend_ids = nullptr;
    int * prev_leaf_backend_ids = nullptr;
    struct ggml_cgraph graph = {};
    struct ggml_backend_sched_split * splits = nullptr;
    int n_splits        = 0;
    int splits_capacity = 0;
    struct ggml_tensor ** graph_inputs = nullptr;
    int n_graph_inputs        = 0;
    int graph_inputs_capacity = 0;
    struct ggml_context * ctx = nullptr;
    char * context_buffer     = nullptr;
    std::unordered_set<const struct ggml_tensor *> * moe_hosted = nullptr;
};

#define GGML_SCHED_MAX_PLANS 8

static void ggml_backend_sched_plan_free(struct ggml_backend_sched_plan * p);

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    // the buffer types given at creation; bufts replaces them with borrowed ones while a
    // MoE cache session lends the compute buffers
    ggml_backend_buffer_type_t bufts_plain[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    // recorded behind the user inputs copied asynchronously to a backend (created at first use; null if the backend
    // has no events), see ggml_backend_sched_compute_splits
    ggml_backend_event_t user_inputs_done[GGML_SCHED_MAX_BACKENDS];
    bool                 user_inputs_done_tried[GGML_SCHED_MAX_BACKENDS];
    struct ggml_tensor ** graph_inputs;
    int n_graph_inputs;
    int graph_inputs_capacity;

    struct ggml_context * ctx;
    void * moe_cache_session;
    // the session belongs to another scheduler (ggml_backend_sched_share_moe_cache)
    bool moe_cache_guest;
    // MUL_MAT_ID nodes of the MoE blocks left on the GPU with their weights in host memory
    // (unless GGML_MOE_GPU=0), found again by every split_graph
    std::unordered_set<const struct ggml_tensor *> * moe_hosted;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    // full-tensor prefetch of offloaded MUL_MAT_ID weights (GGML_SCHED_PREFETCH_EXPERTS)
    // with a large batch virtually every expert is used, so the routing ids are not worth
    // waiting for; uploads run through a second backend instance on the same device so
    // they overlap compute, alternating between two staging slots
    bool prefetch_experts;
    ggml_backend_t prefetch_backend;
    int prefetch_n_slots;
    ggml_backend_buffer_t prefetch_slots[GGML_SCHED_MAX_PREFETCH_SLOTS];
    ggml_backend_event_t prefetch_ready[GGML_SCHED_MAX_PREFETCH_SLOTS];
    ggml_backend_event_t prefetch_free[GGML_SCHED_MAX_PREFETCH_SLOTS];
    bool prefetch_used[GGML_SCHED_MAX_PREFETCH_SLOTS];
    int prefetch_cur;
    // Slot addresses. With an active MoE cache the slots are zones lent by the cache
    // for the duration of a prefill graph; otherwise they are prefetch_slots buffers.
    void * prefetch_slot_data[GGML_SCHED_MAX_PREFETCH_SLOTS];
    bool prefetch_lent;
    ggml_backend_t prefetch_lend_backend;
    // experts the cache holds are read in place (ggml_moe_cache.prefill_residents): a slot ends with the table of the
    // expert addresses its tensor's MMQ reads, copied from prefetch_table; the entries stay pinned until prefill_unpin
    uint64_t * prefetch_table[GGML_SCHED_MAX_PREFETCH_SLOTS];
    bool prefill_pinned;
    // the MoE cache has been created from the first graph (see ggml_backend_sched_prime_moe_cache)
    bool moe_cache_primed;

    // the graph size the per-graph arrays were made for, and the kept plans (ggml_backend_sched_plan_*); the current
    // state's key: the graph it was allocated for when plan_set marked it
    size_t graph_size_max;
    std::vector<ggml_backend_sched_plan *> * plans;
    const struct ggml_cgraph * plan_key;
    int      plan_key_n_nodes;
    uint64_t plan_gen;
    uint64_t plan_clock;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static void ggml_backend_sched_split_inputs_grow(struct ggml_backend_sched_split * split) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (split->inputs_capacity > 0) {
        new_cap = 2*split->inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing split inputs capacity from %d to %d\n", __func__, split->inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) split->inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow split inputs container");
    }
    split->inputs = pnew;
    split->inputs_capacity = new_cap;
}

static void ggml_backend_sched_graph_inputs_grow(ggml_backend_sched_t sched) {
    int new_cap = GGML_SCHED_MAX_SPLIT_INPUTS;
    if (sched->graph_inputs_capacity > 0) {
        new_cap = 2*sched->graph_inputs_capacity;
        GGML_LOG_DEBUG("%s: increasing graph inputs capacity from %d to %d\n", __func__, sched->graph_inputs_capacity, new_cap);
    }
    auto * pnew = (struct ggml_tensor **) realloc((void *) sched->graph_inputs, new_cap * sizeof(struct ggml_tensor *));
    if (pnew == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, new_cap * sizeof(struct ggml_tensor *));
        GGML_ABORT("failed to grow graph inputs container");
    }
    sched->graph_inputs = pnew;
    sched->graph_inputs_capacity = new_cap;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static bool ggml_backend_sched_is_hosted_moe(ggml_backend_sched_t sched, const struct ggml_tensor * node);

static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // a MoE block the GPU hosts: on the first GPU backend (the cache's device)
    if (ggml_backend_sched_is_hosted_moe(sched, tensor)) {
        for (int b = 0; b < sched->n_backends - 1; b++) {
            if (ggml_backend_dev_type(ggml_backend_get_device(sched->backends[b])) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                ggml_backend_supports_op(sched->backends[b], tensor)) {
                SET_CAUSE(tensor, "1.moe");
                return b;
            }
        }
    }

    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    // TODO: there are exceptions (see below) - not an ideal solution
    bool allow = true;

    // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
    allow = allow && tensor->op != GGML_OP_ROPE;

    // skip FLASH_ATTN_EXT since the sinks tensor is too small to choose a based based on it
    allow = allow && tensor->op != GGML_OP_FLASH_ATTN_EXT;

    if (allow) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            const struct ggml_tensor * src = tensor->src[i];
            if (src == NULL) {
                continue;
            }
            if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
                // check if a backend with higher prio wants to offload the op
                if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                    for (int b = 0; b < src_backend_id; b++) {
                        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                            SET_CAUSE(tensor, "1.off");
                            return b;
                        }
                    }
                }
                SET_CAUSE(tensor, "1.wgt%d", i);
                return src_backend_id;
            }
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_desc(node), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static bool ggml_backend_sched_is_hosted_moe(ggml_backend_sched_t sched, const struct ggml_tensor * node) {
    return sched->moe_hosted && !sched->moe_hosted->empty() && sched->moe_hosted->count(node) > 0;
}

// With a MoE cache session (GGML_MOE_GPU=0 turns it off), a MoE block gate / up / swiglu / down whose weights are
// in host memory stays on the GPU for batches of up to 64 routed rows: the GPU computes the rows
// the cache holds, a host thread the others (ggml_cuda_moe_hosted_block); no weight is copied.
static void ggml_backend_sched_find_hosted_moe(ggml_backend_sched_t sched, const struct ggml_cgraph * graph) {
    static const bool enabled = !getenv("GGML_MOE_GPU") || atoi(getenv("GGML_MOE_GPU")) != 0;
    if (sched->moe_hosted) {
        sched->moe_hosted->clear();
    }
    if (!enabled || !sched->moe_cache_session || !ggml_moe_cache.cpu_experts) {
        return;
    }
    if (!sched->moe_hosted) {
        sched->moe_hosted = new std::unordered_set<const struct ggml_tensor *>();
    }
    const auto host_weights = [](const struct ggml_tensor * t) {
        return t->op == GGML_OP_MUL_MAT_ID && t->src[0]->buffer && ggml_backend_buffer_is_host(t->src[0]->buffer) &&
               t->src[0]->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    };
    for (int i = 0; i + 3 < graph->n_nodes; i++) {
        const struct ggml_tensor * glu  = graph->nodes[i + 2];
        const struct ggml_tensor * down = graph->nodes[i + 3];
        if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || down->src[1] != glu) {
            continue;
        }
        const struct ggml_tensor * gate = glu->src[0];
        const struct ggml_tensor * up   = glu->src[1];
        const struct ggml_tensor * ids  = down->src[2];
        if (!((gate == graph->nodes[i] && up == graph->nodes[i + 1]) || (gate == graph->nodes[i + 1] && up == graph->nodes[i])) ||
            !host_weights(gate) || !host_weights(up) || !host_weights(down) ||
            gate->src[2] != ids || up->src[2] != ids || gate->src[1] != up->src[1] ||
            ids->ne[0] * ids->ne[1] > 64) {
            continue;
        }
        sched->moe_hosted->insert(gate);
        sched->moe_hosted->insert(up);
        sched->moe_hosted->insert(down);
        i += 3;
    }
}

void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    ggml_backend_sched_find_hosted_moe(sched, graph);

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // scratch tensors set by a previous split of this graph lived in the context just reset (add_scratch)
    for (int i = 0; i < graph->n_nodes; i++) {
        const char * src = (const char *) graph->nodes[i]->src[GGML_SCHED_SCRATCH_SRC];
        if (src >= (const char *) sched->context_buffer && src < (const char *) sched->context_buffer + sched->context_buffer_size) {
            graph->nodes[i]->src[GGML_SCHED_SCRATCH_SRC] = nullptr;
        }
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                            !(j == 0 && ggml_backend_sched_is_hosted_moe(sched, node))) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    int old_cap = sched->splits_capacity;
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                    for (int k = old_cap; k < sched->splits_capacity; k++) {
                        memset(&sched->splits[k], 0, sizeof(struct ggml_backend_sched_split));
                    }
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                // the weights of a hosted MoE block stay where they are, the block reads them there
                if (j == 0 && ggml_backend_sched_is_hosted_moe(sched, node)) {
                    continue;
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_copy_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= split->inputs_capacity) {
                            ggml_backend_sched_split_inputs_grow(split);
                        }
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // pass 6: collect all input tensors into graph_inputs
    //         this includes inputs not consumed by any node (e.g. the embeddings input of a text-only batch) so that
    //         the graph composition does not depend on which inputs are used, which would otherwise cause graph
    //         reallocations when switching between different types of batches [GGML_SCHED_DEBUG_REALLOC]
    if (sched->n_copies > 1) {
        for (int i = 0; i < graph->n_leafs; i++) {
            struct ggml_tensor * leaf = graph->leafs[i];
            if ((leaf->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
                continue;
            }

            const size_t leaf_id = hash_id(leaf);
            const int leaf_backend_id = tensor_backend_id(leaf);
            GGML_ASSERT(leaf_backend_id != -1); // all leafs should be assigned by now

            if (tensor_id_copy(leaf_id, leaf_backend_id, 0) == NULL) {
                ggml_backend_t backend = sched->backends[leaf_backend_id];
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * tensor_copy;
                    if (c == sched->cur_copy) {
                        tensor_copy = leaf; // use the original tensor as the current copy
                    } else {
                        tensor_copy = ggml_dup_tensor_copy_layout(sched->ctx, leaf);
                        ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), leaf->name, c);
                    }
                    ggml_set_input(tensor_copy);
                    ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                    tensor_id_copy(leaf_id, leaf_backend_id, c) = tensor_copy;
                    SET_CAUSE(tensor_copy, "6.cpy");
                }
            }

            int n_graph_inputs = sched->n_graph_inputs++;
            if (n_graph_inputs >= sched->graph_inputs_capacity) {
                ggml_backend_sched_graph_inputs_grow(sched);
            }
            sched->graph_inputs[n_graph_inputs] = leaf;
        }
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    // optimize the split graphs and collect the allocation dependencies added by the backends
    // this needs to happen before we make graph_copy, so they are in sync
    // TODO: this may create many small allocations in the scheduler, restructure to use a flat array
    std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> alloc_deps;

    struct opt_user_data {
        std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>> * alloc_deps;
        ggml_backend_sched_t sched;
    } opt_ud = { &alloc_deps, sched };

    struct ggml_backend_graph_optimize_params opt_params = {
        /* .add_alloc_dep = */ [](void * user_data, ggml_tensor * tensor, ggml_tensor * until) {
            auto & deps = *((opt_user_data *) user_data)->alloc_deps;
            std::vector<ggml_tensor *> & keep = deps[until];
            if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) {
                keep.push_back(tensor);
            }
        },
        /* .user_data     = */ &opt_ud,
        /* .add_scratch   = */ [](void * user_data, ggml_tensor ** nodes, int n_nodes, size_t size) {
            ggml_backend_sched_t sched = ((opt_user_data *) user_data)->sched;
            // a node source, not a node: ggml-alloc allocates it before its first reader and frees it after
            // its last, and the split inputs are already known, so it is never copied between backends
            ggml_tensor * scratch = ggml_new_tensor_1d(sched->ctx, GGML_TYPE_I8, (int64_t) size);
            ggml_format_name(scratch, "scratch (%s)", nodes[0]->name);
            for (int i = 0; i < n_nodes; i++) {
                GGML_ASSERT(nodes[i]->src[GGML_SCHED_SCRATCH_SRC] == nullptr);
                nodes[i]->src[GGML_SCHED_SCRATCH_SRC] = scratch;
            }
        },
        /* .continues     = */ false,
    };

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        opt_params.continues = i > 0 && sched->splits[i - 1].backend_id == split->backend_id &&
            sched->splits[i - 1].i_end == split->i_start;
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph, &opt_params);
    }

    // each dep is added to graph_copy as a GGML_OP_NONE node with the kept tensors as srcs
    int n_dep_nodes = 0;
    for (const auto & it : alloc_deps) {
        n_dep_nodes += (it.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }

    int total_inputs = sched->n_graph_inputs;
    for (int i = 0; i < sched->n_splits; i++) {
        total_inputs += sched->splits[i].n_inputs;
    }
    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + total_inputs * 2 * sched->n_copies + n_dep_nodes;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    int n_dep_nodes_added = 0;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];

            if (alloc_deps.empty()) {
                continue;
            }

            // add a dependency node so that the kept tensors are not freed before this node is computed
            auto it = alloc_deps.find(graph->nodes[j]);
            if (it != alloc_deps.end()) {
                const std::vector<ggml_tensor *> & keep = it->second;
                for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
                    struct ggml_tensor * dep = ggml_view_tensor(sched->ctx, keep[k]);
                    for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); s++) {
                        dep->src[s] = keep[k + s];
                    }
                    assert(graph_copy->size > graph_copy->n_nodes);
                    sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
                    graph_copy->nodes[graph_copy->n_nodes++] = dep;
                    n_dep_nodes_added++;
                }
            }
        }
    }

    // a mismatch means a backend added a dep with an `until` tensor that is not a node of the optimized graph
    GGML_ASSERT(n_dep_nodes_added == n_dep_nodes);

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        // this graph runs next: a buffer borrowed from the cache may shrink to what it needs
        ggml_gallocr_set_shrink(sched->galloc, true);
        const bool reserved = ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        ggml_gallocr_set_shrink(sched->galloc, false);
        if (!reserved) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

// Give the zones lent by the MoE cache back, once the kernels reading them are done.
// bytes at the end of a prefetch slot for its table of expert addresses (up to LLAMA_MAX_EXPERTS), placed after the
// tensor (ggml_backend_sched_prefetch_table_offset)
#define GGML_SCHED_PREFETCH_TABLE_EXPERTS 1024
#define GGML_SCHED_PREFETCH_TABLE_BYTES (GGML_SCHED_PREFETCH_TABLE_EXPERTS*sizeof(uint64_t) + 256)
static size_t ggml_backend_sched_prefetch_table_offset(size_t tensor_alloc_size) {
    return GGML_PAD(tensor_alloc_size, 256);
}

// the entries the last prefill graph read in place go back to the cache once that graph is done with them (each
// prefetched split records prefetch_free behind its kernels); before anything that may lend, borrow or move entries
static void ggml_backend_sched_prefill_unpin(ggml_backend_sched_t sched) {
    if (!sched->prefill_pinned) {
        return;
    }
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        if (sched->prefetch_used[i]) {
            ggml_backend_event_synchronize(sched->prefetch_free[i]);
        }
    }
    ggml_moe_cache.prefill_unpin(sched->moe_cache_session);
    sched->prefill_pinned = false;
}

static void ggml_backend_sched_prefetch_restore(ggml_backend_sched_t sched) {
    ggml_backend_sched_prefill_unpin(sched);
    if (!sched->prefetch_lent) {
        return;
    }
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        if (sched->prefetch_used[i]) {
            ggml_backend_event_synchronize(sched->prefetch_free[i]);
            sched->prefetch_used[i] = false;
        }
        sched->prefetch_slot_data[i] = NULL;
    }
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(sched->prefetch_backend);
    }
    if (ggml_moe_cache.restore) {
        ggml_moe_cache.restore(sched->moe_cache_session, sched->prefetch_lend_backend);
    }
    sched->prefetch_lent = false;
}

static void ggml_backend_sched_prefetch_disable(ggml_backend_sched_t sched, ggml_backend_t split_backend) {
    sched->prefetch_experts = false;
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(split_backend);
        ggml_backend_synchronize(sched->prefetch_backend);
    }
    ggml_backend_sched_prefetch_restore(sched);
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        ggml_backend_buffer_free(sched->prefetch_slots[i]);
        sched->prefetch_slots[i] = NULL;
        sched->prefetch_slot_data[i] = NULL;
        sched->prefetch_used[i] = false;
    }
}

// slots are sized once for the largest offloaded expert tensor in the current graph so
// that they never need to grow mid-eval
// backends such as CUDA/HIP pad the last row of quantized tensors (MATRIX_ROW_PADDING)
// and their kernels read into that padding, so a slot must hold the padded size, not
// ggml_nbytes, or the kernel reads past the end of the slot
// the size of a weight's copy in a buffer of buft (an expert tensor of a packed layer: its experts at their step)
static size_t ggml_backend_sched_input_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * input) {
    const size_t step = ggml_sched_packed_expert_step(input);
    if (!step) {
        return ggml_backend_buft_get_alloc_size(buft, input);
    }
    ggml_tensor t = *input;
    t.nb[2] = step;
    t.nb[3] = step*t.ne[2];
    t.view_src = nullptr;
    return ggml_backend_buft_get_alloc_size(buft, &t);
}

static size_t ggml_backend_sched_prefetch_max_size(ggml_backend_sched_t sched, ggml_backend_buffer_type_t buft) {
    size_t max_size = 0;
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &sched->splits[split_id];
        if (split->graph.n_nodes == 0 || split->graph.nodes[0]->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            const ggml_tensor * input = split->inputs[input_id];
            if (input->buffer &&
                ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                ggml_backend_buffer_is_host(input->buffer)) {
                max_size = std::max(max_size, ggml_backend_sched_input_alloc_size(buft, input));
            }
        }
    }
    return max_size;
}

// the prefetch's own backend (its stream) and events, on the device of the split it serves
static bool ggml_backend_sched_prefetch_backend_init(ggml_backend_sched_t sched, ggml_backend_dev_t dev) {
    if (sched->prefetch_backend != NULL) {
        return true;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.async || !props.caps.events) {
        sched->prefetch_experts = false;
        return false;
    }
    sched->prefetch_backend = ggml_backend_dev_init(dev, NULL);
    if (sched->prefetch_backend == NULL) {
        sched->prefetch_experts = false;
        return false;
    }
    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        sched->prefetch_ready[i] = ggml_backend_event_new(dev);
        sched->prefetch_free[i]  = ggml_backend_event_new(dev);
        if (sched->prefetch_ready[i] == NULL || sched->prefetch_free[i] == NULL) {
            sched->prefetch_experts = false;
            return false;
        }
    }
    return true;
}

static bool ggml_backend_sched_prefetch_init(ggml_backend_sched_t sched, ggml_backend_t split_backend, size_t size) {
    if (!ggml_backend_sched_prefetch_backend_init(sched, split_backend->device)) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(split_backend);
    size = std::max(size, ggml_backend_sched_prefetch_max_size(sched, buft));
    if (sched->moe_cache_session && ggml_moe_cache.prefill_residents) {
        size = ggml_backend_sched_prefetch_table_offset(size) + GGML_SCHED_PREFETCH_TABLE_BYTES;
    }

    // With an active MoE cache, borrow the slots from it: its memory idles during
    // prefill. The cache is created before the first graph runs (see
    // ggml_backend_sched_prime_moe_cache); if it still cannot lend (dormant, disabled,
    // pools too small for a zone), fall back to slots of our own below.
    if (sched->moe_cache_session && ggml_moe_cache.lend && ggml_moe_cache.set_lend_size) {
        if (sched->prefetch_lent) {
            return true;
        }
        ggml_moe_cache.set_lend_size(sched->moe_cache_session, size, sched->prefetch_n_slots);
        void * zones[GGML_SCHED_MAX_PREFETCH_SLOTS] = {};
        const int n_lent = ggml_moe_cache.lend(sched->moe_cache_session, split_backend, size,
                zones, sched->prefetch_n_slots);
        if (n_lent > 0) {
            if (n_lent < sched->prefetch_n_slots) {
                sched->prefetch_n_slots = n_lent;   // one zone per pool: fewer pools than slots
                sched->prefetch_cur = 0;
            }
            for (int i = 0; i < sched->prefetch_n_slots; i++) {
                sched->prefetch_slot_data[i] = zones[i];
            }
            sched->prefetch_lent = true;
            sched->prefetch_lend_backend = split_backend;
            return true;
        }
        // no zone to borrow: run this graph without prefetch, keep it enabled; slots of our
        // own would be sized into the cache's reserve and never let it lend again
        return false;
    }

    for (int i = 0; i < sched->prefetch_n_slots; i++) {
        if (sched->prefetch_slots[i] == NULL || ggml_backend_buffer_get_size(sched->prefetch_slots[i]) < size) {
            // allocate before freeing so a failure leaves the old slot intact
            ggml_backend_buffer_t new_buf = ggml_backend_buft_alloc_buffer(buft, size);
            if (new_buf == NULL) {
                // overlap needs at least 2 slots, otherwise run with what fits
                if (i >= 2 && sched->prefetch_slots[0] != NULL &&
                    ggml_backend_buffer_get_size(sched->prefetch_slots[0]) >= size) {
                    sched->prefetch_n_slots = i;
                    sched->prefetch_cur = 0;
                    return true;
                }
                ggml_backend_sched_prefetch_disable(sched, split_backend);
                return false;
            }
            if (sched->prefetch_slots[i] != NULL) {
                ggml_backend_synchronize(split_backend);
                ggml_backend_synchronize(sched->prefetch_backend);
                ggml_backend_buffer_free(sched->prefetch_slots[i]);
            }
            // zero once so the row padding never holds garbage (NaN scales)
            ggml_backend_buffer_clear(new_buf, 0);
            sched->prefetch_slots[i] = new_buf;
            sched->prefetch_slot_data[i] = ggml_backend_buffer_get_base(new_buf);
            sched->prefetch_used[i] = false;
        }
    }
    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    struct moe_cache_scope {
        void * session;
        void (*leave)(void *);

        moe_cache_scope(void * session, bool guest)
            : session(session), leave(ggml_moe_cache.session_leave) {
            auto enter = guest ? ggml_moe_cache.session_enter_guest : ggml_moe_cache.session_enter;
            if (enter && leave) {
                enter(session);
            } else {
                leave = nullptr;
            }
        }

        ~moe_cache_scope() {
            if (leave) {
                leave(session);
            }
        }
    } cache_scope(sched->moe_cache_session, sched->moe_cache_guest);

    // Zones lent for the previous graph go back to the cache. A prefill graph borrows
    // them again at its first prefetched split; between two prefill graphs that is
    // almost free, since the cache did not touch them in between.
    ggml_backend_sched_prefetch_restore(sched);

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    int prev_backend_id = -1;

    // backends with user inputs copied asynchronously and not yet waited for: their host source must stay intact until
    // the copies have run. Only a CPU split (its nodes, or its input copies into host memory, may reuse that memory)
    // or the caller (which rewrites the inputs of the next graph) could change it: an event recorded behind the copies
    // is waited for before a CPU split and at the end, so that the GPU split is launched right behind its copies
    // instead of after them (a backend without events synchronizes before its launch, as before)
    bool user_inputs_pending[GGML_SCHED_MAX_BACKENDS] = { false };
    auto wait_user_inputs = [&]() {
        for (int b = 0; b < sched->n_backends; b++) {
            if (user_inputs_pending[b]) {
                ggml_backend_event_synchronize(sched->user_inputs_done[b]);
                user_inputs_pending[b] = false;
            }
        }
    };

    // DIAGNOSTIC (GGML_SCHED_TRACE_INPUTS=N): log every input copy of graphs N to N+29 (all schedulers counted together;
    // each line names its scheduler and its number of splits)
    static const int trace_graph = getenv("GGML_SCHED_TRACE_INPUTS") ? atoi(getenv("GGML_SCHED_TRACE_INPUTS")) : -1;
    static int trace_counter = 0;
    const int trace_c = trace_counter++;
    const bool trace_on = trace_graph >= 0 && trace_c >= trace_graph && trace_c < trace_graph + 30;
    struct trace_input_log {
        bool on; int split; const char * dst; const ggml_tensor * t; const char * src; int64_t t0; const void * sp = nullptr; int ns = 0;
        ~trace_input_log() {
            if (on) {
                GGML_LOG_INFO("trace input %p/%d: split %3d %-6s <- %-6s %-40s %8zu B %s %6lld us\n", sp, ns, split, dst, src, t->name,
                        ggml_nbytes(t), ggml_is_contiguous(t) ? "cont" : "NONCONT", (long long) (ggml_time_us() - t0));
            }
        }
    };
    // DIAGNOSTIC (GGML_SCHED_TIME=1): host time per backend in input copies/syncs and in compute, per scheduler
    static const bool sched_time = getenv("GGML_SCHED_TIME") && atoi(getenv("GGML_SCHED_TIME"));
    struct st_stats { int64_t in[GGML_SCHED_MAX_BACKENDS], cp[GGML_SCHED_MAX_BACKENDS], n[GGML_SCHED_MAX_BACKENDS], calls, total; };
    static std::unordered_map<const void *, st_stats> st_map;
    static st_stats st_unused;
    st_stats & st_s = sched_time ? st_map[sched] : st_unused;
    int64_t * st_in = st_s.in; int64_t * st_cp = st_s.cp; int64_t * st_n = st_s.n; int64_t & st_calls = st_s.calls; int64_t & st_total = st_s.total;
    const int64_t st_t_graph = sched_time ? ggml_time_us() : 0;
    int64_t st_t0 = 0;
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        if (sched_time) st_t0 = ggml_time_us();
        ggml_backend_t split_backend = sched->backends[split_backend_id];
        int split_prefetch_slot = -1;
        const uint64_t * split_prefill_table = NULL;
        ggml_tensor * prefetch_input_cpy = NULL;
        ggml_backend_buffer_t prefetch_saved_buffer = NULL;
        void * prefetch_saved_data = NULL;
        const bool split_is_cpu =
            ggml_backend_dev_type(ggml_backend_get_device(split_backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        ggml_backend_t copies_backend = NULL;
        bool user_inputs_queued = false;
        if (split_is_cpu) {
            wait_user_inputs();
        }
        // the split backend was synchronized in this split: the previous users of its input copies are done, and
        // every copy queued afterwards on its stream is ordered behind that point
        bool dst_synced = false;

        // ensure the previous split's async work has completed before we start
        // this split, the allocator may have reused buffer regions across splits
        if (split->n_inputs == 0 && prev_backend_id >= 0 && prev_backend_id != split_backend_id) {
            if (sched->events[prev_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_synchronize(sched->events[prev_backend_id][sched->cur_copy]);
            } else {
                ggml_backend_synchronize(sched->backends[prev_backend_id]);
            }
        }

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            trace_input_log trace_log{trace_on, split_id, ggml_backend_name(split_backend), input,
                ggml_backend_name(input_backend), trace_on ? ggml_time_us() : 0, sched, sched->n_splits};

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // from host memory to a backend that copies asynchronously: queued, waited for once
                // (see user_inputs_pending; a synchronous copy per input cost ~18 us on ROCm: 39 inputs,
                // ~0.9 ms per generated token, while the GPU waited for the first hosted MoE layer's plan)
                const bool copy_async = !split_is_cpu && split_backend->iface.set_tensor_async && input->buffer &&
                    ggml_backend_buffer_is_host(input->buffer) && ggml_is_contiguous(input) &&
                    input_cpy->buffer && ggml_backend_buffer_get_type(input_cpy->buffer) ==
                        ggml_backend_get_default_buffer_type(split_backend);
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done;
                // a copy queued on the split backend runs after the previous graph's readers of its destination, no wait
                // (it cost ~85 us of wake-up whenever anything was still queued, such as a device state upload)
                if (!copy_async && !user_inputs_queued && !dst_synced) {
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                    } else {
                        ggml_backend_synchronize(split_backend);
                        dst_synced = true;
                    }
                }
                if (copy_async) {
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                    user_inputs_queued = true;
                } else {
                    if (user_inputs_queued) {
                        ggml_backend_synchronize(split_backend);
                        user_inputs_queued = false;
                    }
                    ggml_backend_tensor_copy(input, input_cpy);
                }
            } else {
                // with a large batch virtually every expert is used, so instead of waiting
                // for the routing ids, upload the full tensor through the prefetch backend
                // and let the copy overlap compute of the previous split
                if (sched->prefetch_experts && !sched->callback_eval && split_prefetch_slot == -1 && split->graph.n_nodes > 0) {
                    ggml_tensor * node = split->graph.nodes[0];
                    if (ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                        ggml_backend_buffer_is_host(input->buffer) &&
                        node->op == GGML_OP_MUL_MAT_ID && node->src[0] == input_cpy) {
                        const ggml_tensor * ids = node->src[2];
                        const int64_t n_expert = input->ne[2];
                        if (ids->ne[0]*ids->ne[1] >= 2*n_expert &&
                            ggml_backend_sched_prefetch_init(sched, split_backend,
                                ggml_backend_sched_input_alloc_size(ggml_backend_get_default_buffer_type(split_backend), input))) {
                            const int slot = sched->prefetch_cur;
                            sched->prefetch_cur = (sched->prefetch_cur + 1) % sched->prefetch_n_slots;
                            // wait for the previous user of this slot to finish computing
                            if (sched->prefetch_used[slot]) {
                                ggml_backend_event_wait(sched->prefetch_backend, sched->prefetch_free[slot]);
                            }
                            // point the staging copy at the slot only for the duration of
                            // this split, so a fallback to the regular path on a later
                            // eval can never see a dangling slot pointer
                            prefetch_input_cpy    = input_cpy;
                            prefetch_saved_buffer = input_cpy->buffer;
                            prefetch_saved_data   = input_cpy->data;
                            // only the address changes: the tensor keeps its compute buffer,
                            // which the HIP/CUDA paths only check for its type
                            input_cpy->data   = sched->prefetch_slot_data[slot];
                            // the experts the cache holds are read in place: only the others are copied, and the
                            // slot gets the table of all the experts' addresses (resident: their entry; others: here)
                            const size_t padded_size = ggml_backend_sched_input_alloc_size(
                                ggml_backend_get_default_buffer_type(split_backend), input);
                            // a packed layer: each expert copied with its pad, at its step in the slot (one 2D copy a run)
                            const size_t packed_step = ggml_sched_packed_expert_step(input);
                            int n_resident = -1;
                            if (sched->moe_cache_session && ggml_moe_cache.prefill_residents && sched->prefetch_lent &&
                                    n_expert <= GGML_SCHED_PREFETCH_TABLE_EXPERTS) {
                                if (sched->prefetch_table[slot] == NULL) {
                                    sched->prefetch_table[slot] = (uint64_t *) malloc(GGML_SCHED_PREFETCH_TABLE_EXPERTS*sizeof(uint64_t));
                                }
                                // the previous copy from this slot's table is done
                                if (sched->prefetch_used[slot]) {
                                    ggml_backend_event_synchronize(sched->prefetch_ready[slot]);
                                }
                                n_resident = ggml_moe_cache.prefill_residents(sched->moe_cache_session, split_backend,
                                    node, input, sched->prefetch_table[slot]);
                                sched->prefill_pinned = sched->prefill_pinned || n_resident > 0;
                            }
                            if (n_resident <= 0 && packed_step) {
                                ggml_sched_copy_packed_experts(sched->prefetch_backend, input_cpy, input, 0, n_expert, packed_step);
                            } else if (n_resident <= 0) {
                                ggml_backend_tensor_set_async(sched->prefetch_backend, input_cpy, input->data, 0, ggml_nbytes(input));
                            } else {
                                // runs of copied experts; MMQ reads a little past an expert's last row (up to 512
                                // values): the next expert's first bytes come along when it is not copied
                                const uint64_t * res = sched->prefetch_table[slot];
                                const size_t expert_size = input->nb[2];
                                for (int64_t e0 = 0; e0 < n_expert; ) {
                                    if (res[e0]) {
                                        e0++;
                                        continue;
                                    }
                                    int64_t e1 = e0;
                                    while (e1 < n_expert && !res[e1]) {
                                        e1++;
                                    }
                                    if (packed_step) {
                                        ggml_sched_copy_packed_experts(sched->prefetch_backend, input_cpy, input, e0, e1, packed_step);
                                        e0 = e1;
                                        continue;
                                    }
                                    const size_t off  = e0*expert_size;
                                    const size_t tail = e1 < n_expert ? std::min<size_t>(expert_size, 512) : 0;
                                    ggml_backend_tensor_set_async(sched->prefetch_backend, input_cpy,
                                        (const uint8_t *) input->data + off, off, (e1 - e0)*expert_size + tail);
                                    e0 = e1;
                                }
                            }
                            if (n_resident >= 0) {
                                uint64_t * tab = sched->prefetch_table[slot];
                                const uint64_t base = (uint64_t) (uintptr_t) sched->prefetch_slot_data[slot];
                                for (int64_t e = 0; e < n_expert; e++) {
                                    if (!tab[e]) {
                                        tab[e] = base + e*input_cpy->nb[2];
                                    }
                                }
                                GGML_ASSERT(sched->prefetch_backend->iface.set_tensor_async != NULL);
                                sched->prefetch_backend->iface.set_tensor_async(sched->prefetch_backend, input_cpy, tab,
                                    ggml_backend_sched_prefetch_table_offset(padded_size), n_expert*sizeof(uint64_t));
                                split_prefill_table = (const uint64_t *) ((const char *) sched->prefetch_slot_data[slot] +
                                    ggml_backend_sched_prefetch_table_offset(padded_size));
                            }
                            {
                                // re-zero the row padding: a previous, larger tensor may have
                                // left its bytes there (the public setter stops at ggml_nbytes)
                                static const std::vector<uint8_t> zeros(64*1024, 0);
                                const size_t nbytes = ggml_nbytes(input_cpy);
                                const size_t padded = padded_size;
                                if (padded > nbytes) {
                                    GGML_ASSERT(padded - nbytes <= zeros.size());
                                    GGML_ASSERT(sched->prefetch_backend->iface.set_tensor_async != NULL);
                                    sched->prefetch_backend->iface.set_tensor_async(sched->prefetch_backend, input_cpy,
                                        zeros.data(), nbytes, padded - nbytes);
                                }
                            }
                            ggml_backend_event_record(sched->prefetch_ready[slot], sched->prefetch_backend);
                            ggml_backend_event_wait(split_backend, sched->prefetch_ready[slot]);
                            split_prefetch_slot = slot;
                            continue;
                        }
                    }
                }

                // wait for the split backend to finish using the input before overwriting it (not for a copy from host
                // memory into a GPU split: queued on the split backend after those readers, or done synchronously below
                // after waiting for both backends)
                const bool host_to_gpu = !split_is_cpu && input->buffer && ggml_backend_buffer_is_host(input->buffer);
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else if (!dst_synced && !host_to_gpu) {
                    ggml_backend_synchronize(split_backend);
                    dst_synced = true;
                }

                // an input a CPU split takes from a GPU is queued as an asynchronous copy into
                // its host buffer; one synchronization after the inputs replaces a
                // synchronization and a synchronous copy per input (a MoE layer takes two:
                // generation +5 %, measured). Polling a counter the GPU writes behind the
                // copies instead of synchronizing was measured too: no better (-1 to -3 %).
                if (split_is_cpu && input_backend != split_backend &&
                    input_backend->iface.get_tensor_async && ggml_backend_buffer_is_host(input_cpy->buffer) &&
                    ggml_is_contiguous(input) && ggml_is_contiguous(input_cpy)) {
                    ggml_backend_tensor_get_async(input_backend, input, input_cpy->data, 0, ggml_nbytes(input));
                    copies_backend = input_backend;
                    continue;
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    if (ggml_nelements(ids_tensor) == 0) {
                        continue;
                    }

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    const size_t packed_step = node->op == GGML_OP_MUL_MAT_ID ? ggml_sched_packed_expert_step(input) : 0;
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        if (packed_step) {   // a packed layer: each expert with its pad, one 2D copy
                            ggml_sched_copy_packed_experts(split_backend, input_cpy, input, first_id, last_id + 1, packed_step);
                            return;
                        }
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (id < n_expert && !ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    if (id == n_expert) {
                        // A zero-output token batch selects no experts to transfer.
                        continue;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (copies_backend) {
            ggml_backend_synchronize(copies_backend);
        }
        if (user_inputs_queued) {
            if (!sched->user_inputs_done_tried[split_backend_id]) {
                sched->user_inputs_done_tried[split_backend_id] = true;
                sched->user_inputs_done[split_backend_id] = ggml_backend_event_new(ggml_backend_get_device(split_backend));
            }
            if (sched->user_inputs_done[split_backend_id]) {
                ggml_backend_event_record(sched->user_inputs_done[split_backend_id], split_backend);
                user_inputs_pending[split_backend_id] = true;
            } else {
                ggml_backend_synchronize(split_backend);
            }
        }

        if (trace_on && getenv("GGML_SCHED_TRACE_NODES")) {
            for (int j = 0; j < split->graph.n_nodes; j++) {
                const ggml_tensor * t = split->graph.nodes[j];
                GGML_LOG_INFO("trace node: split %3d %-6s %-14s %-36s [%lld,%lld,%lld,%lld] %s\n", split_id,
                        ggml_backend_name(split_backend), ggml_op_desc(t), t->name,
                        (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                        ggml_type_name(t->type));
            }
        }
        int64_t st_t1 = 0;
        if (sched_time) { st_t1 = ggml_time_us(); st_in[split_backend_id] += st_t1 - st_t0; }
        if (!sched->callback_eval) {
            if (split_prefill_table) {
                ggml_moe_cache.prefill_table(prefetch_input_cpy->data, split_prefill_table);
            }
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (split_prefill_table) {
                ggml_moe_cache.prefill_table(NULL, NULL);
            }
            if (sched_time) { st_cp[split_backend_id] += ggml_time_us() - st_t1; st_n[split_backend_id]++; }
            if (split_prefetch_slot != -1) {
                // the kernels have captured the slot address at launch, safe to restore
                ggml_backend_event_record(sched->prefetch_free[split_prefetch_slot], split_backend);
                sched->prefetch_used[split_prefetch_slot] = true;
                prefetch_input_cpy->buffer = prefetch_saved_buffer;
                prefetch_input_cpy->data   = prefetch_saved_data;
            }
            if (ec != GGML_STATUS_SUCCESS) {
                wait_user_inputs();
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    wait_user_inputs();
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this split
        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
            ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
        }

        prev_backend_id = split_backend_id;
    }

    wait_user_inputs();

    if (sched_time) {
        st_total += ggml_time_us() - st_t_graph;
        static const int st_every = getenv("GGML_SCHED_TIME_EVERY") ? atoi(getenv("GGML_SCHED_TIME_EVERY")) : 250;
        if (++st_calls % st_every == 0) {
            for (int b = 0; b < sched->n_backends; b++) {
                GGML_LOG_INFO("sched time %p: %d graphs, backend %s: %.2f splits, inputs %.1f us, compute %.1f us per graph (graph %.1f us)\n",
                        (void *) sched, st_every, ggml_backend_name(sched->backends[b]), (double) st_n[b] / st_every,
                        (double) st_in[b] / st_every, (double) st_cp[b] / st_every, (double) st_total / st_every);
            }
            memset(&st_s, 0, sizeof(st_s));
        }
    }

    return GGML_STATUS_SUCCESS;
}

// GGML_SCHED_BORROW_COMPUTE=1: take the compute buffers from the MoE cache while it is
// idle (during prefill) instead of allocating them next to it.
static void ggml_backend_sched_use_borrowed_compute(ggml_backend_sched_t sched) {
    static const bool enabled = [] {
        // on by default: the compute buffers live in the MoE cache's memory; GGML_SCHED_BORROW_COMPUTE=0 turns it off
        const char * text = getenv("GGML_SCHED_BORROW_COMPUTE");
        return !text || (text[0] && text[0] != '0');
    }();
    // a borrowed type points at the session: without one, back to the plain types
    for (int b = 0; b < sched->n_backends; b++) {
        sched->bufts[b] = sched->bufts_plain[b];
    }
    if (!enabled || !sched->moe_cache_session || sched->moe_cache_guest || !ggml_moe_cache.compute_buffer_type) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_buffer_type_t buft =
            ggml_moe_cache.compute_buffer_type(sched->moe_cache_session, sched->backends[b]);
        if (buft && ggml_backend_supports_buft(sched->backends[b], buft)) {
            sched->bufts[b] = buft;
        }
    }
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    sched->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    sched->graph_inputs = (struct ggml_tensor **) calloc(sched->graph_inputs_capacity, sizeof(struct ggml_tensor *));

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        sched->bufts_plain[b] = sched->bufts[b];
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    if (ggml_moe_cache.session_create) {
        void * cache_backends[GGML_SCHED_MAX_BACKENDS];
        for (int b = 0; b < n_backends; b++) {
            cache_backends[b] = backends[b];
        }
        sched->moe_cache_session = ggml_moe_cache.session_create(cache_backends, n_backends, nullptr);
    }

    ggml_backend_sched_use_borrowed_compute(sched);

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;
    sched->graph_size_max = graph_size;
    sched->plans = new std::vector<ggml_backend_sched_plan *>();

    // GGML_SCHED_PREFETCH_EXPERTS=1 enables the default slot count, higher values set it
    // directly; more slots let uploads run further ahead of compute at the cost of one
    // max-sized expert tensor of device memory per slot
    // on by default with 4 slots: one per expert tensor of a layer (gate/up/down), plus one so that the copy of the next
    // tensor never waits for the multiplication of the tensor whose slot it takes. With 3 the link idled while a
    // layer's experts were multiplied (19k prefill: 3 slots 1 353, 4 slots 1 446, 5-6 the same: the link is then the
    // limit). GGML_SCHED_PREFETCH_EXPERTS=0 turns it off
    const char * GGML_SCHED_PREFETCH_EXPERTS = getenv("GGML_SCHED_PREFETCH_EXPERTS");
    const int prefetch_n_slots = GGML_SCHED_PREFETCH_EXPERTS ? atoi(GGML_SCHED_PREFETCH_EXPERTS) : 4;
    sched->prefetch_experts = op_offload && prefetch_n_slots > 0;
    sched->prefetch_n_slots = prefetch_n_slots <= 1 ? 4 : std::min(prefetch_n_slots, GGML_SCHED_MAX_PREFETCH_SLOTS);

    ggml_backend_sched_reset(sched);

    return sched;
}

// the scheduler lets go of its session: destroyed if it is its own, left to its owner otherwise
static void ggml_backend_sched_drop_moe_cache(ggml_backend_sched_t sched) {
    if (sched->moe_cache_session && !sched->moe_cache_guest && ggml_moe_cache.session_destroy) {
        ggml_moe_cache.session_destroy(sched->moe_cache_session);
    }
    sched->moe_cache_session = nullptr;
    sched->moe_cache_guest = false;
}

void ggml_backend_sched_set_moe_cache(
        ggml_backend_sched_t sched, enum ggml_moe_cache_mode mode,
        size_t budget_mib, size_t reserve_mib, int expert_parallel, int cpu_overlap, const char * profile_path) {
    GGML_ASSERT(sched);
    GGML_ASSERT(cpu_overlap >= -2 && cpu_overlap <= 8);
    // left to the provider (it creates the session at the first reserve), unless the context sets a reserve
    if (mode == GGML_MOE_CACHE_MODE_UNSPECIFIED && cpu_overlap == -2 && reserve_mib == 0) {
        return;
    }

    // the session changes: the compute buffer types may point at it and the allocator holds
    // them, so both are rebuilt on every path below; buffers borrowed from the old session
    // go back to it before it is destroyed (this runs before any graph is reserved)
    ggml_gallocr_free(sched->galloc);
    sched->galloc = NULL;
    ggml_backend_sched_drop_moe_cache(sched);

    auto rebuild_bufts = [sched]() {
        ggml_backend_sched_use_borrowed_compute(sched);
        sched->galloc = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
    };
    if (mode == GGML_MOE_CACHE_MODE_OFF ||
        !ggml_moe_cache.query_config || !ggml_moe_cache.session_create) {
        rebuild_bufts();
        return;
    }

    ggml_moe_cache_config config = {};
    const int automatic = mode == GGML_MOE_CACHE_MODE_UNSPECIFIED ? -1 :
        mode == GGML_MOE_CACHE_MODE_AUTO ? 1 : 0;
    if (!ggml_moe_cache.query_config(automatic, budget_mib, &config)) {
        rebuild_bufts();
        return;
    }
    if (mode != GGML_MOE_CACHE_MODE_UNSPECIFIED) {
        config.expert_parallel = expert_parallel;
        config.profile_path = profile_path;
    }
    if (cpu_overlap != -2) config.overlap_cpu_rows = cpu_overlap;
    if (reserve_mib > 0 && !config.reserve_explicit) {
        config.reserve_bytes    = reserve_mib << 20;
        config.reserve_explicit = 1;
    }

    void * cache_backends[GGML_SCHED_MAX_BACKENDS];
    for (int index = 0; index < sched->n_backends; index++) {
        cache_backends[index] = sched->backends[index];
    }
    sched->moe_cache_session = ggml_moe_cache.session_create(
            cache_backends, sched->n_backends, &config);
    rebuild_bufts();
}

void ggml_backend_sched_moe_cache_kv_register(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, size_t row_bytes) {
    if (sched && sched->moe_cache_session && !sched->moe_cache_guest && ggml_moe_cache.kv_register) {
        ggml_moe_cache.kv_register(sched->moe_cache_session, tensor, row_bytes);
    }
}

void ggml_backend_sched_moe_cache_kv_set_rows(ggml_backend_sched_t sched, int64_t rows) {
    if (!sched || !sched->moe_cache_session || sched->moe_cache_guest || !ggml_moe_cache.kv_needs_set_rows ||
        !ggml_moe_cache.kv_needs_set_rows(sched->moe_cache_session, rows)) {
        return;   // the step boundary stays: nothing moves (almost every ubatch)
    }
    // entries are about to move: no kernel may still read them, and the hosted MoE unpins its rows once its
    // stream is done
    ggml_backend_sched_synchronize(sched);
    ggml_moe_cache.kv_set_rows(sched->moe_cache_session, rows);
}

int64_t ggml_backend_sched_moe_cache_kv_first_lent_row(ggml_backend_sched_t sched) {
    if (!sched || !sched->moe_cache_session || sched->moe_cache_guest || !ggml_moe_cache.kv_first_lent_row) {
        return 0;
    }
    return ggml_moe_cache.kv_first_lent_row(sched->moe_cache_session);
}

void * ggml_backend_sched_detach_moe_cache(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_backend_sched_prefetch_restore(sched);
    // borrowed compute buffers go back to the session while it is still attached
    ggml_gallocr_free(sched->galloc);
    void * session = sched->moe_cache_session;
    sched->moe_cache_session = nullptr;
    sched->moe_cache_guest = false;
    ggml_backend_sched_use_borrowed_compute(sched);
    sched->galloc = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
    return session;
}

void ggml_backend_sched_attach_moe_cache(ggml_backend_sched_t sched, void * session) {
    GGML_ASSERT(sched);
    ggml_gallocr_free(sched->galloc);
    // the default session of a new scheduler gives way to the kept one
    ggml_backend_sched_drop_moe_cache(sched);
    sched->moe_cache_session = session;
    ggml_backend_sched_use_borrowed_compute(sched);
    sched->galloc = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
}

void * ggml_backend_sched_get_moe_cache(ggml_backend_sched_t sched) {
    return sched ? sched->moe_cache_session : nullptr;
}

void ggml_backend_sched_share_moe_cache(ggml_backend_sched_t sched, void * session) {
    GGML_ASSERT(sched);
    ggml_backend_sched_prefetch_restore(sched);
    ggml_gallocr_free(sched->galloc);
    ggml_backend_sched_drop_moe_cache(sched);
    sched->moe_cache_session = session;
    sched->moe_cache_guest = session != nullptr;
    // the zones the prefetch borrows are the owner's
    sched->prefetch_experts = sched->prefetch_experts && !sched->moe_cache_guest;
    // primed again: its expert tensors are shown to the cache, which takes them in the pools of their shapes
    sched->moe_cache_primed = false;
    ggml_backend_sched_use_borrowed_compute(sched);
    sched->galloc = ggml_gallocr_new_n(sched->bufts, sched->n_backends);
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    ggml_backend_sched_prefetch_restore(sched);
    // the compute buffers may be borrowed from the cache: give them back while the
    // session still exists
    ggml_gallocr_free(sched->galloc);
    sched->galloc = NULL;
    ggml_backend_sched_drop_moe_cache(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
        if (sched->user_inputs_done[b]) {
            ggml_backend_event_free(sched->user_inputs_done[b]);
        }
    }
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(sched->prefetch_backend);
        // the slot count may have been reduced after a failed allocation, free everything
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            ggml_backend_event_free(sched->prefetch_ready[i]);
            ggml_backend_event_free(sched->prefetch_free[i]);
            ggml_backend_buffer_free(sched->prefetch_slots[i]);
        }
        for (int i = 0; i < GGML_SCHED_MAX_PREFETCH_SLOTS; i++) {
            free(sched->prefetch_table[i]);
        }
        ggml_backend_free(sched->prefetch_backend);
    }
    for (auto * p : *sched->plans) {
        ggml_backend_sched_plan_free(p);
    }
    delete sched->plans;
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    for (int i = 0; i < sched->splits_capacity; i++) {
        free(sched->splits[i].inputs);
    }
    free(sched->splits);
    free(sched->graph_inputs);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    delete sched->moe_hosted;
    free(sched);
}

// ggml_backend_sched_plan_*: the scheduler's per-graph state (split, copies, allocation) and a kept plan's swapped
static void ggml_backend_sched_plan_swap(ggml_backend_sched_t sched, ggml_backend_sched_plan * p) {
    std::swap(sched->is_alloc,              p->is_alloc);
    std::swap(sched->is_reset,              p->is_reset);
    std::swap(sched->hash_set,              p->hash_set);
    std::swap(sched->hv_tensor_backend_ids, p->hv_tensor_backend_ids);
    std::swap(sched->hv_tensor_copies,      p->hv_tensor_copies);
    std::swap(sched->node_backend_ids,      p->node_backend_ids);
    std::swap(sched->leaf_backend_ids,      p->leaf_backend_ids);
    std::swap(sched->prev_node_backend_ids, p->prev_node_backend_ids);
    std::swap(sched->prev_leaf_backend_ids, p->prev_leaf_backend_ids);
    std::swap(sched->graph,                 p->graph);
    std::swap(sched->splits,                p->splits);
    std::swap(sched->n_splits,              p->n_splits);
    std::swap(sched->splits_capacity,       p->splits_capacity);
    std::swap(sched->graph_inputs,          p->graph_inputs);
    std::swap(sched->n_graph_inputs,        p->n_graph_inputs);
    std::swap(sched->graph_inputs_capacity, p->graph_inputs_capacity);
    std::swap(sched->ctx,                   p->ctx);
    std::swap(sched->context_buffer,        p->context_buffer);
    std::swap(sched->moe_hosted,            p->moe_hosted);
    std::swap(sched->plan_key,              p->key);
    std::swap(sched->plan_key_n_nodes,      p->key_n_nodes);
    std::swap(sched->plan_gen,              p->gen);
}

// the arrays of a plan, sized as ggml_backend_sched_new sizes the scheduler's
static void ggml_backend_sched_plan_arrays(ggml_backend_sched_t sched, ggml_backend_sched_plan * p) {
    const size_t graph_size = sched->graph_size_max;
    p->hash_set              = ggml_hash_set_new(graph_size);
    p->hv_tensor_backend_ids = (int *) malloc(p->hash_set.size * sizeof(p->hv_tensor_backend_ids[0]));
    p->hv_tensor_copies      = (ggml_tensor **) malloc(p->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
    const size_t nodes_size = graph_size + graph_size*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    p->node_backend_ids      = (int *) calloc(nodes_size, sizeof(int));
    p->leaf_backend_ids      = (int *) calloc(nodes_size, sizeof(int));
    p->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(int));
    p->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(int));
    p->graph = {};
    p->splits_capacity = 16;
    p->splits = (ggml_backend_sched_split *) calloc(p->splits_capacity, sizeof(p->splits[0]));
    p->n_splits = 0;
    p->graph_inputs_capacity = GGML_SCHED_MAX_SPLIT_INPUTS;
    p->graph_inputs = (struct ggml_tensor **) calloc(p->graph_inputs_capacity, sizeof(struct ggml_tensor *));
    p->n_graph_inputs = 0;
    p->ctx = nullptr;
    p->context_buffer = (char *) malloc(sched->context_buffer_size);
    p->moe_hosted = nullptr;
    p->is_alloc = false;
    p->is_reset = false;   // the hash arrays are filled by the next reset
    p->arrays = true;
}

static void ggml_backend_sched_plan_free(ggml_backend_sched_plan * p) {
    if (p->arrays) {
        ggml_free(p->ctx);
        ggml_hash_set_free(&p->hash_set);
        for (int i = 0; i < p->splits_capacity; i++) {
            free(p->splits[i].inputs);
        }
        free(p->splits);
        free(p->graph_inputs);
        free(p->hv_tensor_backend_ids);
        free(p->hv_tensor_copies);
        free(p->node_backend_ids);
        free(p->leaf_backend_ids);
        free(p->prev_node_backend_ids);
        free(p->prev_leaf_backend_ids);
        free(p->context_buffer);
        free(p->graph.nodes);
        free(p->graph.leafs);
        delete p->moe_hosted;
    }
    delete p;
}

static bool ggml_backend_sched_plan_valid(ggml_backend_sched_t sched, const ggml_backend_sched_plan * p) {
    return p->key != nullptr && p->is_alloc && p->gen == ggml_gallocr_get_generation(sched->galloc);
}

// before the current state is reset: kept aside in a free, stale or least recently used plan if plan_set marked it
static void ggml_backend_sched_plan_stash(ggml_backend_sched_t sched) {
    if (sched->plan_key == nullptr || !sched->is_alloc || sched->galloc == nullptr ||
        sched->plan_gen != ggml_gallocr_get_generation(sched->galloc)) {
        sched->plan_key = nullptr;
        return;
    }
    auto & plans = *sched->plans;
    ggml_backend_sched_plan * slot = nullptr;
    for (auto * p : plans) {
        if (!ggml_backend_sched_plan_valid(sched, p)) {
            slot = p;
            break;
        }
        if (slot == nullptr || p->used < slot->used) {
            slot = p;
        }
    }
    if (slot == nullptr || (ggml_backend_sched_plan_valid(sched, slot) && plans.size() < GGML_SCHED_MAX_PLANS)) {
        slot = new ggml_backend_sched_plan();
        plans.push_back(slot);
    }
    if (!slot->arrays) {
        ggml_backend_sched_plan_arrays(sched, slot);
    }
    ggml_backend_sched_plan_swap(sched, slot);
    slot->used = ++sched->plan_clock;
    // the scheduler now holds the arrays of the plan dropped (or new ones): a full reset follows
    sched->plan_key = nullptr;
    sched->is_alloc = false;
    sched->is_reset = false;
}

void ggml_backend_sched_plan_set(ggml_backend_sched_t sched, const struct ggml_cgraph * graph) {
    GGML_ASSERT(sched->is_alloc);
    sched->plan_key         = graph;
    sched->plan_key_n_nodes = graph->n_nodes;
    sched->plan_gen         = ggml_gallocr_get_generation(sched->galloc);
}

bool ggml_backend_sched_plan_use(ggml_backend_sched_t sched, const struct ggml_cgraph * graph) {
    const uint64_t gen = ggml_gallocr_get_generation(sched->galloc);
    if (sched->is_alloc && sched->plan_key == graph && sched->plan_key_n_nodes == graph->n_nodes && sched->plan_gen == gen) {
        return true;
    }
    for (auto * p : *sched->plans) {
        if (p->key == graph && p->key_n_nodes == graph->n_nodes && ggml_backend_sched_plan_valid(sched, p)) {
            // the current state takes the plan's place (kept if it is a valid plan itself)
            const bool keep = sched->plan_key != nullptr && sched->is_alloc && sched->plan_gen == gen;
            ggml_backend_sched_plan_swap(sched, p);
            if (!keep) {
                p->key = nullptr;
            }
            p->used = ++sched->plan_clock;
            return true;
        }
    }
    return false;
}

void ggml_backend_sched_plan_forget(ggml_backend_sched_t sched, const struct ggml_cgraph * graph) {
    if (sched->plan_key == graph) {
        sched->plan_key = nullptr;
    }
    for (auto * p : *sched->plans) {
        if (p->key == graph) {
            p->key = nullptr;
        }
    }
}

void ggml_backend_sched_plan_clear(ggml_backend_sched_t sched) {
    sched->plan_key = nullptr;
    for (auto * p : *sched->plans) {
        p->key = nullptr;
    }
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    ggml_backend_sched_plan_stash(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

static void ggml_backend_sched_prime_moe_cache(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);
    ggml_backend_sched_prefill_unpin(sched);

    // the cache must exist before the compute buffers are allocated: with
    // GGML_SCHED_BORROW_COMPUTE they are taken from it
    ggml_backend_sched_prime_moe_cache(sched, measure_graph);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

// Create the MoE cache before the first graph runs, instead of letting it appear on the
// first generated token. The cache learns tensor shapes from the nodes it is asked to
// serve; here it is shown every host-resident expert tensor of this graph as a one-token
// node (begin, then end, no computation). Two passes: the cache builds its pools once it
// has revisited every tensor it knows. The slot size is stated first, so the lendable
// zones are laid out with the pools, and the prefetch can borrow from the first prefill.
static void ggml_backend_sched_prime_moe_cache(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (sched->moe_cache_primed || !sched->moe_cache_session || !ggml_moe_cache.begin ||
            !ggml_moe_cache.end || !ggml_moe_cache.session_enter || !ggml_moe_cache.session_leave) {
        return;
    }
    sched->moe_cache_primed = true;

    std::vector<const ggml_tensor *> experts;
    size_t slot_bytes = 0;
    int n_mmid = 0;
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(sched->backends[0]);
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }
        n_mmid++;
        const ggml_tensor * w   = node->src[0];
        const ggml_tensor * ids = node->src[2];
        if (!w || !ids || !w->buffer || !w->data || !ggml_backend_buffer_is_host(w->buffer)) {
            continue;
        }
        experts.push_back(node);
        if (!ggml_backend_buft_is_host(gpu_buft)) {
            slot_bytes = std::max(slot_bytes, ggml_backend_buft_get_alloc_size(gpu_buft, const_cast<ggml_tensor *>(w)));
        }
    }
    GGML_LOG_INFO("%s: graph with %d nodes: %d MUL_MAT_ID, %zu with host weights, slot %zu MiB\n",
            __func__, graph->n_nodes, n_mmid, experts.size(), slot_bytes >> 20);
    if (experts.empty()) {
        return;
    }

    // the prefetch's backend now rather than at the first large prefill: its queue then exists when the cache
    // measures the free memory (prepare_queues)
    if (sched->prefetch_experts) {
        ggml_backend_sched_prefetch_backend_init(sched, sched->backends[0]->device);
    }
    // a guest's queues were created with the owner's, before the cache was sized
    if (ggml_moe_cache.prepare_queues && !sched->moe_cache_guest) {
        ggml_moe_cache.prepare_queues(sched->moe_cache_session, graph);
    }

    if (sched->prefetch_experts && slot_bytes && ggml_moe_cache.set_lend_size) {
        ggml_moe_cache.set_lend_size(sched->moe_cache_session, slot_bytes, sched->prefetch_n_slots);
    }

    int accepted[2] = {0, 0};
    (sched->moe_cache_guest && ggml_moe_cache.session_enter_guest ? ggml_moe_cache.session_enter_guest
                                                                  : ggml_moe_cache.session_enter)(sched->moe_cache_session);
    for (int pass = 0; pass < 2; pass++) {
        for (const ggml_tensor * node : experts) {
            const ggml_tensor * w   = node->src[0];
            const ggml_tensor * ids = node->src[2];
            // one token routed to one expert: the warmup graph routes every token to
            // all experts (ids->ne[0] == n_expert), which the cache would reject as a
            // batch above its row limit
            void * probe = ggml_moe_cache.begin(w->name, w->data, w->nb[2],
                    w->ne[0], w->ne[1], w->type, w->ne[2],
                    /*n_tokens =*/ 1, /*n_rows =*/ 1);
            if (probe) {
                accepted[pass]++;
                ggml_moe_cache.end(probe);
            }
        }
    }
    ggml_moe_cache.session_leave(sched->moe_cache_session);
    GGML_LOG_INFO("%s: MoE cache probed with %zu tensors, accepted %d then %d (first: %s, %s, rows %lld)\n",
            __func__, experts.size(), accepted[0], accepted[1],
            experts[0]->src[0]->name, ggml_type_name(experts[0]->src[0]->type),
            (long long)experts[0]->src[2]->ne[0]);
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    // compute buffers may be borrowed from the cache below: no entry may still be pinned
    ggml_backend_sched_prefill_unpin(sched);

    // before split_graph, which redirects weight sources to their device copies
    ggml_backend_sched_prime_moe_cache(sched, graph);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (sched->prefetch_backend) {
        ggml_backend_synchronize(sched->prefetch_backend);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

bool ggml_op_alloc_size_may_expand(enum ggml_op op) {
    switch (op) {
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return true;
        default:
            return false;
    }
}

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name            = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer        = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .alloc_buffer_n      = */ NULL,
            /* .get_alignment       = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size        = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size      = */ NULL, // defaults to ggml_nbytes
            /* .get_alloc_size_n    = */ NULL,
            /* .is_host             = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
