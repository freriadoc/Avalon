#include "speculative_backend_llama.h"

#include <cstdint>
#include <vector>

namespace syclonith::speculative {

const char * LlamaExecutionBackend::name() const {
    return "llama-ggml";
}

int LlamaExecutionBackend::process(llama_context * ctx, common_batch & batch) {
    if (ctx == nullptr || batch.size() <= 0) {
        return -1;
    }

    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
}

void LlamaExecutionBackend::synchronize(llama_context * ctx) {
    if (ctx != nullptr) {
        llama_synchronize(ctx);
    }
}

bool LlamaExecutionBackend::copy_tensor(
        ggml_tensor * src,
        ggml_backend_t src_backend,
        size_t src_offset,
        ggml_tensor * dst,
        ggml_backend_t dst_backend,
        size_t dst_offset,
        size_t n_bytes) {
    if (src == nullptr || dst == nullptr || n_bytes == 0) {
        return false;
    }

    const size_t src_size = ggml_nbytes(src);
    const size_t dst_size = ggml_nbytes(dst);
    if (src_offset > src_size || n_bytes > src_size - src_offset ||
        dst_offset > dst_size || n_bytes > dst_size - dst_offset) {
        return false;
    }

    // Portable correctness-first transfer. Backend-native async/peer copies
    // can replace this implementation without changing the adapter contract.
    if (src_backend != nullptr) {
        ggml_backend_synchronize(src_backend);
    }

    std::vector<uint8_t> staging(n_bytes);
    ggml_backend_tensor_get(src, staging.data(), src_offset, n_bytes);

    if (dst_backend != nullptr) {
        ggml_backend_synchronize(dst_backend);
    }
    ggml_backend_tensor_set(dst, staging.data(), dst_offset, n_bytes);
    if (dst_backend != nullptr) {
        ggml_backend_synchronize(dst_backend);
    }

    return true;
}

bool LlamaExecutionBackend::can_access(ggml_backend_t src, ggml_backend_t dst) const {
    // Backend identity does not imply peer access across distinct instances.
    return src != nullptr && src == dst;
}

} // namespace syclonith::speculative
