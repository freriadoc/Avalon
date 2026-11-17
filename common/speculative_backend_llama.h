#pragma once

#include "speculative_arch.h"

namespace syclonith::speculative {

// Generic llama/ggml execution bridge. Architecture-specific tensor selection
// and row semantics remain the responsibility of the architecture adapter.
class LlamaExecutionBackend final : public ExecutionBackend {
public:
    const char * name() const override;

    // common_batch::get() materializes its llama_batch_ext view, so this must
    // accept a mutable batch even though the adapter does not alter its tokens.
    int process(llama_context * ctx, common_batch & batch) override;
    void synchronize(llama_context * ctx) override;

    bool copy_tensor(
            ggml_tensor * src,
            ggml_backend_t src_backend,
            size_t src_offset,
            ggml_tensor * dst,
            ggml_backend_t dst_backend,
            size_t dst_offset,
            size_t n_bytes) override;

    bool can_access(ggml_backend_t src, ggml_backend_t dst) const override;
};

} // namespace syclonith::speculative
