#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// bytes of scratch the top-k of dst takes instead of pool memory, 0 if none (only the radix path has any)
size_t ggml_cuda_top_k_scratch_size(const ggml_tensor * dst);
