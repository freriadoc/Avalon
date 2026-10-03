#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// bytes of scratch the flash attention of dst takes instead of pool memory (a dry run of its dispatch: the
// same kernel, occupancy and block split), 0 if none
size_t ggml_cuda_flash_attn_ext_scratch_size(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
