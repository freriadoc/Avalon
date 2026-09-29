#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h).

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void ggml_moe_cache_register(const void * owner);

// Surrender the device's cache VRAM under allocator pressure; returns bytes freed.
size_t ggml_moe_cache_trim(int device);

#ifdef __cplusplus
}

struct ggml_backend_cuda_context;
struct ggml_cgraph;

// MoE block (gate / up / swiglu / down) at node i whose weights are in host memory, left on the GPU
// by the scheduler (GGML_MOE_GPU=1): queues the block (resident rows on the GPU, the others on
// the CPU through a host thread) and returns the nodes it covers after i (3), or -1 if node i does
// not start such a block.
int ggml_cuda_moe_hosted_block(ggml_backend_cuda_context & ctx, struct ggml_cgraph * cgraph, int i);

// Copies the CPU rows of the last hosted block in, if 'node' reads its output (or always, node NULL:
// end of the graph). Called before every node the graph evaluation computes.
struct ggml_tensor;
void ggml_cuda_moe_hosted_flush(ggml_backend_cuda_context & ctx, const struct ggml_tensor * node);
#endif
