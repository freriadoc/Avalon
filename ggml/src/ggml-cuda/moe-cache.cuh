#pragma once

// MoE expert cache registration entry point, called from ggml_backend_cuda_reg().
// Populates ggml_moe_cache (see ggml-backend-moe-cache.h).

#include <stddef.h>
#include <stdint.h>

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
// by the scheduler (default; GGML_MOE_GPU=0 turns it off): queues the block (resident rows on the GPU, the others on
// the CPU through a host thread) and returns the nodes it covers after i (3), or -1 if node i does
// not start such a block.
int ggml_cuda_moe_hosted_block(ggml_backend_cuda_context & ctx, struct ggml_cgraph * cgraph, int i);

// rows a graph waits for, written by the host after the launch (see moe-cache.cu): the mapped buffer (nullptr if too large),
// the task that writes it (run by the first wait for the GPU), the node that waits for it (true: the node was launched)
void * ggml_cuda_host_rows_buffer(size_t bytes);
void ggml_cuda_host_rows_task(void (*fn)(void *), void * user);
bool ggml_cuda_host_rows_node(ggml_backend_cuda_context & ctx, struct ggml_tensor * node);

// Copies the CPU rows of the last hosted block in, if 'node' reads its output (or always, node NULL:
// end of the graph). Called before every node the graph evaluation computes.
struct ggml_tensor;
void ggml_cuda_moe_hosted_flush(ggml_backend_cuda_context & ctx, const struct ggml_tensor * node);

// Waits for the stream (or the event, if not null) like cudaStreamSynchronize / cudaEventSynchronize
// while computing the missed experts of the hosted blocks the stream publishes meanwhile. Returns
// false if no block is hosted: the caller synchronizes as usual.
bool ggml_cuda_moe_hosted_wait(void * stream, void * event, int device);
// a hosted MoE block exists (its kernels may wait for the host)
bool ggml_cuda_moe_hosted_active();   // cudaStream_t, cudaEvent_t, the stream's device

// Prefill: the table of expert addresses of a prefetched MUL_MAT_ID weight (one per expert: its entry in the cache,
// or its place in the prefetch slot), registered by the scheduler for the duration of its split; nullptr otherwise.
// MMQ then reads each expert at its address (ggml_cuda_mul_mat_q).
const uint64_t * ggml_cuda_moe_prefill_table(const void * w_data);
// whether ggml_cuda_mul_mat_id computes this node with MMQ (the only path that reads such a table)
bool ggml_cuda_mul_mat_id_uses_mmq(const struct ggml_tensor * dst);
#endif
