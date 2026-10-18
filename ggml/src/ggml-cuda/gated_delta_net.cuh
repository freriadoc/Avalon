#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// the GET_ROWS node behind a gated delta net's state (through reshapes) when the graph evaluation
// left that gather to the kernel, nullptr otherwise
const ggml_tensor * ggml_cuda_gdn_state_gather_of(const ggml_backend_cuda_context & ctx, const ggml_tensor * state);

// whether the multi-task kernel takes this gated delta net (mk.cuh)
bool ggml_cuda_gdn_mk_recordable(const ggml_tensor * dst);
