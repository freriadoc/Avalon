#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst,
                                         const ggml_tensor *         sh       = nullptr,   // shared expert's output, added
                                         const ggml_tensor *         sh_logit = nullptr);  // gated by sigmoid of this (one per token)
