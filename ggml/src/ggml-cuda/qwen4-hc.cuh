#pragma once

#include "common.cuh"

// Fused hyper-connection boundary (combine + grouped norm, down/inject, up + gated mean) for
// batches of up to 8 tokens. Returns the number of nodes to skip, 0 if nothing was fused.
// check_ranges is ggml-cuda's fusion memory-range check.
int ggml_cuda_try_fuse_qwen4_hc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*check_ranges)(const ggml_cgraph *, int, int, const int *, int));
