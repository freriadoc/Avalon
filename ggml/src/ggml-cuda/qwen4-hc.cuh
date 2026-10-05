#pragma once

#include "common.cuh"

// Fused hyper-connection boundary (combine + grouped norm, down/inject, up + gated mean) for
// batches of up to 8 tokens. Returns the number of nodes to skip, 0 if nothing was fused.
// check_ranges is ggml-cuda's fusion memory-range check.
int ggml_cuda_try_fuse_qwen4_hc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*check_ranges)(const ggml_cgraph *, int, int, const int *, int));

// Q8_0 x f32 product of at most 8 columns without quantizing the activation (src[1] that may not be
// converted below F32, ggml_prec_set_src: qwen4exp's hyper-connection gate); the fused up/pre kernel
// computes the same way
bool ggml_cuda_mul_mat_q8_0_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mul_mat_q8_0_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
