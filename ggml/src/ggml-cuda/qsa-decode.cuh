#pragma once

#include "common.cuh"

// ggml_qsa_decode (see ggml.h)
void ggml_cuda_op_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_qsa_decode_supported(const ggml_tensor * op);
