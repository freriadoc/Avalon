#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// SET_ROWS into Q8_0 of rows that are values (contiguous rows, values->ne[0] <= the row) then zeros: the CONCAT of the
// values with a FILL of 0 that dst's source is, without computing it
void ggml_cuda_op_set_rows_padded(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * values);
