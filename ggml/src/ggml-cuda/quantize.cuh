#pragma once

#include "common.cuh"
#include "mmq.cuh"

#include <cstdint>

#define CUDA_QUANTIZE_BLOCK_SIZE     256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128

static_assert(MATRIX_ROW_PADDING %    CUDA_QUANTIZE_BLOCK_SIZE      == 0, "Risk of out-of-bounds access.");
static_assert(MATRIX_ROW_PADDING % (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ) == 0, "Risk of out-of-bounds access.");

typedef void (*quantize_cuda_t)(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy,
        ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_fp4_cuda(const float *   x,
                             const int32_t * ids,
                             void *          vy,
                             float *         scale,
                             ggml_type       type_src0,
                             bool            use_aligned_float8,
                             int64_t         ne00,
                             int64_t         s01,
                             int64_t         s02,
                             int64_t         s03,
                             int64_t         ne0,
                             int64_t         ne1,
                             int64_t         ne2,
                             int64_t         ne3,
                             cudaStream_t    stream);

// quantize each token once and scatter the block to its compact rows (via the inverse map)
void quantize_scatter_mmq_fp4_cuda(const float *   x,
                                   const int32_t * ids_src1_inv,
                                   void *          vy,
                                   float *         scale,
                                   ggml_type       type_src0,
                                   bool            use_aligned_float8,
                                   int64_t         ne00,
                                   int64_t         stride_token,
                                   int64_t         ne0,
                                   int64_t         n_tokens,
                                   int64_t         nrows_dst,
                                   int             n_expert_used,
                                   cudaStream_t    stream);

void quantize_scatter_mmq_q8_1_cuda(const float *   x,
                                    const int32_t * ids_src1_inv,
                                    void *          vy,
                                    ggml_type       type_src0,
                                    int64_t         ne00,
                                    int64_t         stride_token,
                                    int64_t         ne0,
                                    int64_t         n_tokens,
                                    int64_t         nrows_dst,
                                    int             n_expert_used,
                                    cudaStream_t    stream);

// q8_1 rows of the activation x for mmvq (the layout of quantize_row_q8_1_cuda), quantized into the
// context's reuse buffer unless the last call quantized the same activation with the same layout
// and nothing wrote over it since. nullptr when the buffer cannot serve (too small, not the main
// stream): quantize into a temporary buffer as before.
const void * ggml_cuda_q8_1_reuse(ggml_backend_cuda_context & ctx, const float * x,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// a node wrote dst: drop the reused quantization if dst overlaps its activation
void ggml_cuda_q8_1_reuse_written(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// A node whose kernel can write its f32 output together with its q8_1 quantization (the bytes quantize_q8_1
// would write) for the mat-vec product that reads it next: the reuse buffer when the product reads the output as
// it is (same address, two dimensions, contiguous, rows a multiple of MATRIX_ROW_PADDING), else null. The buffer
// is then registered as holding that activation (call it right before the launch).
void * ggml_cuda_q8_1_produce(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int out_idx);
// the q8_1 of node out_idx (F32, contiguous, of a width multiple of QK8_1) written by the kernel that computes it into
// ctx.q8_1_out (rows padded to MATRIX_ROW_PADDING, the padding's blocks zero as quantize_q8_1 writes them), for the first
// product of quantized weights that reads it (ggml_cuda_q8_1_out_match); nullptr if none or it does not qualify
void * ggml_cuda_q8_1_produce_out(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int out_idx);
// q8_1_out's buffer if it holds the activation that ggml_cuda_q8_1_reuse would quantize with these arguments
const void * ggml_cuda_q8_1_out_match(ggml_backend_cuda_context & ctx, const float * x,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
