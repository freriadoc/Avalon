#include <algorithm>
#include <climits>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

static __global__ void argmax_f32(const float * __restrict__ x, int32_t * __restrict__ dst, const int64_t ncols) {
    const int64_t row = blockIdx.x;

    float maxval = -FLT_MAX;
    int   argmax = -1;
    const float * rowx = x + row * ncols;

    for (int32_t col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float val = rowx[col];
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        if (val > maxval) {
            maxval = val;
            argmax = col;
        }
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                if (val > maxval) {
                    maxval = val;
                    argmax = col;
                }
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax;
    }
}

// a long row (a vocabulary's logits): one block reads 1 MB on one CU (~40 us for 248k columns on RDNA3); here
// ARGMAX_SPLIT blocks per row each reduce a slice, then a warp per row reduces their partial maxima. Ties go to the
// smaller column (the CPU's argmax).
#define ARGMAX_SPLIT 64

static __device__ __forceinline__ void argmax_pick(float & maxval, int & argmax, const float val, const int col) {
    if (val > maxval || (val == maxval && col < argmax)) {
        maxval = val;
        argmax = col;
    }
}

static __global__ void argmax_f32_split(const float * __restrict__ x, float * __restrict__ pval, int * __restrict__ pidx,
        const int64_t ncols) {
    const int64_t row = blockIdx.y;
    const int64_t per = (ncols + gridDim.x - 1) / gridDim.x;
    const int64_t c0 = blockIdx.x * per, c1 = min(ncols, c0 + per);
    const float * rowx = x + row * ncols;

    float maxval = -FLT_MAX;
    int   argmax = INT_MAX;
    for (int64_t col = c0 + threadIdx.x; col < c1; col += blockDim.x) {
        argmax_pick(maxval, argmax, rowx[col], (int) col);
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        argmax_pick(maxval, argmax, __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE),
                    __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE));
    }
    __shared__ float smax[1024 / WARP_SIZE];
    __shared__ int   sidx[1024 / WARP_SIZE];
    const int lane_id = threadIdx.x % WARP_SIZE, warp_id = threadIdx.x / WARP_SIZE, n_warps = blockDim.x / WARP_SIZE;
    if (lane_id == 0) {
        smax[warp_id] = maxval;
        sidx[warp_id] = argmax;
    }
    __syncthreads();
    if (warp_id == 0) {
        maxval = lane_id < n_warps ? smax[lane_id] : -FLT_MAX;
        argmax = lane_id < n_warps ? sidx[lane_id] : INT_MAX;
#pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            argmax_pick(maxval, argmax, __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE),
                        __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE));
        }
        if (lane_id == 0) {
            pval[row * gridDim.x + blockIdx.x] = maxval;
            pidx[row * gridDim.x + blockIdx.x] = argmax;
        }
    }
}

static __global__ void argmax_f32_merge(const float * __restrict__ pval, const int * __restrict__ pidx, int32_t * __restrict__ dst,
        const int nsplit) {
    const int64_t row = blockIdx.x;
    float maxval = -FLT_MAX;
    int   argmax = INT_MAX;
    for (int i = threadIdx.x; i < nsplit; i += WARP_SIZE) {
        argmax_pick(maxval, argmax, pval[row * nsplit + i], pidx[row * nsplit + i]);
    }
#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        argmax_pick(maxval, argmax, __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE),
                    __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE));
    }
    if (threadIdx.x == 0) {
        dst[row] = argmax == INT_MAX ? -1 : argmax;
    }
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    if (ne00 >= 32768 && nrows <= 65535) {
        ggml_cuda_pool_alloc<float> pval(ctx.pool(), nrows * ARGMAX_SPLIT);
        ggml_cuda_pool_alloc<int>   pidx(ctx.pool(), nrows * ARGMAX_SPLIT);
        argmax_f32_split<<<dim3(ARGMAX_SPLIT, nrows, 1), 256, 0, stream>>>(src0_d, pval.get(), pidx.get(), ne00);
        argmax_f32_merge<<<nrows, WARP_SIZE, 0, stream>>>(pval.get(), pidx.get(), dst_d, ARGMAX_SPLIT);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, ne00);
}
