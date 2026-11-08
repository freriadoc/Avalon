#include "norm.cuh"
#include "convert.cuh"
#include "norm-gate-body.cuh"
#include "mk.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f,
                                    const ggml_cuda_img img            = ggml_cuda_img()) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;
            // the image of dst for the cuBLAS products after it (ggml_cuda_f16_image_target)
            const int64_t img_off = ((int64_t) (sample*nchannels + channel)*nrows + row)*ncols;

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                        if (img.p) {
                            img.store4(img_off + 4*i, r);   // q8_1: whole warps (rms_norm_f32_dispatch)
                        }
                    }
                }
            } else {
                float tmp = 0.0f; // partial sum for thread in warp

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                const float mean = tmp / ncols;
                const float scale = rsqrtf(mean + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    float r;
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        r = scale * xc[col] * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        r = scale * xc[col] * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        r = scale_out * (scale * xc[col]);
                    } else {
                        r = scale * xc[col];
                    }
                    dstc[col] = r;
                    if (img.p) {
                        img.store(img_off + col, r);
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale>, launch_params, \
            x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            add, add_stride_row, add_stride_channel, add_stride_sample, \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out, img); \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f, const ggml_cuda_img img = ggml_cuda_img()) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;

    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }

    const int n4 = ncols / 4;
    // a q8_1 image is written by whole warps, 8 lanes per block of 32 values: the float4 path, rows of 128 values
    GGML_ASSERT(img.p == nullptr || img.kind != GGML_CUDA_IMG_Q8_1_D4 || (vec && ncols % 128 == 0 && n4 <= 4096));
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream,
                                  const ggml_cuda_img img = ggml_cuda_img()) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (mul_t == nullptr) {
        GGML_ASSERT(img.p == nullptr);
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream, 1.0f, img);
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            add_t, add_stride_row, add_stride_channel, add_stride_sample,
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
        GGML_ASSERT(img.p == nullptr);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, ggml_cuda_img img) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream, img);
}

// rms_norm(x) * w and a product of it with few rows (qwen4exp's hyper-connection inject: 10240 x 4, Q8_0), a block per
// token over its nrows rows: the norm as rms_norm_f32 computes it (1024 threads, a float4 each, the same reduction: the
// same bits), then each value times the dequantized weights, accumulated in F32 and reduced once per output. The
// product (a cuBLAS or MMQ launch over the whole activation for 4 outputs) is not made
// post (the prompt's hc_combine before the norm, kern3): the rows normalized are residual + bo * 2*sigmoid(raw/hc) per
// stream, computed here in hc_post's order (dsv4_hc_post_f32) and written to post_dst, x being the residual; the
// block reads the token's bo row once for its streams
struct rms_norm_inject_post {
    const float * bo = nullptr; int64_t sbo1 = 0;      // the block output [ncols, nt]
    const float * raw = nullptr; int64_t sp0 = 0, sp1 = 0;   // the gate before SCALE -> SIGMOID -> SCALE [nrows, nt]
    float gin = 1.0f, gout = 1.0f;
    float * dst = nullptr;                              // the combined streams, contiguous [ncols, nrows, nt]
};

template <int block_size, int n_out, bool post = false>
static __global__ void __launch_bounds__(block_size) rms_norm_mul_inject_f32(
        const float * x, const float * w, float * dst, const ggml_cuda_img img, const block_q8_0 * wq, const int64_t wq_s1,
        float * inj, const int ncols, const int nrows, const int64_t sx1, const int64_t sx2, const float eps,
        const rms_norm_inject_post pp = rms_norm_inject_post()) {
    const int64_t t   = blockIdx.x;
    const int     tid = threadIdx.x;
    const int     ncols4 = ncols / 4;
    extern __shared__ float s_sum[];

    float acc[n_out];
#pragma unroll
    for (int o = 0; o < n_out; o++) {
        acc[o] = 0.0f;
    }
    for (int c = 0; c < nrows; c++) {
        const float4 * x4 = (const float4 *) (x + t*sx2 + c*sx1);
        float4 xs  = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float  tmp = 0.0f;
        if (tid < ncols4) {
            xs  = x4[tid];
            if constexpr (post) {
                float p = pp.raw[c*pp.sp0 + t*pp.sp1];
                p = pp.gout * (1.0f / (1.0f + expf(-(pp.gin * p))));
                const float4 b = ((const float4 *) (pp.bo + t*pp.sbo1))[tid];
                float sum;
                sum = b.x * p; sum += xs.x; xs.x = sum;
                sum = b.y * p; sum += xs.y; xs.y = sum;
                sum = b.z * p; sum += xs.z; xs.z = sum;
                sum = b.w * p; sum += xs.w; xs.w = sum;
                ((float4 *) (pp.dst + (t*nrows + c)*ncols))[tid] = xs;
            }
            tmp = xs.x*xs.x + xs.y*xs.y + xs.z*xs.z + xs.w*xs.w;
        }
        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
        const float scale = rsqrtf(tmp / ncols + eps);
        if (tid < ncols4) {
            float4 r = make_float4(scale*xs.x, scale*xs.y, scale*xs.z, scale*xs.w);
            const float4 m = ((const float4 *) (w + (int64_t) c*ncols))[tid];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            const int64_t off = (t*nrows + c)*ncols + 4*tid;
            *((float4 *) (dst + off)) = r;
            if (img.p) {
                img.store4(off, r);   // q8_1: whole warps (ggml_cuda_rms_norm_mul_inject_supported: ncols % 128 == 0)
            }
            // the value's place in the token's vector, the product's K
            const int k  = c*ncols + 4*tid;
            const int kb = k / QK8_0;
            const int kq = k % QK8_0;
#pragma unroll
            for (int o = 0; o < n_out; o++) {
                const block_q8_0 * b = wq + o*wq_s1 + kb;
                const float d = __half2float(b->d);
                // the block is 34 bytes: its codes are 2-byte aligned only, read them one by one
                const int8_t * q = b->qs + kq;
                acc[o] += d * (r.x*q[0] + r.y*q[1] + r.z*q[2] + r.w*q[3]);
            }
        }
        __syncthreads();   // s_sum is reused by the next reduction
    }
#pragma unroll
    for (int o = 0; o < n_out; o++) {
        const float v = block_reduce<block_reduce_method::SUM, block_size>(acc[o], s_sum);
        if (tid == 0) {
            inj[t*n_out + o] = v;
        }
        __syncthreads();
    }
}

bool ggml_cuda_rms_norm_mul_inject_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * mm) {
    const ggml_tensor * x  = rms_norm->src[0];
    const ggml_tensor * w  = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * wq = mm->src[0];
    const ggml_tensor * y  = mm->src[1];
    const ggml_tensor * yr = y->view_src ? y->view_src : y;
    const int64_t ncols = x->ne[0];
    return x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && x->ne[3] == 1 && ncols % 4 == 0 && ncols / 4 <= 1024 &&
        ggml_cuda_is_aligned(x, 16) && x->nb[1] % 16 == 0 && x->nb[2] % 16 == 0 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == ncols && w->ne[1] == x->ne[1] &&
        ggml_nelements(w) == ncols * x->ne[1] && ggml_cuda_is_aligned(w, 16) &&
        mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_are_same_shape(mul, x) && ggml_cuda_is_aligned(mul, 16) &&
        yr == mul && y->view_offs == 0 && ggml_is_contiguous(y) && y->ne[0] == ncols * x->ne[1] && y->ne[1] == x->ne[2] &&
        ggml_nrows(y) == x->ne[2] &&
        wq->type == GGML_TYPE_Q8_0 && wq->ne[0] == y->ne[0] && (wq->ne[1] == 4 || wq->ne[1] == 8) && ggml_is_matrix(wq) &&
        wq->nb[1] % sizeof(block_q8_0) == 0 && (int64_t) ggml_nelements(x) < INT32_MAX &&
        mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm) && x->ne[2] <= INT32_MAX;
}

void ggml_cuda_op_rms_norm_mul_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul,
        ggml_tensor * mm, ggml_cuda_img img, const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x  = rms_norm->src[0];
    // with the gate chain: rms_norm's source is the gated hc_post, computed here from its sources
    const bool post = gate_scale_in != nullptr;
    rms_norm_inject_post pp;
    if (post) {
        const ggml_tensor * hp  = x;
        const ggml_tensor * raw = gate_scale_in->src[0];
        pp.bo   = (const float *) hp->src[0]->data;
        pp.sbo1 = hp->src[0]->nb[1] / sizeof(float);
        pp.raw  = (const float *) raw->data;
        pp.sp0  = raw->nb[0] / sizeof(float);
        pp.sp1  = raw->nb[1] / sizeof(float);
        pp.gin  = ggml_get_op_params_f32(gate_scale_in,  0);
        pp.gout = ggml_get_op_params_f32(gate_scale_out, 0);
        pp.dst  = (float *) hp->data;
        x = hp->src[1];   // the residual
    }
    const ggml_tensor * w  = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * wq = mm->src[0];
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    constexpr int block_size = 1024;
    const dim3 grid((unsigned) x->ne[2]);
    const int64_t wq_s1 = wq->nb[1] / sizeof(block_q8_0);
    if (post) {
        GGML_ASSERT(wq->ne[1] == 4);
        rms_norm_mul_inject_f32<block_size, 4, true><<<grid, block_size, 32 * sizeof(float), ctx.stream()>>>(
            (const float *) x->data, (const float *) w->data, (float *) mul->data, img, (const block_q8_0 *) wq->data, wq_s1,
            (float *) mm->data, (int) x->ne[0], (int) x->ne[1], (int64_t) (x->nb[1] / sizeof(float)),
            (int64_t) (x->nb[2] / sizeof(float)), eps, pp);
    } else if (wq->ne[1] == 4) {
        rms_norm_mul_inject_f32<block_size, 4><<<grid, block_size, 32 * sizeof(float), ctx.stream()>>>(
            (const float *) x->data, (const float *) w->data, (float *) mul->data, img, (const block_q8_0 *) wq->data, wq_s1,
            (float *) mm->data, (int) x->ne[0], (int) x->ne[1], (int64_t) (x->nb[1] / sizeof(float)),
            (int64_t) (x->nb[2] / sizeof(float)), eps);
    } else {
        rms_norm_mul_inject_f32<block_size, 8><<<grid, block_size, 32 * sizeof(float), ctx.stream()>>>(
            (const float *) x->data, (const float *) w->data, (float *) mul->data, img, (const block_q8_0 *) wq->data, wq_s1,
            (float *) mm->data, (int) x->ne[0], (int) x->ne[1], (int64_t) (x->nb[1] / sizeof(float)),
            (int64_t) (x->nb[2] / sizeof(float)), eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// rms_norm(x) * w * sigmoid(gate) per row (a gated norm, as qwen4exp ends its linear attention); w is
// one vector of ncols. A block reads its row of x and of gate before writing it: dst may alias x or
// gate exactly. The body: norm-gate-body.cuh.
template <int block_size, bool has_img>
static __global__ void rms_norm_mul_sigmoid_gate_f32(const rms_norm_gate_args a, const ggml_cuda_img img) {
    rms_norm_mul_sigmoid_gate_body<block_size, has_img>(a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, img);
}

void ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * w, const ggml_tensor * gate, ggml_tensor * dst, void * dst_q8_1, ggml_cuda_img img) {
    const ggml_tensor * x = rms_norm->src[0];
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const int64_t ncols = x->ne[0];
    const dim3 blocks_num(x->ne[1], x->ne[2], x->ne[3]);
    const int64_t sx1 = x->nb[1] / sizeof(float), sx2 = x->nb[2] / sizeof(float), sx3 = x->nb[3] / sizeof(float);
    const int64_t sg1 = gate->nb[1] / sizeof(float), sg2 = gate->nb[2] / sizeof(float), sg3 = gate->nb[3] / sizeof(float);
    constexpr int block_size = 128;
    rms_norm_gate_args a;
    a.x = (const float *) x->data; a.w = (const float *) w->data; a.gate = (const float *) gate->data; a.dst = (float *) dst->data;
    a.ncols = (int) ncols; a.sx1 = sx1; a.sx2 = sx2; a.sx3 = sx3; a.sg1 = sg1; a.sg2 = sg2; a.sg3 = sg3; a.eps = eps;
    a.dst_q8_1 = (block_q8_1 *) dst_q8_1;
    a.nrows = (int) x->ne[1]; a.nchannels = (int) x->ne[2]; a.nsamples = (int) x->ne[3];
    if (img.p) {
        ggml_cuda_mk_direct(ctx.mk, ctx.stream());   // the image's variant is not recorded: the pending tasks first
        rms_norm_mul_sigmoid_gate_f32<block_size, true><<<blocks_num, block_size, 32 * sizeof(float), ctx.stream()>>>(a, img);
        return;
    }
    if (ctx.mk && ggml_cuda_mk_add_norm_gate(ctx.mk, a)) {   // the multi-task kernel takes it (mk.cuh)
        return;
    }
    ggml_cuda_mk_direct(ctx.mk, ctx.stream());
    rms_norm_mul_sigmoid_gate_f32<block_size, false><<<blocks_num, block_size, 32 * sizeof(float), ctx.stream()>>>(a, img);
}
