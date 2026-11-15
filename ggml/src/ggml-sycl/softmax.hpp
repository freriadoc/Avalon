#include "softmax.hpp"
#include <cstdint>
#include <utility>
#include <cmath>
#include <cstring>


template <typename T> static __dpct_inline__ float t2f32(T val) {
    return (float) val;
}

template <> float __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  return sycl::vec<sycl::half, 1>(val)
      .convert<float, sycl::rounding_mode::automatic>()[0];
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static void soft_max_f32(const float *         x,
                         const T *             mask,
                         const float *         sinks,
                         float *               dst,
                         const soft_max_params p,
                         uint8_t *             dpct_local) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t ncols = ncols_template == 0 ? p.ncols : ncols_template;
    const int block_size = block_size_template == 0
                               ? item_ct1.get_local_range(2)
                               : block_size_template;
    const int nthreads = block_size;
    const int nwarps = nthreads / WARP_SIZE;
    // Number of warp-sized groups needed to reduce all warp partials.
    const int nreduce = (nwarps + WARP_SIZE - 1) / WARP_SIZE;

    const int tid = item_ct1.get_local_id(2);

    const int64_t i03 = item_ct1.get_group(0);
    const int64_t i02 = item_ct1.get_group(1);
    const int64_t i01 = item_ct1.get_group(2);

    //TODO: noncontigous inputs/outputs
    const int64_t rowx = item_ct1.get_group(2) +
                     item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                     item_ct1.get_group(0) * item_ct1.get_group_range(2) *
                         item_ct1.get_group_range(1);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    if (mask != nullptr) {
        mask += (i11 * p.nb11 + i12 * p.nb12 + i13 * p.nb13) / sizeof(T);
    }
    dst  += int64_t(rowx)*ncols;

    const int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    float * buf_iw = (float *) dpct_local;

    // shared memory buffer to cache values between iterations:
    float *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_SIZE) : dst;
    float max_val = sinks ? sinks[i02] : -INFINITY;
#pragma unroll
    for (int64_t col0 = 0; col0 < ncols; col0 += block_size) {
        const int64_t col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_SIZE>(max_val);

    if (block_size > WARP_SIZE) {
        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        sycl::group_barrier(item_ct1.get_group());

        max_val = -INFINITY;
        for (int i = lane_id; i < nwarps; i += WARP_SIZE) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_SIZE>(max_val);
    }
    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int64_t col0 = 0; col0 < ncols; col0 += block_size) {
        const int64_t col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = sycl::native::exp(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    if (block_size > WARP_SIZE) {
        sycl::group_barrier(item_ct1.get_group());
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            for (int i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_SIZE] = 0.f;
            }
        }
        sycl::group_barrier(item_ct1.get_group());

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        sycl::group_barrier(item_ct1.get_group());

        tmp = buf_iw[lane_id];
        for (int i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum<WARP_SIZE>(tmp);
    }
    if (sinks) {
        tmp += sycl::native::exp(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int64_t col0 = 0; col0 < ncols; col0 += block_size) {
        const int64_t col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static void soft_max_back_f32(const float *grad, const float *dstf, float *dst,
                              const int64_t ncols, const float scale) {
    auto      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int tid      = item_ct1.get_local_id(2);
    const int64_t rowx = item_ct1.get_group(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int64_t col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum<WARP_SIZE>(dgf_dot);

    for (int64_t col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template <int... Ns, typename T>
static void launch_soft_max_kernels(const float *           x,
                                    const T *               mask,
                                    const float *           sinks,
                                    float *                 dst,
                                    const soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              block_dims,
                                    dpct::dim3              block_nums,
                                    size_t                  nbytes_shared)
{
    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);
        if (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        WARP_SIZE)]] {
                        soft_max_f32<true, ncols, block>(
                            x, mask, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_UNUSED(item_ct1);
                    });
            });
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<true, 0, 0>(
                        x, mask, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
    });
}

template <typename T>
static void soft_max_f32_sycl(const float *x, const T *mask,
                              const float *sinks, float *dst,
                              const soft_max_params &params,
                              dpct::queue_ptr stream, int device) {
    int nth = WARP_SIZE;
    int max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < max_block_size) nth *= 2;
    if (nth>max_block_size) nth = max_block_size;

    const dpct::dim3 block_dims(nth, 1, 1);
    const dpct::dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);

    const int id       = get_current_device_id();
    const size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo && ncols_x <= max_block_size) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, mask, sinks, dst, params, stream, block_dims, block_nums,
            nbytes_shared);
    } else {
        const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_low), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    soft_max_f32<false, 0, 0>(
                        x, mask, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_UNUSED(item_ct1);
                });
        });
    }
}

static void soft_max_back_f32_sycl(const float *   grad,
                                   const float *   dstf,
                                   float *         dst,
                                   const int64_t   ncols,
                                   const int64_t   nrows,
                                   const float     scale,
                                   dpct::queue_ptr stream) {
    const dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    const dpct::dim3 block_nums(nrows, 1, 1);

    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             soft_max_back_f32(grad, dstf, dst, ncols, scale);
                             GGML_UNUSED(item_ct1);
                         });
}

void ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[0] > 0);
    GGML_ASSERT(src0->ne[2] > 0);
    GGML_ASSERT(src0->ne[0] == dst->ne[0] &&
                src0->ne[1] == dst->ne[1] &&
                src0->ne[2] == dst->ne[2] &&
                src0->ne[3] == dst->ne[3]);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
    if (src1 != nullptr) {
        GGML_ASSERT(src1->ne[0] >= src0->ne[0]);
        GGML_ASSERT(src1->ne[1] >= src0->ne[1]);
        GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
        GGML_ASSERT(src1->nb[1] % ggml_type_size(src1->type) == 0);
        GGML_ASSERT(src1->nb[2] % ggml_type_size(src1->type) == 0);
        GGML_ASSERT(src1->nb[3] % ggml_type_size(src1->type) == 0);
    }
    if (src2 != nullptr) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src2));
        GGML_ASSERT(src2->ne[0] >= src0->ne[2]);
    }

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    std::memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    GGML_ASSERT(src0->ne[2] > 0);
    GGML_ASSERT(src0->ne[2] <= UINT32_MAX);
    const uint32_t n_head      = (uint32_t) src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    if (use_f16) {
        soft_max_f32_sycl(src0_d, (const sycl::half *)src1_d,
                          (const float *)src2_d, dst_d, params, stream,
                          ctx.device);
    } else {
        soft_max_f32_sycl(src0_d, (const float *)src1_d, (const float *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

void ggml_sycl_op_soft_max_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[0] > 0);
    GGML_ASSERT(src0->ne[0] == src1->ne[0] &&
                src0->ne[1] == src1->ne[1] &&
                src0->ne[2] == src1->ne[2] &&
                src0->ne[3] == src1->ne[3]);
    GGML_ASSERT(src0->ne[0] == dst->ne[0] &&
                src0->ne[1] == dst->ne[1] &&
                src0->ne[2] == dst->ne[2] &&
                src0->ne[3] == dst->ne[3]);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    std::memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    std::memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
