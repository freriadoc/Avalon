//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "conv.hpp"

static void conv_transpose_1d_kernel(
        const int64_t s0, const int64_t output_size,
        const int64_t src0_ne0, const int64_t src0_ne1, const int64_t src0_ne2,
        const int64_t src1_ne0, const int64_t dst_ne0,
        const float * src0, const float * src1, float * dst,
        const sycl::nd_item<3> & item_ct1) {
    const size_t global_index =
        item_ct1.get_local_id(2) +
        item_ct1.get_group(2) * item_ct1.get_local_range(2);

    if (global_index >= static_cast<size_t>(output_size)) {
        return;
    }

    // The output is laid out as [output_length, output_channels].
    const int64_t output_channel = static_cast<int64_t>(global_index) / dst_ne0;
    const int64_t output_position = static_cast<int64_t>(global_index) % dst_ne0;

    float accumulator = 0.0f;

    // src0: [kernel_length, output_channels, input_channels]
    // src1: [input_length, input_channels]
    for (int64_t c = 0; c < src0_ne2; ++c) {
        const int64_t kernel_offset =
            src0_ne0 * src0_ne1 * c + output_channel * src0_ne0;
        const int64_t input_offset = src1_ne0 * c;

        // Only input positions i satisfying
        //   0 <= output_position - i * s0 < src0_ne0
        // contribute to this output position.
        //
        // This avoids scanning the entire input sequence for every output.
        const int64_t first_i =
            output_position >= src0_ne0
                ? (output_position - src0_ne0) / s0 + 1
                : 0;

        const int64_t last_i_unclamped = output_position / s0;
        const int64_t last_i =
            last_i_unclamped < src1_ne0
                ? last_i_unclamped
                : src1_ne0 - 1;

        for (int64_t i = first_i; i <= last_i; ++i) {
            const int64_t weight_idx = output_position - i * s0;

            accumulator +=
                src0[kernel_offset + weight_idx] *
                src1[input_offset + i];
        }
    }

    dst[global_index] = accumulator;
}

static void conv_transpose_1d_f32_f32_sycl(
        const int64_t s0, const int64_t output_size,
        const int64_t src0_ne0, const int64_t src0_ne1, const int64_t src0_ne2,
        const int64_t src1_ne0, const int64_t dst_ne0,
        const float * src0, const float * src1, float * dst,
        const queue_ptr & stream) {
    const size_t block_size = SYCL_CONV_TRANPOSE_1D_BLOCK_SIZE;
    const size_t output_size_size_t = static_cast<size_t>(output_size);
    const size_t num_blocks =
        output_size_size_t / block_size +
        (output_size_size_t % block_size != 0);

    const sycl::range<3> block_dims(1, 1, block_size);
    const sycl::range<3> block_nums(1, 1, num_blocks);

    stream->parallel_for(
        sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_size,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

void ggml_sycl_op_conv_transpose_1d(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    // Match the GGML conv_transpose_1d contract:
    // src0 = [kernel_length, output_channels, input_channels, 1]
    // src1 = [input_length, input_channels, 1, 1]
    // dst  = [output_length, output_channels, 1, 1]
    GGML_ASSERT(src0->ne[0] > 0);
    GGML_ASSERT(src0->ne[1] > 0);
    GGML_ASSERT(src0->ne[2] > 0);
    GGML_ASSERT(src0->ne[3] == 1);

    GGML_ASSERT(ggml_is_matrix(src1));
    GGML_ASSERT(src1->ne[0] > 0);
    GGML_ASSERT(src1->ne[1] == src0->ne[2]);

    GGML_ASSERT(dst->ne[0] > 0);
    GGML_ASSERT(dst->ne[1] == src0->ne[1]);
    GGML_ASSERT(dst->ne[2] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    const int32_t * opts = reinterpret_cast<const int32_t *>(dst->op_params);
    const int64_t s0 = opts[0];

    // This SYCL kernel implements the currently supported GGML case:
    // zero padding and unit dilation.
    GGML_ASSERT(s0 > 0);
    GGML_ASSERT(opts[1] == 0);
    GGML_ASSERT(opts[2] == 1);

    const int64_t expected_output_length =
        (src1->ne[0] - 1) * s0 + src0->ne[0];
    GGML_ASSERT(dst->ne[0] == expected_output_length);

    const int64_t output_size = ggml_nelements(dst);
    GGML_ASSERT(output_size > 0);

    const float * src0_d = static_cast<const float *>(src0->data);
    const float * src1_d = static_cast<const float *>(src1->data);
    float * dst_d = static_cast<float *>(dst->data);

    dpct::queue_ptr stream = ctx.stream();

    conv_transpose_1d_f32_f32_sycl(
        s0, output_size,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}
