#include "ssm_conv.hpp"
#include "common.hpp"
#include "element_wise.hpp"

#include <cstdio>

using namespace sycl;

// One output element of the convolution. DC is a compile-time constant (0 keeps
// the runtime loop). The unfused launch passes false/nullptr for the epilogue.
template <int DC>
static __dpct_inline__ void ssm_conv_element(
    size_t idx,
    const float * src_data,
    const float * weights,
    float * dst_data,
    int64_t d_conv,
    int64_t d_inner,
    int64_t n_t,
    int64_t src_stride_inner,
    int64_t src_stride_seq,
    int64_t weights_stride_inner,
    int64_t dst_stride_channel,
    int64_t dst_stride_token,
    int64_t dst_stride_seq,
    bool apply_silu,
    const float * bias) {
    const size_t tokens_per_seq = static_cast<size_t>(n_t);
    const size_t channels_per_seq = static_cast<size_t>(d_inner);

    const int64_t token = static_cast<int64_t>(idx % tokens_per_seq);
    const int64_t channel =
        static_cast<int64_t>((idx / tokens_per_seq) % channels_per_seq);
    const int64_t seq =
        static_cast<int64_t>(idx / (tokens_per_seq * channels_per_seq));

    const float * s =
        src_data
        + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
        + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
        + static_cast<size_t>(token);

    const float * c =
        weights + static_cast<size_t>(channel) * static_cast<size_t>(weights_stride_inner);

    float sumf = 0.0f;
    if constexpr (DC > 0) {
#pragma unroll
        for (int i0 = 0; i0 < DC; ++i0) {
            sumf += s[i0] * c[i0];
        }
    } else {
        for (int64_t i0 = 0; i0 < d_conv; ++i0) {
            sumf += s[i0] * c[i0];
        }
    }

    // The fused ADD node broadcasts a contiguous 1-D channel bias.
    if (bias != nullptr) {
        sumf += bias[channel];
    }

    const size_t dst_idx =
        static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq)
        + static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token)
        + static_cast<size_t>(channel) * static_cast<size_t>(dst_stride_channel);

    dst_data[dst_idx] = apply_silu ? op_silu(sumf) : sumf;
}

// FUSED=false retains the unfused kernel argument list and lets the compiler
// eliminate the bias/SiLU epilogue.
template <int DC, bool FUSED>
static void kernel_ssm_conv_impl(
    queue & q,
    const float * src_data,
    const float * weights,
    float * dst_data,
    int64_t d_conv,
    int64_t d_inner,
    int64_t n_t,
    int64_t n_s,
    int64_t src_stride_inner,
    int64_t src_stride_seq,
    int64_t weights_stride_inner,
    int64_t dst_stride_channel,
    int64_t dst_stride_token,
    int64_t dst_stride_seq,
    bool apply_silu,
    const float * bias) {
    const size_t total_work =
        static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    constexpr size_t work_group_size = 256;
    const size_t num_work_groups =
        total_work / work_group_size + (total_work % work_group_size != 0);

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    if constexpr (FUSED) {
        q.submit([&](handler & h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(
                        idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                        src_stride_inner, src_stride_seq, weights_stride_inner,
                        dst_stride_channel, dst_stride_token, dst_stride_seq,
                        apply_silu, bias);
                });
        });
    } else {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler & h) {
            h.parallel_for(
                nd_range<1>(global_range, local_range),
                [=](nd_item<1> item) {
                    const size_t idx = item.get_global_id(0);
                    if (idx >= total_work) {
                        return;
                    }

                    ssm_conv_element<DC>(
                        idx, src_data, weights, dst_data, d_conv, d_inner, n_t,
                        src_stride_inner, src_stride_seq, weights_stride_inner,
                        dst_stride_channel, dst_stride_token, dst_stride_seq,
                        false, nullptr);
                });
        });
    }
}

// Local-memory transpose tile. The extra column avoids a power-of-two row
// stride, which can reduce bank conflicts on devices with 32-bank local memory.
template <int DC, int TT, int TC, int WG>
static __dpct_inline__ void ssm_conv_tile(
    nd_item<1> it,
    local_accessor<float, 1> tile,
    const float * src_data,
    const float * weights,
    float * dst_data,
    int64_t n_t,
    size_t nt_tiles,
    size_t nc_tiles,
    int64_t src_stride_inner,
    int64_t src_stride_seq,
    int64_t weights_stride_inner,
    int64_t dst_stride_channel,
    int64_t dst_stride_token,
    int64_t dst_stride_seq,
    bool apply_silu,
    const float * bias) {
    const int lid = static_cast<int>(it.get_local_id(0));
    const size_t group = it.get_group(0);

    const size_t tt = group % nt_tiles;
    const size_t ct = (group / nt_tiles) % nc_tiles;
    const size_t seq = group / (nt_tiles * nc_tiles);

    const size_t t0 = tt * TT;
    const size_t c0 = ct * TC;

    const int ti = lid % TT;
    const int cj = lid / TT;

#pragma unroll
    for (int r = 0; r < TC / (WG / TT); ++r) {
        const int c = cj + r * (WG / TT);
        const size_t channel = c0 + static_cast<size_t>(c);
        const size_t token = t0 + static_cast<size_t>(ti);

        float sumf = 0.0f;
        if (token < static_cast<size_t>(n_t)) {
            const float * s =
                src_data + seq * static_cast<size_t>(src_stride_seq)
                + channel * static_cast<size_t>(src_stride_inner) + token;
            const float * cw =
                weights + channel * static_cast<size_t>(weights_stride_inner);

#pragma unroll
            for (int i = 0; i < DC; ++i) {
                sumf += s[i] * cw[i];
            }

            if (bias != nullptr) {
                sumf += bias[channel];
            }
            if (apply_silu) {
                sumf = op_silu(sumf);
            }
        }

        tile[c * (TT + 1) + ti] = sumf;
    }

    // All work-items must participate: this synchronizes tile production
    // before the transposed store phase reads local memory.
    group_barrier(it.get_group());

    const int cc = lid % TC;
    const int tj = lid / TC;

#pragma unroll
    for (int r = 0; r < TT / (WG / TC); ++r) {
        const int t = tj + r * (WG / TC);
        const size_t token = t0 + static_cast<size_t>(t);
        const size_t channel = c0 + static_cast<size_t>(cc);

        if (token < static_cast<size_t>(n_t)) {
            const size_t dst_idx =
                seq * static_cast<size_t>(dst_stride_seq)
                + token * static_cast<size_t>(dst_stride_token)
                + channel * static_cast<size_t>(dst_stride_channel);

            dst_data[dst_idx] = tile[cc * (TT + 1) + t];
        }
    }
}

// FUSED split mirrors kernel_ssm_conv_impl. For n_t >= 32, bias/SiLU work is
// amortized across the tile.
template <int DC, bool FUSED>
static void kernel_ssm_conv_tiled(
    queue & q,
    const float * src_data,
    const float * weights,
    float * dst_data,
    int64_t d_inner,
    int64_t n_t,
    int64_t n_s,
    int64_t src_stride_inner,
    int64_t src_stride_seq,
    int64_t weights_stride_inner,
    int64_t dst_stride_channel,
    int64_t dst_stride_token,
    int64_t dst_stride_seq,
    bool apply_silu,
    const float * bias) {
    constexpr int TT = 32;
    constexpr int TC = 32;
    constexpr int WG = 256;

    const size_t nt_tiles =
        static_cast<size_t>((n_t + TT - 1) / TT);
    const size_t nc_tiles = static_cast<size_t>(d_inner / TC);
    const size_t groups = nt_tiles * nc_tiles * static_cast<size_t>(n_s);

    if constexpr (FUSED) {
        q.submit([&](handler & h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(
                nd_range<1>(range<1>(groups * WG), range<1>(WG)),
                [=](nd_item<1> it) {
                    ssm_conv_tile<DC, TT, TC, WG>(
                        it, tile, src_data, weights, dst_data, n_t, nt_tiles, nc_tiles,
                        src_stride_inner, src_stride_seq, weights_stride_inner,
                        dst_stride_channel, dst_stride_token, dst_stride_seq,
                        apply_silu, bias);
                });
        });
    } else {
        GGML_UNUSED(apply_silu);
        GGML_UNUSED(bias);

        q.submit([&](handler & h) {
            local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
            h.parallel_for(
                nd_range<1>(range<1>(groups * WG), range<1>(WG)),
                [=](nd_item<1> it) {
                    ssm_conv_tile<DC, TT, TC, WG>(
                        it, tile, src_data, weights, dst_data, n_t, nt_tiles, nc_tiles,
                        src_stride_inner, src_stride_seq, weights_stride_inner,
                        dst_stride_channel, dst_stride_token, dst_stride_seq,
                        false, nullptr);
                });
        });
    }
}

static void kernel_ssm_conv(
    queue & q,
    const float * src_data,
    const float * weights,
    float * dst_data,
    int64_t d_conv,
    int64_t d_inner,
    int64_t n_t,
    int64_t n_s,
    int64_t src_stride_inner,
    int64_t src_stride_seq,
    int64_t weights_stride_inner,
    int64_t dst_stride_channel,
    int64_t dst_stride_token,
    int64_t dst_stride_seq,
    bool apply_silu,
    const float * bias) {
    const bool fused = apply_silu || bias != nullptr;

    // The tiled transpose is selected only when the channel tile is exact and
    // the token dimension is large enough to amortize its synchronization.
    if (d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        if (fused) {
            kernel_ssm_conv_tiled<4, true>(
                q, src_data, weights, dst_data, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, weights_stride_inner,
                dst_stride_channel, dst_stride_token, dst_stride_seq,
                apply_silu, bias);
        } else {
            kernel_ssm_conv_tiled<4, false>(
                q, src_data, weights, dst_data, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, weights_stride_inner,
                dst_stride_channel, dst_stride_token, dst_stride_seq,
                apply_silu, bias);
        }
        return;
    }

    if (d_conv == 4) {
        if (fused) {
            kernel_ssm_conv_impl<4, true>(
                q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, weights_stride_inner,
                dst_stride_channel, dst_stride_token, dst_stride_seq,
                apply_silu, bias);
        } else {
            kernel_ssm_conv_impl<4, false>(
                q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
                src_stride_inner, src_stride_seq, weights_stride_inner,
                dst_stride_channel, dst_stride_token, dst_stride_seq,
                apply_silu, bias);
        }
        return;
    }

    if (fused) {
        kernel_ssm_conv_impl<0, true>(
            q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
            src_stride_inner, src_stride_seq, weights_stride_inner,
            dst_stride_channel, dst_stride_token, dst_stride_seq,
            apply_silu, bias);
    } else {
        kernel_ssm_conv_impl<0, false>(
            q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,
            src_stride_inner, src_stride_seq, weights_stride_inner,
            dst_stride_channel, dst_stride_token, dst_stride_seq,
            apply_silu, bias);
    }
}

inline void ggml_sycl_op_ssm_conv(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * dst,
    ggml_tensor * silu_dst = nullptr,
    const float * bias = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(bias == nullptr || silu_dst != nullptr);

    const int64_t d_conv = src1->ne[0];
    const int64_t d_inner = src0->ne[1];
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];

    GGML_ASSERT(d_conv > 0);
    GGML_ASSERT(d_inner > 0);
    GGML_ASSERT(n_t > 0);
    GGML_ASSERT(n_s > 0);

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src0->ne[2] == n_s);
    GGML_ASSERT(src0->ne[3] == 1);

    GGML_ASSERT(src1->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[2] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[3] == 1);

    ggml_tensor * output = silu_dst != nullptr ? silu_dst : dst;
    if (silu_dst != nullptr) {
        GGML_ASSERT(ggml_are_same_shape(dst, silu_dst));
        GGML_ASSERT(silu_dst->type == GGML_TYPE_F32);
    }

    // The kernels address float elements using tensor byte strides converted
    // to element strides. Dimension 0 must therefore be a scalar float.
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(output->nb[0] % sizeof(float) == 0);
    GGML_ASSERT(src0->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(src0->nb[2] % sizeof(float) == 0);
    GGML_ASSERT(src1->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(output->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(output->nb[2] % sizeof(float) == 0);

    const int64_t src_stride_inner =
        static_cast<int64_t>(src0->nb[1] / sizeof(float));
    const int64_t src_stride_seq =
        static_cast<int64_t>(src0->nb[2] / sizeof(float));
    const int64_t weights_stride_inner =
        static_cast<int64_t>(src1->nb[1] / sizeof(float));

    const int64_t dst_stride_channel =
        static_cast<int64_t>(output->nb[0] / sizeof(float));
    const int64_t dst_stride_token =
        static_cast<int64_t>(output->nb[1] / sizeof(float));
    const int64_t dst_stride_seq =
        static_cast<int64_t>(output->nb[2] / sizeof(float));

    try {
        queue * q = ctx.stream();

        const float * src_data = static_cast<const float *>(src0->data);
        const float * weights = static_cast<const float *>(src1->data);
        const bool apply_silu = silu_dst != nullptr;
        float * dst_data = static_cast<float *>(output->data);

        GGML_ASSERT(src_data != nullptr);
        GGML_ASSERT(weights != nullptr);
        GGML_ASSERT(dst_data != nullptr);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            src_stride_inner,
            src_stride_seq,
            weights_stride_inner,
            dst_stride_channel,
            dst_stride_token,
            dst_stride_seq,
            apply_silu,
            bias);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + ADD + SiLU: write silu(conv(x) + b) directly into silu_dst,
// avoiding a separate SiLU launch and an intermediate convolution-output round trip.
void ggml_sycl_ssm_conv_fused(
    ggml_backend_sycl_context & ctx,
    ggml_tensor * dst,
    ggml_tensor * add,
    ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(silu_dst != nullptr);
    GGML_ASSERT(ggml_are_same_shape(dst, silu_dst));
    GGML_ASSERT(silu_dst->type == GGML_TYPE_F32);

    const float * bias = nullptr;
    if (add != nullptr) {
        GGML_ASSERT(add->src[0] == dst || add->src[1] == dst);

        const ggml_tensor * bias_t =
            (add->src[0] == dst) ? add->src[1] : add->src[0];

        GGML_ASSERT(bias_t != nullptr);
        GGML_ASSERT(bias_t->type == GGML_TYPE_F32);
        GGML_ASSERT(bias_t->ne[0] == dst->ne[0]);
        GGML_ASSERT(bias_t->ne[1] == 1);
        GGML_ASSERT(bias_t->ne[2] == 1);
        GGML_ASSERT(bias_t->ne[3] == 1);
        GGML_ASSERT(ggml_is_contiguous(bias_t));

        bias = static_cast<const float *>(bias_t->data);
        GGML_ASSERT(bias != nullptr);
    }

    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, bias);
}
