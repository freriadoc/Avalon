#pragma once

// rms_norm(x) * w * sigmoid(gate) per row (norm.cu), its body shared with the multi-task kernel (mk.cu): a block of
// block_size threads per row

#include "common.cuh"
#include "convert.cuh"

struct rms_norm_gate_args {
    const float * x; const float * w; const float * gate; float * dst;
    int ncols;
    int64_t sx1, sx2, sx3, sg1, sg2, sg3;
    float eps;
    block_q8_1 * dst_q8_1;
    int nrows, nchannels, nsamples;   // the grid
    bool dst_unused;   // with the image: dst unwritten (its product reads only the image)
};

// has_img: also the image of dst for the cuBLAS products after it (a prompt batch only; out of the arguments and of the
// code of the decode's kernels, which it made slower: 32 -> 40 VGPR, 1.96 -> 2.40 us per launch on the RX 7900 XT)
template <int block_size, bool has_img = false>
static __device__ __forceinline__ void rms_norm_mul_sigmoid_gate_body(const rms_norm_gate_args & a, const int row,
        const int channel, const int sample, const int tid, const ggml_cuda_img img = ggml_cuda_img()) {
    const float * x = a.x; const float * w = a.w; const float * gate = a.gate; float * dst = a.dst;
    const int ncols = a.ncols;
    const int64_t sx1 = a.sx1, sx2 = a.sx2, sx3 = a.sx3, sg1 = a.sg1, sg2 = a.sg2, sg3 = a.sg3;
    const float eps = a.eps; block_q8_1 * dst_q8_1 = a.dst_q8_1;
    const int nrows     = a.nrows;
    const int nchannels = a.nchannels;

    x    += sample*sx3 + channel*sx2 + row*sx1;
    gate += sample*sg3 + channel*sg2 + row*sg1;
    const int64_t dst_off = ((int64_t) (sample*nchannels + channel)*nrows + row)*ncols;
    dst  += dst_off;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const float g = 1.0f / (1.0f + expf(-gate[col]));
        const float v = scale * x[col] * w[col] * g;
        if constexpr (has_img) {
            if (!a.dst_unused) {
                dst[col] = v;
            }
            img.store1(dst_off + col, v);   // q8_1: whole warps (ncols % 128 == 0, ggml_cuda_f16_image_target)
        } else {
            dst[col] = v;
        }
        if (dst_q8_1) {
            // the q8_1 block of the 32 values of this warp, as quantize_q8_1 writes it (lane = index in the block;
            // ncols % QK8_1 == 0, so every warp is whole)
            const float amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float sum  = warp_reduce_sum<QK8_1>(v);
            const float d    = amax / 127.0f;
            const int64_t i  = dst_off + col;
            dst_q8_1[i / QK8_1].qs[i % QK8_1] = amax == 0.0f ? 0 : roundf(v / d);
            if (i % QK8_1 == 0) {
                dst_q8_1[i / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}
