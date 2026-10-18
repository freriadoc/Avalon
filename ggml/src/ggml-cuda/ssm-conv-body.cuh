#pragma once

// ssm_conv with its state in the recurrent cache (ssm-conv.cu), its body shared with the multi-task kernel (mk.cu):
// blocks of 128 threads, a thread per channel

#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

struct ssm_conv_cached_args {
    const float * state; const int32_t * ids; int64_t state_row;
    const char * x; int64_t x_nb0, x_nb1, x_nb2;
    const float * w; int64_t w_nb1; float * dst;
    int64_t n_ch, n_t; bool apply_silu;
    ggml_cuda_ssm_conv_slots slots;
    float * qk_out; int64_t n_qk; float qk_eps, qk_scale;
    int n_s;   // sequences (the grid's y)
};

// block (bx, s) of 128 threads, tid < 128
template <int d_conv, int max_t>
static __device__ __forceinline__ void ssm_conv_cached_body(const ssm_conv_cached_args & a, const int64_t bx, const int64_t s,
        const int tid) {
    const float * state = a.state; const int32_t * ids = a.ids; const int64_t state_row = a.state_row;
    const char * x = a.x; const int64_t x_nb0 = a.x_nb0, x_nb1 = a.x_nb1, x_nb2 = a.x_nb2;
    const float * w = a.w; const int64_t w_nb1 = a.w_nb1; float * dst = a.dst;
    const int64_t n_ch = a.n_ch, n_t = a.n_t; const bool apply_silu = a.apply_silu;
    const ggml_cuda_ssm_conv_slots & slots = a.slots;
    float * qk_out = a.qk_out; const int64_t n_qk = a.n_qk; const float qk_eps = a.qk_eps, qk_scale = a.qk_scale;
    const int64_t c = (int64_t) bx * 128 + tid;
    // a block of the l2-normed channels is one head (128 threads): every thread of it is a channel
    const bool qk = qk_out != nullptr && (int64_t) bx * 128 < n_qk;
    if (c >= n_ch) {
        return;
    }
    __shared__ float qk_sum[32];
    float win[d_conv - 1 + max_t];
    const float * st = state + (ids != nullptr ? (int64_t) ids[s] : s) * state_row + c * (d_conv - 1);
#pragma unroll
    for (int k = 0; k < d_conv - 1; k++) {
        win[k] = st[k];
    }
    for (int64_t t = 0; t < n_t; t++) {
        win[d_conv - 1 + t] = *(const float *) (x + t * x_nb0 + c * x_nb1 + s * x_nb2);
    }
    float wc[d_conv];
#pragma unroll
    for (int k = 0; k < d_conv; k++) {
        wc[k] = w[c * (w_nb1 / sizeof(float)) + k];
    }
    for (int64_t t = 0; t < n_t; t++) {
        float sumf = 0.0f;
#pragma unroll
        for (int k = 0; k < d_conv; k++) {
            sumf += win[t + k] * wc[k];
        }
        sumf += 0.0f;   // the bias term of ssm_conv_f32, absent here: same rounding
        const float y = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
        dst[(s * n_t + t) * n_ch + c] = y;
        if (qk) {
            // rms_norm_f32 with 256 threads per row of 128: these 4 warps' sums at the same lanes, the other 4 zero
            float sq = block_reduce<block_reduce_method::SUM, 128>(y * y, qk_sum);
            const float scale = rsqrtf(sq / 128 + qk_eps);
            qk_out[(s * n_t + t) * n_qk + c] = qk_scale * (scale * y);
            __syncthreads();   // qk_sum is reused by the next token
        }
    }
    for (int k = 0; k < slots.n; k++) {
        float * row = (float *) (slots.dst[k] + s * slots.nb1[k]) + c * (d_conv - 1);
        for (int j = 0; j < d_conv - 1; j++) {
            row[j] = win[slots.tail_col[k] + j];
        }
    }
}
