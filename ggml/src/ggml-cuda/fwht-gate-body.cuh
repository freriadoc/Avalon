#pragma once

// The rotation back of a row of N values (fwht 64 of qwen4exp's attention output) times the sigmoid of its gate, with
// the q8_1 of the product (fwht.cu: fwht_gate_cuda), by one warp; the row may be in shared memory (the flash
// attention's combination writes its result there: fattn-common.cuh, flash_attn_combine_results_gate)

#include "common.cuh"

struct fwht_gate_args {
    const char * gate; int64_t g_ne0, g_ne1, g_ne2; size_t g_nb0, g_nb1, g_nb2, g_nb3;
    float * dst; block_q8_1 * dst_q8_1; float scale;
    int64_t n_dst;   // values of dst
};

// row r (values r*N .. r*N + N - 1 of the flattened tensor) read from row
template <int N>
static __device__ __forceinline__ void fwht_gate_row(const float * row, const int64_t r, const int lane,
                                                     const fwht_gate_args & a) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == QK8_1, "a q8_1 block per register index");
    static constexpr int el_w = N / warp_size;
    float reg[el_w];

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = row[i * warp_size + lane] * a.scale;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int64_t e = r * N + i * warp_size + lane;
        int64_t t = e;
        const int64_t i0 = t % a.g_ne0; t /= a.g_ne0;
        const int64_t i1 = t % a.g_ne1; t /= a.g_ne1;
        const int64_t i2 = t % a.g_ne2; t /= a.g_ne2;
        const float g = *(const float *) (a.gate + i0*a.g_nb0 + i1*a.g_nb1 + i2*a.g_nb2 + t*a.g_nb3);
        const float v = (1.0f / (1.0f + expf(-g))) * reg[i];
        a.dst[e] = v;
        if (a.dst_q8_1) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float sum  = warp_reduce_sum<QK8_1>(v);
            const float d    = amax / 127.0f;
            a.dst_q8_1[e / QK8_1].qs[e % QK8_1] = amax == 0.0f ? 0 : roundf(v / d);
            if (lane == 0) {
                a.dst_q8_1[e / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}
