#pragma once

// the rotation of 64-wide rows written into a quantized cache (fwht.cu), its body shared with the multi-task kernel
// (mk.cu): one warp per row

#include "common.cuh"
#include "cpy-utils.cuh"

struct fwht_set_rows_args {
    const float * src; const int64_t * idx; void * dst;
    int64_t rows_per_set, dst_row_blocks, n_rows;
    float scale;
    bool q4_0;   // the destination's type: Q4_0, else Q8_0
};

// row r by the warp whose lane this is
template <int N, typename B>
static __device__ __forceinline__ void fwht_set_rows_body(const fwht_set_rows_args & a, const int64_t r, const int lane) {
    const int64_t n_rows = a.n_rows;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32 && N % 32 == 0, "a block per register index");

    if (r >= n_rows) {
        return;
    }
    const float * src = a.src + r * N;
    const int64_t * idx = a.idx;
    B * dst = (B *) a.dst;
    const int64_t rows_per_set = a.rows_per_set, dst_row_blocks = a.dst_row_blocks;
    const float scale = a.scale;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
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

    B * y = dst + idx[r / rows_per_set] * dst_row_blocks + (r % rows_per_set) * el_w;
    if constexpr (std::is_same_v<B, block_q4_0>) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            quantize_warp_q4_0(reg[i], y + i, lane);
        }
    } else {
        quantize_warp_q8_0_n<el_w>(reg, y, lane, el_w);
    }
}
