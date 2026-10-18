#pragma once

// The bodies of two kernels of qwen4exp's indexer keys, shared with the multi-task kernel (mk.cu): the raw key padded
// with zeros into its Q8_0 cache row (k_set_rows_quant of ggml_cuda_op_set_rows_padded, set-rows.cu), and the mean of
// a pool's member keys read back from the cache (k_get_rows_mean, getrows.cu). Same arithmetic.

#include "common.cuh"
#include "cpy-utils.cuh"
#include "dequantize.cuh"

struct set_rows_q8_args {
    const float * src0; const void * idx; block_q8_0 * dst;
    int64_t ne_total;
    int64_t s01, s02, s03, s10, s11, s12, s1, s2, s3;
    uint3 ne00, ne01, ne02, ne11_fd, ne12_fd;
    int64_t ne_valid;
    bool idx64;
};

// thread i: block i of the rows (k_set_rows_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>)
static __device__ __forceinline__ void set_rows_q8_body(const set_rows_q8_args & a, const int64_t i) {
    if (i >= a.ne_total) {
        return;
    }
    const int64_t i_base = i * QK8_0;
    uint32_t tmp = (uint32_t) i_base;
    uint2 div_mod;
    div_mod = fast_div_modulo(tmp, a.ne00);
    const int64_t i00 = div_mod.y;
    tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, a.ne01);
    const int64_t i01 = div_mod.y;
    tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, a.ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t) i03, a.ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, a.ne11_fd);
    const int64_t i10 = i01;
    const int64_t off = i10*a.s10 + i11*a.s11 + i12*a.s12;
    const int64_t dst_row = a.idx64 ? ((const int64_t *) a.idx)[off] : (int64_t) ((const int32_t *) a.idx)[off];
    const float * src0_row = a.src0 + i01*a.s01 + i02*a.s02 + i03*a.s03;
    block_q8_0 * dst_block = a.dst + (dst_row*a.s1 + i02*a.s2 + i03*a.s3) / sizeof(block_q8_0) + i00 / QK8_0;
    if (i00 >= a.ne_valid) {
        float zero[QK8_0] = {};
        quantize_f32_q8_0_block(zero, dst_block);
        return;
    }
    quantize_f32_q8_0_block(src0_row + i00, dst_block);
}

struct get_rows_mean_args {
    const void * src0; const int32_t * ids; float * dst;
    int64_t ne00; size_t nb01;
    int n_group; float scale, bias;
    int64_t n_out;
    int type;   // GGML_TYPE_Q8_0 or GGML_TYPE_Q4_0
};

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __device__ __forceinline__ void get_rows_mean_pair(const get_rows_mean_args & a, const int64_t p, const int64_t i00) {
    const int ib   =  i00/qk;
    const int iqs  = (i00%qk)/qr;
    const int iybs = i00 - i00%qk;
    const int y_offset = qr == 1 ? 1 : qk/2;
    float2 sum;
    dequantize_kernel((const char *) a.src0 + (int64_t) a.ids[p*a.n_group]*a.nb01, ib, iqs, sum);
    for (int k = 1; k < a.n_group; ++k) {
        float2 v;
        dequantize_kernel((const char *) a.src0 + (int64_t) a.ids[p*a.n_group + k]*a.nb01, ib, iqs, v);
        sum.x += v.x;
        sum.y += v.y;
    }
    float * dst_row = a.dst + p*a.ne00;
    dst_row[iybs + iqs + 0]        = sum.x*a.scale + a.bias;
    dst_row[iybs + iqs + y_offset] = sum.y*a.scale + a.bias;
}

// thread e: the pair of values e % (ne00/2) of pool e / (ne00/2)
static __device__ __forceinline__ void get_rows_mean_body(const get_rows_mean_args & a, const int64_t e) {
    const int64_t half = a.ne00 / 2;
    const int64_t p = e / half;
    if (p >= a.n_out) {
        return;
    }
    const int64_t i00 = 2*(e % half);
    if (a.type == GGML_TYPE_Q8_0) {
        get_rows_mean_pair<QK8_0, QR8_0, dequantize_q8_0>(a, p, i00);
    } else {
        get_rows_mean_pair<QK4_0, QR4_0, dequantize_q4_0>(a, p, i00);
    }
}
