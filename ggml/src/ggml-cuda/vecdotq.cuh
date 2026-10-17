#pragma once

#include "common.cuh"

#include <cstdint>

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

// Weights read once per token (the main model's large matrices while decoding): with non-temporal loads (HIP: slc
// dlc) they no longer evict from the L2 and the infinity cache the code and the data of the other kernels. One 32-bit
// load at a 2-byte aligned address (the hardware allows it; two 16-bit non-temporal loads would be merged by the
// compiler without the hint). The same values.
static __device__ __forceinline__ int get_int_b2_nt(const void * x, const int & i32) {
#if defined(GGML_USE_HIP)
    return __builtin_nontemporal_load((const int *) ((const uint16_t *) x + 2*i32));
#else
    return get_int_b2(x, i32);
#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ half load_half_nt(const half * x) {
#if defined(GGML_USE_HIP)
    return __ushort_as_half(__builtin_nontemporal_load((const unsigned short *) x));
#else
    return *x;
#endif // defined(GGML_USE_HIP)
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
#if defined(GGML_USE_HIP)
    // Load the 16-byte table into four 32-bit unsigned integers.
    const uint32_t *values = (const uint32_t *)table;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    // Perform lookups in the lower half of the table (indices 0-7).
    uint32_t v_even_low = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    uint32_t v_odd_low = __builtin_amdgcn_perm(values[1], values[0], q_odd & 0x07070707);

    // Perform lookups in the upper half of the table (indices 8-15).
    uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    uint32_t v_odd_high = __builtin_amdgcn_perm(values[3], values[2], q_odd & 0x07070707);

    // Select between the low and high results based on the MSB of each index nibble.
    uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    uint32_t res_x = __builtin_amdgcn_perm(v_even_high, v_even_low, mask_even);
    uint32_t mask_odd = 0x03020100 | ((q_odd & 0x08080808) >> 1);
    uint32_t res_y = __builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd);

    return make_int2(res_x, res_y);
#elif !defined(GGML_USE_MUSA)
    // CUDA does not have an instruction for selecting bytes with 4 bit indices.
    // However, __byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead.
    const uint32_t * table32 = (const uint32_t *) table;

    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Therefore, do 2 iterations over the 32 bits in q4 with 0 and 16 shift.
    // To handle the fourth bit, first call _byte_perm both for the low and the high 64 bit of table, using the low 3 bits.
    // Then, call __byte_perm again to select from the low and high bytes based on the fourth bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;

        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }

    // tmp contains the bytes from tyble in the same order as the 4 bit indices in q4.
    // However, for the result we need ints with all even/odd 4 bit indices in q4.
    // Therefore, 2 more calls to __byte_perm to put the bytes in the correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#else
    // Generic implementation.
    const int      q0_32  = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t * q0_8   = (const int8_t *) &q0_32;
    const char4    val0_8 = make_char4(
        table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int      q1_32  = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t * q1_8   = (const int8_t *) &q1_32;
    const char4    val1_8 = make_char4(
        table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    return make_int2(*((const int *) &val0_8), *((const int *) &val1_8));
#endif
}

static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    // with xor we can "correct" the bit instead of having to mask
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    // broadcast over uint to allow for 0x08040201 / 0x80402010 as selectors
    return s * 0x01010101;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q1_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q1_0_Q8_1_MMQ  4  // Q1_0 has 128 bits (4 ints) per block

#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism
#define VDR_Q2_0_Q8_1_MMQ  2  // Q2_0 group 64: 128 bits (4 ints) per block, 2 32-element chunks

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef FAST_FP16_AVAILABLE
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // FAST_FP16_AVAILABLE

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_16_q8_1_impl(
    const int * v, const int * u, const float * d8_0, const float & d8_1) {

    float sumf = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < vdr; i0 += QI8_0/2) {
        int sumi = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_0/2; ++i) {
            // SIMD dot product of quantized values
            sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
        }

        sumf += d8_0[i0/(QI8_0/2)]*sumi;
    }

    return d8_1*sumf;
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_mxfp4_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b1(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

static __device__ __forceinline__ float vec_dot_nvfp4_q8_1(
                                        const void * __restrict__ vbq,
                                        const block_q8_1 * __restrict__ bq8_1,
                                        const int32_t & kbx,
                                        const int32_t & iqs) {

    const block_nvfp4 * bq4 = (const block_nvfp4 *) vbq + kbx;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        const int32_t iqs0 = iqs + 2*i;
        const int32_t iqs1 = iqs0 + 1;
        const int32_t is = iqs0 >> 1;
        const int2 v0 = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        const int2 v1 = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        const block_q8_1 * bq8 = bq8_1 + (is >> 1);
        const int32_t i8 = ((is & 1) << 2);

        int sumi = ggml_cuda_dp4a(v0.x, get_int_b4(bq8->qs, i8 + 0), 0);
        sumi = ggml_cuda_dp4a(v0.y, get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi = ggml_cuda_dp4a(v1.x, get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi = ggml_cuda_dp4a(v1.y, get_int_b4(bq8->qs, i8 + 3), sumi);

        const float d = ggml_cuda_ue4m3_to_fp32(bq4->d[is]) * __low2float(bq8->ds);
        sum += d * float(sumi);
    }

    return sum;
}
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  4

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous v/x + u/y values
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8, const half2 * s8) {

    float sumf    = 0.0f;
    float sumf_d8 = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QR2_K*VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        const float2 dm2f0 = __half22float2(dm2[i0/(QI8_1/2) + 0]);
        int sumi_d0 = 0;

        const float2 dm2f1 = __half22float2(dm2[i0/(QI8_1/2) + 1]);
        int sumi_d1 = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d0 = ggml_cuda_dp4a(v[i], u[i], sumi_d0);
        }
        sumf_d8 += dm2f0.x * sumi_d0;

#pragma unroll
        for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
            sumi_d1 = ggml_cuda_dp4a(v[i], u[i], sumi_d1);
        }
        sumf_d8 += dm2f1.x * sumi_d1;

        if (i0/QI8_1 < ns8) {
            const float2 s8f = __half22float2(s8[i0/QI8_1]);
            sumf -= dm2f0.y*s8f.x;
            sumf -= dm2f1.y*s8f.y;
        } else {
            int sumi_m0 = 0;
#pragma unroll
            for (int i = i0; i < i0 + QI8_1/2; ++i) {
                sumi_m0 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m0);
            }
            sumf_d8 -= dm2f0.y * sumi_m0;

            int sumi_m1 = 0;
#pragma unroll
            for (int i = i0 + QI8_1/2; i < i0 + QI8_1; ++i) {
                sumi_m1 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m1);
            }
            sumf_d8 -= dm2f1.y * sumi_m1;
        }
    }

    return sumf + d8*sumf_d8;
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsub4(vil, vih);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsub4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous v/x + u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

    const int      sc_packed = get_int_b4(sc, 0);
    const int8_t * sc_reg    = (const int8_t *) &sc_packed;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc_reg[i0/2+0]*sumi_d.x + sc_reg[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q1_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q1_0 * bq1_0 = (const block_q1_0 *) vbq + kbx;

    // Q1_0: 128 elements with ONE scale
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 4 chunks of 32 elements to process (0-3)

    const float     d1 = bq1_0->d;
    const int16_t * qs = (const int16_t *) bq1_0->qs + iqs * 2;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int q  = qs[j];

        const int u0 = get_int_b4(bq8_1_chunk->qs, j*4+0);
        const int u1 = get_int_b4(bq8_1_chunk->qs, j*4+1);
        const int u2 = get_int_b4(bq8_1_chunk->qs, j*4+2);
        const int u3 = get_int_b4(bq8_1_chunk->qs, j*4+3);

        // unpack crumbs into nibble indices
        const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
        const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
        // unpack nibbles into byte values
        const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
        const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
        const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
        const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
        // unshuffle values
        const int v0 = __byte_perm(s0, s1, 0x5410);
        const int v1 = __byte_perm(s0, s1, 0x7632);
        const int v2 = __byte_perm(s2, s3, 0x5410);
        const int v3 = __byte_perm(s2, s3, 0x7632);

        sumi = ggml_cuda_dp4a(v0, u0, sumi);
        sumi = ggml_cuda_dp4a(v1, u1, sumi);
        sumi = ggml_cuda_dp4a(v2, u2, sumi);
        sumi = ggml_cuda_dp4a(v3, u3, sumi);
    }

    // Apply Q1_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d1 * d8 * sumi;
}

static __device__ __forceinline__ float vec_dot_q2_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;

    // Q2_0 (group 64): 64 elements with ONE scale, 2 bits per element (4 elements per byte)
    // Q8_1: 32 elements per block with individual scales
    // iqs selects which of the 2 chunks of 32 elements to process (0-1)

    const float     d2 = bq2_0->d;
    const int16_t * qs = (const int16_t *) bq2_0->qs + iqs * 4;

    // Process only the chunk specified by iqs
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;

#if defined(GGML_USE_HIP)
    // a crumb c of byte b of a 32-bit word is weight 4*b + c: (w >> 2*c) & 0x03030303 gathers crumb c of the 4 bytes,
    // which meets byte c of the 4 activation ints of those weights (the activations transposed by byte permutes, the
    // same for every row a thread computes); the weights are stored as crumb - 1: the sum of the activations is
    // subtracted (same integer as the table decode)
    int sumi = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int w = get_int_b2(bq2_0->qs, 2*iqs + h);
        const int a = get_int_b4(bq8_1_chunk->qs, 4*h + 0);
        const int b = get_int_b4(bq8_1_chunk->qs, 4*h + 1);
        const int c = get_int_b4(bq8_1_chunk->qs, 4*h + 2);
        const int d = get_int_b4(bq8_1_chunk->qs, 4*h + 3);
        const int ab_lo = __builtin_amdgcn_perm(b, a, 0x05010400);
        const int ab_hi = __builtin_amdgcn_perm(b, a, 0x07030602);
        const int cd_lo = __builtin_amdgcn_perm(d, c, 0x05010400);
        const int cd_hi = __builtin_amdgcn_perm(d, c, 0x07030602);
        sumi = ggml_cuda_dp4a((w >> 0) & 0x03030303, __builtin_amdgcn_perm(cd_lo, ab_lo, 0x05040100), sumi);
        sumi = ggml_cuda_dp4a((w >> 2) & 0x03030303, __builtin_amdgcn_perm(cd_lo, ab_lo, 0x07060302), sumi);
        sumi = ggml_cuda_dp4a((w >> 4) & 0x03030303, __builtin_amdgcn_perm(cd_hi, ab_hi, 0x05040100), sumi);
        sumi = ggml_cuda_dp4a((w >> 6) & 0x03030303, __builtin_amdgcn_perm(cd_hi, ab_hi, 0x07060302), sumi);
        sumi -= ggml_cuda_dp4a(0x01010101, a, ggml_cuda_dp4a(0x01010101, b, ggml_cuda_dp4a(0x01010101, c, ggml_cuda_dp4a(0x01010101, d, 0))));
    }
    GGML_UNUSED(qs);
#else
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);

        // unpack even and odd crumbs into byte values
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        // unshuffle values
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);

        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }

#endif // defined(GGML_USE_HIP)

    // Apply Q2_0's single scale and this chunk's Q8_1 scale
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b2(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq + kbx;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_b4(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b2(bq5_0->qs, iqs + i);
        vh[i]    = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq + kbx;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_b4(bq5_1->qs, iqs + i);
        vh[i]    = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_b4(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

// the same with non-temporal loads of the weights (a backend whose weights are read once, see get_int_b2_nt)
static __device__ __forceinline__ float vec_dot_q8_0_q8_1_nt(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2_nt(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, load_half_nt(&bq8_0->d), __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // branchless so nvcc can hoist this out of the ncols_dst loop
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    // same as q4_K
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

#define VDR_IQ2_XXS_Q8_1_MMVQ 2
#define VDR_IQ2_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_XS_Q8_1_MMVQ 2
#define VDR_IQ2_XS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ2_S_Q8_1_MMVQ 2
#define VDR_IQ2_S_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        // a byte 0xff per negative value: bit k of the 4 sign bits to bit 0 of byte k (the shifted copies do not
        // overlap), then times 0xff as (x << 8) - x (a 32-bit multiply by 0xff is a quarter-rate instruction on
        // RDNA3); negated bytewise as (g ^ m) + 1, no carry out of a byte (grid values <= 43)
        const uint32_t sb = signs_packed_8[l0/2];
        const uint32_t one0 = ((sb & 0x0F) * 0x00204081u) & 0x01010101u;
        const uint32_t one1 = ((sb >>   4) * 0x00204081u) & 0x01010101u;
        const uint32_t signs0 = (one0 << 8) - one0;
        const uint32_t signs1 = (one1 << 8) - one1;

        const int grid_l = (grid_pos[0] ^ signs0) + one0;
        const int grid_h = (grid_pos[1] ^ signs1) + one1;

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    // |sumi0|, |sumi1| < 2^17 (16 products of at most 43 x 127): 24-bit multiplies
    const int sumi = (__mul24(sumi0, ls0) + __mul24(sumi1, ls1) + (sumi0 + sumi1)/2)/4;

    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 2
#define VDR_IQ3_XXS_Q8_1_MMQ  2

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ3_S_Q8_1_MMVQ 2
#define VDR_IQ3_S_Q8_1_MMQ  2

// TODO: don't use lookup table for signs
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}

#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = __half2float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = __half22float2(bq8_1[iqs].ds);
    return d1q * (ds.x*sumi + ds.y*delta);
}

#define VDR_IQ1_M_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMQ  1

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq1_m * bq1 = (const block_iq1_m *) vbq + kbx;

    const int       qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    int   sumi[2] = {0};
    float sumf[2] = {0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2*iqs + l0/4] >> (4 * ((l0/2) % 2));

        const int grid = iq1s_grid_gpu[qs[l0/2] | ((qhl & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi[l0/4] = ggml_cuda_dp4a(grid0, u0, sumi[l0/4]);
        sumi[l0/4] = ggml_cuda_dp4a(grid1, u1, sumi[l0/4]);

        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f*IQ1M_DELTA/0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0/4] += delta*sumy;
    }

    const uint16_t * sc = (const uint16_t *) bq1->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);

    const int tmp = sc[iqs/2] >> (6*(iqs%2));
    const int sc0 = 2*((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2*((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

#define VDR_IQ4_NL_Q8_1_MMVQ 2
#define VDR_IQ4_NL_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

#define VDR_IQ4_XS_Q8_1_MMVQ 4
#define VDR_IQ4_XS_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs/4].ds);
    return d * sumi;
}

// a byte of lo (selector 0-3) or hi (4-7) per byte of sel (each byte of sel at most 7)
static __device__ __forceinline__ int ggml_cuda_perm8(const uint32_t hi, const uint32_t lo, const uint32_t sel) {
#if defined(GGML_USE_HIP)
    return (int) __builtin_amdgcn_perm(hi, lo, sel);
#else
    return (int) __byte_perm(lo, hi, (sel & 0x7) | ((sel >> 4) & 0x70) | ((sel >> 8) & 0x700) | ((sel >> 12) & 0x7000));
#endif // defined(GGML_USE_HIP)
}

// iq3_k (ik_llama.cpp's vec_dot_iq3_k_q8_1): a thread takes 8 weights in each of the 4 blocks of 32 of
// one half of the super-block; the 3-bit index and the grid bit (extra) form a nibble for the table
// of 16 (the grid, then the shifted grid)
#define VDR_IQ3_K_Q8_1_MMVQ 4
#define VDR_IQ3_K_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq3_k_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {
    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    const int iqs   = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1: weights 0...127 or 128...255
    const int il8   = iqs%4;  // 0...3: weights 8*il8...8*il8+7 of each block of 32
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    // the grid bit is the same for the 16 weights of a block, and each 4 weights a thread takes lie in one block: the
    // half of the table (grid or shifted grid) is chosen once per block, then one byte permute per 4 weights on the
    // 3-bit index (2 bits of qs, 1 of qh)
    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    const uint32_t * tab = (const uint32_t *) iq3nl_values;
    const uint32_t t0 = tab[0], t1 = tab[1], t2 = tab[2], t3 = tab[3];
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool g = (extra >> (2*j)) & 1;
        lo[j] = g ? t2 : t0;
        hi[j] = g ? t3 : t1;
    }

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        const uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift);

        const uint32_t i0 = ((vl >> 0) & 0x03030303) | ((vh >> 2) & 0x04040404);
        const uint32_t i1 = ((vl >> 2) & 0x03030303) | ((vh >> 3) & 0x04040404);
        const uint32_t i2 = ((vl >> 4) & 0x03030303) | ((vh >> 4) & 0x04040404);
        const uint32_t i3 = ((vl >> 6) & 0x03030303) | ((vh >> 5) & 0x04040404);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[0], lo[0], i0), q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[1], lo[1], i1), q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[2], lo[2], i2), q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[3], lo[3], i3), q8[i], sumi[3]);
    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    return d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);
}

// iq3_ks (ik_llama.cpp's vec_dot_iq3_ks_q8_1 with the scale read from the block): a thread takes 8 weights in each of
// the 4 blocks of 32 of one half of the super-block; the 3-bit index and the block's grid bit form a nibble for the
// table of 16; the signed 5-bit scale of each block of 32
#define VDR_IQ3_KS_Q8_1_MMVQ 4
#define VDR_IQ3_KS_Q8_1_MMQ  4

static __device__ __forceinline__ float vec_dot_iq3_ks_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {
    const block_iq3_ks * bq3 = (const block_iq3_ks *) vbq + kbx;

    const int iqs   = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1: weights 0...127 or 128...255
    const int il8   = iqs%4;  // 0...3: weights 8*il8...8*il8+7 of each block of 32

    // a block is 104 bytes: d, extra and the scales in one 8-byte load, qs and qh in 4-byte loads
    const uint2 head = *(const uint2 *) bq3;
    const uint32_t extra = (head.x >> 16) >> 4*ib128;
    const int * ql = (const int *) bq3->qs + 8*ib128 + 2*il8;
    const int * qh = (const int *) bq3->qh + 2*il8;

    // the grid bit is the same for the 32 weights of a block, and each 4 weights a thread takes lie in one block: the
    // half of the table (grid or shifted grid) is chosen once per block, then one byte permute per 4 weights on the
    // 3-bit index (2 bits of qs, 1 of qh)
    const uint32_t * tab = (const uint32_t *) iq3nl_values;
    const uint32_t t0 = tab[0], t1 = tab[1], t2 = tab[2], t3 = tab[3];
    uint32_t lo[4], hi[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const bool g = (extra >> (8 + j)) & 1;
        lo[j] = g ? t2 : t0;
        hi[j] = g ? t3 : t1;
    }

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[i];
        const uint32_t vh = (uint32_t) qh[i] >> 4*ib128;

        // blocks 0 and 2 from the low and high crumb pairs of each nibble pair, 1 and 3 likewise shifted by 2
        const uint32_t i0 = ((vl >> 0) & 0x03030303) | ((vh << 2) & 0x04040404);
        const uint32_t i1 = ((vl >> 2) & 0x03030303) | ((vh << 1) & 0x04040404);
        const uint32_t i2 = ((vl >> 4) & 0x03030303) | ((vh >> 0) & 0x04040404);
        const uint32_t i3 = ((vl >> 6) & 0x03030303) | ((vh >> 1) & 0x04040404);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[0], lo[0], i0), q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[1], lo[1], i1), q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[2], lo[2], i2), q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(ggml_cuda_perm8(hi[3], lo[3], i3), q8[i], sumi[3]);
    }
    const float d = __half2float(__ushort_as_half((unsigned short) (head.x & 0xffff)));
    // the 4 scales of this half at once: the nibbles, bit 4 from extra bits 0-3 (spread to one bit per byte), minus 16
    const uint32_t sc = ((head.y >> 4*ib128) & 0x0f0f0f0f) | ((((extra & 0x0f) * 0x00204081) & 0x01010101) << 4);
    const int ls0 = (int) ((sc >>  0) & 0xff) - 16;
    const int ls1 = (int) ((sc >>  8) & 0xff) - 16;
    const int ls2 = (int) ((sc >> 16) & 0xff) - 16;
    const int ls3 = (int) ((sc >> 24) & 0xff) - 16;
    return d * (__low2float(bq8_1[4*ib128+0].ds) * ls0 * sumi[0] +
                __low2float(bq8_1[4*ib128+1].ds) * ls1 * sumi[1] +
                __low2float(bq8_1[4*ib128+2].ds) * ls2 * sumi[2] +
                __low2float(bq8_1[4*ib128+3].ds) * ls3 * sumi[3]);
}
