#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

template<typename dst_t, typename src_t>
 __host__ __device__ inline dst_t ggml_cuda_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr(std::is_same_v<dst_t, nv_bfloat16>) {
        return __float2bfloat16(float(x));
    } else if constexpr(std::is_same_v<src_t, nv_bfloat16>) {
        return __bfloat162float(x);
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, half2>) {
        return __float22half2_rn(x);
    } else if constexpr(std::is_same_v<src_t, nv_bfloat162> && std::is_same_v<dst_t, float2>) {
#ifdef GGML_USE_HIP
        return make_float2(__bfloat162float(__low2bfloat16(x)), __bfloat162float(__high2bfloat16(x)));
#else
#if !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
        return __bfloat1622float2(x);
#else
        return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
#endif // !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= 800
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<src_t, float2> && std::is_same_v<dst_t, nv_bfloat162>) {
        // bypass compile error on cuda 12.0.1
#ifdef GGML_USE_HIP
        return __float22bfloat162_rn(x);
#else
        return {x.x, x.y};
#endif // GGML_USE_HIP
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}

// the image of an F32 result for the products that read it, written by the kernel that computes it in their scratch
// (ggml_cuda_f16_image_target): F16 or BF16 for cuBLAS (the same values as its conversion of src1, ggml_cuda_cast), or
// MMQ's quantized activation (block_q8_1_mmq, the D4 layout of Q8_0 weights: the bytes quantize_mmq_q8_1 writes)
enum ggml_cuda_img_kind { GGML_CUDA_IMG_F16 = 0, GGML_CUDA_IMG_BF16 = 1, GGML_CUDA_IMG_Q8_1_D4 = 2 };

struct ggml_cuda_img {
    void *  p    = nullptr;   // nullptr: no image
    int     kind = GGML_CUDA_IMG_F16;
    int64_t ne10 = 0;         // q8_1: the values of a row (the product's K) and the rows (its columns)
    int64_t ne11 = 0;

    // F16 / BF16: value i of the contiguous result
    __device__ __forceinline__ void store(int64_t i, float v) const {
        if (kind == GGML_CUDA_IMG_BF16) {
            ((nv_bfloat16 *) p)[i] = ggml_cuda_cast<nv_bfloat16>(v);
        } else {
            ((half *) p)[i] = ggml_cuda_cast<half>(v);
        }
    }

    // 4 consecutive values per thread, 8 threads per block of 32 (the warp's lanes, all of them here, aligned on 32 values)
    __device__ __forceinline__ void store4(int64_t i0, const float4 r) const {
        if (kind != GGML_CUDA_IMG_Q8_1_D4) {
            store(i0 + 0, r.x);
            store(i0 + 1, r.y);
            store(i0 + 2, r.z);
            store(i0 + 3, r.w);
            return;
        }
        float amax = fabsf(r.x);
        amax = fmaxf(amax, fabsf(r.y));
        amax = fmaxf(amax, fabsf(r.z));
        amax = fmaxf(amax, fabsf(r.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
        }
        const float v[4] = { r.x, r.y, r.z, r.w };
        const float d_inv = 127.0f / amax;
        const int64_t t   = i0 / ne10;
        const int64_t k   = i0 % ne10;
        char * blk = (char *) p + ((k / 128)*ne11 + t)*144;
        const int iqs = (int) (k % 128);
        char4 q;
        q.x = roundf(v[0]*d_inv);
        q.y = roundf(v[1]*d_inv);
        q.z = roundf(v[2]*d_inv);
        q.w = roundf(v[3]*d_inv);
        ((char4 *) (blk + 16))[iqs / 4] = q;
        if (iqs % 32 == 0) {
            ((float *) blk)[iqs / 32] = 1.0f / d_inv;
        }
    }

    // one value per thread, 32 threads per block of 32 (a whole warp, aligned on 32 values)
    __device__ __forceinline__ void store1(int64_t i, const float v) const {
        if (kind != GGML_CUDA_IMG_Q8_1_D4) {
            store(i, v);
            return;
        }
        float amax = fabsf(v);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
        }
        const float d_inv = 127.0f / amax;
        const int64_t t   = i / ne10;
        const int64_t k   = i % ne10;
        char * blk = (char *) p + ((k / 128)*ne11 + t)*144;
        const int iqs = (int) (k % 128);
        blk[16 + iqs] = (int8_t) roundf(v*d_inv);
        if (iqs % 32 == 0) {
            ((float *) blk)[iqs / 32] = 1.0f / d_inv;
        }
    }
};
