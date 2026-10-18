#pragma once

// GET_ROWS of quantized rows into f16 (the K/V cells a small batch attends, qwen4exp's QSA gather), its body shared by
// the multi-task kernel (mk.cu). A warp per row, 16 values per lane (half a block of 32), read as 16-bit words and
// written 32 bytes at a time: the same values as k_get_rows (getrows.cu: dequantize_q8_0 / _q4_0, then a cast to half),
// which takes a block of 256 threads per row of 512 values (a block for every ~1 KiB: the blocks' dispatch limits it).

#include "common.cuh"

struct get_rows_q_args {
    const void * src0; const int32_t * ids; half * dst;
    int64_t ne00; size_t nb01, nb02, nb03;
    int64_t ne10, ne11, ne12;
    size_t s10, s11, s12;   // ids, in elements
    size_t s1, s2, s3;      // dst, in elements (multiples of 8: rows aligned to 16 bytes)
    int type;               // GGML_TYPE_Q8_0 or GGML_TYPE_Q4_0
    int n_rows;             // ne10*ne11*ne12
};

#define GET_ROWS_Q_ROWS_PER_BLOCK 8   // rows per block of 256 threads

// row r of the gather (i10 + ne10*(i11*ne12 + i12), as k_get_rows' grid), by a warp (lane 0 .. 31)
static __device__ __forceinline__ void get_rows_q_row(const get_rows_q_args & a, const int r, const int lane) {
    if (r >= a.n_rows) {
        return;
    }
    // 32-bit divisions (a 64-bit one is a long software sequence)
    const int ne10 = (int) a.ne10, ne12 = (int) a.ne12;
    const int i10 = r % ne10;
    const int z   = r / ne10;
    const int i11 = z / ne12;
    const int i12 = z % ne12;
    const int i01 = a.ids[i10*a.s10 + i11*a.s11 + i12*a.s12];
    half * dst_row = a.dst + i10*a.s1 + i11*a.s2 + i12*a.s3;
    const char * src0_row = (const char *) a.src0 + i01*a.nb01 + i11*a.nb02 + i12*a.nb03;
    const bool q8 = a.type == GGML_TYPE_Q8_0;
    const int bsize = q8 ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0);
    for (int c = lane; c < (int) a.ne00 / 16; c += WARP_SIZE) {
        const int ib = c / 2, h = c % 2;
        const char * blk = src0_row + ib*bsize;
        const float d = __half2float(*(const half *) blk);
        const uint16_t * qs = (const uint16_t *) (blk + sizeof(half));
        half out[16];
        if (q8) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const uint16_t w = qs[h*8 + k];
                out[2*k + 0] = ggml_cuda_cast<half>((float) (int8_t) (w & 0xFF) * d);
                out[2*k + 1] = ggml_cuda_cast<half>((float) (int8_t) (w >> 8) * d);
            }
        } else {
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const uint16_t w = qs[k];
                const int b0 = w & 0xFF, b1 = w >> 8;
                const int n0 = h ? b0 >> 4 : b0 & 0xF;
                const int n1 = h ? b1 >> 4 : b1 & 0xF;
                out[2*k + 0] = ggml_cuda_cast<half>(((float) n0 - 8.0f) * d);
                out[2*k + 1] = ggml_cuda_cast<half>(((float) n1 - 8.0f) * d);
            }
        }
        uint4 * dst4 = (uint4 *) (dst_row + 16*c);
        dst4[0] = ((const uint4 *) out)[0];
        dst4[1] = ((const uint4 *) out)[1];
    }
}

// virtual block vb of 256 threads: rows vb*8 .. vb*8 + 7
static __device__ __forceinline__ void get_rows_q_body(const get_rows_q_args & a, const int vb, const int tid) {
    get_rows_q_row(a, vb*GET_ROWS_Q_ROWS_PER_BLOCK + tid / WARP_SIZE, tid % WARP_SIZE);
}
