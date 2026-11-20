// QSA attention of a prompt, a block per (query, KV head), on the int8 tensor cores.
//
// The query's 12 heads (GQA) are the 16 rows of the products (4 rows of padding), its own chosen cells the other side,
// 32 at a time: nothing is computed for a cell the query did not choose (the tile kernel computes each query against
// the union of its tile's selections, 4.3 selections' worth from 57k cells on). K (q8_0) and V (q4_0) are read as the
// rows the KV cache stores, their codes going to the products as they are:
//   scores: q is quantized once per (head, 32 values) to int8; q.k is an int8 product per block of 32 values (one k32
//           step on NVIDIA), times both scales in FP32
//   output: p * v_scale is quantized per (head, block of 32 values) over the chunk's 32 cells, the product with V's
//           codes runs over the cells (k = 32), its int32 sum times the scale joins the running FP32 sums, rescaled as
//           the online softmax moves its maximum
// NVIDIA (k_qattn_i8_nv): the rows of the next chunk are copied to shared memory (cp.async, whole rows: a block's scale
// comes with its codes) while the current one computes, its cell indices one chunk earlier; q's codes and the
// quantized p * v_scale stay in registers; 31.4 KB of shared memory, three blocks an SM.
// RDNA (k_qattn_i8): the same arithmetic (the int8 product takes 16 values a step there, as the f16 one), the rows read
// as their chunk starts. Reading the next chunk's rows into registers before the scores and storing them after the
// output was tried and removed: the compiler waits for every load before a barrier, the overlap stopped at the scores'
// end (7900 XT: 37.0 - 37.7 ms against 37.2).
//
// GGML_CUDA_QATTN_BENCH="n_kv[,n_queries[,runs[,skip]]]" runs the bench at the backend's creation and exits.
// GGML_CUDA_QATTN_I8=0: off (the f16 kernel of one query instead). On by default for NVIDIA from Ampere on and for RDNA3 /
// RDNA4 (7900 XT, 262k cells, 9536 queries, the node: 45.4 ms with the f16 kernel of one query, 37.1 with this one).

#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ggml_cuda_mma;

struct qattn_args {
    const char    * Q;      // f32 [256] per (query, head)
    const char    * K;      // q8_0 rows of 256 values
    const char    * V;      // q4_0 rows of 256 values
    const int32_t * lists;  // [n_list] per query, then the counts
    const half    * mask;   // [n_list] per query
    float         * out;    // [256] per (query, head)
    int64_t sQ1, sQ2;       // bytes per query, per head
    int64_t sK1, sK2;       // bytes per cell, per KV head
    int64_t sV1, sV2;
    int     n_list, n_q, n_head, n_head_kv;
    float   scale;
    int     skip;           // bench: stages left out
};

// the 4-bit codes of a word, minus 8, as int8
static __device__ __forceinline__ uint32_t qattn_q4(const uint32_t x) {
    const uint32_t m = x ^ 0x08080808u;
    return m | (((m & 0x08080808u) >> 3) * 0xF0u);
}

// 4 values as int8 codes in a word
static __device__ __forceinline__ uint32_t qattn_pack(const float a, const float b, const float c, const float d, const float inv) {
    return  ((uint32_t) (uint8_t) (int8_t) (int) rintf(a*inv))        | (((uint32_t) (uint8_t) (int8_t) (int) rintf(b*inv)) <<  8) |
           (((uint32_t) (uint8_t) (int8_t) (int) rintf(c*inv)) << 16) | (((uint32_t) (uint8_t) (int8_t) (int) rintf(d*inv)) << 24);
}

// rows of 256 bytes at a stride of 256 put every lane of a product's load on the same bank of the shared memory:
// 4 words of padding a row, as the f16 kernel's tiles (7900 XT, 262k cells: the scores 21.9 -> 5.0 ms, the run 54.4 -> 39.2)
#define QATTN_SQ 68   // Q and K rows (words)
#define QATTN_SV  9   // V's transposed rows (words): read by the products' B side only, no alignment asked
#define QATTN_SA 12   // p * v_scale rows (words)
#define QATTN_SP 33   // scores and weights rows (floats)

struct qattn_smem {
    int   Qc[16*QATTN_SQ];        // [head][256 codes]
    int   Kc[32*QATTN_SQ];        // [cell][256 codes]
    int   Vt[256*QATTN_SV + 4];   // [value][32 cells' codes]
    int   Ap[8*16*QATTN_SA];      // [block][head][32 cells' codes of p * v_scale]
    float qs[16*8];               // [head][block]
    float ks[32*8];               // [cell][block]
    float vs[32*8];               // [cell][block]
    float S[2*16*QATTN_SP];       // [part][head][cell]
    float P[16*QATTN_SP];         // [head][cell]
    float pvs[16*8];              // [head][block]
    float R[16];                  // the chunk's rescale of the running sums
    float L[16];
    float maskf[32];
};

// RDNA: the rows read as the chunk computes
static __global__ void k_qattn_i8(const qattn_args a) {
#if defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;
    constexpr int N        = tile_C::J;   // cells (scores) or values (output) per product
    constexpr int NT       = 32/N;        // products per 32 cells or values
    constexpr int QK_SPLIT = 4/NT;        // warps per cell tile in the scores
    constexpr int GPW      = 8/QK_SPLIT;  // blocks of 32 values per warp in the scores

    __shared__ __align__(16) qattn_smem sm;

    const int iq = blockIdx.x, kvh = blockIdx.y;
    const int lane = threadIdx.x, warp = threadIdx.y, t = warp*32 + lane;
    const int n_list = a.n_list, n_head = a.n_head, skip = a.skip;

    const char    * Qq = a.Q + iq*a.sQ1 + (int64_t) kvh*n_head*a.sQ2;
    float         * oq = a.out + ((int64_t) iq*n_head*a.n_head_kv + (int64_t) kvh*n_head)*256;
    const char    * Kg = a.K + kvh*a.sK2;
    const char    * Vg = a.V + kvh*a.sV2;
    const int32_t * lq = a.lists + (int64_t) iq*n_list;
    const half    * mq = a.mask + (int64_t) iq*n_list;
    const int n_chunks = (a.lists[(int64_t) n_list*a.n_q + iq] + 31)/32;

    // q: int8 per (head, block of 32 values)
    {
        const int h = t & 15, g = t >> 4;
        float v[32];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            v[j] = h < n_head ? ((const float *) (Qq + h*a.sQ2))[g*32 + j] : 0.0f;
            amax = fmaxf(amax, fabsf(v[j]));
        }
        const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            sm.Qc[h*QATTN_SQ + g*8 + w] = (int) qattn_pack(v[4*w], v[4*w + 1], v[4*w + 2], v[4*w + 3], inv);
        }
        sm.qs[h*8 + g] = amax/127.0f*a.scale;
    }

    // the online softmax of head hs: 8 threads a head, 4 cells each
    const int hs = t >> 3, c0 = 4*(t & 7);
    float M = -1.0e30f, L = 0.0f;
    float acc[2*NT*tile_C::ne];
#pragma unroll
    for (int i = 0; i < 2*NT*tile_C::ne; ++i) {
        acc[i] = 0.0f;
    }

    // the rows of a chunk: read into registers (load), then put into shared memory (store)
    int4     rk[4];
    uint32_t rv[4][2];
    half     rks[2], rvs[2], rm;
    const auto load = [&](const int ci) {
        const int32_t * l = lq + ci*32;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = t + 128*i;
            const int cell = idx >> 4, piece = idx & 15;
            const int32_t index = l[cell];
            const int64_t row = index ^ (index >> 31);
            memcpy(&rk[i], Kg + row*a.sK1 + (piece >> 1)*34 + 2 + (piece & 1)*16, 16);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = t + 128*i;
            const int cell = idx >> 3, g = idx & 7;
            const int32_t index = l[cell];
            const int64_t row = index ^ (index >> 31);
            rks[i] = *(const half *) (Kg + row*a.sK1 + g*34);
            rvs[i] = *(const half *) (Vg + row*a.sV1 + g*18);
        }
        rm = mq[ci*32 + (t & 31)];
        // V: 16 values of 4 cells (8 bytes of a block: values hb*8 .. in the low halves, 16 + hb*8 .. in the high ones)
        const int cq = t & 7, b = t >> 4, hb = (t >> 3) & 1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int32_t index = l[4*cq + j];
            const int64_t row = index ^ (index >> 31);
            memcpy(rv[j], Vg + row*a.sV1 + b*18 + 2 + hb*8, 8);
        }
    };
    const auto store = [&]() {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = t + 128*i;
            const int cell = idx >> 4, piece = idx & 15;
            *(int4 *) ((char *) sm.Kc + cell*(QATTN_SQ*4) + piece*16) = rk[i];
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            sm.ks[t + 128*i] = __half2float(rks[i]);
            sm.vs[t + 128*i] = __half2float(rvs[i]);
        }
        if (t < 32) {
            sm.maskf[t] = __half2float(rm);
        }
        // V transposed (a word of Vt: one value of 4 cells)
        const int cq = t & 7, b = t >> 4, hb = (t >> 3) & 1;
        uint32_t w[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j][0] = qattn_q4( rv[j][0]       & 0x0F0F0F0Fu);
            w[j][1] = qattn_q4( rv[j][1]       & 0x0F0F0F0Fu);
            w[j][2] = qattn_q4((rv[j][0] >> 4) & 0x0F0F0F0Fu);
            w[j][3] = qattn_q4((rv[j][1] >> 4) & 0x0F0F0F0Fu);
        }
#pragma unroll
        for (int dq = 0; dq < 4; ++dq) {
            const int d0 = b*32 + hb*8 + (dq < 2 ? 4*dq : 16 + 4*(dq - 2));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t x =  ((w[0][dq] >> (8*k)) & 0xFFu)        | (((w[1][dq] >> (8*k)) & 0xFFu) <<  8) |
                                   (((w[2][dq] >> (8*k)) & 0xFFu) << 16) | (((w[3][dq] >> (8*k)) & 0xFFu) << 24);
                sm.Vt[(d0 + k)*QATTN_SV + cq] = (int) x;
            }
        }
    };

    for (int ci = 0; ci < n_chunks; ++ci) {
        __syncthreads();
        load(ci);
        store();
        __syncthreads();

        // scores
        if (!(skip & 4)) {
            const int nt = warp % NT, part = warp / NT;
            float sacc[tile_C::ne];
#pragma unroll
            for (int i = 0; i < tile_C::ne; ++i) {
                sacc[i] = 0.0f;
            }
#pragma unroll
            for (int gi = 0; gi < GPW; ++gi) {
                const int g = part*GPW + gi;
                tile_A A;
                load_ldmatrix(A, sm.Qc + g*8, QATTN_SQ);
                tile_B B;
                load_ldmatrix(B, sm.Kc + nt*N*QATTN_SQ + g*8, QATTN_SQ);
                tile_C C;
                mma(C, A, B);
#pragma unroll
                for (int i = 0; i < tile_C::ne; ++i) {
                    const int h = tile_C::get_i(i), cell = nt*N + tile_C::get_j(i);
                    sacc[i] += (float) C.x[i]*sm.qs[h*8 + g]*sm.ks[cell*8 + g];
                }
            }
#pragma unroll
            for (int i = 0; i < tile_C::ne; ++i) {
                const int h = tile_C::get_i(i), cell = nt*N + tile_C::get_j(i);
                sm.S[(part*16 + h)*QATTN_SP + cell] = sacc[i];
            }
        }
        __syncthreads();

        // online softmax
        if (!(skip & 8)) {
            float s[4];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s[j] = sm.S[hs*QATTN_SP + c0 + j] + sm.maskf[c0 + j];
                if constexpr (QK_SPLIT == 2) {
                    s[j] += sm.S[(16 + hs)*QATTN_SP + c0 + j];
                }
                mx = fmaxf(mx, s[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o, 32));
            }
            const float Mn = fmaxf(M, mx);
            const float r  = expf(M - Mn);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = expf(s[j] - Mn);
                sm.P[hs*QATTN_SP + c0 + j] = p;
                sum += p;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o, 32);
            }
            L = L*r + sum;
            M = Mn;
            if ((t & 7) == 0) {
                sm.R[hs] = r;
            }
        }
        __syncthreads();

        // p * v_scale: int8 per (head, block of 32 values) over the 32 cells
        if (!(skip & 16)) {
            const int h = t & 15, g = t >> 4;
            float pv[32];
            float amax = 0.0f;
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                pv[j] = sm.P[h*QATTN_SP + j]*sm.vs[j*8 + g];
                amax = fmaxf(amax, fabsf(pv[j]));
            }
            const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                sm.Ap[(g*16 + h)*QATTN_SA + w] = (int) qattn_pack(pv[4*w], pv[4*w + 1], pv[4*w + 2], pv[4*w + 3], inv);
            }
            sm.pvs[h*8 + g] = amax/127.0f;
        }
        __syncthreads();

        // output: the warp's 64 values
        if (!(skip & 32))
#pragma unroll
        for (int gi = 0; gi < 2; ++gi) {
            const int g = 2*warp + gi;
            tile_A A;
            load_ldmatrix(A, sm.Ap + g*16*QATTN_SA, QATTN_SA);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                tile_B B;
                load_ldmatrix(B, sm.Vt + (g*32 + nt*N)*QATTN_SV, QATTN_SV);
                tile_C C;
                mma(C, A, B);
#pragma unroll
                for (int i = 0; i < tile_C::ne; ++i) {
                    const int h = tile_C::get_i(i);
                    float & x = acc[(gi*NT + nt)*tile_C::ne + i];
                    x = x*sm.R[h] + (float) C.x[i]*sm.pvs[h*8 + g];
                }
            }
        }
    }

    __syncthreads();
    if ((t & 7) == 0) {
        sm.L[hs] = L;
    }
    __syncthreads();
#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int i = 0; i < tile_C::ne; ++i) {
                const int h = tile_C::get_i(i);
                const int d = (2*warp + gi)*32 + nt*N + tile_C::get_j(i);
                if (h < n_head) {
                    const float ls = sm.L[h];
                    oq[h*256 + d] = ls > 0.0f ? acc[(gi*NT + nt)*tile_C::ne + i]/ls : 0.0f;
                }
            }
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE)
}

// RDNA3 (k_qattn_f16, GGML_CUDA_QATTN_F16=1): the same attention with FP16 operands accumulated in FP32, after
// Strata-rdna3's prompt_attn_wmma_kernel (src/kernels/cuda/qsa_prompt_attn.cu, MIT) on our K and V rows as stored. A
// wave owns 64 values of the heads (two blocks of 32 of K's q8_0 rows and of V's q4_0 rows). K's codes go from memory
// to registers as FP16 with their block's scale (a byte permute and a packed product, as the experts' kernels), so the
// scores need no scale after their products and nothing of q or K is staged in shared memory; q is one FP16 image
// scaled by a power of two. V's codes are staged as int8, p times V's scales is the FP16 row of the output's products.
#if defined(GGML_USE_HIP) && defined(RDNA3)
#define QATTN_F16_DEVICE 1
#endif

typedef _Float16 qattn_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 qattn_h2  __attribute__((ext_vector_type(2)));
typedef float    qattn_f8  __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ qattn_f8 qattn_wmma(const qattn_h16 a, const qattn_h16 b, const qattn_f8 c) {
#if defined(QATTN_F16_DEVICE)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif // defined(QATTN_F16_DEVICE)
}

struct qattn_f16_smem {
    float   part[4][16][33];   // q.k of each wave's 64 values: [wave][head][cell]
    float   p[16][33];         // the weights [head][cell]
    float   vs[4][2][32];      // V's scales [wave][block][cell]
    float   maskf[32];
    uint8_t v[4][32][80];      // [wave][cell]: its 64 values of V as int8
    float   qmax[4];
    float   alpha[16];
    float   lsum[16];
    float   mrow[16];
};

// KSC (GGML_CUDA_QATTN_KSC=0: off): K's codes go to the products as they are (exact in FP16), each product over one q8_0
// block in its own accumulator, the cell's scale applied after them in FP32 (the lane's column is its cell), as the
// fork's int8 K does; off, the codes enter with their block's scale (a packed product per two values).
// ALN (GGML_CUDA_QATTN_ALN=0: off): K's and V's rows read as the 4-byte aligned words they are, when the rows and their
// strides are (qattn_f16_launch): a wave's two blocks of K are the 68 bytes from byte 68w of the row, of V the 36 bytes
// from 36w, read by 16-byte loads of 4-byte alignment, the codes then realigned by a byte shift. Off, 16-byte copies
// the compiler only knows 2-byte aligned (a block is 34 or 18 bytes), which it splits into 4-byte loads: 68 of them
// against the fork's 28 of 16 bytes. RX 7900 XT, 9536 queries of 2048 cells over 24576, standalone bench: 29.8 ms, KSC
// 29.3, KSC + ALN 26.5 (the fork's kernel on its int8 K / V: 29.3), the same output bit for bit with and without ALN;
// a prompt of 19k: the attention 660 -> 589 ms (the fork's: 616).
typedef uint32_t qattn_u4a __attribute__((ext_vector_type(4), aligned(4)));

template <bool KSC, bool ALN>
static __global__ void __launch_bounds__(128) k_qattn_f16(const qattn_args a) {
#if defined(GGML_USE_HIP)
    __shared__ __align__(16) qattn_f16_smem S;
    const int iq = blockIdx.x, kvh = blockIdx.y;
    const int lane = threadIdx.x, warp = threadIdx.y, t = warp*32 + lane;
    const int col = lane & 15, half_ = lane >> 4, dim0 = warp*64;
    const int n_list = a.n_list, n_head = a.n_head;

    const char    * Qq = a.Q + iq*a.sQ1 + (int64_t) kvh*n_head*a.sQ2;
    float         * oq = a.out + ((int64_t) iq*n_head*a.n_head_kv + (int64_t) kvh*n_head)*256;
    const char    * Kg = a.K + kvh*a.sK2;
    const char    * Vg = a.V + kvh*a.sV2;
    const int32_t * lq = a.lists + (int64_t) iq*n_list;
    const half    * mq = a.mask + (int64_t) iq*n_list;
    const int n_chunks = (a.lists[(int64_t) n_list*a.n_q + iq] + 31)/32;

    // q as one FP16 image, scaled by a power of two
    float qm = 0.0f;
    for (int i = t; i < n_head*256; i += 128) {
        qm = fmaxf(qm, fabsf(((const float *) (Qq + (i >> 8)*a.sQ2))[i & 255]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xFFFFFFFFu, qm, o, 32));
    }
    if (lane == 0) {
        S.qmax[warp] = qm;
    }
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) {
        frexpf(qm, &qe);
    }
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(a.scale, qe - 14);
    qattn_h16 qh[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        const float * src = (const float *) (Qq + col*a.sQ2) + dim0 + kk*16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            qh[kk][i] = (_Float16) (col < n_head ? src[i]*qup : 0.0f);
        }
    }

    qattn_f8 acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc[j] = (qattn_f8) 0.0f;
    }
    const qattn_h2 off = {(_Float16) -1152.0f, (_Float16) -1152.0f};

    for (int ci = 0; ci < n_chunks; ++ci) {
        // every read of the chunk first: the lane's cell (V's two blocks of this wave, the mask), then K's fragments
        const int32_t index = lq[ci*32 + lane];
        const int64_t row = index ^ (index >> 31);
        uint32_t vr[2][4];
        half vsc[2];
        const half rm = mq[ci*32 + lane];
        if constexpr (ALN) {
            // 9 words from byte 36w: block 2w's scale in the low half of word 0, its codes from byte 2 (shifted), block
            // 2w+1's scale in the high half of word 4, its codes the words 5 .. 8
            const uint32_t * vp = (const uint32_t *) (Vg + row*a.sV1 + warp*36);
            const qattn_u4a v0 = *(const qattn_u4a *) vp, v1 = *(const qattn_u4a *) (vp + 4);
            const uint32_t d[9] = {v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3], vp[8]};
            vsc[0] = __builtin_bit_cast(half, (uint16_t) (d[0] & 0xFFFFu));
            vsc[1] = __builtin_bit_cast(half, (uint16_t) (d[4] >> 16));
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                vr[0][j] = __builtin_amdgcn_alignbyte(d[j + 1], d[j], 2);
                vr[1][j] = d[5 + j];
            }
        } else {
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const char * vp = Vg + row*a.sV1 + (2*warp + b)*18;
                vsc[b] = *(const half *) vp;
                memcpy(vr[b], vp + 2, 16);
            }
        }
        uint32_t kx[2][4][4];
        half ksc[2][2];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int32_t ix = __shfl_sync(0xFFFFFFFFu, index, nt*16 + col, 32);
            if constexpr (ALN) {
                // 17 words from byte 68w: block 2w's scale in the low half of word 0, its codes from byte 2 (shifted),
                // block 2w+1's scale in the high half of word 8, its codes the words 9 .. 16
                const uint32_t * kp = (const uint32_t *) (Kg + (int64_t) (ix ^ (ix >> 31))*a.sK1 + warp*68);
                uint32_t d[17];
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const qattn_u4a w4 = *(const qattn_u4a *) (kp + 4*q);
                    d[4*q] = w4[0]; d[4*q + 1] = w4[1]; d[4*q + 2] = w4[2]; d[4*q + 3] = w4[3];
                }
                d[16] = kp[16];
                ksc[nt][0] = __builtin_bit_cast(half, (uint16_t) (d[0] & 0xFFFFu));
                ksc[nt][1] = __builtin_bit_cast(half, (uint16_t) (d[8] >> 16));
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    kx[nt][0][w] = __builtin_amdgcn_alignbyte(d[w + 1], d[w], 2);
                    kx[nt][1][w] = __builtin_amdgcn_alignbyte(d[w + 5], d[w + 4], 2);
                    kx[nt][2][w] = d[9 + w];
                    kx[nt][3][w] = d[13 + w];
                }
            } else {
                const char * kp = Kg + (int64_t) (ix ^ (ix >> 31))*a.sK1 + (2*warp)*34;
                ksc[nt][0] = *(const half *) kp;
                ksc[nt][1] = *(const half *) (kp + 34);
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    memcpy(kx[nt][kk], kp + (kk >> 1)*34 + 2 + (kk & 1)*16, 16);
                }
            }
        }
        // V's codes as int8, its scales
        {
            uint4 * dst = (uint4 *) &S.v[warp][lane][0];
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                dst[2*b]     = make_uint4(qattn_q4( vr[b][0]       & 0x0F0F0F0Fu), qattn_q4( vr[b][1]       & 0x0F0F0F0Fu),
                                          qattn_q4( vr[b][2]       & 0x0F0F0F0Fu), qattn_q4( vr[b][3]       & 0x0F0F0F0Fu));
                dst[2*b + 1] = make_uint4(qattn_q4((vr[b][0] >> 4) & 0x0F0F0F0Fu), qattn_q4((vr[b][1] >> 4) & 0x0F0F0F0Fu),
                                          qattn_q4((vr[b][2] >> 4) & 0x0F0F0F0Fu), qattn_q4((vr[b][3] >> 4) & 0x0F0F0F0Fu));
                S.vs[warp][b][lane] = __half2float(vsc[b]);
            }
            if (warp == 0) {
                S.maskf[lane] = __half2float(rm);
            }
        }
        // q.k over the wave's 64 values, two tiles of 16 cells
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            qattn_f8 s = (qattn_f8) 0.0f;
            qattn_f8 sb[2] = {(qattn_f8) 0.0f, (qattn_f8) 0.0f};
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const _Float16 d = __builtin_bit_cast(_Float16, ksc[nt][kk >> 1]);
                const qattn_h2 sc = {d, d};
                uint32_t hw[8];
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t u = kx[nt][kk][w] ^ 0x80808080u;
                    const qattn_h2 c0 = __builtin_bit_cast(qattn_h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07010700u)) + off;
                    const qattn_h2 c1 = __builtin_bit_cast(qattn_h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07030702u)) + off;
                    hw[2*w]     = __builtin_bit_cast(uint32_t, KSC ? c0 : c0*sc);
                    hw[2*w + 1] = __builtin_bit_cast(uint32_t, KSC ? c1 : c1*sc);
                }
                struct { uint32_t w[8]; } tb;
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    tb.w[w] = hw[w];
                }
                if constexpr (KSC) {
                    sb[kk >> 1] = qattn_wmma(qh[kk], __builtin_bit_cast(qattn_h16, tb), sb[kk >> 1]);
                } else {
                    s = qattn_wmma(qh[kk], __builtin_bit_cast(qattn_h16, tb), s);
                }
            }
            if constexpr (KSC) {
                const float d0 = __half2float(ksc[nt][0]), d1 = __half2float(ksc[nt][1]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    s[i] = sb[0][i]*d0 + sb[1][i]*d1;
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                S.part[warp][2*i + half_][nt*16 + col] = s[i];
            }
        }
        __syncthreads();
        // the online softmax: 8 threads a head, 4 cells each
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub*4 + j;
                x[j] = (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c])*qdown + S.maskf[c];
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o, 32));
            }
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : expf(x[j] - m_new);
                S.p[r][sub*4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o, 32);
            }
            if (sub == 0) {
                const float al = m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                S.alpha[r] = al;
                S.lsum[r]  = fmaf(S.lsum[r], al, sum);
                S.mrow[r]  = m_new;
            }
        }
        __syncthreads();
        // p.v over the wave's 64 values: 4 tiles of 16 values, 2 steps of 16 cells; p times the block's V scale is the row
        {
            float vmax = fmaxf(fabsf(S.vs[warp][0][lane]), fabsf(S.vs[warp][1][lane]));
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFFu, vmax, o, 32));
            }
            const float vup = vmax > 0.0f ? 16384.0f/vmax : 0.0f, vdown = vmax*(1.0f/16384.0f);
            qattn_f8 tmp[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                tmp[j] = (qattn_f8) 0.0f;
            }
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                qattn_h16 ah[2];
#pragma unroll
                for (int b = 0; b < 2; ++b) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        ah[b][i] = (_Float16) (S.p[col][ks*16 + i]*(S.vs[warp][b][ks*16 + i]*vup));
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qattn_h16 bv;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        bv[i] = (_Float16) (int) (int8_t) S.v[warp][ks*16 + i][j*16 + col];
                    }
                    tmp[j] = qattn_wmma(ah[j >> 1], bv, tmp[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[j][i] = fmaf(acc[j][i], S.alpha[2*i + half_], tmp[j][i]*vdown);
                }
            }
        }
        // no barrier: v and vs are the wave's own, p, alpha and the mask are written again after the next chunk's first one
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int h = 2*i + half_;
            if (h < n_head) {
                const float l = S.lsum[h];
                oq[h*256 + dim0 + j*16 + col] = l > 0.0f ? acc[j][i]/l : 0.0f;
            }
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP)
}

// RDNA3: k_qattn_f16, unless GGML_CUDA_QATTN_F16=0 (k_qattn_i8). RX 7900 XT, the node, 9536 queries of 2048 cells,
// tiles of 16 / int8 / FP16 a query: 16.3 / 32.9 / 28.5 ms at 4.9k cells, 21.4 / 33.5 / 29.0 at 9.7k, 26.8 / 34.0 / 29.3
// at 16k, 31.4 / 34.3 / 29.5 at 25k, 35.0 / 34.5 / 29.6 at 33k, 50.7 / 35.0 / 30.0 at 66k, 54.3 / 35.9 / 30.8 at 262k;
// against FP64 the largest error 1e-5 (0.01 %) where the int8's is 1.8e-4 (0.20 %)
static bool qattn_ksc() {
    static const bool on = !getenv("GGML_CUDA_QATTN_KSC") || atoi(getenv("GGML_CUDA_QATTN_KSC")) != 0;
    return on;
}

// the k_qattn_f16 variant for these rows: KSC (qattn_ksc), ALN when the rows are 4-byte aligned (GGML_CUDA_QATTN_ALN=0: never)
static void qattn_f16_launch(const qattn_args & a, const dim3 grid, const dim3 block, cudaStream_t stream) {
    static const bool aln_on = !getenv("GGML_CUDA_QATTN_ALN") || atoi(getenv("GGML_CUDA_QATTN_ALN")) != 0;
    const bool aln = aln_on && (uintptr_t) a.K % 4 == 0 && (uintptr_t) a.V % 4 == 0 && a.sK1 % 4 == 0 && a.sK2 % 4 == 0 &&
        a.sV1 % 4 == 0 && a.sV2 % 4 == 0;
    if (qattn_ksc()) {
        if (aln) {
            k_qattn_f16<true, true><<<grid, block, 0, stream>>>(a);
        } else {
            k_qattn_f16<true, false><<<grid, block, 0, stream>>>(a);
        }
    } else {
        if (aln) {
            k_qattn_f16<false, true><<<grid, block, 0, stream>>>(a);
        } else {
            k_qattn_f16<false, false><<<grid, block, 0, stream>>>(a);
        }
    }
}

static bool qattn_f16_on(const int cc) {
    static const bool on = !getenv("GGML_CUDA_QATTN_F16") || atoi(getenv("GGML_CUDA_QATTN_F16")) != 0;
    return on && GGML_CUDA_CC_IS_RDNA3(cc);
}

// the cells from which a list per query replaces the tiles of 16 (ggml_moe_cache.qsa_query_min_kv): 16384 when every
// device runs k_qattn_f16 (the tiles are still 9 % faster there, the list 6 % at 25k), else the graph builder's limit
int64_t ggml_cuda_qattn_query_min_kv() {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    for (int i = 0; i < info.device_count; i++) {
        if (!qattn_f16_on(info.devices[i].cc)) {
            return 0;
        }
    }
    return info.device_count > 0 ? 16384 : 0;
}

struct qattn_nv_smem {
    int   Kr[2][32*68];   // K's rows as stored: [stage][cell][272 bytes]; q's codes at first, in stage 1
    int   Vr[2][32*36];   // V's rows as stored: [stage][cell][144 bytes]
    int   idx[2][32];     // the cells of a chunk
    half  mk[2][32];      // its mask
    float S[16*QATTN_SP]; // [head][cell]; q's scales at first
    float P[16*QATTN_SP]; // [head][cell]
    float R[16];
    float L[16];
};

// a word of 4 bytes at byte offset o of a row of words in shared memory; o is 4-aligned or 2 past it
static __device__ __forceinline__ uint32_t qattn_word(const int * __restrict__ w, const int o) {
    const uint32_t x0 = (uint32_t) w[o >> 2];
    if ((o & 3) == 0) {
        return x0;
    }
    return (x0 >> 16) | ((uint32_t) w[(o >> 2) + 1] << 16);
}

// NVIDIA from Ampere on: the next chunk's rows on their way while this one computes
static __global__ void k_qattn_i8_nv(const qattn_args a) {
#if defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    __shared__ __align__(16) qattn_nv_smem sm;

    const int iq = blockIdx.x, kvh = blockIdx.y;
    const int lane = threadIdx.x, warp = threadIdx.y, t = warp*32 + lane;
    const int n_list = a.n_list, n_head = a.n_head, skip = a.skip;

    const char    * Qq = a.Q + iq*a.sQ1 + (int64_t) kvh*n_head*a.sQ2;
    float         * oq = a.out + ((int64_t) iq*n_head*a.n_head_kv + (int64_t) kvh*n_head)*256;
    const char    * Kg = a.K + kvh*a.sK2;
    const char    * Vg = a.V + kvh*a.sV2;
    const int32_t * lq = a.lists + (int64_t) iq*n_list;
    const half    * mq = a.mask + (int64_t) iq*n_list;
    const int n_chunks = (a.lists[(int64_t) n_list*a.n_q + iq] + 31)/32;

    const unsigned int Kr_s  = ggml_cuda_cvta_generic_to_shared(sm.Kr);
    const unsigned int Vr_s  = ggml_cuda_cvta_generic_to_shared(sm.Vr);
    const unsigned int idx_s = ggml_cuda_cvta_generic_to_shared(sm.idx);
    const unsigned int mk_s  = ggml_cuda_cvta_generic_to_shared(sm.mk);

    // the cells of chunk c on their way to idx[c & 1] (warp 3)
    const auto fetch_cells = [&](const int c) {
        if (warp == 3 && lane < 8 && c < n_chunks) {
            cp_async_cg_16<0>(idx_s + (c & 1)*128 + lane*16, (const char *) (lq + c*32) + lane*16);
        }
    };
    // the rows and the mask of chunk c on their way to stage c & 1; its cells are in idx[c & 1]
    const auto fetch_rows = [&](const int c) {
        if (c >= n_chunks || ((skip & 1) && c > 0)) {
            return;
        }
        const int s = c & 1;
        const int32_t index = sm.idx[s][lane];
        const int64_t row = index ^ (index >> 31);   // a padded slot (complemented index): a valid row, masked
        const char * kr = Kg + row*a.sK1;
        const char * vr = Vg + row*a.sV1;
        const unsigned int kd = Kr_s + s*(32*272) + lane*272;
        const unsigned int vd = Vr_s + s*(32*144) + lane*144;
        // a row of K: 17 pieces of 16 bytes, of V: 9
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            cp_async_cg_16<0>(kd + (warp + 4*i)*16, kr + (warp + 4*i)*16);
        }
        if (warp == 0) {
            cp_async_cg_16<0>(kd + 16*16, kr + 16*16);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            cp_async_cg_16<0>(vd + (warp + 4*i)*16, vr + (warp + 4*i)*16);
        }
        if (warp == 1) {
            cp_async_cg_16<0>(vd + 8*16, vr + 8*16);
        }
        if (warp == 2 && lane < 4) {
            cp_async_cg_16<0>(mk_s + s*64 + lane*16, (const char *) (mq + c*32) + lane*16);
        }
    };

    fetch_cells(0);
    fetch_cells(1);

    // q: int8 per (head, block of 32 values), through stage 1 of K (free until the first chunk computes)
    {
        const int h = t & 15, g = t >> 4;
        float v[32];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            v[j] = h < n_head ? ((const float *) (Qq + h*a.sQ2))[g*32 + j] : 0.0f;
            amax = fmaxf(amax, fabsf(v[j]));
        }
        const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            sm.Kr[1][h*68 + g*8 + w] = (int) qattn_pack(v[4*w], v[4*w + 1], v[4*w + 2], v[4*w + 3], inv);
        }
        sm.S[h*QATTN_SP + g] = amax/127.0f*a.scale;
    }
    cp_async_wait_all();
    __syncthreads();

    // the thread's rows of the products: heads lane / 4 and 8 + lane / 4
    const int r0 = lane >> 2, q4 = lane & 3;
    tile_A QA[8];
    float  qs[2][8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        load_ldmatrix(QA[g], sm.Kr[1] + g*8, 68);
        qs[0][g] = sm.S[ r0     *QATTN_SP + g];
        qs[1][g] = sm.S[(r0 + 8)*QATTN_SP + g];
    }
    fetch_rows(0);
    cp_async_wait_all();
    __syncthreads();

    // the online softmax of head hs: 8 threads a head, 4 cells each
    const int hs = t >> 3, c0 = 4*(t & 7);
    float M = -1.0e30f, L = 0.0f;
    float acc[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        acc[i] = 0.0f;
    }

    // GGML_CUDA_QATTN_I8_AHEAD=0 (the bench's skip bit 2048): each chunk's rows asked for as it starts and waited for,
    // nothing on its way while the products run (to measure what the read-ahead brings)
    const bool ahead = !(skip & 2048);
    for (int ci = 0; ci < n_chunks; ++ci) {
        const int s = ci & 1;
        if (ahead) {
            // the next chunk's rows, the cells of the one after
            fetch_rows(ci + 1);
            fetch_cells(ci + 2);
        } else if (ci > 0) {
            fetch_rows(ci);
            fetch_cells(ci + 1);
            cp_async_wait_all();
            __syncthreads();
        }

        // scores: the warp's 8 cells, the 8 blocks of 32 values
        if (!(skip & 4)) {
            float sacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            const int * kb = sm.Kr[s] + (8*warp + r0)*68;        // the B side: cell lane / 4 of the warp's
            const int * k0 = sm.Kr[s] + (8*warp + 2*q4)*68;      // the C side: cells 2 * (lane % 4) and the next
#pragma unroll
            for (int g = 0; g < 8; ++g) {
                tile_B B;
                B.x[0] = (int) qattn_word(kb, 34*g + 2 + 4*q4);
                B.x[1] = (int) qattn_word(kb, 34*g + 2 + 4*(4 + q4));
                tile_C C;
                mma(C, QA[g], B);
                const float ks0 = __half2float(((const half *) k0)[17*g]);
                const float ks1 = __half2float(((const half *) (k0 + 68))[17*g]);
                sacc[0] += (float) C.x[0]*qs[0][g]*ks0;
                sacc[1] += (float) C.x[1]*qs[0][g]*ks1;
                sacc[2] += (float) C.x[2]*qs[1][g]*ks0;
                sacc[3] += (float) C.x[3]*qs[1][g]*ks1;
            }
            sm.S[ r0     *QATTN_SP + 8*warp + 2*q4    ] = sacc[0];
            sm.S[ r0     *QATTN_SP + 8*warp + 2*q4 + 1] = sacc[1];
            sm.S[(r0 + 8)*QATTN_SP + 8*warp + 2*q4    ] = sacc[2];
            sm.S[(r0 + 8)*QATTN_SP + 8*warp + 2*q4 + 1] = sacc[3];
        }
        __syncthreads();

        // online softmax
        if (!(skip & 8)) {
            float sc[4];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[j] = sm.S[hs*QATTN_SP + c0 + j] + __half2float(sm.mk[s][c0 + j]);
                mx = fmaxf(mx, sc[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o, 32));
            }
            const float Mn = fmaxf(M, mx);
            const float r  = expf(M - Mn);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = expf(sc[j] - Mn);
                sm.P[hs*QATTN_SP + c0 + j] = p;
                sum += p;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) {
                sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o, 32);
            }
            L = L*r + sum;
            M = Mn;
            if ((t & 7) == 0) {
                sm.R[hs] = r;
            }
        }
        __syncthreads();

        // output: the warp's 64 values (blocks 2 * warp and the next)
        if (!(skip & 32)) {
            const float rs0 = sm.R[r0], rs1 = sm.R[r0 + 8];
            const int * vb = sm.Vr[s];
#pragma unroll
            for (int gi = 0; gi < 2; ++gi) {
                const int g = 2*warp + gi;
                // A: p * v_scale of the thread's 2 heads over cells 4 * (lane % 4) .. + 3 and 16 more (the A side of the
                // k32 product: registers 0 and 1 the first 16 cells of rows lane / 4 and 8 more, 2 and 3 the last 16),
                // int8 per (head, block)
                float pv[2][8];
                float amax0 = 0.0f, amax1 = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int cell = (j >> 2)*16 + 4*q4 + (j & 3);
                    const float vs = __half2float(((const half *) (vb + cell*36))[9*g]);
                    pv[0][j] = sm.P[ r0     *QATTN_SP + cell]*vs;
                    pv[1][j] = sm.P[(r0 + 8)*QATTN_SP + cell]*vs;
                    amax0 = fmaxf(amax0, fabsf(pv[0][j]));
                    amax1 = fmaxf(amax1, fabsf(pv[1][j]));
                }
#pragma unroll
                for (int o = 1; o < 4; o <<= 1) {
                    amax0 = fmaxf(amax0, __shfl_xor_sync(0xFFFFFFFFu, amax0, o, 32));
                    amax1 = fmaxf(amax1, __shfl_xor_sync(0xFFFFFFFFu, amax1, o, 32));
                }
                const float inv0 = amax0 > 0.0f ? 127.0f/amax0 : 0.0f;
                const float inv1 = amax1 > 0.0f ? 127.0f/amax1 : 0.0f;
                tile_A A;
                A.x[0] = (int) qattn_pack(pv[0][0], pv[0][1], pv[0][2], pv[0][3], inv0);
                A.x[1] = (int) qattn_pack(pv[1][0], pv[1][1], pv[1][2], pv[1][3], inv1);
                A.x[2] = (int) qattn_pack(pv[0][4], pv[0][5], pv[0][6], pv[0][7], inv0);
                A.x[3] = (int) qattn_pack(pv[1][4], pv[1][5], pv[1][6], pv[1][7], inv1);
                const float ps0 = amax0/127.0f, ps1 = amax1/127.0f;

                // B: values 4 * (lane / 4) .. + 3 of the block, of cells 4 * (lane % 4) .. + 3 and 16 more: product nt
                // takes value nt of each (its column n is value 4 * n + nt of the block)
                uint32_t w[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int cell = (j >> 2)*16 + 4*q4 + (j & 3);
                    const uint32_t x = qattn_word(vb + cell*36, 18*g + 2 + 4*(r0 & 3));
                    w[j] = qattn_q4((r0 < 4 ? x : x >> 4) & 0x0F0F0F0Fu);
                }
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    tile_B B;
                    B.x[0] = (int) ( ((w[0] >> (8*nt)) & 0xFFu)        | (((w[1] >> (8*nt)) & 0xFFu) <<  8) |
                                    (((w[2] >> (8*nt)) & 0xFFu) << 16) | (((w[3] >> (8*nt)) & 0xFFu) << 24));
                    B.x[1] = (int) ( ((w[4] >> (8*nt)) & 0xFFu)        | (((w[5] >> (8*nt)) & 0xFFu) <<  8) |
                                    (((w[6] >> (8*nt)) & 0xFFu) << 16) | (((w[7] >> (8*nt)) & 0xFFu) << 24));
                    tile_C C;
                    mma(C, A, B);
                    float * x = acc + (gi*4 + nt)*4;
                    x[0] = x[0]*rs0 + (float) C.x[0]*ps0;
                    x[1] = x[1]*rs0 + (float) C.x[1]*ps0;
                    x[2] = x[2]*rs1 + (float) C.x[2]*ps1;
                    x[3] = x[3]*rs1 + (float) C.x[3]*ps1;
                }
            }
        }
        cp_async_wait_all();
        __syncthreads();
    }

    if ((t & 7) == 0) {
        sm.L[hs] = L;
    }
    __syncthreads();
    const float l0 = sm.L[r0], l1 = sm.L[r0 + 8];
#pragma unroll
    for (int gi = 0; gi < 2; ++gi) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int h = (i >> 1)*8 + r0;
                const int d = (2*warp + gi)*32 + 4*(2*q4 + (i & 1)) + nt;
                if (h < n_head) {
                    const float ls = i < 2 ? l0 : l1;
                    oq[h*256 + d] = ls > 0.0f ? acc[(gi*4 + nt)*4 + i]/ls : 0.0f;
                }
            }
        }
    }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

static bool qattn_i8_arch(const int cc) {
    static const int force = [] { const char * e = getenv("GGML_CUDA_QATTN_I8"); return e ? atoi(e) : -1; }();
    const bool nv   = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE;
    const bool rdna = GGML_CUDA_CC_IS_RDNA(cc) && amd_wmma_available(cc);
    return force != 0 && (nv || rdna);
}

// whether the int8 kernel takes this attention: a list per query (ggml_qsa_select_tiles of 1), a q8_0 K and a q4_0 V
// of 256 values a head, up to 16 heads a KV head, rows the asynchronous copies can take 16 bytes at a time
bool ggml_cuda_qattn_i8_applies(const int cc, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (!qattn_i8_arch(cc) || !mask || mask->op != GGML_OP_QSA_SELECT ||
            ggml_get_op_params_i32(mask, 1) != GGML_QSA_SELECT_TILE_MASK || mask->ne[1] != 1) {
        return false;
    }
    float max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    return max_bias == 0.0f && logit_softcap == 0.0f &&
        K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q4_0 && Q->type == GGML_TYPE_F32 &&
        Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[3] == 1 && K->ne[2] > 0 &&
        Q->ne[2] % K->ne[2] == 0 && Q->ne[2] / K->ne[2] <= 16 && Q->nb[0] == sizeof(float) &&
        K->nb[1] % 16 == 0 && K->nb[2] % 16 == 0 && V->nb[1] % 16 == 0 && V->nb[2] % 16 == 0 && mask->ne[0] % 32 == 0;
}

void ggml_cuda_qattn_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    GGML_ASSERT(mask->ne[2] == Q->ne[1] && ggml_is_contiguous(dst) && ggml_is_contiguous(mask));
    GGML_ASSERT((uintptr_t) K->data % 16 == 0 && (uintptr_t) V->data % 16 == 0);

    qattn_args a;
    a.Q         = (const char *) Q->data;
    a.K         = (const char *) K->data;
    a.V         = (const char *) V->data;
    a.lists     = (const int32_t *) mask->src[4]->data;
    a.mask      = (const half *) mask->data;
    a.out       = (float *) dst->data;
    a.sQ1       = Q->nb[1];
    a.sQ2       = Q->nb[2];
    a.sK1       = K->nb[1];
    a.sK2       = K->nb[2];
    a.sV1       = V->nb[1];
    a.sV2       = V->nb[2];
    a.n_list    = (int) mask->ne[0];
    a.n_q       = (int) Q->ne[1];
    a.n_head_kv = (int) K->ne[2];
    a.n_head    = (int) (Q->ne[2] / K->ne[2]);
    static const int ahead = [] { const char * e = getenv("GGML_CUDA_QATTN_I8_AHEAD"); return e ? atoi(e) : 1; }();
    a.skip      = ahead ? 0 : 2048;
    memcpy(&a.scale, (const float *) dst->op_params + 0, sizeof(float));

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const dim3 grid((unsigned) a.n_q, (unsigned) a.n_head_kv, 1), block(32, 4, 1);
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        k_qattn_i8_nv<<<grid, block, 0, ctx.stream()>>>(a);
    } else if (qattn_f16_on(cc)) {
        qattn_f16_launch(a, grid, block, ctx.stream());
    } else {
        k_qattn_i8<<<grid, block, 0, ctx.stream()>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
}

static uint64_t qattn_rng(uint64_t & s) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

void ggml_cuda_qattn_i8_bench(int device) {
    long n_kv = 262144;
    int n_q = 9536, runs = 20, skip = 0, uni = 0;
    sscanf(getenv("GGML_CUDA_QATTN_BENCH"), "%ld,%d,%d,%d,%d", &n_kv, &n_q, &runs, &skip, &uni);
    const int n_head = 12, n_head_kv = 2, n_top = 512, kpool = 4, n_sel = n_top*kpool, n_list = GGML_PAD(n_sel + kpool - 1, 256);
    const float scale = 1.0f/16.0f;
    ggml_cuda_set_device(device);
    const int cc = ggml_cuda_info().devices[device].cc;

    // the KV cache's rows: 8 blocks of q8_0 (2 bytes of scale, 32 codes) or of q4_0 (2 bytes of scale, 16 bytes of codes)
    uint64_t s = 88172645463325252ull;
    std::vector<uint8_t>     K((size_t) n_head_kv*n_kv*272), V((size_t) n_head_kv*n_kv*144);
    std::vector<float>       Q((size_t) n_q*n_head*n_head_kv*256);
    std::vector<int32_t>     lists((size_t) n_q*n_list + n_q);
    std::vector<ggml_fp16_t> mask((size_t) n_q*n_list, ggml_fp32_to_fp16(-INFINITY));
    for (size_t i = 0; i + 8 <= K.size(); i += 8) { const uint64_t r = qattn_rng(s); memcpy(&K[i], &r, 8); }
    for (size_t i = 0; i + 8 <= V.size(); i += 8) { const uint64_t r = qattn_rng(s); memcpy(&V[i], &r, 8); }
    for (size_t b = 0; b < K.size()/34; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.002f + 0.004f*(float) (qattn_rng(s) % 1024)/1024.0f);
        memcpy(&K[b*34], &d, 2);
    }
    for (size_t b = 0; b < V.size()/18; ++b) {
        const ggml_fp16_t d = ggml_fp32_to_fp16(0.05f + 0.10f*(float) (qattn_rng(s) % 1024)/1024.0f);
        memcpy(&V[b*18], &d, 2);
    }
    for (auto & x : Q) { x = (float) ((int64_t) (qattn_rng(s) % 2001) - 1000)/1000.0f; }
    // uni (tenths): the selections of 16 consecutive queries drawn from uni / 10 selections' worth of pools, as a deep
    // prompt's (4.3 from 57k cells on); 0: every query on its own
    std::vector<int32_t> pools, cand;
    for (int q = 0; q < n_q; ++q) {
        pools.clear();
        const int n_cand = (int) std::min<int64_t>(n_kv/kpool, (int64_t) uni*n_top/10);
        if (uni > 0 && q % 16 == 0) {
            cand.clear();
            while ((int) cand.size() < n_cand) {
                while ((int) cand.size() < n_cand) {
                    cand.push_back((int32_t) (qattn_rng(s) % (uint64_t) (n_kv/kpool)));
                }
                std::sort(cand.begin(), cand.end());
                cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
            }
        }
        while ((int) pools.size() < n_top) {
            while ((int) pools.size() < n_top) {
                pools.push_back(uni > 0 ? cand[qattn_rng(s) % (uint64_t) n_cand] : (int32_t) (qattn_rng(s) % (uint64_t) (n_kv/kpool)));
            }
            std::sort(pools.begin(), pools.end());
            pools.erase(std::unique(pools.begin(), pools.end()), pools.end());
        }
        int32_t * l = &lists[(size_t) q*n_list];
        for (int p = 0; p < n_top; ++p) {
            for (int c = 0; c < kpool; ++c) {
                l[p*kpool + c] = pools[p]*kpool + c;
            }
        }
        // as ggml_qsa_select_tiles pads a list: the complement of its last cell, masked
        for (int j = n_sel; j < n_list; ++j) {
            l[j] = ~l[n_sel - 1];
        }
        lists[(size_t) n_q*n_list + q] = n_sel;
        // the last cells hidden, as a causal mask hides a query's own future
        for (int j = 0; j < n_sel - (q % 7); ++j) {
            mask[(size_t) q*n_list + j] = ggml_fp32_to_fp16(0.0f);
        }
    }

    const auto up = [](const void * src, size_t bytes) {
        void * p = nullptr;
        CUDA_CHECK(cudaMalloc(&p, bytes));
        CUDA_CHECK(cudaMemcpy(p, src, bytes, cudaMemcpyHostToDevice));
        return p;
    };
    qattn_args a;
    a.Q         = (const char *)    up(Q.data(),     Q.size()*sizeof(float));
    a.K         = (const char *)    up(K.data(),     K.size());
    a.V         = (const char *)    up(V.data(),     V.size());
    a.lists     = (const int32_t *) up(lists.data(), lists.size()*sizeof(int32_t));
    a.mask      = (const half *)    up(mask.data(),  mask.size()*sizeof(ggml_fp16_t));
    a.sQ1       = (int64_t) n_head*n_head_kv*256*sizeof(float);
    a.sQ2       = 256*sizeof(float);
    a.sK1       = 272;
    a.sK2       = (int64_t) n_kv*272;
    a.sV1       = 144;
    a.sV2       = (int64_t) n_kv*144;
    a.n_list    = n_list;
    a.n_q       = n_q;
    a.n_head    = n_head;
    a.n_head_kv = n_head_kv;
    a.scale     = scale;
    a.skip      = skip;
    CUDA_CHECK(cudaMalloc((void **) &a.out, Q.size()*sizeof(float)));

    const bool nv = GGML_CUDA_CC_IS_NVIDIA(cc);
    const dim3 grid((unsigned) n_q, (unsigned) n_head_kv, 1), block(32, 4, 1);
    const bool f16 = qattn_f16_on(ggml_cuda_info().devices[device].cc);
    const auto run = [&]() {
        if (nv) {
            k_qattn_i8_nv<<<grid, block>>>(a);
        } else if (f16) {
            qattn_f16_launch(a, grid, block, 0);
        } else {
            k_qattn_i8<<<grid, block>>>(a);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    for (int i = 0; i < 3; ++i) {
        run();
    }
    {
        cudaFuncAttributes attr = {};
        if (cudaFuncGetAttributes(&attr, nv ? (const void *) k_qattn_i8_nv : f16 ? (qattn_ksc() ? (const void *) k_qattn_f16<true, true> : (const void *) k_qattn_f16<false, true>) : (const void *) k_qattn_i8) == cudaSuccess) {
            printf("qattn %s kernel:", f16 ? "f16" : "i8");
            printf(" %d registers, %zu bytes of scratch a thread, %zu bytes of shared memory\n",
                (int) attr.numRegs, (size_t) attr.localSizeBytes, (size_t) attr.sharedSizeBytes);
        }
    }
    double best = 1e30, sum = 0.0;
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        run();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        best = std::min(best, ms);
        sum += ms;
    }

    // a few queries against the same arithmetic in FP64, from the stored codes
    std::vector<float> o(Q.size());
    CUDA_CHECK(cudaMemcpy(o.data(), a.out, o.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0.0, ref_max = 0.0;
    for (int q : { 0, 5, n_q/2, n_q - 1 }) {
        for (int kvh = 0; kvh < n_head_kv; ++kvh) {
            for (int h = 0; h < n_head; ++h) {
                const float * qp = &Q[((size_t) q*n_head*n_head_kv + (size_t) kvh*n_head + h)*256];
                std::vector<double> sc(n_sel);
                double mx = -1e300;
                for (int j = 0; j < n_sel; ++j) {
                    const uint8_t * row = &K[((size_t) kvh*n_kv + lists[(size_t) q*n_list + j])*272];
                    double d = 0.0;
                    for (int k = 0; k < 256; ++k) {
                        ggml_fp16_t ds;
                        memcpy(&ds, row + (k/32)*34, 2);
                        d += (double) qp[k]*(int8_t) row[(k/32)*34 + 2 + k % 32]*ggml_fp16_to_fp32(ds);
                    }
                    sc[j] = d*scale + ggml_fp16_to_fp32(mask[(size_t) q*n_list + j]);
                    mx = std::max(mx, sc[j]);
                }
                double lsum = 0.0;
                std::vector<double> ov(256, 0.0);
                for (int j = 0; j < n_sel; ++j) {
                    const double p = exp(sc[j] - mx);
                    if (p == 0.0) {
                        continue;
                    }
                    lsum += p;
                    const uint8_t * row = &V[((size_t) kvh*n_kv + lists[(size_t) q*n_list + j])*144];
                    for (int k = 0; k < 256; ++k) {
                        ggml_fp16_t ds;
                        memcpy(&ds, row + (k/32)*18, 2);
                        const int byte = row[(k/32)*18 + 2 + (k % 32) % 16];
                        const int code = ((k % 32) < 16 ? byte & 15 : byte >> 4) - 8;
                        ov[k] += p*code*ggml_fp16_to_fp32(ds);
                    }
                }
                for (int k = 0; k < 256; ++k) {
                    const double ref = ov[k]/lsum;
                    const double got = o[((size_t) q*n_head*n_head_kv + (size_t) kvh*n_head + h)*256 + k];
                    err     = std::max(err, fabs(got - ref));
                    ref_max = std::max(ref_max, fabs(ref));
                }
            }
        }
    }
    printf("qattn i8 (skip %d, union %.1f): %ld cells, %d queries x %d KV heads, %d cells a query: %.2f ms a run (best %.2f, %d runs); "
           "against FP64 on 4 queries: max error %.4g, max value %.4g (%.2f %%)\n",
        skip, uni/10.0, n_kv, n_q, n_head_kv, n_sel, sum/runs, best, runs, err, ref_max, 100.0*err/ref_max);
}
