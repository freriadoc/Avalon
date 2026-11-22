// The gate and the mix of a hyper-connection read (qwen4exp, build_hc_mix) of a prompt batch in one kernel, on the
// RDNA3 matrix cores (GGML_CUDA_HC_UPMIX=0 turns it off):
//   mixed[t][d] = scale sum_c xn[t][c][d] sigmoid(W_up[c n_embd + d] . lo[t])
// in the place of the product W_up lo (hc n_embd values a token: 1.34 GB written at 32768 tokens, then read back) and of
// the mix (GGML_OP_DSV4_HC_PRE, gated). The gate stays in the products' accumulators. W_up is Q8_0, decoded to FP16
// with its scale as the experts' kernels do (moe-f16.cu); lo is converted to FP16 as it is staged; an item is 128 rows of W_up (32
// values d x the 4 streams, row 4 di + c) x 128 tokens, the blocks walk the items. The structure is Strata-rdna3's
// native_h11p_kernel (MIT), as moe-f16.cu's.
// RX 7900 XT, 32768 tokens (standalone bench): 5.0 ms where the product and the mix take 5.9 + 4.9.
// k_hc_upmix2, when K is 320 (lo is first made FP16, in the memory of the product's unwritten result): the block
// of Strata-rdna3's gr_upmix_kernel (MIT), 256 tokens x 16 values d x the 4 streams on 512 threads; its 64 rows of W_up
// are decoded into LDS once for all K and each wave reads its 16 tokens' rows of lo from global memory. One barrier a
// block, 126 registers (12 waves a SIMD, against 6): 3.7 ms on the same bench (4.5 with lo converted from F32 as it is
// read). GGML_CUDA_HC_UPMIX_NT=256: the kernel above.

#include "common.cuh"

#include <cstdlib>

#if defined(GGML_USE_HIP)
#define HC_MIX_HIP 1
#if defined(RDNA3)
#define HC_MIX_DEVICE 1
#endif

namespace {

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float    v8f  __attribute__((ext_vector_type(8)));
typedef _Float16 h2   __attribute__((ext_vector_type(2)));
struct u4 { uint32_t x, y, z, w; };
constexpr int HLD = 72;   // halves per LDS row: 64 values + 8

__device__ __forceinline__ v8f wmma(const v16h a, const v16h b, const v8f c) {
#if defined(HC_MIX_DEVICE)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif // defined(HC_MIX_DEVICE)
}
__device__ __forceinline__ uint32_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }
__device__ __forceinline__ uint32_t ld32(const uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ v16h ldf(const _Float16 * p) {   // 16 halves, 32-byte aligned
    const u4 * q = (const u4 *) p;
    struct { u4 a, b; } t{q[0], q[1]};
    return __builtin_bit_cast(v16h, t);
}
// 4 int8 times a scale as 4 FP16 (two words): a code c as the half 1024 + (c ^ 128), minus 1152, times the scale
__device__ __forceinline__ void conv4(const uint32_t q, const h2 sc, uint32_t & w0, uint32_t & w1) {
    const h2 off = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    const uint32_t u = q ^ 0x80808080u;
    w0 = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07010700u)) + off)*sc);
    w1 = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07030702u)) + off)*sc);
}

struct mix_args {
    const uint8_t * W; int64_t row_bytes;   // row r: W + r row_bytes, Q8_0, K values
    const float * lo; int64_t lo_tok;       // [T][K] (floats between two tokens), converted to FP16 as it is staged
    const _Float16 * lo16;                  // k_hc_upmix2: lo as FP16 rows [T][K]
    const float * xn; int64_t xn_tok;       // [T][4][n_embd] (floats between two tokens)
    // lazy: xn holds the norm's scale of each (token, stream) instead ([T][4]); the normalized value is
    // res[t][c][d] scale[t][c] w[c][d] (res: the streams the norm read; floats between two streams, two tokens)
    bool lazy, res16;                       // res16: res is FP16 (in the first half of its memory)
    const void * res; int64_t res_s1, res_tok;
    const float * w;
    float * out; int64_t out_tok;           // [T][n_embd]
    int T, K, n_embd;
    float scale;
};

__global__ void __launch_bounds__(256) k_hc_upmix(const mix_args a) {
    const int K = a.K, NS = K/64, NE = a.n_embd, NFB = NE/32;
    __shared__ __attribute__((aligned(32))) _Float16 smem[(128 + 128)*HLD];
    _Float16 (* wt)[HLD] = (_Float16 (*)[HLD]) smem;
    _Float16 (* bt)[HLD] = (_Float16 (*)[HLD]) (smem + 128*HLD);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    const int ur = tid >> 1, uj = tid & 1;
    const int n_tiles = (a.T + 127)/128, nwork = n_tiles*NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const int tile = w / NFB, fb = w % NFB, tok0 = tile*128, nrows = min(128, a.T - tok0);
        const int d0 = fb*32;
        // weight row ur of the item: value d0 + ur / 4 of stream ur % 4
        const uint8_t * wrow = a.W + (int64_t) ((ur & 3)*NE + d0 + (ur >> 2))*a.row_bytes + uj*34;
        u4 * bdst = (u4 *) &bt[ur][32*uj];
        u4 * wdst = (u4 *) &wt[ur][32*uj];
        uint32_t raw[9];
        auto load = [&](const int s) {
            const uint8_t * bp = wrow + s*68;
#pragma unroll
            for (int k = 0; k < 8; k++) raw[k] = ld32(bp + 2 + 4*k);
            raw[8] = ld16(bp);
        };
        auto put = [&]() {
            const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[8]);
            const h2 sc = {d, d};
            uint32_t hw[16];
#pragma unroll
            for (int k = 0; k < 8; k++) conv4(raw[k], sc, hw[2*k], hw[2*k + 1]);
#pragma unroll
            for (int k = 0; k < 4; k++) wdst[k] = u4{hw[4*k], hw[4*k + 1], hw[4*k + 2], hw[4*k + 3]};
        };
        __syncthreads();   // the previous item is done with the buffers
        load(0);
        // the 32 values of lo this thread stages, as FP16
        const float4 * bsrc = (const float4 *) (a.lo + (int64_t) (tok0 + min(ur, nrows - 1))*a.lo_tok + 32*uj);
        u4 b0, b1, b2, b3;
        auto bload = [&](const int s) {
            uint32_t hw[16];
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const float4 v = bsrc[16*s + q];
                const h2 x = {(_Float16) fminf(fmaxf(v.x, -65504.0f), 65504.0f), (_Float16) fminf(fmaxf(v.y, -65504.0f), 65504.0f)};
                const h2 y = {(_Float16) fminf(fmaxf(v.z, -65504.0f), 65504.0f), (_Float16) fminf(fmaxf(v.w, -65504.0f), 65504.0f)};
                hw[2*q] = __builtin_bit_cast(uint32_t, x); hw[2*q + 1] = __builtin_bit_cast(uint32_t, y);
            }
            b0 = u4{hw[0], hw[1], hw[2], hw[3]};   b1 = u4{hw[4], hw[5], hw[6], hw[7]};
            b2 = u4{hw[8], hw[9], hw[10], hw[11]}; b3 = u4{hw[12], hw[13], hw[14], hw[15]};
        };
        bload(0);
        put();
        bdst[0] = b0; bdst[1] = b1; bdst[2] = b2; bdst[3] = b3;
        if (NS > 1) {
            load(1);
            bload(1);
        }
        const bool on = 64*wn < nrows;
        v8f acc[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) acc[mt][nt] = (v8f) 0.0f;
#pragma unroll 1
        for (int s = 0; s < NS; s++) {
            __syncthreads();   // stage s is in LDS
            if (on) {
#pragma unroll
                for (int q = 0; q < 4; q++) {
                    const v16h B0 = ldf(&bt[64*wn + l16][16*q]), B1 = ldf(&bt[64*wn + 16 + l16][16*q]);
                    const v16h B2 = ldf(&bt[64*wn + 32 + l16][16*q]), B3 = ldf(&bt[64*wn + 48 + l16][16*q]);
                    const v16h A0 = ldf(&wt[32*wm + l16][16*q]), A1 = ldf(&wt[32*wm + 16 + l16][16*q]);
                    acc[0][0] = wmma(A0, B0, acc[0][0]);
                    acc[0][1] = wmma(A0, B1, acc[0][1]);
                    acc[0][2] = wmma(A0, B2, acc[0][2]);
                    acc[0][3] = wmma(A0, B3, acc[0][3]);
                    acc[1][0] = wmma(A1, B0, acc[1][0]);
                    acc[1][1] = wmma(A1, B1, acc[1][1]);
                    acc[1][2] = wmma(A1, B2, acc[1][2]);
                    acc[1][3] = wmma(A1, B3, acc[1][3]);
                }
            }
            if (s + 1 < NS) {
                __syncthreads();   // every wave is done with stage s
                put();
                bdst[0] = b0; bdst[1] = b1; bdst[2] = b2; bdst[3] = b3;
                if (s + 2 < NS) {
                    load(s + 2);
                    bload(s + 2);
                }
            }
        }
        if (!on) {
            continue;
        }
        // element i of a lane: the item's row 32 wm + 16 mt + 2 i + hi, i.e. value 8 wm + 4 mt + i / 2 of stream 2 (i & 1) + hi
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const int r = 64*wn + 16*nt + l16;
                const int tok = tok0 + min(r, nrows - 1), dd = d0 + 8*wm + 4*mt;
                float xa[4], xb[4];
                if (a.lazy) {
                    float4 r0, r1;
                    if (a.res16) {
                        const _Float16 * rs = (const _Float16 *) a.res + (int64_t) tok*a.res_tok + dd;
                        const _Float16 * p0 = rs + hi*a.res_s1, * p1 = rs + (2 + hi)*a.res_s1;
                        r0 = make_float4((float) p0[0], (float) p0[1], (float) p0[2], (float) p0[3]);
                        r1 = make_float4((float) p1[0], (float) p1[1], (float) p1[2], (float) p1[3]);
                    } else {
                        const float * rs = (const float *) a.res + (int64_t) tok*a.res_tok + dd;
                        r0 = *(const float4 *) (rs + hi*a.res_s1);
                        r1 = *(const float4 *) (rs + (2 + hi)*a.res_s1);
                    }
                    const float4 w0 = *(const float4 *) (a.w + hi*NE + dd), w1 = *(const float4 *) (a.w + (2 + hi)*NE + dd);
                    const float s0 = a.xn[(int64_t) tok*4 + hi], s1 = a.xn[(int64_t) tok*4 + 2 + hi];
                    xa[0] = s0*r0.x*w0.x; xa[1] = s0*r0.y*w0.y; xa[2] = s0*r0.z*w0.z; xa[3] = s0*r0.w*w0.w;
                    xb[0] = s1*r1.x*w1.x; xb[1] = s1*r1.y*w1.y; xb[2] = s1*r1.z*w1.z; xb[3] = s1*r1.w*w1.w;
                } else {
                    const float * xs = a.xn + (int64_t) tok*a.xn_tok + dd;
                    const float4 x0 = *(const float4 *) (xs + hi*NE), x1 = *(const float4 *) (xs + (2 + hi)*NE);
                    xa[0] = x0.x; xa[1] = x0.y; xa[2] = x0.z; xa[3] = x0.w;
                    xb[0] = x1.x; xb[1] = x1.y; xb[2] = x1.z; xb[3] = x1.w;
                }
                float m[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float g0 = acc[mt][nt][2*j], g1 = acc[mt][nt][2*j + 1];
                    m[j] = xa[j]/(1.0f + __expf(-g0)) + xb[j]/(1.0f + __expf(-g1));
                }
#pragma unroll
                for (int j = 0; j < 4; j++) m[j] += __shfl_xor(m[j], 16);
                if (hi == 0 && r < nrows) {
                    *(float4 *) (a.out + (int64_t) (tok0 + r)*a.out_tok + d0 + 8*wm + 4*mt) =
                        make_float4(a.scale*m[0], a.scale*m[1], a.scale*m[2], a.scale*m[3]);
                }
            }
    }
}

// lo [T][K] F32 -> FP16 rows, a block a token
__global__ void k_hc_lo_f16(const float * __restrict__ lo, const int64_t lo_tok, _Float16 * __restrict__ lo16, const int K) {
    const float * row = lo + (int64_t) blockIdx.x*lo_tok;
    _Float16 * dst = lo16 + (int64_t) blockIdx.x*K;
    for (int i = threadIdx.x; i < K/2; i += blockDim.x) {
        const float2 v = ((const float2 *) row)[i];
        const h2 h = {(_Float16) fminf(fmaxf(v.x, -65504.0f), 65504.0f), (_Float16) fminf(fmaxf(v.y, -65504.0f), 65504.0f)};
        ((uint32_t *) dst)[i] = __builtin_bit_cast(uint32_t, h);
    }
}

constexpr int ULD = 328;   // halves per LDS row of W_up: K = 320 + 8
// MODE: 0 the normalized streams read, 1 lazy from F32 streams, 2 lazy from FP16 streams; a template, not a.lazy and
// a.res16: with the three epilogues in one kernel the 512-thread bound of 127 registers no longer holds them (5.8 ms)
template <int NT, int MODE>
__global__ void __launch_bounds__(NT) k_hc_upmix2(const mix_args a) {
    constexpr int K = 320;
    const int NE = a.n_embd;
    __shared__ __attribute__((aligned(32))) _Float16 sW[64][ULD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    // the column tiles fastest: the blocks in flight share their tokens, whose streams the epilogue reads
    const int t0 = blockIdx.y*(NT/2) + 16*wave, d0 = blockIdx.x*16;
    for (int i = tid; i < 64*(K/32); i += NT) {
        const int rr = i/(K/32), q = i%(K/32), c = rr/16, dd = rr%16;
        const uint8_t * bp = a.W + (int64_t) (c*NE + d0 + dd)*a.row_bytes + q*34;
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) ld16(bp));
        const h2 sc = {d, d};
        uint32_t hw[16];
#pragma unroll
        for (int k = 0; k < 8; k++) conv4(ld32(bp + 2 + 4*k), sc, hw[2*k], hw[2*k + 1]);
        u4 * dst = (u4 *) &sW[rr][32*q];
#pragma unroll
        for (int k = 0; k < 4; k++) dst[k] = u4{hw[4*k], hw[4*k + 1], hw[4*k + 2], hw[4*k + 3]};
    }
    __syncthreads();
    if (t0 >= a.T) {
        return;
    }
    const int ta = min(t0 + l16, a.T - 1);
    const _Float16 * arow = a.lo16 + (int64_t) ta*K;
    v8f acc[4];
#pragma unroll
    for (int c = 0; c < 4; c++) acc[c] = (v8f) 0.0f;
#pragma unroll
    for (int ks = 0; ks < K; ks += 16) {
        const v16h A = ldf(arow + ks);
#pragma unroll
        for (int c = 0; c < 4; c++) acc[c] = wmma(A, ldf(&sW[c*16 + l16][ks]), acc[c]);
    }
    // element i of a lane: token t0 + 2 i + hi, value d0 + l16, of stream c in acc[c]
    const int d = d0 + l16;
    float wn[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if constexpr (MODE != 0) {
#pragma unroll
        for (int c = 0; c < 4; c++) wn[c] = a.w[c*NE + d];
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const int t = t0 + 2*i + hi;
        if (t >= a.T) {
            continue;
        }
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            float x;
            if constexpr (MODE == 2) {
                x = a.xn[(int64_t) t*4 + c]*(float) ((const _Float16 *) a.res)[(int64_t) t*a.res_tok + c*a.res_s1 + d]*wn[c];
            } else if constexpr (MODE == 1) {
                x = a.xn[(int64_t) t*4 + c]*((const float *) a.res)[(int64_t) t*a.res_tok + c*a.res_s1 + d]*wn[c];
            } else {
                x = a.xn[(int64_t) t*a.xn_tok + c*NE + d];
            }
            s += x/(1.0f + __expf(-acc[c][i]));
        }
        a.out[(int64_t) t*a.out_tok + d] = a.scale*s;
    }
}

} // namespace
#endif // defined(GGML_USE_HIP)

// NVIDIA from Ampere on (GGML_CUDA_HC_UPMIX=1, off by default: not measured yet): the same kernel on the FP16 tensor
// cores, in the place of cuBLAS's product (W_up converted to FP16 before it) and of the mix. It is k_hc_upmix2's block
// (256 tokens x 16 values d x the 4 streams on 512 threads, the 64 rows of W_up decoded into shared memory once for all
// K, lo read as FP16 rows from global memory); the product in one kernel with the mix, the gate rows never in memory,
// is also Basalt's gr_up_mix (src/models/qwen4exp/prefill/kernels.cu, MIT, Jesus Gulfo). A warp: 16 tokens (the rows of
// m16n8k16's A) x two tiles of 8 values d of each stream (B: W_up's rows as columns).
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define HC_MIX_NV 1

namespace {

struct mix_args_nv {
    const uint8_t * W; int64_t row_bytes;   // row r: W + r row_bytes, Q8_0, K values
    const half * lo16;                      // lo as FP16 rows [T][K]
    const float * xn; int64_t xn_tok;       // [T][4][n_embd] (floats between two tokens); lazy: the norm's scales [T][4]
    const void * res; int64_t res_s1, res_tok;
    const float * w;
    float * out; int64_t out_tok;           // [T][n_embd]
    int T, n_embd;
    float scale;
};

// lo [T][K] F32 -> FP16 rows, a block a token
static __global__ void k_hc_lo_f16_nv(const float * __restrict__ lo, const int64_t lo_tok, half * __restrict__ lo16, const int K) {
    const float * row = lo + (int64_t) blockIdx.x*lo_tok;
    half2 * dst = (half2 *) (lo16 + (int64_t) blockIdx.x*K);
    for (int i = threadIdx.x; i < K/2; i += blockDim.x) {
        const float2 v = ((const float2 *) row)[i];
        dst[i] = __floats2half2_rn(fminf(fmaxf(v.x, -65504.0f), 65504.0f), fminf(fmaxf(v.y, -65504.0f), 65504.0f));
    }
}

constexpr int NV_ULD = 328;   // halves per shared row of W_up: K = 320 + 8
constexpr int NV_NT  = 512;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// c += A (16 x 16 f16, row) * B (16 x 8 f16, col)
static __device__ __forceinline__ void hc_mma(float (& c)[4], const uint32_t (& a)[4], const uint32_t b0, const uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// MODE: 0 the normalized streams read, 1 lazy from F32 streams, 2 lazy from FP16 streams
template <int MODE>
__global__ void __launch_bounds__(NV_NT) k_hc_upmix_nv(const mix_args_nv a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int K = 320;
    const int NE = a.n_embd;
    __shared__ __align__(16) half sW[64][NV_ULD];   // row c*16 + dd: stream c, value d0 + dd
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tig = lane & 3;
    // the column tiles fastest: the blocks in flight share their tokens, whose streams the epilogue reads
    const int t0 = blockIdx.y*(NV_NT/2) + 16*warp, d0 = blockIdx.x*16;
    for (int i = tid; i < 64*(K/32); i += NV_NT) {
        const int rr = i/(K/32), q = i%(K/32), c = rr/16, dd = rr%16;
        const uint8_t * bp = a.W + (int64_t) (c*NE + d0 + dd)*a.row_bytes + q*34;
        const float d = __half2float(*(const half *) bp);
        const int8_t * qs = (const int8_t *) (bp + 2);
        half2 * dst = (half2 *) &sW[rr][32*q];
#pragma unroll
        for (int k = 0; k < 16; k++) dst[k] = __floats2half2_rn(d*(float) qs[2*k], d*(float) qs[2*k + 1]);
    }
    __syncthreads();
    if (t0 >= a.T) {
        return;
    }
    // A: rows t0 + g and t0 + g + 8 (the tokens past the batch repeat its last one, never written)
    const uint32_t * ar0 = (const uint32_t *) (a.lo16 + (int64_t) min(t0 + g,     a.T - 1)*K) + tig;
    const uint32_t * ar1 = (const uint32_t *) (a.lo16 + (int64_t) min(t0 + g + 8, a.T - 1)*K) + tig;
    float acc[4][2][4];
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
        for (int n = 0; n < 2; n++)
#pragma unroll
            for (int q = 0; q < 4; q++) acc[c][n][q] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < K; ks += 16) {
        const uint32_t A[4] = { ar0[ks/2], ar1[ks/2], ar0[ks/2 + 4], ar1[ks/2 + 4] };
#pragma unroll
        for (int c = 0; c < 4; c++)
#pragma unroll
            for (int n = 0; n < 2; n++) {
                const uint32_t * br = (const uint32_t *) &sW[c*16 + 8*n + g][ks] + tig;
                hc_mma(acc[c][n], A, br[0], br[4]);
            }
    }
    // element q of acc[c][n]: token t0 + g + 8 (q >> 1), value d0 + 8 n + 2 tig + (q & 1), of stream c
#pragma unroll
    for (int n = 0; n < 2; n++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int t = t0 + g + 8*(q >> 1), d = d0 + 8*n + 2*tig + (q & 1);
            if (t >= a.T) {
                continue;
            }
            float s = 0.0f;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                float x;
                if constexpr (MODE == 2) {
                    x = a.xn[(int64_t) t*4 + c]*__half2float(((const half *) a.res)[(int64_t) t*a.res_tok + c*a.res_s1 + d])*a.w[c*NE + d];
                } else if constexpr (MODE == 1) {
                    x = a.xn[(int64_t) t*4 + c]*((const float *) a.res)[(int64_t) t*a.res_tok + c*a.res_s1 + d]*a.w[c*NE + d];
                } else {
                    x = a.xn[(int64_t) t*a.xn_tok + c*NE + d];
                }
                s += x/(1.0f + __expf(-acc[c][n][q]));
            }
            a.out[(int64_t) t*a.out_tok + d] = a.scale*s;
        }
#else
    GGML_UNUSED(a);
    NO_DEVICE_CODE;
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
}

} // namespace
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// mm: W_up (Q8_0) lo; pre: the gated mix of xn [n_embd, 4, T] by mm's output. Whether the one kernel takes both
bool ggml_cuda_hc_upmix_supported(const int cc, const ggml_tensor * mm, const ggml_tensor * pre) {
#if defined(HC_MIX_HIP)
    static const bool on = !getenv("GGML_CUDA_HC_UPMIX") || atoi(getenv("GGML_CUDA_HC_UPMIX")) != 0;
    const ggml_tensor * w = mm->src[0], * lo = mm->src[1], * xn = pre->src[0];
    const int64_t K = w->ne[0], T = lo->ne[1], n_embd = xn->ne[0];
    return on && GGML_CUDA_CC_IS_RDNA3(cc) && w->type == GGML_TYPE_Q8_0 && lo->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 &&
        pre->type == GGML_TYPE_F32 && ggml_get_op_params_i32(pre, 1) != 0 && ggml_get_op_params_i32(mm, 3) == GGML_PREC_DEFAULT &&
        K % 64 == 0 && n_embd % 32 == 0 && xn->ne[1] == 4 && w->ne[1] == 4*n_embd && w->ne[2] == 1 && w->ne[3] == 1 &&
        lo->ne[0] == K && lo->ne[2] == 1 && T >= 256 && xn->ne[2] == T && xn->ne[3] == 1 && pre->ne[0] == n_embd && pre->ne[1] == T &&
        w->nb[1] == (size_t) (K/32*34) && (uintptr_t) w->data % 2 == 0 && lo->nb[0] == sizeof(float) && lo->nb[1] % 16 == 0 && (uintptr_t) lo->data % 16 == 0 &&
        xn->nb[0] == sizeof(float) && xn->nb[1] == (size_t) n_embd*sizeof(float) && xn->nb[2] % 16 == 0 && (uintptr_t) xn->data % 16 == 0 &&
        pre->nb[0] == sizeof(float) && pre->nb[1] % 16 == 0 && (uintptr_t) pre->data % 16 == 0;
#elif defined(HC_MIX_NV)
    static const bool on = getenv("GGML_CUDA_HC_UPMIX") && atoi(getenv("GGML_CUDA_HC_UPMIX")) != 0;
    const ggml_tensor * w = mm->src[0], * lo = mm->src[1], * xn = pre->src[0];
    const int64_t K = w->ne[0], T = lo->ne[1], n_embd = xn->ne[0];
    return on && GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_AMPERE &&
        w->type == GGML_TYPE_Q8_0 && lo->type == GGML_TYPE_F32 && xn->type == GGML_TYPE_F32 &&
        pre->type == GGML_TYPE_F32 && ggml_get_op_params_i32(pre, 1) != 0 && ggml_get_op_params_i32(mm, 3) == GGML_PREC_DEFAULT &&
        K == 320 && n_embd % 16 == 0 && xn->ne[1] == 4 && w->ne[1] == 4*n_embd && w->ne[2] == 1 && w->ne[3] == 1 &&
        lo->ne[0] == K && lo->ne[2] == 1 && T >= 256 && xn->ne[2] == T && xn->ne[3] == 1 && pre->ne[0] == n_embd && pre->ne[1] == T &&
        w->nb[1] == (size_t) (K/32*34) && (uintptr_t) w->data % 2 == 0 && lo->nb[0] == sizeof(float) && lo->nb[1] % 8 == 0 && (uintptr_t) lo->data % 8 == 0 &&
        xn->nb[0] == sizeof(float) && xn->nb[1] == (size_t) n_embd*sizeof(float) && pre->nb[0] == sizeof(float);
#else
    GGML_UNUSED_VARS(cc, mm, pre);
    return false;
#endif // defined(HC_MIX_HIP)
}

// res (lazy): the normalized streams were not written; res are the streams the norm read ([n_embd, 4, T]), w its weights
// ([n_embd, 4]), and the memory of the normalized streams holds the norm's scale of each (token, stream)
void ggml_cuda_hc_upmix(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * pre, const ggml_tensor * res,
        const ggml_tensor * w_norm, const bool res16) {
#if defined(HC_MIX_HIP)
    const ggml_tensor * w = mm->src[0], * lo = mm->src[1], * xn = pre->src[0];
    const int64_t K = w->ne[0], T = lo->ne[1], n_embd = xn->ne[0];
    cudaStream_t stream = ctx.stream();
    mix_args a;
    a.W = (const uint8_t *) w->data; a.row_bytes = (int64_t) w->nb[1];
    a.lo = (const float *) lo->data; a.lo_tok = (int64_t) (lo->nb[1]/sizeof(float));
    a.xn = (const float *) xn->data; a.xn_tok = (int64_t) (xn->nb[2]/sizeof(float));
    a.lazy = res != nullptr;
    a.res = res ? res->data : nullptr;
    a.res16 = res16;
    a.res_s1 = res ? (int64_t) (res->nb[1]/sizeof(float)) : 0; a.res_tok = res ? (int64_t) (res->nb[2]/sizeof(float)) : 0;
    a.w = w_norm ? (const float *) w_norm->data : nullptr;
    a.out = (float *) pre->data; a.out_tok = (int64_t) (pre->nb[1]/sizeof(float));
    a.T = (int) T; a.K = (int) K; a.n_embd = (int) n_embd;
    a.scale = ggml_get_op_params_f32(pre, 0);
    static const int nt = getenv("GGML_CUDA_HC_UPMIX_NT") ? atoi(getenv("GGML_CUDA_HC_UPMIX_NT")) : 512;
    // lo as FP16 in the memory of the product's own result, which no kernel writes (the gate stays in the accumulators)
    // and which lives until the mix that reads it
    const auto disjoint = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data, * y0 = (const char *) y->data;
        return !(x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x));
    };
    if (nt == 512 && K == 320 && n_embd % 16 == 0 && ggml_nbytes(mm) >= (size_t) T*K*sizeof(_Float16) && (uintptr_t) mm->data % 16 == 0 &&
            disjoint(mm, pre) && disjoint(mm, lo) && disjoint(mm, xn) && (!res || disjoint(mm, res))) {
        a.lo16 = (const _Float16 *) mm->data;
        k_hc_lo_f16<<<(unsigned) T, 160, 0, stream>>>(a.lo, a.lo_tok, (_Float16 *) mm->data, (int) K);
        const dim3 grid((unsigned) (n_embd/16), (unsigned) ((T + 255)/256));
        if (!a.lazy) {
            k_hc_upmix2<512, 0><<<grid, 512, 0, stream>>>(a);
        } else if (!a.res16) {
            k_hc_upmix2<512, 1><<<grid, 512, 0, stream>>>(a);
        } else {
            k_hc_upmix2<512, 2><<<grid, 512, 0, stream>>>(a);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int64_t items = (T + 127)/128*(n_embd/32);
    k_hc_upmix<<<(unsigned) std::min<int64_t>(items, 4096), 256, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
#elif defined(HC_MIX_NV)
    const ggml_tensor * w = mm->src[0], * lo = mm->src[1], * xn = pre->src[0];
    const int64_t K = w->ne[0], T = lo->ne[1], n_embd = xn->ne[0];
    cudaStream_t stream = ctx.stream();
    mix_args_nv a;
    a.W = (const uint8_t *) w->data; a.row_bytes = (int64_t) w->nb[1];
    a.xn = (const float *) xn->data; a.xn_tok = (int64_t) (xn->nb[2]/sizeof(float));
    a.res = res ? res->data : nullptr;
    a.res_s1 = res ? (int64_t) (res->nb[1]/sizeof(float)) : 0; a.res_tok = res ? (int64_t) (res->nb[2]/sizeof(float)) : 0;
    a.w = w_norm ? (const float *) w_norm->data : nullptr;
    a.out = (float *) pre->data; a.out_tok = (int64_t) (pre->nb[1]/sizeof(float));
    a.T = (int) T; a.n_embd = (int) n_embd;
    a.scale = ggml_get_op_params_f32(pre, 0);
    // lo as FP16 in the memory of the product's own result, which no kernel writes and which lives until the mix that
    // reads it; from the pool when that memory is not usable
    const auto disjoint = [](const ggml_tensor * x, const ggml_tensor * y) {
        const char * x0 = (const char *) x->data, * y0 = (const char *) y->data;
        return !(x0 < y0 + ggml_nbytes(y) && y0 < x0 + ggml_nbytes(x));
    };
    ggml_cuda_pool_alloc<half> lo16_own(ctx.pool());
    half * lo16;
    if (ggml_nbytes(mm) >= (size_t) T*K*sizeof(half) && (uintptr_t) mm->data % 4 == 0 &&
            disjoint(mm, pre) && disjoint(mm, lo) && disjoint(mm, xn) && (!res || disjoint(mm, res))) {
        lo16 = (half *) mm->data;
    } else {
        lo16 = lo16_own.alloc((size_t) T*K);
    }
    a.lo16 = lo16;
    k_hc_lo_f16_nv<<<(unsigned) T, 160, 0, stream>>>((const float *) lo->data, (int64_t) (lo->nb[1]/sizeof(float)), lo16, (int) K);
    const dim3 grid((unsigned) (n_embd/16), (unsigned) ((T + 255)/256));
    if (!res) {
        k_hc_upmix_nv<0><<<grid, NV_NT, 0, stream>>>(a);
    } else if (!res16) {
        k_hc_upmix_nv<1><<<grid, NV_NT, 0, stream>>>(a);
    } else {
        k_hc_upmix_nv<2><<<grid, NV_NT, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(ctx, mm, pre, res, w_norm, res16);
#endif // defined(HC_MIX_HIP)
}
