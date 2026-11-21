// The dense Q8_0 products of a prompt on the RDNA3 matrix cores, in FP16 accumulated in FP32 (GGML_CUDA_DENSE_F16=0: MMQ):
// dst[t][n] = sum_k W[n][k] x[t][k] for a 2D Q8_0 weight and an F32 activation of many tokens (the GDN's qkv, gate and out
// projections, the attention's, the hyper-connections').
//
// The weight is decoded to FP16 with its scales once per product, into the scratch MMQ would have filled with the
// quantized activation (smaller: 2 bytes a weight against 9/8 byte a value of every token), by slices of rows when it
// does not fit. The activation is read in F32 and converted while it is staged, so it needs no copy. A block is 128
// tokens x 256 weight rows, 8 waves of 64 x 64, 64 values of K a stage: the 256 threads stage both operands in shared
// memory as FP16, the next stage read into registers meanwhile (two barriers a stage); the blocks walk the weight rows
// for 8 token tiles at a time, so a group's tokens stay in the caches.
//
// The kernel is Strata-rdna3's prompt GEMM (src/prefill/wmma_gemm.cu, pfg::kernel_bk64, MIT, nicejunjie and the Strata
// contributors) with the activation read in F32. RX 7900 XT, 32768 tokens (standalone bench): 81 - 86 TFLOPS, 21.3 ms for
// 10240 x 2560, 12.2 ms for 6144 x 2560, 12.0 ms for 2560 x 6144, where MMQ's int8 products take 30.8, 18.4 and 18.2 ms
// (~56 TFLOPS) and the quantization of the activation. From 1024 weight rows (GGML_CUDA_DENSE_F16_MIN_ROWS): with the
// products of 512 and 640 rows too, a prompt of 32768 tokens took 11.38 s instead of 11.25 s (one run each).

#include "common.cuh"

#include <cstdlib>

#if defined(GGML_USE_HIP)
#define DENSE_F16_HIP 1
#if defined(RDNA3)
#define DENSE_F16_DEVICE 1
#endif

namespace {

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef _Float16 v2h  __attribute__((ext_vector_type(2)));
typedef float    v8f  __attribute__((ext_vector_type(8)));
struct u4 { uint32_t x, y, z, w; };

constexpr int BM = 128, BN = 256, BK = 64, LDK = BK + 8, GM = 8;

static __device__ __forceinline__ v8f wmma(const v16h a, const v16h b, const v8f c) {
#if defined(DENSE_F16_DEVICE)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif // defined(DENSE_F16_DEVICE)
}

static __device__ __forceinline__ v16h frag(const _Float16 * p) {
    struct { u4 a, b; } t{*(const u4 *) p, *(const u4 *) (p + 8)};
    return __builtin_bit_cast(v16h, t);
}

static __device__ __forceinline__ uint32_t pack(const float a, const float b) {
    const v2h v = {(_Float16) a, (_Float16) b};
    return __builtin_bit_cast(uint32_t, v);
}

// rows [0, n_rows) of a Q8_0 weight (rows of K/32 blocks: a half scale then 32 int8) as FP16 [n_rows][K], a block a thread
static __global__ void k_dense_f16_weight(const char * __restrict__ W, const size_t nbw, const int nblk, const int64_t n_rows,
        _Float16 * __restrict__ out) {
    const int64_t n = n_rows*nblk;
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        const int64_t row = i/nblk;
        const int     b   = (int) (i - row*nblk);
        const char * blk = W + row*nbw + (size_t) b*sizeof(block_q8_0);
        const float  d   = (float) *(const _Float16 *) blk;
        u4 * o = (u4 *) (out + i*32);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t w[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                w[j] = pack(d*(float) (int8_t) blk[2 + 8*q + 2*j], d*(float) (int8_t) blk[2 + 8*q + 2*j + 1]);
            }
            o[q] = u4{w[0], w[1], w[2], w[3]};
        }
    }
}

// BF16 weights [n][K] as FP16 (exact but for the values FP16 cannot hold)
static __global__ void k_dense_f16_weight_bf16(const uint16_t * __restrict__ W, const int64_t n, _Float16 * __restrict__ out) {
    for (int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        out[i] = (_Float16) __uint_as_float((uint32_t) W[i] << 16);
    }
}

// Y[m][n] = sum_k W[n][k] X[m][k]: X F32 (X16: FP16) [M] rows of nbx bytes, W FP16 [N][K], Y F32 rows of ldy floats.
// K % 64 == 0
template <bool X16>
static __global__ void __launch_bounds__(256) k_dense_f16(const char * __restrict__ X, const _Float16 * __restrict__ W,
        float * __restrict__ Y, const int M, const int N, const int K, const size_t ldy, const size_t nbx) {
    __shared__ __align__(16) _Float16 sA[BM][LDK];
    __shared__ __align__(16) _Float16 sB[BN][LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int wm = wave & 1, wn = wave >> 1;
    const int num_m = (M + BM - 1)/BM, num_n = (N + BN - 1)/BN, b = blockIdx.x;
    const int group = b/(GM*num_n), first_m = group*GM, gsize = min(GM, num_m - first_m);
    const int m0 = (first_m + (b % (GM*num_n)) % gsize)*BM, n0 = ((b % (GM*num_n))/gsize)*BN;
    const int sr = tid >> 3, sq = tid & 7;   // 32 rows a pass, 8 values a thread
    const auto ldA = [&](const int k0, const int r) -> u4 {
        if constexpr (X16) {
            return *(const u4 *) (X + (size_t) min(m0 + r, M - 1)*nbx + (size_t) (k0 + 8*sq)*sizeof(_Float16));
        } else {
            const float4 * p = (const float4 *) (X + (size_t) min(m0 + r, M - 1)*nbx + (size_t) (k0 + 8*sq)*sizeof(float));
            const float4 a = p[0], c = p[1];
            return u4{pack(a.x, a.y), pack(a.z, a.w), pack(c.x, c.y), pack(c.z, c.w)};
        }
    };
    const auto ldB = [&](const int k0, const int r) -> u4 {
        return *(const u4 *) (W + (size_t) min(n0 + r, N - 1)*K + k0 + 8*sq);
    };
    u4 ra0, ra1, ra2, ra3, rb0, rb1, rb2, rb3, rb4, rb5, rb6, rb7;
    const auto load = [&](const int k0) {
        ra0 = ldA(k0, sr); ra1 = ldA(k0, sr + 32); ra2 = ldA(k0, sr + 64); ra3 = ldA(k0, sr + 96);
        rb0 = ldB(k0, sr); rb1 = ldB(k0, sr + 32); rb2 = ldB(k0, sr + 64); rb3 = ldB(k0, sr + 96);
        rb4 = ldB(k0, sr + 128); rb5 = ldB(k0, sr + 160); rb6 = ldB(k0, sr + 192); rb7 = ldB(k0, sr + 224);
    };
    const auto store = [&]() {
        *(u4 *) &sA[sr][8*sq] = ra0; *(u4 *) &sA[sr + 32][8*sq] = ra1;
        *(u4 *) &sA[sr + 64][8*sq] = ra2; *(u4 *) &sA[sr + 96][8*sq] = ra3;
        *(u4 *) &sB[sr][8*sq] = rb0; *(u4 *) &sB[sr + 32][8*sq] = rb1;
        *(u4 *) &sB[sr + 64][8*sq] = rb2; *(u4 *) &sB[sr + 96][8*sq] = rb3;
        *(u4 *) &sB[sr + 128][8*sq] = rb4; *(u4 *) &sB[sr + 160][8*sq] = rb5;
        *(u4 *) &sB[sr + 192][8*sq] = rb6; *(u4 *) &sB[sr + 224][8*sq] = rb7;
    };
    v8f acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            acc[i][j] = (v8f) 0.0f;
        }
    }
    load(0);
    store();
    __syncthreads();
    const int ar = 64*wm + l16, br = 64*wn + l16;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool more = k0 + BK < K;
        if (more) {
            load(k0 + BK);
        }
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            const v16h a0 = frag(&sA[ar][ks]), a1 = frag(&sA[ar + 16][ks]), a2 = frag(&sA[ar + 32][ks]), a3 = frag(&sA[ar + 48][ks]);
            const v16h b0 = frag(&sB[br][ks]), b1 = frag(&sB[br + 16][ks]), b2 = frag(&sB[br + 32][ks]), b3 = frag(&sB[br + 48][ks]);
#define DENSE_F16_ROW(i, a)             \
    acc[i][0] = wmma(a, b0, acc[i][0]); \
    acc[i][1] = wmma(a, b1, acc[i][1]); \
    acc[i][2] = wmma(a, b2, acc[i][2]); \
    acc[i][3] = wmma(a, b3, acc[i][3]);
            DENSE_F16_ROW(0, a0) DENSE_F16_ROW(1, a1) DENSE_F16_ROW(2, a2) DENSE_F16_ROW(3, a3)
#undef DENSE_F16_ROW
        }
        if (more) {
            __syncthreads();   // every wave is done reading the stage
            store();
            __syncthreads();
        }
    }
    // element e of a lane: token 2 e + hi of the tile, the lane's weight row
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + 64*wn + 16*j + l16;
            if (n >= N) {
                continue;
            }
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int m = m0 + 64*wm + 16*i + 2*e + hi;
                if (m < M) {
                    Y[(size_t) m*ldy + n] = acc[i][j][e];
                }
            }
        }
    }
}

} // namespace
#endif // defined(GGML_USE_HIP)

// the bytes of scratch a product needs at least: 256 weight rows in FP16
size_t ggml_cuda_dense_f16_min_bytes(const ggml_tensor * src0) {
    return (size_t) std::min<int64_t>(src0->ne[1], 256)*src0->ne[0]*sizeof(half);
}

// a 2D Q8_0 weight against the F32 rows of a prompt batch, on RDNA3
bool ggml_cuda_dense_f16_applies(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
#if defined(DENSE_F16_HIP)
    static const bool on = !getenv("GGML_CUDA_DENSE_F16") || atoi(getenv("GGML_CUDA_DENSE_F16")) != 0;
    static const int64_t min_rows = getenv("GGML_CUDA_DENSE_F16_MIN_ROWS") ? atoll(getenv("GGML_CUDA_DENSE_F16_MIN_ROWS")) : 1024;
    // BF16 weights (the MoE router, the indexer's projections): any width, in place of hipBLAS, whose calls each wait
    // ~1.6 ms on the host after converting the activation to BF16 (72 a ubatch); GGML_CUDA_DENSE_F16_BF16=0: hipBLAS
    static const bool bf16 = !getenv("GGML_CUDA_DENSE_F16_BF16") || atoi(getenv("GGML_CUDA_DENSE_F16_BF16")) != 0;
    // Q8_0 rows at a stride above their size too (a view of some of each row's blocks: the halves of the MTP draft's
    // eh_proj, LLAMA_MTP_PF & 2), which the weight's conversion reads row by row
    const bool w_ok = src0->type == GGML_TYPE_Q8_0 ? src0->ne[1] >= min_rows && src0->nb[0] == ggml_type_size(GGML_TYPE_Q8_0) &&
                                                     src0->nb[1] >= ggml_row_size(GGML_TYPE_Q8_0, src0->ne[0]) &&
                                                     src0->nb[1] % 2 == 0 && (uintptr_t) src0->data % 2 == 0 :
                      src0->type == GGML_TYPE_BF16 ? bf16 && src0->ne[1] >= 64 && src0->nb[1] == (size_t) src0->ne[0]*sizeof(uint16_t) &&
                                                     src0->nb[0] == sizeof(uint16_t) : false;
    static const int64_t min_tok  = getenv("GGML_CUDA_DENSE_F16_MIN_TOKENS") ? atoll(getenv("GGML_CUDA_DENSE_F16_MIN_TOKENS")) : 512;
    return on && GGML_CUDA_CC_IS_RDNA3(cc) && w_ok && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->ne[0] % BK == 0 && src1->ne[1] >= min_tok && src1->ne[0] == src0->ne[0] &&
        src1->nb[0] == sizeof(float) && src1->nb[1] % 16 == 0 && (uintptr_t) src1->data % 16 == 0 &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 &&
        src0->ne[1] < (1 << 30) && src1->ne[1] < (1 << 30);
#else
    GGML_UNUSED_VARS(cc, src0, src1, dst);
    return false;
#endif // defined(DENSE_F16_HIP)
}

// a 2D Q8_0 weight of any number of rows against the FP16 image of a prompt batch's activation (the hyper-connections'
// down, 320 rows x 10240: 4.05 ms at 32768 tokens on the standalone bench, where MMQ on the q8_1 image takes 4.8 - 5.0)
bool ggml_cuda_dense_f16_image_applies(const int cc, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
#if defined(DENSE_F16_HIP)
    static const bool on = !getenv("GGML_CUDA_DENSE_F16") || atoi(getenv("GGML_CUDA_DENSE_F16")) != 0;
    return on && GGML_CUDA_CC_IS_RDNA3(cc) && src0->type == GGML_TYPE_Q8_0 && src0->nb[1] == ggml_row_size(GGML_TYPE_Q8_0, src0->ne[0]) &&
        dst->type == GGML_TYPE_F32 && src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->ne[0] % BK == 0 && src1->ne[1] >= 512 && src1->ne[0] == src0->ne[0] &&
        dst->nb[0] == sizeof(float) && dst->nb[1] % sizeof(float) == 0 && src0->ne[1] < (1 << 30) && src1->ne[1] < (1 << 30);
#else
    GGML_UNUSED_VARS(cc, src0, src1, dst);
    return false;
#endif // defined(DENSE_F16_HIP)
}

// the product; w16: scratch of w16_bytes >= ggml_cuda_dense_f16_min_bytes, aligned to 16 bytes, free until the kernels ran
// x16: the activation as FP16 rows of K values, contiguous and 16-byte aligned (an image its producer wrote), else nullptr
void ggml_cuda_dense_f16(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, char * w16, const size_t w16_bytes,
        cudaStream_t stream, const void * x16) {
#if defined(DENSE_F16_HIP)
    const int64_t K = src0->ne[0], N = src0->ne[1], T = src1->ne[1];
    int64_t rows = (int64_t) (w16_bytes/((size_t) K*sizeof(half)));
    rows = rows >= N ? N : rows/BN*BN;
    GGML_ASSERT(rows > 0);
    for (int64_t r0 = 0; r0 < N; r0 += rows) {
        const int64_t n = std::min(rows, N - r0);
        if (src0->type == GGML_TYPE_BF16) {
            k_dense_f16_weight_bf16<<<(unsigned) std::min<int64_t>((n*K + 255)/256, 8192), 256, 0, stream>>>(
                (const uint16_t *) ((const char *) src0->data + r0*src0->nb[1]), n*K, (_Float16 *) w16);
        } else {
            const int64_t nblk = n*(K/32);
            k_dense_f16_weight<<<(unsigned) std::min<int64_t>((nblk + 255)/256, 8192), 256, 0, stream>>>(
                (const char *) src0->data + r0*src0->nb[1], src0->nb[1], (int) (K/32), n, (_Float16 *) w16);
        }
        const unsigned grid = (unsigned) (((T + BM - 1)/BM)*((n + BN - 1)/BN));
        if (x16) {
            k_dense_f16<true><<<grid, 256, 0, stream>>>((const char *) x16, (const _Float16 *) w16, (float *) dst->data + r0,
                (int) T, (int) n, (int) K, dst->nb[1]/sizeof(float), (size_t) K*sizeof(_Float16));
        } else {
            k_dense_f16<false><<<grid, 256, 0, stream>>>((const char *) src1->data, (const _Float16 *) w16, (float *) dst->data + r0,
                (int) T, (int) n, (int) K, dst->nb[1]/sizeof(float), src1->nb[1]);
        }
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(src0, src1, dst, w16, w16_bytes, stream, x16);
#endif // defined(DENSE_F16_HIP)
}
