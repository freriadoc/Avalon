// The routed experts of a prompt on the RDNA3 matrix cores, in FP16 accumulated in FP32 (GGML_CUDA_MOE_F16=0: off), in place
// of the MMQ products of the grouped MoE block (mmq.cu, moe_block_grouped_impl): gate and up with their SwiGLU written as
// FP16 rows, then down from those rows (plain or weighted into its token's row, ggml_mul_mat_id_sum).
//
// The weights are decoded to FP16 with their scale, the activation is FP16: the products accumulate straight into FP32,
// with no epilogue per block of 32 values (MMQ's int8 products pay one, on the unit the matrix cores share). An item is
// 128 weight rows (gate/up: the 64 gate and 64 up rows of 64 features, interleaved; down: 128 output rows) against up
// to 128 routed rows of one expert; the blocks walk the items. Per stage of 64 values: the 256 threads decode the rows'
// two blocks of 32 and stage the activation in shared memory, the raw bits of the next stage read into registers
// meanwhile; the waves are 4 along the weight rows x 2 along the tokens (6 fragment loads for 8 products), a wave's
// token columns past the item's rows skipped.
//
// The structure, the int8-to-FP16 conversion by byte permutes, Q2_0's and the codebook formats' fast decode are
// Strata-rdna3's native_h11p_kernel (src/prefill/moe_fused_iq.cu, MIT, nicejunjie and the Strata contributors), itself
// after gufo's RoutedF16GEMMKernel (MIT, the gufo contributors). IQ3_K and IQ3_KS are decoded as MMQ's load tiles do.
//
// RX 7900 XT, 512 experts, 10 a token, 9536 tokens (standalone bench): gate/up + SwiGLU 12.5 - 13.6 ms (46 - 50 TFLOPS;
// MMQ: ~19.3 ms and its SwiGLU and quantization), down 6.2 - 6.5 ms (MMQ ~9.1); at 32768 tokens 53 - 54 TFLOPS.

#include "common.cuh"

#include <cstdlib>
#include <mutex>
#include <vector>

struct moe_f16_tile { int e, row0, n; };   // up to 128 routed rows of expert e, from row0

static constexpr int T_IQ4_NL = GGML_TYPE_IQ4_NL, T_IQ2_S = GGML_TYPE_IQ2_S, T_Q2_0 = GGML_TYPE_Q2_0, T_IQ3_K = GGML_TYPE_IQ3_K, T_IQ3_KS = GGML_TYPE_IQ3_KS;
static __host__ __device__ constexpr int block_bytes(int t) { return t == T_IQ2_S ? 82 : t == T_IQ3_K ? 110 : t == T_IQ3_KS ? 104 : 18; }
static __host__ __device__ constexpr int block_vals(int t)  { return t == T_IQ4_NL ? 32 : t == T_Q2_0 ? 64 : 256; }
static __host__ __device__ constexpr int raw_words(int t)   { return t == T_IQ4_NL ? 5 : t == T_IQ3_K || t == T_IQ3_KS ? 10 : 3; }

// the host pass sees the kernels too; only the RDNA3 device pass has the FP16 WMMA
#if defined(GGML_USE_HIP)
#define MOE_F16_HIP 1
#if defined(RDNA3)
#define MOE_F16_DEVICE 1
#endif

namespace {

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float    v8f  __attribute__((ext_vector_type(8)));




typedef _Float16 h2 __attribute__((ext_vector_type(2)));
static __device__ __forceinline__ v8f wmma(const v16h a, const v16h b, const v8f c) {
#if defined(MOE_F16_DEVICE)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif // defined(MOE_F16_HIP)
}
struct u4 { uint32_t x, y, z, w; };
constexpr int HLD = 72;   // halves per LDS row: 64 values + 8 (no bank conflicts between the rows of a fragment)
static __device__ __forceinline__ uint32_t ld16(const uint8_t * p) { return *(const uint16_t *) p; }
static __device__ __forceinline__ uint32_t ld32(const uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
static __device__ __forceinline__ v16h ldf(const _Float16 * p) {   // 16 halves, 32-byte aligned
    const u4 * q = (const u4 *) p;
    struct { u4 a, b; } t{q[0], q[1]};
    return __builtin_bit_cast(v16h, t);
}


struct u2 { uint32_t x, y; };

// 4 nibbles (0..15, one per byte) through a 16-entry int8 table held in 4 words (Strata's table16_one)
static __device__ __forceinline__ uint32_t table16_one(const uint32_t x, const uint32_t t0, const uint32_t t1, const uint32_t t2, const uint32_t t3) {
    const uint32_t p = x & 0x07070707u;
    const uint32_t a = __builtin_amdgcn_perm(t1, t0, p), b = __builtin_amdgcn_perm(t3, t2, p);
    return __builtin_amdgcn_perm(b, a, ((x >> 1) & 0x04040404u) | 0x03020100u);
}
// 4 int8 times a scale as 4 FP16 (two words): a code c as the half 1024 + (c ^ 128), minus 1152, times the scale
static __device__ __forceinline__ void conv4(const uint32_t q, const h2 sc, uint32_t & w0, uint32_t & w1) {
    const h2 off = {(_Float16) -1152.0f, (_Float16) -1152.0f};
    const uint32_t u = q ^ 0x80808080u;
    w0 = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07010700u)) + off)*sc);
    w1 = __builtin_bit_cast(uint32_t, (__builtin_bit_cast(h2, __builtin_amdgcn_perm(0x64646464u, u, 0x07030702u)) + off)*sc);
}

// the raw bits of the 32 values `uj` of stage s (64 values) of a weight row
// pl (IQ3_K, IQ3_KS): the block's bit planes kept from a stage to the next ones that read them: the 2-bit plane of
// 4 units (two stages of a thread), the high-bit plane of the block's 8 units (four stages)
template <int WT> static __device__ __forceinline__ void load_unit(const uint8_t * wrow, const int s, const int uj, uint32_t (& raw)[raw_words(WT)], uint32_t (& pl)[16]) {
    if constexpr (WT == T_IQ4_NL) {
        const uint8_t * bp = wrow + (2*s + uj)*18;
#pragma unroll
        for (int k = 0; k < 4; k++) raw[k] = ld32(bp + 2 + 4*k);
        raw[4] = ld16(bp);
    } else if constexpr (WT == T_Q2_0) {
        const uint8_t * bp = wrow + s*18;
        raw[0] = ld32(bp + 2 + 8*uj); raw[1] = ld32(bp + 6 + 8*uj); raw[2] = ld16(bp);
    } else if constexpr (WT == T_IQ2_S) {
        const uint8_t * bp = wrow + (s >> 2)*82;
        const int ib = 2*(s & 3) + uj;
        raw[0] = ld32(bp + 2 + 4*ib);
        raw[1] = ld32(bp + 34 + 4*ib);
        raw[2] = ld16(bp) | ((uint32_t) bp[66 + ib] << 16) | ((uint32_t) bp[74 + ib] << 24);
    } else {   // IQ3_K, IQ3_KS: the 3-bit indices of the 32 values, a byte each, with their grid bit
        constexpr bool KS = WT == T_IQ3_KS;
        const uint8_t * bp = wrow + (s >> 2)*block_bytes(WT);
        const int ib = 2*(s & 3) + uj;
        const uint8_t * qs = bp + (KS ? 8 : 14) + 32*(ib/4), * qh = bp + (KS ? 72 : 78);
        const uint32_t extra = ld16(bp + 2);
        const uint32_t g0 = KS ? ((extra >> (8 + ib)) & 1)*0x08080808u : ((extra >> (2*ib)) & 1)*0x08080808u;
        const uint32_t g1 = KS ? g0 : ((extra >> (2*ib + 1)) & 1)*0x08080808u;
        if ((s & 1) == 0) {
#pragma unroll
            for (int m = 0; m < 8; m++) pl[m] = ld32(qs + 4*m);
        }
        if ((s & 3) == 0) {
#pragma unroll
            for (int m = 0; m < 8; m++) pl[8 + m] = ld32(qh + 4*m);
        }
#pragma unroll
        for (int m = 0; m < 8; m++) {
            raw[m] = ((pl[m] >> (2*(ib % 4))) & 0x03030303u) | (((pl[8 + m] >> ib) & 0x01010101u) << 2) | (m < 4 ? g0 : g1);
        }
        raw[8] = ld16(bp);
        raw[9] = KS ? (((uint32_t) bp[4 + ib % 4] >> (4*(ib/4))) & 15) | (((extra >> ib) & 1) << 4)
                    : (uint32_t) bp[6 + ib] | (((ld16(bp + 4) >> (2*ib)) & 3) << 8);
    }
}

// the unit as 32 FP16 with their scale, into 4 x 16 bytes of LDS
template <int WT> static __device__ __forceinline__ void put_unit(const uint32_t (& raw)[raw_words(WT)], u4 * wdst, const uint8_t * sgrid, const u2 * ssign) {
    uint32_t hw[16];
    if constexpr (WT == T_IQ4_NL) {
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[4]);
        const h2 sc = {d, d};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            conv4(table16_one(raw[k],      0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u), sc, hw[2*k],     hw[2*k + 1]);
            conv4(table16_one(raw[k] >> 4, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u), sc, hw[2*k + 8], hw[2*k + 9]);
        }
    } else if constexpr (WT == T_Q2_0) {   // a code c as the half 1024 + 256 c, / 256 - 5 = c - 1 (Strata's Q2FAST)
        const _Float16 d = __builtin_bit_cast(_Float16, (uint16_t) raw[2]);
        const h2 sc = {d, d}, k256 = {(_Float16) 0.00390625f, (_Float16) 0.00390625f}, m5 = {(_Float16) -5.0f, (_Float16) -5.0f};
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint32_t x = raw[h];
            const uint32_t A = (x & 0x03030303u) | 0x64646464u, B = ((x >> 2) & 0x03030303u) | 0x64646464u;
            const uint32_t C = ((x >> 4) & 0x03030303u) | 0x64646464u, D = ((x >> 6) & 0x03030303u) | 0x64646464u;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const uint32_t sel = 0x0Cu | ((uint32_t) b << 8) | (0x0Cu << 16) | ((uint32_t) (4 + b) << 24);
                const h2 p0 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(B, A, sel)), p1 = __builtin_bit_cast(h2, __builtin_amdgcn_perm(D, C, sel));
                hw[8*h + 2*b]     = __builtin_bit_cast(uint32_t, (p0*k256 + m5)*sc);
                hw[8*h + 2*b + 1] = __builtin_bit_cast(uint32_t, (p1*k256 + m5)*sc);
            }
        }
    } else if constexpr (WT == T_IQ2_S) {
        const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) raw[2]);
        const uint32_t qh = (raw[2] >> 16) & 255, sc = raw[2] >> 24;
        const _Float16 sa = (_Float16) (d*(0.5f + (float) (sc & 15))*0.25f), sb = (_Float16) (d*(0.5f + (float) (sc >> 4))*0.25f);
#pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint32_t idx = ((raw[0] >> (8*l)) & 255) | (((qh >> (2*l)) & 3) << 8);
            const u4 g = ((const u4 *) sgrid)[idx];
            const uint32_t sg = (raw[1] >> (8*l)) & 255;
            const u2 ma = ssign[sg & 15], mb = ssign[sg >> 4];
            const h2 scl = l >= 2 ? h2{sb, sb} : h2{sa, sa};
            hw[4*l]     = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, g.x ^ ma.x)*scl);
            hw[4*l + 1] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, g.y ^ ma.y)*scl);
            hw[4*l + 2] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, g.z ^ mb.x)*scl);
            hw[4*l + 3] = __builtin_bit_cast(uint32_t, __builtin_bit_cast(h2, g.w ^ mb.y)*scl);
        }
    } else {
        const float d = (float) __builtin_bit_cast(_Float16, (uint16_t) raw[8]);
        float d0, d1;
        if constexpr (WT == T_IQ3_KS) {
            d0 = d1 = d*(float) ((int) raw[9] - 16);
        } else {
            const uint32_t sl = raw[9] & 255, sh = raw[9] >> 8;
            d0 = d*(float) ((int) (2*(sl & 15) + 1)*((sh & 1) ? -1 : 1));
            d1 = d*(float) ((int) (2*(sl >> 4) + 1)*((sh & 2) ? -1 : 1));
        }
        const h2 s0 = {(_Float16) d0, (_Float16) d0}, s1 = {(_Float16) d1, (_Float16) d1};
#pragma unroll
        for (int m = 0; m < 8; m++) {
            conv4(table16_one(raw[m], 0xF6E9D8C1u, 0x2F1C0D01u, 0xFAEDDCC5u, 0x33201105u), m < 4 ? s0 : s1, hw[2*m], hw[2*m + 1]);
        }
    }
#pragma unroll
    for (int k = 0; k < 4; k++) wdst[k] = u4{hw[4*k], hw[4*k + 1], hw[4*k + 2], hw[4*k + 3]};
}

struct exp_args {
    const uint64_t * xw0, * xw1;   // each expert's rows (GU: gate, up; down: xw0), of row_bytes each
    int row_bytes;
    const _Float16 * X;       // GU: the activation [n_tok][2560]; down: H [rows][640]
    const int32_t * ids;      // each routed row's token*n_used + slot
    int n_used;
    _Float16 * out16;         // GU: H [rows][640]
    float * out;              // down summed: [n_tok] rows
    float * outp[5];          // down, not summed: its rows at ids; piece i holds the rows from out_row0[i] on
    int out_row0[5];
    int64_t out_stride;       // floats between two rows
    const float * sum_w;      // summed: the routing weight of each (token, slot)
    const moe_f16_tile * tiles; int n_tiles;
    // IQ2_S: the codebook as FP16 values [1024][8] in global memory (ggml_cuda_moe_f16_iq2s_grid). In shared memory its
    // 16 KB left room for two blocks a WGP only, and the compiler took 254 registers: 152 and three blocks without it.
    // 81 920 rows of 128 experts, random weights (standalone bench): gate/up 10.87 -> 9.00 ms
    const uint8_t * grid16;
};

// the IQ2_S codebook (bytes) as FP16 values
static __global__ void k_iq2s_grid16(uint8_t * out) {
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < 2048; i += gridDim.x*blockDim.x) {
        const uint32_t g = ((const uint32_t *) iq2s_grid)[i];
        const h2 lo = {(_Float16) (float) (g & 255), (_Float16) (float) ((g >> 8) & 255)};
        const h2 hh = {(_Float16) (float) ((g >> 16) & 255), (_Float16) (float) (g >> 24)};
        ((u2 *) out)[i] = u2{__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hh)};
    }
}

// GU: gate and up of 64 features (128 weight rows, interleaved) with their SwiGLU, written as FP16 rows; else down, 128
// output rows, SUM: weighted into its token's row
// at least 6 waves a SIMD (240 registers at most), which leaves room for a third block a WGP: the compiler took 246
// (down Q2_0), 247 (down IQ4_NL) and 256 (gate/up IQ4_NL), a choice and not a need: 238, 147 and 152 with the bound
// (Strata-rdna3's STRATA_NH_WPE does the same for its Q2_0 down). Standalone bench, 81 920 rows of 128 experts: gate/up
// IQ4_NL 8.75 -> 7.93 ms, down IQ4_NL 4.98 -> 4.30. IQ2_S gate/up keeps its 254 (two blocks a WGP by its shared memory)
template <int WT, bool GU, bool SUM>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6))) k_exp_f16(const exp_args a) {
    constexpr int K = GU ? 2560 : 640, NS = K/64, NFB = (GU ? 640 : 2560) / (GU ? 64 : 128);
    constexpr bool HG = WT == T_IQ2_S;
    __shared__ __attribute__((aligned(32))) _Float16 smem[(128 + 128)*HLD];
    const uint8_t * sgrid = a.grid16;
    __shared__ u2 ssign[16];
    __shared__ int srow[128];
    _Float16 (* wt)[HLD] = (_Float16 (*)[HLD]) smem;
    _Float16 (* bt)[HLD] = (_Float16 (*)[HLD]) (smem + 128*HLD);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave & 3, wn = wave >> 2, l16 = lane & 15, hi = lane >> 4;
    const int ur = tid >> 1, uj = tid & 1;
    if constexpr (HG) {   // a sign nibble as the FP16 sign bits of its 4 values
        if (tid < 16) {
            ssign[tid] = u2{(tid & 1)*0x8000u | ((tid >> 1) & 1)*0x80000000u, ((tid >> 2) & 1)*0x8000u | ((tid >> 3) & 1)*0x80000000u};
        }
    }
    const int nwork = a.n_tiles*NFB;
    for (int w = blockIdx.x; w < nwork; w += gridDim.x) {
        const moe_f16_tile t = a.tiles[w / NFB];
        const int fb = w % NFB, nrows = t.n;
        const uint8_t * wrow = GU ? (const uint8_t *) (uintptr_t) ((ur & 1) ? a.xw1 : a.xw0)[t.e] + (int64_t) (fb*64 + (ur >> 1))*a.row_bytes
                                  : (const uint8_t *) (uintptr_t) a.xw0[t.e] + (int64_t) (fb*128 + ur)*a.row_bytes;
        u4 * bdst = (u4 *) &bt[ur][32*uj];
        u4 * wdst = (u4 *) &wt[ur][32*uj];
        uint32_t raw[raw_words(WT)];
        uint32_t pl[16];
        __syncthreads();   // the previous item is done with the buffers (and the tables are written)
        if (tid < 128) {
            const int r = t.row0 + min(tid, nrows - 1);   // rows past the item's end repeat its last one
            srow[tid] = GU ? a.ids[r] / a.n_used : r;
        }
        load_unit<WT>(wrow, 0, uj, raw, pl);
        __syncthreads();
        const u4 * bsrc = (const u4 *) (a.X + (int64_t) srow[ur]*K + 32*uj);
        u4 b0 = bsrc[0], b1 = bsrc[1], b2 = bsrc[2], b3 = bsrc[3];
        put_unit<WT>(raw, wdst, sgrid, ssign);
        bdst[0] = b0; bdst[1] = b1; bdst[2] = b2; bdst[3] = b3;
        load_unit<WT>(wrow, 1, uj, raw, pl);
        b0 = bsrc[8]; b1 = bsrc[9]; b2 = bsrc[10]; b3 = bsrc[11];
        const bool on = 64*wn < nrows;
        const int ncol = min(4, (nrows - 64*wn + 15) >> 4);
        v8f acc[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; mt++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) acc[mt][nt] = (v8f) 0.0f;
#pragma unroll 1
        for (int s = 0; s < NS; s++) {
            __syncthreads();   // stage s is in LDS
            if (on) {
                if (ncol == 4) {
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
                } else {
#pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const v16h A0 = ldf(&wt[32*wm + l16][16*q]), A1 = ldf(&wt[32*wm + 16 + l16][16*q]);
#pragma unroll
                        for (int nt = 0; nt < 3; nt++) {
                            if (nt < ncol) {
                                const v16h Bn = ldf(&bt[64*wn + 16*nt + l16][16*q]);
                                acc[0][nt] = wmma(A0, Bn, acc[0][nt]);
                                acc[1][nt] = wmma(A1, Bn, acc[1][nt]);
                            }
                        }
                    }
                }
            }
            if (s + 1 < NS) {
                __syncthreads();   // every wave is done with stage s
                put_unit<WT>(raw, wdst, sgrid, ssign);
                bdst[0] = b0; bdst[1] = b1; bdst[2] = b2; bdst[3] = b3;
                if (s + 2 < NS) {
                    load_unit<WT>(wrow, s + 2, uj, raw, pl);
                    b0 = bsrc[8*(s + 2)]; b1 = bsrc[8*(s + 2) + 1]; b2 = bsrc[8*(s + 2) + 2]; b3 = bsrc[8*(s + 2) + 3];
                }
            }
        }
        if (!on) {
            continue;
        }
        if constexpr (GU) {
            // lane l < 16: for token column l, the gate of features 16 wm + 8 mt + i; their up rows in lane l + 16
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    float hv[8];
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const float up = __shfl_xor(acc[mt][nt][i], 16);
                        const float gt = acc[mt][nt][i];
                        hv[i] = gt / (1.0f + __expf(-gt)) * up;
                    }
                    const int r = 64*wn + 16*nt + l16;
                    if (hi == 0 && r < nrows) {
                        uint32_t hw[4];
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const h2 v = {(_Float16) fminf(fmaxf(hv[2*j], -65504.0f), 65504.0f), (_Float16) fminf(fmaxf(hv[2*j + 1], -65504.0f), 65504.0f)};
                            hw[j] = __builtin_bit_cast(uint32_t, v);
                        }
                        *(u4 *) (a.out16 + (int64_t) (t.row0 + r)*640 + fb*64 + 16*wm + 8*mt) = u4{hw[0], hw[1], hw[2], hw[3]};
                    }
                }
        } else {
#pragma unroll
            for (int mt = 0; mt < 2; mt++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    const int r = 64*wn + 16*nt + l16;
                    if (r < nrows) {
                        const int idx = a.ids[t.row0 + r];
                        if constexpr (SUM) {   // weighted into its token's row; the adds' order varies
                            float * d = a.out + (int64_t) (idx / a.n_used)*a.out_stride + fb*128 + 32*wm + 16*mt + hi;
                            const float wgt = a.sum_w[idx];
#pragma unroll
                            for (int i = 0; i < 8; i++) unsafeAtomicAdd(d + 2*i, wgt*acc[mt][nt][i]);
                        } else {
                            const int pc = (idx >= a.out_row0[1]) + (idx >= a.out_row0[2]) + (idx >= a.out_row0[3]) + (idx >= a.out_row0[4]);
                            float * d = a.outp[pc] + (int64_t) (idx - a.out_row0[pc])*a.out_stride + fb*128 + 32*wm + 16*mt + hi;
#pragma unroll
                            for (int i = 0; i < 8; i++) d[2*i] = acc[mt][nt][i];
                        }
                    }
                }
        }
    }
}

// the activation of a prompt batch as FP16 rows
__global__ void k_moe_f16_act(const float * __restrict__ x, const int64_t x_stride, _Float16 * __restrict__ y, const int64_t n_embd) {
    const float * xs = x + (int64_t) blockIdx.x*x_stride;
    _Float16 * ys = y + (int64_t) blockIdx.x*n_embd;
    for (int64_t i = threadIdx.x; i < n_embd/2; i += blockDim.x) {
        const float2 v = ((const float2 *) xs)[i];
        const h2 h = {(_Float16) fminf(fmaxf(v.x, -65504.0f), 65504.0f), (_Float16) fminf(fmaxf(v.y, -65504.0f), 65504.0f)};
        ((uint32_t *) ys)[i] = __builtin_bit_cast(uint32_t, h);
    }
}

template <int WT, bool GU, bool SUM> void launch(const exp_args & a, const int n_items, cudaStream_t stream) {
    // a block an item: the items are of unequal work (an expert's last tile has fewer rows), which blocks walking a fixed
    // share of them (504, GGML_CUDA_MOE_F16_BLOCKS=504) do not balance. 81 920 rows of 128 experts (standalone bench on
    // Strata-rdna3's pf_f16_bench data): gate/up IQ3_KS 9.41 -> 8.70 ms, IQ3_K 9.14 -> 8.56, IQ4_NL 9.09 -> 8.79, IQ2_S
    // 10.11 -> 9.99; down IQ4_NL 4.63 -> 4.34, Q2_0 4.60 -> 4.35
    static const int blocks = getenv("GGML_CUDA_MOE_F16_BLOCKS") ? atoi(getenv("GGML_CUDA_MOE_F16_BLOCKS")) : 0;
    const unsigned grid = (unsigned) (blocks > 0 ? std::min(n_items, blocks) : n_items);
    {   // GGML_CUDA_MOE_F16_REGS=1: each instance's registers and shared memory, once
        static bool said = false;
        if (!said && getenv("GGML_CUDA_MOE_F16_REGS")) {
            said = true;
            hipFuncAttributes fa{};
            (void) hipFuncGetAttributes(&fa, (const void *) k_exp_f16<WT, GU, SUM>);
            GGML_LOG_INFO("k_exp_f16 type %d %s%s: %d registers, %d B of shared memory, grid %d for %d items\n", WT, GU ? "gate/up" : "down", SUM ? " summed" : "",
                fa.numRegs, (int) fa.sharedSizeBytes, (int) grid, n_items);
        }
    }
    k_exp_f16<WT, GU, SUM><<<grid, 256, 0, stream>>>(a);
}

} // namespace
#endif // defined(GGML_USE_HIP)

#if defined(MOE_F16_HIP)
// down's separate rows (row t n_used + k: the k-th expert of token t; piece i holds the rows from row0[i] on) weighted
// and summed into the tokens' rows, 4 values a thread
struct moe_f16_pieces {
    const float * p[5];
    int row0[5];
};

static __global__ void k_moe_f16_combine(const moe_f16_pieces D, const float * __restrict__ w, float * __restrict__ out,
        const int64_t out_stride, const int n_used, const int n) {
    const int t = blockIdx.x;
    float wk[16];
    const float4 * rk[16];
#pragma unroll
    for (int k = 0; k < 16; k++) {
        const int r  = t*n_used + min(k, n_used - 1);
        const int pc = (r >= D.row0[1]) + (r >= D.row0[2]) + (r >= D.row0[3]) + (r >= D.row0[4]);
        wk[k] = k < n_used ? w[t*n_used + k] : 0.0f;
        rk[k] = (const float4 *) (D.p[pc] + (int64_t) (r - D.row0[pc])*n);
    }
    for (int c = threadIdx.x; c < n/4; c += blockDim.x) {
        float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        for (int k = 0; k < n_used; k++) {
            const float4 v = rk[k][c];
            s.x += wk[k]*v.x; s.y += wk[k]*v.y; s.z += wk[k]*v.z; s.w += wk[k]*v.w;
        }
        float * o = out + t*out_stride + 4*c;
        o[0] = s.x; o[1] = s.y; o[2] = s.z; o[3] = s.w;
    }
}
#endif // defined(MOE_F16_HIP)

// down's separate rows [n_tok n_used][n] in 5 pieces of memory (16-byte aligned, piece_rows[i] rows in pieces[i], in
// order) into out[n_tok] rows of out_stride floats, weighted by w
void ggml_cuda_moe_f16_combine(float * const * pieces, const int64_t * piece_rows, const float * w, const int64_t n_tok, const int n_used,
        const int64_t n, float * out, const int64_t out_stride, cudaStream_t stream) {
#if defined(MOE_F16_HIP)
    GGML_ASSERT(n_used >= 1 && n_used <= 16 && n % 4 == 0 && n_tok*n_used <= INT32_MAX);
    moe_f16_pieces D;
    int64_t row0 = 0;
    for (int i = 0; i < 5; i++) {
        GGML_ASSERT((uintptr_t) pieces[i] % 16 == 0);
        D.p[i]    = pieces[i];
        D.row0[i] = (int) std::min<int64_t>(row0, INT32_MAX);
        row0 += piece_rows[i];
    }
    GGML_ASSERT(row0 >= n_tok*n_used);
    k_moe_f16_combine<<<(unsigned) n_tok, 256, 0, stream>>>(D, w, out, out_stride, n_used, (int) n);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(pieces, piece_rows, w, n_tok, n_used, n, out, out_stride, stream);
#endif // defined(MOE_F16_HIP)
}

// the grouped MoE block of a prompt by these kernels, for the types they decode at the model's sizes, on RDNA3
// (GGML_CUDA_MOE_F16=0: MMQ)
bool ggml_cuda_moe_f16_applies(const int cc, const ggml_tensor * wg, const ggml_tensor * wd) {
#if defined(MOE_F16_HIP)
    static const bool on = !getenv("GGML_CUDA_MOE_F16") || atoi(getenv("GGML_CUDA_MOE_F16")) != 0;
    const auto gu_ok = [](ggml_type t) { return t == GGML_TYPE_IQ2_S || t == GGML_TYPE_IQ3_K || t == GGML_TYPE_IQ3_KS || t == GGML_TYPE_IQ4_NL; };
    const auto d_ok  = [](ggml_type t) { return t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_Q2_0; };
    // DIAGNOSTIC GGML_CUDA_MOE_F16_ONLY_GU=T / _ONLY_D=T: only the layers whose gate/up (down) is of ggml type T
    static const int only_gu = getenv("GGML_CUDA_MOE_F16_ONLY_GU") ? atoi(getenv("GGML_CUDA_MOE_F16_ONLY_GU")) : -1;
    static const int only_d  = getenv("GGML_CUDA_MOE_F16_ONLY_D")  ? atoi(getenv("GGML_CUDA_MOE_F16_ONLY_D"))  : -1;
    if ((only_gu >= 0 && (int) wg->type != only_gu) || (only_d >= 0 && (int) wd->type != only_d)) {
        return false;
    }
    return on && GGML_CUDA_CC_IS_RDNA3(cc) && gu_ok(wg->type) && d_ok(wd->type) &&
        wg->ne[0] == 2560 && wg->ne[1] == 640 && wd->ne[0] == 640 && wd->ne[1] == 2560 &&
        wg->nb[1] == (size_t) (2560/block_vals(wg->type)*block_bytes(wg->type)) &&
        wd->nb[1] == (size_t) ( 640/block_vals(wd->type)*block_bytes(wd->type));
#else
    GGML_UNUSED_VARS(cc, wg, wd);
    return false;
#endif // defined(MOE_F16_HIP)
}

// the bytes its buffers take: the activation [n_tok][2560] and H [rows][640] as FP16, the tiles, a tail
size_t ggml_cuda_moe_f16_bytes(const int64_t n_tok, const int64_t rows, const int64_t n_expert) {
    return (size_t) n_tok*2560*2 + (size_t) rows*640*2 + (size_t) (rows/128 + n_expert + 1)*sizeof(moe_f16_tile) + 256;
}

// the layer's tiles from the rows' bounds by expert (tile_first: n_expert + 1), and the activation as FP16.
// buf: ggml_cuda_moe_f16_bytes, aligned to 64 bytes
void ggml_cuda_moe_f16_begin(char * buf, const float * x, const int64_t x_stride, const int64_t n_tok, const int64_t rows,
        const std::vector<int32_t> & bounds, std::vector<int32_t> & tile_first, cudaStream_t stream) {
#if defined(MOE_F16_HIP)
    const int64_t n_expert = (int64_t) bounds.size() - 1;
    std::vector<moe_f16_tile> tiles;
    tile_first.assign(n_expert + 1, 0);
    for (int64_t e = 0; e < n_expert; e++) {
        tile_first[e] = (int32_t) tiles.size();
        for (int r = bounds[e]; r < bounds[e + 1]; r += 128) {
            tiles.push_back({(int) e, r, std::min(128, bounds[e + 1] - r)});
        }
    }
    tile_first[n_expert] = (int32_t) tiles.size();
    char * tl = buf + (size_t) n_tok*2560*2 + (size_t) rows*640*2;
    if (!tiles.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(tl, tiles.data(), tiles.size()*sizeof(moe_f16_tile), cudaMemcpyHostToDevice, stream));
    }
    k_moe_f16_act<<<(unsigned) n_tok, 256, 0, stream>>>(x, x_stride, (_Float16 *) buf, 2560);
#else
    GGML_UNUSED_VARS(buf, x, x_stride, n_tok, rows, bounds, tile_first, stream);
#endif // defined(MOE_F16_HIP)
}

// the IQ2_S codebook as FP16 values on the current device (16 KB, made once and kept), ready for the kernels launched on
// `stream` after this call
const uint8_t * ggml_cuda_moe_f16_iq2s_grid(cudaStream_t stream) {
#if defined(MOE_F16_HIP)
    static std::mutex mutex;
    static uint8_t * grids[GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(mutex);
    const int device = ggml_cuda_get_device();
    if (grids[device] == nullptr) {
        CUDA_CHECK(cudaMalloc(&grids[device], 1024*16));
        k_iq2s_grid16<<<8, 256, 0, stream>>>(grids[device]);
        CUDA_CHECK(cudaGetLastError());
        // the kernels of any stream read it from now on
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    return grids[device];
#else
    GGML_UNUSED(stream);
    return nullptr;
#endif // defined(MOE_F16_HIP)
}

// the experts e0 .. e1 - 1 (their weights on the device): gate, up and SwiGLU into H, then down.
// xe: each expert's gate, up and down rows (3 x n_expert addresses, on the device); ids: each row's token*n_used + slot;
// sum_w: ggml_mul_mat_id_sum's weights (down then holds a row per token, zeroed before the first group), else nullptr;
// down_pieces: down's rows, not summed, in 5 pieces of memory of down_piece_rows[i] rows, else in down
void ggml_cuda_moe_f16_group(char * buf, const int64_t n_tok, const int64_t rows, const int64_t n_expert, const std::vector<int32_t> & tile_first,
        const int64_t e0, const int64_t e1, const ggml_type type_gu, const ggml_type type_d, const uint64_t * xe, const size_t row_gu,
        const size_t row_d, const int32_t * ids, const int n_used, float * down, const int64_t down_stride, const float * sum_w,
        cudaStream_t stream, float * const * down_pieces, const int64_t * down_piece_rows) {
#if defined(MOE_F16_HIP)
    const int n_tiles = tile_first[e1] - tile_first[e0];
    if (n_tiles <= 0) {
        return;
    }
    _Float16 * x16 = (_Float16 *) buf, * h16 = x16 + (size_t) n_tok*2560;
    const moe_f16_tile * tiles = (const moe_f16_tile *) (buf + (size_t) n_tok*2560*2 + (size_t) rows*640*2) + tile_first[e0];
    exp_args a = {};
    a.ids = ids; a.n_used = n_used; a.tiles = tiles; a.n_tiles = n_tiles;
    a.grid16 = type_gu == GGML_TYPE_IQ2_S ? ggml_cuda_moe_f16_iq2s_grid(stream) : nullptr;
    a.xw0 = xe; a.xw1 = xe + n_expert; a.row_bytes = (int) row_gu; a.X = x16; a.out16 = h16;
    switch (type_gu) {
        case GGML_TYPE_IQ2_S:  launch<T_IQ2_S,  true, false>(a, n_tiles*10, stream); break;
        case GGML_TYPE_IQ3_K:  launch<T_IQ3_K,  true, false>(a, n_tiles*10, stream); break;
        case GGML_TYPE_IQ3_KS: launch<T_IQ3_KS, true, false>(a, n_tiles*10, stream); break;
        case GGML_TYPE_IQ4_NL: launch<T_IQ4_NL, true, false>(a, n_tiles*10, stream); break;
        default: GGML_ABORT("fatal error");
    }
    a.xw0 = xe + 2*n_expert; a.xw1 = nullptr; a.row_bytes = (int) row_d; a.X = h16; a.out16 = nullptr;
    a.out = down; a.out_stride = down_stride; a.sum_w = sum_w;
    int64_t row0 = 0;
    for (int i = 0; i < 5; i++) {
        a.outp[i]     = down_pieces ? down_pieces[i] : down;
        a.out_row0[i] = down_pieces ? (int) std::min<int64_t>(row0, INT32_MAX) : (i == 0 ? 0 : INT32_MAX);
        row0 += down_pieces ? down_piece_rows[i] : 0;
    }
    if (type_d == GGML_TYPE_IQ4_NL) {
        if (sum_w) launch<T_IQ4_NL, false, true>(a, n_tiles*20, stream); else launch<T_IQ4_NL, false, false>(a, n_tiles*20, stream);
    } else if (type_d == GGML_TYPE_Q2_0) {
        if (sum_w) launch<T_Q2_0, false, true>(a, n_tiles*20, stream); else launch<T_Q2_0, false, false>(a, n_tiles*20, stream);
    } else {
        GGML_ABORT("fatal error");
    }
#else
    GGML_UNUSED_VARS(buf, n_tok, rows, n_expert, tile_first, e0, e1, type_gu, type_d, xe, row_gu, row_d, ids, n_used, down, down_stride, sum_w, stream, down_pieces, down_piece_rows);
#endif // defined(MOE_F16_HIP)
}
