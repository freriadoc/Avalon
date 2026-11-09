#include "gdn-wmma.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <mma.h>
#endif

#include <algorithm>
#include <cstdlib>

// The gated delta rule of a prompt batch in chunks of 16 tokens, on the RDNA3 matrix cores: Strata's gfx11 kernels
// (src/prefill/kernels.cu, gdn_wmma_prep_kernel and gdn_wmma_scan_kernel, MIT, Niko1221 and the Strata
// contributors), read by strides from our q, k, v, gate and beta, with our transposed state, and without the output
// norm (our gated norm follows). Per value head, gamma_t the gate's sum from the chunk's start to t (inclusive) and
// S0 the state before the chunk:
//   A[t][i] = beta_t exp(gamma_t - gamma_i) k_t.k_i (i < t),  T = (I + A)^-1,  P[t][i] = exp(gamma_t - gamma_i) q_t.k_i (i <= t),
//   Y = beta (V - exp(gamma) K S0),  D = T Y,  o = exp(gamma) Q S0 + P D,
//   S = exp(gamma_last) S0 + K^T (exp(gamma_last - gamma) D).
// The prep kernel computes T and P of every chunk (they do not depend on the state); the scan walks the chunks, a
// block per value head and a wave per 16 value columns, the state in the waves' WMMA accumulators (FP32). k, q, T, P,
// D and the state as an operand are rounded to FP16.

#if defined(GGML_USE_HIP) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
                              defined(__gfx1150__) || defined(__gfx1151__))
#define GDN_WMMA_DEVICE 1
#endif

namespace {

constexpr int GW_S   = 128;            // head size
constexpr int GW_HK  = 16;             // key heads
constexpr int GW_HV  = 48;             // value heads
constexpr int GW_CK  = 16;             // tokens per chunk (a WMMA tile)
constexpr int GW_NW  = GW_S / 16;      // the scan's waves per block: one per 16 value columns
constexpr int GW_VPK = GW_HV / GW_HK;  // value heads per key head: the prep's waves per block
constexpr int GW_KP  = GW_S + 8;       // the scan's padded FP16 k / q rows
constexpr int GW_AHEAD = 3;            // the scan warms the cache this many chunks ahead

#ifdef GGML_USE_HIP
typedef _Float16 gw_h16 __attribute__((ext_vector_type(16)));
typedef _Float16 gw_h8  __attribute__((ext_vector_type(8)));
typedef _Float16 gw_h2  __attribute__((ext_vector_type(2)));
typedef float    gw_f8  __attribute__((ext_vector_type(8)));
typedef float    gw_f4  __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ gw_f8 gw_mma(gw_h16 a, gw_h16 b, gw_f8 c) {
#ifdef GDN_WMMA_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b;
    __builtin_trap();   // not an RDNA3 device pass: the host gate never launches it
    return c;
#endif
}

// An accumulator tile (lane l: rows 2v + (l >> 4) for v = 0..7, column l & 15) as a B operand (lane l: column l & 15,
// rows 0..15): pairs of rows rounded to FP16 and packed, the other half's pairs fetched across (permlanex16), then
// interleaved by a byte permute whose selector depends on the lane's half.
struct gw_sel_t { unsigned even, odd; };
static __device__ __forceinline__ gw_sel_t gw_sel(int hi) {
    return hi ? gw_sel_t{0x01000504u, 0x03020706u} : gw_sel_t{0x05040100u, 0x07060302u};
}
static __device__ __forceinline__ gw_h16 gw_acc_to_b(gw_f8 x, gw_sel_t ps) {
    unsigned wd[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const gw_h2 p = {(_Float16) x[2 * m], (_Float16) x[2 * m + 1]};
        const unsigned own = __builtin_bit_cast(unsigned, p);
#ifdef GDN_WMMA_DEVICE
        const unsigned other = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
#else
        const unsigned other = own;
#endif
        wd[2 * m]     = __builtin_amdgcn_perm(other, own, ps.even);
        wd[2 * m + 1] = __builtin_amdgcn_perm(other, own, ps.odd);
    }
    return __builtin_bit_cast(gw_h16, wd);
}
static __device__ __forceinline__ gw_h16 gw_ld(const _Float16 * p) {   // 32 bytes of LDS, 16-byte aligned
    const uint4 a = *reinterpret_cast<const uint4 *>(p), b = *reinterpret_cast<const uint4 *>(p + 8);
    const uint4 ab[2] = {a, b};
    return __builtin_bit_cast(gw_h16, ab);
}
// the wave's own LDS writes complete (read back across its lanes)
static __device__ __forceinline__ void gw_lds_wait() {
#ifdef GDN_WMMA_DEVICE
    __asm__ volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
#endif
}
// a barrier over LDS only: __syncthreads() would also wait for the next chunk's global loads in flight
static __device__ __forceinline__ void gw_lds_barrier() {
#ifdef GDN_WMMA_DEVICE
    __asm__ volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __syncthreads();
#endif
}

// A block per (key head, chunk), a wave per value head of the key head: T then P, FP16 row-major (1 KB), at
// tp[(chunk * HV + value head) * 512].
__global__ void __launch_bounds__(32 * GW_VPK) gdn_wmma_prep(const ggml_cuda_gdn_wmma_args a, _Float16 * __restrict__ tp) {
    __shared__ float sa[GW_VPK][GW_CK * GW_CK], sg[GW_VPK][GW_CK], sb[GW_VPK][GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * GW_HK;
    const int64_t t0 = (int64_t) ch * GW_CK, T = a.T;
    gw_f8 kk = {}, qk = {};   // k_t.k_i and q_t.k_i (lane: column i = c, rows t = 2v + hi)
    {
        const bool ok = t0 + c < T;   // past the end: zero rows
        const int64_t tt = ok ? t0 + c : 0;
        const float * kt = a.k + tt * a.sq2 + qh * a.sq1;
        const float * qt = a.q + tt * a.sq2 + qh * a.sq1;
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            gw_h16 ka, qa;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const gw_f4 k4 = *reinterpret_cast<const gw_f4 *>(kt + 16 * j + 4 * i);
                const gw_f4 q4 = *reinterpret_cast<const gw_f4 *>(qt + 16 * j + 4 * i);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    ka[4 * i + e] = (_Float16) (ok ? k4[e] : 0.0f);
                    qa[4 * i + e] = (_Float16) (ok ? q4[e] : 0.0f);
                }
            }
            kk = gw_mma(ka, ka, kk);   // B = K^T: lane i holds column i of K^T, row i of K - the A operand itself
            qk = gw_mma(qa, ka, qk);
        }
    }
    if (l < GW_CK) {   // past the end: gate 0, beta 0
        const bool ok = t0 + l < T;
        float g = ok ? a.g[(t0 + l) * a.sb2 + vh * a.sb1] : 0.0f;
#pragma unroll
        for (int o = 1; o < GW_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GW_CK);
            if (l >= o) {
                g += u;
            }
        }
        sg[w][l] = g;
        sb[w][l] = ok ? a.beta[(t0 + l) * a.sb2 + vh * a.sb1] : 0.0f;
    }
    gw_lds_wait();
    _Float16 * out = tp + ((int64_t) ch * GW_HV + vh) * (2 * GW_CK * GW_CK);
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const int t = 2 * v + hi;
        sa[w][t * GW_CK + c] = c < t ? sb[w][t] * __expf(sg[w][t] - sg[w][c]) * kk[v] : 0.0f;
        out[GW_CK * GW_CK + t * GW_CK + c] = (_Float16) (c <= t ? __expf(sg[w][t] - sg[w][c]) * qk[v] : 0.0f);
    }
    gw_lds_wait();
    if (l < GW_CK) {   // lane c: column c of T = (I + A)^-1 by forward substitution
        float tc[GW_CK];
#pragma unroll
        for (int t = 0; t < GW_CK; ++t) {
            float v = t == c ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) {
                v = fmaf(-sa[w][t * GW_CK + k], tc[k], v);
            }
            tc[t] = v;
            out[t * GW_CK + c] = (_Float16) v;
        }
    }
}

// A thread's slice of a chunk in the scan (token threadIdx / 16, columns (threadIdx % 16) * 8 .. + 7): k, q and v,
// loaded a chunk ahead, and one word of the chunk's T and P; lanes 0..15 of wave 0 also gate and beta of token l.
// Rows past the end read the last row (finite: their gate and beta become 0 at staging).
struct gw_chunk {
    gw_f4 k0, k1, q0, q1, v0, v1;
    float g, b;
    unsigned tp;
};
static __device__ __forceinline__ void gw_load(gw_chunk & in, const ggml_cuda_gdn_wmma_args & a, const unsigned * __restrict__ tp,
                                               int64_t t0, int vh, int qh) {
    const int t = threadIdx.x >> 4, r = (threadIdx.x & 15) * 8;
    const int64_t tt = min(t0 + t, a.T - 1);
    in.tp = tp[((t0 / GW_CK) * GW_HV + vh) * (GW_CK * GW_CK) + threadIdx.x];
    const float * kt = a.k + tt * a.sq2 + qh * a.sq1 + r;
    const float * qt = a.q + tt * a.sq2 + qh * a.sq1 + r;
    const float * vt = a.v + tt * a.sv2 + vh * a.sv1 + r;
    in.k0 = *reinterpret_cast<const gw_f4 *>(kt);
    in.k1 = *reinterpret_cast<const gw_f4 *>(kt + 4);
    in.q0 = *reinterpret_cast<const gw_f4 *>(qt);
    in.q1 = *reinterpret_cast<const gw_f4 *>(qt + 4);
    in.v0 = *reinterpret_cast<const gw_f4 *>(vt);
    in.v1 = *reinterpret_cast<const gw_f4 *>(vt + 4);
    if (threadIdx.x < GW_CK) {
        const int64_t tl = min(t0 + (int64_t) threadIdx.x, a.T - 1);
        in.g = a.g   [tl * a.sb2 + vh * a.sb1];
        in.b = a.beta[tl * a.sb2 + vh * a.sb1];
    }
}
// brings a later chunk's k, q and v lines into the cache with a load whose result is consumed (by an empty asm) one
// chunk later, so the real loads of that chunk hit the cache
static __device__ __forceinline__ unsigned gw_touch(const ggml_cuda_gdn_wmma_args & a, int64_t t0, int vh, int qh) {
    const int t = threadIdx.x >> 4, part = threadIdx.x & 15, kind = part >> 2;
    if (t0 + t >= a.T || kind == 3) {
        return 0u;
    }
    const float * p = kind == 0 ? a.k + (t0 + t) * a.sq2 + qh * a.sq1
                    : kind == 1 ? a.q + (t0 + t) * a.sq2 + qh * a.sq1
                                : a.v + (t0 + t) * a.sv2 + vh * a.sv1;
    return __float_as_uint(p[32 * (part & 3)]);
}

// Per chunk: stage it (k / q FP16, k also transposed, v, the gate's running sum, beta, T and P); barrier (1);
// X = K S0, QS = Q S0; Y, D, o (written out); the state update; barrier (2), before the next chunk's staging
// overwrites what the slower waves still read.
__global__ void __launch_bounds__(GW_NW * 32) gdn_wmma_scan(const ggml_cuda_gdn_wmma_args a, const unsigned * __restrict__ tp) {
    __shared__ __attribute__((aligned(16))) _Float16 sk[GW_CK * GW_KP], sq[GW_CK * GW_KP], skt[GW_S * GW_CK],
        stp[2 * GW_CK * GW_CK];
    __shared__ __attribute__((aligned(16))) float sv[GW_CK * GW_S];
    __shared__ float sg[GW_CK], sb[GW_CK];
    const _Float16 * st_ = stp, * sp_ = stp + GW_CK * GW_CK;   // T, P
    const int vh = blockIdx.x, qh = vh % GW_HK, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int col = 16 * w + c;                                       // this lane's column in the products
    const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;    // this thread's staging slice
    const int64_t T = a.T;
    const gw_sel_t ps = gw_sel(hi);
    // our state is transposed: M[col][i] = S[i][col], the column contiguous
    const float * s_in = a.state + (a.state_ids ? (int64_t) a.state_ids[0] * a.state_row_stride : 0) + (int64_t) vh * GW_S * GW_S + col * GW_S;
    gw_f8 s[GW_S / 16];   // the state: rows 16j + 2v + hi, column col
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            s[j][v] = s_in[16 * j + 2 * v + hi];
        }
    }
    const float out_scale = rsqrtf((float) GW_S);
    gw_chunk cur;
    gw_load(cur, a, tp, 0, vh, qh);
    unsigned touched = 0;
    for (int64_t t0 = 0; t0 < T; t0 += GW_CK) {
        {   // stage this chunk
            gw_h8 kh, qq;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                kh[i] = (_Float16) (i < 4 ? cur.k0[i] : cur.k1[i - 4]);
                qq[i] = (_Float16) (i < 4 ? cur.q0[i] : cur.q1[i - 4]);
                skt[(sr + i) * GW_CK + st] = kh[i];
            }
            *reinterpret_cast<uint4 *>(&sk[st * GW_KP + sr]) = __builtin_bit_cast(uint4, kh);
            *reinterpret_cast<uint4 *>(&sq[st * GW_KP + sr]) = __builtin_bit_cast(uint4, qq);
            *reinterpret_cast<gw_f4 *>(&sv[st * GW_S + sr])     = cur.v0;
            *reinterpret_cast<gw_f4 *>(&sv[st * GW_S + sr + 4]) = cur.v1;
            reinterpret_cast<unsigned *>(stp)[threadIdx.x] = cur.tp;
            if (threadIdx.x < GW_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + l < T;
                float g = ok ? cur.g : 0.0f;
#pragma unroll
                for (int o = 1; o < GW_CK; o <<= 1) {
                    const float u = __shfl_up(g, o, GW_CK);
                    if (l >= o) {
                        g += u;
                    }
                }
                sg[l] = g;
                sb[l] = ok ? cur.b : 0.0f;
            }
        }
        if (t0 + GW_CK < T) {
            gw_load(cur, a, tp, t0 + GW_CK, vh, qh);   // lands during this chunk
        }
        const unsigned touch_now = gw_touch(a, t0 + GW_AHEAD * GW_CK, vh, qh);
        gw_lds_barrier();   // (1) this chunk staged
        __asm__ volatile("" ::"v"(touched));   // the previous touch, consumed here
        touched = touch_now;
        gw_f8 x = {}, qs = {};   // K S0 and Q S0, this wave's columns
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            const gw_h16 b = gw_acc_to_b(s[j], ps);
            x  = gw_mma(gw_ld(sk + c * GW_KP + 16 * j), b, x);
            qs = gw_mma(gw_ld(sq + c * GW_KP + 16 * j), b, qs);
        }
        gw_f8 y;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = 2 * v + hi;
            y[v] = sb[t] * (sv[t * GW_S + col] - __expf(sg[t]) * x[v]);
        }
        const gw_f8 d = gw_mma(gw_ld(st_ + c * GW_CK), gw_acc_to_b(y, ps), gw_f8{});
        gw_f8 o;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            o[v] = __expf(sg[2 * v + hi]) * qs[v];
        }
        o = gw_mma(gw_ld(sp_ + c * GW_CK), gw_acc_to_b(d, ps), o);
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int64_t t = t0 + 2 * v + hi;
            if (t < T) {
                a.dst[(t * GW_HV + vh) * GW_S + col] = o[v] * out_scale;
            }
        }
        const float gl = sg[GW_CK - 1];
        gw_f8 dd;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            dd[v] = __expf(gl - sg[2 * v + hi]) * d[v];
        }
        const gw_h16 db = gw_acc_to_b(dd, ps);
        const float eg = __expf(gl);
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            s[j] = gw_mma(gw_ld(skt + (16 * j + c) * GW_CK), db, s[j] * eg);
        }
        gw_lds_barrier();   // (2)
    }
    __asm__ volatile("" ::"v"(touched));
    float * s_out = a.state_out + (int64_t) vh * GW_S * GW_S + col * GW_S;
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            s_out[16 * j + 2 * v + hi] = s[j][v];
        }
    }
}
#endif // GGML_USE_HIP

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// CUDA (sm_70 and later): the same chunked algorithm on the tensor cores through nvcuda::wmma (16x16x16, FP16
// operands, FP32 accumulation), its fragments passed through shared memory where the RDNA3 kernels move registers
// across lanes. The chunk algebra is Strata's (src/prefill/kernels.cu and gdn_tc.cuh, MIT, Niko1221 and the Strata
// contributors), as in the HIP kernels above; the prep's dot products in FP32.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define GDN_TC_CUDA_DEVICE 1
#endif

constexpr int GC_SW = 16 * 16;   // a 16 x 16 tile

// A block per (key head, chunk), GW_CK * GW_CK threads: k_t.k_i and q_t.k_i in FP32 (shared by the key head's value
// heads), then for each value head T = (I + A)^-1 by forward substitution and P, FP16 row-major at
// tp[(chunk * HV + value head) * 512] (T, then P), as gdn_wmma_prep.
__global__ void __launch_bounds__(GW_CK * GW_CK) gdn_tc_prep_cuda(const ggml_cuda_gdn_wmma_args a, __half * __restrict__ tp) {
    __shared__ float sk[GW_CK][GW_S + 1], sq[GW_CK][GW_S + 1];
    __shared__ float skk[GW_CK][GW_CK], sqk[GW_CK][GW_CK], sa[GW_CK][GW_CK + 1], sg[GW_CK], sb[GW_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x;
    const int64_t t0 = (int64_t) ch * GW_CK, T = a.T;
    for (int idx = tid; idx < GW_CK * GW_S; idx += blockDim.x) {
        const int t = idx / GW_S, i = idx % GW_S;
        const bool ok = t0 + t < T;
        sk[t][i] = ok ? a.k[(t0 + t) * a.sq2 + qh * a.sq1 + i] : 0.0f;
        sq[t][i] = ok ? a.q[(t0 + t) * a.sq2 + qh * a.sq1 + i] : 0.0f;
    }
    __syncthreads();
    {
        const int t = tid / GW_CK, i = tid % GW_CK;
        float kk = 0.0f, qk = 0.0f;
        for (int e = 0; e < GW_S; ++e) {
            kk = fmaf(sk[t][e], sk[i][e], kk);
            qk = fmaf(sq[t][e], sk[i][e], qk);
        }
        skk[t][i] = kk;
        sqk[t][i] = qk;
    }
    for (int w = 0; w < GW_VPK; ++w) {
        const int vh = qh + w * GW_HK;
        __syncthreads();   // the previous value head's reads of sa, sg, sb are done
        if (tid < GW_CK) {   // past the end: gate 0, beta 0
            const bool ok = t0 + tid < T;
            sg[tid] = ok ? a.g[(t0 + tid) * a.sb2 + vh * a.sb1] : 0.0f;
            sb[tid] = ok ? a.beta[(t0 + tid) * a.sb2 + vh * a.sb1] : 0.0f;
        }
        __syncthreads();
        if (tid == 0) {   // the gate's running sum over the chunk (16 values)
            for (int t = 1; t < GW_CK; ++t) {
                sg[t] += sg[t - 1];
            }
        }
        __syncthreads();
        __half * out = tp + ((int64_t) ch * GW_HV + vh) * (2 * GC_SW);
        {
            const int t = tid / GW_CK, c = tid % GW_CK;
            sa[t][c] = c < t ? sb[t] * __expf(sg[t] - sg[c]) * skk[t][c] : 0.0f;
            out[GC_SW + t * GW_CK + c] = __float2half(c <= t ? __expf(sg[t] - sg[c]) * sqk[t][c] : 0.0f);
        }
        __syncthreads();
        if (tid < GW_CK) {   // thread c: column c of T
            const int c = tid;
            float tc[GW_CK];
#pragma unroll
            for (int t = 0; t < GW_CK; ++t) {
                float v = t == c ? 1.0f : 0.0f;
#pragma unroll
                for (int k = 0; k < t; ++k) {
                    v = fmaf(-sa[t][k], tc[k], v);
                }
                tc[t] = v;
                out[t * GW_CK + c] = __float2half(v);
            }
        }
    }
}

// A block per value head, a warp per 16 value columns (GW_NW warps): the state as GW_S / 16 accumulator fragments a
// warp (rows 16j.., its columns), as gdn_wmma_scan. Per chunk: staged (k, q FP16, k transposed, v, the gate's running
// sum, beta, T and P), then per warp X = K S0 and QS = Q S0, Y = beta (V - exp(gamma) X), D = T Y,
// o = exp(gamma) QS + P D, S = exp(gamma_last) S0 + K^T (exp(gamma_last - gamma) D); a fragment becomes an operand
// through the warp's own tiles in shared memory.
__global__ void __launch_bounds__(GW_NW * 32) gdn_tc_scan_cuda(const ggml_cuda_gdn_wmma_args a, const __half * __restrict__ tp) {
#ifdef GDN_TC_CUDA_DEVICE
    using namespace nvcuda;
    __shared__ __align__(32) __half sk[GW_CK * GW_KP], sq[GW_CK * GW_KP], skt[GW_S * GW_CK], stp[2 * GC_SW];
    __shared__ __align__(32) float sv[GW_CK * GW_S];
    __shared__ __align__(32) float wf[GW_NW][GC_SW];     // a warp's FP32 tile
    __shared__ __align__(32) float wf2[GW_NW][GC_SW];    // a second one (QS)
    __shared__ __align__(32) __half wh[GW_NW][GC_SW];    // a warp's FP16 operand tile
    __shared__ __align__(32) __half wh2[GW_NW][GC_SW];   // a second one (the state update's)
    __shared__ float sg[GW_CK], sb[GW_CK];
    const int vh = blockIdx.x, qh = vh % GW_HK, w = threadIdx.x >> 5, l = threadIdx.x & 31;
    const int64_t T = a.T;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa_t;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb_t;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc_t;
    // the state: s[j] = S[16j .. 16j + 15][16w .. 16w + 15]; ours is transposed, M[col][i] = S[i][col]
    const float * s_in = a.state + (a.state_ids ? (int64_t) a.state_ids[0] * a.state_row_stride : 0) + (int64_t) vh * GW_S * GW_S;
    fc_t s[GW_S / 16];
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
        for (int e = l; e < GC_SW; e += 32) {
            const int i = e / 16, c = e % 16;   // row i of the tile, column c
            wf[w][e] = s_in[(int64_t) (16 * w + c) * GW_S + 16 * j + i];
        }
        __syncwarp();
        wmma::load_matrix_sync(s[j], wf[w], 16, wmma::mem_row_major);
        __syncwarp();
    }
    const float out_scale = rsqrtf((float) GW_S);
    // the warp's FP32 tile (row-major) as an FP16 operand tile
    auto to_half = [&](const float * src) {
        for (int e = l; e < GC_SW; e += 32) {
            wh[w][e] = __float2half(src[e]);
        }
        __syncwarp();
    };
    for (int64_t t0 = 0; t0 < T; t0 += GW_CK) {
        {   // stage this chunk: thread (token st, values sr .. sr + 7); rows past the end read the last row
            const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;
            const int64_t tt = min(t0 + st, T - 1);
            const float * kt = a.k + tt * a.sq2 + qh * a.sq1 + sr;
            const float * qt = a.q + tt * a.sq2 + qh * a.sq1 + sr;
            const float * vt = a.v + tt * a.sv2 + vh * a.sv1 + sr;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __half kh = __float2half(kt[i]);
                sk[st * GW_KP + sr + i] = kh;
                sq[st * GW_KP + sr + i] = __float2half(qt[i]);
                skt[(sr + i) * GW_CK + st] = kh;
                sv[st * GW_S + sr + i] = vt[i];
            }
            const __half * tpc = tp + ((t0 / GW_CK) * GW_HV + vh) * (2 * GC_SW);
            for (int e = threadIdx.x; e < 2 * GC_SW; e += blockDim.x) {
                stp[e] = tpc[e];
            }
            if (threadIdx.x < GW_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + threadIdx.x < T;
                sg[threadIdx.x] = ok ? a.g   [(t0 + threadIdx.x) * a.sb2 + vh * a.sb1] : 0.0f;
                sb[threadIdx.x] = ok ? a.beta[(t0 + threadIdx.x) * a.sb2 + vh * a.sb1] : 0.0f;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                for (int t = 1; t < GW_CK; ++t) {
                    sg[t] += sg[t - 1];
                }
            }
            __syncthreads();   // (1) this chunk staged
        }
        fc_t x, qs;
        wmma::fill_fragment(x, 0.0f);
        wmma::fill_fragment(qs, 0.0f);
#pragma unroll
        for (int j = 0; j < GW_S / 16; ++j) {
            wmma::store_matrix_sync(wf[w], s[j], 16, wmma::mem_row_major);
            __syncwarp();
            to_half(wf[w]);
            fb_t b;
            wmma::load_matrix_sync(b, wh[w], 16);
            fa_t ak, aq;
            wmma::load_matrix_sync(ak, sk + 16 * j, GW_KP);
            wmma::load_matrix_sync(aq, sq + 16 * j, GW_KP);
            wmma::mma_sync(x, ak, b, x);
            wmma::mma_sync(qs, aq, b, qs);
            __syncwarp();
        }
        // Y = beta (V - exp(gamma) X), this warp's columns
        wmma::store_matrix_sync(wf[w], x, 16, wmma::mem_row_major);
        wmma::store_matrix_sync(wf2[w], qs, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = l; e < GC_SW; e += 32) {
            const int t = e / 16, c = e % 16;
            wf[w][e] = sb[t] * (sv[t * GW_S + 16 * w + c] - __expf(sg[t]) * wf[w][e]);
        }
        __syncwarp();
        to_half(wf[w]);
        fc_t d;
        {
            wmma::fill_fragment(d, 0.0f);
            fa_t at;
            fb_t by;
            wmma::load_matrix_sync(at, stp, GW_CK);
            wmma::load_matrix_sync(by, wh[w], 16);
            wmma::mma_sync(d, at, by, d);
        }
        // o = exp(gamma) QS + P D; D as an operand, and exp(gamma_last - gamma) D for the state update
        const float gl = sg[GW_CK - 1];
        wmma::store_matrix_sync(wf[w], d, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = l; e < GC_SW; e += 32) {
            const int t = e / 16;
            wh [w][e] = __float2half(wf[w][e]);
            wh2[w][e] = __float2half(__expf(gl - sg[t]) * wf[w][e]);
        }
        __syncwarp();
        {
            fc_t o;
            wmma::fill_fragment(o, 0.0f);
            fa_t ap;
            fb_t bd;
            wmma::load_matrix_sync(ap, stp + GC_SW, GW_CK);
            wmma::load_matrix_sync(bd, wh[w], 16);
            wmma::mma_sync(o, ap, bd, o);
            __syncwarp();
            const float * ot = wf2[w];   // QS
            wmma::store_matrix_sync(wf[w], o, 16, wmma::mem_row_major);   // P D (wf held D, now in wh / wh2)
            __syncwarp();
            for (int e = l; e < GC_SW; e += 32) {
                const int t = e / 16, c = e % 16;
                if (t0 + t < T) {
                    a.dst[((t0 + t) * GW_HV + vh) * GW_S + 16 * w + c] = (__expf(sg[t]) * ot[e] + wf[w][e]) * out_scale;
                }
            }
            __syncwarp();
        }
        // S = exp(gamma_last) S0 + K^T (exp(gamma_last - gamma) D)
        {
            const float eg = __expf(gl);
            fb_t bdd;
            wmma::load_matrix_sync(bdd, wh2[w], 16);
#pragma unroll
            for (int j = 0; j < GW_S / 16; ++j) {
#pragma unroll
                for (int e = 0; e < s[j].num_elements; ++e) {
                    s[j].x[e] *= eg;
                }
                fa_t akt;
                wmma::load_matrix_sync(akt, skt + 16 * j * GW_CK, GW_CK);
                wmma::mma_sync(s[j], akt, bdd, s[j]);
            }
        }
        __syncthreads();   // (2) before the next chunk's staging overwrites what the slower warps still read
    }
    float * s_out = a.state_out + (int64_t) vh * GW_S * GW_S;
#pragma unroll
    for (int j = 0; j < GW_S / 16; ++j) {
        wmma::store_matrix_sync(wf[w], s[j], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = l; e < GC_SW; e += 32) {
            const int i = e / 16, c = e % 16;
            s_out[(int64_t) (16 * w + c) * GW_S + 16 * j + i] = wf[w][e];
        }
        __syncwarp();
    }
#else
    GGML_UNUSED_VARS(a, tp);
    __trap();
#endif // GDN_TC_CUDA_DEVICE
}
#endif // !GGML_USE_HIP && !GGML_USE_MUSA


#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// ---- GGML_CUDA_GDN_TC2=NP (sm_80 and later): the chunked GDN of a prompt batch in chunks of 32 tokens on mma.sync, the
// state in registers. The structure is Strata's (src/prefill/gdn_tc.cuh, MIT, Niko1221 and the Strata contributors):
// every product transposed so that a warp's slice of the state (16 value columns x 128 rows, S^T) is the mma A operand
// and never leaves its registers; the prep writes the scan's operands as FP16 images (k, q x 2^8, T x 2^et, P x 2^8)
// and the token scalars, which the scan stages by cp.async, double-buffered; super-blocks of 2048 tokens bound the
// scratch; a head split in value tiles (NTL) when the card has the SMs for them. Ours: the operand precision is a
// parameter (NP 1: every product on one FP16 pass, as our RDNA3 kernels and the wmma kernel above; 3: Strata's three
// passes hi.hi + hi.lo + lo.hi, FP32-level; 2: three for the state's products, one for the output's), only the images
// the precision reads are written and staged (one pass: half the scratch and the staging, and v, T and P
// double-buffered too, one barrier a chunk), the q / k / v / gate / beta read by strides, our transposed state read in
// the recurrent cache, the tiles chosen from the SM count for any card.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define GDN_TC2_DEVICE 1
#endif

constexpr int G2_CH = 32;              // tokens per chunk
constexpr int G2_SB = 2048;            // tokens per super-block (one prep and one scan launch)
constexpr int G2_NB = G2_SB / G2_CH;   // chunks per super-block
constexpr int G2_KP = GW_S + 8;        // FP16 k / q rows in shared memory: 272 B, the 8 rows of an ldmatrix on 8 16-B lanes
constexpr int G2_PP = G2_CH + 8;       // FP16 T / P rows: 80 B
constexpr int G2_GP = G2_CH + 1;       // FP32 Gram rows of the prep
constexpr int G2_CW = GW_S / 16;       // warps of a whole head: 16 value columns each

// the images of a (chunk, key head) in the order k hi, q hi, [k lo], [q lo]; of a (chunk, value head) T hi, P hi, [T lo],
// [P lo]: the low halves only for the products on three passes (NPS: the state's products, k.S, Y.T and the update;
// NPO: the output's, q.S and D.P)
static __host__ __device__ constexpr int g2_nimg(int nps, int npo) { return 2 + (nps == 3) + (npo == 3); }
static __host__ __device__ constexpr int g2_klo(int nps)           { return nps == 3 ? 2 : 0; }
static __host__ __device__ constexpr int g2_qlo(int nps, int npo)  { return npo == 3 ? (nps == 3 ? 3 : 2) : 1; }

struct g2_scratch {
    float4   * sc;   // [NB][HV][CH] (beta, exp(gamma), exp(gL - gamma), gamma)
    uint16_t * kq;   // [NB][HK][nimg][CH][S]
    uint16_t * tp;   // [NB][HV][nimg][CH][CH]
    float    * te;   // [NB][HV] 2^-et
};
static size_t g2_scratch_bytes(int nimg) {
    return (size_t) G2_NB * GW_HV * G2_CH * sizeof(float4) + (size_t) G2_NB * GW_HK * nimg * G2_CH * GW_S * 2 +
           (size_t) G2_NB * GW_HV * nimg * G2_CH * G2_CH * 2 + (size_t) G2_NB * GW_HV * sizeof(float);
}
static g2_scratch g2_carve(void * p, int nimg) {
    g2_scratch r;
    unsigned char * b = (unsigned char *) p;
    r.sc = (float4 *) b;   b += (size_t) G2_NB * GW_HV * G2_CH * sizeof(float4);
    r.kq = (uint16_t *) b; b += (size_t) G2_NB * GW_HK * nimg * G2_CH * GW_S * 2;
    r.tp = (uint16_t *) b; b += (size_t) G2_NB * GW_HV * nimg * G2_CH * G2_CH * 2;
    r.te = (float *) b;
    return r;
}

static __device__ __forceinline__ void g2_cp16(void * smem, const void * gmem) {
#ifdef GDN_TC2_DEVICE
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    GGML_UNUSED_VARS(smem, gmem);
#endif
}
static __device__ __forceinline__ void g2_cp4(void * smem, const void * gmem) {
#ifdef GDN_TC2_DEVICE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    GGML_UNUSED_VARS(smem, gmem);
#endif
}
static __device__ __forceinline__ void g2_cp_commit() {
#ifdef GDN_TC2_DEVICE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
static __device__ __forceinline__ void g2_cp_wait_all() {
#ifdef GDN_TC2_DEVICE
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}
// the warp's largest of non-negative floats (their bits order as unsigned integers), under the full mask
static __device__ __forceinline__ float g2_warp_max(float v) {
#ifdef GDN_TC2_DEVICE
    return __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(v)));
#else
    return v;
#endif
}
static __device__ __forceinline__ float g2_pow2(int e) { return __int_as_float((e + 127) << 23); }   // 2^e, |e| < 127
// the power of two that puts m = max |x| in [2^14, 2^15): FP16 hi never overflows and the low part of every entry within
// 2^-17 of the max stays normal; clamped far from FP32's range
static __device__ __forceinline__ int g2_scale_exp(float m) {
    if (m == 0.0f) {
        return 0;
    }
    const int e = 14 - ((__float_as_int(m) >> 23) - 127);
    return e < -40 ? -40 : (e > 40 ? 40 : e);
}
// m16n8k16 fragments (PTX ISA "Matrix Fragments", lane = 4 gid + tig): A a0 = (row gid, cols 2tig, 2tig+1), a1 = (gid+8, ..),
// a2 = (gid, 2tig+8..), a3 = (gid+8, 2tig+8..); B b0 = (k 2tig, 2tig+1; n gid), b1 = (k 2tig+8..; n gid); C c0 = (gid, 2tig),
// c1 = (gid, 2tig+1), c2 = (gid+8, 2tig), c3 = (gid+8, 2tig+1): two adjacent C tiles are the next product's A fragment
static __device__ __forceinline__ void g2_ldsm_x4(uint32_t (&r)[4], const void * smem) {
#ifdef GDN_TC2_DEVICE
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; GGML_UNUSED(smem);
#endif
}
static __device__ __forceinline__ void g2_ldsm_x4_trans(uint32_t (&r)[4], const void * smem) {
#ifdef GDN_TC2_DEVICE
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned) __cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u; GGML_UNUSED(smem);
#endif
}
static __device__ __forceinline__ void g2_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef GDN_TC2_DEVICE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
#endif
}
// a product on NP passes: 3 = lo.hi, hi.lo, hi.hi (small terms first); 1 = hi.hi (the B lo image is not read)
template <int NP>
static __device__ __forceinline__ void g2_mmaN(float (&c)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0,
                                               uint32_t bh1, uint32_t bl0, uint32_t bl1) {
    if constexpr (NP == 3) {
        g2_mma(c, al, bh0, bh1);
        g2_mma(c, ah, bl0, bl1);
    }
    g2_mma(c, ah, bh0, bh1);
}
// x0, x1 -> packed FP16 hi (x0 in the low half) and lo = f16(x - hi) (x - hi exact in FP32)
static __device__ __forceinline__ void g2_split(float x0, float x1, uint32_t & hi, uint32_t & lo) {
    union { __half2 h; uint32_t u; } cv;
    cv.h = __floats2half2_rn(x0, x1);
    hi = cv.u;
    const float2 r = __half22float2(cv.h);
    cv.h = __floats2half2_rn(x0 - r.x, x1 - r.y);
    lo = cv.u;
}
static __device__ __forceinline__ void g2_store4(uint16_t * hi, uint16_t * lo, float4 v, float scale) {
    uint32_t h0, l0, h1, l1;
    g2_split(v.x * scale, v.y * scale, h0, l0);
    g2_split(v.z * scale, v.w * scale, h1, l1);
    *reinterpret_cast<uint2 *>(hi) = make_uint2(h0, h1);
    if (lo) {
        *reinterpret_cast<uint2 *>(lo) = make_uint2(l0, l1);
    }
}
// a float4 of shared memory the compiler does not keep in registers (the token scalars, read in three phases)
static __device__ __forceinline__ float4 g2_lds128(const float4 * p) {
    float4 v;
#ifdef GDN_TC2_DEVICE
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"((unsigned) __cvta_generic_to_shared(p)));
#else
    v = *p;
#endif
    return v;
}

// X^T = S0^T K^T (NPK passes) and QS^T = S0^T Q^T (NPQ): A = the state x sA split per k16-step, B = the k / q images
// ([CH][KP], rows = tokens) by ldmatrix. Out x 2^(8 + es).
template <int NPK, int NPQ>
static __device__ __forceinline__ void g2_prod_state(const float (&s)[16][4], float sA, const uint16_t * kh, const uint16_t * kl,
        const uint16_t * qh, const uint16_t * ql, float (&xt)[4][4], float (&qs)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * G2_KP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int kk = 0; kk < GW_S / 16; ++kk) {
        uint32_t ah[4], al[4];
        g2_split(s[2 * kk][0] * sA,     s[2 * kk][1] * sA,     ah[0], al[0]);
        g2_split(s[2 * kk][2] * sA,     s[2 * kk][3] * sA,     ah[1], al[1]);
        g2_split(s[2 * kk + 1][0] * sA, s[2 * kk + 1][1] * sA, ah[2], al[2]);
        g2_split(s[2 * kk + 1][2] * sA, s[2 * kk + 1][3] * sA, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int o = off + 16 * hf * G2_KP + 16 * kk;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            g2_ldsm_x4(bh, kh + o);
            if constexpr (NPK == 3) {
                g2_ldsm_x4(bl, kl + o);
            }
            g2_mmaN<NPK>(xt[2 * hf],     ah, al, bh[0], bh[1], bl[0], bl[1]);
            g2_mmaN<NPK>(xt[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            g2_ldsm_x4(bh, qh + o);
            if constexpr (NPQ == 3) {
                g2_ldsm_x4(bl, ql + o);
            }
            g2_mmaN<NPQ>(qs[2 * hf],     ah, al, bh[0], bh[1], bl[0], bl[1]);
            g2_mmaN<NPQ>(qs[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// out^T += (sD X^T) B^T for a lower-triangular B (T or P images, [CH][PP], rows = tokens): tokens t < 16 (n-tiles 0, 1)
// see no i >= 16 (k-step 1), skipped
template <int NP>
static __device__ __forceinline__ void g2_prod_tri(const float (&x)[4][4], float sD, const uint16_t * bhp, const uint16_t * blp,
        float (&ot)[4][4]) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * G2_PP + 8 * ((lane >> 3) & 1);
#pragma unroll
    for (int ks = 0; ks < G2_CH / 16; ++ks) {
        uint32_t ah[4], al[4];
        g2_split(x[2 * ks][0] * sD,     x[2 * ks][1] * sD,     ah[0], al[0]);
        g2_split(x[2 * ks][2] * sD,     x[2 * ks][3] * sD,     ah[1], al[1]);
        g2_split(x[2 * ks + 1][0] * sD, x[2 * ks + 1][1] * sD, ah[2], al[2]);
        g2_split(x[2 * ks + 1][2] * sD, x[2 * ks + 1][3] * sD, ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            if (ks == 1 && hf == 0) {
                continue;
            }
            const int o = off + 16 * hf * G2_PP + 16 * ks;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            g2_ldsm_x4(bh, bhp + o);
            if constexpr (NP == 3) {
                g2_ldsm_x4(bl, blp + o);
            }
            g2_mmaN<NP>(ot[2 * hf],     ah, al, bh[0], bh[1], bl[0], bl[1]);
            g2_mmaN<NP>(ot[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}
// S^T += Dd^T (2^8 K): B = the k images read the other way by ldmatrix.trans
template <int NP>
static __device__ __forceinline__ void g2_update_state(float (&s)[16][4], const float (&dd)[4][4], const uint16_t * kh,
        const uint16_t * kl) {
    const int lane = threadIdx.x & 31;
    const int off = (8 * ((lane >> 3) & 1) + (lane & 7)) * G2_KP + 8 * (lane >> 4);
#pragma unroll
    for (int ks = 0; ks < G2_CH / 16; ++ks) {
        uint32_t ah[4], al[4];
        g2_split(dd[2 * ks][0],     dd[2 * ks][1],     ah[0], al[0]);
        g2_split(dd[2 * ks][2],     dd[2 * ks][3],     ah[1], al[1]);
        g2_split(dd[2 * ks + 1][0], dd[2 * ks + 1][1], ah[2], al[2]);
        g2_split(dd[2 * ks + 1][2], dd[2 * ks + 1][3], ah[3], al[3]);
#pragma unroll
        for (int hf = 0; hf < GW_S / 16; ++hf) {
            const int o = off + 16 * ks * G2_KP + 16 * hf;
            uint32_t bh[4], bl[4] = {0u, 0u, 0u, 0u};
            g2_ldsm_x4_trans(bh, kh + o);
            if constexpr (NP == 3) {
                g2_ldsm_x4_trans(bl, kl + o);
            }
            g2_mmaN<NP>(s[2 * hf],     ah, al, bh[0], bh[1], bl[0], bl[1]);
            g2_mmaN<NP>(s[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}

// The prep, a block per (chunk of the super-block, key head), 256 threads: the k and q images (to shared memory, hi and
// lo, and to the scratch, the ones the scan reads), the Gram products k_t.k_i and q_t.k_i on three FP16 passes
// (FP32-level whatever NP: T = (I + A)^-1 amplifies their error), then warps 0..2 (value heads qh + j HK) A, T by forward
// substitution (a lane per column) and T's images x 2^et, warps 3..5 P x 2^8 and the token scalars.
constexpr size_t G2_PREP_SMEM = (size_t) 4 * G2_CH * G2_KP * 2 + 2 * G2_CH * G2_GP * 4;
static_assert(GW_VPK * G2_CH * G2_GP * 4 <= 4 * G2_CH * G2_KP * 2, "A fits where the images were");

template <int NPS, int NPO>
__global__ void __launch_bounds__(256) gdn_tc2_prep(const ggml_cuda_gdn_wmma_args a, const g2_scratch sp, const int64_t t0) {
#ifdef GDN_TC2_DEVICE
    constexpr int NIMG = g2_nimg(NPS, NPO);
    extern __shared__ __align__(16) float g2_psm[];
    uint16_t * im  = reinterpret_cast<uint16_t *>(g2_psm);   // [4][CH][KP] k hi, k lo, q hi, q lo (x 2^8)
    float    * skk = g2_psm + 4 * G2_CH * G2_KP / 2;          // [CH][GP] k_t.k_i
    float    * sqk = skk + G2_CH * G2_GP;                     // [CH][GP] q_t.k_i
    float    * sa  = g2_psm;                                  // [VPK][CH][GP] A (over the images, read by then)
    const int qh = blockIdx.y, ch = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int64_t tb = t0 + (int64_t) ch * G2_CH;
    const int n = (int) min((int64_t) G2_CH, a.T - tb);
    const int j = warp % GW_VPK, vh = qh + j * GW_HK, i = lane;
    const bool pv = warp < 2 * GW_VPK && i < n;   // past the end: gate 0, beta 0
    const float b = pv ? a.beta[(tb + i) * a.sb2 + vh * a.sb1] : 0.0f;
    float gam     = pv ? a.g   [(tb + i) * a.sb2 + vh * a.sb1] : 0.0f;
    {
        constexpr int NR = G2_CH * GW_S / 4 / 256;   // float4 a thread, each of k and q
        float4 kv[NR], qv[NR];
#pragma unroll
        for (int m = 0; m < NR; ++m) {
            const int p = tid + 256 * m, t = p / (GW_S / 4), r = (p % (GW_S / 4)) * 4;
            const float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            kv[m] = t < n ? *reinterpret_cast<const float4 *>(a.k + (tb + t) * a.sq2 + qh * a.sq1 + r) : z4;
            qv[m] = t < n ? *reinterpret_cast<const float4 *>(a.q + (tb + t) * a.sq2 + qh * a.sq1 + r) : z4;
        }
        uint16_t * img = sp.kq + ((size_t) ch * GW_HK + qh) * NIMG * G2_CH * GW_S;
#pragma unroll
        for (int m = 0; m < NR; ++m) {
            const int p = tid + 256 * m, t = p / (GW_S / 4), r = (p % (GW_S / 4)) * 4;
            const int o = t * G2_KP + r, og = t * GW_S + r;
            g2_store4(im + o, im + G2_CH * G2_KP + o, kv[m], 256.0f);
            g2_store4(im + 2 * G2_CH * G2_KP + o, im + 3 * G2_CH * G2_KP + o, qv[m], 256.0f);
            g2_store4(img + og, NPS == 3 ? img + g2_klo(NPS) * G2_CH * GW_S + og : nullptr, kv[m], 256.0f);
            g2_store4(img + G2_CH * GW_S + og, NPO == 3 ? img + g2_qlo(NPS, NPO) * G2_CH * GW_S + og : nullptr, qv[m], 256.0f);
        }
    }
    __syncthreads();
    {   // warp w: product w >> 2 (k.k, q.k), rows 16 ((w >> 1) & 1).., columns 16 (w & 1)..; out x 2^16
        const int pr = warp >> 2, mt = (warp >> 1) & 1, np = warp & 1;
        const uint16_t * ah_ = im + 2 * pr * G2_CH * G2_KP + (16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1)) * G2_KP + 8 * (lane >> 4);
        const uint16_t * bh_ = im + (16 * np + 8 * (lane >> 4) + (lane & 7)) * G2_KP + 8 * ((lane >> 3) & 1);
        float c[2][4] = {};
#pragma unroll
        for (int kk = 0; kk < GW_S / 16; ++kk) {
            uint32_t ah[4], al[4], bh[4], bl[4];
            g2_ldsm_x4(ah, ah_ + 16 * kk);
            g2_ldsm_x4(al, ah_ + G2_CH * G2_KP + 16 * kk);
            g2_ldsm_x4(bh, bh_ + 16 * kk);
            g2_ldsm_x4(bl, bh_ + G2_CH * G2_KP + 16 * kk);
            g2_mmaN<3>(c[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
            g2_mmaN<3>(c[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
        float * g = pr ? sqk : skk;
        const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 16 * mt + gid + 8 * (q >> 1), ii = 16 * np + 8 * jj + 2 * tig + (q & 1);
                g[t * G2_GP + ii] = ii <= t ? c[jj][q] * g2_pow2(-16) : 0.0f;
            }
        }
    }
    __syncthreads();
    if (warp >= 2 * GW_VPK) {
        return;
    }
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) {
            gam += u;
        }
    }
    uint16_t * tpb = sp.tp + ((size_t) ch * GW_HV + vh) * NIMG * G2_CH * G2_CH;
    if (warp >= GW_VPK) {   // P = exp(gamma_t - gamma_i) q_t.k_i (i <= t) x 2^8; the scalars
        uint16_t * Ph = tpb + G2_CH * G2_CH;
        uint16_t * Pl = tpb + g2_qlo(NPS, NPO) * G2_CH * G2_CH;
#pragma unroll
        for (int t = 0; t < G2_CH; ++t) {
            const float gt = __shfl_sync(0xffffffffu, gam, t);
            const float p = (i <= t ? expf(gt - gam) * sqk[t * G2_GP + i] : 0.0f) * 256.0f;
            const __half hi = __float2half_rn(p);
            Ph[t * G2_CH + i] = __half_as_ushort(hi);
            if constexpr (NPO == 3) {
                Pl[t * G2_CH + i] = __half_as_ushort(__float2half_rn(p - __half2float(hi)));
            }
        }
        const float gL = __shfl_sync(0xffffffffu, gam, G2_CH - 1);
        sp.sc[((size_t) ch * GW_HV + vh) * G2_CH + i] = make_float4(b, expf(gam), expf(gL - gam), gam);
        return;
    }
    float * A = sa + j * G2_CH * G2_GP;
#pragma unroll
    for (int t = 0; t < G2_CH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t), bt = __shfl_sync(0xffffffffu, b, t);
        A[t * G2_GP + i] = i < t ? bt * expf(gt - gam) * skk[t * G2_GP + i] : (i == t ? 1.0f : 0.0f);
    }
    __syncwarp();
    // T = (I + A)^-1 row by row, lane i keeping column i in registers (A's row t a broadcast load a term)
    float tv[G2_CH];
    tv[0] = i == 0 ? 1.0f : 0.0f;
#pragma unroll
    for (int t = 1; t < G2_CH; ++t) {
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < t; ++k) {
            v = fmaf(-A[t * G2_GP + k], tv[k], v);
        }
        tv[t] = i < t ? v : (i == t ? 1.0f : 0.0f);
    }
    float mt4[4] = {};
#pragma unroll
    for (int t = 0; t < G2_CH; ++t) {
        mt4[t & 3] = fmaxf(mt4[t & 3], fabsf(tv[t]));
    }
    const int et = g2_scale_exp(g2_warp_max(fmaxf(fmaxf(mt4[0], mt4[1]), fmaxf(mt4[2], mt4[3]))));
    const float sT = g2_pow2(et);
    uint16_t * Th = tpb;
    uint16_t * Tl = tpb + g2_klo(NPS) * G2_CH * G2_CH;
#pragma unroll
    for (int t = 0; t < G2_CH; ++t) {
        const float x = tv[t] * sT;
        const __half hi = __float2half_rn(x);
        Th[t * G2_CH + i] = __half_as_ushort(hi);
        if constexpr (NPS == 3) {
            Tl[t * G2_CH + i] = __half_as_ushort(__float2half_rn(x - __half2float(hi)));
        }
    }
    if (i == 0) {
        sp.te[(size_t) ch * GW_HV + vh] = g2_pow2(-et);
    }
#else
    GGML_UNUSED_VARS(a, sp, t0);
    __trap();
#endif // GDN_TC2_DEVICE
}

// the scan's shared memory: k / q images [2][NIMG][CH][KP], v [NV][CH][VC + 4], T / P images [NV][NIMG][CH][PP], the
// scalars [2][CH], T's scale [2]; with three passes (NPS or NPO 3) v, T and P have one buffer (SPLIT), restaged after the
// chunk's last read of them, so that the plan fits the 99 KB of the sm_86 / sm_89 cards
template <int NPS, int NPO, int NTL>
struct g2_plan {
    static constexpr bool   SPLIT = NPS == 3 || NPO == 3;
    static constexpr int    NIMG  = g2_nimg(NPS, NPO);
    static constexpr int    NV    = SPLIT ? 1 : 2;
    static constexpr int    VC    = GW_S / NTL;    // value columns of a tile
    static constexpr int    VP    = VC + 4;        // lane (gid, tig) reads word 2 tig VP + gid: 32 distinct banks
    static constexpr int    NT    = (G2_CW / NTL) * 32;
    static constexpr size_t KQ    = (size_t) 2 * NIMG * G2_CH * G2_KP * 2;
    static constexpr size_t SV    = (size_t) NV * G2_CH * VP * 4;
    static constexpr size_t TP    = (size_t) NV * NIMG * G2_CH * G2_PP * 2;
    static constexpr size_t SMEM  = KQ + SV + TP + 2 * G2_CH * 16 + 16;
};

// The scan, a block per (value head, tile of VC value columns), a warp per 16 columns, the state slice in registers as
// 16 S^T accumulator tiles a lane; per chunk: (a) X^T = S0^T K^T, QS^T = S0^T Q^T; (b) Y^T = beta (V^T - e X^T);
// (c) D^T = Y^T T^T; (d) o^T = e QS^T + D^T P^T (written out); (e) S^T = eL S0^T + (f D^T) K. Every operand is scaled to
// [2^14, 2^15) by a power of two (exact), so its FP16 images stay in range (and the low halves normal).
template <int NPS, int NPO, int NTL>
__global__ void __launch_bounds__(g2_plan<NPS, NPO, NTL>::NT, 1) gdn_tc2_scan(const ggml_cuda_gdn_wmma_args a, const g2_scratch sp,
        const int64_t t0, const int nch, const bool first) {
#ifdef GDN_TC2_DEVICE
    using P = g2_plan<NPS, NPO, NTL>;
    constexpr int NIMG = P::NIMG, NT = P::NT, VC = P::VC, VP = P::VP;
    extern __shared__ __align__(16) unsigned char g2_ssm[];
    uint16_t * skq = reinterpret_cast<uint16_t *>(g2_ssm);
    float    * sv  = reinterpret_cast<float *>(g2_ssm + P::KQ);
    uint16_t * stp = reinterpret_cast<uint16_t *>(g2_ssm + P::KQ + P::SV);
    float4   * ssc = reinterpret_cast<float4 *>(g2_ssm + P::KQ + P::SV + P::TP);
    float    * ste = reinterpret_cast<float *>(ssc + 2 * G2_CH);
    const int vh = blockIdx.x / NTL, qhd = vh % GW_HK, cb = (blockIdx.x % NTL) * VC;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int64_t T = a.T;
    auto stage = [&](int ch, int b, int parts) {
        const int64_t c_t0 = t0 + (int64_t) ch * G2_CH;
        const int n = (int) min((int64_t) G2_CH, T - c_t0);
        const int bv = P::SPLIT ? 0 : b;
        if (parts & 1) {
            const uint16_t * img = sp.kq + ((size_t) ch * GW_HK + qhd) * NIMG * G2_CH * GW_S;
            for (int pc = tid; pc < NIMG * G2_CH * GW_S / 8; pc += NT) {
                const int m = pc / (G2_CH * GW_S / 8), t = (pc % (G2_CH * GW_S / 8)) / (GW_S / 8), r = (pc % (GW_S / 8)) * 8;
                g2_cp16(skq + ((b * NIMG + m) * G2_CH + t) * G2_KP + r, img + (m * G2_CH + t) * GW_S + r);
            }
            if (tid < G2_CH) {
                g2_cp16(ssc + b * G2_CH + tid, sp.sc + ((size_t) ch * GW_HV + vh) * G2_CH + tid);
            }
            if (tid == 0) {
                g2_cp4(ste + b, sp.te + (size_t) ch * GW_HV + vh);
            }
        }
        if (parts & 2) {
            for (int p = tid; p < G2_CH * VC / 4; p += NT) {
                const int t = p / (VC / 4), r = (p % (VC / 4)) * 4;
                float * dst = sv + (bv * G2_CH + t) * VP + r;
                if (t < n) {
                    g2_cp16(dst, a.v + (c_t0 + t) * a.sv2 + vh * a.sv1 + cb + r);
                } else {
                    *reinterpret_cast<float4 *>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            const uint16_t * tpg = sp.tp + ((size_t) ch * GW_HV + vh) * NIMG * G2_CH * G2_CH;
            for (int pc = tid; pc < NIMG * G2_CH * G2_CH / 8; pc += NT) {
                const int m = pc / (G2_CH * G2_CH / 8), t = (pc % (G2_CH * G2_CH / 8)) / (G2_CH / 8), i = (pc % (G2_CH / 8)) * 8;
                g2_cp16(stp + ((bv * NIMG + m) * G2_CH + t) * G2_PP + i, tpg + (m * G2_CH + t) * G2_CH + i);
            }
        }
        g2_cp_commit();
    };
    const int c0 = 16 * warp;   // the warp's columns c0 + gid and c0 + gid + 8 of the tile
    const int ca = cb + c0;     // in the head
    // our state is transposed, M[col][i] = S[i][col]: s[j][0..1] = S[8j + 2tig .. +1][ca + gid], s[j][2..3] the column + 8
    const float * s_in = first ? a.state + (a.state_ids ? (int64_t) a.state_ids[0] * a.state_row_stride : 0) : a.state_out;
    s_in += (int64_t) vh * GW_S * GW_S;
    float s[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float2 u0 = *reinterpret_cast<const float2 *>(s_in + (int64_t) (ca + gid)     * GW_S + 8 * j + 2 * tig);
        const float2 u1 = *reinterpret_cast<const float2 *>(s_in + (int64_t) (ca + gid + 8) * GW_S + 8 * j + 2 * tig);
        s[j][0] = u0.x; s[j][1] = u0.y; s[j][2] = u1.x; s[j][3] = u1.y;
    }
    stage(0, 0, 3);
    const float out_scale = rsqrtf((float) GW_S);
    for (int ch = 0; ch < nch; ++ch) {
        const int b = ch & 1, bv = P::SPLIT ? 0 : b;
        const int64_t c_t0 = t0 + (int64_t) ch * G2_CH;
        const int n = (int) min((int64_t) G2_CH, T - c_t0);
        g2_cp_wait_all();
        __syncthreads();   // chunk ch's buffers complete; every read of the other ones done
        if (ch + 1 < nch) {
            stage(ch + 1, b ^ 1, P::SPLIT ? 1 : 3);
        }
        const uint16_t * bk  = skq + b * NIMG * G2_CH * G2_KP;
        const uint16_t * bkh = bk;
        const uint16_t * bqh = bk + G2_CH * G2_KP;
        const uint16_t * bkl = bk + g2_klo(NPS) * G2_CH * G2_KP;
        const uint16_t * bql = bk + g2_qlo(NPS, NPO) * G2_CH * G2_KP;
        const uint16_t * bt  = stp + bv * NIMG * G2_CH * G2_PP;
        const uint16_t * bth = bt;
        const uint16_t * bph = bt + G2_CH * G2_PP;
        const uint16_t * btl = bt + g2_klo(NPS) * G2_CH * G2_PP;
        const uint16_t * bpl = bt + g2_qlo(NPS, NPO) * G2_CH * G2_PP;
        const float    * bsv = sv + bv * G2_CH * VP;
        const float4   * bsc = ssc + b * G2_CH;
        // (a) the state x 2^es (its largest entry in [2^14, 2^15)), eight max chains
        float m8[8] = {};
#pragma unroll
        for (int j = 0; j < 16; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                m8[(2 * j + (q >> 1)) & 7] = fmaxf(m8[(2 * j + (q >> 1)) & 7], fabsf(s[j][q]));
            }
        }
        const float m = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])), fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
        const int es = g2_scale_exp(g2_warp_max(m));
        float xt[4][4] = {}, qs[4][4] = {};
        g2_prod_state<NPS, NPO>(s, g2_pow2(es), bkh, bkl, bqh, bql, xt, qs);
        // (b) Y^T = beta_i (V^T - e'_i X^T), e'_i = exp(gamma_i) 2^-(8 + es); lane: tokens 8nt + 2tig (+1), columns c0 + gid (+8)
        const float eS = g2_pow2(-(8 + es));
        float yt[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float4 s0 = g2_lds128(bsc + 8 * nt + 2 * tig), s1 = g2_lds128(bsc + 8 * nt + 2 * tig + 1);
            const float * v = bsv + (8 * nt + 2 * tig) * VP + c0 + gid;
            const float e0 = s0.y * eS, e1 = s1.y * eS;
            yt[nt][0] = s0.x * fmaf(-e0, xt[nt][0], v[0]);
            yt[nt][1] = s1.x * fmaf(-e1, xt[nt][1], v[VP]);
            yt[nt][2] = s0.x * fmaf(-e0, xt[nt][2], v[8]);
            yt[nt][3] = s1.x * fmaf(-e1, xt[nt][3], v[VP + 8]);
        }
        // (c) D^T = Y^T T^T: Y^T x 2^ey, T's images x 2^et; out x 2^(ey + et)
        float my4[4] = {};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                my4[q] = fmaxf(my4[q], fabsf(yt[nt][q]));
            }
        }
        const int ey = g2_scale_exp(g2_warp_max(fmaxf(fmaxf(my4[0], my4[1]), fmaxf(my4[2], my4[3]))));
        float dt[4][4] = {};
        g2_prod_tri<NPS>(yt, g2_pow2(ey), bth, btl, dt);
        const float sYT = g2_pow2(-ey) * ste[b];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                dt[nt][q] *= sYT;
            }
        }
        // (d) o^T = e_t QS^T + D^T P^T: the accumulator starts as 2^(ed + 8) e_t QS^T, D^T in x 2^ed, P x 2^8
        float md4[4] = {};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                md4[q] = fmaxf(md4[q], fabsf(dt[nt][q]));
            }
        }
        const int ed = g2_scale_exp(g2_warp_max(fmaxf(fmaxf(md4[0], md4[1]), fmaxf(md4[2], md4[3]))));
        const float sD = g2_pow2(ed), sO = g2_pow2(ed + 8), sOi = g2_pow2(-(ed + 8)), eO = g2_pow2(ed - es);
        float ot[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float e0 = g2_lds128(bsc + 8 * nt + 2 * tig).y * eO, e1 = g2_lds128(bsc + 8 * nt + 2 * tig + 1).y * eO;
            ot[nt][0] = qs[nt][0] * e0; ot[nt][1] = qs[nt][1] * e1; ot[nt][2] = qs[nt][2] * e0; ot[nt][3] = qs[nt][3] * e1;
        }
        g2_prod_tri<NPO>(dt, sD, bph, bpl, ot);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 8 * nt + 2 * tig + (q & 1);
                if (t < n) {
                    a.dst[((c_t0 + t) * GW_HV + vh) * GW_S + ca + gid + 8 * (q >> 1)] = (ot[nt][q] * sOi) * out_scale;
                }
            }
        }
        if constexpr (P::SPLIT) {   // v, T and P of this chunk are dead: the next chunk's land during (e)
            __syncthreads();
            if (ch + 1 < nch) {
                stage(ch + 1, 0, 2);
            }
        }
        // (e) S^T = eL S0^T + Dd^T K: the accumulator x eL 2^(ed + 8) before, Dd^T = f_i D^T x 2^ed, K x 2^8, x 2^-(ed + 8) after
        const float sPre = g2_lds128(bsc + G2_CH - 1).y * sO;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                s[j][q] *= sPre;
            }
        }
        float dd[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float f0 = g2_lds128(bsc + 8 * nt + 2 * tig).z * sD, f1 = g2_lds128(bsc + 8 * nt + 2 * tig + 1).z * sD;
            dd[nt][0] = dt[nt][0] * f0; dd[nt][1] = dt[nt][1] * f1; dd[nt][2] = dt[nt][2] * f0; dd[nt][3] = dt[nt][3] * f1;
        }
        g2_update_state<NPS>(s, dd, bkh, bkl);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                s[j][q] *= sOi;
            }
        }
    }
    float * s_out = a.state_out + (int64_t) vh * GW_S * GW_S;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        *reinterpret_cast<float2 *>(s_out + (int64_t) (ca + gid)     * GW_S + 8 * j + 2 * tig) = make_float2(s[j][0], s[j][1]);
        *reinterpret_cast<float2 *>(s_out + (int64_t) (ca + gid + 8) * GW_S + 8 * j + 2 * tig) = make_float2(s[j][2], s[j][3]);
    }
#else
    GGML_UNUSED_VARS(a, sp, t0, nch, first);
    __trap();
#endif // GDN_TC2_DEVICE
}

// GGML_CUDA_GDN_TC2: 1 (the default) / 2 / 3 the passes (see above), 0 off; GGML_CUDA_GDN_TC2_TILES=1|2 forces the tiles
static int g2_np() {
    static const int np = [] {
        const char * e = getenv("GGML_CUDA_GDN_TC2");
        const int v = e ? atoi(e) : 1;
        return v == 1 || v == 2 || v == 3 ? v : 0;
    }();
    return np;
}
static int g2_nps(int np) { return np == 1 ? 1 : 3; }
static int g2_npo(int np) { return np == 3 ? 3 : 1; }

template <int NPS, int NPO, int NTL>
static void g2_run(const ggml_cuda_gdn_wmma_args & a, cudaStream_t stream) {
    using P = g2_plan<NPS, NPO, NTL>;
    static bool attr[GGML_CUDA_MAX_DEVICES] = {};
    const int dev = ggml_cuda_get_device();
    if (!attr[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(gdn_tc2_scan<NPS, NPO, NTL>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) P::SMEM));
        attr[dev] = true;
    }
    const g2_scratch sp = g2_carve(a.scratch, P::NIMG);
    for (int64_t t0 = 0; t0 < a.T; t0 += G2_SB) {
        const int nch = (int) ((std::min<int64_t>(G2_SB, a.T - t0) + G2_CH - 1) / G2_CH);
        gdn_tc2_prep<NPS, NPO><<<dim3((unsigned) nch, GW_HK), 256, G2_PREP_SMEM, stream>>>(a, sp, t0);
        gdn_tc2_scan<NPS, NPO, NTL><<<GW_HV * NTL, P::NT, P::SMEM, stream>>>(a, sp, t0, nch, t0 == 0);
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif // !GGML_USE_HIP && !GGML_USE_MUSA

} // namespace

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// the value tiles of a head for the TC2 scan: two when the card holds the 2 x 48 blocks in one wave
static int g2_tiles() {
    static const int forced = getenv("GGML_CUDA_GDN_TC2_TILES") ? atoi(getenv("GGML_CUDA_GDN_TC2_TILES")) : 0;
    if (forced > 0) {
        return forced >= 2 ? 2 : 1;
    }
    return ggml_cuda_info().devices[ggml_cuda_get_device()].nsm >= 2 * GW_HV ? 2 : 1;
}
static size_t g2_smem(int np, int tiles) {
    switch (np * 10 + tiles) {
        case 11: return g2_plan<1, 1, 1>::SMEM;
        case 12: return g2_plan<1, 1, 2>::SMEM;
        case 21: return g2_plan<3, 1, 1>::SMEM;
        case 22: return g2_plan<3, 1, 2>::SMEM;
        case 31: return g2_plan<3, 3, 1>::SMEM;
        default: return g2_plan<3, 3, 2>::SMEM;
    }
}
// GGML_CUDA_GDN_TC2 asked, an sm_80 card with the shared memory of the plan
static bool g2_active() {
    const int np = g2_np();
    if (np == 0) {
        return false;
    }
    const auto & info = ggml_cuda_info().devices[ggml_cuda_get_device()];
    return GGML_CUDA_CC_IS_NVIDIA(info.cc) && info.cc >= GGML_CUDA_CC_AMPERE && info.smpbo >= g2_smem(np, g2_tiles());
}
#endif // !GGML_USE_HIP && !GGML_USE_MUSA

size_t ggml_cuda_gdn_wmma_scratch_size(int64_t T) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (g2_active()) {   // one super-block's, whatever T
        return g2_scratch_bytes(g2_nimg(g2_nps(g2_np()), g2_npo(g2_np())));
    }
#endif
    return (size_t) ((T + GW_CK - 1) / GW_CK) * GW_HV * (2 * GW_CK * GW_CK) * sizeof(uint16_t);
}

bool ggml_cuda_gdn_wmma_usable(const ggml_tensor * dst) {
#if defined(GGML_USE_HIP) || !defined(GGML_USE_MUSA)
    // GGML_CUDA_GDN_WMMA_MIN=N: from N tokens (0: never)
    static const int64_t min_tokens = getenv("GGML_CUDA_GDN_WMMA_MIN") ? atoll(getenv("GGML_CUDA_GDN_WMMA_MIN")) : 128;
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * g = dst->src[3], * beta = dst->src[4];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool aligned16 = [&] {
        for (const ggml_tensor * t : { q, k, v }) {
            if ((uintptr_t) t->data % 16 != 0 || t->nb[0] != sizeof(float) || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0) {
                return false;
            }
        }
        return true;
    }();
#ifdef GGML_USE_HIP
    const bool arch_ok = GGML_CUDA_CC_IS_RDNA3(cc);
#else
    // CUDA: the tensor cores' FP16 wmma (sm_70 and later); GGML_CUDA_GDN_TC=0 keeps the recurrence
    static const bool tc_on = !getenv("GGML_CUDA_GDN_TC") || atoi(getenv("GGML_CUDA_GDN_TC")) != 0;
    const bool arch_ok = tc_on && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_VOLTA;
#endif
    return arch_ok && min_tokens > 0 && v->ne[2] >= min_tokens &&
           v->ne[0] == GW_S && q->ne[0] == GW_S && v->ne[1] == GW_HV && q->ne[1] == GW_HK && k->ne[1] == GW_HK &&
           v->ne[3] == 1 && q->ne[3] == 1 && g->ne[0] == 1 &&               // one sequence, a scalar gate (not KDA)
           ggml_get_op_params_i32(dst, 0) <= 1 &&                           // no snapshots
           // a deferred commit (ggml_gated_delta_net_set_replay) only when the batch is longer than a window: the
           // caller defers no such batch, its prefix is replayed on the state first (gated_delta_net.cu)
           (dst->src[6] == nullptr || v->ne[2] > dst->src[6]->ne[1] / 2) &&
           ggml_are_same_stride(q, k) && ggml_is_contiguous(g) && ggml_is_contiguous(beta) && aligned16;
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
}

void ggml_cuda_gdn_wmma_launch(const ggml_cuda_gdn_wmma_args & a, cudaStream_t stream) {
#ifdef GGML_USE_HIP
    const int64_t nch = (a.T + GW_CK - 1) / GW_CK;
    gdn_wmma_prep<<<dim3(GW_HK, (unsigned) nch), 32 * GW_VPK, 0, stream>>>(a, (_Float16 *) a.scratch);
    gdn_wmma_scan<<<GW_HV, GW_NW * 32, 0, stream>>>(a, (const unsigned *) a.scratch);
    CUDA_CHECK(cudaGetLastError());
#elif !defined(GGML_USE_MUSA)
    if (g2_active()) {
        switch (g2_np() * 10 + g2_tiles()) {
            case 11: g2_run<1, 1, 1>(a, stream); return;
            case 12: g2_run<1, 1, 2>(a, stream); return;
            case 21: g2_run<3, 1, 1>(a, stream); return;
            case 22: g2_run<3, 1, 2>(a, stream); return;
            case 31: g2_run<3, 3, 1>(a, stream); return;
            default: g2_run<3, 3, 2>(a, stream); return;
        }
    }
    const int64_t nch = (a.T + GW_CK - 1) / GW_CK;
    gdn_tc_prep_cuda<<<dim3(GW_HK, (unsigned) nch), GW_CK * GW_CK, 0, stream>>>(a, (__half *) a.scratch);
    gdn_tc_scan_cuda<<<GW_HV, GW_NW * 32, 0, stream>>>(a, (const __half *) a.scratch);
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(a, stream);
    GGML_ABORT("the chunked GDN runs on RDNA3 only");
#endif // GGML_USE_HIP
}
