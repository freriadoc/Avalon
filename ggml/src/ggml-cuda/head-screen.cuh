#pragma once

// GGML_CUDA_DRAFT_HEAD_SCREEN (mmvq.cu): the cut LM head of an MTP draft (Q8_0, one column, the rows of an FR-Spec
// vocabulary mapped back by d2t) screened through a 2-bit copy of its rows. Every row is scored approximately from the
// copy (800 bytes a row of 2560 against 2720 in Q8_0: 52 MiB read instead of 178 MiB for 65 536 rows), the rows within
// HS_DELTA logits of the best approximate score (at most HS_MAX_CAND) are rescored with mul_mat_vec_q's own Q8_0
// arithmetic, in its order, and the head's output holds those exact values, -inf for every other row: the graph's
// fill / set_rows / top_k that follow are unchanged, and their argmax is mmvq's whenever the exact best is among the
// rows rescored.
//
// Ported from Basalt (MIT license; include/basalt/kernels/head_proxy.cuh and src/kernels/cuda/q80_proxy.cu, commits
// 643cb978 and 29e71767): the copy's layout, the fit of a 32-weight group (six trial scales, then UNSHRINK), the scoring
// of a tile of 32 rows by a block of 4 warps (lane = row), the HS_PER best of each tile, the selection of the rows
// rescored (HS_DELTA = 5 logits, HS_MAX_CAND = 64, a radix select of the best 64 past that). Basalt measured on its
// Q8_0 draft head (41 037 rows): the pick equal to the exact argmax in 3565 of 3566 real draft steps, 0 of 400 random
// ones.

#include "common.cuh"
#include "vecdotq.cuh"

#include <cstdint>

static constexpr int   HS_PER           = 4;      // candidates kept per tile of 32 rows
static constexpr int   HS_MAX_CAND      = 64;     // rows rescored exactly
static constexpr float HS_DELTA         = 5.0f;   // a row is rescored when its approximate score is within this of the best
static constexpr int   HS_SCORE_THREADS = 128;    // a tile of 32 rows: 4 warps, each a quarter of the row length
static constexpr int   HS_FINAL_THREADS = 1024;   // the selection and the exact rows: 32 warps, up to 2 rows each
static constexpr int   HS_FINAL_WARPS   = HS_FINAL_THREADS / 32;

static __host__ __device__ constexpr int64_t hs_tiles(int64_t n_rows) { return (n_rows + 31) / 32; }
static __host__ __device__ constexpr int     hs_nw4(int n_in)         { return n_in / 64; }   // uint4 of levels per row

// The copy: per tile of 32 rows and uint4 w4 of a row (two 32-weight groups 2 w4, 2 w4 + 1), the uint4 of levels of
// each of the 32 rows (lane = row, consecutive lanes at consecutive 16 bytes); then, at the same index, the two groups'
// FP16 scales as one 32-bit word. Then the candidates of the scoring pass (HS_PER float2 a tile: score, row).
static __host__ __device__ constexpr size_t hs_levels_bytes(int64_t n_rows, int n_in) {
    return (size_t) hs_tiles(n_rows) * hs_nw4(n_in) * 32 * 16;
}
static __host__ __device__ constexpr size_t hs_scales_bytes(int64_t n_rows, int n_in) {
    return (size_t) hs_tiles(n_rows) * hs_nw4(n_in) * 32 * 4;
}
static __host__ __device__ constexpr size_t hs_cand_bytes(int64_t n_rows) {
    return (size_t) hs_tiles(n_rows) * HS_PER * sizeof(float2);
}

// The level (0..3) of weight w at quantization step sh (a group of zeros has a zero step: any level).
static __device__ __forceinline__ int hs_level(float w, float sh) {
    if (!(sh > 0.0f)) {
        return 2;
    }
    return min(3, max(0, (int) floorf(w / sh + 2.0f)));
}

// One 32-weight group's scale: four levels (l - 1.5) sh with the scale, of six candidates, that leaves the least squared
// error; sh_out gets the step the levels are taken at. UNSHRINK (Basalt): the stored scale is then multiplied by
// sqrt(|w|^2 / <w, w_hat>), the least-squares copy scoring the rows near the best low.
static __device__ __forceinline__ half hs_fit_group(const float * w, float & sh_out) {
    float amax = 1e-9f;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        amax = fmaxf(amax, fabsf(w[l]));
    }
    float best_s = amax / 1.5f;
    float best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s  = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) hs_level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }
    half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    sh_out = sh;
    float ww = 0.0f;
    float wr = 0.0f;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        const float r = ((float) hs_level(w[l], sh) - 1.5f) * sh;
        ww += w[l] * w[l];
        wr += w[l] * r;
    }
    if (wr > 0.0f) {
        hs = __float2half(sh * sqrtf(ww / wr));
    }
    return hs;
}

// One thread per 32-weight group (one Q8_0 block): the group's values fitted into the copy. Word c (0, 1) of a group
// holds its values 16 c .. 16 c + 15: value 16 c + 4 i + b at bits 8 b + 2 i, so that (word >> 2 i) & 0x03030303 is
// the 4 levels that meet the activation's int i of the half (one dp4a).
template <int N_IN>
static __global__ void hs_pack_q8_0(const block_q8_0 * __restrict__ x, const int64_t n_rows, const int64_t stride_row,
        uint4 * __restrict__ wq, uint16_t * __restrict__ sc) {
    constexpr int BPR = N_IN / QK8_0;
    constexpr int NW4 = hs_nw4(N_IN);
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * BPR) {
        return;
    }
    const int64_t row = i / BPR;
    const int     g   = (int) (i % BPR);
    const block_q8_0 * b = x + row * stride_row + g;
    const float d = __half2float(b->d);
    float w[32];
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        w[l] = d * (float) b->qs[l];
    }
    float sh;
    const half hs = hs_fit_group(w, sh);
    const int64_t tile = row >> 5;
    const int     lane = (int) (row & 31);
    const int     w4   = g / 2;
    const int     h    = g & 1;
    uint32_t words[2] = { 0u, 0u };
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int ii = 0; ii < 4; ++ii) {
#pragma unroll
            for (int bb = 0; bb < 4; ++bb) {
                words[c] |= (uint32_t) hs_level(w[c * 16 + 4 * ii + bb], sh) << (8 * bb + 2 * ii);
            }
        }
    }
    const size_t idx = ((size_t) tile * NW4 + w4) * 32 + lane;
    uint32_t * wd = reinterpret_cast<uint32_t *>(wq) + idx * 4 + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[idx * 2 + h] = __half_as_ushort(hs);
}

// The approximate pass, a tile of 32 rows a block of 4 warps: warp k sums the uint4 k NW .. (k + 1) NW - 1 of the rows
// (lane = row), against the activation's q8_1 (as mmvq reads it) staged in shared memory. Writes the tile's HS_PER best
// approximate scores (score, row) and -inf to the tile's rows of the output (the final kernel writes the rescored ones).
template <int N_IN>
static __global__ void __launch_bounds__(HS_SCORE_THREADS) hs_score(
        const uint4 * __restrict__ wq, const uint32_t * __restrict__ sc, const block_q8_1 * __restrict__ x,
        const int n_rows, float2 * __restrict__ cand, float * __restrict__ out) {
    constexpr int NW4 = hs_nw4(N_IN);
    constexpr int NW  = NW4 / 4;
    constexpr int NB  = N_IN / 32;
    static_assert(N_IN % 256 == 0 && NB <= HS_SCORE_THREADS, "rows of whole uint4 quarters, one q8_1 block a thread");

    __shared__ __align__(16) int xi[N_IN / 4];
    __shared__ float xs[NB];
    __shared__ float xdd[NB];
    __shared__ float red[4][32];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tile = blockIdx.x;

    // the copy first, the activation after (its loads are short, from L2)
    const size_t base = ((size_t) tile * NW4 + warp * NW) * 32 + lane;
    uint4    w[NW];
    uint32_t s[NW];
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        w[j] = wq[base + (size_t) j * 32];
        s[j] = sc[base + (size_t) j * 32];
    }

    for (int i = tid; i < N_IN / 4; i += HS_SCORE_THREADS) {
        xi[i] = reinterpret_cast<const int *>(x[i / 8].qs)[i % 8];
    }
    if (tid < NB) {
        const int * q = reinterpret_cast<const int *>(x[tid].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum = ggml_cuda_dp4a(0x01010101, q[i], sum);
        }
        const float d8 = __low2float(x[tid].ds);
        xdd[tid] = d8;
        xs[tid]  = d8 * (float) sum;
    }
    __syncthreads();

    // sum over the group of (l - 1.5) sh d8 q = sh (d8 sum(l q) - 1.5 d8 sum(q))
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < NW; ++j) {
        const int w4 = warp * NW + j;
        const uint32_t ww[4] = { w[j].x, w[j].y, w[j].z, w[j].w };
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * w4 + h;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v  = ww[2 * h + c];
                const int4     x0 = *reinterpret_cast<const int4 *>(&xi[g * 8 + c * 4]);
                P = ggml_cuda_dp4a((int) ( v       & 0x03030303u), x0.x, P);
                P = ggml_cuda_dp4a((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = ggml_cuda_dp4a((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = ggml_cuda_dp4a((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (warp != 0) {
        return;
    }

    const float ninf = -INFINITY;
    const int row = tile * 32 + lane;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= n_rows) {
        v = ninf;
    } else {
        out[row] = ninf;
    }
#pragma unroll
    for (int k = 0; k < HS_PER; ++k) {
        float bv = v;
        int   bi = lane;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, bv, o, 32);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, o, 32);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            cand[tile * HS_PER + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        }
        if (lane == bi) {
            v = ninf;
        }
    }
}

struct hs_final_shared {
    float wmax[HS_FINAL_WARPS];
    int   list[HS_MAX_CAND];
    int   cnt;
    int   hist[256];
    unsigned long long want;
    int   need;
};

// A candidate's selection key: its approximate score, then the lower row on a tie. Keys of distinct rows differ.
static __device__ __forceinline__ unsigned long long hs_cand_key(const float2 c) {
    const unsigned u = __float_as_uint(c.x);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) o << 32) | (unsigned) (0xffffffffu - (unsigned) __float_as_int(c.y));
}

static __device__ __forceinline__ bool hs_in_margin(const float2 c, const float m) {
    return c.x > -INFINITY && c.x >= m - HS_DELTA;
}

// The HS_MAX_CAND-th largest key among the nc candidates within HS_DELTA of m (more than HS_MAX_CAND of them): a radix
// select over the key's bytes, most significant first (Basalt's kth_key).
static __device__ unsigned long long hs_kth_key(hs_final_shared & f, const int nc, const float2 * __restrict__ cand,
        const float m) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (tid == 0) {
        f.want = 0ull;
        f.need = HS_MAX_CAND;
    }
    unsigned long long mask = 0ull;
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) {
            f.hist[tid] = 0;
        }
        __syncthreads();
        const unsigned long long want = f.want;
        for (int i = tid; i < nc; i += HS_FINAL_THREADS) {
            const float2 c = cand[i];
            if (!hs_in_margin(c, m)) {
                continue;
            }
            const unsigned long long k = hs_cand_key(c);
            if ((k & mask) == want) {
                atomicAdd(&f.hist[(int) ((k >> shift) & 255u)], 1);
            }
        }
        __syncthreads();
        if (warp == 0) {
            // lane l holds bins 255 - 8 l .. 248 - 8 l; the digit is the bin where the count from the top reaches need
            int b[8];
            int s = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                b[j] = f.hist[255 - 8 * lane - j];
                s += b[j];
            }
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int t = __shfl_up_sync(0xffffffff, incl, o, 32);
                if (lane >= o) {
                    incl += t;
                }
            }
            const int need = f.need;
            const int excl = incl - s;
            if (excl < need && incl >= need) {
                int a = excl;
                int j = 0;
                while (a + b[j] < need) {
                    a += b[j++];
                }
                f.want = want | ((unsigned long long) (255 - 8 * lane - j) << shift);
                f.need = need - a;
            }
        }
        mask |= 255ull << shift;
        __syncthreads();
    }
    return f.want;
}

// mul_mat_vec_q<GGML_TYPE_Q8_0, 1, ...>'s value of one row, by one warp: mmvq runs the row on NW warps (calc_nwarps),
// thread tid taking the blocks kbx = tid / 4, += 8 NW and the ints kqs = 2 (tid % 4), kqs + 1 of each; lane L stands for
// its threads L + 32 k, whose sums are added in mmvq's cross-warp order (warp 0's, then warp 1's, ...) before the same
// xor tree. The same vec_dot_q8_0_q8_1 (and so the same products, accumulations and contractions): the same bits.
template <int N_IN, int NW>
static __device__ __forceinline__ float hs_exact_row(const void * __restrict__ vx, const int row, const int stride_row_x,
        const block_q8_1 * __restrict__ y, const int lane) {
    constexpr int qk              = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi              = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr             = VDR_Q8_0_Q8_1_MMVQ;
    constexpr int blocks_per_row  = N_IN / qk;
    constexpr int blocks_per_iter = vdr * NW * 32 / qi;
    const int kbx_offset = row * stride_row_x;

    // mmvq's loop kbx = tid / 4; kbx < blocks_per_row; kbx += blocks_per_iter, as a fixed trip count (unrolled, the
    // loads issued together) with the steps past the row skipped: the same sums in the same order
    constexpr int iters = (blocks_per_row + blocks_per_iter - 1) / blocks_per_iter;
    float tmp[NW];
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        const int tid  = 32 * k + lane;
        const int kqs  = vdr * (tid % (qi / vdr));
        const int kbx0 = tid / (qi / vdr);
        tmp[k] = 0.0f;
#pragma unroll
        for (int it = 0; it < iters; ++it) {
            const int kbx = kbx0 + it * blocks_per_iter;
            if (kbx < blocks_per_row) {
                const int kby = kbx * (qk / QK8_1);
                tmp[k] += vec_dot_q8_0_q8_1(vx, &y[kby], kbx_offset + kbx, kqs);
            }
        }
    }
    float sum = tmp[0];
#pragma unroll
    for (int k = 1; k < NW; ++k) {
        sum += tmp[k];
    }
    return warp_reduce_sum<32>(sum);
}

// One block: the best approximate score, the rows within HS_DELTA of it (the HS_MAX_CAND best when more qualify, the
// lower row on a tie: the set depends on the scores alone), then their exact values, a warp a row, into the output.
template <int N_IN, int NW>
static __global__ void __launch_bounds__(HS_FINAL_THREADS) hs_final(
        const void * __restrict__ vx, const int stride_row_x, const block_q8_1 * __restrict__ y,
        const float2 * __restrict__ cand, const int nc, float * __restrict__ out) {
    __shared__ hs_final_shared f;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float ninf = -INFINITY;

    if (tid == 0) {
        f.cnt = 0;
    }
    float m = ninf;
    for (int i = tid; i < nc; i += HS_FINAL_THREADS) {
        m = fmaxf(m, cand[i].x);
    }
    m = warp_reduce_max<32>(m);
    if (lane == 0) {
        f.wmax[warp] = m;
    }
    __syncthreads();
    m = f.wmax[0];
#pragma unroll
    for (int i = 1; i < HS_FINAL_WARPS; ++i) {
        m = fmaxf(m, f.wmax[i]);
    }
    if (m == ninf) {
        return;   // no finite score: every row stays -inf
    }
    for (int i = tid; i < nc; i += HS_FINAL_THREADS) {
        const float2 c = cand[i];
        if (hs_in_margin(c, m)) {
            const int p = atomicAdd(&f.cnt, 1);
            if (p < HS_MAX_CAND) {
                f.list[p] = __float_as_int(c.y);
            }
        }
    }
    __syncthreads();
    const int total = f.cnt;
    if (total > HS_MAX_CAND) {
        const unsigned long long kth = hs_kth_key(f, nc, cand, m);
        if (tid == 0) {
            f.cnt = 0;
        }
        __syncthreads();
        for (int i = tid; i < nc; i += HS_FINAL_THREADS) {
            const float2 c = cand[i];
            if (hs_in_margin(c, m) && hs_cand_key(c) >= kth) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < HS_MAX_CAND) {
                    f.list[p] = __float_as_int(c.y);
                }
            }
        }
        __syncthreads();
    }
    const int n = total < HS_MAX_CAND ? total : HS_MAX_CAND;

    for (int c = warp; c < n; c += HS_FINAL_WARPS) {
        const int   row = f.list[c];
        const float v   = hs_exact_row<N_IN, NW>(vx, row, stride_row_x, y, lane);
        if (lane == 0) {
            out[row] = v;
        }
    }
}

// GGML_CUDA_DRAFT_HEAD_SCREEN=2: mmvq's output (exact) against the screen's, one block. stats: calls, argmax
// mismatches, rescored values whose bits differ from mmvq's, rows rescored. Prints a line per argmax mismatch and a
// summary every 256 calls (a diagnostic: the stats are the head copy's own, on the one stream of the draft).
static __global__ void __launch_bounds__(1024) hs_check(const float * __restrict__ exact, const float * __restrict__ screen,
        const int n, unsigned long long * __restrict__ stats) {
    __shared__ float sv_e[32], sv_s[32];
    __shared__ int   si_e[32], si_s[32], sn_bad[32], sn_cand[32];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float ninf = -INFINITY;

    float ve = ninf, vs = ninf;
    int   ie = 0x7fffffff, is = 0x7fffffff, n_bad = 0, n_cand = 0;
    for (int i = tid; i < n; i += 1024) {
        const float e = exact[i];
        const float s = screen[i];
        if (e > ve || (e == ve && i < ie)) {
            ve = e;
            ie = i;
        }
        if (s > ninf) {
            n_cand++;
            n_bad += __float_as_uint(s) != __float_as_uint(e);
            if (s > vs || (s == vs && i < is)) {
                vs = s;
                is = i;
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ove = __shfl_xor_sync(0xffffffff, ve, o, 32);
        const int   oie = __shfl_xor_sync(0xffffffff, ie, o, 32);
        const float ovs = __shfl_xor_sync(0xffffffff, vs, o, 32);
        const int   ois = __shfl_xor_sync(0xffffffff, is, o, 32);
        if (ove > ve || (ove == ve && oie < ie)) {
            ve = ove;
            ie = oie;
        }
        if (ovs > vs || (ovs == vs && ois < is)) {
            vs = ovs;
            is = ois;
        }
        n_bad  += __shfl_xor_sync(0xffffffff, n_bad,  o, 32);
        n_cand += __shfl_xor_sync(0xffffffff, n_cand, o, 32);
    }
    if (lane == 0) {
        sv_e[warp] = ve; si_e[warp] = ie; sv_s[warp] = vs; si_s[warp] = is; sn_bad[warp] = n_bad; sn_cand[warp] = n_cand;
    }
    __syncthreads();
    if (tid != 0) {
        return;
    }
    ve = sv_e[0]; ie = si_e[0]; vs = sv_s[0]; is = si_s[0]; n_bad = sn_bad[0]; n_cand = sn_cand[0];
    for (int k = 1; k < 32; ++k) {
        if (sv_e[k] > ve || (sv_e[k] == ve && si_e[k] < ie)) {
            ve = sv_e[k];
            ie = si_e[k];
        }
        if (sv_s[k] > vs || (sv_s[k] == vs && si_s[k] < is)) {
            vs = sv_s[k];
            is = si_s[k];
        }
        n_bad  += sn_bad[k];
        n_cand += sn_cand[k];
    }
    const unsigned long long calls = ++stats[0];
    stats[1] += ie != is;
    stats[2] += n_bad;
    stats[3] += n_cand;
    if (ie != is) {
        printf("[head-screen] check %llu: argmax %d (%.4f), screen %d (exact %.4f), %d rows rescored\n",
                calls, ie, ve, is < n ? is : -1, is < n ? exact[is] : 0.0f, n_cand);
    }
    if (calls % 256 == 0) {
        printf("[head-screen] check: %llu calls, %llu argmax mismatches, %llu rescored values not bitwise mmvq's, "
                "%.1f rows rescored a call\n", calls, stats[1], stats[2], (double) stats[3] / (double) calls);
    }
}
