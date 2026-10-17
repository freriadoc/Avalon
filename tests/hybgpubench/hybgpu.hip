// hybgpu : the hosted MoE cache kernels (mul_mat_vec_q_moe with entry pointers: one warp per block, R rows, one block
// row per routed row) extracted with ggml's own vec_dot, for the expert types of our model and of the IQ3_KS hybrid:
// up+gate fused (2560 -> 640) in IQ2_S, IQ3_K, IQ3_KS, IQ4_XS; down (640 -> 2560) in IQ4_NL and Q2_0. Random slots of a
// 2 GB pool (cold, a new set every launch), the same kernel on one slot (hot: compute bound) and a plain read of the
// same bytes; 27 routed rows (3 tokens). Variants: rows per warp, K known at compile time, loads without arithmetic,
// flat (row, chunk) dealing. Checks: Q2_0, IQ3_K, IQ3_KS, IQ2_S dot products against their previous versions
// (*-old.cuh) bit for bit on random blocks. Modes by environment: FLAT, UNROLL, ONLY_Q2, ONLY_KS, ONLY_I2, none.
// build: hybgpu.sh
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include <vector>
#include <random>
#include <cstdio>
template <ggml_type t> struct vd;
// IQ3_KS's loads without its arithmetic (the access pattern alone)
static __device__ __forceinline__ float vec_dot_iq3_ks_loads(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {
    const block_iq3_ks * bq3 = (const block_iq3_ks *) vbq + kbx;
    const int iqs = iiqs/4, ib128 = iqs/4, il8 = iqs%4;
    const uint2 head = *(const uint2 *) bq3;
    const int * ql = (const int *) bq3->qs + 8*ib128 + 2*il8;
    const int * qh = (const int *) bq3->qh + 2*il8;
    int acc = head.x ^ head.y ^ ql[0] ^ ql[1] ^ qh[0] ^ qh[1];
#pragma unroll
    for (int j = 0; j < 4; j++) { const int * q8 = (const int *)bq8_1[4*ib128+j].qs + 2*il8; acc += q8[0] ^ q8[1]; }
    return (float) acc;
}
struct loads_tag {};
// IQ2_S's loads without its arithmetic, without (G = false) or with (G = true) the grid gathers
template <bool G>
static __device__ __forceinline__ float vec_dot_iq2_s_loads(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs/2);
    const int qh = bq2->qh[iqs/2];
    const int signs = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    int acc = qs_packed ^ qh ^ signs ^ bq2->scales[iqs/2] ^ *(const uint16_t *) &bq2->d;
    if constexpr (G) {
        const uint8_t * qs = (const uint8_t *) &qs_packed;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));
            acc += grid_pos[0] ^ grid_pos[1];
        }
    }
#pragma unroll
    for (int l = 0; l < 8; l++) acc += get_int_b4(bq8_1[iqs/2].qs, l);
    return (float) acc;
}
template <> struct vd<GGML_TYPE_IQ2_S> { static constexpr int vdr = VDR_IQ2_S_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_iq2_s_q8_1(a, b, k, i); } };
template <> struct vd<GGML_TYPE_IQ3_K> { static constexpr int vdr = VDR_IQ3_K_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_iq3_k_q8_1(a, b, k, i); } };
template <> struct vd<GGML_TYPE_IQ3_KS> { static constexpr int vdr = VDR_IQ3_KS_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_iq3_ks_q8_1(a, b, k, i); } };
template <> struct vd<GGML_TYPE_IQ4_XS> { static constexpr int vdr = VDR_IQ4_XS_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_iq4_xs_q8_1(a, b, k, i); } };
template <> struct vd<GGML_TYPE_IQ4_NL> { static constexpr int vdr = VDR_IQ4_NL_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_iq4_nl_q8_1(a, b, k, i); } };
template <> struct vd<GGML_TYPE_Q2_0> { static constexpr int vdr = VDR_Q2_0_Q8_1_MMVQ;
    static __device__ __forceinline__ float f(const void * a, const block_q8_1 * b, const int & k, const int & i) { return vec_dot_q2_0_q8_1(a, b, k, i); } };

// the Q2_0 dot product before the shift/mask decode (to check the new one bit for bit)
static __device__ __forceinline__ float vec_dot_q2_0_q8_1_old(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_q2_0 * bq2_0 = (const block_q2_0 *) vbq + kbx;
    const float     d2 = bq2_0->d;
    const int16_t * qs = (const int16_t *) bq2_0->qs + iqs * 4;
    const block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = qs[j];
        const int u  = get_int_b4(bq8_1_chunk->qs, j*2+0);
        const int v  = get_int_b4(bq8_1_chunk->qs, j*2+1);
        const uint32_t qx_indices = (q & 0x03) | ((q & 0x0C) << 6) | ((q & 0x30) << 12) | ((q & 0xC0) << 18);
        const uint32_t qy_bits    = q >> 8;
        const uint32_t qy_indices = (qy_bits & 0x03) | ((qy_bits & 0x0C) << 6) | ((qy_bits & 0x30) << 12) | ((qy_bits & 0xC0) << 18);
        const int qx = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qx_indices);
        const int qy = __builtin_amdgcn_perm(0x020100FF, 0x020100FF, qy_indices);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}
#include "iq3ks-old.cuh"
#include "iq3k-old.cuh"
#include "iq2s-old.cuh"
static __device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16; return x;
}
// N random blocks built in registers (every byte random, d = 1.0 so the result stays finite), each dot product of the
// new and the old version against random activations: count the differences (bits)
template <typename block, int NI, int VDR, float (*f_new)(const void *, const block_q8_1 *, const int &, const int &),
          float (*f_old)(const void *, const block_q8_1 *, const int &, const int &)>
static __global__ void k_check(const block_q8_1 * y, int nblocks, unsigned long long * bad, unsigned long long * n) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nblocks; b += gridDim.x * blockDim.x) {
        union alignas(8) { block blk; uint32_t w[(sizeof(block) + 3) / 4]; } u;
        for (int i = 0; i < (int) ((sizeof(block) + 3) / 4); i++) u.w[i] = hash32(b * 64u + i + 1);
        *(uint16_t *) &u.blk = 0x3c00;
        const block_q8_1 * yy = y + (b % 10) * 8;
        for (int iqs = 0; iqs < NI; iqs += VDR) {
            const float x = f_new(&u.blk, yy, 0, iqs);
            const float z = f_old(&u.blk, yy, 0, iqs);
            if (__float_as_uint(x) != __float_as_uint(z)) atomicAdd(bad, 1ull);
            atomicAdd(n, 1ull);
        }
    }
}

template <ggml_type type, int R, bool gate, int NC = 0, int LOADS = 0>
__launch_bounds__(32, 1)
static __global__ void k_mmv(const char * pool, const block_q8_1 * y, const int * ids, const int * gids, const int * act_ids,
                             float * dst, int ncols, int nrows, int stride_row, int stride_channel, int stride_y) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = vd<type>::vdr;
    const int route = blockIdx.y;
    const int row0 = R * blockIdx.x;
    const int blocks_per_row = (NC ? NC : ncols) / qk;
    constexpr int blocks_per_iter = vdr * 32 / qi;
    const block_q8_1 * yy = y + act_ids[route] * stride_y;
    const int off  = ids[route]  * stride_channel + row0 * stride_row;
    const int goff = gids[route] * stride_channel + row0 * stride_row;
    float tmp[R] = {0}, tg[R] = {0};
#pragma unroll
    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
#pragma unroll
        for (int i = 0; i < R; ++i) {
            if constexpr (LOADS == 2 || LOADS == 3) {
                tmp[i] += vec_dot_iq2_s_loads<LOADS == 3>(pool, &yy[kby], off + i*stride_row + kbx, kqs);
                if (gate) tg[i] += vec_dot_iq2_s_loads<LOADS == 3>(pool, &yy[kby], goff + i*stride_row + kbx, kqs);
            } else if constexpr (LOADS == 1) {
                tmp[i] += vec_dot_iq3_ks_loads(pool, &yy[kby], off + i*stride_row + kbx, kqs);
                if (gate) tg[i] += vec_dot_iq3_ks_loads(pool, &yy[kby], goff + i*stride_row + kbx, kqs);
            } else {
                tmp[i] += vd<type>::f(pool, &yy[kby], off + i*stride_row + kbx, kqs);
                if (gate) tg[i] += vd<type>::f(pool, &yy[kby], goff + i*stride_row + kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) { tmp[i] = warp_reduce_sum<32>(tmp[i]); if (gate) tg[i] = warp_reduce_sum<32>(tg[i]); }
    if (threadIdx.x < R) {
        const float g = tg[threadIdx.x];
        dst[route * nrows + row0 + threadIdx.x] = gate ? tmp[threadIdx.x] * g / (1.0f + expf(-g)) : tmp[threadIdx.x];
    }
}

// flat: the (row, chunk) pairs of the R rows numbered one after the other and dealt to the 32 lanes, so that no lane
// idles at the end of a row (80 chunks a row: 2.5 passes of the warp per row, 5 full passes for 2 rows)
template <ggml_type type, int R, bool gate>
__launch_bounds__(32, 1)
static __global__ void k_mmv_flat(const char * pool, const block_q8_1 * y, const int * ids, const int * gids, const int * act_ids,
                                  float * dst, int ncols, int nrows, int stride_row, int stride_channel, int stride_y) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = vd<type>::vdr;
    constexpr int lanes_per_block = qi/vdr;
    const int route = blockIdx.y;
    const int row0 = R * blockIdx.x;
    const int per_row = (ncols / qk) * lanes_per_block;
    const block_q8_1 * yy = y + act_ids[route] * stride_y;
    const int off  = ids[route]  * stride_channel + row0 * stride_row;
    const int goff = gids[route] * stride_channel + row0 * stride_row;
    float tmp[R] = {0}, tg[R] = {0};
    int r = 0, c = threadIdx.x;
    while (c >= per_row) { c -= per_row; r++; }
    for (; r < R; ) {
        const int kbx = c / lanes_per_block;
        const int kqs = vdr * (c % lanes_per_block);
        const int kby = kbx * (qk/QK8_1);
        const float v = vd<type>::f(pool, &yy[kby], off + r*stride_row + kbx, kqs);
        const float w = gate ? vd<type>::f(pool, &yy[kby], goff + r*stride_row + kbx, kqs) : 0.0f;
#pragma unroll
        for (int i = 0; i < R; ++i) { if (i == r) { tmp[i] += v; if (gate) tg[i] += w; } }
        c += 32;
        while (c >= per_row) { c -= per_row; r++; }
    }
#pragma unroll
    for (int i = 0; i < R; ++i) { tmp[i] = warp_reduce_sum<32>(tmp[i]); if (gate) tg[i] = warp_reduce_sum<32>(tg[i]); }
    if (threadIdx.x < R) {
        const float g = tg[threadIdx.x];
        dst[route * nrows + row0 + threadIdx.x] = gate ? tmp[threadIdx.x] * g / (1.0f + expf(-g)) : tmp[threadIdx.x];
    }
}

static __global__ void k_read(const int4 * pool, const int * ids, const int * gids, size_t slot_int4, int nmat, int * out) {
    const int route = blockIdx.y;
    int acc = 0;
    for (int s = 0; s < nmat; s++) {
        const int4 * p = pool + (size_t) (s ? gids[route] : ids[route]) * slot_int4;
        for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < slot_int4; i += (size_t) gridDim.x * blockDim.x) {
            const int4 v = p[i];
            acc += v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <ggml_type type, bool gate, int R = 2, int NC = 0, int LOADS = 0, bool FLAT = false>
static void run(const char * name, char * pool, size_t pool_bytes, const block_q8_1 * y, float * dst, int * d_sets, int n_sets, int rows, int iters) {
    const int ncols = gate ? 2560 : 640, nrows = gate ? 640 : 2560;
    const size_t row_bytes = ggml_row_size(type, ncols);
    const size_t slot_bytes = row_bytes * nrows;
    const int ts = ggml_type_size(type);
    // slots aligned like cache entries (256 bytes), counted in blocks
    const size_t slot_al = (slot_bytes + 255) & ~(size_t) 255;
    if (slot_al % ts) { printf("%s: slot not a multiple of the block\n", name); return; }
    const int stride_row = row_bytes / ts, stride_channel = slot_al / ts;
    const int n_slots = (int) (pool_bytes / slot_al);
    const int stride_y = 2560 / QK8_1;
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    float ms[3];
    for (int mode = 0; mode < 3; mode++) {
        for (int pass = 0; pass < 2; pass++) {
            hipEventRecord(e0);
            for (int it = 0; it < iters; it++) {
                const int set = mode == 1 ? n_sets : (it % n_sets);
                const int * ids = d_sets + (size_t) set * 3 * rows;
                if (mode < 2 && FLAT) {
                    k_mmv_flat<type, R, gate><<<dim3(nrows / R, rows), 32>>>(pool, y, ids, ids + rows, ids + 2 * rows, dst, ncols, nrows,
                                                                       stride_row, stride_channel, stride_y);
                } else if (mode < 2) {
                    k_mmv<type, R, gate, NC, LOADS><<<dim3(nrows / R, rows), 32>>>(pool, y, ids, ids + rows, ids + 2 * rows, dst, ncols, nrows,
                                                                       stride_row, stride_channel, stride_y);
                } else {
                    k_read<<<dim3(16, rows), 256>>>((const int4 *) pool, ids, ids + rows, slot_al / 16, gate ? 2 : 1, (int *) dst);
                }
            }
            hipEventRecord(e1); hipEventSynchronize(e1);
            if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) { printf("%s: kernel error\n", name); exit(1); }
            hipEventElapsedTime(&ms[mode], e0, e1);
        }
    }
    const double bytes = (gate ? 2.0 : 1.0) * rows * slot_bytes;
    printf("%-14s R%d%s slot %5.0f KB, %5d slots: cold %6.1f us (%4.0f GB/s), hot one slot %6.1f us, plain read %6.1f us (%4.0f GB/s)\n",
           name, R, FLAT ? " flat" : "     ", slot_bytes / 1024.0, n_slots, 1e3 * ms[0] / iters, bytes / (1e-3 * ms[0] / iters) / 1e9, 1e3 * ms[1] / iters,
           1e3 * ms[2] / iters, bytes / (1e-3 * ms[2] / iters) / 1e9);
}

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 2000, rows = 27, n_sets = 997;
    const size_t pool_bytes = (size_t) 2 << 30;
    char * pool; hipMalloc(&pool, pool_bytes);
    {   // small finite halves everywhere (any 2-byte word read as a scale is finite)
        std::vector<uint16_t> h(1 << 24);
        std::mt19937 g(1);
        for (auto & v : h) v = (uint16_t) ((g() & 0x03ff) | 0x2000);
        for (size_t o = 0; o < pool_bytes; o += h.size() * 2) hipMemcpy(pool + o, h.data(), std::min(pool_bytes - o, h.size() * 2), hipMemcpyHostToDevice);
    }
    std::vector<block_q8_1> hy(3 * 2560 / QK8_1);
    std::mt19937 g(2);
    for (auto & b : hy) { for (int i = 0; i < QK8_1; i++) b.qs[i] = (int8_t) (g() % 255 - 127); b.ds = make_half2(0.01f, 0.0f); }
    block_q8_1 * y; hipMalloc(&y, hy.size() * sizeof(block_q8_1)); hipMemcpy(y, hy.data(), hy.size() * sizeof(block_q8_1), hipMemcpyHostToDevice);
    float * dst; hipMalloc(&dst, 64 * 2560 * sizeof(float));
    // slot numbers up to the smallest slot count of all types (the largest slot: IQ4_XS up 2560x640)
    const int max_slots = (int) (pool_bytes / ((ggml_row_size(GGML_TYPE_IQ4_NL, 640) * 2560 + 255) & ~(size_t) 255));
    std::vector<int> hs((size_t) (n_sets + 1) * 3 * rows);
    for (int s = 0; s <= n_sets; s++) for (int r = 0; r < rows; r++) {
        int * p = hs.data() + (size_t) s * 3 * rows;
        p[r] = s == n_sets ? 0 : (int) (g() % max_slots);
        p[rows + r] = s == n_sets ? 1 : (int) (g() % max_slots);
        p[2 * rows + r] = r / 9;
    }
    int * d_sets; hipMalloc(&d_sets, hs.size() * sizeof(int)); hipMemcpy(d_sets, hs.data(), hs.size() * sizeof(int), hipMemcpyHostToDevice);
#define RUNS(T, G, N) run<T, G, 1>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters); \
        run<T, G, 2>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters); \
        run<T, G, 4>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters); \
        run<T, G, 8>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters)
    {
        unsigned long long * d; hipMalloc(&d, 16); hipMemset(d, 0, 16);
        k_check<block_q2_0, 2, 1, vec_dot_q2_0_q8_1, vec_dot_q2_0_q8_1_old><<<1024, 256>>>(y, 64 << 20, d, d + 1);
        unsigned long long h[2]; hipMemcpy(h, d, 16, hipMemcpyDeviceToHost);
        printf("Q2_0 check: %llu dot products, %llu differ\n", h[1], h[0]);
        hipMemset(d, 0, 16);
        k_check<block_iq3_ks, QI4_XS, VDR_IQ3_KS_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1, vec_dot_iq3_ks_q8_1_old><<<1024, 256>>>(y, 16 << 20, d, d + 1);
        hipMemcpy(h, d, 16, hipMemcpyDeviceToHost);
        printf("IQ3_KS check: %llu dot products, %llu differ\n", h[1], h[0]);
        hipMemset(d, 0, 16);
        k_check<block_iq3_k, QI4_XS, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_k_q8_1, vec_dot_iq3_k_q8_1_old><<<1024, 256>>>(y, 16 << 20, d, d + 1);
        hipMemcpy(h, d, 16, hipMemcpyDeviceToHost);
        printf("IQ3_K check: %llu dot products, %llu differ\n", h[1], h[0]);
        hipMemset(d, 0, 16);
        k_check<block_iq2_s, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1, vec_dot_iq2_s_q8_1_old><<<1024, 256>>>(y, 16 << 20, d, d + 1);
        hipMemcpy(h, d, 16, hipMemcpyDeviceToHost);
        printf("IQ2_S check: %llu dot products, %llu differ\n", h[1], h[0]);
    }
    if (getenv("FLAT")) {
#define FL(T, G, Rr, N) run<T, G, Rr>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters); \
        run<T, G, Rr, 0, 0, true>(N, pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters)
        for (int round = 0; round < 2; round++) {
            FL(GGML_TYPE_IQ2_S,  true, 2, "IQ2_S up+gate");
            FL(GGML_TYPE_IQ2_S,  true, 4, "IQ2_S up+gate");
            FL(GGML_TYPE_IQ3_K,  true, 2, "IQ3_K up+gate");
            FL(GGML_TYPE_IQ3_KS, true, 2, "IQ3_KS up+gate");
            FL(GGML_TYPE_IQ4_NL, false, 4, "IQ4_NL down");
            FL(GGML_TYPE_IQ4_NL, false, 8, "IQ4_NL down");
            FL(GGML_TYPE_Q2_0,   false, 4, "Q2_0 down");
            FL(GGML_TYPE_Q2_0,   false, 8, "Q2_0 down");
        }
        return 0;
    }
    if (getenv("ONLY_I2")) {
        for (int round = 0; round < 3; round++) {
            run<GGML_TYPE_IQ2_S, true, 2>("IQ2_S up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ2_S, true, 2, 0, 2>("IQ2_S loads", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ2_S, true, 2, 0, 3>("IQ2_S lds+grid", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
        }
        return 0;
    }
    if (getenv("ONLY_KS")) {
        for (int round = 0; round < 3; round++) {
            run<GGML_TYPE_IQ3_K,  true, 2>("IQ3_K up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_KS, true, 2>("IQ3_KS up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_KS, true, 2, 0, 1>("IQ3_KS loads", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_KS, true, 4, 0, 1>("IQ3_KS loads", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
        }
        return 0;
    }
    if (getenv("UNROLL")) {
        for (int round = 0; round < 2; round++) {
            run<GGML_TYPE_IQ2_S,  true, 2>("IQ2_S up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ2_S,  true, 2, 2560>("IQ2_S up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_K,  true, 2>("IQ3_K up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_K,  true, 2, 2560>("IQ3_K up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_KS, true, 2>("IQ3_KS up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ3_KS, true, 2, 2560>("IQ3_KS up+gate", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ4_NL, false, 4>("IQ4_NL down", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
            run<GGML_TYPE_IQ4_NL, false, 4, 640>("IQ4_NL down", pool, pool_bytes, y, dst, d_sets, n_sets, rows, iters);
        }
        return 0;
    }
    if (getenv("ONLY_Q2")) {
        for (int round = 0; round < 2; round++) { RUNS(GGML_TYPE_Q2_0, false, "Q2_0 down"); }
        return 0;
    }
    for (int round = 0; round < 2; round++) {
        RUNS(GGML_TYPE_IQ2_S,  true,  "IQ2_S up+gate");
        RUNS(GGML_TYPE_IQ3_K,  true,  "IQ3_K up+gate");
        RUNS(GGML_TYPE_IQ3_KS, true,  "IQ3_KS up+gate");
        RUNS(GGML_TYPE_IQ4_NL, false, "IQ4_NL down");
        RUNS(GGML_TYPE_Q2_0,   false, "Q2_0 down");
    }
    return 0;
}
