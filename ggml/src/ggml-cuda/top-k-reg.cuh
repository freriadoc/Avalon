#pragma once

// The top-k of a row with its keys read once and held in registers (HIP, RDNA: a 1,024-thread block has up to 192 VGPRs
// a lane, room for 64 keys a thread, rows of up to 65,536 columns: the QSA blocks of a 262,144-cell context).
// The selection is top_k_radix_row's: the keys above the k-th largest and, of those equal to it, the lowest columns;
// the same radix passes of 11, 11 and 10 bits over top_k_float_to_ordered's keys. Only the order in dst differs (each
// thread's columns, the threads in order: thread t holds columns j*1024 + t), which the QSA selection's readers do not
// see: they mark the chosen cells in a bitmap (tile and query lists) or scatter them (dense mask).
// What top_k_radix_row spends its time on and this kernel does not:
// - the row's first read, a loop of one load and one atomic add (one load in flight a wave), then a second read by the
//   selection after the first pass: here PER independent loads a thread, once;
// - the bins' suffix sum, 11 rounds of two barriers on each of the 3 passes: here a warp scan and one barrier;
// - the masked columns (-inf), all in one bin, 32 lanes adding to one address: one add a warp;
// - passes 2 and 3 when the k-th key's bin holds exactly the keys still to take (all of them are kept);
// - most of the first pass's adds: the keys below the smallest of the threads' largest keys cannot be chosen.
// Ties at the k-th key past the rank: two more passes over the equal keys' columns pick the lowest ones.
// Needs, defined before it is included (top-k.cu, ~/prefetch-experts/tools/qselbench): top_k_float_to_ordered,
// top_k_ncols_live, warp_reduce_sum, warp_prefix_inclusive_sum, ggml_cuda_get_physical_warp_size, CUDA_CHECK.

#define TOP_K_REG_THREADS 1024
#define TOP_K_REG_SUBHIST 2      // sub-histograms (warps modulo 2), as top_k_radix_row
#define TOP_K_REG_MAX_PER 64     // keys a thread at most: the mask of its selected keys is 64 bits

template <int PER>
__launch_bounds__(TOP_K_REG_THREADS, 1)
static __global__ void top_k_reg_row(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k, const int * __restrict__ n_live) {
    constexpr int T = TOP_K_REG_THREADS, NB = 2048, SUB = TOP_K_REG_SUBHIST;
    static_assert(NB == 2 * T, "two bins a thread");
    static_assert(PER <= TOP_K_REG_MAX_PER && PER * T <= 65536, "64-bit masks, columns' keys under 16 bits");
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps = T / warp_size;
    constexpr uint32_t key_ninf = 0x007FFFFFu;   // top_k_float_to_ordered(-INFINITY): the masked columns' key
    constexpr uint32_t no_hot   = 0xFFFFFFFFu;   // above any column's key: the columns' passes have no shared key
    __shared__ int hist[SUB][NB];
    __shared__ int s_wsum[n_warps];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_rank, s_ncand;
    __shared__ uint32_t s_tau;

    const int tid = threadIdx.x;
    const int lane = tid % warp_size, warp = tid / warp_size;
    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int * row_dst = dst + (size_t) blockIdx.x * k;
    const int live = top_k_ncols_live(ncols, k, n_live);
    const int nv = tid < live ? (live - tid + T - 1) / T : 0;   // this thread's keys: columns j*T + tid, j < nv

    // the row, coalesced, every load issued before the first is used (the address clamped into the row, not predicated)
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const uint32_t x = top_k_float_to_ordered(row_src[min(j*T + tid, live - 1)]);
        key[j] = j < nv ? x : 0;
    }
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_rank = k;
        s_tau = 0xFFFFFFFFu;
    }
    // tau, the smallest of the threads' largest keys: each thread with keys holds one at least tau, so at least
    // min(live, T) >= k keys are (k <= T), the k-th largest is one of them and so are all the keys equal to it. The keys
    // below tau can never be chosen and are left out of the passes: the same set (and tie rule) from fewer atomic adds
    // (~10 % of the keys left at 64 a thread, ~60 % at 8, for scores without structure along the row).
    uint32_t t_max = 0;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        t_max = j < nv ? max(t_max, key[j]) : t_max;
    }
    t_max = nv > 0 ? t_max : 0xFFFFFFFFu;
#pragma unroll
    for (int offset = warp_size/2; offset > 0; offset >>= 1) {
        t_max = min(t_max, (uint32_t) __shfl_xor_sync(0xffffffff, t_max, offset, warp_size));
    }
    __syncthreads();   // s_tau set
    if (lane == 0) {
        atomicMin(&s_tau, t_max);
    }
    __syncthreads();
    const uint32_t tau = k <= T ? s_tau : 0;

    // One radix pass over x_of(j), the keys of the eligible elements that match (s_prefix, s_mask): their histogram on
    // (x >> shift) & nbm, then the bin of the s_rank-th largest as top_k_radix_row picks it (the highest bin whose
    // suffix sum reaches the rank, bin 0 if none), from a warp scan of the bins (two a thread) and one barrier.
    // hot: a key many elements may share, counted by a warp sum and one add.
    // One call site an instantiation, in a loop: inlined (the AMDGPU inliner favours calls given a pointer to a private
    // array), so the keys stay in registers; a build must show no scratch for top_k_reg_row (private segment 0).
    const auto pass = [&](auto x_of, auto elig, const int shift, const uint32_t nbm, const uint32_t hot) {
        for (int i = tid; i < SUB*NB; i += T) {
            (&hist[0][0])[i] = 0;
        }
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        const int rank = s_rank;
        int * h = hist[warp % SUB];
        int n_hot = 0;
#pragma unroll
        for (int j = 0; j < PER; j++) {
            const uint32_t x = x_of(j);
            if (j < nv && elig(j) && (x & mask) == prefix) {
                if (x == hot) {
                    n_hot++;
                } else {
                    atomicAdd(&h[(x >> shift) & nbm], 1);
                }
            }
        }
        n_hot = warp_reduce_sum<warp_size>(n_hot);
        if (lane == 0 && n_hot > 0) {
            atomicAdd(&h[(hot >> shift) & nbm], n_hot);
        }
        __syncthreads();
        int c0 = 0, c1 = 0;   // bins 2 tid, 2 tid + 1
#pragma unroll
        for (int s = 0; s < SUB; s++) {
            c0 += hist[s][2*tid];
            c1 += hist[s][2*tid + 1];
        }
        const int loc = c0 + c1;
        const int incl = warp_prefix_inclusive_sum<int, warp_size>(loc);
        if (lane == warp_size - 1) {
            s_wsum[warp] = incl;
        }
        __syncthreads();
        int s_a = loc - incl;   // the keys in bins >= 2 tid: this warp's lanes from this one, the warps after it
        for (int w = warp; w < n_warps; w++) {
            s_a += s_wsum[w];
        }
        const int s_b = s_a - c0;   // bins >= 2 tid + 1
        const int s_c = s_b - c1;   // bins >= 2 tid + 2
        int b = -1, above = 0, cnt = 0;
        if (s_b >= rank && s_c < rank) {
            b = 2*tid + 1; above = s_c; cnt = c1;
        } else if ((s_a >= rank || tid == 0) && s_b < rank) {
            b = 2*tid;     above = s_b; cnt = c0;
        }
        if (b >= 0) {
            s_rank   = rank - above;
            s_prefix = prefix | ((uint32_t) b << shift);
            s_mask   = mask | (nbm << shift);
            s_ncand  = cnt;
        }
        __syncthreads();
    };

    const auto own = [&](int j) { return key[j]; };
    const auto kept = [&](int j) { return key[j] >= tau; };
    for (int p = 0; p < 3 && (p == 0 || s_ncand != s_rank); p++) {
        pass(own, kept, p == 0 ? 21 : p == 1 ? 10 : 0, p == 2 ? 1023u : 2047u, key_ninf);
    }
    // the bin holds exactly the keys still to take: all of them (mask partial after an early stop), else (the whole key
    // after 3 passes) more keys equal to the k-th than the rank: of those, the s_rank lowest columns, i.e. the largest
    // T*PER - 1 - column, by two passes over its 16 bits
    const uint32_t thr = s_prefix, tmask = s_mask;
    const bool ties = s_ncand != s_rank;
    const auto col_key = [&](int j) { return (uint32_t) (T*PER - 1 - (j*T + tid)); };
    uint32_t cthr = 0, cmask = 0;
    if (ties) {
        __syncthreads();   // thr read by every thread
        if (tid == 0) {
            s_prefix = 0;
            s_mask = 0;
        }
        const auto equal = [&](int j) { return key[j] == thr; };
        for (int p = 0; p < 2 && (p == 0 || s_ncand != s_rank); p++) {
            pass(col_key, equal, p == 0 ? 11 : 0, 2047u, no_hot);
        }
        cthr = s_prefix;
        cmask = s_mask;
    }

    uint64_t m = 0;
#pragma unroll
    for (int j = 0; j < PER; j++) {
        const uint32_t x = key[j] & tmask;
        // kept: after an early stop a key below tau may share the bin's high bits
        const bool s = j < nv && kept(j) && (x > thr || (x == thr && (!ties || (col_key(j) & cmask) >= cthr)));
        m |= (uint64_t) s << j;
    }
    // k columns in all: a block scan of the threads' counts places them
    const int cnt = __popcll(m);
    const int incl = warp_prefix_inclusive_sum<int, warp_size>(cnt);
    if (lane == warp_size - 1) {
        s_wsum[warp] = incl;   // the last pass read s_wsum before its final barrier
    }
    __syncthreads();
    int pos = incl - cnt;
    for (int w = 0; w < warp; w++) {
        pos += s_wsum[w];
    }
#pragma unroll
    for (int j = 0; j < PER; j++) {
        if ((m >> j) & 1) {
            row_dst[pos++] = j*T + tid;
        }
    }
}

// the launch for a row width (8 to 64 keys a thread); false past 65,536 columns
static bool top_k_reg_launch(const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream, const int * n_live) {
    constexpr int T = TOP_K_REG_THREADS;
    if (ncols > TOP_K_REG_MAX_PER*T || nrows <= 0) {
        return false;
    }
    if (ncols <= 8*T) {
        top_k_reg_row<8><<<nrows, T, 0, stream>>>(src, dst, ncols, k, n_live);
    } else if (ncols <= 16*T) {
        top_k_reg_row<16><<<nrows, T, 0, stream>>>(src, dst, ncols, k, n_live);
    } else if (ncols <= 32*T) {
        top_k_reg_row<32><<<nrows, T, 0, stream>>>(src, dst, ncols, k, n_live);
    } else if (ncols <= 48*T) {
        top_k_reg_row<48><<<nrows, T, 0, stream>>>(src, dst, ncols, k, n_live);
    } else {
        top_k_reg_row<64><<<nrows, T, 0, stream>>>(src, dst, ncols, k, n_live);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
