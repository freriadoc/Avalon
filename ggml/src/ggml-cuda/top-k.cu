#include "argsort.cuh"
#include "top-k.cuh"
#include "ggml-backend-impl.h"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// the columns a row takes: all, or max(live count, k) (ggml_set_live_count: the columns past the count score -inf)
static __device__ __forceinline__ int top_k_ncols_live(int ncols, int k, const int * __restrict__ n_live) {
    return n_live ? min(ncols, max(*n_live, k)) : ncols;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift,
        int k,
        const int * __restrict__ n_live) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    const int ncols_live = top_k_ncols_live(ncols, k, n_live);
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols_live;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
#pragma unroll 8
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    // the bin of the rank-th largest key, by a suffix sum over the bins (one thread walking the bins from
    // the top took most of the kernel's time); the rank read before the barriers, the chosen thread rewrites it
    const int rank = states[row].rank;
    histogram[tid] = count;
    __syncthreads();
    for (int off = 1; off < NBINS; off <<= 1) {
        const int add = tid + off < NBINS ? histogram[tid + off] : 0;
        __syncthreads();
        histogram[tid] += add;
        __syncthreads();
    }
    // histogram[b]: the keys in bins >= b; the bin is the highest one whose suffix reaches the rank (bin 0
    // if none does, as the walk ended)
    const int suffix = histogram[tid];
    const int above = tid + 1 < NBINS ? histogram[tid + 1] : 0;
    if ((suffix >= rank || tid == 0) && above < rank) {
        top_k_radix_state state = states[row];
        state.rank -= above;
        state.prefix |= (uint32_t) tid << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// The gather is deterministic: each block of a row takes a contiguous run of columns, counts its keys above the
// k-th value's key (prefix) and equal to it, then writes them in column order after the counts of the blocks
// before it. Ties at the k-th value keep the lowest columns. (Gathering through atomic counters, the order of
// the columns and which of several equal keys were kept followed the threads' timing: two runs of an indexer's
// top-k could keep different blocks when scores tie at the cut, and everything after them differed.)
template<int BLOCK_SIZE>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int2 * __restrict__ counts,
        int ncols,
        int chunk,
        int blocks_per_row,
        int k,
        const int * __restrict__ n_live) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t prefix = states[row].prefix;
    const int begin = row_block * chunk;
    const int end = min(begin + chunk, top_k_ncols_live(ncols, k, n_live));

    int g = 0;
    int e = 0;
    for (int col = begin + tid; col < end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        g += key > prefix;
        e += key == prefix;
    }
    // block sums through the warps (a shared counter for every thread serialized them)
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps = BLOCK_SIZE / warp_size;
    __shared__ int warp_g[n_warps];
    __shared__ int warp_e[n_warps];
    g = warp_reduce_sum<warp_size>(g);
    e = warp_reduce_sum<warp_size>(e);
    if (tid % warp_size == 0) {
        warp_g[tid / warp_size] = g;
        warp_e[tid / warp_size] = e;
    }
    __syncthreads();
    if (tid == 0) {
        int greater = 0;
        int equal = 0;
        for (int w = 0; w < n_warps; ++w) {
            greater += warp_g[w];
            equal += warp_e[w];
        }
        counts[(size_t) row * blocks_per_row + row_block] = make_int2(greater, equal);
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int2 * __restrict__ counts,
        int ncols,
        int k,
        int chunk,
        int blocks_per_row,
        const int * __restrict__ n_live) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const uint32_t prefix = states[row].prefix;
    const int rank = states[row].rank;
    const int begin = row_block * chunk;
    const int end = min(begin + chunk, top_k_ncols_live(ncols, k, n_live));

    int g_off = 0;
    int e_off = 0;
#pragma unroll 8
    for (int b = 0; b < row_block; ++b) {
        const int2 c = counts[(size_t) row * blocks_per_row + b];
        g_off += c.x;
        e_off += c.y;
    }

    // a block's positions from the warps' ballots: two barriers per BLOCK_SIZE columns
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int n_warps = BLOCK_SIZE / warp_size;
    __shared__ int warp_g[n_warps];
    __shared__ int warp_e[n_warps];
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const uint64_t below = (1ull << lane) - 1;
    for (int base = begin; base < end; base += BLOCK_SIZE) {
        const int col = base + tid;
        const uint32_t key = col < end ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool is_g = col < end && key > prefix;
        const bool is_e = col < end && key == prefix;
        const uint64_t mg = __ballot(is_g);
        const uint64_t me = __ballot(is_e);
        if (lane == 0) {
            warp_g[warp] = __popcll(mg);
            warp_e[warp] = __popcll(me);
        }
        __syncthreads();
        int pg = 0, pe = 0, tg = 0, te = 0;
        for (int w = 0; w < n_warps; ++w) {
            if (w < warp) {
                pg += warp_g[w];
                pe += warp_e[w];
            }
            tg += warp_g[w];
            te += warp_e[w];
        }
        if (is_g) {
            row_dst[g_off + pg + __popcll(mg & below)] = col;
        }
        if (is_e) {
            const int pos = e_off + pe + __popcll(me & below);
            if (pos < rank) {
                row_dst[k - rank + pos] = col;
            }
        }
        g_off += tg;
        e_off += te;
        __syncthreads();
    }
}

static constexpr int top_k_radix_bins = 1 << 8;

static int top_k_radix_blocks_per_row(int ncols) {
    return std::min((ncols + 1023) / 1024, 64);
}

// sized for the next power of two of blocks_per_row: it grows with ncols (with the context, for an indexer's
// scores), and the legacy pool keeps every buffer it outgrows; this way they add up to at most the last one
static size_t top_k_radix_hist_count(int ncols, int nrows) {
    int hist_blocks = 1;
    while (hist_blocks < top_k_radix_blocks_per_row(ncols)) {
        hist_blocks *= 2;
    }
    return (size_t) nrows * hist_blocks * top_k_radix_bins;
}

static size_t top_k_radix_states_bytes(int nrows) {
    return GGML_PAD(nrows * sizeof(top_k_radix_state), 256);
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool, char * scratch,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream, const int * n_live) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == top_k_radix_bins, "top-k radix bins");
    const int blocks_per_row = top_k_radix_blocks_per_row(ncols);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool);
    top_k_radix_state * states = scratch ? (top_k_radix_state *) scratch : states_alloc.alloc(nrows);
    int * histograms = scratch ? (int *) (scratch + top_k_radix_states_bytes(nrows)) : histograms_alloc.alloc(top_k_radix_hist_count(ncols, nrows));

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift, k, n_live);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    // the histograms are no longer read: they hold the blocks' counts (2 per block <= NBINS per block)
    const int chunk = (ncols + blocks_per_row - 1) / blocks_per_row;
    int2 * counts = (int2 *) histograms;
    top_k_radix_count<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts, ncols, chunk, blocks_per_row, k, n_live);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, counts, ncols, k, chunk, blocks_per_row, n_live);
}

#endif // !defined(GGML_CUDA_USE_CUB) && defined(GGML_USE_HIP)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    // the scheduler's scratch when it gave one (GGML_SCHED_SCRATCH_SRC), else pool memory
    const ggml_tensor * scratch_t = dst->src[GGML_SCHED_SCRATCH_SRC];
    const size_t        need      = ggml_cuda_top_k_scratch_size(dst);
    char * scratch = need > 0 && scratch_t && scratch_t->data && ggml_nbytes(scratch_t) >= need ? (char *) scratch_t->data : nullptr;
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        // the live count (ggml_set_live_count): only that many columns are read
        const ggml_tensor * lc = dst->src[GGML_LIVE_COUNT_SRC];
        const int * n_live = lc ? (const int *) lc->data + ggml_get_op_params_i32(dst, GGML_LIVE_COUNT_PARAM) : nullptr;
        top_k_radix_cuda(pool, scratch, src0_d, dst_d, ncols, nrows, k, stream, n_live);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool);
        int *                     tmp_dst = scratch ? (int *) scratch : temp_dst_alloc.alloc(ncols * nrows);
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

size_t ggml_cuda_top_k_scratch_size(const ggml_tensor * dst) {
#if !defined(GGML_CUDA_USE_CUB)
    const ggml_tensor * src0 = dst->src[0];
    const int ncols = (int) src0->ne[0];
    const int nrows = (int) ggml_nrows(src0);
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        return top_k_radix_states_bytes(nrows) + top_k_radix_hist_count(ncols, nrows) * sizeof(int);
    }
#endif
    // the bitonic sort's full index rows
    return (size_t) ncols * nrows * sizeof(int);
#else
    GGML_UNUSED(dst);
    return 0;
#endif
}
