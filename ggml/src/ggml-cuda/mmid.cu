#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
// One block per expert; each warp of the block scans a contiguous slice of the tokens into its own part of the shared
// store, then the slices are written one after the other: the same order as one warp scanning all of them.
// direct: the store does not fit the shared memory (4 bytes a token: ~16k tokens on RDNA3, ~25k on Ada); the slices are
// scanned twice instead, once to count, once to write each entry at its place.
#define MM_IDS_HELPER_MAX_WARPS 8
template <int n_expert_used_template>
__launch_bounds__(MM_IDS_HELPER_MAX_WARPS*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse,
        const bool direct) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;
    const int n_warps = blockDim.x / warp_size;
    const int warp    = threadIdx.x / warp_size;
    const int lane    = threadIdx.x % warp_size;
    const int slice   = (n_tokens + n_warps - 1) / n_warps;
    const int it_beg  = warp*slice;
    const int it_end  = min(n_tokens, it_beg + slice);

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper + it_beg;
    __shared__ int warp_compact[MM_IDS_HELPER_MAX_WARPS];
    __shared__ int warp_prev[MM_IDS_HELPER_MAX_WARPS];

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    // direct: this warp's first column once known, and whether the scan writes
    int  col_base = 0;
    bool emit     = !direct;
    auto put = [&](const int itc, const int it, const int iex_used) {
        if (!direct) {
            store[itc] = mm_ids_helper_store(it, iex_used);
            return;
        }
        const int col = col_base + itc;
        ids_dst[col] = it*n_expert_used + iex_used;
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = col;
        } else {
            ids_src1[col] = it*sis1 + iex_used % nchannels_y;
        }
    };

    for (int pass = 0; pass < (direct ? 2 : 1); ++pass) {
    if (pass == 1) {
        it_compact = 0;
    }
    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = it_beg; it < it_end; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = lane; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1 && emit) {
                put(it_compact, it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        // the ids of n_loads steps are read before any of them is used (one step per global read latency otherwise);
        // steps past the slice read nothing and change nothing
        constexpr int tokens_per_step = warp_size/neu_padded;
        constexpr int n_loads         = 16;
        const int iex = lane % neu_padded; // The index at which the expert is used, if any.
        for (int it00 = it_beg; it00 < it_end; it00 += n_loads*tokens_per_step) {
            int expert_used_step[n_loads];
#pragma unroll
            for (int s = 0; s < n_loads; ++s) {
                const int it = it00 + s*tokens_per_step + lane / neu_padded;
                expert_used_step[s] = (neu_padded == n_expert_used || iex < n_expert_used) && it < it_end ?
                    ids[it*si1 + iex] : INT_MAX;
            }
#pragma unroll
            for (int s = 0; s < n_loads; ++s) {
                const int it = it00 + s*tokens_per_step + lane / neu_padded;
                const int expert_used = expert_used_step[s];
                const int iex_used = expert_used == expert ? iex : -1;
                nex_prev += expert_used < expert;

                // Whether the threads at this token position have used the expert:
                const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

                // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
                int it_compact_add_lower = 0;
#pragma unroll
                for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                    const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                    if (lane >= offset) {
                        it_compact_add_lower += tmp;
                    }
                }

                if (iex_used != -1 && emit) {
                    put(it_compact + it_compact_add_lower, it, iex_used);
                }

                // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
                it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
            }
        }
    }
    if (pass == 1) {
        break; // the counts are those of the first pass
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    if (lane == 0) {
        warp_compact[warp] = it_compact;
        warp_prev[warp]    = nex_prev;
    }
    __syncthreads();
    if (direct) {
        int c = 0;
        for (int w = 0; w < n_warps; ++w) {
            c += warp_prev[w] + (w < warp ? warp_compact[w] : 0);
        }
        col_base = c;
        emit     = true;
    }
    }

    // this expert's first column, and this warp's place in its compact slice
    int col0 = 0;
    int col_warp = 0;
    int n_compact = 0;
    for (int w = 0; w < n_warps; ++w) {
        col0      += warp_prev[w];
        col_warp  += w < warp ? warp_compact[w] : 0;
        n_compact += warp_compact[w];
    }

    for (int itc = lane; !direct && itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        const int col      = col0 + col_warp + itc;
        ids_dst[col] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = col;
        } else {
            ids_src1[col] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = col0;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = col0 + n_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    // the dynamic limit leaves room for the kernel's static arrays (warp_compact, warp_prev): CUDA refuses a dynamic
    // limit that adds up with them to more than smpbo ("invalid argument", the 19k prefill on an RTX 4090)
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo - 2*MM_IDS_HELPER_MAX_WARPS*sizeof(int));

    // a warp per 512 tokens: a small batch keeps one warp, as before
    const int n_warps = n_expert_used_template == 0 ? 1 : std::max(1, std::min(MM_IDS_HELPER_MAX_WARPS, n_tokens/512));
    const int slice   = (n_tokens + n_warps - 1) / n_warps;
    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(n_warps*warp_size, 1, 1);
    // the static arrays take their part of the block's shared memory
    const size_t nbytes_store  = (size_t) n_warps*slice*sizeof(mm_ids_helper_store);
    const bool   direct        = nbytes_store > smpbo - 2*MM_IDS_HELPER_MAX_WARPS*sizeof(int);
    const size_t nbytes_shared = direct ? 0 : nbytes_store;
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse, direct);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}
