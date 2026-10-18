#pragma once

#include "common.cuh"

// A grid-wide barrier for a kernel whose blocks are all resident at once (grid sized from the occupancy): a counter
// and a generation in the module's memory (no allocation, nothing to set up in a graph), a pair per kernel that may
// run at the same time as another. The kernel reads the generation once at its start
// (ggml_cuda_grid_barrier_start), before any block can pass a barrier. Release and acquire at agent scope: what a
// block wrote before the barrier is seen by every block after it (the per-CU caches invalidated).
// The blocks of a kernel are not all dispatched while another queue's kernels hold the device: on the RX 7900 XT,
// a grid of at most 768 waves always met (blocks of 256 or 128 threads, the device full with another queue's 336
// blocks of 256 threads), 800 never did, not even once the other kernel had ended, and a cooperative launch changes
// nothing (tests/coopbench/coop.hip). A measured bound, of this device: the grid of a kernel with a grid barrier stays
// under it.
#define GGML_CUDA_GRID_BARRIER_MAX_WAVES 768

static int ggml_cuda_grid_barrier_max_blocks(int threads_per_block) {
    const int waves = (threads_per_block + WARP_SIZE - 1) / WARP_SIZE;
    return GGML_CUDA_GRID_BARRIER_MAX_WAVES / waves;
}

static __device__ __forceinline__ unsigned ggml_cuda_grid_barrier_start(const unsigned * gen_var) {
    __shared__ unsigned s_gen;
    if (threadIdx.x == 0) {
        s_gen = __hip_atomic_load(gen_var, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    return s_gen;
}

static __device__ __forceinline__ void ggml_cuda_grid_barrier(unsigned * count, unsigned * gen_var, unsigned & gen) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned target = gen + 1;
        if (atomicAdd(count, 1u) == gridDim.x - 1) {
            atomicExch(count, 0u);
            __threadfence();
            atomicExch(gen_var, target);
        } else {
            while (__hip_atomic_load(gen_var, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != target) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    gen++;
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}
