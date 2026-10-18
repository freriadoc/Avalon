// coop.hip : are all the blocks of a grid-barrier kernel resident at once? (the occupancy API's estimate, with or
// without another queue's kernel running), and can a cooperative launch be captured in a HIP graph?
// Each block announces itself, then waits (with a timeout, no hang) for the whole grid to have announced itself.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s -> %s\n", #x, hipGetErrorString(e)); } } while (0)

__device__ unsigned g_arrived;
__device__ unsigned g_timeouts;

// a block of 256 threads, the dynamic shared memory of k_mk_chain256 (1152 bytes)
template <int NT>
__launch_bounds__(NT, 1)
__global__ void k_meet_t(unsigned long long timeout_cycles) {
    extern __shared__ float smem[];
    if (threadIdx.x == 0) {
        smem[0] = 1.0f;
        atomicAdd(&g_arrived, 1u);
        const unsigned long long t0 = wall_clock64();
        while (__hip_atomic_load(&g_arrived, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < gridDim.x) {
            if (wall_clock64() - t0 > timeout_cycles) {
                atomicAdd(&g_timeouts, 1u);
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
}
#define k_meet k_meet_t<256>

// another queue's work: blocks of 256 threads spinning for a while
__global__ void k_hog(unsigned long long cycles) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < cycles) {
        __builtin_amdgcn_s_sleep(1);
    }
}

static int run(int grid, bool coop, hipStream_t s, unsigned long long timeout) {
    unsigned zero = 0;
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_arrived), &zero, sizeof(zero)));
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_timeouts), &zero, sizeof(zero)));
    if (coop) {
        void * args[] = { &timeout };
        hipError_t e = hipLaunchCooperativeKernel((const void *) k_meet, dim3(grid), dim3(256), args, 1152, s);
        if (e != hipSuccess) {
            printf("  cooperative launch of %d blocks: %s\n", grid, hipGetErrorString(e));
            (void) hipGetLastError();
            return -1;
        }
    } else {
        k_meet<<<grid, 256, 1152, s>>>(timeout);
    }
    CHECK(hipStreamSynchronize(s));
    unsigned to = 0;
    CHECK(hipMemcpyFromSymbol(&to, HIP_SYMBOL(g_timeouts), sizeof(to)));
    return (int) to;
}

int main() {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    int per_cu = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_meet, 256, 1152));
    int clk = 0;
    CHECK(hipDeviceGetAttribute(&clk, hipDeviceAttributeWallClockRate, 0));   // kHz
    const unsigned long long timeout = (unsigned long long) clk * 200;          // 200 ms
    printf("multiProcessorCount %d, occupancy %d blocks per multiprocessor -> %d; cooperativeLaunch %d\n",
           p.multiProcessorCount, per_cu, per_cu * p.multiProcessorCount, p.cooperativeLaunch);
    hipStream_t a, b;
    CHECK(hipStreamCreateWithFlags(&a, hipStreamNonBlocking));
    CHECK(hipStreamCreateWithFlags(&b, hipStreamNonBlocking));
    const int cap = per_cu * p.multiProcessorCount;
    printf("alone, plain launch (blocks that timed out):\n");
    for (int g : { cap / 2, cap - 8, cap, cap + 8, 2 * cap }) {
        printf("  grid %4d: %d\n", g, run(g, false, a, timeout));
    }
    printf("alone, cooperative launch:\n");
    for (int g : { cap / 2, cap, cap + 8, 2 * cap }) {
        const int r = run(g, true, a, timeout);
        if (r >= 0) printf("  grid %4d: %d\n", g, r);
    }
    // another queue busy with 2 x 42 blocks spinning 50 ms, launched just before
    const unsigned long long hog = (unsigned long long) clk * 50;
    for (int coop = 0; coop < 2; coop++) {
        printf("with another queue's kernel (%s launch):\n", coop ? "cooperative" : "plain");
        for (int g : { cap / 2, cap }) {
            k_hog<<<2 * p.multiProcessorCount, 256, 0, b>>>(hog);
            const int r = run(g, coop, a, timeout);
            CHECK(hipStreamSynchronize(b));
            if (r >= 0) printf("  grid %4d: %d\n", g, r);
        }
    }
    // how much room another queue may take: grids of the chain kernels (64 blocks, the top-k's 96, k_mk_chain128's
    // at most 168) against another queue's kernel of 42 to 336 blocks of 256 threads
    printf("plain launch against another queue's kernel (blocks that timed out):\n");
    for (int h : { 168, 252, 336 }) {
        printf("  other queue %3d blocks:", h);
        for (int g : { 96, 104, 112, 120, 128, 144, 168 }) {
            k_hog<<<h, 256, 0, b>>>(hog);
            const int r = run(g, false, a, timeout);
            CHECK(hipStreamSynchronize(b));
            printf("  grid %3d: %3d", g, r);
        }
        printf("\n");
    }
    printf("blocks of 128 threads (k_mk_chain128) against another queue's full kernel (336 x 256):\n");
    for (int g : { 168, 184, 192, 200, 208, 256 }) {
        unsigned z = 0;
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_arrived), &z, sizeof(z)));
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_timeouts), &z, sizeof(z)));
        k_hog<<<336, 256, 0, b>>>(hog);
        k_meet_t<128><<<g, 128, 128, a>>>(timeout);
        CHECK(hipStreamSynchronize(a));
        CHECK(hipStreamSynchronize(b));
        unsigned to = 0;
        CHECK(hipMemcpyFromSymbol(&to, HIP_SYMBOL(g_timeouts), sizeof(to)));
        printf("  grid %3d: %u\n", g, to);
    }
    // capture of a cooperative launch in a graph, replayed
    printf("graph capture of a cooperative launch:\n");
    unsigned zero = 0;
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_arrived), &zero, sizeof(zero)));
    CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_timeouts), &zero, sizeof(zero)));
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    CHECK(hipStreamBeginCapture(a, hipStreamCaptureModeRelaxed));
    unsigned long long to = timeout;
    void * args[] = { &to };
    hipError_t e = hipLaunchCooperativeKernel((const void *) k_meet, dim3(cap / 2), dim3(256), args, 1152, a);
    printf("  launch while capturing: %s\n", hipGetErrorString(e));
    CHECK(hipStreamEndCapture(a, &graph));
    if (graph) {
        size_t n = 0;
        CHECK(hipGraphGetNodes(graph, nullptr, &n));
        printf("  nodes in the graph: %zu\n", n);
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int r = 0; r < 3 && exec; r++) {
            CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_arrived), &zero, sizeof(zero)));
            CHECK(hipGraphLaunch(exec, a));
            CHECK(hipStreamSynchronize(a));
            unsigned t = 0, arr = 0;
            CHECK(hipMemcpyFromSymbol(&t, HIP_SYMBOL(g_timeouts), sizeof(t)));
            CHECK(hipMemcpyFromSymbol(&arr, HIP_SYMBOL(g_arrived), sizeof(arr)));
            printf("  replay %d: %u blocks arrived, %u timeouts so far\n", r, arr, t);
        }
    }
    return 0;
}
