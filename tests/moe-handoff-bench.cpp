// moe-handoff-bench : cost of handing a MoE layer from the GPU to the CPU and back, per layer.
//
// A layer is: a GPU kernel (the mixer and router, spinning PRE_US), the activations (10 KiB) to the
// host, the CPU computing the missed experts (spinning CPU_US), their results (20 KiB) back, a GPU
// kernel adding them (spinning POST_US). Three ways to hand over:
//   V0 host-driven (as the scheduler does): per layer the host launches, copies, synchronizes.
//   V1 queued ahead: the whole pass is queued at once; the GPU writes a flag behind the download
//      (hipStreamWriteValue32), a CPU thread polls it, computes, writes the results into mapped
//      host memory and a flag; the stream waits for that flag (hipStreamWaitValue32) and the
//      kernel reads the results in place.
//   V2 as V1, but the stream waits in a one-thread kernel polling the flag in coherent host memory.
// Printed: the overhead per layer = (wall time - layers * (pre + post + cpu)) / layers.
//
// hipcc -O2 -std=c++17 -o moe-handoff-bench tests/moe-handoff-bench.cpp -lpthread
// ./moe-handoff-bench [CPU_US] [PRE_US] [POST_US]
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1); } } while (0)

static constexpr int LAYERS  = 48;
static constexpr int ACT_F   = 2560;       // activations of a token (10 KiB)
static constexpr int RES_F   = 2 * 2560;   // two missed experts' outputs (20 KiB)
static constexpr int REPS    = 200;

static int64_t g_clock_khz = 0;

__global__ void k_pre(float * act, long long spin, int layer) {
    const long long t0 = wall_clock64();
    while (wall_clock64() - t0 < spin) {}
    for (int i = threadIdx.x; i < ACT_F; i += blockDim.x) act[i] = (float) (layer + i);
}

__global__ void k_post(const float * res, float * acc, long long spin) {
    const long long t0 = wall_clock64();
    float s = 0.0f;
    for (int i = threadIdx.x; i < RES_F; i += blockDim.x) s += res[i];
    atomicAdd(acc, s);
    while (wall_clock64() - t0 < spin) {}
}

__global__ void k_wait_flag(const volatile uint32_t * flag, uint32_t value) {
    while (*flag < value) { __builtin_amdgcn_s_sleep(1); }
    __threadfence_system();
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void cpu_spin(double us) {
    const double t0 = now_us();
    while (now_us() - t0 < us) {}
}

int main(int argc, char ** argv) {
    const double cpu_us  = argc > 1 ? atof(argv[1]) : 120.0;
    const double pre_us  = argc > 2 ? atof(argv[2]) : 20.0;
    const double post_us = argc > 3 ? atof(argv[3]) : 10.0;

    int dev = 0, rate = 0, can_wait = 0;
    CHECK(hipSetDevice(dev));
    CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, dev));
    CHECK(hipDeviceGetAttribute(&can_wait, hipDeviceAttributeCanUseStreamWaitValue, dev));
    g_clock_khz = rate;
    const long long pre_spin  = (long long) (pre_us  * rate / 1000.0);
    const long long post_spin = (long long) (post_us * rate / 1000.0);
    printf("wall clock %d kHz, stream wait value %s; cpu %.0f us, pre %.0f us, post %.0f us per layer\n",
           rate, can_wait ? "supported" : "NOT supported", cpu_us, pre_us, post_us);

    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    float * d_act, * d_res, * d_acc;
    CHECK(hipMalloc(&d_act, ACT_F * sizeof(float)));
    CHECK(hipMalloc(&d_res, LAYERS * RES_F * sizeof(float)));
    CHECK(hipMalloc(&d_acc, sizeof(float)));

    // host side: pinned activations per layer, results in mapped coherent memory
    float * h_act, * h_res;
    CHECK(hipHostMalloc((void **) &h_act, LAYERS * ACT_F * sizeof(float), hipHostMallocDefault));
    CHECK(hipHostMalloc((void **) &h_res, LAYERS * RES_F * sizeof(float), hipHostMallocMapped | hipHostMallocCoherent));
    float * m_res;
    CHECK(hipHostGetDevicePointer((void **) &m_res, h_res, 0));
    std::vector<float> cpu_out(RES_F, 1.0f);

    // flags: A (GPU -> CPU, written by the stream), B (CPU -> GPU)
    uint32_t * sigA, * sigB;           // signal memory, for the stream's write / wait
    CHECK(hipExtMallocWithFlags((void **) &sigA, 8, hipMallocSignalMemory));
    CHECK(hipExtMallocWithFlags((void **) &sigB, 8, hipMallocSignalMemory));
    uint32_t * h_flagB, * m_flagB;     // coherent mapped, for V2's polling kernel
    CHECK(hipHostMalloc((void **) &h_flagB, 64, hipHostMallocMapped | hipHostMallocCoherent));
    CHECK(hipHostGetDevicePointer((void **) &m_flagB, h_flagB, 0));

    // GPU-only reference: pre + post back to back
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    CHECK(hipEventRecord(e0, s));
    for (int l = 0; l < LAYERS; l++) {
        k_pre<<<1, 256, 0, s>>>(d_act, pre_spin, l);
        k_post<<<1, 256, 0, s>>>(d_res + l * RES_F, d_acc, post_spin);
    }
    CHECK(hipEventRecord(e1, s));
    CHECK(hipEventSynchronize(e1));
    float gpu_ms = 0;
    CHECK(hipEventElapsedTime(&gpu_ms, e0, e1));
    const double gpu_layer = gpu_ms * 1000.0 / LAYERS;
    printf("GPU alone: %.1f us per layer (pre + post)\n", gpu_layer);

    auto report = [&](const char * name, double total_us) {
        const double per = total_us / LAYERS;
        printf("%-44s %7.1f us per layer, overhead %6.1f us\n", name, per, per - gpu_layer - cpu_us);
    };

    // V0: host-driven, as the scheduler: launch, copy down, synchronize, compute, copy up
    {
        double best = 1e30;
        for (int r = 0; r < REPS; r++) {
            CHECK(hipStreamSynchronize(s));
            const double t0 = now_us();
            for (int l = 0; l < LAYERS; l++) {
                k_pre<<<1, 256, 0, s>>>(d_act, pre_spin, l);
                CHECK(hipMemcpyAsync(h_act + l * ACT_F, d_act, ACT_F * sizeof(float), hipMemcpyDeviceToHost, s));
                CHECK(hipStreamSynchronize(s));
                cpu_spin(cpu_us);
                memcpy(h_res + l * RES_F, cpu_out.data(), RES_F * sizeof(float));
                CHECK(hipMemcpyAsync(d_res + l * RES_F, h_res + l * RES_F, RES_F * sizeof(float), hipMemcpyHostToDevice, s));
                k_post<<<1, 256, 0, s>>>(d_res + l * RES_F, d_acc, post_spin);
            }
            CHECK(hipStreamSynchronize(s));
            best = std::min(best, now_us() - t0);
        }
        report("V0 host-driven (sync + copies)", best);
    }

    // V1 / V2: the pass queued at once, a CPU thread answering the flags
    for (int variant = 1; variant <= 2; variant++) {
        double best = 1e30;
        uint32_t base = 0;
        for (int r = 0; r < REPS; r++) {
            CHECK(hipStreamSynchronize(s));
            std::atomic<bool> go{false};
            std::thread worker([&, base] {
                while (!go.load(std::memory_order_acquire)) {}
                for (int l = 0; l < LAYERS; l++) {
                    const uint32_t want = base + l + 1;
                    while (__atomic_load_n(sigA, __ATOMIC_ACQUIRE) < want) {}
                    cpu_spin(cpu_us);
                    memcpy(h_res + l * RES_F, cpu_out.data(), RES_F * sizeof(float));
                    if (variant == 1) {
                        __atomic_store_n(sigB, want, __ATOMIC_RELEASE);
                    } else {
                        __atomic_store_n(h_flagB, want, __ATOMIC_RELEASE);
                    }
                }
            });
            const double t0 = now_us();
            go.store(true, std::memory_order_release);
            for (int l = 0; l < LAYERS; l++) {
                const uint32_t want = base + l + 1;
                k_pre<<<1, 256, 0, s>>>(d_act, pre_spin, l);
                CHECK(hipMemcpyAsync(h_act + l * ACT_F, d_act, ACT_F * sizeof(float), hipMemcpyDeviceToHost, s));
                CHECK(hipStreamWriteValue32(s, sigA, want, 0));
                if (variant == 1) {
                    CHECK(hipStreamWaitValue32(s, sigB, want, hipStreamWaitValueGte, 0xFFFFFFFF));
                } else {
                    k_wait_flag<<<1, 1, 0, s>>>(m_flagB, want);
                }
                k_post<<<1, 256, 0, s>>>(m_res + l * RES_F, d_acc, post_spin);
            }
            CHECK(hipStreamSynchronize(s));
            const double t = now_us() - t0;
            worker.join();
            best = std::min(best, t);
            base += LAYERS;
        }
        report(variant == 1 ? "V1 queued, stream write/wait value" : "V2 queued, polling kernel", best);
    }
    return 0;
}
