// moe-plan-bench : latency of planning a MoE layer on the host while the GPU waits, per layer.
//
// The pass is captured in a HIP graph. Per layer the GPU writes the router's 10 expert ids into
// mapped host memory and raises flag A (a one-thread kernel); a CPU thread polling A plans the
// layer with the MoE cache's structures (key, hash, map, slots, LRU, pins, demand counts: copied
// from ggml-cuda/moe-cache.cu), writes the plan (the resident rows' slots, the missed rows) into
// mapped memory and raises flag P; a polling kernel waits for P, then a kernel reads the plan.
// Measured on the GPU's wall clock: flag A written -> flag P seen, per layer; and the host's time
// spent in the plan itself.
//
// Pools as the cache builds them for qwen4exp (gate/up IQ2_S on 36 layers, IQ3_S on 12, down IQ4_NL
// on 48; ~125-156 resident experts per tensor), routing drawn from a Zipf law (s = 1) over the 512
// experts of a layer, the resident ones being each layer's most popular: ~80 % of rows resident.
//
// hipcc -O2 -std=c++17 --offload-arch=gfx1100 -o moe-plan-bench tests/moe-plan-bench.cpp -lpthread
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e)); exit(1); } } while (0)

static constexpr int LAYERS = 48;
static constexpr int N_EXPERT = 512;
static constexpr int TOPK = 10;
static constexpr int PASSES = 200;
static constexpr int PLAN_INTS = 4 + 3 * TOPK + TOPK;   // counts, 3 slots per resident row, missed rows

// ---- the cache's structures (moe-cache.cu)
struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;
    bool operator==(const moe_cache_key & other) const { return tensor == other.tensor && expert == other.expert; }
};
struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};
enum class slot_state { free, copying, valid };
struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1, next = -1, readers = 0;
    slot_state state = slot_state::free;
};
struct moe_cache_pool {
    std::vector<moe_cache_slot> slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1, lru_tail = -1;
};
static void lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) pool.slots[slot.prev].next = slot.next; else pool.lru_head = slot.next;
    if (slot.next >= 0) pool.slots[slot.next].prev = slot.prev; else pool.lru_tail = slot.prev;
    slot.prev = slot.next = -1;
}
static void lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = pool.lru_tail; slot.next = -1;
    if (pool.lru_tail >= 0) pool.slots[pool.lru_tail].next = index; else pool.lru_head = index;
    pool.lru_tail = index;
}

struct layer_tensors { const void * up, * gate, * down; int pool; };

struct planner {
    std::mutex mu;
    moe_cache_pool pools[3];     // 0: IQ2_S gate/up, 1: IQ3_S gate/up, 2: IQ4_NL down
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> demand;
    std::vector<layer_tensors> layers;

    int resident(moe_cache_pool & pool, const void * tensor, int expert) {
        const auto it = pool.map.find({tensor, expert});
        return it != pool.map.end() && pool.slots[it->second].state == slot_state::valid ? it->second : -1;
    }
    // as moe_cache_fused_begin with down: rows whose three tensors are resident are pinned and
    // touched; the others count toward admission
    void plan(int l, const int32_t * ids, int32_t * out) {
        std::lock_guard<std::mutex> lock(mu);
        const layer_tensors & t = layers[l];
        moe_cache_pool & p = pools[t.pool];
        moe_cache_pool & d = pools[2];
        int hits = 0, misses = 0;
        for (int i = 0; i < TOPK; i++) {
            const int su = resident(p, t.up, ids[i]), sg = resident(p, t.gate, ids[i]), sd = resident(d, t.down, ids[i]);
            if (su >= 0 && sg >= 0 && sd >= 0) {
                for (int s : {su, sg}) { p.slots[s].readers++; lru_remove(p, s); lru_push_back(p, s); }
                d.slots[sd].readers++; lru_remove(d, sd); lru_push_back(d, sd);
                out[4 + 3 * hits + 0] = su; out[4 + 3 * hits + 1] = sg; out[4 + 3 * hits + 2] = sd;
                hits++;
            } else {
                out[4 + 3 * TOPK + misses++] = i;
                for (const void * base : {t.up, t.gate, t.down}) {
                    moe_cache_pool & q = base == t.down ? d : p;
                    if (q.map.find({base, ids[i]}) == q.map.end()) demand[{base, ids[i]}]++;
                }
            }
        }
        out[0] = hits; out[1] = misses;
    }
};

__global__ void k_publish_ids(const int32_t * ids, volatile int32_t * m_ids, volatile uint32_t * flag,
                              long long * t_write, uint32_t value) {
    for (int i = 0; i < TOPK; i++) m_ids[i] = ids[i];
    __threadfence_system();
    *t_write = wall_clock64();
    *flag = value;
    __threadfence_system();
}

__global__ void k_wait_flag(const volatile uint32_t * flag, uint32_t value, long long * t_seen) {
    while (*flag < value) { __builtin_amdgcn_s_sleep(1); }
    __threadfence_system();
    *t_seen = wall_clock64();
}

__global__ void k_read_plan(const volatile int32_t * plan, int * acc) {
    int s = 0;
    for (int i = 0; i < PLAN_INTS; i++) s += plan[i];
    atomicAdd(acc, s);
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int rate = 0;
    CHECK(hipDeviceGetAttribute(&rate, hipDeviceAttributeWallClockRate, 0));

    // ---- the pools, filled with each layer's most popular experts
    planner P;
    static char tensor_tags[LAYERS][3];
    const int per_tensor[3] = {125, 125, 156};
    for (int pool = 0; pool < 3; pool++) {
        const int n_tensors = pool == 0 ? 72 : pool == 1 ? 24 : 48;
        P.pools[pool].slots.resize((size_t) n_tensors * per_tensor[pool]);
    }
    int used[3] = {0, 0, 0};
    std::mt19937 rng(1234);
    std::vector<std::vector<int>> popularity(LAYERS, std::vector<int>(N_EXPERT));
    for (int l = 0; l < LAYERS; l++) {
        const int gu_pool = l % 4 == 3 ? 1 : 0;     // 12 IQ3_S layers, 36 IQ2_S
        P.layers.push_back({&tensor_tags[l][0], &tensor_tags[l][1], &tensor_tags[l][2], gu_pool});
        for (int e = 0; e < N_EXPERT; e++) popularity[l][e] = e;
        std::shuffle(popularity[l].begin(), popularity[l].end(), rng);
        for (int k = 0; k < 3; k++) {
            const int pool = k == 2 ? 2 : gu_pool;
            moe_cache_pool & p = P.pools[pool];
            for (int r = 0; r < per_tensor[pool]; r++) {
                const int s = used[pool]++;
                p.slots[s].key = {&tensor_tags[l][k], popularity[l][r]};
                p.slots[s].state = slot_state::valid;
                p.map[p.slots[s].key] = s;
                lru_push_back(p, s);
            }
        }
    }

    // ---- routing: 10 distinct experts per layer, Zipf (s = 1) over the popularity ranks
    std::vector<double> cdf(N_EXPERT);
    double acc = 0;
    for (int r = 0; r < N_EXPERT; r++) { acc += 1.0 / (r + 1); cdf[r] = acc; }
    for (double & c : cdf) c /= acc;
    std::uniform_real_distribution<double> u01(0, 1);
    std::vector<int32_t> ids((size_t) PASSES * LAYERS * TOPK);
    int resident_rows = 0;
    for (int pass = 0; pass < PASSES; pass++) {
        for (int l = 0; l < LAYERS; l++) {
            int32_t * row = &ids[((size_t) pass * LAYERS + l) * TOPK];
            for (int i = 0; i < TOPK; ) {
                const int rank = (int) (std::lower_bound(cdf.begin(), cdf.end(), u01(rng)) - cdf.begin());
                const int e = popularity[l][rank];
                if (std::find(row, row + i, e) != row + i) continue;
                row[i++] = e;
                resident_rows += rank < per_tensor[P.layers[l].pool] && rank < per_tensor[2];
            }
        }
    }
    printf("rows with their three tensors resident: %.1f %%\n", 100.0 * resident_rows / ids.size());

    // ---- device and mapped buffers
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    int32_t * d_ids; int * d_acc; long long * d_t;
    CHECK(hipMalloc(&d_ids, ids.size() * sizeof(int32_t)));
    CHECK(hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMalloc(&d_acc, sizeof(int)));
    CHECK(hipMalloc(&d_t, 2 * LAYERS * sizeof(long long)));
    int32_t * h_ids, * m_ids, * h_plan, * m_plan;
    uint32_t * h_flags, * m_flags;   // [0] A, [16] P (separate cache lines)
    CHECK(hipHostMalloc((void **) &h_ids, LAYERS * TOPK * sizeof(int32_t), hipHostMallocMapped | hipHostMallocCoherent));
    CHECK(hipHostMalloc((void **) &h_plan, LAYERS * PLAN_INTS * sizeof(int32_t), hipHostMallocMapped | hipHostMallocCoherent));
    CHECK(hipHostMalloc((void **) &h_flags, 128, hipHostMallocMapped | hipHostMallocCoherent));
    CHECK(hipHostGetDevicePointer((void **) &m_ids, h_ids, 0));
    CHECK(hipHostGetDevicePointer((void **) &m_plan, h_plan, 0));
    CHECK(hipHostGetDevicePointer((void **) &m_flags, h_flags, 0));

    // ---- one graph per pass (the ids of the pass differ), flags l + 1 per layer, reset per pass
    std::vector<double> lat, plan_us;
    for (int pass = 0; pass < PASSES; pass++) {
        hipGraph_t graph; hipGraphExec_t exec;
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        for (int l = 0; l < LAYERS; l++) {
            k_publish_ids<<<1, 1, 0, s>>>(d_ids + ((size_t) pass * LAYERS + l) * TOPK, m_ids + l * TOPK, m_flags,
                                          d_t + 2 * l, l + 1);
            k_wait_flag<<<1, 1, 0, s>>>(m_flags + 16, l + 1, d_t + 2 * l + 1);
            k_read_plan<<<1, 1, 0, s>>>(m_plan + l * PLAN_INTS, d_acc);
        }
        CHECK(hipStreamEndCapture(s, &graph));
        CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        __atomic_store_n(&h_flags[0], 0u, __ATOMIC_RELEASE);
        __atomic_store_n(&h_flags[16], 0u, __ATOMIC_RELEASE);
        std::atomic<bool> go{false};
        std::vector<double> pl(LAYERS);
        std::thread worker([&] {
            while (!go.load(std::memory_order_acquire)) {}
            for (int l = 0; l < LAYERS; l++) {
                while (__atomic_load_n(&h_flags[0], __ATOMIC_ACQUIRE) < (uint32_t) (l + 1)) {}
                const double t0 = now_us();
                P.plan(l, h_ids + l * TOPK, h_plan + l * PLAN_INTS);
                pl[l] = now_us() - t0;
                __atomic_store_n(&h_flags[16], (uint32_t) (l + 1), __ATOMIC_RELEASE);
            }
        });
        go.store(true, std::memory_order_release);
        CHECK(hipGraphLaunch(exec, s));
        const double t0 = now_us();
        while (hipStreamQuery(s) == hipErrorNotReady) {
            if (now_us() - t0 > 3e6) { printf("pass %d hangs\n", pass); _exit(2); }
        }
        worker.join();
        // the pins taken by the plan are released at the end of the node (moe_cache_end)
        for (auto & pool : P.pools) for (auto & slot : pool.slots) slot.readers = 0;

        std::vector<long long> t(2 * LAYERS);
        CHECK(hipMemcpy(t.data(), d_t, t.size() * sizeof(long long), hipMemcpyDeviceToHost));
        if (pass >= 5) {   // first passes warm up
            for (int l = 0; l < LAYERS; l++) {
                lat.push_back((t[2 * l + 1] - t[2 * l]) * 1000.0 / rate);
                plan_us.push_back(pl[l]);
            }
        }
        CHECK(hipGraphExecDestroy(exec));
        CHECK(hipGraphDestroy(graph));
    }
    auto pct = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[(size_t) (q * (v.size() - 1))]; };
    printf("flag A written -> flag P seen on the GPU: median %.1f us, p90 %.1f us, p99 %.1f us\n",
           pct(lat, 0.5), pct(lat, 0.9), pct(lat, 0.99));
    printf("host time in the plan:                    median %.2f us, p90 %.2f us, p99 %.2f us\n",
           pct(plan_us, 0.5), pct(plan_us, 0.9), pct(plan_us, 0.99));
    return 0;
}
