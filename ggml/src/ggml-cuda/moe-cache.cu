// MoE expert cache by leloch (github.com/leloch/llama.cpp, branch moe-cache-v2-pr;
// RFC ggml-org/llama.cpp#24528); soft-mode partial-eviction fit by giveen
// (TheTom/llama-cpp-turboquant#284). MIT.

#include "moe-cache.cuh"

#if defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

void ggml_moe_cache_register(const void * owner) {
    (void) owner;
}

int ggml_cuda_moe_hosted_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    (void) ctx; (void) cgraph; (void) i;
    return -1;
}

void ggml_cuda_moe_hosted_flush(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    (void) ctx; (void) node;
}

bool ggml_cuda_moe_hosted_wait(void * stream, void * event) {
    (void) stream; (void) event;
    return false;
}

#else

#include "common.cuh"
#include "topk-moe.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"

// EXL3 weights are not part of this tree: every path below that tests for them is dead.
static inline bool ggml_type_is_exl3(ggml_type) { return false; }

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <process.h>
#elif defined(__linux__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <sys/mman.h>
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)

static constexpr int    moe_cache_cc_forced_min               = 700;
static constexpr int    moe_cache_cc_ampere                   = 800;
static constexpr size_t moe_cache_expert_bytes_ampere_min     = 512u << 10;
static constexpr size_t moe_cache_expert_bytes_pre_ampere_min = 1u << 20;
static constexpr int    moe_cache_batch_max                   = 10;
static constexpr int    moe_cache_pool_slots_min              = 64;
static constexpr size_t moe_cache_slab_bytes_auto_min         = 1ull << 30;
static constexpr int    moe_cache_node_rows_max               = 64;
static constexpr int    moe_cache_expert_parallel_max         = 8;

enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

// An entry of the cache holds a whole expert: its up, gate and down tensors one after the other, each at a fixed offset
// (moe_cache_pool::role_offset), in the order the hosted MoE's plan gives their addresses. The hosted MoE computes a
// row on the GPU only with the three tensors of its expert; a pool per tensor shape, each evicting on its own, left
// 3 - 8 % of the cache to tensors of partial experts, useless until the missing ones came back (bench:
// tests/expertcache). One pool per expert shape (the three tensors' sizes and types).
enum { MOE_ROLE_UP = 0, MOE_ROLE_GATE = 1, MOE_ROLE_DOWN = 2, MOE_ROLES = 3 };
// each tensor starts at a multiple of this in its entry (the kernels read 16-byte vectors; the expert tensors of the
// supported types are multiples of 256 bytes, nothing is lost on them)
static constexpr size_t moe_cache_role_align = 256;

// an entry's key: the expert's layer, named by its gate tensor, and the expert
struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
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

struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    uint16_t freq = 0;   // LFU: aged demand count while resident
    moe_cache_slot_state state = moe_cache_slot_state::free;
};

// a run of entries lent to the outside
struct moe_cache_lent_block {
    int begin = 0;
    int count = 0;
};

// one contiguous piece of device memory holding logical slots [first, first + count)
struct moe_cache_range {
    char * base = nullptr;
    int first = 0;
    int count = 0;
    int64_t gpu_offset = 0;   // gpu index of slot 'first'
};

// KV: the tail of one KV tensor attached to a pool (moe_cache_kv_attach). Logical slots [first, first + count) in
// address order from 'start'; the first 'taken' of them are used by the KV (free, out of the free list, the LRU and
// the map), the others belong to the cache.
struct moe_cache_kv_range {
    const char * tensor = nullptr;   // the KV tensor's data
    size_t bytes = 0;                // its size
    size_t row_bytes = 0;            // bytes per cell
    char * start = nullptr;          // first entry boundary in it
    int first = 0;
    int count = 0;
    int taken = 0;
};

struct moe_cache_pool {
    size_t expert_size = 0;                // an entry: a whole expert
    size_t role_bytes[MOE_ROLES] = {};     // per-expert bytes of its up, gate and down tensors
    size_t role_offset[MOE_ROLES] = {};    // their offsets in the entry
    int wtype = -1;                        // the gate tensor's type
    char * slab = nullptr;
    int n_slots = 0;
    bool covers_all_entries = false;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    // the map again, as an array per host tensor indexed by the expert (-1: no entry), kept with it
    // (moe_cache_direct_set); the hosted plan reads it: one load per row instead of a hash chain
    std::unordered_map<const void *, std::vector<int32_t>> direct;
    int lru_head = -1;
    int lru_tail = -1;

    // Tail of the slab that can be lent to the scheduler's expert prefetch during
    // prefill. While lent, its slots are free, outside the free list, the LRU and the
    // map, so nothing in the cache can reach them.
    int zone_begin = INT_MAX;   // lendable zone, inside range 0
    int zone_end = 0;
    int zone_slice = 0;         // entries of one prefetch slot: the zone holds (zone_end - zone_begin) / zone_slice slots
    bool zone_lent = false;

    // runs of entries lent to the outside (compute buffer chunks), taken from the tail
    // of range 0, below the prefetch zone. While lent, their entries are free but out of
    // the free list, the LRU and the map.
    std::vector<moe_cache_lent_block> lent;
    int lent_low = INT_MAX;     // lowest entry lent; zone_begin when nothing is lent

    // Entries lending evicted during the last prefill, coldest first (the order they left
    // the LRU in). Restored hottest first once generation starts (moe_cache_rewarm_feed_locked).
    std::vector<moe_cache_key> rewarm;

    // Address translation. Logical slots are spread over one or more ranges of device
    // memory: range 0 is the slab, the others are borrowed (from the compute buffer).
    // Kernels only take one base and an index, so every range is addressed from a
    // virtual base, low enough and aligned so that each entry sits at
    // gpu_base + k * expert_size.
    std::vector<moe_cache_range> ranges;
    char * gpu_base = nullptr;
    int64_t gpu_extent = 0;     // largest gpu index + 1, passed to kernels as a bound

    std::vector<moe_cache_kv_range> kv;   // KV tails lent to this pool
};

static inline int32_t moe_cache_gpu_index(const moe_cache_pool & pool, int index) {
    for (const moe_cache_range & r : pool.ranges) {
        if (index >= r.first && index < r.first + r.count) {
            return (int32_t)(r.gpu_offset + (index - r.first));
        }
    }
    GGML_ABORT("moe-cache: slot %d outside every range", index);
}

static inline char * moe_cache_slot_ptr(const moe_cache_pool & pool, int index) {
    return pool.gpu_base + (size_t)moe_cache_gpu_index(pool, index) * pool.expert_size;
}

// the device address of an entry, as the kernels take it (ggml_cuda_moe_cache_mmv*: one address per routed row, 0 for a
// row the cache does not hold)
static inline uint64_t moe_cache_entry_addr(const moe_cache_pool & pool, int index) {
    return (uint64_t) (uintptr_t) moe_cache_slot_ptr(pool, index);
}

static inline bool moe_cache_pool_has_zone(const moe_cache_pool & pool) {
    return pool.zone_begin < pool.zone_end;
}

// an expert shape: the three tensors' per-expert sizes and types; experts of this shape (n_entries) and their layers
struct moe_cache_shape {
    size_t role_bytes[MOE_ROLES] = {};
    int role_type[MOE_ROLES] = {-1, -1, -1};
    size_t expert_size = 0;   // the entry (moe_cache_entry_layout)
    uint64_t n_entries = 0;
    int64_t n_layers = 0;
    int pool = -1;
    bool finished = false;
};

// a MoE layer as discovered: its three expert tensors (read from the tensor names: blk.N.ffn_{up,gate,down}_exps)
struct moe_cache_layer {
    const void * tensor[MOE_ROLES] = {};
    size_t bytes[MOE_ROLES] = {};    // per expert (nb[2])
    int type[MOE_ROLES] = {-1, -1, -1};
    int64_t n_expert = 0;
    int shape = -1;                  // once the pools are built
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source[MOE_ROLES] = {};   // the expert's three tensors in host memory
    size_t role_bytes[MOE_ROLES] = {};
    size_t bytes = 0;                      // the entry
    bool rewarm = false;   // an entry lending evicted, restored: joins the LRU at its cold end
    // SPLIT: a tensor of an expert moved to the GPU for the layer being computed; the last one of the expert
    // to arrive raises the layer's flag
    uint32_t * split_flag = nullptr;
    int * split_pending = nullptr;
    uint32_t split_seq = 0;
};

// SPLIT: one tensor of the moved expert done (copied, failed or cancelled): the last one raises the flag
static void moe_cache_split_done_locked(const moe_cache_job & job) {
    if (job.split_flag && --*job.split_pending == 0) {
        __atomic_store_n(job.split_flag, job.split_seq, __ATOMIC_RELEASE);
    }
}

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_config {
    bool enabled = true;
    bool automatic = true;
    int min_devices = 1;
    size_t budget_mb = 0;
    size_t reserve_mb = 3072;
    bool reserve_explicit = false;
    size_t minimum_slab_bytes = moe_cache_slab_bytes_auto_min;
    size_t min_expert_bytes = moe_cache_expert_bytes_pre_ampere_min;
    bool min_expert_explicit = false;
    int max_batch = moe_cache_batch_max;
    bool max_batch_explicit = false;
    // the admission rate, measured with whole-expert entries (replay bench, 3-token passes, 32k and 251k): the experts a
    // plan may admit (inserts_per_plan / 2) on a plateau from 2 to 16; demands before a free entry takes an expert 3
    // (-0.4 % at 32k, -0.2 % at 251k against 2: an admission is three copies now), before it evicts the coldest 8
    // (4: -0.5 % at 32k, +0.9 % at 251k; 2 and 16 worse)
    int inserts_per_plan = 8;
    bool inserts_per_plan_explicit = false;
    int admit_after = 3;
    bool admit_after_explicit = false;
    int readmit_after = 8;
    bool readmit_after_explicit = false;
    // LFU (buun, 766fa07b2): a full pool admits an expert only when its aged demand beats the least recently used
    // entry's; every count halves each lfu_window x entries demands, and is capped at lfu_cap.
    // GGML_CUDA_MOE_CACHE_LFU=0: readmit_after demands, counts never age
    int lfu = 1;
    int lfu_window = 5;
    int lfu_cap = 15;
    int queue_max = 128;
    size_t queue_mb = 512;
    int stats_every = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = moe_cache_cc_forced_min;
    bool serial_fill = true;
    bool serial_fill_explicit = false;
    bool force_dedicated_mmv = false;
    bool force_dedicated_down_mmv = false;
    int overlap_cpu_rows = -1;
    bool overlap_cpu_rows_explicit = false;
    int expert_parallel = 0;
    std::string fail_stage;
    std::string profile_path;
};

struct moe_cache_profile_key {
    uint64_t tensor = 0;
    int32_t expert = -1;

    bool operator==(const moe_cache_profile_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_profile_key_hash {
    size_t operator()(const moe_cache_profile_key & key) const {
        uint64_t value = key.tensor ^
            ((uint64_t)(uint32_t)key.expert * UINT64_C(0x9e3779b97f4a7c15));
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        return (size_t)(value ^ (value >> 31));
    }
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t input = 0;
    size_t q8 = 0;
    size_t out = 0;
};

// device-to-device moves when lending: issued then waited for at once, so they go to the null stream,
// whose hardware queue exists anyway. A stream of their own cost a queue of its own on ROCm (38.4 MiB
// of wave save area on a 7900 XT) for no overlap; every other stream here is non-blocking, so the null
// stream does not synchronize with them.
static constexpr cudaStream_t moe_cache_zone_stream = nullptr;

struct moe_cache_device {
    moe_cache_device(int logical, int physical) : logical(logical), physical(physical) {}

    int logical;
    int physical;
    // KV: the end of the KV tensors lent to the pools, attached once when they are built (moe_cache_attach_kv)
    bool kv_attached = false;
    int64_t kv_step_rows = 0;    // rows per step (a multiple of 256), 0: nothing lent
    int64_t kv_end_rows = 0;     // the KV's rows now, rounded up to a step (at least one step)
    int64_t kv_max_rows = 0;     // rows of the longest KV tensor (cells)
    int64_t kv_takes = 0, kv_gives = 0, kv_moves = 0, kv_evictions = 0;
    double kv_ms = 0;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    std::map<int, moe_cache_layer> layers;              // by layer number
    std::unordered_map<const void *, int> layer_of;      // each expert tensor -> its layer number
    bool pools_built = false;
    // the expert tensors registered in place (page-rounded ranges, one per tensor): fills copied straight from the
    // weights by the SDMA engine, without the memcpy to the stage
    std::vector<std::pair<uintptr_t, uintptr_t>> pinned;
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    uint64_t freq_demands = 0;   // LFU: demands since the last aging
    double freq_period = 0;      // LFU: demands between agings, fixed when first needed (a lent KV tail adds entries later)
    size_t visits_since_new_tensor = 0;
    bool budget_ready = false;
    bool budget_registered = false;
    bool budget_claimed = false;
    size_t automatic_reserve_bytes = 0;
    size_t budget_reserve_bytes = 0;
    size_t budget_limit = 0;
    size_t coordinator_allocated_bytes = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    // PREGATE (moe_pregate_k): prefetches of the experts predicted for the next layer, served only
    // in the window from a layer's CPU rows done to the next layer's publication (CPU idle, GPU on the rest
    // of the layer in VRAM), after the ordinary fills
    std::deque<moe_cache_job> pregate_queue;   // the candidate being reserved (at most one at a time)
    struct pregate_candidate {
        int pool = -1;
        const void * tensor = nullptr;
        int32_t expert = -1;
        size_t size = 0;
    };
    std::vector<pregate_candidate> pg_cands;      // for the next layer, best rank first
    size_t pg_next = 0;
    bool pregate_window = false;
    bool pg_busy = false;                         // a prefetch burst is copying
    int64_t pregate_enqueued = 0, pregate_served = 0, pregate_predicted = 0;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    const void * inflight_source[MOE_ROLES] = {};
    size_t inflight_bytes[MOE_ROLES] = {};
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    std::vector<char *> owned_regions;    // extra regions this device allocated itself (tests)

    // seating of the chunks ggml-alloc announced (moe_cache_plan_borrows): the pool of each
    // chunk size, nullptr once served
    std::vector<size_t> loan_plan_bytes;
    std::vector<moe_cache_pool *> loan_plan_pool;

    // restoring the entries lending evicted (pool.rewarm), from the start of generation
    // until a new prefill lends again
    bool rewarm_active = false;
    size_t rewarm_next_pool = 0;
    long long rewarm_restored = 0;
    char * h_input = nullptr;
    char * d_input = nullptr;
    size_t h_input_cap = 0;
    size_t d_input_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    // hits and misses count tensor-expert probes; fusion counters count row pairs.
    long long hits = 0;
    long long zone_lends = 0;
    long long zone_restores = 0;
    long long zone_moves = 0;
    long long zone_evictions = 0;
    long long zone_checked = 0;
    long long zone_check_failed = 0;
    double zone_lend_ms = 0.0;
    long long misses = 0;
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long activation_dedup = 0;
    long long overlap_rows = 0;
    long long fused_rows = 0;
    long long fused_candidates = 0;
    long long pair_both = 0;
    long long pair_up_only = 0;
    long long pair_gate_only = 0;
    long long pair_neither = 0;
    long long fused_attempts = 0;
    long long fused_nodes = 0;
    long long full_fused_rows = 0;
    long long full_fused_nodes = 0;
    long long dedicated_down_mmv_dispatches = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    // Dispatch mutex contention that turned a potential cache hit into a
    // complete CPU fallback (try_to_lock failed). Measured first so a bounded
    // wait or per-device FIFO can be justified with data.
    long long contention_bypasses = 0;
    std::atomic<int> error_logs{0};
};

struct moe_cache_session {
    moe_cache_config config;
    // prefill detection for the rewarm: the scheduler lends the prefetch zones in every
    // prefill graph and in no other, so the first graph after lending graphs that does
    // not lend is the start of generation
    bool lent_in_last_graph = false;
    bool lent_since_evictions = false;
    size_t lend_bytes = 0;   // size of one prefetch zone, set by the scheduler
    int lend_zones = 2;      // prefetch slots the scheduler wants, one zone per pool
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;
    // KV: tensors of the KV cache whose end may be lent (moe_cache_kv_register), attached when the pools are built
    struct kv_tensor { const char * data; size_t bytes; size_t row_bytes; int logical; };
    std::vector<kv_tensor> kv_tensors;
    std::unordered_map<moe_cache_profile_key, uint32_t,
        moe_cache_profile_key_hash> profile_heat;
    uint64_t profile_generation = 0;
    std::unordered_map<uint64_t, std::vector<int32_t>> profile_hot;
    std::unordered_map<uint64_t, std::vector<uint32_t>> observed_heat;

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    std::atomic<bool> config_announced{false};
    std::atomic<bool> enabled_announced{false};
    std::atomic<bool> batch_bypass_announced{false};
    std::atomic<bool> row_bypass_announced{false};
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;
};

struct moe_cache_pin {
    moe_cache_pool * pool = nullptr;
    int slot = -1;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};
struct moe_cache_physical_budget {
    int participants = 0;
    int prepared = 0;
    std::multiset<size_t> reserves;
    size_t outstanding_bytes = 0;
};
static std::mutex g_budget_mu;
static std::unordered_map<int, moe_cache_physical_budget> g_physical_budgets;
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;
static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);

static void moe_cache_budget_remove_participant(
        moe_cache_physical_budget & state, size_t reserve_bytes) {
    auto reserve = state.reserves.find(reserve_bytes);
    if (reserve != state.reserves.end()) {
        state.reserves.erase(reserve);
    }
    state.participants = std::max(state.participants - 1, 0);
}

static bool moe_cache_budget_register(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    std::vector<moe_cache_device *> registered;
    try {
        registered.reserve(session.devices.size());
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            moe_cache_physical_budget & state = g_physical_budgets[device.physical];
            const size_t reserve_bytes = ggml_moe_cache_effective_reserve_bytes(
                    session.config.reserve_explicit,
                    session.config.reserve_mb << 20,
                    device.automatic_reserve_bytes);
            state.reserves.insert(reserve_bytes);
            state.participants++;
            device.budget_registered = true;
            device.budget_reserve_bytes = reserve_bytes;
            registered.push_back(&device);
        }
        return true;
    } catch (...) {
        for (moe_cache_device * device : registered) {
            auto found = g_physical_budgets.find(device->physical);
            if (found != g_physical_budgets.end()) {
                moe_cache_budget_remove_participant(
                        found->second, device->budget_reserve_bytes);
                if (found->second.participants == 0) {
                    g_physical_budgets.erase(found);
                }
            }
            device->budget_registered = false;
            device->budget_reserve_bytes = 0;
        }
        for (auto found = g_physical_budgets.begin();
             found != g_physical_budgets.end();) {
            if (found->second.participants == 0) {
                found = g_physical_budgets.erase(found);
            } else {
                ++found;
            }
        }
        return false;
    }
}

static void moe_cache_budget_allocation(
        moe_cache_device & device, size_t bytes, bool allocated) {
    if (!device.budget_registered || !device.budget_claimed || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }
    moe_cache_physical_budget & state = found->second;
    if (allocated) {
        const size_t remaining = device.budget_limit > device.coordinator_allocated_bytes
            ? device.budget_limit - device.coordinator_allocated_bytes : 0;
        const size_t tracked = std::min(bytes, remaining);
        device.coordinator_allocated_bytes += tracked;
        state.outstanding_bytes = tracked <= state.outstanding_bytes
            ? state.outstanding_bytes - tracked : 0;
    } else {
        const size_t tracked = std::min(bytes, device.coordinator_allocated_bytes);
        device.coordinator_allocated_bytes -= tracked;
        if (tracked <= SIZE_MAX - state.outstanding_bytes) {
            state.outstanding_bytes += tracked;
        } else {
            state.outstanding_bytes = SIZE_MAX;
        }
    }
}

static void moe_cache_budget_reallocation(
        moe_cache_device & device, size_t old_bytes, size_t new_bytes) {
    if (!device.budget_registered || !device.budget_claimed || old_bytes == new_bytes) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }

    moe_cache_physical_budget & state = found->second;
    const size_t tracked_old = std::min(old_bytes, device.coordinator_allocated_bytes);
    const size_t base = device.coordinator_allocated_bytes - tracked_old;
    const size_t remaining = device.budget_limit > base
        ? device.budget_limit - base : 0;
    const size_t tracked_new = std::min(new_bytes, remaining);
    device.coordinator_allocated_bytes = base + tracked_new;

    if (tracked_new >= tracked_old) {
        const size_t consumed = tracked_new - tracked_old;
        state.outstanding_bytes = consumed <= state.outstanding_bytes
            ? state.outstanding_bytes - consumed : 0;
    } else {
        const size_t released = tracked_old - tracked_new;
        state.outstanding_bytes = released <= SIZE_MAX - state.outstanding_bytes
            ? state.outstanding_bytes + released : SIZE_MAX;
    }
}

static void moe_cache_budget_unregister(moe_cache_device & device) {
    if (!device.budget_registered) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found != g_physical_budgets.end()) {
        moe_cache_physical_budget & state = found->second;
        if (device.budget_claimed) {
            const size_t unallocated = device.budget_limit > device.coordinator_allocated_bytes
                ? device.budget_limit - device.coordinator_allocated_bytes : 0;
            state.outstanding_bytes = unallocated <= state.outstanding_bytes
                ? state.outstanding_bytes - unallocated : 0;
            state.prepared = std::max(state.prepared - 1, 0);
        }
        moe_cache_budget_remove_participant(
                state, device.budget_reserve_bytes);
        if (state.participants == 0) {
            g_physical_budgets.erase(found);
        }
    }
    device.budget_registered = false;
    device.budget_claimed = false;
    device.budget_ready = false;
    device.budget_reserve_bytes = 0;
    device.budget_limit = 0;
    device.coordinator_allocated_bytes = 0;
}

static size_t moe_cache_budget_claim(
        moe_cache_session & session, moe_cache_device & device,
        size_t free_memory, size_t & reserve_bytes) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    reserve_bytes = 0;
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end() || !device.budget_registered) {
        return 0;
    }
    moe_cache_physical_budget & state = found->second;
    reserve_bytes = state.reserves.empty()
        ? 0 : *state.reserves.rbegin();
    const size_t available_after_reserve = free_memory > reserve_bytes
        ? free_memory - reserve_bytes : 0;
    const size_t unclaimed = available_after_reserve > state.outstanding_bytes
        ? available_after_reserve - state.outstanding_bytes : 0;
    const int remaining_participants = std::max(
            state.participants - state.prepared, 1);
    size_t claim = unclaimed / (size_t)remaining_participants;
    if (session.config.budget_mb > 0) {
        claim = std::min(claim, session.config.budget_mb << 20);
    }
    state.prepared++;
    state.outstanding_bytes += claim;
    device.budget_claimed = true;
    return claim;
}

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static int moe_cache_min_compute_capability() {
    // Both modes use the same SM70-capable kernels. Mode changes budgeting and
    // admission policy, not the minimum architecture supported by execution.
    int result = moe_cache_cc_forced_min;
    int64_t value = 0;
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        result = (int)value;
    }
    return result;
}

static size_t moe_cache_default_min_expert_bytes(int compute_capability) {
    return compute_capability >= moe_cache_cc_ampere
        ? moe_cache_expert_bytes_ampere_min
        : moe_cache_expert_bytes_pre_ampere_min;
}

static void moe_cache_apply_mode_defaults(moe_cache_config & config) {
    config.minimum_slab_bytes = config.automatic
        ? moe_cache_slab_bytes_auto_min : 0;
    if (!config.min_expert_explicit) {
        config.min_expert_bytes = config.automatic
            ? moe_cache_expert_bytes_ampere_min
            : moe_cache_expert_bytes_pre_ampere_min;
    }
    if (!config.max_batch_explicit) {
        config.max_batch = moe_cache_batch_max;
    }
    if (!config.overlap_cpu_rows_explicit) {
        config.overlap_cpu_rows = -1;
    }
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
        config.reserve_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
        config.min_expert_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, moe_cache_batch_max, value)) {
        config.max_batch = (int)value;
        config.max_batch_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
        config.inserts_per_plan_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
        config.admit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
        config.readmit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_LFU", 0, 1, value)) {
        config.lfu = (int) value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_LFU_WINDOW", 1, 1000, value)) {
        config.lfu_window = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_LFU_CAP", 2, 65535, value)) {
        config.lfu_cap = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
        config.serial_fill_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DOWN_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_down_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", 0, 8, value)) {
        config.overlap_cpu_rows = (int)value;
        config.overlap_cpu_rows_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_EXPERT_PARALLEL",
            -1, moe_cache_expert_parallel_max, value)) {
        config.expert_parallel = (int)value;
    }
    moe_cache_apply_mode_defaults(config);
    config.min_compute_capability = moe_cache_min_compute_capability();
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    return config;
}

// device memory kept out of the cache: what is allocated after it is sized, which depends on the model and the batch,
// not on the card's size (the cache itself takes the free memory minus this). Measured (11 Oct., Qwen3.8-Flash-Next,
// 256k, MTP, 8192-token ubatches, LD_PRELOAD tools/kfdalloc): ~340 MiB at the peak, of which the flash attention
// scratch of two queues (2 x 44.6 MiB, the D=256 prefill kernel spills on RDNA3), the CWSR areas of two queues created
// later (2 x 38.4), rocBLAS (12.7), kernel code (~20), ~140 outside the KFD allocations (rings, page tables); the
// ggml-cuda pool no longer grows (temporaries in the compute buffer). Peak at 824: 19.435 GiB, at 600: 19.652, the limit
// before spilling to GTT 19.91 on a 20 GiB card: 700 keeps ~360 MiB of margin. Also the default margin of --fit, which
// sets the reserve when it runs.
static size_t moe_cache_recommended_reserve_bytes(size_t total_vram) {
    constexpr size_t MiB = 1024 * 1024;
    return std::min<size_t>(700 * MiB, total_vram / 4);
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static uint64_t moe_cache_profile_tensor_hash(
        const char * name, size_t expert_size, int wtype, int64_t n_expert) {
    uint64_t hash = moe_cache_name_hash(name);
    const uint64_t fields[] = {
        (uint64_t)expert_size, (uint64_t)(uint32_t)wtype, (uint64_t)n_expert,
    };
    for (uint64_t field : fields) {
        hash ^= field;
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static uint32_t moe_cache_profile_u32(const unsigned char * data) {
    return uint32_t(data[0]) | uint32_t(data[1]) << 8 |
        uint32_t(data[2]) << 16 | uint32_t(data[3]) << 24;
}

static uint64_t moe_cache_profile_u64(const unsigned char * data) {
    return uint64_t(moe_cache_profile_u32(data)) |
        uint64_t(moe_cache_profile_u32(data + 4)) << 32;
}

static void moe_cache_profile_put_u32(std::vector<unsigned char> & out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back((unsigned char)(value >> shift));
    }
}

static void moe_cache_profile_put_u64(std::vector<unsigned char> & out, uint64_t value) {
    moe_cache_profile_put_u32(out, (uint32_t)value);
    moe_cache_profile_put_u32(out, (uint32_t)(value >> 32));
}

static uint64_t moe_cache_profile_checksum(const unsigned char * data, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static void moe_cache_profile_load(moe_cache_session & session, bool announce = true) {
    if (session.config.profile_path.empty()) {
        return;
    }
    std::ifstream input(session.config.profile_path, std::ios::binary | std::ios::ate);
    if (!input) {
        return;
    }
    const std::streamoff end = input.tellg();
    constexpr size_t header_v1_size = 24;
    constexpr size_t header_v2_size = 32;
    constexpr size_t record_size = 16;
    constexpr uint32_t max_records = 1u << 20;
    if (end < (std::streamoff)header_v1_size || end > (std::streamoff)(header_v2_size +
            (size_t)max_records * record_size)) {
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid expert profile size: %s\n",
                    session.config.profile_path.c_str());
        }
        return;
    }
    std::vector<unsigned char> bytes((size_t)end);
    input.seekg(0);
    if (!input.read((char *)bytes.data(), bytes.size())) {
        return;
    }
    static constexpr unsigned char magic[8] = {'G','G','M','L','M','H','C','1'};
    const uint32_t version = moe_cache_profile_u32(bytes.data() + 8);
    const uint32_t count = moe_cache_profile_u32(bytes.data() + 12);
    const size_t header_size = version == 1 ? header_v1_size :
        version == 2 ? header_v2_size : 0;
    const uint64_t generation = version == 2 && bytes.size() >= header_v2_size
        ? moe_cache_profile_u64(bytes.data() + 16) : 0;
    const uint64_t checksum = header_size != 0 && bytes.size() >= header_size
        ? moe_cache_profile_u64(bytes.data() + (version == 2 ? 24 : 16)) : 0;
    if (memcmp(bytes.data(), magic, sizeof(magic)) != 0 || header_size == 0 ||
            count > max_records || bytes.size() != header_size + (size_t)count * record_size ||
            checksum != moe_cache_profile_checksum(bytes.data() + header_size,
                bytes.size() - header_size)) {
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] ignoring corrupt or incompatible expert profile: %s\n",
                    session.config.profile_path.c_str());
        }
        return;
    }
    try {
        session.profile_generation = generation;
        session.profile_heat.reserve(count);
        const unsigned char * record = bytes.data() + header_size;
        for (uint32_t i = 0; i < count; ++i, record += record_size) {
            const moe_cache_profile_key key{
                moe_cache_profile_u64(record),
                (int32_t)moe_cache_profile_u32(record + 8),
            };
            const uint32_t heat = moe_cache_profile_u32(record + 12);
            if (key.tensor != 0 && key.expert >= 0 && heat > 0) {
                session.profile_heat[key] = heat;
            }
        }
        std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, int32_t>>> ranked;
        for (const auto & item : session.profile_heat) {
            ranked[item.first.tensor].push_back({item.second, item.first.expert});
        }
        session.profile_hot.reserve(ranked.size());
        for (auto & tensor : ranked) {
            std::sort(tensor.second.begin(), tensor.second.end(), [](const auto & a, const auto & b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            if (tensor.second.size() > 32) {
                tensor.second.resize(32);
            }
            std::vector<int32_t> & hot = session.profile_hot[tensor.first];
            hot.reserve(tensor.second.size());
            for (const auto & item : tensor.second) {
                hot.push_back(item.second);
            }
        }
        if (announce) {
            MOE_CACHE_LOG("[moe-cache] loaded %zu expert heat entries from %s\n",
                    session.profile_heat.size(), session.config.profile_path.c_str());
        }
    } catch (...) {
        session.profile_heat.clear();
        session.profile_hot.clear();
        session.profile_generation = 0;
    }
}

static std::mutex g_profile_io_mu;

struct moe_cache_profile_file_lock {
    bool locked = false;
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
    int fd = -1;
#endif

    explicit moe_cache_profile_file_lock(const std::string & path) {
        const std::string lock_path = path + ".lock";
#if defined(_WIN32)
        handle = CreateFileA(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle != INVALID_HANDLE_VALUE) {
            OVERLAPPED overlap = {};
            locked = LockFileEx(handle, LOCKFILE_EXCLUSIVE_LOCK, 0,
                    MAXDWORD, MAXDWORD, &overlap) != 0;
        }
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0600);
        locked = fd >= 0 && flock(fd, LOCK_EX) == 0;
#else
        locked = true;
#endif
    }

    ~moe_cache_profile_file_lock() {
#if defined(_WIN32)
        if (handle != INVALID_HANDLE_VALUE) {
            if (locked) {
                OVERLAPPED overlap = {};
                UnlockFileEx(handle, 0, MAXDWORD, MAXDWORD, &overlap);
            }
            CloseHandle(handle);
        }
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        if (fd >= 0) {
            if (locked) {
                flock(fd, LOCK_UN);
            }
            close(fd);
        }
#endif
    }
};

static void moe_cache_profile_save(moe_cache_session & session) {
    if (session.config.profile_path.empty() || session.observed_heat.empty()) {
        return;
    }
    try {
        std::lock_guard<std::mutex> process_lock(g_profile_io_mu);
        moe_cache_profile_file_lock file_lock(session.config.profile_path);
        if (!file_lock.locked) {
            return;
        }
        moe_cache_session latest;
        latest.config.profile_path = session.config.profile_path;
        moe_cache_profile_load(latest, false);
        auto & merged_heat = latest.profile_heat;
        if (latest.profile_generation == session.profile_generation) {
            for (auto it = merged_heat.begin(); it != merged_heat.end();) {
                it->second /= 2;
                if (it->second == 0) {
                    it = merged_heat.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (const auto & tensor : session.observed_heat) {
            for (size_t expert = 0; expert < tensor.second.size(); ++expert) {
                const uint32_t observed = tensor.second[expert];
                if (observed == 0 || expert > INT32_MAX) {
                    continue;
                }
                uint32_t & heat = merged_heat[
                    {tensor.first, (int32_t)expert}];
                const uint64_t merged = uint64_t(heat) + observed;
                heat = (uint32_t)std::min<uint64_t>(merged, UINT32_MAX);
            }
        }
        std::vector<std::pair<moe_cache_profile_key, uint32_t>> records(
                merged_heat.begin(), merged_heat.end());
        std::sort(records.begin(), records.end(), [](const auto & a, const auto & b) {
            return a.first.tensor < b.first.tensor ||
                (a.first.tensor == b.first.tensor && a.first.expert < b.first.expert);
        });
        if (records.size() > (1u << 20)) {
            records.resize(1u << 20);
        }
        std::vector<unsigned char> payload;
        payload.reserve(records.size() * 16);
        for (const auto & record : records) {
            moe_cache_profile_put_u64(payload, record.first.tensor);
            moe_cache_profile_put_u32(payload, (uint32_t)record.first.expert);
            moe_cache_profile_put_u32(payload, record.second);
        }
        std::vector<unsigned char> bytes;
        bytes.insert(bytes.end(), {'G','G','M','L','M','H','C','1'});
        moe_cache_profile_put_u32(bytes, 2);
        moe_cache_profile_put_u32(bytes, (uint32_t)records.size());
        moe_cache_profile_put_u64(bytes, latest.profile_generation + 1);
        moe_cache_profile_put_u64(bytes,
                moe_cache_profile_checksum(payload.data(), payload.size()));
        bytes.insert(bytes.end(), payload.begin(), payload.end());

        char suffix[80];
#if defined(_WIN32)
        const long process_id = _getpid();
#elif defined(__linux__) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__DragonFly__)
        const long process_id = (long)getpid();
#else
        const long process_id = 0;
#endif
        snprintf(suffix, sizeof(suffix), ".tmp-%ld-%p", process_id, (void *)&session);
        const std::string temporary = session.config.profile_path + suffix;
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output.write((const char *)bytes.data(), bytes.size())) {
            output.close();
            std::remove(temporary.c_str());
            return;
        }
        output.close();
#ifdef _WIN32
        const bool replaced = MoveFileExA(temporary.c_str(),
                session.config.profile_path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        const bool replaced = std::rename(
                temporary.c_str(), session.config.profile_path.c_str()) == 0;
#endif
        if (replaced) {
            MOE_CACHE_LOG("[moe-cache] saved %zu expert heat entries to %s\n",
                    records.size(), session.config.profile_path.c_str());
        } else {
            std::remove(temporary.c_str());
        }
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] could not update expert profile: %s\n",
                session.config.profile_path.c_str());
    }
}




static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static bool moe_cache_tensor_name_supported(const char * name) {
    return strstr(name, "_exps") || strstr(name, "_chexps");
}

static bool moe_cache_type_supported(ggml_type type) {
    if (ggml_type_is_exl3(type)) return true;
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static void moe_cache_lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) {
        pool.slots[slot.prev].next = slot.next;
    } else {
        pool.lru_head = slot.next;
    }
    if (slot.next >= 0) {
        pool.slots[slot.next].prev = slot.prev;
    } else {
        pool.lru_tail = slot.prev;
    }
    slot.prev = -1;
    slot.next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = pool.lru_tail;
    slot.next = -1;
    if (pool.lru_tail >= 0) {
        pool.slots[pool.lru_tail].next = index;
    } else {
        pool.lru_head = index;
    }
    pool.lru_tail = index;
}

static void moe_cache_lru_push_front(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = -1;
    slot.next = pool.lru_head;
    if (pool.lru_head >= 0) {
        pool.slots[pool.lru_head].prev = index;
    } else {
        pool.lru_tail = index;
    }
    pool.lru_head = index;
}

static void moe_cache_direct_set(moe_cache_pool & pool, const moe_cache_key & key, int index) {
    std::vector<int32_t> & d = pool.direct[key.tensor];
    if ((int) d.size() <= key.expert) {
        d.resize(key.expert + 1, -1);
    }
    d[key.expert] = index;
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
        moe_cache_direct_set(pool, slot.key, -1);
    }
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.state == moe_cache_slot_state::valid) {
        moe_cache_lru_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    slot.key = {};
    slot.generation++;
    slot.readers = 0;
    slot.freq = 0;
    slot.state = moe_cache_slot_state::free;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
}


static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = moe_cache_node_rows_max;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t ids_bytes = max_rows * (3 * sizeof(uint64_t) + sizeof(int32_t));   // up, gate, down entries; activation map
    const size_t act_bytes = max_rows * (size_t)n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return false;
    }
    const size_t capacities[] = {
        moe_cache_growth_capacity(0, ids_bytes + act_bytes),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.input, additional->input) : current.input,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static int moe_cache_query_config(
        int automatic, size_t budget_mib, ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability();
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->reserve_explicit = config.reserve_explicit;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->automatic = config.automatic;
    result->min_devices = config.min_devices;
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    result->expert_parallel = config.expert_parallel;
    result->profile_path = nullptr;
    return 1;
}

static int moe_cache_query_device(
        void * opaque, const ggml_moe_cache_config * config,
        ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !ggml_moe_cache.owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != ggml_moe_cache.owner) {
        return 0;
    }

    int logical = -1;
    const size_t count = ggml_backend_reg_dev_count(reg);
    for (size_t index = 0; index < count; index++) {
        if (ggml_backend_reg_dev_get(reg, index) == device) {
            logical = (int)index;
            break;
        }
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (logical < 0 || logical >= info.device_count) {
        return 0;
    }

    const auto & cuda_device = info.devices[logical];
    if (cuda_device.cc < config->min_compute_capability) {
        return 0;
    }
    result->logical_device = logical;
    result->physical_device = cuda_device.physical_device;
    result->compute_capability = cuda_device.cc;
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(cuda_device.cc);
    result->recommended_reserve_bytes = ggml_moe_cache_effective_reserve_bytes(
            config->reserve_explicit,
            config->reserve_bytes,
            moe_cache_recommended_reserve_bytes(cuda_device.total_vram));
    return 1;
}

static int moe_cache_query_shape(
        int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
        size_t expert_size,
        ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        (ggml_type_is_exl3((ggml_type)wtype) && (n_in % 128 || n_out % 128)) ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size != (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    moe_cache_scratch scratch;
    if (!moe_cache_scratch_requirements(n_in, n_out, scratch)) {
        return 0;
    }
    const size_t scratch_bytes = moe_cache_scratch_total(scratch);
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (scratch_bytes == SIZE_MAX || pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}


static void moe_cache_rewarm_feed_locked(moe_cache_session & session, moe_cache_device & device);
static bool moe_cache_enqueue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker, bool allow_eviction, bool rewarm, bool pregate, uint16_t freq = 0, bool contest = false);
static void moe_cache_rewarm_stop_locked(moe_cache_device & device);

// PREGATE: experts predicted per token for the next layer of a hosted pass of at least 2 tokens (MTP
// verification); GGML_CUDA_MOE_PREGATE=k enables it, top-k predictions (6 was the best: top-4 and 8, a threshold of
// 1 or 3 predictions, all the layers, measured below it). Off by default: it gained 3 - 5 % on 4 Oct.; since then
// (larger cache with the KV tail lent, faster GPU kernels, the dispatcher, whole-expert entries) it costs 1.7 % at
// 32k and 4 - 5 % at 251k (replay bench, 3-token passes), whatever was tried: its router on a side stream (6 of its
// ~18 us of GPU work off the critical path), its experts at the cold end of the LRU (worse), without the dispatcher
// (still -2.5 %). Not explained: perhaps its copies taking RAM bandwidth from the CPU rows (IQ2_S experts on 8 threads
// are bound by it). To be measured again with PCIe 4.0 and faster RAM.
static int moe_pregate_k() {
    static const int k = getenv("GGML_CUDA_MOE_PREGATE") ? atoi(getenv("GGML_CUDA_MOE_PREGATE")) : 0;
    return k;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    cudaStream_t stream = nullptr;
    char * stage2 = nullptr;          // PREGATE: the second buffer of the prefetch pipeline
    size_t stage2_capacity = 0;

    const auto grow = [](char *& buf, size_t & cap, size_t bytes) {
        if (cap >= bytes) {
            return true;
        }
        char * fresh = nullptr;
        if (cudaMallocHost((void **) &fresh, bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        if (buf) {
            cudaFreeHost(buf);
        }
        buf = fresh;
        cap = bytes;
        return true;
    };
    // whether a source lies in a registered expert tensor (copied straight from it)
    const auto pinned = [&](const void * source, size_t bytes) {
        const uintptr_t src = (uintptr_t) source;
        auto range = std::upper_bound(device->pinned.begin(), device->pinned.end(), std::make_pair(src, UINTPTR_MAX));
        return range != device->pinned.begin() && src + bytes <= (--range)->second;
    };
    // the tensors of a job not registered in place, to the staging buffer at their offsets in the entry
    const auto stage_job = [&](const moe_cache_job & j, const moe_cache_pool * p, char * buf) {
        for (int r = 0; r < MOE_ROLES; r++) {
            if (!pinned(j.source[r], j.role_bytes[r])) {
                memcpy(buf + p->role_offset[r], j.source[r], j.role_bytes[r]);
            }
        }
    };
    // the job's three copies, each from the weights or from the staging buffer
    const auto copy_job = [&](const moe_cache_job & j, const moe_cache_pool * p, char * dst, const char * buf) {
        cudaError_t err = cudaSuccess;
        for (int r = 0; r < MOE_ROLES && err == cudaSuccess; r++) {
            const void * src = pinned(j.source[r], j.role_bytes[r]) ? j.source[r] : (const void *) (buf + p->role_offset[r]);
            err = cudaMemcpyAsync(dst + p->role_offset[r], src, j.role_bytes[r], cudaMemcpyHostToDevice, stream);
        }
        return err;
    };
    // the entry a job fills (session lock taken)
    // Both staging buffers sized once to the largest expert of the pools: freeing pinned memory
    // synchronizes every stream (each gives its SDMA engine back) and, once generation runs, the fill
    // stream could lose its engine to the hosted MoE or draft streams for good.
    const auto ensure_stages = [&](size_t bytes) {
        size_t need = bytes;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            for (auto & p : device->pools) {
                need = std::max(need, p->expert_size);
            }
        }
        if (!grow(stage, stage_capacity, need) || (moe_pregate_k() > 0 && !grow(stage2, stage2_capacity, need))) {
            (void)cudaGetLastError();
        }
    };
    const auto destination_of = [&](const moe_cache_job & j, moe_cache_pool *& pool) -> char * {
        std::lock_guard<std::mutex> lock(session->mu);
        pool = nullptr;
        if (j.pool >= 0 && j.pool < (int) device->pools.size()) {
            pool = device->pools[j.pool].get();
            if (pool->slab && j.slot >= 0 && j.slot < pool->n_slots) {
                return moe_cache_slot_ptr(*pool, j.slot);
            }
        }
        return nullptr;
    };
    // a copy done: the entry becomes valid (or is released), as for an ordinary fill
    const auto complete = [&](const moe_cache_job & j, moe_cache_pool * pool, cudaError_t error) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (pool && j.slot >= 0 && j.slot < pool->n_slots) {
            moe_cache_slot & slot = pool->slots[j.slot];
            if (slot.state == moe_cache_slot_state::copying && slot.generation == j.generation && slot.key == j.key) {
                if (error == cudaSuccess) {
                    slot.state = moe_cache_slot_state::valid;
                    moe_cache_lru_push_back(*pool, j.slot);
                    device->fills++;
                } else {
                    moe_cache_slot_reset(*pool, j.slot, true);
                    device->fill_failures++;
                }
            }
        }
        session->idle_cv.notify_all();
    };
    // the best remaining candidate still absent, its entry reserved now (session lock held)
    const auto take_candidate = [&](moe_cache_job & out) -> bool {
        while (device->pg_next < device->pg_cands.size()) {
            const auto c = device->pg_cands[device->pg_next++];
            if (c.pool < 0 || c.pool >= (int) device->pools.size()) {
                continue;
            }
            moe_cache_pool & pool = *device->pools[c.pool];
            if (pool.map.count(moe_cache_key{c.tensor, c.expert})) {
                continue;
            }
            int inserts_left = 1;
            bool wake = false;
            if (moe_cache_enqueue_locked(*session, *device, pool, c.pool, c.tensor, c.expert, c.size, inserts_left, wake,
                                         true, false, true)) {
                out = device->pregate_queue.back();
                device->pregate_queue.pop_back();
                device->queued_bytes = out.bytes <= device->queued_bytes ? device->queued_bytes - out.bytes : 0;
                device->pregate_served++;
                return true;
            }
        }
        return false;
    };
    // PREGATE: while the window is open and no ordinary fill waits, the candidates in rank order; the
    // memcpy of the next one into the other buffer overlaps the copy of the current one
    const auto pregate_burst = [&]() {
        moe_cache_job cur;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (!device->pregate_window || !device->queue.empty() || session->stopping || !take_candidate(cur)) {
                return;
            }
            device->pg_busy = true;
        }
        ggml_cuda_set_device(device->logical);
        cudaError_t error = cudaSuccess;
        if (!stream) {
            error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }
        char * bufs[2];
        size_t * caps[2] = {&stage_capacity, &stage2_capacity};
        int b = 0;
        moe_cache_pool * cur_pool = nullptr;
        char * cur_dst = destination_of(cur, cur_pool);
        if (stage_capacity < cur.bytes || stage2_capacity < cur.bytes) {
            ensure_stages(cur.bytes);
        }
        if (error != cudaSuccess || !cur_dst || stage_capacity < cur.bytes) {
            error = error != cudaSuccess ? error : cudaErrorInvalidValue;
        } else {
            bufs[0] = stage;
            stage_job(cur, cur_pool, bufs[b]);
            error = copy_job(cur, cur_pool, cur_dst, bufs[b]);
        }
        for (;;) {
            moe_cache_job next;
            bool have = false;
            if (error == cudaSuccess) {
                std::lock_guard<std::mutex> lock(session->mu);
                have = device->pregate_window && device->queue.empty() && !session->stopping && take_candidate(next);
            }
            moe_cache_pool * next_pool = nullptr;
            char * next_dst = nullptr;
            if (have) {
                next_dst = destination_of(next, next_pool);
                char *& nb = b == 0 ? stage2 : stage;
                if (!next_dst || *caps[b ^ 1] < next.bytes) {
                    complete(next, next_pool, cudaErrorInvalidValue);
                    have = false;
                } else {
                    bufs[b ^ 1] = nb;
                    stage_job(next, next_pool, bufs[b ^ 1]);
                }
            }
            if (error == cudaSuccess) {
                while ((error = cudaStreamQuery(stream)) == cudaErrorNotReady) {
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                }
            }
            complete(cur, cur_pool, error);
            if (!have) {
                break;
            }
            if (error != cudaSuccess) {
                complete(next, next_pool, error);
                break;
            }
            b ^= 1;
            cur = next;
            cur_pool = next_pool;
            cur_dst = next_dst;
            error = copy_job(cur, cur_pool, cur_dst, bufs[b]);
        }
        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->pg_busy = false;
            session->idle_cv.notify_all();
        }
        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert prefetch", true);
            moe_cache_trim_session(*session, device->physical);
        }
    };

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty() || (device->pregate_window && device->pg_next < device->pg_cands.size());
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            if (device->queue.empty()) {
                // no ordinary fill waits: the next layer's prefetches, while the window is open
                lock.unlock();
                pregate_burst();
                continue;
            }
            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            for (int r = 0; r < MOE_ROLES; r++) {
                device->inflight_source[r] = job.source[r];
                device->inflight_bytes[r] = job.role_bytes[r];
            }
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->logical);

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream) {
            int least_priority = 0;
            int greatest_priority = 0;
            error = cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority);
            if (error == cudaSuccess) {
                error = cudaStreamCreateWithPriority(
                        &stream, cudaStreamNonBlocking, least_priority);
            } else {
                (void)cudaGetLastError();
                error = cudaStreamCreateWithFlags(
                        &stream, cudaStreamNonBlocking);
            }
        }
        if (error == cudaSuccess && stage_capacity < job.bytes) {
            ensure_stages(job.bytes);
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = moe_cache_slot_ptr(*pool, job.slot);
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                // DIAGNOSTIC (GGML_CUDA_MOE_FILL_STATS=1): per 2000 fills, time in the copy to the staging
                // buffer, in the copy to the device and its wait (of which the call), between fills
                static const bool ft = getenv("GGML_CUDA_MOE_FILL_STATS") != nullptr;
                static int64_t f_n = 0, f_cpy = 0, f_dma = 0, f_idle = 0, f_last = 0;
                const int64_t t0 = ft ? ggml_time_us() : 0;
                stage_job(job, pool, stage);
                const int64_t t1 = ft ? ggml_time_us() : 0;
                error = copy_job(job, pool, destination, stage);
                static int64_t f_call = 0;
                const int64_t t1b = ft ? ggml_time_us() : 0;
                if (error == cudaSuccess) {
                    // Poll, do not synchronize. ROCclr (7.2) gives a stream an SDMA engine of its own
                    // at its first copy and takes it back only when the stream is synchronized; a
                    // stream that finds both engines of the card taken falls back to ROCr's shared
                    // copy path and queues behind their copies. The hosted MoE blocks and the MTP
                    // draft poll their streams, which keep an engine each: a synchronized fill
                    // stream lost its own at every fill and waited ~1.2 ms behind a copy of the main
                    // stream (itself waiting on a kernel) for a fill of ~60 us. Polled, it keeps its
                    // engine; a fill lasts tens of microseconds, this thread has nothing else to do.
                    while ((error = cudaStreamQuery(stream)) == cudaErrorNotReady) {
#if defined(__x86_64__)
                        __builtin_ia32_pause();
#endif
                    }
                }
                if (ft) {
                    const int64_t t2 = ggml_time_us();
                    f_call += t1b - t1;
                    f_cpy += t1 - t0; f_dma += t2 - t1;
                    if (f_last > 0 && t0 - f_last < 5000) f_idle += t0 - f_last;
                    f_last = t2;
                    if (++f_n % 2000 == 0) {
                        GGML_LOG_INFO("fill: %lld fills, per fill: memcpy %.1f us, copy+sync %.1f us (call %.1f), between fills %.1f us\n",
                                (long long) f_n, (double) f_cpy / 2000, (double) f_dma / 2000, (double) f_call / 2000, (double) f_idle / 2000);
                        f_cpy = f_dma = f_idle = f_call = 0;
                    }
                }
            } else if (error == cudaSuccess) {
                for (int r = 0; r < MOE_ROLES && error == cudaSuccess; r++) {
                    error = cudaMemcpy(destination + pool->role_offset[r], job.source[r], job.role_bytes[r],
                                       cudaMemcpyHostToDevice);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            device->inflight = false;
            for (int r = 0; r < MOE_ROLES; r++) {
                device->inflight_source[r] = nullptr;
                device->inflight_bytes[r] = 0;
            }

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (error == cudaSuccess) {
                        slot.state = moe_cache_slot_state::valid;
                        if (job.rewarm) {
                            // restored hottest first, each one colder than the last: at the
                            // cold end they take back their place in the LRU
                            moe_cache_lru_push_front(*pool, job.slot);
                            device->rewarm_restored++;
                        } else {
                            moe_cache_lru_push_back(*pool, job.slot);
                        }
                        device->demand_count.erase(job.key);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        device->fill_failures++;
                    }
                }
            }
            moe_cache_split_done_locked(job);
            moe_cache_rewarm_feed_locked(*session, *device);
            session->idle_cv.notify_all();
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    if (stage2) {
        cudaFreeHost(stage2);
    }
    if (stage) {
        cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device);
        device.worker_started = true;
        return true;
    } catch (...) {
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

// the most entries a pool may hold: the kernels address an entry, its index is an int (ggml_moe_cache_max_pool_slots
// asked for a size multiple of the type's block, true of one tensor, not of a whole expert)
static size_t moe_cache_max_slots(size_t entry_size) {
    return entry_size ? std::min<size_t>(SIZE_MAX / entry_size, INT_MAX) : 0;
}

// The entry layout of an expert shape: the offset of each tensor, each starting at a multiple of moe_cache_role_align.
// Returns the entry's size.
static size_t moe_cache_entry_layout(const size_t bytes[MOE_ROLES], size_t offset[MOE_ROLES]) {
    size_t off = 0;
    for (int r = 0; r < MOE_ROLES; r++) {
        offset[r] = off;
        off += (bytes[r] + moe_cache_role_align - 1) / moe_cache_role_align * moe_cache_role_align;
    }
    return off;
}

// the role of an expert tensor from its name (blk.N.ffn_up_exps, ffn_gate_exps, ffn_down_exps), -1 for another one
// (a fused gate_up tensor among them)
static int moe_cache_role_of_name(const char * name) {
    if (!name || strstr(name, "gate_up") || strstr(name, "up_gate")) {
        return -1;
    }
    if (strstr(name, "_gate_")) return MOE_ROLE_GATE;
    if (strstr(name, "_up_"))   return MOE_ROLE_UP;
    if (strstr(name, "_down_")) return MOE_ROLE_DOWN;
    return -1;
}

// the layer of an expert tensor (any of the three, an entry's key among them), nullptr if unknown
static const moe_cache_layer * moe_cache_layer_of(const moe_cache_device & device, const void * tensor) {
    const auto it = device.layer_of.find(tensor);
    if (it == device.layer_of.end()) {
        return nullptr;
    }
    const auto l = device.layers.find(it->second);
    return l == device.layers.end() ? nullptr : &l->second;
}

// the pool of an expert tensor's layer, -1 if none
static int moe_cache_pool_of(const moe_cache_device & device, const void * tensor) {
    const moe_cache_layer * layer = moe_cache_layer_of(device, tensor);
    if (!layer || layer->shape < 0 || layer->shape >= (int) device.shapes.size()) {
        return -1;
    }
    return device.shapes[layer->shape].pool;
}

// the shape of a complete layer (its three tensors seen, one expert count): the existing one, or a new one while the
// pools are not built; -1 otherwise
static int moe_cache_shape_for(moe_cache_device & device, const moe_cache_layer & layer) {
    for (int r = 0; r < MOE_ROLES; r++) {
        if (!layer.tensor[r] || layer.bytes[r] == 0) {
            return -1;
        }
    }
    for (int i = 0; i < (int) device.shapes.size(); i++) {
        const moe_cache_shape & sh = device.shapes[i];
        bool same = true;
        for (int r = 0; r < MOE_ROLES; r++) {
            same = same && sh.role_bytes[r] == layer.bytes[r] && sh.role_type[r] == layer.type[r];
        }
        if (same) {
            return i;
        }
    }
    if (device.pools_built) {
        return -1;
    }
    moe_cache_shape sh;
    size_t offset[MOE_ROLES];
    for (int r = 0; r < MOE_ROLES; r++) {
        sh.role_bytes[r] = layer.bytes[r];
        sh.role_type[r] = layer.type[r];
    }
    sh.expert_size = moe_cache_entry_layout(sh.role_bytes, offset);
    device.shapes.push_back(sh);
    return (int) device.shapes.size() - 1;
}

#if defined(GGML_USE_HIP)
// amdgpu driver counters for the device: /sys/bus/pci/devices/<bus id>/mem_info_vram_{total,used}.
// On ROCm, hipMemGetInfo can report noticeably less free VRAM than these counters.
static bool moe_cache_sysfs_free(int physical, size_t & free_out, size_t & total_out) {
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), physical) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    std::string id(bus);
    for (char & c : id) {
        c = (char) tolower((unsigned char) c);
    }
    const std::string base = "/sys/bus/pci/devices/" + id + "/mem_info_vram_";
    const auto read_u64 = [](const std::string & path, unsigned long long & v) {
        FILE * f = fopen(path.c_str(), "r");
        if (!f) {
            return false;
        }
        const bool ok = fscanf(f, "%llu", &v) == 1;
        fclose(f);
        return ok;
    };
    unsigned long long total = 0;
    unsigned long long used  = 0;
    if (!read_u64(base + "total", total) || !read_u64(base + "used", used) || used > total) {
        return false;
    }
    free_out  = (size_t) (total - used);
    total_out = (size_t) total;
    return true;
}
#endif

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.budget_ready) {
        return device.budget_limit > 0;
    }
    device.budget_ready = true;

    ggml_cuda_set_device(device.logical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

#if defined(GGML_USE_HIP)
    // GGML_CUDA_MOE_CACHE_FREE_SOURCE=sysfs sizes the cache from the driver counters
    // instead of hipMemGetInfo; falls back to the HIP figure if they cannot be read
    {
        const char * source = getenv("GGML_CUDA_MOE_CACHE_FREE_SOURCE");
        if (source && strcmp(source, "sysfs") == 0) {
            size_t sysfs_free  = 0;
            size_t sysfs_total = 0;
            if (moe_cache_sysfs_free(device.physical, sysfs_free, sysfs_total)) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d free source=sysfs: hip=%zu MiB sysfs=%zu MiB (total %zu MiB)\n",
                        device.physical, free_memory >> 20, sysfs_free >> 20, sysfs_total >> 20);
                free_memory = sysfs_free;
            } else {
                MOE_CACHE_LOG("[moe-cache] CUDA%d free source=sysfs unavailable, using hip=%zu MiB\n",
                        device.physical, free_memory >> 20);
            }
        }
    }
#endif

    size_t reserve_bytes = 0;
    const size_t available = moe_cache_budget_claim(
            session, device, free_memory, reserve_bytes);
    device.budget_limit = available;

    if (session.config.budget_mb > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: cap=%zu MiB granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, session.config.budget_mb, available >> 20,
                free_memory >> 20, reserve_bytes >> 20);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: policy=free-minus-reserve granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, available >> 20, free_memory >> 20,
                reserve_bytes >> 20);
    }
    return available > 0;
}

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_layers <= 0 || shape.n_entries == 0) {
        return false;
    }
    shape.finished = true;

    size_t slots_by_budget = budget / shape.expert_size;
    size_t slot_count = std::min<size_t>(
            slots_by_budget,
            (size_t)std::min<uint64_t>(shape.n_entries, INT_MAX));
    slot_count = std::min(slot_count,
            moe_cache_max_slots(shape.expert_size));
    if (slot_count < moe_cache_pool_slots_min) {
        return false;
    }

    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= moe_cache_pool_slots_min) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < moe_cache_pool_slots_min) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }
    const size_t allocated = slot_count * shape.expert_size;
    moe_cache_budget_allocation(device, allocated, true);

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }
    try {
        pool->expert_size = moe_cache_entry_layout(shape.role_bytes, pool->role_offset);
        for (int r = 0; r < MOE_ROLES; r++) {
            pool->role_bytes[r] = shape.role_bytes[r];
        }
        pool->wtype = shape.role_type[MOE_ROLE_GATE];
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        pool->gpu_base = slab;
        pool->gpu_extent = (int64_t)slot_count;
        pool->zone_end = (int)slot_count;
        pool->ranges.push_back({slab, 0, (int)slot_count, 0});
        pool->covers_all_entries = (uint64_t)slot_count >= shape.n_entries;
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        cudaFree(device.pools.back()->slab);
        moe_cache_budget_allocation(device, allocated, false);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: experts %s/%s/%s (up/gate/down) entry=%zu KiB slots=%zu experts=%llu coverage=%s total=%zu MiB\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_UP]),
            ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_GATE]), ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_DOWN]),
            shape.expert_size >> 10, slot_count,
            (unsigned long long)shape.n_entries,
            device.pools.back()->covers_all_entries ? "complete" : "partial",
            allocated >> 20);
    bool expected = false;
    if (session.enabled_announced.compare_exchange_strong(expected, true)) {
        MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on CUDA%d\n", device.physical);
    }

    return true;
}

static size_t moe_cache_zone_bytes(const moe_cache_session & session);
static void moe_cache_setup_zone(moe_cache_device & device, size_t zone_bytes, int n_zones);
static void moe_cache_attach_test_region(moe_cache_device & device);
static void moe_cache_attach_kv(moe_cache_session & session, moe_cache_device & device);

// Every expert tensor registered in place (hipHostRegister, read-only; GGML_CUDA_MOE_PIN=0 disables), once the
// tensors are known and before any graph runs:
// - registering synchronizes every stream, which then gives back its SDMA engine; once generation runs, the
//   draft and main streams took both engines back before the fill stream (fills 0.6 - 1.25 ms instead of
//   ~60 us): here no stream holds one yet;
// - the range of a tensor is rounded outwards to whole pages: a prefetch copies a whole tensor, and a copy
//   that ends outside a registered range went through ROCm's staged path on registered pages, prefill -8 %
//   (and with one range per expert, 73 728 of them: -25 %);
// - the pages are not locked in RAM (the GPU mapping follows them by MMU notifiers); the GPU page tables
//   (~2 MiB per GiB, ~94 MiB here) come after the budget was fixed: out of the reserve.
// Measured (32k MTP, 3 pairs): fills 115 -> 61 us, CPU per hosted layer -4 to -6 %, generation ~+1.6 %,
// prefill unchanged.
static void moe_cache_pin_experts(moe_cache_device & device) {
    static const bool enabled = !getenv("GGML_CUDA_MOE_PIN") || atoi(getenv("GGML_CUDA_MOE_PIN")) != 0;
    if (!enabled || !device.pinned.empty()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    ggml_cuda_set_device(device.logical);
    size_t bytes = 0, fail = 0;
    for (const auto & t : device.seen_tensors) {
        const uintptr_t b = (uintptr_t) t.first;
        const uintptr_t ib = b & ~(uintptr_t) 4095;
        const uintptr_t ie = GGML_PAD(b + (size_t) t.second.n_expert * t.second.expert_size, 4096);
        if (cudaHostRegister((void *) ib, ie - ib, cudaHostRegisterReadOnly) == cudaSuccess) {
            device.pinned.push_back({ib, ie});
            bytes += ie - ib;
        } else {
            (void) cudaGetLastError();
            fail++;
        }
    }
    std::sort(device.pinned.begin(), device.pinned.end());
    MOE_CACHE_LOG("[moe-cache] CUDA%d pin: %zu expert tensors registered (%zu MiB) in %.1f s, %zu failed\n",
            device.physical, device.pinned.size(), bytes >> 20, (ggml_time_us() - t0) / 1e6, fail);
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    // the complete layers, grouped by expert shape
    for (auto & [number, layer] : device.layers) {
        GGML_UNUSED(number);
        if (layer.shape < 0) {
            layer.shape = moe_cache_shape_for(device, layer);
            if (layer.shape >= 0) {
                device.shapes[layer.shape].n_entries += (uint64_t) layer.n_expert;
                device.shapes[layer.shape].n_layers++;
            }
        }
    }
    device.pools_built = true;
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_layers > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    moe_cache_pin_experts(device);

    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve);
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_layers > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    long double total_weight = 0.0;
    size_t minimum_remaining = 0;
    for (moe_cache_shape & shape : device.shapes) {
        // by weight: the entry size times the experts of the shape (each shape holds the same share of its experts)
        if (!shape.finished && shape.n_layers > 0) {
            pending.push_back(&shape);
            total_weight +=
                (long double)shape.expert_size * (long double)shape.n_entries;
            const size_t minimum = shape.expert_size * moe_cache_pool_slots_min;
            minimum_remaining = minimum <= SIZE_MAX - minimum_remaining
                ? minimum_remaining + minimum : SIZE_MAX;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const long double lhs_weight =
            (long double)lhs->expert_size * (long double)lhs->n_entries;
        const long double rhs_weight =
            (long double)rhs->expert_size * (long double)rhs->n_entries;
        return lhs_weight > rhs_weight;
    });

    auto weighted_share = [](size_t available, long double weight, long double total) {
        if (available == 0 || weight <= 0.0 || total <= 0.0) {
            return (size_t)0;
        }
        const long double value = (long double)available * weight / total;
        return value >= (long double)available ? available : (size_t)value;
    };
    const bool complete_pools = minimum_remaining <= remaining;
    for (moe_cache_shape * shape : pending) {
        const long double weight =
            (long double)shape->expert_size * (long double)shape->n_entries;
        const size_t minimum = shape->expert_size * moe_cache_pool_slots_min;
        size_t share = weighted_share(remaining, weight, total_weight);
        if (complete_pools) {
            const size_t extra = remaining - minimum_remaining;
            const size_t bonus = weighted_share(extra, weight, total_weight);
            share = minimum + bonus;
        } else if (share < minimum && remaining >= minimum) {
            share = minimum;
        }
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
        if (complete_pools) {
            minimum_remaining -= minimum;
        }
    }

    moe_cache_setup_zone(device, moe_cache_zone_bytes(session), session.lend_zones);
    moe_cache_attach_test_region(device);
    moe_cache_attach_kv(session, device);
}

// An expert tensor shown to the cache (moe_cache_begin): recorded in its layer (number and role read from its name).
// Once every tensor known has been seen again without a new one (two passes of the scheduler's priming), the complete
// layers are grouped by expert shape and a pool is built per shape. A layer complete after that joins the pool of its
// shape, if there is one. Returns the pool of the tensor's layer, -1 if none (yet).
static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device, const char * name,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert) {
    int layer_number = -1;
    const int role = moe_cache_role_of_name(name);
    if (role < 0 || !moe_cache_layer_number(name, layer_number)) {
        return -1;
    }
    const bool first_visit = device.seen_tensors.emplace(
            host_base, moe_cache_seen_tensor{tensor_size, expert_size, wtype, n_expert}).second;
    moe_cache_layer & layer = device.layers[layer_number];
    if (first_visit) {
        layer.tensor[role] = host_base;
        layer.bytes[role] = expert_size;
        layer.type[role] = wtype;
        layer.n_expert = layer.n_expert ? std::min(layer.n_expert, n_expert) : n_expert;
        device.layer_of[host_base] = layer_number;
        device.visits_since_new_tensor = 0;
    } else if (device.visits_since_new_tensor < SIZE_MAX) {
        device.visits_since_new_tensor++;
    }
    if (layer.shape < 0 && device.pools_built) {
        // a layer completed after the pools were built: the pool of its shape takes its experts too
        const int shape = moe_cache_shape_for(device, layer);
        if (shape >= 0) {
            layer.shape = shape;
            moe_cache_shape & sh = device.shapes[shape];
            sh.n_entries += (uint64_t) layer.n_expert;
            sh.n_layers++;
            if (sh.pool >= 0 && sh.pool < (int) device.pools.size()) {
                device.pools[sh.pool]->covers_all_entries = (uint64_t) device.pools[sh.pool]->n_slots >= sh.n_entries;
            }
        }
    }
    if (!device.pools_built && device.visits_since_new_tensor >= device.seen_tensors.size()) {
        moe_cache_build_pending(session, device);
    }
    return moe_cache_pool_of(device, host_base);
}

static void moe_cache_log_stats(moe_cache_device & device) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = device.hits + device.misses;
    MOE_CACHE_LOG("[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld act-dedup=%lld cpu-overlap=%lld fusion=%lld/%lld pairs=%lld/%lld/%lld/%lld fusion-attempts=%lld fusion-nodes=%lld full-fusion=%lld/%lld down-mmv-dedicated=%lld bypass=%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, device.inserts, device.fills, device.fill_failures,
            device.evictions, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures,
            device.activation_dedup, device.overlap_rows,
            device.fused_rows, device.fused_candidates,
            device.pair_both, device.pair_up_only,
            device.pair_gate_only, device.pair_neither,
            device.fused_attempts,
            device.fused_nodes,
            device.full_fused_rows, device.full_fused_nodes,
            device.dedicated_down_mmv_dispatches,
            device.contention_bypasses);

    size_t zone_slots = 0;
    int zone_pools = 0;
    for (const auto & pool_ptr : device.pools) {
        if (moe_cache_pool_has_zone(*pool_ptr)) {
            zone_slots += (size_t)(pool_ptr->zone_end - pool_ptr->zone_begin);
            zone_pools++;
        }
    }
    if (zone_pools) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d zone: pools=%d slots=%zu lends=%lld restores=%lld moved=%lld evicted=%lld last-lend=%.2f ms checked=%lld failed=%lld\n",
                device.physical, zone_pools, zone_slots, device.zone_lends,
                device.zone_restores, device.zone_moves, device.zone_evictions,
                device.zone_lend_ms, device.zone_checked, device.zone_check_failed);
    }
}

static void moe_cache_log_configuration(moe_cache_session & session) {
    bool expected = false;
    if (!session.config_announced.compare_exchange_strong(expected, true)) {
        return;
    }

    const std::string budget = session.config.budget_mb > 0
        ? std::to_string(session.config.budget_mb) + " MiB cap"
        : "free-minus-reserve";
    const std::string overlap_cpu_rows = session.config.overlap_cpu_rows < 0
        ? "auto" : std::to_string(session.config.overlap_cpu_rows);
    const std::string admission = session.config.admit_after_explicit
        ? std::to_string(session.config.admit_after) + "-fixed/" +
            std::to_string(session.config.readmit_after) + "-replace"
        : "1-complete/" + std::to_string(session.config.admit_after) +
            "-partial/" + std::to_string(session.config.readmit_after) + "-replace";
    MOE_CACHE_LOG("[moe-cache] configured: mode=%s devices=%zu budget=%s reserve=%zu MiB min-slab=%zu MiB min-expert=%zu KiB max-batch=%d inserts=%d admit=%s cpu-overlap=%s fills=%s expert-parallel=%d down-mmv=%s\n",
            session.config.automatic ? "auto" : "on",
            session.devices.size(), budget.c_str(),
            session.config.reserve_mb,
            session.config.minimum_slab_bytes >> 20,
            session.config.min_expert_bytes >> 10,
            session.config.max_batch,
            session.config.inserts_per_plan,
            admission.c_str(),
            overlap_cpu_rows.c_str(),
            session.config.serial_fill ? "serial" : "parallel",
            session.config.expert_parallel,
            session.config.force_dedicated_down_mmv ? "dedicated" : "generic");
}

static void * moe_cache_session_create(
        void * const * backends, int n_backends,
        const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->reserve_explicit < 0 ||
                supplied_config->reserve_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->automatic < 0 ||
                supplied_config->automatic > 1 ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8 ||
                supplied_config->expert_parallel < -1 ||
                supplied_config->expert_parallel > moe_cache_expert_parallel_max) {
                return nullptr;
            }
            config.enabled = true;
            config.automatic = supplied_config->automatic != 0;
            config.min_devices = supplied_config->min_devices;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.reserve_explicit = supplied_config->reserve_explicit;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
            config.expert_parallel = supplied_config->expert_parallel;
            config.profile_path = supplied_config->profile_path
                ? supplied_config->profile_path : "";
        }
        if (!config.enabled) {
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        size_t default_min_expert_bytes = moe_cache_expert_bytes_ampere_min;
        int minimum_capability = INT_MAX;
        const auto add_backend = [&](ggml_backend_t backend) {
            if (!backend || !ggml_backend_is_cuda(backend) ||
                (int)session->devices.size() >= session->config.max_devices) {
                return;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int logical = context->device;
            const ggml_cuda_device_info & info = ggml_cuda_info();
            if (logical < 0 || logical >= info.device_count) {
                return;
            }
            const int physical = info.devices[logical].physical_device;
            if (!seen_devices.insert(physical).second) {
                return;
            }

            ggml_cuda_set_device(logical);
            const int capability = info.devices[logical].cc;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d is below %d\n",
                        physical, capability,
                        session->config.min_compute_capability);
                return;
            }

            default_min_expert_bytes = std::max(
                    default_min_expert_bytes,
                    moe_cache_default_min_expert_bytes(capability));
            minimum_capability = std::min(minimum_capability, capability);
            session->devices.emplace_back(new moe_cache_device(logical, physical));
            session->devices.back()->automatic_reserve_bytes =
                moe_cache_recommended_reserve_bytes(info.devices[logical].total_vram);
        };
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (ggml_backend_is_meta(backend)) {
                const size_t n_simple = ggml_backend_meta_n_backends(backend);
                for (size_t i = 0; i < n_simple; ++i) {
                    add_backend(ggml_backend_meta_simple_backend(backend, i));
                }
            } else {
                add_backend(backend);
            }
        }

        if ((int) session->devices.size() < session->config.min_devices) {
            return nullptr;
        }
        if (!session->config.reserve_explicit) {
            size_t reserve_bytes = 0;
            for (const auto & device_ptr : session->devices) {
                reserve_bytes = std::max(reserve_bytes,
                        device_ptr->automatic_reserve_bytes);
            }
            // Configuration logging reports the largest automatic reserve;
            // budget registration retains each device's own value.
            session->config.reserve_mb = reserve_bytes >> 20;
        }
        if (!session->config.min_expert_explicit) {
            session->config.min_expert_bytes = default_min_expert_bytes;
        }
        if (!session->config.serial_fill_explicit) {
            session->config.serial_fill =
                session->devices.size() < 2 || minimum_capability < moe_cache_cc_ampere;
        }
        if (session->config.expert_parallel < 0) {
            session->config.expert_parallel = std::min<int>(
                    3, session->devices.size());
        }
        if (session->config.expert_parallel > 0) {
            // Expert-parallel verification probes more routed rows per node and
            // consumes complete up/gate/down bundles. Admit enough rows to warm a
            // bundle promptly, then require sustained demand before displacing a
            // resident bundle. Serial cache policy keeps its established defaults.
            if (!session->config.inserts_per_plan_explicit) {
                session->config.inserts_per_plan = 16;
            }
            if (!session->config.readmit_after_explicit) {
                session->config.readmit_after = 40;
            }
        }
        if (!moe_cache_budget_register(*session)) {
            return nullptr;
        }
        moe_cache_profile_load(*session);

        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        } catch (...) {
            for (const auto & device_ptr : session->devices) {
                moe_cache_budget_unregister(*device_ptr);
            }
            throw;
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

// whether a fill reads from [base, base + size)
static bool moe_cache_job_overlaps(const moe_cache_job & job, const void * base, size_t size) {
    for (int r = 0; r < MOE_ROLES; r++) {
        if (moe_cache_ranges_overlap(job.source[r], job.role_bytes[r], base, size)) {
            return true;
        }
    }
    return false;
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (std::deque<moe_cache_job> * q : {&device.queue, &device.pregate_queue})
    for (auto it = q->begin(); it != q->end();) {
        if (all || moe_cache_job_overlaps(*it, base, size)) {
            moe_cache_split_done_locked(*it);   // a GPU waiting for it must not wait forever
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = q->erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.logical);
    for (const auto & r : device.pinned) {
        (void) cudaHostUnregister((void *) r.first);
    }
    device.pinned.clear();
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            moe_cache_budget_allocation(
                    device, (size_t)pool_ptr->n_slots * pool_ptr->expert_size, false);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_input) {
        cudaFree(device.d_input);
        moe_cache_budget_allocation(device, device.d_input_cap, false);
        device.d_input = nullptr;
    }
    if (device.d_act_q8) {
        cudaFree(device.d_act_q8);
        moe_cache_budget_allocation(device, device.act_q8_cap, false);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        cudaFree(device.d_out);
        moe_cache_budget_allocation(device, device.d_out_cap, false);
        device.d_out = nullptr;
    }
    if (device.h_input) {
        cudaFreeHost(device.h_input);
        device.h_input = nullptr;
    }
    if (device.h_out) {
        cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.compute_stream) {
        cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    for (char * region : device.owned_regions) {
        cudaFree(region);
    }
    device.owned_regions.clear();
    device.pools.clear();
    device.freq_period = 0;
    device.allocated_bytes = 0;
    device.d_input_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.h_input_cap = 0;
    device.h_out_cap = 0;
}

static void moe_hosted_shutdown(moe_cache_session * session);

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    // the host thread of the GPU-hosted MoE blocks plans with this session
    moe_hosted_shutdown(session);

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_rewarm_stop_locked(*device_ptr);
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
    }
    moe_cache_profile_save(*session);

    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    delete session;
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    if (session->active_scopes == 0) {
        // a graph starts: after prefill graphs (they lend), the first one that did not lend
        // means generation has started
        if (session->lent_in_last_graph) {
            session->lent_since_evictions = true;
        } else if (session->lent_since_evictions) {
            session->lent_since_evictions = false;
            for (auto & device_ptr : session->devices) {
                moe_cache_device & device = *device_ptr;
                size_t pending = 0;
                for (auto & pool_ptr : device.pools) {
                    pending += pool_ptr->rewarm.size();
                }
                if (pending && !device.rewarm_active) {
                    device.rewarm_active = true;
                    device.rewarm_restored = 0;
                    MOE_CACHE_LOG("[moe-cache] CUDA%d rewarm: %zu entries to restore\n",
                            device.physical, pending);
                    moe_cache_rewarm_feed_locked(*session, device);
                }
            }
        }
        session->lent_in_last_graph = false;
    }
    session->active_scopes++;
}

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        active->idle_cv.notify_all();
    }
}



// The cache learns the expert tensors from the nodes it is shown (ggml_backend_sched_prime_moe_cache: every host
// expert tensor of the graph, twice, before the first graph) and serves them through the MoE block hosted by the GPU
// only (ggml_cuda_moe_hosted_block): no node is computed here, the CPU's MUL_MAT_ID computes the batches the hosted MoE
// does not take. Returns nullptr.
static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
        int64_t n_tokens, int64_t n_rows) {
    GGML_UNUSED(n_rows);
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (!name || !host_base || !moe_cache_tensor_name_supported(name) || n_tokens < 1 ||
        expert_size < ggml_moe_cache_effective_min_expert_bytes(wtype,
            session->config.min_expert_explicit, session->config.min_expert_bytes) ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 || !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }
    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size || expert_size != (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / expert_size || expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }
    const size_t tensor_size = (size_t)n_expert * expert_size;
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->stopping) {
        return nullptr;
    }
    for (const auto & device_ptr : session->devices) {
        moe_cache_device & device = *device_ptr;
        if (device.dead.load() || !moe_cache_prepare_budget(*session, device)) {
            continue;
        }
        try {
            (void) moe_cache_discover_pool(*session, device, name, host_base, tensor_size, expert_size, wtype, n_expert);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d: tensor discovery failed (host allocation)\n", device.physical);
        }
    }
    return nullptr;
}

static bool moe_cache_enqueue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker, bool allow_eviction, bool rewarm, bool pregate, uint16_t freq, bool contest) {
    GGML_UNUSED(expert_size);   // the entry: pool.expert_size
    const moe_cache_key key{host_base, expert};
    const moe_cache_layer * layer = moe_cache_layer_of(device, host_base);
    const size_t queue_limit = session.config.queue_mb << 20;
    if (!layer || expert < 0 || expert >= layer->n_expert || inserts_left <= 0 ||
        (int)device.queue.size() >= session.config.queue_max ||
        pool.expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
        device.insert_skips++;
        return false;
    }

    int slot_index = -1;
    if (!pool.free_slots.empty()) {
        slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();
    } else if (allow_eviction) {
        int candidate = pool.lru_head;
        while (candidate >= 0 && pool.slots[candidate].readers > 0) {
            candidate = pool.slots[candidate].next;
        }
        if (candidate < 0) {
            device.insert_skips++;
            return false;
        }
        const uint16_t victim_freq = pool.slots[candidate].freq;
        if (contest && victim_freq >= freq) {
            device.admission_skips++;
            return false;
        }
        const moe_cache_key victim = pool.slots[candidate].key;
        slot_index = candidate;
        moe_cache_slot_reset(pool, slot_index, false);
        device.evictions++;
        if (session.config.lfu && victim_freq > 0) {
            try {
                device.demand_count[victim] = {victim_freq, pool.expert_size};   // its history follows it out
            } catch (...) {
            }
        }
    } else {
        return false;
    }

    moe_cache_slot & slot = pool.slots[slot_index];
    slot.key = key;
    slot.generation++;
    slot.readers = 0;
    slot.freq = freq;
    slot.state = moe_cache_slot_state::copying;
    try {
        const auto inserted = pool.map.emplace(key, slot_index);
        if (!inserted.second) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            return false;
        }
        moe_cache_direct_set(pool, key, slot_index);

        moe_cache_job job;
        job.pool = pool_index;
        job.slot = slot_index;
        job.generation = slot.generation;
        job.key = key;
        for (int r = 0; r < MOE_ROLES; r++) {
            job.source[r] = (const char *)layer->tensor[r] + (size_t)expert * layer->bytes[r];
            job.role_bytes[r] = layer->bytes[r];
        }
        job.bytes = pool.expert_size;
        job.rewarm = rewarm;
        (pregate ? device.pregate_queue : device.queue).push_back(job);
        device.queued_bytes += pool.expert_size;
    } catch (...) {
        moe_cache_slot_reset(pool, slot_index, true);
        device.insert_skips++;
        return false;
    }
    device.inserts++;
    inserts_left--;
    wake_worker = true;
    return true;
}

// Restore the entries lending evicted, hottest first, into free entries only (never
// evicting), and only when the fill worker has nothing else queued: an admission waits
// for at most one restoration. Called when generation starts and after every fill.
static void moe_cache_rewarm_feed_locked(moe_cache_session & session, moe_cache_device & device) {
    if (!device.rewarm_active || !device.queue.empty() || session.stopping || device.dead.load()) {
        return;
    }
    const size_t n = device.pools.size();
    for (size_t step = 0; step < n; step++) {
        const size_t index = (device.rewarm_next_pool + step) % n;
        moe_cache_pool & pool = *device.pools[index];
        while (!pool.rewarm.empty() && !pool.free_slots.empty()) {
            const moe_cache_key key = pool.rewarm.back();
            pool.rewarm.pop_back();
            if (pool.map.count(key)) {
                continue;   // admitted again in the meantime
            }
            int inserts_left = 1;
            bool wake_worker = false;
            if (moe_cache_enqueue_locked(session, device, pool, (int)index, key.tensor, key.expert,
                    pool.expert_size, inserts_left, wake_worker, false, true, false)) {
                device.rewarm_next_pool = index + 1;
                session.cv.notify_all();
                return;
            }
        }
    }
    // nothing left to restore, or no free entry left for it
    device.rewarm_active = false;
    size_t left = 0;
    for (auto & pool_ptr : device.pools) {
        left += pool_ptr->rewarm.size();
        pool_ptr->rewarm.clear();
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d rewarm: %lld entries restored, %zu dropped\n",
            device.physical, device.rewarm_restored, left);
}

// Stop restoring: a new prefill is lending (what is left belongs to the previous one),
// or the host weights or the session are going away.
static void moe_cache_rewarm_stop_locked(moe_cache_device & device) {
    if (!device.rewarm_active) {
        return;
    }
    device.rewarm_active = false;
    for (auto & pool_ptr : device.pools) {
        pool_ptr->rewarm.clear();
    }
}

static int moe_cache_lookup_or_queue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker) {
    const moe_cache_key key{host_base, expert};
    auto found = pool.map.find(key);
    device.freq_demands++;
    const uint16_t freq_cap = session.config.lfu ? (uint16_t) session.config.lfu_cap : std::numeric_limits<uint16_t>::max();
    if (found != pool.map.end()) {
        moe_cache_slot & slot = pool.slots[found->second];
        slot.freq += slot.freq < freq_cap;
        return slot.state == moe_cache_slot_state::valid ? found->second : -1;
    }

    moe_cache_demand * demand = nullptr;
    try {
        demand = &device.demand_count[key];
    } catch (...) {
        device.insert_skips++;
        return -1;
    }
    demand->expert_size = expert_size;
    demand->count += demand->count < freq_cap;
    const int initial_admit_after =
        pool.covers_all_entries && !session.config.admit_after_explicit
            ? 1 : session.config.admit_after;
    const bool contest = session.config.lfu && pool.free_slots.empty();
    const int admit_after = pool.free_slots.empty() && !contest
        ? std::max(initial_admit_after, session.config.readmit_after)
        : initial_admit_after;
    if (demand->count < admit_after) {
        device.admission_skips++;
        return -1;
    }
    // the entry carries the count from here (enqueue may rehash demand_count)
    if (moe_cache_enqueue_locked(session, device, pool, pool_index, host_base, expert, expert_size, inserts_left,
                                 wake_worker, true, false, false, demand->count, contest) &&
        session.config.lfu) {
        device.demand_count.erase(key);
    }
    return -1;
}

// LFU aging: every count halves once per lfu_window x entries demands, so recent demand outweighs history
static void moe_cache_freq_age_locked(const moe_cache_session & session, moe_cache_device & device) {
    if (!session.config.lfu) {
        return;
    }
    if (device.freq_period <= 0) {
        size_t entries = 0;
        for (const auto & pool : device.pools) {
            entries += pool->n_slots;
        }
        device.freq_period = (double) session.config.lfu_window * (double) std::max<size_t>(entries, 1);
    }
    if ((double) device.freq_demands < device.freq_period) {
        return;
    }
    device.freq_demands = 0;
    for (auto & pool : device.pools) {
        for (auto & slot : pool->slots) {
            slot.freq >>= 1;
        }
    }
    for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
        it->second.count >>= 1;
        it = it->second.count ? std::next(it) : device.demand_count.erase(it);
    }
}


static void moe_cache_end(void * opaque) {
    GGML_UNUSED(opaque);   // moe_cache_begin hands out no node
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    session.tensor_devices.erase(base);
    for (auto & device_ptr : session.devices) {
        // host weights are going away: a restoration must not queue a read of them again
        moe_cache_rewarm_stop_locked(*device_ptr);
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            for (int r = 0; r < MOE_ROLES && device_ptr->inflight; r++) {
                if (moe_cache_ranges_overlap(device_ptr->inflight_source[r], device_ptr->inflight_bytes[r], base, size)) {
                    return false;
                }
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        // the layers reading from the range: their entries reset, their tensors forgotten
        std::vector<int> gone;
        for (auto & [number, layer] : device.layers) {
            for (int r = 0; r < MOE_ROLES; r++) {
                if (layer.tensor[r] && moe_cache_ranges_overlap(layer.tensor[r], (size_t) layer.n_expert * layer.bytes[r], base, size)) {
                    gone.push_back(number);
                    break;
                }
            }
        }
        for (int number : gone) {
            moe_cache_layer & layer = device.layers[number];
            const void * key = layer.tensor[MOE_ROLE_GATE];
            if (layer.shape >= 0 && layer.shape < (int) device.shapes.size()) {
                moe_cache_shape & shape = device.shapes[layer.shape];
                if (shape.pool >= 0 && shape.pool < (int) device.pools.size()) {
                    moe_cache_pool & pool = *device.pools[shape.pool];
                    for (int index = 0; index < pool.n_slots; index++) {
                        moe_cache_slot & slot = pool.slots[index];
                        if (slot.state != moe_cache_slot_state::free && slot.key.tensor == key) {
                            moe_cache_slot_reset(pool, index, true);
                        }
                    }
                    pool.direct.erase(key);
                }
                shape.n_entries = (uint64_t) layer.n_expert <= shape.n_entries ? shape.n_entries - (uint64_t) layer.n_expert : 0;
                shape.n_layers = std::max<int64_t>(shape.n_layers - 1, 0);
                if (shape.pool >= 0 && shape.pool < (int) device.pools.size()) {
                    device.pools[shape.pool]->covers_all_entries = (uint64_t) device.pools[shape.pool]->n_slots >= shape.n_entries;
                }
            }
            for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
                it = it->first.tensor == key ? device.demand_count.erase(it) : std::next(it);
            }
            for (int r = 0; r < MOE_ROLES; r++) {
                if (layer.tensor[r]) {
                    session.tensor_devices.erase(layer.tensor[r]);
                    device.seen_tensors.erase(layer.tensor[r]);
                    device.layer_of.erase(layer.tensor[r]);
                }
            }
            device.layers.erase(number);
            device.visits_since_new_tensor = 0;
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    std::unique_lock<std::mutex> lock(session.mu);
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->zone_lent) {
            // the prefetch is using part of a slab: freeing it now would pull the
            // memory from under the running prefill
            MOE_CACHE_LOG("[moe-cache] CUDA%d trim refused: a zone is lent to the prefetch\n",
                    selected->physical);
            return 0;
        }
    }
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    freed += selected->d_out_cap + selected->d_input_cap +
             selected->act_q8_cap;
    lock.unlock();
    moe_cache_free_device(*selected);
    moe_cache_budget_unregister(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count) {
        return 0;
    }
    const int physical_device = info.devices[device].physical_device;
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, physical_device);
    }
    return freed;
}

// ---------------------------------------------------------------------------
// Lending the tail of the largest pools to the scheduler's expert prefetch.
//
// During prefill the cache is bypassed (batches above the row limit), so its memory
// idles while the prefetch needs two slots of the size of the largest offloaded
// expert tensor. Instead of allocating them next to the cache, the prefetch borrows
// a zone at the tail of each of the two largest pools for the duration of the
// prefill and gives it back before generation resumes.
// ---------------------------------------------------------------------------

// Size of one zone: set by the scheduler, or GGML_CUDA_MOE_CACHE_ZONE_MB for testing.
static size_t moe_cache_zone_bytes(const moe_cache_session & session) {
    if (session.lend_bytes) {
        return session.lend_bytes;
    }
    static const size_t env_bytes = [] () -> size_t {
        const char * text = getenv("GGML_CUDA_MOE_CACHE_ZONE_MB");
        const long long value = text && text[0] ? atoll(text) : 0;
        return value > 0 ? (size_t)value << 20 : 0;
    }();
    return env_bytes;
}

// Mark the tail of the first n_zones eligible pools as lendable: one prefetch slot each,
// so that a prefill evicts a little of every expert type rather than a lot of one.
// Nothing else changes: until lent, the zone is ordinary cache.
static void moe_cache_setup_zone(moe_cache_device & device, size_t zone_bytes, int n_zones) {
    if (!zone_bytes || n_zones <= 0) {
        return;
    }
    // the pools that can lend, and the entries of one slot in each
    std::vector<moe_cache_pool *> eligible;
    std::vector<int> slice, count;
    int equipped = 0;
    for (auto & pool_ptr : device.pools) {
        moe_cache_pool & pool = *pool_ptr;
        if (moe_cache_pool_has_zone(pool)) {
            equipped += (pool.zone_end - pool.zone_begin) / std::max(1, pool.zone_slice);
            continue;
        }
        if (!pool.slab || !pool.expert_size || ggml_type_is_exl3((ggml_type)pool.wtype)) {
            continue;
        }
        const int region0 = pool.ranges.empty() ? 0 : pool.ranges[0].count;
        const size_t zone_slots = (zone_bytes + pool.expert_size - 1) / pool.expert_size;
        if (!zone_slots || zone_slots > (size_t)region0 / 2) {
            continue;   // at most half of a pool, so the cache keeps a working set
        }
        eligible.push_back(&pool);
        slice.push_back((int)zone_slots);
        count.push_back(0);
    }
    // a slot for each pool first (each expert shape lends a little), then each further slot to the pool where the zone
    // would weigh least (its share of range 0), never past half of it
    for (int z = equipped; z < n_zones; z++) {
        int best = -1;
        double best_share = 2.0;
        for (size_t i = 0; i < eligible.size(); i++) {
            const int region0 = eligible[i]->ranges[0].count;
            if ((count[i] + 1) * (int64_t)slice[i] > region0 / 2) {
                continue;
            }
            const double share = count[i] == 0 ? -1.0 : (double)(count[i] + 1) * slice[i] / region0;
            if (share < best_share) {
                best_share = share;
                best = (int)i;
            }
        }
        if (best < 0) {
            break;
        }
        count[best]++;
        equipped++;
    }
    for (size_t i = 0; i < eligible.size(); i++) {
        if (count[i] == 0) {
            continue;
        }
        moe_cache_pool & pool = *eligible[i];
        const int region0 = pool.ranges[0].count;
        pool.zone_slice = slice[i];
        pool.zone_begin = region0 - count[i] * slice[i];
        pool.zone_end = region0;
        MOE_CACHE_LOG("[moe-cache] CUDA%d zone: type=%s %d slot(s) of %d entries of %d (%zu MiB) at entry %d\n",
                device.physical, ggml_type_name((ggml_type)pool.wtype), count[i], slice[i], region0,
                ((size_t)count[i] * slice[i] * pool.expert_size) >> 20, pool.zone_begin);
    }
    if (equipped < n_zones) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d zone: only %d of %d prefetch slots reserved%s\n",
                device.physical, equipped, n_zones,
                equipped < 2 ? ", the prefetch will not borrow" : "");
    }
}


// Give a pool a new range of device memory. The range is aligned so that every entry
// sits at gpu_base + k * expert_size (losing less than one entry), lowering the virtual
// base if the range sits below the slab. Returns the bytes actually taken.
static size_t moe_cache_pool_attach_range(moe_cache_pool & pool, char * ptr, size_t bytes) {
    const size_t es = pool.expert_size;
    if (!es || bytes < es) {
        return 0;
    }
    char * start = ptr;
    if (start < pool.gpu_base) {
        // the virtual base must stay below every range, on an entry boundary
        const size_t below = (size_t)(pool.gpu_base - start);
        const size_t steps = (below + es - 1) / es;
        pool.gpu_base -= steps * es;
        for (moe_cache_range & r : pool.ranges) {
            r.gpu_offset += (int64_t)steps;
        }
        pool.gpu_extent += (int64_t)steps;
    }
    const size_t misalign = (size_t)(start - pool.gpu_base) % es;
    if (misalign) {
        start += es - misalign;
    }
    if (start >= ptr + bytes) {
        return 0;
    }
    const int count = (int)((size_t)(ptr + bytes - start) / es);
    if (count <= 0) {
        return 0;
    }
    const int64_t offset = (int64_t)((size_t)(start - pool.gpu_base) / es);
    if (offset + count > (int64_t)moe_cache_max_slots(es)) {
        return 0;   // the kernels index entries as int32 from the virtual base
    }

    const int first = pool.n_slots;
    try {
        pool.slots.resize((size_t)first + count);
        pool.ranges.push_back({start, first, count, offset});
        pool.free_slots.reserve(pool.free_slots.size() + count);
    } catch (...) {
        pool.slots.resize((size_t)first);
        return 0;
    }
    for (int i = first + count - 1; i >= first; i--) {
        pool.free_slots.push_back(i);
    }
    pool.n_slots += count;
    pool.gpu_extent = std::max(pool.gpu_extent, offset + count);
    return (size_t)count * es;
}

// Share a region of device memory between the pools of a device, in proportion to what
// each already holds, so that every expert type grows by the same fraction.
static size_t moe_cache_region_attach(moe_cache_device & device, char * ptr, size_t bytes) {
    size_t weight = 0;
    for (const auto & pool_ptr : device.pools) {
        if (!ggml_type_is_exl3((ggml_type)pool_ptr->wtype)) {
            weight += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    if (!weight || !bytes) {
        return 0;
    }
    size_t used = 0;
    for (size_t i = 0; i < device.pools.size(); i++) {
        moe_cache_pool & pool = *device.pools[i];
        if (ggml_type_is_exl3((ggml_type)pool.wtype)) {
            continue;
        }
        const size_t share = i + 1 == device.pools.size()
            ? bytes - used
            : (size_t)((double)bytes * ((double)((size_t)pool.n_slots * pool.expert_size) / (double)weight));
        if (share < pool.expert_size || used + share > bytes) {
            continue;
        }
        const size_t taken = moe_cache_pool_attach_range(pool, ptr + used, share);
        MOE_CACHE_LOG("[moe-cache] CUDA%d region: type=%s +%zu MiB (%d slots), pool now %d slots, gpu extent %lld\n",
                device.physical, ggml_type_name((ggml_type)pool.wtype), taken >> 20,
                (int)(taken / pool.expert_size), pool.n_slots, (long long)pool.gpu_extent);
        used += share;
    }
    return used;
}

// GGML_CUDA_MOE_CACHE_TEST_REGION_MB: allocate that much device memory and hand it to
// the pools, to exercise multi-range addressing before the compute buffer lends its own.
static void moe_cache_attach_test_region(moe_cache_device & device) {
    static const size_t bytes = [] () -> size_t {
        const char * text = getenv("GGML_CUDA_MOE_CACHE_TEST_REGION_MB");
        const long long value = text && text[0] ? atoll(text) : 0;
        return value > 0 ? (size_t)value << 20 : 0;
    }();
    if (!bytes || !device.owned_regions.empty()) {
        return;
    }
    ggml_cuda_set_device(device.logical);
    char * ptr = nullptr;
    if (cudaMalloc((void **)&ptr, bytes) != cudaSuccess || !ptr) {
        (void) cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] CUDA%d test region of %zu MiB refused\n", device.physical, bytes >> 20);
        return;
    }
    device.owned_regions.push_back(ptr);
    moe_cache_region_attach(device, ptr, bytes);
}

// GGML_CUDA_MOE_CACHE_VERIFY=N: after a lend, read one moved entry out of N back and
// compare it with the weights in host memory.
static long long moe_cache_verify_period() {
    static const long long period = [] () -> long long {
        const char * text = getenv("GGML_CUDA_MOE_CACHE_VERIFY");
        const long long value = text && text[0] ? atoll(text) : 0;
        return value > 0 ? value : 0;
    }();
    return period;
}

static void moe_cache_zone_check(moe_cache_device & device, moe_cache_pool & pool, int index) {
    const moe_cache_slot & slot = pool.slots[index];
    if (slot.state != moe_cache_slot_state::valid || !slot.key.tensor) {
        return;
    }
    std::vector<char> host_copy;
    try {
        host_copy.resize(pool.expert_size);
    } catch (...) {
        return;
    }
    if (cudaMemcpy(host_copy.data(), moe_cache_slot_ptr(pool, index),
                pool.expert_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    const moe_cache_layer * layer = moe_cache_layer_of(device, slot.key.tensor);
    if (!layer) {
        return;
    }
    device.zone_checked++;
    bool same = true;
    for (int r = 0; r < MOE_ROLES; r++) {
        const char * source = (const char *)layer->tensor[r] + (size_t)slot.key.expert * layer->bytes[r];
        same = same && memcmp(host_copy.data() + pool.role_offset[r], source, pool.role_bytes[r]) == 0;
    }
    if (!same) {
        device.zone_check_failed++;
        MOE_CACHE_LOG("[moe-cache] CUDA%d ZONE CHECK FAILED slot=%d expert=%d\n",
                device.physical, index, slot.key.expert);
    }
}

// Empty a run of entries of one pool. Preconditions (caller holds session.mu): no fill queued
// or in flight, no reader, zone not lent. Every slot is therefore either free or valid
// and linked in the LRU. Copies are issued on moe_cache_zone_stream and waited for by the
// caller. The destinations of the moved entries are appended to 'moved'.
static void moe_cache_evacuate(
        moe_cache_device & device, moe_cache_pool & pool, int b, int e, std::vector<int> & moved) {
    const int zone_slots = e - b;

    const auto in_zone = [&](int index) { return index >= b && index < e; };

    // 1. free list: slots outside the zone become holes, slots inside leave with it
    std::vector<int> holes;
    holes.reserve(pool.free_slots.size() + (size_t)zone_slots);
    int free_in_zone = 0;
    for (int index : pool.free_slots) {
        if (in_zone(index)) {
            free_in_zone++;
        } else {
            holes.push_back(index);
        }
    }
    pool.free_slots.clear();

    // 2. evict the coldest entries, from the LRU head, until every entry still in the
    //    zone has a hole to move to
    int occupied = zone_slots - free_in_zone;
    while (occupied > (int)holes.size() && pool.lru_head >= 0) {
        const int index = pool.lru_head;
        pool.rewarm.push_back(pool.slots[index].key);   // coldest first
        moe_cache_slot_reset(pool, index, false);   // unlinks the head, erases the key
        if (in_zone(index)) {
            occupied--;
        } else {
            holes.push_back(index);
        }
        device.zone_evictions++;
    }
    if (pool.rewarm.size() > (size_t)pool.n_slots) {
        // prefills with no generation after them: keep the most recent (hottest) ones
        pool.rewarm.erase(pool.rewarm.begin(), pool.rewarm.end() - pool.n_slots);
    }

    // 3. move the zone entries that survived into the holes, keeping their LRU place
    for (int source = b; source < e; source++) {
        moe_cache_slot & src = pool.slots[source];
        if (src.state != moe_cache_slot_state::valid) {
            continue;
        }
        GGML_ASSERT(!holes.empty() && "zone evacuation: not enough holes");
        const int destination = holes.back();
        holes.pop_back();

        cudaMemcpyAsync(moe_cache_slot_ptr(pool, destination),
                moe_cache_slot_ptr(pool, source),
                pool.expert_size, cudaMemcpyDeviceToDevice, moe_cache_zone_stream);

        moe_cache_slot & dst = pool.slots[destination];
        dst.key = src.key;
        dst.generation++;
        dst.readers = 0;
        dst.state = moe_cache_slot_state::valid;
        dst.prev = src.prev;
        dst.next = src.next;
        if (dst.prev >= 0) {
            pool.slots[dst.prev].next = destination;
        } else {
            pool.lru_head = destination;
        }
        if (dst.next >= 0) {
            pool.slots[dst.next].prev = destination;
        } else {
            pool.lru_tail = destination;
        }
        auto it = pool.map.find(dst.key);
        GGML_ASSERT(it != pool.map.end() && it->second == source);
        it->second = destination;
        moe_cache_direct_set(pool, dst.key, destination);

        src.key = {};
        src.generation++;
        src.readers = 0;
        src.state = moe_cache_slot_state::free;
        src.prev = -1;
        src.next = -1;
        moved.push_back(destination);
        device.zone_moves++;
    }

    // 4. what is left are free slots outside the zone
    pool.free_slots.swap(holes);
}

// ---------------------------------------------------------------------------
// The end of the KV cache lent to the pools (by default; GGML_CUDA_MOE_CACHE_KV_LEND=0 disables it). The KV tensors are
// allocated for the whole context, but attention only reads the n_kv cells in use: the rows past the step above them
// hold experts. Each KV tensor's tail is attached once to a pool when the pools are built, before any graph captures
// the virtual base; taking rows back is the zone evacuation on a run of logical slots, giving them back puts free
// slots back. Bench: tests/kvlendtest (the same functions on host memory, ASan + UBSan, random operations, invariants
// and bytes).
// ---------------------------------------------------------------------------

// on by default; GGML_CUDA_MOE_CACHE_KV_LEND=0 keeps the whole KV cache out of the expert cache
static bool moe_cache_kv_lend_enabled() {
    static const bool enabled = !getenv("GGML_CUDA_MOE_CACHE_KV_LEND") || atoi(getenv("GGML_CUDA_MOE_CACHE_KV_LEND")) != 0;
    return enabled;
}

// the tail of a KV tensor, from byte 'from' to its end, as a range of the pool, lent to it whole. Returns the bytes.
static size_t moe_cache_kv_attach(moe_cache_pool & pool, const char * tensor, size_t bytes, size_t row_bytes, size_t from) {
    if (from >= bytes) {
        return 0;
    }
    const int first = pool.n_slots;
    const size_t got = moe_cache_pool_attach_range(pool, (char *) tensor + from, bytes - from);
    if (!got) {
        return 0;
    }
    const moe_cache_range & r = pool.ranges.back();
    pool.kv.push_back({tensor, bytes, row_bytes, r.base, first, r.count, 0});
    return got;
}

// The KV uses its tensor up to byte 'end': the entries that start before 'end' leave the cache (from the lowest; their
// memory zeroed: attention reads masked cells up to n_kv, and a NaN in them would come through the WMMA's zero
// weights), those at or after 'end' come back. Preconditions of moe_cache_evacuate. Copies and memsets on the zone
// stream, waited for by the caller.
static void moe_cache_kv_set_end(moe_cache_device & device, moe_cache_pool & pool, moe_cache_kv_range & kv, size_t end,
                                 std::vector<int> & moved) {
    int want = 0;
    if (kv.tensor + end > kv.start) {
        const size_t over = (size_t) (kv.tensor + end - kv.start);
        want = (int) std::min<size_t>((over + pool.expert_size - 1) / pool.expert_size, (size_t) kv.count);
    }
    if (want > kv.taken) {
        moe_cache_evacuate(device, pool, kv.first + kv.taken, kv.first + want, moved);
        cudaMemsetAsync(moe_cache_slot_ptr(pool, kv.first + kv.taken), 0, (size_t) (want - kv.taken) * pool.expert_size,
                        moe_cache_zone_stream);
        kv.taken = want;
    } else if (want < kv.taken) {
        for (int i = kv.first + kv.taken - 1; i >= kv.first + want; i--) {
            GGML_ASSERT(pool.slots[i].state == moe_cache_slot_state::free);
            pool.free_slots.push_back(i);
        }
        kv.taken = want;
    }
}

// Once per device, when its pools are built: the step (rows) is the smallest multiple of 256 (the padding of n_kv)
// in which the KV tensor with the fewest bytes per row holds the largest expert; the first step is never lent. Each
// tensor goes to the pool whose share of KV bytes (against its slab) is smallest, the largest tensors first, so that
// no expert type alone carries the loan. The kernels take each entry's address: a range may lie anywhere.
static void moe_cache_attach_kv(moe_cache_session & session, moe_cache_device & device) {
    if (device.kv_attached || !moe_cache_kv_lend_enabled()) {
        return;
    }
    device.kv_attached = true;   // once: a pool rebuilt later (trim) never takes the KV again, its base is captured
    std::vector<moe_cache_session::kv_tensor> mine;
    size_t min_row = SIZE_MAX, max_es = 0;
    for (const auto & t : session.kv_tensors) {
        if (t.logical == device.logical && t.row_bytes > 0) {
            mine.push_back(t);
            min_row = std::min(min_row, t.row_bytes);
        }
    }
    std::vector<moe_cache_pool *> pools;
    for (auto & p : device.pools) {
        if (p->slab && !ggml_type_is_exl3((ggml_type) p->wtype)) {
            pools.push_back(p.get());
            max_es = std::max(max_es, p->expert_size);
        }
    }
    if (mine.empty() || pools.empty()) {
        return;
    }
    const int64_t step = GGML_PAD((int64_t) ((max_es + min_row - 1) / min_row), 256);
    std::sort(mine.begin(), mine.end(), [](const auto & a, const auto & b) { return a.bytes > b.bytes; });
    std::vector<double> kv_bytes(pools.size(), 0.0);
    size_t attached = 0;
    int n_ranges = 0;
    for (const auto & t : mine) {
        size_t best = 0;
        double best_share = 1e300;
        for (size_t i = 0; i < pools.size(); i++) {
            const double share = (kv_bytes[i] + (double) t.bytes) / ((double) pools[i]->n_slots * pools[i]->expert_size);
            if (share < best_share) {
                best_share = share;
                best = i;
            }
        }
        const size_t got = moe_cache_kv_attach(*pools[best], t.data, t.bytes, t.row_bytes, (size_t) step * t.row_bytes);
        if (got) {
            kv_bytes[best] += (double) got;
            attached += got;
            n_ranges++;
        } else {
            MOE_CACHE_LOG("[moe-cache] CUDA%d KV tensor of %zu MiB not lent (tail of %zu KiB, under one entry)\n",
                    device.physical, t.bytes >> 20, (t.bytes - std::min(t.bytes, (size_t) step * t.row_bytes)) >> 10);
        }
    }
    if (!attached) {
        return;
    }
    device.kv_step_rows = step;
    device.kv_end_rows = step;
    for (const auto & t : mine) {
        device.kv_max_rows = std::max<int64_t>(device.kv_max_rows, (int64_t) (t.bytes / t.row_bytes));
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d KV lent: %d of %zu tensors, %zu MiB past the first %lld rows (step %lld rows)\n",
            device.physical, n_ranges, mine.size(), attached >> 20, (long long) step, (long long) step);
}

static void moe_cache_kv_register(void * opaque, const struct ggml_tensor * tensor, size_t row_bytes) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session || !tensor || !tensor->data || !moe_cache_kv_lend_enabled()) {
        return;
    }
    // the device holding it (a KV tensor in host memory is not lent)
#if defined(GGML_USE_HIP)
    hipPointerAttribute_t attr = {};
    if (hipPointerGetAttributes(&attr, tensor->data) != hipSuccess || attr.type != hipMemoryTypeDevice) {
#else
    cudaPointerAttributes attr = {};
    if (cudaPointerGetAttributes(&attr, tensor->data) != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
#endif
        (void) cudaGetLastError();
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    session->kv_tensors.push_back({(const char *) tensor->data, ggml_nbytes(tensor), row_bytes, attr.device});
}

static int64_t moe_cache_kv_end_rows(const moe_cache_device & device, int64_t rows) {
    const int64_t s = device.kv_step_rows;
    rows = std::min(rows, device.kv_max_rows);   // "everything" (a state restore) is the whole tensor
    return std::max<int64_t>(s, (rows + s - 1) / s * s);
}

static bool moe_cache_kv_needs_set_rows(void * opaque, int64_t rows) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return false;
    }
    for (auto & device_ptr : session->devices) {
        const moe_cache_device & d = *device_ptr;
        if (d.kv_step_rows > 0 && !d.dead.load() && moe_cache_kv_end_rows(d, rows) != d.kv_end_rows) {
            return true;
        }
    }
    return false;
}

static void moe_hosted_unpin_session(moe_cache_session & session);

static void moe_cache_kv_set_rows(void * opaque, int64_t rows) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    for (auto & device_ptr : session->devices) {
        moe_cache_device & device = *device_ptr;
        if (device.kv_step_rows <= 0 || device.dead.load()) {
            continue;
        }
        const int64_t end_rows = moe_cache_kv_end_rows(device, rows);
        if (end_rows == device.kv_end_rows) {
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const bool take = end_rows > device.kv_end_rows;
        if (take) {
            // as for a zone loan: no prefetch, fill or restoration may be copying an entry meanwhile, and no row pinned
            device.pregate_window = false;
            device.pg_cands.clear();
            device.pg_next = 0;
            session->idle_cv.wait(lock, [&] { return !device.pg_busy; });
            session->idle_cv.wait(lock, [&] { return device.queue.empty() && !device.inflight; });
            moe_cache_rewarm_stop_locked(device);
            moe_hosted_unpin_session(*session);
        }
        ggml_cuda_set_device(device.logical);
        const int64_t ev0 = device.zone_evictions, mv0 = device.zone_moves;
        std::vector<int> moved;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (moe_cache_kv_range & kv : pool.kv) {
                const size_t end = std::min(kv.bytes, (size_t) end_rows * kv.row_bytes);
                moe_cache_kv_set_end(device, pool, kv, end, moved);
            }
        }
        if (take) {
            const cudaError_t error = cudaStreamSynchronize(moe_cache_zone_stream);
            if (error != cudaSuccess) {
                moe_cache_cuda_ok(device, error, "KV evacuation", true);   // the bookkeeping points at the copies
            }
            device.kv_takes++;
        } else {
            device.kv_gives++;
        }
        device.kv_moves += device.zone_moves - mv0;
        device.kv_evictions += device.zone_evictions - ev0;
        device.zone_moves = mv0;          // not counted as prefetch zone traffic
        device.zone_evictions = ev0;
        device.kv_end_rows = end_rows;
        device.kv_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        static const bool log_each = getenv("GGML_CUDA_MOE_CACHE_KV_LOG") != nullptr;
        if (log_each) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d KV %s to %lld rows: %zu moved, %lld evicted, total %.1f ms (%lld takes, %lld gives)\n",
                    device.physical, take ? "grows" : "shrinks", (long long) end_rows, moved.size(),
                    (long long) (device.kv_evictions), device.kv_ms, (long long) device.kv_takes, (long long) device.kv_gives);
        }
    }
}

static int64_t moe_cache_kv_first_lent_row(void * opaque) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return 0;
    }
    int64_t first = 0;
    for (auto & device_ptr : session->devices) {
        if (device_ptr->kv_step_rows > 0) {
            first = first ? std::min(first, device_ptr->kv_step_rows) : device_ptr->kv_step_rows;
        }
    }
    return first;
}

static moe_cache_device * moe_cache_device_for_backend(
        moe_cache_session & session, void * backend_ptr) {
    ggml_backend_t backend = (ggml_backend_t)backend_ptr;
    if (!backend || !ggml_backend_is_cuda(backend)) {
        return nullptr;
    }
    const int logical = ((ggml_backend_cuda_context *)backend->context)->device;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->logical == logical) {
            return device_ptr.get();
        }
    }
    return nullptr;
}

static void moe_cache_set_lend_size(void * opaque, size_t bytes, int n_zones) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->lend_bytes == bytes && session->lend_zones == n_zones) {
        return;
    }
    session->lend_bytes = bytes;
    session->lend_zones = n_zones;
    // pools created before the size was known get their zone now (idempotent)
    for (auto & device_ptr : session->devices) {
        if (device_ptr->budget_ready && !device_ptr->dead.load()) {
            moe_cache_setup_zone(*device_ptr, bytes, n_zones);
        }
    }
}

// Lend up to n_zones zones of at least 'bytes' on the device of 'backend', one per pool.
// Returns the number lent and fills 'zones', or 0 and lends nothing when fewer than two
// zones are available (the prefetch needs two slots to overlap copies and compute).
static int moe_cache_lend(void * opaque, void * backend, size_t bytes, void ** zones, int n_zones) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !zones || n_zones <= 0) {
        return 0;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return 0;
    }

    std::unique_lock<std::mutex> lock(session->mu);
    // PREGATE: a prefill moves entries: no prefetch may be copying meanwhile
    device->pregate_window = false;
    device->pg_cands.clear();
    device->pg_next = 0;
    session->idle_cv.wait(lock, [&] { return !device->pg_busy; });
    if (device->dead.load() || !device->budget_ready) {
        return 0;   // no pool yet (first prefill) or cache disabled after a trim
    }
    session->lent_in_last_graph = true;
    moe_cache_rewarm_stop_locked(*device);
    // the pools whose zone has room, and the slots each gives (a slot: zone_slice contiguous entries)
    std::vector<moe_cache_pool *> chosen;
    int n_slots = 0;
    for (auto & pool_ptr : device->pools) {
        moe_cache_pool & pool = *pool_ptr;
        if (n_slots < n_zones && moe_cache_pool_has_zone(pool) && !pool.zone_lent && pool.zone_slice > 0 &&
                (size_t)pool.zone_slice * pool.expert_size >= bytes) {
            chosen.push_back(&pool);
            n_slots += (pool.zone_end - pool.zone_begin) / pool.zone_slice;
        }
    }
    if (n_slots < 2) {
        return 0;
    }

    // let the fills already decided land; nobody enqueues while we wait, since
    // admissions come from this very thread
    session->idle_cv.wait(lock, [&] {
        return device->queue.empty() && !device->inflight;
    });
    if (device->dead.load()) {
        return 0;
    }
    for (moe_cache_pool * pool : chosen) {
        for (const moe_cache_slot & slot : pool->slots) {
            if (slot.readers > 0) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d lend refused: expert still being read\n",
                        device->physical);
                return 0;
            }
        }
    }

    ggml_cuda_set_device(device->logical);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<int>> moved(chosen.size());
    for (size_t i = 0; i < chosen.size(); i++) {
        moe_cache_evacuate(*device, *chosen[i], chosen[i]->zone_begin, chosen[i]->zone_end, moved[i]);
    }
    const cudaError_t error = cudaStreamSynchronize(moe_cache_zone_stream);
    if (error != cudaSuccess) {
        // the bookkeeping already points at the destinations: the copies must land
        moe_cache_cuda_ok(*device, error, "zone evacuation", true);
    }
    int given = 0;
    for (size_t i = 0; i < chosen.size(); i++) {
        moe_cache_pool & pool = *chosen[i];
        pool.zone_lent = true;
        for (int first = pool.zone_begin; first + pool.zone_slice <= pool.zone_end && given < n_zones; first += pool.zone_slice) {
            zones[given++] = moe_cache_slot_ptr(pool, first);   // range 0: contiguous
        }
    }
    device->zone_lends++;
    device->zone_lend_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

    const long long period = moe_cache_verify_period();
    if (period) {
        for (size_t i = 0; i < chosen.size(); i++) {
            moe_cache_pool & pool = *chosen[i];
            for (size_t m = 0; m < moved[i].size(); m += (size_t)period) {
                moe_cache_zone_check(*device, pool, moved[i][m]);
            }
            for (const auto & kv : pool.map) {
                if (kv.second >= pool.zone_begin && kv.second < pool.zone_end) {
                    device->zone_check_failed++;
                    MOE_CACHE_LOG("[moe-cache] CUDA%d ZONE CHECK FAILED: map entry in lent zone\n",
                            device->physical);
                    break;
                }
            }
        }
    }
    return given;
}


// ---------------------------------------------------------------------------
// Lending plain memory to the rest of the backend (the compute buffer of a prefill
// graph). Same machinery as the prefetch zones, but for an arbitrary size: entries are
// taken from the tail of range 0, under the prefetch zone, and given back later.
// ---------------------------------------------------------------------------

// How much of a pool may be lent out at once, prefetch zone included. The cache is
// bypassed while the memory is lent (it is lent for a prefill), so this floor only has
// to keep the pool usable for the single-token nodes of the last layer.
static constexpr int moe_cache_lend_percent = 95;

// How far a pool may be lent beyond the least-lent one that could also take the block.
// Measured over the real chunk list at two cache sizes: at 5 or 10 points the early
// choices leave later blocks unplaceable, at 30 one pool gets drained; 15 gives the
// smallest spread with no refusal.
static constexpr double moe_cache_lend_fairness = 0.15;

// lowest entry currently lent, or the start of the prefetch zone
static int moe_cache_pool_lend_low(const moe_cache_pool & pool) {
    const int region0 = pool.ranges.empty() ? 0 : pool.ranges[0].count;
    if (pool.lent_low < region0) {
        return pool.lent_low;
    }
    return pool.zone_begin < region0 ? pool.zone_begin : region0;
}

// entries this pool could still lend
static int moe_cache_pool_lendable(const moe_cache_pool & pool) {
    if (pool.ranges.empty() || !pool.expert_size || ggml_type_is_exl3((ggml_type)pool.wtype)) {
        return 0;
    }
    const int region0 = pool.ranges[0].count;
    const int floor = region0 - (int)((int64_t)region0 * moe_cache_lend_percent / 100);
    const int low = moe_cache_pool_lend_low(pool);
    return low > floor ? low - floor : 0;
}

// Seat a whole list of chunks at once: every assignment of the chunks to the pools that
// can lend, keeping the one whose most-lent pool is the least lent (then the smallest sum
// of squared loads). No rule looking at one chunk at a time finds it in every case, and a
// chunk left without a pool is allocated on top of the cache. The chunks are placed one
// by one; a branch stops as soon as a pool would overflow or be more lent than the best
// complete assignment so far (loads only grow along a branch, so it cannot do better).
// The prefill chunks of today (8 over 3 pools): 0.02 ms; 16, the most ggml-alloc makes:
// under 80 ms in the worst of 20 000 random lists.
struct moe_cache_loan_search {
    size_t n = 0, np = 0;
    std::vector<int> want;   // want[i * np + k]: entries chunk i takes in pool k
    std::vector<int> room, low, region0, used, pick, best;
    double best_max = 2.0, best_sq = 0.0;

    double lent(size_t k, int extra) const {
        return (double)(region0[k] - (low[k] - used[k] - extra)) / (double)region0[k];
    }

    void place(size_t i) {
        if (i == n) {
            double max_lent = 0.0, sq = 0.0;
            for (size_t k = 0; k < np; k++) {
                max_lent = std::max(max_lent, lent(k, 0));
                sq += lent(k, 0) * lent(k, 0);
            }
            if (max_lent < best_max || (max_lent == best_max && sq < best_sq)) {
                best_max = max_lent;
                best_sq = sq;
                best = pick;
            }
            return;
        }
        for (size_t k = 0; k < np; k++) {
            const int w = want[i * np + k];
            if (used[k] + w > room[k] || lent(k, w) > best_max) {
                continue;
            }
            used[k] += w;
            pick[i] = (int)k;
            place(i + 1);
            used[k] -= w;
        }
    }
};

static bool moe_cache_borrows(ggml_backend_buffer_type_t buft) {
    void * opaque = nullptr;
    ggml_backend_t backend = nullptr;
    return ggml_backend_cuda_borrowed_buffer_type_source(buft, &opaque, &backend) && opaque;
}

static void moe_cache_plan_borrows(ggml_backend_buffer_type_t buft, const size_t * sizes, int n_sizes) {
    void * opaque = nullptr;
    ggml_backend_t backend = nullptr;
    if (!ggml_backend_cuda_borrowed_buffer_type_source(buft, &opaque, &backend)) {
        return;
    }
    moe_cache_session * session = (moe_cache_session *)opaque;
    moe_cache_device * device = session ? moe_cache_device_for_backend(*session, backend) : nullptr;
    if (!device) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    device->loan_plan_bytes.clear();
    device->loan_plan_pool.clear();
    if (device->dead.load() || !device->budget_ready) {
        return;
    }

    std::vector<moe_cache_pool *> lenders;
    for (auto & pool_ptr : device->pools) {
        if (moe_cache_pool_lendable(*pool_ptr) > 0) {
            lenders.push_back(pool_ptr.get());
        }
    }
    std::vector<size_t> blocks;
    for (int i = 0; i < n_sizes; i++) {
        if (sizes[i]) {
            blocks.push_back(sizes[i]);   // ggml-alloc allocates nothing for an empty chunk
        }
    }
    const size_t np = lenders.size(), n = blocks.size();
    if (!n || !np) {
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    moe_cache_loan_search search;
    search.n = n;
    search.np = np;
    search.want.resize(n * np);
    search.room.resize(np);
    search.low.resize(np);
    search.region0.resize(np);
    search.used.assign(np, 0);
    search.pick.resize(n);
    for (size_t k = 0; k < np; k++) {
        search.room[k] = moe_cache_pool_lendable(*lenders[k]);
        search.low[k] = moe_cache_pool_lend_low(*lenders[k]);
        search.region0[k] = lenders[k]->ranges[0].count;
        for (size_t i = 0; i < n; i++) {
            search.want[i * np + k] = (int)((blocks[i] + lenders[k]->expert_size - 1) / lenders[k]->expert_size);
        }
    }
    search.place(0);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (search.best.empty()) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d loan plan: no seating for %zu chunks (%.2f ms)\n",
                device->physical, n, ms);
        return;
    }
    for (size_t i = 0; i < n; i++) {
        device->loan_plan_bytes.push_back(blocks[i]);
        device->loan_plan_pool.push_back(lenders[search.best[i]]);
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d loan plan: %zu chunks over %zu pools, most lent %d%% (%.2f ms)\n",
            device->physical, n, np, (int)(100 * search.best_max), ms);
}

// Borrow 'bytes' of contiguous device memory from the cache of this backend's device.
// Returns nullptr when no pool can spare that much.
static void * moe_cache_borrow(void * opaque, void * backend, size_t bytes) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !bytes) {
        return nullptr;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return nullptr;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    if (device->dead.load() || !device->budget_ready) {
        return nullptr;
    }

    moe_cache_rewarm_stop_locked(*device);

    // The plan first: the pool it chose for a chunk of this size, if it can still take it.
    moe_cache_pool * chosen = nullptr;
    int slots = 0;
    for (size_t i = 0; i < device->loan_plan_pool.size(); i++) {
        moe_cache_pool * pool = device->loan_plan_pool[i];
        if (!pool || device->loan_plan_bytes[i] != bytes) {
            continue;
        }
        const int want = (int)((bytes + pool->expert_size - 1) / pool->expert_size);
        if (moe_cache_pool_lendable(*pool) >= want) {
            chosen = pool;
            slots = want;
            device->loan_plan_pool[i] = nullptr;
        }
        break;
    }

    // Otherwise, like a waiter seating a party: the smallest table that fits, so that the
    // large pools stay free for the large blocks (a tensor bigger than the chunk limit
    // gets a chunk of its own: the QSA scores are 2 GiB). But never seat a party at a
    // table already much more crowded than another that would do, so that no expert type
    // ends up carrying the whole loan.
    if (!chosen) {
        struct candidate { moe_cache_pool * pool; int slots; int64_t room; double lent_after; };
        std::vector<candidate> candidates;
        double least_lent = 1.0;
        for (auto & pool_ptr : device->pools) {
            moe_cache_pool & pool = *pool_ptr;
            const int want = (int)((bytes + pool.expert_size - 1) / pool.expert_size);
            const int room = moe_cache_pool_lendable(pool);
            if (room < want) {
                continue;
            }
            const int region0 = pool.ranges[0].count;
            const double lent_after = region0 > 0
                ? (double)(region0 - (moe_cache_pool_lend_low(pool) - want)) / (double)region0 : 1.0;
            candidates.push_back({&pool, want, static_cast<int64_t>((int64_t)room * (int64_t)pool.expert_size), lent_after});
            least_lent = std::min(least_lent, lent_after);
        }
        int64_t smallest_room = 0;
        for (const candidate & c : candidates) {
            if (c.lent_after > least_lent + moe_cache_lend_fairness) {
                continue;
            }
            if (!chosen || c.room < smallest_room) {
                chosen = c.pool;
                slots = c.slots;
                smallest_room = c.room;
            }
        }
    }
    if (!chosen) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d borrow: no pool can lend %zu MiB in one piece\n",
                device->physical, bytes >> 20);
        return nullptr;
    }

    // let the fills already decided land; nobody enqueues while we wait
    session->idle_cv.wait(lock, [&] {
        return device->queue.empty() && !device->inflight;
    });
    if (device->dead.load()) {
        return nullptr;
    }
    for (const moe_cache_slot & slot : chosen->slots) {
        if (slot.readers > 0) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d borrow refused: expert still being read\n", device->physical);
            return nullptr;
        }
    }

    ggml_cuda_set_device(device->logical);

    const int low = moe_cache_pool_lend_low(*chosen);
    const int begin = low - slots;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<int> moved;
    moe_cache_evacuate(*device, *chosen, begin, low, moved);
    const cudaError_t error = cudaStreamSynchronize(moe_cache_zone_stream);
    if (error != cudaSuccess) {
        moe_cache_cuda_ok(*device, error, "borrow evacuation", true);
    }
    chosen->lent.push_back({begin, slots});
    chosen->lent_low = begin;
    device->zone_lends++;
    device->zone_lend_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    MOE_CACHE_LOG("[moe-cache] CUDA%d borrow: %zu MiB from the %s pool (%d entries at %d, %d%% of it lent), %.2f ms\n",
            device->physical, bytes >> 20, ggml_type_name((ggml_type)chosen->wtype), slots, begin,
            (int)(100 * (chosen->ranges[0].count - begin) / chosen->ranges[0].count),
            device->zone_lend_ms);

    const long long period = moe_cache_verify_period();
    if (period) {
        for (size_t m = 0; m < moved.size(); m += (size_t)period) {
            moe_cache_zone_check(*device, *chosen, moved[m]);
        }
    }
    return moe_cache_slot_ptr(*chosen, begin);
}

// The buffer type the scheduler uses for its compute buffers when it wants them taken
// from this session's cache.
static ggml_backend_buffer_type_t moe_cache_compute_buffer_type(void * opaque, void * backend) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !backend || !ggml_backend_is_cuda((ggml_backend_t)backend)) {
        return nullptr;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return nullptr;
    }
    return ggml_backend_cuda_borrowed_buffer_type(device->logical, opaque, (ggml_backend_t)backend);
}

// Give back memory obtained from moe_cache_borrow.
static void moe_cache_give_back(void * opaque, void * backend, void * ptr) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !ptr) {
        return;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    for (auto & pool_ptr : device->pools) {
        moe_cache_pool & pool = *pool_ptr;
        for (size_t i = 0; i < pool.lent.size(); i++) {
            if (moe_cache_slot_ptr(pool, pool.lent[i].begin) != (char *)ptr) {
                continue;
            }
            const moe_cache_lent_block block = pool.lent[i];
            pool.lent.erase(pool.lent.begin() + i);
            for (int index = block.begin + block.count - 1; index >= block.begin; index--) {
                pool.free_slots.push_back(index);   // already free, unlinked and unmapped
            }
            const int region0 = pool.ranges[0].count;
            int low = pool.zone_begin < region0 ? pool.zone_begin : region0;
            for (const moe_cache_lent_block & other : pool.lent) {
                low = std::min(low, other.begin);
            }
            pool.lent_low = pool.lent.empty() ? INT_MAX : low;
            device->zone_restores++;
            MOE_CACHE_LOG("[moe-cache] CUDA%d give back: %d entries at %d to the %s pool\n",
                    device->physical, block.count, block.begin,
                    ggml_type_name((ggml_type)pool.wtype));
            return;
        }
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d give back: unknown pointer\n", device->physical);
}

// Give every lent zone of the device of 'backend' back to the cache.
static void moe_cache_restore(void * opaque, void * backend) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    bool restored = false;
    for (auto & pool_ptr : device->pools) {
        moe_cache_pool & pool = *pool_ptr;
        if (!pool.zone_lent) {
            continue;
        }
        for (int index = pool.zone_begin; index < pool.zone_end; index++) {
            pool.free_slots.push_back(index);   // already free, unlinked and unmapped
        }
        pool.zone_lent = false;
        restored = true;
    }
    if (restored) {
        device->zone_restores++;
    }
}

// ---------------------------------------------------------------------------------------------
// MoE block hosted by the GPU (default, GGML_MOE_GPU=0 turns it off; generation batches).
//
// The scheduler leaves the MoE block gate / up / swiglu / down of a layer on the GPU with its
// weights in host memory. Per layer the stream: copies the activations and the router's ids to
// mapped host memory, writes a mailbox (layer, sequence); waits (polling kernel) for the plan a host
// thread makes with the cache's structures: the rows whose three tensors are resident get their
// slots, the others -1; computes the resident rows with the cache's kernels (a row at -1 is
// skipped); waits for the host thread's results for the other rows, computed whole on the CPU
// (ggml_moe_cache.cpu_experts), and copies them in. Only kernels and copies are queued, the
// variable parts (sequence, plan) being read from memory, so the block can be captured in a graph.
// The kernels keep the pools' base addresses of the capture: a plan made while they differ (pools
// rebuilt) sends every row to the CPU.

static constexpr int moe_hosted_rows_max = 64;

struct moe_hosted_mailbox {
    uint32_t seq;
    uint32_t slot;
    uint32_t n_tokens;
    uint32_t pad[13];
};

struct moe_hosted_layer {
    const ggml_tensor * gate_w = nullptr;
    const ggml_tensor * up_w = nullptr;
    const ggml_tensor * down_w = nullptr;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int n_used = 0;
    int pool_index = -1;
    int down_pool_index = -1;
    char * base = nullptr;          // pool bases the kernels were captured with
    char * down_base = nullptr;
    // mapped host memory (host pointer, device pointer)
    float * h_act = nullptr;
    block_q8_0 * h_act8 = nullptr;   // the same activations as q8_0 blocks, the bytes of the GPU's q8_1
    int32_t * h_ids = nullptr;
    uint64_t * h_plan = nullptr;    // up, gate, down entry addresses (0: a CPU row): rows each
    float * h_res = nullptr;        // rows x n_embd
    uint32_t * h_flags = nullptr;   // [0] plan ready, [16] CPU rows ready, [24] moved expert ready (SPLIT)
    uint64_t * h_plan2 = nullptr;   // SPLIT: the moved expert's rows (up, gate, down entry addresses; 0 elsewhere)
    uint64_t * m_plan2 = nullptr;
    uint64_t * d_plan2 = nullptr;
    int split_pending = 0;          // SPLIT: its tensors still copying (session lock)
    float * h_chk = nullptr;        // DIAGNOSTIC (GGML_CUDA_MOE_SPLIT_VERIFY=1): the GPU results of the moved rows (rows x n_embd)
    float * m_chk = nullptr;
    int32_t * h_pred = nullptr;     // PREGATE: [0] stamp (publication it belongs to), [1] count, then expert ids
    int last_tokens = 1;            // PREGATE: tokens of the last publication (read by the serving thread)
    int32_t * m_pred = nullptr;
    uint64_t * m_plan = nullptr;
    float * m_res = nullptr;
    float * m_act = nullptr;
    block_q8_0 * m_act8 = nullptr;
    int32_t * m_ids = nullptr;
    uint32_t * m_flags = nullptr;
    // device memory
    uint64_t * d_plan = nullptr;
    int32_t * d_act_ids = nullptr;  // row -> token
    uint32_t * d_expect = nullptr;  // sequence number of the layer's last publication
};

struct moe_hosted_state {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    std::mutex mu;
    std::unordered_map<const void *, int> slot_of;          // gate weights -> slot
    std::vector<std::unique_ptr<moe_hosted_layer>> layers;
    moe_hosted_mailbox * h_mail = nullptr;
    moe_hosted_mailbox * m_mail = nullptr;
    uint32_t * d_seq = nullptr;
    // quantized activations of the block and of its intermediate result: the block's own tensors
    // cannot hold them, ggml-alloc may place down's output over gate's or up's once swiglu has read
    // them. Sized once for the largest block (addresses captured in graphs).
    float * d_pg_logits = nullptr;                           // PREGATE: next layer's router logits
    std::unordered_map<uint32_t, int> pg_demand;             // PREGATE: predictions per (layer, expert)
    char * d_q8_act = nullptr;
    char * d_q8_mid = nullptr;
    size_t q8_act_bytes = 0;
    size_t q8_mid_bytes = 0;
    int n_threads = 1;
    double pcie_bytes_per_us = 13600.0;                      // SPLIT: host -> device copy rate (moe_cache_pcie_bytes_per_us)
    std::vector<moe_cache_pin> pins;                         // held by the last plan
    int64_t split_moved = 0;                                 // SPLIT: experts moved to the GPU
    // served by the thread that waits for the stream (ggml_cuda_moe_hosted_wait)
    std::mutex serve_mu;
    uint32_t last_seq = 0;
    std::vector<int32_t> cpu_rows, experts;
    std::vector<float> out, act;
    std::vector<uint8_t> act8;   // the CPU rows' activations as q8_0 blocks (from h_act8)
    int64_t st_n = 0, st_gap = 0, st_gaps = 0, st_plan = 0, st_cpu = 0, st_rows = 0, t_done = 0;
    // DIAGNOSTIC (GGML_CUDA_MOE_TIMELINE): host times per layer (seq % MOE_TL_RING): publication seen, plan flag
    // written, CPU rows' flag written (ns); sums of the derived intervals; the host's stream queries
    std::vector<int64_t> tl_host;
    double tl_sum[20] = {0};
    int64_t tl_n = 0, tl_nw = 0, tl_q_n = 0, tl_q_ns = 0;
    double tl_tick_ns = 0;
    // CPU rows by what they have resident: up+gate (not down), down only, up or gate only, nothing
};

static std::mutex g_hosted_mu;
static std::unordered_map<moe_cache_session *, std::unique_ptr<moe_hosted_state>> g_hosted;

// a hosted block queued but whose CPU rows are not copied in yet (graph evaluation thread)
struct moe_hosted_pending {
    moe_hosted_layer * layer = nullptr;
    const ggml_tensor * down = nullptr;
    int rows = 0;
    bool split = false;   // SPLIT: the merge skips the rows of the moved expert
};
static thread_local moe_hosted_pending g_hosted_pending;

static bool moe_hosted_host_alloc(void ** ptr, size_t bytes) {
#if defined(GGML_USE_HIP)
    if (hipHostMalloc(ptr, bytes, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
#else
    if (cudaHostAlloc(ptr, bytes, cudaHostAllocMapped) != cudaSuccess) {
#endif
        (void) cudaGetLastError();
        *ptr = nullptr;
        return false;
    }
    memset(*ptr, 0, bytes);
    return true;
}

template <typename T>
static T * moe_hosted_device_ptr(T * host) {
    void * dev = nullptr;
    return cudaHostGetDevicePointer(&dev, host, 0) == cudaSuccess ? (T *) dev : nullptr;
}

// DIAGNOSTIC (GGML_CUDA_MOE_TIMELINE=N, HIP): a timeline of the hosted layers without a profiler, every N layers. The
// GPU writes its wall clock to mapped host memory, per layer (seq % MOE_TL_RING): [0] publication, [1] plan seen,
// [2] the CPU rows' flag seen by the merge (0: no CPU row). The host keeps its own times: publication seen, plan
// flag written, CPU rows' flag written. No clock synchronization: the relays (GPU -> host -> GPU) come out of
// differences of durations measured on one side each.
#define MOE_TL_RING 1024
static unsigned long long * g_tl_h = nullptr;   // host view
static unsigned long long * g_tl_m = nullptr;   // device view of the same memory
static int64_t moe_tl_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000 + ts.tv_nsec;
}
static __device__ __forceinline__ unsigned long long moe_tl_clock() {
#if defined(GGML_USE_HIP)
    return wall_clock64();
#else
    return 0;
#endif
}

static __device__ __forceinline__ void moe_hosted_pause() {
#if defined(GGML_USE_HIP)
    __builtin_amdgcn_s_sleep(127);
#else
    __nanosleep(2000);
#endif
}

// Entering a hosted block, in one kernel: the activations and the router's ids to mapped host memory
// (written by a kernel: a copy node of the graph would wait in a copy engine's queue for the previous
// layers, holding back the cache's fills on that engine), every block its share; the last block to
// finish (blocks done counted in d_seq[1]) publishes the layer in the mailbox, waits for the host's
// plan and copies it to device memory. (Three kernels before: stage, publish, wait_plan.)
// PREGATE: the next layer's router applied to this layer's MoE input (a block per expert), then per token
// its k best experts written to host memory with the stamp of the publication that follows
// 128 threads per expert (two experts per block): each thread issues its 5 float4 weight loads at once
// (n_embd = 2560 = 128 x 5 x 4), multiplies them by every token's activations, then warp + block sums
// the router's weights in F32 or in BF16 (converted exactly: the same products)
static __device__ __forceinline__ float4 pregate_w4(const float * w, int i) {
    return ((const float4 *) w)[i];
}
static __device__ __forceinline__ float4 pregate_w4(const uint16_t * w, int i) {
    const uint2 u = ((const uint2 *) w)[i];
    return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xFFFF0000u),
                       __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xFFFF0000u));
}

template <int max_tokens, int per_thread, typename wt>
static __global__ void k_pregate_logits(const wt * w, const float * act, float * logits, int n_embd, int n_expert, int n_tokens) {
    const int e = blockIdx.x * 2 + threadIdx.x / 128;
    const int tid = threadIdx.x % 128;
    const int lane = threadIdx.x % warpSize;
    const int warp_in_expert = tid / warpSize;
    __shared__ float part[2][4][max_tokens];
    float sum[max_tokens] = {0.0f};
    if (e < n_expert) {
        const wt * we = w + (size_t) e * n_embd;
        float4 wv[per_thread];
#pragma unroll
        for (int c = 0; c < per_thread; c++) {
            wv[c] = pregate_w4(we, tid + c * 128);
        }
#pragma unroll
        for (int t = 0; t < max_tokens; t++) {
            if (t < n_tokens) {
                const float4 * a4 = (const float4 *) (act + (size_t) t * n_embd);
#pragma unroll
                for (int c = 0; c < per_thread; c++) {
                    const float4 a = a4[tid + c * 128];
                    sum[t] += wv[c].x * a.x + wv[c].y * a.y + wv[c].z * a.z + wv[c].w * a.w;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < max_tokens; t++) {
        float v = sum[t];
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            v += __shfl_xor(v, off);
        }
        if (lane == 0) {
            part[threadIdx.x / 128][warp_in_expert][t] = v;
        }
    }
    __syncthreads();
    if (e < n_expert && tid < n_tokens) {
        const int g = threadIdx.x / 128;
        logits[tid * n_expert + e] = part[g][0][tid] + part[g][1][tid] + part[g][2][tid] + part[g][3][tid];
    }
}

// a block per token, a thread per expert (n_expert <= blockDim.x): k rounds of a warp-shuffle argmax
static __global__ void k_pregate_topk(const float * logits, int n_expert, int n_tokens, int k, const uint32_t * d_seq, int32_t * pred) {
    const int t = blockIdx.x;
    const int e = threadIdx.x;
    float v = e < n_expert ? logits[t * n_expert + e] : -INFINITY;
    __shared__ float wv[32];
    __shared__ int   wi[32];
    const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize, nwarps = blockDim.x / warpSize;
    for (int j = 0; j < k; j++) {
        float bv = v; int bi = e;
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off);
            const int   oi = __shfl_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[warp] = bv; wi[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = lane < nwarps ? wv[lane] : -INFINITY; bi = lane < nwarps ? wi[lane] : -1;
            for (int off = warpSize / 2; off > 0; off >>= 1) {
                const float ov = __shfl_xor(bv, off);
                const int   oi = __shfl_xor(bi, off);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { wi[0] = bi; pred[2 + t * k + j] = bi; }
        }
        __syncthreads();
        if (e == wi[0]) { v = -INFINITY; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        __threadfence_system();   // only thread 0 wrote to the host
    }
    if (t == 0 && threadIdx.x == 0) {
        pred[1] = n_tokens * k;
        pred[0] = (int32_t) d_seq[0];   // the publication of this layer's block (queued after it)
        __threadfence_system();
    }
}

static __global__ void k_moe_hosted_enter(const float * act, float * h_act, block_q8_0 * h_act8, int64_t n_act,
                                          const float * tk_logits, float * tk_weights, float tk_clamp, float tk_scale,
                                          int32_t * ids, int64_t ids_nb1, int32_t * h_ids, int n_used, int n_tokens,
                                          uint32_t * d_seq, uint32_t * d_expect, volatile moe_hosted_mailbox * mail,
                                          uint32_t slot, const volatile uint32_t * flag,
                                          const volatile uint64_t * m_plan, uint64_t * d_plan, int n_plan,
                                          volatile unsigned long long * tl) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_act; i += (int64_t) gridDim.x * blockDim.x) {
        h_act[i] = act[i];
    }
    // and quantized for the CPU's experts as quantize_q8_1 does (a group of 32 threads per block): the CPU starts
    // from these bytes without quantizing them again
    {
        const int lane   = threadIdx.x % QK8_0;
        const int group  = (blockIdx.x * blockDim.x + threadIdx.x) / QK8_0;
        const int groups = gridDim.x * blockDim.x / QK8_0;
        for (int64_t ib = group; ib < n_act / QK8_0; ib += groups) {
            const float xi   = act[ib * QK8_0 + lane];
            const float amax = warp_reduce_max<QK8_0>(fabsf(xi));
            const float d    = amax / 127.0f;
            const int8_t q   = amax == 0.0f ? 0 : roundf(xi / d);
            h_act8[ib].qs[lane] = q;
            if (lane == 0) {
                h_act8[ib].d = __float2half(d);
            }
        }
    }
    const int n_ids = n_used * n_tokens;
    if (tk_logits) {
        // the MoE top-k of the block's tokens (left to this kernel by ggml_cuda_topk_moe_defer): a warp per token in
        // block 0, then its ids to the host
        if (blockIdx.x == 0) {
            const int warp = threadIdx.x / WARP_SIZE;
            const int t    = warp < n_tokens ? warp : n_tokens - 1;
            topk_moe_config config;
            config.use_sigmoid       = false;
            config.use_sqrt_softplus = false;
            config.with_norm         = true;
            config.delayed_softmax   = false;
            topk_moe_warp<TOPK_MOE_HOSTED_N_EXPERT, false, true>(warp < n_tokens,
                tk_logits + (int64_t) TOPK_MOE_HOSTED_N_EXPERT * t, tk_weights + (int64_t) n_used * t,
                (int32_t *) ((char *) ids + (int64_t) t * ids_nb1), nullptr, n_used, tk_clamp, tk_scale, config,
                threadIdx.x % WARP_SIZE);
            __syncthreads();
            for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
                const int tt = i / n_used;
                h_ids[i] = *(const int32_t *) ((const char *) ids + (int64_t) tt * ids_nb1 + (int64_t) (i % n_used) * sizeof(int32_t));
            }
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ids; i += gridDim.x * blockDim.x) {
            const int t = i / n_used;
            h_ids[i] = *(const int32_t *) ((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) (i % n_used) * sizeof(int32_t));
        }
    }
    // this block's writes reach the host before it counts itself
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        last = atomicAdd(d_seq + 1, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) {
        return;
    }
    if (threadIdx.x == 0) {
        d_seq[1] = 0;   // for the next layer (stream-ordered)
        __threadfence();
        const uint32_t value = d_seq[0] + 1;
        d_seq[0]  = value;
        *d_expect = value;
        mail->slot     = slot;
        mail->n_tokens = (uint32_t) n_tokens;
        if (tl) {
            tl[(value % MOE_TL_RING) * 8 + 2] = 0;
            tl[(value % MOE_TL_RING) * 8 + 3] = 0;
            tl[(value % MOE_TL_RING) * 8 + 0] = moe_tl_clock();
        }
        __threadfence_system();
        mail->seq = value;
        __threadfence_system();
        while ((int32_t) (*flag - value) < 0) {
            moe_hosted_pause();
        }
        __threadfence_system();
        if (tl) {
            tl[(value % MOE_TL_RING) * 8 + 1] = moe_tl_clock();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_plan; i += blockDim.x) {
        d_plan[i] = m_plan[i];
    }
}

// a block per row: the rows the CPU computed (down id -1) are copied in once they are ready
// SPLIT: the moved expert arrived (or none was moved): its plan to the device
static __global__ void k_moe_split_wait(const uint32_t * d_expect, const volatile uint32_t * flag,
                                        const volatile uint64_t * m_plan2, uint64_t * d_plan2, int n_plan,
                                        volatile unsigned long long * tl) {
    if (threadIdx.x == 0) {
        const uint32_t expect = *d_expect;
        const unsigned long long t0 = tl ? moe_tl_clock() : 0;
        while ((int32_t) (*flag - expect) < 0) {
            moe_hosted_pause();
        }
        __threadfence_system();
        if (tl) {
            tl[(expect % MOE_TL_RING) * 8 + 3] = moe_tl_clock() - t0;   // the wait for the moved expert
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_plan; i += blockDim.x) {
        d_plan2[i] = m_plan2[i];
    }
}

// DIAGNOSTIC (GGML_CUDA_MOE_SPLIT_VERIFY=1): the moved rows as the GPU computed them, to the host; then flag [28]
static __global__ void k_moe_split_check(const uint32_t * d_expect, const uint64_t * d_moved_down_ids, const float * dst,
                                         float * m_chk, uint32_t * flag, int n_embd, int rows) {
    for (int row = 0; row < rows; row++) {
        if (d_moved_down_ids[row] == 0) {
            continue;
        }
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            m_chk[(size_t) row * n_embd + i] = dst[(size_t) row * n_embd + i];
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *(volatile uint32_t *) flag = *d_expect;
        __threadfence_system();
    }
}

static __global__ void k_moe_hosted_merge(const uint32_t * d_expect, const volatile uint32_t * flag,
                                          const uint64_t * d_down_ids, const uint64_t * d_moved_down_ids,
                                          const volatile float * m_res, float * dst, int n_embd,
                                          volatile unsigned long long * tl) {
    const int row = blockIdx.x;
    if (d_down_ids[row] != 0 || (d_moved_down_ids && d_moved_down_ids[row] != 0)) {
        return;   // resident, or moved to the GPU (SPLIT)
    }
    __shared__ uint32_t expect;
    if (threadIdx.x == 0) {
        expect = *d_expect;
        const unsigned long long t0 = tl ? moe_tl_clock() : 0;
        while ((int32_t) (*flag - expect) < 0) {
            moe_hosted_pause();
        }
        __threadfence_system();
        if (tl) {
            tl[(expect % MOE_TL_RING) * 8 + 4] = t0;   // the merge reached (one of its waiting blocks)
            tl[(expect % MOE_TL_RING) * 8 + 2] = moe_tl_clock();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[(size_t) row * n_embd + i] = m_res[(size_t) row * n_embd + i];
    }
}

// Unpin the rows the GPU read for the last planned layer (session lock held).
static void moe_hosted_unpin_locked(moe_hosted_state & st) {
    for (const moe_cache_pin & pin : st.pins) {
        if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots && pin.pool->slots[pin.slot].readers > 0) {
            pin.pool->slots[pin.slot].readers--;
        }
    }
    st.pins.clear();
}

// KV: the rows the hosted plan pinned for the session's last layer, unpinned (its stream is done: the caller synchronized)
static void moe_hosted_unpin_session(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    auto it = g_hosted.find(&session);
    if (it != g_hosted.end() && it->second) {
        moe_hosted_unpin_locked(*it->second);
    }
}

// Plan a layer's rows under the session lock: resident rows (three tensors valid) are pinned and
// touched, as moe_cache_fused_begin does; the others count toward admission. out: up, gate, down
// gpu ids (-1 for a row the CPU computes). Returns the rows the CPU computes. The GPU waits for the
// plan, so it is released (flag = seq) as soon as the resident rows are pinned and out is written;
// what it does not need (LRU order of the hits, admissions, statistics) follows under the same lock,
// in the same row order. The cache's structures are cold by then (the previous layer's CPU rows
// evicted them): 12.5 us for the 30 rows of a 3-token pass, 2.6 of them in that bookkeeping. The
// resident rows are found through the pools' direct index, not the hash map (no chain of dependent loads).
static int moe_hosted_plan(moe_hosted_state & st, moe_hosted_layer & layer, const int32_t * ids, int rows,
                           uint64_t * out, int32_t * cpu_rows, uint32_t * flag, uint32_t seq) {
    moe_cache_session & session = *st.session;
    moe_cache_device & device = *st.device;
    int n_cpu = 0;
    bool wake_worker = false;
    {
        std::lock_guard<std::mutex> lock(session.mu);
        moe_hosted_unpin_locked(st);

        moe_cache_pool * pool = nullptr;
        if (!session.stopping && !session.dormant && !device.dead.load() &&
            layer.pool_index >= 0 && layer.pool_index < (int) device.pools.size()) {
            pool = device.pools[layer.pool_index].get();
            if (!pool || !pool->slab || pool->gpu_base != layer.base) {
                pool = nullptr;
            }
        }
        const void * key = layer.gate_w->data;
        const int64_t n_expert = layer.gate_w->ne[2];
        // the layer's direct index once, then one load per row (independent of the other rows)
        const int32_t * direct = nullptr;
        int n_direct = 0;
        if (pool) {
            const auto d = pool->direct.find(key);
            if (d != pool->direct.end()) {
                direct = d->second.data();
                n_direct = (int) d->second.size();
            }
        }
        int slots[moe_hosted_rows_max];
        for (int row = 0; row < rows; row++) {
            out[row] = out[rows + row] = out[2 * rows + row] = 0;
            slots[row] = -1;
            const int32_t expert = ids[row];
            if (!pool || expert < 0 || expert >= n_expert) {
                continue;
            }
            const int s = expert < n_direct ? direct[expert] : -1;
            if (s >= 0 && pool->slots[s].state == moe_cache_slot_state::valid) {
                slots[row] = s;
                pool->slots[s].readers++;
                st.pins.push_back({pool, s});
                const uint64_t entry = moe_cache_entry_addr(*pool, s);
                for (int k = 0; k < MOE_ROLES; k++) {
                    out[k * rows + row] = entry + pool->role_offset[k];
                }
            }
        }
        __atomic_store_n(flag, seq, __ATOMIC_RELEASE);
        if (!st.tl_host.empty()) {
            st.tl_host[(seq % MOE_TL_RING) * 3 + 1] = moe_tl_now_ns();
        }

        // admissions: inserts_per_plan / 2 experts per plan, the rate the pools per tensor shape had (a budget of
        // inserts_per_plan tensors for the gate / up pool, two tensors an expert)
        int inserts = std::max(1, session.config.inserts_per_plan / 2);
        for (int row = 0; row < rows; row++) {
            const int32_t expert = ids[row];
            if (!pool || expert < 0 || expert >= n_expert) {
                cpu_rows[n_cpu++] = row;
                continue;
            }
            if (slots[row] >= 0) {
                device.hits++;
                device.freq_demands++;
                moe_cache_slot & slot = pool->slots[slots[row]];
                slot.freq += slot.freq < (session.config.lfu ? session.config.lfu_cap : std::numeric_limits<uint16_t>::max());
                moe_cache_lru_remove(*pool, slots[row]);
                moe_cache_lru_push_back(*pool, slots[row]);
                continue;
            }
            cpu_rows[n_cpu++] = row;
            if (moe_cache_lookup_or_queue_locked(session, device, *pool, layer.pool_index, key, expert, pool->expert_size,
                                                 inserts, wake_worker) >= 0) {
                device.hits++;
            } else {
                device.misses++;
            }
        }
        device.nodes++;
        moe_cache_freq_age_locked(session, device);
    }
    if (wake_worker) {
        session.cv.notify_all();
    }
    return n_cpu;
}

// SPLIT (GGML_CUDA_MOE_SPLIT=0 disables): in a pass of 2 tokens or more, one or two experts of the CPU rows are moved to
// the GPU when that ends the MoE window sooner. Their missing tensors take cache entries (an admission) and go to the
// front of the fill queue; the last one to arrive raises the layer's flag, which a kernel queued after the resident rows
// waits for. The window ends with the later of the CPU rows and the GPU's work: the resident rows, the wait for the
// moved experts, then the work queued behind it; costs: see moe_split_cpu_fixed_us. Returns the CPU rows left.
// SPLIT: the dispatcher's model of a 2 - 8 token hosted layer, fitted on ~15 000 layers of 3-token MTP verifications in a
// serving process (replay bench, 32k context; the residual standard deviation of each fit in brackets):
// - CPU time of the missed rows: 2.0 + 55.8 per distinct expert + 14.5 per row + 5.2 per tensor copied meanwhile (the
//   copy reads the same RAM) [21.6 us];
// - arrival of the moved experts' tensors (one copy at a time): 9.1 + per tensor 11.4 + bytes / the link's copy rate
//   [16.6 us]; the rate is the link's (moe_cache_pcie_bytes_per_us) times 86 % (13.6 GB/s on PCIe 3.0 x16);
// - GPU work of the window: before the wait for the moved experts (the resident rows) 23.3 + 2.97 per resident row, and
//   after it (the moved rows, the pregate and the shared expert, queued behind the wait) 66.4 + 1.5 per moved row
//   [3.4 us]; in all 89.7 + 2.97 per resident row + 3.46 per moved row [9.5 us].
// Copies of expert tensors alone, for comparison (tests/pciebench): 6.2 us + bytes / 14.1 GB/s.
static constexpr double moe_split_cpu_fixed_us = 2.0, moe_split_cpu_expert_us = 55.8, moe_split_cpu_row_us = 14.5;
static constexpr double moe_split_cpu_copy_us = 5.2;
static constexpr double moe_split_copy_start_us = 9.1, moe_split_copy_tensor_us = 11.4;
static constexpr double moe_cache_pcie_copy_efficiency = 13.6 / (8.0 * 16 * 128.0 / 130.0 / 8.0);
static constexpr double moe_split_gpu_before_us = 23.3, moe_split_gpu_row_us = 2.97;
static constexpr double moe_split_gpu_after_us = 66.4, moe_split_gpu_moved_row_us = 1.5;

static bool moe_split_enabled() {
    static const bool enabled = !getenv("GGML_CUDA_MOE_SPLIT") || atoi(getenv("GGML_CUDA_MOE_SPLIT")) != 0;
    return enabled;
}

static int moe_hosted_split(moe_hosted_state * st, moe_hosted_layer * layer, int rows, int n_tokens, uint32_t seq,
                            int32_t * cpu_rows, int n_cpu) {
    if (!moe_split_enabled() || n_tokens < 2) {
        return n_cpu;
    }
    uint64_t * plan2 = layer->h_plan2;
    for (int i = 0; i < 3 * rows; i++) {
        plan2[i] = 0;
    }
    int jobs = 0;
    if (n_cpu > 0) {
        moe_cache_session & session = *st->session;
        moe_cache_device & device = *st->device;
        std::lock_guard<std::mutex> lock(session.mu);
        const bool usable = !session.stopping && !session.dormant && !device.dead.load() &&
            layer->pool_index >= 0 && layer->pool_index < (int) device.pools.size() &&
            layer->down_pool_index >= 0 && layer->down_pool_index < (int) device.pools.size();
        if (usable) {
            moe_cache_pool * pool = device.pools[layer->pool_index].get();
            const void * key = layer->gate_w->data;
            const int idx = layer->pool_index;
            const double rate = st->pcie_bytes_per_us;   // bytes per microsecond
            // a candidate: an expert of the CPU rows, its entry (absent: -1, or copying), the time of its three copies one
            // after another
            struct cand { int32_t e; int rows; int slot; double copies; int tensors; };
            std::vector<cand> c;
            for (int j = 0; j < n_cpu; j++) {
                const int32_t e = layer->h_ids[cpu_rows[j]];
                bool seen = false;
                for (auto & x : c) {
                    if (x.e == e) { x.rows++; seen = true; }
                }
                if (!seen) {
                    cand x = {e, 1, -1, 0.0, MOE_ROLES};
                    const auto f = pool->map.find(moe_cache_key{key, e});
                    if (f != pool->map.end()) {
                        x.slot = f->second;
                    }
                    for (int k = 0; k < MOE_ROLES; k++) {
                        x.copies += moe_split_copy_tensor_us + pool->role_bytes[k] / rate;
                    }
                    c.push_back(x);
                }
            }
            // the window's end: the later of the CPU rows and the GPU's work (the resident rows, the wait for the moved
            // experts, then the work queued behind it)
            const auto cpu_time = [](int d, int r, int tensors) {
                return d > 0 ? moe_split_cpu_fixed_us + moe_split_cpu_expert_us * d + moe_split_cpu_row_us * r +
                               moe_split_cpu_copy_us * tensors : 0.0;
            };
            const double gpu_before = moe_split_gpu_before_us + moe_split_gpu_row_us * (rows - n_cpu);
            const auto window = [&](int d, int r, int moved_rows, int tensors, double copies) {
                const double arrival = tensors > 0 ? moe_split_copy_start_us + copies : 0.0;
                const double gpu_after = moe_split_gpu_after_us + moe_split_gpu_moved_row_us * moved_rows;
                return std::max(cpu_time(d, r, tensors), std::max(gpu_before, arrival) + gpu_after);
            };
            const int D = (int) c.size();
            const double all = window(D, n_cpu, 0, 0, 0.0);
            // the best single expert or pair, if it ends the window sooner
            std::vector<int> best;
            double best_gain = 0.0;
            for (int i = 0; i < D; i++) {
                const double crit = window(D - 1, n_cpu - c[i].rows, c[i].rows, c[i].tensors, c[i].copies);
                if (all - crit > best_gain) {
                    best_gain = all - crit;
                    best = {i};
                }
            }
            for (int i = 0; i < D; i++) {
                for (int j = i + 1; j < D; j++) {
                    const double crit = window(D - 2, n_cpu - c[i].rows - c[j].rows, c[i].rows + c[j].rows,
                                               c[i].tensors + c[j].tensors, c[i].copies + c[j].copies);
                    if (all - crit > best_gain) {
                        best_gain = all - crit;
                        best = {i, j};
                    }
                }
            }
            static const bool verify = getenv("GGML_CUDA_MOE_SPLIT_VERIFY") && atoi(getenv("GGML_CUDA_MOE_SPLIT_VERIFY")) > 0;
            std::vector<moe_cache_job> moved;
            std::vector<int32_t> moved_experts;
            // an expert copying already (in flight, or a prefetch) cannot be waited for: not moved; a queued admission
            // is promoted, an absent expert takes an entry (an admission)
            for (int b : best) {
                cand & x = c[b];
                moe_cache_job job;
                bool ok = false;
                if (x.slot >= 0) {
                    for (auto it = device.queue.begin(); it != device.queue.end(); ++it) {
                        if (it->key == moe_cache_key{key, x.e} && it->slot == x.slot) {
                            job = *it;
                            device.queue.erase(it);
                            ok = true;
                            break;
                        }
                    }
                } else {
                    int inserts_left = 1;
                    bool wake = false;
                    if (moe_cache_enqueue_locked(session, device, *pool, idx, key, x.e, pool->expert_size, inserts_left, wake,
                                                 true, false, false)) {
                        job = device.queue.back();
                        device.queue.pop_back();
                        x.slot = job.slot;
                        ok = true;
                    }
                }
                if (!ok) {
                    continue;
                }
                pool->slots[x.slot].readers++;
                st->pins.push_back({pool, x.slot});
                moved.push_back(job);
                const uint64_t entry = moe_cache_entry_addr(*pool, x.slot);
                for (int j = 0; j < n_cpu; j++) {
                    const int row = cpu_rows[j];
                    if (layer->h_ids[row] == x.e) {
                        for (int k = 0; k < MOE_ROLES; k++) {
                            plan2[k * rows + row] = entry + pool->role_offset[k];
                        }
                    }
                }
                moved_experts.push_back(x.e);
            }
            if (!moved.empty()) {
                for (auto & j : moved) {
                    j.split_flag = layer->h_flags + 24;
                    j.split_pending = &layer->split_pending;
                    j.split_seq = seq;
                }
                for (auto it = moved.rbegin(); it != moved.rend(); ++it) {
                    device.queue.push_front(*it);
                }
                jobs = (int) moved.size();
                layer->split_pending = jobs;
                int left = 0;
                for (int j = 0; j < n_cpu; j++) {
                    const int row = cpu_rows[j];
                    const bool gone = std::find(moved_experts.begin(), moved_experts.end(), layer->h_ids[row]) != moved_experts.end();
                    if (!gone || verify) {
                        cpu_rows[left++] = row;   // with VERIFY computed on both sides, compared
                    }
                }
                n_cpu = left;
                st->split_moved += (int64_t) moved_experts.size();
                session.cv.notify_all();
            }
        }
    }
    if (jobs == 0) {
        __atomic_store_n(&layer->h_flags[24], seq, __ATOMIC_RELEASE);
    }
    return n_cpu;
}

// One publication of the stream: plan the layer, raise the plan flag, compute the missed rows on
// the CPU (with this thread's OpenMP team, the one of the graph's other CPU work), raise the rows
// flag. The cache is suspended meanwhile: the thread may be inside a session of the scheduler.
// PREGATE: after a layer's CPU rows, the experts predicted for the next layer that are not resident are
// queued for prefetch (at least PREGATE_PF_T predictions, default 2; not for the layers in
// PREGATE_SKIP, default the poorly predicted 1,2,3,15,18,23,31,32,39), and the prefetch window opens
static void moe_hosted_pregate(moe_hosted_state * st, uint32_t slot, moe_hosted_layer * layer, uint32_t seq) {
    static const int pf_t = getenv("GGML_CUDA_MOE_PREGATE_PF_T") ? atoi(getenv("GGML_CUDA_MOE_PREGATE_PF_T")) : 2;
    static const std::vector<int> skip = [] {
        std::vector<int> v;
        const char * e = getenv("GGML_CUDA_MOE_PREGATE_SKIP");
        std::string str = e ? e : "1,2,3,15,18,23,31,32,39";
        size_t p = 0;
        while (p < str.size()) {
            const size_t q = str.find(',', p);
            v.push_back(atoi(str.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
            if (q == std::string::npos) break;
            p = q + 1;
        }
        return v;
    }();
    moe_hosted_layer * next = nullptr;
    {
        std::lock_guard<std::mutex> lock(st->mu);
        next = slot + 1 < st->layers.size() ? st->layers[slot + 1].get() : nullptr;
    }
    moe_cache_session & session = *st->session;
    moe_cache_device & device = *st->device;
    std::lock_guard<std::mutex> lock(session.mu);
    const int32_t * pred = layer->h_pred;
    device.pg_cands.clear();
    device.pg_next = 0;
    if (next && std::find(skip.begin(), skip.end(), (int) slot + 1) == skip.end() &&
            (uint32_t) __atomic_load_n(&pred[0], __ATOMIC_ACQUIRE) == seq && next->pool_index >= 0 && next->down_pool_index >= 0) {
        const int n = pred[1];
        const int k = std::max(1, n / std::max(1, layer->last_tokens));
        const void * key = next->gate_w->data;
        moe_cache_pool & pool = *device.pools[next->pool_index];
        // rank-major: the best prediction of every token first
        for (int j = 0; j < k; j++) {
            for (int t = 0; t * k + j < n; t++) {
                const int32_t e = pred[2 + t * k + j];
                if (e < 0) {
                    continue;
                }
                device.pregate_predicted++;
                if (pool.map.count(moe_cache_key{key, e})) {
                    continue;   // valid or copying
                }
                int & d = st->pg_demand[(slot + 1) << 16 | (uint32_t) e];
                if (++d < pf_t) {
                    continue;
                }
                d = 0;
                bool dup = false;
                for (const auto & c : device.pg_cands) {
                    dup |= c.tensor == key && c.expert == e;
                }
                if (!dup) {
                    device.pg_cands.push_back({next->pool_index, key, e, pool.expert_size});
                    device.pregate_enqueued++;
                }
            }
        }
    }
    device.pregate_window = true;
    session.cv.notify_all();
}

// DIAGNOSTIC (GGML_CUDA_MOE_TIMELINE): called once the layer seq's CPU rows are handed back; accounts the layer
// seq - 1 (its merge is done: the GPU published seq after it) with the publication of seq
static void moe_tl_account(moe_hosted_state * st, uint32_t seq) {
    static const int every = atoi(getenv("GGML_CUDA_MOE_TIMELINE"));
    const uint32_t p = seq - 1;
    int64_t * h  = st->tl_host.data();
    int64_t * hp = h + (p % MOE_TL_RING) * 3, * hs = h + (seq % MOE_TL_RING) * 3;
    hs[2] = moe_tl_now_ns();
    const volatile unsigned long long * gp = g_tl_h + (p % MOE_TL_RING) * 8, * gs = g_tl_h + (seq % MOE_TL_RING) * 8;
    const double k = st->tl_tick_ns;
    const unsigned long long ge1 = gp[0], ge2 = gp[1], gm = gp[2], gs1 = gs[0];
    const int64_t gap_host = hs[0] - hp[2];
    if (hp[0] == 0 || hp[1] == 0 || hp[2] == 0 || ge1 == 0 || ge2 < ge1 || gap_host <= 0 || gap_host > 2000000) {
        return;
    }
    const double plan_rt   = (double) (ge2 - ge1) * k;       // GPU: publication -> plan seen
    const double plan_host = (double) (hp[1] - hp[0]);       // host: publication seen -> plan flag
    st->tl_sum[0] += plan_rt;
    st->tl_sum[1] += plan_host;
    st->tl_n++;
    if (gm != 0 && gm >= ge2 && gs1 >= gm) {
        const double win_gpu  = (double) (gm - ge2) * k;     // GPU: plan seen -> CPU rows' flag seen
        const double win_host = (double) (hp[2] - hp[1]);    // host: plan flag -> CPU rows' flag
        const double seg_gpu  = (double) (gs1 - gm) * k;     // GPU: CPU rows' flag seen -> next publication
        st->tl_sum[2] += win_gpu;
        st->tl_sum[3] += win_host;
        st->tl_sum[4] += seg_gpu;
        st->tl_sum[5] += (double) gap_host;                  // host: CPU rows' flag -> next publication seen
        st->tl_nw++;
        // the GPU later than the CPU at the merge (relays included): waiting for the moved expert's copy, or not
        const double late = win_gpu - win_host;
        const double copy_wait = (double) gp[3] * k;
        if (late > 10e3) {
            const int c = copy_wait > 5e3 ? 0 : 1;
            // the GPU's work in the window: plan seen -> merge reached, without the wait for the moved expert
            const double gpu_work = (double) (gp[4] - ge2) * k - copy_wait;
            st->tl_sum[14 + c] += gpu_work;
            st->tl_sum[6 + c] += 1;
            st->tl_sum[8 + c] += late;
            st->tl_sum[10 + c] += copy_wait;
        }
        st->tl_sum[12] += late > 0 ? late : 0;
        st->tl_sum[13] += copy_wait;
        st->tl_sum[16] += (double) (gp[4] - ge2) * k - copy_wait;
        st->tl_sum[17] += win_host;
    }
    if (st->tl_n % every == 0) {
        const double n = (double) st->tl_n, nw = std::max<double>(1, (double) st->tl_nw);
        const double plan_rt = st->tl_sum[0] / n, plan_host = st->tl_sum[1] / n;
        const double win_gpu = st->tl_sum[2] / nw, win_host = st->tl_sum[3] / nw, seg = st->tl_sum[4] / nw, gap = st->tl_sum[5] / nw;
        GGML_LOG_INFO("moe timeline: %lld layers (%lld with CPU rows), us per layer: plan round trip GPU %.1f, plan host %.1f "
                "-> relays publication + plan %.1f; window GPU %.1f, CPU host %.1f -> relay rows - relay plan %.1f; "
                "GPU segment %.1f, host gap %.1f -> relays rows + publication %.1f; stream query %.2f us (%lld)\n",
                (long long) st->tl_n, (long long) st->tl_nw, plan_rt / 1e3, plan_host / 1e3, (plan_rt - plan_host) / 1e3,
                win_gpu / 1e3, win_host / 1e3, (win_gpu - win_host) / 1e3, seg / 1e3, gap / 1e3, (gap - seg) / 1e3,
                st->tl_q_n ? (double) st->tl_q_ns / st->tl_q_n / 1e3 : 0.0, (long long) st->tl_q_n);
        const double * a = st->tl_sum;
        GGML_LOG_INFO("moe timeline: GPU late at the merge %.1f us per layer, moved expert wait %.1f us per layer; layers late "
                "> 10 us: waiting for the moved expert %.1f %% (late %.1f us, copy wait %.1f us), otherwise %.1f %% (late %.1f us)\n",
                a[12] / nw / 1e3, a[13] / nw / 1e3, 100 * a[6] / nw, a[6] ? a[8] / a[6] / 1e3 : 0.0, a[6] ? a[10] / a[6] / 1e3 : 0.0,
                100 * a[7] / nw, a[7] ? a[9] / a[7] / 1e3 : 0.0);
        GGML_LOG_INFO("moe timeline: GPU work in the window (plan seen -> merge, without the moved expert wait) %.1f us, "
                "CPU window %.1f us; in the late layers without copy wait: GPU work %.1f us\n",
                a[16] / nw / 1e3, a[17] / nw / 1e3, a[7] ? a[15] / a[7] / 1e3 : 0.0);
    }
}

static void moe_hosted_serve_one(moe_hosted_state * st, uint32_t seq) {
    // DIAGNOSTIC (GGML_CUDA_MOE_HOSTED_STATS=N): every N layers, per layer: the GPU's time between the
    // CPU rows handed back and the next publication (merge + the next layer's GPU work; long gaps,
    // between graphs, left out), the plan, the CPU rows, the CPU rows per layer
    static const int stats_every = getenv("GGML_CUDA_MOE_HOSTED_STATS") ? atoi(getenv("GGML_CUDA_MOE_HOSTED_STATS")) : 0;
    const int64_t t_arrive = stats_every > 0 ? ggml_time_us() : 0;
    if (!st->tl_host.empty()) {
        st->tl_host[(seq % MOE_TL_RING) * 3 + 0] = moe_tl_now_ns();
    }
    if (stats_every > 0 && st->t_done > 0 && t_arrive - st->t_done < 2000) {
        st->st_gap += t_arrive - st->t_done;
        st->st_gaps++;
    }
    const int pregate_k = moe_pregate_k();
    if (pregate_k > 0) {
        std::lock_guard<std::mutex> lock(st->session->mu);
        st->device->pregate_window = false;   // this layer's exchanges with the host begin: its prefetches
        st->device->pg_cands.clear();         // not started missed their window
        st->device->pg_next = 0;
    }
    const uint32_t slot = __atomic_load_n(&st->h_mail->slot, __ATOMIC_RELAXED);
    const int n_tokens = (int) __atomic_load_n(&st->h_mail->n_tokens, __ATOMIC_RELAXED);
    moe_hosted_layer * layer;
    {
        std::lock_guard<std::mutex> lock(st->mu);
        layer = slot < st->layers.size() ? st->layers[slot].get() : nullptr;
    }
    if (!layer) {
        return;   // cannot happen: the stream publishes registered slots only
    }
    std::vector<int32_t> & cpu_rows = st->cpu_rows;
    std::vector<int32_t> & ids = st->experts;
    std::vector<float> & out = st->out;
    std::vector<float> & act = st->act;
    cpu_rows.resize(moe_hosted_rows_max);
    const int rows = layer->n_used * n_tokens;
    layer->last_tokens = n_tokens;
    int n_cpu = moe_hosted_plan(*st, *layer, layer->h_ids, rows, layer->h_plan, cpu_rows.data(), &layer->h_flags[0], seq);
    n_cpu = moe_hosted_split(st, layer, rows, n_tokens, seq, cpu_rows.data(), n_cpu);
    const int64_t t_plan = stats_every > 0 ? ggml_time_us() : 0;

    // the CPU rows in one call, one column each (k = 1) with its token's activations: exactly the
    // missed rows (a k-per-token layout padded the tokens with fewer misses: 8.1 rows computed for
    // 5.9, CPU time +12 % per layer); MUL_MAT_ID groups the rows of an expert, which is read once
    if (n_cpu > 0) {
        ids.resize(n_cpu);
        act.resize((size_t) n_cpu * layer->n_embd);
        const size_t q8_row = (size_t) (layer->n_embd / QK8_0) * sizeof(block_q8_0);
        st->act8.resize((size_t) n_cpu * q8_row);
        for (int j = 0; j < n_cpu; j++) {
            const int t = cpu_rows[j] / layer->n_used;
            ids[j] = layer->h_ids[cpu_rows[j]];
            memcpy(act.data() + (size_t) j * layer->n_embd, layer->h_act + (size_t) t * layer->n_embd, layer->n_embd * sizeof(float));
            memcpy(st->act8.data() + (size_t) j * q8_row, (const char *) layer->h_act8 + (size_t) t * q8_row, q8_row);
        }
        out.resize((size_t) n_cpu * layer->n_embd);
        g_session_suppressed++;
        ggml_moe_cache.cpu_experts(layer->gate_w, layer->up_w, layer->down_w, act.data(), st->act8.data(), ids.data(), 1, n_cpu,
                                   out.data(), st->n_threads);
        g_session_suppressed--;
        for (int j = 0; j < n_cpu; j++) {
            memcpy(layer->h_res + (size_t) cpu_rows[j] * layer->n_embd, out.data() + (size_t) j * layer->n_embd,
                   layer->n_embd * sizeof(float));
        }
    }
    __atomic_store_n(&layer->h_flags[16], seq, __ATOMIC_RELEASE);
    if (!st->tl_host.empty()) {
        moe_tl_account(st, seq);
    }
{
        // DIAGNOSTIC (GGML_CUDA_MOE_SPLIT_VERIFY=1): the moved rows, CPU against GPU (relative L2 error per row)
        static const bool verify = getenv("GGML_CUDA_MOE_SPLIT_VERIFY") && atoi(getenv("GGML_CUDA_MOE_SPLIT_VERIFY")) > 0;
        static int64_t vn = 0, vbad = 0;
        static double vsum = 0, vmax = 0;
        if (verify && n_tokens >= 2) {
            bool any = false;
            for (int r = 0; r < rows; r++) any |= layer->h_plan2[2 * rows + r] != 0;
            if (any) {
                while ((int32_t) (__atomic_load_n(&layer->h_flags[28], __ATOMIC_ACQUIRE) - seq) < 0) {
#if defined(__x86_64__)
                    __builtin_ia32_pause();
#endif
                }
                for (int r = 0; r < rows; r++) {
                    if (layer->h_plan2[2 * rows + r] == 0) continue;
                    double d2 = 0, n2 = 0;
                    for (int i = 0; i < layer->n_embd; i++) {
                        const double a = layer->h_res[(size_t) r * layer->n_embd + i], b = layer->h_chk[(size_t) r * layer->n_embd + i];
                        d2 += (a - b) * (a - b); n2 += a * a;
                    }
                    const double rel = sqrt(d2 / std::max(n2, 1e-30));
                    vn++; vsum += rel; vmax = std::max(vmax, rel); vbad += rel > 0.2;
                    if (rel > 0.2) {
                        double g2 = 0;
                        for (int i = 0; i < layer->n_embd; i++) g2 += (double) layer->h_chk[(size_t) r * layer->n_embd + i] * layer->h_chk[(size_t) r * layer->n_embd + i];
                        GGML_LOG_INFO("split verify: slot %u row %d expert %d relative error %.3f, |cpu| %.3e |gpu| %.3e\n",
                                      slot, r, layer->h_ids[r], rel, sqrt(n2), sqrt(g2));
                    }
                }
                if (vn % 2000 < 4) {
                    GGML_LOG_INFO("split verify: %lld rows, relative error mean %.4f, max %.4f, above 0.2: %lld\n",
                                  (long long) vn, vsum / vn, vmax, (long long) vbad);
                }
            }
        }
    }
    if (pregate_k > 0) {
        moe_hosted_pregate(st, slot, layer, seq);
    }
    if (stats_every > 0) {
        st->t_done = ggml_time_us();
        st->st_plan += t_plan - t_arrive;
        st->st_cpu += st->t_done - t_plan;
        st->st_rows += n_cpu;
        if (++st->st_n % stats_every == 0) {
            const double n = (double) stats_every;
            if (pregate_k > 0) {
                std::lock_guard<std::mutex> lock(st->session->mu);
                GGML_LOG_INFO("pregate: predicted %lld, candidates %lld, copied %lld\n",
                        (long long) st->device->pregate_predicted, (long long) st->device->pregate_enqueued,
                        (long long) st->device->pregate_served);
            }
            GGML_LOG_INFO("moe hosted: %lld experts moved to the GPU\n", (long long) st->split_moved);
            GGML_LOG_INFO("moe hosted: %lld layers: GPU between layers %.1f us, plan %.1f us, CPU rows %.1f us (%.2f rows)\n",
                    (long long) st->st_n, st->st_gaps ? (double) st->st_gap / st->st_gaps : 0.0,
                    (double) st->st_plan / st->st_n, (double) st->st_cpu / st->st_n, (double) st->st_rows / st->st_n);
            {
                std::lock_guard<std::mutex> lock(st->session->mu);
                moe_cache_log_stats(*st->device);
            }
        }
    }
}

// Wait for the stream (or the event) like cudaStreamSynchronize / cudaEventSynchronize, answering the
// hosted MoE blocks it publishes meanwhile: the thread that waits for the GPU computes the missed
// experts. Returns false when no block is hosted (the caller synchronizes as usual).
bool ggml_cuda_moe_hosted_wait(void * stream_ptr, void * event_ptr) {
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    cudaEvent_t event = (cudaEvent_t) event_ptr;
    moe_hosted_state * st;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        if (g_hosted.empty()) {
            return false;
        }
        st = g_hosted.begin()->second.get();
    }
    std::unique_lock<std::mutex> serving(st->serve_mu, std::try_to_lock);
    const bool tl_on = !st->tl_host.empty();
    for (;;) {
        const int64_t tq = tl_on ? moe_tl_now_ns() : 0;
        const cudaError_t done = event ? cudaEventQuery(event) : cudaStreamQuery(stream);
        if (tl_on) {
            st->tl_q_ns += moe_tl_now_ns() - tq;
            st->tl_q_n++;
        }
        if (done != cudaErrorNotReady) {
            CUDA_CHECK(done);
            if (!event) {
                // the stream is done, the last hosted layer's rows too: unpin them (a plan unpins the
                // previous layer's; left pinned after a graph, they made the cache refuse to lend its
                // zones to the next prefill, which then ran without prefetch)
                std::lock_guard<std::mutex> lock(st->session->mu);
                moe_hosted_unpin_locked(*st);
            }
            return true;
        }
        const uint32_t seq = __atomic_load_n(&st->h_mail->seq, __ATOMIC_ACQUIRE);
        if (serving.owns_lock() && seq != st->last_seq) {
            st->last_seq = seq;
            moe_hosted_serve_one(st, seq);
            continue;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        if (!serving.owns_lock()) {
            serving.try_lock();
        }
    }
}

// SPLIT: the host -> device bytes per microsecond a copy of an expert gets. The link's rate is the smallest of the maximum
// link capabilities on the path from the device to the root port: the current ones drop while the link idles, and a
// Radeon's own link to the switch inside the card reads 16 GT/s x16 in any slot. Times the encoding (8b/10b up to
// 5 GT/s, 128b/130b above), times the share a copy gets (moe_cache_pcie_copy_efficiency). 0 if the path cannot be read.
static double moe_cache_pcie_bytes_per_us(int physical, std::string & desc) {
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), physical) != cudaSuccess) {
        (void) cudaGetLastError();
        return 0.0;
    }
    std::string id(bus);
    for (char & c : id) {
        c = (char) tolower((unsigned char) c);
    }
    char real[PATH_MAX];
    if (!realpath(("/sys/bus/pci/devices/" + id).c_str(), real)) {
        return 0.0;
    }
    double best = 0.0, best_gt = 0.0;
    int best_width = 0;
    for (std::string p = real; p.size() > 1; p = p.substr(0, p.rfind('/'))) {
        double gt = 0.0;
        int width = 0;
        FILE * f = fopen((p + "/max_link_speed").c_str(), "r");
        if (!f) {
            break;   // above the root port
        }
        const bool ok_speed = fscanf(f, "%lf", &gt) == 1;
        fclose(f);
        f = fopen((p + "/max_link_width").c_str(), "r");
        const bool ok_width = f && fscanf(f, "%d", &width) == 1;
        if (f) {
            fclose(f);
        }
        if (!ok_speed || !ok_width || gt <= 0.0 || width <= 0) {
            continue;
        }
        const double gbs = gt * width * (gt >= 8.0 ? 128.0 / 130.0 : 8.0 / 10.0) / 8.0;
        if (best == 0.0 || gbs < best) {
            best = gbs;
            best_gt = gt;
            best_width = width;
        }
    }
    if (best == 0.0) {
        return 0.0;
    }
    char text[128];
    snprintf(text, sizeof(text), "%.1f GT/s x%d on the path: %.2f GB/s, %.2f GB/s for a copy", best_gt, best_width, best,
             best * moe_cache_pcie_copy_efficiency);
    desc = text;
    return best * moe_cache_pcie_copy_efficiency * 1000.0;   // 1 GB/s = 1000 bytes per microsecond
}

static moe_hosted_state * moe_hosted_state_for(moe_cache_session & session, moe_cache_device & device) {
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    auto & entry = g_hosted[&session];
    if (entry) {
        return entry.get();
    }
    std::unique_ptr<moe_hosted_state> st(new moe_hosted_state());
    st->session = &session;
    st->device = &device;
    const char * threads = getenv("GGML_CUDA_MOE_HOSTED_THREADS");
    const int hw = (int) std::thread::hardware_concurrency();
    st->n_threads = threads ? std::max(1, atoi(threads)) : std::max(1, hw / 2);
    {
        std::string desc;
        const double rate = moe_cache_pcie_bytes_per_us(device.physical, desc);
        if (rate > 0.0) {
            st->pcie_bytes_per_us = rate;
            MOE_CACHE_LOG("[moe-cache] CUDA%d PCIe: %s\n", device.physical, desc.c_str());
        } else {
            MOE_CACHE_LOG("[moe-cache] CUDA%d PCIe: link not readable, %.1f GB/s assumed for a copy\n", device.physical,
                    st->pcie_bytes_per_us / 1000.0);
        }
    }
    if (!moe_hosted_host_alloc((void **) &st->h_mail, sizeof(moe_hosted_mailbox)) ||
        !(st->m_mail = moe_hosted_device_ptr(st->h_mail)) ||
        cudaMalloc((void **) &st->d_seq, 2 * sizeof(uint32_t)) != cudaSuccess ||   // [0] seq, [1] blocks done (stage)
        cudaMemset(st->d_seq, 0, 2 * sizeof(uint32_t)) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    if (getenv("GGML_CUDA_MOE_TIMELINE") && atoi(getenv("GGML_CUDA_MOE_TIMELINE")) > 0) {
        int rate_khz = 0;
#if defined(GGML_USE_HIP)
        (void) hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, device.logical);
#endif
        if (rate_khz > 0 && (g_tl_h || moe_hosted_host_alloc((void **) &g_tl_h, MOE_TL_RING * 8 * sizeof(unsigned long long)))) {
            g_tl_m = moe_hosted_device_ptr(g_tl_h);
            st->tl_tick_ns = 1e6 / rate_khz;
            st->tl_host.assign(MOE_TL_RING * 3, 0);
            GGML_LOG_INFO("moe timeline: GPU wall clock %d kHz\n", rate_khz);
        }
    }
    entry = std::move(st);
    return entry.get();
}

static void moe_hosted_shutdown(moe_cache_session * session) {
    std::unique_ptr<moe_hosted_state> st;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        auto it = g_hosted.find(session);
        if (it == g_hosted.end()) {
            return;
        }
        st = std::move(it->second);
        g_hosted.erase(it);
    }
    for (auto & layer : st->layers) {
        for (void * h : {(void *) layer->h_act, (void *) layer->h_act8, (void *) layer->h_ids, (void *) layer->h_plan, (void *) layer->h_res,
                         (void *) layer->h_flags}) {
            if (h) cudaFreeHost(h);
        }
        for (void * d : {(void *) layer->d_plan, (void *) layer->d_act_ids, (void *) layer->d_expect}) {
            if (d) cudaFree(d);
        }
    }
    if (st->h_mail) cudaFreeHost(st->h_mail);
    if (st->d_seq) cudaFree(st->d_seq);
    if (st->d_q8_act) cudaFree(st->d_q8_act);
    if (st->d_q8_mid) cudaFree(st->d_q8_mid);
    // the pins of the last plan go with the pools the session is about to free
}

static moe_hosted_layer * moe_hosted_register(moe_hosted_state & st, moe_cache_device & device,
                                              const ggml_tensor * gate_w, const ggml_tensor * up_w,
                                              const ggml_tensor * down_w, int n_used, int & slot) {
    std::lock_guard<std::mutex> lock(st.mu);
    const auto found = st.slot_of.find(gate_w->data);
    if (found != st.slot_of.end()) {
        slot = found->second;
        return st.layers[slot].get();
    }
    std::unique_ptr<moe_hosted_layer> layer(new moe_hosted_layer());
    layer->gate_w = gate_w;
    layer->up_w = up_w;
    layer->down_w = down_w;
    layer->n_embd = gate_w->ne[0];
    layer->n_ff = gate_w->ne[1];
    layer->n_used = n_used;
    {
        std::lock_guard<std::mutex> session_lock(st.session->mu);
        // the pool of the layer's expert shape, the same for its three tensors (an entry holds a whole expert)
        layer->pool_index = moe_cache_pool_of(device, gate_w->data);
        layer->down_pool_index = layer->pool_index;
        if (layer->pool_index < 0 || moe_cache_pool_of(device, up_w->data) != layer->pool_index ||
            moe_cache_pool_of(device, down_w->data) != layer->pool_index) {
            return nullptr;
        }
        const moe_cache_pool & pool = *device.pools[layer->pool_index];
        if (pool.role_bytes[MOE_ROLE_UP] != (size_t) up_w->nb[2] || pool.role_bytes[MOE_ROLE_GATE] != (size_t) gate_w->nb[2] ||
            pool.role_bytes[MOE_ROLE_DOWN] != (size_t) down_w->nb[2]) {
            return nullptr;
        }
        layer->base = device.pools[layer->pool_index]->gpu_base;
        layer->down_base = device.pools[layer->down_pool_index]->gpu_base;
    }
    const int rows = moe_hosted_rows_max;
    const int tokens = rows / n_used;
    GGML_ASSERT(layer->n_embd % QK8_0 == 0);
    bool ok = moe_hosted_host_alloc((void **) &layer->h_act, (size_t) tokens * layer->n_embd * sizeof(float)) &&
              moe_hosted_host_alloc((void **) &layer->h_act8, (size_t) tokens * (layer->n_embd / QK8_0) * sizeof(block_q8_0)) &&
              moe_hosted_host_alloc((void **) &layer->h_ids, rows * sizeof(int32_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_plan, 3 * rows * sizeof(uint64_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_res, (size_t) rows * layer->n_embd * sizeof(float)) &&
              moe_hosted_host_alloc((void **) &layer->h_flags, 128) &&
              moe_hosted_host_alloc((void **) &layer->h_pred, (2 + rows) * sizeof(int32_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_plan2, 3 * rows * sizeof(uint64_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_chk, (size_t) rows * layer->n_embd * sizeof(float));
    ok = ok && (layer->m_act = moe_hosted_device_ptr(layer->h_act)) && (layer->m_act8 = (block_q8_0 *) moe_hosted_device_ptr(layer->h_act8)) &&
         (layer->m_ids = moe_hosted_device_ptr(layer->h_ids)) &&
         (layer->m_plan = moe_hosted_device_ptr(layer->h_plan)) && (layer->m_res = moe_hosted_device_ptr(layer->h_res)) &&
         (layer->m_flags = moe_hosted_device_ptr(layer->h_flags)) && (layer->m_pred = moe_hosted_device_ptr(layer->h_pred)) &&
         (layer->m_plan2 = moe_hosted_device_ptr(layer->h_plan2)) && (layer->m_chk = moe_hosted_device_ptr(layer->h_chk));
    ok = ok && cudaMalloc((void **) &layer->d_plan, 3 * rows * sizeof(uint64_t)) == cudaSuccess &&
         cudaMalloc((void **) &layer->d_plan2, 3 * rows * sizeof(uint64_t)) == cudaSuccess &&
         cudaMalloc((void **) &layer->d_act_ids, rows * sizeof(int32_t)) == cudaSuccess &&
         cudaMalloc((void **) &layer->d_expect, sizeof(uint32_t)) == cudaSuccess;
    if (ok) {
        std::vector<int32_t> act_ids(rows);
        for (int row = 0; row < rows; row++) {
            act_ids[row] = row / n_used;
        }
        ok = cudaMemcpy(layer->d_act_ids, act_ids.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemset(layer->d_expect, 0, sizeof(uint32_t)) == cudaSuccess;
    }
    if (!ok) {
        (void) cudaGetLastError();
        GGML_ABORT("MoE block hosted by the GPU: allocation failed");
    }
    {
        const size_t act_bytes = (size_t) tokens * (GGML_PAD(layer->n_embd, MATRIX_ROW_PADDING) / QK8_1) * sizeof(block_q8_1);
        const size_t mid_bytes = (size_t) rows * (GGML_PAD(layer->n_ff, MATRIX_ROW_PADDING) / QK8_1) * sizeof(block_q8_1);
        if (!st.d_q8_act) {
            if (cudaMalloc((void **) &st.d_q8_act, act_bytes) != cudaSuccess ||
                cudaMalloc((void **) &st.d_q8_mid, mid_bytes) != cudaSuccess) {
                GGML_ABORT("MoE block hosted by the GPU: allocation failed");
            }
            st.q8_act_bytes = act_bytes;
            st.q8_mid_bytes = mid_bytes;
        }
        GGML_ASSERT(act_bytes <= st.q8_act_bytes && mid_bytes <= st.q8_mid_bytes);   // one shape per model
    }
    slot = (int) st.layers.size();
    st.slot_of[gate_w->data] = slot;
    st.layers.push_back(std::move(layer));
    return st.layers.back().get();
}

int ggml_cuda_moe_hosted_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 3 >= cgraph->n_nodes) {
        return -1;
    }
    ggml_tensor * n0 = cgraph->nodes[i];
    ggml_tensor * n1 = cgraph->nodes[i + 1];
    ggml_tensor * glu = cgraph->nodes[i + 2];
    ggml_tensor * down = cgraph->nodes[i + 3];
    if (glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
        n0->op != GGML_OP_MUL_MAT_ID || n1->op != GGML_OP_MUL_MAT_ID || down->op != GGML_OP_MUL_MAT_ID) {
        return -1;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up = glu->src[1];
    if (!((gate == n0 && up == n1) || (gate == n1 && up == n0)) || down->src[1] != glu) {
        return -1;
    }
    const ggml_tensor * act = gate->src[1];
    const ggml_tensor * ids = gate->src[2];
    const int n_used = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    const int rows = n_used * n_tokens;
    if (up->src[1] != act || up->src[2] != ids || down->src[2] != ids || rows > moe_hosted_rows_max ||
        act->type != GGML_TYPE_F32 || act->ne[1] != 1 || !ggml_is_contiguous(act) || !ggml_is_contiguous(glu) ||
        !ggml_is_contiguous(down) || gate->src[0]->type != up->src[0]->type) {
        return -1;
    }
    if (g_session_stack.empty() || !g_session_stack.back().active) {
        return -1;
    }
    moe_cache_session & session = *g_session_stack.back().active;
    moe_cache_device * device = nullptr;
    for (auto & d : session.devices) {
        if (d->logical == ctx.device) {
            device = d.get();
        }
    }
    if (!device) {
        return -1;
    }
    moe_hosted_state * st = moe_hosted_state_for(session, *device);
    if (!st) {
        return -1;
    }
    int slot = -1;
    moe_hosted_layer * layer = moe_hosted_register(*st, *device, gate->src[0], up->src[0], down->src[0], n_used, slot);
    if (!layer) {
        return -1;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_embd = layer->n_embd;
    const int64_t n_ff = layer->n_ff;
    const ggml_type wtype = gate->src[0]->type;
    const int64_t padded_embd = GGML_PAD(n_embd, MATRIX_ROW_PADDING);
    const int64_t padded_ff = GGML_PAD(n_ff, MATRIX_ROW_PADDING);

    // the top-k of these ids, left to this kernel (ggml_cuda_topk_moe_defer)
    auto & tp = ctx.topk_pending;
    const bool tk = tp.logits && tp.ids_t == ids;
    GGML_ASSERT(!tk || (tp.n_rows == n_tokens && tp.n_used == n_used && n_tokens * WARP_SIZE <= 256));
    if (tp.logits && !tk) {
        ggml_cuda_topk_moe_pending_flush(ctx);
    }

    // activations and ids to the host, then the mailbox
    k_moe_hosted_enter<<<8, 256, 0, stream>>>((const float *) act->data, layer->m_act, layer->m_act8, (int64_t) n_tokens * n_embd,
                                              tk ? tp.logits : nullptr, tk ? tp.weights : nullptr, tp.clamp, tp.scale,
                                              (int32_t *) ids->data, ids->nb[1], layer->m_ids, n_used, n_tokens,
                                              st->d_seq, layer->d_expect, st->m_mail, (uint32_t) slot, layer->m_flags,
                                              layer->m_plan, layer->d_plan, 3 * rows, g_tl_m);
    if (tk) {
        tp.logits = nullptr;
    }

    // resident rows: activations quantized, up/gate/swiglu into the glu result (alive until down has
    // read it), its quantization, down into the block's output
    // the activations are quantized through the context's reuse buffer: the shared expert's
    // gate / up read the same ones
    const char * q8_act = (const char *) ggml_cuda_q8_1_reuse(ctx, (const float *) act->data, n_embd, n_embd,
                                                              n_tokens * n_embd, n_tokens * n_embd, padded_embd, n_tokens, 1, 1, stream);
    if (!q8_act) {
        quantize_row_q8_1_cuda((const float *) act->data, nullptr, st->d_q8_act, wtype, n_embd, n_embd, n_tokens * n_embd,
                               n_tokens * n_embd, padded_embd, n_tokens, 1, 1, stream);
        q8_act = st->d_q8_act;
    }
    char * q8_mid = st->d_q8_mid;
    ggml_cuda_moe_cache_mmv_fused(wtype, q8_act, layer->d_plan, layer->d_plan + rows,
                                  layer->d_act_ids, (float *) glu->data, n_embd, n_ff,
                                  rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY, stream);
    quantize_row_q8_1_cuda((const float *) glu->data, nullptr, q8_mid, wtype, n_ff, n_ff, (int64_t) rows * n_ff,
                           (int64_t) rows * n_ff, padded_ff, rows, 1, 1, stream);
    ggml_cuda_moe_cache_mmv(down->src[0]->type, q8_mid, layer->d_plan + 2 * rows, nullptr,
                            (float *) down->data, n_ff, n_embd, rows, rows, stream);
    // SPLIT: the rows of the expert moved to the GPU, once it arrived; the kernels skip every other row
    const bool split_on = moe_split_enabled() && n_tokens >= 2;
    if (split_on) {
        ctx.split_wait_queued = true;
        ctx.split_wait_captured = true;
        k_moe_split_wait<<<1, 256, 0, stream>>>(layer->d_expect, layer->m_flags + 24, layer->m_plan2, layer->d_plan2, 3 * rows, g_tl_m);
        ggml_cuda_moe_cache_mmv_fused(wtype, q8_act, layer->d_plan2, layer->d_plan2 + rows,
                                      layer->d_act_ids, (float *) glu->data, n_embd, n_ff,
                                      rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY, stream);
        quantize_row_q8_1_cuda((const float *) glu->data, nullptr, q8_mid, wtype, n_ff, n_ff, (int64_t) rows * n_ff,
                               (int64_t) rows * n_ff, padded_ff, rows, 1, 1, stream);
        ggml_cuda_moe_cache_mmv(down->src[0]->type, q8_mid, layer->d_plan2 + 2 * rows, nullptr,
                                (float *) down->data, n_ff, n_embd, rows, rows, stream);
        static const bool verify = getenv("GGML_CUDA_MOE_SPLIT_VERIFY") && atoi(getenv("GGML_CUDA_MOE_SPLIT_VERIFY")) > 0;
        if (verify) {
            k_moe_split_check<<<1, 256, 0, stream>>>(layer->d_expect, layer->d_plan2 + 2 * rows, (const float *) down->data,
                                                     layer->m_chk, layer->m_flags + 28, (int) n_embd, rows);
        }
    }
    // PREGATE: the next layer's router (the next MUL_MAT with as many outputs as experts, F32 or BF16 weights) on this
    // layer's MoE input; queued after the resident rows, it runs while the GPU would wait for the CPU rows
    const int pregate_k = moe_pregate_k();
    // from 2 tokens: with one, the CPU rows (~1.5 per layer) are not the critical path and it does not pay
    if (pregate_k > 0 && n_tokens >= 2 && n_tokens * pregate_k <= rows) {
        const int64_t n_expert = gate->src[0]->ne[2];
        const ggml_tensor * w_next = nullptr;
        for (int j = i + 4; j < cgraph->n_nodes && !w_next; j++) {
            const ggml_tensor * n = cgraph->nodes[j];
            if (n->op == GGML_OP_MUL_MAT && (n->src[0]->type == GGML_TYPE_F32 || n->src[0]->type == GGML_TYPE_BF16) && n->src[0]->ne[1] == n_expert &&
                n->src[0]->ne[0] == n_embd && ggml_is_contiguous(n->src[0])) {
                w_next = n->src[0];
            }
            if (n->op == GGML_OP_MUL_MAT_ID) {
                break;   // reached the next block without its router: none
            }
        }
        // the kernels' shapes: n_embd = 128 threads x 5 float4, a thread per expert in the top-k
        if (w_next && w_next->data && n_tokens <= 8 && n_embd == 128 * 5 * 4 && n_expert <= 512) {
            if (!st->d_pg_logits) {
                CUDA_CHECK(cudaMalloc((void **) &st->d_pg_logits, (size_t) moe_hosted_rows_max * n_expert * sizeof(float)));
            }
            // two experts per block; up to 8 tokens (the MTP verification has 3)
            if (w_next->type == GGML_TYPE_BF16) {
                k_pregate_logits<8, 5><<<(unsigned) ((n_expert + 1) / 2), 256, 0, stream>>>((const uint16_t *) w_next->data, (const float *) act->data,
                                                                    st->d_pg_logits, (int) n_embd, (int) n_expert, n_tokens);
            } else {
                k_pregate_logits<8, 5><<<(unsigned) ((n_expert + 1) / 2), 256, 0, stream>>>((const float *) w_next->data, (const float *) act->data,
                                                                    st->d_pg_logits, (int) n_embd, (int) n_expert, n_tokens);
            }
            k_pregate_topk<<<(unsigned) n_tokens, 512, 0, stream>>>(st->d_pg_logits, (int) n_expert, n_tokens, pregate_k, st->d_seq, layer->m_pred);
        }
    }

    // the CPU rows are copied in when a node first reads the block's output (the shared expert,
    // queued in between, runs while the CPU computes them)
    CUDA_CHECK(cudaGetLastError());
    g_hosted_pending = {layer, down, rows, split_on};
    return 3;
}

void ggml_cuda_moe_hosted_flush(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    moe_hosted_pending & p = g_hosted_pending;
    if (!p.layer) {
        return;
    }
    if (node) {
        bool reads = false;
        for (int j = 0; j < GGML_MAX_SRC && !reads; j++) {
            const ggml_tensor * src = node->src[j];
            reads = src && (src == p.down || src->view_src == p.down);
        }
        if (!reads) {
            return;
        }
    }
    ggml_cuda_q8_1_reuse_written(ctx, p.down);
    k_moe_hosted_merge<<<p.rows, 256, 0, ctx.stream()>>>(p.layer->d_expect, p.layer->m_flags + 16,
                                                          p.layer->d_plan + 2 * p.rows, p.split ? p.layer->d_plan2 + 2 * p.rows : nullptr,
                                                          p.layer->m_res,
                                                          (float *) p.down->data, (int) p.layer->n_embd, g_tl_m);
    CUDA_CHECK(cudaGetLastError());
    p = {};
}

void ggml_moe_cache_register(const void * owner) {
    if (ggml_moe_cache.owner && ggml_moe_cache.owner != owner) {
        return;
    }
    ggml_moe_cache.owner = owner;
    ggml_moe_cache.query_config = moe_cache_query_config;
    ggml_moe_cache.query_device = moe_cache_query_device;
    ggml_moe_cache.query_shape = moe_cache_query_shape;
    ggml_moe_cache.session_create = moe_cache_session_create;
    ggml_moe_cache.session_destroy = moe_cache_session_destroy;
    ggml_moe_cache.session_enter = moe_cache_session_enter;
    ggml_moe_cache.kv_register = moe_cache_kv_register;
    ggml_moe_cache.kv_needs_set_rows = moe_cache_kv_needs_set_rows;
    ggml_moe_cache.kv_set_rows = moe_cache_kv_set_rows;
    ggml_moe_cache.kv_first_lent_row = moe_cache_kv_first_lent_row;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.plan = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.dispatch = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.collect = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.fused_begin = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.invalidate = moe_cache_invalidate;
    ggml_moe_cache.set_lend_size = moe_cache_set_lend_size;
    ggml_moe_cache.lend = moe_cache_lend;
    ggml_moe_cache.restore = moe_cache_restore;
    ggml_moe_cache.borrow = moe_cache_borrow;
    ggml_moe_cache.give_back = moe_cache_give_back;
    ggml_moe_cache.compute_buffer_type = moe_cache_compute_buffer_type;
    ggml_moe_cache.plan_borrows = moe_cache_plan_borrows;
    ggml_moe_cache.borrows = moe_cache_borrows;
}

#endif
