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

int ggml_cuda_moe_ring_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    (void) ctx; (void) cgraph; (void) i;
    return 0;
}

void ggml_cuda_moe_hosted_flush(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    (void) ctx; (void) node;
}

bool ggml_cuda_moe_hosted_wait(void * stream, void * event, int device) {
    (void) stream; (void) event; (void) device;
    return false;
}

void ggml_cuda_moe_split_capture_begin(int device, const void * owner) { (void) device; (void) owner; }
void ggml_cuda_moe_split_capture_end(int device, const void * owner) { (void) device; (void) owner; }
void ggml_cuda_moe_split_graph_launch(int device, const void * owner) { (void) device; (void) owner; }

void * ggml_cuda_host_rows_buffer(size_t bytes) {
    (void) bytes;
    return nullptr;
}

void ggml_cuda_host_rows_task(void (*fn)(void *), void * user) {
    (void) fn; (void) user;
}

bool ggml_cuda_host_rows_node(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    (void) ctx; (void) node;
    return false;
}

#else

#include "common.cuh"
#include "topk-moe.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "convert.cuh"
#include "fattn.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"
#include "moe-arena.h"
#include "moe-ram-budget.h"
#include "moe-ring.h"
#include "mmq.cuh"
#include <functional>

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
#include <functional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <process.h>
#  if defined(GGML_USE_HIP)
#    include <winternl.h>   // NTSTATUS, for d3dkmthk.h
#    include <d3dkmthk.h>   // D3DKMTQueryStatistics: the VRAM resident for every process
#  endif
#elif defined(__linux__)
#  include <fcntl.h>
#  include <sched.h>
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
// the floor applies to each expert tensor, but an entry is a whole expert (up, gate, down): 256 KiB, not buun's 512,
// so that a layer's Q2_0 down (640 x 2560, 450 KiB: ISTA's experts, the IQ3_KS hybrid and Strata's IQ3_S GGUF) is
// cached with its gate and up. The reference has run at 256 (GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB) since Oct 15
static constexpr size_t moe_cache_expert_bytes_ampere_min     = 256u << 10;
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

// ARENA: one per expert shape (moe-arena.h)
using moe_expert_arena = moe_arena<moe_cache_key, moe_cache_key_hash>;

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
    size_t role_pad[MOE_ROLES] = {};       // after each, a copy of its first bytes (moe_cache_role_pad)
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

    // Entries lending evicted during the last prefill and their LFU counts, coldest first (the order they left the
    // LRU in). Restored once generation starts (moe_cache_rewarm_start_locked).
    std::vector<std::pair<moe_cache_key, uint16_t>> rewarm;

    // Address translation. Logical slots are spread over one or more ranges of device
    // memory: range 0 is the slab, the others are borrowed (from the compute buffer).
    // Kernels only take one base and an index, so every range is addressed from a
    // virtual base, low enough and aligned so that each entry sits at
    // gpu_base + k * expert_size.
    std::vector<moe_cache_range> ranges;
    char * gpu_base = nullptr;
    int64_t gpu_extent = 0;     // largest gpu index + 1, passed to kernels as a bound

    std::vector<moe_cache_kv_range> kv;   // KV tails lent to this pool

    // BANDS (moe_cache_allocate_banded): the pools of a device share one slab cut into bands of band_bytes, every band
    // holding entries of every pool (as many as keep each pool at its share of the slab, so the counts differ from band
    // to band; a band's last entry may run into the next). A run of bands is a run of slots in every pool, at its tail:
    // zones and loans take whole bands, from every expert shape alike.
    char * band_base = nullptr;
    size_t band_bytes = 0;
    std::vector<char *> band_ptr;   // per slot of the bands: its address, increasing
    int band_slots = 0;             // the slots of the bands
    std::vector<uint8_t> band_refs;   // per slot of the bands: the loans covering it (an entry astride two loans: 2)
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
    if (index < pool.band_slots) {
        return pool.band_ptr[(size_t)index];
    }
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
    size_t role_pad[MOE_ROLES] = {};
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
    size_t bytes[MOE_ROLES] = {};    // per expert
    size_t stride[MOE_ROLES] = {};   // from one expert to the next in host memory (nb[2]: bytes, or a packed entry)
    int64_t ne0[MOE_ROLES] = {};     // values per row
    int type[MOE_ROLES] = {-1, -1, -1};
    int64_t n_expert = 0;
    int shape = -1;                  // once the pools are built
    bool guest = false;              // shown by a scheduler sharing the session (the MTP draft): not a main-model layer
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;         // the range: from the first expert to the end of the last one
    size_t expert_size = 0;   // its bytes per expert
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
    // the next expert's tensor follows in host memory: a tensor's pad (moe_cache_role_pad) is copied with it, from
    // the next expert's first blocks (any blocks of its type do), in the same copy
    bool pad_follows[MOE_ROLES] = {};
    bool whole = false;                    // a packed layer laid out as the entry: one copy of bytes from source[0]
    size_t bytes = 0;                      // the entry
    bool rewarm = false;   // an entry lending evicted, restored: joins the LRU at its cold end
    // SPLIT: a tensor of an expert moved to the GPU for the layer being computed; the last one of the expert
    // to arrive raises the layer's flag
    uint32_t * split_flag = nullptr;
    int * split_pending = nullptr;
    uint32_t split_seq = 0;
    // ARENA: the arena slot the fill copies from, held (a reader) from the admission to the end of the copy; -1 when
    // the arena is off or the expert is registered in place
    int arena_hold = -1;
    moe_expert_arena * arena = nullptr;   // the arena of arena_hold
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
    // the window rises from 2 to lfu_window over lfu_ramp x entries demands after the content was renewed (the
    // preload, the restoration after a prefill): forget a poor start fast, then keep the history (0 = fixed). Replay,
    // 3-token passes, against a fixed 5: code (a profile serving it poorly) -2.8 % (32k) / -2.1 % (251k), prose
    // (served well) +0.3 / +0.1 %; a fixed 2 cost prose +0.25 %. 20: the best of 5 - 40 in the cache simulator
    int lfu_ramp = 20;
    int lfu_cap = 15;
    // the window follows the context depth (KV rows): lfu_window up to depth_window[0], depth_window[1] from
    // depth_window[2], linear between, without the ramp; a deep context's demand moves faster.
    // GGML_CUDA_MOE_CACHE_LFU_WINDOW_DEPTH="d0,w1,d1", "0" for the fixed window. MEASURED: see the commit
    double depth_window[3] = {32768, 2, 250000};
    // deferred admissions (GGML_CUDA_MOE_CACHE_DEFER=0 disables): see moe_hosted_admit_deferred
    bool defer = true;
    // a pass's deferred admissions can come in bursts (up to ~180 after a long prefill)
    int queue_max = 1024;
    size_t queue_mb = 4096;
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

// a pool lends at most this share of its slab (moe_cache_pool_lendable)
static constexpr int moe_cache_lend_percent = 95;

// ARENA: threads for the CPU path's O_DIRECT reads (an NVMe drive needs several reads in flight for its rate)
struct moe_arena_readers {
    std::mutex mu;
    std::condition_variable cv, done_cv;
    std::deque<std::function<void()>> q;
    bool stop = false;
    std::vector<std::thread> threads;

    explicit moe_arena_readers(int n) {
        for (int i = 0; i < n; i++) {
            threads.emplace_back([this] {
                for (;;) {
                    std::function<void()> f;
                    {
                        std::unique_lock<std::mutex> lock(mu);
                        cv.wait(lock, [this] { return stop || !q.empty(); });
                        if (q.empty()) {
                            return;
                        }
                        f = std::move(q.front());
                        q.pop_front();
                    }
                    f();
                }
            });
        }
    }
    ~moe_arena_readers() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : threads) {
            t.join();
        }
    }
    // runs the tasks on the threads ahead of what post() queued (the CPU rows and the prefill wait for them), returns
    // when they are done
    void run(std::vector<std::function<void()>> & tasks) {
        if (tasks.empty()) {
            return;
        }
        int left = (int) tasks.size();
        std::unique_lock<std::mutex> lock(mu);
        for (auto it = tasks.rbegin(); it != tasks.rend(); ++it) {
            q.push_front([this, &left, f = std::move(*it)] {
                f();
                std::lock_guard<std::mutex> l(mu);
                if (--left == 0) {
                    done_cv.notify_all();
                }
            });
        }
        cv.notify_all();
        done_cv.wait(lock, [&] { return left == 0; });
    }
    // a task behind the others, not waited for (a restoration's reads)
    void post(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lock(mu);
            q.push_back(std::move(f));
        }
        cv.notify_one();
    }
};

struct moe_cache_device {
    // the largest piece the device lends while the prefetch holds its zones, set once with the zones (the bands: all of their
    // free room; pools apart: one pool's, 1 GiB at most), the compute buffers' chunk (moe_cache_largest_lend)
    std::atomic<size_t> largest_lend_bytes{0};
    // BANDS: the slab of the banded pools (moe_cache_allocate_banded). Zones and loans are byte ranges of it: the
    // prefetch zones at its top (band_zones slots of band_zone_bytes from band_zone_begin), the compute buffer chunks in
    // the free space under them (band_loans, sorted), above a floor of 5 % that stays cached
    char * band_slab = nullptr;
    size_t band_bytes = 0;
    int n_bands = 0;
    size_t band_zone_begin = 0;
    size_t band_zone_bytes = 0;
    int band_zones = 0;
    std::vector<std::pair<size_t, size_t>> band_loans;
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
    bool pin_full = false;   // KFD's limit was reached: nothing more is registered
    // PIN, direct load (moe_cache_pin_load_direct): the expert ranges made anonymous memory filled from the files with
    // O_DIRECT before they are registered; each one's file and offset, to map it back if KFD refuses it
    bool pin_direct = false;
    size_t pin_quota = SIZE_MAX;   // the bytes this card may register (moe-ram-budget.h, several cards that do not all fit)
    bool arena_off = false, pin_off = false;   // GGML_CUDA_MOE_ARENA_MB=auto's choice (moe_cache_build_pending)
    struct pin_file { uintptr_t start, end; std::string path; uint64_t offset; };
    std::vector<pin_file> pin_files;
    // the expert arena (moe-arena.h, GGML_CUDA_MOE_ARENA_MB): one per expert shape (index of shapes, null for a shape
    // without one), its slots under the session lock; one mapping, registered in pieces of whole slots of each
    std::vector<std::shared_ptr<moe_expert_arena>> arenas;
    // GGML_CUDA_MOE_RING_PREFILL: the prompt's expert ring in the zones the scheduler borrowed (moe_cache_ring_zone)
    struct moe_ring_state * ring = nullptr;
    // EXPERT SPLIT: the experts' host memory (the pin, the arena) is the session's, taken once by the first card
    // (host_ready once done) and shared by the others (host_shared: they register, read and unmap nothing themselves)
    bool host_ready = false;
    bool host_shared = false;
    bool host_keep = false;   // trimmed (moe_cache_trim_session) while other cards share its host memory: kept until the end
    char * arena_base = nullptr;
    size_t arena_mapped = 0;                              // bytes mapped
    std::vector<std::pair<char *, size_t>> arena_pieces;   // the registered pieces
    // the mappings of the model files behind the expert tensors: [start, end) is the file from offset, read with O_DIRECT
    struct arena_file { uintptr_t start, end; int fd; uint64_t offset; };
    std::vector<arena_file> arena_files;
    std::map<std::string, int> arena_fds;   // the files opened with O_DIRECT, by path
    uint64_t arena_gen = 0;   // bumped by a prefill's lending: a restoration's reads still queued give up
    std::unique_ptr<moe_arena_readers> arena_readers;   // the CPU path's O_DIRECT reads, in parallel
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    uint64_t freq_demands = 0;   // LFU: demands since the last aging
    uint64_t renewed_demands = 0;   // LFU: demands since the content was renewed (lfu_ramp)
    // GGML_CUDA_MOE_CACHE_PROFILE_SAVE: routed rows per (layer, expert) seen by the hosted plans, written at the end
    std::vector<uint32_t> heat;
    int64_t heat_experts = 0;
    double freq_period = 0;      // LFU: demands between agings, fixed when first needed (a lent KV tail adds entries later)
    size_t visits_since_new_tensor = 0;
    bool budget_ready = false;
    bool budget_registered = false;
    bool budget_claimed = false;
    size_t automatic_reserve_bytes = 0;
    size_t budget_reserve_bytes = 0;
    size_t budget_limit = 0;
    size_t preload_bytes = 0;   // what moe_cache_preload installed (EXPERT SPLIT, DEAL=2: where the next card's band starts)
    int64_t preload_us = 0;     // the time its copies took, from the first to the end of the last (the link's measured rate)
    size_t coordinator_allocated_bytes = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;

    std::deque<moe_cache_job> queue;
    // deferred admissions (moe_hosted_admit_deferred): the candidates of the pass's main-model layers so far, one list
    // kept by decreasing count, arrival order among equals (moe_cache_defer_add), and the main model's first and last
    // layers (the layers known when the pools are built: the MTP draft's joins later and admits as before)
    struct deferred_candidate {
        int pool = -1;
        moe_cache_key key;
        uint16_t freq = 0;
        bool contest = false;
    };
    std::vector<deferred_candidate> deferred;
    bool deferring = false;   // a main-model layer's plan runs: an admission is recorded, not made
    int main_first = -1;
    int main_last = -1;
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
    bool inflight_rewarm = false;   // the job in flight restores an entry (moe_cache_rewarm_feed_locked)
    const void * inflight_source[MOE_ROLES] = {};
    size_t inflight_bytes[MOE_ROLES] = {};
    std::thread worker;

    cudaStream_t compute_stream = nullptr;
    ggml_backend_cuda_context * backend_ctx = nullptr; // the context of the scheduler that created the session
    cudaStream_t fill_stream = nullptr;                // the fill worker's stream, created and dispatched before sizing
    std::vector<char *> owned_regions;    // extra regions this device allocated itself (tests)

    // seating of the chunks ggml-alloc announced (moe_cache_plan_borrows): the pool of each
    // chunk size, nullptr once served
    std::vector<size_t> loan_plan_bytes;
    std::vector<moe_cache_pool *> loan_plan_pool;

    // restoring the entries lending evicted (pool.rewarm) and the ranking's pairs, from the start of generation until
    // a new prefill lends again
    bool rewarm_active = false;
    bool rewarm_armed = false;   // after a prefill: the restoration starts at the next pass's first hosted layer
    bool rewarm_bulk_pending = false;   // the restoration at once, handed to the fill worker (moe_cache_rewarm_bulk_locked)
    std::vector<std::pair<int, moe_cache_key>> rewarm_order;   // (pool, key), what the restoration brings back, in order
    size_t rewarm_next = 0;
    long long rewarm_restored = 0;
    bool lfu = false;   // session.config.lfu: an entry a loan evicts keeps its count (demand_count)
    // the preload's ranking (pool, key, share of the demands), hottest first: with the entries lending evicted, what
    // the restoration after a prefill brings back (empty without GGML_CUDA_MOE_CACHE_PRELOAD)
    struct ranked { int pool; moe_cache_key key; float share; };
    std::vector<ranked> ranking;
    // the preload's ranking as read, (layer, expert, share) hottest first: translated again into ranking when a layer
    // joins the pools later (the MTP draft's, shown when it shares the session), so that its pairs take their place
    struct ranked_src { int layer; int expert; float share; };
    std::vector<ranked_src> ranking_src;
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
    // EXPERT SPLIT (GGML_MOE_CACHE_EXPERT_SPLIT=1): the hosted rows this card planned, by where their expert was (here,
    // in another card's cache, nowhere), and its admissions sent to another card; the rows another card's plans found here
    long long split_local = 0, split_remote = 0, split_absent = 0, split_admit_away = 0, split_served = 0;
    long long split_moved_away = 0;   // experts its dispatcher moved to another card
    long long zone_lends = 0;
    long long prefill_resident = 0;   // experts of prefetched tensors read in place
    long long prefill_streamed = 0;   // experts of prefetched tensors copied
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
    // entries a prefill graph reads in place (moe_cache_prefill_residents), until moe_cache_prefill_unpin
    std::vector<std::pair<moe_cache_pool *, int>> prefill_pins;
    // ARENA: the arena slots a prefetch slot's copies read, by prefetch slot (moe_cache_prefill_sources)
    std::map<int, std::vector<std::pair<moe_expert_arena *, int>>> prefill_arena_pins;
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
    bool guest = false;   // a scheduler sharing another's session (session_enter_guest)
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

// NVIDIA: at 400 MiB a 19k prefill ran out of memory once in a few servers (RTX 4090, 25 Oct.: an 8 MiB stream-k
// temporary of MMQ from the VMM pool, the card full); at 1024 none did, and the peak left the whole reserve free (no
// growth seen at 10 Hz, so a transient: cause not found). The reserve, explicit or not, is at least 1024 there.
static size_t moe_cache_platform_reserve_bytes(size_t reserve_bytes) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return reserve_bytes;
#else
    return std::max<size_t>(reserve_bytes, 1024ull * 1024 * 1024);
#endif
}

static bool moe_cache_budget_register(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    std::vector<moe_cache_device *> registered;
    try {
        registered.reserve(session.devices.size());
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            moe_cache_physical_budget & state = g_physical_budgets[device.physical];
            const size_t reserve_bytes = moe_cache_platform_reserve_bytes(ggml_moe_cache_effective_reserve_bytes(
                    session.config.reserve_explicit,
                    session.config.reserve_mb << 20,
                    device.automatic_reserve_bytes));
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
    // DIAGNOSTIC (GGML_MOE_CACHE_VIRTUAL_DEVICES=1): free_memory is already this virtual device's own
    // (ggml_cuda_virtual_memory), not the physical GPU's to split between its caches
    if (getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES") && atoi(getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES")) != 0) {
        claim = available_after_reserve;
    }
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
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_LFU_RAMP", 0, 1000, value)) {
        config.lfu_ramp = (int)value;
    }
    if (const char * a = getenv("GGML_CUDA_MOE_CACHE_LFU_WINDOW_DEPTH")) {
        double d0 = 0, w1 = 0, d1 = 0;
        if (sscanf(a, "%lf,%lf,%lf", &d0, &w1, &d1) == 3 && d1 > d0 && w1 >= 1) {
            config.depth_window[0] = d0;
            config.depth_window[1] = w1;
            config.depth_window[2] = d1;
        } else {
            config.depth_window[0] = config.depth_window[2] = 0;   // the fixed window
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DEFER", 0, 1, value)) {
        config.defer = value != 0;
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
// not on the card's size (the cache itself takes the free memory minus this). The queues and their scratch are created
// before (moe_cache_prepare_queues); measured after (11 Oct., Qwen3.8-Flash-Next, 256k, MTP, 8192-token ubatches,
// LD_PRELOAD tools/kfdalloc): ~170 MiB at the peak, of which rocBLAS and graph buffers ~25, ~145 outside the KFD
// allocations (queue rings, page tables). 500 covers it about three times. Also the default margin of --fit, which sets
// the reserve when it runs.
static size_t moe_cache_recommended_reserve_bytes(size_t total_vram) {
    constexpr size_t MiB = 1024 * 1024;
    return std::min<size_t>(400 * MiB, total_vram / 4);
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
        case GGML_TYPE_IQ3_KS:
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
    result->recommended_reserve_bytes = moe_cache_platform_reserve_bytes(ggml_moe_cache_effective_reserve_bytes(
            config->reserve_explicit,
            config->reserve_bytes,
            moe_cache_recommended_reserve_bytes(cuda_device.total_vram)));
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

    // expert_size: nb[2], the expert's bytes or, in a packed layer, the whole entry
    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size < (size_t)n_out * row_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }
    expert_size = (size_t)n_out * row_size;

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
static int moe_cache_rewarm_from();
static bool moe_cache_enqueue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker, bool allow_eviction, bool rewarm, bool pregate, uint16_t freq = 0, bool contest = false);
static void moe_cache_rewarm_stop_locked(moe_cache_device & device);
static void moe_cache_rewarm_start_locked(moe_cache_session & session, moe_cache_device & device);
static bool moe_cache_rewarm_bulk_on();
static void moe_cache_rewarm_bulk_locked(moe_cache_session & session, moe_cache_device & device,
                                         std::unique_lock<std::mutex> & lock);
static double moe_cache_freq_period(const moe_cache_session & session, moe_cache_device & device);

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

static __global__ void k_moe_cache_queue_prime() {}

// before the cache sizes itself (ggml_backend_sched_prime_moe_cache): the queues and scratch the later work needs are
// created now (ggml_cuda_prime_queues), so that the free memory it measures has them already. The target's flash
// attention kernels, picked from its reserve graph, set the private memory of its queue; the fill worker's stream is
// created and dispatched here.
static void moe_cache_prepare_queues(void * opaque, const struct ggml_cgraph * graph) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return;
    }
    for (auto & device_ptr : session->devices) {
        moe_cache_device & device = *device_ptr;
        if (device.budget_ready) {
            continue;
        }
        if (device.backend_ctx && graph) {
            for (int i = 0; i < graph->n_nodes; ++i) {
                ggml_tensor * node = graph->nodes[i];
                if (node->op == GGML_OP_FLASH_ATTN_EXT && ggml_cuda_flash_attn_ext_supported(device.logical, node)) {
                    ggml_cuda_flash_attn_ext_scratch_size(*device.backend_ctx, node);
                }
            }
        }
        ggml_cuda_set_device(device.logical);
        if (!device.fill_stream) {
            int least_priority = 0;
            int greatest_priority = 0;
            if (cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) != cudaSuccess ||
                    cudaStreamCreateWithPriority(&device.fill_stream, cudaStreamNonBlocking, least_priority) != cudaSuccess) {
                (void) cudaGetLastError();
                device.fill_stream = nullptr;
            } else {
                k_moe_cache_queue_prime<<<1, 1, 0, device.fill_stream>>>();
                (void) cudaStreamSynchronize(device.fill_stream);
            }
        }
        ggml_cuda_prime_queues(device.logical);
    }
}

static bool moe_arena_load(const moe_cache_device & d, moe_expert_arena & a, int s, const moe_cache_job & j,
                           const moe_cache_pool & p);

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
        if (j.whole) {
            if (!pinned(j.source[0], j.bytes)) {
                memcpy(buf, j.source[0], j.bytes);
            }
            return;
        }
        for (int r = 0; r < MOE_ROLES; r++) {
            const size_t bytes = j.role_bytes[r] + (j.pad_follows[r] ? p->role_pad[r] : 0);
            if (!pinned(j.source[r], bytes)) {
                memcpy(buf + p->role_offset[r], j.source[r], bytes);
            }
        }
    };
    // the job's three copies, each from the weights, from the staging buffer or from the expert's arena slot (parts)
    const auto copy_job = [&](const moe_cache_job & j, const moe_cache_pool * p, char * dst, const char * buf,
                              const char * const * parts) {
        cudaError_t err = cudaSuccess;
        if (j.whole) {   // a packed layer: the expert's up, gate, down and pads in one copy
            const void * src = parts ? (const void *) parts[0] : pinned(j.source[0], j.bytes) ? j.source[0] : (const void *) buf;
            err = cudaMemcpyAsync(dst, src, j.bytes, cudaMemcpyHostToDevice, stream);
        }
        for (int r = 0; r < MOE_ROLES && err == cudaSuccess && !j.whole; r++) {
            const size_t bytes = j.role_bytes[r] + (j.pad_follows[r] ? p->role_pad[r] : 0);
            const void * src = parts ? (const void *) parts[r] :
                pinned(j.source[r], bytes) ? j.source[r] : (const void *) (buf + p->role_offset[r]);
            err = cudaMemcpyAsync(dst + p->role_offset[r], src, bytes, cudaMemcpyHostToDevice, stream);
            if (err == cudaSuccess && p->role_pad[r] && !j.pad_follows[r]) {   // the last expert: its own first bytes
                err = cudaMemcpyAsync(dst + p->role_offset[r] + p->role_bytes[r], src, p->role_pad[r], cudaMemcpyHostToDevice, stream);
            }
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
            error = copy_job(cur, cur_pool, cur_dst, bufs[b], nullptr);
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
            error = copy_job(cur, cur_pool, cur_dst, bufs[b], nullptr);
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

    // the queue's head taken (session lock held): marked in flight, its entry's address found
    const auto take_locked = [&](moe_cache_job & j, moe_cache_pool *& p, char *& dst) {
        j = device->queue.front();
        device->queue.pop_front();
        device->queued_bytes = j.bytes <= device->queued_bytes ? device->queued_bytes - j.bytes : 0;
        device->inflight = true;
        device->inflight_rewarm = j.rewarm;
        for (int r = 0; r < MOE_ROLES; r++) {
            device->inflight_source[r] = j.source[r];
            device->inflight_bytes[r] = j.role_bytes[r];
        }
        p = nullptr;
        dst = nullptr;
        if (j.pool >= 0 && j.pool < (int) device->pools.size()) {
            p = device->pools[j.pool].get();
            if (p->slab && j.slot >= 0 && j.slot < p->n_slots) {
                dst = moe_cache_slot_ptr(*p, j.slot);
            }
        }
    };
    // a fill's closing takes the next job under the same lock when one waits (one lock per fill while the queue holds
    // jobs, instead of three: take, the entry's address, close)
    bool have_next = false;
    moe_cache_job job;
    moe_cache_pool * pool = nullptr;
    char * destination = nullptr;

    for (;;) {
        if (have_next) {
            have_next = false;
        } else {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() || device->rewarm_bulk_pending ||
                    !device->queue.empty() || (device->pregate_window && device->pg_next < device->pg_cands.size());
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }
            if (device->rewarm_bulk_pending) {
                // the restoration at once: this thread waits for its copies, never the one serving the hosted MoE's
                // plans (the GPU spins in a kernel for its plan: a wait there for the fill stream never ended on an RTX
                // 4090 after a prompt of 131k at -ub 27648)
                device->rewarm_bulk_pending = false;
                moe_cache_rewarm_bulk_locked(*session, *device, lock);
                moe_cache_rewarm_feed_locked(*session, *device);
                continue;
            }

            if (device->queue.empty()) {
                // no ordinary fill waits: the next layer's prefetches, while the window is open
                lock.unlock();
                pregate_burst();
                continue;
            }
            take_locked(job, pool, destination);
        }

        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->logical);
        int arena_slot = -1;
        bool arena_read = false;

        if (device->dead.load() || moe_cache_fail(*session, "insert")) {
            error = cudaErrorUnknown;
        }
        if (error == cudaSuccess && !stream && device->fill_stream) {
            stream = device->fill_stream; // created by moe_cache_prepare_queues, the worker owns it from here
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

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            // ARENA: copied from the expert's slot, held since its admission
            arena_slot = job.arena ? job.arena_hold : -1;
            arena_read = arena_slot >= 0;
            const char * const * parts = arena_slot >= 0 ? job.arena->slots[arena_slot].part : nullptr;
            if (error == cudaSuccess && (arena_slot >= 0 || (stage && stage_capacity >= job.bytes))) {
                // DIAGNOSTIC (GGML_CUDA_MOE_FILL_STATS=1): per 2000 fills, time in the copy to the staging
                // buffer, in the copy to the device and its wait (of which the call), between fills
                static const bool ft = getenv("GGML_CUDA_MOE_FILL_STATS") != nullptr;
                static int64_t f_n = 0, f_cpy = 0, f_dma = 0, f_idle = 0, f_last = 0;
                const int64_t t0 = ft ? ggml_time_us() : 0;
                if (arena_slot < 0) {
                    stage_job(job, pool, stage);
                }
                const int64_t t1 = ft ? ggml_time_us() : 0;
                error = copy_job(job, pool, destination, stage, parts);
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
            } else if (error == cudaSuccess && job.whole) {
                error = cudaMemcpy(destination, job.source[0], job.bytes, cudaMemcpyHostToDevice);
            } else if (error == cudaSuccess) {
                for (int r = 0; r < MOE_ROLES && error == cudaSuccess; r++) {
                    error = cudaMemcpy(destination + pool->role_offset[r], job.source[r],
                                       job.role_bytes[r] + (job.pad_follows[r] ? pool->role_pad[r] : 0), cudaMemcpyHostToDevice);
                    if (error == cudaSuccess && pool->role_pad[r] && !job.pad_follows[r]) {
                        error = cudaMemcpy(destination + pool->role_offset[r] + pool->role_bytes[r], job.source[r],
                                           pool->role_pad[r], cudaMemcpyHostToDevice);
                    }
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
            if (arena_read) {
                job.arena->release(arena_slot);
                job.arena->hits++;
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
            if (error == cudaSuccess && !session->stopping && !device->dead.load() && !device->queue.empty()) {
                take_locked(job, pool, destination);
                have_next = true;
            }
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

// A prefill reads an entry in place with MMQ (moe_cache_prefill_residents), which reads a row in steps of 256 values
// (512 at most): a tensor whose rows are not a multiple is read past the end of its last row, by up to this many bytes.
// The entry repeats the tensor's first bytes there (blocks of its own type, whatever their values: finite scales; the
// next tensor's bytes, read as scales of this type, could be NaN, and NaN times the zero padding of the activations is
// NaN: seen with down, 640 values). Rounded to moe_cache_role_align.
static size_t moe_cache_role_pad(int type, int64_t ne0) {
    if (type < 0 || ne0 <= 0) {
        return 0;
    }
    const size_t over = (size_t) ((GGML_PAD(ne0, 512) - ne0) / ggml_blck_size((ggml_type) type)) * ggml_type_size((ggml_type) type);
    return (over + moe_cache_role_align - 1) / moe_cache_role_align * moe_cache_role_align;
}

// The entry layout of an expert shape: the offset of each tensor, each starting at a multiple of moe_cache_role_align,
// each followed by its pad. Returns the entry's size.
static size_t moe_cache_entry_layout(const size_t bytes[MOE_ROLES], const size_t pad[MOE_ROLES], size_t offset[MOE_ROLES]) {
    size_t off = 0;
    for (int r = 0; r < MOE_ROLES; r++) {
        offset[r] = off;
        off += (bytes[r] + moe_cache_role_align - 1) / moe_cache_role_align * moe_cache_role_align + pad[r];
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
// EXPERT SPLIT (GGML_MOE_CACHE_EXPERT_SPLIT=1): see moe_split_find_locked
#define MOE_SPLIT_MAX_CARDS 4
// EXPERT SPLIT, PSUM: up to this many tokens, another card returns one weighted sum per token of its rows instead of
// each row (GGML_MOE_SPLIT_PSUM=0: each row)
#define MOE_SPLIT_PSUM_TOKENS 8
#define MOE_SPLIT_MAX_SLOTS 256
// a wait of the cards' chains or of the merge for a card longer than this (clock ticks: 100 MHz under HIP, ns under
// CUDA) gives up and is reported (moe_split_diag) rather than freezing the GPU
#if defined(GGML_USE_HIP)
#define MOE_SPLIT_WAIT_BUDGET 500000000ull
#else
#define MOE_SPLIT_WAIT_BUDGET 5000000000ull
#endif
static bool moe_cache_expert_split() {
    static const bool on = getenv("GGML_MOE_CACHE_EXPERT_SPLIT") && atoi(getenv("GGML_MOE_CACHE_EXPERT_SPLIT")) != 0;
    return on;
}

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
            same = same && sh.role_bytes[r] == layer.bytes[r] && sh.role_type[r] == layer.type[r] &&
                sh.role_pad[r] == moe_cache_role_pad(layer.type[r], layer.ne0[r]);
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
        sh.role_pad[r] = moe_cache_role_pad(layer.type[r], layer.ne0[r]);
    }
    sh.expert_size = moe_cache_entry_layout(sh.role_bytes, sh.role_pad, offset);
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

#if defined(GGML_USE_HIP) && defined(_WIN32)
// the free VRAM of the device's adapter by WDDM's segment statistics, every process counted (hipMemGetInfo counts only
// this one's: the desktop and a browser held ~2.5 GiB on a 24 GiB card, and past the physical VRAM WDDM moves
// allocations to system memory, the cache read over PCIe): the local segments' commit limit minus their resident
// bytes. The adapter is found by the LUID HIP gives; D3DKMTQueryStatistics is taken from gdi32 at run time.
static bool moe_cache_d3dkmt_free(int physical, size_t & free_out, size_t & total_out) {
    using query_fn = NTSTATUS (APIENTRY *)(const D3DKMT_QUERYSTATISTICS *);
    static const query_fn query = [] {
        HMODULE gdi = LoadLibraryA("gdi32.dll");
        return gdi ? (query_fn) GetProcAddress(gdi, "D3DKMTQueryStatistics") : nullptr;
    }();
    if (!query) {
        return false;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, physical) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    LUID luid;
    static_assert(sizeof(luid) == sizeof(prop.luid), "LUID size");
    memcpy(&luid, prop.luid, sizeof(luid));

    D3DKMT_QUERYSTATISTICS q = {};
    q.Type        = D3DKMT_QUERYSTATISTICS_ADAPTER;
    q.AdapterLuid = luid;
    if (query(&q) != 0) {
        return false;
    }
    const ULONG nsegments = q.QueryResult.AdapterInformation.NbSegments;
    unsigned long long limit = 0, resident = 0;
    for (ULONG i = 0; i < nsegments; ++i) {
        D3DKMT_QUERYSTATISTICS qs = {};
        qs.Type                   = D3DKMT_QUERYSTATISTICS_SEGMENT;
        qs.AdapterLuid            = luid;
        qs.QuerySegment.SegmentId = i;
        if (query(&qs) != 0) {
            return false;
        }
        const D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION & seg = qs.QueryResult.SegmentInformation;
        if (seg.Aperture) {
            continue;   // system memory seen through the GPU's aperture
        }
        limit    += seg.CommitLimit;
        resident += seg.BytesResident;
    }
    if (limit == 0) {
        return false;
    }
    total_out = (size_t) limit;
    free_out  = resident < limit ? (size_t) (limit - resident) : 0;
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
    // the cache is sized from the amdgpu driver counters (sysfs), which hipMemGetInfo underestimates (+1.8 GiB of cache
    // on a 20 GiB card); falls back to the HIP figure if they cannot be read. GGML_CUDA_MOE_CACHE_FREE_SOURCE=hip keeps
    // the HIP figure (on Windows also without the share of other processes)
    {
        const char * source = getenv("GGML_CUDA_MOE_CACHE_FREE_SOURCE");
        if (!source || strcmp(source, "hip") != 0) {
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
#if defined(_WIN32)
            // hipMemGetInfo does not count the VRAM of other processes: the free VRAM by WDDM's statistics, read once
            // when the cache is sized; failing that, 3072 MiB left to them (the desktop and a browser held ~2.5 GiB)
            size_t d3d_free  = 0;
            size_t d3d_total = 0;
            if (moe_cache_d3dkmt_free(device.physical, d3d_free, d3d_total)) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d free source=d3dkmt: hip=%zu MiB d3dkmt=%zu MiB (total %zu MiB)\n",
                        device.physical, free_memory >> 20, d3d_free >> 20, d3d_total >> 20);
                free_memory = std::min(free_memory, d3d_free);
            } else {
                constexpr size_t other_processes = (size_t) 3072 << 20;
                free_memory = free_memory > other_processes ? free_memory - other_processes : 0;
                MOE_CACHE_LOG("[moe-cache] CUDA%d free source=d3dkmt unavailable, hip minus %zu MiB for other processes: %zu MiB\n",
                        device.physical, other_processes >> 20, free_memory >> 20);
            }
#endif
        }
    }
#endif

    // DIAGNOSTIC (GGML_MOE_CACHE_VIRTUAL_DEVICES=1, one cache per virtual device): the memory of a card of its own
    if (getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES") && atoi(getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES")) != 0) {
        size_t vfree = free_memory, vtotal = total_memory;
        ggml_cuda_virtual_memory(device.logical, &vfree, &vtotal);
        if (vtotal != total_memory) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d virtual device %d: free %zu of %zu MiB\n", device.physical, device.logical,
                    vfree >> 20, vtotal >> 20);
            free_memory = vfree;
        }
    }

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
        pool->expert_size = moe_cache_entry_layout(shape.role_bytes, shape.role_pad, pool->role_offset);
        for (int r = 0; r < MOE_ROLES; r++) {
            pool->role_bytes[r] = shape.role_bytes[r];
            pool->role_pad[r] = shape.role_pad[r];
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

// BANDS: the pools of a device in one slab of bands of 128 MiB; GGML_MOE_CACHE_BAND_MB=N another band size, 0 a slab
// per pool (as before)
static size_t moe_cache_band_bytes() {
    static const size_t bytes = [] () -> size_t {
        const char * text = getenv("GGML_MOE_CACHE_BAND_MB");
        const long long value = text && text[0] ? atoll(text) : 128;
        return value > 0 ? (size_t)value << 20 : 0;
    }();
    return bytes;
}

// BANDS: the bytes of the slab and its floor (the bottom 5 %, never lent)
static size_t moe_cache_band_total(const moe_cache_device & device) {
    return (size_t)device.n_bands * device.band_bytes;
}
static size_t moe_cache_band_floor(const moe_cache_device & device) {
    return moe_cache_band_total(device) - moe_cache_band_total(device) * moe_cache_lend_percent / 100;
}

// BANDS: where a chunk of 'bytes' goes under the zones and above the floor (aligned to 4 KiB), SIZE_MAX: nowhere. The
// small ones (under 256 MiB: the MTP draft's compute buffer, kept for good, and generation's, kept for an answer) at the
// bottom of the lowest gap that holds them, the large ones (a prefill's, given back after each ubatch) at the top of the
// highest: the small ones lasting, a draft's 127 MiB placed between a prefill's chunks would cut the free space in two
static size_t moe_cache_band_find(const moe_cache_device & device, size_t bytes) {
    const bool small = bytes < ((size_t)256 << 20);
    const size_t n = device.band_loans.size();
    for (size_t k = 0; k <= n; k++) {
        const size_t i = small ? k : n - k;   // the gap between loan i - 1 and loan i
        const size_t bottom = i > 0 ? device.band_loans[i - 1].second : moe_cache_band_floor(device);
        const size_t top = i < n ? device.band_loans[i].first : device.band_zone_begin;
        if (top < bottom || top - bottom < bytes) {
            continue;
        }
        const size_t start = small ? ((bottom + 4095) & ~(size_t)4095) : ((top - bytes) & ~(size_t)4095);
        if (start >= bottom && start + bytes <= top) {
            return start;
        }
    }
    return SIZE_MAX;
}
static void moe_cache_band_room(const moe_cache_device & device, size_t & free_bytes, size_t & largest) {
    free_bytes = largest = 0;
    size_t bottom = moe_cache_band_floor(device);
    for (size_t i = 0; i <= device.band_loans.size(); i++) {
        const size_t top = i < device.band_loans.size() ? device.band_loans[i].first : device.band_zone_begin;
        const size_t gap = top > bottom ? top - bottom : 0;
        free_bytes += gap;
        largest = std::max(largest, gap);
        if (i < device.band_loans.size()) {
            bottom = device.band_loans[i].second;
        }
    }
}

// BANDS: the slots of a banded pool that the byte range [a, b) of the slab overlaps: one run [s0, s1) (an entry that
// starts before a or ends after b included)
static void moe_cache_band_slots(const moe_cache_pool & pool, size_t a, size_t b, int & s0, int & s1) {
    s0 = s1 = 0;
    if (pool.band_ptr.empty() || b <= a) {
        return;
    }
    const char * pa = pool.band_base + a, * pb = pool.band_base + b;
    const size_t es = pool.expert_size;
    // the first entry ending after a, and the first starting at b or later
    s0 = (int)(std::upper_bound(pool.band_ptr.begin(), pool.band_ptr.end(), pa,
                                [es](const char * at, const char * e) { return at < e + es; }) - pool.band_ptr.begin());
    s1 = (int)(std::lower_bound(pool.band_ptr.begin(), pool.band_ptr.end(), pb) - pool.band_ptr.begin());
    if (s1 < s0) {
        s1 = s0;
    }
}

static void moe_cache_evacuate(moe_cache_device & device, moe_cache_pool & pool, int b, int e, std::vector<int> & moved);

// BANDS: lend [a, b) of the slab: in every banded pool, the slots it overlaps leave the cache (those not already lent
// by a neighbour moved out or evicted), their loan count up. The caller synchronizes moe_cache_zone_stream after.
static void moe_cache_band_take(moe_cache_device & device, size_t a, size_t b,
        std::vector<std::pair<moe_cache_pool *, std::vector<int>>> & moved) {
    for (auto & pool_ptr : device.pools) {
        moe_cache_pool & pool = *pool_ptr;
        int s0, s1;
        moe_cache_band_slots(pool, a, b, s0, s1);
        if (s0 >= s1) {
            continue;
        }
        int e0 = s0, e1 = s1;   // a neighbour's loan can only share the first or the last slot
        while (e0 < e1 && pool.band_refs[(size_t)e0] > 0) {
            e0++;
        }
        while (e1 > e0 && pool.band_refs[(size_t)e1 - 1] > 0) {
            e1--;
        }
        moved.push_back({&pool, {}});
        if (e0 < e1) {
            moe_cache_evacuate(device, pool, e0, e1, moved.back().second);
        }
        for (int i = s0; i < s1; i++) {
            pool.band_refs[(size_t)i]++;
        }
    }
}

// BANDS: give [a, b) of the slab back: a slot no other loan covers returns to the free list (it is free, unlinked and
// unmapped since its loan)
static void moe_cache_band_release(moe_cache_device & device, size_t a, size_t b) {
    for (auto & pool_ptr : device.pools) {
        moe_cache_pool & pool = *pool_ptr;
        int s0, s1;
        moe_cache_band_slots(pool, a, b, s0, s1);
        for (int i = s1 - 1; i >= s0; i--) {
            if (pool.band_refs[(size_t)i] > 0 && --pool.band_refs[(size_t)i] == 0) {
                pool.free_slots.push_back(i);
            }
        }
    }
}

// BANDS: where the entries of each shape sit in the slab (offsets, increasing): each shape's share of the slab in
// proportion to its weight (entry size x experts, as the pools' budgets), band after band the whole number that keeps its
// running count closest to that share (a band holds 1.35 entries of a small shape: 1 or 2 by turns), the band's tail
// given to the shapes furthest behind it (the smallest entries, given every tail, would grow past their share). A band
// may end up to one entry into the next (the last one within the slab): nothing is lost between bands. A shape that
// gets no entry: false.
static bool moe_cache_band_layout(const std::vector<moe_cache_shape *> & shapes, size_t band, int n_bands,
        std::vector<std::vector<size_t>> & offs) {
    const size_t n = shapes.size();
    long double total = 0.0;
    size_t largest = 0;
    for (const moe_cache_shape * sh : shapes) {
        total += (long double)sh->expert_size * (long double)sh->n_entries;
        largest = std::max(largest, sh->expert_size);
    }
    std::vector<long double> share(n);   // entries per band
    for (size_t i = 0; i < n; i++) {
        share[i] = (long double)band * (long double)shapes[i]->n_entries / total;
    }
    offs.assign(n, {});
    std::vector<long long> have(n, 0);
    size_t at = 0;   // where the next band's entries start (past the band's beginning when the previous one ran over)
    for (int c = 0; c < n_bands; c++) {
        // the band's entries end at most one entry past its end (the last band's within the slab)
        const size_t room = (size_t)(c + 1) * band + (c + 1 < n_bands ? largest : 0) - at;
        std::vector<long long> k(n);
        size_t used = 0;
        for (size_t i = 0; i < n; i++) {
            k[i] = std::max<long long>(0, (long long)(share[i] * (long double)(c + 1)) - have[i]);
            used += (size_t)k[i] * shapes[i]->expert_size;
        }
        // behind (> 0) or ahead (< 0) of its share after this band, in its own entries per band
        auto lag = [&](size_t i) { return (share[i] * (long double)(c + 1) - (long double)(have[i] + k[i])) / share[i]; };
        while (used > room) {
            size_t best = n;
            for (size_t i = 0; i < n; i++) {
                if (k[i] > 0 && (best == n || lag(i) < lag(best))) {
                    best = i;
                }
            }
            k[best]--;
            used -= shapes[best]->expert_size;
        }
        for (;;) {
            size_t best = n;
            for (size_t i = 0; i < n; i++) {
                if (used + shapes[i]->expert_size <= room && lag(i) > 0 && (best == n || lag(i) > lag(best))) {
                    best = i;
                }
            }
            if (best == n) {
                break;
            }
            k[best]++;
            used += shapes[best]->expert_size;
        }
        for (size_t i = 0; i < n; i++) {
            for (long long j = 0; j < k[i]; j++) {
                offs[i].push_back(at);
                at += shapes[i]->expert_size;
            }
            have[i] += k[i];
        }
    }
    for (size_t i = 0; i < n; i++) {
        if (have[i] <= 0 || have[i] > INT_MAX / 2) {
            return false;
        }
    }
    return true;
}

// BANDS: the shapes' pools in one slab of bands, each band holding whole entries of every shape (moe_cache_band_layout):
// what stays at a band's tail is lost for good (~0.7 % at 128 MiB). A run of bands is then a run of slots at the same
// place in every pool, so a loan of any size up to the slab takes from every shape alike and finds one contiguous piece.
static bool moe_cache_allocate_banded(moe_cache_session & session, moe_cache_device & device,
        const std::vector<moe_cache_shape *> & shapes, size_t budget) {
    const size_t band = moe_cache_band_bytes();
    if (!band || shapes.empty()) {
        return false;
    }
    for (const moe_cache_shape * sh : shapes) {
        if (sh->expert_size == 0 || sh->n_layers <= 0 || sh->n_entries == 0 || ggml_type_is_exl3((ggml_type)sh->role_type[MOE_ROLE_GATE]) ||
                sh->expert_size > band) {
            return false;
        }
    }
    int n_bands = (int)std::min<size_t>(budget / band, INT_MAX / 2);
    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    while (n_bands >= 8) {
        if (!moe_cache_fail(session, "slab") && cudaMalloc((void **)&slab, (size_t)n_bands * band) == cudaSuccess) {
            break;
        }
        (void)cudaGetLastError();
        slab = nullptr;
        n_bands /= 2;
    }
    if (!slab) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d banded slab refused: a slab per pool\n", device.physical);
        return false;
    }
    std::vector<std::vector<size_t>> offs;
    if (!moe_cache_band_layout(shapes, band, n_bands, offs)) {
        (void)cudaFree(slab);
        return false;   // a band too small for a share of every shape
    }
    const size_t allocated = (size_t)n_bands * band;
    moe_cache_budget_allocation(device, allocated, true);
    device.band_slab = slab;
    device.band_bytes = band;
    device.n_bands = n_bands;
    device.band_zone_begin = (size_t)n_bands * band;
    size_t placed = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        moe_cache_shape & shape = *shapes[i];
        shape.finished = true;
        std::unique_ptr<moe_cache_pool> pool(new moe_cache_pool());
        const int slot_count = (int)offs[i].size();
        pool->expert_size = moe_cache_entry_layout(shape.role_bytes, shape.role_pad, pool->role_offset);
        for (int r = 0; r < MOE_ROLES; r++) {
            pool->role_bytes[r] = shape.role_bytes[r];
            pool->role_pad[r] = shape.role_pad[r];
        }
        pool->wtype = shape.role_type[MOE_ROLE_GATE];
        pool->slab = slab;   // shared: freed once, with the device's band_slab
        pool->n_slots = slot_count;
        pool->gpu_base = slab;
        pool->gpu_extent = (int64_t)slot_count;
        pool->zone_end = slot_count;
        pool->ranges.push_back({slab, 0, slot_count, 0});
        pool->covers_all_entries = (uint64_t)slot_count >= shape.n_entries;
        pool->band_base = slab;
        pool->band_bytes = band;
        pool->band_ptr.resize((size_t)slot_count);
        for (int j = 0; j < slot_count; j++) {
            pool->band_ptr[(size_t)j] = slab + offs[i][(size_t)j];
        }
        placed += (size_t)slot_count * pool->expert_size;
        pool->band_slots = slot_count;
        pool->band_refs.assign((size_t)slot_count, 0);
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);   // the low bands fill first, the tail (lent first) last
        }
        device.pools.push_back(std::move(pool));
        shape.pool = (int)device.pools.size() - 1;
        MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: experts %s/%s/%s (up/gate/down) entry=%zu KiB slots=%d (%.2f a band) experts=%llu coverage=%s total=%zu MiB\n",
                device.physical, shape.pool, ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_UP]),
                ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_GATE]), ggml_type_name((ggml_type)shape.role_type[MOE_ROLE_DOWN]),
                shape.expert_size >> 10, slot_count, (double)slot_count / n_bands, (unsigned long long)shape.n_entries,
                device.pools.back()->covers_all_entries ? "complete" : "partial", ((size_t)slot_count * shape.expert_size) >> 20);
    }
    device.allocated_bytes += allocated;
    MOE_CACHE_LOG("[moe-cache] CUDA%d banded slab: %d bands of %zu MiB (%zu MiB), %zu KiB unused (%.2f %%)\n",
            device.physical, n_bands, band >> 20, allocated >> 20, (allocated - placed) >> 10, 100.0 * (double)(allocated - placed) / (double)allocated);
    if (!moe_cache_start_worker(session, device)) {
        GGML_ABORT("moe-cache: the fill worker did not start");
    }
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
static void moe_cache_preload(const moe_cache_session & session, moe_cache_device & device);
static void moe_cache_arena_seed(moe_cache_device & device);
static void moe_cache_profile_save_heat(moe_cache_session & session);

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
static bool moe_cache_pin_enabled() {
    static const bool enabled = !getenv("GGML_CUDA_MOE_PIN") || atoi(getenv("GGML_CUDA_MOE_PIN")) != 0;
    return enabled;
}

// KFD charges a registered range against its system memory limit until the mapping itself goes away: hipHostUnregister
// keeps it (ROCm caches the registration), and a refused registration stays half done ("already allocated by SVM"), the
// copies that later read the range hang. Mapping the same file pages again at the same address (MAP_FIXED, as llama.cpp
// maps the model: shared, read-only) drops the old mapping with its registration and its charge; the contents, the
// address and the page cache are unchanged. Linux only, where the limit is.
static bool moe_cache_remap_range(uintptr_t ib, uintptr_t ie) {
#if defined(__linux__)
    FILE * maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        return false;
    }
    char line[4096];
    bool done = false;
    while (!done && fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0, end = 0, offset = 0;
        char path[4096] = {};
        if (sscanf(line, "%llx-%llx %*s %llx %*s %*s %4095[^\n]", &start, &end, &offset, path) < 4 ||
            path[0] != '/' || ib < start || ie > end) {
            continue;
        }
        const int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            void * p = mmap((void *) ib, ie - ib, PROT_READ, MAP_SHARED | MAP_FIXED, fd, (off_t) (offset + (ib - start)));
            close(fd);
            if (p == (void *) ib) {
                (void) posix_madvise(p, ie - ib, POSIX_MADV_RANDOM);
                done = true;
            }
        }
        break;
    }
    fclose(maps);
    return done;
#else
    GGML_UNUSED(ib);
    GGML_UNUSED(ie);
    return false;
#endif
}

// a range back to its file's pages: one loaded directly (anonymous now, its file recorded), else through /proc/self/maps
static bool moe_cache_remap_any(const moe_cache_device & device, uintptr_t ib, uintptr_t ie) {
#if defined(__linux__)
    for (const auto & f : device.pin_files) {
        if (ib >= f.start && ie <= f.end) {
            const int fd = open(f.path.c_str(), O_RDONLY);
            if (fd < 0) {
                return false;
            }
            void * p = mmap((void *) ib, ie - ib, PROT_READ, MAP_SHARED | MAP_FIXED, fd, (off_t) (f.offset + (ib - f.start)));
            close(fd);
            if (p != (void *) ib) {
                return false;
            }
            (void) posix_madvise(p, ie - ib, POSIX_MADV_RANDOM);
            return true;
        }
    }
#endif
    return moe_cache_remap_range(ib, ie);
}

// one expert tensor registered in place (its range rounded outwards to pages), false if ROCm refuses it. Portable: with
// several GPUs a range registered for one device's cache is also copied by another (the first GPU's prefetch copies
// every layer); a range another device of the session registered is not registered again
static bool moe_cache_pin_tensor(const moe_cache_session & session, moe_cache_device & device, const void * base,
                                 const moe_cache_seen_tensor & t, size_t & bytes) {
    const uintptr_t b = (uintptr_t) base;
    const uintptr_t ib = b & ~(uintptr_t) 4095;
    const uintptr_t ie = GGML_PAD(b + t.bytes, 4096);
    for (const auto & d : session.devices) {
        for (const auto & range : d->pinned) {   // a packed layer: its three tensors share one range
            if (ib >= range.first && ie <= range.second) {
                return true;
            }
        }
    }
    // mapped: the dispatcher's kernel copy reads a moved expert there (moe_split_copy_by_kernel). Read-only where the
    // platform has it: some refuse the flag (cudaDevAttrHostRegisterReadOnlySupported 0, a rented RTX 4090 with HMM
    // addressing), which then registered nothing at all, read as KFD's limit
    const unsigned flags = cudaHostRegisterPortable | cudaHostRegisterMapped;
    cudaError_t err = cudaHostRegister((void *) ib, ie - ib, flags | cudaHostRegisterReadOnly);
    if (err == cudaErrorNotSupported) {
        (void) cudaGetLastError();
        err = cudaHostRegister((void *) ib, ie - ib, flags);
    }
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        (void) moe_cache_remap_any(device, ib, ie);   // the half-done registration goes with the old mapping
        device.pin_full = true;
        return false;
    }
    device.pinned.push_back({ib, ie});
    bytes += ie - ib;
    return true;
}

// PIN, direct load: each expert range (merged, page-aligned) replaced by anonymous memory at the same address (MAP_FIXED)
// and filled from its file with O_DIRECT by reader threads (GGML_CUDA_MOE_ARENA_READERS, 8), in 64 MiB pieces; left[i]
// reaches 0 when range i is filled (loaded[i] false on a failed read), load_cv notified. The experts' host copy is then
// this memory, not the page cache (the same bytes, the same addresses). A range not inside one file mapping stays as it
// is. Returns the bytes read
static size_t moe_cache_pin_load_direct(moe_cache_device & device, const std::vector<std::pair<uintptr_t, uintptr_t>> & merged,
                                        std::vector<std::atomic<int>> & left, std::vector<char> & loaded,
                                        std::vector<std::thread> & readers, std::mutex & mu, std::condition_variable & cv) {
#if defined(__linux__)
    struct mapping { uintptr_t start, end; uint64_t offset; std::string path; };
    std::vector<mapping> maps_;
    {
        FILE * maps = fopen("/proc/self/maps", "r");
        char line[4096];
        while (maps && fgets(line, sizeof(line), maps)) {
            unsigned long long start = 0, end = 0, offset = 0;
            char path[4096] = {};
            if (sscanf(line, "%llx-%llx %*s %llx %*s %*s %4095[^\n]", &start, &end, &offset, path) == 4 && path[0] == '/') {
                maps_.push_back({(uintptr_t) start, (uintptr_t) end, (uint64_t) offset, path});
            }
        }
        if (maps) {
            fclose(maps);
        }
    }
    struct piece { size_t range; uintptr_t at; size_t len; int fd; uint64_t off; };
    std::vector<piece> pieces;
    std::map<std::string, int> fds;
    size_t total = 0;
    const size_t piece_bytes = (size_t) 64 << 20;
    for (size_t i = 0; i < merged.size(); i++) {
        left[i].store(0);
        const auto & r = merged[i];
        const mapping * m = nullptr;
        for (const auto & x : maps_) {
            if (r.first >= x.start && r.second <= x.end) {
                m = &x;
                break;
            }
        }
        if (!m) {
            continue;
        }
        auto it = fds.find(m->path);
        if (it == fds.end()) {
            it = fds.emplace(m->path, open(m->path.c_str(), O_RDONLY | O_DIRECT)).first;
        }
        if (it->second < 0) {
            continue;
        }
        void * p = mmap((void *) r.first, r.second - r.first, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        if (p != (void *) r.first) {
            continue;   // MAP_FIXED failing leaves the old mapping
        }
        const uint64_t off = m->offset + (r.first - m->start);
        device.pin_files.push_back({r.first, r.second, m->path, off});
        int n = 0;
        for (uintptr_t a = r.first; a < r.second; a += piece_bytes, n++) {
            pieces.push_back({i, a, std::min(piece_bytes, (size_t) (r.second - a)), it->second, off + (a - r.first)});
        }
        left[i].store(n);
        total += r.second - r.first;
    }
    static const int n_threads = getenv("GGML_CUDA_MOE_ARENA_READERS") ? std::max(1, atoi(getenv("GGML_CUDA_MOE_ARENA_READERS"))) : 8;
    auto next = std::make_shared<std::atomic<size_t>>(0);
    auto files = std::make_shared<std::map<std::string, int>>(std::move(fds));
    auto jobs = std::make_shared<std::vector<piece>>(std::move(pieces));
    auto open_readers = std::make_shared<std::atomic<int>>(n_threads);
    for (int t = 0; t < n_threads; t++) {
        readers.emplace_back([&, next, files, jobs, open_readers] {
            for (size_t k; (k = next->fetch_add(1)) < jobs->size();) {
                const piece & q = (*jobs)[k];
                size_t got = 0;
                bool eof = false;
                while (got < q.len) {
                    const ssize_t n = pread(q.fd, (char *) q.at + got, q.len - got, (off_t) (q.off + got));
                    if (n < 0 && errno == EINTR) {
                        continue;
                    }
                    eof = n == 0;
                    if (n <= 0) {
                        break;
                    }
                    got += (size_t) n;
                }
                // short only at the end of the file, where the range's last page goes past it (zeros there, as mapped)
                const bool ok = got == q.len || (eof && got > 0);
                std::lock_guard<std::mutex> lock(mu);
                if (!ok) {
                    loaded[q.range] = 0;
                }
                if (left[q.range].fetch_sub(1) == 1) {
                    cv.notify_all();
                }
            }
            if (open_readers->fetch_sub(1) == 1) {
                for (const auto & f : *files) {
                    if (f.second >= 0) {
                        close(f.second);
                    }
                }
            }
        });
    }
    return total;
#else
    GGML_UNUSED(device); GGML_UNUSED(merged); GGML_UNUSED(left); GGML_UNUSED(loaded); GGML_UNUSED(readers);
    GGML_UNUSED(mu); GGML_UNUSED(cv);
    return 0;
#endif
}

static void moe_cache_pin_experts(const moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_pin_enabled() || !device.pinned.empty()) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    ggml_cuda_set_device(device.logical);
    size_t bytes = 0, fail = 0;
    // the tensors' page ranges, merged where they overlap (the three tensors of a packed layer), registered once each
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    for (const auto & t : device.seen_tensors) {
        const uintptr_t b = (uintptr_t) t.first;
        ranges.push_back({b & ~(uintptr_t) 4095, GGML_PAD(b + t.second.bytes, 4096)});
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto & r : ranges) {
        if (!merged.empty() && r.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, r.second);
        } else {
            merged.push_back(r);
        }
    }
    // several cards that do not all fit (moe-ram-budget.h): the ranges under this card's quota only; the others stay
    // as mapped, in the page cache, not registered (fills copied through the staging buffer)
    size_t budget_left_out = 0, budget_left_bytes = 0;
    if (device.pin_quota != SIZE_MAX) {
        std::vector<char> keep(merged.size(), 0);
        (void) moe_ram_select(merged, device.pin_quota, keep);
        std::vector<std::pair<uintptr_t, uintptr_t>> kept;
        for (size_t i = 0; i < merged.size(); i++) {
            if (keep[i]) {
                kept.push_back(merged[i]);
            } else {
                budget_left_out++;
                budget_left_bytes += merged[i].second - merged[i].first;
            }
        }
        merged.swap(kept);
    }
    // direct load (device.pin_direct): reader threads fill the ranges, each registered as soon as it is filled
    std::vector<std::atomic<int>> left(merged.size());
    std::vector<char> loaded(merged.size(), 1);
    std::vector<std::thread> readers;
    std::mutex load_mu;
    std::condition_variable load_cv;
    size_t direct_bytes = 0;
    if (device.pin_direct) {
        direct_bytes = moe_cache_pin_load_direct(device, merged, left, loaded, readers, load_mu, load_cv);
    }
    // until KFD's limit (about RAM - RAM/64 - 1.5 GiB for all processes: 48.6 GiB of experts here, more than a 32 or
    // 48 GB machine allows) refuses one; then 1 GiB of the last ones is given back (unregistered and mapped again) to
    // the pinned buffers allocated after this. The others are copied through a staging buffer, as with
    // GGML_CUDA_MOE_PIN=0
    for (size_t i = 0; i < merged.size(); i++) {
        const auto & r = merged[i];
        if (!readers.empty()) {
            std::unique_lock<std::mutex> lock(load_mu);
            load_cv.wait(lock, [&] { return left[i].load() == 0; });
        }
        if (!loaded[i]) {
            (void) moe_cache_remap_any(device, r.first, r.second);   // a failed read: the file's pages again
            continue;
        }
        moe_cache_seen_tensor whole;
        whole.bytes = r.second - r.first;
        if (!moe_cache_pin_tensor(session, device, (const void *) r.first, whole, bytes)) {
            break;
        }
    }
    for (auto & t : readers) {
        t.join();
    }
    if (device.pin_full && device.pin_direct) {
        // the ranges after the refused one were filled but are not registered: their file's pages again, the RAM back
        for (const auto & r : merged) {
            bool registered = false;
            for (const auto & q : device.pinned) {
                registered = registered || (r.first >= q.first && r.second <= q.second);
            }
            if (!registered) {
                (void) moe_cache_remap_any(device, r.first, r.second);
            }
        }
    }
    if (device.pin_full) {
        size_t freed = 0;
        while (freed < ((size_t) 1 << 30) && !device.pinned.empty()) {
            const auto r = device.pinned.back();
            (void) cudaHostUnregister((void *) r.first);
            (void) moe_cache_remap_any(device, r.first, r.second);
            freed += r.second - r.first;
            bytes -= r.second - r.first;
            device.pinned.pop_back();
        }
        fail = merged.size() - device.pinned.size();
    }
    std::sort(device.pinned.begin(), device.pinned.end());
    fail += budget_left_out;
    MOE_CACHE_LOG("[moe-cache] CUDA%d pin: %zu expert tensors registered (%zu MiB) in %.1f s, %zu left out%s%s%s\n",
            device.physical, device.pinned.size(), bytes >> 20, (ggml_time_us() - t0) / 1e6, fail,
            budget_left_out ? (" (" + std::to_string(budget_left_out) + " over the RAM budget, " +
                std::to_string(budget_left_bytes >> 20) + " MiB in the page cache)").c_str() : "",
            device.pin_full ? " (KFD's system memory limit reached)" : "",
            device.pin_direct ? (" (read with O_DIRECT: " + std::to_string(direct_bytes >> 20) + " MiB)").c_str() : "");
}

// The expert arena (moe-arena.h), GGML_CUDA_MOE_ARENA_MB=N: N MiB of anonymous memory
// registered once, now, before the expert tensors are (it comes first under KFD's limit). One arena per expert shape,
// its slots the size of that shape's expert read with O_DIRECT; never more slots than the shape has experts. When the
// size cannot hold them all, it is shared out by the shapes' bytes (as the VRAM cache's budget). Registered in pieces of
// whole slots (~1 GiB), the shapes in turn, until KFD refuses one; the refused piece and the rest are unmapped, which
// drops a half-done registration with its charge.
// auto (SIZE_MAX), the default on Linux: see moe_cache_build_pending; 0: none, the experts registered in place from the
// page cache (the default elsewhere: the arena and the direct load read /proc and use O_DIRECT)
static size_t moe_cache_arena_mb() {
    const char * v = getenv("GGML_CUDA_MOE_ARENA_MB");
#if defined(__linux__)
    const size_t dflt = SIZE_MAX;
#else
    const size_t dflt = 0;
#endif
    static const size_t mb = !v ? dflt : strcmp(v, "auto") == 0 ? SIZE_MAX : (size_t) atoll(v);
    return mb;
}

// a number from a file: its first line ("max": SIZE_MAX), or the value after key; SIZE_MAX if absent
static size_t moe_arena_read_number(const std::string & path, const char * key) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (!key) {
            return line == "max" ? SIZE_MAX : (size_t) strtoull(line.c_str(), nullptr, 10);
        }
        if (line.compare(0, strlen(key), key) == 0) {
            return (size_t) strtoull(line.c_str() + strlen(key), nullptr, 10);
        }
    }
    return SIZE_MAX;
}

// GGML_CUDA_MOE_ARENA_MB=auto (buun's rule for his RAM tier): the memory free to take now, less a margin
// (GGML_CUDA_MOE_ARENA_MARGIN_MB, 4096): the smaller of MemAvailable and, under every cgroup v2 limit above this
// process, the limit less what cannot be reclaimed without swap (anonymous, shared, kernel; the page cache can be)
static size_t moe_cache_room_bytes() {
    size_t room = moe_arena_read_number("/proc/meminfo", "MemAvailable:");
    room = room == SIZE_MAX ? SIZE_MAX : room << 10;
    std::ifstream cg("/proc/self/cgroup");
    std::string line;
    while (std::getline(cg, line)) {
        if (line.compare(0, 3, "0::") != 0) {
            continue;
        }
        std::string dir = "/sys/fs/cgroup" + line.substr(3);
        while (dir.size() > strlen("/sys/fs/cgroup")) {
            const size_t limit = moe_arena_read_number(dir + "/memory.max", nullptr);
            if (limit != SIZE_MAX) {
                size_t used = 0;
                for (const char * key : {"anon ", "shmem ", "kernel "}) {
                    const size_t b = moe_arena_read_number(dir + "/memory.stat", key);
                    used += b == SIZE_MAX ? 0 : b;
                }
                room = std::min(room, used < limit ? limit - used : 0);
            }
            dir.resize(dir.rfind('/'));
        }
    }
    return room;
}

static size_t moe_cache_arena_margin_mb() {
    const char * m = getenv("GGML_CUDA_MOE_ARENA_MARGIN_MB");
    return m ? (size_t) atoll(m) : 4096;
}

static size_t moe_cache_arena_auto_bytes() {
    const size_t room = moe_cache_room_bytes();
    const size_t margin = moe_cache_arena_margin_mb() << 20;
    return room != SIZE_MAX && room > margin ? room - margin : 0;
}

// ARENA: what a restoration brings back once generation starts. GGML_CUDA_MOE_ARENA_RESTORE=prompt: the experts the
// prompt routed most rows to (counted on the device during its prefill, ggml_cuda_moe_prefill_counts), else (evicted,
// the default) the experts the prefill's zone evicted, hottest first
static bool moe_arena_restore_prompt() {
    static const bool prompt = getenv("GGML_CUDA_MOE_ARENA_RESTORE") && !strcmp(getenv("GGML_CUDA_MOE_ARENA_RESTORE"), "prompt");
    return prompt;
}

// the prefill's row counts: one row of n experts per layer number, on the device (one device)
static std::atomic<int *> g_prefill_counts{nullptr};
static int g_prefill_counts_n = 0, g_prefill_counts_rows = 0;

int * ggml_cuda_moe_prefill_counts(const char * w_name, int n_expert) {
    int * d = g_prefill_counts.load(std::memory_order_acquire);
    if (!d || !w_name || n_expert != g_prefill_counts_n) {
        return nullptr;
    }
    const char * b = strstr(w_name, "blk.");
    int layer = -1, end = 0;
    if (!b || sscanf(b, "blk.%d.%n", &layer, &end) != 1 || strncmp(b + end, "ffn_gate_exps", 13) != 0 ||
            layer < 0 || layer >= g_prefill_counts_rows) {
        return nullptr;
    }
    return d + (size_t) layer * n_expert;
}

static __global__ void k_moe_count_rows(const int * __restrict__ bounds, int * __restrict__ counts, int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n) {
        counts[e] += bounds[e + 1] - bounds[e];
    }
}

void ggml_cuda_moe_count_rows(const int * expert_bounds, int * counts, int n_expert, void * stream) {
    k_moe_count_rows<<<(n_expert + 255) / 256, 256, 0, (cudaStream_t) stream>>>(expert_bounds, counts, n_expert);
}

// ARENA: the mappings of the model files behind the expert tensors seen so far, each file opened with O_DIRECT once;
// again for a layer shown later (the MTP draft's, from its own file), before its first graph (the readers do not run then)
static void moe_arena_open_files(moe_cache_device & device) {
#if defined(__linux__)
    FILE * maps = fopen("/proc/self/maps", "r");
    char line[4096];
    while (maps && fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0, end = 0, offset = 0;
        char path[4096] = {};
        if (sscanf(line, "%llx-%llx %*s %llx %*s %*s %4095[^\n]", &start, &end, &offset, path) < 4 || path[0] != '/') {
            continue;
        }
        bool known = false;
        for (const auto & f : device.arena_files) {
            known = known || (f.start <= (uintptr_t) start && (uintptr_t) end <= f.end);   // merged ranges hold their pieces
        }
        bool experts = false;
        for (const auto & t : device.seen_tensors) {
            const uintptr_t b = (uintptr_t) t.first;
            experts = experts || (b < end && b + t.second.bytes > start);
        }
        if (known || !experts) {
            continue;
        }
        auto it = device.arena_fds.find(path);
        if (it == device.arena_fds.end()) {
            it = device.arena_fds.emplace(path, open(path, O_RDONLY | O_DIRECT)).first;
        }
        if (it->second >= 0) {
            device.arena_files.push_back({(uintptr_t) start, (uintptr_t) end, it->second, (uint64_t) offset});
        }
    }
    if (maps) {
        fclose(maps);
    }
    std::sort(device.arena_files.begin(), device.arena_files.end(),
              [](const moe_cache_device::arena_file & a, const moe_cache_device::arena_file & b) { return a.start < b.start; });
    // one file shows as several consecutive mappings where a madvise split it (the lazy PLE table, the experts left out
    // of the prefetch): merged, or an expert across a split would be in none of them
    std::vector<moe_cache_device::arena_file> merged;
    for (const auto & f : device.arena_files) {
        if (!merged.empty() && merged.back().fd == f.fd && merged.back().end == f.start &&
                merged.back().offset + (merged.back().end - merged.back().start) == f.offset) {
            merged.back().end = f.end;
        } else {
            merged.push_back(f);
        }
    }
    device.arena_files = std::move(merged);
#else
    GGML_UNUSED(device);
#endif
}

static void moe_cache_arena_create(moe_cache_device & device) {
    size_t want = moe_cache_arena_mb() == SIZE_MAX ? moe_cache_arena_auto_bytes() : moe_cache_arena_mb() << 20;
    if (want == 0 || device.arena_base) {
        return;
    }
    // a slot holds an expert read with O_DIRECT: each tensor (with its pad) in the page-aligned span around it, or a
    // packed expert in one
    struct sub { int shape; size_t slot; uint64_t experts; size_t bytes = 0, piece = 0, done = 0; char * base = nullptr; };
    std::vector<sub> subs;
    double need = 0;
    for (size_t k = 0; k < device.shapes.size(); k++) {
        const moe_cache_shape & shape = device.shapes[k];
        if (shape.n_layers > 0 && shape.n_entries > 0) {
            size_t roles = 0;
            for (int r = 0; r < MOE_ROLES; r++) {
                roles += GGML_PAD(shape.role_bytes[r] + shape.role_pad[r] + 4096, 4096);
            }
            subs.push_back({(int) k, std::max(roles, (size_t) GGML_PAD(shape.expert_size + 4096, 4096)), shape.n_entries});
            need += (double) subs.back().slot * subs.back().experts;
        }
    }
    if (subs.empty()) {
        return;
    }
    // under KFD's limit for registered memory (RAM - RAM/64 - 1.5 GiB for all processes), less 2 GiB for what ROCm
    // locks itself (a copy from pageable memory) and the pinned buffers allocated later
    {
        const size_t total = moe_arena_read_number("/proc/meminfo", "MemTotal:");
        if (total != SIZE_MAX) {
            const size_t ram = total << 10;
            const size_t kfd = ram - ram / 64 - ((size_t) 3 << 29) - ((size_t) 2 << 30);
            want = std::min(want, kfd);
        }
    }
    // the shares by bytes, each at least one layer and a half of its experts (a prefill holds a layer's experts while it
    // copies them: below that, a shape's zone cannot take them, they are copied from the mapping), taken from the shapes
    // above their floor. MEASURED: the two one-layer shapes at 243 slots of 512 experts, ~1 450 copies from the mapping
    // per 19k prompt
    std::vector<uint64_t> n_of(subs.size()), floor_of(subs.size());
    double above = 0;
    size_t total = 0;
    for (size_t k = 0; k < subs.size(); k++) {
        const sub & x = subs[k];
        const moe_cache_shape & sh = device.shapes[x.shape];
        const double per_layer = sh.n_layers > 0 ? (double) sh.n_entries / sh.n_layers : 0.0;
        floor_of[k] = std::min<uint64_t>(x.experts, (uint64_t) std::ceil(1.5 * per_layer));
        n_of[k] = want >= need ? x.experts : (uint64_t) ((double) want * x.slot * x.experts / need / x.slot);
        n_of[k] = std::max(n_of[k], floor_of[k]);
        total += (size_t) n_of[k] * x.slot;
        above += (double) (n_of[k] - floor_of[k]) * x.slot;
    }
    if (total > want && above > 0) {
        const double cut = std::min(1.0, (double) (total - want) / above);
        for (size_t k = 0; k < subs.size(); k++) {
            n_of[k] -= (uint64_t) std::ceil(cut * (double) (n_of[k] - floor_of[k]));
        }
    }
    size_t bytes = 0;
    for (size_t k = 0; k < subs.size(); k++) {
        sub & x = subs[k];
        x.bytes = (size_t) n_of[k] * x.slot;
        x.piece = std::max<size_t>(1, ((size_t) 1 << 30) / x.slot) * x.slot;   // a copy reads one registration
        bytes += x.bytes;
    }
    if (bytes == 0) {
        return;
    }
#if defined(__linux__)
    const int64_t t0 = ggml_time_us();
    moe_arena_open_files(device);
    if (device.arena_files.empty()) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d arena: no model file opened with O_DIRECT, no arena\n", device.physical);
        return;
    }
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d arena: mmap of %zu MiB failed\n", device.physical, bytes >> 20);
        return;
    }
    ggml_cuda_set_device(device.logical);
    char * base = (char *) p;
    {
        char * cur = base;
        for (sub & x : subs) {
            x.base = cur;
            cur += x.bytes;
        }
    }
    // DIAGNOSTIC GGML_CUDA_MOE_ARENA_MAPPED=0: not mapped into the GPU's address space (only SDMA copies read it)
    static const bool mapped = !getenv("GGML_CUDA_MOE_ARENA_MAPPED") || atoi(getenv("GGML_CUDA_MOE_ARENA_MAPPED")) != 0;
    bool refused = false;
    for (bool more = true; more && !refused;) {
        more = false;
        for (sub & x : subs) {
            if (x.done == x.bytes) {
                continue;
            }
            const size_t n = std::min(x.piece, x.bytes - x.done);
            if (cudaHostRegister(x.base + x.done, n, cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0)) != cudaSuccess) {
                (void) cudaGetLastError();
                refused = true;
                break;
            }
            device.arena_pieces.push_back({x.base + x.done, n});
            x.done += n;
            more = more || x.done < x.bytes;
        }
    }
    size_t done = 0;
    int n_slots = 0;
    device.arenas.clear();
    device.arenas.resize(device.shapes.size());
    // GGML_CUDA_MOE_ARENA_EVICT=0: once full, an arena keeps its experts
    const bool evict = !getenv("GGML_CUDA_MOE_ARENA_EVICT") || atoi(getenv("GGML_CUDA_MOE_ARENA_EVICT")) != 0;
    // DIAGNOSTIC GGML_CUDA_MOE_ARENA_IDLE=1: registered, never used
    const bool idle = getenv("GGML_CUDA_MOE_ARENA_IDLE") && atoi(getenv("GGML_CUDA_MOE_ARENA_IDLE")) != 0;
    std::string shapes;
    for (sub & x : subs) {
        if (x.done < x.bytes) {
            (void) munmap(x.base + x.done, x.bytes - x.done);   // the refused piece's half-done registration goes with it
        }
        const int n = (int) (x.done / x.slot);
        if (n > 0) {
            auto a = std::make_unique<moe_expert_arena>();
            a->init(x.base, x.slot, idle ? 0 : n);
            a->evict = evict;
            // the zone lent to a prefill: the experts it reads that the arena does not hold, for the layers it has in
            // flight (a read slot is held until its prefetch slot is used again, 4 tensors later: about 2.3 layers),
            // up to the whole arena of a small shape (restored after). MEASURED: 2 layers capped at half the arena
            // left ~2 700 reads per 19k prompt to the mapping
            const moe_cache_shape & sh = device.shapes[x.shape];
            const double per_layer = sh.n_layers > 0 ? (double) sh.n_entries / sh.n_layers : 0.0;
            const double absent = x.experts > 0 ? 1.0 - std::min(1.0, (double) n / x.experts) : 0.0;
            a->zone_slots = std::min(n, (int) std::ceil(2.5 * per_layer * absent));
            device.arenas[x.shape] = std::shared_ptr<moe_expert_arena>(std::move(a));
        }
        done += x.done;
        n_slots += n;
        char b[96];
        snprintf(b, sizeof(b), "%s%d x %zu KiB (%llu experts)", shapes.empty() ? "" : ", ", n, x.slot >> 10,
                 (unsigned long long) x.experts);
        shapes += b;
    }
    if (n_slots == 0) {
        for (const auto & piece : device.arena_pieces) {
            (void) cudaHostUnregister(piece.first);
        }
        device.arena_pieces.clear();
        device.arenas.clear();
        (void) munmap(base, bytes);
        MOE_CACHE_LOG("[moe-cache] CUDA%d arena: nothing registered (KFD's system memory limit)\n", device.physical);
        return;
    }
    device.arena_base = base;
    device.arena_mapped = bytes;
    static const int n_readers = getenv("GGML_CUDA_MOE_ARENA_READERS") ? std::max(1, atoi(getenv("GGML_CUDA_MOE_ARENA_READERS"))) : 8;
    device.arena_readers.reset(new moe_arena_readers(n_readers));
    if (moe_arena_restore_prompt() && !g_prefill_counts.load() && !device.layers.empty()) {
        int rows = 0, n = 0;
        for (const auto & [number, layer] : device.layers) {
            rows = std::max(rows, number + 1);
            n = std::max(n, (int) layer.n_expert);
        }
        int * d = nullptr;
        if (cudaMalloc((void **) &d, (size_t) rows * n * sizeof(int)) == cudaSuccess &&
                cudaMemset(d, 0, (size_t) rows * n * sizeof(int)) == cudaSuccess) {
            g_prefill_counts_n = n;
            g_prefill_counts_rows = rows;
            g_prefill_counts.store(d, std::memory_order_release);
        } else {
            (void) cudaGetLastError();
        }
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d arena: %d slots (%zu MiB of %zu asked, %.0f MiB hold every expert) registered in %.2f s%s: %s\n",
            device.physical, n_slots, done >> 20, want >> 20, need / 1048576.0, (ggml_time_us() - t0) / 1e6,
            refused ? " (KFD's system memory limit reached)" : "", shapes.c_str());
#else
    GGML_UNUSED(device);
#endif
}

static void moe_cache_arena_free(moe_cache_device & device) {
    device.arena_readers.reset();   // first: the restoration's reads still queued run (or give up) before the unmapping
    if (int * d = g_prefill_counts.exchange(nullptr)) {
        (void) cudaFree(d);
    }
    for (const auto & f : device.arena_fds) {
        if (f.second >= 0) {
            close(f.second);
        }
    }
    device.arena_fds.clear();
    device.arena_files.clear();
    if (!device.arena_base) {
        return;
    }
    moe_expert_arena t;   // the counters, summed
    for (const auto & a : device.arenas) {
        if (a) {
            t.writes += a->writes; t.cpu_loads += a->cpu_loads; t.prefill_loads += a->prefill_loads;
            t.load_failures += a->load_failures; t.hits += a->hits; t.cpu_hits += a->cpu_hits;
            t.evictions += a->evictions; t.busy += a->busy; t.refused += a->refused;
            t.zone_busy += a->zone_busy; t.restored += a->restored; t.zone_slots += a->zone_slots;
        }
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d arena: %llu reads with O_DIRECT (%llu for the CPU, %llu for prefills, %llu failed), "
            "%llu fills from it, %llu CPU experts from it, %llu evictions, %llu busy, %llu admissions refused (not in it); "
            "prefill zone %d slots, %llu reads with it full, %llu restored\n", device.physical,
            (unsigned long long) t.writes, (unsigned long long) t.cpu_loads, (unsigned long long) t.prefill_loads,
            (unsigned long long) t.load_failures, (unsigned long long) t.hits,
            (unsigned long long) t.cpu_hits, (unsigned long long) t.evictions, (unsigned long long) t.busy,
            (unsigned long long) t.refused, t.zone_slots, (unsigned long long) t.zone_busy, (unsigned long long) t.restored);
    for (const auto & piece : device.arena_pieces) {
        (void) cudaHostUnregister(piece.first);
    }
    device.arena_pieces.clear();
#if defined(__linux__)
    (void) munmap(device.arena_base, device.arena_mapped);
#endif
    device.arena_base = nullptr;
    device.arenas.clear();
    device.arena_readers.reset();
}

// ARENA: the arena of a layer's shape, null when none is in use
static moe_expert_arena * moe_arena_of(const moe_cache_device & d, const moe_cache_layer * cl) {
    if (!cl || cl->shape < 0 || cl->shape >= (int) d.arenas.size()) {
        return nullptr;
    }
    moe_expert_arena * a = d.arenas[cl->shape].get();
    return a && a->active() ? a : nullptr;
}

// a job's host side: the expert's tensors in the weights (one range for a packed layer)
static void moe_cache_job_sources(const moe_cache_layer & layer, const moe_cache_pool & pool, int32_t expert,
                                  moe_cache_job & job) {
    job.whole = true;
    for (int r = 0; r < MOE_ROLES; r++) {
        job.source[r] = (const char *) layer.tensor[r] + (size_t) expert * layer.stride[r];
        // the pad copied from the next expert's first bytes when they follow (stride = bytes)
        job.pad_follows[r] = expert + 1 < layer.n_expert && layer.stride[r] == layer.bytes[r];
        job.role_bytes[r] = layer.bytes[r];
        job.whole = job.whole && layer.stride[r] == pool.expert_size &&
            (const char *) job.source[r] == (const char *) job.source[0] + pool.role_offset[r];
    }
    job.bytes = pool.expert_size;
}

// ARENA: the host ranges a job reads (one for a packed layer, else one per tensor with its pad)
static int moe_arena_ranges(const moe_cache_job & j, const moe_cache_pool & p, const char ** src, size_t * n) {
    if (j.whole) {
        src[0] = (const char *) j.source[0];
        n[0] = j.bytes;
        return 1;
    }
    for (int r = 0; r < MOE_ROLES; r++) {
        src[r] = (const char *) j.source[r];
        n[r] = j.role_bytes[r] + (j.pad_follows[r] ? p.role_pad[r] : 0);
    }
    return MOE_ROLES;
}

// ARENA: a job's expert read from the model file with O_DIRECT into arena slot s (taken, being written): each range as
// the page-aligned span around it, the slot's part[] at its first byte. False on any failure (the slot is then aborted)
static bool moe_arena_load(const moe_cache_device & d, moe_expert_arena & ar, int s, const moe_cache_job & j,
                           const moe_cache_pool & p) {
#if defined(__linux__)
    const char * src[MOE_ROLES];
    size_t n[MOE_ROLES];
    const int nr = moe_arena_ranges(j, p, src, n);
    char * cur = ar.ptr(s);
    char * const end = cur + ar.slot_bytes;
    const char * part[MOE_ROLES] = {};
    for (int r = 0; r < nr; r++) {
        const uintptr_t a = (uintptr_t) src[r];
        auto f = std::upper_bound(d.arena_files.begin(), d.arena_files.end(), a,
                                  [](uintptr_t x, const moe_cache_device::arena_file & y) { return x < y.start; });
        if (f != d.arena_files.begin() && a >= (f - 1)->start && a < (f - 1)->end && a + n[r] > (f - 1)->end) {
            // the last expert of a file: its entry's pad goes past the file's mapping (zeros there, as mapped)
            const size_t inside = (size_t) ((f - 1)->end - a);
            memset(cur + (a & 4095) + inside, 0, std::min(n[r] - inside, (size_t) (end - (cur + (a & 4095) + inside))));
            n[r] = inside;
        }
        if (f == d.arena_files.begin() || a + n[r] > (--f)->end) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 4) {
                GGML_LOG_WARN("[moe-cache] arena: read of %p (%zu bytes) failed: in no model file opened with O_DIRECT\n",
                              (const void *) a, n[r]);
            }
            return false;
        }
        const uint64_t off = f->offset + (a - f->start);
        const uint64_t aligned = off & ~(uint64_t) 4095;
        const size_t lead = (size_t) (off - aligned);
        const size_t len = GGML_PAD(lead + n[r], 4096);
        if (cur + len > end) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 4) {
                GGML_LOG_WARN("[moe-cache] arena: read of %p failed: %zu bytes past its slot of %zu\n",
                              (const void *) a, (size_t) (cur + len - end), ar.slot_bytes);
            }
            return false;
        }
        size_t got = 0;
        while (got < len) {
            const ssize_t k = pread(f->fd, cur + got, len - got, (off_t) (aligned + got));
            if (k < 0 && errno == EINTR) {
                continue;
            }
            if (k <= 0) {
                break;   // the end of the file: the span's tail is past it
            }
            got += (size_t) k;
        }
        if (got < lead + n[r]) {
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1) < 4) {
                GGML_LOG_WARN("[moe-cache] arena: read of %p failed: %zu of %zu bytes at file offset %llu (errno %d)\n",
                              (const void *) a, got, lead + n[r], (unsigned long long) aligned, errno);
            }
            return false;
        }
        part[r] = cur + lead;
        cur += len;
    }
    auto & x = ar.slots[s];
    for (int r = 0; r < MOE_ROLES; r++) {
        x.part[r] = j.whole ? part[0] + p.role_offset[r] : part[r];
    }
    return true;
#else
    GGML_UNUSED(d); GGML_UNUSED(ar); GGML_UNUSED(s); GGML_UNUSED(j); GGML_UNUSED(p);
    return false;
#endif
}

static size_t moe_cache_env_mb(const char * name, size_t dflt) {
    const char * v = getenv(name);
    return v ? (size_t) atoll(v) : dflt;
}

// whether this card's expert ranges are registered in place (all, or its quota of the session's budget) or go to the
// arena (moe-ram-budget.h): one card, every range when they fit both the arena's room (moe_cache_arena_auto_bytes) and
// KFD's limit for registered memory (RAM - RAM/64 - 1.5 GiB for all processes) less 1 GiB for the pinned buffers
// allocated later; several cards, the budget decided once at the first card's decision. Sets device.pin_quota

// the experts' host RAM, decided once for all the cards of the process (moe-ram-budget.h): one plan for every session,
// as each stage of a pipeline over several GPUs has a session of its own (a plan per session decided again at the second
// card's session, with the room the first card had already taken: 9 of 22 GiB registered, the rest faulted in at 175 t/s)
static moe_ram_plan g_ram_plan;
static std::mutex g_ram_plan_mu;   // taken after a session's mu, never before

static moe_ram_mode moe_cache_experts_fit(moe_cache_session & session, moe_cache_device & device) {
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    for (const auto & t : device.seen_tensors) {
        const uintptr_t b = (uintptr_t) t.first;
        ranges.push_back({b & ~(uintptr_t) 4095, GGML_PAD(b + t.second.bytes, 4096)});
    }
    std::sort(ranges.begin(), ranges.end());
    size_t bytes = 0;
    uintptr_t end = 0;
    for (const auto & r : ranges) {
        const uintptr_t b = std::max(r.first, end);
        bytes += r.second > b ? r.second - b : 0;
        end = std::max(end, r.second);
    }
    const size_t total = moe_arena_read_number("/proc/meminfo", "MemTotal:");
    if (total == SIZE_MAX) {
        return MOE_RAM_ARENA;
    }
    const size_t ram = total << 10;
    const size_t kfd = ram - ram / 64 - ((size_t) 3 << 29);
    std::lock_guard<std::mutex> ram_lock(g_ram_plan_mu);
    moe_ram_plan & plan = g_ram_plan;
    const bool first = !plan.decided;
    // GGML_CUDA_MOE_PIN_RESERVE_MB: the whole reserve of several cards; else GGML_CUDA_MOE_PIN_RESERVE_BASE_MB (10240)
    // plus GGML_CUDA_MOE_PIN_RESERVE_CARD_MB (3072) per card after the first
    const size_t reserve_one = moe_ram_reserve_bytes(1, moe_cache_arena_margin_mb(), 0, 0);
    const size_t reserve_multi = getenv("GGML_CUDA_MOE_PIN_RESERVE_MB")
        ? moe_cache_env_mb("GGML_CUDA_MOE_PIN_RESERVE_MB", 0) * moe_ram_mib
        : moe_ram_reserve_bytes(plan.n_devices, moe_cache_arena_margin_mb(),
              moe_cache_env_mb("GGML_CUDA_MOE_PIN_RESERVE_BASE_MB", 10240), moe_cache_env_mb("GGML_CUDA_MOE_PIN_RESERVE_CARD_MB", 3072));
    const size_t room_raw = moe_cache_room_bytes();
    const moe_ram_choice c = moe_ram_choose(plan, bytes, room_raw, kfd, reserve_one, reserve_multi);
    device.pin_quota = c.mode == MOE_RAM_PIN_PART ? c.quota : SIZE_MAX;
    if (plan.n_devices <= 1 || plan.total == 0) {
        const size_t room = room_raw == SIZE_MAX ? 0 : (room_raw > reserve_one ? room_raw - reserve_one : 0);
        MOE_CACHE_LOG("[moe-cache] CUDA%d experts: %zu MiB, room %zu MiB, KFD limit ~%zu MiB: %s\n", device.physical,
                bytes >> 20, room >> 20, kfd >> 20, c.mode == MOE_RAM_PIN_ALL ? "registered in place, read with O_DIRECT" : "the arena");
    } else {
        if (first) {
            MOE_CACHE_LOG("[moe-cache] host RAM for the experts of %d cards: %zu MiB in all, room %zu MiB, reserve %zu MiB, "
                    "KFD limit ~%zu MiB: budget %zu MiB, %s\n", plan.n_devices, plan.total >> 20,
                    plan.room >> 20, reserve_multi >> 20, kfd >> 20, plan.budget >> 20,
                    plan.all_fit ? "every card registers all its experts" : "each card registers its share");
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d experts: %zu MiB, room %zu MiB (session budget %zu MiB, %zu taken): %s %zu MiB, "
                "read with O_DIRECT%s\n", device.physical, bytes >> 20, plan.room >> 20, plan.budget >> 20, plan.used >> 20,
                c.mode == MOE_RAM_PIN_ALL ? "registered in place" : "its share registered in place,", c.quota >> 20,
                c.mode == MOE_RAM_PIN_ALL ? "" : ", the rest left in the page cache");
    }
    return c.mode;
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
    // the main model's layers (a guest's, counted in the pools' shares, are not: the deferred admissions follow the
    // main model's pass)
    for (const auto & [number, layer] : device.layers) {
        if (!layer.guest) {
            device.main_first = device.main_first < 0 ? number : std::min(device.main_first, number);
            device.main_last = std::max(device.main_last, number);
        }
    }
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_layers > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    // GGML_CUDA_MOE_ARENA_MB=auto: every expert registered in place and read with O_DIRECT (moe_cache_pin_load_direct)
    // when they all fit (the arena's room and KFD's limit), else the arena of slots without registration in place.
    // GGML_CUDA_MOE_PIN_DIRECT=1: the direct load with the in-place registration whatever the size
    moe_cache_device * owner = session.devices.empty() ? nullptr : session.devices[0].get();
    const bool share_host = moe_cache_expert_split() && owner && owner != &device && owner->host_ready;
    if (!share_host) {
        const char * d = getenv("GGML_CUDA_MOE_PIN_DIRECT");
        device.pin_direct = d && atoi(d) != 0;
        if (moe_cache_arena_mb() == SIZE_MAX && moe_cache_pin_enabled()) {
            device.pin_direct = moe_cache_experts_fit(session, device) != MOE_RAM_ARENA;
            device.arena_off = device.pin_direct;
            device.pin_off = !device.pin_direct;
        }
    }
    if (share_host) {
        // EXPERT SPLIT: the first card's pin and arena, registered Portable: this card's fills read them as they are
        device.pin_direct = owner->pin_direct;
        device.arena_off = owner->arena_off;
        device.pin_off = owner->pin_off;
        device.pinned = owner->pinned;
        device.arenas = owner->arenas;
        device.host_shared = true;
        MOE_CACHE_LOG("[moe-cache] CUDA%d (device %d) expert split: the experts' host memory of device %d shared (%zu "
                "registered ranges, %zu arenas)\n", device.physical, device.logical, owner->logical, device.pinned.size(),
                device.arenas.size());
    } else {
        if (moe_cache_expert_split() && owner && owner != &device) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d (device %d) expert split: the first card has no host memory yet, this one "
                    "takes its own\n", device.physical, device.logical);
        }
        if (!device.arena_off) {
            moe_cache_arena_create(device);
        }
        if (!device.pin_off) {
            moe_cache_pin_experts(session, device);
        }
        device.host_ready = true;
    }

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
    // each shape's share of the budget by weight, so that every layer holds the same share of its experts whatever its
    // entry size; the floor (moe_cache_pool_slots_min entries) is a minimum, not a bonus (added to the share, it gave
    // the one-layer pools of a seven-shape model +50 % of their part), and the shapes after this one keep theirs
    // BANDS: the first build of a device, every shape in one slab of bands
    if (device.pools.empty() && moe_cache_allocate_banded(session, device, pending, remaining)) {
        pending.clear();
    }
    for (moe_cache_shape * shape : pending) {
        const long double weight =
            (long double)shape->expert_size * (long double)shape->n_entries;
        const size_t minimum = shape->expert_size * moe_cache_pool_slots_min;
        minimum_remaining -= minimum;   // the floors of the shapes after this one
        size_t share = std::max(minimum, weighted_share(remaining, weight, total_weight));
        if (remaining > minimum_remaining && share > remaining - minimum_remaining) {
            share = remaining - minimum_remaining;
        }
        if (share > remaining) {
            share = remaining;
        }
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
    }

    moe_cache_setup_zone(device, moe_cache_zone_bytes(session), session.lend_zones);
    moe_cache_attach_test_region(device);
    moe_cache_attach_kv(session, device);
    moe_cache_arena_seed(device);
    moe_cache_preload(session, device);
}

// the ranking (ranking_src) translated into (pool, key) for the layers the pools hold, its order kept
static void moe_cache_ranking_resolve(moe_cache_device & device) {
    device.ranking.clear();
    for (const auto & r : device.ranking_src) {
        const auto l = device.layers.find(r.layer);
        if (l == device.layers.end() || l->second.shape < 0 || r.expert >= l->second.n_expert) {
            continue;
        }
        const int pool_index = device.shapes[l->second.shape].pool;
        if (pool_index >= 0 && pool_index < (int) device.pools.size()) {
            device.ranking.push_back({pool_index, moe_cache_key{l->second.tensor[MOE_ROLE_GATE], r.expert}, r.share});
        }
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d ranking: %zu of %zu pairs in the pools\n", device.physical, device.ranking.size(),
            device.ranking_src.size());
}

// An expert tensor shown to the cache (moe_cache_begin): recorded in its layer (number and role read from its name).
// Once every tensor known has been seen again without a new one (two passes of the scheduler's priming), the complete
// layers are grouped by expert shape and a pool is built per shape. A layer complete after that joins the pool of its
// shape, if there is one. Returns the pool of the tensor's layer, -1 if none (yet).
static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device, const char * name,
        const void * host_base, size_t tensor_size, size_t expert_size, size_t stride,
        int wtype, int64_t n_expert, int64_t n_in) {
    int layer_number = -1;
    const int role = moe_cache_role_of_name(name);
    if (role < 0 || !moe_cache_layer_number(name, layer_number)) {
        return -1;
    }
    const bool first_visit = device.seen_tensors.emplace(
            host_base, moe_cache_seen_tensor{tensor_size, expert_size, wtype, n_expert}).second;
    moe_cache_layer & layer = device.layers[layer_number];
    if (first_visit) {
        layer.guest = !g_session_stack.empty() && g_session_stack.back().guest;
        layer.tensor[role] = host_base;
        layer.bytes[role] = expert_size;
        layer.stride[role] = stride;
        layer.ne0[role] = n_in;
        layer.type[role] = wtype;
        layer.n_expert = layer.n_expert ? std::min(layer.n_expert, n_expert) : n_expert;
        device.layer_of[host_base] = layer_number;
        if (!device.arenas.empty()) {
            moe_arena_open_files(device);   // a layer shown after the arena was built: its file too
        }
        device.visits_since_new_tensor = 0;
        size_t pinned_bytes = 0;   // under a RAM budget (moe-ram-budget.h), a late tensor registered only within the quota
        for (const auto & r : device.pinned) {
            pinned_bytes += r.second - r.first;
        }
        const bool in_quota = device.pin_quota == SIZE_MAX ||
            pinned_bytes + GGML_PAD(tensor_size + 4095, 4096) <= device.pin_quota;
        if (device.pools_built && !device.pinned.empty() && !device.pin_full && in_quota) {
            // a tensor of a scheduler sharing the session (MTP draft), shown after the pools were built: registered
            // like the others, still before its first graph (registering synchronizes every stream)
            size_t bytes = 0;
            ggml_cuda_set_device(device.logical);
            if (moe_cache_pin_tensor(session, device, host_base, device.seen_tensors.at(host_base), bytes)) {
                std::sort(device.pinned.begin(), device.pinned.end());
                MOE_CACHE_LOG("[moe-cache] CUDA%d pin: %s registered late (%zu MiB)\n", device.physical, name, bytes >> 20);
            }
        }
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
            if (!device.ranking_src.empty()) {
                moe_cache_ranking_resolve(device);   // its pairs of the ranking, for the restorations after a prefill
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
    if (moe_cache_expert_split()) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d expert split: planned rows here %lld, in another card's cache %lld, nowhere %lld; "
                "admissions sent to another card %lld; experts moved to another card %lld; rows other cards found here %lld\n",
                device.physical, device.split_local, device.split_remote, device.split_absent, device.split_admit_away,
                device.split_moved_away, device.split_served);
    }
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
    if (device.prefill_resident + device.prefill_streamed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d prefill: experts read in place %lld, copied %lld (%.1f %% in place)\n",
                device.physical, device.prefill_resident, device.prefill_streamed,
                100.0 * device.prefill_resident / (device.prefill_resident + device.prefill_streamed));
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
            // one cache per physical GPU; DIAGNOSTIC (GGML_MOE_CACHE_VIRTUAL_DEVICES=1, with GGML_CUDA_DEVICES and a
            // --moe-cache budget): one per virtual device, to run the multi-GPU paths on a single GPU
            static const bool per_virtual = getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES") && atoi(getenv("GGML_MOE_CACHE_VIRTUAL_DEVICES")) != 0;
            if (!seen_devices.insert(per_virtual ? logical : physical).second) {
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
            session->devices.back()->lfu = session->config.lfu;
            session->devices.back()->backend_ctx = context;
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
            // serial with several GPUs too: the concurrent fills of buun's multi-GPU path were never measured with the
            // hosted MoE (one session lock, one fill lock)
            session->config.serial_fill = true;
            GGML_UNUSED(minimum_capability);
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
            if (it->arena) {
                it->arena->release(it->arena_hold);
            }
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
    if (device.host_shared) {   // the first card's: it unregisters and unmaps them
        device.pinned.clear();
        device.arenas.clear();
    } else if (device.host_keep) {
        // the other cards still read it: released with the session (moe_cache_session_free clears host_keep)
    } else {
        for (const auto & r : device.pinned) {
            (void) cudaHostUnregister((void *) r.first);
        }
        device.pinned.clear();
        moe_cache_arena_free(device);
    }
    if (device.compute_stream) {
        cudaStreamSynchronize(device.compute_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab && pool_ptr->band_slots) {
            pool_ptr->slab = nullptr;   // BANDS: the device's slab, freed below
        } else if (pool_ptr->slab) {
            cudaFree(pool_ptr->slab);
            moe_cache_budget_allocation(
                    device, (size_t)pool_ptr->n_slots * pool_ptr->expert_size, false);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.band_slab) {
        cudaFree(device.band_slab);
        moe_cache_budget_allocation(device, (size_t)device.n_bands * device.band_bytes, false);
        device.band_slab = nullptr;
        device.n_bands = 0;
        device.band_loans.clear();
        device.band_zones = 0;
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
    moe_cache_profile_save_heat(*session);
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

    // the last cards first: they may share the first card's host memory (EXPERT SPLIT), which it frees
    for (auto it = session->devices.rbegin(); it != session->devices.rend(); ++it) {
        auto & device_ptr = *it;
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        device_ptr->host_keep = false;   // the cards sharing it are freed already
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    delete session;
}

// guest: a scheduler sharing the session of another (the MTP draft's, ggml_backend_sched_share_moe_cache); its graphs
// do not count for the restoration after a prefill (a draft graph runs between two prefill ubatches of the main model)
static void moe_cache_session_enter_impl(void * opaque, bool guest) {
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
        g_session_stack.push_back({session, session, guest});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    if (session->active_scopes == 0 && !guest) {
        // a graph starts: after prefill graphs (they lend), the first one that did not lend
        // means generation has started. Zones still lent here were the previous graph's too: a prefill graph after
        // another keeps them without lending again (ggml_backend_sched_compute_splits); counted as lending, else the
        // last ubatch armed the restoration and the generation's compute buffer borrow disarmed it before its start
        bool zones_out = false;
        for (const auto & device_ptr : session->devices) {
            for (const auto & pool_ptr : device_ptr->pools) {
                zones_out = zones_out || pool_ptr->zone_lent;
            }
        }
        if (session->lent_in_last_graph || zones_out) {
            session->lent_since_evictions = true;
        } else if (session->lent_since_evictions) {
            // the restoration starts at this graph's first hosted layer (moe_hosted_plan): a prompt chunk too short to
            // lend but too long to be hosted (the server's checkpoint chunk before the prompt's end) is followed by
            // more prefill, which would evict what it restored
            session->lent_since_evictions = false;
            for (auto & device_ptr : session->devices) {
                device_ptr->rewarm_armed = true;
            }
        }
        session->lent_in_last_graph = false;
    }
    session->active_scopes++;
}

static void moe_cache_session_enter(void * opaque) {
    moe_cache_session_enter_impl(opaque, false);
}

static void moe_cache_session_enter_guest(void * opaque) {
    moe_cache_session_enter_impl(opaque, true);
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
    // expert_size: the step from one expert to the next (nb[2]), its bytes or, in a packed layer, the whole entry
    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    const size_t stride = expert_size;
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size || stride < (size_t)n_out * row_size ||
        (uint64_t)n_expert > SIZE_MAX / stride || stride > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }
    expert_size = (size_t)n_out * row_size;
    const size_t tensor_size = (size_t)(n_expert - 1) * stride + expert_size;
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->stopping) {
        return nullptr;
    }
    // a tensor tied to its layer's device (moe_cache_assign) is learned there only
    const auto tied = session->tensor_devices.find(host_base);
    const int only = tied == session->tensor_devices.end() ? -1 : tied->second;
    for (const auto & device_ptr : session->devices) {
        moe_cache_device & device = *device_ptr;
        if ((only >= 0 && device.logical != only) || device.dead.load() || !moe_cache_prepare_budget(*session, device)) {
            continue;
        }
        try {
            (void) moe_cache_discover_pool(*session, device, name, host_base, tensor_size, expert_size, stride, wtype, n_expert, n_in);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d: tensor discovery failed (host allocation)\n", device.physical);
        }
    }
    return nullptr;
}

static constexpr size_t moe_cache_deferred_max = 4096;   // candidates of a pass kept for its end (a bound only)
// a candidate into the pass's list, by decreasing count, arrival order among equals; demanded again (a plan routes up to
// 3 tokens): its count up, moved after the ones still above it. Short (~100 at most): a scan beats a map
static void moe_cache_defer_add(moe_cache_device & device, int pool_index, const moe_cache_key & key, uint16_t freq, bool contest) {
    auto & c = device.deferred;
    const auto place = [&](auto end) {
        return std::find_if(c.begin(), end, [&](const auto & x) { return x.freq < freq; });
    };
    const auto f = std::find_if(c.begin(), c.end(), [&](const auto & x) { return x.key == key; });
    if (f != c.end()) {
        if (freq > f->freq) {
            f->freq = freq;
            std::rotate(place(f), f, f + 1);
        }
    } else if (c.size() < moe_cache_deferred_max) {
        try {
            c.insert(place(c.end()), {pool_index, key, freq, contest});
        } catch (...) {
        }
    }
}

// ARENA: whether an expert's host copy is the arena: its shape has one (a) and none of its tensors is registered in place
// (a registered tensor is copied and read where it is)
static bool moe_arena_wanted(const moe_cache_device & d, const moe_expert_arena * a, const moe_cache_job & j,
                             const moe_cache_pool & p) {
    if (!a) {
        return false;
    }
    const auto pinned = [&](const void * source, size_t bytes) {
        const uintptr_t src = (uintptr_t) source;
        auto range = std::upper_bound(d.pinned.begin(), d.pinned.end(), std::make_pair(src, UINTPTR_MAX));
        return range != d.pinned.begin() && src + bytes <= (--range)->second;
    };
    if (j.whole) {
        return !pinned(j.source[0], j.bytes);
    }
    for (int r = 0; r < MOE_ROLES; r++) {
        if (pinned(j.source[r], j.role_bytes[r] + (j.pad_follows[r] ? p.role_pad[r] : 0))) {
            return false;
        }
    }
    return true;
}

// ARENA: whether the arena should hold the expert and does not (an expert the tail skip left out, a rewarm's): not proposed
// for admission, since only the arena's experts are. MEASURED against reading such an expert in the background for its
// next demand (16 and 32 GB, 2 ABBA pairs each): the same steps, ~5 000 fewer reads from the file a run and 4 – 10 %
// fewer arena evictions (session lock held)
static bool moe_arena_absent_locked(moe_cache_device & device, const moe_cache_pool & pool, const moe_cache_key & key) {
    const moe_cache_layer * layer = moe_cache_layer_of(device, key.tensor);
    moe_expert_arena * ar = layer ? moe_arena_of(device, layer) : nullptr;
    if (!ar) {
        return false;
    }
    moe_cache_job probe;
    moe_cache_job_sources(*layer, pool, key.expert, probe);
    if (!moe_arena_wanted(device, ar, probe, pool)) {
        return false;
    }
    const auto it = ar->map.find(key);
    return it == ar->map.end() || !ar->slots[it->second].valid;
}

static bool moe_cache_enqueue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker, bool allow_eviction, bool rewarm, bool pregate, uint16_t freq, bool contest) {
    GGML_UNUSED(expert_size);   // the entry: pool.expert_size
    const moe_cache_key key{host_base, expert};
    if (device.deferring && !rewarm && !pregate) {
        moe_cache_defer_add(device, pool_index, key, freq, contest);
        return false;
    }
    const moe_cache_layer * layer = moe_cache_layer_of(device, host_base);
    const size_t queue_limit = session.config.queue_mb << 20;
    if (!layer || expert < 0 || expert >= layer->n_expert || inserts_left <= 0 ||
        (int)device.queue.size() >= session.config.queue_max ||
        pool.expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
        device.insert_skips++;
        return false;
    }
    // ARENA: only an expert the arena holds is admitted, its slot held until the copy is done (the CPU rows or a
    // prefill read it there first; the others are not proposed, moe_arena_absent_locked; the fill thread reads no file).
    // The pre-gating's prefetches (off by default) copy from the mapping
    int hold = -1;
    moe_expert_arena * ar = pregate ? nullptr : moe_arena_of(device, layer);
    if (ar) {
        moe_cache_job probe;
        moe_cache_job_sources(*layer, pool, expert, probe);
        if (moe_arena_wanted(device, ar, probe, pool)) {
            hold = ar->acquire(key);
            if (hold < 0) {
                ar->refused++;
                return false;
            }
        } else {
            ar = nullptr;
        }
    }
    const auto unhold = [&] {
        if (hold >= 0) {
            ar->release(hold);
        }
    };

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
            unhold();
            return false;
        }
        const uint16_t victim_freq = pool.slots[candidate].freq;
        if (contest && victim_freq >= freq) {
            device.admission_skips++;
            unhold();
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
        unhold();
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
            unhold();
            return false;
        }
        moe_cache_direct_set(pool, key, slot_index);

        moe_cache_job job;
        job.pool = pool_index;
        job.slot = slot_index;
        job.generation = slot.generation;
        job.key = key;
        moe_cache_job_sources(*layer, pool, expert, job);
        job.rewarm = rewarm;
        job.arena_hold = hold;
        job.arena = hold >= 0 ? ar : nullptr;
        (pregate ? device.pregate_queue : device.queue).push_back(job);
        device.queued_bytes += pool.expert_size;
    } catch (...) {
        moe_cache_slot_reset(pool, slot_index, true);
        device.insert_skips++;
        unhold();
        return false;
    }
    device.inserts++;
    inserts_left--;
    wake_worker = true;
    return true;
}

// What the restoration after a prefill brings back (GGML_CUDA_MOE_CACHE_REWARM_FROM): the entries the loans evicted
// and the ranking's pairs, merged by LFU count ("both", default), the ranking only ("ranking") or the evicted entries
// only ("evicted"). An evicted entry's count is the one it had when the loan evicted it; a ranked pair's is where
// demand at its share of the ranking would hold it at the current aging period (moe_cache_freq_period), so a ranked
// pair comes before an evicted one only when it is expected to be demanded more. The evicted entries held the previous
// answer's experts, which the next turn reads again when it continues the same text; after a prompt on another subject
// the ranking serves better (simulator, cachesim/restore.py, the hybrid's depth trace: per request, the evicted ones
// first win 65k and 131k by 0.15 - 0.3 point of hits, the ranking first wins 8k and 16k by 0.4 - 0.5; merged by count:
// within 0.01 point of the better in total, 81.01 % against 80.89 % for the evicted then the ranking).
static int moe_cache_rewarm_from() {
    static const int from = [] {
        const char * v = getenv("GGML_CUDA_MOE_CACHE_REWARM_FROM");
        return !v ? 2 : !strcmp(v, "ranking") ? 0 : !strcmp(v, "evicted") ? 1 : 2;
    }();
    return from;
}

// The order of the restoration, built once when it starts: the evicted entries, hottest first, pools in turn, then
// grouped by count (counting sort: counts are at most lfu_cap), merged with the ranking (hottest first, so its counts
// never rise); an evicted entry first on equal counts. Linear in the lists, no sort.
static void moe_cache_rewarm_start_locked(moe_cache_session & session, moe_cache_device & device) {
    device.rewarm_order.clear();
    device.rewarm_next = 0;
    device.renewed_demands = 0;
    const bool use_ranking = moe_cache_rewarm_from() != 1;
    const bool use_evicted = moe_cache_rewarm_from() != 0;
    std::vector<std::pair<int, moe_cache_key>> evicted;
    std::vector<uint16_t> counts;
    if (use_evicted) {
        size_t longest = 0;
        for (const auto & pool_ptr : device.pools) {
            longest = std::max(longest, pool_ptr->rewarm.size());
        }
        for (size_t i = 0; i < longest; i++) {
            for (size_t p = 0; p < device.pools.size(); p++) {
                const auto & list = device.pools[p]->rewarm;   // coldest first
                if (i < list.size()) {
                    evicted.push_back({(int) p, list[list.size() - 1 - i].first});
                    counts.push_back(list[list.size() - 1 - i].second);
                }
            }
        }
    }
    for (auto & pool_ptr : device.pools) {
        pool_ptr->rewarm.clear();
    }
    const int cap = session.config.lfu ? session.config.lfu_cap : 0;
    std::vector<std::vector<std::pair<int, moe_cache_key>>> by_count(cap + 1);
    for (size_t i = 0; i < evicted.size(); i++) {
        by_count[std::min<int>(counts[i], cap)].push_back(evicted[i]);
    }
    const double period = moe_cache_freq_period(session, device);
    size_t r = 0;
    const size_t n_ranked = use_ranking ? device.ranking.size() : 0;
    for (int c = cap; c >= 0; c--) {
        for (; r < n_ranked && session.config.lfu &&
               std::min<double>(cap, std::lround(2.0 * device.ranking[r].share * period)) > c; r++) {
            device.rewarm_order.push_back({device.ranking[r].pool, device.ranking[r].key});
        }
        device.rewarm_order.insert(device.rewarm_order.end(), by_count[c].begin(), by_count[c].end());
    }
    for (; r < n_ranked; r++) {
        device.rewarm_order.push_back({device.ranking[r].pool, device.ranking[r].key});
    }
    if (!device.rewarm_order.empty() && !device.rewarm_active) {
        device.rewarm_active = true;
        device.rewarm_restored = 0;
        MOE_CACHE_LOG("[moe-cache] CUDA%d rewarm: %zu evicted entries and %zu ranked pairs to restore\n",
                device.physical, evicted.size(), n_ranked);
        if (!moe_cache_rewarm_bulk_on()) {   // else at once by the fill worker (moe_cache_rewarm_bulk_locked), then queued
            moe_cache_rewarm_feed_locked(session, device);
        }
    }
}

// Restore the next entry of the order (moe_cache_rewarm_start_locked) not in the cache into a free entry of its pool
// (never evicting), and only when the fill worker has nothing else queued: an admission waits for at most one
// restoration. A restored entry joins the LRU at its cold end with a count of 1. Called when the restoration starts
// and after every fill.
static void moe_cache_rewarm_feed_locked(moe_cache_session & session, moe_cache_device & device) {
    if (!device.rewarm_active || !device.queue.empty() || session.stopping || device.dead.load()) {
        return;
    }
    while (device.rewarm_next < device.rewarm_order.size()) {
        const auto & [pool_index, key] = device.rewarm_order[device.rewarm_next++];
        if (pool_index >= (int) device.pools.size()) {
            continue;   // the pools were rebuilt with fewer shapes
        }
        moe_cache_pool & pool = *device.pools[pool_index];
        if (pool.free_slots.empty() || pool.map.count(key)) {
            continue;
        }
        if (moe_arena_absent_locked(device, pool, key)) {
            continue;
        }
        int inserts_left = 1;
        bool wake_worker = false;
        if (moe_cache_enqueue_locked(session, device, pool, pool_index, key.tensor, key.expert,
                pool.expert_size, inserts_left, wake_worker, false, true, false, 1)) {
            session.cv.notify_all();
            return;
        }
    }
    // nothing left to restore, or no free entry left for it
    device.rewarm_active = false;
    MOE_CACHE_LOG("[moe-cache] CUDA%d rewarm: %lld entries restored\n", device.physical, device.rewarm_restored);
    device.rewarm_order.clear();
}

// The restoration at once (as Strata gives its loan back before a decode window reads): every entry of the order
// (moe_cache_rewarm_start_locked) not in the cache and registered in place goes into a free entry of its pool, the
// copies queued back to back on the fill stream and waited for once (an event: the stream keeps its SDMA engine) by the
// fill worker, the answer's passes going on meanwhile (an entry being copied is a miss); the answer's own admissions
// queue behind it.
// What the arena holds instead (not registered in place) is left to the queued restoration, which takes over. Called
// by the fill worker (moe_cache_worker) with the session lock held, released during the copies; asked for by the first
// hosted layer of the pass (moe_hosted_plan).
// Off by default (GGML_CUDA_MOE_REWARM_BULK=1 turns it on): no gain measured on the replay bench, and steps a little longer
// at the server after long prompts (not explained); a lead, to restore only the entries the loan evicted.
static bool moe_cache_rewarm_bulk_on() {
    static const bool on = getenv("GGML_CUDA_MOE_REWARM_BULK") && atoi(getenv("GGML_CUDA_MOE_REWARM_BULK")) != 0;
    return on;
}

static void moe_cache_rewarm_bulk_locked(moe_cache_session & session, moe_cache_device & device,
                                         std::unique_lock<std::mutex> & lock) {
    if (!device.rewarm_active || !device.fill_stream || session.stopping || device.dead.load()) {
        return;
    }
    const auto pinned = [&](const void * source, size_t bytes) {
        const uintptr_t src = (uintptr_t) source;
        auto range = std::upper_bound(device.pinned.begin(), device.pinned.end(), std::make_pair(src, UINTPTR_MAX));
        return range != device.pinned.begin() && src + bytes <= (--range)->second;
    };
    const int64_t t0 = ggml_time_us();
    std::vector<moe_cache_job> jobs;
    std::vector<std::pair<int, moe_cache_key>> left;   // for the queued restoration (the arena's, not registered)
    size_t bytes = 0;
    for (; device.rewarm_next < device.rewarm_order.size(); device.rewarm_next++) {
        const auto & [pool_index, key] = device.rewarm_order[device.rewarm_next];
        if (pool_index >= (int) device.pools.size()) {
            continue;
        }
        moe_cache_pool & pool = *device.pools[pool_index];
        if (pool.free_slots.empty() || pool.map.count(key) || !pool.slab) {
            continue;
        }
        const moe_cache_layer * layer = moe_cache_layer_of(device, key.tensor);
        if (!layer || key.expert < 0 || key.expert >= layer->n_expert) {
            continue;
        }
        moe_cache_job job;
        moe_cache_job_sources(*layer, pool, key.expert, job);
        bool direct = job.whole ? pinned(job.source[0], job.bytes) : true;
        for (int r = 0; r < MOE_ROLES && !job.whole && direct; r++) {
            direct = pinned(job.source[r], job.role_bytes[r] + (job.pad_follows[r] ? pool.role_pad[r] : 0));
        }
        if (!direct) {
            left.push_back({pool_index, key});
            continue;
        }
        const int s = pool.free_slots.back();
        pool.free_slots.pop_back();
        moe_cache_slot & slot = pool.slots[s];
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.freq = 1;
        slot.state = moe_cache_slot_state::copying;
        if (!pool.map.emplace(key, s).second) {
            moe_cache_slot_reset(pool, s, true);
            continue;
        }
        moe_cache_direct_set(pool, key, s);
        job.pool = pool_index;
        job.slot = s;
        job.generation = slot.generation;
        job.key = key;
        job.rewarm = true;
        jobs.push_back(job);
        bytes += job.bytes;
    }
    if (jobs.empty()) {
        device.rewarm_order = std::move(left);
        device.rewarm_next = 0;
        return;
    }
    // the copies, the lock released (the entries are copying: nothing reads, evicts or moves them)
    std::vector<char *> dst(jobs.size());
    for (size_t i = 0; i < jobs.size(); i++) {
        dst[i] = moe_cache_slot_ptr(*device.pools[jobs[i].pool], jobs[i].slot);
    }
    lock.unlock();
    ggml_cuda_set_device(device.logical);
    cudaStream_t st = device.fill_stream;
    cudaError_t err = cudaSuccess;
    for (size_t i = 0; i < jobs.size() && err == cudaSuccess; i++) {
        const moe_cache_job & j = jobs[i];
        const moe_cache_pool & p = *device.pools[j.pool];
        if (j.whole) {
            err = cudaMemcpyAsync(dst[i], j.source[0], j.bytes, cudaMemcpyHostToDevice, st);
            continue;
        }
        for (int r = 0; r < MOE_ROLES && err == cudaSuccess; r++) {
            const size_t n = j.role_bytes[r] + (j.pad_follows[r] ? p.role_pad[r] : 0);
            err = cudaMemcpyAsync(dst[i] + p.role_offset[r], j.source[r], n, cudaMemcpyHostToDevice, st);
            if (err == cudaSuccess && p.role_pad[r] && !j.pad_follows[r]) {
                err = cudaMemcpyAsync(dst[i] + p.role_offset[r] + p.role_bytes[r], j.source[r], p.role_pad[r], cudaMemcpyHostToDevice, st);
            }
        }
    }
    cudaEvent_t ev = nullptr;
    if (err == cudaSuccess) {
        err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    }
    if (err == cudaSuccess) {
        err = cudaEventRecord(ev, st);
    }
    if (err == cudaSuccess) {
        err = cudaEventSynchronize(ev);
    } else {
        (void) cudaGetLastError();
        (void) cudaStreamSynchronize(st);   // whatever was queued is done before the entries are released
    }
    if (ev) {
        (void) cudaEventDestroy(ev);
    }
    lock.lock();
    long long restored = 0;
    for (const moe_cache_job & j : jobs) {
        if (j.pool >= (int) device.pools.size()) {
            continue;
        }
        moe_cache_pool & pool = *device.pools[j.pool];
        moe_cache_slot & slot = pool.slots[j.slot];
        if (slot.state != moe_cache_slot_state::copying || slot.generation != j.generation || !(slot.key == j.key)) {
            continue;
        }
        if (err == cudaSuccess) {
            slot.state = moe_cache_slot_state::valid;
            moe_cache_lru_push_front(pool, j.slot);   // hottest first: each one colder than the last
            device.demand_count.erase(j.key);
            device.fills++;
            restored++;
        } else {
            moe_cache_slot_reset(pool, j.slot, true);
            device.fill_failures++;
        }
    }
    device.rewarm_restored += restored;
    MOE_CACHE_LOG("[moe-cache] CUDA%d rewarm: %lld entries (%.0f MiB) restored at once in %.1f ms%s, %zu left to the queue\n",
            device.physical, restored, bytes / 1048576.0, (ggml_time_us() - t0) / 1000.0,
            err == cudaSuccess ? "" : " (copy failed)", left.size());
    device.rewarm_order = std::move(left);
    device.rewarm_next = 0;
    session.idle_cv.notify_all();
}

// Stop restoring: a new prefill is lending (what is left belongs to the previous one),
// or the host weights or the session are going away.
static void moe_cache_rewarm_stop_locked(moe_cache_device & device) {
    device.rewarm_armed = false;
    device.rewarm_bulk_pending = false;
    if (!device.rewarm_active) {
        return;
    }
    device.rewarm_active = false;
    device.rewarm_order.clear();
}

static int moe_cache_lookup_or_queue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker) {
    const moe_cache_key key{host_base, expert};
    auto found = pool.map.find(key);
    device.freq_demands++;
    device.renewed_demands++;
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
// the LFU aging period now: lfu_window x entries demands, the window following the ramp and the context depth
static double moe_cache_freq_period(const moe_cache_session & session, moe_cache_device & device) {
    if (device.freq_period <= 0) {
        size_t entries = 0;
        for (const auto & pool : device.pools) {
            entries += pool->n_slots;
        }
        device.freq_period = (double) session.config.lfu_window * (double) std::max<size_t>(entries, 1);
    }
    double period = device.freq_period;
    if (session.config.lfu_ramp > 0 && session.config.lfu_window > 2) {
        const double entries = device.freq_period / session.config.lfu_window;
        const double w = std::min<double>(session.config.lfu_window,
            2.0 + (session.config.lfu_window - 2.0) * (double) device.renewed_demands / (session.config.lfu_ramp * entries));
        period = w * entries;
    }
    if (session.config.depth_window[2] > session.config.depth_window[0]) {
        const double entries = device.freq_period / session.config.lfu_window;
        const double d0 = session.config.depth_window[0], w1 = session.config.depth_window[1], d1 = session.config.depth_window[2];
        const double w0 = session.config.lfu_window;
        const double d = (double) device.kv_end_rows;
        const double w = d <= d0 ? w0 : d >= d1 ? w1 : w0 + (w1 - w0) * (d - d0) / (d1 - d0);
        period = w * entries;
    }
    return period;
}

static void moe_cache_freq_age_locked(const moe_cache_session & session, moe_cache_device & device) {
    if (!session.config.lfu) {
        return;
    }
    if ((double) device.freq_demands < moe_cache_freq_period(session, device)) {
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
                if (layer.tensor[r] && moe_cache_ranges_overlap(layer.tensor[r], (size_t) (layer.n_expert - 1) * layer.stride[r] + layer.bytes[r], base, size)) {
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
    // EXPERT SPLIT: the host memory of a card the others share stays (only its VRAM goes)
    for (const auto & d : session.devices) {
        selected->host_keep = selected->host_keep || (d.get() != selected && d->host_shared);
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
    // BANDS: the zones at the top of the slab, one after the other (a prefetch slot: zone_bytes rounded to 4 KiB), at
    // most half of it; each banded pool's zone is the run of its slots they overlap
    if (device.band_slab) {
        if (device.band_zones > 0) {
            return;
        }
        const size_t total = moe_cache_band_total(device);
        const size_t slot = (zone_bytes + 4095) & ~(size_t)4095;
        int zones = n_zones;
        while (zones > 0 && (size_t)zones * slot > total / 2) {
            zones--;
        }
        device.band_zones = zones;
        device.band_zone_bytes = slot;
        device.band_zone_begin = total - (size_t)zones * slot;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            if (!pool.band_slots || zones == 0) {
                continue;
            }
            int s0, s1;
            moe_cache_band_slots(pool, device.band_zone_begin, total, s0, s1);
            pool.zone_begin = s0;
            pool.zone_end = s1;
            pool.zone_slice = s1 - s0;   // the whole zone: the device's lend gives the slots' pointers
        }
        device.largest_lend_bytes.store(device.band_zone_begin > moe_cache_band_floor(device) ?
                device.band_zone_begin - moe_cache_band_floor(device) : 0);
        MOE_CACHE_LOG("[moe-cache] CUDA%d zone: %d prefetch slot(s) of %zu MiB at %zu MiB of %zu\n", device.physical, zones,
                slot >> 20, device.band_zone_begin >> 20, total >> 20);
        if (zones < n_zones) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d zone: only %d of %d prefetch slots reserved%s\n",
                    device.physical, zones, n_zones, zones < 2 ? ", the prefetch will not borrow" : "");
        }
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
        // compute buffer chunks lent from the tail of range 0 (the MTP draft keeps its own for good): a zone set there
        // now would cover them, and its entries, back in the cache after the prefill, would hold the chunk's tensors
        if (!pool.lent.empty()) {
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
    {
        size_t best = 0;
        for (const auto & pool_ptr : device.pools) {
            const moe_cache_pool & pool = *pool_ptr;
            if (pool.ranges.empty() || !pool.expert_size) {
                continue;
            }
            const int region0 = pool.ranges[0].count;
            const int zone    = pool.zone_end > pool.zone_begin ? pool.zone_end - pool.zone_begin : 0;
            const int lend    = (int) ((int64_t) region0 * moe_cache_lend_percent / 100) - zone;
            best = std::max(best, (size_t) std::max(lend, 0) * pool.expert_size);
        }
        // a chunk of 1 GiB at most: the loan plan seats every chunk list it can (see moe_cache_plan_borrows), so the
        // limit only trades fragmentation against how finely chunks spread over the pools. Measured on the qwen4exp
        // graph (7.2 GiB in one piece): 512 MiB chunks lose 654 MiB to fragmentation, 1 GiB 292 MiB at the same most-lent
        // pool (71 %), 2 GiB 237 MiB but 77 %. Measured, not derived: the best value moves with the graph. (The bands lend
        // one slab: their largest lend is all of its free room, a graph's buffer in one chunk)
        device.largest_lend_bytes.store(std::min(best, (size_t) 1 << 30));
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
        const char * source = (const char *)layer->tensor[r] + (size_t)slot.key.expert * layer->stride[r];
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
        const moe_cache_slot & victim = pool.slots[index];
        pool.rewarm.push_back({victim.key, victim.freq});   // coldest first
        if (device.lfu && victim.freq > 0) {
            // its history follows it out, as at an admission's eviction: demanded again during the answer, it comes
            // back sooner than a never-seen expert (simulator: +0.1 to +0.25 point of hits, restoration or not)
            try {
                device.demand_count[victim.key] = {victim.freq, pool.expert_size};
            } catch (...) {
            }
        }
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
        dst.freq = src.freq;   // its LFU count moves with it
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
        src.freq = 0;
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
// Expert rankings, hottest first, in Strata's format: "STRP", then version, layers, experts, slots, pairs (uint32);
// version 1 (Strata's data/expert-profile.bin) then the pairs as (layer, expert) uint16, version 2 (ours) then
// (layer uint16, expert uint16, heat uint32): the routed rows counted (GGML_CUDA_MOE_CACHE_PROFILE_SAVE).
struct moe_cache_ranked_expert {
    int layer = 0;
    int expert = 0;
    uint32_t heat = 0;   // 0 in version 1
};

static bool moe_cache_ranking_read(const char * path, std::vector<moe_cache_ranked_expert> & out, uint32_t & n_layers,
                                   uint32_t & n_experts) {
    std::FILE * f = std::fopen(path, "rb");
    char magic[4] = {};
    uint32_t hdr[5] = {};
    if (!f || std::fread(magic, 1, 4, f) != 4 || std::fread(hdr, 4, 5, f) != 5 || memcmp(magic, "STRP", 4) != 0 ||
            (hdr[0] != 1 && hdr[0] != 2)) {
        if (f) std::fclose(f);
        return false;
    }
    n_layers = hdr[1];
    n_experts = hdr[2];
    out.resize(hdr[4]);
    bool ok = true;
    for (auto & r : out) {
        uint16_t le[2];
        uint32_t heat = 0;
        if (std::fread(le, 2, 2, f) != 2 || (hdr[0] == 2 && std::fread(&heat, 4, 1, f) != 1)) {
            ok = false;
            break;
        }
        r = { le[0], le[1], heat };
    }
    std::fclose(f);
    return ok;
}

static const char * moe_cache_profile_save_path() {
    static const char * path = getenv("GGML_CUDA_MOE_CACHE_PROFILE_SAVE") && *getenv("GGML_CUDA_MOE_CACHE_PROFILE_SAVE")
        ? getenv("GGML_CUDA_MOE_CACHE_PROFILE_SAVE") : nullptr;
    return path;
}

// one hosted plan's routed rows, counted for the saved ranking (session lock held)
static void moe_cache_heat_note(moe_cache_device & device, int layer_number, const int32_t * ids, int rows, int64_t n_expert) {
    if (layer_number < 0 || layer_number > 4095 || n_expert <= 0 || n_expert > 65535) {
        return;
    }
    if (device.heat_experts == 0) {
        device.heat_experts = n_expert;
    }
    if (device.heat_experts != n_expert) {
        return;
    }
    const size_t need = (size_t) (layer_number + 1) * n_expert;
    if (device.heat.size() < need) {
        device.heat.resize(need, 0);
    }
    for (int row = 0; row < rows; row++) {
        if (ids[row] >= 0 && ids[row] < n_expert && device.heat[(size_t) layer_number * n_expert + ids[row]] < UINT32_MAX) {
            device.heat[(size_t) layer_number * n_expert + ids[row]]++;
        }
    }
}

// GGML_CUDA_MOE_CACHE_PROFILE_SAVE=FILE: at the end of the session, the routed rows counted, added to the file's (a
// version 2 ranking of the same model), written back as a version 2 ranking (GGML_CUDA_MOE_CACHE_PRELOAD reads it)
static void moe_cache_profile_save_heat(moe_cache_session & session) {
    const char * path = moe_cache_profile_save_path();
    if (!path || session.devices.empty() || session.devices[0]->heat.empty()) {
        return;
    }
    // every device's counts (with several GPUs each one holds its own layers')
    const int64_t ne = session.devices[0]->heat_experts;
    std::vector<uint64_t> total;
    for (const auto & d : session.devices) {
        if (d->heat_experts != ne) {
            continue;
        }
        if (total.size() < d->heat.size()) {
            total.resize(d->heat.size(), 0);
        }
        for (size_t i = 0; i < d->heat.size(); i++) {
            total[i] += d->heat[i];
        }
    }
    std::vector<moe_cache_ranked_expert> old;
    uint32_t nl = 0, nx = 0;
    if (moe_cache_ranking_read(path, old, nl, nx) && nx == (uint32_t) ne) {
        for (const auto & r : old) {
            const size_t i = (size_t) r.layer * ne + r.expert;
            if (r.expert < ne) {
                if (total.size() <= i) total.resize(i + 1, 0);
                total[i] += r.heat;
            }
        }
    }
    std::vector<moe_cache_ranked_expert> ranked;
    for (size_t i = 0; i < total.size(); i++) {
        if (total[i]) {
            ranked.push_back({ (int) (i / ne), (int) (i % ne), (uint32_t) std::min<uint64_t>(total[i], UINT32_MAX) });
        }
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto & a, const auto & b) {
        return a.heat != b.heat ? a.heat > b.heat : a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
    });
    const std::string tmp = std::string(path) + ".tmp";
    std::FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        return;
    }
    const uint32_t hdr[5] = { 2, (uint32_t) ((total.size() + ne - 1) / ne), (uint32_t) ne, (uint32_t) ranked.size(), (uint32_t) ranked.size() };
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(hdr, 4, 5, f) == 5;
    for (const auto & r : ranked) {
        const uint16_t le[2] = { (uint16_t) r.layer, (uint16_t) r.expert };
        ok = ok && std::fwrite(le, 2, 2, f) == 2 && std::fwrite(&r.heat, 4, 1, f) == 1;
    }
    ok = std::fclose(f) == 0 && ok;
    if (ok && std::rename(tmp.c_str(), path) == 0) {
        MOE_CACHE_LOG("[moe-cache] profile: %zu (layer, expert) pairs written to %s\n", ranked.size(), path);
    } else {
        std::remove(tmp.c_str());
    }
}

// GGML_CUDA_MOE_CACHE_PRELOAD=FILE: the pools are filled before the first graph from a ranking of the experts, hottest
// first (moe_cache_ranking_read). Nothing runs yet: each pair takes a free entry of its layer's pool and is copied in
// place, without the admission and the fill thread; the LRU follows the ranking (the coldest are evicted first, by the
// prefill's loans too). The prefetch zone and the KV's rows are left empty (a prefill or a long context takes them
// before they serve); the ranking is kept, with each pair's share of the demands (version 2 heats): after each prefill
// the restoration merges its pairs not in the cache with the evicted entries (moe_cache_rewarm_start_locked). Each
// preloaded entry starts with an LFU count of 1, not where demand at its share would hold it (2 q P, capped): on
// another text those counts held the ranking's experts in the cache too long (simulator, fresh servers on the hybrid's
// depth trace: 80.56 % of hits with 1 against 80.44 % with the heat; 0 and 2 within 0.03 point of 1). Without a
// ranking the cache starts empty and fills during generation: the first prompt read everything over PCIe.
// ARENA: filled once at start (buun's seed reads), before the preload, which copies from it: the preload's ranking first
// (GGML_CUDA_MOE_CACHE_PRELOAD), then every other expert layer by layer, until each shape's arena is full (no eviction).
// Read with O_DIRECT by the reader threads, in batches; the session lock is held (the pools are being built)
static void moe_cache_arena_seed(moe_cache_device & device) {
    if (device.arenas.empty() || !device.arena_readers) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    std::vector<std::pair<int, int>> order;   // (layer number, expert)
    if (const char * path = getenv("GGML_CUDA_MOE_CACHE_PRELOAD"); path && *path) {
        std::vector<moe_cache_ranked_expert> ranking;
        uint32_t n_layers = 0, n_experts = 0;
        if (moe_cache_ranking_read(path, ranking, n_layers, n_experts)) {
            for (const auto & r : ranking) {
                order.push_back({r.layer, r.expert});
            }
        }
    }
    for (const auto & [number, layer] : device.layers) {
        for (int64_t e = 0; e < layer.n_expert; e++) {
            order.push_back({number, (int) e});
        }
    }
    struct load { moe_expert_arena * ar; int slot; moe_cache_job job; const moe_cache_pool * pool; };
    std::vector<load> loads;
    size_t n_read = 0, n_failed = 0, bytes = 0;
    const auto flush = [&] {
        std::vector<char> ok(loads.size(), 0);
        std::vector<std::function<void()>> tasks;
        for (size_t i = 0; i < loads.size(); i++) {
            tasks.push_back([&, i] { ok[i] = moe_arena_load(device, *loads[i].ar, loads[i].slot, loads[i].job, *loads[i].pool); });
        }
        device.arena_readers->run(tasks);
        for (size_t i = 0; i < loads.size(); i++) {
            if (ok[i]) {
                loads[i].ar->commit(loads[i].slot);
                n_read++;
                bytes += loads[i].job.bytes;
            } else {
                loads[i].ar->abort(loads[i].slot);
                loads[i].ar->load_failures++;
                n_failed++;
            }
        }
        loads.clear();
    };
    for (const auto & [number, expert] : order) {
        const auto l = device.layers.find(number);
        if (l == device.layers.end() || l->second.shape < 0 || expert >= l->second.n_expert) {
            continue;
        }
        const moe_cache_layer & cl = l->second;
        moe_expert_arena * ar = moe_arena_of(device, &cl);
        const int pi = device.shapes[cl.shape].pool;
        if (!ar || ar->free_list.empty() || pi < 0 || pi >= (int) device.pools.size()) {
            continue;
        }
        const moe_cache_pool & pool = *device.pools[pi];
        moe_cache_job jb;
        moe_cache_job_sources(cl, pool, expert, jb);
        if (!moe_arena_wanted(device, ar, jb, pool)) {
            continue;
        }
        const int t = ar->take(moe_cache_key{cl.tensor[MOE_ROLE_GATE], expert}, pi);
        if (t >= 0) {
            loads.push_back({ar, t, jb, &pool});
            if (loads.size() >= 256) {
                flush();
            }
        }
    }
    flush();
    const double sec = (ggml_time_us() - t0) / 1e6;
    MOE_CACHE_LOG("[moe-cache] CUDA%d arena: %zu experts (%.2f GiB) read at start in %.2f s (%.2f GB/s), %zu failed\n",
            device.physical, n_read, bytes / 1073741824.0, sec, sec > 0 ? bytes / sec / 1e9 : 0.0, n_failed);
}

// GGML_CUDA_MOE_CACHE_PRELOAD_ZONES=0: the preload leaves the prefetch zones empty (as before)
static bool moe_cache_preload_zones() {
    static const bool on = !getenv("GGML_CUDA_MOE_CACHE_PRELOAD_ZONES") || atoi(getenv("GGML_CUDA_MOE_CACHE_PRELOAD_ZONES")) != 0;
    return on;
}

static void moe_cache_preload(const moe_cache_session & session, moe_cache_device & device) {
    const char * path = getenv("GGML_CUDA_MOE_CACHE_PRELOAD");
    if (!path || !*path) {
        return;
    }
    std::vector<moe_cache_ranked_expert> ranking;
    uint32_t n_layers = 0, n_experts = 0;
    if (!moe_cache_ranking_read(path, ranking, n_layers, n_experts)) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d preload: %s is not an expert ranking (or is truncated)\n", device.physical, path);
        return;
    }
    double heat_total = 0;
    for (const auto & r : ranking) {
        heat_total += r.heat;
    }
    // EXPERT SPLIT: the ranking dealt to the cards in proportion to their budgets, so that every band of it is shared
    // the same way and no expert is preloaded (or restored after a prefill) on two cards; every card deals the same
    // ranking with the same budgets and keeps its own hand. GGML_MOE_SPLIT_DEAL=0: every card takes the whole ranking
    // GGML_MOE_SPLIT_DEAL=2: in bands instead, the hottest to card 0 up to its budget, the next ones to card 1, ...
    static const int deal_mode = getenv("GGML_MOE_SPLIT_DEAL") ? atoi(getenv("GGML_MOE_SPLIT_DEAL")) : 1;
    const bool deal = deal_mode != 0;
    if (deal && moe_cache_expert_split() && session.devices.size() > 1) {
        std::vector<double> budget, dealt;
        int self = -1;
        for (int di = 0; di < (int) session.devices.size(); di++) {
            const moe_cache_device & dk = *session.devices[di];
            // DEAL=2: a card that preloaded already ends its band where its slots ran out
            budget.push_back(dk.dead.load() ? 0.0 : deal_mode == 2 && &dk != &device && dk.preload_bytes ? (double) dk.preload_bytes : (double) dk.budget_limit);
            dealt.push_back(0.0);
            if (&dk == &device) {
                self = di;
            }
        }
        std::vector<moe_cache_ranked_expert> hand;
        int band = 0;   // DEAL=2: the card being filled; dealt[] in bytes, each entry at this card's size for its layer
        for (const auto & r : ranking) {
            if (deal_mode == 2) {
                const auto l = device.layers.find(r.layer);
                const int pi = l == device.layers.end() || l->second.shape < 0 ? -1 : device.shapes[l->second.shape].pool;
                const double size = pi >= 0 && pi < (int) device.pools.size() ? (double) device.pools[pi]->expert_size : 0.0;
                while (band < (int) budget.size() && dealt[band] + size > budget[band]) {
                    band++;
                }
                if (band < (int) budget.size()) {
                    dealt[band] += size;
                    if (band == self) {
                        hand.push_back(r);
                    }
                }
                continue;
            }
            int t = -1;
            double best = 0;
            for (int di = 0; di < (int) budget.size(); di++) {
                const double v = (dealt[di] + 1) / budget[di];
                if (budget[di] > 0 && (t < 0 || v < best)) {
                    t = di;
                    best = v;
                }
            }
            if (t >= 0) {
                dealt[t] += 1;
            }
            if (t == self) {
                hand.push_back(r);
            }
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d preload: %zu of %zu ranked experts dealt to this card (budgets %s)\n", device.physical,
                hand.size(), ranking.size(), [&] { std::string b; for (double x : budget) { b += std::to_string((long long) (x / (1 << 20))) + " MiB "; } return b; }().c_str());
        ranking.swap(hand);
    }
    const uint32_t n_ranked = (uint32_t) ranking.size();
    device.ranking_src.clear();
    for (const auto & r : ranking) {
        device.ranking_src.push_back({r.layer, r.expert, heat_total > 0 ? (float) (r.heat / heat_total) : 0.0f});
    }
    ggml_cuda_set_device(device.logical);
    const int64_t t0 = ggml_time_us();
    size_t loaded = 0, bytes = 0, skipped = 0;
    std::vector<int> installed;   // (pool, slot) in ranking order, pushed to the LRU's cold end at the end
    std::vector<moe_cache_pool *> installed_pool;
    std::vector<std::pair<moe_expert_arena *, int>> arena_held;
    cudaError_t err = cudaSuccess;
    for (uint32_t i = 0; i < n_ranked && err == cudaSuccess; i++) {
        const int layer_number = ranking[i].layer, expert = ranking[i].expert;
        const auto l = device.layers.find(layer_number);
        if (l == device.layers.end() || l->second.shape < 0 || expert >= l->second.n_expert) {
            skipped++;
            continue;
        }
        const moe_cache_layer & layer = l->second;
        const int pool_index = device.shapes[layer.shape].pool;
        if (pool_index < 0 || pool_index >= (int) device.pools.size()) {
            skipped++;
            continue;
        }
        moe_cache_pool & pool = *device.pools[pool_index];
        const moe_cache_key key{layer.tensor[MOE_ROLE_GATE], expert};
        device.ranking.push_back({pool_index, key, device.ranking_src[i].share});
        if (pool.map.count(key)) {
            continue;
        }
        // first the entries no loan takes (the hottest experts there), then the prefetch zone: a short prompt's
        // answers (no prefetch, nothing lent) find a full cache, the first long prompt lends the zone (its experts,
        // the ranking's least demanded, moved or evicted, then restored as any lent entry); never the KV's rows, which
        // a long context takes back
        int s = -1, zone_slot = -1;
        size_t zone_f = 0;
        for (size_t f = pool.free_slots.size(); f-- > 0;) {
            const int c = pool.free_slots[f];
            const bool zone = c >= pool.zone_begin && c < pool.zone_end;
            bool kv_row = false;
            for (const auto & kv : pool.kv) {
                kv_row = kv_row || (c >= kv.first && c < kv.first + kv.count);
            }
            if (!zone && !kv_row) {
                s = c;
                pool.free_slots.erase(pool.free_slots.begin() + f);
                break;
            }
            if (zone && !kv_row && zone_slot < 0) {
                zone_slot = c;
                zone_f = f;
            }
        }
        if (s < 0 && zone_slot >= 0 && moe_cache_preload_zones()) {
            s = zone_slot;
            pool.free_slots.erase(pool.free_slots.begin() + zone_f);
        }
        if (s < 0) {
            continue;
        }
        char * dst = moe_cache_slot_ptr(pool, s);
        // ARENA: from the expert's slot (read at start), held until the copies are done
        moe_expert_arena * ar = moe_arena_of(device, &layer);
        const int a = ar ? ar->acquire(key) : -1;
        if (a >= 0) {
            arena_held.push_back({ar, a});
        }
        for (int r = 0; r < MOE_ROLES && err == cudaSuccess; r++) {
            const char * src = a >= 0 ? ar->slots[a].part[r] : (const char *) layer.tensor[r] + (size_t) expert * layer.stride[r];
            const bool follows = expert + 1 < layer.n_expert && layer.stride[r] == layer.bytes[r];
            err = cudaMemcpyAsync(dst + pool.role_offset[r], src, pool.role_bytes[r] + (follows ? pool.role_pad[r] : 0),
                                  cudaMemcpyHostToDevice, moe_cache_zone_stream);
            if (err == cudaSuccess && pool.role_pad[r] && !follows) {
                err = cudaMemcpyAsync(dst + pool.role_offset[r] + pool.role_bytes[r], src, pool.role_pad[r],
                                      cudaMemcpyHostToDevice, moe_cache_zone_stream);
            }
        }
        moe_cache_slot & slot = pool.slots[s];
        slot.key = key;
        slot.generation++;
        slot.readers = 0;
        slot.freq = 1;
        slot.state = moe_cache_slot_state::valid;
        pool.map.emplace(key, s);
        moe_cache_direct_set(pool, key, s);
        installed.push_back(s);
        installed_pool.push_back(&pool);
        loaded++;
        bytes += pool.expert_size;
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(moe_cache_zone_stream);
    }
    if (err != cudaSuccess) {
        (void) cudaStreamSynchronize(moe_cache_zone_stream);   // no copy may still read a slot given back
    }
    for (const auto & h : arena_held) {
        h.first->release(h.second);
    }
    // hottest first in the list: each pushed to the front ends up hotter than the ones pushed after it
    for (size_t k = 0; k < installed.size(); k++) {
        moe_cache_pool & pool = *installed_pool[k];
        if (err == cudaSuccess) {
            moe_cache_lru_push_front(pool, installed[k]);
        } else {
            pool.slots[installed[k]].state = moe_cache_slot_state::free;   // reset below puts it back
            moe_cache_map_erase(pool, installed[k]);
            pool.slots[installed[k]].key = {};
            pool.free_slots.push_back(installed[k]);
        }
    }
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] CUDA%d preload failed: %s\n", device.physical, cudaGetErrorString(err));
        return;
    }
    device.preload_bytes = bytes;
    device.preload_us = ggml_time_us() - t0;
    MOE_CACHE_LOG("[moe-cache] CUDA%d preload: %zu experts (%.2f GiB) from %s in %.2f s, %zu of %u pairs outside the model\n",
            device.physical, loaded, bytes / 1073741824.0, path, (ggml_time_us() - t0) / 1e6, skipped, n_ranked);
}

// the model's host expert bytes and cards (moe-ram-budget.h): kept until the first card of the process decides, ignored after
static void moe_cache_host_experts(void * opaque, size_t bytes, int n_devices) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> ram_lock(g_ram_plan_mu);
    if (!g_ram_plan.decided) {
        g_ram_plan.total = bytes;
        g_ram_plan.n_devices = n_devices;
    }
}

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

static void moe_hosted_unpin_session(moe_cache_session & session, const moe_cache_device * device);

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
            moe_hosted_unpin_session(*session, &device);
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

// the first tie wins: every scheduler rebuilt by a reservation (llama_context::sched_reserve) computes the layers' devices
// again (the free memory has moved), and a tensor moved to another device would find its layer's block there without
// its registration (MUL_MAT_ID with host weights outside a hosted MoE block)
static void moe_cache_assign(void * opaque, const void * host_base, void * backend) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !host_base) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    const moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (device) {
        session->tensor_devices.emplace(host_base, device->logical);
    }
}

static int moe_cache_assigned(void * opaque, const void * host_base, void * backend) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !host_base) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    const auto tied = session->tensor_devices.find(host_base);
    if (tied == session->tensor_devices.end()) {
        return -1;
    }
    const moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    return device && device->logical == tied->second ? 1 : 0;
}

static int moe_cache_hosts(void * opaque, const void * gate, const void * up, const void * down) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || !gate || !up || !down) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    bool refused = false;
    for (const auto & device_ptr : session->devices) {
        const moe_cache_device & device = *device_ptr;
        if (!device.pools_built || device.dead.load()) {
            continue;
        }
        const int pool = moe_cache_pool_of(device, gate);
        if (pool >= 0 && moe_cache_pool_of(device, up) == pool && moe_cache_pool_of(device, down) == pool) {
            return 1;
        }
        refused = refused || moe_cache_layer_of(device, gate) || moe_cache_layer_of(device, up) ||
                  moe_cache_layer_of(device, down);
    }
    return refused ? 0 : 1;
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

// Prefill: the experts of host tensor w the cache holds, read in place by the node's MMQ (moe-cache.cuh,
// ggml_cuda_moe_prefill_table). addrs[e]: the device address of w's part of expert e's entry, 0 if not resident. The
// entries are pinned until moe_cache_prefill_unpin. Returns the number of resident experts, -1 if the node does not
// read a table (not MMQ). The reads past a tensor's last row land in its pad (moe_cache_role_pad).
static int moe_cache_prefill_residents(void * opaque, void * backend, const ggml_tensor * node, const ggml_tensor * w,
                                       uint64_t * addrs) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    // GGML_CUDA_MOE_PREFILL_INPLACE=0: every expert copied, as before
    static const bool enabled = !getenv("GGML_CUDA_MOE_PREFILL_INPLACE") || atoi(getenv("GGML_CUDA_MOE_PREFILL_INPLACE"));
    if (!enabled || !session || !node || !w || !addrs || node->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_mul_mat_id_uses_mmq(node)) {
        return -1;
    }
    const int64_t n_expert = w->ne[2];
    for (int64_t e = 0; e < n_expert; e++) {
        addrs[e] = 0;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->stopping || session->dormant || device->dead.load() || !device->budget_ready) {
        return 0;
    }
    const moe_cache_layer * layer = moe_cache_layer_of(*device, w->data);
    const int pool_index = moe_cache_pool_of(*device, w->data);
    const int role = moe_cache_role_of_name(w->name);
    if (!layer || pool_index < 0 || pool_index >= (int) device->pools.size() || role < 0 ||
            layer->tensor[role] != w->data || layer->n_expert != n_expert) {
        return 0;
    }
    moe_cache_pool & pool = *device->pools[pool_index];
    if (!pool.slab || pool.role_bytes[role] != w->nb[1]*w->ne[1] || pool.role_pad[role] < moe_cache_role_pad(w->type, w->ne[0])) {
        device->prefill_streamed += n_expert;
        return 0;
    }
    const auto d = pool.direct.find(layer->tensor[MOE_ROLE_GATE]);
    int n = 0;
    if (d != pool.direct.end()) {
        const std::vector<int32_t> & direct = d->second;
        for (int64_t e = 0; e < n_expert && e < (int64_t) direct.size(); e++) {
            const int s = direct[e];
            if (s >= 0 && pool.slots[s].state == moe_cache_slot_state::valid) {
                pool.slots[s].readers++;
                session->prefill_pins.push_back({&pool, s});
                addrs[e] = moe_cache_entry_addr(pool, s) + pool.role_offset[role];
                n++;
            }
        }
    }
    device->prefill_resident += n;
    device->prefill_streamed += n_expert - n;
    return n;
}

static void moe_cache_prefill_unpin_locked(moe_cache_session & session) {
    for (auto & kv : session.prefill_arena_pins) {
        for (const auto & pin : kv.second) {
            pin.first->release(pin.second);
        }
    }
    session.prefill_arena_pins.clear();
    for (const auto & pin : session.prefill_pins) {
        if (pin.second >= 0 && pin.second < pin.first->n_slots && pin.first->slots[pin.second].readers > 0) {
            pin.first->slots[pin.second].readers--;
        }
    }
    session.prefill_pins.clear();
}

// ARENA: the host sources of the experts a prefetch copies (ggml_moe_cache.prefill_sources)
static int moe_cache_prefill_sources(void * opaque, void * backend, const ggml_tensor * w, const uint64_t * resident,
                                     const void ** srcs, size_t * pad, int pslot) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    const int64_t n_expert = w ? w->ne[2] : 0;
    for (int64_t e = 0; e < n_expert; e++) {
        srcs[e] = nullptr;
    }
    *pad = 0;
    moe_cache_device * device = session ? moe_cache_device_for_backend(*session, backend) : nullptr;
    if (!device || device->arenas.empty()) {
        return 0;
    }
    struct load { int slot; int32_t e; moe_cache_job job; };
    std::vector<load> loads;
    const moe_cache_pool * pp = nullptr;
    moe_expert_arena * ar = nullptr;
    const void * key = nullptr;
    int role = -1;
    int n = 0;
    {
        std::lock_guard<std::mutex> lock(session->mu);
        // the previous copies from this prefetch slot are done: their arena slots go back
        auto & pins = session->prefill_arena_pins[pslot];
        for (const auto & pin : pins) {
            pin.first->release(pin.second);
        }
        pins.clear();
        if (session->stopping || session->dormant || device->dead.load() || !device->budget_ready) {
            return 0;
        }
        const moe_cache_layer * cl = moe_cache_layer_of(*device, w->data);
        role = moe_cache_role_of_name(w->name);
        const int pi = cl && cl->shape >= 0 ? device->shapes[cl->shape].pool : -1;
        ar = moe_arena_of(*device, cl);
        if (!cl || !ar || role < 0 || cl->tensor[role] != w->data || cl->n_expert != n_expert || pi < 0 ||
                pi >= (int) device->pools.size()) {
            return 0;
        }
        pp = device->pools[pi].get();
        key = cl->tensor[MOE_ROLE_GATE];
        *pad = pp->role_pad[role];
        for (int64_t e = 0; e < n_expert; e++) {
            if (resident && resident[e]) {
                continue;
            }
            const int a = ar->acquire(moe_cache_key{key, (int32_t) e}, false);
            if (a >= 0) {
                pins.push_back({ar, a});
                srcs[e] = ar->slots[a].part[role];
                n++;
                continue;
            }
            moe_cache_job jb;
            moe_cache_job_sources(*cl, *pp, (int32_t) e, jb);
            if (!moe_arena_wanted(*device, ar, jb, *pp)) {
                continue;
            }
            // read into the zone the prefill borrows (lent at its first read, given back when generation starts):
            // copied from there, never admitted; the zone full, the expert is copied from the mapping
            if (!ar->zone_active) {
                ar->lend();
                device->arena_gen++;   // a restoration still under way gives up
            }
            const int t = ar->transit_take(moe_cache_key{key, (int32_t) e}, pi);
            if (t >= 0) {
                loads.push_back({t, (int32_t) e, jb});
            }
        }
    }
    if (loads.empty()) {
        return n;
    }
    std::vector<char> ok(loads.size(), 0);
    std::vector<std::function<void()>> tasks;
    for (size_t i = 0; i < loads.size(); i++) {
        tasks.push_back([&, i] { ok[i] = moe_arena_load(*device, *ar, loads[i].slot, loads[i].job, *pp); });
    }
    device->arena_readers->run(tasks);
    std::lock_guard<std::mutex> lock(session->mu);
    auto & pins = session->prefill_arena_pins[pslot];
    for (size_t i = 0; i < loads.size(); i++) {
        if (ok[i]) {
            ar->commit(loads[i].slot);
            const int a = ar->acquire(moe_cache_key{key, loads[i].e}, false);
            pins.push_back({ar, a});
            srcs[loads[i].e] = ar->slots[a].part[role];
            ar->prefill_loads++;
            n++;
        } else {
            ar->abort(loads[i].slot);
            ar->load_failures++;
        }
    }
    return n;
}

static void moe_ring_drain(moe_cache_device & device);

static void moe_cache_prefill_unpin(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }
    // the prefill ring's reads of the entries it pinned are done first (GGML_CUDA_MOE_RING_PREFILL)
    for (auto & d : session->devices) {
        moe_ring_drain(*d);
    }
    std::lock_guard<std::mutex> lock(session->mu);
    moe_cache_prefill_unpin_locked(*session);
}

// the table the split being computed reads (the scheduler's thread computes the split)
static thread_local const void *     g_prefill_table_w = nullptr;
static thread_local const uint64_t * g_prefill_table   = nullptr;

static void moe_cache_prefill_table(const void * w_data, const uint64_t * table) {
    g_prefill_table_w = table ? w_data : nullptr;
    g_prefill_table   = table;
}

void ggml_cuda_moe_prefill_table_set(const void * w_data, const uint64_t * table) {
    moe_cache_prefill_table(w_data, table);
}

const uint64_t * ggml_cuda_moe_prefill_table(const void * w_data) {
    return w_data && w_data == g_prefill_table_w ? g_prefill_table : nullptr;
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
    // BANDS: the zones are a byte range at the top of the slab, every banded pool lending the slots it overlaps
    const bool banded = device->band_slab && device->band_zones > 0;
    bool band_ready = false;
    if (banded && device->band_zone_bytes >= bytes) {
        band_ready = true;
        for (auto & pool_ptr : device->pools) {
            if (pool_ptr->band_slots && pool_ptr->zone_lent) {
                band_ready = false;
            }
        }
        n_slots = band_ready ? std::min(n_zones, device->band_zones) : 0;
    }
    for (auto & pool_ptr : device->pools) {
        if (banded) {
            break;
        }
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
    for (auto & pool_ptr : device->pools) {
        if (std::find(chosen.begin(), chosen.end(), pool_ptr.get()) == chosen.end() && !(banded && pool_ptr->band_slots)) {
            continue;
        }
        for (const moe_cache_slot & slot : pool_ptr->slots) {
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
    std::vector<std::pair<moe_cache_pool *, std::vector<int>>> band_moved;
    if (banded) {
        moe_cache_band_take(*device, device->band_zone_begin, moe_cache_band_total(*device), band_moved);
        for (auto & m : band_moved) {
            m.first->zone_lent = true;
        }
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
        for (int first = pool.zone_begin; !banded && first + pool.zone_slice <= pool.zone_end && given < n_zones; first += pool.zone_slice) {
            zones[given++] = moe_cache_slot_ptr(pool, first);   // range 0: contiguous
        }
    }
    for (int z = 0; banded && z < n_slots; z++) {
        zones[given++] = device->band_slab + device->band_zone_begin + (size_t)z * device->band_zone_bytes;
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

static size_t moe_cache_largest_lend(void * opaque, void * backend) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return 0;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    return device && !device->dead.load() ? device->largest_lend_bytes.load() : 0;
}

static size_t moe_cache_lend_room(void * opaque, void * backend) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    if (!session) {
        return 0;
    }
    moe_cache_device * device = moe_cache_device_for_backend(*session, backend);
    if (!device || device->dead.load()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (!device->budget_ready) {
        return 0;
    }
    if (device->band_slab) {
        size_t free_bytes = 0, largest = 0;
        moe_cache_band_room(*device, free_bytes, largest);
        return free_bytes;
    }
    size_t room = 0;
    for (const auto & pool_ptr : device->pools) {
        room += (size_t) moe_cache_pool_lendable(*pool_ptr) * pool_ptr->expert_size;
    }
    return room;
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

    // BANDS: no seating to plan, only the room to count
    if (device->band_slab) {
        size_t left = 0, largest = 0, need = 0;
        moe_cache_band_room(*device, left, largest);
        std::string desc;
        for (int i = 0; i < n_sizes; i++) {
            if (sizes[i]) {
                need += (sizes[i] + 4095) & ~(size_t)4095;
                desc += " " + std::to_string((sizes[i] + (1 << 19)) >> 20);
            }
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d loan plan: chunks MiB%s, %zu MiB of %zu free (largest gap %zu)%s\n", device->physical,
                desc.c_str(), need >> 20, left >> 20, largest >> 20, need <= left ? "" : ": NO ROOM");
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
    // what the plan was given: the chunks, and per pool its entry size, lendable entries, lowest lent entry, range 0
    {
        std::string desc = "chunks MiB";
        for (size_t i = 0; i < n; i++) {
            desc += " " + std::to_string((blocks[i] + (1 << 19)) >> 20);
        }
        desc += "; pools (entry KiB, room, low, region0)";
        for (size_t k = 0; k < np; k++) {
            desc += " (" + std::to_string(lenders[k]->expert_size >> 10) + "," + std::to_string(search.room[k]) + "," +
                    std::to_string(search.low[k]) + "," + std::to_string(search.region0[k]) + ")";
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d loan plan input: %s\n", device->physical, desc.c_str());
    }
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

    // BANDS: the highest free gap under the zones that holds the chunk, above the floor of 5 %; every banded pool gives
    // the slots the chunk overlaps (a slot astride a neighbouring loan is already out)
    if (device->band_slab) {
        const size_t size = (bytes + 4095) & ~(size_t)4095;
        const size_t a = moe_cache_band_find(*device, size);
        if (a == SIZE_MAX) {
            size_t left = 0, largest = 0;
            moe_cache_band_room(*device, left, largest);
            MOE_CACHE_LOG("[moe-cache] CUDA%d borrow: no room for %zu MiB (%zu MiB free, largest gap %zu)\n", device->physical,
                    bytes >> 20, left >> 20, largest >> 20);
            return nullptr;
        }
        session->idle_cv.wait(lock, [&] {
            return device->queue.empty() && !device->inflight;
        });
        if (device->dead.load()) {
            return nullptr;
        }
        for (auto & pool_ptr : device->pools) {
            for (const moe_cache_slot & slot : pool_ptr->slots) {
                if (pool_ptr->band_slots && slot.readers > 0) {
                    MOE_CACHE_LOG("[moe-cache] CUDA%d borrow refused: expert still being read\n", device->physical);
                    return nullptr;
                }
            }
        }
        ggml_cuda_set_device(device->logical);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::pair<moe_cache_pool *, std::vector<int>>> moved;
        moe_cache_band_take(*device, a, a + size, moved);
        const cudaError_t error = cudaStreamSynchronize(moe_cache_zone_stream);
        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "borrow evacuation", true);
        }
        device->band_loans.insert(std::upper_bound(device->band_loans.begin(), device->band_loans.end(), std::make_pair(a, a + size)),
                std::make_pair(a, a + size));
        device->zone_lends++;
        device->zone_lend_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        MOE_CACHE_LOG("[moe-cache] CUDA%d borrow: %zu MiB at %zu MiB of the slab, %.2f ms\n", device->physical, bytes >> 20, a >> 20,
                device->zone_lend_ms);
        const long long period = moe_cache_verify_period();
        for (auto & m : moved) {
            for (size_t k = 0; period && k < m.second.size(); k += (size_t)period) {
                moe_cache_zone_check(*device, *m.first, m.second[k]);
            }
        }
        return device->band_slab + a;
    }

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
    // BANDS: a byte range of the slab, back to every banded pool
    if (device->band_slab && (char *)ptr >= device->band_slab && (char *)ptr < device->band_slab + moe_cache_band_total(*device)) {
        const size_t a = (size_t)((char *)ptr - device->band_slab);
        for (size_t i = 0; i < device->band_loans.size(); i++) {
            if (device->band_loans[i].first != a) {
                continue;
            }
            const size_t b = device->band_loans[i].second;
            device->band_loans.erase(device->band_loans.begin() + i);
            moe_cache_band_release(*device, a, b);
            device->zone_restores++;
            MOE_CACHE_LOG("[moe-cache] CUDA%d give back: %zu MiB at %zu MiB\n", device->physical, (b - a) >> 20, a >> 20);
            return;
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d give back: unknown range of the slab\n", device->physical);
        return;
    }
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
    // BANDS: the zones' byte range back, a slot astride a loan staying out until that loan returns
    bool band_lent = false;
    for (auto & pool_ptr : device->pools) {
        if (pool_ptr->band_slots && pool_ptr->zone_lent) {
            band_lent = true;
            pool_ptr->zone_lent = false;
        }
    }
    if (band_lent) {
        moe_cache_band_release(*device, device->band_zone_begin, moe_cache_band_total(*device));
        restored = true;
    }
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

// the buffers' routed rows: a block of up to moe_hosted_max_rows() (GGML_MOE_HOSTED_MAX_ROWS, default 8191) is hosted,
// prompts of up to ~800 tokens included. A hosted block copies no weight: on the 7900 XT in PCIe 3.0 a prompt of 164 /
// 692 / 853 tokens takes 0.62 / 1.89 / 2.24 s against ~3.05 s for the prefetch, which copies every expert whatever the
// batch (1054 tokens: 3.17 against 3.07). Generation and the MTP verification use 10 to 30 rows. And at most
// moe_hosted_tokens_max tokens (a context sharing the session, the MTP draft, may use fewer experts per token): the
// activations' buffers are sized for them
static constexpr int moe_hosted_rows_max = 8192;
static constexpr int moe_hosted_tokens_max = 1024;
static int moe_hosted_max_rows() {
    static const int n = [] {
        const char * v = getenv("GGML_MOE_HOSTED_MAX_ROWS");
        const int r = v ? atoi(v) : moe_hosted_rows_max - 1;
        return r < 1 ? 1 : (r > moe_hosted_rows_max ? moe_hosted_rows_max : r);
    }();
    return n;
}

struct moe_hosted_mailbox {
    uint32_t seq;
    uint32_t slot;
    uint32_t n_tokens;
    uint32_t pad[13];
};

struct moe_hosted_layer {
    const ggml_tensor * gate_w = nullptr;
    int layer_number = -2;           // its layer (moe_cache_device::layer_of), -2 until looked up
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
    // SPLIT, kernel copy: [0] count, then per moved expert its packed entry in RAM (device address), its cache entry, its
    // bytes; copied to the device by k_moe_split_wait, read by k_moe_split_fetch
    uint64_t * h_fetch = nullptr;
    uint64_t * m_fetch = nullptr;
    uint64_t * d_fetch = nullptr;
    float * h_chk = nullptr;        // DIAGNOSTIC (GGML_CUDA_MOE_SPLIT_VERIFY=1): the GPU results of the moved rows (rows x n_embd)
    float * m_chk = nullptr;
    int32_t * h_pred = nullptr;     // PREGATE: [0] stamp (publication it belongs to), [1] count, then expert ids
    int last_tokens = 1;            // PREGATE: tokens of the last publication (read by the serving thread)
    int32_t * m_pred = nullptr;
    uint64_t * m_plan = nullptr;
    float * m_res = nullptr;
    float * h_w = nullptr;          // EXPERT SPLIT, PSUM: the rows' routing weights, read by the other cards
    float * m_w = nullptr;
    float * m_act = nullptr;
    block_q8_0 * m_act8 = nullptr;
    int32_t * m_ids = nullptr;
    uint32_t * m_flags = nullptr;
    // device memory
    uint64_t * d_plan = nullptr;
    int32_t * d_act_ids = nullptr;  // row -> token
    // EXPERT SPLIT: per other card k (1..MOE_SPLIT_MAX_CARDS), its rows' up, gate, down entry addresses in its cache
    // (3 x rows each, 0 elsewhere), and each row's producer for the merge (0: here or the CPU, k: card k); its done flag
    // is h_flags[32 + 8 k]
    uint64_t * h_plank = nullptr;
    uint64_t * m_plank = nullptr;
    int8_t * h_src = nullptr;
    int8_t * m_src = nullptr;
    // EXPERT SPLIT, the dispatcher: per other card, the rows of the experts moved to it (3 x rows each); its moved experts'
    // flag is h_flags[34 + 8 k], raised by its fill thread when they arrived (split_pending_k copies left)
    uint64_t * h_plan2k = nullptr;
    uint64_t * m_plan2k = nullptr;
    int split_pending_k[MOE_SPLIT_MAX_CARDS + 1] = {};
    uint32_t * d_expect = nullptr;  // sequence number of the layer's last publication
};

// SPLIT: the dispatcher's CPU time model (moe_split_cpu_*) was fitted on one machine; on another the CPU rows take
// longer or shorter. Its scale is measured once, on the layers with CPU rows left after a split: the first
// moe_split_cal_skip warm the caches, the next moe_split_cal_layers are timed, then the scale is fixed (no timing after)
static constexpr int moe_split_cal_skip = 500, moe_split_cal_layers = 2000;

// EXPERT SPLIT: another card computing the rows of the experts its cache holds, its chain queued with the hosting card's
// blocks (a wait for the plan, its rows, its flag): its stream and its buffers, shared by the layers (they run in order)
struct moe_split_card {
    int index = 0;                    // 1..MOE_SPLIT_MAX_CARDS: its plan and its flag in the layers' buffers
    moe_cache_device * device = nullptr;
    cudaStream_t stream = nullptr;
    uint32_t * d_last = nullptr;      // per layer slot: the last publication it served
    uint32_t * d_seq = nullptr;       // the publication being served (the wait kernel -> the done kernel)
    uint64_t * d_plan = nullptr;      // 3 x rows entry addresses
    uint64_t * d_plan2 = nullptr;     // the rows of the experts the dispatcher moved to this card
    double rate = 13600.0;            // its link's copy rate, bytes per microsecond (moe_cache_pcie_bytes_per_us)
    int32_t * d_act_ids = nullptr;    // row -> token
    char * d_q8_act = nullptr;        // the activations as q8_1, quantized here as the hosting card does
    float * d_act = nullptr;          // the activations (F32, copied from the hosting card's mapped buffer)
    float * d_glu = nullptr;
    char * d_q8_mid = nullptr;
    unsigned * d_q8_count = nullptr;
    float * d_res = nullptr;          // PSUM: its rows' down outputs (MOE_SPLIT_PSUM_TOKENS x n_used x n_embd), summed here
    float * d_w = nullptr;            // PSUM: the routing weights, copied here with the activations (read once over the link)
    int64_t n_embd = 0, n_ff = 0;
};

struct moe_hosted_state {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    std::vector<moe_split_card> split_cards;   // EXPERT SPLIT: the other cards (created by the first block)
    // EXPERT SPLIT: a wait that ran out of time (mapped): [0] 1 a card's plan wait, 2 the merge's card wait; [1] the card,
    // [2] the layer slot or row, [3] the value it held, [4] the value seen; logged once by the serving thread
    uint32_t * h_split_diag = nullptr;
    uint32_t * m_split_diag = nullptr;
    bool split_diag_logged = false;
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
    // the layers' activations, ids, plans and CPU results, shared like d_q8_act: the layers run one after the other on
    // the stream, a layer's enter (which rewrites them) queued after the previous layer's merge (which waits for its
    // CPU rows and reads its results). Sized once by the first layer (addresses captured in graphs)
    float * h_act = nullptr;
    block_q8_0 * h_act8 = nullptr;
    float * h_res = nullptr;
    float * h_w = nullptr;          // EXPERT SPLIT, PSUM: the rows' routing weights (rows)
    int32_t * h_ids = nullptr;
    uint64_t * h_plan = nullptr;
    uint64_t * h_plan2 = nullptr;
    uint64_t * d_plan = nullptr;
    uint64_t * d_plan2 = nullptr;
    int64_t sh_embd = 0;
    unsigned * d_q8_count = nullptr;   // the up / gate kernel's warps done per q8_1 block of d_q8_mid (zero between launches)
    unsigned * d_router_done = nullptr;   // the router's rows done (k_moe_hosted_enter), zero between layers
    uint64_t * d_router_tagged = nullptr; // the router's rows, {layer tag, logit} (k_moe_hosted_enter)
    size_t q8_act_bytes = 0;
    size_t q8_mid_bytes = 0;
    int n_threads = 1;
    double pcie_bytes_per_us = 13600.0;                      // SPLIT: host -> device copy rate (moe_cache_pcie_bytes_per_us)
    std::vector<moe_cache_pin> pins;                         // held by the last plan
    std::vector<moe_cache_job> gpu_fills;                    // SPLIT, kernel copy: entries the last plan's layer fills
    int64_t split_moved = 0;                                 // SPLIT: experts moved to the GPU
    int split_tensors = 0;                                   // SPLIT: tensors copied for the last layer's moved experts
    // SPLIT: the CPU term of the dispatcher's model scaled to this machine (moe_split_calibrate), measured once
    std::atomic<double> cpu_scale{1.0};
    // SPLIT: the GPU terms of the dispatcher's model scaled to this card, measured once at the first layer
    // (moe_split_gpu_bench; GGML_CUDA_MOE_SPLIT_GPU_SCALE=x: fixed, not measured)
    double gpu_scale = 1.0;
    bool gpu_benched = false;
    int cal_left = moe_split_cal_skip + moe_split_cal_layers;   // GGML_CUDA_MOE_SPLIT_CPU_SCALE=x: fixed, not measured
    double cal_measured = 0.0, cal_model = 0.0;
    // served by the thread that waits for the stream (ggml_cuda_moe_hosted_wait)
    std::mutex serve_mu;
    // and by a thread of its own while blocks launched outside a graph capture are pending (moe_hosted_server).
    // The launching thread can block in a CUDA call that waits for the device (a full launch queue, a lazily loaded
    // module, a first cuBLAS GEMM) while a block waits for its mailbox to be served: three hangs on an RTX 4090, and on
    // HIP a first rocBLAS GEMM loading its Tensile code object (R9700, gfx1201, ROCm 7.1: every first prompt hung)
    std::thread server;
    std::mutex server_mu;
    std::condition_variable server_cv;
    uint32_t server_pending = 0;   // blocks launched outside a capture whose publication is not served yet
    std::atomic<bool> server_stop{false};
    uint32_t last_seq = 0;
    std::vector<int32_t> cpu_rows, experts;
    std::vector<float> out, act;
    std::vector<const void *> arena_src;   // ARENA: per CPU row, its expert's gate, up and down in the arena (or null)
    std::vector<int> arena_held;           // the slots read, released after the CPU rows
    std::vector<uint8_t> act8;   // the CPU rows' activations as q8_0 blocks (from h_act8)
    int64_t st_n = 0, st_gap = 0, st_gaps = 0, st_plan = 0, st_cpu = 0, st_rows = 0, t_done = 0;
    // DIAGNOSTIC (GGML_CUDA_MOE_TIMELINE): host times per layer (seq % MOE_TL_RING): publication seen, plan flag
    // written, CPU rows' flag written (ns); sums of the derived intervals; the host's stream queries
    std::vector<int64_t> tl_host;
    double tl_sum[20] = {0};
    double tlk_sum[8][10] = {};   // EXPERT SPLIT: per other card, its chain (moe_tl_account)
    int64_t tl_n = 0, tl_nw = 0, tl_q_n = 0, tl_q_ns = 0;
    double tl_tick_ns = 0;
    // CPU rows by what they have resident: up+gate (not down), down only, up or gate only, nothing
};

static std::mutex g_hosted_mu;
// one state per (session, device logical index): with the layers split over several GPUs, each device hosts its own layers
static std::map<std::pair<moe_cache_session *, int>, std::unique_ptr<moe_hosted_state>> g_hosted;

// a hosted block queued but whose CPU rows are not copied in yet (graph evaluation thread)
#define MOE_HOSTED_ENTER_BLOCKS 8

// the shared expert's gate (a row of F32 weights times the block's activations), computed by one more workgroup of the
// merge with the arithmetic of mul_mat_vec_f<float, float, ncols, 256> (the node is then skipped)
#define MOE_SGATE_MAX_COLS 8
struct moe_sgate_args {
    const float * x = nullptr;
    const float * y = nullptr;
    float * dst = nullptr;
    int ncols2 = 0;
    int stride_col_y2 = 0;
    int stride_col_dst = 0;
    int ncols = 0;
};

struct moe_hosted_pending {
    moe_hosted_layer * layer = nullptr;
    const ggml_tensor * down = nullptr;
    int rows = 0;
    bool split = false;   // SPLIT: the merge skips the rows of the moved expert
    moe_sgate_args sg;    // the shared expert's gate, or none
    bool cards = false;   // EXPERT SPLIT: rows of other cards (each waits for its producer's flag)
    uint32_t * diag = nullptr;
    int n_used = 0;
    bool psum = false;    // EXPERT SPLIT, PSUM: each card's rows of a token come as one weighted sum
};
static thread_local moe_hosted_pending g_hosted_pending;

// Portable: every GPU reads and writes these buffers (several cards: the mailbox, the plan, the activations, the rows);
// without it a mapping can be made for the current device only (CUDA)
static bool moe_hosted_host_alloc(void ** ptr, size_t bytes) {
#if defined(GGML_USE_HIP)
    if (hipHostMalloc(ptr, bytes, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable) != hipSuccess) {
#else
    if (cudaHostAlloc(ptr, bytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
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

// DIAGNOSTIC (GGML_CUDA_MOE_TIMELINE=N; HIP: wall clock, CUDA: global timer): a timeline of the hosted layers without a profiler, every N layers. The
// GPU writes its wall clock to mapped host memory, per layer (seq % MOE_TL_RING): [0] publication, [1] plan seen,
// [2] the CPU rows' flag seen by the merge (0: no CPU row). The host keeps its own times: publication seen, plan
// flag written, CPU rows' flag written. No clock synchronization: the relays (GPU -> host -> GPU) come out of
// differences of durations measured on one side each.
#define MOE_TL_RING 1024
static unsigned long long * g_tl_h = nullptr;   // host view
static unsigned long long * g_tl_m = nullptr;   // device view of the same memory
// EXPERT SPLIT: each other card's chain, per publication (seq % MOE_TL_RING) x 8, its own clock: [0] the chain reached the
// plan wait, [1] plan seen, [2] plan and activations copied, [3] the moved experts' wait reached, [4] passed, [5] done,
// [6] the publication
static unsigned long long * g_tlk_h[8] = {};
static unsigned long long * g_tlk_m[8] = {};
static int64_t moe_tl_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static __device__ __forceinline__ unsigned long long moe_tl_clock() {
#if defined(GGML_USE_HIP)
    return wall_clock64();
#elif !defined(GGML_USE_MUSA)
    unsigned long long t;   // CUDA: the global timer, in nanoseconds
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
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

// Rows a graph reads from the host (qwen4exp's PLE rows, read after the launch the way Strata does): the host writes them
// into a mapped buffer while the GPU computes the layers before their first reader, a node named HOST_ROWS_NODE (any op, its
// sources not read) whose kernel waits for the host's count of written batches and copies the rows into its output. The
// writer is a task the graph's user registers before the compute (ggml_cuda_host_rows_task); the first wait for the GPU after
// the launch runs it (moe_host_rows_run), before serving any mailbox.
#define HOST_ROWS_NODE "ple_host_wait"
#define HOST_ROWS_CAP  (1u << 20)
static void * g_rows_h = nullptr;                // mapped host buffer, HOST_ROWS_CAP bytes
static void * g_rows_m = nullptr;                // its device address
static volatile uint32_t * g_rows_flag_h = nullptr;   // batches written by the host (mapped)
static uint32_t * g_rows_flag_m = nullptr;
static uint32_t * g_rows_ticket = nullptr;       // batches read by the GPU: in the memory of the GPU that runs the node
static int g_rows_ticket_device = -1;            // that GPU (allocated at the node's first launch)
static uint32_t g_rows_count = 0;
static std::atomic<void (*)(void *)> g_rows_fn{nullptr};
static void * g_rows_user = nullptr;

void * ggml_cuda_host_rows_buffer(size_t bytes) {
    // on by default where it was measured to pay (RX 7900 XT: replay of 3 tokens -0.5 %); an RTX 4090 lost to it in both
    // rounds (17.43 / 17.23 against 17.45 / 17.64 ms per step): there the rows are read before the launch.
    // GGML_CUDA_HOST_ROWS=1 / 0 forces it
#if defined(GGML_USE_HIP)
    constexpr bool on_default = true;
#else
    constexpr bool on_default = false;
#endif
    static const bool on = getenv("GGML_CUDA_HOST_ROWS") ? atoi(getenv("GGML_CUDA_HOST_ROWS")) != 0 : on_default;
    if (!on || bytes > HOST_ROWS_CAP) {
        return nullptr;
    }
    if (!g_rows_h) {
        void * flag = nullptr;
        if (!moe_hosted_host_alloc(&g_rows_h, HOST_ROWS_CAP) || !moe_hosted_host_alloc(&flag, 64)) {
            (void) cudaGetLastError();
            g_rows_h = nullptr;
            return nullptr;
        }
        g_rows_flag_h = (volatile uint32_t *) flag;
        g_rows_m      = moe_hosted_device_ptr(g_rows_h);
        g_rows_flag_m = moe_hosted_device_ptr((uint32_t *) flag);
        if (!g_rows_m || !g_rows_flag_m) {
            g_rows_h = nullptr;
            return nullptr;
        }
    }
    return g_rows_h;
}

void ggml_cuda_host_rows_task(void (*fn)(void *), void * user) {
    g_rows_user = user;
    g_rows_fn.store(fn, std::memory_order_release);
}

// the pending task, then the count the GPU waits for
static void moe_host_rows_run() {
    void (*fn)(void *) = g_rows_fn.exchange(nullptr, std::memory_order_acq_rel);
    if (!fn) {
        return;
    }
    fn(g_rows_user);
    std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(__x86_64__)
    __builtin_ia32_sfence();
#endif
    *g_rows_flag_h = ++g_rows_count;
}

// one block: the count waited for, the rows copied (volatile: the mapped memory changes between replays), the ticket moved
static __global__ void k_host_rows_wait(const volatile uint32_t * flag, uint32_t * ticket, const volatile uint64_t * src,
                                        uint64_t * dst, int n) {
    __shared__ uint32_t want;
    if (threadIdx.x == 0) {
        want = ticket[0] + 1;
        while ((int32_t) (*flag - want) < 0) {
            moe_hosted_pause();
        }
        __threadfence_system();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ticket[0] = want;
    }
}

bool ggml_cuda_host_rows_node(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    if (strcmp(node->name, HOST_ROWS_NODE) != 0) {
        return false;
    }
    const size_t bytes = ggml_nbytes(node);
    GGML_ASSERT(g_rows_h && bytes <= HOST_ROWS_CAP && bytes % sizeof(uint64_t) == 0 && ggml_is_contiguous(node));
    // launched now rather than recorded into a graph (a first run, the warm-up): the host writes the rows first, the wait
    // for the GPU that would run the task may come after a synchronization inside this graph's evaluation (CUDA: its
    // blocks launched outside a capture are served by the server thread)
#if defined(GGML_USE_HIP)
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    const bool now = hipStreamIsCapturing(ctx.stream(), &capture) == hipSuccess && capture == hipStreamCaptureStatusNone;
#else
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    const bool now = cudaStreamIsCapturing(ctx.stream(), &capture) == cudaSuccess && capture == cudaStreamCaptureStatusNone;
#endif
    // the ticket is read and written by this kernel only: in this GPU's memory, allocated by the first launch (the buffers
    // above are mapped host memory, the host writes them). Not in ggml_cuda_host_rows_buffer: the graph's builder calls it
    // without a device, and a cudaMalloc there took the thread's current device (with two GPUs the second one, which the
    // first then faulted on without P2P, Avalon#26)
    if (!g_rows_ticket) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaMalloc((void **) &g_rows_ticket, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(g_rows_ticket, 0, sizeof(uint32_t)));
        g_rows_ticket_device = ctx.device;
        MOE_CACHE_LOG("[moe-cache] CUDA%d host rows: ticket allocated%s\n", ctx.device, now ? "" : " during a capture");
    }
    GGML_ASSERT(g_rows_ticket_device == ctx.device && "the host rows node moved to another GPU");
    if (now) {
        moe_host_rows_run();
    }
    (void) cudaGetLastError();
    k_host_rows_wait<<<1, 1024, 0, ctx.stream()>>>(g_rows_flag_m, g_rows_ticket, (const volatile uint64_t *) g_rows_m,
                                                   (uint64_t *) node->data, (int) (bytes / sizeof(uint64_t)));
    CUDA_CHECK(cudaGetLastError());
    return true;
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
            v += __shfl_xor_sync(0xffffffff, v, off, warpSize);
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
            const float ov = __shfl_xor_sync(0xffffffff, bv, off, warpSize);
            const int   oi = __shfl_xor_sync(0xffffffff, bi, off, warpSize);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[warp] = bv; wi[warp] = bi; }
        __syncthreads();
        if (warp == 0) {
            bv = lane < nwarps ? wv[lane] : -INFINITY; bi = lane < nwarps ? wi[lane] : -1;
            for (int off = warpSize / 2; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffff, bv, off, warpSize);
                const int   oi = __shfl_xor_sync(0xffffffff, bi, off, warpSize);
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

// the MoE router computed by the hosted block's entry kernel (ggml_cuda_router_defer_ok): BF16 weights, ncols token
// columns, a workgroup of 256 threads per row with mul_mat_vec_f's arithmetic (4 values per load), the top-k waiting
// on done for all nrows rows
#define MOE_ROUTER_MAX_COLS 8
struct moe_router_args {
    const nv_bfloat16 * x = nullptr;
    const float * y = nullptr;
    float * dst = nullptr;
    int ncols2 = 0;            // row width / 2
    int stride_row = 0;        // elements between two rows of x
    int stride_col_y2 = 0;     // float2 between two columns of y
    int stride_col_dst = 0;
    int ncols = 0;
    int nrows = 0;
    unsigned * done = nullptr;
    uint64_t * tagged = nullptr;   // the rows' logits as {layer tag, value} words (no fence, no counter), else done
};

// a 64-bit word written or read in one access by another block (device scope, relaxed)
static __device__ __forceinline__ void moe_store_relaxed_device(uint64_t * p, uint64_t v) {
#if defined(GGML_USE_HIP)
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    *(volatile uint64_t *) p = v;
#endif // defined(GGML_USE_HIP)
}
static __device__ __forceinline__ uint64_t moe_load_relaxed_device(const uint64_t * p) {
#if defined(GGML_USE_HIP)
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *(const volatile uint64_t *) p;
#endif // defined(GGML_USE_HIP)
}

static __device__ void moe_router_row(const moe_router_args & a, const int row, const uint32_t tag) {
    constexpr int block_size = 256;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;
    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    float sumf[MOE_ROUTER_MAX_COLS] = {0.0f};
    const int2   * x4 = (const int2 *) (a.x + (int64_t) row * a.stride_row);
    const float4 * y4 = (const float4 *) a.y;
    for (int col4 = tid; col4 < a.ncols2/2; col4 += block_size) {
        const int2 tmpx = x4[col4];
#pragma unroll
        for (int j = 0; j < MOE_ROUTER_MAX_COLS; ++j) {
            if (j < a.ncols) {
                const float4 tmpy = y4[j*(a.stride_col_y2/2) + col4];
                ggml_cuda_mad(sumf[j], ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx.x)[0]), tmpy.x);
                ggml_cuda_mad(sumf[j], ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx.x)[1]), tmpy.y);
                ggml_cuda_mad(sumf[j], ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx.y)[0]), tmpy.z);
                ggml_cuda_mad(sumf[j], ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx.y)[1]), tmpy.w);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MOE_ROUTER_MAX_COLS; ++j) {
        if (j < a.ncols) {
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }
            __syncthreads();
        }
    }
    if (tid < a.ncols) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < MOE_ROUTER_MAX_COLS; ++j) {
            v = j == tid ? sumf[j] : v;
        }
        if (a.tagged) {
            moe_store_relaxed_device(a.tagged + (int64_t) tid*a.nrows + row, (uint64_t) tag << 32 | __float_as_uint(v));
        } else {
            a.dst[tid*a.stride_col_dst + row] = v;
        }
    }
}

// The chunk list of a plan (MOE_GROUP_* in mmvq.cuh) from its up entries, already on the device and visible to the block
// (written by it before a __syncthreads): one thread per routed row (rows <= MOE_GROUP_MAX_ROWS <= blockDim.x). A row
// heads a chunk when the rows of its entry before it are a multiple of MOE_GROUP_COLS; chunks numbered in row order.
static __device__ void moe_group_list(const uint64_t * up, int rows, int32_t * grp) {
    __shared__ uint64_t s_entry[MOE_GROUP_MAX_ROWS];
    __shared__ bool     s_head[MOE_GROUP_MAX_ROWS];
    const int t = threadIdx.x;
    uint64_t e = 0;
    bool head = false;
    if (t < rows) {
        e = up[t];
        s_entry[t] = e;
    }
    __syncthreads();
    if (t < rows) {
        int rank = 0;
        for (int j = 0; j < t; j++) {
            rank += s_entry[j] == e;
        }
        head = e != 0 && rank % MOE_GROUP_COLS == 0;
        s_head[t] = head;
    }
    __syncthreads();
    if (head) {
        int c = 0;
        for (int j = 0; j < t; j++) {
            c += s_head[j];
        }
        int n = 0;
        for (int j = t; j < rows && n < MOE_GROUP_COLS; j++) {
            if (s_entry[j] == e) {
                grp[1 + MOE_GROUP_MAX_ROWS + MOE_GROUP_COLS * c + n++] = j;
            }
        }
        grp[1 + c] = n;
    }
    if (t == 0) {
        int n = 0;
        for (int j = 0; j < rows; j++) {
            n += s_head[j];
        }
        grp[0] = n;
    }
}

static __global__ void k_moe_hosted_enter(const float * act, float * h_act, block_q8_0 * h_act8, int64_t n_act,
                                          const float * tk_logits, float * tk_weights, float tk_clamp, float tk_scale,
                                          int32_t * ids, int64_t ids_nb1, int32_t * h_ids, int n_used, int n_tokens,
                                          uint32_t * d_seq, uint32_t * d_expect, volatile moe_hosted_mailbox * mail,
                                          uint32_t slot, const volatile uint32_t * flag,
                                          const volatile uint64_t * m_plan, uint64_t * d_plan, int n_plan,
                                          volatile unsigned long long * tl, const moe_router_args rt,
                                          const float * w_src, float * h_w) {
    // this layer's publication number (d_seq[0] moves at the publication, after block 0 has read the router's rows): the
    // tag of the router's rows
    const uint32_t pub_tag = d_seq[0] + 1;
    // the router's workgroups after the entry's MOE_HOSTED_ENTER_BLOCKS
    if ((int) blockIdx.x >= MOE_HOSTED_ENTER_BLOCKS) {
        moe_router_row(rt, (int) blockIdx.x - MOE_HOSTED_ENTER_BLOCKS, pub_tag);
        if (rt.tagged) {
            // no fence: a device-scope fence here waits for the entry blocks' writes to the host still in flight
            // (twice the router's time on an RTX 4090)
            return;
        }
        ggml_cuda_fence_release_device();
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(rt.done, 1u);
        }
        return;
    }
    const int n_blocks = MOE_HOSTED_ENTER_BLOCKS;
    // the activations to the host: without block 0 when it computes the top-k of a tagged router (its system fence
    // would come before the top-k)
    const int b0 = tk_logits && rt.tagged ? 1 : 0;
    const int bx = (int) blockIdx.x - b0, n_copy = n_blocks - b0;
    for (int64_t i = bx * blockDim.x + threadIdx.x; h_act && bx >= 0 && i < n_act; i += (int64_t) n_copy * blockDim.x) {
        h_act[i] = act[i];
    }
    // and quantized for the CPU's experts as quantize_q8_1 does (a group of 32 threads per block): the CPU starts
    // from these bytes without quantizing them again
    {
        const int lane   = threadIdx.x % QK8_0;
        const int group  = (bx * (int) blockDim.x + (int) threadIdx.x) / QK8_0;
        const int groups = n_copy * blockDim.x / QK8_0;
        for (int64_t ib = group; bx >= 0 && ib < n_act / QK8_0; ib += groups) {
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
            if (rt.x && rt.tagged) {   // the router's rows, tagged with this layer
                const int n = rt.ncols * rt.nrows;
                for (int i = threadIdx.x; i < n; i += blockDim.x) {
                    uint64_t w = moe_load_relaxed_device(rt.tagged + i);
                    while ((uint32_t) (w >> 32) != pub_tag) {
                        ggml_cuda_poll_pause();
                        w = moe_load_relaxed_device(rt.tagged + i);
                    }
                    rt.dst[(i / rt.nrows) * rt.stride_col_dst + i % rt.nrows] = __uint_as_float((uint32_t) w);
                }
                __syncthreads();
            } else if (rt.x) {   // the router's rows
                if (threadIdx.x == 0) {
                    while (ggml_cuda_load_relaxed_device(rt.done) < (unsigned) rt.nrows) {
                        ggml_cuda_poll_pause();
                    }
                    *rt.done = 0;   // for the next layer (stream-ordered)
                }
                __syncthreads();
                ggml_cuda_fence_acquire_device();
            }
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
                if (h_w) {
                    h_w[i] = tk_weights[i];   // PSUM: the weights the top-k just wrote, for the other cards
                }
            }
        }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_ids; i += n_blocks * blockDim.x) {
            const int t = i / n_used;
            h_ids[i] = *(const int32_t *) ((const char *) ids + (int64_t) t * ids_nb1 + (int64_t) (i % n_used) * sizeof(int32_t));
            if (h_w) {
                h_w[i] = w_src[i];   // PSUM: the routing weights (computed before the block), for the other cards
            }
        }
    }
    // this block's writes reach the host before it counts itself
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        last = atomicAdd(d_seq + 1, 1u) == (unsigned) n_blocks - 1;
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
    // the plan's chunk list after it, for the grouped kernels (verification: several tokens)
    const int rows = n_plan / 3;
    if (n_tokens >= 2 && rows <= MOE_GROUP_MAX_ROWS) {
        __syncthreads();
        moe_group_list(d_plan, rows, (int32_t *) (d_plan + n_plan));
    }
}

// a block per row: the rows the CPU computed (down id -1) are copied in once they are ready
// SPLIT, kernel copy: moved experts per layer at most (a list of 1 + 3 x this many words)
#define MOE_SPLIT_FETCH_MAX 16

// SPLIT: the moved expert arrived (or none was moved): its plan to the device, and the kernel copy's list
static __global__ void k_moe_split_wait(const uint32_t * d_expect, const volatile uint32_t * flag,
                                        const volatile uint64_t * m_plan2, uint64_t * d_plan2, int n_plan,
                                        const volatile uint64_t * m_fetch, uint64_t * d_fetch,
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
    if (n_plan / 3 <= MOE_GROUP_MAX_ROWS) {
        __syncthreads();
        moe_group_list(d_plan2, n_plan / 3, (int32_t *) (d_plan2 + n_plan));
    }
    if (m_fetch) {
        const int n_fetch = 1 + 3 * (int) min((uint64_t) MOE_SPLIT_FETCH_MAX, m_fetch[0]);
        for (int i = threadIdx.x; i < n_fetch; i += blockDim.x) {
            d_fetch[i] = m_fetch[i];
        }
    }
}

// SPLIT, kernel copy: the moved experts read from their packed entries in RAM (mapped) and written to their cache entries,
// 16 bytes a thread, on the compute stream after the resident rows (the copy of Strata's PCIe share, fetch_blobs)
static __global__ void k_moe_split_fetch(const uint64_t * __restrict__ d_fetch) {
    const int n = (int) min((uint64_t) MOE_SPLIT_FETCH_MAX, d_fetch[0]);
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int k = 0; k < n; k++) {
        const uint4 * src = (const uint4 *) d_fetch[1 + 3 * k];
        uint4 * dst = (uint4 *) d_fetch[2 + 3 * k];
        const int64_t m = (int64_t) (d_fetch[3 + 3 * k] / sizeof(uint4));
        for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < m; i += stride) {
            dst[i] = src[i];
        }
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

// mul_mat_vec_f<float, float, ncols, 256> (no fusion) for one row: the same loads, products and reductions
static __device__ void moe_sgate_body(const moe_sgate_args & a) {
    constexpr int block_size = 256;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    const int tid = threadIdx.x;
    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();
    float sumf[MOE_SGATE_MAX_COLS] = {0.0f};
    const float2 * x2 = (const float2 *) a.x;
    const float2 * y2 = (const float2 *) a.y;
    for (int col2 = tid; col2 < a.ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < MOE_SGATE_MAX_COLS; ++j) {
            if (j < a.ncols) {
                const float2 tmpy = y2[j*a.stride_col_y2 + col2];
                ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < MOE_SGATE_MAX_COLS; ++j) {
        if (j < a.ncols) {
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            buf_iw[tid/warp_size] = sumf[j];
            __syncthreads();
            if (tid < warp_size) {
                sumf[j] = buf_iw[tid];
                sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
            }
            __syncthreads();
        }
    }
    if (tid < a.ncols) {
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < MOE_SGATE_MAX_COLS; ++j) {
            v = j == tid ? sumf[j] : v;
        }
        a.dst[tid*a.stride_col_dst] = v;
    }
}

// EXPERT SPLIT, on another card's stream: wait for the layer's plan (its flag changes from the publication this card
// served last), copy this card's part of it and the activations (F32: quantized here by the hosting card's kernel, the
// same q8_1 bytes; its q8_0 for the CPU differ in the block sum, which some types read)
static __device__ __forceinline__ unsigned long long moe_tl_clock();

static __global__ void k_split_card_wait(const volatile uint32_t * flag0, uint32_t * d_last, uint32_t * d_seq,
                                         const uint64_t * m_plan, uint64_t * d_plan, int n_plan,
                                         const float * m_act, float * d_act, int n_act,
                                         volatile uint32_t * diag, int card, int slot, volatile unsigned long long * tlk,
                                         int n_tokens, const float * m_w, float * d_w, int n_w) {
    __shared__ uint32_t seen;
    if (threadIdx.x == 0) {
        const uint32_t last = *d_last;
        uint32_t v = *flag0;
        const unsigned long long t0 = moe_tl_clock();
        while (v == last) {
            if (moe_tl_clock() - t0 > MOE_SPLIT_WAIT_BUDGET) {
                if (diag && diag[0] == 0) {
                    diag[1] = card; diag[2] = slot; diag[3] = last; diag[4] = v;
                    __threadfence_system();
                    diag[0] = 1;
                }
                break;
            }
            ggml_cuda_poll_pause();
            v = *flag0;
        }
        __threadfence_system();
        *d_last = v;
        *d_seq = v;
        seen = v;
        if (tlk) {
            tlk[(v % MOE_TL_RING) * 8 + 0] = t0;
            tlk[(v % MOE_TL_RING) * 8 + 1] = moe_tl_clock();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_plan; i += blockDim.x) {
        d_plan[i] = ((const volatile uint64_t *) m_plan)[i];
    }
    // PSUM: the routing weights, written with the activations (the plan's flag comes after both)
    for (int i = threadIdx.x; m_w && i < n_w; i += blockDim.x) {
        d_w[i] = ((const volatile float *) m_w)[i];
    }
    // the plan's chunk list after it, as the hosting card makes it: the rows of one expert computed together
    if (n_tokens >= 2 && n_plan / 3 <= MOE_GROUP_MAX_ROWS) {
        __syncthreads();
        moe_group_list(d_plan, n_plan / 3, (int32_t *) (d_plan + n_plan));
    }
    // the activations over the link: on CUDA 16-byte reads that skip the caches (__ldcv: the buffer is rewritten every layer),
    // four in flight per thread (one 4-byte volatile read at a time took ~35 us for 3 tokens)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (n_act % 4 == 0 && ((uintptr_t) m_act % 16) == 0 && ((uintptr_t) d_act % 16) == 0) {
        const float4 * src = (const float4 *) m_act;
        float4 * dst = (float4 *) d_act;
        const int n4 = n_act / 4, step = blockDim.x;
        int i = threadIdx.x;
        for (; i + 3 * step < n4; i += 4 * step) {
            const float4 a = __ldcv(src + i), b = __ldcv(src + i + step), c = __ldcv(src + i + 2 * step), e = __ldcv(src + i + 3 * step);
            dst[i] = a; dst[i + step] = b; dst[i + 2 * step] = c; dst[i + 3 * step] = e;
        }
        for (; i < n4; i += step) {
            dst[i] = __ldcv(src + i);
        }
    } else
#endif
    {
        for (int i = threadIdx.x; i < n_act; i += blockDim.x) {
            d_act[i] = ((const volatile float *) m_act)[i];
        }
    }
    if (tlk) {
        __syncthreads();
        if (threadIdx.x == 0) {
            tlk[(seen % MOE_TL_RING) * 8 + 2] = moe_tl_clock();
        }
    }
}

// EXPERT SPLIT: the experts the dispatcher moved to this card arrived (its flag reached the publication): their rows' part of
// the plan
static __global__ void k_split_card_wait2(const volatile uint32_t * flag, const uint32_t * d_seq, const uint64_t * m_plan2,
                                          uint64_t * d_plan2, int n_plan, volatile uint32_t * diag, int card,
                                          volatile unsigned long long * tlk, int n_tokens) {
    __shared__ uint32_t seq;
    if (threadIdx.x == 0) {
        seq = *d_seq;
        const unsigned long long t0 = moe_tl_clock();
        if (tlk) {
            tlk[(seq % MOE_TL_RING) * 8 + 3] = t0;
        }
        while ((int32_t) (*flag - seq) < 0) {
            if (moe_tl_clock() - t0 > MOE_SPLIT_WAIT_BUDGET) {
                if (diag && diag[0] == 0) {
                    diag[1] = card; diag[2] = 0; diag[3] = seq; diag[4] = *flag;
                    __threadfence_system();
                    diag[0] = 3;
                }
                break;
            }
            ggml_cuda_poll_pause();
        }
        __threadfence_system();
        if (tlk) {
            tlk[(seq % MOE_TL_RING) * 8 + 4] = moe_tl_clock();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_plan; i += blockDim.x) {
        d_plan2[i] = ((const volatile uint64_t *) m_plan2)[i];
    }
    if (n_tokens >= 2 && n_plan / 3 <= MOE_GROUP_MAX_ROWS) {
        __syncthreads();
        moe_group_list(d_plan2, n_plan / 3, (int32_t *) (d_plan2 + n_plan));
    }
}

// EXPERT SPLIT: this card's rows are in the hosting card's result buffer: its flag
static __global__ void k_split_card_done(const uint32_t * d_seq, volatile uint32_t * flag, volatile unsigned long long * tlk) {
    __threadfence_system();
    const uint32_t seq = *d_seq;
    *flag = seq;
    if (tlk) {
        tlk[(seq % MOE_TL_RING) * 8 + 5] = moe_tl_clock();
        tlk[(seq % MOE_TL_RING) * 8 + 6] = seq;
    }
}

// EXPERT SPLIT, PSUM: on another card, a block per token: its rows of the token (resident in its cache or moved to it)
// summed with their routing weights, written to the result row of the first of them divided by that row's weight; the
// merge writes 0 to the others, and the weighting that follows the block gives back the sum
static __global__ void k_split_card_psum(const uint64_t * d_plan, const uint64_t * d_plan2, int rows, int n_used,
                                         const float * d_w, const float * d_res, float * m_res, int n_embd) {
    const int t = blockIdx.x;
    int own[16];
    int n = 0;
    for (int s = 0; s < n_used && n < 16; s++) {
        const int r = t * n_used + s;
        if (d_plan[2 * rows + r] != 0 || d_plan2[2 * rows + r] != 0) {
            own[n++] = r;
        }
    }
    if (n == 0) {
        return;
    }
    // the weights from this card's memory: read from host memory, every thread's own reads crossed the link (a round
    // trip each, in turn), in proportion to the card's rows
    const float wd = d_w[own[0]];
    float scale[16];
    for (int j = 0; j < n; j++) {
        scale[j] = wd != 0.0f ? d_w[own[j]] / wd : (j == 0 ? 1.0f : 0.0f);
    }
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float sum = d_res[(size_t) own[0] * n_embd + i];
        for (int j = 1; j < n; j++) {
            sum += scale[j] * d_res[(size_t) own[j] * n_embd + i];
        }
        m_res[(size_t) own[0] * n_embd + i] = sum;
    }
}

static __global__ void k_moe_hosted_merge(const uint32_t * d_expect, const volatile uint32_t * flag,
                                          const uint64_t * d_down_ids, const uint64_t * d_moved_down_ids,
                                          const volatile float * m_res, float * dst, int n_embd,
                                          volatile unsigned long long * tl, const moe_sgate_args sg,
                                          const volatile int8_t * m_src, const volatile uint32_t * flags,
                                          volatile uint32_t * diag, int n_used, int psum) {
    if (sg.x && blockIdx.x == gridDim.x - 1) {
        moe_sgate_body(sg);   // the last workgroup: the shared expert's gate
        return;
    }
    const int row = blockIdx.x;
    if (d_down_ids[row] != 0 || (d_moved_down_ids && d_moved_down_ids[row] != 0)) {
        return;   // resident, or moved to the GPU (SPLIT)
    }
    __shared__ uint32_t expect;
    __shared__ int zero;
    if (threadIdx.x == 0) {
        zero = 0;
        expect = *d_expect;
        const unsigned long long t0 = tl ? moe_tl_clock() : 0;
        while ((int32_t) (*flag - expect) < 0) {
            moe_hosted_pause();
        }
        __threadfence_system();
        // EXPERT SPLIT: a row of another card waits for that card's flag too; its producer is read after the CPU's flag,
        // which the host raises after the dispatcher wrote the rows it moved to a card
        const int src = m_src ? m_src[row] : 0;
        // PSUM: a card's row after its first one in the token: in that first row's weighted sum, here 0
        if (src > 0 && psum) {
            for (int r = (row / n_used) * n_used; r < row && !zero; r++) {
                zero = m_src[r] == src;
            }
        }
        if (src > 0 && !zero) {
            const volatile uint32_t * kflag = flags + 32 + 8 * src;
            const unsigned long long tw = moe_tl_clock();
            while ((int32_t) (*kflag - expect) < 0) {
                if (diag && moe_tl_clock() - tw > MOE_SPLIT_WAIT_BUDGET) {
                    if (diag[0] == 0) {
                        diag[1] = src; diag[2] = row; diag[3] = expect; diag[4] = *kflag;
                        __threadfence_system();
                        diag[0] = 2;
                    }
                    break;
                }
                moe_hosted_pause();
            }
            __threadfence_system();
        }
        if (tl) {
            tl[(expect % MOE_TL_RING) * 8 + 4] = t0;   // the merge reached (one of its waiting blocks)
            tl[(expect % MOE_TL_RING) * 8 + 2] = moe_tl_clock();
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[(size_t) row * n_embd + i] = zero ? 0.0f : m_res[(size_t) row * n_embd + i];
    }
}

// Unpin the rows the GPU read for the last planned layer (session lock held). The GPU is past that layer (the next one's
// publication, or the stream synchronized): the entries its kernel copy filled are valid, as the fill thread makes them.
static void moe_hosted_unpin_locked(moe_hosted_state & st) {
    for (const moe_cache_job & job : st.gpu_fills) {
        moe_cache_device & device = *st.device;
        if (job.pool < 0 || job.pool >= (int) device.pools.size() || !device.pools[job.pool]) {
            continue;
        }
        moe_cache_pool & pool = *device.pools[job.pool];
        if (job.slot < 0 || job.slot >= pool.n_slots) {
            continue;
        }
        moe_cache_slot & slot = pool.slots[job.slot];
        if (slot.state == moe_cache_slot_state::copying && slot.generation == job.generation && slot.key == job.key) {
            slot.state = moe_cache_slot_state::valid;
            moe_cache_lru_push_back(pool, job.slot);
            device.demand_count.erase(job.key);
            device.fills++;
        }
    }
    st.gpu_fills.clear();
    for (const moe_cache_pin & pin : st.pins) {
        if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots && pin.pool->slots[pin.slot].readers > 0) {
            pin.pool->slots[pin.slot].readers--;
        }
    }
    st.pins.clear();
}

// KV: the rows the hosted plan pinned for the session's last layer, unpinned (its stream is done: the caller synchronized)
// (device: that device's only, nullptr: every device's)
static void moe_hosted_unpin_session(moe_cache_session & session, const moe_cache_device * device) {
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    for (auto & entry : g_hosted) {
        if (entry.first.first == &session && entry.second && (!device || entry.first.second == device->logical)) {
            moe_hosted_unpin_locked(*entry.second);
        }
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
// Deferred admissions (GGML_CUDA_MOE_CACHE_DEFER=0 disables). A copy into the cache reads the RAM at ~13 GB/s: during
// the CPU rows (RAM bound, ~31 GB/s) it slowed them by ~30 %. The plan of a main-model layer only records the
// candidates the admission rule passes; at the end of the layer's CPU rows its most demanded one is admitted, its copy
// running while the GPU finishes the layer; at the end of the pass's last layer, all the others still absent, most
// demanded first: their copies run during the MTP draft, and the next pass's first layer waits for them. The victim is
// chosen then, not at the plan. MEASURED: see the commit.
static void moe_hosted_admit_deferred(moe_hosted_state & st, moe_hosted_layer & layer) {
    moe_cache_session & session = *st.session;
    moe_cache_device & device = *st.device;
    if (!session.config.defer) {
        return;
    }
    std::lock_guard<std::mutex> lock(session.mu);
    if (device.main_first < 0 || layer.layer_number < device.main_first || layer.layer_number > device.main_last) {
        return;
    }
    auto & c = device.deferred;
    bool wake = false;
    const auto admit = [&](const moe_cache_device::deferred_candidate & k) -> bool {
        if (session.stopping || session.dormant || device.dead.load() || k.pool < 0 || k.pool >= (int) device.pools.size()) {
            return false;
        }
        moe_cache_pool & pool = *device.pools[k.pool];
        if (!pool.slab || pool.map.count(k.key)) {
            return false;
        }
        int inserts_left = 1;
        if (!moe_cache_enqueue_locked(session, device, pool, k.pool, k.key.tensor, k.key.expert, pool.expert_size,
                                      inserts_left, wake, true, false, false, k.freq, k.contest)) {
            return false;
        }
        if (session.config.lfu) {
            device.demand_count.erase(k.key);
        }
        return true;
    };
    if (!device.arenas.empty()) {   // ARENA: the candidates the arena does not hold leave the list
        c.erase(std::remove_if(c.begin(), c.end(), [&](const auto & k) {
            return k.pool >= 0 && k.pool < (int) device.pools.size() &&
                   moe_arena_absent_locked(device, *device.pools[k.pool], k.key);
        }), c.end());
    }
    if (layer.layer_number == device.main_last) {   // the pass's end: all of them, in order
        for (const auto & k : c) {
            admit(k);
        }
        c.clear();
    } else {   // a layer's end: the first of the whole list the rule admits leaves it; the refused stay
        for (auto it = c.begin(); it != c.end(); ++it) {
            if (admit(*it)) {
                c.erase(it);
                break;
            }
        }
    }
    if (wake) {
        session.cv.notify_all();
    }
}

// ARENA: generation starts: each arena's zone given back to it, and the experts its lending evicted read back, hottest
// first, into its free slots by the reader threads behind any read the CPU rows wait for. A prefill lending again
// (arena_gen bumped) makes the reads still queued give up
// ARENA: one expert read back into a free slot of its arena by the reader threads, behind any read the CPU rows wait for;
// a prefill lending again (arena_gen bumped) makes it give up (session lock held)
static void moe_arena_restore_one_locked(moe_cache_session & sess, moe_cache_device & device, moe_expert_arena * ar,
                                         const moe_cache_key & key, int pi, uint64_t gen) {
    if (ar->free_list.empty() || pi < 0 || pi >= (int) device.pools.size()) {
        return;   // restored into free slots only: no eviction for it
    }
    // an expert the VRAM cache holds is restored too, as the arena keeps the VRAM's experts everywhere. MEASURED (32 GB):
    // skipped, 730 of the 3 456 a lending evicted were not restored and short-prompt steps took +2.1 % (2 ABBA pairs);
    // the preload's copies given back and an expert moved to the cold end at its fill: the CPU's reads from the file
    // x1.9, refused admissions x1.8, steps +30 %. The VRAM cache turns over (~50 000 evictions a run) and an expert it
    // evicts comes back through the arena
    const moe_cache_layer * cl = moe_cache_layer_of(device, key.tensor);
    if (!cl) {
        return;
    }
    const moe_cache_pool * pp = device.pools[pi].get();
    moe_cache_job jb;
    moe_cache_job_sources(*cl, *pp, key.expert, jb);
    const int t = ar->take(key, pi);
    if (t < 0) {
        return;   // there already (read by the CPU rows meanwhile)
    }
    moe_cache_device * d = &device;
    moe_cache_session * session = &sess;
    device.arena_readers->post([d, session, ar, t, jb, pp, gen] {
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            ok = d->arena_gen == gen;
        }
        ok = ok && moe_arena_load(*d, *ar, t, jb, *pp);
        std::lock_guard<std::mutex> lock(session->mu);
        if (ok) {
            ar->commit(t);
            ar->restored++;
        } else {
            ar->abort(t);
        }
    });
}

// ARENA: generation starts: each arena's zone given back to it, then the restoration (moe_arena_restore_prompt). The
// prompt's counts are copied from the device on a reader thread, not here: this thread answers the hosted layer the GPU
// waits for, and a copy queued behind that wait would never run
static void moe_arena_restore_start_locked(moe_cache_session & sess, moe_cache_device & device) {
    if (device.arenas.empty() || !device.arena_readers) {
        return;
    }
    const uint64_t gen = device.arena_gen;
    std::vector<std::pair<moe_expert_arena *, std::vector<std::pair<moe_cache_key, int>>>> evicted;
    for (auto & a : device.arenas) {
        if (a && a->zone_active) {
            evicted.push_back({a.get(), a->give_back()});
        }
    }
    if (evicted.empty()) {
        return;
    }
    if (!moe_arena_restore_prompt() || !g_prefill_counts.load()) {
        for (auto & [ar, keys] : evicted) {
            for (const auto & [key, pi] : keys) {
                moe_arena_restore_one_locked(sess, device, ar, key, pi, gen);
            }
        }
        return;
    }
    moe_cache_device * d = &device;
    moe_cache_session * session = &sess;
    device.arena_readers->post([d, session, gen] {
        const int rows = g_prefill_counts_rows, n = g_prefill_counts_n;
        int * dev = g_prefill_counts.load();
        if (!dev) {
            return;
        }
        static thread_local cudaStream_t stream = nullptr;
        ggml_cuda_set_device(d->logical);
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        std::vector<int> counts((size_t) rows * n);
        if (cudaMemcpyAsync(counts.data(), dev, counts.size() * sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                cudaMemsetAsync(dev, 0, counts.size() * sizeof(int), stream) != cudaSuccess ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
            (void) cudaGetLastError();
            return;
        }
        std::lock_guard<std::mutex> lock(session->mu);
        if (d->arena_gen != gen) {
            return;
        }
        // the experts the prompt routed rows to that the arena does not hold, most rows first
        struct cand { int rows; moe_expert_arena * ar; moe_cache_key key; int pi; };
        std::vector<cand> c;
        for (const auto & [number, layer] : d->layers) {
            moe_expert_arena * ar = moe_arena_of(*d, &layer);
            if (!ar || number >= rows || layer.shape < 0) {
                continue;
            }
            const int pi = d->shapes[layer.shape].pool;
            for (int e = 0; e < n && e < layer.n_expert; e++) {
                const int r = counts[(size_t) number * n + e];
                const moe_cache_key key{layer.tensor[MOE_ROLE_GATE], e};
                if (r > 0 && !ar->map.count(key)) {
                    c.push_back({r, ar, key, pi});
                }
            }
        }
        std::stable_sort(c.begin(), c.end(), [](const cand & a, const cand & b) { return a.rows > b.rows; });
        for (const cand & x : c) {
            moe_arena_restore_one_locked(*session, *d, x.ar, x.key, x.pi, gen);
        }
    });
}

// EXPERT SPLIT (GGML_MOE_CACHE_EXPERT_SPLIT=1, several GPUs): every card's cache holds experts of every layer, none twice;
// the card that hosts a layer's block plans its rows by where each expert is: its own cache, another card's, or none (the
// CPU, or an admission to the card with the most room); the rows of another card's cache are computed by that card
// (moe_split_card), the merge waits for each row's producer

// another card's cache entry for (key, expert): its device index in the session and its slot, else -1 (session lock)
static int moe_split_find_locked(moe_cache_session & session, const moe_cache_device & self, const void * key, int32_t expert,
                                 int * slot_out) {
    for (int di = 0; di < (int) session.devices.size(); di++) {
        moe_cache_device & dk = *session.devices[di];
        if (&dk == &self || dk.dead.load() || !dk.budget_ready) {
            continue;
        }
        const int pi = moe_cache_pool_of(dk, key);
        if (pi < 0 || pi >= (int) dk.pools.size() || !dk.pools[pi]->slab) {
            continue;
        }
        moe_cache_pool & pk = *dk.pools[pi];
        const auto d = pk.direct.find(key);
        if (d == pk.direct.end() || expert >= (int) d->second.size()) {
            continue;
        }
        const int s = d->second[expert];
        if (s >= 0 && pk.slots[s].state == moe_cache_slot_state::valid) {
            *slot_out = s;
            return di;
        }
    }
    return -1;
}

// the card an absent expert is admitted to: the one whose pool for the layer has the most free entries, else the one whose
// coldest entry is the least demanded (session lock). Returns its device index, -1 for none
static int moe_split_target_locked(moe_cache_session & session, const void * key) {
    int best = -1;
    long best_free = -1;
    int best_cold = INT_MAX;
    for (int di = 0; di < (int) session.devices.size(); di++) {
        moe_cache_device & dk = *session.devices[di];
        if (dk.dead.load() || !dk.budget_ready) {
            continue;
        }
        const int pi = moe_cache_pool_of(dk, key);
        if (pi < 0 || pi >= (int) dk.pools.size() || !dk.pools[pi]->slab) {
            continue;
        }
        const moe_cache_pool & pk = *dk.pools[pi];
        const long free = (long) pk.free_slots.size();
        const int cold = pk.lru_head >= 0 ? (int) pk.slots[pk.lru_head].freq : 0;
        if (free > best_free || (free == 0 && best_free == 0 && cold < best_cold)) {
            best = di;
            best_free = free;
            best_cold = cold;
        }
    }
    return best;
}

// an absent expert's demand counted by the planning card (one count whatever card admits it), its admission decided on the
// target card's pool and queued to that card's fill thread (session lock)
static void moe_split_queue_away_locked(moe_cache_session & session, moe_cache_device & owner, moe_cache_device & target,
                                        int pool_index, const void * host_base, int32_t expert, int & inserts_left,
                                        bool & wake_worker) {
    moe_cache_pool & pool = *target.pools[pool_index];
    const moe_cache_key key{host_base, expert};
    owner.freq_demands++;
    owner.renewed_demands++;
    const uint16_t freq_cap = session.config.lfu ? (uint16_t) session.config.lfu_cap : std::numeric_limits<uint16_t>::max();
    moe_cache_demand * demand = nullptr;
    try {
        demand = &owner.demand_count[key];
    } catch (...) {
        owner.insert_skips++;
        return;
    }
    demand->expert_size = pool.expert_size;
    demand->count += demand->count < freq_cap;
    const int initial_admit_after = pool.covers_all_entries && !session.config.admit_after_explicit ? 1 : session.config.admit_after;
    const bool contest = session.config.lfu && pool.free_slots.empty();
    const int admit_after = pool.free_slots.empty() && !contest ? std::max(initial_admit_after, session.config.readmit_after)
                                                                : initial_admit_after;
    if (demand->count < admit_after) {
        owner.admission_skips++;
        return;
    }
    const bool deferring = target.deferring;
    target.deferring = false;   // its own list is replayed by its own hosted layers only
    if (moe_cache_enqueue_locked(session, target, pool, pool_index, host_base, expert, pool.expert_size, inserts_left,
                                 wake_worker, true, false, false, demand->count, contest)) {
        owner.split_admit_away++;
        if (session.config.lfu) {
            owner.demand_count.erase(key);
        }
    }
    target.deferring = deferring;
}

static int moe_hosted_plan(moe_hosted_state & st, moe_hosted_layer & layer, const int32_t * ids, int rows,
                           uint64_t * out, int32_t * cpu_rows, uint32_t * flag, uint32_t seq) {
    moe_cache_session & session = *st.session;
    moe_cache_device & device = *st.device;
    int n_cpu = 0;
    bool wake_worker = false;
    {
        std::unique_lock<std::mutex> lock(session.mu);
        if (layer.layer_number == -2) {
            const auto it = device.layer_of.find(layer.gate_w->data);
            layer.layer_number = it != device.layer_of.end() ? it->second : -1;
        }
        const bool defer = session.config.defer && device.main_first >= 0 &&
            layer.layer_number >= device.main_first && layer.layer_number <= device.main_last;
        if (layer.layer_number == device.main_first && device.rewarm_armed) {
            device.rewarm_armed = false;
            moe_cache_rewarm_start_locked(session, device);
            if (moe_cache_rewarm_bulk_on() && device.rewarm_active) {
                device.rewarm_bulk_pending = true;   // the fill worker does it (moe_cache_worker)
                session.cv.notify_all();
            }
            moe_arena_restore_start_locked(session, device);
        }
        if (defer && layer.layer_number == device.main_first) {
            device.deferred.clear();   // a new pass, a new list
            // the previous pass's admissions land before this one computes (they copied during the draft); not the
            // restoration, which keeps one job queued until it ends (waiting for it held the answer's first pass for
            // its whole ~0.3 s)
            session.idle_cv.wait(lock, [&] {
                return ((!device.inflight || device.inflight_rewarm) &&
                        std::all_of(device.queue.begin(), device.queue.end(), [](const moe_cache_job & j) { return j.rewarm; })) ||
                    session.stopping || device.dead.load();
            });
        }
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
        // EXPERT SPLIT: the rows not resident here, looked up in the other cards' caches
        const bool split = moe_cache_expert_split() && session.devices.size() > 1;
        int remote[moe_hosted_rows_max];
        int remote_slot[moe_hosted_rows_max];
        bool remote_run[moe_hosted_rows_max];   // computed by that card (its chain is queued), not by the CPU
        if (split && layer.h_src) {
            memset(layer.h_src, 0, rows);
            memset(layer.h_plank, 0, (size_t) MOE_SPLIT_MAX_CARDS * 3 * rows * sizeof(uint64_t));
        }
        for (int row = 0; row < rows; row++) {
            remote[row] = -1;
            remote_run[row] = false;
            if (split && slots[row] < 0 && ids[row] >= 0 && ids[row] < n_expert) {
                remote[row] = moe_split_find_locked(session, device, key, ids[row], &remote_slot[row]);
            }
            if (remote[row] < 0 || !layer.h_src) {
                continue;
            }
            moe_cache_device & dk = *session.devices[remote[row]];
            for (const moe_split_card & card : st.split_cards) {
                if (card.device != &dk) {
                    continue;
                }
                moe_cache_pool & pk = *dk.pools[moe_cache_pool_of(dk, key)];
                const uint64_t entry = moe_cache_entry_addr(pk, remote_slot[row]);
                uint64_t * plan = layer.h_plank + (size_t) (card.index - 1) * 3 * rows;
                for (int k = 0; k < MOE_ROLES; k++) {
                    plan[k * rows + row] = entry + pk.role_offset[k];
                }
                layer.h_src[row] = (int8_t) card.index;
                pk.slots[remote_slot[row]].readers++;
                st.pins.push_back({&pk, remote_slot[row]});
                remote_run[row] = true;
            }
        }
        __atomic_store_n(flag, seq, __ATOMIC_RELEASE);
        if (!st.tl_host.empty()) {
            st.tl_host[(seq % MOE_TL_RING) * 3 + 1] = moe_tl_now_ns();
        }

        if (moe_cache_profile_save_path()) {
            if (layer.layer_number == -2) {
                const auto it = device.layer_of.find(key);
                layer.layer_number = it != device.layer_of.end() ? it->second : -1;
            }
            moe_cache_heat_note(device, layer.layer_number, ids, rows, n_expert);
        }

        // admissions: inserts_per_plan / 2 experts per plan, the rate the pools per tensor shape had (a budget of
        // inserts_per_plan tensors for the gate / up pool, two tensors an expert)
        int inserts = std::max(1, session.config.inserts_per_plan / 2);
        device.deferring = defer;
        for (int row = 0; row < rows; row++) {
            const int32_t expert = ids[row];
            if (!pool || expert < 0 || expert >= n_expert) {
                cpu_rows[n_cpu++] = row;
                continue;
            }
            if (slots[row] >= 0) {
                device.hits++;
                device.split_local += split;
                device.freq_demands++;
                device.renewed_demands++;
                moe_cache_slot & slot = pool->slots[slots[row]];
                slot.freq += slot.freq < (session.config.lfu ? session.config.lfu_cap : std::numeric_limits<uint16_t>::max());
                moe_cache_lru_remove(*pool, slots[row]);
                moe_cache_lru_push_back(*pool, slots[row]);
                continue;
            }
            if (!remote_run[row]) {
                cpu_rows[n_cpu++] = row;
            }
            if (split) {
                if (remote[row] >= 0) {
                    // in another card's cache: its demand counted there (frequency, recency); still a CPU row (step 2)
                    moe_cache_device & dk = *session.devices[remote[row]];
                    moe_cache_pool & pk = *dk.pools[moe_cache_pool_of(dk, key)];
                    moe_cache_slot & slot = pk.slots[remote_slot[row]];
                    slot.freq += slot.freq < (session.config.lfu ? session.config.lfu_cap : std::numeric_limits<uint16_t>::max());
                    moe_cache_lru_remove(pk, remote_slot[row]);
                    moe_cache_lru_push_back(pk, remote_slot[row]);
                    dk.split_served++;
                    device.split_remote++;
                    device.hits += remote_run[row];
                    device.misses += !remote_run[row];
                    continue;
                }
                device.split_absent++;
                device.misses++;
                const int t = moe_split_target_locked(session, key);
                if (t >= 0 && session.devices[t].get() != &device) {
                    moe_cache_device & dt = *session.devices[t];
                    moe_split_queue_away_locked(session, device, dt, moe_cache_pool_of(dt, key), key, expert, inserts,
                                                wake_worker);
                    continue;
                }
                (void) moe_cache_lookup_or_queue_locked(session, device, *pool, layer.pool_index, key, expert,
                                                        pool->expert_size, inserts, wake_worker);
                continue;
            }
            if (moe_cache_lookup_or_queue_locked(session, device, *pool, layer.pool_index, key, expert, pool->expert_size,
                                                 inserts, wake_worker) >= 0) {
                device.hits++;
            } else {
                device.misses++;
            }
        }
        device.deferring = false;
        device.nodes++;
        moe_cache_freq_age_locked(session, device);
    }
    if (wake_worker) {
        session.cv.notify_all();
    }
    return n_cpu;
}

// SPLIT (GGML_CUDA_MOE_SPLIT=0 disables): in a pass of 2 tokens or more, experts of the CPU rows are moved to the GPU when
// that ends the MoE window sooner: the best one or pair, then one more at a time while that still gains. Their missing tensors take cache entries (an admission) and go to the
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
// the per-expert and per-row terms were fitted with a CPU team of 8 threads (Ryzen 7 3800XT): they are divided among the
// team's threads, the fixed term and the concurrent copies' (they read the same RAM) are not. The measured scale
// (moe_split_calibrate) is left with the rest (clock, RAM): 8 threads give the fitted model as it was, a 20-thread team
// of a 2.3 GHz EPYC a 0.4 x model that its scale lifts again (it measured 0.57 x without the team in the model)
static constexpr double moe_split_cpu_fit_threads = 8.0;

static double moe_split_cpu_model_us(int n_threads, int d, int r, int tensors) {
    const double share = moe_split_cpu_fit_threads / std::max(1, n_threads);
    return moe_split_cpu_fixed_us + (moe_split_cpu_expert_us * d + moe_split_cpu_row_us * r) * share +
           moe_split_cpu_copy_us * tensors;
}
static constexpr double moe_split_copy_start_us = 9.1, moe_split_copy_tensor_us = 11.4;
static constexpr double moe_cache_pcie_copy_efficiency = 13.6 / (8.0 * 16 * 128.0 / 130.0 / 8.0);
static constexpr double moe_split_gpu_before_us = 23.3, moe_split_gpu_row_us = 2.97;
static constexpr double moe_split_gpu_after_us = 66.4, moe_split_gpu_moved_row_us = 1.5;

// SPLIT: a moved expert of a packed layer copied by k_moe_split_fetch on the compute stream, into the entry it is admitted
// to (GGML_CUDA_MOE_SPLIT_COPY=kernel), or by the fill thread's DMA (=dma): no queue, one read of the entry, no flag to
// wait for. Default: kernel on NVIDIA; DMA under HIP, where a kernel reading RAM slows the other queues' kernels
static bool moe_split_copy_by_kernel() {
    static const bool kernel = [] {
        const char * v = getenv("GGML_CUDA_MOE_SPLIT_COPY");
        if (v) {
            return strcmp(v, "kernel") == 0;
        }
#if defined(GGML_USE_HIP)
        return false;
#else
        return true;
#endif
    }();
    return kernel;
}

static bool moe_split_enabled() {
#if defined(_WIN32)
    // off by default: on Windows the GPU can hang in k_moe_split_wait until the driver resets it (TDR)
    static const bool enabled = getenv("GGML_CUDA_MOE_SPLIT") && atoi(getenv("GGML_CUDA_MOE_SPLIT")) != 0;
#else
    static const bool enabled = !getenv("GGML_CUDA_MOE_SPLIT") || atoi(getenv("GGML_CUDA_MOE_SPLIT")) != 0;
#endif
    return enabled;
}

// EXPERT SPLIT: the moved experts' flag of every card the dispatcher sent none to (its chain waits for it in every layer)
static void moe_split_cards_release(moe_hosted_state * st, moe_hosted_layer * layer, uint32_t seq) {
    for (const moe_split_card & card : st->split_cards) {
        if (__atomic_load_n(&layer->split_pending_k[card.index], __ATOMIC_ACQUIRE) == 0) {
            __atomic_store_n(&layer->h_flags[34 + 8 * card.index], seq, __ATOMIC_RELEASE);
        }
    }
}

static int moe_hosted_split(moe_hosted_state * st, moe_hosted_layer * layer, int rows, int n_tokens, uint32_t seq,
                            int32_t * cpu_rows, int n_cpu) {
    if (!st->split_cards.empty() && layer->h_plan2k) {
        memset(layer->h_plan2k, 0, (size_t) MOE_SPLIT_MAX_CARDS * 3 * rows * sizeof(uint64_t));
    }
    if (!moe_split_enabled() || n_tokens < 2) {
        moe_split_cards_release(st, layer, seq);
        return n_cpu;
    }
    st->split_tensors = 0;
    uint64_t * plan2 = layer->h_plan2;
    for (int i = 0; i < 3 * rows; i++) {
        plan2[i] = 0;
    }
    layer->h_fetch[0] = 0;
    int n_fetch = 0;
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
            const bool by_kernel = moe_split_copy_by_kernel();
            // a packed layer fills an entry in one copy (the fill's whole job): one copy and one tensor per moved expert
            // in the model, not three (the model was fitted before the packing)
            bool packed = false;
            if (const moe_cache_layer * cl = moe_cache_layer_of(device, key)) {
                packed = true;
                for (int r = 0; r < MOE_ROLES; r++) {
                    packed = packed && cl->stride[r] == pool->expert_size &&
                        (const char *) cl->tensor[r] == (const char *) cl->tensor[0] + pool->role_offset[r];
                }
            }
            // a candidate: an expert of the CPU rows, its entry (absent: -1, or copying), the time of its three copies one
            // after another
            // ARENA: an absent expert the arena does not hold is not a candidate (its copy would wait for a read of
            // the file); it stays on the CPU rows, which read it into the arena
            struct cand { int32_t e; int rows; int slot; double copies; int tensors; };
            std::vector<cand> c;
            std::vector<int32_t> unheld;
            const moe_cache_layer * acl = device.arenas.empty() ? nullptr : moe_cache_layer_of(device, key);
            moe_expert_arena * aar = moe_arena_of(device, acl);
            for (int j = 0; j < n_cpu; j++) {
                const int32_t e = layer->h_ids[cpu_rows[j]];
                bool seen = std::find(unheld.begin(), unheld.end(), e) != unheld.end();
                for (auto & x : c) {
                    if (x.e == e) { x.rows++; seen = true; }
                }
                if (!seen) {
                    cand x = {e, 1, -1, 0.0, packed ? 1 : MOE_ROLES};
                    const auto f = pool->map.find(moe_cache_key{key, e});
                    if (f != pool->map.end()) {
                        x.slot = f->second;
                    } else if (aar) {
                        moe_cache_job probe;
                        moe_cache_job_sources(*acl, *pool, e, probe);
                        const auto a = aar->map.find(moe_cache_key{key, e});
                        if (moe_arena_wanted(device, aar, probe, *pool) &&
                                (a == aar->map.end() || !aar->slots[a->second].valid)) {
                            unheld.push_back(e);
                            continue;
                        }
                    }
                    if (packed) {
                        // kernel copy: the entry at the link's rate, no copy overhead nor queue
                        x.copies = (by_kernel ? 0.0 : moe_split_copy_tensor_us) + pool->expert_size / rate;
                    }
                    for (int k = 0; k < MOE_ROLES && !packed; k++) {
                        x.copies += moe_split_copy_tensor_us + pool->role_bytes[k] / rate;
                    }
                    c.push_back(x);
                }
            }
            // the window's end: the later of the CPU rows and the GPU's work (the resident rows, the wait for the moved
            // experts, then the work queued behind it)
            const double scale = st->cpu_scale;
            const int n_threads = st->n_threads;
            const auto cpu_time = [scale, n_threads](int d, int r, int tensors) {
                return d > 0 ? scale * moe_split_cpu_model_us(n_threads, d, r, tensors) : 0.0;
            };
            const double gs = st->gpu_scale;   // the GPU terms scaled to this card (moe_split_gpu_bench)
            const double gpu_before = gs * (moe_split_gpu_before_us + moe_split_gpu_row_us * (rows - n_cpu));
            // kernel copy (packed layers): it runs after the resident rows on the same stream, so it adds to the GPU's path
            const auto window = [&](int d, int r, int moved_rows, int tensors, double copies) {
                const double arrival = tensors == 0 ? 0.0 : by_kernel && packed ? gpu_before + copies : moe_split_copy_start_us + copies;
                const double gpu_after = gs * (moe_split_gpu_after_us + moe_split_gpu_moved_row_us * moved_rows);
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
            // past the best pair, one more expert at a time while one still ends the window sooner (the pairs are
            // searched in full: one expert alone rarely pays for its copy, two may)
            if (best.size() == 2) {
                int r = c[best[0]].rows + c[best[1]].rows, t = c[best[0]].tensors + c[best[1]].tensors;
                double cp = c[best[0]].copies + c[best[1]].copies;
                for (;;) {
                    int add = -1;
                    for (int i = 0; i < D; i++) {
                        if (std::find(best.begin(), best.end(), i) != best.end()) {
                            continue;
                        }
                        const int n = (int) best.size() + 1;
                        const double crit = window(D - n, n_cpu - r - c[i].rows, r + c[i].rows, t + c[i].tensors, cp + c[i].copies);
                        if (all - crit > best_gain) {
                            best_gain = all - crit;
                            add = i;
                        }
                    }
                    if (add < 0) {
                        break;
                    }
                    best.push_back(add);
                    r += c[add].rows;
                    t += c[add].tensors;
                    cp += c[add].copies;
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
                    // its demand count goes with it (at 0 it would be the next victim of a contested admission, just
                    // after the copy it was moved for)
                    uint16_t freq = 0;
                    if (session.config.lfu) {
                        const auto d = device.demand_count.find(moe_cache_key{key, x.e});
                        freq = d != device.demand_count.end() ? d->second.count : 0;
                    }
                    if (moe_cache_enqueue_locked(session, device, *pool, idx, key, x.e, pool->expert_size, inserts_left, wake,
                                                 true, false, false, freq)) {
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
                const uint64_t entry = moe_cache_entry_addr(*pool, x.slot);
                // kernel copy: a packed entry registered in place (mapped), aligned for 16-byte reads
                const void * src_dev = nullptr;
                if (moe_split_copy_by_kernel() && job.whole && n_fetch < MOE_SPLIT_FETCH_MAX && job.bytes % 16 == 0 &&
                        (uintptr_t) job.source[0] % 16 == 0 && entry % 16 == 0) {
                    const uintptr_t b = (uintptr_t) job.source[0], e = b + job.bytes;
                    for (const auto & range : device.pinned) {
                        if (b >= range.first && e <= range.second) {
                            src_dev = moe_hosted_device_ptr((void *) job.source[0]);
                            break;
                        }
                    }
                }
                if (src_dev) {
                    layer->h_fetch[1 + 3 * n_fetch] = (uint64_t) (uintptr_t) src_dev;
                    layer->h_fetch[2 + 3 * n_fetch] = entry;
                    layer->h_fetch[3 + 3 * n_fetch] = job.bytes;
                    n_fetch++;
                    st->gpu_fills.push_back(job);
                    // out of the fill queue's accounting, as the fill thread does when it takes a job
                    device.queued_bytes = job.bytes <= device.queued_bytes ? device.queued_bytes - job.bytes : 0;
                } else {
                    moved.push_back(job);
                }
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
            layer->h_fetch[0] = (uint64_t) n_fetch;
            if (!moved_experts.empty()) {
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
                st->split_tensors = (packed ? 1 : MOE_ROLES) * (int) moved_experts.size();
                session.cv.notify_all();
            }
            // EXPERT SPLIT: the other cards as destinations, one expert at a time while one still ends the window sooner.
            // A card's term: its resident rows, then the moved experts' copies over its own link one after another, their
            // rows and the crossing of its rows through host memory (the same model's terms as card 0's)
            if (!st->split_cards.empty() && layer->h_plan2k && layer->h_src && n_cpu > 0) {
                const int ncard = (int) st->split_cards.size();
                std::vector<bool> taken(c.size(), false);
                for (int b : best) {
                    taken[b] = true;
                }
                std::vector<int> krows(ncard + 1, 0), kmoved(ncard + 1, 0);
                std::vector<double> karr(ncard + 1, 0.0), kend(ncard + 1, 0.0);
                for (int row = 0; row < rows; row++) {
                    const int src = layer->h_src[row];
                    if (src > 0 && src <= ncard) {
                        krows[src]++;
                    }
                }
                const double cross_us = 10.0;   // a card's rows through host memory and its flag (tests/mgproto: ~18 us a layer)
                const auto card_end = [&](int k, int moved_rows, double arrival) {
                    // the other cards scaled as this one (the same kind of card, as in the setups tested)
                    const double before = krows[k] ? gs * (moe_split_gpu_before_us + moe_split_gpu_row_us * krows[k]) : 0.0;
                    if (moved_rows == 0) {
                        return krows[k] ? before + cross_us : 0.0;
                    }
                    return std::max(before, arrival) + gs * moe_split_gpu_moved_row_us * moved_rows + 2 * cross_us;
                };
                for (int k = 1; k <= ncard; k++) {
                    kend[k] = card_end(k, 0, 0.0);
                }
                int moved0_rows = 0, moved0_tensors = 0;
                double moved0_copies = 0.0;
                for (int b : best) {
                    if (std::find(moved_experts.begin(), moved_experts.end(), c[b].e) != moved_experts.end()) {
                        moved0_rows += c[b].rows;
                        moved0_tensors += c[b].tensors;
                        moved0_copies += c[b].copies;
                    }
                }
                const double arrival0 = moved0_tensors == 0 ? 0.0 : by_kernel && packed ? gpu_before + moved0_copies :
                                        moe_split_copy_start_us + moved0_copies;
                const double gpu0 = std::max(gpu_before, arrival0) + gs * (moe_split_gpu_after_us + moe_split_gpu_moved_row_us * moved0_rows);
                int d_left = 0;
                for (size_t i = 0; i < c.size(); i++) {
                    d_left += !taken[i];
                }
                int tensors_all = moved0_tensors;
                const auto window_now = [&](int d, int r, int t, int kk, double kk_end) {
                    double w = std::max(cpu_time(d, r, t), gpu0);
                    for (int k = 1; k <= ncard; k++) {
                        w = std::max(w, k == kk ? kk_end : kend[k]);
                    }
                    return w;
                };
                double W = window_now(d_left, n_cpu, tensors_all, 0, 0.0);
                for (;;) {
                    int bi = -1, bk = -1;
                    double bW = W - 1.0, b_end = 0.0, b_arr = 0.0;
                    for (size_t i = 0; i < c.size(); i++) {
                        if (taken[i] || c[i].slot >= 0) {
                            continue;
                        }
                        for (int k = 1; k <= ncard; k++) {
                            moe_cache_device & dk = *st->split_cards[k - 1].device;
                            const int pik = moe_cache_pool_of(dk, key);
                            if (pik < 0 || pik >= (int) dk.pools.size() || !dk.pools[pik]->slab ||
                                    dk.pools[pik]->map.count(moe_cache_key{key, c[i].e})) {
                                continue;
                            }
                            const double arr = karr[k] + moe_split_copy_start_us + dk.pools[pik]->expert_size / st->split_cards[k - 1].rate;
                            const double end = card_end(k, kmoved[k] + c[i].rows, arr);
                            const double w = window_now(d_left - 1, n_cpu - c[i].rows, tensors_all + c[i].tensors, k, end);
                            if (w < bW) {
                                bW = w; bi = (int) i; bk = k; b_end = end; b_arr = arr;
                            }
                        }
                    }
                    if (bi < 0) {
                        break;
                    }
                    taken[bi] = true;
                    // the admission to card bk, its copy first in that card's fill queue, its rows to that card
                    moe_split_card & card = st->split_cards[bk - 1];
                    moe_cache_device & dk = *card.device;
                    const int pik = moe_cache_pool_of(dk, key);
                    moe_cache_pool & pk = *dk.pools[pik];
                    int inserts_left = 1;
                    bool wake = false;
                    uint16_t freq = 0;
                    if (session.config.lfu) {
                        const auto d = device.demand_count.find(moe_cache_key{key, c[bi].e});
                        freq = d != device.demand_count.end() ? d->second.count : 0;
                    }
                    const bool deferring = dk.deferring;
                    dk.deferring = false;
                    const bool ok = moe_cache_enqueue_locked(session, dk, pk, pik, key, c[bi].e, pk.expert_size, inserts_left,
                                                             wake, true, false, false, freq);
                    dk.deferring = deferring;
                    if (!ok) {
                        continue;   // no entry for it there: it stays on the CPU rows
                    }
                    moe_cache_job job = dk.queue.back();
                    dk.queue.pop_back();
                    job.split_flag = layer->h_flags + 34 + 8 * card.index;
                    job.split_pending = &layer->split_pending_k[card.index];
                    job.split_seq = seq;
                    __atomic_add_fetch(&layer->split_pending_k[card.index], 1, __ATOMIC_ACQ_REL);
                    dk.queue.push_front(job);
                    pk.slots[job.slot].readers++;
                    st->pins.push_back({&pk, job.slot});
                    const uint64_t entry = moe_cache_entry_addr(pk, job.slot);
                    uint64_t * plan2k = layer->h_plan2k + (size_t) (card.index - 1) * 3 * rows;
                    int left = 0;
                    for (int j = 0; j < n_cpu; j++) {
                        const int row = cpu_rows[j];
                        if (layer->h_ids[row] == c[bi].e) {
                            for (int r = 0; r < MOE_ROLES; r++) {
                                plan2k[r * rows + row] = entry + pk.role_offset[r];
                            }
                            layer->h_src[row] = (int8_t) card.index;
                        } else {
                            cpu_rows[left++] = row;
                        }
                    }
                    n_cpu = left;
                    d_left--;
                    tensors_all += c[bi].tensors;
                    karr[bk] = b_arr;
                    kmoved[bk] += c[bi].rows;
                    kend[bk] = b_end;
                    W = bW;
                    device.split_moved_away++;
                    session.cv.notify_all();
                }
            }
        }
    }
    if (jobs == 0) {
        __atomic_store_n(&layer->h_flags[24], seq, __ATOMIC_RELEASE);
    }
    moe_split_cards_release(st, layer, seq);
    return n_cpu;
}

// SPLIT: the dispatcher's GPU terms were fitted on a 7900 XT (moe_split_gpu_*). Once, outside any graph capture, the
// resident rows' kernels (up + gate fused, then down) of a verification pass (3 tokens, 28 rows) run on the card's own
// cache: distinct entries every call, as generation reads them (cold), timed by events after a warm-up; their time over
// the same bench's on the 7900 XT (moe_split_gpu_bench_ref_us) scales every GPU term
static constexpr double moe_split_gpu_bench_ref_us = 147.2;   // the 7900 XT (ROCm 7.2.4): 147.7 and 146.7 us on two starts
static void moe_split_gpu_bench(moe_hosted_state * st, moe_cache_device & device, const moe_hosted_layer * layer,
                                ggml_type wtype, ggml_type dtype, cudaStream_t stream) {
    st->gpu_benched = true;
    if (const char * fixed = getenv("GGML_CUDA_MOE_SPLIT_GPU_SCALE")) {
        st->gpu_scale = atof(fixed);
        return;
    }
    const int R = 28, tokens = 3, iters = 40, warm = 20;
    std::vector<uint64_t> h_plan;
    {
        std::lock_guard<std::mutex> lock(st->session->mu);
        const moe_cache_pool & pool = *device.pools[layer->pool_index];
        const int n_entries = pool.band_slots ? pool.band_slots : pool.ranges[0].count;
        if (n_entries < R) {
            return;
        }
        h_plan.resize((size_t) (warm + iters) * 3 * R);
        for (int it = 0; it < warm + iters; it++) {
            for (int r = 0; r < R; r++) {
                const int slot = (int) (((int64_t) it * R + r) % n_entries);
                const uint64_t e = moe_cache_entry_addr(pool, slot);
                for (int k = 0; k < MOE_ROLES; k++) {
                    h_plan[((size_t) it * 3 + k) * R + r] = e + pool.role_offset[k];
                }
            }
        }
    }
    const int64_t n_embd = layer->n_embd, n_ff = layer->n_ff;
    const int64_t padded_embd = GGML_PAD(n_embd, MATRIX_ROW_PADDING), padded_ff = GGML_PAD(n_ff, MATRIX_ROW_PADDING);
    uint64_t * d_plan = nullptr; int32_t * d_ids = nullptr; char * d_act = nullptr, * d_mid = nullptr;
    float * d_glu = nullptr, * d_out = nullptr; unsigned * d_count = nullptr;
    // on the block's own stream (not capturing here): a stream of its own kept a 7th hardware queue for good (ROCm keeps
    // a destroyed stream's queue), 38 MiB of context save area and one more queue for the scheduler
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    const size_t mid_blocks = (size_t) R * (padded_ff / QK8_1);
    bool ok = cudaEventCreateWithFlags(&e0, 0) == cudaSuccess && cudaEventCreateWithFlags(&e1, 0) == cudaSuccess &&
              cudaMalloc((void **) &d_plan, h_plan.size() * sizeof(uint64_t)) == cudaSuccess &&
              cudaMalloc((void **) &d_ids, R * sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc((void **) &d_act, (size_t) tokens * (padded_embd / QK8_1) * sizeof(block_q8_1)) == cudaSuccess &&
              cudaMalloc((void **) &d_mid, mid_blocks * sizeof(block_q8_1)) == cudaSuccess &&
              cudaMalloc((void **) &d_count, mid_blocks * sizeof(unsigned)) == cudaSuccess &&
              cudaMalloc((void **) &d_glu, (size_t) R * n_ff * sizeof(float)) == cudaSuccess &&
              cudaMalloc((void **) &d_out, (size_t) R * n_embd * sizeof(float)) == cudaSuccess;
    if (ok) {
        std::vector<int32_t> ids(R);
        for (int r = 0; r < R; r++) {
            ids[r] = r * tokens / R;
        }
        ok = cudaMemcpyAsync(d_plan, h_plan.data(), h_plan.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemcpyAsync(d_ids, ids.data(), R * sizeof(int32_t), cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemsetAsync(d_act, 0, (size_t) tokens * (padded_embd / QK8_1) * sizeof(block_q8_1), stream) == cudaSuccess &&
             cudaMemsetAsync(d_count, 0, mid_blocks * sizeof(unsigned), stream) == cudaSuccess;
    }
    double us = 0.0;
    if (ok) {
        const auto call = [&](int it) {
            const uint64_t * plan = d_plan + (size_t) it * 3 * R;
            ggml_cuda_moe_cache_mmv_fused(wtype, d_act, plan, plan + R, d_ids, d_glu, n_embd, n_ff, R, tokens,
                                          -INFINITY, INFINITY, -INFINITY, INFINITY, stream, d_mid, padded_ff / QK8_1, d_count,
                                          nullptr, tokens);
            ggml_cuda_moe_cache_mmv(dtype, d_mid, plan + 2 * R, nullptr, d_out, n_ff, n_embd, R, R, stream, nullptr, tokens);
        };
        for (int it = 0; it < warm; it++) {
            call(it);
        }
        cudaEventRecord(e0, stream);
        for (int it = warm; it < warm + iters; it++) {
            call(it);
        }
        cudaEventRecord(e1, stream);
        float ms = 0.0f;
        ok = cudaEventSynchronize(e1) == cudaSuccess && cudaEventElapsedTime(&ms, e0, e1) == cudaSuccess;
        us = 1e3 * ms / iters;
    }
    (void) cudaGetLastError();
    cudaStreamSynchronize(stream);
    cudaFree(d_plan); cudaFree(d_ids); cudaFree(d_act); cudaFree(d_mid); cudaFree(d_count); cudaFree(d_glu); cudaFree(d_out);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (!ok || us <= 0.0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d split: GPU bench failed, the model's GPU terms kept\n", device.physical);
        return;
    }
    if (moe_split_gpu_bench_ref_us > 0.0) {
        st->gpu_scale = us / moe_split_gpu_bench_ref_us;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d split: GPU bench %.1f us for %d resident rows (3 tokens), %.2f x the model's GPU terms\n",
                  device.physical, us, R, st->gpu_scale);
}

// SPLIT: one timed layer of the calibration (moe_split_cal_skip): its CPU time from the plan to the CPU rows' flag (what
// the model was fitted on) against the model's for the rows left; at the end the scale is fixed and logged
// (ids: the CPU rows' experts, copied before the flag; the layers' shared h_ids may already hold the next layer's)
static void moe_split_calibrate(moe_hosted_state * st, const int32_t * ids, int n_cpu, int64_t us) {
    if (--st->cal_left >= moe_split_cal_layers) {
        return;
    }
    int d = 0;
    for (int j = 0; j < n_cpu; j++) {
        bool seen = false;
        for (int i = 0; i < j && !seen; i++) {
            seen = ids[i] == ids[j];
        }
        d += !seen;
    }
    st->cal_measured += (double) us;
    st->cal_model += moe_split_cpu_model_us(st->n_threads, d, n_cpu, st->split_tensors);
    if (st->cal_left == 0 && st->cal_model > 0.0) {
        st->cpu_scale = st->cal_measured / st->cal_model;
        MOE_CACHE_LOG("[moe-cache] CUDA%d split: CPU rows take %.2f x the model's time (%d layers, %.1f us per layer)\n",
                      st->device->physical, st->cpu_scale.load(), moe_split_cal_layers, st->cal_measured / moe_split_cal_layers);
    }
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
    // EXPERT SPLIT: each other card's chain for the layer p, against the hosting card's publication and merge (two clocks:
    // %globaltimer on CUDA, assumed close; the durations within a card need no assumption)
    for (const moe_split_card & card : st->split_cards) {
        const volatile unsigned long long * c = g_tlk_h[card.index] ? g_tlk_h[card.index] + (p % MOE_TL_RING) * 8 : nullptr;
        if (!c || c[6] != p || c[0] == 0 || c[5] < c[0]) {
            continue;
        }
        double * a = st->tlk_sum[card.index];
        a[0] += 1;
        a[1] += ((double) c[1] - (double) ge1) * k;   // publication -> plan seen there
        a[2] += ((double) c[2] - (double) c[1]) * k;  // plan and activations copied
        a[3] += ((double) c[3] - (double) c[2]) * k;  // its rows (quantize, up+gate, down)
        a[4] += ((double) c[4] - (double) c[3]) * k;  // the moved experts' wait
        a[5] += ((double) c[5] - (double) c[4]) * k;  // their rows (and the weighted sum)
        a[6] += ((double) c[5] - (double) c[1]) * k;  // plan seen -> done
        a[7] += gp[4] ? ((double) gp[2] - (double) c[5]) * k : 0;   // done -> merge passed on the hosting card
        a[8] += gp[4] ? ((double) gp[4] - (double) ge2) * k : 0;    // hosting card: plan seen -> merge reached
        a[9] += ((double) c[1] - (double) c[0]) > 0 ? ((double) c[1] - (double) c[0]) * k : 0;   // idle before the plan
    }
    if (st->tl_n % every == 0) {
        for (const moe_split_card & card : st->split_cards) {
            const double * a = st->tlk_sum[card.index];
            if (a[0] > 0) {
                const double m = a[0] * 1e3;
                GGML_LOG_INFO("moe timeline: card %d chain, us per layer (%lld): publication -> plan seen %.1f; copies %.1f; "
                        "its rows %.1f; moved experts' wait %.1f, their rows %.1f; plan seen -> done %.1f; done -> merge passed "
                        "%.1f (hosting card plan seen -> merge reached %.1f); idle before the plan %.1f\n", card.index,
                        (long long) a[0], a[1] / m, a[2] / m, a[3] / m, a[4] / m, a[5] / m, a[6] / m, a[7] / m, a[8] / m, a[9] / m);
            }
        }
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

// the activations to the host as F32 too (GGML_CUDA_MOE_HOSTED_F32=1), not only as q8_0 blocks: ~30 KB more over PCIe per
// layer of 3 tokens, which the system fence before the publication waits for
static bool moe_hosted_act_f32() {
    static const bool on = getenv("GGML_CUDA_MOE_HOSTED_F32") && atoi(getenv("GGML_CUDA_MOE_HOSTED_F32")) != 0;
    return on;
}

// GGML_CUDA_MOE_TAIL_SKIP=R (Strata's STRATA_ROUTE_TAIL_SKIP, the same rule; default 7 as Strata, 0 turns it off; changes
// the output): a missed expert that every token of the pass routes only at rank R or lower (its position in the top-k, by
// weight; R from 1 to 9) is neither moved to the GPU nor computed: its rows are zero, the other experts keep the weights the
// router gave them (no renormalisation). The main model's layers only, as Strata (not the MTP draft's).
// Returns the CPU rows kept; counted as Strata reports them (missed experts skipped, rows skipped).
static int moe_hosted_tail_skip(moe_hosted_state & st, moe_hosted_layer & layer, int rows, int32_t * cpu_rows, int n_cpu) {
    static const int rank = [] {
        const char * v = getenv("GGML_CUDA_MOE_TAIL_SKIP");
        const int r = v ? atoi(v) : 7;
        return r > 0 && r < 10 ? r : 0;
    }();
    if (rank == 0 || n_cpu == 0) {
        return n_cpu;
    }
    const moe_cache_device & device = *st.device;
    if (layer.layer_number < device.main_first || layer.layer_number > device.main_last) {
        return n_cpu;
    }
    static std::atomic<int64_t> n_experts{0}, n_rows{0};
    const int32_t * ids = layer.h_ids;
    const int n_used = layer.n_used;
    // the missed experts (once each, by their first CPU row) routed only at rank R or lower
    int32_t skip[moe_hosted_rows_max];
    int n_skip = 0;
    for (int j = 0; j < n_cpu; j++) {
        const int32_t e = ids[cpu_rows[j]];
        bool seen = false, tail = true;
        for (int q = 0; q < rows; q++) {
            if (ids[q] == e) {
                seen = seen || q < cpu_rows[j];
                tail = tail && q % n_used >= rank;
            }
        }
        if (!seen && tail) {
            skip[n_skip++] = e;
        }
    }
    n_experts += n_skip;
    int keep = 0;
    for (int j = 0; j < n_cpu; j++) {
        const int r = cpu_rows[j];
        if (std::find(skip, skip + n_skip, ids[r]) == skip + n_skip) {
            cpu_rows[keep++] = r;
            continue;
        }
        memset(layer.h_res + (size_t) r * layer.n_embd, 0, layer.n_embd * sizeof(float));
        n_rows++;
    }
    static std::atomic<int64_t> next_log{20000};
    if (n_rows >= next_log) {
        next_log += 20000;
        GGML_LOG_INFO("[moe-cache] tail skip (rank >= %d): %lld missed experts skipped (%lld rows) since the start\n",
                      rank, (long long) n_experts.load(), (long long) n_rows.load());
    }
    return keep;
}

static void moe_hosted_serve_one(moe_hosted_state * st, uint32_t seq) {
    if (st->h_split_diag && !st->split_diag_logged && __atomic_load_n(&st->h_split_diag[0], __ATOMIC_ACQUIRE)) {
        st->split_diag_logged = true;
        const uint32_t * d = st->h_split_diag;
        GGML_LOG_ERROR("[moe-cache] EXPERT SPLIT: a wait ran out of time: %s, card %u, %s %u, held %u, saw %u (serving publication %u)\n",
                       d[0] == 1 ? "a card's wait for its plan" : "the merge's wait for a card", d[1],
                       d[0] == 1 ? "layer slot" : "row", d[2], d[3], d[4], seq);
    }
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
    n_cpu = moe_hosted_tail_skip(*st, *layer, rows, cpu_rows.data(), n_cpu);
    const int64_t t_cal = st->cal_left > 0 && n_tokens >= 2 && moe_split_enabled() ? ggml_time_us() : 0;
    int64_t t_reads = 0;   // ARENA: the CPU rows' reads of the file, out of the calibration (the model times the compute)
    n_cpu = moe_hosted_split(st, layer, rows, n_tokens, seq, cpu_rows.data(), n_cpu);
    const int64_t t_plan = stats_every > 0 ? ggml_time_us() : 0;

    // the CPU rows in one call, one column each (k = 1) with its token's activations: exactly the
    // missed rows (a k-per-token layout padded the tokens with fewer misses: 8.1 rows computed for
    // 5.9, CPU time +12 % per layer); MUL_MAT_ID groups the rows of an expert, which is read once
    if (n_cpu > 256) {
        std::stable_sort(cpu_rows.begin(), cpu_rows.begin() + n_cpu,
                         [&](int32_t a, int32_t b) { return layer->h_ids[a] < layer->h_ids[b]; });
    }
    if (n_cpu > 0) {
        ids.resize(n_cpu);
        act.resize((size_t) n_cpu * layer->n_embd);
        const size_t q8_row = (size_t) (layer->n_embd / QK8_0) * sizeof(block_q8_0);
        st->act8.resize((size_t) n_cpu * q8_row);
        for (int j = 0; j < n_cpu; j++) {
            const int t = cpu_rows[j] / layer->n_used;
            ids[j] = layer->h_ids[cpu_rows[j]];
            memcpy(st->act8.data() + (size_t) j * q8_row, (const char *) layer->h_act8 + (size_t) t * q8_row, q8_row);
            if (moe_hosted_act_f32()) {
                memcpy(act.data() + (size_t) j * layer->n_embd, layer->h_act + (size_t) t * layer->n_embd, layer->n_embd * sizeof(float));
            }
        }
        out.resize((size_t) n_cpu * layer->n_embd);
        g_session_suppressed++;
        // ARENA: the absent experts the arena holds are read there; the others are read into slots with O_DIRECT first (in
        // parallel), then computed from them (from the weights when no slot is free). Slots held meanwhile
        std::vector<const void *> & src = st->arena_src;
        std::vector<int> & held = st->arena_held;
        moe_expert_arena * har = nullptr;   // the arena of held
        src.clear();
        held.clear();
        if (!st->device->arenas.empty()) {
            moe_cache_device & dev = *st->device;
            const void * key = layer->gate_w->data;
            struct load { int slot; int32_t e; moe_cache_job job; };
            std::vector<load> loads;
            std::vector<std::pair<int32_t, int>> slot_of;   // distinct expert -> its slot
            const moe_cache_pool * pp = nullptr;
            {
                std::lock_guard<std::mutex> lock(st->session->mu);
                const moe_cache_layer * cl = moe_cache_layer_of(dev, key);
                const int pi = cl && cl->shape >= 0 ? dev.shapes[cl->shape].pool : -1;
                har = moe_arena_of(dev, cl);
                if (har && pi >= 0 && pi < (int) dev.pools.size()) {
                    pp = dev.pools[pi].get();
                    for (int j = 0; j < n_cpu; j++) {
                        const int32_t e = ids[j];
                        bool seen = false;
                        for (const auto & x : slot_of) {
                            seen = seen || x.first == e;
                        }
                        if (seen) {
                            continue;
                        }
                        const int a = har->acquire(moe_cache_key{key, e});
                        if (a >= 0) {
                            held.push_back(a);
                            slot_of.push_back({e, a});
                            har->cpu_hits++;
                            continue;
                        }
                        moe_cache_job jb;
                        moe_cache_job_sources(*cl, *pp, e, jb);
                        if (!moe_arena_wanted(dev, har, jb, *pp)) {
                            continue;
                        }
                        const int t = har->take(moe_cache_key{key, e}, pi);
                        if (t >= 0) {
                            loads.push_back({t, e, jb});
                        }
                    }
                }
            }
            if (!loads.empty()) {
                std::vector<char> ok(loads.size(), 0);
                std::vector<std::function<void()>> tasks;
                for (size_t i = 0; i < loads.size(); i++) {
                    tasks.push_back([&, i] { ok[i] = moe_arena_load(dev, *har, loads[i].slot, loads[i].job, *pp); });
                }
                const int64_t t_read = t_cal > 0 ? ggml_time_us() : 0;
                dev.arena_readers->run(tasks);
                if (t_cal > 0) {
                    t_reads += ggml_time_us() - t_read;
                }
                std::lock_guard<std::mutex> lock(st->session->mu);
                for (size_t i = 0; i < loads.size(); i++) {
                    if (ok[i]) {
                        har->commit(loads[i].slot);
                        const int a = har->acquire(moe_cache_key{key, loads[i].e});
                        held.push_back(a);
                        slot_of.push_back({loads[i].e, a});
                        har->cpu_loads++;
                    } else {
                        har->abort(loads[i].slot);
                        har->load_failures++;
                    }
                }
            }
            if (!slot_of.empty()) {
                src.assign((size_t) 3 * n_cpu, nullptr);
                for (int j = 0; j < n_cpu; j++) {
                    for (const auto & x : slot_of) {
                        if (x.first == ids[j]) {
                            const auto & sl = har->slots[x.second];
                            src[3 * j + 0] = sl.part[MOE_ROLE_GATE];
                            src[3 * j + 1] = sl.part[MOE_ROLE_UP];
                            src[3 * j + 2] = sl.part[MOE_ROLE_DOWN];
                            break;
                        }
                    }
                }
            }
        }
        // without the F32 rows, the generic path of cpu_experts takes them from the q8_0 blocks; a prompt's block (more
        // than 256 rows, the multi-row kernels' limit) in calls of 256 rows, its rows sorted by expert first
        const int call_rows = 256;
        for (int j0 = 0; j0 < n_cpu; j0 += call_rows) {
            const int n = std::min(call_rows, n_cpu - j0);
            ggml_moe_cache.cpu_experts(layer->gate_w, layer->up_w, layer->down_w,
                                       moe_hosted_act_f32() ? act.data() + (size_t) j0 * layer->n_embd : nullptr,
                                       st->act8.data() + (size_t) j0 * q8_row, ids.data() + j0,
                                       src.empty() ? nullptr : src.data() + (size_t) 3 * j0, 1, n,
                                       out.data() + (size_t) j0 * layer->n_embd, st->n_threads);
        }
        for (const int a : held) {
            har->release(a);
        }
        g_session_suppressed--;
        for (int j = 0; j < n_cpu; j++) {
            memcpy(layer->h_res + (size_t) cpu_rows[j] * layer->n_embd, out.data() + (size_t) j * layer->n_embd,
                   layer->n_embd * sizeof(float));
        }
    }
    __atomic_store_n(&layer->h_flags[16], seq, __ATOMIC_RELEASE);
    if (t_cal > 0 && n_cpu > 0) {
        moe_split_calibrate(st, ids.data(), n_cpu, ggml_time_us() - t_cal - t_reads);
    }
    moe_hosted_admit_deferred(*st, *layer);
    if (!st->tl_host.empty()) {
        moe_tl_account(st, seq);
    }
{
        // DIAGNOSTIC (GGML_CUDA_MOE_SPLIT_VERIFY=1): the moved rows, CPU against GPU (relative L2 error per row); read after
        // the flag, the shared h_ids / h_plan2 / h_res may already hold the next layer's: a mismatch can be that
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

bool ggml_cuda_moe_hosted_active() {
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    return !g_hosted.empty();
}

// Wait for the stream (or the event) like cudaStreamSynchronize / cudaEventSynchronize, answering the
// hosted MoE blocks it publishes meanwhile: the thread that waits for the GPU computes the missed
// experts. Returns false when no block is hosted (the caller synchronizes as usual).
// GGML_CUDA_MOE_HOSTED_SERVER=1: the hosted blocks of every device are served by one thread, with one OpenMP team for the
// missing experts, while any thread waits for a GPU; the waiting threads only poll their stream. Without it each waiting
// thread serves: with the layers split in pipeline stages, each stage's thread served its device's blocks with its own
// OpenMP team, and the two teams, pinned to the same cores, spun while the other computed
static bool moe_hosted_server_on() {
    static const bool on = getenv("GGML_CUDA_MOE_HOSTED_SERVER") && atoi(getenv("GGML_CUDA_MOE_HOSTED_SERVER")) != 0;
    return on;
}
static std::atomic<int>        g_srv_waiters{0};
static std::mutex              g_srv_mu;
static std::condition_variable g_srv_cv;

static void moe_hosted_server_loop() {
    constexpr int max_states = 8;
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(g_srv_mu);
            g_srv_cv.wait(lock, [] { return g_srv_waiters.load(std::memory_order_acquire) > 0; });
        }
        moe_hosted_state * sts[max_states];
        int n_st = 0;
        {
            std::lock_guard<std::mutex> lock(g_hosted_mu);
            for (auto & entry : g_hosted) {
                if (entry.second && n_st < max_states) {
                    sts[n_st++] = entry.second.get();
                }
            }
        }
        std::unique_lock<std::mutex> serving[max_states];
        for (int k = 0; k < n_st; k++) {
            serving[k] = std::unique_lock<std::mutex>(sts[k]->serve_mu, std::try_to_lock);
        }
        while (g_srv_waiters.load(std::memory_order_acquire) > 0) {
            bool served = false;
            for (int k = 0; k < n_st; k++) {
                moe_hosted_state * s = sts[k];
                const uint32_t seq = __atomic_load_n(&s->h_mail->seq, __ATOMIC_ACQUIRE);
                if (serving[k].owns_lock() && seq != s->last_seq) {
                    s->last_seq = seq;
                    moe_hosted_serve_one(s, seq);
                    served = true;
                } else if (!serving[k].owns_lock()) {
                    serving[k].try_lock();
                }
            }
            if (!served) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
        }
    }
}

bool ggml_cuda_moe_hosted_wait(void * stream_ptr, void * event_ptr, int device) {
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    cudaEvent_t event = (cudaEvent_t) event_ptr;
    // the rows a launched graph waits for, before anything else (and without a hosted MoE: the wait would never end)
    moe_host_rows_run();
    if (moe_hosted_server_on()) {
        {
            std::lock_guard<std::mutex> lock(g_hosted_mu);
            if (g_hosted.empty()) {
                return false;
            }
        }
        static std::once_flag started;
        std::call_once(started, [] { std::thread(moe_hosted_server_loop).detach(); });
        {
            std::lock_guard<std::mutex> lock(g_srv_mu);
            g_srv_waiters.fetch_add(1, std::memory_order_acq_rel);
        }
        g_srv_cv.notify_one();
        for (;;) {
            const cudaError_t done = event ? cudaEventQuery(event) : cudaStreamQuery(stream);
            if (done != cudaErrorNotReady) {
                g_srv_waiters.fetch_sub(1, std::memory_order_acq_rel);
                CUDA_CHECK(done);
                if (!event) {
                    // as below: the stream's device's rows, once its stream is done
                    std::lock_guard<std::mutex> lock(g_hosted_mu);
                    for (auto & entry : g_hosted) {
                        moe_hosted_state * s = entry.second.get();
                        if (!s || (g_hosted.size() > 1 && s->device->logical != device)) {
                            continue;
                        }
                        std::lock_guard<std::mutex> lock_s(s->session->mu);
                        moe_hosted_unpin_locked(*s);
                        moe_cache_prefill_unpin_locked(*s->session);
                    }
                }
                return true;
            }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
    // one state per (session, device): every mailbox is served while waiting, whichever device the stream is on (with the
    // layers split over several GPUs, a block of one device may wait for its CPU rows while the host waits for another)
    constexpr int max_states = 8;
    moe_hosted_state * sts[max_states];
    int n_st = 0;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        if (g_hosted.empty()) {
            return false;
        }
        for (auto & entry : g_hosted) {
            if (entry.second && n_st < max_states) {
                sts[n_st++] = entry.second.get();
            }
        }
    }
    std::unique_lock<std::mutex> serving[max_states];
    for (int k = 0; k < n_st; k++) {
        serving[k] = std::unique_lock<std::mutex>(sts[k]->serve_mu, std::try_to_lock);
    }
    moe_hosted_state * st = sts[0];
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
                // zones to the next prefill, which then ran without prefetch). Only the stream's device's: another
                // device may still be computing with its pinned rows
                for (int k = 0; k < n_st; k++) {
                    if (n_st > 1 && sts[k]->device->logical != device) {
                        continue;
                    }
                    std::lock_guard<std::mutex> lock(sts[k]->session->mu);
                    moe_hosted_unpin_locked(*sts[k]);
                    moe_cache_prefill_unpin_locked(*sts[k]->session);
                }
            }
            return true;
        }
        bool served = false;
        for (int k = 0; k < n_st; k++) {
            moe_hosted_state * s = sts[k];
            const uint32_t seq = __atomic_load_n(&s->h_mail->seq, __ATOMIC_ACQUIRE);
            if (serving[k].owns_lock() && seq != s->last_seq) {
                s->last_seq = seq;
                moe_hosted_serve_one(s, seq);
                served = true;
            }
        }
        if (served) {
            continue;
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#endif
        for (int k = 0; k < n_st; k++) {
            if (!serving[k].owns_lock()) {
                serving[k].try_lock();
            }
        }
    }
}

// SPLIT: the host -> device bytes per microsecond a copy of an expert gets. The link's rate is the smallest of the maximum
// link capabilities on the path from the device to the root port: the current ones drop while the link idles, and a
// Radeon's own link to the switch inside the card reads 16 GT/s x16 in any slot. Times the encoding (8b/10b up to
// 5 GT/s, 128b/130b above), times the share a copy gets (moe_cache_pcie_copy_efficiency). 0 if the path cannot be read.
static double moe_cache_pcie_bytes_per_us(int physical, std::string & desc) {
#if !defined(__linux__)
    GGML_UNUSED(physical);
    GGML_UNUSED(desc);
    return 0.0;
#else
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
#endif
}

// the copy rate of a card's link, bytes per microsecond: the preload's (GiB of experts copied from host memory at start,
// measured once, already done: the link as it is, a slot or a host that halves it included), else the link's
// capabilities (moe_cache_pcie_bytes_per_us) when they are known and the measure is not well below them. GGML_CUDA_MOE_PCIE_MEASURED=0: the
// capabilities only. 0 when neither is known
static double moe_cache_link_rate(const moe_cache_device & device, std::string & desc) {
    const double caps = moe_cache_pcie_bytes_per_us(device.physical, desc);
    static const bool use_measured = !getenv("GGML_CUDA_MOE_PCIE_MEASURED") || atoi(getenv("GGML_CUDA_MOE_PCIE_MEASURED")) != 0;
    if (!use_measured || device.preload_bytes < ((size_t) 1 << 30) || device.preload_us <= 0) {
        return caps;
    }
    // the measured rate replaces the capabilities' only when well below them (a link that does not give what it reads,
    // 11 - 12 GB/s on a 16 GT/s x16 path read as 27.2 on a rented host): the dispatcher's terms were fitted on the
    // capabilities' rate of a link that gives it (13.6 GB/s, PCIe 3.0 x16 at 86 %), which a preload measures within ~10 %
    const double measured = (double) device.preload_bytes / (double) device.preload_us;
    const bool take = caps <= 0.0 || measured < 0.75 * caps;
    char text[192];
    snprintf(text, sizeof(text), "%s%s%.2f GB/s measured by the preload (%.2f GiB), %s", desc.c_str(), desc.empty() ? "" : "; ",
             measured / 1000.0, device.preload_bytes / 1073741824.0, take ? "taken" : "the capabilities' kept");
    desc = text;
    return take ? measured : caps;
}

// the CPU rows' team: one thread per physical core the process may run on (its affinity mask, SMT siblings counted
// once); elsewhere, or when the topology is not readable, half the hardware threads
static int moe_hosted_default_threads() {
    const int hw = std::max(1, (int) std::thread::hardware_concurrency());
#if defined(__linux__)
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        std::set<std::string> cores;
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
            if (!CPU_ISSET(cpu, &mask)) {
                continue;
            }
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
            std::string siblings;
            if (!std::getline(f, siblings)) {
                return std::max(1, hw / 2);
            }
            cores.insert(siblings);
        }
        if (!cores.empty()) {
            // a container's CPU quota (cgroup v2 cpu.max: "quota period", or "max"; cgroup v1 cpu.cfs_quota_us and
            // cpu.cfs_period_us, -1 for none): rented GPU machines showed 48 hardware threads, 24 cores, for 16 and
            // 20.4 vCPU quotas, and a team of 24 spinning threads on them (MTP 70 t/s, 81 with 16 threads)
            int n = (int) cores.size();
            long long quota = -1, period = 0;
            std::ifstream q2("/sys/fs/cgroup/cpu.max");
            std::string q2s;
            if (q2 >> q2s >> period && q2s != "max") {
                quota = std::atoll(q2s.c_str());
            } else {
                std::ifstream q1("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), p1("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
                if (!(q1 >> quota && p1 >> period)) {
                    quota = -1;
                }
            }
            if (quota > 0 && period > 0) {
                n = std::min(n, std::max(1, (int) (quota / period)));
            }
            return n;
        }
    }
#endif
    return std::max(1, hw / 2);
}

// serves the mailbox while blocks launched outside a capture are pending; asleep otherwise. The thread that waits
// for the stream serves it too (ggml_cuda_moe_hosted_wait): whoever holds serve_mu serves, the other one counts it
static void moe_hosted_server(moe_hosted_state * st) {
    ggml_cuda_set_device(st->device->logical);
    uint32_t seen;
    {
        std::lock_guard<std::mutex> lock(st->serve_mu);
        seen = st->last_seq;
    }
    for (;;) {
        {
            std::unique_lock<std::mutex> lock(st->server_mu);
            st->server_cv.wait(lock, [st] { return st->server_stop || st->server_pending > 0; });
            if (st->server_stop) {
                return;
            }
        }
        for (;;) {
            if (st->server_stop.load(std::memory_order_relaxed)) {
                return;
            }
            // no driver call here: a first cuBLAS product of a shape (more than 8 columns, in a hosted prompt block)
            // loads a cuBLASLt module (cuLibraryLoadData) that waits for the device while holding a driver lock;
            // a cudaStreamQuery each poll then waited for that lock while the device waited for this mailbox
            // (RTX 4090, 30 Oct.: a fresh server's first request hung with blocks of more than 128 rows)
            uint32_t advanced = 0;
            {
                // blocking: a thread waiting for the stream holds serve_mu and serves the mailbox meanwhile (the last
                // block of a prefill, launched with its graph, published seconds later: polled for nothing before)
                std::unique_lock<std::mutex> serving(st->serve_mu);
                const uint32_t seq = __atomic_load_n(&st->h_mail->seq, __ATOMIC_ACQUIRE);
                if (seq != st->last_seq) {
                    st->last_seq = seq;
                    moe_hosted_serve_one(st, seq);
                }
                advanced = st->last_seq - seen;
                seen = st->last_seq;
            }
            if (advanced) {
                std::lock_guard<std::mutex> lock(st->server_mu);
                st->server_pending = st->server_pending > advanced ? st->server_pending - advanced : 0;
                if (st->server_pending == 0 || st->server_stop) {
                    break;
                }
            }
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
    }
}

static moe_hosted_state * moe_hosted_state_for(moe_cache_session & session, moe_cache_device & device) {
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    auto & entry = g_hosted[{&session, device.logical}];
    if (entry) {
        return entry.get();
    }
    std::unique_ptr<moe_hosted_state> st(new moe_hosted_state());
    st->session = &session;
    st->device = &device;
    const char * threads = getenv("GGML_CUDA_MOE_HOSTED_THREADS");
    st->n_threads = threads ? std::max(1, atoi(threads)) : moe_hosted_default_threads();
    if (const char * scale = getenv("GGML_CUDA_MOE_SPLIT_CPU_SCALE")) {
        st->cpu_scale = atof(scale);
        st->cal_left = 0;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d CPU rows: %d threads\n", device.physical, st->n_threads);
    {
        std::string desc;
        const double rate = moe_cache_link_rate(device, desc);
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
#elif !defined(GGML_USE_MUSA)
        rate_khz = 1000000;   // %globaltimer counts nanoseconds
#endif
        if (rate_khz > 0 && (g_tl_h || moe_hosted_host_alloc((void **) &g_tl_h, MOE_TL_RING * 8 * sizeof(unsigned long long)))) {
            g_tl_m = moe_hosted_device_ptr(g_tl_h);
            st->tl_tick_ns = 1e6 / rate_khz;
            st->tl_host.assign(MOE_TL_RING * 3, 0);
            GGML_LOG_INFO("moe timeline: GPU wall clock %d kHz\n", rate_khz);
        }
    }
    st->server = std::thread(moe_hosted_server, st.get());
    entry = std::move(st);
    return entry.get();
}

// The scheduler hosts a MoE block when that beats its prefetch, which copies the block's expert tensors whatever the
// batch: hosted if fixed_us + rows * row_us < the tensors' bytes / copy_bytes_per_us. Fitted on the 7900 XT in PCIe 3.0
// (IQ3_KS hybrid, 48 MoE layers, 8 CPU threads, CPU scale 0.95): a prompt of 164 / 278 / 413 / 692 / 853 tokens takes
// 0.61 / 0.82 / 1.14 / 1.95 / 2.24 s hosted, ~0.235 s + 2.35 ms per token, i.e. per layer 4.9 ms + 4.9 us per routed
// row, against 3.05 s for the prefetch (its bytes / the link's copy rate, 13.6 GB/s). The row term is the CPU's (the
// GPU computes the resident rows meanwhile): it follows the CPU team (its threads, the measured scale of
// moe_split_calibrate, 1.0 until it is fixed), the fixed term does not. 7900 XT: ~12 000 rows (above the buffers' 8191);
// RTX 4090 in PCIe 4.0 (27.2 GB/s, 13 threads at 1.71): ~5 000 rows, ~500 tokens
static constexpr double moe_hosted_fit_fixed_us = 4900.0, moe_hosted_fit_row_us = 4.9, moe_hosted_fit_scale = 0.95;

static bool moe_hosted_cost(void * session_ptr, double * copy_bytes_per_us, double * fixed_us, double * row_us) {
    moe_cache_session * session = static_cast<moe_cache_session *>(session_ptr);
    if (!session || session->devices.empty() || !session->devices[0]) {
        return false;
    }
    moe_hosted_state * st = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        auto it = g_hosted.find({session, session->devices[0]->logical});
        st = it == g_hosted.end() ? nullptr : it->second.get();
    }
    if (!st) {
        return false;   // no hosted block yet: the scheduler keeps its default
    }
    *copy_bytes_per_us = st->pcie_bytes_per_us;
    *fixed_us = moe_hosted_fit_fixed_us;
    *row_us = moe_hosted_fit_row_us * (st->cpu_scale.load() / moe_hosted_fit_scale) *
              (moe_split_cpu_fit_threads / std::max(1, st->n_threads));
    return true;
}

static void moe_hosted_shutdown(moe_cache_session * session) {
    std::vector<std::unique_ptr<moe_hosted_state>> states;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        for (auto it = g_hosted.begin(); it != g_hosted.end();) {
            if (it->first.first == session) {
                states.push_back(std::move(it->second));
                it = g_hosted.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto & st : states) {
        if (!st) {
            continue;
        }
        if (st->server.joinable()) {
            {
                std::lock_guard<std::mutex> lock(st->server_mu);
                st->server_stop = true;
            }
            st->server_cv.notify_one();
            st->server.join();
        }
        for (void * h : {(void *) st->h_act, (void *) st->h_act8, (void *) st->h_res, (void *) st->h_w, (void *) st->h_ids,
                         (void *) st->h_plan, (void *) st->h_plan2}) {
            if (h) cudaFreeHost(h);
        }
        for (void * d : {(void *) st->d_plan, (void *) st->d_plan2}) {
            if (d) cudaFree(d);
        }
        for (auto & layer : st->layers) {
            for (void * h : {(void *) layer->h_flags}) {
                if (h) cudaFreeHost(h);
            }
            for (void * d : {(void *) layer->d_act_ids, (void *) layer->d_expect}) {
                if (d) cudaFree(d);
            }
        }
        if (st->h_mail) cudaFreeHost(st->h_mail);
        if (st->d_seq) cudaFree(st->d_seq);
        if (st->d_q8_act) cudaFree(st->d_q8_act);
        if (st->d_q8_mid) cudaFree(st->d_q8_mid);
        if (st->d_q8_count) cudaFree(st->d_q8_count);
        if (st->d_router_done) cudaFree(st->d_router_done);
        if (st->d_router_tagged) cudaFree(st->d_router_tagged);
        // the pins of the last plan go with the pools the session is about to free
    }
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
        // an expert's bytes (nb[2] is an entry in a packed layer)
        if (pool.role_bytes[MOE_ROLE_UP] != (size_t) (up_w->nb[1]*up_w->ne[1]) ||
            pool.role_bytes[MOE_ROLE_GATE] != (size_t) (gate_w->nb[1]*gate_w->ne[1]) ||
            pool.role_bytes[MOE_ROLE_DOWN] != (size_t) (down_w->nb[1]*down_w->ne[1])) {
            return nullptr;
        }
        layer->base = device.pools[layer->pool_index]->gpu_base;
        layer->down_base = device.pools[layer->down_pool_index]->gpu_base;
    }
    const int rows = moe_hosted_rows_max;
    const int tokens = moe_hosted_tokens_max;
    GGML_ASSERT(layer->n_embd % QK8_0 == 0);
    if (!st.h_res) {
        st.sh_embd = layer->n_embd;
        if (!moe_hosted_host_alloc((void **) &st.h_act, (size_t) tokens * layer->n_embd * sizeof(float)) ||
            !moe_hosted_host_alloc((void **) &st.h_act8, (size_t) tokens * (layer->n_embd / QK8_0) * sizeof(block_q8_0)) ||
            !moe_hosted_host_alloc((void **) &st.h_res, (size_t) rows * layer->n_embd * sizeof(float)) ||
            !moe_hosted_host_alloc((void **) &st.h_w, rows * sizeof(float)) ||
            !moe_hosted_host_alloc((void **) &st.h_ids, rows * sizeof(int32_t)) ||
            !moe_hosted_host_alloc((void **) &st.h_plan, 3 * rows * sizeof(uint64_t)) ||
            !moe_hosted_host_alloc((void **) &st.h_plan2, 3 * rows * sizeof(uint64_t)) ||
            // each plan followed by its chunk list (moe_group_list)
            cudaMalloc((void **) &st.d_plan, 3 * rows * sizeof(uint64_t) + MOE_GROUP_WORDS * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void **) &st.d_plan2, 3 * rows * sizeof(uint64_t) + MOE_GROUP_WORDS * sizeof(int32_t)) != cudaSuccess) {
            GGML_ABORT("MoE block hosted by the GPU: allocation failed");
        }
    }
    GGML_ASSERT(layer->n_embd == st.sh_embd);
    layer->h_act = st.h_act;
    layer->h_act8 = st.h_act8;
    layer->h_res = st.h_res;
    layer->h_w = st.h_w;
    layer->h_ids = st.h_ids;
    layer->h_plan = st.h_plan;
    layer->h_plan2 = st.h_plan2;
    layer->d_plan = st.d_plan;
    layer->d_plan2 = st.d_plan2;
    // EXPERT SPLIT only: the other cards' plans and each row's card (1.5 MiB a layer), read nowhere else (every use
    // checks the split or its cards first)
    const bool split = moe_cache_expert_split();
    bool ok = moe_hosted_host_alloc((void **) &layer->h_flags, 512) &&
              (!split || moe_hosted_host_alloc((void **) &layer->h_plan2k, (size_t) MOE_SPLIT_MAX_CARDS * 3 * rows * sizeof(uint64_t))) &&
              (!split || moe_hosted_host_alloc((void **) &layer->h_plank, (size_t) MOE_SPLIT_MAX_CARDS * 3 * rows * sizeof(uint64_t))) &&
              (!split || moe_hosted_host_alloc((void **) &layer->h_src, (size_t) rows)) &&
              moe_hosted_host_alloc((void **) &layer->h_pred, (2 + rows) * sizeof(int32_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_fetch, (1 + 3 * MOE_SPLIT_FETCH_MAX) * sizeof(uint64_t)) &&
              moe_hosted_host_alloc((void **) &layer->h_chk, (size_t) (getenv("GGML_CUDA_MOE_SPLIT_VERIFY") ? rows : 1) *
                                    layer->n_embd * sizeof(float));
    ok = ok && (layer->m_act = moe_hosted_device_ptr(layer->h_act)) && (layer->m_act8 = (block_q8_0 *) moe_hosted_device_ptr(layer->h_act8)) &&
         (layer->m_ids = moe_hosted_device_ptr(layer->h_ids)) &&
         (layer->m_plan = moe_hosted_device_ptr(layer->h_plan)) && (layer->m_res = moe_hosted_device_ptr(layer->h_res)) &&
         (layer->m_w = moe_hosted_device_ptr(layer->h_w)) &&
         (layer->m_flags = moe_hosted_device_ptr(layer->h_flags)) && (layer->m_pred = moe_hosted_device_ptr(layer->h_pred)) &&
         (layer->m_plan2 = moe_hosted_device_ptr(layer->h_plan2)) && (layer->m_chk = moe_hosted_device_ptr(layer->h_chk)) &&
         (layer->m_fetch = moe_hosted_device_ptr(layer->h_fetch)) &&
         (!split || ((layer->m_plank = moe_hosted_device_ptr(layer->h_plank)) && (layer->m_src = moe_hosted_device_ptr(layer->h_src)) &&
                     (layer->m_plan2k = moe_hosted_device_ptr(layer->h_plan2k))));
    ok = ok && cudaMalloc((void **) &layer->d_fetch, (1 + 3 * MOE_SPLIT_FETCH_MAX) * sizeof(uint64_t)) == cudaSuccess &&
         cudaMemset(layer->d_fetch, 0, sizeof(uint64_t)) == cudaSuccess &&
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
                cudaMalloc((void **) &st.d_q8_mid, mid_bytes) != cudaSuccess ||
                cudaMalloc((void **) &st.d_q8_count, mid_bytes / sizeof(block_q8_1) * sizeof(unsigned)) != cudaSuccess ||
                cudaMemset(st.d_q8_count, 0, mid_bytes / sizeof(block_q8_1) * sizeof(unsigned)) != cudaSuccess) {
                GGML_ABORT("MoE block hosted by the GPU: allocation failed");
            }
            st.q8_act_bytes = act_bytes;
            st.q8_mid_bytes = mid_bytes;
        }
        // addresses captured in graphs: one embedding and expert width per session
        GGML_ASSERT(act_bytes <= st.q8_act_bytes && mid_bytes <= st.q8_mid_bytes);
    }
    slot = (int) st.layers.size();
    st.slot_of[gate_w->data] = slot;
    st.layers.push_back(std::move(layer));
    return st.layers.back().get();
}

// the shared expert's gate after the hosted block at node i: a MUL_MAT of a row of F32 weights (device memory) by the
// block's activations, that mul_mat_vec_f would compute with 256 threads; taken by the merge when the merge is launched
// by the node after it at the latest (one of the 3 nodes after it reads the block's output: the hosted flush window) and
// no node between the block and it writes where its result goes (ggml-alloc may give that memory to an earlier tensor).
// GGML_CUDA_MOE_SGATE=0: computed by its own node
static moe_sgate_args moe_sgate_find(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
                                     const ggml_tensor * act, const ggml_tensor * down, int n_tokens) {
    static const bool on = !getenv("GGML_CUDA_MOE_SGATE") || atoi(getenv("GGML_CUDA_MOE_SGATE")) != 0;
    const int64_t n_embd = act->ne[0];
    if (!on || n_tokens > MOE_SGATE_MAX_COLS || n_embd % 2 != 0) {
        return {};
    }
    // mul_mat_vec_f's block size for this width (ggml_cuda_mul_mat_vec_f): 256 only
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    int64_t bs_best = warp_size, niter_best = (n_embd + 2*warp_size - 1) / (2*warp_size);
    for (int64_t bs = 2*warp_size; bs <= 256; bs += warp_size) {
        const int64_t niter = (n_embd + 2*bs - 1) / (2*bs);
        if (niter < niter_best) {
            niter_best = niter;
            bs_best = bs;
        }
    }
    if (bs_best != 256) {
        return {};
    }
    const auto reads_down = [&](const ggml_tensor * t) {
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = t->src[j];
            if (src && (src == down || src->view_src == down)) {
                return true;
            }
        }
        return false;
    };
    for (int g = i + 4; g < std::min(i + 24, cgraph->n_nodes); g++) {
        const ggml_tensor * node = cgraph->nodes[g];
        if (reads_down(node)) {
            return {};   // past the merge
        }
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const ggml_tensor * w = node->src[0];
        const ggml_tensor * y = node->src[1];
        if (w->type != GGML_TYPE_F32 || w->ne[0] != n_embd || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 ||
                !ggml_is_contiguous(w) || !w->buffer || ggml_backend_buffer_is_host(w->buffer) ||
                y->type != GGML_TYPE_F32 || y->data != act->data || y->ne[0] != n_embd || y->ne[1] != n_tokens ||
                y->ne[2] != 1 || y->ne[3] != 1 || y->nb[0] != sizeof(float) ||
                node->type != GGML_TYPE_F32 || node->ne[0] != 1 || node->ne[1] != n_tokens) {
            continue;
        }
        bool next_reads = false;
        for (int k = g + 1; k <= g + 3 && k < cgraph->n_nodes; k++) {
            next_reads = next_reads || reads_down(cgraph->nodes[k]);
        }
        const char * d0 = (const char *) node->data;
        const char * d1 = d0 + ggml_nbytes(node);
        bool clash = false;
        for (int k = i + 4; k < g && !clash; k++) {
            const ggml_tensor * t = cgraph->nodes[k];
            const char * t0 = (const char *) t->data;
            clash = t0 && t0 < d1 && d0 < t0 + ggml_nbytes(t);
        }
        if (!next_reads || clash) {
            return {};
        }
        ctx.absorbed.insert(node);
        moe_sgate_args a;
        a.x = (const float *) w->data;
        a.y = (const float *) y->data;
        a.dst = (float *) node->data;
        a.ncols2 = (int) (n_embd / 2);
        a.stride_col_y2 = (int) (y->nb[1] / sizeof(float) / 2);
        a.stride_col_dst = (int) (node->nb[1] / sizeof(float));
        a.ncols = n_tokens;
        return a;
    }
    return {};
}

// EXPERT SPLIT, HIP / CUDA graphs: while the hosting card's stream is captured, every other card's stream is captured too
// (a separate graph per card, the same blocks in the same order); each launch of the hosting card's graph launches theirs
// first. The cards' chains wait for the plans, so their order against the hosting card's graph does not matter
struct moe_split_graph {
    int logical = -1;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};
static thread_local const void * g_split_capture_owner = nullptr;
static thread_local int g_split_capture_device = -1;
static thread_local std::vector<moe_split_graph> g_split_capturing;   // streams being captured with the owner's
static std::map<const void *, std::vector<moe_split_graph>> g_split_graphs;   // under g_hosted_mu

static void moe_split_capture_card(const moe_split_card & card) {
    ggml_cuda_set_device(card.device->logical);
    CUDA_CHECK(cudaStreamBeginCapture(card.stream, cudaStreamCaptureModeRelaxed));
    moe_split_graph g;
    g.logical = card.device->logical;
    g.stream = card.stream;
    g_split_capturing.push_back(g);
}

void ggml_cuda_moe_split_capture_begin(int device, const void * owner) {
    if (!moe_cache_expert_split()) {
        return;
    }
    g_split_capture_owner = owner;
    g_split_capture_device = device;
    g_split_capturing.clear();
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    for (auto & entry : g_hosted) {
        if (entry.first.second != device || !entry.second) {
            continue;
        }
        for (const moe_split_card & card : entry.second->split_cards) {
            moe_split_capture_card(card);
        }
    }
    ggml_cuda_set_device(device);
}

void ggml_cuda_moe_split_capture_end(int device, const void * owner) {
    if (!moe_cache_expert_split() || g_split_capture_owner != owner) {
        return;
    }
    std::vector<moe_split_graph> done;
    for (moe_split_graph g : g_split_capturing) {
        ggml_cuda_set_device(g.logical);
        CUDA_CHECK(cudaStreamEndCapture(g.stream, &g.graph));
        CUDA_CHECK(cudaGraphInstantiate(&g.exec, g.graph, NULL, NULL, 0));
        done.push_back(g);
    }
    g_split_capturing.clear();
    g_split_capture_owner = nullptr;
    g_split_capture_device = -1;
    {
        std::lock_guard<std::mutex> lock(g_hosted_mu);
        auto & old = g_split_graphs[owner];
        for (const moe_split_graph & g : old) {
            ggml_cuda_set_device(g.logical);
            (void) cudaGraphExecDestroy(g.exec);
            (void) cudaGraphDestroy(g.graph);
        }
        old = std::move(done);
    }
    ggml_cuda_set_device(device);
}

void ggml_cuda_moe_split_graph_launch(int device, const void * owner) {
    if (!moe_cache_expert_split()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_hosted_mu);
    const auto it = g_split_graphs.find(owner);
    if (it == g_split_graphs.end()) {
        return;
    }
    for (const moe_split_graph & g : it->second) {
        ggml_cuda_set_device(g.logical);
        CUDA_CHECK(cudaGraphLaunch(g.exec, g.stream));
    }
    ggml_cuda_set_device(device);
}

// EXPERT SPLIT: the other cards of the session, each with a stream and buffers for the largest block (made once, by the
// hosting card's first block)
static void moe_split_cards_create(moe_cache_session & session, moe_hosted_state & st, moe_cache_device & host,
                                   int64_t n_embd, int64_t n_ff, int n_used) {
    const int rows = moe_hosted_rows_max;
    const int64_t padded_embd = GGML_PAD(n_embd, MATRIX_ROW_PADDING), padded_ff = GGML_PAD(n_ff, MATRIX_ROW_PADDING);
    const int tokens = (rows + n_used - 1) / n_used;
    for (auto & dp : session.devices) {
        if (dp.get() == &host || (int) st.split_cards.size() >= MOE_SPLIT_MAX_CARDS) {
            continue;
        }
        moe_split_card card;
        card.index = (int) st.split_cards.size() + 1;
        card.device = dp.get();
        card.n_embd = n_embd;
        card.n_ff = n_ff;
        ggml_cuda_set_device(card.device->logical);
        const size_t mid_blocks = (size_t) rows * (padded_ff / QK8_1);
        std::vector<int32_t> act_ids(rows);
        for (int r = 0; r < rows; r++) {
            act_ids[r] = r / n_used;
        }
        const bool ok = cudaStreamCreateWithFlags(&card.stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaMalloc((void **) &card.d_last, MOE_SPLIT_MAX_SLOTS * sizeof(uint32_t)) == cudaSuccess &&
            cudaMemset(card.d_last, 0, MOE_SPLIT_MAX_SLOTS * sizeof(uint32_t)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_seq, sizeof(uint32_t)) == cudaSuccess &&
            // each plan followed by its chunk list (moe_group_list), as on the hosting card
            cudaMalloc((void **) &card.d_plan, (size_t) 3 * rows * sizeof(uint64_t) + MOE_GROUP_WORDS * sizeof(int32_t)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_plan2, (size_t) 3 * rows * sizeof(uint64_t) + MOE_GROUP_WORDS * sizeof(int32_t)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_act_ids, rows * sizeof(int32_t)) == cudaSuccess &&
            cudaMemcpy(card.d_act_ids, act_ids.data(), rows * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMalloc((void **) &card.d_q8_act, (size_t) tokens * (padded_embd / QK8_1) * sizeof(block_q8_1)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_act, (size_t) tokens * n_embd * sizeof(float)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_glu, (size_t) rows * n_ff * sizeof(float)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_q8_mid, mid_blocks * sizeof(block_q8_1)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_q8_count, mid_blocks * sizeof(unsigned)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_res, (size_t) MOE_SPLIT_PSUM_TOKENS * n_used * n_embd * sizeof(float)) == cudaSuccess &&
            cudaMalloc((void **) &card.d_w, (size_t) rows * sizeof(float)) == cudaSuccess &&
            cudaMemset(card.d_q8_count, 0, mid_blocks * sizeof(unsigned)) == cudaSuccess;
        if (!ok) {
            (void) cudaGetLastError();
            GGML_ABORT("EXPERT SPLIT: allocation on device %d failed", card.device->logical);
        }
        if (!st.h_split_diag) {
            if (!moe_hosted_host_alloc((void **) &st.h_split_diag, 64) ||
                !(st.m_split_diag = moe_hosted_device_ptr(st.h_split_diag))) {
                GGML_ABORT("EXPERT SPLIT: diagnostic word allocation failed");
            }
        }
        {
            std::string desc;
            const double rate = moe_cache_link_rate(*card.device, desc);
            if (rate > 0.0) {
                card.rate = rate;
            }
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d expert split: device %d computes the rows of its cache for device %d's blocks "
                "(card %d, %.1f GB/s for a copy)\n", card.device->physical, card.device->logical, host.logical, card.index,
                card.rate / 1000.0);
        if (g_tl_h && !g_tlk_h[card.index] &&
                moe_hosted_host_alloc((void **) &g_tlk_h[card.index], MOE_TL_RING * 8 * sizeof(unsigned long long))) {
            g_tlk_m[card.index] = moe_hosted_device_ptr(g_tlk_h[card.index]);
        }
        st.split_cards.push_back(card);
        if (g_split_capture_owner && g_split_capture_device == host.logical) {
            moe_split_capture_card(card);   // made during a capture of the hosting card's graph
        }
    }
    ggml_cuda_set_device(host.logical);
}


// ---------------------------------------------------------------------------------------------------------------------
// GGML_CUDA_MOE_RING_PREFILL: a prompt's MoE blocks hosted, their experts streamed through a ring of entries in the
// zones the scheduler borrowed for the prefill (ggml_moe_cache.ring_zone), computed by groups (mmq.cu,
// ggml_cuda_moe_block_grouped_xe). The experts the cache holds are read in place; the others are copied in a fixed
// order (layer, expert), a packed layer's row in one copy, each into the next bytes of the ring (an entry of its layer's
// size: moe-ring.h) once the groups that read what they cover are done (an event on the compute stream, waited for on the
// copy stream): the copies of a layer start before its block,
// behind the previous layer's groups.
struct moe_ring_state {
    char * base = nullptr;
    size_t bytes = 0;
    const void * owner = nullptr;    // the backend context of the scheduler that lent the zones
    moe_vram_ring<moe_cache_key, moe_cache_key_hash> ring;
    cudaStream_t copy = nullptr;
    static constexpr int P = 4096;   // events: groups in flight stay far below
    std::vector<cudaEvent_t> done, copied;
    int64_t seq = 0;                 // the next group's number
    int64_t waited = -1;             // the last group whose done event the copy stream waits for (they are in order)
    std::vector<std::pair<moe_cache_pool *, int>> pins, pins_prev;
    uint64_t copies = 0, in_place = 0;
};

static void moe_ring_drain(moe_cache_device & device) {
    if (!device.ring || !device.ring->copy) {
        return;
    }
    moe_ring_state & r = *device.ring;
    ggml_cuda_set_device(device.logical);
    CUDA_CHECK(cudaStreamSynchronize(r.copy));
    if (r.seq > 0) {
        CUDA_CHECK(cudaEventSynchronize(r.done[(r.seq - 1) % moe_ring_state::P]));
    }
}

static void moe_ring_unpin(moe_cache_session & session, std::vector<std::pair<moe_cache_pool *, int>> & pins) {
    std::lock_guard<std::mutex> lock(session.mu);
    for (const auto & pin : pins) {
        if (pin.second >= 0 && pin.second < pin.first->n_slots && pin.first->slots[pin.second].readers > 0) {
            pin.first->slots[pin.second].readers--;
        }
    }
    pins.clear();
}

static void moe_cache_ring_zone(void * opaque, void * backend, void * base, size_t bytes) {
    moe_cache_session * session = (moe_cache_session *) opaque;
    moe_cache_device * device = session ? moe_cache_device_for_backend(*session, backend) : nullptr;
    if (!device) {
        return;
    }
    ggml_cuda_set_device(device->logical);
    if (!device->ring) {
        device->ring = new moe_ring_state();
        moe_ring_state & r = *device->ring;
        CUDA_CHECK(cudaStreamCreateWithFlags(&r.copy, cudaStreamNonBlocking));
        r.done.resize(moe_ring_state::P);
        r.copied.resize(moe_ring_state::P);
        for (int k = 0; k < moe_ring_state::P; k++) {
            CUDA_CHECK(cudaEventCreateWithFlags(&r.done[k], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&r.copied[k], cudaEventDisableTiming));
        }
    }
    moe_ring_state & r = *device->ring;
    if (!base) {
        if (r.base) {
            // nothing may still copy into the zones or read them: they go back to the cache
            CUDA_CHECK(cudaStreamSynchronize(r.copy));
            if (r.seq > 0) {
                CUDA_CHECK(cudaEventSynchronize(r.done[(r.seq - 1) % moe_ring_state::P]));
            }
            moe_ring_unpin(*session, r.pins_prev);
            moe_ring_unpin(*session, r.pins);
            MOE_CACHE_LOG("[moe-cache] CUDA%d prefill ring: %llu experts copied, %llu read in place (%.1f %% in place)\n",
                    device->physical, (unsigned long long) r.copies, (unsigned long long) r.in_place,
                    100.0 * r.in_place / std::max<uint64_t>(1, r.copies + r.in_place));
        }
        r.base = nullptr;
        r.bytes = 0;
        r.owner = nullptr;
        r.ring.give_back();
        return;
    }
    r.owner = backend ? ((ggml_backend_t) backend)->context : nullptr;
    if (r.base == base && r.bytes == bytes) {
        return;
    }
    // an entry: its layer's expert, with what MMQ reads past its last row (ggml_cuda_moe_ring_block); the log gives the
    // entries the zone holds for the largest and the smallest expert of the device's pools
    size_t largest = 0, smallest = SIZE_MAX;
    for (const auto & pool : device->pools) {
        largest = std::max(largest, pool->expert_size);
        smallest = std::min(smallest, pool->expert_size);
    }
    r.base = (char *) base;
    r.bytes = bytes;
    r.ring.lend(bytes, {});
    // finite bytes past each expert's rows (the zones held cache entries; zeroed once)
    CUDA_CHECK(cudaMemsetAsync(base, 0, bytes, r.copy));
    const size_t e_hi = largest ? GGML_PAD(largest + 1024, 256) : 0, e_lo = smallest != SIZE_MAX ? GGML_PAD(smallest + 1024, 256) : 0;
    MOE_CACHE_LOG("[moe-cache] CUDA%d prefill ring: %zu MiB, %zu entries of %zu KiB to %zu of %zu KiB (each layer's own size)\n",
            device->physical, bytes >> 20, r.ring.entries(e_hi), e_hi >> 10, r.ring.entries(e_lo), e_lo >> 10);
}

bool ggml_cuda_moe_ring_takes(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    static const bool ring_prefill = getenv("GGML_CUDA_MOE_RING_PREFILL") && atoi(getenv("GGML_CUDA_MOE_RING_PREFILL")) > 0;
    if (!ring_prefill || node->op != GGML_OP_MUL_MAT_ID || !node->src[2] || node->src[2]->ne[1] < 128 || !node->src[0]->buffer ||
            !ggml_backend_buffer_is_host(node->src[0]->buffer)) {
        return false;
    }
    if (node->src[2]->ne[0] * node->src[2]->ne[1] > moe_hosted_max_rows() || node->src[2]->ne[1] > moe_hosted_tokens_max) {
        return true;
    }
    if (g_session_stack.empty() || !g_session_stack.back().active) {
        return false;
    }
    for (auto & d : g_session_stack.back().active->devices) {
        if (d->logical == ctx.device && d->ring && d->ring->base && d->ring->owner == (const void *) &ctx) {
            return true;
        }
    }
    return false;
}

int ggml_cuda_moe_ring_block(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 3 >= cgraph->n_nodes || g_session_stack.empty() || !g_session_stack.back().active) {
        return 0;
    }
    ggml_tensor * gate = cgraph->nodes[i], * up = cgraph->nodes[i + 1], * down = cgraph->nodes[i + 3];
    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID || down->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    moe_cache_session & session = *g_session_stack.back().active;
    moe_cache_device * device = nullptr;
    for (auto & d : session.devices) {
        if (d->logical == ctx.device) {
            device = d.get();
        }
    }
    if (!device || !device->ring || !device->ring->base || device->ring->owner != (const void *) &ctx) {
        return 0;
    }
    moe_ring_state & r = *device->ring;
    const ggml_tensor * w[3] = { gate->src[0], up->src[0], down->src[0] };
    const int64_t n_expert = w[0]->ne[2];
    const size_t stride = w[0]->nb[2];
    const char * lo = nullptr;
    for (const ggml_tensor * t : w) {
        if (t->nb[2] != stride || t->ne[2] != n_expert || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
            return 0;
        }
        lo = lo ? std::min(lo, (const char *) t->data) : (const char *) t->data;
    }
    size_t role_off[3];
    // the bytes of an expert's row its three parts cover: the row's tail past them is the margin MMQ reads past the
    // down's last row (256 B), which for a layer's last expert lies past its tensors, maybe outside the registered
    // memory (blk.22 of the IQ3_KS hybrid: invalid argument from the copy, CUDA and HIP): that one copies its parts
    // and zeroes the margin
    size_t span = 0;
    for (int k = 0; k < 3; k++) {
        role_off[k] = (const char *) w[k]->data - lo;
        // a packed layer: the three parts within one row of `stride` bytes
        if (role_off[k] + w[k]->nb[1]*w[k]->ne[1] > stride) {
            return 0;
        }
        span = std::max(span, role_off[k] + w[k]->nb[1]*w[k]->ne[1]);
    }
    // GGML_CUDA_MOE_RING_GROUP=G: the experts of a group (32 by default): a larger group, fewer launches and fewer idle
    // SMs in each launch's last wave, against a ring that must hold a group and later reads of a group's first copies
    static const int G = getenv("GGML_CUDA_MOE_RING_GROUP") && atoi(getenv("GGML_CUDA_MOE_RING_GROUP")) > 0 ?
                         atoi(getenv("GGML_CUDA_MOE_RING_GROUP")) : 32;
    // this layer's entry: its expert's row and what MMQ reads past the down's last row (72 B; the zones are zeroed once,
    // then hold copied experts: finite)
    const size_t eb = GGML_PAD(stride + 1024, 256);
    if (eb < stride + 512) {
        return 0;
    }
    if (n_expert % G != 0 || G >= n_expert || !r.ring.fits(G, eb)) {
        return 0;
    }
    // the experts the cache holds (a whole expert: every role), pinned until the next block's synchronization
    std::vector<uint64_t> xe[3];
    for (int k = 0; k < 3; k++) {
        xe[k].assign(n_expert, 0);
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        const moe_cache_layer * layer = moe_cache_layer_of(*device, w[0]->data);
        const int pool_index = moe_cache_pool_of(*device, w[0]->data);
        if (!session.stopping && !session.dormant && !device->dead.load() && device->budget_ready && layer &&
                pool_index >= 0 && pool_index < (int) device->pools.size() && layer->n_expert == n_expert) {
            moe_cache_pool & pool = *device->pools[pool_index];
            int role_of[3];
            bool ok = pool.slab != nullptr;
            for (int k = 0; k < 3 && ok; k++) {
                role_of[k] = moe_cache_role_of_name(w[k]->name);
                ok = role_of[k] >= 0 && layer->tensor[role_of[k]] == w[k]->data;
            }
            const auto d = ok ? pool.direct.find(layer->tensor[MOE_ROLE_GATE]) : pool.direct.end();
            if (d != pool.direct.end()) {
                for (int64_t e = 0; e < n_expert && e < (int64_t) d->second.size(); e++) {
                    const int sl = d->second[e];
                    if (sl >= 0 && pool.slots[sl].state == moe_cache_slot_state::valid) {
                        pool.slots[sl].readers++;
                        session.prefill_pins.push_back({&pool, sl});   // released by prefill_unpin (after moe_ring_drain)
                        for (int k = 0; k < 3; k++) {
                            xe[k][e] = moe_cache_entry_addr(pool, sl) + pool.role_offset[role_of[k]];
                        }
                    }
                }
            }
        }
    }
    // the others: an entry each, in order; a copy waits for the group that read its entry last
    cudaStream_t cs = ctx.stream();
    const int64_t seq0 = r.seq;
    const int64_t n_groups = n_expert / G;
    struct pend { int64_t e; int64_t wait; int64_t off; bool issued; };
    std::vector<std::vector<pend>> by_group(n_groups);
    for (int64_t e = 0; e < n_expert; e++) {
        if (xe[0][e]) {
            r.in_place++;
            continue;
        }
        int64_t wait = -1;
        const int64_t off = r.ring.take(moe_cache_key{w[0]->data, (int32_t) e}, seq0 + e / G, eb, &wait);
        GGML_ASSERT(off >= 0);
        for (int k = 0; k < 3; k++) {
            xe[k][e] = (uint64_t) (uintptr_t) (r.base + (size_t) off + role_off[k]);
        }
        by_group[e / G].push_back({e, wait, off, false});
    }
    // GGML_CUDA_MOE_GROUPED_CHECK computes the block again after its groups, reading the entries: a ring smaller than
    // the block's copies has overwritten its first entries by then (the reference differs, not the block)
    static const bool check = getenv("GGML_CUDA_MOE_GROUPED_CHECK") && atoi(getenv("GGML_CUDA_MOE_GROUPED_CHECK")) != 0;
    size_t n_copies = 0;
    for (const auto & bg : by_group) {
        n_copies += bg.size();
    }
    if (check && n_copies > r.ring.entries(eb)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("%s: GGML_CUDA_MOE_GROUPED_CHECK with a ring of %zu entries for %llu copies a block: the check reads "
                    "overwritten entries, use a ring of a layer's experts (GGML_CUDA_MOE_RING_PREFILL=1800)\n", __func__,
                    r.ring.entries(eb), (unsigned long long) n_copies);
        }
    }
    static const bool validate = getenv("GGML_CUDA_MOE_RING_VALIDATE") && atoi(getenv("GGML_CUDA_MOE_RING_VALIDATE")) != 0;
    if (validate) {
        const uint64_t s0 = (uint64_t) (uintptr_t) device->band_slab, s1 = s0 + moe_cache_band_total(*device);
        const uint64_t q0 = (uint64_t) (uintptr_t) r.base, q1 = q0 + r.bytes;
        int bad = 0, in_slab = 0, in_ring = 0;
        for (int k = 0; k < 3; k++) {
            for (int64_t e = 0; e < n_expert; e++) {
                const uint64_t a = xe[k][e];
                const bool ring_a = a >= q0 && a < q1, slab_a = a >= s0 && a < s1;
                in_ring += ring_a; in_slab += slab_a && !ring_a;
                if (!ring_a && !slab_a && bad++ < 4) {
                    GGML_LOG_INFO("ring validate %s: role %d expert %lld at %#llx outside slab [%#llx, %#llx) and ring [%#llx, %#llx)\n",
                        w[0]->name, k, (long long) e, (unsigned long long) a, (unsigned long long) s0, (unsigned long long) s1,
                        (unsigned long long) q0, (unsigned long long) q1);
                }
            }
        }
        GGML_LOG_INFO("ring validate %s: %d in the slab, %d in the ring, %d elsewhere; stride %zu entry %zu offsets %zu %zu %zu\n", w[0]->name,
                in_slab, in_ring, bad, stride, eb, role_off[0], role_off[1], role_off[2]);
    }
    const bool two_streams = ggml_cuda_moe_grouped_streams() == 2;
    cudaStream_t gstream = cs;          // the group's stream (set by the grouped block before each hook)
    int64_t recorded = seq0;            // groups below have their done event recorded
    int64_t next_group_to_close = 0;    // groups below have every copy issued (and their copied event)
    // the copies in order, group by group, as far as their waits are recorded: those that wait for nothing of this
    // block go now (they run during the routing and the previous layers)
    auto advance = [&]() {
        while (next_group_to_close < n_groups) {
            bool all = true;
            for (auto & p : by_group[next_group_to_close]) {
                if (p.issued) {
                    continue;
                }
                if (p.wait >= recorded) {
                    all = false;
                    break;
                }
                // one wait per group read, not per copy: the done events are recorded in order on the compute stream, a
                // later one covers the earlier (ROCm 7.2.4: a wait before every copy cost the copy stream 3.7 %, ringsim)
                if (p.wait > r.waited) {
                    CUDA_CHECK(cudaStreamWaitEvent(r.copy, r.done[p.wait % moe_ring_state::P], 0));
                    // two streams (GGML_CUDA_MOE_GROUPED_STREAMS=2): a done event covers its stream's groups only, the
                    // previous group's (the other stream) too then
                    if (two_streams && p.wait - 1 > r.waited) {
                        CUDA_CHECK(cudaStreamWaitEvent(r.copy, r.done[(p.wait - 1) % moe_ring_state::P], 0));
                    }
                    r.waited = p.wait;
                }
                char * dst = r.base + (size_t) p.off;
                const size_t bytes = p.e == n_expert - 1 ? span : stride;
                CUDA_CHECK(cudaMemcpyAsync(dst, lo + (size_t) p.e*stride, bytes, cudaMemcpyHostToDevice, r.copy));
                if (bytes < stride) {
                    CUDA_CHECK(cudaMemsetAsync(dst + bytes, 0, stride - bytes, r.copy));
                }
                r.copies++;
                p.issued = true;
            }
            if (!all) {
                break;
            }
            if (!by_group[next_group_to_close].empty()) {
                CUDA_CHECK(cudaEventRecord(r.copied[(seq0 + next_group_to_close) % moe_ring_state::P], r.copy));
            }
            next_group_to_close++;
        }
    };
    advance();
    std::function<void(int64_t)> before = [&](int64_t g) {
        advance();
        GGML_ASSERT(next_group_to_close > g && "a group's copies wait for itself");
        if (!by_group[g].empty()) {
            CUDA_CHECK(cudaStreamWaitEvent(gstream, r.copied[(seq0 + g) % moe_ring_state::P], 0));
        }
    };
    std::function<void(int64_t)> after = [&](int64_t g) {
        CUDA_CHECK(cudaEventRecord(r.done[(seq0 + g) % moe_ring_state::P], gstream));
        recorded = seq0 + g + 1;
        advance();
    };
    std::function<void()> synced = [&]() {   // the previous block's reads are done: its pins go
        moe_ring_unpin(session, r.pins_prev);
    };
    // GGML_CUDA_MOE_RING_DUMP=FILE: each block's routing appended (tests/ringsim reads it): "RNG1", layer, tokens,
    // experts, the expert's stride, gate and down types, the rows of each expert, whether the cache held it
    static const char * dump_path = getenv("GGML_CUDA_MOE_RING_DUMP");
    std::vector<int32_t> hb;
    std::vector<uint8_t> held(n_expert);
    for (int64_t e = 0; e < n_expert; e++) {
        held[e] = !(xe[0][e] >= (uint64_t) (uintptr_t) r.base && xe[0][e] < (uint64_t) (uintptr_t) (r.base + r.bytes));
    }
    const int covered = ggml_cuda_moe_block_grouped_xe(ctx, cgraph, i, G, xe, &before, &after, &synced, dump_path ? &hb : nullptr,
        &gstream);
    GGML_ASSERT(covered > 0 && "a ring prompt MoE block the grouped path does not take");
    if (two_streams && n_groups > 0) {
        // the block's last done event again on its stream, after the join: it covers both streams (moe_ring_drain, the
        // next block's first waits); the copies that already wait for the first record keep it
        CUDA_CHECK(cudaEventRecord(r.done[(seq0 + n_groups - 1) % moe_ring_state::P], cs));
    }
    if (dump_path && (int64_t) hb.size() == n_expert + 1) {
        static FILE * f = fopen(dump_path, "ab");
        if (f) {
            int layer = -1;
            sscanf(w[0]->name, "blk.%d.", &layer);
            const int32_t hdr[4] = { 0x31474e52, layer, (int32_t) gate->src[2]->ne[1], (int32_t) n_expert };
            const uint64_t st = stride;
            const int32_t types[2] = { (int32_t) w[0]->type, (int32_t) w[2]->type };
            std::vector<int32_t> rows(n_expert);
            for (int64_t e = 0; e < n_expert; e++) {
                rows[e] = hb[e + 1] - hb[e];
            }
            fwrite(hdr, sizeof(hdr), 1, f);
            fwrite(&st, sizeof(st), 1, f);
            fwrite(types, sizeof(types), 1, f);
            fwrite(rows.data(), sizeof(int32_t), n_expert, f);
            fwrite(held.data(), 1, n_expert, f);
            fflush(f);
        }
    }
    r.seq = seq0 + n_groups;
    r.pins_prev.insert(r.pins_prev.end(), r.pins.begin(), r.pins.end());
    r.pins.clear();
    return covered;
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
    if (up->src[1] != act || up->src[2] != ids || down->src[2] != ids || rows > moe_hosted_max_rows() || n_tokens > moe_hosted_tokens_max ||
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
    if (!st->gpu_benched) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ctx.stream(), &capture) == cudaSuccess && capture == cudaStreamCaptureStatusNone) {
            moe_split_gpu_bench(st, *device, layer, gate->src[0]->type, down->src[0]->type, ctx.stream());
        }
        (void) cudaGetLastError();
    }

    cudaStream_t stream = ctx.stream();
    const int64_t n_embd = layer->n_embd;
    const int64_t n_ff = layer->n_ff;
    // EXPERT SPLIT: the other cards, before the first block's activations are published (they read the F32 ones)
    if (moe_cache_expert_split() && session.devices.size() > 1 && st->split_cards.empty()) {
        moe_split_cards_create(session, *st, *device, n_embd, n_ff, layer->n_used);
    }
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

    // EXPERT SPLIT, PSUM: the routing weights the weighting after the block reads (the first node to read its output),
    // published with the ids for the other cards' weighted sums; computed by this kernel's top-k, or before the block
    const float * w_src = nullptr;
    static const bool psum_on = !getenv("GGML_MOE_SPLIT_PSUM") || atoi(getenv("GGML_MOE_SPLIT_PSUM")) != 0;
    if (psum_on && !st->split_cards.empty() && n_tokens <= MOE_SPLIT_PSUM_TOKENS) {
        for (int g = i + 4; g < std::min(i + 24, cgraph->n_nodes); g++) {
            const ggml_tensor * node = cgraph->nodes[g];
            const ggml_tensor * dsrc = nullptr, * wt = nullptr;
            for (int j = 0; j < 2; j++) {
                const ggml_tensor * a = node->src[j];
                if (a && (a == down || a->view_src == down)) {
                    dsrc = a;
                    wt = node->src[1 - j];
                }
            }
            if (!dsrc) {
                bool reads = false;
                for (int j = 0; j < GGML_MAX_SRC && !reads; j++) {
                    reads = node->src[j] && (node->src[j] == down || node->src[j]->view_src == down);
                }
                if (reads) {
                    break;
                }
                continue;
            }
            if (node->op == GGML_OP_MUL && wt && wt->type == GGML_TYPE_F32 && ggml_nelements(wt) == rows &&
                    ggml_is_contiguous(wt)) {
                bool ready = true;   // not written by a node between the block and its weighting
                for (int h = i + 4; h < g && ready; h++) {
                    ready = cgraph->nodes[h] != wt && cgraph->nodes[h] != wt->view_src;
                }
                if (tk && (const float *) wt->data == tp.weights) {
                    w_src = tp.weights;
                } else if (ready) {
                    w_src = (const float *) wt->data;
                }
            }
            break;
        }
    }

    // activations and ids to the host, then the mailbox
    // the router whose logits the top-k reads, left to this kernel too (ggml_cuda_router_defer_ok)
    moe_router_args rta;
    if (tk && ctx.router_pending && (const float *) ctx.router_pending->data == tp.logits) {
        const ggml_tensor * rn = ctx.router_pending;
        const ggml_tensor * w = rn->src[0];
        const ggml_tensor * y = rn->src[1];
        rta.x = (const nv_bfloat16 *) w->data;
        rta.y = (const float *) y->data;
        rta.dst = (float *) rn->data;
        rta.ncols2 = (int) (w->ne[0] / 2);
        rta.stride_row = (int) (w->nb[1] / ggml_type_size(w->type));
        rta.stride_col_y2 = (int) (y->nb[1] / sizeof(float) / 2);
        rta.stride_col_dst = (int) (rn->nb[1] / sizeof(float));
        rta.ncols = (int) y->ne[1];
        rta.nrows = (int) w->ne[1];
        if (!st->d_router_done) {
            CUDA_CHECK(cudaMalloc((void **) &st->d_router_done, 64));
            CUDA_CHECK(cudaMemsetAsync(st->d_router_done, 0, 64, stream));
        }
        rta.done = st->d_router_done;
        // the rows tagged with the layer's publication number, which the kernel reads (GGML_CUDA_MOE_ROUTER_TAG=0: a counter
        // after a fence)
        static const bool router_tag = !getenv("GGML_CUDA_MOE_ROUTER_TAG") || atoi(getenv("GGML_CUDA_MOE_ROUTER_TAG")) != 0;
        if (router_tag && rta.ncols <= MOE_ROUTER_MAX_COLS) {
            if (!st->d_router_tagged) {
                const size_t bytes = (size_t) MOE_ROUTER_MAX_COLS * TOPK_MOE_HOSTED_N_EXPERT * sizeof(uint64_t);
                CUDA_CHECK(cudaMalloc((void **) &st->d_router_tagged, bytes));
                CUDA_CHECK(cudaMemsetAsync(st->d_router_tagged, 0, bytes, stream));
            }
            rta.tagged = st->d_router_tagged;
        }
        ctx.router_pending = nullptr;
        ctx.router_next = -1;
    } else {
        ggml_cuda_router_flush(ctx);
    }
    k_moe_hosted_enter<<<MOE_HOSTED_ENTER_BLOCKS + (rta.x ? rta.nrows : 0), 256, 0, stream>>>((const float *) act->data,
                                              moe_hosted_act_f32() || !st->split_cards.empty() ? layer->m_act : nullptr, layer->m_act8, (int64_t) n_tokens * n_embd,
                                              tk ? tp.logits : nullptr, tk ? tp.weights : nullptr, tp.clamp, tp.scale,
                                              (int32_t *) ids->data, ids->nb[1], layer->m_ids, n_used, n_tokens,
                                              st->d_seq, layer->d_expect, st->m_mail, (uint32_t) slot, layer->m_flags,
                                              layer->m_plan, layer->d_plan, 3 * rows, g_tl_m, rta,
                                              w_src, w_src ? layer->m_w : nullptr);
    // a block launched now (not recorded into a graph) waits for its mailbox: the server thread serves it
    {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture) == cudaSuccess && capture == cudaStreamCaptureStatusNone) {
            {
                std::lock_guard<std::mutex> lock(st->server_mu);
                st->server_pending++;
            }
            st->server_cv.notify_one();
        }
        (void) cudaGetLastError();
    }
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
    // up / gate / swiglu, and the q8_1 of the result for down written by the same kernel (GGML_CUDA_MOE_Q8_FUSED=0: a
    // quantization kernel after it)
    static const bool q8_fused = !getenv("GGML_CUDA_MOE_Q8_FUSED") || atoi(getenv("GGML_CUDA_MOE_Q8_FUSED")) != 0;
    // the plan's chunk list (written after it by the kernel that copies it): the rows of one expert computed together
    const auto groups_of = [&](const uint64_t * plan) -> const int32_t * {
        return n_tokens >= 2 && rows <= MOE_GROUP_MAX_ROWS ? (const int32_t *) (plan + 3 * rows) : nullptr;
    };
    const auto up_gate = [&](const uint64_t * plan) {
        if (q8_fused && n_ff % QK8_1 == 0) {
            ggml_cuda_moe_cache_mmv_fused(wtype, q8_act, plan, plan + rows, layer->d_act_ids, (float *) glu->data, n_embd, n_ff,
                                          rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY, stream, q8_mid,
                                          padded_ff / QK8_1, st->d_q8_count, groups_of(plan), (int) n_tokens);
            return;
        }
        ggml_cuda_moe_cache_mmv_fused(wtype, q8_act, plan, plan + rows, layer->d_act_ids, (float *) glu->data, n_embd, n_ff,
                                      rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY, stream, nullptr, 0, nullptr,
                                      groups_of(plan), (int) n_tokens);
        quantize_row_q8_1_cuda((const float *) glu->data, nullptr, q8_mid, wtype, n_ff, n_ff, (int64_t) rows * n_ff,
                               (int64_t) rows * n_ff, padded_ff, rows, 1, 1, stream);
    };
    up_gate(layer->d_plan);
    ggml_cuda_moe_cache_mmv(down->src[0]->type, q8_mid, layer->d_plan + 2 * rows, nullptr,
                            (float *) down->data, n_ff, n_embd, rows, rows, stream, groups_of(layer->d_plan), (int) n_tokens);
#if defined(GGML_USE_HIP)
    // any hosted block: until the stream is synchronized, its copies to and from pinned host memory (the graph's
    // outputs) are made by a kernel, as for the dispatcher's wait. ROCr keeps one SDMA ring per engine: an output copy
    // that waits for this graph blocks the ring, the cache's fills queued behind it wait too, and the plan of a pass's
    // first layer waits for the previous pass's fills while the GPU waits for that plan (hung at 16k, 1-token passes,
    // a fill in flight and 130 queued; all copies by kernel, GPU_FORCE_BLIT_COPY_SIZE=8192: no hang)
    ctx.split_wait_queued = true;
    ctx.split_wait_captured = true;
#endif
    // SPLIT: the rows of the expert moved to the GPU, once it arrived; the kernels skip every other row
    const bool split_on = moe_split_enabled() && n_tokens >= 2;
    if (split_on) {
        ctx.split_wait_queued = true;
        ctx.split_wait_captured = true;
        const bool by_kernel = moe_split_copy_by_kernel();
        k_moe_split_wait<<<1, 256, 0, stream>>>(layer->d_expect, layer->m_flags + 24, layer->m_plan2, layer->d_plan2, 3 * rows,
                                                by_kernel ? layer->m_fetch : nullptr, layer->d_fetch, g_tl_m);
        if (by_kernel) {
            k_moe_split_fetch<<<192, 256, 0, stream>>>(layer->d_fetch);
        }
        up_gate(layer->d_plan2);
        ggml_cuda_moe_cache_mmv(down->src[0]->type, q8_mid, layer->d_plan2 + 2 * rows, nullptr,
                                (float *) down->data, n_ff, n_embd, rows, rows, stream, groups_of(layer->d_plan2), (int) n_tokens);
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
    // EXPERT SPLIT: every other card's chain for this block: wait for the plan, its rows into the result buffer, its flag
    bool cards = false;
    if (moe_cache_expert_split() && session.devices.size() > 1 && layer->h_plank && slot >= 0 && slot < MOE_SPLIT_MAX_SLOTS) {
        // a plan's chunk list (written after it by the card's wait kernels), as for the hosting card's kernels
        static const bool card_grouped = !getenv("GGML_MOE_SPLIT_GROUPS") || atoi(getenv("GGML_MOE_SPLIT_GROUPS")) != 0;
        const auto card_groups = [&](const uint64_t * plan) -> const int32_t * {
            return card_grouped && n_tokens >= 2 && rows <= MOE_GROUP_MAX_ROWS ? (const int32_t *) (plan + 3 * rows) : nullptr;
        };
        for (moe_split_card & card : st->split_cards) {
            ggml_cuda_set_device(card.device->logical);
            k_split_card_wait<<<1, 256, 0, card.stream>>>(layer->m_flags, card.d_last + slot, card.d_seq,
                                                          layer->m_plank + (size_t) (card.index - 1) * 3 * rows, card.d_plan,
                                                          3 * rows, layer->m_act, card.d_act, (int) (n_tokens * n_embd),
                                                          st->m_split_diag, card.index, slot, g_tlk_m[card.index], (int) n_tokens,
                                                          w_src ? layer->m_w : nullptr, card.d_w, rows);
            quantize_row_q8_1_cuda(card.d_act, nullptr, card.d_q8_act, wtype, n_embd, n_embd, n_tokens * n_embd,
                                   n_tokens * n_embd, padded_embd, n_tokens, 1, 1, card.stream);
            if (q8_fused && n_ff % QK8_1 == 0) {
                ggml_cuda_moe_cache_mmv_fused(wtype, card.d_q8_act, card.d_plan, card.d_plan + rows, card.d_act_ids, card.d_glu,
                                              n_embd, n_ff, rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY,
                                              card.stream, card.d_q8_mid, padded_ff / QK8_1, card.d_q8_count, card_groups(card.d_plan),
                                              (int) n_tokens);
            } else {
                ggml_cuda_moe_cache_mmv_fused(wtype, card.d_q8_act, card.d_plan, card.d_plan + rows, card.d_act_ids, card.d_glu,
                                              n_embd, n_ff, rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY,
                                              card.stream, nullptr, 0, nullptr, card_groups(card.d_plan), (int) n_tokens);
                quantize_row_q8_1_cuda(card.d_glu, nullptr, card.d_q8_mid, wtype, n_ff, n_ff, (int64_t) rows * n_ff,
                                       (int64_t) rows * n_ff, padded_ff, rows, 1, 1, card.stream);
            }
            float * card_out = w_src ? card.d_res : layer->m_res;   // PSUM: summed below
            ggml_cuda_moe_cache_mmv(down->src[0]->type, card.d_q8_mid, card.d_plan + 2 * rows, nullptr, card_out, n_ff,
                                    n_embd, rows, rows, card.stream, card_groups(card.d_plan), (int) n_tokens);
            // the experts the dispatcher moved to this card, once copied (the flag comes without any)
            k_split_card_wait2<<<1, 256, 0, card.stream>>>(layer->m_flags + 34 + 8 * card.index, card.d_seq,
                                                           layer->m_plan2k + (size_t) (card.index - 1) * 3 * rows, card.d_plan2,
                                                           3 * rows, st->m_split_diag, card.index, g_tlk_m[card.index], (int) n_tokens);
            // their rows quantized by the fused kernel as above (a separate quantization went over every row, moved or not)
            static const bool fused2 = !getenv("GGML_MOE_SPLIT_FUSED2") || atoi(getenv("GGML_MOE_SPLIT_FUSED2")) != 0;
            if (fused2 && q8_fused && n_ff % QK8_1 == 0) {
                ggml_cuda_moe_cache_mmv_fused(wtype, card.d_q8_act, card.d_plan2, card.d_plan2 + rows, card.d_act_ids, card.d_glu,
                                              n_embd, n_ff, rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY,
                                              card.stream, card.d_q8_mid, padded_ff / QK8_1, card.d_q8_count, card_groups(card.d_plan2),
                                              (int) n_tokens);
            } else {
                ggml_cuda_moe_cache_mmv_fused(wtype, card.d_q8_act, card.d_plan2, card.d_plan2 + rows, card.d_act_ids, card.d_glu,
                                              n_embd, n_ff, rows, n_tokens, -INFINITY, INFINITY, -INFINITY, INFINITY,
                                              card.stream, nullptr, 0, nullptr, card_groups(card.d_plan2), (int) n_tokens);
                quantize_row_q8_1_cuda(card.d_glu, nullptr, card.d_q8_mid, wtype, n_ff, n_ff, (int64_t) rows * n_ff,
                                       (int64_t) rows * n_ff, padded_ff, rows, 1, 1, card.stream);
            }
            ggml_cuda_moe_cache_mmv(down->src[0]->type, card.d_q8_mid, card.d_plan2 + 2 * rows, nullptr, card_out, n_ff,
                                    n_embd, rows, rows, card.stream, card_groups(card.d_plan2), (int) n_tokens);
            if (w_src) {
                k_split_card_psum<<<(unsigned) n_tokens, 256, 0, card.stream>>>(card.d_plan, card.d_plan2, rows, n_used,
                                                                               card.d_w, card.d_res, layer->m_res, (int) n_embd);
            }
            k_split_card_done<<<1, 1, 0, card.stream>>>(card.d_seq, layer->m_flags + 32 + 8 * card.index, g_tlk_m[card.index]);
            CUDA_CHECK(cudaGetLastError());
            cards = true;
        }
        ggml_cuda_set_device(ctx.device);
    }
    g_hosted_pending = {layer, down, rows, split_on, moe_sgate_find(ctx, cgraph, i, act, down, (int) n_tokens), cards,
                        st->m_split_diag, n_used, cards && w_src != nullptr};
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
    k_moe_hosted_merge<<<p.rows + (p.sg.x ? 1 : 0), 256, 0, ctx.stream()>>>(p.layer->d_expect, p.layer->m_flags + 16,
                                                          p.layer->d_plan + 2 * p.rows, p.split ? p.layer->d_plan2 + 2 * p.rows : nullptr,
                                                          p.layer->m_res,
                                                          (float *) p.down->data, (int) p.layer->n_embd, g_tl_m, p.sg,
                                                          p.cards ? p.layer->m_src : nullptr, p.layer->m_flags,
                                                          p.cards ? p.diag : nullptr, p.n_used, p.psum ? 1 : 0);
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
    ggml_moe_cache.session_enter_guest = moe_cache_session_enter_guest;
    ggml_moe_cache.kv_register = moe_cache_kv_register;
    ggml_moe_cache.kv_needs_set_rows = moe_cache_kv_needs_set_rows;
    ggml_moe_cache.kv_set_rows = moe_cache_kv_set_rows;
    ggml_moe_cache.kv_first_lent_row = moe_cache_kv_first_lent_row;
    ggml_moe_cache.host_experts = moe_cache_host_experts;
    ggml_moe_cache.session_leave = moe_cache_session_leave;
    ggml_moe_cache.begin = moe_cache_begin;
    ggml_moe_cache.assign = moe_cache_assign;
    ggml_moe_cache.assigned = moe_cache_assigned;
    ggml_moe_cache.hosts = moe_cache_hosts;
    ggml_moe_cache.prepare_queues = moe_cache_prepare_queues;
    ggml_moe_cache.hosted_max_rows = moe_hosted_max_rows;
    ggml_moe_cache.largest_lend = moe_cache_largest_lend;
    ggml_moe_cache.lend_room = moe_cache_lend_room;
    ggml_moe_cache.hosted_cost = moe_hosted_cost;
    ggml_moe_cache.plan = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.dispatch = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.collect = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.end = moe_cache_end;
    ggml_moe_cache.fused_begin = nullptr;   // nodes are served by the hosted MoE only (moe_cache_begin)
    ggml_moe_cache.invalidate = moe_cache_invalidate;
    ggml_moe_cache.set_lend_size = moe_cache_set_lend_size;
    ggml_moe_cache.prefill_residents = moe_cache_prefill_residents;
    ggml_moe_cache.prefill_table = moe_cache_prefill_table;
    ggml_moe_cache.prefill_unpin = moe_cache_prefill_unpin;
    ggml_moe_cache.prefill_sources = moe_cache_prefill_sources;
    ggml_moe_cache.lend = moe_cache_lend;
    ggml_moe_cache.restore = moe_cache_restore;
    ggml_moe_cache.ring_zone = moe_cache_ring_zone;
    ggml_moe_cache.borrow = moe_cache_borrow;
    ggml_moe_cache.give_back = moe_cache_give_back;
    ggml_moe_cache.compute_buffer_type = moe_cache_compute_buffer_type;
    ggml_moe_cache.plan_borrows = moe_cache_plan_borrows;
    ggml_moe_cache.borrows = moe_cache_borrows;
}

#endif
