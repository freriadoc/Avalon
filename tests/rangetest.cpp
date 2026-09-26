
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
typedef int cudaError_t; static const cudaError_t cudaSuccess=0;
struct CUstream_st; typedef CUstream_st* cudaStream_t;
enum cudaMemcpyKind { cudaMemcpyDeviceToDevice, cudaMemcpyDeviceToHost };
static cudaError_t cudaMemcpyAsync(void*d,const void*s,size_t n,cudaMemcpyKind,cudaStream_t){memmove(d,s,n);return cudaSuccess;}
static cudaError_t cudaMalloc(void**p,size_t n){*p=aligned_alloc(4096,n);return *p?cudaSuccess:1;}
static cudaError_t cudaFree(void*p){free(p);return cudaSuccess;}
static cudaError_t cudaGetLastError(){return cudaSuccess;}
static void ggml_cuda_set_device(int){}
enum ggml_type { GGML_TYPE_F32 };
static bool ggml_type_is_exl3(int){return false;}
static const char * ggml_type_name(int){return "iq4_nl";}
static size_t ggml_moe_cache_max_pool_slots(int, size_t){return 40000;}
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr,"ASSERT %s:%d\n", #x, __LINE__); abort(); } } while (0)
#define GGML_ABORT(...) do { fprintf(stderr, __VA_ARGS__); fprintf(stderr,"\n"); abort(); } while (0)
#define MOE_CACHE_LOG(...) do { if (getenv("V")) printf(__VA_ARGS__); } while (0)
enum class moe_cache_slot_state : unsigned char { free, copying, valid };
struct moe_cache_key { const void * tensor=nullptr; int32_t expert=-1;
  bool operator==(const moe_cache_key&o)const{return tensor==o.tensor&&expert==o.expert;} };
struct moe_cache_key_hash { size_t operator()(const moe_cache_key&k)const{return ((size_t)(uintptr_t)k.tensor)^((size_t)k.expert*0x9e3779b9u);} };
struct moe_cache_slot { moe_cache_key key; unsigned long long generation=0; int prev=-1,next=-1,readers=0;
  moe_cache_slot_state state=moe_cache_slot_state::free; };
// one contiguous piece of device memory holding logical slots [first, first + count)
struct moe_cache_range {
    char * base = nullptr;
    int first = 0;
    int count = 0;
    int64_t gpu_offset = 0;   // gpu index of slot 'first'
};

struct moe_cache_pool {
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;
    bool covers_all_entries = false;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1;
    int lru_tail = -1;

    // Tail of the slab that can be lent to the scheduler's expert prefetch during
    // prefill. While lent, its slots are free, outside the free list, the LRU and the
    // map, so nothing in the cache can reach them.
    int zone_begin = INT_MAX;   // lendable zone, inside range 0
    int zone_end = 0;
    int zone_units = 0;         // prefetch slots this zone can hold
    int zone_unit_slots = 0;    // entries per prefetch slot
    bool zone_lent = false;

    // Address translation. Logical slots are spread over one or more ranges of device
    // memory: range 0 is the slab, the others are borrowed (from the compute buffer).
    // Kernels only take one base and an index, so every range is addressed from a
    // virtual base, low enough and aligned so that each entry sits at
    // gpu_base + k * expert_size.
    std::vector<moe_cache_range> ranges;
    char * gpu_base = nullptr;
    int64_t gpu_extent = 0;     // largest gpu index + 1, passed to kernels as a bound
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

static inline bool moe_cache_pool_has_zone(const moe_cache_pool & pool) {
    return pool.zone_begin < pool.zone_end;
}

#include <memory>

struct moe_cache_device {
    int physical=0, logical=0;
    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<char *> owned_regions;
    cudaStream_t zone_stream=nullptr;
    long long zone_moves=0, zone_evictions=0;
};
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

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
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
    slot.state = moe_cache_slot_state::free;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free) {
        pool.free_slots.push_back(index);
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
    if (offset + count > (int64_t)ggml_moe_cache_max_pool_slots(pool.wtype, es)) {
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

static void moe_cache_zone_evacuate(
        moe_cache_device & device, moe_cache_pool & pool, std::vector<int> & moved) {
    const int b = pool.zone_begin;
    const int zone_slots = pool.zone_end - b;

    const auto in_zone = [&](int index) { return index >= b && index < pool.zone_end; };

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
        moe_cache_slot_reset(pool, index, false);   // unlinks the head, erases the key
        if (in_zone(index)) {
            occupied--;
        } else {
            holes.push_back(index);
        }
        device.zone_evictions++;
    }

    // 3. move the zone entries that survived into the holes, keeping their LRU place
    for (int source = b; source < pool.zone_end; source++) {
        moe_cache_slot & src = pool.slots[source];
        if (src.state != moe_cache_slot_state::valid) {
            continue;
        }
        GGML_ASSERT(!holes.empty() && "zone evacuation: not enough holes");
        const int destination = holes.back();
        holes.pop_back();

        cudaMemcpyAsync(moe_cache_slot_ptr(pool, destination),
                moe_cache_slot_ptr(pool, source),
                pool.expert_size, cudaMemcpyDeviceToDevice, device.zone_stream);

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


// ---------------- driver ----------------
static const size_t ES = 48;
static std::vector<char> g_host;
static const void * HOST = nullptr;

static bool admit(moe_cache_pool & pool, int expert) {
    const moe_cache_key key{HOST, expert};
    int i;
    if (!pool.free_slots.empty()) { i = pool.free_slots.back(); pool.free_slots.pop_back(); }
    else { i = pool.lru_head; while (i >= 0 && pool.slots[i].readers > 0) i = pool.slots[i].next;
           if (i < 0) return false; moe_cache_slot_reset(pool, i, false); }
    GGML_ASSERT(!(pool.zone_lent && i >= pool.zone_begin && i < pool.zone_end));
    auto & sl = pool.slots[i]; sl.key = key; sl.generation++; sl.state = moe_cache_slot_state::copying;
    if (!pool.map.emplace(key, i).second) { moe_cache_slot_reset(pool, i, true); return false; }
    memcpy(moe_cache_slot_ptr(pool, i), (const char*)HOST + (size_t)expert*ES, ES);  // the fill
    sl.state = moe_cache_slot_state::valid; moe_cache_lru_push_back(pool, i); return true;
}
static bool hit(moe_cache_pool & pool, int expert) {
    auto it = pool.map.find(moe_cache_key{HOST, expert});
    if (it == pool.map.end()) return false;
    // what a kernel would read: gpu_base + gpu_index * expert_size
    const char * addr = pool.gpu_base + (size_t)moe_cache_gpu_index(pool, it->second) * ES;
    GGML_ASSERT(memcmp(addr, (const char*)HOST + (size_t)expert*ES, ES) == 0);
    moe_cache_lru_remove(pool, it->second); moe_cache_lru_push_back(pool, it->second); return true;
}
static int check(moe_cache_pool & pool, const char * where) {
    int e = 0, valid = 0, c = 0;
    std::vector<int> seen(pool.n_slots, 0);
    for (auto & sl : pool.slots) if (sl.state == moe_cache_slot_state::valid) valid++;
    for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
        if (++c > pool.n_slots) { printf("%s: cycle\n", where); return 1; }
        if (seen[i]++) { printf("%s: dup %d\n", where, i); e++; }
    }
    int back = 0; for (int i = pool.lru_tail; i >= 0; i = pool.slots[i].prev) back++;
    if (back != c || c != valid) { printf("%s: LRU %d back %d valid %d\n", where, c, back, valid); e++; }
    for (int i : pool.free_slots) if (seen[i]++) { printf("%s: %d free+used\n", where, i); e++; }
    for (int i = 0; i < pool.n_slots; i++) {
        const bool lent = pool.zone_lent && i >= pool.zone_begin && i < pool.zone_end;
        if (!seen[i] && !lent) { printf("%s: slot %d lost\n", where, i); e++; }
    }
    for (auto & kv : pool.map) {
        if (memcmp(moe_cache_slot_ptr(pool, kv.second), (const char*)HOST + (size_t)kv.first.expert*ES, ES)) {
            printf("%s: WRONG WEIGHTS expert %d slot %d\n", where, kv.first.expert, kv.second); e++; }
        if (pool.zone_lent && kv.second >= pool.zone_begin && kv.second < pool.zone_end) {
            printf("%s: map into lent zone\n", where); e++; }
    }
    // every gpu index must be unique and inside the extent
    std::unordered_map<int64_t,int> gpu;
    for (int i = 0; i < pool.n_slots; i++) {
        const int64_t g = moe_cache_gpu_index(pool, i);
        if (g < 0 || g >= pool.gpu_extent) { printf("%s: gpu index %lld out of extent\n", where, (long long)g); e++; }
        if (!gpu.emplace(g, i).second) { printf("%s: gpu index %lld twice\n", where, (long long)g); e++; }
        if (moe_cache_slot_ptr(pool, i) != pool.gpu_base + (size_t)g*ES) { printf("%s: ptr mismatch\n", where); e++; }
    }
    return e;
}
int main(int argc, char ** argv) {
    const int N = 64, Z = 12;
    g_host.resize(8192*ES);
    for (int x = 0; x < 8192; x++) memset(&g_host[(size_t)x*ES], 1 + x % 251, ES);
    HOST = g_host.data();
    moe_cache_device dev;
    srand(argc > 1 ? atoi(argv[1]) : 1);

    // slab, plus two extra regions: one above, one below (the virtual base must move)
    // one arena: the three pieces must stay close, as they are on a real device
    char * arena = (char *)aligned_alloc(4096, 1 << 20);
    char * below = arena + 11;              // misaligned, under the slab
    char * slab  = arena + 64*1024;
    char * above = arena + 256*1024 + 37;   // misaligned, over the slab
    auto pool_ptr = std::unique_ptr<moe_cache_pool>(new moe_cache_pool());
    moe_cache_pool & pool = *pool_ptr;
    pool.expert_size = ES; pool.slab = slab; pool.n_slots = N; pool.slots.resize(N);
    for (int i = N-1; i >= 0; i--) pool.free_slots.push_back(i);
    pool.gpu_base = slab; pool.gpu_extent = N; pool.zone_end = N; pool.zone_begin = N - Z; pool.zone_units = 2; pool.zone_unit_slots = Z/2;
    pool.ranges.push_back({slab, 0, N, 0});
    dev.pools.push_back(std::move(pool_ptr));

    printf("attach above: %zu B, below: %zu B\n",
           moe_cache_pool_attach_range(pool, above, 40*ES),
           moe_cache_pool_attach_range(pool, below, 24*ES));
    printf("slots=%d ranges=%zu extent=%lld\n", pool.n_slots, pool.ranges.size(), (long long)pool.gpu_extent);
    int errors = check(pool, "after attach");

    for (int round = 0; round < 2000 && !errors; round++) {
        for (int s = 0, n = rand() % 300; s < n; s++) { const int x = rand() % 200; if (!hit(pool, x)) admit(pool, x); }
        if ((rand() % 5) == 0) for (int h = rand() % 20; h > 0; h--) {
            const int i = rand() % pool.n_slots;
            if (pool.slots[i].state == moe_cache_slot_state::valid) moe_cache_slot_reset(pool, i, true); }
        errors += check(pool, "generation");
        std::vector<int> moved;
        moe_cache_zone_evacuate(dev, pool, moved); pool.zone_lent = true;
        memset(moe_cache_slot_ptr(pool, pool.zone_begin), 0xEE, (size_t)Z*ES);   // the prefetch writes
        errors += check(pool, "lent");
        for (int i = pool.zone_begin; i < pool.zone_end; i++) pool.free_slots.push_back(i);
        pool.zone_lent = false;
        errors += check(pool, "restored");
    }
    printf("moves=%lld evictions=%lld errors=%d\n", dev.zone_moves, dev.zone_evictions, errors);
    return errors != 0;
}
