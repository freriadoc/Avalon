
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
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "ASSERT %s line %d\n", #x, __LINE__); abort(); } } while (0)
enum class moe_cache_slot_state : unsigned char { free, copying, valid };
struct moe_cache_key { const void * tensor=nullptr; int32_t expert=-1;
  bool operator==(const moe_cache_key&o)const{return tensor==o.tensor&&expert==o.expert;} };
struct moe_cache_key_hash { size_t operator()(const moe_cache_key&k)const{return ((size_t)(uintptr_t)k.tensor) ^ ((size_t)k.expert*0x9e3779b97f4a7c15ULL); } };
struct moe_cache_slot {
    moe_cache_key key;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    moe_cache_slot_state state = moe_cache_slot_state::free;
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
    int zone_begin = INT_MAX;   // >= n_slots when the pool has no zone
    bool zone_lent = false;
};

static inline bool moe_cache_pool_has_zone(const moe_cache_pool & pool) {
    return pool.zone_begin < pool.n_slots;
}


struct moe_cache_device { cudaStream_t zone_stream=nullptr; long long zone_moves=0, zone_evictions=0; };
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

static void moe_cache_zone_evacuate(
        moe_cache_device & device, moe_cache_pool & pool, std::vector<int> & moved) {
    const int b = pool.zone_begin;
    const int zone_slots = pool.n_slots - b;

    // 1. free list: slots outside the zone become holes, slots inside leave with it
    std::vector<int> holes;
    holes.reserve(pool.free_slots.size() + (size_t)zone_slots);
    const size_t free_total = pool.free_slots.size();
    for (int index : pool.free_slots) {
        if (index < b) {
            holes.push_back(index);
        }
    }
    pool.free_slots.clear();

    // 2. evict the coldest entries, from the LRU head, until the pool has as many free
    //    slots as the zone has slots
    int to_evict = zone_slots > (int)free_total ? zone_slots - (int)free_total : 0;
    while (to_evict > 0 && pool.lru_head >= 0) {
        const int index = pool.lru_head;
        moe_cache_slot_reset(pool, index, false);   // unlinks the head, erases the key
        if (index < b) {
            holes.push_back(index);
        }
        device.zone_evictions++;
        to_evict--;
    }

    // 3. move the zone entries that survived into the holes, keeping their LRU place
    for (int source = b; source < pool.n_slots; source++) {
        moe_cache_slot & src = pool.slots[source];
        if (src.state != moe_cache_slot_state::valid) {
            continue;
        }
        GGML_ASSERT(!holes.empty() && "zone evacuation: not enough holes");
        const int destination = holes.back();
        holes.pop_back();

        cudaMemcpyAsync(pool.slab + (size_t)destination * pool.expert_size,
                pool.slab + (size_t)source * pool.expert_size,
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


static const size_t ES = 48;
static std::vector<char> g_host; static const void * HOST=nullptr;

static bool admit(moe_cache_pool & pool, int expert) {
    const moe_cache_key key{HOST, expert};
    int i;
    if (!pool.free_slots.empty()) { i = pool.free_slots.back(); pool.free_slots.pop_back(); }
    else { i = pool.lru_head; while (i >= 0 && pool.slots[i].readers > 0) i = pool.slots[i].next;
           if (i < 0) return false; moe_cache_slot_reset(pool, i, false); }
    GGML_ASSERT(!(pool.zone_lent && i >= pool.zone_begin));   // never admit into a lent zone
    auto & sl = pool.slots[i]; sl.key = key; sl.generation++; sl.state = moe_cache_slot_state::copying;
    if (!pool.map.emplace(key, i).second) { moe_cache_slot_reset(pool, i, true); return false; }
    memcpy(pool.slab + (size_t)i*ES, (const char*)HOST + (size_t)expert*ES, ES);
    sl.state = moe_cache_slot_state::valid; moe_cache_lru_push_back(pool, i); return true;
}
static bool hit(moe_cache_pool & pool, int expert) {
    auto it = pool.map.find(moe_cache_key{HOST, expert});
    if (it == pool.map.end()) return false;
    moe_cache_lru_remove(pool, it->second); moe_cache_lru_push_back(pool, it->second); return true;
}
static void restore(moe_cache_pool & pool) {
    for (int i = pool.zone_begin; i < pool.n_slots; i++) pool.free_slots.push_back(i);
    pool.zone_lent = false;
}
static int check(moe_cache_pool & pool, const char * where) {
    int e = 0; std::vector<int> seen(pool.n_slots, 0); int valid = 0;
    for (auto & sl : pool.slots) if (sl.state == moe_cache_slot_state::valid) valid++;
    int c = 0;
    for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
        if (++c > pool.n_slots) { printf("%s: cycle\n", where); return 1; }
        if (pool.slots[i].state != moe_cache_slot_state::valid) { printf("%s: non-valid in LRU %d\n", where, i); e++; }
        if (pool.zone_lent && i >= pool.zone_begin) { printf("%s: lent slot %d in LRU\n", where, i); e++; }
        if (seen[i]++) { printf("%s: dup %d\n", where, i); e++; }
    }
    int back = 0; for (int i = pool.lru_tail; i >= 0; i = pool.slots[i].prev) back++;
    if (back != c || c != valid) { printf("%s: LRU count %d back %d valid %d\n", where, c, back, valid); e++; }
    for (int i : pool.free_slots) {
        if (pool.zone_lent && i >= pool.zone_begin) { printf("%s: lent slot %d free-listed\n", where, i); e++; }
        if (seen[i]++) { printf("%s: slot %d free and used\n", where, i); e++; }
    }
    for (int i = 0; i < pool.n_slots; i++) {
        const bool zone = pool.zone_lent && i >= pool.zone_begin;
        if (!seen[i] && !zone) { printf("%s: slot %d lost\n", where, i); e++; }
        if (zone && pool.slots[i].state != moe_cache_slot_state::free) { printf("%s: zone slot %d not free\n", where, i); e++; }
    }
    if (pool.map.size() != (size_t)valid) { printf("%s: map %zu valid %d\n", where, pool.map.size(), valid); e++; }
    for (auto & kv : pool.map) {
        if (!(pool.slots[kv.second].key == kv.first)) { printf("%s: map/key mismatch\n", where); e++; }
        if (pool.zone_lent && kv.second >= pool.zone_begin) { printf("%s: map into lent zone\n", where); e++; }
        if (memcmp(pool.slab + (size_t)kv.second*ES, (const char*)HOST + (size_t)kv.first.expert*ES, ES)) {
            printf("%s: WRONG WEIGHTS expert %d slot %d\n", where, kv.first.expert, kv.second); e++; }
    }
    return e;
}
// ---- several pools: moe_cache_setup_zone and the pool choice of moe_cache_lend ----
// one zone per pool, at most half of a pool, the first n_zones eligible pools in order
static void setup_zones(std::vector<moe_cache_pool> & pools, size_t zone_bytes, int n_zones) {
    int equipped = 0;
    for (moe_cache_pool & pool : pools) {
        if (equipped >= n_zones) break;
        if (moe_cache_pool_has_zone(pool)) { equipped++; continue; }
        const size_t zone_slots = (zone_bytes + pool.expert_size - 1) / pool.expert_size;
        if (!zone_slots || zone_slots > (size_t)pool.n_slots / 2) continue;
        pool.zone_begin = pool.n_slots - (int)zone_slots;
        equipped++;
    }
}
// up to n_zones pools with a free zone of at least zone_bytes, or none below two
static std::vector<moe_cache_pool *> choose_lend(std::vector<moe_cache_pool> & pools, size_t bytes, int n_zones) {
    std::vector<moe_cache_pool *> chosen;
    for (moe_cache_pool & pool : pools) {
        if (moe_cache_pool_has_zone(pool) && !pool.zone_lent &&
                (size_t)(pool.n_slots - pool.zone_begin) * pool.expert_size >= bytes) {
            chosen.push_back(&pool);
            if ((int)chosen.size() == n_zones) break;
        }
    }
    if (chosen.size() < 2) chosen.clear();
    return chosen;
}

static int multi_pool(int seed) {
    srand(seed);
    int errors = 0;
    long long lends = 0, partial = 0;
    for (int t = 0; t < 2000 && !errors; t++) {
        // 1 to 4 pools, entries of 1 to 4 units; a zone of 'zb' bytes
        const int n_pools = 1 + rand() % 4;
        const size_t zb = (size_t)(1 + rand() % 24) * ES;
        std::vector<std::vector<char>> slabs(n_pools);
        std::vector<moe_cache_pool> pools(n_pools);
        for (int k = 0; k < n_pools; k++) {
            moe_cache_pool & pool = pools[k];
            pool.expert_size = ES;   // admit/check copy ES bytes per entry
            pool.n_slots = 8 + rand() % 60;
            slabs[k].resize((size_t)pool.n_slots * ES);
            pool.slab = slabs[k].data();
            pool.slots.resize(pool.n_slots);
            for (int i = pool.n_slots - 1; i >= 0; i--) pool.free_slots.push_back(i);
        }
        const int want = 2 + rand() % 3;
        setup_zones(pools, zb, want);
        int eligible = 0, zoned = 0;
        for (moe_cache_pool & pool : pools) {
            const size_t zs = (zb + ES - 1) / ES;
            eligible += zs <= (size_t)pool.n_slots / 2;
            zoned += moe_cache_pool_has_zone(pool);
            if (moe_cache_pool_has_zone(pool) && pool.n_slots - pool.zone_begin > pool.n_slots / 2) {
                printf("case %d: zone above half a pool\n", t); errors++;
            }
        }
        if (zoned != std::min(want, eligible)) {
            printf("case %d: %d zones for %d wanted, %d eligible\n", t, zoned, want, eligible); errors++;
        }
        // setting up again with the same request changes nothing (idempotent)
        std::vector<int> before; for (auto & p : pools) before.push_back(p.zone_begin);
        setup_zones(pools, zb, want);
        for (int k = 0; k < n_pools; k++) if (pools[k].zone_begin != before[k]) { printf("case %d: not idempotent\n", t); errors++; }

        moe_cache_device dev;
        for (int round = 0; round < 20 && !errors; round++) {
            for (moe_cache_pool & pool : pools)
                for (int s = rand() % 200; s > 0; s--) { const int x = rand() % 150; if (!hit(pool, x)) admit(pool, x); }
            std::vector<moe_cache_pool *> chosen = choose_lend(pools, zb, want);
            const int expect = zoned >= 2 ? zoned : 0;
            if ((int)chosen.size() != expect) { printf("case %d: lent %zu, expected %d\n", t, chosen.size(), expect); errors++; break; }
            if (chosen.empty()) break;
            partial += (int)chosen.size() < want;
            for (moe_cache_pool * pool : chosen) {
                std::vector<int> moved;
                moe_cache_zone_evacuate(dev, *pool, moved);
                pool->zone_lent = true;
                memset(pool->slab + (size_t)pool->zone_begin * ES, 0xEE, (size_t)(pool->n_slots - pool->zone_begin) * ES);
            }
            lends++;
            for (moe_cache_pool & pool : pools) errors += check(pool, "multi lent");
            // a second lend while lent finds nothing left to lend
            if (!choose_lend(pools, zb, want).empty()) { printf("case %d: lent twice\n", t); errors++; }
            for (moe_cache_pool & pool : pools)
                for (int s = 0; s < 10; s++) { const int x = rand() % 150; if (!hit(pool, x)) admit(pool, x); }
            for (moe_cache_pool * pool : chosen) restore(*pool);
            for (moe_cache_pool & pool : pools) errors += check(pool, "multi restored");
        }
    }
    printf("multi-pool: lends=%lld partial=%lld errors=%d\n", lends, partial, errors);
    return errors;
}

int main(int argc, char ** argv) {
    const int N = 64, Z = 12;
    g_host.resize(4096*ES); for (int x = 0; x < 4096; x++) memset(&g_host[(size_t)x*ES], 1 + x % 251, ES); HOST = g_host.data();
    if (multi_pool(argc > 1 ? atoi(argv[1]) : 1)) {
        return 1;
    }
    std::vector<char> slab(N*ES);
    moe_cache_pool pool; pool.expert_size = ES; pool.slab = slab.data(); pool.n_slots = N;
    pool.slots.resize(N); for (int i = N-1; i >= 0; i--) pool.free_slots.push_back(i);
    pool.zone_begin = N - Z;
    moe_cache_device dev; int errors = 0; srand(argc > 1 ? atoi(argv[1]) : 1);
    long long lends = 0;
    for (int round = 0; round < 3000; round++) {
        // generation phase: random admissions / hits, sometimes leaving free slots
        const int steps = rand() % 300;
        for (int s = 0; s < steps; s++) { const int x = rand() % 150; if (!hit(pool, x)) admit(pool, x); }
        if ((rand() % 5) == 0) {                     // punch random holes (invalidations)
            for (int h = rand() % 20; h > 0; h--) { const int i = rand() % N;
                if (pool.slots[i].state == moe_cache_slot_state::valid) moe_cache_slot_reset(pool, i, true); } }
        errors += check(pool, "before lend");
        std::vector<int> moved;
        moe_cache_zone_evacuate(dev, pool, moved); pool.zone_lent = true; lends++;
        memset(pool.slab + (size_t)pool.zone_begin*ES, 0xEE, (size_t)Z*ES);   // prefetch scribbles the zone
        errors += check(pool, "lent");
        // prefill last layer may still admit / hit while lent
        for (int s = 0; s < 20; s++) { const int x = rand() % 150; if (!hit(pool, x)) admit(pool, x); }
        errors += check(pool, "lent+admit");
        restore(pool);
        errors += check(pool, "restored");
        if (errors) break;
    }
    printf("lends=%lld moves=%lld evictions=%lld errors=%d\n", lends, dev.zone_moves, dev.zone_evictions, errors);
    return errors != 0;
}
