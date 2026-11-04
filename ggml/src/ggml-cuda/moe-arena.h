#pragma once

// The idea of a host RAM tier of hot experts under the VRAM cache comes from buun's expert store
// (spiritbuun/buun-llama-cpp: ggml-moe-store, moe-tier.cu).
//
// The expert arena: anonymous host memory registered with the GPU once at start, in slots large enough for any expert
// read with O_DIRECT (its tensors each in a page-aligned piece of the slot: part[] points at them). With the arena, the
// experts' host copy is the arena, not the page cache: an expert the CPU or a prefill needs that the arena does not hold
// is read from the file into a slot with O_DIRECT (one large read, nothing kept in the page cache); fills copy only
// experts the arena holds.
// Eviction is LRU, as the VRAM cache's: a read or a write puts the slot at the hot end, the victim is the coldest slot
// not being read or written. The VRAM cache's experts are kept like the others (buun's "VRAM first" rule measured
// against: the VRAM cache turns over, the experts it evicts come back through the arena).
// A prefill borrows a zone (the VRAM cache's loans, applied to the arena): lend() takes the coldest slots out of the
// LRU, the experts they held noted; the prefill reads the experts the arena does not hold into them in turn
// (transit_take), copied from there and never admitted; its reads of what the arena holds do not move it (acquire
// without touch). give_back() returns the zone at the start of generation, with the experts to restore.
//
// Not thread-safe: every call but release() under the cache session's lock. release() (a reader done) is atomic.
// Host only, no HIP: tests/arenatest.cpp exercises it with ASan + UBSan.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename Key, typename Hash>
struct moe_arena {
    struct slot {
        Key key{};
        int pool = -1;
        bool valid = false;     // its bytes are the expert's
        bool writing = false;   // taken: being written (then copied to the device by a fill)
        bool lent = false;      // in the zone lent to a prefill: neither in the LRU nor free
        const char * part[3] = {};   // where each tensor of the expert starts in the slot (set by the writer)
        std::atomic<int> readers{0};
        int prev = -1, next = -1;   // LRU list of the valid and written slots: head the coldest
    };

    char * base = nullptr;
    size_t slot_bytes = 0;
    int n_slots = 0;
    std::unique_ptr<slot[]> slots;
    std::unordered_map<Key, int, Hash> map;   // valid slots and slots being written, by key (the zone's too)
    std::vector<int> free_list;
    int lru_head = -1, lru_tail = -1;
    bool evict = true;   // false: once full, the arena keeps what it holds (take() returns -1)
    // the zone lent to a prefill: its slots, the next one to try, the experts the lending evicted (coldest first)
    int zone_slots = 0;   // the zone's size, set once (the experts of the layers a prefill has in flight)
    std::vector<int> zone;
    bool zone_active = false;   // lent (between lend() and give_back())
    size_t zone_next = 0;
    std::vector<std::pair<Key, int>> zone_evicted;
    // counters
    uint64_t hits = 0, writes = 0, evictions = 0, busy = 0, cpu_hits = 0, cpu_loads = 0, prefill_loads = 0, load_failures = 0,
             refused = 0,      // admissions of an expert the arena does not hold
             zone_busy = 0,    // a prefill's read with every slot of the zone in use
             restored = 0;     // experts read back after a zone was given back

    void init(char * b, size_t bytes_per_slot, int n) {
        base = b;
        slot_bytes = bytes_per_slot;
        n_slots = n;
        slots.reset(n > 0 ? new slot[n] : nullptr);
        map.clear();
        free_list.clear();
        for (int i = n - 1; i >= 0; i--) {
            free_list.push_back(i);
        }
        lru_head = lru_tail = -1;
        zone.clear();
        zone_active = false;
        zone_next = 0;
        zone_evicted.clear();
    }

    bool active() const { return n_slots > 0; }

    char * ptr(int s) const { return base + (size_t) s * slot_bytes; }

    bool in_use(int s) const { return slots[s].writing || slots[s].readers.load(std::memory_order_acquire) != 0; }

    void lru_remove(int s) {
        slot & x = slots[s];
        (x.prev >= 0 ? slots[x.prev].next : lru_head) = x.next;
        (x.next >= 0 ? slots[x.next].prev : lru_tail) = x.prev;
        x.prev = x.next = -1;
    }
    void lru_push_hot(int s) {
        slot & x = slots[s];
        x.prev = lru_tail;
        x.next = -1;
        (lru_tail >= 0 ? slots[lru_tail].next : lru_head) = s;
        lru_tail = s;
    }
    void clear_slot(slot & x) {
        x.key = Key{};
        x.pool = -1;
        x.valid = false;
        x.writing = false;
        x.part[0] = x.part[1] = x.part[2] = nullptr;
    }

    // a valid slot of the key, a reader taken (release() when done), or -1; touch: to the hot end (not for a prefill)
    int acquire(const Key & key, bool touch = true) {
        const auto it = map.find(key);
        if (it == map.end() || !slots[it->second].valid) {
            return -1;
        }
        slot & x = slots[it->second];
        x.readers.fetch_add(1, std::memory_order_relaxed);
        if (touch && !x.lent) {
            lru_remove(it->second);
            lru_push_hot(it->second);
        }
        return it->second;
    }

    void release(int s) { slots[s].readers.fetch_sub(1, std::memory_order_release); }

    // a slot to write key into, marked writing (commit() or abort() when the write is done), or -1 when the key is
    // already there or being written, or when every slot is busy (or full without eviction)
    int take(const Key & key, int pool) {
        if (map.count(key)) {
            return -1;
        }
        int s = -1;
        if (!free_list.empty()) {
            s = free_list.back();
            free_list.pop_back();
        } else if (!evict) {
            busy++;
            return -1;
        } else {
            for (int c = lru_head; c >= 0; c = slots[c].next) {
                if (!in_use(c)) {
                    s = c;
                    break;
                }
            }
            if (s < 0) {
                busy++;
                return -1;
            }
            lru_remove(s);
            map.erase(slots[s].key);
            evictions++;
        }
        slot & x = slots[s];
        clear_slot(x);
        x.key = key;
        x.pool = pool;
        x.writing = true;
        map[key] = s;
        lru_push_hot(s);
        writes++;
        return s;
    }

    void commit(int s) {
        slots[s].writing = false;
        slots[s].valid = true;
    }

    void abort(int s) {
        slot & x = slots[s];
        map.erase(x.key);
        clear_slot(x);
        if (!x.lent) {
            lru_remove(s);
            free_list.push_back(s);
        }
    }

    // the zone for a prefill, zone_slots of the free then coldest slots not in use (with those a give_back() kept);
    // what they held is noted for give_back(). Nothing when it is already lent. Returns the zone's size
    int lend() {
        if (zone_active) {
            return (int) zone.size();
        }
        zone_active = true;
        while ((int) zone.size() < zone_slots && !free_list.empty()) {
            const int s = free_list.back();
            free_list.pop_back();
            slots[s].lent = true;
            zone.push_back(s);
        }
        for (int c = lru_head; c >= 0 && (int) zone.size() < zone_slots;) {
            const int next = slots[c].next;
            if (!in_use(c)) {
                slot & x = slots[c];
                lru_remove(c);
                map.erase(x.key);
                zone_evicted.push_back({x.key, x.pool});
                clear_slot(x);
                x.lent = true;
                zone.push_back(c);
                evictions++;
            }
            c = next;
        }
        zone_next = 0;
        return (int) zone.size();
    }

    // a slot of the zone to read key into (writing; commit() or abort() when done), its previous expert dropped; -1
    // when the key is there or being written, or when every slot of the zone is in use
    int transit_take(const Key & key, int pool) {
        if (map.count(key)) {
            return -1;
        }
        for (size_t i = 0; i < zone.size(); i++) {
            const size_t k = (zone_next + i) % zone.size();
            const int s = zone[k];
            if (in_use(s)) {
                continue;
            }
            slot & x = slots[s];
            if (x.valid) {
                map.erase(x.key);
            }
            clear_slot(x);
            x.key = key;
            x.pool = pool;
            x.writing = true;
            map[key] = s;
            zone_next = (k + 1) % zone.size();
            writes++;
            return s;
        }
        zone_busy++;
        return -1;
    }

    // the zone back to the free list (a slot still in use stays lent until the next call), and the experts the lending
    // evicted, hottest first, to restore
    std::vector<std::pair<Key, int>> give_back() {
        std::vector<int> kept;
        for (const int s : zone) {
            if (in_use(s)) {
                kept.push_back(s);
                continue;
            }
            slot & x = slots[s];
            if (x.valid) {
                map.erase(x.key);
            }
            clear_slot(x);
            x.lent = false;
            free_list.push_back(s);
        }
        zone = kept;
        zone_active = false;
        zone_next = 0;
        std::vector<std::pair<Key, int>> out(zone_evicted.rbegin(), zone_evicted.rend());
        zone_evicted.clear();
        return out;
    }
};
