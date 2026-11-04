// Bench of the expert arena (ggml/src/ggml-cuda/moe-arena.h): random writes, readers held across operations, a zone
// lent to a "prefill" and given back. Invariants after every operation: map <-> slots, every slot in exactly one of the
// free list, the LRU list (linked both ways) and the zone, lent <-> in the zone, a slot's bytes are its key's, a slot
// read or written is never taken, the victim is the coldest slot not busy, a read or a write puts its slot at the hot
// end (a read without touch leaves it where it is), a transit write takes a slot of the zone, give_back() frees the
// zone
// but its slots in use and returns what the lending evicted, hottest first.
// c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I ggml/src/ggml-cuda tests/arenatest.cpp -o arenatest && ./arenatest

#include "moe-arena.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct key_t_ {
    const void * tensor = nullptr;
    int32_t expert = -1;
    bool operator==(const key_t_ & o) const { return tensor == o.tensor && expert == o.expert; }
};
struct key_hash {
    size_t operator()(const key_t_ & k) const { return std::hash<const void *>()(k.tensor) ^ ((size_t) k.expert * 0x9e3779b97f4a7c15ULL); }
};

static uint8_t byte_of(const key_t_ & k, size_t i) {
    return (uint8_t) (((uintptr_t) k.tensor >> 4) * 131 + (uint32_t) k.expert * 31 + i * 7);
}

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s (seed %u, op %d)\n", __FILE__, __LINE__, #c, seed, op); exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 200;
    const int ops = argc > 2 ? atoi(argv[2]) : 20000;
    uint64_t total = 0, lends = 0, transits = 0, gives = 0;
    for (unsigned seed = 1; seed <= (unsigned) seeds; seed++) {
        std::mt19937 rng(seed);
        const int n_slots = 1 + rng() % 200;
        const size_t slot_bytes = 64 + rng() % 256;
        std::vector<char> mem((size_t) n_slots * slot_bytes);
        moe_arena<key_t_, key_hash> a;
        a.init(mem.data(), slot_bytes, n_slots);
        a.zone_slots = (int) (rng() % (n_slots / 2 + 1));
        const int n_tensors = 1 + rng() % 8, n_experts = 1 + rng() % 600;
        static char tensors[8];
        struct held { int s; key_t_ k; bool writer; };
        std::vector<held> hold;
        std::vector<std::pair<key_t_, int>> lend_evicted;   // what the current lending evicted (the bench's copy)
        int op = 0;
        const auto rand_key = [&] { return key_t_{&tensors[rng() % n_tensors], (int32_t) (rng() % n_experts)}; };
        const auto lru = [&] {
            std::vector<int> v;
            for (int c = a.lru_head; c >= 0; c = a.slots[c].next) v.push_back(c);
            return v;
        };
        const auto busy_slot = [&](int c) { return a.slots[c].writing || a.slots[c].readers.load() != 0; };
        const auto in_zone = [&](int s) { return std::find(a.zone.begin(), a.zone.end(), s) != a.zone.end(); };
        for (op = 0; op < ops; op++) {
            const int r = rng() % 100;
            if (r < 30) {   // a write
                const key_t_ k = rand_key();
                const int had = a.map.count(k) ? a.map.at(k) : -1;
                const std::vector<int> before = lru();
                const uint64_t ev = a.evictions;
                const int s = a.take(k, (int) (rng() % 3));
                if (s >= 0) {
                    CHECK(had < 0);
                    for (const auto & h : hold) CHECK(h.s != s);   // never a busy slot
                    CHECK(a.lru_tail == s && !in_zone(s));
                    if (a.evictions != ev) {   // the victim: the coldest slot not busy
                        for (int c : before) {
                            if (c == s) break;
                            CHECK(busy_slot(c));
                        }
                    }
                    for (size_t i = 0; i < slot_bytes; i++) a.ptr(s)[i] = (char) byte_of(k, i);
                    hold.push_back({s, k, true});
                }
            } else if (r < 50) {   // a reader (CPU, a fill, a prefill's copy)
                const key_t_ k = rand_key();
                const bool touch = rng() % 2;
                const std::vector<int> before = lru();
                const int s = a.acquire(k, touch);
                if (s >= 0) {
                    CHECK(a.slots[s].valid && a.slots[s].key == k);
                    if (touch && !a.slots[s].lent) CHECK(a.lru_tail == s); else CHECK(lru() == before);
                    hold.push_back({s, k, false});
                }
            } else if (r < 75) {   // one done
                if (!hold.empty()) {
                    const size_t i = rng() % hold.size();
                    const held h = hold[i];
                    hold.erase(hold.begin() + i);
                    for (size_t b = 0; b < slot_bytes; b++) CHECK((uint8_t) a.ptr(h.s)[b] == byte_of(h.k, b));
                    if (h.writer) {
                        if (rng() % 20 == 0) a.abort(h.s); else a.commit(h.s);
                    } else {
                        a.release(h.s);
                    }
                }
            } else if (r < 81) {   // a prefill lends a zone
                if (!a.zone_active) {
                    std::vector<int> cold;   // what the lending may take, in order: the free slots, then the LRU's
                    for (int c : lru()) if (!busy_slot(c)) cold.push_back(c);
                    const size_t kept = a.zone.size(), free_before = a.free_list.size();
                    std::vector<std::pair<key_t_, int>> expect;
                    size_t want = a.zone_slots > (int) kept ? a.zone_slots - kept : 0;
                    const size_t from_free = std::min(want, free_before);
                    for (size_t i = 0; i < cold.size() && i < want - from_free; i++) {
                        expect.push_back({a.slots[cold[i]].key, a.slots[cold[i]].pool});
                    }
                    a.lend();
                    lends++;
                    CHECK(a.zone_active);
                    CHECK(a.zone_evicted.size() == expect.size());
                    for (size_t i = 0; i < expect.size(); i++) {
                        CHECK(a.zone_evicted[i].first == expect[i].first && a.zone_evicted[i].second == expect[i].second);
                        CHECK(!a.map.count(expect[i].first));
                    }
                    lend_evicted = expect;
                }
            } else if (r < 94) {   // a prefill's read of an expert the arena does not hold
                if (a.zone_active) {
                    const key_t_ k = rand_key();
                    const int had = a.map.count(k) ? a.map.at(k) : -1;
                    const int s = a.transit_take(k, (int) (rng() % 3));
                    if (s >= 0) {
                        transits++;
                        CHECK(had < 0 && in_zone(s) && a.slots[s].lent);
                        for (const auto & h : hold) CHECK(h.s != s);
                        for (size_t i = 0; i < slot_bytes; i++) a.ptr(s)[i] = (char) byte_of(k, i);
                        hold.push_back({s, k, true});
                    } else if (had < 0) {
                        for (int z : a.zone) CHECK(busy_slot(z));   // refused only when the whole zone is in use
                    }
                }
            } else {   // generation starts: the zone back
                if (a.zone_active) {
                    std::vector<int> busy_zone;
                    for (int z : a.zone) if (busy_slot(z)) busy_zone.push_back(z);
                    const auto back = a.give_back();
                    gives++;
                    CHECK(!a.zone_active);
                    CHECK(a.zone == busy_zone);
                    CHECK(back.size() == lend_evicted.size());
                    for (size_t i = 0; i < back.size(); i++) {   // hottest first: the lending's order reversed
                        CHECK(back[i].first == lend_evicted[lend_evicted.size() - 1 - i].first);
                    }
                    lend_evicted.clear();
                }
            }
            // invariants
            std::vector<int> seen(n_slots, 0);
            for (const auto & [k, s] : a.map) {
                CHECK(s >= 0 && s < n_slots && a.slots[s].key == k);
                CHECK(a.slots[s].valid != a.slots[s].writing);
            }
            for (int s : a.free_list) {
                CHECK(s >= 0 && s < n_slots);
                CHECK(!a.slots[s].valid && !a.slots[s].writing && !a.slots[s].lent && a.slots[s].readers.load() == 0);
                seen[s]++;
            }
            size_t lent_in_map = 0;
            for (int s : a.zone) {
                CHECK(s >= 0 && s < n_slots && a.slots[s].lent && a.slots[s].prev < 0 && a.slots[s].next < 0);
                if (a.slots[s].valid || a.slots[s].writing) {
                    CHECK(a.map.count(a.slots[s].key) && a.map.at(a.slots[s].key) == s);
                    lent_in_map++;
                }
                seen[s]++;
            }
            {
                const std::vector<int> l = lru();
                CHECK(l.size() + lent_in_map == a.map.size());
                int prev = -1;
                for (int c : l) {
                    CHECK(!a.slots[c].lent);
                    CHECK(a.map.count(a.slots[c].key) && a.map.at(a.slots[c].key) == c);
                    CHECK(a.slots[c].prev == prev);
                    prev = c;
                    seen[c]++;
                }
                CHECK(a.lru_tail == prev);
            }
            for (int s = 0; s < n_slots; s++) {
                CHECK(seen[s] == 1);
                if (a.slots[s].valid) {
                    for (size_t b = 0; b < slot_bytes; b++) CHECK((uint8_t) a.ptr(s)[b] == byte_of(a.slots[s].key, b));
                }
                int readers = 0;
                for (const auto & h : hold) readers += !h.writer && h.s == s;
                CHECK(a.slots[s].readers.load() == readers);
            }
            total++;
        }
    }
    printf("arenatest: %d seeds, %llu operations (%llu lendings, %llu transit reads, %llu give-backs), invariants held\n",
           seeds, (unsigned long long) total, (unsigned long long) lends, (unsigned long long) transits,
           (unsigned long long) gives);
    return 0;
}
