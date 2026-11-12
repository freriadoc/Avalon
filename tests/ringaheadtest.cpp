// Bench of the prefill ring's reads ahead (ggml/src/ggml-cuda/moe-ring-ahead.h) over the expert arena (moe-arena.h).
// Three threads on one lock, as in the cache: the reads-ahead thread, a "ring" that goes through the layers of several
// "ubatches" the way ggml_cuda_moe_ring_block does (wait for the layer, take its slots over, read what is left itself,
// keep the slots until the next block's synchronization), and a "CPU path" that writes experts into the arena, evicting.
// Random: arena and zone sizes (down to a zone smaller than a layer's absent experts), layers, experts, what the "VRAM
// cache" holds (it changes while the thread reads), read durations, read failures, plans dropped in the middle of a
// ubatch, the zone given back.
// Invariants:
// - a slot handed to the ring is valid, holds the expert asked for, and its bytes are that expert's, when it is handed
//   over and again when the ring releases it (it was never recycled while held);
// - every expert the cache does not hold gets a source (ahead, the arena, the block's own read) unless a read failed or
//   the zone is smaller than what one block holds;
// - no deadlock (an alarm), whatever the zone's size;
// - at the end: no reader left, no slot being written, map <-> slots, every slot in exactly one of the free list, the LRU
//   list and the zone.
// c++ -std=c++17 -O1 -g -fsanitize=address,undefined -pthread -I ggml/src/ggml-cuda tests/ringaheadtest.cpp -o ringaheadtest && ./ringaheadtest
// c++ -std=c++17 -O1 -g -fsanitize=thread -pthread -I ggml/src/ggml-cuda tests/ringaheadtest.cpp -o ringaheadtest-tsan && ./ringaheadtest-tsan 60

#include "moe-arena.h"
#include "moe-ring-ahead.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <unistd.h>
#include <vector>

struct key_t_ {
    const void * tensor = nullptr;
    int32_t expert = -1;
    bool operator==(const key_t_ & o) const { return tensor == o.tensor && expert == o.expert; }
};
struct key_hash {
    size_t operator()(const key_t_ & k) const { return std::hash<const void *>()(k.tensor) ^ ((size_t) k.expert * 0x9e3779b97f4a7c15ULL); }
};
struct job_t { int32_t e = -1; };

typedef moe_arena<key_t_, key_hash> arena_t;
typedef moe_ring_ahead<arena_t, key_t_, job_t> ahead_t;

static uint8_t byte_of(const key_t_ & k, size_t i) {
    return (uint8_t) (((uintptr_t) k.tensor >> 4) * 131 + (uint32_t) k.expert * 31 + i * 7);
}

static unsigned g_seed = 0;
static const char * g_where = "";
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s (seed %u, %s)\n", __FILE__, __LINE__, #c, g_seed, g_where); exit(1); } } while (0)

static void on_alarm(int) {
    const char msg[] = "FAIL: deadlock (alarm)\n";
    (void) !write(2, msg, sizeof(msg) - 1);
    _exit(2);
}

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 300;
    signal(SIGALRM, on_alarm);
    uint64_t n_ahead = 0, n_arena = 0, n_own = 0, n_none = 0, n_left = 0, n_drops = 0, n_blocks = 0, n_fail = 0, n_waits = 0;
    for (unsigned seed = 1; seed <= (unsigned) seeds; seed++) {
        g_seed = seed;
        alarm(60);
        std::mt19937 rng(seed);
        const int n_layers = 1 + rng() % 10;
        const int n_expert = 4 + rng() % 60;
        const int n_slots = 8 + rng() % 400;
        const size_t slot_bytes = 32 + rng() % 96;
        std::vector<char> mem((size_t) n_slots * slot_bytes);
        std::vector<char> layer_ids(n_layers);   // their addresses are the layers' ids
        arena_t ar;
        ar.init(mem.data(), slot_bytes, n_slots);
        // from a zone smaller than a layer to one of several layers
        ar.zone_slots = std::min(n_slots - 1, (int) (1 + rng() % (3 * n_expert)));
        std::mutex mu;
        std::vector<std::vector<char>> cached(n_layers, std::vector<char>(n_expert, 0));   // the "VRAM cache", under mu
        const int fail_1_in = rng() % 3 ? 0 : 50 + rng() % 400;
        const bool slow = rng() % 4 == 0;
        std::atomic<uint64_t> load_rng{seed * 7919u};
        const auto write_slot = [&](int s, const key_t_ & k) {
            char * p = ar.ptr(s);
            for (size_t i = 0; i < slot_bytes; i++) {
                p[i] = (char) byte_of(k, i);
            }
        };
        const auto slot_is = [&](int s, const key_t_ & k) {
            const char * p = ar.ptr(s);
            for (size_t i = 0; i < slot_bytes; i++) {
                if ((uint8_t) p[i] != byte_of(k, i)) {
                    return false;
                }
            }
            return true;
        };
        const auto key_of = [&](int l, int32_t e) { return key_t_{&layer_ids[l], e}; };
        const auto layer_of = [&](const void * id) { return (int) ((const char *) id - layer_ids.data()); };
        // the reads: without the lock, a slot's bytes written while it is "writing"
        const auto do_load = [&](const std::vector<std::pair<int, key_t_>> & reads, std::vector<char> & ok) {
            for (size_t b = 0; b < reads.size(); b++) {
                const uint64_t r = load_rng.fetch_add(0x9e3779b97f4a7c15ULL) >> 33;
                if (slow && r % 16 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(r % 300));
                }
                write_slot(reads[b].first, reads[b].second);
                ok[b] = !(fail_1_in && r % fail_1_in == 0);
            }
        };
        // the arena starts partly full
        {
            const int n0 = (int) (rng() % (n_slots + 1));
            for (int i = 0; i < n0; i++) {
                const key_t_ k = key_of(rng() % n_layers, rng() % n_expert);
                const int s = ar.take(k, 0);
                if (s >= 0) {
                    write_slot(s, k);
                    ar.commit(s);
                }
            }
            for (int l = 0; l < n_layers; l++) {
                for (int e = 0; e < n_expert; e++) {
                    cached[l][e] = rng() % 4 == 0;
                }
            }
        }
        ahead_t A;
        A.wanted = [&](const ahead_t::layer & L, int32_t e, job_t & j) { j.e = e; return !cached[layer_of(L.id)][e]; };
        A.key = [&](const ahead_t::layer & L, int32_t e) { return key_of(layer_of(L.id), e); };
        A.lend = [&](arena_t & a) { if (!a.zone_active) { a.lend(); } };
        A.load = [&](const ahead_t::layer & L, const std::vector<ahead_t::read> & batch, std::vector<char> & ok) {
            std::vector<std::pair<int, key_t_>> reads;
            for (const auto & b : batch) {
                CHECK(b.job.e == b.e);
                reads.push_back({b.slot, key_of(layer_of(L.id), b.e)});
            }
            do_load(reads, ok);
        };
        // the "CPU path": experts written into the arena (evicting the coldest), the "VRAM cache" changing
        std::atomic<bool> cpu_stop{false};
        std::thread cpu([&] {
            std::mt19937 r2(seed * 31 + 7);
            while (!cpu_stop.load()) {
                int s = -1;
                key_t_ k;
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (r2() % 3 == 0) {
                        cached[r2() % n_layers][r2() % n_expert] ^= 1;
                    }
                    k = key_of(r2() % n_layers, r2() % n_expert);
                    s = ar.take(k, 0);
                }
                if (s >= 0) {
                    write_slot(s, k);
                    std::lock_guard<std::mutex> lock(mu);
                    ar.commit(s);
                }
                if (r2() % 8 == 0) {
                    std::this_thread::yield();
                }
            }
        });
        // the ring
        std::vector<std::pair<int, key_t_>> pins, pins_prev;   // (slot, key), a reader held on each
        const auto release = [&](std::vector<std::pair<int, key_t_>> & v) {
            for (const auto & p : v) {
                CHECK(slot_is(p.first, p.second));   // never recycled while held
                ar.release(p.first);
            }
            v.clear();
        };
        const int n_ubatch = 1 + rng() % 5;
        for (int u = 0; u < n_ubatch; u++) {
            const int first = rng() % n_layers;
            const int drop_at = rng() % 5 == 0 ? first + (int) (rng() % (n_layers - first)) : -1;
            for (int l = first; l < n_layers; l++) {
                n_blocks++;
                g_where = "block";
                std::vector<int> src(n_expert, -1);   // the slot each expert not cached is copied from
                std::vector<char> was_cached(n_expert);
                {
                    std::unique_lock<std::mutex> lk(mu);
                    int idx = A.find_locked(&layer_ids[l]);
                    if (idx < 0) {
                        std::vector<ahead_t::layer> plan;
                        for (int q = l; q < n_layers; q++) {
                            ahead_t::layer L;
                            L.id = &layer_ids[q];
                            L.ar = &ar;
                            L.pool = 0;
                            L.n_expert = n_expert;
                            plan.push_back(std::move(L));
                        }
                        A.set_plan_locked(std::move(plan));
                        A.start_locked(mu);
                        idx = A.find_locked(&layer_ids[l]);
                        CHECK(idx == 0);
                    }
                    n_waits += !A.plan[idx].done;
                    const bool ok = A.wait(lk, idx, [&] { release(pins_prev); });
                    CHECK(ok);   // nothing else drops the plan while a block waits
                    for (int e = 0; e < n_expert; e++) {
                        was_cached[e] = cached[l][e];
                    }
                    for (const auto & pin : A.take_locked(idx)) {
                        const key_t_ k = key_of(l, pin.first);
                        CHECK(ar.slots[pin.second].valid && ar.slots[pin.second].key == k && ar.slots[pin.second].lent);
                        CHECK(ar.slots[pin.second].readers.load() > 0);
                        CHECK(slot_is(pin.second, k));
                        if (!was_cached[pin.first] && src[pin.first] < 0) {
                            src[pin.first] = pin.second;
                            pins.push_back({pin.second, k});
                            n_ahead++;
                        } else {
                            ar.release(pin.second);
                        }
                    }
                }
                // the block's own sources for what is left: the arena, else a read into the zone (twice: the second time
                // after the previous block's slots are freed)
                for (int pass = 0; pass < 2; pass++) {
                    std::vector<std::pair<int, key_t_>> reads;
                    std::vector<int32_t> es;
                    bool full = false;
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        for (int e = 0; e < n_expert; e++) {
                            if (was_cached[e] || src[e] >= 0) {
                                continue;
                            }
                            const key_t_ k = key_of(l, e);
                            const int a = ar.acquire(k, false);
                            if (a >= 0) {
                                CHECK(slot_is(a, k));
                                src[e] = a;
                                pins.push_back({a, k});
                                n_arena++;
                                continue;
                            }
                            if (ar.map.count(k)) {
                                continue;   // being written by the CPU path: left out (the cache copies it from the mapping)
                            }
                            if (!ar.zone_active) {
                                ar.lend();
                            }
                            const int t = ar.transit_take(k, 0);
                            if (t >= 0) {
                                reads.push_back({t, k});
                                es.push_back(e);
                            } else {
                                full = true;
                            }
                        }
                    }
                    std::vector<char> ok(reads.size(), 0);
                    do_load(reads, ok);
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        for (size_t b = 0; b < reads.size(); b++) {
                            if (ok[b]) {
                                ar.commit(reads[b].first);
                                const int a = ar.acquire(reads[b].second, false);
                                CHECK(a == reads[b].first);
                                src[es[b]] = a;
                                pins.push_back({a, reads[b].second});
                                n_own++;
                            } else {
                                ar.abort(reads[b].first);
                                n_fail++;
                            }
                        }
                    }
                    if (!full) {
                        break;
                    }
                    release(pins_prev);
                    A.freed();
                }
                for (int e = 0; e < n_expert; e++) {
                    n_none += !was_cached[e] && src[e] < 0;
                }
                if (rng() % 3 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(rng() % 200));   // the GPU computes
                }
                // the next block's synchronization: the previous block's slots go
                release(pins_prev);
                A.freed();
                pins_prev.swap(pins);
                if (l == drop_at) {
                    g_where = "drop";
                    n_drops++;
                    release(pins_prev);
                    std::lock_guard<std::mutex> lock(mu);
                    A.cancel_locked();
                    if (ar.zone_active && rng() % 2) {
                        ar.give_back();   // generation starts: the zone back (slots still being read stay lent)
                    }
                    break;
                }
            }
            g_where = "ubatch end";
            release(pins_prev);
            A.freed();
            if (rng() % 2) {
                std::lock_guard<std::mutex> lock(mu);
                A.cancel_locked();
                if (ar.zone_active) {
                    ar.give_back();
                }
            }
        }
        g_where = "end";
        cpu_stop = true;
        cpu.join();
        A.stop_join();
        n_left += A.left;
        {
            std::lock_guard<std::mutex> lock(mu);
            if (ar.zone_active) {
                ar.give_back();
            }
            ar.give_back();   // the slots a first call kept (they were in use)
            CHECK(ar.zone.empty());
            std::vector<int> where(n_slots, 0);
            for (const int s : ar.free_list) {
                where[s]++;
            }
            int n_lru = 0;
            for (int c = ar.lru_head, prev = -1; c >= 0; prev = c, c = ar.slots[c].next) {
                CHECK(ar.slots[c].prev == prev);
                where[c]++;
                CHECK(++n_lru <= n_slots);
            }
            for (int s = 0; s < n_slots; s++) {
                CHECK(where[s] == 1);
                CHECK(ar.slots[s].readers.load() == 0 && !ar.slots[s].writing && !ar.slots[s].lent);
            }
            size_t n_valid = 0;
            for (int s = 0; s < n_slots; s++) {
                if (ar.slots[s].valid) {
                    n_valid++;
                    const auto it = ar.map.find(ar.slots[s].key);
                    CHECK(it != ar.map.end() && it->second == s);
                    CHECK(slot_is(s, ar.slots[s].key));
                }
            }
            CHECK(n_valid == ar.map.size());
        }
    }
    alarm(0);
    printf("ok: %d seeds, %llu blocks (%llu waited for the thread), %llu plans dropped; sources: %llu read ahead, %llu in the "
           "arena, %llu read by the block, %llu none (%llu reads failed); %llu experts left by the thread to their block\n",
           seeds, (unsigned long long) n_blocks, (unsigned long long) n_waits, (unsigned long long) n_drops,
           (unsigned long long) n_ahead, (unsigned long long) n_arena, (unsigned long long) n_own, (unsigned long long) n_none,
           (unsigned long long) n_fail, (unsigned long long) n_left);
    return 0;
}
