// expertringtest.cpp : bench of moe_vram_ring (ggml/src/ggml-cuda/moe-ring.h). A model of the prefill: layers of 512
// experts in groups of G, each layer's experts of one size (its own, as the pools of the IQ3_KS hybrid), a random share of
// each group absent (copied through the ring), copies on one stream in order and groups computed on another in order, a
// copy into bytes waiting for the read of the last group among the entries they cover. Checks: every group reads its
// entries with the bytes of its own experts (a shadow of the zone, unit by unit: no byte overwritten before its reader is
// done), every entry within the zone, the ring never asks a group to wait for itself or a later group, the copy stream
// never deadlocks, give_back returns the evicted experts hottest first and the held ones with their groups. Random G,
// zone and entry sizes, absent shares and seeds.
// c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I ggml/src/ggml-cuda tests/expertringtest.cpp
#include "moe-ring.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct key { int l, e; bool operator==(const key & o) const { return l == o.l && e == o.e; } };
struct key_hash { size_t operator()(const key & k) const { return (size_t) k.l * 1024 + k.e; } };

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s (seed %u)\n", __FILE__, __LINE__, #c, seed); fails++; return; } } while (0)

static void run(unsigned seed) {
    std::mt19937 rng(seed);
    const int G = 1 << (rng() % 6);              // 1 .. 32 experts a group
    const int NE = 512, L = 1 + rng() % 8;
    const size_t U = 256;                        // the unit of the shadow (entries are multiples of 256 B)
    std::vector<size_t> lsize(L);                // each layer's entry, in units
    size_t smax = 0;
    for (int l = 0; l < L; l++) {
        lsize[l] = 1 + rng() % 40;
        smax = std::max(smax, lsize[l]);
    }
    const size_t units = (G + 1) * smax + rng() % (400 * smax);   // at least one group's experts of the largest layer
    const double absent = (rng() % 100) / 100.0;
    moe_vram_ring<key, key_hash> r;
    std::vector<std::pair<key, int>> ev;
    for (int i = 0; i < 37; i++) ev.push_back({key{-1, i}, i % 7});
    r.lend(units * U, ev);
    for (int l = 0; l < L; l++) CHECK(r.fits(G, lsize[l] * U));
    std::vector<key> shadow(units, key{-9, -9});   // what each unit holds
    // the device: the groups queued (their entries) and the last group computed
    struct rd { size_t off, n; key k; };
    struct grp { std::vector<rd> reads; };
    std::vector<grp> groups;
    int64_t done = -1;                           // groups computed: 0 .. done
    auto compute_until = [&](int64_t g) {        // the compute stream runs the groups up to g (in order)
        while (done < g) {
            done++;
            for (auto & x : groups[done].reads) {
                for (size_t u = x.off; u < x.off + x.n; u++) {
                    if (!(shadow[u] == x.k)) { fprintf(stderr, "FAIL bytes overwritten before their read (seed %u)\n", seed); fails++; return; }
                }
            }
        }
    };
    int64_t g = 0;
    for (int l = 0; l < L && !fails; l++) {
        const size_t b = lsize[l] * U;
        for (int e0 = 0; e0 < NE && !fails; e0 += G, g++) {
            groups.push_back({});
            for (int e = e0; e < e0 + G && e < NE; e++) {
                if ((rng() % 1000) / 1000.0 >= absent) continue;   // resident: read in place
                int64_t wait;
                const int64_t off = r.take(key{l, e}, g, b, &wait);
                CHECK(off >= 0 && off % U == 0 && (size_t) off + b <= units * U);
                CHECK(wait < g);                                    // never itself nor a later group
                if (wait >= 0) {
                    compute_until(wait);                            // the copy stream waits for that read
                    if (fails) return;
                }
                CHECK(wait <= done);
                for (size_t u = off / U; u < off / U + lsize[l]; u++) shadow[u] = key{l, e};   // the copy lands
                groups[g].reads.push_back({(size_t) off / U, lsize[l], key{l, e}});
            }
            if (rng() % 3 == 0) compute_until(g);                   // the device sometimes catches up
        }
    }
    if (!fails) compute_until(g - 1);
    std::vector<std::pair<key, int64_t>> held;
    auto back = r.give_back(&held);
    CHECK(back.size() == ev.size() && back.front().first == ev.back().first);
    for (auto & h : held) CHECK(h.second < g);
    CHECK(!r.active && r.live.empty());
}

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 2000;
    for (int s = 0; s < seeds && !fails; s++) run(0xA11CE + s);
    printf("%s: %d seeds\n", fails ? "FAILED" : "ok", seeds);
    return fails ? 1 : 0;
}
