// prefetchring.cpp : bench of ggml_sched_ring (ggml/src/ggml-sched-ring.h), the byte ring of the prefill's expert
// prefetch. A model of the device: chunks are copied then read (in order on the compute stream, after a random delay);
// the ring's wait(rec) makes the copy stream wait for the read of that record's last chunk. Checks, at every write:
// no byte of the new chunk belongs to a chunk whose read has not been waited for; at every read: the chunk still holds
// its own bytes (a pattern per chunk, checked byte by byte on a shadow of the ring); records reused only after their
// chunk is forgotten; chunk within the ring. Random sizes (with tensors larger than half the ring), random record
// counts, random seeds. Build: c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I ggml/src tests/prefetchring.cpp
#include "ggml-sched-ring.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s (seed %u op %d)\n", __FILE__, __LINE__, #c, seed, op); fails++; return; } } while (0)

static void run(unsigned seed, int n_ops) {
    std::mt19937 rng(seed);
    const size_t cap = 1024 + rng() % 8192;
    const int n_rec = 2 + rng() % 7;
    ggml_sched_ring r = {};
    r.reset(cap);
    std::vector<uint32_t> shadow(cap, 0);          // the id of the chunk whose bytes each ring byte holds
    struct rec_state { bool live = false; uint32_t id = 0; size_t off = 0, size = 0; bool read_done = false, waited = true; };
    std::vector<rec_state> recs(n_rec);
    uint32_t next_id = 1;
    int cur = 0;
    int op = 0;
    std::vector<int> pending;                       // records copied, not yet read (in order)
    auto do_read = [&](int k) {
        rec_state & s = recs[k];
        for (size_t i = 0; i < s.size; i++) {
            if (shadow[s.off + i] != s.id) { fprintf(stderr, "FAIL corrupted chunk (seed %u op %d)\n", seed, op); fails++; return; }
        }
        s.read_done = true;
    };
    for (op = 0; op < n_ops && !fails; op++) {
        // the device reads some pending chunks (in order: one compute stream)
        while (!pending.empty() && rng() % 3 == 0) {
            do_read(pending.front());
            pending.erase(pending.begin());
        }
        const int k = cur;
        cur = (cur + 1) % n_rec;
        size_t size = 1 + rng() % (rng() % 4 == 0 ? cap : cap / 3 + 1);
        auto wait = [&](int rec) {
            // the copy stream waits for that record's read: the device must have read it (we read it now if not yet,
            // as the stream would block until it is done)
            if (!recs[rec].read_done) {
                while (!pending.empty()) {
                    const int f = pending.front();
                    pending.erase(pending.begin());
                    do_read(f);
                    if (f == rec) break;
                }
            }
            recs[rec].waited = true;
        };
        const size_t off = r.alloc(size, k, wait);
        if (size > cap) { CHECK(off == SIZE_MAX); continue; }
        CHECK(off != SIZE_MAX && off + size <= cap);
        // the record's previous chunk was waited for and forgotten
        CHECK(!recs[k].live || recs[k].waited);
        // no live, unwaited chunk overlaps
        for (int j = 0; j < n_rec; j++) {
            if (j == k || !recs[j].live || recs[j].waited) continue;
            const bool ov = recs[j].off < off + size && off < recs[j].off + recs[j].size;
            CHECK(!ov);
        }
        for (int i = 0; i < r.n_live; i++) CHECK(r.live[i].off + r.live[i].size <= cap);
        CHECK(r.n_live <= n_rec);
        // the copy writes the chunk
        rec_state & s = recs[k];
        s = rec_state();
        s.live = true; s.id = next_id++; s.off = off; s.size = size; s.waited = false;
        for (size_t i = 0; i < size; i++) shadow[off + i] = s.id;
        pending.push_back(k);
    }
    while (!pending.empty() && !fails) { do_read(pending.front()); pending.erase(pending.begin()); }
}

int main(int argc, char ** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 500;
    const int ops = argc > 2 ? atoi(argv[2]) : 5000;
    for (int s = 0; s < seeds && !fails; s++) run(0xC0FFEEu + s, ops);
    // a deliberate error must be caught: a ring that never waits
    printf("%s: %d seeds x %d ops\n", fails ? "FAILED" : "ok", seeds, ops);
    return fails ? 1 : 0;
}
