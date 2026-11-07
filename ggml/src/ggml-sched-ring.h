#pragma once
// A byte ring for the expert prefetch of a prefill (ggml-backend.cpp, GGML_SCHED_PREFETCH_RING_MB): each prefetched
// expert tensor takes one chunk (its table of expert addresses, then the experts the MoE cache does not hold, packed
// back to back), in allocation order, wrapping to the start when a chunk does not fit before the end. A chunk is tied to
// a record (the slot whose events guard it: copied, then read); before a chunk is written, every live chunk it overlaps
// must have been read: alloc() calls wait(record) for each of them (a device-side wait on the copy stream) and forgets
// them, as it does the record's own previous chunk. No host-side wait, no timer.
// Plain data (the scheduler is calloc'ed); self-contained (no ggml): tests/prefetchring.cpp drives it with ASan + UBSan.
#include <cstddef>
#include <cstdint>

#define GGML_SCHED_RING_MAX_LIVE 16

struct ggml_sched_ring {
    struct chunk {
        size_t off;
        size_t size;
        int    rec;
    };
    size_t cap;
    size_t head;
    int    n_live;
    chunk  live[GGML_SCHED_RING_MAX_LIVE];   // allocation order

    void reset(size_t capacity) {
        cap    = capacity;
        head   = 0;
        n_live = 0;
    }

    static bool overlaps(const chunk & c, size_t off, size_t size) {
        return c.off < off + size && off < c.off + c.size;
    }

    void erase(int i) {
        for (int j = i + 1; j < n_live; j++) {
            live[j - 1] = live[j];
        }
        n_live--;
    }

    // forget the chunk of record rec (its record is about to guard a new chunk): wait(rec) if it is live
    template <typename W> void retire(int rec, W && wait) {
        for (int i = 0; i < n_live; i++) {
            if (live[i].rec == rec) {
                wait(rec);
                erase(i);
                return;
            }
        }
    }

    // a chunk of size bytes for record rec (rec < GGML_SCHED_RING_MAX_LIVE); SIZE_MAX if it can never fit
    template <typename W> size_t alloc(size_t size, int rec, W && wait) {
        if (size == 0 || size > cap || rec < 0 || rec >= GGML_SCHED_RING_MAX_LIVE) {
            return SIZE_MAX;
        }
        retire(rec, wait);
        const size_t off = head + size <= cap ? head : 0;
        for (int i = 0; i < n_live;) {
            if (overlaps(live[i], off, size)) {
                wait(live[i].rec);
                erase(i);
            } else {
                i++;
            }
        }
        live[n_live++] = {off, size, rec};   // at most one live chunk per record: n_live <= MAX_LIVE
        head = off + size;
        return off;
    }
};
