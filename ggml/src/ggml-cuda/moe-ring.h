#pragma once

// The prefill's expert ring in VRAM (the plan: a prefill MoE computed by groups of experts, GGML_CUDA_MMQ_EXPERT_GROUP,
// its absent experts streamed through a ring smaller than a layer). Modeled on the expert arena's zone (moe-arena.h):
// lend() takes the zone (cache entries, the coldest first, their experts noted), take() gives the next bytes in turn
// for an absent expert, give_back() returns the zone with the experts to restore.
// The zone is a FIFO of bytes, each expert taking the size of its own layer's entry (the experts of a layer are all the
// same size, the copies go in (layer, expert) order): the entries of a layer follow each other, the size changes only
// from one layer to the next, and an expert that does not fit before the zone's end starts again at its beginning (the
// tail left is lost for that turn). With the largest expert's size for every entry, the IQ3_KS hybrid's 7 shapes
// (1 475 - 2 600 KiB) lost ~22 % of the zone.
// The release differs from the arena: a GPU reads an entry, not a CPU thread, so instead of a reader count each entry
// remembers the group (a record of the compute stream, its event recorded after the group's down product) that reads it
// last; take() returns the latest group among the entries its bytes cover (and the tail it skips), whose event the copy
// stream waits for (device side) before writing them. Groups are numbered in compute order and the entries are taken in
// that order, so that group covers the others; an entry is never asked for the group that is still filling it (the ring
// must hold one group's absent experts: check fits()).
// Host only, no HIP: tests/expertringtest.cpp drives it with ASan + UBSan.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename Key, typename Hash>
struct moe_vram_ring {
    struct entry {
        Key key{};
        size_t off = 0, bytes = 0;
        int64_t group = -1;   // the last group that reads it (-1: none)
    };
    std::deque<entry> live;   // the entries that hold an expert, oldest first (the next ones overwritten)
    std::vector<std::pair<Key, int>> zone_evicted;   // what lend() took the entries from, coldest first
    size_t cap = 0;           // the zone's bytes
    size_t head = 0;          // where the next entry starts
    bool active = false;
    uint64_t takes = 0, waits = 0;

    // the zone of `bytes` bytes; evicted: the experts their cache entries held (coldest first)
    void lend(size_t bytes, std::vector<std::pair<Key, int>> evicted) {
        live.clear();
        zone_evicted = std::move(evicted);
        cap = bytes;
        head = 0;
        active = true;
    }

    // entries of `bytes` the zone holds at once
    size_t entries(size_t bytes) const { return bytes ? cap / bytes : 0; }

    // a group of k absent experts of `bytes` each fits: it never waits for itself (the tail an entry skips at the zone's
    // end is less than one entry)
    bool fits(size_t k, size_t bytes) const { return bytes > 0 && (k + 1) * bytes <= cap; }

    // the bytes for key (an entry of `bytes`), read by group g: their offset in the zone (-1: none), and in *wait the
    // group whose reads must be done before the copy (-1: none). g must not decrease from one call to the next
    int64_t take(const Key & key, int64_t g, size_t bytes, int64_t * wait) {
        *wait = -1;
        if (bytes == 0 || bytes > cap) {
            return -1;
        }
        size_t off = head;
        if (off + bytes > cap) {
            // the tail is skipped: the entries in it are dropped (no copy lands there this turn, but their groups are
            // older than what follows, and waiting for them keeps the rule simple)
            while (!live.empty() && live.front().off >= off) {
                *wait = std::max(*wait, live.front().group);
                live.pop_front();
            }
            off = 0;
        }
        // the entries the new bytes cover are the oldest ones, right after `off`
        while (!live.empty() && live.front().off < off + bytes && off < live.front().off + live.front().bytes) {
            *wait = std::max(*wait, live.front().group);
            live.pop_front();
        }
        if (*wait >= 0) {
            waits++;
        }
        live.push_back({key, off, bytes, g});
        head = off + bytes;
        takes++;
        return (int64_t) off;
    }

    // the zone back: the experts to restore (hottest first), and the experts the ring holds now (each in an entry of
    // its own, readable once their last group is done: candidates for an admission without a copy)
    std::vector<std::pair<Key, int>> give_back(std::vector<std::pair<Key, int64_t>> * held = nullptr) {
        if (held) {
            held->clear();
            for (const entry & e : live) {
                held->push_back({e.key, e.group});
            }
        }
        std::vector<std::pair<Key, int>> out(zone_evicted.rbegin(), zone_evicted.rend());
        live.clear();
        zone_evicted.clear();
        cap = 0;
        head = 0;
        active = false;
        return out;
    }
};
