#pragma once

// The prefill's expert ring in VRAM (the plan: a prefill MoE computed by groups of experts, GGML_CUDA_MMQ_EXPERT_GROUP,
// its absent experts streamed through a ring smaller than a layer). Modeled on the expert arena's zone (moe-arena.h):
// lend() takes the zone's entries (cache entries of the largest shape, the coldest first, their experts noted),
// take() gives the next entry in turn for an absent expert, give_back() returns the zone with the experts to restore.
// The release differs: a GPU reads an entry, not a CPU thread, so instead of a reader count each entry remembers the
// group (a record of the compute stream, its event recorded after the group's down product) that reads it last; take()
// returns that group, whose event the copy stream waits for (device side) before writing the entry. Groups are numbered
// in compute order; an entry is never asked for the group that is still filling it (the ring must hold one group's
// absent experts: check fits()).
// Host only, no HIP: tests/expertringtest.cpp drives it with ASan + UBSan.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename Key, typename Hash>
struct moe_vram_ring {
    struct entry {
        Key key{};
        bool valid = false;
        int64_t group = -1;   // the last group that reads it (-1: none)
    };
    std::vector<entry> zone;
    std::vector<std::pair<Key, int>> zone_evicted;   // what lend() took the entries from, coldest first
    size_t next = 0;
    bool active = false;
    uint64_t takes = 0, waits = 0;

    // the zone of n entries; evicted: the experts their cache entries held (coldest first)
    void lend(size_t n, std::vector<std::pair<Key, int>> evicted) {
        zone.assign(n, entry{});
        zone_evicted = std::move(evicted);
        next = 0;
        active = true;
    }

    // a group of k absent experts fits: it never waits for itself
    bool fits(size_t k) const { return k <= zone.size(); }

    // the entry for key, read by group g: its index, and in *wait the group whose read must be done before the copy
    // (-1: none). g must not decrease from one call to the next
    int take(const Key & key, int64_t g, int64_t * wait) {
        if (zone.empty()) {
            *wait = -1;
            return -1;
        }
        const size_t i = next;
        next = (next + 1) % zone.size();
        entry & e = zone[i];
        *wait = e.group;
        if (e.group >= 0) {
            waits++;
        }
        e.key = key;
        e.valid = true;
        e.group = g;
        takes++;
        return (int) i;
    }

    // the zone back: the experts to restore (hottest first), and the experts the ring holds now (each in an entry of
    // its own, readable once their last group is done: candidates for an admission without a copy)
    std::vector<std::pair<Key, int>> give_back(std::vector<std::pair<Key, int64_t>> * held = nullptr) {
        if (held) {
            held->clear();
            for (const entry & e : zone) {
                if (e.valid) {
                    held->push_back({e.key, e.group});
                }
            }
        }
        std::vector<std::pair<Key, int>> out(zone_evicted.rbegin(), zone_evicted.rend());
        zone.clear();
        zone_evicted.clear();
        next = 0;
        active = false;
        return out;
    }
};
