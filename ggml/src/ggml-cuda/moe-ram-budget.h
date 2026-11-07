#pragma once

// The host RAM taken by the experts of a MoE cache session, decided once for all its cards (moe_cache_build_pending).
//
// One card (or a session that does not know its total): the rule of before, per card and measured at its decision:
// every expert range registered in place (read with O_DIRECT into anonymous memory) when they fit both the room
// (moe_cache_arena_auto_bytes: MemAvailable and the cgroup v2 limits, less the margin, 4 GiB) and KFD's limit less
// 1 GiB, else the arena.
//
// Several cards (pipeline stages or layers split by card; each card holds its own layers' experts, disjoint): the room
// is measured once, at the first card's decision, before any card took its share, and frozen; the budget is that room
// less a reserve for what is allocated after the decision (see moe_ram_reserve_bytes), and less 1 GiB under KFD's limit.
// When the session's experts fit the budget, every card registers all of its own (as before); else each card registers
// its share of the budget (by its experts' bytes), never more than what the budget still has, and leaves the rest of its
// experts in the page cache (mapped, not registered: fills copied through the staging buffer, a prefill reads them from
// the mapping). Never the arena because of several cards.
//
// Host only: tests/rambudgettest.cpp exercises it with ASan + UBSan.

#include <algorithm>
#include <cstddef>
#include <cstdint>

enum moe_ram_mode {
    MOE_RAM_PIN_ALL  = 0,   // every expert range registered in place
    MOE_RAM_ARENA    = 1,   // the arena (one card that does not fit)
    MOE_RAM_PIN_PART = 2,   // several cards that do not all fit: quota bytes registered, the rest left in the page cache
};

struct moe_ram_choice {
    moe_ram_mode mode;
    size_t quota;   // bytes this card may register (PIN_PART), its experts' bytes otherwise
};

// the session's part, set once
struct moe_ram_plan {
    size_t total = 0;       // the bytes of every host expert range of the model (all cards), 0: unknown
    int n_devices = 0;      // the GPUs the experts are spread over
    bool decided = false;   // the first card's decision is made: room, budget frozen
    size_t room = 0;        // the room at that decision, before any margin
    size_t budget = 0;      // what all the cards together may register
    bool all_fit = false;
    size_t used = 0;        // registered so far by the session's cards (their quotas)
};

static constexpr size_t moe_ram_mib = (size_t) 1 << 20;

// what is allocated after the decision, which the room measured at it does not see.
// One card: the margin (GGML_CUDA_MOE_ARENA_MARGIN_MB, 4096), the rule of before, unchanged.
// Several cards (2 x RTX 5060 Ti, cgroup of 61 648 MiB, 49 743 MiB of experts, 7 Nov.): one card lived with 10 581 MiB
// of room left above its experts (room 60 324 before the margin), 0.5 GiB to spare at its peak; with two cards, per
// card more pinned host memory allocated after the decision (shmem +2 214 MiB), more page cache in use (file +1 688 MiB),
// and the server was killed with 10 425 MiB left. Base: the one-card need measured there (10 GiB), plus 3 GiB for each
// further card (the 2.5 GiB measured for the second, rounded up); per_card_mib and base_mib override them
static inline size_t moe_ram_reserve_bytes(int n_devices, size_t margin_mib, size_t base_mib, size_t per_card_mib) {
    if (n_devices <= 1) {
        return margin_mib * moe_ram_mib;
    }
    return (base_mib + (size_t) (n_devices - 1) * per_card_mib) * moe_ram_mib;
}

static inline size_t moe_ram_sub(size_t a, size_t b) {
    return a > b ? a - b : 0;
}

// one card's choice, its expert bytes given. room_raw: the room now, before any margin (SIZE_MAX if unknown); kfd: KFD's
// limit for registered memory (RAM - RAM/64 - 1.5 GiB); reserve_one / reserve_multi: moe_ram_reserve_bytes for one card
// and for plan.n_devices. Updates plan (the first card of several freezes the room and the budget, each takes its quota)
static inline moe_ram_choice moe_ram_choose(moe_ram_plan & plan, size_t bytes, size_t room_raw, size_t kfd,
                                            size_t reserve_one, size_t reserve_multi) {
    const size_t one_gib = (size_t) 1 << 30;
    if (plan.n_devices <= 1 || plan.total == 0) {
        // the rule of before, per card, at its own decision
        const size_t room = room_raw == SIZE_MAX ? 0 : moe_ram_sub(room_raw, reserve_one);
        const bool fit = bytes + one_gib <= kfd && bytes <= room;
        return { fit ? MOE_RAM_PIN_ALL : MOE_RAM_ARENA, bytes };
    }
    if (!plan.decided) {
        plan.decided = true;
        plan.room = room_raw == SIZE_MAX ? 0 : room_raw;
        plan.budget = std::min(moe_ram_sub(plan.room, reserve_multi), moe_ram_sub(kfd, one_gib));
        // a card whose experts the total did not count (a total too small): the session's real total is at least this
        plan.all_fit = std::max(plan.total, bytes) <= plan.budget;
    }
    const size_t left = moe_ram_sub(plan.budget, plan.used);
    size_t quota = bytes;
    if (!plan.all_fit) {
        const size_t total = std::max(plan.total, bytes);
        quota = (size_t) ((long double) plan.budget * (long double) bytes / (long double) total);
    }
    quota = std::min(quota, left);
    plan.used += quota;
    return { quota >= bytes ? MOE_RAM_PIN_ALL : MOE_RAM_PIN_PART, quota };
}

// the ranges a card registers under its quota (sorted, merged, page-aligned): the first ones in address order (the
// layers in order) while they fit, a range that does not fit skipped for the smaller ones after it. keep[i] set for
// each range kept; returns the bytes kept
template <typename Ranges, typename Keep>
static inline size_t moe_ram_select(const Ranges & ranges, size_t quota, Keep & keep) {
    size_t kept = 0;
    for (size_t i = 0; i < ranges.size(); i++) {
        const size_t len = ranges[i].second - ranges[i].first;
        keep[i] = kept + len <= quota;
        kept += keep[i] ? len : 0;
    }
    return kept;
}
