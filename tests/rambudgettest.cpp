// Bench of the experts' host RAM decision (ggml/src/ggml-cuda/moe-ram-budget.h): the measured cases (2 x RTX 5060 Ti in
// a cgroup of 61 648 MiB, 7 Nov.), then random sessions. Invariants: one card (or an unknown total) decides exactly as
// before; several cards never take the arena; the quotas never exceed the budget; when the session fits, every card
// registers all its experts; the ranges kept by moe_ram_select stay within the quota.
// c++ -std=c++17 -O1 -g -fsanitize=address,undefined -I ggml/src/ggml-cuda tests/rambudgettest.cpp -o rambudgettest && ./rambudgettest

#include "moe-ram-budget.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s (case %s)\n", __FILE__, __LINE__, #c, name); exit(1); } } while (0)

static const size_t MiB = (size_t) 1 << 20;
static const size_t GiB = (size_t) 1 << 30;

// the rule of before (moe_cache_experts_fit before this change)
static bool legacy_fit(size_t bytes, size_t room_raw, size_t kfd, size_t margin) {
    const size_t room = room_raw != SIZE_MAX && room_raw > margin ? room_raw - margin : 0;
    return bytes + GiB <= kfd && bytes <= room;
}

struct card_result { moe_ram_choice c; };

// a session: each card decides in turn, the room measured at its decision falling by what the cards before registered
static std::vector<moe_ram_choice> run(const char * name, const std::vector<size_t> & cards, size_t room0, size_t kfd,
                                       bool known_total = true, size_t base_mib = 10240, size_t card_mib = 3072) {
    moe_ram_plan plan;
    size_t total = 0;
    for (size_t b : cards) {
        total += b;
    }
    plan.total = known_total ? total : 0;
    plan.n_devices = (int) cards.size();
    const size_t reserve_one = moe_ram_reserve_bytes(1, 4096, 0, 0);
    const size_t reserve_multi = moe_ram_reserve_bytes(plan.n_devices, 4096, base_mib, card_mib);
    std::vector<moe_ram_choice> out;
    size_t room = room0, sum = 0;
    for (size_t b : cards) {
        const moe_ram_choice c = moe_ram_choose(plan, b, room, kfd, reserve_one, reserve_multi);
        out.push_back(c);
        const size_t taken = c.mode == MOE_RAM_ARENA ? 0 : std::min(c.quota, b);
        sum += c.mode == MOE_RAM_ARENA ? 0 : c.quota;
        room = room == SIZE_MAX ? room : (room > taken ? room - taken : 0);
        if (cards.size() <= 1 || !known_total) {
            CHECK((c.mode == MOE_RAM_PIN_ALL) == legacy_fit(b, room + taken, kfd, 4 * 1024 * MiB));
            CHECK(c.mode != MOE_RAM_PIN_PART);
        } else {
            CHECK(c.mode != MOE_RAM_ARENA);
            CHECK(c.quota <= b || c.mode == MOE_RAM_PIN_ALL);
            CHECK(sum <= plan.budget || plan.all_fit);
            if (plan.all_fit) {
                CHECK(c.mode == MOE_RAM_PIN_ALL && c.quota == b);
            }
        }
    }
    printf("%-34s", name);
    for (size_t i = 0; i < cards.size(); i++) {
        printf(" | card %zu: %6zu MiB -> %s %6zu", i, cards[i] / MiB,
               out[i].mode == MOE_RAM_PIN_ALL ? "all " : out[i].mode == MOE_RAM_ARENA ? "arena" : "part", out[i].quota / MiB);
    }
    if (plan.decided) {
        printf(" | budget %zu", plan.budget / MiB);
    }
    printf("\n");
    return out;
}

int main(int argc, char ** argv) {
    const char * name = "measured";
    const size_t kfd_pod = 248 * GiB;   // the pod's host RAM is large; only the cgroup limits
    // measured: one card, 49 743 MiB of experts, room 56 228 MiB after the 4 GiB margin (60 324 before): registered
    auto one = run("1 card fits (pod, 1 x 5060 Ti)", {49743 * MiB}, 60324 * MiB, kfd_pod);
    CHECK(one[0].mode == MOE_RAM_PIN_ALL);
    // one card that does not fit: the arena, as before
    auto big = run("1 card does not fit", {58000 * MiB}, 60324 * MiB, kfd_pod);
    CHECK(big[0].mode == MOE_RAM_ARENA);
    // two cards, plenty of RAM: every card all
    auto two = run("2 cards fit (125 GiB room)", {27728 * MiB, 22015 * MiB}, 117911 * MiB, kfd_pod);
    CHECK(two[0].mode == MOE_RAM_PIN_ALL && two[1].mode == MOE_RAM_PIN_ALL);
    // the pod: 2 cards, 27 728 + 22 015 MiB, room 60 168 MiB at the first decision (56 072 + the 4 GiB margin); killed
    // before with everything registered
    auto pod = run("2 cards in 60 GiB (the pod)", {27728 * MiB, 22015 * MiB}, 60168 * MiB, kfd_pod);
    CHECK(pod[0].mode == MOE_RAM_PIN_PART && pod[1].mode == MOE_RAM_PIN_PART);
    CHECK(pod[0].quota + pod[1].quota <= 60168 * MiB - (10240 + 3072) * MiB);
    printf("  pod: %zu MiB registered, %zu MiB left in the page cache\n", (pod[0].quota + pod[1].quota) / MiB,
           (27728 + 22015) * MiB / MiB - (pod[0].quota + pod[1].quota) / MiB);
    // three cards
    auto three = run("3 cards in 60 GiB", {17000 * MiB, 16000 * MiB, 16743 * MiB}, 60168 * MiB, kfd_pod);
    for (auto & c : three) CHECK(c.mode == MOE_RAM_PIN_PART);
    auto three_ok = run("3 cards in 120 GiB", {17000 * MiB, 16000 * MiB, 16743 * MiB}, 120000 * MiB, kfd_pod);
    for (auto & c : three_ok) CHECK(c.mode == MOE_RAM_PIN_ALL);
    // no cgroup, no MemAvailable (room unknown): one card the arena as before (room 0), several cards nothing registered
    auto none1 = run("1 card, room unknown", {49743 * MiB}, SIZE_MAX, kfd_pod);
    CHECK(none1[0].mode == MOE_RAM_ARENA);
    // cgroup absent, MemAvailable only (a 128 GB machine, 2 cards)
    auto nocg = run("2 cards, no cgroup, 120 GiB available", {27728 * MiB, 22015 * MiB}, 120 * GiB, kfd_pod);
    CHECK(nocg[0].mode == MOE_RAM_PIN_ALL && nocg[1].mode == MOE_RAM_PIN_ALL);
    // KFD's limit binds (ROCm, 2 cards): RAM 50 GiB -> limit 50 - 0.78 - 1.5
    const size_t ram = 50 * GiB, kfd64 = ram - ram / 64 - 3 * (GiB / 2);
    auto kfd2 = run("2 cards, KFD limit binds", {27728 * MiB, 22015 * MiB}, 90 * GiB, kfd64);
    CHECK(kfd2[0].mode == MOE_RAM_PIN_PART && kfd2[0].quota + kfd2[1].quota <= kfd64 - GiB);
    // the total unknown (an old caller): the rule of before per card
    run("2 cards, total unknown (legacy)", {27728 * MiB, 22015 * MiB}, 60168 * MiB, kfd_pod, false);

    // moe_ram_select: ranges kept within the quota, in order, a large one skipped for smaller ones after it
    {
        name = "select";
        std::vector<std::pair<uintptr_t, uintptr_t>> r = {{0, 1000 * MiB}, {1000 * MiB, 3000 * MiB}, {3000 * MiB, 3500 * MiB}};
        std::vector<char> keep(r.size());
        const size_t kept = moe_ram_select(r, 1600 * MiB, keep);
        CHECK(keep[0] && !keep[1] && keep[2] && kept == 1500 * MiB);
        CHECK(moe_ram_select(r, 0, keep) == 0 && !keep[0] && !keep[1] && !keep[2]);
    }

    // random sessions
    const int n = argc > 1 ? atoi(argv[1]) : 200000;
    std::mt19937_64 rng(1);
    char buf[64];
    name = buf;
    for (int it = 0; it < n; it++) {
        snprintf(buf, sizeof(buf), "random %d", it);
        const int cards = 1 + (int) (rng() % 4);
        std::vector<size_t> b(cards);
        size_t total = 0;
        for (auto & x : b) {
            x = (rng() % (60 * 1024)) * MiB + (rng() % 2) * 4096;
            total += x;
        }
        const size_t room = rng() % 16 == 0 ? SIZE_MAX : (rng() % (200 * 1024)) * MiB;
        const size_t kfd = (rng() % (256 * 1024)) * MiB;
        const bool known = rng() % 8 != 0;
        moe_ram_plan plan;
        plan.total = known ? total : 0;
        plan.n_devices = cards;
        const size_t r1 = moe_ram_reserve_bytes(1, 4096, 0, 0), rm = moe_ram_reserve_bytes(cards, 4096, 10240, 3072);
        size_t sum = 0;
        for (int i = 0; i < cards; i++) {
            const moe_ram_choice c = moe_ram_choose(plan, b[i], room, kfd, r1, rm);
            if (cards == 1 || !known) {
                CHECK((c.mode == MOE_RAM_PIN_ALL) == legacy_fit(b[i], room, kfd, 4096 * MiB));
                CHECK(c.mode != MOE_RAM_PIN_PART && c.quota == b[i]);
            } else {
                CHECK(c.mode != MOE_RAM_ARENA);
                sum += c.quota;
                CHECK(sum <= plan.budget);
                CHECK(plan.all_fit == (total <= plan.budget));
                CHECK(!plan.all_fit || (c.mode == MOE_RAM_PIN_ALL && c.quota == b[i]));
                CHECK(c.mode == MOE_RAM_PIN_ALL ? c.quota == b[i] : c.quota < b[i]);
                // a part is the card's share by its bytes (or what the budget still has)
                CHECK(c.quota <= (size_t) ((long double) plan.budget * b[i] / total) + 1);
            }
            // a card's ranges under its quota
            std::vector<std::pair<uintptr_t, uintptr_t>> r;
            uintptr_t at = 0;
            for (int k = 0; k < 1 + (int) (rng() % 30); k++) {
                const uintptr_t len = (1 + rng() % 2000) * MiB;
                r.push_back({at, at + len});
                at += len + (rng() % 2) * 4096;
            }
            std::vector<char> keep(r.size());
            const size_t kept = moe_ram_select(r, c.quota, keep);
            size_t check = 0;
            for (size_t k = 0; k < r.size(); k++) {
                check += keep[k] ? r[k].second - r[k].first : 0;
            }
            CHECK(kept == check && kept <= c.quota);
        }
    }
    printf("random: %d sessions, every invariant held\n", n);
    return 0;
}
