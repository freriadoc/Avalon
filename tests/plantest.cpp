// Planned seating of compute-buffer loans in the MoE cache pools.
//
// ggml-alloc knows every chunk it will ask for before the first request (ggml_vbuffer_alloc
// walks a finished list). The plan sees that list, picks the assignment of blocks to pools
// that minimises the most-lent pool, and each borrow is then served from the plan. The
// online rule (moe_cache_borrow as of 13225587) is kept as the fallback and as the reference.
//
// build: g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
//            tests/plantest.cpp -o /tmp/plantest && /tmp/plantest

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------- pool state, as in moe-cache.cu ----------------
static constexpr int    moe_cache_lend_percent   = 95;
static constexpr double moe_cache_lend_fairness  = 0.15;

struct pool_t {
    const char * name;
    size_t expert_size;
    int region0;                 // ranges[0].count
    int zone_begin;              // INT_MAX: no prefetch zone
    std::vector<std::pair<int, int>> lent;   // (begin, count)
    int lent_low = INT_MAX;
};

static int lend_low(const pool_t & p) {
    if (p.lent_low < p.region0) return p.lent_low;
    return p.zone_begin < p.region0 ? p.zone_begin : p.region0;
}
static int lend_floor(const pool_t & p) {
    return p.region0 - (int)((int64_t)p.region0 * moe_cache_lend_percent / 100);
}
static int lendable(const pool_t & p) {
    const int low = lend_low(p), fl = lend_floor(p);
    return low > fl ? low - fl : 0;
}
static int want_slots(const pool_t & p, size_t bytes) {
    return (int)((bytes + p.expert_size - 1) / p.expert_size);
}
static double lent_after(const pool_t & p, int want) {
    return (double)(p.region0 - (lend_low(p) - want)) / (double)p.region0;
}

// ---------------- the online rule (moe_cache_borrow) ----------------
static int seat_online(const std::vector<pool_t> & pools, size_t bytes) {
    double least = 1.0;
    for (const pool_t & p : pools) {
        const int w = want_slots(p, bytes);
        if (lendable(p) >= w) least = std::min(least, lent_after(p, w));
    }
    int chosen = -1; int64_t smallest = 0;
    for (size_t i = 0; i < pools.size(); i++) {
        const pool_t & p = pools[i];
        const int w = want_slots(p, bytes);
        if (lendable(p) < w || lent_after(p, w) > least + moe_cache_lend_fairness) continue;
        const int64_t room = (int64_t)lendable(p) * (int64_t)p.expert_size;
        if (chosen < 0 || room < smallest) { chosen = (int)i; smallest = room; }
    }
    return chosen;
}

// ---------------- the plan ----------------
// As moe_cache_plan_borrows: every assignment of the chunks to the pools, keeping the one
// whose most-lent pool is the least lent (then the smallest sum of squared loads). The
// chunks are placed one by one; a branch stops as soon as a pool would overflow or be more
// lent than the best complete assignment so far.
struct plan_t {
    std::vector<size_t> bytes;   // block sizes, in the order ggml-alloc will ask
    std::vector<int>    pool;    // pool of each block
    std::vector<bool>   served;
};

struct loan_search {
    size_t n = 0, np = 0;
    std::vector<int> want, room, low, region0, used, pick, best;
    double best_max = 2.0, best_sq = 0.0;

    double lent(size_t k, int extra) const {
        return (double)(region0[k] - (low[k] - used[k] - extra)) / (double)region0[k];
    }
    void place(size_t i) {
        if (i == n) {
            double max_lent = 0.0, sq = 0.0;
            for (size_t k = 0; k < np; k++) { max_lent = std::max(max_lent, lent(k, 0)); sq += lent(k, 0) * lent(k, 0); }
            if (max_lent < best_max || (max_lent == best_max && sq < best_sq)) { best_max = max_lent; best_sq = sq; best = pick; }
            return;
        }
        for (size_t k = 0; k < np; k++) {
            const int w = want[i * np + k];
            if (used[k] + w > room[k] || lent(k, w) > best_max) continue;
            used[k] += w; pick[i] = (int)k;
            place(i + 1);
            used[k] -= w;
        }
    }
};

static bool plan_loans(const std::vector<pool_t> & pools, const std::vector<size_t> & bytes, plan_t & out) {
    const size_t n = bytes.size(), np = pools.size();
    out = {};
    if (!n || !np) return false;
    loan_search s;
    s.n = n; s.np = np; s.want.resize(n * np); s.room.resize(np); s.low.resize(np); s.region0.resize(np);
    s.used.assign(np, 0); s.pick.resize(n);
    for (size_t k = 0; k < np; k++) {
        s.room[k] = lendable(pools[k]); s.low[k] = lend_low(pools[k]); s.region0[k] = pools[k].region0;
        for (size_t i = 0; i < n; i++) s.want[i * np + k] = want_slots(pools[k], bytes[i]);
    }
    s.place(0);
    if (s.best.empty()) return false;
    out.bytes = bytes; out.pool = s.best; out.served.assign(n, false);
    return true;
}

// reference: the plain loop over all np^n assignments, no cut (small n only)
static bool plan_reference(const std::vector<pool_t> & pools, const std::vector<size_t> & bytes, double & best_max) {
    const size_t n = bytes.size(), np = pools.size();
    size_t total = 1;
    for (size_t i = 0; i < n; i++) total *= np;
    best_max = 2.0;
    std::vector<int> used(np);
    for (size_t c = 0; c < total; c++) {
        std::fill(used.begin(), used.end(), 0);
        size_t x = c;
        for (size_t i = 0; i < n; i++, x /= np) used[x % np] += want_slots(pools[x % np], bytes[i]);
        bool fits = true; double m = 0;
        for (size_t k = 0; k < np; k++) {
            fits = fits && used[k] <= lendable(pools[k]);
            m = std::max(m, (double)(pools[k].region0 - (lend_low(pools[k]) - used[k])) / pools[k].region0);
        }
        if (fits) best_max = std::min(best_max, m);
    }
    return best_max < 2.0;
}

// ---------------- lending bookkeeping (borrow / give back) ----------------
static void lend(pool_t & p, int want) {
    const int low = lend_low(p);
    p.lent.push_back({low - want, want});
    p.lent_low = low - want;
}
static void give_back_all(pool_t & p) {
    p.lent.clear();
    p.lent_low = INT_MAX;
}

// one borrow: the plan first, the online rule when the request is not in the plan
static int borrow(std::vector<pool_t> & pools, plan_t * plan, size_t bytes) {
    int k = -1;
    if (plan) {
        for (size_t i = 0; i < plan->bytes.size(); i++) {
            if (plan->served[i] || plan->bytes[i] != bytes) continue;
            const int c = plan->pool[i];
            if (lendable(pools[c]) >= want_slots(pools[c], bytes)) { plan->served[i] = true; k = c; }
            break;
        }
    }
    if (k < 0) k = seat_online(pools, bytes);
    if (k >= 0) lend(pools[k], want_slots(pools[k], bytes));
    return k;
}

// ---------------- helpers ----------------
static const size_t MiB = 1u << 20, KiB = 1u << 10;

static std::vector<pool_t> real_pools(double scale = 1.0) {
    // CUDA0 pools after loading, reserve 2304 MiB (journal of the 23 Sept. start)
    std::vector<pool_t> v = {
        {"iq4_nl", 900 * KiB, 5928, 5415, {}, INT_MAX},
        {"iq2_s",  512 * KiB, 8862, 7962, {}, INT_MAX},
        {"iq3_s",  687 * KiB, 2997, INT_MAX, {}, INT_MAX},
    };
    if (scale != 1.0) {
        for (pool_t & p : v) {
            const int zone = p.zone_begin < p.region0 ? p.region0 - p.zone_begin : 0;
            p.region0 = (int)(p.region0 * scale);
            p.zone_begin = zone ? p.region0 - zone : INT_MAX;
        }
    }
    return v;
}
static std::vector<size_t> mib(std::initializer_list<int> l) {
    std::vector<size_t> v; for (int x : l) v.push_back((size_t)x * MiB); return v;
}
static const std::vector<size_t> BURST1 = mib({416, 503, 500, 400, 1024});
static const std::vector<size_t> BURST2 = mib({424, 480, 500, 512, 512, 1024, 2048, 2048});

// serve a burst in the given order; returns the number of refusals
static int serve(std::vector<pool_t> & pools, const std::vector<size_t> & req, bool planned,
                 bool verbose = false) {
    plan_t plan;
    const bool have = planned && plan_loans(pools, req, plan);
    int refused = 0;
    for (size_t b : req) {
        const int k = borrow(pools, have ? &plan : nullptr, b);
        if (k < 0) refused++;
        if (verbose) {
            if (k < 0) printf("    %5zu MiB  REFUSED\n", b / MiB);
            else printf("    %5zu MiB  -> %-6s (%3.0f%% lent)\n", b / MiB, pools[k].name,
                        100.0 * (pools[k].region0 - lend_low(pools[k])) / pools[k].region0);
        }
    }
    return refused;
}
static double max_load(const std::vector<pool_t> & pools) {
    double m = 0;
    for (const pool_t & p : pools) m = std::max(m, (double)(p.region0 - lend_low(p)) / p.region0);
    return m;
}
static void print_loads(const std::vector<pool_t> & pools) {
    for (const pool_t & p : pools)
        printf(" %s %.0f%%", p.name, 100.0 * (p.region0 - lend_low(p)) / p.region0);
    printf("\n");
}

int main() {
    int errors = 0;

    // A. the journal, replayed
    printf("A. start of 23 Sept., journal order\n");
    for (int planned = 0; planned < 2; planned++) {
        std::vector<pool_t> pools = real_pools();
        printf("  %s\n  burst 1\n", planned ? "planned" : "online rule");
        int r = serve(pools, BURST1, planned, true);
        for (pool_t & p : pools) give_back_all(p);
        printf("  burst 2\n");
        r += serve(pools, BURST2, planned, true);
        printf("  refused %d, loads", r); print_loads(pools);
        if (!planned && r != 1) { printf("  ERROR: the online rule should reproduce the one refusal of the journal\n"); errors++; }
        if (planned && r != 0) { printf("  ERROR: the plan refused a block\n"); errors++; }
    }

    // B. every order of burst 2
    {
        std::vector<size_t> req = BURST2; std::sort(req.begin(), req.end());
        int orders = 0, fail_online = 0, fail_plan = 0; double lo = 2, hi = 0;
        do {
            orders++;
            std::vector<pool_t> a = real_pools(), b = real_pools();
            if (serve(a, req, false)) fail_online++;
            if (serve(b, req, true)) fail_plan++;
            lo = std::min(lo, max_load(b)); hi = std::max(hi, max_load(b));
        } while (std::next_permutation(req.begin(), req.end()));
        printf("\nB. burst 2, %d orders: online rule fails %d, plan fails %d, plan max load %.1f..%.1f%%\n",
               orders, fail_online, fail_plan, 100 * lo, 100 * hi);
        if (fail_plan || lo != hi) { printf("  ERROR: the plan must not depend on the order\n"); errors++; }
    }

    // C. smaller caches: where does each policy stop fitting burst 2? The plan does not
    //    depend on the order (B), so it is computed once per size; the online rule is
    //    tried over every order.
    {
        printf("\nC. cache scaled down, burst 2\n");
        int first_fail_online = 0, first_fail_plan = 0;
        for (int pct = 100; pct >= 50 && !(first_fail_online && first_fail_plan); pct--) {
            std::vector<pool_t> b = real_pools(pct / 100.0);
            if (serve(b, BURST2, true) && !first_fail_plan) first_fail_plan = pct;
            std::vector<size_t> req = BURST2; std::sort(req.begin(), req.end());
            int fo = 0;
            do {
                std::vector<pool_t> a = real_pools(pct / 100.0);
                if (serve(a, req, false)) fo++;
            } while (!first_fail_online && std::next_permutation(req.begin(), req.end()));
            if (fo && !first_fail_online) first_fail_online = pct;
        }
        printf("  online rule first fails at %d%% of the cache, plan at %d%%\n", first_fail_online, first_fail_plan);
    }

    // D. random pools and bursts of 1..16 blocks: the plan is optimal (against the plain
    //    loop, up to 8 blocks); served in any order it never refuses, lands where it said,
    //    and does at least as well as the online rule
    {
        srand(1);
        int cases = 0, planned = 0, online_ok = 0;
        static const char * names[] = {"p0", "p1", "p2"};
        for (int t = 0; t < 20000; t++) {
            const int n_blocks = t % 4 == 3 ? 9 + rand() % 8 : 1 + rand() % 8;
            std::vector<pool_t> pools;
            for (int k = 0; k < 3; k++) {
                const size_t es = (size_t)(256 + rand() % 800) * KiB;
                const int r0 = 1000 + rand() % 9000;
                const int zone = rand() % 2 ? rand() % (r0 / 8) : 0;
                pools.push_back({names[k], es, r0, zone ? r0 - zone : INT_MAX, {}, INT_MAX});
            }
            std::vector<size_t> req;
            for (int i = 0; i < n_blocks; i++) {
                static const int sizes[] = {64, 128, 256, 400, 512, 512, 1024, 2048};
                req.push_back((size_t)(sizes[rand() % 8] + (rand() % 2 ? 0 : rand() % 100)) * MiB);
            }
            cases++;
            std::vector<pool_t> on = pools;
            const bool on_ok = serve(on, req, false) == 0;
            online_ok += on_ok;
            plan_t plan;
            const bool have = plan_loans(pools, req, plan);
            if (req.size() <= 8) {
                double ref;
                const bool rok = plan_reference(pools, req, ref);
                if (rok != have) { printf("  ERROR case %d: plan %d reference %d\n", t, have, rok); errors++; continue; }
                if (have) {
                    std::vector<int> u(3, 0);
                    for (size_t i = 0; i < req.size(); i++) u[plan.pool[i]] += want_slots(pools[plan.pool[i]], req[i]);
                    double m = 0;
                    for (int k = 0; k < 3; k++) m = std::max(m, (double)(pools[k].region0 - (lend_low(pools[k]) - u[k])) / pools[k].region0);
                    if (std::abs(m - ref) > 1e-12) { printf("  ERROR case %d: plan %.4f reference %.4f\n", t, m, ref); errors++; }
                }
            }
            if (!have) {
                if (on_ok) { printf("  ERROR case %d: no plan but the online rule placed everything\n", t); errors++; }
                continue;
            }
            planned++;
            std::vector<int> used(3, 0);
            for (size_t i = 0; i < req.size(); i++) used[plan.pool[i]] += want_slots(pools[plan.pool[i]], req[i]);
            double m = 0;
            for (int k = 0; k < 3; k++) m = std::max(m, (double)(pools[k].region0 - (lend_low(pools[k]) - used[k])) / pools[k].region0);
            if (on_ok && m > max_load(on) + 1e-12) { printf("  ERROR case %d: plan %.4f worse than online %.4f\n", t, m, max_load(on)); errors++; }
            std::vector<size_t> shuffled = req;
            for (size_t i = shuffled.size(); i > 1; i--) std::swap(shuffled[i - 1], shuffled[rand() % i]);
            std::vector<pool_t> live = pools;
            int refused = 0;
            for (size_t b : shuffled) if (borrow(live, &plan, b) < 0) refused++;
            if (refused || std::abs(max_load(live) - m) > 1e-12) {
                printf("  ERROR case %d: %d refused, load %.4f vs plan %.4f\n", t, refused, max_load(live), m); errors++;
            }
        }
        printf("\nD. %d random cases: %d with a plan, %d placed by the online rule\n", cases, planned, online_ok);
    }

    // E. requests outside the plan go to the online rule
    {
        std::vector<pool_t> pools = real_pools();
        plan_t plan; plan_loans(pools, BURST1, plan);
        int refused = 0;
        for (size_t b : BURST2) if (borrow(pools, &plan, b) < 0) refused++;
        printf("\nE. burst 2 served against the plan of burst 1: %d refused (online rule fallback)\n", refused);
    }

    printf("\nerrors=%d\n", errors);
    return errors != 0;
}
