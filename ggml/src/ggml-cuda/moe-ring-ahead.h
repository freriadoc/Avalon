#pragma once

// The prefill ring's reads ahead, with the expert arena (moe-arena.h) holding only a part of the experts.
//
// The ring (GGML_CUDA_MOE_RING_PREFILL) copies to the GPU, layer after layer, every expert the VRAM cache does not hold.
// One the arena does not hold either must be read from the model file with O_DIRECT into a slot of the zone the arena
// lends a prefill, and copied from there. This thread does those reads before the ring needs them: it walks the plan
// (the layers of the graph, in order) and, for each, reads the experts that are neither cached nor in the arena into
// the zone, so that the disk works while the GPU computes and the host queues the previous layers.
//
// - A slot read ahead is held (a reader) until the layer's block takes it over (take_locked) or the plan is dropped
//   (cancel_locked): the zone never recycles it meanwhile.
// - The zone bounds how far ahead the thread runs. Full, it waits (cv) for the ring to free a block's slots (freed()).
// - The ring's block of a layer waits for the thread to be done with that layer (wait()). It first frees the previous
//   block's slots (the caller's free_previous, run without the lock); if the zone is still full then, the thread leaves
//   the rest of the layer to the block (waiting == the layer): no deadlock, whatever the zone's size.
// - A dropped plan (a new graph, generation starts, the zones go back) bumps gen: what the thread was reading for it is
//   aborted when its reads return.
//
// Every member is used under the caller's lock (*mu, the cache session's), which cv is waited for with; load() alone runs
// without it. Host only, no HIP: tests/ringaheadtest.cpp exercises it with ASan + UBSan and with TSan.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

template <typename Arena, typename Key, typename Job>
struct moe_ring_ahead {
    struct layer {
        const void * id = nullptr;   // the caller's layer
        Arena * ar = nullptr;
        int pool = -1;
        int32_t n_expert = 0;
        bool done = false;           // read as far as the zone allowed
        bool consumed = false;       // its block took its slots
        std::vector<std::pair<int32_t, int>> pins;   // (expert, arena slot), a reader held on each
    };
    struct read {
        int slot;
        int32_t e;
        Job job;
    };

    // set once, before start_locked()
    std::function<bool()> usable;                                         // locked: false, the layer is left to its block
    std::function<bool(const layer &, int32_t, Job &)> wanted;            // locked: expert e is to be read (not cached), its job
    std::function<Key(const layer &, int32_t)> key;                       // locked
    std::function<void(Arena &)> lend;                                    // locked: the zone is needed (lent if it is not)
    std::function<void(const layer &, const std::vector<read> &, std::vector<char> &)> load;   // without the lock: ok[b]

    std::vector<layer> plan;
    size_t next = 0;      // the layer the thread reads
    uint64_t gen = 0;     // bumped when the plan is dropped
    int waiting = -1;     // the layer whose block waits for the thread, the previous block's slots freed
    bool stop = false;
    std::mutex * mu = nullptr;
    std::condition_variable cv;
    std::thread thread;
    uint64_t loads = 0, failures = 0, left = 0;   // experts read ahead, reads failed, experts left to their block

    // the plan dropped: the slots held ahead are released
    void cancel_locked() {
        gen++;
        for (auto & L : plan) {
            for (const auto & pin : L.pins) {
                L.ar->release(pin.second);
            }
        }
        plan.clear();
        next = 0;
        waiting = -1;
        cv.notify_all();
    }

    void set_plan_locked(std::vector<layer> p) {
        cancel_locked();
        plan = std::move(p);
        cv.notify_all();
    }

    void start_locked(std::mutex & m) {
        if (!thread.joinable()) {
            mu = &m;
            thread = std::thread([this] { run(); });
        }
    }

    // not under the lock
    void stop_join() {
        if (!thread.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(*mu);
            stop = true;
            cancel_locked();
        }
        cv.notify_all();
        thread.join();
    }

    // the first layer of the plan with this id whose block has not come yet, or -1
    int find_locked(const void * id) const {
        for (size_t q = 0; q < plan.size(); q++) {
            if (plan[q].id == id && !plan[q].consumed) {
                return (int) q;
            }
        }
        return -1;
    }

    // the block of layer idx waits for the thread to be done with it; free_previous() (the previous block's slots
    // released, without the lock) runs first when it is not. False when the plan was dropped meanwhile
    template <typename F>
    bool wait(std::unique_lock<std::mutex> & lk, int idx, F && free_previous) {
        const uint64_t g = gen;
        if (!plan[idx].done) {
            lk.unlock();
            free_previous();
            lk.lock();
            if (g != gen) {
                return false;
            }
            waiting = idx;
            cv.notify_all();
            cv.wait(lk, [&] { return g != gen || plan[idx].done; });
            if (g != gen) {
                return false;
            }
            waiting = -1;
        }
        return true;
    }

    // the block takes the layer's slots over (their readers are now the block's)
    std::vector<std::pair<int32_t, int>> take_locked(int idx) {
        std::vector<std::pair<int32_t, int>> pins = std::move(plan[idx].pins);
        plan[idx].pins.clear();
        plan[idx].consumed = true;
        return pins;
    }

    // slots of the zone were released (not under the lock): the thread, if it waits for room, tries again
    void freed() {
        if (mu) {
            { std::lock_guard<std::mutex> lock(*mu); }   // the thread is either waiting or will see the freed slots
            cv.notify_all();
        }
    }

    void run() {
        std::unique_lock<std::mutex> lk(*mu);
        for (;;) {
            cv.wait(lk, [&] { return stop || next < plan.size(); });
            if (stop) {
                return;
            }
            const uint64_t g = gen;
            const size_t idx = next;
            layer L;   // the layer without its pins (the plan may be dropped while the reads run)
            L.id = plan[idx].id;
            L.ar = plan[idx].ar;
            L.pool = plan[idx].pool;
            L.n_expert = plan[idx].n_expert;
            std::vector<std::pair<int32_t, Job>> todo;
            if (!usable || usable()) {
                for (int32_t e = 0; e < L.n_expert; e++) {
                    Job j{};
                    if (!L.ar->map.count(key(L, e)) && wanted(L, e, j)) {
                        todo.push_back({e, j});
                    }
                }
            }
            size_t k = 0;
            while (k < todo.size() && g == gen && !stop) {
                std::vector<read> batch;
                lend(*L.ar);
                for (; k < todo.size(); k++) {
                    const Key ek = key(L, todo[k].first);
                    if (L.ar->map.count(ek)) {
                        continue;   // read meanwhile (the CPU path)
                    }
                    const int t = L.ar->transit_take(ek, L.pool);
                    if (t < 0) {
                        break;   // the zone is full
                    }
                    batch.push_back({t, todo[k].first, todo[k].second});
                }
                if (batch.empty()) {
                    if (k >= todo.size()) {
                        break;
                    }
                    if (waiting == (int) idx) {
                        left += todo.size() - k;
                        break;   // its block waits and has freed what it could: the rest is the block's
                    }
                    cv.wait(lk);   // for the ring to free a block's slots, or for this layer's block
                    continue;
                }
                std::vector<char> ok(batch.size(), 0);
                lk.unlock();
                load(L, batch, ok);
                lk.lock();
                const bool current = g == gen && !stop;
                for (size_t b = 0; b < batch.size(); b++) {
                    if (ok[b] && current) {
                        L.ar->commit(batch[b].slot);
                        const int a = L.ar->acquire(key(L, batch[b].e), false);
                        plan[idx].pins.push_back({batch[b].e, a});
                        loads++;
                    } else {
                        L.ar->abort(batch[b].slot);
                        failures += !ok[b];
                    }
                }
                cv.notify_all();
            }
            if (g == gen && !stop) {
                plan[idx].done = true;
                next = idx + 1;
                cv.notify_all();
            }
        }
    }
};
