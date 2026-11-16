#pragma once

// Offline placement of a graph's buffer blocks (ggml-alloc's reservation): every block's size and lifetime are known
// once the graph has been walked, so the blocks are placed in a set order (from the largest down, or one of ggml_alloc_plan's others), each at the lowest address free of
// the blocks already placed that are alive at the same time, never across a chunk boundary (a chunk is a backend
// buffer of its own, at most chunk_cap bytes). The walk-order placement it replaces decided each block on arrival,
// without the sizes to come: on qwen4exp's prefill graph 3 075 MiB for 2 666 live at most, and up to 4 180 MiB after
// changes that did not raise the live peak (two inputs fewer at the start of the graph). The greedy-by-size rule is
// the one of TFLite's and TVM's memory planners. Header-only, plain C, so that a standalone bench (tests/allocplan)
// runs the same code.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_alloc_plan_block {
    size_t  size;   // bytes, a multiple of the alignment
    int64_t t0;     // allocated at this event
    int64_t t1;     // freed at this event (INT64_MAX: never)
    // result
    int     chunk;
    size_t  offset;
};

struct ggml_alloc_plan_iv {
    size_t lo, hi; // [lo, hi) in the plan's linear address space
};

// the rank of a block: its size, weighted by its life up to life_cap events (0: the size alone)
static uint64_t ggml_alloc_plan_rank(const struct ggml_alloc_plan_block * x, int64_t life_cap) {
    const int64_t life = x->t1 - x->t0;
    return (uint64_t) x->size * (uint64_t) (life_cap <= 0 ? 1 : (life < life_cap ? (life > 0 ? life : 1) : life_cap));
}

struct ggml_alloc_plan_sort {
    const struct ggml_alloc_plan_block * blocks;
    int64_t life_cap;
};

static int ggml_alloc_plan_cmp_size(const void * a, const void * b, void * sort_v) {
    const struct ggml_alloc_plan_sort * sort = (const struct ggml_alloc_plan_sort *) sort_v;
    const struct ggml_alloc_plan_block * x = &sort->blocks[*(const int *) a];
    const struct ggml_alloc_plan_block * y = &sort->blocks[*(const int *) b];
    const uint64_t rx = ggml_alloc_plan_rank(x, sort->life_cap), ry = ggml_alloc_plan_rank(y, sort->life_cap);
    if (rx != ry) {
        return rx > ry ? -1 : 1;
    }
    if (x->size != y->size) {
        return x->size > y->size ? -1 : 1;
    }
    // equal sizes: the longer-lived first, then the earlier
    const int64_t lx = x->t1 - x->t0, ly = y->t1 - y->t0;
    if (lx != ly) {
        return lx > ly ? -1 : 1;
    }
    return x->t0 < y->t0 ? -1 : (x->t0 > y->t0 ? 1 : 0);
}

static int ggml_alloc_plan_cmp_iv(const void * a, const void * b) {
    const struct ggml_alloc_plan_iv * x = (const struct ggml_alloc_plan_iv *) a;
    const struct ggml_alloc_plan_iv * y = (const struct ggml_alloc_plan_iv *) b;
    return x->lo < y->lo ? -1 : (x->lo > y->lo ? 1 : 0);
}

// the lowest start >= a at which size bytes stay inside one chunk of cap bytes
static size_t ggml_alloc_plan_chunk_fit(size_t a, size_t size, size_t cap) {
    if ((a % cap) + size > cap) {
        a = (a / cap + 1) * cap;
    }
    return a;
}

// a merge sort of the block indices keeps this header free of qsort_r, which differs between platforms
static void ggml_alloc_plan_sort_order(int * order, int n, const struct ggml_alloc_plan_block * blocks, int64_t life_cap) {
    struct ggml_alloc_plan_sort sort = { blocks, life_cap };
    if (n < 2) {
        return;
    }
    int * tmp = (int *) malloc((size_t) n * sizeof(int));
    if (!tmp) {
        return;
    }
    for (int w = 1; w < n; w *= 2) {
        for (int lo = 0; lo < n; lo += 2*w) {
            const int mid = lo + w < n ? lo + w : n;
            const int hi  = lo + 2*w < n ? lo + 2*w : n;
            int i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                tmp[k++] = ggml_alloc_plan_cmp_size(&order[j], &order[i], (void *) &sort) < 0 ? order[j++] : order[i++];
            }
            while (i < mid) tmp[k++] = order[i++];
            while (j < hi)  tmp[k++] = order[j++];
        }
        for (int i = 0; i < n; i++) {
            order[i] = tmp[i];
        }
    }
    free(tmp);
}

// Places the n blocks; chunk_cap is raised to the largest block when one exceeds it. Fills chunk_sizes[0..n_chunks)
// (each the end of its highest block) and returns n_chunks, or -1 when more than max_chunks would be needed or on an
// allocation failure (the caller keeps its own placement).
// the placed blocks are found by time: each is listed in the buckets of the time axis its life covers, and a block to
// place looks only at the buckets of its own life; the long-lived ones (more than GGML_ALLOC_PLAN_LONG buckets: the
// pipeline's copies, the graph's kept inputs) are in a list of their own, looked at by every block. Every list is kept
// sorted by address: a block's lists are walked together up to the first hole that fits, nothing is gathered or sorted
// per block. qwen4exp's prefill graph at 32768 tokens (8 674 blocks, 1 284 of them long-lived, ~600 alive with a block
// on average): 235 ms -> 30 ms, the same placement
struct ggml_alloc_plan_bucket {
    int * ids;
    int n, cap;
};

#define GGML_ALLOC_PLAN_BUCKETS 256
#define GGML_ALLOC_PLAN_LONG    32
#define GGML_ALLOC_PLAN_WALK    4    // buckets a block may span for its lists to be walked

static int ggml_alloc_plan_push(struct ggml_alloc_plan_bucket * bk, int id) {
    if (bk->n == bk->cap) {
        const int cap = bk->cap ? 2*bk->cap : 16;
        int * ids = (int *) realloc(bk->ids, (size_t) cap * sizeof(int));
        if (!ids) {
            return 0;
        }
        bk->ids = ids;
        bk->cap = cap;
    }
    bk->ids[bk->n++] = id;
    return 1;
}

// a block added to a list kept sorted by address
static int ggml_alloc_plan_push_sorted(struct ggml_alloc_plan_bucket * bk, int id, const size_t * start) {
    if (!ggml_alloc_plan_push(bk, id)) {
        return 0;
    }
    int j = bk->n - 1;
    for (; j > 0 && start[bk->ids[j - 1]] > start[id]; j--) {
        bk->ids[j] = bk->ids[j - 1];
    }
    bk->ids[j] = id;
    return 1;
}

// Places the n blocks; chunk_cap is raised to the largest block when one exceeds it. Fills chunk_sizes[0..n_chunks)
// (each the end of its highest block) and returns n_chunks, or -1 when more than max_chunks would be needed or on an
// allocation failure (the caller keeps its own placement).
// life_cap: the order of placement, see ggml_alloc_plan_rank
static int ggml_alloc_plan_order(struct ggml_alloc_plan_block * blocks, int n, size_t chunk_cap, size_t * chunk_sizes, int max_chunks,
        int64_t life_cap) {
    size_t cap = chunk_cap;
    int64_t tmax = 1;
    for (int i = 0; i < n; i++) {
        if (blocks[i].size > cap) {
            cap = blocks[i].size;
        }
        if (blocks[i].t0 + 1 > tmax) {
            tmax = blocks[i].t0 + 1;
        }
        if (blocks[i].t1 != INT64_MAX && blocks[i].t1 > tmax) {
            tmax = blocks[i].t1;
        }
    }
    if (cap == 0) {
        cap = 1;
    }
    const int64_t width = (tmax + GGML_ALLOC_PLAN_BUCKETS - 1) / GGML_ALLOC_PLAN_BUCKETS;
    const int nn = n > 0 ? n : 1;
    int * order = (int *) malloc((size_t) nn * sizeof(int));
    int * stamp = (int *) malloc((size_t) nn * sizeof(int));
    size_t * start = (size_t *) malloc((size_t) nn * sizeof(size_t));
    struct ggml_alloc_plan_bucket * bks = (struct ggml_alloc_plan_bucket *) calloc(GGML_ALLOC_PLAN_BUCKETS + 2, sizeof(struct ggml_alloc_plan_bucket));
    int ok = order && stamp && start && bks;
    struct ggml_alloc_plan_bucket * longs = bks ? &bks[GGML_ALLOC_PLAN_BUCKETS] : NULL;
    struct ggml_alloc_plan_bucket * all   = bks ? &bks[GGML_ALLOC_PLAN_BUCKETS + 1] : NULL;
    int n_chunks = 0;
    if (ok) {
        for (int i = 0; i < n; i++) {
            order[i] = i;
            stamp[i] = -1;
        }
        ggml_alloc_plan_sort_order(order, n, blocks, life_cap);
        for (int c = 0; c < max_chunks; c++) {
            chunk_sizes[c] = 0;
        }
    }
    for (int k = 0; k < n && ok; k++) {
        const int id = order[k];
        struct ggml_alloc_plan_block * b = &blocks[id];
        const int64_t b_end = b->t1 == INT64_MAX ? tmax : b->t1;
        const int lo_bk = (int) (b->t0 / width);
        const int hi_bk = (int) ((b_end - 1) / width);
        // the placed blocks alive at the same time: those listed in this block's buckets and the long-lived ones. Every
        // list is kept sorted by address as the blocks are placed, and the lists are walked together in address order,
        // up to the first hole that fits: no list is gathered and sorted per block (in qwen4exp's prefill graph of
        // 32768 tokens a block has ~600 placed blocks alive with it and over a thousand long-lived ones, the copies of
        // the layers' inputs; gathering and sorting them for each of 8 674 blocks took 56 to 111 ms a ubatch)
        const int n_list = hi_bk - lo_bk + 2;
        int pos[GGML_ALLOC_PLAN_BUCKETS + 2];
        for (int l = 0; l < n_list; l++) {
            pos[l] = 0;
        }
        size_t a = 0;
        if (n_list > GGML_ALLOC_PLAN_WALK + 1) {
            // a block over many buckets (the long-lived ones above all, placed last: each is alive with nearly every
            // block): the list of every placed block, sorted by address, walked alone
            for (int q = 0; q < all->n; q++) {
                const int pid = all->ids[q];
                const struct ggml_alloc_plan_block * p = &blocks[pid];
                if (!(p->t0 < b->t1 && b->t0 < p->t1 && p->size > 0)) {
                    continue;
                }
                a = ggml_alloc_plan_chunk_fit(a, b->size, cap);
                if (start[pid] >= a + b->size) {
                    break;
                }
                if (start[pid] + p->size > a) {
                    a = start[pid] + p->size;
                }
            }
        } else
        for (;;) {
            // the lowest next block of the lists, alive with this one and not seen in another bucket
            int best = -1;
            size_t best_lo = 0;
            for (int l = 0; l < n_list; l++) {
                const struct ggml_alloc_plan_bucket * bk = l == n_list - 1 ? longs : &bks[lo_bk + l];
                while (pos[l] < bk->n) {
                    const int pid = bk->ids[pos[l]];
                    const struct ggml_alloc_plan_block * p = &blocks[pid];
                    if (stamp[pid] != k && p->t0 < b->t1 && b->t0 < p->t1 && p->size > 0) {
                        break;
                    }
                    pos[l]++;
                }
                if (pos[l] < bk->n && (best < 0 || start[bk->ids[pos[l]]] < best_lo)) {
                    best = l;
                    best_lo = start[bk->ids[pos[l]]];
                }
            }
            if (best < 0) {
                break;
            }
            const struct ggml_alloc_plan_bucket * bk = best == n_list - 1 ? longs : &bks[lo_bk + best];
            const int pid = bk->ids[pos[best]++];
            stamp[pid] = k;
            a = ggml_alloc_plan_chunk_fit(a, b->size, cap);
            if (best_lo >= a + b->size) {
                break;
            }
            if (best_lo + blocks[pid].size > a) {
                a = best_lo + blocks[pid].size;
            }
        }
        a = ggml_alloc_plan_chunk_fit(a, b->size, cap);
        start[id] = a;
        b->chunk  = (int) (a / cap);
        b->offset = a % cap;
        if (b->chunk >= max_chunks) {
            ok = 0;
            break;
        }
        if (b->chunk + 1 > n_chunks) {
            n_chunks = b->chunk + 1;
        }
        if (b->offset + b->size > chunk_sizes[b->chunk]) {
            chunk_sizes[b->chunk] = b->offset + b->size;
        }
        // list it where the next blocks will look, each list sorted by address
        ok = ggml_alloc_plan_push_sorted(all, id, start);
        if (!ok) {
            break;
        }
        if (hi_bk - lo_bk + 1 > GGML_ALLOC_PLAN_LONG) {
            ok = ggml_alloc_plan_push_sorted(longs, id, start);
        } else {
            for (int bi = lo_bk; bi <= hi_bk && ok; bi++) {
                ok = ggml_alloc_plan_push_sorted(&bks[bi], id, start);
            }
        }
    }
    if (bks) {
        for (int bi = 0; bi <= GGML_ALLOC_PLAN_BUCKETS + 1; bi++) {
            free(bks[bi].ids);
        }
    }
    free(order); free(stamp); free(start); free(bks);
    return ok ? n_chunks : -1;
}

// The placement of the smallest total among a few orders: the largest block first (life_cap 0), and the size weighted by
// the life up to a few events, which ranks a short-lived temporary after the smaller blocks that outlive it. No order
// wins on every graph and the total moves by hundreds of MiB from one to the next (qwen4exp's prefill graph at 32768
// tokens, 7 029 MiB live at most: 7 541 by size, 7 156 / 7 029 / 7 361 / 7 259 / 7 349 at 4 / 8 / 16 / 32 / 64 events,
// 8 629 at 192), so each is tried, up to the first that reaches live_max (the bound of any placement, 0: unknown). A
// plan is made once per graph shape (ggml-alloc keeps it), ~30 ms per order on that graph.
// 8 first: it reaches live_max on qwen4exp's prompt graphs, whose plan is made again for the first prompt of each depth
// (10 ms instead of 20 with the size alone tried before it)
static const int64_t ggml_alloc_plan_life_caps[] = { 8, 0, 32, 4, 16, 64, 24, 256 };

static int ggml_alloc_plan(struct ggml_alloc_plan_block * blocks, int n, size_t chunk_cap, size_t * chunk_sizes, int max_chunks,
        size_t live_max) {
    const int n_orders = (int) (sizeof(ggml_alloc_plan_life_caps) / sizeof(ggml_alloc_plan_life_caps[0]));
    struct ggml_alloc_plan_block * best = (struct ggml_alloc_plan_block *) malloc((size_t) (n > 0 ? n : 1) * sizeof(struct ggml_alloc_plan_block));
    size_t * best_sizes = (size_t *) malloc((size_t) (max_chunks > 0 ? max_chunks : 1) * sizeof(size_t));
    if (!best || !best_sizes) {
        free(best); free(best_sizes);
        return ggml_alloc_plan_order(blocks, n, chunk_cap, chunk_sizes, max_chunks, 0);
    }
    int    best_chunks = -1;
    size_t best_total  = 0;
    for (int o = 0; o < n_orders; o++) {
        const int n_chunks = ggml_alloc_plan_order(blocks, n, chunk_cap, chunk_sizes, max_chunks, ggml_alloc_plan_life_caps[o]);
        if (n_chunks < 0) {
            continue;
        }
        size_t total = 0;
        for (int c = 0; c < n_chunks; c++) {
            total += chunk_sizes[c];
        }
        if (best_chunks < 0 || total < best_total) {
            best_chunks = n_chunks;
            best_total  = total;
            for (int i = 0; i < n; i++) {
                best[i] = blocks[i];
            }
            for (int c = 0; c < max_chunks; c++) {
                best_sizes[c] = chunk_sizes[c];
            }
        }
        if (best_total <= live_max) {
            break;
        }
    }
    if (best_chunks >= 0) {
        for (int i = 0; i < n; i++) {
            blocks[i] = best[i];
        }
        for (int c = 0; c < max_chunks; c++) {
            chunk_sizes[c] = best_sizes[c];
        }
    }
    free(best); free(best_sizes);
    return best_chunks;
}

// the plan before the buckets: every placed block tested, kept for tests/allocplan to check that both place alike
static int ggml_alloc_plan_ref(struct ggml_alloc_plan_block * blocks, int n, size_t chunk_cap, size_t * chunk_sizes, int max_chunks) {
    size_t cap = chunk_cap;
    for (int i = 0; i < n; i++) {
        if (blocks[i].size > cap) {
            cap = blocks[i].size;
        }
    }
    if (cap == 0) {
        cap = 1;
    }
    int * order = (int *) malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
    size_t * start = (size_t *) malloc((size_t) (n > 0 ? n : 1) * sizeof(size_t));
    struct ggml_alloc_plan_iv * ivs = (struct ggml_alloc_plan_iv *) malloc((size_t) (n > 0 ? n : 1) * sizeof(struct ggml_alloc_plan_iv));
    if (!order || !start || !ivs) {
        free(order); free(start); free(ivs);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        order[i] = i;
    }
    ggml_alloc_plan_sort_order(order, n, blocks, 0);
    int n_chunks = 0;
    for (int c = 0; c < max_chunks; c++) {
        chunk_sizes[c] = 0;
    }
    int ok = 1;
    for (int k = 0; k < n && ok; k++) {
        struct ggml_alloc_plan_block * b = &blocks[order[k]];
        int m = 0;
        for (int q = 0; q < k; q++) {
            const struct ggml_alloc_plan_block * p = &blocks[order[q]];
            if (p->t0 < b->t1 && b->t0 < p->t1 && p->size > 0) {
                ivs[m].lo = start[order[q]];
                ivs[m].hi = start[order[q]] + p->size;
                m++;
            }
        }
        qsort(ivs, (size_t) m, sizeof(ivs[0]), ggml_alloc_plan_cmp_iv);
        size_t a = 0;
        for (int q = 0; q < m; q++) {
            a = ggml_alloc_plan_chunk_fit(a, b->size, cap);
            if (ivs[q].lo >= a + b->size) {
                break;
            }
            if (ivs[q].hi > a) {
                a = ivs[q].hi;
            }
        }
        a = ggml_alloc_plan_chunk_fit(a, b->size, cap);
        start[order[k]] = a;
        b->chunk  = (int) (a / cap);
        b->offset = a % cap;
        if (b->chunk >= max_chunks) {
            ok = 0;
            break;
        }
        if (b->chunk + 1 > n_chunks) {
            n_chunks = b->chunk + 1;
        }
        if (b->offset + b->size > chunk_sizes[b->chunk]) {
            chunk_sizes[b->chunk] = b->offset + b->size;
        }
    }
    free(order); free(start); free(ivs);
    return ok ? n_chunks : -1;
}

#ifdef __cplusplus
}
#endif
