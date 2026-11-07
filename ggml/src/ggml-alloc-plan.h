#pragma once

// Offline placement of a graph's buffer blocks (ggml-alloc's reservation): every block's size and lifetime are known
// once the graph has been walked, so the blocks are placed from the largest down, each at the lowest address free of
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

static int ggml_alloc_plan_cmp_size(const void * a, const void * b, void * blocks_v) {
    const struct ggml_alloc_plan_block * blocks = (const struct ggml_alloc_plan_block *) blocks_v;
    const struct ggml_alloc_plan_block * x = &blocks[*(const int *) a];
    const struct ggml_alloc_plan_block * y = &blocks[*(const int *) b];
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

// the neighbours of a block by address: a few dozen at a time, where qsort (glibc: a buffer allocated and copied per
// call) took most of the plan's time; insertion sort up to 32, a Shell sort above
static void ggml_alloc_plan_sort_iv(struct ggml_alloc_plan_iv * v, int m) {
    int gap = 1;
    if (m > 32) {
        while (gap < m / 3) {
            gap = 3*gap + 1;
        }
    }
    for (; gap >= 1; gap /= 3) {
        for (int i = gap; i < m; i++) {
            const struct ggml_alloc_plan_iv x = v[i];
            int j = i;
            while (j >= gap && v[j - gap].lo > x.lo) {
                v[j] = v[j - gap];
                j -= gap;
            }
            v[j] = x;
        }
    }
}

// the lowest start >= a at which size bytes stay inside one chunk of cap bytes
static size_t ggml_alloc_plan_chunk_fit(size_t a, size_t size, size_t cap) {
    if ((a % cap) + size > cap) {
        a = (a / cap + 1) * cap;
    }
    return a;
}

// a merge sort of the block indices keeps this header free of qsort_r, which differs between platforms
static void ggml_alloc_plan_sort_order(int * order, int n, const struct ggml_alloc_plan_block * blocks) {
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
                tmp[k++] = ggml_alloc_plan_cmp_size(&order[j], &order[i], (void *) blocks) < 0 ? order[j++] : order[i++];
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
// place looks only at the buckets of its own life (most tensors live a few nodes: a few dozen neighbours instead of
// every placed block, 15 ms -> well under 1 ms for qwen4exp's 3 469 blocks); the long-lived ones (more than
// GGML_ALLOC_PLAN_LONG buckets: the pipeline's copies, the graph's kept inputs) are looked at by every block
struct ggml_alloc_plan_bucket {
    int * ids;
    int n, cap;
};

#define GGML_ALLOC_PLAN_BUCKETS 256
#define GGML_ALLOC_PLAN_LONG    32

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

// Places the n blocks; chunk_cap is raised to the largest block when one exceeds it. Fills chunk_sizes[0..n_chunks)
// (each the end of its highest block) and returns n_chunks, or -1 when more than max_chunks would be needed or on an
// allocation failure (the caller keeps its own placement).
static int ggml_alloc_plan(struct ggml_alloc_plan_block * blocks, int n, size_t chunk_cap, size_t * chunk_sizes, int max_chunks) {
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
    struct ggml_alloc_plan_iv * ivs = (struct ggml_alloc_plan_iv *) malloc((size_t) nn * sizeof(struct ggml_alloc_plan_iv));
    struct ggml_alloc_plan_bucket * bks = (struct ggml_alloc_plan_bucket *) calloc(GGML_ALLOC_PLAN_BUCKETS + 1, sizeof(struct ggml_alloc_plan_bucket));
    int ok = order && stamp && start && ivs && bks;
    struct ggml_alloc_plan_bucket * longs = bks ? &bks[GGML_ALLOC_PLAN_BUCKETS] : NULL;
    int n_chunks = 0;
    if (ok) {
        for (int i = 0; i < n; i++) {
            order[i] = i;
            stamp[i] = -1;
        }
        ggml_alloc_plan_sort_order(order, n, blocks);
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
        // the placed blocks alive at the same time: those listed in this block's buckets, and the long-lived ones
        int m = 0;
        for (int pass = 0; pass < 2; pass++) {
            const int from = pass == 0 ? lo_bk : GGML_ALLOC_PLAN_BUCKETS;
            const int to   = pass == 0 ? hi_bk : GGML_ALLOC_PLAN_BUCKETS;
            for (int bi = from; bi <= to; bi++) {
                const struct ggml_alloc_plan_bucket * bk = &bks[bi];
                for (int q = 0; q < bk->n; q++) {
                    const int pid = bk->ids[q];
                    if (stamp[pid] == k) {
                        continue;
                    }
                    stamp[pid] = k;
                    const struct ggml_alloc_plan_block * p = &blocks[pid];
                    if (p->t0 < b->t1 && b->t0 < p->t1 && p->size > 0) {
                        ivs[m].lo = start[pid];
                        ivs[m].hi = start[pid] + p->size;
                        m++;
                    }
                }
            }
        }
        ggml_alloc_plan_sort_iv(ivs, m);
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
        // list it where the next blocks will look
        if (hi_bk - lo_bk + 1 > GGML_ALLOC_PLAN_LONG) {
            ok = ggml_alloc_plan_push(longs, id);
        } else {
            for (int bi = lo_bk; bi <= hi_bk && ok; bi++) {
                ok = ggml_alloc_plan_push(&bks[bi], id);
            }
        }
    }
    if (bks) {
        for (int bi = 0; bi <= GGML_ALLOC_PLAN_BUCKETS; bi++) {
            free(bks[bi].ids);
        }
    }
    free(order); free(stamp); free(start); free(ivs); free(bks);
    return ok ? n_chunks : -1;
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
    ggml_alloc_plan_sort_order(order, n, blocks);
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
