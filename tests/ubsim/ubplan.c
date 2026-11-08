// ubplan DUMP [CAP_MIB...] : the blocks of each reservation in a GGML_ALLOC_PLAN_DUMP file (tests/ubsim/ubsim.cpp, or
// any run with the variable set), placed offline by ggml-alloc's own planner at each chunk cap; the live peak (the
// lower bound of any placement) and the tensors alive at it, grouped by name without the layer number.
//   cc -O2 -I ggml/src tests/ubsim/ubplan.c -o ubplan
// UBPLAN_ALL=1: every allocator of the file (default: the device one with the largest live peak); UBPLAN_TOP=N: rows
#include "ggml-alloc-plan.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct blk { size_t size; int64_t t0, t1; char op[48], type[16], name[96]; };
struct sec { char alloc[96]; size_t cap, walk, live_max; int n; struct blk * b; };

static void base_name(const char * in, char * out, size_t n) {
    // "ffn_moe_down-17" -> "ffn_moe_down-#", "attn_norm-3 (view)" -> "attn_norm-# (view)"
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 2 < n; i++) {
        if (in[i] == '-' && isdigit((unsigned char) in[i + 1])) {
            out[j++] = '-'; out[j++] = '#';
            while (isdigit((unsigned char) in[i + 1])) i++;
        } else {
            out[j++] = in[i];
        }
    }
    out[j] = 0;
}

static int cmp_sz(const void * a, const void * b) {
    const struct blk * x = *(const struct blk * const *) a, * y = *(const struct blk * const *) b;
    return x->size < y->size ? 1 : x->size > y->size ? -1 : 0;
}

struct grp { char name[96]; char op[48]; size_t bytes; int n; size_t biggest; };
static int cmp_grp(const void * a, const void * b) {
    const struct grp * x = (const struct grp *) a, * y = (const struct grp *) b;
    return x->bytes < y->bytes ? 1 : x->bytes > y->bytes ? -1 : 0;
}

static void analyse(const struct sec * s, const double * caps, int ncaps, int top) {
    const double M = 1048576.0;
    printf("== %s: %d blocks, walk %.0f MiB, live max %.0f MiB (cap in the run %.0f MiB)\n", s->alloc, s->n, s->walk / M,
           s->live_max / M, s->cap / M);
    // the live peak by a sweep over the events
    int64_t tmax = 0;
    for (int i = 0; i < s->n; i++) {
        if (s->b[i].t0 > tmax) tmax = s->b[i].t0;
        if (s->b[i].t1 != INT64_MAX && s->b[i].t1 > tmax) tmax = s->b[i].t1;
    }
    long long * delta = calloc((size_t) tmax + 2, sizeof(long long));
    for (int i = 0; i < s->n; i++) {
        delta[s->b[i].t0] += (long long) s->b[i].size;
        if (s->b[i].t1 != INT64_MAX) delta[s->b[i].t1] -= (long long) s->b[i].size;
    }
    long long live = 0, peak = 0; int64_t tpeak = 0;
    for (int64_t t = 0; t <= tmax; t++) {
        live += delta[t];   // a block freed at t is gone before one born at t (as the walk frees before it allocates)
        if (live > peak) { peak = live; tpeak = t; }
    }
    free(delta);
    printf("   live peak %.1f MiB at event %lld of %lld\n", peak / M, (long long) tpeak, (long long) tmax);
    // tensors alive at the peak, grouped
    struct grp * g = calloc((size_t) s->n, sizeof(struct grp)); int ng = 0;
    for (int i = 0; i < s->n; i++) {
        const struct blk * b = &s->b[i];
        if (!(b->t0 <= tpeak && (b->t1 == INT64_MAX || b->t1 > tpeak))) continue;
        char bn[96]; base_name(b->name, bn, sizeof(bn));
        int k = 0;
        for (; k < ng; k++) if (!strcmp(g[k].name, bn) && !strcmp(g[k].op, b->op)) break;
        if (k == ng) { strcpy(g[k].name, bn); strcpy(g[k].op, b->op); ng++; }
        g[k].bytes += b->size; g[k].n++; if (b->size > g[k].biggest) g[k].biggest = b->size;
    }
    qsort(g, (size_t) ng, sizeof(struct grp), cmp_grp);
    printf("   alive at the peak (%d groups):\n", ng);
    for (int k = 0; k < ng && k < top; k++)
        printf("     %8.1f MiB  x%-3d (largest %7.1f)  %-12s %s\n", g[k].bytes / M, g[k].n, g[k].biggest / M, g[k].op, g[k].name);
    free(g);
    // the largest blocks
    struct blk ** p = malloc((size_t) s->n * sizeof(*p));
    for (int i = 0; i < s->n; i++) p[i] = &s->b[i];
    qsort(p, (size_t) s->n, sizeof(*p), cmp_sz);
    printf("   largest blocks:\n");
    for (int i = 0; i < s->n && i < 8; i++)
        printf("     %8.1f MiB  [%lld, %lld)  %-12s %-6s %s\n", p[i]->size / M, (long long) p[i]->t0,
               p[i]->t1 == INT64_MAX ? -1LL : (long long) p[i]->t1, p[i]->op, p[i]->type, p[i]->name);
    free(p);
    // the offline placement at each cap
    struct ggml_alloc_plan_block * pb = malloc((size_t) s->n * sizeof(*pb));
    for (int c = 0; c < ncaps; c++) {
        for (int i = 0; i < s->n; i++) { pb[i].size = s->b[i].size; pb[i].t0 = s->b[i].t0; pb[i].t1 = s->b[i].t1; pb[i].chunk = 0; pb[i].offset = 0; }
        size_t cs[64] = {0};
        const size_t cap = caps[c] > 0 ? (size_t) (caps[c] * M) : (SIZE_MAX / 4);
        const int nc = ggml_alloc_plan(pb, s->n, cap, cs, 64);
        size_t tot = 0;
        if (caps[c] > 0) printf("   cap %6.0f MiB: ", caps[c]); else printf("   cap    inf MiB: ");
        for (int k = 0; k < nc; k++) { tot += cs[k]; printf("%s%.0f", k ? " + " : "", cs[k] / M); }
        printf(" = %.0f MiB in %d chunks (%.0f MiB over the live peak)\n", tot / M, nc, (tot - (double) peak) / M);
    }
    free(pb);
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: ubplan DUMP [CAP_MIB...] (0 = no cap)\n"); return 1; }
    FILE * f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    double caps[32]; int ncaps = 0;
    for (int i = 2; i < argc && ncaps < 32; i++) caps[ncaps++] = atof(argv[i]);
    if (ncaps == 0) { const double d[] = {512, 1024, 1536, 2048, 0}; for (int i = 0; i < 5; i++) caps[ncaps++] = d[i]; }
    const int top = getenv("UBPLAN_TOP") ? atoi(getenv("UBPLAN_TOP")) : 25;
    const int all = getenv("UBPLAN_ALL") && atoi(getenv("UBPLAN_ALL"));
    struct sec secs[256]; int ns = 0;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "allocator ", 10)) {
            struct sec * s = &secs[ns++];
            memset(s, 0, sizeof(*s));
            if (sscanf(line, "allocator %95s cap %zu walk %zu live_max %zu blocks %d", s->alloc, &s->cap, &s->walk, &s->live_max, &s->n) != 5) {
                fprintf(stderr, "bad header: %s", line); return 1;
            }
            s->b = calloc((size_t) s->n, sizeof(struct blk));
            for (int i = 0; i < s->n; i++) {
                if (!fgets(line, sizeof(line), f)) { fprintf(stderr, "short section\n"); return 1; }
                struct blk * b = &s->b[i];
                long long t0, t1;
                if (sscanf(line, "%zu %lld %lld %47s %15s %95[^\n]", &b->size, &t0, &t1, b->op, b->type, b->name) < 5) {
                    fprintf(stderr, "bad block: %s", line); return 1;
                }
                b->t0 = t0; b->t1 = t1;
            }
            if (ns == 256) break;
        }
    }
    fclose(f);
    int best = -1;
    for (int i = 0; i < ns; i++) {
        printf("section %d: %s, %d blocks, live max %.0f MiB, walk %.0f MiB\n", i, secs[i].alloc, secs[i].n,
               secs[i].live_max / 1048576.0, secs[i].walk / 1048576.0);
        if (strstr(secs[i].alloc, "Host") || strstr(secs[i].alloc, "CPU")) continue;
        if (best < 0 || secs[i].live_max > secs[best].live_max) best = i;
    }
    for (int i = 0; i < ns; i++)
        if (all || i == best) analyse(&secs[i], caps, ncaps, top);
    return 0;
}
