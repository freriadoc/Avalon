#include "qsa-select.cuh"
#include "qsa-select-body.cuh"
#include "mk.cuh"
#include "top-k.cuh"
#include "ggml-backend-impl.h"

// see ggml_qsa_select: a thread per (slot, query) (the body: qsa-select-body.cuh)
template <int mode, bool by_pos>
static __global__ void k_qsa_select(const qsa_select_args a) {
    k_qsa_select_body<mode, by_pos>(a, (int64_t) blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y);
}

static __global__ void k_qsa_fill_neg_inf(half * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(-INFINITY);
    }
}

template <int mode, bool by_pos>
static void qsa_select_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, const dim3 grid, const int threads,
                              cudaStream_t stream, const int64_t n_sel, const int64_t n_top) {
    const ggml_tensor * sel       = dst->src[0];
    const ggml_tensor * top_score = dst->src[1];
    const ggml_tensor * vis       = dst->src[2];
    const ggml_tensor * tok_pos   = dst->src[3];
    qsa_select_args a;
    a.sel = sel ? (const int32_t *) sel->data : nullptr; a.top_score = top_score ? (const float *) top_score->data : nullptr;
    a.vis = vis->data; a.tok_pos = tok_pos ? (const int32_t *) tok_pos->data : nullptr; a.dst = dst->data;
    a.n_sel = n_sel; a.n_top = n_top; a.n_kv = vis->ne[0]; a.kpool = ggml_get_op_params_i32(dst, 0);
    a.s_sel = sel ? sel->nb[1] / sizeof(int32_t) : 0; a.s_score = top_score ? top_score->nb[1] / sizeof(float) : 0;
    a.s_mask = by_pos ? 0 : vis->nb[1] / sizeof(half); a.s_dst = dst->nb[1] / ggml_type_size(dst->type);
    a.mode = mode; a.by_pos = by_pos; a.n_t = (int) grid.y;
    // the multi-task kernel takes it (mk.cuh); not the dense mode, whose fill must come first
    if (mode != GGML_QSA_SELECT_DENSE && ctx.mk && ggml_cuda_mk_add_qsa_select(ctx.mk, a)) {
        return;
    }
    k_qsa_select<mode, by_pos><<<grid, threads, 0, stream>>>(a);
}

// see ggml_qsa_pool_layout: a thread per cell scatters it to its member slot (or block, for the reps); the same thread
// index also writes the padded slots, which no cell reaches
template <int mode>
static __global__ void k_qsa_pool_layout(const int32_t * __restrict__ cell_pos, const int32_t * __restrict__ meta,
        int32_t * __restrict__ out, const int64_t n_kv, const int64_t n_pool, const int32_t kpool) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t pos_min = meta[0], n_real = meta[1];
    if constexpr (mode == GGML_QSA_SELECT_POOL_END) {
        if (i < n_pool) {
            out[i] = i < n_real ? pos_min + (int32_t) i*kpool + kpool - 1 : -1;
        }
        return;
    }
    constexpr bool members = mode == GGML_QSA_SELECT_POOL_MEMBERS;
    const int64_t n_out  = members ? n_pool*kpool : n_pool;
    const int64_t n_live = members ? (int64_t) n_real*kpool : n_real;
    if (i >= n_live && i < n_out) {
        out[i] = members ? meta[3] : meta[2];
    }
    if (i < n_kv) {
        const int32_t p = cell_pos[i];
        const int64_t d = (int64_t) p - pos_min;
        if (p >= 0 && d >= 0 && d < (int64_t) n_real*kpool) {
            if (members) {
                out[d] = (int32_t) i;
            } else if (d % kpool == kpool - 1) {
                out[d / kpool] = (int32_t) i;
            }
        }
    }
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t mode   = ggml_get_op_params_i32(dst, 1);
    if (mode >= GGML_QSA_SELECT_POOL_MEMBERS) {
        const int32_t kpool  = ggml_get_op_params_i32(dst, 0);
        const int64_t n_kv   = dst->src[2]->ne[0];
        const int64_t n_pool = mode == GGML_QSA_SELECT_POOL_MEMBERS ? dst->ne[1] : dst->ne[0];
        const int64_t n      = std::max(n_kv, mode == GGML_QSA_SELECT_POOL_MEMBERS ? n_pool*kpool : n_pool);
        const int threads = 256;
        const int blocks  = (int) ((n + threads - 1) / threads);
        const int32_t * cp = (const int32_t *) dst->src[2]->data;
        const int32_t * mt = (const int32_t *) dst->src[3]->data;
        int32_t * out = (int32_t *) dst->data;
        cudaStream_t stream = ctx.stream();
        switch (mode) {
            case GGML_QSA_SELECT_POOL_MEMBERS: k_qsa_pool_layout<GGML_QSA_SELECT_POOL_MEMBERS><<<blocks, threads, 0, stream>>>(cp, mt, out, n_kv, n_pool, kpool); break;
            case GGML_QSA_SELECT_POOL_REP:     k_qsa_pool_layout<GGML_QSA_SELECT_POOL_REP>    <<<blocks, threads, 0, stream>>>(cp, mt, out, n_kv, n_pool, kpool); break;
            default:                           k_qsa_pool_layout<GGML_QSA_SELECT_POOL_END>    <<<blocks, threads, 0, stream>>>(cp, mt, out, n_kv, n_pool, kpool); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool    by_pos = dst->src[2]->type == GGML_TYPE_I32;
    const int64_t n_t    = dst->ne[1];
    const int64_t n_sel  = mode == GGML_QSA_SELECT_CAUSAL ? dst->ne[0] : dst->src[0]->ne[0];
    const int64_t n_top  = mode == GGML_QSA_SELECT_CAUSAL ? 0 : dst->src[1]->ne[0];
    const int threads = 256;
    const dim3 grid((n_sel + threads - 1) / threads, n_t, 1);
    cudaStream_t stream = ctx.stream();
    if (mode == GGML_QSA_SELECT_DENSE) {
        const int64_t n = ggml_nelements(dst);
        k_qsa_fill_neg_inf<<<(n + threads - 1) / threads, threads, 0, stream>>>((half *) dst->data, n);
    }
#define QSA_LAUNCH(M) (by_pos ? qsa_select_launch<M, true>(ctx, dst, grid, threads, stream, n_sel, n_top) \
                              : qsa_select_launch<M, false>(ctx, dst, grid, threads, stream, n_sel, n_top))
    switch (mode) {
        case GGML_QSA_SELECT_ROWS:   QSA_LAUNCH(GGML_QSA_SELECT_ROWS);   break;
        case GGML_QSA_SELECT_MASK:   QSA_LAUNCH(GGML_QSA_SELECT_MASK);   break;
        case GGML_QSA_SELECT_DENSE:  QSA_LAUNCH(GGML_QSA_SELECT_DENSE);  break;
        default:                     QSA_LAUNCH(GGML_QSA_SELECT_CAUSAL); break;
    }
#undef QSA_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// the dense mask of ggml_qsa_select from the selection's parts: slot j < n_top*kpool is cell r = j % kpool of the pool
// top_k[p = j / kpool] (pool_idxs [kpool, n_pool]), live unless the pool's score is -inf; the slots after are the tail
template <bool by_pos>
static __global__ void k_qsa_select_dense_pooled(const int32_t * top_k, const int32_t * pool_idxs, const int32_t * tail,
        const float * score, const void * vis, const int32_t * tok_pos, half * dst,
        const int64_t n_top, const int64_t n_tail, const int64_t n_kv, const int32_t kpool,
        const int64_t s_top, const int64_t s_pool, const int64_t s_tail, const int64_t s_score, const int64_t s_mask,
        const int64_t s_dst) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t n_pooled = n_top*kpool;
    if (j >= n_pooled + n_tail) {
        return;
    }
    int32_t c;
    bool    live;
    if (j < n_pooled) {
        const int32_t pool = top_k[t*s_top + j / kpool];
        c    = pool_idxs[pool*s_pool + j % kpool];
        live = score[t*s_score + pool] != -INFINITY;
    } else {
        c    = tail[t*s_tail + (j - n_pooled)];
        live = true;
    }
    if (live && c >= 0 && c < n_kv) {   // live slots name distinct cells
        dst[t*s_dst + c] = qsa_visibility<by_pos>(vis, tok_pos, s_mask, t, c);
    }
}

static bool qsa_pooled_is_view(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW || t->op == GGML_OP_NONE;
}

// t, through reshapes of contiguous tensors
static const ggml_tensor * qsa_pooled_root(const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        t = t->src[0];
    }
    return t;
}

// the next node after k that is not a view, -1 if none
static int qsa_pooled_next(const ggml_cgraph * cgraph, int k) {
    for (k++; k < cgraph->n_nodes; k++) {
        if (!qsa_pooled_is_view(cgraph->nodes[k])) {
            return k;
        }
    }
    return -1;
}

int ggml_cuda_try_fuse_qsa_select_pooled(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_QSA_POOLED") != nullptr;
    const ggml_tensor * cells = cgraph->nodes[i];
    if (disabled || cells->op != GGML_OP_GET_ROWS || cells->type != GGML_TYPE_I32) {
        return 0;
    }
    const int ic = qsa_pooled_next(cgraph, i);
    const int ig = ic < 0 ? -1 : qsa_pooled_next(cgraph, ic);
    const int iq = ig < 0 ? -1 : qsa_pooled_next(cgraph, ig);
    if (iq < 0) {
        return 0;
    }
    const ggml_tensor * cat = cgraph->nodes[ic];
    const ggml_tensor * gs  = cgraph->nodes[ig];
    ggml_tensor *       qs  = cgraph->nodes[iq];
    if (cat->op != GGML_OP_CONCAT || gs->op != GGML_OP_GET_ROWS || qs->op != GGML_OP_QSA_SELECT ||
            ggml_get_op_params_i32(qs, 1) != GGML_QSA_SELECT_DENSE || ggml_get_op_params_i32(cat, 0) != 0) {
        return 0;
    }
    const ggml_tensor * pool_idxs = cells->src[0];
    const ggml_tensor * top_k     = qsa_pooled_root(cells->src[1]);
    const ggml_tensor * tail      = cat->src[1];
    const ggml_tensor * score     = qsa_pooled_root(gs->src[0]);
    const int32_t       kpool     = ggml_get_op_params_i32(qs, 0);
    // the selection is the concat of the pools' cells and the tail, the scores are taken at the same top-k
    if (top_k->op != GGML_OP_TOP_K || top_k->type != GGML_TYPE_I32 || !ggml_is_contiguous(top_k) ||
            qsa_pooled_root(cat->src[0]) != cells || gs->src[1] != top_k ||
            qs->src[0] != cat || qsa_pooled_root(qs->src[1]) != gs ||
            pool_idxs->type != GGML_TYPE_I32 || pool_idxs->ne[0] != kpool || pool_idxs->nb[0] != sizeof(int32_t) ||
            tail->type != GGML_TYPE_I32 || tail->nb[0] != sizeof(int32_t) || score->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(score) || !ggml_is_contiguous(cat) || score->ne[1] != top_k->ne[1] ||
            tail->ne[1] != top_k->ne[1] || cat->ne[0] != kpool*top_k->ne[0] + tail->ne[0] || cat->ne[1] != top_k->ne[1] ||
            qs->src[1]->ne[0] != top_k->ne[0]) {
        return 0;
    }
    // the nodes from i to iq, the views of a node outside the group (the scores' reshape) left out
    int     nodes[16];
    ggml_op ops[16];
    int     n_nodes = 0;
    if (iq - i + 1 > 16) {
        return 0;
    }
    for (int k = i; k <= iq; k++) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (qsa_pooled_is_view(t)) {
            bool inside = false;
            for (int m = 0; m < n_nodes; m++) {
                inside = inside || cgraph->nodes[nodes[m]] == t->view_src || cgraph->nodes[nodes[m]] == t->src[0];
            }
            if (!inside) {
                continue;
            }
        }
        nodes[n_nodes] = k;
        ops[n_nodes++] = t->op;
    }
    int out = iq;
    if (!ggml_can_fuse_subgraph_ext(cgraph, nodes, n_nodes, ops, &out, 1)) {
        return 0;
    }
    const ggml_tensor * vis     = qs->src[2];
    const ggml_tensor * tok_pos = qs->src[3];
    // ggml-alloc may place the mask over a tensor whose last reader in the graph ran before it (the scores and the
    // top-k are last read by the scores' GET_ROWS): the fused kernel reads them while it writes the mask
    {
        const char * ml = (const char *) qs->data, * mh = ml + ggml_nbytes(qs);
        for (const ggml_tensor * t : { top_k, pool_idxs, tail, score, vis, tok_pos }) {
            if (t) {
                const char * tl = (const char *) t->data, * th = tl + ggml_nbytes(t);
                if (tl < mh && ml < th) {
                    return 0;
                }
            }
        }
    }
    const bool by_pos  = vis->type == GGML_TYPE_I32;
    const int64_t n_t   = qs->ne[1];
    const int64_t n_top = top_k->ne[0];
    const int64_t n_sel = cat->ne[0];
    const int threads = 256;
    cudaStream_t stream = ctx.stream();
    const int64_t n = ggml_nelements(qs);
    k_qsa_fill_neg_inf<<<(n + threads - 1) / threads, threads, 0, stream>>>((half *) qs->data, n);
    const dim3 grid((n_sel + threads - 1) / threads, n_t, 1);
    const int64_t s_mask = by_pos ? 0 : vis->nb[1] / sizeof(half);
    const int64_t s_dst  = qs->nb[1] / ggml_type_size(qs->type);
    if (by_pos) {
        k_qsa_select_dense_pooled<true><<<grid, threads, 0, stream>>>((const int32_t *) top_k->data,
            (const int32_t *) pool_idxs->data, (const int32_t *) tail->data, (const float *) score->data, vis->data,
            (const int32_t *) tok_pos->data, (half *) qs->data, n_top, tail->ne[0], vis->ne[0], kpool,
            top_k->nb[1] / sizeof(int32_t), pool_idxs->nb[1] / sizeof(int32_t), tail->nb[1] / sizeof(int32_t),
            score->nb[1] / sizeof(float), s_mask, s_dst);
    } else {
        k_qsa_select_dense_pooled<false><<<grid, threads, 0, stream>>>((const int32_t *) top_k->data,
            (const int32_t *) pool_idxs->data, (const int32_t *) tail->data, (const float *) score->data, vis->data,
            nullptr, (half *) qs->data, n_top, tail->ne[0], vis->ne[0], kpool,
            top_k->nb[1] / sizeof(int32_t), pool_idxs->nb[1] / sizeof(int32_t), tail->nb[1] / sizeof(int32_t),
            score->nb[1] / sizeof(float), s_mask, s_dst);
    }
    CUDA_CHECK(cudaGetLastError());
    return iq - i;
}

int ggml_cuda_try_fuse_qsa_select_rows(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    ggml_tensor * top_k = cgraph->nodes[i];
    if (top_k->op != GGML_OP_TOP_K || top_k->type != GGML_TYPE_I32 || !ggml_is_contiguous(top_k)) {
        return 0;
    }
    const int ir = qsa_pooled_next(cgraph, i);
    const int ic = ir < 0 ? -1 : qsa_pooled_next(cgraph, ir);
    const int ig = ic < 0 ? -1 : qsa_pooled_next(cgraph, ic);
    const int iq = ig < 0 ? -1 : qsa_pooled_next(cgraph, ig);
    if (iq < 0) {
        return 0;
    }
    const ggml_tensor * cells = cgraph->nodes[ir];
    const ggml_tensor * cat   = cgraph->nodes[ic];
    const ggml_tensor * gs    = cgraph->nodes[ig];
    const ggml_tensor * qs    = cgraph->nodes[iq];
    if (cells->op != GGML_OP_GET_ROWS || cells->type != GGML_TYPE_I32 || cat->op != GGML_OP_CONCAT ||
            gs->op != GGML_OP_GET_ROWS || qs->op != GGML_OP_QSA_SELECT ||
            ggml_get_op_params_i32(qs, 1) != GGML_QSA_SELECT_ROWS || ggml_get_op_params_i32(cat, 0) != 0) {
        return 0;
    }
    const ggml_tensor * pool_idxs = cells->src[0];
    const ggml_tensor * tail      = cat->src[1];
    const ggml_tensor * score     = qsa_pooled_root(gs->src[0]);
    const int32_t       kpool     = ggml_get_op_params_i32(qs, 0);
    // as ggml_cuda_try_fuse_qsa_select_pooled; the selection and the scores at the top-k are read again (the mask's
    // QSA_SELECT), the rows by the gathers
    if (qsa_pooled_root(cells->src[1]) != top_k || score != top_k->src[0] || gs->src[1] != top_k ||
            qsa_pooled_root(cat->src[0]) != cells || qs->src[0] != cat || qsa_pooled_root(qs->src[1]) != gs ||
            pool_idxs->type != GGML_TYPE_I32 || pool_idxs->ne[0] != kpool || pool_idxs->nb[0] != sizeof(int32_t) ||
            tail->type != GGML_TYPE_I32 || tail->nb[0] != sizeof(int32_t) || score->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(score) || !ggml_is_contiguous(cat) || !ggml_is_contiguous(gs) || !ggml_is_contiguous(qs) ||
            score->ne[1] != top_k->ne[1] || tail->ne[1] != top_k->ne[1] || cat->ne[0] != kpool*top_k->ne[0] + tail->ne[0] ||
            cat->ne[1] != top_k->ne[1] || qs->ne[0] != cat->ne[0] || qs->ne[1] != top_k->ne[1] ||
            ggml_nelements(gs) != ggml_nelements(top_k) || qs->type != GGML_TYPE_I32) {
        return 0;
    }
    // the mask's QSA_SELECT right after (qwen4exp builds it there), from the positions
    const int im = qsa_pooled_next(cgraph, iq);
    const ggml_tensor * qm = im < 0 ? nullptr : cgraph->nodes[im];
    if (qm && !(qm->op == GGML_OP_QSA_SELECT && ggml_get_op_params_i32(qm, 1) == GGML_QSA_SELECT_MASK &&
                ggml_get_op_params_i32(qm, 0) == kpool && qm->src[0] == cat && qsa_pooled_root(qm->src[1]) == gs &&
                qm->src[2] && qm->src[2]->type == GGML_TYPE_I32 && qm->src[2]->ne[0] == qs->src[2]->ne[0] &&
                qm->src[3] && qm->src[3]->type == GGML_TYPE_I32 && qm->type == GGML_TYPE_F16 && ggml_is_contiguous(qm) &&
                qm->ne[0] == cat->ne[0] && qm->ne[1] == cat->ne[1])) {
        qm = nullptr;
    }
    const int last = qm ? im : iq;
    int     nodes[16];
    ggml_op ops[16];
    int     n_nodes = 0;
    if (last - i + 1 > 16) {
        return 0;
    }
    for (int k = i; k <= last; k++) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (qsa_pooled_is_view(t)) {
            bool inside = false;
            for (int m = 0; m < n_nodes; m++) {
                inside = inside || cgraph->nodes[nodes[m]] == t->view_src || cgraph->nodes[nodes[m]] == t->src[0];
            }
            if (!inside) {
                continue;
            }
        }
        nodes[n_nodes] = k;
        ops[n_nodes++] = t->op;
    }
    // everything is written but the pools' cells (and their reshape), which only the CONCAT may read
    int outs[16];
    int n_outs = 0;
    for (int m = 0; m < n_nodes; m++) {
        const ggml_tensor * t = cgraph->nodes[nodes[m]];
        if (t != cells && t->src[0] != cells) {
            outs[n_outs++] = nodes[m];
        }
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, nodes, n_nodes, ops, outs, n_outs)) {
        return 0;
    }
    // the selection's results are written after everything was read (a grid barrier between): they may overlap the
    // inputs. The top-k and its scratch are written before: against what is read
    {
        const ggml_tensor * scratch = top_k->src[GGML_SCHED_SCRATCH_SRC];
        const ggml_tensor * all[] = { top_k, score, pool_idxs, tail, scratch, top_k->src[GGML_LIVE_COUNT_SRC],
                                      qm ? qm->src[2] : nullptr, qm ? qm->src[3] : nullptr };
        const ggml_tensor * written[] = { top_k, scratch };
        for (const ggml_tensor * w : written) {
            if (!w || !w->data) {
                continue;
            }
            const char * wl = (const char *) w->data, * wh = wl + ggml_nbytes(w);
            for (const ggml_tensor * t : all) {
                if (t && t != w && t->data) {
                    const char * tl = (const char *) t->data, * th = tl + ggml_nbytes(t);
                    if (tl < wh && wl < th) {
                        return 0;
                    }
                }
            }
        }
    }
    top_k_qsa_tail qt;
    qt.pool_idxs = (const int32_t *) pool_idxs->data; qt.tail = (const int32_t *) tail->data;
    qt.sel = (int32_t *) cat->data; qt.top_score = (float *) gs->data; qt.rows = (int32_t *) qs->data;
    qt.s_pool = pool_idxs->nb[1] / sizeof(int32_t); qt.s_tail = tail->nb[1] / sizeof(int32_t);
    qt.s_sel = cat->nb[1] / sizeof(int32_t); qt.s_rows = qs->nb[1] / sizeof(int32_t);
    qt.n_tail = tail->ne[0]; qt.n_kv = qs->src[2]->ne[0]; qt.kpool = kpool;
    qt.mask = qm ? (half *) qm->data : nullptr;
    qt.cell_pos = qm ? (const int32_t *) qm->src[2]->data : nullptr;
    qt.tok_pos  = qm ? (const int32_t *) qm->src[3]->data : nullptr;
    qt.s_mask   = qm ? qm->nb[1] / sizeof(half) : 0;
    if (!ggml_cuda_top_k_qsa_rows(ctx, top_k, qt)) {
        return 0;
    }
    return last - i;
}
