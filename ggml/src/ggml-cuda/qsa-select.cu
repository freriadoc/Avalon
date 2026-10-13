#include "qsa-select.cuh"

// visibility of cell c for query t: the dense mask, or 0 / -inf from the positions
template <bool by_pos>
static __device__ __forceinline__ half qsa_visibility(const void * vis, const int32_t * tok_pos, const int64_t s_mask,
                                                      const int64_t t, const int32_t c) {
    if constexpr (by_pos) {
        const int32_t pc = ((const int32_t *) vis)[c];
        return pc >= 0 && pc <= tok_pos[t] ? __float2half(0.0f) : __float2half(-INFINITY);
    } else {
        return ((const half *) vis)[t*s_mask + c];
    }
}

// see ggml_qsa_select: a thread per (slot, query)
template <int mode, bool by_pos>
static __global__ void k_qsa_select(const int32_t * sel, const float * top_score, const void * vis, const int32_t * tok_pos,
        void * dst, const int64_t n_sel, const int64_t n_top, const int64_t n_kv, const int32_t kpool,
        const int64_t s_sel, const int64_t s_score, const int64_t s_mask, const int64_t s_dst) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (j >= n_sel) {
        return;
    }
    if constexpr (mode == GGML_QSA_SELECT_CAUSAL) {
        ((half *) dst)[t*s_dst + j] = qsa_visibility<by_pos>(vis, tok_pos, s_mask, t, (int32_t) j);
        return;
    }
    const int32_t c    = sel[t*s_sel + j];
    const bool    cell = c >= 0 && c < n_kv;
    const bool    live = cell && (j >= n_top*kpool || top_score[t*s_score + j/kpool] != -INFINITY);
    if constexpr (mode == GGML_QSA_SELECT_ROWS) {
        ((int32_t *) dst)[t*s_dst + j] = cell ? c : 0;
    } else if constexpr (mode == GGML_QSA_SELECT_MASK) {
        ((half *) dst)[t*s_dst + j] = live ? qsa_visibility<by_pos>(vis, tok_pos, s_mask, t, c) : __float2half(-INFINITY);
    } else {
        // live slots name distinct cells
        if (live) {
            ((half *) dst)[t*s_dst + c] = qsa_visibility<by_pos>(vis, tok_pos, s_mask, t, c);
        }
    }
}

static __global__ void k_qsa_fill_neg_inf(half * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(-INFINITY);
    }
}

template <int mode, bool by_pos>
static void qsa_select_launch(const ggml_tensor * dst, const dim3 grid, const int threads, cudaStream_t stream,
                              const int64_t n_sel, const int64_t n_top) {
    const ggml_tensor * sel       = dst->src[0];
    const ggml_tensor * top_score = dst->src[1];
    const ggml_tensor * vis       = dst->src[2];
    const ggml_tensor * tok_pos   = dst->src[3];
    k_qsa_select<mode, by_pos><<<grid, threads, 0, stream>>>(
        sel ? (const int32_t *) sel->data : nullptr, top_score ? (const float *) top_score->data : nullptr, vis->data,
        tok_pos ? (const int32_t *) tok_pos->data : nullptr, dst->data, n_sel, n_top, vis->ne[0],
        ggml_get_op_params_i32(dst, 0),
        sel ? sel->nb[1] / sizeof(int32_t) : 0, top_score ? top_score->nb[1] / sizeof(float) : 0,
        by_pos ? 0 : vis->nb[1] / sizeof(half), dst->nb[1] / ggml_type_size(dst->type));
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t mode   = ggml_get_op_params_i32(dst, 1);
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
#define QSA_LAUNCH(M) (by_pos ? qsa_select_launch<M, true>(dst, grid, threads, stream, n_sel, n_top) \
                              : qsa_select_launch<M, false>(dst, grid, threads, stream, n_sel, n_top))
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
