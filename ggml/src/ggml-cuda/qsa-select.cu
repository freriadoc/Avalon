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
