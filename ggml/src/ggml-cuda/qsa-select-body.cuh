#pragma once

// QSA_SELECT's slot kernel (qsa-select.cu), its body shared with the multi-task kernel (mk.cu): a thread per (slot,
// query)

#include "common.cuh"

struct qsa_select_args {
    const int32_t * sel; const float * top_score; const void * vis; const int32_t * tok_pos; void * dst;
    int64_t n_sel, n_top, n_kv; int32_t kpool;
    int64_t s_sel, s_score, s_mask, s_dst;
    int mode; bool by_pos;   // the kernel's template arguments
    int n_t;                 // queries (the grid's y)
};

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

template <int mode, bool by_pos>
static __device__ __forceinline__ void k_qsa_select_body(const qsa_select_args & a, const int64_t j, const int64_t t) {
    const int32_t * sel = a.sel; const float * top_score = a.top_score; const void * vis = a.vis;
    const int32_t * tok_pos = a.tok_pos; void * dst = a.dst;
    const int64_t n_sel = a.n_sel, n_top = a.n_top, n_kv = a.n_kv;
    const int32_t kpool = a.kpool;
    const int64_t s_sel = a.s_sel, s_score = a.s_score, s_mask = a.s_mask, s_dst = a.s_dst;
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
