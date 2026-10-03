#include "qsa-select.cuh"

// see ggml_qsa_select: a thread per (slot, query)
template <int mode>
static __global__ void k_qsa_select(const int32_t * sel, const float * top_score, const half * kq_mask, void * dst,
        const int64_t n_sel, const int64_t n_top, const int64_t n_kv, const int32_t kpool,
        const int64_t s_sel, const int64_t s_score, const int64_t s_mask, const int64_t s_dst) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (j >= n_sel) {
        return;
    }
    const int32_t c    = sel[t*s_sel + j];
    const bool    cell = c >= 0 && c < n_kv;
    const bool    live = cell && (j >= n_top*kpool || top_score[t*s_score + j/kpool] != -INFINITY);
    if constexpr (mode == GGML_QSA_SELECT_ROWS) {
        ((int32_t *) dst)[t*s_dst + j] = cell ? c : 0;
    } else if constexpr (mode == GGML_QSA_SELECT_MASK) {
        ((half *) dst)[t*s_dst + j] = live ? kq_mask[t*s_mask + c] : __float2half(-INFINITY);
    } else {
        // live slots name distinct cells
        if (live) {
            ((half *) dst)[t*s_dst + c] = kq_mask[t*s_mask + c];
        }
    }
}

static __global__ void k_qsa_fill_neg_inf(half * dst, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(-INFINITY);
    }
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel       = dst->src[0];
    const ggml_tensor * top_score = dst->src[1];
    const ggml_tensor * kq_mask   = dst->src[2];
    const int32_t kpool = ggml_get_op_params_i32(dst, 0);
    const int32_t mode  = ggml_get_op_params_i32(dst, 1);
    const int64_t n_sel = sel->ne[0];
    const int64_t n_t   = sel->ne[1];
    const int threads = 256;
    const dim3 grid((n_sel + threads - 1) / threads, n_t, 1);
    const int64_t s_sel   = sel->nb[1] / sizeof(int32_t);
    const int64_t s_score = top_score->nb[1] / sizeof(float);
    const int64_t s_mask  = kq_mask->nb[1] / sizeof(half);
    const int64_t s_dst   = dst->nb[1] / ggml_type_size(dst->type);
    cudaStream_t stream = ctx.stream();
    const int32_t * sd = (const int32_t *) sel->data;
    const float   * td = (const float *) top_score->data;
    const half    * md = (const half *) kq_mask->data;
    switch (mode) {
        case GGML_QSA_SELECT_ROWS:
            k_qsa_select<GGML_QSA_SELECT_ROWS><<<grid, threads, 0, stream>>>(sd, td, md, dst->data, n_sel, top_score->ne[0],
                kq_mask->ne[0], kpool, s_sel, s_score, s_mask, s_dst);
            break;
        case GGML_QSA_SELECT_MASK:
            k_qsa_select<GGML_QSA_SELECT_MASK><<<grid, threads, 0, stream>>>(sd, td, md, dst->data, n_sel, top_score->ne[0],
                kq_mask->ne[0], kpool, s_sel, s_score, s_mask, s_dst);
            break;
        default: {
            const int64_t n = ggml_nelements(dst);
            k_qsa_fill_neg_inf<<<(n + threads - 1) / threads, threads, 0, stream>>>((half *) dst->data, n);
            k_qsa_select<GGML_QSA_SELECT_DENSE><<<grid, threads, 0, stream>>>(sd, td, md, dst->data, n_sel, top_score->ne[0],
                kq_mask->ne[0], kpool, s_sel, s_score, s_mask, s_dst);
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
}
