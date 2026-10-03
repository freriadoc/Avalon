#include "qsa-select.cuh"

// see ggml_qsa_select: a thread per (slot, query)
template <bool mask>
static __global__ void k_qsa_select(const int32_t * sel, const float * top_score, const half * kq_mask, void * dst,
        const int64_t n_sel, const int64_t n_top, const int64_t n_kv, const int32_t kpool,
        const int64_t s_sel, const int64_t s_score, const int64_t s_mask) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    if (j >= n_sel) {
        return;
    }
    const int32_t c    = sel[t*s_sel + j];
    const bool    cell = c >= 0 && c < n_kv;
    if constexpr (mask) {
        const bool live = cell && (j >= n_top*kpool || top_score[t*s_score + j/kpool] != -INFINITY);
        ((half *) dst)[t*n_sel + j] = live ? kq_mask[t*s_mask + c] : __float2half(-INFINITY);
    } else {
        ((int32_t *) dst)[t*n_sel + j] = cell ? c : 0;
    }
}

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sel       = dst->src[0];
    const ggml_tensor * top_score = dst->src[1];
    const ggml_tensor * kq_mask   = dst->src[2];
    const int32_t kpool = ggml_get_op_params_i32(dst, 0);
    const bool    mask  = ggml_get_op_params_i32(dst, 1) != 0;
    const int64_t n_sel = sel->ne[0];
    const int64_t n_t   = sel->ne[1];
    const int threads = 256;
    const dim3 grid((n_sel + threads - 1) / threads, n_t, 1);
    const int64_t s_sel   = sel->nb[1] / sizeof(int32_t);
    const int64_t s_score = top_score->nb[1] / sizeof(float);
    const int64_t s_mask  = kq_mask->nb[1] / sizeof(half);
    if (mask) {
        k_qsa_select<true><<<grid, threads, 0, ctx.stream()>>>((const int32_t *) sel->data, (const float *) top_score->data,
            (const half *) kq_mask->data, dst->data, n_sel, top_score->ne[0], kq_mask->ne[0], kpool, s_sel, s_score, s_mask);
    } else {
        k_qsa_select<false><<<grid, threads, 0, ctx.stream()>>>((const int32_t *) sel->data, (const float *) top_score->data,
            (const half *) kq_mask->data, dst->data, n_sel, top_score->ne[0], kq_mask->ne[0], kpool, s_sel, s_score, s_mask);
    }
    CUDA_CHECK(cudaGetLastError());
}
