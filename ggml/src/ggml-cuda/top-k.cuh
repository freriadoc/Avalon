#include "common.cuh"

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// bytes of scratch the top-k of dst takes instead of pool memory, 0 if none (only the radix path has any)
size_t ggml_cuda_top_k_scratch_size(const ggml_tensor * dst);

// the generation's QSA selection after an indexer's top-k (ggml_cuda_try_fuse_qsa_select_rows): slot j < n_top*kpool is
// cell j % kpool of the pool top_k[j / kpool] (pool_idxs), the slots after are the tail cells; sel (the CONCAT), top_score
// (the GET_ROWS of the scores at the top-k) and rows (QSA_SELECT ROWS: the cell, 0 if none) written
// and, if mask is set, the mask (QSA_SELECT MASK from the positions: 0 at a live slot whose cell the query sees, -inf
// elsewhere)
struct top_k_qsa_tail {
    const int32_t * pool_idxs; const int32_t * tail;
    int32_t * sel; float * top_score; int32_t * rows;
    int64_t s_pool, s_tail, s_sel, s_rows, n_tail, n_kv;
    int32_t kpool;
    half * mask; const int32_t * cell_pos; const int32_t * tok_pos; int64_t s_mask;
};

// the top-k dst and its selection computed by one kernel; false if it cannot (dst untouched)
bool ggml_cuda_top_k_qsa_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const top_k_qsa_tail & tail);
