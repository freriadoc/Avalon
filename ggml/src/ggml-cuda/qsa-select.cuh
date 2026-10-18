#include "common.cuh"

void ggml_cuda_op_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// QSA prefill (ggml_qsa_select DENSE behind the selection the graph assembles): GET_ROWS of the pools' cells at the
// top-k pools, CONCAT of the tail cells, GET_ROWS of the scores at the top-k pools, QSA_SELECT; computed in one kernel
// from the top-k, the pool table, the tail and the scores. Returns the nodes covered after i (0: no match).
int ggml_cuda_try_fuse_qsa_select_pooled(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// QSA generation (few queries): the indexer's TOP_K, GET_ROWS of the pools' cells, CONCAT of the tail cells, GET_ROWS
// of the scores at the top-k, QSA_SELECT ROWS (and QSA_SELECT MASK if it follows), in one kernel (ggml_cuda_top_k_qsa_rows). Returns the nodes covered
// after i (0: no match).
int ggml_cuda_try_fuse_qsa_select_rows(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
