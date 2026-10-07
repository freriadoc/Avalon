#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#define GGML_CUDA_SSM_CONV_CACHED_MAX_T     16
#define GGML_CUDA_SSM_CONV_CACHED_MAX_SLOTS 4

// rollback slots of a conv state in a recurrent cache: slot k's rows start at dst[k] (one row per
// sequence, nb1[k] bytes apart) and take the window's columns from tail_col[k]
struct ggml_cuda_ssm_conv_slots {
    int     n = 0;
    char *  dst[GGML_CUDA_SSM_CONV_CACHED_MAX_SLOTS]      = {nullptr};
    int64_t nb1[GGML_CUDA_SSM_CONV_CACHED_MAX_SLOTS]      = {0};
    int     tail_col[GGML_CUDA_SSM_CONV_CACHED_MAX_SLOTS] = {0};
};

// ssm_conv (d_conv 4, at most GGML_CUDA_SSM_CONV_CACHED_MAX_T tokens) whose window is the conv state
// (rows of state_row floats, row ids[s], or row s when ids is null) followed by x [n_t, n_ch, n_s]
// (strided), optionally followed by silu, writing the new tails into the cache slots
// qk_out (optional): the l2 norm of the first n_qk output channels, heads of 128 (rms_norm with eps qk_eps, then
// scale qk_scale), [n_qk, n_t, n_s], computed as rms_norm_f32 does (the same bytes)
void ggml_cuda_op_ssm_conv_cached(ggml_backend_cuda_context & ctx, const float * state, const int32_t * ids, int64_t state_row,
        const ggml_tensor * x, const ggml_tensor * w, float * dst, int64_t n_ch, int64_t n_t, int64_t n_s, bool apply_silu,
        const ggml_cuda_ssm_conv_slots & slots,
        float * qk_out = nullptr, int64_t n_qk = 0, float qk_eps = 0.0f, float qk_scale = 1.0f);
