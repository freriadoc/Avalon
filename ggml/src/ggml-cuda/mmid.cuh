#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// the same result as ggml_cuda_launch_mm_ids_helper with write_inverse, in one pass over the ids (distinct within a token)
size_t ggml_cuda_mm_ids_sorted_scratch(int n_experts, int n_tokens);
void ggml_cuda_mm_ids_sorted(const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds, int32_t * scratch,
        int n_experts, int n_tokens, int n_expert_used, int si1, cudaStream_t stream);
