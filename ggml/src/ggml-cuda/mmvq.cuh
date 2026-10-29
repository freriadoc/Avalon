#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

// n (2 to GGML_CUDA_MMVQ_MULTI_MAX) products of the activation src1 by matrices of one type and width,
// in one launch; the rows of every matrix but the last must be a multiple of 8 (a block never spans two)
void ggml_cuda_mul_mat_vec_q_multi(ggml_backend_cuda_context & ctx, const ggml_tensor * const * src0, ggml_tensor * const * dst,
        int n, const ggml_tensor * src1);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
// MoE cache: one batched matvec over cached experts given by their entry addresses on the device (64 bits each, 0 for a
// row the cache does not hold: left untouched). Output row c is W[entries[c]] times activation row act_ids[c] (row c
// modulo act_rows without act_ids). act_q8 uses the padded q8_1 layout of quantize_row_q8_1_cuda. groups (optional, from
// 2 tokens: max_group, the most rows an entry can have): the routed rows of each entry in chunks of at most
// MOE_GROUP_COLS (moe_group_list), each chunk computed by one warp that decodes the weights once for its rows.
void ggml_cuda_moe_cache_mmv(
    ggml_type type0, const char * act_q8, const uint64_t * entries_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_hits, int64_t act_rows, cudaStream_t stream,
    const int32_t * groups = nullptr, int max_group = 1);

// up * GLU(gate) for rows whose up and gate experts (two cache entries) are both resident.
void ggml_cuda_moe_cache_mmv_fused(
    ggml_type type0, const char * act_q8, const uint64_t * up_entries_dev,
    const uint64_t * gate_entries_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_hits, int64_t act_rows,
    float up_min, float up_max, float gate_min, float gate_max,
    cudaStream_t stream, void * q8_dst = nullptr, int64_t q8_stride = 0, unsigned * q8_count = nullptr,
    const int32_t * groups = nullptr, int max_group = 1);

// The chunk list of a hosted MoE plan (written by the kernel that copies the plan to the device, read by the grouped
// kernels): [0] the chunk count, [1 + c] chunk c's row count (1 to MOE_GROUP_COLS), [1 + MOE_GROUP_MAX_ROWS +
// MOE_GROUP_COLS * c + k] its rows, increasing. A chunk is up to MOE_GROUP_COLS rows of one entry (the tokens routed to
// one expert), in order of their first row.
#define MOE_GROUP_MAX_ROWS 64
#define MOE_GROUP_COLS     4
#define MOE_GROUP_WORDS    (1 + MOE_GROUP_MAX_ROWS + MOE_GROUP_COLS * MOE_GROUP_MAX_ROWS)
