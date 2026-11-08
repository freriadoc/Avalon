#include "common.cuh"
#include "mmq.cuh"

// each expert's first column tile of width J in the compact MUL_MAT_ID grid (prefix[e]: tiles of the experts before e,
// prefix[n]: all of them), from mm_ids_helper's row bounds; one thread: 512 experts in a few microseconds
static __global__ void k_mmq_tile_prefix(const int32_t * __restrict__ bounds, const int n, const int J, int32_t * __restrict__ prefix) {
    if (threadIdx.x != 0) {
        return;
    }
    int t = 0;
    for (int e = 0; e < n; ++e) {
        prefix[e] = t;
        t += (bounds[e + 1] - bounds[e] + J - 1) / J;
    }
    prefix[n] = t;
}

void ggml_cuda_mmq_tile_prefix(const int32_t * expert_bounds, int n_experts, int J, int32_t * prefix, cudaStream_t stream) {
    k_mmq_tile_prefix<<<1, 32, 0, stream>>>(expert_bounds, n_experts, J, prefix);
}
#include "quantize.cuh"
#include "mmid.cuh"
#include "unary.cuh"
#include "moe-cache.cuh"
#include "ggml-backend-impl.h"

#include <cstdint>
#include <functional>
#include <cstring>

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_q_case<GGML_TYPE_IQ3_K, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_q_case<GGML_TYPE_IQ3_KS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL, GGML_PREC_Q8>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q8>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

// The scratch of a MUL_MAT(_ID) node: the expert ids helper's outputs first (fixed sizes), then the
// quantized activation, whose tail (padding read by the kernels) depends on the weight type.
static constexpr size_t mmq_scratch_align = 256;

// a tile of J columns may read up to J - 1 past the end of the quantized activation and of ids_dst (#29953), in loads
// of nthreads ints; every config has J <= 128 and nthreads <= 512 (static_asserts of mmq.cuh): the bounds of both
static constexpr int    mmq_J_bound       = 128;
static constexpr size_t mmq_src1_tail     = GGML_PAD(mmq_J_bound*sizeof(block_q8_1_mmq), 512*sizeof(int));

struct mmq_scratch_layout {
    size_t ids_src1      = 0;
    size_t ids_dst       = 0;
    size_t expert_bounds = 0;
    size_t src1_q8_1     = 0;
    size_t size          = 0;
};

static mmq_scratch_layout mmq_scratch_get_layout(const ggml_tensor * dst, int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2] : nullptr;

    const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    const bool    fallback    = src0->ne[1] % 128 != 0;
    // MUL_MAT_ID: the tokens are src1->ne[2] (#29941: J_max of the tokens, not of the experts per token)
    const int64_t j_max       = mmq_J_bound;
    const size_t  q8_1_tail   = mmq_src1_tail;

    mmq_scratch_layout l;
    size_t off = 0;
    if (ids) {
        const int64_t ne_get_rows = src1->ne[2] * ids->ne[0];
        l.ids_src1      = off; off += GGML_PAD(ne_get_rows*sizeof(int32_t), mmq_scratch_align);
        l.ids_dst       = off; off += GGML_PAD((ne_get_rows + j_max - 1)*sizeof(int32_t), mmq_scratch_align); // read up to J - 1 past the end (#29953)
        l.expert_bounds = off; off += GGML_PAD((src0->ne[2] + 1)*sizeof(int32_t), mmq_scratch_align);
        l.src1_q8_1     = off; off += ne_get_rows*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + q8_1_tail;
    } else {
        l.src1_q8_1     = off;
        off += src1->ne[3]*src1->ne[2] * src1->ne[1]*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ + q8_1_tail;
    }
    l.size = off;
    return l;
}

size_t ggml_cuda_mmq_scratch_size(const ggml_tensor * dst, int cc, int64_t key[GGML_CUDA_MMQ_SCRATCH_KEY]) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->op == GGML_OP_MUL_MAT_ID ? dst->src[2] : nullptr;

    // the native FP4 path also needs per-row scales: it keeps the pool
    if (ggml_cuda_mmq_get_prec_src1(src0, dst, cc) != GGML_PREC_Q8) {
        return 0;
    }

    // the quantized activation depends on the activation, on the weight type only through the layout of
    // the block scales, and for MUL_MAT_ID on the ids (the activation is scattered to the experts' rows)
    for (int i = 0; i < GGML_CUDA_MMQ_SCRATCH_KEY; i++) {
        key[i] = 0;
    }
    key[0] = (int64_t) (intptr_t) src1;
    key[1] = (int64_t) (intptr_t) ids;
    key[2] = (int64_t) mmq_get_q8_1_ds_layout(src0->type);
    for (int i = 0; i < 4; i++) {
        key[3 + i] = src1->ne[i];
    }
    if (ids) {
        key[7] = ids->ne[0];
        key[8] = ids->ne[1];
        key[9] = src0->ne[2];
    }
    return mmq_scratch_get_layout(dst, cc).size;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int    id    = ggml_cuda_get_device();
    const int    cc    = ggml_cuda_info().devices[id].cc;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding. Not for the prefetch ring's chunk (its table
    // at src0->data, GGML_SCHED_PREFETCH_RING_MB): its experts sit at the table's addresses, each with its pad
    const uint64_t * ring_table = ggml_cuda_moe_prefill_table(src0->data);
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            !(ring_table && (const void *) ring_table == src0->data)) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            // contiguous, or experts at a step above their bytes (a packed layer's copy): nothing past ggml_nbytes either way
            GGML_ASSERT(ggml_is_contiguously_allocated(src0) ||
                (src0->nb[1] == ggml_row_size(src0->type, src0->ne[0]) && src0->nb[2] >= src0->nb[1]*src0->ne[1] &&
                 src0->nb[3] >= src0->nb[2]*src0->ne[2]));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ggml_cuda_mmq_needs_fallback(ne01);

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    // the scratch the scheduler placed in the compute buffer (GGML_SCHED_SCRATCH_SRC), else pool memory;
    // a node sharing the previous node's scratch and key finds its quantized activation (and ids) there
    int64_t key[GGML_CUDA_MMQ_SCRATCH_KEY] = {0};
    const size_t        scratch_need = ggml_cuda_mmq_scratch_size(dst, cc, key);
    const ggml_tensor * scratch_t    = dst->src[GGML_SCHED_SCRATCH_SRC];
    char * scratch = scratch_need > 0 && scratch_t && scratch_t->data && ggml_nbytes(scratch_t) >= scratch_need ?
        (char *) scratch_t->data : nullptr;
    const bool scratch_reuse = scratch && ctx.mmq_scratch_last == scratch_t &&
        memcmp(ctx.mmq_scratch_key, key, sizeof(key)) == 0;
    ctx.mmq_scratch_last = scratch ? scratch_t : nullptr;
    memcpy(ctx.mmq_scratch_key, key, sizeof(key));
    const mmq_scratch_layout scratch_l = scratch ? mmq_scratch_get_layout(dst, cc) : mmq_scratch_layout();

    int J_best        = 0;
    int nthreads_best = 0;
    {
        int64_t ncols_opt = ne11;
        if (ids) {
            const int64_t n_expert_used = ids->ne[0];
            ncols_opt = ne12;

            // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
            // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
            if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
            }
        }

        int ntiles_J_best = INT_MAX;

        for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
            const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J, fallback, cc, prec_src1);
            if (config.type == GGML_TYPE_COUNT) {
                continue;
            }

            if (mmq_get_nbytes_shared(config, cc) > smpbo) {
                continue;
            }

            const int ntiles_x = (ncols_opt + config.J - 1) / config.J;

            if (ntiles_x < ntiles_J_best) {
                J_best = J;
                nthreads_best = config.nthreads;
                ntiles_J_best = ntiles_x;
            }
        }
    }
    GGML_ASSERT(J_best > 0);

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            mmq_src1_tail;
        ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
        char * src1_q8_1 = scratch ? scratch + scratch_l.src1_q8_1 : src1_q8_1_pool.alloc(nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        if (!scratch_reuse) {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        // GGML_CUDA_F16_IMAGE_CHECK=1 (diagnostic, synchronizes: with GGML_CUDA_DISABLE_GRAPHS=1): a quantized src1 found
        // in the scratch (left by the group's previous product or written by the producer, ggml_cuda_f16_image_target)
        // compared with a quantization made now
        static const bool img_check = getenv("GGML_CUDA_F16_IMAGE_CHECK") != nullptr;
        if (img_check && scratch_reuse && !use_native_fp4) {
            const size_t n = nbytes_src1_q8_1 - mmq_src1_tail;
            ggml_cuda_pool_alloc<char> ref(ctx.pool(), nbytes_src1_q8_1);
            quantize_mmq_q8_1_cuda(src1_d, nullptr, ref.get(), src0->type, ne10, src1->nb[1] / ts_src1, src1->nb[2] / ts_src1,
                                   src1->nb[3] / ts_src1, ne10_padded, ne11, ne12, ne13, stream);
            std::vector<uint8_t> a(n), b(n);
            CUDA_CHECK(cudaMemcpyAsync(a.data(), src1_q8_1, n, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(b.data(), ref.get(), n, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int64_t bad = 0;
            for (size_t k = 0; k < n; k++) {
                bad += a[k] != b[k];
            }
            static int64_t n_checked = 0, n_bad = 0;
            n_checked++; n_bad += bad != 0;
            fprintf(stderr, "Q8_IMAGE_CHECK %s (%s): %lld / %lld bytes differ; products %lld, with differences %lld\n", dst->name,
                    src1->name, (long long) bad, (long long) n, (long long) n_checked, (long long) n_bad);
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, J_best};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1_pool(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_dst_pool(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds_pool(ctx.pool());
    int32_t * ids_src1      = scratch ? (int32_t *) (scratch + scratch_l.ids_src1)      : ids_src1_pool.alloc(ne_get_rows);
    int32_t * ids_dst       = scratch ? (int32_t *) (scratch + scratch_l.ids_dst)       : ids_dst_pool.alloc(ne_get_rows + J_best - 1);
    int32_t * expert_bounds = scratch ? (int32_t *) (scratch + scratch_l.expert_bounds) : expert_bounds_pool.alloc(ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    if (!scratch_reuse) {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1, ids_dst, expert_bounds,
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
        if (int * counts = ggml_cuda_moe_prefill_counts(src0->name, (int) ne02)) {
            ggml_cuda_moe_count_rows(expert_bounds, counts, (int) ne02, stream);
        }
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        mmq_src1_tail;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    char * src1_q8_1 = scratch ? scratch + scratch_l.src1_q8_1 : src1_q8_1_pool.alloc(nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    if (!scratch_reuse) {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1, src1_q8_1, src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1, src1_q8_1, src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1, src1_q8_1, src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1, src1_q8_1, src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1, ids_dst, expert_bounds, dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, J_best, ggml_cuda_moe_prefill_table(src0->data)};

    // the grid over the tiles that hold rows (mmq_tile_remap): each expert's first column tile for J_best, from the
    // bounds mm_ids_helper wrote. GGML_CUDA_MMQ_COMPACT=0: every expert times the batch's tiles, as before
    static const bool compact = !getenv("GGML_CUDA_MMQ_COMPACT") || atoi(getenv("GGML_CUDA_MMQ_COMPACT")) != 0;
    ggml_cuda_pool_alloc<int32_t> tile_prefix(ctx.pool());
    mmq_args args_c = args;
    // GGML_CUDA_MMQ_EXPERT_GROUP=G (experiment, the prefill's MoE by groups of experts): the experts in groups of G, one
    // launch each over its experts' rows (the bounds, the addresses and the weights from expert g0 on); the same tiles,
    // the same sums: the result is the ungrouped one bit for bit. The grid of a group is sized for every row (its rows
    // are known on the device only): the tiles past its own exit at once
    static const int expert_group = getenv("GGML_CUDA_MMQ_EXPERT_GROUP") ? atoi(getenv("GGML_CUDA_MMQ_EXPERT_GROUP")) : 0;
    if (compact && expert_group > 0 && expert_group < ne02 && ne02 % expert_group == 0) {
        const int G = expert_group;
        tile_prefix.alloc(G + 1);
        for (int64_t g0 = 0; g0 < ne02; g0 += G) {
            mmq_args a = args;
            a.x             = args.x_experts ? args.x : args.x + g0*src0->nb[2];
            a.x_experts     = args.x_experts ? args.x_experts + g0 : nullptr;
            a.expert_bounds = expert_bounds + g0;
            a.nchannels_x   = G;
            a.nchannels_y   = G;
            ggml_cuda_mmq_tile_prefix(expert_bounds + g0, G, J_best, tile_prefix.get(), stream);
            a.tile_prefix = tile_prefix.get();
            ggml_cuda_mul_mat_q_switch_type(ctx, a, stream, prec_src1);
        }
        return;
    }
    if (compact) {
        tile_prefix.alloc(ne02 + 1);
        ggml_cuda_mmq_tile_prefix(expert_bounds, (int) ne02, J_best, tile_prefix.get(), stream);
        args_c.tile_prefix = tile_prefix.get();
    }

    ggml_cuda_mul_mat_q_switch_type(ctx, args_c, stream, prec_src1);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

// ---------------------------------------------------------------------------------------------------------------------
// GGML_CUDA_MOE_GROUPED=G (experiment, off by default): a prompt's MoE block, gate MUL_MAT_ID, up MUL_MAT_ID, SWIGLU and
// down MUL_MAT_ID, computed by groups of G experts: per group gate, up, their swiglu, the down's activation quantized and
// down, over that group's rows only (the condition of an expert ring smaller than a layer: a group's experts are done
// with once its down is). The rows of every expert are read back to the host once per block (a 2 KB copy and a stream
// synchronization), so that each launch has the grid of its own rows. The same kernels, the same tiles, the same sums,
// the swiglu row by row as unary_gated_op_kernel: the result is the ungrouped one bit for bit.

// out[row] = silu(gate[row]) * up[row] for the compact rows r0 .. r1 - 1 (ids_dst: compact row -> row of the outputs)
static __global__ void k_moe_grouped_swiglu(const float * __restrict__ gate, const float * __restrict__ up, float * __restrict__ out,
        const int32_t * __restrict__ ids_dst, const int r0, const int64_t n, const int64_t sg, const int64_t su, const int64_t so) {
    const int64_t row = ids_dst[r0 + blockIdx.y];
    for (int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; j < n; j += (int64_t) gridDim.x*blockDim.x) {
        const float g = gate[row*sg + j];
        out[row*so + j] = (g / (1.0f + expf(-g))) * up[row*su + j];
    }
}

static int mmq_pick_J(ggml_type type, int64_t ncols_opt, bool fallback, int cc, size_t smpbo, ggml_prec prec_src1) {
    int J_best = 0, ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int ntiles_x = (ncols_opt + config.J - 1) / config.J;
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

// the block's group size: GGML_CUDA_MOE_GROUPED=G, else 32 when the prompt's experts come through the ring path
static int moe_grouped_size() {
    static const int G = getenv("GGML_CUDA_MOE_GROUPED") ? atoi(getenv("GGML_CUDA_MOE_GROUPED")) :
                         getenv("GGML_CUDA_MOE_RING_PREFILL") && atoi(getenv("GGML_CUDA_MOE_RING_PREFILL")) > 0 ? 32 : 0;
    return G;
}

// xe_host: per role (gate, up, down), each expert's address (a packed layer: its stride is no whole number of
// blocks, MMQ then reads the experts by address, as the prefill's in-place reads)
struct moe_grouped_hooks {
    int G = 0;                                     // the group size (0: moe_grouped_size())
    std::function<void(int64_t)> * before = nullptr, * after = nullptr;   // around each group (empty ones too)
    std::function<void()> * synced = nullptr;      // once the stream is synchronized (the rows read back)
    std::vector<int32_t> * bounds = nullptr;       // the rows' bounds by expert, read back (n_expert + 1)
};

static int moe_block_grouped_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
        const ggml_tensor * wg_dev, const ggml_tensor * wu_dev, const ggml_tensor * wd_dev, const std::vector<uint64_t> * xe_host,
        const moe_grouped_hooks * hooks = nullptr);

int ggml_cuda_moe_block_grouped(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    return moe_block_grouped_impl(ctx, cgraph, i, nullptr, nullptr, nullptr, nullptr);
}

int ggml_cuda_moe_block_grouped_xe(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int G,
        const std::vector<uint64_t> * xe, std::function<void(int64_t)> * before, std::function<void(int64_t)> * after,
        std::function<void()> * synced, std::vector<int32_t> * bounds) {
    const ggml_tensor * gate = cgraph->nodes[i], * up = cgraph->nodes[i + 1], * down = cgraph->nodes[i + 3];
    moe_grouped_hooks h;
    h.G = G; h.before = before; h.after = after; h.synced = synced; h.bounds = bounds;
    return moe_block_grouped_impl(ctx, cgraph, i, gate->src[0], up->src[0], down->src[0], xe, &h);
}

// GGML_CUDA_MOE_RING_PREFILL (step 1, the expert ring to come): a prompt's MoE block with its experts in host memory
// (the scheduler hosts it: no split, no prefetch), its layer copied whole to the device (one copy when its three
// tensors are one packed range, else three), then computed by groups. The copy is pool memory: run with a larger
// GGML_CUDA_MOE_CACHE_RESERVE_MB (a layer is ~1.1 GiB) until the ring takes its place
int ggml_cuda_moe_block_grouped_host(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (moe_grouped_size() <= 0 || i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * gate = cgraph->nodes[i], * up = cgraph->nodes[i + 1], * down = cgraph->nodes[i + 3];
    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID || down->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    const ggml_tensor * w[3] = { gate->src[0], up->src[0], down->src[0] };
    const char * lo = nullptr, * hi = nullptr;
    size_t sum = 0;
    for (const ggml_tensor * t : w) {
        if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) || t->view_src) {
            return 0;
        }
        const char * a = (const char *) t->data, * b = a + ggml_nbytes(t);
        lo = lo ? std::min(lo, a) : a;
        hi = hi ? std::max(hi, b) : b;
        sum += ggml_nbytes(t);
    }
    cudaStream_t stream = ctx.stream();
    const bool one_range = (size_t) (hi - lo) <= sum + sum / 16;   // a packed layer: its tensors interleaved per expert
    // MMQ reads a little past an expert's last row (up to 512 values): a zeroed tail after the copy
    constexpr size_t tail = 64*1024;
    const size_t span = one_range ? (size_t) (hi - lo) : sum + 2*tail;
    ggml_cuda_pool_alloc<char> buf(ctx.pool(), span + tail);
    CUDA_CHECK(cudaMemsetAsync(buf.get() + span, 0, tail, stream));
    ggml_tensor dw[3];
    size_t off = 0;
    for (int k = 0; k < 3; k++) {
        dw[k] = *w[k];   // its host buffer kept (weights: MMQ clears no padding past it); its data the device copy's
        if (one_range) {
            dw[k].data = buf.get() + ((const char *) w[k]->data - lo);
        } else {
            dw[k].data = buf.get() + off;
            CUDA_CHECK(cudaMemcpyAsync(dw[k].data, w[k]->data, ggml_nbytes(w[k]), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemsetAsync((char *) dw[k].data + ggml_nbytes(w[k]), 0, tail, stream));
            off += ggml_nbytes(w[k]) + tail;
        }
    }
    if (one_range) {
        CUDA_CHECK(cudaMemcpyAsync(buf.get(), lo, (size_t) (hi - lo), cudaMemcpyHostToDevice, stream));
    }
    std::vector<uint64_t> xe[3];
    for (int k = 0; k < 3; k++) {
        xe[k].resize(w[k]->ne[2]);
        for (int64_t e = 0; e < w[k]->ne[2]; e++) {
            xe[k][e] = (uint64_t) (uintptr_t) ((const char *) dw[k].data + e*w[k]->nb[2]);
        }
    }
    const int covered = moe_block_grouped_impl(ctx, cgraph, i, &dw[0], &dw[1], &dw[2], xe);
    GGML_ASSERT(covered > 0 && "a hosted prompt MoE block the grouped path does not take");
    return covered;
}

static int moe_block_grouped_impl(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
        const ggml_tensor * wg_dev, const ggml_tensor * wu_dev, const ggml_tensor * wd_dev, const std::vector<uint64_t> * xe_host,
        const moe_grouped_hooks * hooks) {
    const int G = hooks && hooks->G > 0 ? hooks->G : moe_grouped_size();
    if (G <= 0 || i + 3 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * gate = cgraph->nodes[i], * up = cgraph->nodes[i + 1], * glu = cgraph->nodes[i + 2], * down = cgraph->nodes[i + 3];
    if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID || glu->op != GGML_OP_GLU ||
            down->op != GGML_OP_MUL_MAT_ID || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
            glu->src[0] != gate || glu->src[1] != up || down->src[1] != glu) {
        return 0;
    }
    const ggml_tensor * x = gate->src[1], * ids = gate->src[2];
    const bool host_w = wg_dev != nullptr;   // the weights copied to the device (ggml_cuda_moe_block_grouped_host)
    const ggml_tensor * wg = host_w ? wg_dev : gate->src[0], * wu = host_w ? wu_dev : up->src[0], * wd = host_w ? wd_dev : down->src[0];
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t n_expert = wg->ne[2], n_used = ids->ne[0], n_tok = ids->ne[1];
    // one sequence of a prompt, the activation broadcast to the experts (ne11 == 1), every tensor on the device, MMQ
    // types, contiguous outputs, the experts dividing into groups, no prefill table (weights read in place)
    if (up->src[1] != x || up->src[2] != ids || down->src[2] != ids || x->ne[1] != 1 || x->type != GGML_TYPE_F32 ||
            n_expert % G != 0 || G >= n_expert || n_tok < 128 || wg->ne[3] != 1 ||
            !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || !ggml_is_contiguous(glu) || !ggml_is_contiguous(down) ||
            !ggml_is_contiguous(x) || wg->type != wu->type || wg->ne[1] != wu->ne[1] ||
            ggml_cuda_moe_prefill_table(wg->data) || ggml_cuda_moe_prefill_table(wu->data) || ggml_cuda_moe_prefill_table(wd->data)) {
        return 0;
    }
    for (const ggml_tensor * w : { wg, wu, wd }) {
        if ((!host_w && (!w->buffer || ggml_backend_buffer_is_host(w->buffer))) ||
                !ggml_cuda_should_use_mmq(w->type, cc, n_tok, n_expert) || w->type == GGML_TYPE_NVFP4 || w->type == GGML_TYPE_MXFP4 ||
                ggml_cuda_mmq_get_prec_src1(w, w == wd ? down : gate, cc) != GGML_PREC_Q8) {
            return 0;
        }
    }
    // by groups, gate, up and the swiglu stay live until the last group's down: the allocator, which sees them dead
    // once the swiglu (down) has read them, may have placed a later output over them. The swiglu over its inputs is
    // fine (row by row); anything else over them: the ungrouped way
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
        return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
    };
    if (overlap(down, gate) || overlap(down, up) || overlap(down, glu) || overlap(down, x) || overlap(gate, up) ||
            overlap(glu, x) || overlap(gate, x) || overlap(up, x)) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s: outputs over the block's live tensors, computed ungrouped\n", __func__, down->name);
        }
        return 0;
    }
    cudaStream_t stream = ctx.stream();
    {
        // the bounds are read back to the host: not inside a graph capture (the block is then computed ungrouped)
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
            (void) cudaGetLastError();
            return 0;
        }
    }
    const size_t smpbo = ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo;
    const int64_t rows = n_tok * n_used;
    const int64_t n_embd = x->ne[0], n_ff = gate->ne[0];
    const int64_t ne10p_gu = GGML_PAD(n_embd, MATRIX_ROW_PADDING), ne10p_d = GGML_PAD(n_ff, MATRIX_ROW_PADDING);
    const bool fb_gu = ggml_cuda_mmq_needs_fallback(wg->ne[1]), fb_d = ggml_cuda_mmq_needs_fallback(wd->ne[1]);
    int64_t ncols_opt = n_tok;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (rows + n_expert - 1) / n_expert;
    }
    const int J_gu = mmq_pick_J(wg->type, ncols_opt, fb_gu, cc, smpbo, GGML_PREC_Q8);
    const int J_d  = mmq_pick_J(wd->type, ncols_opt, fb_d,  cc, smpbo, GGML_PREC_Q8);
    GGML_ASSERT(J_gu > 0 && J_d > 0);

    // the scratch the scheduler placed in the compute buffer for gate (its ids and its quantized activation, shared with
    // up) and for down (its quantized activation), as the ungrouped products use it; else pool memory
    const mmq_scratch_layout lg = mmq_scratch_get_layout(gate, cc), ld = mmq_scratch_get_layout(down, cc);
    const ggml_tensor * sg_t = gate->src[GGML_SCHED_SCRATCH_SRC], * sd_t = down->src[GGML_SCHED_SCRATCH_SRC];
    char * sg = sg_t && sg_t->data && ggml_nbytes(sg_t) >= lg.size ? (char *) sg_t->data : nullptr;
    char * sd = sd_t && sd_t->data && ggml_nbytes(sd_t) >= ld.size ? (char *) sd_t->data : nullptr;
    if (sg && sd && sg == sd) {
        sd = nullptr;   // one scratch for both: down's activation apart
    }
    // by groups the scratch must outlive every output of the block, which the allocator may have placed over it (it
    // sees gate's scratch dead after up, down's before down): then pool memory
    auto over = [&](const char * p, size_t n) {
        for (const ggml_tensor * t : { (const ggml_tensor *) gate, (const ggml_tensor *) up, (const ggml_tensor *) glu,
                                       (const ggml_tensor *) down, x, ids }) {
            const char * t0 = (const char *) t->data;
            if (t0 < p + n && p < t0 + ggml_nbytes(t)) {
                return true;
            }
        }
        return false;
    };
    if (sg && over(sg, lg.size)) {
        sg = nullptr;
    }
    if (sd && over(sd, ld.size)) {
        sd = nullptr;
    }
    ctx.mmq_scratch_last = nullptr;   // what the scratch holds is no longer what a next product would expect
    const size_t q8_gu_bytes = rows*ne10p_gu*sizeof(block_q8_1_mmq)/QK8_1_MMQ + mmq_src1_tail;
    const size_t q8_d_bytes  = rows*ne10p_d *sizeof(block_q8_1_mmq)/QK8_1_MMQ + mmq_src1_tail;
    ggml_cuda_pool_alloc<int32_t> ids_src1_p(ctx.pool()), ids_dst_p(ctx.pool()), bounds_p(ctx.pool());
    ggml_cuda_pool_alloc<char> q8_gu_p(ctx.pool()), q8_d_p(ctx.pool());
    struct ptr_i32 { int32_t * p; int32_t * get() const { return p; } };
    struct ptr_c { char * p; char * get() const { return p; } };
    const ptr_i32 ids_src1 { sg ? (int32_t *) (sg + lg.ids_src1)      : ids_src1_p.alloc(rows) };
    const ptr_i32 ids_dst  { sg ? (int32_t *) (sg + lg.ids_dst)       : ids_dst_p.alloc(rows + std::max(J_gu, J_d) - 1) };
    const ptr_i32 bounds   { sg ? (int32_t *) (sg + lg.expert_bounds) : bounds_p.alloc(n_expert + 1) };
    const ptr_c   q8_gu    { sg ? sg + lg.src1_q8_1 : q8_gu_p.alloc(q8_gu_bytes) };
    const ptr_c   q8_d     { sd ? sd + ld.src1_q8_1 : q8_d_p.alloc(q8_d_bytes) };
    // the pool frees in reverse order of its allocations (VMM pool: asserted): allocated after the ones above, which
    // their declarations precede
    ggml_cuda_pool_alloc<int32_t> prefix(ctx.pool(), G + 1);

    // the rows by expert (the inverse map for the broadcast activation), the activation quantized once for gate and up
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(), n_expert, n_tok,
        n_used, 1, ids->nb[1] / ggml_element_size(ids), x->nb[2] / x->nb[1], /*write_inverse =*/ true, stream);
    quantize_scatter_mmq_q8_1_cuda((const float *) x->data, ids_src1.get(), q8_gu.get(), wg->type, n_embd,
        x->nb[2] / sizeof(float), ne10p_gu, n_tok, rows, (int) n_used, stream);
    ggml_cuda_pool_alloc<uint64_t> xe_dev(ctx.pool(), xe_host ? 3*n_expert : 1);
    if (xe_host) {
        for (int k = 0; k < 3; k++) {
            CUDA_CHECK(cudaMemcpyAsync(xe_dev.get() + k*n_expert, xe_host[k].data(), n_expert*sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        }
    }
    std::vector<int32_t> hb(n_expert + 1);
    CUDA_CHECK(cudaMemcpyAsync(hb.data(), bounds.get(), (n_expert + 1)*sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    // GGML_CUDA_MOE_GROUPED_TIMING=1: the host's wait in this synchronization, and the stream's idle time after it (the
    // time to queue the first group), every 48 blocks
    static const bool timing = getenv("GGML_CUDA_MOE_GROUPED_TIMING") && atoi(getenv("GGML_CUDA_MOE_GROUPED_TIMING")) != 0;
    const int64_t t_sync0 = timing ? ggml_time_us() : 0;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    static int64_t t_wait = 0, n_blocks = 0, t_gap = 0;
    const int64_t t_sync1 = timing ? ggml_time_us() : 0;

    if (hooks && hooks->bounds) {
        *hooks->bounds = hb;
    }
    if (hooks && hooks->synced) {
        (*hooks->synced)();
    }

    auto launch = [&](const ggml_tensor * w, ggml_tensor * dst, const char * q8, int64_t ne10p, int64_t ne11, int J, bool fallback,
                      int64_t g0, int r0, int r1, int role) {
        const size_t ts = ggml_type_size(w->type);
        mmq_args a = {
            (const char *) w->data + g0*w->nb[2], w->type, (const int *) q8, ids_dst.get(), bounds.get() + g0, (float *) dst->data,
            nullptr,
            w->ne[0], w->ne[1], rows, (int64_t) (w->nb[1]/ts), rows, (int64_t) (dst->nb[1]/sizeof(float)),
            G, G, (int64_t) (w->nb[2]/ts), (int64_t) (ne11*ne10p*sizeof(block_q8_1)/(QK8_1*sizeof(int))), (int64_t) (dst->nb[2]/sizeof(float)),
            1, 1, (int64_t) (w->nb[3]/ts), 0, (int64_t) (dst->nb[3]/sizeof(float)),
            n_tok, J};
        if (xe_host) {
            a.x_experts = xe_dev.get() + role*n_expert + g0;
        }
        // the grid of the group's own rows (the compact grid bound for them)
        a.ncols_dst = r1 - r0;
        ggml_cuda_mmq_tile_prefix(bounds.get() + g0, G, J, prefix.get(), stream);
        a.tile_prefix = prefix.get();
        GGML_UNUSED(fallback);
        ggml_cuda_mul_mat_q_switch_type(ctx, a, stream, GGML_PREC_Q8);
    };
    bool first_launch = true;
    for (int64_t g0 = 0; g0 < n_expert; g0 += G) {
        const int r0 = hb[g0], r1 = hb[g0 + G];
        if (timing && first_launch && r1 > r0) {
            // the stream idles from the synchronization to the first group's launch
            t_gap += ggml_time_us() - t_sync1;
            first_launch = false;
        }
        if (hooks && hooks->before) {
            (*hooks->before)(g0 / G);
        }
        if (r1 <= r0) {
            if (hooks && hooks->after) {
                (*hooks->after)(g0 / G);
            }
            continue;
        }
        launch(wg, gate, q8_gu.get(), ne10p_gu, 1, J_gu, fb_gu, g0, r0, r1, 0);
        launch(wu, up,   q8_gu.get(), ne10p_gu, 1, J_gu, fb_gu, g0, r0, r1, 1);
        {
            const dim3 grid((unsigned) std::min<int64_t>((n_ff + 255)/256, 8), (unsigned) (r1 - r0));
            k_moe_grouped_swiglu<<<grid, 256, 0, stream>>>((const float *) gate->data, (const float *) up->data, (float *) glu->data,
                ids_dst.get(), r0, n_ff, gate->nb[1]/sizeof(float), up->nb[1]/sizeof(float), glu->nb[1]/sizeof(float));
        }
        quantize_mmq_q8_1_rows_cuda((const float *) glu->data, ids_dst.get(), q8_d.get(), wd->type, n_ff,
            glu->nb[1]/sizeof(float), ne10p_d, r0, r1 - r0, (int) rows, stream);
        launch(wd, down, q8_d.get(), ne10p_d, n_used, J_d, fb_d, g0, r0, r1, 2);
        if (hooks && hooks->after) {
            (*hooks->after)(g0 / G);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    if (timing) {
        t_wait += t_sync1 - t_sync0;
        if (++n_blocks % 48 == 0) {
            GGML_LOG_INFO("%s: %lld blocks: host wait in the synchronizations %.1f ms, stream idle after them %.2f ms (%.1f us a block)\n",
                    __func__, (long long) n_blocks, t_wait / 1e3, t_gap / 1e3, (double) t_gap / n_blocks);
        }
    }

    // GGML_CUDA_MOE_GROUPED_CHECK=1: the block again the ungrouped way into copies of its outputs, compared bit for bit
    static const bool check = getenv("GGML_CUDA_MOE_GROUPED_CHECK") && atoi(getenv("GGML_CUDA_MOE_GROUPED_CHECK")) != 0;
    if (check) {
        ggml_tensor tg = *gate, tu = *up, th = *glu, td = *down;
        ggml_cuda_pool_alloc<char> bg(ctx.pool(), ggml_nbytes(gate)), bu(ctx.pool(), ggml_nbytes(up)),
            bh(ctx.pool(), ggml_nbytes(glu)), bd(ctx.pool(), ggml_nbytes(down));
        tg.data = bg.get(); tu.data = bu.get(); th.data = bh.get(); td.data = bd.get();
        tg.src[GGML_SCHED_SCRATCH_SRC] = tu.src[GGML_SCHED_SCRATCH_SRC] = td.src[GGML_SCHED_SCRATCH_SRC] = nullptr;
        th.src[0] = &tg; th.src[1] = &tu; td.src[1] = &th;
        auto table = [&](const ggml_tensor * w, int role) {
            ggml_cuda_moe_prefill_table_set(xe_host ? w->data : nullptr, xe_host ? xe_dev.get() + role*n_expert : nullptr);
        };
        table(wg, 0); ggml_cuda_mul_mat_q(ctx, wg, x, ids, &tg);
        table(wu, 1); ggml_cuda_mul_mat_q(ctx, wu, x, ids, &tu);
        ggml_cuda_op_swiglu(ctx, &th);
        table(wd, 2); ggml_cuda_mul_mat_q(ctx, wd, &th, ids, &td);
        ggml_cuda_moe_prefill_table_set(nullptr, nullptr);
        std::vector<char> ga(ggml_nbytes(gate)), gb(ggml_nbytes(gate)), ua(ggml_nbytes(up)), ub(ggml_nbytes(up));
        CUDA_CHECK(cudaMemcpyAsync(ga.data(), gate->data, ga.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(gb.data(), tg.data,    gb.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ua.data(), up->data,   ua.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(ub.data(), tu.data,    ub.size(), cudaMemcpyDeviceToHost, stream));
        std::vector<char> a(ggml_nbytes(down)), b(ggml_nbytes(down)), c(ggml_nbytes(glu)), d(ggml_nbytes(glu));
        CUDA_CHECK(cudaMemcpyAsync(a.data(), down->data, a.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), td.data,    b.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(c.data(), glu->data,  c.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(d.data(), th.data,    d.size(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        static int n_checked = 0, n_diff = 0;
        n_checked++;
        const bool same_d = memcmp(a.data(), b.data(), a.size()) == 0, same_h = memcmp(c.data(), d.data(), c.size()) == 0;
        if (!same_d || !same_h) {
            n_diff++;
        }
        if (!same_d || !same_h || n_checked <= 2 || n_checked % 48 == 0) {
            const bool same_g = memcmp(ga.data(), gb.data(), ga.size()) == 0, same_u = memcmp(ua.data(), ub.data(), ua.size()) == 0;
            size_t ndiff = 0; double maxd = 0.0;
            for (size_t k = 0; k < ga.size()/4; k++) {
                const float p0 = ((const float *) ga.data())[k], p1 = ((const float *) gb.data())[k];
                if (p0 != p1) { ndiff++; maxd = std::max(maxd, (double) fabsf(p0 - p1)); }
            }
            GGML_LOG_INFO("%s: %s: %d blocks checked, %d differ (gate %s: %zu values, max %.3g; up %s, swiglu %s, down %s)\n", __func__,
                    down->name, n_checked, n_diff, same_g ? "identical" : "DIFFERS", ndiff, maxd, same_u ? "identical" : "DIFFERS",
                    same_h ? "identical" : "DIFFERS", same_d ? "identical" : "DIFFERS");
        }
    }
    return 3;
}
