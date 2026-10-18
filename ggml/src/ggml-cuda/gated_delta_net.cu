#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include "gdn-body.cuh"
#include "mk.cuh"

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp, bool replay = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const gated_delta_net_args a) {
    gated_delta_net_body<S_v, KDA, keep_rs_t, cols_per_warp, replay>(a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.y,
                                                                      blockDim.y, threadIdx.x);
}

template <bool KDA, bool keep_rs_t, bool replay>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const int32_t * state_ids, int64_t state_row_stride,
        const gdn_replay_args & rp, cudaStream_t stream, ggml_cuda_mk * mk = nullptr) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    // four columns per warp (see the kernel); shrink the CTA when the wider CTA would leave
    // SMs without a CTA, so small head counts keep the device filled. A single token keeps one column per warp:
    // its pass is mostly the read of the state, probably better spread over four times more CTAs (replay, RDNA3:
    // one-token passes -0.4 %; from 3 tokens on, four columns are as fast or faster)
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = n_tokens == 1 ? 1 : gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    // one CTA covers cols_per_warp*num_warps columns (see the kernel)
    dim3      grid_dims(H, n_seqs, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    gated_delta_net_args a;
    a.q = q_d; a.k = k_d; a.v = v_d; a.g = g_d; a.beta = b_d; a.curr_state = s_d; a.dst = dst_d; a.state = state_d;
    a.H = H; a.n_tokens = n_tokens; a.n_seqs = n_seqs; a.sq1 = sq1; a.sq2 = sq2; a.sq3 = sq3; a.sv1 = sv1; a.sv2 = sv2; a.sv3 = sv3;
    a.sb1 = sb1; a.sb2 = sb2; a.sb3 = sb3;
    a.neqk1_magic = init_fastdiv_values(neqk1); a.rq3_magic = init_fastdiv_values(rq3);
    a.scale = scale; a.state_slot_stride = state_slot_stride; a.K = K; a.state_ids = state_ids; a.state_row_stride = state_row_stride;
    a.rp = rp; a.gz = (int) grid_dims.z; a.nwarps = num_warps;
    // the multi-task kernel takes it (mk.cuh): S_v 128, four columns and four warps, no KDA, replay or kept states
    if (mk && S_v == 128 && !KDA && !keep_rs_t && !replay && cols_per_warp == gdn_cols_per_warp && num_warps == 4 &&
            warp_size == 32 && ggml_cuda_mk_add_gdn(mk, a)) {
        return;
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<16, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<16, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params, a);
            break;
        case 32:
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<32, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<32, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params, a);
            break;
        case 64: {
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<64, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<64, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params, a);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<128, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<128, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params, a);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // the state's GET_ROWS left to this kernel (ggml_cuda_gdn_state_gather): read the cache rows
    const int32_t * state_ids        = nullptr;
    int64_t         state_row_stride = 0;
    if (const ggml_tensor * gather = ggml_cuda_gdn_state_gather_of(ctx, src_state)) {
        s_d              = (const float *) gather->src[0]->data;
        state_ids        = (const int32_t *) gather->src[1]->data;
        state_row_stride = (int64_t) (gather->src[0]->nb[1] / sizeof(float));
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state) || state_ids != nullptr);

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    gdn_replay_args rp = { nullptr, nullptr, 0, 0, neqk1 };
    if (const ggml_tensor * stash = dst->src[6]) {
        GGML_ASSERT(!keep_rs && n_seqs == 1);
        rp.stash = (float *) stash->data;
        rp.ctl   = (const int *) dst->src[7]->data;
        rp.row   = stash->ne[0];
        rp.w     = stash->ne[1] / 2;
        // the replayed k rows are read by key head with the q/k head size
        GGML_ASSERT(src_k->ne[0] == S_v);
        if (kda) {
            launch_gated_delta_net<true, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        } else {
            launch_gated_delta_net<false, false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        }
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        } else {
            launch_gated_delta_net<true, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        } else {
            launch_gated_delta_net<false, false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream, ctx.mk);
        }
    }
}

// whether launch_gated_delta_net records this node's kernel in the multi-task kernel (its conditions, mk.cuh)
bool ggml_cuda_gdn_mk_recordable(const ggml_tensor * dst) {
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const int64_t S_v = v->ne[0], H = v->ne[1], n_tokens = v->ne[2], n_seqs = v->ne[3];
    if (S_v != 128 || g->ne[0] == S_v || ggml_get_op_params_i32(dst, 0) > 1 || dst->src[6] != nullptr || n_tokens == 1 ||
            ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size != 32) {
        return false;
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    int num_warps = 4;
    while (num_warps > 1 && H*n_seqs*(S_v / (gdn_cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    return num_warps == 4;
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}

const ggml_tensor * ggml_cuda_gdn_state_gather_of(const ggml_backend_cuda_context & ctx, const ggml_tensor * state) {
    return ggml_cuda_gathered_rows_of(ctx, state);
}
