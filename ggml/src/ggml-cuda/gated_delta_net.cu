#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

constexpr int gdn_cols_per_warp = 4;

// the deferred commit of ggml_gated_delta_net_set_replay
struct gdn_replay_args {
    float *     stash;   // [row, 2*w] (null: no replay)
    const int * ctl;     // { n_prefix, read_half, defer }
    int64_t     row;     // floats per token row: k [S_v*n_k_heads], v [S_v*H], g [H*(KDA ? S_v : 1)], beta [H]
    int64_t     w;       // token rows per half
    int64_t     n_k_heads;
};

template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp = gdn_cols_per_warp, bool replay = false>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_ids,
                                     int64_t       state_row_stride,
                                     const gdn_replay_args rp) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    // with state_ids, curr_state is the cache itself and the sequence's row is gathered here
    const int64_t state_in_offset      = (state_ids != nullptr ? (int64_t) state_ids[sequence] * state_row_stride
                                                               : (int64_t) sequence * H * S_v * S_v) + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * lanes_per_col + lane_in_col;
        s_shard[r]  = curr_state[i];
    }

    // deferred commit: the kept tokens of the previous window first (the same arithmetic as in the window, without the
    // output), then the state to commit; the window's inputs are kept in the other half
    bool          defer  = false;
    float *       st_wr  = nullptr;
    const int64_t off_v  = (int64_t) S_v * rp.n_k_heads;
    const int64_t off_g  = off_v + (int64_t) S_v * H;
    const int64_t off_b  = off_g + (KDA ? (int64_t) S_v : 1) * H;
    if constexpr (replay) {
        const int     n_prefix  = rp.ctl[0];
        const int     read_half = rp.ctl[1];
        defer = rp.ctl[2] != 0;
        const float * st_rd = rp.stash + (int64_t) read_half * rp.w * rp.row;
        st_wr = rp.stash + (int64_t) (1 - read_half) * rp.w * rp.row;
        for (int t = 0; t < n_prefix; t++) {
            const float * row    = st_rd + t * rp.row;
            const float * k_t    = row + iq1 * S_v;
            const float * g_t    = row + off_g + h_idx * (KDA ? S_v : 1);
            const float beta_val = row[off_b + h_idx];
            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * lanes_per_col + lane_in_col];
            }
            float kv_shard = 0.0f;
            float g_val    = 0.0f;
            if constexpr (!KDA) {
                g_val = expf(*g_t);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    kv_shard += s_shard[r] * k_reg[r];
                }
            } else {
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
                }
            }
            const float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);
            const float v_col  = row[off_v + h_idx * S_v + col];
            if constexpr (!KDA) {
                const float delta_col = (v_col - g_val * kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                }
            } else {
                const float delta_col = (v_col - kv_col) * beta_val;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    s_shard[r] = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                }
            }
        }
        if (defer) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * lanes_per_col + lane_in_col;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * lanes_per_col + lane_in_col;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (replay) {
            if (defer) {
                // one writer per value: k by the first column group of the first CTA of each key head, v by the
                // column's lane 0, g and beta by lane 0 of the first CTA
                float * row = st_wr + t * rp.row;
                const bool first = blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0;
                if (first && (int64_t) h_idx < rp.n_k_heads) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        row[h_idx * S_v + r * lanes_per_col + lane_in_col] = k_reg[r];
                    }
                }
                if (lane_in_col == 0) {
                    row[off_v + h_idx * S_v + col] = v_t[col];
                }
                if constexpr (KDA) {
                    if (first) {
#pragma unroll
                        for (int r = 0; r < rows_per_lane; r++) {
                            const int i = r * lanes_per_col + lane_in_col;
                            row[off_g + h_idx * S_v + i] = g_t[i];
                        }
                    }
                } else if (first && lane == 0) {
                    row[off_g + h_idx] = *g_t;
                }
                if (first && lane == 0) {
                    row[off_b + h_idx] = beta_val;
                }
            }
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * lanes_per_col + lane_in_col;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
        if (!defer) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i          = r * lanes_per_col + lane_in_col;
                state[col * S_v + i] = s_shard[r];
            }
        }
    }
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
        const gdn_replay_args & rp, cudaStream_t stream) {
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

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<16, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<16, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride, rp);
            break;
        case 32:
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<32, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<32, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride, rp);
            break;
        case 64: {
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<64, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<64, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride, rp);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(cols_per_warp == 1 ? gated_delta_net_cuda<128, KDA, keep_rs_t, 1, replay> : gated_delta_net_cuda<128, KDA, keep_rs_t, gdn_cols_per_warp, replay>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_ids, state_row_stride, rp);
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
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_ids, state_row_stride, rp, stream);
        }
    }
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
