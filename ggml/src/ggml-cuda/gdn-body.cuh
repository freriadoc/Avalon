#pragma once

// the gated delta net's recurrence (gated_delta_net.cu), its body shared with the multi-task kernel (mk.cu): a block
// of warps per head and sequence, each warp a set of state columns

#include "common.cuh"

constexpr int gdn_cols_per_warp = 4;

// the deferred commit of ggml_gated_delta_net_set_replay
struct gdn_replay_args {
    float *     stash;   // [row, 2*w] (null: no replay)
    const int * ctl;     // { n_prefix, read_half, defer }
    int64_t     row;     // floats per token row: k [S_v*n_k_heads], v [S_v*H], g [H*(KDA ? S_v : 1)], beta [H]
    int64_t     w;       // token rows per half
    int64_t     n_k_heads;
};

struct gated_delta_net_args {
    const float * q; const float * k; const float * v; const float * g; const float * beta; const float * curr_state;
    float * dst; float * state;
    int64_t H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3;
    uint3 neqk1_magic, rq3_magic;
    float scale; int64_t state_slot_stride; int K; const int32_t * state_ids; int64_t state_row_stride;
    gdn_replay_args rp;
    int gz, nwarps;   // the grid's z (column blocks) and the warps of a block
};

// head h_idx, sequence, column block bz, warp ty of nwarps, lane
template <int S_v, bool KDA, bool keep_rs_t, int cols_per_warp, bool replay>
static __device__ __forceinline__ void gated_delta_net_body(const gated_delta_net_args & a, const uint32_t h_idx,
        const uint32_t sequence, const uint32_t bz, const int ty, const int nwarps, const int lane_) {
    auto q = a.q;
    auto k = a.k;
    auto v = a.v;
    auto g = a.g;
    auto beta = a.beta;
    auto curr_state = a.curr_state;
    auto dst = a.dst;
    auto state = a.state;
    const auto H = a.H;
    const auto n_tokens = a.n_tokens;
    const auto n_seqs = a.n_seqs;
    const auto sq1 = a.sq1;
    const auto sq2 = a.sq2;
    const auto sq3 = a.sq3;
    const auto sv1 = a.sv1;
    const auto sv2 = a.sv2;
    const auto sv3 = a.sv3;
    const auto sb1 = a.sb1;
    const auto sb2 = a.sb2;
    const auto sb3 = a.sb3;
    const auto neqk1_magic = a.neqk1_magic;
    const auto rq3_magic = a.rq3_magic;
    const auto scale = a.scale;
    const auto state_slot_stride = a.state_slot_stride;
    const auto K = a.K;
    const auto state_ids = a.state_ids;
    const auto state_row_stride = a.state_row_stride;
    const auto rp = a.rp;


    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    // the warp is split into cols_per_warp segments of lanes_per_col lanes; each segment owns
    // one state column and reduces within itself
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = lane_;
    const int col_in_warp = lane / lanes_per_col;              // column slot within the warp
    const int lane_in_col = lane - col_in_warp * lanes_per_col;  // lane within the column's reduction segment
    const int col = (bz * nwarps + ty) * cols_per_warp + col_in_warp;

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
                const bool first = bz == 0 && ty == 0 && col_in_warp == 0;
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
