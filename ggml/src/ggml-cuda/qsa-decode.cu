// ggml_qsa_decode: the QSA attention of a few queries in one op, every size the cache's and the live counts read on the
// device (meta), so that a graph captured once serves every context length (as Basalt's step state). Five launches:
//   1. the indexer's score of every live pool visible to the query, -inf for the others (up to max(live, n_top) columns,
//      what the top-k reads);
//   2. the top n_top pools (the radix top-k; with no more live pools than n_top, written without its passes);
//   3. the visible cells of the chosen pools and of the tail, in order, first: their count, the attention's mask (0 for
//      them, -inf after) and its KV length (the count rounded up to FATTN_KQ_STRIDE);
//   4. those cells' K and V rows to F16 (zeros up to the KV length, nothing past it);
//   5. the flash attention over them, stopped at that KV length.
#include "qsa-decode.cuh"
#include "fattn.cuh"
#include "fattn-common.cuh"
#include "mk.cuh"
#include "top-k.cuh"
#include "getrows-body.cuh"

#include <cstdint>

static __device__ __forceinline__ float qsa_dec_load(const char * row, const int type, const int i) {
    switch (type) {
        case GGML_TYPE_F32: return ((const float *) row)[i];
        case GGML_TYPE_F16: return __half2float(((const half *) row)[i]);
        case GGML_TYPE_Q8_0: {
            const block_q8_0 * b = (const block_q8_0 *) row + i / QK8_0;
            return __half2float(b->d) * (float) b->qs[i % QK8_0];
        }
        default: {   // GGML_TYPE_Q4_0
            const block_q4_0 * b = (const block_q4_0 *) row + i / QK4_0;
            const int j = i % QK4_0;
            const int q = j < QK4_0/2 ? (b->qs[j] & 0xF) : (b->qs[j - QK4_0/2] >> 4);
            return __half2float(b->d) * (float) (q - 8);
        }
    }
}

struct qsa_dec_args {
    const float * idx_q; const char * idx_k; const int32_t * members; const int32_t * tail; const int32_t * cell_pos;
    const int32_t * tok_pos; const int32_t * meta; const char * k; const char * v;
    float idx_scale;
    int idx_dim, n_idx_h, kpool, n_pool, n_top, n_tail, n_cells, n_t, n_kvh, D, n_sel;
    int idx_type, k_type, v_type;
    size_t s_idx_q1, s_idx_q2, s_idx_k, s_tail, k_nb1, k_nb2, v_nb1, v_nb2;
    float * score; int32_t * top; int32_t * rows; int32_t * cnt; int32_t * kv_max; half * mask; half * k16; half * v16;
};

// 1. a warp per pool (its key read once, the dimensions over the lanes), the query's indexer heads in shared memory. With no
// more live pools than n_top every visible one is chosen: 0 for them, no product
static constexpr int QSA_DEC_SCORE_WARPS = 8;

static __global__ void k_qsa_dec_score(const qsa_dec_args a) {
    extern __shared__ float s_q[];   // [n_idx_h * idx_dim]
    const int t = blockIdx.y;
    const int n_real = min(a.n_pool, max(a.meta[1], 0));
    const int p0 = blockIdx.x*QSA_DEC_SCORE_WARPS;
    if (p0 >= max(n_real, a.n_top) || p0 >= a.n_pool) {
        return;   // never read by the top-k
    }
    const bool all = n_real <= a.n_top;
    if (!all) {
        for (int i = threadIdx.x; i < a.n_idx_h*a.idx_dim; i += blockDim.x) {
            const int h = i / a.idx_dim, d = i % a.idx_dim;
            s_q[i] = *(const float *) ((const char *) a.idx_q + h*a.s_idx_q1 + t*a.s_idx_q2 + d*sizeof(float));
        }
        __syncthreads();
    }
    const int p = p0 + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (p >= max(n_real, a.n_top) || p >= a.n_pool) {
        return;
    }
    float * dst = a.score + (size_t) t*a.n_pool;
    if (p >= n_real || a.meta[0] + (p + 1)*a.kpool - 1 > a.tok_pos[t]) {
        if (lane == 0) {
            dst[p] = -INFINITY;
        }
        return;
    }
    if (all) {
        if (lane == 0) {
            dst[p] = 0.0f;
        }
        return;
    }
    const char * kr = a.idx_k + (size_t) a.members[(size_t) p*a.kpool + a.kpool - 1]*a.s_idx_k;
    float kv[8];   // idx_dim <= 8*WARP_SIZE
    const int nd = (a.idx_dim + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = lane + j*WARP_SIZE;
        kv[j] = j < nd && d < a.idx_dim ? qsa_dec_load(kr, a.idx_type, d) : 0.0f;
    }
    float sc = 0.0f;
    for (int h = 0; h < a.n_idx_h; ++h) {
        float d = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int dd = lane + j*WARP_SIZE;
            if (j < nd && dd < a.idx_dim) {
                d += s_q[h*a.idx_dim + dd]*kv[j];
            }
        }
        d = warp_reduce_sum(d);
        sc += a.idx_scale*fmaxf(d, 0.0f);
    }
    if (lane == 0) {
        dst[p] = sc;
    }
}

// 1, a q8_0 indexer key (idx_dim = 128): 8 lanes per pool (16 values each, half a block), 4 pools per warp, every query
// of the batch in the block (each key read once): a pool's product per head reduced over its 8 lanes
static constexpr int QSA_DEC_SQ_POOLS = 32;   // per block of 256 threads
static constexpr int QSA_DEC_SQ_MAX_T = 8;

static __global__ void k_qsa_dec_score_q8(const qsa_dec_args a) {
    extern __shared__ float s_q[];   // [n_t][n_idx_h][128]
    const int n_real = min(a.n_pool, max(a.meta[1], 0));
    const int p0 = blockIdx.x*QSA_DEC_SQ_POOLS;
    const int n_cols = max(n_real, a.n_top);
    if (p0 >= n_cols || p0 >= a.n_pool) {
        return;
    }
    const bool all = n_real <= a.n_top;
    const int nq = a.n_idx_h*128;
    if (!all) {
        for (int i = threadIdx.x; i < a.n_t*nq; i += blockDim.x) {
            const int t = i / nq, h = (i % nq) / 128, d = i % 128;
            s_q[i] = *(const float *) ((const char *) a.idx_q + h*a.s_idx_q1 + t*a.s_idx_q2 + d*sizeof(float));
        }
        __syncthreads();
    }
    const int p = p0 + threadIdx.x / 8;
    const int l = threadIdx.x % 8;   // values 16*l .. 16*l + 15
    if (p >= n_cols || p >= a.n_pool) {
        return;
    }
    const bool live = p < n_real;
    float kv[16];
    if (live && !all) {
        const block_q8_0 * b = (const block_q8_0 *) (a.idx_k + (size_t) a.members[(size_t) p*a.kpool + a.kpool - 1]*a.s_idx_k) + l / 2;
        const float d = __half2float(b->d);
        const uint16_t * qs = (const uint16_t *) b->qs + (l % 2)*8;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint16_t w = qs[j];
            kv[2*j + 0] = d * (float) (int8_t) (w & 0xFF);
            kv[2*j + 1] = d * (float) (int8_t) (w >> 8);
        }
    }
    for (int t = 0; t < a.n_t; ++t) {
        const bool vis = live && a.meta[0] + (p + 1)*a.kpool - 1 <= a.tok_pos[t];
        float sc = 0.0f;
        if (vis && !all) {
            for (int h = 0; h < a.n_idx_h; ++h) {
                const float * qh = s_q + t*nq + h*128 + 16*l;
                float dp = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    dp += qh[j]*kv[j];
                }
                dp += __shfl_xor_sync(0xffffffff, dp, 1, 8);
                dp += __shfl_xor_sync(0xffffffff, dp, 2, 8);
                dp += __shfl_xor_sync(0xffffffff, dp, 4, 8);
                sc += a.idx_scale*fmaxf(dp, 0.0f);
            }
        }
        if (l == 0) {
            a.score[(size_t) t*a.n_pool + p] = vis ? sc : -INFINITY;
        }
    }
}

// 3. a block per query: the visible cells in entry order (the chosen pools' members, then the tail), by chunks per thread
static constexpr int QSA_DEC_THREADS = 256;
static constexpr int QSA_DEC_CT_THREADS = 1024;   // the compaction's

static __device__ __forceinline__ int32_t qsa_dec_cell(const qsa_dec_args & a, const int t, const int j, const int n_real) {
    const int n_pooled = a.n_top*a.kpool;
    int32_t c;
    if (j < n_pooled) {
        const int p = a.top[(size_t) t*a.n_top + j / a.kpool];
        if (p < 0 || p >= n_real || a.meta[0] + (p + 1)*a.kpool - 1 > a.tok_pos[t]) {
            return -1;
        }
        c = a.members[(size_t) p*a.kpool + j % a.kpool];
    } else {
        c = *(const int32_t *) ((const char *) a.tail + t*a.s_tail + (j - n_pooled)*sizeof(int32_t));
    }
    if (c < 0 || c >= a.n_cells) {
        return -1;
    }
    const int32_t pc = a.cell_pos[c];
    return pc >= 0 && pc <= a.tok_pos[t] ? c : -1;
}

static __global__ void k_qsa_dec_compact(const qsa_dec_args a) {
    const int t = blockIdx.x;
    const int n_real = min(a.n_pool, max(a.meta[1], 0));
    const int n_e = a.n_top*a.kpool + a.n_tail;
    const int per = (n_e + QSA_DEC_CT_THREADS - 1) / QSA_DEC_CT_THREADS;
    const int j0 = threadIdx.x*per, j1 = min(n_e, j0 + per);
    int n = 0;
    for (int j = j0; j < j1; ++j) {
        n += qsa_dec_cell(a, t, j, n_real) >= 0;
    }
    __shared__ int s_n[QSA_DEC_CT_THREADS];
    s_n[threadIdx.x] = n;
    __syncthreads();
    for (int off = 1; off < QSA_DEC_CT_THREADS; off <<= 1) {   // inclusive scan
        const int add = threadIdx.x >= off ? s_n[threadIdx.x - off] : 0;
        __syncthreads();
        s_n[threadIdx.x] += add;
        __syncthreads();
    }
    int o = s_n[threadIdx.x] - n;
    int32_t * rows = a.rows + (size_t) t*a.n_sel;
    for (int j = j0; j < j1; ++j) {
        const int32_t c = qsa_dec_cell(a, t, j, n_real);
        if (c >= 0) {
            rows[o++] = c;
        }
    }
    const int total = s_n[QSA_DEC_CT_THREADS - 1];
    const int kv_max = min(a.n_sel, (total + FATTN_KQ_STRIDE - 1) / FATTN_KQ_STRIDE * FATTN_KQ_STRIDE);
    half * mask = a.mask + (size_t) t*a.n_sel;
    for (int i = threadIdx.x; i < a.n_sel; i += QSA_DEC_CT_THREADS) {
        mask[i] = i < total ? __float2half(0.0f) : __float2half(-INFINITY);
        if (i >= total) {
            // past the count: cell 0 (finite, masked) up to the KV length, then nothing to gather (get_rows_q skips -2).
            // Under CUDA every row past the count is cell 0: the tensor-core attention (stream-k, several blocks a tile)
            // runs a block's first batch even past the KV length, and a NaN left in rows that were not gathered makes
            // the output NaN (QSA_DECODE with 2 and 4 queries; a short answer made of token 0, no draft accepted).
            // RTX 4090, a launch: the gather 6.7 us against 3.5 on a short context, 11.1 either way after 19k tokens;
            // the attention a block a tile instead (it stops at the KV length) 8.4 us against 7.0 + 2.6, 40.8 against
            // 11.6 after 19k tokens
#if defined(GGML_USE_HIP)
            rows[i] = i < kv_max ? 0 : -2;
#else
            rows[i] = 0;
#endif // defined(GGML_USE_HIP)
        }
    }
    if (threadIdx.x == 0) {
        a.cnt[t]    = total;
        a.kv_max[t] = kv_max;
    }
}

// 4. a warp per (row, query): its K and V rows of every KV head to F16, zeros past the count, nothing past the KV length
static constexpr int QSA_DEC_GATHER_ROWS = QSA_DEC_THREADS / WARP_SIZE;

static __global__ void k_qsa_dec_gather(const qsa_dec_args a) {
    const int t = blockIdx.y;
    const int i = blockIdx.x*QSA_DEC_GATHER_ROWS + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
#if defined(GGML_USE_HIP)
    if (i >= a.kv_max[t]) {
        return;
    }
#endif // defined(GGML_USE_HIP)
    const int32_t c = i < a.cnt[t] ? a.rows[(size_t) t*a.n_sel + i] : -1;
    const size_t o = ((size_t) t*a.n_sel + i)*a.n_kvh*a.D;
    for (int h = 0; h < a.n_kvh; ++h) {
        const char * kr = c >= 0 ? a.k + (size_t) c*a.k_nb2 + h*a.k_nb1 : nullptr;
        const char * vr = c >= 0 ? a.v + (size_t) c*a.v_nb2 + h*a.v_nb1 : nullptr;
        for (int d = lane; d < a.D; d += WARP_SIZE) {
            a.k16[o + (size_t) h*a.D + d] = __float2half(kr ? qsa_dec_load(kr, a.k_type, d) : 0.0f);
            a.v16[o + (size_t) h*a.D + d] = __float2half(vr ? qsa_dec_load(vr, a.v_type, d) : 0.0f);
        }
    }
}

// 4, a q8_0 / q4_0 K or V whose cells hold their heads adjacent: get_rows_q's rows (K and V in one launch, blockIdx.z)
static __global__ void k_qsa_dec_gather_q(const get_rows_q_args ak, const get_rows_q_args av) {
    get_rows_q_body(blockIdx.z == 0 ? ak : av, blockIdx.x, threadIdx.x);
}

static ggml_tensor qsa_dec_view(ggml_type type, void * data, ggml_backend_buffer_t buffer,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3) {
    ggml_tensor t = {};
    t.type = type; t.data = data; t.buffer = buffer;
    t.ne[0] = ne0; t.ne[1] = ne1; t.ne[2] = ne2; t.ne[3] = ne3;
    t.nb[0] = ggml_type_size(type); t.nb[1] = nb1; t.nb[2] = nb2; t.nb[3] = nb3;
    return t;
}

void ggml_cuda_op_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q        = dst->src[0];
    const ggml_tensor * idx_q    = dst->src[1];
    const ggml_tensor * idx_k    = dst->src[2];
    const ggml_tensor * members  = dst->src[3];
    const ggml_tensor * tail     = dst->src[4];
    const ggml_tensor * cell_pos = dst->src[5];
    const ggml_tensor * tok_pos  = dst->src[6];
    const ggml_tensor * meta     = dst->src[7];
    const ggml_tensor * k        = dst->src[8];
    const ggml_tensor * v        = dst->src[9];
    cudaStream_t stream = ctx.stream();
    ggml_cuda_mk_direct(ctx.mk, stream);

    qsa_dec_args a = {};
    a.idx_q = (const float *) idx_q->data; a.idx_k = (const char *) idx_k->data; a.members = (const int32_t *) members->data;
    a.tail = (const int32_t *) tail->data; a.cell_pos = (const int32_t *) cell_pos->data; a.tok_pos = (const int32_t *) tok_pos->data;
    a.meta = (const int32_t *) meta->data; a.k = (const char *) k->data; a.v = (const char *) v->data;
    a.idx_scale = ggml_get_op_params_f32(dst, 1);
    a.idx_dim = (int) idx_q->ne[0]; a.n_idx_h = (int) idx_q->ne[1]; a.kpool = (int) members->ne[0]; a.n_pool = (int) members->ne[1];
    a.n_top = std::min<int>(ggml_get_op_params_i32(dst, 2), a.n_pool); a.n_tail = (int) tail->ne[0];
    a.n_cells = (int) k->ne[2]; a.n_t = (int) q->ne[2]; a.n_kvh = (int) k->ne[1]; a.D = (int) q->ne[0];
    a.n_sel = GGML_PAD(a.n_top*a.kpool + a.n_tail, FATTN_KQ_STRIDE);
    a.idx_type = idx_k->type; a.k_type = k->type; a.v_type = v->type;
    a.s_idx_q1 = idx_q->nb[1]; a.s_idx_q2 = idx_q->nb[2]; a.s_idx_k = idx_k->nb[1]; a.s_tail = tail->nb[1];
    a.k_nb1 = k->nb[1]; a.k_nb2 = k->nb[2]; a.v_nb1 = v->nb[1]; a.v_nb2 = v->nb[2];

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float>   score(pool, (size_t) a.n_pool*a.n_t);
    ggml_cuda_pool_alloc<int32_t> top(pool, (size_t) a.n_top*a.n_t);
    ggml_cuda_pool_alloc<int32_t> rows(pool, (size_t) a.n_sel*a.n_t);
    ggml_cuda_pool_alloc<int32_t> cnt(pool, 2*(size_t) a.n_t);
    ggml_cuda_pool_alloc<half>    mask(pool, (size_t) a.n_sel*a.n_t);
    ggml_cuda_pool_alloc<half>    k16(pool, (size_t) a.n_sel*a.n_t*a.n_kvh*a.D);
    ggml_cuda_pool_alloc<half>    v16(pool, (size_t) a.n_sel*a.n_t*a.n_kvh*a.D);
    a.score = score.get(); a.top = top.get(); a.rows = rows.get(); a.cnt = cnt.get(); a.kv_max = cnt.get() + a.n_t;
    a.mask = mask.get(); a.k16 = k16.get(); a.v16 = v16.get();

    // 1, 2
    if (idx_k->type == GGML_TYPE_Q8_0 && a.idx_dim == 128 && a.n_t <= QSA_DEC_SQ_MAX_T) {
        const size_t smem = (size_t) a.n_t*a.n_idx_h*128*sizeof(float);
        k_qsa_dec_score_q8<<<(a.n_pool + QSA_DEC_SQ_POOLS - 1) / QSA_DEC_SQ_POOLS, 8*QSA_DEC_SQ_POOLS, smem, stream>>>(a);
    } else {
        const size_t smem = (size_t) a.n_idx_h*a.idx_dim*sizeof(float);
        k_qsa_dec_score<<<dim3((a.n_pool + QSA_DEC_SCORE_WARPS - 1) / QSA_DEC_SCORE_WARPS, a.n_t), QSA_DEC_SCORE_WARPS*WARP_SIZE, smem, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_top_k_live(pool, a.score, a.top, a.n_pool, a.n_t, a.n_top, a.meta + 1, stream);
    // 3, 4
    k_qsa_dec_compact<<<a.n_t, QSA_DEC_CT_THREADS, 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    const auto quant = [](ggml_type t) { return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0; };
    const bool rows_q = quant(k->type) && quant(v->type) && (a.n_kvh*a.D) % 16 == 0 &&
        k->nb[1] == ggml_row_size(k->type, a.D) && v->nb[1] == ggml_row_size(v->type, a.D);
    if (rows_q) {
        const auto args = [&](const ggml_tensor * src, half * dst16) {
            get_rows_q_args g = {};
            g.src0 = src->data; g.ids = a.rows; g.dst = dst16;
            g.ne00 = (int64_t) a.n_kvh*a.D; g.nb01 = src->nb[2]; g.nb02 = 0; g.nb03 = 0;
            g.ne10 = a.n_sel; g.ne11 = a.n_t; g.ne12 = 1;
            g.s10 = 1; g.s11 = a.n_sel; g.s12 = 0;
            g.s1 = (size_t) a.n_kvh*a.D; g.s2 = (size_t) a.n_sel*a.n_kvh*a.D; g.s3 = 0;
            g.type = src->type; g.n_rows = a.n_sel*a.n_t;
            return g;
        };
        const get_rows_q_args ak = args(k, a.k16), av = args(v, a.v16);
        k_qsa_dec_gather_q<<<dim3((ak.n_rows + GET_ROWS_Q_ROWS_PER_BLOCK - 1) / GET_ROWS_Q_ROWS_PER_BLOCK, 1, 2), 256, 0, stream>>>(ak, av);
    } else {
        k_qsa_dec_gather<<<dim3((a.n_sel + QSA_DEC_GATHER_ROWS - 1) / QSA_DEC_GATHER_ROWS, a.n_t), QSA_DEC_THREADS, 0, stream>>>(a);
    }
    CUDA_CHECK(cudaGetLastError());

    // 5: a query per stream, as the gathered attention of qwen4exp's graph (K/V [D, n_sel, n_kvh, n_t] of the rows)
    const int64_t D = a.D, n_head = q->ne[1], n_t = a.n_t, n_sel = a.n_sel, n_kvh = a.n_kvh;
    ggml_tensor tq = qsa_dec_view(GGML_TYPE_F32, q->data, q->buffer, D, 1, n_head, n_t, q->nb[2], q->nb[1], q->nb[2]);
    const size_t hb = D*sizeof(half);
    ggml_tensor tk = qsa_dec_view(GGML_TYPE_F16, a.k16, dst->buffer, D, n_sel, n_kvh, n_t, n_kvh*hb, hb, n_sel*n_kvh*hb);
    ggml_tensor tv = qsa_dec_view(GGML_TYPE_F16, a.v16, dst->buffer, D, n_sel, n_kvh, n_t, n_kvh*hb, hb, n_sel*n_kvh*hb);
    ggml_tensor tm = qsa_dec_view(GGML_TYPE_F16, a.mask, dst->buffer, n_sel, 1, 1, n_t,
                                  n_sel*sizeof(half), n_sel*sizeof(half), n_sel*sizeof(half));
    ggml_tensor fa = *dst;
    fa.op = GGML_OP_FLASH_ATTN_EXT;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        fa.src[i] = nullptr;
    }
    memset(fa.op_params, 0, sizeof(fa.op_params));
    const float params[3] = { ggml_get_op_params_f32(dst, 0), 0.0f, 0.0f };
    memcpy(fa.op_params, params, sizeof(params));
    fa.src[0] = &tq; fa.src[1] = &tk; fa.src[2] = &tv; fa.src[3] = &tm;
    ggml_prec_set_acc(&fa, GGML_PREC_F32);
    ctx.fattn_kv_max = a.kv_max;
    ggml_cuda_flash_attn_ext(ctx, &fa);
    ctx.fattn_kv_max = nullptr;
}

bool ggml_cuda_qsa_decode_supported(const ggml_tensor * op) {
    const ggml_tensor * idx_k = op->src[2];
    const ggml_tensor * k = op->src[8];
    const ggml_tensor * v = op->src[9];
    const auto kv_ok = [](ggml_type t) { return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0; };
    return (idx_k->type == GGML_TYPE_F32 || idx_k->type == GGML_TYPE_F16 || idx_k->type == GGML_TYPE_Q8_0) &&
           kv_ok(k->type) && kv_ok(v->type) && op->src[0]->ne[0] % WARP_SIZE == 0 && op->src[1]->ne[0] <= 8*WARP_SIZE &&
           (int64_t) op->src[1]->ne[0]*op->src[1]->ne[1]*sizeof(float) <= 48*1024;
}
