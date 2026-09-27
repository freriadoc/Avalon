// Fused hyper-connection boundary for small batches (qwen4exp decoding).
//
// Between two blocks the graph runs, per token, 15 nodes on a [n_embd, hc] residual:
//   combine: SCALE, SIGMOID, SCALE, DSV4_HC_POST       residual += block * (2*sigmoid(inject/hc))
//   norm:    RMS_NORM, MUL                             xn = rms_norm(stream) * gamma
//   mix:     MUL_MAT inject, MUL_MAT down, SCALE, SILU  inject = W_i xn, lo = silu(W_d xn / hc)
//            MUL_MAT up, DSV4_HC_PRE                   mixed = mean_c xn_c * sigmoid(W_u lo)_c
// For one token these tensors are a few KiB: every node is a kernel launch and a pass over
// data that fits in cache, while only the three low-rank weights are real memory traffic.
// Three kernels do the same work:
//   K1 combine + grouped norm   (reads the residual once, writes the new residual and xn)
//   K2 W_i and W_d over xn      (one pass; xn quantized once per workgroup, dp4a)
//   K3 W_u with the gated stream mean as its epilogue (the four rows of an embedding
//      channel, e + c*n_embd, are computed by the same warp)

#include "qwen4-hc.cuh"
#include "common.cuh"
#include "vecdotq.cuh"

#include <cmath>

static constexpr int QWEN4_HC_MAX_TOKENS = 8;

static bool qwen4_hc_is_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

static bool qwen4_hc_f32_cont(const ggml_tensor * t) {
    return t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

// Collects the nodes of an expected op sequence starting at node i, allowing view-like nodes
// (no computation) in between. Fills idx[] with the positions of the expected ops, all[] and ops[]
// with every node of the span; views are reported separately so they can be declared as outputs.
struct qwen4_hc_span {
    int idx[8];
    int n_idx = 0;
    int all[24];
    ggml_op ops[24];
    int n_all = 0;
    int views[16];
    int n_views = 0;
};

static bool qwen4_hc_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW;
}

static bool qwen4_hc_collect(const ggml_cgraph * cgraph, int i, const ggml_op * expected, int n_expected,
        qwen4_hc_span & span) {
    int j = i;
    for (int k = 0; k < n_expected; ++k) {
        while (k > 0 && j < cgraph->n_nodes && qwen4_hc_is_view(cgraph->nodes[j]) && expected[k] != cgraph->nodes[j]->op) {
            if (span.n_all >= 24 || span.n_views >= 16) {
                return false;
            }
            span.views[span.n_views++] = j;
            span.ops[span.n_all] = cgraph->nodes[j]->op;
            span.all[span.n_all++] = j;
            ++j;
        }
        if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != expected[k] || span.n_all >= 24) {
            return false;
        }
        span.idx[span.n_idx++] = j;
        span.ops[span.n_all] = cgraph->nodes[j]->op;
        span.all[span.n_all++] = j;
        ++j;
    }
    return true;
}

// the span is fusable if nothing outside uses an elided node; outputs are the given expected-op
// positions plus every view (views write nothing but may be read later)
static bool qwen4_hc_span_fusable(const ggml_cgraph * cgraph, const qwen4_hc_span & span,
        const int * out_k, int n_out_k) {
    int outputs[24];
    int n = 0;
    for (int k = 0; k < n_out_k; ++k) {
        outputs[n++] = span.idx[out_k[k]];
    }
    for (int k = 0; k < span.n_views; ++k) {
        outputs[n++] = span.views[k];
    }
    return ggml_can_fuse_subgraph_ext(cgraph, span.all, span.n_all, span.ops, outputs, n);
}

static const ggml_tensor * qwen4_hc_view_base(const ggml_tensor * t) {
    while (t && qwen4_hc_is_view(t)) {
        t = t->src[0];
    }
    return t;
}

// ---------------------------------------------------------------- K1

// One workgroup per token handles all its streams: every input of the token (residual, block
// output, inject) is read before the barrier of the norm reduction and every output is written
// after it, so outputs may alias inputs of the same token (ggml-alloc often places xn over the
// old residual or the block output).
static constexpr int QWEN4_HC_K1_THREADS = 1024;
static constexpr int QWEN4_HC_K1_PER_THREAD = 4; // n_embd <= 4096

static __global__ void qwen4_hc_combine_norm_f32(
        const float * residual, const float * block, const float * inject, const float * gamma,
        float * res_out, float * xn,
        int n_embd, int hc, float s1, float b1, float s2, float b2, float eps) {
    const int t = blockIdx.x;

    float w[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float v = c < hc ? inject[c + t*hc] * s1 + b1 : 0.0f;
        w[c] = (1.0f / (1.0f + expf(-v))) * s2 + b2;
    }

    const float * r_in = residual + (size_t) t*hc*n_embd;
    const float * b_in = block + (size_t) t*n_embd;

    float r[QWEN4_HC_K1_PER_THREAD][4];
    float ss[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (int k = 0; k < QWEN4_HC_K1_PER_THREAD; ++k) {
        const int e = threadIdx.x + k*QWEN4_HC_K1_THREADS;
        const float bv = e < n_embd ? b_in[e] : 0.0f;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            r[k][c] = (e < n_embd && c < hc) ? r_in[(size_t) c*n_embd + e] + bv * w[c] : 0.0f;
            ss[c] += r[k][c] * r[k][c];
        }
    }

    __shared__ float partial[QWEN4_HC_K1_THREADS / 32][4];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        ss[c] = warp_reduce_sum(ss[c]);
    }
    if (lane == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            partial[warp][c] = ss[c];
        }
    }
    __syncthreads(); // every input of the token has been read
    float scale[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float v = lane < (int) (blockDim.x / WARP_SIZE) ? partial[lane][c] : 0.0f;
        v = warp_reduce_sum(v);
        scale[c] = rsqrtf(v / n_embd + eps);
    }

    float * r_out = res_out + (size_t) t*hc*n_embd;
    float * x_out = xn + (size_t) t*hc*n_embd;
#pragma unroll
    for (int k = 0; k < QWEN4_HC_K1_PER_THREAD; ++k) {
        const int e = threadIdx.x + k*QWEN4_HC_K1_THREADS;
        if (e < n_embd) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (c < hc) {
                    r_out[(size_t) c*n_embd + e] = r[k][c];
                    x_out[(size_t) c*n_embd + e] = r[k][c] * scale[c] * gamma[(size_t) c*n_embd + e];
                }
            }
        }
    }
}

static bool qwen4_hc_match_k1(const ggml_cgraph * cgraph, int i) {
    static const ggml_op ops[] = { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST,
                                   GGML_OP_RMS_NORM, GGML_OP_MUL };
    const int outputs[] = { i + 3, i + 5 };
    if (!ggml_can_fuse_subgraph(cgraph, i, 6, ops, outputs, 2)) {
        return false;
    }
    const ggml_tensor * sc1  = cgraph->nodes[i];
    const ggml_tensor * sig  = cgraph->nodes[i + 1];
    const ggml_tensor * sc2  = cgraph->nodes[i + 2];
    const ggml_tensor * post = cgraph->nodes[i + 3];
    const ggml_tensor * rms  = cgraph->nodes[i + 4];
    const ggml_tensor * mul  = cgraph->nodes[i + 5];

    const ggml_tensor * inject   = sc1->src[0];
    const ggml_tensor * block    = post->src[0];
    const ggml_tensor * residual = post->src[1];
    const ggml_tensor * gamma    = mul->src[1];

    const int64_t n_embd = residual->ne[0];
    const int64_t hc     = residual->ne[1];
    const int64_t T      = residual->ne[2];

    return qwen4_hc_is_unary(sig, GGML_UNARY_OP_SIGMOID) &&
        sig->src[0] == sc1 && sc2->src[0] == sig && post->src[2] == sc2 && post->src[3] == nullptr &&
        rms->src[0] == post && mul->src[0] == rms &&
        qwen4_hc_f32_cont(inject) && qwen4_hc_f32_cont(block) && qwen4_hc_f32_cont(residual) &&
        qwen4_hc_f32_cont(gamma) && qwen4_hc_f32_cont(post) && qwen4_hc_f32_cont(mul) &&
        residual->ne[3] == 1 && T >= 1 && T <= QWEN4_HC_MAX_TOKENS &&
        hc <= 4 && n_embd <= QWEN4_HC_K1_THREADS*QWEN4_HC_K1_PER_THREAD &&
        inject->ne[0] == hc && ggml_nelements(inject) == hc*T &&
        block->ne[0] == n_embd && ggml_nelements(block) == n_embd*T &&
        gamma->ne[0] == n_embd && gamma->ne[1] == hc && ggml_nelements(gamma) == n_embd*hc &&
        ggml_are_same_shape(post, residual) && ggml_are_same_shape(mul, residual);
}

// ---------------------------------------------------------------- K2

// Activations are quantized like quantize_q8_1 does for mul_mat_vec_q (per block of 32:
// d = amax/127, q = round(x/d), d then kept in f16) into shared memory, once per workgroup, and dotted with the
// Q8_0 weights by dp4a: the fused kernels give the numbers of the unfused path.
static __device__ __forceinline__ void qwen4_hc_quantize_shared(
        const float * x, int n, int8_t * xq, float * xd) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    for (int ib = warp; ib < n / QK8_1; ib += nwarps) {
        const float v = x[ib*QK8_1 + lane];
        const float amax = warp_reduce_max(fabsf(v));
        const float d = amax / 127.0f;
        xq[ib*QK8_1 + lane] = amax == 0.0f ? 0 : (int8_t) roundf(v / d);
        if (lane == 0) {
            xd[ib] = __half2float(__float2half(d)); // block_q8_1 keeps d in f16
        }
    }
}

static constexpr int QWEN4_HC_K2_CHUNK = 4; // tokens quantized in shared memory at a time

// one warp per output row: rows [0, n_a) from W_a, [n_a, n_a + n_b) from W_b (then scale + silu);
// a lane takes whole Q8_0 blocks, neighbouring lanes neighbouring blocks
static __global__ void qwen4_hc_down_inject_q8_0(
        const block_q8_0 * wa, const block_q8_0 * wb, const float * x,
        float * out_a, float * out_b,
        int n_in, int n_a, int n_b, int T, float s, float b) {
    extern __shared__ int qwen4_hc_k2_smem[];
    const int nb = n_in / QK8_0;
    int8_t * xq = (int8_t *) qwen4_hc_k2_smem;                        // [CHUNK][n_in]
    float  * xd = (float *) (xq + (size_t) QWEN4_HC_K2_CHUNK*n_in);    // [CHUNK][nb]

    const int row  = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const block_q8_0 * w = row < n_a ? wa + (size_t) row * nb : wb + (size_t) (row - n_a) * nb;

    for (int t0 = 0; t0 < T; t0 += QWEN4_HC_K2_CHUNK) {
        const int tn = min(QWEN4_HC_K2_CHUNK, T - t0);
        __syncthreads();
        for (int t = 0; t < tn; ++t) {
            qwen4_hc_quantize_shared(x + (size_t) (t0 + t)*n_in, n_in, xq + (size_t) t*n_in, xd + (size_t) t*nb);
        }
        __syncthreads();
        if (row >= n_a + n_b) {
            continue;
        }

        float acc[QWEN4_HC_K2_CHUNK] = { 0.0f };
        for (int ib = lane; ib < nb; ib += WARP_SIZE) {
            const float dw = __half2float(w[ib].d);
            int wi[QK8_0/4];
#pragma unroll
            for (int k = 0; k < QK8_0/4; ++k) {
                wi[k] = get_int_b2(w[ib].qs, k);
            }
            for (int t = 0; t < tn; ++t) {
                const int * xi = (const int *) (xq + (size_t) t*n_in + ib*QK8_0);
                int sumi = 0;
#pragma unroll
                for (int k = 0; k < QK8_0/4; ++k) {
                    sumi = ggml_cuda_dp4a(wi[k], xi[k], sumi);
                }
                acc[t] += dw * xd[t*nb + ib] * (float) sumi;
            }
        }
        for (int t = 0; t < tn; ++t) {
            const float v = warp_reduce_sum(acc[t]);
            if (lane == 0) {
                if (row < n_a) {
                    out_a[(size_t) (t0 + t)*n_a + row] = v;
                } else {
                    const float y = v * s + b;
                    out_b[(size_t) (t0 + t)*n_b + row - n_a] = y / (1.0f + expf(-y));
                }
            }
        }
    }
}

static bool qwen4_hc_match_k2(const ggml_cgraph * cgraph, int i, qwen4_hc_span & span) {
    static const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_SCALE, GGML_OP_UNARY };
    const int out_k[] = { 0, 3 };
    if (!qwen4_hc_collect(cgraph, i, ops, 4, span) || !qwen4_hc_span_fusable(cgraph, span, out_k, 2)) {
        return false;
    }
    const ggml_tensor * mm_a = cgraph->nodes[span.idx[0]];
    const ggml_tensor * mm_b = cgraph->nodes[span.idx[1]];
    const ggml_tensor * sc   = cgraph->nodes[span.idx[2]];
    const ggml_tensor * silu = cgraph->nodes[span.idx[3]];
    const ggml_tensor * x    = mm_a->src[1];

    return qwen4_hc_is_unary(silu, GGML_UNARY_OP_SILU) && silu->src[0] == sc && sc->src[0] == mm_b &&
        mm_b->src[1] == x &&
        mm_a->src[0]->type == GGML_TYPE_Q8_0 && mm_b->src[0]->type == GGML_TYPE_Q8_0 &&
        ggml_is_contiguous(mm_a->src[0]) && ggml_is_contiguous(mm_b->src[0]) &&
        ggml_nelements(mm_a->src[0]) == mm_a->src[0]->ne[0]*mm_a->src[0]->ne[1] &&
        ggml_nelements(mm_b->src[0]) == mm_b->src[0]->ne[0]*mm_b->src[0]->ne[1] &&
        qwen4_hc_f32_cont(x) && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->ne[1] >= 1 && x->ne[1] <= QWEN4_HC_MAX_TOKENS && x->ne[0] % QK8_0 == 0 &&
        QWEN4_HC_K2_CHUNK * x->ne[0] * (1 + sizeof(float) / QK8_0) <= 48*1024 &&
        mm_a->src[0]->ne[0] == x->ne[0] && mm_b->src[0]->ne[0] == x->ne[0] &&
        qwen4_hc_f32_cont(mm_a) && qwen4_hc_f32_cont(silu);
}

// ---------------------------------------------------------------- K3

// one warp per embedding channel e: the rows e + c*n_embd of W_u (c < hc <= 4) go to four groups
// of 8 lanes, lane k of a group takes int k of every block (a group reads a block's 32 bytes
// together); each group reduces its gate, then the gated mean over the streams is the epilogue
static __global__ void qwen4_hc_up_pre_q8_0(
        const block_q8_0 * w, const float * lo, const float * xn, float * mixed,
        int n_lo, int n_embd, int hc, int T, float scale) {
    extern __shared__ int qwen4_hc_k3_smem[];
    const int nb = n_lo / QK8_0;
    int8_t * lq = (int8_t *) qwen4_hc_k3_smem;                 // [T][n_lo]
    float  * ld = (float *) (lq + (size_t) T*n_lo);             // [T][nb]
    for (int t = 0; t < T; ++t) {
        qwen4_hc_quantize_shared(lo + (size_t) t*n_lo, n_lo, lq + (size_t) t*n_lo, ld + (size_t) t*nb);
    }
    __syncthreads();

    const int e = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (e >= n_embd) {
        return;
    }
    const int c = lane >> 3;
    const int k = lane & 7;

    float acc[QWEN4_HC_MAX_TOKENS] = { 0.0f };
    if (c < hc) {
        const block_q8_0 * wr = w + ((size_t) (e + c*n_embd)) * nb;
        for (int ib = 0; ib < nb; ++ib) {
            const float dw = __half2float(wr[ib].d);
            const int wi = get_int_b2(wr[ib].qs, k);
            for (int t = 0; t < T; ++t) {
                const int xi = ((const int *) (lq + (size_t) t*n_lo + ib*QK8_0))[k];
                acc[t] += dw * ld[t*nb + ib] * (float) ggml_cuda_dp4a(wi, xi, 0);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        const float g = warp_reduce_sum<8>(acc[t]);
        float m = (c < hc && k == 0) ? xn[((size_t) t*hc + c) * n_embd + e] / (1.0f + expf(-g)) : 0.0f;
        m += __shfl_xor_sync(0xffffffff, m, 8, WARP_SIZE);
        m += __shfl_xor_sync(0xffffffff, m, 16, WARP_SIZE);
        if (lane == 0) {
            mixed[(size_t) t*n_embd + e] = scale * m;
        }
    }
}

static bool qwen4_hc_match_k3(const ggml_cgraph * cgraph, int i, qwen4_hc_span & span) {
    static const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_DSV4_HC_PRE };
    const int out_k[] = { 1 };
    if (!qwen4_hc_collect(cgraph, i, ops, 2, span) || !qwen4_hc_span_fusable(cgraph, span, out_k, 1)) {
        return false;
    }
    const ggml_tensor * mm  = cgraph->nodes[span.idx[0]];
    const ggml_tensor * pre = cgraph->nodes[span.idx[1]];
    const ggml_tensor * lo  = mm->src[1];
    const ggml_tensor * xn  = pre->src[0];

    const int64_t n_embd = xn->ne[0];
    const int64_t hc     = xn->ne[1];
    const int64_t T      = xn->ne[2];

    return qwen4_hc_view_base(pre->src[1]) == mm && ggml_is_contiguous(pre->src[1]) &&
        ggml_get_op_params_i32(pre, 1) != 0 &&
        mm->src[0]->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(mm->src[0]) &&
        qwen4_hc_f32_cont(lo) && qwen4_hc_f32_cont(xn) && qwen4_hc_f32_cont(pre) && qwen4_hc_f32_cont(mm) &&
        hc >= 1 && hc <= 4 && T >= 1 && T <= QWEN4_HC_MAX_TOKENS && xn->ne[3] == 1 &&
        lo->ne[0] % QK8_0 == 0 && ggml_nelements(lo) == lo->ne[0]*T &&
        mm->src[0]->ne[0] == lo->ne[0] && mm->src[0]->ne[1] == n_embd*hc &&
        ggml_nelements(pre) == n_embd*T;
}

// ---------------------------------------------------------------- entry

int ggml_cuda_try_fuse_qwen4_hc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*check_ranges)(const ggml_cgraph *, int, int, const int *, int)) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_QWEN4_HC") != nullptr; // A/B against the unfused ops
    if (disabled) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    cudaStream_t stream = ctx.stream();

    if (node->op == GGML_OP_SCALE && qwen4_hc_match_k1(cgraph, i)) {
        // one token: a single workgroup reads all inputs before writing, aliasing is safe
        const int outputs[] = { i + 3, i + 5 };
        if (cgraph->nodes[i + 3]->src[1]->ne[2] != 1 && !check_ranges(cgraph, i, 6, outputs, 2)) {
            return 0;
        }
        const ggml_tensor * sc1  = cgraph->nodes[i];
        const ggml_tensor * sc2  = cgraph->nodes[i + 2];
        ggml_tensor *       post = cgraph->nodes[i + 3];
        const ggml_tensor * rms  = cgraph->nodes[i + 4];
        ggml_tensor *       mul  = cgraph->nodes[i + 5];
        const ggml_tensor * residual = post->src[1];
        const int n_embd = (int) residual->ne[0];
        const int hc     = (int) residual->ne[1];
        const int T      = (int) residual->ne[2];
        float eps;
        memcpy(&eps, rms->op_params, sizeof(float));
        qwen4_hc_combine_norm_f32<<<T, QWEN4_HC_K1_THREADS, 0, stream>>>(
                (const float *) residual->data, (const float *) post->src[0]->data,
                (const float *) sc1->src[0]->data, (const float *) mul->src[1]->data,
                (float *) post->data, (float *) mul->data, n_embd, hc,
                ggml_get_op_params_f32(sc1, 0), ggml_get_op_params_f32(sc1, 1),
                ggml_get_op_params_f32(sc2, 0), ggml_get_op_params_f32(sc2, 1), eps);
        return 5;
    }

    qwen4_hc_span span2;
    if (node->op == GGML_OP_MUL_MAT && qwen4_hc_match_k2(cgraph, i, span2)) {
        // views write nothing: only the two real outputs enter the overlap check
        const int outputs[] = { span2.idx[0], span2.idx[3] };
        if (!check_ranges(cgraph, i, span2.n_all, outputs, 2)) {
            return 0;
        }
        ggml_tensor * mm_a = cgraph->nodes[span2.idx[0]];
        const ggml_tensor * mm_b = cgraph->nodes[span2.idx[1]];
        const ggml_tensor * sc   = cgraph->nodes[span2.idx[2]];
        ggml_tensor * silu = cgraph->nodes[span2.idx[3]];
        const ggml_tensor * x = mm_a->src[1];
        const int n_in = (int) x->ne[0];
        const int T    = (int) x->ne[1];
        const int n_a  = (int) mm_a->src[0]->ne[1];
        const int n_b  = (int) mm_b->src[0]->ne[1];
        const int warps = 8;
        const size_t smem = (size_t) QWEN4_HC_K2_CHUNK * (n_in + (n_in / QK8_0) * sizeof(float));
        qwen4_hc_down_inject_q8_0<<<(n_a + n_b + warps - 1) / warps, warps*WARP_SIZE, smem, stream>>>(
                (const block_q8_0 *) mm_a->src[0]->data, (const block_q8_0 *) mm_b->src[0]->data,
                (const float *) x->data, (float *) mm_a->data, (float *) silu->data,
                n_in, n_a, n_b, T, ggml_get_op_params_f32(sc, 0), ggml_get_op_params_f32(sc, 1));
        return span2.n_all - 1;
    }

    qwen4_hc_span span3;
    if (node->op == GGML_OP_MUL_MAT && qwen4_hc_match_k3(cgraph, i, span3)) {
        const int outputs[] = { span3.idx[1] };
        if (!check_ranges(cgraph, i, span3.n_all, outputs, 1)) {
            return 0;
        }
        const ggml_tensor * mm  = cgraph->nodes[span3.idx[0]];
        ggml_tensor *       pre = cgraph->nodes[span3.idx[1]];
        const ggml_tensor * lo  = mm->src[1];
        const ggml_tensor * xn  = pre->src[0];
        const int n_lo   = (int) lo->ne[0];
        const int n_embd = (int) xn->ne[0];
        const int hc     = (int) xn->ne[1];
        const int T      = (int) xn->ne[2];
        const int warps = 8;
        const size_t smem = (size_t) T * (n_lo + (n_lo / QK8_0) * sizeof(float));
        qwen4_hc_up_pre_q8_0<<<(n_embd + warps - 1) / warps, warps*WARP_SIZE, smem, stream>>>(
                (const block_q8_0 *) mm->src[0]->data, (const float *) lo->data, (const float *) xn->data,
                (float *) pre->data, n_lo, n_embd, hc, T, ggml_get_op_params_f32(pre, 0));
        return span3.n_all - 1;
    }

    return 0;
}
