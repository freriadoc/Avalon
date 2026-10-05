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
//   K2 W_i and W_d over xn      (one pass; xn in f16, written by K1 when the two follow each other)
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

static bool qwen4_hc_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * al = (const char *) a->data, * ah = al + ggml_nbytes(a);
    const char * bl = (const char *) b->data, * bh = bl + ggml_nbytes(b);
    return al < bh && bl < ah;
}

static const ggml_tensor * qwen4_hc_view_base(const ggml_tensor * t) {
    while (t && qwen4_hc_is_view(t)) {
        t = t->src[0];
    }
    return t;
}

// ---------------------------------------------------------------- K1

// One workgroup per (stream, token): the norm is over the embedding of one stream. Every input of
// the stream (residual, block output, inject) is read before the barrier of the norm reduction and
// every output is written after it, so an output may alias the residual exactly (ggml-alloc gives
// the new residual, or xn, the old residual's memory); any other overlap refuses the fusion, since
// the block output and the inject are shared by the streams of a token.
static constexpr int QWEN4_HC_K1_THREADS = 1024;
static constexpr int QWEN4_HC_K1_PER_THREAD = 4; // n_embd <= 4096

static __global__ void qwen4_hc_combine_norm_f32(
        const float * residual, const float * block, const float * inject, const float * gamma,
        float * res_out, float * xn, half * xh,
        int n_embd, int hc, float s1, float b1, float s2, float b2, float eps) {
    const int c = blockIdx.x;
    const int t = blockIdx.y;

    const float v = inject[c + t*hc] * s1 + b1;
    const float w = (1.0f / (1.0f + expf(-v))) * s2 + b2;

    const float * r_in = residual + ((size_t) t*hc + c)*n_embd;
    const float * b_in = block + (size_t) t*n_embd;

    float r[QWEN4_HC_K1_PER_THREAD];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < QWEN4_HC_K1_PER_THREAD; ++k) {
        const int e = threadIdx.x + k*QWEN4_HC_K1_THREADS;
        r[k] = e < n_embd ? r_in[e] + b_in[e] * w : 0.0f;
        ss += r[k] * r[k];
    }

    __shared__ float partial[QWEN4_HC_K1_THREADS / 32];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    ss = warp_reduce_sum(ss);
    if (lane == 0) {
        partial[warp] = ss;
    }
    __syncthreads(); // every input of the stream has been read
    float sum = lane < (int) (blockDim.x / WARP_SIZE) ? partial[lane] : 0.0f;
    sum = warp_reduce_sum(sum);
    const float scale = rsqrtf(sum / n_embd + eps);

    float * r_out = res_out + ((size_t) t*hc + c)*n_embd;
    float * x_out = xn + ((size_t) t*hc + c)*n_embd;
    const float * g = gamma + (size_t) c*n_embd;
#pragma unroll
    for (int k = 0; k < QWEN4_HC_K1_PER_THREAD; ++k) {
        const int e = threadIdx.x + k*QWEN4_HC_K1_THREADS;
        if (e < n_embd) {
            r_out[e] = r[k];
            const float xv = r[k] * scale * g[e];
            x_out[e] = xv;
            if (xh) {
                xh[((size_t) t*hc + c)*n_embd + e] = __float2half(xv);   // for K2, the same layout as xn
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

// xn is multiplied in f16 (the graph allows converting it down to F16, ggml_prec_set_src): the Q8_0 weights
// convert exactly, the products of two f16 are exact in f32 and are summed in f32. The activation is 10240
// per token and every row's workgroup reads it all, so its size matters: f16 halves the f32 traffic, and the
// conversion is free when K1 writes it (otherwise one launch, like the q8 quantization it replaces).
// Against an f64 reference (tests/hcbench, random data) the NMSE is ~3e-8, against 2 - 6e-5 for q8_1.
static __global__ void qwen4_hc_to_f16(const float * x, half * xh, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        xh[i] = __float2half(x[i]);
    }
}

// c + a.x*b.x + a.y*b.y with f32 accumulation
static __device__ __forceinline__ float qwen4_hc_dot2(const half2 a, const half2 b, const float c) {
#if defined(GGML_USE_HIP) && (defined(RDNA2) || defined(RDNA3) || defined(RDNA4) || defined(CDNA))
    typedef _Float16 h2 __attribute__((ext_vector_type(2)));
    h2 av, bv;
    memcpy(&av, &a, 4);
    memcpy(&bv, &b, 4);
    return __builtin_amdgcn_fdot2(av, bv, c, false);
#else
    const float2 af = __half22float2(a);
    const float2 bf = __half22float2(b);
    return fmaf(af.y, bf.y, fmaf(af.x, bf.x, c));
#endif
}

static constexpr int QWEN4_HC_K2_WARPS = 4; // warps sharing one output row

// one workgroup per output row: rows [0, n_a) from W_a, [n_a, n_a + n_b) from W_b (then scale + silu);
// a thread takes whole Q8_0 blocks, neighbouring threads neighbouring blocks. With n_b = 0 it is a plain
// product (ggml_cuda_mul_mat_q8_0_f16, the unfused path), with the same numbers.
template <int T>
static __global__ void qwen4_hc_down_inject_f16(
        const block_q8_0 * wa, const block_q8_0 * wb, const half * xh,
        float * out_a, float * out_b, int n_in, int n_a, int n_b, float s, float b) {
    const int nb   = n_in / QK8_0;
    const int row  = blockIdx.x;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const block_q8_0 * w = row < n_a ? wa + (size_t) row * nb : wb + (size_t) (row - n_a) * nb;

    float acc[T] = { 0.0f };
    for (int ib = threadIdx.x; ib < nb; ib += blockDim.x) {
        const float dw = __half2float(w[ib].d);
        half2 wh[QK8_0/2];
#pragma unroll
        for (int j = 0; j < QK8_0/4; ++j) {
            const int wi = get_int_b2(w[ib].qs, j);
            wh[2*j + 0] = make_half2(__int2half_rn((int8_t) wi),         __int2half_rn((int8_t) (wi >> 8)));
            wh[2*j + 1] = make_half2(__int2half_rn((int8_t) (wi >> 16)), __int2half_rn((int8_t) (wi >> 24)));
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const int4 * xb = (const int4 *) (xh + (size_t) t*n_in + ib*QK8_0);
            float sum = 0.0f;
#pragma unroll
            for (int q = 0; q < QK8_0/8; ++q) {
                const int4 v = xb[q];
                const int vv[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    half2 xv;
                    memcpy(&xv, &vv[u], 4);
                    sum = qwen4_hc_dot2(wh[4*q + u], xv, sum);
                }
            }
            acc[t] += dw * sum;
        }
    }

    __shared__ float partial[QWEN4_HC_K2_WARPS][T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float v = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            partial[warp][t] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        const int t = threadIdx.x;
        float v = 0.0f;
#pragma unroll
        for (int k = 0; k < QWEN4_HC_K2_WARPS; ++k) {
            v += partial[k][t];
        }
        if (row < n_a) {
            out_a[(size_t) t*n_a + row] = v;
        } else {
            const float y = v * s + b;
            out_b[(size_t) t*n_b + row - n_a] = y / (1.0f + expf(-y));
        }
    }
}

template <int T>
static void qwen4_hc_down_inject_launch(
        const block_q8_0 * wa, const block_q8_0 * wb, const half * xh,
        float * out_a, float * out_b, int n_in, int n_a, int n_b, int T_run, float s, float b, cudaStream_t stream) {
    if constexpr (T > 1) {
        if (T_run < T) {
            qwen4_hc_down_inject_launch<T - 1>(wa, wb, xh, out_a, out_b, n_in, n_a, n_b, T_run, s, b, stream);
            return;
        }
    }
    qwen4_hc_down_inject_f16<T><<<n_a + n_b, QWEN4_HC_K2_WARPS*WARP_SIZE, 0, stream>>>(
            wa, wb, xh, out_a, out_b, n_in, n_a, n_b, s, b);
}

static void qwen4_hc_to_f16_launch(const float * x, half * xh, int n, cudaStream_t stream) {
    qwen4_hc_to_f16<<<(n + 255) / 256, 256, 0, stream>>>(x, xh, n);
}

bool ggml_cuda_mul_mat_q8_0_f16_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->ne[1] >= 1 && src1->ne[1] <= QWEN4_HC_MAX_TOKENS && src0->ne[0] % QK8_0 == 0;
}

void ggml_cuda_mul_mat_q8_0_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_q8_0_f16_supported(src0, src1, dst));
    const int n = (int) src0->ne[0];
    const int T = (int) src1->ne[1];
    ggml_cuda_pool_alloc<half> xh(ctx.pool(), (size_t) T * n);
    qwen4_hc_to_f16_launch((const float *) src1->data, xh.get(), T * n, ctx.stream());
    qwen4_hc_down_inject_launch<QWEN4_HC_MAX_TOKENS>((const block_q8_0 *) src0->data, nullptr, xh.get(),
            (float *) dst->data, nullptr, n, (int) src0->ne[1], 0, T, 1.0f, 0.0f, ctx.stream());
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

    // the fused kernel multiplies xn in f16, as the unfused products do when the graph allows it
    return qwen4_hc_is_unary(silu, GGML_UNARY_OP_SILU) && silu->src[0] == sc && sc->src[0] == mm_b &&
        mm_b->src[1] == x &&
        (ggml_prec) ggml_get_op_params_i32(mm_a, 3) == GGML_PREC_F16 &&
        (ggml_prec) ggml_get_op_params_i32(mm_b, 3) == GGML_PREC_F16 &&
        mm_a->src[0]->type == GGML_TYPE_Q8_0 && mm_b->src[0]->type == GGML_TYPE_Q8_0 &&
        ggml_is_contiguous(mm_a->src[0]) && ggml_is_contiguous(mm_b->src[0]) &&
        ggml_nelements(mm_a->src[0]) == mm_a->src[0]->ne[0]*mm_a->src[0]->ne[1] &&
        ggml_nelements(mm_b->src[0]) == mm_b->src[0]->ne[0]*mm_b->src[0]->ne[1] &&
        qwen4_hc_f32_cont(x) && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->ne[1] >= 1 && x->ne[1] <= QWEN4_HC_MAX_TOKENS && x->ne[0] % QK8_0 == 0 &&
        mm_a->src[0]->ne[0] == x->ne[0] && mm_b->src[0]->ne[0] == x->ne[0] &&
        qwen4_hc_f32_cont(mm_a) && qwen4_hc_f32_cont(silu);
}

// ---------------------------------------------------------------- K3

// one warp per embedding channel e: the rows e + c*n_embd of W_u (c < hc <= 4) go to four groups
// of 8 lanes (qwen4_hc_row_q8_0_f32); each group reduces its gate, then the gated mean over the streams
// is the epilogue
static constexpr int QWEN4_HC_K3_THREADS = 256;
static constexpr int QWEN4_HC_K3_MAX_LO  = 512;

// Q8_0 weights times f32 activations without quantizing them, for a row of nb blocks read by a group of
// 8 lanes: lane k takes the k-th int (4 weights) of every block and the group reduces. x is [T][nb*32] (shared
// memory). Used both by the fused up/pre kernel and by the unfused product whose src[1] may not be
// converted below F32 (ggml_prec_set_src, ggml_cuda_mul_mat_q8_0_f32): the two paths compute the same numbers.
template <int T>
static __device__ __forceinline__ void qwen4_hc_row_q8_0_f32(
        const block_q8_0 * wr, int nb, const float * x, int k, float * acc) {
    const float4 * x4 = (const float4 *) x;   // lane k reads the k-th float4 of a block: no bank conflict
    const int n4 = nb*QK8_0/4;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.0f;
    }
#pragma unroll 2
    for (int ib = 0; ib < nb; ++ib) {
        const float dw = __half2float(wr[ib].d);
        const int   wi = get_int_b2(wr[ib].qs, k);
        const float w0 = (float) (int8_t) (wi      );
        const float w1 = (float) (int8_t) (wi >>  8);
        const float w2 = (float) (int8_t) (wi >> 16);
        const float w3 = (float) (int8_t) (wi >> 24);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 xv = x4[t*n4 + ib*(QK8_0/4) + k];
            acc[t] += dw * (w0*xv.x + w1*xv.y + w2*xv.z + w3*xv.w);
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = warp_reduce_sum<8>(acc[t]);
    }
}

// x (n floats, contiguous) into shared memory, n <= MAX: every load is issued before the first store, so
// the copy costs one memory latency rather than one per round
template <int MAX>
static __device__ __forceinline__ void qwen4_hc_load_shared(const float * x, int n, float * xs) {
    constexpr int ROUNDS = (MAX + 255) / 256;   // workgroups of at least 256 threads
    float v[ROUNDS];
#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int i = threadIdx.x + r*blockDim.x;
        v[r] = i < n ? x[i] : 0.0f;
    }
#pragma unroll
    for (int r = 0; r < ROUNDS; ++r) {
        const int i = threadIdx.x + r*blockDim.x;
        if (i < n) {
            xs[i] = v[r];
        }
    }
}

template <int T>
static __global__ void qwen4_hc_up_pre_q8_0(
        const block_q8_0 * w, const float * lo, const float * xn, float * mixed,
        int n_lo, int n_embd, int hc, float scale) {
    extern __shared__ float qwen4_hc_k3_smem[];
    const int nb = n_lo / QK8_0;
    qwen4_hc_load_shared<T*QWEN4_HC_K3_MAX_LO>(lo, T*n_lo, qwen4_hc_k3_smem);
    __syncthreads();

    const int e = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (e >= n_embd) {
        return;
    }
    const int c = lane >> 3;
    const int k = lane & 7;

    float acc[T];
    qwen4_hc_row_q8_0_f32<T>(w + ((size_t) (e + min(c, hc - 1)*n_embd)) * nb, nb, qwen4_hc_k3_smem, k, acc);
#pragma unroll
    for (int t = 0; t < T; ++t) {
        float m = (c < hc && k == 0) ? xn[((size_t) t*hc + c) * n_embd + e] / (1.0f + expf(-acc[t])) : 0.0f;
        m += __shfl_xor_sync(0xffffffff, m, 8, WARP_SIZE);
        m += __shfl_xor_sync(0xffffffff, m, 16, WARP_SIZE);
        if (lane == 0) {
            mixed[(size_t) t*n_embd + e] = scale * m;
        }
    }
}

static constexpr int QWEN4_HC_MM_MAX_N = 1024;

// dst[t][r] = W[r] . x[t] for Q8_0 W [nrows][n <= QWEN4_HC_MM_MAX_N], f32 x [T][n]: four rows per warp
template <int T>
static __global__ void qwen4_hc_mul_mat_q8_0_f32(const block_q8_0 * w, const float * x, float * dst, int n, int nrows) {
    extern __shared__ float qwen4_hc_mm_smem[];
    qwen4_hc_load_shared<T*QWEN4_HC_MM_MAX_N>(x, T*n, qwen4_hc_mm_smem);
    __syncthreads();
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = (blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE) * 4 + (lane >> 3);
    if (row - (lane >> 3) >= nrows) {
        return;
    }
    const int nb = n / QK8_0;
    float acc[T];
    qwen4_hc_row_q8_0_f32<T>(w + (size_t) min(row, nrows - 1) * nb, nb, qwen4_hc_mm_smem, lane & 7, acc);
    if (row < nrows && (lane & 7) == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            dst[(size_t) t*nrows + row] = acc[t];
        }
    }
}

template <int T>
static void qwen4_hc_mul_mat_launch(const block_q8_0 * w, const float * x, float * dst, int n, int nrows, int T_run,
        cudaStream_t stream) {
    if constexpr (T > 1) {
        if (T_run < T) {
            qwen4_hc_mul_mat_launch<T - 1>(w, x, dst, n, nrows, T_run, stream);
            return;
        }
    }
    const int warps = 8;
    qwen4_hc_mul_mat_q8_0_f32<T><<<(nrows + 4*warps - 1) / (4*warps), warps*WARP_SIZE, (size_t) T*n*sizeof(float), stream>>>(
            w, x, dst, n, nrows);
}

bool ggml_cuda_mul_mat_q8_0_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src1->ne[1] >= 1 && src1->ne[1] <= QWEN4_HC_MAX_TOKENS && src0->ne[0] % QK8_0 == 0 &&
        src0->ne[0] <= QWEN4_HC_MM_MAX_N;
}

void ggml_cuda_mul_mat_q8_0_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_q8_0_f32_supported(src0, src1, dst));
    qwen4_hc_mul_mat_launch<QWEN4_HC_MAX_TOKENS>((const block_q8_0 *) src0->data, (const float *) src1->data,
            (float *) dst->data, (int) src0->ne[0], (int) src0->ne[1], (int) src1->ne[1], ctx.stream());
}

template <int T>
static void qwen4_hc_up_pre_launch(int grid, int block, size_t smem, cudaStream_t stream,
        const block_q8_0 * w, const float * lo, const float * xn, float * mixed,
        int n_lo, int n_embd, int hc, int T_run, float scale) {
    if constexpr (T > 1) {
        if (T_run < T) {
            qwen4_hc_up_pre_launch<T - 1>(grid, block, smem, stream, w, lo, xn, mixed, n_lo, n_embd, hc, T_run, scale);
            return;
        }
    }
    qwen4_hc_up_pre_q8_0<T><<<grid, block, smem, stream>>>(w, lo, xn, mixed, n_lo, n_embd, hc, scale);
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

    // the fused kernel multiplies lo in f32, as the unfused product does when the graph forbids converting lo
    return qwen4_hc_view_base(pre->src[1]) == mm && ggml_is_contiguous(pre->src[1]) &&
        ggml_get_op_params_i32(pre, 1) != 0 && (ggml_prec) ggml_get_op_params_i32(mm, 3) == GGML_PREC_F32 &&
        mm->src[0]->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(mm->src[0]) &&
        qwen4_hc_f32_cont(lo) && qwen4_hc_f32_cont(xn) && qwen4_hc_f32_cont(pre) && qwen4_hc_f32_cont(mm) &&
        hc >= 1 && hc <= 4 && T >= 1 && T <= QWEN4_HC_MAX_TOKENS && xn->ne[3] == 1 &&
        lo->ne[0] % QK8_0 == 0 && lo->ne[0] <= QWEN4_HC_K3_MAX_LO && ggml_nelements(lo) == lo->ne[0]*T &&
        mm->src[0]->ne[0] == lo->ne[0] && mm->src[0]->ne[1] == n_embd*hc &&
        ggml_nelements(pre) == n_embd*T;
}

// ---------------------------------------------------------------- entry

// views write nothing: only K2's two real outputs enter the overlap check
static bool qwen4_hc_k2_ranges_ok(const ggml_cgraph * cgraph, int i, const qwen4_hc_span & span,
        bool (*check_ranges)(const ggml_cgraph *, int, int, const int *, int)) {
    const int outputs[] = { span.idx[0], span.idx[3] };
    return check_ranges(cgraph, i, span.n_all, outputs, 2);
}

static void qwen4_hc_k2_launch(const ggml_cgraph * cgraph, const qwen4_hc_span & span, const half * xh, cudaStream_t stream) {
    ggml_tensor *       mm_a = cgraph->nodes[span.idx[0]];
    const ggml_tensor * mm_b = cgraph->nodes[span.idx[1]];
    const ggml_tensor * sc   = cgraph->nodes[span.idx[2]];
    ggml_tensor *       silu = cgraph->nodes[span.idx[3]];
    const ggml_tensor * x    = mm_a->src[1];
    qwen4_hc_down_inject_launch<QWEN4_HC_MAX_TOKENS>(
            (const block_q8_0 *) mm_a->src[0]->data, (const block_q8_0 *) mm_b->src[0]->data, xh,
            (float *) mm_a->data, (float *) silu->data, (int) x->ne[0], (int) mm_a->src[0]->ne[1], (int) mm_b->src[0]->ne[1],
            (int) x->ne[1], ggml_get_op_params_f32(sc, 0), ggml_get_op_params_f32(sc, 1), stream);
}

int ggml_cuda_try_fuse_qwen4_hc(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*check_ranges)(const ggml_cgraph *, int, int, const int *, int)) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_QWEN4_HC") != nullptr; // A/B against the unfused ops
    if (disabled) {
        return 0;
    }
    const ggml_tensor * node = cgraph->nodes[i];
    cudaStream_t stream = ctx.stream();

    if (node->op == GGML_OP_SCALE && qwen4_hc_match_k1(cgraph, i)) {
        // A workgroup per (stream, token) reads all of its stream's inputs before writing its stream's
        // outputs: an output may alias the residual exactly (same layout: ggml-alloc gives the new
        // residual the previous one's memory), whatever the number of tokens. Any other overlap refuses the fusion
        // (the block output and the inject have another per-token layout, gamma is shared).
        const ggml_tensor * post_n = cgraph->nodes[i + 3];
        const ggml_tensor * mul_n  = cgraph->nodes[i + 5];
        const ggml_tensor * ins[3] = { post_n->src[0], cgraph->nodes[i]->src[0], mul_n->src[1] };
        const ggml_tensor * res    = post_n->src[1];
        const ggml_tensor * outs[2] = { post_n, mul_n };
        bool ok = !qwen4_hc_overlap(post_n, mul_n);
        for (const ggml_tensor * o : outs) {
            for (const ggml_tensor * in : ins) {
                ok = ok && !qwen4_hc_overlap(o, in);
            }
            ok = ok && (!qwen4_hc_overlap(o, res) ||
                        (o->data == res->data && ggml_nbytes(o) == ggml_nbytes(res) && ggml_is_contiguous(res) && ggml_is_contiguous(o)));
        }
        if (!ok) {
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

        // K2 right after (views in between) on this xn: K1 also writes xn in f16 for it, no conversion launch
        int j = i + 6;
        while (j < cgraph->n_nodes && qwen4_hc_is_view(cgraph->nodes[j])) {
            ++j;
        }
        qwen4_hc_span span2;
        const bool with_k2 = j < cgraph->n_nodes && cgraph->nodes[j]->op == GGML_OP_MUL_MAT &&
            qwen4_hc_match_k2(cgraph, j, span2) && qwen4_hc_view_base(cgraph->nodes[span2.idx[0]]->src[1]) == mul &&
            qwen4_hc_k2_ranges_ok(cgraph, j, span2, check_ranges);
        ggml_cuda_pool_alloc<half> xh(ctx.pool());
        if (with_k2) {
            xh.alloc((size_t) T * hc * n_embd);
        }
        qwen4_hc_combine_norm_f32<<<dim3(hc, T), QWEN4_HC_K1_THREADS, 0, stream>>>(
                (const float *) residual->data, (const float *) post->src[0]->data,
                (const float *) sc1->src[0]->data, (const float *) mul->src[1]->data,
                (float *) post->data, (float *) mul->data, with_k2 ? xh.get() : nullptr, n_embd, hc,
                ggml_get_op_params_f32(sc1, 0), ggml_get_op_params_f32(sc1, 1),
                ggml_get_op_params_f32(sc2, 0), ggml_get_op_params_f32(sc2, 1), eps);
        if (!with_k2) {
            return 5;
        }
        qwen4_hc_k2_launch(cgraph, span2, xh.get(), stream);
        return (j - i) + span2.n_all - 1;
    }

    qwen4_hc_span span2;
    if (node->op == GGML_OP_MUL_MAT && qwen4_hc_match_k2(cgraph, i, span2) && qwen4_hc_k2_ranges_ok(cgraph, i, span2, check_ranges)) {
        const ggml_tensor * x = cgraph->nodes[span2.idx[0]]->src[1];
        ggml_cuda_pool_alloc<half> xh(ctx.pool(), (size_t) ggml_nelements(x));
        qwen4_hc_to_f16_launch((const float *) x->data, xh.get(), (int) ggml_nelements(x), stream);
        qwen4_hc_k2_launch(cgraph, span2, xh.get(), stream);
        return span2.n_all - 1;
    }

    qwen4_hc_span span3;
    if (node->op == GGML_OP_MUL_MAT && qwen4_hc_match_k3(cgraph, i, span3)) {
        const ggml_tensor * mm  = cgraph->nodes[span3.idx[0]];
        ggml_tensor *       pre = cgraph->nodes[span3.idx[1]];
        const ggml_tensor * lo  = mm->src[1];
        const ggml_tensor * xn  = pre->src[0];
        // the kernel reads the weights, lo (every workgroup, into shared memory) and xn. ggml-alloc may
        // give the output the memory of lo, dead once the product read it: lo is then copied first (two
        // launches, against three unfused), so the fusion no longer depends on the placement. An
        // overlap with xn refuses it.
        if (qwen4_hc_overlap(pre, xn) || !ggml_is_contiguous(lo)) {
            return 0;
        }
        GGML_UNUSED(check_ranges);
        ggml_cuda_pool_alloc<float> lo_copy(ctx.pool());
        const float * lo_d = (const float *) lo->data;
        if (qwen4_hc_overlap(pre, lo)) {
            lo_copy.alloc(ggml_nelements(lo));
            CUDA_CHECK(cudaMemcpyAsync(lo_copy.get(), lo->data, ggml_nbytes(lo), cudaMemcpyDeviceToDevice, stream));
            lo_d = lo_copy.get();
        }
        const int n_lo   = (int) lo->ne[0];
        const int n_embd = (int) xn->ne[0];
        const int hc     = (int) xn->ne[1];
        const int T      = (int) xn->ne[2];
        const int warps = QWEN4_HC_K3_THREADS / WARP_SIZE;
        const size_t smem = (size_t) T * n_lo * sizeof(float);
        qwen4_hc_up_pre_launch<QWEN4_HC_MAX_TOKENS>((n_embd + warps - 1) / warps, warps*WARP_SIZE, smem, stream,
                (const block_q8_0 *) mm->src[0]->data, lo_d, (const float *) xn->data,
                (float *) pre->data, n_lo, n_embd, hc, T, ggml_get_op_params_f32(pre, 0));
        return span3.n_all - 1;
    }

    return 0;
}
