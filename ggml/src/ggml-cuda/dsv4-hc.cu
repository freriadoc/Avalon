#include "common.cuh"
#include "dsv4-hc.cuh"
#include "convert.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

// a thread per (column, token): grid x the column blocks, grid y the tokens (looped past the grid's limit)
template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int n_embd,
        int hc,
        int n_tokens,
        int sx0,
        int sx1,
        int64_t sx2,
        int sw0,
        int sw1,
        int64_t sw2,
        int sd0,
        int64_t sd1,
        float   scale,
        const ggml_cuda_img img) {
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    for (int it = blockIdx.y; it < n_tokens; it += gridDim.y) {
        float sum = 0.0f;
        for (int ih = 0; ih < hc; ++ih) {
            const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
            float wv;
            if constexpr (gated) {
                wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            } else {
                wv = weights[ih*sw0 + it*sw1];
            }
            sum += xv * wv;
        }

        dst[i0*sd0 + it*sd1] = scale * sum;
        if (img.p) {
            // the image of a contiguous dst (ggml_cuda_f16_image_target)
            img.store1((int64_t) it*n_embd + i0, scale * sum);   // q8_1: whole warps (n_embd % 32 == 0)
        }
    }
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
// a thread per (column, token) computes the column's hc streams, reading the residual's hc values once; the same
// arithmetic, in the same order, as one thread per output
constexpr int DSV4_HC_POST_MAX = 8;
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int n_embd,
        int hc,
        int n_tokens,
        int sx0,
        int64_t sx1,
        int sr0,
        int sr1,
        int64_t sr2,
        int sp0,
        int64_t sp1,
        int sc0,
        int sc1,
        int64_t sc2,
        int sd0,
        int sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i0 >= n_embd) {
        return;
    }

    ggml_cuda_pdl_sync();

    for (int it = blockIdx.y; it < n_tokens; it += gridDim.y) {
        const float xv = x[i0*sx0 + it*sx1];
        float r[DSV4_HC_POST_MAX];
        for (int isrc = 0; isrc < hc; ++isrc) {
            r[isrc] = residual[i0*sr0 + isrc*sr1 + it*sr2];
        }
        for (int idst = 0; idst < hc; ++idst) {
            float p = post[idst*sp0 + it*sp1];
            if constexpr (gated) {
                p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
            }

            float sum = xv * p;
            if constexpr (has_comb) {
                for (int isrc = 0; isrc < hc; ++isrc) {
                    sum += r[isrc] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } else {
                sum += r[idst];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        }
    }
}

// the gated hc_post (no comb) and the next half's rms_norm * w_norm in one pass (qwen4exp's prompt: hc_combine, then
// build_hc_pre's grouped norm of its result). A block per (stream, token) row with the vectorized rms_norm kernel's
// mapping (block_size threads, a float4 each) and reduction, and hc_post's arithmetic in its order: the two outputs are
// the bits of the two kernels, and the norm does not read the combined streams back. A thread reads its float4 of the
// residual before the block's barrier and writes its float4 of xn after it: xn may be the residual itself
template <int block_size>
static __global__ void dsv4_hc_post_norm_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * w_norm,
        float * dst,
        float * xn,
        int n_embd4,
        int n_tokens,
        int64_t sx1,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sw1,
        float gate_scale_in,
        float gate_scale_out,
        float eps,
        const ggml_cuda_img img) {
    ggml_cuda_pdl_lc();
    const int hc  = gridDim.x;
    const int c   = blockIdx.x;
    const int tid = threadIdx.x;
    const int ncols = 4*n_embd4;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    for (int it = blockIdx.y; it < n_tokens; it += gridDim.y) {
        float p = post[c*sp0 + it*sp1];
        p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));

        const int64_t row = (int64_t) it*hc + c;
        float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float tmp = 0.0f;
        if (tid < n_embd4) {
            const float4 xv = ((const float4 *) (x + it*sx1))[tid];
            const float4 rv = ((const float4 *) (residual + c*sr1 + it*sr2))[tid];
            float sum;
            sum = xv.x * p; sum += rv.x; y.x = sum;
            sum = xv.y * p; sum += rv.y; y.y = sum;
            sum = xv.z * p; sum += rv.z; y.z = sum;
            sum = xv.w * p; sum += rv.w; y.w = sum;
            ((float4 *) (dst + row*ncols))[tid] = y;
            tmp += y.x*y.x + y.y*y.y + y.z*y.z + y.w*y.w;
        }
        tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
        const float scale = rsqrtf(tmp / ncols + eps);

        if (tid < n_embd4) {
            float4 r = make_float4(scale*y.x, scale*y.y, scale*y.z, scale*y.w);
            const float4 m = ((const float4 *) (w_norm + c*sw1))[tid];
            r.x *= m.x;
            r.y *= m.y;
            r.z *= m.z;
            r.w *= m.w;
            ((float4 *) (xn + row*ncols))[tid] = r;
            if (img.p) {
                img.store4(row*ncols + 4*tid, r);   // q8_1: whole warps (ncols % 128 == 0, the caller's check)
            }
        }

        if constexpr (block_size > WARP_SIZE) {
            // s_sum is reused by the next token's reduction
            __syncthreads();
        }
    }
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_img img) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, (unsigned) std::min<int64_t>(n_tokens, 65535), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale, img);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    GGML_ASSERT(hc <= DSV4_HC_POST_MAX);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, (unsigned) std::min<int64_t>(n_tokens, 65535), 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}

bool ggml_cuda_dsv4_hc_post_norm_supported(const ggml_tensor * post_node, const ggml_tensor * mul) {
    const ggml_tensor * x        = post_node->src[0];
    const ggml_tensor * residual = post_node->src[1];
    const ggml_tensor * w_norm   = mul->src[1];
    const int64_t n_embd = x->ne[0];
    if (n_embd % 4 != 0 || n_embd/4 > 1024 || post_node->src[3] != nullptr) {
        return false;
    }
    if (x->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || w_norm->type != GGML_TYPE_F32 ||
            post_node->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    // the shapes of qwen4exp's hc tensors: block output [n_embd, nt], residual and both outputs [n_embd, hc, nt],
    // the norm weights [n_embd, hc]; rows contiguous and 16-byte aligned (the rms_norm kernel's vectorized path)
    const int64_t hc = residual->ne[1];
    const int64_t nt = residual->ne[2];
    if (x->ne[1] != nt || x->ne[2] != 1 || x->ne[3] != 1 || residual->ne[0] != n_embd || residual->ne[3] != 1 ||
            !ggml_are_same_shape(post_node, residual) || !ggml_are_same_shape(mul, residual) ||
            w_norm->ne[0] != n_embd || w_norm->ne[1] != hc || w_norm->ne[2] != 1 || w_norm->ne[3] != 1 || hc > 65535) {
        return false;
    }
    if (x->nb[0] != sizeof(float) || residual->nb[0] != sizeof(float) || w_norm->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(post_node) || !ggml_is_contiguous(mul)) {
        return false;
    }
    const auto aligned = [](const ggml_tensor * t, int dim) {
        return ((uintptr_t) t->data % 16) == 0 && (dim < 1 || t->nb[1] % 16 == 0) && (dim < 2 || t->nb[2] % 16 == 0);
    };
    return aligned(x, 1) && aligned(residual, 2) && aligned(w_norm, 1) && aligned(post_node, 0) && aligned(mul, 0);
}

void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post_node,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out, const ggml_tensor * rms_norm,
        ggml_tensor * mul, const ggml_cuda_img img) {
    const ggml_tensor * x        = post_node->src[0];
    const ggml_tensor * residual = post_node->src[1];
    const ggml_tensor * post     = gate_scale_in->src[0];
    const ggml_tensor * w_norm   = mul->src[1];

    float eps;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const int64_t n_embd4  = x->ne[0] / 4;
    const int64_t hc       = residual->ne[1];
    const int64_t n_tokens = residual->ne[2];

    const dim3 grid_dims((unsigned) hc, (unsigned) std::min<int64_t>(n_tokens, UINT16_MAX), 1);
    const float gate_in  = ggml_get_op_params_f32(gate_scale_in,  0);
    const float gate_out = ggml_get_op_params_f32(gate_scale_out, 0);

#define HC_POST_NORM_LAUNCH(BLOCK_SIZE) \
    do { \
        const ggml_cuda_kernel_launch_params launch_params(grid_dims, dim3(BLOCK_SIZE, 1, 1), \
                (BLOCK_SIZE) > WARP_SIZE ? 32 * sizeof(float) : 0, ctx.stream()); \
        ggml_cuda_kernel_launch(dsv4_hc_post_norm_f32<BLOCK_SIZE>, launch_params, \
            (const float *) x->data, (const float *) residual->data, (const float *) post->data, \
            (const float *) w_norm->data, (float *) post_node->data, (float *) mul->data, \
            (int) n_embd4, (int) n_tokens, \
            (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (residual->nb[1] / sizeof(float)), \
            (int64_t) (residual->nb[2] / sizeof(float)), \
            (int64_t) (post->nb[0] / sizeof(float)), (int64_t) (post->nb[1] / sizeof(float)), \
            (int64_t) (w_norm->nb[1] / sizeof(float)), gate_in, gate_out, eps, img); \
    } while (0)

    // the rms_norm dispatcher's block sizes for a row of n_embd4 float4 (rms_norm_f32_dispatch)
    if (n_embd4 <= 32) {
        HC_POST_NORM_LAUNCH(32);
    } else if (n_embd4 <= 128) {
        HC_POST_NORM_LAUNCH(128);
    } else if (n_embd4 <= 256) {
        HC_POST_NORM_LAUNCH(256);
    } else if (n_embd4 <= 512) {
        HC_POST_NORM_LAUNCH(512);
    } else {
        HC_POST_NORM_LAUNCH(1024);
    }
#undef HC_POST_NORM_LAUNCH
}
