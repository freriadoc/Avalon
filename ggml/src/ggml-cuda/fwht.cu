#include "common.cuh"
#include "convert.cuh"
#include "cpy-utils.cuh"
#include "fwht.cuh"
#include "quantize.cuh"

#include <vector>

// wide FWHT blocks use one row per thread block with this many threads
#define GGML_CUDA_FWHT_BLOCK_NT 256

template <int N, typename T>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * warp_size + lane]) * scale;
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Wide blocks: one row per thread block instead of per warp, so each thread keeps N/NT
// values rather than N/32. Stages below the warp width still shuffle, those up to the
// block width go through shared memory, and the rest stay in registers.
template <int N, int NT, typename T>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    ggml_cuda_pdl_sync();

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = ggml_cuda_cast<float>(src[i * NT + tid]) * scale;
    }

    // stages within a warp: partner differs in the lane bits
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }
}

template <typename T>
static bool ggml_cuda_op_fwht_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    const int     n    = src->ne[0];
    const int64_t rows = ggml_nrows(src);

    const T *     src_d = (const T *) src->data;
    float *       dst_d = (float *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;

    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;

    cudaStream_t                         stream = ctx.stream();
    dim3                                 grid_dims(num_blocks, 1, 1);
    dim3                                 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    const float scale = 1 / sqrtf(n);

    switch (n) {
        case 64:
            ggml_cuda_kernel_launch(fwht_cuda<64, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 128:
            ggml_cuda_kernel_launch(fwht_cuda<128, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 256:
            ggml_cuda_kernel_launch(fwht_cuda<256, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        case 512:
            ggml_cuda_kernel_launch(fwht_cuda<512, T>, launch_params, src_d, dst_d, rows, scale);
            return true;
        default:
            break;
    }

    // wide blocks: one row per thread block
    {
        constexpr int nt = GGML_CUDA_FWHT_BLOCK_NT;

        dim3 grid_dims_w(rows, 1, 1);
        dim3 block_dims_w(nt, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params_w =
            ggml_cuda_kernel_launch_params(grid_dims_w, block_dims_w, 0, stream);

        switch (n) {
            case 1024:
                ggml_cuda_kernel_launch(fwht_cuda_block<1024, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 2048:
                ggml_cuda_kernel_launch(fwht_cuda_block<2048, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
            case 4096:
                ggml_cuda_kernel_launch(fwht_cuda_block<4096, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#if !defined(GGML_USE_MUSA)
            // 32 KB of shared memory, above the MUSA limit; falls back there
            case 8192:
                ggml_cuda_kernel_launch(fwht_cuda_block<8192, nt, T>, launch_params_w, src_d, dst_d, rows, scale);
                return true;
#endif // !defined(GGML_USE_MUSA)
            default:
                return false;
        }
    }
}

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op) {
    const struct ggml_tensor * a = op->src[0];
    const struct ggml_tensor * b = op->src[1];

    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
           a->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
           (b->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F16) && ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
           ggml_are_same_shape(b, op);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    switch (src->type) {
        case GGML_TYPE_F32:
            return ggml_cuda_op_fwht_impl<float>(ctx, src, dst);
        case GGML_TYPE_F16:
            return ggml_cuda_op_fwht_impl<half>(ctx, src, dst);
        default:
            return false;
    }
}

// ---------------------------------------------------------------- rotation back x sigmoid(gate), with its q8_1

// One warp per row of N values: the transform of fwht_cuda<N, float> (same arithmetic), then each value times the
// sigmoid of its gate (unary_gated_op_kernel's op_sigmoid(g) * x), and the q8_1 blocks of the product as
// quantize_q8_1 writes them (N % 32 == 0: a register index of the warp is a block).
template <int N>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_gate_cuda(const float * src, const char * gate, const int64_t g_ne0, const int64_t g_ne1,
        const int64_t g_ne2, const size_t g_nb0, const size_t g_nb1, const size_t g_nb2, const size_t g_nb3,
        float * dst, block_q8_1 * dst_q8_1, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == QK8_1, "a q8_1 block per register index");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }
    src += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        const int64_t e = r * N + i * warp_size + lane;
        int64_t t = e;
        const int64_t i0 = t % g_ne0; t /= g_ne0;
        const int64_t i1 = t % g_ne1; t /= g_ne1;
        const int64_t i2 = t % g_ne2; t /= g_ne2;
        const float g = *(const float *) (gate + i0*g_nb0 + i1*g_nb1 + i2*g_nb2 + t*g_nb3);
        const float v = (1.0f / (1.0f + expf(-g))) * reg[i];
        dst[e] = v;
        if (dst_q8_1) {
            const float amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float sum  = warp_reduce_sum<QK8_1>(v);
            const float d    = amax / 127.0f;
            dst_q8_1[e / QK8_1].qs[e % QK8_1] = amax == 0.0f ? 0 : roundf(v / d);
            if (lane == 0) {
                dst_q8_1[e / QK8_1].ds = make_half2(d, sum);
            }
        }
    }
}

static bool fwht_gate_is_view(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

static const ggml_tensor * fwht_gate_root(const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        t = t->src[0];
    }
    return t;
}

// nodes i (fwht), cont, sigmoid, mul; views and reshapes may sit in between
static bool fwht_gate_match(const ggml_cgraph * cgraph, int i, int idx[4]) {
    const ggml_tensor * mm = cgraph->nodes[i];
    if (!ggml_cuda_op_mul_mat_use_fwht(mm) || mm->ne[0] != 64 || mm->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    idx[0] = i;
    int k = i + 1;
    for (int j = 1; j < 4; j++) {
        while (k < cgraph->n_nodes && fwht_gate_is_view(cgraph->nodes[k])) {
            k++;
        }
        if (k >= cgraph->n_nodes) {
            return false;
        }
        idx[j] = k++;
    }
    const ggml_tensor * cont = cgraph->nodes[idx[1]];
    const ggml_tensor * sig  = cgraph->nodes[idx[2]];
    const ggml_tensor * mul  = cgraph->nodes[idx[3]];
    if (cont->op != GGML_OP_CONT || sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID ||
        mul->op != GGML_OP_MUL || fwht_gate_root(sig->src[0]) != cont) {
        return false;
    }
    const ggml_tensor * other = mul->src[0] == sig ? mul->src[1] : mul->src[1] == sig ? mul->src[0] : nullptr;
    const ggml_tensor * gate  = cont->src[0];
    const int64_t n = ggml_nelements(mm);
    return other && fwht_gate_root(other) == mm && cont->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32 &&
        sig->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 && ggml_is_contiguous(mul) && ggml_is_contiguous(cont) &&
        ggml_nelements(cont) == n && ggml_nelements(sig) == n && ggml_nelements(mul) == n && ggml_nelements(gate) == n &&
        ggml_are_same_shape(mul, sig);   // cont: the gate view in its row-major order (CONT_2D of a 3D view)
}

ggml_tensor * ggml_cuda_match_fwht_gate(const ggml_cgraph * cgraph, int i, ggml_tensor ** rows, ggml_tensor ** gate_src) {
    int idx[4];
    if (!fwht_gate_match(cgraph, i, idx)) {
        return nullptr;
    }
    *rows     = cgraph->nodes[i]->src[1];
    *gate_src = cgraph->nodes[idx[1]]->src[0];
    return cgraph->nodes[idx[3]];
}

int ggml_cuda_try_fuse_fwht_gate(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_FWHT_GATE") != nullptr;
    int idx[4];
    if (disabled || !fwht_gate_match(cgraph, i, idx)) {
        return 0;
    }
    // the nodes of the group: the four computing ones and the views of them in between (a view of a tensor computed
    // before the group, the gate's, stays out: it is an input)
    std::vector<int> nodes;
    std::vector<ggml_op> ops;
    for (int k = i; k <= idx[3]; k++) {
        const ggml_tensor * n = cgraph->nodes[k];
        bool in = k == idx[0] || k == idx[1] || k == idx[2] || k == idx[3];
        if (!in && fwht_gate_is_view(n)) {
            for (const ggml_tensor * v = n->view_src ? n->view_src : n->src[0]; v && !in; v = v->view_src) {
                for (int j : nodes) {
                    in = in || cgraph->nodes[j] == v;
                }
            }
        }
        if (in) {
            nodes.push_back(k);
            ops.push_back(n->op);
        }
    }
    const int out_idx = idx[3];
    if (!ggml_can_fuse_subgraph_ext(cgraph, nodes.data(), (int) nodes.size(), ops.data(), &out_idx, 1)) {
        return 0;
    }
    const ggml_tensor * mm   = cgraph->nodes[i];
    const ggml_tensor * rows = mm->src[1];
    const ggml_tensor * gate = cgraph->nodes[idx[1]]->src[0];
    ggml_tensor *       mul  = cgraph->nodes[idx[3]];
    // a warp reads its row and the gate values of its row, other warps write theirs: no overlap at all
    const auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * al = (const char *) a->data, * ah = al + ggml_nbytes(a);
        const char * bl = (const char *) b->data, * bh = bl + ggml_nbytes(b);
        return al < bh && bl < ah;
    };
    if (!ggml_is_contiguous(rows) || overlap(mul, rows) || overlap(mul, gate)) {
        return 0;
    }
    const int64_t n_rows = ggml_nelements(mm) / 64;
    block_q8_1 * q8_1 = (block_q8_1 *) ggml_cuda_q8_1_produce(ctx, cgraph, idx[3]);
    const int rows_per_block = 4;
    const dim3 grid((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, rows_per_block, 1);
    fwht_gate_cuda<64><<<grid, block, 0, ctx.stream()>>>((const float *) rows->data, (const char *) gate->data,
            gate->ne[0], gate->ne[1], gate->ne[2], gate->nb[0], gate->nb[1], gate->nb[2], gate->nb[3],
            (float *) mul->data, q8_1, n_rows, 1.0f / sqrtf(64.0f));
    return idx[3] - i;
}

// ---------------------------------------------------------------- rotation written into a quantized cache

// One warp per row of N values: the transform of fwht_cuda<N, float> (same arithmetic), then the row quantized into
// the set_rows destination as k_set_rows_quant writes it (a register index of the warp is a block). Row r is part
// rr = r % rows_per_set of the set_rows row r / rows_per_set, which goes to destination row idx[r / rows_per_set].
template <int N, typename B>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_set_rows_cuda(const float * src, const int64_t * idx, B * dst, const int64_t rows_per_set,
        const int64_t dst_row_blocks, const int64_t n_rows, const float scale) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    static_assert(warp_size == 32 && N % 32 == 0, "a block per register index");

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n_rows) {
        return;
    }
    src += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = src[i * warp_size + lane] * scale;
    }
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    B * y = dst + idx[r / rows_per_set] * dst_row_blocks + (r % rows_per_set) * el_w;
    if constexpr (std::is_same_v<B, block_q4_0>) {
#pragma unroll
        for (int i = 0; i < el_w; ++i) {
            quantize_warp_q4_0(reg[i], y + i, lane);
        }
    } else {
        quantize_warp_q8_0_n<el_w>(reg, y, lane, el_w);
    }
}

// nodes i (fwht of 64-wide rows), views of it, then the SET_ROWS of its rows into a Q4_0 / Q8_0 tensor
static bool fwht_set_rows_match(const ggml_cgraph * cgraph, int i, int * sr_idx) {
    const ggml_tensor * mm = cgraph->nodes[i];
    if (!ggml_cuda_op_mul_mat_use_fwht(mm) || mm->ne[0] != 64 || mm->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    int k = i + 1;
    while (k < cgraph->n_nodes && fwht_gate_is_view(cgraph->nodes[k]) && cgraph->nodes[k]->view_src == mm &&
           cgraph->nodes[k]->view_offs == 0) {
        k++;
    }
    if (k >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * sr  = cgraph->nodes[k];
    if (sr->op != GGML_OP_SET_ROWS) {
        return false;
    }
    const ggml_tensor * src = sr->src[0];
    const ggml_tensor * ids = sr->src[1];
    *sr_idx = k;
    return (src == mm || (src->view_src == mm && src->view_offs == 0)) && ggml_is_contiguous(src) &&
        (sr->type == GGML_TYPE_Q4_0 || sr->type == GGML_TYPE_Q8_0) && src->type == GGML_TYPE_F32 &&
        ids->type == GGML_TYPE_I64 && ggml_is_contiguous(ids) &&
        src->ne[0] % 64 == 0 && src->ne[2] == 1 && src->ne[3] == 1 && ids->ne[0] == src->ne[1] &&
        ggml_nelements(ids) == ids->ne[0] && ggml_nelements(src) == ggml_nelements(mm) &&
        sr->ne[0] == src->ne[0] && sr->nb[1] % ggml_type_size(sr->type) == 0;
}

int ggml_cuda_try_fuse_fwht_set_rows(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_FWHT_SET_ROWS") != nullptr;
    int k;
    if (disabled || !fwht_set_rows_match(cgraph, i, &k)) {
        return 0;
    }
    int nodes[8];
    ggml_op ops[8];
    int n = 0;
    if (k - i + 1 > 8) {
        return 0;
    }
    for (int j = i; j <= k; j++) {
        nodes[n] = j;
        ops[n++] = cgraph->nodes[j]->op;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, nodes, n, ops, &k, 1)) {
        return 0;
    }
    const ggml_tensor * mm   = cgraph->nodes[i];
    const ggml_tensor * rows = mm->src[1];
    ggml_tensor *       sr   = cgraph->nodes[k];
    // the warps read the rows and write the destination's blocks (another tensor: the cache)
    const char * rl = (const char *) rows->data, * rh = rl + ggml_nbytes(rows);
    const char * dl = (const char *) sr->data,   * dh = dl + ggml_nbytes(sr);
    if (!ggml_is_contiguous(rows) || (rl < dh && dl < rh)) {
        return 0;
    }
    const ggml_tensor * src = sr->src[0];
    const int64_t n_rows         = ggml_nelements(mm) / 64;
    const int64_t rows_per_set   = src->ne[0] / 64;
    const int64_t dst_row_blocks = sr->nb[1] / ggml_type_size(sr->type);
    const int rows_per_block = 4;
    const dim3 grid((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, rows_per_block, 1);
    const int64_t * idx = (const int64_t *) sr->src[1]->data;
    const float scale = 1.0f / sqrtf(64.0f);
    if (sr->type == GGML_TYPE_Q4_0) {
        fwht_set_rows_cuda<64, block_q4_0><<<grid, block, 0, ctx.stream()>>>((const float *) rows->data, idx,
                (block_q4_0 *) sr->data, rows_per_set, dst_row_blocks, n_rows, scale);
    } else {
        fwht_set_rows_cuda<64, block_q8_0><<<grid, block, 0, ctx.stream()>>>((const float *) rows->data, idx,
                (block_q8_0 *) sr->data, rows_per_set, dst_row_blocks, n_rows, scale);
    }
    return k - i;
}
