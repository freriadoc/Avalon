#include "mk.cuh"
#include "grid-barrier.cuh"

// the multi-task kernel: a block finds its task, then runs the original kernel's body for its virtual block
__launch_bounds__(GGML_CUDA_MK_THREADS, 1)
static __global__ void k_mk(const ggml_cuda_mk_tasks T) {
    int i = 0;
    while (i + 1 < T.n && (int) blockIdx.x >= T.t[i + 1].first_block) {
        i++;
    }
    const ggml_cuda_mk_task & t = T.t[i];
    const int vb = blockIdx.x - t.first_block;
    switch (t.kind) {
        case GGML_CUDA_MK_ROPE:
        case GGML_CUDA_MK_ROPE_FF: {
            const rms_norm_mul_rope_args & a = t.p.rope;
            const int row     = vb % a.n_rows;
            const int channel = (vb / a.n_rows) % a.n_channels;
            const int sample  = vb / (a.n_rows * a.n_channels);
            if (t.kind == GGML_CUDA_MK_ROPE) {
                rms_norm_mul_rope_f32_body<GGML_CUDA_MK_THREADS, false, float>(a, row, channel, sample);
            } else {
                rms_norm_mul_rope_f32_body<GGML_CUDA_MK_THREADS, true, float>(a, row, channel, sample);
            }
        } break;
        case GGML_CUDA_MK_FWHT_SR: {
            const fwht_set_rows_args & a = t.p.fwht;
            const int64_t r = (int64_t) vb * (GGML_CUDA_MK_THREADS / WARP_SIZE) + threadIdx.x / WARP_SIZE;
            if (a.q4_0) {
                fwht_set_rows_body<64, block_q4_0>(a, r, threadIdx.x % WARP_SIZE);
            } else {
                fwht_set_rows_body<64, block_q8_0>(a, r, threadIdx.x % WARP_SIZE);
            }
        } break;
        case GGML_CUDA_MK_QSA_SEL: {
            const qsa_select_args & a = t.p.sel;
            const int bx = (int) ((a.n_sel + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS);
            const int64_t j = (int64_t) (vb % bx) * GGML_CUDA_MK_THREADS + threadIdx.x;
            const int64_t q = vb / bx;
#define MK_SEL(M) (a.by_pos ? k_qsa_select_body<M, true>(a, j, q) : k_qsa_select_body<M, false>(a, j, q))
            switch (a.mode) {
                case GGML_QSA_SELECT_ROWS: MK_SEL(GGML_QSA_SELECT_ROWS); break;
                case GGML_QSA_SELECT_MASK: MK_SEL(GGML_QSA_SELECT_MASK); break;
                default:                   MK_SEL(GGML_QSA_SELECT_CAUSAL); break;
            }
#undef MK_SEL
        } break;
        case GGML_CUDA_MK_SSM_CONV: {
            // blocks of 128 threads: the other half of the block leaves (the original kernel's threads past the
            // channels leave before its barriers too)
            if (threadIdx.x >= 128) {
                return;
            }
            const ssm_conv_cached_args & a = t.p.conv;
            const int bx = (int) ((a.n_ch + 127) / 128);
            ssm_conv_cached_body<4, GGML_CUDA_SSM_CONV_CACHED_MAX_T>(a, vb % bx, vb / bx, threadIdx.x);
        } break;
        case GGML_CUDA_MK_GDN_GATES:
            gdn_gates_body(t.p.gates, (int64_t) vb * GGML_CUDA_MK_THREADS + threadIdx.x);
            break;
        case GGML_CUDA_MK_GATHER:
            get_rows_q_body(t.p.gather, vb, threadIdx.x);
            break;
        case GGML_CUDA_MK_FILL: {
            const int64_t j = (int64_t) vb * GGML_CUDA_MK_THREADS + threadIdx.x;
            if (j < t.p.fill.n) {
                t.p.fill.dst[j] = t.p.fill.value;
            }
        } break;
        case GGML_CUDA_MK_SR_Q8:
            set_rows_q8_body(t.p.sr_q8, (int64_t) vb * GGML_CUDA_MK_THREADS + threadIdx.x);
            break;
        case GGML_CUDA_MK_GR_MEAN:
            get_rows_mean_body(t.p.gr_mean, (int64_t) vb * GGML_CUDA_MK_THREADS + threadIdx.x);
            break;
    }
}

// task classes: the generic kernel's blocks of 256 threads (A), the chain kernel's of 128 (B), either (C)
static bool mk_class_a(int k) {
    return k == GGML_CUDA_MK_ROPE || k == GGML_CUDA_MK_ROPE_FF || k == GGML_CUDA_MK_FWHT_SR || k == GGML_CUDA_MK_QSA_SEL ||
           k == GGML_CUDA_MK_GATHER || k == GGML_CUDA_MK_FILL || k == GGML_CUDA_MK_SR_Q8 || k == GGML_CUDA_MK_GR_MEAN;
}
static bool mk_class_b(int k) {
    return k == GGML_CUDA_MK_GDN || k == GGML_CUDA_MK_NORM_GATE;
}

// the virtual blocks of 128 threads of a task in the chain kernel
static int mk_vblocks128(const ggml_cuda_mk_task & t) {
    switch (t.kind) {
        case GGML_CUDA_MK_SSM_CONV:  return (int) ((t.p.conv.n_ch + 127) / 128) * t.p.conv.n_s;
        case GGML_CUDA_MK_GDN_GATES: return (int) ((t.p.gates.n + 127) / 128);
        case GGML_CUDA_MK_GDN:       return (int) (t.p.gdn.H * t.p.gdn.n_seqs) * t.p.gdn.gz;
        case GGML_CUDA_MK_NORM_GATE: return t.p.norm_gate.nrows * t.p.norm_gate.nchannels * t.p.norm_gate.nsamples;
        default:                     return 0;
    }
}

// the chain kernel's grid barrier (one chain kernel at a time on the device: the evaluation's stream)
static __device__ unsigned g_mk_bar_count = 0;
static __device__ unsigned g_mk_bar_gen   = 0;

// stages of tasks of class B or C, blocks of 128 threads looping over each task's virtual blocks, a grid barrier
// between two stages (tasks sorted by stage)
__launch_bounds__(128, 1)
static __global__ void k_mk_chain128(const ggml_cuda_mk_tasks T, const int n_stages) {
    unsigned gen = ggml_cuda_grid_barrier_start(&g_mk_bar_gen);
    const int tid = threadIdx.x;
    int i = 0;
    for (int stage = 0; stage < n_stages; stage++) {
        for (; i < T.n && T.t[i].stage == stage; i++) {
            const ggml_cuda_mk_task & t = T.t[i];
            switch (t.kind) {
                case GGML_CUDA_MK_SSM_CONV: {
                    const ssm_conv_cached_args & a = t.p.conv;
                    const int bx = (int) ((a.n_ch + 127) / 128);
                    for (int vb = blockIdx.x; vb < bx * a.n_s; vb += gridDim.x) {
                        ssm_conv_cached_body<4, GGML_CUDA_SSM_CONV_CACHED_MAX_T>(a, vb % bx, vb / bx, tid);
                    }
                } break;
                case GGML_CUDA_MK_GDN_GATES: {
                    const gdn_gates_args & a = t.p.gates;
                    for (int vb = blockIdx.x; (int64_t) vb * 128 < a.n; vb += gridDim.x) {
                        gdn_gates_body(a, (int64_t) vb * 128 + tid);
                    }
                } break;
                case GGML_CUDA_MK_GDN: {
                    const gated_delta_net_args & a = t.p.gdn;
                    const int nb = (int) (a.H * a.n_seqs) * a.gz;
                    for (int vb = blockIdx.x; vb < nb; vb += gridDim.x) {
                        const int h = vb % a.H, rest = vb / a.H;
                        gated_delta_net_body<128, false, false, gdn_cols_per_warp, false>(a, h, rest % a.n_seqs, rest / a.n_seqs,
                                                                                         tid / WARP_SIZE, 4, tid % WARP_SIZE);
                    }
                } break;
                case GGML_CUDA_MK_NORM_GATE: {
                    const rms_norm_gate_args & a = t.p.norm_gate;
                    const int nb = a.nrows * a.nchannels * a.nsamples;
                    for (int vb = blockIdx.x; vb < nb; vb += gridDim.x) {
                        rms_norm_mul_sigmoid_gate_body<128>(a, vb % a.nrows, (vb / a.nrows) % a.nchannels,
                                                            vb / (a.nrows * a.nchannels), tid);
                    }
                } break;
            }
        }
        if (stage + 1 < n_stages) {
            ggml_cuda_grid_barrier(&g_mk_bar_count, &g_mk_bar_gen, gen);
        }
    }
}

static void mk_launch_chain128(const ggml_cuda_mk_tasks & T, int n_stages, cudaStream_t stream) {
    static int cap = 0;
    if (cap == 0) {
        int per_cu = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_mk_chain128, 128, 32 * sizeof(float)));
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        // every block resident at once (else the barrier waits for ever): at most what the device holds, and under
        // the waves it dispatches whatever other queues hold (grid-barrier.cuh)
        cap = std::max(1, std::min(std::min(per_cu, 4) * nsm, ggml_cuda_grid_barrier_max_blocks(128)));
    }
    int need = 1;
    for (int i = 0; i < T.n; i++) {
        need = std::max(need, mk_vblocks128(T.t[i]));
    }
    k_mk_chain128<<<std::min(need, cap), 128, 32 * sizeof(float), stream>>>(T, n_stages);
    CUDA_CHECK(cudaGetLastError());
}

// the chain kernel of blocks of 256 threads: stages of the indexer key's tasks (mk_class_chain256), each looping over
// its virtual blocks, a grid barrier between two stages (its own pair: it never runs at the same time as k_mk_chain128)
static __device__ __attribute__((aligned(256))) unsigned g_mk_bar256[128];   // count [0], generation [64]

// the chain kernel's grid (at most: a few blocks, all resident at once whatever runs on other queues, grid-barrier.cuh;
// at the full occupancy, 7 x 42 blocks, it waited for ever in a prefill), and the largest task it takes (a few loops
// of its blocks)
#define GGML_CUDA_MK_CHAIN256_GRID       64
#define GGML_CUDA_MK_CHAIN256_MAX_BLOCKS 256

__launch_bounds__(GGML_CUDA_MK_THREADS, 1)
static __global__ void k_mk_chain256(const ggml_cuda_mk_tasks T, const int n_stages) {
    unsigned gen = ggml_cuda_grid_barrier_start(&g_mk_bar256[64]);
    const int tid = threadIdx.x;
    int i = 0;
    for (int stage = 0; stage < n_stages; stage++) {
        for (; i < T.n && T.t[i].stage == stage; i++) {
            const ggml_cuda_mk_task & t = T.t[i];
            for (int vb = blockIdx.x; vb < t.n_blocks; vb += gridDim.x) {
                switch (t.kind) {
                    case GGML_CUDA_MK_SR_Q8:
                        set_rows_q8_body(t.p.sr_q8, (int64_t) vb * GGML_CUDA_MK_THREADS + tid);
                        break;
                    case GGML_CUDA_MK_GR_MEAN:
                        get_rows_mean_body(t.p.gr_mean, (int64_t) vb * GGML_CUDA_MK_THREADS + tid);
                        break;
                    case GGML_CUDA_MK_FILL: {
                        const int64_t j = (int64_t) vb * GGML_CUDA_MK_THREADS + tid;
                        if (j < t.p.fill.n) {
                            t.p.fill.dst[j] = t.p.fill.value;
                        }
                    } break;
                    case GGML_CUDA_MK_QSA_SEL: {
                        const qsa_select_args & a = t.p.sel;
                        const int bx = (int) ((a.n_sel + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS);
                        const int64_t j = (int64_t) (vb % bx) * GGML_CUDA_MK_THREADS + tid;
                        const int64_t q = vb / bx;
#define MK_SEL(M) (a.by_pos ? k_qsa_select_body<M, true>(a, j, q) : k_qsa_select_body<M, false>(a, j, q))
                        switch (a.mode) {
                            case GGML_QSA_SELECT_ROWS: MK_SEL(GGML_QSA_SELECT_ROWS); break;
                            case GGML_QSA_SELECT_MASK: MK_SEL(GGML_QSA_SELECT_MASK); break;
                            default:                   MK_SEL(GGML_QSA_SELECT_CAUSAL); break;
                        }
#undef MK_SEL
                    } break;
                    case GGML_CUDA_MK_ROPE: {
                        const rms_norm_mul_rope_args & a = t.p.rope;
                        rms_norm_mul_rope_f32_body<GGML_CUDA_MK_THREADS, false, float>(a, vb % a.n_rows,
                            (vb / a.n_rows) % a.n_channels, vb / (a.n_rows * a.n_channels));
                        __syncthreads();   // its shared memory, before the next virtual block's
                    } break;
                }
            }
        }
        if (stage + 1 < n_stages) {
            ggml_cuda_grid_barrier(&g_mk_bar256[0], &g_mk_bar256[64], gen);
        }
    }
}

static void mk_launch_chain256(const ggml_cuda_mk_tasks & T, int n_stages, cudaStream_t stream) {
    const size_t smem = (32 + 256) * sizeof(float);
    static int cap = 0;
    if (cap == 0) {
        int per_cu = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, k_mk_chain256, GGML_CUDA_MK_THREADS, smem));
        cap = std::max(1, per_cu * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm);
    }
    int need = 1;
    for (int i = 0; i < T.n; i++) {
        need = std::max(need, T.t[i].n_blocks);
    }
    k_mk_chain256<<<std::min(std::min(need, cap), std::min(GGML_CUDA_MK_CHAIN256_GRID, ggml_cuda_grid_barrier_max_blocks(GGML_CUDA_MK_THREADS))),
        GGML_CUDA_MK_THREADS, smem, stream>>>(T, n_stages);
    CUDA_CHECK(cudaGetLastError());
}

static bool mk_class_chain256(int k);

static bool mk_add(ggml_cuda_mk * mk, int kind, int n_blocks, size_t smem, const ggml_cuda_mk_payload & p) {
    if (mk->tasks.n >= GGML_CUDA_MK_MAX_TASKS || n_blocks <= 0) {
        return false;
    }
    for (int i = 0; i < mk->tasks.n; i++) {   // the classes do not mix (a kernel of 256 or of 128 threads)
        if ((mk_class_a(kind) && mk_class_b(mk->tasks.t[i].kind)) || (mk_class_b(kind) && mk_class_a(mk->tasks.t[i].kind))) {
            return false;
        }
    }
    // stages run in a chain kernel, which has the bodies of its own kinds only (k_mk_chain256 or k_mk_chain128)
    if (mk->n_stages > 1 && mk->tasks.n > 0 &&
            (mk_class_chain256(kind) != mk_class_chain256(mk->tasks.t[0].kind) || (!mk_class_chain256(kind) && mk_class_a(kind)))) {
        return false;
    }
    ggml_cuda_mk_task & t = mk->tasks.t[mk->tasks.n++];
    t.kind        = kind;
    t.stage       = mk->n_stages - 1;
    t.first_block = mk->n_blocks;
    t.n_blocks    = n_blocks;
    t.p           = p;
    mk->n_blocks += n_blocks;
    mk->smem      = std::max(mk->smem, smem);
    return true;
}

bool ggml_cuda_mk_add_rope(ggml_cuda_mk * mk, const rms_norm_mul_rope_args & a, bool has_ff) {
    ggml_cuda_mk_payload p;
    p.rope = a;
    return mk_add(mk, has_ff ? GGML_CUDA_MK_ROPE_FF : GGML_CUDA_MK_ROPE, a.n_rows * a.n_channels * a.n_samples,
                  (32 + 256) * sizeof(float), p);
}

bool ggml_cuda_mk_add_fwht_set_rows(ggml_cuda_mk * mk, const fwht_set_rows_args & a) {
    ggml_cuda_mk_payload p;
    p.fwht = a;
    const int rows_per_block = GGML_CUDA_MK_THREADS / WARP_SIZE;
    return mk_add(mk, GGML_CUDA_MK_FWHT_SR, (int) ((a.n_rows + rows_per_block - 1) / rows_per_block), 0, p);
}

bool ggml_cuda_mk_add_qsa_select(ggml_cuda_mk * mk, const qsa_select_args & a) {
    if (a.mode == GGML_QSA_SELECT_DENSE) {
        return false;
    }
    ggml_cuda_mk_payload p;
    p.sel = a;
    return mk_add(mk, GGML_CUDA_MK_QSA_SEL, (int) ((a.n_sel + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS) * a.n_t, 0, p);
}

bool ggml_cuda_mk_add_ssm_conv(ggml_cuda_mk * mk, const ssm_conv_cached_args & a) {
    ggml_cuda_mk_payload p;
    p.conv = a;
    return mk_add(mk, GGML_CUDA_MK_SSM_CONV, (int) ((a.n_ch + 127) / 128) * a.n_s, 0, p);
}

bool ggml_cuda_mk_add_gdn_gates(ggml_cuda_mk * mk, const gdn_gates_args & a) {
    ggml_cuda_mk_payload p;
    p.gates = a;
    return mk_add(mk, GGML_CUDA_MK_GDN_GATES, (int) ((a.n + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS), 0, p);
}

bool ggml_cuda_mk_add_gdn(ggml_cuda_mk * mk, const gated_delta_net_args & a) {
    ggml_cuda_mk_payload p;
    p.gdn = a;
    return mk_add(mk, GGML_CUDA_MK_GDN, (int) (a.H * a.n_seqs) * a.gz, 0, p);
}

bool ggml_cuda_mk_add_norm_gate(ggml_cuda_mk * mk, const rms_norm_gate_args & a) {
    ggml_cuda_mk_payload p;
    p.norm_gate = a;
    return mk_add(mk, GGML_CUDA_MK_NORM_GATE, a.nrows * a.nchannels * a.nsamples, 32 * sizeof(float), p);
}

bool ggml_cuda_mk_add_gather(ggml_cuda_mk * mk, const get_rows_q_args & a) {
    if ((a.type != GGML_TYPE_Q8_0 && a.type != GGML_TYPE_Q4_0) || a.ne00 % 32 != 0 || (uintptr_t) a.dst % 16 != 0 ||
            a.s1 % 8 != 0 || a.s2 % 8 != 0 || a.s3 % 8 != 0 || (uintptr_t) a.src0 % 2 != 0 || a.nb01 % 2 != 0 ||
            a.nb02 % 2 != 0 || a.nb03 % 2 != 0) {
        return false;
    }
    ggml_cuda_mk_payload p;
    p.gather = a;
    return mk_add(mk, GGML_CUDA_MK_GATHER, (a.n_rows + GET_ROWS_Q_ROWS_PER_BLOCK - 1) / GET_ROWS_Q_ROWS_PER_BLOCK, 0, p);
}

bool ggml_cuda_mk_add_fill(ggml_cuda_mk * mk, const mk_fill_args & a) {
    ggml_cuda_mk_payload p;
    p.fill = a;
    return mk_add(mk, GGML_CUDA_MK_FILL, (int) ((a.n + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS), 0, p);
}

bool ggml_cuda_mk_add_set_rows_q8(ggml_cuda_mk * mk, const set_rows_q8_args & a) {
    ggml_cuda_mk_payload p;
    p.sr_q8 = a;
    return mk_add(mk, GGML_CUDA_MK_SR_Q8, (int) ((a.ne_total + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS), 0, p);
}

bool ggml_cuda_mk_add_get_rows_mean(ggml_cuda_mk * mk, const get_rows_mean_args & a) {
    ggml_cuda_mk_payload p;
    p.gr_mean = a;
    const int64_t n = a.n_out * (a.ne00 / 2);
    return mk_add(mk, GGML_CUDA_MK_GR_MEAN, (int) ((n + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS), 0, p);
}

// the kinds of k_mk_chain256
static bool mk_class_chain256(int k) {
    return k == GGML_CUDA_MK_SR_Q8 || k == GGML_CUDA_MK_GR_MEAN || k == GGML_CUDA_MK_ROPE || k == GGML_CUDA_MK_FILL ||
           k == GGML_CUDA_MK_QSA_SEL;
}

bool ggml_cuda_mk_chainable256(const ggml_cuda_mk * mk) {
    if (mk->n_stages >= GGML_CUDA_MK_MAX_STAGES || mk->tasks.n == 0 || mk->tasks.n >= GGML_CUDA_MK_MAX_TASKS) {
        return false;
    }
    // small tasks only (the generation's): a few loops of the chain kernel's grid
    for (int i = 0; i < mk->tasks.n; i++) {
        if (!mk_class_chain256(mk->tasks.t[i].kind) || mk->tasks.t[i].n_blocks > GGML_CUDA_MK_CHAIN256_MAX_BLOCKS) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_mk_chainable(const ggml_cuda_mk * mk) {
    if (mk->n_stages >= GGML_CUDA_MK_MAX_STAGES) {
        return false;
    }
    for (int i = 0; i < mk->tasks.n; i++) {
        if (mk_class_a(mk->tasks.t[i].kind)) {
            return false;
        }
        // the convolution's threads past its channels leave before its block barriers: whole blocks only
        if (mk->tasks.t[i].kind == GGML_CUDA_MK_SSM_CONV && mk->tasks.t[i].p.conv.n_ch % 128 != 0) {
            return false;
        }
    }
    return true;
}

static ggml_cuda_mk::range mk_range_of(const ggml_tensor * t) {
    const char * lo = (const char *) t->data;
    return { lo, lo + ggml_nbytes(t) };
}

// a task alone: the original kernel's configuration (block size, registers), not the multi-task kernel's
template <bool has_ff>
static __global__ void k_mk_rope(const rms_norm_mul_rope_args a) {
    rms_norm_mul_rope_f32_body<256, has_ff, float>(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

template <typename B>
__launch_bounds__(4*WARP_SIZE, 1)
static __global__ void k_mk_fwht_set_rows(const fwht_set_rows_args a) {
    fwht_set_rows_body<64, B>(a, (int64_t) blockIdx.x * blockDim.y + threadIdx.y, threadIdx.x);
}

template <int mode, bool by_pos>
static __global__ void k_mk_qsa_select(const qsa_select_args a) {
    k_qsa_select_body<mode, by_pos>(a, (int64_t) blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y);
}

static __global__ void k_mk_ssm_conv(const ssm_conv_cached_args a) {
    ssm_conv_cached_body<4, GGML_CUDA_SSM_CONV_CACHED_MAX_T>(a, blockIdx.x, blockIdx.y, threadIdx.x);
}

static __global__ void k_mk_gdn_gates(const gdn_gates_args a) {
    gdn_gates_body(a, (int64_t) blockIdx.x * blockDim.x + threadIdx.x);
}

static __global__ void k_mk_gdn1(const gated_delta_net_args a) {
    gated_delta_net_body<128, false, false, gdn_cols_per_warp, false>(a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.y,
                                                                     blockDim.y, threadIdx.x);
}

static __global__ void k_mk_sr_q8(const set_rows_q8_args a) {
    set_rows_q8_body(a, (int64_t) blockIdx.x * blockDim.x + threadIdx.x);
}

static __global__ void k_mk_gr_mean(const get_rows_mean_args a) {
    get_rows_mean_body(a, (int64_t) blockIdx.x * blockDim.x + threadIdx.x);
}

static __global__ void k_mk_fill1(const mk_fill_args a) {
    const int64_t j = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (j < a.n) {
        a.dst[j] = a.value;
    }
}

static __global__ void k_mk_gather1(const get_rows_q_args a) {
    get_rows_q_body(a, blockIdx.x, threadIdx.x);
}

static __global__ void k_mk_norm_gate1(const rms_norm_gate_args a) {
    rms_norm_mul_sigmoid_gate_body<128>(a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);
}

static void mk_launch_single(const ggml_cuda_mk_task & t, cudaStream_t stream) {
    switch (t.kind) {
        case GGML_CUDA_MK_GDN: {
            const gated_delta_net_args & a = t.p.gdn;
            k_mk_gdn1<<<dim3(a.H, a.n_seqs, a.gz), dim3(WARP_SIZE, a.nwarps, 1), 0, stream>>>(a);
        } break;
        case GGML_CUDA_MK_NORM_GATE: {
            const rms_norm_gate_args & a = t.p.norm_gate;
            k_mk_norm_gate1<<<dim3(a.nrows, a.nchannels, a.nsamples), 128, 32 * sizeof(float), stream>>>(a);
        } break;
        case GGML_CUDA_MK_ROPE:
        case GGML_CUDA_MK_ROPE_FF: {
            const rms_norm_mul_rope_args & a = t.p.rope;
            const dim3 grid(a.n_rows, a.n_channels, a.n_samples);
            const size_t smem = (32 + 256) * sizeof(float);
            if (t.kind == GGML_CUDA_MK_ROPE) {
                k_mk_rope<false><<<grid, 256, smem, stream>>>(a);
            } else {
                k_mk_rope<true><<<grid, 256, smem, stream>>>(a);
            }
        } break;
        case GGML_CUDA_MK_FWHT_SR: {
            const fwht_set_rows_args & a = t.p.fwht;
            const dim3 grid((a.n_rows + 3) / 4, 1, 1), block(WARP_SIZE, 4, 1);
            if (a.q4_0) {
                k_mk_fwht_set_rows<block_q4_0><<<grid, block, 0, stream>>>(a);
            } else {
                k_mk_fwht_set_rows<block_q8_0><<<grid, block, 0, stream>>>(a);
            }
        } break;
        case GGML_CUDA_MK_QSA_SEL: {
            const qsa_select_args & a = t.p.sel;
            const dim3 grid((a.n_sel + 255) / 256, a.n_t, 1);
#define MK_SEL1(M) (a.by_pos ? k_mk_qsa_select<M, true><<<grid, 256, 0, stream>>>(a) : k_mk_qsa_select<M, false><<<grid, 256, 0, stream>>>(a))
            switch (a.mode) {
                case GGML_QSA_SELECT_ROWS: MK_SEL1(GGML_QSA_SELECT_ROWS); break;
                case GGML_QSA_SELECT_MASK: MK_SEL1(GGML_QSA_SELECT_MASK); break;
                default:                   MK_SEL1(GGML_QSA_SELECT_CAUSAL); break;
            }
#undef MK_SEL1
        } break;
        case GGML_CUDA_MK_SSM_CONV: {
            const ssm_conv_cached_args & a = t.p.conv;
            k_mk_ssm_conv<<<dim3((a.n_ch + 127) / 128, a.n_s, 1), 128, 0, stream>>>(a);
        } break;
        case GGML_CUDA_MK_GDN_GATES: {
            const gdn_gates_args & a = t.p.gates;
            k_mk_gdn_gates<<<(a.n + 255) / 256, 256, 0, stream>>>(a);
        } break;
        case GGML_CUDA_MK_FILL: {
            const mk_fill_args & a = t.p.fill;
            k_mk_fill1<<<(int) ((a.n + 255) / 256), 256, 0, stream>>>(a);
        } break;
        case GGML_CUDA_MK_SR_Q8:
            k_mk_sr_q8<<<t.n_blocks, 256, 0, stream>>>(t.p.sr_q8);
            break;
        case GGML_CUDA_MK_GR_MEAN:
            k_mk_gr_mean<<<t.n_blocks, 256, 0, stream>>>(t.p.gr_mean);
            break;
        case GGML_CUDA_MK_GATHER: {
            const get_rows_q_args & a = t.p.gather;
            k_mk_gather1<<<(a.n_rows + GET_ROWS_Q_ROWS_PER_BLOCK - 1) / GET_ROWS_Q_ROWS_PER_BLOCK, 256, 0, stream>>>(a);
        } break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// the gated delta net's convolution and gates (the two independent kernels of a linear attention layer), specialised:
// blocks of 128 threads, the convolution's (its own size) first, then the gates'
static __global__ void k_mk_conv_gates(const ssm_conv_cached_args c, const gdn_gates_args g, const int conv_bx, const int conv_blocks) {
    const int b = blockIdx.x;
    if (b < conv_blocks) {
        ssm_conv_cached_body<4, GGML_CUDA_SSM_CONV_CACHED_MAX_T>(c, b % conv_bx, b / conv_bx, threadIdx.x);
    } else {
        gdn_gates_body(g, (int64_t) (b - conv_blocks) * 128 + threadIdx.x);
    }
}

// the attention layer's independent kernels before the flash attention (Q and K rope, V rotated into its cache, the
// selection of the attended cells, the indexer's weights), specialised: blocks of 256 threads, at most 4 tasks, the selection's mode fixed
struct ggml_cuda_mk_tasks4 {
    int n;
    ggml_cuda_mk_task t[4];
};

template <int sel_mode, bool sel_by_pos>
__launch_bounds__(GGML_CUDA_MK_THREADS, 1)
static __global__ void k_mk_qsa(const ggml_cuda_mk_tasks4 T) {
    int i = 0;
    while (i + 1 < T.n && (int) blockIdx.x >= T.t[i + 1].first_block) {
        i++;
    }
    const ggml_cuda_mk_task & t = T.t[i];
    const int vb = blockIdx.x - t.first_block;
    if (t.kind == GGML_CUDA_MK_ROPE) {
        const rms_norm_mul_rope_args & a = t.p.rope;
        rms_norm_mul_rope_f32_body<GGML_CUDA_MK_THREADS, false, float>(a, vb % a.n_rows, (vb / a.n_rows) % a.n_channels,
                                                                      vb / (a.n_rows * a.n_channels));
    } else if (t.kind == GGML_CUDA_MK_FWHT_SR) {
        const fwht_set_rows_args & a = t.p.fwht;
        const int64_t r = (int64_t) vb * (GGML_CUDA_MK_THREADS / WARP_SIZE) + threadIdx.x / WARP_SIZE;
        if (a.q4_0) {
            fwht_set_rows_body<64, block_q4_0>(a, r, threadIdx.x % WARP_SIZE);
        } else {
            fwht_set_rows_body<64, block_q8_0>(a, r, threadIdx.x % WARP_SIZE);
        }
    } else if (t.kind == GGML_CUDA_MK_FILL) {
        const int64_t j = (int64_t) vb * GGML_CUDA_MK_THREADS + threadIdx.x;
        if (j < t.p.fill.n) {
            t.p.fill.dst[j] = t.p.fill.value;
        }
    } else {
        const qsa_select_args & a = t.p.sel;
        const int bx = (int) ((a.n_sel + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS);
        k_qsa_select_body<sel_mode, sel_by_pos>(a, (int64_t) (vb % bx) * GGML_CUDA_MK_THREADS + threadIdx.x, vb / bx);
    }
}

// the attention's gathers of the selected K and V cells and the selection's mask (independent of each other, before the
// flash attention), specialised: blocks of 256 threads, at most 4 tasks, the selection's mode fixed
template <int sel_mode, bool sel_by_pos>
__launch_bounds__(GGML_CUDA_MK_THREADS, 1)
static __global__ void k_mk_gather(const ggml_cuda_mk_tasks4 T) {
    int i = 0;
    while (i + 1 < T.n && (int) blockIdx.x >= T.t[i + 1].first_block) {
        i++;
    }
    const ggml_cuda_mk_task & t = T.t[i];
    const int vb = blockIdx.x - t.first_block;
    if (t.kind == GGML_CUDA_MK_GATHER) {
        get_rows_q_body(t.p.gather, vb, threadIdx.x);
    } else {
        const qsa_select_args & a = t.p.sel;
        const int bx = (int) ((a.n_sel + GGML_CUDA_MK_THREADS - 1) / GGML_CUDA_MK_THREADS);
        k_qsa_select_body<sel_mode, sel_by_pos>(a, (int64_t) (vb % bx) * GGML_CUDA_MK_THREADS + threadIdx.x, vb / bx);
    }
}

// a specialised kernel for the pending tasks, if there is one for them
static bool mk_launch_special(const ggml_cuda_mk_tasks & T, int n, cudaStream_t stream) {
    // the attention's gathers and at most one selection (not the causal one)
    if (n <= 4) {
        int n_sel = 0, mode = -1;
        bool by_pos = false, ok = true;
        for (int i = 0; i < n && ok; i++) {
            const int k = T.t[i].kind;
            ok = k == GGML_CUDA_MK_GATHER || k == GGML_CUDA_MK_QSA_SEL;
            if (k == GGML_CUDA_MK_QSA_SEL) {
                n_sel++;
                mode = T.t[i].p.sel.mode;
                by_pos = T.t[i].p.sel.by_pos;
            }
        }
        if (ok && n_sel <= 1 && n_sel < n && (n_sel == 0 || mode == GGML_QSA_SELECT_ROWS || mode == GGML_QSA_SELECT_MASK)) {
            ggml_cuda_mk_tasks4 t4;
            t4.n = n;
            for (int i = 0; i < n; i++) {
                t4.t[i] = T.t[i];
            }
            const int n_blocks = T.t[n - 1].first_block + T.t[n - 1].n_blocks;
#define MK_GATHER(M, B) k_mk_gather<M, B><<<n_blocks, GGML_CUDA_MK_THREADS, 0, stream>>>(t4)
            if (mode == GGML_QSA_SELECT_ROWS) {
                if (by_pos) MK_GATHER(GGML_QSA_SELECT_ROWS, true); else MK_GATHER(GGML_QSA_SELECT_ROWS, false);
            } else {
                if (by_pos) MK_GATHER(GGML_QSA_SELECT_MASK, true); else MK_GATHER(GGML_QSA_SELECT_MASK, false);
            }
#undef MK_GATHER
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
    }
    // the attention layer's: rope (without frequency factors), rotation into a cache, at most one selection
    if (n <= 4) {
        int n_sel = 0, mode = -1;
        bool by_pos = false, ok = true;
        for (int i = 0; i < n && ok; i++) {
            const int k = T.t[i].kind;
            ok = k == GGML_CUDA_MK_ROPE || k == GGML_CUDA_MK_FWHT_SR || k == GGML_CUDA_MK_QSA_SEL || k == GGML_CUDA_MK_FILL;
            if (k == GGML_CUDA_MK_QSA_SEL) {
                n_sel++;
                mode = T.t[i].p.sel.mode;
                by_pos = T.t[i].p.sel.by_pos;
            }
        }
        if (ok && n_sel <= 1 && (n_sel == 0 || mode == GGML_QSA_SELECT_CAUSAL || mode == GGML_QSA_SELECT_ROWS ||
                                 mode == GGML_QSA_SELECT_MASK)) {
            ggml_cuda_mk_tasks4 t4;
            t4.n = n;
            for (int i = 0; i < n; i++) {
                t4.t[i] = T.t[i];
            }
            const int n_blocks = T.t[n - 1].first_block + T.t[n - 1].n_blocks;
            size_t smem = 0;
            for (int i = 0; i < n; i++) {
                smem = std::max(smem, T.t[i].kind == GGML_CUDA_MK_ROPE ? (32 + 256) * sizeof(float) : (size_t) 0);
            }
#define MK_QSA(M, B) k_mk_qsa<M, B><<<n_blocks, GGML_CUDA_MK_THREADS, smem, stream>>>(t4)
            if (mode == GGML_QSA_SELECT_ROWS) {
                if (by_pos) MK_QSA(GGML_QSA_SELECT_ROWS, true); else MK_QSA(GGML_QSA_SELECT_ROWS, false);
            } else if (mode == GGML_QSA_SELECT_MASK) {
                if (by_pos) MK_QSA(GGML_QSA_SELECT_MASK, true); else MK_QSA(GGML_QSA_SELECT_MASK, false);
            } else {
                if (by_pos) MK_QSA(GGML_QSA_SELECT_CAUSAL, true); else MK_QSA(GGML_QSA_SELECT_CAUSAL, false);
            }
#undef MK_QSA
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
    }
    if (n != 2) {
        return false;
    }
    const ggml_cuda_mk_task * conv = nullptr, * gates = nullptr;
    for (int i = 0; i < 2; i++) {
        if (T.t[i].kind == GGML_CUDA_MK_SSM_CONV) conv = &T.t[i];
        if (T.t[i].kind == GGML_CUDA_MK_GDN_GATES) gates = &T.t[i];
    }
    if (!conv || !gates) {
        return false;
    }
    const ssm_conv_cached_args & c = conv->p.conv;
    const gdn_gates_args & g = gates->p.gates;
    const int conv_bx = (int) ((c.n_ch + 127) / 128);
    const int conv_blocks = conv_bx * c.n_s;
    const int gate_blocks = (int) ((g.n + 127) / 128);
    k_mk_conv_gates<<<conv_blocks + gate_blocks, 128, 0, stream>>>(c, g, conv_bx, conv_blocks);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// tasks launched: one alone with its own kernel; stages, or tasks of class B, by the chain kernel; else a specialised
// kernel if there is one, else the generic one
static void mk_launch(const ggml_cuda_mk_tasks & T, int n_blocks, size_t smem, cudaStream_t stream) {
    int n_stages = 1;
    bool chain = false;
    for (int i = 0; i < T.n; i++) {
        n_stages = std::max(n_stages, T.t[i].stage + 1);
        chain = chain || mk_class_b(T.t[i].kind);
    }
    if (T.n == 1) {
        mk_launch_single(T.t[0], stream);
    } else if (n_stages > 1 && mk_class_chain256(T.t[0].kind)) {
        mk_launch_chain256(T, n_stages, stream);
    } else if (n_stages > 1 || chain) {
        mk_launch_chain128(T, n_stages, stream);
    } else if (!mk_launch_special(T, T.n, stream)) {
        k_mk<<<n_blocks, GGML_CUDA_MK_THREADS, smem, stream>>>(T);
        CUDA_CHECK(cudaGetLastError());
    }
}

// the first n pending tasks launched as one kernel, the others kept (their blocks renumbered, a first stage)
static void mk_flush_first(ggml_cuda_mk * mk, int n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    ggml_cuda_mk_tasks first = mk->tasks;
    first.n = n;
    const int n_blocks = n < mk->tasks.n ? mk->tasks.t[n].first_block : mk->n_blocks;
    mk_launch(first, n_blocks, mk->smem, stream);
    mk->launches++;
    mk->merged += n;
    int k = 0;
    for (int i = n; i < mk->tasks.n; i++, k++) {
        mk->tasks.t[k] = mk->tasks.t[i];
        mk->tasks.t[k].first_block -= n_blocks;
        mk->tasks.t[k].stage = 0;
    }
    mk->tasks.n   = k;
    mk->n_blocks -= n_blocks;
    mk->reads.clear();
    mk->writes.clear();
    mk->stage_reads.clear();
    mk->stage_writes.clear();
    mk->n_stages = 1;
    mk->ranged = 0;
}

static bool mk_node_conflicts(const ggml_cuda_mk * mk, const ggml_tensor * node);

static bool mk_is_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// the nodes of a unit (indices into the graph) that it writes for others: the ones read after it, marked as outputs,
// or writing into another tensor (a view: a cache row, an in-place result); the others are the intermediate results
// of a fused unit, never written
static bool mk_writes_for_others(const ggml_cuda_mk * mk, const ggml_cgraph * cgraph, const int * idx, int n, int k) {
    const ggml_tensor * t = cgraph->nodes[idx[k]];
    if (mk_is_noop(t) || !t->data) {
        return false;
    }
    if (t->view_src || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return true;
    }
    int last = idx[0];
    for (int u = 0; u < n; u++) {
        last = std::max(last, idx[u]);
    }
    const auto it = mk->last_read.find(t);
    return it != mk->last_read.end() && it->second > last;
}

void ggml_cuda_mk_begin_graph(ggml_cuda_mk * mk, const ggml_cgraph * cgraph) {
    mk->last_read.clear();
    for (int m = 0; m < cgraph->n_nodes; m++) {
        const ggml_tensor * c = cgraph->nodes[m];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            for (const ggml_tensor * s = c->src[j]; s; s = s->view_src) {
                mk->last_read[s] = m;
            }
        }
    }
}

// a unit (indices into the graph) conflicts with the pending tasks: what it writes for others against what they read
// and write, what it reads from outside itself against what they write
static bool mk_unit_conflicts_with(const ggml_cuda_mk * mk, const ggml_cgraph * cgraph, const int * idx, int n,
        const std::vector<ggml_cuda_mk::range> & reads, const std::vector<ggml_cuda_mk::range> & writes);

bool ggml_cuda_mk_unit_conflicts(const ggml_cuda_mk * mk, const ggml_cgraph * cgraph, const int * idx, int n) {
    if (mk->ranged == 0) {
        return false;
    }
    return mk_unit_conflicts_with(mk, cgraph, idx, n, mk->reads, mk->writes);
}

static bool mk_unit_conflicts_with(const ggml_cuda_mk * mk, const ggml_cgraph * cgraph, const int * idx, int n,
        const std::vector<ggml_cuda_mk::range> & reads, const std::vector<ggml_cuda_mk::range> & writes) {
    const auto overlaps = [](const ggml_cuda_mk::range & a, const ggml_cuda_mk::range & b) {
        return a.lo < b.hi && b.lo < a.hi;
    };
    const auto inside = [&](const ggml_tensor * t) {
        for (; t; t = t->view_src) {
            for (int k = 0; k < n; k++) {
                if (cgraph->nodes[idx[k]] == t) {
                    return true;
                }
            }
        }
        return false;
    };
    for (int k = 0; k < n; k++) {
        if (mk_writes_for_others(mk, cgraph, idx, n, k)) {
            const ggml_cuda_mk::range w = mk_range_of(cgraph->nodes[idx[k]]);
            for (const auto & r : reads) {
                if (overlaps(w, r)) {
                    return true;
                }
            }
            for (const auto & r : writes) {
                if (overlaps(w, r)) {
                    return true;
                }
            }
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = cgraph->nodes[idx[k]]->src[j];
            if (src && src->data && !inside(src)) {
                const ggml_cuda_mk::range r = mk_range_of(src);
                for (const auto & w : writes) {
                    if (overlaps(r, w)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool ggml_cuda_mk_unit_done(ggml_cuda_mk * mk, const ggml_cgraph * cgraph, int first, int last, cudaStream_t stream) {

    const bool recorded = mk->ranged < mk->tasks.n;
    // the unit's nodes against the tasks pending before it (the first was checked before the unit ran)
    std::vector<int> idx;
    for (int k = first; k <= last && k < cgraph->n_nodes; k++) {
        idx.push_back(k);
    }
    bool conflict = false;
    if (recorded) {
        conflict = ggml_cuda_mk_unit_conflicts(mk, cgraph, idx.data(), (int) idx.size());
    } else {
        for (int k = first; k <= last && k < cgraph->n_nodes && !conflict; k++) {
            conflict = !mk_is_noop(cgraph->nodes[k]) && mk_node_conflicts(mk, cgraph->nodes[k]);
        }
    }
    bool new_stage = false;
    if (conflict) {
        if (!recorded) {
            return false;   // launched before the tasks it depends on
        }
        if (ggml_cuda_mk_chainable(mk) || ggml_cuda_mk_chainable256(mk)) {
            // after a grid barrier: a new stage if it depends on the last one, else in it (the barrier before the last
            // stage orders it after the earlier ones)
            new_stage = mk_unit_conflicts_with(mk, cgraph, idx.data(), (int) idx.size(), mk->stage_reads, mk->stage_writes);
        } else {
            mk_flush_first(mk, mk->ranged, stream);   // the earlier tasks first, the unit's own stay pending
        }
    }
    if (!recorded) {
        return true;
    }
    if (new_stage) {
        for (int i = mk->ranged; i < mk->tasks.n; i++) {
            mk->tasks.t[i].stage = mk->n_stages;
        }
        mk->n_stages++;
        mk->stage_reads.clear();
        mk->stage_writes.clear();
    }
    // what the unit writes for others, and what it reads from outside itself
    const auto inside = [&](const ggml_tensor * t) {
        for (; t; t = t->view_src) {
            for (int k = first; k <= last && k < cgraph->n_nodes; k++) {
                if (cgraph->nodes[k] == t) {
                    return true;
                }
            }
        }
        return false;
    };
    for (int k = 0; k < (int) idx.size(); k++) {
        if (mk_writes_for_others(mk, cgraph, idx.data(), (int) idx.size(), k)) {
            mk->writes.push_back(mk_range_of(cgraph->nodes[idx[k]]));
            mk->stage_writes.push_back(mk_range_of(cgraph->nodes[idx[k]]));
        }
        const ggml_tensor * n = cgraph->nodes[idx[k]];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * src = n->src[j];
            if (src && src->data && !inside(src)) {
                mk->reads.push_back(mk_range_of(src));
                mk->stage_reads.push_back(mk_range_of(src));
            }
        }
    }
    mk->ranged = mk->tasks.n;
    return true;
}

bool ggml_cuda_mk_conflicts(const ggml_cuda_mk * mk, const ggml_tensor * node) {
    return mk_node_conflicts(mk, node);
}

static bool mk_node_conflicts(const ggml_cuda_mk * mk, const ggml_tensor * node) {
    if (mk->ranged == 0) {
        return false;
    }
    const auto overlaps = [](const ggml_cuda_mk::range & a, const ggml_cuda_mk::range & b) {
        return a.lo < b.hi && b.lo < a.hi;
    };
    if (node->data) {
        const ggml_cuda_mk::range w = mk_range_of(node);
        for (const auto & r : mk->reads) {
            if (overlaps(w, r)) {
                return true;
            }
        }
        for (const auto & r : mk->writes) {
            if (overlaps(w, r)) {
                return true;
            }
        }
    }
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        const ggml_tensor * s = node->src[j];
        if (s && s->data) {
            const ggml_cuda_mk::range r = mk_range_of(s);
            for (const auto & w : mk->writes) {
                if (overlaps(r, w)) {
                    return true;
                }
            }
        }
    }
    return false;
}

void ggml_cuda_mk_flush(ggml_cuda_mk * mk, cudaStream_t stream) {
    if (mk->tasks.n == 0) {
        return;
    }
    mk_launch(mk->tasks, mk->n_blocks, mk->smem, stream);
    mk->launches++;
    mk->merged += mk->tasks.n;
    mk->tasks.n  = 0;
    mk->n_blocks = 0;
    mk->smem     = 0;
    mk->ranged   = 0;
    mk->reads.clear();
    mk->writes.clear();
    mk->stage_reads.clear();
    mk->stage_writes.clear();
    mk->n_stages = 1;
}
