// Source attribution and MIT license: docs/third-party/sm80-gdn.txt.
#include "gdn-sm80-chunked.cuh"

#include <algorithm>
#include <climits>
#include <mutex>

bool ggml_cuda_gdn_sm80_chunked_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_GDN_SM80_CHUNKED");
        return value && strcmp(value, "1") == 0;
    }();
    return enabled;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
namespace {
constexpr int S = 128, HK = 16, HV = 48, VPK = HV / HK;
constexpr int GCH = 32, GSB = 2048, GTP = GCH + 1;

// Async input copies; pre-Ampere targets use a compile-only fallback.
__device__ __forceinline__ void gdn_cp4(float * smem, const float * gmem) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *smem = *gmem;
#endif
}
__device__ __forceinline__ void gdn_cp16(float * smem, const float * gmem) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((unsigned) __cvta_generic_to_shared(smem)), "l"(gmem));
#else
    *reinterpret_cast<float4 *>(smem) = *reinterpret_cast<const float4 *>(gmem);
#endif
}
__device__ __forceinline__ void gdn_cp_commit() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
}
__device__ __forceinline__ void gdn_cp_wait_all() {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;");
#endif
}

#include "gdn-sm80-chunked-impl.cuh"

constexpr size_t state_bytes = size_t(S) * HV * S * sizeof(float);

// Recurrent state is [head][value][key]; the chunked scan uses [key][head][value].
template <bool TO_SCAN>
__global__ void gdn_chunked_state_transpose(const float * src, float * dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * HV * S) {
        return;
    }
    const int key = i % S, value = (i / S) % S, head = i / (S * S);
    const int j = (key * HV + head) * S + value;
    if (TO_SCAN) {
        dst[j] = src[i];
    } else {
        dst[i] = src[j];
    }
}

bool gdn_chunked_init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        const auto & info = ggml_cuda_info().devices[device];
        if (info.smpbo < kTcScanSmem<false> || info.nsm < HV) {
            return;
        }
        const auto status = cudaFuncSetAttribute(gdn_tc_scan_kernel<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, int(kTcScanSmem<false>));
        available[device] = status == cudaSuccess;
        if (status != cudaSuccess) {
            (void) cudaGetLastError();
        }
    });
    return available[device];
}
} // namespace
#endif

bool ggml_cuda_gdn_sm80_chunked_usable(int device, const ggml_cuda_gdn_sm80_chunked_args & a) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(device, a);
    return false;
#else
    if (!ggml_cuda_gdn_sm80_chunked_enabled() || !a.eligible || device < 0 || device >= GGML_CUDA_MAX_DEVICES ||
        a.H != HV || a.H_k != HK || a.n_seqs != 1 || a.rq3 != 1 || a.n_tokens < 128 || a.n_tokens > INT64_MAX - GSB ||
        ggml_cuda_info().devices[device].cc != GGML_CUDA_CC_AMPERE) {
        return false;
    }
    // float4 loads and cp.async need 16-byte aligned rows. Q and K have equal strides.
    if ((reinterpret_cast<uintptr_t>(a.q) | reinterpret_cast<uintptr_t>(a.k) |
         reinterpret_cast<uintptr_t>(a.v)) % 16 || a.sq1 % 4 || a.sq2 % 4 || a.sv1 % 4 || a.sv2 % 4) {
        return false;
    }
    return gdn_chunked_init(device);
#endif
}

size_t ggml_cuda_gdn_sm80_chunked_scratch_size(int64_t tokens) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(tokens);
    return 0;
#else
    if (tokens <= 0) {
        return 0;
    }
    const size_t nb = (std::min<int64_t>(tokens, GSB) + GCH - 1) / GCH;
    return state_bytes + nb * (kTcTiBytes + kTcTeBytes + kTcScBytes + kTcKqBytes + kTcPiBytes) / GTC_NB;
#endif
}

size_t ggml_cuda_gdn_sm80_chunked_scratch_size_for(const ggml_tensor * dst, int device) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(dst, device);
    return 0;
#else
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * v = dst->src[2];
    const auto & info = ggml_cuda_info().devices[device];
    // The scheduler queries this before tensor data pointers are assigned.
    if (!ggml_cuda_gdn_sm80_chunked_enabled() || info.cc != GGML_CUDA_CC_AMPERE || info.nsm < HV ||
        info.smpbo < kTcScanSmem<false> || v->ne[0] != S || v->ne[1] != HV || q->ne[1] != HK ||
        v->ne[2] < 128 || v->ne[3] != 1 || q->ne[3] != 1 || dst->src[3]->ne[0] != 1 ||
        ggml_get_op_params_i32(dst, 0) != 1 || (dst->src[6] && v->ne[2] <= dst->src[6]->ne[1] / 2)) {
        return 0;
    }
    return ggml_cuda_gdn_sm80_chunked_scratch_size(v->ne[2]);
#endif
}

void ggml_cuda_gdn_sm80_chunked_launch(const ggml_cuda_gdn_sm80_chunked_args & a, void * scratch, cudaStream_t stream) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(a, scratch, stream);
    GGML_ABORT("Chunked GDN requires CUDA SM80");
#else
    GGML_ASSERT(scratch && a.state && a.state_out);
    static std::once_flag notice;
    std::call_once(notice, [] {
        GGML_LOG_WARN("GDN SM80 chunked path active: FP16 scaled high/residual triple products, FP32 state\n");
    });
    const size_t nb = (std::min<int64_t>(a.n_tokens, GSB) + GCH - 1) / GCH;
    auto * state = reinterpret_cast<float *>(scratch);
    auto * p = reinterpret_cast<unsigned char *>(scratch) + state_bytes;
    GtcScratch sc;
    sc.ti = reinterpret_cast<uint16_t *>(p); p += nb * kTcTiBytes / GTC_NB;
    sc.te = reinterpret_cast<float *>(p); p += nb * kTcTeBytes / GTC_NB;
    sc.sc = reinterpret_cast<float4 *>(p); p += nb * kTcScBytes / GTC_NB;
    sc.kq = reinterpret_cast<uint16_t *>(p); p += nb * kTcKqBytes / GTC_NB;
    sc.pi = reinterpret_cast<uint16_t *>(p);
    gdn_chunked_state_transpose<true><<<S * HV * S / 256, 256, 0, stream>>>(a.state, state);
    CUDA_CHECK(cudaGetLastError());
    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += GSB) {
        const int nch = (std::min<int64_t>(GSB, a.n_tokens - t0) + GCH - 1) / GCH;
        gdn_tc_prep_kernel<<<dim3(nch, HK), 256, kTcPrepSmem, stream>>>(a, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, t0, a.n_tokens);
        CUDA_CHECK(cudaGetLastError());
        gdn_tc_scan_kernel<false><<<HV, 256, kTcScanSmem<false>, stream>>>(
            state, a, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, a.dst, t0, nch, a.n_tokens);
        CUDA_CHECK(cudaGetLastError());
    }
    gdn_chunked_state_transpose<false><<<S * HV * S / 256, 256, 0, stream>>>(state, a.state_out);
    CUDA_CHECK(cudaGetLastError());
#endif
}
