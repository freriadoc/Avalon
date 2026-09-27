// IQ2_S x Q8_K dot product on AVX2: reference kernel (ggml-cpu/arch/x86/quants.c) against
// variants, bit-exact check and timing per row (one thread).
//
// build: g++ -std=c++17 -O3 -mavx2 -mfma -mf16c -mbmi2 -I ggml/src -I ggml/include tests/iq2sbench.cpp -o /tmp/iq2sbench

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static inline float fp16_to_fp32(ggml_half h) { return _cvtsh_ss(h); }

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}


static const uint8_t k_mask1[32] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3};
static const uint8_t k_mask2[32] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

#define IQ2S_G(qs, h, k) iq2s_grid[(qs)[k] | ((((uint32_t) (h)) << (8 - 2*(k))) & 0x300)]

// W0: reference (ggml-cpu/arch/x86/quants.c, verbatim apart from the block loop body macro)
#define IQ2S_BODY(Q2_1, Q2_2)                                                                       \
            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));             \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                   \
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);                                 \
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);             \
            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));                    \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                   \
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);                                 \
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);             \
            signs += 4;                                                                            \
            const __m256i dot1  = _mm256_maddubs_epi16(Q2_1, q8s_1);                               \
            const __m256i dot2  = _mm256_maddubs_epi16(Q2_2, q8s_2);                               \
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0))); \
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1))); \
            sumi1 = _mm256_add_epi32(sumi1, p1);                                                   \
            sumi2 = _mm256_add_epi32(sumi2, p2);

#define IQ2S_PRE                                                                                   \
        const float d = fp16_to_fp32(x[i].d) * y[i].d;                                              \
        const uint8_t * qs = x[i].qs;                                                               \
        const uint8_t * qh = x[i].qh;                                                               \
        const uint16_t * signs = (const uint16_t *)(x[i].qs + QK_K/8);                             \
        const int8_t  * q8 = y[i].qs;                                                               \
        uint64_t aux64; memcpy(&aux64, x[i].scales, 8);                                             \
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1); \
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);                                     \
        __m256i sumi1 = _mm256_setzero_si256();                                                     \
        __m256i sumi2 = _mm256_setzero_si256();

#define IQ2S_CONSTS                                                                                \
    const int nb = n / QK_K;                                                                        \
    const __m128i m4 = _mm_set1_epi8(0xf);                                                          \
    const __m128i m1 = _mm_set1_epi8(1);                                                            \
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);                              \
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);                              \
    __m256 accumf = _mm256_setzero_ps();

static void dot_ref(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}

// W1: grid entries written to an aligned scratch, read back as vectors
static void dot_w1(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    alignas(32) uint64_t g[8];
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const uint32_t h0 = qh[ib32], h1 = qh[ib32+1];
            for (int k = 0; k < 4; ++k) { g[k] = IQ2S_G(qs, h0, k); g[k+4] = IQ2S_G(qs+4, h1, k); }
            const __m256i q2_1 = _mm256_load_si256((const __m256i *) g);
            const __m256i q2_2 = _mm256_load_si256((const __m256i *) (g + 4));
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}

// W2: whole super-block of grid entries first (32 scalar loads into a scratch), then the vector part
static void dot_w2(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    alignas(32) uint64_t g[32];
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint32_t h = qh[ib];
            for (int k = 0; k < 4; ++k) g[4*ib + k] = IQ2S_G(qs + 4*ib, h, k);
        }
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_load_si256((const __m256i *) (g + 4*ib32));
            const __m256i q2_2 = _mm256_load_si256((const __m256i *) (g + 4*ib32 + 4));
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}

// W3: each 64-bit entry loaded straight into a vector register, pairs joined by unpacklo,
// halves by inserti128 (no pinsrq, 2 uops on Zen 2)
static inline __m128i iq2s_ld(uint64_t v_idx) { return _mm_loadl_epi64((const __m128i *) (iq2s_grid + v_idx)); }
static inline __m256i iq2s_q2(const uint8_t * qs, uint32_t h) {
    const __m128i lo = _mm_unpacklo_epi64(iq2s_ld(qs[0] | ((h << 8) & 0x300)), iq2s_ld(qs[1] | ((h << 6) & 0x300)));
    const __m128i hi = _mm_unpacklo_epi64(iq2s_ld(qs[2] | ((h << 4) & 0x300)), iq2s_ld(qs[3] | ((h << 2) & 0x300)));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}
static void dot_w3(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = iq2s_q2(qs, qh[ib32+0]);
            const __m256i q2_2 = iq2s_q2(qs + 4, qh[ib32+1]);
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}


// W5: reference with every grid index pinned to an integer register (clang otherwise builds the
// indices in a vector and extracts them with vpextrw, 2 uops each on Zen 2)
static inline uint32_t iq2s_pin(uint32_t v) { __asm__("" : "+r"(v)); return v; }
#define IQ2S_I(qs, h, k) iq2s_grid[iq2s_pin((qs)[k] | ((((uint32_t) (h)) << (8 - 2*(k))) & 0x300))]
static void dot_w5(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            const __m256i q2_1 = _mm256_set_epi64x(IQ2S_I(qs, h0, 3), IQ2S_I(qs, h0, 2), IQ2S_I(qs, h0, 1), IQ2S_I(qs, h0, 0));
            const __m256i q2_2 = _mm256_set_epi64x(IQ2S_I(qs+4, h1, 3), IQ2S_I(qs+4, h1, 2), IQ2S_I(qs+4, h1, 1), IQ2S_I(qs+4, h1, 0));
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}

// U1: reference, loop fully unrolled
static void dot_u1(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
#pragma GCC unroll 4
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}


// U2: pinned indices, loop fully unrolled
static void dot_u2(int n, float * s, const block_iq2_s * x, const block_q8_K * y) {
    IQ2S_CONSTS
    for (int i = 0; i < nb; ++i) {
        IQ2S_PRE
#pragma GCC unroll 4
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            const __m256i q2_1 = _mm256_set_epi64x(IQ2S_I(qs, h0, 3), IQ2S_I(qs, h0, 2), IQ2S_I(qs, h0, 1), IQ2S_I(qs, h0, 0));
            const __m256i q2_2 = _mm256_set_epi64x(IQ2S_I(qs+4, h1, 3), IQ2S_I(qs+4, h1, 2), IQ2S_I(qs+4, h1, 1), IQ2S_I(qs+4, h1, 0));
            qs += 8;
            IQ2S_BODY(q2_1, q2_2)
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = 0.125f * hsum_float_8(accumf);
}


// ---------------------------------------------------------------- harness
typedef void (*dot_fn)(int, float *, const block_iq2_s *, const block_q8_K *);

int main(int argc, char ** argv) {
    const int n = 2560, rows = 640, reps = argc > 1 ? atoi(argv[1]) : 200;
    const int nb = n / QK_K;
    std::mt19937 rng(1);
    std::vector<block_iq2_s> w((size_t) rows * nb);
    for (auto & b : w) {
        uint8_t * p = (uint8_t *) &b;
        for (size_t k = 0; k < sizeof(b); ++k) p[k] = (uint8_t) rng();
        b.d = _cvtss_sh(0.01f + (rng() % 100) * 1e-4f, 0);
    }
    std::vector<block_q8_K> a(nb);
    for (auto & b : a) {
        for (int k = 0; k < QK_K; ++k) b.qs[k] = (int8_t) (rng() % 255 - 127);
        b.d = 0.02f;
    }
    struct { const char * name; dot_fn fn; } fns[] = { { "ref", dot_ref }, { "w1-store", dot_w1 }, { "w2-block", dot_w2 }, { "w3-unpack", dot_w3 }, { "w5-pin", dot_w5 }, { "u1-unroll", dot_u1 }, { "u2-pin-unr", dot_u2 } };
    std::vector<float> out_ref(rows);
    for (int r = 0; r < rows; ++r) dot_ref(n, &out_ref[r], &w[(size_t) r*nb], a.data());
    int errors = 0;
    for (auto & f : fns) {
        std::vector<float> out(rows);
        for (int r = 0; r < rows; ++r) f.fn(n, &out[r], &w[(size_t) r*nb], a.data());
        const bool same = memcmp(out.data(), out_ref.data(), rows*sizeof(float)) == 0;
        errors += !same;
        float sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; ++k) {
            for (int r = 0; r < rows; ++r) { float v; f.fn(n, &v, &w[(size_t) r*nb], a.data()); sink += v; }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("%-10s %s  %8.1f us per expert (640 rows, 1 thread)  %6.3f ns per lookup  [%g]\n", f.name, same ? "exact" : "DIFF ",
               us, us * 1e3 / (rows * nb * 32.0), sink * 0);
    }
    return errors != 0;
}
