// IQ3_S x Q8_K dot product on AVX2: reference kernel (ggml-cpu/arch/x86/quants.c) against
// variants, bit-exact check and timing per row (one thread).
//
// build: g++ -std=c++17 -O3 -mavx2 -mfma -mf16c -mbmi2 -I ggml/src -I ggml/include tests/iq3sbench.cpp -o /tmp/iq3sbench

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

// ---------------------------------------------------------------- reference (verbatim)
static void dot_ref(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    const int nb = n / QK_K;
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);
    typedef union { __m256i vec[2]; uint32_t index[16]; } index_t;
    index_t idx;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint16_t * signs = (const uint16_t *)x[i].signs;
        const int8_t  * q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]);
            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
            signs += 4;
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// ---------------------------------------------------------------- variants
#include "iq3sbench-variants.inc"

// ---------------------------------------------------------------- harness
typedef void (*dot_fn)(int, float *, const block_iq3_s *, const block_q8_K *);

int main(int argc, char ** argv) {
    const int n = 2560, rows = 640, reps = argc > 1 ? atoi(argv[1]) : 200;
    const int nb = n / QK_K;
    std::mt19937 rng(1);
    std::vector<block_iq3_s> w((size_t) rows * nb);
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
    struct { const char * name; dot_fn fn; } fns[] = { { "ref", dot_ref }, VARIANTS };
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
               us, us * 1e3 / (rows * nb * 64.0), sink * 0);
    }
    return errors != 0;
}
