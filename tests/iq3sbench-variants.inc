// variants of the IQ3_S dot product, included by iq3sbench.cpp

// grid entry of weight group k (4 weights) of a 32-weight block: 8 bits in qs, bit k of qh
#define IQ3S_G(qs, qh, k) iq3s_grid[(qs)[k] | ((((uint32_t) (qh)) << (8 - (k))) & 256)]

// the sign and scale half, shared by all variants
#define IQ3S_TAIL                                                                                  \
            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));                         \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                    \
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);                                  \
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);              \
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));                                \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                    \
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);                                  \
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);              \
            signs += 4;                                                                             \
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);                                \
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);                                \
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;                                         \
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;                                         \
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));                 \
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));                 \
            sumi1 = _mm256_add_epi32(sumi1, p1);                                                    \
            sumi2 = _mm256_add_epi32(sumi2, p2);

#define IQ3S_HEAD                                                                                   \
    const int nb = n / QK_K;                                                                        \
    static const uint8_t k_mask1[32] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3}; \
    static const uint8_t k_mask2[32] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,                  \
                                        1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};                 \
    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);                              \
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);                              \
    __m256 accumf = _mm256_setzero_ps();

// V1: indices in scalar registers, same 32-bit inserts
static void dot_v1(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
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
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            const __m256i q2_1 = _mm256_set_epi32(IQ3S_G(qs, h0, 7), IQ3S_G(qs, h0, 6), IQ3S_G(qs, h0, 5), IQ3S_G(qs, h0, 4),
                                                  IQ3S_G(qs, h0, 3), IQ3S_G(qs, h0, 2), IQ3S_G(qs, h0, 1), IQ3S_G(qs, h0, 0));
            const __m256i q2_2 = _mm256_set_epi32(IQ3S_G(qs+8, h1, 7), IQ3S_G(qs+8, h1, 6), IQ3S_G(qs+8, h1, 5), IQ3S_G(qs+8, h1, 4),
                                                  IQ3S_G(qs+8, h1, 3), IQ3S_G(qs+8, h1, 2), IQ3S_G(qs+8, h1, 1), IQ3S_G(qs+8, h1, 0));
            qs += 16;
            IQ3S_TAIL
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// V2: pairs of entries combined in 64-bit integer registers, 4 inserts per vector
#define IQ3S_P(qs, h, k) ((uint64_t) IQ3S_G(qs, h, k) | ((uint64_t) IQ3S_G(qs, h, (k)+1) << 32))
static void dot_v2(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
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
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            const __m256i q2_1 = _mm256_set_epi64x(IQ3S_P(qs, h0, 6), IQ3S_P(qs, h0, 4), IQ3S_P(qs, h0, 2), IQ3S_P(qs, h0, 0));
            const __m256i q2_2 = _mm256_set_epi64x(IQ3S_P(qs+8, h1, 6), IQ3S_P(qs+8, h1, 4), IQ3S_P(qs+8, h1, 2), IQ3S_P(qs+8, h1, 0));
            qs += 16;
            IQ3S_TAIL
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// V3: entries written to an aligned scratch by scalar stores, read back as vectors
static void dot_v3(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
    alignas(32) uint32_t g[16];
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
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            for (int k = 0; k < 8; ++k) {
                g[k]     = IQ3S_G(qs, h0, k);
                g[k + 8] = IQ3S_G(qs + 8, h1, k);
            }
            const __m256i q2_1 = _mm256_load_si256((const __m256i *) g);
            const __m256i q2_2 = _mm256_load_si256((const __m256i *) (g + 8));
            qs += 16;
            IQ3S_TAIL
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}


// V4: indices computed in scalar registers, entries fetched by vpgatherdd
static void dot_v4(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);
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
            __m256i i0 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32+0]), idx_shift), idx_mask);
            __m256i i1 = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh[ib32+1]), idx_shift), idx_mask);
            i0 = _mm256_or_si256(i0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            i1 = _mm256_or_si256(i1, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_i32gather_epi32((const int *) iq3s_grid, i0, 4);
            const __m256i q2_2 = _mm256_i32gather_epi32((const int *) iq3s_grid, i1, 4);
            IQ3S_TAIL
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// V5: V1 with two super-blocks interleaved (their sums kept apart, added in the same order)
static void dot_v5(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
    int i = 0;
    for (; i + 1 < nb; i += 2) {
        __m256i sa1 = _mm256_setzero_si256(), sa2 = _mm256_setzero_si256();
        __m256i sb1 = _mm256_setzero_si256(), sb2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
#define IQ3S_ONE(I, SUM1, SUM2) {                                                                   \
            const uint8_t * qs = x[I].qs + 8*ib32;                                                  \
            const uint16_t * signs = (const uint16_t *) x[I].signs + 2*ib32;                        \
            const int8_t * q8 = y[I].qs + 32*ib32;                                                  \
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);                           \
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));                    \
            const uint32_t h0 = x[I].qh[ib32+0], h1 = x[I].qh[ib32+1];                              \
            const __m256i q2_1 = _mm256_set_epi32(IQ3S_G(qs, h0, 7), IQ3S_G(qs, h0, 6), IQ3S_G(qs, h0, 5), IQ3S_G(qs, h0, 4), \
                                                  IQ3S_G(qs, h0, 3), IQ3S_G(qs, h0, 2), IQ3S_G(qs, h0, 1), IQ3S_G(qs, h0, 0)); \
            const __m256i q2_2 = _mm256_set_epi32(IQ3S_G(qs+8, h1, 7), IQ3S_G(qs+8, h1, 6), IQ3S_G(qs+8, h1, 5), IQ3S_G(qs+8, h1, 4), \
                                                  IQ3S_G(qs+8, h1, 3), IQ3S_G(qs+8, h1, 2), IQ3S_G(qs+8, h1, 1), IQ3S_G(qs+8, h1, 0)); \
            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));                        \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                   \
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);                                 \
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);             \
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));                               \
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);                   \
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);                                 \
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);             \
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);                               \
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);                               \
            const uint16_t ls1 = x[I].scales[ib32/2] & 0xf;                                        \
            const uint16_t ls2 = x[I].scales[ib32/2] >>  4;                                        \
            SUM1 = _mm256_add_epi32(SUM1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1)));   \
            SUM2 = _mm256_add_epi32(SUM2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1)));   \
        }
            IQ3S_ONE(i, sa1, sa2)
            IQ3S_ONE(i + 1, sb1, sb2)
        }
        const float da = fp16_to_fp32(x[i].d) * y[i].d;
        const float db = fp16_to_fp32(x[i+1].d) * y[i+1].d;
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(da), _mm256_cvtepi32_ps(_mm256_add_epi32(sa1, sa2)), accumf);
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(db), _mm256_cvtepi32_ps(_mm256_add_epi32(sb1, sb2)), accumf);
    }
    for (; i < nb; ++i) {
        __m256i sa1 = _mm256_setzero_si256(), sa2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            IQ3S_ONE(i, sa1, sa2)
        }
        const float da = fp16_to_fp32(x[i].d) * y[i].d;
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(da), _mm256_cvtepi32_ps(_mm256_add_epi32(sa1, sa2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

// V6: V1 with every grid index pinned to an integer register (see iq2sbench.cpp W5)
static inline uint32_t iq3s_pin(uint32_t v) { __asm__("" : "+r"(v)); return v; }
#define IQ3S_GP(qs, qh, k) iq3s_grid[iq3s_pin((qs)[k] | ((((uint32_t) (qh)) << (8 - (k))) & 256))]
static void dot_v6(int n, float * s, const block_iq3_s * x, const block_q8_K * y) {
    IQ3S_HEAD
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
            const uint32_t h0 = qh[ib32+0], h1 = qh[ib32+1];
            const __m256i q2_1 = _mm256_set_epi32(IQ3S_GP(qs, h0, 7), IQ3S_GP(qs, h0, 6), IQ3S_GP(qs, h0, 5), IQ3S_GP(qs, h0, 4),
                                                  IQ3S_GP(qs, h0, 3), IQ3S_GP(qs, h0, 2), IQ3S_GP(qs, h0, 1), IQ3S_GP(qs, h0, 0));
            const __m256i q2_2 = _mm256_set_epi32(IQ3S_GP(qs+8, h1, 7), IQ3S_GP(qs+8, h1, 6), IQ3S_GP(qs+8, h1, 5), IQ3S_GP(qs+8, h1, 4),
                                                  IQ3S_GP(qs+8, h1, 3), IQ3S_GP(qs+8, h1, 2), IQ3S_GP(qs+8, h1, 1), IQ3S_GP(qs+8, h1, 0));
            qs += 16;
            IQ3S_TAIL
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    *s = hsum_float_8(accumf);
}

#define VARIANTS { "v1-scal", dot_v1 }, { "v2-pair64", dot_v2 }, { "v3-store", dot_v3 }, { "v4-gather", dot_v4 }, { "v5-2blocks", dot_v5 }, { "v6-pin", dot_v6 }
