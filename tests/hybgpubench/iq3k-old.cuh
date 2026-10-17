static __device__ __forceinline__ float vec_dot_iq3_k_q8_1_old(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iiqs) {
    const block_iq3_k * bq3 = (const block_iq3_k *) vbq + kbx;

    const int iqs   = iiqs/4;
    const int ib128 = iqs/4;  // 0 or 1: weights 0...127 or 128...255
    const int il8   = iqs%4;  // 0...3: weights 8*il8...8*il8+7 of each block of 32
    const int shift = 4*(il8/2);

    const uint16_t * ql = (const uint16_t *)bq3->qs + 16*ib128 + 4*il8;
    const uint16_t * qh = (const uint16_t *)bq3->qh + 4*il8;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    const int hshift = 4*(1-ib128);
    const uint16_t sh = bq3->scales_h >> (8*ib128 + il8/2);

    const uint8_t extra = bq3->extra >> (8*ib128 + il8/2);
    const uint32_t extra32 = uint32_t(extra) * 0x01010101;
    const uint32_t extra32_1 = (extra32 << 3) & 0x88888888;
    const uint32_t extra32_2 = (extra32 << 1) & 0x88888888;

    const int * q8;
    int sumi[4] = {0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t vl = ql[2*i+0] | (ql[2*i+1] << 16);
        const uint32_t vh = ((qh[2*i+0] | (qh[2*i+1] << 16)) << hshift);

        const uint32_t val1 = ((vl >> 0) & 0x33333333) | extra32_1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
        const uint32_t val2 = ((vl >> 2) & 0x33333333) | extra32_2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);
        const int2 v1 = get_int_from_table_16(val1, iq3nl_values);
        const int2 v2 = get_int_from_table_16(val2, iq3nl_values);

        q8 = (const int *)bq8_1[4*ib128+0].qs + 2*il8;
        sumi[0] = ggml_cuda_dp4a(v1.x, q8[i], sumi[0]);

        q8 += sizeof(block_q8_1)/4;
        sumi[1] = ggml_cuda_dp4a(v2.x, q8[i], sumi[1]);

        q8 += sizeof(block_q8_1)/4;
        sumi[2] = ggml_cuda_dp4a(v1.y, q8[i], sumi[2]);

        q8 += sizeof(block_q8_1)/4;
        sumi[3] = ggml_cuda_dp4a(v2.y, q8[i], sumi[3]);
    }
    const float d = __half2float(bq3->d);
    const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
    aux32 = ((((sl16[0] | (sl16[1] << 16)) >> shift) & 0x0f0f0f0f) << 1) | 0x01010101;
    return d * (__low2float(bq8_1[4*ib128+0].ds) * aux8[0] * (sh & 0x01 ? -1 : 1) * sumi[0] +
                __low2float(bq8_1[4*ib128+1].ds) * aux8[1] * (sh & 0x04 ? -1 : 1) * sumi[1] +
                __low2float(bq8_1[4*ib128+2].ds) * aux8[2] * (sh & 0x10 ? -1 : 1) * sumi[2] +
                __low2float(bq8_1[4*ib128+3].ds) * aux8[3] * (sh & 0x40 ? -1 : 1) * sumi[3]);
}
