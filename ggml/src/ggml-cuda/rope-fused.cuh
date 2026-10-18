#pragma once

// RMS_NORM + MUL + ROPE (+ VIEW + SET_ROWS, + the 256-wide rotation) fused (rope.cu), its body shared with the
// multi-task kernel (mk.cu): the helpers of the rope kernels and the body of one block given its row, channel, sample

#include "common.cuh"
#include "cpy-utils.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

// the arguments of the fused kernel (one struct: the multi-task kernel records it, mk.cuh)
struct rms_norm_mul_rope_args {
    const float * x; void * dst; int ncols;
    int64_t s01, s02, s03, s1, s2, s3;
    float eps;
    const float * mul;
    int64_t mul_s01, mul_s02, mul_s03;
    uint3 mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed;
    int n_dims; const int32_t * pos;
    float freq_scale, ext_factor, attn_factor;
    rope_corr_dims corr_dims; float theta_scale;
    const float * freq_factors;
    const int64_t * row_indices; int set_rows_stride;
    bool is_neox; mrope_sections sections; int mrope; float fwht_scale;
    block_q8_0 * dst_q8;
    int n_rows, n_channels, n_samples;   // the grid
};

// one row of the fused kernel by a block of block_size threads; shared memory: (32 + 256) floats
template <int block_size, bool has_ff, typename D>
static __device__ __forceinline__ void rms_norm_mul_rope_f32_body(const rms_norm_mul_rope_args & a, const int row,
        const int channel, const int sample) {
    const int n_channels = a.n_channels;
    const float * x = a.x;
    D * dst = (D *) a.dst;
    const float * mul = a.mul;
    const auto ncols = a.ncols;
    const auto s01 = a.s01;
    const auto s02 = a.s02;
    const auto s03 = a.s03;
    const auto s1 = a.s1;
    const auto s2 = a.s2;
    const auto s3 = a.s3;
    const auto eps = a.eps;
    const auto mul_s01 = a.mul_s01;
    const auto mul_s02 = a.mul_s02;
    const auto mul_s03 = a.mul_s03;
    const auto mul_ncols_packed = a.mul_ncols_packed;
    const auto mul_nrows_packed = a.mul_nrows_packed;
    const auto mul_nchannels_packed = a.mul_nchannels_packed;
    const auto mul_nsamples_packed = a.mul_nsamples_packed;
    const auto n_dims = a.n_dims;
    const auto pos = a.pos;
    const auto freq_scale = a.freq_scale;
    const auto ext_factor = a.ext_factor;
    const auto attn_factor = a.attn_factor;
    const auto corr_dims = a.corr_dims;
    const auto theta_scale = a.theta_scale;
    const auto freq_factors = a.freq_factors;
    const auto row_indices = a.row_indices;
    const auto set_rows_stride = a.set_rows_stride;
    const auto is_neox = a.is_neox;
    const auto sections = a.sections;
    const auto mrope = a.mrope;
    const auto fwht_scale = a.fwht_scale;
    const auto dst_q8 = a.dst_q8;

    const int tid     = threadIdx.x;

    x += sample*s03 + channel*s02 + row*s01;

    const uint32_t mul_row     = fastmodulo(row,     mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample,  mul_nsamples_packed);
    mul += mul_sample*mul_s03 + mul_channel*mul_s02 + mul_row*mul_s01;

    float tmp = 0.0f;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float scale = rsqrtf(tmp/ncols + eps);
    // fwht_scale != 0: the row (ncols == 256) goes to shared memory, then is rotated (see the end); dst_q8 != nullptr:
    // the row (ncols <= 256) goes there too, then is written into the set_rows destination in Q8_0
    float * s_row = s_sum + 32;
    const bool fwht  = fwht_scale != 0.0f;
    const bool stage = fwht || dst_q8 != nullptr;

    if (dst_q8 == nullptr) {
        int64_t idst = sample*s3 + channel*s2 + row*s1;
        if (set_rows_stride != 0) {
            idst = row*s1 + row_indices[channel]*set_rows_stride;
        }
        dst += idst;
    }

    for (int i0 = 2*tid; i0 < ncols; i0 += 2*block_size) {
        int ix0;
        int ix1;
        if (is_neox && i0 < n_dims) {
            ix0 = i0/2;
            ix1 = i0/2 + n_dims/2;
        } else {
            ix0 = i0 + 0;
            ix1 = i0 + 1;
        }

        const float x0 = scale * x[ix0] * mul[fastmodulo(ix0, mul_ncols_packed)];
        const float x1 = scale * x[ix1] * mul[fastmodulo(ix1, mul_ncols_packed)];

        if (i0 >= n_dims) {
            if (stage) {
                s_row[ix0] = x0;
                s_row[ix1] = x1;
                continue;
            }
            dst[ix0] = ggml_cuda_cast<D>(x0);
            dst[ix1] = ggml_cuda_cast<D>(x1);
            continue;
        }

        // M-RoPE (mrope 1, interleaved: 2): the position of the sector, as rope_multi takes it (channel = token)
        int ip = channel;
        if (mrope != 0) {
            const int ne02      = n_channels;
            const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
            const int sec_w     = sections.v[1] + sections.v[0];
            const int sector    = (i0 / 2) % sect_dims;
            int k = 0;
            if (mrope == 2) {
                k = sector % 3 == 1 && sector < 3 * sections.v[1] ? 1 :
                    sector % 3 == 2 && sector < 3 * sections.v[2] ? 2 :
                    sector % 3 == 0 && sector < 3 * sections.v[0] ? 0 : 3;
            } else {
                k = sector < sections.v[0] ? 0 : sector < sec_w ? 1 : sector < sec_w + sections.v[2] ? 2 : 3;
            }
            ip = channel + ne02 * k;
        }
        const float theta_base  = pos[ip]*powf(theta_scale, i0/2.0f);
        const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

        float cos_theta;
        float sin_theta;
        rope_yarn<true>(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, cos_theta, sin_theta);

        if (stage) {
            s_row[ix0] = x0*cos_theta - x1*sin_theta;
            s_row[ix1] = x0*sin_theta + x1*cos_theta;
            continue;
        }
        dst[ix0] = ggml_cuda_cast<D>(x0*cos_theta - x1*sin_theta);
        dst[ix1] = ggml_cuda_cast<D>(x0*sin_theta + x1*cos_theta);
    }

    if (stage) {
        // by one warp, lane l holding values i*32 + l: the rotation (MUL_MAT by a Hadamard matrix) of the row with
        // fwht_cuda<256>'s arithmetic, then the row written; in Q8_0, register index i is block i (k_set_rows_quant's
        // bytes)
        __syncthreads();
        if (tid < WARP_SIZE) {
            constexpr int N = 256;
            constexpr int el_max = N / WARP_SIZE;
            const int el_w = ncols / WARP_SIZE;
            const int lane = tid;
            float reg[el_max];
#pragma unroll
            for (int i = 0; i < el_max; ++i) {
                reg[i] = i < el_w ? s_row[i * WARP_SIZE + lane] : 0.0f;
            }
            if (fwht) {
#pragma unroll
                for (int i = 0; i < el_max; ++i) {
                    reg[i] *= fwht_scale;
                }
#pragma unroll
                for (int h = 1; h < WARP_SIZE; h *= 2) {
#pragma unroll
                    for (int j = 0; j < el_max; j++) {
                        const float val  = reg[j];
                        const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, WARP_SIZE);
                        reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
                    }
                }
#pragma unroll
                for (int h = WARP_SIZE; h < N; h *= 2) {
                    const int step = h / WARP_SIZE;
#pragma unroll
                    for (int j = 0; j < el_max; j += 2 * step) {
#pragma unroll
                        for (int k = 0; k < step; k++) {
                            const float a = reg[j + k];
                            const float b = reg[j + k + step];
                            reg[j + k]        = a + b;
                            reg[j + k + step] = a - b;
                        }
                    }
                }
            }
            if (dst_q8 != nullptr) {
                quantize_warp_q8_0_n<el_max>(reg, dst_q8 + row*el_w + row_indices[channel]*set_rows_stride, lane, el_w);
            } else {
#pragma unroll
                for (int i = 0; i < el_max; ++i) {
                    if (i < el_w) {
                        dst[i * WARP_SIZE + lane] = ggml_cuda_cast<D>(reg[i]);
                    }
                }
            }
        }
    }
}


