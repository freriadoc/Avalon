#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_sm80_chunked_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
};

bool ggml_cuda_gdn_sm80_chunked_enabled();
bool ggml_cuda_gdn_sm80_chunked_usable(int device, const ggml_cuda_gdn_sm80_chunked_args & args);
size_t ggml_cuda_gdn_sm80_chunked_scratch_size(int64_t tokens);
size_t ggml_cuda_gdn_sm80_chunked_scratch_size_for(const ggml_tensor * dst, int device);
void ggml_cuda_gdn_sm80_chunked_launch(const ggml_cuda_gdn_sm80_chunked_args & args, void * scratch, cudaStream_t stream);
