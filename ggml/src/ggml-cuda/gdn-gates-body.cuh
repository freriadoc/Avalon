#pragma once

// the gated delta net's gates (unary.cu), its body shared with the multi-task kernel (mk.cu): g = softplus(alpha + dt) * A
// (dt, A: one value per head, the first dimension of alpha) and beta = sigmoid(b), elementwise; an output may alias
// its input

#include "common.cuh"
#include "unary.cuh"

// unary.cu's op_softplus and op_sigmoid
static __device__ __forceinline__ float gdn_gates_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float gdn_gates_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

struct gdn_gates_args {
    const float * alpha; const float * dt; const float * a; const float * b;
    float * g; float * beta;
    int64_t n, n_heads;
};

static __device__ __forceinline__ void gdn_gates_body(const gdn_gates_args & a, const int64_t i) {
    if (i >= a.n) {
        return;
    }
    const int64_t h = i % a.n_heads;
    const float g_i = gdn_gates_softplus(a.alpha[i] + a.dt[h]) * a.a[h];
    const float b_i = gdn_gates_sigmoid(a.b[i]);
    a.g[i]    = g_i;
    a.beta[i] = b_i;
}
