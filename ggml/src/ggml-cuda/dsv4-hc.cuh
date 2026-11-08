#include "common.cuh"
#include "convert.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// img: also the image of a contiguous dst (ggml_cuda_f16_image_target)
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_img img = ggml_cuda_img());
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

// the gated, comb-less hc_post (gate chain gate_scale_in -> SIGMOID -> gate_scale_out) followed by rms_norm of its
// result and the MUL by [n_embd, hc] weights, in one kernel; the outputs are post_node's and mul's
bool ggml_cuda_dsv4_hc_post_norm_supported(const ggml_tensor * post_node, const ggml_tensor * mul);
void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, ggml_tensor * post_node,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out, const ggml_tensor * rms_norm,
        ggml_tensor * mul);
