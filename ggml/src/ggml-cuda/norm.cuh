#include "common.cuh"
#include "convert.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// img: also the image of the result (ggml_cuda_f16_image_target)
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, ggml_cuda_img img = ggml_cuda_img());

// rms_norm(x) * w (mul) and mm = W x reshape(mul), W Q8_0 of 4 or 8 rows (qwen4exp's hyper-connection inject), one kernel
bool ggml_cuda_rms_norm_mul_inject_supported(const ggml_tensor * rms_norm, const ggml_tensor * mul, const ggml_tensor * mm);
void ggml_cuda_op_rms_norm_mul_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, ggml_tensor * mul,
        ggml_tensor * mm, ggml_cuda_img img = ggml_cuda_img());

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// rms_norm(x) * w * sigmoid(gate) in one kernel (gated norm)
void ggml_cuda_op_rms_norm_mul_sigmoid_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * w, const ggml_tensor * gate, ggml_tensor * dst,
        void * dst_q8_1 = nullptr, ggml_cuda_img img = ggml_cuda_img());
