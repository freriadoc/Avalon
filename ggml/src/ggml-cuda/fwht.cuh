#include "common.cuh"

bool ggml_cuda_op_mul_mat_use_fwht(const struct ggml_tensor * op);

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);

// The rotation of 64-wide rows back (MUL_MAT by a Hadamard matrix), times the sigmoid of a gate read through its view
// (the gate's CONT skipped), the f32 product written with its q8_1 for the quantized product that reads it next:
// MUL_MAT (fwht), CONT, SIGMOID, MUL in one kernel (qwen4exp's gated attention output). Returns the nodes to skip
// after i, 0 if not fused.
int ggml_cuda_try_fuse_fwht_gate(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
// the same match, for the allocation dependencies: the product and the two inputs it reads (rotated rows, gate's
// source), nullptr if no match at i
ggml_tensor * ggml_cuda_match_fwht_gate(const ggml_cgraph * cgraph, int i, ggml_tensor ** rows, ggml_tensor ** gate_src);
