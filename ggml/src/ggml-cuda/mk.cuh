#pragma once

// Multi-task kernel (on by default, GGML_CUDA_MK=0 turns it off): consecutive independent small kernels launched as one. Each launch costs
// ~2.4 us on RDNA3 between two dependent kernels of a graph (tests/gapbench); kernels that do not depend on each
// other need no such wait. While the graph is evaluated, the launchers that support it record their task here
// instead of launching it, with the memory its nodes read and write; before a node that reads what a pending task
// writes, or writes what one reads or writes, the pending tasks are launched together (ggml_cuda_mk_flush), each block
// of the kernel taking a block of one task and running the original kernel's body (rope-fused.cuh, fwht-sr.cuh,
// qsa-select-body.cuh), so the results are the same bit for bit. A launcher that does not record launches as before:
// it conflicts with no pending task, which may run after it.

#include "common.cuh"
#include "rope-fused.cuh"
#include "fwht-sr.cuh"
#include "qsa-select-body.cuh"
#include "ssm-conv-body.cuh"
#include "gdn-gates-body.cuh"
#include "gdn-body.cuh"
#include "norm-gate-body.cuh"
#include "getrows-body.cuh"
#include "idx-key-body.cuh"

#include <vector>
#include <unordered_map>

#define GGML_CUDA_MK_MAX_TASKS 8
#define GGML_CUDA_MK_THREADS   256

enum ggml_cuda_mk_kind {
    GGML_CUDA_MK_ROPE,      // rms_norm_mul_rope_f32<256, false, float>
    GGML_CUDA_MK_ROPE_FF,   // rms_norm_mul_rope_f32<256, true, float>
    GGML_CUDA_MK_FWHT_SR,   // fwht_set_rows_cuda<64, block_q4_0 or block_q8_0>, 8 rows per block
    GGML_CUDA_MK_QSA_SEL,   // k_qsa_select<mode, by_pos>
    GGML_CUDA_MK_SSM_CONV,  // ssm_conv_cached_f32<4, GGML_CUDA_SSM_CONV_CACHED_MAX_T>, 128 threads of the block
    GGML_CUDA_MK_GDN_GATES, // k_gdn_gates
    GGML_CUDA_MK_GDN,       // gated_delta_net_cuda<128, false, false, 4, false>, 4 warps
    GGML_CUDA_MK_NORM_GATE, // rms_norm_mul_sigmoid_gate_f32<128>
    GGML_CUDA_MK_GATHER,    // get_rows of q8_0 / q4_0 rows into f16 (getrows-body.cuh), 8 rows per block
    GGML_CUDA_MK_FILL,      // fill_kernel<float>
    GGML_CUDA_MK_SR_Q8,     // k_set_rows_quant into Q8_0, rows padded with zeros (idx-key-body.cuh)
    GGML_CUDA_MK_GR_MEAN,   // k_get_rows_mean (idx-key-body.cuh), a thread per pair of values
};

struct mk_fill_args {
    float * dst; int64_t n; float value;
};

#define GGML_CUDA_MK_MAX_STAGES 4

union ggml_cuda_mk_payload {
    rms_norm_mul_rope_args rope;
    fwht_set_rows_args     fwht;
    qsa_select_args        sel;
    ssm_conv_cached_args   conv;
    gdn_gates_args         gates;
    gated_delta_net_args   gdn;
    rms_norm_gate_args     norm_gate;
    get_rows_q_args        gather;
    mk_fill_args           fill;
    set_rows_q8_args       sr_q8;
    get_rows_mean_args     gr_mean;
    ggml_cuda_mk_payload() {}
};

struct ggml_cuda_mk_task {
    int kind;
    int first_block;   // its blocks in the kernel: first_block .. first_block + n_blocks - 1
    int n_blocks;
    int stage;         // tasks of a later stage run after a grid barrier (k_mk_chain128)
    ggml_cuda_mk_payload p;
};

struct ggml_cuda_mk_tasks {
    int n;
    ggml_cuda_mk_task t[GGML_CUDA_MK_MAX_TASKS];
};

// the recorder of a CUDA context
struct ggml_cuda_mk {
    struct range {
        const char * lo;
        const char * hi;
    };
    ggml_cuda_mk_tasks tasks;
    int    n_blocks = 0;
    size_t smem     = 0;
    int    ranged   = 0;   // tasks whose ranges are known (recorded by a unit already done)
    std::vector<range> reads, writes;   // of all the pending tasks
    std::vector<range> stage_reads, stage_writes;   // of the last stage's tasks
    int n_stages = 1;
    int new_stage_at = -1;   // the task (index) that opens a stage, chosen by the unit's check
    int64_t launches = 0, merged = 0;   // statistics: kernels launched, tasks they ran
    // the graph being evaluated: the index of the last node that reads each tensor (through views), set once per
    // evaluation (ggml_cuda_mk_begin_graph)
    std::unordered_map<const ggml_tensor *, int> last_read;

    ggml_cuda_mk() { tasks.n = 0; }
};

// a task recorded (true), or not (false: the caller launches it): the recorder is full
bool ggml_cuda_mk_add_rope(ggml_cuda_mk * mk, const rms_norm_mul_rope_args & a, bool has_ff);
bool ggml_cuda_mk_add_fwht_set_rows(ggml_cuda_mk * mk, const fwht_set_rows_args & a);
bool ggml_cuda_mk_add_qsa_select(ggml_cuda_mk * mk, const qsa_select_args & a);
bool ggml_cuda_mk_add_ssm_conv(ggml_cuda_mk * mk, const ssm_conv_cached_args & a);
bool ggml_cuda_mk_add_gdn_gates(ggml_cuda_mk * mk, const gdn_gates_args & a);
bool ggml_cuda_mk_add_gdn(ggml_cuda_mk * mk, const gated_delta_net_args & a);
bool ggml_cuda_mk_add_norm_gate(ggml_cuda_mk * mk, const rms_norm_gate_args & a);
bool ggml_cuda_mk_add_gather(ggml_cuda_mk * mk, const get_rows_q_args & a);
bool ggml_cuda_mk_add_fill(ggml_cuda_mk * mk, const mk_fill_args & a);
bool ggml_cuda_mk_add_set_rows_q8(ggml_cuda_mk * mk, const set_rows_q8_args & a);
bool ggml_cuda_mk_add_get_rows_mean(ggml_cuda_mk * mk, const get_rows_mean_args & a);
// whether a task can be chained after the pending ones behind a grid barrier: of class B or C (k_mk_chain128); of the
// kinds of k_mk_chain256 (the indexer's key: its cache row, the pool's mean, its rope into the cache) when every
// pending task is too (256)
bool ggml_cuda_mk_chainable(const ggml_cuda_mk * mk);
bool ggml_cuda_mk_chainable256(const ggml_cuda_mk * mk);

// the evaluation loop's hooks: the unit of nodes first .. last done (the ranges of the tasks it recorded); whether a
// node conflicts with a pending task; the pending tasks launched as one kernel
// returns false when the unit launched a kernel the normal way that conflicts with a pending task (an ordering error)
bool ggml_cuda_mk_unit_done(ggml_cuda_mk * mk, const ggml_cgraph * cgraph, int first, int last, cudaStream_t stream);
bool ggml_cuda_mk_conflicts(const ggml_cuda_mk * mk, const ggml_tensor * node);
bool ggml_cuda_mk_unit_conflicts(const ggml_cuda_mk * mk, const ggml_cgraph * cgraph, const int * idx, int n);
void ggml_cuda_mk_flush(ggml_cuda_mk * mk, cudaStream_t stream);
void ggml_cuda_mk_begin_graph(ggml_cuda_mk * mk, const ggml_cgraph * cgraph);
static inline int ggml_cuda_mk_pending(const ggml_cuda_mk * mk) { return mk->tasks.n; }
