#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    bool get_causal_attn() const;

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    uint32_t reserve_outputs(int32_t n_outputs) { return output_reserve(n_outputs); }   // llama_output_reserve
    void set_nextn_rows_cb(llama_nextn_rows_cb cb, void * user);
    bool set_mtp_h_rows(const ggml_tensor * rows, int64_t n_rows, ggml_backend_t backend, const float * row_first);
    bool set_mtp_h_rows_ev(const ggml_tensor * rows, int64_t n_rows, ggml_backend_t backend, const float * row_first,
            ggml_backend_event_t ev_ready);
    bool mtp_h_rows_record(ggml_backend_t backend, ggml_backend_event_t ev);
    bool share_moe_cache(llama_context & owner);                                            // llama_share_moe_cache
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    bool set_mtp_h_from(llama_context & tgt, const float * row_first);                    // llama_set_mtp_h_from
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // the row from which the KV cache is lent to the MoE expert cache, given to the memory module
    void kv_lent_from_update();

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    // launch = false: everything but the launch (memory, graph, allocation, inputs), done by process_ubatch_launch: a
    // pipeline stage prepares its ubatch while the previous stage computes
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret,
                              bool   launch = true);
    ggml_status process_ubatch_launch(llm_graph_result * res, const llama_ubatch & ubatch);

    int encode(const llama_batch_ext & batch_inp);
    int decode(const llama_batch_ext & batch_inp);

    // the first stage of a pipeline over several GPUs (the default, LLAMA_STAGES=0 turns it off): a batch processed ubatch by ubatch, split as the
    // last stage splits it (same breaks, same n_ubatch); each step writes its layers' output into `out`
    int  stage_begin(const llama_batch_ext & batch_inp, const std::vector<int32_t> & breaks, bool embd_all);
    // the buffers between the stages, for ubatches of up to n_rows tokens: a prefill's taken from the MoE caches of the
    // two devices (as the compute buffers are), a few rows otherwise; false when they cannot be allocated
    bool stage_buffers(uint32_t n_rows);
    void stage_ubatch_for(uint32_t n_tokens);   // LLAMA_STAGES_FULL_UBATCH_FROM: this batch's ubatch (no reservation)
    uint32_t ubatch_reserved() const;           // the ubatch the graphs are reserved for: the largest a batch may take
    // the ubatch a prompt batch of n_tokens is split at: n_ubatch, or less when the compute buffers are borrowed from the
    // MoE cache and it cannot lend a ubatch that large now (llama_context::ubatch_fit)
    uint32_t ubatch_fit(uint32_t n_tokens);
    bool stage_step(ggml_tensor * out, uint32_t & n_tokens);
    void stage_end();

    // compat version
    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    bool extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // pipeline stages over several GPUs (the default, LLAMA_STAGES=0 turns it off): this context is the last stage (its device's layers, the head,
    // the outputs), stage_first the first (the layers before, the embeddings); between them the layers' output of each
    // ubatch, written on the first stage's device (stage_out) and copied to this one's (stage_in), two of each in turn
    std::unique_ptr<llama_context> stage_first;
    ggml_context_ptr        stage_ctx;
    ggml_backend_buffer_ptr stage_buf_out[2];
    ggml_backend_buffer_ptr stage_buf_in[2];
    uint32_t stage_rows = 0;   // the ubatch size the buffers hold
    uint32_t stage_ub_half = 0;   // pipeline stages: the ubatch they split a batch into (the requested one halved)
    uint32_t stage_ub_full = 0;   // and the requested one (LLAMA_STAGES_FULL_UBATCH_FROM)

    // the device compute buffer of a prompt ubatch, a + b x tokens (b in bytes per token), from two measures of the
    // worst-case graph at the first reservation whose buffers are borrowed (ubatch_fit); 0: not known
    double   ub_cb_fixed     = 0.0;
    double   ub_cb_per_token = 0.0;
    uint32_t ub_fit_last     = 0;
    // the stage buffers each kept graph was built with (process_ubatch reuses a graph for the same ones only)
    std::map<const void *, std::pair<const ggml_tensor *, const ggml_tensor *>> stage_graph_bufs;
    ggml_tensor * stage_out[2] = { nullptr, nullptr };
    ggml_tensor * stage_in [2] = { nullptr, nullptr };
    // recorded by this stage behind the graph that read stage_in[i]: the first stage waits for it (on its GPU) before it
    // copies the ubatch two later there
    ggml_backend_event_t stage_ev_read[2] = { nullptr, nullptr };
    // a first stage's batch in progress
    llama_memory_context_ptr stage_mctx;
    bool     stage_more   = false;
    uint32_t stage_n_left = 0;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // the nextn rows of the last decode on the device when it was one ubatch with every row (an MTP draft reads them
    // there, llama_set_mtp_h_from); valid until the next decode allocates its graph
    const ggml_tensor * h_nextn_dev      = nullptr;
    ggml_backend_t      h_nextn_backend  = nullptr;   // its backend: copies from it are queued on its stream, never
                                                      // synchronous ones (a hosted MoE graph may wait for the host)
    int64_t             h_nextn_dev_rows = 0;

    // the nextn rows of a decode of several ubatches gathered on the device in batch order, for the MTP draft as
    // h_nextn_dev: each ubatch's rows are overwritten by the next one in the compute buffer, and through the host the
    // draft's input cost ~0.4 s per 19k-token prompt. Borrowed from the expert cache (the compute buffers' type), given
    // back at the next decode
    // the rows of the last decode in the host output buffer in batch order (embd_nextn), when not on the device
    int64_t                 h_nextn_host_rows = 0;
    int64_t                 h_host_n  = 0;
    bool                    h_host_ok = false;

    ggml_context_ptr        h_rows_ctx;
    ggml_backend_buffer_ptr h_rows_buf;
    ggml_tensor *           h_rows         = nullptr;
    ggml_backend_t          h_rows_backend = nullptr;
    int64_t                 h_rows_n       = 0;
    bool                    h_rows_ok      = false;

    // this context's hidden-state input read from another's device rows (cparams.mtp_h_src points here)
    llama_mtp_h_src mtp_h_src;

    // pipeline stages: the MTP draft fed per ubatch (llama_set_nextn_rows_cb); nextn_rows_want: the unmasked nextn rows
    // the host output buffer holds (-1: n_batch, without the cb or the stages)
    llama_nextn_rows_cb nextn_cb      = nullptr;
    void *              nextn_cb_user = nullptr;
    bool                nextn_cb_on   = false;   // the current decode's batch is consumed by the cb
    int64_t             nextn_rows_want = -1;

    // LLAMA_MTP_ROWS_ASYNC=1 (one device, the cb on): each ubatch's nextn rows copied on the device into this staging
    // buffer (one ubatch of rows, borrowed from the expert cache like the compute buffers), the draft reading them there
    // behind h_stage_ready while the target goes on; the next copy waits on the GPU for h_stage_done (the draft's reads of
    // the previous ones). Kept over the prompt's decodes, released (after h_stage_done) at the first decode the cb does not
    // take
    ggml_context_ptr        h_stage_ctx;
    ggml_backend_buffer_ptr h_stage_buf;
    ggml_tensor *           h_stage         = nullptr;
    ggml_backend_t          h_stage_backend = nullptr;
    ggml_backend_event_t    h_stage_ready   = nullptr;
    ggml_backend_event_t    h_stage_done    = nullptr;
    bool                    h_stage_busy    = false;   // h_stage_done recorded behind reads not waited for yet
    std::vector<float>      h_stage_last;
    void h_stage_release();

    // tokens of the decode call still to come after the current ubatch: the KV rows they will need are taken from the MoE
    // expert cache with the current ubatch's (process_ubatch)
    int64_t kv_rows_ahead = 0;

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;
        // the samplers at the last scheduler reservation: setting one of them again needs no new one
        std::map<llama_seq_id, llama_sampler *> samplers_reserved;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    size_t sched_max_nodes = 0; // the size of sched, see sched_reserve
    void * moe_cache_shared = nullptr; // the MoE cache session of another context, see share_moe_cache
    std::vector<const ggml_tensor *> moe_cache_guest_weights; // a sharing context's expert weights, see share_moe_cache

    bool sched_need_reserve = true;
    bool sched_measure_only = false; // sched_reserve splits the graphs (kernel needs recorded) and allocates nothing

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // the graphs of small ubatches (decoding, speculative verification and drafting), one per shape, kept built
    // with their split and allocation (ggml_backend_sched_plan_*): a change of shape replays the graph of the new
    // shape instead of building, splitting and allocating it again; most recently used first
    std::vector<std::pair<uint32_t, llm_graph_result_ptr>> gf_res_small;
    llm_graph_result * get_gf_res_small(const llama_ubatch & ubatch);
    void reset_graph_results();

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

public:
    // llama_set_seq_snapshots / llama_get_seq_snapshot
    void set_seq_snapshots(llama_seq_id seq_id, const llama_pos * pos, int32_t n_pos, llama_state_seq_flags flags);

    // llama_state_seq_capture / llama_state_seq_capture_data
    size_t state_seq_capture(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_capture_data(uint8_t * dst, size_t size);
    size_t get_seq_snapshot(llama_seq_id seq_id, llama_pos pos, uint8_t * dst, size_t size);

private:
    // capture the requested snapshots that end with this ubatch (decode)
    void seq_snapshots_capture(const llama_ubatch & ubatch);

    struct seq_snapshot {
        llama_pos pos = -1;
        bool      taken = false;
        size_t    size = 0;
        ggml_backend_buffer_ptr buf; // pinned host memory: the serialized state, filled by copies on the streams
    };
    llama_seq_id snap_seq = -1;
    llama_state_seq_flags snap_flags = 0;
    std::vector<seq_snapshot> snaps;

    ggml_backend_buffer_ptr capture_buf;   // llama_state_seq_capture: pinned, kept from a capture to the next
    size_t capture_size = 0;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
    int64_t ut_after_us = 0;      // LLAMA_UBATCH_TIMING: when the previous timed ubatch ended
};
