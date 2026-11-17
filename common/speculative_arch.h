#pragma once

// Experimental architecture boundary for Syclonith speculative decoding.
//
// This is an interface proposal, not yet wired into common_speculative_init()
// or the server request loop. Keep the existing implementation operational
// until an adapter has been migrated and validated.

#include "common.h"
#include "llama.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace syclonith::speculative {

// -----------------------------------------------------------------------------
// Generic controller <-> adapter contract
// -----------------------------------------------------------------------------

struct SequenceId {
    llama_seq_id value = -1;
};

struct DraftRequest {
    llama_seq_id seq_id = -1;
    llama_pos   pos0 = 0;
    llama_token id_last = LLAMA_TOKEN_NULL;

    // The prompt is owned by the caller and remains valid for this call.
    const llama_tokens * prompt = nullptr;

    // Controller-selected limit. The adapter may reduce it only when required
    // by an architecture capability (e.g. number of trained prediction heads).
    int32_t n_draft = 0;

    float    temperature = 1.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
};

struct DraftResult {
    std::vector<llama_token> tokens;

    // Optional candidate distributions. Empty when the adapter/backend samples
    // on-device or when the controller does not need q distributions.
    std::vector<std::vector<llama_token_data>> candidates;

    void clear() {
        tokens.clear();
        candidates.clear();
    }
};

struct AcceptanceFeedback {
    llama_seq_id seq_id = -1;
    uint32_t n_drafted = 0;
    uint32_t n_accepted = 0;
    bool accepted_other = false;
};

struct AdapterCapabilities {
    int32_t max_draft_tokens = 1;
    bool supports_multiple_sequences = false;
    bool supports_device_sampling = false;
    bool requires_draft_context = false;
    bool uses_target_hidden_states = false;
    bool shares_target_kv = false;
    bool has_multiple_prediction_heads = false;
};

// -----------------------------------------------------------------------------
// Backend execution contract
// -----------------------------------------------------------------------------

// Backend owns execution and transfer policy, not model-family interpretation.
// Architecture adapters decide WHAT must be executed/transferred; the backend
// decides HOW to perform it on SYCL, oneDNN, CPU, or another backend.
class ExecutionBackend {
public:
    virtual ~ExecutionBackend() = default;

    virtual const char * name() const = 0;

    // Execute a llama batch in the supplied context. Implementations may route
    // through llama_process() and the selected ggml backend.
    virtual int process(llama_context * ctx, common_batch & batch) = 0;

    virtual void synchronize(llama_context * ctx) = 0;

    // Generic tensor transfer primitive. The adapter supplies the tensor
    // semantics and offsets; the backend handles device placement/copy policy.
    virtual bool copy_tensor(
            ggml_tensor * src,
            ggml_backend_t src_backend,
            size_t src_offset,
            ggml_tensor * dst,
            ggml_backend_t dst_backend,
            size_t dst_offset,
            size_t n_bytes) = 0;

    virtual bool can_access(ggml_backend_t src, ggml_backend_t dst) const = 0;
};

// -----------------------------------------------------------------------------
// Architecture adapter contract
// -----------------------------------------------------------------------------

struct AdapterInit {
    llama_model   * target_model = nullptr;
    llama_context * target_ctx = nullptr;

    // May be null for self-contained target-model prediction heads.
    llama_model   * draft_model = nullptr;
    llama_context * draft_ctx = nullptr;

    uint32_t n_sequences = 1;
    common_params_speculative params;
    ExecutionBackend * backend = nullptr;
};

class ArchitectureAdapter {
public:
    virtual ~ArchitectureAdapter() = default;

    virtual const char * id() const = 0;
    virtual const char * architecture() const = 0;

    // Must perform all compatibility checks before allocating persistent state.
    // Return false and explain why in reason when the model is unsupported.
    virtual bool supports(
            const llama_model * target,
            const llama_model * draft,
            std::string & reason) const = 0;

    virtual AdapterCapabilities capabilities() const = 0;

    virtual bool initialize(const AdapterInit & init, std::string & reason) = 0;

    // Called once for a new generation/request on this sequence.
    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    // Called as target batches are processed. Adapters that do not need target
    // observations should return true without doing work.
    virtual bool observe_target_batch(const common_batch & batch) = 0;

    // Generate up to request.n_draft tokens. The adapter must not independently
    // change controller policy; architecture-imposed limits belong in
    // capabilities() and are enforced by the controller before this call.
    virtual bool draft(const DraftRequest & request, DraftResult & result) = 0;

    virtual void accept(const AcceptanceFeedback & feedback) = 0;

    // Optional per-sequence checkpoint support.
    virtual bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const {
        (void) seq_id;
        (void) data;
        return false;
    }

    virtual void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) {
        (void) seq_id;
        (void) data;
    }
};

// -----------------------------------------------------------------------------
// Generic policy/controller contract
// -----------------------------------------------------------------------------

struct ControllerConfig {
    int32_t n_draft_min = 1;
    int32_t n_draft_max = 6;

    // Adaptive policy parameters are generic. They must not depend on model
    // architecture or backend implementation.
    bool adaptive = false;
    float acceptance_ema_alpha = 0.25f;
    float initial_acceptance_estimate = 2.0f;
};

struct ControllerDecision {
    int32_t n_draft = 1;
    const char * strategy = "fixed";
};

class SpeculativeController {
public:
    virtual ~SpeculativeController() = default;

    virtual void reset(llama_seq_id seq_id) = 0;

    virtual ControllerDecision choose(
            llama_seq_id seq_id,
            int32_t configured_max,
            int32_t adapter_max) = 0;

    virtual void observe(const AcceptanceFeedback & feedback) = 0;
};

// The factory registry will own adapter construction. Registration should be
// explicit and deterministic; do not use static initialisers with hidden order.
using AdapterFactory = std::unique_ptr<ArchitectureAdapter> (*)();

} // namespace syclonith::speculative
