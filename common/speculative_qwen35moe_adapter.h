#pragma once

#include "llama.h"

#include <cstdint>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// Layer-2 architecture inspection for the Qwen3.5-MoE single-head MTP path.
// This adapter describes model semantics and validates the loaded model; it
// does not execute graphs or perform backend transfers.
namespace syclonith::speculative {

struct Qwen35MoEMTPAdapterInfo {
    std::string architecture;
    int32_t hidden_size = 0;
    int32_t nextn_layers = 0;
    std::string tensor_prefix;
    std::vector<std::string> required_tensors;
};

class Qwen35MoEMTPAdapter {
public:
    static constexpr const char * ID = "qwen35moe_mtp";
    static constexpr const char * ARCHITECTURE = "qwen35moe";

    // Returns true only when target/draft metadata and the loaded MTP tensors
    // satisfy the currently verified Qwen35MoE single-head contract.
    static bool inspect(
            const llama_model * target,
            const llama_model * draft,
            Qwen35MoEMTPAdapterInfo & info,
            std::string & reason);
};


// Runtime state for the Qwen35MoE single-head MTP contract. This owns the
// sequence-local hidden-state frontier and verification rows. It deliberately
// does not perform backend copies or graph execution; callers pass host-visible
// rows after the backend has made them available.
class Qwen35MoEMTPRuntime {
public:
    Qwen35MoEMTPRuntime(uint32_t n_sequences, int32_t hidden_size)
        : hidden_size_(hidden_size),
          pending_(n_sequences, std::vector<float>(static_cast<size_t>(hidden_size), 0.0f)),
          verification_(n_sequences),
          verification_rows_(n_sequences, 0) {
    }

    int32_t hidden_size() const { return hidden_size_; }
    uint32_t n_sequences() const { return static_cast<uint32_t>(pending_.size()); }

    bool valid_sequence(llama_seq_id seq_id) const {
        return seq_id >= 0 && static_cast<size_t>(seq_id) < pending_.size();
    }

    float * pending_data(llama_seq_id seq_id) {
        return valid_sequence(seq_id) ? pending_[seq_id].data() : nullptr;
    }

    const float * pending_data(llama_seq_id seq_id) const {
        return valid_sequence(seq_id) ? pending_[seq_id].data() : nullptr;
    }

    // Start a new generation on a sequence. The pending frontier is retained:
    // prefill has already produced the hidden row needed by the first draft.
    void begin_sequence(llama_seq_id seq_id) {
        if (!valid_sequence(seq_id)) return;
        verification_[seq_id].clear();
        verification_rows_[seq_id] = 0;
    }

    // Allocate storage for the target rows retained for verifier acceptance.
    float * prepare_verification(llama_seq_id seq_id, int32_t n_rows) {
        if (!valid_sequence(seq_id) || n_rows <= 0) return nullptr;
        verification_[seq_id].resize(static_cast<size_t>(n_rows) * hidden_size_);
        verification_rows_[seq_id] = n_rows;
        return verification_[seq_id].data();
    }

    // Publish captured rows and advance the frontier to the last observed row.
    bool finish_verification(llama_seq_id seq_id) {
        if (!valid_sequence(seq_id) || verification_rows_[seq_id] <= 0) return false;
        const size_t offset = static_cast<size_t>(verification_rows_[seq_id] - 1) * hidden_size_;
        std::memcpy(pending_[seq_id].data(), verification_[seq_id].data() + offset,
                static_cast<size_t>(hidden_size_) * sizeof(float));
        return true;
    }

    int32_t verification_rows(llama_seq_id seq_id) const {
        return valid_sequence(seq_id) ? verification_rows_[seq_id] : 0;
    }

    // Select the hidden row associated with the last accepted token. The
    // caller separately trims deferred token/KV work to the same frontier.
    int32_t accepted_row(llama_seq_id seq_id, uint16_t n_accepted) const {
        const int32_t n_rows = verification_rows(seq_id);
        return n_rows > 0 ? std::min<int32_t>(n_accepted, n_rows - 1) : -1;
    }

    bool accept(llama_seq_id seq_id, uint16_t n_accepted) {
        const int32_t row = accepted_row(seq_id, n_accepted);
        if (row < 0) return false;
        const size_t offset = static_cast<size_t>(row) * hidden_size_;
        std::memcpy(pending_[seq_id].data(), verification_[seq_id].data() + offset,
                static_cast<size_t>(hidden_size_) * sizeof(float));
        return true;
    }

    // Qwen35MoE pairs the first token for a sequence with the previous
    // frontier; later tokens pair with the preceding target row in the batch.
    const float * paired_target_row(
            llama_seq_id seq_id, int32_t batch_index, int32_t sequence_begin,
            const float * target_rows) const {
        if (!valid_sequence(seq_id) || batch_index < 0 || sequence_begin < 0 || batch_index < sequence_begin) {
            return nullptr;
        }
        return batch_index == sequence_begin
            ? pending_[seq_id].data()
            : target_rows + static_cast<size_t>(batch_index - 1) * hidden_size_;
    }

private:
    int32_t hidden_size_;
    std::vector<std::vector<float>> pending_;
    std::vector<std::vector<float>> verification_;
    std::vector<int32_t> verification_rows_;
};

} // namespace syclonith::speculative
