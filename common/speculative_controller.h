#pragma once

// Architecture-independent draft-length policy used by speculative methods.
// This class intentionally has no llama/ggml/backend dependencies so policy
// can be tested independently of model loading and device execution.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

class common_speculative_draft_controller {
public:
    static constexpr float DEFAULT_EMA_ALPHA = 0.25f;
    static constexpr float DEFAULT_FULL_ACCEPT_PROBE = 1.0f;
    static constexpr float DEFAULT_INITIAL_ESTIMATE = 2.0f;

    explicit common_speculative_draft_controller(
            uint32_t n_seq,
            float alpha = DEFAULT_EMA_ALPHA,
            float full_accept_probe = DEFAULT_FULL_ACCEPT_PROBE,
            float initial_estimate = DEFAULT_INITIAL_ESTIMATE)
        : alpha_(alpha),
          full_accept_probe_(full_accept_probe),
          initial_estimate_(initial_estimate),
          estimates_(n_seq, initial_estimate),
          last_draft_(n_seq, 0) {
    }

    void set_enabled(bool enabled) {
        enabled_ = enabled;
    }

    bool enabled() const {
        return enabled_;
    }

    void reset(int32_t seq_id) {
        if (!valid_seq(seq_id)) {
            return;
        }
        estimates_[seq_id] = initial_estimate_;
        last_draft_[seq_id] = 0;
    }

    // A full acceptance is a censored observation: the target may have
    // accepted more tokens had the draft been longer. Probe upward by one.
    // A partial acceptance is an exact stopping point and updates the EMA.
    void observe_acceptance(int32_t seq_id, uint32_t n_accepted) {
        if (!valid_seq(seq_id)) {
            return;
        }

        const int32_t drafted = last_draft_[seq_id];
        if (drafted > 0 && n_accepted >= static_cast<uint32_t>(drafted)) {
            estimates_[seq_id] += full_accept_probe_;
        } else {
            estimates_[seq_id] = (1.0f - alpha_) * estimates_[seq_id]
                               + alpha_ * static_cast<float>(n_accepted);
        }
        last_draft_[seq_id] = 0;
    }

    // Preserve the existing policy semantics: disabled/invalid sequence returns
    // configured maximum; adaptive mode is always clamped to at least one.
    int32_t choose_draft_length(int32_t seq_id, int32_t configured_max, int32_t configured_min) {
        if (!enabled_ || !valid_seq(seq_id)) {
            return configured_max;
        }

        const int32_t wanted = static_cast<int32_t>(std::lround(estimates_[seq_id]));
        const int32_t selected = std::max(
                std::max(1, configured_min),
                std::min(configured_max, wanted));

        estimates_[seq_id] = std::min(estimates_[seq_id], static_cast<float>(configured_max));
        last_draft_[seq_id] = selected;
        return selected;
    }

    float acceptance_estimate(int32_t seq_id) const {
        return valid_seq(seq_id) ? estimates_[seq_id] : 0.0f;
    }

    int32_t last_draft_length(int32_t seq_id) const {
        return valid_seq(seq_id) ? last_draft_[seq_id] : 0;
    }

private:
    bool valid_seq(int32_t seq_id) const {
        return seq_id >= 0 && static_cast<size_t>(seq_id) < estimates_.size();
    }

    const float alpha_;
    const float full_accept_probe_;
    const float initial_estimate_;
    bool enabled_ = false;
    std::vector<float> estimates_;
    std::vector<int32_t> last_draft_;
};
