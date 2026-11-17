#include "../common/speculative_controller.h"

#include <cassert>
#include <cmath>

int main() {
    common_speculative_draft_controller controller(2);

    // Fixed-length mode preserves the configured maximum.
    assert(controller.choose_draft_length(0, 6, 0) == 6);

    controller.set_enabled(true);

    // Initial estimate and per-sequence isolation.
    assert(controller.choose_draft_length(0, 6, 0) == 2);
    assert(controller.choose_draft_length(1, 6, 0) == 2);

    // Full acceptance is censored and probes upward by one.
    controller.observe_acceptance(0, 2);
    assert(std::fabs(controller.acceptance_estimate(0) - 3.0f) < 0.0001f);
    assert(controller.choose_draft_length(0, 6, 0) == 3);

    // Partial acceptance updates the EMA.
    controller.observe_acceptance(0, 1);
    assert(std::fabs(controller.acceptance_estimate(0) - 2.5f) < 0.0001f);

    // Minimum draft length is clamped to at least one.
    controller.reset(0);
    assert(controller.choose_draft_length(0, 6, 0) == 2);
    assert(controller.choose_draft_length(0, 6, 1) == 2);

    // Invalid sequence IDs fall back to the configured maximum.
    assert(controller.choose_draft_length(-1, 6, 0) == 6);
    assert(controller.choose_draft_length(2, 6, 0) == 6);
}
