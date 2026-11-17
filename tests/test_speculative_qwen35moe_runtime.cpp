#include "speculative_qwen35moe_adapter.h"

#include <cassert>
#include <cmath>

using syclonith::speculative::Qwen35MoEMTPRuntime;

int main() {
    Qwen35MoEMTPRuntime runtime(/*n_sequences*/ 2, /*hidden_size*/ 3);

    assert(runtime.n_sequences() == 2);
    assert(runtime.hidden_size() == 3);
    assert(runtime.valid_sequence(0));
    assert(!runtime.valid_sequence(-1));
    assert(!runtime.valid_sequence(2));

    // The first token in a sequence pairs with its previous frontier; later
    // tokens pair with the immediately preceding target output row.
    float target_rows[] = {
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
    };
    float * pending0 = runtime.pending_data(0);
    pending0[0] = 1.0f;
    pending0[1] = 2.0f;
    pending0[2] = 3.0f;

    const float * first = runtime.paired_target_row(0, 2, 2, target_rows);
    assert(first[0] == 1.0f && first[2] == 3.0f);
    const float * second = runtime.paired_target_row(0, 3, 2, target_rows);
    assert(second[0] == 30.0f && second[2] == 32.0f);

    // Capture verifier rows. The last observed row becomes the next frontier.
    float * verify = runtime.prepare_verification(0, 3);
    assert(verify != nullptr);
    const float rows[] = {
        100.0f, 101.0f, 102.0f,
        200.0f, 201.0f, 202.0f,
        300.0f, 301.0f, 302.0f,
    };
    for (int i = 0; i < 9; ++i) verify[i] = rows[i];
    assert(runtime.finish_verification(0));
    assert(runtime.pending_data(0)[0] == 300.0f);
    assert(runtime.verification_rows(0) == 3);

    // Acceptance selects the hidden row associated with the last accepted token.
    assert(runtime.accepted_row(0, 0) == 0);
    assert(runtime.accepted_row(0, 1) == 1);
    assert(runtime.accepted_row(0, 99) == 2);
    assert(runtime.accept(0, 1));
    assert(runtime.pending_data(0)[0] == 200.0f);
    assert(runtime.pending_data(0)[2] == 202.0f);

    // Sequence state is isolated, and begin clears verifier rows but retains
    // the frontier produced by prefill for the next generation.
    assert(runtime.pending_data(1)[0] == 0.0f);
    runtime.begin_sequence(0);
    assert(runtime.verification_rows(0) == 0);
    assert(runtime.pending_data(0)[0] == 200.0f);
    assert(runtime.accept(0, 0) == false);
}
