# Qwen35MoE MTP adapter: first integration slice

This change introduces a Layer-2 structural inspector for the Qwen35MoE MTP
contract and selects it through an explicit, deterministic architecture
registry when the existing MTP speculative implementation is constructed.

## What it checks

- Both target and draft GGUF metadata report `general.architecture=qwen35moe`.
- Target and draft output embedding widths match.
- The draft model reports exactly one `nextn` prediction layer.
- The loaded draft model contains one matching MTP tensor group:
  - `*.nextn.eh_proj.weight`
  - `*.nextn.enorm.weight`
  - `*.nextn.hnorm.weight`
  - `*.nextn.shared_head_norm.weight`
- The projection and normalization tensor dimensions agree with the output
  embedding width.

The inspector uses the model's loaded tensor map, not just GGUF metadata. It
therefore validates the tensors that the runtime actually loaded. The registry
currently registers only `qwen35moe` and deliberately has no implicit static
registration or ordering.

## What it does not claim

This is a structural compatibility check, not runtime qualification. In
particular, it does not prove hidden-state row pairing, sequence lifecycle,
KV-cache correctness, acceptance behavior, or SYCL/backend transfer correctness
for Sage or Ornith. Those still require runtime tests with each model.

The registry selects an architecture inspector, not a new graph executor.
The existing `common_speculative_impl_draft_mtp` remains the executor, and
selection does not change draft behavior. Unknown architectures are reported
as unregistered; a registered adapter that rejects the model pair produces a
warning. In both cases, the existing MTP path is retained unchanged. This
avoids regressing other architectures while the Layer-2/Layer-3 migration is
incomplete.

## Build and test

The controller test is standalone and can be built with:

```bash
cmake --build build --target test_speculative_controller
ctest --test-dir build -R '^test_speculative_controller$' --output-on-failure
```

The two changed speculative translation units should also compile as part of
the normal `llama-common` build. Runtime qualification should then exercise
prompt prefill, short and long generation, multiple sequences, cancellation /
slot reuse, and repeated runs on the actual SYCL backend.
