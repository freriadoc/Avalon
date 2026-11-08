# SM80 chunked GDN prefill

`GGML_CUDA_GDN_SM80_CHUNKED=1` enables an opt-in tensor-core prefill path. The
variable defaults off. The existing recurrent/WMMA dispatch handles unsupported
calls. No separate research kernel or environment variable is required.

The supported shape has head size 128, 16 Q/K heads, 48 value heads, one sequence,
scalar decay, K=1 final state and at least 128 tokens. The device must have compute
capability exactly 8.0, at least 48 SMs and 124,944 bytes of opt-in shared memory
per block. Q/K/V base addresses and row strides must permit 16-byte input copies.
A failed CUDA function-attribute setup declines before state mutation. A launch
failure after mutation remains fatal, as in the existing backend.

The scan processes 32-token chunks within 2048-token superblocks. It splits scaled
FP16 operands into high/residual parts, evaluates three tensor products, and keeps
state and accumulation in FP32. Input convolution, normalization, gate activation,
output scaling and the external graph contract remain unchanged.

A transpose adapts the existing state layout before the scan, then returns the
final state to the existing output/cache destination. Gathered state and accepted
MTP prefixes are materialized with the original recurrence first, on the same
stream. Short deferred replay windows retain the existing path. Scratch is
reserved by the scheduler or owned by the context's pool, never a global device
buffer. Its maximum size is 63,451,136 bytes per active node. Allocation failure
uses the backend's existing fatal allocation behavior.

The CPU-reference tests cover strided inputs, nonzero initial state, weak decay,
chunk tails, multiple superblocks, accepted replay prefixes, state gathers and
fused cache writes. Unsupported head ratios, KDA, multiple sequences and snapshot
counts retain fallback coverage. Existing numerical tolerances are unchanged.
Run GATED_DELTA_NET backend comparisons with the flag unset and set before
full-model evaluation. Native timing fixtures include accepted MTP prefixes.

Source attribution and the full third-party MIT license are in
[sm80-gdn.txt](../third-party/sm80-gdn.txt). Retain that notice in distributions.
