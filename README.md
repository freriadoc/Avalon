# Syclonith

**SYCL-powered LLM inference for Intel GPUs.**

Syclonith is an experimental inference-engine project derived from [Avalon](https://github.com/freriadoc/Avalon), itself based on [llama.cpp](https://github.com/ggml-org/llama.cpp).

The project explores bringing advanced heterogeneous LLM inference optimisations to Intel GPU hardware using **Intel oneAPI, SYCL and oneDNN**.

The goal is to make efficient local LLM inference possible on Intel hardware, with particular attention to memory management, GPU utilisation, CPU/GPU cooperation and Mixture-of-Experts (MoE) workloads.

## Project goals

Syclonith aims to explore and develop:

- **Intel SYCL backend:** GPU computation using Intel's SYCL implementation and oneAPI ecosystem.
- **Heterogeneous inference:** Efficient cooperation between CPU and Intel GPU resources.
- **MoE optimisation:** Expert caching, scheduling and memory-transfer optimisations.
- **Memory efficiency:** Better utilisation of system RAM, GPU memory and shared resources.
- **Attention optimisation:** Investigating efficient attention implementations and oneDNN integration.
- **Inference performance:** Optimising token generation, prompt processing and long-context workloads.

The project builds upon the architectural work and optimisations introduced by Avalon and the broader llama.cpp ecosystem.

## Hardware and software

The primary development environment targets Intel hardware.

| Component | Target |
|---|---|
| CPU | Intel x86-64 processors |
| GPU | Intel integrated and discrete GPUs |
| GPU programming | SYCL |
| Compute ecosystem | Intel oneAPI |
| Math libraries | oneDNN |
| Operating system | Linux |
| Build system | CMake |

Actual hardware compatibility and performance depend on the GPU architecture, driver, oneAPI version and supported SYCL operations.

## Current development status

Syclonith is under active development.

| Area | Status |
|---|---|
| Intel SYCL backend | Experimental |
| oneDNN integration | Under investigation |
| CPU/GPU cooperation | Under development |
| MoE optimisation | Under development |
| Flash Attention | Experimental |
| Performance benchmarking | In progress |

Features and compatibility will be documented as they are implemented and validated.

## Building

### Requirements

- Linux
- CMake
- Intel oneAPI Base Toolkit
- Intel SYCL compiler and runtime
- Supported Intel GPU and drivers

### Build with Intel SYCL

```bash
cmake -S . -B build-sycl \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SYCL=ON

cmake --build build-sycl \
    --parallel \
    --target llama-server
```

Additional compiler and runtime configuration may be necessary depending on the Intel GPU and oneAPI installation.

Refer to the build documentation for hardware-specific configuration and troubleshooting.

## Performance

Performance measurements will be published as the implementation matures.

Benchmarks will aim to cover:

- Prompt processing throughput.
- Token generation throughput.
- GPU utilisation.
- CPU utilisation.
- Memory bandwidth and transfer overhead.
- MoE expert caching efficiency.
- Long-context inference.
- Flash Attention performance.

All benchmark results should include hardware specifications, software versions, model quantisation, context size and inference parameters to make comparisons reproducible.

## Project background and acknowledgements

Syclonith is an independent derivative project.

It builds upon work from several open-source projects and contributors:

- [Avalon](https://github.com/freriadoc/Avalon) — upstream project and inference optimisations.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — foundational inference engine.
- [GGML](https://github.com/ggml-org/llama.cpp/tree/master/ggml) — tensor computation infrastructure.
- [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) — SYCL development ecosystem.
- [oneDNN](https://github.com/uxlfoundation/oneDNN) — performance-optimised deep learning primitives.

We acknowledge the original authors and contributors whose work makes this project possible.

## Licensing

Syclonith's source code is distributed under the MIT License, retaining applicable original copyright and licence notices.

The project incorporates work originating from Avalon and llama.cpp. Please refer to the relevant upstream repositories and third-party notices for additional attribution.

Model weights and other externally distributed assets are subject to their own licences. They are not automatically covered by the project's MIT licence.

## Contributing

Contributions, experiments, performance measurements and Intel GPU compatibility reports are welcome.

The project particularly welcomes contributions around:

- SYCL kernel optimisation.
- oneDNN integration.
- Memory management.
- MoE inference.
- CPU/GPU scheduling.
- Performance benchmarking.

---

**Syclonith — exploring efficient local LLM inference on Intel hardware.**
