#!/bin/bash
# hybgpu.sh [ITERS] : builds and runs hybgpu (the flags of ggml-cuda's mmvq.cu)
set -e; cd ~/llama-fred/tests/hybgpubench; L=~/llama-fred
/opt/rocm/lib/llvm/bin/clang++ -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -I$L/ggml/src -I$L/ggml/include -O3 -DNDEBUG -std=gnu++17 \
  --offload-arch=gfx1100 -x hip hybgpu.hip -L/opt/rocm/lib -lamdhip64 -L$L/build-hip/bin -lggml-base -Wl,-rpath,$L/build-hip/bin -o hybgpu -w
[ "${BUILD_ONLY:-0}" = 1 ] || ./hybgpu ${1:-2000}
