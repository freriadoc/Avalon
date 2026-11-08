#!/bin/bash
# ringsim on the pod (RTX 4090), ~5 min: from /root/avalon-test, ringsim.cu and route-81k-ub9536.bin in ./ringsim/.
# Nothing else on the GPU meanwhile.
cd /root/avalon-test/ringsim || exit 1
nvcc -O2 -std=c++17 -arch=sm_89 -o ringsim ringsim.cu -lpthread || exit 1
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm --format=csv,noheader
D=route-81k-ub9536.bin
echo "=== A. calibration: ungrouped (G = 512), its compute against the 4090 trace (a layer at 9536: dense 34.3 ms + MoE 12.7)"
timeout 60 ./ringsim --dump $D --passes 1 --G 512 --ring 1800 --issuer block --sync 1
echo "=== B. 9536 tokens, sync per block (the real one): group size x ring x issuer x waits per copy (0) / per group (1)"
timeout 170 ./ringsim --dump $D --passes 1 --G 8,16,32,64 --ring 900,1800 --issuer block,thread,hthread --wait1 0,1 --sync 1
echo "=== C. 9536 tokens without the synchronization"
timeout 40 ./ringsim --dump $D --passes 1 --G 32 --ring 900,1800 --issuer block,ahead --wait1 1 --sync 0
echo "=== D. 28608 tokens (3 passes merged)"
timeout 120 ./ringsim --dump $D --merge 3 --passes 1 --G 16,32,64 --ring 900,1800 --issuer block,thread --wait1 1 --sync 1
