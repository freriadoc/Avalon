#pragma once
// NVIDIA Volta+ (mmq-config-ampere.cuh): the MMQ config of the types Avalon's models use (Q8_0, IQ2_S, IQ3_K, IQ3_KS,
// IQ4_NL) in one place, to try others on a rented card: threads per block, blocks per SM (launch bounds), rows per tile,
// stream-k decomposition. Stream-k off: 19k prefill on an RTX 4090 2 414 - 2 429 -> 2 688 - 2 700 t/s (+11 %); the
// other combinations measured there: 2 blocks per SM -4 %, (128, 2, 64) +0 %, with stream-k off +1 %.
#ifndef AVL_MMQ_NTH
#define AVL_MMQ_NTH 256
#endif
#ifndef AVL_MMQ_OCC
#define AVL_MMQ_OCC 1
#endif
#ifndef AVL_MMQ_I
#define AVL_MMQ_I   128
#endif
#ifndef AVL_MMQ_SK
#define AVL_MMQ_SK  false
#endif
