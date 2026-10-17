#!/usr/bin/env python3
# check-packed-experts.py SRC_DIR PACKED_DIR [N] : the packed model against its source: every other tensor (name, type, shape,
# bytes: a hash of the whole data), every layer's metadata, and the three roles and the pads of N random experts per layer
import sys, os, re, glob, hashlib, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
from gguf import GGUFReader
src, dst = sys.argv[1], sys.argv[2]
n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 16
S = {t.name: t for f in sorted(glob.glob(src + '/*.gguf')) for t in GGUFReader(f).tensors}
R = [GGUFReader(f) for f in sorted(glob.glob(dst + '/*.gguf'))]
D = {t.name: t for r in R for t in r.tensors}
bad = 0
for name, t in S.items():
    if re.search(r'_exps\.weight$', name) and 'ffn_' in name and 'chexps' not in name:
        continue
    u = D.get(name)
    if u is None or u.tensor_type != t.tensor_type or list(u.shape) != list(t.shape) or \
       hashlib.sha1(np.asarray(u.data).tobytes()).digest() != hashlib.sha1(np.asarray(t.data).tobytes()).digest():
        print('MISMATCH', name); bad += 1
rng = random.Random(1)
for key, field in R[0].fields.items():
    m = re.match(r'avalon\.experts_packed\.blk\.(\d+)$', key)
    if not m:
        continue
    layer = int(m.group(1)); v = [int(x) for x in field.contents()]
    stride, n_expert = v[0], v[1]
    p = np.asarray(D[f'blk.{layer}.ffn_exps_packed'].data).view(np.uint8).reshape(n_expert, stride)
    for k, role in enumerate(('up', 'gate', 'down')):
        typ, ne0, ne1, off = v[2 + 4*k: 6 + 4*k]
        t = S[f'blk.{layer}.ffn_{role}_exps.weight']
        assert int(t.tensor_type) == typ and int(t.shape[0]) == ne0 and int(t.shape[1]) == ne1
        nb = t.n_bytes // n_expert
        a = np.asarray(t.data).reshape(-1).view(np.uint8).reshape(n_expert, nb)
        for e in [0, n_expert - 1] + [rng.randrange(n_expert) for _ in range(n_check)]:
            if not np.array_equal(p[e, off:off + nb], a[e]):
                print('EXPERT MISMATCH', layer, role, e); bad += 1
            nxt = off + nb
            pad_end = v[2 + 4*(k+1) + 3] if k < 2 else stride
            pad = pad_end - nxt
            if pad > 0:
                ref = a[e + 1, :pad] if e + 1 < n_expert else a[e, :pad]
                if not np.array_equal(p[e, nxt:nxt + pad], ref):
                    print('PAD MISMATCH', layer, role, e); bad += 1
print(f'{len(S)} source tensors, {len(D)} packed tensors, {bad} mismatches')
