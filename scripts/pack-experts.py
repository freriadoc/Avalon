#!/usr/bin/env python3
# pack-experts.py SRC_DIR DST_DIR : rewrites a split GGUF model so that each MoE layer's experts are stored one after
# the other, each expert as the MoE cache lays an entry out (up | gate | down, each at a multiple of 256 bytes and
# followed by its pad), in one I8 tensor per layer, blk.N.ffn_exps_packed, of shape [stride, n_expert]. A fill of the
# expert cache then copies one contiguous range instead of three. The three tensors ffn_{up,gate,down}_exps are no
# longer in the file: the loader (fred/master) builds them as views of the packed tensor from the key
# avalon.experts_packed.blk.N of the first split, an array of 14 uint64:
#   stride, n_expert, then per role (up, gate, down): ggml type, ne0, ne1, offset in the entry.
# A pad (the MoE cache's moe_cache_role_pad: what MMQ reads past a row when ne0 is not a multiple of 512) holds the
# next expert's first bytes of that tensor (its own first bytes for the last expert), as the cache copies them today.
# Every other tensor and key is copied as is; a layer's packed tensor goes into the split of its first expert tensor.
import sys, os, re, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGML_QUANT_SIZES, GGMLQuantizationType

ALIGN = 256
ROLES = ('up', 'gate', 'down')   # the MoE cache's MOE_ROLE_UP, _GATE, _DOWN
SKIP_KEYS = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.architecture'}


def role_pad(qtype, ne0):
    blck, size = GGML_QUANT_SIZES[GGMLQuantizationType(qtype)]
    over = ((ne0 + 511) // 512 * 512 - ne0) // blck * size
    return (over + ALIGN - 1) // ALIGN * ALIGN


def main(src, dst):
    files = sorted(glob.glob(os.path.join(src, '*.gguf')))
    readers = [GGUFReader(f) for f in files]
    # the expert tensors of each layer, and the split of the first one
    exps = {}
    for i, r in enumerate(readers):
        for t in r.tensors:
            m = re.match(r'blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$', t.name)
            if m:
                exps.setdefault(int(m.group(1)), {})[m.group(2)] = (i, t)
    layouts = {}
    for layer, roles in exps.items():
        assert set(roles) == set(ROLES), f'layer {layer}: {sorted(roles)}'
        meta, off = [], 0
        n_expert = None
        for role in ROLES:
            t = roles[role][1]
            ne0, ne1, ne2 = (int(x) for x in t.shape[:3])
            assert n_expert in (None, ne2)
            n_expert = ne2
            nbytes = t.n_bytes // ne2
            pad = role_pad(int(t.tensor_type), ne0)
            meta.append((int(t.tensor_type), ne0, ne1, off, nbytes, pad))
            off += (nbytes + ALIGN - 1) // ALIGN * ALIGN + pad
        layouts[layer] = (off, n_expert, meta, min(i for i, _ in roles.values()))
    n_tensors = sum(len(r.tensors) for r in readers) - 2 * len(layouts)
    os.makedirs(dst, exist_ok=True)
    for i, r in enumerate(readers):
        out = os.path.join(dst, os.path.basename(files[i]))
        w = GGUFWriter(out, arch=readers[0].fields['general.architecture'].contents(), use_temp_file=False)
        if 'general.architecture' not in r.fields:   # only the first split has it
            for kv in w.kv_data:
                kv.pop('general.architecture', None)
        for name, f in r.fields.items():
            if name in SKIP_KEYS:
                continue
            vtype = f.types[0]
            val = f.contents()
            if name == 'split.tensors.count':
                val = n_tensors
            if vtype == GGUFValueType.ARRAY:
                w.add_key_value(name, val, vtype, sub_type=f.types[1])
            else:
                w.add_key_value(name, val, vtype)
        if i == 0:
            for layer, (stride, n_expert, meta, _) in sorted(layouts.items()):
                vals = [stride, n_expert]
                for (qtype, ne0, ne1, off, _, _) in meta:
                    vals += [qtype, ne0, ne1, off]
                w.add_key_value(f'avalon.experts_packed.blk.{layer}', vals, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
        for t in r.tensors:
            m = re.match(r'blk\.(\d+)\.ffn_(up|gate|down)_exps\.weight$', t.name)
            if not m:
                w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
                continue
            layer = int(m.group(1))
            stride, n_expert, meta, home = layouts[layer]
            if home != i or m.group(2) != next(ro for ro in ROLES if exps[layer][ro][0] == i):
                continue   # the packed tensor is written once, with the first expert tensor of its split
            packed = np.zeros((n_expert, stride), dtype=np.uint8)
            for role, (qtype, ne0, ne1, off, nbytes, pad) in zip(ROLES, meta):
                src_t = exps[layer][role][1]
                data = np.asarray(src_t.data).reshape(-1).view(np.uint8).reshape(n_expert, nbytes)
                packed[:, off:off + nbytes] = data
                if pad:
                    packed[:-1, off + nbytes:off + nbytes + pad] = data[1:, :pad]
                    packed[-1, off + nbytes:off + nbytes + pad] = data[-1, :pad]
            w.add_tensor(f'blk.{layer}.ffn_exps_packed', packed.view(np.int8), raw_dtype=GGMLQuantizationType.I8)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=False)
        w.close()
        print(f'{os.path.basename(out)}: {len(r.tensors)} tensors read', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
