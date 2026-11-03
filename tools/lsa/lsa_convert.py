"""Convert an LSA projection map (`lsa distill export`'s flat torch file) into the Maya engine's `.lsa` binary.

Usage (needs torch and numpy; run it from the repository, not from the directory the map was downloaded to):
    python -I tools/lsa/lsa_convert.py <map.pt> <out.lsa> [--top-k 2048 --top-p 512 --late-pos 1048576 --late-top-k 4096
                                               --late-top-p 1024 --pool 4 --block 512 --buckets 64 --cap 16
                                               --window 512 --sink 4]

The defaults are the GLM-5.3-Flash D-LSA Flat recipe (configs/evaluation/accuracy/glm53_flash/lsa_flat_early_budget_tp8.yaml
of Prefrontal-AI/latent-sparse-attention): index dim 64, pools of 4 keys, cluster blocks of 512 keys, 64 buckets a block
(average 8 keys, cap 16), window 512, 4 sinks, top-k 2048 / top-p 512 below position 1,048,576 and 4096 / 1024 from it.

File layout (little endian):
    u32 magic 'LSAM' (0x4D41534C), u32 version 1, u32 n_layers, u32 input_dim, u32 index_dim,
    u32 pool, u32 block, u32 buckets, u32 cap, u32 window, u32 sink,
    u32 top_k, u32 top_p, u64 late_pos, u32 late_top_k, u32 late_top_p,
    then per layer: i32 layer, f32 W_Q[input_dim][index_dim], f32 W_K[input_dim][index_dim]  (row major: input, index)
"""

import argparse
import struct
import sys

import numpy as np
import torch

MAGIC = 0x4D41534C   # 'LSAM'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--top-p", type=int, default=512)
    ap.add_argument("--late-pos", type=int, default=1048576)
    ap.add_argument("--late-top-k", type=int, default=4096)
    ap.add_argument("--late-top-p", type=int, default=1024)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--buckets", type=int, default=64)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--sink", type=int, default=4)
    a = ap.parse_args()

    blob = torch.load(a.src, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not blob:
        print("not a projection file", file=sys.stderr)
        return 1
    found: dict[int, dict[str, torch.Tensor]] = {}
    for name, t in blob.items():
        layer, _, kind = str(name).partition(".")
        if not layer.isdigit() or kind not in ("W_Q_idx", "W_K_idx") or not isinstance(t, torch.Tensor):
            print(f"unexpected entry {name!r}", file=sys.stderr)
            return 1
        found.setdefault(int(layer), {})[kind] = t.float().contiguous()
    layers = sorted(found)
    shapes = {tuple(found[l][k].shape) for l in layers for k in ("W_Q_idx", "W_K_idx") if k in found[l]}
    if len(shapes) != 1 or any(len(found[l]) != 2 for l in layers):
        print(f"inconsistent map: shapes {shapes}, layers {layers}", file=sys.stderr)
        return 1
    (input_dim, index_dim), = shapes
    if a.block % a.pool or (a.block // a.buckets) % a.pool or a.cap < 2 * (a.block // a.buckets):
        print("sizes do not nest (pool | bucket size | block, cap >= 2 * bucket size)", file=sys.stderr)
        return 1
    with open(a.dst, "wb") as f:
        f.write(struct.pack("<IIIII", MAGIC, 1, len(layers), input_dim, index_dim))
        f.write(struct.pack("<IIIIII", a.pool, a.block, a.buckets, a.cap, a.window, a.sink))
        f.write(struct.pack("<IIQII", a.top_k, a.top_p, a.late_pos, a.late_top_k, a.late_top_p))
        for l in layers:
            f.write(struct.pack("<i", l))
            f.write(np.ascontiguousarray(found[l]["W_Q_idx"].numpy(), dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(found[l]["W_K_idx"].numpy(), dtype="<f4").tobytes())
    print(f"wrote {a.dst}: layers {layers}, W [{input_dim} x {index_dim}], top_k {a.top_k} / top_p {a.top_p} "
          f"(from position {a.late_pos}: {a.late_top_k} / {a.late_top_p}), width {a.top_k + a.window + a.sink} "
          f"(late {a.late_top_k + a.window + a.sink})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
