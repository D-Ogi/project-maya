"""Reference data for the Maya LSA kernels (lsa_test.cu): random unit index keys and queries, the repo's own torch oracle
(lsa.core.reference.build_buckets / read) run on them, everything written to one binary file the test reads.

Usage:  python -I lsa_oracle.py <repo src dir> <out.bin> [--keys 8192] [--queries 64] [--seed 1] [--late-pos N]

Layout (little endian): u32 magic 'LSAT', u32 n_keys, u32 n_queries, u32 pool, u32 block, u32 buckets, u32 cap,
u32 window, u32 sink, u32 top_k, u32 top_p, u64 late_pos, u32 late_top_k, u32 late_top_p, u32 width;
f32 keys[n_keys][64]; f32 queries[n_queries][64]; i64 positions[n_queries];
f32 centroids[n_blocks * buckets][64]; i32 members[n_blocks * buckets][cap]; i32 selected[n_queries][width];
i32 lengths[n_queries].
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_src")
    ap.add_argument("out")
    ap.add_argument("--keys", type=int, default=8192)
    ap.add_argument("--queries", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--late-pos", type=int, default=1 << 20)
    ap.add_argument("--clustered", action="store_true", help="keys drawn around a few centers (ties and near-ties)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo_src).resolve()))
    from lsa.budget import EarlyBudget
    from lsa.core.reference import build_buckets, read

    pool, block, buckets, cap, window, sink = 4, 512, 64, 16, 512, 4
    # the design's budget (late) and the early budget below late_pos - the engine's LsaBudget names the early one
    # top_k / top_p and the design's late_top_k / late_top_p
    late_k, late_p = 4096, 1024
    top_k, top_p = 2048, 512
    early = EarlyBudget(position=a.late_pos, top_k=top_k, top_p=top_p)
    g = torch.Generator().manual_seed(a.seed)
    if a.clustered:
        centers = torch.randn(32, 64, generator=g)
        pick = torch.randint(0, 32, (a.keys,), generator=g)
        keys = centers[pick] + 0.3 * torch.randn(a.keys, 64, generator=g)
    else:
        keys = torch.randn(a.keys, 64, generator=g)
    keys = torch.nn.functional.normalize(keys, dim=-1, eps=1e-12).float().contiguous()
    queries = torch.nn.functional.normalize(torch.randn(a.queries, 64, generator=g), dim=-1, eps=1e-12).float()
    # positions spread over the context, the last ones at the end (every block closed), a few early (no block)
    positions = torch.sort(torch.randint(0, a.keys, (a.queries,), generator=g))[0].to(torch.int64)
    positions[-1] = a.keys - 1
    positions[0] = 3
    positions[1] = 511
    positions[2] = 512
    b = build_buckets(keys, pool, 8, cap, block)
    selected, lengths = read(queries, positions, b, late_p, late_k, sink, window, early_budget=early)
    width = late_k + window + sink
    assert selected.shape == (a.queries, width)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<IIIIIIIIIIIQIII", 0x5441534C, a.keys, a.queries, pool, block, buckets, cap, window, sink,
                            top_k, top_p, a.late_pos, late_k, late_p, width))
        f.write(keys.numpy().astype("<f4").tobytes())
        f.write(queries.numpy().astype("<f4").tobytes())
        f.write(positions.numpy().astype("<i8").tobytes())
        f.write(b.bucket_centroids.numpy().astype("<f4").tobytes())
        f.write(b.bucket_members.numpy().astype("<i4").tobytes())
        f.write(selected.numpy().astype("<i4").tobytes())
        f.write(lengths.numpy().astype("<i4").tobytes())
    n_blocks = a.keys // block
    print(f"wrote {a.out}: {a.keys} keys ({n_blocks} blocks, {n_blocks * buckets} buckets), {a.queries} queries, "
          f"width {width}, lengths {lengths.min().item()}..{lengths.max().item()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
