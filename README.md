# Project Maya + Latent Sparse Attention (a fork)

This is a fork of [Project Maya](https://github.com/mw00/project-maya) (the upstream README is
[README.upstream.md](README.upstream.md)) with one experiment on top: **Latent Sparse Attention (LSA)** from
Prefrontal AI ([paper, arXiv 2610.42147](https://arxiv.org/abs/2610.42147),
[code and distilled maps](https://github.com/Prefrontal-AI/latent-sparse-attention)) ported into Maya's GLM-5.3-Flash
engine as a drop-in replacement for the model's own sparse-attention indexer.

**Short version:** it works, it is verified against the reference implementation bit for bit where that is possible,
and on the machine it was built for it makes nothing faster. It is 2–5 % slower. The reasons are below, and they are
about the hardware class, not about LSA. If you run a 321 B MoE on one 24 GB card, the attention indexer was never
your problem.

## What LSA is, in one paragraph

GLM-5.3-Flash uses DeepSeek-style sparse attention (DSA): for every query, a small "lightning indexer" scores every
cached position and the attention reads the top 2,048 of them. LSA replaces the scoring. Each cached position gets a
64-dimensional *index key* from the layer's 512-dimensional latent through a distilled map; keys are grouped in pools
of 4, and every closed block of 512 keys is clustered (Ward's method) into 64 buckets of 8 keys on average (cap 16), each
with a centroid. A query is projected into the same 64-dimensional space, scored against the *centroids* only, the best
512 buckets are opened best-first until 2,048 keys are picked, and a 512-token window plus 4 sink tokens are added:
2,564 cells per query. From position 1,048,576 the budget doubles. The only trained part is the pair of 512×64
projection maps per layer (2.9 MB for the whole model), distilled by Prefrontal AI against the original indexer; the
model weights are untouched, so it works with Maya's quantized packs as it is.

## What this fork adds

- `include/strata/kernels/glm_lsa.hpp`, `src/kernels/cuda/glm_lsa.cu` — the CUDA kernels: index keys from the latent
  (with the cache's fp16 / INT8 rounding replayed, so a key rebuilt from the cache equals the key computed on the fly),
  the Ward clustering of a block (one thread block per 512-key block, 16 blocks a launch), the query projection (for
  decode: a per-layer fold of the absorbed-query matrices, one GEMV over the raw query; for prefill: the absorbed queries
  that already exist), centroid scoring, and the stable top-p / best-first top-k selection with window and sinks.
- `src/core/glm_lsa.cu` — the engine side: loading the map, per-sequence state (a ring of the last 9,216 keys, the
  centroids and members of up to 512 blocks per layer: 135 MB per sequence at a 262K context), lazy bucket builds when
  a block closes (from the ring, or from the latent cache after a snapshot or slot restore), and the selection calls.
- Hooks in the four places the engine selects cells: single-token decode (`glm_fast_path.cu`), prompt prefill
  (`glm_prefill.cu`, sub-batches up to 8,192 tokens), the DFlash verify window (`glm_window.cu`) and parallel decode rows
  (`glm_rows.cu`). The selection is written in the native `cells` format, so KV streaming, conversation slots and
  snapshots work unchanged. The native indexer keeps computing its caches, so a slot saved with LSA on loads with it off.
- A short-context shortcut: below 2,560 positions LSA's selection is provably every key, so one fill kernel replaces the
  query, score and select launches (this is what the native path does too).
- `--lsa FILE` on the engine (`--lsa off` disables it; the environment variable is `STRATA_GLM_LSA`). The engine prints
  one line at start: `glm lsa: CUDA0 11 DSA layers select by LSA (...)`.
- `tools/lsa/`: `lsa_convert.py` turns Prefrontal AI's map file into the engine's `.lsa` binary, `lsa_oracle.py` runs
  the reference implementation on random keys and writes an oracle file, `lsa_test.cu` checks the CUDA kernels against
  it, `lsa_bench.py` runs the end-to-end comparison below.

The NextN (MTP) block keeps the native indexer. The port is CUDA only (tested on an RTX 4090, sm_89); the HIP build is
untouched but untested, and prefill under HIP needs the F32 absorbed queries.

## Verification

**Kernels against the reference.** `lsa_oracle.py` runs the repository's own torch implementation
(`build_buckets` / `read`) on 8K–16K random and clustered unit keys with 48–96 queries; `lsa_test.cu` runs Maya's
kernels on the same keys. On all three oracle sets: every bucket partition identical, centroids within 1.2e-7,
every selection identical in content and order except one near-tie (two buckets with scores equal to fp32 rounding)
ordered differently with the same set of cells; the batched form equals the one-by-one form; the shortcut equals the
oracle below 2,560 positions. Score + select cost 28–40 µs per query over 1–2K buckets, a block's Ward clustering
10–27 µs.

**End to end.** `lsa_bench.py` starts Maya's server twice with the same config except `--lsa`, asks four short
questions and a 250-word story (thinking off, temperature 0), then plants three secret codes at 15 / 50 / 85 % of a
40K-token haystack built from Maya's own docs and asks for each one, then measures decode at that context.
Machine: one RTX 4090 (24 GB), a Raptor Lake desktop CPU (the engine's CPU lane on 19 threads), the experts streamed
from a 75 GB pinned RAM tier; Windows 11, CUDA 13.0; the Maya-S24 pack, 262,144-token context, KV streaming on, the
DFlash2 drafter on.

| | native DSA indexer | LSA |
|---|---|---|
| needles found (3 codes, 40,263-token prompt) | 3/3 | 3/3 |
| prefill of that prompt | 2,192 tok/s | 2,143 tok/s |
| decode at 40K context | 24.3 tok/s | 23.8 tok/s |
| decode, short prompt (story) | 26.1 tok/s | 24.8 tok/s |
| model load | 60 s | 60 s |

Answers are coherent and, at temperature 0, mostly identical between the two (the same Fibonacci function, the same
arithmetic, a different but correct story). No errors in the engine log.

## Why the results are underwhelming

1. **The indexer is not where the time goes on this hardware.** A decode step takes about 40 ms here, and almost all
   of it is moving experts: the 321 B model's experts live in RAM, the GPU pool holds 14 GB of them, and every token
   fetches 10–30 experts over PCIe or computes them on the CPU. The native indexer's share of a step at 40K tokens is a
   fraction of a millisecond. Replacing it cannot show up in tokens per second, and it does not.
2. **The paper's speedups live at a context this box never reaches.** The 22.7× over GLM-5.3-Flash's own sparse
   attention is measured at 10 M tokens on an H200, with the whole model resident. There the native indexer, which
   touches every cached position per query, is linear in the context and dominates; LSA scores one centroid per 8
   keys and reads only the opened buckets. At 40K–256K on one 24 GB card both are noise next to the expert traffic.
3. **LSA is added, not substituted.** The native indexer still runs its key projection and pool caches on every token so
   that snapshots and slots stay compatible between runs with and without `--lsa`. Only its scoring and selection are
   skipped. LSA then adds its own key projection, a bucket build when a block closes, and a query fold, score and select
   per layer per token. On a card that is busy with experts those extra launches and the host work cost 2–5 %.
4. **Our quality test cannot separate the two.** Both methods find three needles in 40K tokens, and so would a much
   worse selector. The paper reports LSA matching or beating DSA on RULER-style suites at 128K and beyond; running those
   on this machine takes hours per point and was not done. Treat the quality side as "not broken", not as "as good".
5. **The part that could matter here is untested.** Buckets built lazily from the latent cache after a slot restore, the
   doubled budget past 1 M tokens, several sequences at once — all implemented and unit-checked, none exercised in a
   long session.

If you have a multi-GPU rig where the experts are resident and you run contexts in the hundreds of thousands of
tokens, the picture may differ: then the indexer *is* a measurable share of a step and LSA's selection is cheaper. That
is a measurement somebody with that hardware could take with `tools/lsa/lsa_bench.py`.

## Trying it

1. Get the distilled map for GLM-5.3-Flash from Prefrontal AI's release `d-lsa-weights-v1`
   (`glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt`, Apache-2.0):
   ```bash
   gh release download d-lsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention --pattern 'glm53_flash*latent.pt' --dir maps
   ```
2. Convert it (needs torch and numpy):
   ```bash
   python -I tools/lsa/lsa_convert.py maps/glm53_flash__long-16k-1M-heavy-d2048-lr3e-4-latent.pt glm53_flash-latent.lsa
   ```
3. Build Maya as usual (`setup.sh` / `START-MAYA.bat`), then add to the engine's arguments in your `maya-*.json`:
   `"--lsa", "<path>/glm53_flash-latent.lsa"`. Start it and look for the `glm lsa:` line in the engine log.
4. To compare, run `tools/lsa/lsa_bench.py` once per config (`--tag native`, `--tag lsa`) and then `--compare`.

To check the kernels on your card (any oracle size; the `-arch` is yours):
```bash
pip install torch numpy
git clone https://github.com/Prefrontal-AI/latent-sparse-attention lsa-ref
python -I tools/lsa/lsa_oracle.py lsa-ref/src oracle.bin --keys 16384 --queries 96 --clustered
nvcc -O2 -std=c++17 -arch=sm_89 -I include -I src/kernels/cuda tools/lsa/lsa_test.cu src/kernels/cuda/glm_lsa.cu -o lsa_test
./lsa_test oracle.bin
```

## Credits and license

Project Maya is by Peasant Smith and contributors, MIT license (see [README.upstream.md](README.upstream.md) and
[LICENSE](LICENSE)). Latent Sparse Attention is by Prefrontal AI, Apache-2.0; this port uses their published map and
their reference implementation as the test oracle, not their H200/B300 kernels. The port itself was written with
Claude (Anthropic) and is under Maya's MIT license.
