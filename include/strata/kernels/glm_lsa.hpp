// include/strata/kernels/glm_lsa.hpp - Latent Sparse Attention (D-LSA Flat) for the glm5-next DSA layers: the kernels.
//
// LSA (Kubicek et al., arXiv 2610.42147; Prefrontal-AI/latent-sparse-attention) replaces the model's own indexer with a
// learned 64-dim index space: every position's index KEY is the normalized projection of its latent (the KV-cache row,
// kv_a_norm's output) through W_K [kv_lora x 64]; a query's index QUERY is the normalized projection of its absorbed
// queries summed over the heads through W_Q [kv_lora x 64].  Keys are pooled four at a time, each closed cluster block
// of 512 keys is merged by Ward into 64 buckets (average 8 keys, 16 at most), and a query scores the bucket CENTROIDS
// (the normalized key sums) of every block closed before its own, picks the top_p best, reads their keys best bucket
// first up to top_k, and always reads its window of 512 recent positions and the 4 sinks.  The attention kernel is
// unchanged: LSA only decides which cells it reads - here the same `cells` list (int32 positions, -1 padded) the
// native selection writes, top_k + window + sink wide.
//
// The reference is src/lsa/core/reference.py (build_buckets, read) and ward.py of the repo; these kernels follow it
// step for step (the tie rules included) in fp32.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace strata::kernels::glml {

constexpr int kLsaDim = 64;        // the index dim these kernels are written for
constexpr int kLsaPoolsMax = 128;  // pools a cluster block at most (block / pool)
constexpr int kLsaCapMax = 32;     // keys a bucket at most

struct LsaGeom {
    int pool = 4;       // keys a pool
    int block = 512;    // keys a cluster block
    int buckets = 64;   // buckets a block ends with
    int cap = 16;       // keys a bucket at most
    int window = 512;   // recent positions always read (the query's own included)
    int sink = 4;       // first positions always read
};
struct LsaBudget {
    int top_k = 2048, top_p = 512;        // keys read from the buckets at most, buckets picked
    int64_t late_pos = 1 << 20;           // from this position the late budget applies
    int late_top_k = 4096, late_top_p = 1024;
};

/// The index keys of T positions p0 .. p0 + T - 1 from kv_a's raw output (rows of ld floats; the rmsnorm with `norm_w`
/// and `eps` as dsa_prep applies it, then the cache's own rounding - FP16, or the INT8 record with `q8` - so the key is
/// the cached latent's), through W_K [kv_lora x 64], unit rows.  Written to the ring at row (p0 + t) % ring.
void lsa_keys_raw(const float* kv_raw, int64_t ld, const float* norm_w, float eps, int kv_lora, bool q8,
                  const float* W_K, int T, int64_t p0, int ring, float* keys, cudaStream_t s);
/// The same from cached latent rows `row0` .. `row0` + T - 1 of `lat` (FP16 rows, or INT8 records with `q8`; `lat` may
/// be a device-mapped host copy), written densely to keys[t * 64] (ring 0) or to the ring at (p0 + t) % ring.
void lsa_keys_lat(const uint16_t* lat, bool q8, int kv_lora, int64_t row0, int T, const float* W_K, int64_t p0,
                  int ring, float* keys, cudaStream_t s);
/// Ward buckets of the `n_blocks` cluster blocks from block b0: the key of position p is keys[(p % ring) * 64]
/// (ring > 0) or keys[(p - p_first) * 64].  cent[(b * buckets + k) * 64]: bucket k of block b's centroid;
/// memb[(b * buckets + k) * cap]: its keys' positions in position order, -1 padded.
void lsa_build(const float* keys, int ring, int64_t p_first, int64_t b0, int n_blocks, LsaGeom g, float* cent,
               int* memb, cudaStream_t s);
/// M[(h * qk_nope + e) * 64 + d] = sum_c k_b[h][c][e] * W_Q[c][d] - the fold of the absorbed queries' sum over the
/// heads with W_Q, so an index query is one GEMV of the plain query (q_b's output) against M.  k_b: BF16, per head
/// kv_lora rows of qk_nope (mla()'s layout).
void lsa_fold(const uint16_t* k_b, int n_head, int kv_lora, int qk_nope, const float* W_Q, float* M, cudaStream_t s);
/// q_idx[t] = normalize(q[t] . M) for T queries of K = n_head * qk_nope floats (rows of ld_q); part: scratch of
/// T * 16 * 64 floats.
void lsa_query_fold(const float* q, int64_t ld_q, int K, const float* M, int T, float* part, float* q_idx,
                    cudaStream_t s);
/// q_idx[t] = normalize((sum_h q_abs[t][h]) . W_Q) for T tokens' absorbed queries (rows of ld floats, n_head x kv_lora
/// each).
void lsa_query_abs(const float* q_abs, int64_t ld, int n_head, int kv_lora, const float* W_Q, int T, float* q_idx,
                   cudaStream_t s);
/// scores[t * ld + b] = cent[b] . q_idx[t] for the buckets b visible to position p0 + t: the first
/// ((p0 + t) / block) * buckets of them.
void lsa_score(const float* q_idx, const float* cent, int buckets, int block, int64_t p0, int T, float* scores,
               int64_t ld, cudaStream_t s);
/// Per token t at position p0 + t: the top_p visible buckets by score (stable: a tie to the lower bucket), their
/// members best bucket first - a member in the sinks or the window left out - up to top_k, then the window
/// [max(0, p + 1 - window), p], then the sinks [0, min(sink, window start)); cells[t * ld_cells .. + width) with -1
/// past the count, n_sel[t] the count (may be null).  top_p <= 1024.
void lsa_select(const float* scores, int64_t ld, const int* memb, LsaGeom g, LsaBudget bud, int64_t p0, int T,
                int width, int* cells, int64_t ld_cells, int* n_sel, cudaStream_t s);
/// The selection of a position p = p0 + t below top_k + block, where it is every key [0, p] (the closed blocks' keys
/// fit the top-k, the open block lies in the window): cells[t * ld_cells + i] = i for i <= p, -1 past it up to width,
/// n_sel[t] = p + 1 (may be null).  What lsa_score + lsa_select would write, without the scores.
void lsa_select_all(int64_t p0, int T, int width, int* cells, int64_t ld_cells, int* n_sel, cudaStream_t s);

}  // namespace strata::kernels::glml
