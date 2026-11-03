// src/kernels/cuda/glm_lsa.cu - Latent Sparse Attention (D-LSA Flat) kernels; see include/strata/kernels/glm_lsa.hpp.
//
// Index keys and queries (projection + unit rows, fp32), the Ward merge of a cluster block's pools into buckets (one
// block of threads a cluster block, the reference's rounds: every pool's cheapest allowed partner, the mutual pairs
// merged cheapest first, no further than the block's count), bucket scores (one warp a bucket), and the selection
// (the radix top-p of dsa_topk.cuh with pools of one, the expansion by a block-wide scan, then the window and the
// sinks).
#include "strata/kernels/glm_lsa.hpp"

#include "dsa_topk.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace strata::kernels::glml {
namespace {

constexpr int D = kLsaDim;
constexpr float kNormEps = 1e-12f;     // F.normalize's eps (lsa.constants.NORM_EPS)
constexpr float kCentEps = 1e-30f;     // build_buckets' clamp of a bucket's key-sum norm

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) std::fprintf(stderr, "glm lsa: %s: %s\n", what, cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// a 512-thread block's sum (16 warps); red: 32 floats of shared memory
__device__ __forceinline__ float block_sum_512(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) red[w] = v;
    __syncthreads();
    float t = lane < 16 ? red[lane] : 0.0f;
    t = warp_sum(t);
    __syncthreads();
    return t;
}
__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float((uint32_t) v << 16); }
__device__ __forceinline__ float inf_f() { return __int_as_float(0x7f800000); }
__device__ __forceinline__ float f16f(uint16_t v) { return __half2float(__ushort_as_half(v)); }

// out[d] = normalize(y . W) for y[in_dim] in shared memory and W [in_dim x 64] (unit row, a zero row stays zero);
// 512 threads: 8 groups of 64 threads, each over in_dim / 8 inputs.  part: 8 * 64 floats of shared memory, red: 2.
__device__ void proj_unit(const float* s_y, int in_dim, const float* __restrict__ W, float* part, float* red, float* out) {
    const int tid = threadIdx.x, grp = tid >> 6, d = tid & 63;
    const int per = in_dim >> 3, c0 = grp * per;
    float acc = 0.0f;
    for (int c = c0; c < c0 + per; ++c) acc += s_y[c] * W[(size_t) c * D + d];
    part[grp * D + d] = acc;
    __syncthreads();
    if (tid < D) {
        float v = 0.0f;
        for (int g = 0; g < 8; ++g) v += part[g * D + d];
        part[d] = v;   // (group 0's row reused for the finished sums)
        const float s = warp_sum(v * v);
        if ((tid & 31) == 0) red[tid >> 5] = s;
    }
    __syncthreads();
    if (tid < D) {
        const float norm = sqrtf(red[0] + red[1]);
        out[d] = part[d] / fmaxf(norm, kNormEps);
    }
}

// ---- index keys from kv_a's raw rows (the latent as the cache rounds it), T blocks of 512 threads
__global__ void __launch_bounds__(512) keys_raw_kernel(const float* __restrict__ kv_raw, int64_t ld,
                                                       const float* __restrict__ norm_w, float eps, int kv_lora,
                                                       int q8, const float* __restrict__ W_K, int64_t p0, int ring,
                                                       float* __restrict__ keys) {
    __shared__ float s_y[1024];
    __shared__ float s_part[8 * D];
    __shared__ float s_red[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* row = kv_raw + (size_t) t * ld;
    float ss = 0.0f;
    for (int c = tid; c < kv_lora; c += 512) ss += row[c] * row[c];
    ss = block_sum_512(ss, s_red);
    const float inv = rsqrtf(ss / (float) kv_lora + eps);
    for (int c = tid; c < kv_lora; c += 512) {   // (kv_lora % 32 == 0: whole warps per 32-group)
        float y = row[c] * inv * norm_w[c];
        if (q8) {
            const float amax = warp_max(fabsf(y));
            const __half d16 = __float2half(amax / 127.0f);
            const float dd = __half2float(d16);
            const int q = dd > 0.0f ? max(-127, min(127, __float2int_rn(y / dd))) : 0;
            y = (float) q * dd;
        } else {
            y = __half2float(__float2half(y));
        }
        s_y[c] = y;
    }
    __syncthreads();
    const int64_t p = p0 + t;
    proj_unit(s_y, kv_lora, W_K, s_part, s_red, keys + (size_t) (ring > 0 ? p % ring : (int64_t) t) * D);
}

// ---- index keys from cached latent rows
__global__ void __launch_bounds__(512) keys_lat_kernel(const uint16_t* __restrict__ lat, int q8, int kv_lora,
                                                       int64_t row0, const float* __restrict__ W_K, int64_t p0, int ring,
                                                       float* __restrict__ keys) {
    __shared__ float s_y[1024];
    __shared__ float s_part[8 * D];
    __shared__ float s_red[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int64_t r = row0 + t;
    if (q8) {
        const int rec_bytes = kv_lora + kv_lora / 32 * 2;
        const uint8_t* rec = (const uint8_t*) lat + (size_t) rec_bytes * r;
        for (int c = tid; c < kv_lora; c += 512)
            s_y[c] = (float) ((const int8_t*) rec)[c] * __half2float(((const __half*) (rec + kv_lora))[c >> 5]);
    } else {
        for (int c = tid; c < kv_lora; c += 512) s_y[c] = f16f(lat[(size_t) kv_lora * r + c]);
    }
    __syncthreads();
    proj_unit(s_y, kv_lora, W_K, s_part, s_red, keys + (size_t) (ring > 0 ? (p0 + t) % ring : (int64_t) t) * D);
}

// ---- Ward: one block of 128 threads a cluster block, a thread a pool
constexpr int P_MAX = kLsaPoolsMax;
constexpr int KS_LD = D + 1;   // (an odd stride: a thread's row walk and the partners' row reads stay off one bank)

__global__ void __launch_bounds__(P_MAX) build_kernel(const float* __restrict__ keys, int ring, int64_t p_first, int64_t b0,
                                                      int pool, int block, int buckets, int cap, float* __restrict__ cent,
                                                      int* __restrict__ memb) {
    __shared__ float ks[P_MAX * KS_LD];
    __shared__ float sz[P_MAX], sq[P_MAX], al[P_MAX], bestc[P_MAX];
    __shared__ int best[P_MAX], parent[P_MAX], root[P_MAX], pref[P_MAX];
    __shared__ unsigned char alive[P_MAX];
    __shared__ int s_cnt, s_any;
    const int P = block / pool;
    const int i = threadIdx.x;
    const int64_t b = b0 + blockIdx.x;
    const int64_t pos0 = b * block;
    const auto key_row = [&](int64_t p) -> const float* {
        return keys + (size_t) (ring > 0 ? p % ring : p - p_first) * D;
    };
    // the pool keys: each pool's keys summed in position order
    if (i < P) {
        float acc[D];
#pragma unroll
        for (int d = 0; d < D; ++d) acc[d] = 0.0f;
        for (int m = 0; m < pool; ++m) {
            const float* k = key_row(pos0 + (int64_t) i * pool + m);
#pragma unroll
            for (int d = 0; d < D; ++d) acc[d] += k[d];
        }
#pragma unroll
        for (int d = 0; d < D; ++d) ks[i * KS_LD + d] = acc[d];
        sz[i] = (float) pool;
        alive[i] = 1;
        parent[i] = i;
        best[i] = -1;
        bestc[i] = inf_f();
    }
    __syncthreads();
    for (int round = 0; round < 256; ++round) {
        if (i == 0) s_cnt = 0;
        __syncthreads();
        if (i < P && alive[i]) atomicAdd(&s_cnt, 1);
        __syncthreads();
        const int merges_left = s_cnt - buckets;
        if (merges_left <= 0) break;
        if (i < P && alive[i]) {
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) s += ks[i * KS_LD + d] * ks[i * KS_LD + d];
            sq[i] = s;
            al[i] = sqrtf(fmaxf(s, 0.0f));
        }
        __syncthreads();
        // every live pool's cheapest allowed partner (a tie to the lower index)
        int bj = -1;
        float bc = inf_f();
        if (i < P && alive[i]) {
            const float szi = sz[i], sqi = sq[i], ali = al[i];
            for (int j = 0; j < P; ++j) {
                if (j == i || !alive[j] || szi + sz[j] > (float) cap) continue;
                float g = 0.0f;
#pragma unroll 8
                for (int d = 0; d < D; ++d) g += ks[i * KS_LD + d] * ks[j * KS_LD + d];
                const float merged = sqrtf(fmaxf(sqi + sq[j] + 2.0f * g, 0.0f));
                const float c = ali + al[j] - merged;
                if (c < bc) {
                    bc = c;
                    bj = j;
                }
            }
        }
        if (i < P) {
            best[i] = bj;
            bestc[i] = bc;
        }
        __syncthreads();
        // the mutual pairs, flagged at their lower index; their rank by cost (a tie to the lower index)
        const bool mutual = i < P && alive[i] && bj >= 0 && best[bj] == i && i < bj;
        int rank = 0;
        if (mutual)
            for (int k = 0; k < P; ++k) {
                if (k == i || !alive[k]) continue;
                const int bk = best[k];
                if (bk < 0 || best[bk] != k || k >= bk) continue;
                if (bestc[k] < bc || (bestc[k] == bc && k < i)) ++rank;
            }
        const bool merge = mutual && rank < merges_left;
        if (i == 0) s_any = 0;
        __syncthreads();
        if (merge) s_any = 1;
        __syncthreads();
        if (!s_any) break;
        if (merge) {
#pragma unroll
            for (int d = 0; d < D; ++d) ks[i * KS_LD + d] += ks[bj * KS_LD + d];
            sz[i] += sz[bj];
            alive[bj] = 0;
            parent[bj] = i;
        }
        __syncthreads();
    }
    __syncthreads();
    // every pool's surviving pool; the survivors numbered in index order
    if (i < P) {
        int r = i;
        while (!alive[r]) r = parent[r];
        root[i] = r;
        pref[i] = alive[i] ? 1 : 0;
    }
    __syncthreads();
    for (int off = 1; off < P_MAX; off <<= 1) {   // inclusive scan of the live flags
        const int v = i >= off && i < P ? pref[i - off] : 0;
        __syncthreads();
        if (i < P) pref[i] += v;
        __syncthreads();
    }
    if (i < P && alive[i]) {
        const int L = pref[i] - 1;
        if (L < buckets) {
            // the bucket's key sum over its keys in position order (as the reference's index_add), its members
            float acc[D];
#pragma unroll
            for (int d = 0; d < D; ++d) acc[d] = 0.0f;
            int* mrow = memb + (size_t) (b * buckets + L) * cap;
            int n = 0;
            for (int j = 0; j < P; ++j) {
                if (root[j] != i) continue;
                for (int m = 0; m < pool; ++m) {
                    const int64_t p = pos0 + (int64_t) j * pool + m;
                    const float* k = key_row(p);
#pragma unroll
                    for (int d = 0; d < D; ++d) acc[d] += k[d];
                    if (n < cap) mrow[n++] = (int) p;
                }
            }
            for (; n < cap; ++n) mrow[n] = -1;
            float s = 0.0f;
#pragma unroll
            for (int d = 0; d < D; ++d) s += acc[d] * acc[d];
            const float inv = 1.0f / fmaxf(sqrtf(s), kCentEps);
            float* crow = cent + (size_t) (b * buckets + L) * D;
#pragma unroll
            for (int d = 0; d < D; ++d) crow[d] = acc[d] * inv;
        }
    }
}

// ---- the fold M = k_b (per head, transposed) . W_Q: a block an (head, 64 e) tile, 256 threads x 16 outputs
__global__ void __launch_bounds__(256) fold_kernel(const uint16_t* __restrict__ k_b, int kv_lora, int qk_nope,
                                                   const float* __restrict__ W_Q, float* __restrict__ M) {
    __shared__ float s_k[64], s_w[D];
    const int h = blockIdx.x, e0 = blockIdx.y * 64;
    const int tid = threadIdx.x, d = tid & 63, eg = tid >> 6;   // 4 groups of 16 e each
    float acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
    const uint16_t* kb = k_b + (size_t) h * kv_lora * qk_nope;
    for (int c = 0; c < kv_lora; ++c) {
        if (tid < 64) s_k[tid] = e0 + tid < qk_nope ? bf16f(kb[(size_t) c * qk_nope + e0 + tid]) : 0.0f;
        else if (tid < 128) s_w[tid - 64] = W_Q[(size_t) c * D + (tid - 64)];
        __syncthreads();
        const float w = s_w[d];
#pragma unroll
        for (int j = 0; j < 16; ++j) acc[j] += s_k[eg * 16 + j] * w;
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int e = e0 + eg * 16 + j;
        if (e < qk_nope) M[((size_t) h * qk_nope + e) * D + d] = acc[j];
    }
}

// ---- q . M in parts: grid (T, splits), 32 warps a block, a warp over its rows with a lane per two dims
constexpr int FOLD_SPLITS = 16;
__global__ void __launch_bounds__(1024) query_fold_kernel(const float* __restrict__ q, int64_t ld_q, int K,
                                                          const float* __restrict__ M, float* __restrict__ part) {
    __shared__ float red[32 * D];
    const int t = blockIdx.x, sp = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int per_block = (K + FOLD_SPLITS - 1) / FOLD_SPLITS;
    const int per_warp = (per_block + 31) / 32;
    const int k0 = sp * per_block + w * per_warp, k1 = min(K, min(sp * per_block + per_block, k0 + per_warp));
    const float* qt = q + (size_t) t * ld_q;
    float2 acc = make_float2(0.0f, 0.0f);
    for (int k = k0; k < k1; ++k) {
        const float2 m = ((const float2*) (M + (size_t) k * D))[lane];
        const float qk = qt[k];
        acc.x += qk * m.x;
        acc.y += qk * m.y;
    }
    red[w * D + 2 * lane] = acc.x;
    red[w * D + 2 * lane + 1] = acc.y;
    __syncthreads();
    if (tid < D) {
        float v = 0.0f;
        for (int i = 0; i < 32; ++i) v += red[i * D + tid];
        part[((size_t) t * FOLD_SPLITS + sp) * D + tid] = v;
    }
}
__global__ void __launch_bounds__(64) query_fold_finish(const float* __restrict__ part, float* __restrict__ q_idx) {
    __shared__ float red[2];
    const int t = blockIdx.x, d = threadIdx.x;
    float v = 0.0f;
    for (int sp = 0; sp < FOLD_SPLITS; ++sp) v += part[((size_t) t * FOLD_SPLITS + sp) * D + d];
    const float s = warp_sum(v * v);
    if ((d & 31) == 0) red[d >> 5] = s;
    __syncthreads();
    q_idx[(size_t) t * D + d] = v / fmaxf(sqrtf(red[0] + red[1]), kNormEps);
}

// ---- the absorbed queries summed over the heads, then W_Q: a block a token, 512 threads
__global__ void __launch_bounds__(512) query_abs_kernel(const float* __restrict__ q_abs, int64_t ld, int n_head,
                                                        int kv_lora, const float* __restrict__ W_Q,
                                                        float* __restrict__ q_idx) {
    __shared__ float s_y[1024];
    __shared__ float s_part[8 * D];
    __shared__ float s_red[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* row = q_abs + (size_t) t * ld;
    for (int c = tid; c < kv_lora; c += 512) {
        float s = 0.0f;
        for (int h = 0; h < n_head; ++h) s += row[(size_t) h * kv_lora + c];
        s_y[c] = s;
    }
    __syncthreads();
    proj_unit(s_y, kv_lora, W_Q, s_part, s_red, q_idx + (size_t) t * D);
}

// ---- scores: a warp a bucket, a lane two dims
__global__ void __launch_bounds__(1024) score_kernel(const float* __restrict__ q_idx, const float* __restrict__ cent,
                                                     int buckets, int block, int64_t p0, float* __restrict__ scores,
                                                     int64_t ld) {
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    const int64_t vis = ((p0 + t) / block) * buckets;
    const int64_t b = (int64_t) blockIdx.x * 32 + (threadIdx.x >> 5);
    if (b >= vis) return;
    const float2 qv = ((const float2*) (q_idx + (size_t) t * D))[lane];
    const float2 cv = ((const float2*) (cent + (size_t) b * D))[lane];
    const float s = warp_sum(qv.x * cv.x + qv.y * cv.y);
    if (lane == 0) scores[(size_t) t * ld + b] = s;
}

// ---- the selection: a block a token
__global__ void __launch_bounds__(1024) select_kernel(const float* __restrict__ scores, int64_t ld,
                                                      const int* __restrict__ memb, LsaGeom g, LsaBudget bud, int64_t p0,
                                                      int width, int* __restrict__ cells, int64_t ld_cells,
                                                      int* __restrict__ n_sel) {
    __shared__ int s_pick[1024];
    __shared__ int s_cnt[1024];
    const int t = blockIdx.x, tid = threadIdx.x;
    const int64_t p = p0 + t;
    const int64_t vis64 = (p / g.block) * g.buckets;
    const int vis = (int) min(vis64, (int64_t) 0x7fffffff);
    const int top_k = p < bud.late_pos ? bud.top_k : bud.late_top_k;
    const int top_p = p < bud.late_pos ? bud.top_p : bud.late_top_p;
    const int top = min(min(top_p, vis), 1024);
    const float* sc = scores + (size_t) t * ld;
    int* out = cells + (size_t) t * ld_cells;
    if (top > 0) dsa::select_top_pools(sc, vis, top, 1, s_pick);
    __syncthreads();
    const int64_t wstart = p + 1 - g.window > 0 ? p + 1 - g.window : 0;
    // each picked bucket's members outside the sinks and the window, in position order; their offsets by a scan
    int cnt = 0;
    const int* mrow = nullptr;
    if (tid < top) {
        mrow = memb + (size_t) s_pick[tid] * g.cap;
        for (int j = 0; j < g.cap; ++j) {
            const int v = mrow[j];
            if (v >= g.sink && v < wstart) ++cnt;
        }
    }
    s_cnt[tid] = cnt;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int v = tid >= off ? s_cnt[tid - off] : 0;
        __syncthreads();
        s_cnt[tid] += v;
        __syncthreads();
    }
    const int n_mem = min(s_cnt[1023], top_k);
    if (tid < top) {
        int o = s_cnt[tid] - cnt;
        for (int j = 0; j < g.cap && o < top_k; ++j) {
            const int v = mrow[j];
            if (v >= g.sink && v < wstart) out[o++] = v;
        }
    }
    const int n_win = (int) (p - wstart + 1);
    const int n_sink = (int) min((int64_t) g.sink, wstart);
    for (int i = tid; i < n_win; i += 1024) out[n_mem + i] = (int) (wstart + i);
    for (int i = tid; i < n_sink; i += 1024) out[n_mem + n_win + i] = i;
    const int n = n_mem + n_win + n_sink;
    for (int i = n + tid; i < width; i += 1024) out[i] = -1;
    if (tid == 0 && n_sel != nullptr) n_sel[t] = n;
}

}  // namespace

void lsa_keys_raw(const float* kv_raw, int64_t ld, const float* norm_w, float eps, int kv_lora, bool q8,
                  const float* W_K, int T, int64_t p0, int ring, float* keys, cudaStream_t s) {
    if (T <= 0) return;
    keys_raw_kernel<<<T, 512, 0, s>>>(kv_raw, ld, norm_w, eps, kv_lora, q8 ? 1 : 0, W_K, p0, ring, keys);
    check("keys_raw");
}

void lsa_keys_lat(const uint16_t* lat, bool q8, int kv_lora, int64_t row0, int T, const float* W_K, int64_t p0,
                  int ring, float* keys, cudaStream_t s) {
    if (T <= 0) return;
    keys_lat_kernel<<<T, 512, 0, s>>>(lat, q8 ? 1 : 0, kv_lora, row0, W_K, p0, ring, keys);
    check("keys_lat");
}

void lsa_build(const float* keys, int ring, int64_t p_first, int64_t b0, int n_blocks, LsaGeom g, float* cent,
               int* memb, cudaStream_t s) {
    if (n_blocks <= 0) return;
    build_kernel<<<n_blocks, P_MAX, 0, s>>>(keys, ring, p_first, b0, g.pool, g.block, g.buckets, g.cap, cent, memb);
    check("build");
}

void lsa_fold(const uint16_t* k_b, int n_head, int kv_lora, int qk_nope, const float* W_Q, float* M, cudaStream_t s) {
    const dim3 grid(n_head, (qk_nope + 63) / 64);
    fold_kernel<<<grid, 256, 0, s>>>(k_b, kv_lora, qk_nope, W_Q, M);
    check("fold");
}

void lsa_query_fold(const float* q, int64_t ld_q, int K, const float* M, int T, float* part, float* q_idx,
                    cudaStream_t s) {
    if (T <= 0) return;
    query_fold_kernel<<<dim3(T, FOLD_SPLITS), 1024, 0, s>>>(q, ld_q, K, M, part);
    query_fold_finish<<<T, 64, 0, s>>>(part, q_idx);
    check("query_fold");
}

void lsa_query_abs(const float* q_abs, int64_t ld, int n_head, int kv_lora, const float* W_Q, int T, float* q_idx,
                   cudaStream_t s) {
    if (T <= 0) return;
    query_abs_kernel<<<T, 512, 0, s>>>(q_abs, ld, n_head, kv_lora, W_Q, q_idx);
    check("query_abs");
}

void lsa_score(const float* q_idx, const float* cent, int buckets, int block, int64_t p0, int T, float* scores,
               int64_t ld, cudaStream_t s) {
    if (T <= 0) return;
    const int64_t vis_max = ((p0 + T - 1) / block) * buckets;
    if (vis_max <= 0) return;
    const dim3 grid((unsigned) ((vis_max + 31) / 32), (unsigned) T);
    score_kernel<<<grid, 1024, 0, s>>>(q_idx, cent, buckets, block, p0, scores, ld);
    check("score");
}

void lsa_select(const float* scores, int64_t ld, const int* memb, LsaGeom g, LsaBudget bud, int64_t p0, int T,
                int width, int* cells, int64_t ld_cells, int* n_sel, cudaStream_t s) {
    if (T <= 0) return;
    select_kernel<<<T, 1024, 0, s>>>(scores, ld, memb, g, bud, p0, width, cells, ld_cells, n_sel);
    check("select");
}

__global__ void select_all_kernel(int64_t p0, int width, int* cells, int64_t ld_cells, int* n_sel) {
    const int t = (int) blockIdx.y;
    const int64_t p = p0 + t;
    int* const row = cells + (size_t) t * ld_cells;
    for (int i = (int) (blockIdx.x * blockDim.x + threadIdx.x); i < width; i += (int) (gridDim.x * blockDim.x))
        row[i] = (int64_t) i <= p ? i : -1;
    if (n_sel != nullptr && blockIdx.x == 0 && threadIdx.x == 0) n_sel[t] = (int) (p + 1 < width ? p + 1 : width);
}

void lsa_select_all(int64_t p0, int T, int width, int* cells, int64_t ld_cells, int* n_sel, cudaStream_t s) {
    if (T <= 0 || width <= 0) return;
    const dim3 grid((unsigned) ((width + 255) / 256), (unsigned) T);
    select_all_kernel<<<grid, 256, 0, s>>>(p0, width, cells, ld_cells, n_sel);
    check("select_all");
}

}  // namespace strata::kernels::glml
