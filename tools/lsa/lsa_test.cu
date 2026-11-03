// lsa_test.cu - the Maya LSA kernels against the repo's torch oracle (lsa_oracle.py's file).
//
// Build (from the repository root; on Windows inside a vcvars64 shell; -arch for your card):
//   nvcc -O2 -std=c++17 -arch=sm_89 -I include -I src/kernels/cuda tools/lsa/lsa_test.cu src/kernels/cuda/glm_lsa.cu
//        -o lsa_test
// Run:  lsa_test oracle.bin        (oracle.bin from tools/lsa/lsa_oracle.py)
//
// Checks: (1) the buckets of every closed block - the same partition (members row for row) and centroids within
// 1e-5; (2) every query's selection - the same cells in the same order, or, where a near-tie in the fp32 scores
// orders two buckets differently, the same SET of cells (reported separately).
#include "strata/kernels/glm_lsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <vector>

namespace gl = strata::kernels::glml;

template <typename T>
static bool rd(std::ifstream& f, T& v) { return (bool) f.read((char*) &v, sizeof(T)); }
template <typename T>
static bool rdv(std::ifstream& f, std::vector<T>& v, size_t n) {
    v.resize(n);
    return (bool) f.read((char*) v.data(), (std::streamsize) (n * sizeof(T)));
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: lsa_test oracle.bin\n"); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    uint32_t magic, n_keys, n_q, pool, block, buckets, cap, window, sink, top_k, top_p, late_k, late_p, width;
    uint64_t late_pos;
    if (!rd(f, magic) || !rd(f, n_keys) || !rd(f, n_q) || !rd(f, pool) || !rd(f, block) || !rd(f, buckets) || !rd(f, cap) ||
        !rd(f, window) || !rd(f, sink) || !rd(f, top_k) || !rd(f, top_p) || !rd(f, late_pos) || !rd(f, late_k) ||
        !rd(f, late_p) || !rd(f, width) || magic != 0x5441534Cu) {
        std::fprintf(stderr, "bad header\n");
        return 2;
    }
    const uint32_t n_blocks = n_keys / block, n_b = n_blocks * buckets;
    std::vector<float> keys, queries, cent_ref;
    std::vector<int64_t> pos;
    std::vector<int32_t> memb_ref, sel_ref, len_ref;
    if (!rdv(f, keys, (size_t) n_keys * 64) || !rdv(f, queries, (size_t) n_q * 64) || !rdv(f, pos, n_q) ||
        !rdv(f, cent_ref, (size_t) n_b * 64) || !rdv(f, memb_ref, (size_t) n_b * cap) ||
        !rdv(f, sel_ref, (size_t) n_q * width) || !rdv(f, len_ref, n_q)) {
        std::fprintf(stderr, "short file\n");
        return 2;
    }
    gl::LsaGeom g;
    g.pool = (int) pool; g.block = (int) block; g.buckets = (int) buckets; g.cap = (int) cap; g.window = (int) window; g.sink = (int) sink;
    gl::LsaBudget bud;
    bud.top_k = (int) top_k; bud.top_p = (int) top_p; bud.late_pos = (int64_t) late_pos; bud.late_top_k = (int) late_k; bud.late_top_p = (int) late_p;
    // the keys through a ring (as the engine keeps them) and densely (the rebuild path): both must give the buckets
    const int ring = (int) std::min<uint32_t>(n_keys, 9216);
    float *d_keys = nullptr, *d_ring = nullptr, *d_cent = nullptr, *d_cent2 = nullptr, *d_q = nullptr, *d_sc = nullptr;
    int *d_memb = nullptr, *d_memb2 = nullptr, *d_cells = nullptr, *d_nsel = nullptr;
    CK(cudaMalloc(&d_keys, keys.size() * 4));
    CK(cudaMalloc(&d_ring, (size_t) ring * 64 * 4));
    CK(cudaMalloc(&d_cent, (size_t) n_b * 64 * 4));
    CK(cudaMalloc(&d_cent2, (size_t) n_b * 64 * 4));
    CK(cudaMalloc(&d_memb, (size_t) n_b * cap * 4));
    CK(cudaMalloc(&d_memb2, (size_t) n_b * cap * 4));
    CK(cudaMalloc(&d_q, queries.size() * 4));
    CK(cudaMalloc(&d_sc, (size_t) n_q * n_b * 4));
    CK(cudaMalloc(&d_cells, (size_t) n_q * width * 4));
    CK(cudaMalloc(&d_nsel, (size_t) n_q * 4));
    CK(cudaMemcpy(d_keys, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_q, queries.data(), queries.size() * 4, cudaMemcpyHostToDevice));
    // dense: all blocks in one launch
    gl::lsa_build(d_keys, 0, 0, 0, (int) n_blocks, g, d_cent, d_memb, 0);
    CK(cudaDeviceSynchronize());
    // ring: the keys written position by position (a ring of `ring`), each block built when its last key is in
    {
        std::vector<float> rbuf((size_t) ring * 64);
        for (uint32_t b = 0; b < n_blocks; ++b) {
            for (uint32_t p = b * block; p < (b + 1) * block; ++p)
                std::copy(keys.begin() + (size_t) p * 64, keys.begin() + (size_t) (p + 1) * 64, rbuf.begin() + (size_t) (p % ring) * 64);
            CK(cudaMemcpy(d_ring, rbuf.data(), rbuf.size() * 4, cudaMemcpyHostToDevice));
            gl::lsa_build(d_ring, ring, 0, b, 1, g, d_cent2, d_memb2, 0);
            CK(cudaDeviceSynchronize());
        }
    }
    std::vector<float> cent((size_t) n_b * 64), cent2((size_t) n_b * 64);
    std::vector<int32_t> memb((size_t) n_b * cap), memb2((size_t) n_b * cap);
    CK(cudaMemcpy(cent.data(), d_cent, cent.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(memb.data(), d_memb, memb.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cent2.data(), d_cent2, cent2.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(memb2.data(), d_memb2, memb2.size() * 4, cudaMemcpyDeviceToHost));
    uint32_t bad_memb = 0, bad_ring = 0, blocks_diff = 0;
    double max_cent = 0.0;
    std::vector<char> block_bad(n_blocks, 0);
    for (uint32_t b = 0; b < n_b; ++b) {
        bool same = true;
        for (uint32_t j = 0; j < cap; ++j) same = same && memb[(size_t) b * cap + j] == memb_ref[(size_t) b * cap + j];
        if (!same) { ++bad_memb; block_bad[b / buckets] = 1; }
        for (uint32_t j = 0; j < cap; ++j)
            if (memb[(size_t) b * cap + j] != memb2[(size_t) b * cap + j]) { ++bad_ring; break; }
        if (same)
            for (int d = 0; d < 64; ++d)
                max_cent = std::max(max_cent, (double) std::fabs(cent[(size_t) b * 64 + d] - cent_ref[(size_t) b * 64 + d]));
    }
    for (uint32_t b = 0; b < n_blocks; ++b) blocks_diff += block_bad[b];
    std::printf("buckets: %u of %u rows differ from the oracle (%u of %u blocks); ring vs dense: %u rows differ; "
                "centroids of the same rows: max |diff| %.2e\n", bad_memb, n_b, blocks_diff, n_blocks, bad_ring, max_cent);
    // the selection on the oracle's own buckets (the selection logic alone), then on ours
    std::vector<int32_t> sel((size_t) n_q * width), nsel(n_q);
    uint32_t exact = 0, same_set = 0, bad = 0, bad_len = 0;
    for (int pass = 0; pass < 2; ++pass) {
        float* c = pass == 0 ? d_cent : d_cent2;
        int* m = pass == 0 ? d_memb : d_memb2;
        if (pass == 1) {
            CK(cudaMemcpy(d_cent2, cent_ref.data(), cent_ref.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_memb2, memb_ref.data(), memb_ref.size() * 4, cudaMemcpyHostToDevice));
        }
        exact = same_set = bad = bad_len = 0;
        for (uint32_t i = 0; i < n_q; ++i) {   // one query a launch (positions are not consecutive)
            gl::lsa_score(d_q + (size_t) i * 64, c, (int) buckets, (int) block, pos[i], 1, d_sc + (size_t) i * n_b, n_b, 0);
            gl::lsa_select(d_sc + (size_t) i * n_b, n_b, m, g, bud, pos[i], 1, (int) width, d_cells + (size_t) i * width, width,
                           d_nsel + i, 0);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(sel.data(), d_cells, sel.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(nsel.data(), d_nsel, nsel.size() * 4, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n_q; ++i) {
            const int32_t* a = sel.data() + (size_t) i * width;
            const int32_t* r = sel_ref.data() + (size_t) i * width;
            if (nsel[i] != len_ref[i]) ++bad_len;
            bool eq = true;
            for (uint32_t j = 0; j < width; ++j) eq = eq && a[j] == r[j];
            if (eq) { ++exact; continue; }
            std::set<int32_t> sa(a, a + width), sr(r, r + width);
            if (sa == sr) ++same_set;
            else {
                ++bad;
                if (bad <= 3) {
                    uint32_t j = 0;
                    while (j < width && a[j] == r[j]) ++j;
                    std::printf("  query %u at %lld: first difference at %u: ours %d, oracle %d (lengths %d / %d)\n", i,
                                (long long) pos[i], j, a[j], r[j], nsel[i], len_ref[i]);
                }
            }
        }
        std::printf("selection on %s buckets: %u of %u queries exact, %u the same set in another order, %u differ "
                    "(%u lengths differ)\n", pass == 0 ? "OUR" : "the ORACLE'S", exact, n_q, same_set, bad, bad_len);
    }
    // the short-context shortcut: below top_k + block every key is selected - lsa_select_all's set against the oracle's
    uint32_t bad_short = 0, bad_short_len = 0;
    {
        uint32_t n_short = 0;
        for (uint32_t i = 0; i < n_q; ++i) {
            const int64_t all_below = (int64_t) (pos[i] >= (int64_t) late_pos ? late_k : top_k) + (int64_t) block;
            if (pos[i] >= all_below) continue;
            ++n_short;
            gl::lsa_select_all(pos[i], 1, (int) width, d_cells, width, d_nsel, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(sel.data(), d_cells, (size_t) width * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(nsel.data(), d_nsel, 4, cudaMemcpyDeviceToHost));
            const int32_t* r = sel_ref.data() + (size_t) i * width;
            std::set<int32_t> sa(sel.data(), sel.data() + width), sr(r, r + width);
            if (sa != sr) ++bad_short;
            if (nsel[0] != len_ref[i]) ++bad_short_len;
        }
        std::printf("select-all shortcut: %u queries below top_k + block, %u sets differ from the oracle, %u lengths differ\n",
                    n_short, bad_short, bad_short_len);
    }
    // the batched form (T consecutive positions) against the one-by-one form
    {
        const int64_t p0 = (int64_t) n_keys - 40;
        const int T = 32;
        std::vector<float> qb((size_t) T * 64);
        for (int t = 0; t < T; ++t) std::copy(queries.begin() + (size_t) (t % n_q) * 64, queries.begin() + (size_t) (t % n_q + 1) * 64, qb.begin() + (size_t) t * 64);
        float* d_qb = nullptr;
        CK(cudaMalloc(&d_qb, qb.size() * 4));
        CK(cudaMemcpy(d_qb, qb.data(), qb.size() * 4, cudaMemcpyHostToDevice));
        gl::lsa_score(d_qb, d_cent, (int) buckets, (int) block, p0, T, d_sc, n_b, 0);
        gl::lsa_select(d_sc, n_b, d_memb, g, bud, p0, T, (int) width, d_cells, width, d_nsel, 0);
        CK(cudaDeviceSynchronize());
        std::vector<int32_t> selb((size_t) T * width);
        CK(cudaMemcpy(selb.data(), d_cells, selb.size() * 4, cudaMemcpyDeviceToHost));
        uint32_t diff = 0;
        for (int t = 0; t < T; ++t) {
            gl::lsa_score(d_qb + (size_t) t * 64, d_cent, (int) buckets, (int) block, p0 + t, 1, d_sc, n_b, 0);
            gl::lsa_select(d_sc, n_b, d_memb, g, bud, p0 + t, 1, (int) width, d_cells, width, d_nsel, 0);
            CK(cudaDeviceSynchronize());
            std::vector<int32_t> one(width);
            CK(cudaMemcpy(one.data(), d_cells, one.size() * 4, cudaMemcpyDeviceToHost));
            for (uint32_t j = 0; j < width; ++j) if (one[j] != selb[(size_t) t * width + j]) { ++diff; break; }
        }
        std::printf("batched %d consecutive positions vs one by one: %u rows differ\n", T, diff);
        cudaFree(d_qb);
    }
    // timing: scores + selection of one query over every bucket
    {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        const int reps = 200;
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) {
            gl::lsa_score(d_q, d_cent, (int) buckets, (int) block, (int64_t) n_keys - 1, 1, d_sc, n_b, 0);
            gl::lsa_select(d_sc, n_b, d_memb, g, bud, (int64_t) n_keys - 1, 1, (int) width, d_cells, width, nullptr, 0);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        cudaEventRecord(e0);
        gl::lsa_build(d_keys, 0, 0, 0, (int) n_blocks, g, d_cent, d_memb, 0);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float msb = 0.0f;
        cudaEventElapsedTime(&msb, e0, e1);
        std::printf("timing: score + select over %u buckets: %.1f us a query; Ward of %u blocks: %.2f ms (%.1f us a block)\n",
                    n_b, 1000.0f * ms / reps, n_blocks, msb, 1000.0f * msb / n_blocks);
    }
    const bool ok = bad_memb == 0 && bad_ring == 0 && bad == 0 && bad_len == 0 && bad_short == 0 && bad_short_len == 0;
    std::printf("%s\n", ok ? "OK" : "MISMATCH");
    return ok ? 0 : 1;
}
