// src/core/glm_lsa.cu - Latent Sparse Attention (D-LSA Flat) on the fast path's DSA layers; see
// include/strata/kernels/glm_lsa.hpp for the design and src/kernels/cuda/glm_lsa.cu for the kernels.
//
// STRATA_GLM_LSA=<map .lsa> (generate: --lsa FILE; P:\llm\lsa\work\lsa_convert.py turns `lsa distill export`'s torch
// file into it) switches the trunk's DSA layers that the map covers to LSA's selection: the native indexer still runs
// (its key / gate caches and pools stay valid, so snapshots, checkpoints and slots are what they were and a slot saved
// here loads without LSA), only its scores and selection are replaced.  The NextN draft block, which the map does not
// cover, keeps the native indexer.
//
// Per sequence and layer the engine keeps: the index KEYS of the last `ring` positions (written as the positions'
// latents are, from kv_a's raw rows), the BUCKETS of every closed cluster block (centroids and member positions,
// built when the first query past the block arrives - from the ring, or from the latent cache after a restore that
// replaced the latents: a checkpoint, a slot) and how many blocks stand (`built`).  A rollback of the position (a
// verify window's rejected drafts, a snapshot) only lowers `built` to the blocks the kept positions close - the
// buckets of a block are a function of its 512 keys alone, so the ones still closed are untouched.
#include "glm_fast_state.hpp"

#include "strata/kernels/glm_lsa.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace strata::core {

namespace gl = strata::kernels::glml;

struct Glm5Model::Lsa {
    std::string path;
    gl::LsaGeom geom;
    gl::LsaBudget bud;
    int width = 0;             // cells a selection (top_k + window + sink of the widest budget in the context)
    int input_dim = 0;         // the projections' input: kv_lora
    int ring = 0;              // positions of index keys kept
    std::vector<int> layers;   // the map's layers, file order
    std::vector<std::vector<float>> WQ_h, WK_h;   // their projections [input_dim x 64], row major
    // per layer index (n_layers + 1 entries; null: the layer is not LSA's here)
    std::vector<float*> WQ, WK, M;
    struct Seq {
        int64_t ctx = 0, max_blocks = 0;
        void* arena = nullptr;
        std::vector<float*> keys, cent;
        std::vector<int*> memb;
        std::vector<int64_t> built;       // the buckets of blocks [0, built) stand
        std::vector<int64_t> ring_from;   // the ring holds the keys of positions from here (below: the latent cache's)
    };
    std::vector<Seq> seqs;
    // scratch (this part, every sequence): index queries of up to q_rows tokens, the fold path's partials for up to
    // part_rows, scores of up to score_rows tokens (the verify window's), keys of up to tmp_blocks cluster blocks
    // from the latent cache
    float* q_idx = nullptr;
    int q_rows = 0;
    float* part = nullptr;
    int part_rows = 0;
    float* scores = nullptr;
    int score_rows = 0;
    int64_t score_ld = 0;
    float* keys_tmp = nullptr;
    int tmp_blocks = 0;
    double build_ms = 0;   // (statistics)
    int64_t blocks_built = 0, blocks_from_lat = 0;
};

namespace {
constexpr uint32_t kLsaMagic = 0x4D41534Cu;   // 'LSAM'

template <typename T>
bool rd(std::ifstream& f, T& v) {
    return (bool) f.read((char*) &v, sizeof(T));
}
}  // namespace

int Glm5Model::lsa_width() const { return lsa_ != nullptr ? lsa_->width : 0; }

bool Glm5Model::lsa_load(int64_t max_ctx, std::string& err) {
    const char* path = getenv("STRATA_GLM_LSA");
    if (path == nullptr || path[0] == '\0' || std::strcmp(path, "off") == 0 || std::strcmp(path, "0") == 0) return true;
    if (!fast_mode_) {
        std::fprintf(stderr, "glm lsa: the reference path (STRATA_GLM_SLOW) keeps the native indexer - %s ignored\n", path);
        return true;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = std::string("lsa: cannot read ") + path;
        return false;
    }
    auto L = std::make_unique<Lsa>();
    L->path = path;
    uint32_t magic = 0, version = 0, n_layers = 0, input_dim = 0, index_dim = 0;
    uint32_t pool = 0, block = 0, buckets = 0, cap = 0, window = 0, sink = 0, top_k = 0, top_p = 0, late_k = 0, late_p = 0;
    uint64_t late_pos = 0;
    if (!rd(f, magic) || !rd(f, version) || !rd(f, n_layers) || !rd(f, input_dim) || !rd(f, index_dim) || !rd(f, pool) ||
        !rd(f, block) || !rd(f, buckets) || !rd(f, cap) || !rd(f, window) || !rd(f, sink) || !rd(f, top_k) ||
        !rd(f, top_p) || !rd(f, late_pos) || !rd(f, late_k) || !rd(f, late_p)) {
        err = std::string("lsa: ") + path + " is too short for a header";
        return false;
    }
    if (magic != kLsaMagic || version != 1) {
        err = std::string("lsa: ") + path + " is not a Maya LSA map (lsa_convert.py writes one)";
        return false;
    }
    const Glm5Geometry& g = g_;
    if ((int) input_dim != g.kv_lora || (int) index_dim != gl::kLsaDim) {
        err = "lsa: the map projects [" + std::to_string(input_dim) + " x " + std::to_string(index_dim) +
              "], this model's latent is " + std::to_string(g.kv_lora) + " wide and the kernels index " +
              std::to_string(gl::kLsaDim) + " dims";
        return false;
    }
    if (pool < 1 || block < pool || block % pool != 0 || block / pool > (uint32_t) gl::kLsaPoolsMax || buckets < 1 ||
        buckets > block / pool || cap < 2 * (block / buckets) || cap > (uint32_t) gl::kLsaCapMax || window < 1 ||
        top_k < 1 || top_p < 1 || top_p > 1024 || late_k < 1 || late_p < 1 || late_p > 1024 || n_layers < 1 ||
        n_layers > 4096 || g.kv_lora % 32 != 0 || g.kv_lora > 1024) {
        err = std::string("lsa: ") + path + " holds sizes these kernels do not run (pool " + std::to_string(pool) +
              ", block " + std::to_string(block) + ", buckets " + std::to_string(buckets) + ", cap " + std::to_string(cap) +
              ", top_p " + std::to_string(top_p) + " / " + std::to_string(late_p) + ")";
        return false;
    }
    L->geom.pool = (int) pool;
    L->geom.block = (int) block;
    L->geom.buckets = (int) buckets;
    L->geom.cap = (int) cap;
    L->geom.window = (int) window;
    L->geom.sink = (int) sink;
    L->bud.top_k = (int) top_k;
    L->bud.top_p = (int) top_p;
    L->bud.late_pos = (int64_t) late_pos;
    L->bud.late_top_k = (int) late_k;
    L->bud.late_top_p = (int) late_p;
    L->input_dim = (int) input_dim;
    const size_t n = (size_t) input_dim * index_dim;
    for (uint32_t i = 0; i < n_layers; ++i) {
        int32_t layer = -1;
        std::vector<float> wq(n), wk(n);
        if (!rd(f, layer) || !f.read((char*) wq.data(), (std::streamsize) (n * sizeof(float))) ||
            !f.read((char*) wk.data(), (std::streamsize) (n * sizeof(float)))) {
            err = std::string("lsa: ") + path + " ends inside layer " + std::to_string(i);
            return false;
        }
        L->layers.push_back(layer);
        L->WQ_h.push_back(std::move(wq));
        L->WK_h.push_back(std::move(wk));
    }
    // this part's trunk DSA layers must all be in the map (the NextN block is never: it keeps the native indexer)
    lsa_on_.assign((size_t) g.n_layers + 1, 0);
    L->WQ.assign((size_t) g.n_layers + 1, nullptr);
    L->WK.assign((size_t) g.n_layers + 1, nullptr);
    L->M.assign((size_t) g.n_layers + 1, nullptr);
    std::string missing;
    int n_on = 0;
    cudaSetDevice(dev_);
    for (int il = l0_; il < l1_; ++il) {
        if (g.is_recr(il)) continue;
        const auto it = std::find(L->layers.begin(), L->layers.end(), il);
        if (it == L->layers.end()) {
            missing += (missing.empty() ? "" : ", ") + std::to_string(il);
            continue;
        }
        const size_t k = (size_t) (it - L->layers.begin());
        float *wq = nullptr, *wk = nullptr;
        if (cudaMalloc(&wq, n * sizeof(float)) != cudaSuccess || cudaMalloc(&wk, n * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(wq, L->WQ_h[k].data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(wk, L->WK_h[k].data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError();
            err = "lsa: the projections did not allocate on CUDA" + std::to_string(dev_);
            return false;
        }
        L->WQ[(size_t) il] = wq;
        L->WK[(size_t) il] = wk;
        lsa_on_[(size_t) il] = 1;
        ++n_on;
    }
    if (!missing.empty()) {
        err = std::string("lsa: ") + path + " has no projections for DSA layers " + missing + " of this model";
        return false;
    }
    // the widest selection a position of this context takes: the late budget's from late_pos on
    const bool late = max_ctx > L->bud.late_pos;
    L->width = (late ? L->bud.late_top_k : L->bud.top_k) + L->geom.window + L->geom.sink;
    g_.lsa_n_sel = L->width;
    // the key ring: the prompt path's largest sub-batch (<= 8192) and the block before it, rounded to blocks
    const int64_t want = std::min<int64_t>((max_ctx + block - 1) / block * block, 8192 + 2 * (int64_t) block);
    L->ring = (int) std::max<int64_t>(2 * block, want);
    // scratch
    const int64_t max_blocks = max_ctx / block + 1;
    L->q_rows = (int) std::min<int64_t>(8192, std::max<int64_t>(64, max_ctx));
    L->part_rows = 16;
    L->score_rows = 16;
    L->score_ld = max_blocks * buckets;
    L->tmp_blocks = 16;
    if (cudaMalloc(&L->q_idx, (size_t) L->q_rows * gl::kLsaDim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&L->part, (size_t) L->part_rows * 16 * gl::kLsaDim * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&L->scores, (size_t) L->score_rows * L->score_ld * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&L->keys_tmp, (size_t) L->tmp_blocks * block * gl::kLsaDim * sizeof(float)) != cudaSuccess) {
        cudaGetLastError();
        err = "lsa: the scratch did not allocate on CUDA" + std::to_string(dev_);
        return false;
    }
    lsa_ = L.release();
    if (!lsa_seq_alloc(0, max_ctx, err)) return false;
    std::fprintf(stderr, "glm lsa: CUDA%d %d DSA layers select by LSA (%s): index %d, pools of %d, blocks of %d -> %d "
                         "buckets (cap %d), window %d + %d sinks, top-k %d / top-p %d%s, %d cells a query, keys ring %d, "
                         "buckets of %lld blocks a layer (%.1f MB a sequence)\n",
                 dev_, n_on, path, gl::kLsaDim, (int) pool, (int) block, (int) buckets, (int) cap, (int) window, (int) sink,
                 (int) top_k, (int) top_p, late ? (" (" + std::to_string(late_k) + " / " + std::to_string(late_p) +
                                                    " from " + std::to_string(late_pos) + ")").c_str() : "",
                 lsa_->width, lsa_->ring, (long long) (max_ctx / block),
                 (double) n_on * ((double) lsa_->ring * gl::kLsaDim * 4 + (double) max_blocks * buckets * (gl::kLsaDim * 4 + cap * 4)) /
                     1048576.0);
    return true;
}

bool Glm5Model::lsa_seq_alloc(int seq, int64_t ctx, std::string& err) {
    Lsa& L = *lsa_;
    if ((int) L.seqs.size() <= seq) L.seqs.resize((size_t) seq + 1);
    Lsa::Seq& S = L.seqs[(size_t) seq];
    if (S.arena != nullptr) return true;
    const Glm5Geometry& g = g_;
    S.ctx = ctx;
    S.max_blocks = ctx / L.geom.block + 1;
    const size_t nl = (size_t) g.n_layers + 1;
    S.keys.assign(nl, nullptr);
    S.cent.assign(nl, nullptr);
    S.memb.assign(nl, nullptr);
    S.built.assign(nl, 0);
    S.ring_from.assign(nl, 0);
    const size_t keys_b = (size_t) L.ring * gl::kLsaDim * sizeof(float);
    const size_t cent_b = (size_t) S.max_blocks * L.geom.buckets * gl::kLsaDim * sizeof(float);
    const size_t memb_b = (size_t) S.max_blocks * L.geom.buckets * L.geom.cap * sizeof(int);
    const auto up = [](size_t b) { return (b + 255) / 256 * 256; };
    size_t bytes = 0;
    for (int il = l0_; il < l1_; ++il)
        if (lsa_on(il)) bytes += up(keys_b) + up(cent_b) + up(memb_b);
    if (bytes == 0) return true;
    cudaSetDevice(dev_);
    uint8_t* base = nullptr;
    if (cudaMalloc(&base, bytes) != cudaSuccess) {
        cudaGetLastError();
        err = "lsa: the buckets of sequence " + std::to_string(seq) + " (" + std::to_string(bytes >> 20) +
              " MB) do not fit on CUDA" + std::to_string(dev_);
        return false;
    }
    cudaMemset(base, 0, bytes);
    S.arena = base;
    size_t at = 0;
    for (int il = l0_; il < l1_; ++il) {
        if (!lsa_on(il)) continue;
        S.keys[(size_t) il] = (float*) (base + at);
        at += up(keys_b);
        S.cent[(size_t) il] = (float*) (base + at);
        at += up(cent_b);
        S.memb[(size_t) il] = (int*) (base + at);
        at += up(memb_b);
    }
    return true;
}

// fast_setup, after the layer weights are bound: each LSA layer's fold M = k_b . W_Q
bool Glm5Model::lsa_fold_all(std::string& err) {
    if (lsa_ == nullptr || fast_ == nullptr) return true;
    Lsa& L = *lsa_;
    const Glm5Geometry& g = g_;
    FastState* F = fast_;
    cudaSetDevice(dev_);
    const size_t m_floats = (size_t) g.n_head * g.qk_nope * gl::kLsaDim;
    for (int il = l0_; il < l1_; ++il) {
        if (!lsa_on(il)) continue;
        const auto& Ly = F->L[(size_t) il];
        if (Ly.k_b == nullptr) {
            err = "lsa: layer " + std::to_string(il) + " has no BF16 k_b for the fold";
            return false;
        }
        float* M = L.M[(size_t) il];
        if (M == nullptr && cudaMalloc(&M, m_floats * sizeof(float)) != cudaSuccess) {
            cudaGetLastError();
            err = "lsa: the folds did not allocate on CUDA" + std::to_string(dev_);
            return false;
        }
        L.M[(size_t) il] = M;
        gl::lsa_fold(Ly.k_b, g.n_head, g.kv_lora, g.qk_nope, L.WQ[(size_t) il], M, F->cs);
    }
    if (!glmfast::cuda_ok(cudaStreamSynchronize(F->cs), "lsa: the folds", err)) return false;
    return true;
}

void Glm5Model::lsa_invalidate(int seq, int64_t pos) {
    if (lsa_ == nullptr || seq < 0 || (size_t) seq >= lsa_->seqs.size()) return;
    Lsa::Seq& S = lsa_->seqs[(size_t) seq];
    for (size_t il = 0; il < S.built.size(); ++il) {
        S.built[il] = 0;
        S.ring_from[il] = pos;
    }
}

void Glm5Model::lsa_free() {
    if (lsa_ == nullptr) return;
    cudaSetDevice(dev_);
    Lsa& L = *lsa_;
    for (auto& S : L.seqs)
        if (S.arena) cudaFree(S.arena);
    for (float* p : L.WQ) if (p) cudaFree(p);
    for (float* p : L.WK) if (p) cudaFree(p);
    for (float* p : L.M) if (p) cudaFree(p);
    if (L.q_idx) cudaFree(L.q_idx);
    if (L.part) cudaFree(L.part);
    if (L.scores) cudaFree(L.scores);
    if (L.keys_tmp) cudaFree(L.keys_tmp);
    delete lsa_;
    lsa_ = nullptr;
    lsa_on_.clear();
}

void Glm5Model::lsa_keys(int il, int seq, const float* kv_raw, int64_t ld, int T, int64_t p0, cudaStream_t s) {
    Lsa& L = *lsa_;
    const Glm5Geometry& g = g_;
    const auto& Ly = fast_->L[(size_t) il];
    gl::lsa_keys_raw(kv_raw, ld, Ly.kv_a_norm, g.norm_eps, g.kv_lora, lat_q8_, L.WK[(size_t) il], T, p0, L.ring,
                     L.seqs[(size_t) seq].keys[(size_t) il], s);
}

bool Glm5Model::lsa_select(int il, int seq, int64_t p0, int T, const float* q, const float* q_abs, int64_t ld_q,
                           float* scores, int64_t ld_scores, int* cells, int64_t ld_cells, int* n_sel,
                           const uint16_t* lat_rows, std::string& err, cudaStream_t s) {
    Lsa& L = *lsa_;
    const Glm5Geometry& g = g_;
    if (seq < 0 || (size_t) seq >= L.seqs.size() || L.seqs[(size_t) seq].arena == nullptr) {
        err = "lsa: sequence " + std::to_string(seq) + " has no bucket state";
        return false;
    }
    Lsa::Seq& S = L.seqs[(size_t) seq];
    const int block = L.geom.block;
    const int64_t last = p0 + T - 1;
    const int64_t need = last / block;   // the blocks closed before the last query's own: [0, need)
    if (need > S.max_blocks) {
        err = "lsa: position " + std::to_string(last) + " is past the sequence's context";
        return false;
    }
    int64_t& built = S.built[(size_t) il];
    if (built > p0 / block) built = p0 / block;   // (a rollback: the blocks the kept positions do not close are gone)
    float* const cent = S.cent[(size_t) il];
    int* const memb = S.memb[(size_t) il];
    while (built < need) {
        const int64_t b = built;
        // the ring holds the block when its keys were written after the last restore and are not overwritten yet
        const bool from_ring = (int64_t) b * block >= S.ring_from[(size_t) il] && (int64_t) b * block > last - L.ring;
        if (from_ring) {
            gl::lsa_build(S.keys[(size_t) il], L.ring, 0, b, (int) (need - b), L.geom, cent, memb, s);
            L.blocks_built += need - b;
            built = need;
            break;
        }
        if (lat_rows == nullptr) {
            err = "lsa: layer " + std::to_string(il) + " needs its latent cache to rebuild the buckets of block " +
                  std::to_string(b);
            return false;
        }
        // from the latent cache, tmp_blocks blocks a launch: those whose keys the ring does not hold
        int64_t b1 = b + 1;
        while (b1 < need && b1 - b < L.tmp_blocks &&
               !((int64_t) b1 * block >= S.ring_from[(size_t) il] && (int64_t) b1 * block > last - L.ring))
            ++b1;
        const int nb = (int) (b1 - b);
        gl::lsa_keys_lat(lat_rows, lat_q8_, g.kv_lora, (int64_t) b * block, nb * block, L.WK[(size_t) il], 0, 0,
                         L.keys_tmp, s);
        gl::lsa_build(L.keys_tmp, 0, (int64_t) b * block, b, nb, L.geom, cent, memb, s);
        L.blocks_built += nb;
        L.blocks_from_lat += nb;
        built = b1;
    }
    // below top_k + block every key is selected (the closed blocks' keys fit the top-k, the open block lies in the
    // window) - the native path's "select all" of a short context, without the queries and scores
    const int64_t all_below = (int64_t) (p0 >= L.bud.late_pos ? L.bud.late_top_k : L.bud.top_k) + block;
    if (last < all_below) {
        gl::lsa_select_all(p0, T, L.width, cells, ld_cells, n_sel, s);
        return true;
    }
    // the queries, scores and selections, in pieces the scratch holds
    float* const sc = scores != nullptr ? scores : L.scores;
    const int64_t sld = scores != nullptr ? ld_scores : L.score_ld;
    int piece = std::min(T, L.q_rows);
    if (scores == nullptr) piece = std::min(piece, L.score_rows);
    if (q_abs == nullptr) piece = std::min(piece, L.part_rows);
    const int K = g.n_head * g.qk_nope;
    for (int t0 = 0; t0 < T; t0 += piece) {
        const int tn = std::min(piece, T - t0);
        if (q_abs != nullptr)
            gl::lsa_query_abs(q_abs + (size_t) t0 * ld_q, ld_q, g.n_head, g.kv_lora, L.WQ[(size_t) il], tn, L.q_idx, s);
        else
            gl::lsa_query_fold(q + (size_t) t0 * ld_q, ld_q, K, L.M[(size_t) il], tn, L.part, L.q_idx, s);
        float* const sct = scores != nullptr ? sc + (size_t) t0 * sld : sc;
        gl::lsa_score(L.q_idx, cent, L.geom.buckets, block, p0 + t0, tn, sct, sld, s);
        gl::lsa_select(sct, sld, memb, L.geom, L.bud, p0 + t0, tn, L.width, cells + (size_t) t0 * ld_cells, ld_cells,
                       n_sel != nullptr ? n_sel + t0 : nullptr, s);
    }
    return true;
}

void Glm5Model::lsa_stats(uint64_t& blocks, uint64_t& from_lat) const {
    blocks = lsa_ != nullptr ? (uint64_t) lsa_->blocks_built : 0;
    from_lat = lsa_ != nullptr ? (uint64_t) lsa_->blocks_from_lat : 0;
}

}  // namespace strata::core
