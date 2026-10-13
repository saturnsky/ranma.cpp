#pragma once

// RANMA_PREFILL_GPU_STAMP (ranma, HIP only): GPU stream time of a prefill ubatch per layer, attention chunk and section.
// Off by default. It needs RANMA_PREFILL_ACCOUNT=1 (ggml-account.h): the accounting record names the ubatch and its
// token count, and the totals fill its gpu_* fields.
//
//   RANMA_PREFILL_GPU_STAMP=1             stamps for every ubatch of at least RANMA_PREFILL_GPU_STAMP_MIN_TOKENS (64)
//   RANMA_PREFILL_GPU_STAMP=2             the same and the interval of every stamped launch ("nodes")
//   RANMA_PREFILL_FA_OVERLAP=1            (with mode 1 or 2) per flash attention call: how much the queries of an aligned
//                                         16-row block share their visible columns (the top-k lists of a CSA/HCA layer).
//                                         Adds "ov":[[layer,chunk,rows,n_comp,v0..v51],...] to the line. v0..v3: sums over
//                                         all columns of N (selected cells), T (pairs of rows x shared columns), U16 (union
//                                         per 16-row block), U4 (union per 4-row group); v4..v7 the same for the columns
//                                         below n_comp (the compressed rows = the top-k part); v8..v27 histogram of the
//                                         per-block overlap T/(pairs*N/rows) of that part in steps of 0.05; v28..v51
//                                         histogram of its per-block union / mean list length, bin = floor((x-1)*1.5)
//   RANMA_PREFILL_GPU_STAMP_OUT=<path>  the records go to this file (JSONL), else to stderr; with mode 2 one node table
//                                         per distinct graph goes to <path>.nodes.<n_nodes>.json
//
// A one-thread kernel stores the GPU wall clock into mapped coherent host memory before every executed node of a CUDA
// graph call, once more in front of the main kernel of a flash attention launch (so that mask compaction and the K/V
// conversion are told apart from the kernel) and once at the end of the call. hipEvents are not used: on this stack
// their time is not aligned with the completion of the kernels in front of them (docs/hip-copy-api-overheads.ko.md 2.7).
// Nothing synchronizes per op: the host reads a record once its last stamp has landed, at the start of a later graph
// call or right after a stream synchronize, and a ubatch may consist of several graph calls (scheduler splits). A
// stamped call runs without CUDA graph capture.
//
// The section of a node comes from the cb() names of the model graph (DeepSeek V4 names, others land in "other"): an
// op override (FLASH_ATTN_EXT, LIGHTNING_INDEXER, DSV4_*, MOE_FFN_HAD, MUL_MAT_ID), then the name, and an unnamed node
// takes the section and layer of the next named node. Inside a layer the k-th FLASH_ATTN_EXT closes attention chunk k:
// the chunk-local sections in front of it belong to chunk k, the output projection after it as well.
//
// Record (one JSON line per ubatch):
//   {"kind":"prefill_gpu_stamp","v":1,"seq":n,"key":k,"tokens":t,"calls":c,"n_exec":n,"stamps":n,"overflow":0,
//    "concurrent":n,"cal_us":[a,b],"stream_ms":f,"group_ms":{"attn":f,"moe":f,"other":f},
//    "class_ms":{"<class>":f,...},"kind_ms":{"raw":{...},"csa":{...},"hca":{...}},
//    "segs":[[layer,chunk,class,us,launches],...],
//    "fa":[[layer,chunk,i0,n,n_kv,n_kv_max,mask_ne0,kernel,sparse,ncols1,ncols2,nbatch_fa,nwarps,gx,gy,gz,stream_k,
//           k_cell0,nj,n_comp,prep_us,kernel_us],...],
//    "lid":[[layer,chunk,rows,width,k,fused],...],
//    "moe":[[layer,tokens,n_used,rows,chunk_rows,mt_v,mt_r,path,remote,n_chunks,us],...],
//    "ops":{"FLASH_ATTN_EXT":n,"LIGHTNING_INDEXER":n,"DSV4_COMPRESS":n,"MOE_FFN_HAD":n,"MUL_MAT_ID":n,"CONCAT":n},
//    "nodes":[[node,class,us,fused],...]}   (mode 2)
// chunk -1: a whole-ubatch section. kernel: BEST_FATTN_KERNEL_* of fattn.cu (-1 unknown). path: 0 direct, 1 staged,
// 2 streamed. n_chunks: the chunk table of the expert-major op as the routing kernel left it. Each interval includes
// one stamp launch (cal_us: two back-to-back stamp intervals of the record's first call).

#include "common.cuh"

#include <cstdint>

// hooks of the other translation units; each returns at once when the switch is off
void ggml_cuda_pstamp_fa_kernel(int kernel);
void ggml_cuda_pstamp_fa(cudaStream_t stream, const ggml_tensor * dst, bool sparse, int ncols1, int ncols2, int nbatch_fa,
                         int nwarps, int gx, int gy, int gz, bool stream_k, int64_t n_kv);
void ggml_cuda_pstamp_moe(cudaStream_t stream, const int * n_chunks_dev, int64_t n_tokens, int n_used, int64_t n_rows,
                          int64_t chunk_rows, int mt_v, int mt_r, int path, bool remote);
// RANMA_PREFILL_FA_OVERLAP=1 (needs the stamp): after the launches of a flash attention call, aggregates on the GPU how
// much the 16-row blocks of the mask share their visible columns; no host synchronization
void ggml_cuda_pstamp_fa_overlap(ggml_backend_cuda_context & ctx, cudaStream_t stream, const ggml_tensor * dst);

#if defined(GGML_CUDA_PSTAMP_IMPL)

#include "ggml-account.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#if defined(GGML_USE_HIP)

namespace ggml_cuda_pstamp {

enum cls_t : uint8_t {
    C_HC_NORM, C_QR_KV, C_Q, C_COMP_PROJ, C_COMP, C_LID_Q, C_LID_SCORE, C_LID_TOPK, C_MASK, C_KV_CONCAT, C_MASK_CONCAT,
    C_FA_PREP, C_FA, C_O_PROJ, C_ROUTER, C_MOE, C_SHEXP, C_HEAD, C_OTHER, C_COUNT,
    C_CAL = 254, C_END = 255,
};

static const char * const cls_names[C_COUNT] = {
    "hc_norm", "qr_kv", "q", "comp_proj", "comp", "lid_q", "lid_score", "lid_topk", "mask", "kv_concat", "mask_concat",
    "fa_prep", "fa", "o_proj", "router", "moe", "shexp", "head", "other",
};

// 0 attention, 1 routed MoE (router and experts), 2 other
static int group_of(int c) {
    return c >= C_QR_KV && c <= C_O_PROJ ? 0 : c == C_ROUTER || c == C_MOE ? 1 : 2;
}

// the sections that repeat per attention chunk
static bool chunk_local(int c) {
    return c == C_Q || c == C_LID_Q || c == C_LID_SCORE || c == C_LID_TOPK || c == C_MASK || c == C_KV_CONCAT ||
           c == C_MASK_CONCAT || c == C_FA || c == C_O_PROJ;
}

static int mode() {
    static const int m = [] {
        const char * v = getenv("RANMA_PREFILL_GPU_STAMP");
        const int k = v != nullptr ? atoi(v) : 0;
        if (k <= 0) {
            return 0;
        }
        if (!ggml_account_enabled()) {
            GGML_LOG_WARN("RANMA_PREFILL_GPU_STAMP needs RANMA_PREFILL_ACCOUNT=1; off\n");
            return 0;
        }
        return k >= 2 ? 2 : 1;
    }();
    return m;
}

static int min_tokens() {
    static const int n = [] {
        const char * v = getenv("RANMA_PREFILL_GPU_STAMP_MIN_TOKENS");
        return v != nullptr ? std::max(1, atoi(v)) : 64;
    }();
    return n;
}

static const char * out_path() {
    static const char * p = [] {
        const char * v = getenv("RANMA_PREFILL_GPU_STAMP_OUT");
        return v != nullptr && v[0] != '\0' ? v : (const char *) nullptr;
    }();
    return p;
}

static constexpr uint32_t n_slots = 1u << 17;   // 1 MiB of stamps

struct ent_t {
    uint32_t slot;
    int32_t  node;    // -1 for cal/end
    int16_t  layer;
    int16_t  chunk;
    uint8_t  cls;
    uint8_t  kind;    // 0 raw, 1 csa, 2 hca
    int16_t  fused;
};

struct fa_t {
    int v[20];        // layer .. n_comp, see the header comment
    int ent_prep = -1, ent_kern = -1;
    uint32_t ov_slot = UINT32_MAX;   // first of ov_n slots of RANMA_PREFILL_FA_OVERLAP
};

static constexpr int ov_n = 52;

struct moe_t {
    int v[9];         // layer, tokens, n_used, rows, chunk_rows, mt_v, mt_r, path, remote
    uint32_t val_slot;
    int ent = -1;
};

struct rec_t {
    uint64_t key = 0;
    int tokens = 0;
    int calls = 0;
    int n_exec = 0;
    int concurrent = 0;
    bool overflow = false;
    uint32_t last_slot = UINT32_MAX;
    std::vector<ent_t> ents;
    std::vector<fa_t> fa;
    std::vector<moe_t> moe;
    std::vector<std::vector<int>> lid;
    int ops[6] = {};
};

// per graph: the section, layer, chunk and FA facts of every node
struct graph_info {
    const ggml_cgraph * g = nullptr;
    int n = -1;
    const ggml_tensor * first = nullptr, * last = nullptr;
    std::vector<uint8_t> cls;
    std::vector<int16_t> layer;
    std::vector<int16_t> chunk;
    std::vector<uint8_t> kind;
    std::map<int, std::vector<int>> fa;   // node -> layer, chunk, i0, n, n_kv, n_kv_max, mask_ne0, k_cell0, nj, n_comp
    std::vector<std::vector<int>> lid;
    int ops[6] = {};
};

struct state_t {
    std::mutex mutex;
    bool init = false, failed = false;
    int device = -1;
    double ticks_per_us = 0.0;
    uint64_t * host = nullptr;
    uint64_t * dev  = nullptr;
    uint32_t pos = 0;
    FILE * file = nullptr;
    uint64_t seq = 0;
    rec_t cur;
    bool cur_open = false;
    bool active = false;          // the current graph call is stamped
    int cur_node = -1;
    int fa_kernel = -1;
    std::vector<rec_t> pending;
    graph_info gi;
    std::vector<int> tables;      // n_nodes of the node tables written
};

static state_t & st() {
    static state_t * s = new state_t;   // never destroyed
    return *s;
}

static __global__ void stamp_kernel(uint64_t * out) {
    const uint64_t t = (uint64_t) wall_clock64();
    *out = t != 0 ? t : 1;
    __threadfence_system();
}

static __global__ void capture_kernel(const int * src, uint64_t * out) {
    *out = (1ull << 63) | (uint64_t) (uint32_t) *src;
    __threadfence_system();
}


static bool overlap_enabled() {
    static const bool e = [] {
        const char * v = getenv("RANMA_PREFILL_FA_OVERLAP");
        return v != nullptr && atoi(v) != 0;
    }();
    return e;
}

// One block per aligned 16-row block of the mask. A column is selected by a row if its mask value is finite (the test
// of the sparse index compaction). acc: 8 u64 sums (see the header), hist: 44 u32 bins.
static __global__ void overlap_kernel(const half * mask, int64_t s31, int ne30, int rows, int n_comp,
                                      unsigned long long * acc, unsigned int * hist) {
    const int row0 = blockIdx.x*16;
    const int m = min(16, rows - row0);
    unsigned long long loc[8] = {};
    for (int j = threadIdx.x; j < ne30; j += blockDim.x) {
        unsigned bits = 0;
        for (int r = 0; r < m; ++r) {
            bits |= (unsigned) isfinite(__half2float(mask[(int64_t) (row0 + r)*s31 + j])) << r;
        }
        if (bits == 0) {
            continue;
        }
        const unsigned c = __popc(bits);
        const unsigned u4 = ((bits & 0xF) != 0) + ((bits & 0xF0) != 0) + ((bits & 0xF00) != 0) + ((bits & 0xF000) != 0);
        const int o = j < n_comp ? 4 : 0;
        loc[o + 0] += c;
        loc[o + 1] += (unsigned long long) c*(c - 1)/2;
        loc[o + 2] += 1;
        loc[o + 3] += u4;
        if (j < n_comp) {   // the all-columns sums include the compressed part
            loc[0] += c;
            loc[1] += (unsigned long long) c*(c - 1)/2;
            loc[2] += 1;
            loc[3] += u4;
        }
    }
    __shared__ unsigned long long sh[8];
    if (threadIdx.x < 8) {
        sh[threadIdx.x] = 0;
    }
    __syncthreads();
    for (int i = 0; i < 8; ++i) {
        atomicAdd(&sh[i], loc[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; ++i) {
            atomicAdd(&acc[i], sh[i]);
        }
        const double pairs = 0.5*m*(m - 1);
        const double len = m > 0 ? (double) sh[4]/m : 0.0;
        if (m >= 2 && len > 0.0) {
            const double rho = (double) sh[5]/(pairs*len);
            atomicAdd(&hist[min(19, (int) (rho*20.0))], 1u);
            const double ratio = (double) sh[6]/len;
            atomicAdd(&hist[20 + min(23, max(0, (int) ((ratio - 1.0)*1.5)))], 1u);
        }
    }
}

static __global__ void overlap_publish(const unsigned long long * acc, const unsigned int * hist, uint64_t * out) {
    const int i = threadIdx.x;
    if (i < 8) {
        out[i] = (1ull << 63) | acc[i];
    } else if (i < ov_n) {
        out[i] = (1ull << 63) | hist[i - 8];
    }
    __threadfence_system();
}

static bool open_once(int device) {
    state_t & s = st();
    if (s.init) {
        return !s.failed && device == s.device;
    }
    s.init = true;
    s.device = device;
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device) != hipSuccess || khz <= 0) {
        (void) hipGetLastError();
        GGML_LOG_WARN("RANMA_PREFILL_GPU_STAMP: no wall clock rate; off\n");
        s.failed = true;
        return false;
    }
    s.ticks_per_us = khz/1000.0;
    if (hipHostMalloc((void **) &s.host, sizeof(uint64_t)*n_slots, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess ||
            hipHostGetDevicePointer((void **) &s.dev, s.host, 0) != hipSuccess) {
        (void) hipGetLastError();
        GGML_LOG_WARN("RANMA_PREFILL_GPU_STAMP: the stamp buffer could not be allocated; off\n");
        s.failed = true;
        return false;
    }
    memset(s.host, 0, sizeof(uint64_t)*n_slots);
    if (out_path() != nullptr) {
        s.file = fopen(out_path(), "ab");
        if (s.file == nullptr) {
            GGML_LOG_WARN("RANMA_PREFILL_GPU_STAMP: cannot open %s, writing to stderr\n", out_path());
        }
    }
    GGML_LOG_INFO("RANMA_PREFILL_GPU_STAMP: mode %d, ubatches of %d tokens or more, wall clock %d kHz, records to %s\n",
        mode(), min_tokens(), khz, s.file != nullptr ? out_path() : "stderr");
    return true;
}

// one stamp (or captured value) slot of the current record, -1 when the buffer is full
static int64_t take_slot() {
    state_t & s = st();
    if (s.pos >= n_slots) {
        s.cur.overflow = true;
        return -1;
    }
    const uint32_t slot = s.pos++;
    s.host[slot] = 0;
    s.cur.last_slot = slot;
    return slot;
}

static void stamp(cudaStream_t stream, int node, uint8_t cls, int16_t layer, int16_t chunk, uint8_t kind) {
    state_t & s = st();
    const int64_t slot = take_slot();
    if (slot < 0) {
        return;
    }
    stamp_kernel<<<1, 1, 0, stream>>>(s.dev + slot);
    s.cur.ents.push_back({ (uint32_t) slot, node, layer, chunk, cls, kind, 0 });
}

// "name-12" -> base "name", layer 12; a derived name ("x-3 (cont)") counts as unnamed
static int split_name(const char * name, char * base, size_t cap, bool & named) {
    const size_t n = strnlen(name, GGML_MAX_NAME);
    named = n > 0 && strstr(name, " (") == nullptr;
    size_t i = n;
    while (i > 0 && name[i - 1] >= '0' && name[i - 1] <= '9') {
        --i;
    }
    int layer = -1;
    size_t len = n;
    if (i > 0 && i < n && name[i - 1] == '-') {
        layer = atoi(name + i);
        len = i - 1;
    }
    len = std::min(len, cap - 1);
    memcpy(base, name, len);
    base[len] = '\0';
    return layer;
}

static bool starts(const char * s, const char * p) {
    return strncmp(s, p, strlen(p)) == 0;
}

static int class_of(const ggml_tensor * t, const char * base, bool named) {
    switch (t->op) {
        case GGML_OP_FLASH_ATTN_EXT:    return C_FA;
        case GGML_OP_LIGHTNING_INDEXER: return C_LID_SCORE;
        case GGML_OP_DSV4_COMPRESS:     return C_COMP;
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
        case GGML_OP_DSV4_HC_COEF:      return C_HC_NORM;
        case GGML_OP_MOE_FFN_HAD:
        case GGML_OP_MUL_MAT_ID:        return C_MOE;
        default: break;
    }
    if (!named) {
        return -1;
    }
    if (strstr(base, "k_all") != nullptr && t->op == GGML_OP_CONCAT) {
        return C_KV_CONCAT;
    }
    if (strstr(base, "kq_mask") != nullptr || !strcmp(base, "chunk_mask")) {
        return t->op == GGML_OP_CONCAT ? C_MASK_CONCAT : C_MASK;
    }
    if (starts(base, "hc_") || !strcmp(base, "attn_norm") || !strcmp(base, "ffn_norm") || !strcmp(base, "l_last") ||
            !strcmp(base, "layer_inp") || !strcmp(base, "result_norm")) {
        return C_HC_NORM;
    }
    if (!strcmp(base, "qr") || !strcmp(base, "qr_norm") || !strcmp(base, "kv") || !strcmp(base, "kv_norm") ||
            !strcmp(base, "kv_rot")) {
        return C_QR_KV;
    }
    if (!strcmp(base, "q") || !strcmp(base, "q_norm") || !strcmp(base, "q_rot")) {
        return C_Q;
    }
    if (starts(base, "csa_state_") || starts(base, "hca_state_") || starts(base, "lid_state_")) {
        return strstr(base, "compress") != nullptr ? C_COMP : C_COMP_PROJ;
    }
    if (starts(base, "lid_kq") || starts(base, "lid_score")) {
        return C_LID_SCORE;
    }
    if (starts(base, "lid_top_k")) {
        return C_LID_TOPK;
    }
    if (starts(base, "lid_")) {
        return C_LID_Q;
    }
    if (starts(base, "attn_")) {
        return C_O_PROJ;
    }
    if (starts(base, "ffn_moe_")) {
        const char * r = base + 8;
        return starts(r, "logits") || starts(r, "probs") || starts(r, "topk") || starts(r, "argsort") ||
               starts(r, "group_topk") || starts(r, "weights") ? C_ROUTER : C_MOE;
    }
    if (!strcmp(base, "ffn_out")) {
        return C_OTHER;
    }
    if (starts(base, "ffn_")) {
        return C_SHEXP;
    }
    if (!strcmp(base, "result_output")) {
        return C_HEAD;
    }
    return -1;
}

// the non-view source behind views/permutes
static const ggml_tensor * unview(const ggml_tensor * t) {
    while (t != nullptr && (t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_RESHAPE ||
                            t->op == GGML_OP_TRANSPOSE) && t->src[0] != nullptr) {
        t = t->src[0];
    }
    return t;
}

static void classify(const ggml_cgraph * g) {
    graph_info & gi = st().gi;
    if (g == gi.g && g->n_nodes == gi.n && g->nodes[0] == gi.first && g->nodes[g->n_nodes - 1] == gi.last) {
        return;
    }
    const int n = g->n_nodes;
    gi.g = g; gi.n = n; gi.first = g->nodes[0]; gi.last = g->nodes[n - 1];
    gi.cls.assign((size_t) n, C_OTHER);
    gi.layer.assign((size_t) n, -1);
    gi.chunk.assign((size_t) n, -1);
    gi.kind.assign((size_t) n, 0);
    gi.fa.clear();
    gi.lid.clear();
    memset(gi.ops, 0, sizeof(gi.ops));

    std::vector<int> own_cls((size_t) n, -1), own_layer((size_t) n, -1);
    std::map<int, uint8_t> layer_kind;
    char base[GGML_MAX_NAME + 1];
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = g->nodes[i];
        bool named = false;
        const int layer = split_name(t->name, base, sizeof(base), named);
        own_cls[(size_t) i] = class_of(t, base, named);
        own_layer[(size_t) i] = named ? layer : -1;
        if (named && layer >= 0) {
            if (starts(base, "csa_") || starts(base, "lid_")) {
                layer_kind[layer] = 1;
            } else if (starts(base, "hca_") && layer_kind[layer] != 1) {
                layer_kind[layer] = 2;
            }
        }
        switch (t->op) {
            case GGML_OP_FLASH_ATTN_EXT:    gi.ops[0]++; break;
            case GGML_OP_LIGHTNING_INDEXER: gi.ops[1]++; break;
            case GGML_OP_DSV4_COMPRESS:     gi.ops[2]++; break;
            case GGML_OP_MOE_FFN_HAD:       gi.ops[3]++; break;
            case GGML_OP_MUL_MAT_ID:        gi.ops[4]++; break;
            case GGML_OP_CONCAT:            gi.ops[5]++; break;
            default: break;
        }
    }
    // an unnamed node belongs to the next named node (names mark the output of a block)
    int cur_cls = C_OTHER, cur_layer = -1;
    for (int i = n - 1; i >= 0; --i) {
        if (own_cls[(size_t) i] >= 0) {
            cur_cls = own_cls[(size_t) i];
        }
        if (own_layer[(size_t) i] >= 0) {
            cur_layer = own_layer[(size_t) i];
        }
        gi.cls[(size_t) i]   = (uint8_t) (own_cls[(size_t) i] >= 0 ? own_cls[(size_t) i] : cur_cls);
        gi.layer[(size_t) i] = (int16_t) (own_layer[(size_t) i] >= 0 ? own_layer[(size_t) i] : cur_layer);
    }
    // chunks: the k-th FLASH_ATTN_EXT of a layer closes chunk k
    std::map<int, int> fa_seen, i0_of;
    std::map<int, bool> lid_fused;
    for (int i = 0; i < n; ++i) {
        if (g->nodes[i]->op == GGML_OP_LIGHTNING_INDEXER) {
            lid_fused[gi.layer[(size_t) i]] = true;
        }
    }
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * t = g->nodes[i];
        const int l = gi.layer[(size_t) i];
        const int c = gi.cls[(size_t) i];
        gi.kind[(size_t) i] = layer_kind.count(l) ? layer_kind[l] : 0;
        if (!chunk_local(c)) {
            continue;
        }
        int & seen = fa_seen[l];
        gi.chunk[(size_t) i] = (int16_t) (c == C_O_PROJ ? std::max(0, seen - 1) : seen);
        if (c == C_LID_TOPK && t->op == GGML_OP_TOP_K && t->src[0] != nullptr) {
            gi.lid.push_back({ l, seen, (int) t->src[0]->ne[1], (int) t->src[0]->ne[0], (int) t->ne[0], lid_fused[l] ? 1 : 0 });
        }
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            const ggml_tensor * Q = t->src[0], * K = t->src[1], * mask = t->src[3];
            const int64_t n_kv = K->ne[1];
            int k_cell0 = -1, nj = (int) n_kv, n_comp = 0;
            const ggml_tensor * kb = unview(K);
            if (kb != nullptr && kb->op == GGML_OP_CONCAT && kb->src[0] != nullptr && kb->src[1] != nullptr) {
                const int dim = ggml_get_op_params_i32(kb, 0);
                const ggml_tensor * a = kb->src[0];
                nj = (int) a->ne[dim];
                n_comp = (int) kb->src[1]->ne[dim];
                k_cell0 = a->view_src != nullptr && a->nb[dim] > 0 ? (int) (a->view_offs/a->nb[dim]) : 0;
            } else if (kb != nullptr && kb->view_src != nullptr) {
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    if (kb->ne[d] == n_kv && kb->nb[d] > 0) {
                        k_cell0 = (int) (kb->view_offs/kb->nb[d]);
                        break;
                    }
                }
            }
            int & i0 = i0_of[l];
            gi.fa[i] = { l, seen, i0, (int) Q->ne[1], (int) n_kv, ggml_get_op_params_i32(t, 4),
                         mask != nullptr ? (int) mask->ne[0] : 0, k_cell0, nj, n_comp };
            i0 += (int) Q->ne[1];
            seen++;
        }
    }
}

// a finished record: reads the stamps, writes the line, settles the accounting record
static void emit(rec_t & r) {
    state_t & s = st();
    const double tpu = s.ticks_per_us;
    auto val = [&](uint32_t slot) { return s.host[slot]; };
    auto span_us = [&](size_t k) -> double {   // interval of entry k up to the next stamp of the same call
        if (k + 1 >= r.ents.size() || r.ents[k].cls == C_END) {
            return 0.0;
        }
        const uint64_t a = val(r.ents[k].slot), b = val(r.ents[k + 1].slot);
        return b > a ? (double) (b - a)/tpu : 0.0;
    };
    double cls_us[C_COUNT] = {};
    double kind_us[3][C_COUNT] = {};
    double group_us[3] = {};
    double stream_us = 0.0;
    std::vector<double> cal;
    std::map<std::tuple<int, int, int>, std::pair<double, int>> segs;
    std::vector<std::tuple<int, int, int>> seg_order;
    std::vector<double> ent_us(r.ents.size(), 0.0);
    const bool ok = !r.overflow;
    for (size_t k = 0; ok && k < r.ents.size(); ++k) {
        ent_t & e = r.ents[k];
        if (e.cls == C_END) {
            continue;
        }
        if (e.cls == C_FA_PREP && (k + 1 >= r.ents.size() || r.ents[k + 1].node != e.node || r.ents[k + 1].cls != C_FA)) {
            e.cls = C_FA;   // a flash attention path that does not go through launch_fattn: no split
        }
        const double us = span_us(k);
        ent_us[k] = us;
        if (e.cls == C_CAL) {
            if (k + 1 < r.ents.size() && r.ents[k + 1].cls == C_CAL) {
                cal.push_back(us);
            }
            continue;
        }
        stream_us += us;
        cls_us[e.cls] += us;
        kind_us[e.kind][e.cls] += us;
        group_us[group_of(e.cls)] += us;
        const auto key = std::make_tuple((int) e.layer, (int) e.chunk, (int) e.cls);
        auto it = segs.find(key);
        if (it == segs.end()) {
            segs[key] = { us, 1 };
            seg_order.push_back(key);
        } else {
            it->second.first += us;
            it->second.second++;
        }
    }

    std::string out;
    out.reserve(1 << 16);
    char buf[512];
    snprintf(buf, sizeof(buf), "{\"kind\":\"prefill_gpu_stamp\",\"v\":1,\"seq\":%llu,\"key\":%llu,\"tokens\":%d,\"calls\":%d,"
        "\"n_exec\":%d,\"stamps\":%zu,\"overflow\":%d,\"concurrent\":%d,\"cal_us\":[",
        (unsigned long long) s.seq, (unsigned long long) r.key, r.tokens, r.calls, r.n_exec, r.ents.size(),
        r.overflow ? 1 : 0, r.concurrent);
    out += buf;
    for (size_t k = 0; k < cal.size(); ++k) {
        snprintf(buf, sizeof(buf), "%s%.2f", k ? "," : "", cal[k]);
        out += buf;
    }
    snprintf(buf, sizeof(buf), "],\"stream_ms\":%.3f,\"group_ms\":{\"attn\":%.3f,\"moe\":%.3f,\"other\":%.3f},\"class_ms\":{",
        stream_us/1e3, group_us[0]/1e3, group_us[1]/1e3, group_us[2]/1e3);
    out += buf;
    for (int c = 0; c < C_COUNT; ++c) {
        snprintf(buf, sizeof(buf), "%s\"%s\":%.3f", c ? "," : "", cls_names[c], cls_us[c]/1e3);
        out += buf;
    }
    out += "},\"kind_ms\":{";
    static const char * const kinds[3] = { "raw", "csa", "hca" };
    for (int k = 0; k < 3; ++k) {
        out += std::string(k ? "," : "") + "\"" + kinds[k] + "\":{";
        bool first = true;
        for (int c = 0; c < C_COUNT; ++c) {
            if (kind_us[k][c] > 0.0) {
                snprintf(buf, sizeof(buf), "%s\"%s\":%.3f", first ? "" : ",", cls_names[c], kind_us[k][c]/1e3);
                out += buf;
                first = false;
            }
        }
        out += "}";
    }
    out += "},\"segs\":[";
    for (size_t k = 0; k < seg_order.size(); ++k) {
        const auto & sg = segs[seg_order[k]];
        snprintf(buf, sizeof(buf), "%s[%d,%d,\"%s\",%.1f,%d]", k ? "," : "", std::get<0>(seg_order[k]),
            std::get<1>(seg_order[k]), cls_names[std::get<2>(seg_order[k])], sg.first, sg.second);
        out += buf;
    }
    out += "],\"fa\":[";
    for (size_t k = 0; k < r.fa.size(); ++k) {
        const fa_t & f = r.fa[k];
        out += k ? ",[" : "[";
        for (int j = 0; j < 20; ++j) {
            snprintf(buf, sizeof(buf), "%s%d", j ? "," : "", f.v[j]);
            out += buf;
        }
        const double prep = ok && f.ent_prep >= 0 ? ent_us[(size_t) f.ent_prep] : -1.0;
        const double kern = ok && f.ent_kern >= 0 ? ent_us[(size_t) f.ent_kern] : -1.0;
        snprintf(buf, sizeof(buf), ",%.1f,%.1f]", prep, kern);
        out += buf;
    }
    out += "]";
    {
        bool any = false;
        for (size_t k = 0; k < r.fa.size(); ++k) {
            const fa_t & f = r.fa[k];
            if (f.ov_slot == UINT32_MAX) {
                continue;
            }
            out += any ? ",[" : ",\"ov\":[[";
            any = true;
            snprintf(buf, sizeof(buf), "%d,%d,%d,%d", f.v[0], f.v[1], f.v[3], f.v[19]);
            out += buf;
            for (int j = 0; j < ov_n; ++j) {
                snprintf(buf, sizeof(buf), ",%llu", (unsigned long long) (val(f.ov_slot + (uint32_t) j) & ~(1ull << 63)));
                out += buf;
            }
            out += "]";
        }
        if (any) {
            out += "]";
        }
    }
    out += ",\"lid\":[";
    for (size_t k = 0; k < r.lid.size(); ++k) {
        const std::vector<int> & l = r.lid[k];
        snprintf(buf, sizeof(buf), "%s[%d,%d,%d,%d,%d,%d]", k ? "," : "", l[0], l[1], l[2], l[3], l[4], l[5]);
        out += buf;
    }
    out += "],\"moe\":[";
    for (size_t k = 0; k < r.moe.size(); ++k) {
        const moe_t & m = r.moe[k];
        const uint64_t v = val(m.val_slot);
        const int n_chunks = (v >> 63) ? (int) (uint32_t) v : -1;
        out += k ? ",[" : "[";
        for (int j = 0; j < 9; ++j) {
            snprintf(buf, sizeof(buf), "%s%d", j ? "," : "", m.v[j]);
            out += buf;
        }
        snprintf(buf, sizeof(buf), ",%d,%.1f]", n_chunks, ok && m.ent >= 0 ? ent_us[(size_t) m.ent] : -1.0);
        out += buf;
    }
    snprintf(buf, sizeof(buf), "],\"ops\":{\"FLASH_ATTN_EXT\":%d,\"LIGHTNING_INDEXER\":%d,\"DSV4_COMPRESS\":%d,"
        "\"MOE_FFN_HAD\":%d,\"MUL_MAT_ID\":%d,\"CONCAT\":%d}", r.ops[0], r.ops[1], r.ops[2], r.ops[3], r.ops[4], r.ops[5]);
    out += buf;
    if (mode() >= 2) {
        out += ",\"nodes\":[";
        bool first = true;
        for (size_t k = 0; ok && k < r.ents.size(); ++k) {
            const ent_t & e = r.ents[k];
            if (e.cls >= C_COUNT) {
                continue;
            }
            snprintf(buf, sizeof(buf), "%s[%d,\"%s\",%.2f,%d]", first ? "" : ",", e.node, cls_names[e.cls], ent_us[k], e.fused);
            out += buf;
            first = false;
        }
        out += "]";
    }
    out += "}\n";
    FILE * f = s.file != nullptr ? s.file : stderr;
    fwrite(out.data(), 1, out.size(), f);
    fflush(f);

    if (ok) {
        ggml_account_add_for(r.key, GGML_ACCOUNT_GPU_STREAM_MS, stream_us/1e3);
        ggml_account_add_for(r.key, GGML_ACCOUNT_GPU_ATTN_MS,   group_us[0]/1e3);
        ggml_account_add_for(r.key, GGML_ACCOUNT_GPU_MOE_MS,    group_us[1]/1e3);
        ggml_account_add_for(r.key, GGML_ACCOUNT_GPU_OTHER_MS,  group_us[2]/1e3);
    }
    ggml_account_note_for(r.key, "gpu_stamp_seq", std::to_string(s.seq).c_str());
    ggml_account_settle(r.key, 1);
    s.seq++;
}

// writes the finished records whose last stamp has landed (all of them after a stream synchronize)
static void collect(bool synced) {
    state_t & s = st();
    size_t k = 0;
    for (; k < s.pending.size(); ++k) {
        rec_t & r = s.pending[k];
        if (!synced && r.last_slot != UINT32_MAX && __atomic_load_n(&s.host[r.last_slot], __ATOMIC_ACQUIRE) == 0) {
            break;   // records finish in stream order
        }
        emit(r);
    }
    s.pending.erase(s.pending.begin(), s.pending.begin() + k);
}

static void close_cur() {
    state_t & s = st();
    if (s.cur_open) {
        s.pending.push_back(std::move(s.cur));
        s.cur = rec_t();
        s.cur_open = false;
    }
}

static void write_table(const ggml_cgraph * g) {
    state_t & s = st();
    if (mode() < 2 || out_path() == nullptr || s.tables.size() >= 8 ||
            std::find(s.tables.begin(), s.tables.end(), g->n_nodes) != s.tables.end()) {
        return;
    }
    s.tables.push_back(g->n_nodes);
    const std::string p = std::string(out_path()) + ".nodes." + std::to_string(g->n_nodes) + ".json";
    FILE * f = fopen(p.c_str(), "wb");
    if (f == nullptr) {
        return;
    }
    const graph_info & gi = s.gi;
    fprintf(f, "{\"n_nodes\":%d,\"nodes\":[", g->n_nodes);
    for (int i = 0; i < g->n_nodes; ++i) {
        const ggml_tensor * t = g->nodes[i];
        fprintf(f, "%s[%d,\"%s\",\"%s\",%d,%d,\"%s\",\"%s\",[%lld,%lld,%lld,%lld]]", i ? "," : "", i, ggml_op_name(t->op), t->name,
            (int) gi.layer[(size_t) i], (int) gi.chunk[(size_t) i], cls_names[gi.cls[(size_t) i]], ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    }
    fprintf(f, "]}\n");
    fclose(f);
}

// start of a graph call; true when this call is stamped (the caller then runs it without CUDA graph capture)
static bool begin(ggml_backend_cuda_context * ctx, const ggml_cgraph * g) {
    if (mode() == 0 || g->n_nodes == 0) {
        return false;
    }
    state_t & s = st();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.active = false;
    if (!open_once(ctx->device)) {
        return false;
    }
    collect(false);
    const uint64_t key = ggml_account_current();
    if (s.cur_open && s.cur.key != key) {
        close_cur();
        collect(false);
    }
    if (key == 0) {
        return false;
    }
    if (!s.cur_open) {
        const int tokens = ggml_account_tokens(key);
        if (tokens < min_tokens() || !ggml_account_expect(key, 1)) {
            return false;
        }
        if (s.pending.empty()) {
            s.pos = 0;
        }
        s.cur = rec_t();
        s.cur.key = key;
        s.cur.tokens = tokens;
        s.cur_open = true;
    }
    classify(g);
    write_table(g);
    for (int k = 0; k < 6; ++k) {
        s.cur.ops[k] += s.gi.ops[k];
    }
    s.cur.lid.insert(s.cur.lid.end(), s.gi.lid.begin(), s.gi.lid.end());
    if (s.cur.calls++ == 0) {
        for (int k = 0; k < 3; ++k) {   // back-to-back stamps: the cost of one stamp interval
            stamp(ctx->stream(), -1, C_CAL, -1, -1, 0);
        }
    }
    s.active = true;
    return true;
}

// before node i is executed (views and no-ops never get here); `other_stream` skips nodes of a concurrent region
static void node(ggml_backend_cuda_context * ctx, int i, bool other_stream) {
    state_t & s = st();
    if (!s.active) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    s.cur.n_exec++;
    s.cur_node = i;
    s.fa_kernel = -1;
    if (other_stream) {
        s.cur.concurrent++;
        return;
    }
    const graph_info & gi = s.gi;
    uint8_t cls = gi.cls[(size_t) i];
    if (cls == C_FA) {
        cls = C_FA_PREP;   // the FA launch stamps the main kernel itself
    }
    stamp(ctx->stream(), i, cls, gi.layer[(size_t) i], gi.chunk[(size_t) i], gi.kind[(size_t) i]);
}

// the node passed to node() was fused with the n following nodes into one launch
static void fused(int n) {
    state_t & s = st();
    if (s.active && !s.cur.ents.empty()) {
        s.cur.ents.back().fused = (int16_t) n;
    }
}

static void end(ggml_backend_cuda_context * ctx) {
    state_t & s = st();
    if (!s.active) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    stamp(ctx->stream(), -1, C_END, -1, -1, 0);
    (void) hipStreamQuery(ctx->stream());
    s.active = false;
    s.cur_node = -1;
}

// after a stream synchronize: every stamp so far has landed
static void on_sync() {
    if (mode() == 0) {
        return;
    }
    state_t & s = st();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.init || s.failed) {
        return;
    }
    if (s.cur_open && !s.active && s.cur.key != ggml_account_current()) {
        close_cur();
    }
    collect(true);
}

} // namespace ggml_cuda_pstamp

void ggml_cuda_pstamp_fa_kernel(int kernel) {
    ggml_cuda_pstamp::state_t & s = ggml_cuda_pstamp::st();
    if (s.active) {
        s.fa_kernel = kernel;
    }
}

void ggml_cuda_pstamp_fa(cudaStream_t stream, const ggml_tensor * dst, bool sparse, int ncols1, int ncols2, int nbatch_fa,
                         int nwarps, int gx, int gy, int gz, bool stream_k, int64_t n_kv) {
    using namespace ggml_cuda_pstamp;
    state_t & s = st();
    if (!s.active) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    const int i = s.cur_node;
    if (i < 0 || i >= s.gi.n || s.gi.g->nodes[i] != dst) {
        return;
    }
    fa_t f;
    auto it = s.gi.fa.find(i);
    std::vector<int> g = it != s.gi.fa.end() ? it->second : std::vector<int>(10, -1);
    // layer, chunk, i0, n, n_kv (the launch's), n_kv_max, mask_ne0, kernel, sparse, ncols1, ncols2, nbatch_fa, nwarps,
    // gx, gy, gz, stream_k, k_cell0, nj, n_comp
    const int v[20] = { g[0], g[1], g[2], g[3], (int) n_kv, g[5], g[6], s.fa_kernel, sparse ? 1 : 0, ncols1, ncols2,
                        nbatch_fa, nwarps, gx, gy, gz, stream_k ? 1 : 0, g[7], g[8], g[9] };
    memcpy(f.v, v, sizeof(v));
    // the node stamp opened fa_prep; the main kernel starts here
    if (!s.cur.ents.empty() && s.cur.ents.back().node == i && s.cur.ents.back().cls == C_FA_PREP) {
        f.ent_prep = (int) s.cur.ents.size() - 1;
        const ent_t prep = s.cur.ents.back();
        stamp(stream, i, C_FA, prep.layer, prep.chunk, prep.kind);
        f.ent_kern = (int) s.cur.ents.size() - 1;
    }
    s.cur.fa.push_back(f);
}


void ggml_cuda_pstamp_fa_overlap(ggml_backend_cuda_context & ctx, cudaStream_t stream, const ggml_tensor * dst) {
    using namespace ggml_cuda_pstamp;
    state_t & s = st();
    if (!s.active || !overlap_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    const int i = s.cur_node;
    const ggml_tensor * mask = dst->src[3];
    if (i < 0 || i >= s.gi.n || s.gi.g->nodes[i] != dst || s.cur.fa.empty() || mask == nullptr ||
            mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || dst->src[0]->ne[1] < 16) {
        return;
    }
    if (s.pos + ov_n >= n_slots) {
        s.cur.overflow = true;
        return;
    }
    const int rows = (int) dst->src[0]->ne[1];
    auto it = s.gi.fa.find(i);
    const int n_comp = it != s.gi.fa.end() ? it->second[9] : 0;
    ggml_cuda_pool_alloc<unsigned long long> acc(ctx.pool(), 8 + (44 + 1)/2);
    unsigned int * hist = (unsigned int *) (acc.ptr + 8);
    CUDA_CHECK(cudaMemsetAsync(acc.ptr, 0, sizeof(unsigned long long)*(8 + 22), stream));
    overlap_kernel<<<(rows + 15)/16, 256, 0, stream>>>((const half *) mask->data, mask->nb[1]/sizeof(half), (int) mask->ne[0],
        rows, n_comp, acc.ptr, hist);
    const uint32_t first = s.pos;
    for (int k = 0; k < ov_n; ++k) {
        (void) take_slot();
    }
    overlap_publish<<<1, 64, 0, stream>>>(acc.ptr, hist, s.dev + first);
    s.cur.fa.back().ov_slot = first;
}

void ggml_cuda_pstamp_moe(cudaStream_t stream, const int * n_chunks_dev, int64_t n_tokens, int n_used, int64_t n_rows,
                          int64_t chunk_rows, int mt_v, int mt_r, int path, bool remote) {
    using namespace ggml_cuda_pstamp;
    state_t & s = st();
    if (!s.active) {
        return;
    }
    std::lock_guard<std::mutex> lock(s.mutex);
    const int i = s.cur_node;
    moe_t m;
    const int v[9] = { i >= 0 && i < s.gi.n ? (int) s.gi.layer[(size_t) i] : -1, (int) n_tokens, n_used, (int) n_rows,
                       (int) chunk_rows, mt_v, mt_r, path, remote ? 1 : 0 };
    memcpy(m.v, v, sizeof(v));
    const int64_t slot = take_slot();
    if (slot < 0) {
        return;
    }
    m.val_slot = (uint32_t) slot;
    capture_kernel<<<1, 1, 0, stream>>>(n_chunks_dev, s.dev + slot);
    if (!s.cur.ents.empty() && s.cur.ents.back().node == i) {
        m.ent = (int) s.cur.ents.size() - 1;
    }
    s.cur.moe.push_back(m);
}

#else // GGML_USE_HIP

namespace ggml_cuda_pstamp {
static bool begin(ggml_backend_cuda_context *, const ggml_cgraph *) { return false; }
static void node(ggml_backend_cuda_context *, int, bool) {}
static void fused(int) {}
static void end(ggml_backend_cuda_context *) {}
static void on_sync() {}
} // namespace ggml_cuda_pstamp

void ggml_cuda_pstamp_fa_kernel(int) {}
void ggml_cuda_pstamp_fa_overlap(ggml_backend_cuda_context &, cudaStream_t, const ggml_tensor *) {}
void ggml_cuda_pstamp_fa(cudaStream_t, const ggml_tensor *, bool, int, int, int, int, int, int, int, bool, int64_t) {}
void ggml_cuda_pstamp_moe(cudaStream_t, const int *, int64_t, int, int64_t, int64_t, int, int, int, bool) {}

#endif // GGML_USE_HIP

#endif // GGML_CUDA_PSTAMP_IMPL
