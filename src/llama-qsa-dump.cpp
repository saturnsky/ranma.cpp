#include "llama-qsa-dump.h"

#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

static const char * const LLAMA_QSA_DUMP_PREFIX = "indexer_top_k_dump-";

//
// minimal SHA-1, enough to fingerprint one selection set
//

namespace {

struct qsa_sha1 {
    uint32_t h[5] = { 0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u };
    uint8_t  block[64];
    size_t   n_block = 0;
    uint64_t n_total = 0;

    static uint32_t rol(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }

    void compress() {
        uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t) block[4*i + 0] << 24) | ((uint32_t) block[4*i + 1] << 16) |
                   ((uint32_t) block[4*i + 2] <<  8) | ((uint32_t) block[4*i + 3]);
        }
        for (int i = 16; i < 80; ++i) {
            w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f;
            uint32_t k;
            if (i < 20) {
                f = (b & c) | (~b & d);          k = 0x5A827999u;
            } else if (i < 40) {
                f = b ^ c ^ d;                   k = 0x6ED9EBA1u;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu;
            } else {
                f = b ^ c ^ d;                   k = 0xCA62C1D6u;
            }
            const uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }

    void update(const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *) data;
        n_total += len;
        while (len > 0) {
            const size_t n = std::min(len, sizeof(block) - n_block);
            memcpy(block + n_block, p, n);
            n_block += n;
            p       += n;
            len     -= n;
            if (n_block == sizeof(block)) {
                compress();
                n_block = 0;
            }
        }
    }

    std::string final_hex() {
        const uint64_t bits = n_total*8;
        const uint8_t  pad  = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (n_block != 56) {
            update(&zero, 1);
        }
        uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) {
            len_be[i] = (uint8_t) (bits >> (56 - 8*i));
        }
        // the length itself must not be counted, so write it straight into the block
        memcpy(block + n_block, len_be, 8);
        n_block = 64;
        compress();

        char out[41];
        for (int i = 0; i < 5; ++i) {
            snprintf(out + 8*i, 9, "%08x", h[i]);
        }
        return std::string(out, 40);
    }
};

struct qsa_dump_state {
    std::mutex mutex;

    FILE * file = nullptr;

    // one graph evaluation is one step; layers come out in increasing order, so a
    // layer id that does not grow means a new graph
    int64_t step   = -1;
    int     last_il = 1 << 30;

    std::map<int, int64_t> n_kv;

    std::vector<int32_t> cell_blk;
    int64_t cell_blk_n_kv     = 0;
    int64_t cell_blk_n_stream = 0;

    std::vector<int32_t> buf;
    std::vector<int32_t> blocks;
    std::vector<llama_qsa_dump_block> valid_blocks;
};

qsa_dump_state & qsa_dump_get() {
    static qsa_dump_state state;
    return state;
}

} // namespace

bool llama_qsa_dump_enabled() {
    static const bool enabled = []() {
        const char * path = getenv("LLAMA_QSA_DUMP");
        if (path == nullptr || path[0] == '\0') {
            return false;
        }

        FILE * f = fopen(path, "wb");
        if (f == nullptr) {
            LLAMA_LOG_ERROR("%s: cannot open LLAMA_QSA_DUMP file '%s'\n", __func__, path);
            return false;
        }

        qsa_dump_get().file = f;

        LLAMA_LOG_INFO("%s: QSA top-k dump enabled, writing '%s'\n", __func__, path);

        return true;
    }();

    return enabled;
}

void llama_qsa_dump_set_n_kv(int il, int64_t n_kv) {
    auto & state = qsa_dump_get();

    std::lock_guard<std::mutex> lock(state.mutex);
    state.n_kv[il] = n_kv;
}

void llama_qsa_dump_set_cell_blk(const int32_t * cell_blk, int64_t n_kv, int64_t n_stream) {
    auto & state = qsa_dump_get();

    std::lock_guard<std::mutex> lock(state.mutex);

    state.cell_blk.assign(cell_blk, cell_blk + n_kv*n_stream);
    state.cell_blk_n_kv     = n_kv;
    state.cell_blk_n_stream = n_stream;
}

void llama_qsa_dump_set_blocks(const std::vector<llama_qsa_dump_block> & blocks) {
    auto & state = qsa_dump_get();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.valid_blocks = blocks;
    std::sort(state.valid_blocks.begin(), state.valid_blocks.end(),
            [](const llama_qsa_dump_block & a, const llama_qsa_dump_block & b) {
                return a.seq != b.seq ? a.seq < b.seq : a.start < b.start;
            });
}

bool llama_qsa_dump_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const bool is_dbg = strncmp(t->name, "indexer_dbg_", 12) == 0;

    const bool is_trans = strncmp(t->name, "indexer_ktrans_dump-", 20) == 0;

    if (!is_trans && !is_dbg && strncmp(t->name, LLAMA_QSA_DUMP_PREFIX, strlen(LLAMA_QSA_DUMP_PREFIX)) != 0) {
        return false;
    }

    if (ask) {
        return true;
    }

    auto & state = qsa_dump_get();

    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.file == nullptr) {
        return true;
    }

    if (is_trans) {
        const int il = atoi(t->name + 20);
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        qsa_sha1 meta, values;
        const size_t row_bytes = t->ne[0]*sizeof(float);
        for (const auto & block : state.valid_blocks) {
            GGML_ASSERT((block.row + 1)*row_bytes <= raw.size());
            const int32_t key[] = {block.seq, block.start, block.pos[0], block.pos[1], block.pos[2], block.pos[3]};
            meta.update(key, sizeof(key));
            values.update(raw.data() + block.row*row_bytes, row_bytes);
        }
        const int64_t step = state.step + (il <= state.last_il ? 1 : 0);
        fprintf(state.file, "# ktrans %" PRId64 " %d %zu %s %s\n", step, il, state.valid_blocks.size(),
                meta.final_hex().c_str(), values.final_hex().c_str());
        fflush(state.file);
        return true;
    }

    if (is_dbg) {
        const char * dash = strrchr(t->name, '-');
        const int il_dbg = dash ? atoi(dash + 1) : -1;
        const size_t nb = ggml_nbytes(t);
        std::vector<char> raw(nb);
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
        qsa_sha1 dbg;
        dbg.update(raw.data(), nb);
        fprintf(state.file, "# dbg %" PRId64 " %d %s %zu %s\n", state.step, il_dbg, t->name, nb, dbg.final_hex().c_str());
        fflush(state.file);
        return true;
    }

    const int il = atoi(t->name + strlen(LLAMA_QSA_DUMP_PREFIX));

    if (il <= state.last_il) {
        state.step++;
    }
    state.last_il = il;

    GGML_ASSERT(t->type == GGML_TYPE_I32);

    const int64_t width = t->ne[0];
    const int64_t n_all = ggml_nelements(t);

    state.buf.resize(n_all);
    ggml_backend_tensor_get(t, state.buf.data(), 0, n_all*sizeof(int32_t));

    const int64_t n_rows = n_all/width;

    // rows are laid out stream after stream, matching top_k [width, n_tps, 1, n_stream]
    const int64_t n_tps = state.cell_blk_n_stream > 0 ? n_rows/state.cell_blk_n_stream : n_rows;

    qsa_sha1 sha;
    qsa_sha1 sha_blk;

    for (int64_t row = 0; row < n_rows; ++row) {
        int32_t * p = state.buf.data() + row*width;
        std::sort(p, p + width);
        sha.update(p, width*sizeof(int32_t));

        if (state.cell_blk.empty()) {
            continue;
        }

        const int64_t s = n_tps > 0 ? row/n_tps : 0;
        const int32_t * map = state.cell_blk.data() + s*state.cell_blk_n_kv;

        state.blocks.clear();
        for (int64_t i = 0; i < width; ++i) {
            if (p[i] >= 0 && p[i] < state.cell_blk_n_kv) {
                state.blocks.push_back(map[p[i]]);
            }
        }
        std::sort(state.blocks.begin(), state.blocks.end());
        state.blocks.erase(std::unique(state.blocks.begin(), state.blocks.end()), state.blocks.end());
        sha_blk.update(state.blocks.data(), state.blocks.size()*sizeof(int32_t));
    }

    const auto it = state.n_kv.find(il);

    fprintf(state.file, "%" PRId64 " %d %" PRId64 " %" PRId64 " %s %s\n",
            state.step, il, it == state.n_kv.end() ? -1 : it->second, n_all,
            sha.final_hex().c_str(), sha_blk.final_hex().c_str());
    fflush(state.file);

    return true;
}
