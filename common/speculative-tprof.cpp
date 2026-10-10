#include "speculative-tprof.h"

#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#define TPF_INF(fmt, ...) LOG_INF("trigram profile: " fmt, __VA_ARGS__)
#define TPF_WRN(fmt, ...) LOG_WRN("trigram profile: " fmt, __VA_ARGS__)

static const char     TPROF_MAGIC[8] = { 'R', 'T', 'P', 'R', 'O', 'F', '1', '\n' };
static const uint32_t TPROF_VERSION  = 1;

//
// hashing
//

static uint64_t tp_mix(uint64_t x) { // splitmix64 finaliser
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint64_t tp_fnv(const void * data, size_t n, uint64_t h = 0xcbf29ce484222325ULL) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// gear table of the content-defined chunking
struct tp_gear {
    uint64_t g[256];
    tp_gear() {
        uint64_t s = 0x6a09e667f3bcc909ULL;
        for (auto & x : g) {
            s += 0x9e3779b97f4a7c15ULL;
            x = tp_mix(s);
        }
    }
};
static const tp_gear TP_GEAR;

static bool tp_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

//
// open addressing tables (linear probing, capacity a power of 2, deletion by backward shift)
//

struct tp_node { // key (a, b, c); c < 0: empty slot
    int32_t  a, b, c;
    float    n[3];
    uint32_t t;
};

struct tp_head { // key (a, b); a < 0: empty slot
    int32_t  a, b;
    float    N[3];
    uint32_t t;
};

struct tp_chunk { // key id; id 0: empty slot
    uint32_t lo, hi;
    uint32_t seen;
    uint64_t id() const { return ((uint64_t) hi << 32) | lo; }
};

template <typename E, typename K>
struct tp_table {
    std::vector<E> slots;
    size_t         n     = 0;
    size_t         cap_max = 0; // slots at most

    using key_type = K;

    static constexpr double LOAD = 0.7;

    size_t mask() const { return slots.size() - 1; }
    size_t limit() const { return (size_t) (cap_max * LOAD); } // entries at most
    double mib() const { return (double) slots.size() * sizeof(E) / (1024.0 * 1024.0); }

    void init(size_t cap0, size_t cmax) {
        cap_max = cmax;
        slots.assign(std::min(cap0, cmax), K::empty());
        n = 0;
    }

    // the slot of key, or the empty slot where it goes
    size_t find(const E & key) const {
        size_t i = K::hash(key) & mask();
        while (!K::is_empty(slots[i]) && !K::eq(slots[i], key)) {
            i = (i + 1) & mask();
        }
        return i;
    }

    const E * get(const E & key) const {
        const size_t i = find(key);
        return K::is_empty(slots[i]) ? nullptr : &slots[i];
    }

    // the entry of key, inserted (zero) when missing; nullptr: full (the caller compacts first)
    E * put(const E & key) {
        size_t i = find(key);
        if (!K::is_empty(slots[i])) {
            return &slots[i];
        }
        if ((double) (n + 1) > slots.size() * LOAD) {
            if (slots.size() >= cap_max) {
                if ((double) (n + 1) > slots.size() * 0.9) {
                    return nullptr;
                }
            } else {
                grow();
                i = find(key);
            }
        }
        slots[i] = K::fresh(key);
        n++;
        return &slots[i];
    }

    void grow() {
        std::vector<E> old;
        old.swap(slots);
        slots.assign(std::min(old.size() * 2, cap_max), K::empty());
        for (const auto & e : old) {
            if (!K::is_empty(e)) {
                slots[find(e)] = e;
            }
        }
    }

    // removes every entry for which drop(e) holds (backward shift: no tombstones)
    template <typename F>
    size_t remove_if(const F & drop) {
        size_t removed = 0;
        const size_t cap = slots.size();
        // start after an empty slot, so that every cluster is seen whole
        size_t start = 0;
        while (start < cap && !K::is_empty(slots[start])) {
            start++;
        }
        if (start == cap) {
            start = 0; // full table (not reached: the load is at most 0.9)
        }
        for (size_t k = 1; k <= cap; ++k) {
            const size_t i = (start + k) & mask();
            if (K::is_empty(slots[i]) || !drop(slots[i])) {
                continue;
            }
            // delete i and shift the following entries of the cluster back
            size_t hole = i;
            size_t j    = i;
            for (;;) {
                j = (j + 1) & mask();
                if (K::is_empty(slots[j])) {
                    break;
                }
                const size_t home = K::hash(slots[j]) & mask();
                // j can move to hole when its home is not in (hole, j]
                const bool in = hole <= j ? (home > hole && home <= j) : (home > hole || home <= j);
                if (!in) {
                    slots[hole] = slots[j];
                    hole = j;
                }
            }
            slots[hole] = K::empty();
            n--;
            removed++;
            // the entry shifted into i has not been looked at yet
            if (!K::is_empty(slots[i])) {
                k--;
            }
        }
        return removed;
    }
};

struct tp_node_k {
    static tp_node  empty() { tp_node e{}; e.c = -1; return e; }
    static bool     is_empty(const tp_node & e) { return e.c < 0; }
    static bool     eq(const tp_node & x, const tp_node & y) { return x.a == y.a && x.b == y.b && x.c == y.c; }
    static uint64_t hash(const tp_node & e) {
        return tp_mix(((uint64_t) (uint32_t) e.a << 32 | (uint32_t) e.b) ^ tp_mix((uint64_t) (uint32_t) e.c + 0x51ed27ULL));
    }
    static tp_node fresh(const tp_node & key) { tp_node e{}; e.a = key.a; e.b = key.b; e.c = key.c; return e; }
};

struct tp_head_k {
    static tp_head  empty() { tp_head e{}; e.a = -1; return e; }
    static bool     is_empty(const tp_head & e) { return e.a < 0; }
    static bool     eq(const tp_head & x, const tp_head & y) { return x.a == y.a && x.b == y.b; }
    static uint64_t hash(const tp_head & e) { return tp_mix((uint64_t) (uint32_t) e.a << 32 | (uint32_t) e.b); }
    static tp_head fresh(const tp_head & key) { tp_head e{}; e.a = key.a; e.b = key.b; return e; }
};

struct tp_chunk_k {
    static tp_chunk empty() { return tp_chunk{ 0, 0, 0 }; }
    static bool     is_empty(const tp_chunk & e) { return e.lo == 0 && e.hi == 0; }
    static bool     eq(const tp_chunk & x, const tp_chunk & y) { return x.lo == y.lo && x.hi == y.hi; }
    static uint64_t hash(const tp_chunk & e) { return tp_mix(e.id()); }
    static tp_chunk fresh(const tp_chunk & key) { return tp_chunk{ key.lo, key.hi, 0 }; }
};

static tp_chunk tp_chunk_key(uint64_t id) {
    return tp_chunk{ (uint32_t) id, (uint32_t) (id >> 32), 0 };
}

//
// the profile
//

struct tp_chunks {
    std::vector<uint64_t> id;      // per chunk
    std::vector<uint32_t> msg;     // per chunk: its message
    std::vector<uint32_t> b0, b1;  // per chunk: its bytes in norm
    std::vector<int32_t>  of_tok;  // per token: its chunk (-1: none)
    std::string           norm;    // the normalised bytes
};

struct common_spec_tprof_impl {
    common_spec_tprof_config cfg;
    common_spec_tprof_vocab  voc;

    bool        on        = false;
    bool        read_only = false;
    double      decay_k   = 0.0;  // ln 2 / half_life
    double      clock     = 0.0;  // generated tokens of the profile
    uint32_t    req       = 0;    // commits of the profile
    bool        dirty     = false;
    int32_t     req_since_save = 0;
    std::chrono::steady_clock::time_point t_save = std::chrono::steady_clock::now();

    tp_table<tp_node, tp_node_k>   nodes;
    tp_table<tp_head, tp_head_k>   heads;
    tp_table<tp_chunk, tp_chunk_k> chunks;
    std::vector<std::string>       ring;  // generated answers, normalised (the newest last)
    std::vector<std::pair<uint64_t, int32_t>> regen; // prompt hash -> answers so far (the newest last)
    std::vector<common_spec_tprof_coef> coefs;

    common_spec_tprof_stats st;

    // the save thread
    std::thread       writer;
    std::atomic<bool> writing{ false };

#ifdef _WIN32
    HANDLE lock = INVALID_HANDLE_VALUE;
#else
    int lock = -1;
#endif

    float decay(uint32_t t) const {
        const double dt = clock - (double) t;
        return dt <= 0.0 ? 1.0f : (float) std::exp(-decay_k * dt);
    }

    void sizes(double cap_mib) {
        // the chunk table is fixed, the context table has half the slots of the node table, which takes the largest
        // power of 2 that fits
        const double  total  = cap_mib * 1024.0 * 1024.0;
        const size_t  ccap   = 1u << 19;
        const double  rest   = total - (double) ccap * sizeof(tp_chunk);
        size_t        ncap   = 1u << 12;
        while ((double) (ncap * 2) * sizeof(tp_node) + (double) ncap * sizeof(tp_head) <= rest) {
            ncap *= 2;
        }
        nodes.init(4096, ncap);
        heads.init(2048, std::max<size_t>(ncap / 2, 2048));
        chunks.init(4096, ccap);
    }

    void stats_sizes() {
        st.nodes  = nodes.n;
        st.heads  = heads.n;
        st.chunks = chunks.n;
        st.mib    = nodes.mib() + heads.mib() + chunks.mib();
        st.clock  = clock;
    }

    // the normalised bytes of the tokens, cut into chunks: at a special token, at a message start, else at a gear
    // anchor (at least 16 bytes, at most 256). A token belongs to the chunk of its first byte; a token without bytes to
    // the next chunk of its message, else to the chunk before it. A chunk of fewer than 16 bytes has an id that also
    // depends on the chunk before it in its message, so that short common texts are not taken for each other.
    void cut(const int32_t * toks, size_t n, const std::vector<uint8_t> * msg_start, tp_chunks & out) const {
        out = {};
        out.of_tok.assign(n, -1);
        std::vector<size_t> pending;
        uint32_t msg   = 0;
        bool     open  = false; // a chunk is open
        uint64_t gh    = 0;
        uint64_t prev  = 0;     // id of the chunk before, in the same message and after the last special token
        auto close = [&]() {
            if (!open) {
                return;
            }
            const size_t c   = out.id.size() - 1;
            const size_t len = out.b1[c] - out.b0[c];
            uint64_t h = tp_fnv(out.norm.data() + out.b0[c], len);
            if (len < 16) {
                h = tp_mix(h ^ prev);
            }
            h = h == 0 ? 1 : h;
            out.id[c] = h;
            prev      = h;
            open      = false;
        };
        for (size_t i = 0; i < n; ++i) {
            const int32_t t = toks[i];
            const bool special = t >= 0 && (size_t) t < voc.special.size() && voc.special[t];
            const bool starts  = msg_start != nullptr && (*msg_start)[i];
            if (starts) {
                close();
                // tokens without bytes left at the end of the message: to its last chunk
                if (!pending.empty() && !out.id.empty() && out.msg.back() == msg) {
                    for (size_t j : pending) {
                        out.of_tok[j] = (int32_t) out.id.size() - 1;
                    }
                }
                pending.clear();
                msg++;
                prev = 0;
            }
            if (special) {
                close();
                prev = 0;
                pending.push_back(i);
                continue;
            }
            const std::string & pc = t >= 0 && (size_t) t < voc.piece.size() ? voc.piece[t] : std::string();
            bool first = true;
            for (unsigned char ch : pc) {
                if (tp_space(ch)) {
                    continue;
                }
                if (!open) {
                    out.id.push_back(0);
                    out.msg.push_back(msg);
                    out.b0.push_back((uint32_t) out.norm.size());
                    out.b1.push_back((uint32_t) out.norm.size());
                    open = true;
                    gh   = 0;
                }
                const int32_t c = (int32_t) out.id.size() - 1;
                if (first) {
                    out.of_tok[i] = c;
                    for (size_t j : pending) {
                        out.of_tok[j] = c;
                    }
                    pending.clear();
                    first = false;
                }
                out.norm.push_back((char) ch);
                out.b1.back() = (uint32_t) out.norm.size();
                gh = (gh << 1) + TP_GEAR.g[ch];
                const size_t len = out.b1.back() - out.b0.back();
                if ((len >= 16 && (gh >> 58) == 0) || len >= 256) {
                    close();
                }
            }
            if (first) {
                pending.push_back(i); // only white space
            }
        }
        close();
        if (!pending.empty() && !out.id.empty() && out.msg.back() == msg) {
            for (size_t j : pending) {
                out.of_tok[j] = (int32_t) out.id.size() - 1;
            }
        }
    }

    bool chunk_known(uint64_t id) const {
        return chunks.get(tp_chunk_key(id)) != nullptr;
    }

    void chunk_mark(uint64_t id) {
        tp_chunk * e = chunks.put(tp_chunk_key(id));
        if (e == nullptr) {
            evict_chunks();
            e = chunks.put(tp_chunk_key(id));
        }
        if (e != nullptr) {
            e->seen = req;
        }
    }

    // the chunk table is full: the 25 % seen longest ago go
    void evict_chunks() {
        std::vector<uint32_t> seen;
        seen.reserve(chunks.n);
        for (const auto & e : chunks.slots) {
            if (!tp_chunk_k::is_empty(e)) {
                seen.push_back(e.seen);
            }
        }
        if (seen.empty()) {
            return;
        }
        const size_t q = seen.size() / 4;
        std::nth_element(seen.begin(), seen.begin() + q, seen.end());
        const uint32_t thr = seen[q];
        chunks.remove_if([&](const tp_chunk & e) { return e.seen < thr; });
        if (chunks.n > chunks.limit() * 0.8) {
            chunks.remove_if([&](const tp_chunk & e) { return e.seen <= thr; });
        }
    }

    void add(int32_t a, int32_t b, int32_t c, int32_t s, float w) {
        tp_node key{};
        key.a = a; key.b = b; key.c = c;
        tp_node * e = nodes.put(key);
        if (e == nullptr) {
            compact(0.6);
            e = nodes.put(key);
            if (e == nullptr) {
                return;
            }
        }
        const uint32_t now = (uint32_t) clock;
        const float f = decay(e->t);
        for (float & x : e->n) {
            x *= f;
        }
        e->t = now;
        e->n[s] += w;

        tp_head hk{};
        hk.a = a; hk.b = b;
        tp_head * h = heads.put(hk);
        if (h == nullptr) {
            compact(0.6);
            h = heads.put(hk);
            if (h == nullptr) {
                return;
            }
        }
        const float fh = decay(h->t);
        for (float & x : h->N) {
            x *= fh;
        }
        h->t = now;
        h->N[s] += w;
    }

    // drops the dead trigrams (decayed weight below eps) and, while a table holds more than frac of its limit, the
    // lightest ones
    void compact(double frac) {
        const auto t0 = std::chrono::steady_clock::now();
        auto wn = [&](const tp_node & e) { return (e.n[0] + e.n[1] + e.n[2]) * decay(e.t); };
        auto wh = [&](const tp_head & e) { return (e.N[0] + e.N[1] + e.N[2]) * decay(e.t); };
        const float eps = (float) cfg.eps;
        nodes.remove_if([&](const tp_node & e) { return wn(e) < eps; });
        heads.remove_if([&](const tp_head & e) { return wh(e) < eps; });
        auto shrink = [&](auto & tab, auto w) {
            const size_t target = (size_t) (tab.limit() * frac);
            if (tab.n <= target) {
                return;
            }
            std::vector<float> ws;
            ws.reserve(tab.n);
            for (const auto & e : tab.slots) {
                if (!std::decay_t<decltype(tab)>::key_type::is_empty(e)) {
                    ws.push_back(w(e));
                }
            }
            const size_t q = tab.n - target;
            std::nth_element(ws.begin(), ws.begin() + q, ws.end());
            const float thr = ws[q];
            tab.remove_if([&](const auto & e) { return w(e) < thr; });
            // ties at the threshold: as many as are still over the target
            if (tab.n > target) {
                size_t extra = tab.n - target;
                tab.remove_if([&](const auto & e) {
                    if (extra > 0 && w(e) <= thr) {
                        extra--;
                        return true;
                    }
                    return false;
                });
            }
        };
        shrink(nodes, wn);
        shrink(heads, wh);
        st.compact_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    //
    // file
    //

    static void put_u32(std::string & o, uint32_t v) { o.append((const char *) &v, 4); }
    static void put_u64(std::string & o, uint64_t v) { o.append((const char *) &v, 8); }
    static void put_f64(std::string & o, double v) { o.append((const char *) &v, 8); }
    static void put_str(std::string & o, const std::string & s) { put_u32(o, (uint32_t) s.size()); o += s; }

    std::string snapshot() const {
        std::string o;
        o.reserve(64 + nodes.n * sizeof(tp_node) + heads.n * sizeof(tp_head) + chunks.n * 12);
        o.append(TPROF_MAGIC, 8);
        put_u32(o, TPROF_VERSION);
        put_str(o, cfg.tok_key);
        put_f64(o, clock);
        put_f64(o, cfg.half_life);
        put_u32(o, req);
        put_u64(o, nodes.n);
        for (const auto & e : nodes.slots) {
            if (!tp_node_k::is_empty(e)) {
                o.append((const char *) &e, sizeof(e));
            }
        }
        put_u64(o, heads.n);
        for (const auto & e : heads.slots) {
            if (!tp_head_k::is_empty(e)) {
                o.append((const char *) &e, sizeof(e));
            }
        }
        put_u64(o, chunks.n);
        for (const auto & e : chunks.slots) {
            if (!tp_chunk_k::is_empty(e)) {
                put_u64(o, e.id());
                put_u32(o, e.seen);
            }
        }
        put_u32(o, (uint32_t) ring.size());
        for (const auto & r : ring) {
            put_str(o, r);
        }
        put_u32(o, (uint32_t) coefs.size());
        for (const auto & c : coefs) {
            put_str(o, c.key);
            put_u32(o, (uint32_t) c.b.size());
            for (double v : c.b) {
                put_f64(o, v);
            }
            for (double v : c.G) {
                put_f64(o, v);
            }
            put_u64(o, (uint64_t) c.updates);
        }
        put_u64(o, tp_fnv(o.data(), o.size()));
        return o;
    }

    struct reader {
        const std::string & s;
        size_t pos = 0;
        bool   ok  = true;
        bool take(void * dst, size_t n) {
            if (!ok || pos + n > s.size()) {
                ok = false;
                return false;
            }
            memcpy(dst, s.data() + pos, n);
            pos += n;
            return true;
        }
        uint32_t u32() { uint32_t v = 0; take(&v, 4); return v; }
        uint64_t u64() { uint64_t v = 0; take(&v, 8); return v; }
        double   f64() { double v = 0; take(&v, 8); return v; }
        std::string str() {
            const uint32_t n = u32();
            if (!ok || pos + n > s.size()) {
                ok = false;
                return {};
            }
            std::string r = s.substr(pos, n);
            pos += n;
            return r;
        }
    };

    // 1: loaded, 0: no file, -1: not usable (why)
    int load_file(const std::string & path, std::string & why) {
        std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
        if (!f) {
            return 0;
        }
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (s.size() < 8 + 4 + 8 || memcmp(s.data(), TPROF_MAGIC, 8) != 0) {
            why = "unknown format";
            return -1;
        }
        uint64_t sum = 0;
        memcpy(&sum, s.data() + s.size() - 8, 8);
        if (sum != tp_fnv(s.data(), s.size() - 8)) {
            why = "damaged (checksum)";
            return -1;
        }
        reader r{ s };
        r.pos = 8;
        const uint32_t ver = r.u32();
        if (ver != TPROF_VERSION) {
            why = "unknown version " + std::to_string(ver);
            return -1;
        }
        const std::string key = r.str();
        if (key != cfg.tok_key) {
            why = "another tokenizer (profile " + key + ", model " + cfg.tok_key + ")";
            return -2;
        }
        clock = r.f64();
        r.f64(); // the half-life it was written with (this process uses its own)
        req = r.u32();
        const uint64_t nn = r.u64();
        for (uint64_t i = 0; i < nn && r.ok; ++i) {
            tp_node e;
            if (!r.take(&e, sizeof(e)) || e.c < 0) {
                r.ok = false;
                break;
            }
            tp_node * d = nodes.put(e);
            if (d == nullptr) {
                compact(0.75);
                d = nodes.put(e);
            }
            if (d != nullptr) {
                *d = e;
            }
        }
        const uint64_t nh = r.u64();
        for (uint64_t i = 0; i < nh && r.ok; ++i) {
            tp_head e;
            if (!r.take(&e, sizeof(e)) || e.a < 0) {
                r.ok = false;
                break;
            }
            tp_head * d = heads.put(e);
            if (d == nullptr) {
                compact(0.75);
                d = heads.put(e);
            }
            if (d != nullptr) {
                *d = e;
            }
        }
        const uint64_t nc = r.u64();
        for (uint64_t i = 0; i < nc && r.ok; ++i) {
            const uint64_t id   = r.u64();
            const uint32_t seen = r.u32();
            if (id == 0) {
                r.ok = false;
                break;
            }
            tp_chunk * d = chunks.put(tp_chunk_key(id));
            if (d == nullptr) {
                evict_chunks();
                d = chunks.put(tp_chunk_key(id));
            }
            if (d != nullptr) {
                d->seen = seen;
            }
        }
        const uint32_t nr = r.u32();
        for (uint32_t i = 0; i < nr && r.ok; ++i) {
            ring.push_back(r.str());
        }
        const uint32_t ncf = r.u32();
        for (uint32_t i = 0; i < ncf && r.ok; ++i) {
            common_spec_tprof_coef c;
            c.key = r.str();
            const uint32_t d = r.u32();
            if (d > 64) {
                r.ok = false;
                break;
            }
            c.b.resize(d);
            c.G.resize(d);
            for (auto & v : c.b) {
                v = r.f64();
            }
            for (auto & v : c.G) {
                v = r.f64();
            }
            c.updates = (int64_t) r.u64();
            coefs.push_back(std::move(c));
        }
        if (!r.ok || r.pos != s.size() - 8) {
            why = "damaged (layout)";
            return -1;
        }
        return 1;
    }

    void reset_tables() {
        sizes(cfg.cap_mib);
        ring.clear();
        coefs.clear();
        clock = 0.0;
        req   = 0;
    }

    bool take_lock() {
        const std::string lp = cfg.path + ".lock";
#ifdef _WIN32
        lock = CreateFileW(std::filesystem::u8path(lp).wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                           OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        return lock != INVALID_HANDLE_VALUE;
#else
        lock = ::open(lp.c_str(), O_RDWR | O_CREAT, 0644);
        if (lock < 0) {
            return false;
        }
        if (flock(lock, LOCK_EX | LOCK_NB) != 0) {
            ::close(lock);
            lock = -1;
            return false;
        }
        return true;
#endif
    }

    void drop_lock() {
#ifdef _WIN32
        if (lock != INVALID_HANDLE_VALUE) {
            CloseHandle(lock);
            lock = INVALID_HANDLE_VALUE;
        }
#else
        if (lock >= 0) {
            flock(lock, LOCK_UN);
            ::close(lock);
            unlink((cfg.path + ".lock").c_str());
            lock = -1;
        }
#endif
    }

    static bool write_file(const std::string & path, const std::string & data) {
        namespace fs = std::filesystem;
        const fs::path p   = fs::u8path(path);
        const fs::path tmp = fs::u8path(path + ".tmp");
        const fs::path bak = fs::u8path(path + ".bak");
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) {
                return false;
            }
            f.write(data.data(), (std::streamsize) data.size());
            f.flush();
            if (!f) {
                return false;
            }
        }
        std::error_code ec;
        if (fs::exists(p, ec)) {
            fs::remove(bak, ec);
            ec.clear();
            fs::rename(p, bak, ec);
            if (ec) {
                return false;
            }
        }
        fs::rename(tmp, p, ec);
        return !ec;
    }

    void save_now(bool sync) {
        if (!on || read_only) {
            return;
        }
        if (writer.joinable()) {
            if (!sync && writing.load()) {
                return; // the last save is still being written: the next one takes this state too
            }
            writer.join();
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string data = snapshot();
        st.save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        dirty          = false;
        req_since_save = 0;
        t_save         = std::chrono::steady_clock::now();
        const std::string path = cfg.path;
        if (sync) {
            if (!write_file(path, data)) {
                TPF_WRN("cannot write %s\n", path.c_str());
            }
            return;
        }
        writing = true;
        writer = std::thread([this, path, data = std::move(data)]() {
            if (!write_file(path, data)) {
                TPF_WRN("cannot write %s\n", path.c_str());
            }
            writing = false;
        });
    }
};

common_spec_tprof::common_spec_tprof(const common_spec_tprof_config & cfg, common_spec_tprof_vocab vocab)
    : impl(std::make_unique<common_spec_tprof_impl>()) {
    impl->cfg     = cfg;
    impl->cfg.half_life = std::max(1.0, cfg.half_life);
    impl->cfg.sat = std::max<int32_t>(1, cfg.sat);
    impl->cfg.rho = std::clamp(cfg.rho, 0.0, 1.0);
    impl->voc     = std::move(vocab);
    impl->decay_k = std::log(2.0) / impl->cfg.half_life;
    impl->reset_tables();
}

common_spec_tprof::~common_spec_tprof() {
    if (impl->writer.joinable()) {
        impl->writer.join();
    }
    if (impl->on && !impl->read_only && impl->dirty) {
        impl->save_now(true);
        TPF_INF("saved %s (%zu trigrams, %zu contexts, %zu chunks)\n", impl->cfg.path.c_str(), impl->nodes.n,
                impl->heads.n, impl->chunks.n);
    }
    impl->drop_lock();
}

bool common_spec_tprof::open(std::string * why) {
    auto & m = *impl;
    std::string w;
    int r = m.load_file(m.cfg.path, w);
    if (r == -1) {
        // a damaged main file: the previous one
        m.reset_tables();
        std::string w2;
        const int r2 = m.load_file(m.cfg.path + ".bak", w2);
        if (r2 == 1) {
            TPF_WRN("%s %s, the previous save %s.bak is used\n", m.cfg.path.c_str(), w.c_str(), m.cfg.path.c_str());
            r = 1;
        } else if (r2 == -2) {
            w = w2;
            r = -2;
        }
    } else if (r == 0) {
        // no main file (a save stopped between its renames): the previous one if there is one
        std::string w2;
        const int r2 = m.load_file(m.cfg.path + ".bak", w2);
        if (r2 == 1 || r2 == -2) {
            r = r2;
            w = w2;
        } else if (r2 == -1) {
            m.reset_tables();
        }
    }
    if (r < 0) {
        m.reset_tables();
        m.on = false;
        if (why) {
            *why = w;
        }
        return false;
    }
    m.on = true;
    if (!m.take_lock()) {
        m.read_only = true;
        TPF_WRN("%s.lock is held by another process: the profile is read-only (no saves)\n", m.cfg.path.c_str());
    }
    m.stats_sizes();
    TPF_INF("%s %s: %zu trigrams, %zu contexts, %zu chunks, %zu answers, clock %.0f generated tokens, half-life %.0f, "
            "tables %.1f MiB (at most %.1f MiB, %zu trigrams)\n", m.cfg.path.c_str(), r == 1 ? "loaded" : "new",
            m.nodes.n, m.heads.n, m.chunks.n, m.ring.size(), m.clock, m.cfg.half_life, m.st.mib,
            (double) (m.nodes.cap_max * sizeof(tp_node) + m.heads.cap_max * sizeof(tp_head) +
                      m.chunks.cap_max * sizeof(tp_chunk)) / (1024.0 * 1024.0), m.nodes.limit());
    return true;
}

bool common_spec_tprof::on() const { return impl->on; }
bool common_spec_tprof::read_only() const { return impl->read_only; }

void common_spec_tprof::lookup(int32_t a, int32_t b, const int32_t * toks, int32_t n, double * cnt, double * N) const {
    const auto & m = *impl;
    for (int32_t i = 0; i < N_SRC * n; ++i) {
        cnt[i] = 0.0;
    }
    N[0] = N[1] = N[2] = 0.0;
    if (!m.on) {
        return;
    }
    tp_head hk{};
    hk.a = a; hk.b = b;
    const tp_head * h = m.heads.get(hk);
    if (h == nullptr) {
        return;
    }
    const float fh = m.decay(h->t);
    for (int32_t s = 0; s < N_SRC; ++s) {
        N[s] = h->N[s] * fh;
    }
    tp_node key{};
    key.a = a; key.b = b;
    for (int32_t i = 0; i < n; ++i) {
        key.c = toks[i];
        if (key.c < 0) {
            continue;
        }
        const tp_node * e = m.nodes.get(key);
        if (e == nullptr) {
            continue;
        }
        const float f = m.decay(e->t);
        for (int32_t s = 0; s < N_SRC; ++s) {
            cnt[N_SRC * i + s] = std::min<double>(e->n[s] * f, N[s]); // a context dropped in part: n <= N
        }
    }
}

void common_spec_tprof::commit(const int32_t * prompt, size_t n_prompt, const int32_t * gen, size_t n_gen,
                               const common_spec_roles & roles) {
    auto & m = *impl;
    if (!m.on) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    common_spec_tprof_stats & st = m.st;
    st = common_spec_tprof_stats();
    m.req++;
    st.req = m.req;

    // the messages of the prompt: a control token starts one (the markers start with one)
    const std::vector<uint8_t> & ctrl = !roles.ctrl.empty() ? roles.ctrl : m.voc.special;
    std::vector<uint8_t> msg_start(n_prompt, 0);
    for (size_t i = 0; i < n_prompt; ++i) {
        const int32_t t = prompt[i];
        msg_start[i] = t >= 0 && (size_t) t < ctrl.size() && ctrl[t];
    }
    const std::vector<uint8_t> role = roles.of(prompt, n_prompt, SIZE_MAX);

    tp_chunks pc;
    m.cut(prompt, n_prompt, &msg_start, pc);

    // the status of every chunk: known, X, U, or known by absorption (an edge of a generated answer)
    enum : uint8_t { KNOWN = 0, NEW_X = 1, NEW_U = 2, ABSORBED = 3 };
    const size_t nc = pc.id.size();
    std::vector<uint8_t> status(nc, KNOWN);
    std::vector<uint8_t> msg_assist; // per message: an assistant message
    {
        const uint32_t n_msg = nc > 0 ? pc.msg.back() + 1 : 0;
        msg_assist.assign(n_msg, 0);
        // the role of a message: that of its last token with a chunk
        for (size_t i = 0; i < n_prompt; ++i) {
            const int32_t c = pc.of_tok[i];
            if (c >= 0) {
                msg_assist[pc.msg[c]] = role[i] == common_spec_roles::ASSIST;
            }
        }
        std::vector<int32_t> kn(n_msg, 0), un(n_msg, 0);
        for (size_t c = 0; c < nc; ++c) {
            const bool k = m.chunk_known(pc.id[c]);
            status[c] = k ? KNOWN : NEW_U;
            (k ? kn : un)[pc.msg[c]]++;
        }
        for (size_t c = 0; c < nc; ++c) {
            if (status[c] == KNOWN || !msg_assist[pc.msg[c]]) {
                continue;
            }
            if (kn[pc.msg[c]] == 0) {
                status[c] = NEW_X; // an assistant message this server did not write
                continue;
            }
            // a partly known assistant message: an edge that a template, a front end or a tokenizer changed is a part
            // of a recent answer; the rest is an edit (U)
            const std::string bytes = pc.norm.substr(pc.b0[c], pc.b1[c] - pc.b0[c]);
            for (const auto & r : m.ring) {
                if (r.find(bytes) != std::string::npos) {
                    status[c] = ABSORBED;
                    break;
                }
            }
        }
    }

    // the trigrams to count, saturated per request and source
    std::unordered_map<uint64_t, uint32_t> occ; // key of (a, b, c, s) -> occurrences
    std::vector<std::pair<std::array<int32_t, 4>, uint64_t>> keys;
    auto event = [&](int32_t a, int32_t b, int32_t c, int32_t s) {
        const uint64_t k = tp_mix(tp_mix(((uint64_t) (uint32_t) a << 32) | (uint32_t) b) ^ (((uint64_t) (uint32_t) c << 2) | (uint32_t) s));
        auto it = occ.find(k);
        if (it == occ.end()) {
            occ.emplace(k, 1);
            keys.push_back({ { a, b, c, s }, k });
        } else {
            it->second++;
        }
    };
    int32_t last_chunk = -1; // the chunk of the token before (a token without one takes it)
    for (size_t i = 0; i < n_prompt; ++i) {
        int32_t c = pc.of_tok[i];
        if (c < 0) {
            c = last_chunk;
        } else {
            last_chunk = c;
        }
        const uint8_t s = c >= 0 ? status[c] : (uint8_t) KNOWN;
        if (s == KNOWN) {
            st.tok_known += 1;
        } else if (s == ABSORBED) {
            st.tok_absorbed += 1;
        } else {
            (s == NEW_X ? st.tok_x : st.tok_u) += 1;
            if (i >= 2) {
                event(prompt[i - 2], prompt[i - 1], prompt[i], s == NEW_X ? SRC_X : SRC_U);
            }
        }
    }
    auto at = [&](size_t i) { return i < n_prompt ? prompt[i] : gen[i - n_prompt]; };
    for (size_t j = 0; j < n_gen; ++j) {
        const size_t i = n_prompt + j;
        if (i >= 2) {
            event(at(i - 2), at(i - 1), at(i), SRC_M);
        }
    }
    st.tok_m = (double) n_gen;

    // the answer's number for this prompt
    const uint64_t ph = tp_fnv(prompt, n_prompt * sizeof(int32_t));
    int32_t mth = 1;
    {
        auto it = std::find_if(m.regen.begin(), m.regen.end(), [&](const auto & e) { return e.first == ph; });
        if (it != m.regen.end()) {
            mth = it->second + 1;
            m.regen.erase(it);
        }
        if (n_gen > 0) {
            m.regen.emplace_back(ph, mth);
            if (m.regen.size() > 1024) {
                m.regen.erase(m.regen.begin());
            }
        } else {
            mth = std::max(1, mth - 1);
        }
    }
    st.regen_m = mth;
    st.weight  = 1.0 / (1.0 + (mth - 1) * m.cfg.rho);

    for (const auto & [k4, k] : keys) {
        const uint32_t o = occ[k];
        const uint32_t c = std::min<uint32_t>(o, (uint32_t) m.cfg.sat);
        const float    w = (float) (k4[3] == SRC_M ? c * st.weight : (double) c);
        st.sat_cut += (double) (o - c);
        m.add(k4[0], k4[1], k4[2], k4[3], w);
    }

    // the chunks: the prompt's are known from now on, and so are those of the generated tokens
    for (size_t c = 0; c < nc; ++c) {
        m.chunk_mark(pc.id[c]);
    }
    if (n_gen > 0) {
        tp_chunks gc;
        m.cut(gen, n_gen, nullptr, gc);
        for (uint64_t id : gc.id) {
            m.chunk_mark(id);
        }
        m.ring.push_back(gc.norm.size() > 65536 ? gc.norm.substr(gc.norm.size() - 65536) : gc.norm);
        if ((int32_t) m.ring.size() > std::max<int32_t>(1, m.cfg.ring_n)) {
            m.ring.erase(m.ring.begin());
        }
    }

    m.clock += (double) n_gen;
    if (m.nodes.n > m.nodes.limit() || m.heads.n > m.heads.limit()) {
        m.compact(0.75);
    }
    m.dirty = true;
    m.req_since_save++;
    m.stats_sizes();
    st.commit_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void common_spec_tprof::maybe_save(bool force) {
    auto & m = *impl;
    if (!m.on || m.read_only || !m.dirty) {
        return;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - m.t_save).count();
    if (force || m.req_since_save >= m.cfg.save_every_req || s >= m.cfg.save_every_s) {
        m.save_now(false);
    }
}

bool common_spec_tprof::get_coef(const std::string & key, common_spec_tprof_coef & c) const {
    for (const auto & x : impl->coefs) {
        if (x.key == key) {
            c = x;
            return true;
        }
    }
    return false;
}

void common_spec_tprof::put_coef(const common_spec_tprof_coef & c) {
    for (auto & x : impl->coefs) {
        if (x.key == c.key) {
            x = c;
            impl->dirty = true;
            return;
        }
    }
    impl->coefs.push_back(c);
    impl->dirty = true;
}

const common_spec_tprof_stats & common_spec_tprof::stats() const { return impl->st; }
const common_spec_tprof_config & common_spec_tprof::config() const { return impl->cfg; }

double common_spec_tprof::weight(int32_t a, int32_t b, int32_t c, int32_t s) const {
    tp_node key{};
    key.a = a; key.b = b; key.c = c;
    const tp_node * e = impl->nodes.get(key);
    return e == nullptr ? 0.0 : e->n[s] * impl->decay(e->t);
}

double common_spec_tprof::context_weight(int32_t a, int32_t b, int32_t s) const {
    tp_head key{};
    key.a = a; key.b = b;
    const tp_head * e = impl->heads.get(key);
    return e == nullptr ? 0.0 : e->N[s] * impl->decay(e->t);
}

std::vector<uint64_t> common_spec_tprof::chunk_ids(const int32_t * toks, size_t n) const {
    tp_chunks c;
    impl->cut(toks, n, nullptr, c);
    return c.id;
}

bool common_spec_tprof::known(uint64_t id) const { return impl->chunk_known(id); }
size_t common_spec_tprof::n_nodes() const { return impl->nodes.n; }
