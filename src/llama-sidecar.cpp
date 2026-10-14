#include "llama-sidecar.h"

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-expert-sidecar.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
extern char ** environ;
#endif

// File layout: docs/ranma/expert-sidecar.md. Everything little-endian; the writer assumes a little-endian host.

namespace {

enum record_type : uint16_t { REC_CTX = 1, REC_CALL = 2, REC_EVENT = 3, REC_NOTE = 4 };
enum event_dropped : uint16_t { EVENT_DROPPED = 4 };
enum call_mode : uint8_t { MODE_IDS = 0, MODE_COUNTS = 1, MODE_TOKENS = 2 };
enum call_source : uint8_t { SOURCE_NONE = 0, SOURCE_DEVICE = 1, SOURCE_EVAL = 2 };
enum call_flags : uint16_t { CALL_TRUNCATED = 1, CALL_TIER = 2, CALL_FORCED_SYNC = 4, CALL_NO_SOURCE = 8 };

constexpr uint32_t block_magic    = 0x4B4C4253; // "SBLK"
constexpr size_t   block_target   = 256*1024;
constexpr int64_t  block_age_us   = 1000*1000;
constexpr size_t   backlog_limit  = 64u*1024*1024;

uint32_t crc32_ieee(const uint8_t * p, size_t n) {
    static uint32_t table[256];
    static std::once_flag once;
    std::call_once(once, [] {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = c & 1 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            }
            table[i] = c;
        }
    });
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; ++i) {
        c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

struct bytes {
    std::vector<uint8_t> v;
    template <typename T> void put(T x) { put_raw(&x, sizeof(T)); }
    void put_raw(const void * p, size_t n) {
        const size_t at = v.size();
        v.resize(at + n);
        if (n) { memcpy(v.data() + at, p, n); }
    }
    void pad4() { v.resize((v.size() + 3) & ~(size_t) 3, 0); }
};

std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += (char) c;
                }
        }
    }
    return o;
}

std::string q(const std::string & s) { return "\"" + json_escape(s) + "\""; }

#ifdef _WIN32
std::string utf8(const wchar_t * w, int n = -1) {
    if (w == nullptr) { return std::string(); }
    const int len = WideCharToMultiByte(CP_UTF8, 0, w, n, nullptr, 0, nullptr, nullptr);
    if (len <= 0) { return std::string(); }
    std::string s((size_t) len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, n, s.data(), len, nullptr, nullptr);
    if (n < 0 && !s.empty() && s.back() == '\0') { s.pop_back(); }
    return s;
}
#endif

std::string process_cmdline() {
#ifdef _WIN32
    return utf8(GetCommandLineW());
#else
    std::string s;
    if (FILE * f = fopen("/proc/self/cmdline", "rb")) {
        int c;
        while ((c = fgetc(f)) != EOF) { s += c == 0 ? ' ' : (char) c; }
        fclose(f);
    }
    return s;
#endif
}

std::string process_exe() {
#ifdef _WIN32
    wchar_t buf[32768];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) (sizeof(buf)/sizeof(buf[0])));
    return utf8(buf, (int) n);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
#endif
}

uint32_t process_id() {
#ifdef _WIN32
    return (uint32_t) GetCurrentProcessId();
#else
    return (uint32_t) getpid();
#endif
}

// "name":"value" pairs of the variables that set up a run
std::string process_env_json() {
    static const char * prefixes[] = { "RANMA_", "LLAMA_", "GGML_", "LLM_", "HIP_" };
    std::map<std::string, std::string> vars;
    auto take = [&](const std::string & kv) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) { return; }
        const std::string k = kv.substr(0, eq);
        for (const char * p : prefixes) {
            if (k.compare(0, strlen(p), p) == 0) {
                vars[k] = kv.substr(eq + 1);
                return;
            }
        }
    };
#ifdef _WIN32
    if (wchar_t * env = GetEnvironmentStringsW()) {
        for (const wchar_t * p = env; *p; p += wcslen(p) + 1) {
            take(utf8(p));
        }
        FreeEnvironmentStringsW(env);
    }
#else
    for (char ** e = environ; e && *e; ++e) { take(*e); }
#endif
    std::string s = "{";
    for (const auto & kv : vars) {
        s += (s.size() > 1 ? "," : "") + q(kv.first) + ":" + q(kv.second);
    }
    return s + "}";
}

struct settings {
    std::string path;
    int  prefill      = MODE_TOKENS; // MODE_TOKENS (off), MODE_COUNTS or MODE_IDS
    int  rows         = 64;
    int  prefill_rows = 4096;
    bool tier         = false;
    std::string source = "auto";      // auto | device | eval
};

const settings & get_settings() {
    static const settings s = [] {
        settings r;
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR")) {
            r.path = v;
        }
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR_PREFILL")) {
            if (strcmp(v, "counts") == 0) { r.prefill = MODE_COUNTS; }
            else if (strcmp(v, "ids") == 0) { r.prefill = MODE_IDS; }
            else if (v[0] != '\0' && strcmp(v, "off") != 0) {
                LLAMA_LOG_WARN("%s: RANMA_EXPERT_SIDECAR_PREFILL=%s is not off|counts|ids; off\n", __func__, v);
            }
        }
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR_ROWS")) {
            r.rows = std::max(1, atoi(v));
        }
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR_PREFILL_ROWS")) {
            r.prefill_rows = std::max(1, atoi(v));
        }
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR_TIER")) {
            r.tier = v[0] != '\0' && strcmp(v, "0") != 0;
        }
        if (const char * v = getenv("RANMA_EXPERT_SIDECAR_SOURCE")) {
            if (strcmp(v, "device") == 0 || strcmp(v, "eval") == 0 || strcmp(v, "auto") == 0) {
                r.source = v;
            } else {
                LLAMA_LOG_WARN("%s: RANMA_EXPERT_SIDECAR_SOURCE=%s is not auto|device|eval; auto\n", __func__, v);
            }
        }
        return r;
    }();
    return s;
}

// One file per process. The compute threads append records to the open block; a full or old block goes to the
// writer thread, which writes and flushes whole blocks.
class writer {
public:
    static writer * get() {
        static writer * w = [] () -> writer * {
            const settings & s = get_settings();
            if (s.path.empty()) {
                return nullptr;
            }
            static writer inst;
            return inst.open(s) ? &inst : nullptr;
        }();
        return w;
    }

    int64_t now_us() const { return ggml_time_us() - t0_us_; }

    uint32_t new_ctx_id() { return next_ctx_.fetch_add(1); }

    void append(uint16_t type, uint16_t flags, const bytes & body, bool hand_over = false) {
        std::unique_lock<std::mutex> lock(mutex_);
        const uint32_t n = (uint32_t) (8 + ((body.v.size() + 3) & ~(size_t) 3));
        const size_t at = cur_.size();
        cur_.resize(at + n, 0);
        memcpy(cur_.data() + at, &type, 2);
        memcpy(cur_.data() + at + 2, &flags, 2);
        memcpy(cur_.data() + at + 4, &n, 4);
        if (!body.v.empty()) {
            memcpy(cur_.data() + at + 8, body.v.data(), body.v.size());
        }
        if (cur_records_ == 0) {
            cur_t0_ = ggml_time_us();
        }
        cur_records_++;
        if (hand_over || cur_.size() >= block_target || ggml_time_us() - cur_t0_ >= block_age_us) {
            hand_over_locked();
        }
    }

    void flush() {
        std::unique_lock<std::mutex> lock(mutex_);
        hand_over_locked();
        drained_.wait(lock, [this] { return queue_.empty() && !writing_; });
    }

    ~writer() {
        // Static destruction at process exit: on Windows the writer thread may already be gone. Never block here.
        if (!mutex_.try_lock()) {
            return;
        }
        hand_over_locked();
        stop_ = true;
        std::deque<std::vector<uint8_t>> rest;
        if (!writing_) {
            rest.swap(queue_);
        }
        mutex_.unlock();
        cv_.notify_all();
        for (auto & b : rest) {
            write_block(b);
        }
        if (thread_.joinable()) {
            thread_.detach();
        }
    }

private:
    bool open(const settings & s) {
        std::string path = s.path;
        const size_t at = path.find("{pid}");
        if (at != std::string::npos) {
            path.replace(at, 5, std::to_string(process_id()));
        }
#ifdef _WIN32
        std::wstring wpath;
        {
            const int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
            wpath.resize(n > 0 ? (size_t) n : 0);
            if (n > 0) { MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), n); }
        }
        f_ = _wfopen(wpath.c_str(), L"wb");
#else
        f_ = fopen(path.c_str(), "wb");
#endif
        if (f_ == nullptr) {
            LLAMA_LOG_ERROR("%s: cannot open the expert sidecar '%s'; sidecar off\n", __func__, path.c_str());
            return false;
        }
        t0_us_ = ggml_time_us();
        const int64_t unix_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const char * prefill = s.prefill == MODE_IDS ? "ids" : s.prefill == MODE_COUNTS ? "counts" : "off";
        const std::string json = std::string("{\"version\":1") +
            ",\"pid\":" + std::to_string(process_id()) +
            ",\"t0_unix_us\":" + std::to_string(unix_us) +
            ",\"exe\":" + q(process_exe()) +
            ",\"cmdline\":" + q(process_cmdline()) +
            ",\"env\":" + process_env_json() +
            ",\"settings\":{\"prefill\":" + q(prefill) + ",\"rows\":" + std::to_string(s.rows) +
            ",\"prefill_rows\":" + std::to_string(s.prefill_rows) + ",\"tier\":" + (s.tier ? "true" : "false") +
            ",\"source\":" + q(s.source) + "}}";
        bytes h;
        h.put_raw("RNMSIDE1", 8);
        h.put<uint32_t>(1);
        h.put<uint32_t>((uint32_t) json.size());
        h.put_raw(json.data(), json.size());
        h.v.resize((h.v.size() + 7) & ~(size_t) 7, 0);
        fwrite(h.v.data(), 1, h.v.size(), f_);
        fflush(f_);
        thread_ = std::thread([this] { run(); });
        LLAMA_LOG_INFO("%s: expert sidecar -> '%s' (prefill %s, rows %d, prefill rows %d, tier %d, source %s)\n",
                __func__, path.c_str(), prefill, s.rows, s.prefill_rows, s.tier ? 1 : 0, s.source.c_str());
        return true;
    }

    void hand_over_locked() {
        if (cur_records_ == 0) {
            return;
        }
        std::vector<uint8_t> blk(24 + cur_.size());
        const uint32_t hdr[6] = { block_magic, seq_++, (uint32_t) cur_.size(), cur_records_,
                                  crc32_ieee(cur_.data(), cur_.size()), 0 };
        memcpy(blk.data(), hdr, sizeof(hdr));
        memcpy(blk.data() + 24, cur_.data(), cur_.size());
        cur_.clear();
        cur_records_ = 0;
        if (queued_bytes_ + blk.size() > backlog_limit) {
            dropped_blocks_++;
            dropped_bytes_ += blk.size();
            return;
        }
        if (dropped_blocks_ > 0) {
            // the loss is recorded in the next block that is kept
            bytes ev;
            ev.put<uint32_t>(0xFFFFFFFFu);
            ev.put<uint16_t>(EVENT_DROPPED);
            ev.put<uint16_t>(0);
            ev.put<int32_t>(-1);
            ev.put<uint64_t>((uint64_t) now_us());
            ev.put<int64_t>((int64_t) dropped_blocks_);
            ev.put<int64_t>((int64_t) dropped_bytes_);
            ev.put<int64_t>(0);
            ev.put<int64_t>(0);
            dropped_blocks_ = 0;
            dropped_bytes_  = 0;
            const uint32_t n = (uint32_t) (8 + ev.v.size());
            const uint16_t t = REC_EVENT, fl = 0;
            std::vector<uint8_t> rec(n);
            memcpy(rec.data(), &t, 2);
            memcpy(rec.data() + 2, &fl, 2);
            memcpy(rec.data() + 4, &n, 4);
            memcpy(rec.data() + 8, ev.v.data(), ev.v.size());
            std::vector<uint8_t> payload(blk.begin() + 24, blk.end());
            payload.insert(payload.end(), rec.begin(), rec.end());
            const uint32_t hdr2[6] = { block_magic, hdr[1], (uint32_t) payload.size(), hdr[3] + 1,
                                       crc32_ieee(payload.data(), payload.size()), 0 };
            blk.resize(24 + payload.size());
            memcpy(blk.data(), hdr2, sizeof(hdr2));
            memcpy(blk.data() + 24, payload.data(), payload.size());
        }
        queued_bytes_ += blk.size();
        queue_.push_back(std::move(blk));
        cv_.notify_one();
    }

    void write_block(const std::vector<uint8_t> & b) {
        fwrite(b.data(), 1, b.size(), f_);
        fflush(f_);
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
                if (stop_) { break; }
                continue;
            }
            std::vector<uint8_t> b = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
            lock.unlock();
            write_block(b);
            lock.lock();
            writing_ = false;
            queued_bytes_ -= b.size();
            if (queue_.empty()) {
                drained_.notify_all();
            }
        }
        drained_.notify_all();
    }

    FILE * f_ = nullptr;
    int64_t t0_us_ = 0;
    std::atomic<uint32_t> next_ctx_{ 0 };
    std::mutex mutex_;
    std::condition_variable cv_, drained_;
    std::vector<uint8_t> cur_;
    uint32_t cur_records_ = 0;
    int64_t  cur_t0_ = 0;
    uint32_t seq_ = 0;
    std::deque<std::vector<uint8_t>> queue_;
    size_t queued_bytes_ = 0;
    bool writing_ = false;
    bool stop_ = false;
    uint64_t dropped_blocks_ = 0, dropped_bytes_ = 0;
    std::thread thread_;
};

// host-built layer record (eval source)
struct eval_layer {
    uint32_t rows = 0, stored = 0, launches = 0;
    uint16_t n_used = 0;
    std::vector<uint16_t> ids;
    std::vector<uint32_t> counts;
};

struct pending_call {
    uint32_t call_seq = 0;
    int32_t  req_id = -1;
    uint8_t  phase = 0, mode = MODE_IDS, source = SOURCE_NONE;
    uint16_t flags = 0;
    bool     prefill = false;
    uint32_t ticket = 0;
    uint64_t t_begin_us = 0;
    std::vector<int32_t> token, pos, seq;
    std::vector<uint8_t> output;
    std::map<int, eval_layer> layers; // eval source
};

bool parse_topk_layer(const char * name, int & layer) {
    static constexpr const char prefix[] = "ffn_moe_topk-";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }
    char * end = nullptr;
    const long l = strtol(name + sizeof(prefix) - 1, &end, 10);
    if (end == name + sizeof(prefix) - 1 || *end != '\0' || l < 0 || l > 0xFFFF) {
        return false;
    }
    layer = (int) l;
    return true;
}

const ggml_expert_sidecar_iface * device_iface(ggml_backend_t backend) {
    if (backend == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }
    auto * get = (ggml_backend_expert_sidecar_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
            GGML_EXPERT_SIDECAR_PROC_NAME);
    const ggml_expert_sidecar_iface * t = get ? get() : nullptr;
    return t && t->version == GGML_EXPERT_SIDECAR_VERSION ? t : nullptr;
}

} // namespace

struct llama_sidecar_ctx {
    llama_context * lctx = nullptr;
    writer * w = nullptr;
    uint32_t ctx_id = 0;
    uint8_t  source = SOURCE_NONE;
    ggml_backend_t backend = nullptr;              // device source
    const ggml_expert_sidecar_iface * dev = nullptr;
    bool     dev_ready = false;
    bool     dev_warned = false;
    int      depth = 0;                            // decode calls on the stack
    uint32_t next_seq = 0;
    int64_t  last_seq = -1;
    int32_t  req_id = -1;
    int32_t  phase = LLAMA_SIDECAR_PHASE_AUTO;
    std::deque<pending_call> pending;
    pending_call * current = nullptr;              // the call being computed (eval source)
    std::vector<uint8_t> buf;

    void write_call(pending_call & c) {
        const settings & s = get_settings();
        std::vector<uint8_t> layers;
        int32_t n_blocks = 0;
        if (c.source == SOURCE_DEVICE && c.ticket != 0 && dev != nullptr) {
            bool cut = false;
            size_t n = dev->end(backend, c.ticket, s.tier && c.mode == MODE_IDS, buf.data(), buf.size(), &n_blocks, &cut);
            if (n > buf.size()) {
                buf.resize(n + n/4);
                n = dev->end(backend, c.ticket, s.tier && c.mode == MODE_IDS, buf.data(), buf.size(), &n_blocks, &cut);
            }
            if (n <= buf.size()) {
                layers.assign(buf.begin(), buf.begin() + n);
            } else {
                n_blocks = 0;
            }
            if (cut) { c.flags |= CALL_TRUNCATED; }
            if (s.tier && c.mode == MODE_IDS) { c.flags |= CALL_TIER; }
        } else if (c.source == SOURCE_EVAL) {
            bytes b;
            for (auto & kv : c.layers) {
                const eval_layer & L = kv.second;
                b.put<uint16_t>((uint16_t) kv.first);
                if (c.mode == MODE_IDS) {
                    b.put<uint16_t>(L.n_used);
                    b.put<uint32_t>(L.rows);
                    b.put<uint32_t>(L.stored);
                    b.put<uint32_t>(L.launches);
                    b.put_raw(L.ids.data(), L.ids.size()*sizeof(uint16_t));
                    b.pad4();
                    if (L.stored < L.rows) { c.flags |= CALL_TRUNCATED; }
                } else {
                    b.put<uint16_t>((uint16_t) L.counts.size());
                    b.put<uint32_t>(L.rows);
                    b.put<uint32_t>(0);
                    b.put<uint32_t>(L.launches);
                    b.put_raw(L.counts.data(), L.counts.size()*sizeof(uint32_t));
                }
                n_blocks++;
            }
            layers.swap(b.v);
        }
        if (c.mode == MODE_TOKENS) {
            layers.clear();
            n_blocks = 0;
        }
        const uint32_t n = (uint32_t) c.token.size();
        bytes r;
        r.v.reserve(48 + (size_t) n*13 + layers.size());
        r.put<uint32_t>(ctx_id);
        r.put<uint32_t>(c.call_seq);
        r.put<int32_t>(c.req_id);
        r.put<uint8_t>(c.phase);
        r.put<uint8_t>(c.mode);
        r.put<uint8_t>(c.source);
        r.put<uint8_t>(0);
        r.put<uint64_t>(c.t_begin_us);
        r.put<uint32_t>((uint32_t) std::max<int64_t>(0, w->now_us() - (int64_t) c.t_begin_us));
        r.put<uint32_t>(n);
        r.put<uint32_t>((uint32_t) n_blocks);
        r.put<uint32_t>(0);
        r.put_raw(c.token.data(), (size_t) n*4);
        r.put_raw(c.pos.data(), (size_t) n*4);
        r.put_raw(c.seq.data(), (size_t) n*4);
        r.put_raw(c.output.data(), n);
        r.pad4();
        r.put_raw(layers.data(), layers.size());
        w->append(REC_CALL, c.flags, r);
    }

    void collect_all() {
        while (!pending.empty()) {
            write_call(pending.front());
            pending.pop_front();
        }
    }

    // eval callback: the router selections of host tensors
    static bool eval_cb(ggml_tensor * t, bool ask, void * ud) {
        auto * sc = (llama_sidecar_ctx *) ud;
        int layer = -1;
        if (t == nullptr || t->type != GGML_TYPE_I32 || !parse_topk_layer(t->name, layer)) {
            return ask ? false : true;
        }
        if (ask) {
            return t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer) && t->ne[2] == 1 && t->ne[3] == 1 &&
                   t->nb[0] == sizeof(int32_t);
        }
        pending_call * c = sc->current;
        if (c == nullptr || c->mode == MODE_TOKENS || t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
            return true;
        }
        const settings & s = get_settings();
        const int n_used = (int) t->ne[0];
        const int n_rows = (int) t->ne[1];
        eval_layer & L = c->layers[layer];
        L.launches++;
        L.n_used = (uint16_t) n_used;
        const uint32_t cap_rows = (uint32_t) (c->prefill ? s.prefill_rows : s.rows);
        const int n_expert = (int) sc->lctx->get_model().hparams.n_expert;
        if (c->mode == MODE_COUNTS && L.counts.empty()) {
            L.counts.assign((size_t) std::max(1, n_expert), 0);
        }
        for (int r = 0; r < n_rows; ++r) {
            const int32_t * row = (const int32_t *) ((const char *) t->data + (size_t) r*t->nb[1]);
            const bool keep = c->mode == MODE_IDS && L.stored < cap_rows;
            for (int k = 0; k < n_used; ++k) {
                const int e = row[k];
                if (keep) {
                    L.ids.push_back(e < 0 || e > 0xFFFE ? (uint16_t) 0xFFFF : (uint16_t) e);
                } else if (c->mode == MODE_COUNTS && e >= 0 && (size_t) e < L.counts.size()) {
                    L.counts[(size_t) e]++;
                }
            }
            if (keep) { L.stored++; }
            L.rows++;
        }
        return true;
    }
};

void llama_sidecar_ctx_deleter::operator()(llama_sidecar_ctx * p) const {
    if (p == nullptr) {
        return;
    }
    p->collect_all();
    if (p->w) {
        p->w->flush();
    }
    delete p;
}

llama_sidecar_ctx_ptr llama_sidecar_ctx_init(llama_context * lctx, ggml_backend_t expert_backend, bool gpu_compute,
        ggml_backend_sched_eval_callback * cb, void ** cb_ud) {
    writer * w = writer::get();
    if (w == nullptr || lctx == nullptr) {
        return nullptr;
    }
    const settings & s = get_settings();
    llama_sidecar_ctx_ptr sc(new llama_sidecar_ctx());
    sc->lctx   = lctx;
    sc->w      = w;
    sc->ctx_id = w->new_ctx_id();

    const ggml_expert_sidecar_iface * dev = s.source != "eval" ? device_iface(expert_backend) : nullptr;
    if (dev != nullptr) {
        sc->source  = SOURCE_DEVICE;
        sc->backend = expert_backend;
        sc->dev     = dev;
    } else if (s.source != "device" && (!gpu_compute || s.source == "eval") && cb != nullptr && *cb == nullptr) {
        sc->source = SOURCE_EVAL;
        *cb    = &llama_sidecar_ctx::eval_cb;
        *cb_ud = sc.get();
    }

    const llama_model & model = lctx->get_model();
    const llama_cparams & cp  = lctx->get_cparams();
    std::string cache = "null";
    if (lctx->expert_iface != nullptr && lctx->expert_iface->status != nullptr) {
        ggml_expert_status st = {};
        if (lctx->expert_iface->status(&st)) {
            cache = std::string("{\"installed\":") + (st.installed ? "true" : "false") +
                    ",\"device_bytes\":" + std::to_string(st.device_bytes) +
                    ",\"host_bytes\":" + std::to_string(st.host_bytes) + "}";
        }
    }
    const char * src = sc->source == SOURCE_DEVICE ? "device" : sc->source == SOURCE_EVAL ? "eval" : "none";
    const std::string json = "{\"ctx_id\":" + std::to_string(sc->ctx_id) +
        ",\"arch\":" + q(llm_arch_name(model.arch)) +
        ",\"desc\":" + q(model.desc()) +
        ",\"n_layer\":" + std::to_string(model.hparams.n_layer()) +
        ",\"n_layer_all\":" + std::to_string(model.hparams.n_layer_all) +
        ",\"n_expert\":" + std::to_string(model.hparams.n_expert) +
        ",\"n_expert_used\":" + std::to_string(model.hparams.n_expert_used_max()) +
        ",\"n_ctx\":" + std::to_string(cp.n_ctx) +
        ",\"n_batch\":" + std::to_string(cp.n_batch) +
        ",\"n_ubatch\":" + std::to_string(cp.n_ubatch) +
        ",\"n_seq_max\":" + std::to_string(cp.n_seq_max) +
        ",\"source\":" + q(src) +
        ",\"expert_cache\":" + cache + "}";
    bytes b;
    b.put<uint32_t>(sc->ctx_id);
    b.put<uint32_t>((uint32_t) json.size());
    b.put_raw(json.data(), json.size());
    w->append(REC_CTX, 0, b);
    LLAMA_LOG_INFO("%s: expert sidecar ctx %u: source %s\n", __func__, sc->ctx_id, src);
    return sc;
}

void llama_sidecar_call_begin(llama_sidecar_ctx * sc, const llama_batch_ext & batch, ggml_backend_sched_t sched) {
    if (sc == nullptr) {
        return;
    }
    const settings & s = get_settings();
    uint16_t flags = 0;
    if (sc->depth == 0 && sc->source == SOURCE_DEVICE && sc->pending.size() >= GGML_EXPERT_SIDECAR_BANKS) {
        // the oldest bank would be reused before it was read
        if (sched != nullptr) {
            ggml_backend_sched_synchronize(sched);
        }
        sc->collect_all();
        flags |= CALL_FORCED_SYNC;
    }
    sc->depth++;

    pending_call c;
    c.call_seq   = sc->next_seq++;
    c.req_id     = sc->req_id;
    c.t_begin_us = (uint64_t) sc->w->now_us();
    c.source     = sc->source;
    c.flags      = flags;
    const size_t n = batch.tokens.size();
    c.token.resize(n);
    c.pos.resize(n);
    c.seq.resize(n);
    c.output.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const auto & t = batch.tokens[i];
        c.token[i]  = t.id;
        c.pos[i]    = t.pos[0];
        int32_t sq = -1;
        for (llama_seq_id id : t.seq_ids) {
            sq = sq < 0 ? id : std::min(sq, id);
        }
        c.seq[i]    = sq;
        c.output[i] = t.output ? 1 : 0;
    }
    int32_t phase = sc->phase;
    if (phase == LLAMA_SIDECAR_PHASE_AUTO) {
        phase = n == 1 ? LLAMA_SIDECAR_PHASE_DECODE : LLAMA_SIDECAR_PHASE_PREFILL;
    }
    c.phase   = (uint8_t) phase;
    // a long call of another phase (the draft model's prompt catch-up) is a prefill too; a verify call stays one
    c.prefill = phase == LLAMA_SIDECAR_PHASE_PREFILL || phase == LLAMA_SIDECAR_PHASE_MIXED ||
                (phase != LLAMA_SIDECAR_PHASE_VERIFY && n > (size_t) s.rows);
    c.mode    = c.prefill ? (uint8_t) s.prefill : (uint8_t) MODE_IDS;
    if (c.source == SOURCE_DEVICE) {
        // a joint cache installs only once every model joined: until then each call tries again
        if (!sc->dev_ready) {
            sc->dev_ready = sc->dev->configure(s.rows, s.prefill == MODE_IDS ? s.prefill_rows : 0);
            if (!sc->dev_ready && !sc->dev_warned) {
                sc->dev_warned = true;
                LLAMA_LOG_WARN("%s: expert sidecar ctx %u: the expert cache is not installed yet; calls without ids until it is\n",
                        __func__, sc->ctx_id);
            }
        }
        if (sc->dev_ready) {
            // a tokens-only call still opens a bank (mode none), so that its kernels do not add rows to an earlier call
            const auto mode = c.mode == MODE_TOKENS ? GGML_EXPERT_SIDECAR_NONE :
                              c.mode == MODE_COUNTS ? GGML_EXPERT_SIDECAR_COUNTS : GGML_EXPERT_SIDECAR_IDS;
            c.ticket = sc->dev->begin(sc->backend, mode, c.prefill);
        }
    }
    if (c.source == SOURCE_NONE || (c.source == SOURCE_DEVICE && c.ticket == 0 && c.mode != MODE_TOKENS)) {
        c.flags |= CALL_NO_SOURCE;
    }
    sc->last_seq = c.call_seq;
    sc->pending.push_back(std::move(c));
    sc->current = &sc->pending.back();
}

void llama_sidecar_call_end(llama_sidecar_ctx * sc) {
    if (sc == nullptr) {
        return;
    }
    sc->depth = std::max(0, sc->depth - 1);
    if (sc->depth == 0) {
        sc->current = nullptr;
    }
}

void llama_sidecar_collect(llama_sidecar_ctx * sc) {
    if (sc == nullptr || sc->depth > 0 || sc->pending.empty()) {
        return;
    }
    sc->collect_all();
}

// ---- C API ----

bool llama_sidecar_enabled(void) {
    return writer::get() != nullptr;
}

void llama_sidecar_set_phase(llama_context * ctx, int32_t req_id, llama_sidecar_phase phase) {
    if (ctx == nullptr || !ctx->sidecar) {
        return;
    }
    ctx->sidecar->req_id = req_id;
    ctx->sidecar->phase  = (int32_t) phase;
}

int64_t llama_sidecar_call_seq(const llama_context * ctx) {
    return ctx != nullptr && ctx->sidecar ? ctx->sidecar->last_seq : -1;
}

int32_t llama_sidecar_ctx_id(const llama_context * ctx) {
    return ctx != nullptr && ctx->sidecar ? (int32_t) ctx->sidecar->ctx_id : -1;
}

void llama_sidecar_event(llama_context * ctx, llama_sidecar_event_kind kind, int32_t req_id, int64_t a, int64_t b, int64_t c,
        int64_t d) {
    writer * w = writer::get();
    if (w == nullptr) {
        return;
    }
    bytes r;
    r.put<uint32_t>(ctx != nullptr && ctx->sidecar ? ctx->sidecar->ctx_id : 0xFFFFFFFFu);
    r.put<uint16_t>((uint16_t) kind);
    r.put<uint16_t>(0);
    r.put<int32_t>(req_id);
    r.put<uint64_t>((uint64_t) w->now_us());
    r.put<int64_t>(a);
    r.put<int64_t>(b);
    r.put<int64_t>(c);
    r.put<int64_t>(d);
    w->append(REC_EVENT, 0, r, kind == LLAMA_SIDECAR_EVENT_REQ_END);
}

void llama_sidecar_note(const char * key, const char * value) {
    writer * w = writer::get();
    if (w == nullptr || key == nullptr) {
        return;
    }
    const size_t nk = strlen(key);
    const size_t nv = value ? strlen(value) : 0;
    bytes r;
    r.put<uint32_t>((uint32_t) nk);
    r.put<uint32_t>((uint32_t) nv);
    r.put_raw(key, nk);
    r.put_raw(value, nv);
    w->append(REC_NOTE, 0, r);
}

void llama_sidecar_flush(void) {
    if (writer * w = writer::get()) {
        w->flush();
    }
}
