#include "expert-controller.cuh"

#if defined(GGML_USE_HIP)

#include "expert-geometry.h"
#include "expert-hash-early.h"
#include "expert-hash-stage.cuh"
#include "expert-host.cuh"
#include "expert-l2.cuh"
#include "expert-os.h"
#include "expert-plan.h"
#include "expert-profiler.cuh"
#include "expert-redraw.h"
#include "expert-profile-store.h"
#include "expert-storage.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// Histograms are cheap (n_counts * 4 bytes each), so a fixed number of banks is allocated at
// finalize time and bank_open hands them out; the caller's policy decides what each one means.
static constexpr uint32_t max_banks = 4;

// Models of one joint cache, and compute contexts that run them.
static constexpr size_t max_members  = 4;
static constexpr int    max_attached = 16;

enum class state_t { unconfigured, configured, registered, installed, disabled };

static const char * state_name(state_t s) {
    switch (s) {
        case state_t::unconfigured: return "unconfigured";
        case state_t::configured:   return "configured";
        case state_t::registered:   return "registered";
        case state_t::installed:    return "installed";
        case state_t::disabled:     return "disabled";
    }
    return "?";
}

// A bank counts every model of the cache: its histogram spans the joint layers, and each model
// keeps its own store of records, so each model's half-life score comes from its own records.
struct bank_state {
    std::string label;
    bool prompt = false;                                // the caller's policy says this bank counts prompt processing;
                                                        // set by whichever bank_open names it so, never cleared
    std::vector<std::unique_ptr<profile_store>> stores; // [member]; empty when profiling is off
    std::vector<uint8_t> usable;                        // [member] open() returned ok/missing/corrupt, not incompatible
    ggml_expert_plan_id latest_plan = GGML_EXPERT_PLAN_NONE;
    uint64_t commits = 0;

    bool any_usable() const {
        for (size_t m = 0; m < stores.size(); ++m) { if (stores[m] && usable[m]) { return true; } }
        return false;
    }
    uint64_t total_selections() const {
        uint64_t total = 0;
        for (size_t m = 0; m < stores.size(); ++m) { if (stores[m] && usable[m]) { total += stores[m]->total_selections(); } }
        return total;
    }
    size_t records() const {
        size_t n = 0;
        for (size_t m = 0; m < stores.size(); ++m) { if (stores[m] && usable[m]) { n = std::max(n, stores[m]->window().size()); } }
        return n;
    }
};

// One model whose routed experts live in the cache. A cache of one model has one member at offset
// 0, and its geometry is the model's own. In a joint cache the members' layers are laid end to
// end in the joint geometry (member m's layer l is joint layer offset + l), and the load-time plan
// and the arenas cover every member. A member declared by the first model's config is described by
// the tensor metadata of its GGUF file until its own weight context arrives and is bound.
struct member_state {
    std::string key;         // "<arch>-<16 hex digits of the geometry signature hash>", the profile directory
    std::string identity;    // architecture name
    std::string path;        // first GGUF file (declared members)
    std::string signature;   // identity, a newline and the geometry signature: the profile compatibility key
    geometry    geo;         // the member's own layout; tensors point into `meta` until bound
    int   offset   = 0;      // first joint layer
    float weight   = 1.0f;   // multiplier on the host read cost in the joint plan
    bool  accepted = false;  // its config was accepted
    bool  bound    = false;  // its weight context is adopted, the joint geometry holds its tensors
    bool  loaded   = false;  // its load finished (finalize)
    bool  released = false;
    bool  dropped  = false;  // exclusive only: its context did not match; its slots stay unused
    bool  tier_ready = false; // its files are open and its backing is checked (SSD tier)
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t exclusive_buffer = nullptr; // owned by the model
    std::vector<gguf_context *> gguf;                 // declared metadata, freed once bound
    std::vector<ggml_context *> meta;

    void free_meta() {
        for (gguf_context * g : gguf) { gguf_free(g); }
        for (ggml_context * c : meta) { ggml_free(c); }
        gguf.clear();
        meta.clear();
    }
};

// Plans are kept only as long as they can still be installed: the newest plan of every bank, plus
// the one that is installed. A commit drops the plan it replaces, and an install drops the plan it
// replaces, so the map holds at most n_banks + 1 entries. Ids are never reused, so an id that is
// gone is reported as stale instead of resolving to another plan.
struct plan_state {
    ggml_expert_bank_id bank = GGML_EXPERT_BANK_NONE;
    std::vector<std::vector<int32_t>> selected;
    placement_stats stats;
    // The SSD tier cuts the host tier from the same scores, but only at install time, because the
    // host capacity depends on the ring class the plan turns out to carry.
    std::vector<uint64_t> scores;
    // [member] joint cache: the model does not run in this bank's phase (a draft model during
    // prompt processing), so the plan leaves its VRAM slices as they are when it is installed
    std::vector<uint8_t> keep;
    // the size-class split the plan was made for (expert-redraw.h): a plan of an older one is placed
    // again from its scores before it is installed
    uint32_t epoch = 0;
};

// Defined below the controller: the exclusive buffer type needs the controller instance.
static ggml_backend_buffer_t exclusive_buffer_create(ggml_context * ctx, ggml_backend_buffer_type_t buft);
// The buffer that holds the tensors of the routed context that are not routed experts, or null.
static ggml_backend_buffer_t exclusive_delegate_buffer(ggml_backend_buffer_t buffer);
bool is_exclusive_buft(ggml_backend_buffer_type_t buft);

// 128-bit FNV-1a over 64-bit words, two lanes with different primes. Exclusive mode has no host
// master to compare a slice against, so the digest of the bytes the loader wrote (which are the
// bytes of the GGUF) is recorded per (layer, kind, expert) and every slot is checked against it.
struct slice_digest {
    uint64_t a = 0xcbf29ce484222325ull;
    uint64_t b = 0x9e3779b97f4a7c15ull;

    bool operator==(const slice_digest & other) const { return a == other.a && b == other.b; }
    bool operator!=(const slice_digest & other) const { return !(*this == other); }
};

static slice_digest digest_of(const void * data, size_t bytes) {
    slice_digest out;
    const unsigned char * p = static_cast<const unsigned char *>(data);
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= bytes; i += sizeof(uint64_t)) {
        uint64_t word = 0;
        memcpy(&word, p + i, sizeof(word));
        out.a = (out.a ^ word)*0x100000001b3ull;
        out.b = (out.b ^ word)*0xc2b2ae3d27d4eb4full;
    }
    for (; i < bytes; ++i) {
        out.a = (out.a ^ (uint64_t) p[i])*0x100000001b3ull;
        out.b = (out.b ^ (uint64_t) p[i])*0xc2b2ae3d27d4eb4full;
    }
    return out;
}

static uint64_t now_s() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// All L2 environment overrides are read here. Normal options arrive in cfg.
static l2_config l2_debug_config(const ggml_expert_config & cfg, bool verify_all) {
    l2_config out;
    out.prefill_ring_bytes = cfg.l2_prefill_ring_bytes;
    out.decode_ring_bytes = cfg.l2_decode_ring_bytes;
    out.worker_cpu = cfg.l2_worker_cpu;
    out.experts_used = cfg.l2_experts_used;
    out.prefill_rows = cfg.l2_prefill_rows;
    out.decode_rows = cfg.l2_decode_rows;
    out.phase_rings = cfg.l2_phase_rings;
    out.log_mask = cfg.log_mask;
    out.verify = verify_all;
    auto number = [](const char * name, int64_t fallback, int64_t minimum, int64_t maximum) {
        const char * text = getenv(name);
        if (!text || !*text) { return fallback; }
        char * end = nullptr;
        errno = 0;
        const long long value = strtoll(text, &end, 10);
        if (errno || end == text || *end || value < minimum || value > maximum) {
            GGML_LOG_WARN("expert cache: ignoring invalid %s='%s'\n", name, text);
            return fallback;
        }
        GGML_LOG_INFO("expert cache: %s sets %lld\n", name, value);
        return int64_t(value);
    };
    out.verify = number("RANMA_EXPERT_L2_VERIFY", out.verify ? 1 : 0, 0, 1) != 0;
    out.read_wait_ms = number("RANMA_EXPERT_L2_WAIT", out.read_wait_ms, 1, INT64_MAX);
    out.queue_depth = int(number("RANMA_EXPERT_L2_QD", out.queue_depth, 1, expert_os::max_queue_depth));
    // 0 = off, 1 = the ubatches with more rows than the decode bound, N > 1 = ubatches of at least N rows.
    const int64_t staged = number("RANMA_EXPERT_L2_STAGED", l2_staged_default ? 1 : 0, 0, INT32_MAX);
    out.staged_min_rows = staged == 1 ? int64_t(std::max<uint64_t>(out.decode_rows, 1)) + 1 : staged;
    out.staged_drain = number("RANMA_EXPERT_L2_STAGED_DRAIN", 0, 0, 1) != 0;
    // The staging ring lives in the per storage class host arenas, next to the residents and at
    // their pitch.
    out.class_layout = true;
    // Class layout: each storage class ring is this many times its floor, as far as the budget
    // allows; 1 = the floors. A real number of at least 1.
    if (const char * text = getenv("RANMA_EXPERT_L2_RING_FACTOR"); text != nullptr && *text != '\0') {
        char * end = nullptr;
        errno = 0;
        const double value = strtod(text, &end);
        if (errno || end == text || *end || !(value >= 1.0) || !(value <= 1024.0)) {
            GGML_LOG_WARN("expert cache: ignoring invalid RANMA_EXPERT_L2_RING_FACTOR='%s'\n", text);
        } else {
            out.ring_factor = value;
            GGML_LOG_INFO("expert cache: RANMA_EXPERT_L2_RING_FACTOR sets %g\n", value);
        }
    }
    return out;
}

// RANMA_EXPERT_L1_REDRAW_BENEFIT and RANMA_EXPERT_L1_REDRAW_MOVE, read once per process.
static const redraw_params & redraw_env() {
    static const redraw_params params = [] {
        std::string warnings;
        const redraw_params p = redraw_params_from([](const char * name) { return (const char *) getenv(name); }, warnings);
        if (!warnings.empty()) {
            GGML_LOG_WARN("expert cache: L1 redraw: %s\n", warnings.c_str());
        }
        return p;
    }();
    return params;
}

class controller {
public:
    // ---- lifecycle -------------------------------------------------------------------------

    bool configure(const ggml_expert_config * config) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (config == nullptr || config->abi_version != GGML_EXPERT_ABI_VERSION) {
            GGML_LOG_ERROR("expert cache: configure rejected: abi %u, expected %u\n",
                config ? config->abi_version : 0u, (unsigned) GGML_EXPERT_ABI_VERSION);
            return false;
        }
        if (state_ != state_t::unconfigured) {
            // A later model joins when the first model's config declared its file.
            if (config->model_path != nullptr && config->model_path[0] != '\0' && state_ != state_t::disabled) {
                for (size_t m = 1; m < members_.size(); ++m) {
                    member_state & mb = members_[m];
                    if (!mb.accepted && !mb.dropped && mb.path == config->model_path) {
                        mb.accepted = true;
                        accepting_  = (int) m;
                        GGML_LOG_INFO("expert cache: model %s (%s) joins the cache as model %zu\n",
                            mb.key.c_str(), mb.path.c_str(), m);
                        return true;
                    }
                }
            }
            // Any other model in the same process runs without the cache.
            GGML_LOG_WARN("expert cache: already %s for another model; the new model is not cached\n", state_name(state_));
            return false;
        }
        cfg_ = *config;
        profile_dir_  = config->profile_dir  ? config->profile_dir  : "";
        initial_bank_ = config->initial_bank ? config->initial_bank : "";
        cfg_.profile_dir  = profile_dir_.c_str();
        cfg_.initial_bank = initial_bank_.c_str();
        if (cfg_.l1_bytes == 0 && cfg_.l2_bytes == 0 && cfg_.policy != GGML_EXPERT_POLICY_OFF) {
            // Nothing to do for this model; stay unconfigured so that a later model may configure.
            return false;
        }
        if (cfg_.mode != GGML_EXPERT_MODE_INCLUSIVE && cfg_.mode != GGML_EXPERT_MODE_EXCLUSIVE) {
            GGML_LOG_WARN("expert cache: unknown mode %d; cache off\n", (int) cfg_.mode);
            return false;
        }
        if (cfg_.policy < GGML_EXPERT_POLICY_ADAPTIVE || cfg_.policy > GGML_EXPERT_POLICY_OFF ||
                (cfg_.policy == GGML_EXPERT_POLICY_OFF && cfg_.l1_bytes != 0)) {
            GGML_LOG_ERROR("expert cache: invalid placement policy or nonzero L1 in OFF policy\n");
            return false;
        }
        if (profile_dir_.empty()) {
            cfg_.policy = cfg_.l1_bytes == 0 ? GGML_EXPERT_POLICY_OFF : GGML_EXPERT_POLICY_STATIC;
            cfg_.freeze = true;
            GGML_LOG_INFO("expert cache: no profile directory; seeded fixed placement, no records or installs\n");
        }
        if (cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE || cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE || !cfg_.l1_bytes) { cfg_.spare_slots = 0; }
        if (cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE || cfg_.l2_bytes != 0 || cfg_.policy == GGML_EXPERT_POLICY_OFF) {
            // validate_expert_params refuses this before the model loads; this is the backend-side
            // guard for a caller that built the config by hand.
            if (!expert_os::supported()) {
                GGML_LOG_WARN("expert cache: owned host storage needs the OS address reservation, which this "
                              "platform does not implement; cache off\n");
                return false;
            }
            if (cfg_.spare_slots < 0 || (cfg_.spare_slots == 0 && cfg_.policy == GGML_EXPERT_POLICY_ADAPTIVE && cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE && cfg_.l1_bytes > 0)) {
                GGML_LOG_WARN("expert cache: exclusive mode needs spare_slots > 0; cache off\n");
                return false;
            }
        }
        // Debug override: read every resident slice back after each install and compare it with
        // the host tensor. Slow (the whole arena crosses PCIe twice): for a correctness check,
        // not for serving.
        const char * verify = getenv("RANMA_EXPERT_VERIFY");
        verify_all_ = verify != nullptr && strcmp(verify, "0") != 0;
        if (verify_all_) {
            GGML_LOG_INFO("expert cache: RANMA_EXPERT_VERIFY set, every install is verified against the host weights\n");
        }
        configure_hash_early_locked(config);
        redraw_ = redraw_env();
        clear_members_locked();
        member_state first;
        first.path     = config->model_path ? config->model_path : "";
        first.weight   = config->model_weight > 0.0f ? config->model_weight : 1.0f;
        first.accepted = true;
        members_.push_back(std::move(first));
        for (uint32_t i = 0; i < config->n_join && config->join_paths != nullptr; ++i) {
            const char * path   = config->join_paths[i];
            const float  weight = config->join_weights != nullptr && config->join_weights[i] > 0.0f ? config->join_weights[i] : 1.0f;
            if (members_.size() >= max_members) {
                GGML_LOG_WARN("expert cache: at most %zu models can share the cache; %s is not cached\n", max_members, path ? path : "(null)");
                continue;
            }
            member_state mb;
            std::string why;
            if (path == nullptr || !declare_member(path, weight, mb, why)) {
                GGML_LOG_WARN("expert cache: %s cannot join the cache: %s\n", path ? path : "(null)", why.c_str());
                mb.free_meta();
                continue;
            }
            GGML_LOG_INFO("expert cache: model %s (%s) is declared to join: %d routed layers x %d experts, weight %.3f\n",
                mb.key.c_str(), mb.path.c_str(), mb.geo.n_routed_layers(), mb.geo.n_experts, (double) mb.weight);
            members_.push_back(std::move(mb));
        }
        state_ = state_t::configured;
        return true;
    }

    // Reads the tensor layout of a model that will join later from its GGUF file(s), without data.
    static bool declare_member(const char * path, float weight, member_state & out, std::string & why) {
        out = member_state();
        out.path   = path;
        out.weight = weight;
        std::vector<std::string> files = { out.path };
        std::vector<const ggml_tensor *> tensors;
        for (size_t f = 0; f < files.size(); ++f) {
            ggml_context * meta = nullptr;
            gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ &meta };
            gguf_context * g = gguf_init_from_file(files[f].c_str(), params);
            if (g == nullptr) {
                why = "cannot read the GGUF metadata of " + files[f];
                return false;
            }
            out.gguf.push_back(g);
            if (meta != nullptr) {
                out.meta.push_back(meta);
            }
            if (f == 0) {
                const int64_t arch = gguf_find_key(g, "general.architecture");
                if (arch < 0 || gguf_get_kv_type(g, arch) != GGUF_TYPE_STRING) {
                    why = "no general.architecture";
                    return false;
                }
                out.identity = gguf_get_val_str(g, arch);
                // the same naming rule as llama_split_path: <prefix>-%05d-of-%05d.gguf
                const int64_t split = gguf_find_key(g, "split.count");
                const int n_split = split >= 0 && gguf_get_kv_type(g, split) == GGUF_TYPE_UINT16 ? (int) gguf_get_val_u16(g, split) : 1;
                if (n_split > 1) {
                    char tail[64];
                    snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", 1, n_split);
                    const size_t n = strlen(tail);
                    if (out.path.size() <= n || out.path.compare(out.path.size() - n, n, tail) != 0) {
                        why = "split file name does not end in " + std::string(tail);
                        return false;
                    }
                    const std::string prefix = out.path.substr(0, out.path.size() - n);
                    for (int i = 2; i <= n_split; ++i) {
                        snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", i, n_split);
                        files.push_back(prefix + tail);
                    }
                }
            }
            for (ggml_tensor * t = meta ? ggml_get_first_tensor(meta) : nullptr; t != nullptr; t = ggml_get_next_tensor(meta, t)) {
                tensors.push_back(t);
            }
        }
        if (!build_geometry_tensors(tensors, out.geo, why)) {
            return false;
        }
        if (out.geo.n_layers == 0) {
            why = "the model has no routed experts";
            return false;
        }
        out.signature = out.identity + "\n" + out.geo.signature();
        out.key       = member_key(out.identity, out.signature);
        return true;
    }

    static std::string member_key(const std::string & identity, const std::string & signature) {
        char hex[32];
        snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) geometry_hash(signature));
        std::string name = identity.empty() ? std::string("model") : identity;
        for (char & c : name) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
            if (!ok) { c = '_'; }
        }
        return name + "-" + hex;
    }

    void clear_members_locked() {
        for (member_state & m : members_) {
            m.free_meta();
        }
        members_.clear();
        members_.reserve(max_members);
        accepting_ = -1;
        joint_     = false;
        n_attached_.store(0, std::memory_order_release);
    }

    // Owned host storage redirects loader writes for exclusive, finite inclusive and Off policies.
    // Unlimited inclusive uses the ordinary host buffer and register_context.
    ggml_backend_buffer_t alloc_context(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (accepting_ >= 0) {
            // a model that joins: only the owned host storage allocates here, inclusive registers
            if (!l1_ || !l1_->owns_host_storage() || state_ != state_t::installed || ctx == nullptr || buft == nullptr ||
                    !ggml_backend_buft_is_host(buft)) {
                return nullptr;
            }
            const int m = join_context_locked(ctx, buft, identity);
            if (m < 0) {
                return nullptr;
            }
            if (tier_) {
                // the loader of the joining model adds files and backing the worker reads; nothing
                // computes while a model loads, and the worker restarts when the load is final
                tier_->stop_worker();
            }
            ggml_backend_buffer_t buffer = exclusive_buffer_create(ctx, buft);
            if (buffer == nullptr) {
                abort_locked("exclusive address reservation of a joined model failed");
            }
            members_[m].exclusive_buffer = buffer;
            return buffer;
        }
        if (state_ != state_t::configured || (cfg_.mode != GGML_EXPERT_MODE_EXCLUSIVE && cfg_.l2_bytes == 0 && cfg_.policy != GGML_EXPERT_POLICY_OFF) ||
                ctx == nullptr || buft == nullptr) {
            return nullptr;
        }
        if (!ggml_backend_buft_is_host(buft)) {
            return nullptr;
        }
        geometry geo;
        std::string reason;
        if (!build_geometry(ctx, geo, reason)) {
            disable_locked(("routed expert tensors rejected: " + reason).c_str());
            return nullptr;
        }
        if (geo.n_layers == 0) {
            return nullptr; // no routed experts in this context
        }
        if (!adopt_context_locked(ctx, buft, identity, geo, /*owns_host =*/ true)) {
            return nullptr;
        }
        if (!prepare_exclusive_locked()) {
            return nullptr;
        }
        ggml_backend_buffer_t buffer = exclusive_buffer_create(ctx, buft);
        if (buffer == nullptr) {
            disable_locked("exclusive address reservation failed");
            return nullptr;
        }
        exclusive_buffer_ = buffer;
        return buffer;
    }

    // Routed by the exclusive buffer type's set_tensor/get_tensor.
    bool buffer_io(const ggml_tensor * tensor, void * data, size_t offset, size_t size, bool write) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!l1_ || !l1_->owns_host_storage()) {
            return false;
        }
        const ggml_tensor * root = tensor;
        while (root->view_src != nullptr) {
            offset += root->view_offs;
            root = root->view_src;
        }
        int layer = -1;
        int kind  = -1;
        if (!resolve(root, layer, kind)) {
            return false;
        }
        if (!l1_->logical_io(layer, kind, data, offset, size, write)) { return false; }
        if (write && digests_enabled_) {
            record_digest_locked(layer, kind, offset, size, data);
        }
        return true;
    }

    bool register_context(ggml_context * ctx, ggml_backend_buffer_t buffer, const char * identity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (accepting_ >= 0) {
            // inclusive: the arenas are built once every member is registered and loaded
            if (state_ != state_t::registered || (l1_ && l1_->owns_host_storage()) || ctx == nullptr || buffer == nullptr ||
                    !ggml_backend_buffer_is_host(buffer)) {
                return false;
            }
            return join_context_locked(ctx, ggml_backend_buffer_get_type(buffer), identity) >= 0;
        }
        if (state_ != state_t::configured || ctx == nullptr || buffer == nullptr) {
            return false;
        }
        geometry geo;
        std::string reason;
        if (!build_geometry(ctx, geo, reason)) {
            disable_locked(("routed expert tensors rejected: " + reason).c_str());
            return false;
        }
        if (geo.n_layers == 0) {
            return false; // no routed experts in this context
        }
        // The model has one weight context per buffer type: the routed layers that fit in VRAM
        // (-ncmoe leaves the last ones there) arrive in a device context and are not this cache's
        // business; only the context in the HIP host buffer type is registered.
        if (!ggml_backend_buffer_is_host(buffer)) {
            GGML_LOG_DEBUG("expert cache: %d routed layers already live in device memory; not cached\n", geo.n_routed_layers());
            return false;
        }
        return adopt_context_locked(ctx, ggml_backend_buffer_get_type(buffer), identity, geo, /*owns_host =*/ false);
    }

    bool finalize() {
        std::lock_guard<std::mutex> lock(mutex_);
        return finalize_locked();
    }

    bool finalize_locked() {
        // The load of every bound member is complete now.
        bool newly_loaded = false;
        for (member_state & m : members_) {
            if (m.bound && !m.loaded) {
                m.loaded     = true;
                newly_loaded = true;
            }
        }
        if (state_ == state_t::installed && newly_loaded && l1_ && l1_->owns_host_storage()) {
            // A model joined an installed owned-host cache: its slices are in their homes.
            return finalize_joined_locked();
        }
        if (state_ != state_t::registered) {
            return false;
        }
        const bool owns_host = l1_ && l1_->owns_host_storage();
        if (!owns_host) {
            // Inclusive: the arenas are planned over every member at once, so the cache waits for
            // the models that were declared to join. Until then lookups miss and the host tensors
            // are read, as without the cache.
            size_t waiting = 0;
            for (const member_state & m : members_) {
                waiting += m.loaded ? 0 : 1;
            }
            if (waiting != 0) {
                GGML_LOG_INFO("expert cache: waiting for %zu more model(s) to join before the arenas are built\n", waiting);
                return true;
            }
        }
        ggml_cuda_set_device(device_);

        if (owns_host) {
            // The delegate buffer is not in the model's buffer list, so the backend's own
            // post-load step for host buffers has not seen it. Do it here, while the loader's
            // writes to it are final, exactly as llama does for the buffers it knows.
            if (!finalize_delegate_buffer_locked(exclusive_buffer_)) {
                abort_locked("delegate buffer finalization failed");
            }
            // The loader has written every slice to its home; register the host arenas as coarse
            // mapped memory so the kernels can read them, and the cache is live.
            if (!host_->map()) {
                abort_locked("host arena registration failed");
            }
            if (tier_ && !start_tier_locked()) {
                return false;
            }
            state_ = state_t::installed;
            std::string reason;
            if (!l1_->verify_current_assignment(reason)) {
                abort_locked(("exclusive assignment is inconsistent: " + reason).c_str());
            }
            verify_all_locked();
            GGML_LOG_INFO("expert cache: owned host install: %zu MiB in VRAM (%zu of %zu slices), "
                          "%zu MiB in the host arena, %d spare slots per class\n",
                l1_->device_bytes()/(1024*1024), resident_slices_locked(),
                size_t(geo_.n_routed_layers())*size_t(geo_.n_experts),
                host_->host_bytes()/(1024*1024), cfg_.spare_slots);
            if (tier_) {
                report_tier_plan_locked("at model load");
            }
            log_members_locked("at model load");
            return true;
        }

        const size_t table_bytes = table_bytes_locked(false);
        const size_t overhead = budget_overhead_locked(false);
        if (cfg_.l1_bytes <= overhead) {
            disable_locked("budget smaller than the cache tables");
            return false;
        }
        const size_t remaining = cfg_.l1_bytes - overhead;

        // The bank that seeds the load-time plan. Its stored profile decides the frozen capacities.
        std::vector<uint64_t> scores;
        const uint64_t * counts = seed_counts_locked(scores);

        placement_inputs in;
        in.geo          = &geo_;
        in.counts       = counts;
        in.budget_bytes = remaining;
        in.exclusive    = false;
        placement initial = plan_placement(in);
        if (!any_capacity(initial.capacities)) {
            disable_locked("budget holds no expert slice");
            return false;
        }
        capacities_ = initial.capacities;
        if (carve_stage_locked()) {
            // the static plan keeps the capacity the staging slots leave it
            placement_inputs fixed = in;
            fixed.fixed_capacities = &capacities_;
            initial = plan_placement(fixed);
        }

        if (!allocate_profiler_locked()) { return false; }
        l1_.reset(new l1_arena(geo_));
        redraw_prepare_locked();
        if (!l1_->allocate(capacities_, device_, 0, stage_class_slots_.empty() ? nullptr : &stage_class_slots_)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
        redraw_started_locked(counts == nullptr);
        allocate_stage_locked();
        l1_install_stats stats;
        if (!l1_->install(initial.selected, /*retain=*/false, stats)) {
            disable_locked("initial install failed");
            return false;
        }
        // One full slice read back and compared with its host source: a mismatch means the stride
        // or source assumptions are wrong and the kernels would read wrong weights. That is fatal.
        for (int l = geo_.n_layers - 1; l >= 0; --l) {
            if (initial.selected[l].empty()) {
                continue;
            }
            if (!l1_->verify_slice(l, 2, initial.selected[l][0])) {
                GGML_ABORT("expert cache: H2D verification of layer %d expert %d failed", l, initial.selected[l][0]);
            }
            break;
        }
        state_ = state_t::installed;
        verify_all_locked();
        if (counts != nullptr) { log_plan_locked(initial.stats); }
        GGML_LOG_INFO("expert cache: installed %zu MiB in VRAM (%zu slices, %.1f ms); tables and profiler %zu KiB%s\n",
            l1_->device_bytes()/(1024*1024), stats.copied, stats.ms,
            (table_bytes + (profiler_ ? profiler_->device_bytes() : 0))/1024, models_text_locked().c_str());
        log_members_locked("at model load");
        return true;
    }

    void release(ggml_context * ctx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == state_t::unconfigured) {
            return;
        }
        // A joint cache lives until the last of its models is freed: the others may still compute.
        bool member = false;
        bool others = false;
        for (member_state & m : members_) {
            if (m.bound && m.ctx == ctx) {
                m.released = true;
                member     = true;
            }
        }
        for (const member_state & m : members_) {
            others |= m.bound && !m.released;
        }
        if (!member && ctx_ != nullptr && ctx != ctx_) {
            return;
        }
        if (others) {
            return;
        }
        state_ = state_t::unconfigured;
        ggml_cuda_set_device(device_ >= 0 ? device_ : 0);
        plans_.clear();
        next_plan_id_   = 0;
        installed_plan_ = GGML_EXPERT_PLAN_NONE;
        banks_.clear();
        n_banks_.store(0, std::memory_order_release);
        // the worker must stop before the arenas it reads into go away
        tier_.reset();
        free_stage_locked();
        l1_.reset();
        host_.reset();
        profiler_.reset();
        exclusive_buffer_ = nullptr;
        digests_.clear();
        digest_known_.clear();
        digests_enabled_ = false;
        geo_ = geometry();
        ctx_ = nullptr;
        clear_members_locked();
        device_ = -1;
        disabled_reason_ = "";
        capacities_.clear();
        redraw_on_ = false;
        redraw_gate_ = redraw_gate();
        redraw_epoch_ = 0;
        redraw_max_static_.clear();
        redraw_unequal_ = false;
    }

    bool status(ggml_expert_status * out) {
        if (out == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        *out = {};
        out->configured      = state_ != state_t::unconfigured;
        out->registered      = state_ == state_t::registered || state_ == state_t::installed;
        out->installed       = state_ == state_t::installed;
        out->disabled        = state_ == state_t::disabled;
        out->disabled_reason = disabled_reason_.c_str();
        out->device_bytes    = (l1_ ? l1_->device_bytes() : 0) + (profiler_ ? profiler_->device_bytes() : 0) +
                               (host_ ? host_->device_bytes() : 0);
        out->host_bytes      = (host_ ? host_->host_bytes() : 0) + (tier_ ? tier_->host_bytes() : 0);
        out->n_banks         = (uint32_t) banks_.size();
        return true;
    }

    // ---- banks and plans ------------------------------------------------------------------

    bool bank_open(const char * label, bool prompt_bank, ggml_expert_bank_id * out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!profiling_enabled() || state_ != state_t::installed || label == nullptr || label[0] == '\0' || out == nullptr) {
            return false;
        }
        return open_bank_locked(label, prompt_bank, out);
    }

    // Mark and discard both drop what the bank counted so far.
    bool bank_zero(ggml_expert_bank_id bank) {
        std::lock_guard<std::mutex> lock(mutex_);
        return bank_ok_locked(bank) && profiler_->zero_bank(bank);
    }

    bool bank_commit(ggml_expert_bank_id bank, const ggml_expert_record * record, ggml_expert_plan_id * out_plan) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!bank_ok_locked(bank) || record == nullptr || out_plan == nullptr) {
            return false;
        }
        bank_state & b = banks_[bank];
        if (!b.any_usable()) {
            return false;
        }
        std::vector<uint64_t> delta;
        if (!profiler_->read_bank(bank, delta, /*zero_after=*/true)) {
            return false;
        }
        if (stage_dev_ != nullptr) {
            // the device is synchronized by the read-back above
            log_stage_counters_locked(b.label);
        }
        if (tier_ && !early_decl_.empty() && !b.prompt) {
            log_early_counters_locked(b.label);
        }
        uint64_t total = 0;
        for (uint64_t v : delta) {
            total += v;
        }
        profile_record_meta meta;
        meta.timestamp_s   = now_s();
        meta.request_count = record->request_count;
        meta.input_tokens  = record->input_tokens;
        meta.output_tokens = record->output_tokens;
        meta.bank_tokens   = record->bank_tokens;
        const auto save_start = std::chrono::steady_clock::now();
        // every model stores its own part of the interval; a model that routed nothing in it gets
        // no record, so its window is not diluted by intervals in which it did not run
        for (size_t m = 0; m < b.stores.size(); ++m) {
            if (!b.stores[m] || !b.usable[m]) {
                continue;
            }
            const member_state & mb = members_[m];
            const size_t begin = size_t(mb.offset)*geo_.n_experts;
            const size_t end   = begin + size_t(mb.geo.n_layers)*geo_.n_experts;
            std::vector<uint64_t> part;
            if (!joint_) {
                part = delta;
            } else {
                part.assign(delta.begin() + begin, delta.begin() + std::min(end, delta.size()));
            }
            uint64_t part_total = 0;
            for (uint64_t v : part) {
                part_total += v;
            }
            if (part_total != 0 && !b.stores[m]->checkpoint(part, meta)) {
                GGML_LOG_WARN("expert cache: bank '%s' of model %s failed to store a record: %s\n", b.label.c_str(),
                    mb.key.c_str(), b.stores[m]->last_error().c_str());
                return false;
            }
        }
        const double save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - save_start).count();
        ++b.commits;
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) { report_round_locked(b.label, b.commits, *record, delta, save_ms); }
        std::vector<uint64_t> scores;
        std::vector<uint8_t> keep;
        if (joint_) {
            std::vector<const profile_store *> sources = joint_sources_locked(b);
            if (b.prompt) {
                keep_members_locked(b, sources, keep);
            }
            assemble_joint_scores_locked(sources, scores, b.label.c_str());
        } else {
            joint_scores_locked(b, scores);
        }
        placement_inputs in;
        in.geo              = &geo_;
        in.counts           = b.total_selections() != 0 ? scores.data() : nullptr;
        in.budget_bytes     = SIZE_MAX;
        in.exclusive        = l1_ && l1_->owns_host_storage();
        in.fixed_capacities = &capacities_;
        placement next = plan_placement(in);

        plan_state plan;
        plan.bank     = bank;
        plan.selected = std::move(next.selected);
        plan.stats    = next.stats;
        plan.keep     = std::move(keep);
        plan.epoch    = redraw_epoch_;
        if ((tier_ || redraw_on_) && b.total_selections() != 0) {
            plan.scores = scores;
        }
        const ggml_expert_plan_id id = next_plan_id_++;
        plans_[id] = std::move(plan);
        // one live plan per bank: the plan this one replaces is unreachable unless it is installed
        if (b.latest_plan != GGML_EXPERT_PLAN_NONE && b.latest_plan != installed_plan_) {
            plans_.erase(b.latest_plan);
        }
        b.latest_plan = id;
        *out_plan = id;
        if (cfg_.log_mask & GGML_EXPERT_LOG_PROFILE) {
            GGML_LOG_INFO("expert cache: bank '%s' commit %llu: %llu selections, %zu records, plan %u selection-hit=%.2f%% byte-hit=%.2f%%\n",
                b.label.c_str(), (unsigned long long) b.commits, (unsigned long long) total, b.records(),
                id, 100.0*plans_[id].stats.selection_hit(), 100.0*plans_[id].stats.byte_hit());
        }
        return true;
    }

    bool plan_install(ggml_expert_plan_id id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != state_t::installed) {
            return false;
        }
        const auto it = plans_.find(id);
        if (it == plans_.end()) {
            GGML_LOG_WARN("expert cache: plan %u is stale or unknown; not installed\n", id);
            return false;
        }
        const plan_state & committed = it->second;
        if (banks_[committed.bank].latest_plan != id) {
            GGML_LOG_WARN("expert cache: plan %u of bank '%s' is stale; not installed\n", id, banks_[committed.bank].label.c_str());
            return false;
        }
        if (cfg_.freeze || cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE) {
            GGML_LOG_INFO("expert cache: frozen; plan %u not installed\n", id);
            return false;
        }
        // a plan committed before a size-class redraw is placed again in the current capacities
        plan_state replaced;
        if (committed.epoch != redraw_epoch_) {
            replaced = committed;
            replan_locked(replaced);
        }
        const plan_state & stored = committed.epoch != redraw_epoch_ ? replaced : committed;
        // A model that does not run in the plan's phase keeps the VRAM slices it has now. Its size
        // classes are its own (keep_members_locked checks that), so the class capacities still hold.
        plan_state patched;
        bool kept = false;
        for (size_t m = 0; m < stored.keep.size() && m < members_.size(); ++m) {
            if (!stored.keep[m]) {
                continue;
            }
            if (!kept) {
                patched = stored;
                kept    = true;
            }
            const member_state & mb = members_[m];
            for (int l = mb.offset; l < mb.offset + mb.geo.n_layers && l < geo_.n_layers; ++l) {
                patched.selected[(size_t) l] = l1_->selected()[(size_t) l];
            }
        }
        const plan_state & plan = kept ? patched : stored;
        // The size-class redraw's gate: one decision per generation install (expert-redraw.h).
        std::vector<int> redraw_to;
        if (redraw_on_ && !banks_[plan.bank].prompt) {
            redraw_to = redraw_evaluate_locked(id, plan);
        }
        // Installing what is already in the arena costs nothing: no drain, no copies, one line.
        if (redraw_to.empty() && !tier_ && plan.selected == l1_->selected()) {
            finish_plan_locked(id);
            if (cfg_.log_mask & GGML_EXPERT_LOG_INSTALL) {
                GGML_LOG_INFO("expert cache: plan %u of bank '%s' is already installed; nothing to do\n",
                    id, banks_[plan.bank].label.c_str());
            }
            return true;
        }
        ggml_cuda_set_device(device_);
        // The caller synchronized its context; the device-wide drain covers every stream that could
        // still read the arena or the tables.
        CUDA_CHECK(cudaDeviceSynchronize());
        // the install republishes the slot tables: the staging entries go, and the staging forgets its
        // experts (the next hint copies again; an install is rare next to the hints)
        for (std::vector<int32_t> & owners : stage_owner_) { stage_forget(owners); }
        if (!redraw_to.empty()) {
            return redraw_locked(id, plan, redraw_to);
        }
        return install_once_locked(id, plan);
    }

    bool install_once_locked(ggml_expert_plan_id id, const plan_state & plan) {
        if (tier_) {
            return install_tier_locked(id, plan, /*seed =*/ false);
        }
        return install_plain_locked(id, plan);
    }

    // An install without the SSD tier.
    bool install_plain_locked(ggml_expert_plan_id id, const plan_state & plan) {
        l1_install_stats stats;
        if (!l1_->install(plan.selected, cfg_.delta_install, stats)) {
            disable_locked("install failed");
            return false;
        }
        if (l1_->owns_host_storage()) {
            GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': mode=%s retained=%zu exchanged=%zu "
                          "h2d_bytes=%zu MiB d2h_bytes=%zu MiB in %.1f ms\n",
                id, banks_[plan.bank].label.c_str(), mode_name(), stats.retained, stats.copied,
                stats.bytes/(1024*1024), stats.d2h_bytes/(1024*1024), stats.ms);
            std::string reason;
            if (!l1_->verify_current_assignment(reason)) {
                GGML_ABORT("expert cache: exclusive assignment is inconsistent after a swap: %s", reason.c_str());
            }
        } else {
            GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': retained=%zu copied=%zu bytes=%zu MiB in %.1f ms\n",
                id, banks_[plan.bank].label.c_str(), stats.retained, stats.copied, stats.bytes/(1024*1024), stats.ms);
        }
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) {
            GGML_LOG_INFO("expert_metrics {\"kind\":\"install\",\"bank\":\"%s\",\"plan\":%u,\"install_ms\":%.6f,"
                          "\"h2d_bytes\":%zu,\"d2h_bytes\":%zu,\"ssd_bytes\":0}\n",
                banks_[plan.bank].label.c_str(), id, stats.ms, stats.bytes, stats.d2h_bytes);
        }
        // the plan that was installed until now is only reachable while it is a bank's newest one
        finish_plan_locked(id);
        verify_all_locked();
        if (joint_) { log_members_locked("after install"); }
        return true;
    }

    bool profile_select(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank) {
        if (state_ != state_t::installed || cfg_.policy == GGML_EXPERT_POLICY_OFF || backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
        if (cuda_ctx->device != device_ || !profiler_) {
            return false;
        }
        const int member = member_of(cuda_ctx);
        if (member < 0) {
            return false;
        }
        ggml_cuda_set_device(device_);
        if (tier_ && bank < n_banks_.load(std::memory_order_acquire)) {
            tier_->set_phase(bank_prompt_[bank].load(std::memory_order_acquire) != 0);
        }
        return profiler_->select(row_begin, row_end, bank, cuda_ctx->stream(), (uint32_t) member);
    }

    // ---- hot path ---------------------------------------------------------------------------

    ggml_cuda_expert_lookup lookup(const ggml_tensor * tensor) const noexcept {
        if (state_ != state_t::installed || tensor == nullptr) {
            return {};
        }
        int layer = -1;
        int kind  = -1;
        if (!resolve(tensor, layer, kind) || ggml_cuda_get_device() != device_) {
            return {};
        }
        return l1_->lookup(layer, kind);
    }

    // Joint layer and kind of a routed expert tensor of one of the cached models. With one model
    // the joint layer is the tensor's own layer. Hot path: no lock; the member table is fixed
    // before any compute of the cached models.
    bool resolve(const ggml_tensor * tensor, int & layer, int & kind) const noexcept {
        if (!parse_expert_tensor_name(tensor->name, layer, kind)) {
            return false;
        }
        if (!joint_) {
            return layer < geo_.n_layers && geo_.tensors[layer][kind] == tensor;
        }
        for (const member_state & m : members_) {
            if (layer < m.geo.n_layers && m.offset + layer < geo_.n_layers && geo_.tensors[m.offset + layer][kind] == tensor) {
                layer += m.offset;
                return true;
            }
        }
        return false;
    }

    // The member whose model runs on this compute context, -1 when none was attached.
    int member_of(const void * backend_ctx) const noexcept {
        const int n = n_attached_.load(std::memory_order_acquire);
        for (int i = 0; i < n; ++i) {
            if (attached_ctx_[i].load(std::memory_order_relaxed) == backend_ctx) {
                return attached_member_[i];
            }
        }
        return -1;
    }

    // `weights` null detaches the context (it is being freed); its slot can then be reused.
    bool attach_backend(ggml_backend_t backend, ggml_context * weights) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        const void * key = backend->context;
        const int n = n_attached_.load(std::memory_order_acquire);
        if (weights == nullptr) {
            for (int i = 0; i < n; ++i) {
                if (attached_ctx_[i].load(std::memory_order_relaxed) == key) {
                    attached_ctx_[i].store(nullptr, std::memory_order_release);
                }
            }
            return true;
        }
        if (state_ == state_t::unconfigured) {
            return false;
        }
        int member = -1;
        for (size_t m = 0; m < members_.size(); ++m) {
            if (members_[m].bound && members_[m].ctx == weights) {
                member = (int) m;
            }
        }
        if (member < 0) {
            return false;
        }
        int slot = -1;
        for (int i = 0; i < n && slot < 0; ++i) {
            if (attached_ctx_[i].load(std::memory_order_relaxed) == key) {
                slot = i;
            }
        }
        for (int i = 0; i < n && slot < 0; ++i) {
            if (attached_ctx_[i].load(std::memory_order_relaxed) == nullptr) {
                slot = i;
            }
        }
        if (slot < 0) {
            if (n >= max_attached) {
                GGML_LOG_WARN("expert cache: more than %d compute contexts; the new one is not profiled\n", max_attached);
                return false;
            }
            slot = n;
        }
        attached_member_[slot] = member;
        attached_ctx_[slot].store(key, std::memory_order_release);
        if (slot == n) {
            n_attached_.store(n + 1, std::memory_order_release);
        }
        if (joint_) {
            GGML_LOG_INFO("expert cache: compute context %d runs model %d (%s)\n", slot, member, members_[member].key.c_str());
        }
        return true;
    }

    // ---- early routes (expert-hash-early.h) ------------------------------------------------------

    // The experts that the next graph on `backend` selects in early-route layers. Called by the
    // compute thread before it builds or launches that graph; never waits for the device.
    bool route_hint(ggml_backend_t backend, const ggml_expert_route_hint * hints, int32_t n_hints) {
        if (!hash_mode_.any() || early_decl_.empty()) {
            return false;
        }
        if (state_ != state_t::installed || backend == nullptr || !ggml_backend_is_cuda(backend) || hints == nullptr ||
                n_hints <= 0) {
            return true;
        }
        if (!(hash_mode_.ssd && tier_ && tier_->early()) && stage_dev_ == nullptr) {
            return false;   // installed, and neither an SSD tier to read early nor staging slots
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
        if (cuda_ctx->device != device_ || member_of(cuda_ctx) != 0) {
            return true;   // only the model that declared the layers
        }
        // generation and verification batches; prompt ubatches have the staged tier service
        const int rows = hints[0].rows;
        if (rows <= 0 || (uint64_t) rows > std::max<uint64_t>(cfg_.l2_decode_rows, 1)) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != state_t::installed) {
            return true;
        }
        const std::vector<uint8_t> early = early_joint_locked();
        const int offset = members_.empty() ? 0 : members_[0].offset;
        std::vector<std::pair<int, std::vector<int>>> layers;
        for (int32_t i = 0; i < n_hints; ++i) {
            const ggml_expert_route_hint & h = hints[i];
            const int joint = offset + h.layer;
            if (h.layer < 0 || joint >= geo_.n_layers || !early[(size_t) joint]) {
                continue;
            }
            layers.push_back({joint, hint_distinct_ids(h.ids, h.n_ids, geo_.n_experts)});
        }
        std::sort(layers.begin(), layers.end(),
            [](const std::pair<int, std::vector<int>> & a, const std::pair<int, std::vector<int>> & b) { return a.first < b.first; });
        if (layers.empty()) {
            return true;
        }
        if (hash_mode_.ssd && tier_ && tier_->early()) {
            std::vector<l2_early_layer> job;
            for (const auto & l : layers) {
                l2_early_layer item;
                item.layer     = l.first;
                item.published = tier_->published(l.first);
                item.ids       = l.second;
                job.push_back(std::move(item));
            }
            tier_->post_early(std::move(job));
        }
        if (stage_dev_ != nullptr) {
            stage_hint_locked(*cuda_ctx, layers);
        }
        return true;
    }

    void profile_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids) {
        if (state_ != state_t::installed || ids == nullptr || ctx.device != device_) {
            return;
        }
        static constexpr const char prefix[] = "ffn_moe_topk-";
        if (strncmp(ids->name, prefix, sizeof(prefix) - 1) != 0) {
            return;
        }
        char * end = nullptr;
        const long layer = strtol(ids->name + sizeof(prefix) - 1, &end, 10);
        if (end == ids->name + sizeof(prefix) - 1 || *end != '\0' || layer < 0) {
            return;
        }
        if (ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t)) {
            return;
        }
        // The model is the one whose context computes: the layer names of two models overlap, so
        // the name alone would count one model's selections into the other's layers. A context
        // that runs no cached model (a draft that is not in the cache) is never profiled.
        const int member = member_of(&ctx);
        long joint = -1;
        if (member >= 0 && layer < members_[member].geo.n_layers) {
            joint = members_[member].offset + layer;
        }
        if (member < 0 || joint < 0 || joint >= geo_.n_layers || geo_.layer_class[joint] < 0) {
            return;
        }
        if (stage_dev_ != nullptr && stage_index_[(size_t) joint] >= 0) {
            // VRAM staging of an early-route layer: its matmuls read the table only after the
            // staging of this graph's hint (if any) is final
            const int idx = stage_index_[(size_t) joint];
            const ggml_cuda_kernel_launch_params wait_launch(dim3(1), dim3(64), 0, ctx.stream());
            ggml_cuda_kernel_launch(stage_wait_kernel, wait_launch, stage_dev_ + idx, (const int32_t *) ids->data,
                (int) ids->ne[1], (int) ids->ne[0], (int) (ids->nb[1]/sizeof(int32_t)), geo_.n_experts,
                (const int32_t *) l1_->lookup((int) joint, 0).slots, stage_first_[(size_t) joint],
                stage_first_[(size_t) joint] + stage_slots_);
        }
        if (profiler_) {
        launch_profile_ids((const int32_t *) ids->data, (int) ids->ne[1], (int) (ids->nb[1]/sizeof(int32_t)), (int) ids->ne[0],
            geo_.n_experts, profiler_->counts_base() + size_t(joint)*geo_.n_experts, profiler_->n_counts(), profiler_->n_banks(),
            profiler_->selection((uint32_t) member), ctx.stream());
        }
        // The SSD tier needs the same ids: this is the one site that sees both the fused and the
        // unfused router, and it is ordered before the layer's MUL_MAT_ID on the same stream.
        tier_route(ctx, (int) joint, ids);
    }

    void before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_before_read(ctx, src0);
    }

    void layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_layer_done(ctx, src0);
    }

    // The member whose joint layer range holds `layer`; a layer past the joint geometry counts as
    // the first model's.
    int member_of_layer(int layer) const {
        for (size_t m = 0; m < members_.size(); ++m) {
            if (layer >= members_[m].offset && layer < members_[m].offset + members_[m].geo.n_layers) {
                return (int) m;
            }
        }
        return 0;
    }

    // ---- the SSD tier -------------------------------------------------------------------------

    // Called by the loader for every routed expert tensor in the exclusive buffer type.
    bool tier_set_backing(const ggml_tensor * tensor, int file_index, const char * path, uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        int layer = -1;
        int kind  = -1;
        if (!tier_ || tensor == nullptr || !resolve(tensor, layer, kind)) {
            return false;
        }
        tier_->set_backing(layer, kind, file_index, path, offset);
        return true;
    }

    // False for an expert whose bytes stay in the file, so the loader skips them.
    bool tier_load_wanted(const ggml_tensor * tensor, int expert) {
        std::lock_guard<std::mutex> lock(mutex_);
        int layer = -1;
        int kind  = -1;
        if (!tier_ || tensor == nullptr || !resolve(tensor, layer, kind) || expert < 0 || expert >= geo_.n_experts) {
            return true;
        }
        return tier_->wanted(layer, expert);
    }

    void tier_route(ggml_backend_cuda_context & ctx, int layer, const ggml_tensor * ids) {
        if (tier_ && state_ == state_t::installed && ctx.device == device_) {
            tier_->publish_and_wait(layer, ids, ctx.stream());
        }
    }

    // Staged service: the kernel about to read `src0` waits for that kind's slices only.
    void tier_before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        if (!tier_ || !tier_->staged() || !tier_->any_ssd() || state_ != state_t::installed || src0 == nullptr ||
                ctx.device != device_) {
            return;
        }
        int layer = -1;
        int kind  = -1;
        if (resolve(src0, layer, kind)) {
            tier_->wait_kind(layer, kind, ctx.stream());
        }
    }

    void tier_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        if (!tier_ || state_ != state_t::installed || src0 == nullptr || ctx.device != device_) {
            return;
        }
        int layer = -1;
        int kind  = -1;
        if (resolve(src0, layer, kind) && kind == 2) {
            tier_->mark_done(layer, ctx.stream());
        }
    }

private:
    // Shared tail of alloc_context and register_context: find the device of the buffer type and
    // adopt the routed-expert context.
    bool adopt_context_locked(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity,
            geometry & geo, bool owns_host) {
        const int device = ggml_backend_cuda_dev_index(ggml_backend_buft_get_device(buft));
        if (device < 0) {
            GGML_LOG_INFO("expert cache: routed experts are in a CPU buffer, not the HIP host buffer type; not cached\n");
            return false;
        }
        member_state & first = members_[0];
        first.identity  = identity ? identity : "";
        first.geo       = geo;
        first.signature = first.identity + "\n" + first.geo.signature();
        first.key       = member_key(first.identity, first.signature);
        first.ctx       = ctx;
        first.bound     = true;
        // Declared members must route as many experts as this one; the others cannot join.
        for (size_t m = members_.size(); m-- > 1; ) {
            if (members_[m].geo.n_experts != geo.n_experts) {
                GGML_LOG_WARN("expert cache: model %s routes %d experts, this model %d; it is not cached\n",
                    members_[m].key.c_str(), members_[m].geo.n_experts, geo.n_experts);
                members_[m].free_meta();
                members_.erase(members_.begin() + m);
            }
        }
        joint_ = members_.size() > 1;
        if (joint_) {
            std::vector<const geometry *> parts;
            for (const member_state & m : members_) {
                parts.push_back(&m.geo);
            }
            std::vector<int> offsets;
            std::string reason;
            if (!concat_geometry(parts, geo_, offsets, reason)) {
                GGML_LOG_ERROR("expert cache: joint geometry rejected: %s\n", reason.c_str());
                return false;
            }
            for (size_t m = 0; m < members_.size(); ++m) {
                members_[m].offset = offsets[m];
            }
        } else {
            geo_ = std::move(geo);
        }
        ctx_       = ctx;
        device_    = device;
        identity_  = first.identity;
        signature_ = identity_ + "\n" + geo_.signature();
        state_     = state_t::registered;
        GGML_LOG_INFO("expert cache: registered %s%s: %d routed layers x %d experts, %zu size classes, device %d\n",
            identity_.c_str(), owns_host ? " for owned host storage" : "", geo_.n_routed_layers(),
            geo_.n_experts, geo_.class_bytes.size(), device_);
        if (joint_) {
            for (size_t m = 0; m < members_.size(); ++m) {
                const member_state & mb = members_[m];
                GGML_LOG_INFO("expert cache: model %zu %s: joint layers %d..%d, weight %.3f%s\n", m, mb.key.c_str(),
                    mb.offset, mb.offset + mb.geo.n_layers - 1, (double) mb.weight, mb.bound ? "" : ", joins later");
            }
            for (size_t cls = 0; cls < geo_.class_bytes.size(); ++cls) {
                GGML_LOG_INFO("expert cache: size class %zu: %d layers, %zu KiB per expert\n",
                    cls, geo_.class_layers[cls], geo_.class_total_bytes((int) cls)/1024);
            }
        }
        return true;
    }

    // A declared model's weight context arrives: its layout must be the one read from its file.
    // Returns the member index, or -1 when the context holds no routed experts (the next context of
    // the same model may) or does not match (the model is then not cached).
    int join_context_locked(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity) {
        geometry geo;
        std::string reason;
        if (!build_geometry(ctx, geo, reason)) {
            GGML_LOG_WARN("expert cache: routed expert tensors of a joining model rejected: %s\n", reason.c_str());
            drop_member_locked(accepting_);
            return -1;
        }
        if (geo.n_layers == 0) {
            return -1;
        }
        const int m = accepting_;
        member_state & mb = members_[m];
        const int device = ggml_backend_cuda_dev_index(ggml_backend_buft_get_device(buft));
        // the layout decides; the architecture name of the key is the one the file declares
        GGML_UNUSED(identity);
        if (device != device_) {
            GGML_LOG_WARN("expert cache: model %s puts its routed experts on device %d, the cache is on %d; it is not cached\n",
                mb.key.c_str(), device, device_);
            drop_member_locked(m);
            return -1;
        }
        if (geo.signature() != mb.geo.signature()) {
            GGML_LOG_WARN("expert cache: model %s: the routed expert tensors of the loaded model differ from the layout read "
                          "from %s (all of its routed experts must be in host memory); it is not cached\n",
                mb.key.c_str(), mb.path.c_str());
            drop_member_locked(m);
            return -1;
        }
        for (int l = 0; l < geo.n_layers; ++l) {
            for (int k = 0; k < geometry::n_kinds; ++k) {
                geo_.tensors[mb.offset + l][k] = geo.tensors[l][k];
            }
        }
        mb.geo   = std::move(geo);
        mb.ctx   = ctx;
        mb.bound = true;
        mb.free_meta();
        accepting_ = -1;
        GGML_LOG_INFO("expert cache: model %d %s bound: joint layers %d..%d%s\n", m, mb.key.c_str(), mb.offset,
            mb.offset + mb.geo.n_layers - 1, l1_ && l1_->owns_host_storage() ? ", its slices are written to their homes" : "");
        return m;
    }

    // A member that cannot join. Before the arenas exist it leaves the joint geometry; after that
    // (owned host storage) its layers keep their slots, unused, and it is only marked.
    void drop_member_locked(int m) {
        accepting_ = -1;
        if (m <= 0 || m >= (int) members_.size()) {
            return;
        }
        members_[m].dropped = true;
        if (l1_ || host_) {
            // the joint geometry still points at its file metadata; freed with the cache
            return;
        }
        members_[m].free_meta();
        members_.erase(members_.begin() + m);
        joint_ = members_.size() > 1;
        std::vector<const geometry *> parts;
        for (const member_state & mb : members_) {
            parts.push_back(&mb.geo);
        }
        std::vector<int> offsets;
        std::string reason;
        if (joint_) {
            if (!concat_geometry(parts, geo_, offsets, reason)) {
                disable_locked(("joint geometry: " + reason).c_str());
                return;
            }
            for (size_t i = 0; i < members_.size(); ++i) {
                members_[i].offset = offsets[i];
            }
        } else {
            geo_ = members_[0].geo;
            members_[0].offset = 0;
        }
        signature_ = identity_ + "\n" + geo_.signature();
        // an inclusive cache that waited for this model builds its arenas now
        bool all_loaded = true;
        for (const member_state & mb : members_) {
            all_loaded &= mb.loaded;
        }
        if (state_ == state_t::registered && all_loaded) {
            finalize_locked();
        }
    }

    // A model joined the installed owned-host cache and its loader has written every slice.
    bool finalize_joined_locked() {
        ggml_cuda_set_device(device_);
        for (member_state & m : members_) {
            if (m.bound && m.loaded && m.exclusive_buffer != nullptr && !finalize_delegate_buffer_locked(m.exclusive_buffer)) {
                abort_locked("delegate buffer finalization of a joined model failed");
            }
        }
        // the device must see the loader's writes to the mapped host arena
        CUDA_CHECK(cudaDeviceSynchronize());
        std::string reason;
        if (tier_) {
            // the joined model's files: open them and check its backing like at the first load,
            // then let the worker (stopped when the model joined) serve every model again
            const std::vector<uint8_t> layers = member_layers_locked(true);
            if (!tier_->open_files(reason, &layers)) {
                abort_locked(("SSD tier: " + reason).c_str());
            }
            verify_tier_backing_locked(&layers);
            for (member_state & m : members_) {
                m.tier_ready = m.tier_ready || (m.bound && !m.dropped);
            }
            // the joined model's tensors have their sector shifts now, and with the class layout
            // its host residents are addressed at them
            if (!tier_->worker_running()) { tier_->refresh_addresses(); }
            tier_->start_worker(); // no-op when it runs
            report_tier_plan_locked("after a model joined");
        }
        if (!l1_->verify_current_assignment(reason)) {
            abort_locked(("exclusive assignment is inconsistent: " + reason).c_str());
        }
        verify_all_locked();
        log_members_locked("after a model joined");
        if (redraw_on_) {
            // the class mix changed: the next installs may redraw on their first proposal
            redraw_mark_fresh(redraw_gate_, redraw_);
        }
        return true;
    }

    // " models=<key>*<weight>,..." for the plan and install lines.
    std::string models_text_locked() const {
        std::string out = " models=";
        for (size_t m = 0; m < members_.size(); ++m) {
            char w[32];
            snprintf(w, sizeof(w), "*%.3g", (double) members_[m].weight);
            out += (m ? "," : "") + members_[m].key + w;
        }
        return out;
    }

    // Per model: experts and bytes in VRAM and in host memory (and in the file with the SSD tier).
    void log_members_locked(const char * what) const {
        if (!l1_ || members_.empty()) {
            return;
        }
        const std::vector<std::vector<int32_t>> & selected = l1_->selected();
        for (size_t m = 0; m < members_.size(); ++m) {
            const member_state & mb = members_[m];
            size_t n_l1 = 0, b_l1 = 0, n_host = 0, b_host = 0, n_file = 0, b_file = 0;
            for (int l = mb.offset; l < mb.offset + mb.geo.n_layers && l < geo_.n_layers; ++l) {
                const int cls = geo_.layer_class[l];
                if (cls < 0) {
                    continue;
                }
                const size_t bytes = geo_.class_total_bytes(cls);
                std::vector<uint8_t> in_l1(geo_.n_experts, 0);
                if ((size_t) l < selected.size()) {
                    for (int32_t e : selected[l]) { in_l1[e] = 1; }
                }
                for (int e = 0; e < geo_.n_experts; ++e) {
                    if (in_l1[e]) { n_l1++; b_l1 += bytes; continue; }
                    if (tier_ && !tier_->locations()[l][e].resident()) { n_file++; b_file += bytes; continue; }
                    n_host++; b_host += bytes;
                }
            }
            GGML_LOG_INFO("expert cache: model %zu %s %s: L1 %zu experts / %zu MiB, host %zu experts / %zu MiB%s%s\n",
                m, mb.key.c_str(), what, n_l1, b_l1/(1024*1024), n_host, b_host/(1024*1024),
                tier_ ? (", file " + std::to_string(n_file) + " experts / " + std::to_string(b_file/(1024*1024)) + " MiB").c_str() : "",
                mb.dropped ? " (not cached, slots unused)" : (mb.bound ? "" : " (joins later)"));
        }
    }

    void finish_plan_locked(ggml_expert_plan_id id) {
        if (installed_plan_ != GGML_EXPERT_PLAN_NONE && installed_plan_ != id) {
            const auto old = plans_.find(installed_plan_);
            if (old != plans_.end() && banks_[old->second.bank].latest_plan != installed_plan_) {
                plans_.erase(old);
            }
        }
        installed_plan_ = id;
    }

    // ---- size-class redraw of the VRAM tier (expert-redraw.h) -------------------------------------

    void set_capacities_locked(const std::vector<int> & caps) {
        capacities_ = caps;
        l1_->set_static_capacities(caps);
    }

    // A plan committed before a redraw: the same scores placed in the current capacities.
    void replan_locked(plan_state & plan) const {
        placement_inputs in;
        in.geo              = &geo_;
        in.counts           = plan.scores.empty() ? nullptr : plan.scores.data();
        in.budget_bytes     = SIZE_MAX;
        in.exclusive        = l1_ && l1_->owns_host_storage();
        in.fixed_capacities = &capacities_;
        placement next = plan_placement(in);
        plan.selected = std::move(next.selected);
        plan.stats    = next.stats;
        plan.epoch    = redraw_epoch_;
    }

    // Before the arena is allocated: with a redraw condition set and a mode the redraw supports,
    // the arena is built on HIP virtual memory, with address ranges for the largest static capacity
    // each class can reach within the current static bytes.
    void redraw_prepare_locked() {
        redraw_on_ = false;
        redraw_max_static_.clear();
        if (!redraw_.enabled) {
            return;
        }
        const char * why = nullptr;
        if (cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE || cfg_.freeze) {
            why = "no plan installs (frozen or not adaptive)";
        } else if (cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE && cfg_.l2_bytes == 0) {
            why = "exclusive mode with an unlimited host tier keeps one host arena per class that the L1 split sizes; "
                  "the redraw needs a finite L2 or inclusive mode";
        }
        if (why != nullptr) {
            GGML_LOG_WARN("expert cache: L1 redraw requested but off: %s\n", why);
            return;
        }
        const size_t budget = redraw_static_bytes(geo_, capacities_);
        redraw_max_static_.assign(geo_.class_bytes.size(), 0);
        for (size_t c = 0; c < redraw_max_static_.size(); ++c) {
            const size_t bytes = geo_.class_total_bytes((int) c);
            const size_t n = bytes == 0 ? 0 : budget/bytes;
            redraw_max_static_[c] = (int) std::min<size_t>(n, size_t(geo_.class_layers[c])*size_t(geo_.n_experts));
        }
        l1_->enable_vmm(redraw_max_static_, redraw_.handle_bytes);
    }

    static std::string split_text(const std::vector<int> & v) {
        std::string out;
        for (size_t c = 0; c < v.size(); ++c) { out += (c ? "," : "") + std::to_string(v[c]); }
        return out;
    }

    void redraw_started_locked(bool cold) {
        redraw_gate_ = redraw_gate();
        if (!redraw_.enabled || redraw_max_static_.empty()) {
            return;
        }
        if (!l1_->vmm()) {
            GGML_LOG_WARN("expert cache: L1 redraw requested but off: no VMM arena (%s)\n", l1_->vmm_reason().c_str());
            return;
        }
        redraw_on_ = true;
        if (cold) {
            redraw_mark_fresh(redraw_gate_, redraw_);
        }
        char move[32], benefit[32];
        snprintf(move, sizeof(move), redraw_.move_pct > 0.0 ? "> %g %%" : "not set", redraw_.move_pct);
        snprintf(benefit, sizeof(benefit), redraw_.benefit > 0.0 ? "> %g x cost" : "not set", redraw_.benefit);
        GGML_LOG_INFO("expert cache: L1 redraw on: VMM class arenas, %zu MiB handles, %.1f GiB of address range for static "
                      "capacities up to %s (now %s); gate: move %s of the L1 budget, benefit over %d tokens %s, %d consecutive "
                      "proposals (none in the first installs after a cold load or member join)%s, hold-off %d installs, at most "
                      "%zu MiB per install\n",
            redraw_.handle_bytes >> 20, double(l1_->vmm_reserved_bytes())/double(size_t(1) << 30),
            split_text(redraw_max_static_).c_str(), split_text(capacities_).c_str(), move, redraw_.horizon, benefit,
            redraw_.confirm, cold ? ", cold load" : "", redraw_.holdoff, redraw_.cap_bytes >> 20);
    }

    // [class] the static capacity a redraw may give a class: at most every expert of the class and
    // what its address range holds; with a finite inclusive tier the host cut must still hold the
    // class's VRAM slots (the host plan's mandatory masters); with a finite exclusive tier a layer
    // may keep no more experts in the file than its storage class ring holds, so the class keeps at
    // least all experts less its host cut less a full ring per layer.
    void redraw_bounds_locked(std::vector<int> & lo, std::vector<int> & hi) const {
        const size_t classes = geo_.class_bytes.size();
        lo.assign(classes, 0);
        hi.assign(classes, 0);
        for (size_t c = 0; c < classes; ++c) {
            hi[c] = std::min(redraw_max_static_[c], geo_.class_layers[c]*geo_.n_experts);
            if (tier_ && c < host_capacities_base_.size()) {
                const int cut = host_capacities_base_[c];
                if (cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE) {
                    hi[c] = std::min(hi[c], cut);
                } else if (c < file_cap_.size()) {
                    lo[c] = std::max(0, geo_.class_layers[c]*(geo_.n_experts - file_cap_[c]) - cut);
                }
            }
            lo[c] = std::min(lo[c], capacities_[c]);
            hi[c] = std::max(hi[c], capacities_[c]);
        }
    }

    // The gate for the plan about to be installed; the new static capacities when it fires, else empty.
    std::vector<int> redraw_evaluate_locked(ggml_expert_plan_id id, const plan_state & plan) {
        const std::vector<int> now = capacities_;
        const size_t budget = redraw_static_bytes(geo_, now);
        std::vector<int> lo, hi;
        redraw_bounds_locked(lo, hi);
        const bool owned = l1_->owns_host_storage();
        const uint64_t * counts = plan.scores.empty() ? nullptr : plan.scores.data();
        std::vector<int> proposed = redraw_capacities(geo_, counts, budget, owned, now, lo, hi);
        proposed = redraw_limit(geo_, now, proposed, redraw_.cap_bytes, budget);
        placement_inputs in;
        in.geo              = &geo_;
        in.counts           = counts;
        in.budget_bytes     = SIZE_MAX;
        in.exclusive        = owned;
        in.fixed_capacities = &proposed;
        const placement next = plan_placement(in);
        const expert_slot_table & gpu = l1_->host_slots();
        const expert_locations & homes = l1_->locations();
        const bool file_tier = tier_ != nullptr;
        auto in_file = [&](int l, int e) {
            return file_tier && gpu[(size_t) l][(size_t) e] < 0 && (size_t) l < homes.size() &&
                homes[(size_t) l][(size_t) e].storage == expert_storage::file;
        };
        const size_t l1_budget = cfg_.l1_bytes != 0 ? cfg_.l1_bytes : redraw_static_bytes(geo_, l1_->slot_counts());
        const redraw_estimate est = redraw_estimate_of(geo_, counts, cfg_.l2_experts_used, plan.selected, next.selected, in_file,
            cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE, now, proposed, l1_budget, redraw_);
        const redraw_verdict v = redraw_decide(redraw_, est, redraw_gate_);
        char move[48], benefit[48];
        if (redraw_.move_pct > 0.0) { snprintf(move, sizeof(move), "> %g %%: %s", redraw_.move_pct, v.move_ok ? "yes" : "no"); }
        else { snprintf(move, sizeof(move), "not set"); }
        if (redraw_.benefit > 0.0) { snprintf(benefit, sizeof(benefit), "> %g: %s", redraw_.benefit, v.benefit_ok ? "yes" : "no"); }
        else { snprintf(benefit, sizeof(benefit), "not set"); }
        GGML_LOG_INFO("expert cache: L1 redraw check, plan %u of bank '%s': static split %s -> %s (floors %s, ceilings %s), "
                      "moves %.1f MiB = %.3f %% of the L1 budget (%s); benefit %.5f ms/token x %d tokens = %.2f ms, cost %.2f ms "
                      "(promote %.2f, demote %.2f, remap %.2f, evacuate %.2f; %zu slices in, %zu out, %zu handles), "
                      "ratio %.3f (%s); streak %d, hold-off %d: %s\n",
            id, banks_[plan.bank].label.c_str(), split_text(now).c_str(), split_text(proposed).c_str(), split_text(lo).c_str(),
            split_text(hi).c_str(), double(est.moved_bytes)/double(1 << 20), est.move_pct, move, est.benefit_ms_token,
            redraw_.horizon, est.benefit_ms, est.cost_ms, est.cost_promote_ms, est.cost_demote_ms, est.cost_remap_ms,
            est.cost_evacuate_ms, est.enter_slices, est.leave_slices, est.handles, est.ratio, benefit, redraw_gate_.streak,
            redraw_gate_.holdoff, v.why.c_str());
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) {
            GGML_LOG_INFO("expert_metrics {\"kind\":\"l1_redraw_check\",\"plan\":%u,\"moved_bytes\":%zu,\"move_pct\":%.6f,"
                          "\"benefit_ms_token\":%.9f,\"horizon\":%d,\"benefit_ms\":%.6f,\"cost_ms\":%.6f,\"cost_promote_ms\":%.6f,"
                          "\"cost_demote_ms\":%.6f,\"cost_remap_ms\":%.6f,\"cost_evacuate_ms\":%.6f,\"ratio\":%.6f,"
                          "\"move_ok\":%d,\"benefit_ok\":%d,\"fire\":%d}\n",
                id, est.moved_bytes, est.move_pct, est.benefit_ms_token, redraw_.horizon, est.benefit_ms, est.cost_ms,
                est.cost_promote_ms, est.cost_demote_ms, est.cost_remap_ms, est.cost_evacuate_ms,
                std::isfinite(est.ratio) ? est.ratio : -1.0, v.move_ok ? 1 : 0, v.benefit_ok ? 1 : 0, v.fire ? 1 : 0);
        }
        return v.fire ? proposed : std::vector<int>();
    }

    // The redraw, inside the install's quiescence (the device is drained, the mutex held):
    //   1. the shrinking classes' kept residents above their new slot count move down (VRAM copies,
    //      swapped with residents the new plan drops when no slot below is free);
    //   2. an install of the plan with the shrinking classes at their new capacity and the growing ones
    //      at the old: the slots above the new count retire;
    //   3. the arena moves the handles (expert-l1.cu, resize);
    //   4. an install of the plan in the new capacities: the growing classes fill their new slots.
    // Every step keeps one home per expert (exclusive) and the file as the backing of what it drops.
    bool redraw_locked(ggml_expert_plan_id id, const plan_state & plan, const std::vector<int> & caps_new) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t classes = geo_.class_bytes.size();
        const std::vector<int> caps_old = capacities_, slots_old = l1_->slot_counts();
        std::vector<int> caps_shrink(classes), limit(classes), slots_new(classes);
        for (size_t c = 0; c < classes; ++c) {
            const int fixed = slots_old[c] - caps_old[c];   // the spares
            caps_shrink[c] = std::min(caps_old[c], caps_new[c]);
            limit[c]       = caps_shrink[c] + fixed;
            slots_new[c]   = caps_new[c] + fixed;
        }
        const bool owned = l1_->owns_host_storage();
        auto place = [&](const std::vector<int> & caps) {
            placement_inputs in;
            in.geo              = &geo_;
            in.counts           = plan.scores.empty() ? nullptr : plan.scores.data();
            in.budget_bytes     = SIZE_MAX;
            in.exclusive        = owned;
            in.fixed_capacities = &caps;
            return plan_placement(in).selected;
        };
        plan_state step = plan;
        step.selected = place(caps_shrink);
        // 1. the spares stay where they are
        std::vector<std::vector<uint8_t>> other(classes);
        for (size_t c = 0; c < classes; ++c) { other[c].assign((size_t) std::max(slots_old[c], 0), 0); }
        const auto & spares = l1_->gpu_spares();
        for (size_t c = 0; c < spares.size() && c < classes; ++c) {
            for (int s : spares[c]) { if (s >= 0 && s < slots_old[c]) { other[c][(size_t) s] = 1; } }
        }
        const evac_plan ev = plan_evacuation(geo_.layer_class, geo_.n_experts, l1_->host_slots(), step.selected, other, slots_old, limit);
        if (!ev.valid) { abort_locked(("L1 redraw: " + ev.reason).c_str()); }
        double evac_ms = 0.0;
        if (!l1_->evacuate(ev.ops, evac_ms)) { abort_locked("L1 redraw: the evacuation copies failed"); }
        size_t swaps = 0, evac_bytes = 0;
        for (const evac_op & op : ev.ops) {
            swaps += op.victim_layer >= 0 ? 1 : 0;
            evac_bytes += geo_.class_total_bytes(op.cls)*(op.victim_layer >= 0 ? 3 : 1);
        }
        GGML_LOG_INFO("expert cache: L1 redraw %llu, plan %u: static %s -> %s (slots %s -> %s); step 1: %zu kept residents moved "
                      "below the new slot counts (%zu swaps, %.1f MiB copied) in %.1f ms\n",
            (unsigned long long) redraw_count_ + 1, id, split_text(caps_old).c_str(), split_text(caps_new).c_str(),
            split_text(slots_old).c_str(), split_text(slots_new).c_str(), ev.ops.size(), swaps,
            double(evac_bytes)/double(1 << 20), evac_ms);
        // 2. the shrink install
        const auto t1 = std::chrono::steady_clock::now();
        set_capacities_locked(caps_shrink);
        l1_->set_retire(limit);
        redraw_unequal_ = cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE;
        l1_->set_unequal(redraw_unequal_);
        const bool shrunk = install_once_locked(id, step);
        l1_->set_retire({});
        if (!shrunk) {
            redraw_unequal_ = false;
            l1_->set_unequal(false);
            return false;
        }
        if (tier_) {
            tier_->stop_worker();
        }
        const double shrink_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        // 3. the handles
        l1_arena::resize_stats rs;
        std::string why;
        if (!l1_->resize(slots_new, rs, why)) { abort_locked(("L1 redraw resize: " + why).c_str()); }
        set_capacities_locked(caps_new);
        ++redraw_epoch_;
        GGML_LOG_INFO("expert cache: L1 redraw %llu: step 2: shrink install in %.1f ms; step 3: %zu handles moved, %zu created, "
                      "%zu released, %zu partial units replaced (%.1f MiB copied), %.1f MiB zeroed, VRAM %.1f -> %.1f MiB, in %.1f ms\n",
            (unsigned long long) redraw_count_ + 1, shrink_ms, rs.handles_moved, rs.handles_created, rs.handles_released,
            rs.units_replaced, double(rs.copied_bytes)/double(1 << 20), double(rs.zeroed_bytes)/double(1 << 20),
            double(rs.mapped_before)/double(1 << 20), double(rs.mapped_after)/double(1 << 20), rs.ms);
        // 4. the grow install
        const auto t2 = std::chrono::steady_clock::now();
        step.selected = place(caps_new);
        step.epoch    = redraw_epoch_;
        const bool grown = install_once_locked(id, step);
        redraw_unequal_ = false;
        l1_->set_unequal(false);
        ++redraw_count_;
        const double grow_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        GGML_LOG_INFO("expert cache: L1 redraw %llu done: step 4: grow install in %.1f ms; %.1f ms in all, epoch %u\n",
            (unsigned long long) redraw_count_, grow_ms, ms, redraw_epoch_);
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) {
            GGML_LOG_INFO("expert_metrics {\"kind\":\"l1_redraw\",\"plan\":%u,\"evacuated\":%zu,\"swaps\":%zu,\"evacuate_ms\":%.6f,"
                          "\"shrink_ms\":%.6f,\"handles_moved\":%zu,\"handles_created\":%zu,\"handles_released\":%zu,"
                          "\"units_replaced\":%zu,\"resize_ms\":%.6f,\"grow_ms\":%.6f,\"total_ms\":%.6f}\n",
                id, ev.ops.size(), swaps, evac_ms, shrink_ms, rs.handles_moved, rs.handles_created, rs.handles_released,
                rs.units_replaced, rs.ms, grow_ms, ms);
        }
        return grown;
    }

    tier_inputs tier_inputs_locked(const std::vector<uint64_t> & scores) const {
        tier_inputs in;
        in.inclusive = cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE;
        in.minimum_capacities = in.inclusive ? &capacities_ : nullptr;
        in.geo = &geo_;
        in.counts = scores.empty() ? nullptr : scores.data();
        in.vram = &tier_vram_;
        in.slot_pitch = &host_pitch_;
        in.file_cap = file_cap_.empty() ? nullptr : &file_cap_;
        return in;
    }

    // Sizes the ring out of the host budget and cuts the rest three ways. `host_capacities` comes in
    // as the two-tier capacity (everything that is not in VRAM) and goes out as the finite one.
    bool size_tier_locked(const std::vector<std::vector<int32_t>> & selected, std::vector<int> & host_capacities) {
        l2_config tier_cfg = l2_debug_config(cfg_, verify_all_);
        tier_cfg.early = hash_mode_.ssd && !early_decl_.empty();
        tier_.reset(new l2_tier(geo_, tier_cfg));
        if (!tier_->sized()) {
            tier_.reset();
            disable_locked("the SSD tier could not size its staging ring");
            return false;
        }
        file_cap_.clear();
        if (tier_->class_layout()) {
            return size_class_tier_locked(selected, host_capacities);
        }
        const size_t fixed = tier_->fixed_bytes() + spare_bytes_locked() + arena_tail_total(geo_);
        const auto minimum = minimum_host_budget(geo_, capacities_, fixed, cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE);
        if (!minimum.valid || cfg_.l2_bytes < minimum.bytes) {
            GGML_ABORT("expert cache: L2 budget %zu bytes is below minimum %zu bytes (L1 payload upper bound %zu bytes, ring/metadata/padding/spares %zu bytes); uncached loading refused",
                cfg_.l2_bytes, minimum.bytes, minimum.l1_payload, fixed);
        }

        host_pitch_.assign(geo_.class_bytes.size(), 0);
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            host_pitch_[cls] = geo_.class_total_bytes(cls);
        }
        vram_table_locked(selected, tier_vram_);

        tier_inputs in = tier_inputs_locked(tier_scores_);
        in.budget_bytes = cfg_.l2_bytes - fixed;
        const tier_plan base = plan_host_tier(in);
        if (!base.valid) { disable_locked(base.reason.c_str()); return false; }
        host_capacities_base_ = base.capacities;
        host_capacities       = host_capacities_base_;
        const auto expanded = plan_borrowed_capacities(in, host_capacities_base_,
            size_t(tier_->prompt_slots() - tier_->decode_slots()));
        lent_capacities_.resize(expanded.size());
        lent_starts_.resize(expanded.size());
        int next = tier_->decode_slots();
        for (size_t cls = 0; cls < expanded.size(); ++cls) {
            lent_capacities_[cls] = expanded[cls] - host_capacities_base_[cls];
            lent_starts_[cls] = next;
            next += lent_capacities_[cls];
        }
        GGML_ASSERT(next <= tier_->prompt_slots());
        GGML_LOG_INFO("expert cache: SSD tier budget %zu MiB: ring %zu MiB, tables %zu MiB, "
                      "host residents %zu MiB, %zu slices / %zu MiB left in the file\n",
            cfg_.l2_bytes/(1024*1024), tier_->ring_bytes(tier_->prompt_slots())/(1024*1024),
            tier_->metadata_bytes()/(1024*1024), base.resident_bytes/(1024*1024),
            base.ssd_slices, base.ssd_bytes/(1024*1024));
        return true;
    }

    // The class layout: the storage classes, their rings and the host cut, from the plan that seeds
    // the load. The ring of a storage class is its prompt floor: the most file residents one of its
    // layers keeps under the seed plan, bounded by what one ubatch can demand (the bound itself
    // without a stored profile). The floor is found by growing the rings from zero until the cut
    // under them keeps no layer above its ring. An explicit ring size is the total over the classes;
    // what it adds above the floors is split by the bytes each class leaves in the file. Both phases
    // use the same rings. Later plans are held to the rings (tier_inputs::file_cap).
    bool size_class_tier_locked(const std::vector<std::vector<int32_t>> & selected, std::vector<int> & host_capacities) {
        const size_t classes = geo_.class_bytes.size();
        const bool inclusive = cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE;
        std::vector<std::array<size_t, 3>> tails(classes);
        for (size_t c = 0; c < classes; ++c) {
            for (int k = 0; k < geometry::n_kinds; ++k) { tails[c][k] = arena_tail_bytes(geo_, (int) c, k); }
        }
        const int bound = storage_demand_bound(geo_.n_experts, cfg_.l2_experts_used, cfg_.l2_prefill_rows);
        class_storage st = plan_storage_classes(geo_, tails, expert_os::io_alignment, bound);
        if (!st.valid() || bound <= 0) {
            tier_.reset();
            disable_locked("the SSD tier could not lay out its storage classes");
            return false;
        }
        const int n = st.storages();
        host_pitch_.assign(classes, 0);
        for (size_t c = 0; c < classes; ++c) { host_pitch_[c] = st.class_pitch((int) c); }
        vram_table_locked(selected, tier_vram_);
        size_t spares = 0;
        for (size_t c = 0; c < classes; ++c) { spares += size_t(cfg_.spare_slots)*host_pitch_[c]; }
        const size_t base_fixed = tier_->metadata_bytes() + spares + st.tail_bytes();
        auto ring_bytes = [&](const std::vector<int> & ring) {
            size_t bytes = 0;
            for (int s = 0; s < n; ++s) { bytes += size_t(ring[s])*st.stride(s); }
            return bytes;
        };
        auto ring_text = [&](const std::vector<int> & ring) {
            std::string text;
            for (int s = 0; s < n; ++s) { text += (s ? "," : "") + std::to_string(ring[s]); }
            return text;
        };
        // refuses the load when the budget cannot hold this ring
        auto require = [&](const std::vector<int> & ring) {
            const size_t fixed = base_fixed + ring_bytes(ring);
            const auto minimum = minimum_host_budget(geo_, capacities_, fixed, inclusive);
            if (!minimum.valid || cfg_.l2_bytes < minimum.bytes) {
                GGML_ABORT("expert cache: L2 budget %zu bytes is below minimum %zu bytes (L1 payload upper bound %zu bytes, "
                           "ring/metadata/padding/spares %zu bytes, of which ring %zu bytes = slots %s of the storage classes); "
                           "uncached loading refused",
                    cfg_.l2_bytes, minimum.bytes, minimum.l1_payload, fixed, ring_bytes(ring), ring_text(ring).c_str());
            }
        };
        auto cut = [&](const std::vector<int> & ring) {
            tier_inputs in = tier_inputs_locked(tier_scores_);
            in.budget_bytes = cfg_.l2_bytes - base_fixed - ring_bytes(ring);
            return plan_host_tier(in);
        };
        std::vector<int> ring(n, bound);
        tier_plan base;
        if (!tier_scores_.empty()) {
            std::fill(ring.begin(), ring.end(), 0);
            for (int round = 0; round < 1024; ++round) {
                require(ring);
                base = cut(ring);
                if (!base.valid) { disable_locked(base.reason.c_str()); return false; }
                const std::vector<int> need = storage_floors(geo_, st, &tier_vram_, &base.selected, bound);
                bool grew = false;
                for (int s = 0; s < n; ++s) {
                    if (need[s] > ring[s]) { ring[s] = need[s]; grew = true; }
                }
                if (!grew) { break; }
            }
        }
        const std::vector<int> floors = ring;
        // Without a stored profile the floors are the per ubatch bound, already the most one layer
        // can demand; the factor applies only to floors that come from a seed plan.
        const bool seeded = !tier_scores_.empty();
        const double factor = seeded ? tier_->ring_factor() : 1.0;
        std::string note;
        if (!seeded && tier_->ring_factor() > 1.0 && cfg_.l2_prefill_ring_bytes == 0) {
            char text[64];
            snprintf(text, sizeof(text), "%g", tier_->ring_factor());
            note = std::string("no seed profile: the floors are the per ubatch bound, the ring factor ") + text + " is not applied";
        }
        const size_t requested = cfg_.l2_prefill_ring_bytes;
        if (requested != 0) {
            require(ring);
            if (!base.valid) { base = cut(ring); }
            if (!base.valid) { disable_locked(base.reason.c_str()); return false; }
            std::vector<uint64_t> demand(n, 0);
            std::vector<int> limit(n, 0);
            for (int l = 0; l < geo_.n_layers; ++l) {
                const int c = geo_.layer_class[l];
                if (c < 0) { continue; }
                const int s = st.storage_of[c];
                limit[s] += geo_.n_experts;
                int resident = 0;
                for (int e = 0; e < geo_.n_experts; ++e) {
                    resident += tier_vram_[l][e] >= 0 || std::binary_search(base.selected[l].begin(), base.selected[l].end(), e) ? 1 : 0;
                }
                demand[s] += uint64_t(geo_.n_experts - resident)*geo_.class_total_bytes(c);
            }
            ring = split_ring(st, floors, demand, limit, requested);
            if (ring_bytes(floors) > requested) {
                note = "the requested ring of " + std::to_string(requested/(1024*1024)) + " MiB was raised to its floors";
            }
        } else if (factor > 1.0) {
            // factor x floor per class, as far as the budget left above the minimum (the floors) allows
            std::vector<int> limit(n, 0);
            for (int c = 0; c < (int) classes; ++c) { limit[st.storage_of[c]] += geo_.class_layers[c]*geo_.n_experts; }
            const std::vector<int> want = factor_rings(floors, factor, limit);
            require(floors);   // the minimum stays the floors; the factor never refuses a budget
            const auto minimum = minimum_host_budget(geo_, capacities_, base_fixed + ring_bytes(floors), inclusive);
            ring = shrink_rings(st, floors, want, cfg_.l2_bytes - minimum.bytes);
            // A larger ring leaves fewer host residents, so a layer can keep more of its experts in
            // the file than it did at the floors; such a ring grows to that count while it fits.
            for (int round = 0; round < 1024; ++round) {
                const tier_plan grown = cut(ring);
                if (!grown.valid) { break; }
                const std::vector<int> need = storage_floors(geo_, st, &tier_vram_, &grown.selected, bound);
                bool grew = false;
                std::vector<int> next = ring;
                for (int s = 0; s < n; ++s) {
                    if (need[s] > next[s]) { next[s] = need[s]; grew = true; }
                }
                if (!grew) { break; }
                if (cfg_.l2_bytes < minimum_host_budget(geo_, capacities_, base_fixed + ring_bytes(next), inclusive).bytes) {
                    ring = floors;
                    note = "the rings above their floors did not fit with the file residents they add, the floors are used";
                    break;
                }
                ring = next;
            }
            bool short_of_want = false;
            for (int s = 0; s < n; ++s) { short_of_want = short_of_want || ring[s] < want[s]; }
            if (short_of_want && note.empty()) {
                char text[64];
                snprintf(text, sizeof(text), "%g", factor);
                note = "the budget gives the rings " + ring_text(ring) + " slots instead of " + ring_text(want) +
                    " (factor " + text + " of the floors), shrunk toward the floors in proportion";
            }
        }
        if (cfg_.l2_phase_rings && cfg_.l2_decode_ring_bytes != cfg_.l2_prefill_ring_bytes) {
            note += std::string(note.empty() ? "" : "; ") + "the class layout keeps one ring size in both phases, the decode ring size is not used";
        }
        file_cap_.assign(classes, 0);
        for (size_t c = 0; c < classes; ++c) { file_cap_[c] = ring[st.storage_of[c]]; }
        require(ring);
        base = cut(ring);
        if (!base.valid) { disable_locked(base.reason.c_str()); return false; }
        host_capacities_base_ = base.capacities;
        host_capacities       = host_capacities_base_;
        lent_capacities_.assign(classes, 0);
        lent_starts_.assign(classes, 0);
        std::vector<int> residents(classes);
        for (size_t c = 0; c < classes; ++c) { residents[c] = host_capacities_base_[c] + cfg_.spare_slots; }
        if (!st.place(residents, ring)) { disable_locked("the SSD tier could not place its storage slots"); return false; }
        st.floor = floors;
        st.factor = requested != 0 ? 0.0 : factor;
        tier_->set_storage(st);
        if (!tier_->sized()) {
            tier_.reset();
            disable_locked("the SSD tier refused its storage class layout");
            return false;
        }
        GGML_LOG_INFO("expert cache: SSD tier budget %zu MiB (class layout): ring %zu MiB (slots %s, floors %s, "
                      "factor %g%s, per ubatch bound %d%s), tables %zu MiB, spares %zu MiB, host residents %zu MiB, "
                      "%zu slices / %zu MiB left in the file\n",
            cfg_.l2_bytes/(1024*1024), st.ring_bytes()/(1024*1024), ring_text(ring).c_str(), ring_text(floors).c_str(),
            factor, requested != 0 ? " not used, explicit total" : "", bound, tier_scores_.empty() ? ", no seed profile" : "", tier_->metadata_bytes()/(1024*1024),
            spares/(1024*1024), base.resident_bytes/(1024*1024), base.ssd_slices, base.ssd_bytes/(1024*1024));
        if (!note.empty()) {
            GGML_LOG_INFO("expert cache: %s\n", note.c_str());
        }
        return true;
    }

    bool allocate_tier_locked(const std::vector<std::vector<int32_t>> & selected) {
        if (!tier_->allocate(device_)) {
            tier_.reset();
            host_.reset();
            l1_.reset();
            profiler_.reset();
            disable_locked("SSD tier allocation failed");
            return false;
        }
        tier_inputs in = tier_inputs_locked(tier_scores_);
        in.fixed_capacities = &host_capacities_base_;
        const tier_plan host_plan = plan_host_tier(in);
        if (!host_plan.valid) { abort_locked(host_plan.reason.c_str()); }

        // The load time assignment is not a move: the loader writes every slice to the home this
        // hands out, so it is built directly instead of through the mover.
        std::vector<std::vector<int32_t>> gpu(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
        std::vector<std::vector<int32_t>> host(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
        std::vector<int> next_gpu(geo_.class_bytes.size(), 0), next_host(geo_.class_bytes.size(), 0);
        bool ok = true;
        for (int layer = 0; layer < geo_.n_layers && ok; ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0) {
                continue;
            }
            std::vector<bool> in_gpu(geo_.n_experts, false), in_host(geo_.n_experts, false);
            for (int32_t expert : selected[layer]) {
                in_gpu[expert] = true;
            }
            for (int32_t expert : host_plan.selected[layer]) {
                in_host[expert] = true;
            }
            for (int expert = 0; expert < geo_.n_experts; ++expert) {
                if (in_gpu[expert]) {
                    ok = ok && next_gpu[cls] < capacities_[cls];
                    if (ok) { gpu[layer][expert] = next_gpu[cls]++; }
                }
                if (in_host[expert]) {
                    ok = ok && next_host[cls] < host_capacities_base_[cls];
                    if (ok) { host[layer][expert] = next_host[cls]++; }
                }
            }
        }
        std::vector<std::vector<int>> spares(geo_.class_bytes.size());
        std::vector<int> host_caps = host_capacities_base_;
        for (size_t cls = 0; cls < geo_.class_bytes.size(); ++cls) {
            host_caps[cls] += cfg_.spare_slots;
            for (int i = 0; i < cfg_.spare_slots; ++i) {
                spares[cls].push_back(capacities_[cls] + i);
            }
        }
        if (!ok || !l1_->assign_tier(gpu, host_locations(host), tier_layout_locked(true), selected, spares)) {
            tier_.reset();
            host_.reset();
            l1_.reset();
            profiler_.reset();
            disable_locked("three-tier slot assignment failed");
            return false;
        }
        tier_->set_homes(gpu, l1_->locations(), host_geometry_locked());
        l2_tier * tier = tier_.get();
        l1_->attach_locations([tier](int layer, int kind, expert_location at) { return tier->location_address(layer, kind, at); });
        l1_arena::tier_reader reader;
        reader.read    = [tier](std::vector<l2_read> & reads, std::string & why) { return tier->read_install(reads, why); };
        reader.address = [tier](const l2_read & read) { return tier->read_address(read); };
        reader.direct_host = tier_->class_layout();
        // Constant across a repartition: install_read_slots is also bounded by the decode ring,
        // which is the smaller of the two ring sizes.
        reader.slots   = tier_->install_read_slots();
        l1_->attach_tier_reader(std::move(reader));
        return true;
    }

    l2_host_geometry host_geometry_locked() const {
        l2_host_geometry out;
        if (host_->class_layout()) {
            const int storages = tier_->storage().storages();
            out.device_base.assign(storages, {nullptr, nullptr, nullptr});
            out.host_base.assign(storages, {nullptr, nullptr, nullptr});
            for (int s = 0; s < storages; ++s) {
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    out.device_base[s][kind] = host_->storage_base(s, kind, true);
                    out.host_base[s][kind]   = host_->storage_base(s, kind, false);
                }
            }
            return out;
        }
        const size_t classes = geo_.class_bytes.size();
        out.device_base.assign(classes, {nullptr, nullptr, nullptr});
        out.host_base.assign(classes, {nullptr, nullptr, nullptr});
        for (size_t cls = 0; cls < classes; ++cls) {
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                out.device_base[cls][kind] = const_cast<void *>(host_->device_data((int) cls, kind));
                out.host_base[cls][kind]   = host_->slice((int) cls, kind, 0);
            }
        }
        return out;
    }

    // The joint layers of the bound members that are not dropped; with `newly_joined` only those whose
    // files the SSD tier has not opened yet.
    std::vector<uint8_t> member_layers_locked(bool newly_joined) const {
        std::vector<uint8_t> mask((size_t) geo_.n_layers, 0);
        for (const member_state & m : members_) {
            if (!m.bound || m.dropped || (newly_joined && m.tier_ready)) {
                continue;
            }
            for (int l = m.offset; l < m.offset + m.geo.n_layers && l < geo_.n_layers; ++l) {
                mask[(size_t) l] = 1;
            }
        }
        return mask;
    }

    bool start_tier_locked() {
        if (!tier_->map()) {
            abort_locked("SSD tier ring registration failed");
        }
        // Models that join later have no backing yet; their files are opened when they load.
        const std::vector<uint8_t> layers = member_layers_locked(false);
        std::string reason;
        if (!tier_->open_files(reason, &layers)) {
            abort_locked(("SSD tier: " + reason).c_str());
        }
        verify_tier_backing_locked(&layers);
        for (member_state & m : members_) {
            m.tier_ready = m.bound && !m.dropped;
        }
        l2_tier * tier = tier_.get();
        l1_->attach_addresses([tier](int layer, int kind) { return tier->addresses(layer, kind); });
        tier_->set_homes(l1_->host_slots(), l1_->locations(), host_geometry_locked());
        tier_->set_early_layers(early_joint_locked());
        tier_->start_worker();
        return true;
    }

    // The tier reads the GGUF itself, so its idea of where an expert lives has to agree with the
    // loader's. One host resident expert per routed layer and kind is read back from the file and
    // compared with the bytes the loader put in the arena. Always on: it is a few slices and it is
    // the only check that the tier and the loader read the same offsets.
    void verify_tier_backing_locked(const std::vector<uint8_t> * layers) {
        const std::vector<std::vector<int32_t>> & gpu_table  = l1_->host_slots();
        const std::vector<std::vector<int32_t>> & host_table = l1_->arena_slots();
        std::vector<uint8_t> from_file, loaded_copy;
        size_t checked = 0, checked_host = 0, checked_vram = 0, unchecked = 0;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0 || (layers != nullptr && !(*layers)[(size_t) layer])) {
                continue;
            }
            // A host resident is compared where the loader wrote it; a layer without one (a joined
            // model whose whole host share went to the file) is compared against a VRAM resident,
            // read back. Only a layer with no loaded expert at all has nothing to compare with.
            int expert = -1;
            for (int candidate = 0; candidate < geo_.n_experts && expert < 0; ++candidate) {
                if (gpu_table[layer][candidate] < 0 && host_table[layer][candidate] >= 0) {
                    expert = candidate;
                }
            }
            bool in_vram = false;
            for (int candidate = 0; candidate < geo_.n_experts && expert < 0; ++candidate) {
                if (gpu_table[layer][candidate] >= 0) {
                    expert  = candidate;
                    in_vram = true;
                }
            }
            if (expert < 0) {
                ++unchecked;
                GGML_LOG_WARN("expert cache: SSD tier backing of layer %d cannot be checked: no expert of it was loaded\n", layer);
                continue;
            }
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                const size_t stride = geo_.class_bytes[cls][kind];
                from_file.assign(stride, 0);
                std::string why;
                if (!tier_->read_slice(layer, kind, expert, from_file.data(), why)) {
                    abort_locked(("SSD tier backing check: " + why).c_str());
                }
                const void * loaded = nullptr;
                if (in_vram) {
                    loaded_copy.assign(stride, 0);
                    if (l1_->read_slice(layer, kind, gpu_table[layer][expert], loaded_copy.data())) {
                        loaded = loaded_copy.data();
                    }
                } else {
                    loaded = tier_->host_address(layer, kind, expert);
                }
                if (loaded == nullptr || memcmp(from_file.data(), loaded, stride) != 0) {
                    GGML_ABORT("expert cache: the SSD tier reads layer %d kind %d expert %d from the wrong "
                               "place in the file", layer, kind, expert);
                }
                ++checked;
                (in_vram ? checked_vram : checked_host) += 1;
            }
        }
        GGML_LOG_INFO("expert cache: SSD tier backing checked on %zu slices against the loaded weights "
                      "(%zu host residents, %zu VRAM residents), %zu routed layers without a loaded expert\n",
            checked, checked_host, checked_vram, unchecked);
    }

    void vram_table_locked(const std::vector<std::vector<int32_t>> & selected,
            std::vector<std::vector<int32_t>> & table) const {
        table.assign(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1));
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            for (int32_t expert : selected[layer]) {
                table[layer][expert] = 0;
            }
        }
    }

    install_layout tier_layout_locked(bool prompt) const {
        install_layout out;
        out.gpu = capacities_; out.host = host_capacities_base_;
        out.lent_begin = lent_starts_; out.lent_count = lent_capacities_;
        for (size_t c = 0; c < out.gpu.size(); ++c) {
            out.gpu[c] += cfg_.spare_slots; out.host[c] += cfg_.spare_slots;
            if (prompt) { out.lent_count[c] = 0; }
        }
        return out;
    }

    bool install_tier_locked(ggml_expert_plan_id id, const plan_state & plan, bool seed) {
        const auto t0 = std::chrono::steady_clock::now();
        tier_->stop_worker();
        bool has_prefill = false;
        for (const bank_state & bank : banks_) { has_prefill |= bank.prompt && bank.latest_plan != GGML_EXPERT_PLAN_NONE; }
        const bool prompt = plan_uses_prompt_ring(cfg_.l2_phase_rings, !banks_[plan.bank].prompt, has_prefill, seed);
        const int ring_before = tier_->ring_count();
        const auto layout = tier_layout_locked(prompt);
        auto cut = host_capacities_base_;
        for (size_t c = 0; c < cut.size(); ++c) { cut[c] += layout.lent_count[c]; }
        vram_table_locked(plan.selected, tier_vram_);
        tier_inputs in = tier_inputs_locked(plan.scores);
        in.fixed_capacities = &cut;
        const tier_plan host_plan = plan_host_tier(in);
        if (!host_plan.valid) { abort_locked(host_plan.reason.c_str()); }
        const auto tx = ggml_cuda_expert::plan_install(geo_, plan.selected, host_plan.selected, l1_->host_slots(), l1_->locations(),
            capacities_, l1_->layout(), layout, l1_->gpu_spares(),
            {cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE, cfg_.spare_slots}, true, /*retain =*/ true,
            /*unequal =*/ redraw_unequal_ && cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE);
        if (!tx.valid) { abort_locked(("install transaction: " + tx.reason).c_str()); }
        std::string reason;
        // Class layout: the moves run as relabels of the storage slots (the rings keep their
        // contents); the separate ring of direct tier users goes through the mover.
        const bool relabel = tier_->relabel_installs();
        l2_install_stats moved;
        if (relabel) {
            if (!tier_->install_relabel(tx, l1_->locations(), *l1_, moved, reason)) {
                abort_locked(("install relabel: " + reason).c_str());
            }
        } else {
            if (!l1_->execute(tx)) { abort_locked("the install mover refused a move"); }
            for (const auto & move : tx.moves) for (int k = 0; k < geometry::n_kinds; ++k) {
                tier_->finish_write(move.layer, k, move.to);
            }
            moved.read_slices = tx.ssd_slices;
            moved.read_bytes  = tx.ssd_bytes;
        }
        // The old lent sources stay readable until every move has completed. The class layout has
        // no lent slots and one ring size: a prompt/decode switch is an install like any other.
        const bool repartition = !tier_->class_layout() && tier_->set_prompt_ring(prompt);
        if (!l1_->assign_tier(tx.gpu_slots, tx.host, layout, plan.selected, tx.gpu_spares) ||
                !l1_->verify_current_assignment(reason)) { abort_locked(("install assignment: " + reason).c_str()); }
        // the tables are published after every read and copy of the transaction completed
        tier_->set_homes(tx.gpu_slots, tx.host, host_geometry_locked(), relabel);
        const size_t host_total = host_->host_bytes() + tier_->host_bytes();
        GGML_ASSERT(host_total <= cfg_.l2_bytes);
        if (repartition) {
            GGML_LOG_INFO("expert cache: L2 repartition ring=%d->%d lent_slots=%d host_total=%zu\n",
                ring_before, tier_->ring_count(), tier_->prompt_slots() - tier_->ring_count(), host_total);
        }
        tier_->start_worker();
        log_budget_split_locked(host_plan, prompt);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string relabel_text;
        if (relabel) {
            char text[256];
            snprintf(text, sizeof(text), " relabel=%zu slices / %zu MiB ring-copy=%zu slices / %zu MiB read=%zu slices / %zu MiB "
                     "ring kept %zu/%zu occupants (%zu after, %zu residents demoted into the rings, %d read batches)",
                moved.relabel_slices, moved.relabel_bytes/(1024*1024), moved.ring_copy_slices, moved.ring_copy_bytes/(1024*1024),
                moved.read_slices, moved.read_bytes/(1024*1024), moved.ring_kept, moved.ring_before, moved.ring_after,
                moved.demoted, moved.batches);
            relabel_text = text;
        }
        GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': mode=%s-l2 ring=%s%s retained=%zu/%zu "
                      "h2d=%zu slices / %zu MiB d2h=%zu slices / %zu MiB ssd=%zu slices / %zu MiB%s in %.1f ms\n",
            id, banks_[plan.bank].label.c_str(), cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE ? "inclusive" : "exclusive",
            prompt ? "prompt" : "decode", repartition ? " (changed)" : "", tx.retained_gpu, tx.retained_host,
            tx.h2d_slices, tx.h2d_bytes/(1024*1024), tx.d2h_slices, tx.d2h_bytes/(1024*1024),
            tx.ssd_slices, tx.ssd_bytes/(1024*1024), relabel_text.c_str(), ms);
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) {
            // ssd_bytes: every move from the file, as before; read_bytes: what was actually read
            // (equal to ssd_bytes without relabel installs)
            GGML_LOG_INFO("expert_metrics {\"kind\":\"install\",\"bank\":\"%s\",\"plan\":%u,\"install_ms\":%.6f,"
                          "\"h2d_bytes\":%zu,\"d2h_bytes\":%zu,\"ssd_bytes\":%zu,\"ring_before\":%d,\"ring_after\":%d,"
                          "\"relabel\":%d,\"read_slices\":%zu,\"read_bytes\":%zu,\"relabel_slices\":%zu,\"relabel_bytes\":%zu,"
                          "\"ring_copy_slices\":%zu,\"ring_copy_bytes\":%zu,\"ring_occupants_before\":%zu,\"ring_kept\":%zu,"
                          "\"ring_occupants_after\":%zu,\"demoted_to_ring\":%zu,\"read_batches\":%d}\n",
                banks_[plan.bank].label.c_str(), id, ms, tx.h2d_bytes, tx.d2h_bytes, tx.ssd_bytes, ring_before, tier_->ring_count(),
                relabel ? 1 : 0, moved.read_slices, moved.read_bytes, moved.relabel_slices, moved.relabel_bytes,
                moved.ring_copy_slices, moved.ring_copy_bytes, moved.ring_before, moved.ring_kept, moved.ring_after,
                moved.demoted, moved.batches);
        }
        finish_plan_locked(id); verify_all_locked();
        if (joint_) { log_members_locked("after install"); }
        return true;
    }

    void report_tier_plan_locked(const char * what) const {
        size_t vram = 0, host = 0, ssd = 0, borrowed = 0;
        const std::vector<std::vector<int32_t>> & gpu_table  = l1_->host_slots();
        const std::vector<std::vector<int32_t>> & host_table = l1_->arena_slots();
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            if (geo_.layer_class[layer] < 0) {
                continue;
            }
            for (int expert = 0; expert < geo_.n_experts; ++expert) {
                if (gpu_table[layer][expert] >= 0)       { ++vram; }
                else if (host_table[layer][expert] >= 0) { ++host; }
                else if (tier_->locations()[layer][expert].storage == expert_storage::lent) { ++borrowed; }
                else                                     { ++ssd;  }
            }
        }
        GGML_LOG_INFO("expert cache: SSD tier %s: %zu slices in VRAM, %zu in the host arena, %zu borrowed, %zu in the file\n",
            what, vram, host, borrowed, ssd);
    }

    // One line per install: where the L2 budget went. `resident_payload` is what the arena slot
    // table actually holds after the transaction, not what the cut planned. Lent slices live in
    // ring slots, so their bytes are already inside `ring` and are reported separately, not added.
    void log_budget_split_locked(const tier_plan & host_plan, bool prompt) const {
        const std::vector<std::vector<int32_t>> & host_table = l1_->arena_slots();
        size_t payload = 0, placed = 0, lent = 0, lent_bytes = 0;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0) {
                continue;
            }
            for (int expert = 0; expert < geo_.n_experts; ++expert) {
                if (host_table[layer][expert] >= 0) {
                    payload += geo_.class_total_bytes(cls);
                    ++placed;
                } else if (tier_->locations()[layer][expert].storage == expert_storage::lent) {
                    lent_bytes += geo_.class_total_bytes(cls);
                    ++lent;
                }
            }
        }
        const bool   classes = tier_->class_layout();
        const int    slots  = classes ? tier_->ring_count() : tier_->prompt_slots();
        const size_t ring   = tier_->ring_total_bytes();
        const size_t meta   = tier_->metadata_bytes();
        const size_t pad    = classes ? tier_->storage().tail_bytes() : arena_tail_total(geo_);
        const size_t spare  = classes ? host_spare_bytes_locked() : spare_bytes_locked();
        const size_t tables = table_bytes_locked(true);
        // class layout: every resident slot costs the storage pitch; what it holds beyond the
        // payload (sector room, MMQ tail, and slots the cut left empty) is `slot_padding`
        size_t slot_padding = 0;
        std::string storage_text;
        if (classes) {
            const class_storage & st = tier_->storage();
            size_t reserved = 0;
            for (size_t c = 0; c < host_capacities_base_.size(); ++c) { reserved += size_t(host_capacities_base_[c])*host_pitch_[c]; }
            slot_padding = reserved - std::min(reserved, payload);
            storage_text = ",\"layout\":\"class\",\"storage_classes\":[";
            for (int s = 0; s < st.storages(); ++s) {
                std::string members;
                for (int c : st.members[s]) { members += (members.empty() ? "" : ",") + std::to_string(c); }
                storage_text += std::string(s ? "," : "") + "{\"classes\":[" + members + "],\"pitch\":[" +
                    std::to_string(st.pitch[s][0]) + "," + std::to_string(st.pitch[s][1]) + "," + std::to_string(st.pitch[s][2]) +
                    "],\"ring_slots\":" + std::to_string(st.ring_slots[s]) + ",\"floor\":" + std::to_string(st.floor[s]) + "}";
            }
            storage_text += "],\"slot_padding\":" + std::to_string(slot_padding);
        }
        const size_t accounted = payload + slot_padding + ring + meta + pad + spare;
        // Written as an `expert_metrics` record and not gated by the log mask: it is a default
        // level line, and llama-bench's log callback only lets that prefix through.
        GGML_LOG_INFO("expert_metrics {\"kind\":\"budget\",\"phase\":\"%s\",\"l2_budget\":%zu,"
                      "\"resident_payload\":%zu,\"resident_slices\":%zu,\"host_arena\":%zu,"
                      "\"ring\":%zu,\"ring_slots\":%d,\"ring_pitch\":[%zu,%zu,%zu],"
                      "\"lent_slices\":%zu,\"lent_payload\":%zu,\"metadata\":%zu,\"padding\":%zu,"
                      "\"spare\":%zu,\"vram_tables\":%zu,\"file_slices\":%zu,\"file_bytes\":%zu,"
                      "\"remainder\":%lld%s}\n",
            prompt ? "prompt" : "decode", cfg_.l2_bytes, payload, placed,
            host_->host_bytes(), ring, slots,
            tier_->slot_pitch(0), tier_->slot_pitch(1), tier_->slot_pitch(2),
            lent, lent_bytes, meta, pad, spare, tables,
            host_plan.ssd_slices, host_plan.ssd_bytes,
            (long long) cfg_.l2_bytes - (long long) accounted, storage_text.c_str());
    }

    const char * mode_name() const { return cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE ? "inclusive" : "exclusive"; }

    // Once the arenas own the expert bytes there is no uncached fallback: the routed tensors hold
    // logical addresses and no second copy exists. Every failure past that point ends the process.
    [[noreturn]] void abort_locked(const char * reason) const {
        GGML_ABORT("expert cache: owned host storage cannot continue: %s", reason);
    }

    void disable_locked(const char * reason) {
        if (l1_ && l1_->owns_host_storage()) {
            abort_locked(reason);
        }
        // Pointers stay allocated: a captured graph may hold them. Lookups miss from now on.
        GGML_LOG_WARN("expert cache: disabled: %s\n", reason);
        disabled_reason_ = reason;
        state_ = state_t::disabled;
    }

    void verify_all_locked() {
        if (!verify_all_ || !l1_) {
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (l1_->owns_host_storage()) {
            size_t vram = 0;
            size_t host = 0;
            size_t skipped = 0;
            if (!verify_exclusive_locked(vram, host, skipped)) {
                GGML_ABORT("expert cache: an expert slice differs from the bytes the loader wrote");
            }
            GGML_LOG_INFO("expert cache: verified %zu VRAM and %zu host slices against the GGUF bytes "
                          "(%zu without a digest) in %.1f ms\n", vram, host, skipped,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            return;
        }
        const long checked = l1_->verify_resident(0);
        if (checked < 0) {
            GGML_ABORT("expert cache: a resident slice differs from its host tensor");
        }
        GGML_LOG_INFO("expert cache: verified %ld resident slices against the host weights in %.1f ms\n", checked,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }

    // The delegate buffer of the exclusive weight buffer may be a host buffer that the backend
    // registers as coarse-grained mapped memory only after the model is loaded. llama runs that step
    // for every buffer of the model, but the delegate is owned by the exclusive buffer and is not in
    // that list, so it is run here instead. Buffers that need nothing answer true.
    bool finalize_delegate_buffer_locked(ggml_backend_buffer_t exclusive_buffer) {
        ggml_backend_buffer_t delegate = exclusive_buffer != nullptr ?
            exclusive_delegate_buffer(exclusive_buffer) : nullptr;
        if (delegate == nullptr) {
            return true;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(delegate));
        if (dev == nullptr) {
            return true;
        }
        using finalize_host_buffer_t = bool (*)(ggml_backend_buffer_t);
        finalize_host_buffer_t finalize_host = (finalize_host_buffer_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_finalize_host_buffer");
        return finalize_host == nullptr || finalize_host(delegate);
    }

    size_t resident_slices_locked() const {
        size_t count = 0;
        for (const std::vector<int32_t> & layer : l1_->selected()) {
            count += layer.size();
        }
        return count;
    }

    static bool any_capacity(const std::vector<int> & capacities) {
        for (int c : capacities) { if (c > 0) { return true; } }
        return false;
    }

    void log_plan_locked(const placement_stats & stats) const {
        GGML_LOG_INFO("expert cache: plan budget=%zu MiB choices=%d per-layer=%d..%d selection-hit=%.2f%% byte-hit=%.2f%%%s\n",
            cfg_.l1_bytes/(1024*1024), stats.choices, stats.per_layer_min, stats.per_layer_max,
            100.0*stats.selection_hit(), 100.0*stats.byte_hit(), models_text_locked().c_str());
    }

    // Free slots that the owned-host exchange rotates through.
    size_t spare_bytes_locked() const {
        size_t bytes = 0;
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            bytes += size_t(cfg_.spare_slots)*geo_.class_total_bytes(cls);
        }
        return bytes;
    }

    // The host arena's spare slots; with the class layout at the storage pitch.
    size_t host_spare_bytes_locked() const {
        if (!tier_ || !tier_->class_layout()) { return spare_bytes_locked(); }
        size_t bytes = 0;
        for (size_t cls = 0; cls < host_pitch_.size(); ++cls) { bytes += size_t(cfg_.spare_slots)*host_pitch_[cls]; }
        return bytes;
    }

    // Owned host storage keeps a second slot table, for the host arena.
    size_t table_bytes_locked(bool owns_host) const {
        return (owns_host ? 2 : 1)*geo_.n_counts()*sizeof(int32_t);
    }

    // Everything inside the budget that is not an expert slice.
    size_t budget_overhead_locked(bool owns_host) const {
        return table_bytes_locked(owns_host) + arena_tail_total(geo_) +
            (owns_host ? spare_bytes_locked() : 0) +
            (profiling_enabled() ? geo_.n_counts()*sizeof(uint32_t)*max_banks : 0);
    }

    bool profiling_enabled() const { return cfg_.policy != GGML_EXPERT_POLICY_OFF && !profile_dir_.empty(); }
    bool allocate_profiler_locked() {
        if (!profiling_enabled()) { return true; }
        profiler_.reset(new profiler(geo_, max_banks, (uint32_t) std::max<size_t>(members_.size(), 1)));
        if (profiler_->allocate(device_)) { return true; }
        profiler_.reset(); disable_locked("profiler allocation failed"); return false;
    }

    // ---- early routes (expert-hash-early.h, expert-hash-stage.cuh) --------------------------------

    // RANMA_EXPERT_HASH_EARLY = 0 (default) | ssd | vram | both, read once per process, and the
    // early-route layers the first model declares. 0 changes nothing: route_hint answers false, no
    // slot is carved, no kernel is added.
    void configure_hash_early_locked(const ggml_expert_config * config) {
        hash_mode_ = hash_early_mode();
        early_decl_.clear();
        stage_slots_ = 0;
        for (uint32_t i = 0; config->early_route_layers != nullptr && i < config->n_early_route_layers; ++i) {
            if (config->early_route_layers[i] >= 0) {
                early_decl_.push_back(config->early_route_layers[i]);
            }
        }
        const char * env = getenv("RANMA_EXPERT_HASH_EARLY");
        if (env != nullptr && env[0] != '\0') {
            hash_early_mode mode;
            if (parse_hash_early(env, mode)) {
                hash_mode_ = mode;
            } else {
                GGML_LOG_WARN("expert cache: ignoring invalid RANMA_EXPERT_HASH_EARLY='%s', expected 0, ssd, vram or both\n", env);
            }
        }
        if (!hash_mode_.any()) {
            return;
        }
        if (early_decl_.empty()) {
            GGML_LOG_INFO("expert cache: RANMA_EXPERT_HASH_EARLY=%s: the model routes no layer by token id; nothing to do\n",
                hash_early_name(hash_mode_));
            hash_mode_ = hash_early_mode();
            return;
        }
        if (hash_mode_.vram) {
            stage_slots_ = 6;
            if (const char * text = getenv("RANMA_EXPERT_HASH_STAGE_SLOTS"); text != nullptr && text[0] != '\0') {
                char * end = nullptr;
                errno = 0;
                const long n = strtol(text, &end, 10);
                if (errno == 0 && end != text && *end == '\0' && n >= 1 && n <= stage_max_list) {
                    stage_slots_ = (int) n;
                } else {
                    GGML_LOG_WARN("expert cache: ignoring invalid RANMA_EXPERT_HASH_STAGE_SLOTS='%s', expected 1..%d\n",
                        text, stage_max_list);
                }
            }
            const char * why = nullptr;
            if (cfg_.policy == GGML_EXPERT_POLICY_OFF || cfg_.l1_bytes == 0) {
                why = "the cache has no VRAM budget";
            } else if (cfg_.mode != GGML_EXPERT_MODE_EXCLUSIVE && cfg_.l2_bytes == 0) {
                why = "it needs owned host storage (exclusive mode or a finite host tier)";
            } else if (early_decl_.size() > 8) {
                why = "more than 8 early-route layers";
            }
            if (why != nullptr) {
                GGML_LOG_WARN("expert cache: RANMA_EXPERT_HASH_EARLY=%s: VRAM staging off: %s\n", hash_early_name(hash_mode_), why);
                hash_mode_.vram = false;
                stage_slots_ = 0;
            }
        }
        std::string layers;
        for (int32_t l : early_decl_) {
            layers += (layers.empty() ? "" : ",") + std::to_string(l);
        }
        const std::string staging = hash_mode_.vram ? std::to_string(stage_slots_) + " slots per layer" : std::string("off");
        GGML_LOG_INFO("expert cache: RANMA_EXPERT_HASH_EARLY=%s: early-route layers %s; early SSD reads %s, VRAM staging %s\n",
            env, layers.c_str(), hash_mode_.ssd ? "on (finite host tier only)" : "off", staging.c_str());
    }

    // [joint layer] 1 for the early-route layers of the first model that the cache routes.
    std::vector<uint8_t> early_joint_locked() const {
        std::vector<uint8_t> out((size_t) std::max(geo_.n_layers, 0), 0);
        if (early_decl_.empty() || members_.empty()) {
            return out;
        }
        const int offset = members_[0].offset;
        const int n = joint_ ? members_[0].geo.n_layers : geo_.n_layers;
        for (int32_t l : early_decl_) {
            const int joint = offset + l;
            if (l < n && joint < geo_.n_layers && geo_.layer_class[joint] >= 0) {
                out[(size_t) joint] = 1;
            }
        }
        return out;
    }

    // Load time, right after the load-time plan froze the capacities: every early-route layer takes
    // stage_slots_ slots of its size class for its VRAM staging. The static plans keep the rest; the
    // slots sit after the spares in the arena and belong to no plan. Returns true when capacities_
    // changed.
    bool carve_stage_locked() {
        stage_class_slots_.clear();
        stage_first_.assign((size_t) geo_.n_layers, -1);
        stage_index_.assign((size_t) geo_.n_layers, -1);
        stage_layer_.clear();
        if (!hash_mode_.vram || stage_slots_ <= 0) {
            return false;
        }
        const std::vector<uint8_t> early = early_joint_locked();
        std::vector<int> per_class(geo_.class_bytes.size(), 0);
        size_t bytes = 0;
        for (int l = 0; l < geo_.n_layers; ++l) {
            if (!early[(size_t) l]) {
                continue;
            }
            const int cls = geo_.layer_class[l];
            if (capacities_[(size_t) cls] - per_class[(size_t) cls] < stage_slots_) {
                GGML_LOG_WARN("expert cache: VRAM staging: size class %d has fewer than %d static slots left; layer %d is not staged\n",
                    cls, stage_slots_, l);
                continue;
            }
            stage_index_[(size_t) l] = (int) stage_layer_.size();
            stage_layer_.push_back(l);
            per_class[(size_t) cls] += stage_slots_;
            bytes += size_t(stage_slots_)*geo_.class_total_bytes(cls);
        }
        if (stage_layer_.empty()) {
            hash_mode_.vram = false;
            return false;
        }
        for (size_t cls = 0; cls < per_class.size(); ++cls) {
            capacities_[cls] -= per_class[cls];
        }
        stage_class_slots_ = per_class;
        GGML_LOG_INFO("expert cache: VRAM staging of early-route layers: %d slots x %zu layers = %.1f MiB taken from the "
                      "static VRAM capacity\n", stage_slots_, stage_layer_.size(), double(bytes)/(1024.0*1024.0));
        return true;
    }

    // After the arena exists, before any graph capture.
    void allocate_stage_locked() {
        if (stage_layer_.empty() || !l1_) {
            return;
        }
        std::vector<int> next(geo_.class_bytes.size(), 0);
        for (int l : stage_layer_) {
            const int cls = geo_.layer_class[l];
            stage_first_[(size_t) l] = l1_->stage_base(cls) + next[(size_t) cls];
            next[(size_t) cls] += stage_slots_;
        }
        const size_t n = stage_layer_.size();
        if (cudaMalloc((void **) &stage_dev_, n*sizeof(stage_state)) != cudaSuccess) {
            (void) cudaGetLastError();
            stage_dev_ = nullptr;
            GGML_LOG_WARN("expert cache: VRAM staging state allocation failed; staging off (its slots stay unused)\n");
            hash_mode_.vram = false;
            return;
        }
        CUDA_CHECK(cudaMemset(stage_dev_, 0, n*sizeof(stage_state)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stage_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&stage_event_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&stage_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaDeviceSynchronize());
        stage_owner_.assign(n, std::vector<int32_t>((size_t) stage_slots_, -1));
        stage_gen_.assign(n, 0);
        stage_last_.assign(n, stage_state{});
        stage_hints_ = stage_issues_ = stage_copies_ = stage_bytes_ = stage_kept_ = stage_over_ = stage_verified_ = 0;
        stage_us_ = 0.0;
    }

    void free_stage_locked() {
        if (stage_stream_ != nullptr) {
            (void) cudaStreamSynchronize(stage_stream_);
            (void) cudaStreamDestroy(stage_stream_);
            stage_stream_ = nullptr;
        }
        if (stage_event_ != nullptr) {
            (void) cudaEventDestroy(stage_event_);
            stage_event_ = nullptr;
        }
        if (stage_done_ != nullptr) {
            (void) cudaEventDestroy(stage_done_);
            stage_done_ = nullptr;
        }
        if (stage_dev_ != nullptr) {
            (void) cudaFree(stage_dev_);
            stage_dev_ = nullptr;
        }
        stage_class_slots_.clear();
        stage_first_.clear();
        stage_index_.clear();
        stage_layer_.clear();
        stage_owner_.clear();
        stage_gen_.clear();
        stage_last_.clear();
    }

    // One hint's VRAM staging (see expert-hash-stage.cuh): the hinted experts of each staged layer
    // that live in host memory go to the layer's staging slots.
    void stage_hint_locked(ggml_backend_cuda_context & ctx, const std::vector<std::pair<int, std::vector<int>>> & layers) {
        const auto t0 = std::chrono::steady_clock::now();
        ++stage_hints_;
        struct work {
            int idx, layer;
            stage_plan plan;
        };
        std::vector<work> works;
        const std::vector<std::vector<int32_t>> & vram = l1_->host_slots();
        const expert_locations & homes = l1_->locations();
        for (const auto & l : layers) {
            const int idx = stage_index_[(size_t) l.first];
            if (idx < 0) {
                continue;
            }
            std::vector<int> wanted;
            for (int e : l.second) {
                if (vram[(size_t) l.first][(size_t) e] < 0 && (size_t) l.first < homes.size() &&
                        homes[(size_t) l.first][(size_t) e].storage == expert_storage::host) {
                    wanted.push_back(e);
                }
            }
            stage_plan plan = plan_stage(stage_owner_[(size_t) idx], wanted);
            stage_over_ += (uint64_t) plan.over;
            stage_kept_ += (uint64_t) plan.keeps.size();
            if (!plan.changed()) {
                continue;
            }
            works.push_back({idx, l.first, std::move(plan)});
        }
        if (works.empty()) {
            return;
        }
        ggml_cuda_set_device(device_);
        const cudaStream_t cs = ctx.stream();
        stage_targets targets;
        for (const work & w : works) {
            targets.index[targets.n] = w.idx;
            targets.value[targets.n] = ++stage_gen_[(size_t) w.idx];
            ++targets.n;
        }
        const ggml_cuda_kernel_launch_params one(dim3(1), dim3(stage_max_list), 0, cs);
        ggml_cuda_kernel_launch(stage_target_kernel, one, stage_dev_, targets);
        // nothing below may start before every graph launched so far has finished with the slots
        CUDA_CHECK(cudaEventRecord(stage_event_, cs));
        CUDA_CHECK(cudaStreamWaitEvent(stage_stream_, stage_event_, 0));
        const ggml_cuda_kernel_launch_params side(dim3(1), dim3(stage_max_list), 0, stage_stream_);
        for (size_t i = 0; i < works.size(); ++i) {
            const work & w = works[i];
            const int cls = geo_.layer_class[w.layer];
            const int first = stage_first_[(size_t) w.layer];
            int32_t * table = const_cast<int32_t *>(l1_->lookup(w.layer, 0).slots);
            if (!w.plan.clears.empty()) {
                stage_list clears;
                for (const auto & c : w.plan.clears) {
                    clears.expert[clears.n] = c.first;
                    clears.slot[clears.n]   = first + c.second;
                    ++clears.n;
                }
                ggml_cuda_kernel_launch(stage_clear_kernel, side, table, clears);
            }
            for (const auto & c : w.plan.copies) {
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    const void * src = l1_->host_address(w.layer, kind, c.first);
                    GGML_ASSERT(src != nullptr);
                    const size_t bytes = geo_.class_bytes[(size_t) cls][kind];
                    CUDA_CHECK(cudaMemcpyAsync(l1_->gpu_slice_address(cls, kind, first + c.second), src, bytes,
                        cudaMemcpyHostToDevice, stage_stream_));
                    stage_bytes_ += bytes;
                }
            }
            stage_copies_ += (uint64_t) w.plan.copies.size();
            stage_list sets;
            for (const auto & c : w.plan.keeps) {
                sets.expert[sets.n] = c.first;
                sets.slot[sets.n]   = first + c.second;
                ++sets.n;
            }
            for (const auto & c : w.plan.copies) {
                sets.expert[sets.n] = c.first;
                sets.slot[sets.n]   = first + c.second;
                ++sets.n;
            }
            ggml_cuda_kernel_launch(stage_publish_kernel, side, table, sets, stage_dev_ + w.idx, targets.value[i]);
        }
        // A submit point for the side stream: without it the runtime may hold the queued commands on
        // the host while the graph's wait kernel spins for them (seen in the fixture as a hang).
        CUDA_CHECK(cudaEventRecord(stage_done_, stage_stream_));
        ++stage_issues_;
        if (verify_all_) {
            verify_stage_locked();
        }
        stage_us_ += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    }

    // RANMA_EXPERT_VERIFY: after the staging is done, every staging slot the table names holds exactly
    // its expert's host bytes, and the table names every slot the host believes it does.
    void verify_stage_locked() {
        CUDA_CHECK(cudaStreamSynchronize(stage_stream_));
        std::vector<char> back;
        for (size_t idx = 0; idx < stage_layer_.size(); ++idx) {
            const int layer = stage_layer_[idx];
            const int cls = geo_.layer_class[layer];
            std::vector<int32_t> table((size_t) geo_.n_experts);
            CUDA_CHECK(cudaMemcpy(table.data(), l1_->lookup(layer, 0).slots, table.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::string why;
            if (!stage_table_check(stage_owner_[idx], table, stage_first_[(size_t) layer], why)) {
                GGML_ABORT("expert cache: VRAM staging of layer %d: %s", layer, why.c_str());
            }
            for (int k = 0; k < stage_slots_; ++k) {
                const int e = stage_owner_[idx][(size_t) k];
                const int slot = stage_first_[(size_t) layer] + k;
                if (e < 0) {
                    continue;
                }
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    const size_t bytes = geo_.class_bytes[(size_t) cls][kind];
                    back.resize(bytes);
                    CUDA_CHECK(cudaMemcpy(back.data(), l1_->gpu_slice_address(cls, kind, slot), bytes, cudaMemcpyDeviceToHost));
                    if (memcmp(back.data(), l1_->host_address(layer, kind, e), bytes) != 0) {
                        GGML_ABORT("expert cache: VRAM staging of layer %d expert %d kind %d differs from its host bytes", layer, e, kind);
                    }
                    ++stage_verified_;
                }
            }
        }
    }

    // Per generation bank commit, for a model that declares early-route layers (also with the switch
    // off, for the comparison): the decode wait of those layers and the early read counters, cumulative.
    void log_early_counters_locked(const std::string & label) {
        const l2_counters c = tier_->counters_snapshot();
        const double khz = tier_->steady_khz() > 0 ? tier_->steady_khz() : 1.0;
        GGML_LOG_INFO("expert_metrics {\"kind\":\"hash_early\",\"bank\":\"%s\",\"mode\":\"%s\",\"decode_tokens\":%llu,"
                      "\"decode_wait_ms\":%.3f,\"early_layer_decode_wait_ms\":%.3f,\"early_layer_samples\":%llu,"
                      "\"reads\":%llu,\"ssd_bytes\":%llu,\"ring_hits\":%llu,\"early_jobs\":%llu,\"early_layers\":%llu,"
                      "\"early_late\":%llu,\"early_reads\":%llu,\"early_bytes\":%llu,\"early_already\":%llu,"
                      "\"early_hits\":%llu,\"early_unused\":%llu,\"early_lost\":%llu,\"early_inline\":%llu,\"verify_bad\":%llu}\n",
            label.c_str(), hash_early_name(hash_mode_), (unsigned long long) c.decode_tokens,
            double(c.decode_wait_ticks)/khz, double(c.early_layer_wait_ticks)/khz, (unsigned long long) c.early_layer_samples,
            (unsigned long long) c.ssd_reads, (unsigned long long) c.ssd_bytes, (unsigned long long) c.ring_hits,
            (unsigned long long) c.early_jobs, (unsigned long long) c.early_layers, (unsigned long long) c.early_late,
            (unsigned long long) c.early_reads, (unsigned long long) c.early_bytes, (unsigned long long) c.early_already,
            (unsigned long long) c.early_hits, (unsigned long long) c.early_unused, (unsigned long long) c.early_lost,
            (unsigned long long) c.early_inline, (unsigned long long) c.verify_bad);
    }

    // Per bank commit (the device is synchronized): the staging counters since the last commit.
    void log_stage_counters_locked(const std::string & label) {
        std::vector<stage_state> now(stage_layer_.size());
        CUDA_CHECK(cudaMemcpy(now.data(), stage_dev_, now.size()*sizeof(stage_state), cudaMemcpyDeviceToHost));
        std::string layers;
        for (size_t i = 0; i < now.size(); ++i) {
            const stage_state & a = now[i];
            const stage_state & b = stage_last_[i];
            layers += std::string(i ? "," : "") + "{\"layer\":" + std::to_string(stage_layer_[i]) +
                ",\"hits\":" + std::to_string(a.hits - b.hits) + ",\"host\":" + std::to_string(a.host - b.host) +
                ",\"waits\":" + std::to_string(a.waits - b.waits) + ",\"launches\":" + std::to_string(a.launches - b.launches) + "}";
        }
        GGML_LOG_INFO("expert_metrics {\"kind\":\"hash_stage\",\"bank\":\"%s\",\"slots\":%d,\"hints\":%llu,\"issues\":%llu,"
                      "\"copies\":%llu,\"bytes\":%llu,\"kept\":%llu,\"over\":%llu,\"verified\":%llu,\"host_us\":%.1f,\"layers\":[%s]}\n",
            label.c_str(), stage_slots_, (unsigned long long) stage_hints_, (unsigned long long) stage_issues_,
            (unsigned long long) stage_copies_, (unsigned long long) stage_bytes_, (unsigned long long) stage_kept_,
            (unsigned long long) stage_over_, (unsigned long long) stage_verified_, stage_us_, layers.c_str());
        stage_last_ = now;
    }

    // Exclusive: the whole budget is planned and allocated before the loader writes, so that every
    // slice can be written straight to its final home (design section 6).
    // Anything that fails here leaves the cache disabled and the model loading uncached.
    bool prepare_exclusive_locked() {
        ggml_cuda_set_device(device_);

        const size_t table_bytes = table_bytes_locked(true);
        const size_t overhead = budget_overhead_locked(true);
        if (cfg_.l1_bytes > 0 && cfg_.l1_bytes <= overhead) {
            disable_locked("budget smaller than the cache tables and spare slots");
            return false;
        }

        std::vector<uint64_t> scores;
        const uint64_t * counts = seed_counts_locked(scores);
        if (counts != nullptr) {
            tier_scores_ = scores;
        }

        placement_inputs in;
        in.geo          = &geo_;
        in.counts       = counts;
        in.budget_bytes = cfg_.l1_bytes == 0 ? 0 : cfg_.l1_bytes - overhead;
        in.exclusive    = true; // every VRAM slot is filled, zero-count experts included
        placement initial = plan_placement(in);
        if (!any_capacity(initial.capacities) && cfg_.l1_bytes > 0) {
            disable_locked("budget holds no expert slice");
            return false;
        }
        capacities_ = initial.capacities;
        if (carve_stage_locked()) {
            // the static plan keeps the capacity the staging slots leave it; the host capacities below
            // count the carved experts too
            placement_inputs fixed = in;
            fixed.fixed_capacities = &capacities_;
            initial = plan_placement(fixed);
        }

        std::vector<int> host_capacities(geo_.class_bytes.size(), 0);
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            host_capacities[cls] = geo_.class_layers[cls]*geo_.n_experts - capacities_[cls];
            if (host_capacities[cls] < 0) {
                disable_locked("more VRAM slots than experts");
                return false;
            }
        }
        if (cfg_.l2_bytes != 0 && !size_tier_locked(initial.selected, host_capacities)) {
            return false;
        }

        // Plan time, before the first expert byte is copied: a host requirement larger than the
        // installed memory can never be met, so it is a configuration error, not a slow run.
        size_t host_required = cfg_.l2_bytes;
        if (host_required == 0) {
            for (size_t cls = 0; cls < host_capacities.size(); ++cls) {
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    host_required += size_t(host_capacities[cls] + cfg_.spare_slots)*geo_.class_bytes[cls][kind] +
                        arena_tail_bytes(geo_, (int) cls, kind);
                }
            }
        }
        const auto memory = check_host_memory(host_required, expert_os::total_physical_bytes());
        if (!memory.ok) {
            GGML_ABORT("expert cache: the host tier needs %.1f GiB but the machine has %.1f GiB of physical memory; lower --expert-l2-mib or raise --expert-l1-mib",
                double(memory.required_bytes)/(1024.0*1024.0*1024.0), double(memory.total_bytes)/(1024.0*1024.0*1024.0));
        }

        if (!allocate_profiler_locked()) { return false; }
        l1_.reset(new l1_arena(geo_));
        redraw_prepare_locked();
        if (!l1_->allocate(capacities_, device_, cfg_.spare_slots, stage_class_slots_.empty() ? nullptr : &stage_class_slots_)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
        redraw_started_locked(counts == nullptr);
        allocate_stage_locked();
        host_.reset(new host_arena(geo_));
        if (!host_->allocate(host_capacities, cfg_.spare_slots, device_,
                tier_ && tier_->class_layout() ? &tier_->storage() : nullptr)) {
            host_.reset();
            l1_.reset();
            profiler_.reset();
            disable_locked("host arena allocation failed");
            return false;
        }
        l1_->attach_host(host_.get(), cfg_.spare_slots, cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE);
        if (tier_) {
            if (!allocate_tier_locked(initial.selected)) {
                return false;
            }
        } else if (!l1_->assign_exclusive(initial.selected)) {
            host_.reset();
            l1_.reset();
            profiler_.reset();
            disable_locked("exclusive slot assignment failed");
            return false;
        }
        if (verify_all_) {
            digests_enabled_ = true;
            digests_.assign(size_t(geo_.n_layers)*geometry::n_kinds*geo_.n_experts, slice_digest());
            digest_known_.assign(digests_.size(), 0);
        }
        if (counts != nullptr) { log_plan_locked(initial.stats); }
        GGML_LOG_INFO("expert cache: owned host arenas ready: VRAM %zu MiB, host %zu MiB, tables and profiler %zu KiB\n",
            l1_->device_bytes()/(1024*1024), host_->host_bytes()/(1024*1024),
            (table_bytes + (profiler_ ? profiler_->device_bytes() : 0))/1024);
        return true;
    }

    size_t digest_index_locked(int layer, int kind, int expert) const {
        return (size_t(layer)*geometry::n_kinds + size_t(kind))*size_t(geo_.n_experts) + size_t(expert);
    }

    // Records the digest of the bytes the loader writes. A write that does not cover whole expert
    // slices leaves the touched experts without a digest, and verify reports them as skipped rather
    // than pretending to have checked them.
    void record_digest_locked(int layer, int kind, size_t offset, size_t size, const void * data) {
        const int cls = geo_.layer_class[layer];
        if (cls < 0) {
            return;
        }
        const size_t stride = geo_.class_bytes[cls][kind];
        if (stride == 0) {
            return;
        }
        const bool aligned = offset % stride == 0 && size % stride == 0;
        const int  first   = int(offset/stride);
        const int  last    = int((offset + size + stride - 1)/stride);
        for (int expert = first; expert < last && expert < geo_.n_experts; ++expert) {
            const size_t index = digest_index_locked(layer, kind, expert);
            if (!aligned) {
                digest_known_[index] = 0;
                continue;
            }
            digests_[index] = digest_of(static_cast<const char *>(data) + (size_t(expert) - first)*stride, stride);
            digest_known_[index] = 1;
        }
    }

    // Compares every VRAM slot and every host slot with the digest of the bytes the loader wrote.
    bool verify_exclusive_locked(size_t & vram, size_t & host, size_t & skipped) {
        if (digests_.empty()) { return true; }
        std::vector<uint8_t> back;
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0) { continue; }
            // a joined model that has not loaded yet (or was dropped) has no bytes and no file backing
            if (joint_) {
                const member_state & mb = members_[(size_t) member_of_layer(l)];
                if (!mb.loaded || mb.dropped) { continue; }
            }
            for (int e = 0; e < geo_.n_experts; ++e) for (int k = 0; k < geometry::n_kinds; ++k) {
                const int gpu = l1_->host_slots()[l][e];
                const void * home = l1_->host_address(l, k, e);
                if ((gpu < 0 && !home) || (!digest_known_[digest_index_locked(l, k, e)] && !tier_)) { skipped += k == 0; continue; }
                auto check = [&](const void * data) {
                    const size_t i = digest_index_locked(l, k, e);
                    const auto found = digest_of(data, geo_.class_bytes[cls][k]);
                    if (!digest_known_[i]) {
                        std::string reason;
                        if (!tier_->verify_resident(l, k, e, data, reason)) { GGML_LOG_ERROR("expert cache: %s\n", reason.c_str()); return false; }
                        digests_[i] = found; digest_known_[i] = 1;
                    }
                    if (digests_[i] == found) { return true; }
                    GGML_LOG_ERROR("expert cache: layer %d kind %d expert %d differs from GGUF bytes\n", l, k, e); return false;
                };
                if (gpu >= 0) {
                    back.resize(geo_.class_bytes[cls][k]);
                    if (!l1_->read_slice(l, k, gpu, back.data()) || !check(back.data())) { return false; }
                    vram += k == 0;
                }
                if (home) { if (!check(home)) { return false; } host += k == 0; }
            }
        }
        return true;
    }

    bool bank_ok_locked(ggml_expert_bank_id bank) const {
        return state_ == state_t::installed && bank < banks_.size() && profiler_ != nullptr;
    }

    // What the interval just committed found in the VRAM arena. `delta` is the bank histogram of
    // the interval, indexed [layer*n_experts + expert]; the plan that was installed while the
    // interval ran is exactly l1_->selected(). This is the only measurement of what a plan buys
    // during prompt processing, where the kernels read resident experts from the arena and
    // everything else across PCIe, so it is reported per commit rather than per kernel call: the
    // histogram is already in VRAM and the commit already reads it back.
    void report_round_locked(const std::string & label, uint64_t round, const ggml_expert_record & record,
            const std::vector<uint64_t> & delta, double save_ms) const {
        uint64_t vram = 0, host = 0, file = 0;
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0) { continue; }
            for (int e = 0; e < geo_.n_experts; ++e) {
                const uint64_t bytes = delta[size_t(l)*geo_.n_experts + e]*geo_.class_total_bytes(cls);
                if (l1_->host_slots()[l][e] >= 0) { vram += bytes; }
                else if (!tier_ || tier_->locations()[l][e].resident()) { host += bytes; }
                else { file += bytes; }
            }
        }
        GGML_LOG_INFO("expert_metrics {\"kind\":\"round\",\"bank\":\"%s\",\"round\":%llu,\"tokens\":%llu,"
                      "\"vram_bytes\":%llu,\"host_bytes\":%llu,\"file_bytes\":%llu,\"profile_save_ms\":%.6f}\n",
            label.c_str(), (unsigned long long) round, (unsigned long long) record.bank_tokens,
            (unsigned long long) vram, (unsigned long long) host, (unsigned long long) file, save_ms);
    }

    // The stored profile that seeds the plan installed at model load. `initial_bank` is a
    // preference list of bank labels separated by commas; the first one that has stored records
    // wins, so the caller expresses "the plan of the next phase, and the other bank if that one is
    // still empty" without the backend knowing what a phase is. Returns null when no bank has a
    // profile, which means the arenas start cold.
    const uint64_t * seed_counts_locked(std::vector<uint64_t> & scores) {
        if (cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE) {
            scores = seeded_expert_scores(geo_, cfg_.random_seed);
            uint64_t hash = UINT64_C(1469598103934665603);
            for (uint64_t score : scores) { hash = (hash ^ score)*UINT64_C(1099511628211); }
            GGML_LOG_INFO("expert cache: fixed seeded placement policy=%d seed=%u scores=%016llx\n",
                (int) cfg_.policy, cfg_.random_seed, (unsigned long long) hash);
            return scores.data();
        }
        if (initial_bank_.empty() || profile_dir_.empty()) {
            return nullptr;
        }
        if (joint_) {
            return seed_joint_counts_locked(scores);
        }
        for (size_t pos = 0; pos <= initial_bank_.size(); ) {
            const size_t sep = std::min(initial_bank_.find(',', pos), initial_bank_.size());
            const std::string label = initial_bank_.substr(pos, sep - pos);
            pos = sep + 1;
            if (label.empty()) {
                continue;
            }
            ggml_expert_bank_id bank = GGML_EXPERT_BANK_NONE;
            // A seed bank is opened before the caller names its phase; init reopens it by label.
            if (open_bank_locked(label.c_str(), false, &bank) && banks_[bank].any_usable() &&
                    banks_[bank].total_selections() != 0) {
                joint_scores_locked(banks_[bank], scores);
                GGML_LOG_INFO("expert cache: bank '%s' seeds the plan from %zu stored records\n",
                    label.c_str(), banks_[bank].records());
                if (joint_) {
                    for (size_t m = 0; m < members_.size(); ++m) {
                        const profile_store * st = banks_[bank].stores[m].get();
                        GGML_LOG_INFO("expert cache: bank '%s' model %zu %s: %zu records, %llu selections, weight %.3f\n",
                            label.c_str(), m, members_[m].key.c_str(), st && banks_[bank].usable[m] ? st->window().size() : 0,
                            (unsigned long long) (st && banks_[bank].usable[m] ? st->total_selections() : 0), (double) members_[m].weight);
                    }
                }
                return scores.data();
            }
            GGML_LOG_INFO("expert cache: bank '%s' has no usable profile\n", label.c_str());
        }
        GGML_LOG_INFO("expert cache: no bank of '%s' has a stored profile; starting cold\n", initial_bank_.c_str());
        return nullptr;
    }

    // Joint cache: each model is seeded from the first bank of the preference list that has stored
    // records for it, so a draft model that only runs in generation is seeded from its generation
    // bank while the target is seeded from the prompt bank when the prefill swap is on.
    const uint64_t * seed_joint_counts_locked(std::vector<uint64_t> & scores) {
        std::vector<std::string> labels;
        for (size_t pos = 0; pos <= initial_bank_.size(); ) {
            const size_t sep = std::min(initial_bank_.find(',', pos), initial_bank_.size());
            const std::string label = initial_bank_.substr(pos, sep - pos);
            pos = sep + 1;
            if (!label.empty()) {
                labels.push_back(label);
            }
        }
        std::vector<const profile_store *> sources(members_.size(), nullptr);
        bool any = false;
        for (size_t m = 0; m < members_.size(); ++m) {
            std::string from;
            uint64_t selections = 0;
            size_t records = 0;
            for (const std::string & label : labels) {
                ggml_expert_bank_id bank = GGML_EXPERT_BANK_NONE;
                if (!open_bank_locked(label.c_str(), false, &bank)) {
                    continue;
                }
                const bank_state & bs = banks_[bank];
                if (m < bs.stores.size() && bs.stores[m] && bs.usable[m] && bs.stores[m]->total_selections() != 0) {
                    sources[m] = bs.stores[m].get(); // owned by a unique_ptr: stable while banks_ grows
                    selections = sources[m]->total_selections();
                    records    = sources[m]->window().size();
                    from       = bs.label;
                    break;
                }
            }
            any = any || selections != 0;
            GGML_LOG_INFO("expert cache: model %zu %s seeds the plan from bank '%s': %zu records, %llu selections, weight %.3f\n",
                m, members_[m].key.c_str(), from.empty() ? "(none)" : from.c_str(), records, (unsigned long long) selections,
                (double) members_[m].weight);
        }
        if (!any) {
            GGML_LOG_INFO("expert cache: no bank of '%s' has a stored profile; starting cold\n", initial_bank_.c_str());
            return nullptr;
        }
        assemble_joint_scores_locked(sources, scores, "seed");
        return scores.data();
    }

    // The scores of a bank over the joint layers: each model's own half-life scores at its layers,
    // times its weight (the relative cost of reading one of its bytes from host memory). The
    // greedy then ranks every (model, layer, expert) by weighted frequency per byte. A cache of one
    // model uses its scores unchanged.
    void joint_scores_locked(const bank_state & b, std::vector<uint64_t> & out) const {
        if (!joint_) {
            if (!b.stores.empty() && b.stores[0] && b.usable[0]) {
                out = b.stores[0]->scores();
            } else {
                out.assign(geo_.n_counts(), 0);
            }
            return;
        }
        assemble_joint_scores_locked(joint_sources_locked(b), out, nullptr);
    }

    // The usable store of every member in bank b (null where there is none).
    std::vector<const profile_store *> joint_sources_locked(const bank_state & b) const {
        std::vector<const profile_store *> out(members_.size(), nullptr);
        for (size_t m = 0; m < members_.size() && m < b.stores.size(); ++m) {
            if (b.stores[m] && b.usable[m]) {
                out[m] = b.stores[m].get();
            }
        }
        return out;
    }

    // Selections per record of a store: the half-life weighted mean of the record totals over the
    // records its scores use (same window and weights as score_records).
    static double selection_rate(const profile_store & store) {
        const std::vector<profile_record> & records = store.window();
        const scoring_params & sp = store.params().scoring;
        const size_t window = sp.window > 0 ? (size_t) sp.window : records.size();
        const size_t used   = std::min(window, records.size());
        const size_t first  = records.size() - used;
        double sum = 0.0, weights = 0.0;
        for (size_t i = 0; i < used; ++i) {
            const double w = std::exp2(-(double) (used - 1 - i)/sp.half_life_turns);
            uint64_t total = 0;
            for (uint64_t v : records[first + i].counts) {
                total += v;
            }
            sum     += w*(double) total;
            weights += w;
        }
        return weights > 0.0 ? sum/weights : 0.0;
    }

    // Joint scores. Each model's stored scores are normalized to a total of score_scale, so they
    // are brought back to an absolute rate first: score x (selections per record) x weight. The
    // greedy then ranks every (model, layer, expert) by expected selections per request x weight
    // per byte. Only the ratios between the models matter, so every model is divided by the same
    // sum of rate x weight: the joint total stays at score_scale, the range a single model's
    // scores have, and nothing the planners sum or multiply by bytes can overflow uint64.
    void assemble_joint_scores_locked(const std::vector<const profile_store *> & sources, std::vector<uint64_t> & out,
            const char * what) const {
        out.assign(geo_.n_counts(), 0);
        std::vector<double> factor(members_.size(), 0.0), rate(members_.size(), 0.0);
        double total = 0.0;
        for (size_t m = 0; m < members_.size() && m < sources.size(); ++m) {
            if (sources[m] == nullptr || sources[m]->total_selections() == 0) {
                continue;
            }
            rate[m]   = selection_rate(*sources[m]);
            factor[m] = rate[m]*(double) members_[m].weight;
            total    += factor[m];
        }
        for (size_t m = 0; m < members_.size() && m < sources.size(); ++m) {
            if (factor[m] <= 0.0 || total <= 0.0) {
                continue;
            }
            const double f = factor[m]/total;
            const std::vector<uint64_t> & part = sources[m]->scores();
            const size_t begin = size_t(members_[m].offset)*geo_.n_experts;
            const size_t n     = size_t(members_[m].geo.n_layers)*geo_.n_experts;
            for (size_t i = 0; i < part.size() && i < n && begin + i < out.size(); ++i) {
                out[begin + i] = (uint64_t) std::llround(double(part[i])*f);
            }
        }
        if (what != nullptr) {
            for (size_t m = 0; m < members_.size(); ++m) {
                GGML_LOG_INFO("expert cache: plan '%s' model %zu %s: %.1f selections per record, weight %.3f, share of the score mass %.4f\n",
                    what, m, members_[m].key.c_str(), rate[m], (double) members_[m].weight, total > 0.0 ? factor[m]/total : 0.0);
            }
        }
    }

    // True when no other member has a layer in one of member m's size classes.
    bool member_classes_own_locked(size_t m) const {
        const member_state & mb = members_[m];
        for (int l = 0; l < geo_.n_layers; ++l) {
            if (l >= mb.offset && l < mb.offset + mb.geo.n_layers) {
                continue;
            }
            const int cls = geo_.layer_class[l];
            for (int k = mb.offset; cls >= 0 && k < mb.offset + mb.geo.n_layers; ++k) {
                if (geo_.layer_class[k] == cls) {
                    return false;
                }
            }
        }
        return true;
    }

    // A prompt bank of a joint cache: the models that never route in prompt processing (no stored
    // selections in this bank) keep their VRAM slices when the plan is installed, and their share
    // of the plan is ranked with the scores of their generation bank, so that the host tier (which
    // the plan also cuts) keeps what generation needs.
    void keep_members_locked(const bank_state & b, std::vector<const profile_store *> & sources, std::vector<uint8_t> & keep) const {
        keep.assign(members_.size(), 0);
        for (size_t m = 0; m < members_.size(); ++m) {
            const bool silent = m >= b.stores.size() || !b.stores[m] || !b.usable[m] || b.stores[m]->total_selections() == 0;
            if (!silent || !members_[m].bound || members_[m].dropped) {
                continue;
            }
            if (!member_classes_own_locked(m)) {
                GGML_LOG_WARN("expert cache: model %s shares a size class with another model; the '%s' plan re-plans it\n",
                    members_[m].key.c_str(), b.label.c_str());
                continue;
            }
            keep[m] = 1;
            sources[m] = nullptr;
            for (const bank_state & other : banks_) {
                if (!other.prompt && m < other.stores.size() && other.stores[m] && other.usable[m] &&
                        other.stores[m]->total_selections() != 0) {
                    sources[m] = other.stores[m].get();
                    break;
                }
            }
        }
    }

    bool open_bank_locked(const char * label, bool prompt_bank, ggml_expert_bank_id * out) {
        for (size_t i = 0; i < banks_.size(); ++i) {
            if (banks_[i].label == label) {
                *out = (ggml_expert_bank_id) i;
                // The load-time seed opens a bank before the caller names its phase; the caller's
                // later open as a prompt bank makes it one, exactly as if it had been opened first.
                if (prompt_bank && !banks_[i].prompt) {
                    banks_[i].prompt = true;
                    bank_prompt_[i].store(1u, std::memory_order_release);
                }
                return true;
            }
        }
        if (banks_.size() >= max_banks) {
            GGML_LOG_WARN("expert cache: no free bank for '%s' (%u banks)\n", label, (unsigned) max_banks);
            return false;
        }
        bank_state b;
        b.label  = label;
        b.prompt = prompt_bank;
        if (!profile_dir_.empty()) {
            // one store per model at <dir>/<model key>/<bank>, alone or in a joint cache: the key,
            // signature and layer count are the model's own, so a model finds the same records either
            // way; the manifest signature keeps a store of another model or geometry from being applied
            b.stores.resize(members_.size());
            b.usable.assign(members_.size(), 0);
            for (size_t m = 0; m < members_.size(); ++m) {
                const member_state & mb = members_[m];
                profile_store_params p;
                p.bank_dir  = std::filesystem::path(profile_dir_)/mb.key/label;
                p.label     = label;
                p.signature = mb.signature;
                p.n_layers  = mb.geo.n_layers;
                p.n_experts = geo_.n_experts;
                p.archive   = cfg_.profile_archive;
                b.stores[m].reset(new profile_store(p));
                if (cfg_.profile_reset) {
                    b.stores[m]->reset();
                }
                const profile_store_status st = b.stores[m]->open();
                b.usable[m] = st != profile_store_status::incompatible && !mb.dropped;
                GGML_LOG_INFO("expert cache: bank '%s' of model %zu opens %s (%zu records)\n", label, m,
                    p.bank_dir.string().c_str(), b.usable[m] ? b.stores[m]->window().size() : (size_t) 0);
                if (st == profile_store_status::incompatible) {
                    GGML_LOG_WARN("expert cache: profile at %s belongs to another model or format; ignored (use --expert-profile-reset to replace it)\n",
                        p.bank_dir.string().c_str());
                } else if (b.stores[m]->skipped_records() != 0) {
                    GGML_LOG_WARN("expert cache: profile at %s: %zu unreadable records skipped\n",
                        p.bank_dir.string().c_str(), b.stores[m]->skipped_records());
                }
            }
        }
        banks_.push_back(std::move(b));
        *out = (ggml_expert_bank_id) (banks_.size() - 1);
        // profile_select runs on the hot path without the mutex, so its copy of the flag is a fixed
        // array of atomics instead of the growing bank vector.
        bank_prompt_[*out].store(prompt_bank ? 1u : 0u, std::memory_order_release);
        n_banks_.store((uint32_t) banks_.size(), std::memory_order_release);
        return true;
    }

    std::mutex mutex_;
    std::atomic<state_t> state_ { state_t::unconfigured };
    ggml_expert_config cfg_ = {};
    std::string profile_dir_;
    std::string initial_bank_;
    std::string identity_;
    std::string signature_;
    geometry geo_;
    std::vector<member_state> members_;   // [0] = the model that configured the cache
    bool joint_ = false;                   // more than one member; fixed before any compute
    int  accepting_ = -1;                  // member whose config was accepted and whose context is awaited
    std::array<std::atomic<const void *>, max_attached> attached_ctx_{};  // compute contexts, read without the mutex
    std::array<int, max_attached> attached_member_{};
    std::atomic<int> n_attached_{ 0 };
    ggml_context * ctx_ = nullptr;
    int device_ = -1;
    std::string disabled_reason_;
    bool verify_all_ = false;
    std::vector<int> capacities_;
    std::unique_ptr<profiler> profiler_;
    // early routes (expert-hash-early.h)
    hash_early_mode hash_mode_;                // RANMA_EXPERT_HASH_EARLY
    std::vector<int32_t> early_decl_;          // the first model's early-route layers (its own indices)
    int  stage_slots_ = 0;                     // VRAM staging slots per early layer
    std::vector<int> stage_class_slots_;       // [class] staging slots after the spares; empty = none
    std::vector<int> stage_first_;             // [joint layer] arena slot of its first staging slot, -1
    std::vector<int> stage_index_;             // [joint layer] index into stage_dev_, -1
    std::vector<int> stage_layer_;             // [index] joint layer
    std::vector<std::vector<int32_t>> stage_owner_;  // [index][slot] the expert a staging slot holds, -1
    std::vector<uint32_t> stage_gen_;          // [index] the last staging generation issued
    stage_state * stage_dev_ = nullptr;        // [index] VRAM
    cudaStream_t  stage_stream_ = nullptr;
    cudaEvent_t   stage_event_  = nullptr;     // compute stream -> side stream
    cudaEvent_t   stage_done_   = nullptr;     // recorded after each hint's side-stream work (a submit point)
    uint64_t stage_hints_ = 0, stage_issues_ = 0, stage_copies_ = 0, stage_bytes_ = 0, stage_kept_ = 0;
    uint64_t stage_over_ = 0, stage_verified_ = 0;
    double   stage_us_ = 0.0;                  // host time of the hints that issued staging work
    std::vector<stage_state> stage_last_;      // [index] device counters at the previous commit
    std::unique_ptr<l1_arena> l1_;
    std::unique_ptr<host_arena> host_;
    std::unique_ptr<l2_tier> tier_;
    std::vector<size_t> host_pitch_;
    std::vector<int>    host_capacities_base_;
    std::vector<int>    lent_capacities_, lent_starts_;
    std::vector<int>    file_cap_;   // class layout: [class] the ring slots of its storage class
    std::vector<uint64_t> tier_scores_;
    std::vector<std::vector<int32_t>> tier_vram_;
    // size-class redraw of the VRAM tier (RANMA_EXPERT_L1_REDRAW_BENEFIT/_MOVE, expert-redraw.h)
    redraw_params redraw_;
    bool          redraw_on_ = false;          // requested, the mode supports it and the arenas are VMM
    redraw_gate   redraw_gate_;
    uint32_t      redraw_epoch_ = 0;           // the split's generation; plans of an older one are placed again
    std::vector<int> redraw_max_static_;       // [class] static capacity the address ranges hold
    bool          redraw_unequal_ = false;     // exclusive installs of a redraw: unequal exchange
    uint64_t      redraw_count_ = 0;
    ggml_backend_buffer_t exclusive_buffer_ = nullptr; // owned by the model, not by the controller
    bool digests_enabled_ = false;
    std::vector<slice_digest> digests_;      // [(layer*n_kinds + kind)*n_experts + expert]
    std::vector<uint8_t>      digest_known_;
    std::vector<bank_state> banks_;
    std::array<std::atomic<uint32_t>, max_banks> bank_prompt_{};  // read by profile_select without the mutex
    std::atomic<uint32_t> n_banks_{ 0 };
    std::map<ggml_expert_plan_id, plan_state> plans_;
    ggml_expert_plan_id next_plan_id_  = 0;
    ggml_expert_plan_id installed_plan_ = GGML_EXPERT_PLAN_NONE;
};

static controller & instance() {
    static controller c;
    return c;
}

// ---- the exclusive buffer type ---------------------------------------------------------------
//
// Address-only storage: the buffer's base is an expert_os reservation with no pages behind it, so
// every routed expert tensor gets a distinct, stable logical address while its bytes live in the
// VRAM arena or in the host arena. Everything that touches tensor data goes through logical_io.

namespace {

struct exclusive_buffer_context {
    expert_os::reservation res;
    ggml_context * tensors = nullptr;
    // The weight context of the HIP host buffer type is not routed experts only: the token
    // embedding of a fully offloaded model lands in the same buffer type. Those tensors keep real
    // storage of that same buffer type, in a delegate buffer this one owns, so the cache does not
    // move them and the graph is the one a run without the cache builds.
    ggml_backend_buffer_t delegate_buffer = nullptr;
};

const char * exclusive_buft_name(ggml_backend_buffer_type_t) {
    return "ROCm_ExpertExclusive";
}

size_t exclusive_buft_alignment(ggml_backend_buffer_type_t) {
    return 128;
}

ggml_backend_buffer_t exclusive_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // Anything allocated on this buffer type that is not a routed expert weight needs real device
    // storage. A LoRA adapter tensor takes the buffer type of its base tensor (src/llama-adapter.cpp),
    // so a base tensor that lives here would otherwise hand the adapter a null allocator.
    return ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(buft->device), size);
}

void * exclusive_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<exclusive_buffer_context *>(buffer->context)->res.base;
}

void exclusive_free_buffer(ggml_backend_buffer_t buffer) {
    exclusive_buffer_context * ctx = static_cast<exclusive_buffer_context *>(buffer->context);
    if (ctx->delegate_buffer != nullptr) {
        ggml_backend_buffer_free(ctx->delegate_buffer);
    }
    expert_os::release(ctx->res);
    delete ctx;
}

void exclusive_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!instance().buffer_io(tensor, const_cast<void *>(data), offset, size, /*write =*/ true)) {
        GGML_ABORT("expert cache: exclusive write to '%s' failed", ggml_get_name(tensor));
    }
}

void exclusive_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!instance().buffer_io(tensor, data, offset, size, /*write =*/ false)) {
        GGML_ABORT("expert cache: exclusive read of '%s' failed", ggml_get_name(tensor));
    }
}

void exclusive_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
        size_t offset, size_t size) {
    std::vector<uint8_t> chunk(std::min<size_t>(size, 8*1024*1024), value);
    while (size != 0) {
        const size_t n = std::min(size, chunk.size());
        exclusive_set_tensor(buffer, tensor, chunk.data(), offset, n);
        offset += n;
        size   -= n;
    }
}

bool exclusive_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_are_same_layout(src, dst)) {
        return false;
    }
    const size_t total = ggml_nbytes(src);
    std::vector<uint8_t> chunk(std::min<size_t>(total, 8*1024*1024));
    for (size_t offset = 0; offset < total; offset += chunk.size()) {
        const size_t n = std::min(chunk.size(), total - offset);
        ggml_backend_tensor_get(src, chunk.data(), offset, n);
        exclusive_set_tensor(buffer, dst, chunk.data(), offset, n);
    }
    return true;
}

void exclusive_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    exclusive_buffer_context * ctx = static_cast<exclusive_buffer_context *>(buffer->context);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx->tensors); t != nullptr;
            t = ggml_get_next_tensor(ctx->tensors, t)) {
        if (t->buffer == buffer) { // the delegated tensors belong to the delegate buffer
            exclusive_memset_tensor(buffer, t, value, 0, ggml_nbytes(t));
        }
    }
}

ggml_backend_buffer_type g_exclusive_buft = {
    /* .iface   = */ { exclusive_buft_name, exclusive_buft_alloc_buffer, nullptr, exclusive_buft_alignment,
                       nullptr, nullptr, nullptr, nullptr },
    /* .device  = */ nullptr,
    /* .context = */ nullptr,
};

} // namespace

static bool is_routed_expert(const ggml_tensor * tensor) {
    int layer = -1;
    int kind  = -1;
    return parse_expert_tensor_name(ggml_get_name(tensor), layer, kind);
}

static ggml_backend_buffer_t exclusive_buffer_create(ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    // The tensors of this context that are not routed experts are delegated to a buffer of `buft`,
    // the buffer type the context was going to use anyway, so they keep the storage, alloc size,
    // alignment and row padding of that type and stay exactly where inclusive mode and a run
    // without the cache put them. Only the routed experts get logical addresses.
    const size_t delegate_align = ggml_backend_buft_get_alignment(buft);
    size_t logical_bytes  = 0;
    size_t delegate_bytes = 0;
    int    delegated      = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        if (is_routed_expert(t)) {
            logical_bytes += GGML_PAD(ggml_nbytes(t), 128);
        } else {
            delegate_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), delegate_align);
            ++delegated;
        }
    }
    std::optional<expert_os::reservation> res = expert_os::reserve(logical_bytes);
    if (!res) {
        return nullptr;
    }
    exclusive_buffer_context * context = new exclusive_buffer_context();
    context->res     = *res;
    context->tensors = ctx;
    if (delegate_bytes != 0) {
        context->delegate_buffer = ggml_backend_buft_alloc_buffer(buft, delegate_bytes);
        if (context->delegate_buffer == nullptr) {
            GGML_LOG_ERROR("expert cache: could not allocate %.2f MiB of %s for the %d non-expert tensors of the "
                           "routed expert context\n", delegate_bytes/1024.0/1024.0,
                           ggml_backend_buft_name(buft), delegated);
            expert_os::release(context->res);
            delete context;
            return nullptr;
        }
        ggml_backend_buffer_set_usage(context->delegate_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    g_exclusive_buft.device = buft->device;
    ggml_backend_buffer_i iface = {};
    iface.free_buffer    = exclusive_free_buffer;
    iface.get_base       = exclusive_get_base;
    iface.memset_tensor  = exclusive_memset_tensor;
    iface.set_tensor     = exclusive_set_tensor;
    iface.get_tensor     = exclusive_get_tensor;
    iface.cpy_tensor     = exclusive_cpy_tensor;
    iface.clear          = exclusive_clear;

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(&g_exclusive_buft, iface, context, logical_bytes);
    char * delegate_base = context->delegate_buffer != nullptr ?
        static_cast<char *>(ggml_backend_buffer_get_base(context->delegate_buffer)) : nullptr;
    size_t offset          = 0;
    size_t delegate_offset = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        const bool expert = is_routed_expert(t);
        ggml_backend_buffer_t owner = expert ? buffer : context->delegate_buffer;
        char * address = expert ? static_cast<char *>(context->res.base) + offset : delegate_base + delegate_offset;
        if (ggml_backend_tensor_alloc(owner, t, address) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            return nullptr;
        }
        if (expert) {
            offset += GGML_PAD(ggml_nbytes(t), 128);
        } else {
            delegate_offset += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), delegate_align);
        }
    }
    GGML_LOG_INFO("expert cache: owned host buffer: %.2f MiB of address space for the routed experts, "
                  "%.2f MiB of %s for %d other tensors of the same context\n",
        logical_bytes/1024.0/1024.0, delegate_bytes/1024.0/1024.0, ggml_backend_buft_name(buft), delegated);
    return buffer;
}

static ggml_backend_buffer_t exclusive_delegate_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || !is_exclusive_buft(buffer->buft)) {
        return nullptr;
    }
    return static_cast<exclusive_buffer_context *>(buffer->context)->delegate_buffer;
}

bool is_exclusive_buft(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == exclusive_buft_name;
}

} // namespace ggml_cuda_expert

using ggml_cuda_expert::instance;

ggml_cuda_expert_lookup ggml_cuda_expert_lookup_tensor(const ggml_tensor * src0) {
    return instance().lookup(src0);
}

void ggml_cuda_expert_profile_ids(ggml_backend_cuda_context & ctx, const ggml_tensor * ids) {
    instance().profile_ids(ctx, ids);
}

void ggml_cuda_expert_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    instance().layer_done(ctx, src0);
}

void ggml_cuda_expert_before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    instance().before_read(ctx, src0);
}

bool ggml_cuda_expert_is_exclusive_buffer_type(ggml_backend_buffer_type_t buft) {
    return ggml_cuda_expert::is_exclusive_buft(buft);
}

static bool iface_configure(const ggml_expert_config * config) { return instance().configure(config); }
static ggml_backend_buffer_t iface_alloc_context(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity) {
    return instance().alloc_context(ctx, buft, identity);
}
static bool iface_register_context(ggml_context * ctx, ggml_backend_buffer_t buffer, const char * identity) {
    return instance().register_context(ctx, buffer, identity);
}
static bool iface_finalize(void) { return instance().finalize(); }
static void iface_release(ggml_context * ctx) { instance().release(ctx); }
static bool iface_status(ggml_expert_status * out) { return instance().status(out); }
static bool iface_bank_open(const char * label, bool prompt_bank, ggml_expert_bank_id * out) { return instance().bank_open(label, prompt_bank, out); }
static bool iface_bank_mark(ggml_expert_bank_id bank) { return instance().bank_zero(bank); }
static bool iface_bank_commit(ggml_expert_bank_id bank, const ggml_expert_record * record, ggml_expert_plan_id * out) {
    return instance().bank_commit(bank, record, out);
}
static bool iface_bank_discard(ggml_expert_bank_id bank) { return instance().bank_zero(bank); }
static bool iface_plan_install(ggml_expert_plan_id plan) { return instance().plan_install(plan); }
static bool iface_profile_select(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank) {
    return instance().profile_select(backend, row_begin, row_end, bank);
}
static bool iface_load_wanted(const ggml_tensor * tensor, int32_t expert) {
    return instance().tier_load_wanted(tensor, expert);
}
static bool iface_set_backing(const ggml_tensor * tensor, int32_t file_index, const char * path, uint64_t file_offset) {
    return instance().tier_set_backing(tensor, file_index, path, file_offset);
}
static bool iface_memory(ggml_backend_buffer_t buffer, size_t * host_bytes, size_t * device_bytes) {
    if (buffer == nullptr || !ggml_cuda_expert::is_exclusive_buft(buffer->buft)) {
        return false;
    }
    ggml_expert_status status = {};
    if (!instance().status(&status)) {
        return false;
    }
    if (host_bytes != nullptr) {
        *host_bytes = status.host_bytes;
    }
    if (device_bytes != nullptr) {
        *device_bytes = status.device_bytes;
    }
    return true;
}

static bool iface_attach_backend(ggml_backend_t backend, ggml_context * weights) {
    return instance().attach_backend(backend, weights);
}

static bool iface_route_hint(ggml_backend_t backend, const ggml_expert_route_hint * hints, int32_t n_hints) {
    return instance().route_hint(backend, hints, n_hints);
}

static const ggml_expert_iface g_expert_iface = {
    /* .abi_version      = */ GGML_EXPERT_ABI_VERSION,
    /* .configure        = */ iface_configure,
    /* .alloc_context    = */ iface_alloc_context,
    /* .register_context = */ iface_register_context,
    /* .finalize         = */ iface_finalize,
    /* .release          = */ iface_release,
    /* .status           = */ iface_status,
    /* .bank_open        = */ iface_bank_open,
    /* .bank_mark        = */ iface_bank_mark,
    /* .bank_commit      = */ iface_bank_commit,
    /* .bank_discard     = */ iface_bank_discard,
    /* .plan_install     = */ iface_plan_install,
    /* .profile_select   = */ iface_profile_select,
    /* .load_wanted      = */ iface_load_wanted,
    /* .set_backing      = */ iface_set_backing,
    /* .memory           = */ iface_memory,
    /* .attach_backend   = */ iface_attach_backend,
    /* .route_hint       = */ iface_route_hint,
};

const ggml_expert_iface * ggml_backend_cuda_expert_iface(void) {
    return &g_expert_iface;
}

#endif // GGML_USE_HIP
