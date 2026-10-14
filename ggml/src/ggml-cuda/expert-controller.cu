#include "expert-controller.cuh"

#if defined(GGML_USE_HIP)

#include "expert-geometry.h"
#include "expert-hash-early.h"
#include "expert-hash-stage.cuh"
#include "expert-host.cuh"
#include "expert-l2.cuh"
#include "expert-lend.h"
#include "expert-os.h"
#include "expert-plan.h"
#include "expert-profiler.cuh"
#include "expert-redraw.h"
#include "expert-profile-store.h"
#include "expert-storage.h"
#include "expert-ub-pool.cuh"
#include "mapped-host.cuh"

#include "ggml-account.h"
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
#include <thread>
#include <string>
#include <vector>

// a device buffer over memory the cache owns (ggml-cuda.cu)
ggml_backend_buffer_t ggml_backend_cuda_buffer_wrap(int device, void * ptr, size_t size);

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
};

// ---- the prefill lend's restore kernel (expert-lend.h) -------------------------------------------------------------

// Per routed layer, on the device: where a restored expert comes from and goes to.
struct lend_restore_desc {
    int32_t        * slots        = nullptr;   // the layer's VRAM slot table
    int32_t        * restore      = nullptr;   // [expert] lent slot to restore into, -1
    uint32_t       * bits         = nullptr;   // [words] restored experts (the SSD tier's publish skips them), or null
    char           * dst[3]       = {};        // arena base per kind
    const char     * src[3]       = {};        // host arena base (exclusive) or the mapped tensor (inclusive)
    const int32_t  * host_slots   = nullptr;   // exclusive: host slot of an expert, null otherwise
    const uint64_t * addresses[3] = {};        // the address table the kernels take first (the lookup's host_addresses), or null
    uint64_t         bytes[3]     = {};        // slot stride and copy size per kind
    int32_t          enabled      = 0;
};

struct lend_live {
    unsigned long long restores, bytes;
};

static constexpr int lend_restore_max    = 64;
static constexpr int lend_restore_blocks = 64;

// One decode launch of a layer: every block decides the same list (lend_restore_list), copies its share of the bytes
// and the last block publishes the slots. Nothing reads the tables a block decides from until the last block, so all
// blocks see the same list. The layer's matmuls run after this kernel on the same stream.
static __global__ void lend_restore_kernel(const lend_restore_desc * descs, int layer, const int32_t * ids, int rows,
        int used, int stride, int n_experts, unsigned int * tickets, lend_live * live) {
    const lend_restore_desc & d = descs[layer];
    __shared__ int32_t list[lend_restore_max];
    __shared__ int n;
    __shared__ int last;
    if (threadIdx.x == 0) {
        n = d.enabled ? lend_restore_list(ids, rows, used, stride, n_experts, d.restore, d.slots, list, lend_restore_max) : 0;
        // an expert without a readable source (no address and no host slot) stays where it is
        int m = 0;
        for (int i = 0; i < n; ++i) {
            const int e = list[i];
            bool ok = true;
            for (int k = 0; k < 3; ++k) {
                const uint64_t a = d.addresses[k] != nullptr ? d.addresses[k][e] : 0;
                if (a == 0 && d.host_slots != nullptr && d.host_slots[e] < 0) { ok = false; }
            }
            if (ok) { list[m++] = e; }
        }
        n = m;
    }
    __syncthreads();
    if (n == 0) {
        return;
    }
    const size_t tid = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    const size_t nth = (size_t) gridDim.x*blockDim.x;
    for (int i = 0; i < n; ++i) {
        const int e    = list[i];
        const int slot = d.restore[e];
        for (int k = 0; k < 3; ++k) {
            const size_t b = (size_t) d.bytes[k];
            if (b == 0) {
                continue;
            }
            const uint64_t a = d.addresses[k] != nullptr ? d.addresses[k][e] : 0;
            const char * src = a != 0 ? (const char *) (uintptr_t) a :
                d.host_slots != nullptr ? d.src[k] + (size_t) d.host_slots[e]*b : d.src[k] + (size_t) e*b;
            char * dst = d.dst[k] + (size_t) slot*b;
            const size_t n16 = (((uintptr_t) src | (uintptr_t) dst) & 15) == 0 ? b/16 : 0;
            for (size_t u = tid; u < n16; u += nth) {
                ((int4 *) dst)[u] = ((const int4 *) src)[u];
            }
            for (size_t u = n16*16 + tid; u < b; u += nth) {
                dst[u] = src[u];
            }
        }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(tickets + layer, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) {
        return;
    }
    unsigned long long bytes = 0;
    for (int i = 0; i < n; ++i) {
        const int e = list[i];
        d.slots[e]   = d.restore[e];
        d.restore[e] = -1;
        if (d.bits != nullptr) {
            atomicOr(d.bits + e/32, 1u << (e%32));
        }
        bytes += d.bytes[0] + d.bytes[1] + d.bytes[2];
    }
    tickets[layer] = 0;
    if (live != nullptr) {
        live->restores += (unsigned long long) n;
        live->bytes    += bytes;
    }
    __threadfence_system();
}

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
    // The unified host policy (expert-l2-class-ledger.h) with its defaults: ghost 32 decode generations, probation 0.10,
    // probation slots = the class's ring, prompt generations follow the decode rules, installs reseed in score order.
    out.unified.on = l2_unified_default;
    return out;
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
            // the aux tensors (EXL3 rotations) are in the delegate buffer and final now: the VRAM slots get their rows
            l1_->start_aux();
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
                l1_->device_bytes()/(1024*1024), resident_slices_locked(), total_slices_locked(),
                host_->host_bytes()/(1024*1024), cfg_.spare_slots);
            log_aux_locked("at model load");
            if (tier_) {
                report_tier_plan_locked("at model load");
            }
            log_members_locked("at model load");
            allocate_ub_pool_locked();
            lend_setup_locked();
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
        lend_scores_ = counts != nullptr ? scores : std::vector<uint64_t>();

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
        if (!l1_->allocate(capacities_, device_, 0, stage_class_slots_.empty() ? nullptr : &stage_class_slots_)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
        allocate_stage_locked();
        // every member is loaded: the installs bring the aux rows (EXL3 rotations) of the slots they fill
        l1_->start_aux();
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
        log_aux_locked("at model load");
        log_members_locked("at model load");
        allocate_ub_pool_locked();
        lend_setup_locked();
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
        lend_free_locked();
        free_ub_pool_locked();
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
        unequal_ = false;
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
            std::vector<uint64_t> part;
            if (!joint_) {
                part = delta;
            } else {
                part = member_rows_locked(mb, delta);
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
        if ((tier_ || cfg_.lend) && b.total_selections() != 0) {
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
        const plan_state & stored = committed;
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
        // a prefill lease: the plan goes into the kept slots, the rest of it into the restore space
        if (lend_on_.load(std::memory_order_acquire) && lend_.state != lend_state::none) {
            return lend_install_locked(id, plan);
        }
        // Installing what is already in the arena costs nothing: no drain, no copies, one line. The unified
        // host policy also skips it unless the rings are reseeded from the plan's profile at every install.
        const bool tier_static = !tier_ || (tier_->unified() && tier_->unified_reseed() == 0);
        if (tier_static && plan.selected == l1_->selected()) {
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
        // the decode pool's entries go back to their homes before the install republishes the tables
        ub_reclaim_sync_locked("install");
        // the install republishes the slot tables: the staging entries go, and the staging forgets its
        // experts (the next hint copies again; an install is rare next to the hints)
        for (std::vector<int32_t> & owners : stage_owner_) { stage_forget(owners); }
        return install_once_locked(id, plan);
    }

    bool install_once_locked(ggml_expert_plan_id id, const plan_state & plan) {
        const bool ok = tier_ ? install_tier_locked(id, plan, /*seed =*/ false) : install_plain_locked(id, plan);
        if (ok && id != GGML_EXPERT_PLAN_NONE) {
            lend_scores_ = plan.scores;
            lend_set_want_locked(plan);
        }
        return ok;
    }

    // The experts a restore space takes back: those the installed plan keeps in L1, when it is a decode plan with a
    // profile; none for a prompt plan or a cold one.
    void lend_set_want_locked(const plan_state & plan) {
        const bool decode = plan.bank >= banks_.size() || !banks_[plan.bank].prompt;
        lend_set_want_locked(decode && !plan.scores.empty() ? &plan.selected : nullptr);
    }

    void lend_set_want_locked(const expert_slot_table * selected) {
        lend_want_.clear();
        if (selected == nullptr) {
            return;
        }
        const size_t E = (size_t) geo_.n_experts;
        lend_want_.assign((size_t) geo_.n_layers*E, 0);
        for (size_t l = 0; l < selected->size() && l < (size_t) geo_.n_layers; ++l) {
            for (int32_t e : (*selected)[l]) {
                if (e >= 0 && (size_t) e < E) { lend_want_[l*E + (size_t) e] = 1; }
            }
        }
    }

    const char * bank_label_locked(ggml_expert_bank_id bank) const {
        return bank < banks_.size() ? banks_[bank].label.c_str() : "prefill lend";
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
                          "h2d_bytes=%zu MiB d2h_bytes=%zu MiB in %.1f ms%s\n",
                id, banks_[plan.bank].label.c_str(), mode_name(), stats.retained, stats.copied,
                stats.bytes/(1024*1024), stats.d2h_bytes/(1024*1024), stats.ms, aux_text(stats).c_str());
            std::string reason;
            if (!l1_->verify_current_assignment(reason)) {
                GGML_ABORT("expert cache: exclusive assignment is inconsistent after a swap: %s", reason.c_str());
            }
        } else {
            GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': retained=%zu copied=%zu bytes=%zu MiB in %.1f ms%s\n",
                id, banks_[plan.bank].label.c_str(), stats.retained, stats.copied, stats.bytes/(1024*1024), stats.ms,
                aux_text(stats).c_str());
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
            if (ub_ && ub_->lent() && ub_owner_ == key) {
                ub_reclaim_sync_locked("context freed");
            }
            if (lend_.state == lend_state::held && lend_.owner == key) {
                lend_end_locked("context freed");
            }
            if (ub_last_cctx_ == key) {
                ub_last_cctx_   = nullptr;
                ub_last_stream_ = nullptr;
            }
            ub_ring_judged_.erase(std::remove(ub_ring_judged_.begin(), ub_ring_judged_.end(), key), ub_ring_judged_.end());
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
        // The pool's step also counts the profile of a launch that gets one (one kernel less per layer)
        const bool ub_step = ub_on_.load(std::memory_order_acquire) && member == ub_member_ && ub_layer_[(size_t) joint] &&
                ids->ne[1] <= (int64_t) ub_rows_;
        if (profiler_ && !ub_step) {
        launch_profile_ids((const int32_t *) ids->data, (int) ids->ne[1], (int) (ids->nb[1]/sizeof(int32_t)), (int) ids->ne[0],
            geo_.n_experts, profiler_->counts_base() + size_t(joint)*geo_.n_experts, profiler_->n_counts(), profiler_->n_banks(),
            profiler_->selection((uint32_t) member), ctx.stream());
        }
        // The decode pool (expert-ub-pool.cuh): the step and the copies of a launch of few rows, before
        // the tier and every matmul of the layer, on the same stream. The kernels run whether or not the
        // pool is lent (r = 0 makes them return), so a captured graph stays valid either way.
        if (ub_step) {
            ub_->launch_step((int) joint, (const int32_t *) ids->data, (int) ids->ne[1], (int) (ids->nb[1]/sizeof(int32_t)),
                (int) ids->ne[0], true, ctx.stream(),
                profiler_ ? profiler_->counts_base() + size_t(joint)*geo_.n_experts : nullptr,
                profiler_ ? profiler_->n_counts() : 0, profiler_ ? profiler_->n_banks() : 0,
                profiler_ ? profiler_->selection((uint32_t) member) : nullptr);
            ub_->launch_copy((int) joint, ctx.stream());
        }
        // The SSD tier needs the same ids: this is the one site that sees both the fused and the
        // unfused router, and it is ordered before the layer's MUL_MAT_ID on the same stream.
        tier_route(ctx, (int) joint, ids);
        // The prefill lend: a decode launch copies the evicted experts it selects back into their lent slots, after the
        // tier made their file bytes readable and before the layer's matmuls. It runs whether or not a restore is due
        // (an empty list returns at once), so a captured graph stays valid either way.
        if (lend_on_.load(std::memory_order_acquire) && member == 0 && ids->ne[1] <= (int64_t) lend_rows_) {
            const ggml_cuda_kernel_launch_params lp(dim3(lend_restore_blocks), dim3(256), 0, ctx.stream());
            ggml_cuda_kernel_launch(lend_restore_kernel, lp, (const lend_restore_desc *) lend_desc_dev_, (int) joint,
                (const int32_t *) ids->data, (int) ids->ne[1], (int) ids->ne[0], (int) (ids->nb[1]/sizeof(int32_t)),
                geo_.n_experts, lend_tickets_dev_, lend_live_dev_);
        }
    }

    // ---- the decode pool (expert-ub-pool.cuh) ----------------------------------------------------

    // Called by the compute thread before every graph on `backend`: [base, base + size) is its
    // compute buffer and the graph uses it up to `extent`; `n_tokens` is the batch (<= 0: not a
    // batch, e.g. a K-shift). Lends the part above the decode graphs, and takes it back before a
    // graph that reaches into it.
    bool compute_spare(ggml_backend_t backend, void * base, size_t size, size_t extent, int32_t n_tokens) {
        if (lend_on_.load(std::memory_order_acquire) && backend != nullptr && ggml_backend_is_cuda(backend) &&
                n_tokens >= 1 && n_tokens <= lend_rows_) {
            // a decode graph: the prompt's implicit lend (graph_begin) ends here, without a refill
            ggml_backend_cuda_context * lctx = (ggml_backend_cuda_context *) backend->context;
            std::lock_guard<std::mutex> lock(mutex_);
            if (lctx->device == device_ && state_ == state_t::installed && lend_.state == lend_state::held && !lend_.explicit_) {
                lend_end_locked("a decode graph");
            }
        }
        if (!ub_on_.load(std::memory_order_acquire) || backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
        if (cctx->device != device_) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ub_ || state_ != state_t::installed) {
            return false;
        }
        ggml_cuda_set_device(device_);
        cudaStream_t stream = cctx->stream();
        if (ub_->ring()) {
            ub_ring_order_locked(cctx, stream, base);
        }
        if (ub_->lent()) {
            // an unknown buffer (null base) may be the lent one
            const bool same = base == nullptr || base == ub_lent_base_;
            if (same && extent <= ub_begin_) {
                return true;
            }
            if (!same && (const void *) cctx != ub_owner_) {
                return true;   // a buffer the pool does not use
            }
            if ((const void *) cctx != ub_owner_) {
                // the graphs of the lending context read the pool on its own stream
                CUDA_CHECK(cudaStreamSynchronize(ub_stream_));
            }
            xp_pause_locked(stream);
            ub_reclaim_locked(stream, same ? "a graph reaches into the pool" : "the compute buffer moved");
        }
        if (base == nullptr) {
            return true;
        }
        // Only the largest compute buffer lends: the target's, not the smaller one of a draft context that computes on
        // its own (contexts that share one buffer, a drafting context and its target, see views of different sizes at
        // one base).
        // The decode high-water mark is kept per buffer.
        ub_max_size_ = std::max(ub_max_size_, size);
        ub_buf_base_ = base;
        if (size >= ub_max_size_ && member_of(cctx) == ub_member_) {
            ub_member_base_ = base;   // the buffer the pool lends from (ring admission: a joint member is judged against it)
        }
        size_t & hwm = ub_hwm_of_[base];
        if (ub_hwm_of_.size() > 16) {
            ub_hwm_of_.clear();   // buffers come and go with the contexts; a forgotten mark is measured again
        }
        if (n_tokens <= 0 || n_tokens > ub_rows_) {
            return true;
        }
        // A small graph of any context counts for its buffer's mark: with shared compute buffers (a drafting context
        // computes in its target's) a draft's decode graphs run in the target's buffer, and a
        // lend above them is not taken back by every draft forward. Only the cached model's own contexts lend.
        const bool lends = member_of(cctx) == ub_member_;
        ub_hwm_ = hwm;
        if (ub_cfg_.log && (ub_calls_ < 16 || extent > ub_hwm_)) {
            GGML_LOG_INFO("expert cache: decode pool: a %d-row graph%s reaches %.1f MiB of the %.1f MiB compute buffer\n",
                n_tokens, lends ? "" : " of a draft", extent/1048576.0, size/1048576.0);
        }
        ++ub_calls_;
        ub_hwm_ = std::max(ub_hwm_, extent);
        ub_hwm_of_[base] = ub_hwm_;
        if (!lends) {
            return true;
        }
        const size_t align = size_t(2) << 20;
        size_t begin = (ub_hwm_ + std::max(ub_cfg_.margin, ub_hwm_/4) + align - 1)/align*align;
        if (size < ub_max_size_) {
            return true;
        }
        if (begin >= size || size - begin < ub_cfg_.min_bytes) {
            return true;
        }
        const int r = ub_->lend((char *) base + begin, size - begin, stream);
        if (r > 0) {
            xp_resume_locked();
            ub_owner_     = cctx;
            ub_stream_    = stream;
            ub_begin_     = begin;
            ub_lent_base_ = base;
            ++ub_lends_;
            if (ub_cfg_.log || ub_lends_ == 1) {
                GGML_LOG_INFO("expert cache: decode pool lent %.1f MiB above %.1f MiB of a %.1f MiB compute buffer (decode "
                              "graphs reach %.1f MiB): %d positions per layer, %d layers (lend %llu)\n",
                    (size - begin)/1048576.0, begin/1048576.0, size/1048576.0, ub_hwm_/1048576.0, r, ub_n_layers_,
                    (unsigned long long) ub_lends_);
            }
        }
        return true;
    }

    // ---- the prefill lend (expert-lend.h, ggml_expert_config::lend) -------------------------------------------------
    //
    // A prompt borrows its buffers from L1 (nothing from host memory). The lease (lend_book):
    //   none/restore -> held (lend_take_locked): the lowest ranked L1 residents leave (an install that retires the top
    //     slots of each class, after the kept residents above them moved down), and the pieces point into the retired
    //     slots;
    //   held -> restore (lend_end_locked): the explicit release, or the first decode graph after an implicit lend
    //     (graph_begin). Every lent slot is given an evicted expert (highest rank first), and a decode launch that
    //     reads such an expert copies it into its slot first (lend_restore_kernel);
    //   restore -> held: the restored copies leave the tables (they return to the mirror); more residents leave only
    //     when the pieces need more slots. A plan install under a lease installs the plan inside the kept slots, and
    //     the rest of the plan becomes the restore set (lend_install_locked).

    void lend_setup_locked() {
        lend_on_.store(false, std::memory_order_release);
        if (!cfg_.lend) {
            return;
        }
        const char * why = nullptr;
        if (joint_ || members_.size() > 1) {
            why = "a joint cache";
        } else if (!l1_ || !l1_->allocated()) {
            why = "no L1 arena";
        } else if (l1_->owns_host_storage() && !tier_ && (!host_ || !host_->chunked())) {
            why = "the exclusive host arena is not chunked";
        }
        if (why != nullptr) {
            GGML_LOG_WARN("expert cache: prefill lend off: %s; the prompt buffers use the pool\n", why);
            return;
        }
        ggml_cuda_set_device(device_);
        const size_t L = (size_t) geo_.n_layers, E = (size_t) geo_.n_experts, W = (E + 31)/32;
        CUDA_CHECK(cudaMalloc((void **) &lend_restore_dev_, L*E*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(lend_restore_dev_, 0xff, L*E*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void **) &lend_bits_dev_, L*W*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(lend_bits_dev_, 0, L*W*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void **) &lend_desc_dev_, L*sizeof(lend_restore_desc)));
        CUDA_CHECK(cudaMemset(lend_desc_dev_, 0, L*sizeof(lend_restore_desc)));
        CUDA_CHECK(cudaMalloc((void **) &lend_tickets_dev_, L*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(lend_tickets_dev_, 0, L*sizeof(unsigned int)));
        CUDA_CHECK(cudaHostAlloc((void **) &lend_live_host_, sizeof(lend_live), cudaHostAllocMapped));
        memset(lend_live_host_, 0, sizeof(lend_live));
        CUDA_CHECK(cudaHostGetDevicePointer((void **) &lend_live_dev_, lend_live_host_, 0));
        CUDA_CHECK(cudaStreamCreateWithFlags(&lend_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaDeviceSynchronize());
        if (tier_) {
            tier_->set_restored_bits(lend_bits_dev_);
        }
        lend_rows_ = (int) std::max<uint64_t>(1, cfg_.l2_decode_rows);
        lend_ = lend_book();
        lend_.slots = l1_->slot_counts();
        lend_.limit = lend_.slots;
        lend_guard_.assign(geo_.class_bytes.size()*geometry::n_kinds, 0);
        for (size_t c = 0; c < geo_.class_bytes.size(); ++c) {
            for (int k = 0; k < geometry::n_kinds; ++k) {
                lend_guard_[c*geometry::n_kinds + (size_t) k] = GGML_PAD(arena_tail_bytes(geo_, (int) c, k), 256);
            }
        }
        lend_upload_descs_locked();
        GGML_LOG_INFO("expert cache: prefill lend on: prompt buffers from the lowest ranked L1 experts (%s slots per class), "
                      "none from host memory; the decode copies an evicted expert back when it reads it (launches of up to %d "
                      "rows)\n", split_text(lend_.slots).c_str(), lend_rows_);
        lend_on_.store(true, std::memory_order_release);
    }

    void lend_free_locked() {
        lend_on_.store(false, std::memory_order_release);
        if (device_ >= 0) {
            ggml_cuda_set_device(device_);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (lend_held_piece & pc : lend_pieces_) {
            if (pc.buf != nullptr) {
                ggml_backend_buffer_free(pc.buf);
            }
        }
        lend_pieces_.clear();
        if (lend_restore_dev_)  { (void) cudaFree(lend_restore_dev_); }
        if (lend_bits_dev_)     { (void) cudaFree(lend_bits_dev_); }
        if (lend_desc_dev_)     { (void) cudaFree(lend_desc_dev_); }
        if (lend_tickets_dev_)  { (void) cudaFree(lend_tickets_dev_); }
        if (lend_live_host_)    { (void) cudaFreeHost(lend_live_host_); }
        if (lend_stream_)       { (void) cudaStreamDestroy(lend_stream_); }
        lend_restore_dev_ = nullptr;
        lend_bits_dev_ = nullptr;
        lend_desc_dev_ = nullptr;
        lend_tickets_dev_ = nullptr;
        lend_live_host_ = nullptr;
        lend_live_dev_ = nullptr;
        lend_stream_ = nullptr;
        lend_holder_stream_ = nullptr;
        lend_ = lend_book();
        lend_scores_.clear();
        lend_want_.clear();
        lend_counts_ = {};
        lend_live_mark_ = 0;
    }

    // The per-layer source and destination of the restore kernel (the tier's address tables exist once it started).
    void lend_upload_descs_locked() {
        std::vector<lend_restore_desc> d((size_t) geo_.n_layers);
        const size_t W = ((size_t) geo_.n_experts + 31)/32;
        const bool owned = l1_->owns_host_storage();
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int c = geo_.layer_class[l];
            if (c < 0) {
                continue;
            }
            lend_restore_desc & x = d[(size_t) l];
            for (int k = 0; k < geometry::n_kinds; ++k) {
                const ggml_cuda_expert_lookup lk = l1_->lookup(l, k);
                if (k == 0) {
                    x.slots      = const_cast<int32_t *>(lk.slots);
                    x.host_slots = lk.host_slots;
                }
                x.dst[k] = (char *) const_cast<void *>(lk.data);
                // inclusive: the mapped tensor is the home; exclusive: the host arena, or only the address table
                x.src[k] = lk.host_data != nullptr ? (const char *) lk.host_data :
                    !owned && geo_.tensors[l][k] != nullptr ? (const char *) geo_.tensors[l][k]->data : nullptr;
                x.addresses[k] = lk.host_addresses;
                x.bytes[k] = geo_.class_bytes[(size_t) c][(size_t) k];
            }
            x.restore = lend_restore_dev_ + (size_t) l*(size_t) geo_.n_experts;
            x.bits    = tier_ ? lend_bits_dev_ + (size_t) l*W : nullptr;
            x.enabled = x.slots != nullptr && x.dst[0] != nullptr ? 1 : 0;
        }
        CUDA_CHECK(cudaMemcpyAsync(lend_desc_dev_, d.data(), d.size()*sizeof(lend_restore_desc), cudaMemcpyHostToDevice, lend_stream_));
        CUDA_CHECK(cudaStreamSynchronize(lend_stream_));
    }

    // The restore table of lend_.restore and the restored bitmap of `kept` (experts already back at their slot).
    void lend_upload_restore_locked(const std::vector<lend_restore> & kept) {
        const size_t L = (size_t) geo_.n_layers, E = (size_t) geo_.n_experts, W = (E + 31)/32;
        std::vector<int32_t> t(L*E, -1);
        for (const lend_restore & r : lend_.restore) {
            t[(size_t) r.layer*E + (size_t) r.expert] = r.slot;
        }
        std::vector<uint32_t> bits(L*W, 0);
        for (const lend_restore & r : kept) {
            bits[(size_t) r.layer*W + (size_t) r.expert/32] |= 1u << (r.expert%32);
        }
        CUDA_CHECK(cudaMemcpyAsync(lend_restore_dev_, t.data(), t.size()*sizeof(int32_t), cudaMemcpyHostToDevice, lend_stream_));
        CUDA_CHECK(cudaMemcpyAsync(lend_bits_dev_, bits.data(), bits.size()*sizeof(uint32_t), cudaMemcpyHostToDevice, lend_stream_));
        CUDA_CHECK(cudaStreamSynchronize(lend_stream_));
        lend_upload_descs_locked();
    }

    std::vector<int> lend_held_locked() const {
        std::vector<int> held(lend_.slots.size(), 0);
        for (size_t c = 0; c < held.size(); ++c) {
            held[c] = lend_.slots[c] - lend_.limit[c];
        }
        return held;
    }

    std::vector<lend_class> lend_classes_locked() const {
        const size_t nc = geo_.class_bytes.size();
        std::vector<lend_class> out(nc);
        for (size_t c = 0; c < nc; ++c) {
            for (int k = 0; k < geometry::n_kinds; ++k) {
                out[c].kind_bytes.push_back(geo_.class_bytes[c][(size_t) k]);
                out[c].guard.push_back(lend_guard_[c*geometry::n_kinds + (size_t) k]);
            }
            out[c].slots = lend_.slots[c];
        }
        const auto & gpu = l1_->host_slots();
        const size_t E = (size_t) geo_.n_experts;
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int c = geo_.layer_class[l];
            if (c < 0) {
                continue;
            }
            for (int e = 0; e < geo_.experts(l); ++e) {
                const int32_t g = gpu[(size_t) l][(size_t) e];
                if (g < 0) {
                    continue;
                }
                lend_resident r;
                r.layer  = l;
                r.expert = e;
                r.slot   = g;
                r.score  = lend_scores_.empty() ? 0 : lend_scores_[(size_t) l*E + (size_t) e];
                out[(size_t) c].residents.push_back(r);
            }
        }
        return out;
    }

    bool lend_host_layout_locked() const {
        return !tier_ || tier_->class_layout();
    }

    // The L1 plan of `req`: the fewest lowest ranked residents whose slots hold every piece, over the slots held already.
    lend_l1_plan lend_plan_locked(const std::vector<ggml_expert_lend_piece> & req) {
        std::vector<size_t> bytes;
        for (const auto & pc : req) {
            bytes.push_back(pc.bytes);
        }
        const std::vector<int> held = lend_held_locked();
        return plan_l1_lend(lend_classes_locked(), bytes, 256, &held);
    }

    // Whether `req` could be lent now (nothing moves).
    bool lend_fits_locked(const std::vector<ggml_expert_lend_piece> & req) {
        return lend_plan_locked(req).ok;
    }

    // An install of the kept L1 residents with the top `plan.evict` slots of each class retired (a shrink install,
    // expert-redraw.h's evacuation): the kept residents above the new slot counts move down first; the evicted
    // experts go to their host home (exclusive without a finite tier: the host arena grows by them), the ring or the file.
    bool lend_evict_locked(const lend_l1_plan & plan) {
        const size_t nc = capacities_.size();
        const std::vector<int> full = capacities_;
        std::vector<int> caps(nc);
        const std::vector<int> held = lend_held_locked();
        for (size_t c = 0; c < nc; ++c) {
            caps[c] = full[c] - plan.evict[c];
            if (caps[c] < 0) {
                return false;
            }
        }
        const size_t L = (size_t) geo_.n_layers, E = (size_t) geo_.n_experts;
        std::vector<std::vector<uint8_t>> gone(L, std::vector<uint8_t>(E, 0));
        std::vector<int> per_class(nc, 0);
        for (const lend_resident & v : plan.victims) {
            gone[(size_t) v.layer][(size_t) v.expert] = 1;
            per_class[(size_t) geo_.layer_class[(size_t) v.layer]]++;
        }
        const auto & gpu = l1_->host_slots();
        expert_slot_table keep(L);
        for (size_t l = 0; l < L; ++l) {
            for (size_t e = 0; e < E; ++e) {
                if (gpu[l][e] >= 0 && !gone[l][e]) {
                    keep[l].push_back((int32_t) e);
                }
            }
        }
        const std::vector<int> slots_now = l1_->slot_counts();
        std::vector<std::vector<uint8_t>> other(nc);
        for (size_t c = 0; c < nc; ++c) {
            other[c].assign((size_t) std::max(slots_now[c], 0), 0);
        }
        const auto & spares = l1_->gpu_spares();
        for (size_t c = 0; c < spares.size() && c < nc; ++c) {
            for (int sl : spares[c]) {
                if (sl >= 0 && sl < slots_now[c]) { other[c][(size_t) sl] = 1; }
            }
        }
        const evac_plan ev = plan_evacuation(geo_.layer_class, geo_.n_experts, gpu, keep, other, slots_now, plan.limit);
        if (!ev.valid) {
            GGML_LOG_WARN("expert cache: prefill lend: %s\n", ev.reason.c_str());
            return false;
        }
        double evac_ms = 0.0;
        if (!l1_->evacuate(ev.ops, evac_ms)) {
            abort_locked("prefill lend: the evacuation copies failed");
        }
        if (l1_->owns_host_storage() && !tier_) {
            std::vector<int> host_up = host_->capacities();
            for (size_t c = 0; c < nc; ++c) {
                host_up[c] += per_class[c];
            }
            l1_arena::host_resize_stats hs;
            std::string why;
            if (!l1_->resize_host(host_up, hs, why)) {
                abort_locked(("prefill lend: host arena growth: " + why).c_str());
            }
        }
        set_capacities_locked(caps);
        l1_->set_retire(plan.limit);
        unequal_ = cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE;
        l1_->set_unequal(unequal_);
        plan_state step;
        step.selected = keep;
        step.scores   = lend_scores_;
        bool ok = false;
        if (tier_) {
            ok = install_tier_locked(GGML_EXPERT_PLAN_NONE, step, /*seed =*/ false);
        } else {
            l1_install_stats st;
            ok = l1_->install(keep, /*retain =*/ true, st);
            if (ok && l1_->owns_host_storage()) {
                std::string reason;
                if (!l1_->verify_current_assignment(reason)) {
                    GGML_ABORT("expert cache: exclusive assignment is inconsistent after the prefill lend: %s", reason.c_str());
                }
            }
            lend_last_d2h_ = st.d2h_bytes;
        }
        l1_->set_retire({});
        unequal_ = false;
        l1_->set_unequal(false);
        set_capacities_locked(full);
        if (!ok) {
            disable_locked("prefill lend: the install that evicts the lowest ranked experts failed");
            return false;
        }
        lend_last_moves_ = ev.ops.size();
        lend_last_d2h_   = install_last_d2h_;
        GGML_UNUSED(held);
        return true;
    }

    // One line per lend after its summary: where each piece went (role, MiB, class.kind), what each lent region holds
    // (used / capacity MiB past its zero guard of `g` bytes), and the residents it evicted per class (count, MiB). The
    // difference of the evicted and the lent bytes is the capacity of the regions the pieces leave unused.
    void lend_log_packing_locked(uint64_t n, const std::vector<ggml_expert_lend_piece> & req, const lend_l1_plan & plan,
                                 const std::vector<lend_resident> & victims) const {
        static const char * role_name[] = { "buf", "scratch0", "scratch1", "scratch2", "stage0", "T", "stage1" };
        std::vector<size_t> bytes;
        for (const auto & pc : req) { bytes.push_back(pc.bytes); }
        std::string pieces;
        for (size_t i = 0; i < req.size() && i < plan.fits.size(); ++i) {
            if (req[i].bytes == 0) { continue; }
            const int role = req[i].role;
            const lend_region & r = plan.regions[(size_t) plan.fits[i].region];
            char t[96];
            snprintf(t, sizeof(t), "%s%s %.1f@%d.%d", pieces.empty() ? "" : ", ",
                role >= 0 && role < (int) (sizeof(role_name)/sizeof(role_name[0])) ? role_name[role] : "?", req[i].bytes/1048576.0,
                r.cls, r.kind);
            pieces += t;
        }
        const std::vector<size_t> used = lend_region_used(plan.regions, plan.fits, bytes);
        std::string regions;
        for (size_t i = 0; i < plan.regions.size(); ++i) {
            const lend_region & r = plan.regions[i];
            char t[96];
            snprintf(t, sizeof(t), "%s%d.%d %.1f/%.1f g%zu", regions.empty() ? "" : ", ", r.cls, r.kind, used[i]/1048576.0,
                r.bytes/1048576.0, lend_guard_[(size_t) r.cls*3 + (size_t) r.kind]);
            regions += t;
        }
        std::vector<size_t> n_cls(lend_.slots.size(), 0), b_cls(lend_.slots.size(), 0);
        for (const lend_resident & v : victims) {
            const int c = v.layer >= 0 && v.layer < geo_.n_layers ? geo_.layer_class[(size_t) v.layer] : -1;
            if (c < 0 || (size_t) c >= n_cls.size()) { continue; }
            n_cls[(size_t) c]++;
            b_cls[(size_t) c] += geo_.class_total_bytes(c);
        }
        std::string ev;
        for (size_t c = 0; c < n_cls.size(); ++c) {
            if (n_cls[c] == 0) { continue; }
            char t[64];
            snprintf(t, sizeof(t), "%s%zu: %zu x %.1f", ev.empty() ? "" : ", ", c, n_cls[c], b_cls[c]/1048576.0);
            ev += t;
        }
        GGML_LOG_INFO("expert cache: prefill lend %llu packing: pieces (role MiB@class.kind) %s; regions (class.kind used/capacity "
                      "MiB, zero guard bytes) %s; evicted per class (n x MiB) %s, %d of the last tie given back (a class not "
                      "needed)\n", (unsigned long long) n, pieces.c_str(), regions.c_str(), ev.empty() ? "none" : ev.c_str(),
            plan.untied);
    }

    bool lend_take_locked(ggml_backend_cuda_context * cctx, const std::vector<ggml_expert_lend_piece> & req, bool expl,
                          ggml_expert_lend_grant * out) {
        if (!lend_on_.load(std::memory_order_acquire) || state_ != state_t::installed || req.empty() ||
                req.size() > GGML_EXPERT_LEND_MAX_PIECES) {
            return false;
        }
        if (lend_.state == lend_state::held) {
            if (lend_.owner == cctx && lend_same_locked(req)) {
                lend_.explicit_ = lend_.explicit_ || expl;
                lend_grant_locked(out);
                return true;
            }
            if (lend_.explicit_ && lend_.owner != cctx) {
                return false;   // another context's prompt holds it
            }
            lend_end_locked("a new lend");
        }
        const auto t0 = std::chrono::steady_clock::now();
        ggml_cuda_set_device(device_);
        CUDA_CHECK(cudaDeviceSynchronize());
        ub_reclaim_sync_locked("prefill lend");
        for (std::vector<int32_t> & owners : stage_owner_) { stage_forget(owners); }
        const bool was_restore = lend_.state == lend_state::restore;
        if (was_restore) {
            lend_drop_restore_locked("a new lend");
        }
        const std::vector<int> held = lend_held_locked();
        lend_l1_plan plan = lend_plan_locked(req);
        if (!plan.ok) {
            return false;
        }
        bool more = false;
        for (size_t c = 0; c < held.size(); ++c) {
            more = more || plan.evict[c] > held[c];
        }
        lend_last_d2h_ = 0;
        lend_last_moves_ = 0;
        if (more && !lend_evict_locked(plan)) {
            return false;
        }
        // the pieces
        lend_pieces_.clear();
        size_t di = 0, dev_bytes = 0, n_dev = 0;
        for (size_t i = 0; i < req.size(); ++i) {
            const auto & pc = req[i];
            lend_held_piece h;
            h.req = pc;
            if (pc.bytes == 0) {
                lend_pieces_.push_back(h);
                di++;
                continue;
            }
            const lend_fit f = plan.fits[di++];
            const lend_region & r = plan.regions[(size_t) f.region];
            h.ptr = l1_->gpu_slice_address(r.cls, r.kind, 0) + r.begin + f.offset;
            h.buf = ggml_backend_cuda_buffer_wrap(device_, h.ptr, pc.bytes);
            dev_bytes += pc.bytes;
            n_dev++;
            if (h.buf == nullptr) {
                lend_pieces_.push_back(h);
                GGML_LOG_WARN("expert cache: prefill lend: a piece of %.1f MiB has no buffer\n", pc.bytes/1048576.0);
                lend_.hold(plan.limit, plan.victims, cctx, expl);
                lend_holder_stream_ = cctx->stream();
                lend_end_locked("a piece failed");
                return false;
            }
            lend_pieces_.push_back(h);
        }
        // the head of the first lent slot stays zero: MMQ reads past the last row of the slot below it
        for (size_t c = 0; c < plan.limit.size(); ++c) {
            if (plan.limit[c] < lend_.slots[c]) {
                size_t g[3];
                for (int k = 0; k < 3; ++k) { g[k] = lend_guard_[c*3 + (size_t) k]; }
                l1_->zero_heads((int) c, plan.limit[c], 1, g);
            }
        }
        const size_t new_victims = plan.victims.size();
        uint64_t lo = 0, hi_score = 0;
        if (new_victims > 0) {
            lo = plan.victims.front().score;
            hi_score = plan.victims.back().score;
        }
        lend_.hold(plan.limit, plan.victims, cctx, expl);
        lend_holder_stream_ = cctx->stream();
        lend_counts_.acquires += more ? 1 : 0;
        lend_counts_.retakes  += more ? 0 : 1;
        lend_counts_.evicted  += new_victims;
        size_t ev_bytes = 0;
        for (const lend_resident & v : plan.victims) {
            ev_bytes += geo_.class_total_bytes(geo_.layer_class[(size_t) v.layer]);
        }
        lend_counts_.evicted_bytes += ev_bytes;
        lend_grant_locked(out);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        int held_slots = 0;
        for (int h : lend_held_locked()) { held_slots += h; }
        size_t residents = 0;
        for (const auto & row : l1_->host_slots()) { for (int32_t g : row) { residents += g >= 0 ? 1 : 0; } }
        // the bytes to host are the install's own copies out of VRAM (its d2h), the moves its relocations in VRAM
        GGML_LOG_INFO("expert cache: prefill lend %llu (%s): %.1f MiB VRAM in %zu pieces from %d lent L1 slots (%s per class kept), "
                      "%zu experts evicted now (%.1f MiB, the lowest ranked: scores %llu..%llu, %zu kept residents, %zu moved down, "
                      "%.1f MiB copied to host), nothing from host memory in %.1f ms\n",
            (unsigned long long) (lend_counts_.acquires + lend_counts_.retakes), expl ? "layer-major" : "ubatch op",
            dev_bytes/1048576.0, n_dev, held_slots, split_text(plan.limit).c_str(), new_victims, ev_bytes/1048576.0,
            (unsigned long long) lo, (unsigned long long) hi_score, residents, lend_last_moves_, lend_last_d2h_/1048576.0, ms);
        lend_log_packing_locked(lend_counts_.acquires + lend_counts_.retakes, req, plan, plan.victims);
        return true;
    }

    bool lend_same_locked(const std::vector<ggml_expert_lend_piece> & req) const {
        if (req.size() != lend_pieces_.size()) {
            return false;
        }
        for (size_t i = 0; i < req.size(); ++i) {
            const auto & h = lend_pieces_[i].req;
            if (h.role != req[i].role || h.bytes < req[i].bytes) {
                return false;
            }
        }
        return true;
    }

    void lend_grant_locked(ggml_expert_lend_grant * out) const {
        if (out == nullptr) {
            return;
        }
        memset(out, 0, sizeof(*out));
        for (size_t i = 0; i < lend_pieces_.size(); ++i) {
            out->buffers[i] = lend_pieces_[i].buf;
        }
    }

    // The holder's graphs that used the pieces are done: the lent L1 slots become the restore space of the evicted experts.
    void lend_end_locked(const char * why) {
        if (lend_.state != lend_state::held) {
            return;
        }
        ggml_cuda_set_device(device_);
        if (lend_holder_stream_ != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(lend_holder_stream_));
        }
        for (lend_held_piece & pc : lend_pieces_) {
            if (pc.buf != nullptr) {
                ggml_backend_buffer_free(pc.buf);
            }
        }
        lend_pieces_.clear();
        lend_.open_restore(geo_.layer_class, &lend_want_, geo_.n_experts);
        int held_slots = 0;
        for (size_t c = 0; c < lend_.slots.size(); ++c) {
            if (!lend_.lent((int) c)) {
                continue;
            }
            size_t g[3];
            for (int k = 0; k < 3; ++k) { g[k] = lend_guard_[c*3 + (size_t) k]; }
            l1_->zero_heads((int) c, lend_.limit[c], lend_.slots[c] - lend_.limit[c], g);
            held_slots += lend_.slots[c] - lend_.limit[c];
        }
        std::vector<l1_arena::slot_entry> ent;
        for (const lend_restore & r : lend_.restore) {
            ent.push_back({r.layer, r.expert, r.slot});
        }
        if (!l1_->stage_aux(ent)) {
            abort_locked("prefill lend: the aux rows of the restore slots");
        }
        lend_upload_restore_locked({});
        lend_holder_stream_ = nullptr;
        lend_counts_.releases++;
        size_t wanted = 0;
        for (const lend_resident & v : lend_.victims) {
            wanted += lend_wanted(lend_want_, geo_.n_experts, v.layer, v.expert) ? 1 : 0;
        }
        GGML_LOG_INFO("expert cache: prefill lend ends (%s): %zu evicted experts, %zu of them in the installed decode plan%s; "
                      "%zu may come back into %d lent L1 slots as the decode reads them (no refill), the others are read "
                      "where the L2 has them\n", why, lend_.victims.size(), wanted, lend_want_.empty() ? " (none: no decode plan)" : "",
            lend_.restore.size(), held_slots);
    }

    // A restore space taken back (the device is idle): the restored copies leave the tables.
    void lend_drop_restore_locked(const char * why) {
        if (lend_.state != lend_state::restore) {
            return;
        }
        if (!l1_->publish_overlay({})) {
            abort_locked("prefill lend: the tables of the restore space");
        }
        lend_.restore.clear();
        lend_upload_restore_locked({});
        lend_log_restores_locked(why);
    }

    void lend_log_restores_locked(const char * why) {
        if (lend_live_host_ == nullptr) {
            return;
        }
        const volatile lend_live * live = lend_live_host_;
        const unsigned long long n = live->restores, b = live->bytes;
        lend_counts_.restores      = n;
        lend_counts_.restore_bytes = b;
        if (n != lend_live_mark_) {
            GGML_LOG_INFO("expert cache: prefill lend: the decode restored %llu experts (%llu in all, %.1f MiB) before %s\n",
                n - lend_live_mark_, n, b/1048576.0, why);
        }
        lend_live_mark_ = n;
    }

    // A plan install under a lease: the plan inside the kept slots; the rest of it is the restore set, and the experts the
    // decode restored already stay where the plan still wants them.
    bool lend_install_locked(ggml_expert_plan_id id, const plan_state & plan) {
        if (lend_.state == lend_state::held) {
            if (lend_.explicit_) {
                GGML_LOG_WARN("expert cache: plan %u not installed: a prompt holds the prefill lend\n", id);
                return false;
            }
            lend_end_locked("a plan install");
        }
        ggml_cuda_set_device(device_);
        CUDA_CHECK(cudaDeviceSynchronize());
        ub_reclaim_sync_locked("install");
        for (std::vector<int32_t> & owners : stage_owner_) { stage_forget(owners); }
        lend_log_restores_locked("a plan install");
        std::vector<std::vector<int32_t>> dev;
        if (!l1_->read_device_tables(dev)) {
            abort_locked("prefill lend: the device tables");
        }
        const auto mirror = l1_->host_slots();
        std::vector<lend_restore> restored;
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int c = geo_.layer_class[l];
            if (c < 0) {
                continue;
            }
            for (int e = 0; e < geo_.experts(l); ++e) {
                const int32_t g = dev[(size_t) l][(size_t) e];
                if (g >= lend_.limit[(size_t) c] && g < lend_.slots[(size_t) c] && mirror[(size_t) l][(size_t) e] < 0) {
                    restored.push_back({l, e, c, g});
                }
            }
        }
        const size_t nc = capacities_.size();
        const std::vector<int> full = capacities_;
        std::vector<int> caps(nc);
        for (size_t c = 0; c < nc; ++c) {
            caps[c] = full[c] - (lend_.slots[c] - lend_.limit[c]);
        }
        placement_inputs in;
        in.geo              = &geo_;
        in.counts           = plan.scores.empty() ? nullptr : plan.scores.data();
        in.budget_bytes     = SIZE_MAX;
        in.exclusive        = l1_->owns_host_storage();
        in.fixed_capacities = &caps;
        plan_state step = plan;
        step.selected = plan_placement(in).selected;
        set_capacities_locked(caps);
        l1_->set_retire(lend_.limit);
        unequal_ = cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE;
        l1_->set_unequal(unequal_);
        const bool ok = install_once_locked(id, step);
        l1_->set_retire({});
        unequal_ = false;
        l1_->set_unequal(false);
        set_capacities_locked(full);
        if (!ok) {
            return false;
        }
        lend_scores_ = plan.scores;
        lend_set_want_locked(plan);
        // the restore set: what the plan holds beyond the kept slots
        const size_t E = (size_t) geo_.n_experts;
        std::vector<lend_resident> victims;
        for (int l = 0; l < geo_.n_layers; ++l) {
            const auto & want = plan.selected[(size_t) l];
            const auto & got  = step.selected[(size_t) l];
            for (int32_t e : want) {
                if (std::find(got.begin(), got.end(), e) != got.end()) {
                    continue;
                }
                lend_resident r;
                r.layer  = l;
                r.expert = e;
                r.score  = lend_scores_.empty() ? 0 : lend_scores_[(size_t) l*E + (size_t) e];
                victims.push_back(r);
            }
        }
        lend_rank_order(victims);
        std::vector<lend_restore> kept;
        for (const lend_restore & r : restored) {
            if (!lend_wanted(lend_want_, geo_.n_experts, r.layer, r.expert)) {
                continue;
            }
            for (const lend_resident & v : victims) {
                if (v.layer == r.layer && v.expert == r.expert) {
                    kept.push_back(r);
                    break;
                }
            }
        }
        lend_.victims = victims;
        lend_.restore = plan_restore(victims, geo_.layer_class, lend_.limit, lend_.slots, &kept, &lend_want_, geo_.n_experts);
        std::vector<l1_arena::slot_entry> ent, keep_ent;
        for (const lend_restore & r : lend_.restore) {
            ent.push_back({r.layer, r.expert, r.slot});
        }
        for (const lend_restore & r : kept) {
            keep_ent.push_back({r.layer, r.expert, r.slot});
        }
        if (!l1_->stage_aux(ent) || !l1_->publish_overlay(keep_ent)) {
            abort_locked("prefill lend: the restore space after an install");
        }
        lend_upload_restore_locked(kept);
        GGML_LOG_INFO("expert cache: prefill lend: plan %u installed in the %s kept slots per class; %zu experts of it wait in the "
                      "restore space (%zu restored copies kept)\n", id, split_text(lend_.limit).c_str(), lend_.restore.size(), kept.size());
        return true;
    }

    // ---- the iface entries and the op's hooks ----

    void graph_begin(ggml_backend_cuda_context * cctx, int n_op, int64_t n_tok) {
        if (n_op == 0 || !lend_on_.load(std::memory_order_acquire) || cctx->device != device_ || n_tok <= lend_rows_) {
            return;
        }
        if (cfg_.lend_scratch[0] + cfg_.lend_scratch[1] + cfg_.lend_scratch[2] == 0) {
            return;
        }
        // the blocks of this call's width, not of the configured ubatch: a short prompt borrows (and evicts) only what its op
        // reads. A held lend of this holder stays when its pieces cover the width, else it is taken again at the larger one.
        size_t need[3] = { cfg_.lend_scratch[0], cfg_.lend_scratch[1], cfg_.lend_scratch[2] };
        if (cfg_.lend_scratch_of != nullptr) {
            size_t w[3] = { 0, 0, 0 };
            cfg_.lend_scratch_of(cfg_.lend_scratch_user, n_tok, w);
            for (int i = 0; i < 3; ++i) { need[i] = std::min(need[i], w[i]); }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (lend_.state == lend_state::held && lend_.explicit_) {
            return;
        }
        if (lend_.state == lend_state::held && lend_.owner == cctx) {
            bool covered = lend_pieces_.size() == 3;
            for (const lend_held_piece & pc : lend_pieces_) {
                const int i = pc.req.role - GGML_EXPERT_LEND_SCRATCH0;
                if (i < 0 || i >= 3) { covered = false; continue; }
                covered = covered && pc.req.bytes >= need[i];
                need[i] = std::max(need[i], pc.req.bytes);
            }
            if (covered) {
                return;
            }
            // wider than the held pieces: the lend is taken again at the larger blocks (the held ones end first)
        }
        std::vector<ggml_expert_lend_piece> req;
        for (int i = 0; i < 3; ++i) {
            ggml_expert_lend_piece pc = {};
            pc.bytes = need[i];
            pc.role  = GGML_EXPERT_LEND_SCRATCH0 + i;
            req.push_back(pc);
        }
        if (!lend_take_locked(cctx, req, false, nullptr) && !lend_scratch_warned_) {
            lend_scratch_warned_ = true;
            GGML_LOG_WARN("expert cache: prefill lend: the expert-major op's scratch (%.1f MiB) does not fit in L1; the op uses "
                          "the pool\n", (need[0] + need[1] + need[2])/1048576.0);
        }
    }

    bool lend_is_on() const {
        return lend_on_.load(std::memory_order_acquire);
    }

    // The lent piece of `role` for an op on `cctx` that needs `bytes`, or null (the op then uses the pool).
    void * lent(ggml_backend_cuda_context * cctx, int role, size_t bytes) {
        if (!lend_on_.load(std::memory_order_acquire) || cctx->device != device_) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (lend_.state != lend_state::held || lend_.owner != (const void *) cctx) {
            return nullptr;
        }
        for (const lend_held_piece & pc : lend_pieces_) {
            if (pc.req.role == role && pc.req.bytes >= bytes && pc.ptr != nullptr) {
                return pc.ptr;
            }
        }
        return nullptr;
    }

    static bool lend_request_of(const ggml_expert_lend_request * r, std::vector<ggml_expert_lend_piece> & out) {
        out.clear();
        if (r == nullptr || r->n_pieces <= 0 || r->n_pieces > GGML_EXPERT_LEND_MAX_PIECES) {
            return false;
        }
        out.assign(r->pieces, r->pieces + r->n_pieces);
        return true;
    }

    bool iface_lend_fits(ggml_backend_t backend, const ggml_expert_lend_request * request) {
        std::vector<ggml_expert_lend_piece> req;
        if (!lend_on_.load(std::memory_order_acquire) || backend == nullptr || !ggml_backend_is_cuda(backend) ||
                !lend_request_of(request, req)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return state_ == state_t::installed && lend_fits_locked(req);
    }

    bool iface_lend_acquire(ggml_backend_t backend, const ggml_expert_lend_request * request, ggml_expert_lend_grant * out) {
        std::vector<ggml_expert_lend_piece> req;
        if (!lend_on_.load(std::memory_order_acquire) || backend == nullptr || !ggml_backend_is_cuda(backend) ||
                !lend_request_of(request, req)) {
            return false;
        }
        ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
        if (cctx->device != device_) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return lend_take_locked(cctx, req, true, out);
    }

    bool iface_lend_release(ggml_backend_t backend) {
        if (!lend_on_.load(std::memory_order_acquire) || backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
        std::lock_guard<std::mutex> lock(mutex_);
        if (lend_.state != lend_state::held || lend_.owner != (const void *) cctx) {
            return false;
        }
        lend_end_locked("released");
        return true;
    }

    bool iface_lend_counts(ggml_expert_lend_counts * out) {
        if (out == nullptr) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (lend_live_host_ != nullptr) {
            const volatile lend_live * live = lend_live_host_;
            lend_counts_.restores      = live->restores;
            lend_counts_.restore_bytes = live->bytes;
        }
        lend_counts_.held_slots = 0;
        for (int h : lend_held_locked()) { lend_counts_.held_slots += h; }
        lend_counts_.state = (int32_t) lend_.state;
        lend_counts_.lendable_bytes = 0;
        if (lend_on_.load(std::memory_order_acquire) && state_ == state_t::installed) {
            size_t total = 0, largest = 0;
            lend_l1_capacity(lend_classes_locked(), lend_held_locked(), total, largest);
            lend_counts_.lendable_bytes = total;
        }
        *out = lend_counts_;
        return lend_on_.load(std::memory_order_acquire);
    }

    // Ring admission: a step copies from a ring slot that only the stream order protects
    // (the worker serves only while that stream waits at a readiness). Each llama context computes on its own
    // stream, and nothing orders the graphs of two contexts on the device, so before a graph of another context the
    // previous context's stream is drained (normally already idle: its outputs were read).
    // A context of a joint cache member (a draft model in the cache) is judged at its first graph with a known buffer:
    // a graph at the base of the cached model's compute buffer runs in a view of it (the shared buffer types, which a
    // drafting context in the cache always takes), and those synchronize the other context's backend before every graph
    // (llama_context::share_compute_wait), so the ring admission stays on. A graph elsewhere (no sharing: only with
    // pipeline parallelism) turns it off for good: the positions from the ring go back before that graph, and the next
    // lends admit host residents only. Kept as a guard.
    void ub_ring_order_locked(ggml_backend_cuda_context * cctx, cudaStream_t stream, const void * base) {
        const int member = member_of(cctx);
        // before the cached model computed, nothing is lent and there is nothing to judge against: the next graph is judged
        if (member >= 0 && member != ub_member_ && base != nullptr && ub_member_base_ != nullptr &&
                std::find(ub_ring_judged_.begin(), ub_ring_judged_.end(), (const void *) cctx) == ub_ring_judged_.end()) {
            if (base == ub_member_base_) {
                ub_ring_judged_.push_back(cctx);
                GGML_LOG_INFO("expert cache: decode pool: ring admission stays on: the draft model's context "
                              "computes in the target's compute buffer (shared, one context at a time)\n");
            } else {
                GGML_LOG_WARN("expert cache: decode pool: ring admission is off from now on: the draft model's "
                              "context computes in its own compute buffer, not in a view of the target's\n");
                if (ub_->lent()) {
                    if ((const void *) cctx != ub_owner_) {
                        CUDA_CHECK(cudaStreamSynchronize(ub_stream_));
                    }
                    xp_pause_locked(stream);
                    ub_reclaim_locked(stream, "ring admission off");
                }
                ub_->set_ring(false);
                ub_last_cctx_   = nullptr;
                ub_last_stream_ = nullptr;
                return;
            }
        }
        if (ub_last_cctx_ != nullptr && ub_last_cctx_ != (const void *) cctx && ub_last_stream_ != nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(ub_last_stream_));
        }
        ub_last_cctx_   = cctx;
        ub_last_stream_ = stream;
    }

    int32_t compact_rows() const {
        return ub_on_.load(std::memory_order_acquire) ? ub_rows_ : 0;
    }

    // The counts of the pool from host memory (ggml_expert_iface::pool_counts); never waits for the device.
    bool pool_counts(ggml_expert_pool_counts * out) {
        if (out == nullptr || !ub_on_.load(std::memory_order_acquire)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ub_) {
            return false;
        }
        ggml_expert_pool_counts c{};
        if (!ub_->read_live(c.hits, c.fills, c.fill_bytes)) {
            return false;
        }
        c.lends    = ub_lends_;
        c.reclaims = ub_reclaims_;
        *out = c;
        return true;
    }

    void ub_reclaim_locked(cudaStream_t stream, const char * why) {
        ub_->reclaim(stream);
        ++ub_reclaims_;
        if (ub_cfg_.log) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            ub_log_counters_locked(why);
        }
        ub_owner_ = nullptr;
    }

    // While nothing computes (or before the device drain of the caller): back to the homes, synchronously.
    void ub_reclaim_sync_locked(const char * why) {
        if (!ub_ || !ub_->lent()) {
            return;
        }
        ggml_cuda_set_device(device_);
        xp_pause_locked(nullptr);
        CUDA_CHECK(cudaStreamSynchronize(ub_stream_));
        if (verify_all_) {
            std::string reason;
            const long checked = ub_->verify(reason);
            if (checked < 0) {
                GGML_ABORT("expert cache: decode pool: %s", reason.c_str());
            }
            GGML_LOG_INFO("expert cache: decode pool: verified %ld slots against their homes\n", checked);
        }
        ub_->reclaim(ub_stream_);
        CUDA_CHECK(cudaStreamSynchronize(ub_stream_));
        ++ub_reclaims_;
        if (ub_cfg_.log) {
            ub_log_counters_locked(why);
        }
        ub_owner_ = nullptr;
    }

    void ub_log_counters_locked(const char * why) const {
        std::array<uint64_t, ub_pool_n_counters> c{};
        if (!ub_->read_counters(c)) {
            return;
        }
        char ring_text[48] = "";
        if (ub_->ring_enabled()) {
            snprintf(ring_text, sizeof(ring_text), " ring_fills=%llu", (unsigned long long) c[ub_pool_ring_fills]);
        }
        GGML_LOG_INFO("expert cache: decode pool counters (%s): lends=%llu reclaims=%llu "
                      "steps=%llu hits=%llu fills=%llu "
                      "(%.1f MiB) over=%llu evictions=%llu rejected=%llu prefetch_fills=%llu prefetch_arrived=%llu%s\n", why,
            (unsigned long long) ub_lends_, (unsigned long long) ub_reclaims_, (unsigned long long) c[ub_pool_steps], (unsigned long long) c[ub_pool_hits],
            (unsigned long long) c[ub_pool_fills], c[ub_pool_fill_bytes]/1048576.0, (unsigned long long) c[ub_pool_over],
            (unsigned long long) c[ub_pool_evictions], (unsigned long long) c[ub_pool_rejected],
            (unsigned long long) c[ub_pool_pf_fills], (unsigned long long) c[ub_pool_pf_arrived], ring_text);
    }

    void allocate_ub_pool_locked() {
        if (ub_ || !l1_) {
            return;
        }
        ub_cfg_ = ub_pool_config_from_env();
        // The unified host policy (after its gate) serves every host expert from the rings: the pool takes
        // them through its ring admission, which is then on. The admission's own checks below and its runtime
        // judgement (ub_ring_order_locked) apply.
        if (tier_ && tier_->unified()) {
            ub_cfg_.ring = true;
            GGML_LOG_INFO("expert cache: decode pool: ring admission on under the unified host policy\n");
        }
        if (tier_ && !tier_->class_layout()) {
            GGML_LOG_WARN("expert cache: decode pool needs the class layout of the SSD tier; off\n");
            return;
        }
        if (geo_.n_experts > ub_pool_max_experts) {
            GGML_LOG_WARN("expert cache: decode pool supports up to %d experts per layer; off\n", ub_pool_max_experts);
            return;
        }
        const int L = geo_.n_layers;
        const member_state & mb = members_[0];
        const std::vector<uint8_t> early = early_joint_locked();
        // without a table from the tier or the host arena the pool brings its own (0 = not pooled)
        bool own_tables = false;
        for (int l = mb.offset; l < mb.offset + mb.geo.n_layers && l < L; ++l) {
            if (geo_.layer_class[l] >= 0 && l1_->lookup(l, 0).host_addresses == nullptr) {
                own_tables = true;
            }
        }
        if (own_tables) {
            const size_t bytes = size_t(L)*geometry::n_kinds*size_t(geo_.n_experts)*sizeof(uint64_t);
            if (cudaMalloc((void **) &ub_tables_, bytes) != cudaSuccess) {
                (void) cudaGetLastError();
                ub_tables_ = nullptr;
                GGML_LOG_WARN("expert cache: decode pool: address tables not allocated; off\n");
                return;
            }
            CUDA_CHECK(cudaMemset(ub_tables_, 0, bytes));
            uint64_t * tables = ub_tables_;
            const int E = geo_.n_experts;
            l1_->attach_pool_addresses([tables, E](int layer, int kind) {
                return (const uint64_t *) (tables + (size_t(layer)*geometry::n_kinds + size_t(kind))*size_t(E));
            });
        }
        std::vector<ub_pool_layer> layers((size_t) L);
        std::vector<std::array<size_t, 3>> bytes((size_t) L, {0, 0, 0}), tails((size_t) L, {0, 0, 0});
        ub_layer_.assign((size_t) L, 0);
        ub_n_layers_ = 0;
        for (int l = mb.offset; l < mb.offset + mb.geo.n_layers && l < L; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0 || early[(size_t) l] || (!stage_index_.empty() && stage_index_[(size_t) l] >= 0)) {
                continue;
            }
            ub_pool_layer & u = layers[(size_t) l];
            bool ok = true;
            for (int k = 0; k < geometry::n_kinds && ok; ++k) {
                const ggml_cuda_expert_lookup lk = l1_->lookup(l, k);
                const ggml_tensor * t = geo_.tensors[l][k];
                ub_pool_kind & c = u.kind[k];
                if (lk.host_data != nullptr) {
                    c.tensor = (const char *) lk.host_data;
                } else if (t != nullptr && t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
                    c.tensor = (const char *) ggml_hip_mapped_host_device_alias(t);
                }
                c.host_slots = lk.host_slots;
                c.addresses  = const_cast<uint64_t *>(lk.host_addresses);
                c.stride     = geo_.class_bytes[cls][k];
                ok = lk.slots != nullptr && c.addresses != nullptr && (c.tensor != nullptr || tier_);
                u.slots = lk.slots;
                bytes[(size_t) l][k] = geo_.class_bytes[cls][k];
                tails[(size_t) l][k] = arena_tail_bytes(geo_, cls, k);
            }
            if (!ok) {
                u = ub_pool_layer();
                continue;
            }
            u.serve   = tier_ ? tier_->serve_bitmap(l) : nullptr;
            u.enabled = 1;
            ub_layer_[(size_t) l] = 1;
            ++ub_n_layers_;
        }
        if (ub_n_layers_ == 0) {
            GGML_LOG_WARN("expert cache: decode pool: no routed layer can take positions; off\n");
            return;
        }
        ub_.reset(new ub_pool(L, geo_.n_experts, ub_cfg_.r_cap));
        if (!ub_->allocate(device_, layers, bytes, tails, ub_cfg_.ghost)) {
            ub_.reset();
            GGML_LOG_WARN("expert cache: decode pool: state not allocated; off\n");
            return;
        }
        if (tier_) {
            tier_->set_served_only();
        }
        if (ub_cfg_.ring) {
            // ring admission (expert-ub-pool.cuh): only where the stream order alone protects the ring slot it copies
            const char * why = nullptr;
            if (!tier_) {
                why = "there is no SSD tier";
            } else if (hash_mode_.ssd) {
                why = "RANMA_EXPERT_HASH_EARLY reads from the SSD while the graphs run";
            } else if (!ub_->enable_ring()) {
                why = "its state could not be allocated";
            } else if (!tier_->set_pool_bits(ub_->ring_bits_host(), ub_->ring_bits_device(), ub_->ring_words())) {
                ub_->set_ring(false);
                why = "the tier's bitmap layout differs";
            }
            if (why != nullptr) {
                GGML_LOG_WARN("expert cache: decode pool: ring admission is off: %s\n", why);
            }
        }
        ub_member_ = 0;
        ub_rows_   = ub_cfg_.rows > 0 ? ub_cfg_.rows : (int) std::max<uint64_t>(cfg_.l2_decode_rows, 1);
        ub_on_.store(true, std::memory_order_release);
        char filter_text[32];
        if (ub_cfg_.ghost > 0) {
            snprintf(filter_text, sizeof(filter_text), "ghost:%d", ub_cfg_.ghost);
        } else {
            snprintf(filter_text, sizeof(filter_text), "none");
        }
        GGML_LOG_INFO("expert cache: decode pool on: %d layers, up to %d positions each, launches of up to %d rows, "
                      "%zu KiB of state, admission filter %s%s%s\n", ub_n_layers_, ub_cfg_.r_cap, ub_rows_,
            ub_->device_bytes()/1024, filter_text, own_tables ? ", own address tables" : "",
            ub_->ring() ? ", ring occupants admitted" : "");
        xp_start_locked();
    }

    // ---- the expert prefetch (RANMA_EXPERT_PREFETCH; expert-ub-pool.cuh "prefetch staging") ----------
    //
    // The model publishes, after the output of layer l, a device prediction of the experts of the next
    // routed layer T (the argsort "ffn_moe_pred-<T>"). Mode inplace (default): the publish kernel reserves LRU
    // tail positions of T for the best predicted experts and lists them in a mapped mailbox; the relay thread
    // below copies each expert from its host home into its position with the copy engine (hipMemcpyAsync, then
    // hipStreamWriteValue32 of the position's done word) and the step of T settles the finished ones into owned
    // positions, no copy (see the pool). Mode fill: the relay picks the experts and copies them into staging
    // slots; the step's fill copies from the slot. Before a reclaim the relay stops and the compute stream waits
    // for its stream; it resumes after the next lend.
    //
    // The budget (experts per layer and publish) adapts per layer and row count (1, 2-3, 4+ rows): the step
    // records its clock and which slots had arrived, so the window (publish -> step) and whether the last
    // copy made it are known one token later. Cold start: `cold` publishes with no copies (windows only),
    // then the budget that fits the 10th percentile window under the transfer model (start + b * bytes / bw);
    // afterwards every `window` publishes: down when fewer than 90 % of the last copies arrived (at once when
    // the last 8 fall below 75 %), up when at least 98 % arrived and one more expert fits. A window that
    // changes with the context depth moves the recent samples, so the budget follows it.

    struct xp_config {
        bool   on = false;
        bool   inplace = true;
        int    slots = 4, ring = 4, k = 8, cold = 8, window = 32, test = 0;
        int    hold = 512, probe = 4096;   // predicted layer set: kept at least `hold` graphs, all layers again every `probe`
        bool   skip = true;                // layers whose budget is 0 do not predict
        double bw_gbps = 55.0, start_us = 70.0, margin_us = 10.0;
        bool   log = false;
    };

    struct xp_bucket {
        int b = 0, seen = 0, since = 0;
        bool init = false;
        float w[64] = {};      // recent windows, us
        int8_t ok[64] = {};    // recent: 1 the last issued copy arrived, 0 not, -1 nothing issued
        int n = 0, at = 0;
    };

    struct xp_layer {
        uint32_t last_gen = 0;      // generation of the last publish handled
        int      issued = 0, bucket = 0;
        uint64_t t_pub = 0;
        xp_bucket bk[3];
        uint64_t n_pub = 0, n_issue = 0, n_arrive = 0, n_take = 0, n_skip_lock = 0;
    };

    static xp_config xp_config_from_env() {
        auto num = [](const char * name, double def) {
            const char * v = getenv(name);
            return v != nullptr && *v != '\0' ? atof(v) : def;
        };
        xp_config c;
        c.on        = num("RANMA_EXPERT_PREFETCH", 0) != 0;
        if (const char * m = getenv("RANMA_EXPERT_PREFETCH_MODE"); m != nullptr && *m != '\0') {
            if (strcmp(m, "fill") == 0) {
                c.inplace = false;
            } else if (strcmp(m, "inplace") != 0) {
                GGML_LOG_WARN("expert cache: RANMA_EXPERT_PREFETCH_MODE='%s' is not inplace or fill; inplace\n", m);
            }
        }
        c.slots     = std::min(std::max((int) num("RANMA_EXPERT_PREFETCH_SLOTS", c.slots), 1), ub_pf_max_slots);
        c.ring      = std::min(std::max((int) num("RANMA_EXPERT_PREFETCH_RING", c.ring), 2), 16);
        c.k         = std::min(std::max((int) num("RANMA_EXPERT_PREFETCH_K", c.k), 1), ub_pf_max_k);
        c.cold      = std::max((int) num("RANMA_EXPERT_PREFETCH_COLD", c.cold), 1);
        c.window    = std::min(std::max((int) num("RANMA_EXPERT_PREFETCH_WINDOW", c.window), 8), 64);
        c.test      = (int) num("RANMA_EXPERT_PREFETCH_TEST", 0);
        c.bw_gbps   = std::max(num("RANMA_EXPERT_PREFETCH_BW_GBPS", c.bw_gbps), 1.0);
        c.start_us  = std::max(num("RANMA_EXPERT_PREFETCH_START_US", c.start_us), 0.0);
        c.margin_us = std::max(num("RANMA_EXPERT_PREFETCH_MARGIN_US", c.margin_us), 0.0);
        c.log       = num("RANMA_EXPERT_PREFETCH_LOG", 0) != 0;
        c.hold      = std::max((int) num("RANMA_EXPERT_PREFETCH_HOLD", c.hold), 16);
        c.probe     = std::max((int) num("RANMA_EXPERT_PREFETCH_PROBE", c.probe), c.hold);
        c.skip      = num("RANMA_EXPERT_PREFETCH_SKIP", 1) != 0;
        return c;
    }

    // Once, right after the pool is on (before any graph capture).
    void xp_start_locked() {
        xp_cfg_ = xp_config_from_env();
        if (!xp_cfg_.on) {
            return;
        }
        if (!ub_->enable_prefetch(xp_cfg_.ring, xp_cfg_.slots, xp_cfg_.inplace)) {
            GGML_LOG_WARN("expert cache: prefetch: staging not allocated; off\n");
            return;
        }
        int khz = 0;
        if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device_) != hipSuccess || khz <= 0) {
            (void) hipGetLastError();
            khz = 100000;
        }
        xp_tick_us_ = 1000.0/(double) khz;
        CUDA_CHECK(cudaStreamCreateWithFlags(&xp_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&xp_event_, cudaEventDisableTiming));
        xp_paused_ = false;
        xp_l_.assign((size_t) geo_.n_layers, xp_layer{});
        xp_reset_.assign((size_t) geo_.n_layers*3, 0);
        xp_layers_.clear();
        for (int l = 0; l < geo_.n_layers; ++l) {
            if (ub_layer_[(size_t) l]) {
                xp_layers_.push_back(l);
            }
        }
        {
            std::lock_guard<std::mutex> lk(xp_mask_mutex_);
            for (int b = 0; b < 3; ++b) {
                xp_profile_[b] = 2;   // every layer learns first
            }
            xp_graphs_ = 0;
            xp_next_eval_ = (uint64_t) xp_cfg_.hold;
            xp_next_probe_ = (uint64_t) xp_cfg_.probe;
            xp_mask_changes_ = 0;
        }
        xp_stop_.store(false);
        xp_thread_ = std::thread([this] { xp_loop(); });
        xp_on_.store(true, std::memory_order_release);
        GGML_LOG_INFO("expert cache: prefetch on (%s): %d layers, %s %d x %d of %.2f MiB, top %d per row, "
                      "adaptive budget (start %.0f us, %.0f GB/s, margin %.0f us, cold %d, window %d)%s\n",
            xp_cfg_.inplace ? "inplace" : "fill", (int) xp_layers_.size(), xp_cfg_.inplace ? "positions per publish up to" : "staging",
            xp_cfg_.inplace ? 1 : xp_cfg_.ring, xp_cfg_.slots, ub_->prefetch().slot_bytes/1048576.0, xp_cfg_.k,
            xp_cfg_.start_us, xp_cfg_.bw_gbps, xp_cfg_.margin_us, xp_cfg_.cold,
            xp_cfg_.window, xp_cfg_.test ? " (test mode)" : "");
    }

    void xp_stop_locked() {
        if (!xp_on_.load(std::memory_order_acquire)) {
            return;
        }
        xp_on_.store(false, std::memory_order_release);
        xp_stop_.store(true);
        if (xp_thread_.joinable()) {
            xp_thread_.join();
        }
        if (xp_stream_ != nullptr) {
            (void) cudaStreamSynchronize(xp_stream_);
            (void) cudaStreamDestroy(xp_stream_);
            xp_stream_ = nullptr;
        }
        if (xp_event_ != nullptr) {
            (void) cudaEventDestroy(xp_event_);
            xp_event_ = nullptr;
        }
        uint64_t pub = 0, iss = 0, arr = 0, take = 0, skip = 0;
        std::string budgets;
        for (int l : xp_layers_) {
            const xp_layer & s = xp_l_[(size_t) l];
            pub += s.n_pub; iss += s.n_issue; arr += s.n_arrive; take += s.n_take; skip += s.n_skip_lock;
            char b[32];
            snprintf(b, sizeof(b), " %d:%d/%d/%d", l, s.bk[0].b, s.bk[1].b, s.bk[2].b);
            budgets += b;
        }
        GGML_LOG_INFO("expert cache: prefetch: %llu publishes, %llu experts issued, %llu arrived in the window, %llu filled from "
                      "staging, %llu publishes skipped (busy), %llu profile changes\n", (unsigned long long) pub,
            (unsigned long long) iss, (unsigned long long) arr, (unsigned long long) take, (unsigned long long) skip,
            (unsigned long long) xp_mask_changes_);
        if (xp_cfg_.log) {
            GGML_LOG_INFO("expert cache: prefetch budgets (layer:1/2-3/4+ rows):%s\n", budgets.c_str());
        }
    }

    // ggml_expert_iface::prefetch_layers: the layers a graph of n_rows rows predicts, as one of three profiles per row
    // bucket (1, 2-3, 4+ rows): 0 none, 1 the MLA layers (joint layer % 4 == 3), 2 every pool layer. The profile is the
    // graph's reuse key, so each (shape, profile) graph is built once and picked again later. Called by the compute
    // thread for every graph's parameters, so the calls are the clock: a profile is kept at least `hold` calls; then it
    // is 2 when at least a quarter of the KDA layers still learn or have a budget, else 1 when an MLA layer does, else 0;
    // every `probe` calls it is 2 again and the zero budgets start over.
    static int xp_profile_has(int profile, int layer) {
        return profile == 2 || (profile == 1 && layer % 4 == 3);
    }

    uint32_t prefetch_layers(int32_t n_rows, uint64_t * mask, int32_t n_words) {
        if (!xp_on_.load(std::memory_order_acquire) || mask == nullptr || n_words <= 0) {
            return 0;
        }
        std::lock_guard<std::mutex> lk(xp_mask_mutex_);
        ++xp_graphs_;
        if (xp_cfg_.skip && xp_graphs_ >= xp_next_eval_) {
            xp_next_eval_ = xp_graphs_ + (uint64_t) xp_cfg_.hold;
            const bool probe = xp_graphs_ >= xp_next_probe_;
            if (probe) {
                xp_next_probe_ = xp_graphs_ + (uint64_t) xp_cfg_.probe;
            }
            for (int b = 0; b < 3; ++b) {
                int kda = 0, kda_on = 0, mla_on = 0;
                for (int l : xp_layers_) {
                    // read without the relay's ordering: a stale value only delays the change by one evaluation
                    xp_bucket & bk = xp_l_[(size_t) l].bk[b];
                    const bool on = !bk.init || bk.b > 0;
                    if (l % 4 == 3) {
                        mla_on += on;
                    } else {
                        ++kda;
                        kda_on += on;
                    }
                    if (probe && bk.init && bk.b == 0) {
                        xp_reset_[(size_t) l*3 + b] = 1;   // the relay restarts this bucket's learning
                    }
                }
                const int prof = probe || (kda > 0 && 4*kda_on >= kda) ? 2 : mla_on > 0 ? 1 : 0;
                if (prof != xp_profile_[b]) {
                    xp_profile_[b] = prof;
                    ++xp_mask_changes_;
                    if (xp_cfg_.log) {
                        GGML_LOG_INFO("expert cache: prefetch: %s rows predict profile %d (KDA %d/%d, MLA %d; after %llu graphs%s)\n",
                            b == 0 ? "1" : b == 1 ? "2-3" : "4+", prof, kda_on, kda, mla_on, (unsigned long long) xp_graphs_,
                            probe ? ", probe" : "");
                    }
                }
            }
        }
        const int b = n_rows <= 1 ? 0 : n_rows <= 3 ? 1 : 2;
        const int prof = xp_profile_[b];
        uint64_t m[2] = { 0, 0 };
        const int moff = members_[ub_member_].offset;
        for (int l : xp_layers_) {
            const int ml = l - moff;
            if (ml >= 0 && ml < 128 && xp_profile_has(prof, l)) {
                m[ml/64] |= 1ull << (ml%64);
            }
        }
        for (int w = 0; w < n_words; ++w) {
            mask[w] = w < 2 ? m[w] : 0;
        }
        return 1 + (uint32_t) prof;   // the reuse key: one graph per (shape, profile)
    }

    // inplace: after an up/gate matmul of a routed layer, before its down matmul.
    void xp_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        if (!xp_on_.load(std::memory_order_acquire) || !xp_cfg_.inplace || src0 == nullptr || ctx.device != device_) {
            return;
        }
        int layer = -1, kind = -1;
        if (!resolve(src0, layer, kind) || kind == 2 || layer < 0 || layer >= geo_.n_layers || !ub_layer_[(size_t) layer] ||
                member_of(&ctx) != ub_member_) {
            return;
        }
        ub_->launch_settle_down(layer, ctx.stream());
    }

    // Before a reclaim: no further copies, and `stream` (null: the host) waits for the issued ones, so that no copy
    // lands in a region the compute takes back. The lock is the relay's issue section.
    void xp_pause_locked(cudaStream_t stream) {
        if (!xp_on_.load(std::memory_order_acquire)) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(xp_issue_mutex_);
            xp_paused_ = true;
        }
        if (stream != nullptr) {
            CUDA_CHECK(cudaEventRecord(xp_event_, xp_stream_));
            CUDA_CHECK(cudaStreamWaitEvent(stream, xp_event_, 0));
        } else {
            CUDA_CHECK(cudaStreamSynchronize(xp_stream_));
        }
    }

    void xp_resume_locked() {
        if (!xp_on_.load(std::memory_order_acquire)) {
            return;
        }
        std::lock_guard<std::mutex> lk(xp_issue_mutex_);
        xp_paused_ = false;
    }

    // The publish after the predictor of layer T (`order`: "ffn_moe_pred-<l>", I32 argsort [n_experts, n_rows],
    // descending). On the compute stream; no mutex (like the pool's step).
    void xp_publish(ggml_backend_cuda_context & ctx, const ggml_tensor * order) {
        if (!xp_on_.load(std::memory_order_acquire) || order->type != GGML_TYPE_I32 || ctx.device != device_) {
            return;
        }
        static constexpr const char prefix[] = "ffn_moe_pred-";
        char * end = nullptr;
        const long layer = strtol(order->name + sizeof(prefix) - 1, &end, 10);
        if (end == order->name + sizeof(prefix) - 1 || *end != '\0' || layer < 0) {
            return;
        }
        const int member = member_of(&ctx);
        if (member != ub_member_ || layer >= members_[member].geo.n_layers) {
            return;
        }
        const long joint = members_[member].offset + layer;
        if (joint < 0 || joint >= geo_.n_layers || !ub_layer_[(size_t) joint] || order->ne[1] > std::min<int64_t>(ub_rows_, ub_pf_max_rows) ||
                order->ne[2] != 1 || order->nb[0] != sizeof(int32_t)) {
            return;
        }
        ub_->launch_publish((int) joint, (const int32_t *) order->data, (int) order->ne[1], (int) (order->nb[1]/sizeof(int32_t)),
            (int) std::min<int64_t>(xp_cfg_.k, order->ne[0]), ctx.stream());
    }

    // Budget of layer T for a publish of `bucket` rows; `s` is updated by xp_learn.
    int xp_budget(int T, xp_bucket & bk) {
        const int cap = xp_cfg_.slots;
        ++bk.seen;
        if (bk.seen <= xp_cfg_.cold || bk.n < 4) {
            return 0;
        }
        const double us_per_expert = (double) ub_->prefetch().slot_bytes/(xp_cfg_.bw_gbps*1e3);
        auto w10 = [&bk]() {
            float v[64];
            const int n = bk.n;
            for (int i = 0; i < n; ++i) { v[i] = bk.w[i]; }
            std::sort(v, v + n);
            return (double) v[n/10];
        };
        // the step needs up and gate (2/3 of an expert) of the last copy; the down kind has until the down matmul
        auto fits = [&](int b) {
            const double need = xp_cfg_.inplace ? (b - 1.0/3.0)*us_per_expert : b*us_per_expert;
            return xp_cfg_.start_us + need <= w10() - xp_cfg_.margin_us;
        };
        if (!bk.init) {
            bk.init = true;
            int b = 0;
            while (b < cap && fits(b + 1)) { ++b; }
            bk.b = b;
            bk.since = 0;
            return bk.b;
        }
        ++bk.since;
        // arrival rate of the last copy over the recent publishes that issued the current budget
        auto rate = [&bk](int last) {
            int n = 0, ok = 0;
            for (int i = 0; i < std::min(last, bk.n); ++i) {
                const int8_t o = bk.ok[(bk.at - 1 - i + 64) % 64];
                if (o >= 0) { ++n; ok += o; }
            }
            return n == 0 ? -1.0 : (double) ok/n;
        };
        const double r8 = rate(8);
        if (bk.b > 0 && r8 >= 0.0 && r8 < 0.75) {
            --bk.b;
            bk.since = 0;
        } else if (bk.since >= xp_cfg_.window) {
            const double r = rate(xp_cfg_.window);
            if (bk.b > 0 && r >= 0.0 && r < 0.90) {
                --bk.b;
            } else if (bk.b < cap && (bk.b == 0 || r >= 0.98) && fits(bk.b + 1)) {
                ++bk.b;
            }
            bk.since = 0;
        }
        GGML_UNUSED(T);
        return bk.b;
    }

    // The step record of the previous publish of layer T (stat.gen == its generation): window and arrivals.
    void xp_learn(int T, xp_layer & s, const ub_pf_stat & st) {
        const uint32_t sg = __atomic_load_n(&st.gen, __ATOMIC_ACQUIRE);
        if (s.last_gen == 0 || sg != s.last_gen) {
            return;
        }
        xp_bucket & bk = s.bk[s.bucket];
        const double w = (double) (st.t_step - s.t_pub)*xp_tick_us_;
        const uint32_t mask = s.issued > 0 ? (1u << s.issued) - 1 : 0;
        bk.w[bk.at] = (float) w;
        bk.ok[bk.at] = s.issued == 0 ? -1 : (int8_t) ((st.arrived >> (s.issued - 1)) & 1);
        bk.at = (bk.at + 1) % 64;
        bk.n = std::min(bk.n + 1, 64);
        s.n_arrive += (uint64_t) __builtin_popcount(st.arrived & mask);
        s.n_take   += (uint64_t) __builtin_popcount(st.taken & mask);
        s.last_gen = 0;
        GGML_UNUSED(T);
    }

    void xp_handle(int T, uint32_t gen, const ub_pf_mail & m, const ub_pf_stat & st) {
        xp_layer & s = xp_l_[(size_t) T];
        for (int b = 0; b < 3; ++b) {
            uint8_t & r = xp_reset_[(size_t) T*3 + b];
            if (__atomic_load_n(&r, __ATOMIC_ACQUIRE)) {
                __atomic_store_n(&r, (uint8_t) 0, __ATOMIC_RELEASE);
                s.bk[b] = xp_bucket{};
            }
        }
        xp_learn(T, s, st);
        const int rows = std::min(std::max(m.n_rows, 0), ub_pf_max_rows);
        const int k    = std::min(std::max(m.k, 0), ub_pf_max_k);
        const int bucket = rows <= 1 ? 0 : rows <= 3 ? 1 : 2;
        int b = xp_budget(T, s.bk[bucket]);
        ++s.n_pub;
        s.t_pub  = m.t_pub;
        s.bucket = bucket;
        s.issued = 0;
        s.last_gen = gen;
        if (xp_cfg_.inplace) {
            // the device reserved with the budget of the previous publish; this one sets the next
            ub_->pf_budget_host()[T*3 + bucket] = b;
            xp_copy_inplace(T, gen, m, s);
            return;
        }
        if (b <= 0 || rows == 0 || k == 0) {
            return;
        }
        // candidates: the predicted experts by rank summed over the rows, without L1 residents and the experts
        // the layer's positions held at its last step
        const int E = geo_.n_experts;
        std::vector<int> score((size_t) E, 0);
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < k; ++i) {
                const int e = m.ids[r*ub_pf_max_k + i];
                if ((unsigned) e < (unsigned) E) {
                    score[(size_t) e] += k - i;
                }
            }
        }
        std::vector<int> cand;
        for (int e = 0; e < E; ++e) {
            if (score[(size_t) e] > 0 && !((st.pooled[e/32] >> (e%32)) & 1u)) {
                cand.push_back(e);
            }
        }
        std::stable_sort(cand.begin(), cand.end(), [&score](int a, int c) { return score[(size_t) a] > score[(size_t) c]; });
        if (xp_cfg_.test == 2) {
            // test: a wrong prediction (the lowest-ranked candidates first)
            std::reverse(cand.begin(), cand.end());
        }
        struct pick_t { int e; const void * src[3]; };
        std::vector<pick_t> pick;
        {
            std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
            const auto t0 = std::chrono::steady_clock::now();
            while (!lock.try_lock()) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(50)) {
                    ++s.n_skip_lock;
                    return;
                }
                std::this_thread::yield();
            }
            if (!l1_ || (size_t) T >= l1_->host_slots().size()) {
                return;
            }
            const std::vector<int32_t> & vram = l1_->host_slots()[(size_t) T];
            const auto & homes = l1_->locations()[(size_t) T];
            for (int e : cand) {
                if ((int) pick.size() >= b) {
                    break;
                }
                if (vram[(size_t) e] >= 0 || homes[(size_t) e].storage != expert_storage::host) {
                    continue;
                }
                pick_t p{ e, { nullptr, nullptr, nullptr } };
                bool ok = true;
                for (int kind = 0; kind < geometry::n_kinds && ok; ++kind) {
                    p.src[kind] = l1_->host_address(T, kind, e);
                    ok = p.src[kind] != nullptr;
                }
                if (ok) {
                    pick.push_back(p);
                }
            }
        }
        if (pick.empty()) {
            return;
        }
        if (xp_cfg_.test == 3) {
            // test: a late copy (arrives after the step)
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        const ub_pf_dev & pf = ub_->prefetch();
        const int set = T % pf.ring;
        cudaStream_t st_ = xp_stream_;
        bool ok = true;
        for (size_t j = 0; j < pick.size(); ++j) {
            uint32_t * g = const_cast<uint32_t *>(pf.gen) + set*pf.slots + (int) j;
            ok = ok && hipStreamWriteValue32(st_, g, 0, 0) == hipSuccess;
        }
        for (size_t j = 0; j < pick.size() && ok; ++j) {
            for (int kind = 0; kind < geometry::n_kinds && ok; ++kind) {
                ok = cudaMemcpyAsync(ub_->pf_kind_address(T, set, (int) j, kind), pick[j].src[kind], ub_->kind_bytes(T, kind),
                    cudaMemcpyHostToDevice, st_) == cudaSuccess;
            }
            int32_t * o = const_cast<int32_t *>(pf.owner) + set*pf.slots + (int) j;
            ok = ok && hipStreamWriteValue32(st_, o, (uint32_t) pick[j].e, 0) == hipSuccess;
        }
        const uint32_t mark = xp_cfg_.test == 1 ? gen ^ 0x80u : gen;   // test 1: a generation that never matches
        for (size_t j = 0; j < pick.size() && ok; ++j) {
            uint32_t * g = const_cast<uint32_t *>(pf.gen) + set*pf.slots + (int) j;
            ok = hipStreamWriteValue32(st_, g, mark, 0) == hipSuccess;
        }
        (void) hipStreamQuery(st_);
        if (!ok) {
            (void) cudaGetLastError();
            return;
        }
        s.issued = (int) pick.size();
        s.n_issue += (uint64_t) pick.size();
    }

    // inplace: copy the reserved (position, expert) pairs of the mailbox, then each position's done word.
    void xp_copy_inplace(int T, uint32_t gen, const ub_pf_mail & m, xp_layer & s) {
        const int n = std::min(std::max(m.n_res, 0), ub_pf_max_slots);
        if (n == 0) {
            return;
        }
        struct pick_t { int q; int e; const void * src[3]; };
        pick_t pick[ub_pf_max_slots];
        int np = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
            const auto t0 = std::chrono::steady_clock::now();
            while (!lock.try_lock()) {
                if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(50)) {
                    ++s.n_skip_lock;
                    return;   // the positions stay pending; a later publish reserves them again
                }
                std::this_thread::yield();
            }
            if (!l1_) {
                return;
            }
            for (int j = 0; j < n; ++j) {
                pick_t p{ m.res_q[j], m.res_e[j], { nullptr, nullptr, nullptr } };
                if ((unsigned) p.e >= (unsigned) geo_.n_experts) {
                    continue;
                }
                bool ok = true;
                for (int kind = 0; kind < geometry::n_kinds && ok; ++kind) {
                    p.src[kind] = l1_->host_address(T, kind, p.e);
                    ok = p.src[kind] != nullptr;
                }
                if (ok) {
                    pick[np++] = p;
                }
            }
        }
        if (xp_cfg_.test == 3) {
            // test: a late copy (finishes after the step)
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
        std::lock_guard<std::mutex> lk(xp_issue_mutex_);
        if (xp_paused_) {
            return;
        }
        const ub_pf_dev & pf = ub_->prefetch();
        cudaStream_t st_ = xp_stream_;
        bool ok = true;
        int issued = 0;
        // test 1: done words that never match (the positions stay pending)
        const uint32_t mark = xp_cfg_.test == 1 ? gen ^ 0x40u : gen;
        for (int j = 0; j < np && ok; ++j) {
            // up and gate, their done word, then down and its own: the down matmul comes later
            for (int kind = 0; kind < 2 && ok; ++kind) {
                ok = cudaMemcpyAsync((void *) (uintptr_t) ub_->pos_kind_address(pick[j].q, kind), pick[j].src[kind],
                    ub_->kind_bytes(T, kind), cudaMemcpyHostToDevice, st_) == cudaSuccess;
            }
            ok = ok && hipStreamWriteValue32(st_, const_cast<uint32_t *>(pf.done) + size_t(pick[j].q)*2, mark, 0) == hipSuccess;
        }
        for (int j = 0; j < np && ok; ++j) {
            ok = cudaMemcpyAsync((void *) (uintptr_t) ub_->pos_kind_address(pick[j].q, 2), pick[j].src[2],
                ub_->kind_bytes(T, 2), cudaMemcpyHostToDevice, st_) == cudaSuccess;
            ok = ok && hipStreamWriteValue32(st_, const_cast<uint32_t *>(pf.done) + size_t(pick[j].q)*2 + 1, mark, 0) == hipSuccess;
            issued += ok ? 1 : 0;
        }
        (void) hipStreamQuery(st_);
        if (!ok) {
            (void) cudaGetLastError();
        }
        s.issued = n;   // the device's reservation count: the last of them decides the arrival record
        s.n_issue += (uint64_t) issued;
    }

    void xp_loop() {
        ggml_cuda_set_device(device_);
        const ub_pf_mail * mail = ub_->pf_mail_host();
        const ub_pf_stat * stat = ub_->pf_stat_host();
        std::vector<uint32_t> seen((size_t) geo_.n_layers, 0);
        while (!xp_stop_.load(std::memory_order_relaxed)) {
            bool any = false;
            for (int T : xp_layers_) {
                const uint32_t g = __atomic_load_n(&mail[T].gen, __ATOMIC_ACQUIRE);
                if (g == seen[(size_t) T]) {
                    continue;
                }
                seen[(size_t) T] = g;
                any = true;
                xp_handle(T, g, mail[T], stat[T]);
            }
            if (!any) {
                std::this_thread::yield();
            }
        }
    }

    void free_ub_pool_locked() {
        xp_stop_locked();
        ub_on_.store(false, std::memory_order_release);
        if (tier_ && ub_ && ub_->ring_enabled()) {
            tier_->set_pool_bits(nullptr, nullptr, ub_->ring_words());   // the worker must not read the freed bitmap
        }
        ub_.reset();
        if (ub_tables_ != nullptr) {
            (void) cudaFree(ub_tables_);
            ub_tables_ = nullptr;
        }
        ub_owner_ = nullptr;
        ub_stream_ = nullptr;
        ub_buf_base_ = nullptr;
        ub_lent_base_ = nullptr;
        ub_hwm_ = ub_begin_ = 0;
        ub_max_size_ = 0;
        ub_hwm_of_.clear();
    }

    void before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_before_read(ctx, src0);
    }

    void layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_layer_done(ctx, src0);
        xp_layer_done(ctx, src0);
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
        if (!tier_ || tensor == nullptr || !resolve(tensor, layer, kind) || expert < 0 || expert >= geo_.experts(layer)) {
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

    bool tier_layer_staged(const ggml_tensor * src0) {
        int layer = -1;
        int kind  = -1;
        return tier_ && state_ == state_t::installed && src0 != nullptr && resolve(src0, layer, kind) && tier_->layer_staged(layer);
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
        // Declared members may route another number of experts: the joint tables take the largest count as
        // their stride, and a layer's experts past its own count do not exist (geometry::layer_experts).
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
                GGML_LOG_INFO("expert cache: model %zu %s: joint layers %d..%d, %d experts, weight %.3f%s\n", m, mb.key.c_str(),
                    mb.offset, mb.offset + mb.geo.n_layers - 1, mb.geo.n_experts, (double) mb.weight,
                    mb.bound ? "" : ", joins later");
            }
            for (size_t cls = 0; cls < geo_.class_bytes.size(); ++cls) {
                GGML_LOG_INFO("expert cache: size class %zu: %d layers, %d experts, %zu KiB per expert\n",
                    cls, geo_.class_layers[cls], geo_.class_slices((int) cls), geo_.class_total_bytes((int) cls)/1024);
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
        // the aux rows are not in the signature: their layout must match the file's as well
        if (geo.aux_nb != mb.geo.aux_nb) {
            GGML_LOG_WARN("expert cache: model %s: the aux rows of the loaded model differ from the layout read from %s; "
                          "it is not cached\n", mb.key.c_str(), mb.path.c_str());
            drop_member_locked(m);
            return -1;
        }
        for (int l = 0; l < geo.n_layers; ++l) {
            for (int k = 0; k < geometry::n_kinds; ++k) {
                geo_.tensors[mb.offset + l][k] = geo.tensors[l][k];
            }
            // the joint geometry held the file's metadata tensors until now; they are freed below
            if ((size_t) (mb.offset + l) < geo_.aux.size() && (size_t) l < geo.aux.size()) {
                geo_.aux[mb.offset + l] = geo.aux[l];
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
        // the joined model's aux tensors are final now; slots published while it loaded may name its experts
        if (!l1_->refresh_aux()) {
            abort_locked("the aux rows of a joined model could not be copied");
        }
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
            if (!tier_->worker_running()) {
                ub_reclaim_sync_locked("model join");
                tier_->refresh_addresses();
            }
        }
        if (!l1_->verify_current_assignment(reason)) {
            abort_locked(("exclusive assignment is inconsistent: " + reason).c_str());
        }
        // what the loader wrote is verified against its digests while it is still in the host residents; the
        // joined model's classes then join the unified policy (with the worker stopped) before the worker serves
        verify_all_locked();
        if (tier_) {
            unify_joined_locked();
            tier_->start_worker(); // no-op when it runs
            report_tier_plan_locked("after a model joined");
        }
        log_members_locked("after a model joined");
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

    // Expert slices of every routed layer of the cache.
    size_t total_slices_locked() const {
        size_t n = 0;
        for (size_t c = 0; c < geo_.class_bytes.size(); ++c) {
            n += (size_t) std::max(geo_.class_slices((int) c), 0);
        }
        return n;
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
                for (int e = 0; e < geo_.experts(l); ++e) {
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

    // ---- the static capacities (the prefill lend's shrink install) --------------------------------

    void set_capacities_locked(const std::vector<int> & caps) {
        capacities_ = caps;
        l1_->set_static_capacities(caps);
    }

    static std::string split_text(const std::vector<int> & v) {
        std::string out;
        for (size_t c = 0; c < v.size(); ++c) { out += (c ? "," : "") + std::to_string(v[c]); }
        return out;
    }

    // The unified host policy runs only where the host slots are touched by
    // nothing but the tier's own installs and worker: exclusive mode, no early routes, no prefetch copies,
    // and every storage class holding the layers of one model (checked once the tier knows its storage
    // classes, in size_class_tier_locked). Models that join later take part: their classes join the policy
    // when they have bound and loaded (unify_joined_locked).
    //
    // Where the gate refuses it, the policy only stays off (info).
    void unified_off_locked(const char * why) const {
        GGML_LOG_INFO("expert cache: the unified host policy (default) is off: %s\n", why);
    }

    void unified_gate_locked(l2_config & c) {
        if (!c.unified.on) {
            return;
        }
        const char * why = nullptr;
        if (!c.class_layout) {
            why = "it needs the class layout";
        } else if (cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE) {
            why = "it supports exclusive mode only";
        } else if (hash_mode_.any()) {
            why = "RANMA_EXPERT_HASH_EARLY is set";
        } else if (xp_config_from_env().on) {
            why = "RANMA_EXPERT_PREFETCH is set";
        }
        if (why != nullptr) {
            unified_off_locked(why);
            c.unified.on = false;
        }
    }

    // Every non-VRAM expert as a file resident: the host locations of the unified policy. The layers of
    // `hold` (storage classes outside the policy, see unified_hold_layers_locked) keep `kept`'s locations.
    expert_locations unified_homes_locked(const std::vector<uint8_t> & hold = {}, const expert_locations * kept = nullptr) const {
        expert_locations homes = host_locations(expert_slot_table(geo_.n_layers, std::vector<int32_t>(geo_.n_experts, -1)));
        for (size_t l = 0; l < hold.size() && kept != nullptr; ++l) {
            if (hold[l]) { homes[l] = (*kept)[l]; }
        }
        return homes;
    }

    // Per storage class: whether the layers of every model that is not dropped are bound and loaded, so the
    // class can join the unified policy. A class of a model that joins later waits (unify_joined_locked).
    std::vector<uint8_t> unified_ready_storages_locked() const {
        const int n = tier_->storage().storages();
        std::vector<uint8_t> has((size_t) n, 0), waits((size_t) n, 0), ready((size_t) n, 0);
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0) { continue; }
            const member_state & mb = members_[(size_t) member_of_layer(l)];
            if (mb.dropped) { continue; }
            const int s = tier_->storage().storage_of[cls];
            has[(size_t) s] = 1;
            if (!mb.bound || !mb.loaded) { waits[(size_t) s] = 1; }
        }
        for (int s = 0; s < n; ++s) { ready[(size_t) s] = has[(size_t) s] && !waits[(size_t) s] ? 1 : 0; }
        return ready;
    }

    // Per layer, 1 where the layer's storage class is outside the policy for now (empty when none is): a
    // model that has not bound yet. Its residents and ring are untouched by installs until it joins.
    std::vector<uint8_t> unified_hold_layers_locked() const {
        std::vector<uint8_t> hold;
        if (!tier_ || !tier_->unified()) { return hold; }
        bool any = false;
        hold.assign((size_t) geo_.n_layers, 0);
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls >= 0 && !tier_->unified_storage(tier_->storage().storage_of[cls])) { hold[(size_t) l] = 1; any = true; }
        }
        if (!any) { hold.clear(); }
        return hold;
    }

    // Under the unified policy every slot of a storage class is a ring slot: one layer's demand, at most
    // its non-VRAM experts or what one ubatch can select, has to fit in its class. Stops the process otherwise.
    void check_unified_slots_locked(const std::vector<std::vector<int32_t>> & vram, const std::vector<uint8_t> & hold = {}) {
        const int bound = storage_demand_bound(geo_.n_experts, cfg_.l2_experts_used, cfg_.l2_prefill_rows);
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0 || (!hold.empty() && hold[(size_t) l])) { continue; }
            int in_vram = 0;
            for (int e = 0; e < geo_.experts(l); ++e) { in_vram += vram[(size_t) l][(size_t) e] >= 0 ? 1 : 0; }
            const int s = tier_->storage().storage_of[cls];
            const int demand = std::min(geo_.experts(l) - in_vram, bound);
            if (demand > tier_->class_slots(s)) {
                char text[256];
                snprintf(text, sizeof(text), "unified host policy: layer %d can demand %d experts outside VRAM at once, "
                         "its storage class %d has %d slots", l, demand, s, tier_->class_slots(s));
                abort_locked(text);
            }
        }
    }

    // The unified policy at the start, before the worker runs: the residents the loader wrote dissolve
    // into the rings, which are then filled with the best experts of the seed profile, in score order.
    //
    // A joint cache puts only the storage classes of the models that are bound and loaded into the policy.
    // The classes of a model that joins later keep their residents and rings: the install leaves their layers
    // as they are (their current host residents stay the plan's), and the model's slices are written to those
    // homes when it binds. unify_joined_locked dissolves them then.
    void unify_tier_locked() {
        const auto t0 = std::chrono::steady_clock::now();
        if (joint_) { tier_->set_unified_active(unified_ready_storages_locked()); }
        const std::vector<uint8_t> hold = unified_hold_layers_locked();
        const auto layout = tier_layout_locked(true);
        expert_slot_table none((size_t) geo_.n_layers);
        for (size_t l = 0; l < hold.size(); ++l) {
            for (int e = 0; hold[l] && e < geo_.experts((int) l); ++e) {
                if (l1_->locations()[l][(size_t) e].storage == expert_storage::host) { none[l].push_back(e); }
            }
        }
        const auto tx = ggml_cuda_expert::plan_install(geo_, l1_->selected(), none, l1_->host_slots(), l1_->locations(),
            capacities_, l1_->layout(), layout, l1_->gpu_spares(), {false, cfg_.spare_slots}, true, /*retain =*/ true,
            /*unequal =*/ unequal_);
        if (!tx.valid) { abort_locked(("unified host policy: " + tx.reason).c_str()); }
        l2_install_stats moved;
        std::string reason;
        if (!tier_->install_relabel(tx, l1_->locations(), *l1_, moved, reason) ||
                !tier_->reseed_unified(tier_scores_, tx.gpu_slots, true, moved, reason)) {
            abort_locked(("unified host policy: " + reason).c_str());
        }
        const expert_locations homes = unified_homes_locked(hold, &tx.host);
        if (!l1_->assign_tier(tx.gpu_slots, homes, layout, l1_->selected(), tx.gpu_spares) ||
                !l1_->verify_current_assignment(reason)) { abort_locked(("unified host policy: " + reason).c_str()); }
        tier_->set_homes(tx.gpu_slots, homes, host_geometry_locked(), true);
        check_unified_slots_locked(tx.gpu_slots, hold);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        GGML_LOG_INFO("expert cache: unified host policy: %zu residents dissolved into the rings, %zu slices / %zu MiB of the "
                      "seed profile read into their empty slots, rings in score order%s, in %.1f ms\n", moved.demoted,
            moved.reseed_slices, moved.reseed_bytes/(1024*1024), tier_scores_.empty() ? " (no seed profile: slot order)" : "", ms);
        if (!hold.empty()) { log_unified_waiting_locked(); }
    }

    void log_unified_waiting_locked() const {
        for (int s = 0; s < tier_->storage().storages(); ++s) {
            if (tier_->unified_storage(s)) { continue; }
            std::string models;
            for (size_t m = 0; m < members_.size(); ++m) {
                bool owns = false;
                for (int l = members_[m].offset; l < members_[m].offset + members_[m].geo.n_layers && l < geo_.n_layers; ++l) {
                    owns = owns || (geo_.layer_class[l] >= 0 && tier_->storage().storage_of[geo_.layer_class[l]] == s);
                }
                if (owns) { models += (models.empty() ? "" : ",") + std::to_string(m); }
            }
            GGML_LOG_INFO("expert cache: unified host policy: storage class %d (model %s) stays as residents and ring until its "
                          "model has bound and loaded\n", s, models.c_str());
        }
    }

    // A model joined the installed cache and its slices are in their homes: its storage classes join the unified
    // policy now, like the others did at the start. Its residents (what the loader wrote to host homes) dissolve
    // into the class ring in place, the best experts of the seed profile fill the empty slots (read from the
    // model's file, opened by the caller), the list is laid out in score order, and the model's experts outside
    // VRAM become file homes. Only these classes change; the others keep running as they are. The worker is
    // stopped, nothing computes while a model loads, and the pool is back at its homes.
    void unify_joined_locked() {
        if (!tier_ || !tier_->unified()) { return; }
        std::vector<uint8_t> scope = unified_ready_storages_locked();
        bool any = false;
        for (int s = 0; s < (int) scope.size(); ++s) {
            scope[(size_t) s] = scope[(size_t) s] && !tier_->unified_storage(s) ? 1 : 0;
            any = any || scope[(size_t) s];
        }
        if (!any) { return; }
        const auto t0 = std::chrono::steady_clock::now();
        tier_->stop_worker();
        ub_reclaim_sync_locked("model join");
        ggml_cuda_set_device(device_);
        CUDA_CHECK(cudaDeviceSynchronize());
        const expert_slot_table gpu_slots = l1_->host_slots();
        const expert_slot_table selected  = l1_->selected();
        const auto spares = l1_->gpu_spares();
        const install_layout layout = l1_->layout();
        l2_install_stats moved;
        std::string reason;
        if (!tier_->unify_storages(scope, tier_scores_, gpu_slots, moved, reason)) {
            abort_locked(("unified host policy: " + reason).c_str());
        }
        // the classes' experts outside VRAM live in the file now
        expert_locations homes = l1_->locations();
        for (int l = 0; l < geo_.n_layers; ++l) {
            const int cls = geo_.layer_class[l];
            if (cls < 0 || !scope[(size_t) tier_->storage().storage_of[cls]]) { continue; }
            for (size_t e = 0; e < homes[(size_t) l].size(); ++e) { homes[(size_t) l][e] = expert_location(); }
        }
        if (!l1_->assign_tier(gpu_slots, homes, layout, selected, spares) || !l1_->verify_current_assignment(reason)) {
            abort_locked(("unified host policy: " + reason).c_str());
        }
        tier_->set_homes(gpu_slots, homes, host_geometry_locked(), true);
        check_unified_slots_locked(gpu_slots, unified_hold_layers_locked());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string classes;
        for (int s = 0; s < (int) scope.size(); ++s) { if (scope[(size_t) s]) { classes += (classes.empty() ? "" : ",") + std::to_string(s); } }
        GGML_LOG_INFO("expert cache: unified host policy after a model joined: storage class %s: %zu residents dissolved into the "
                      "ring, %zu slices / %zu MiB of the seed profile read into its empty slots, ring in score order%s, in %.1f ms\n",
            classes.c_str(), moved.demoted, moved.reseed_slices, moved.reseed_bytes/(1024*1024),
            tier_scores_.empty() ? " (no seed profile: slot order)" : "", ms);
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
        if (joint_) {
            for (int l = 0; l < geo_.n_layers; ++l) { tier_cfg.layer_owner.push_back(member_of_layer(l)); }
        }
        unified_gate_locked(tier_cfg);
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
                limit[s] += geo_.experts(l);
                int resident = 0;
                for (int e = 0; e < geo_.experts(l); ++e) {
                    resident += tier_vram_[l][e] >= 0 || std::binary_search(base.selected[l].begin(), base.selected[l].end(), e) ? 1 : 0;
                }
                demand[s] += uint64_t(geo_.experts(l) - resident)*geo_.class_total_bytes(c);
            }
            ring = split_ring(st, floors, demand, limit, requested);
            if (ring_bytes(floors) > requested) {
                note = "the requested ring of " + std::to_string(requested/(1024*1024)) + " MiB was raised to its floors";
            }
        } else if (factor > 1.0) {
            // factor x floor per class, as far as the budget left above the minimum (the floors) allows
            std::vector<int> limit(n, 0);
            for (int c = 0; c < (int) classes; ++c) { limit[st.storage_of[c]] += geo_.class_slices(c); }
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
        if (tier_->unified_shared_storage() >= 0) {
            const std::string why = "storage class " + std::to_string(tier_->unified_shared_storage()) +
                " holds the layers of several models (the policy takes a class of one model)";
            unified_off_locked(why.c_str());
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
        if (tier_->unified()) {
            unify_tier_locked();
        }
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
            for (int candidate = 0; candidate < geo_.experts(layer) && expert < 0; ++candidate) {
                if (gpu_table[layer][candidate] < 0 && host_table[layer][candidate] >= 0) {
                    expert = candidate;
                }
            }
            bool in_vram = false;
            for (int candidate = 0; candidate < geo_.experts(layer) && expert < 0; ++candidate) {
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

    bool install_tier_locked(ggml_expert_plan_id id, const plan_state & requested, bool seed) {
        const auto t0 = std::chrono::steady_clock::now();
        // A storage class outside the unified policy (the model of its layers has not bound yet) is not
        // installed: the layers keep their VRAM slices and host residents until unify_joined_locked.
        const std::vector<uint8_t> hold = unified_hold_layers_locked();
        plan_state held;
        if (!hold.empty()) {
            held = requested;
            for (int l = 0; l < geo_.n_layers; ++l) {
                if (hold[(size_t) l]) { held.selected[(size_t) l] = l1_->selected()[(size_t) l]; }
            }
        }
        const plan_state & plan = hold.empty() ? requested : held;
        tier_->stop_worker();
        bool has_prefill = false;
        for (const bank_state & bank : banks_) { has_prefill |= bank.prompt && bank.latest_plan != GGML_EXPERT_PLAN_NONE; }
        const bool decode_bank = plan.bank < banks_.size() && !banks_[plan.bank].prompt;
        const bool prompt = plan_uses_prompt_ring(cfg_.l2_phase_rings, decode_bank, has_prefill, seed);
        const int ring_before = tier_->ring_count();
        const auto layout = tier_layout_locked(prompt);
        auto cut = host_capacities_base_;
        for (size_t c = 0; c < cut.size(); ++c) { cut[c] += layout.lent_count[c]; }
        vram_table_locked(plan.selected, tier_vram_);
        const bool unified = tier_->unified();
        tier_plan host_plan;
        if (unified) {
            // No host resident but the demotions from VRAM, which the install copies back into the rings.
            host_plan.valid = true;
            host_plan.selected.assign((size_t) geo_.n_layers, {});
            std::vector<int> room(layout.host.size(), 0);
            for (size_t c = 0; c < room.size(); ++c) { room[c] = layout.host[c] + layout.lent_count[c]; }
            const auto & gpu_now = l1_->host_slots();
            for (int l = 0; l < geo_.n_layers; ++l) {
                const int c = geo_.layer_class[l];
                if (c < 0) { continue; }
                const auto & keep = plan.selected[(size_t) l];
                if (!hold.empty() && hold[(size_t) l]) {
                    for (int e = 0; e < geo_.experts(l); ++e) {
                        if (l1_->locations()[(size_t) l][(size_t) e].storage == expert_storage::host) { host_plan.selected[(size_t) l].push_back(e); }
                    }
                    continue;
                }
                for (int e = 0; e < geo_.experts(l); ++e) {
                    if (gpu_now[(size_t) l][(size_t) e] >= 0 && room[(size_t) c] > 0 &&
                            std::find(keep.begin(), keep.end(), e) == keep.end()) {
                        host_plan.selected[(size_t) l].push_back(e);
                        --room[(size_t) c];
                    }
                }
            }
        } else {
            tier_inputs in = tier_inputs_locked(plan.scores);
            in.fixed_capacities = &cut;
            host_plan = plan_host_tier(in);
            if (!host_plan.valid) { abort_locked(host_plan.reason.c_str()); }
        }
        const auto tx = ggml_cuda_expert::plan_install(geo_, plan.selected, host_plan.selected, l1_->host_slots(), l1_->locations(),
            capacities_, l1_->layout(), layout, l1_->gpu_spares(),
            {cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE, cfg_.spare_slots}, true, /*retain =*/ true,
            /*unequal =*/ unequal_ && cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE);
        if (!tx.valid) { abort_locked(("install transaction: " + tx.reason).c_str()); }
        install_last_d2h_ = tx.d2h_bytes;
        std::string reason;
        // Class layout: the moves run as relabels of the storage slots (the rings keep their
        // contents); the separate ring of direct tier users goes through the mover.
        const bool relabel = tier_->relabel_installs();
        l2_install_stats moved;
        if (relabel) {
            if (!tier_->install_relabel(tx, l1_->locations(), *l1_, moved, reason)) {
                abort_locked(("install relabel: " + reason).c_str());
            }
            // the unified policy: the rings start every plan from its profile
            if (unified && tier_->unified_reseed() > 0 &&
                    !tier_->reseed_unified(plan.scores, tx.gpu_slots, tier_->unified_reseed() == 2, moved, reason)) {
                abort_locked(("unified reseed: " + reason).c_str());
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
        // the unified policy leaves every non-VRAM expert to the rings: no host home
        const expert_locations homes = unified ? unified_homes_locked(hold, &tx.host) : tx.host;
        if (!l1_->assign_tier(tx.gpu_slots, homes, layout, plan.selected, tx.gpu_spares) ||
                !l1_->verify_current_assignment(reason)) { abort_locked(("install assignment: " + reason).c_str()); }
        // the tables are published after every read and copy of the transaction completed
        tier_->set_homes(tx.gpu_slots, homes, host_geometry_locked(), relabel);
        if (unified) { check_unified_slots_locked(tx.gpu_slots, hold); }
        const size_t host_total = host_->host_bytes() + tier_->host_bytes();
        GGML_ASSERT(host_total <= cfg_.l2_bytes);
        if (repartition) {
            GGML_LOG_INFO("expert cache: L2 repartition ring=%d->%d lent_slots=%d host_total=%zu\n",
                ring_before, tier_->ring_count(), tier_->prompt_slots() - tier_->ring_count(), host_total);
        }
        tier_->start_worker();
        if (!unified) {
            log_budget_split_locked(host_plan, prompt);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string relabel_text;
        if (unified) {
            char text[224];
            snprintf(text, sizeof(text), " unified: %zu demotions into the rings, reseed-read %zu slices / %zu MiB in %.1f ms",
                moved.unified_demoted, moved.reseed_slices, moved.reseed_bytes/(1024*1024), moved.reseed_ms);
            relabel_text = text;
        }
        if (relabel) {
            char text[256];
            snprintf(text, sizeof(text), " relabel=%zu slices / %zu MiB ring-copy=%zu slices / %zu MiB read=%zu slices / %zu MiB "
                     "ring kept %zu/%zu occupants (%zu after, %zu residents demoted into the rings, %d read batches)",
                moved.relabel_slices, moved.relabel_bytes/(1024*1024), moved.ring_copy_slices, moved.ring_copy_bytes/(1024*1024),
                moved.read_slices, moved.read_bytes/(1024*1024), moved.ring_kept, moved.ring_before, moved.ring_after,
                moved.demoted, moved.batches);
            relabel_text += text;
        }
        GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': mode=%s-l2 ring=%s%s retained=%zu/%zu "
                      "h2d=%zu slices / %zu MiB d2h=%zu slices / %zu MiB ssd=%zu slices / %zu MiB%s in %.1f ms\n",
            id, bank_label_locked(plan.bank), cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE ? "inclusive" : "exclusive",
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
                bank_label_locked(plan.bank), id, ms, tx.h2d_bytes, tx.d2h_bytes, tx.ssd_bytes, ring_before, tier_->ring_count(),
                relabel ? 1 : 0, moved.read_slices, moved.read_bytes, moved.relabel_slices, moved.relabel_bytes,
                moved.ring_copy_slices, moved.ring_copy_bytes, moved.ring_before, moved.ring_kept, moved.ring_after,
                moved.demoted, moved.batches);
        }
        if (id != GGML_EXPERT_PLAN_NONE) {
            finish_plan_locked(id);
        }
        verify_all_locked();
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
            for (int expert = 0; expert < geo_.experts(layer); ++expert) {
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
            for (int expert = 0; expert < geo_.experts(layer); ++expert) {
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
        if (l1_->aux_enabled()) {
            const auto t0 = std::chrono::steady_clock::now();
            // a joined model that has not loaded yet (or was dropped) has no rows
            std::vector<uint8_t> layers((size_t) geo_.n_layers, 1);
            for (int l = 0; joint_ && l < geo_.n_layers; ++l) {
                const member_state & mb = members_[(size_t) member_of_layer(l)];
                layers[(size_t) l] = mb.loaded && !mb.dropped ? 1 : 0;
            }
            std::string reason;
            const long checked = l1_->verify_aux(reason, &layers);
            if (checked < 0) {
                GGML_ABORT("expert cache: aux rows: %s", reason.c_str());
            }
            GGML_LOG_INFO("expert cache: verified the aux rows of %ld VRAM residents against their tensors in %.1f ms\n", checked,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
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

    // The same in VRAM, where a slot also holds the aux rows.
    size_t vram_spare_bytes_locked() const {
        size_t bytes = 0;
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            bytes += size_t(cfg_.spare_slots)*geo_.class_vram_bytes(cls);
        }
        return bytes;
    }

    // " aux_rows=<n> aux_bytes=<KiB> KiB" for an install line of a bank with aux rows, else nothing.
    std::string aux_text(const l1_install_stats & stats) const {
        if (!l1_ || !l1_->aux_enabled()) {
            return "";
        }
        return " aux_rows=" + std::to_string(stats.aux_rows) + " aux_bytes=" + std::to_string(stats.aux_bytes/1024) + " KiB";
    }

    // The aux arenas of a bank with aux rows: their VRAM, what one resident expert adds, and the rows copied so far.
    void log_aux_locked(const char * what) const {
        if (!l1_ || !l1_->aux_enabled()) {
            return;
        }
        std::string per;
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            char item[96];
            snprintf(item, sizeof(item), "%s%zu+%zu", cls ? "," : "", geo_.class_total_bytes(cls), geo_.class_aux_total_bytes(cls));
            per += item;
        }
        GGML_LOG_INFO("expert cache: aux rows (EXL3 rotations) %s: %.1f MiB of VRAM arenas at the weight slots, bytes per "
                      "expert (weights+aux) by class %s, %zu experts' rows copied (%.1f MiB)\n", what,
            double(l1_->aux_device_bytes())/(1024.0*1024.0), per.c_str(), l1_->aux_rows_copied(),
            double(l1_->aux_bytes_copied())/(1024.0*1024.0));
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
            (owns_host ? vram_spare_bytes_locked() : 0) +
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
            bytes += size_t(stage_slots_)*geo_.class_vram_bytes(cls);
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
        const uint64_t stage_bytes_before = stage_bytes_;
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
                    // the aux rows of the slot, before the publish below names it
                    for (int which = 0; l1_->aux_enabled() && which < geometry::n_aux; ++which) {
                        const size_t aux_bytes = geo_.class_aux_bytes(cls, kind, which);
                        if (aux_bytes == 0) {
                            continue;
                        }
                        const char * aux_src = l1_->aux_host_address(w.layer, kind, which, c.first);
                        GGML_ASSERT(aux_src != nullptr);
                        CUDA_CHECK(cudaMemcpyAsync(l1_->aux_slice_address(cls, kind, which, first + c.second), aux_src,
                            aux_bytes, cudaMemcpyHostToDevice, stage_stream_));
                        stage_bytes_ += aux_bytes;
                    }
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
        ggml_account_add(GGML_ACCOUNT_CTRL_STAGE_BYTES, (double) (stage_bytes_ - stage_bytes_before));
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
                    for (int which = 0; l1_->aux_enabled() && which < geometry::n_aux; ++which) {
                        const size_t aux_bytes = geo_.class_aux_bytes(cls, kind, which);
                        if (aux_bytes == 0) {
                            continue;
                        }
                        back.resize(aux_bytes);
                        CUDA_CHECK(cudaMemcpy(back.data(), l1_->aux_slice_address(cls, kind, which, slot), aux_bytes,
                            cudaMemcpyDeviceToHost));
                        if (memcmp(back.data(), l1_->aux_host_address(layer, kind, which, e), aux_bytes) != 0) {
                            GGML_ABORT("expert cache: VRAM staging of layer %d expert %d kind %d: aux row %d differs from its tensor",
                                layer, e, kind, which);
                        }
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
        lend_scores_ = counts != nullptr ? scores : std::vector<uint64_t>();
        lend_want_.clear();
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
            host_capacities[cls] = geo_.class_slices((int) cls) - capacities_[cls];
            if (host_capacities[cls] < 0) {
                disable_locked("more VRAM slots than experts");
                return false;
            }
        }
        if (cfg_.l2_bytes != 0 && !size_tier_locked(initial.selected, host_capacities)) {
            return false;
        }
        // the seed plan is the installed plan until the first install (a cold cache has none)
        lend_set_want_locked(counts != nullptr ? &initial.selected : nullptr);

        // Plan time, before the first expert byte is copied: a host requirement larger than the
        // installed memory can never be met, so it is a configuration error, not a slow run.
        size_t host_required = cfg_.l2_bytes;
        // exclusive mode without a finite tier: the prefill lend grows the host arenas (expert-host-layout.h)
        const bool host_chunks = !tier_ && cfg_.lend;
        if (host_required == 0) {
            for (size_t cls = 0; cls < host_capacities.size(); ++cls) {
                for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                    const int slots = host_capacities[cls] + cfg_.spare_slots;
                    host_required += host_chunks ?
                        host_chunked_bytes(geo_.class_bytes[cls][kind], arena_tail_bytes(geo_, (int) cls, kind), slots,
                            host_chunk_slots(geo_, (int) cls, host_chunk_mib << 20)) :
                        size_t(slots)*geo_.class_bytes[cls][kind] + arena_tail_bytes(geo_, (int) cls, kind);
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
        if (!l1_->allocate(capacities_, device_, cfg_.spare_slots, stage_class_slots_.empty() ? nullptr : &stage_class_slots_)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
        allocate_stage_locked();
        host_.reset(new host_arena(geo_));
        if (!tier_ && cfg_.lend) {
            // the prefill lend grows the host capacity by the experts it evicts: an address table and chunked arenas
            host_->enable_addresses();
            host_->enable_chunks(host_chunk_mib << 20);
        }
        if (!host_->allocate(host_capacities, cfg_.spare_slots, device_,
                tier_ && tier_->class_layout() ? &tier_->storage() : nullptr)) {
            host_.reset();
            l1_.reset();
            profiler_.reset();
            disable_locked("host arena allocation failed");
            return false;
        }
        if (host_->addresses_enabled()) {
            size_t table = size_t(geo_.n_routed_layers())*geometry::n_kinds*size_t(geo_.n_experts)*sizeof(uint64_t);
            std::string per;
            for (size_t c = 0; c < host_capacities.size(); ++c) { per += (c ? "," : "") + std::to_string(host_->chunk_slots((int) c)); }
            GGML_LOG_INFO("expert cache: host address table on (%zu KiB of routed-layer entries); host arenas in %zu chunks "
                          "of about %zu MiB (slots per chunk %s), %.1f MiB of chunk slack\n", table/1024, host_->chunk_count(),
                host_chunk_mib, per.c_str(), double(host_->slack_bytes())/double(1 << 20));
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
        for (int expert = first; expert < last && expert < geo_.experts(layer); ++expert) {
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
            for (int e = 0; e < geo_.experts(l); ++e) for (int k = 0; k < geometry::n_kinds; ++k) {
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
            for (int e = 0; e < geo_.experts(l); ++e) {
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

    // A member's rows of a joint [layer][expert] table, in the member's own stride: what its profile store keeps.
    // With one expert count this is the plain slice of the member's layers.
    std::vector<uint64_t> member_rows_locked(const member_state & mb, const std::vector<uint64_t> & joint) const {
        const size_t width = (size_t) std::max(mb.geo.n_experts, 0), stride = (size_t) geo_.n_experts;
        std::vector<uint64_t> out;
        out.reserve(size_t(mb.geo.n_layers)*width);
        for (int l = 0; l < mb.geo.n_layers; ++l) {
            const size_t row = size_t(mb.offset + l)*stride;
            for (size_t e = 0; e < width && row + e < joint.size(); ++e) {
                out.push_back(joint[row + e]);
            }
        }
        return out;
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
            // the store keeps the model's own [layer][expert] rows (its expert count); the joint rows
            // have the joint stride, and the experts a model does not have stay 0
            const size_t width = (size_t) std::max(members_[m].geo.n_experts, 0);
            const size_t n     = size_t(members_[m].geo.n_layers)*width;
            for (size_t i = 0; i < part.size() && i < n; ++i) {
                const size_t at = (size_t(members_[m].offset) + i/width)*size_t(geo_.n_experts) + i%width;
                if (at < out.size()) {
                    out[at] = (uint64_t) std::llround(double(part[i])*f);
                }
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
                p.n_experts = mb.geo.n_experts;   // the model's own count: its records do not depend on the joint stride
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
    bool          unequal_ = false;            // the prefill lend's shrink install: exclusive unequal exchange
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
    // decode pool (expert-ub-pool.cuh)
    ub_pool_config ub_cfg_;
    std::unique_ptr<ub_pool> ub_;
    std::atomic<bool> ub_on_{ false };      // the step runs; read by profile_ids without the mutex
    std::vector<uint8_t> ub_layer_;         // [joint layer] has positions; fixed while ub_on_
    int ub_member_ = 0, ub_rows_ = 1, ub_n_layers_ = 0;
    uint64_t * ub_tables_ = nullptr;        // the pool's own address tables [layer][kind][expert], or null
    const void * ub_owner_ = nullptr;       // the compute context that lent its buffer
    cudaStream_t ub_stream_ = nullptr;      // its stream
    const void * ub_last_cctx_ = nullptr;   // ring admission: the context of the last graph, and its stream
    cudaStream_t ub_last_stream_ = nullptr;
    const void * ub_member_base_ = nullptr;          // base of the cached model's largest compute buffer
    std::vector<const void *> ub_ring_judged_;       // joint member contexts found computing in that buffer
    void * ub_buf_base_ = nullptr;          // the buffer of the last graph
    void * ub_lent_base_ = nullptr;         // the buffer the pool is lent from
    size_t ub_hwm_ = 0, ub_begin_ = 0;      // largest decode extent seen, start of the lent region (offsets)
    uint64_t ub_lends_ = 0, ub_reclaims_ = 0, ub_calls_ = 0;
    size_t   ub_max_size_ = 0;              // the largest compute buffer (view) seen
    std::map<const void *, size_t> ub_hwm_of_;   // decode high-water mark per compute buffer base
    // the prefill lend (expert-lend.h)
    std::atomic<bool> lend_on_{ false };           // read by compute_spare, graph_begin and profile_ids without the mutex
    int lend_rows_ = 1;                            // decode launches of at most this many rows end an implicit lend
    lend_book lend_;
    std::vector<uint64_t> lend_scores_;            // the installed plan's scores, empty = cold
    std::vector<uint8_t>  lend_want_;              // [layer*experts + expert] the installed decode plan's L1 experts, empty = none
    struct lend_held_piece {
        ggml_expert_lend_piece req{};
        void * ptr = nullptr;
        ggml_backend_buffer_t buf = nullptr;
    };
    std::vector<lend_held_piece> lend_pieces_;
    cudaStream_t lend_holder_stream_ = nullptr;
    cudaStream_t lend_stream_ = nullptr;           // the lend's own copies (tables, heads)
    std::vector<size_t> lend_guard_;               // [class*3 + kind] zero bytes at the start of a lent slot
    size_t lend_last_d2h_ = 0, lend_last_moves_ = 0;   // the last eviction install's VRAM -> host bytes and its moves
    size_t install_last_d2h_ = 0;                  // the VRAM -> host bytes of the last tier install (its d2h)
    int32_t * lend_restore_dev_ = nullptr;         // [layer][expert]
    uint32_t * lend_bits_dev_ = nullptr;           // [layer][words]
    lend_restore_desc * lend_desc_dev_ = nullptr;  // [layer]
    unsigned int * lend_tickets_dev_ = nullptr;    // [layer]
    lend_live * lend_live_host_ = nullptr;         // mapped
    lend_live * lend_live_dev_ = nullptr;
    unsigned long long lend_live_mark_ = 0;        // restores at the last log line
    ggml_expert_lend_counts lend_counts_{};
    bool lend_scratch_warned_ = false;
    // the expert prefetch (RANMA_EXPERT_PREFETCH)
    xp_config xp_cfg_;
    std::atomic<bool> xp_on_{ false };      // read by xp_publish without the mutex
    std::atomic<bool> xp_stop_{ false };
    std::thread xp_thread_;
    cudaStream_t xp_stream_ = nullptr;
    cudaEvent_t xp_event_ = nullptr;
    std::mutex xp_issue_mutex_;             // the relay's issue section against a reclaim
    bool xp_paused_ = false;                // under xp_issue_mutex_
    double xp_tick_us_ = 0.01;
    std::vector<xp_layer> xp_l_;            // [joint layer], the relay thread's own
    std::vector<uint8_t> xp_reset_;         // [joint layer][3] restart learning (set by prefetch_layers, cleared by the relay)
    std::mutex xp_mask_mutex_;
    int xp_profile_[3] = { 2, 2, 2 };       // predicted layer profile per row bucket (see prefetch_layers)
    uint64_t xp_graphs_ = 0, xp_next_eval_ = 0, xp_next_probe_ = 0, xp_mask_changes_ = 0;
    std::vector<int> xp_layers_;            // the layers with positions
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

void ggml_cuda_expert_prefetch_publish(ggml_backend_cuda_context & ctx, const ggml_tensor * order) {
    instance().xp_publish(ctx, order);
}

void ggml_cuda_expert_graph_begin(ggml_backend_cuda_context & ctx, int n_op, int64_t n_tok) {
    instance().graph_begin(&ctx, n_op, n_tok);
}

void * ggml_cuda_expert_lent(ggml_backend_cuda_context & ctx, int role, size_t bytes) {
    return instance().lent(&ctx, role, bytes);
}

bool ggml_cuda_expert_lend_on() {
    return instance().lend_is_on();
}

void ggml_cuda_expert_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    instance().layer_done(ctx, src0);
}

void ggml_cuda_expert_before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    instance().before_read(ctx, src0);
}

bool ggml_cuda_expert_layer_staged(const ggml_tensor * src0) {
    return instance().tier_layer_staged(src0);
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
static bool iface_compute_spare(ggml_backend_t backend, void * base, size_t size, size_t extent, int32_t n_tokens) {
    return instance().compute_spare(backend, base, size, extent, n_tokens);
}
static int32_t iface_compact_rows(void) { return instance().compact_rows(); }
static uint32_t iface_prefetch_layers(int32_t n_rows, uint64_t * mask, int32_t n_words) {
    return instance().prefetch_layers(n_rows, mask, n_words);
}
static bool iface_pool_counts(ggml_expert_pool_counts * out) { return instance().pool_counts(out); }
static bool iface_lend_fits(ggml_backend_t backend, const ggml_expert_lend_request * request) {
    return instance().iface_lend_fits(backend, request);
}
static bool iface_lend_acquire(ggml_backend_t backend, const ggml_expert_lend_request * request, ggml_expert_lend_grant * out) {
    return instance().iface_lend_acquire(backend, request, out);
}
static bool iface_lend_release(ggml_backend_t backend) { return instance().iface_lend_release(backend); }
static bool iface_lend_counts(ggml_expert_lend_counts * out) { return instance().iface_lend_counts(out); }

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
    /* .compute_spare    = */ iface_compute_spare,
    /* .compact_rows     = */ iface_compact_rows,
    /* .pool_counts      = */ iface_pool_counts,
    /* .prefetch_layers  = */ iface_prefetch_layers,
    /* .lend_fits        = */ iface_lend_fits,
    /* .lend_acquire     = */ iface_lend_acquire,
    /* .lend_release     = */ iface_lend_release,
    /* .lend_counts      = */ iface_lend_counts,
};

const ggml_expert_iface * ggml_backend_cuda_expert_iface(void) {
    return &g_expert_iface;
}

#endif // GGML_USE_HIP
