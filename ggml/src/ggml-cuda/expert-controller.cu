#include "expert-controller.cuh"

#if defined(GGML_USE_HIP)

#include "expert-geometry.h"
#include "expert-host.cuh"
#include "expert-l2.cuh"
#include "expert-os.h"
#include "expert-plan.h"
#include "expert-profiler.cuh"
#include "expert-profile-store.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

// Histograms are cheap (n_counts * 4 bytes each), so a fixed number of banks is allocated at
// finalize time and bank_open hands them out; the caller's policy decides what each one means.
static constexpr uint32_t max_banks = 4;

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

struct bank_state {
    std::string label;
    bool prompt = false;                    // the caller's policy says this bank counts prompt processing
    std::unique_ptr<profile_store> store;   // null when profiling is off for this bank
    bool store_usable = false;              // open() returned ok/missing/corrupt, not incompatible
    ggml_expert_plan_id latest_plan = GGML_EXPERT_PLAN_NONE;
    uint64_t commits = 0;
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
            // A second model in the same process runs without the cache (decision: one model per process).
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
        state_ = state_t::configured;
        return true;
    }

    // Owned host storage redirects loader writes for exclusive, finite inclusive and Off policies.
    // Unlimited inclusive uses the ordinary host buffer and register_context.
    ggml_backend_buffer_t alloc_context(ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * identity) {
        std::lock_guard<std::mutex> lock(mutex_);
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
        if (!parse_expert_tensor_name(ggml_get_name(root), layer, kind) || layer >= geo_.n_layers ||
                geo_.tensors[layer][kind] != root) {
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
        if (state_ != state_t::registered) {
            return false;
        }
        ggml_cuda_set_device(device_);

        if (l1_ && l1_->owns_host_storage()) {
            // The delegate buffer is not in the model's buffer list, so the backend's own
            // post-load step for host buffers has not seen it. Do it here, while the loader's
            // writes to it are final, exactly as llama does for the buffers it knows.
            if (!finalize_delegate_buffer_locked()) {
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

        if (!allocate_profiler_locked()) { return false; }
        l1_.reset(new l1_arena(geo_));
        if (!l1_->allocate(capacities_, device_)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
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
        GGML_LOG_INFO("expert cache: installed %zu MiB in VRAM (%zu slices, %.1f ms); tables and profiler %zu KiB\n",
            l1_->device_bytes()/(1024*1024), stats.copied, stats.ms,
            (table_bytes + (profiler_ ? profiler_->device_bytes() : 0))/1024);
        return true;
    }

    void release(ggml_context * ctx) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == state_t::unconfigured || (ctx_ != nullptr && ctx != ctx_)) {
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
        l1_.reset();
        host_.reset();
        profiler_.reset();
        exclusive_buffer_ = nullptr;
        digests_.clear();
        digest_known_.clear();
        digests_enabled_ = false;
        geo_ = geometry();
        ctx_ = nullptr;
        device_ = -1;
        disabled_reason_ = "";
        capacities_.clear();
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
        if (!b.store_usable) {
            return false;
        }
        std::vector<uint64_t> delta;
        if (!profiler_->read_bank(bank, delta, /*zero_after=*/true)) {
            return false;
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
        if (total != 0 && !b.store->checkpoint(delta, meta)) {
            GGML_LOG_WARN("expert cache: bank '%s' failed to store a record: %s\n", b.label.c_str(), b.store->last_error().c_str());
            return false;
        }
        const double save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - save_start).count();
        ++b.commits;
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) { report_round_locked(b.label, b.commits, *record, delta, save_ms); }
        const std::vector<uint64_t> & scores = b.store->scores();
        placement_inputs in;
        in.geo              = &geo_;
        in.counts           = b.store->total_selections() != 0 ? scores.data() : nullptr;
        in.budget_bytes     = SIZE_MAX;
        in.exclusive        = l1_ && l1_->owns_host_storage();
        in.fixed_capacities = &capacities_;
        placement next = plan_placement(in);

        plan_state plan;
        plan.bank     = bank;
        plan.selected = std::move(next.selected);
        plan.stats    = next.stats;
        if (tier_ && b.store->total_selections() != 0) {
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
                b.label.c_str(), (unsigned long long) b.commits, (unsigned long long) total, b.store->window().size(),
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
        const plan_state & plan = it->second;
        if (banks_[plan.bank].latest_plan != id) {
            GGML_LOG_WARN("expert cache: plan %u of bank '%s' is stale; not installed\n", id, banks_[plan.bank].label.c_str());
            return false;
        }
        if (cfg_.freeze || cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE) {
            GGML_LOG_INFO("expert cache: frozen; plan %u not installed\n", id);
            return false;
        }
        // Installing what is already in the arena costs nothing: no drain, no copies, one line.
        if (!tier_ && plan.selected == l1_->selected()) {
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
        if (tier_) {
            return install_tier_locked(id, plan, /*seed =*/ false);
        }
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
        return true;
    }

    bool profile_select(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank) {
        if (state_ != state_t::installed || cfg_.policy == GGML_EXPERT_POLICY_OFF || backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
        if (cuda_ctx->device != device_) {
            return false;
        }
        ggml_cuda_set_device(device_);
        if (tier_ && bank < n_banks_.load(std::memory_order_acquire)) {
            tier_->set_phase(bank_prompt_[bank].load(std::memory_order_acquire) != 0);
        }
        return profiler_->select(row_begin, row_end, bank, cuda_ctx->stream());
    }

    // ---- hot path ---------------------------------------------------------------------------

    ggml_cuda_expert_lookup lookup(const ggml_tensor * tensor) const noexcept {
        if (state_ != state_t::installed || tensor == nullptr) {
            return {};
        }
        int layer = -1;
        int kind  = -1;
        if (!parse_expert_tensor_name(tensor->name, layer, kind) || layer >= geo_.n_layers ||
                geo_.tensors[layer][kind] != tensor || ggml_cuda_get_device() != device_) {
            return {};
        }
        return l1_->lookup(layer, kind);
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
        if (end == ids->name + sizeof(prefix) - 1 || *end != '\0' || layer < 0 || layer >= geo_.n_layers ||
                geo_.layer_class[layer] < 0) {
            return;
        }
        if (ids->type != GGML_TYPE_I32 || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t)) {
            return;
        }
        if (profiler_) {
        launch_profile_ids((const int32_t *) ids->data, (int) ids->ne[1], (int) (ids->nb[1]/sizeof(int32_t)), (int) ids->ne[0],
            geo_.n_experts, profiler_->counts_base() + size_t(layer)*geo_.n_experts, profiler_->n_counts(), profiler_->n_banks(),
            profiler_->selection(), ctx.stream());
        }
        // The SSD tier needs the same ids: this is the one site that sees both the fused and the
        // unfused router, and it is ordered before the layer's MUL_MAT_ID on the same stream.
        tier_route(ctx, layer, ids);
    }

    void before_read(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_before_read(ctx, src0);
    }

    void layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        tier_layer_done(ctx, src0);
    }

    // ---- the SSD tier -------------------------------------------------------------------------

    // Called by the loader for every routed expert tensor in the exclusive buffer type.
    bool tier_set_backing(const ggml_tensor * tensor, int file_index, const char * path, uint64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        int layer = -1;
        int kind  = -1;
        if (!tier_ || tensor == nullptr || !parse_expert_tensor_name(ggml_get_name(tensor), layer, kind) ||
                layer >= geo_.n_layers || geo_.tensors[layer][kind] != tensor) {
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
        if (!tier_ || tensor == nullptr || !parse_expert_tensor_name(ggml_get_name(tensor), layer, kind) ||
                layer >= geo_.n_layers || expert < 0 || expert >= geo_.n_experts) {
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
        if (parse_expert_tensor_name(src0->name, layer, kind) && layer < geo_.n_layers &&
                geo_.tensors[layer][kind] == src0) {
            tier_->wait_kind(layer, kind, ctx.stream());
        }
    }

    void tier_layer_done(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
        if (!tier_ || state_ != state_t::installed || src0 == nullptr || ctx.device != device_) {
            return;
        }
        int layer = -1;
        int kind  = -1;
        if (parse_expert_tensor_name(src0->name, layer, kind) && kind == 2 && layer < geo_.n_layers &&
                geo_.tensors[layer][kind] == src0) {
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
        geo_       = std::move(geo);
        ctx_       = ctx;
        device_    = device;
        identity_  = identity ? identity : "";
        signature_ = identity_ + "\n" + geo_.signature();
        state_     = state_t::registered;
        GGML_LOG_INFO("expert cache: registered %s%s: %d routed layers x %d experts, %zu size classes, device %d\n",
            identity_.c_str(), owns_host ? " for owned host storage" : "", geo_.n_routed_layers(),
            geo_.n_experts, geo_.class_bytes.size(), device_);
        return true;
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

    tier_inputs tier_inputs_locked(const std::vector<uint64_t> & scores) const {
        tier_inputs in;
        in.inclusive = cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE;
        in.minimum_capacities = in.inclusive ? &capacities_ : nullptr;
        in.geo = &geo_;
        in.counts = scores.empty() ? nullptr : scores.data();
        in.vram = &tier_vram_;
        in.slot_pitch = &host_pitch_;
        return in;
    }

    // Sizes the ring out of the host budget and cuts the rest three ways. `host_capacities` comes in
    // as the two-tier capacity (everything that is not in VRAM) and goes out as the finite one.
    bool size_tier_locked(const std::vector<std::vector<int32_t>> & selected, std::vector<int> & host_capacities) {
        tier_.reset(new l2_tier(geo_, l2_debug_config(cfg_, verify_all_)));
        if (!tier_->sized()) {
            tier_.reset();
            disable_locked("the SSD tier could not size its staging ring");
            return false;
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
        l1_->attach_locations([tier](int cls, int kind, expert_location at) { return tier->location_address(cls, kind, at); });
        l1_arena::tier_reader reader;
        reader.read    = [tier](const std::vector<l2_read> & reads, std::string & why) { return tier->read_install(reads, why); };
        reader.address = [tier](const l2_read & read) { return tier->read_address(read); };
        // Constant across a repartition: install_read_slots is also bounded by the decode ring,
        // which is the smaller of the two ring sizes.
        reader.slots   = tier_->install_read_slots();
        l1_->attach_tier_reader(std::move(reader));
        return true;
    }

    l2_host_geometry host_geometry_locked() const {
        l2_host_geometry out;
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

    bool start_tier_locked() {
        if (!tier_->map()) {
            abort_locked("SSD tier ring registration failed");
        }
        std::string reason;
        if (!tier_->open_files(reason)) {
            abort_locked(("SSD tier: " + reason).c_str());
        }
        verify_tier_backing_locked();
        l2_tier * tier = tier_.get();
        l1_->attach_addresses([tier](int layer, int kind) { return tier->addresses(layer, kind); });
        tier_->set_homes(l1_->host_slots(), l1_->locations(), host_geometry_locked());
        tier_->start_worker();
        return true;
    }

    // The tier reads the GGUF itself, so its idea of where an expert lives has to agree with the
    // loader's. One host resident expert per routed layer and kind is read back from the file and
    // compared with the bytes the loader put in the arena. Always on: it is a few slices and it is
    // the only check that the tier and the loader read the same offsets.
    void verify_tier_backing_locked() {
        const std::vector<std::vector<int32_t>> & gpu_table  = l1_->host_slots();
        const std::vector<std::vector<int32_t>> & host_table = l1_->arena_slots();
        std::vector<uint8_t> from_file;
        size_t checked = 0;
        for (int layer = 0; layer < geo_.n_layers; ++layer) {
            const int cls = geo_.layer_class[layer];
            if (cls < 0) {
                continue;
            }
            int expert = -1;
            for (int candidate = 0; candidate < geo_.n_experts && expert < 0; ++candidate) {
                if (gpu_table[layer][candidate] < 0 && host_table[layer][candidate] >= 0) {
                    expert = candidate;
                }
            }
            if (expert < 0) {
                continue;
            }
            for (int kind = 0; kind < geometry::n_kinds; ++kind) {
                const size_t stride = geo_.class_bytes[cls][kind];
                from_file.assign(stride, 0);
                std::string why;
                if (!tier_->read_slice(layer, kind, expert, from_file.data(), why)) {
                    abort_locked(("SSD tier backing check: " + why).c_str());
                }
                const void * arena = host_->slice(cls, kind, host_table[layer][expert]);
                if (arena == nullptr || memcmp(from_file.data(), arena, stride) != 0) {
                    GGML_ABORT("expert cache: the SSD tier reads layer %d kind %d expert %d from the wrong "
                               "place in the file", layer, kind, expert);
                }
                ++checked;
            }
        }
        GGML_LOG_INFO("expert cache: SSD tier backing checked on %zu slices against the loaded weights\n", checked);
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
            {cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE, cfg_.spare_slots}, true);
        if (!tx.valid) { abort_locked(("install transaction: " + tx.reason).c_str()); }
        if (!l1_->execute(tx)) { abort_locked("the install mover refused a move"); }
        std::string reason;
        for (const auto & move : tx.moves) for (int k = 0; k < geometry::n_kinds; ++k) {
            tier_->finish_write(move.cls, k, move.to);
        }
        // The old lent sources stay readable until every move has completed.
        const bool repartition = tier_->set_prompt_ring(prompt);
        if (!l1_->assign_tier(tx.gpu_slots, tx.host, layout, plan.selected, tx.gpu_spares) ||
                !l1_->verify_current_assignment(reason)) { abort_locked(("install assignment: " + reason).c_str()); }
        tier_->set_homes(tx.gpu_slots, tx.host, host_geometry_locked());
        const size_t host_total = host_->host_bytes() + tier_->host_bytes();
        GGML_ASSERT(host_total <= cfg_.l2_bytes);
        if (repartition) {
            GGML_LOG_INFO("expert cache: L2 repartition ring=%d->%d lent_slots=%d host_total=%zu\n",
                ring_before, tier_->ring_count(), tier_->prompt_slots() - tier_->ring_count(), host_total);
        }
        tier_->start_worker();
        log_budget_split_locked(host_plan, prompt);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        GGML_LOG_INFO("expert cache: installed plan %u of bank '%s': mode=%s-l2 ring=%s%s retained=%zu/%zu "
                      "h2d=%zu slices / %zu MiB d2h=%zu slices / %zu MiB ssd=%zu slices / %zu MiB in %.1f ms\n",
            id, banks_[plan.bank].label.c_str(), cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE ? "inclusive" : "exclusive",
            prompt ? "prompt" : "decode", repartition ? " (changed)" : "", tx.retained_gpu, tx.retained_host,
            tx.h2d_slices, tx.h2d_bytes/(1024*1024), tx.d2h_slices, tx.d2h_bytes/(1024*1024),
            tx.ssd_slices, tx.ssd_bytes/(1024*1024), ms);
        if (cfg_.log_mask & GGML_EXPERT_LOG_L2) {
            GGML_LOG_INFO("expert_metrics {\"kind\":\"install\",\"bank\":\"%s\",\"plan\":%u,\"install_ms\":%.6f,"
                          "\"h2d_bytes\":%zu,\"d2h_bytes\":%zu,\"ssd_bytes\":%zu,\"ring_before\":%d,\"ring_after\":%d}\n",
                banks_[plan.bank].label.c_str(), id, ms, tx.h2d_bytes, tx.d2h_bytes, tx.ssd_bytes, ring_before, tier_->ring_count());
        }
        finish_plan_locked(id); verify_all_locked();
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
        const int    slots  = tier_->prompt_slots();
        const size_t ring   = tier_->ring_bytes(slots);
        const size_t meta   = tier_->metadata_bytes();
        const size_t pad    = arena_tail_total(geo_);
        const size_t spare  = spare_bytes_locked();
        const size_t tables = table_bytes_locked(true);
        const size_t accounted = payload + ring + meta + pad + spare;
        // Written as an `expert_metrics` record and not gated by the log mask: it is a default
        // level line, and llama-bench's log callback only lets that prefix through.
        GGML_LOG_INFO("expert_metrics {\"kind\":\"budget\",\"phase\":\"%s\",\"l2_budget\":%zu,"
                      "\"resident_payload\":%zu,\"resident_slices\":%zu,\"host_arena\":%zu,"
                      "\"ring\":%zu,\"ring_slots\":%d,\"ring_pitch\":[%zu,%zu,%zu],"
                      "\"lent_slices\":%zu,\"lent_payload\":%zu,\"metadata\":%zu,\"padding\":%zu,"
                      "\"spare\":%zu,\"vram_tables\":%zu,\"file_slices\":%zu,\"file_bytes\":%zu,"
                      "\"remainder\":%lld}\n",
            prompt ? "prompt" : "decode", cfg_.l2_bytes, payload, placed,
            host_->host_bytes(), ring, slots,
            tier_->slot_pitch(0), tier_->slot_pitch(1), tier_->slot_pitch(2),
            lent, lent_bytes, meta, pad, spare, tables,
            host_plan.ssd_slices, host_plan.ssd_bytes,
            (long long) cfg_.l2_bytes - (long long) accounted);
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
    bool finalize_delegate_buffer_locked() {
        ggml_backend_buffer_t delegate = exclusive_buffer_ != nullptr ?
            exclusive_delegate_buffer(exclusive_buffer_) : nullptr;
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
        GGML_LOG_INFO("expert cache: plan budget=%zu MiB choices=%d per-layer=%d..%d selection-hit=%.2f%% byte-hit=%.2f%%\n",
            cfg_.l1_bytes/(1024*1024), stats.choices, stats.per_layer_min, stats.per_layer_max,
            100.0*stats.selection_hit(), 100.0*stats.byte_hit());
    }

    // Free slots that the owned-host exchange rotates through.
    size_t spare_bytes_locked() const {
        size_t bytes = 0;
        for (int cls = 0; cls < (int) geo_.class_bytes.size(); ++cls) {
            bytes += size_t(cfg_.spare_slots)*geo_.class_total_bytes(cls);
        }
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
        profiler_.reset(new profiler(geo_, max_banks));
        if (profiler_->allocate(device_)) { return true; }
        profiler_.reset(); disable_locked("profiler allocation failed"); return false;
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
        if (!l1_->allocate(capacities_, device_, cfg_.spare_slots)) {
            l1_.reset();
            profiler_.reset();
            disable_locked("arena allocation failed");
            return false;
        }
        host_.reset(new host_arena(geo_));
        if (!host_->allocate(host_capacities, cfg_.spare_slots, device_)) {
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
        for (size_t pos = 0; pos <= initial_bank_.size(); ) {
            const size_t sep = std::min(initial_bank_.find(',', pos), initial_bank_.size());
            const std::string label = initial_bank_.substr(pos, sep - pos);
            pos = sep + 1;
            if (label.empty()) {
                continue;
            }
            ggml_expert_bank_id bank = GGML_EXPERT_BANK_NONE;
            // A seed bank is opened before the caller names its phase; init reopens it by label.
            if (open_bank_locked(label.c_str(), false, &bank) && banks_[bank].store_usable &&
                    banks_[bank].store->total_selections() != 0) {
                scores = banks_[bank].store->scores();
                GGML_LOG_INFO("expert cache: bank '%s' seeds the plan from %zu stored records\n",
                    label.c_str(), banks_[bank].store->window().size());
                return scores.data();
            }
            GGML_LOG_INFO("expert cache: bank '%s' has no usable profile\n", label.c_str());
        }
        GGML_LOG_INFO("expert cache: no bank of '%s' has a stored profile; starting cold\n", initial_bank_.c_str());
        return nullptr;
    }

    bool open_bank_locked(const char * label, bool prompt_bank, ggml_expert_bank_id * out) {
        for (size_t i = 0; i < banks_.size(); ++i) {
            if (banks_[i].label == label) {
                *out = (ggml_expert_bank_id) i;
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
            profile_store_params p;
            p.bank_dir  = std::filesystem::path(profile_dir_)/label;
            p.label     = label;
            p.signature = signature_;
            p.n_layers  = geo_.n_layers;
            p.n_experts = geo_.n_experts;
            p.archive   = cfg_.profile_archive;
            b.store.reset(new profile_store(p));
            if (cfg_.profile_reset) {
                b.store->reset();
            }
            const profile_store_status st = b.store->open();
            b.store_usable = st != profile_store_status::incompatible;
            if (!b.store_usable) {
                GGML_LOG_WARN("expert cache: profile at %s belongs to another model or format; ignored (use --expert-profile-reset to replace it)\n",
                    p.bank_dir.string().c_str());
            } else if (b.store->skipped_records() != 0) {
                GGML_LOG_WARN("expert cache: profile at %s: %zu unreadable records skipped\n",
                    p.bank_dir.string().c_str(), b.store->skipped_records());
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
    ggml_context * ctx_ = nullptr;
    int device_ = -1;
    std::string disabled_reason_;
    bool verify_all_ = false;
    std::vector<int> capacities_;
    std::unique_ptr<profiler> profiler_;
    std::unique_ptr<l1_arena> l1_;
    std::unique_ptr<host_arena> host_;
    std::unique_ptr<l2_tier> tier_;
    std::vector<size_t> host_pitch_;
    std::vector<int>    host_capacities_base_;
    std::vector<int>    lent_capacities_, lent_starts_;
    std::vector<uint64_t> tier_scores_;
    std::vector<std::vector<int32_t>> tier_vram_;
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
    /* .iface   = */ { exclusive_buft_name, exclusive_buft_alloc_buffer, exclusive_buft_alignment,
                       nullptr, nullptr, nullptr },
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
};

const ggml_expert_iface * ggml_backend_cuda_expert_iface(void) {
    return &g_expert_iface;
}

#endif // GGML_USE_HIP
