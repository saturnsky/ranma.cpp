#include "expert-controller.cuh"

#if defined(GGML_USE_HIP)

#include "expert-geometry.h"
#include "expert-host.cuh"
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
        if (cfg_.l1_bytes == 0) {
            // Nothing to do for this model; stay unconfigured so that a later model may configure.
            return false;
        }
        if (cfg_.mode != GGML_EXPERT_MODE_INCLUSIVE && cfg_.mode != GGML_EXPERT_MODE_EXCLUSIVE) {
            GGML_LOG_WARN("expert cache: unknown mode %d; cache off\n", (int) cfg_.mode);
            return false;
        }
        if (cfg_.policy < GGML_EXPERT_POLICY_ADAPTIVE || cfg_.policy > GGML_EXPERT_POLICY_STATIC) {
            GGML_LOG_ERROR("expert cache: invalid placement policy\n");
            return false;
        }
        if (profile_dir_.empty()) {
            cfg_.policy = GGML_EXPERT_POLICY_STATIC;
            cfg_.freeze = true;
            GGML_LOG_INFO("expert cache: no profile directory; seeded fixed placement, no records or installs\n");
        }
        if (cfg_.mode == GGML_EXPERT_MODE_INCLUSIVE || cfg_.policy != GGML_EXPERT_POLICY_ADAPTIVE || !cfg_.l1_bytes) { cfg_.spare_slots = 0; }
        if (cfg_.mode == GGML_EXPERT_MODE_EXCLUSIVE) {
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
        // the host tensor. Slow (the whole arena crosses PCIe twice), meant for the verification
        // gate of a port stage, not for serving.
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
        if (state_ != state_t::configured || cfg_.mode != GGML_EXPERT_MODE_EXCLUSIVE ||
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
        out->host_bytes      = host_ ? host_->host_bytes() : 0;
        out->n_banks         = (uint32_t) banks_.size();
        return true;
    }

    // ---- banks and plans ------------------------------------------------------------------

    bool bank_open(const char * label, ggml_expert_bank_id * out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!profiling_enabled() || state_ != state_t::installed || label == nullptr || label[0] == '\0' || out == nullptr) {
            return false;
        }
        return open_bank_locked(label, out);
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
        (void) save_start;
        ++b.commits;
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
        if (plan.selected == l1_->selected()) {
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
        // the plan that was installed until now is only reachable while it is a bank's newest one
        finish_plan_locked(id);
        verify_all_locked();
        return true;
    }

    bool profile_select(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank) {
        if (state_ != state_t::installed || backend == nullptr || !ggml_backend_is_cuda(backend)) {
            return false;
        }
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
        if (cuda_ctx->device != device_) {
            return false;
        }
        ggml_cuda_set_device(device_);
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

    bool profiling_enabled() const { return !profile_dir_.empty(); }
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
        if (!l1_->assign_exclusive(initial.selected)) {
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
                if (gpu < 0 && !home) { skipped += k == 0; continue; }
                if (!digest_known_[digest_index_locked(l, k, e)]) { skipped += k == 0; continue; }
                auto check = [&](const void * data) {
                    const size_t i = digest_index_locked(l, k, e);
                    const auto found = digest_of(data, geo_.class_bytes[cls][k]);
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
            if (open_bank_locked(label.c_str(), &bank) && banks_[bank].store_usable &&
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

    bool open_bank_locked(const char * label, ggml_expert_bank_id * out) {
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
    ggml_backend_buffer_t exclusive_buffer_ = nullptr; // owned by the model, not by the controller
    bool digests_enabled_ = false;
    std::vector<slice_digest> digests_;      // [(layer*n_kinds + kind)*n_experts + expert]
    std::vector<uint8_t>      digest_known_;
    std::vector<bank_state> banks_;
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
static bool iface_bank_open(const char * label, ggml_expert_bank_id * out) { return instance().bank_open(label, out); }
static bool iface_bank_mark(ggml_expert_bank_id bank) { return instance().bank_zero(bank); }
static bool iface_bank_commit(ggml_expert_bank_id bank, const ggml_expert_record * record, ggml_expert_plan_id * out) {
    return instance().bank_commit(bank, record, out);
}
static bool iface_bank_discard(ggml_expert_bank_id bank) { return instance().bank_zero(bank); }
static bool iface_plan_install(ggml_expert_plan_id plan) { return instance().plan_install(plan); }
static bool iface_profile_select(ggml_backend_t backend, int32_t row_begin, int32_t row_end, ggml_expert_bank_id bank) {
    return instance().profile_select(backend, row_begin, row_end, bank);
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
    /* .memory           = */ iface_memory,
};

const ggml_expert_iface * ggml_backend_cuda_expert_iface(void) {
    return &g_expert_iface;
}

#endif // GGML_USE_HIP
