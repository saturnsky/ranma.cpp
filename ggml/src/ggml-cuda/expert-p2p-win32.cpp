// Peer-to-peer reads into VRAM for the expert cache. See expert-p2p.h. The Windows implementation
// needs D3D12, DXGI and the IoRing API; all three are resolved at run time, so the backend links
// no new library and runs unchanged where they are missing. Other platforms get the stub below.

#include "expert-p2p.h"

#include <cstdlib>
#include <cstring>

namespace ggml_cuda_expert {

p2p_config p2p_config_from_env(std::string & notes) {
    p2p_config cfg;
    auto number = [&](const char * name, long long fallback, long long lo, long long hi) {
        const char * text = getenv(name);
        if (text == nullptr || *text == '\0') { return fallback; }
        char * end = nullptr;
        const long long v = strtoll(text, &end, 10);
        if (end == text || *end != '\0' || v < lo || v > hi) {
            notes += std::string(notes.empty() ? "" : "; ") + "ignoring invalid " + name + "='" + text + "'";
            return fallback;
        }
        return v;
    };
    cfg.staging_bytes = size_t(number("RANMA_EXPERT_P2P_STAGING_MIB", (long long) (cfg.staging_bytes >> 20), 16, 4096)) << 20;
    cfg.chunk_bytes   = size_t(number("RANMA_EXPERT_P2P_CHUNK_KIB", (long long) (cfg.chunk_bytes >> 10), 64, 65536)) << 10;
    cfg.chunk_bytes   = (cfg.chunk_bytes + 65535) & ~size_t(65535);
    cfg.queue_depth   = int(number("RANMA_EXPERT_P2P_QD", cfg.queue_depth, 1, 256));
    cfg.pregate       = number("RANMA_EXPERT_P2P_PREGATE", cfg.pregate ? 1 : 0, 0, 1) != 0;
    cfg.wait_ms       = number("RANMA_EXPERT_P2P_WAIT_MS", cfg.wait_ms, 100, 600000);
    if (cfg.chunk_bytes > cfg.staging_bytes) {
        notes += std::string(notes.empty() ? "" : "; ") + "the chunk is larger than the staging, using the staging size";
        cfg.chunk_bytes = cfg.staging_bytes;
    }
    return cfg;
}

} // namespace ggml_cuda_expert

#if defined(_WIN32)

#include <hip/hip_runtime_api.h>

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <ioringapi.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <atomic>
#include <chrono>

// __uuidof names the COM interfaces without linking dxguid
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#endif
#include <cstdio>
#include <vector>

namespace ggml_cuda_expert {

namespace {

std::string hr_text(HRESULT hr) {
    char b[32];
    snprintf(b, sizeof(b), "0x%08lx", (unsigned long) hr);
    return b;
}

using clock_type = std::chrono::steady_clock;
double ms_since(clock_type::time_point t) { return std::chrono::duration<double, std::milli>(clock_type::now() - t).count(); }

// The IoRing API, resolved at run time.
struct ring_api {
    decltype(&::QueryIoRingCapabilities) caps = nullptr;
    decltype(&::CreateIoRing) create = nullptr;
    decltype(&::SubmitIoRing) submit = nullptr;
    decltype(&::CloseIoRing) close = nullptr;
    decltype(&::PopIoRingCompletion) pop = nullptr;
    decltype(&::BuildIoRingReadFile) read = nullptr;
    decltype(&::BuildIoRingRegisterFileHandles) regfiles = nullptr;
    decltype(&::BuildIoRingRegisterBuffers) regbufs = nullptr;
    IORING_CAPABILITIES cap = {};
};

bool resolve_ring(ring_api & api, std::string & reason) {
    for (const char * name : {"kernelbase.dll", "api-ms-win-core-ioring-l1-1-0.dll"}) {
        HMODULE m = LoadLibraryA(name);
        if (m == nullptr) { continue; }
        ring_api a;
#define RANMA_RESOLVE(field, sym) a.field = (decltype(a.field)) (void *) GetProcAddress(m, sym)
        RANMA_RESOLVE(caps, "QueryIoRingCapabilities"); RANMA_RESOLVE(create, "CreateIoRing");
        RANMA_RESOLVE(submit, "SubmitIoRing"); RANMA_RESOLVE(close, "CloseIoRing"); RANMA_RESOLVE(pop, "PopIoRingCompletion");
        RANMA_RESOLVE(read, "BuildIoRingReadFile"); RANMA_RESOLVE(regfiles, "BuildIoRingRegisterFileHandles");
        RANMA_RESOLVE(regbufs, "BuildIoRingRegisterBuffers");
#undef RANMA_RESOLVE
        if (!a.caps || !a.create || !a.submit || !a.close || !a.pop || !a.read || !a.regfiles || !a.regbufs) { continue; }
        const HRESULT hr = a.caps(&a.cap);
        if (FAILED(hr) || a.cap.MaxVersion == IORING_VERSION_INVALID) {
            reason = "QueryIoRingCapabilities " + hr_text(hr);
            return false;
        }
        if (a.cap.FeatureFlags & IORING_FEATURE_UM_EMULATION) {
            reason = "the IoRing is emulated in user mode";
            return false;
        }
        api = a;
        return true;
    }
    reason = "the IoRing API is not available";
    return false;
}

bool ring_timeout(HRESULT hr) {
    return hr == HRESULT_FROM_WIN32(WAIT_TIMEOUT) || hr == (HRESULT) WAIT_TIMEOUT || hr == HRESULT_FROM_WIN32(ERROR_TIMEOUT);
}

D3D12_RESOURCE_DESC buffer_desc(size_t bytes) {
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = bytes; rd.Height = 1; rd.DepthOrArraySize = 1;
    rd.MipLevels = 1; rd.Format = DXGI_FORMAT_UNKNOWN; rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return rd;
}

inline void cpu_relax() { YieldProcessor(); }

class p2p_win32 final : public p2p_engine {
public:
    ~p2p_win32() override { release(); }

    bool init(int device, const p2p_config & cfg, std::string & reason) {
        cfg_ = cfg;
        device_ = device;
        if (!resolve_ring(api_, reason)) { return false; }
        hipDeviceProp_t prop = {};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void) hipGetLastError();
            reason = "hipGetDeviceProperties failed";
            return false;
        }
        d3d12_dll_ = LoadLibraryA("d3d12.dll");
        dxgi_dll_  = LoadLibraryA("dxgi.dll");
        if (d3d12_dll_ == nullptr || dxgi_dll_ == nullptr) { reason = "d3d12.dll or dxgi.dll is missing"; return false; }
        using create_factory_fn = HRESULT (WINAPI *)(REFIID, void **);
        auto create_factory = (create_factory_fn) (void *) GetProcAddress(dxgi_dll_, "CreateDXGIFactory1");
        auto create_device  = (PFN_D3D12_CREATE_DEVICE) (void *) GetProcAddress(d3d12_dll_, "D3D12CreateDevice");
        if (create_factory == nullptr || create_device == nullptr) { reason = "D3D12CreateDevice or CreateDXGIFactory1 is missing"; return false; }
        IDXGIFactory1 * factory = nullptr;
        HRESULT hr = create_factory(__uuidof(IDXGIFactory1), (void **) &factory);
        if (FAILED(hr)) { reason = "CreateDXGIFactory1 " + hr_text(hr); return false; }
        IDXGIAdapter1 * a = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d = {};
            a->GetDesc1(&d);
            if (adapter_ == nullptr && memcmp(&d.AdapterLuid, prop.luid, sizeof(LUID)) == 0) {
                adapter_ = a;
                char name[128] = {};
                WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
                adapter_name_ = name;
            } else {
                a->Release();
            }
        }
        factory->Release();
        if (adapter_ == nullptr) { reason = "no DXGI adapter has the LUID of the HIP device"; return false; }
        hr = create_device(adapter_, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void **) &dev_);
        if (FAILED(hr)) { reason = "D3D12CreateDevice " + hr_text(hr); return false; }
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16 = {};
        hr = dev_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof(o16));
        if (FAILED(hr) || !o16.GPUUploadHeapSupported) { reason = "GPU upload heaps are not supported (" + hr_text(hr) + ")"; return false; }
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        hr = dev_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), (void **) &queue_);
        if (FAILED(hr)) { reason = "CreateCommandQueue " + hr_text(hr); return false; }
        hr = dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void **) &fence_);
        if (FAILED(hr)) { reason = "CreateFence " + hr_text(hr); return false; }
        hr = dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), (void **) &gate_);
        if (FAILED(hr)) { reason = "CreateFence (gate) " + hr_text(hr); return false; }
        lists_.resize(16);
        for (auto & l : lists_) {
            hr = dev_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, __uuidof(ID3D12CommandAllocator), (void **) &l.allocator);
            if (SUCCEEDED(hr)) {
                hr = dev_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, l.allocator, nullptr, __uuidof(ID3D12GraphicsCommandList), (void **) &l.list);
            }
            if (FAILED(hr)) { reason = "command list " + hr_text(hr); return false; }
            l.list->Close();
        }
        // staging: buffers of at most 1 GiB (a larger mapped allocation failed on the target)
        const size_t chunk = cfg_.chunk_bytes;
        size_t left = std::max(cfg_.staging_bytes, chunk);
        const size_t per_buffer = std::max(chunk, (size_t(1) << 30)/chunk*chunk);
        while (left >= chunk) {
            const size_t bytes = std::min(left, per_buffer)/chunk*chunk;
            staging_buf b;
            if (!upload_buffer(bytes, b.res, b.map, reason)) { return false; }
            b.bytes = bytes;
            staging_.push_back(b);
            for (size_t off = 0; off + chunk <= bytes; off += chunk) {
                slots_.push_back({(int) staging_.size() - 1, off});
            }
            left -= bytes;
            staging_total_ += bytes;
        }
        for (int i = (int) slots_.size() - 1; i >= 0; --i) { free_slots_.push_back(i); }
        slot_tag_.assign(slots_.size(), 0);
        // the source of the tail clears
        if (!upload_buffer(zero_bytes_, zero_res_, zero_map_, reason)) { return false; }
        memset(zero_map_, 0, zero_bytes_);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        staging_total_ += zero_bytes_;
        if (!open_ring(reason)) { return false; }
        return true;
    }

    int add_file(const std::string & path, std::string & reason) override {
        for (size_t i = 0; i < paths_.size(); ++i) {
            if (paths_[i] == path) { return (int) i; }
        }
        HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            reason = "cannot open '" + path + "' for unbuffered reading (" + std::to_string(GetLastError()) + ")";
            return -1;
        }
        paths_.push_back(path);
        handles_.push_back(h);
        files_dirty_ = true;
        return (int) paths_.size() - 1;
    }

    int alloc_region(size_t bytes, void ** device, std::string & reason) override {
        if (bytes == 0 || bytes > max_region_bytes) { reason = "region size " + std::to_string(bytes) + " out of range"; return -1; }
        region r;
        r.bytes = bytes;
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        const D3D12_RESOURCE_DESC rd = buffer_desc(bytes);
        HRESULT hr = dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), (void **) &r.res);
        if (FAILED(hr)) { reason = "CreateCommittedResource(DEFAULT, SHARED) " + hr_text(hr); return -1; }
        hr = dev_->CreateSharedHandle(r.res, nullptr, GENERIC_ALL, nullptr, &r.handle);
        if (FAILED(hr)) { reason = "CreateSharedHandle " + hr_text(hr); r.handle = nullptr; free_region(r); return -1; }
        const D3D12_RESOURCE_ALLOCATION_INFO ai = dev_->GetResourceAllocationInfo(0, 1, &rd);
        if (hipSetDevice(device_) != hipSuccess) { (void) hipGetLastError(); reason = "hipSetDevice failed"; free_region(r); return -1; }
        hipExternalMemoryHandleDesc d = {};
        d.type = hipExternalMemoryHandleTypeD3D12Resource;
        d.handle.win32.handle = r.handle;
        d.size = ai.SizeInBytes;
        d.flags = hipExternalMemoryDedicated;
        hipError_t e = hipImportExternalMemory(&r.memory, &d);
        if (e != hipSuccess) {
            (void) hipGetLastError(); r.memory = nullptr;
            reason = std::string("hipImportExternalMemory ") + hipGetErrorName(e);
            free_region(r);
            return -1;
        }
        hipExternalMemoryBufferDesc bd = {};
        bd.offset = 0; bd.size = bytes; bd.flags = 0;
        e = hipExternalMemoryGetMappedBuffer(&r.device, r.memory, &bd);
        if (e != hipSuccess || r.device == nullptr) {
            (void) hipGetLastError(); r.device = nullptr;
            reason = std::string("hipExternalMemoryGetMappedBuffer ") + hipGetErrorName(e);
            free_region(r);
            return -1;
        }
        regions_.push_back(r);
        region_total_ += bytes;
        *device = r.device;
        return (int) regions_.size() - 1;
    }

    void * region_device(int region) const override {
        return region >= 0 && region < (int) regions_.size() ? regions_[(size_t) region].device : nullptr;
    }

    bool begin(const p2p_read * reads, size_t n, std::string & reason) override {
        if (running_) { reason = "a P2P run is already in progress"; return false; }
        if (failed_) { reason = "the P2P engine failed earlier: " + fail_reason_; return false; }
        if (files_dirty_ && !register_files(reason)) { return fail(reason); }
        t0_ = clock_type::now();
        reqs_.assign(reads, reads + n);
        pieces_.clear();
        req_left_.assign(n, 0);
        next_piece_ = 0; inflight_ = 0; prefix_ = 0; copied_ = 0;
        ++run_id_;
        const uint64_t chunk = cfg_.chunk_bytes, sector = 4096;
        for (size_t i = 0; i < n; ++i) {
            const p2p_read & r = reqs_[i];
            if (r.file < 0 || r.file >= (int) handles_.size() || r.region < 0 || r.region >= (int) regions_.size() ||
                    r.bytes == 0 || r.dst + r.bytes + r.zero_tail > regions_[(size_t) r.region].bytes) {
                reason = "a P2P read out of range (file " + std::to_string(r.file) + ", region " + std::to_string(r.region) +
                    ", dst " + std::to_string(r.dst) + ", bytes " + std::to_string(r.bytes) + ")";
                return fail(reason);
            }
            const uint64_t end = r.offset + r.bytes;
            const uint64_t aligned_end = (end + sector - 1) & ~(sector - 1);
            for (uint64_t fs = r.offset & ~(sector - 1); fs < aligned_end; fs += chunk) {
                const uint64_t fe = std::min(fs + chunk, aligned_end);
                const uint64_t ps = std::max(fs, r.offset), pe = std::min(fe, end);
                piece p;
                p.req = (int) i; p.file = r.file; p.foff = fs; p.rbytes = uint32_t(fe - fs);
                p.need = uint32_t(pe - fs);
                p.src_off = uint32_t(ps - fs); p.region = r.region; p.dst = r.dst + size_t(ps - r.offset);
                p.cbytes = uint32_t(pe - ps);
                pieces_.push_back(p);
                ++req_left_[i];
            }
            pieces_.back().zero_tail = r.zero_tail;
            stats_.bytes += r.bytes;
            stats_.zero_bytes += r.zero_tail;
        }
        stats_.reads += n;
        stats_.pieces += pieces_.size();
        ++stats_.runs;
        running_ = true;
        reads_done_ = false;
        gated_ = false;
        if (!issue(reason)) { return fail(reason); }
        if (cfg_.pregate && next_piece_ == pieces_.size() && !pieces_.empty()) {
            // every read is in flight: its copies go to the queue now, behind a wait on the gate fence
            // that the CPU signals once the last read completed (no ExecuteCommandLists after the reads)
            if (!open_list(reason)) { return fail(reason); }
            for (piece & p : pieces_) {
                if (!record_copy(p, reason)) { return fail(reason); }
            }
            const UINT64 gate = ++gate_value_;
            if (FAILED(queue_->Wait(gate_, gate))) { return fail(reason = "copy queue Wait on the gate failed"); }
            const UINT64 tag = submit_list(reason);
            if (tag == 0) { signal_gate(); return fail(reason); }
            for (piece & p : pieces_) { p.tag = tag; slot_tag_[(size_t) p.slot] = tag; }
            gated_ = true;
            ++stats_.gated_runs;
        }
        return true;
    }

    bool finish(std::string & reason, const std::function<bool(size_t)> & progress) override {
        if (!running_) { reason = "no P2P run in progress"; return false; }
        const auto start = clock_type::now();
        size_t reported = 0;
        auto report = [&]() {
            if (progress && prefix_ > reported) {
                reported = prefix_;
                return progress(prefix_);
            }
            return true;
        };
        while (prefix_ < reqs_.size()) {
            bool moved = false;
            const UINT64 done = fence_->GetCompletedValue();
            if (done == UINT64_MAX) { return fail(reason = "the D3D12 device was removed"); }
            // copies the queue has finished: their reads are complete and their staging is free
            for (size_t i = copied_; i < next_piece_; ++i) {
                piece & p = pieces_[i];
                if (p.state == piece::copying && p.tag <= done) { p.state = piece::done; --req_left_[(size_t) p.req]; moved = true; }
            }
            while (copied_ < pieces_.size() && pieces_[copied_].state == piece::done) { ++copied_; }
            while (prefix_ < reqs_.size() && req_left_[prefix_] == 0) { ++prefix_; moved = true; }
            if (!report()) { return fail(reason = "the P2P progress callback refused"); }
            if (prefix_ >= reqs_.size()) { break; }
            // new reads while there is staging and queue room
            const size_t before = next_piece_;
            if (!issue(reason)) { return fail(reason); }
            moved = moved || next_piece_ != before;
            // completed reads become copies, one command list per sweep
            int recorded = 0;
            IORING_CQE cqe = {};
            while (pop_cqe(cqe)) {
                const UINT_PTR ud = cqe.UserData;
                if ((ud >> 40) != (run_id_ & 0xFFFFFF)) { continue; }   // a completion of an earlier run
                const size_t index = size_t(ud & ((UINT_PTR(1) << 40) - 1));
                if (index >= pieces_.size() || pieces_[index].state != piece::reading) { continue; }
                piece & p = pieces_[index];
                --inflight_;
                if (FAILED(cqe.ResultCode) || cqe.Information < p.need) {
                    reason = "P2P read of file " + std::to_string(p.file) + " at " + std::to_string(p.foff) + " failed: " +
                        hr_text(cqe.ResultCode) + ", " + std::to_string((unsigned long long) cqe.Information) + " of " +
                        std::to_string(p.rbytes) + " bytes";
                    if (gated_) { signal_gate(); }
                    return fail(reason);
                }
                if (gated_) {
                    p.state = piece::copying;   // its copy is queued behind the gate
                    moved = true;
                    continue;
                }
                if (!record_copy(p, reason)) { return fail(reason); }
                p.state = piece::recorded;
                ++recorded;
            }
            if (!reads_done_ && inflight_ == 0 && next_piece_ == pieces_.size()) {
                reads_done_ = true;
                reads_done_at_ = clock_type::now();
                if (gated_) { signal_gate(); }
            }
            if (recorded > 0) {
                const UINT64 tag = submit_list(reason);
                if (tag == 0) { return fail(reason); }
                for (size_t i = copied_; i < next_piece_; ++i) {
                    piece & p = pieces_[i];
                    if (p.state == piece::recorded) { p.state = piece::copying; p.tag = tag; slot_tag_[(size_t) p.slot] = tag; }
                }
                moved = true;
            }
            if (!moved) {
                const auto w = clock_type::now();
                if (inflight_ > 0) {
                    // spin briefly on the completion queue, then sleep in SubmitIoRing for at most 1 ms
                    bool got = false;
                    for (int i = 0; i < 4000 && !got; ++i) { cpu_relax(); got = ring_has_completion(); }
                    if (!got) {
                        UINT32 submitted = 0;
                        const HRESULT hr = api_.submit(ring_, 1, 1, &submitted);
                        if (FAILED(hr) && !ring_timeout(hr)) { return fail(reason = "SubmitIoRing (wait) " + hr_text(hr)); }
                    }
                    stats_.io_wait_ms += ms_since(w);
                } else {
                    for (int i = 0; i < 256; ++i) { cpu_relax(); }
                    stats_.fence_wait_ms += ms_since(w);
                }
                if (ms_since(start) > double(cfg_.wait_ms)) {
                    char state[256];
                    snprintf(state, sizeof(state), " (reads in flight %d, issued %zu of %zu pieces, copied %zu, requests done %zu of %zu, "
                        "copy fence %llu of %llu, gate %s)", inflight_, next_piece_, pieces_.size(), copied_, prefix_, reqs_.size(),
                        (unsigned long long) fence_->GetCompletedValue(), (unsigned long long) fence_value_, gated_ ? "on" : "off");
                    return fail(reason = "a P2P run did not complete within " + std::to_string(cfg_.wait_ms) + " ms" + state);
                }
            }
        }
        running_ = false;
        stats_.run_ms += ms_since(t0_);
        if (reads_done_) {
            stats_.tail_ms += ms_since(reads_done_at_);
            ++stats_.tail_runs;
        }
        return true;
    }

    bool read_back(int region, size_t offset, void * dst, size_t bytes, std::string & reason) override {
        const span one = {region, offset, dst, bytes};
        return read_back_many(&one, 1, reason);
    }

    bool read_back_many(const span * spans, size_t n, std::string & reason) override {
        if (failed_) { reason = "the P2P engine failed earlier: " + fail_reason_; return false; }
        size_t total = 0;
        for (size_t i = 0; i < n; ++i) {
            const span & sp = spans[i];
            if (sp.region < 0 || sp.region >= (int) regions_.size() || sp.offset + sp.bytes > regions_[(size_t) sp.region].bytes) {
                reason = "a read back out of range";
                return false;
            }
            total += sp.bytes;
        }
        // one buffer of up to 64 MiB, grown on demand
        const size_t want = std::min<size_t>(std::max<size_t>(total, size_t(4) << 20), size_t(64) << 20);
        if (readback_ == nullptr || readback_bytes_ < want) {
            if (readback_ != nullptr) { readback_->Unmap(0, nullptr); readback_->Release(); readback_ = nullptr; readback_map_ = nullptr; }
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_READBACK;
            const D3D12_RESOURCE_DESC rd = buffer_desc(want);
            HRESULT hr = dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                __uuidof(ID3D12Resource), (void **) &readback_);
            if (FAILED(hr)) { readback_ = nullptr; reason = "CreateCommittedResource(READBACK) " + hr_text(hr); return false; }
            void * p = nullptr;
            hr = readback_->Map(0, nullptr, &p);
            if (FAILED(hr) || p == nullptr) { reason = "Map(READBACK) " + hr_text(hr); return false; }
            readback_map_ = static_cast<unsigned char *>(p);
            readback_bytes_ = want;
        }
        // a list of a run in progress is never left open between sweeps; these are submitted at once
        if (open_ >= 0) { reason = "a read back while a command list is open"; return false; }
        struct piece_out { char * dst; size_t at, bytes; };
        std::vector<piece_out> outs;
        size_t i = 0, done_in_span = 0;
        while (i < n) {
            if (!open_list(reason)) { return false; }
            ID3D12GraphicsCommandList * cl = lists_[(size_t) open_].list;
            outs.clear();
            size_t used = 0;
            while (i < n && used < readback_bytes_) {
                const span & sp = spans[i];
                const size_t m = std::min(sp.bytes - done_in_span, readback_bytes_ - used);
                cl->CopyBufferRegion(readback_, used, regions_[(size_t) sp.region].res, sp.offset + done_in_span, m);
                outs.push_back({static_cast<char *>(sp.dst) + done_in_span, used, m});
                used += m;
                done_in_span += m;
                if (done_in_span == sp.bytes) { ++i; done_in_span = 0; }
            }
            const UINT64 tag = submit_list(reason);
            if (tag == 0 || !wait_fence(tag, reason)) { return false; }
            for (const piece_out & o : outs) { memcpy(o.dst, readback_map_ + o.at, o.bytes); }
        }
        return true;
    }

    bool write(int region, size_t offset, const void * src, size_t bytes, size_t zero_tail, std::string & reason) override {
        if (running_) { reason = "a P2P write during a run"; return false; }
        if (failed_) { reason = "the P2P engine failed earlier: " + fail_reason_; return false; }
        if (region < 0 || region >= (int) regions_.size() || offset + bytes + zero_tail > regions_[(size_t) region].bytes) {
            reason = "a P2P write out of range";
            return false;
        }
        ID3D12Resource * dst = regions_[(size_t) region].res;
        std::string why;
        if (!issue(why)) { reason = why; return false; }   // frees the staging slots whose copies are done
        size_t done = 0;
        while (done < bytes || zero_tail > 0) {
            if (!open_list(reason)) { return false; }
            ID3D12GraphicsCommandList * cl = lists_[(size_t) open_].list;
            std::vector<int> used;
            while (done < bytes && !free_slots_.empty()) {
                const int slot = free_slots_.back();
                free_slots_.pop_back();
                used.push_back(slot);
                const stage_slot & st = slots_[(size_t) slot];
                const size_t n = std::min(cfg_.chunk_bytes, bytes - done);
                memcpy(staging_[(size_t) st.buffer].map + st.offset, static_cast<const char *>(src) + done, n);
                cl->CopyBufferRegion(dst, offset + done, staging_[(size_t) st.buffer].res, st.offset, n);
                done += n;
            }
            if (done >= bytes) {
                for (size_t z = 0; z < zero_tail;) {
                    const size_t n = std::min(zero_bytes_, zero_tail - z);
                    cl->CopyBufferRegion(dst, offset + bytes + z, zero_res_, 0, n);
                    z += n;
                }
                zero_tail = 0;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);   // the write-combined staging stores are out
            const UINT64 tag = submit_list(reason);
            if (tag == 0 || !wait_fence(tag, reason)) { return false; }
            for (int slot : used) { free_slots_.push_back(slot); }
            if (used.empty() && done < bytes) { reason = "no free staging slot for a P2P write"; return false; }
        }
        return true;
    }

    const p2p_stats & stats() const override { return stats_; }

    std::string description() const override {
        char b[256];
        snprintf(b, sizeof(b), "adapter '%s', staging %zu MiB in %zu buffers (GPU_UPLOAD), chunk %zu KiB, queue depth %d, IoRing v%d, copies %s",
            adapter_name_.c_str(), (staging_total_ - zero_bytes_) >> 20, staging_.size(), cfg_.chunk_bytes >> 10,
            cfg_.queue_depth, (int) api_.cap.MaxVersion, cfg_.pregate ? "queued before the reads behind a CPU-signalled gate" :
            "submitted as the reads complete");
        return b;
    }
    size_t staging_bytes() const override { return staging_total_; }
    size_t region_bytes() const override { return region_total_; }

private:
    struct staging_buf { ID3D12Resource * res = nullptr; unsigned char * map = nullptr; size_t bytes = 0; };
    struct stage_slot { int buffer = 0; size_t offset = 0; };
    struct region {
        ID3D12Resource * res = nullptr;
        HANDLE handle = nullptr;
        hipExternalMemory_t memory = nullptr;
        void * device = nullptr;
        size_t bytes = 0;
    };
    struct list_slot { ID3D12CommandAllocator * allocator = nullptr; ID3D12GraphicsCommandList * list = nullptr; UINT64 tag = 0; };
    struct piece {
        enum state_t : uint8_t { pending, reading, recorded, copying, done };
        int req = 0, file = 0, region = 0, slot = -1;
        uint64_t foff = 0;
        uint32_t rbytes = 0, need = 0, src_off = 0, cbytes = 0;
        size_t dst = 0, zero_tail = 0;
        UINT64 tag = 0;
        state_t state = pending;
    };

    void signal_gate() {
        if (gate_ != nullptr) { (void) gate_->Signal(gate_value_); }
    }

    bool fail(const std::string & why) {
        failed_ = true;
        fail_reason_ = why;
        running_ = false;
        return false;
    }

    bool upload_buffer(size_t bytes, ID3D12Resource *& res, unsigned char *& map, std::string & reason) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
        const D3D12_RESOURCE_DESC rd = buffer_desc(bytes);
        HRESULT hr = dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), (void **) &res);
        if (FAILED(hr)) { res = nullptr; reason = "CreateCommittedResource(GPU_UPLOAD, " + std::to_string(bytes >> 20) + " MiB) " + hr_text(hr); return false; }
        void * p = nullptr;
        hr = res->Map(0, nullptr, &p);
        if (FAILED(hr) || p == nullptr) { res->Release(); res = nullptr; reason = "Map(GPU_UPLOAD) " + hr_text(hr); return false; }
        map = static_cast<unsigned char *>(p);
        return true;
    }

    HRESULT ring_wait(UINT_PTR ud) {
        const auto t = clock_type::now();
        UINT32 n = 0;
        HRESULT hr = api_.submit(ring_, 0, 0, &n);
        if (FAILED(hr)) { return hr; }
        for (;;) {
            IORING_CQE cqe = {};
            if (api_.pop(ring_, &cqe) == S_OK) {
                if (cqe.UserData == ud) { return cqe.ResultCode; }
                continue;
            }
            if (ms_since(t) > 30000.0) { return HRESULT_FROM_WIN32(ERROR_TIMEOUT); }
            hr = api_.submit(ring_, 1, 100, &n);
            if (FAILED(hr) && !ring_timeout(hr)) { return hr; }
        }
    }

    // The ring with the staging buffers registered, one entry per buffer (about 0.5 s per GiB).
    bool open_ring(std::string & reason) {
        const UINT32 want = (UINT32) std::max(64, 2*cfg_.queue_depth + 8);
        UINT32 sq = 32;
        while (sq < want) { sq <<= 1; }
        if (api_.cap.MaxSubmissionQueueSize && sq > api_.cap.MaxSubmissionQueueSize) { sq = api_.cap.MaxSubmissionQueueSize; }
        UINT32 cq = 2*sq;
        if (api_.cap.MaxCompletionQueueSize && cq > api_.cap.MaxCompletionQueueSize) { cq = api_.cap.MaxCompletionQueueSize; }
        IORING_CREATE_FLAGS fl = {IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_CREATE_ADVISORY_FLAGS_NONE};
        HRESULT hr = api_.create(api_.cap.MaxVersion, fl, sq, cq, &ring_);
        if (FAILED(hr)) { ring_ = nullptr; reason = "CreateIoRing " + hr_text(hr); return false; }
        std::vector<IORING_BUFFER_INFO> entries;
        for (const staging_buf & b : staging_) {
            IORING_BUFFER_INFO e;
            e.Address = b.map;
            e.Length = (UINT32) b.bytes;
            entries.push_back(e);
        }
        hr = api_.regbufs(ring_, (UINT32) entries.size(), entries.data(), (UINT_PTR) 0xB0FFull);
        if (SUCCEEDED(hr)) { hr = ring_wait((UINT_PTR) 0xB0FFull); }
        if (FAILED(hr)) { reason = "BuildIoRingRegisterBuffers " + hr_text(hr); return false; }
        return true;
    }

    // A ring's file table is registered once; a later file reopens the ring.
    bool register_files(std::string & reason) {
        if (files_registered_) {
            api_.close(ring_);
            ring_ = nullptr;
            files_registered_ = false;
            if (!open_ring(reason)) { return false; }
        }
        HRESULT hr = api_.regfiles(ring_, (UINT32) handles_.size(), handles_.data(), (UINT_PTR) 0xF11Eull);
        if (SUCCEEDED(hr)) { hr = ring_wait((UINT_PTR) 0xF11Eull); }
        if (FAILED(hr)) { reason = "BuildIoRingRegisterFileHandles " + hr_text(hr); return false; }
        files_registered_ = true;
        files_dirty_ = false;
        return true;
    }

    // A completion taken while waiting is parked until the next sweep.
    bool ring_has_completion() {
        IORING_CQE cqe = {};
        if (api_.pop(ring_, &cqe) != S_OK) { return false; }
        parked_.push_back(cqe);
        return true;
    }
    bool pop_cqe(IORING_CQE & cqe) {
        if (!parked_.empty()) {
            cqe = parked_.back();
            parked_.pop_back();
            return true;
        }
        return api_.pop(ring_, &cqe) == S_OK;
    }

    // Reads go out while a staging slot is free (its last copy done) and the queue has room.
    bool issue(std::string & reason) {
        const UINT64 done = fence_->GetCompletedValue();
        if (done == UINT64_MAX) { reason = "the D3D12 device was removed"; return false; }
        // free the slots whose copies are done
        for (size_t i = 0; i < busy_slots_.size();) {
            const int s = busy_slots_[i];
            if (slot_tag_[(size_t) s] != 0 && slot_tag_[(size_t) s] <= done) {
                slot_tag_[(size_t) s] = 0;
                free_slots_.push_back(s);
                busy_slots_[i] = busy_slots_.back();
                busy_slots_.pop_back();
            } else {
                ++i;
            }
        }
        int issued = 0;
        while (next_piece_ < pieces_.size() && inflight_ < cfg_.queue_depth && !free_slots_.empty()) {
            piece & p = pieces_[next_piece_];
            p.slot = free_slots_.back();
            free_slots_.pop_back();
            busy_slots_.push_back(p.slot);
            slot_tag_[(size_t) p.slot] = 0;
            const stage_slot & s = slots_[(size_t) p.slot];
            const UINT_PTR ud = (UINT_PTR(run_id_ & 0xFFFFFF) << 40) | UINT_PTR(next_piece_);
            const HRESULT hr = api_.read(ring_, IoRingHandleRefFromIndex((UINT32) p.file),
                IoRingBufferRefFromIndexAndOffset((UINT32) s.buffer, (UINT32) s.offset), p.rbytes, p.foff, ud, IOSQE_FLAGS_NONE);
            if (FAILED(hr)) { reason = "BuildIoRingReadFile " + hr_text(hr); return false; }
            p.state = piece::reading;
            ++inflight_;
            ++next_piece_;
            ++issued;
        }
        if (issued > 0) {
            UINT32 n = 0;
            const HRESULT hr = api_.submit(ring_, 0, 0, &n);
            if (FAILED(hr)) { reason = "SubmitIoRing " + hr_text(hr); return false; }
        }
        return true;
    }

    // Opens a command list when none is open; an allocator is reused once its last list is done.
    bool open_list(std::string & reason) {
        if (open_ >= 0) { return true; }
        const auto t = clock_type::now();
        for (;;) {
            const UINT64 done = fence_->GetCompletedValue();
            if (done == UINT64_MAX) { reason = "the D3D12 device was removed"; return false; }
            for (int i = 0; i < (int) lists_.size(); ++i) {
                const int k = (next_list_ + i) % (int) lists_.size();
                if (lists_[(size_t) k].tag <= done) {
                    list_slot & l = lists_[(size_t) k];
                    HRESULT hr = l.allocator->Reset();
                    if (SUCCEEDED(hr)) { hr = l.list->Reset(l.allocator, nullptr); }
                    if (FAILED(hr)) { reason = "command list reset " + hr_text(hr); return false; }
                    open_ = k;
                    next_list_ = (k + 1) % (int) lists_.size();
                    return true;
                }
            }
            if (ms_since(t) > double(cfg_.wait_ms)) { reason = "no command allocator came free"; return false; }
            cpu_relax();
        }
    }

    bool record_copy(const piece & p, std::string & reason) {
        if (!open_list(reason)) { return false; }
        ID3D12GraphicsCommandList * cl = lists_[(size_t) open_].list;
        const stage_slot & s = slots_[(size_t) p.slot];
        ID3D12Resource * dst = regions_[(size_t) p.region].res;
        cl->CopyBufferRegion(dst, p.dst, staging_[(size_t) s.buffer].res, s.offset + p.src_off, p.cbytes);
        for (size_t z = 0; z < p.zero_tail;) {
            const size_t n = std::min(zero_bytes_, p.zero_tail - z);
            cl->CopyBufferRegion(dst, p.dst + p.cbytes + z, zero_res_, 0, n);
            z += n;
        }
        return true;
    }

    bool wait_fence(UINT64 tag, std::string & reason) {
        const auto t = clock_type::now();
        for (;;) {
            const UINT64 done = fence_->GetCompletedValue();
            if (done == UINT64_MAX) { reason = "the D3D12 device was removed"; return false; }
            if (done >= tag) { return true; }
            if (ms_since(t) > double(cfg_.wait_ms)) {
                reason = "the copy queue did not signal in time (fence " + std::to_string((unsigned long long) done) + " of " +
                    std::to_string((unsigned long long) tag) + ")";
                return false;
            }
            cpu_relax();
        }
    }

    UINT64 submit_list(std::string & reason) {
        list_slot & l = lists_[(size_t) open_];
        HRESULT hr = l.list->Close();
        if (FAILED(hr)) { reason = "command list close " + hr_text(hr); return 0; }
        ID3D12CommandList * cls[] = {l.list};
        queue_->ExecuteCommandLists(1, cls);
        const UINT64 tag = ++fence_value_;
        hr = queue_->Signal(fence_, tag);
        if (FAILED(hr)) { reason = "copy queue Signal " + hr_text(hr); return 0; }
        l.tag = tag;
        open_ = -1;
        ++stats_.lists;
        return tag;
    }

    void free_region(region & r) {
        if (r.device) { (void) hipFree(r.device); }
        if (r.memory) { (void) hipDestroyExternalMemory(r.memory); }
        (void) hipGetLastError();
        if (r.handle) { CloseHandle(r.handle); }
        if (r.res) { r.res->Release(); }
        r = region();
    }

    void release() {
        if (fence_ != nullptr && fence_value_ > 0) {
            const auto t = clock_type::now();
            while (fence_->GetCompletedValue() < fence_value_ && ms_since(t) < 10000.0) { Sleep(0); }
        }
        if (!regions_.empty()) {
            (void) hipSetDevice(device_);
            (void) hipDeviceSynchronize();
            for (region & r : regions_) { free_region(r); }
            regions_.clear();
        }
        if (ring_ != nullptr) { api_.close(ring_); ring_ = nullptr; }
        for (HANDLE h : handles_) { CloseHandle(h); }
        handles_.clear();
        for (staging_buf & b : staging_) { if (b.res) { b.res->Unmap(0, nullptr); b.res->Release(); } }
        staging_.clear();
        if (zero_res_) { zero_res_->Unmap(0, nullptr); zero_res_->Release(); zero_res_ = nullptr; }
        if (readback_) { readback_->Unmap(0, nullptr); readback_->Release(); readback_ = nullptr; }
        for (list_slot & l : lists_) { if (l.list) { l.list->Release(); } if (l.allocator) { l.allocator->Release(); } }
        lists_.clear();
        if (fence_) { fence_->Release(); fence_ = nullptr; }
        if (gate_) { gate_->Release(); gate_ = nullptr; }
        if (queue_) { queue_->Release(); queue_ = nullptr; }
        if (dev_) { dev_->Release(); dev_ = nullptr; }
        if (adapter_) { adapter_->Release(); adapter_ = nullptr; }
        // the DLLs stay loaded: COM objects of other components may still use them
    }

    p2p_config cfg_;
    int device_ = 0;
    ring_api api_;
    HMODULE d3d12_dll_ = nullptr, dxgi_dll_ = nullptr;
    IDXGIAdapter1 * adapter_ = nullptr;
    std::string adapter_name_;
    ID3D12Device * dev_ = nullptr;
    ID3D12CommandQueue * queue_ = nullptr;
    ID3D12Fence * fence_ = nullptr;
    ID3D12Fence * gate_ = nullptr;      // pre-gated runs: the copy queue waits on it, the CPU signals it
    UINT64 gate_value_ = 0;
    UINT64 fence_value_ = 0;
    std::vector<list_slot> lists_;
    int open_ = -1, next_list_ = 0;
    std::vector<staging_buf> staging_;
    std::vector<stage_slot> slots_;
    std::vector<int> free_slots_, busy_slots_;
    std::vector<UINT64> slot_tag_;
    ID3D12Resource * zero_res_ = nullptr;
    unsigned char * zero_map_ = nullptr;
    const size_t zero_bytes_ = size_t(64) << 10;
    size_t staging_total_ = 0, region_total_ = 0;
    std::vector<region> regions_;
    std::vector<std::string> paths_;
    std::vector<HANDLE> handles_;
    bool files_dirty_ = false, files_registered_ = false;
    HIORING ring_ = nullptr;
    std::vector<IORING_CQE> parked_;
    ID3D12Resource * readback_ = nullptr;   // read_back: a READBACK heap buffer, created on first use
    unsigned char * readback_map_ = nullptr;
    size_t readback_bytes_ = 0;
    // the current run
    std::vector<p2p_read> reqs_;
    std::vector<piece> pieces_;
    std::vector<int> req_left_;
    size_t next_piece_ = 0, prefix_ = 0, copied_ = 0;
    int inflight_ = 0;
    uint64_t run_id_ = 0;
    bool running_ = false, failed_ = false;
    bool gated_ = false, reads_done_ = false;
    clock_type::time_point reads_done_at_;
    std::string fail_reason_;
    clock_type::time_point t0_;
    p2p_stats stats_;
};

} // namespace

std::unique_ptr<p2p_engine> p2p_engine::create(int device, const p2p_config & cfg, std::string & reason) {
    std::unique_ptr<p2p_win32> e(new p2p_win32());
    if (!e->init(device, cfg, reason)) {
        return nullptr;
    }
    return e;
}

} // namespace ggml_cuda_expert

#else

namespace ggml_cuda_expert {

std::unique_ptr<p2p_engine> p2p_engine::create(int, const p2p_config &, std::string & reason) {
    reason = "peer-to-peer reads are implemented for Windows only";
    return nullptr;
}

} // namespace ggml_cuda_expert

#endif
