#pragma once

// One worker thread keeps the process' GPU residency alive. While the server is idle, and while the model is torn
// down, it records one event on every device that holds model buffers, once per interval. Windows' video memory
// manager evicts a process' whole VRAM residency after ~10 s without a submission, and a single event per interval
// is enough to reset that policy. The worker owns no model data: one extra backend and one event per device.

#include "ggml.h"
#include "ggml-backend.h"

#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct server_gpu_heartbeat {
    // returns true when a worker is pulsing at least one device
    bool init(const std::vector<ggml_backend_dev_t> & devices, float interval_seconds) {
        release();

        if (!(interval_seconds > 0.0f) || devices.empty()) {
            return false;
        }

        devs.clear();
        for (ggml_backend_dev_t dev : devices) {
            if (dev == nullptr) {
                continue;
            }
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            if (std::find(devs.begin(), devs.end(), dev) == devs.end()) {
                devs.push_back(dev);
            }
        }

        if (devs.empty()) {
            return false;
        }

        interval_ms = std::max<int64_t>(1, (int64_t) std::ceil((double) interval_seconds * 1000.0));
        state       = phase::busy;
        ready = available = stopping = executing = pending = immediate = false;
        n_pulses = n_teardown_pulses = 0;

        try {
            worker = std::thread([this] { run(); });
        } catch (const std::exception & e) {
            SRV_WRN("gpu heartbeat: cannot start worker: %s\n", e.what());
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this] { return ready; });
        const bool ok = available;
        lock.unlock();

        if (!ok) {
            release();
        }

        return ok;
    }

    // going busy waits for an in-flight pulse, so the caller can touch model GPU state afterwards
    void set_idle(bool idle) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!available || state == phase::shutdown) {
            return;
        }

        const phase wanted = idle ? phase::idle : phase::busy;
        if (state != wanted) {
            state   = wanted;
            next_ms = ggml_time_ms() + interval_ms;
            condition.notify_all();
        }

        if (!idle) {
            condition.wait(lock, [this] { return !executing && !pending; });
        }
    }

    // pulse now, then keep pulsing while the model buffers are freed
    void begin_shutdown() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!available || state == phase::shutdown) {
            return;
        }

        const uint64_t previous = n_teardown_pulses;

        state     = phase::shutdown;
        immediate = true;
        condition.notify_all();
        condition.wait(lock, [this, previous] { return n_teardown_pulses > previous || !available; });
    }

    void release() {
        if (!worker.joinable()) {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            condition.notify_all();
        }
        worker.join();
        devs.clear();
    }

    ~server_gpu_heartbeat() { release(); }

private:
    enum class phase { busy, idle, shutdown };

    struct device_state {
        ggml_backend_dev_t   dev     = nullptr;
        ggml_backend_t       backend = nullptr;
        ggml_backend_event_t event   = nullptr;
    };

    std::mutex              mutex;
    std::condition_variable condition;
    std::thread             worker;

    std::vector<ggml_backend_dev_t> devs;

    phase state     = phase::busy;
    bool  ready     = false;
    bool  available = false;
    bool  stopping  = false;
    bool  executing = false;
    bool  pending   = false;
    bool  immediate = false;

    int64_t  interval_ms        = 0;
    int64_t  next_ms            = 0;
    uint64_t n_pulses           = 0;
    uint64_t n_teardown_pulses  = 0;

    void run() {
        std::vector<device_state> active;
        std::string names;

        for (ggml_backend_dev_t dev : devs) {
            device_state ds;
            ds.dev     = dev;
            ds.backend = ggml_backend_dev_init(dev, nullptr);
            if (ds.backend == nullptr) {
                SRV_WRN("gpu heartbeat: skipping device %s, backend initialization failed\n", ggml_backend_dev_name(dev));
                continue;
            }
            ds.event = ggml_backend_event_new(dev);
            if (ds.event == nullptr) {
                SRV_WRN("gpu heartbeat: skipping device %s, events are not supported\n", ggml_backend_dev_name(dev));
                ggml_backend_free(ds.backend);
                continue;
            }
            names += names.empty() ? "" : ", ";
            names += ggml_backend_dev_name(dev);
            active.push_back(ds);
        }

        const bool ok = !active.empty();
        if (ok) {
            SRV_INF("gpu heartbeat: interval=%.3f s, devices=%s\n", interval_ms / 1000.0, names.c_str());
        } else {
            SRV_WRN("%s", "gpu heartbeat: no usable device, disabled\n");
        }

        uint64_t pulses = 0, teardown_pulses = 0;
        int64_t  max_us = 0;
        bool     awaiting = false, awaiting_teardown = false;

        // at most one pulse is outstanding: the previous one is completed before the next is recorded
        auto complete = [&]() {
            if (!awaiting) {
                return;
            }
            const int64_t t_start_us = ggml_time_us();
            for (auto & ds : active) {
                ggml_backend_event_synchronize(ds.event);
                // also drain the backend: Metal keeps a command buffer per recorded event until the backend is synchronized
                ggml_backend_synchronize(ds.backend);
            }
            const int64_t elapsed_us = ggml_time_us() - t_start_us;
            awaiting = false;
            ++pulses;
            if (awaiting_teardown) {
                ++teardown_pulses;
            }
            max_us = std::max(max_us, elapsed_us);
            SRV_DBG("gpu heartbeat: pulse %" PRIu64 " done, phase=%s, wait=%" PRId64 " us\n",
                    pulses, awaiting_teardown ? "teardown" : "idle", elapsed_us);
        };

        {
            std::lock_guard<std::mutex> lock(mutex);
            ready     = true;
            available = ok;
            condition.notify_all();
        }

        std::unique_lock<std::mutex> lock(mutex);
        while (ok && !stopping) {
            if (state == phase::busy) {
                if (awaiting) {
                    executing = true;
                    lock.unlock();
                    complete();
                    lock.lock();
                    pending = executing = false;
                    n_pulses = pulses;
                    n_teardown_pulses = teardown_pulses;
                    condition.notify_all();
                }
                condition.wait(lock, [this] { return stopping || state != phase::busy; });
                continue;
            }

            const int64_t delay_ms = next_ms - ggml_time_ms();
            if (!immediate && delay_ms > 0) {
                condition.wait_for(lock, std::chrono::milliseconds(delay_ms));
                continue;
            }

            const bool teardown = state == phase::shutdown;
            const bool wait_now = immediate;
            immediate = false;
            executing = true;
            lock.unlock();

            complete();
            for (auto & ds : active) {
                ggml_backend_event_record(ds.event, ds.backend);
            }
            awaiting          = true;
            awaiting_teardown = teardown;
            if (wait_now) {
                complete();
            }

            lock.lock();
            pending   = awaiting;
            executing = false;
            n_pulses          = pulses;
            n_teardown_pulses = teardown_pulses;
            next_ms = ggml_time_ms() + interval_ms;
            condition.notify_all();
        }
        available = false;
        condition.notify_all();
        lock.unlock();

        complete();

        lock.lock();
        pending = executing = false;
        n_pulses = pulses;
        n_teardown_pulses = teardown_pulses;
        condition.notify_all();
        lock.unlock();

        if (ok) {
            SRV_INF("gpu heartbeat: released, pulses=%" PRIu64 " (teardown %" PRIu64 "), max wait=%" PRId64 " us\n",
                    pulses, teardown_pulses, max_us);
        }

        for (auto & ds : active) {
            ggml_backend_event_free(ds.event);
            ggml_backend_free(ds.backend);
        }
    }
};
