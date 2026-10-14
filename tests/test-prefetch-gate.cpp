#include "llama-prefetch.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// The probe links the internal helper directly, without a model or llama DLL.
void llama_log_internal(ggml_log_level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}

struct mapping {
    void * data = nullptr;
    size_t size;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE map = nullptr;
#else
    int file = -1;
#endif
    mapping(const char * path, size_t size) : size(size) {
#ifdef _WIN32
        file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (map) {
                data = MapViewOfFile(map, FILE_MAP_READ, 0, 0, size);
            }
        }
#else
        file = open(path, O_RDONLY);
        if (file >= 0) {
            data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file, 0);
            if (data == MAP_FAILED) {
                data = nullptr;
            }
        }
#endif
    }
    ~mapping() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); }
        if (map) { CloseHandle(map); }
        if (file != INVALID_HANDLE_VALUE) { CloseHandle(file); }
#else
        if (data) { munmap(data, size); }
        if (file >= 0) { close(file); }
#endif
    }
};

static void require(bool ok, const char * text) {
    if (!ok) {
        throw std::runtime_error(text);
    }
}

struct ready_state {
    llama_prefetch_batch * batch;
    std::atomic<bool> * released;
    bool gated = false;
    bool violation = false;
    std::promise<void> * release;
};

static void ready(ggml_cgraph * graph, void * opaque) {
    auto & s = *static_cast<ready_state *>(opaque);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_GET_ROWS) {
            if (!s.gated) {
                s.release->set_value();
            }
            s.batch->wait(node->src[0]);
            s.violation |= !s.released->load();
            s.gated = true;
        }
    }
}

int main(int argc, char ** argv) {
    // A new file per invocation approximates cold input without evicting the host's page cache.
    if (argc < 3) {
        fprintf(stderr, "usage: %s NEW_FILE serial|parallel [rows_per_piece=262144] [global_ids=359] [delay_ms=100]\n", argv[0]);
        return 2;
    }
    try {
        const std::string path = argv[1];
        const std::string mode = argv[2];
        require(mode == "serial" || mode == "parallel", "unknown mode");
        const size_t rows = argc > 3 ? std::stoull(argv[3]) : 262144;
        const size_t ids_count = argc > 4 ? std::stoull(argv[4]) : 359;
        const size_t delay_ms = argc > 5 ? std::stoull(argv[5]) : 100;
        require(rows > 0 && rows <= 1048576 && ids_count >= 256 && ids_count <= 1048576 && delay_ms <= 60000, "probe size outside safety bounds");
        const size_t pieces = 3;
        const size_t width = 66; // 264 bytes per row, as in the engram table; F32 gives an independent oracle.
        const size_t bytes = pieces*rows*width*sizeof(float);
        FILE * existing = fopen(path.c_str(), "rb");
        if (existing) {
            fclose(existing);
            throw std::runtime_error("probe requires a new file; existing file preserved");
        }
        FILE * file = fopen(path.c_str(), "wbx");
        require(file != nullptr, "create synthetic file");
        std::vector<float> line(width);
        bool written = true;
        for (size_t p = 0; p < pieces && written; ++p) {
            for (size_t row = 0; row < rows && written; ++row) {
                std::fill(line.begin(), line.end(), float(p*rows + row));
                written = fwrite(line.data(), sizeof(float), width, file) == width;
            }
        }
        const bool closed = fclose(file) == 0;
        require(written && closed, "write synthetic file");
        mapping mapped(path.c_str(), bytes);
        require(mapped.data != nullptr, "read-only mmap");

        ggml_init_params ip = { 8*1024*1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        require(ctx != nullptr, "metadata context");
#ifdef TEST_CPU_BACKEND_PATH
        ggml_backend_reg_t cpu_reg = ggml_backend_load(TEST_CPU_BACKEND_PATH);
        require(cpu_reg != nullptr, "load CPU backend only");
        ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_reg_dev_get(cpu_reg, 0), nullptr);
#else
        ggml_backend_t cpu = ggml_backend_cpu_init();
#endif
        require(cpu != nullptr, "CPU backend");
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
        auto * set_threads = reinterpret_cast<decltype(ggml_backend_cpu_set_n_threads) *>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads) {
            set_threads(cpu, (int) std::max(1u, std::thread::hardware_concurrency()));
        }
        ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, nullptr, 1, 128, false, false);
        require(sched != nullptr, "CPU scheduler");
        ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(mapped.data, bytes);
        require(buffer != nullptr, "mmap buffer");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
        std::vector<ggml_tensor *> tables, indices, outputs;
        ggml_tensor * ids_root = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pieces*ids_count);
        ggml_set_input(ids_root);
        std::vector<std::vector<int32_t>> local(pieces, std::vector<int32_t>(ids_count, 0));
        uint64_t rng = 1;
        for (size_t i = 0; i < ids_count; ++i) {
            rng = rng*6364136223846793005ULL + 1442695040888963407ULL;
            const size_t global = (rng >> 16) % (pieces*rows);
            local[global/rows][i] = int32_t(global % rows);
        }
        for (size_t p = 0; p < pieces; ++p) {
            ggml_tensor * table = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
            table->data = static_cast<char *>(mapped.data) + p*rows*width*sizeof(float);
            table->buffer = buffer;
            table->flags |= GGML_TENSOR_FLAG_LAZY;
            ggml_tensor * ids = ggml_view_1d(ctx, ids_root, ids_count, p*ids_count*sizeof(int32_t));
            ggml_tensor * out = ggml_get_rows(ctx, table, ids);
            ggml_set_output(out);
            ggml_build_forward_expand(graph, out);
            tables.push_back(table);
            indices.push_back(ids);
            outputs.push_back(out);
        }
        require(ggml_backend_sched_alloc_graph(sched, graph), "allocate CPU graph");
        for (size_t p = 0; p < pieces; ++p) {
            ggml_backend_tensor_set(indices[p], local[p].data(), 0, ids_count*sizeof(int32_t));
        }

        // Measure the production queue and OS helper with identical ID lists in both modes.
        const int64_t begin = ggml_time_us();
        {
            llama_prefetch_batch batch;
            for (size_t p = 0; p < pieces; ++p) {
                batch.add(tables[p], local[p], LLAMA_PLE_PREFETCH_PREFILL, true);
            }
            batch.start(mode == "parallel");
            batch.finish();
        }
        printf("{\"kind\":\"pieces\",\"mode\":\"%s\",\"rows_per_piece\":%zu,\"global_ids\":%zu,\"row_bytes\":264,\"us\":%lld}\n",
               mode.c_str(), rows, ids_count, (long long) (ggml_time_us() - begin));

        // Hold the real hint behind a latch, then use the production scheduler hook to gate GET_ROWS.
        std::promise<void> entered, release;
        auto released_future = release.get_future().share();
        std::atomic<bool> released(false);
        const auto submitting_thread = std::this_thread::get_id();
        {
            llama_prefetch_batch batch(sched);
            batch.add(tables[0], local[0], LLAMA_PLE_PREFETCH_PREFILL, true);
            batch.start(false, [&] {
                entered.set_value();
                // Thread-creation fallback runs on the caller and cannot await its future CPU callback.
                if (std::this_thread::get_id() != submitting_thread) {
                    released_future.wait();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                released.store(true);
            });
            entered.get_future().wait();
            ready_state state = { &batch, &released, false, false, &release };
            ggml_backend_sched_set_cpu_ready_callback(sched, ready, &state);
            const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
            if (!state.gated) {
                release.set_value();
            }
            batch.finish();
            require(status == GGML_STATUS_SUCCESS && state.gated && !state.violation, "gated CPU compute");
            printf("{\"kind\":\"gate\",\"delay_ms\":%zu,\"wait_us\":%lld,\"released_before_gather\":true}\n", delay_ms, (long long) batch.wait_us());
        }
        std::vector<float> gathered(width*ids_count);
        const auto verify = [&] {
            for (size_t p = 0; p < pieces; ++p) {
                ggml_backend_tensor_get(outputs[p], gathered.data(), 0, gathered.size()*sizeof(float));
                for (size_t i = 0; i < ids_count; ++i) {
                    for (size_t j = 0; j < width; ++j) {
                        require(gathered[i*width + j] == float(p*rows + local[p][i]), "gather oracle");
                    }
                }
            }
        };
        verify();
        {
            llama_prefetch_batch automatic(sched);
            automatic.prepare(graph, LLAMA_PLE_PREFETCH_PREFILL, 2);
            require(automatic.task_count() == pieces, "automatic lazy-table discovery with input views");
            require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "automatic CPU compute");
            automatic.finish();
        }
        verify();
        ggml_backend_sched_free(sched);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        printf("OK (synthetic file retained)\n");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
