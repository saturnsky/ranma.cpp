#pragma once

#include "bench-expert-options.h"

#include <chrono>
#include <filesystem>
#include <cstdio>

// Owns only the unique temporary directory it creates. The input profile is read-only.
struct bench_expert_profiles {
    std::filesystem::path root, active, seed;
    bool keep = false;

    explicit bench_expert_profiles(const bench_expert_options & options) : keep(options.keep) {
        if (!options.enabled() || options.mode == "off") { return; }
        const auto source = std::filesystem::path(options.profile);
        if (options.mode == "cold") {
            if (std::filesystem::exists(source) &&
                    (!std::filesystem::is_directory(source) || !std::filesystem::is_empty(source))) {
                throw std::runtime_error("cold profile directory must be empty or absent");
            }
            active = source;
            return;
        }
        if (!std::filesystem::is_directory(source) || std::filesystem::is_symlink(source)) {
            throw std::runtime_error("warm profile source must be an existing directory, not a symlink");
        }
        for (const auto & entry : std::filesystem::recursive_directory_iterator(source)) {
            if (entry.is_symlink()) { throw std::runtime_error("warm profiles must not contain symlinks"); }
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() / ("llama-bench-expert-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(root)) { throw std::runtime_error("cannot create unique profile workspace"); }
        try {
            active = root / "run";
            if (options.restore_each) {
                seed = root / "seed";
                std::filesystem::copy(source, seed, std::filesystem::copy_options::recursive);
                restore();
            } else { std::filesystem::copy(source, active, std::filesystem::copy_options::recursive); }
        } catch (...) {
            std::error_code ec; std::filesystem::remove_all(root, ec);
            root.clear(); throw;
        }
        fprintf(stderr, "expert benchmark: source %s, process profile %s (%s at exit); repetitions evolve in this process\n",
            source.string().c_str(), active.string().c_str(), keep ? "kept" : "removed");
    }

    bench_expert_profiles(const bench_expert_profiles &) = delete;
    bench_expert_profiles & operator=(const bench_expert_profiles &) = delete;

    void restore() {
        if (root.empty() || seed.empty() || active.parent_path() != root) {
            throw std::runtime_error("profile restore has no owned workspace");
        }
        std::filesystem::remove_all(active);
        std::filesystem::copy(seed, active, std::filesystem::copy_options::recursive);
    }

    ~bench_expert_profiles() {
        if (!root.empty() && !keep) {
            std::error_code ec; std::filesystem::remove_all(root, ec);
            if (ec) { fprintf(stderr, "expert benchmark: cannot remove %s: %s\n", root.string().c_str(), ec.message().c_str()); }
        }
    }
};
