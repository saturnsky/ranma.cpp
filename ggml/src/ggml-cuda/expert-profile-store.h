// ranma: on-disk expert selection profile store, format version 2.
//
// A record is one self-describing binary file; the scores are recomputed in
// memory from the records inside the window, so nothing derived is persisted.
//
// The store carries no device code and no CUDA dependency on purpose: it is a
// plain C++ translation unit so it can be built and tested without a GPU.

#pragma once

#include "expert-score.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ggml_cuda_expert {

struct profile_record_meta {
    uint64_t timestamp_s = 0;      // unix seconds
    uint64_t request_count = 0;
    uint64_t input_tokens = 0;
    uint64_t output_tokens = 0;
    uint64_t bank_tokens = 0;      // tokens attributed to this bank in the interval
};

struct profile_record {
    uint64_t sequence = 0;
    profile_record_meta meta;
    std::vector<uint64_t> counts;  // n_layers * n_experts
};

struct profile_store_params {
    std::filesystem::path bank_dir;  // <profile-dir>/<bank label>
    std::string label;               // bank label, stored in the manifest only
    std::string signature;           // geometry signature; a manifest with a different signature makes the store incompatible
    int n_layers = 0;
    int n_experts = 0;
    scoring_params scoring;
    bool archive = false;            // true: records that fall out of the window move to <bank_dir>/archive/ instead of being deleted
};

enum class profile_store_status { ok, missing, incompatible, corrupt };

class profile_store {
public:
    explicit profile_store(profile_store_params params);

    // Reads the manifest and the newest `scoring.window` records. Creates nothing on disk.
    // missing      = no manifest (cold start, checkpoint() will create the directory)
    // incompatible = manifest signature/geometry/format differ, or the manifest does not parse
    //                (the store refuses to write and reports so)
    // corrupt      = manifest ok but every record was unreadable (still usable: cold)
    // ok           = at least the manifest is fine
    // Records with a bad CRC, a short length or the wrong geometry are skipped and counted.
    profile_store_status open();
    profile_store_status status() const;

    // Deletes every record and the manifest (used by an explicit --expert-profile-reset),
    // then behaves like a missing store.
    bool reset();

    // Appends one record and refreshes the manifest, the window and the scores.
    // delta.size() must equal n_layers*n_experts; a delta whose total is 0 is not
    // written (returns true, nothing changes). Refuses when status() == incompatible.
    bool checkpoint(const std::vector<uint64_t> & delta, const profile_record_meta & meta);

    const std::vector<profile_record> & window() const;   // oldest first
    const std::vector<uint64_t> & scores() const;         // n_counts fixed-point scores; all zero when the window is empty
    uint64_t total_selections() const;                    // sum of all counts in the window
    size_t   skipped_records() const;                     // unreadable records seen by open()
    uint64_t next_sequence() const;
    const profile_store_params & params() const;

    // Last failure seen by open()/checkpoint()/reset(). The store never logs by
    // itself so the caller decides how (and whether) to report.
    const std::string & last_error() const;

private:
    std::filesystem::path records_dir() const;
    std::filesystem::path archive_dir() const;
    std::filesystem::path manifest_path() const;
    std::filesystem::path record_path(uint64_t sequence) const;

    bool read_manifest();
    bool write_manifest(uint64_t updated_s);
    bool read_record(const std::filesystem::path & path, profile_record & out);
    bool write_record(const profile_record & record);
    void trim_window();
    void recompute();

    profile_store_params  store_params;
    size_t                n_counts = 0;
    profile_store_status  store_status = profile_store_status::missing;
    std::vector<profile_record> store_window;   // oldest first
    std::vector<uint64_t> store_scores;
    uint64_t              store_total = 0;
    size_t                store_skipped = 0;
    uint64_t              store_next_sequence = 1;
    std::string           store_last_error;
};

} // namespace ggml_cuda_expert
