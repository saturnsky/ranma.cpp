// ranma: standalone test for the v2 expert selection profile store.
// No GPU, no llama model: it only exercises the on-disk format and the window.

#include "expert-profile-store.h"
#include "expert-score.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using ggml_cuda_expert::profile_record;
using ggml_cuda_expert::profile_record_meta;
using ggml_cuda_expert::profile_store;
using ggml_cuda_expert::profile_store_params;
using ggml_cuda_expert::profile_store_status;
using ggml_cuda_expert::score_records;
using ggml_cuda_expert::scoring_params;

static int g_failures = 0;

static void check(bool condition, const char * what) {
    printf("%s %s\n", condition ? "ok  " : "FAIL", what);
    if (!condition) {
        g_failures++;
    }
}

static const int n_layers  = 3;
static const int n_experts = 4;
static const size_t n_counts = (size_t) n_layers * (size_t) n_experts;

static std::vector<uint64_t> make_delta(uint64_t base) {
    std::vector<uint64_t> delta(n_counts, 0);
    for (size_t i = 0; i < n_counts; i++) {
        delta[i] = base + (uint64_t) i;
    }
    return delta;
}

static profile_record_meta make_meta(uint64_t sequence) {
    profile_record_meta meta;
    meta.timestamp_s   = 1700000000ULL + sequence;
    meta.request_count = sequence;
    meta.input_tokens  = 100 * sequence;
    meta.output_tokens = 10 * sequence;
    meta.bank_tokens   = 7 * sequence;
    return meta;
}

static profile_store_params make_params(const fs::path & dir, const std::string & signature, int window, bool archive) {
    profile_store_params params;
    params.bank_dir  = dir;
    params.label     = "bank0";
    params.signature = signature;
    params.n_layers  = n_layers;
    params.n_experts = n_experts;
    params.scoring.window = window;
    params.archive   = archive;
    return params;
}

static size_t count_bin_files(const fs::path & dir) {
    std::error_code ec;
    size_t n = 0;
    if (!fs::is_directory(dir, ec)) {
        return 0;
    }
    for (const fs::directory_entry & entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".bin") {
            n++;
        }
    }
    return n;
}

static bool truncate_file(const fs::path & path, size_t keep_bytes) {
    std::vector<unsigned char> data;
    FILE * f = fopen(path.string().c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    unsigned char buffer[512];
    size_t got = 0;
    while ((got = fread(buffer, 1, sizeof(buffer), f)) > 0) {
        data.insert(data.end(), buffer, buffer + got);
    }
    fclose(f);
    if (data.size() <= keep_bytes) {
        return false;
    }
    f = fopen(path.string().c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    const bool ok = fwrite(data.data(), 1, keep_bytes, f) == keep_bytes;
    fclose(f);
    return ok;
}

static bool flip_byte(const fs::path & path, size_t offset) {
    FILE * f = fopen(path.string().c_str(), "r+b");
    if (f == nullptr) {
        return false;
    }
    if (fseek(f, (long) offset, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    int c = fgetc(f);
    if (c == EOF) {
        fclose(f);
        return false;
    }
    if (fseek(f, (long) offset, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    const bool ok = fputc(c ^ 0xFF, f) != EOF;
    fclose(f);
    return ok;
}

int main() {
    const uint64_t stamp = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    char unique[64];
    snprintf(unique, sizeof(unique), "ranma-expert-store-%llu", (unsigned long long) stamp);
    const fs::path root = fs::temp_directory_path() / unique;

    std::error_code ec;
    fs::remove_all(root, ec);

    // --- cold start -------------------------------------------------------
    {
        const fs::path dir = root / "cold";
        profile_store store(make_params(dir, "sig-a", 10, false));
        check(store.open() == profile_store_status::missing, "open() on a missing directory reports missing");
        check(!fs::exists(dir), "open() creates nothing on disk");
        check(store.next_sequence() == 1, "cold store starts at sequence 1");
        check(store.scores().size() == n_counts && store.scores()[0] == 0, "cold scores are all zero");

        const std::vector<uint64_t> delta = make_delta(1);
        check(store.checkpoint(delta, make_meta(1)), "first checkpoint succeeds");
        check(fs::exists(dir / "manifest.json"), "checkpoint writes the manifest");
        check(fs::exists(dir / "records" / "00000001.bin"), "checkpoint writes 00000001.bin");
        check(!fs::exists(dir / "records" / "00000001.bin.tmp"), "no temporary file is left behind");

        profile_store reopened(make_params(dir, "sig-a", 10, false));
        check(reopened.open() == profile_store_status::ok, "reopen reports ok");
        check(reopened.window().size() == 1, "reopened window holds one record");
        check(reopened.window()[0].sequence == 1, "reopened record keeps its sequence");
        check(reopened.window()[0].counts == delta, "reopened counts round trip");
        const profile_record_meta & meta = reopened.window()[0].meta;
        const profile_record_meta expected = make_meta(1);
        check(meta.timestamp_s == expected.timestamp_s && meta.request_count == expected.request_count &&
              meta.input_tokens == expected.input_tokens && meta.output_tokens == expected.output_tokens &&
              meta.bank_tokens == expected.bank_tokens, "reopened meta round trips");
        check(reopened.next_sequence() == 2, "next sequence after one record is 2");

        std::vector<std::vector<uint64_t>> expected_records;
        expected_records.push_back(delta);
        scoring_params scoring;
        scoring.window = 10;
        check(reopened.scores() == score_records(expected_records, scoring), "scores match score_records of the window");
        uint64_t total = 0;
        for (const uint64_t c : delta) {
            total += c;
        }
        check(reopened.total_selections() == total, "total_selections sums the window");
        check(reopened.skipped_records() == 0, "nothing skipped on a healthy store");
    }

    // --- window trimming, delete mode ------------------------------------
    {
        const fs::path dir = root / "window";
        profile_store store(make_params(dir, "sig-a", 10, false));
        store.open();
        for (uint64_t i = 1; i <= 12; i++) {
            if (!store.checkpoint(make_delta(i), make_meta(i))) {
                check(false, "checkpoint in the window loop");
                break;
            }
        }
        check(count_bin_files(dir / "records") == 10, "only 10 record files remain after 12 checkpoints");
        bool sequences_ok = store.window().size() == 10;
        for (size_t i = 0; sequences_ok && i < store.window().size(); i++) {
            sequences_ok = store.window()[i].sequence == (uint64_t) (3 + i);
        }
        check(sequences_ok, "in-memory window holds sequences 3..12 oldest first");
        check(!fs::exists(dir / "archive"), "no archive directory without archive mode");

        profile_store reopened(make_params(dir, "sig-a", 10, false));
        check(reopened.open() == profile_store_status::ok, "reopen after trimming is ok");
        bool reopened_ok = reopened.window().size() == 10;
        for (size_t i = 0; reopened_ok && i < reopened.window().size(); i++) {
            reopened_ok = reopened.window()[i].sequence == (uint64_t) (3 + i);
        }
        check(reopened_ok, "reopened window holds sequences 3..12 oldest first");
        check(reopened.next_sequence() == 13, "next sequence after 12 records is 13");
        check(reopened.scores() == store.scores(), "reopened scores match the live ones");

        // zero-total delta
        const std::vector<uint64_t> zero(n_counts, 0);
        check(reopened.checkpoint(zero, make_meta(13)), "a zero-total delta reports success");
        check(count_bin_files(dir / "records") == 10, "a zero-total delta writes no record");
        check(reopened.next_sequence() == 13, "a zero-total delta does not consume a sequence");
    }

    // --- window trimming, archive mode -----------------------------------
    {
        const fs::path dir = root / "archive";
        profile_store store(make_params(dir, "sig-a", 10, true));
        store.open();
        for (uint64_t i = 1; i <= 12; i++) {
            store.checkpoint(make_delta(i), make_meta(i));
        }
        check(count_bin_files(dir / "records") == 10, "archive mode also keeps 10 records");
        check(fs::exists(dir / "archive" / "00000001.bin") && fs::exists(dir / "archive" / "00000002.bin"),
            "the two oldest records moved to archive/");
        check(count_bin_files(dir / "archive") == 2, "archive holds exactly the two evicted records");
    }

    // --- incompatible signature ------------------------------------------
    {
        const fs::path dir = root / "signature";
        {
            profile_store store(make_params(dir, "sig-a", 10, false));
            store.open();
            store.checkpoint(make_delta(1), make_meta(1));
        }
        const auto manifest_before = fs::last_write_time(dir / "manifest.json");
        const size_t records_before = count_bin_files(dir / "records");

        profile_store other(make_params(dir, "sig-b", 10, false));
        check(other.open() == profile_store_status::incompatible, "a different signature reports incompatible");
        check(!other.last_error().empty(), "incompatible open sets last_error");
        check(!other.checkpoint(make_delta(2), make_meta(2)), "checkpoint refuses an incompatible store");
        check(count_bin_files(dir / "records") == records_before, "no record written to an incompatible store");
        check(fs::last_write_time(dir / "manifest.json") == manifest_before, "the manifest is untouched");

        profile_store geometry(make_params(dir, "sig-a", 10, false));
        profile_store_params geometry_params = make_params(dir, "sig-a", 10, false);
        geometry_params.n_experts = n_experts + 1;
        profile_store wrong_geometry(geometry_params);
        check(wrong_geometry.open() == profile_store_status::incompatible, "a different geometry reports incompatible");
        check(geometry.open() == profile_store_status::ok, "the matching store still opens");
    }

    // --- truncated newest record -----------------------------------------
    {
        const fs::path dir = root / "truncated";
        profile_store store(make_params(dir, "sig-a", 10, false));
        store.open();
        for (uint64_t i = 1; i <= 4; i++) {
            store.checkpoint(make_delta(i), make_meta(i));
        }
        check(truncate_file(dir / "records" / "00000004.bin", 80 + 8), "truncate the newest record");

        profile_store reopened(make_params(dir, "sig-a", 10, false));
        check(reopened.open() == profile_store_status::ok, "a store with one bad record still opens");
        check(reopened.skipped_records() == 1, "the truncated record is counted as skipped");
        check(reopened.window().size() == 3, "the window holds the remaining records");
        check(reopened.next_sequence() == 5, "next sequence is still max+1 after a truncated record");
        check(reopened.checkpoint(make_delta(5), make_meta(5)), "checkpoint after a truncated record succeeds");
        check(fs::exists(dir / "records" / "00000005.bin"), "the new record uses the next sequence");
    }

    // --- flipped payload byte --------------------------------------------
    {
        const fs::path dir = root / "crc";
        profile_store store(make_params(dir, "sig-a", 10, false));
        store.open();
        for (uint64_t i = 1; i <= 3; i++) {
            store.checkpoint(make_delta(i), make_meta(i));
        }
        check(flip_byte(dir / "records" / "00000002.bin", 80 + 3), "flip one payload byte");

        profile_store reopened(make_params(dir, "sig-a", 10, false));
        check(reopened.open() == profile_store_status::ok, "a store with a bad CRC still opens");
        check(reopened.skipped_records() == 1, "the CRC mismatch is counted as skipped");
        check(reopened.window().size() == 2, "the bad record is left out of the window");
        bool kept_ok = reopened.window().size() == 2 &&
            reopened.window()[0].sequence == 1 && reopened.window()[1].sequence == 3;
        check(kept_ok, "the surviving records keep their order");
    }

    // --- every record unreadable -> corrupt -------------------------------
    {
        const fs::path dir = root / "allbad";
        profile_store store(make_params(dir, "sig-a", 10, false));
        store.open();
        store.checkpoint(make_delta(1), make_meta(1));
        check(flip_byte(dir / "records" / "00000001.bin", 80), "flip the only record's payload");

        profile_store reopened(make_params(dir, "sig-a", 10, false));
        check(reopened.open() == profile_store_status::corrupt, "a manifest with no readable record reports corrupt");
        check(reopened.window().empty(), "the corrupt store has an empty window");
        check(reopened.checkpoint(make_delta(2), make_meta(2)), "a corrupt store still accepts a checkpoint");
    }

    // --- garbage manifest -------------------------------------------------
    {
        const fs::path dir = root / "manifest";
        profile_store store(make_params(dir, "sig-a", 10, false));
        store.open();
        store.checkpoint(make_delta(1), make_meta(1));

        FILE * f = fopen((dir / "manifest.json").string().c_str(), "wb");
        check(f != nullptr, "open the manifest for rewriting");
        if (f != nullptr) {
            fprintf(f, "{\"format_version\": 2, \"labl\": \"bank0\", \"signature\": \"sig-a\"}\n");
            fclose(f);
        }
        profile_store broken(make_params(dir, "sig-a", 10, false));
        check(broken.open() == profile_store_status::incompatible, "a manifest with a bad key reports incompatible");
        check(!broken.checkpoint(make_delta(2), make_meta(2)), "a broken manifest blocks checkpoints");
        check(count_bin_files(dir / "records") == 1, "nothing was written past the broken manifest");

        f = fopen((dir / "manifest.json").string().c_str(), "wb");
        if (f != nullptr) {
            fprintf(f, "not json at all\n");
            fclose(f);
        }
        profile_store garbage(make_params(dir, "sig-a", 10, false));
        check(garbage.open() == profile_store_status::incompatible, "garbage in the manifest reports incompatible");
    }

    // --- reset ------------------------------------------------------------
    {
        const fs::path dir = root / "reset";
        profile_store store(make_params(dir, "sig-a", 10, true));
        store.open();
        for (uint64_t i = 1; i <= 12; i++) {
            store.checkpoint(make_delta(i), make_meta(i));
        }
        check(store.reset(), "reset() succeeds");
        check(!fs::exists(dir / "manifest.json"), "reset() removes the manifest");
        check(count_bin_files(dir / "records") == 0, "reset() removes every record");
        check(count_bin_files(dir / "archive") == 0, "reset() removes the archive");
        check(store.status() == profile_store_status::missing, "a reset store behaves like a missing one");
        check(store.next_sequence() == 1, "a reset store restarts at sequence 1");
        check(store.checkpoint(make_delta(1), make_meta(1)), "checkpoint after reset succeeds");
        check(fs::exists(dir / "records" / "00000001.bin"), "the first record after reset is 00000001.bin");
    }

    // --- manifest string escaping round trip ------------------------------
    {
        const fs::path dir = root / "escape";
        const std::string signature = "geometry \"v2\"\nlayers\\experts\ttab\nend";
        profile_store store(make_params(dir, signature, 10, false));
        store.open();
        check(store.checkpoint(make_delta(1), make_meta(1)), "checkpoint with an escaped signature");

        profile_store reopened(make_params(dir, signature, 10, false));
        check(reopened.open() == profile_store_status::ok, "a signature with quotes, backslashes and newlines round trips");

        profile_store different(make_params(dir, signature + "x", 10, false));
        check(different.open() == profile_store_status::incompatible, "an almost-equal signature is still incompatible");
    }

    fs::remove_all(root, ec);
    check(!fs::exists(root), "the temporary directory is cleaned up");

    printf("%s: %d failure(s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
