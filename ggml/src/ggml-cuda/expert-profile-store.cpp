// ranma: on-disk expert selection profile store, format version 2. See expert-profile-store.h.

#include "expert-profile-store.h"
#include "expert-os.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <system_error>

namespace fs = std::filesystem;

namespace ggml_cuda_expert {

namespace {

constexpr uint32_t record_format_version = 2;
constexpr char     record_magic[8]       = { 'R', 'N', 'E', 'X', 'P', 'R', 'F', '2' };
constexpr size_t   record_header_bytes   = 80;
constexpr size_t   record_header_crc_bytes = 72; // everything before header_crc32

// The header is written field by field as stored little-endian integers. Every
// target of this fork is little-endian x86/x86_64, so a plain memcpy of the
// packed struct is the on-disk representation; a big-endian port would have to
// byte-swap here and nowhere else.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma pack(push, 1)
#define RANMA_PACKED
#else
#define RANMA_PACKED __attribute__((packed))
#endif
struct RANMA_PACKED record_header {
    char     magic[8];
    uint32_t version;
    uint32_t n_layers;
    uint32_t n_experts;
    uint32_t payload_crc32;
    uint64_t sequence;
    uint64_t timestamp_s;
    uint64_t request_count;
    uint64_t input_tokens;
    uint64_t output_tokens;
    uint64_t bank_tokens;
    uint32_t header_crc32;
    uint32_t reserved;
};
#if defined(_MSC_VER) && !defined(__clang__)
#pragma pack(pop)
#endif

static_assert(sizeof(record_header) == record_header_bytes, "record header must be exactly 80 bytes on disk");

using expert_os::atomic_replace;

std::string escape_json(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (const char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            case '\r': out += "\\r";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

void skip_spaces(const std::string & text, size_t & pos) {
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) {
        pos++;
    }
}

// Strict scanner for the fixed manifest layout: locate "<key>" and step past the
// colon. No JSON library, and no tolerance for a key that is missing.
bool find_key(const std::string & text, const char * key, size_t & pos) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t at = text.find(needle);
    if (at == std::string::npos) {
        return false;
    }
    pos = at + needle.size();
    skip_spaces(text, pos);
    if (pos >= text.size() || text[pos] != ':') {
        return false;
    }
    pos++;
    skip_spaces(text, pos);
    return true;
}

bool parse_u64(const std::string & text, const char * key, uint64_t & out) {
    size_t pos = 0;
    if (!find_key(text, key, pos)) {
        return false;
    }
    const char * begin = text.c_str() + pos;
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = strtoull(begin, &end, 10);
    if (end == begin || errno != 0) {
        return false;
    }
    out = (uint64_t) value;
    return true;
}

bool parse_double(const std::string & text, const char * key, double & out) {
    size_t pos = 0;
    if (!find_key(text, key, pos)) {
        return false;
    }
    const char * begin = text.c_str() + pos;
    char * end = nullptr;
    errno = 0;
    const double value = strtod(begin, &end);
    if (end == begin || errno == ERANGE) {
        return false;
    }
    out = value;
    return true;
}

bool parse_string(const std::string & text, const char * key, std::string & out) {
    size_t pos = 0;
    if (!find_key(text, key, pos)) {
        return false;
    }
    if (pos >= text.size() || text[pos] != '"') {
        return false;
    }
    pos++;
    std::string value;
    while (pos < text.size()) {
        const char c = text[pos++];
        if (c == '"') {
            out = value;
            return true;
        }
        if (c != '\\') {
            value += c;
            continue;
        }
        if (pos >= text.size()) {
            return false;
        }
        const char esc = text[pos++];
        switch (esc) {
            case '\\': value += '\\'; break;
            case '"':  value += '"';  break;
            case 'n':  value += '\n'; break;
            case 't':  value += '\t'; break;
            case 'r':  value += '\r'; break;
            case '/':  value += '/';  break;
            default:   return false; // unknown escape: the manifest is not ours
        }
    }
    return false;
}

bool read_whole_file(const fs::path & path, std::string & out) {
    FILE * f = nullptr;
#if defined(_WIN32)
    f = _wfopen(path.c_str(), L"rb");
#else
    f = fopen(path.c_str(), "rb");
#endif
    if (f == nullptr) {
        return false;
    }
    out.clear();
    char buffer[4096];
    size_t got = 0;
    while ((got = fread(buffer, 1, sizeof(buffer), f)) > 0) {
        out.append(buffer, got);
    }
    const bool bad = ferror(f) != 0;
    fclose(f);
    return !bad;
}

FILE * open_write(const fs::path & path) {
#if defined(_WIN32)
    return _wfopen(path.c_str(), L"wb");
#else
    return fopen(path.c_str(), "wb");
#endif
}

FILE * open_read(const fs::path & path) {
#if defined(_WIN32)
    return _wfopen(path.c_str(), L"rb");
#else
    return fopen(path.c_str(), "rb");
#endif
}

// Record files are named NNNNNNNN.bin; anything else in records/ is ignored.
bool sequence_from_filename(const fs::path & path, uint64_t & out) {
    if (path.extension() != ".bin") {
        return false;
    }
    const std::string stem = path.stem().string();
    if (stem.size() != 8) {
        return false;
    }
    uint64_t value = 0;
    for (const char c : stem) {
        if (c < '0' || c > '9') {
            return false;
        }
        value = value * 10 + (uint64_t) (c - '0');
    }
    out = value;
    return true;
}

} // namespace

profile_store::profile_store(profile_store_params params) : store_params(std::move(params)) {
    if (store_params.scoring.window < 1) {
        store_params.scoring.window = 1;
    }
    n_counts = (size_t) std::max(store_params.n_layers, 0) * (size_t) std::max(store_params.n_experts, 0);
    store_scores.assign(n_counts, 0);
}

fs::path profile_store::records_dir() const {
    return store_params.bank_dir / "records";
}

fs::path profile_store::archive_dir() const {
    return store_params.bank_dir / "archive";
}

fs::path profile_store::manifest_path() const {
    return store_params.bank_dir / "manifest.json";
}

fs::path profile_store::record_path(uint64_t sequence) const {
    char name[32];
    snprintf(name, sizeof(name), "%08llu.bin", (unsigned long long) sequence);
    return records_dir() / name;
}

const std::vector<profile_record> & profile_store::window() const {
    return store_window;
}

const std::vector<uint64_t> & profile_store::scores() const {
    return store_scores;
}

uint64_t profile_store::total_selections() const {
    return store_total;
}

size_t profile_store::skipped_records() const {
    return store_skipped;
}

uint64_t profile_store::next_sequence() const {
    return store_next_sequence;
}

const profile_store_params & profile_store::params() const {
    return store_params;
}

const std::string & profile_store::last_error() const {
    return store_last_error;
}

profile_store_status profile_store::status() const {
    return store_status;
}

bool profile_store::read_manifest() {
    std::string text;
    if (!read_whole_file(manifest_path(), text)) {
        store_last_error = "manifest unreadable: " + manifest_path().string();
        return false;
    }

    uint64_t format_version = 0;
    uint64_t layers = 0;
    uint64_t experts = 0;
    uint64_t window_size = 0;
    uint64_t updated_s = 0;
    double   half_life = 0.0;
    double   raw_weight = 0.0;
    double   balanced_weight = 0.0;
    std::string label;
    std::string signature;

    if (!parse_u64(text, "format_version", format_version) ||
        !parse_string(text, "label", label) ||
        !parse_string(text, "signature", signature) ||
        !parse_u64(text, "n_layers", layers) ||
        !parse_u64(text, "n_experts", experts) ||
        !parse_u64(text, "window", window_size) ||
        !parse_double(text, "half_life_turns", half_life) ||
        !parse_double(text, "raw_weight", raw_weight) ||
        !parse_double(text, "balanced_weight", balanced_weight) ||
        !parse_u64(text, "updated_s", updated_s)) {
        store_last_error = "manifest does not parse: " + manifest_path().string();
        return false;
    }
    (void) label;
    (void) window_size;
    (void) half_life;
    (void) raw_weight;
    (void) balanced_weight;
    (void) updated_s;

    if (format_version != 2) {
        store_last_error = "manifest format_version is not 2";
        return false;
    }
    if (layers != (uint64_t) store_params.n_layers || experts != (uint64_t) store_params.n_experts) {
        store_last_error = "manifest geometry differs from the running model";
        return false;
    }
    if (signature != store_params.signature) {
        store_last_error = "manifest signature differs from the running model";
        return false;
    }
    return true;
}

bool profile_store::write_manifest(uint64_t updated_s) {
    std::error_code ec;
    fs::create_directories(store_params.bank_dir, ec);
    if (ec) {
        store_last_error = "failed to create " + store_params.bank_dir.string() + ": " + ec.message();
        return false;
    }

    const fs::path target = manifest_path();
    const fs::path temporary = fs::path(target.string() + ".tmp");
    FILE * f = open_write(temporary);
    if (f == nullptr) {
        store_last_error = "failed to open " + temporary.string();
        return false;
    }
    const int written = fprintf(f,
        "{\"format_version\": 2, \"label\": \"%s\", \"signature\": \"%s\", "
        "\"n_layers\": %d, \"n_experts\": %d, \"window\": %d, "
        "\"half_life_turns\": %.17g, \"raw_weight\": %.17g, \"balanced_weight\": %.17g, "
        "\"updated_s\": %llu}\n",
        escape_json(store_params.label).c_str(),
        escape_json(store_params.signature).c_str(),
        store_params.n_layers,
        store_params.n_experts,
        store_params.scoring.window,
        store_params.scoring.half_life_turns,
        store_params.scoring.raw_weight,
        store_params.scoring.balanced_weight,
        (unsigned long long) updated_s);
    const bool ok = written > 0 && fflush(f) == 0;
    fclose(f);
    if (!ok) {
        store_last_error = "failed to write " + temporary.string();
        fs::remove(temporary, ec);
        return false;
    }
    if (!atomic_replace(temporary, target, store_last_error)) {
        fs::remove(temporary, ec);
        return false;
    }
    return true;
}

bool profile_store::read_record(const fs::path & path, profile_record & out) {
    FILE * f = open_read(path);
    if (f == nullptr) {
        return false;
    }

    unsigned char raw[record_header_bytes];
    if (fread(raw, 1, record_header_bytes, f) != record_header_bytes) {
        fclose(f);
        return false;
    }
    record_header header;
    memcpy(&header, raw, record_header_bytes);

    if (memcmp(header.magic, record_magic, sizeof(record_magic)) != 0 ||
        header.version != record_format_version) {
        fclose(f);
        return false;
    }
    if ((uint32_t) crc32(raw, record_header_crc_bytes) != header.header_crc32) {
        fclose(f);
        return false;
    }
    if (header.n_layers != (uint32_t) store_params.n_layers ||
        header.n_experts != (uint32_t) store_params.n_experts) {
        fclose(f);
        return false;
    }

    std::vector<uint64_t> counts(n_counts, 0);
    const size_t payload_bytes = n_counts * sizeof(uint64_t);
    if (payload_bytes > 0 && fread(counts.data(), 1, payload_bytes, f) != payload_bytes) {
        fclose(f);
        return false; // truncated tail of a crashed checkpoint
    }
    // Trailing bytes beyond the payload mean the file is not what the header claims.
    unsigned char extra = 0;
    const bool has_extra = fread(&extra, 1, 1, f) == 1;
    fclose(f);
    if (has_extra) {
        return false;
    }
    if ((uint32_t) crc32(counts.data(), payload_bytes) != header.payload_crc32) {
        return false;
    }

    out.sequence = header.sequence;
    out.meta.timestamp_s   = header.timestamp_s;
    out.meta.request_count = header.request_count;
    out.meta.input_tokens  = header.input_tokens;
    out.meta.output_tokens = header.output_tokens;
    out.meta.bank_tokens   = header.bank_tokens;
    out.counts = std::move(counts);
    return true;
}

bool profile_store::write_record(const profile_record & record) {
    std::error_code ec;
    fs::create_directories(records_dir(), ec);
    if (ec) {
        store_last_error = "failed to create " + records_dir().string() + ": " + ec.message();
        return false;
    }

    record_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, record_magic, sizeof(record_magic));
    header.version       = record_format_version;
    header.n_layers      = (uint32_t) store_params.n_layers;
    header.n_experts     = (uint32_t) store_params.n_experts;
    header.sequence      = record.sequence;
    header.timestamp_s   = record.meta.timestamp_s;
    header.request_count = record.meta.request_count;
    header.input_tokens  = record.meta.input_tokens;
    header.output_tokens = record.meta.output_tokens;
    header.bank_tokens   = record.meta.bank_tokens;
    header.reserved      = 0;

    const size_t payload_bytes = record.counts.size() * sizeof(uint64_t);
    header.payload_crc32 = (uint32_t) crc32(record.counts.data(), payload_bytes);

    unsigned char raw[record_header_bytes];
    memcpy(raw, &header, record_header_bytes);
    header.header_crc32 = (uint32_t) crc32(raw, record_header_crc_bytes);
    memcpy(raw, &header, record_header_bytes);

    const fs::path target = record_path(record.sequence);
    const fs::path temporary = fs::path(target.string() + ".tmp");
    FILE * f = open_write(temporary);
    if (f == nullptr) {
        store_last_error = "failed to open " + temporary.string();
        return false;
    }
    bool ok = fwrite(raw, 1, record_header_bytes, f) == record_header_bytes;
    if (ok && payload_bytes > 0) {
        ok = fwrite(record.counts.data(), 1, payload_bytes, f) == payload_bytes;
    }
    ok = ok && fflush(f) == 0;
    fclose(f);
    if (!ok) {
        store_last_error = "failed to write " + temporary.string();
        fs::remove(temporary, ec);
        return false;
    }
    if (!atomic_replace(temporary, target, store_last_error)) {
        fs::remove(temporary, ec);
        return false;
    }
    return true;
}

void profile_store::recompute() {
    store_total = 0;
    std::vector<std::vector<uint64_t>> records;
    records.reserve(store_window.size());
    for (const profile_record & record : store_window) {
        for (const uint64_t c : record.counts) {
            store_total += c;
        }
        records.push_back(record.counts);
    }
    if (records.empty()) {
        store_scores.assign(n_counts, 0);
        return;
    }
    store_scores = score_records(records, store_params.scoring);
    if (store_scores.size() != n_counts) {
        store_scores.assign(n_counts, 0);
    }
}

profile_store_status profile_store::open() {
    store_window.clear();
    store_scores.assign(n_counts, 0);
    store_total = 0;
    store_skipped = 0;
    store_next_sequence = 1;
    store_last_error.clear();

    std::error_code ec;
    if (!fs::exists(manifest_path(), ec) || ec) {
        store_status = profile_store_status::missing;
        return store_status;
    }
    if (!read_manifest()) {
        // A manifest that is present but unusable is never overwritten except by reset().
        store_status = profile_store_status::incompatible;
        return store_status;
    }

    // Collect candidate record files. The highest sequence ever seen - even in a
    // file that fails to load - determines the next sequence, so a crashed
    // checkpoint never gets its number reused.
    std::vector<std::pair<uint64_t, fs::path>> candidates;
    if (fs::is_directory(records_dir(), ec)) {
        for (const fs::directory_entry & entry : fs::directory_iterator(records_dir(), ec)) {
            uint64_t sequence = 0;
            if (!entry.is_regular_file(ec) || !sequence_from_filename(entry.path(), sequence)) {
                continue;
            }
            candidates.emplace_back(sequence, entry.path());
            store_next_sequence = std::max(store_next_sequence, sequence + 1);
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const std::pair<uint64_t, fs::path> & a, const std::pair<uint64_t, fs::path> & b) {
            return a.first < b.first;
        });

    const size_t window_size = (size_t) store_params.scoring.window;
    size_t first = 0;
    if (candidates.size() > window_size) {
        first = candidates.size() - window_size;
    }
    for (size_t i = first; i < candidates.size(); i++) {
        profile_record record;
        if (!read_record(candidates[i].second, record)) {
            // Bad files are left in place: the next checkpoint uses a higher
            // sequence, and the caller can inspect what crashed.
            store_skipped++;
            continue;
        }
        store_next_sequence = std::max(store_next_sequence, record.sequence + 1);
        store_window.push_back(std::move(record));
    }
    recompute();

    const bool considered_any = candidates.size() > first;
    if (considered_any && store_window.empty()) {
        store_status = profile_store_status::corrupt;
    } else {
        store_status = profile_store_status::ok;
    }
    return store_status;
}

bool profile_store::reset() {
    store_last_error.clear();
    std::error_code ec;
    fs::remove_all(records_dir(), ec);
    if (ec) {
        store_last_error = "failed to remove " + records_dir().string() + ": " + ec.message();
        return false;
    }
    fs::remove_all(archive_dir(), ec);
    if (ec) {
        store_last_error = "failed to remove " + archive_dir().string() + ": " + ec.message();
        return false;
    }
    fs::remove(manifest_path(), ec);
    if (ec) {
        store_last_error = "failed to remove " + manifest_path().string() + ": " + ec.message();
        return false;
    }

    store_window.clear();
    store_scores.assign(n_counts, 0);
    store_total = 0;
    store_skipped = 0;
    store_next_sequence = 1;
    store_status = profile_store_status::missing;
    return true;
}

void profile_store::trim_window() {
    std::error_code ec;
    std::vector<std::pair<uint64_t, fs::path>> files;
    if (!fs::is_directory(records_dir(), ec)) {
        return;
    }
    for (const fs::directory_entry & entry : fs::directory_iterator(records_dir(), ec)) {
        uint64_t sequence = 0;
        if (!entry.is_regular_file(ec) || !sequence_from_filename(entry.path(), sequence)) {
            continue;
        }
        files.emplace_back(sequence, entry.path());
    }
    const size_t window_size = (size_t) store_params.scoring.window;
    if (files.size() <= window_size) {
        return;
    }
    std::sort(files.begin(), files.end(),
        [](const std::pair<uint64_t, fs::path> & a, const std::pair<uint64_t, fs::path> & b) {
            return a.first < b.first;
        });
    const size_t drop = files.size() - window_size;
    for (size_t i = 0; i < drop; i++) {
        if (store_params.archive) {
            fs::create_directories(archive_dir(), ec);
            const fs::path destination = archive_dir() / files[i].second.filename();
            std::string error;
            if (!atomic_replace(files[i].second, destination, error)) {
                store_last_error = error;
            }
        } else {
            fs::remove(files[i].second, ec);
            if (ec) {
                store_last_error = "failed to remove " + files[i].second.string() + ": " + ec.message();
            }
        }
    }
}

bool profile_store::checkpoint(const std::vector<uint64_t> & delta, const profile_record_meta & meta) {
    store_last_error.clear();
    if (store_status == profile_store_status::incompatible) {
        store_last_error = "store is incompatible; refusing to write";
        return false;
    }
    if (delta.size() != n_counts) {
        store_last_error = "delta size does not match the model geometry";
        return false;
    }

    uint64_t total = 0;
    for (const uint64_t c : delta) {
        total += c;
    }
    if (total == 0) {
        // Nothing was selected in this interval: an empty record would only age
        // the window without adding information.
        return true;
    }

    profile_record record;
    record.sequence = store_next_sequence;
    record.meta = meta;
    record.counts = delta;

    if (!write_record(record)) {
        return false;
    }
    if (!write_manifest(meta.timestamp_s)) {
        return false;
    }
    store_next_sequence++;

    store_window.push_back(std::move(record));
    const size_t window_size = (size_t) store_params.scoring.window;
    if (store_window.size() > window_size) {
        const ptrdiff_t drop = (ptrdiff_t) (store_window.size() - window_size);
        store_window.erase(store_window.begin(), store_window.begin() + drop);
    }
    trim_window();
    recompute();
    store_status = profile_store_status::ok;
    return true;
}

} // namespace ggml_cuda_expert
