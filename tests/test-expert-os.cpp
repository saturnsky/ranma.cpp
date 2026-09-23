// ranma: unbuffered read queue of the expert cache SSD tier (expert-os.h).
//
// The tier reads whole 4 KiB sector ranges straight into ring slots, so the three things that can
// go wrong are the alignment arithmetic around a slice that does not start on a sector boundary,
// a read that runs past the end of the file, and a read that never completes. One temporary file
// covers all three. Nothing here needs a GPU or a model.

#include "expert-os.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>
#include <thread>

using namespace ggml_cuda_expert;

static int failures = 0;

static void check(bool condition, const char * what) {
    if (!condition) {
        printf("FAIL %s\n", what);
        ++failures;
    }
}

// Deterministic content so a read can be checked against the offset it came from.
static uint8_t byte_at(size_t offset) {
    return (uint8_t) ((offset*131u + (offset >> 8)*7u + 17u) & 0xFF);
}

int main() {
    uint32_t mailbox = 0;
    uint64_t wide = UINT64_C(0x123456789abcdef0);
    int published = 0;
    std::thread worker([&] { published = 73; expert_os::store_release(&mailbox, 1); });
    while (!expert_os::load_acquire(&mailbox)) { std::this_thread::yield(); }
    check(published == 73 && expert_os::load_acquire(&wide) == wide, "mailbox acquire observes the published payload");
    worker.join();
    std::string pin_error;
    std::thread pinned([&] { check(expert_os::pin_current_thread(-1, pin_error), "last logical CPU pin"); });
    pinned.join();

    check(expert_os::align_down_io(0) == 0, "align_down_io(0)");
    check(expert_os::align_down_io(4095) == 0, "align_down_io(4095)");
    check(expert_os::align_down_io(4096) == 4096, "align_down_io(4096)");
    check(expert_os::align_down_io(5000) == 4096, "align_down_io(5000)");
    check(expert_os::align_up_io(0) == 0, "align_up_io(0)");
    check(expert_os::align_up_io(1) == 4096, "align_up_io(1)");
    check(expert_os::align_up_io(4096) == 4096, "align_up_io(4096)");
    check(expert_os::align_up_io(4097) == 8192, "align_up_io(4097)");

    if (!expert_os::supported()) {
        printf("expert-os: no implementation on this platform, read queue tests skipped\n");
        printf("%s\n", failures == 0 ? "OK" : "FAILED");
        return failures == 0 ? 0 : 1;
    }

    // A file of three whole sectors plus a partial one, so the last read is short.
    const size_t whole_sectors = 3;
    const size_t tail_bytes    = 1234;
    const size_t file_bytes    = whole_sectors*expert_os::io_alignment + tail_bytes;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path()/("ranma-expert-os-" + std::to_string((unsigned long long) time(nullptr)) + ".bin");
    {
        std::vector<uint8_t> content(file_bytes);
        for (size_t i = 0; i < file_bytes; ++i) {
            content[i] = byte_at(i);
        }
        FILE * f = fopen(path.string().c_str(), "wb");
        check(f != nullptr, "temporary file created");
        if (f == nullptr) {
            printf("FAILED\n");
            return 1;
        }
        check(fwrite(content.data(), 1, file_bytes, f) == file_bytes, "temporary file written");
        fclose(f);
    }

    expert_os::file_handle file = expert_os::open_unbuffered(path.string().c_str());
    check(file != nullptr, "open_unbuffered");
    if (file == nullptr) {
        std::filesystem::remove(path);
        printf("FAILED\n");
        return 1;
    }
    check(expert_os::file_size(file) == file_bytes, "file_size");

    const size_t buffer_bytes = 4*expert_os::io_alignment;
    uint8_t * buffer = (uint8_t *) expert_os::aligned_alloc(buffer_bytes, expert_os::io_alignment);
    check(buffer != nullptr, "aligned buffer");

    {
        // whole first sector
        expert_os::read_queue queue(4);
        check(queue.valid(), "read queue valid");
        memset(buffer, 0, buffer_bytes);
        expert_os::read_op op;
        op.file   = file;
        op.offset = 0;
        op.dst    = buffer;
        op.bytes  = expert_os::io_alignment;
        check(queue.submit(&op, 1), "submit aligned read");
        std::string reason;
        check(queue.wait_all(30000, &reason), ("wait_all aligned read: " + reason).c_str());
        check(queue.results().size() == 1 && queue.results()[0].got == expert_os::io_alignment,
            "aligned read delivered a whole sector");
        bool same = true;
        for (size_t i = 0; i < expert_os::io_alignment; ++i) {
            same = same && buffer[i] == byte_at(i);
        }
        check(same, "aligned read content");
    }

    {
        // a slice that does not start on a sector boundary: read the covering sector range and take
        // the payload at slot + h, which is what the ring does
        const uint64_t slice_offset = 5000;
        const size_t   slice_bytes  = 3000;
        const uint64_t aligned      = expert_os::align_down_io(slice_offset);
        const size_t   h            = (size_t) (slice_offset - aligned);
        const size_t   span         = expert_os::align_up_io(h + slice_bytes);
        check(h != 0, "the slice offset is not sector aligned");
        check(span <= buffer_bytes, "covering range fits the buffer");

        expert_os::read_queue queue(4);
        memset(buffer, 0, buffer_bytes);
        expert_os::read_op op;
        op.file   = file;
        op.offset = aligned;
        op.dst    = buffer;
        op.bytes  = span;
        check(queue.submit(&op, 1), "submit unaligned slice");
        std::string reason;
        check(queue.wait_all(30000, &reason), ("wait_all unaligned slice: " + reason).c_str());
        check(queue.results()[0].got >= h + slice_bytes, "covering range delivered");
        bool same = true;
        for (size_t i = 0; i < slice_bytes; ++i) {
            same = same && buffer[h + i] == byte_at((size_t) slice_offset + i);
        }
        check(same, "unaligned slice content at slot + h");
    }

    {
        // past the last whole sector: the operating system delivers what is there and no more
        expert_os::read_queue queue(4);
        memset(buffer, 0, buffer_bytes);
        expert_os::read_op op;
        op.file   = file;
        op.offset = whole_sectors*expert_os::io_alignment;
        op.dst    = buffer;
        op.bytes  = 2*expert_os::io_alignment;
        check(queue.submit(&op, 1), "submit read at end of file");
        std::string reason;
        check(queue.wait_all(30000, &reason), ("wait_all at end of file: " + reason).c_str());
        check(queue.results()[0].got == tail_bytes, "short read reports the bytes that exist");
        bool same = true;
        for (size_t i = 0; i < tail_bytes; ++i) {
            same = same && buffer[i] == byte_at(whole_sectors*expert_os::io_alignment + i);
        }
        check(same, "short read content");
    }

    {
        // misaligned descriptors are refused before anything is issued
        expert_os::read_queue queue(4);
        expert_os::read_op op;
        op.file   = file;
        op.offset = 1;
        op.dst    = buffer;
        op.bytes  = expert_os::io_alignment;
        check(!queue.submit(&op, 1), "submit refuses an unaligned offset");
        op.offset = 0;
        op.bytes  = 100;
        check(!queue.submit(&op, 1), "submit refuses an unaligned length");
        op.bytes  = expert_os::io_alignment;
        op.dst    = buffer + 1;
        check(!queue.submit(&op, 1), "submit refuses an unaligned destination");
    }

    {
        // A closed file handle fails at ReadFile; the same queue can then serve a valid read.
        expert_os::read_queue queue(2);
        expert_os::file_handle closed = expert_os::open_unbuffered(path.string().c_str());
        check(closed != nullptr, "open handle for error-path test");
        expert_os::read_op op;
        op.file = closed;
        op.offset = 0;
        op.dst = buffer;
        op.bytes = expert_os::io_alignment;
        expert_os::close_file(closed);
        check(queue.submit(&op, 1), "queue accepts descriptor before it checks the handle");
        std::string reason;
        check(!queue.wait_all(30000, &reason), "a closed read handle fails");
        check(!reason.empty(), "the failed read gives a reason");
        check(!queue.results()[0].ok, "the failed read is not marked successful");
        op.file = file;
        check(queue.submit(&op, 1), "queue accepts a read after an error");
        reason.clear();
        check(queue.wait_all(30000, &reason), "queue completes a read after an error");
        check(queue.results()[0].got == expert_os::io_alignment, "retry has a complete result");
    }

    {
        // progress: each report names a completed prefix in submit order whose bytes are in place,
        // the reports grow strictly, and the last one covers every read
        const size_t n = 12;
        uint8_t * many = (uint8_t *) expert_os::aligned_alloc(n*expert_os::io_alignment, expert_os::io_alignment);
        check(many != nullptr, "progress buffer");
        expert_os::read_queue queue(2);
        std::vector<expert_os::read_op> ops;
        for (size_t i = 0; i < n; ++i) {
            expert_os::read_op op;
            op.file   = file;
            op.offset = (i % whole_sectors)*expert_os::io_alignment;
            op.dst    = many + i*expert_os::io_alignment;
            op.bytes  = expert_os::io_alignment;
            ops.push_back(op);
        }
        memset(many, 0, n*expert_os::io_alignment);
        check(queue.submit(ops.data(), ops.size()), "submit the progress batch");
        std::vector<size_t> reports;
        bool in_place = true;
        std::string reason;
        const bool ok = queue.wait_all(30000, &reason, [&](size_t count) {
            reports.push_back(count);
            for (size_t i = 0; i < count; ++i) {
                const size_t base = (i % whole_sectors)*expert_os::io_alignment;
                for (size_t b = 0; b < expert_os::io_alignment; b += 511) {
                    in_place = in_place && many[i*expert_os::io_alignment + b] == byte_at(base + b);
                }
            }
            return true;
        });
        check(ok, ("wait_all with progress: " + reason).c_str());
        bool growing = !reports.empty();
        for (size_t i = 1; i < reports.size(); ++i) {
            growing = growing && reports[i] > reports[i - 1];
        }
        check(growing && reports.back() == n, "progress reports grow to the whole batch");
        check(in_place, "a reported prefix is in place");

        // a reader that gives up fails the wait, and the queue still works afterwards
        check(queue.submit(ops.data(), ops.size()), "submit the refused batch");
        reason.clear();
        check(!queue.wait_all(30000, &reason, [](size_t) { return false; }), "a refusing reader fails the wait");
        check(!reason.empty(), "a refusing reader gives a reason");
        check(queue.submit(ops.data(), 1), "the queue takes work after a refusal");
        reason.clear();
        check(queue.wait_all(30000, &reason), ("the queue works after a refusal: " + reason).c_str());
        expert_os::aligned_free(many);
    }

    {
        // a deadline that has already passed must not leave the queue unusable
        expert_os::read_queue queue(2);
        std::vector<expert_os::read_op> ops;
        for (int i = 0; i < 64; ++i) {
            expert_os::read_op op;
            op.file   = file;
            op.offset = 0;
            op.dst    = buffer;
            op.bytes  = expert_os::io_alignment;
            ops.push_back(op);
        }
        check(queue.submit(ops.data(), ops.size()), "submit the deadline batch");
        std::string reason;
        const bool ok = queue.wait_all(0, &reason);
        check(ok || !reason.empty(), "a missed deadline says why");

        expert_os::read_op op;
        op.file   = file;
        op.offset = 0;
        op.dst    = buffer;
        op.bytes  = expert_os::io_alignment;
        check(queue.submit(&op, 1), "the queue takes work after a deadline");
        std::string second;
        check(queue.wait_all(30000, &second), ("the queue works after a deadline: " + second).c_str());
        check(queue.results().size() == 1 && queue.results()[0].got == expert_os::io_alignment,
            "the read after a deadline delivered");
    }

    expert_os::aligned_free(buffer);
    expert_os::close_file(file);
    std::filesystem::remove(path);

    printf("%s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
