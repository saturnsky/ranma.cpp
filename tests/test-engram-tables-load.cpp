// The loader's engram table set rules on small fake files (no model is run; every load stops at the hyperparameters,
// after the tables were attached or refused). docs/ranma/engram-tables.md:
//  - a model whose index lists tables it does not hold takes the set given with --engram-tables (a file of any name, or a
//    folder), else the one <P>-engram*.gguf next to it (<P>-00001-of-0000K.gguf / <P>.gguf): a file of its own or the
//    first file -00001-of-0000M of a split set, the other files of a split set not counted
//  - no such set stops the load naming <P>-engram*.gguf and --engram-tables; two stop it listing both; one that does not
//    fit stops it naming what differs (no other set is tried)
//  - a set fits by architecture, table names, rows and row widths; the type and the pieces may differ
//  - a model that holds its tables keeps them without --engram-tables (a set next to it is only mentioned in a warning),
//    and --engram-tables replaces them (file or folder, same rules), here F32 tables by MXFP8
//  - a set with another table name, row count, row width or architecture is refused

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_INFO) {
        g_log += text;
    }
}

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); g_fail++; } } while (0)

static uint64_t fnv1a64(const uint8_t * data, size_t n, uint64_t h = 0xcbf29ce484222325ULL) {
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ data[i]) * 0x100000001b3ULL;
    }
    return h;
}

// a GGUF of one tensor (or none) with the given keys
struct file_spec {
    std::string arch = "deepseek41";
    std::string general_type;
    int split_no = -1, split_count = 0, split_tensors = 0;
    // the index of a set: names, parts, rows, part_rows, kind, type, piece_check
    std::vector<std::string> names;
    std::vector<uint32_t> parts;
    std::vector<uint64_t> rows, part_rows, piece_check;
    bool own_check = false;     // the piece check of its own tensor (a set of one file)
    // the index of a model that needs a set: names, rows, width
    std::vector<uint64_t> width;
    std::string tname;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne0 = 0, ne1 = 0;
    uint8_t fill = 0;
};

static uint64_t write_gguf(const std::string & path, const file_spec & f) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", f.arch.c_str());
    if (!f.general_type.empty()) {
        gguf_set_val_str(g, "general.type", f.general_type.c_str());
    }
    if (f.split_no >= 0) {
        gguf_set_val_u16(g, "split.no", (uint16_t) f.split_no);
        gguf_set_val_u16(g, "split.count", (uint16_t) f.split_count);
        gguf_set_val_i32(g, "split.tensors.count", f.split_tensors);
    }
    uint64_t check = 0;
    ggml_context * ctx = nullptr;
    ggml_tensor * t = nullptr;
    if (!f.tname.empty()) {
        ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() + ggml_row_size(f.type, f.ne0)*f.ne1 + 256,
                                /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
        ctx = ggml_init(ip);
        t = ggml_new_tensor_2d(ctx, f.type, f.ne0, f.ne1);
        ggml_set_name(t, f.tname.c_str());
        uint8_t * d = (uint8_t *) t->data;
        const size_t n = ggml_nbytes(t);
        for (size_t i = 0; i < n; ++i) { d[i] = (uint8_t) (f.fill + i*7); }
        const size_t m = n < 4096 ? n : 4096;
        check = fnv1a64(d + n - m, m, fnv1a64(d, m));
    }
    if (!f.names.empty()) {
        std::vector<const char *> n;
        for (const auto & s : f.names) { n.push_back(s.c_str()); }
        gguf_set_arr_str(g, "engram_tables.names", n.data(), n.size());
        gguf_set_arr_data(g, "engram_tables.rows", GGUF_TYPE_UINT64, f.rows.data(), f.rows.size());
        if (!f.width.empty()) {
            gguf_set_arr_data(g, "engram_tables.width", GGUF_TYPE_UINT64, f.width.data(), f.width.size());
        } else {
            gguf_set_arr_data(g, "engram_tables.parts", GGUF_TYPE_UINT32, f.parts.data(), f.parts.size());
            gguf_set_arr_data(g, "engram_tables.part_rows", GGUF_TYPE_UINT64, f.part_rows.data(), f.part_rows.size());
            gguf_set_val_str(g, "engram_tables.kind", "engram");
            gguf_set_val_str(g, "engram_tables.type", f.type == GGML_TYPE_MXFP8 ? "MXFP8" : ggml_type_name(f.type));
            if (f.own_check) {
                gguf_set_arr_data(g, "engram_tables.piece_check", GGUF_TYPE_UINT64, &check, 1);
            } else {
                gguf_set_arr_data(g, "engram_tables.piece_check", GGUF_TYPE_UINT64, f.piece_check.data(), f.piece_check.size());
            }
        }
    }
    if (t) {
        gguf_add_tensor(g, t);
    }
    gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    if (ctx) { ggml_free(ctx); }
    return check;
}

// a set "<dir>/<name>-0000i-of-00003.gguf" of one table: index + 2 MXFP8 pieces of rows/2
static std::string write_set(const std::string & dir, const std::string & name, const std::string & arch,
                             const std::string & table, int64_t rows, int64_t width) {
    file_spec p;
    p.arch = arch; p.split_count = 3; p.type = GGML_TYPE_MXFP8; p.ne0 = width; p.ne1 = rows/2;
    std::vector<uint64_t> checks;
    char buf[64];
    for (int i = 1; i <= 2; ++i) {
        p.split_no = i; p.split_tensors = 2; p.tname = table + ".p" + std::to_string(i - 1) + ".weight"; p.fill = (uint8_t) i;
        snprintf(buf, sizeof(buf), "-%05d-of-00003.gguf", i + 1);
        checks.push_back(write_gguf(dir + "/" + name + buf, p));
    }
    file_spec x;
    x.arch = arch; x.general_type = "engram_tables"; x.split_no = 0; x.split_count = 3; x.split_tensors = 2;
    x.type = GGML_TYPE_MXFP8;
    x.names = { table }; x.parts = { 2 }; x.rows = { (uint64_t) rows }; x.part_rows = { (uint64_t) (rows/2) };
    x.piece_check = checks;
    write_gguf(dir + "/" + name + "-00001-of-00003.gguf", x);
    return dir + "/" + name + "-00001-of-00003.gguf";
}

// a set of one file "<dir>/<file>": the index with the one piece of blk.1.engram_embd
static std::string write_single(const std::string & dir, const std::string & file, int64_t rows, int64_t width,
                                ggml_type type = GGML_TYPE_MXFP8) {
    file_spec x;
    x.general_type = "engram_tables"; x.own_check = true;
    x.names = { "blk.1.engram_embd" }; x.parts = { 1 }; x.rows = { (uint64_t) rows }; x.part_rows = { (uint64_t) rows };
    x.tname = "blk.1.engram_embd.p0.weight"; x.type = type; x.ne0 = width; x.ne1 = rows; x.fill = 3;
    write_gguf(dir + "/" + file, x);
    return dir + "/" + file;
}

// a model index that needs a set: blk.1.engram_embd, 64 rows of 256 values, no tensor
static void write_needing(const std::string & path) {
    file_spec m;
    m.names = { "blk.1.engram_embd" }; m.rows = { 64 }; m.width = { 256 };
    write_gguf(path, m);
}

static std::string load(const std::string & model, const char * tables) {
    g_log.clear();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers  = 0;
    mp.engram_tables = tables;
    llama_model * m = llama_model_load_from_file(model.c_str(), mp);
    if (m) { llama_model_free(m); }
    return g_log;
}

static bool has(const std::string & log, const std::string & s) {
    return log.find(s) != std::string::npos;
}

int main() {
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "test-engram-tables-load";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const std::string dir = root.generic_u8string();
    auto mkdir = [&](const char * sub) {
        std::filesystem::create_directories(root / sub);
        return dir + "/" + sub;
    };
    auto expect = [&](const char * what, const std::string & log, const std::vector<std::string> & yes,
                      const std::vector<std::string> & no) {
        bool ok = true;
        for (const auto & s : yes) { ok = ok && has(log, s); }
        for (const auto & s : no)  { ok = ok && !has(log, s); }
        if (!ok) {
            fprintf(stderr, "FAIL %s:\n%s\n", what, log.c_str());
            g_fail++;
        }
    };

    const std::string attached = "attach_engram_tables: engram table set ";
    const std::string hparams  = "hyperparameters";

    // ---- a model that needs a set
    // the one split set next to the model (its other files are no candidates); a set of another model is not looked at
    {
        const std::string d = mkdir("named");
        write_needing(d + "/M-00001-of-00002.gguf");
        write_set(d, "Other-engram", "deepseek41", "blk.1.engram_embd", 64, 256);
        write_set(d, "M-engram", "deepseek41", "blk.1.engram_embd", 64, 256);
        expect("split set next to the model", load(d + "/M-00001-of-00002.gguf", nullptr),
               { attached + d + "/M-engram-00001-of-00003.gguf: 3 files", "(from the folder of the model)", hparams }, {});
    }
    // a set of one file of another type (F32) and other pieces, in a folder given with --engram-tables; without it the
    // folder of the model holds none: the load stops naming M-engram*.gguf and --engram-tables
    {
        const std::string d  = mkdir("single-model");
        const std::string ds = mkdir("single-set");
        write_needing(d + "/M.gguf");
        write_single(ds, "M-engram-f32.gguf", 64, 256, GGML_TYPE_F32);
        expect("one-file F32 set in a folder given", load(d + "/M.gguf", ds.c_str()),
               { attached + ds + "/M-engram-f32.gguf: 1 file, engram f32", hparams }, {});
        expect("no set next to the model", load(d + "/M.gguf", nullptr),
               { "holds no M-engram*.gguf", "--engram-tables" }, { attached });
    }
    // a file given: any name
    {
        const std::string d = mkdir("anyname");
        write_needing(d + "/M.gguf");
        const std::string f = write_single(d, "tables.gguf", 64, 256);
        expect("file of any name", load(d + "/M.gguf", f.c_str()), { attached + f + ": 1 file," }, {});
    }
    // two sets of the model: stop, list both and say how to choose
    {
        const std::string d = mkdir("two");
        write_needing(d + "/M-00001-of-00002.gguf");
        write_set(d, "M-engram", "deepseek41", "blk.1.engram_embd", 64, 256);
        write_single(d, "M-engram-f16.gguf", 64, 256);
        expect("two sets", load(d + "/M-00001-of-00002.gguf", nullptr),
               { "holds 2 engram table sets", "M-engram-00001-of-00003.gguf", "M-engram-f16.gguf", "--engram-tables <file>" },
               { attached });
        // the user chooses one
        expect("two sets, one given", load(d + "/M-00001-of-00002.gguf", (d + "/M-engram-f16.gguf").c_str()),
               { attached + d + "/M-engram-f16.gguf" }, {});
    }
    // one set that does not fit: stop with what differs, no other set is tried
    {
        const std::string d = mkdir("misfit");
        write_needing(d + "/M.gguf");
        write_set(d, "M-engram", "deepseek41", "blk.1.engram_embd", 32, 256);
        expect("set of other rows", load(d + "/M.gguf", nullptr),
               { "does not fit the tables of the model", "the model has 64 rows, the set 32" }, { attached });
    }

    // ---- a model that holds its table: blk.1.engram_embd.weight F32 [256, 64]
    {
        file_spec m;
        m.tname = "blk.1.engram_embd.weight"; m.ne0 = 256; m.ne1 = 64;
        write_gguf(dir + "/model.gguf", m);
    }
    const std::string replaced = "are replaced by the engram table set";

    // no set given: the model's own table stays (the load stops at the hyperparameters)
    expect("embedded, nothing given", load(dir + "/model.gguf", nullptr), { hparams }, { replaced, "own engram tables are used" });
    // a set next to it without --engram-tables: kept, with a warning
    {
        const std::string d = mkdir("embedded-next");
        std::filesystem::copy_file(root / "model.gguf", root / "embedded-next" / "model.gguf");
        write_single(d, "model-engram.gguf", 64, 256);
        expect("embedded, a set next to it", load(d + "/model.gguf", nullptr),
               { "found " + d + "/model-engram.gguf, the model's own engram tables are used; give --engram-tables", hparams },
               { replaced });
    }
    // a matching set, as its first file (any name) and as a folder (the one model-engram*.gguf): replaced
    {
        const std::string d = mkdir("ok");
        const std::string first = write_set(d, "S-engram", "deepseek41", "blk.1.engram_embd", 64, 256);
        expect("replace by file", load(dir + "/model.gguf", first.c_str()), { replaced, hparams }, {});
        expect("folder without model-engram*", load(dir + "/model.gguf", d.c_str()), { "holds no model-engram*.gguf" },
               { replaced });
        write_set(d, "model-engram", "deepseek41", "blk.1.engram_embd", 64, 256);
        expect("replace by folder", load(dir + "/model.gguf", d.c_str()),
               { replaced + " " + d + "/model-engram-00001-of-00003.gguf", hparams }, {});
        write_single(d, "model-engram-x.gguf", 64, 256);
        expect("replace, two in the folder", load(dir + "/model.gguf", d.c_str()), { "holds 2 engram table sets" },
               { replaced });
    }
    // another table name, row count, row width or architecture: refused
    struct bad { const char * sub; const char * arch; const char * table; int64_t rows, width; const char * says; };
    const bad cases[] = {
        { "name",  "deepseek41", "blk.2.engram_embd", 64, 256, "the set has the table blk.2.engram_embd" },
        { "rows",  "deepseek41", "blk.1.engram_embd", 32, 256, "the model has 64 rows, the set 32" },
        { "width", "deepseek41", "blk.1.engram_embd", 64, 512, "wide in the model" },
        { "arch",  "qwen4exp",   "blk.1.engram_embd", 64, 256, "the set is for qwen4exp" },
    };
    for (const bad & b : cases) {
        const std::string d = mkdir(b.sub);
        const std::string first = write_set(d, "model-engram", b.arch, b.table, b.rows, b.width);
        for (const std::string & given : { first, d }) {
            expect(b.sub, load(dir + "/model.gguf", given.c_str()), { b.says }, { replaced });
        }
    }
    // a model without engram tables: the set is not used, with a warning
    {
        file_spec n;
        n.tname = "blk.0.ffn_down.weight"; n.ne0 = 256; n.ne1 = 4;
        write_gguf(dir + "/plain.gguf", n);
        expect("no tables", load(dir + "/plain.gguf", (dir + "/ok/S-engram-00001-of-00003.gguf").c_str()),
               { "holds no engram table" }, {});
    }

    std::filesystem::remove_all(root);
    llama_backend_free();
    if (g_fail) {
        fprintf(stderr, "%d failure(s)\n", g_fail);
        return 1;
    }
    printf("PASS: engram table sets: the one <P>-engram*.gguf next to the model or in a folder given, a file of any name, "
           "a set of one file of another type; none / two / a misfit stop the load; embedded tables kept (a set next to "
           "them mentioned), replaced by a set given (file and folder), refused by name, rows, width and architecture; a "
           "model without tables warns\n");
    return 0;
}
