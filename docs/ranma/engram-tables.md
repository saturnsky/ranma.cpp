# Engram tables as a GGUF set of their own

DeepSeek V4.1 (engram tables, `engram-mxfp8.md`) and Qwen3.8 Flash Next (PLE engram table, `ple-prefetch.md`) have
tables that are far larger than the rest of the model (about 101 GB per table in the original format). A split
conversion writes them as a GGUF set of their own, which the model set lists and does not contain:

    <P>-00001-of-0000K.gguf                the model index (metadata only), listing the tables it needs
    <P>-00002-of-0000K.gguf ..             the weight shards
    <P>-engram-00001-of-0000M.gguf         the index of the table set (metadata only)
    <P>-engram-00002-of-0000M.gguf ..      one piece of a table per file, the pieces of each table in order, the tables in order

or, when all pieces together fit the split size, one file `<P>-engram.gguf`: the index with all pieces, without split keys.
`<P>` is the name of the model output (`--outfile <P>.gguf`).

The DeepSeek V4.1 tables come from the FP8 checkpoint unchanged (MXFP8 is the FP8 row and its scales), so they are the
same bytes for every quantization of the weights: one table set (6 pieces, 203 GB) serves the 3.0 bpw, the 2.9 bpw and any
later conversion. On disk each model gets the set under its own name, as hard links (`<P1>-engram-…`, `<P2>-engram-…`).

Without `--split-max-size` the output is one GGUF and each table is one tensor of it (`blk.N.engram_embd.weight`,
`per_layer_token_embd.weight`), the format of the upstream PR. A model split afterwards with `llama-gguf-split` keeps that
tensor in one of its shards (the published Qwen3.8 EXL3 sets are such files). The loader reads both, and a table set can
replace such embedded tables (below). The earlier layouts of 2026-10-10 .. 2026-10-12 (table shards at the end of the model
set; a set named by `ngram_tables.base` and identified by a fingerprint) were never published and are not read.

## Pieces

The table is cut by the size of the ORIGINAL table (the format of the checkpoint, before any quantization) and the split
size (`--split-max-size 50G` = 50 x 10^9 bytes, the limit of a Hugging Face file):

    row_bytes = bytes of a row in the original checkpoint
    P = the smallest count for which a piece of ceil(rows / P) original rows is at most the split size
    R = ceil(rows / P), piece i = rows [i*R, min((i+1)*R, rows)), and P = ceil(rows / R) so that no piece is empty

A row is the unit (rows are independent in every table type). Every variant of a table that the converter writes (the
original rows, an EXL3 row codec) has the same P and the same row boundaries; the loader does not require it.

| Table | original row | rows | P (50G) |
| --- | --- | --- | --- |
| DS V4.1 engram, layer 1 / 14 | FP8 values + E8M0 scales = 264 B (the MXFP8 row) | 384,006,168 / 384,016,682 | 3 each, 6 pieces |
| Qwen3.8 PLE, 160 values per row | BF16 (the dtype of the model's config) = 320 B | 320,001,536 | 3 (102.4 GB / 3) |

Tensor names: `<table>.p<i>.weight`, `<table>` = `blk.N.engram_embd` or `per_layer_token_embd`. The per-head bias of a
Qwen3.8 EXL3 row codec table, `per_layer_token_embd.bias` (F16 `[160, heads]`), sits next to piece 0 in the same file; the
loader requires it for an EXL3 row codec table. The converter writes no bias for a table of the original dtype.

## Keys

`gguf-py/gguf/constants.py`, `Keys.EngramTables`.

The model index (a model that needs a set) lists the tables and names no file, type or pieces:

| Key | Type | Meaning |
| --- | --- | --- |
| `engram_tables.names` | string[tables] | `blk.1.engram_embd`, `blk.14.engram_embd` / `per_layer_token_embd` |
| `engram_tables.rows` | uint64[tables] | rows of the whole table |
| `engram_tables.width` | uint64[tables] | values per row (ne0 of the pieces) |

Its `split.count` and `split.tensors.count` count the weights only.

The table set index has `general.type = engram_tables`, `general.architecture` (the model's), `general.name`, and:

| Key | Type | Meaning |
| --- | --- | --- |
| `engram_tables.version` | uint32 | 1 |
| `engram_tables.kind` | string | `engram` or `ple` |
| `engram_tables.type` | string | ggml type of the pieces, e.g. `MXFP8` |
| `engram_tables.names` | string[tables] | as above |
| `engram_tables.parts` | uint32[tables] | pieces per table |
| `engram_tables.rows` | uint64[tables] | rows of the whole table |
| `engram_tables.part_rows` | uint64[tables] | rows of a piece (the last holds the rest) |
| `engram_tables.piece_check` | uint64[pieces] | below |
| `engram_tables.source` | string | the checkpoint name: `--engram-source-name`, else the name of the folder the tables were read from |
| `engram_tables.source_revision` | string | when known: `--engram-source-rev`, else the commit that `hf download --local-dir` recorded in `.cache/huggingface/download/config.json.metadata` of that folder |

## What makes a set fit

A set fits a model by its structure: the same `general.architecture`, the same table names, the same rows per table and the
same row width per table (ne0 of piece 0). The type and the pieces (count, rows per piece) may differ, so an F16 or EXL3
variant of the tables, or a set cut with another split size, serves the same model. A table of the same shape with other
contents (for instance from a fine-tune that changed the engram tables) also fits: the loader cannot tell it apart.

Piece check = FNV-1a 64 of the first 4 KiB followed by the last 4 KiB of the data of a piece, one per piece in the table
set index (`gguf-py/gguf/engram_tables.py`). The loader reads both ends of every piece (8 KiB) and compares, so a piece file
of another set or of another run of the converter under a name of this set stops the load. It is no content hash: damage
between the ends is not detected at load time; `verify_engram_tables.py --source` compares every row, and the sha256 of
every file in the manifest of an upload covers the bytes.

## Loader

`llama_model_loader::attach_engram_tables`, for the model `<P>-00001-of-0000K.gguf` or `<P>.gguf` in folder F:

1. A model whose index has `engram_tables.names` needs a set:
   - `--engram-tables PATH` (`llama_model_params::engram_tables`; else the environment variable `LLAMA_ARG_ENGRAM_TABLES`,
     read by the library too, so `llama-bench` and other tools without the option honour it): a readable file is the set
     (its first file, any name; a set of several files needs the name `<S>-00001-of-0000M.gguf` so that the other files can
     be found); anything else is a folder searched as F below.
   - else F holds the set: the files `<P>-engram*.gguf` that are a file of their own (no `-0000N-of-0000M` in the name)
     or the first file of a split set (`-00001-of-0000M`); the other files of a split set do not count.
   - none: the load stops, naming `<P>-engram*.gguf`, the folder and `--engram-tables`. Two or more: the load stops,
     listing them; give one with `--engram-tables <file>` (or a folder that holds only it). One that does not fit: the load
     stops, listing what differs; no other set is tried.
2. A model that holds its tables (`blk.N.engram_embd.weight`, `per_layer_token_embd.weight`, in any shard of a model split
   by `llama-gguf-split`) keeps them, unless `--engram-tables` / `LLAMA_ARG_ENGRAM_TABLES` gives a set: then the set is
   found as in 1 and replaces them when it fits (any type, e.g. the MXFP8 tables of the checkpoint for the Q8_0 tables of a
   quantized model); the replaced tensors are neither read nor allocated. Without the option, a `<P>-engram*.gguf` next to
   the model is only mentioned in a warning. A model without tables ignores the option with a warning.
3. The set index must have `general.type = engram_tables` and, for a split set, `split.no` 0 and no tensor; the other files
   their `split.no` / `split.count` and together `split.tensors.count` tensors. Their tensors join the model's (a tensor in
   both stops the load).
4. Every piece must have the type of `engram_tables.type` and pass its piece check.
5. The layout keys of the set (names, parts, rows, part_rows) replace those of the model metadata; the model code reads the
   pieces of each table from them (`llama_table_layout_get`), or the one tensor `<table>.weight` without them.

`llama-quantize` does not attach the set: its output keeps the keys and needs the same set.

## Converter

    --split-max-size 50G     the tables become a set of their own (without it: one tensor each, one file)
    --engram-stage all        (default) the model set and the table set
    --engram-stage weights    the model set (no table data is read)
    --engram-stage tables     the table set only (no weight, no model metadata; --outfile must be a file name)
    --engram-tables-out P     folder of the table set (created when missing), or a file name <S>.gguf that sets its name
                             (default: the folder of --outfile, the name <outfile name>-engram)
    --engram-source DIR       the original checkpoint the tables are read from (DeepSeek V4.1: the FP8 checkpoint; Qwen3.8:
                             the HF checkpoint, the table is then written in its own dtype instead of the EXL3 row codec)
    --engram-source-name / --engram-source-rev    the records of the table set index
    --engram-parts / --engram-orig-dtype          override P / the original dtype that sizes the pieces

The model index of a split n-gram model holds no tensor (as with `--no-tensor-first-split`).

## Resume

Every file of both sets is written to `<final name>.part` and renamed after its last byte was written, flushed and fsynced,
so a file under a final name is complete. `--resume` keeps a file whose size and header (header, KV, tensor info, padding)
equal what this run generates, without reading or computing its tensors, and writes every other file again (a leftover
`.part` is deleted first). `--resume-check` reports skip / write for every file of both sets, writes, deletes and creates
nothing, computes no tensor and exits; it reads the ends of the pieces for the piece checks. Checked by
`gguf-py/tests/test_resume_shards.py` and `gguf-py/tests/test_engram_tables.py` (fake data).

## Check

`scripts/verify_engram_tables.py TABLES|FOLDER [--model MODEL-00001-of-0000K.gguf] [--source DIR --source-kind
engram-fp8|ple-original|ple-exl3]` (copy in `docs/hf-upload/` of the workspace): the layout and the piece checks
recomputed from the data; with `--model` what the loader compares, that the model set holds no table tensor and whether the
loader takes the set next to the model by itself; with `--source` every row against the checkpoint.

## Runtime

The graph does not change shape with the rows. For a table of P pieces the graph input has one `I32` list of
`P * n` local row numbers and one `F32` `[1, P * n]` mask (`n` = rows gathered by the ubatch, a function of the ubatch size
only). Piece k gathers all `n` rows (the rows it does not hold read its row 0), the result is multiplied by the mask of the
piece and the P results are added: `llama_weight_aux_get_rows_pieces()`. Exactly one mask is 1 per row, so the sum is the row
itself. A table of one piece is the gather of before, without a mask.

- Cost: P gathers instead of one (DS: 3, Qwen: 3). Not measured.
- Prefetch (`--ple-prefetch`) is issued per piece with its local rows; the threshold counts the rows of the whole gather
  (`llama_prefetch_table_pieces_rows`).
- The CPU backend's threaded gather is selected by tensor name; the pieces `per_layer_token_embd.p<i>.weight` and
  `blk.N.engram_embd.p<i>.weight` are included.

## Commands (not run on a real model)

    # DeepSeek V4.1: EXL3 weights, MXFP8 engram tables from the FP8 checkpoint; the table set <name>-engram-… next to the model
    python convert_hf_to_gguf.py <exl3-folder> --outfile OUT/<name>.gguf --split-max-size 50G --engram-source <fp8-folder>
    # the same in two runs
    python convert_hf_to_gguf.py <exl3-folder> --outfile OUT/<name>.gguf --split-max-size 50G --engram-source <fp8-folder> --engram-stage weights
    python convert_hf_to_gguf.py <exl3-folder> --outfile OUT/<name>.gguf --split-max-size 50G --engram-source <fp8-folder> --engram-stage tables

    # another quantization in the same folder uses the same set under its own name (hard links)
    for f in OUT/<name>-engram-*.gguf; do ln "$f" "OUT/<name2>-engram-${f##*-engram-}"; done

    # run with the table set in another folder, or a variant of it
    llama-server -m OUT/<name>-00001-of-0000K.gguf --engram-tables T/<name>-engram-00001-of-00007.gguf
