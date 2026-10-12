"""The engram tables (DeepSeek V4.1 engram, Qwen3.8 PLE n-gram) of a conversion, written as a GGUF set of their own.

A split conversion (--split-max-size) writes the model set WITHOUT the tables, and the table set:

    <P>-00001-of-0000K.gguf ..         the index (metadata only) and the weight shards; the index lists the tables (names,
                                       rows, row widths) and names no file
    <P>-engram-00001-of-0000M.gguf ..  the index of the tables, then one piece of a table per file
    <P>-engram.gguf                    instead, when all pieces fit the split size: one file, the index with the pieces

<P> is the name of the model output (--outfile without .gguf). The loader takes the one set <P>-engram*.gguf next to the
model, or the set given with --engram-tables, when its architecture, table names, rows and row widths fit
(docs/ranma/engram-tables.md).

The table of each n-gram layer is cut into P pieces of R rows, one tensor `<table>.p<i>.weight` per piece. P and R follow
from the table as the ORIGINAL checkpoint stores it (before any quantization) and the split size:

    P = the smallest count for which a piece of the original rows is at most the split size
    R = ceil(rows / P)                 (a row is the unit: no other alignment, rows are independent)
    piece i = rows [i*R, min((i+1)*R, rows)), P is then lowered to ceil(rows / R) so that no piece is empty

The table set does not depend on the weights: the tables of DeepSeek V4.1 come from the FP8 checkpoint whatever the
quantization of the weights, so one table set serves every quantization of the model (a link under the name of another
model, <P2>-engram..., lets it find the set). The piece check is defined in gguf-py/gguf/engram_tables.py.

Without --split-max-size the output is one GGUF and each table is one tensor of it (`blk.N.engram_embd.weight`,
`per_layer_token_embd.weight`).

--engram-stage selects the sets of a split conversion: all (default, both), weights (the model set; no table data is
read), tables (the table set only).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

import gguf
from gguf import engram_tables as nt

logger = logging.getLogger("engram_tables")

STAGES = ("all", "weights", "tables")

# element sizes of the dtypes an original table can have (safetensors and torch spellings)
ORIG_DTYPE_BYTES = {
    "F32": 4, "FP32": 4, "float32": 4, "torch.float32": 4,
    "F16": 2, "FP16": 2, "float16": 2, "torch.float16": 2,
    "BF16": 2, "bfloat16": 2, "torch.bfloat16": 2,
}


def orig_dtype_bytes(dtype: str) -> int:
    try:
        return ORIG_DTYPE_BYTES[dtype]
    except KeyError:
        raise ValueError(f"original table dtype {dtype!r} is not one of {sorted(set(ORIG_DTYPE_BYTES))}") from None


def plan_parts(rows: int, row_bytes: int, limit: int) -> tuple[int, int]:
    """(P, R) for a table of `rows` rows of `row_bytes` bytes (original format) and a shard limit of `limit` bytes."""
    if rows <= 0 or row_bytes <= 0:
        raise ValueError(f"table of {rows} rows x {row_bytes} bytes")
    if row_bytes > limit:
        raise ValueError(f"a row of {row_bytes} bytes does not fit the split size {limit}")
    parts = max(1, -(-rows * row_bytes // limit))
    while -(-rows // parts) * row_bytes > limit:
        parts += 1
    part_rows = -(-rows // parts)
    return -(-rows // part_rows), part_rows


@dataclass
class TableSource:
    """One table as the converter reads it."""
    name: str                                     # base tensor name, e.g. blk.1.engram_embd, per_layer_token_embd
    rows: int
    orig_row_bytes: int                           # bytes of a row in the ORIGINAL checkpoint format (plans the pieces)
    row_bytes: int                                # stored bytes of a row of this variant
    qtype: gguf.GGMLQuantizationType
    read_rows: Callable[[int, int], np.ndarray]   # stored rows [r0, r1) -> uint8 [n, row_bytes] (samples)
    piece: Callable[[int, int], Any]              # tensor of the stored rows [r0, r1) for GGUFWriter.add_tensor (uint8 [n, row_bytes])


def hf_revision(folder: Path) -> str | None:
    """The commit of a folder written by `hf download --local-dir` (.cache/huggingface/download/<file>.metadata)."""
    meta = folder / ".cache" / "huggingface" / "download"
    for name in ("config.json.metadata", "model.safetensors.index.json.metadata"):
        p = meta / name
        if p.is_file():
            first = p.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
            if first and len(first[0].strip()) == 40:
                return first[0].strip()
    return None


class EngramTables:
    """The tables of one conversion: the plan, the keys of the model index and the table set."""

    def __init__(self, model: Any):
        self.model = model
        self.stage: str = model.engram_stage
        self.limit: int = model.split_max_size
        self.kind: str | None = None
        self.sources: list[TableSource] = []
        self.plans: list[nt.TableDesc] = []
        self.extras: list[tuple[str, np.ndarray]] = []
        self.source_folder: Path | None = None
        if self.stage not in STAGES:
            raise ValueError(f"--engram-stage {self.stage!r} is not one of {STAGES}")
        if self.stage != "all" and self.limit <= 0:
            raise ValueError("--engram-stage weights / tables needs --split-max-size (the tables are a set of their own)")

    @property
    def active(self) -> bool:
        """The tables are a set of their own when the output is split by size; otherwise each is one tensor of the file."""
        return self.limit > 0

    @property
    def registered(self) -> bool:
        return bool(self.sources)

    # -- planning
    def set_tables(self, kind: str, sources: Sequence[TableSource], source_folder: Path,
                   extras: Sequence[tuple[str, np.ndarray]] = ()) -> None:
        """Register the tables (once): plan the pieces (no table data is read)."""
        if self.sources:
            raise ValueError("the engram tables are registered twice")
        if not sources:
            raise ValueError("no engram table to register")
        if len({s.qtype for s in sources}) != 1:
            raise ValueError(f"the engram tables have different types {sorted({s.qtype.name for s in sources})}")
        self.kind = kind
        self.sources = list(sources)
        self.extras = [(n, np.ascontiguousarray(a)) for n, a in extras]
        self.source_folder = source_folder
        override = self.model.engram_parts
        plans = []
        for s in sources:
            parts, part_rows = plan_parts(s.rows, s.orig_row_bytes, self.limit)
            if override is not None:
                if override < 1:
                    raise ValueError("--engram-parts must be at least 1")
                part_rows = -(-s.rows // override)
                parts = -(-s.rows // part_rows)
                if parts != override:
                    raise ValueError(f"{s.name}: {s.rows} rows cannot be cut into {override} pieces")
            if part_rows * s.row_bytes > self.limit:
                logger.warning("%s: pieces of %.1f GB are above the split size %.1f GB", s.name, part_rows * s.row_bytes / 1e9,
                               self.limit / 1e9)
            width = gguf.quant_shape_from_byte_shape((s.row_bytes,), s.qtype)[0]
            plans.append(nt.TableDesc(s.name, s.rows, parts, part_rows, s.row_bytes, width))
            logger.info("engram table %s: %d rows x %d B (original) = %.1f GB -> %d pieces of %d rows, %s %d B per row",
                        s.name, s.rows, s.orig_row_bytes, s.rows * s.orig_row_bytes / 1e9, parts, part_rows, s.qtype.name,
                        s.row_bytes)
        self.plans = plans

    def piece_checks(self) -> list[int]:
        """The piece checks (the ends of every piece, read from the table source)."""
        def read(ti: int, a: int, b: int) -> bytes:
            return np.ascontiguousarray(self.sources[ti].read_rows(a, b), dtype=np.uint8).tobytes()
        return nt.piece_checks(self.plans, read)

    @property
    def type_name(self) -> str:
        return self.sources[0].qtype.name

    @property
    def single_file(self) -> bool:
        """All pieces (and the bias) fit the split size: the set is one file, the index holding the pieces."""
        total = sum(p.rows * p.row_bytes for p in self.plans) + sum(a.nbytes for _, a in self.extras)
        return total <= self.limit

    @property
    def split_count(self) -> int:
        return 1 if self.single_file else 1 + sum(p.parts for p in self.plans)

    @property
    def base_name(self) -> str:
        """<P>-engram for the model output <P>.gguf (the name the loader looks for next to the model), or the name that
        --engram-tables-out gives."""
        out = self.model.engram_tables_out
        if out is not None and out.name.endswith(".gguf"):
            m = nt.SHARD_RE.match(out.name)
            return m.group(1) if m else out.name[:-len(".gguf")]
        return nt.set_base(self.model_prefix())

    def model_prefix(self) -> str:
        """<P> of the model output <P>.gguf (the model set is <P>-00001-of-0000K.gguf)."""
        fo = Path(self.model.fname_out)
        if fo.is_dir():
            raise ValueError("--engram-stage tables with --outfile a folder: the name of the model output is not known before "
                             "its metadata; give --outfile <name>.gguf (the same as in the weights stage) or "
                             "--engram-tables-out <name>-engram.gguf")
        ftype = getattr(self.model, "ftype", None)
        name = gguf.fill_templated_filename(fo.name, ftype.name.partition("_")[2] if ftype is not None else None)
        name = name[:-len(".gguf")] if name.endswith(".gguf") else name
        m = re.match(r"^(.*)-\d{5}-of-\d{5}$", name)
        return m.group(1) if m else name

    def table_set_folder(self) -> Path:
        """--engram-tables-out: a name ending in .gguf is a file name of the set, anything else a folder (created when
        missing); default: the folder of --outfile."""
        out = self.model.engram_tables_out
        if out is not None:
            return out.parent if out.name.endswith(".gguf") else out
        fo = Path(self.model.fname_out)
        return fo if fo.is_dir() else fo.parent

    # -- the model index
    def write_index_keys(self, w: gguf.GGUFWriter) -> None:
        """The keys of the model index that list the tables the set must have (a no-op when the tables are not a set of
        their own): names, rows, row widths. No file name, type or pieces: any set with these tables fits."""
        if not self.sources:
            return
        K = gguf.Keys.EngramTables
        w.add_array(K.NAMES, [p.name for p in self.plans])
        w.add_key_value(K.ROWS, [p.rows for p in self.plans], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
        w.add_key_value(K.WIDTH, [p.width for p in self.plans], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)

    # -- the table set
    def write_table_set(self) -> list[Path]:
        """Write the table set (resume and .part as for the model set); returns its file names."""
        if not self.sources:
            raise ValueError("this model has no engram tables to write")
        K = gguf.Keys.EngramTables
        model = self.model
        folder = self.table_set_folder()
        if not model.resume_check:
            folder.mkdir(parents=True, exist_ok=True)
        base = self.base_name
        if "-engram" not in base:
            logger.warning("engram table set %s: the name is not <model>-engram..., so the loader finds it only by "
                           "--engram-tables / LLAMA_ARG_ENGRAM_TABLES", base)
        path = folder / (base + ".gguf")
        single = self.single_file
        # a set of one file is the index with the pieces (no split keys); otherwise the index is metadata only and every
        # piece has a file of its own
        w = gguf.GGUFWriter(None, gguf.MODEL_ARCH_NAMES[model.model_arch], small_first_shard=not single,
                            resume=model.resume_shards, resume_check=model.resume_check)
        logger.info("engram table set %s: %d file%s", path.name if single else nt.shard_name(base, 0, self.split_count),
                    self.split_count, "" if single else "s")
        w.add_type(K.GENERAL_TYPE)
        w.add_name(base)
        w.add_uint32(K.VERSION, nt.VERSION)
        assert self.kind is not None
        w.add_string(K.KIND, self.kind)
        w.add_string(K.TYPE, self.type_name)
        w.add_array(K.NAMES, [p.name for p in self.plans])
        w.add_key_value(K.PARTS, [p.parts for p in self.plans], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
        w.add_key_value(K.ROWS, [p.rows for p in self.plans], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
        w.add_key_value(K.PART_ROWS, [p.part_rows for p in self.plans], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
        w.add_key_value(K.PIECE_CHECK, self.piece_checks(), gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)
        assert self.source_folder is not None
        w.add_string(K.SOURCE, model.engram_source_name or self.source_folder.name)
        rev = model.engram_source_rev or hf_revision(self.source_folder)
        if rev:
            w.add_string(K.SOURCE_REVISION, rev)
        first = True
        for s, p in zip(self.sources, self.plans):
            for i in range(p.parts):
                if not first and not single:
                    w.new_shard()
                r0, r1 = p.span(i)
                logger.info("engram table set: %s rows [%d, %d), %.1f GB, %s", p.piece_name(i), r0, r1,
                            (r1 - r0) * p.row_bytes / 1e9, s.qtype.name)
                w.add_tensor(p.piece_name(i), s.piece(r0, r1), raw_dtype=s.qtype)
                if first:
                    for name, arr in self.extras:
                        w.add_tensor(name, arr)
                first = False
        w.write_header_to_file(path=path)
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=True)
        w.close()
        return w.shard_paths


def chunked_reader(chunks: list[Callable[[], np.ndarray]], row_bytes: int) -> np.ndarray:
    """The rows of a list of chunk loaders, joined (small reads: the ends of the pieces for the piece checks)."""
    parts = [np.ascontiguousarray(c()).view(np.uint8).reshape(-1, row_bytes) for c in chunks]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, row_bytes), dtype=np.uint8)


# ---- the original (unquantized) engram table of a checkpoint folder: safetensors shards `<module>.ngram_embedding.shard_N` -----

CHUNK_BYTES = 64 * 2**20
_TORCH_TO_GGML = {"F32": gguf.GGMLQuantizationType.F32, "F16": gguf.GGMLQuantizationType.F16, "BF16": gguf.GGMLQuantizationType.BF16}


@dataclass
class _Shard:
    path: Path
    offset: int      # absolute file offset of the data
    rows: int


@dataclass
class OriginalTable:
    shards: list[_Shard]
    rows: int
    cols: int
    dtype: str       # safetensors spelling: F32, F16 or BF16

    @property
    def row_bytes(self) -> int:
        return self.cols * orig_dtype_bytes(self.dtype)

    @property
    def ggml_type(self) -> gguf.GGMLQuantizationType:
        return _TORCH_TO_GGML[self.dtype]

    def chunks(self, r0: int, r1: int) -> list:
        """Callables that return the rows [r0, r1) as uint8 [n, row_bytes] arrays, a bounded number of rows at a time."""
        rb = self.row_bytes
        step = max(1, CHUNK_BYTES // rb)
        out = []
        base = 0
        for sh in self.shards:
            lo, hi = max(r0, base), min(r1, base + sh.rows)
            for a in range(lo, hi, step):
                b = min(a + step, hi)
                out.append(lambda sh=sh, a=a - base, b=b - base: _read_rows(sh, rb, a, b))
            base += sh.rows
        return out


def _read_rows(sh: _Shard, row_bytes: int, a: int, b: int):
    data = np.memmap(sh.path, mode="r", dtype=np.uint8, offset=sh.offset + a * row_bytes, shape=(b - a, row_bytes))
    out = np.array(data)
    del data
    return out


def find_original_table(folder: Path) -> OriginalTable:
    """The shards `<module>.ngram_embedding.shard_N` of the *.safetensors files in `folder`, in row order."""
    import json
    import struct
    marker = ".ngram_embedding.shard_"
    found: dict[int, tuple[Path, int, str, tuple[int, ...]]] = {}
    for path in sorted(folder.glob("*.safetensors")):
        with open(path, "rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, e in header.items():
            if marker not in name:
                continue
            tail = name.split(marker, 1)[1].split(".")[0]
            if not tail.isdigit():
                raise ValueError(f"n-gram shard tensor {name!r} in {path.name}")
            if int(tail) in found:
                raise ValueError(f"n-gram shard {tail} is in both {found[int(tail)][0].name} and {path.name}")
            found[int(tail)] = (path, 8 + n + e["data_offsets"][0], e["dtype"], tuple(e["shape"]))
            if e["data_offsets"][1] - e["data_offsets"][0] != int(np.prod(e["shape"])) * orig_dtype_bytes(e["dtype"]):
                raise ValueError(f"{name}: byte size does not match shape {e['shape']} {e['dtype']}")
    if not found:
        raise ValueError(f"no `{marker}N` n-gram table shards in the safetensors files of {folder}")
    if sorted(found) != list(range(len(found))):
        raise ValueError(f"n-gram table shards {sorted(found)} of {folder} are not 0..{len(found) - 1}")
    first = found[0]
    shards = []
    for i in range(len(found)):
        path, off, dtype, shape = found[i]
        if dtype != first[2] or len(shape) != 2 or shape[1] != first[3][1]:
            raise ValueError(f"n-gram table shard {i}: {dtype} {list(shape)} differs from shard 0 {first[2]} {list(first[3])}")
        shards.append(_Shard(path, off, shape[0]))
    # rows map to shards by a plain division (official NGramEmbedding.load): equal shards, the last may be short
    if any(s.rows != shards[0].rows for s in shards[:-1]) or shards[-1].rows > shards[0].rows:
        raise ValueError(f"n-gram table shard row counts {[s.rows for s in shards]}")
    for s in shards:
        if s.offset + s.rows * first[3][1] * orig_dtype_bytes(first[2]) > s.path.stat().st_size:
            raise ValueError(f"{s.path.name} is shorter than its header says (incomplete download?)")
    return OriginalTable(shards, sum(s.rows for s in shards), first[3][1], first[2])


def original_source(name: str, src: OriginalTable) -> TableSource:
    """The original table of a checkpoint as a table source, in its own dtype (no bias: the loader reads one for an EXL3
    row codec table only)."""
    def piece(r0: int, r1: int):
        return gguf.LazyChunkedTensor(src.chunks(r0, r1), shape=(r1 - r0, src.row_bytes), dtype=np.uint8)
    return TableSource(name, src.rows, src.row_bytes, src.row_bytes, src.ggml_type,
                       lambda r0, r1: chunked_reader(src.chunks(r0, r1), src.row_bytes), piece)


def config_dtype(hparams: dict[str, Any], override: str | None) -> str:
    """Dtype of the original table when no checkpoint of it is at hand: --engram-orig-dtype, else the dtype of the config."""
    if override:
        return {"bf16": "BF16", "f16": "F16", "f32": "F32"}[override]
    for d in (hparams, hparams.get("text_config") or {}):
        for key in ("dtype", "torch_dtype"):
            if d.get(key):
                return str(d[key])
    raise ValueError("the dtype of the original engram table is unknown: pass --engram-source (the original checkpoint) "
                     "or --engram-orig-dtype")
