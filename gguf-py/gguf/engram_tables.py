"""Engram tables (DeepSeek V4.1 engram, Qwen3.8 PLE n-gram) in a GGUF set of their own.

A table set is `<S>-00001-of-0000M.gguf` .. `<S>-0000M-of-0000M.gguf`: file 1 is the index (metadata only), every other
file holds one piece `<table>.p<i>.weight` of a table (the per-head bias of a Qwen3.8 EXL3 row codec table sits next to piece 0
of its table); or one file `<S>.gguf`, the index with all pieces (no split keys). For the model `<P>-00001-of-0000K.gguf` /
`<P>.gguf`, `<S>` starts with `<P>-engram` (the converter writes `<P>-engram`; variants such as `<P>-engram-f16` are found
too). A model that uses the set has no table tensor; its index lists the tables (names, rows, row widths) and names no file.
The loader takes the one set `<P>-engram*.gguf` next to the model (find_table_set), or the file or folder given with
--engram-tables. A set fits a model by its structure: the architecture, the table names, the rows and the row widths; the
type and the pieces may differ. See docs/ranma/engram-tables.md.

This module holds what the converter and the verifier (scripts/verify_engram_tables.py) share: the piece check and the
reading of a table set.

Piece check: FNV-1a 64 of the first min(4096, n) bytes followed by the last min(4096, n) bytes of the data of a piece (n bytes).
The loader computes it for every piece when it attaches a set, so a piece file of another set or of another run of the
converter, put under a name of this set, stops the load. It is no content hash: the bytes between the ends are not covered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .constants import GGMLQuantizationType, Keys

CHECK_BYTES = 4096
VERSION = 1
KINDS = ("engram", "ple")

_FNV_OFFSET = 0xcbf29ce484222325
_FNV_PRIME = 0x100000001b3
_MASK64 = (1 << 64) - 1

SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
SET_SUFFIX = "-engram"


def model_prefix(path: Path) -> str:
    """<P> of a model file <P>-0000N-of-0000K.gguf or <P>.gguf."""
    m = SHARD_RE.match(path.name)
    if m:
        return m.group(1)
    return path.name[:-len(".gguf")] if path.name.endswith(".gguf") else path.name


def set_base(prefix: str) -> str:
    """The name <S> the converter gives the table set of the model <P>: <P>-engram."""
    return prefix + SET_SUFFIX


def shard_name(base: str, i: int, n: int) -> str:
    """File name of file i (0-based) of n of the set `base`."""
    return f"{base}-{i + 1:05d}-of-{n:05d}.gguf"


def split_base(path: Path) -> tuple[str, int]:
    """(base name, file count) from the name of a file of a split set."""
    m = SHARD_RE.match(path.name)
    if not m:
        raise ValueError(f"{path.name} is not a file of a split set (<base>-0000N-of-0000M.gguf)")
    return m.group(1), int(m.group(3))


def set_files(first: Path, n: int) -> list[Path]:
    """The files of a set of n files from its first file (n == 1: the file itself)."""
    if n == 1:
        return [first]
    base, count = split_base(first)
    if count != n:
        raise ValueError(f"{first.name}: the name says {count} files, the index {n}")
    return [first.with_name(shard_name(base, i, n)) for i in range(n)]


def candidates(folder: Path, prefix: str) -> list[Path]:
    """The table sets of the model <prefix> in `folder`, one per set, as the loader counts them: <P>-engram*.gguf that is a
    file of its own (no -0000N-of-0000M in the name) or the first file of a split set; sorted by name."""
    head = set_base(prefix)
    out = []
    for p in folder.iterdir():
        if not (p.is_file() and p.name.startswith(head) and p.name.endswith(".gguf")):
            continue
        m = SHARD_RE.match(p.name)
        if m and m.group(2) != "00001":
            continue
        out.append(p)
    return sorted(out)


@dataclass(frozen=True)
class TableDesc:
    name: str        # base tensor name, e.g. blk.1.engram_embd
    rows: int
    parts: int
    part_rows: int
    row_bytes: int   # stored bytes per row
    width: int = 0   # values per row (ne0 of a piece)

    def span(self, i: int) -> tuple[int, int]:
        return i * self.part_rows, min((i + 1) * self.part_rows, self.rows)

    def piece_name(self, i: int) -> str:
        return f"{self.name}.p{i}.weight"


def fnv1a64(data: bytes, h: int = _FNV_OFFSET) -> int:
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


def check_rows(n_rows: int, row_bytes: int) -> int:
    """Rows to read at each end of a piece to have the bytes of its piece check."""
    return min(n_rows, -(-CHECK_BYTES // row_bytes))


def piece_check(head_rows: bytes, tail_rows: bytes, nbytes: int) -> int:
    """The piece check from the bytes of the first and the last check_rows() rows of a piece of nbytes bytes."""
    k = min(CHECK_BYTES, nbytes)
    head, tail = head_rows[:k], tail_rows[len(tail_rows) - k:]
    if len(head) != k or len(tail) != k:
        raise ValueError(f"piece check: {len(head_rows)} / {len(tail_rows)} bytes for the ends of a piece of {nbytes} bytes")
    return fnv1a64(tail, fnv1a64(head))


RowReader = Callable[[int, int, int], bytes]   # (table index, first row, end row) -> stored bytes of those rows


def piece_checks(tables: Sequence[TableDesc], read_rows: RowReader) -> list[int]:
    out = []
    for ti, t in enumerate(tables):
        for i in range(t.parts):
            r0, r1 = t.span(i)
            k = check_rows(r1 - r0, t.row_bytes)
            out.append(piece_check(read_rows(ti, r0, r0 + k), read_rows(ti, r1 - k, r1), (r1 - r0) * t.row_bytes))
    return out


# ---- reading a table set (index + pieces) ---------------------------------------------------------------------------

def _field(reader, key: str):
    f = reader.get_field(key)
    return None if f is None else f.contents()


@dataclass
class Piece:
    table: int
    index: int
    name: str
    path: Path
    offset: int      # absolute file offset of the data
    nbytes: int
    ggml_type: int
    rows: int


@dataclass
class TableSet:
    first: Path
    files: list[Path]
    arch: str
    kind: str
    type_name: str
    tables: list[TableDesc]
    piece_check: list[int]
    source: str | None
    source_revision: str | None
    version: int | None
    pieces: list[Piece] = field(default_factory=list)
    extras: list[tuple[str, Path, int, int]] = field(default_factory=list)   # (name, path, offset, nbytes)
    problems: list[str] = field(default_factory=list)

    def read_rows(self, ti: int, a: int, b: int) -> bytes:
        """Stored bytes of the rows [a, b) of table ti, which must lie in one piece."""
        t = self.tables[ti]
        i = a // t.part_rows
        r0, r1 = t.span(i)
        if not r0 <= a < b <= r1:
            raise ValueError(f"rows [{a}, {b}) of {t.name} are not inside one piece")
        p = next(p for p in self.pieces if p.table == ti and p.index == i)
        with open(p.path, "rb") as f:
            f.seek(p.offset + (a - r0) * t.row_bytes)
            return f.read((b - a) * t.row_bytes)

    def extra_bytes(self) -> list[tuple[str, bytes]]:
        out = []
        for name, path, off, n in self.extras:
            with open(path, "rb") as f:
                f.seek(off)
                out.append((name, f.read(n)))
        return out


def structure_from_index(reader) -> tuple[list[str], list[int], list[int], list[int]] | None:
    names = _field(reader, Keys.EngramTables.NAMES)
    if names is None:
        return None
    return ([str(x) for x in names], [int(x) for x in _field(reader, Keys.EngramTables.PARTS)],
            [int(x) for x in _field(reader, Keys.EngramTables.ROWS)], [int(x) for x in _field(reader, Keys.EngramTables.PART_ROWS)])


def open_table_set(first: Path) -> TableSet:
    """The index and the pieces of a table set, from its first file. Layout problems are collected in .problems."""
    from .gguf_reader import GGUFReader

    problems: list[str] = []
    r0 = GGUFReader(first, "r")
    n = _field(r0, Keys.Split.LLM_KV_SPLIT_COUNT) or 1
    files = set_files(first, n)
    if _field(r0, Keys.General.TYPE) != Keys.EngramTables.GENERAL_TYPE:
        problems.append(f"{first.name}: general.type is {_field(r0, Keys.General.TYPE)!r}, not {Keys.EngramTables.GENERAL_TYPE!r}")
    st = structure_from_index(r0)
    if st is None:
        raise ValueError(f"{first.name} has no {Keys.EngramTables.NAMES}: not the index of an engram table set")
    names, parts, rows, part_rows = st
    if n > 1 and r0.tensors:
        problems.append(f"{first.name}: the index of a set of {n} files holds {len(r0.tensors)} tensors")
    pc = _field(r0, Keys.EngramTables.PIECE_CHECK)
    ts = TableSet(first, files, str(_field(r0, Keys.General.ARCHITECTURE)), str(_field(r0, Keys.EngramTables.KIND)),
                  str(_field(r0, Keys.EngramTables.TYPE)), [], [int(x) for x in pc] if pc is not None else [],
                  _field(r0, Keys.EngramTables.SOURCE), _field(r0, Keys.EngramTables.SOURCE_REVISION),
                  _field(r0, Keys.EngramTables.VERSION), problems=problems)
    total = _field(r0, Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT)
    del r0

    found: dict[str, tuple[Path, int, int, int, list[int]]] = {}
    n_tensors = 0
    for i, path in enumerate(files):
        if not path.is_file():
            problems.append(f"{path.name} is missing")
            continue
        r = GGUFReader(path, "r")
        if n > 1 and (_field(r, Keys.Split.LLM_KV_SPLIT_NO) != i or _field(r, Keys.Split.LLM_KV_SPLIT_COUNT) != n):
            problems.append(f"{path.name}: split.no/count {_field(r, Keys.Split.LLM_KV_SPLIT_NO)}/{_field(r, Keys.Split.LLM_KV_SPLIT_COUNT)}, "
                            f"expected {i}/{n}")
        for t in r.tensors:
            n_tensors += 1
            found[t.name] = (path, int(t.data_offset), int(t.n_bytes), int(t.tensor_type), [int(x) for x in t.shape])
        del r
    if n > 1 and total != n_tensors:
        problems.append(f"split.tensors.count {total}, the files hold {n_tensors}")

    used = set()
    for ti, name in enumerate(names):
        info = [found.get(f"{name}.p{i}.weight") for i in range(parts[ti])]
        if any(x is None for x in info):
            problems.append(f"{name}: pieces {[i for i, x in enumerate(info) if x is None]} missing")
            ts.tables.append(TableDesc(name, rows[ti], parts[ti], part_rows[ti], 0))
            continue
        row_bytes = info[0][2] // max(1, info[0][4][-1])
        t = TableDesc(name, rows[ti], parts[ti], part_rows[ti], row_bytes, info[0][4][0])
        ts.tables.append(t)
        for i, (path, off, nbytes, gtype, shape) in enumerate(info):  # type: ignore[misc]
            r_a, r_b = t.span(i)
            used.add(t.piece_name(i))
            if shape[-1] != r_b - r_a or nbytes != (r_b - r_a) * row_bytes or shape[0] != t.width:
                problems.append(f"{t.piece_name(i)}: {shape} / {nbytes} bytes, the index implies {r_b - r_a} rows of {t.width} "
                                f"values, {row_bytes} bytes")
            if GGMLQuantizationType(gtype).name != ts.type_name:
                problems.append(f"{t.piece_name(i)} is {GGMLQuantizationType(gtype).name}, the index says {ts.type_name}")
            ts.pieces.append(Piece(ti, i, t.piece_name(i), path, off, nbytes, gtype, r_b - r_a))
    for name, (path, off, nbytes, _, _) in found.items():
        if name not in used:
            ts.extras.append((name, path, off, nbytes))
    return ts


def recompute(ts: TableSet) -> list[int]:
    """The piece checks of a table set computed from its data (the piece ends)."""
    return piece_checks(ts.tables, ts.read_rows)


def check_table_set(ts: TableSet) -> list[str]:
    """What the loader checks of a set, plus the layout (one piece per file, in order, for a split set)."""
    problems = list(ts.problems)
    if ts.kind not in KINDS:
        problems.append(f"{Keys.EngramTables.KIND} is {ts.kind!r}")
    order = [(p.table, p.index) for p in ts.pieces]
    if order != sorted(order):
        problems.append("the pieces are not in table / piece order")
    if len(ts.files) > 1:
        for k, p in enumerate(ts.pieces):
            if k + 1 < len(ts.files) and p.path != ts.files[k + 1]:
                problems.append(f"{p.name} is in {p.path.name}, expected {ts.files[k + 1].name}")
        if len(ts.files) != 1 + sum(t.parts for t in ts.tables):
            problems.append(f"{len(ts.files)} files for {sum(t.parts for t in ts.tables)} pieces + the index")
    for name, path, _, _ in ts.extras:
        if ts.pieces and path != ts.pieces[0].path:
            problems.append(f"{name} is in {path.name}, not next to the first piece")
    if problems:
        return problems
    checks = recompute(ts)
    if checks != ts.piece_check:
        bad = [ts.pieces[i].name for i, (a, b) in enumerate(zip(checks, ts.piece_check)) if a != b]
        problems.append(f"{Keys.EngramTables.PIECE_CHECK} differs for {bad or 'the count of pieces'}")
    return problems


def check_model_index(model_first: Path, ts: TableSet) -> list[str]:
    """The keys of a model index against a table set: what the loader compares (architecture, table names, rows, row
    widths), and that no file of the model set holds a table tensor."""
    from .gguf_reader import GGUFReader

    K = Keys.EngramTables
    problems = []
    r = GGUFReader(model_first, "r")
    want = {K.NAMES: [t.name for t in ts.tables], K.ROWS: [t.rows for t in ts.tables], K.WIDTH: [t.width for t in ts.tables]}
    got = {}
    for key in want:
        v = _field(r, key)
        got[key] = [x if isinstance(x, str) else int(x) for x in v] if isinstance(v, (list, tuple)) else v
    if got[K.NAMES] is None:
        problems.append(f"model index has no {K.NAMES}: the model does not use a table set")
    else:
        mine = {n: (rows, w) for n, rows, w in zip(got[K.NAMES], got[K.ROWS] or [], got[K.WIDTH] or [])}
        theirs = {t.name: (t.rows, t.width) for t in ts.tables}
        if mine != theirs:
            problems.append(f"model index tables (name: rows, width) {mine}, the table set has {theirs}")
    if _field(r, Keys.General.ARCHITECTURE) != ts.arch:
        problems.append(f"model architecture {_field(r, Keys.General.ARCHITECTURE)!r}, table set {ts.arch!r}")
    count = _field(r, Keys.Split.LLM_KV_SPLIT_COUNT) or 1
    del r
    files = [model_first] if count == 1 else [model_first.with_name(shard_name(split_base(model_first)[0], i, count))
                                               for i in range(count)]
    table_names = tuple(t.name + "." for t in ts.tables)
    for f in files:
        if not f.is_file():
            problems.append(f"model file {f.name} is missing")
            continue
        rf = GGUFReader(f, "r")
        held = [t.name for t in rf.tensors if t.name.startswith(table_names)]
        if held:
            problems.append(f"{f.name} holds table tensors {held[:3]}: the model set must not contain the tables")
        del rf
    return problems


def find_table_set(model_first: Path, folder: Path | None = None) -> tuple[Path | None, str]:
    """The table set of a model as the loader finds it in `folder` (default: the folder of the model): the one set
    <P>-engram*.gguf there. Returns (first file, "") or (None, why)."""
    folder = folder or model_first.parent
    prefix = model_prefix(model_first)
    found = candidates(folder, prefix)
    if not found:
        return None, f"{folder} holds no {set_base(prefix)}*.gguf"
    if len(found) > 1:
        return None, f"{folder} holds {len(found)} table sets of the model: {[p.name for p in found]}; choose one"
    return found[0], ""
