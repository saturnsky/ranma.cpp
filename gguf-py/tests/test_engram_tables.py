#!/usr/bin/env python3

# Engram tables in a GGUF set of their own (docs/ranma/engram-tables.md), with fake data and no model:
#  - the converter's table set (conversion/engram_tables.py) and the model index keys pass the checks of the verifier
#  - the set is named after the model (<P>-engram...), one file when all pieces fit the split size
#  - the set is found as the loader finds it: the one <P>-engram*.gguf next to the model (none / two: not found)
#  - an interrupted table set is completed by resume, byte for byte
#  - the checks catch a piece of another set, a model index of other tables and a model set that still holds a table

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
from gguf import engram_tables as nt  # noqa: E402
from gguf.gguf_writer import GGUFWriter  # noqa: E402


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def conversion_tables():
    sys.path.insert(0, str(ROOT))
    import conversion.engram_tables as ct  # noqa: E402
    return ct


MX = gguf.GGMLQuantizationType.MXFP8
ROW = 264                                        # an MXFP8 row (256 values)


class Boom(RuntimeError):
    pass


class Piece:
    """uint8 rows for GGUFWriter.add_tensor; can be made to fail while it is written."""

    def __init__(self, data: np.ndarray, fail: bool = False):
        self.data, self.fail = data, fail
        self.dtype, self.shape, self.nbytes = data.dtype, data.shape, data.nbytes

    def tofile(self, f):
        if self.fail:
            raise Boom()
        self.data.tofile(f)

    def byteswap(self, inplace=False):
        return self


def engram_tables(seed: int, rows=(1000, 1210)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (r, ROW), dtype=np.uint8) for r in rows]


def fake_model(out: Path, limit: int, **kw) -> SimpleNamespace:
    m = SimpleNamespace(engram_stage="all", split_max_size=limit, engram_parts=None, engram_tables_out=None, fname_out=out,
                        model_arch=gguf.MODEL_ARCH.DEEPSEEK41, resume_shards=False, resume_check=False,
                        engram_source_name=None, engram_source_rev=None)
    m.__dict__.update(kw)
    return m


LIMIT = 100_000      # 1000 x 264 B = 264 KB -> 3 pieces of 334 rows; 1210 rows -> 4 pieces


def converter_tables(ct, folder: Path, tables: list[np.ndarray], fail_piece: int | None = None, limit: int = LIMIT,
                     out: str = "m.gguf", **kw):
    """The converter's EngramTables for fake engram tables of layers 1 and 14 (source folder 'Fake-Model'), model output
    <folder>/<out>."""
    model = fake_model(folder / out, limit, **kw)
    t = ct.EngramTables(model)
    calls = {"n": 0}

    def piece_of(a):
        def piece(r0, r1):
            k = calls["n"]
            calls["n"] += 1
            return Piece(a[r0:r1], fail=(k == fail_piece))
        return piece
    sources = [ct.TableSource(f"blk.{bid}.engram_embd", a.shape[0], ROW, ROW, MX, (lambda a: lambda r0, r1: a[r0:r1])(a), piece_of(a))
               for bid, a in zip((1, 14), tables)]
    t.set_tables("engram", sources, folder / "Fake-Model")
    return t


def write_model_set(out: Path, extra_kv=None, table_tensors: list[tuple[str, np.ndarray]] | None = None,
                    tables=None, arch: str = "deepseek41", raw_dtype=MX) -> GGUFWriter:
    """A small model set: index (metadata only) + weight shards (+ table tensors after them: a set that still holds a table)."""
    w = GGUFWriter(None, arch, split_max_size=10_000, small_first_shard=True)
    w.add_name("fake")
    w.add_string("general.base_model.0.repo_url", "https://huggingface.co/org/Fake-Model")
    if extra_kv:
        extra_kv(w)
    if tables is not None:
        tables.write_index_keys(w)
    rng = np.random.default_rng(7)
    for i in range(7):
        w.add_tensor(f"blk.{i}.w", rng.standard_normal(300 + 50 * i).astype(np.float32))
    w.split_max_size = 0          # the old converter placed the pieces itself: one per shard, the bias next to piece 0
    for name, data in table_tensors or []:
        if not name.endswith(".bias"):
            w.new_shard()
        w.add_tensor(name, data, raw_dtype=raw_dtype if data.dtype == np.uint8 else None)
    w.write_header_to_file(path=out)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return w


def snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir()) if p.is_file()}


class EngramTablesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ct = conversion_tables()

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, name: str) -> Path:
        d = self.root / name
        d.mkdir()
        return d

    def test_fnv(self):
        self.assertEqual(nt.fnv1a64(b"a"), 0xaf63dc4c8601ec8c)
        self.assertEqual(nt.fnv1a64(b""), 0xcbf29ce484222325)
        self.assertEqual(nt.check_rows(1000, 264), 16)

    def converted(self, name="conv", **kw) -> tuple[Path, Path]:
        d = self.folder(name)
        tables = engram_tables(1)
        t = converter_tables(self.ct, d, tables, engram_source_name="org/Fake-Model", **kw)
        self.assertEqual([(p.parts, p.part_rows) for p in t.plans], [(3, 334), (4, 303)])
        write_model_set(d / "m.gguf", tables=t)
        files = t.write_table_set()
        self.assertEqual(files[0].name, "m-engram-00001-of-00008.gguf")
        return d / "m-00001-of-00003.gguf", files[0]

    def test_converter_set(self):
        model, first = self.converted()
        ts = nt.open_table_set(first)
        self.assertEqual(nt.check_table_set(ts), [])
        self.assertEqual(nt.check_model_index(model, ts), [])
        self.assertEqual(ts.source, "org/Fake-Model")
        # the pieces hold the rows of the tables
        tables = engram_tables(1)
        for p in ts.pieces:
            t = ts.tables[p.table]
            r0, r1 = t.span(p.index)
            with open(p.path, "rb") as f:
                f.seek(p.offset)
                self.assertEqual(f.read(p.nbytes), tables[p.table][r0:r1].tobytes())
        # one piece per file after the index, the index holds no tensor
        r = gguf.GGUFReader(first, "r")
        self.assertEqual(len(r.tensors), 0)
        del r

    def test_tables_out(self):
        d = self.folder("out")
        t = converter_tables(self.ct, d, engram_tables(1), engram_tables_out=d / "tables")   # a folder that does not exist yet
        self.assertEqual(t.write_table_set()[0], d / "tables" / "m-engram-00001-of-00008.gguf")
        t = converter_tables(self.ct, d, engram_tables(1), engram_tables_out=d / "x" / "T-engram.gguf")
        self.assertEqual(t.base_name, "T-engram")
        self.assertEqual(t.write_table_set()[-1], d / "x" / "T-engram-00008-of-00008.gguf")

    def test_model_index_keys(self):
        """The model index lists the tables (names, rows, row widths) and names no file, type or pieces."""
        model, _ = self.converted("keys")
        r = gguf.GGUFReader(model, "r")
        keys = sorted(k for k in r.fields if k.startswith("engram_tables."))
        self.assertEqual(keys, ["engram_tables.names", "engram_tables.rows", "engram_tables.width"])
        self.assertEqual([int(x) for x in r.get_field("engram_tables.width").contents()], [256, 256])
        del r

    def test_single_file_set(self):
        """All pieces within the split size: one file <P>-engram.gguf, the index with the pieces and no split keys."""
        d = self.folder("single")
        tables = engram_tables(1)
        t = converter_tables(self.ct, d, tables, limit=1_000_000, out="Model-Q3.gguf", engram_source_name="org/Fake-Model")
        self.assertEqual([(p.parts, p.part_rows) for p in t.plans], [(1, 1000), (1, 1210)])
        write_model_set(d / "Model-Q3.gguf", tables=t)
        files = t.write_table_set()
        self.assertEqual(files, [d / "Model-Q3-engram.gguf"])
        r = gguf.GGUFReader(files[0], "r")
        self.assertEqual(len(r.tensors), 2)
        self.assertIsNone(r.get_field("split.count"))
        del r
        ts = nt.open_table_set(files[0])
        self.assertEqual(nt.check_table_set(ts), [])
        self.assertEqual(nt.check_model_index(d / "Model-Q3-00001-of-00003.gguf", ts), [])

    def test_find_table_set(self):
        """The one <P>-engram*.gguf next to the model: a file of its own or the first file of a split set."""
        model, first = self.converted("find")
        d = first.parent
        self.assertEqual(nt.find_table_set(model), (first, ""))
        self.assertEqual(nt.candidates(d, "m"), [first])
        # another model of the same tables in the folder: its own name finds nothing
        other = d / "other-00001-of-00003.gguf"
        other.write_bytes(model.read_bytes())
        found, why = nt.find_table_set(other)
        self.assertIsNone(found)
        self.assertIn("other-engram*.gguf", why)
        # a link under its name: found
        (d / "other-engram-f16.gguf").write_bytes(b"GGUF")
        self.assertEqual(nt.find_table_set(other)[0], d / "other-engram-f16.gguf")
        # two sets of the model (a variant next to it): ambiguous
        (d / "m-engram-f16.gguf").write_bytes(b"GGUF")
        found, why = nt.find_table_set(model)
        self.assertIsNone(found)
        self.assertIn("2 table sets", why)

    def test_mxfp8_samples(self):
        """DeepSeek V4.1: the rows the piece checks read (read_rows) are the bytes the table set gets (tofile)."""
        import io
        from conversion.deepseek import _Mxfp8Table, _SafetensorsEntry
        rng = np.random.default_rng(5)
        rows = 3000
        w = rng.integers(0, 256, (rows, 256), dtype=np.uint8)
        sc = rng.integers(100, 140, (rows, 8), dtype=np.uint8)
        f = self.root / "t.bin"
        f.write_bytes(b"x" * 7 + w.tobytes() + sc.tobytes())
        table = _Mxfp8Table(_SafetensorsEntry(f, 7, w.nbytes, "F8_E4M3", (rows, 256)),
                            _SafetensorsEntry(f, 7 + w.nbytes, sc.nbytes, "F8_E8M0", (rows, 8)))
        for a, b in ((0, 1), (100, 1124), (1976, 3000)):
            buf = io.BytesIO()
            table.piece(a, b).tofile(buf)
            self.assertEqual(table.read_rows(a, b).tobytes(), buf.getvalue())
            self.assertEqual(len(buf.getvalue()), (b - a) * ROW)

    def test_resume_check_table_set(self):
        """--resume-check: reports, writes nothing, creates no folder and computes no piece."""
        _, first = self.converted("chk")
        d = first.parent
        victim = first.with_name(nt.shard_name("m-engram", 3, 8))
        victim.write_bytes(victim.read_bytes()[:-1])
        before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in d.iterdir()}
        t = converter_tables(self.ct, d, engram_tables(1), engram_source_name="org/Fake-Model", resume_shards=True,
                             resume_check=True, fail_piece=-1)
        for s in t.sources:      # any piece that is computed raises
            s.piece = lambda r0, r1, a=None: Piece(np.zeros((r1 - r0, ROW), dtype=np.uint8), fail=True)
        files = t.write_table_set()
        self.assertEqual(len(files), 8)
        self.assertEqual({p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in d.iterdir()}, before)
        t = converter_tables(self.ct, d, engram_tables(1), resume_shards=True, resume_check=True,
                             engram_tables_out=d / "not-there")
        t.write_table_set()
        self.assertFalse((d / "not-there").exists())

    def test_resume_table_set(self):
        _, first = self.converted("ref")
        ref = {n: b for n, b in snapshot(first.parent).items() if n.startswith("m-engram")}
        d = self.folder("run")
        tables = engram_tables(1)
        t = converter_tables(self.ct, d, tables, engram_source_name="org/Fake-Model", fail_piece=4)
        with self.assertRaises(Boom):
            t.write_table_set()
        self.assertTrue(any(n.endswith(".part") for n in snapshot(d)))
        t = converter_tables(self.ct, d, tables, engram_source_name="org/Fake-Model", resume_shards=True)
        t.write_table_set()
        self.assertEqual(snapshot(d), ref)

    def test_checks_catch(self):
        model, first = self.converted("a")
        d2 = self.folder("b")
        t2 = converter_tables(self.ct, d2, engram_tables(2), engram_source_name="org/Fake-Model")
        files2 = t2.write_table_set()
        # a piece of another set under a name of this one
        victim = first.with_name(nt.shard_name("m-engram", 2, 8))
        victim.write_bytes(files2[2].read_bytes())
        ts = nt.open_table_set(first)
        problems = nt.check_table_set(ts)
        self.assertTrue(any("piece_check" in p for p in problems), problems)
        # a set of the same structure with other contents fits the model index (the structure is what is compared)
        ts2 = nt.open_table_set(files2[0])
        self.assertEqual(nt.check_model_index(model, ts2), [])
        # a model index of other tables: rows differ
        d3 = self.folder("c")
        t3 = converter_tables(self.ct, d3, engram_tables(1, rows=(1000, 1200)), engram_source_name="org/Fake-Model")
        write_model_set(d3 / "m.gguf", tables=t3)
        self.assertTrue(any("model index tables" in p for p in nt.check_model_index(d3 / "m-00001-of-00003.gguf", ts2)))

    def test_verify_script(self):
        model, first = self.converted()
        ver = load_script("verify_engram_tables")
        argv = sys.argv
        try:
            sys.argv = ["verify", str(first), "--model", str(model)]
            self.assertEqual(ver.main(), 0)
            # a model set that still holds a table tensor
            d = self.folder("held")
            t = converter_tables(self.ct, d, engram_tables(1), engram_source_name="org/Fake-Model")
            write_model_set(d / "m.gguf", tables=t, table_tensors=[("blk.1.engram_embd.weight", engram_tables(1)[0])])
            sys.argv = ["verify", str(first), "--model", str(d / "m-00001-of-00004.gguf")]
            self.assertEqual(ver.main(), 1)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
