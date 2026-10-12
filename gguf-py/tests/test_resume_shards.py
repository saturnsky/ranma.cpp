#!/usr/bin/env python3

# Test GGUFWriter shard resume: a split set written in pieces (interrupted, then continued with resume=True) must equal
# the set written in one go, byte for byte; complete shards are not rewritten and their tensors are not computed.
# Pure Python, fake data, no model.

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf  # noqa: E402
from gguf.gguf_writer import GGUFWriter  # noqa: E402


class Boom(RuntimeError):
    pass


class Fake:
    """A tensor that only exists as a recipe: computing it (tofile) is recorded, and can be made to fail."""
    computed: list[str] = []

    def __init__(self, name: str, n: int, fail: bool = False):
        self.name, self.n, self.fail = name, n, fail
        self.dtype = np.dtype(np.int8)
        self.shape = (n,)
        self.nbytes = n

    def tofile(self, f) -> None:
        if self.fail:
            raise Boom(self.name)
        Fake.computed.append(self.name)
        f.write(np.random.default_rng(sum(self.name.encode())).integers(0, 256, self.n, dtype=np.uint8).tobytes())

    def byteswap(self, inplace=False):
        return self


WEIGHTS = [(f"w{i}", 40 + 13 * i) for i in range(9)]   # packed into several shards by the size limit
PIECES = [("t.p0", 70), ("t.bias", 6), ("t.p1", 90), ("t.p2", 33)]  # one shard each by new_shard(), the bias next to p0
SPLIT = 150


def build(out: Path, resume: bool = False, check: bool = False, fail: str | None = None,
          description: str = "d") -> GGUFWriter:
    w = GGUFWriter(None, "llama", split_max_size=SPLIT, small_first_shard=True, resume=resume, resume_check=check)
    w.add_description(description)
    w.add_block_count(3)
    for name, n in WEIGHTS:
        w.add_tensor(name, Fake(name, n, fail=(name == fail)))   # type: ignore[arg-type]
    for name, n in PIECES:
        if name != "t.bias":
            w.new_shard()
        w.add_tensor(name, Fake(name, n, fail=(name == fail)))   # type: ignore[arg-type]
    w.write_header_to_file(path=out)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return w


def snapshot(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Fake.computed = []

    def tearDown(self):
        self.tmp.cleanup()

    def folder(self, name: str) -> Path:
        d = self.root / name
        d.mkdir()
        return d

    def reference(self) -> dict[str, bytes]:
        d = self.folder("ref")
        build(d / "m.gguf")
        return snapshot(d)

    def test_interrupted_all(self):
        ref = self.reference()
        self.assertGreater(len(ref), 5)
        for fail in ("w5", "t.p1"):
            d = self.folder("run_" + fail)
            with self.assertRaises(Boom):
                build(d / "m.gguf", fail=fail)
            part = snapshot(d)
            self.assertTrue(any(n.endswith(".part") for n in part), "an unfinished shard keeps the .part name")
            done = {n: (d / n).stat().st_mtime_ns for n in part if not n.endswith(".part")}
            self.assertGreater(len(done), 0)
            for n in done:
                self.assertEqual(part[n], ref[n], "a shard under its final name is complete")
            Fake.computed = []
            build(d / "m.gguf", resume=True)
            self.assertEqual(snapshot(d), ref)
            for n, t in done.items():
                self.assertEqual((d / n).stat().st_mtime_ns, t, f"{n} was rewritten")
            # no tensor of a kept shard was computed
            kept = set()
            r = gguf.GGUFReader  # noqa: F841
            for n in done:
                rd = gguf.GGUFReader(d / n, "r")
                kept |= {t.name for t in rd.tensors}
                del rd
            self.assertFalse(kept & set(Fake.computed), "tensors of kept shards were computed")

    def test_old_run_without_part(self):
        """A truncated file under the final name (earlier runs wrote it directly) is rewritten, complete ones are kept."""
        ref = self.reference()
        d = self.folder("old")
        for n, b in ref.items():
            (d / n).write_bytes(b)
        names = sorted(ref)
        cut = names[4]
        (d / cut).write_bytes(ref[cut][: len(ref[cut]) // 2])
        for n in names[5:]:
            (d / n).unlink()
        m0 = {n: (d / n).stat().st_mtime_ns for n in names[:4]}
        build(d / "m.gguf", resume=True)
        self.assertEqual(snapshot(d), ref)
        for n, t in m0.items():
            self.assertEqual((d / n).stat().st_mtime_ns, t)

    def test_same_size_other_header(self):
        ref = self.reference()
        d = self.folder("hdr")
        for n, b in ref.items():
            (d / n).write_bytes(b)
        names = sorted(ref)
        bad = names[2]
        b = bytearray(ref[bad])
        i = b.index(b"w") if b"w" in b[:200] else 20      # inside the tensor info: the name of the first tensor
        b[i] ^= 1
        self.assertEqual(len(b), len(ref[bad]))
        (d / bad).write_bytes(bytes(b))
        before = {n: (d / n).stat().st_mtime_ns for n in names if n != bad}
        build(d / "m.gguf", resume=True)
        self.assertEqual(snapshot(d), ref)
        for n, t in before.items():
            self.assertEqual((d / n).stat().st_mtime_ns, t, f"{n} was rewritten")

    def test_leftover_part(self):
        ref = self.reference()
        d = self.folder("part")
        for n, b in ref.items():
            (d / n).write_bytes(b)
        names = sorted(ref)
        (d / (names[3] + ".part")).write_bytes(b"junk")       # stale .part next to a complete shard: removed, shard kept
        gone = names[4]
        (d / gone).unlink()
        (d / (gone + ".part")).write_bytes(b"junk" * 1000)    # .part of a missing shard: removed, shard written
        build(d / "m.gguf", resume=True)
        self.assertEqual(snapshot(d), ref)

    def test_no_resume_overwrites(self):
        ref = self.reference()
        d = self.folder("plain")
        build(d / "m.gguf")
        (d / sorted(ref)[1]).write_bytes(b"x")
        build(d / "m.gguf")
        self.assertEqual(snapshot(d), ref)

    def test_new_shard_layout(self):
        """new_shard() puts each piece in a shard of its own, after the weight shards; the bias stays with p0."""
        ref = self.reference()
        names = sorted(ref)
        tensors = []
        for n in names:
            r = gguf.GGUFReader(self.root / "ref" / n, "r")
            tensors.append([t.name for t in r.tensors])
            del r
        self.assertEqual(tensors[0], [])
        self.assertEqual(tensors[-3:], [["t.p0", "t.bias"], ["t.p1"], ["t.p2"]])

    def test_index_changed(self):
        """Other metadata in shard 1: the index is rewritten, the weight shards stay."""
        d = self.folder("idx")
        build(d / "m.gguf", description="old")
        mt = {n: (d / n).stat().st_mtime_ns for n in sorted(os.listdir(d))}
        build(d / "m.gguf", resume=True, description="new")
        ref_new = self.folder("ref_new")
        build(ref_new / "m.gguf", description="new")
        self.assertEqual(snapshot(d), snapshot(ref_new))
        names = sorted(mt)
        self.assertNotEqual((d / names[0]).stat().st_mtime_ns, mt[names[0]])
        for n in names[1:]:
            self.assertEqual((d / n).stat().st_mtime_ns, mt[n])

    def test_check_writes_nothing(self):
        ref = self.reference()
        d = self.folder("chk")
        for n, b in ref.items():
            (d / n).write_bytes(b)
        names = sorted(ref)
        (d / names[3]).write_bytes(ref[names[3]][:-1])
        Fake.computed = []
        before = {n: ((d / n).stat().st_mtime_ns, (d / n).stat().st_size) for n in names}
        w = build(d / "m.gguf", check=True)
        self.assertEqual(w.shard_skipped, [n != names[3] for n in names])
        self.assertEqual(sorted(os.listdir(d)), names)
        self.assertEqual({n: ((d / n).stat().st_mtime_ns, (d / n).stat().st_size) for n in names}, before)
        self.assertEqual(Fake.computed, [])


if __name__ == "__main__":
    unittest.main()
