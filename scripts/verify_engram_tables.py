#!/usr/bin/env python3
"""Check an engram table set (DeepSeek V4.1 engram, Qwen3.8 PLE n-gram) and the model that uses it (read-only).

A table set is <P>-engram*-00001-of-0000M.gguf (the index) and one piece <table>.p<i>.weight per further file, or one file
<P>-engram*.gguf with the index and all pieces (docs/ranma/engram-tables.md). Checks, exit 0 only if all hold:
  1. the set: general.type, split.no / split.count / split.tensors.count, one piece per file in table / piece order (a set
     of several files), the piece shapes against the structure keys of the index, one type for all pieces;
  2. the piece checks (engram_tables.piece_check) recomputed from the data (the ends of every piece);
  3. with --model MODEL-00001-of-0000K.gguf: what the loader compares (architecture, table names, rows, row widths) and
     that no file of the model set holds a table tensor; it also says whether the loader takes the set next to the model by
     itself (the one <P>-engram*.gguf there);
  4. with --source DIR --source-kind engram-fp8|ple-original|ple-exl3: every row of every piece equals the source, byte for
     byte (reads the whole set and the source: about 200 GB for DeepSeek V4.1), and the PLE bias (EXL3 row codec) equals the
     head_bias of the source.

Usage:
  verify_engram_tables.py TABLES-00001-of-0000M.gguf|TABLES.gguf|FOLDER [--model MODEL-00001-of-0000K.gguf]
  (a folder, or no TABLES with --model: the one <P>-engram*.gguf in the folder / next to the model, as the loader takes it)
                         [--source DIR --source-kind engram-fp8|ple-original|ple-exl3] [--json out.json]
Needs gguf-py of a ranma.cpp checkout: set GGUF_PY=<ranma.cpp>/gguf-py (or run from the checkout). Python deps: numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
elif (Path(__file__).resolve().parent.parent / "gguf-py").is_dir():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gguf-py"))

import numpy as np  # noqa: E402
from gguf import engram_tables as nt  # noqa: E402

STEP_BYTES = 1 << 26
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2}


# ---- safetensors (header only, rows are read with memmap) --------------------------------------------------------------

def st_entries(folder, want):
    """{name: (path, absolute data offset, dtype, shape)} of the tensors for which want(name) is true."""
    out = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(folder, fn)
        with open(path, "rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        for name, e in hdr.items():
            if name != "__metadata__" and want(name):
                out[name] = (path, 8 + n + e["data_offsets"][0], e["dtype"], tuple(e["shape"]))
    return out


def st_rows(entry, row_bytes, a, b):
    path, off, _, _ = entry
    return np.array(np.memmap(path, mode="r", dtype=np.uint8, offset=off + a * row_bytes, shape=(b - a, row_bytes)))


class Source:
    """table(name) -> (rows, row_bytes, rows_fn): rows_fn(a, b) gives the stored bytes of the source rows [a, b)."""

    def __init__(self, kind, folder):
        self.kind, self.folder = kind, folder
        self.bias = None

    def table(self, name):
        if self.kind == "engram-fp8":
            bid = int(re.match(r"blk\.(\d+)\.engram_embd$", name).group(1))
            wn, sn = f"layers.{bid}.engram.embed.weight", f"layers.{bid}.engram.embed.scale"
            ents = st_entries(self.folder, lambda n: n in (wn, sn))
            w, s = ents.get(wn), ents.get(sn)
            if w is None or s is None:
                sys.exit(f"source: {wn} / .scale not found in {self.folder}")
            if w[2] != "F8_E4M3" or s[2] != "F8_E8M0":
                sys.exit(f"source: engram dtypes {w[2]}/{s[2]}")
            return w[3][0], w[3][1] + s[3][1], lambda a, b: np.concatenate([st_rows(w, w[3][1], a, b), st_rows(s, s[3][1], a, b)], axis=1)
        marker = ".ngram_embedding.shard_"
        suffix = ".trellis" if self.kind == "ple-exl3" else ""
        ents = st_entries(self.folder, lambda n: marker in n and (n.endswith(suffix) if suffix else True))
        idx = {int(n.split(marker, 1)[1].split(".")[0]): e for n, e in ents.items()}
        if not idx or sorted(idx) != list(range(len(idx))):
            sys.exit(f"source: n-gram shards {sorted(idx)} in {self.folder}")
        first = idx[0]
        eb = 2 if self.kind == "ple-exl3" else DTYPE_BYTES[first[2]]
        row_bytes = first[3][1] * eb
        bases = np.cumsum([0] + [idx[i][3][0] for i in range(len(idx))])

        def rows_fn(a, b):
            parts = []
            for i in range(len(idx)):
                lo, hi = max(a, bases[i]), min(b, bases[i + 1])
                if lo < hi:
                    parts.append(st_rows(idx[i], row_bytes, lo - bases[i], hi - bases[i]))
            return np.concatenate(parts, axis=0)
        if self.kind == "ple-exl3":
            hb = st_entries(self.folder, lambda n: n.endswith(".ngram_embedding.head_bias"))
            if len(hb) != 1:
                sys.exit("source: head_bias not found")
            (path, off, _, shape), = hb.values()
            self.bias = np.array(np.memmap(path, mode="r", dtype=np.float16, offset=off, shape=shape))
        return int(bases[-1]), row_bytes, rows_fn


def compare_source(ts: nt.TableSet, kind: str, folder: str) -> list[str]:
    problems = []
    src = Source(kind, folder)
    for ti, t in enumerate(ts.tables):
        rows, row_bytes, rows_fn = src.table(t.name)
        if rows != t.rows or row_bytes != t.row_bytes:
            problems.append(f"{t.name}: {t.rows} rows of {t.row_bytes} B, the source has {rows} rows of {row_bytes} B")
            continue
        step = max(1, STEP_BYTES // row_bytes)
        for p in (p for p in ts.pieces if p.table == ti):
            r0, r1 = t.span(p.index)
            data = np.memmap(p.path, mode="r", dtype=np.uint8, offset=p.offset, shape=(p.nbytes,))
            for x in range(r0, r1, step):
                y = min(x + step, r1)
                if not np.array_equal(data[(x - r0) * row_bytes:(y - r0) * row_bytes], rows_fn(x, y).reshape(-1)):
                    problems.append(f"{p.name}: rows {x}..{y} differ from the source")
                    break
            del data
            print(f"{p.name}: {r1 - r0} rows compared", flush=True)
    bias = [e for e in ts.extras if e[0] == "per_layer_token_embd.bias"]
    if kind == "ple-exl3":
        if len(bias) != 1:
            problems.append("per_layer_token_embd.bias missing (an EXL3 row codec table needs it)")
        else:
            got = np.frombuffer(dict(ts.extra_bytes())["per_layer_token_embd.bias"], dtype=np.float16)
            if not np.array_equal(got, np.ascontiguousarray(src.bias).reshape(-1)):
                problems.append("per_layer_token_embd.bias differs from head_bias of the source")
    elif bias:
        # the zero bias of the earlier table shard layout, kept by a set converted from it; the loader does not read it
        got = np.frombuffer(dict(ts.extra_bytes())["per_layer_token_embd.bias"], dtype=np.float16)
        if np.any(got != 0):
            problems.append("per_layer_token_embd.bias of a table of the original dtype is not zero")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", type=Path, nargs="?", help="the first file (the index) of the table set, or a folder to search")
    ap.add_argument("--model", type=Path, help="the first file of the model set that uses it")
    ap.add_argument("--source")
    ap.add_argument("--source-kind", choices=["engram-fp8", "ple-original", "ple-exl3"])
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if bool(a.source) != bool(a.source_kind):
        ap.error("--source and --source-kind go together")

    if a.tables is None and a.model is None:
        ap.error("give the table set, or --model to look for it next to the model")
    first = a.tables
    if first is None or first.is_dir():
        if a.model is None:
            ap.error("a folder needs --model (the set is found by the model's name)")
        first, why = nt.find_table_set(a.model, first)
        if first is None:
            print(f"MISMATCH: no table set for {a.model.name}: {why}")
            return 1
        print(f"found {first}")
    ts = nt.open_table_set(first)
    print(f"table set {ts.first.name}: {len(ts.files)} files, {ts.kind} {ts.type_name}, tables {[t.name for t in ts.tables]}, "
          f"source {ts.source!r} rev {ts.source_revision!r}")
    problems = nt.check_table_set(ts)
    if a.model:
        problems += nt.check_model_index(a.model, ts)
        if not problems:
            found, why = nt.find_table_set(a.model)
            if found is None or found.resolve() != ts.first.resolve():
                print(f"note: the loader does not take this set next to {a.model.name} by itself "
                      f"({why or 'it takes ' + str(found)}); give it with --engram-tables")
            else:
                print("the loader takes this set next to the model by itself")
    if a.source and not problems:
        problems += compare_source(ts, a.source_kind, a.source)
    res = {"tables": [str(f) for f in ts.files], "model": str(a.model) if a.model else None,
           "problems": problems, "match": not problems}
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("MATCH" if not problems else f"MISMATCH ({len(problems)} problems)")
    for p in problems[:50]:
        print("  ", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
