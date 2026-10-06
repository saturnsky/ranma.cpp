"""EXL3 (ExLlamaV3) checkpoints as a source for convert_hf_to_gguf.py.

An EXL3 linear is stored as `<key>.trellis` (int16 [K/16, N/16, 16*bits], 16x16 trellis tiles), `<key>.suh` / `<key>.svh`
(F16 [K] / [N]; old files: `.su` / `.sv` packed signs), an optional codebook marker (`.mul1`, `.mcg`; none = 3inst) and
an optional `.bias`. Here it enters the existing model class as the HF tensor `<key>.weight`: a symbolic tensor of
shape [N, K] (the padded dims as stored) that records which source rows and columns each of its rows and columns comes
from. The model class's tensor rules run unchanged on it. Only moves of whole 128-row groups and 128-column blocks of
one source, and stacks of such tensors, can be stored as EXL3 (W = diag(svh) H128 W_raw H128 diag(suh), H block
diagonal): any other result (a transpose, a split or permutation inside 128, rows of different sources, any change of
values) is refused with the name of the tensor. The output is the same tensor name with the EXL3 type of its bitrate
and codebook (gguf.exl3, tile bytes kept, tiles in 128-row strips), plus `<base>.rot_in` (suh) and `<base>.rot_out`
(svh) as F16 [K(, E)] / [N(, E)] with the original bytes. Other tensors keep their source dtype.

Model classes that permute the Q/K rows for RoPE (undo_permute) run without that permutation, and the file says
`<arch>.rope.style = "neox"`: the unpermuted (HF) rows with NEOX rope are the same operation.
"""

from __future__ import annotations

import copy
import json
import logging
import struct
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

import gguf
from gguf import exl3 as gexl3

from .base import LazyTorchTensor

logger = logging.getLogger("exl3")

MARKERS = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}
CHUNK_BYTES = 64 * 2**20

class Exl3Error(ValueError):
    pass


class Exl3MtpUnsupported(Exl3Error):
    """The MTP layers of the model cannot be stored as EXL3: convert again with no_mtp."""


def is_exl3(hparams: dict[str, Any]) -> bool:
    qc = hparams.get("quantization_config")
    return isinstance(qc, dict) and qc.get("quant_method") == "exl3"


def check_args(args: Any) -> None:
    """Options that change tensor values or join tensors cannot apply to EXL3 weights."""
    refused = [opt for opt, on in (
        ("--outtype", args.outtype != "auto"),
        ("--fuse-gate-up-exps", args.fuse_gate_up_exps),  # gate and up have their own rot_in
        ("--fuse-qkv", args.fuse_qkv),                    # q, k and v have their own rot_in
        ("--bigendian", args.bigendian),
        ("--remote", args.remote),
        ("--mmproj", args.mmproj),                        # quantized vision towers are not converted yet
        ("--dspark", args.dspark),
    ) if on]
    if refused:
        raise Exl3Error(f"EXL3 source: {', '.join(refused)} cannot be used with EXL3 weights")


# ---- safetensors headers -----------------------------------------------------------------------------------------

@dataclass
class StEntry:
    path: Path
    offset: int
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def array(self, dtype: Any) -> np.ndarray:
        if self.nbytes == 0:
            return np.zeros(self.shape, dtype=dtype)
        return np.memmap(self.path, mode="r", dtype=dtype, offset=self.offset, shape=self.shape)


def read_headers(dir_model: Path) -> dict[str, StEntry]:
    """Every tensor of every *.safetensors file in the directory (the official loader reads all of them)."""
    entries: dict[str, StEntry] = {}
    for path in sorted(dir_model.glob("*.safetensors")):
        with open(path, "rb") as f:
            n, = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for name, e in header.items():
            if name in entries:
                raise Exl3Error(f"tensor {name!r} is in both {entries[name].path.name} and {path.name}")
            begin, end = e["data_offsets"]
            entries[name] = StEntry(path, 8 + n + begin, end - begin, e["dtype"], tuple(e["shape"]))
    return entries


# ---- EXL3 sources ------------------------------------------------------------------------------------------------

@dataclass
class Source:
    prefix: str         # name after the model class's filter_tensors
    source: str         # name in the checkpoint
    trellis: StEntry
    suh: StEntry | None
    svh: StEntry | None
    su: StEntry | None
    sv: StEntry | None
    codebook: str
    qtype: gguf.GGMLQuantizationType
    k: int
    n: int
    rows_used: dict[int, int] = field(default_factory=dict)  # 128-row group -> times written

    def rot(self, side: str) -> np.ndarray:
        """suh / svh as F16 (old files: packed signs, bit set = -1, 16 per int16 word, low bit first)."""
        scale, signs = (self.suh, self.su) if side == "in" else (self.svh, self.sv)
        if scale is not None:
            return scale.array("<f2")
        assert signs is not None
        bits = signs.array("<u2").astype(np.uint32)
        expanded = ((bits[:, None] >> np.arange(16, dtype=np.uint32)) & 1).reshape(-1)
        return (1.0 - 2.0 * expanded).astype("<f2")


def _codebook(entries: dict[str, StEntry], key: str) -> str:
    found = [m for m in MARKERS if key + "." + m in entries]
    if len(found) > 1:
        raise Exl3Error(f"{key}: more than one codebook marker")
    for m in found:
        e = entries[key + "." + m]
        value = int(e.array("<u4").reshape(-1)[0]) if e.nbytes == 4 else None
        if e.dtype != "I32" or value != MARKERS[m]:
            raise Exl3Error(f"{key}.{m}: unexpected marker {e.dtype} {value}")
    return found[0] if found else "3inst"


# ---- symbolic EXL3 tensor ----------------------------------------------------------------------------------------

_TAGS = ("r", "c", "e")  # source row (output), source column (input), stacked source; "1" = size-1 axis without data


class Exl3Tensor:
    """A tensor whose element (i, j, ...) is element (rows, cols) of source srcs, in index arrays per axis kind.

    Each axis is a source-row axis, a source-column axis, a stack axis or a size-1 axis. arrays["r"] holds the source
    row of every combination of the row axes (in their logical order), likewise "c" and "e" (source ids). Only
    operations that keep the axis kinds apart are possible."""

    def __init__(self, state: Exl3State, shape: Sequence[int], tags: Sequence[str], arrays: dict[str, np.ndarray],
                 dtype: torch.dtype = torch.float16):
        self._state = state
        self.shape = torch.Size(int(s) for s in shape)
        self._tags = tuple(tags)
        self._arrays = arrays
        self.dtype = dtype
        for t in _TAGS:
            assert arrays[t].shape == tuple(s for s, g in zip(self.shape, self._tags) if g == t), (t, arrays[t].shape)

    @classmethod
    def source(cls, state: Exl3State, sid: int) -> Exl3Tensor:
        src = state.sources[sid]
        return cls(state, (src.n, src.k), ("r", "c"),
                   {"r": np.arange(src.n), "c": np.arange(src.k), "e": np.array(sid)})

    # -- properties used by model classes
    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return self.ndim

    def size(self, d: int | None = None):
        return self.shape if d is None else self.shape[d]

    def numel(self) -> int:
        return prod(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def device(self) -> torch.device:
        return torch.device("meta")

    @property
    def T(self) -> Exl3Tensor:
        return self.permute(*reversed(range(self.ndim)))

    def __repr__(self) -> str:
        return f"Exl3Tensor(shape={tuple(self.shape)}, axes={''.join(self._tags)})"

    def _fail(self, what: str) -> Exl3Error:
        return Exl3Error(f"{what} is not possible on an EXL3 weight {self!r}")

    def __getattr__(self, name: str):
        # an AttributeError keeps hasattr() working; the message names the operation
        raise AttributeError(f"EXL3 weight {self!r} has no {name!r}: only moves of 128-row groups / 128-column blocks are possible")

    # -- value-preserving no-ops
    def contiguous(self, *args, **kwargs) -> Exl3Tensor:
        return self

    def to(self, *args, **kwargs) -> Exl3Tensor:
        dtype = kwargs.get("dtype", next((a for a in args if isinstance(a, torch.dtype)), None))
        if dtype is not None and dtype not in (torch.float16, torch.float32, torch.float64):
            raise self._fail(f"conversion to {dtype}")
        return Exl3Tensor(self._state, self.shape, self._tags, self._arrays, dtype or self.dtype)

    def float(self) -> Exl3Tensor:
        return self.to(torch.float32)

    # -- shape operations
    def _with(self, shape, tags, arrays) -> Exl3Tensor:
        return Exl3Tensor(self._state, shape, tags, arrays, self.dtype)

    def reshape(self, *shape) -> Exl3Tensor:
        if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
            shape = tuple(shape[0])
        shape = [int(s) for s in shape]
        if shape.count(-1) > 1:
            raise self._fail("reshape with two -1")
        if -1 in shape:
            known = prod(s for s in shape if s != -1)
            shape[shape.index(-1)] = self.numel() // known if known else 0
        if prod(shape) != self.numel():
            raise self._fail(f"reshape to {shape}")
        # size-1 axes carry no data: drop them from the index arrays, then align the other axes in segments of equal size
        old = [(s, t) for s, t in zip(self.shape, self._tags) if s != 1]
        arrays = {t: self._arrays[t].reshape([s for s, g in old if g == t]) for t in _TAGS}
        new_tags = ["1"] * len(shape)
        new_sizes: dict[str, list[int]] = {t: [] for t in _TAGS}
        i = 0
        j_list = [j for j, s in enumerate(shape) if s != 1]
        j = 0
        while i < len(old) or j < len(j_list):
            seg_old, seg_new = [i], [j_list[j]] if j < len(j_list) else []
            if i >= len(old) or j >= len(j_list):
                raise self._fail(f"reshape to {shape}")
            po, pn = old[i][0], shape[j_list[j]]
            i += 1; j += 1
            while po != pn:
                if po < pn:
                    if i >= len(old):
                        raise self._fail(f"reshape to {shape}")
                    po *= old[i][0]; seg_old.append(i); i += 1
                else:
                    if j >= len(j_list):
                        raise self._fail(f"reshape to {shape}")
                    pn *= shape[j_list[j]]; seg_new.append(j_list[j]); j += 1
            tags = {old[k][1] for k in seg_old}
            if len(tags) != 1:
                raise self._fail(f"reshape to {shape} (it mixes rows, columns or sources)")
            t = tags.pop()
            for jj in seg_new:
                new_tags[jj] = t
                new_sizes[t].append(shape[jj])
        arrays = {t: arrays[t].reshape(new_sizes[t]) for t in _TAGS}
        return self._with(shape, new_tags, arrays)

    view = reshape

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Exl3Tensor:
        end_dim %= self.ndim
        shape = list(self.shape[:start_dim]) + [prod(self.shape[start_dim:end_dim + 1])] + list(self.shape[end_dim + 1:])
        return self.reshape(shape)

    def squeeze(self, dim: int | None = None) -> Exl3Tensor:
        if dim is None:
            return self.reshape([s for s in self.shape if s != 1])
        dim %= self.ndim
        if self.shape[dim] != 1:
            return self
        return self.reshape([s for d, s in enumerate(self.shape) if d != dim])

    def unsqueeze(self, dim: int) -> Exl3Tensor:
        dim %= self.ndim + 1
        shape = list(self.shape)
        shape.insert(dim, 1)
        tags = list(self._tags)
        tags.insert(dim, "1")
        return self._with(shape, tags, self._arrays)

    def permute(self, *dims) -> Exl3Tensor:
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        dims = [d % self.ndim for d in dims]
        if sorted(dims) != list(range(self.ndim)):
            raise self._fail(f"permute {dims}")
        arrays = {}
        for t in _TAGS:
            order = [d for d in range(self.ndim) if self._tags[d] == t]
            arrays[t] = np.transpose(self._arrays[t], [order.index(d) for d in dims if self._tags[d] == t])
        return self._with([self.shape[d] for d in dims], [self._tags[d] for d in dims], arrays)

    def transpose(self, d0: int, d1: int) -> Exl3Tensor:
        dims = list(range(self.ndim))
        dims[d0], dims[d1] = dims[d1], dims[d0]
        return self.permute(dims)

    swapaxes = transpose

    def t(self) -> Exl3Tensor:
        return self.transpose(0, 1)

    def __getitem__(self, key) -> Exl3Tensor:
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            e = key.index(Ellipsis)
            n_fill = self.ndim - (len(key) - 1 - sum(k is None for k in key))
            key = key[:e] + (slice(None),) * n_fill + key[e + 1:]
        shape, tags = [], []
        index: dict[str, list[Any]] = {t: [] for t in _TAGS}
        d = 0
        for k in key:
            if k is None:
                shape.append(1); tags.append("1")
                continue
            if d >= self.ndim:
                raise self._fail(f"index {key}")
            t = self._tags[d]
            if isinstance(k, slice):
                r = range(*k.indices(self.shape[d]))
                if t != "1":
                    index[t].append(slice(k.start, k.stop, k.step))
                if len(r) == 0:
                    raise self._fail(f"empty slice {key}")
                shape.append(len(r)); tags.append(t if t != "1" or len(r) != 1 else "1")
            elif isinstance(k, int):
                if t != "1":
                    index[t].append(k)
            else:
                raise self._fail(f"index {key!r}")
            d += 1
        for dd in range(d, self.ndim):
            shape.append(self.shape[dd]); tags.append(self._tags[dd])
            if self._tags[dd] != "1":
                index[self._tags[dd]].append(slice(None))
        arrays = {t: self._arrays[t][tuple(index[t])] if index[t] else self._arrays[t] for t in _TAGS}
        return self._with(shape, tags, arrays)

    def split(self, split_size, dim: int = 0) -> tuple[Exl3Tensor, ...]:
        dim %= self.ndim
        n = self.shape[dim]
        sizes = [split_size] * (n // split_size) + ([n % split_size] if n % split_size else []) if isinstance(split_size, int) else list(split_size)
        if sum(sizes) != n:
            raise self._fail(f"split {split_size}")
        out, start = [], 0
        for s in sizes:
            key = [slice(None)] * self.ndim
            key[dim] = slice(start, start + s)
            out.append(self[tuple(key)])
            start += s
        return tuple(out)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Exl3Tensor, ...]:
        n = self.shape[dim % self.ndim]
        return self.split((n + chunks - 1) // chunks, dim)

    def _promote(self, dim: int) -> Exl3Tensor:
        """A size-1 axis without data becomes a stack axis of one source (for cat / stack)."""
        if self._tags[dim] != "1":
            return self
        tags = list(self._tags)
        tags[dim] = "e"
        pos = sum(1 for d in range(dim) if self._tags[d] == "e")
        arrays = dict(self._arrays)
        arrays["e"] = np.expand_dims(self._arrays["e"], pos)
        return self._with(self.shape, tags, arrays)

    @classmethod
    def _cat(cls, tensors: list[Exl3Tensor], dim: int) -> Exl3Tensor:
        first = tensors[0]
        dim %= first.ndim
        tensors = [t._promote(dim) for t in tensors]
        tags = tensors[0]._tags
        for t in tensors[1:]:
            if t._tags != tags or any(a != b for d, (a, b) in enumerate(zip(t.shape, tensors[0].shape)) if d != dim):
                raise first._fail("concatenation of tensors with different layouts")
        kind = tags[dim]
        pos = sum(1 for d in range(dim) if tags[d] == kind)
        arrays = {}
        for t in _TAGS:
            if t == kind:
                arrays[t] = np.concatenate([x._arrays[t] for x in tensors], axis=pos)
            else:
                if any(not np.array_equal(x._arrays[t], tensors[0]._arrays[t]) for x in tensors[1:]):
                    what = {"r": "rows", "c": "columns", "e": "sources"}[t]
                    raise first._fail(f"concatenation along a {'row' if kind == 'r' else 'column' if kind == 'c' else 'stack'} "
                                      f"axis of tensors with different {what} (different EXL3 sources have different rot_in / rot_out)")
                arrays[t] = tensors[0]._arrays[t]
        shape = list(tensors[0].shape)
        shape[dim] = sum(x.shape[dim] for x in tensors)
        return tensors[0]._with(shape, tags, arrays)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.cat, torch.concat, torch.concatenate):
            tensors = list(args[0])
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if not all(isinstance(t, cls) for t in tensors):
                raise Exl3Error("concatenation of an EXL3 weight with another tensor")
            return cls._cat(tensors, dim)
        if func is torch.stack:
            tensors = list(args[0])
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if not all(isinstance(t, cls) for t in tensors):
                raise Exl3Error("stack of an EXL3 weight with another tensor")
            return cls._cat([t.unsqueeze(dim) for t in tensors], dim)
        method = {torch.reshape: "reshape", torch.permute: "permute", torch.transpose: "transpose", torch.swapaxes: "swapaxes",
                  torch.squeeze: "squeeze", torch.unsqueeze: "unsqueeze", torch.split: "split", torch.chunk: "chunk",
                  torch.flatten: "flatten", torch.t: "t"}.get(func)
        if method is not None and isinstance(args[0], cls):
            return getattr(args[0], method)(*args[1:], **kwargs)
        raise Exl3Error(f"torch.{getattr(func, '__name__', func)} is not possible on an EXL3 weight")

    def _arith(self, *args, **kwargs):
        raise self._fail("a change of values")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __neg__ = __pow__ = _arith
    __matmul__ = __rmatmul__ = _arith


# ---- conversion state ----------------------------------------------------------------------------------------------

@dataclass
class Exl3State:
    dir_model: Path
    ngram: str = "refuse"
    dry: bool = False  # check the EXL3 weights the class writes, write nothing (MTP check)
    sources: list[Source] = field(default_factory=list)
    source_dtypes: dict[str, str] = field(default_factory=dict)
    ngram_tensors: list[str] = field(default_factory=list)
    written: list[dict[str, Any]] = field(default_factory=list)

    def install(self, model: Any) -> None:
        """Replace the EXL3 groups of model.model_tensors (names after filter_tensors) by symbolic `<key>.weight`."""
        entries = read_headers(self.dir_model)
        cls = type(model)
        renamed: dict[str, str] = {}
        for name in entries:
            item = cls.filter_tensors((name, lambda: None))
            if item is not None:
                renamed[item[0]] = name
        for new, old in renamed.items():
            self.source_dtypes[new] = entries[old].dtype
        mt = model.model_tensors

        # n-gram row codecs (Qwen3.8): a trellis without suh/svh
        rows = [n for n in renamed if n.endswith(".trellis") and not any(n[:-8] + s in renamed for s in (".suh", ".su"))]
        self.ngram_tensors = sorted(renamed[n] for n in rows)
        if rows:
            if self.ngram == "refuse":
                raise Exl3Error(f"EXL3 n-gram row codec tensors ({len(rows)}, e.g. {renamed[rows[0]]!r}) need the EXL3 row codec "
                                "type, which is not implemented yet; --exl3-ngram omit converts without the n-gram table")
            base = {n.rsplit(".shard_", 1)[0] if ".shard_" in n else n[:-8] for n in rows}
            # the table belongs to a PLE module (`<layer>.ple.ple_embedding.ngram_embedding`): the module's own tensors
            # (key / value projections, norms, conv) are of no use without it, and the loader of a file without
            # PLE layers does not create them
            base = {b[:b.index(".ple.") + 4] if ".ple." in b else b for b in base}
            dropped = sorted(n for n in mt if any(n.startswith(b + ".") or n == b for b in base))
            for n in dropped:
                del mt[n]
            logger.warning("EXL3: %d source tensors omitted with the n-gram table: %s", len(dropped), ", ".join(dropped))
            if "ple_layer_ids" in model.hparams:
                model.hparams["ple_layer_ids"] = []
            logger.warning("EXL3: n-gram table omitted (%d row codec tensors); the file has no PLE n-gram embedding", len(rows))

        keys = sorted(n[:-8] for n in renamed if n.endswith(".trellis") and n not in rows)
        # every file is indexed (index_tensors); a linear the class did not see would be lost
        outside = [k for k in keys if k + ".trellis" not in mt]
        if outside:
            raise Exl3Error(f"{len(outside)} EXL3 linear groups are outside the indexed model files, e.g. {renamed[outside[0] + '.trellis']!r} "
                            f"in {entries[renamed[outside[0] + '.trellis']].path.name}")
        for key in keys:
            ent = {s: entries[renamed[key + s]] if key + s in renamed else None
                   for s in (".trellis", ".suh", ".svh", ".su", ".sv")}
            tr = ent[".trellis"]
            assert tr is not None
            if (ent[".suh"] is None and ent[".su"] is None) or (ent[".svh"] is None and ent[".sv"] is None):
                raise Exl3Error(f"{key}: EXL3 trellis without input / output scales")
            if tr.dtype != "I16" or len(tr.shape) != 3:
                raise Exl3Error(f"{key}.trellis: {tr.dtype} {list(tr.shape)}")
            kt, nt, words = tr.shape
            cb = _codebook({key + "." + m: entries[renamed[key + "." + m]] for m in MARKERS if key + "." + m in renamed}, key)
            try:
                qtype = gexl3.qtype_for_tile(words, cb)
            except ValueError as e:
                raise Exl3Error(f"{key}: {e}") from e
            k, n = kt * 16, nt * 16
            if k % 128 or n % 128:
                raise Exl3Error(f"{key}: K = {k}, N = {n} are not multiples of 128")
            for side, dim in (("suh", k), ("svh", n)):
                e = ent["." + side]
                if e is not None and (e.dtype != "F16" or e.shape != (dim,)):
                    raise Exl3Error(f"{key}.{side}: {e.dtype} {list(e.shape)}, expected F16 [{dim}]")
            for side, dim in (("su", k), ("sv", n)):
                e = ent["." + side]
                if e is not None and (e.dtype != "I16" or e.shape != (dim // 16,)):
                    raise Exl3Error(f"{key}.{side}: {e.dtype} {list(e.shape)}, expected I16 [{dim // 16}]")
            src = Source(key, renamed[key + ".trellis"][:-len(".trellis")], tr, ent[".suh"], ent[".svh"], ent[".su"], ent[".sv"], cb, qtype, k, n)
            sid = len(self.sources)
            self.sources.append(src)
            if key + ".weight" in mt:
                raise Exl3Error(f"{key}: both an EXL3 group and a dense weight")
            for s in (".trellis", ".suh", ".svh", ".su", ".sv", ".mul1", ".mcg"):
                mt.pop(key + s, None)
            mt[key + ".weight"] = lambda sid=sid: Exl3Tensor.source(self, sid)
            self.source_dtypes[key + ".weight"] = "EXL3"
        logger.info("EXL3: %d linear groups (%s)", len(self.sources),
                    ", ".join(f"{q.name} {c}" for q, c in sorted(self._type_counts().items())))

    def _type_counts(self) -> dict[gguf.GGMLQuantizationType, int]:
        counts: dict[gguf.GGMLQuantizationType, int] = {}
        for s in self.sources:
            counts[s.qtype] = counts.get(s.qtype, 0) + 1
        return counts

    def check_config(self, hparams: dict[str, Any]) -> None:
        """The codebook of the config and of quantization_config.json (bits, multiplier) against the tensor headers."""
        qc = hparams["quantization_config"]
        configured = qc.get("codebook", "3inst")
        found = {s.codebook for s in self.sources}
        if found and found != {configured}:
            raise Exl3Error(f"EXL3 codebook {configured!r} in quantization_config, tensors use {sorted(found)}")
        path = self.dir_model / "quantization_config.json"
        if not path.is_file():
            return
        storage = json.loads(path.read_text(encoding="utf-8")).get("tensor_storage", {})
        checked = 0
        for s in self.sources:
            info = storage.get(s.source)
            if info is None or info.get("quant_format") != "exl3":
                continue
            bits = gexl3.bits2(s.qtype) / 2
            if info.get("bits_per_weight") is not None and float(info["bits_per_weight"]) != bits:
                raise Exl3Error(f"{s.source}: bits_per_weight {info['bits_per_weight']} in quantization_config.json, tiles give {bits}")
            mult = {"mul1": "mul1_multiplier", "mcg": "mcg_multiplier"}
            for cb, keyname in mult.items():
                if (keyname in info) != (s.codebook == cb) or (keyname in info and info[keyname] != MARKERS[cb]):
                    raise Exl3Error(f"{s.source}: {keyname} in quantization_config.json does not match the {s.codebook} tensors")
            checked += 1
        logger.info("EXL3: %d of %d groups checked against quantization_config.json", checked, len(self.sources))

    # -- output
    def emit(self, model: Any, name: str, t: Exl3Tensor) -> None:
        if not name.endswith(".weight"):
            raise Exl3Error(f"{name}: an EXL3 weight must be written as a .weight tensor")
        base = name[:-len(".weight")]
        tags = t._tags
        if t.ndim == 2 and tags == ("r", "c"):
            srcs = [int(t._arrays["e"])]
            stacked = False
        elif t.ndim == 3 and tags[1:] == ("r", "c") and tags[0] in ("e", "1"):
            srcs = [int(s) for s in np.atleast_1d(t._arrays["e"]).reshape(-1)]
            stacked = True
        else:
            hint = " (a transpose)" if set(tags[-2:]) == {"r", "c"} else ""
            raise Exl3Error(f"{name}: EXL3 weight {t!r} is not [N, K] or [E, N, K] of source rows and columns{hint}")
        rows, cols = t._arrays["r"].reshape(-1), t._arrays["c"].reshape(-1)
        n_blocks = self._blocks(name, rows, "rows", "a RoPE permutation inside a head, or a split not at a 128-row boundary")
        k_blocks = self._blocks(name, cols, "columns", "a split or permutation of the input not at 128-column blocks")
        sources = [self.sources[s] for s in srcs]
        qtype = sources[0].qtype
        for s in sources:
            if s.qtype != qtype:
                raise Exl3Error(f"{name}: stacked sources of types {qtype.name} and {s.qtype.name} ({s.prefix})")
            if (int(max(n_blocks)) + 1) * 128 > s.n or (int(max(k_blocks)) + 1) * 128 > s.k:
                raise Exl3Error(f"{name}: rows or columns outside {s.prefix}")
            for b in n_blocks:
                s.rows_used[int(b)] = s.rows_used.get(int(b), 0) + 1
        if self.dry:
            return
        n, k = 128 * len(n_blocks), 128 * len(k_blocks)
        row_bytes = k * gexl3.bits2(qtype) // 16

        def weight_chunks() -> list[Callable[[], np.ndarray]]:
            groups = max(1, CHUNK_BYTES // (128 * row_bytes))
            out = []
            for s in sources:
                for g0 in range(0, len(n_blocks), groups):
                    out.append(lambda s=s, g0=g0: _relayout(s, n_blocks[g0:g0 + groups], k_blocks))
            return out

        def rot_chunks(side: str, idx: np.ndarray) -> list[Callable[[], np.ndarray]]:
            return [lambda s=s: np.ascontiguousarray(s.rot(side)[idx]) for s in sources]

        lead = (len(sources),) if stacked else ()
        w = gguf.LazyChunkedTensor(weight_chunks(), shape=(*lead, n, row_bytes), dtype=np.uint8)
        rin = gguf.LazyChunkedTensor(rot_chunks("in", cols), shape=(*lead, k), dtype=np.float16)
        rout = gguf.LazyChunkedTensor(rot_chunks("out", rows), shape=(*lead, n), dtype=np.float16)
        model.gguf_writer.add_tensor(name, w, raw_dtype=qtype)
        model.gguf_writer.add_tensor(base + ".rot_in", rin)
        model.gguf_writer.add_tensor(base + ".rot_out", rout)
        moved = not (np.array_equal(n_blocks, np.arange(sources[0].n // 128)) and np.array_equal(k_blocks, np.arange(sources[0].k // 128)))
        shape = ", ".join(str(x) for x in (k, n, *lead))
        logger.info(f"{name + ',':<48} exl3 --> {qtype.name}, shape = {{{shape}}}{' (128-row groups moved)' if moved else ''}")
        self.written.append(dict(name=name, type=qtype.name, k=k, n=n, experts=len(sources) if stacked else 0,
                                 sources=[s.prefix for s in sources] if len(sources) <= 4 else [sources[0].prefix, "...", sources[-1].prefix],
                                 n_blocks=None if not moved else [int(b) for b in n_blocks],
                                 k_blocks=None if not moved else [int(b) for b in k_blocks]))

    @staticmethod
    def _blocks(name: str, idx: np.ndarray, what: str, hint: str) -> np.ndarray:
        if len(idx) % 128:
            raise Exl3Error(f"{name}: {len(idx)} {what} are not whole 128-blocks ({hint})")
        b = idx.reshape(-1, 128)
        if np.any(b[:, 0] % 128) or np.any(b - b[:, :1] != np.arange(128)):
            raise Exl3Error(f"{name}: {what} are not whole 128-blocks of the source ({hint})")
        return b[:, 0] // 128

    def finish(self) -> None:
        unused = [s.prefix for s in self.sources if not s.rows_used]
        if unused:
            raise Exl3Error(f"EXL3 groups not written by the model class: {unused[:6]}{' ...' if len(unused) > 6 else ''}")
        partial = [s.prefix for s in self.sources if len(s.rows_used) != s.n // 128]
        if partial:
            raise Exl3Error(f"EXL3 groups written only in part: {partial[:6]}{' ...' if len(partial) > 6 else ''}")
        reused = [s.prefix for s in self.sources if max(s.rows_used.values()) > 1]
        if reused:
            logger.warning("EXL3: %d groups have rows written more than once, e.g. %s", len(reused), reused[0])

    def add_metadata(self, model: Any, rope_neox: bool) -> None:
        qc = model.hparams["quantization_config"]
        w = model.gguf_writer
        w.add_string("quantize.exl3.version", str(qc.get("version", "")))
        if "bits" in qc:
            w.add_float32("quantize.exl3.bits", float(qc["bits"]))
        if "head_bits" in qc:
            w.add_float32("quantize.exl3.head_bits", float(qc["head_bits"]))
        w.add_string("quantize.exl3.codebook", qc.get("codebook", "3inst"))
        if "out_scales" in qc:
            w.add_string("quantize.exl3.out_scales", str(qc["out_scales"]))
        cal = qc.get("calibration") or {}
        if "rows" in cal:
            w.add_uint32("quantize.exl3.calibration.rows", int(cal["rows"]))
        if "cols" in cal:
            w.add_uint32("quantize.exl3.calibration.cols", int(cal["cols"]))
        if self.ngram_tensors and self.ngram == "omit":
            w.add_string("quantize.exl3.ngram", "omitted")
        if rope_neox:
            w.add_string(gguf.Keys.Rope.STYLE.format(arch=w.arch), "neox")


def _relayout(s: Source, n_blocks: np.ndarray, k_blocks: np.ndarray) -> np.ndarray:
    """GGUF bytes [len(n_blocks)*128, K*bits/8] of the 128-row groups n_blocks with the 128-column blocks k_blocks."""
    t = s.trellis.array("<i2")  # [K/16, N/16, words]
    nt = np.concatenate([np.arange(8 * b, 8 * b + 8) for b in n_blocks])
    kt = np.concatenate([np.arange(8 * b, 8 * b + 8) for b in k_blocks])
    if np.array_equal(nt, np.arange(nt[0], nt[0] + len(nt))):
        sub = t[:, nt[0]:nt[0] + len(nt)]
    else:
        sub = t[:, nt]
    if not np.array_equal(kt, np.arange(t.shape[0])):
        sub = sub[kt]
    out = gexl3.to_ggml(np.ascontiguousarray(sub))
    del t
    return out


# ---- the model class ------------------------------------------------------------------------------------------------

class _Exl3Model:
    """Mixed in before the model class (see adapt())."""

    _exl3: Exl3State
    _exl3_ngram: str = "refuse"
    _exl3_rope_neox: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # ty: ignore[too-many-positional-arguments]
        self._exl3 = Exl3State(self.dir_model, ngram=self._exl3_ngram)  # ty: ignore[unresolved-attribute]
        self._exl3_ftype = self.ftype  # ty: ignore[unresolved-attribute]
        if not self.no_mtp:  # ty: ignore[unresolved-attribute]
            self._exl3_check_mtp()

    def _exl3_check_mtp(self):
        """Run the class's tensor rules on the MTP tensors before anything is written. When one of them cannot be
        stored as EXL3, raise Exl3MtpUnsupported: the caller converts the model without its MTP layers (--no-mtp)."""
        mtp = type(self)._exl3_mtp_names  # ty: ignore[unresolved-attribute]
        if not any(n in mtp for n in self.model_tensors):  # ty: ignore[unresolved-attribute]
            return
        trial = copy.copy(self)
        for k, v in list(trial.__dict__.items()):
            if isinstance(v, (dict, list, set)):
                setattr(trial, k, copy.copy(v))
        trial._exl3 = Exl3State(self.dir_model, ngram=self._exl3_ngram, dry=True)  # ty: ignore[unresolved-attribute]
        trial._exl3.install(trial)
        groups = {n[:-len(".trellis")] + ".weight" for n in mtp if n.endswith(".trellis")}
        names = [n for n in trial.model_tensors if n in mtp or n in groups]
        try:
            for name in names:
                data = trial.model_tensors[name]()
                if not isinstance(data, Exl3Tensor) and data.dtype not in (torch.float16, torch.float32):
                    data = data.to(torch.float32)
                bid = next((int(p) for p in name.split(".") if p.isdecimal()), None)
                for _ in trial.modify_tensors(data, name, bid):
                    pass
        except Exl3Error as e:
            msg = f"EXL3: the MTP layers of this model cannot be converted from EXL3 for now: {e}"
            if self.mtp_only:  # ty: ignore[unresolved-attribute]
                raise Exl3Error(msg + "; --mtp has nothing to write") from e
            logger.error("%s; converting the model without its MTP layers (as with --no-mtp)", msg)
            raise Exl3MtpUnsupported(msg) from e
        logger.info("EXL3: %d MTP tensors checked", len(names))

    @classmethod
    def filter_tensors(cls, item):
        titem = cls._exl3_filter(item)
        if titem is not None and getattr(cls, "supports_mtp_export", False) and not cls.no_mtp:  # ty: ignore[unresolved-attribute]
            # an MTP tensor is one the class keeps, but drops with --no-mtp
            own = cls.__dict__.get("no_mtp")
            cls.no_mtp = True
            try:
                if cls._exl3_filter(item) is None:
                    cls._exl3_mtp_names.add(titem[0])  # ty: ignore[unresolved-attribute]
            finally:
                if own is None:
                    del cls.no_mtp
                else:
                    cls.no_mtp = own
        return titem

    @classmethod
    def _exl3_filter(cls, item):
        # the class's rules name a linear by its `<key>.weight`: apply them to the parts of an EXL3 group under that name
        name, gen = item
        key, _, part = name.rpartition(".")
        if key and part in ("trellis", "suh", "svh", "su", "sv", "mul1", "mcg"):
            titem = super().filter_tensors((key + ".weight", gen))  # ty: ignore[unresolved-attribute]
            if titem is None:
                return None
            if titem[0].endswith(".weight"):
                return titem[0][:-len("weight")] + part, titem[1]
        return super().filter_tensors(item)  # ty: ignore[unresolved-attribute]

    def index_tensors(self, remote_hf_model_id=None):
        tensors = super().index_tensors(remote_hf_model_id)  # ty: ignore[unresolved-attribute]
        # the official loader reads every *.safetensors file of the directory; model.safetensors.index.json lists
        # only the main shards (Qwen3.8 n-gram tables and MTP patch, DeepSeek-V4 hc_head patch and MTP)
        index = self.dir_model / "model.safetensors.index.json"  # ty: ignore[unresolved-attribute]
        if remote_hf_model_id is None and index.is_file():
            listed = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            for path in sorted(self.dir_model.glob("*.safetensors")):  # ty: ignore[unresolved-attribute]
                if path.name in listed:
                    continue
                added = 0
                for name, data in gguf.utility.SafetensorsLocal(path).tensors.items():
                    item = self.filter_tensors((name, lambda data=data: LazyTorchTensor.from_local_tensor(data)))  # ty: ignore[unresolved-attribute]
                    if item is None:
                        continue
                    if item[0] in tensors:
                        raise Exl3Error(f"tensor {item[0]!r} of {path.name} is also in the indexed files")
                    tensors[item[0]] = item[1]
                    added += 1
                logger.info("EXL3: %d tensors from %s (not in the index)", added, path.name)
        return tensors

    def dequant_model(self):
        self._exl3.install(self)
        self._exl3.check_config(self.hparams)  # ty: ignore[unresolved-attribute]
        # the class's own handling of other quantized tensors still runs, without the EXL3 method it does not know
        qc = self.hparams.pop("quantization_config")  # ty: ignore[unresolved-attribute]
        try:
            super().dequant_model()  # ty: ignore[unresolved-attribute]
        finally:
            self.hparams["quantization_config"] = qc  # ty: ignore[unresolved-attribute]

    def modify_tensors(self, data_torch, name, bid) -> Iterable[tuple[str, Any]]:
        try:
            for new_name, t in super().modify_tensors(data_torch, name, bid):  # ty: ignore[unresolved-attribute]
                if isinstance(t, Exl3Tensor):
                    self._exl3.emit(self, new_name, t)
                else:
                    yield new_name, t
        except Exl3Error as e:
            raise Exl3Error(f"{name}: {e}") from e

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # tensors other than EXL3 keep their source type: the file type that prepare_tensors falls back to follows
        # the source tensor, after the converter's own rules (F32 for 1-D, norms, routers, convolutions, ...)
        self.ftype = {"F16": gguf.LlamaFileType.MOSTLY_F16, "BF16": gguf.LlamaFileType.MOSTLY_BF16,
                      "F32": gguf.LlamaFileType.ALL_F32}.get(self._exl3.source_dtypes.get(name, ""), self._exl3_ftype)
        return super().tensor_force_quant(name, new_name, bid, n_dims)  # ty: ignore[unresolved-attribute]

    def prepare_tensors(self):
        super().prepare_tensors()  # ty: ignore[unresolved-attribute]
        self._exl3.finish()

    def prepare_metadata(self, vocab_only: bool):
        self.ftype = gguf.LlamaFileType.MOSTLY_EXL3
        super().prepare_metadata(vocab_only)  # ty: ignore[unresolved-attribute]
        self._exl3.add_metadata(self, self._exl3_rope_neox)


def adapt(model_class: type, ngram: str = "refuse") -> type:
    """The model class with EXL3 weights; RoPE Q/K permutation off (NEOX key instead). The MTP layers follow the class
    (default, --no-mtp, --mtp); when they cannot be stored as EXL3, construction raises Exl3MtpUnsupported."""
    attrs: dict[str, Any] = dict(model_arch=model_class.model_arch, _exl3_ngram=ngram, _exl3_mtp_names=set())
    if getattr(model_class, "undo_permute", False):
        attrs["undo_permute"] = False
        attrs["_exl3_rope_neox"] = True
    return type("Exl3" + model_class.__name__, (_Exl3Model, model_class), attrs)
