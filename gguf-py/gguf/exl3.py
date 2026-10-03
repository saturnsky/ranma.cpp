# EXL3 trellis weights (GGML_TYPE_EXL3_*), see ggml/src/ggml-exl3.c
# format and codebooks (mul1, mcg, 3inst): ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
#
# The official tensor `trellis` is int16 [K/16, N/16, 16*bits] (K = in, N = out) in 16x16 tiles.
# GGUF keeps the tile bytes and only changes the tile order to [N/128][K/16][8][tile], so that a ggml
# tensor ne = [K, N(, E)] has the usual row stride K*bits/8 at 128-row boundaries. The type also names the
# codebook: an official tensor with a `mul1` marker is mul1, with `mcg` mcg, without a marker 3inst.

from __future__ import annotations

import numpy as np

from .constants import GGMLQuantizationType

QTYPE_BY_BITS: dict[float, GGMLQuantizationType] = {
    1: GGMLQuantizationType.EXL3_M1,
    2: GGMLQuantizationType.EXL3_M2,
    3: GGMLQuantizationType.EXL3_M3,
    4: GGMLQuantizationType.EXL3_M4,
    5: GGMLQuantizationType.EXL3_M5,
    6: GGMLQuantizationType.EXL3_M6,
    7: GGMLQuantizationType.EXL3_M7,
    8: GGMLQuantizationType.EXL3_M8,
    1.5: GGMLQuantizationType.EXL3_M1H,
    2.5: GGMLQuantizationType.EXL3_M2H,
    3.5: GGMLQuantizationType.EXL3_M3H,
}

CODEBOOKS = ("mul1", "mcg", "3inst")

# mcg and 3inst have integer bitrates only
QTYPE_BY_CODEBOOK_BITS: dict[tuple[str, float], GGMLQuantizationType] = {
    **{("mul1", bits): t for bits, t in QTYPE_BY_BITS.items()},
    **{("mcg", bits): GGMLQuantizationType[f"EXL3_G{bits}"] for bits in range(1, 9)},
    **{("3inst", bits): GGMLQuantizationType[f"EXL3_T{bits}"] for bits in range(1, 9)},
}

QTYPES = frozenset(QTYPE_BY_CODEBOOK_BITS.values())


def bits2(qtype: GGMLQuantizationType) -> int:
    """Twice the bits per weight."""
    for (_, bits), t in QTYPE_BY_CODEBOOK_BITS.items():
        if t == qtype:
            return int(bits * 2)
    raise ValueError(f"{qtype!r} is not an EXL3 trellis type")


def codebook(qtype: GGMLQuantizationType) -> str:
    """Codebook name of an EXL3 type."""
    for (cb, _), t in QTYPE_BY_CODEBOOK_BITS.items():
        if t == qtype:
            return cb
    raise ValueError(f"{qtype!r} is not an EXL3 trellis type")


def qtype_for_tile(tile_words: int, codebook: str = "mul1") -> GGMLQuantizationType:
    """Type for an official trellis whose last dim has `tile_words` int16 words (16*bits) and its codebook."""
    bits = tile_words / 16 if tile_words % 16 else tile_words // 16
    if tile_words % 8 != 0 or (codebook, bits) not in QTYPE_BY_CODEBOOK_BITS:
        raise ValueError(f"unsupported EXL3 tile size of {tile_words} words for the {codebook} codebook")
    return QTYPE_BY_CODEBOOK_BITS[(codebook, bits)]


def to_ggml(trellis: np.ndarray) -> np.ndarray:
    """Official [..., K/16, N/16, words] int16 -> GGUF bytes uint8 [..., N, K*bits/8]."""
    *lead, kt, nt, words = trellis.shape
    if trellis.dtype not in (np.int16, np.uint16) or nt % 8 != 0 or kt % 8 != 0:
        raise ValueError(f"unsupported EXL3 trellis {trellis.dtype} {trellis.shape}: K and N must be multiples of 128")
    if words % 8 != 0 or words // 8 not in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16):
        raise ValueError(f"unsupported EXL3 tile size of {words} words")
    t = trellis.reshape(*lead, kt, nt // 8, 8, words)
    t = np.ascontiguousarray(np.swapaxes(t, -4, -3))
    return t.view(np.uint8).reshape(*lead, nt * 16, kt * words // 8)


def from_ggml(data: np.ndarray, qtype: GGMLQuantizationType, k: int, n: int) -> np.ndarray:
    """GGUF bytes [..., N, K*bits/8] (or any shape with the same bytes) -> official [..., K/16, N/16, words] int16."""
    words = 8 * bits2(qtype)
    if k % 128 != 0 or n % 128 != 0:
        raise ValueError("K and N must be multiples of 128")
    data = np.ascontiguousarray(data).view(np.uint8)
    lead = data.shape[:-2] if data.ndim > 2 else ()
    if data.size != int(np.prod(lead, dtype=np.int64)) * k * n * words // 128:
        raise ValueError("EXL3 data size does not match K, N and type")
    t = data.reshape(*lead, n // 128, k // 16, 8, words * 2)
    t = np.ascontiguousarray(np.swapaxes(t, -4, -3)).view(np.int16)
    return t.reshape(*lead, k // 16, n // 16, words)


def _positions() -> np.ndarray:
    # ring position of tile element (r = row in K, c = column in N)
    r, c = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    return ((c % 8) * 4 + (r % 8) // 2) * 8 + r % 2 + (r >= 8) * 2 + (c >= 8) * 4


def mul1(states: np.ndarray) -> np.ndarray:
    """mul1 codebook, fp16."""
    x = (states.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    s = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    k_inv = np.array([0x1eee], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    k_bias = np.array([0xc931], dtype=np.uint16).view(np.float16).astype(np.float32)[0]
    return ((1024 + s).astype(np.float32) * k_inv + k_bias).astype(np.float16)


def _half_sum(x: np.ndarray) -> np.ndarray:
    x = ((x & 0x8FFF8FFF) ^ 0x3B603B60).astype(np.uint32)
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return (lo + hi).astype(np.float16)


def mcg(states: np.ndarray) -> np.ndarray:
    """mcg codebook, fp16."""
    return _half_sum((states.astype(np.uint64) * 0xCBAC1FED) & 0xFFFFFFFF)


def inst3(states: np.ndarray) -> np.ndarray:
    """3inst codebook, fp16."""
    return _half_sum((states.astype(np.uint64) * 89226354 + 64248484) & 0xFFFFFFFF)


CODEBOOK_FUNCTIONS = {"mul1": mul1, "mcg": mcg, "3inst": inst3}


def decode_raw(data: np.ndarray, qtype: GGMLQuantizationType, k: int, n: int) -> np.ndarray:
    """Raw codebook values W_raw as fp16 [..., N, K] (ggml orientation, row = output channel)."""
    b2 = bits2(qtype)
    ring = 128 * b2
    t = from_ggml(data, qtype, k, n)
    lead = t.shape[:-3]
    w = t.reshape(-1, k // 16, n // 16, 4 * b2 * 2).view("<u4").astype(np.uint64)  # [B, kt, nt, ring/32]
    p = np.arange(256)
    start = ((((p + 1) * b2) >> 1) - 16) % ring
    lo = start // 32
    hi = (lo + 1) % (ring // 32)
    window = (w[..., lo] << np.uint64(32)) | w[..., hi]
    states = (window >> (48 - start % 32).astype(np.uint64)) & np.uint64(0xffff)
    values = CODEBOOK_FUNCTIONS[codebook(qtype)](states)[..., _positions()]  # [B, kt, nt, r, c]
    out = np.transpose(values, (0, 2, 4, 1, 3))  # [B, nt, c, kt, r]
    return out.reshape(*lead, n, k)


def hadamard_128(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalized Sylvester Hadamard over blocks of 128 along `axis` (float64)."""
    h = np.array([[1.0]])
    while h.shape[0] < 128:
        h = np.block([[h, h], [h, -h]])
    h /= np.sqrt(128.0)
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, -1)
    shape = x.shape
    if shape[-1] % 128 != 0:
        raise ValueError("Hadamard axis must be a multiple of 128")
    y = (x.reshape(*shape[:-1], shape[-1] // 128, 128) @ h).reshape(shape)
    return np.moveaxis(y, -1, axis)


def effective_weight(data: np.ndarray, qtype: GGMLQuantizationType, k: int, n: int,
                     rot_in: np.ndarray, rot_out: np.ndarray) -> np.ndarray:
    """Dense float64 weight [..., N, K] of y = rot_out * H(W_raw^T H(rot_in * x)), one expert per leading index."""
    raw = decode_raw(data, qtype, k, n).astype(np.float64)
    w = hadamard_128(hadamard_128(raw, axis=-1), axis=-2)
    return w * np.asarray(rot_out, dtype=np.float64)[..., :, None] * np.asarray(rot_in, dtype=np.float64)[..., None, :]


# ---- row codec (GGML_TYPE_EXL3R_M*: ExLlamaV3 exl3_ngram_trellis tables) ------------------------------------------
#
# A row of ROW_DIM = 160 values is (1 + 10*bits) little-endian uint16 words, the official packed row: word 0 is the fp16
# scale, then a ring of 160*bits bits read LSB first. Position i owns bits [i*bits, (i+1)*bits); its state is the 16 ring
# bits that end there. Value = fp16 mul1(state) * scale (f32, exact). The table's per-head bias and the final fp16
# rounding of the official reconstruction are applied by the graph, not by the row.

ROW_DIM = 160

ROW_QTYPES: dict[int, GGMLQuantizationType] = {bits: GGMLQuantizationType[f"EXL3R_M{bits}"] for bits in range(1, 9)}


def row_bits(qtype: GGMLQuantizationType) -> int:
    if qtype not in ROW_QTYPES.values():
        raise ValueError(f"{qtype.name} is not an EXL3 row codec type")
    return int(qtype) - int(GGMLQuantizationType.EXL3R_M1) + 1


def row_qtype_for_words(words: int) -> GGMLQuantizationType:
    """Type of an official packed row of `words` int16 words (1 + ROW_DIM*bits/16)."""
    bits = (words - 1) * 16 // ROW_DIM
    if bits not in ROW_QTYPES or words != 1 + ROW_DIM * bits // 16:
        raise ValueError(f"an EXL3 n-gram row of {words} words is not 1 + 10*bits with bits 1..8")
    return ROW_QTYPES[bits]


def row_states(data: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows [n, 2 + 20*bits] uint8 -> (states [n, 160] uint32, scales [n] float16)."""
    data = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1, 2 + ROW_DIM * bits // 8)
    scales = data[:, :2].copy().view("<f2").reshape(-1)
    stream = np.unpackbits(data[:, 2:], axis=1, bitorder="little").astype(np.uint32)  # [n, 160*bits]
    i = np.arange(ROW_DIM).reshape(-1, 1)
    m = np.arange(16).reshape(1, -1)
    src = ((i - m // bits) % ROW_DIM) * bits + m % bits                                  # [160, 16]
    states = (stream[:, src] << m.astype(np.uint32)).sum(axis=-1, dtype=np.uint32)
    return states, scales


def decode_rows(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    """to_float of a row codec type: rows [n, 2 + 20*bits] uint8 -> float32 [n, 160] (no bias, no fp16 rounding)."""
    states, scales = row_states(data, row_bits(qtype))
    return mul1(states).astype(np.float32) * scales.astype(np.float32)[:, None]
