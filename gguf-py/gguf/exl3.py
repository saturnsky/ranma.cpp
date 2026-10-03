# EXL3 trellis weights (GGML_TYPE_EXL3_*), see ggml/src/ggml-exl3.c
# format and mul1 codebook: ExLlamaV3, Copyright (c) 2025 Turboderp, MIT license
#
# The official tensor `trellis` is int16 [K/16, N/16, 16*bits] (K = in, N = out) in 16x16 tiles.
# GGUF keeps the tile bytes and only changes the tile order to [N/128][K/16][8][tile], so that a ggml
# tensor ne = [K, N(, E)] has the usual row stride K*bits/8 at 128-row boundaries.

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

QTYPES = frozenset(QTYPE_BY_BITS.values())


def bits2(qtype: GGMLQuantizationType) -> int:
    """Twice the bits per weight."""
    for bits, t in QTYPE_BY_BITS.items():
        if t == qtype:
            return int(bits * 2)
    raise ValueError(f"{qtype!r} is not an EXL3 trellis type")


def qtype_for_tile(tile_words: int) -> GGMLQuantizationType:
    """Type for an official trellis whose last dim has `tile_words` int16 words (16*bits)."""
    if tile_words % 8 != 0 or (tile_words // 8) not in [bits2(t) for t in QTYPES]:
        raise ValueError(f"unsupported EXL3 tile size of {tile_words} words")
    return QTYPE_BY_BITS[tile_words / 16 if tile_words % 16 else tile_words // 16]


def to_ggml(trellis: np.ndarray) -> np.ndarray:
    """Official [..., K/16, N/16, words] int16 -> GGUF bytes uint8 [..., N, K*bits/8]."""
    *lead, kt, nt, words = trellis.shape
    if trellis.dtype not in (np.int16, np.uint16) or nt % 8 != 0 or kt % 8 != 0:
        raise ValueError(f"unsupported EXL3 trellis {trellis.dtype} {trellis.shape}: K and N must be multiples of 128")
    qtype_for_tile(words)
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
    values = mul1(states)[..., _positions()]  # [B, kt, nt, r, c]
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
