# GGML_TYPE_MXFP8 (layout of upstream PR ggml-org/llama.cpp#26157), see ggml/src/ggml-quants.c
#
# A block is 264 bytes: 256 E4M3 values, then 8 E8M0 scales (one per 32 values); a value is e4m3(q) * 2^(e - 127).
# DeepSeek V4.1 stores an engram table as `.weight` F8_E4M3 [rows, 256] and `.scale` F8_E8M0 [rows, 8], so a row of
# the table is the weight row followed by the scale row, and repacking is a concatenation of bytes.
# Reference for the dequantization: DeepSeek-V4.1-Flash inference/model.py, ParallelEngramEmbedding.forward.

from __future__ import annotations

import numpy as np

BLOCK = 256
SUB = 32
BLOCK_BYTES = BLOCK + BLOCK // SUB


def repack_rows(values: np.ndarray, scales: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """E4M3 bytes [n, 256] and E8M0 bytes [n, 8] -> MXFP8 rows uint8 [n, 264]."""
    values = np.asarray(values, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    n = values.shape[0]
    if values.shape != (n, BLOCK) or scales.shape != (n, BLOCK // SUB):
        raise ValueError(f"MXFP8 repack: values {values.shape} and scales {scales.shape} are not [n, {BLOCK}] and [n, {BLOCK // SUB}]")
    if out is None:
        out = np.empty((n, BLOCK_BYTES), dtype=np.uint8)
    out[:, :BLOCK] = values
    out[:, BLOCK:] = scales
    return out
