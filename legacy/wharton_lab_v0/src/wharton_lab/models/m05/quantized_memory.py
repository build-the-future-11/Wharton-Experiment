"""Genuine packed expert weight storage (int8 codebook + residuals)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class PackedExpertBlock:
    """One expert's weights stored as codebook indices + residual bytes."""

    shape: tuple[int, ...]
    lo: float
    hi: float
    codebook: np.ndarray  # int8 (K,)
    indices: np.ndarray  # uint8 per element
    residual_bytes: bytes
    residual_dtype: np.dtype

    def nbytes(self) -> int:
        return (
            self.codebook.nbytes
            + self.indices.nbytes
            + len(self.residual_bytes)
            + 32  # shape + dtype metadata in blob header
        )


def _quantize_vector(vec: np.ndarray, codebook_size: int = 256) -> PackedExpertBlock:
    vec = np.asarray(vec, dtype=np.float64).ravel()
    lo, hi = float(vec.min()), float(vec.max())
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    codebook = np.linspace(lo, hi, codebook_size, dtype=np.float64)
    codebook_i8 = np.round(
        (codebook - lo) / (hi - lo) * 255.0 - 128.0
    ).astype(np.int8)
    # decode book for assignment
    book_f = (codebook_i8.astype(np.float32) + 128.0) / 255.0 * (hi - lo) + lo
    dist = np.abs(vec[:, None] - book_f[None, :])
    indices = np.argmin(dist, axis=1).astype(np.uint8)
    recon = book_f[indices]
    residual = (vec - recon).astype(np.float16)
    residual_bytes = zlib.compress(residual.tobytes(), level=1)
    return PackedExpertBlock(
        shape=vec.shape,
        lo=lo,
        hi=hi,
        codebook=codebook_i8,
        indices=indices,
        residual_bytes=residual_bytes,
        residual_dtype=np.dtype(np.float16),
    )


def _dequantize_vector(block: PackedExpertBlock) -> np.ndarray:
    span = block.hi - block.lo
    book_f = (block.codebook.astype(np.float64) + 128.0) / 255.0 * span + block.lo
    raw = zlib.decompress(block.residual_bytes)
    residual = np.frombuffer(raw, dtype=block.residual_dtype).astype(np.float64)
    recon = book_f[block.indices]
    return (recon + residual).reshape(block.shape)


class QuantizedExpertMemory:
    """Budgeted store of packed expert tensors."""

    def __init__(self, cap_bytes: int, codebook_size: int = 256) -> None:
        self.cap_bytes = cap_bytes
        self.codebook_size = codebook_size
        self._blocks: dict[str, PackedExpertBlock] = {}

    def used_bytes(self) -> int:
        return sum(b.nbytes() for b in self._blocks.values())

    def pack(self, expert_id: str, weights: np.ndarray) -> bool:
        block = _quantize_vector(weights, self.codebook_size)
        trial = dict(self._blocks)
        trial[expert_id] = block
        if sum(b.nbytes() for b in trial.values()) > self.cap_bytes:
            return False
        self._blocks = trial
        return True

    def unpack(self, expert_id: str) -> np.ndarray | None:
        block = self._blocks.get(expert_id)
        if block is None:
            return None
        return _dequantize_vector(block)

    def remove(self, expert_id: str) -> None:
        self._blocks.pop(expert_id, None)

    def serialize(self) -> Mapping[str, Any]:
        entries = {}
        for eid, block in self._blocks.items():
            entries[eid] = {
                "shape": block.shape,
                "lo": block.lo,
                "hi": block.hi,
                "codebook": block.codebook.tobytes(),
                "indices": block.indices.tobytes(),
                "residual": block.residual_bytes,
                "residual_dtype": str(block.residual_dtype),
            }
        return {"cap_bytes": self.cap_bytes, "codebook_size": self.codebook_size, "entries": entries}

    def deserialize(self, state: Mapping[str, Any]) -> None:
        self.cap_bytes = int(state["cap_bytes"])
        self.codebook_size = int(state["codebook_size"])
        self._blocks = {}
        for eid, ent in state.get("entries", {}).items():
            dt = np.dtype(ent["residual_dtype"])
            block = PackedExpertBlock(
                shape=tuple(ent["shape"]),
                lo=float(ent["lo"]),
                hi=float(ent["hi"]),
                codebook=np.frombuffer(ent["codebook"], dtype=np.int8).copy(),
                indices=np.frombuffer(ent["indices"], dtype=np.uint8).copy(),
                residual_bytes=bytes(ent["residual"]),
                residual_dtype=dt,
            )
            self._blocks[eid] = block

    def blob_nbytes(self) -> int:
        """Total serialized payload size (zlib residuals included)."""
        total = 0
        for ent in self.serialize()["entries"].values():
            total += len(ent["codebook"]) + len(ent["indices"]) + len(ent["residual"])
        return total
