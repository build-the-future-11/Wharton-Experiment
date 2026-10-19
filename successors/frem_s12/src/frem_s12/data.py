"""Strict six-portfolio archive parsing and generated panel contracts."""

from __future__ import annotations

import csv
import io
import itertools
import zipfile
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s12.errors import ContractError

DIMENSIONS = 6


@dataclass(frozen=True, slots=True)
class ReturnPanel:
    """One contiguous monthly six-dimensional return panel."""

    family: str
    months: NDArray[np.int64]
    returns: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own arrays and enforce finite contiguous chronology."""
        months = np.array(self.months, dtype=np.int64, copy=True)
        returns = np.array(self.returns, dtype=np.float64, copy=True)
        if not self.family or months.ndim != 1 or returns.shape != (len(months), DIMENSIONS):
            raise ContractError("panel shape or family is invalid")
        if len(months) < 24 or not np.isfinite(returns).all() or len(np.unique(months)) != len(months):
            raise ContractError("panel values or months are invalid")
        if np.any(np.diff(months) <= 0):
            raise ContractError("panel months are not increasing")
        for left, right in itertools.pairwise(months):
            year, month = divmod(int(left), 100)
            expected = year * 100 + month + 1 if month < 12 else (year + 1) * 100 + 1
            if int(right) != expected:
                raise ContractError("panel calendar is not contiguous")
        months.setflags(write=False)
        returns.setflags(write=False)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "returns", returns)


def _archive_member(payload: bytes) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            bounded = [name for name in members if name.lower().endswith((".csv", ".txt"))]
            if len(bounded) != 1:
                raise ContractError("archive must contain one CSV or TXT member")
            info = archive.getinfo(bounded[0])
            if info.file_size > 10_000_000 or info.compress_size <= 0:
                raise ContractError("archive member exceeds bounded contract")
            return bounded[0], archive.read(bounded[0])
    except zipfile.BadZipFile as exc:
        raise ContractError("input is not a valid ZIP archive") from exc


def _month(value: str) -> int | None:
    stripped = value.strip()
    if len(stripped) != 6 or not stripped.isdigit():
        return None
    numeric = int(stripped)
    return numeric if 1 <= numeric % 100 <= 12 else None


def _numeric(fields: list[str]) -> list[float]:
    if len(fields) != DIMENSIONS:
        raise ContractError("monthly row does not contain six portfolios")
    values = np.asarray([float(value.strip()) for value in fields], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= -99.0):
        raise ContractError("monthly row contains missing or nonfinite values")
    return [float(value) for value in values / 100.0]


def _table_rows(name: str, text: str) -> list[list[str]]:
    if name.lower().endswith(".csv"):
        return list(csv.reader(io.StringIO(text)))
    return [line.split() for line in text.splitlines()]


def parse_archive(payload: bytes, *, family: str, cutoff: int) -> ReturnPanel:
    """Parse the first value-weighted monthly six-portfolio table through cutoff."""
    name, member = _archive_member(payload)
    try:
        text = member.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ContractError("archive member is not UTF-8 text") from exc
    months: list[int] = []
    rows: list[list[float]] = []
    for fields in _table_rows(name, text):
        if not fields:
            continue
        month = _month(fields[0])
        if month is None:
            if months:
                break
            continue
        if month <= cutoff:
            months.append(month)
            rows.append(_numeric(fields[1:]))
    if not months or months[-1] != cutoff:
        raise ContractError("archive does not reach the exact cutoff")
    return ReturnPanel(family, np.asarray(months), np.asarray(rows))
