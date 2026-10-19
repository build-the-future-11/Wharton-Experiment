"""Strict parser for release-vintage Fama/French 2x3 portfolio archives."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from frem_s11.errors import DataContractError

ASSETS = ("small_growth", "small_neutral", "small_value", "big_growth", "big_neutral", "big_value")
_HEADER = ("SMALL LoBM", "ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM")
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
_MAX_MEMBER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReturnPanel:
    """Immutable monthly total-return panel in decimal units."""

    months: NDArray[np.int64]
    returns: NDArray[np.float64]
    release_month: int
    source_sha256: str

    def __post_init__(self) -> None:
        """Own arrays and enforce the monthly-panel contract."""
        months = np.array(self.months, dtype=np.int64, copy=True)
        values = np.array(self.returns, dtype=np.float64, copy=True)
        if months.ndim != 1 or values.shape != (len(months), len(ASSETS)) or len(months) < 24:
            raise DataContractError("monthly panel shape is invalid")
        if not all(_valid_month(int(month)) for month in months):
            raise DataContractError("panel contains a malformed month")
        expected = np.where(months[:-1] % 100 == 12, (months[:-1] // 100 + 1) * 100 + 1, months[:-1] + 1)
        if len(np.unique(months)) != len(months) or np.any(months[1:] != expected):
            raise DataContractError("months must be unique, increasing, and contiguous")
        if not np.isfinite(values).all() or np.any(np.abs(values) > 2.0):
            raise DataContractError("returns must be finite decimals within the declared bound")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise DataContractError("source SHA-256 is malformed")
        if not _valid_month(self.release_month):
            raise DataContractError("release month is malformed")
        if months[-1] >= self.release_month:
            raise DataContractError("panel observations must predate the declared release")
        months.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "returns", values)

    def through(self, end_inclusive: int) -> ReturnPanel:
        """Return a cutoff view without inspecting or retaining later numeric rows."""
        if not _valid_month(end_inclusive):
            raise DataContractError("cutoff month is malformed")
        stop = int(np.searchsorted(self.months, end_inclusive, side="right"))
        if stop < 24 or self.months[stop - 1] != end_inclusive:
            raise DataContractError("cutoff leaves insufficient history")
        return ReturnPanel(self.months[:stop], self.returns[:stop], self.release_month, self.source_sha256)


def _valid_month(value: int) -> bool:
    return 190001 <= value <= 220012 and 1 <= value % 100 <= 12


def _member(archive: bytes) -> tuple[str, bytes]:
    if not archive or len(archive) > _MAX_ARCHIVE_BYTES:
        raise DataContractError("archive is empty or exceeds the byte limit")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            files = [item for item in bundle.infolist() if not item.is_dir()]
            if len(files) != 1:
                raise DataContractError("archive must contain exactly one regular file")
            item = files[0]
            if item.filename.startswith(("/", "\\")) or ".." in item.filename.replace("\\", "/").split("/"):
                raise DataContractError("archive member path is unsafe")
            if item.file_size > _MAX_MEMBER_BYTES or item.compress_size <= 0:
                raise DataContractError("archive member size is invalid")
            return item.filename, bundle.read(item)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise DataContractError("archive cannot be read") from exc


def _csv_monthly_rows(text: str, end_inclusive: int) -> tuple[list[int], list[list[float]]]:
    reader = csv.reader(io.StringIO(text))
    header_seen = False
    months: list[int] = []
    rows: list[list[float]] = []
    for raw in reader:
        fields = [item.strip() for item in raw]
        if not header_seen:
            header_seen = len(fields) >= 7 and tuple(fields[1:7]) == _HEADER
            continue
        if not fields or re.fullmatch(r"[0-9]{6}", fields[0]) is None:
            if months:
                break
            continue
        month = int(fields[0])
        if month > end_inclusive:
            break
        months.append(month)
        rows.append(_return_row(fields, month))
    if not header_seen or not rows or months[-1] != end_inclusive:
        raise DataContractError("declared monthly table or cutoff is unavailable")
    return months, rows


def _txt_monthly_rows(text: str, end_inclusive: int) -> tuple[list[int], list[list[float]]]:
    """Read the official whitespace-delimited historical TXT monthly table."""
    header_seen = False
    months: list[int] = []
    rows: list[list[float]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        normalized = " ".join(stripped.upper().split())
        if not header_seen:
            header_seen = all(token in normalized for token in ("SMALL LOBM", "ME1 BM2", "BIG HIBM"))
            continue
        fields = stripped.split()
        if not fields or re.fullmatch(r"[0-9]{6}", fields[0]) is None:
            if months:
                break
            continue
        month = int(fields[0])
        if month > end_inclusive:
            break
        months.append(month)
        rows.append(_return_row(fields, month))
    if not header_seen or not rows or months[-1] != end_inclusive:
        raise DataContractError("declared TXT monthly table or cutoff is unavailable")
    return months, rows


def _return_row(fields: list[str], month: int) -> list[float]:
    if not _valid_month(month) or len(fields) != 7:
        raise DataContractError("monthly row month or shape is invalid")
    try:
        values = [float(item) for item in fields[1:]]
    except ValueError as exc:
        raise DataContractError("monthly return is not numeric") from exc
    if any(value in (-99.99, -999.0) for value in values):
        raise DataContractError("monthly table contains a documented missing value")
    return [value / 100.0 for value in values]


def parse_french_archive(archive: bytes, *, release_month: int, end_inclusive: int) -> ReturnPanel:
    """Parse only the first monthly value-weighted table through a declared cutoff."""
    if not _valid_month(release_month) or not _valid_month(end_inclusive) or end_inclusive >= release_month:
        raise DataContractError("release/cutoff chronology is invalid")
    member_name, payload = _member(archive)
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataContractError("archive member is not UTF-8 text") from exc
    suffix = member_name.lower().rsplit(".", maxsplit=1)[-1]
    if suffix == "csv":
        months, rows = _csv_monthly_rows(text, end_inclusive)
    elif suffix == "txt":
        months, rows = _txt_monthly_rows(text, end_inclusive)
    else:
        raise DataContractError("archive member must use the declared CSV or TXT format")
    return ReturnPanel(
        np.asarray(months, dtype=np.int64),
        np.asarray(rows, dtype=np.float64),
        release_month,
        hashlib.sha256(archive).hexdigest(),
    )
