from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from frem_s12.data import ReturnPanel, parse_archive
from frem_s12.errors import ContractError


def archive(text: str, name: str = "six.csv") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(name, text)
    return output.getvalue()


def rows(start: int, count: int) -> str:
    year, month = divmod(start, 100)
    result = ["header"]
    for _index in range(count):
        current = year * 100 + month
        result.append(f"{current},1,2,3,4,5,6")
        month += 1
        if month == 13:
            year += 1
            month = 1
    result.append("annual section")
    return "\n".join(result)


def test_parse_csv_archive_and_percent_scale() -> None:
    panel = parse_archive(archive(rows(200001, 24)), family="x", cutoff=200112)
    assert panel.returns.shape == (24, 6)
    assert panel.returns[0, 0] == pytest.approx(0.01)


def test_parse_text_archive() -> None:
    text = rows(200001, 24).replace(",", " ")
    panel = parse_archive(archive(text, "six.txt"), family="x", cutoff=200112)
    assert panel.months[-1] == 200112


def test_missing_sentinel_rejected() -> None:
    with pytest.raises(ContractError, match="missing"):
        parse_archive(archive(rows(200001, 24).replace("1,2,3", "-99.99,2,3", 1)), family="x", cutoff=200112)


def test_calendar_gap_rejected() -> None:
    text = rows(200001, 24).replace("200006,", "200007,", 1)
    with pytest.raises(ContractError):
        parse_archive(archive(text), family="x", cutoff=200112)


def test_return_panel_owns_input() -> None:
    months = np.arange(200001, 200025)
    with pytest.raises(ContractError):
        ReturnPanel("x", months, np.zeros((24, 6)))
