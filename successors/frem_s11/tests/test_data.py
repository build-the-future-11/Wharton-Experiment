import csv
import io
import zipfile

import numpy as np
import pytest

from frem_s11.data import ASSETS, ReturnPanel, parse_french_archive
from frem_s11.errors import DataContractError


def _archive(rows: int = 36, missing: bool = False) -> bytes:
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["release-vintage generated fixture"])
    writer.writerow(["", "SMALL LoBM", "ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM"])
    year, month = 2000, 1
    for index in range(rows):
        value = -99.99 if missing and index == 2 else 1.0 + index / 100
        writer.writerow([year * 100 + month, value, 2, 3, 4, 5, 6])
        month += 1
        if month == 13:
            year, month = year + 1, 1
    writer.writerow([])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("6_Portfolios_2x3.csv", stream.getvalue())
    return output.getvalue()


def _txt_archive(rows: int = 36) -> bytes:
    lines = [
        "release-vintage generated fixture",
        "            SMALL LoBM    ME1 BM2    SMALL HiBM    BIG LoBM    ME2 BM2    BIG HiBM",
    ]
    year, month = 2000, 1
    for index in range(rows):
        values = [1.0 + index / 100, 2, 3, 4, 5, 6]
        lines.append(f"{year * 100 + month} " + " ".join(str(value) for value in values))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    lines.append("")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("6_Portfolios_2x3.txt", "\n".join(lines))
    return output.getvalue()


def test_parser_reads_exact_monthly_table_and_decimal_units() -> None:
    panel = parse_french_archive(_archive(), release_month=200401, end_inclusive=200212)
    assert panel.returns.shape == (36, len(ASSETS))
    assert panel.months[0] == 200001 and panel.months[-1] == 200212
    np.testing.assert_allclose(panel.returns[0], np.arange(1, 7) / 100)
    assert not panel.returns.flags.writeable
    assert ASSETS == (
        "small_growth",
        "small_neutral",
        "small_value",
        "big_growth",
        "big_neutral",
        "big_value",
    )


def test_official_txt_shape_matches_csv_semantics() -> None:
    csv_panel = parse_french_archive(_archive(), release_month=200401, end_inclusive=200212)
    txt_panel = parse_french_archive(_txt_archive(), release_month=200401, end_inclusive=200212)
    np.testing.assert_array_equal(txt_panel.months, csv_panel.months)
    np.testing.assert_allclose(txt_panel.returns, csv_panel.returns)


def test_parser_rejects_missing_values_and_unavailable_cutoff() -> None:
    with pytest.raises(DataContractError, match="missing"):
        parse_french_archive(_archive(missing=True), release_month=200401, end_inclusive=200212)
    with pytest.raises(DataContractError, match="cutoff"):
        parse_french_archive(_archive(), release_month=200501, end_inclusive=200312)


def test_panel_rejects_month_gaps_and_observations_after_release() -> None:
    values = np.zeros((24, 6))
    digest = "0" * 64
    gap = np.asarray(
        [2000 * 100 + month for month in range(1, 13)]
        + [2001 * 100 + month for month in range(2, 13)]
        + [200201]
    )
    with pytest.raises(DataContractError, match="contiguous"):
        ReturnPanel(gap, values, 200301, digest)
    contiguous = np.asarray(
        [2000 * 100 + month for month in range(1, 13)] + [2001 * 100 + month for month in range(1, 13)]
    )
    with pytest.raises(DataContractError, match="predate"):
        ReturnPanel(contiguous, values, 200112, digest)
