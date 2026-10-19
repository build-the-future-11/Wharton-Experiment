import hashlib

import numpy as np

from frem_s11.backtest import VARIANTS, run_backtest
from frem_s11.data import ReturnPanel


def _panel() -> ReturnPanel:
    rng = np.random.default_rng(29)
    length = 264
    common = rng.normal(0.005, 0.03, size=(length, 1))
    returns = common + rng.normal(0, 0.015, size=(length, 6))
    months = []
    year, month = 1990, 1
    for _ in range(length):
        months.append(year * 100 + month)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return ReturnPanel(np.asarray(months), returns, 201301, hashlib.sha256(b"fixture").hexdigest())


def test_generated_backtest_retains_all_variants_and_future_invariance() -> None:
    panel = _panel()
    result = run_backtest(panel, start_month=200001, end_month=200912)
    assert result.scores.shape == (120, len(VARIANTS))
    assert result.allocations.shape == (120, len(VARIANTS), 7)
    assert result.wealth.shape == result.scores.shape
    assert result.benchmark_wealth.shape == (120, 2)
    assert set(result.intervals) == {"equal_minus_stacked", "recent_minus_stacked"}
    assert set(result.relative_improvements) == {"equal", "recent"}
    assert isinstance(result.primary_supported, bool)
    assert isinstance(result.secondary_supported, bool)
    assert not result.scores.flags.writeable
    changed = np.array(panel.returns, copy=True)
    origin = int(np.flatnonzero(panel.months == 200501)[0])
    changed[origin:] += 0.5
    altered = ReturnPanel(panel.months, changed, panel.release_month, hashlib.sha256(b"changed").hexdigest())
    before = run_backtest(panel, start_month=200001, end_month=200412)
    after = run_backtest(altered, start_month=200001, end_month=200412)
    np.testing.assert_array_equal(before.scores, after.scores)
