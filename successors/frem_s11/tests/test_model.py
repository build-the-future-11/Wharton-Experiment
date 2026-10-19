import numpy as np

from frem_s11.model import EXPERTS, EnsembleConfig, issue_forecast
from frem_s11.scores import energy_score


def _returns(length: int = 180) -> np.ndarray:
    rng = np.random.default_rng(11)
    common = rng.normal(0.006, 0.035, size=(length, 1))
    style = rng.normal(0, 0.018, size=(length, 6))
    values = common + style
    values[120:132] -= 0.04
    return values


def test_specialists_are_finite_distinct_and_causal() -> None:
    values = _returns()
    bundle = issue_forecast(values[:150])
    assert len(bundle.specialists) == len(EXPERTS)
    assert np.isclose(bundle.stack_weights.sum(), 1)
    assert len({law.probabilities.tobytes() for law in bundle.specialists}) == len(EXPERTS)
    changed = values.copy()
    changed[150:] = 1000
    other = issue_forecast(changed[:150])
    np.testing.assert_array_equal(bundle.stacked.atoms, other.stacked.atoms)
    np.testing.assert_array_equal(bundle.stacked.probabilities, other.stacked.probabilities)


def test_delayed_scores_change_stack_without_exceeding_cap() -> None:
    values = _returns()
    baseline = issue_forecast(values[:150])
    losses = np.tile(np.array([0.1, 1.0, 1.2, 1.4]), (48, 1))
    fitted = issue_forecast(values[:150], loss_history=losses, config=EnsembleConfig())
    assert fitted.stack_weights[0] > baseline.stack_weights[0]
    assert fitted.stack_weights.max() <= 0.60 + 1e-12
    assert energy_score(fitted.stacked, values[150]) >= 0


def test_value_state_is_high_minus_low_book_to_market() -> None:
    values = np.zeros((120, 6), dtype=np.float64)
    values[:, [2, 5]] = 0.02
    bundle = issue_forecast(values)
    np.testing.assert_allclose(bundle.current_state[4], 0.02)
