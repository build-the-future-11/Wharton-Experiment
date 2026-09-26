from datetime import datetime, timedelta

import numpy as np

from wharton_lab.causal.maturity_gated import MaturityGatedLinearForecaster, decision_time_series


def test_future_label_change_does_not_change_earlier_prediction():
    rng = np.random.default_rng(11)
    n = 40
    start = datetime(2020, 1, 1)
    times = decision_time_series(start, n, step_days=1)
    X = rng.normal(size=(n, 3))
    true_w = np.array([0.5, -0.2, 0.1])
    y = X @ true_w + rng.normal(scale=0.05, size=n)

    as_of = start + timedelta(days=25)
    model = MaturityGatedLinearForecaster(horizon_days=5, reporting_lag_days=0)
    model.fit(X, y, times, as_of)
    x_query = X[10:12]
    pred_before = model.predict(x_query)

    y_mutated = y.copy()
    y_mutated[30:] = y_mutated[30:] + 100.0
    model2 = MaturityGatedLinearForecaster(horizon_days=5, reporting_lag_days=0)
    model2.fit(X, y_mutated, times, as_of)
    pred_after = model2.predict(x_query)

    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-5, atol=1e-5)
