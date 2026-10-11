import numpy as np
import pytest

from wharton_lab.audit.leaky import LeakyFeatureError, assert_no_feature_leak


def test_reject_leaky_fixture():
    rng = np.random.default_rng(7)
    y = rng.normal(size=50)
    leaky_features = y.copy()
    with pytest.raises(LeakyFeatureError):
        assert_no_feature_leak(leaky_features, y)


def test_accept_lagged_fixture():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    safe = np.roll(y, 1)
    safe[0] = 0.0
    assert_no_feature_leak(safe, y)
