import numpy as np

from wharton_lab.models.m10 import M10Model, M10Config
from wharton_lab.models.m10.flow import bridge_sample, bridge_velocity, euler_integrate
from wharton_lab.evaluation.metrics import energy_score


def test_flow_bridge():
    z = np.array([0.0, 0.0])
    y = np.array([1.0, 2.0])
    mid = bridge_sample(z, y, 0.5)
    np.testing.assert_allclose(mid, [0.5, 1.0])
    np.testing.assert_allclose(bridge_velocity(z, y), [1.0, 2.0])


def test_m10_smoke_and_energy():
    m = M10Model(M10Config(n_assets=3, path_steps=4, n_ode_steps=5))
    out = m.smoke_forward(n_features=6)
    assert out["path_shape"] == (4, 4, 3)
    assert out["finite"]
    samples = np.random.default_rng(0).normal(size=(5, 20, 3))
    y = np.random.default_rng(1).normal(size=(5, 3))
    es = energy_score(samples, y)
    assert np.isfinite(es)


def test_student_t_noise():
    m = M10Model(M10Config(student_t_df=4.0, n_assets=2, path_steps=2))
    out = m.smoke_forward()
    assert out["finite"]
