import numpy as np

from wharton_lab.models.m12 import M12Model, M12Config


def test_m12_smoke():
    m = M12Model(M12Config(n_assets=3, latent_dim=3))
    out = m.smoke_forward(n_features=9, n_samples=40)
    assert out["finite"]


def test_market_forecast_invariant_to_liability():
    rng = np.random.default_rng(0)
    X = rng.normal(scale=0.01, size=(30, 9))
    y = rng.normal(scale=0.01, size=30)
    m = M12Model(M12Config(n_assets=3))
    m.fit(X, y)
    p1 = m.predict_market(X[:5])
    H, k = 5, 3
    scen = rng.normal(scale=0.01, size=(8, H, k))
    liab_a = np.full(H, 0.03)
    liab_b = np.full(H, 0.20)
    m.plan(scen, liab_a, cash=1.0)
    p2 = m.predict_market(X[:5])
    m.plan(scen, liab_b, cash=1.0)
    p3 = m.predict_market(X[:5])
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(p2, p3)


def test_ablation_no_liability():
    m = M12Model(M12Config(no_liability=True, n_assets=4))
    scen = np.zeros((4, 5, 4))
    w = m.plan(scen, np.ones(5))
    np.testing.assert_allclose(w, 0.25)
