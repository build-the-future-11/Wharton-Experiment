import numpy as np
import pytest
from wharton_lab.research.causal_portfolio import make_dataset, WORLDS, point_features, windows, future_paths, panel_features, energy_losses

@pytest.mark.parametrize('world',WORLDS)
def test_horizon_purge_and_future_suffix_noninterference(world):
    d=make_dataset(world,431)
    assert d.train_origins[-1]+d.horizon-1<d.test_origins[0]
    r=d.normalized.copy(); altered=r.copy(); altered[260:]+=10000
    origins=np.arange(240,261)
    np.testing.assert_array_equal(point_features(r,origins),point_features(altered,origins))
    np.testing.assert_array_equal(windows(r,origins,20),windows(altered,origins,20))
    for a,b in zip(panel_features(r,origins),panel_features(altered,origins)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(future_paths(r,d.train_origins,5),future_paths(altered,d.train_origins,5))


def test_energy_score_dirac_and_dimension_guard():
    truth=np.array([[1.,2.]])
    samples=np.repeat(truth[:,None,:],4,axis=1)
    assert energy_losses(samples,truth)[0]==pytest.approx(0)
    assert energy_losses(samples,truth+1)[0]==pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError): energy_losses(samples,np.zeros((1,3)))
