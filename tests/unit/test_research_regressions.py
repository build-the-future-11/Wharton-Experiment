"""Adversarial scientific regressions added after the spent legacy lockbox."""
import numpy as np
import pytest
from wharton_lab.models.m09 import M09Model, M09Config
from wharton_lab.models.m10 import M10Model, M10Config
from wharton_lab.models.m12 import M12Model, M12Config
from wharton_lab.models.m07 import EigenJEPAModel
from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m06 import FIJEPAModel
from wharton_lab.models.m06.config import M06Config


def test_m09_one_step_ridge_uses_labels():
    rng = np.random.default_rng(811)
    X = rng.normal(scale=.01, size=(40, 4)); y = 2 * X[:, 0] + .1
    a = M09Model(M09Config(groupdro_steps=1)).fit(X, y)
    b = M09Model(M09Config(groupdro_steps=1)).fit(X, -y)
    assert np.linalg.norm(a.coef_) > .01
    np.testing.assert_allclose(a.coef_, -b.coef_, atol=1e-10)


def test_m07_covariance_loss_reaches_encoder():
    rng = np.random.default_rng(811)
    X = rng.normal(size=(12, 4, 2))
    model = EigenJEPAModel(M07Config(windows=(4,), n_assets=2, epochs=1, device='cpu'))
    before = {k: v.detach().clone() for k, v in model.encoders.state_dict().items()}
    future = np.repeat(np.eye(2)[None], len(X), axis=0)
    model.fit(X, np.zeros(len(X)), future_covariances={4: future})
    assert any(not np.array_equal(v.detach().numpy(), before[k].numpy())
               for k, v in model.encoders.state_dict().items())


def test_m06_holdout_does_not_change_training():
    rng = np.random.default_rng(811)
    X = rng.normal(size=(20, 4, 2)); y = rng.normal(size=20)
    cfg = M06Config(seq_len=4, input_dim=2, epochs=1, batch_size=16,
                    holdout_future_frac=.2, device='cpu')
    a = FIJEPAModel(cfg).fit(X, y)
    changed = X.copy(); changed[16:] += 100
    b = FIJEPAModel(cfg).fit(changed, y)
    for key, val in a.context.state_dict().items():
        np.testing.assert_array_equal(val.detach().numpy(), b.context.state_dict()[key].detach().numpy())


def test_m10_no_fabricated_future_dimensions():
    rng = np.random.default_rng(811)
    X = rng.normal(size=(16, 4)); y = rng.normal(size=(16, 2))
    m = M10Model(M10Config(n_assets=2, path_steps=3)).fit(X, y)
    with pytest.raises(ValueError, match='path|dimension'):
        m.sample_paths(X[:1], 4)


def test_m10_rng_continuation_after_save(tmp_path):
    rng = np.random.default_rng(811)
    X = rng.normal(size=(16, 4)); y = rng.normal(size=(16, 4))
    m = M10Model(M10Config(n_assets=2, path_steps=2)).fit(X, y)
    m.sample_paths(X[:1], 3)
    path = tmp_path/'flow.pkl'; m.save(path); loaded = M10Model.load(path)
    np.testing.assert_array_equal(m.sample_paths(X[:1], 3), loaded.sample_paths(X[:1], 3))


def test_m12_nondefault_serialization(tmp_path):
    rng = np.random.default_rng(811)
    X = rng.normal(scale=.01, size=(20, 10)); y = rng.normal(size=20)
    m = M12Model(M12Config(latent_dim=3, memory_slots=6, random_state=17)).fit(X, y)
    path = tmp_path/'stack.pkl'; m.save(path); loaded = M12Model.load(path)
    np.testing.assert_allclose(m.predict(X[:5]), loaded.predict(X[:5]), atol=1e-12)


def test_m05_expert_uncertainty_does_not_cancel():
    from wharton_lab.models.m05.model import QAPENModel
    m=QAPENModel()
    weights=np.array([[.5,.5]]); predictions=np.array([[0.,10.]])
    a=m._aggregate(weights,predictions,np.array([[.01,10000.]]))
    b=m._aggregate(weights,predictions,np.array([[10000.,.01]]))
    assert a[0]<1 and b[0]>9


def test_m07_forward_predictions_condition_on_context():
    rng=np.random.default_rng(66); X=rng.normal(size=(15,4,2))
    targets=np.array([np.eye(2)*(1+abs(x[0,0])) for x in X])
    m=EigenJEPAModel(M07Config(windows=(4,),n_assets=2,epochs=2,device='cpu')).fit(X,np.zeros(15),future_covariances={4:targets})
    p=m.predict_covariances(X,4)
    assert p.shape==(15,2,2) and np.linalg.eigvalsh(p).min()>0
    assert np.linalg.norm(p[0]-p[1])>1e-6


def test_m09_prediction_prefix_causality():
    rng=np.random.default_rng(23); X=rng.normal(size=(70,4)); y=rng.normal(size=70)
    m=M09Model().fit(X[:40],y[:40])
    a=m.predict(X[40:]); changed=X[40:].copy(); changed[10:]+=100
    np.testing.assert_array_equal(a[:10],m.predict(changed)[:10])


def test_m06_forward_targets_purged_from_internal_holdout():
    rng=np.random.default_rng(84); X=rng.normal(size=(30,4,2)); y=rng.normal(size=30)
    future={2:rng.normal(size=(30,2,2)),4:rng.normal(size=(30,4,2))}
    cfg=M06Config(seq_len=4,input_dim=2,horizons=(2,4),epochs=1,batch_size=32,holdout_future_frac=.2,device='cpu')
    a=FIJEPAModel(cfg).fit(X,y,future_windows=future)
    future2={h:v.copy() for h,v in future.items()}
    # split=24, maxh=4 => last three training origins must be purged.
    for value in future2.values(): value[21:]+=100
    b=FIJEPAModel(cfg).fit(X,y,future_windows=future2)
    for k,v in a.context.state_dict().items():
        np.testing.assert_array_equal(v.detach().numpy(),b.context.state_dict()[k].detach().numpy())
