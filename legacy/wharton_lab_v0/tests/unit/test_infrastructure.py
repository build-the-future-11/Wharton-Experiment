import numpy as np

from wharton_lab.data.synthetic import SyntheticWorldConfig, WorldKind, generate_world
from wharton_lab.data.etf_track import load_etf_track, write_data_manifest
from wharton_lab.splits.walk_forward import walk_forward_folds
from wharton_lab.splits.clock import InformationClock
from datetime import datetime, timedelta
from wharton_lab.evaluation.metrics import pinball_loss, spearman_ic
from wharton_lab.orchestration.runner import ExperimentRunner, TaskSpec
from wharton_lab.orchestration.profiles import get_profile, ProfileName


def test_synthetic_worlds():
    for kind in WorldKind:
        w = generate_world(SyntheticWorldConfig(kind=kind, n_steps=50, seed=1))
        assert w.returns.shape[0] == 50


def test_etf_manifest(tmp_path):
    track = load_etf_track(cache_dir=tmp_path / "etf")
    manifest = write_data_manifest(track, tmp_path / "manifest.yaml")
    assert manifest.exists()


def test_walk_forward_and_clock():
    folds = walk_forward_folds(200, min_train=50, test_size=10, purge_gap=2, lockbox_size=20)
    assert len(folds) >= 2
    assert folds[-1].is_lockbox
    clk = InformationClock(datetime(2020, 1, 1), horizon_days=5, reporting_lag_days=2)
    assert clk.label_maturity == datetime(2020, 1, 1) + timedelta(days=7)


def test_metrics_and_runner(tmp_path):
    y = np.array([0.0, 1.0, 2.0])
    q = np.array([0.1, 0.9, 1.8])
    assert pinball_loss(y, q, 0.5) >= 0
    ic = spearman_ic(y, q)
    assert np.isfinite(ic)
    runner = ExperimentRunner(tmp_path)
    spec = TaskSpec(task_id=TaskSpec.deterministic_id("t", {"a": 1}), name="t")
    r1 = runner.run_task(spec, lambda: {"x": 1})
    r2 = runner.run_task(spec, lambda: {"x": 2})
    assert r1.status == "ok"
    assert r2.output["x"] == 1
    prof = get_profile(ProfileName.SMOKE)
    assert prof.seeds == (11,)
