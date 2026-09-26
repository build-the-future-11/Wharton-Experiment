"""Generate protocol/EXPERIMENT_MANIFEST.csv (idempotent merge)."""

from __future__ import annotations

from pathlib import Path

from wharton_lab.orchestration.manifest import ManifestRow, load_manifest, manifest_path, write_manifest
from wharton_lab.orchestration.profiles import ProfileName, get_profile

MODELS = [f"M{i:02d}" for i in range(1, 13)]
BASELINES = ["B_RIDGE", "B_PERSISTENCE", "B_EWMA_COV"]
DATASETS = ["synthetic_track_c", "etf_track_a"]

ABLATIONS: list[tuple[str, str, str]] = [
    ("M01", "ablation_no_regime", "synthetic_track_c"),
    ("M01", "ablation_soft_regime", "synthetic_track_c"),
    ("M04", "ablation_mlp_backbone", "synthetic_track_c"),
    ("M05", "ablation_linear_only", "synthetic_track_c"),
    ("M12", "ablation_no_graph", "synthetic_track_c"),
    ("M12", "ablation_no_memory", "synthetic_track_c"),
    ("M12", "ablation_no_liability", "synthetic_track_c"),
]

METRICS = {
    "M01": "pinball_loss",
    "M02": "spearman_ic",
    "M03": "mse",
    "M04": "mse",
    "M05": "mse",
    "M06": "latent_mse",
    "M07": "cov_forecast_mse",
    "M08": "graph_latent_mse",
    "M09": "mse",
    "M10": "energy_score_proxy_mse",
    "M11": "funding_shortfall",
    "M12": "market_mse",
    "B_RIDGE": "mse",
    "B_PERSISTENCE": "mse",
    "B_EWMA_COV": "frobenius_cov_error",
}


def _rows_for_profile(profile: ProfileName) -> list[ManifestRow]:
    prof = get_profile(profile)
    key = profile.value.lower()
    out: list[ManifestRow] = []
    seeds = prof.seeds
    folds = range(prof.max_folds) if profile != ProfileName.SMOKE else (0,)

    def add(model_id: str, variant: str, dataset: str, seed: int, fold: int) -> None:
        run_id = f"{key}_{model_id}_{variant}_{dataset}_f{fold}_s{seed}_h{prof.primary_horizon}"
        out.append(
            ManifestRow(
                run_id=run_id,
                model_id=model_id,
                variant=variant,
                dataset=dataset,
                fold=fold,
                seed=seed,
                horizon=prof.primary_horizon,
                profile=key,
                config_hash="",
                dependencies="",
                primary_metric=METRICS.get(model_id, "mse"),
                budget="smoke_tiny" if profile == ProfileName.SMOKE else profile.value.lower(),
                status="PENDING",
                result_path="",
                failure_reason="",
            )
        )

    for seed in seeds:
        for fold in folds:
            for model_id in MODELS:
                for dataset in DATASETS if profile == ProfileName.SMOKE else DATASETS:
                    add(model_id, "default", dataset, seed, fold)
            for baseline in BASELINES:
                add(baseline, "default", "synthetic_track_c", seed, fold)
                if profile != ProfileName.SMOKE:
                    add(baseline, "default", "etf_track_a", seed, fold)

    if profile in (ProfileName.SMOKE, ProfileName.PILOT):
        for model_id, variant, dataset in ABLATIONS:
            seed = seeds[0]
            add(model_id, variant, dataset, seed, 0)

    return out


def bootstrap(repo_root: Path | str) -> None:
    root = Path(repo_root)
    path = manifest_path(root)
    existing = {r.run_id: r for r in load_manifest(path)}
    for profile in ProfileName:
        for row in _rows_for_profile(profile):
            if row.run_id not in existing:
                existing[row.run_id] = row
    write_manifest(path, sorted(existing.values(), key=lambda r: r.run_id))


if __name__ == "__main__":
    bootstrap(Path(__file__).resolve().parents[3])
