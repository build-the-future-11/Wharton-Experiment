"""Repaired-pipeline exploratory evaluation (post-lockbox; NOT confirmatory).

Causal features (``data/causal_tracks.py``), disjoint expanding-window folds, 1-step
daily label. ETF test windows exclude the spent lockbox rows; synthetic uses the frozen
full seeds only (never the lockbox seed). Comparators and the primary contrast are fixed
in ``PRIMARY`` / ``ARMS`` below before any result is produced.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy import stats

from wharton_lab.backtesting.walk_forward import iter_walk_forward
from wharton_lab.baselines.ridge import ridge_forecast
from wharton_lab.data.causal_tracks import (
    FEATURE_NAMES,
    CausalPanel,
    causal_return_features,
    causal_synthetic_panel,
    legacy_synthetic_panel,
)
from wharton_lab.evaluation.multiplicity import benjamini_hochberg
from wharton_lab.orchestration.lockbox import LOCKBOX_ETF_OFFSET, LOCKBOX_N_STEPS, LOCKBOX_SEED

FULL_SEEDS: tuple[int, ...] = (11, 23, 47, 89, 131)
SYNTH_N_STEPS = 1200
ETF_CACHE = Path("data/raw/etf/SPY_AGG_GLD_VNQ_EFA_returns.npy")
FOLD_KW = {"n_folds": 8, "min_train": 300, "test_size": 100}

PRIMARY = ("ridge_std", "mean")
CHALLENGERS = ("ridge_legacy", "ridge_std")
BENCHMARKS = ("zero", "mean", "persistence_1step", "static_last")


def _ridge_std(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z_tr, Z_te = (X_tr - mu) / sd, (X_te - mu) / sd
    y_mu = y_tr.mean()
    return y_mu + ridge_forecast(Z_tr, y_tr - y_mu, Z_te, alpha=1.0)


ARMS: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = {
    "zero": lambda X_tr, y_tr, X_te: np.zeros(len(X_te)),
    "mean": lambda X_tr, y_tr, X_te: np.full(len(X_te), y_tr.mean()),
    "persistence_1step": lambda X_tr, y_tr, X_te: X_te[:, FEATURE_NAMES.index("lag1")].copy(),
    "static_last": lambda X_tr, y_tr, X_te: np.full(len(X_te), y_tr[-1]),
    "ridge_legacy": lambda X_tr, y_tr, X_te: ridge_forecast(X_tr, y_tr, X_te),
    "ridge_std": _ridge_std,
}


@dataclass(frozen=True)
class Contrast:
    dataset: str
    challenger: str
    benchmark: str
    n_obs: int
    n_folds: int
    mse_challenger: float
    mse_benchmark: float
    oos_r2_vs_benchmark: float
    dm_stat: float
    dm_p: float
    fold_wilcoxon_p: float | None
    bh_q: float = float("nan")


def diebold_mariano(d: np.ndarray, lag: int | None = None) -> tuple[float, float]:
    """DM test on loss differential ``d`` with Newey–West variance; two-sided normal p."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2 / 9)))
    dc = d - d.mean()
    var = float(dc @ dc) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * float(dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))


def _eval_panel(
    panel: CausalPanel,
    *,
    exclude_rows: tuple[int, int] | None = None,
    extra_arms: dict[str, tuple[CausalPanel, Callable]] | None = None,
) -> dict[str, Any]:
    folds_out = []
    errors: dict[str, list[np.ndarray]] = {}
    excluded = []
    for f in iter_walk_forward(len(panel.y), **FOLD_KW):
        t_rows = panel.row_t[f.test_idx]
        if exclude_rows and (t_rows.max() >= exclude_rows[0]) and (t_rows.min() < exclude_rows[1]):
            excluded.append({"fold": f.fold, "rows": [int(t_rows.min()), int(t_rows.max())]})
            continue
        X_tr, y_tr = panel.X[f.train_idx], panel.y[f.train_idx]
        X_te, y_te = panel.X[f.test_idx], panel.y[f.test_idx]
        rec: dict[str, Any] = {"fold": f.fold, "rows": [int(t_rows.min()), int(t_rows.max())]}
        for name, fn in ARMS.items():
            e = y_te - fn(X_tr, y_tr, X_te)
            errors.setdefault(name, []).append(e)
            rec[name] = float(np.mean(e**2))
        for name, (alt, fn) in (extra_arms or {}).items():
            pos = {int(t): i for i, t in enumerate(alt.row_t)}
            tr = np.array([pos[int(t)] for t in panel.row_t[f.train_idx]])
            te = np.array([pos[int(t)] for t in t_rows])
            e = alt.y[te] - fn(alt.X[tr], alt.y[tr], alt.X[te])
            errors.setdefault(name, []).append(e)
            rec[name] = float(np.mean(e**2))
        folds_out.append(rec)
    return {"folds": folds_out, "excluded_folds": excluded, "errors": errors}


def _contrasts(dataset: str, res: dict[str, Any]) -> list[Contrast]:
    out = []
    errs = res["errors"]
    challengers = [c for c in CHALLENGERS if c in errs] + [
        c for c in errs if c not in ARMS and c not in BENCHMARKS
    ]
    for ch in challengers:
        for bm in BENCHMARKS:
            e_c = np.concatenate(errs[ch])
            e_b = np.concatenate(errs[bm])
            d = e_c**2 - e_b**2
            stat, p = diebold_mariano(d)
            fold_d = np.array([np.mean(a**2) - np.mean(b**2) for a, b in zip(errs[ch], errs[bm])])
            wp = None
            if len(fold_d) >= 6 and not np.allclose(fold_d, 0):
                wp = float(stats.wilcoxon(fold_d, alternative="two-sided").pvalue)
            mse_c, mse_b = float(np.mean(e_c**2)), float(np.mean(e_b**2))
            out.append(
                Contrast(
                    dataset=dataset,
                    challenger=ch,
                    benchmark=bm,
                    n_obs=len(d),
                    n_folds=len(fold_d),
                    mse_challenger=mse_c,
                    mse_benchmark=mse_b,
                    oos_r2_vs_benchmark=1.0 - mse_c / mse_b if mse_b > 0 else float("nan"),
                    dm_stat=stat,
                    dm_p=p,
                    fold_wilcoxon_p=wp,
                )
            )
    return out


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def run_repaired_eval(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    etf_path = root / ETF_CACHE
    if not etf_path.exists():
        raise FileNotFoundError(f"{etf_path} missing — ETF cache is not versioned; see DATA_MANIFEST")
    etf_rets = np.asarray(np.load(etf_path), dtype=float)
    etf_panel = causal_return_features(etf_rets, source="etf_cache")
    lock_rows = (LOCKBOX_ETF_OFFSET, LOCKBOX_ETF_OFFSET + LOCKBOX_N_STEPS)

    datasets: dict[str, dict[str, Any]] = {}
    contrasts: list[Contrast] = []

    r = _eval_panel(etf_panel, exclude_rows=lock_rows)
    datasets["etf_track_a"] = {"folds": r["folds"], "excluded_folds": r["excluded_folds"]}
    contrasts += _contrasts("etf_track_a", r)

    for seed in FULL_SEEDS:
        assert seed != LOCKBOX_SEED
        panel = causal_synthetic_panel(seed, SYNTH_N_STEPS)
        leaky = legacy_synthetic_panel(seed, SYNTH_N_STEPS)
        r = _eval_panel(
            panel, extra_arms={"ridge_legacy_on_leaky_features": (leaky, ARMS["ridge_legacy"])}
        )
        name = f"synthetic_signal_s{seed}"
        datasets[name] = {"folds": r["folds"], "excluded_folds": r["excluded_folds"]}
        contrasts += _contrasts(name, r)

    # BH family = all non-leakage-demo contrasts
    family = [i for i, c in enumerate(contrasts) if c.challenger in CHALLENGERS]
    qs = benjamini_hochberg([contrasts[i].dm_p for i in family])
    contrasts = [
        Contrast(**{**c.__dict__, "bh_q": qs[family.index(i)]}) if i in family else c
        for i, c in enumerate(contrasts)
    ]

    payload = {
        "generation": "repaired_pipeline_exploratory_post_lockbox",
        "confirmatory": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(root),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "label": "1-step daily return of asset 0",
            "features": list(FEATURE_NAMES),
            "folds": FOLD_KW,
            "synthetic_seeds": list(FULL_SEEDS),
            "synthetic_n_steps": SYNTH_N_STEPS,
            "etf_cache": str(ETF_CACHE),
            "etf_cache_sha256": _sha256(etf_path),
            "etf_rows": int(etf_rets.shape[0]),
            "etf_lockbox_rows_excluded_from_test": list(lock_rows),
            "primary_contrast": list(PRIMARY),
            "bh_family": "all challenger x benchmark x dataset DM p-values (leakage demo excluded)",
        },
        "datasets": datasets,
        "contrasts": [c.__dict__ for c in contrasts],
    }
    out_dir = root / "runs" / "repaired"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPAIRED_RESULTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_summary(root, payload, contrasts)
    return payload


def _write_summary(root: Path, payload: dict[str, Any], contrasts: list[Contrast]) -> None:
    cfg = payload["config"]
    lines = [
        "# Repaired-pipeline exploratory evaluation",
        "",
        "**Generation:** repaired pipeline, exploratory, post-lockbox. **Not confirmatory**; "
        "the lockbox is spent and is not reopened.",
        "",
        f"- Label: {cfg['label']}; causal features {', '.join(cfg['features'])} (rows < t only).",
        f"- Folds: expanding window, {cfg['folds']['n_folds']} × {cfg['folds']['test_size']} "
        f"disjoint test rows, min train {cfg['folds']['min_train']}.",
        f"- ETF: `{cfg['etf_cache']}` ({cfg['etf_rows']} rows, sha256 `{cfg['etf_cache_sha256']}`); "
        f"test folds overlapping lockbox rows {cfg['etf_lockbox_rows_excluded_from_test']} dropped.",
        f"- Synthetic: SIGNAL world, seeds {cfg['synthetic_seeds']} (lockbox seed excluded), "
        f"{cfg['synthetic_n_steps']} steps.",
        f"- Primary contrast (pre-declared): `{cfg['primary_contrast'][0]}` vs "
        f"`{cfg['primary_contrast'][1]}`.",
        f"- Git HEAD at run: `{payload['git_head']}`.",
        "",
        "OOS R² = 1 − MSE(challenger)/MSE(benchmark); positive = challenger better. "
        "DM = Diebold–Mariano (Newey–West); q = Benjamini–Hochberg over the challenger family.",
        "",
    ]
    for ds in payload["datasets"]:
        excl = payload["datasets"][ds]["excluded_folds"]
        lines.append(f"## {ds}")
        if excl:
            lines.append(f"Excluded folds (lockbox overlap): {excl}")
        lines += [
            "",
            "| challenger | benchmark | n | MSE chall. | MSE bench. | OOS R² | DM p | BH q | fold Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in contrasts:
            if c.dataset != ds:
                continue
            q = "—" if np.isnan(c.bh_q) else f"{c.bh_q:.3g}"
            wp = "—" if c.fold_wilcoxon_p is None else f"{c.fold_wilcoxon_p:.3g}"
            lines.append(
                f"| {c.challenger} | {c.benchmark} | {c.n_obs} | {c.mse_challenger:.4e} | "
                f"{c.mse_benchmark:.4e} | {c.oos_r2_vs_benchmark:+.4f} | {c.dm_p:.3g} | {q} | {wp} |"
            )
        lines.append("")
    lines += [
        "`ridge_legacy_on_leaky_features` re-fits the frozen B_RIDGE on the legacy Track C "
        "features `[latent_t, latent_{t-1}]` over the same test rows. It is a **leakage "
        "demonstration**, not a forecast, and is excluded from the BH family.",
        "",
    ]
    out = root / "reports" / "evidence_bank" / "REPAIRED_EVAL_SUMMARY.md"
    out.write_text("\n".join(lines))
