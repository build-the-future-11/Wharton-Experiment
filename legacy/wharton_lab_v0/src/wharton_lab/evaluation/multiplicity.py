"""Multiple-testing correction for paired model comparisons."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


MIN_UNIQUE_PAIRS = 6  # exact two-sided Wilcoxon cannot reach p < 0.05 with n <= 5

LEAKY_DATASETS = {
    "synthetic_track_c": "legacy features include contemporaneous latent_t (D-050)",
    "etf_track_a": "legacy expanding std includes the label r_t (D-051)",
}


@dataclass(frozen=True)
class PairTest:
    profile: str
    dataset: str
    challenger: str
    baseline: str
    n_raw_pairs: int
    n_unique_pairs: int
    mean_delta_mse: float
    testable: bool
    wilcoxon_p: float | None
    bh_q: float | None
    significant_at_0_05: bool


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    ranked = np.asarray(p_values, dtype=float)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def _load_mse_index(root: Path, profile: str) -> dict[tuple, float]:
    """Key: (dataset, fold, seed, horizon, model_id) -> primary mse."""
    out: dict[tuple, float] = {}
    manifest = root / "protocol" / "EXPERIMENT_MANIFEST.csv"
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r.get("status") != "COMPLETED":
            continue
        if r.get("profile") != profile:
            continue
        if r.get("primary_metric") != "mse":
            continue
        rp = root / r["result_path"]
        if not rp.exists():
            continue
        data = json.loads(rp.read_text())
        metrics = data.get("metrics") or {}
        val = metrics.get("primary_value")
        if not isinstance(val, (int, float)):
            continue
        key = (
            str(r["dataset"]),
            int(r["fold"]),
            int(r["seed"]),
            int(r["horizon"]),
            str(r["model_id"]),
        )
        out[key] = float(val)
    return out


def paired_vs_persistence(
    root: Path | str,
    *,
    profile: str = "full",
    challengers: tuple[str, ...] = ("B_RIDGE", "M03", "M04", "M05", "M09"),
    baseline: str = "B_PERSISTENCE",
    alpha: float = 0.05,
) -> list[PairTest]:
    root = Path(root)
    idx = _load_mse_index(root, profile)
    raw: list[tuple[str, str, str, int, list[float]]] = []
    for chall in challengers:
        for dataset in sorted({k[0] for k in idx}):
            pairs: list[tuple[float, float]] = []
            for (ds, fold, seed, horizon, mid), mse in idx.items():
                if ds != dataset or mid != chall:
                    continue
                base_key = (ds, fold, seed, horizon, baseline)
                if base_key not in idx:
                    continue
                pairs.append((mse, idx[base_key]))
            if not pairs:
                continue
            # Legacy folds share one split and ETF seeds share one data panel, so repeated
            # cells are exact copies; collapse them rather than count them as evidence.
            unique = sorted({(round(c, 15), round(b, 15)) for c, b in pairs})
            # negative delta => challenger better (lower MSE)
            raw.append((dataset, chall, baseline, len(pairs), [c - b for c, b in unique]))

    pvals: list[float] = []
    testable_idx: list[int] = []
    for i, (_, _, _, _, deltas) in enumerate(raw):
        arr = np.asarray(deltas, dtype=float)
        if len(arr) < MIN_UNIQUE_PAIRS:
            continue
        if np.allclose(arr, 0):
            p = 1.0
        else:
            p = float(stats.wilcoxon(arr, alternative="two-sided").pvalue)
        pvals.append(p)
        testable_idx.append(i)

    qvals = benjamini_hochberg(pvals)
    results: list[PairTest] = []
    for i, (dataset, chall, base, n_raw, deltas) in enumerate(raw):
        testable = i in testable_idx
        p = pvals[testable_idx.index(i)] if testable else None
        q = qvals[testable_idx.index(i)] if testable else None
        results.append(
            PairTest(
                profile=profile,
                dataset=dataset,
                challenger=chall,
                baseline=base,
                n_raw_pairs=n_raw,
                n_unique_pairs=len(deltas),
                mean_delta_mse=float(np.mean(deltas)),
                testable=testable,
                wilcoxon_p=p,
                bh_q=q,
                significant_at_0_05=bool(testable and q <= alpha and np.mean(deltas) < 0),
            )
        )
    return results


def write_multiplicity_report(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    tests = paired_vs_persistence(root)
    n_testable = sum(1 for t in tests if t.testable)
    payload = {
        "generation": "legacy_lean_matrix_full_profile (leakage-contaminated; see D-050..D-052)",
        "method": (
            "Wilcoxon signed-rank on paired MSE deltas vs B_PERSISTENCE after collapsing exact "
            "duplicate cells; BH FDR over testable comparisons"
        ),
        "profile": "full",
        "alpha": 0.05,
        "min_unique_pairs": MIN_UNIQUE_PAIRS,
        "leakage": LEAKY_DATASETS,
        "tests": [asdict(t) for t in tests],
        "n_testable": n_testable,
        "n_significant_better": sum(1 for t in tests if t.significant_at_0_05),
        "lockbox_note": (
            "Lockbox H1 remains a pre-registered single-shot claim and is NOT "
            "included in this BH family (already opened once)."
        ),
    }
    out_dir = root / "reports" / "evidence_bank"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MULTIPLICITY_RESULTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    lines = [
        "# Multiplicity-corrected comparisons (legacy full profile)",
        "",
        "**Generation:** legacy lean matrix — leakage-contaminated on both tracks "
        "(D-050, D-051); folds degenerate and ETF seeds identical (D-052). "
        "Not evidence of forecasting skill.",
        "",
        "Paired Wilcoxon on MSE(challenger) − MSE(persistence) after collapsing exact duplicate "
        f"cells; comparisons with fewer than {MIN_UNIQUE_PAIRS} unique pairs are untestable. "
        "Benjamini–Hochberg over testable comparisons. "
        "Significant = q ≤ 0.05 **and** mean delta < 0.",
        "",
        "| dataset | challenger | raw cells | unique pairs | mean ΔMSE | testable | p | BH q | sig better |",
        "|---|---|---:|---:|---:|:---:|---:|---:|:---:|",
    ]
    for t in tests:
        p = "—" if t.wilcoxon_p is None else f"{t.wilcoxon_p:.3g}"
        q = "—" if t.bh_q is None else f"{t.bh_q:.3g}"
        lines.append(
            f"| {t.dataset} | {t.challenger} | {t.n_raw_pairs} | {t.n_unique_pairs} | "
            f"{t.mean_delta_mse:.3e} | {'Y' if t.testable else 'N'} | {p} | {q} | "
            f"{'Y' if t.significant_at_0_05 else 'N'} |"
        )
    lines.extend(
        [
            "",
            f"Testable: **{n_testable}** / {len(tests)}. "
            f"Significant better after BH: **{payload['n_significant_better']}**.",
            "",
            "Lockbox H1 is excluded from this family (already opened once; see LOCKBOX_SUMMARY).",
            "",
        ]
    )
    (out_dir / "MULTIPLICITY_SUMMARY.md").write_text("\n".join(lines))
    return payload
