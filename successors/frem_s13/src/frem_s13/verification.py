"""Independent result readback from raw laws and cash-flow identities."""

import json

import numpy as np

from .config import RunConfig
from .evaluation import holm, paired_bootstrap, rebalance, score_law, summary


def verify_semantics(root):
    cfg = RunConfig.from_dict(json.loads((root / "config.json").read_text()))
    reference = None
    cells = 0
    date_scores = {}
    for arm in cfg.arms:
        for seed in cfg.seeds:
            cell = root / f"{arm}-s{seed}"
            with np.load(cell / "predictions.npz", allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            calendar = [raw[k] for k in ("y", "times", "ends", "assets")]
            if reference is None:
                reference = calendar
            elif any(not np.array_equal(a, b) for a, b in zip(reference, calendar, strict=True)):
                raise ValueError("cross-cell outcomes or calendar differ")
            scores = score_law(raw, raw["y"], cfg.model.student_df)
            weights = raw["weights"][:, 0]
            mean = (weights * raw["mu"]).sum(1)
            variance = (
                weights
                * (
                    raw["scale"] ** 2 * cfg.model.student_df / (cfg.model.student_df - 2)
                    + raw["mu"] ** 2
                )
            ).sum(1) - mean**2
            if not np.allclose(mean, raw["mean"], atol=1e-6, rtol=1e-5):
                raise ValueError("mixture mean differs")
            if not np.allclose(np.sqrt(np.maximum(variance, 0)), raw["std"], atol=1e-6, rtol=1e-4):
                raise ValueError("mixture standard deviation differs")
            if not np.allclose(raw["nll"], scores["nll"], rtol=3e-5, atol=3e-5):
                raise ValueError("retained likelihood differs from independent score")
            reconstructed = summary(raw, raw["y"], raw["times"], cfg.model.student_df)
            saved = json.loads((cell / "metrics.json").read_text())
            for key, value in reconstructed.items():
                if value is None:
                    if saved[key] is not None:
                        raise ValueError("undefined statistic was invented")
                elif not np.isclose(value, saved[key], atol=1e-10, rtol=1e-10):
                    raise ValueError(f"summary differs: {key}")
            for multiplier in (1, 2, 4):
                verify_ledger(
                    json.loads((cell / f"portfolio_cost{multiplier}.json").read_text()), raw
                )
            date_scores.setdefault(arm, []).append(
                np.array([scores["nll"][raw["times"] == d].mean() for d in np.unique(raw["times"])])
            )
            cells += 1
    for policy in ("equal_weight", "cash"):
        verify_ledger(json.loads((root / f"{policy}.json").read_text()), raw)
    comparisons = {}
    if "full" in date_scores:
        candidate = np.mean(date_scores["full"], axis=0)
        for arm in cfg.arms:
            if arm != "full":
                difference = np.mean(date_scores[arm], axis=0) - candidate
                comparisons[arm] = (
                    {"status": "INSUFFICIENT_DATES_FOR_DECLARED_BLOCK"}
                    if len(difference) < 2 * cfg.block_dates
                    else paired_bootstrap(difference, cfg.block_dates, cfg.bootstrap_reps)
                )
        available = [v for v in comparisons.values() if "p_two_sided" in v]
        for value, adjusted in zip(
            available, holm([v["p_two_sided"] for v in available]), strict=True
        ):
            value["p_holm"] = float(adjusted)
    if comparisons != json.loads((root / "comparisons.json").read_text())["comparisons"]:
        raise ValueError("seed aggregation or paired inference differs from raw scores")
    return {"cells": cells, "law_scores_and_ledger_arithmetic_verified": True}


def verify_ledger(report, raw):
    wealth, end, previous = 1.0, -1, None
    fee = report["cost_bps"] / 10000
    for row in report["ledger"]:
        if row["time"] < end:
            raise ValueError("overlapping holding periods")
        ix = raw["times"] == row["time"]
        if not ix.any() or not np.array_equal(raw["y"][ix], row["returns"]):
            raise ValueError("ledger outcomes differ from retained outcomes")
        if not np.all(raw["ends"][ix] == row["end"]):
            raise ValueError("ledger end differs from retained maturity")
        before = np.asarray(row["previous"])
        if previous is None:
            previous = np.zeros_like(before)
        if not np.allclose(before, previous, atol=1e-12, rtol=0):
            raise ValueError("holdings continuity failed")
        factor, previous, cost = rebalance(before, row["target"], row["returns"], fee)
        wealth *= factor
        if not np.allclose(
            [factor, cost, wealth],
            [row["factor"], row["cost_fraction"], row["wealth"]],
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("cash-flow reconstruction failed")
        if not np.allclose(previous, row["post"], atol=1e-12, rtol=0):
            raise ValueError("post-return holdings differ")
        end = row["end"]
    if previous is None:
        raise ValueError("empty ledger")
    liquidation, _, _ = rebalance(previous, np.zeros_like(previous), np.zeros_like(previous), fee)
    if not np.allclose(
        [liquidation, wealth * liquidation],
        [report["liquidation_factor"], report["ending_wealth"]],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("terminal liquidation differs")
