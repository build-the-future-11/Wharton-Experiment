"""Independent stdlib review of retained S6-S8 identities and S8 claims.

No fitting, forecast generation, policy replay, or protected data access.
Run from the repository root; output must not already exist.
"""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    freeze_paths = {
        "s6": "research/frem_s6_real_20261014/freeze2/source_hashes.json",
        "s7": "research/frem_s7_20261014/freeze2.json",
        "s8": "research/frem_s8_20261014/freeze2.json",
    }
    source_counts = {}
    for name, path in freeze_paths.items():
        document = read(path)
        files = document if name == "s6" else document["sources"]
        for filename, expected in files.items():
            assert sha(filename) == expected, filename
        source_counts[name] = len(files)

    base = Path("research/frem_s8_20261014")
    output = Path("outputs/frem_s8_2020_2022_20261014_attempt1")
    data = Path("data/s8_etf_2020_2022_20261014_attempt2")
    original, corrected = read(base / "freeze.json"), read(base / "freeze2.json")
    consumed = read(base / "evaluation_consumed.json")
    acquired = [read(data / f"{asset}.receipt.json")["retrieved_at"] for asset in ("SPY", "TLT", "GLD")]
    parse = datetime.fromisoformat
    assert parse(original["created_at"]) < min(map(parse, acquired))
    assert max(map(parse, acquired)) < parse(corrected["created_at"]) < parse(consumed["started_at"])
    assert corrected["original_preacquisition_freeze_sha256"] == sha(base / "freeze.json")
    assert consumed["freeze_sha256"] == sha(base / "freeze2.json")
    for filename, expected in corrected["acquired_raw_identities"].items():
        assert sha(filename) == expected, filename

    records = read(output / "ensembles.json")
    scores, comparisons = read(output / "scores.json"), read(output / "comparisons.json")
    variants = sorted({r["variant"] for r in records})
    assert len(variants) == 15
    by_variant = {}
    score_residual = 0.0
    for variant in variants:
        rows = sorted((r for r in records if r["variant"] == variant), key=lambda r: r["date"])
        assert len(rows) == 153 and len({r["date"] for r in rows}) == 153
        assert all(2020 <= datetime.fromtimestamp(r["date"], timezone.utc).year <= 2022 for r in rows)
        by_variant[variant] = rows
        value = statistics.mean(statistics.mean(r["nll"]) for r in rows)
        score_residual = max(score_residual, abs(value - scores[variant]["ensemble_nll"]))
    assert score_residual < 1e-12
    predictive = {}
    for control in ("equal", "memory", "fixed_half", "s8_global"):
        differences = [
            statistics.mean(c["nll"]) - statistics.mean(s["nll"])
            for c, s in zip(by_variant[control], by_variant["s8"], strict=True)
            if c["date"] == s["date"]
        ]
        assert len(differences) == 153
        point = statistics.mean(differences)
        for block in ("6", "12"):
            assert abs(point - comparisons[control][block]["mean"]) < 1e-12
        predictive[control] = {
            "gain_nats": point,
            "retained_intervals": {b: comparisons[control][b]["interval95"] for b in ("6", "12")},
            "both_lower_bounds_exceed_0_01": all(comparisons[control][b]["interval95"][0] > 0.01 for b in ("6", "12")),
        }

    economics, benchmarks = read(output / "economics.json"), read(output / "benchmarks.json")
    candidate = economics["s8"]["10"]["s8_h4"]["terminal_wealth"]
    controls = {
        "equal_forecast_h4": economics["equal"]["10"]["s8_h4"]["terminal_wealth"],
        "s6_memory_h4": economics["memory"]["10"]["s8_h4"]["terminal_wealth"],
        "cash": benchmarks["cash"]["10"]["terminal_wealth"],
        "equal_weight_assets": benchmarks["equal_weight"]["10"]["terminal_wealth"],
        "buy_hold_assets": benchmarks["buy_hold"]["10"]["terminal_wealth"],
    }
    economic = {
        key: {
            "benchmark_wealth": value,
            "candidate_relative_uplift": candidate / value - 1,
            "minimum_target_wealth": 1.06 * value,
            "stretch_target_wealth": 1.36 * value,
            "minimum_target_met": candidate >= 1.06 * value,
            "stretch_target_met": candidate >= 1.36 * value,
        }
        for key, value in controls.items()
    }
    eq = economic["equal_weight_assets"]
    eq["minimum_dollar_gap"] = eq["minimum_target_wealth"] - candidate
    eq["stretch_dollar_gap"] = eq["stretch_target_wealth"] - candidate
    eq["minimum_improvement_over_candidate"] = eq["minimum_target_wealth"] / candidate - 1
    eq["stretch_improvement_over_candidate"] = eq["stretch_target_wealth"] / candidate - 1
    audit = read("research/finish_all_20261014/review/s8_saved_audit.json")
    assert audit["status"] == "PASS"
    assert not any(v["minimum_target_met"] for v in economic.values())
    assert not predictive["equal"]["both_lower_bounds_exceed_0_01"]
    assert not predictive["memory"]["both_lower_bounds_exceed_0_01"]
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_BOUNDED_CLAIM_AND_IDENTITY_CHECK",
        "source_identities_verified": source_counts,
        "freeze_chronology": {
            "original_before_acquisition": True,
            "corrected_after_acquisition_before_execution": True,
            "original_created_at": original["created_at"],
            "corrected_created_at": corrected["created_at"],
            "execution_started_at": consumed["started_at"],
        },
        "variant_count": len(variants),
        "evaluation_dates": 153,
        "score_max_abs_difference": score_residual,
        "predictive_comparisons": predictive,
        "candidate_wealth_10bps": candidate,
        "candidate_wealth_0bps": economics["s8"]["0"]["s8_h4"]["terminal_wealth"],
        "wealth_target_comparisons": economic,
        "saved_output_audit": audit,
        "scope": "Same-host independent agent review, arithmetic and immutable identity checks. Bootstrap endpoints retained and checked against frozen thresholds, not independently resampled here. No market fitting, policy replay, protected access, clean-environment reproduction, human review or distribution authorization.",
        "review_script_sha256": sha(__file__),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
