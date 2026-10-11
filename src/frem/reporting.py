"""Evidence-derived reports with complete pairing and frozen analysis settings."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import atomic_text, file_digest, object_digest, read_json, write_json


def _complete_run(run: Path) -> dict[str, Any]:
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Incomplete, interrupted or failed matrices cannot establish complete evidence")
    if manifest.get("phase") not in {"development", "confirmatory"}:
        raise ValueError("Unknown evidence phase")
    cfg = manifest["configuration"]
    models, seeds = list(cfg["models"]), list(cfg["seeds"])
    if not models or not seeds or len(models) != len(set(models)) or len(seeds) != len(set(seeds)):
        raise ValueError("Matrix model/seed family must be nonempty and unique")
    if any(not isinstance(model, str) or not model.replace("_", "").isalnum() for model in models):
        raise ValueError("Unsafe or invalid model identifier")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Configured seeds must be nonnegative integers")
    expected = {(model, int(seed)) for model in models for seed in seeds}
    completed = read_json(run / "completed_cells.json")
    cells = [(row["model"], int(row["seed"])) for row in completed]
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise ValueError("Completed-cell registry differs from the full configured model/seed matrix")
    for model, seed in expected:
        cell = run / model / f"seed_{seed}"
        if not (cell / "predictions.csv").is_file() or not (cell / "metrics.json").is_file():
            raise ValueError(f"Missing prediction or metric artifact for {model}/seed_{seed}")
    if manifest.get("phase") == "confirmatory" and not manifest.get("identity", {}).get("freeze_sha256"):
        raise ValueError("Confirmatory evidence lacks a recorded freeze identity")
    return manifest


def _matching_evidence(
    left: pd.DataFrame, right: pd.DataFrame, keys: tuple[str, ...] = ("time", "asset_id", "seed")
) -> None:
    evidence = ["actual", "label_time"]
    for field in ("entry_time", "exit_time", "source_id", "cohort_id", "protocol_id", "panel_fingerprint"):
        if (field in left) != (field in right):
            raise ValueError(f"Methods or seeds differ in saved {field} evidence")
        if field in left:
            evidence.append(field)
    required = set(keys) | set(evidence)
    if required - set(left) or required - set(right):
        raise ValueError("Missing pairing keys or outcome evidence")
    joined = left[list(keys) + evidence].merge(
        right[list(keys) + evidence],
        on=list(keys),
        how="outer",
        indicator=True,
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Unmatched observation keys; failed or missing rows cannot be silently removed")
    for field in evidence:
        if not np.array_equal(joined[field + "_left"].to_numpy(), joined[field + "_right"].to_numpy()):
            raise ValueError(f"Methods or seeds use different {field} evidence")


def _predictions(run: Path, model: str, manifest: dict) -> pd.DataFrame:
    from .evaluation import score_predictions
    from .statistics import average_seeds

    expected = {int(seed) for seed in manifest["configuration"]["seeds"]}
    directories = [path for path in (run / model).glob("seed_*") if path.is_dir()]
    try:
        present = {int(path.name.removeprefix("seed_")) for path in directories}
    except ValueError:
        raise ValueError(f"Malformed seed directory for {model}") from None
    if present != expected or {path.name for path in directories} != {f"seed_{seed}" for seed in expected}:
        raise ValueError(f"Incomplete seed family for {model}: expected {expected}, found {present}")
    frames: list[pd.DataFrame] = []
    for seed in sorted(expected):
        frame = pd.read_csv(run / model / f"seed_{seed}" / "predictions.csv")
        if "seed" not in frame or "model" not in frame:
            raise ValueError("Prediction file lacks its model/seed identity")
        if set(frame["seed"].unique()) != {seed} or set(frame["model"].unique()) != {model}:
            raise ValueError("Prediction identity does not match the experiment cell")
        frame = score_predictions(frame, student_df=None)
        if frames:
            _matching_evidence(frames[0], frame, keys=("time", "asset_id"))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    # Check the full seed-observation grid, not only the seed directory names.
    average_seeds(result, ["nll"])
    return result


def _protocol_for_run(run: Path, manifest: dict) -> tuple[dict[str, Any], str]:
    from .config import project_root

    expected = manifest.get("identity", {}).get("code", {}).get("files", {}).get("research/protocol.json")
    archived = run / "protocol.json"
    source = archived if archived.is_file() else project_root() / "research" / "protocol.json"
    actual = file_digest(source)
    if expected is None:
        raise ValueError("Run identity did not preserve the research protocol hash")
    if actual != expected:
        raise ValueError("Analysis protocol differs from the protocol frozen with the run")
    if manifest.get("phase") == "confirmatory":
        files = manifest["identity"]["code"]["files"]
        for name in ("artifacts", "config", "evaluation", "metrics", "reporting", "statistics"):
            relative = f"src/frem/{name}.py"
            if files.get(relative) != file_digest(project_root() / relative):
                raise ValueError(f"Confirmatory analysis source differs from its frozen version: {relative}")
    return read_json(source), actual


def _settings(
    manifest: dict,
    protocol: dict,
    *,
    block_length: int | None,
    n_resamples: int | None,
    confidence: float | None,
    threshold: float | None,
    bootstrap_seed: int | None,
) -> dict[str, Any]:
    from .config import finite_number, positive_integer

    stats = protocol["statistics"]
    primary = protocol["primary_comparison"]
    confirmatory = manifest["phase"] == "confirmatory"
    if confirmatory and "bootstrap_seed" not in stats:
        raise ValueError("The frozen confirmatory protocol must explicitly declare bootstrap_seed")
    prescribed = {
        "block_length": positive_integer(stats["block_dates"], "block_dates"),
        "n_resamples": positive_integer(stats["bootstrap_repetitions"], "bootstrap_repetitions", 199),
        "confidence": finite_number(stats["confidence"], "confidence"),
        "threshold": finite_number(primary["minimum_practical_improvement_nats"], "threshold"),
        "bootstrap_seed": positive_integer(stats.get("bootstrap_seed", 1901), "bootstrap_seed", 0),
    }
    cfg_eval = manifest["configuration"].get("evaluation", {})
    cfg_fields = (
        ("block_length", "bootstrap_block_dates"),
        ("n_resamples", "bootstrap_repetitions"),
        ("confidence", "bootstrap_confidence"),
        ("bootstrap_seed", "bootstrap_seed"),
    )
    defaults = dict(prescribed)
    if confirmatory:
        for key, cfg_key in cfg_fields:
            if cfg_key in cfg_eval and cfg_eval[cfg_key] != prescribed[key]:
                raise ValueError(f"Frozen configuration and protocol disagree on {key}")
    else:
        for key, cfg_key in cfg_fields:
            if cfg_key in cfg_eval:
                defaults[key] = cfg_eval[cfg_key]
    chosen = {
        "block_length": defaults["block_length"] if block_length is None else block_length,
        "n_resamples": defaults["n_resamples"] if n_resamples is None else n_resamples,
        "confidence": defaults["confidence"] if confidence is None else confidence,
        "threshold": defaults["threshold"] if threshold is None else threshold,
        "bootstrap_seed": defaults["bootstrap_seed"] if bootstrap_seed is None else bootstrap_seed,
    }
    for key, minimum in (("block_length", 1), ("n_resamples", 199), ("bootstrap_seed", 0)):
        chosen[key] = positive_integer(chosen[key], key, minimum)
    for key in ("confidence", "threshold"):
        chosen[key] = finite_number(chosen[key], key)
    allowed_blocks = {prescribed["block_length"]} | {
        positive_integer(x, "sensitivity_block_dates") for x in stats.get("sensitivity_block_dates", [])
    }
    if confirmatory:
        if chosen["block_length"] not in allowed_blocks:
            raise ValueError("Confirmatory block length was not in the frozen primary/sensitivity family")
        for key in ("n_resamples", "confidence", "threshold", "bootstrap_seed"):
            if chosen[key] != prescribed[key]:
                raise ValueError(f"Confirmatory {key} cannot differ from its frozen value")
    if (
        chosen["block_length"] < 1
        or chosen["n_resamples"] < 199
        or chosen["bootstrap_seed"] < 0
        or not 0 < chosen["confidence"] < 1
        or not np.isfinite(chosen["threshold"])
        or chosen["threshold"] < 0
    ):
        raise ValueError("Invalid statistical settings")
    chosen["prescribed"] = prescribed
    chosen["primary_block"] = chosen["block_length"] == prescribed["block_length"]
    chosen["allowed_frozen_block_family"] = sorted(allowed_blocks)
    return chosen


def _paired_analysis(
    left: pd.DataFrame, right: pd.DataFrame, settings: dict, horizon: int
) -> tuple[dict, int]:
    from .statistics import average_seeds, minimum_overlap_block_length, paired_date_test

    _matching_evidence(left, right)
    minimum_block = max(horizon, minimum_overlap_block_length(left))
    left_mean, right_mean = average_seeds(left, ["nll"]), average_seeds(right, ["nll"])
    result = paired_date_test(
        left_mean,
        right_mean,
        "nll",
        block_length=settings["block_length"],
        n_resamples=settings["n_resamples"],
        seed=settings["bootstrap_seed"],
        minimum_block_length=minimum_block,
        confidence=settings["confidence"],
    )
    return result, minimum_block


def _input_digests(run: Path, models: list[str], seeds: list[int]) -> dict[str, str]:
    paths = [run / model / f"seed_{seed}" / "predictions.csv" for model in models for seed in seeds]
    return {str(path.relative_to(run)): file_digest(path) for path in paths}


def _preserve_analysis(run: Path, stem: str, result: dict) -> Path:
    """Different statistical settings or source files produce different artifacts."""
    digest = object_digest(result)
    destination = run / "analysis" / f"{stem}_{digest[:16]}.json"
    if destination.exists():
        if object_digest(read_json(destination)) != digest:
            raise ValueError("Existing analysis artifact is inconsistent with its identity")
    else:
        write_json(destination, result)
    return destination


def compare_models(
    run_path: str | Path,
    candidate: str = "frem",
    control: str = "no_memory",
    *,
    block_length: int | None = None,
    n_resamples: int | None = None,
    threshold: float | None = None,
    bootstrap_seed: int | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    run = Path(run_path)
    manifest = _complete_run(run)
    cfg = manifest["configuration"]
    if candidate == control or candidate not in cfg["models"] or control not in cfg["models"]:
        raise ValueError("Distinct compared methods must be included in the completed matrix")
    protocol, protocol_hash = _protocol_for_run(run, manifest)
    primary_spec = protocol["primary_comparison"]
    primary = candidate == primary_spec["candidate"] and control == primary_spec["control"]
    allowed_controls = {primary_spec["control"]} | set(protocol["mechanism_comparisons"])
    if manifest["phase"] == "confirmatory" and (
        candidate != primary_spec["candidate"] or control not in allowed_controls
    ):
        raise ValueError("Confirmatory comparison is outside the frozen primary/secondary family")
    settings = _settings(
        manifest,
        protocol,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        threshold=threshold,
        bootstrap_seed=bootstrap_seed,
    )
    left, right = _predictions(run, control, manifest), _predictions(run, candidate, manifest)
    inference, minimum_block = _paired_analysis(left, right, settings, int(cfg["data"].get("horizon", 5)))
    statistical_result = (
        "INTERVAL_ABOVE_PRACTICAL_THRESHOLD"
        if inference["ci_lower"] > settings["threshold"]
        else "NONPOSITIVE_UPPER_BOUND"
        if inference["ci_upper"] <= 0
        else "INCONCLUSIVE"
    )
    primary_eligible = manifest["phase"] == "confirmatory" and primary and settings["primary_block"]
    result = {
        "candidate": candidate,
        "control": control,
        "phase": manifest["phase"],
        "effect": "control_nll_minus_candidate_nll; positive favors candidate",
        "practical_threshold_nats": settings["threshold"],
        "settings": settings,
        "primary_comparison": primary,
        "prescribed_primary_block": settings["primary_block"],
        "analysis_role": "primary"
        if primary_eligible
        else "secondary_or_sensitivity"
        if manifest["phase"] == "confirmatory"
        else "development",
        "statistical_result": statistical_result,
        "primary_statistical_criterion_met": primary_eligible
        and inference["ci_lower"] > settings["threshold"],
        "minimum_overlap_block_dates": minimum_block,
        "identity": manifest["identity"],
        "protocol_sha256": protocol_hash,
        "inference": inference,
        "source_prediction_sha256": _input_digests(run, [candidate, control], list(cfg["seeds"])),
        "limitations": "Date-block inference needs adequate temporal stability and independent blocks; optimizer seeds do not create new market histories.",
        "disposition": "A primary statistical criterion still requires the full integrity audit. Secondary p-values need the complete Holm family; sensitivity/development results do not establish primary support.",
    }
    artifact = _preserve_analysis(run, f"{candidate}_vs_{control}", result)
    return {**result, "analysis_artifact": str(artifact)}


def compare_family(
    run_path: str | Path,
    *,
    block_length: int | None = None,
    n_resamples: int | None = None,
    confidence: float | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Apply Holm to the complete protocol-defined secondary mechanism family."""
    from .statistics import holm_adjust

    run = Path(run_path)
    manifest = _complete_run(run)
    cfg = manifest["configuration"]
    protocol, protocol_hash = _protocol_for_run(run, manifest)
    candidate = str(protocol["primary_comparison"]["candidate"])
    controls = list(protocol["mechanism_comparisons"])
    if not controls or len(controls) != len(set(controls)) or candidate in controls:
        raise ValueError("Invalid or empty predeclared mechanism family")
    if {candidate, *controls} - set(cfg["models"]):
        raise ValueError("The complete prescribed mechanism family is absent from this matrix")
    settings = _settings(
        manifest,
        protocol,
        block_length=block_length,
        n_resamples=n_resamples,
        confidence=confidence,
        threshold=None,
        bootstrap_seed=bootstrap_seed,
    )
    candidate_frame = _predictions(run, candidate, manifest)
    comparisons = []
    p_values = {}
    for control in controls:
        control_frame = _predictions(run, control, manifest)
        inference, bound = _paired_analysis(
            control_frame, candidate_frame, settings, int(cfg["data"].get("horizon", 5))
        )
        name = f"{candidate}_vs_{control}"
        p_values[name] = float(inference["p_two_sided"])
        comparisons.append(
            {
                "hypothesis": name,
                "candidate": candidate,
                "control": control,
                "inference": inference,
                "minimum_overlap_block_dates": bound,
            }
        )
    adjusted = holm_adjust(p_values)
    alpha = 1 - settings["confidence"]
    confirmatory_family = manifest["phase"] == "confirmatory" and settings["primary_block"]
    for result in comparisons:
        result["holm_adjusted_p"] = adjusted[result["hypothesis"]]
        result["reject_equal_mean_loss_after_holm"] = result["holm_adjusted_p"] <= alpha
        result["candidate_gain"] = result["inference"]["estimate"] > 0
        result["confirmatory_secondary_criterion_met"] = (
            confirmatory_family and result["reject_equal_mean_loss_after_holm"] and result["candidate_gain"]
        )
    output = {
        "phase": manifest["phase"],
        "candidate": candidate,
        "analysis_role": "secondary_family"
        if confirmatory_family
        else "sensitivity"
        if manifest["phase"] == "confirmatory"
        else "development",
        "family": controls,
        "family_complete": True,
        "multiplicity": "Holm step-down",
        "alpha": alpha,
        "settings": settings,
        "comparisons": comparisons,
        "intervals_are_marginal_not_simultaneous": True,
        "identity": manifest["identity"],
        "protocol_sha256": protocol_hash,
        "source_prediction_sha256": _input_digests(run, [candidate] + controls, list(cfg["seeds"])),
        "interpretation": "Adjusted tests concern zero mean loss difference within this declared seven-control family, not simultaneous selection across block lengths. Only the prescribed primary block supports confirmatory secondary decisions; a rejection alone does not establish economic value or primary practical support.",
    }
    artifact = _preserve_analysis(run, "secondary_mechanism_family", output)
    return {**output, "analysis_artifact": str(artifact)}


def render_report(run_path: str | Path, destination: str | Path | None = None) -> Path:
    """Rank only a completed matrix scored on identical outcomes and time evidence."""
    run = Path(run_path)
    manifest = _complete_run(run)
    _, protocol_hash = _protocol_for_run(run, manifest)
    rows = []
    reference: pd.DataFrame | None = None
    for model in manifest["configuration"]["models"]:
        frame = _predictions(run, model, manifest)
        if reference is None:
            reference = frame
        else:
            _matching_evidence(reference, frame)
        frame["utc_date"] = pd.to_datetime(frame["time"], utc=True).dt.normalize()
        per_seed_asset_date = frame.groupby(["seed", "utc_date", "asset_id"], observed=True)["nll"].mean()
        per_seed_date = per_seed_asset_date.groupby(level=["seed", "utc_date"]).mean()
        seed_mean = per_seed_date.groupby(level="seed").mean()
        rows.append(
            {
                "model": model,
                "mean_nll": float(seed_mean.mean()),
                "seed_sd": float(seed_mean.std(ddof=1)) if len(seed_mean) > 1 else 0.0,
                "seeds": len(seed_mean),
                "decision_dates": frame["utc_date"].nunique(),
            }
        )
    # All evidence is checked before a ranked figure or report directory is written.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest = Path(destination) if destination else run / "report"
    if (dest / "index.html").exists():
        raise FileExistsError("A report already exists; choose a new path to retain earlier evidence")
    dest.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("mean_nll")
    table.to_csv(dest / "predictive_scores.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(table))))
    ax.barh(table["model"], table["mean_nll"], color="#174d5a")
    ax.invert_yaxis()
    ax.set_xlabel("Date-balanced Student-t NLL (lower is better)")
    ax.set_title("Observed predictive scores — no profitability inference")
    fig.tight_layout()
    fig.savefig(dest / "predictive_scores.png", dpi=180)
    plt.close(fig)
    audit = {
        "run": str(run.resolve()),
        "run_id": manifest["run_id"],
        "identity": manifest["identity"],
        "phase": manifest["phase"],
        "protocol_sha256": protocol_hash,
        "weighting": "mean within seed/date/asset, then equal assets per date, then equal dates and seeds",
        "source_prediction_sha256": _input_digests(
            run, list(manifest["configuration"]["models"]), list(manifest["configuration"]["seeds"])
        ),
    }
    write_json(dest / "report_provenance.json", audit)
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>FREM evidence report</title>
<style>body{{font:16px system-ui;max-width:1040px;margin:48px auto;line-height:1.6;color:#16323a}}table{{border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>FREM evidence report</h1><p>Phase: {html.escape(manifest["phase"])}. Run: <code>{html.escape(manifest["run_id"])}</code>.</p>
<p>Every method and configured seed uses matching outcomes, label availability and execution timestamps. Repeated observations are averaged within an asset and UTC date before assets and dates are weighted equally.</p>
<p>Seed standard deviation describes optimizer variability; it is not a confidence interval across independent financial histories. Use the paired date-block analysis and the declared complete multiplicity family for inference.</p>
<img src="predictive_scores.png" alt="Predictive likelihood by method">{table.to_html(index=False, escape=True)}
<h2>Evidence identity</h2><p>Source SHA-256: <code>{html.escape(manifest["identity"]["code"]["sha256"])}</code>.</p>
<p>Raw forecasts, exact configurations, retained failures, training records and prediction-file hashes remain in the experiment directory. No strategy-return plot is made without an explicit realizable portfolio ledger.</p></html>"""
    atomic_text(dest / "index.html", content)
    return dest / "index.html"


def plot_equity(ledger_path: str | Path, destination: str | Path) -> Path:
    """Render wealth only from validated nonoverlapping net-return accounting."""
    from .evaluation import evaluate_portfolio

    ledger = pd.read_csv(ledger_path)
    evaluate_portfolio(ledger)
    ledger = ledger.sort_values("entry_time").reset_index(drop=True)
    growth = (1.0 + ledger["net_return"].to_numpy()).cumprod()
    dates = pd.to_datetime(ledger["exit_time"], utc=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dates, growth, color="#174d5a", linewidth=1.6)
    ax.set(xlabel="Exit date", ylabel="Wealth per unit initial capital", title="Realized net strategy path")
    fig.tight_layout()
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError("Wealth plot already exists; choose a new output path")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    return dest
