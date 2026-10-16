"""Render bounded saved-statistic summaries without importing or running a model."""

import math

from frem_s10.errors import ContractError

from .ablations import AblationResult
from .endpoints import HypothesisResult, StudyResult


def _number(value: float | None) -> str:
    """Keep undefined values distinct from exact zero; JSON retains full precision."""
    if value is not None and not math.isfinite(value):
        raise ContractError("nonfinite report number")
    return "UNDEFINED" if value is None else format(value, ".12g")


def _cell(value: str) -> str:
    """Keep source descriptions from changing Markdown table structure."""
    return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _hypothesis(name: str, result: HypothesisResult | None) -> list[str]:
    if result is None:
        return [f"## {name}", "", "Not evaluated; retain the study-level failure reason.", ""]
    rows = [
        f"## {name}: {result.status}",
        "",
        "All required controls and block lengths are retained. Values are fractions, not percentages.",
        "",
        "| Control | Block length | Fold A | Fold B | Pooled | Lower 95% | Upper 95% | Reason |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result.comparisons:
        a, b = item.fold_points if item.fold_points is not None else (None, None)
        values = (a, b, item.pooled_point, item.lower, item.upper)
        rows.append(
            f"| {_cell(item.control)} | {item.block} | "
            + " | ".join(_number(value) for value in values)
            + f" | {_cell(item.reason or 'none')} |"
        )
    rows.extend(["", "Point gates:", ""])
    rows.extend(f"- {_cell(name)}: {'PASS' if passed else 'FAIL'}" for name, passed in result.point_gates)
    rows.append("")
    return rows


def _complete_hypothesis(
    result: HypothesisResult, controls: tuple[str, str], gate_names: tuple[str, ...]
) -> None:
    expected = {(control, block) for control in controls for block in (2, 4)}
    actual = [(row.control, row.block) for row in result.comparisons]
    if len(actual) != 4 or set(actual) != expected:
        raise ContractError("report requires each declared control/block comparison exactly once")
    if tuple(name for name, _ in result.point_gates) != gate_names or any(
        type(passed) is not bool for _, passed in result.point_gates
    ):
        raise ContractError("report requires every declared point gate")
    if result.status not in ("PASS", "FAIL", "INCONCLUSIVE"):
        raise ContractError("unrecognized conditional hypothesis disposition")


def _validate_completeness(study: StudyResult, ablations: AblationResult) -> None:
    expected = {(f"A{number}", block) for number in range(1, 9) for block in (2, 4)}
    actual = [(row.ablation_id, row.block) for row in ablations.comparisons]
    if len(actual) != 16 or set(actual) != expected:
        raise ContractError("report requires all sixteen distinct ablation comparisons")
    if study.disposition == "INVALID":
        if study.h1 is not None or study.h2 is not None or study.h3 or study.bootstrap or not study.reason:
            raise ContractError("invalid study must retain its reason without unsupported primary endpoints")
        return
    if study.h1 is None or study.h2 is None or tuple(row.fold_id for row in study.h3) != ("A", "B", "POOLED"):
        raise ContractError("complete H1/H2 and ordered A/B/POOLED descriptive utility required")
    _complete_hypothesis(
        study.h1,
        ("global_stack", "equal"),
        ("global_stack/both_folds_positive", "equal/both_folds_positive"),
    )
    gates = tuple(
        f"{fold}/{name}"
        for fold in ("A", "B")
        for name in (
            "equal100/at_least_6pct",
            "buy_hold100/at_least_6pct",
            "beats_cash",
            "drawdown_no_worse_than_equal80",
        )
    )
    _complete_hypothesis(study.h2, ("equal100", "buy_hold100"), gates)
    pair = (study.h1.status, study.h2.status)
    if "INCONCLUSIVE" in pair:
        expected_status = "INCONCLUSIVE"
    elif pair == ("PASS", "PASS"):
        expected_status = "SUPPORTED"
    elif pair == ("FAIL", "FAIL"):
        expected_status = "NEGATIVE"
    else:
        expected_status = "MIXED"
    if study.disposition != expected_status:
        raise ContractError("study disposition contradicts retained hypothesis dispositions")


def render_statistics(
    study: StudyResult,
    ablations: AblationResult,
    *,
    evidence_sha256: str,
    scope: str,
) -> str:
    """Render every supplied comparison with explicit computational evidence scope.

    The caller verifies the referenced inventory and obtains these statistics by
    read-only regeneration. A content identity is a link, never source authority.
    Full precision inputs and bootstrap arrays remain in their numeric artifacts.
    """
    if (
        len(evidence_sha256) != 64
        or any(char not in "0123456789abcdef" for char in evidence_sha256)
        or scope not in ("GENERATED_ENGINEERING", "SOURCE_AUTHORITY_UNVERIFIED")
        or study.confirmation_claim
        or study.profitability_guarantee
        or study.original_nll_hypothesis_satisfied
        or ablations.confirmation_claim
        or ablations.changes_primary
    ):
        raise ContractError("report requires an exact evidence link and bounded nonconfirmatory scope")
    _validate_completeness(study, ablations)
    rows = [
        "# S10 computational evidence summary",
        "",
        f"Evidence scope: **{scope}**. Inventory SHA-256: `{evidence_sha256}`.",
        "",
        f"Conditional numerical disposition: **{study.disposition}**.",
        "This is a calculation over the supplied evidence. It does not establish source admission, "
        "empirical support, confirmation, the original NLL claim or future profitability.",
        f"Study-level reason: {_cell(study.reason or 'none')}.",
        "",
    ]
    rows.extend(_hypothesis("H1 total-return energy score", study.h1))
    rows.extend(_hypothesis("H2 net economic endpoints", study.h2))
    rows.extend(
        [
            "## H3 descriptive CRRA utility",
            "",
            "| Fold | Blocks | Robust mean block | Nominal mean block | Robust terminal | Nominal terminal |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in study.h3:
        values = (
            item.robust_mean_block_utility,
            item.nominal_mean_block_utility,
            item.robust_terminal_utility,
            item.nominal_terminal_utility,
        )
        rows.append(f"| {item.fold_id} | {item.observations} | " + " | ".join(map(_number, values)) + " |")
    rows.extend(
        [
            "",
            "## Secondary ablations",
            "",
            "Positive ES difference favors the fixed primary. One Holm family contains all 16 tests. "
            "A4/A5 are policy-only and have structural zero score differences; no ablation replaces primary.",
            "",
            "| Ablation | Block | Fold A delta | Fold B delta | Pooled delta | Raw p | Holm p | Structural zero |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for comparison in ablations.comparisons:
        ablation_values = (
            *comparison.fold_means,
            comparison.pooled_mean,
            comparison.raw_p,
            comparison.holm_p,
        )
        rows.append(
            f"| {comparison.ablation_id} | {comparison.block} | "
            + " | ".join(map(_number, ablation_values))
            + f" | {comparison.structural_zero} |"
        )
    rows.extend(
        ["", "Bootstrap: 5,000 paired draws per fixed fold/block family; saved arrays are required.", ""]
    )
    return "\n".join(rows)
