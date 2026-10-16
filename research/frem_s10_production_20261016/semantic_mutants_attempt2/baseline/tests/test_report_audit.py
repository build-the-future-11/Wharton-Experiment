"""Report conservation: no required comparison, undefined result or scope is omitted."""

from dataclasses import replace

import pytest
from test_ablations_audit import fold as ablation_fold
from test_endpoints_audit import fold

from frem_s10.analysis.ablations import assess_ablations
from frem_s10.analysis.endpoints import assess_study
from frem_s10.analysis.report import render_statistics
from frem_s10.errors import ContractError


def test_report_retains_all_comparisons_and_rejects_unsupported_claim_scope() -> None:
    study = assess_study((fold("A"), fold("B")), integrity_verified=True, complete_matrix=True)
    ablations = assess_ablations((ablation_fold("A", 24), ablation_fold("B", 24)))
    text = render_statistics(study, ablations, evidence_sha256="a" * 64, scope="GENERATED_ENGINEERING")
    assert "GENERATED_ENGINEERING" in text and "does not establish source admission" in text
    for row in ablations.comparisons:
        assert text.count(f"| {row.ablation_id} | {row.block} |") == 1
    for hypothesis in (study.h1, study.h2):
        assert hypothesis is not None
        for primary_row in hypothesis.comparisons:
            assert text.count(f"| {primary_row.control} | {primary_row.block} |") == 1
    for invalid_study in (
        replace(study, confirmation_claim=True),
        replace(study, profitability_guarantee=True),
        replace(study, original_nll_hypothesis_satisfied=True),
    ):
        with pytest.raises(ContractError):
            render_statistics(
                invalid_study,
                ablations,
                evidence_sha256="a" * 64,
                scope="GENERATED_ENGINEERING",
            )
    assert study.h1 is not None
    undefined = replace(study.h1.comparisons[0], lower=None, reason="undefined | retained")
    mixed = replace(study, h1=replace(study.h1, comparisons=(undefined, *study.h1.comparisons[1:])))
    text = render_statistics(mixed, ablations, evidence_sha256="a" * 64, scope="SOURCE_AUTHORITY_UNVERIFIED")
    assert "UNDEFINED" in text and "undefined \\| retained" in text


@pytest.mark.parametrize("part", ("h1", "h2", "h3", "ablations"))
@pytest.mark.parametrize("mutation", ("drop", "duplicate"))
def test_report_rejects_missing_or_duplicate_required_rows(part: str, mutation: str) -> None:
    study = assess_study((fold("A"), fold("B")), integrity_verified=True, complete_matrix=True)
    ablations = assess_ablations((ablation_fold("A", 24), ablation_fold("B", 24)))
    if part in ("h1", "h2"):
        hypothesis = study.h1 if part == "h1" else study.h2
        assert hypothesis is not None
        rows = hypothesis.comparisons
        changed = rows[:-1] if mutation == "drop" else (rows[0], *rows[:-1])
        hypothesis = replace(hypothesis, comparisons=changed)
        study = replace(study, h1=hypothesis) if part == "h1" else replace(study, h2=hypothesis)
    elif part == "h3":
        utilities = study.h3[:-1] if mutation == "drop" else (study.h3[0], study.h3[0])
        study = replace(study, h3=utilities)
    else:
        comparisons = ablations.comparisons
        comparisons = comparisons[:-1] if mutation == "drop" else (comparisons[0], *comparisons[:-1])
        ablations = replace(ablations, comparisons=comparisons)
    with pytest.raises(ContractError):
        render_statistics(study, ablations, evidence_sha256="a" * 64, scope="GENERATED_ENGINEERING")


def test_invalid_report_retains_reason_without_primary_support() -> None:
    study = assess_study((fold("A"), fold("B")), integrity_verified=False, complete_matrix=False)
    ablations = assess_ablations((ablation_fold("A", 24), ablation_fold("B", 24)))
    text = render_statistics(study, ablations, evidence_sha256="a" * 64, scope="GENERATED_ENGINEERING")
    assert "**INVALID**" in text and "integrity or complete matrix not established" in text
    assert text.count("Not evaluated;") == 2
