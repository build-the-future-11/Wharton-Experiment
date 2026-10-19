"""Campaign completeness, ownership, and terminal analysis."""

from __future__ import annotations

from pathlib import Path

from frem_s12.analysis import PrimaryDecision, decide
from frem_s12.artifacts import read_fold
from frem_s12.campaign import cells
from frem_s12.errors import ContractError
from frem_s12.evidence import read_json, validate_cell_provenance
from frem_s12.experiment import FoldResult


def discover_results(root: Path) -> tuple[FoldResult, ...]:
    """Load only complete hash-valid fold bundles under one campaign root."""
    if not root.is_dir() or root.is_symlink():
        raise ContractError("fixed campaign root is not a regular directory")
    expected_identities = {(value.family, value.fold, value.seed) for value in cells()}
    expected_files = {
        (Path(f"family={family}") / f"fold={fold}" / f"seed={seed}" / name).as_posix()
        for family, fold, seed in expected_identities
        for name in ("arrays.npz", "summary.json", "manifest.json")
    }
    observed_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ContractError("fixed campaign contains a symlink or special file")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
    if observed_files != expected_files:
        raise ContractError("fixed campaign file inventory is incomplete or foreign")
    manifests = sorted(root.glob("family=*/fold=*/seed=*/manifest.json"))
    results = tuple(read_fold(path.parent) for path in manifests)
    environment_hashes: set[str] = set()
    selection_hashes: set[str] = set()
    family_gates: dict[str, set[tuple[str, str, str]]] = {}
    consumption_hashes: set[str] = set()
    for path, result in zip(manifests, results, strict=True):
        expected_path = root / f"family={result.family}" / f"fold={result.fold}" / f"seed={result.seed}"
        if path.parent != expected_path:
            raise ContractError("campaign contains a misplaced cell bundle")
        summary = read_json(path.parent / "summary.json")
        provenance = validate_cell_provenance(
            summary.get("provenance"),
            phase="fixed",
            identity={"family": result.family, "fold": result.fold, "seed": result.seed},
        )
        environment_hashes.add(provenance["environment_receipt_sha256"])
        selection_hashes.add(provenance["selection_receipt_sha256"])
        family_gates.setdefault(result.family, set()).add(
            (
                provenance["data_receipt_sha256"],
                provenance["acquisition_receipt_sha256"],
                provenance["archive_sha256"],
            )
        )
        consumption_hashes.add(provenance["consumption_sha256"])
    identities = {(value.family, value.fold, value.seed) for value in results}
    expected = expected_identities
    if identities != expected or len(results) != len(expected):
        missing = sorted(expected - identities)
        extra = sorted(identities - expected)
        raise ContractError(
            f"campaign is incomplete or has foreign cells: missing={missing[:3]} extra={extra[:3]}"
        )
    if len(environment_hashes) != 1 or len(selection_hashes) != 1:
        raise ContractError("fixed cells do not share one environment and one selection receipt")
    if set(family_gates) != {value.family for value in cells()} or any(
        len(values) != 1 for values in family_gates.values()
    ):
        raise ContractError("fixed cells do not share one data gate per family")
    if len(consumption_hashes) != len(expected):
        raise ContractError("fixed cells do not have unique consumption claims")
    return results


def analyze_campaign(root: Path, *, draws: int = 10_000, seed: int = 1912) -> PrimaryDecision:
    """Reconstruct the frozen decision only from a complete campaign."""
    return decide(discover_results(root), draws=draws, seed=seed)
