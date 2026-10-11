"""Read/write protocol/EXPERIMENT_MANIFEST.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MANIFEST_HEADER: tuple[str, ...] = (
    "run_id",
    "model_id",
    "variant",
    "dataset",
    "fold",
    "seed",
    "horizon",
    "profile",
    "config_hash",
    "dependencies",
    "primary_metric",
    "budget",
    "status",
    "result_path",
    "failure_reason",
)


@dataclass
class ManifestRow:
    run_id: str
    model_id: str
    variant: str
    dataset: str
    fold: int
    seed: int
    horizon: int
    profile: str
    config_hash: str
    dependencies: str
    primary_metric: str
    budget: str
    status: str
    result_path: str
    failure_reason: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "variant": self.variant,
            "dataset": self.dataset,
            "fold": self.fold,
            "seed": self.seed,
            "horizon": self.horizon,
            "profile": self.profile,
            "config_hash": self.config_hash,
            "dependencies": self.dependencies,
            "primary_metric": self.primary_metric,
            "budget": self.budget,
            "status": self.status,
            "result_path": self.result_path,
            "failure_reason": self.failure_reason,
        }


def manifest_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "protocol" / "EXPERIMENT_MANIFEST.csv"


def load_manifest(path: Path | str) -> list[ManifestRow]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[ManifestRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                ManifestRow(
                    run_id=raw["run_id"],
                    model_id=raw["model_id"],
                    variant=raw["variant"],
                    dataset=raw["dataset"],
                    fold=int(raw["fold"]),
                    seed=int(raw["seed"]),
                    horizon=int(raw["horizon"]),
                    profile=raw["profile"],
                    config_hash=raw.get("config_hash", ""),
                    dependencies=raw.get("dependencies", ""),
                    primary_metric=raw.get("primary_metric", ""),
                    budget=raw.get("budget", ""),
                    status=raw.get("status", "PENDING"),
                    result_path=raw.get("result_path", ""),
                    failure_reason=raw.get("failure_reason", ""),
                )
            )
    return rows


def write_manifest(path: Path | str, rows: Sequence[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MANIFEST_HEADER))
        writer.writeheader()
        for row in rows:
            d = row.as_dict()
            writer.writerow({k: d[k] for k in MANIFEST_HEADER})


def update_row_status(
    rows: list[ManifestRow],
    run_id: str,
    *,
    status: str,
    result_path: str = "",
    failure_reason: str = "",
) -> None:
    for i, row in enumerate(rows):
        if row.run_id == run_id:
            rows[i] = ManifestRow(
                **{
                    **row.as_dict(),
                    "status": status,
                    "result_path": result_path,
                    "failure_reason": failure_reason,
                }
            )
            return


def filter_profile(rows: Iterable[ManifestRow], profile: str) -> list[ManifestRow]:
    key = profile.lower()
    return [r for r in rows if r.profile.lower() == key]
