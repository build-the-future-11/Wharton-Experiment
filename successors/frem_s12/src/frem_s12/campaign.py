"""Immutable search/fixed-matrix registries and shard ownership."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from frem_s12.errors import ContractError
from frem_s12.experiment import Fold

FAMILIES = ("size_bm", "size_op", "size_inv")
SEEDS = (11, 29, 47, 71, 97)
FOLDS = (
    Fold("A", 199801, 199912, 200001, 200112),
    Fold("B", 200001, 200112, 200201, 200312),
    Fold("C", 200201, 200312, 200401, 200512),
    Fold("D", 200401, 200512, 200601, 200712),
    Fold("E", 200601, 200712, 200801, 200912),
    Fold("F", 200801, 200912, 201001, 201112),
    Fold("G", 201001, 201112, 201201, 201312),
    Fold("H", 201201, 201312, 201401, 201512),
)


@dataclass(frozen=True, slots=True)
class Cell:
    """One fixed family/fold/seed cell."""

    family: str
    fold: str
    seed: int


def cells() -> tuple[Cell, ...]:
    """Return the complete 120-cell P0 fixed registry."""
    return tuple(
        Cell(family, fold.identifier, seed) for family in FAMILIES for fold in FOLDS for seed in SEEDS
    )


def registry() -> dict[str, object]:
    """Return canonical fixed-matrix registry and identity."""
    payload = [asdict(cell) for cell in cells()]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return {
        "schema": "frem-s12-fixed-registry-v1",
        "cells": payload,
        "cell_count": len(payload),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def shard_cells(shard: int, total_shards: int) -> tuple[Cell, ...]:
    """Assign each fixed cell to exactly one deterministic shard."""
    if total_shards < 1 or not 0 <= shard < total_shards:
        raise ContractError("shard coordinates are invalid")
    return tuple(cell for index, cell in enumerate(cells()) if index % total_shards == shard)


def search_space() -> tuple[dict[str, int | float], ...]:
    """Return exactly 24 predeclared router/reservoir trials."""
    trials = []
    for reservoir_width in (16, 32, 64):
        for router_hidden in (8, 16):
            for learning_rate in (0.001, 0.003):
                for capacity in (60, 120):
                    trials.append(
                        {
                            "reservoir_width": reservoir_width,
                            "router_hidden": router_hidden,
                            "router_learning_rate": learning_rate,
                            "engram_capacity": capacity,
                        }
                    )
    if len(trials) != 24:
        raise AssertionError("search registry must contain 24 trials")
    return tuple(trials)
