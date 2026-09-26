"""Experiment size profiles — lean defaults for fast local runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ProfileName(str, Enum):
    SMOKE = "SMOKE"
    PILOT = "PILOT"
    OVERNIGHT = "OVERNIGHT"
    FULL = "FULL"


@dataclass(frozen=True)
class ExperimentProfile:
    name: ProfileName
    seeds: tuple[int, ...]
    max_folds: int
    n_scenarios: int
    n_steps: int
    primary_horizon: int
    context_len: int
    description: str


# NEW_DEFAULT 2026-09-26: tightened folds/steps so full matrix finishes on 16GB laptop.
_PROFILES: dict[ProfileName, ExperimentProfile] = {
    ProfileName.SMOKE: ExperimentProfile(
        name=ProfileName.SMOKE,
        seeds=(11,),
        max_folds=1,
        n_scenarios=4,
        n_steps=48,
        primary_horizon=20,
        context_len=64,
        description="Minimal sanity run",
    ),
    ProfileName.PILOT: ExperimentProfile(
        name=ProfileName.PILOT,
        seeds=(11, 23, 47),
        max_folds=2,
        n_scenarios=16,
        n_steps=96,
        primary_horizon=20,
        context_len=64,
        description="Fast exploratory run",
    ),
    ProfileName.OVERNIGHT: ExperimentProfile(
        name=ProfileName.OVERNIGHT,
        seeds=(11, 23, 47, 89),
        max_folds=3,
        n_scenarios=32,
        n_steps=128,
        primary_horizon=20,
        context_len=64,
        description="Medium batch (lean)",
    ),
    ProfileName.FULL: ExperimentProfile(
        name=ProfileName.FULL,
        seeds=(11, 23, 47, 89, 131),
        max_folds=4,
        n_scenarios=64,
        n_steps=160,
        primary_horizon=20,
        context_len=64,
        description="Full registered grid (lean folds/steps)",
    ),
}


def get_profile(name: str | ProfileName) -> ExperimentProfile:
    key = ProfileName(name) if isinstance(name, str) else name
    return _PROFILES[key]


def seed_list(profile: ExperimentProfile) -> Sequence[int]:
    return profile.seeds
