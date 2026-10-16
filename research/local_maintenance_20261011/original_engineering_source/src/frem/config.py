"""Strict JSON configuration. Relative paths resolve from the project root."""

from __future__ import annotations

import copy
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import read_json

ALLOWED_TOP = {
    "schema_version",
    "name",
    "data",
    "split",
    "model",
    "training",
    "evaluation",
    "models",
    "seeds",
    "output_dir",
    "budget",
    "description",
}
SUPPORTED_MODELS = frozenset(
    {
        "frem",
        "no_memory",
        "no_spectral",
        "no_reservoir",
        "no_jepa",
        "no_gate",
        "no_controller",
        "no_ngram",
        "fixed_scale",
        "fifo_memory",
        "uniform_memory",
        "hybrid_xgboost",
        "zero",
        "ridge",
        "reservoir_ridge",
        "xgboost",
        "gru",
        "dlinear",
        "patch_transformer",
    }
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    return validate_config(read_json(path), root=root)


def _finite_tree(value: Any, location: str = "configuration") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} requires string object keys")
            _finite_tree(item, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")


def positive_integer(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_config(configuration: dict[str, Any], root: str | Path | None = None) -> dict[str, Any]:
    """Validate and normalize every fully resolved configuration, without data I/O.

    File loading, public execution calls, and each merged suite variant all use
    this same function. It copies its input and resolves relative paths from the
    explicit project root, including paths introduced by later variant merges.
    It never opens datasets, constructs a model, or imports the training stack.
    """
    if not isinstance(configuration, dict):
        raise ValueError("Configuration must be a JSON object")
    cfg = copy.deepcopy(configuration)
    _finite_tree(cfg)
    unknown = sorted(set(cfg) - ALLOWED_TOP)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    if type(cfg.get("schema_version")) is not int or cfg["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    for key in ("data", "split", "model", "training", "evaluation", "budget"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(f"Missing object: {key}")
    if not isinstance(cfg["data"].get("path"), str) or not cfg["data"]["path"].strip():
        raise ValueError("data.path is required")
    if not cfg["split"].get("protected_start"):
        raise ValueError("split.protected_start is required; choose before inspecting holdout")
    if cfg["split"]["protected_start"] == "SET_BEFORE_FIRST_DATA_INSPECTION":
        raise ValueError("Set the prospective protected-start timestamp before data inspection")
    boundary = cfg["split"]["protected_start"]
    if not isinstance(boundary, str):
        raise ValueError("protected_start must be an ISO timestamp with a timezone")
    try:
        parsed_boundary = datetime.fromisoformat(boundary)
    except ValueError as error:
        raise ValueError("protected_start must be an ISO timestamp with a timezone") from error
    if parsed_boundary.utcoffset() is None:
        raise ValueError("protected_start must include its timezone")
    context, horizon = cfg["data"].get("context", 64), cfg["data"].get("horizon", 5)
    positive_integer(context, "data.context", 4)
    positive_integer(horizon, "data.horizon")
    positive_integer(cfg["split"].get("embargo_bars", 5), "split.embargo_bars", 0)
    if (
        not isinstance(cfg.get("models"), list)
        or not cfg["models"]
        or not all(isinstance(m, str) for m in cfg["models"])
    ):
        raise ValueError("models must be a nonempty list of names")
    if len(set(cfg["models"])) != len(cfg["models"]):
        raise ValueError("models must not contain duplicate variants")
    unsupported = sorted(set(cfg["models"]) - SUPPORTED_MODELS)
    if unsupported:
        raise ValueError(f"Unknown model variants: {unsupported}")
    seeds = cfg.get("seeds", [])
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(s, int) and not isinstance(s, bool) and s >= 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    maximum_runs = positive_integer(cfg["budget"].get("max_runs"), "budget.max_runs")
    if maximum_runs < len(cfg["models"]) * len(seeds):
        raise ValueError("Requested matrix exceeds budget.max_runs")
    if finite_number(cfg["budget"].get("max_wall_seconds"), "budget.max_wall_seconds") <= 0:
        raise ValueError("budget.max_wall_seconds must be positive")
    training_limit = cfg["training"].get("max_wall_seconds")
    if training_limit is not None and finite_number(training_limit, "training.max_wall_seconds") <= 0:
        raise ValueError("training.max_wall_seconds must be positive or null")
    for field in ("epochs", "batch_size"):
        if field in cfg["training"]:
            positive_integer(cfg["training"][field], f"training.{field}")
    train_fraction = finite_number(cfg["split"].get("train_fraction", 0.6), "split.train_fraction")
    val_fraction = finite_number(cfg["split"].get("validation_fraction", 0.2), "split.validation_fraction")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Require positive train/validation fractions with sum<1")
    noise = finite_number(cfg["evaluation"].get("noise_std", 0), "evaluation.noise_std")
    missing = finite_number(cfg["evaluation"].get("missing_probability", 0), "evaluation.missing_probability")
    fraction = finite_number(cfg["evaluation"].get("training_fraction", 1), "evaluation.training_fraction")
    if noise < 0 or not 0 <= missing < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid evaluation noise, missingness, or training fraction")
    if "online_update" in cfg["evaluation"] and not isinstance(cfg["evaluation"]["online_update"], bool):
        raise ValueError("evaluation.online_update must be a boolean")
    if "held_out_assets" in cfg["evaluation"]:
        names = cfg["evaluation"]["held_out_assets"]
        if (
            not isinstance(names, list)
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("held_out_assets must be a list of unique nonempty names")
    base = Path(root or project_root()).resolve()

    def resolved(value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty path string")
        return str((base / Path(value).expanduser()).resolve())

    cfg["data"]["path"] = resolved(cfg["data"]["path"], "data.path")
    for key in ("macro_path",):
        if cfg["data"].get(key):
            cfg["data"][key] = resolved(cfg["data"][key], f"data.{key}")
    cfg["output_dir"] = resolved(cfg.get("output_dir", "outputs/runs"), "output_dir")
    return cfg
