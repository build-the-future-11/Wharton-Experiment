"""Fail-closed command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import PanelManifest, validate_panel
from .experiments import run_engineering_matrix


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="reng")
    commands = root.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser(
        "engineering-matrix", help="run labeled synthetic compatibility datasets"
    )
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--seed", type=int, default=11)
    validate = commands.add_parser(
        "data-validate", help="validate an admitted canonical parquet/csv panel"
    )
    validate.add_argument("--panel", required=True)
    validate.add_argument("--manifest", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "engineering-matrix":
        print(json.dumps(run_engineering_matrix(args.output, seed=args.seed), indent=2))
        return 0
    if args.command == "data-validate":
        import pandas as pd

        path = Path(args.panel)
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        manifest = PanelManifest.from_json(args.manifest)
        print(json.dumps(validate_panel(frame, manifest), indent=2))
        return 0
    raise AssertionError("unreachable")
