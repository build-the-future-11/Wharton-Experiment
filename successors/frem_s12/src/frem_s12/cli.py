"""Auditable FREM-S12 command-line research surface."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import NoReturn

from frem_s12.acquisition import acquire
from frem_s12.admission import (
    admit_archive,
    qualify_environment,
    qualify_kaggle_environment,
    require_execution_receipts,
)
from frem_s12.artifacts import decision_json, write_fold
from frem_s12.campaign import FAMILIES, FOLDS, SEEDS, registry, search_space, shard_cells
from frem_s12.campaign_io import analyze_campaign
from frem_s12.checkpoints import read_checkpoint, write_checkpoint
from frem_s12.consumption import claim_consumption
from frem_s12.data import parse_archive
from frem_s12.errors import S12Error
from frem_s12.evidence import atomic_json, package_identity, read_json, sha256_file
from frem_s12.experiment import Fold, run_fold
from frem_s12.generated import generated_panel
from frem_s12.run_bundle import verify_run_bundle
from frem_s12.saved_versions import audit_saved_versions
from frem_s12.search import (
    SEARCH_SEEDS,
    config_from_selection,
    run_search_cell,
    select_search,
    write_search_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="s12-research")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="print the immutable search and fixed registries")

    qualify = commands.add_parser("qualify", help="write a bounded environment receipt")
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--benchmark-size", type=int, default=384)

    qualify_kaggle = commands.add_parser(
        "qualify-kaggle", help="write a Kaggle-native scientific admission receipt"
    )
    qualify_kaggle.add_argument("--output", type=Path, required=True)

    acquisition = commands.add_parser("acquire", help="download one exact allowlisted public archive")
    acquisition.add_argument("--family", choices=FAMILIES, required=True)
    acquisition.add_argument("--release", choices=(201608, 202408), type=int, required=True)
    acquisition.add_argument("--output", type=Path, required=True)
    acquisition.add_argument("--execute", action="store_true", required=True)

    admission = commands.add_parser("admit-data", help="parse and bind an acquired archive")
    admission.add_argument("--archive", type=Path, required=True)
    admission.add_argument("--acquisition-receipt", type=Path, required=True)
    admission.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run-cell", help="execute one gated fixed-matrix cell")
    run.add_argument("--family", choices=FAMILIES, required=True)
    run.add_argument("--fold", choices=tuple(value.identifier for value in FOLDS), required=True)
    run.add_argument("--seed", choices=SEEDS, type=int, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--acquisition-receipt", type=Path, required=True)
    run.add_argument("--data-receipt", type=Path, required=True)
    run.add_argument("--environment-receipt", type=Path, required=True)
    run.add_argument("--selection-receipt", type=Path, required=True)
    run.add_argument("--consumption-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-dir", type=Path)
    run.add_argument("--resume", type=Path)
    run.add_argument("--checkpoint-every", type=int, default=12)
    run.add_argument("--execute", action="store_true", required=True)

    search = commands.add_parser("run-search-cell", help="execute one gated Fold-A search cell")
    search.add_argument("--trial", type=int, choices=range(24), required=True)
    search.add_argument("--family", choices=FAMILIES, required=True)
    search.add_argument("--seed", choices=SEARCH_SEEDS, type=int, required=True)
    search.add_argument("--archive", type=Path, required=True)
    search.add_argument("--acquisition-receipt", type=Path, required=True)
    search.add_argument("--data-receipt", type=Path, required=True)
    search.add_argument("--environment-receipt", type=Path, required=True)
    search.add_argument("--consumption-root", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--execute", action="store_true", required=True)

    selection = commands.add_parser("select-search", help="select from a complete search registry")
    selection.add_argument("--campaign", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)

    smoke = commands.add_parser("generated-smoke", help="run engineering-only generated evidence")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--seed", type=int, default=11)

    analysis = commands.add_parser("analyze", help="analyze a complete 120-cell campaign")
    analysis.add_argument("--campaign", type=Path, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--draws", type=int, default=10_000)

    shards = commands.add_parser("shard-plan", help="print cells owned by a deterministic shard")
    shards.add_argument("--shard", type=int, required=True)
    shards.add_argument("--total-shards", type=int, required=True)

    verify_bundle = commands.add_parser(
        "verify-run-bundle", help="independently verify a downloaded Kaggle output bundle"
    )
    verify_bundle.add_argument("--bundle", type=Path, required=True)
    verify_bundle.add_argument("--expected-wheel-sha256")
    verify_bundle.add_argument("--expected-package-version")
    verify_bundle.add_argument(
        "--expected-mode",
        choices=(
            "qualify-kaggle",
            "generated-smoke",
            "run-search-cell",
            "select-search",
            "run-cell",
            "analyze",
        ),
    )
    verify_bundle.add_argument("--output", type=Path)
    saved_versions = commands.add_parser(
        "audit-saved-versions", help="verify the complete external Kaggle saved-version inventory"
    )
    saved_versions.add_argument("--ledger", type=Path, required=True)
    saved_versions.add_argument("--output", type=Path)
    return parser


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _fold(identifier: str) -> Fold:
    return next(value for value in FOLDS if value.identifier == identifier)


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    """Execute one explicit no-overwrite S12 operation."""
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "plan":
            _emit({"package": package_identity(), "fixed": registry(), "search": search_space()})
        elif arguments.command == "qualify":
            value = qualify_environment(benchmark_size=arguments.benchmark_size)
            atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "qualify-kaggle":
            value = qualify_kaggle_environment()
            atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "acquire":
            _emit(acquire(family=arguments.family, release=arguments.release, output=arguments.output))
        elif arguments.command == "admit-data":
            _, value = admit_archive(arguments.archive, arguments.acquisition_receipt)
            atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "run-cell":
            provenance = require_execution_receipts(
                environment_receipt=arguments.environment_receipt,
                data_receipt=arguments.data_receipt,
                acquisition_receipt=arguments.acquisition_receipt,
                archive=arguments.archive,
                family=arguments.family,
            )
            data = read_json(arguments.data_receipt)
            panel = parse_archive(
                arguments.archive.read_bytes(), family=arguments.family, cutoff=int(data["cutoff"])
            )
            config = config_from_selection(arguments.selection_receipt)
            provenance["selection_receipt_sha256"] = sha256_file(arguments.selection_receipt)
            resume_state = read_checkpoint(arguments.resume) if arguments.resume else None
            gate_hashes = dict(provenance)
            consumption = claim_consumption(
                arguments.consumption_root,
                phase="fixed",
                identity={"family": arguments.family, "fold": arguments.fold, "seed": arguments.seed},
                gate_hashes=gate_hashes,
                resume=resume_state is not None,
            )
            provenance["consumption"] = consumption
            if resume_state is not None:
                provenance["resumed_checkpoint_sha256"] = resume_state.pop("checkpoint_sha256")
            callback = None
            if arguments.checkpoint_dir is not None:
                callback = partial(write_checkpoint, arguments.checkpoint_dir)
            fold_result = run_fold(
                panel,
                _fold(arguments.fold),
                seed=arguments.seed,
                config=config,
                resume_state=resume_state,
                checkpoint_callback=callback,
                checkpoint_every=arguments.checkpoint_every,
            )
            _emit(write_fold(arguments.output, fold_result, provenance=provenance))
        elif arguments.command == "run-search-cell":
            provenance = require_execution_receipts(
                environment_receipt=arguments.environment_receipt,
                data_receipt=arguments.data_receipt,
                acquisition_receipt=arguments.acquisition_receipt,
                archive=arguments.archive,
                family=arguments.family,
            )
            data = read_json(arguments.data_receipt)
            panel = parse_archive(
                arguments.archive.read_bytes(), family=arguments.family, cutoff=int(data["cutoff"])
            )
            consumption = claim_consumption(
                arguments.consumption_root,
                phase="search",
                identity={
                    "trial": arguments.trial,
                    "family": arguments.family,
                    "seed": arguments.seed,
                },
                gate_hashes=provenance,
            )
            provenance["consumption"] = consumption
            search_result = run_search_cell(panel, trial=arguments.trial, seed=arguments.seed)
            _emit(write_search_result(arguments.output, search_result, provenance=provenance))
        elif arguments.command == "select-search":
            if arguments.output.resolve().is_relative_to(arguments.campaign.resolve()):
                raise ValueError("selection output must be outside the immutable search campaign")
            value = select_search(arguments.campaign)
            atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "generated-smoke":
            panel = generated_panel(months=432, seed=811)
            result = run_fold(panel, _fold("A"), seed=arguments.seed)
            _emit(
                write_fold(
                    arguments.output,
                    result,
                    provenance={"engineering_only": True, "generated": True},
                )
            )
        elif arguments.command == "analyze":
            if arguments.output.resolve().is_relative_to(arguments.campaign.resolve()):
                raise ValueError("analysis output must be outside the immutable fixed campaign")
            value = decision_json(analyze_campaign(arguments.campaign, draws=arguments.draws))
            atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "shard-plan":
            _emit(
                {
                    "shard": arguments.shard,
                    "total_shards": arguments.total_shards,
                    "cells": [
                        {"family": value.family, "fold": value.fold, "seed": value.seed}
                        for value in shard_cells(arguments.shard, arguments.total_shards)
                    ],
                }
            )
        elif arguments.command == "verify-run-bundle":
            if arguments.output is not None and arguments.output.resolve().is_relative_to(
                arguments.bundle.resolve()
            ):
                raise ValueError("verification output must be outside the immutable run bundle")
            value = verify_run_bundle(
                arguments.bundle,
                expected_wheel_sha256=arguments.expected_wheel_sha256,
                expected_package_version=arguments.expected_package_version,
                expected_mode=arguments.expected_mode,
            )
            if arguments.output is not None:
                atomic_json(arguments.output, value)
            _emit(value)
        elif arguments.command == "audit-saved-versions":
            value = audit_saved_versions(arguments.ledger)
            if arguments.output is not None:
                atomic_json(arguments.output, value)
            _emit(value)
        else:
            _fail("unknown command")
    except (S12Error, OSError, ValueError, KeyError) as exc:
        _fail(f"FREM-S12 failed closed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
