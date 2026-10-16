"""Only the consumed fixed generated calendar-scale workload may enter this worker."""

import argparse
from pathlib import Path

from frem_s10.errors import ExecutionStateError, IntegrityError, S10Error
from frem_s10.identities import JSONObject, checked_json, digest_bytes, object_json
from frem_s10.runtime.artifacts import publish_json, read_verified_json
from frem_s10.runtime.qualification import (
    QUALIFICATION_CONTRACT,
    deep_worker_evidence,
    file_reference,
    qualification_calendar,
    validate_request,
)


def run_generated(
    request: JSONObject, *, artifact_root: Path, output: Path, identity: str, attempt: str
) -> JSONObject:
    """Exercise full dates, six causal kernels and all192 ledgers without market bytes."""
    from dataclasses import replace

    import numpy as np

    from frem_s10.data.production import _sessions
    from frem_s10.domain import SessionPanel
    from frem_s10.experiment.model_archive import ModelArchiveWriter
    from frem_s10.experiment.replay import SessionTiming, prepare_campaign
    from frem_s10.runtime.worker import run_provided_campaign

    clocks = _sessions(dict(qualification_calendar(request["calendar"], request["clock"], artifact_root)))
    count = len(clocks)
    index = np.arange(count, dtype=np.float64)[:, None]
    phase = np.array([0.0, 0.7, 1.4])[None, :]
    prices = 100 * np.exp(
        0.00008 * index + 0.035 * np.sin(index / 19 + phase) + 0.018 * np.sin(index / 71 + 2 * phase)
    )
    closes = prices * (1 + 0.002 * np.sin(index / 7 + phase))
    dividends = np.zeros((count, 3), dtype=np.float64)
    dividends[::63, 0], dividends[::21, 1] = 0.25, 0.1
    aligned: list[tuple[str | None, str | None, str | None]] = []
    timing: list[SessionTiming] = []
    for row, clock in enumerate(clocks):
        first = f"generated-SPY-{row}" if dividends[row, 0] else None
        second = f"generated-TLT-{row}" if dividends[row, 1] else None
        aligned.append((first, second, None))
        timing.append(
            SessionTiming(
                clock.open_us,
                clock.close_us,
                (() if first is None else (first,), () if second is None else (second,), ()),
            )
        )
    panel = SessionPanel(
        tuple(clock.day for clock in clocks),
        prices,
        closes,
        dividends,
        ("generated-SPY", "generated-TLT", "generated-GLD"),
        tuple(tuple(item for item in row if item is not None) for row in aligned),
        asset_event_ids=tuple(aligned),
    )
    inputs = replace(prepare_campaign(panel, tuple(timing)), evidence_scope="GENERATED_ENGINEERING")
    archive = output / "model_archive"
    archive.mkdir(exist_ok=False)
    writer = ModelArchiveWriter(
        archive, experiment_id=attempt, evidence_scope="GENERATED_ENGINEERING", binding_id=identity
    )
    result = run_provided_campaign(inputs, output, writer, attempt)
    archive_id = writer.finish()
    archive_document = object_json(
        read_verified_json(archive / "manifest.json", archive_id, maximum_bytes=4_194_304)
    )
    counts = object_json(archive_document["shape_counts"])
    stress = stress_actions()
    publish_json(output, "qualification-stress.json", stress)
    if counts["max_library"] != 520:
        raise IntegrityError("generated run did not reach declared support/action bounds")
    evidence = object_json(result["evidence"])
    audit = object_json(result["audit"])
    result = {
        "schema": "s10-qualification-worker-v1",
        "request_sha256": identity,
        "contract": QUALIFICATION_CONTRACT,
        "counts": {
            "sessions": count,
            "model_variants": 6,
            "transitions": counts["transitions"],
            "ledgers": 192,
            "maximum_library": counts["max_library"],
            "maximum_actions": 610,
            "fold_a_blocks": len(inputs.windows[0].origin_indices),
            "fold_b_blocks": len(inputs.windows[1].origin_indices),
        },
        "archive": file_reference(archive / "manifest.json", artifact_root),
        "campaign": file_reference(output / str(evidence["path"]), artifact_root),
        "analysis": file_reference(output / str(audit["path"]), artifact_root),
        "stress": file_reference(output / "qualification-stress.json", artifact_root),
    }
    del inputs, panel, prices, closes, dividends, index, phase, timing, aligned, writer
    deep_worker_evidence(result, request=request, identity=identity, artifact_root=artifact_root)
    return result


def stress_actions() -> JSONObject:
    """Actually evaluate both policies over610 actions at all fixed library ladder sizes."""
    from dataclasses import asdict
    from time import monotonic

    import numpy as np

    from frem_s10.domain import ForecastDistribution
    from frem_s10.portfolio.actions import Valuation
    from frem_s10.portfolio.policy import select_policy

    rows = []
    valuation = Valuation(np.array([20_001.0, 20_003.0, 20_007.0]), 39_989.0)
    for n in (64, 128, 260, 520):
        index = np.arange(n, dtype=np.float64)[:, None]
        g = 1 + 0.05 * np.sin(index / 7 + np.array([0.0, 1.0, 2.0])[None, :])
        law = ForecastDistribution(f"generated-stress-{n}", g, np.zeros_like(g), np.full(n, 1 / n))
        for cost in (0, 10, 20, 40):
            for robust in (False, True):
                started = monotonic()
                decision = select_policy(valuation, law, one_way_cost=cost / 10000, robust=robust)
                if decision.certificate.candidate_count != 610:
                    raise IntegrityError("maximum-action stress did not evaluate610 actual candidates")
                rows.append(
                    {
                        "atoms": n,
                        "cost_bps": cost,
                        "robust": robust,
                        "wall_seconds": monotonic() - started,
                        "certificate": asdict(decision.certificate),
                    }
                )
    return {
        "schema": "s10-generated-policy-stress-v1",
        "evidence_scope": "GENERATED_ENGINEERING",
        "rows": checked_json(rows),
    }


def execute_qualification_worker(
    *,
    request_path: Path,
    identity: str,
    artifact_root: Path,
    source_root: Path,
    consumption_root: Path,
    attempt: str,
    output: Path,
) -> None:
    """Recheck exact consumption, source identity and environment before NumPy import."""
    request = object_json(read_verified_json(request_path, identity))
    marker_path = consumption_root / f"campaign-{identity}.json"
    marker = object_json(read_verified_json(marker_path, digest_bytes(marker_path.read_bytes())))
    if (
        marker.get("state"),
        marker.get("freeze_id"),
        marker.get("attempt_id"),
        marker.get("output_path"),
        marker.get("resume_permitted"),
    ) != ("CONSUMED", identity, attempt, str(output), False):
        raise ExecutionStateError("exact consumed qualification attempt required")
    validate_request(request, artifact_root=artifact_root, source_root=source_root)
    publish_json(
        output,
        "qualification-started.json",
        {
            "request_sha256": identity,
            "attempt_id": attempt,
            "scope": "GENERATED_ENGINEERING",
            "scientific_support_claimed": False,
        },
    )
    result = run_generated(
        request, artifact_root=artifact_root, output=output, identity=identity, attempt=attempt
    )
    publish_json(output, "qualification.json", result)


def main(argv: list[str] | None = None) -> int:
    """Expose only controller-generated arguments, with no arbitrary worker code."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "artifact-root", "source-root", "consumption-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args(argv)
    try:
        execute_qualification_worker(
            request_path=args.request,
            identity=args.sha256,
            artifact_root=args.artifact_root,
            source_root=args.source_root,
            consumption_root=args.consumption_root,
            attempt=args.attempt_id,
            output=Path.cwd(),
        )
    except (S10Error, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
