"""CLI: python -m wharton_lab.orchestration.cli smoke|pilot|overnight|full"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from wharton_lab.orchestration.executor import execute_row
from wharton_lab.orchestration.manifest import (
    ManifestRow,
    filter_profile,
    load_manifest,
    manifest_path,
    write_manifest,
)
from wharton_lab.orchestration.profiles import ProfileName, get_profile


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_execution_state(
    root: Path,
    profile: str,
    *,
    completed: int,
    skipped: int,
    failed: int,
    total: int,
    elapsed_sec: float,
) -> None:
    state = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_profile": profile,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "total": total,
        "elapsed_sec": round(elapsed_sec, 2),
    }
    (root / "EXECUTION_STATE.json").write_text(json.dumps(state, indent=2, sort_keys=True))
    next_lines = [
        f"# Next action ({profile})",
        "",
        f"- Completed: {completed}/{total} (skipped resume: {skipped}, failed: {failed})",
        f"- Last run elapsed: {elapsed_sec:.1f}s",
        "",
    ]
    if failed:
        next_lines.append("- Fix failed manifest rows and re-run with `--resume`.")
    elif completed + skipped < total:
        next_lines.append(f"- Re-run: `python -m wharton_lab.orchestration.cli {profile.lower()} --resume`")
    else:
        nxt = {"smoke": "pilot", "pilot": "overnight", "overnight": "full", "full": None}
        follow = nxt.get(profile.lower())
        if follow:
            next_lines.append(f"- Profile complete. Suggested next: `{follow}`.")
        else:
            next_lines.append("- All registered full-profile cells complete. Run `bash scripts/build_reports.sh`.")
    (root / "NEXT_ACTION.md").write_text("\n".join(next_lines) + "\n")


def run_profile(
    profile_name: str,
    *,
    resume: bool = False,
    hours: float | None = None,
) -> int:
    root = repo_root()
    prof = get_profile(profile_name.upper())
    key = prof.name.value.lower()
    mpath = manifest_path(root)
    rows = load_manifest(mpath)
    subset = filter_profile(rows, key)
    if not subset:
        print(f"No manifest rows for profile={key}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    deadline = t0 + hours * 3600 if hours else None
    completed = skipped = failed = 0

    for row in subset:
        if deadline and time.perf_counter() >= deadline:
            break
        if resume and row.status == "COMPLETED":
            skipped += 1
            continue
        try:
            _, receipt_path = execute_row(
                row,
                repo_root=root,
                profile=prof,
                n_steps=prof.n_steps,
            )
            for i, r in enumerate(rows):
                if r.run_id == row.run_id:
                    d = r.as_dict()
                    d.update(
                        status="COMPLETED",
                        result_path=str(receipt_path.relative_to(root)),
                        failure_reason="",
                    )
                    rows[i] = ManifestRow(**d)
                    break
            completed += 1
        except Exception as exc:
            for i, r in enumerate(rows):
                if r.run_id == row.run_id:
                    d = r.as_dict()
                    d.update(status="FAILED", failure_reason=str(exc)[:500])
                    rows[i] = ManifestRow(**d)
                    break
            failed += 1
        # Flush periodically (not every cell) — 591KB CSV rewrite was a bottleneck.
        if (completed + failed) % 25 == 0 or (completed + failed) == 1:
            write_manifest(mpath, rows)
            elapsed = time.perf_counter() - t0
            write_execution_state(
                root,
                key,
                completed=completed,
                skipped=skipped,
                failed=failed,
                total=len(subset),
                elapsed_sec=elapsed,
            )
            print(
                f"[{key}] done={completed} skip={skipped} fail={failed} "
                f"total={len(subset)} elapsed={elapsed:.1f}s last={row.run_id}",
                flush=True,
            )

    write_manifest(mpath, rows)
    elapsed = time.perf_counter() - t0
    write_execution_state(
        root,
        key,
        completed=completed,
        skipped=skipped,
        failed=failed,
        total=len(subset),
        elapsed_sec=elapsed,
    )
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wharton experiment orchestration")
    parser.add_argument(
        "profile",
        choices=["smoke", "pilot", "overnight", "full"],
        help="Experiment profile",
    )
    parser.add_argument("--resume", action="store_true", help="Skip COMPLETED manifest rows")
    parser.add_argument("--hours", type=float, default=None, help="Time budget (overnight)")
    args = parser.parse_args(argv)
    return run_profile(args.profile, resume=args.resume, hours=args.hours)


if __name__ == "__main__":
    raise SystemExit(main())
