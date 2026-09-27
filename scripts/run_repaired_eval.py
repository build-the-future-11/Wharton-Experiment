"""CLI: repaired-pipeline exploratory evaluation (post-lockbox, not confirmatory)."""

from __future__ import annotations

from wharton_lab.evaluation.repaired import run_repaired_eval
from wharton_lab.utils.paths import repo_root


def main() -> int:
    payload = run_repaired_eval(repo_root())
    print(f"Wrote runs/repaired/REPAIRED_RESULTS.json ({len(payload['contrasts'])} contrasts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
