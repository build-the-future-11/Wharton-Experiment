"""CLI: multiplicity-corrected scoreboard claims."""

from __future__ import annotations

from pathlib import Path

from wharton_lab.evaluation.multiplicity import write_multiplicity_report
from wharton_lab.utils.paths import repo_root


def main() -> int:
    payload = write_multiplicity_report(repo_root())
    print(f"Wrote multiplicity report; significant={payload['n_significant_better']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
