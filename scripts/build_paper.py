"""Render the methods manuscript with LaTeX; does not execute research code."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    paper = Path(__file__).resolve().parents[1] / "paper"
    for name in ("pdflatex", "bibtex"):
        if shutil.which(name) is None:
            print(f"Required document tool is not installed: {name}", file=sys.stderr)
            return 2
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    # A fresh document build prevents stale auxiliary files from hiding a
    # bibliography repair or changing the generated page references.
    with tempfile.TemporaryDirectory(prefix="frem_paper_") as temporary:
        build = Path(temporary)
        for filename in ("main.tex", "references.bib"):
            shutil.copyfile(paper / filename, build / filename)
        for command in commands:
            result = subprocess.run(
                command, cwd=build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
            )
            if result.returncode:
                (paper / "build_failure.log").write_text(result.stdout, encoding="utf-8")
                print(result.stdout[-5000:], file=sys.stderr)
                return result.returncode
        for filename in ("main.pdf", "main.bbl", "main.log"):
            shutil.copyfile(build / filename, paper / filename)
    print(f"Rendered document: {paper / 'main.pdf'}")
    print("Inspect rendered pages and the LaTeX log before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
