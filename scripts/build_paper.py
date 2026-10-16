"""Render the methods manuscript with LaTeX; does not execute research code."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    paper = Path(__file__).resolve().parents[1] / "paper"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New output directory; refuses an existing path")
    args = parser.parse_args()
    output = args.output_dir.resolve() if args.output_dir else paper
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    engine = "pdflatex+bibtex"
    if not all(shutil.which(name) for name in ("pdflatex", "bibtex")):
        if shutil.which("tectonic") is None:
            print("Install pdflatex + bibtex, or tectonic, to render the manuscript.", file=sys.stderr)
            return 2
        engine = "tectonic"
        commands = [["tectonic", "--only-cached", "--untrusted", "--keep-logs",
                     "--keep-intermediates", "main.tex"]]
    # A fresh document build prevents stale auxiliary files from hiding a
    # bibliography repair or changing the generated page references.
    with tempfile.TemporaryDirectory(prefix="frem_paper_") as temporary:
        build = Path(temporary)
        sources = sorted([*paper.glob("*.tex"), *paper.glob("*.bib")])
        source_hashes = {}
        for source in sources:
            shutil.copyfile(source, build / source.name)
            source_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        for command in commands:
            result = subprocess.run(
                command, cwd=build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
            )
            if result.returncode:
                (output / "build_failure.log").write_text(result.stdout, encoding="utf-8")
                print(result.stdout[-5000:], file=sys.stderr)
                return result.returncode
        for filename in ("main.pdf", "main.bbl", "main.log"):
            shutil.copyfile(build / filename, output / filename)
        (output / "build_receipt.json").write_text(json.dumps({
            "engine": engine, "source_sha256": source_hashes,
            "pdf_sha256": hashlib.sha256((output / "main.pdf").read_bytes()).hexdigest(),
            "scope": "Document compilation only; no research execution or submission readiness",
        }, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered document: {output / 'main.pdf'}")
    print("Inspect rendered pages and the LaTeX log before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
