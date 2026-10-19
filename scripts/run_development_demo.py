#!/usr/bin/env python3
"""Write the fixed, labelled synthetic development demonstration to a new directory."""
import argparse
import os
from pathlib import Path
import sys

# Set before NumPy/sklearn imports; the library also scopes BLAS to one thread.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wharton_lab.research.development_demo import run_development_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_development_demo(args.output_dir)
    print(f"SYNTHETIC DEVELOPMENT ONLY: {len(result['evaluated_samples'])} scored sessions; "
          f"receipts written to {args.output_dir}")


if __name__ == "__main__":
    main()
