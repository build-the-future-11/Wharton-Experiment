"""Figure: out-of-sample R² vs historical mean, repaired causal pipeline vs legacy leaky features.

Source data: runs/repaired/REPAIRED_RESULTS.json (written by scripts/run_repaired_eval.py).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs" / "repaired" / "REPAIRED_RESULTS.json"
OUT = ROOT / "reports" / "figures"
ARMS = (
    ("ridge_legacy", "Frozen B_RIDGE, causal features"),
    ("ridge_std", "Standardised ridge, causal features"),
    ("ridge_legacy_on_leaky_features", "Frozen B_RIDGE, legacy leaky features"),
)


def main() -> int:
    payload = json.loads(SRC.read_text())
    rows = [c for c in payload["contrasts"] if c["benchmark"] == "mean"]
    datasets = list(payload["datasets"])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "repaired_oos_r2.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "challenger", "oos_r2_vs_mean", "dm_p", "bh_q", "n_obs"])
        for c in rows:
            w.writerow([c["dataset"], c["challenger"], c["oos_r2_vs_benchmark"], c["dm_p"], c["bh_q"], c["n_obs"]])

    fig, (ax_c, ax_l) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    width = 0.38
    x = range(len(datasets))
    for k, (arm, label) in enumerate(ARMS[:2]):
        vals = [
            100 * next(c["oos_r2_vs_benchmark"] for c in rows if c["dataset"] == d and c["challenger"] == arm)
            for d in datasets
        ]
        ax_c.bar([i + (k - 0.5) * width for i in x], vals, width, label=label)
    ax_c.axhline(0, color="black", lw=0.8)
    ax_c.set_xticks(list(x))
    ax_c.set_xticklabels([d.replace("synthetic_signal_", "synth ").replace("etf_track_a", "ETF (AGG)") for d in datasets])
    ax_c.set_ylabel("OOS R² vs historical mean (%)")
    ax_c.set_title("Repaired causal pipeline: no skill over the mean")
    ax_c.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    synth = [d for d in datasets if d.startswith("synthetic")]
    leaky = [
        100 * next(c["oos_r2_vs_benchmark"] for c in rows if c["dataset"] == d and c["challenger"] == ARMS[2][0])
        for d in synth
    ]
    ax_l.bar(range(len(synth)), leaky, color="tab:red")
    ax_l.set_xticks(range(len(synth)))
    ax_l.set_xticklabels([d.replace("synthetic_signal_", "") for d in synth])
    ax_l.set_ylim(0, 100)
    ax_l.set_ylabel("OOS R² vs historical mean (%)")
    ax_l.set_title("Legacy features (leak demo)")
    fig.suptitle(
        "1-step daily return, 8×100 disjoint expanding-window test folds (ETF: lockbox-overlap fold dropped)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT / "repaired_oos_r2.png", dpi=160)
    fig.savefig(OUT / "repaired_oos_r2.pdf")
    print(f"Wrote {OUT / 'repaired_oos_r2.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
