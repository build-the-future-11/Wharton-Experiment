"""Display the frozen primary controls; never select or refit a policy."""
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).parent
run = Path("outputs/frem_s8_2020_2022_20261014_attempt1")
summary = json.loads((root / "SUMMARY.json").read_text())
economics = json.loads((run / "economics.json").read_text())
benchmarks = json.loads((run / "benchmarks.json").read_text())
figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
series = [
    ("S8, four-week holdings", economics["s8"]["10"]["s8_h4"], "#2166ac"),
    ("S6 memory, same policy", economics["memory"]["10"]["s8_h4"], "#b35806"),
    ("Fixed forecast average, same policy", economics["equal"]["10"]["s8_h4"], "#777777"),
    ("Equal-weight assets", benchmarks["equal_weight"]["10"], "#1b7837"),
    ("Cash", benchmarks["cash"]["10"], "#333333"),
]
for name, ledger, color in series:
    rows = ledger["periods"]
    times = [datetime.fromtimestamp(rows[0]["time"], UTC)] + [
        datetime.fromtimestamp(r["end"], UTC) for r in rows
    ]
    wealth = [100000] + [r["nav_end"] for r in rows]
    wealth[-1] = ledger["terminal_wealth"]
    axes[0].plot(times, [v / 1000 for v in wealth], label=name, color=color, linewidth=1.5)
axes[0].set_title("Net wealth · 10 bps one-way costs")
axes[0].set_ylabel("USD thousands · initial capital $100,000")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for y, control in enumerate(("equal", "memory")):
    for offset, block, color in ((-.07, "6", "#2166ac"), (.07, "12", "#777777")):
        value = summary["paired_prediction"][control][block]
        mean = value["mean"]
        lo, hi = value["interval95"]
        axes[1].errorbar(mean, y + offset, xerr=[[mean-lo], [hi-mean]], fmt="o",
                         color=color, capsize=3, label=f"{block}-date blocks" if y == 0 else None)
axes[1].axvline(0, color="#555555", linewidth=.8)
axes[1].axvline(.01, color="#b2182b", linestyle="--", label="Practical threshold")
axes[1].set_yticks([0, 1], ["Fixed averaging − S8", "S6 memory − S8"])
axes[1].set_ylim(-.5, 1.5)
axes[1].set_title("Predictive gain and 95% date-block intervals")
axes[1].set_xlabel("NLL improvement (nats)\nPositive favors S8", fontsize=10)
axes[1].set_xticks([-.005, 0, .005, .01])
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
for axis in axes:
    axis.grid(alpha=.18)
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle("S8 development evaluation · 2020–2022 · three seeds", fontsize=13)
figure.tight_layout(rect=(0, .04, 1, .94))
figure.text(.5, .01, "Retrospective ETF development; historical exposure exists. No 2019 tuning or protected confirmation.",
            ha="center", fontsize=8, color="#555555")
path = root / "RESULTS_FINAL.png"
if path.exists():
    raise FileExistsError(path)
figure.savefig(path, dpi=170)
plt.close(figure)
