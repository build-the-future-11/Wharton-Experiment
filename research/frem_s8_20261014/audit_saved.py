"""Separate standard-library arithmetic and chronology audit, no fitting imports."""

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def day(t):
    return datetime.fromtimestamp(t, UTC).date().isoformat()


def verify(root, data_root, freeze, predecessors):
    identity = read(root / "data_identity.json")
    frozen = read(freeze)
    for files in [frozen["sources"], read(predecessors)["files"], identity["files"]]:
        for name, expected in files.items():
            assert digest(name) == expected, name
    assert identity["freeze_sha256"] == digest(freeze)
    data = read(data_root / "windows.json")
    expected_rows = {}
    max_return_error, feature_error, raw_count = 0.0, 0.0, 0
    for name, part in data["partitions"].items():
        for asset in ("SPY", "TLT", "GLD"):
            raw = read(
                (data_root if name == "evaluation" else Path("data/s6_real_etf_20261014")) / f"{asset}.json"
            )
            prices = {day(r["timestamp"]): r["open"] for r in raw["rows"]}
            divs = [(day(d["date"]), d["amount"]) for d in raw["dividends"].values()]
            features = []
            for left, right in zip(raw["rows"][:-1], raw["rows"][1:], strict=True):
                midnight = (
                    datetime.fromisoformat(day(right["timestamp"])).replace(tzinfo=UTC).timestamp() + 86400
                )
                features.append((midnight, math.log(right["close"] / left["close"])))
            for i, a in enumerate(part["assets"]):
                if a != asset:
                    continue
                start, end = part["origin_times"][i], part["maturity_times"][i]
                s, e = day(start), day(end)
                assert not s.startswith("2019") and e < "2023-01-01"
                expected = (prices[e] + sum(v for d, v in divs if s < d <= e)) / prices[s] - 1
                max_return_error = max(
                    max_return_error,
                    abs(expected - part["simple_returns"][i]),
                    abs(math.expm1(part["y"][i]) - expected),
                )
                past = [x for x in features if x[0] < start][-8:]
                assert [x[0] for x in past] == part["window_times"][i] == part["window_available"][i]
                feature_error = max(
                    feature_error, *(abs(x[1] - v[0]) for x, v in zip(past, part["windows"][i], strict=True))
                )
                if name == "train":
                    future = [x for x in features if x[0] > start][:8]
                    assert [x[0] for x in future] == part["future_times"][i] == part["future_available"][i]
                    feature_error = max(
                        feature_error,
                        *(abs(x[1] - v[0]) for x, v in zip(future, part["future_windows"][i], strict=True)),
                    )
                expected_rows[(start, asset)] = (end, part["available_times"][i], part["y"][i], expected)
                raw_count += 1
    assert max_return_error < 1e-12 and feature_error < 1e-12
    completion = read(root / "completion.json")
    variants = set(completion["variants"])
    assert len(variants) == 15 and completion["seeds"] == [11, 29, 47]
    arrays = [
        read(root / f"seed{s}/{stage}_predictions.json")
        for s in (11, 29, 47)
        for stage in ("selection", "evaluation")
    ] + [read(root / "ensembles.json")]
    max_nll_error, nll_count, gate_checks = 0.0, 0, 0
    for records in arrays:
        dates = sorted({r["date"] for r in records})
        assert len(records) == len(dates) * len(variants)
        assert len({(r["date"], r["variant"]) for r in records}) == len(records)
        for r in records:
            assert r["feature_available_max"] < r["date"] < r["end"] <= r["available"]
            assert r["assets"] == ["SPY", "TLT", "GLD"]
            nlls = []
            for i, asset in enumerate(r["assets"]):
                end, avail, y, ret = expected_rows[(r["date"], asset)]
                assert r["end"] == end and r["available"] == avail
                assert r["y"][i] == y and abs(r["simple_returns"][i] - ret) < 1e-12
                ws = r["weights"][i]
                assert abs(sum(ws) - 1) < 1e-10 and all(w >= 0 for w in ws)
                logs = [
                    math.log(w)
                    + math.lgamma((df + 1) / 2)
                    - math.lgamma(df / 2)
                    - 0.5 * math.log(df * math.pi)
                    - math.log(s)
                    - (df + 1) / 2 * math.log1p(((y - mu) / s) ** 2 / df)
                    for mu, s, df, w in zip(r["mean"][i], r["scale"][i], r["df"][i], ws, strict=True)
                    if w
                ]
                top = max(logs)
                value = -top - math.log(sum(math.exp(x - top) for x in logs))
                max_nll_error = max(max_nll_error, abs(value - r["nll"][i]))
                nlls.append(value)
                nll_count += 1
            max_nll_error = max(max_nll_error, abs(sum(nlls) / 3 - r["date_nll"]))
    assert max_nll_error < 1e-10
    # Reconstruct the soft gate independently from saved strategy scores and release times.
    for seed in (11, 29, 47):
        records = read(root / f"seed{seed}/selection_predictions.json") + read(
            root / f"seed{seed}/evaluation_predictions.json"
        )
        lookup = {(r["date"], r["variant"]): r for r in records}
        trace = read(root / f"seed{seed}/trace.json")
        for gate_name, strategy, weight_name in [
            ("local_gate", "mixture_ungated", "local_expert_weights"),
            ("global_gate", "mixture_global", "global_expert_weights"),
        ]:
            q, pending = [0.5, 0.5], []
            for t in trace:
                now = t["date"]
                ready = sorted((p for p in pending if p[0] < now), key=lambda p: p[0])
                for _, losses in ready:
                    top = max(math.log(q[i]) - losses[i] for i in range(2))
                    z = [math.exp(math.log(q[i]) - losses[i] - top) for i in range(2)]
                    q = [(1 - 1 / 52) * v / sum(z) + 1 / 104 for v in z]
                pending = [p for p in pending if p[0] >= now]
                assert max(abs(a - b) for a, b in zip(q, t[gate_name], strict=True)) < 1e-12
                name = "s8" if gate_name == "local_gate" else "s8_global"
                weights = [q[0] / 6 + q[1] * v for v in t[weight_name]]
                for row in lookup[(now, name)]["weights"]:
                    assert max(abs(a - b) for a, b in zip(row, weights, strict=True)) < 1e-12
                pending.append(
                    (
                        t["available"],
                        [lookup[(now, "equal")]["date_nll"], lookup[(now, strategy)]["date_nll"]],
                    )
                )
                gate_checks += 1
    returns = {
        r["date"]: dict(zip(r["assets"], r["simple_returns"], strict=True))
        for r in read(root / "ensembles.json")
        if r["variant"] == "equal"
    }
    economics, benchmarks = read(root / "economics.json"), read(root / "benchmarks.json")
    ledgers = [
        (v, c, p, ledger)
        for v, costs in economics.items()
        for c, policies in costs.items()
        for p, ledger in policies.items()
    ]
    ledgers += [(v, c, "benchmark", ledger) for v, costs in benchmarks.items() for c, ledger in costs.items()]
    residual = 0.0
    for variant, cost, policy, ledger in ledgers:
        rate = int(cost) / 10000
        holdings, cash = dict.fromkeys(("SPY", "TLT", "GLD"), 0.0), 100000.0
        assert len(ledger["periods"]) == completion["evaluation_dates"]
        for step, r in enumerate(ledger["periods"]):
            nav = cash + sum(holdings.values())
            post = {a: r["holdings"][a] / (1 + returns[r["time"]][a]) for a in holdings}
            notional = sum(abs(post[a] - holdings[a]) for a in holdings)
            residual = max(
                residual,
                abs(nav - r["nav_start"]),
                abs(notional - r["notional"]),
                abs(rate * notional - r["fee"]),
                abs(nav - r["fee"] - sum(post.values()) - r["cash"]),
            )
            gross = r["cash"] + sum(r["holdings"].values())
            residual = max(residual, abs(gross - r.get("liquidation_fee", 0) - r["nav_end"]))
            if policy == "s8_h4" and step % 4:
                assert r["decision"] is None and r["notional"] == 0
            holdings, cash = r["holdings"], r["cash"]
        terminal = cash + sum(holdings.values()) - rate * sum(holdings.values())
        residual = max(
            residual,
            abs(terminal - ledger["terminal_wealth"]),
            abs(sum(r["fee"] for r in ledger["periods"]) + rate * sum(holdings.values()) - ledger["fees"]),
        )
    assert residual < 1e-7
    return {
        "status": "PASS",
        "frozen_files": len(frozen["sources"]),
        "predecessor_files_unchanged": len(read(predecessors)["files"]),
        "raw_labels_reconciled": raw_count,
        "max_return_error": max_return_error,
        "max_feature_error": feature_error,
        "nll_values_recomputed": nll_count,
        "max_nll_error": max_nll_error,
        "soft_gate_decisions_reconstructed": gate_checks,
        "ledgers_verified": len(ledgers),
        "max_accounting_residual": residual,
        "2019_inputs_used": False,
        "protected_access": False,
        "scope": "Separate saved-output arithmetic and chronology audit; same host, no independent retraining.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.root, args.data, args.freeze, args.freeze.parent / "predecessor_manifest.json")
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
