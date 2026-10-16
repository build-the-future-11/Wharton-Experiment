"""Attribute saved continuous long-only, zero-cash-yield ledgers without replay."""

import math
from datetime import UTC, datetime

from .contracts import finite


def audit_ledger(ledger, *, cost_bps, tolerance=1e-6, decision_assets=None):
    """Reconcile realized dollar contributions, fees and origin-year segments.

    This consumes existing decisions only. It neither chooses new targets nor
    creates a counterfactual strategy. Supports the S8 zero-cash-yield contract.
    """
    cost_bps = finite(cost_bps, "cost bps")
    tolerance = finite(tolerance, "tolerance")
    if not 0 <= cost_bps < 10000 or tolerance <= 0:
        raise ValueError("invalid cost or tolerance")
    rows = ledger["periods"]
    if not rows:
        raise ValueError("nonempty ledger required")
    names = tuple(sorted(rows[0]["holdings"]))
    if not names:
        raise ValueError("nonempty asset set required")
    contributions = {a: [] for a in names}
    exposures = {a: [] for a in names}
    years = {}
    trade_fees, max_error, previous_end, previous_nav = [], 0.0, None, None
    max_ratio_error = 0.0
    decisions, positive_means = 0, {a: 0 for a in names}
    initial = finite(rows[0]["nav_start"], "initial NAV")
    if initial <= 0:
        raise ValueError("positive initial NAV required")
    peak, max_drawdown = initial, 0.0

    def close(left, right, name, *, ratio=False):
        nonlocal max_error, max_ratio_error
        left, right = finite(left, name), finite(right, name)
        error = abs(left - right)
        if ratio:
            max_ratio_error = max(max_ratio_error, error)
        else:
            max_error = max(max_error, error)
        if error > (1e-12 if ratio else tolerance):
            raise ValueError(f"{name} mismatch: {left} != {right}")

    for index, row in enumerate(rows):
        start, end, nav_start, nav_end, fee, notional, cash = [
            finite(row[k], k) for k in ("time", "end", "nav_start", "nav_end", "fee", "notional", "cash")
        ]
        if start >= end or min(nav_start, nav_end) <= 0 or min(fee, notional, cash) < 0:
            raise ValueError("invalid ledger time, NAV or cash/cost")
        if previous_end is not None:
            if start != previous_end:
                raise ValueError("ledger gap or overlap")
            close(nav_start, previous_nav, "NAV continuity")
        if set(row["holdings"]) != set(names):
            raise ValueError("asset grid changed")
        holdings = {a: finite(row["holdings"][a], "holding") for a in names}
        post = nav_start - fee
        if post <= 0:
            raise ValueError("nonpositive post-fee NAV")
        if "weights" in row:
            if set(row["weights"]) != set(names):
                raise ValueError("weight asset grid changed")
            weights = {a: finite(row["weights"][a], "weight") for a in names}
        else:
            if set(row.get("contributions", {})) != set(names):
                raise ValueError("weights or complete contributions required")
            weights = {
                a: (holdings[a] - finite(row["contributions"][a], "contribution")) / post
                for a in names
            }
        if min(weights.values()) < 0 or min(holdings.values()) < 0 or math.fsum(weights.values()) > 1 + 1e-10:
            raise ValueError("invalid long-only weights or holdings")
        close(fee, cost_bps / 10000 * notional, "trade fee")
        close(cash + post * math.fsum(weights.values()), post, "self-financing cash")
        final = index == len(rows) - 1
        root_only_liquidation = final and 'liquidation_fee' not in row
        liquidation = finite(
            ledger.get('liquidation_fee', 0) if root_only_liquidation else row.get('liquidation_fee', 0),
            "liquidation fee",
        )
        stored_end = nav_end
        if root_only_liquidation:
            # S7 stores terminal liquidation outside its last period; S8 and
            # the signal ledger store it inside the last period. Normalize
            # only the audit view; never rewrite either historical format.
            nav_end -= liquidation
        if liquidation < 0 or (index != len(rows) - 1 and liquidation != 0):
            raise ValueError("invalid liquidation placement")
        if index == len(rows) - 1:
            close(liquidation, cost_bps / 10000 * math.fsum(holdings.values()), "liquidation fee")
        close(cash + math.fsum(holdings.values()) - liquidation, nav_end, "ending holdings")
        if "net_factor" in row:
            close(row["net_factor"], stored_end / nav_start, "net factor", ratio=True)
        year = str(datetime.fromtimestamp(start, UTC).year)
        part = years.setdefault(year, {
            "periods": 0, "start_nav": nav_start, "end_nav": nav_start,
            "asset_contributions": {a: 0.0 for a in names},
            "fees": 0.0, "risky_weight_sum": 0.0,
        })
        per_asset = {}
        for a in names:
            pnl = holdings[a] - post * weights[a]
            per_asset[a] = pnl
            contributions[a].append(pnl)
            exposures[a].append(weights[a])
            part["asset_contributions"][a] += pnl
            if "contributions" in row:
                if set(row["contributions"]) != set(names):
                    raise ValueError("contribution asset grid changed")
                close(pnl, row["contributions"][a], "saved contribution")
        close(math.fsum(per_asset.values()) - fee - liquidation, nav_end - nav_start, "period PnL")
        part["periods"] += 1
        part["end_nav"] = nav_end
        part["fees"] += fee + liquidation
        part["risky_weight_sum"] += math.fsum(weights.values())
        trade_fees.append(fee)
        decision = row.get("decision")
        if decision is not None:
            # Positional inputs need an explicit source order; JSON object key
            # ordering is not an asset-identity contract.
            if (decision_assets is None or set(decision_assets) != set(names)
                    or len(decision_assets) != len(names)):
                raise ValueError("explicit decision asset order required")
            ordered = list(decision_assets)
            if len(decision["mean"]) != len(ordered) or len(decision["target"]) != len(ordered):
                raise ValueError("decision dimension mismatch")
            decisions += 1
            for a, mu, target in zip(ordered, decision["mean"], decision["target"]):
                positive_means[a] += finite(mu, "decision mean") > 0
                close(finite(target, "decision target"), weights[a], "decision target", ratio=True)
        peak = max(peak, nav_end + liquidation)
        max_drawdown = max(max_drawdown, 1 - nav_end / peak)
        previous_end, previous_nav = end, nav_end
    liquidation = finite(rows[-1].get("liquidation_fee", ledger.get('liquidation_fee', 0)), "liquidation")
    total_fees = math.fsum(trade_fees) + liquidation
    gross_pnl = math.fsum(math.fsum(v) for v in contributions.values())
    close(total_fees, ledger["fees"], "total fees")
    close(previous_nav, ledger["terminal_wealth"], "terminal wealth")
    close(initial + gross_pnl - total_fees, previous_nav, "whole-path attribution")
    close(ledger["net_return"], previous_nav / initial - 1, "net return", ratio=True)
    if "max_drawdown" in ledger:
        close(ledger["max_drawdown"], max_drawdown, "drawdown", ratio=True)
    for part in years.values():
        close(math.fsum(part["asset_contributions"].values()) - part["fees"],
              part["end_nav"] - part["start_nav"], "annual attribution")
        part["net_return"] = part["end_nav"] / part["start_nav"] - 1
        part["average_risky_weight"] = part.pop("risky_weight_sum") / part["periods"]
    return {
        "periods": len(rows), "initial_nav": initial, "terminal_wealth": previous_nav,
        "gross_dollar_contribution": gross_pnl, "fees": total_fees,
        "asset_contributions": {a: math.fsum(v) for a, v in contributions.items()},
        "average_asset_weights": {a: math.fsum(v) / len(v) for a, v in exposures.items()},
        "annual": years, "decisions_with_recorded_inputs": decisions,
        "positive_log_mean_decisions": positive_means,
        "max_absolute_accounting_error": max_error,
        "max_ratio_error": max_ratio_error, "max_drawdown": max_drawdown,
        "scope": "Attribution of saved decisions only; no new policy, fit, or causal effect estimate.",
    }
