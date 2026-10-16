"""Descriptive calibration diagnostics; these never choose model settings."""

from __future__ import annotations

import math
from collections import Counter


def calibration_diagnostics(candidates, counts, records, first_decision):
    """Explain rejection of memory using validation only and saved memory metadata."""
    if not candidates or not counts:
        raise ValueError("Diagnostics require calibration candidates and retrieval counts")
    if any(not math.isfinite(c["validation_nll"]) for c in candidates):
        raise ValueError("Nonfinite calibration score")
    zero = [c["validation_nll"] for c in candidates if c["weight"] == 0]
    if not zero:
        raise ValueError("Missing zero-weight comparator")
    baseline = min(zero)
    penalties = {
        str(weight): min(c["validation_nll"] for c in candidates if c["weight"] == weight) - baseline
        for weight in sorted({c["weight"] for c in candidates})
    }
    ages = [(first_decision - r["label_time"]) / 86_400_000_000_000 for r in records]
    if ages and min(ages) <= 0:
        raise ValueError("Memory outcomes must mature before validation")
    return {
        "validation_nll_penalty_by_weight": penalties,
        "retrieval_count_histogram": dict(Counter(int(n) for n in counts)),
        "records_per_asset": dict(Counter(r["asset_id"] for r in records)),
        "record_age_days_at_validation_start": [min(ages), max(ages)] if ages else None,
        "interpretation": "Descriptive only; unchanged declared calibration grid and selection rule",
    }
