"""Deterministic hashing for configs and receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(obj: Any, *, n: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:n]
