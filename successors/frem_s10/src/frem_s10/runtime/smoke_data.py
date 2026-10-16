"""Small deterministic generated inputs for the maintained S10 kernel exercise."""

from dataclasses import dataclass

import numpy as np

from frem_s10.domain import OriginRecord, Transform
from frem_s10.model.records import IssueRequest, scientific_id

WEEK = 604800_000_000
BASE = 946684800_000_000
START = 70


@dataclass(frozen=True)
class GeneratedInputs:
    """Regenerable data and a burn-in-only transform, never market observations."""

    records: tuple[OriginRecord, ...]
    transform: Transform
    identity: str


def generated_inputs(seed: int, steps: int) -> GeneratedInputs:
    """Generate a fixed seeded panel once, then reuse it across every variant.

    All randomness is consumed here using a private PCG64 generator. The online
    kernel is deterministic and has no optimizer, sampler, or global RNG state.
    Regeneration and exact data identity replace a mutable sampler checkpoint.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    count = START + steps + 5
    contexts = rng.normal(0, 0.02, size=(count, 12))
    g = 1 + rng.uniform(-0.04, 0.04, size=(count, 3))
    d = rng.uniform(0, 0.001, size=(count, 3))
    identity = scientific_id("s10-generated-smoke-data-v1", (seed, steps), contexts, g, d)
    records = tuple(
        OriginRecord(
            f"{identity}:{i}",
            i,
            BASE + i * WEEK,
            BASE + (i + 4) * WEEK,
            BASE + (i + 4) * WEEK + WEEK // 2,
            BASE + i * WEEK - 1,
            contexts[i],
            g[i],
            d[i],
            1 + (g[i] - 1) / 4,
            d[i] / 4,
        )
        for i in range(count)
    )
    burnin = contexts[:64]
    scale = np.maximum(burnin.std(axis=0), 1e-12)
    score_scale = np.maximum((g[:64] + d[:64] - 1).std(axis=0), 1e-12)
    transform_id = scientific_id(
        "s10-generated-smoke-transform-v1",
        tuple(r.record_id for r in records[:64]),
        burnin.mean(axis=0),
        scale,
        score_scale,
    )
    transform = Transform(
        burnin.mean(axis=0),
        scale,
        score_scale,
        tuple(r.record_id for r in records[:64]),
        transform_id,
    )
    return GeneratedInputs(records, transform, identity)


def issue_request(record: OriginRecord) -> IssueRequest:
    """Strip all target values before the actual forecasting transition."""
    return IssueRequest(
        record.record_id,
        record.origin_us,
        record.endpoint_us,
        record.available_us,
        record.feature_available_us,
        record.x,
        record.anchor_index,
    )
