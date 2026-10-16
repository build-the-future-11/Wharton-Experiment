"""Delayed fixed-share aggregation of two complete density strategies."""

import math

import numpy as np


class SoftGate:
    def __init__(self, share=1 / 52):
        if not math.isfinite(share) or not 0 < share <= 1:
            raise ValueError("share must be in (0,1]")
        self.share = float(share)
        self.probabilities = np.array([0.5, 0.5])
        self.pending = []
        self.seen = set()
        self.last_issue = None
        self.updates = []

    def weights(self, issue):
        if not math.isfinite(issue) or (self.last_issue is not None and issue <= self.last_issue):
            raise ValueError("issue times must strictly advance")
        ready = sorted((r for r in self.pending if r[1] < issue), key=lambda r: (r[1], r[0]))
        for origin, available, losses in ready:
            logits = np.log(self.probabilities) - losses
            p = np.exp(logits - logits.max())
            p /= p.sum()
            self.probabilities = (1 - self.share) * p + self.share / 2
            self.updates.append(
                {
                    "origin": origin,
                    "available": available,
                    "used_at": issue,
                    "losses": losses.tolist(),
                    "weights": self.probabilities.tolist(),
                }
            )
        self.pending = [r for r in self.pending if r[1] >= issue]
        self.last_issue = float(issue)
        return self.probabilities.copy()

    def observe(self, origin, available, losses):
        losses = np.asarray(losses, float)
        if (
            origin != self.last_issue
            or origin in self.seen
            or not math.isfinite(available)
            or available <= origin
            or losses.shape != (2,)
            or not np.isfinite(losses).all()
        ):
            raise ValueError("one finite complete feedback pair per issued date required")
        self.pending.append((float(origin), float(available), losses.copy()))
        self.seen.add(origin)

    def state_dict(self):
        return {
            "share": self.share,
            "probabilities": self.probabilities.tolist(),
            "last_issue": self.last_issue,
            "seen": sorted(self.seen),
            "updates": self.updates,
            "pending": [{"origin": t, "available": a, "losses": x.tolist()} for t, a, x in self.pending],
        }
