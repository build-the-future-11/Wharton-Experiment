"""Mixture likelihood routing. No outcome may enter before availability."""
import numpy as np


def mixture_weights(log_density, date_weights=None, prior_strength=12.0):
    """EM with uniform Dirichlet pseudocounts; rows are [date, asset, expert]."""
    x = np.asarray(log_density, dtype=float)
    if x.ndim != 3 or not np.isfinite(x).all() or not len(x):
        raise ValueError('finite nonempty date/asset/expert densities required')
    n, _, e = x.shape
    q = np.ones(n) if date_weights is None else np.asarray(date_weights, dtype=float)
    if q.shape != (n,) or not np.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        raise ValueError('invalid date weights')
    q = q / q.sum() * n
    density = np.exp(x - x.max(axis=2, keepdims=True))
    w = np.ones(e) / e
    for _ in range(60):
        responsibility = density * w
        responsibility /= responsibility.sum(axis=2, keepdims=True)
        w = (np.sum(q[:, None] * responsibility.mean(axis=1), axis=0)
             + prior_strength / e) / (n + prior_strength)
    return w


class MixtureMemory:
    def __init__(self, experts=6):
        self.experts = experts
        self.records = []

    def add(self, key, log_density, available):
        key, density = np.asarray(key, float), np.asarray(log_density, float)
        if key.ndim != 1 or density.ndim != 2 or density.shape[1] != self.experts:
            raise ValueError('invalid memory shape')
        if not np.isfinite(key).all() or not np.isfinite(density).all() or not np.isfinite(available):
            raise ValueError('nonfinite memory')
        self.records.append((float(available), key.copy(), density.copy()))

    def weights(self, key, issue, contextual=True):
        rows = sorted((r for r in self.records if r[0] < issue), key=lambda r:r[0])[-60:]
        if not rows:
            return np.ones(self.experts) / self.experts
        keys = np.stack([r[1] for r in rows])
        distance = np.mean((keys - np.asarray(key))**2, axis=1)
        q = np.exp(-.5 * distance) if contextual else np.ones(len(rows))
        return mixture_weights(np.stack([r[2] for r in rows]), q)


def select_strength(gains):
    """Require >0.01 nats in each chronological validation half; ties favor zero."""
    best, best_score = 0.0, 0.01
    for alpha in (.25, .5, 1.0):
        values = np.asarray(gains[alpha], float)
        if len(values) < 8 or not np.isfinite(values).all():
            raise ValueError('insufficient finite chronological calibration gains')
        score = min(float(x.mean()) for x in np.array_split(values, 2))
        if score > best_score:
            best, best_score = alpha, score
    return best
