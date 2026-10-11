"""M01 configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence, Tuple


DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


@dataclass
class M01Config:
    quantiles: Sequence[float] = field(default_factory=lambda: DEFAULT_QUANTILES)
    use_regimes: bool = True
    soft_vs_hard: bool = False  # False = hard regime assignment
    distributional_vs_point: bool = True
    calibrate: bool = False
    vol_window: int = 20
    vol_threshold: float = 0.02
    return_sign_feature_col: int = 0
    n_estimators: int = 50
    max_depth: int = 3
    learning_rate: float = 0.1
    min_samples_leaf: int = 5
    random_state: int = 42
    use_hist_gb: bool = True
    # Fast path: append causal regime id as a feature (1 GBM/quantile)
    # instead of fitting separate trees per regime (4×).
    regime_as_feature: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M01Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
