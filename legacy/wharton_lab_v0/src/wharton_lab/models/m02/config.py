from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M02Config:
    factor_window: int = 60
    rank_lr: float = 0.05
    rank_epochs: int = 80
    rank_hidden: int = 16
    l2: float = 1e-4
    max_pairs_per_date: int = 256
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M02Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
