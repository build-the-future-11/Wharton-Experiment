from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M03Config:
    n_factors: int = 3
    mlp_hidden: int = 32
    mlp_epochs: int = 120
    mlp_lr: float = 0.01
    ar1_fit_min: int = 30
    pca_on_train: bool = True
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M03Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
