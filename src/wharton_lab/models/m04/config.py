from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M04Config:
    horizon_steps: int = 5
    maturity_delay: int = 2
    memory_capacity: int = 512
    similarity_threshold: float = 0.35
    novelty_threshold: float = 0.85
    min_confidence: float = 0.2
    ridge_alpha: float = 1.0
    mlp_hidden: int = 16
    use_mlp_backbone: bool = False
    retrieval_k: int = 8
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M04Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
