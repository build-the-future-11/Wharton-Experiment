"""M05 Q-APEN configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class M05Config:
    max_experts: int = 8
    top_k: int = 2
    memory_cap_bytes: int = 65536
    codebook_size: int = 256
    residual_dtype: str = "float16"
    create_loss_threshold: float = 0.15
    merge_similarity_threshold: float = 0.98
    prune_worst_fraction: float = 0.25
    min_matured_before_lifecycle: int = 4
    expert_types: tuple[str, ...] = ("linear", "gaussian")
    random_state: int = 42
    disagreement_eps: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> M05Config:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
