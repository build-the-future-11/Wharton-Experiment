"""Instantiate lab models — lean defaults for fast orchestration."""

from __future__ import annotations

from typing import Any

from wharton_lab.models.m01 import M01Config, M01RegimeDistributionalGBM
from wharton_lab.models.m02 import M02Config, M02FactorResidualRanker
from wharton_lab.models.m03 import M03Config, M03ConditionalNonlinearFactor
from wharton_lab.models.m04 import M04Config, M04FinancialAPEN
from wharton_lab.models.m05 import QAPENModel
from wharton_lab.models.m05.config import M05Config
from wharton_lab.models.m06 import FIJEPAModel
from wharton_lab.models.m06.config import M06Config
from wharton_lab.models.m07 import EigenJEPAModel
from wharton_lab.models.m07.config import M07Config
from wharton_lab.models.m08 import GraphFIJEPAModel
from wharton_lab.models.m08.config import M08Config
from wharton_lab.models.m09 import M09Config, M09Model
from wharton_lab.models.m10 import M10Config, M10Model
from wharton_lab.models.m11 import M11Config, M11Model
from wharton_lab.models.m12 import M12Config, M12Model


def build_model(model_id: str, variant: str, *, smoke: bool = False) -> Any:
    """Return a model instance or None for baseline ids handled elsewhere."""
    mid = model_id.upper()
    var = variant.lower()

    if mid == "M01":
        soft = var in ("soft_regime", "ablation_soft_regime")
        cfg = M01Config(
            n_estimators=6 if smoke else 12,
            max_depth=2,
            min_samples_leaf=8,
            learning_rate=0.15,
            use_regimes=var not in ("no_regime", "ablation_no_regime"),
            soft_vs_hard=soft,
            distributional_vs_point=var not in ("point_only", "ablation_point_only"),
            regime_as_feature=not soft,
        )
        return M01RegimeDistributionalGBM(cfg)
    if mid == "M02":
        cfg = M02Config(
            rank_epochs=4 if smoke else 16,
            max_pairs_per_date=32 if smoke else 64,
            rank_hidden=8 if smoke else 12,
        )
        return M02FactorResidualRanker(cfg)
    if mid == "M03":
        cfg = M03Config(mlp_epochs=5 if smoke else 40)
        return M03ConditionalNonlinearFactor(cfg)
    if mid == "M04":
        cfg = M04Config(use_mlp_backbone=var in ("mlp_backbone", "ablation_mlp_backbone"))
        return M04FinancialAPEN(cfg)
    if mid == "M05":
        cfg = M05Config(max_experts=3 if smoke else 5)
        if var in ("linear_only", "ablation_linear_only"):
            cfg.expert_types = ("linear",)
        return QAPENModel(cfg)
    if mid == "M06":
        cfg = M06Config(
            epochs=1,
            device="cpu",
            hidden_dim=16 if smoke else 24,
            latent_dim=8 if smoke else 12,
            seq_len=4 if smoke else 6,
            batch_size=32,
        )
        return FIJEPAModel(cfg)
    if mid == "M07":
        cfg = M07Config(epochs=1 if smoke else 2, device="cpu")
        return EigenJEPAModel(cfg)
    if mid == "M08":
        cfg = M08Config(
            epochs=1 if smoke else 2,
            device="cpu",
            n_nodes=4 if smoke else 5,
            seq_len=5 if smoke else 8,
            node_dim=2 if smoke else 3,
        )
        return GraphFIJEPAModel(cfg)
    if mid == "M09":
        return M09Model(M09Config(groupdro_steps=1 if smoke else 3))
    if mid == "M10":
        return M10Model(M10Config(path_steps=2 if smoke else 3))
    if mid == "M11":
        return M11Model(M11Config(horizon=4 if smoke else 5))
    if mid == "M12":
        cfg = M12Config(n_assets=3 if smoke else 5)
        if var in ("no_graph", "ablation_no_graph"):
            cfg.no_graph = True
        if var in ("no_memory", "ablation_no_memory"):
            cfg.no_memory = True
        if var in ("no_latent", "ablation_no_latent"):
            cfg.no_latent = True
        if var in ("no_liability", "ablation_no_liability"):
            cfg.no_liability = True
        return M12Model(cfg)
    if mid.startswith("B_"):
        return None
    raise ValueError(f"Unknown model_id {model_id}")
