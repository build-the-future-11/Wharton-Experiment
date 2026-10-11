"""Ground-up Reservoir-Engram JEPA research core."""

__version__ = "0.3.0"

from .baselines import CausalContinuationRetriever, CausalStreamingResidualMemory, ResidualAlignment
from .case import LauraMandate
from .checkpoint import load_checkpoint, save_checkpoint
from .engram import SpectralEngramMemory
from .mandate import (
    ClientMandate,
    FacilityContributionPolicy,
    FundingFrontier,
    LiabilitySchedule,
    LiquidityReserve,
    dynamic_sleeve_weights,
    evaluate_funding_frontier,
)
from .model import REJEPA, REJEPAConfig, REJEPAOutput, rejepa_loss
from .training import TrainingBatch, TrainingConfig, TrainingResult, fit_model, predict_frozen

__all__ = [
    "REJEPA",
    "CausalContinuationRetriever",
    "CausalStreamingResidualMemory",
    "ClientMandate",
    "FacilityContributionPolicy",
    "FundingFrontier",
    "LauraMandate",
    "LiabilitySchedule",
    "LiquidityReserve",
    "REJEPAConfig",
    "REJEPAOutput",
    "ResidualAlignment",
    "SpectralEngramMemory",
    "TrainingBatch",
    "TrainingConfig",
    "TrainingResult",
    "__version__",
    "dynamic_sleeve_weights",
    "evaluate_funding_frontier",
    "fit_model",
    "load_checkpoint",
    "predict_frozen",
    "rejepa_loss",
    "save_checkpoint",
]
