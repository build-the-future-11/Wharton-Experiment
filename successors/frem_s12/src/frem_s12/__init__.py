"""FREM-S12 causal multiscale memory-router research package."""

from frem_s12.experiment import Fold, FoldResult, run_fold
from frem_s12.model import ModelConfig

__all__ = ["Fold", "FoldResult", "ModelConfig", "run_fold"]
__version__ = "0.1.4"
