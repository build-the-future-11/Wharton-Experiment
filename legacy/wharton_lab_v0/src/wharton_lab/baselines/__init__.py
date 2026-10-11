"""Simple baselines for benchmarking."""

from wharton_lab.baselines.bootstrap_scenarios import bootstrap_scenario_paths
from wharton_lab.baselines.ewma_cov import ewma_covariance
from wharton_lab.baselines.persistence import persistence_forecast
from wharton_lab.baselines.ridge import ridge_forecast
from wharton_lab.baselines.strategic_allocation import equal_weight_targets

__all__ = [
    "persistence_forecast",
    "ridge_forecast",
    "ewma_covariance",
    "bootstrap_scenario_paths",
    "equal_weight_targets",
]
