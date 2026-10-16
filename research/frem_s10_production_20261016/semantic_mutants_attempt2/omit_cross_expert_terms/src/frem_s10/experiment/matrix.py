"""Exact fixed development identities, without data access or execution."""

from dataclasses import dataclass

from frem_s10.errors import ContractError

FORECAST_ARMS = (
    "unconditional",
    "recent",
    "context",
    "equal",
    "global_stack",
    "contextual_stack",
    "individual_score_mix",
    "zero",
)
BENCHMARKS = ("cash", "equal100", "equal80", "buy_hold100", "inverse_vol80", "shrink_min_variance80")
ABLATIONS = (
    "repeat_h1",
    "frozen_state",
    "no_shrink",
    "force_liquidate",
    "no_exact_hold",
    "window260",
    "neighbors16",
    "neighbors64",
)
MODEL_VARIANTS = ("primary", "repeat_h1", "no_shrink", "window260", "neighbors16", "neighbors64")
COSTS_BPS = (0, 10, 20, 40)


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One immutable declared ledger, independently initialized within its fold."""

    fold_id: str
    kind: str
    arm: str
    cost_bps: int
    policy: str = "robust"

    def __post_init__(self) -> None:
        """Reject undeclared combinations rather than silently widening the matrix."""
        if self.fold_id not in ("A", "B") or type(self.cost_bps) is not int or self.cost_bps not in COSTS_BPS:
            raise ContractError("cell fold or cost is outside the frozen matrix")
        allowed = {"strategy": FORECAST_ARMS, "benchmark": BENCHMARKS, "ablation": ABLATIONS}
        if self.kind not in allowed or self.arm not in allowed[self.kind]:
            raise ContractError("undeclared cell kind or arm")
        if self.policy not in ("nominal", "robust"):
            raise ContractError("undeclared policy")
        if self.kind != "strategy" and self.policy != "robust":
            raise ContractError("only strategy cells have a policy cross product")
        if self.kind == "ablation" and self.cost_bps != 10:
            raise ContractError("ablations use only the frozen primary 10 bps cost")

    @property
    def cell_id(self) -> str:
        """Match the metadata planner's established human-readable identity."""
        stem = f"{self.fold_id}/{self.kind}/{self.arm}"
        return (
            f"{stem}/{self.policy}/{self.cost_bps}" if self.kind == "strategy" else f"{stem}/{self.cost_bps}"
        )

    @property
    def one_way_cost(self) -> float:
        """Convert explicit basis points to a dimensionless one-way fee rate."""
        return self.cost_bps / 10000.0

    @property
    def variant(self) -> str:
        """Select a separate causal model only for a forecast-changing ablation."""
        return self.arm if self.kind == "ablation" and self.arm in MODEL_VARIANTS else "primary"

    @property
    def forecast_arm(self) -> str:
        """Return the exact shared law used by this cell."""
        if self.kind == "strategy":
            return self.arm
        return "unconditional" if self.kind == "benchmark" else "contextual_stack"


def campaign_matrix() -> tuple[CellSpec, ...]:
    """Enumerate precisely 128 strategies, 48 benchmarks and 16 fixed ablations."""
    result: list[CellSpec] = []
    for fold in ("A", "B"):
        for cost in COSTS_BPS:
            result.extend(
                CellSpec(fold, "strategy", arm, cost, policy)
                for arm in FORECAST_ARMS
                for policy in ("nominal", "robust")
            )
            result.extend(CellSpec(fold, "benchmark", arm, cost) for arm in BENCHMARKS)
        result.extend(CellSpec(fold, "ablation", arm, 10) for arm in ABLATIONS)
    return tuple(result)
