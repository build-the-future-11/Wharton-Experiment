"""Typed FREM-S12 failures."""


class S12Error(RuntimeError):
    """Base S12 failure."""


class ContractError(S12Error):
    """Input, identity, chronology, or evidence contract failure."""


class NumericalError(S12Error):
    """Nonfinite or mathematically invalid numerical state."""


class ConsumptionError(S12Error):
    """Attempted reuse or conflict in a one-use execution namespace."""
