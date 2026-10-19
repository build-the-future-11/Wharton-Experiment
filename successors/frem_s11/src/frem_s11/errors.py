"""Typed FREM-S11 failures."""


class S11Error(RuntimeError):
    """Base error for S11."""


class DataContractError(S11Error):
    """Raised when release-vintage data violate the frozen contract."""


class NumericalError(S11Error):
    """Raised when finite numerical contracts fail."""
