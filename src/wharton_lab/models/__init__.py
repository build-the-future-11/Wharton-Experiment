"""Models with optional backends loaded on demand.

NumPy accounting does not require optional neural models. Public class names
and their defining modules are unchanged, preserving serialized identities.
"""
from importlib import import_module

_EXPORTS = {
    "BaseModel": ".base", "M01RegimeDistributionalGBM": ".m01",
    "M02FactorResidualRanker": ".m02", "M03ConditionalNonlinearFactor": ".m03",
    "M04FinancialAPEN": ".m04", "QAPENModel": ".m05", "FIJEPAModel": ".m06",
    "EigenJEPAModel": ".m07", "GraphFIJEPAModel": ".m08",
    "M09Model": ".m09", "M09Config": ".m09", "M10Model": ".m10",
    "M10Config": ".m10", "M11Model": ".m11", "M11Config": ".m11",
    "M12Model": ".m12", "M12Config": ".m12",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
