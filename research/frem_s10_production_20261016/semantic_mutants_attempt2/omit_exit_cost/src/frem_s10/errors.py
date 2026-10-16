"""Typed, non-silent failures shared by the isolated S10 package."""

from collections.abc import Mapping


class S10Error(Exception):
    """Base failure with bounded machine-readable diagnostic metadata."""

    code = "S10_ERROR"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        record_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.record_id = record_id
        self.details = dict(details or {})


class ContractError(S10Error):
    """A shape, type, unit, schema or finite-value contract was violated."""

    code = "CONFIG_CONTRACT"


class ChronologyError(S10Error):
    """Information or state violates the strict research clock."""

    code = "CHRONOLOGY"


class DataAdmissionError(S10Error):
    """The proposed source cannot support the declared research data contract."""

    code = "SOURCE_ADMISSION"


class NumericalError(S10Error):
    """A finite or certified numerical result could not be produced."""

    code = "NUMERICAL"


class IntegrityError(S10Error):
    """An identity or retained-evidence invariant failed."""

    code = "SOURCE_INTEGRITY"


class ResourceLimitError(S10Error):
    """A resource admission or execution ceiling failed."""

    code = "RESOURCE_BUDGET"


class ArtifactIOError(S10Error):
    """A validated artifact could not be read or durably published."""

    code = "STORAGE_INTEGRITY"


class ExecutionStateError(S10Error):
    """The requested action is invalid for the consumed execution state."""

    code = "STATE_CONFLICT"
