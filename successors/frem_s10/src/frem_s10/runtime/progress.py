"""Durable observed progress without optional accounts or numerical imports."""

import math
import resource
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from frem_s10.errors import ContractError, S10Error
from frem_s10.identities import JSONObject, canonical_bytes
from frem_s10.runtime.journal import Journal

if TYPE_CHECKING:
    from frem_s10.experiment.model_archive import ModelArchiveWriter
    from frem_s10.model.records import ForecastTransition


def _console(line: str) -> None:
    print(line, flush=True)


class ProgressLog:
    """Own one fresh hash-linked JSONL file; console failure cannot lose its bytes.

    Disk failures remain explicit: reporting success without durable local logs
    would be misleading. Optional console failure is isolated after publication.
    No exception message, environment, credential or command line is captured.
    """

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        console: Callable[[str], None] = _console,
    ) -> None:
        """Create a new local log, rejecting existing files and symlink parents."""
        if (
            not isinstance(run_id, str)
            or not run_id
            or len(run_id) > 160
            or any(character.isspace() for character in run_id)
        ):
            raise ContractError("bounded run identity without whitespace required")
        self.clock = clock
        self.started = clock()
        if not math.isfinite(self.started):
            raise ContractError("finite progress clock required")
        self.last = self.started
        self.run_id = run_id
        self.console = console
        self.journal = Journal(path)

    def emit(self, event: JSONObject) -> str:
        """Persist actual event fields before writing timestamped flushed console output."""
        reserved = {"schema", "timestamp", "run_id", "elapsed_seconds", "process_peak_rss_mib"}
        if set(event).intersection(reserved):
            raise ContractError("progress event cannot replace observed envelope fields")
        now = self.clock()
        if not math.isfinite(now) or now < self.last:
            raise ContractError("progress clock regressed or became unavailable")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        observed: JSONObject = {
            "schema": "s10-progress-v1",
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "elapsed_seconds": now - self.started,
            "process_peak_rss_mib": peak / (2**20 if sys.platform == "darwin" else 1024),
            **event,
        }
        digest = self.journal.append(observed)
        self.last = now
        # A detached notebook/closed stdout is not a scientific failure.
        with suppress(OSError, ValueError):
            self.console(canonical_bytes(observed).decode("utf-8"))
        return digest

    def failure(self, phase: str, error: BaseException) -> None:
        """Retain an error type only, without masking the original operation failure."""
        # The operation still raises its original error; existing bytes stay.
        with suppress(S10Error, OSError):
            self.emit({"event": "FAILED", "phase": phase, "error_type": type(error).__name__})


def transition_sink(
    writer: "ModelArchiveWriter", log: ProgressLog
) -> Callable[[str, "ForecastTransition"], None]:
    """Log only transitions whose exact immutable checkpoint already exists."""

    def record(variant: str, step: "ForecastTransition") -> None:
        writer.sink(variant, step)
        request = step.batch.get("contextual_stack").request
        mixture = step.batch.mixtures
        global_fit, local_fit = mixture.global_certificate, mixture.local_certificate
        log.emit(
            {
                "event": "ORIGIN_ARCHIVED",
                "phase": "forecast",
                "variant": variant,
                "global_step": len(writer.rows),
                "anchor_index": request.anchor_index,
                "state_id": step.state.record_id,
                "checkpoint_path": str(writer.directory / f"weekly-{len(writer.rows) - 1:04}.json"),
                "matured_feedback_count": len(step.matured_feedback),
                "pending_count": len(step.state.pending),
                "library_count": len(step.state.library.record_ids),
                "global_regularized_energy_objective": None if global_fit is None else global_fit.objective,
                "local_regularized_energy_objective": None if local_fit is None else local_fit.objective,
                "local_fallback_reason": mixture.fallback_reason,
            }
        )

    return record
