"""Resumable experiment runner with atomic checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


@dataclass
class TaskSpec:
    task_id: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def deterministic_id(name: str, params: Mapping[str, Any]) -> str:
        payload = json.dumps({"name": name, "params": dict(params)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class TaskResult:
    task_id: str
    status: str
    output: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExperimentRunner:
    """Append-only progress log and atomic checkpoint files."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.run_dir / "progress.jsonl"
        self.checkpoint_dir = self.run_dir / "checkpoints"

    def _checkpoint_path(self, task_id: str) -> Path:
        return self.checkpoint_dir / f"{task_id}.json"

    def is_complete(self, task_id: str) -> bool:
        p = self._checkpoint_path(task_id)
        return p.exists()

    def load_result(self, task_id: str) -> Optional[TaskResult]:
        p = self._checkpoint_path(task_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return TaskResult(**data)

    def _append_progress(self, record: Mapping[str, Any]) -> None:
        with self.progress_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def run_task(
        self,
        spec: TaskSpec,
        fn: Callable[[], Mapping[str, Any]],
        *,
        force: bool = False,
    ) -> TaskResult:
        if not force and self.is_complete(spec.task_id):
            existing = self.load_result(spec.task_id)
            if existing is not None:
                return existing

        self._append_progress({"event": "start", "task_id": spec.task_id, "name": spec.name})
        try:
            output = fn()
            result = TaskResult(task_id=spec.task_id, status="ok", output=dict(output))
            self._atomic_write(
                self._checkpoint_path(spec.task_id),
                json.dumps(asdict(result), sort_keys=True),
            )
            self._append_progress({"event": "done", "task_id": spec.task_id, "status": "ok"})
            return result
        except Exception as exc:
            result = TaskResult(
                task_id=spec.task_id,
                status="error",
                error=str(exc),
            )
            self._atomic_write(
                self._checkpoint_path(spec.task_id),
                json.dumps(asdict(result), sort_keys=True),
            )
            self._append_progress(
                {"event": "done", "task_id": spec.task_id, "status": "error", "error": str(exc)}
            )
            return result
