"""Allowlisted, hash-bound local JSON input without implicit source discovery."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..errors import ArtifactIOError, ContractError, IntegrityError
from ..metadata import ASSETS, strict_int


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate JSON key", field=key)
        result[key] = value
    return result


def _nonfinite_constant(value: str) -> object:
    raise ContractError("nonfinite JSON numeric constant", field=value)


def strict_json_loads(payload: str | bytes) -> dict[str, object]:
    """Decode an object, rejecting duplicate keys and nonstandard numeric values."""
    try:
        decoded: object = json.loads(
            payload, object_pairs_hook=_unique_object, parse_constant=_nonfinite_constant
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError("malformed UTF-8 JSON object") from exc
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise ContractError("JSON object required")
    return decoded


def require_fields(value: dict[str, object], fields: set[str], *, name: str) -> None:
    """Require exactly the versioned field set; never ignore unknown keys."""
    if set(value) != fields:
        raise ContractError(
            f"{name}: fields differ from supported schema",
            field=name,
            details={"missing": sorted(fields - set(value)), "unknown": sorted(set(value) - fields)},
        )


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Declared source locator and exact raw-byte identity, not a data permission."""

    logical_path: str
    sha256: str
    symbol: str
    retrieved_at_us: int

    def __post_init__(self) -> None:
        """Validate an exact allowlist locator without opening its contents."""
        if not isinstance(self.logical_path, str):
            raise ContractError("source path must be a normalized relative string path")
        path = PurePosixPath(self.logical_path)
        if (
            not self.logical_path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.logical_path
            or str(path) != self.logical_path
        ):
            raise ContractError("source path must be a normalized relative path")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ContractError("lowercase SHA-256 source identity required")
        if self.symbol not in ASSETS:
            raise ContractError("unsupported source symbol")
        strict_int(self.retrieved_at_us, name="retrieved_at_us")


class ScopedSourceReader:
    """Read only explicitly declared files beneath a fixed root after hash checks."""

    def __init__(
        self, root: Path, identities: tuple[SourceIdentity, ...], *, max_bytes: int = 16 * 1024 * 1024
    ) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ContractError("source root must be an existing directory")
        self.max_bytes = strict_int(max_bytes, name="max_source_bytes", minimum=1)
        self.identities = {identity.logical_path: identity for identity in identities}
        if len(self.identities) != len(identities):
            raise ContractError("duplicate source locator in allowlist")

    def _path(self, logical_path: str) -> Path:
        if logical_path not in self.identities:
            raise IntegrityError("source locator is not allowlisted", field="logical_path")
        candidate = self.root / logical_path
        current = self.root
        for component in PurePosixPath(logical_path).parts:
            current = current / component
            if current.is_symlink():
                raise IntegrityError("source symlinks are unsupported", field="logical_path")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ArtifactIOError("declared source does not exist") from exc
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise IntegrityError("source path escapes declared root or is not a regular file")
        return resolved

    def read_json(self, logical_path: str) -> dict[str, object]:
        """Verify bytes before decoding; returned source still requires scope filtering."""
        path = self._path(logical_path)
        try:
            with path.open("rb") as stream:
                payload = stream.read(self.max_bytes + 1)
        except OSError as exc:
            raise ArtifactIOError("cannot read declared source") from exc
        if len(payload) > self.max_bytes:
            raise IntegrityError("source exceeds declared byte ceiling")
        if hashlib.sha256(payload).hexdigest() != self.identities[logical_path].sha256:
            raise IntegrityError("retained source SHA-256 mismatch", field="sha256")
        return strict_json_loads(payload)
