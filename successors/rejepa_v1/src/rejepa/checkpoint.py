"""Hash-bound, atomic RE-JEPA checkpoint serialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from .engram import SpectralEngramMemory
from .model import REJEPA, REJEPAConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor content canonically, independent of torch archive metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def save_checkpoint(
    directory: str | Path,
    model: REJEPA,
    *,
    metadata: dict[str, Any],
    memory: SpectralEngramMemory | None = None,
) -> dict[str, str]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=target, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, target / "model.pt")
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(target / "config.json", asdict(model.config))
    _atomic_json(target / "metadata.json", metadata)
    if memory is not None:
        _atomic_json(target / "engram.json", memory.export())
    hashes = {
        path.name: _sha256(path)
        for path in sorted(target.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    _atomic_json(
        target / "manifest.json",
        {"sha256": hashes, "model_state_sha256": model_state_sha256(model.state_dict())},
    )
    return hashes


def load_checkpoint(
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[REJEPA, dict[str, Any], SpectralEngramMemory | None]:
    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Checkpoint manifest is missing hashes")
    for name, expected in hashes.items():
        path = source / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Checkpoint integrity failed for {name}")
    config_payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    model = REJEPA(REJEPAConfig(**config_payload))
    state = torch.load(source / "model.pt", map_location=map_location, weights_only=True)
    expected_state_hash = manifest.get("model_state_sha256")
    if expected_state_hash is not None and model_state_sha256(state) != expected_state_hash:
        raise ValueError("Checkpoint model state integrity failed")
    model.load_state_dict(state, strict=True)
    model.to(map_location)
    model.eval()
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    memory_path = source / "engram.json"
    memory = None
    if memory_path.exists():
        memory = SpectralEngramMemory.from_export(json.loads(memory_path.read_text(encoding="utf-8")))
    return model, metadata, memory
