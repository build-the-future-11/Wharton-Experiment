"""Governance checks authored prospectively and now verified at runtime."""

import json

import pytest

from frem.artifacts import object_digest, run_directory
from frem.governance import append_event


def test_hash_is_key_order_invariant():
    assert object_digest({"a": 1, "b": 2}) == object_digest({"b": 2, "a": 1})


def test_failure_is_retained(tmp_path):
    with (
        pytest.raises(ValueError, match="declared failure"),
        run_directory(tmp_path, {}, "development", {}) as directory,
    ):
        raise ValueError("declared failure")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert "declared failure" in (directory / "failure.txt").read_text()


def test_journal_detects_earlier_edit(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"hypothesis": "v1"})
    path.write_text(path.read_text().replace('"v1"', '"v2"'))
    with pytest.raises(ValueError, match="hash chain"):
        append_event(path, {"next": True})


def test_journal_chains_events(tmp_path):
    path = tmp_path / "events.jsonl"
    first = append_event(path, {"decision": "A"})
    second = append_event(path, {"decision": "B"})
    assert second["previous"] == first["sha256"]
