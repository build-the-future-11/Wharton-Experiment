from __future__ import annotations

import json
from pathlib import Path

import pytest

from frem_s12.errors import ContractError
from frem_s12.saved_versions import audit_saved_versions


def _ledger(path: Path, *, complete: bool, duplicate: bool = False) -> None:
    claims = [{"phase": "search", "identity": "trial=0/family=size_bm/seed=11"}]
    attempts = [
        {"saved_version_id": 101, "status": "REJECTED", "claims": []},
        {"saved_version_id": 102, "status": "SCIENTIFIC", "claims": claims},
    ]
    if duplicate:
        attempts.append({"saved_version_id": 103, "status": "SCIENTIFIC", "claims": claims})
    path.write_text(
        json.dumps(
            {
                "schema": "frem-s12-saved-version-ledger-v1",
                "scope": "kaggle",
                "complete": complete,
                "attempts": attempts,
            }
        )
    )


def test_saved_versions_require_external_completion(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    _ledger(path, complete=False)
    with pytest.raises(ContractError, match="not externally complete"):
        audit_saved_versions(path)


def test_saved_versions_reject_duplicate_scientific_claims(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    _ledger(path, complete=True, duplicate=True)
    with pytest.raises(ContractError, match="duplicate scientific claim"):
        audit_saved_versions(path)


def test_saved_versions_accept_complete_fork_free_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    _ledger(path, complete=True)
    assert audit_saved_versions(path)["scientific_claims"] == 1
