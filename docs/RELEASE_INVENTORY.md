# Release inventories and historical source identity

The old root `RELEASE_MANIFEST.json` is a retained historical snapshot. It differs from the current source and must not certify the current checkout. Existing wheels likewise identify earlier builds; their existence does not establish current-source correspondence.

The inventory builder now requires a new output directory, derives package versions from each `pyproject.toml`, and records `SOURCE_INVENTORY_ONLY`. It does not run tests, lint, builds, or scientific evaluation. Those checks require their own dated, source-bound receipts. It preserves prior manifests and hashes files in bounded memory, rejects source symlinks and changing files, and prunes raw data, run outputs, caches, temporary trees, and the separate legacy/successor boundaries before traversal. A manifest is a local identity record, not distribution approval.

```sh
rtk proxy .venv/bin/python scripts/build_release_manifests.py --output-dir research/NEW_RELEASE_SNAPSHOT
rtk proxy .venv/bin/python scripts/build_release_manifests.py --verify research/NEW_RELEASE_SNAPSHOT/FREM.json
rtk proxy .venv/bin/python scripts/build_release_manifests.py --root successors/rejepa_v1 --verify research/NEW_RELEASE_SNAPSHOT/RE-JEPA.json
rtk proxy .venv/bin/python scripts/build_release_manifests.py --root successors/reng_jepa_v01 --verify research/NEW_RELEASE_SNAPSHOT/RENG-JEPA.json
```

Verification checks every listed source/build identity and the referenced previous manifest, rejects missing/changed/duplicate entries, and exits nonzero on failure. It does not certify unlisted files, fresh builds, numerical behavior, dataset rights, or scientific validity. Do not generate a release snapshot while another writer is changing its source tree.

## Re-auditing the original synthetic run

The original four-cell run saved hashes of 36 source/config files. Later S2 changes to `model.py` and `training.py` correctly fail a check against live source. The exact original files were recovered from Git commit `09268ecb01ac645eeee18cc9fcbc6db247b6ec89`, checked against all 36 original hashes, and saved under `research/local_maintenance_20261011/original_engineering_source/`. The archive receipt records their origin; no old source was executed.

Use the explicit archive to recheck the original result:

```sh
rtk proxy .venv/bin/python scripts/audit_engineering_evidence.py --source-root research/local_maintenance_20261011/original_engineering_source --output research/NEW_HISTORICAL_EVIDENCE_AUDIT
```

Every original source hash, data/receipt identity, paired outcome, timing boundary and score check still applies. A missing or altered archived file fails. This reconstructs the retained negative engineering result without refitting, replacing data, or weakening the run's identity. The original bootstrap interval is retained, not independently recomputed.
