Private Wharton ledger correction: require boolean terminal flags

The existing R2 core accepted terminal="false" and liquidated100 currency units of holdings at an explicit10% fee. This is a malformed-input defect, not a proposed liquidation policy. State.closed also accepted non-booleans.

The isolated candidate adds two fail-closed bool checks and annotates the existing terminal parameter. Valid True/False behavior and both mark/liquidate policies remain unchanged. Invalid strings, numbers, null and containers raise ValueError.

Baseline: four new test methods,16 failing invalid-value subtests. Fix: all25 accounting tests pass (21 existing,4 new). No model imports, training, market fixtures, tuning or benchmarks.

From candidate/: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -p "test_accounting*.py" -v

The candidate contains the actual existing ledger core, package initializer and accounting tests. terminal-flag-validation.patch targets the full R2 package for a separately owned integration. Both frozen R2 copies and the negative synthetic result remain unchanged.

No point-in-time data claim is added. Authentic market vintages/availability, source archive access, economic adoption and canonical integration remain separate gates.
