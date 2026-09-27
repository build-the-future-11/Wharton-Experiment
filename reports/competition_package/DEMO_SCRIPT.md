# Local decision-tool demonstration

**A demonstration of real local calculation and saved evidence, not simulated customer traction or executed WInS trades.** No login, network service or trading permission is needed.

1. Run `PYTHONPATH=src .venv/bin/python scripts/run_competition_stress.py`, then `.venv/bin/python scripts/build_competition_artifacts.py`. Open `reports/competition_package/DECISION_ROOM.html` locally.
2. Read the banner: all inputs are assumptions, initial wealth is normalized, and no portfolio is approved for the client. Show the visible source hashes and config path.
3. Select `steady_assumption`, `liability_reserve`, 10 bps. Read end wealth, funded share, unpaid balance and transaction costs. These values come from the retained JSON trajectory.
4. Switch to `early_joint_crash`, then `late_joint_crash` without changing policy or cost. Explain the payment timing effect; do not call either path a forecast.
5. Select `funding_failure`. Show the first missed payment and unpaid balance. Switch policies and retain the adverse finding. Cash changes exposure; it does not eliminate an infeasible obligation schedule.
6. Change costs from 0 to 50 bps. Explain that the values are cost sensitivities, not verified execution costs. Cash interest and taxes are omitted assumptions.
7. Open `runs/competition_stress/SUMMARY.csv` or the JSON and cross-check the displayed combination. The complete monthly ledger records starting wealth, trades, fees, payment and ending wealth.
8. Open `EVIDENCE_APPENDIX.md`. Explain why legacy model results and the old synthetic win cannot be used as support. Show the client-source and trading-record blockers.

Expected outcome: the user can trace a displayed number to a deterministic calculation and identify its limits. No result establishes client suitability, empirical superiority, or submission readiness.

Fallback: If interactive scripting is unavailable, use `FINANCIAL_STRESS.md` and the JSON trajectories. Do not replace missing outputs with static invented numbers.
