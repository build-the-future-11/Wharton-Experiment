"""Inert account fixtures; no platform calls, numerical imports or compute jobs."""

import json
import math
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path

from account_budget import (
    AccountLedger, Blocked, DEADLINE, Observation, PhaseCost, Plan, StopRequired, caps,
)


class FakeClock:
    def __init__(self):
        self.now = DEADLINE - 12 * 3600

    def __call__(self):
        return self.now


def plan(quota=1.0):
    phase = PhaseCost(90.0, 0.025, quota / 4)
    return Plan(phase, phase, phase, phase, 1, 1.0, "simulated-conversion")


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.path = Path(self.directory.name) / "account.jsonl"
        self.clock = FakeClock()

    def tearDown(self):
        self.directory.cleanup()

    def ledger(self, **changes):
        options = dict(account_id="fixture-account", quota_period_id="fixture-period", initial_quota_hours=30.0,
                       allocation_enabled=True, account_ready=True, allocation_approval_id="simulated-approval")
        options.update(changes)
        return AccountLedger.create(self.path, clock=self.clock, **options)

    def observation(self, ledger, **changes):
        observation = Observation("fixture-account", "fixture-period", 30.0, self.clock.now, (), self.clock.now,
                                  "simulated-quota", "simulated-sessions", "simulated-coordination",
                                  self.clock.now, ledger.summary()["ledger_id"])
        return replace(observation, **changes)

    def finish(self, ledger, name, cost=1.0):
        ledger.reserve(name, plan(cost), self.observation(ledger))
        permit = ledger.authorize_launch(name, self.observation(ledger))
        ledger.record_outcome(name, outcome="failed", observed_wall_seconds=30, observed_device_hours=0.01,
                              evidence_id="simulated-failure")
        self.clock.now = permit["stop_by"]
        ledger.confirm_platform_stopped(name, self.observation(ledger))

    def test_caps_and_unknown_or_zero_quota(self):
        self.assertEqual(caps(None), {"preferred": 0, "expanded": 0})
        self.assertEqual(caps(30), {"preferred": 6, "expanded": 10})
        self.assertEqual(caps(25), {"preferred": 1, "expanded": 5})
        for index, q0 in enumerate((None, 0, 20, 24)):
            self.path = self.path.parent / f"low-{index}.jsonl"
            ledger = self.ledger(initial_quota_hours=q0)
            with self.assertRaises(Blocked):
                ledger.reserve("r", plan(), self.observation(ledger, remaining_quota_hours=q0 or 0))

    def test_disabled_default_and_phone_or_account_not_ready(self):
        ledger = AccountLedger.create(self.path, account_id="fixture-account", quota_period_id="fixture-period",
                                      initial_quota_hours=30, clock=self.clock)
        self.assertFalse(ledger.summary()["allocation_enabled"])
        with self.assertRaisesRegex(Blocked, "disabled"):
            ledger.reserve("r", plan(), self.observation(ledger))
        self.path = self.path.parent / "not-ready.jsonl"
        ledger = self.ledger(account_ready=False)
        with self.assertRaisesRegex(Blocked, "readiness"):
            ledger.reserve("r", plan(), self.observation(ledger))

    def test_expanded_requires_approval_and_ten_percent_reserve(self):
        with self.assertRaisesRegex(Blocked, "expanded approval"):
            self.ledger(expanded=True)
        ledger = self.ledger(expanded=True, expanded_approval_id="simulated-explicit-expanded-approval")
        self.assertEqual(ledger.summary()["spendable_hours"], 9.0)
        with self.assertRaisesRegex(Blocked, "budget"):
            ledger.reserve("r", plan(9.1), self.observation(ledger))
        ledger.reserve("r", plan(9.0), self.observation(ledger))

    def test_preferred_budget_is_five_point_four_for_simulated_thirty(self):
        ledger = self.ledger()
        self.assertEqual(ledger.summary()["spendable_hours"], 5.4)
        with self.assertRaisesRegex(Blocked, "budget"):
            ledger.reserve("r", plan(5.5), self.observation(ledger))

    def test_stale_future_missing_or_wrong_scope_observations_block(self):
        ledger = self.ledger()
        invalid = (
            {"quota_observed_at": self.clock.now - 61},
            {"sessions_observed_at": self.clock.now - 61},
            {"coordinated_at": self.clock.now + 1},
            {"coordinated_at": self.clock.now - 1},
            {"quota_evidence_id": ""}, {"sessions_evidence_id": ""}, {"coordination_evidence_id": ""},
            {"coordination_ledger_id": "different-ledger"}, {"account_id": "other-account"},
            {"quota_period_id": "reset-period"}, {"active_session_ids": ("uncoordinated-session",)},
            {"remaining_quota_hours": 31},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(Blocked):
                ledger.reserve("r", plan(), self.observation(ledger, **changes))
        self.assertEqual(ledger.summary()["runs"], 0)

    def test_nonfinite_negative_and_boolean_counters_fail(self):
        for value in (math.nan, math.inf, -1, True, "30", 10**1000):
            with self.subTest(value=repr(value)[:20]), self.assertRaises(Blocked):
                caps(value)
        ledger = self.ledger()
        for value in (math.nan, math.inf, -1, True):
            with self.subTest(value=value), self.assertRaises(Blocked):
                ledger.reserve("r", plan(), self.observation(ledger, remaining_quota_hours=value))
        with self.assertRaises(Blocked):
            plan(math.inf).totals()

    def test_explicit_phase_conversion_bounds(self):
        valid = plan()
        self.assertEqual(valid.totals()["wall_seconds_upper"], 360)
        self.assertEqual(valid.totals()["device_hours_upper"], 0.1)
        for bad in (replace(valid, setup=None), replace(valid, device_count=2),
                    replace(valid, shutdown=PhaseCost(90, 0.025, 0.001)),
                    replace(valid, conversion_evidence_id=""), replace(valid, quota_per_device_hour_upper=0)):
            with self.subTest(bad=bad), self.assertRaises(Blocked):
                bad.totals()

    def test_deadline_rechecked_before_launch_and_timeout_required(self):
        ledger = self.ledger()
        self.clock.now = DEADLINE - 360
        ledger.reserve("r", plan(), self.observation(ledger))
        self.clock.now += 1
        with self.assertRaisesRegex(Blocked, "deadline"):
            ledger.authorize_launch("r", self.observation(ledger))
        with self.assertRaises(Blocked):
            ledger.assert_within_timeout("r")
        self.path = self.path.parent / "timeout.jsonl"
        self.clock.now = DEADLINE - 1000
        ledger = self.ledger()
        ledger.reserve("r", plan(), self.observation(ledger))
        launch = ledger.authorize_launch("r", self.observation(ledger))
        ledger.assert_within_timeout("r")
        self.clock.now = launch["stop_by"]
        with self.assertRaises(StopRequired):
            ledger.assert_within_timeout("r")
        self.clock.now = DEADLINE
        with self.assertRaises(Blocked):
            ledger.reserve("after", plan(), self.observation(ledger))

    def test_launch_once_and_needs_new_fresh_observations(self):
        ledger = self.ledger()
        old = self.observation(ledger)
        self.clock.now += 1
        ledger.reserve("r", plan(), old)
        with self.assertRaisesRegex(Blocked, "after reservation"):
            ledger.authorize_launch("r", old)
        ledger.authorize_launch("r", self.observation(ledger))
        with self.assertRaisesRegex(Blocked, "consumed"):
            ledger.authorize_launch("r", self.observation(ledger))

    def test_failed_run_stays_charged_and_local_exit_does_not_stop_platform(self):
        ledger = self.ledger()
        ledger.reserve("failed", plan(3), self.observation(ledger))
        permit = ledger.authorize_launch("failed", self.observation(ledger))
        ledger.record_outcome("failed", outcome="failed", observed_wall_seconds=2, observed_device_hours=0.001,
                              evidence_id="simulated-failure")
        with self.assertRaisesRegex(Blocked, "unresolved"):
            ledger.reserve("other", plan(), self.observation(ledger))
        with self.assertRaisesRegex(Blocked, "reserved interval"):
            ledger.confirm_platform_stopped("failed", self.observation(ledger))
        self.clock.now = permit["stop_by"]
        ledger.confirm_platform_stopped("failed", self.observation(ledger))
        self.assertEqual(ledger.summary()["reserved_estimated_quota_hours"], 3)
        with self.assertRaisesRegex(Blocked, "budget"):
            ledger.reserve("too-much", plan(3), self.observation(ledger))
        ledger.reserve("fits", plan(2), self.observation(ledger))

    def test_account_floor_includes_unfinalized_estimates(self):
        ledger = self.ledger(expanded=True, expanded_approval_id="fixture-expanded")
        self.finish(ledger, "r1", cost=4)
        with self.assertRaisesRegex(Blocked, "floor"):
            ledger.reserve("r2", plan(2), self.observation(ledger, remaining_quota_hours=25))

    def test_external_observed_usage_counts_against_campaign(self):
        ledger = self.ledger()
        with self.assertRaisesRegex(Blocked, "budget"):
            ledger.reserve("r", plan(1), self.observation(ledger, remaining_quota_hours=25))

    def test_overrun_blocks_later_work_even_if_quota_billing_lags(self):
        ledger = self.ledger()
        ledger.reserve("over", plan(), self.observation(ledger))
        permit = ledger.authorize_launch("over", self.observation(ledger))
        ledger.record_outcome("over", outcome="timeout", observed_wall_seconds=361, observed_device_hours=0.1,
                              evidence_id="simulated-overrun")
        self.clock.now = permit["stop_by"] + 1
        ledger.confirm_platform_stopped("over", self.observation(ledger))
        with self.assertRaisesRegex(Blocked, "exceeded"):
            ledger.reserve("new", plan(), self.observation(ledger))

    def test_concurrent_reservations_have_one_winner(self):
        ledger = self.ledger()
        observation = self.observation(ledger)
        barrier = threading.Barrier(2)
        outcomes = []
        def reserve(index):
            own = AccountLedger(self.path, clock=self.clock)
            barrier.wait()
            try:
                own.reserve(f"r{index}", plan(3), observation)
                outcomes.append("accepted")
            except Blocked:
                outcomes.append("blocked")
        threads = [threading.Thread(target=reserve, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertCountEqual(outcomes, ["accepted", "blocked"])
        self.assertEqual(ledger.summary()["runs"], 1)

    def test_append_only_integrity_restart_and_clock_regression(self):
        ledger = self.ledger()
        initial = self.path.read_bytes()
        ledger.reserve("r", plan(), self.observation(ledger))
        self.assertTrue(self.path.read_bytes().startswith(initial))
        reopened = AccountLedger(self.path, clock=self.clock)
        self.assertEqual(reopened.summary()["runs"], 1)
        with self.assertRaises(FileExistsError):
            self.ledger()
        self.clock.now -= 1
        with self.assertRaisesRegex(Blocked, "regressed"):
            reopened.summary()
        self.clock.now += 1
        with self.path.open("ab") as stream:
            stream.write(b'{"interrupted_append":')
        with self.assertRaisesRegex(Blocked, "incomplete"):
            reopened.summary()

    def test_tamper_and_symlink_rejected(self):
        ledger = self.ledger()
        row = json.loads(self.path.read_text())
        row["payload"]["policy"]["allocation_enabled"] = False
        self.path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(Blocked, "hash"):
            ledger.summary()
        link = self.path.parent / "link.jsonl"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(Blocked, "symlinks"):
            AccountLedger(link)


if __name__ == "__main__":
    unittest.main()
