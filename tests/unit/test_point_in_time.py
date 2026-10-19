"""Point-in-time admission contracts; all inputs are artificial fixtures."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from wharton_lab.data.point_in_time import (
    DataManifest, MembershipUpdate, Observation, admit_data_file,
    build_asof_snapshot, eligible_assets,
)


def moment(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=timezone.utc)


def member(asset: str = "A") -> MembershipUpdate:
    return MembershipUpdate(asset, moment(1), moment(1), True)


class PointInTimeTests(unittest.TestCase):
    def snapshot(self, observations, **kwargs):
        settings = dict(decision_time=moment(5), assets=["A"], fields=["close"],
                        membership_updates=[member()])
        settings.update(kwargs)
        return build_asof_snapshot(observations, **settings)

    def test_publication_delay_and_equal_decision_availability(self):
        delayed = Observation("A", "close", moment(3), moment(6), 10)
        with self.assertRaisesRegex(ValueError, "missing as-of"):
            self.snapshot([delayed])
        result = self.snapshot([delayed], decision_time=moment(6))
        self.assertEqual(result.values, ((10.0,),))
        self.assertEqual(result.records[0][0], delayed)

    def test_revision_not_visible_until_available_and_event_priority(self):
        old = Observation("A", "close", moment(2), moment(2), 10, "original")
        revision = replace(old, available_time=moment(6), value=20, revision="revised")
        self.assertEqual(self.snapshot([revision, old]).values, ((10.0,),))
        self.assertEqual(self.snapshot([old, revision], decision_time=moment(7)).values, ((20.0,),))
        newer_event = Observation("A", "close", moment(3), moment(3), 30)
        self.assertEqual(self.snapshot([old, revision, newer_event], decision_time=moment(7)).values,
                         ((30.0,),))

    def test_future_perturbations_leave_snapshot_unchanged(self):
        past = Observation("A", "close", moment(3), moment(3), 10)
        future_event = Observation("A", "close", moment(5), moment(5), 100)
        unpublished = replace(past, available_time=moment(6), value=200, revision="future")
        expected = self.snapshot([past])
        self.assertEqual(self.snapshot([past, future_event, unpublished]), expected)
        self.assertEqual(self.snapshot([replace(unpublished, value=-999), past,
                                       replace(future_event, value=999)]), expected)

    def test_default_exclusion_and_effective_and_publication_clocks(self):
        updates = [MembershipUpdate("A", moment(3), moment(6), True),
                   MembershipUpdate("B", moment(7), moment(2), True)]
        self.assertEqual(eligible_assets(updates, decision_time=moment(5), assets=["A", "B"]), ())
        self.assertEqual(eligible_assets(updates, decision_time=moment(6), assets=["A", "B"]), ("A",))
        self.assertEqual(eligible_assets(updates, decision_time=moment(7), assets=["B", "A"]), ("B", "A"))
        self.assertEqual(eligible_assets([], decision_time=moment(7), assets=["A"]), ())

    def test_removal_and_membership_revision_use_available_vintage(self):
        removed = MembershipUpdate("A", moment(3), moment(4), False)
        corrected = replace(removed, available_time=moment(6), included=True, revision="correction")
        updates = [member(), removed, corrected]
        self.assertEqual(eligible_assets(updates, decision_time=moment(5), assets=["A"]), ())
        self.assertEqual(eligible_assets(updates, decision_time=moment(6), assets=["A"]), ("A",))

    def test_order_and_excluded_assets_need_no_missing_data(self):
        observations = [Observation(a, f, moment(3), moment(3), value)
                        for a, f, value in [("A", "close", 1), ("A", "volume", 2),
                                            ("B", "close", 3), ("B", "volume", 4)]]
        result = self.snapshot(observations, assets=["B", "UNKNOWN", "A"],
                               fields=["volume", "close"], membership_updates=[member("A"), member("B")])
        self.assertEqual(result.assets, ("B", "A"))
        self.assertEqual(result.excluded_assets, ("UNKNOWN",))
        self.assertEqual(result.values, ((4.0, 3.0), (2.0, 1.0)))
        self.assertEqual(result.membership_records, (member("B"), member("A")))

    def test_missing_and_stale_values_are_never_filled(self):
        old = Observation("A", "close", moment(2), moment(2), 10)
        with self.assertRaisesRegex(ValueError, "missing as-of"):
            self.snapshot([])
        with self.assertRaisesRegex(ValueError, "stale"):
            self.snapshot([old], max_age=timedelta(days=2))
        result = self.snapshot([old], max_age=timedelta(days=2), allow_missing=True)
        self.assertEqual(result.values, ((None,),))
        self.assertEqual(result.records, ((None,),))
        self.assertEqual(self.snapshot([old], max_age=timedelta(days=3)).values, ((10.0,),))

    def test_nonfinite_missing_and_boolean_observation_values_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), None, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Observation("A", "close", moment(2), moment(2), value)

    def test_naive_times_rejected_and_timezones_compared_by_instant(self):
        naive = datetime(2026, 1, 2)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            Observation("A", "close", naive, moment(3), 1)
        with self.assertRaises(ValueError):
            MembershipUpdate("A", moment(1), naive, True)
        with self.assertRaises(ValueError):
            self.snapshot([], decision_time=naive)
        india = timezone(timedelta(hours=5, minutes=30))
        value = Observation("A", "close", moment(2).astimezone(india), moment(3).astimezone(india), 4)
        self.assertEqual(value.event_time, moment(2))
        self.assertEqual(self.snapshot([value]).values, ((4.0,),))

    def test_conflicting_vintages_rejected_exact_duplicates_idempotent(self):
        value = Observation("A", "close", moment(3), moment(3), 1)
        self.assertEqual(self.snapshot([value, value]), self.snapshot([value]))
        for conflict in (replace(value, value=2), replace(value, revision="different"),
                         replace(value, available_time=moment(4))):
            with self.subTest(conflict=conflict), self.assertRaisesRegex(ValueError, "vintage"):
                self.snapshot([value, conflict])
        with self.assertRaisesRegex(ValueError, "vintage"):
            eligible_assets([member(), replace(member(), included=False)],
                            decision_time=moment(5), assets=["A"])

    def test_invalid_orders_and_age_rejected(self):
        for kwargs in ({"assets": ["A", "A"]}, {"fields": []}, {"fields": "close"}, {"assets": {"A"}},
                       {"max_age": timedelta(days=-1)}, {"max_age": 3}, {"allow_missing": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.snapshot([], **kwargs)


class ManifestAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "explicit.csv"
        self.raw = b"timestamp,value\r\n2026-01-01,1\r\n"
        self.path.write_bytes(self.raw)
        self.manifest = DataManifest(
            source="local deterministic unittest fixture", licence="locally generated test fixture",
            data_kind="synthetic", sha256=hashlib.sha256(self.raw).hexdigest(),
            availability_semantics="fixture available_time records the publication instant",
            corporate_action_policy="no corporate actions in this artificial fixture",
            missingness_policy="reject missing cells", universe_policy="explicit membership updates",
        )

    def test_exact_bytes_returned_without_file_rewrite(self):
        admitted = admit_data_file(self.path, self.manifest)
        self.assertEqual(admitted.raw, self.raw)
        self.assertEqual(admitted.sha256, self.manifest.sha256)
        self.assertFalse(admitted.market_eligible)
        self.assertEqual(self.path.read_bytes(), self.raw)
        self.path.write_bytes(b"changed later")
        self.assertEqual(admitted.raw, self.raw)

    def test_even_newline_change_invalidates_expected_hash(self):
        self.path.write_bytes(self.raw.replace(b"\r\n", b"\n"))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            admit_data_file(self.path, self.manifest)

    def test_synthetic_is_never_market_evidence(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            admit_data_file(self.path, self.manifest, for_market=True)
        with self.assertRaises(ValueError):
            replace(self.manifest, data_kind="unknown_cache")

    def test_market_declaration_path_is_structural_not_external_certification(self):
        # This exercises the declaration format with artificial bytes. It does
        # not assert that these bytes are market data or independently licensed.
        declared = replace(self.manifest, data_kind="market",
                           source="https://example.invalid/provenance-format-fixture",
                           licence="fictional licence declaration for a software test")
        admitted = admit_data_file(self.path, declared, for_market=True)
        self.assertTrue(admitted.market_eligible)
        self.assertEqual(admitted.raw, self.raw)

    def test_missing_provenance_rejected(self):
        fields = ("source", "licence", "availability_semantics", "corporate_action_policy",
                  "missingness_policy", "universe_policy")
        for field in fields:
            for missing in ("", "unknown", "none", None):
                with self.subTest(field=field, missing=missing), self.assertRaises(ValueError):
                    replace(self.manifest, **{field: missing})
        with self.assertRaises(ValueError):
            admit_data_file(self.path, None)
        with self.assertRaises(ValueError):
            replace(self.manifest, sha256="bad")

    def test_missing_explicit_file_never_falls_back_to_nearby_cache(self):
        with self.assertRaises(FileNotFoundError):
            admit_data_file(self.path.with_name("missing.csv"), self.manifest)
        self.assertEqual(self.path.read_bytes(), self.raw)


if __name__ == "__main__":
    unittest.main()
