"""Deterministic artificial ledger contracts, never market-performance evidence."""
import unittest
from decimal import Decimal as D
from wharton_final.accounting import State, Policy, transition


INVALID_FLAGS = ('false', 'true', '', 0, 1, None, [], {})


class TerminalFlagTests(unittest.TestCase):
    def state(self, **changes):
        values = dict(at=0, units=(10,), prices=(10,), cash=5)
        values.update(changes)
        return State(**values)

    def step(self, state, terminal, treatment='liquidate'):
        return transition(state, at=1, returns=[0],
            policy=Policy('cash_only', 'carry', 'after_market', treatment, '.1'),
            cash_rate=0, return_kind='price', terminal=terminal)

    def test_non_boolean_terminal_rejected_before_events(self):
        # An input string "false" must not silently cause a $100 sale and $10 fee.
        for flag in INVALID_FLAGS:
            with self.subTest(flag=repr(flag)):
                state = self.state()
                with self.assertRaisesRegex(ValueError, 'terminal.*bool'):
                    self.step(state, flag)
                self.assertEqual(state.units, (D(10),))
                self.assertEqual(state.cash, D(5))
                self.assertIs(state.closed, False)

    def test_deserialized_state_requires_boolean_closed(self):
        for flag in INVALID_FLAGS:
            with self.subTest(flag=repr(flag)):
                with self.assertRaisesRegex(ValueError, 'closed.*bool'):
                    self.state(closed=flag)

    def test_false_preserves_inventory_without_terminal_fees(self):
        for treatment in ('mark', 'liquidate'):
            with self.subTest(treatment=treatment):
                receipt = self.step(self.state(), False, treatment)
                self.assertEqual(receipt.state.units, (D(10),))
                self.assertEqual(receipt.state.cash, D(5))
                self.assertEqual(receipt.state.wealth, D(105))
                self.assertEqual(receipt.fees, D(0))
                self.assertEqual(receipt.turnover, D(0))
                self.assertIs(receipt.state.closed, False)
                receipt.verify()

    def test_true_keeps_explicit_terminal_treatment_and_exact_fees(self):
        marked = self.step(self.state(), True, 'mark')
        self.assertEqual(marked.state.units, (D(10),))
        self.assertEqual(marked.state.cash, D(5))
        self.assertEqual(marked.fees, D(0))
        liquidated = self.step(self.state(), True, 'liquidate')
        self.assertEqual(liquidated.turnover, D(100))
        self.assertEqual(liquidated.fees, D(10))
        self.assertEqual(liquidated.state.units, (D(0),))
        self.assertEqual(liquidated.state.cash, D(95))
        for receipt in (marked, liquidated):
            self.assertIs(receipt.state.closed, True)
            receipt.verify()
            with self.assertRaisesRegex(ValueError, 'open portfolio'):
                transition(receipt.state, at=2, returns=[0],
                    policy=Policy('cash_only', 'carry', 'after_market', 'mark', '.1'),
                    cash_rate=0, return_kind='price')
