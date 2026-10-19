"""Bounded stdlib regression: ambiguous network failures must fail closed."""

import errno
import unittest
from unittest.mock import MagicMock, patch

from frem_s10.errors import IntegrityError
from frem_s10.runtime import kaggle_contract


class NetworkDenialTests(unittest.TestCase):
    def check_result(self, error, accepted):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.connect.side_effect = error
        with patch.object(kaggle_contract, "require_platform"), patch.object(
            kaggle_contract.socket, "socket", return_value=connection
        ):
            if accepted:
                kaggle_contract.network_unreachable()
            else:
                with self.assertRaises(IntegrityError):
                    kaggle_contract.network_unreachable()

    def test_timeout_is_unknown_not_denial(self):
        self.check_result(TimeoutError("timed out"), False)

    def test_unspecified_error_is_unknown(self):
        self.check_result(OSError("unknown failure"), False)

    def test_reachable_network_rejected(self):
        self.check_result(None, False)

    def test_connection_refused_does_not_prove_isolation(self):
        self.check_result(ConnectionRefusedError(errno.ECONNREFUSED, "refused"), False)

    def test_explicit_denials_retained(self):
        for number in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM, errno.EACCES):
            with self.subTest(errno=number):
                self.check_result(OSError(number, "denied"), True)


if __name__ == "__main__":
    unittest.main()
