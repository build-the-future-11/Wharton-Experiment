"""Adversarial source-readback tests without importing the numerical stack."""

import ast
import copy
import importlib.util
import json
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_s14_saved_notebook.py"
SPEC = importlib.util.spec_from_file_location("saved_notebook", SOURCE)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class SavedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local = verifier.NOTEBOOK.read_bytes()
        cls.release = json.loads(verifier.RELEASE.read_bytes())
        cls.archive = (verifier.PACKAGE / "dist" / cls.release["archive"]).read_bytes()

    def check(self, notebook):
        return verifier.compare(self.local, json.dumps(notebook).encode(), self.release, self.archive)

    def test_real_release_self_check_does_not_claim_remote_identity(self):
        result = verifier.compare(self.local, self.local, self.release, self.archive)
        self.assertEqual(result["status"], "PASS_SOURCE_CONTENT_MATCH")
        self.assertEqual(result["cells_compared"], 4)
        self.assertFalse(result["remote_identity_verified"])
        self.assertFalse(result["runtime_qualified"])

    def test_line_arrays_and_strings_match_with_different_metadata(self):
        notebook = json.loads(self.local)
        for cell in notebook["cells"]:
            cell["source"] = "".join(cell["source"])
            cell["metadata"] = {"downloaded": True}
            cell["outputs"] = []
        self.assertEqual(self.check(notebook)["status"], "PASS_SOURCE_CONTENT_MATCH")

    def test_one_character_source_mutation_is_rejected(self):
        notebook = json.loads(self.local)
        notebook["cells"][1]["source"][0] += " "
        with self.assertRaisesRegex(ValueError, "content/order/types"):
            self.check(notebook)

    def test_cell_order_and_missing_cells_are_rejected(self):
        for operation in ("reverse", "drop", "append"):
            notebook = json.loads(self.local)
            if operation == "reverse":
                notebook["cells"].reverse()
            elif operation == "drop":
                notebook["cells"].pop()
            else:
                notebook["cells"].append(copy.deepcopy(notebook["cells"][-1]))
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "content/order/types"):
                self.check(notebook)

    def test_cell_type_and_malformed_source_are_rejected(self):
        notebook = json.loads(self.local)
        notebook["cells"][0]["cell_type"] = "code"
        with self.assertRaises(ValueError):
            self.check(notebook)
        notebook["cells"][0]["source"] = [17]
        with self.assertRaises(ValueError):
            self.check(notebook)

    def test_trusted_local_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local notebook"):
            verifier.compare(self.local + b" ", self.local, self.release, self.archive)
        with self.assertRaisesRegex(ValueError, "local archive"):
            verifier.compare(self.local, self.local, self.release, self.archive + b" ")

    def test_duplicate_json_and_nonfinite_constants_are_rejected(self):
        for data in (b'{"nbformat": 3, "nbformat": 4}', b'{"nbformat": NaN}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                verifier.normalized_cells(data)

    def test_embedded_archive_is_checked_even_with_self_consistent_notebook_hash(self):
        notebook = json.loads(self.local)
        source = "".join(notebook["cells"][1]["source"])
        # Mutate the literal, then bind the altered notebook as a fixture.
        tree = ast.parse(source)
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "RELEASE"
        )
        literal = ast.literal_eval(assignment.value.args[0])
        notebook["cells"][1]["source"] = source.replace(literal, "dGFtcGVyZWQ=", 1)
        changed = json.dumps(notebook).encode()
        release = {**self.release, "notebook_sha256": verifier.digest(changed)}
        with self.assertRaisesRegex(ValueError, "embedded archive"):
            verifier.compare(changed, changed, release, self.archive)


if __name__ == "__main__":
    unittest.main()
