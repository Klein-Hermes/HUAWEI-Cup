from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "src" / "f_q1_3_mixture_loss.py").is_file()
)
SCRIPT = PROJECT_ROOT / "src" / "f_q1_3_mixture_loss.py"
SPEC = importlib.util.spec_from_file_location("f_q1_3_mixture_loss", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AuditModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name) / "regmix_tables"
        self.root.mkdir()
        self.output = Path(self._temporary.name) / "audit"
        self._write_valid_fixture()

    def tearDown(self) -> None:
        self._temporary.cleanup()

    @staticmethod
    def _write_csv(path: Path, header: tuple[str, ...] | list[str], row: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerow(row)

    def _write_valid_fixture(self) -> None:
        mixture_row = ["recipe-1", "1", *("0" for _ in range(16))]
        loss_row = ["recipe-1", *("2.5" for _ in range(13))]
        for _, _, mixture_name, _, loss_name, _ in MODULE.AUDIT_SPLITS:
            self._write_csv(self.root / mixture_name, MODULE.CANONICAL_MIXTURE_COLUMNS, mixture_row)
            self._write_csv(self.root / loss_name, MODULE.CANONICAL_LOSS_COLUMNS, loss_row)

    def _run(self, output_name: str = "audit") -> dict:
        MODULE.DATA_ROOT = self.root
        return MODULE.run_audit(Path(self._temporary.name) / output_name)

    def test_valid_inputs_pass_and_outputs_are_isolated(self) -> None:
        summary = self._run()
        self.assertEqual(summary["status"], "PASS")
        self.assertFalse(summary["isolation_guards"]["model_functions_called"])
        self.assertFalse(summary["isolation_guards"]["closure_computed_or_saved"])
        self.assertEqual(
            {path.name for path in self.output.glob("*")},
            {"audit_summary.json", "file_roles.csv", "pair_integrity.csv", "train_composition_qc.csv"},
        )

    def test_duplicate_header_fails_with_machine_readable_result(self) -> None:
        bad_header = list(MODULE.CANONICAL_MIXTURE_COLUMNS)
        bad_header[-1] = bad_header[-2]
        bad_row = ["recipe-1", "1", *("0" for _ in range(16))]
        self._write_csv(self.root / "test_mixture_1m.csv", bad_header, bad_row)
        summary = self._run()
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any(issue["code"] == "SCHEMA_MISMATCH" for issue in summary["issues"]))
        persisted = json.loads((self.output / "audit_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "FAIL")

    def test_same_inputs_have_same_conclusion_hash(self) -> None:
        first = self._run("audit_first")
        second = self._run("audit_second")
        self.assertEqual(first["audit_conclusion_sha256"], second["audit_conclusion_sha256"])
        self.assertNotEqual(first["started_at_utc"], second["started_at_utc"])

    def test_pair_key_mismatch_fails(self) -> None:
        loss_row = ["different-recipe", *("2.5" for _ in range(13))]
        self._write_csv(
            self.root / "test_pile_loss_60m.csv", MODULE.CANONICAL_LOSS_COLUMNS, loss_row
        )
        summary = self._run()
        self.assertEqual(summary["status"], "FAIL")
        self.assertTrue(any(issue["code"] == "PAIR_KEY_MISMATCH" for issue in summary["issues"]))

    def test_invalid_training_share_fails_without_repair(self) -> None:
        mixture_row = ["recipe-1", "-1", "2", *("0" for _ in range(15))]
        self._write_csv(
            self.root / "train_mixture_1m.csv", MODULE.CANONICAL_MIXTURE_COLUMNS, mixture_row
        )
        summary = self._run()
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["training_value_audit"]["negative_share_values"], 1)
        self.assertFalse(summary["isolation_guards"]["rows_deleted_or_repaired"])

    def test_cli_returns_nonzero_for_failed_audit(self) -> None:
        bad_header = list(MODULE.CANONICAL_LOSS_COLUMNS[:-1])
        bad_row = ["recipe-1", *("2.5" for _ in range(12))]
        self._write_csv(self.root / "test_pile_loss_1B.csv", bad_header, bad_row)
        argv = [
            str(SCRIPT), "--mode", "audit", "--data-root", str(self.root),
            "--output-dir", str(Path(self._temporary.name) / "cli_failure"),
        ]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            self.assertEqual(MODULE.main(), 2)


if __name__ == "__main__":
    unittest.main()
