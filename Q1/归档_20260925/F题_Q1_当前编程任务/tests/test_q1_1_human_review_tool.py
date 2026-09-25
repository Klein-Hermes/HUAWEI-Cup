from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "q1_1_human_review_tool.py"
SPEC = importlib.util.spec_from_file_location("q1_1_human_review_tool", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_source_form(path: Path, ids: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(MODULE.SOURCE_HEADER))
        writer.writeheader()
        for blind_id in ids:
            writer.writerow({"blind_id": blind_id})


def write_packet(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# packet\n\n")
        for blind_id, content in records:
            fence = MODULE.choose_fence(content)
            stream.write(f"## {blind_id}\n\n{fence}text\n{content}\n{fence}\n\n")


def fill_form(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["blind_id", *MODULE.DIMENSIONS])
        writer.writeheader()
        for row in rows:
            writer.writerow({"blind_id": row["blind_id"], **{dimension: 3 for dimension in MODULE.DIMENSIONS}})


class HumanReviewToolTests(unittest.TestCase):
    def test_prepare_validate_merge_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            ids = [f"Q11-{i:04d}" for i in range(1, 5)]
            source_form = tmp_path / "source.csv"
            packet = tmp_path / "packet.md"
            write_source_form(source_form, ids)
            write_packet(packet, [(blind_id, f"text {blind_id} ~~~ data") for blind_id in ids])

            output = tmp_path / "prepared"
            manifest = MODULE.prepare_packages(packet, source_form, output, seeds=(11, 22), items_per_part=2, expected_count=4)
            self.assertEqual(manifest["input_audit"]["sample_count"], 4)
            self.assertNotEqual(manifest["raters"][0]["order_sha256"], manifest["raters"][1]["order_sha256"])
            self.assertEqual(len(list((output / "rater_1").glob("packet_part_*.md"))), 2)

            r1 = output / "rater_1" / "rater_1_form.csv"
            r2 = output / "rater_2" / "rater_2_form.csv"
            fill_form(r1)
            fill_form(r2)
            merged = tmp_path / "merged.csv"
            result = MODULE.merge_forms(source_form, r1, r2, "R1", "R2", merged, expected_count=4)
            self.assertEqual(result["status"], "ready")
            with merged.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["blind_id"] for row in rows], ids)
            self.assertTrue(all(row["rater_1_overall"] == "3" and row["rater_2_overall"] == "3" for row in rows))

    def test_blank_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            ids = ["Q11-0001"]
            source_form = tmp_path / "source.csv"
            write_source_form(source_form, ids)
            ratings = tmp_path / "ratings.csv"
            with ratings.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["blind_id", *MODULE.DIMENSIONS])
                writer.writeheader()
                writer.writerow({"blind_id": ids[0], **{dimension: "" for dimension in MODULE.DIMENSIONS}})
            with self.assertRaises(MODULE.ContractError):
                MODULE.validate_rater_form(ratings, set(ids), expected_count=1)

    def test_packet_and_form_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source_form = tmp_path / "source.csv"
            packet = tmp_path / "packet.md"
            write_source_form(source_form, ["Q11-0001"])
            write_packet(packet, [("Q11-0002", "text")])
            with self.assertRaises(MODULE.ContractError):
                MODULE.validate_frozen_inputs(packet, source_form, expected_count=1)

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            ids = ["Q11-0001"]
            source_form = tmp_path / "source.csv"
            packet = tmp_path / "packet.md"
            write_source_form(source_form, ids)
            write_packet(packet, [(ids[0], "text")])
            output = tmp_path / "prepared"
            output.mkdir()
            with self.assertRaises(MODULE.ContractError):
                MODULE.prepare_packages(packet, source_form, output, expected_count=1)

    def test_same_rater_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            ids = ["Q11-0001"]
            source_form = tmp_path / "source.csv"
            write_source_form(source_form, ids)
            ratings = tmp_path / "ratings.csv"
            with ratings.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["blind_id", *MODULE.DIMENSIONS])
                writer.writeheader()
                writer.writerow({"blind_id": ids[0], **{dimension: 3 for dimension in MODULE.DIMENSIONS}})
            with self.assertRaises(MODULE.ContractError):
                MODULE.merge_forms(source_form, ratings, ratings, "R1", "R2", tmp_path / "merged.csv", expected_count=1)

    def test_existing_merge_manifest_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            ids = ["Q11-0001"]
            source_form = tmp_path / "source.csv"
            write_source_form(source_form, ids)
            forms = []
            for number in (1, 2):
                ratings = tmp_path / f"ratings_{number}.csv"
                with ratings.open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["blind_id", *MODULE.DIMENSIONS])
                    writer.writeheader()
                    writer.writerow({"blind_id": ids[0], **{dimension: 3 for dimension in MODULE.DIMENSIONS}})
                forms.append(ratings)
            merged = tmp_path / "merged.csv"
            sidecar = merged.with_suffix(merged.suffix + ".manifest.json")
            sidecar.write_text("sentinel", encoding="utf-8")
            with self.assertRaises(MODULE.ContractError):
                MODULE.merge_forms(source_form, forms[0], forms[1], "R1", "R2", merged, expected_count=1)
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "sentinel")
            self.assertFalse(merged.exists())

    def test_multiline_fence_and_trailing_newline_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            records = [
                ("Q11-0001", "first line\n## Q11-9999\n~~~~~~ inside\nlast line\n"),
                ("Q11-0002", "x" * 100_000),
            ]
            source = tmp_path / "source.md"
            write_packet(source, records)
            parsed = MODULE.parse_blind_packet(source)
            self.assertEqual(parsed, records)
            regenerated = tmp_path / "regenerated.md"
            MODULE.write_packet_part(regenerated, 1, 1, 1, parsed)
            self.assertEqual(MODULE.parse_blind_packet(regenerated), records)


if __name__ == "__main__":
    unittest.main()
