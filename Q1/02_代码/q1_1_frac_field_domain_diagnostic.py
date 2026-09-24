#!/usr/bin/env python3
"""Report exact-formula match rates by the frozen A1 source-domain mapping."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_MARKER = Path("中文题目/F题/real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz")
ROOT = next((p for p in Path(__file__).resolve().parents if (p / ROOT_MARKER).is_file()), Path(__file__).resolve().parents[1])
PREPROCESSED = ROOT / "results/q1_common_preprocess/v1/a1_preprocessed.csv.gz"
AUDIT_DIR = ROOT / "results/q1_1/fraction_recalculation_audit/v2"
TOLERANCE = 0.01
EXPECTED_ROWS = 51_230
CANDIDATES = {
    "rps_doc_frac_unique_words": ["calc_rps_doc_frac_unique_words_pct"],
    "rps_doc_frac_no_alph_words": [
        "calc_rps_doc_frac_no_alph_words_pct",
        "calc_complement_of_ascii_english_letter_character_fraction_pct",
    ],
    "rps_doc_frac_chars_top_2gram": ["calc_rps_doc_frac_chars_top_2gram_pct"],
    "rps_doc_frac_chars_top_3gram": ["calc_rps_doc_frac_chars_top_3gram_pct"],
    "rps_lines_uppercase_letter_fraction": [
        "calc_uppercase_mean_all_lines_pct",
        "calc_uppercase_mean_nonempty_lines_pct",
        "calc_uppercase_char_weighted_pct",
    ],
    "rps_lines_ending_with_terminal_punctution_mark": [
        "calc_terminal_mean_all_lines_pct",
        "calc_terminal_mean_nonempty_lines_pct",
    ],
    "rps_lines_numerical_chars_fraction": [
        "calc_numerical_mean_all_lines_pct",
        "calc_numerical_mean_nonempty_lines_pct",
        "calc_numerical_char_weighted_pct",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed", type=Path, default=PREPROCESSED)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args()

    preprocessed = args.preprocessed.resolve()
    audit_dir = args.audit_dir.resolve()
    manifest_path = audit_dir / "manifest.json"
    detail_path = audit_dir / "a1_record_recalculation.csv.gz"
    if not preprocessed.is_file() or not manifest_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError("Required frozen preprocessing or full audit inputs are missing")
    audit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("status") != "full_a1_complete" or audit_manifest.get("rows_recalculated") != EXPECTED_ROWS:
        raise ValueError("The audit directory is not a completed 51,230-row A1 run")
    if sha256(detail_path) != audit_manifest["output_sha256"][detail_path.name]:
        raise ValueError("Full audit detail SHA-256 does not match its manifest")

    counts = defaultdict(lambda: [0, 0, 0.0, 0.0])
    first_domains = []
    seen = 0
    with gzip.open(preprocessed, "rt", encoding="utf-8", newline="") as pre_f, gzip.open(
        detail_path, "rt", encoding="utf-8", newline=""
    ) as detail_f:
        pre_reader = csv.DictReader(pre_f)
        detail_reader = csv.DictReader(detail_f)
        for row_number, (pre, detail) in enumerate(zip(pre_reader, detail_reader), 1):
            seen += 1
            if pre["id"] != detail["sample_id"]:
                raise ValueError(f"A1 ID/order mismatch at row {row_number}")
            domain = pre["source_domain"] or "unknown"
            if row_number <= 50:
                first_domains.append(domain)
            for field, calc_columns in CANDIDATES.items():
                stored = float(detail[f"stored_{field}"])
                for calc_column in calc_columns:
                    calculated = float(detail[calc_column])
                    key = (domain, field, calc_column)
                    aggregate = counts[key]
                    aggregate[0] += 1
                    difference = abs(stored - calculated)
                    aggregate[1] += int(difference <= TOLERANCE)
                    aggregate[2] += difference
                    aggregate[3] = max(aggregate[3], difference)
        if next(pre_reader, None) is not None or next(detail_reader, None) is not None:
            raise ValueError("A1 input row counts differ")
    if seen != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} aligned rows, got {seen}")

    output_rows = []
    for (domain, field, candidate), (n, matched, abs_error_sum, max_error) in sorted(counts.items()):
        output_rows.append({
            "source_domain": domain,
            "field": field,
            "candidate": candidate,
            "n": n,
            "n_within_0_01pp": matched,
            "match_rate": matched / n,
            "mean_absolute_error_pp": abs_error_sum / n,
            "max_absolute_error_pp": max_error,
        })
    output_path = audit_dir / "domain_match_by_source.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    script_path = Path(__file__).resolve()
    diagnostic_manifest = {
        "schema_version": "q1-1-frac-domain-diagnostic-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_aligned_by_id_and_order": seen,
        "first_50_source_domains": sorted(set(first_domains)),
        "first_50_all_same_domain": len(set(first_domains)) == 1,
        "comparison_tolerance_percentage_points": TOLERANCE,
        "preprocessed_a1": {"path": str(preprocessed.relative_to(ROOT)), "sha256": sha256(preprocessed)},
        "full_audit_detail": {"path": str(detail_path.relative_to(ROOT)), "sha256": sha256(detail_path)},
        "script": {"path": str(script_path.relative_to(ROOT)), "sha256": sha256(script_path)},
        "output": {"path": str(output_path.relative_to(ROOT)), "sha256": sha256(output_path)},
        "no_candidate_selected": True,
    }
    manifest_out = audit_dir / "domain_diagnostic_manifest.json"
    manifest_out.write_text(json.dumps(diagnostic_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"rows_aligned={seen}; first50_domains={diagnostic_manifest['first_50_source_domains']}")
    print(f"output={output_path}")
    print(f"manifest={manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
