"""Read-only Unicode integrity audit for the F-question A1-A3 text records.

Run only after reading 题目分析报告.md, especially §§3 and 8. This script
never prints or writes raw content, never modifies source JSONL, and does not
change the frozen Q1 preprocessing outputs. It writes record-level hashes and
character counts so Q1.1 can join integrity flags without exposing text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_ROOT = ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value"
DEFAULT_OUTPUT = ROOT / "results" / "q1_1"

ZERO_WIDTH_CODEPOINTS = {
    0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF,
}
BIDI_CODEPOINTS = (
    {0x061C, 0x200E, 0x200F}
    | set(range(0x202A, 0x202F))
    | set(range(0x2066, 0x206A))
)
LINE_BREAKS = {"\t", "\n", "\r"}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def identity_hash(dataset: str, sample_id: Any) -> str:
    value = f"{dataset}\0{sample_id if sample_id is not None else ''}"
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def input_paths(a_root: Path) -> dict[str, list[Path]]:
    paths = {
        "A1": [a_root / "slimpajama_quality_signal_sample.jsonl.xz"],
        "A2": sorted((a_root / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz")),
        "A3": sorted((a_root / "slimpajama_quality_extended").glob("github_*.jsonl.xz")),
    }
    if not paths["A1"][0].exists() or not paths["A2"] or not paths["A3"]:
        raise FileNotFoundError("A1/A2/A3 raw JSONL inputs are incomplete")
    return paths


def record_domain(dataset: str, path: Path, record: dict[str, Any]) -> str:
    if dataset == "A1":
        return str(record.get("_source_domain") or "unknown")
    return path.name.split("_", maxsplit=1)[0]


def audit_content(text: str) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    codepoint_counts: Counter[int] = Counter()
    n_zero_width = 0
    n_bidi = 0
    n_other_control = 0
    n_line_separator = 0
    n_private_use = 0
    n_noncharacter = 0

    for char in text:
        codepoint = ord(char)
        category = unicodedata.category(char)
        if category == "Cf":
            category_counts["Cf_format"] += 1
            codepoint_counts[codepoint] += 1
        elif category == "Cc" and char not in LINE_BREAKS:
            category_counts["Cc_control"] += 1
            codepoint_counts[codepoint] += 1
            n_other_control += 1
        elif category in {"Zl", "Zp"}:
            category_counts[category] += 1
            codepoint_counts[codepoint] += 1
            n_line_separator += 1
        elif category == "Co":
            category_counts["Co_private_use"] += 1
            codepoint_counts[codepoint] += 1
            n_private_use += 1

        n_zero_width += codepoint in ZERO_WIDTH_CODEPOINTS
        n_bidi += codepoint in BIDI_CODEPOINTS
        n_noncharacter += (
            0xFDD0 <= codepoint <= 0xFDEF
            or codepoint & 0xFFFF in {0xFFFE, 0xFFFF}
        )

    return {
        "category_counts": category_counts,
        "codepoint_counts": codepoint_counts,
        "zero_width_count": n_zero_width,
        "bidi_control_count": n_bidi,
        "other_control_count": n_other_control,
        "line_separator_count": n_line_separator,
        "private_use_count": n_private_use,
        "noncharacter_count": n_noncharacter,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Unicode integrity of F-question A1-A3 content fields.")
    parser.add_argument("--a-root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    totals: dict[str, Any] = {}
    for dataset, files in input_paths(args.a_root).items():
        dataset_rows = 0
        missing_content = 0
        parse_errors = 0
        file_hashes: dict[str, str] = {}
        for path in files:
            file_digest = hashlib.sha256()
            with path.open("rb") as raw_handle:
                for chunk in iter(lambda: raw_handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            file_hashes[path.name] = file_digest.hexdigest()

            with lzma.open(path, "rt", encoding="utf-8", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    dataset_rows += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    sample_id = record.get("id")
                    domain = record_domain(dataset, path, record)
                    raw_content = record.get("content")
                    content_present = isinstance(raw_content, str)
                    if not content_present:
                        missing_content += 1
                        audited = None
                        raw_content = ""
                        nfc_content = ""
                        category_counts = Counter()
                        codepoint_counts = Counter()
                        any_flag = None
                    else:
                        audited = audit_content(raw_content)
                        nfc_content = unicodedata.normalize("NFC", raw_content)
                        category_counts = audited["category_counts"]
                        codepoint_counts = audited["codepoint_counts"]
                        any_flag = bool(category_counts or audited["noncharacter_count"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "source_domain": domain,
                            "sample_key_sha256": identity_hash(dataset, sample_id),
                            "content_field_present": content_present,
                            "content_sha256": sha256_text(raw_content) if content_present else "",
                            "content_nfc_sha256": sha256_text(nfc_content) if content_present else "",
                            "nfc_changed": raw_content != nfc_content if content_present else None,
                            "format_char_count": category_counts["Cf_format"] if content_present else None,
                            "other_control_count": audited["other_control_count"] if content_present else None,
                            "line_separator_count": audited["line_separator_count"] if content_present else None,
                            "private_use_count": audited["private_use_count"] if content_present else None,
                            "noncharacter_count": audited["noncharacter_count"] if content_present else None,
                            "zero_width_count": audited["zero_width_count"] if content_present else None,
                            "bidi_control_count": audited["bidi_control_count"] if content_present else None,
                            "unicode_category_counts": json.dumps(
                                dict(category_counts) if content_present else None,
                                ensure_ascii=False, sort_keys=True
                            ),
                            "unicode_codepoint_counts": json.dumps(
                                {
                                    f"U+{cp:04X}": count
                                    for cp, count in sorted(codepoint_counts.items())
                                } if content_present else None,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "unicode_integrity_flag": any_flag,
                        }
                    )

        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            if row["dataset"] != dataset:
                continue
            group = grouped[row["source_domain"]]
            group["rows"] += 1
            group["content_field_present_rows"] += row["content_field_present"]
            if not row["content_field_present"]:
                group["content_field_missing_rows"] += 1
                continue
            group["format_flag_rows"] += row["format_char_count"] > 0
            group["other_control_flag_rows"] += row["other_control_count"] > 0
            group["line_separator_flag_rows"] += row["line_separator_count"] > 0
            group["unicode_integrity_flag_rows"] += row["unicode_integrity_flag"]
            group["nfc_changed_rows"] += row["nfc_changed"]

        totals[dataset] = {
            "rows": dataset_rows,
            "missing_content_rows": missing_content,
            "content_audit_status": (
                "complete" if missing_content == 0
                else "not_applicable_no_content_field" if missing_content == dataset_rows
                else "partial_content_field_coverage"
            ),
            "json_parse_errors": parse_errors,
            "files": file_hashes,
            "by_source_domain": {
                domain: dict(counts) for domain, counts in sorted(grouped.items())
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "text_integrity_audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "audit_version": "q1-text-integrity-v1",
        "read_only": True,
        "raw_content_exported": False,
        "raw_files_modified": False,
        "detection_scope": [
            "Unicode general category Cf",
            "Unicode category Cc excluding tab/newline/carriage-return",
            "Unicode categories Zl/Zp",
            "Unicode category Co",
            "Unicode noncharacters",
            "explicit zero-width and bidi-control codepoint counts",
        ],
        "nfc_change_is_diagnostic_only": True,
        "report_baseline_invisible_rows_a1": 334,
        "datasets": totals,
        "record_audit_csv": str(csv_path),
        "record_audit_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    summary_path = args.output_dir / "text_integrity_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "record_audit": str(csv_path),
                "datasets": totals,
                "raw_content_exported": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
