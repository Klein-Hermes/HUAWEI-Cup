#!/usr/bin/env python
"""Audit invisible Unicode in A1 content without exposing or scoring text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_DIR = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "q1_2_conflict_model" / "content_integrity"
STANDARD_TEXT_CONTROLS = {"\t", "\n", "\r"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_content(value: object) -> Tuple[set[str], set[str], bool, int, int, str, bool]:
    content = value if isinstance(value, str) else ""
    nfkc = unicodedata.normalize("NFKC", content)
    codepoints = set()
    categories = set()
    has_nonascii_space = False
    for char in content:
        category = unicodedata.category(char)
        if category == "Cf" or (category == "Cc" and char not in STANDARD_TEXT_CONTROLS):
            codepoints.add(f"U+{ord(char):04X}")
            categories.add(category)
        if category == "Zs" and char != " ":
            has_nonascii_space = True
    return (
        codepoints,
        categories,
        has_nonascii_space,
        len(content),
        len(nfkc),
        hashlib.sha256(nfkc.encode("utf-8")).hexdigest(),
        nfkc != content,
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    preprocess_dir = Path(args.preprocess_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preprocess_manifest_path = preprocess_dir / "manifest.json"
    manifest = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("freeze_status") != "FROZEN":
        raise ValueError("公共预处理未冻结，停止内容完整性审计")
    a1_input = next(row for row in manifest["input_files"] if row["dataset"] == "a1")
    input_path = Path(a1_input["path"])
    expected_hash = str(a1_input["sha256"])
    actual_hash = sha256_file(input_path)
    if actual_hash != expected_hash:
        raise ValueError("A1 原始 JSONL.XZ 的 SHA-256 与冻结清单不符")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    category_rows: Counter[str] = Counter()
    codepoint_rows: Counter[str] = Counter()
    nfkc_changed_rows = 0
    records = 0
    with lzma.open(input_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"A1 JSONL 第 {line_number} 行无法解析") from exc
            records += 1
            content = record.get("content", "")
            codepoints, categories, nonascii_space, raw_len, nfkc_len, nfkc_hash, nfkc_changed = inspect_content(content)
            if nfkc_changed:
                nfkc_changed_rows += 1
            for category in categories:
                category_rows[category] += 1
            for codepoint in codepoints:
                codepoint_rows[codepoint] += 1
            if codepoints:
                rows.append({
                    "id": str(record.get("id", "")),
                    "source_domain": str(record.get("_source_domain") or "unknown"),
                    "content_sha256_nfkc": nfkc_hash,
                    "content_length_chars": raw_len,
                    "nfkc_length_chars": nfkc_len,
                    "nfkc_changed": nfkc_changed,
                    "has_format_control_cf": "Cf" in categories,
                    "has_nonstandard_control_cc": "Cc" in categories,
                    "has_nonascii_space": nonascii_space,
                    "invisible_codepoints": ";".join(sorted(codepoints)),
                })

    if records != int(manifest["summaries"]["a1"]["rows"]):
        raise ValueError(f"扫描行数 {records} 与冻结清单不符")
    if sha256_file(input_path) != actual_hash:
        raise ValueError("A1 原始文件在扫描过程中发生变化")

    flagged_path = output_dir / "a1_invisible_content_ids.csv"
    write_csv(
        flagged_path,
        rows,
        ["id", "source_domain", "content_sha256_nfkc", "content_length_chars", "nfkc_length_chars",
         "nfkc_changed", "has_format_control_cf", "has_nonstandard_control_cc", "has_nonascii_space",
         "invisible_codepoints"],
    )
    audit = {
        "schema_version": "q1-a1-content-integrity-audit-v1",
        "status": "complete",
        "input_dataset": "a1",
        "input_file": str(input_path),
        "input_sha256": actual_hash,
        "records_scanned": records,
        "records_flagged_cf_or_nonstandard_cc": len(rows),
        "records_with_nfkc_changes": nfkc_changed_rows,
        "flag_rule": "Unicode General Category Cf, plus Cc excluding TAB/LF/CR; report non-ASCII Zs separately",
        "normalization": "NFKC used to compute per-record digest only; no text is written or used as a model feature",
        "reported_prior_scan_count": 334,
        "reported_count_matches_this_rule": len(rows) == 334,
        "category_record_counts": dict(category_rows),
        "codepoint_record_counts": dict(sorted(codepoint_rows.items())),
        "a2_a3_prior_scan_note": "The referenced analysis report states no same-type invisible-Unicode hits for A2/A3; this script audits A1 only.",
        "output_sha256": {flagged_path.name: sha256_file(flagged_path)},
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": audit["status"],
        "records_scanned": records,
        "records_flagged_cf_or_nonstandard_cc": len(rows),
        "reported_prior_scan_count": 334,
        "reported_count_matches_this_rule": audit["reported_count_matches_this_rule"],
        "records_with_nfkc_changes": nfkc_changed_rows,
        "category_record_counts": dict(category_rows),
        "codepoint_record_counts": dict(sorted(codepoint_rows.items())),
        "output_dir": str(output_dir),
        "input_sha256": actual_hash,
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-dir", default=str(DEFAULT_PREPROCESS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
