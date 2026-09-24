#!/usr/bin/env python3
"""Independently recalculate seven RedPajama-style Q1 A1 indicators.

This is an audit-only tool. It reads the hashed A1 JSONL.XZ attachment and
compares public RedPajama-style formulas against stored scalar values. It
never writes to or recalculates Q1/Q1.1/Q1.2/Q1.3 model outputs.

The OpenDataLab generation commit and its list-to-document aggregation code
are not present in the project. Accordingly, line metrics are reported under
several explicit aggregation candidates; the tool does not select one by fit.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import math
import re
import statistics
import string
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_INPUT_RELATIVE = Path(
    "中文题目/F题/real_attachments/A_data_value/"
    "slimpajama_quality_signal_sample.jsonl.xz"
)
ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / _INPUT_RELATIVE).is_file()),
    Path(__file__).resolve().parents[1],
)
INPUT_PATH = (
    ROOT / _INPUT_RELATIVE
)
PREPROCESS_MANIFEST = ROOT / "Q1" / "03_结果" / "00_公共预处理" / "v1" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "q1_1" / "fraction_recalculation_audit" / "v1"
ABS_TOLERANCE = 0.01  # percentage points
UPSTREAM_PRECISION = 8
EXPECTED_A1_ROWS = 51_230

FIELDS = (
    "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction",
)

UPSTREAM_URLS = {
    "normalization": "https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/utilities/text/normalization.py",
    "document": "https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/document.py",
    "natural_language": "https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/natural_language.py",
    "lines": "https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/lines.py",
    "repetitions": "https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/repetitions.py",
    "opendatalab_card": "https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater",
}

ASCII_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
WORDPUNCT_PATTERN = re.compile(r"\w+|[^\w\s]+", flags=re.UNICODE)
ALPHABET_PATTERN = re.compile(r"[a-zA-Z]")
LINE_PATTERN = re.compile(r"([^\n]*\n|[^\n]+$)")
TERMINAL_MARKS = (".", "!", "?", "”")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_redpajama(text: str) -> str:
    """Mirror the public RedPajama normalization sequence."""
    text = text.translate(ASCII_PUNCTUATION_TABLE)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return unicodedata.normalize("NFD", text)


def split_redpajama_lines(text: str) -> List[str]:
    """Mirror RedPajama split_paragraphs(..., remove_empty=False)."""
    return [match.group(1) for match in LINE_PATTERN.finditer(text)]


def ngram_fraction_percent(words: Sequence[str], n: int) -> float:
    if not words:
        return 0.0
    grams = (tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))
    top = Counter(grams).most_common(1)
    if not top:
        return 0.0
    gram, count = top[0]
    if count <= 1:
        return 0.0
    total_chars = sum(map(len, words))
    if total_chars == 0:
        return 0.0
    # The official top-ngram formula counts overlapping windows repeatedly.
    score = sum(map(len, gram)) * count / total_chars
    return 100.0 * round(score, UPSTREAM_PRECISION)


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(statistics.fmean(values)) if values else None


def pooled_fraction_or_none(numerators: Sequence[int], denominators: Sequence[int]) -> Optional[float]:
    denominator = sum(denominators)
    return 100.0 * sum(numerators) / denominator if denominator else None


def recalculate_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    text = record.get("content")
    if not isinstance(text, str):
        raise ValueError(f"Record {record.get('id')!r} has no string content")

    normalized_words = normalize_redpajama(text).split()
    raw_words = WORDPUNCT_PATTERN.findall(text)

    out: Dict[str, Any] = {
        "sample_id": str(record.get("id", "")),
        "content_chars": len(text),
        "raw_line_count": 0,
        "raw_nonempty_line_count": 0,
        "normalized_nonempty_line_count": 0,
    }

    n_words = len(normalized_words)
    out["calc_rps_doc_frac_unique_words_pct"] = (
        100.0 * round(len(set(normalized_words)) / n_words, UPSTREAM_PRECISION)
        if n_words
        else None
    )
    out["calc_rps_doc_frac_no_alph_words_pct"] = (
        100.0
        * round(
            1.0
            - sum(ALPHABET_PATTERN.search(word) is not None for word in raw_words)
            / len(raw_words),
            UPSTREAM_PRECISION,
        )
        if raw_words
        else None
    )
    out["calc_rps_doc_frac_chars_top_2gram_pct"] = ngram_fraction_percent(normalized_words, 2)
    out["calc_rps_doc_frac_chars_top_3gram_pct"] = ngram_fraction_percent(normalized_words, 3)

    raw_lines = split_redpajama_lines(text)
    normalized_lines = [normalize_redpajama(line) for line in raw_lines]
    out["raw_line_count"] = len(raw_lines)
    out["raw_nonempty_line_count"] = sum(bool(line) for line in raw_lines)
    out["normalized_nonempty_line_count"] = sum(bool(line) for line in normalized_lines)

    uppercase_num: List[int] = []
    uppercase_den: List[int] = []
    terminal_values: List[float] = []
    numerical_num: List[int] = []
    numerical_den: List[int] = []

    for raw_line, normalized_line in zip(raw_lines, normalized_lines):
        uppercase_num.append(round(sum(ch.isupper() for ch in raw_line) / len(raw_line), UPSTREAM_PRECISION) if raw_line else 0.0)
        uppercase_den.append(len(raw_line))
        terminal_values.append(float(raw_line.rstrip().endswith(TERMINAL_MARKS)))
        numerical_num.append(round(sum(ch.isnumeric() for ch in normalized_line) / len(normalized_line), UPSTREAM_PRECISION) if normalized_line else 0.0)
        numerical_den.append(len(normalized_line))

    out["calc_uppercase_mean_all_lines_pct"] = mean_or_none(
        [100.0 * n for n in uppercase_num]
    )
    out["calc_uppercase_mean_nonempty_lines_pct"] = mean_or_none(
        [100.0 * n for n, d in zip(uppercase_num, uppercase_den) if d]
    )
    out["calc_uppercase_char_weighted_pct"] = pooled_fraction_or_none(
        [round(n * d) for n, d in zip(uppercase_num, uppercase_den)], uppercase_den
    )

    terminal_mean_all = mean_or_none(terminal_values)
    terminal_mean_nonempty = mean_or_none(
        [value for value, line in zip(terminal_values, raw_lines) if line.strip()]
    )
    out["calc_terminal_mean_all_lines_pct"] = (
        100.0 * terminal_mean_all if terminal_mean_all is not None else None
    )
    out["calc_terminal_mean_nonempty_lines_pct"] = (
        100.0 * terminal_mean_nonempty if terminal_mean_nonempty is not None else None
    )

    out["calc_numerical_mean_all_lines_pct"] = mean_or_none(
        [100.0 * n for n in numerical_num]
    )
    out["calc_numerical_mean_nonempty_lines_pct"] = mean_or_none(
        [100.0 * n for n, d in zip(numerical_num, numerical_den) if d]
    )
    out["calc_numerical_char_weighted_pct"] = pooled_fraction_or_none(
        [round(n * d) for n, d in zip(numerical_num, numerical_den)], numerical_den
    )

    for field in FIELDS:
        value = record.get(field)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")
        out[f"stored_{field}"] = numeric_value

    return out


def candidates_for_field(field: str) -> List[Tuple[str, str]]:
    if field == "rps_doc_frac_unique_words":
        return [("public_document_formula_x100", "calc_rps_doc_frac_unique_words_pct")]
    if field == "rps_doc_frac_no_alph_words":
        return [("public_document_formula_x100", "calc_rps_doc_frac_no_alph_words_pct")]
    if field == "rps_doc_frac_chars_top_2gram":
        return [("public_top_ngram_formula_x100", "calc_rps_doc_frac_chars_top_2gram_pct")]
    if field == "rps_doc_frac_chars_top_3gram":
        return [("public_top_ngram_formula_x100", "calc_rps_doc_frac_chars_top_3gram_pct")]
    if field == "rps_lines_uppercase_letter_fraction":
        return [
            ("mean_all_lines", "calc_uppercase_mean_all_lines_pct"),
            ("mean_nonempty_lines", "calc_uppercase_mean_nonempty_lines_pct"),
            ("character_weighted_all_lines", "calc_uppercase_char_weighted_pct"),
        ]
    if field == "rps_lines_ending_with_terminal_punctution_mark":
        return [
            ("mean_all_lines", "calc_terminal_mean_all_lines_pct"),
            ("mean_nonempty_lines", "calc_terminal_mean_nonempty_lines_pct"),
        ]
    if field == "rps_lines_numerical_chars_fraction":
        return [
            ("mean_all_lines", "calc_numerical_mean_all_lines_pct"),
            ("mean_nonempty_lines", "calc_numerical_mean_nonempty_lines_pct"),
            ("character_weighted_normalized_lines", "calc_numerical_char_weighted_pct"),
        ]
    raise KeyError(field)


def compare_values(stored: float, calculated: Optional[float]) -> Optional[float]:
    if calculated is None or not math.isfinite(stored) or not math.isfinite(calculated):
        return None
    return abs(stored - calculated)


def build_summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for field in FIELDS:
        stored_vals = [float(row[f"stored_{field}"]) for row in rows if math.isfinite(float(row[f"stored_{field}"]))]
        if stored_vals:
            stored_min, stored_max = min(stored_vals), max(stored_vals)
            stored_gt1 = sum(value > 1.0 for value in stored_vals)
            stored_gt100 = sum(value > 100.0 for value in stored_vals)
        else:
            stored_min = stored_max = float("nan")
            stored_gt1 = stored_gt100 = 0

        for candidate_name, calc_column in candidates_for_field(field):
            pairs = [
                (float(row[f"stored_{field}"]), row.get(calc_column))
                for row in rows
                if math.isfinite(float(row[f"stored_{field}"])) and row.get(calc_column) is not None
            ]
            errors = [abs(stored - float(calc)) for stored, calc in pairs]
            calculated_values = [float(calc) for _, calc in pairs]
            result.append(
                {
                    "field": field,
                    "candidate_aggregation": candidate_name,
                    "n_compared": len(pairs),
                    "n_missing_or_invalid_stored": len(rows) - len(stored_vals),
                    "stored_min": stored_min,
                    "stored_max": stored_max,
                    "stored_n_gt_1": stored_gt1,
                    "stored_n_gt_100": stored_gt100,
                    "calculated_min": min(calculated_values) if calculated_values else float("nan"),
                    "calculated_max": max(calculated_values) if calculated_values else float("nan"),
                    "match_tolerance_percentage_points": ABS_TOLERANCE,
                    "n_within_tolerance": sum(error <= ABS_TOLERANCE for error in errors),
                    "fraction_within_tolerance": (
                        sum(error <= ABS_TOLERANCE for error in errors) / len(errors) if errors else float("nan")
                    ),
                    "mean_absolute_error": statistics.fmean(errors) if errors else float("nan"),
                    "median_absolute_error": statistics.median(errors) if errors else float("nan"),
                    "max_absolute_error": max(errors) if errors else float("nan"),
                }
            )
    return result


def csv_write(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def csv_gzip_write(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}g}"


def write_summary_markdown(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    input_sha256: str,
    expected_sha256: str,
    input_rows: int,
    full_run: bool,
    command: str,
) -> None:
    lines = [
        "# Q1.1 七个 frac 字段独立复算审计",
        "",
        f"- 运行类型：{'全量 A1' if full_run else '最小试运行（非最终结论）'}",
        f"- 输入行数：{len(rows):,} / {EXPECTED_A1_ROWS:,}（manifest 预期共 {input_rows:,}）",
        f"- A1 原始输入 SHA-256：`{input_sha256}`",
        f"- 预处理 manifest 记录 SHA-256：`{expected_sha256}`（{'匹配' if input_sha256 == expected_sha256 else '不匹配'}）",
        f"- 比较容差：±{ABS_TOLERANCE} 个百分点",
        "- 所有计算只读取原始文本和指标值；不修改 Q1 冻结分数与历史输出。",
        "",
        "## 复算口径",
        "",
        "文档级指标按公开 RedPajama 实现从 content 独立计算，再乘 100 与存储值比较。行级指标先依公开代码逐行生成值；OpenDataLab 数据卡未给出列表折叠为文档标量的具体公式，因此同时报告明确列出的候选折叠方式，不按误差大小替数据集挑选 winner。公开实现来自 `main` 动态分支，不能证明它就是 OpenDataLab 生成这批数据时使用的提交版本。",
        "",
        "## 字段结果",
        "",
        "| 字段 | 文档/聚合候选 | 存储范围 | 存储值>1 | 存储值>100 | 容差内匹配 | 平均绝对误差 | 最大绝对误差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    stored_meta: Dict[str, Mapping[str, Any]] = {}
    for row in summary_rows:
        stored_meta.setdefault(str(row["field"]), row)
        lines.append(
            "| `{field}` | {candidate} | {lo}–{hi} | {gt1} | {gt100} | {matches}/{n} ({fraction}) | {mae} | {maxae} |".format(
                field=row["field"],
                candidate=row["candidate_aggregation"],
                lo=fmt_number(row["stored_min"]),
                hi=fmt_number(row["stored_max"]),
                gt1=row["stored_n_gt_1"],
                gt100=row["stored_n_gt_100"],
                matches=row["n_within_tolerance"],
                n=row["n_compared"],
                fraction=fmt_number(row["fraction_within_tolerance"], 4),
                mae=fmt_number(row["mean_absolute_error"]),
                maxae=fmt_number(row["max_absolute_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 匹配表示当前公开公式在本次 A1 文本上可复现存储值；不匹配只表示当前公开公式/明确候选聚合未复现，不能单独证明数据损坏。",
            "- 大于 1 是百分数尺度线索；top n-gram 计数允许重叠窗口重复累计，故大于 100 本身不构成错误证据。",
            "- 若行级字段在所有列出的聚合方式下都无法复现，结论应保留“OpenDataLab 生成版本/文档聚合尚未追溯”，不能将任一候选聚合说成已确认原公式。",
            "- 该审计不决定候选 Q，也不改变 `q_huber` 的操作性冻结。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            command,
            "```",
            "",
            "## 参考实现与数据卡",
            "",
        ]
    )
    for key, url in UPSTREAM_URLS.items():
        lines.append(f"- {key}: {url}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_expected_input_hash() -> Tuple[str, int]:
    manifest = json.loads(PREPROCESS_MANIFEST.read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("input_files", []) if item.get("dataset") == "a1"]
    if len(matches) != 1:
        raise ValueError("Could not identify exactly one A1 input in the frozen preprocessing manifest")
    return str(matches[0]["sha256"]), int(matches[0]["size"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N rows; for P1 smoke-check only")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir.resolve()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_sha, expected_size = load_expected_input_hash()
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)
    actual_size = INPUT_PATH.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"A1 input size mismatch: expected {expected_size}, got {actual_size}")
    actual_sha = sha256_file(INPUT_PATH)
    if actual_sha != expected_sha:
        raise ValueError(f"A1 input SHA-256 mismatch: expected {expected_sha}, got {actual_sha}")

    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    total_seen = 0
    with lzma.open(INPUT_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total_seen += 1
            audit_row = recalculate_record(record)
            sample_id = audit_row["sample_id"]
            if not sample_id or sample_id in seen_ids:
                raise ValueError(f"Missing or duplicate A1 id at record {total_seen}: {sample_id!r}")
            seen_ids.add(sample_id)
            rows.append(audit_row)
            if total_seen % args.progress_every == 0:
                print(f"processed {total_seen:,} A1 rows", flush=True)
            if args.limit is not None and total_seen >= args.limit:
                break

    if not rows:
        raise ValueError("No A1 rows were recalculated")

    summary_rows = build_summary(rows)
    summary_columns = list(summary_rows[0].keys())
    csv_write(output_dir / "field_recalculation_metrics.csv", summary_rows, summary_columns)

    detail_columns = list(rows[0].keys())
    csv_gzip_write(output_dir / "a1_record_recalculation.csv.gz", rows, detail_columns)

    command = (
        "python src/q1_1_frac_field_recalculation_audit.py "
        f"--output-dir {args.output_dir.as_posix()}"
    )
    if args.limit is not None:
        command += f" --limit {args.limit}"
    write_summary_markdown(
        output_dir / "summary.md",
        rows=rows,
        summary_rows=summary_rows,
        input_sha256=actual_sha,
        expected_sha256=expected_sha,
        input_rows=EXPECTED_A1_ROWS,
        full_run=args.limit is None and total_seen == EXPECTED_A1_ROWS,
        command=command,
    )

    output_files = [
        "field_recalculation_metrics.csv",
        "a1_record_recalculation.csv.gz",
        "summary.md",
    ]
    output_hashes = {name: sha256_file(output_dir / name) for name in output_files}
    manifest = {
        "schema_version": "q1-1-frac-recalculation-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_a1_complete" if args.limit is None and total_seen == EXPECTED_A1_ROWS else "partial_smoke_run",
        "rows_recalculated": len(rows),
        "expected_a1_rows": EXPECTED_A1_ROWS,
        "unique_sample_ids": len(seen_ids),
        "input": {
            "path": str(INPUT_PATH.relative_to(ROOT)),
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "matches_frozen_preprocessing_manifest": actual_sha == expected_sha,
            "preprocessing_manifest": str(PREPROCESS_MANIFEST.relative_to(ROOT)),
        },
        "script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "method": {
            "public_reference_branch": "togethercomputer/RedPajama-Data main (dynamic branch; source generation commit for OpenDataLab not identified)",
            "retrieved_date_local": datetime.now().astimezone().date().isoformat(),
            "upstream_precision_digits": UPSTREAM_PRECISION,
            "unit_comparison": "stored values compared with independently calculated ratios multiplied by 100",
            "line_aggregation": "not specified by OpenDataLab card; arithmetic all-line, arithmetic nonempty-line, and character-weighted candidates reported separately",
            "absolute_tolerance_percentage_points": ABS_TOLERANCE,
            "random_seed": None,
            "randomness": "none",
            "source_urls": UPSTREAM_URLS,
        },
        "command": command,
        "output_sha256": output_hashes,
        "frozen_q1_outputs_modified": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"status={manifest['status']}")
    print(f"rows={len(rows)}; unique_ids={len(seen_ids)}")
    print(f"input_sha256={actual_sha}; manifest_match={actual_sha == expected_sha}")
    print(f"output_dir={output_dir}")
    print(f"script_sha256={manifest['script']['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # report deterministic audit failure with a nonzero status
        print(f"AUDIT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
