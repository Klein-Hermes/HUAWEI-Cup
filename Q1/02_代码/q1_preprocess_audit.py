"""Q1 A1-A3 公共预处理的审计、验收与冻结辅助脚本。

本脚本不重新拟合标准化参数，也不改变预处理结果。它生成：

* data_overview.csv
* quality_indicator_dictionary.csv
* duplicate_id_report.csv
* preprocess_qc_report.csv

原始 content 只用于重复记录的哈希比较，不进入质量指标，也不执行其文本内容。
运行前应由调用命令同时加载题目分析报告，遵守其中的数据污染隔离规则。
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
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from f_q1_common_preprocess import (
    FIELD_SPECS,
    NONNEGATIVE_FIELDS,
    UNIT_SUSPECT_FIELDS,
    scalarize_record,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_ROOT = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"


INDICATOR_DESCRIPTIONS: Dict[str, str] = {
    "fineweb_edu": "FineWeb 教育质量分数",
    "fluency_en": "英文流畅概率",
    "modernbert_cleanliness": "ModernBERT 清洁度等级期望",
    "modernbert_readability": "ModernBERT 可读性等级期望",
    "modernbert_reasoning": "ModernBERT 推理性等级期望",
    "modernbert_professionalism": "ModernBERT 专业性等级期望",
    "dsir_books": "DSIR 书籍域信号",
    "dsir_wiki": "DSIR Wikipedia 域信号",
    "dsir_math": "DSIR 数学域信号",
    "qurater": "Qurater 列表均值",
    "ad_en": "英文无广告概率",
    "rps_doc_word_count": "文档词数",
    "rps_doc_num_sentences": "文档句数",
    "rps_doc_unigram_entropy": "词一元熵",
    "rps_doc_frac_unique_words": "唯一词比例/分数",
    "rps_doc_frac_no_alph_words": "非字母词比例/分数",
    "rps_doc_frac_chars_top_2gram": "高频二元字符比例/分数",
    "rps_doc_frac_chars_top_3gram": "高频三元字符比例/分数",
    "rps_lines_uppercase_letter_fraction": "大写字母比例/分数",
    "rps_lines_ending_with_terminal_punctution_mark": "句末终止标点比例",
    "rps_lines_numerical_chars_fraction": "数字字符比例/分数",
    "rps_doc_mean_word_length": "平均词长",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and freeze Q1 A1-A3 common preprocessing outputs.")
    parser.add_argument("--a-root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def input_paths(a_root: Path) -> Dict[str, List[Path]]:
    paths = {
        "a1": [a_root / "slimpajama_quality_signal_sample.jsonl.xz"],
        "a2": sorted((a_root / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz")),
        "a3": sorted((a_root / "slimpajama_quality_extended").glob("github_*.jsonl.xz")),
    }
    if not paths["a1"][0].exists() or not paths["a2"] or not paths["a3"]:
        raise FileNotFoundError("A1/A2/A3 input files are incomplete")
    return paths


def open_text(path: Path):
    return lzma.open(path, "rt", encoding="utf-8") if path.suffix == ".xz" else path.open("rt", encoding="utf-8")


def source_domain(record: Mapping[str, Any], dataset: str, path: Path) -> str:
    if dataset == "a1":
        return str(record.get("_source_domain") or "unknown")
    match = re.match(r"(arxiv|github)_", path.name)
    return match.group(1) if match else "unknown"


def source_corpus(record: Mapping[str, Any]) -> str:
    for key in ("corpus", "_corpus", "source_corpus"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def content_hash(record: Mapping[str, Any]) -> str:
    value = record.get("content")
    if value is None:
        return "<missing-content>"
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def metric_signature(record: Mapping[str, Any]) -> str:
    scalars, _ = scalarize_record(record)
    clean = {
        field: (None if not math.isfinite(value) else float(value))
        for field, value in scalars.items()
    }
    return json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def scan_raw_dataset(dataset: str, paths: Sequence[Path]) -> Dict[str, Any]:
    raw_columns = set()
    ids: Dict[str, Dict[str, Any]] = {}
    rows = 0
    empty_ids = 0
    domains = set()
    corpora = set()

    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                rows += 1
                raw_columns.update(record.keys())
                sample_id = str(record.get("id") or "")
                domain = source_domain(record, dataset, path)
                corpus = source_corpus(record)
                domains.add(domain)
                if corpus:
                    corpora.add(corpus)
                empty_ids += int(sample_id == "")

                meta = ids.setdefault(
                    sample_id,
                    {
                        "count": 0,
                        "content_hashes": set(),
                        "metric_signatures": set(),
                        "domains": set(),
                        "corpora": set(),
                    },
                )
                meta["count"] += 1
                meta["content_hashes"].add(content_hash(record))
                meta["metric_signatures"].add(metric_signature(record))
                meta["domains"].add(domain)
                if corpus:
                    meta["corpora"].add(corpus)

    duplicate_groups = {sample_id: meta for sample_id, meta in ids.items() if meta["count"] > 1}
    return {
        "rows": rows,
        "raw_columns": raw_columns,
        "ids": ids,
        "duplicate_groups": duplicate_groups,
        "empty_ids": empty_ids,
        "domains": domains,
        "corpora": corpora,
    }


def scan_processed_dataset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    qc: Dict[str, Dict[str, Any]] = {
        field: {"n": 0, "missing": 0, "invalid": 0, "unit_suspect": 0, "outlier": 0, "norm": []}
        for field in FIELD_SPECS
    }
    rows = 0
    header: List[str] = []
    nonfinite_norm = 0
    out_of_range_norm = 0
    unexplained_blank_norm = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        for row in reader:
            rows += 1
            for field in FIELD_SPECS:
                item = qc[field]
                item["n"] += 1
                item["missing"] += int(row.get(f"missing_{field}") or 0)
                item["invalid"] += int(row.get(f"invalid_{field}") or 0)
                item["unit_suspect"] += int(row.get(f"unit_suspect_{field}") or 0)
                item["outlier"] += int(row.get(f"outlier_{field}") or 0)
                value = row.get(f"norm_{field}", "")
                if value in (None, ""):
                    if int(row.get(f"missing_{field}") or 0) == 0 and int(row.get(f"invalid_{field}") or 0) == 0:
                        unexplained_blank_norm += 1
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    nonfinite_norm += 1
                    continue
                if numeric < 0.0 or numeric > 1.0:
                    out_of_range_norm += 1
                item["norm"].append(numeric)
    return {
        "rows": rows,
        "columns": header,
        "qc": qc,
        "sha256": sha256_file(path),
        "nonfinite_norm": nonfinite_norm,
        "out_of_range_norm": out_of_range_norm,
        "unexplained_blank_norm": unexplained_blank_norm,
    }


def duplicate_type(sample_id: str, meta: Mapping[str, Any]) -> str:
    if sample_id == "":
        return "empty_id"
    same_content = len(meta["content_hashes"]) == 1
    same_metrics = len(meta["metric_signatures"]) == 1
    scoped = len(meta["domains"]) > 1 or len(meta["corpora"]) > 1
    if same_content and same_metrics:
        return "exact_duplicate_scoped" if scoped else "exact_duplicate"
    if scoped:
        return "scoped_id_collision"
    if same_content:
        return "same_content_metric_conflict"
    if same_metrics:
        return "same_metrics_content_conflict"
    return "id_collision"


def duplicate_action(kind: str) -> str:
    if kind == "exact_duplicate":
        return "retain_all; optional_dedup_sensitivity"
    if kind == "exact_duplicate_scoped":
        return "retain_all; use_composite_key"
    if kind == "empty_id":
        return "retain_all; assign_surrogate_key_later"
    if kind == "scoped_id_collision":
        return "retain_all; use_composite_key"
    return "retain_all; block_unique_id_assumption"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_indicator_dictionary() -> List[Dict[str, Any]]:
    rows = []
    scalar_rules = {
        "single": "finite scalar",
        "single_list": "length=1, take the unique element",
        "prob_index_1": "stable softmax, take class index 1",
        "ordinal_logits": "stable softmax, expected level / 5",
        "mean_list": "mean of all finite list elements",
    }
    for indicator, spec in FIELD_SPECS.items():
        direction = spec["direction"]
        normalization = (
            "A1 clipped ECDF 0.5%-99.5%"
            if direction in {"positive", "negative"}
            else "A1 robust centrality exp(-abs(x-median)/scale)"
        )
        valid_domain = "finite and >=0" if indicator in NONNEGATIVE_FIELDS else "finite"
        rows.append(
            {
                "indicator": indicator,
                "meaning": INDICATOR_DESCRIPTIONS.get(indicator, ""),
                "raw_type": spec["kind"],
                "scalar_rule": scalar_rules[spec["kind"]],
                "transform": spec["transform"],
                "direction": direction,
                "normalization": normalization,
                "valid_domain": valid_domain,
                "unit_suspect_rule": "flag if value < 0 or value > 100; do not auto-convert"
                if indicator in UNIT_SUSPECT_FIELDS
                else "not applicable",
                "missing_policy": "leave norm blank and preserve missing flag",
                "outlier_policy": "robust z-score > 3.5; retain value and flag; robust aggregation later",
            }
        )
    return rows


def build_qc_rows(dataset: str, processed: Mapping[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for indicator, item in processed["qc"].items():
        n = int(item["n"])
        missing = int(item["missing"])
        invalid = int(item["invalid"])
        unit_suspect = int(item["unit_suspect"])
        outlier = int(item["outlier"])
        values = np.asarray(item["norm"], dtype=float)
        finite_values = values[np.isfinite(values)]
        result.append(
            {
                "dataset": dataset,
                "indicator": indicator,
                "n": n,
                "valid_n": n - missing - invalid,
                "valid_rate": (n - missing - invalid) / n if n else "",
                "missing_n": missing,
                "missing_rate": missing / n if n else "",
                "invalid_n": invalid,
                "invalid_rate": invalid / n if n else "",
                "unit_suspect_n": unit_suspect,
                "unit_suspect_rate": unit_suspect / n if n else "",
                "outlier_n": outlier,
                "outlier_rate": outlier / n if n else "",
                "norm_valid_n": int(finite_values.size),
                "norm_zero_n": int(np.sum(finite_values == 0.0)),
                "norm_zero_rate": float(np.mean(finite_values == 0.0)) if finite_values.size else "",
                "norm_one_n": int(np.sum(finite_values == 1.0)),
                "norm_one_rate": float(np.mean(finite_values == 1.0)) if finite_values.size else "",
                "norm_mean": float(np.mean(finite_values)) if finite_values.size else "",
                "norm_median": float(np.median(finite_values)) if finite_values.size else "",
                "norm_min": float(np.min(finite_values)) if finite_values.size else "",
                "norm_max": float(np.max(finite_values)) if finite_values.size else "",
                "norm_nonfinite_n": processed["nonfinite_norm"],
                "norm_out_of_range_n": processed["out_of_range_norm"],
                "norm_unexplained_blank_n": processed["unexplained_blank_norm"],
            }
        )
    return result


def main() -> int:
    args = parse_args()
    paths = input_paths(args.a_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_info: Dict[str, Dict[str, Any]] = {}
    processed_info: Dict[str, Dict[str, Any]] = {}
    for dataset in ("a1", "a2", "a3"):
        raw_info[dataset] = scan_raw_dataset(dataset, paths[dataset])
        processed_info[dataset] = scan_processed_dataset(args.output_dir / f"{dataset}_preprocessed.csv.gz")

    overview_rows = []
    duplicate_rows = []
    qc_rows = []
    blockers = []
    for dataset in ("a1", "a2", "a3"):
        raw = raw_info[dataset]
        processed = processed_info[dataset]
        duplicate_groups = raw["duplicate_groups"]
        overview_rows.append(
            {
                "dataset": dataset.upper(),
                "n_rows_raw": raw["rows"],
                "n_rows_processed": processed["rows"],
                "row_delta": processed["rows"] - raw["rows"],
                "n_columns_raw": len(raw["raw_columns"]),
                "n_columns_processed": len(processed["columns"]),
                "n_quality_indicators": len(FIELD_SPECS),
                "duplicate_id_count": len(duplicate_groups),
                "duplicate_row_count": sum(meta["count"] - 1 for meta in duplicate_groups.values()),
                "empty_id_count": raw["empty_ids"],
                "domain_count": len(raw["domains"]),
                "corpus_count": len(raw["corpora"]),
                "input_files": "|".join(str(path) for path in paths[dataset]),
                "input_sha256": "|".join(sha256_file(path) for path in paths[dataset]),
                "processed_file": str(args.output_dir / f"{dataset}_preprocessed.csv.gz"),
                "processed_sha256": processed["sha256"],
            }
        )
        for sample_id, meta in sorted(duplicate_groups.items()):
            kind = duplicate_type(sample_id, meta)
            action = duplicate_action(kind)
            if kind in {"id_collision", "same_content_metric_conflict", "same_metrics_content_conflict", "scoped_id_collision"}:
                blockers.append(f"{dataset}:{sample_id}:{kind}")
            duplicate_rows.append(
                {
                    "dataset": dataset.upper(),
                    "sample_id": sample_id,
                    "duplicate_count": meta["count"],
                    "same_content": len(meta["content_hashes"]) == 1,
                    "same_metrics": len(meta["metric_signatures"]) == 1,
                    "domain_count": len(meta["domains"]),
                    "corpus_count": len(meta["corpora"]),
                    "duplicate_type": kind,
                    "action": action,
                }
            )
        qc_rows.extend(build_qc_rows(dataset.upper(), processed))

    write_csv(
        args.output_dir / "data_overview.csv",
        [
            "dataset", "n_rows_raw", "n_rows_processed", "row_delta", "n_columns_raw",
            "n_columns_processed", "n_quality_indicators", "duplicate_id_count",
            "duplicate_row_count", "empty_id_count", "domain_count", "corpus_count",
            "input_files", "input_sha256", "processed_file", "processed_sha256",
        ],
        overview_rows,
    )
    write_csv(
        args.output_dir / "quality_indicator_dictionary.csv",
        [
            "indicator", "meaning", "raw_type", "scalar_rule", "transform", "direction",
            "normalization", "valid_domain", "unit_suspect_rule", "missing_policy", "outlier_policy",
        ],
        build_indicator_dictionary(),
    )
    write_csv(
        args.output_dir / "duplicate_id_report.csv",
        [
            "dataset", "sample_id", "duplicate_count", "same_content", "same_metrics",
            "domain_count", "corpus_count", "duplicate_type", "action",
        ],
        duplicate_rows,
    )
    write_csv(
        args.output_dir / "preprocess_qc_report.csv",
        [
            "dataset", "indicator", "n", "valid_n", "valid_rate", "missing_n", "missing_rate",
            "invalid_n", "invalid_rate", "unit_suspect_n", "unit_suspect_rate", "outlier_n",
            "outlier_rate", "norm_valid_n", "norm_zero_n", "norm_zero_rate", "norm_one_n",
            "norm_one_rate", "norm_mean", "norm_median", "norm_min", "norm_max",
            "norm_nonfinite_n", "norm_out_of_range_n", "norm_unexplained_blank_n",
        ],
        qc_rows,
    )

    all_norm_values = [row for row in qc_rows if row["norm_max"] != ""]
    norm_range_ok = all(0.0 <= float(row["norm_min"]) <= float(row["norm_max"]) <= 1.0 for row in all_norm_values)
    rows_preserved = all(row["row_delta"] == 0 for row in overview_rows)
    dictionary_ok = len(build_indicator_dictionary()) == 22
    qc_pass = bool(norm_range_ok and rows_preserved and dictionary_ok)
    qc_pass = qc_pass and all(
        processed_info[dataset]["nonfinite_norm"] == 0
        and processed_info[dataset]["out_of_range_norm"] == 0
        and processed_info[dataset]["unexplained_blank_norm"] == 0
        for dataset in ("a1", "a2", "a3")
    )

    manifest_path = args.output_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "preprocessing_version": "q1-common-v1.1",
            "status": "frozen" if qc_pass and not blockers else "audit_blocked",
            "freeze_blockers": blockers,
            "audit_checks": {
                **manifest.get("audit_checks", {}),
                "dictionary_22_of_22": dictionary_ok,
                "rows_preserved": rows_preserved,
                "norm_range_0_1": norm_range_ok,
                "norm_nonfinite_zero": all(processed_info[dataset]["nonfinite_norm"] == 0 for dataset in ("a1", "a2", "a3")),
                "norm_out_of_range_zero": all(processed_info[dataset]["out_of_range_norm"] == 0 for dataset in ("a1", "a2", "a3")),
                "norm_blank_explained": all(processed_info[dataset]["unexplained_blank_norm"] == 0 for dataset in ("a1", "a2", "a3")),
                "raw_content_used_as_metric": False,
                "a2_a3_refit_calibration": False,
            },
            "audit_artifacts": sorted(set(manifest.get("audit_artifacts", [])) | {
                "data_overview.csv",
                "quality_indicator_dictionary.csv",
                "duplicate_id_report.csv",
                "preprocess_qc_report.csv",
            }),
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"], "freeze_blockers": blockers, "audit_checks": manifest["audit_checks"]}, ensure_ascii=False, indent=2))
    return 0 if qc_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
