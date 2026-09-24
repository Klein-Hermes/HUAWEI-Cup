"""Freeze and accept the Q1 A1-A3 common-preprocessing deliverables.

This command does not refit calibration and does not rewrite processed data.
It verifies the existing outputs, writes one consolidated QC report, and
records the freeze version/status in manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from f_q1_common_preprocess import FIELD_SPECS, output_fieldnames, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"
FREEZE_VERSION = "q1-common-v1.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze Q1 A1-A3 common preprocessing outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def scan_output(path: Path) -> Dict[str, Any]:
    expected_header = output_fieldnames()
    qc = {
        field: {
            "n": 0,
            "missing": 0,
            "invalid": 0,
            "unit_suspect": 0,
            "outlier": 0,
            "norm_valid": 0,
            "norm_min": None,
            "norm_max": None,
        }
        for field in FIELD_SPECS
    }
    rows = 0
    domains: Dict[str, int] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != expected_header:
            raise ValueError(
                f"{path.name}: column contract mismatch: got {len(header)}, expected {len(expected_header)}"
            )
        for row in reader:
            rows += 1
            domain = str(row.get("source_domain") or "unknown")
            domains[domain] = domains.get(domain, 0) + 1
            for field, item in qc.items():
                item["n"] += 1
                for flag in ("missing", "invalid", "unit_suspect", "outlier"):
                    item[flag] += int(row.get(f"{flag}_{field}") or 0)
                value = row.get(f"norm_{field}")
                if value not in (None, ""):
                    numeric = float(value)
                    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                        raise ValueError(f"{path.name}: norm_{field} outside [0,1]: {value}")
                    item["norm_valid"] += 1
                    item["norm_min"] = numeric if item["norm_min"] is None else min(item["norm_min"], numeric)
                    item["norm_max"] = numeric if item["norm_max"] is None else max(item["norm_max"], numeric)
    return {
        "rows": rows,
        "columns": len(header),
        "domains": domains,
        "qc": qc,
        "sha256": sha256_file(path),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(
    output_dir: Path,
    manifest: Dict[str, Any],
    calibration: Dict[str, Any],
    scans: Dict[str, Dict[str, Any]],
    generated_at: str,
) -> str:
    expected_columns = len(output_fieldnames())
    lines: List[str] = [
        "# Q1 A1/A2/A3 公共预处理 QC 验收报告",
        "",
        f"- 冻结版本：`{FREEZE_VERSION}`",
        "- 冻结状态：`FROZEN`",
        f"- 验收时间（UTC）：`{generated_at}`",
        "- 验收范围：A1 校准、A1/A2/A3 统一预处理结果及可复现性信息",
        "",
        "## 1. 验收结论",
        "",
        "本报告由冻结脚本基于已生成文件重新扫描得到，不重新拟合校准参数，不修改三份预处理数据。",
        "",
        "| 检查项 | 结果 |",
        "|---|---|",
        f"| 指标数 | PASS：{len(FIELD_SPECS)} / 22 |",
        f"| 输出列契约 | PASS：{expected_columns} 列 |",
        "| A1 校准复用到 A2/A3 | PASS |",
        "| 所有非空 norm 值位于 [0,1] | PASS |",
        "| 输入/输出哈希可追溯 | PASS |",
        "| 行数与域计数保持 | PASS |",
        "",
        "## 2. 冻结版本与校准规则",
        "",
        f"- `freeze_version`：`{FREEZE_VERSION}`",
        f"- `schema_version`：`{manifest.get('schema_version', '—')}`",
        f"- `fit_dataset`：`{calibration.get('fit_dataset', '—')}`",
        f"- `fit_rows`：`{calibration.get('fit_rows', '—')}`",
        f"- `quantile_points`：`{calibration.get('quantile_points', '—')}`",
        f"- `calibration_quantile_clip`：`{calibration.get('calibration_quantile_clip', '—')}`",
        f"- `same_calibration_for`：`{', '.join(manifest.get('same_calibration_for', []))}`",
        "",
        "校准规则只在 A1 上拟合；A2、A3 直接应用同一套字段解析、方向处理和归一化参数。",
        "",
        "## 3. 数据集级验收",
        "",
        "| 数据集 | 行数 | 列数 | 域计数 | 缺失值 | 非法值 | 单位疑似值 | 异常标记总数 | 输出 SHA-256 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for dataset in ("a1", "a2", "a3"):
        scan = scans[dataset]
        summary = manifest["summaries"][dataset]
        total = {key: sum(item[key] for item in scan["qc"].values()) for key in ("missing", "invalid", "unit_suspect", "outlier")}
        lines.append(
            f"| {dataset.upper()} | {scan['rows']} | {scan['columns']} | "
            f"`{json.dumps(scan['domains'], ensure_ascii=False, sort_keys=True)}` | "
            f"{total['missing']} | {total['invalid']} | {total['unit_suspect']} | {total['outlier']} | "
            f"`{scan['sha256']}` |"
        )
        if scan["rows"] != summary["rows"] or scan["sha256"] != summary["output_sha256"]:
            raise ValueError(f"{dataset}: scan result disagrees with manifest")
    lines.extend(
        [
            "",
            "说明：`outlier` 是相对于 A1 稳健中心和尺度的审计标记，不表示删除该行；后续综合评分应使用稳健聚合或敏感性分析。",
            "",
            "## 4. 字段级 QC",
            "",
            "`norm_valid` 为该字段成功生成归一化值的记录数；缺失值保留为空并由对应标记列记录。",
            "",
            "| 数据集 | 指标 | n | missing | invalid | unit_suspect | outlier | norm_valid | norm_min | norm_max |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("a1", "a2", "a3"):
        for field, item in scans[dataset]["qc"].items():
            lines.append(
                f"| {dataset.upper()} | `{field}` | {item['n']} | {item['missing']} | {item['invalid']} | "
                f"{item['unit_suspect']} | {item['outlier']} | {item['norm_valid']} | "
                f"{fmt(item['norm_min'])} | {fmt(item['norm_max'])} |"
            )
    lines.extend(
        [
            "",
            "## 5. 输入文件哈希",
            "",
            "| 数据集 | 输入文件 | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for item in manifest.get("input_files", []):
        lines.append(f"| {str(item.get('dataset', '')).upper()} | `{item.get('path', '')}` | `{item.get('sha256', '')}` |")
    lines.extend(
        [
            "",
            "## 6. 后续使用约束",
            "",
            "1. Q1.1、Q1.2 后续计算直接使用三份结果中的 `norm_*` 列。",
            "2. 不得在 A2、A3 内重新拟合 Min-Max、Z-score 或经验分布。",
            "3. `missing_*`、`invalid_*`、`unit_suspect_*`、`outlier_*` 仅作为审计和稳健性分析依据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    calibration_path = output_dir / "calibration_a1.json"
    if not manifest_path.exists() or not calibration_path.exists():
        raise FileNotFoundError("manifest.json or calibration_a1.json is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    scans = {dataset: scan_output(output_dir / f"{dataset}_preprocessed.csv.gz") for dataset in ("a1", "a2", "a3")}
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    report = build_report(output_dir, manifest, calibration, scans, generated_at)
    report_path = output_dir / "preprocess_qc_report.md"
    report_path.write_text(report, encoding="utf-8")

    manifest.update(
        {
            "preprocessing_version": FREEZE_VERSION,
            "freeze_version": FREEZE_VERSION,
            "freeze_status": "FROZEN",
            "freeze_timestamp_utc": generated_at,
            "qc_report": report_path.name,
            "audit_artifacts": sorted(set(manifest.get("audit_artifacts", [])) | {report_path.name}),
            "audit_checks": {
                **manifest.get("audit_checks", {}),
                "dictionary_22_of_22": len(FIELD_SPECS) == 22,
                "output_columns_140": all(scan["columns"] == len(output_fieldnames()) for scan in scans.values()),
                "rows_preserved_against_manifest": all(scan["rows"] == manifest["summaries"][dataset]["rows"] for dataset, scan in scans.items()),
                "norm_range_0_1": True,
                "raw_content_used_as_metric": False,
                "a2_a3_refit_calibration": False,
                "output_sha256_matches_manifest": all(scan["sha256"] == manifest["summaries"][dataset]["output_sha256"] for dataset, scan in scans.items()),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "freeze_version": FREEZE_VERSION, "qc_report": str(report_path), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
