"""F题 Q1：A1-A3 公共质量指标预处理。

设计原则
--------
1. 22 个指标使用同一份字段配置和标量化规则。
2. 所有标准化参数只在 A1 上拟合；A2、A3 只应用 A1 参数。
3. 结构性无效值进入缺失；统计异常和单位可疑值保留原值并单独标记。
4. 原始输入保持只读，输出保存原始标量、统一方向后的 [0, 1] 值和审计标记。

本脚本只负责公共预处理，不计算最终综合质量分 Q，也不进行配比-Loss 回归。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import lzma
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A_ROOT = PROJECT_ROOT / "中文题目" / "F题" / "real_attachments" / "A_data_value"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"
OUTLIER_THRESHOLD = 3.5


# direction 的含义：
# positive = 原值越大越好；negative = 原值越小越好；central = 越接近 A1 的稳健中心越好。
# transform 用于拟合 A1 校准参数前的尺度变换，不改变输出中的 scalar 原值。
FIELD_SPECS: "OrderedDict[str, Dict[str, str]]" = OrderedDict(
    [
        ("fineweb_edu", {"kind": "single_list", "direction": "positive", "transform": "identity"}),
        ("fluency_en", {"kind": "prob_index_1", "direction": "positive", "transform": "identity"}),
        ("modernbert_cleanliness", {"kind": "ordinal_logits", "direction": "positive", "transform": "identity"}),
        ("modernbert_readability", {"kind": "ordinal_logits", "direction": "positive", "transform": "identity"}),
        ("modernbert_reasoning", {"kind": "ordinal_logits", "direction": "positive", "transform": "identity"}),
        ("modernbert_professionalism", {"kind": "ordinal_logits", "direction": "positive", "transform": "identity"}),
        ("dsir_books", {"kind": "single", "direction": "positive", "transform": "identity"}),
        ("dsir_wiki", {"kind": "single", "direction": "positive", "transform": "identity"}),
        ("dsir_math", {"kind": "single", "direction": "positive", "transform": "identity"}),
        ("qurater", {"kind": "mean_list", "direction": "positive", "transform": "identity"}),
        ("ad_en", {"kind": "prob_index_1", "direction": "positive", "transform": "identity"}),
        ("rps_doc_word_count", {"kind": "single", "direction": "central", "transform": "log1p"}),
        ("rps_doc_num_sentences", {"kind": "single", "direction": "central", "transform": "log1p"}),
        ("rps_doc_unigram_entropy", {"kind": "single", "direction": "positive", "transform": "identity"}),
        ("rps_doc_frac_unique_words", {"kind": "single", "direction": "positive", "transform": "identity"}),
        ("rps_doc_frac_no_alph_words", {"kind": "single", "direction": "negative", "transform": "identity"}),
        ("rps_doc_frac_chars_top_2gram", {"kind": "single", "direction": "negative", "transform": "identity"}),
        ("rps_doc_frac_chars_top_3gram", {"kind": "single", "direction": "negative", "transform": "identity"}),
        ("rps_lines_uppercase_letter_fraction", {"kind": "single", "direction": "negative", "transform": "identity"}),
        (
            "rps_lines_ending_with_terminal_punctution_mark",
            {"kind": "single", "direction": "positive", "transform": "identity"},
        ),
        # 数字比例和平均词长在不同域中不具有统一的单调方向，按“接近 A1 稳健中心”处理。
        ("rps_lines_numerical_chars_fraction", {"kind": "single", "direction": "central", "transform": "identity"}),
        ("rps_doc_mean_word_length", {"kind": "single", "direction": "central", "transform": "identity"}),
    ]
)

# 这些字段的定义域明确要求非负。负值不能通过 max(value, 0) 静默修复，
# 而应被标为 invalid，并在本层作为缺失处理。
NONNEGATIVE_FIELDS = frozenset(
    {
        "rps_doc_word_count",
        "rps_doc_num_sentences",
        "rps_doc_unigram_entropy",
        "rps_doc_frac_unique_words",
        "rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_lines_numerical_chars_fraction",
        "rps_doc_mean_word_length",
    }
)

# 报告显示这些字段存在百分数/分数混用风险，且个别值超过 100。
# 本层不擅自除以 100，也不截断，只保留原值并打 unit_suspect 标记。
UNIT_SUSPECT_FIELDS = frozenset(
    {
        "rps_doc_frac_unique_words",
        "rps_doc_frac_no_alph_words",
        "rps_doc_frac_chars_top_2gram",
        "rps_doc_frac_chars_top_3gram",
        "rps_lines_uppercase_letter_fraction",
        "rps_lines_ending_with_terminal_punctution_mark",
        "rps_lines_numerical_chars_fraction",
    }
)

# 用 A1 的主体分布建立公共经验 CDF，尾部极端值不参与坐标轴跨度。
# 超出该范围的有效值仍会被映射到 0/1；统计异常只标记，不删除记录。
CALIBRATION_Q_LOW = 0.005
CALIBRATION_Q_HIGH = 0.995


def finite_float(value: Any) -> float:
    """将标量安全转换为有限 float；失败返回 NaN。"""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def stable_softmax(values: Sequence[Any]) -> Optional[np.ndarray]:
    arr = np.asarray([finite_float(v) for v in values], dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    shifted = arr - np.max(arr)
    exp_values = np.exp(shifted)
    denom = float(exp_values.sum())
    if not math.isfinite(denom) or denom <= 0:
        return None
    return exp_values / denom


def scalarize(value: Any, kind: str) -> float:
    """把 22 个字段统一压缩为一个标量。"""

    if kind == "single":
        return finite_float(value)

    if kind == "single_list":
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return finite_float(value[0])
        return float("nan")

    if not isinstance(value, (list, tuple)):
        return float("nan")

    if kind == "prob_index_1":
        probabilities = stable_softmax(value)
        return float(probabilities[1]) if probabilities is not None and len(probabilities) > 1 else float("nan")

    if kind == "ordinal_logits":
        probabilities = stable_softmax(value)
        if probabilities is None or len(probabilities) != 6:
            return float("nan")
        levels = np.arange(6, dtype=float)
        return float(np.dot(levels, probabilities) / 5.0)

    if kind == "mean_list":
        arr = np.asarray([finite_float(v) for v in value], dtype=float)
        return float(np.mean(arr)) if arr.size and np.all(np.isfinite(arr)) else float("nan")

    raise ValueError(f"unknown scalarization kind: {kind}")


def apply_base_transform(value: float, transform: str, field: Optional[str] = None) -> float:
    if not math.isfinite(value):
        return float("nan")
    if field in NONNEGATIVE_FIELDS and value < 0:
        return float("nan")
    if transform == "identity":
        return value
    if transform == "log1p":
        # 负的计数不是“很小的计数”，不能静默截到 0。
        if value < 0:
            return float("nan")
        return math.log1p(value)
    raise ValueError(f"unknown transform: {transform}")


def robust_scale(values: np.ndarray) -> Tuple[float, float]:
    """返回 median 和非零稳健尺度，优先 MAD，退化时回退 IQR/1.349 或 std。"""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad > 0 and math.isfinite(mad):
        return median, 1.4826 * mad
    q25, q75 = np.percentile(finite, [25, 75])
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale > 0 and math.isfinite(iqr_scale):
        return median, iqr_scale
    std = float(np.std(finite))
    return median, std if std > 0 and math.isfinite(std) else 1.0


def robust_z_outlier(value: float, median: float, scale: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(median) or not math.isfinite(scale) or scale <= 0:
        return False
    return abs((value - median) / scale) > OUTLIER_THRESHOLD


def empirical_cdf(value: float, grid_values: np.ndarray, grid_probs: np.ndarray) -> float:
    if not math.isfinite(value):
        return float("nan")
    result = float(np.interp(value, grid_values, grid_probs, left=0.0, right=1.0))
    return float(np.clip(result, 0.0, 1.0))


def scalarize_record(record: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, bool]]:
    scalars: Dict[str, float] = {}
    missing: Dict[str, bool] = {}
    for field, spec in FIELD_SPECS.items():
        value = scalarize(record.get(field), spec["kind"])
        scalars[field] = value
        missing[field] = not math.isfinite(value)
    return scalars, missing


def fit_calibration(a1_path: Path, quantile_points: int = 2001, max_rows: Optional[int] = None) -> Dict[str, Any]:
    columns: Dict[str, List[float]] = {field: [] for field in FIELD_SPECS}
    rows = 0
    for record, _ in stream_records(a1_path, dataset="a1", max_rows=max_rows):
        scalars, _ = scalarize_record(record)
        rows += 1
        for field, spec in FIELD_SPECS.items():
            value = apply_base_transform(scalars[field], spec["transform"], field)
            columns[field].append(value)

    if rows == 0:
        raise ValueError("A1 contains no usable records")

    calibration: Dict[str, Any] = {
        "schema_version": "q1-common-preprocess-v1",
        "fit_dataset": "A1",
        "fit_rows": rows,
        "quantile_points": quantile_points,
        "calibration_quantile_clip": [CALIBRATION_Q_LOW, CALIBRATION_Q_HIGH],
        "fields": {},
    }
    probs = np.linspace(CALIBRATION_Q_LOW, CALIBRATION_Q_HIGH, quantile_points)
    for field, spec in FIELD_SPECS.items():
        values = np.asarray(columns[field], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"field {field} has no finite A1 values")
        median, scale = robust_scale(finite)
        grid_values = np.quantile(finite, probs)
        # np.interp 需要单调自变量；重复分位点在常量/离散字段中很常见。
        # 对重复值统一取最后一次出现的经验概率，避免同一取值对应多个 CDF 值。
        last_indices = np.r_[np.flatnonzero(np.diff(grid_values) != 0), len(grid_values) - 1]
        unique_values = grid_values[last_indices]
        unique_probs = probs[last_indices]
        calibration["fields"][field] = {
            "kind": spec["kind"],
            "direction": spec["direction"],
            "transform": spec["transform"],
            "n_finite": int(finite.size),
            "n_missing": int(values.size - finite.size),
            "median": median,
            "robust_scale": scale,
            "grid_values": unique_values.tolist(),
            "grid_probs": unique_probs.tolist(),
        }
    return calibration


def stream_records(path: Path, dataset: str, max_rows: Optional[int] = None) -> Iterable[Tuple[Dict[str, Any], str]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_rows is not None and index >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if dataset == "a1":
                domain = str(record.get("_source_domain") or "unknown")
            else:
                match = re.match(r"(arxiv|github)_", path.name)
                domain = match.group(1) if match else "unknown"
            yield record, domain


def apply_calibration(
    record: Mapping[str, Any],
    domain: str,
    dataset: str,
    calibration: Mapping[str, Any],
) -> Dict[str, Any]:
    scalars, missing = scalarize_record(record)
    output: Dict[str, Any] = {
        "dataset": dataset,
        "source_domain": domain,
        "id": record.get("id", ""),
        "sub_path": record.get("sub_path", ""),
    }
    missing_count = 0
    invalid_count = 0
    unit_suspect_count = 0
    outlier_count = 0
    for field, spec in FIELD_SPECS.items():
        scalar = scalars[field]
        output[f"scalar_{field}"] = "" if not math.isfinite(scalar) else scalar
        transformed = apply_base_transform(scalar, spec["transform"], field)
        field_cal = calibration["fields"][field]
        is_missing = bool(missing[field])
        is_invalid = not is_missing and not math.isfinite(transformed)
        is_unit_suspect = (
            not is_missing
            and not is_invalid
            and field in UNIT_SUSPECT_FIELDS
            and (scalar < 0 or scalar > 100)
        )
        is_outlier = False if (is_missing or is_invalid) else robust_z_outlier(
            transformed,
            float(field_cal["median"]),
            float(field_cal["robust_scale"]),
        )
        output[f"missing_{field}"] = int(is_missing)
        output[f"invalid_{field}"] = int(is_invalid)
        output[f"unit_suspect_{field}"] = int(is_unit_suspect)
        output[f"outlier_{field}"] = int(is_outlier)
        if is_missing or is_invalid:
            output[f"norm_{field}"] = ""
            missing_count += int(is_missing)
            invalid_count += int(is_invalid)
            continue
        grid_values = np.asarray(field_cal["grid_values"], dtype=float)
        grid_probs = np.asarray(field_cal["grid_probs"], dtype=float)
        cdf = empirical_cdf(transformed, grid_values, grid_probs)
        if spec["direction"] == "positive":
            normalized = cdf
        elif spec["direction"] == "negative":
            normalized = 1.0 - cdf
        elif spec["direction"] == "central":
            scale = max(float(field_cal["robust_scale"]), 1e-12)
            normalized = math.exp(-abs(transformed - float(field_cal["median"])) / scale)
        else:
            raise ValueError(f"unknown direction: {spec['direction']}")
        output[f"norm_{field}"] = float(np.clip(normalized, 0.0, 1.0))
        unit_suspect_count += int(is_unit_suspect)
        outlier_count += int(is_outlier)

    output["missing_count"] = missing_count
    output["invalid_count"] = invalid_count
    output["unit_suspect_count"] = unit_suspect_count
    output["outlier_count"] = outlier_count
    return output


def output_fieldnames() -> List[str]:
    fields = ["dataset", "source_domain", "id", "sub_path"]
    for prefix in ("scalar", "norm", "missing", "invalid", "unit_suspect", "outlier"):
        fields.extend(f"{prefix}_{field}" for field in FIELD_SPECS)
    fields.extend(["missing_count", "invalid_count", "unit_suspect_count", "outlier_count"])
    return fields


def process_dataset(
    input_paths: Sequence[Path],
    dataset: str,
    output_path: Path,
    calibration: Mapping[str, Any],
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    total = 0
    missing_total = 0
    invalid_total = 0
    unit_suspect_total = 0
    outlier_total = 0
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames())
        writer.writeheader()
        for path in input_paths:
            for record, domain in stream_records(path, dataset=dataset, max_rows=max_rows):
                row = apply_calibration(record, domain, dataset, calibration)
                writer.writerow(row)
                total += 1
                counts[domain] = counts.get(domain, 0) + 1
                missing_total += int(row["missing_count"])
                invalid_total += int(row["invalid_count"])
                unit_suspect_total += int(row["unit_suspect_count"])
                outlier_total += int(row["outlier_count"])
    return {
        "dataset": dataset,
        "rows": total,
        "domains": counts,
        "missing_values": missing_total,
        "invalid_values": invalid_total,
        "unit_suspect_values": unit_suspect_total,
        "outlier_flags": outlier_total,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def input_paths(a_root: Path) -> Dict[str, List[Path]]:
    a1 = [a_root / "slimpajama_quality_signal_sample.jsonl.xz"]
    a2 = sorted((a_root / "slimpajama_quality_extended").glob("arxiv_*.jsonl.xz"))
    a3 = sorted((a_root / "slimpajama_quality_extended").glob("github_*.jsonl.xz"))
    if not a1[0].exists():
        raise FileNotFoundError(a1[0])
    if not a2 or not a3:
        raise FileNotFoundError("A2/A3 extended JSONL.XZ files were not found")
    return {"a1": a1, "a2": a2, "a3": a3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit A1 calibration and apply common preprocessing to A1-A3.")
    parser.add_argument("--a-root", type=Path, default=DEFAULT_A_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=["a1", "a2", "a3"], default=["a1", "a2", "a3"])
    parser.add_argument("--max-rows", type=int, default=None, help="每个输入文件最多读取多少行，仅用于 smoke test")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = input_paths(args.a_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "calibration_a1.json"
    manifest_path = args.output_dir / "manifest.json"
    if not args.overwrite:
        existing = [args.output_dir / f"{dataset}_preprocessed.csv.gz" for dataset in args.datasets]
        existing = [path for path in existing if path.exists()]
        if existing:
            raise FileExistsError(f"output exists; use --overwrite: {existing}")

    calibration = fit_calibration(paths["a1"][0], max_rows=args.max_rows)
    calibration["field_specs"] = FIELD_SPECS
    calibration["nonnegative_fields"] = sorted(NONNEGATIVE_FIELDS)
    calibration["unit_suspect_fields"] = sorted(UNIT_SUSPECT_FIELDS)
    calibration["input_a1"] = {
        "path": str(paths["a1"][0]),
        "size": paths["a1"][0].stat().st_size,
        "sha256": sha256_file(paths["a1"][0]),
    }
    calibration_path.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding="utf-8")

    summaries: Dict[str, Any] = {}
    for dataset in args.datasets:
        output_path = args.output_dir / f"{dataset}_preprocessed.csv.gz"
        summaries[dataset] = process_dataset(
            paths[dataset],
            dataset,
            output_path,
            calibration,
            max_rows=args.max_rows,
        )

    manifest = {
        "schema_version": "q1-common-preprocess-v1",
        "preprocessing_version": "q1-common-v1.1",
        "command": " ".join(sys.argv),
        "project_root": str(PROJECT_ROOT),
        "a_root": str(args.a_root),
        "output_dir": str(args.output_dir),
        "fit_dataset": "A1",
        "same_calibration_for": args.datasets,
        "max_rows": args.max_rows,
        "field_specs": FIELD_SPECS,
        "nonnegative_fields": sorted(NONNEGATIVE_FIELDS),
        "unit_suspect_fields": sorted(UNIT_SUSPECT_FIELDS),
        "outlier_rule": {
            "method": "robust_z_score",
            "scale": "1.4826*MAD; fallback IQR/1.349, then SD",
            "threshold": OUTLIER_THRESHOLD,
            "comparison": "abs((value - A1_median) / A1_robust_scale) > threshold",
            "action": "flag_only; retain scalar and normalized value",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "preprocess_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "summaries": summaries,
        "input_files": [
            {
                "dataset": dataset,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for dataset in args.datasets
            for path in paths[dataset]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": str(calibration_path), "manifest": str(manifest_path), "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
