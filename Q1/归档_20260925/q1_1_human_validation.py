#!/usr/bin/env python3
"""Q1.1 A1 双人盲评的预注册分析程序。

本程序只分析真实回填的评分，不生成、补齐或模拟人工分数。输入不完整时
返回退出码 2，且不创建正式分析输出。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = ROOT / "results" / "q1_1" / "v1" / "review_packet"
DEFAULT_RATINGS = DEFAULT_REVIEW_DIR / "blind_review_form.csv"
DEFAULT_MAPPING = DEFAULT_REVIEW_DIR / "restricted_id_mapping.csv"
DEFAULT_FRAME = DEFAULT_REVIEW_DIR / "sampling_frame.csv"
DEFAULT_SCORES = ROOT / "results" / "q1_1" / "v1" / "sample_scores.csv.gz"
DEFAULT_SENSITIVITY_SUMMARY = ROOT / "results" / "q1_1" / "v1" / "sample_score_sensitivity.csv"
DEFAULT_OUTPUT_DIR = ROOT / "Q1" / "03_结果" / "Q1.1" / "human_validation"

DIMENSIONS = ("education", "readability", "coherence", "information", "overall")
RATING_COLUMNS = tuple(
    f"rater_{rater}_{dimension}"
    for rater in (1, 2)
    for dimension in DIMENSIONS
)
EXPECTED_SAMPLE_COUNT = 793
EXPECTED_DOMAINS = 7
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260924
MIN_VALID_BOOTSTRAP_PROPORTION = 0.95
MAX_CI_HALF_WIDTH = 0.15
MIN_OVERALL_KAPPA = 0.40


class InputError(ValueError):
    """输入结构或取值违反冻结合同。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise InputError(f"输入文件不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError(f"CSV 缺少表头：{path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def require_columns(path: Path, header: Sequence[str], required: Iterable[str]) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise InputError(f"{path} 缺少字段：{', '.join(missing)}")


def parse_int_score(raw: str, column: str, blind_id: str) -> int:
    text = raw.strip()
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise InputError(f"{blind_id} 的 {column} 不是数值：{raw!r}") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise InputError(f"{blind_id} 的 {column} 必须是 1–5 整数：{raw!r}")
    integer = int(value)
    if integer < 1 or integer > 5:
        raise InputError(f"{blind_id} 的 {column} 超出 1–5：{raw!r}")
    return integer


def load_mapping(path: Path) -> list[dict[str, Any]]:
    header, rows = read_csv(path)
    required = (
        "blind_id", "dataset", "source_domain", "raw_id", "q_equal_tertile",
        "broad_unicode_flag", "N_h", "n_h", "design_weight", "content_sha256",
    )
    require_columns(path, header, required)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise InputError(f"映射应为 {EXPECTED_SAMPLE_COUNT} 行，实际 {len(rows)} 行")
    seen: set[str] = set()
    seen_raw_ids: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        blind_id = row["blind_id"].strip()
        if not blind_id or blind_id in seen:
            raise InputError(f"映射 blind_id 为空或重复：{blind_id!r}")
        seen.add(blind_id)
        if row["dataset"].strip().lower() != "a1":
            raise InputError(f"盲评映射只能包含 A1：{blind_id}")
        raw_id = row["raw_id"].strip()
        if not raw_id or raw_id in seen_raw_ids:
            raise InputError(f"映射 raw_id 为空或重复：{raw_id!r}")
        seen_raw_ids.add(raw_id)
        try:
            tertile = int(row["q_equal_tertile"])
            unicode_flag = int(row["broad_unicode_flag"])
            population_n = int(row["N_h"])
            sample_n = int(row["n_h"])
            design_weight = float(row["design_weight"])
        except ValueError as exc:
            raise InputError(f"映射数值字段不能解析：{blind_id}") from exc
        if tertile not in (1, 2, 3) or unicode_flag not in (0, 1):
            raise InputError(f"映射分层值非法：{blind_id}")
        if population_n <= 0 or sample_n <= 0 or sample_n > population_n:
            raise InputError(f"映射 N_h/n_h 非法：{blind_id}")
        expected_weight = population_n / sample_n
        if not math.isclose(design_weight, expected_weight, rel_tol=1e-10, abs_tol=1e-12):
            raise InputError(f"映射设计权重与 N_h/n_h 不一致：{blind_id}")
        parsed.append({
            "blind_id": blind_id,
            "dataset": "a1",
            "source_domain": row["source_domain"].strip(),
            "raw_id": raw_id,
            "q_equal_tertile": tertile,
            "broad_unicode_flag": unicode_flag,
            "N_h": population_n,
            "n_h": sample_n,
            "design_weight": design_weight,
            "content_sha256": row["content_sha256"].strip(),
        })
    domains = {row["source_domain"] for row in parsed}
    if len(domains) != EXPECTED_DOMAINS:
        raise InputError(f"盲评映射应覆盖 {EXPECTED_DOMAINS} 个 A1 域，实际 {len(domains)} 个")
    return parsed


def validate_sampling_frame(path: Path, mapping: Sequence[Mapping[str, Any]]) -> None:
    header, rows = read_csv(path)
    required = (
        "dataset", "source_domain", "q_equal_tertile", "broad_unicode_flag",
        "N_h", "n_h", "design_weight",
    )
    require_columns(path, header, required)
    frame: dict[tuple[str, int, int], tuple[int, int, float | None]] = {}
    for row in rows:
        if row["dataset"].strip().lower() != "a1":
            raise InputError("sampling_frame.csv 含非 A1 记录")
        key = (
            row["source_domain"].strip(),
            int(row["q_equal_tertile"]),
            int(row["broad_unicode_flag"]),
        )
        if key in frame:
            raise InputError(f"sampling_frame.csv 分层重复：{key}")
        population_n = int(row["N_h"])
        sample_n = int(row["n_h"])
        weight = float(row["design_weight"]) if row["design_weight"].strip() else None
        frame[key] = (population_n, sample_n, weight)
    counts: dict[tuple[str, int, int], int] = defaultdict(int)
    for row in mapping:
        key = (
            str(row["source_domain"]),
            int(row["q_equal_tertile"]),
            int(row["broad_unicode_flag"]),
        )
        counts[key] += 1
    for key, selected_n in counts.items():
        if key not in frame:
            raise InputError(f"映射分层不在 sampling_frame.csv：{key}")
        population_n, sample_n, weight = frame[key]
        if selected_n != sample_n or weight is None:
            raise InputError(f"映射样本数与 sampling_frame.csv 不一致：{key}")
        if population_n <= 0 or not math.isclose(weight, population_n / sample_n, rel_tol=1e-10):
            raise InputError(f"sampling_frame.csv 设计权重非法：{key}")
    if sum(int(row["n_h"]) for row in rows) != EXPECTED_SAMPLE_COUNT:
        raise InputError("sampling_frame.csv 的 n_h 合计不是 793")


def load_ratings(
    path: Path,
    expected_ids: set[str],
) -> tuple[str, dict[str, dict[str, int]], dict[str, Any]]:
    header, rows = read_csv(path)
    require_columns(path, header, ("blind_id", *RATING_COLUMNS))
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise InputError(f"评分表应为 {EXPECTED_SAMPLE_COUNT} 行，实际 {len(rows)} 行")
    seen: set[str] = set()
    missing_cells: list[tuple[str, str]] = []
    parsed: dict[str, dict[str, int]] = {}
    for row in rows:
        blind_id = row["blind_id"].strip()
        if not blind_id or blind_id in seen:
            raise InputError(f"评分表 blind_id 为空或重复：{blind_id!r}")
        seen.add(blind_id)
        values: dict[str, int] = {}
        for column in RATING_COLUMNS:
            raw = (row.get(column) or "").strip()
            if raw == "":
                missing_cells.append((blind_id, column))
            else:
                values[column] = parse_int_score(raw, column, blind_id)
        parsed[blind_id] = values
    unknown = sorted(seen - expected_ids)
    absent = sorted(expected_ids - seen)
    if unknown or absent:
        raise InputError(
            f"评分表与盲评映射 ID 不一致：未知 {len(unknown)}，缺失 {len(absent)}"
        )
    total_cells = EXPECTED_SAMPLE_COUNT * len(RATING_COLUMNS)
    filled_cells = total_cells - len(missing_cells)
    readiness = {
        "required_rows": EXPECTED_SAMPLE_COUNT,
        "required_score_cells": total_cells,
        "filled_score_cells": filled_cells,
        "missing_score_cells": len(missing_cells),
        "complete_rows": sum(len(values) == len(RATING_COLUMNS) for values in parsed.values()),
    }
    if missing_cells:
        status = "not_ready_empty" if filled_cells == 0 else "not_ready_partial"
        return status, parsed, readiness
    return "ready", parsed, readiness


def load_scores(path: Path, mapping: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    if not path.is_file():
        raise InputError(f"候选分文件不存在：{path}")
    targets = {str(row["raw_id"]): str(row["source_domain"]) for row in mapping}
    found: dict[str, tuple[float, float]] = {}
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    if path.suffix.lower() == ".gz":
        stream_context = opener(path, "rt", encoding="utf-8-sig", newline="")
    else:
        stream_context = opener(path, "r", encoding="utf-8-sig", newline="")
    with stream_context as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise InputError("候选分文件缺少表头")
        require_columns(path, reader.fieldnames, ("dataset", "source_domain", "id", "q_equal", "q_huber"))
        for row in reader:
            if row["dataset"].strip().lower() != "a1":
                continue
            raw_id = row["id"].strip()
            if raw_id not in targets:
                continue
            if raw_id in found:
                raise InputError(f"候选分文件 A1 ID 重复：{raw_id}")
            if row["source_domain"].strip() != targets[raw_id]:
                raise InputError(f"候选分来源域与盲评映射不一致：{raw_id}")
            try:
                q_equal = float(row["q_equal"])
                q_huber = float(row["q_huber"])
            except ValueError as exc:
                raise InputError(f"候选分不能解析：{raw_id}") from exc
            if not all(math.isfinite(value) for value in (q_equal, q_huber)):
                raise InputError(f"候选分非有限值：{raw_id}")
            found[raw_id] = (q_equal, q_huber)
    missing = sorted(set(targets) - set(found))
    if missing:
        raise InputError(f"793 条盲评记录中有 {len(missing)} 条无法连接候选分")
    return found


def build_records(
    mapping: Sequence[Mapping[str, Any]],
    ratings: Mapping[str, Mapping[str, int]],
    scores: Mapping[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for mapped in mapping:
        blind_id = str(mapped["blind_id"])
        raw_id = str(mapped["raw_id"])
        record = dict(mapped)
        record.update(ratings[blind_id])
        record["q_equal"], record["q_huber"] = scores[raw_id]
        record["human_overall"] = (
            record["rater_1_overall"] + record["rater_2_overall"]
        ) / 2.0
        record["stratum"] = (
            record["source_domain"],
            record["q_equal_tertile"],
            record["broad_unicode_flag"],
        )
        records.append(record)
    return records


def weighted_quadratic_kappa(
    first: Sequence[int],
    second: Sequence[int],
    sample_weights: Sequence[float],
) -> float | None:
    if not (len(first) == len(second) == len(sample_weights)) or not first:
        return None
    observed = [[0.0] * 5 for _ in range(5)]
    total = 0.0
    for left, right, weight in zip(first, second, sample_weights):
        if weight <= 0 or not math.isfinite(weight):
            return None
        observed[left - 1][right - 1] += weight
        total += weight
    if total <= 0:
        return None
    observed = [[cell / total for cell in row] for row in observed]
    first_margin = [sum(row) for row in observed]
    second_margin = [sum(observed[i][j] for i in range(5)) for j in range(5)]
    observed_agreement = 0.0
    expected_agreement = 0.0
    for i in range(5):
        for j in range(5):
            agreement_weight = 1.0 - ((i + 1) - (j + 1)) ** 2 / 16.0
            observed_agreement += agreement_weight * observed[i][j]
            expected_agreement += agreement_weight * first_margin[i] * second_margin[j]
    denominator = 1.0 - expected_agreement
    if abs(denominator) <= 1e-15:
        return None
    value = (observed_agreement - expected_agreement) / denominator
    return value if math.isfinite(value) else None


def weighted_midranks(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    grouped: dict[float, float] = defaultdict(float)
    for value, weight in zip(values, weights):
        grouped[float(value)] += float(weight)
    rank_by_value: dict[float, float] = {}
    cumulative = 0.0
    for value in sorted(grouped):
        group_weight = grouped[value]
        rank_by_value[value] = cumulative + 0.5 * group_weight
        cumulative += group_weight
    return [rank_by_value[float(value)] for value in values]


def weighted_pearson(
    first: Sequence[float],
    second: Sequence[float],
    weights: Sequence[float],
) -> float | None:
    total = sum(weights)
    if total <= 0 or not (len(first) == len(second) == len(weights)):
        return None
    first_mean = sum(w * x for x, w in zip(first, weights)) / total
    second_mean = sum(w * y for y, w in zip(second, weights)) / total
    covariance = sum(
        w * (x - first_mean) * (y - second_mean)
        for x, y, w in zip(first, second, weights)
    )
    first_variance = sum(w * (x - first_mean) ** 2 for x, w in zip(first, weights))
    second_variance = sum(w * (y - second_mean) ** 2 for y, w in zip(second, weights))
    denominator = math.sqrt(first_variance * second_variance)
    if denominator <= 1e-15:
        return None
    value = covariance / denominator
    return value if math.isfinite(value) else None


def weighted_spearman(
    first: Sequence[float],
    second: Sequence[float],
    weights: Sequence[float],
) -> float | None:
    if not first or not (len(first) == len(second) == len(weights)):
        return None
    return weighted_pearson(
        weighted_midranks(first, weights),
        weighted_midranks(second, weights),
        weights,
    )


def domain_correlations(
    records: Sequence[Mapping[str, Any]], candidate: str
) -> tuple[list[dict[str, Any]], float | None]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_domain"])].append(record)
    rows: list[dict[str, Any]] = []
    correlations: list[float] = []
    for domain in sorted(grouped):
        subset = grouped[domain]
        value = weighted_spearman(
            [float(row[candidate]) for row in subset],
            [float(row["human_overall"]) for row in subset],
            [float(row["design_weight"]) for row in subset],
        )
        rows.append({
            "source_domain": domain,
            "candidate": candidate,
            "n_reviewed": len(subset),
            "design_weighted_spearman": value,
        })
        if value is None:
            return rows, None
        correlations.append(value)
    if len(correlations) != EXPECTED_DOMAINS:
        return rows, None
    return rows, sum(correlations) / len(correlations)


def percentile(values: Sequence[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def summarize_draws(name: str, point: float | None, draws: Sequence[float]) -> dict[str, Any]:
    low = percentile(draws, 0.025)
    high = percentile(draws, 0.975)
    valid_count = len(draws)
    return {
        "statistic": name,
        "point_estimate": point,
        "ci_low": low,
        "ci_high": high,
        "ci_half_width": (high - low) / 2.0 if low is not None and high is not None else None,
        "valid_bootstrap_count": valid_count,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "valid_bootstrap_proportion": valid_count / BOOTSTRAP_REPETITIONS,
        "ci_method": "paired stratified percentile bootstrap",
    }


def run_bootstrap(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    strata: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        strata[tuple(record["stratum"])].append(index)  # type: ignore[arg-type]
    ordered_strata = sorted(strata)
    rng = random.Random(BOOTSTRAP_SEED)
    draws: dict[str, list[float]] = {
        **{f"kappa_{dimension}": [] for dimension in DIMENSIONS},
        "macro_q_equal": [],
        "macro_q_huber": [],
        "macro_difference_huber_minus_equal": [],
    }
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled_indices: list[int] = []
        for stratum in ordered_strata:
            indices = strata[stratum]
            sampled_indices.extend(rng.choice(indices) for _ in range(len(indices)))
        sample = [records[index] for index in sampled_indices]
        weights = [float(row["design_weight"]) for row in sample]
        for dimension in DIMENSIONS:
            value = weighted_quadratic_kappa(
                [int(row[f"rater_1_{dimension}"]) for row in sample],
                [int(row[f"rater_2_{dimension}"]) for row in sample],
                weights,
            )
            if value is not None:
                draws[f"kappa_{dimension}"].append(value)
        _, equal_value = domain_correlations(sample, "q_equal")
        _, huber_value = domain_correlations(sample, "q_huber")
        if equal_value is not None:
            draws["macro_q_equal"].append(equal_value)
        if huber_value is not None:
            draws["macro_q_huber"].append(huber_value)
        if equal_value is not None and huber_value is not None:
            draws["macro_difference_huber_minus_equal"].append(huber_value - equal_value)
    return draws


def evaluate_selection(
    agreement: Mapping[str, Mapping[str, Any]],
    bootstrap_summary: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    overall = agreement["overall"]
    reliability_pass = bool(
        overall["design_weighted_kappa"] is not None
        and overall["design_weighted_kappa"] >= MIN_OVERALL_KAPPA
        and overall["ci_low"] is not None
        and overall["ci_low"] > 0
        and overall["valid_bootstrap_proportion"] >= MIN_VALID_BOOTSTRAP_PROPORTION
    )
    precision_names = (
        "macro_q_equal", "macro_q_huber", "macro_difference_huber_minus_equal"
    )
    precision_checks = {
        name: bool(
            bootstrap_summary[name]["ci_half_width"] is not None
            and bootstrap_summary[name]["ci_half_width"] <= MAX_CI_HALF_WIDTH
            and bootstrap_summary[name]["valid_bootstrap_proportion"]
            >= MIN_VALID_BOOTSTRAP_PROPORTION
        )
        for name in precision_names
    }
    precision_pass = all(precision_checks.values())
    sensitivity_status = "not_evaluable_from_aggregate_sensitivity_outputs"
    statistical_primary: str | None = None
    if not reliability_pass:
        status = "no_selection_rater_reliability_gate_failed"
    elif not precision_pass:
        status = "no_selection_precision_gate_failed"
    else:
        equal_low = bootstrap_summary["macro_q_equal"]["ci_low"]
        huber_low = bootstrap_summary["macro_q_huber"]["ci_low"]
        difference_low = bootstrap_summary["macro_difference_huber_minus_equal"]["ci_low"]
        if huber_low is not None and huber_low > 0 and difference_low is not None and difference_low > 0:
            status = "no_unique_selection_sensitivity_reversal_not_evaluable"
        elif equal_low is not None and equal_low > 0:
            statistical_primary = "q_equal"
            status = "q_equal_statistical_primary_a1_scoped"
        else:
            status = "no_selection_candidate_validity_not_demonstrated"
    return {
        "status": status,
        "statistical_primary_candidate": statistical_primary,
        "governance_operational_interface": "q_huber",
        "rater_reliability_gate_pass": reliability_pass,
        "precision_gate_pass": precision_pass,
        "precision_checks": precision_checks,
        "sensitivity_reversal_status": sensitivity_status,
        "claim_scope": "A1 original-text validation only; A2/A3 original text and verified ID mapping unavailable",
    }


def format_number(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    readiness: Mapping[str, Any],
    agreement_rows: Sequence[Mapping[str, Any]],
    correlation_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    inputs: Mapping[str, Path],
    rater_1_id: str,
    rater_2_id: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        output_dir / "agreement_by_dimension.csv",
        agreement_rows,
        (
            "dimension", "design_weighted_kappa", "unweighted_sample_kappa",
            "ci_low", "ci_high", "valid_bootstrap_count",
            "bootstrap_repetitions", "valid_bootstrap_proportion",
        ),
    )
    write_csv_rows(
        output_dir / "correlation_by_domain.csv",
        correlation_rows,
        ("source_domain", "candidate", "n_reviewed", "design_weighted_spearman"),
    )
    write_csv_rows(
        output_dir / "bootstrap_summary.csv",
        bootstrap_rows,
        (
            "statistic", "point_estimate", "ci_low", "ci_high", "ci_half_width",
            "valid_bootstrap_count", "bootstrap_repetitions",
            "valid_bootstrap_proportion", "ci_method",
        ),
    )
    summary = {
        "schema_version": "q1-1-human-validation-v1",
        "status": selection["status"],
        "readiness": dict(readiness),
        "rater_provenance": {
            "rater_1_id": rater_1_id or "not_recorded",
            "rater_2_id": rater_2_id or "not_recorded",
            "distinct_ids_confirmed": bool(rater_1_id and rater_2_id and rater_1_id != rater_2_id),
        },
        "selection": dict(selection),
        "agreement": {row["dimension"]: dict(row) for row in agreement_rows},
        "bootstrap": {row["statistic"]: dict(row) for row in bootstrap_rows},
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overall = next(row for row in agreement_rows if row["dimension"] == "overall")
    bootstrap_lookup = {row["statistic"]: row for row in bootstrap_rows}
    report_lines = [
        "# Q1.1 A1 双人盲评验证结果",
        "",
        f"- 统计状态：`{selection['status']}`",
        "- 下游治理接口：`q_huber`（治理冻结，不等于统计胜出）",
        f"- 统计主候选：`{selection['statistical_primary_candidate'] or 'none'}`",
        "- 验证范围：仅 A1 可追溯原文；A2/A3 缺原文与可核验 ID 映射。",
        "",
        "## 评审可靠性",
        "",
        "| 维度 | 设计加权 Kappa | 样本内 Kappa | 95% 区间 | 有效重复比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in agreement_rows:
        report_lines.append(
            f"| {row['dimension']} | {format_number(row['design_weighted_kappa'])} | "
            f"{format_number(row['unweighted_sample_kappa'])} | "
            f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}] | "
            f"{format_number(row['valid_bootstrap_proportion'], 3)} |"
        )
    report_lines += [
        "",
        f"整体质量可靠性门禁：**{'通过' if selection['rater_reliability_gate_pass'] else '未通过'}**。",
        "",
        "## 候选与人工主效标",
        "",
        "| 统计量 | 点估计 | 95% 区间 | 半宽 | 有效重复比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("macro_q_equal", "macro_q_huber", "macro_difference_huber_minus_equal"):
        row = bootstrap_lookup[name]
        report_lines.append(
            f"| {name} | {format_number(row['point_estimate'])} | "
            f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}] | "
            f"{format_number(row['ci_half_width'])} | "
            f"{format_number(row['valid_bootstrap_proportion'], 3)} |"
        )
    report_lines += [
        "",
        f"估计精度门禁：**{'通过' if selection['precision_gate_pass'] else '未通过'}**。",
        "",
        "## 选择与限制",
        "",
        f"预注册选择状态为 `{selection['status']}`。",
        "现有敏感性文件只含全量汇总，不含 793 条评审样本在各情景下的逐记录候选分；"
        "因此不能据其判断人工相关差是否发生实质反转。若基础结果指向 Huber 优于等权，"
        "该缺口会阻止唯一统计选择。",
        "",
        "无论统计分支为何，队伍冻结的 `q_huber` 仍单独记录为下游操作接口；"
        "不得把治理状态改写成人工效度胜出。",
        "",
        "A2/A3 没有可追溯原文及一一 ID 映射，本结果不能支持三数据集全面内容效度。",
        "",
        "## 复现参数",
        "",
        f"- 分层配对 bootstrap：{BOOTSTRAP_REPETITIONS} 次；种子 {BOOTSTRAP_SEED}。",
        f"- 整体质量 Kappa 门槛：点估计 ≥ {MIN_OVERALL_KAPPA}、区间下限 > 0、有效比例 ≥ {MIN_VALID_BOOTSTRAP_PROPORTION}。",
        f"- 三项相关精度门槛：区间半宽 ≤ {MAX_CI_HALF_WIDTH}、有效比例 ≥ {MIN_VALID_BOOTSTRAP_PROPORTION}。",
        f"- 评审身份：rater_1={rater_1_id or 'not_recorded'}；rater_2={rater_2_id or 'not_recorded'}。",
        "",
        f"整体质量设计加权 Kappa 点估计：{format_number(overall['design_weighted_kappa'])}。",
    ]
    (output_dir / "q1_1_human_validation_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "q1-1-human-validation-repro-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "parameters": {
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_valid_bootstrap_proportion": MIN_VALID_BOOTSTRAP_PROPORTION,
            "maximum_ci_half_width": MAX_CI_HALF_WIDTH,
            "minimum_overall_design_weighted_kappa": MIN_OVERALL_KAPPA,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in inputs.items()
            if path.is_file()
        },
        "outputs": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.name == "repro_manifest.json" or not path.is_file():
            continue
        manifest["outputs"][path.name] = sha256_file(path)
    (output_dir / "repro_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def self_test() -> None:
    perfect = weighted_quadratic_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1] * 5)
    if perfect is None or not math.isclose(perfect, 1.0, abs_tol=1e-12):
        raise AssertionError("perfect Kappa self-test failed")
    increasing = weighted_spearman([1, 2, 3, 4], [2, 4, 6, 8], [1, 2, 1, 3])
    decreasing = weighted_spearman([1, 2, 3, 4], [8, 6, 4, 2], [1, 2, 1, 3])
    if increasing is None or not math.isclose(increasing, 1.0, abs_tol=1e-12):
        raise AssertionError("increasing Spearman self-test failed")
    if decreasing is None or not math.isclose(decreasing, -1.0, abs_tol=1e-12):
        raise AssertionError("decreasing Spearman self-test failed")
    if percentile([0.0, 1.0], 0.25) != 0.25:
        raise AssertionError("percentile self-test failed")
    print(json.dumps({"status": "self_test_pass", "tests": 4}, ensure_ascii=False))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--sampling-frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--sensitivity-summary", type=Path, default=DEFAULT_SENSITIVITY_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rater-1-id", default="")
    parser.add_argument("--rater-2-id", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    rater_1_id = args.rater_1_id.strip()
    rater_2_id = args.rater_2_id.strip()
    if rater_1_id and rater_2_id and rater_1_id == rater_2_id:
        raise InputError("两名评审的记录标识不能相同")
    mapping = load_mapping(args.mapping)
    validate_sampling_frame(args.sampling_frame, mapping)
    expected_ids = {str(row["blind_id"]) for row in mapping}
    status, ratings, readiness = load_ratings(args.ratings, expected_ids)
    readiness_payload = {"status": status, **readiness}
    if status != "ready":
        print(json.dumps(readiness_payload, ensure_ascii=False, indent=2))
        return 2
    scores = load_scores(args.scores, mapping)
    if args.check_only:
        print(json.dumps({
            "status": "ready",
            **readiness,
            "candidate_score_links": len(scores),
            "rater_identity_check": (
                "confirmed_distinct"
                if rater_1_id and rater_2_id and rater_1_id != rater_2_id
                else "not_required_for_check_only"
            ),
        }, ensure_ascii=False, indent=2))
        return 0
    if not rater_1_id or not rater_2_id:
        raise InputError(
            "正式分析必须同时提供非空且不同的 --rater-1-id 与 --rater-2-id；"
            "--check-only 不要求评审标识"
        )
    records = build_records(mapping, ratings, scores)
    weights = [float(record["design_weight"]) for record in records]
    bootstrap_draws = run_bootstrap(records)

    agreement_rows: list[dict[str, Any]] = []
    agreement_lookup: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        first = [int(row[f"rater_1_{dimension}"]) for row in records]
        second = [int(row[f"rater_2_{dimension}"]) for row in records]
        design_kappa = weighted_quadratic_kappa(first, second, weights)
        sample_kappa = weighted_quadratic_kappa(first, second, [1.0] * len(records))
        summary = summarize_draws(
            f"kappa_{dimension}", design_kappa, bootstrap_draws[f"kappa_{dimension}"]
        )
        row = {
            "dimension": dimension,
            "design_weighted_kappa": design_kappa,
            "unweighted_sample_kappa": sample_kappa,
            **{key: value for key, value in summary.items() if key != "statistic"},
        }
        agreement_rows.append(row)
        agreement_lookup[dimension] = row

    equal_domain_rows, equal_macro = domain_correlations(records, "q_equal")
    huber_domain_rows, huber_macro = domain_correlations(records, "q_huber")
    correlation_rows = equal_domain_rows + huber_domain_rows
    point_values = {
        "macro_q_equal": equal_macro,
        "macro_q_huber": huber_macro,
        "macro_difference_huber_minus_equal": (
            huber_macro - equal_macro
            if huber_macro is not None and equal_macro is not None
            else None
        ),
    }
    bootstrap_rows = [
        summarize_draws(name, point_values[name], bootstrap_draws[name])
        for name in point_values
    ]
    bootstrap_lookup = {row["statistic"]: row for row in bootstrap_rows}
    selection = evaluate_selection(agreement_lookup, bootstrap_lookup)
    inputs = {
        "ratings": args.ratings,
        "restricted_id_mapping": args.mapping,
        "sampling_frame": args.sampling_frame,
        "sample_scores": args.scores,
        "aggregate_sensitivity_summary": args.sensitivity_summary,
    }
    write_outputs(
        args.output_dir,
        readiness_payload,
        agreement_rows,
        correlation_rows,
        bootstrap_rows,
        selection,
        inputs,
        rater_1_id,
        rater_2_id,
    )
    print(json.dumps({
        "status": selection["status"],
        "output_dir": str(args.output_dir.resolve()),
        "statistical_primary_candidate": selection["statistical_primary_candidate"],
        "governance_operational_interface": "q_huber",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as exc:
        print(json.dumps({"status": "input_error", "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2)
