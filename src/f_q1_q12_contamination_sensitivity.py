#!/usr/bin/env python
"""Q1.2 sensitivity to A1 rows with invisible Unicode in content.

The frozen norm_* values remain unchanged. This analysis removes only flagged
A1 rows from Q1.2's A1-derived thresholds, weights, Huber delta, and pair
rankings, then reapplies those estimates to the unchanged A2/A3 data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_DIR = PROJECT_ROOT / "results" / "q1_common_preprocess" / "v1"
DEFAULT_MAIN_DIR = PROJECT_ROOT / "results" / "q1_2_conflict_model" / "v1"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "results" / "q1_1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "q1_2_conflict_model" / "contamination_sensitivity"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
import f_q1_2_conflict_model as q12  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"拒绝写空表：{path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def subset_data(data: q12.DatasetData, keep: np.ndarray) -> q12.DatasetData:
    return q12.DatasetData(
        dataset=data.dataset,
        ids=data.ids[keep],
        domains=data.domains[keep],
        scores=data.scores[keep, :],
        outlier=data.outlier[keep, :],
        unit_suspect=data.unit_suspect[keep, :],
        input_rows=int(np.sum(keep)),
    )


def pair_gap_map(
    data: q12.DatasetData, indicators: Sequence[str]
) -> Dict[str, Dict[Tuple[int, int], float]]:
    _, gaps = q12.pair_diagnostics(data, indicators)
    return gaps


def macro_gaps(
    gap_by_domain: Mapping[str, Mapping[Tuple[int, int], float]],
    domains: Sequence[str],
    indicator_count: int,
) -> Dict[Tuple[int, int], float]:
    out: Dict[Tuple[int, int], float] = {}
    for j in range(indicator_count):
        for k in range(j + 1, indicator_count):
            values = [gap_by_domain[d].get((j, k), float("nan")) for d in domains]
            finite = [float(v) for v in values if math.isfinite(float(v))]
            out[(j, k)] = float(np.mean(finite)) if finite else float("nan")
    return out


def rank_pairs(gaps: Mapping[Tuple[int, int], float], indicators: Sequence[str]) -> List[Tuple[int, int]]:
    return sorted(
        gaps,
        key=lambda pair: (
            -float(gaps[pair]) if math.isfinite(float(gaps[pair])) else float("inf"),
            indicators[pair[0]], indicators[pair[1]],
        ),
    )


def run(args: argparse.Namespace) -> int:
    started = time.time()
    preprocess_dir = Path(args.preprocess_dir).resolve()
    main_dir = Path(args.main_dir).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    preprocess_manifest_path = preprocess_dir / "manifest.json"
    main_manifest_path = main_dir / "manifest.json"
    audit_manifest_path = audit_dir / "text_integrity_summary.json"
    audit_file_path = audit_dir / "text_integrity_audit.csv"
    preprocess_manifest = json.loads(preprocess_manifest_path.read_text(encoding="utf-8"))
    main_manifest = json.loads(main_manifest_path.read_text(encoding="utf-8"))
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if preprocess_manifest.get("freeze_status") != "FROZEN" or preprocess_manifest.get("status") != "frozen":
        raise ValueError("输入公共预处理结果未冻结")
    if main_manifest.get("status") != "full":
        raise ValueError("Q1.2 主结果不是 full 运行")
    raw_a1 = next(row for row in preprocess_manifest["input_files"] if row["dataset"] == "a1")
    raw_a1_path = Path(raw_a1["path"])
    if not raw_a1_path.is_absolute():
        raw_a1_path = PROJECT_ROOT / raw_a1_path
    if not raw_a1_path.is_file():
        raise FileNotFoundError(f"冻结清单中的原始 A1 文件不存在：{raw_a1_path}")
    raw_a1_actual_sha256 = sha256_file(raw_a1_path)
    if raw_a1_actual_sha256 != raw_a1["sha256"]:
        raise ValueError("当前原始 A1 文件 SHA-256 与冻结预处理清单不一致")
    if audit_manifest["datasets"]["A1"]["files"].get(Path(raw_a1["path"]).name) != raw_a1["sha256"]:
        raise ValueError("Unicode 审计使用的原始 A1 文件哈希与冻结清单不一致")
    if hashlib.sha256(audit_file_path.read_bytes()).hexdigest() != audit_manifest["record_audit_sha256"]:
        raise ValueError("原文完整性逐记录审计表哈希与 summary 不一致")

    indicators = list(preprocess_manifest["field_specs"].keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    data: Dict[str, q12.DatasetData] = {}
    input_hashes = {}
    for key in q12.DATASETS:
        path = preprocess_dir / f"{key}_preprocessed.csv.gz"
        expected_hash = preprocess_manifest["summaries"][key]["output_sha256"]
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{key} 预处理结果 SHA-256 校验失败")
        input_hashes[key] = actual_hash
        data[key] = q12._read_rows(
            path,
            indicators,
            int(preprocess_manifest["summaries"][key]["rows"]),
            None,
            args.seed,
        )

    audit_rows = read_csv(audit_file_path)
    if args.audit_rule == "cf_cc":
        flagged_rows = [
            row for row in audit_rows
            if row["dataset"] == "A1"
            and (int(row["format_char_count"] or 0) > 0 or int(row["other_control_count"] or 0) > 0)
        ]
        audit_rule_description = "A1 content 中存在 Unicode Cf，或除 TAB/LF/CR 外的 Cc"
    else:
        flagged_rows = [
            row for row in audit_rows
            if row["dataset"] == "A1" and str(row["unicode_integrity_flag"]).strip().lower() == "true"
        ]
        audit_rule_description = "A1 content 中存在 Cf/Cc/Zl/Zp/Co 或 Unicode noncharacter"
    flagged_keys = {row["sample_key_sha256"] for row in flagged_rows}
    a1 = data["a1"]
    a1_keys = np.array([
        hashlib.sha256(f"A1\0{sample_id if sample_id is not None else ''}".encode("utf-8")).hexdigest()
        for sample_id in a1.ids
    ], dtype=object)
    flagged_mask = np.isin(a1_keys, list(flagged_keys))
    if int(flagged_mask.sum()) != len(flagged_keys):
        raise ValueError("Unicode 审计样本键与 A1 冻结预处理 ID 未一一匹配")
    flagged_count = len(flagged_keys)
    clean_a1 = subset_data(a1, ~flagged_mask)
    if len(clean_a1.ids) < 3:
        raise ValueError("剔除 Unicode 候选行后 A1 有效样本不足")

    d_values = {key: q12.conflict_intensity(value.scores) for key, value in data.items()}
    clean_d_values = q12.conflict_intensity(clean_a1.scores)
    clean_indices_by_domain = q12._domain_indices(clean_a1.domains)
    full_indices_by_domain = q12._domain_indices(a1.domains)
    thresholds_main: Dict[str, float] = {}
    thresholds_clean: Dict[str, float] = {}
    threshold_cis_clean: Dict[str, Tuple[float, float]] = {}
    threshold_rows: List[Dict[str, object]] = []
    domain_rows: List[Dict[str, object]] = []
    for domain, clean_idx in clean_indices_by_domain.items():
        full_idx = full_indices_by_domain[domain]
        full_values = d_values["a1"][full_idx]
        clean_values = clean_d_values[clean_idx]
        clean_values = clean_values[np.isfinite(clean_values)]
        main_tau = q12._quantile(full_values, 0.95)
        clean_tau = q12._quantile(clean_values, 0.95)
        ci = q12._bootstrap_quantile_ci(
            clean_values, 0.95, args.threshold_bootstrap,
            q12._stable_seed(args.seed, f"unicode-clean-threshold-{domain}"),
        )
        thresholds_main[domain] = main_tau
        thresholds_clean[domain] = clean_tau
        threshold_cis_clean[domain] = ci
        flagged_domain = full_idx[flagged_mask[full_idx]]
        clean_domain_d = clean_d_values[clean_idx]
        flag_d = d_values["a1"][flagged_domain]
        domain_rows.append({
            "row_type": "a1_reference",
            "dataset": "a1",
            "source_domain": domain,
            "flagged_n": int(len(flagged_domain)),
            "clean_n": int(len(clean_idx)),
            "main_a1_q95": main_tau,
            "clean_a1_q95": clean_tau,
            "clean_threshold_ci_low": ci[0],
            "clean_threshold_ci_high": ci[1],
            "threshold_shift_clean_minus_main": clean_tau - main_tau,
            "flagged_median_D": float(np.nanmedian(flag_d)) if len(flag_d) else float("nan"),
            "flagged_rate_above_main_threshold": float(np.mean(flag_d > main_tau)) if len(flag_d) else float("nan"),
            "clean_a1_median_D": float(np.nanmedian(clean_domain_d)),
            "clean_a1_rate_above_clean_threshold": float(np.mean(clean_domain_d > clean_tau)),
        })
        threshold_rows.append({
            "source_domain": domain,
            "n_full_a1": int(len(full_idx)),
            "n_flagged_removed": int(len(flagged_domain)),
            "n_clean_a1": int(len(clean_idx)),
            "q95_full": main_tau,
            "q95_clean": clean_tau,
            "clean_q95_ci_low": ci[0],
            "clean_q95_ci_high": ci[1],
            "q95_shift": clean_tau - main_tau,
        })

    # The frozen norm_* calibration is untouched. Only A1-derived Q1.2
    # thresholds/weights/delta and pair rankings are re-estimated here.
    weights_clean, clean_weight_rows = q12.derive_reliability_weights(
        clean_a1, indicators, args.weight_bootstrap,
        q12._stable_seed(args.seed, "weights-a1"),
    )
    delta_clean, scale_clean, method_clean = q12.derive_huber_delta(clean_a1.scores)
    with (main_dir / "indicator_weights.csv").open(encoding="utf-8-sig", newline="") as stream:
        main_weight_rows = list(csv.DictReader(stream))
    weights_main = np.array([float(row["weight"]) for row in main_weight_rows], dtype=float)
    weight_l1 = float(np.sum(np.abs(weights_clean - weights_main)))
    weight_rank_rho = q12._spearman(weights_clean, weights_main, min_n=3)
    for row, main_weight in zip(clean_weight_rows, weights_main):
        row["primary_weight"] = float(main_weight)
        row["weight_change_clean_minus_primary"] = float(row["weight"] - main_weight)

    # Compare the full-A1 and clean-A1 pair gap rankings, both domain-wise and macro.
    full_gaps = pair_gap_map(a1, indicators)
    clean_gaps = pair_gap_map(clean_a1, indicators)
    main_a1_macro = macro_gaps(full_gaps, sorted(full_gaps), len(indicators))
    clean_a1_macro = macro_gaps(clean_gaps, sorted(clean_gaps), len(indicators))
    main_macro_order = rank_pairs(main_a1_macro, indicators)
    clean_macro_order = rank_pairs(clean_a1_macro, indicators)
    pair_rows: List[Dict[str, object]] = []
    pair_rows.append({
        "comparison": "a1_macro_full_vs_clean",
        "source_domain": "macro_7_domain",
        "extension_dataset": "",
        "pair_rank_spearman": q12._spearman(
            np.array([main_a1_macro[p] for p in main_macro_order]),
            np.array([clean_a1_macro[p] for p in main_macro_order]),
        ),
        "top10_overlap_n": len(set(main_macro_order[:10]) & set(clean_macro_order[:10])),
        "top10_overlap_rate": len(set(main_macro_order[:10]) & set(clean_macro_order[:10])) / 10.0,
    })
    ext_pair_gaps = {key: pair_gap_map(data[key], indicators) for key in ("a2", "a3")}
    for domain in sorted(clean_gaps):
        full_order = rank_pairs(full_gaps[domain], indicators)
        clean_order = rank_pairs(clean_gaps[domain], indicators)
        overlap_n = len(set(full_order[:10]) & set(clean_order[:10]))
        pair_rows.append({
            "comparison": "a1_same_domain_full_vs_clean",
            "source_domain": domain,
            "extension_dataset": "",
            "pair_rank_spearman": q12._spearman(
                np.array([full_gaps[domain][p] for p in full_order]),
                np.array([clean_gaps[domain][p] for p in full_order]),
            ),
            "top10_overlap_n": overlap_n,
            "top10_overlap_rate": overlap_n / 10.0,
        })

    clean_pair_validation: List[Dict[str, object]] = []
    for ext_key in ("a2", "a3"):
        ext_data = data[ext_key]
        for domain, ext_idx in q12._domain_indices(ext_data.domains).items():
            if domain not in clean_indices_by_domain:
                continue
            ref_idx = clean_indices_by_domain[domain]
            ref_d = q12.conflict_intensity(clean_a1.scores[ref_idx, :])
            ext_d = d_values[ext_key][ext_idx]
            tau = thresholds_clean[domain]
            diff_ci = q12._bootstrap_group_differences(
                ref_d, ext_d, tau, args.validation_bootstrap,
                q12._stable_seed(args.seed, f"unicode-clean-validation-{ext_key}-{domain}"),
            )
            ref_gap = clean_gaps[domain]
            ext_gap = ext_pair_gaps[ext_key][domain]
            keys = sorted(ref_gap)
            rank_rho = q12._spearman(
                np.array([ref_gap[p] for p in keys]), np.array([ext_gap[p] for p in keys])
            )
            ref_top10 = set(rank_pairs(ref_gap, indicators)[:10])
            ext_top10 = set(rank_pairs(ext_gap, indicators)[:10])
            overlap_n = len(ref_top10 & ext_top10)
            clean_pair_validation.append({
                "extension_dataset": ext_key,
                "source_domain": domain,
                "clean_a1_n": int(np.isfinite(ref_d).sum()),
                "extension_n": int(np.isfinite(ext_d).sum()),
                "clean_a1_threshold": tau,
                "clean_threshold_ci_low": threshold_cis_clean[domain][0],
                "clean_threshold_ci_high": threshold_cis_clean[domain][1],
                "median_conflict_diff": float(np.nanmedian(ext_d) - np.nanmedian(ref_d)),
                "median_diff_ci_low": diff_ci["median_diff_low"],
                "median_diff_ci_high": diff_ci["median_diff_high"],
                "clean_a1_high_conflict_rate": float(np.mean(ref_d > tau)),
                "extension_high_conflict_rate": float(np.mean(ext_d > tau)),
                "high_conflict_rate_diff": float(np.mean(ext_d > tau) - np.mean(ref_d > tau)),
                "rate_diff_ci_low": diff_ci["rate_diff_low"],
                "rate_diff_ci_high": diff_ci["rate_diff_high"],
                "pair_rank_spearman": rank_rho,
                "same_domain_top10_overlap_n": overlap_n,
                "same_domain_top10_overlap_rate": overlap_n / 10.0,
            })
            pair_rows.append({
                "comparison": "a1_clean_vs_extension",
                "source_domain": domain,
                "extension_dataset": ext_key,
                "pair_rank_spearman": rank_rho,
                "top10_overlap_n": overlap_n,
                "top10_overlap_rate": overlap_n / 10.0,
            })

    # See whether cleaned-A1 weighting changes the composite score itself.
    primary_delta = float(main_manifest["model_parameters"]["huber_delta"])
    q_change_rows: List[Dict[str, object]] = []
    for key, item in data.items():
        _, qh_primary, _ = q12._score_summary(item.scores, weights_main, primary_delta)
        _, qh_clean, dq_clean = q12._score_summary(item.scores, weights_clean, delta_clean)
        valid = np.isfinite(qh_primary) & np.isfinite(qh_clean)
        change = qh_clean[valid] - qh_primary[valid]
        q_change_rows.append({
            "dataset": key,
            "n": int(valid.sum()),
            "primary_huber_delta": primary_delta,
            "clean_a1_huber_delta": delta_clean,
            "median_abs_qh_change": float(np.median(np.abs(change))) if len(change) else float("nan"),
            "p95_abs_qh_change": q12._quantile(np.abs(change), 0.95),
            "max_abs_qh_change": float(np.max(np.abs(change))) if len(change) else float("nan"),
            "clean_weight_delta_q_median_a1": float(np.nanmedian(dq_clean)) if key == "a1" else float("nan"),
        })

    write_csv(output_dir / "a1_clean_thresholds.csv", threshold_rows)
    write_csv(output_dir / "contamination_domain_sensitivity.csv", domain_rows)
    write_csv(output_dir / "pair_ranking_sensitivity.csv", pair_rows)
    write_csv(output_dir / "clean_a1_indicator_weights.csv", clean_weight_rows)
    write_csv(output_dir / "huber_score_sensitivity.csv", q_change_rows)
    write_csv(output_dir / "clean_a1_external_validation.csv", clean_pair_validation)

    report_lines = [
        "# Q1.2 数据污染敏感性分析",
        "",
        "本分析保留全量冻结 `norm_*` 作为主结果；敏感性版本不修改原始文本或冻结预处理文件，只从 A1 的阈值、指标权重、Huber δ 和指标对排序估计中排除 Unicode 审计命中的记录，再将这些 A1 估计应用到未变更的扩展集。标记只表示字符完整性风险，不等于恶意投毒。",
        "",
        f"本次以题目分析报告中的正式逐记录审计为准，规则‘{audit_rule_description}’命中 {flagged_count} 条；审计表与 A1 冻结原始文件哈希均已校验。报告旧基线 334 条使用的历史规则无法复现，较宽审计规则命中 463 条。规则范围不同导致计数不同，不能解释为新增恶意记录或恶意样本。审计来源为 `../q1_1/text_integrity_summary.json` 与 `../q1_1/text_integrity_audit.csv`。",
        "",
        f"A1 排除后保留 {len(clean_a1.ids):,}/{len(a1.ids):,} 条。A1 稳定性候选权重 L1 变化={weight_l1:.6f}，权重排序 Spearman={q12._format(weight_rank_rho)}；Huber δ 从 {primary_delta:.6f} 变为 {delta_clean:.6f}（{method_clean}）。",
        "",
        "## 按 A1 来源域的阈值影响",
        "",
        "完整阈值与清洁子集阈值都基于同一冻结 `norm_*` 坐标；清洁子集阈值区间以该子集内 B_C 次 bootstrap 估计。具体计数、阈值和被排除行的超阈比例见 `a1_clean_thresholds.csv` 与 `contamination_domain_sensitivity.csv`。",
        "",
        "## A1 指标对排序影响",
        "",
        "`pair_ranking_sensitivity.csv` 比较全 A1 与排除标记行后的 A1 排序，并用清洁 A1 同域排序复核 A2/A3。所有 Top-10 重合分母为 10；这些比较是描述统计，不构造 231 对的独立显著性区间。",
        "",
        "## Q1.1 综合分候选影响",
        "",
        "`huber_score_sensitivity.csv` 汇总按清洁 A1 重新估计权重和 δ 后，对同一冻结输入样本的 Huber 分变化。它只表示校准敏感性，不表示清洁候选比主分析更有效；最终 Q 仍需盲评和 Q1.1 的预定义选择规则。",
        "",
        "## 边界",
        "",
        "该分析检验的是记录级影响，不重算上游各评分器的原始输出，也不能确定不可见字符是否已经影响这些预计算指标。由于公共预处理明示没有把 `content` 直接作为指标，本分析保留冻结主结果，并将排除标记记录的结果作为风险敏感性，不作全体 JSON 被恶意投毒的断言。",
        "",
    ]
    report_path = output_dir / "contamination_sensitivity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    output_files = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "manifest.json")
    output_hashes = {path.name: sha256_file(path) for path in output_files}
    run_manifest = {
        "schema_version": "q1-2-contamination-sensitivity-v1",
        "status": "full_sensitivity",
        "primary_q1_2_manifest_sha256": sha256_file(main_manifest_path),
        "preprocess_manifest_sha256": sha256_file(preprocess_manifest_path),
        "verified_raw_a1_source_sha256": raw_a1_actual_sha256,
        "unicode_audit_manifest_sha256": sha256_file(audit_manifest_path),
        "unicode_audit_summary_sha256": sha256_file(audit_manifest_path),
        "unicode_record_audit_sha256": sha256_file(audit_file_path),
        "preprocessed_input_sha256": input_hashes,
        "records_excluded_from_a1_q1_2_calibration": flagged_count,
        "records_retained_in_a1_sensitivity": int(len(clean_a1.ids)),
        "unicode_flag_rule": audit_rule_description,
        "bootstrap": {
            "threshold_repetitions": args.threshold_bootstrap,
            "external_validation_repetitions": args.validation_bootstrap,
            "weight_repetitions": args.weight_bootstrap,
            "seed": args.seed,
            "ci": "95% percentile",
        },
        "clean_a1_huber_scale_method": method_clean,
        "clean_a1_huber_delta": delta_clean,
        "clean_a1_huber_scale": scale_clean,
        "weight_l1_change": weight_l1,
        "weight_rank_spearman": weight_rank_rho,
        "runtime_seconds": round(time.time() - started, 3),
        "outputs_sha256": output_hashes,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "status": run_manifest["status"],
        "output_dir": str(output_dir),
        "flagged_rows": flagged_count,
        "clean_a1_rows": int(len(clean_a1.ids)),
        "weight_l1_change": weight_l1,
        "weight_rank_spearman": weight_rank_rho,
        "primary_delta": primary_delta,
        "clean_delta": delta_clean,
        "clean_external_validation": clean_pair_validation,
        "elapsed_seconds": run_manifest["runtime_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocess-dir", default=str(DEFAULT_PREPROCESS_DIR))
    parser.add_argument("--main-dir", default=str(DEFAULT_MAIN_DIR))
    parser.add_argument("--audit-dir", default=str(DEFAULT_AUDIT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit-rule", choices=("cf_cc", "broad"), default="cf_cc")
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--threshold-bootstrap", type=int, default=2000)
    parser.add_argument("--validation-bootstrap", type=int, default=2000)
    parser.add_argument("--weight-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    for name in ("threshold_bootstrap", "validation_bootstrap", "weight_bootstrap"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} 至少为 2")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
