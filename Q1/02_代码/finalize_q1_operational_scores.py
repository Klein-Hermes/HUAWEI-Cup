#!/usr/bin/env python3
"""Freeze the existing Q1.1 equal-weight score as the operational primary Q.

This script does not recompute scores or fit models. It exports and audits the
existing Q1.1/Q1.2/Q1.3 evidence into a versioned closeout bundle.
Run from the repository root: python src/finalize_q1_operational_scores.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
Q11 = ROOT / "results/q1_1/v1"
Q12 = ROOT / "results/q1_2_conflict_model/v1"
HUMAN = ROOT / "Q1/03_结果/Q1.1/human_spotcheck_final/received_20260924"
Q13_PQ = ROOT / "results/q1_3/pq_extension_v1"
Q13_PONLY_REPORT = ROOT / "Q1/03_结果/Q1.3/v1/q1_3_run_report.md"
OUT = ROOT / "results/q1_1/operational_final_v1"
MIRROR = ROOT / "Q1/03_结果/Q1.1/operational_final_v1"

EXPECTED_DATASET_ROWS = {"a1": 51230, "a2": 17523, "a3": 203752}
EXPECTED_DOMAIN_ROWS = {
    ("a1", "arxiv"): 1419,
    ("a1", "book"): 171,
    ("a1", "c4"): 10000,
    ("a1", "commoncrawl"): 9640,
    ("a1", "github"): 10000,
    ("a1", "stackexchange"): 10000,
    ("a1", "wikipedia"): 10000,
    ("a2", "arxiv"): 17523,
    ("a3", "github"): 203752,
}
Q13_DIRECT_NEAR_DOMAINS = {"arxiv", "book", "commoncrawl", "github", "stackexchange", "wikipedia"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path, compressed: bool = False) -> List[dict]:
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_bytes(rows: Sequence[dict], columns: Sequence[str]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(csv_bytes(rows, columns))


def write_deterministic_gzip_csv(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = csv_bytes(rows, columns)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            gz.write(payload)


def json_write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_score(value: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0 or x > 1:
        raise ValueError(f"Q score outside [0,1] or non-finite: {value}")
    return x


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    sample_path = Q11 / "sample_scores.csv.gz"
    source_rows = read_csv(sample_path, compressed=True)
    if not source_rows:
        raise ValueError("Q1.1 sample score source is empty")
    input_columns = list(source_rows[0].keys())
    required_columns = {
        "dataset", "source_domain", "id", "sub_path", "q_equal", "q_huber",
        "n_valid_indicators", "indicator_coverage", "weighted_coverage",
        "outlier_indicator_count", "unit_suspect_indicator_count",
        "unicode_cf_cc_flag", "unicode_broad_flag",
    }
    if not required_columns.issubset(input_columns):
        raise ValueError(f"Unexpected Q1.1 score schema; missing {sorted(required_columns - set(input_columns))}")

    counts = defaultdict(int)
    seen_keys = set()
    coverage = defaultdict(lambda: {
        "n_records": 0, "n_scored_main": 0, "n_scored_huber": 0,
        "n_missing_main": 0, "n_missing_huber": 0,
        "n_valid_min": 10**9, "n_valid_max": -1, "n_valid_sum": 0,
        "mean_indicator_coverage_sum": 0.0, "n_unit_suspect_records": 0,
        "n_unicode_cf_cc_records": 0, "n_unicode_broad_records": 0,
    })
    operational_rows = []
    for row in source_rows:
        dataset, domain = row["dataset"], row["source_domain"]
        key = (dataset, domain, row["id"], row["sub_path"])
        if key in seen_keys:
            raise ValueError(f"Duplicate sample key: {key}")
        seen_keys.add(key)
        qe = finite_score(row["q_equal"])
        qh = finite_score(row["q_huber"])
        n_valid = int(row["n_valid_indicators"])
        if n_valid < 0 or n_valid > 22:
            raise ValueError(f"Invalid indicator count {n_valid} at {key}")
        g = coverage[(dataset, domain)]
        g["n_records"] += 1
        g["n_scored_main"] += 1
        g["n_scored_huber"] += 1
        g["n_valid_min"] = min(g["n_valid_min"], n_valid)
        g["n_valid_max"] = max(g["n_valid_max"], n_valid)
        g["n_valid_sum"] += n_valid
        g["mean_indicator_coverage_sum"] += float(row["indicator_coverage"])
        g["n_unit_suspect_records"] += int(float(row["unit_suspect_indicator_count"]) > 0)
        g["n_unicode_cf_cc_records"] += int(float(row["unicode_cf_cc_flag"]) > 0)
        g["n_unicode_broad_records"] += int(float(row["unicode_broad_flag"]) > 0)
        counts[dataset] += 1
        out = dict(row)
        out["q_operational_main"] = f"{qe:.17g}"
        out["q_huber_sensitivity"] = f"{qh:.17g}"
        out["selected_candidate"] = "q_equal"
        out["score_role"] = "operational_primary"
        operational_rows.append(out)

    if counts != EXPECTED_DATASET_ROWS:
        raise ValueError(f"Unexpected dataset row counts: {dict(counts)}")
    observed_domain_rows = {(d, s): v["n_records"] for (d, s), v in coverage.items()}
    if observed_domain_rows != EXPECTED_DOMAIN_ROWS:
        raise ValueError(f"Unexpected dataset/domain coverage: {observed_domain_rows}")
    if len(seen_keys) != sum(EXPECTED_DATASET_ROWS.values()):
        raise ValueError("Sample key uniqueness/count assertion failed")

    sample_out_cols = input_columns + ["q_operational_main", "q_huber_sensitivity", "selected_candidate", "score_role"]
    sample_out = OUT / "sample_scores_operational.csv.gz"
    write_deterministic_gzip_csv(sample_out, operational_rows, sample_out_cols)

    domain_source = read_csv(Q11 / "domain_summary.csv")
    domain_ci_source = read_csv(Q11 / "domain_confidence_intervals.csv")
    ci_map = {(r["dataset"], r["source_domain"], r["candidate"]): r for r in domain_ci_source}
    domain_rows = []
    seen_domain_keys = set()
    for row in domain_source:
        if row["candidate"] not in {"q_equal", "q_huber"}:
            continue
        key = (row["dataset"], row["source_domain"], row["candidate"])
        if key in seen_domain_keys:
            raise ValueError(f"Duplicate domain summary key: {key}")
        seen_domain_keys.add(key)
        ci = ci_map.get(key)
        if ci is None:
            raise ValueError(f"No confidence interval for domain score: {key}")
        domain_rows.append({
            "dataset": row["dataset"], "source_domain": row["source_domain"],
            "candidate": row["candidate"],
            "score_role": "operational_primary" if row["candidate"] == "q_equal" else "sensitivity",
            "n_records": row["n_records"], "n_scored": row["n_scored"],
            "scored_rate": row["scored_rate"], "domain_score": row["domain_score"],
            "mean_sample_score": row["mean_sample_score"], "median_sample_score": row["median_sample_score"],
            "mean_indicator_coverage": row["mean_indicator_coverage"],
            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "bootstrap_repetitions": ci["bootstrap_repetitions"],
            "ci_method": ci["ci_method"],
        })
    if len(domain_rows) != 18:
        raise ValueError(f"Expected 18 candidate/domain scores, got {len(domain_rows)}")
    domain_cols = [
        "dataset", "source_domain", "candidate", "score_role", "n_records", "n_scored",
        "scored_rate", "domain_score", "mean_sample_score", "median_sample_score",
        "mean_indicator_coverage", "ci_low", "ci_high", "bootstrap_repetitions", "ci_method",
    ]
    write_csv(OUT / "domain_scores_operational.csv", domain_rows, domain_cols)

    macros = read_csv(Q11 / "a1_macro_summary.csv")
    domain_lookup = {(r["dataset"], r["source_domain"], r["candidate"]): r for r in domain_rows}
    dataset_rows = []
    for candidate in ("q_equal", "q_huber"):
        macro = next(r for r in macros if r["dataset"] == "a1" and r["candidate"] == candidate)
        dataset_rows.append({
            "dataset": "a1", "candidate": candidate,
            "score_role": "operational_primary" if candidate == "q_equal" else "sensitivity",
            "n_records": sum(EXPECTED_DOMAIN_ROWS[("a1", d)] for d in ["arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"]),
            "n_domains": macro["n_domains"], "q_dataset_score": macro["macro_domain_score"],
            "ci_low": macro["ci_low"], "ci_high": macro["ci_high"],
            "summary_method": macro["definition"],
        })
        for dataset, domain in (("a2", "arxiv"), ("a3", "github")):
            row = domain_lookup[(dataset, domain, candidate)]
            dataset_rows.append({
                "dataset": dataset, "candidate": candidate,
                "score_role": "operational_primary" if candidate == "q_equal" else "sensitivity",
                "n_records": row["n_records"], "n_domains": 1,
                "q_dataset_score": row["domain_score"], "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "summary_method": f"single-domain Huber location: {domain}",
            })
    dataset_cols = ["dataset", "candidate", "score_role", "n_records", "n_domains", "q_dataset_score", "ci_low", "ci_high", "summary_method"]
    write_csv(OUT / "dataset_summary_operational.csv", dataset_rows, dataset_cols)

    coverage_rows = []
    for (dataset, domain), g in sorted(coverage.items()):
        n = g["n_records"]
        required = dataset == "a1" or (dataset, domain) in {("a2", "arxiv"), ("a3", "github")}
        coverage_rows.append({
            "dataset": dataset, "source_domain": domain,
            "q1_1_required_domain": str(required).lower(),
            "q1_3_direct_or_near_mapping": str(domain in Q13_DIRECT_NEAR_DOMAINS and dataset in {"a1", "a2", "a3"}).lower(),
            "n_records": n, "n_scored_main": g["n_scored_main"], "n_missing_main": g["n_missing_main"],
            "n_scored_huber": g["n_scored_huber"], "n_missing_huber": g["n_missing_huber"],
            "scored_rate_main": g["n_scored_main"] / n, "scored_rate_huber": g["n_scored_huber"] / n,
            "n_valid_indicators_min": g["n_valid_min"], "n_valid_indicators_max": g["n_valid_max"],
            "mean_n_valid_indicators": g["n_valid_sum"] / n,
            "mean_indicator_coverage": g["mean_indicator_coverage_sum"] / n,
            "n_unit_suspect_records": g["n_unit_suspect_records"],
            "n_unicode_cf_cc_records": g["n_unicode_cf_cc_records"],
            "n_unicode_broad_records": g["n_unicode_broad_records"],
        })
    if any(r["n_missing_main"] != 0 or r["n_missing_huber"] != 0 for r in coverage_rows):
        raise ValueError("Q score coverage is incomplete")
    coverage_cols = list(coverage_rows[0].keys())
    write_csv(OUT / "score_coverage.csv", coverage_rows, coverage_cols)

    # Confirm that Q1.2 consumed the exact same sample-level candidate scores.
    q12_rows = read_csv(Q12 / "sample_scores.csv.gz", compressed=True)
    q12_by_key = {(r["dataset"], r["source_domain"], r["id"]): r for r in q12_rows}
    if len(q12_by_key) != len(q12_rows) or len(q12_rows) != len(source_rows):
        raise ValueError("Q1.2 sample score keys/count do not match Q1.1")
    max_abs_equal = 0.0
    max_abs_huber = 0.0
    for row in source_rows:
        peer = q12_by_key[(row["dataset"], row["source_domain"], row["id"])]
        max_abs_equal = max(max_abs_equal, abs(float(row["q_equal"]) - float(peer["q_equal_mean"])))
        max_abs_huber = max(max_abs_huber, abs(float(row["q_huber"]) - float(peer["q_huber"])))
    if max_abs_equal > 1e-12 or max_abs_huber > 1e-12:
        raise ValueError(f"Q1.2 candidate parity check failed: equal={max_abs_equal}, huber={max_abs_huber}")

    human_metrics = {r["metric"]: r["value"] for r in read_csv(HUMAN / "human_check_metrics.csv")}
    paired = {r["metric"]: r for r in read_csv(HUMAN / "paired_bootstrap.csv")}
    human_summary = (HUMAN / "summary.md").read_text(encoding="utf-8")
    human_manifest = json.loads((HUMAN / "decision_status.json").read_text(encoding="utf-8"))
    if human_manifest["n_total"] != 30 or human_manifest["candidate_selected"] is not None:
        raise ValueError("Human review status changed from the recorded 30-row non-selection")
    if "顺序独立随机化未满足" not in human_summary:
        raise ValueError("Expected human review order deviation is not documented")

    rank_sensitivity = read_csv(Q11 / "domain_rank_sensitivity.csv")
    rank_by_candidate = {c: {} for c in ("q_equal", "q_huber")}
    for row in rank_sensitivity:
        rank_by_candidate[row["candidate"]][row["scenario"]] = float(row["seven_domain_rank_spearman"])
    if len(rank_by_candidate["q_equal"]) != 10 or set(rank_by_candidate) - {"q_equal", "q_huber"}:
        raise ValueError("Unexpected Q1.1 ranking sensitivity scenarios")
    rank_one_counts = {
        candidate: sum(abs(rho - 1.0) < 1e-12 for rho in scenarios.values())
        for candidate, scenarios in rank_by_candidate.items()
    }
    conflict_domains = [r for r in read_csv(Q12 / "domain_summary.csv") if r["dataset"] == "a1"]
    if len(conflict_domains) != 7:
        raise ValueError("Expected seven A1 Q1.2 conflict summary domains")
    mean_abs_delta_high = sum(float(r["median_abs_delta_q_high_conflict"]) for r in conflict_domains) / len(conflict_domains)
    mean_abs_delta_other = sum(float(r["median_abs_delta_q_other"]) for r in conflict_domains) / len(conflict_domains)
    dataset_scores = {(r["dataset"], r["candidate"]): r for r in dataset_rows}

    decision = {
        "schema_version": "q1-operational-q-decision-v1",
        "status": "operational_primary_frozen_with_limited_validation",
        "decision_date": "2026-09-24",
        "scope": "Q1.1 operational use across A1/A2/A3; not a claim of ground-truth quality or poisoning removal",
        "selected_candidate": "q_equal",
        "operational_score_column": "q_operational_main",
        "sensitivity_candidate": "q_huber",
        "sensitivity_score_column": "q_huber_sensitivity",
        "decision_basis": [
            f"Q1.1 domain-rank sensitivity retained the complete seven-domain A1 ranking in {rank_one_counts['q_equal']}/10 scenarios for q_equal and {rank_one_counts['q_huber']}/10 for q_huber; excluding the seven unresolved frac fields lowers rank Spearman to {rank_by_candidate['q_equal']['exclude_7_unit_ambiguous_fields']:.4f} and {rank_by_candidate['q_huber']['exclude_7_unit_ambiguous_fields']:.4f}, so this is relative stability evidence rather than immunity to the flagged fields.",
            f"Q1.2 conflict diagnostics show the mean across seven A1 domains of median |q_huber-q_equal| is {mean_abs_delta_high:.4f} among high-conflict records and {mean_abs_delta_other:.4f} among other records. This establishes Huber as a useful conflict-sensitivity comparison, not as a more correct score.",
            f"A1/A2/A3 score comparisons do not show uniform candidate dominance: q_equal/q_huber are {float(dataset_scores[('a1','q_equal')]['q_dataset_score']):.6f}/{float(dataset_scores[('a1','q_huber')]['q_dataset_score']):.6f} for A1, {float(dataset_scores[('a2','q_equal')]['q_dataset_score']):.6f}/{float(dataset_scores[('a2','q_huber')]['q_dataset_score']):.6f} for A2/arxiv, and {float(dataset_scores[('a3','q_equal')]['q_dataset_score']):.6f}/{float(dataset_scores[('a3','q_huber')]['q_dataset_score']):.6f} for A3/github; the direction reverses across A2 and A3, and both same-domain transfer intervals include zero.",
            "The two candidates tie on the 24 representative cases: Spearman 0.566692 each; paired-case bootstrap difference CI [-0.058987, 0.062315] crosses zero.",
            "The 30 paired human forms are a micro-sample; they do not validate all 793 review candidates or establish a general winner. They used Chinese machine-translation-assisted text, and returned reviewer order was not independently randomized.",
            "Taken together, the sensitivity ranking favors q_equal modestly, while Q1.2 and A1/A2/A3 score levels do not establish that either candidate is more correct. With no human or transfer winner, q_equal is the simpler transparent operational baseline; q_huber remains the sensitivity alternative.",
        ],
        "evidence_summary": {
            "q1_1_a1_domain_rank_sensitivity_scenarios": 10,
            "q_equal_rank_spearman_one_scenarios": rank_one_counts["q_equal"],
            "q_huber_rank_spearman_one_scenarios": rank_one_counts["q_huber"],
            "exclude_7_frac_fields_rank_spearman": {
                "q_equal": rank_by_candidate["q_equal"]["exclude_7_unit_ambiguous_fields"],
                "q_huber": rank_by_candidate["q_huber"]["exclude_7_unit_ambiguous_fields"],
            },
            "q1_2_mean_domain_median_abs_candidate_difference": {
                "high_conflict": mean_abs_delta_high,
                "other": mean_abs_delta_other,
            },
            "dataset_domain_scores": {
                f"{dataset}/{candidate}": float(row["q_dataset_score"])
                for (dataset, candidate), row in dataset_scores.items()
            },
        },
        "human_review_status": {
            "n_total": 30,
            "n_representative_used_for_candidate_correlation": 24,
            "n_diagnostic": 6,
            "candidate_selected_by_human_review": None,
            "reviewer_order_independent_randomization_met": False,
            "machine_translation_assisted": True,
            "overall_quality_quadratic_weighted_kappa": human_metrics["quadratic_weighted_cohen_kappa_overall_quality"],
            "overall_quality_spearman_between_reviewers": human_metrics["spearman_overall_quality"],
            "q_equal_vs_consensus_spearman": paired["spearman_q_equal_vs_consensus"]["point_estimate"],
            "q_huber_vs_consensus_spearman": paired["spearman_q_huber_vs_consensus"]["point_estimate"],
            "delta_huber_minus_equal_ci95": [paired["delta_huber_minus_equal"]["ci_95_low"], paired["delta_huber_minus_equal"]["ci_95_high"]],
        },
        "scope_limits": [
            "Do not describe the 30 rows as full 793-row human validation.",
            "Seven rps_*frac* indicator meanings/units remain unresolved; no manual unit conversion was applied.",
            "Source and Unicode contamination flags are preserved and prior contamination sensitivities remain; this does not prove all poisoning has been removed.",
            "A2/A3 original-text mapping remains incomplete; no claim of broad original-text validation is made.",
            "The selected score is an operational primary for the Q1 analysis, not a causal measure or a proven uniquely correct quality score.",
        ],
        "human_analysis_decision_status_file": "../human_spotcheck_final/received_20260924/decision_status.json",
        "human_analysis_candidate_selected": None,
    }
    json_write(OUT / "decision_status.json", decision)

    score_by_candidate = {r["candidate"]: r for r in dataset_rows if r["dataset"] == "a1"}
    a1_equal = score_by_candidate["q_equal"]
    a1_huber = score_by_candidate["q_huber"]
    report = f"""# Q1 实验收口报告（2026-09-24）

## 收口决定

Q1.1 的操作主分冻结为 `q_equal`，导出列名为 `q_operational_main`；`q_huber` 保留为敏感性对照，导出列名为 `q_huber_sensitivity`。这是在现有证据无法区分两候选时，采用假设更少的等权基线作操作决策；**不是 30 条人工评审判定 q_equal 胜出**，也不是声称它是无误差的真实质量。

取舍综合了四类证据：Q1.1 的 A1 七领域排序敏感性中，q_equal 在 10 个情景中的 9 个保持基准排序不变，q_huber 为 6/10；排除 7 个未核实 frac 字段后，两者的排序 Spearman 分别降为 0.6786 和 0.7500。Q1.2 中高冲突样本的候选差异更大（七领域内中位数差的平均值 {mean_abs_delta_high:.6f}，其他样本 {mean_abs_delta_other:.6f}），说明 Huber 是有用的冲突敏感性对照，但不能证明其更正确。A1/A2/A3 的候选分方向并不一致：A2/arxiv q_equal/q_huber = {float(dataset_scores[('a2','q_equal')]['q_dataset_score']):.6f}/{float(dataset_scores[('a2','q_huber')]['q_dataset_score']):.6f}，A3/github = {float(dataset_scores[('a3','q_equal')]['q_dataset_score']):.6f}/{float(dataset_scores[('a3','q_huber')]['q_dataset_score']):.6f}；同域迁移差均近零且区间跨 0。30 条评审也没有区分候选。因此按更简单透明的基线选择 q_equal 为操作主分，并保留 q_huber 作敏感性分析。

A1 七领域等权域级汇总：q_equal = {float(a1_equal['q_dataset_score']):.6f}（95% CI [{float(a1_equal['ci_low']):.6f}, {float(a1_equal['ci_high']):.6f}]）；q_huber = {float(a1_huber['q_dataset_score']):.6f}（95% CI [{float(a1_huber['ci_low']):.6f}, {float(a1_huber['ci_high']):.6f}]）。A2/arxiv 与 A3/github 的单领域分数、区间和每条样本分数见 `dataset_summary_operational.csv`、`domain_scores_operational.csv` 和 `sample_scores_operational.csv.gz`。

## 证据边界

- 两位评审各评 30 条，整体质量二次加权 kappa = {float(human_metrics['quadratic_weighted_cohen_kappa_overall_quality']):.6f}，评审间 Spearman = {float(human_metrics['spearman_overall_quality']):.6f}。候选相关只基于 24 条 representative：q_equal 与 q_huber 均为 0.566692；差值区间 [{float(paired['delta_huber_minus_equal']['ci_95_low']):.6f}, {float(paired['delta_huber_minus_equal']['ci_95_high']):.6f}] 跨 0。它是微型小样本核查，不能外推成 793 条全量验证。
- 评审使用中文机译辅助文本；两份表行序相同，独立随机顺序未满足。六条 diagnostic 是案例材料，不并入候选总体相关。
- Q1.1 在 10 个 A1 排序敏感性情景中 q_equal 有 9 个保持基准排序、q_huber 有 6 个；7 个 frac 字段排除情景下分别为 0.6786 和 0.7500。Q1.2 高冲突记录中的候选差异大于其他记录，但只说明 Huber 对冲突更敏感，不判定其更正确。
- A1/A2/A3 候选域级结果方向不统一（表中列值），A2 与 A3 方向相反；两个候选的同域迁移差值区间都跨 0。因此跨数据集表现不能选出普遍优胜者。
- A1 有 7 个领域；A2/arxiv 与 A3/github 也都有域级 Q。所有 272,505 条样本均有 q_equal 和 q_huber，覆盖审计保留指标覆盖、单位疑点及 Unicode 标记。Q1.3 direct/near 的 6 个领域（arxiv、book、commoncrawl、github、stackexchange、wikipedia）均有 A1 评分；A2 arxiv、A3 github 按同领域比较输出。
- 七个 `rps_*frac*` 字段仍有语义/单位未完全核实的问题；未手动换算。数据源和 Unicode 污染标记保留，已有严格/宽松污染敏感性分析，不据此声称投毒已被彻底清除。
- A2/A3 同领域迁移差值均接近 0 且区间跨 0。q_equal 在两个迁移差值上都略接近 0，但这不足以声称显著优于 q_huber。

## Q1.2 与 Q1.3 收口

- Q1.2 复用 Q1.1 候选分做冲突诊断，不另造 Q，也不与 Q1.1 做算术合并。其价值是展示高冲突记录中候选分差异更明显；它不是选择赢家的证据。
- p-only 保持 Q1.3 主模型。既有 A4/A5 分组嵌套验证的主模型为 `simplex_ridge`，标准化 RMSE 为 0.633048；A6–A11 冻结评估已生成，不重跑。A12–A15 是估计表一致性核对，不是实际测试准确率。
- p+Q 受限扩展已比较现有 q_equal/q_huber 情景，不重跑。`Q_mix = p^T q` 在线性 p-only 中无独立秩增量；`Q_covered(p)` 是配比的确定性非线性派生特征。已有比较在 A4/A5 与 1M、1B 上未见普遍改善，60M 出现局部改善，故 p+Q 只作为探索性结果，不能宣称 Q 带来独立质量效应或稳定预测收益。

## 冻结交付文件

- `sample_scores_operational.csv.gz`：272,505 条样本分数，包含主分、Huber 敏感性分和污染审计标记。
- `domain_scores_operational.csv`、`dataset_summary_operational.csv`：A1/A2/A3 域级和数据集级结果及既有区间。
- `score_coverage.csv`：各数据集/领域的计分覆盖和数据质量标记计数。
- `decision_status.json`：操作主分选择及证据限制。
- `manifest.json`：来源和交付物哈希、行数/覆盖审计与 Q1.2 分数一致性核对。

交付副本位于 `Q1/03_结果/Q1.1/operational_final_v1/`。本轮只整理冻结既有计算结果和决策，不重算候选评分、不增加人工评审、不重跑 Q1.2 或 Q1.3 模型，也未修改 `题目分析报告.md`。
"""
    (OUT / "summary.md").write_text(report, encoding="utf-8")

    input_paths = [
        Q11 / "sample_scores.csv.gz", Q11 / "domain_summary.csv", Q11 / "domain_confidence_intervals.csv",
        Q11 / "a1_macro_summary.csv", Q11 / "same_domain_differences.csv", Q11 / "manifest.json",
        Q12 / "sample_scores.csv.gz", Q12 / "manifest.json", HUMAN / "summary.md",
        HUMAN / "decision_status.json", HUMAN / "human_check_metrics.csv",
        HUMAN / "candidate_human_comparison.csv", HUMAN / "paired_bootstrap.csv",
        Q13_PQ / "manifest.json", Q13_PQ / "q1_3_pq_extension_report.md", Q13_PONLY_REPORT,
        Path(__file__),
    ]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in input_paths}
    output_names = [
        "sample_scores_operational.csv.gz", "domain_scores_operational.csv",
        "dataset_summary_operational.csv", "score_coverage.csv", "decision_status.json", "summary.md",
    ]
    output_hashes = {name: sha256(OUT / name) for name in output_names}
    manifest = {
        "schema_version": "q1-operational-q-bundle-v1",
        "status": "PASS",
        "generated_date": "2026-09-24",
        "operation": "export_existing_q1_1_candidate_scores_and_freeze_operational_primary",
        "no_model_recalculation": True,
        "selected_candidate": "q_equal",
        "sensitivity_candidate": "q_huber",
        "source_sample_count": len(source_rows),
        "source_rows_by_dataset": dict(sorted(counts.items())),
        "source_rows_by_dataset_domain": {f"{d}/{s}": n for (d, s), n in sorted(observed_domain_rows.items())},
        "required_domain_coverage": {
            "a1_domains": 7,
            "a2_domains": ["arxiv"],
            "a3_domains": ["github"],
            "all_required_groups_scored": True,
        },
        "sample_score_integrity": {
            "unique_dataset_domain_id_subpath_keys": len(seen_keys),
            "candidate_scores_in_unit_interval": True,
            "n_valid_indicators_min": min(int(r["n_valid_indicators"]) for r in source_rows),
            "n_valid_indicators_max": max(int(r["n_valid_indicators"]) for r in source_rows),
            "missing_q_equal": 0,
            "missing_q_huber": 0,
        },
        "q1_2_candidate_score_parity": {
            "matched_rows": len(source_rows), "max_abs_q_equal_difference": max_abs_equal,
            "max_abs_q_huber_difference": max_abs_huber, "tolerance": 1e-12,
        },
        "human_review_scope": {
            "n_reviewed": 30, "n_representative_for_candidate_comparison": 24,
            "reviewer_candidate_selected": None, "machine_translation_assisted": True,
            "reviewer_order_independently_randomized": False,
        },
        "source_sha256": source_hashes,
        "outputs_sha256": output_hashes,
    }
    json_write(OUT / "manifest.json", manifest)

    # Keep Q1's organized delivery copy byte-identical to the computation bundle.
    MIRROR.mkdir(parents=True, exist_ok=True)
    for artifact in OUT.iterdir():
        if artifact.is_file():
            shutil.copyfile(artifact, MIRROR / artifact.name)

    # Finalization integrity checks: generated files, mirror, and declared hashes.
    for name, digest in output_hashes.items():
        if sha256(OUT / name) != digest or sha256(MIRROR / name) != digest:
            raise ValueError(f"Output hash verification failed for {name}")
    if (OUT / "manifest.json").read_bytes() != (MIRROR / "manifest.json").read_bytes():
        raise ValueError("Mirrored manifest differs from canonical manifest")
    if (OUT / "summary.md").read_bytes() != (MIRROR / "summary.md").read_bytes():
        raise ValueError("Mirrored summary differs from canonical summary")

    print(json.dumps({
        "status": "PASS",
        "source_rows": len(source_rows),
        "rows_by_dataset": dict(counts),
        "domain_groups": len(coverage_rows),
        "q1_2_max_abs_diff": {"q_equal": max_abs_equal, "q_huber": max_abs_huber},
        "canonical_output": str(OUT),
        "mirror_output": str(MIRROR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
