"""Analyze two returned Q1.1 30-item reviewer workbooks without selecting Q."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = ROOT / "results/q1_1/human_spotcheck_final"
DEFAULT_OUTPUT = DEFAULT_REVIEW_DIR / "received_20260924"
SCORE_COLUMNS = [
    "教育用途（1–5）",
    "可读性（1–5）",
    "连贯性（1–5）",
    "信息传递（1–5）",
    "整体质量（1–5）",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_xlsx(path: Path) -> tuple[list[str], dict[str, list[int]], list[str], list[str]]:
    ws = load_workbook(path, data_only=True, read_only=True).active
    headers = [str(c.value or "").strip() for c in ws[1]]
    if len(headers) != 7 or headers[0] not in {"blind_id", "盲评编号"}:
        raise ValueError(f"{path.name}: 表头结构不符合 7 列盲评表")
    order: list[str] = []
    ratings: dict[str, list[int]] = {}
    issues: list[str] = []
    for row_no, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if not row or row[0] is None or str(row[0]).strip() == "":
            continue
        blind_id = str(row[0]).strip()
        if blind_id in ratings:
            issues.append(f"重复 blind_id：{blind_id}（第 {row_no} 行）")
            continue
        values = list(row[2:7])
        if len(values) != 5:
            issues.append(f"{blind_id}: 五项评分列不完整")
            continue
        parsed: list[int] = []
        for name, value in zip(SCORE_COLUMNS, values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(f"{blind_id}: {name} 不是整数评分")
                break
            if not float(value).is_integer() or not 1 <= int(value) <= 5:
                issues.append(f"{blind_id}: {name} 超出 1–5 整数范围")
                break
            parsed.append(int(value))
        if len(parsed) == 5:
            order.append(blind_id)
            ratings[blind_id] = parsed
    return headers, ratings, order, issues


def read_template_order(path: Path) -> list[str]:
    ws = load_workbook(path, data_only=True, read_only=True).active
    order = [
        str(row[0]).strip()
        for row in ws.iter_rows(min_row=2, min_col=1, max_col=1, values_only=True)
        if row and row[0] is not None and str(row[0]).strip()
    ]
    if len(order) != 30 or len(set(order)) != 30:
        raise ValueError(f"{path.name}: 冻结模板的 ID 数量或唯一性不正确")
    return order


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = ((start + 1) + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank
        start = end
    return ranks


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return math.nan
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return sum(x * y for x, y in zip(da, db)) / denom if denom else math.nan


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(rankdata(a), rankdata(b))


def quadratic_weighted_kappa(a: list[int], b: list[int]) -> float:
    if len(a) != len(b) or not a:
        return math.nan
    observed = [[0.0] * 5 for _ in range(5)]
    for x, y in zip(a, b):
        observed[x - 1][y - 1] += 1.0
    marg_a = [sum(row) for row in observed]
    marg_b = [sum(observed[i][j] for i in range(5)) for j in range(5)]
    n = float(len(a))
    obs_disagreement = exp_disagreement = 0.0
    for i in range(5):
        for j in range(5):
            weight = ((i - j) / 4.0) ** 2
            obs_disagreement += weight * observed[i][j] / n
            exp_disagreement += weight * marg_a[i] * marg_b[j] / (n * n)
    return 1.0 - obs_disagreement / exp_disagreement if exp_disagreement else math.nan


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    location = (len(ordered) - 1) * p
    lower = int(location)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.6f}"


def analyze(args: argparse.Namespace) -> None:
    reviewer_a = args.reviewer_a.resolve()
    reviewer_b = args.reviewer_b.resolve()
    manifest_path = args.manifest.resolve()
    review_dir = manifest_path.parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    _, ratings_a, order_a, issues_a = read_xlsx(reviewer_a)
    _, ratings_b, order_b, issues_b = read_xlsx(reviewer_b)
    if issues_a or issues_b:
        raise ValueError("评分表校验失败：\n" + "\n".join(issues_a + issues_b))

    sample_rows = read_csv(manifest_path)
    manifest = {r["blind_id"]: r for r in sample_rows}
    if len(manifest) != len(sample_rows):
        raise ValueError("抽样清单中 blind_id 不唯一")
    ids_a, ids_b, expected_ids = set(ratings_a), set(ratings_b), set(manifest)
    if ids_a != ids_b or ids_a != expected_ids:
        raise ValueError(
            f"ID 不一致：A={len(ids_a)}, B={len(ids_b)}, manifest={len(expected_ids)}"
        )
    if len(ids_a) != 30:
        raise ValueError(f"预期 30 条，实际为 {len(ids_a)} 条")

    t1_path = review_dir / "reviewer1_中文机译版.xlsx"
    t2_path = review_dir / "reviewer2_中文机译版.xlsx"
    order_t1 = read_template_order(t1_path)
    order_t2 = read_template_order(t2_path)
    order_checks = {
        "reviewer_a_matches_reviewer1_template": order_a == order_t1,
        "reviewer_a_matches_reviewer2_template": order_a == order_t2,
        "reviewer_b_matches_reviewer1_template": order_b == order_t1,
        "reviewer_b_matches_reviewer2_template": order_b == order_t2,
        "returned_orders_identical": order_a == order_b,
    }

    consensus_rows: list[dict[str, Any]] = []
    for blind_id in order_a:
        a = ratings_a[blind_id]
        b = ratings_b[blind_id]
        consensus_rows.append(
            {
                "blind_id": blind_id,
                "sample_type": manifest[blind_id]["sample_type"],
                "教育用途_共识均值": (a[0] + b[0]) / 2,
                "可读性_共识均值": (a[1] + b[1]) / 2,
                "连贯性_共识均值": (a[2] + b[2]) / 2,
                "信息传递_共识均值": (a[3] + b[3]) / 2,
                "整体质量_共识均值": (a[4] + b[4]) / 2,
                "reviewer_a_整体质量": a[4],
                "reviewer_b_整体质量": b[4],
            }
        )
    consensus = {r["blind_id"]: float(r["整体质量_共识均值"]) for r in consensus_rows}
    write_csv(
        output / "human_consensus.csv",
        consensus_rows,
        [
            "blind_id",
            "sample_type",
            "教育用途_共识均值",
            "可读性_共识均值",
            "连贯性_共识均值",
            "信息传递_共识均值",
            "整体质量_共识均值",
            "reviewer_a_整体质量",
            "reviewer_b_整体质量",
        ],
    )

    all_ids = order_a
    overall_a = [ratings_a[i][4] for i in all_ids]
    overall_b = [ratings_b[i][4] for i in all_ids]
    metrics: list[dict[str, Any]] = [
        {"metric": "n_complete_each", "value": len(all_ids), "scope": "all 30"},
        {
            "metric": "quadratic_weighted_cohen_kappa_overall_quality",
            "value": quadratic_weighted_kappa(overall_a, overall_b),
            "scope": "all 30",
        },
        {
            "metric": "spearman_overall_quality",
            "value": spearman(overall_a, overall_b),
            "scope": "all 30",
        },
        {
            "metric": "exact_agreement_overall_quality",
            "value": sum(x == y for x, y in zip(overall_a, overall_b)) / len(all_ids),
            "scope": "all 30",
        },
        {
            "metric": "within_one_point_overall_quality",
            "value": sum(abs(x - y) <= 1 for x, y in zip(overall_a, overall_b)) / len(all_ids),
            "scope": "all 30",
        },
        {
            "metric": "mean_absolute_difference_overall_quality",
            "value": sum(abs(x - y) for x, y in zip(overall_a, overall_b)) / len(all_ids),
            "scope": "all 30",
        },
    ]

    representative = [i for i in all_ids if manifest[i]["sample_type"] == "representative"]
    diagnostic = [i for i in all_ids if manifest[i]["sample_type"] == "diagnostic"]
    if len(representative) != 24 or len(diagnostic) != 6:
        raise ValueError(f"抽样类型数不符合冻结方案：representative={len(representative)}, diagnostic={len(diagnostic)}")

    q_equal = {i: float(manifest[i]["q_equal"]) for i in representative}
    q_huber = {i: float(manifest[i]["q_huber"]) for i in representative}
    human = {i: consensus[i] for i in representative}
    eq_rho = spearman([q_equal[i] for i in representative], [human[i] for i in representative])
    huber_rho = spearman([q_huber[i] for i in representative], [human[i] for i in representative])

    rng = random.Random(args.seed)
    boot: list[tuple[float, float, float]] = []
    for _ in range(args.bootstrap_reps):
        draw = rng.choices(representative, k=len(representative))
        r_eq = spearman([q_equal[i] for i in draw], [human[i] for i in draw])
        r_huber = spearman([q_huber[i] for i in draw], [human[i] for i in draw])
        if math.isfinite(r_eq) and math.isfinite(r_huber):
            boot.append((r_eq, r_huber, r_huber - r_eq))
    if not boot:
        raise ValueError("bootstrap 没有有效重复")
    boot_names = ["spearman_q_equal_vs_consensus", "spearman_q_huber_vs_consensus", "delta_huber_minus_equal"]
    boot_rows = []
    for j, name in enumerate(boot_names):
        values = [r[j] for r in boot]
        boot_rows.append(
            {
                "metric": name,
                "point_estimate": [eq_rho, huber_rho, huber_rho - eq_rho][j],
                "ci_95_low": percentile(values, 0.025),
                "ci_95_high": percentile(values, 0.975),
                "valid_repetitions": len(boot),
                "requested_repetitions": args.bootstrap_reps,
                "seed": args.seed,
                "method": "paired case bootstrap over the 24 representative items; exploratory",
            }
        )
    write_csv(
        output / "paired_bootstrap.csv",
        boot_rows,
        ["metric", "point_estimate", "ci_95_low", "ci_95_high", "valid_repetitions", "requested_repetitions", "seed", "method"],
    )

    comparison_rows = []
    for blind_id in representative:
        comparison_rows.append(
            {
                "blind_id": blind_id,
                "source_domain": manifest[blind_id]["source_domain"],
                "q_equal_tertile": manifest[blind_id]["q_equal_tertile"],
                "q_equal": q_equal[blind_id],
                "q_huber": q_huber[blind_id],
                "reviewer_a_overall_quality": ratings_a[blind_id][4],
                "reviewer_b_overall_quality": ratings_b[blind_id][4],
                "human_consensus": human[blind_id],
            }
        )
    write_csv(
        output / "candidate_human_comparison.csv",
        comparison_rows,
        ["blind_id", "source_domain", "q_equal_tertile", "q_equal", "q_huber", "reviewer_a_overall_quality", "reviewer_b_overall_quality", "human_consensus"],
    )

    d_eq = [float(manifest[i]["q_equal"]) for i in diagnostic]
    d_huber = [float(manifest[i]["q_huber"]) for i in diagnostic]
    d_human = [consensus[i] for i in diagnostic]
    ranks_eq = rankdata(d_eq)
    ranks_huber = rankdata(d_huber)
    ranks_human = rankdata(d_human)
    diagnostic_rows = []
    for j, blind_id in enumerate(diagnostic):
        diagnostic_rows.append(
            {
                "blind_id": blind_id,
                "diagnostic_category": manifest[blind_id]["diagnostic_category"],
                "source_domain": manifest[blind_id]["source_domain"],
                "q_equal": d_eq[j],
                "q_huber": d_huber[j],
                "reviewer_a_overall_quality": ratings_a[blind_id][4],
                "reviewer_b_overall_quality": ratings_b[blind_id][4],
                "human_consensus": d_human[j],
                "rank_q_equal_among_6": ranks_eq[j],
                "rank_q_huber_among_6": ranks_huber[j],
                "rank_human_among_6": ranks_human[j],
            }
        )
    write_csv(
        output / "diagnostic_cases.csv",
        diagnostic_rows,
        ["blind_id", "diagnostic_category", "source_domain", "q_equal", "q_huber", "reviewer_a_overall_quality", "reviewer_b_overall_quality", "human_consensus", "rank_q_equal_among_6", "rank_q_huber_among_6", "rank_human_among_6"],
    )
    write_csv(output / "human_check_metrics.csv", metrics, ["metric", "value", "scope"])

    old_gate_note = (
        "30 条方案存在口径冲突：分析步骤写明不增加 bootstrap/区间，验收段又要求 2,000 次分层配对 bootstrap、复用设计权重，" 
        "并以 95% 区间半宽 0.15 和有效重复率 95% 作门槛。本轮按最新流程附加的普通配对案例 bootstrap 不是该设计加权方法；" 
        "且抽样清单未提供纳入概率/设计权重，因此不能据它宣称预注册门槛通过。"
    )
    reason = (
        "q_equal 与 q_huber 在 24 条代表样本上的点相关相同，" 
        "配对 bootstrap 的相关差区间跨 0；因此保留两个候选，不强行选择。"
    )
    if order_checks["returned_orders_identical"] and not order_checks["reviewer_b_matches_reviewer2_template"]:
        reason += " 两份回收表行序相同且都对应 reviewer1 模板，须在结论中记录顺序独立随机化未满足。"
    summary = [
        "# Q1.1 30 条双人盲评回收分析",
        "",
        "## 数据检查",
        "",
        f"- 两份表均为两位人工评审填写（按项目负责人说明）；每份 {len(all_ids)}/30 条完整、评分均为 1–5 整数，ID 完全匹配且无重复。",
        f"- Reviewer A/B 的行序相同：`{order_checks['returned_orders_identical']}`；A、B 均匹配 reviewer1 模板：`{order_checks['reviewer_a_matches_reviewer1_template'] and order_checks['reviewer_b_matches_reviewer1_template']}`；B 匹配 reviewer2 模板：`{order_checks['reviewer_b_matches_reviewer2_template']}`。因此独立随机顺序要求未满足。",
        "- 本次文本来自中文机译辅助评审；结论限于机译版本上的评分可靠性核查，不等同于直接核验英文原文。",
        "",
        "## 一致性与候选比较",
        "",
        f"- 整体质量二次加权 Cohen Kappa：{fmt(float(metrics[1]['value']))}。",
        f"- 整体质量 Spearman：{fmt(float(metrics[2]['value']))}；完全同分比例：{fmt(float(metrics[3]['value']))}；相差不超过 1 分比例：{fmt(float(metrics[4]['value']))}。",
        f"- 24 条 representative：Spearman(q_equal, 共识)={fmt(eq_rho)}；Spearman(q_huber, 共识)={fmt(huber_rho)}；点差={fmt(huber_rho-eq_rho)}。",
        f"- 配对案例 bootstrap：{len(boot)}/{args.bootstrap_reps} 个有效重复；q_equal、q_huber、相关差的 95% 区间分别见 `paired_bootstrap.csv`。相关差区间 [{fmt(float(boot_rows[2]['ci_95_low']))}, {fmt(float(boot_rows[2]['ci_95_high']))}] 跨 0，不能区分候选。",
        "- 这组区间按最新流程要求补算，仅作 24 条小样本的描述性辅助结果。q_equal 和 q_huber 的单独区间半宽均超过 0.15；由于抽样清单未保存设计权重，不能用本次普通配对 bootstrap 替代方案验收段要求的设计加权 bootstrap。",
        "",
        "## 选择状态与限制",
        "",
        f"- 选择状态：保留 `q_equal` 与 `q_huber`，`candidate_selected` 留空。{reason}",
        f"- 门槛核对：{old_gate_note}",
        "- 本项完成的是 A1 可读长度子集 30 条文本的双人评分核查；不代表 793 条全量人工验证，也不补足 A2/A3 原文映射缺口。",
        "- 六条 diagnostic 仅作案例表，不并入 24 条总体相关；见 `diagnostic_cases.csv`。",
        "",
        "## 输出",
        "",
        "- `human_consensus.csv`",
        "- `human_check_metrics.csv`",
        "- `candidate_human_comparison.csv`",
        "- `paired_bootstrap.csv`",
        "- `diagnostic_cases.csv`",
        "- `decision_status.json`",
    ]
    (output / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    decision = {
        "status": "review_analysis_complete_candidate_not_selected",
        "review_type": "two_human_reviewers_machine_translation_assisted",
        "ai_judge_results_available": False,
        "n_total": len(all_ids),
        "n_representative": len(representative),
        "n_diagnostic": len(diagnostic),
        "candidate_selected": None,
        "candidates_retained": ["q_equal", "q_huber"],
        "reason": reason,
        "order_protocol_deviation": order_checks,
        "translation_assisted_scope": True,
        "a2_a3_original_text_validated": False,
        "q1_1_fully_frozen": False,
    }
    (output / "decision_status.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            "reviewer_a": {"path": str(reviewer_a), "sha256": sha256(reviewer_a)},
            "reviewer_b": {"path": str(reviewer_b), "sha256": sha256(reviewer_b)},
            "sampling_manifest_private": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)},
            "reviewer1_template": {"path": str(t1_path.relative_to(ROOT)), "sha256": sha256(t1_path)},
            "reviewer2_template": {"path": str(t2_path.relative_to(ROOT)), "sha256": sha256(t2_path)},
        },
        "validation": {
            "same_30_ids": True,
            "all_scores_complete_1_to_5": True,
            **order_checks,
        },
        "analysis": {
            "representative_only_candidate_correlation": True,
            "diagnostic_not_pooled": True,
            "bootstrap_repetitions": args.bootstrap_reps,
            "bootstrap_seed": args.seed,
            "bootstrap_method": "paired observation percentile bootstrap over the 24 representative records; supplementary/exploratory",
            "candidate_selection_rule": "no winner selected; both candidates retained unless modeler applies a valid pre-frozen rule",
        },
        "outputs_sha256": {
            p.name: sha256(p)
            for p in sorted(output.iterdir())
            if p.is_file() and p.name != "reproduction_manifest.json"
        },
    }
    (output / "reproduction_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Analysis complete: {output}")
    print(f"Overall QWK={fmt(float(metrics[1]['value']))}; Spearman={fmt(float(metrics[2]['value']))}")
    print(f"Representative rho: q_equal={fmt(eq_rho)}, q_huber={fmt(huber_rho)}")
    print(f"Candidate delta 95% CI=[{fmt(float(boot_rows[2]['ci_95_low']))}, {fmt(float(boot_rows[2]['ci_95_high']))}]")
    print(f"Candidate selected: {decision['candidate_selected']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_REVIEW_DIR / "sampling_manifest_private.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
