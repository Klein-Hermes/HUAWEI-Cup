"""Build a source-linked model registry and cross-source coverage report for Q4.

AI assistance: OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater
checked 2026-09-25, up_to_date); developer OpenAI. Model selector name: ; model
version/publication date: . These fields are blank per user instruction.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
DATA = ROOT / "中文题目" / "F题" / "real_attachments" / "C_efficiency_evolution"
RESULTS = Q4 / "03_结果" / "results"

BASE_FIELDS = [
    "Model", "#Params (B)", "Submission Date", "Hub License", "Type",
    "Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO",
]
NUMERIC_BASE_FIELDS = {
    "#Params (B)", "Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO",
}
C1_FIELD_NAMES = {
    "Model": "c1_model_name_raw", "#Params (B)": "c1_parameter_count_B_raw",
    "Submission Date": "c1_submission_date_raw", "Hub License": "c1_hub_license_raw",
    "Type": "c1_type_raw", "Average ⬆️": "c1_average_raw", "IFEval": "c1_ifeval_raw",
    "BBH": "c1_bbh_raw", "MATH Lvl 5": "c1_math_lvl5_raw", "GPQA": "c1_gpqa_raw",
    "MUSR": "c1_musr_raw", "MMLU-PRO": "c1_mmlu_pro_raw",
}
C2_EXTRA_FIELDS = ["Epoch_AI_Publication_Date", "Epoch_AI_Organization", "Epoch_AI_Open_Weights"]
C3_KEEP_FIELDS = ["Model", "Year", "Params_B", "Average", "IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO", "Source"]
C4_KEEP_FIELDS = [
    "Model", "Parameters", "Training compute (FLOP)", "Training compute lower bound",
    "Training compute upper bound", "Open model weights?",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_row_ids(value: str) -> list[int]:
    try:
        return [int(item) for item in json.loads(value or "[]")]
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def base_row_aligned(c1: dict[str, str], c2: dict[str, str]) -> bool:
    for field in BASE_FIELDS:
        left = (c1.get(field) or "").strip()
        right = (c2.get(field) or "").strip()
        if field in NUMERIC_BASE_FIELDS:
            if not left and not right:
                continue
            try:
                if abs(float(left) - float(right)) > 5e-13:
                    return False
            except ValueError:
                return False
        elif left != right:
            return False
    return True


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    candidate_path = RESULTS / "q4_c8_c1_match_candidates.csv"
    candidate_rows = read_csv(candidate_path)
    c1_rows = read_csv(DATA / "leaderboard_cleaned.csv")
    c2_rows = read_csv(DATA / "leaderboard_enhanced.csv")
    c3_rows = read_csv(DATA / "leaderboard_extended_timeseries.csv")
    c4_rows = read_csv(DATA / "epoch_all_ai_models.csv")

    alignment = read_csv(RESULTS / "q4_c1_c2_alignment_summary.csv")[0]
    if alignment.get("one_to_one_full_row_match_with_numeric_tolerance_5e_13") != "True":
        raise ValueError("C1/C2 row linkage is not independently confirmed one-to-one")
    if len(c1_rows) != len(c2_rows) or len(c1_rows) != int(alignment["c1_rows"]):
        raise ValueError("C1/C2 source row counts no longer match the alignment audit")

    c1_by_id = {row_id: row for row_id, row in enumerate(c1_rows, start=2)}
    c2_by_id = {row_id: row for row_id, row in enumerate(c2_rows, start=2)}
    c3_by_name: dict[str, list[dict[str, str]]] = {}
    for row in c3_rows:
        name = (row.get("Model") or "").strip()
        if name:
            c3_by_name.setdefault(name, []).append(row)
    c4_by_name: dict[str, list[dict[str, str]]] = {}
    for row in c4_rows:
        name = (row.get("Model") or row.get("model") or "").strip()
        if name:
            c4_by_name.setdefault(name, []).append(row)

    registry_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        name = (candidate.get("c8_model_name_raw") or "").strip()
        ids = parse_row_ids(candidate.get("c1_record_ids", ""))
        c1 = c1_by_id.get(ids[0], {}) if len(ids) == 1 else {}
        c2 = c2_by_id.get(ids[0], {}) if len(ids) == 1 else {}
        if c1 and not base_row_aligned(c1, c2):
            raise ValueError(f"C1/C2 base-field alignment failed for record {ids[0]}")
        c3 = c3_by_name.get(name, [])
        c4 = c4_by_name.get(name, [])
        c3_selected = [{key: row.get(key, "") for key in C3_KEEP_FIELDS} for row in c3]
        c4_selected = [{key: row.get(key, "") for key in C4_KEEP_FIELDS if key in row} for row in c4]
        is_unique = candidate.get("match_status") == "exact_model_id_unique_record"
        row: dict[str, Any] = {
            "registry_id": f"C8::{candidate.get('c8_model_dir', '')}",
            "model_name_raw": name,
            "c8_model_dir": candidate.get("c8_model_dir", ""),
            "c8_run_id": candidate.get("run_id", ""),
            "c8_model_revision": candidate.get("c8_model_revision", ""),
            "c8_model_sha": candidate.get("c8_model_sha", ""),
            "version_match_status": candidate.get("version_match_status", "source_version_missing"),
            "strict_version_confirmed": "False",
            "c1_record_count": candidate.get("c1_record_count", "0"),
            "c1_record_ids": candidate.get("c1_record_ids", "[]"),
            "c1_match_status": candidate.get("match_status", ""),
            "provisional_unique_name_candidate": str(is_unique),
            "c1_row_number": ids[0] if len(ids) == 1 else "",
        }
        for source_field, registry_field in C1_FIELD_NAMES.items():
            row[registry_field] = c1.get(source_field, "")
        row["c2_record_number"] = ids[0] if len(ids) == 1 and c2 else ""
        row["c2_alignment_status"] = "one_to_one_base_row_confirmed" if c2 else "not_joined"
        row["c2_epoch_publication_date"] = c2.get("Epoch_AI_Publication_Date", "")
        row["c2_epoch_organization"] = c2.get("Epoch_AI_Organization", "")
        row["c2_epoch_open_weights"] = c2.get("Epoch_AI_Open_Weights", "")
        row["c3_exact_name_record_count"] = len(c3)
        row["c3_years_json"] = json.dumps(sorted({r.get("Year", "") for r in c3 if r.get("Year")}), ensure_ascii=False)
        row["c3_sources_json"] = json.dumps(sorted({r.get("Source", "") for r in c3 if r.get("Source")}), ensure_ascii=False)
        row["c3_candidate_metrics_json"] = json.dumps(c3_selected, ensure_ascii=False, separators=(",", ":"))
        row["c4_exact_name_record_count"] = len(c4)
        row["c4_metadata_candidates_json"] = json.dumps(c4_selected, ensure_ascii=False, separators=(",", ":"))
        registry_rows.append(row)

        audit_rows.append({
            "model_name_raw": name,
            "c8_model_dir": candidate.get("c8_model_dir", ""),
            "run_id": candidate.get("run_id", ""),
            "c8_model_revision": candidate.get("c8_model_revision", ""),
            "c8_model_sha": candidate.get("c8_model_sha", ""),
            "c1_record_count": candidate.get("c1_record_count", "0"),
            "c1_record_ids": candidate.get("c1_record_ids", "[]"),
            "match_status": candidate.get("match_status", ""),
            "c1_match_basis": "exact C8 model-name string; unique C1 row required for candidate" if is_unique else "exact C8 model-name string only; duplicate or unmatched source rows prevent unique join",
            "c2_record_number": ids[0] if len(ids) == 1 and c2 else "",
            "c2_match_status": "one_to_one_base_row_confirmed" if c2 else "not_joined",
            "c2_match_basis": "same C1/C2 row number; verified across all 12 base fields (text exact; numeric tolerance 5e-13)" if c2 else "no unique C1 row to carry through verified C1/C2 alignment",
            "c3_match_basis": "exact model-name intersection only; source/year metrics retained separately" if c3 else "no exact model-name intersection",
            "c4_match_basis": "exact model-name intersection only; metadata kept as candidates" if c4 else "no exact model-name intersection",
            "alias_match_used": "False",
            "human_confirmed_version_match": "False",
            "version_match_status": candidate.get("version_match_status", "source_version_missing"),
            "strict_version_confirmed": "False",
            "c3_exact_name_record_count": len(c3),
            "c4_exact_name_record_count": len(c4),
            "provisional_unique_name_candidate": str(is_unique),
            "interpretation": "Exact-name joins are candidates only; C1 has no model revision/SHA field.",
        })

    registry_path = RESULTS / "q4_model_registry.csv"
    audit_path = RESULTS / "q4_match_audit.csv"
    write_csv(registry_path, registry_rows)
    write_csv(audit_path, audit_rows)

    aggregate = pd.read_csv(RESULTS / "q4_task_aggregate.csv", low_memory=False)
    selected = pd.read_csv(RESULTS / "q4_selected_task_scores.csv", low_memory=False)
    families = pd.read_csv(RESULTS / "q4_selected_metric_coverage.csv")
    source_coverage = pd.read_csv(RESULTS / "q4_auxiliary_source_coverage.csv")
    monthly = pd.read_csv(RESULTS / "q4_scale_tech_monthly.csv")
    c8_manifest = json.loads((RESULTS / "q4_c8_manifest.json").read_text(encoding="utf-8"))

    candidate_count = sum(row.get("match_status") == "exact_model_id_unique_record" for row in candidate_rows)
    if candidate_count != len(aggregate):
        raise ValueError("Registry candidate count and modeling aggregate row count differ")
    core = aggregate["main_cohort_complete_core6"].astype(bool)
    full7 = aggregate["main_cohort_complete_full7"].astype(bool)
    params = pd.to_numeric(aggregate["parameter_count_B"], errors="coerce").gt(0)
    dates = pd.to_datetime(aggregate["submission_date"], errors="coerce").notna()
    open_yes = aggregate["open_weights_status"].eq("Yes")

    coverage_rows: list[dict[str, Any]] = []

    def add(dimension: str, level: str, count: int, denominator: int, source: str, note: str) -> None:
        coverage_rows.append({
            "dimension": dimension, "level": level, "n_models": int(count),
            "denominator": int(denominator), "coverage_fraction": (float(count) / denominator if denominator else ""),
            "source": source, "note": note,
        })

    all_c8 = len(candidate_rows)
    add("identity", "C8 latest parseable model directories", all_c8, all_c8, "C8", "One row per latest parseable model directory.")
    add("identity", "Exact-name unique C1 row candidates", candidate_count, all_c8, "C8+C1", "Candidate join only; strict version confirmation remains zero.")
    add("identity", "Strict version-confirmed candidates", 0, candidate_count, "C8+C1", "C1 exposes no revision/SHA field to validate C8 identity.")
    for label, mask, note in [
        ("C1 positive parameter count among candidates", params, "C1 #Params (B) is not name-imputed."),
        ("C1 Submission Date among candidates", dates, "Submission date is a leaderboard timestamp, not release date."),
        ("Complete six-family score index", core, "Requires all selected leaf tasks for all six primary task families."),
        ("Complete seven-family score index", full7, "Requires ARC-Challenge in addition to all six core families."),
        ("Core6 + positive parameters", core & params, "Intersection of complete score index and observed positive C1 parameter count."),
        ("Core6 + submission date", core & dates, "Intersection of complete score index and observed C1 Submission Date."),
        ("Core6 + positive parameters + submission date", core & params & dates, "Scale/time model universe before sparse-month support restriction."),
        ("Core6 + explicit C2 Open Weights=Yes", core & open_yes, "Known-open subset; unknown and No remain distinct."),
        ("Core6 + explicit open + positive parameters + date", core & open_yes & params & dates, "Intersection for open-weight scale/time descriptions."),
    ]:
        add("analysis_intersection", label, int(mask.sum()), candidate_count, "C1+C2+C8", note)

    for _, row in families.iterrows():
        family = str(row["family"])
        add("task_family", f"{family} selected-score models", int(row["models_with_selected_scores"]), candidate_count, "C8 selected metric", f"version={row['version']}; metric={row['metric']}; expected leaf tasks={int(row['expected_tasks'])}.")
        add("task_family", f"{family} complete family", int(row["complete_family_models"]), candidate_count, "C8 selected metric", "Complete requires every declared leaf task, selected metric/version, numeric score, and known higher-is-better direction.")

    for task, group in selected.groupby("task_key", sort=True):
        n_models = int(group["model_name_raw"].nunique())
        n_eff = pd.to_numeric(group["n_effective"], errors="coerce").dropna()
        add("leaf_task", str(task), n_models, candidate_count, "C8 selected metric", f"family={group['family'].iloc[0]}; version={group['task_version'].iloc[0]}; metric={group['metric_name'].iloc[0]}; mean_n_effective={float(n_eff.mean()) if len(n_eff) else 'NA'}.")

    for _, row in source_coverage.iterrows():
        add("source_intersection", str(row["source_id"]), int(row["exact_overlap_with_1819_unique_name_candidates"]), candidate_count, str(row["file"]), "Exact model-name intersection only; no version confirmation.")

    for _, row in monthly.sort_values("submission_month").iterrows():
        add("time_month", str(row["submission_month"]), int(row["n_models"]), candidate_count, "C1 Submission Date + complete Core6", f"n>=20 decomposition support={str(row['decomposition_supported_month_n_ge_20']).lower()}; rows below 20 remain descriptive only.")

    coverage_path = RESULTS / "q4_coverage.csv"
    write_csv(coverage_path, coverage_rows)

    family_table = families[["family", "version", "metric", "expected_tasks", "models_with_selected_scores", "complete_family_models"]]
    family_md = markdown_table(family_table)
    source_md = markdown_table(source_coverage[["source_id", "row_count", "exact_overlap_with_1819_unique_name_candidates", "file"]])
    month_table = monthly[["submission_month", "n_models", "decomposition_supported_month_n_ge_20"]].copy()
    month_md = markdown_table(month_table)
    intersection = pd.DataFrame([
        ("唯一名称候选", candidate_count, all_c8),
        ("完整六族指数", int(core.sum()), candidate_count),
        ("完整七族指数", int(full7.sum()), candidate_count),
        ("六族 + 正参数 + 提交日期", int((core & params & dates).sum()), candidate_count),
        ("六族 + 明确开放权重", int((core & open_yes).sum()), candidate_count),
    ], columns=["队列交集", "模型数", "候选分母"])
    intersection_md = markdown_table(intersection)
    report = [
        "# Q4 覆盖率与有效样本报告", "",
        "> AI 辅助说明：OpenAI Codex 桌面版 26.917.71314（build 10954，prod；2026-09-25 更新器核验为 up_to_date），开发机构 OpenAI；模型选择器显示名称：；模型版本/发布日期：。（按用户指示暂空。）数据字段、合并和结论须由队伍逐项复核。", "",
        "本报告按 C8 最新可解析模型目录建立唯一名称候选，并将 C1/C2 行级配准、C3/C4 精确名称交集和任务分数覆盖分开列示。所有名称连接均是候选连接，不等于版本确认。", "",
        "## 交集样本", "", intersection_md, "",
        "ARC 完整族有 83 个模型，但它们与完整六族样本的交集为 0；因此七族完整综合分没有可计算样本。主分析保留六族，ARC 单独报告覆盖与任务分数，不把空交集误写为有统计量的敏感性结果。", "",
        "## 任务族覆盖", "", family_md, "",
        "叶任务逐项有效模型数及平均 `n_effective` 见 `../03_结果/results/q4_coverage.csv` 的 `leaf_task` 行；模型交集与缺失分母也在同表。", "",
        "## 外部来源名称交集", "", source_md, "",
        "C3/C4 只作为来源覆盖审计。C3 不同来源的评测口径不合并，C4 无精确模型名交集；C5/C6/C7 的局部数据覆盖见 `q4_auxiliary_coverage_review.md` 和桥接质量表。", "",
        "## 时间与规模样本", "", month_md, "",
        "回归只使用月样本量至少 20 的支持月份；较早的稀疏月份保留为描述记录。包含参数、提交日期及完整六族分数的模型交集用于尺度/时间可行性判断，不据缺失字段补齐样本。", "",
        f"C8 文件级解析计数见 `../03_结果/results/q4_c8_manifest.json`（扫描 {c8_manifest.get('c8_files_in_scope', 'NA')} 个文件，解析 {c8_manifest.get('c8_parseable_files_in_scope', 'NA')} 个）；C1/C2 完整行配准见 `../03_结果/results/q4_c1_c2_alignment_summary.csv`。", "",
        "## 主要表格", "",
        "- `../03_结果/results/q4_model_registry.csv`：每个 C8 最新模型目录一行，保留 C1/C2 原始候选字段、C3 匹配记录摘要和 C4 可用元数据候选。",
        "- `../03_结果/results/q4_match_audit.csv`：C8→C1/C3/C4 名称匹配状态和版本未确认原因。",
        "- `../03_结果/results/q4_coverage.csv`：来源、任务、族、交集和月份有效样本数。",
        "- `../03_结果/results/q4_selected_metric_coverage.csv`：所选任务族的主指标、版本和完整队列计数。", "",
    ]
    report_path = Q4 / "01_方案说明" / "q4_coverage_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    input_paths = [
        DATA / "leaderboard_cleaned.csv", DATA / "leaderboard_enhanced.csv",
        DATA / "leaderboard_extended_timeseries.csv", DATA / "epoch_all_ai_models.csv",
        RESULTS / "q4_c8_c1_match_candidates.csv", RESULTS / "q4_c1_c2_alignment_summary.csv",
        RESULTS / "q4_task_aggregate.csv", RESULTS / "q4_selected_task_scores.csv",
        RESULTS / "q4_selected_metric_coverage.csv", RESULTS / "q4_auxiliary_source_coverage.csv",
        RESULTS / "q4_scale_tech_monthly.csv", RESULTS / "q4_c8_manifest.json",
    ]
    outputs = [registry_path, audit_path, coverage_path, report_path]
    manifest = {
        "status": "provisional_user_decision_accepted_ai_disclosure_pending",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ai_disclosure": "OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater checked 2026-09-25, up_to_date); developer OpenAI; model selector name: ; model version/publication date: .",
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in input_paths},
        "outputs_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in outputs},
        "registry_rows": len(registry_rows), "match_audit_rows": len(audit_rows),
        "unique_name_candidates": candidate_count,
        "strict_version_confirmed": 0,
        "coverage_rows": len(coverage_rows),
        "limitations": [
            "Exact model-name joins do not establish C1/C8 model-version identity.",
            "C1/C2 are joined by verified one-to-one source row correspondence.",
            "C3 exact-name rows can represent distinct years or evaluation sources and are preserved as raw candidates.",
            "The C4 exact-name intersection is zero for the current candidate cohort.",
        ],
    }
    manifest_path = RESULTS / "q4_registry_coverage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"registry={len(registry_rows)} candidates={candidate_count} coverage_rows={len(coverage_rows)}")
    print(f"ARC_and_core6={(aggregate['ARC_complete'].astype(bool) & core).sum()} full7={int(full7.sum())}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
