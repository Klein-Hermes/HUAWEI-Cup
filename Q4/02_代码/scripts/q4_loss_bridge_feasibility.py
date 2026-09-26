"""Run the pre-fit Q4 Loss–Benchmark feasibility audit only.

AI disclosure: OpenAI Codex desktop 26.917.71314 (build 10954; prod), model ID
gpt-6-luna (GPT-6 Luna), developer OpenAI, official release date 2026-09-22.

This script never fits a bridge, tunes a model, or creates predictions.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

Q4 = Path(__file__).resolve().parents[2]
ROOT = Q4.parent
ATT = ROOT / "中文题目" / "F题" / "real_attachments"
CE = ATT / "C_efficiency_evolution"
BS = ATT / "B_scaling_laws"
Q2R = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1"
OUT = Q4 / "03_结果" / "results"
CONTRACT = Q4 / "01_方案说明" / "q4_model_contract_v0.6.md"
AI_DISCLOSURE = (
    "OpenAI Codex desktop 26.917.71314 (build 10954; prod); model ID "
    "gpt-6-luna (GPT-6 Luna); developer OpenAI; official release date 2026-09-22."
)

C5 = CE / "loss_benchmark_bridge.csv"
C6 = CE / "loss_benchmark_bridge_expanded.csv"
B1 = BS / "pythia_training_log_existing.csv"
B1_FIT = Q2R / "b1_fitted_predictions.csv"
LOSS_PROTOCOL = ROOT / "Q2" / "05_审计交付" / "loss_protocol_matrix.csv"
DATA_NOTE = ROOT / "中文题目" / "F题" / "数据说明(无隐藏字段版本）.pdf"

BASE_COLS = ["N_params_B", "D_tokens_B", "Val_Loss"]
LB_COLS = [
    "LB_Average", "LB_IFEval", "LB_BBH", "LB_MATH", "LB_GPQA", "LB_MUSR", "LB_MMLU_PRO"
]
NUMERIC_COLS = BASE_COLS + LB_COLS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")
    return path


def normalized_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and math.isfinite(float(numeric)):
        return format(float(numeric), ".15g")
    return str(value).strip()


def family_hint(source: str, model: str) -> str:
    text = f"{source} {model}".lower()
    if "pythia" in text:
        return "Pythia (source-label hint; version/family independence unverified)"
    return "unresolved"


def load_bridge(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = sorted(set(NUMERIC_COLS + ["Model", "Loss_Source", "Loss_Comparability"]) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")
    frame.insert(0, "source_table", source)
    frame.insert(1, "source_row_1based", np.arange(2, len(frame) + 2))
    frame["source_model_family_hint"] = [
        family_hint(source_label, model_name)
        for source_label, model_name in zip(frame["Loss_Source"], frame["Model"])
    ]
    frame["Model"] = frame["Model"].astype(str).str.strip()
    for column in NUMERIC_COLS:
        frame[column + "_numeric"] = pd.to_numeric(frame[column], errors="coerce")
    frame["loss_finite"] = np.isfinite(frame["Val_Loss_numeric"].to_numpy(dtype=float))
    frame["lb_average_finite"] = np.isfinite(frame["LB_Average_numeric"].to_numpy(dtype=float))
    frame["high_comparability_label"] = frame["Loss_Comparability"].str.lower().str.startswith("high")
    frame["all_lb_finite"] = np.isfinite(frame[[c + "_numeric" for c in LB_COLS]].to_numpy(dtype=float)).all(axis=1)
    frame["high_paired_finite"] = (
        frame["high_comparability_label"] & frame["loss_finite"] & frame["lb_average_finite"]
    )
    return frame


def content_key(row: pd.Series) -> str:
    # The contract key includes the raw source and model name. A name is not
    # revision-level identity evidence, so exact duplicates remain content-only.
    values = [str(row["Loss_Source"]), str(row["Model"])]
    values.extend(normalized_value(row.get(column, "")) for column in NUMERIC_COLS)
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def protocol_evidence() -> tuple[dict[str, str], str]:
    frame = pd.read_csv(LOSS_PROTOCOL, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    row = frame.loc[frame["dataset"].eq("B1")]
    if row.empty:
        return {}, "No B1 row in Q2 loss protocol matrix"
    record = row.iloc[0].to_dict()
    details = (
        f"Q2 loss_protocol_matrix.csv: unit={record.get('loss_unit', 'unknown')}; "
        f"evaluation_dataset={record.get('evaluation_dataset', 'unknown')}; "
        f"split={record.get('split', 'unknown')}; tokenizer={record.get('tokenizer', 'unknown')}; "
        f"preprocessing={record.get('preprocessing', 'unknown')}"
    )
    return record, details


def terminal_per_size(frame: pd.DataFrame, value_column: str, source: str) -> pd.DataFrame:
    columns = ["N_params_B", "D_tokens_B", value_column]
    work = frame[columns].copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["N_params_B", "D_tokens_B", value_column])
    idx = work.groupby("N_params_B")["D_tokens_B"].idxmax()
    final = work.loc[idx].sort_values("N_params_B").reset_index(drop=True)
    final.insert(0, "target_source", source)
    final.insert(1, "target_scope", "terminal observed D checkpoint by parameter-size group")
    final.insert(2, "target_record_id", [f"{source}:N={n:g}:D={d:g}" for n, d in zip(final.N_params_B, final.D_tokens_B)])
    return final


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_paths = [C5, C6, B1, B1_FIT, LOSS_PROTOCOL, DATA_NOTE, CONTRACT]
    absent = [str(p) for p in input_paths if not p.exists()]
    if absent:
        raise FileNotFoundError(f"Required audit inputs missing: {absent}")

    c5_raw = load_bridge(C5, "C5")
    c6_raw = load_bridge(C6, "C6")
    bridge = pd.concat([c5_raw, c6_raw], ignore_index=True)
    bridge["content_key_sha256"] = bridge.apply(content_key, axis=1)
    bridge["exact_content_occurrences"] = bridge.groupby("content_key_sha256")["content_key_sha256"].transform("size")
    bridge["cross_table_duplicate"] = bridge.groupby("content_key_sha256")["source_table"].transform("nunique").gt(1)
    bridge["pooled_duplicate_rank"] = bridge.groupby("content_key_sha256").cumcount() + 1
    bridge["kept_in_pooled_unique_table"] = bridge["pooled_duplicate_rank"].eq(1)
    c5 = bridge.loc[bridge["source_table"].eq("C5")].copy()
    c6 = bridge.loc[bridge["source_table"].eq("C6")].copy()
    bridge["model_identity_status"] = "model name present; revision/SHA and leaderboard submission/run ID unavailable"
    bridge["loss_unit_status"] = "unknown in bridge CSV; must be source-verified"
    bridge["evaluation_protocol_status"] = "unknown beyond supplied comparability label"
    bridge["benchmark_definition_status"] = "metric names present; evaluation-set/aggregation details not row-verified"
    pair_columns = [
        "source_table", "source_row_1based", "Model", "Loss_Source", "Loss_Comparability",
        "source_model_family_hint", "N_params_B", "D_tokens_B", "Val_Loss",
        *LB_COLS, "high_comparability_label", "loss_finite", "lb_average_finite",
        "all_lb_finite", "high_paired_finite", "content_key_sha256", "cross_table_duplicate",
        "pooled_duplicate_rank", "kept_in_pooled_unique_table", "model_identity_status",
        "loss_unit_status", "evaluation_protocol_status", "benchmark_definition_status",
    ]
    pair_audit = bridge[pair_columns].copy()
    pair_path = write_csv(pair_audit, "q4_loss_bridge_pair_audit.csv")

    protocol_row, protocol_text = protocol_evidence()
    loss_unit = str(protocol_row.get("loss_unit", "UNKNOWN")).strip() or "UNKNOWN"
    eval_dataset = str(protocol_row.get("evaluation_dataset", "UNKNOWN")).strip() or "UNKNOWN"
    split = str(protocol_row.get("split", "UNKNOWN")).strip() or "UNKNOWN"
    tokenizer = str(protocol_row.get("tokenizer", "UNKNOWN")).strip() or "UNKNOWN"
    preprocessing = str(protocol_row.get("preprocessing", "UNKNOWN")).strip() or "UNKNOWN"

    b1 = pd.read_csv(B1, encoding="utf-8-sig")
    b1_fit = pd.read_csv(B1_FIT, encoding="utf-8-sig")
    b1_final = terminal_per_size(b1, "val_loss", "Q2_B1_observed")
    fit_final = terminal_per_size(b1_fit, "predicted_val_loss", "Q2_M0_B1_fitted")
    targets = []
    for final_frame, value_col in [(b1_final, "val_loss"), (fit_final, "predicted_val_loss")]:
        for row in final_frame.to_dict("records"):
            targets.append({
                "target_source": row["target_source"], "target_scope": row["target_scope"],
                "target_record_id": row["target_record_id"], "N_params_B": row["N_params_B"],
                "D_tokens_B": row["D_tokens_B"], "target_loss": row[value_col],
                "target_loss_unit": loss_unit, "unit_protocol_verified": False,
                "target_role": "candidate only; Q2 endpoint, not an approved future Q1-Q3 conversion target",
            })
    target_frame = pd.DataFrame(targets)
    high_unique = bridge.loc[bridge["high_comparability_label"] & bridge["high_paired_finite"] & bridge["kept_in_pooled_unique_table"]].copy()
    if high_unique.empty:
        high_min = high_max = np.nan
    else:
        high_min = float(high_unique["Val_Loss_numeric"].min())
        high_max = float(high_unique["Val_Loss_numeric"].max())
    target_frame["within_high_comparability_loss_range"] = (
        target_frame["target_loss"].between(high_min, high_max, inclusive="both") if math.isfinite(high_min) else False
    )
    targets_path = write_csv(target_frame, "q4_loss_bridge_target_audit.csv")

    def summary_row(label: str, frame: pd.DataFrame, unique: bool = False) -> dict[str, Any]:
        selected = frame[frame["high_comparability_label"]].copy()
        if unique:
            selected = selected[selected["kept_in_pooled_unique_table"]]
        finite = selected[selected["loss_finite"] & selected["lb_average_finite"]]
        family_score_cols = [column + "_numeric" for column in LB_COLS[1:]]
        formula_rows = selected.dropna(subset=["LB_Average_numeric", *family_score_cols])
        formula_gap = (
            formula_rows["LB_Average_numeric"] - formula_rows[family_score_cols].mean(axis=1)
        ).abs()
        unique_pairs = selected["content_key_sha256"].nunique()
        return {
            "scope": label,
            "input_rows": int(len(frame)),
            "high_comparability_labeled_rows": int(selected.shape[0]),
            "high_rows_finite_loss_and_lb_average": int(len(finite)),
            "unique_high_content_pairs": int(unique_pairs),
            "distinct_high_loss_values": int(selected["Val_Loss_numeric"].nunique(dropna=True)),
            "distinct_parameter_sizes": int(selected["N_params_B_numeric"].nunique(dropna=True)),
            "unique_model_names": int(selected["Model"].nunique()),
            "verified_model_version_count": 0,
            "model_names": "; ".join(sorted(set(selected["Model"]))),
            "source_family_hints": "; ".join(sorted(set(selected["source_model_family_hint"]))),
            "independent_verified_family_count": 0,
            "high_loss_min": float(selected["Val_Loss_numeric"].min()) if len(selected) else np.nan,
            "high_loss_max": float(selected["Val_Loss_numeric"].max()) if len(selected) else np.nan,
            "high_lb_average_min": float(selected["LB_Average_numeric"].min()) if len(selected) else np.nan,
            "high_lb_average_max": float(selected["LB_Average_numeric"].max()) if len(selected) else np.nan,
            "high_rows_with_all_lb_metrics_finite": int(selected["all_lb_finite"].sum()),
            "lb_average_formula_check_rows": int(len(formula_rows)),
            "lb_average_formula_max_abs_difference": float(formula_gap.max()) if len(formula_gap) else np.nan,
            "lb_average_formula_matches_at_1e_8": int(formula_gap.le(1e-8).sum()) if len(formula_gap) else 0,
            "c5_c6_exact_cross_table_duplicates": int(selected["cross_table_duplicate"].sum()),
            "pooled_unique_high_pairs": int(unique_pairs),
            "target_candidate_rows": int(len(target_frame)),
            "target_candidate_rows_inside_high_loss_range": int(target_frame["within_high_comparability_loss_range"].sum()),
            "target_candidate_min": float(target_frame["target_loss"].min()) if len(target_frame) else np.nan,
            "target_candidate_max": float(target_frame["target_loss"].max()) if len(target_frame) else np.nan,
            "loss_unit": loss_unit,
            "evaluation_dataset": eval_dataset,
            "split": split,
            "tokenizer": tokenizer,
            "preprocessing": preprocessing,
            "benchmark_definition_status": "six official task names/direction and arithmetic-average identity checked; exact historical run/version-to-score provenance unavailable",
            "target_scope_status": "Q2 final checkpoints are candidates only; no approved future target loss grid found in Q4 contract",
            "prefit_gate_status": "BLOCKED_NO_FIT",
            "blocked_reasons": "Loss unit/evaluation split/tokenizer/preprocessing are UNKNOWN in Q2 protocol audit; exact historical model revision-to-score provenance is unavailable; Q2 candidate endpoints are same-family training records, not an approved future target schedule.",
            "protocol_evidence": protocol_text,
            "no_fit_executed": True,
        }

    summaries = [summary_row("C5", c5), summary_row("C6", c6), summary_row("pooled_C5_C6_deduplicated", bridge, unique=True)]
    feas = pd.DataFrame(summaries)
    feas_path = write_csv(feas, "q4_loss_benchmark_feasibility.csv")

    reason_lines = [
        "# Q4 Loss–Benchmark 预拟合可行性门控报告",
        "",
        f"> 门控仅做来源、字段、重复、量尺与支持范围审计；拟合执行数 = 0。合同：`{rel(CONTRACT)}`。",
        "",
        "## 判定",
        "",
        "预拟合状态：`BLOCKED_NO_FIT`。现阶段不得拟合 Loss–Benchmark 映射。",
        "",
        "阻断理由：",
        "",
        "- Q2 `loss_protocol_matrix.csv` 对 B1 的 Loss 单位、评测集/切分、tokenizer 与预处理均标为 UNKNOWN；因此虽有 C5/C6 的“高可比”文字标签，仍不能独立核验实际协议与单位。",
        "- C5/C6 含有 `Model` 模型名称和 `Loss_Source` 来源字段；模型名称、原始来源与数值字段已纳入完全重复键，并在逐行审计表中保留。表内没有 revision/SHA 或排行榜提交/评测运行 ID，因此无法把名称对应到确切历史版本与评测运行。六个 `LB_*` 字段按 Open LLM Leaderboard 官方说明映射到 IFEval、BBH、MATH、GPQA、MuSR、MMLU-Pro，门控核对了 `LB_Average` 与六项算术均值的一致性。",
        "- 高可比记录的模型名称为 Pythia 各规模模型；C5/C6 的 7 条高可比内容记录跨表逐行重复。按模型名、来源及数值去重后有 7 个名称、1 个来源家族提示，但没有经核验的模型 revision 或评测运行，不能将名称数当作独立版本数，也不能验证跨家族迁移。官方 Pythia 说明 8 个主模型规模各训练约 299.893B tokens，与桥接表 D 相符；这不能确认本地 Loss 的评测切分或具体版本。",
        "- Q2 B1 每个参数规模的终点观测与 M0 终点拟合值已列为候选 Loss 目标；Q4 尚未指定需要转换的未来 `(N,D)` 目标网格，因此不能把训练样本范围冒充预测目标支持。",
        "- Open LLM Leaderboard 的任务定义及高分方向见 [官方任务与评测说明](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about)；Pythia 训练 token 数与模型版本信息见 [EleutherAI Pythia 官方仓库](https://github.com/EleutherAI/pythia)。这些公开说明用于核验字段语义，不补出附件中缺失的行级版本和 Loss protocol。",
        "",
        "## 门控产物",
        "",
        f"- `{rel(feas_path)}`：按 C5、C6 与跨表去重合并表给出状态、配对数、支持范围与阻断原因。",
        f"- `{rel(pair_path)}`：逐行审计表；源表行号、可比性标签、数值字段、跨表重复键和身份/单位限制均保留。",
        f"- `{rel(targets_path)}`：Q2 B1/M0 每个规模终点的候选 Loss 值及其相对 C5/C6 高可比范围位置。它们不是已批准的未来转换目标，也不证明单位/协议一致。",
        "",
        "## 证据限制",
        "",
        "本门控没有读入或改写 Q1–Q3 原始数据；只读复核 Q2 的冻结 B1 与 M0 结果文件。没有识别到已批准的 Q1/Q3 直接 Loss-to-Benchmark 转换目标。继续桥接前须由队伍明确未来目标网格及 Benchmark 定义，并从来源补足 Loss 单位/评测协议证据；取得证据后重做门控。",
        "",
    ]
    report_path = OUT / "q4_loss_benchmark_feasibility_report.md"
    report_path.write_text("\n".join(reason_lines), encoding="utf-8")

    outputs = [feas_path, pair_path, targets_path, report_path]
    manifest = {
        "status": "BLOCKED_NO_FIT",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ai_disclosure": AI_DISCLOSURE,
        "script": rel(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "contract_sha256": sha256(CONTRACT),
        "inputs_sha256": {rel(path): sha256(path) for path in input_paths},
        "outputs_sha256": {rel(path): sha256(path) for path in outputs},
        "external_sources": [
            {"url": "https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about", "use": "Official task names, score directions, and evaluation metrics; accessed 2026-09-25."},
            {"url": "https://github.com/EleutherAI/pythia", "use": "Official Pythia training token count and model/version notes; accessed 2026-09-25."},
        ],
        "fit_executed": False,
        "gate_rows": summaries,
        "limitations": [
            "Comparability labels were retained but not treated as a substitute for row-level protocol evidence.",
            "C5/C6 model names are present, but revision/SHA and leaderboard submission/run identifiers are absent; exact content duplication is not version identity.",
            "Q2 endpoint values are candidate support checks only; Q4 has no approved future target-loss grid.",
        ],
    }
    manifest_path = OUT / "q4_loss_benchmark_feasibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={manifest['status']} fit_executed={manifest['fit_executed']}")
    for row in summaries:
        print(
            f"{row['scope']}: high={row['high_comparability_labeled_rows']} "
            f"unique_pairs={row['unique_high_content_pairs']} "
            f"loss_range=[{row['high_loss_min']:.6g},{row['high_loss_max']:.6g}]"
        )
    print(f"targets={len(target_frame)} within_high_loss_range={int(target_frame['within_high_comparability_loss_range'].sum())}")
    print(f"report={rel(report_path)} manifest={rel(manifest_path)}")


if __name__ == "__main__":
    main()
