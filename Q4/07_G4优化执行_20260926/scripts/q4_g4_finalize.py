"""Finalize the isolated G4 report and verify the frozen Q4 files stayed unchanged.

AI-use note for any contest code submission: assisted by OpenAI Codex (GPT-6
family) for report and code drafting; developer OpenAI. Exact model/version and
official release date are unavailable in this task record. The team must verify
and complete those fields and the full tool-participation scope before submitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
G4 = ROOT / "Q4" / "07_G4优化执行_20260926"
SNAPSHOT = G4 / "frozen_q4_baseline_sha256.json"
MANIFEST = G4 / "g4_manifest.json"
REPORT = G4 / "g4_execution_report.md"
P2_REVIEW_RECEIPT = G4 / "q4_g4_task3_p2_review_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task3_p2_status() -> str:
    if not P2_REVIEW_RECEIPT.is_file():
        return "PENDING_EXTERNAL_REVIEW"
    receipt = read_json(P2_REVIEW_RECEIPT)
    if receipt.get("gate") != "P2" or receipt.get("scope") != "task3_forecast":
        return "INVALID_REVIEW_RECEIPT"
    if receipt.get("status") != "PASS":
        return str(receipt.get("status", "INVALID_REVIEW_RECEIPT"))
    reviewed_files = receipt.get("reviewed_files", {})
    if not reviewed_files:
        return "INVALID_REVIEW_RECEIPT"
    for relative_path, expected_hash in reviewed_files.items():
        path = ROOT / Path(relative_path)
        if not path.is_file() or sha256(path) != expected_hash:
            return "STALE_REVIEW_RECEIPT"
    return "PASS"


def verify_frozen_snapshot() -> dict:
    snapshot = read_json(SNAPSHOT)
    root_results = []
    unchanged_all = True
    for entry in snapshot["roots"]:
        expected = {item["path"]: item["sha256"] for item in entry["files"]}
        root_abs = ROOT / Path(entry["root"])
        actual_files = sorted(path for path in root_abs.rglob("*") if path.is_file())
        actual = {path.relative_to(ROOT).as_posix(): sha256(path) for path in actual_files}
        changed = sorted(path for path in expected.keys() & actual.keys() if expected[path] != actual[path])
        added = sorted(actual.keys() - expected.keys())
        removed = sorted(expected.keys() - actual.keys())
        ok = not changed and not added and not removed
        unchanged_all = unchanged_all and ok
        root_results.append({
            "root": entry["root"],
            "expected_file_count": len(expected),
            "actual_file_count": len(actual),
            "unchanged": ok,
            "changed": changed,
            "added": added,
            "removed": removed,
        })
    return {"unchanged": unchanged_all, "snapshot_created_at_local": snapshot["created_at_local"], "roots": root_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    decomposition_dir = G4 / "results" / "full" / "decomposition"
    forecast_dir = G4 / "results" / "forecast"
    required = [
        decomposition_dir / "q4_g4_decomposition.csv",
        decomposition_dir / "q4_g4_share_fieller.csv",
        decomposition_dir / "q4_g4_publisher_bootstrap_summary.csv",
        decomposition_dir / "q4_g4_decomposition_diagnostics.json",
        forecast_dir / "q4_frontier_scenario_forecasts.csv",
        forecast_dir / "q4_frontier_candidate_summary.csv",
        forecast_dir / "q4_frontier_model_diagnostics.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Full outputs are incomplete: " + ", ".join(missing))

    frozen = verify_frozen_snapshot()
    decomp = read_json(decomposition_dir / "q4_g4_decomposition_diagnostics.json")
    forecast_diag = read_json(forecast_dir / "q4_frontier_model_diagnostics.json")
    contrib = pd.read_csv(decomposition_dir / "q4_g4_decomposition.csv", encoding="utf-8-sig")
    fieller = pd.read_csv(decomposition_dir / "q4_g4_share_fieller.csv", encoding="utf-8-sig")
    bootstrap = pd.read_csv(decomposition_dir / "q4_g4_publisher_bootstrap_summary.csv", encoding="utf-8-sig")
    forecasts = pd.read_csv(forecast_dir / "q4_frontier_scenario_forecasts.csv", encoding="utf-8-sig")
    candidates = pd.read_csv(forecast_dir / "q4_frontier_candidate_summary.csv", encoding="utf-8-sig")

    passed_baseline = max(decomp["analysis"]["frozen_baseline_max_abs_differences"].values()) <= 1e-8
    all_ratio_unstable = bool((~fieller["ratio_stably_identified"].astype(bool)).all())
    sign_flip_pct = float(decomp["analysis"]["publisher_cluster_bootstrap"]["denominator_sign_opposite_original_pct"])
    bootstrap_diag = decomp["analysis"]["publisher_cluster_bootstrap"]
    task3_p2 = task3_p2_status()
    analysis_valid = passed_baseline and frozen["unchanged"] and all_ratio_unstable
    if not analysis_valid:
        outcome = "verification_failed"
    elif task3_p2 != "PASS":
        outcome = "analysis_ready_pending_task3_p2"
    else:
        outcome = "analysis_complete_with_scoped_limitations"

    component_labels = {
        "scale_composition": "规模组成项",
        "type_composition": "类型组成项",
        "month_fixed_effect": "月份固定项",
        "non_scale_total": "非规模合计",
        "predicted_mean_change": "模型预测均值差",
        "observed_mean_change": "观测均值差",
    }
    contributions_of_interest = contrib[contrib["component"].isin(component_labels)]
    contribution_lines = [
        f"| {component_labels[row['component']]} | {float(row['estimate_index_points']):+.4f} | [{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] |"
        for _, row in contributions_of_interest.iterrows()
    ]
    forecast_rows = forecasts[forecasts["is_primary_alpha_half"].astype(bool)].sort_values(["horizon_months", "scenario"])
    forecast_lines = [
        f"| {int(row['horizon_months'])} | {row['target_month']} | {row['scenario_label_zh']} | {float(row['frontier_point_C1_Average_P90']):.2f} | {float(row['parameter_anchor_only_q025']):.2f}–{float(row['parameter_anchor_only_q975']):.2f} | {float(row['conditional_simulation_q025']):.2f}–{float(row['conditional_simulation_q975']):.2f} | {float(row['future_shock_band_width_increment']):.2f} |"
        for _, row in forecast_rows.iterrows()
    ]
    candidate_lines = [
        f"| {row['method']} | {int(row['origins'])} | {float(row['MAE']):.3f} |"
        for _, row in candidates.sort_values("MAE").iterrows()
    ]
    fieller_shapes = ", ".join(f"{row['component']}={row['fieller_shape']}" for _, row in fieller.iterrows())
    failed_roots = [root for root in frozen["roots"] if not root["unchanged"]]
    freeze_text = "所有基线快照文件哈希一致" if frozen["unchanged"] else f"基线目录检测到变动：{', '.join(root['root'] for root in failed_roots)}；请检查 manifest 详细差异。"
    timing_items = []
    for target_month, timing in forecast_diag["target_timing_as_of_run"].items():
        month_delta = int(timing["months_from_run_month"])
        if month_delta < 0:
            timing_items.append(f"{target_month} 已过去 {-month_delta} 个自然月")
        elif month_delta == 0:
            timing_items.append(f"{target_month} 是本次运行所在月")
        else:
            timing_items.append(f"{target_month} 距本次运行月还有 {month_delta} 个自然月")
    version_status_summary = ", ".join(
        f"{key}={value}" for key, value in sorted(
            forecast_diag["target_queue_version_match_status_counts"].items()))
    report = f"""# G4 执行报告（{datetime.now(timezone.utc).date().isoformat()}）

**执行状态：** `{outcome}`。M1=PASS；P1=PASS；Task 3 独立 P2={task3_p2}；AI 披露核对=PENDING_TEAM_VERIFICATION。基线重现={'PASS' if passed_baseline else 'FAIL'}；冻结目录哈希={'PASS' if frozen['unchanged'] else 'FAIL'}。

## 做了什么

1. 在独立目录复现 2024-11 到 2025-03 的共同支持分解；输入支持区间为 log10(参数量/B)=[{decomp['analysis']['support_log10_params_B'][0]:.6f}, {decomp['analysis']['support_log10_params_B'][1]:.6f}]，n={decomp['analysis']['fit_rows']}，设计秩 {decomp['analysis']['design_rank']}/{decomp['analysis']['design_columns']}，残差自由度 {decomp['analysis']['residual_df']}。
2. 对观测端点均值差 (D_{{obs}}) 构造 HC3 联合协方差和 Fieller 比值集合，并用发布者簇 bootstrap 固定共同支持重拟合两个估计量，以诊断份额是否可识别；正文报告分解量与区间，不报告点份额。
3. 在隔离目录重跑 C1 Average P90 三档 C4 条件情景，分别导出无未来扰动参数/锚点带和叠加未来月度误差后的模拟带。Loss–Benchmark 桥接不拟合。

## 分解结果

表中区间均为固定早晚期经验协变量分布条件下的 HC3 95% 线性对比区间，单位为六族指数点；观测均值差区间另由端点两组模型估计。

| 分解项 | 指数点 | HC3 95% 区间 |
|---|---:|---:|
{chr(10).join(contribution_lines)}

- 端点观测净变化为 {decomp['analysis']['observed_endpoint_mean_difference']:.4f}，HC3 SE={decomp['analysis']['observed_change_HC3_se']:.4f}，95% CI=[{fieller['denominator_ci95_low'].iloc[0]:.4f}, {fieller['denominator_ci95_high'].iloc[0]:.4f}]。区间跨 0；Fieller 集合形状为 {fieller_shapes}，因此分量与净变化之比无法稳定识别。正文和摘要只报告分解量与各自 HC3 区间，不列分量百分比或贡献率。
- 发布者簇数 {bootstrap_diag['publisher_clusters']}；bootstrap {bootstrap_diag['successful_replicates']}/{bootstrap_diag['requested_replicates']} 成功，失败 {bootstrap_diag['failed_replicates']}；分母符号与原点估计相反的成功复本占 {sign_flip_pct:.2f}%。该不稳定性诊断及失败原因见内部 CSV/JSON，不作为贡献百分比结论。
- 分解与冻结基线的最大绝对差为 {max(decomp['analysis']['frozen_baseline_max_abs_differences'].values()):.3g}；分解加总闭合误差 {decomp['analysis']['arithmetic_component_closure_abs_error']:.3g}。

## 前沿条件情景（主映射 α=0.5）

| 期限(月) | 目标月 | 算力情景 | 点情景 | 无未来扰动参数/起点带 | 叠加未来扰动总带 | 带宽差 |
|---:|---|---|---:|---:|---:|---:|
{chr(10).join(forecast_lines)}

一步对比（规模—时间模型 MAE 应与持平基线一起解读）：

| 候选 | 起点数 | MAE |
|---|---:|---:|
{chr(10).join(candidate_lines)}

- 数据原点为 {forecast_diag['forecast_origin']}，目标月为 {', '.join(forecast_diag['target_months'])}；这些目标不是从本次运行日期 {forecast_diag['analysis_date_utc'][:10]} 起算。按当前运行月，{'；'.join(timing_items)}。现有 C1 数据只有 {forecast_diag['n_month_bins']} 个月、{forecast_diag['annual_horizon_rolling_origins']} 个同口径年度回测起点，故结果只可称条件情景，年度预测技能未验证，模拟带覆盖率未知；没有同口径观测来回测已过去的目标月或从当前月份更新预测。
- α=0.73 结构敏感性、全部越界标志、bootstrap 诊断及各输入/输出哈希见 `results/forecast/` 下 CSV/JSON。情景值不做静默截断；C4 三点算力序列只锚定假设，不与 C1 个体匹配。
- 目标队列版本核验：开放权重筛选共 {forecast_diag['n_open_weight_model_rows']} 条模型行，严格 revision/SHA 确认 {forecast_diag['strict_version_confirmed_rows']} 条；名称级版本状态为 {version_status_summary}。因此模型队列按冻结数据中的 C2 `Open Weights=Yes` 名称级标签构造，不应表述为逐模型的精确版本已核验。

## 未完成 / 采用边界

- Loss–Benchmark 桥接仍为 `BLOCKED_NO_FIT`，没有用 Loss 补造分数或替换 C1 预测。
- Q4 冻结基线保护检查：{freeze_text}。完整逐文件对照见 [`frozen_q4_baseline_sha256.json`](frozen_q4_baseline_sha256.json)。
- 当前会话的 AI 工具为 Codex / GPT-6 家族，但本任务可见记录不能确认具体模型变体及官方发布日期；G4 脚本头部已加待核验注释。如将结果用于论文或随程序提交，队伍仍须核对完整 AI 参与范围并按[2026 年竞赛 AI 工具规则](https://www.cmathc.org.cn/cpmcm/news/643.html)补齐开发机构、模型/版本、日期、输入与输出处理披露。因此本包是分析增量，不是参赛提交级最终包。
- Task 3 P2 状态来自 `q4_g4_task3_p2_review_receipt.json`；缺失或所绑定文件哈希变化时，状态自动退回待复核/失效。当前结论不替代整套 Q4 论文的最终检查。

## 主要产物

- 本计划：[`g4_execution_plan.md`](g4_execution_plan.md)
- 分解、Fieller 与发布者簇敏感性：`results/full/decomposition/`
- 前沿条件情景、回测及诊断：`results/forecast/`
- 文件 SHA-256 清单：[`g4_manifest.json`](g4_manifest.json)
"""
    REPORT.write_text(report, encoding="utf-8")

    included_files = sorted(
        path for path in G4.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts and path.suffix.lower() != ".pyc"
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": outcome,
        "review_gates": {
            "M1_model_contract": "PASS",
            "P1_minimum_execution": "PASS",
            "P2_task3_forecast": task3_p2,
            "decomposition_P2": "NOT_RECHECKED_IN_THIS_TASK3_TURN",
            "contest_submission_ai_disclosure": "PENDING_TEAM_VERIFICATION",
        },
        "frozen_baseline_verification": frozen,
        "analysis_metrics": {
            "decomposition_max_abs_baseline_difference": max(decomp["analysis"]["frozen_baseline_max_abs_differences"].values()),
            "common_support_n": decomp["analysis"]["fit_rows"],
            "observed_endpoint_change": decomp["analysis"]["observed_endpoint_mean_difference"],
            "ratio_shares_stably_identified": bool((fieller["ratio_stably_identified"].astype(bool)).all()),
            "publisher_bootstrap_successes": bootstrap_diag["successful_replicates"],
            "publisher_bootstrap_requested": bootstrap_diag["requested_replicates"],
            "forecast_annual_rolling_origins": forecast_diag["annual_horizon_rolling_origins"],
            "forecast_analysis_date_utc": forecast_diag["analysis_date_utc"],
            "forecast_target_timing_as_of_run": forecast_diag["target_timing_as_of_run"],
        },
        "files": {path.relative_to(ROOT).as_posix(): sha256(path) for path in included_files},
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={outcome} frozen_unchanged={frozen['unchanged']} report={REPORT} files={len(included_files)}")


if __name__ == "__main__":
    main()
