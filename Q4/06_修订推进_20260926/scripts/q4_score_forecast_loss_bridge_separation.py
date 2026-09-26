"""Build a focused evidence snapshot for Q4's independent forecast/bridge tracks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
Q4 = ROOT / "Q4"
REVISION = Q4 / "06_修订推进_20260926"
BASE_RESULTS = Q4 / "03_结果" / "results"
SMOKE_FORECAST = REVISION / "results" / "smoke" / "forecast_model"
FINAL_FORECAST = REVISION / "results" / "forecast"
OUT = REVISION / "results" / "smoke"

FINAL_FORECAST_MANIFEST = FINAL_FORECAST / "q4_frontier_forecast_manifest.json"
if FINAL_FORECAST_MANIFEST.is_file():
    _final_forecast_meta = json.loads(FINAL_FORECAST_MANIFEST.read_text(encoding="utf-8"))
    SCENARIO_DIR = FINAL_FORECAST if not _final_forecast_meta.get("smoke_run", True) else SMOKE_FORECAST
else:
    SCENARIO_DIR = SMOKE_FORECAST

INPUTS = {
    "frontier": BASE_RESULTS / "q4_frontier_monthly.csv",
    "forecast_feasibility": BASE_RESULTS / "q4_forecast_feasibility.csv",
    "ar1_backtest": BASE_RESULTS / "q4_ar1_backtest_summary.csv",
    "short_backtest": BASE_RESULTS / "q4_forecast_backtest_1month_summary.csv",
    "bridge_gate": BASE_RESULTS / "q4_loss_benchmark_feasibility.csv",
    "bridge_manifest": BASE_RESULTS / "q4_loss_benchmark_feasibility_manifest.json",
    "c4_year_summary": BASE_RESULTS / "q4_c4_year_summary.csv",
    "modeling_report": Q4 / "01_方案说明" / "q4_modeling_report.md",
    "historical_plan": Q4 / "05_审计交付" / "q4_next_steps_plan.md",
    "historical_contract": Q4 / "01_方案说明" / "q4_model_contract_v0.6.md",
    "forecast_code": Q4 / "02_代码" / "scripts" / "q4_modeling_analysis.py",
    "bridge_code": Q4 / "02_代码" / "scripts" / "q4_loss_bridge_feasibility.py",
    "contract_v0_7": REVISION / "q4_model_contract_v0.7_addendum.md",
    "contract_v0_8": REVISION / "q4_task3_forecast_amendment_v0.8.md",
    "forecast_method": REVISION / "q4_frontier_forecast_method_v0.1.md",
    "forecast_scenario_code": REVISION / "scripts" / "q4_frontier_scenario_model.py",
    "forecast_manifest": SCENARIO_DIR / "q4_frontier_forecast_manifest.json",
    "forecast_report": SCENARIO_DIR / "q4_frontier_scenario_report.md",
    "scenario_forecasts": SCENARIO_DIR / "q4_frontier_scenario_forecasts.csv",
    "compute_scenarios": SCENARIO_DIR / "q4_c4_compute_scenarios.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def month_ordinal(month: str) -> int:
    year, mon = (int(part) for part in month.split("-"))
    return year * 12 + mon - 1


def month_string(ordinal: int) -> str:
    year, zero_based_month = divmod(ordinal, 12)
    return f"{year:04d}-{zero_based_month + 1:02d}"


def exact_horizon_origins(months: set[str], horizon: int) -> int:
    """Count observed t,t+h pairs; this is not a count of fitted validation folds."""
    return sum(month_string(month_ordinal(month) + horizon) in months for month in months)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def verify_forecast_manifest(manifest: dict, forecast_rows: list[dict[str, str]]) -> dict[str, object]:
    """Audit the direct-score smoke artifact without promoting stale output values."""
    provenance_mismatches: list[str] = []
    if manifest.get("script_sha256") != sha256(INPUTS["forecast_scenario_code"]):
        provenance_mismatches.append("forecast script SHA-256 differs from the smoke manifest")
    if "NOT_VALIDATED" not in manifest.get("annual_validation_status", ""):
        raise ValueError("C1 scenario manifest must keep annual forecast skill unvalidated")
    if int(manifest.get("simulations_per_scenario_horizon", 0)) <= 0:
        raise ValueError("C1 scenario manifest is missing its simulation count")

    contract_method_hashes = manifest.get("contract_and_method_sha256", {})
    expected_contracts = {
        "q4_task3_forecast_amendment_v0.8.md": INPUTS["contract_v0_8"],
        "q4_frontier_forecast_method_v0.1.md": INPUTS["forecast_method"],
    }
    for name, path in expected_contracts.items():
        if contract_method_hashes.get(name) != sha256(path):
            provenance_mismatches.append(f"{name} SHA-256 differs from the smoke manifest")

    # The direct-score model declares only C1/C4/backtest inputs; it must not import Loss data.
    declared_inputs = manifest.get("inputs", {})
    if any("loss" in name.lower() or "benchmark" in name.lower() for name in declared_inputs):
        raise ValueError("C1 scenario manifest unexpectedly declares a Loss/Benchmark input")
    for relative_name, expected_hash in declared_inputs.items():
        source = ROOT / Path(relative_name)
        if not source.is_file() or sha256(source) != expected_hash:
            raise ValueError(f"C1 scenario input is missing or changed: {relative_name}")

    manifest_outputs = manifest.get("outputs", {})
    for name, expected_hash in manifest_outputs.items():
        output = SCENARIO_DIR / name
        if not output.is_file() or sha256(output) != expected_hash:
            raise ValueError(f"C1 scenario smoke output is missing or changed: {name}")

    if len(forecast_rows) != 12:
        raise ValueError(f"Expected 12 scenario smoke rows (2 horizons × 3 paths × 2 exponents), got {len(forecast_rows)}")
    if {int(row["horizon_months"]) for row in forecast_rows} != {12, 24}:
        raise ValueError("C1 scenario smoke output must cover 12 and 24 months")
    if {row["scenario"] for row in forecast_rows} != {"historical_rate", "half_log_growth", "flat_compute"}:
        raise ValueError("C1 scenario smoke output does not contain the three contracted C4 paths")
    if {float(row["scaling_exponent_alpha_N_proportional_C_alpha"]) for row in forecast_rows} != {0.5, 0.73}:
        raise ValueError("C1 scenario smoke output is missing the alpha=0.5/0.73 pair")
    if any(row["simulation_band_coverage_calibrated"].lower() != "false" for row in forecast_rows):
        raise ValueError("C1 simulation bands must remain explicitly uncalibrated")
    if any(int(row["annual_horizon_rolling_origin_count"]) != 0 for row in forecast_rows):
        raise ValueError("C1 scenario artifact conflicts with the zero annual-origin status")
    for row in forecast_rows:
        for column in ("frontier_point_C1_Average_P90", "conditional_simulation_q025", "conditional_simulation_q975"):
            if as_float(row[column]) is None:
                raise ValueError(f"C1 scenario value is missing: {column}")

    return {
        "scenario_output_hashes_match": True,
        "source_and_contract_hashes_match": not provenance_mismatches,
        "provenance_mismatches": provenance_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a focused Q4 score/bridge evidence snapshot.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUT,
        help="Output directory (defaults to the revision's results/smoke folder).",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("Output directory must be inside the project root") from error

    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required evidence files are missing:\n" + "\n".join(missing))

    frontier = read_csv(INPUTS["frontier"])
    forecast_feasibility = read_csv(INPUTS["forecast_feasibility"])
    ar1_rows = read_csv(INPUTS["ar1_backtest"])
    short_rows = read_csv(INPUTS["short_backtest"])
    gate_rows = read_csv(INPUTS["bridge_gate"])
    bridge_manifest = json.loads(INPUTS["bridge_manifest"].read_text(encoding="utf-8"))
    scenario_manifest = json.loads(INPUTS["forecast_manifest"].read_text(encoding="utf-8"))
    scenario_rows = read_csv(INPUTS["scenario_forecasts"])
    compute_scenarios = read_csv(INPUTS["compute_scenarios"])
    scenario_report = INPUTS["forecast_report"].read_text(encoding="utf-8")

    if not frontier:
        raise ValueError("C1 frontier table has no observed months")
    required_frontier = {"submission_month", "monthly_entry_p90_frontier_C1_average_0_100"}
    if not required_frontier.issubset(frontier[0]):
        raise ValueError(f"C1 frontier table is missing {sorted(required_frontier - set(frontier[0]))}")
    months = {row["submission_month"] for row in frontier}
    if len(months) != len(frontier):
        raise ValueError("C1 frontier must have at most one row per submission month")
    scores = [as_float(row["monthly_entry_p90_frontier_C1_average_0_100"]) for row in frontier]
    if any(value is None or not 0 <= value <= 100 for value in scores):
        raise ValueError("C1 Average P90 values must be finite and within the source 0-100 scale")
    scenario_provenance = verify_forecast_manifest(scenario_manifest, scenario_rows)
    if len(compute_scenarios) != 3:
        raise ValueError("C4 scenario evidence must contain the three declared paths")
    for marker in ("conditional_scenario_only", "NOT_VALIDATED", "UNKNOWN"):
        if marker not in scenario_report:
            raise ValueError(f"C1 scenario report is missing the required label: {marker}")

    forecast_by_horizon = {int(row["horizon_months"]): row for row in forecast_feasibility}
    ar1 = ar1_rows[0] if ar1_rows else {}
    short_by_method = {row["method"]: row for row in short_rows}
    pooled = next((row for row in gate_rows if row["scope"] == "pooled_C5_C6_deduplicated"), None)
    if pooled is None:
        raise ValueError("Pooled C5/C6 Loss gate row is missing")

    bridge_status = pooled["prefit_gate_status"]
    fit_executed = bool(bridge_manifest.get("fit_executed"))
    if bridge_status != bridge_manifest.get("status"):
        raise ValueError("Loss bridge CSV status and manifest status disagree")
    if bridge_status == "BLOCKED_NO_FIT" and fit_executed:
        raise ValueError("Blocked Loss bridge manifest incorrectly reports a fit")

    span = max(month_ordinal(month) for month in months) - min(month_ordinal(month) for month in months)
    row_template = {
        "track": "", "target": "", "horizon_months": "", "target_month": "", "scenario": "",
        "scaling_exponent": "", "status": "", "evidence_stage": "", "point_forecast": "",
        "simulation_q025": "", "simulation_q975": "", "annual_forecast_skill": "",
        "simulation_band_coverage": "", "fit_executed": "", "loss_used": "",
        "observed_month_bins": "", "calendar_span_months": "", "exact_horizon_origins": "",
        "unique_bridge_pairs": "", "verified_model_versions": "", "loss_unit": "", "reason": "",
    }
    rows: list[dict[str, object]] = []

    for horizon in (12, 24):
        source = forecast_by_horizon[horizon]
        if source["status"] != "not_estimable_from_supplied_history":
            raise ValueError(f"Unexpected unconditional C1 status for horizon {horizon}")
        if any(as_float(source[column]) is not None for column in (
            "point_forecast", "prediction_interval_low", "prediction_interval_high"
        )):
            raise ValueError(f"Unconditional C1 status is inconsistent with numeric output for horizon {horizon}")
        row = dict(row_template)
        row.update({
            "track": "unconditional_direct_C1_score_trend",
            "target": f"monthly_open_weight_C1_Average_P90_{horizon}m",
            "horizon_months": horizon,
            "target_month": source["target_month"],
            "status": "not_estimable_from_supplied_history",
            "evidence_stage": "historical_feasibility",
            "annual_forecast_skill": "NOT_ESTIMABLE_FROM_HISTORY",
            "simulation_band_coverage": "NOT_APPLICABLE",
            "fit_executed": "not_applicable_to_feasibility_snapshot",
            "loss_used": "false",
            "observed_month_bins": len(frontier),
            "calendar_span_months": span,
            "exact_horizon_origins": exact_horizon_origins(months, horizon),
            "reason": "C1 history and exact-horizon support are insufficient; this is independent of Loss bridge eligibility.",
        })
        rows.append(row)

    scenario_status = "conditional_scenario_only"
    if scenario_manifest.get("smoke_run"):
        scenario_stage = "legacy_smoke_source_mismatch" if not scenario_provenance["source_and_contract_hashes_match"] else "smoke_only"
    else:
        scenario_stage = "full_run" if scenario_provenance["source_and_contract_hashes_match"] else "full_run_source_mismatch"
    coverage_status = "UNKNOWN"
    for source in scenario_rows:
        horizon = int(source["horizon_months"])
        row = dict(row_template)
        row.update({
            "track": "conditional_direct_C1_score_scenario",
            "target": f"monthly_open_weight_C1_Average_P90_{horizon}m",
            "horizon_months": horizon,
            "target_month": source["target_month"],
            "scenario": source["scenario_label_zh"],
            "scaling_exponent": source["scaling_exponent_alpha_N_proportional_C_alpha"],
            "status": scenario_status,
            "evidence_stage": scenario_stage,
            "point_forecast": source["frontier_point_C1_Average_P90"] if scenario_provenance["source_and_contract_hashes_match"] else "",
            "simulation_q025": source["conditional_simulation_q025"] if scenario_provenance["source_and_contract_hashes_match"] else "",
            "simulation_q975": source["conditional_simulation_q975"] if scenario_provenance["source_and_contract_hashes_match"] else "",
            "annual_forecast_skill": "NOT_VALIDATED",
            "simulation_band_coverage": coverage_status,
            "fit_executed": "true",
            "loss_used": "false",
            "observed_month_bins": len(frontier),
            "calendar_span_months": span,
            "exact_horizon_origins": exact_horizon_origins(months, horizon),
            "reason": (
                "Direct C1 Average scenario output; annual skill is unvalidated and band coverage is unknown."
                if scenario_provenance["source_and_contract_hashes_match"] else
                "Direct C1 Average scenario output; values omitted because source/contract hashes do not match its manifest. Annual skill is unvalidated and band coverage is unknown."
            ),
        })
        rows.append(row)

    row = dict(row_template)
    row.update({
        "track": "loss_benchmark_bridge",
        "target": "validated_Loss_to_C1_Benchmark_mapping",
        "status": bridge_status,
        "evidence_stage": "prefit_gate",
        "annual_forecast_skill": "NOT_APPLICABLE",
        "simulation_band_coverage": "NOT_APPLICABLE",
        "fit_executed": str(fit_executed).lower(),
        "loss_used": "true",
        "unique_bridge_pairs": pooled["unique_high_content_pairs"],
        "verified_model_versions": pooled["verified_model_version_count"],
        "loss_unit": pooled["loss_unit"],
        "reason": pooled["blocked_reasons"],
    })
    rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "q4_score_forecast_loss_bridge_separation.csv"
    write_csv(table_path, rows)

    last_value = short_by_method.get("last_value", {})
    linear = short_by_method.get("linear_time_trend", {})
    ar1_mae = as_float(ar1.get("ar1_mae", ""))
    persistence_mae = as_float(ar1.get("persistence_mae", ""))
    size_time = next((item for item in read_csv(SCENARIO_DIR / "q4_frontier_candidate_summary.csv") if item["method"] == "size_time_C4_ref"), None)
    if size_time is None:
        raise ValueError("C1 smoke candidate summary is missing size_time_C4_ref")
    loss_low = float(pooled["high_loss_min"])
    loss_high = float(pooled["high_loss_max"])
    report_path = output_dir / "q4_score_forecast_loss_bridge_separation_report.md"

    primary_rows = [r for r in scenario_rows if float(r["scaling_exponent_alpha_N_proportional_C_alpha"]) == 0.5]
    primary_rows.sort(key=lambda r: (int(r["horizon_months"]), r["scenario_label_zh"]))
    scenario_table = "\n".join(
        f"| {r['horizon_months']} | {r['target_month']} | {r['scenario_label_zh']} | "
        f"{float(r['frontier_point_C1_Average_P90']):.2f} | {float(r['conditional_simulation_q025']):.2f} | "
        f"{float(r['conditional_simulation_q975']):.2f} |"
        for r in primary_rows
    ) if scenario_provenance["source_and_contract_hashes_match"] else "| — | — | 当前版本尚无可核验数值 | — | — | — |"
    sensitivity_rows = [r for r in scenario_rows if float(r["scaling_exponent_alpha_N_proportional_C_alpha"]) == 0.73]
    sensitivity_table = "\n".join(
        f"| {r['horizon_months']} | {r['scenario_label_zh']} | {float(r['frontier_point_C1_Average_P90']):.2f} |"
        for r in sorted(sensitivity_rows, key=lambda r: (int(r["horizon_months"]), r["scenario_label_zh"]))
    ) if scenario_provenance["source_and_contract_hashes_match"] else "| — | 当前版本尚无可核验敏感性数值 | — |"

    report = f"""# C1 分数预测与 Loss–Benchmark 桥接分轨核查

**范围：** 完成截图中的 Task 5：把 Loss–Benchmark 桥接作为独立交付。C1 直接分数预测单独核对；不拿它替代 Loss 桥接，也不因为 Loss 门控未过而阻断其 C1 分数响应模型。

## 结论

| 路径 | 响应量/目标 | 当前状态 | 证据成熟度 |
|---|---|---|---|
| 无条件 C1 分数趋势 | C1 `Average` 原始分数构造的开放权重月度 P90 | `not_estimable_from_supplied_history`（12/24 月） | 历史可行性核查；年度同口径滚动起点为 0 |
| C4 条件下的直接 C1 分数情景 | C1 `Average` 原始分数 P90 | `{scenario_status}`；年度技能 `NOT_VALIDATED`，模拟带覆盖率 `UNKNOWN` | `{scenario_stage}`，共 {len(scenario_rows)} 个情景输出 |
| Loss–Benchmark 桥接 | 有协议与目标支持的 Loss → Benchmark 映射 | `{bridge_status}` | 独立预拟合门控；拟合执行 `{str(fit_executed).lower()}` |

### 1. 无条件分数趋势

目标是每月新提交且 C2 明确 `Open Weights=Yes` 的模型 C1 `Average`（0–100）分数 P90。现有同口径序列共 {len(frontier)} 个月、跨 {span} 个月；精确 12/24 月历史配对数为 {exact_horizon_origins(months, 12)}/{exact_horizon_origins(months, 24)}。因此当前不能报告经相应期限验证的无条件点预测或区间。这个限制来自 C1 历史与验证支持，不来自 Loss 门控。

- AR(1) 一步扩展回测：{ar1.get('ar1_origins', 'NA')} 个起点，MAE {float(ar1.get('ar1_mae', 'nan')):.3f}；持平基线 MAE {float(ar1.get('persistence_mae', 'nan')):.3f}。
- 另一个一步比较中，持平基线 MAE {float(last_value.get('mae', 'nan')):.3f}，线性时间趋势 MAE {float(linear.get('mae', 'nan')):.3f}。

### 2. C4 条件情景：直接预测 C1 分数

当前直接分数情景以 C1 原始 `Average` 为响应，不经过 Loss 桥；使用 C4 年度算力 P90 锚定三条假设路径，并使用 `N∝C^0.5` 主假设及 `N∝C^0.73` 敏感性。C4 与 C1 没有个体精确名称匹配，故这只是条件情景外推。当前产物阶段为 `{scenario_stage}`，manifest、脚本、合同、方法及输入/输出哈希均核对通过={str(scenario_provenance['source_and_contract_hashes_match']).lower()}。

主假设 `N∝C^0.5` 的条件数值（不是年度验证预测）：

| 期限(月) | 目标月 | C4 算力情景 | C1 Average P90 点值 | 模拟分位数 2.5% | 模拟分位数 97.5% |
|---:|---|---|---:|---:|---:|
{scenario_table}

结构敏感性 `N∝C^0.73`：

| 期限(月) | C4 算力情景 | C1 Average P90 点值 |
|---:|---|---:|
{sensitivity_table}

规模—时间模型一步 MAE 为 {float(size_time['MAE']):.3f}，持平基线为 {float(last_value.get('mae', 'nan')):.3f}；前者没有优于持平基线。年度技能仍 `NOT_VALIDATED`，模拟带覆盖率仍 `UNKNOWN`。这条 C1 路径与 Loss 桥分开；数值只按假设情景解释，不称作经验证预测。

### 3. Loss–Benchmark 桥接：独立阻断

- 高可比 Loss 范围：{loss_low:.4f}–{loss_high:.4f}；合并去重后 {pooled['unique_high_content_pairs']} 个配对，已核验模型版本数 {pooled['verified_model_version_count']}。
- C5/C6 高可比记录存在 {pooled['c5_c6_exact_cross_table_duplicates']} 条跨表完全重复；Loss 单位为 `{pooled['loss_unit']}`，评测协议、revision/运行身份与批准目标网格没有核实。
- 状态继续为 `{bridge_status}`，`fit_executed={str(fit_executed).lower()}`。没有拟合，也没有用缺少可核验适用范围与误差的文献关系替代。
- C1 情景脚本的 manifest 输入不含 Loss/Benchmark 数据；这是直接分数模型与桥接分轨的可复核证据。

## 合同与追溯

`q4_model_contract_v0.7_addendum.md` 记录 C1 与桥接拆分；`q4_task3_forecast_amendment_v0.8.md` 允许题目所需的条件情景数值展示，但保留年度技能和模拟带覆盖率限制。本次输出复核当前情景 manifest 的输入/输出哈希及脚本、合同、方法哈希；仅将哈希相符的产物纳入数值表。

独立状态表：`q4_score_forecast_loss_bridge_separation.csv`。本报告自身的复现 manifest 记录所有源文件与产物 SHA-256。
"""
    report_path.write_text(report, encoding="utf-8")

    source_hashes = {name: sha256(path) for name, path in INPUTS.items()}
    outputs = [table_path, report_path]
    final_run_verified = SCENARIO_DIR == FINAL_FORECAST and scenario_provenance["source_and_contract_hashes_match"]
    manifest_path = output_dir / "q4_score_forecast_loss_bridge_separation_manifest.json"
    manifest = {
        "status": "separation_snapshot_generated",
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "script_sha256": sha256(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): source_hashes[name] for name, path in INPUTS.items()},
        "outputs_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in outputs},
        "unconditional_score_trend_track": {
            "status": "not_estimable_from_supplied_history",
            "observed_month_bins": len(frontier),
            "calendar_span_months": span,
            "rolling_origins_12m": exact_horizon_origins(months, 12),
            "rolling_origins_24m": exact_horizon_origins(months, 24),
        },
        "compute_conditioned_score_track": {
            "status": scenario_status,
            "evidence_stage": scenario_stage,
            "annual_forecast_skill": "NOT_VALIDATED",
            "simulation_band_coverage": coverage_status,
            "scenario_row_count": len(scenario_rows),
            "loss_used": False,
            "scenario_provenance": scenario_provenance,
            "full_run_manifest_verified": final_run_verified,
        },
        "loss_bridge_track": {"status": bridge_status, "fit_executed": fit_executed},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"unconditional_C1=not_estimable_from_supplied_history bins={len(frontier)} span={span} origins_12m={exact_horizon_origins(months, 12)} origins_24m={exact_horizon_origins(months, 24)}")
    print(f"conditional_C1={scenario_status} stage={scenario_stage} rows={len(scenario_rows)} source_hashes_match={str(scenario_provenance['source_and_contract_hashes_match']).lower()} annual_skill=NOT_VALIDATED band_coverage=UNKNOWN")
    print(f"loss_bridge={bridge_status} fit_executed={str(fit_executed).lower()} unique_pairs={pooled['unique_high_content_pairs']} verified_versions={pooled['verified_model_version_count']}")
    print(f"table={table_path.relative_to(ROOT).as_posix()}")
    print(f"report={report_path.relative_to(ROOT).as_posix()}")
    print(f"manifest={manifest_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
