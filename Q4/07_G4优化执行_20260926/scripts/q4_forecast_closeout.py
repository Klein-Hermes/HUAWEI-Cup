#!/usr/bin/env python3
"""Formalize and audit the existing Q4 conditional forecast outputs.

This closeout does not refit the forecast model or modify the frozen Q4 v0.6
baseline. It independently recomputes one-step validation metrics from the
saved rolling-origin predictions, checks the saved scenario table, exports
figures, and writes a hash-bound closeout report/manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
# Static publication audit also checks the explicit svg.fonttype='none' setting.
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
G4 = ROOT / "Q4" / "07_G4优化执行_20260926"
FORECAST = G4 / "results" / "forecast"
FIGURES = G4 / "figures"
MANIFEST = FORECAST / "q4_forecast_closeout_manifest.json"
REPORT = FORECAST / "q4_forecast_closeout.md"
FIGURE_CONTRACT = FORECAST / "q4_forecast_figure_contract.md"
UPSTREAM_FORECAST = FORECAST / "q4_frontier_scenario_forecasts.csv"
UPSTREAM_BACKTEST = FORECAST / "q4_frontier_candidate_backtest.csv"
UPSTREAM_SUMMARY = FORECAST / "q4_frontier_candidate_summary.csv"
UPSTREAM_MONTHLY = FORECAST / "q4_frontier_monthly_rebuilt.csv"
UPSTREAM_MODEL_MANIFEST = FORECAST / "q4_frontier_forecast_manifest.json"
UPSTREAM_MODEL_DIAGNOSTICS = FORECAST / "q4_frontier_model_diagnostics.json"
G4_MANIFEST = G4 / "g4_manifest.json"
BASELINE_SNAPSHOT = G4 / "frozen_q4_baseline_sha256.json"

SKILL_ROOT = Path(r"C:\Users\86147\.codex\skills\math-modeling")
FIGURE_TOOLS = SKILL_ROOT / "tools" / "figure" / "scripts"
ROLE_UTILS = G4 / "utils"
sys.path.insert(0, str(ROLE_UTILS))
sys.path.insert(0, str(FIGURE_TOOLS))

from plot_style import PALETTE, add_panel_labels, apply_publication_style  # noqa: E402
from export_figure import export_figure  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


SCENARIO_ORDER = ["flat_compute", "half_log_growth", "historical_rate"]
SCENARIO_LABELS = {
    "flat_compute": "强放缓",
    "half_log_growth": "中度放缓",
    "historical_rate": "历史速度延续",
}
SCENARIO_MARKERS = {
    "flat_compute": "o",
    "half_log_growth": "s",
    "historical_rate": "^",
}
SCENARIO_COLORS = {
    "flat_compute": PALETTE["primary"],
    "half_log_growth": PALETTE["secondary"],
    "historical_rate": PALETTE["positive"],
}
METHOD_LABELS = {
    "last_value": "持平基线",
    "AR1": "AR(1)",
    "linear_time_trend": "线性时间趋势",
    "size_time_C4_ref": "规模—时间 + C4 情景",
}
METHOD_STYLES = {
    "last_value": (PALETTE["primary"], "o", "-"),
    "AR1": (PALETTE["accent"], "D", "--"),
    "linear_time_trend": (PALETTE["contrast"], "^", ":"),
    "size_time_C4_ref": (PALETTE["secondary"], "s", "-."),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    forecasts = pd.read_csv(UPSTREAM_FORECAST)
    backtest = pd.read_csv(UPSTREAM_BACKTEST, parse_dates=["origin_month", "target_month"])
    upstream_summary = pd.read_csv(UPSTREAM_SUMMARY)
    upstream_manifest = json.loads(UPSTREAM_MODEL_MANIFEST.read_text(encoding="utf-8"))
    return forecasts, backtest, upstream_summary, upstream_manifest


def validate_inputs(forecasts: pd.DataFrame, backtest: pd.DataFrame,
                    upstream_summary: pd.DataFrame, upstream_manifest: dict) -> None:
    required_forecast = {
        "forecast_origin", "horizon_months", "target_month", "scenario",
        "scaling_exponent_alpha_N_proportional_C_alpha",
        "frontier_point_C1_Average_P90", "conditional_simulation_q025",
        "conditional_simulation_q975", "annual_horizon_rolling_origin_count",
        "simulation_band_coverage_calibrated", "point_outside_score_domain_0_100",
        "simulation_band_outside_score_domain_0_100",
    }
    required_backtest = {
        "origin_month", "target_month", "method", "predicted_p90",
        "observed_p90", "error_pred_minus_observed", "absolute_error",
    }
    missing = required_forecast - set(forecasts.columns)
    if missing:
        raise ValueError(f"Forecast output missing columns: {sorted(missing)}")
    missing = required_backtest - set(backtest.columns)
    if missing:
        raise ValueError(f"Backtest output missing columns: {sorted(missing)}")
    if len(forecasts) != 12:
        raise ValueError(f"Expected 12 scenario rows, got {len(forecasts)}")
    if set(forecasts["horizon_months"].astype(int)) != {12, 24}:
        raise ValueError("Forecast horizons must be exactly 12 and 24 months")
    if set(np.round(forecasts["scaling_exponent_alpha_N_proportional_C_alpha"], 2)) != {0.50, 0.73}:
        raise ValueError("Expected alpha=0.50 primary and alpha=0.73 sensitivity rows")
    if set(forecasts["scenario"]) != set(SCENARIO_ORDER):
        raise ValueError("Forecast table does not contain exactly the three registered scenarios")
    if forecasts.duplicated(["horizon_months", "scenario", "scaling_exponent_alpha_N_proportional_C_alpha"]).any():
        raise ValueError("Duplicate horizon/scenario/alpha forecast rows")
    lower = forecasts["conditional_simulation_q025"].astype(float)
    point = forecasts["frontier_point_C1_Average_P90"].astype(float)
    upper = forecasts["conditional_simulation_q975"].astype(float)
    if not ((lower <= point) & (point <= upper)).all():
        raise ValueError("A conditional point forecast falls outside its simulated quantile band")
    if not ((lower >= 0) & (upper <= 100) & (point >= 0) & (point <= 100)).all():
        raise ValueError("A point or simulated band endpoint is outside the 0–100 score domain")
    if forecasts["annual_horizon_rolling_origin_count"].astype(int).ne(0).any():
        raise ValueError("Annual rolling-origin count changed; reassess the forecast-status conclusion")
    if forecasts["simulation_band_coverage_calibrated"].astype(bool).any():
        raise ValueError("Source table incorrectly marks the simulation bands calibrated")
    if forecasts["point_outside_score_domain_0_100"].astype(bool).any() or forecasts["simulation_band_outside_score_domain_0_100"].astype(bool).any():
        raise ValueError("Source table flags an out-of-domain forecast")
    if len(backtest) != 17:
        raise ValueError(f"Expected 17 one-step backtest rows, got {len(backtest)}")
    if set(backtest["method"]) != set(METHOD_LABELS):
        raise ValueError("Unexpected or missing one-step backtest method")
    expected_error = (backtest["predicted_p90"].astype(float)
                      - backtest["observed_p90"].astype(float))
    actual_error = backtest["error_pred_minus_observed"].astype(float)
    actual_absolute_error = backtest["absolute_error"].astype(float)
    if not np.allclose(actual_error, expected_error, rtol=0, atol=1e-10):
        raise ValueError("Backtest signed error is not predicted_p90 minus observed_p90")
    if not np.allclose(actual_absolute_error, np.abs(expected_error), rtol=0, atol=1e-10):
        raise ValueError("Backtest absolute_error is inconsistent with prediction and observation")
    if set(upstream_summary["method"]) != set(METHOD_LABELS):
        raise ValueError("Upstream summary and row-level backtest methods differ")
    if upstream_manifest.get("annual_validation_status", "").split(";")[0] != "NOT_VALIDATED":
        raise ValueError("Upstream annual-validation status is not NOT_VALIDATED")
    if int(upstream_manifest.get("simulations_per_scenario_horizon", 0)) != 12000:
        raise ValueError("Expected 12,000 simulation draws per scenario and horizon")


def recompute_backtest(backtest: pd.DataFrame,
                       upstream_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = (backtest.loc[backtest["method"].eq("last_value")]
                .set_index("target_month")["absolute_error"])
    rows = []
    for method in ["last_value", "AR1", "linear_time_trend", "size_time_C4_ref"]:
        group = backtest.loc[backtest["method"].eq(method)].sort_values("target_month").copy()
        matched = group["target_month"].isin(baseline.index)
        paired_baseline = baseline.loc[group.loc[matched, "target_month"]].to_numpy(dtype=float)
        method_paired_abs = group.loc[matched, "absolute_error"].to_numpy(dtype=float)
        baseline_mae = float(np.mean(paired_baseline)) if len(paired_baseline) else float("nan")
        mae = float(group["absolute_error"].mean())
        rmse = float(np.sqrt(np.mean(np.square(group["error_pred_minus_observed"].astype(float)))))
        median_ae = float(group["absolute_error"].median())
        mean_error = float(group["error_pred_minus_observed"].mean())
        paired_mae = float(method_paired_abs.mean()) if len(method_paired_abs) else float("nan")
        base_mae_for_same_rows = baseline_mae
        rows.append({
            "method": method,
            "method_label_zh": METHOD_LABELS[method],
            "n_one_step_origins": int(len(group)),
            "target_months": ", ".join(group["target_month"].dt.strftime("%Y-%m").tolist()),
            "mae_score_points": mae,
            "rmse_score_points": rmse,
            "median_absolute_error_score_points": median_ae,
            "mean_signed_error_pred_minus_observed": mean_error,
            "matched_last_value_origins": int(matched.sum()),
            "matched_last_value_mae_score_points": base_mae_for_same_rows,
            "paired_method_mae_score_points": paired_mae,
            "mae_difference_vs_matched_last_value": paired_mae - base_mae_for_same_rows,
            "mae_ratio_vs_matched_last_value": paired_mae / base_mae_for_same_rows if base_mae_for_same_rows else float("nan"),
            "interpretation": "small-sample one-step diagnostic; not annual skill validation",
        })
    result = pd.DataFrame(rows)
    upstream = upstream_summary.set_index("method")
    for row in result.itertuples(index=False):
        expected = float(upstream.loc[row.method, "MAE"])
        if abs(row.mae_score_points - expected) > 1e-9:
            raise ValueError(f"Recomputed MAE differs from upstream for {row.method}: {row.mae_score_points} vs {expected}")
    size = result.set_index("method").loc["size_time_C4_ref"]
    flat = result.set_index("method").loc["last_value"]
    if not (size["mae_score_points"] > flat["mae_score_points"]):
        raise ValueError("Expected the size-time candidate not to beat the flat baseline; review this conclusion")
    return result


def make_interval_status(forecasts: pd.DataFrame, upstream_manifest: dict) -> pd.DataFrame:
    rows = []
    for horizon, group in forecasts.groupby("horizon_months", sort=True):
        first = group.iloc[0]
        rows.append({
            "forecast_origin": str(first["forecast_origin"]),
            "horizon_months": int(horizon),
            "target_month": str(first["target_month"]),
            "as_of_run_status": str(first["target_status_as_of_run"]),
            "annual_rolling_origin_folds": int(first["annual_horizon_rolling_origin_count"]),
            "interval_type": "fixed-scenario empirical 2.5%–97.5% Monte Carlo quantiles",
            "empirical_coverage_status": "UNKNOWN_NOT_CALIBRATED",
            "reason": "zero same-protocol annual rolling origins; the target cannot be scored against a matched observation",
            "simulation_draws_per_scenario": int(upstream_manifest["simulations_per_scenario_horizon"]),
        })
    return pd.DataFrame(rows)


def plot_forecast_intervals(forecasts: pd.DataFrame) -> tuple[plt.Figure, str]:
    apply_publication_style(language="zh", width="report")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), constrained_layout=False)
    panel_specs = [(0.50, axes[0], "(a) 主设定  α = 0.50"), (0.73, axes[1], "(b) 结构敏感性  α = 0.73")]
    horizon_x = {12: 0.0, 24: 1.0}
    offsets = {"flat_compute": -0.16, "half_log_growth": 0.0, "historical_rate": 0.16}
    for alpha, ax, title in panel_specs:
        subset = forecasts.loc[np.isclose(forecasts["scaling_exponent_alpha_N_proportional_C_alpha"].astype(float), alpha)]
        for scenario in SCENARIO_ORDER:
            rows = subset.loc[subset["scenario"].eq(scenario)].sort_values("horizon_months")
            xs, ys, lows, highs = [], [], [], []
            for row in rows.itertuples(index=False):
                xs.append(horizon_x[int(row.horizon_months)] + offsets[scenario])
                ys.append(float(row.frontier_point_C1_Average_P90))
                lows.append(float(row.frontier_point_C1_Average_P90) - float(row.conditional_simulation_q025))
                highs.append(float(row.conditional_simulation_q975) - float(row.frontier_point_C1_Average_P90))
            ax.errorbar(
                xs, ys, yerr=np.array([lows, highs]),
                fmt=SCENARIO_MARKERS[scenario], color=SCENARIO_COLORS[scenario],
                markersize=4.5, capsize=2.5, elinewidth=1.0, linewidth=0,
                markeredgecolor="white", markeredgewidth=0.45,
                label=SCENARIO_LABELS[scenario], zorder=3,
            )
        ax.set_title(title, loc="left", pad=6)
        ax.set_xticks([0, 1], ["12个月\n2026-03", "24个月\n2027-03"])
        ax.set_xlim(-0.42, 1.42)
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("C1 Average P90（分，0–100）")
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.45, alpha=0.55)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.52, 0.91), frameon=False,
               columnspacing=1.05, handletextpad=0.4)
    fig.suptitle("C4 算力路径下的能力前沿条件情景", x=0.06, y=0.985, ha="left", fontsize=10)
    origin = str(forecasts["forecast_origin"].iloc[0])
    timing_parts = []
    for horizon in sorted(forecasts["horizon_months"].unique()):
        row = forecasts[forecasts["horizon_months"].eq(horizon)].iloc[0]
        months_from_run = int(row["target_months_from_run_month"])
        if months_from_run < 0:
            timing_parts.append(f"{row['target_month']} 已过去 {-months_from_run} 个月")
        elif months_from_run == 0:
            timing_parts.append(f"{row['target_month']} 为本次运行月")
        else:
            timing_parts.append(f"{row['target_month']} 距运行月 {months_from_run} 个月")
    fig.text(0.5, 0.052, f"数据原点 {origin}；" + "；".join(timing_parts) + "。", ha="center", va="bottom", fontsize=7)
    fig.text(0.5, 0.025, "点为条件情景值；误差线为经验 2.5%–97.5% 模拟分位数，未校准为预测区间。", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.21, wspace=0.27)
    return fig, "result_q4_conditional_forecast_intervals"


def plot_one_step_backtest(backtest: pd.DataFrame,
                           summary: pd.DataFrame) -> tuple[plt.Figure, str]:
    apply_publication_style(language="zh", width="report")
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=False)
    for method in ["last_value", "AR1", "linear_time_trend", "size_time_C4_ref"]:
        group = backtest.loc[backtest["method"].eq(method)].sort_values("target_month")
        metrics = summary.set_index("method").loc[method]
        color, marker, linestyle = METHOD_STYLES[method]
        label = f"{METHOD_LABELS[method]}（MAE={metrics['mae_score_points']:.2f}, n={len(group)}）"
        ax.plot(
            group["target_month"], group["error_pred_minus_observed"],
            color=color, marker=marker, linestyle=linestyle,
            markersize=4.0, linewidth=1.0, label=label, zorder=3,
        )
    ax.axhline(0, color="#222222", linewidth=0.75, linestyle="--", zorder=1)
    ax.set_ylabel("一步误差（预测 − 观测，分）")
    ax.set_xlabel("回测目标月")
    ax.set_title("一步滚动回测：误差围绕零线；起点数很少", loc="left", pad=7)
    ax.set_xticks(sorted(backtest["target_month"].unique()))
    ax.set_xticklabels(sorted(backtest["target_month"].dt.strftime("%Y-%m").unique()), rotation=0)
    y_min = min(float(backtest["error_pred_minus_observed"].min()) - 0.4, -0.4)
    y_max = max(float(backtest["error_pred_minus_observed"].max()) + 0.6, 0.6)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.23), ncol=2, frameon=False, columnspacing=1.2, handletextpad=0.5)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.32)
    return fig, "process_q4_one_step_backtest"


def export_and_audit(fig: plt.Figure, stem_name: str) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / stem_name
    preview = FIGURES / f"{stem_name}_preview.png"
    render_preview(fig, str(preview), dpi=600)
    axis_positions = [ax.get_position().bounds for ax in fig.axes]
    audit_verdict = print_report(audit_layout(fig))
    # visual_qa.audit_layout draws the figure; Matplotlib's tight-save side effect
    # may recalculate axes geometry, so restore the authored layout before export.
    for ax, position in zip(fig.axes, axis_positions):
        ax.set_position(position)
    if audit_verdict == "FAIL":
        raise RuntimeError(f"Visual layout audit failed for {stem_name}")
    paths = export_figure(
        fig,
        basename=str(stem),
        formats=["pdf", "svg", "png", "tiff"],  # bundle includes .pdf, .svg, .tiff
        size_inches=(7.2, 4.5 if stem_name.startswith("result_") else 3.5),
        dpi=600,
        grayscale_preview=False,
        tight=False,
    )
    grayscale_path = FIGURES / f"{stem_name}_grayscale.png"
    with Image.open(stem.with_suffix(".png")) as exported_png:
        exported_png.convert("L").save(grayscale_path, dpi=(600, 600))
    plt.close(fig)
    return [Path(path) for path in paths] + [grayscale_path, preview]


def write_contract() -> None:
    FIGURE_CONTRACT.write_text(
        "# Q4 前沿预测图表契约\n\n"
        "本补充只绘制已冻结预测结果的条件情景和一步验证，不重估模型、不添加题外预测。\n\n"
        "Source text stays editable: svg.fonttype=\"none\"; pdf.fonttype=42.\n\n"
        "| 图 | 类别/论点 | 证据与统计口径 | 版型与导出 |\n"
        "|---|---|---|---|\n"
        "| `result_q4_conditional_forecast_intervals` | 结果图：三种 C4 算力路径下的 12/24 月条件情景；读者能看出情景差异及区间较宽 | 12 条模型输出；两种 alpha 分面；点为条件情景点值，误差线为 12,000 次模拟经验 2.5%–97.5% 分位数；并非经校准预测区间 | 双栏宽 7.2×4.5 in；点区间图；SVG、PDF、600 dpi PNG/TIFF、灰度预览 |\n"
        "| `process_q4_one_step_backtest` | 过程/验证图：规模—时间模型一步误差整体高于持平基线，年度外推不能因此视为已验证 | 17 条滚动一步预测记录；模型组 n=5、AR(1) n=2；纵轴为预测减观测；不计算显著性 | 双栏宽 7.2×3.5 in；时间序列误差点线图；SVG、PDF、600 dpi PNG/TIFF、灰度预览 |\n\n"
        "统一图注边界：前沿响应是 C1 官方 Average P90（0–100 分）；数据原点 2025-03，目标 2026-03/2027-03，单独阅读图表时须结合本轮运行日期判定目标是否已到期。队列按 C2 名称级 `Open Weights=Yes` 标签筛选，模型未逐项经 revision/SHA 精确确认；C4 只作为情景锚点，不与 C1 个体匹配。误差带不是置信区间或已校准预测区间。现有 Q4 原始数据图和总体流程图仍见 `Q4/04_图表/`。\n",
        encoding="utf-8",
    )


def write_report(forecasts: pd.DataFrame, summary: pd.DataFrame,
                 interval_status: pd.DataFrame, upstream_manifest: dict) -> None:
    model_diagnostics = json.loads(UPSTREAM_MODEL_DIAGNOSTICS.read_text(encoding="utf-8"))
    version_status_summary = ", ".join(
        f"{key}={value}" for key, value in sorted(
            model_diagnostics["target_queue_version_match_status_counts"].items()))
    scenario_rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    ordered = forecasts.assign(_scenario_rank=forecasts["scenario"].map(scenario_rank))
    primary = ordered.loc[ordered["is_primary_alpha_half"].astype(bool)].sort_values(["horizon_months", "_scenario_rank"])
    sensitivity = ordered.loc[~ordered["is_primary_alpha_half"].astype(bool)].sort_values(["horizon_months", "_scenario_rank"])
    def forecast_lines(frame: pd.DataFrame) -> str:
        return "\n".join(
            f"| {int(r.horizon_months)} | {r.target_month} | {r.scenario_label_zh} | {float(r.frontier_point_C1_Average_P90):.2f} | {float(r.conditional_simulation_q025):.2f}–{float(r.conditional_simulation_q975):.2f} |"
            for r in frame.itertuples(index=False)
        )
    validation_lines = "\n".join(
        f"| {r.method_label_zh} | {int(r.n_one_step_origins)} | {r.mae_score_points:.3f} | {r.rmse_score_points:.3f} | {r.median_absolute_error_score_points:.3f} | {r.mean_signed_error_pred_minus_observed:+.3f} | {r.mae_difference_vs_matched_last_value:+.3f} |"
        for r in summary.itertuples(index=False)
    )
    interval_lines = "\n".join(
        f"| {int(r.horizon_months)} | {r.target_month} | {int(r.annual_rolling_origin_folds)} | `{r.empirical_coverage_status}` |"
        for r in interval_status.itertuples(index=False)
    )
    sims = int(upstream_manifest["simulations_per_scenario_horizon"])
    report = f"""# Q4 12/24 月前沿预测收口复核

**状态：** `CONDITIONAL_SCENARIOS_FORMALIZED; ANNUAL_SKILL_NOT_VALIDATED; INTERVAL_COVERAGE_UNKNOWN`。本补充把现有预测、有限回测、模拟区间及图表收为可复核增量；没有改写 Q4 v0.6 冻结基线。

## 目标与时间口径

- 预测原点：2025-03；目标月：2026-03（12 个月）和 2027-03（24 个月）。本轮计算运行于 2026-09-26，因此前者已过去约 6 个月，后者距运行月约 6 个月；它不是以 2026-09 为原点的实时 12/24 月预测。
- 响应量：当月新提交、C2 标为 Open Weights=Yes 的 C1 官方 `Average` 分数 P90，取值范围 0–100。它不同于 Q4 六族相对百分位指数，也不经 Loss–Benchmark 桥接。
- 版本核验：该开放权重目标队列有 {model_diagnostics['n_open_weight_model_rows']} 条模型行，严格 revision/SHA 确认 {model_diagnostics['strict_version_confirmed_rows']} 条；名称级状态为 `{version_status_summary}`。因此结果只适用于按 C2 名称级 `Open Weights=Yes` 标签构造的队列，不应表述为逐模型精确版本已核验。
- 输入月序列共 10 个月（2024-06 至 2025-03）；同口径 12/24 月滚动回测起点均为 0。Open LLM Leaderboard v2 于 2025-03 退役，无法用同协议附件观测更新当前原点或回测已过目标。

## 模型与条件假设

横截面模型为 `Average_i = β0 + βN·log₁₀(N_i) + βt·month_i + Type_i·θ + ε_i`。条件点值从最新前沿 `F0=41.2745` 出发：

`F(h,s,α) = F0 + h·βt + βN·α·log₁₀(G_s^(h/12))`，其中 `G_s` 是 C4 2022–2024 年训练算力 P90 所锚定的路径增长倍数。三情景分别为算力持平（强放缓）、对数增长率减半（中度放缓）和历史对数趋势延续；主映射 `N∝C^0.50`，`α=0.73` 为结构敏感性。C4 与 C1 没有精确个体匹配，因此不识别个体能力—算力关系。

该外推还条件于候选类型构成、残差分布和评分协议保持稳定。它满足题面所需的数值情景表达，但不是已验证的未来预测技能。

## 主设定结果：α=0.50

| 期限（月） | 目标月 | C4 情景 | 条件点值 | 模拟分位数带 2.5%–97.5% |
|---:|---|---|---:|---:|
{forecast_lines(primary)}

## 结构敏感性：α=0.73

| 期限（月） | 目标月 | C4 情景 | 条件点值 | 模拟分位数带 2.5%–97.5% |
|---:|---|---|---:|---:|
{forecast_lines(sensitivity)}

## 点预测验证

采用现有 1 个月滚动起点作诊断。`MAE=mean(|预测−观测|)`，`RMSE=sqrt(mean((预测−观测)^2))`；“相对持平 MAE 差”是在同一组目标月上与 last-value 基线配对计算。只有 5 个可用目标月，AR(1) 只有 2 个起点，不作显著性检验。

| 方法 | 起点数 | MAE（分） | RMSE（分） | 中位绝对误差（分） | 平均有符号误差（分） | 相对同目标持平 MAE 差（分） |
|---|---:|---:|---:|---:|---:|---:|
{validation_lines}

规模—时间 + C4 情景模型的一步 MAE 为 {summary.set_index('method').loc['size_time_C4_ref', 'mae_score_points']:.3f}，比相同 5 个目标月的持平基线 {summary.set_index('method').loc['last_value', 'mae_score_points']:.3f} 高 {summary.set_index('method').loc['size_time_C4_ref', 'mae_difference_vs_matched_last_value']:.3f} 分。它没有优于持平基线，因此只作为条件情景生成器；该一步比较不能代替年度验证。

## 区间与验证边界

| 期限（月） | 目标月 | 年度滚动起点 | 区间覆盖校准状态 |
|---:|---|---:|---|
{interval_lines}

每个情景/期限运行 {sims:,} 次模拟。分位数带传播月份簇系数 bootstrap、2025-03 发布者簇前沿 bootstrap，以及最近 5 个一步持平误差经中心化后的独立逐月重抽样。系数与起点前沿分别重抽样，未传播二者协方差；也未传播 C4 增速与 α 的不确定性。区间覆盖率未知，**不得称为经校准的 95% 预测区间**。全部 12 个条件点值和分位数端点均在 0–100 内，这只是范围核验，不是覆盖率证据。

年度区间的实际覆盖率无法验证：没有同口径年度滚动起点，且目标之后缺少相同排行榜协议的观测。若要把结论升级为经验证的预测，需要新增可比的后续月度前沿数据，形成至少多个年度滚动起点，并在留出期检验点误差和区间覆盖；现有附件无法完成这一步。

## 图表与复现

- 条件点值与模拟带：[`../../figures/result_q4_conditional_forecast_intervals.svg`](../../figures/result_q4_conditional_forecast_intervals.svg) / PNG；图中误差线为模拟分位数带，未校准。
- 一步误差：[`../../figures/process_q4_one_step_backtest.svg`](../../figures/process_q4_one_step_backtest.svg) / PNG。
- 逐方法指标：`q4_forecast_one_step_validation.csv`；区间验证状态：`q4_forecast_horizon_validation_status.csv`。
- 图表契约：`q4_forecast_figure_contract.md`。完整 12 行条件情景及原始回测仍以本目录的 `q4_frontier_scenario_forecasts.csv` 和 `q4_frontier_candidate_backtest.csv` 为源。
- 唯一复现命令：`D:/Anaconda/python.exe Q4/07_G4优化执行_20260926/scripts/q4_forecast_closeout.py`（从项目根目录运行）。输入与产物 SHA-256 见 `q4_forecast_closeout_manifest.json`。

**使用边界：** 本报告和图表是竞赛论文撰写参考材料；队伍需核对口径并用自己的语言写入论文。提交版文字、图号和交叉引用由队伍定稿。
"""
    REPORT.write_text(report, encoding="utf-8")


def write_manifest(inputs: list[Path], outputs: list[Path], summary: pd.DataFrame,
                   upstream_manifest: dict, p2_status: str,
                   figure_format_status: str,
                   question_figure_coverage_status: str) -> None:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
    }
    metrics = summary.set_index("method")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "conditional_scenarios_formalized_with_scoped_validation_limits",
        "forecast_origin": "2025-03",
        "scope": "supplemental forecast closeout only; frozen Q4 v0.6 results remain unchanged",
        "source_forecast_manifest": rel(UPSTREAM_MODEL_MANIFEST),
        "source_simulations_per_scenario_horizon": int(upstream_manifest["simulations_per_scenario_horizon"]),
        "annual_forecast_skill": "NOT_VALIDATED",
        "annual_rolling_origin_count": 0,
        "simulation_interval_coverage": "UNKNOWN_NOT_CALIBRATED",
        "review_gates": {
            "environment_check": "PASS",
            "data_profile": "PASS",
            "source_output_invariants": "PASS",
            "figure_layout_programmatic": "PASS",
            "figure_format_audit": figure_format_status,
            "Q4_question_figure_coverage": question_figure_coverage_status,
            "independent_P2": p2_status,
        },
        "one_step_validation": {
            "size_time_c4_ref_origins": int(metrics.loc["size_time_C4_ref", "n_one_step_origins"]),
            "size_time_c4_ref_mae": float(metrics.loc["size_time_C4_ref", "mae_score_points"]),
            "last_value_origins": int(metrics.loc["last_value", "n_one_step_origins"]),
            "last_value_mae": float(metrics.loc["last_value", "mae_score_points"]),
            "paired_mae_difference_vs_last_value": float(metrics.loc["size_time_C4_ref", "mae_difference_vs_matched_last_value"]),
            "AR1_origins": int(metrics.loc["AR1", "n_one_step_origins"]),
        },
        "environment": versions,
        "seed_in_upstream_forecast": int(upstream_manifest["seed"]),
        "input_sha256": {rel(path): sha256(path) for path in inputs},
        "output_sha256": {rel(path): sha256(path) for path in outputs},
        "frozen_Q4_baseline_status": json.loads(G4_MANIFEST.read_text(encoding="utf-8"))["frozen_baseline_verification"]["unchanged"],
        "frozen_Q4_baseline_snapshot_sha256": sha256(BASELINE_SNAPSHOT),
        "command": "D:/Anaconda/python.exe Q4/07_G4优化执行_20260926/scripts/q4_forecast_closeout.py",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_review_gates(p2_status: str, figure_format_status: str,
                        question_figure_coverage_status: str) -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Cannot record review gates before closeout manifest exists: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["review_gates"]["independent_P2"] = p2_status
    manifest["review_gates"]["figure_format_audit"] = figure_format_status
    manifest["review_gates"]["Q4_question_figure_coverage"] = question_figure_coverage_status
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="validate source rows and recompute metrics without writing figures/reports")
    parser.add_argument("--p2-status", choices=["PENDING", "PASS", "FAIL", "BLOCKED"], default="PENDING")
    parser.add_argument("--figure-format-status", choices=["PENDING", "PASS", "FAIL", "BLOCKED"], default="PENDING")
    parser.add_argument("--question-figure-coverage-status", choices=["PENDING", "PASS", "FAIL", "BLOCKED"], default="PENDING")
    parser.add_argument("--record-gates-only", action="store_true", help="update external/independent review statuses in the existing manifest without regenerating artifacts")
    args = parser.parse_args()

    if args.record_gates_only:
        record_review_gates(args.p2_status, args.figure_format_status,
                            args.question_figure_coverage_status)
        print(f"REVIEW_GATES_RECORDED manifest={rel(MANIFEST)} p2={args.p2_status} figures={args.figure_format_status} coverage={args.question_figure_coverage_status}")
        return 0

    forecasts, backtest, upstream_summary, upstream_manifest = load_inputs()
    validate_inputs(forecasts, backtest, upstream_summary, upstream_manifest)
    summary = recompute_backtest(backtest, upstream_summary)
    interval_status = make_interval_status(forecasts, upstream_manifest)
    if args.smoke:
        print(f"P1_SMOKE_OK forecasts={len(forecasts)} backtest_rows={len(backtest)}")
        print(summary[["method", "n_one_step_origins", "mae_score_points", "rmse_score_points"]].to_string(index=False))
        print("annual_origins=0 annual_skill=NOT_VALIDATED interval_coverage=UNKNOWN_NOT_CALIBRATED")
        return 0

    FORECAST.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summary_path = FORECAST / "q4_forecast_one_step_validation.csv"
    interval_path = FORECAST / "q4_forecast_horizon_validation_status.csv"
    summary.to_csv(summary_path, index=False, float_format="%.12g", encoding="utf-8-sig")
    interval_status.to_csv(interval_path, index=False, encoding="utf-8-sig")
    write_contract()
    exported: list[Path] = []
    for fig, stem in [plot_forecast_intervals(forecasts), plot_one_step_backtest(backtest, summary)]:
        exported.extend(export_and_audit(fig, stem))
    write_report(forecasts, summary, interval_status, upstream_manifest)

    # G4_MANIFEST is read for the frozen-baseline boolean but omitted from this
    # hash list to avoid a circular manifest dependency; the baseline snapshot itself is hashed.
    inputs = [Path(__file__), UPSTREAM_FORECAST, UPSTREAM_BACKTEST, UPSTREAM_SUMMARY, UPSTREAM_MONTHLY,
              UPSTREAM_MODEL_DIAGNOSTICS,
              UPSTREAM_MODEL_MANIFEST, BASELINE_SNAPSHOT,
              G4 / "q4_frontier_forecast_method_v0.1.md",
              G4 / "q4_task3_forecast_amendment_v0.8.md",
              G4 / "scripts" / "q4_g4_frontier_scenario_model.py",
              G4 / "utils" / "plot_style.py"]
    outputs = [summary_path, interval_path, FIGURE_CONTRACT, REPORT, *exported]
    write_manifest(inputs, outputs, summary, upstream_manifest, args.p2_status,
                   args.figure_format_status, args.question_figure_coverage_status)
    print(f"CLOSEOUT_OK report={rel(REPORT)} figures={len(exported)} p2_status={args.p2_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
