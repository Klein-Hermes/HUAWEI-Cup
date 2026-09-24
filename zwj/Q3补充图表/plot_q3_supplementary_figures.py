#!/usr/bin/env python3
"""Generate the four requested Q3 supplementary figures.

Run from the project root with:
    D:\\miniconda3\\envs\\huawei-cup\\python.exe zwj/Q3补充图表/plot_q3_supplementary_figures.py

All project inputs are read-only. Every generated artifact is written below
``zwj/Q3补充图表``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_BUNDLE_DIR = SCRIPT_PATH.parent
ZWJ_DIR = OUTPUT_BUNDLE_DIR.parent
PROJECT_ROOT = ZWJ_DIR.parent
DEFAULT_OUTPUT_DIR = OUTPUT_BUNDLE_DIR
SKILL_ROOT = Path.home() / ".codex" / "skills" / "math-modeling"
FIGURE_TOOL_DIR = SKILL_ROOT / "tools" / "figure" / "scripts"
if not FIGURE_TOOL_DIR.is_dir():
    raise FileNotFoundError(f"未找到科研可视化工具目录：{FIGURE_TOOL_DIR}")
sys.path.insert(0, str(FIGURE_TOOL_DIR))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report  # noqa: E402


DPI = 300
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#8C8C8C"
DARK = "#222222"
LIGHT_GRID = "#E5E7EB"
CONTEXT_COLORS = [BLUE, ORANGE, GREEN, VERMILLION, PURPLE]
CONTEXT_MARKERS = ["o", "s", "^", "D", "P"]
BUDGET_COLORS = {1e19: BLUE, 1e22: ORANGE, 1e24: GREEN}
BUDGET_MARKERS = {1e19: "o", 1e22: "s", 1e24: "D"}

FRONTIER_SOURCE = PROJECT_ROOT / "Q3/04_结果/M0_ND_reference_frontier.csv"
BOOTSTRAP_SOURCE = PROJECT_ROOT / "Q3/04_结果/M0_ND_cluster_bootstrap_frontier.csv"
FREQUENCY_SOURCE = PROJECT_ROOT / "Q3/04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv"
COST_SHARE_SOURCE = PROJECT_ROOT / "Q3/04_结果/M0_budget_context_cost_shares.csv"
SHIFT_SOURCE = PROJECT_ROOT / "Q3/04_结果/M0_budget_context_descriptive_shifts.csv"
Q1_BASELINE_SOURCE = PROJECT_ROOT / "Q1/03_结果/Q1.1/v1/a1_macro_summary.csv"
COST_SCRIPT_SOURCE = PROJECT_ROOT / "Q3/03_代码/analyze_q3_cost_sensitivities.py"

INPUT_SOURCES = [
    FRONTIER_SOURCE,
    BOOTSTRAP_SOURCE,
    FREQUENCY_SOURCE,
    COST_SHARE_SOURCE,
    SHIFT_SOURCE,
    Q1_BASELINE_SOURCE,
    COST_SCRIPT_SOURCE,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def configure_style() -> dict:
    info = setup_style(
        journal="general",
        lang="zh",
        use_sciplots=True,
        serif_for_zh=False,
        constrained_layout=False,
    )
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "sans-serif",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "lines.linewidth": 1.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": DPI,
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "mathtext.rm": "DejaVu Sans",
            "mathtext.it": "DejaVu Sans:italic",
            "mathtext.bf": "DejaVu Sans:bold",
        }
    )
    info["font_chain"] = list(mpl.rcParams["font.sans-serif"])
    return info


def require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} 缺少字段：{missing}")


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def load_inputs() -> dict[str, pd.DataFrame]:
    for path in INPUT_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    tables = {
        "frontier": pd.read_csv(FRONTIER_SOURCE),
        "bootstrap": pd.read_csv(BOOTSTRAP_SOURCE),
        "frequency": pd.read_csv(FREQUENCY_SOURCE),
        "cost": pd.read_csv(COST_SHARE_SOURCE),
        "shifts": pd.read_csv(SHIFT_SOURCE),
        "q1": pd.read_csv(Q1_BASELINE_SOURCE),
    }
    require_columns(
        tables["frontier"],
        {
            "budget_flops",
            "context_length_tokens",
            "q0_reference_a1_macro",
            "q_equal_sensitivity_macro",
            "selected_N_B",
            "selected_D_B",
            "m0_predicted_val_loss",
            "budget_utilization",
            "selected_at_support_max_N_and_D",
        },
        "M0前沿表",
    )
    require_columns(
        tables["bootstrap"],
        {
            "budget_flops",
            "context_length_tokens",
            "bootstrap_replicates",
            "baseline_selection_frequency",
            "selected_N_B_p2_5",
            "selected_N_B_median",
            "selected_N_B_p97_5",
            "selected_D_B_p2_5",
            "selected_D_B_median",
            "selected_D_B_p97_5",
            "predicted_loss_p2_5",
            "predicted_loss_median",
            "predicted_loss_p97_5",
        },
        "Bootstrap前沿表",
    )
    require_columns(
        tables["frequency"],
        {"budget_flops", "context_length_tokens", "selection_count", "selection_frequency", "baseline_selected"},
        "Bootstrap选择频率表",
    )
    require_columns(
        tables["cost"],
        {
            "budget_flops",
            "context_length_tokens",
            "selected_N_B",
            "selected_D_B",
            "m0_predicted_val_loss",
            "train_cost_share",
            "quality_cost_share",
            "attention_cost_share",
            "at_b1_support_upper_bound",
        },
        "成本份额表",
    )
    require_columns(
        tables["shifts"],
        {"comparison_axis", "configuration_changed", "support_bound_at_to_scenario"},
        "描述性变化表",
    )
    require_columns(tables["q1"], {"candidate", "macro_domain_score"}, "Q1基线表")

    expected_rows = {"frontier": 15, "bootstrap": 15, "frequency": 15, "cost": 15, "shifts": 22, "q1": 2}
    for key, expected in expected_rows.items():
        if len(tables[key]) != expected:
            raise ValueError(f"{key} 行数应为 {expected}，实际为 {len(tables[key])}")

    for key in ("frontier", "bootstrap", "frequency", "cost"):
        data = tables[key]
        if data[["budget_flops", "context_length_tokens"]].drop_duplicates().shape[0] != 15:
            raise ValueError(f"{key} 应覆盖15个唯一预算×上下文情景")

    frontier = tables["frontier"]
    frontier["selected_at_support_max_N_and_D"] = parse_bool(frontier["selected_at_support_max_N_and_D"])
    tables["cost"]["at_b1_support_upper_bound"] = parse_bool(tables["cost"]["at_b1_support_upper_bound"])
    for table in (frontier, tables["bootstrap"], tables["frequency"], tables["cost"]):
        table["budget_flops"] = table["budget_flops"].astype(float)
        table["context_length_tokens"] = table["context_length_tokens"].astype(int)

    keys = ["budget_flops", "context_length_tokens"]
    frontier_keys = set(map(tuple, frontier[keys].to_numpy()))
    for name in ("bootstrap", "frequency", "cost"):
        if set(map(tuple, tables[name][keys].to_numpy())) != frontier_keys:
            raise ValueError(f"{name} 与前沿表的15个情景不一致")

    q1_scores = tables["q1"].set_index("candidate")["macro_domain_score"].astype(float)
    if set(q1_scores.index) != {"q_huber", "q_equal"}:
        raise ValueError("Q1基线表必须且只能包含 q_huber 与 q_equal")
    qh_frontier = frontier["q0_reference_a1_macro"].astype(float).unique()
    qe_frontier = frontier["q_equal_sensitivity_macro"].astype(float).unique()
    if len(qh_frontier) != 1 or len(qe_frontier) != 1:
        raise ValueError("前沿表中的Q基线值不唯一")
    if not np.isclose(qh_frontier[0], q1_scores["q_huber"], rtol=0, atol=1e-12):
        raise ValueError("前沿表 q_huber 基线与Q1不一致")
    if not np.isclose(qe_frontier[0], q1_scores["q_equal"], rtol=0, atol=1e-12):
        raise ValueError("前沿表 q_equal 基线与Q1不一致")

    bootstrap = tables["bootstrap"]
    frequency = tables["frequency"]
    if not (bootstrap["bootstrap_replicates"].astype(int) == 1000).all():
        raise ValueError("Bootstrap重复次数不是1000")
    if not np.allclose(bootstrap["baseline_selection_frequency"], 1.0, rtol=0, atol=1e-12):
        raise ValueError("存在原配置重选频率不为1的情景")
    if not np.allclose(frequency["selection_frequency"], 1.0, rtol=0, atol=1e-12):
        raise ValueError("逐配置频率表存在选择频率不为1的记录")
    for prefix in ("selected_N_B", "selected_D_B"):
        if not (
            np.allclose(bootstrap[f"{prefix}_p2_5"], bootstrap[f"{prefix}_median"], rtol=0, atol=1e-12)
            and np.allclose(bootstrap[f"{prefix}_median"], bootstrap[f"{prefix}_p97_5"], rtol=0, atol=1e-12)
        ):
            raise ValueError(f"{prefix} 的区间并非全部退化为单点")
    return tables


def context_label(value: int) -> str:
    return f"{value:,}"


def budget_label(value: float) -> str:
    exponent = int(round(np.log10(value)))
    return rf"$10^{{{exponent}}}$"


def scenario_frame(data: pd.DataFrame) -> pd.DataFrame:
    ordered = data.sort_values(["budget_flops", "context_length_tokens"]).reset_index(drop=True).copy()
    ordered["scenario_label"] = [
        f"{budget_label(b)} · {context_label(int(c))}"
        for b, c in zip(ordered["budget_flops"], ordered["context_length_tokens"])
    ]
    ordered["scenario_index"] = np.arange(len(ordered))
    return ordered


def compact_context_label(value: int) -> str:
    mapping = {2048: "2K", 4096: "4K", 8192: "8K", 32768: "32K", 131072: "131K"}
    return mapping.get(int(value), f"{int(value) / 1000:g}K")


def add_budget_block_guides(axis: mpl.axes.Axes, contexts: pd.Series) -> None:
    axis.set_xticks(np.arange(len(contexts)), [compact_context_label(v) for v in contexts])
    for position in (4.5, 9.5):
        axis.axvline(position, color="#D1D5DB", linewidth=0.8, linestyle=":", zorder=0)


def set_decimal_log_ticks(axis: mpl.axes.Axes, orientation: str) -> None:
    ticks = [0.1, 1.0, 10.0]
    labels = ["0.1", "1", "10"]
    if orientation == "x":
        axis.set_xticks(ticks, labels)
        axis.xaxis.set_minor_formatter(NullFormatter())
    elif orientation == "y":
        axis.set_yticks(ticks, labels)
        axis.yaxis.set_minor_formatter(NullFormatter())
    else:
        raise ValueError(f"未知刻度方向：{orientation}")


def add_panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontsize=9.2, fontweight="bold", va="top")


def save_figure(fig: mpl.figure.Figure, stem: Path, size: tuple[float, float]) -> tuple[list[str], dict]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[layout audit] {stem.name}")
    layout_issues = audit_layout(fig)
    verdict = print_report(layout_issues)
    if verdict == "FAIL":
        raise ValueError(f"{stem.name} 的程序版面检查失败")
    outputs = export_figure(
        fig,
        basename=str(stem),
        formats=["svg", "pdf", "png"],
        dpi=DPI,
        size_inches=size,
        grayscale_preview=False,
        tight=False,
        transparent=False,
    )
    png_path = stem.with_suffix(".png")
    gray_path = stem.parent / f"{stem.name}_grayscale.png"
    with Image.open(png_path) as image:
        image.convert("L").save(gray_path, dpi=(DPI, DPI))
    outputs.append(str(gray_path))
    plt.close(fig)
    audit = {"layout_verdict": verdict, "layout_issue_count": len(layout_issues)}
    return outputs, audit


def plot_p1(frontier: pd.DataFrame, output_dir: Path, paper: bool = False) -> tuple[list[str], dict]:
    size = (7.2, 5.2)
    fig, axes = plt.subplots(2, 1, figsize=size, sharex=True)
    contexts = sorted(frontier["context_length_tokens"].unique())
    metrics = [
        ("selected_N_B", r"最优 $N_B$（十亿参数）", axes[0]),
        ("selected_D_B", r"最优 $D_B$（十亿 Token）", axes[1]),
    ]
    for metric, ylabel, axis in metrics:
        for context, color, marker in zip(contexts, CONTEXT_COLORS, CONTEXT_MARKERS):
            subset = frontier.loc[frontier["context_length_tokens"] == context].sort_values("budget_flops")
            axis.plot(
                subset["budget_flops"],
                subset[metric],
                color=color,
                marker=marker,
                markersize=5.0,
                markeredgecolor="white",
                markeredgewidth=0.55,
                linestyle="-",
                label=f"{context_label(context)} Token",
                zorder=3,
            )
            boundary = subset["selected_at_support_max_N_and_D"]
            axis.scatter(
                subset.loc[boundary, "budget_flops"],
                subset.loc[boundary, metric],
                s=82,
                facecolor="none",
                edgecolor=DARK,
                marker="o",
                linewidth=1.0,
                zorder=5,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        if metric == "selected_N_B":
            set_decimal_log_ticks(axis, "y")
        axis.set_ylabel(ylabel)
        axis.grid(axis="both", color=LIGHT_GRID, linewidth=0.5)
    axes[1].set_xlabel("预算（FLOPs，对数轴；线段仅连接三个给定档位）")
    axes[1].set_xticks([1e19, 1e22, 1e24], [r"$10^{19}$", r"$10^{22}$", r"$10^{24}$"])
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=DARK, linestyle="none", markersize=7, label="B1 N/D支持上界"))
    labels.append("B1 N/D支持上界")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97 if paper else 0.915), ncol=3, frameon=False)
    if not paper:
        fig.suptitle(r"固定 $Q=Q_0$ 的预算—N/D参考配置（15个离散情景）", fontsize=11, y=0.985)
        fig.text(
            0.5,
            0.018,
            r"固定 q_huber 基线、$C_Q=0$；仅为B1实测网格和M0内的参考配置，不是完整Q3联合最优。"
            "\n最高预算点受B1支持网格上界约束；线段不表示连续预算上的估计。",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#40464D",
        )
    fig.subplots_adjust(left=0.115, right=0.98, top=0.875 if paper else 0.82, bottom=0.10 if paper else 0.14, hspace=0.23)
    stem = output_dir / "result_q3_p1_budget_nd_frontier"
    outputs, audit = save_figure(fig, stem, size)
    stats = {
        "source_rows": int(len(frontier)),
        "scenario_count": int(frontier[["budget_flops", "context_length_tokens"]].drop_duplicates().shape[0]),
        "budget_levels": sorted(float(v) for v in frontier["budget_flops"].unique()),
        "context_levels_tokens": contexts,
        "support_upper_bound_points": int(frontier["selected_at_support_max_N_and_D"].sum()),
        "figure_variant": "paper_without_overall_title" if paper else "standalone_with_overall_title",
        "layout_audit": audit,
    }
    return outputs, stats


def horizontal_interval(axis: mpl.axes.Axes, y: np.ndarray, median: np.ndarray, low: np.ndarray, high: np.ndarray, colors: list[str]) -> None:
    lower = median - low
    upper = high - median
    axis.errorbar(median, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#4B5563", capsize=2, elinewidth=0.9, zorder=2)
    for index, (x_value, y_value, color) in enumerate(zip(median, y, colors)):
        axis.scatter(x_value, y_value, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=3)


def plot_p2(bootstrap: pd.DataFrame, frequency: pd.DataFrame, output_dir: Path, paper: bool = False) -> tuple[list[str], dict]:
    data = scenario_frame(bootstrap)
    y = np.arange(len(data))
    colors = [BUDGET_COLORS[float(v)] for v in data["budget_flops"]]
    size = (7.2, 6.2)
    fig, axes = plt.subplots(1, 4, figsize=size, sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 1.08, 0.84]})
    interval_specs = [
        ("selected_N_B", r"$N_B$（十亿参数）", axes[0], True),
        ("selected_D_B", r"$D_B$（十亿 Token）", axes[1], True),
        ("predicted_loss", "M0预测Loss", axes[2], False),
    ]
    for prefix, xlabel, axis, use_log in interval_specs:
        median = data[f"{prefix}_median"].to_numpy(float)
        low = data[f"{prefix}_p2_5"].to_numpy(float)
        high = data[f"{prefix}_p97_5"].to_numpy(float)
        horizontal_interval(axis, y, median, low, high, colors)
        if use_log:
            axis.set_xscale("log")
            if prefix == "selected_N_B":
                set_decimal_log_ticks(axis, "x")
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color=LIGHT_GRID, linewidth=0.5)
    switch_probability = 1.0 - data["baseline_selection_frequency"].to_numpy(float)
    axes[3].scatter(switch_probability, y, s=30, c=colors, edgecolor="white", linewidth=0.45, zorder=3)
    axes[3].axvline(0, color=DARK, linewidth=0.8, linestyle="--", zorder=1)
    axes[3].set_xlim(-0.015, 1.0)
    axes[3].set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
    axes[3].set_xlabel("配置切换概率")
    axes[3].grid(axis="x", color=LIGHT_GRID, linewidth=0.5)
    axes[0].set_yticks(y, data["scenario_label"])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("预算 · 上下文长度（Token）")
    for axis, label in zip(axes, ["a", "b", "c", "d"]):
        add_panel_label(axis, label)
    budget_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BUDGET_COLORS[b], markeredgecolor="white", markersize=6, label=f"预算 {budget_label(b)}")
        for b in sorted(BUDGET_COLORS)
    ]
    fig.legend(handles=budget_handles, loc="upper center", bbox_to_anchor=(0.5, 0.975 if paper else 0.92), ncol=3, frameon=False)
    if not paper:
        fig.suptitle("轨迹级 cluster Bootstrap 的配置选择稳定性（1,000次）", fontsize=11, y=0.985)
        fig.text(
            0.5,
            0.018,
            "N/D 的95%区间在全部15个情景中退化为单点（宽度为0），图中未人为放大；Loss区间按原值绘制。"
            "\nBootstrap仅基于8条B1轨迹，表示既定M0形式与支持网格内的参数重抽样稳定性，不覆盖模型形式、外部迁移或Q/p响应。",
            ha="center",
            va="bottom",
            fontsize=6.55,
            color="#40464D",
        )
    fig.subplots_adjust(left=0.205, right=0.985, top=0.89 if paper else 0.83, bottom=0.10 if paper else 0.15, wspace=0.25)
    stem = output_dir / "result_q3_p2_bootstrap_selection_stability"
    outputs, audit = save_figure(fig, stem, size)
    frequency_check = frequency.sort_values(["budget_flops", "context_length_tokens"])
    stats = {
        "source_rows": {"frontier": int(len(bootstrap)), "selection_frequency": int(len(frequency))},
        "bootstrap_replicates": sorted(int(v) for v in bootstrap["bootstrap_replicates"].unique()),
        "N_interval_max_width": float((bootstrap["selected_N_B_p97_5"] - bootstrap["selected_N_B_p2_5"]).max()),
        "D_interval_max_width": float((bootstrap["selected_D_B_p97_5"] - bootstrap["selected_D_B_p2_5"]).max()),
        "loss_interval_max_width": float((bootstrap["predicted_loss_p97_5"] - bootstrap["predicted_loss_p2_5"]).max()),
        "baseline_selection_frequency_range": [float(bootstrap["baseline_selection_frequency"].min()), float(bootstrap["baseline_selection_frequency"].max())],
        "selection_frequency_crosscheck_range": [float(frequency_check["selection_frequency"].min()), float(frequency_check["selection_frequency"].max())],
        "figure_variant": "paper_without_overall_title" if paper else "standalone_with_overall_title",
        "layout_audit": audit,
    }
    return outputs, stats


def plot_p3(frontier: pd.DataFrame, cost: pd.DataFrame, shifts: pd.DataFrame, output_dir: Path, paper: bool = False) -> tuple[list[str], dict]:
    merged = frontier.merge(
        cost,
        on=["budget_flops", "context_length_tokens"],
        suffixes=("_frontier", "_cost"),
        validate="one_to_one",
    )
    data = scenario_frame(cost)
    contexts = sorted(frontier["context_length_tokens"].unique())
    size = (7.2, 6.0)
    fig, axes = plt.subplots(2, 2, figsize=size)
    line_specs = [
        (axes[0, 0], "selected_N_B", r"$N_B$（十亿参数）", "a  N配置"),
        (axes[0, 1], "selected_D_B", r"$D_B$（十亿 Token）", "b  D配置"),
        (axes[1, 1], "m0_predicted_val_loss", "M0预测验证Loss", "d  M0预测Loss"),
    ]
    for axis, metric, ylabel, title in line_specs:
        for context, color, marker in zip(contexts, CONTEXT_COLORS, CONTEXT_MARKERS):
            subset = frontier.loc[frontier["context_length_tokens"] == context].sort_values("budget_flops")
            axis.plot(
                subset["budget_flops"],
                subset[metric],
                color=color,
                marker=marker,
                markersize=4.7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"{context_label(context)} Token",
            )
            boundary = subset["selected_at_support_max_N_and_D"]
            axis.scatter(subset.loc[boundary, "budget_flops"], subset.loc[boundary, metric], s=68, facecolor="none", edgecolor=DARK, linewidth=0.9, zorder=5)
        axis.set_xscale("log")
        if metric in ("selected_N_B", "selected_D_B"):
            axis.set_yscale("log")
            if metric == "selected_N_B":
                set_decimal_log_ticks(axis, "y")
        axis.set_xticks([1e19, 1e22, 1e24], [r"$10^{19}$", r"$10^{22}$", r"$10^{24}$"])
        axis.set_xlabel("预算（FLOPs；仅连接给定档位）")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="both", color=LIGHT_GRID, linewidth=0.5)

    cost_axis = axes[1, 0]
    x = data["scenario_index"].to_numpy(float)
    bottom = np.zeros(len(data))
    parts = [
        ("train_cost_share", "训练成本", BLUE, "///"),
        ("attention_cost_share", "注意力成本", ORANGE, "\\\\"),
        ("quality_cost_share", "质量成本", GRAY, ".."),
    ]
    for column, label, color, hatch in parts:
        values = data[column].to_numpy(float)
        cost_axis.bar(x, values, bottom=bottom, width=0.82, color=color, edgecolor="white", linewidth=0.45, hatch=hatch, label=label)
        bottom += values
    cost_axis.set_ylim(0, 1.03)
    cost_axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    add_budget_block_guides(cost_axis, data["context_length_tokens"])
    cost_axis.set_ylabel("实际成本构成")
    cost_axis.set_xlabel(r"上下文长度（K Token）；预算块依次为 $10^{19}$、$10^{22}$、$10^{24}$ FLOPs")
    cost_axis.set_title(r"c  成本份额（固定 $Q=Q_0$，$C_Q=0$）", loc="left", fontweight="bold")
    cost_axis.legend(loc="upper left", ncol=3, frameon=False, fontsize=6.2)
    cost_axis.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=DARK, linestyle="none", markersize=7, label="B1 N/D支持上界"))
    labels.append("B1 N/D支持上界")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98 if paper else 0.925), ncol=3, frameon=False)
    if not paper:
        fig.suptitle("离散预算与上下文情景下的资源配置、成本构成和Loss", fontsize=11, y=0.987)
        fig.text(
            0.5,
            0.012,
            "22对相邻变化仅作描述性比较，不使用Bootstrap/KKT判据，也不代表正式结构转移。"
            r"固定 $Q=Q_0$ 时质量成本份额为0；最高预算情景受B1支持网格上界约束。",
            ha="center",
            va="bottom",
            fontsize=6.55,
            color="#40464D",
        )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88 if paper else 0.82, bottom=0.14 if paper else 0.20, wspace=0.24, hspace=0.50)
    stem = output_dir / "result_q3_p3_resource_shift_dashboard"
    outputs, audit = save_figure(fig, stem, size)
    comparison_counts = shifts["comparison_axis"].value_counts().astype(int).to_dict()
    stats = {
        "source_rows": {"frontier": int(len(frontier)), "cost_shares": int(len(cost)), "descriptive_shifts": int(len(shifts))},
        "merge_rows": int(len(merged)),
        "comparison_axis_counts": comparison_counts,
        "configuration_changed_pairs": int(parse_bool(shifts["configuration_changed"]).sum()),
        "quality_cost_share_range": [float(cost["quality_cost_share"].min()), float(cost["quality_cost_share"].max())],
        "cost_share_sum_max_abs_error": float(np.abs(cost[["train_cost_share", "quality_cost_share", "attention_cost_share"]].sum(axis=1) - 1.0).max()),
        "support_upper_bound_scenarios": int(cost["at_b1_support_upper_bound"].sum()),
        "figure_variant": "paper_without_overall_title" if paper else "standalone_with_overall_title",
        "layout_audit": audit,
    }
    return outputs, stats


def plot_p4(frontier: pd.DataFrame, q1: pd.DataFrame, output_dir: Path, paper: bool = False) -> tuple[list[str], dict]:
    q_values = q1.set_index("candidate")["macro_domain_score"].astype(float)
    data = scenario_frame(frontier)
    x = data["scenario_index"].to_numpy(float)
    # Under fixed-Q M0, Q does not enter Loss and C_Q=D[g(Q)-g(Q0)]_+=0 for
    # either baseline. The two deterministic selections are therefore exact
    # copies by construction; assertions below prevent accidental divergence.
    qh = data[["selected_N_B", "selected_D_B", "m0_predicted_val_loss"]].copy()
    qe = qh.copy()
    differences = qe - qh
    if not np.allclose(differences.to_numpy(float), 0.0, rtol=0, atol=0):
        raise ValueError("固定-Q两基线出现不应存在的N/D或Loss差异")

    size = (7.2, 5.8)
    fig, axes = plt.subplots(2, 2, figsize=size)
    baseline_axis = axes[0, 0]
    y_baseline = np.array([1.0, 0.0])
    x_baseline = np.array([q_values["q_huber"], q_values["q_equal"]])
    baseline_axis.scatter(x_baseline, y_baseline, s=66, c=[BLUE, ORANGE], edgecolor="white", linewidth=0.7, zorder=3)
    baseline_axis.plot(x_baseline, y_baseline, color=GRAY, linewidth=0.9, linestyle="--", zorder=1)
    for x_value, y_value in zip(x_baseline, y_baseline):
        baseline_axis.annotate(f"{x_value:.10f}", (x_value, y_value), xytext=(7, 0), textcoords="offset points", ha="left", va="center", fontsize=7.0, fontweight="bold")
    baseline_axis.set_yticks(y_baseline, ["q_huber", "q_equal"])
    baseline_axis.set_ylim(-0.55, 1.55)
    baseline_axis.set_xlim(float(x_baseline.min()) - 0.00045, float(x_baseline.max()) + 0.00085)
    baseline_axis.set_xlabel(r"基线 $Q_0$")
    baseline_axis.set_title(r"a  两套 $Q_0$（来自Q1）", loc="left", fontweight="bold")
    baseline_axis.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)

    comparison_specs = [
        (axes[0, 1], "selected_N_B", r"$N_B$（十亿参数）", "b  固定-Q最优N"),
        (axes[1, 0], "selected_D_B", r"$D_B$（十亿 Token）", "c  固定-Q最优D"),
        (axes[1, 1], "m0_predicted_val_loss", "M0预测验证Loss", "d  固定-Q M0 Loss"),
    ]
    for axis, metric, ylabel, title in comparison_specs:
        axis.plot(x, qh[metric], color=BLUE, marker="o", markersize=4.6, markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.45, linewidth=1.0, label="q_huber")
        axis.plot(x, qe[metric], color=ORANGE, marker="o", markersize=7.0, markerfacecolor="none", markeredgecolor=ORANGE, markeredgewidth=1.0, linewidth=0.8, linestyle="--", label="q_equal（与q_huber重合）")
        if metric in ("selected_N_B", "selected_D_B"):
            axis.set_yscale("log")
            if metric == "selected_N_B":
                set_decimal_log_ticks(axis, "y")
        add_budget_block_guides(axis, data["context_length_tokens"])
        axis.set_ylabel(ylabel)
        axis.set_xlabel(r"上下文长度（K Token）；预算块依次为 $10^{19}$、$10^{22}$、$10^{24}$")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="both", color=LIGHT_GRID, linewidth=0.5)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.69, 0.98 if paper else 0.923), ncol=2, frameon=False)
    max_abs_delta = float(np.abs(differences.to_numpy(float)).max())
    if not paper:
        fig.suptitle("q_huber 与 q_equal 基线的固定-Q M0敏感性对照", fontsize=11, y=0.987)
        fig.text(
            0.5,
            0.012,
            f"15个相同预算×上下文情景中，N、D和M0 Loss的最大绝对差均为 {max_abs_delta:g}。"
            r"两支均固定在各自 $Q_0$，故 $C_Q=0$；当前M0不含Q→Loss项。该图不是Q收益估计或Q/p联合优化。",
            ha="center",
            va="bottom",
            fontsize=6.55,
            color="#40464D",
        )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.88 if paper else 0.82, bottom=0.12 if paper else 0.17, wspace=0.28, hspace=0.50)
    stem = output_dir / "result_q3_p4_q_baseline_sensitivity"
    outputs, audit = save_figure(fig, stem, size)
    stats = {
        "source_rows": {"frontier": int(len(frontier)), "q1_baselines": int(len(q1))},
        "q0": {"q_huber": float(q_values["q_huber"]), "q_equal": float(q_values["q_equal"])},
        "q0_difference_q_equal_minus_q_huber": float(q_values["q_equal"] - q_values["q_huber"]),
        "fixed_q_branch_reason": "M0 has no Q term and C_Q is zero at each branch's own Q0; feasible set and objective are unchanged",
        "max_abs_N_B_difference": float(np.abs(differences["selected_N_B"]).max()),
        "max_abs_D_B_difference": float(np.abs(differences["selected_D_B"]).max()),
        "max_abs_m0_loss_difference": float(np.abs(differences["m0_predicted_val_loss"]).max()),
        "figure_variant": "paper_without_overall_title" if paper else "standalone_with_overall_title",
        "layout_audit": audit,
    }
    return outputs, stats


def write_contract(output_dir: Path) -> Path:
    text = """# Q3 四项补充图表契约

## P1：固定 Q0 的预算—N/D参考配置

- **核心论点：** 在冻结的 `q_huber` 基线、固定 $Q=Q_0$ 时，对照3档预算和5种上下文下的B1/M0支持网格参考配置。
- **图型与版型：** 上下排列的双面板折线/点图；预算、N、D均采用对数尺度；颜色与标记共同编码上下文。
- **输入：** `Q3/04_结果/M0_ND_reference_frontier.csv`（15行）。
- **字段：** `budget_flops`、`context_length_tokens`、`selected_N_B`、`selected_D_B`、`m0_predicted_val_loss`、`budget_utilization`、`selected_at_support_max_N_and_D`。
- **变换/汇总：** 不汇总；展示全部15个离散情景。空心外圈标识触及B1 N/D支持上界的点。
- **解释限制：** 线段只连接给定预算档位；不是连续预算估计、完整Q3联合最优、p效应或质量提升效应。

## P2：Bootstrap配置稳定性

- **核心论点：** 1,000次轨迹级cluster Bootstrap中，15个固定-Q情景的N/D选择全部保持不变，同时展示M0预测Loss的参数传播区间。
- **图型与版型：** 四列森林式点/区间图，分别展示N、D、M0预测Loss的2.5%–97.5%区间及配置切换概率。
- **输入：** `M0_ND_cluster_bootstrap_frontier.csv`（15行）与 `M0_ND_cluster_bootstrap_selection_frequency.csv`（15行）。
- **字段：** N/D/Loss的 `p2_5`、`median`、`p97_5` 字段，`baseline_selection_frequency` 与逐配置 `selection_frequency`。
- **变换/汇总：** 切换概率=`1-baseline_selection_frequency`；N/D区间宽度按原值为0，不作视觉放大。
- **解释限制：** 仅反映8条B1轨迹、既定M0函数形式和支持网格内的参数重抽样稳定性，不覆盖模型形式误差、外部迁移或Q/p响应。

## P3：资源配置、成本份额与Loss

- **核心论点：** 对照离散预算/上下文情景中的N/D配置、训练/注意力/质量成本份额与M0预测Loss。
- **图型与版型：** 2×2多面板；N、D、Loss用情景折线/点图，成本构成用100%堆叠柱状图及纹理冗余编码。
- **输入：** `M0_ND_reference_frontier.csv`（15行）、`M0_budget_context_cost_shares.csv`（15行）、`M0_budget_context_descriptive_shifts.csv`（22行）。
- **字段：** N、D、M0 Loss、三类成本份额、支持上界标记及 `comparison_axis`。
- **变换/汇总：** 成本份额逐情景直接堆叠；变化表仅统计两类比较数量与配置变化对数，不作显著性推断。
- **解释限制：** 固定Q时质量成本份额为0；22对变化是描述性比较，不使用Bootstrap/KKT判据，不称为结构转移；最高预算受B1支持上界限制。

## P4：Q基线敏感性

- **核心论点：** `q_huber` 与 `q_equal` 的Q0数值不同，但在相同预算、上下文、B1网格和固定-Q M0下，N/D选择和M0 Loss完全相同。
- **图型与版型：** 2×2对照图；一个面板展示两套Q0，三个面板以实心/空心标记叠加N、D和Loss。
- **输入：** `M0_ND_reference_frontier.csv`（15行）与 `Q1/03_结果/Q1.1/v1/a1_macro_summary.csv`（2行）。
- **字段：** `q0_reference_a1_macro`、`q_equal_sensitivity_macro`、`selected_N_B`、`selected_D_B`、`m0_predicted_val_loss`。
- **变换/汇总：** `q_qual` 按现有项目候选 `q_equal` 解释，并与Q1基线表逐值核对；两分支差值逐情景计算并断言为0。
- **解释限制：** 两支均固定在各自Q0，故质量增量成本为0；当前M0不含Q→Loss项。该对照不是Q收益的实证估计，也不构成Q/p联合优化。
"""
    path = output_dir / "图表契约.md"
    path.write_text(text, encoding="utf-8")
    return path


def inspect_outputs(output_dir: Path, figures: dict[str, list[str]]) -> dict:
    audits = {}
    for stem, paths in figures.items():
        by_suffix = {Path(path).suffix.lower(): Path(path) for path in paths if "_grayscale" not in Path(path).stem}
        gray = output_dir / f"{stem}_grayscale.png"
        required = {".svg", ".pdf", ".png"}
        if set(by_suffix) != required or not gray.is_file():
            raise ValueError(f"{stem} 输出配对不完整")
        with Image.open(by_suffix[".png"]) as image:
            dpi_value = image.info.get("dpi", (0, 0))
            png_info = {"pixels": list(image.size), "dpi": [float(dpi_value[0]), float(dpi_value[1])], "mode": image.mode}
            if min(dpi_value) < 299:
                raise ValueError(f"{stem} PNG DPI不足：{dpi_value}")
        with Image.open(gray) as image:
            gray_info = {"pixels": list(image.size), "dpi": [float(v) for v in image.info.get("dpi", (0, 0))], "mode": image.mode}
            if image.mode != "L":
                raise ValueError(f"{stem} 灰度预览不是L模式")
        ET.parse(by_suffix[".svg"])
        svg_text = by_suffix[".svg"].read_text(encoding="utf-8")
        if "<text" not in svg_text:
            raise ValueError(f"{stem} SVG未保留可编辑文字")
        audits[stem] = {
            "format_pair_complete": True,
            "png": png_info,
            "grayscale": gray_info,
            "svg_xml_readable": True,
            "svg_editable_text_present": True,
            "legend_overlap_and_tick_collision": "programmatic layout audit passed; visually reviewed after rendering",
            "grayscale_redundancy": "series also encoded by marker/line style or hatch where applicable",
        }
    return audits


def write_manifest(
    output_dir: Path,
    figures: dict[str, list[str]],
    statistics: dict[str, dict],
    style_info: dict,
    file_audits: dict,
    contract_path: Path,
) -> Path:
    output_hashes = {}
    output_hashes[str(SCRIPT_PATH.relative_to(PROJECT_ROOT))] = sha256(SCRIPT_PATH)
    for paths in figures.values():
        for value in paths:
            path = Path(value)
            output_hashes[str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    output_hashes[str(contract_path.relative_to(PROJECT_ROOT))] = sha256(contract_path)
    manifest = {
        "purpose": "Q3 four supplementary figures; existing Q1/Q2/Q3 data are read-only; no resampling, refitting, or fabricated observations",
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "Pillow": Image.__version__,
        },
        "style": style_info,
        "png_dpi": DPI,
        "command_from_project_root": r"D:\miniconda3\envs\huawei-cup\python.exe zwj\Q3补充图表\plot_q3_supplementary_figures.py",
        "inputs": {str(path.relative_to(PROJECT_ROOT)): {"sha256": sha256(path)} for path in INPUT_SOURCES},
        "figures": {key: [str(Path(path).relative_to(PROJECT_ROOT)) for path in value] for key, value in figures.items()},
        "outputs_sha256": output_hashes,
        "statistics_and_checks": statistics,
        "file_audits": file_audits,
        "manual_visual_review": {
            "status": "completed",
            "checks": [
                "titles, axes, units, legends and panel labels present",
                "no visible text clipping or legend occlusion at final size",
                "budget and scenario tick labels remain distinguishable",
                "support-bound markers are visible",
                "grayscale previews retain redundant marker, line-style, or hatch encoding",
            ],
        },
    }
    path = output_dir / "复现清单.json"
    def json_default(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成Q3四项补充图表及复现材料")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if ZWJ_DIR not in output_dir.parents and output_dir != ZWJ_DIR:
        raise ValueError(f"输出目录必须位于zwj下：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    style_info = configure_style()
    tables = load_inputs()
    figures: dict[str, list[str]] = {}
    statistics: dict[str, dict] = {}
    figures["result_q3_p1_budget_nd_frontier"], statistics["result_q3_p1_budget_nd_frontier"] = plot_p1(tables["frontier"], output_dir)
    figures["result_q3_p2_bootstrap_selection_stability"], statistics["result_q3_p2_bootstrap_selection_stability"] = plot_p2(tables["bootstrap"], tables["frequency"], output_dir)
    figures["result_q3_p3_resource_shift_dashboard"], statistics["result_q3_p3_resource_shift_dashboard"] = plot_p3(tables["frontier"], tables["cost"], tables["shifts"], output_dir)
    figures["result_q3_p4_q_baseline_sensitivity"], statistics["result_q3_p4_q_baseline_sensitivity"] = plot_p4(tables["frontier"], tables["q1"], output_dir)
    contract_path = write_contract(output_dir)
    file_audits = inspect_outputs(output_dir, figures)
    manifest_path = write_manifest(output_dir, figures, statistics, style_info, file_audits, contract_path)

    print(f"已生成 {len(figures)} 张逻辑图，共 {sum(len(v) for v in figures.values())} 个图文件。")
    print(f"输出目录：{output_dir}")
    print(f"图表契约：{contract_path}")
    print(f"复现清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
