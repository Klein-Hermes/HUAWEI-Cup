#!/usr/bin/env python3
"""Generate the three required Q2 supplementary figures.

Run from the project root with:
    & "D:\\Anaconda\\python.exe" src/q2_supplementary_figures.py --output-dir Q2/04_图表/q2_supplementary

All inputs are read-only. All generated files are written below
``Q2/04_图表/q2_supplementary``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in SCRIPT_PATH.parents
        if (parent / "Q1" / "03_结果").is_dir()
        and (parent / "Q2" / "03_结果").is_dir()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError(f"无法从脚本路径定位项目根目录：{SCRIPT_PATH}")
SUPPLEMENTARY_ROOT = PROJECT_ROOT / "Q2" / "04_图表" / "q2_supplementary"
DEFAULT_OUTPUT_DIR = SUPPLEMENTARY_ROOT
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
VERMILLION = "#D55E00"
GRAY = "#8C8C8C"
WARNING_RED = "#B33A3A"
DARK = "#222222"
LIGHT_BLUE = "#DCEAF7"
LIGHT_ORANGE = "#FCE8C3"
LIGHT_RED = "#F7D9D3"
LIGHT_GRAY = "#ECEFF1"

FIG1_SOURCE = PROJECT_ROOT / "Q2/03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_by_checkpoint.csv"
FIG2_SOURCE = PROJECT_ROOT / "Q1/03_结果/Q1.3/v1/holdout_metrics.csv"
FIG2_REPORT = PROJECT_ROOT / "Q1/03_结果/Q1.3/v1/q1_3_run_report.md"
P_AUDIT_JSON = PROJECT_ROOT / "Q2/03_结果/p配比可行性审计/v1/q2_p_audit_summary.json"
P_MAPPING = PROJECT_ROOT / "Q2/03_结果/p配比可行性审计/v1/q2_p_trajectory_mapping.csv"
VALIDATION_JSON = PROJECT_ROOT / "Q2/03_结果/经典ScalingLaw基线/v1/validation_comparability.json"
LOSS_PROTOCOL = PROJECT_ROOT / "Q2/03_结果/经典ScalingLaw基线/v1/loss_protocol_matrix.csv"
P_AUDIT_REPORT = PROJECT_ROOT / "Q2/03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md"
LOSS_AUDIT_REPORT = PROJECT_ROOT / "Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md"


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
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "lines.linewidth": 1.25,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": DPI,
        }
    )
    # Preserve the CJK font selected by setup_style().  Do not replace it with
    # a machine-specific list that may be absent on another workstation.
    info["font_chain"] = list(mpl.rcParams["font.sans-serif"])
    return info


def save_figure(fig: mpl.figure.Figure, stem: Path, size: tuple[float, float]) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[layout audit] {stem.name}")
    layout_issues = audit_layout(fig)
    verdict = print_report(layout_issues)
    if verdict == "FAIL":
        raise ValueError(f"{stem.name} 的程序版面检查失败")
    # Publication vectors: figure.svg and figure.pdf; raster preview: figure.png.
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
        # Pillow can fall back to a narrow Windows path API on some builds.
        # Encode in memory, then let pathlib perform the Unicode-safe write.
        gray_buffer = io.BytesIO()
        image.convert("L").save(gray_buffer, format="PNG", dpi=(DPI, DPI))
        gray_path.write_bytes(gray_buffer.getvalue())
    outputs.append(str(gray_path))
    plt.close(fig)
    return outputs


def padded_limits(lower: pd.Series, upper: pd.Series, fraction: float = 0.06) -> tuple[float, float]:
    lo = float(lower.min())
    hi = float(upper.max())
    pad = max((hi - lo) * fraction, 1e-6)
    return lo - pad, hi + pad


def plot_m0_elasticity(
    output_dir: Path,
    *,
    show_title: bool = True,
    show_footer: bool = True,
) -> tuple[list[str], dict]:
    required = [
        "B12_model_size",
        "N_params_B",
        "D_tokens_B",
        "elasticity_N",
        "elasticity_N_lower_95",
        "elasticity_N_upper_95",
        "elasticity_D",
        "elasticity_D_lower_95",
        "elasticity_D_upper_95",
    ]
    data = pd.read_csv(FIG1_SOURCE)
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"图1缺少字段：{missing_columns}")
    if len(data) != 1176 or data[required].isna().any().any():
        raise ValueError("图1输入应为1176行且必需字段无缺失")
    group_sizes = data.groupby("B12_model_size").size()
    if len(group_sizes) != 8 or not (group_sizes == 147).all():
        raise ValueError(f"图1应为8个模型规模、每组147点，实际为：{group_sizes.to_dict()}")
    if (data["D_tokens_B"] <= 0).any():
        raise ValueError("D_tokens_B 含非正值，不能使用对数轴")

    size_table = (
        data[["B12_model_size", "N_params_B"]]
        .drop_duplicates()
        .sort_values("N_params_B")
        .reset_index(drop=True)
    )
    ordered_labels = size_table["B12_model_size"].tolist()
    small_labels, large_labels = ordered_labels[:4], ordered_labels[4:]

    n_limits = padded_limits(data["elasticity_N_lower_95"], data["elasticity_N_upper_95"])
    d_limits = padded_limits(data["elasticity_D_lower_95"], data["elasticity_D_upper_95"])

    # The frozen bootstrap intervals occupy far less than one raster pixel on
    # the trajectory scale.  Keep the main axes truthful, and use a separate
    # inset coordinate system for the signed deviation from the estimate.
    # Multiplication by 1e5 is an explicit unit conversion, not data inflation.
    deviation_scale = 1e5
    deviation_limits: dict[str, tuple[float, float]] = {}
    for metric in ("N", "D"):
        estimate_col = f"elasticity_{metric}"
        lower_col = f"elasticity_{metric}_lower_95"
        upper_col = f"elasticity_{metric}_upper_95"
        deviations = pd.concat(
            [
                (data[lower_col] - data[estimate_col]) * deviation_scale,
                (data[upper_col] - data[estimate_col]) * deviation_scale,
            ],
            ignore_index=True,
        )
        bound = float(np.abs(deviations).max()) * 1.10
        deviation_limits[estimate_col] = (-bound, bound)

    size = (7.2, 9.0 if show_footer else 8.35)
    fig = plt.figure(figsize=size)
    outer_grid = fig.add_gridspec(
        4,
        4,
        left=0.10,
        right=0.985,
        top=0.905 if show_title else 0.945,
        bottom=0.125 if show_footer else 0.075,
        wspace=0.16,
        hspace=0.42,
    )
    axes = np.empty((4, 4), dtype=object)
    ci_axes = np.empty((4, 4), dtype=object)
    for grid_row in range(4):
        for grid_col in range(4):
            cell_grid = outer_grid[grid_row, grid_col].subgridspec(
                2,
                1,
                height_ratios=[2.45, 1.0],
                hspace=0.08,
            )
            axes[grid_row, grid_col] = fig.add_subplot(cell_grid[0, 0])
            ci_axes[grid_row, grid_col] = fig.add_subplot(cell_grid[1, 0])
    metric_rows = [
        (small_labels, "elasticity_N", "elasticity_N_lower_95", "elasticity_N_upper_95", BLUE, n_limits),
        (small_labels, "elasticity_D", "elasticity_D_lower_95", "elasticity_D_upper_95", ORANGE, d_limits),
        (large_labels, "elasticity_N", "elasticity_N_lower_95", "elasticity_N_upper_95", BLUE, n_limits),
        (large_labels, "elasticity_D", "elasticity_D_lower_95", "elasticity_D_upper_95", ORANGE, d_limits),
    ]
    for row, (labels, estimate, lower, upper, color, limits) in enumerate(metric_rows):
        for col, label in enumerate(labels):
            axis = axes[row, col]
            subset = data.loc[data["B12_model_size"] == label].sort_values("D_tokens_B")
            x = subset["D_tokens_B"].to_numpy(float)
            y = subset[estimate].to_numpy(float)
            lo = subset[lower].to_numpy(float)
            hi = subset[upper].to_numpy(float)
            axis.fill_between(x, lo, hi, color=color, alpha=0.30, linewidth=0, zorder=2)
            axis.plot(x, lo, color=color, linewidth=0.48, linestyle=(0, (1.4, 1.4)), alpha=0.92, zorder=3)
            axis.plot(x, hi, color=color, linewidth=0.48, linestyle=(0, (1.4, 1.4)), alpha=0.92, zorder=3)
            axis.plot(x, y, color=color, linewidth=1.15, zorder=4)
            axis.set_xscale("log")
            axis.set_xlim(float(data["D_tokens_B"].min()) * 0.92, float(data["D_tokens_B"].max()) * 1.08)
            axis.set_xticks([])
            axis.set_ylim(*limits)
            axis.axhline(0, color="#B7BDC3", linewidth=0.6, zorder=0)
            axis.grid(axis="y", color="#E6E8EB", linewidth=0.45)
            if row in (0, 2):
                n_value = float(subset["N_params_B"].iloc[0])
                axis.set_title(f"{label}（N={n_value:g}B）", pad=3)
            axis.tick_params(axis="y", labelleft=col == 0)
            if col == 0:
                metric_label = r"$\varepsilon_N$" if estimate == "elasticity_N" else r"$\varepsilon_D$"
                axis.set_ylabel(f"M0 蕴含弹性 {metric_label}")

            # Magnified uncertainty strip: the dark zero line is the estimate;
            # the light band and dashed edges are the exact lower/upper bounds
            # after subtracting the estimate and expressing the result in 1e-5.
            ci_axis = ci_axes[row, col]
            delta_lo = (lo - y) * deviation_scale
            delta_hi = (hi - y) * deviation_scale
            ci_axis.fill_between(x, delta_lo, delta_hi, color=color, alpha=0.30, linewidth=0)
            ci_axis.plot(x, delta_lo, color=color, linewidth=0.72, linestyle=(0, (2.0, 1.4)))
            ci_axis.plot(x, delta_hi, color=color, linewidth=0.72, linestyle=(0, (2.0, 1.4)))
            ci_axis.axhline(0, color=color, linewidth=0.82)
            ci_axis.set_xscale("log")
            ci_axis.set_xlim(float(data["D_tokens_B"].min()) * 0.92, float(data["D_tokens_B"].max()) * 1.08)
            ci_axis.set_ylim(*deviation_limits[estimate])
            ci_axis.set_xticks([1, 10, 100], labels=["1", "10", "100"])
            ci_axis.tick_params(axis="x", labelbottom=row in (1, 3), length=2.0, width=0.45, pad=1)
            bound = deviation_limits[estimate][1]
            ci_axis.set_yticks([-bound, 0, bound])
            if col == 0:
                ci_axis.set_yticklabels([f"{-bound:.1f}", "0", f"{bound:.1f}"], fontsize=6.0)
                ci_axis.set_ylabel("95%区间偏差\n(10^-5)", fontsize=6.0, labelpad=2)
            else:
                ci_axis.set_yticklabels([])
            ci_axis.tick_params(axis="y", length=1.8, width=0.45, pad=1)
            ci_axis.spines["top"].set_visible(True)
            ci_axis.spines["right"].set_visible(True)
            for spine in ci_axis.spines.values():
                spine.set_color("#AEB4BA")
                spine.set_linewidth(0.45)
            ci_axis.set_facecolor("#FAFAFA")
            ci_axis.text(
                0.04,
                0.90,
                "区间放大",
                transform=ci_axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.0,
                color="#40464D",
            )

    if show_title:
        fig.suptitle("冻结经典 M0 中 N、D 弹性随训练进度变化", fontsize=11, y=0.988)
    max_ci_width = max(
        float((data["elasticity_N_upper_95"] - data["elasticity_N_lower_95"]).max()),
        float((data["elasticity_D_upper_95"] - data["elasticity_D_lower_95"]).max()),
    )
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.4, label="N 弹性估计"),
        mpl.patches.Patch(facecolor=BLUE, alpha=0.30, edgecolor=BLUE, label="N 的 95% 区间（见嵌框）"),
        Line2D([0], [0], color=ORANGE, lw=1.4, label="D 弹性估计"),
        mpl.patches.Patch(facecolor=ORANGE, alpha=0.30, edgecolor=ORANGE, label="D 的 95% 区间（见嵌框）"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.956 if show_title else 0.992),
        ncol=4,
        frameon=False,
        fontsize=6.6,
    )
    if show_footer:
        fig.text(
            0.5,
            0.006,
            "区间来自 8 条完整 Pythia 轨迹的 cluster Bootstrap；独立轨迹较少，仅作稳定性提示。\n"
            f"主轴区间全宽至多 {max_ci_width:.1e}；嵌框显示相对估计值的原始上下界，纵轴单位为 10^-5。\n"
            "弹性是冻结 M0 的模型蕴含量，表示 N、D 效应，不是 p/Q 效应或因果效应。",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#40464D",
        )
    fig.supxlabel("训练量 D（十亿 Token，对数轴）", x=0.54, y=0.076 if show_footer else 0.025, fontsize=8.0)
    stem = output_dir / "result_q2_m0_elasticity_profile"
    outputs = save_figure(fig, stem, size)
    stats = {
        "source_rows": int(len(data)),
        "model_scales": int(data["B12_model_size"].nunique()),
        "checkpoints_per_scale": sorted(int(v) for v in group_sizes.unique()),
        "D_range_B_tokens": [float(data["D_tokens_B"].min()), float(data["D_tokens_B"].max())],
        "max_95_interval_full_width": max_ci_width,
        "interval_rendering": "truth-scale band on main axes plus per-panel signed-deviation inset in 1e-5 units; exact bounds, no data inflation",
        "layout": "4x4 single figure: two 2x4 N/D blocks for readability and complete coverage of 8 scales",
        "overall_title_included": show_title,
        "inline_footer_included": show_footer,
    }
    return outputs, stats


def plot_a_cross_scale(
    output_dir: Path,
    *,
    show_title: bool = True,
    show_footer: bool = True,
) -> tuple[list[str], dict]:
    data = pd.read_csv(FIG2_SOURCE)
    required = {"split", "role", "model", "target", "n", "rmse", "r2", "variant"}
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"图2缺少字段：{missing_columns}")
    split_order = ["test_1m", "test_60m", "test_1b"]
    filtered = data.loc[
        (data["model"] == "simplex_ridge")
        & (data["variant"] == "closed")
        & (data["split"].isin(split_order))
    ].copy()
    if len(filtered) != 39 or not (filtered.groupby("split")["target"].nunique() == 13).all():
        raise ValueError("图2筛选后应为3个尺度×13个目标=39行")
    expected_n = {"test_1m": 256, "test_60m": 256, "test_1b": 64}
    for split, n_value in expected_n.items():
        if set(filtered.loc[filtered["split"] == split, "n"].astype(int)) != {n_value}:
            raise ValueError(f"{split} 的 n 与预期不一致")

    macro = filtered.groupby("split", sort=False)[["rmse", "r2"]].mean().reindex(split_order)
    expected_macro = {
        "rmse": np.array([0.453863022539131, 1.5590162377435113, 3.1969897294857073]),
        "r2": np.array([0.6188879050595115, -7.933569486978265, -810.1195857238465]),
    }
    for metric in ("rmse", "r2"):
        if not np.allclose(macro[metric].to_numpy(float), expected_macro[metric], rtol=0, atol=1e-10):
            raise ValueError(f"图2的宏平均 {metric} 与冻结报告不一致")

    size = (7.2, 3.9 if show_footer else 3.45)
    fig, axes = plt.subplots(1, 2, figsize=size)
    colors = [BLUE, ORANGE, GREEN]
    x_positions = np.arange(3, dtype=float)
    x_labels = ["1M\n同尺度", "60M\n跨尺度", "1B\n跨尺度"]
    metrics = [
        ("rmse", "逐目标 RMSE", axes[0]),
        ("r2", r"逐目标 $R^2$（symlog）", axes[1]),
    ]
    for metric, ylabel, axis in metrics:
        for index, split in enumerate(split_order):
            subset = filtered.loc[filtered["split"] == split].sort_values("target")
            jitter = np.linspace(-0.17, 0.17, len(subset))
            axis.scatter(
                np.full(len(subset), x_positions[index]) + jitter,
                subset[metric],
                s=20,
                facecolor=colors[index],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.76,
                zorder=2,
            )
            mean_value = float(macro.loc[split, metric])
            axis.scatter(
                x_positions[index],
                mean_value,
                marker="D",
                s=56,
                facecolor=colors[index],
                edgecolor=DARK,
                linewidth=0.85,
                zorder=4,
            )
            text_value = f"{mean_value:.3f}" if metric == "rmse" else (f"{mean_value:.3f}" if abs(mean_value) < 10 else f"{mean_value:.1f}")
            axis.annotate(
                text_value,
                (x_positions[index], mean_value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                fontweight="bold",
                color=DARK,
            )
        axis.set_xticks(x_positions, x_labels)
        axis.set_xlim(-0.45, 2.45)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("已有评估切分")
        axis.grid(axis="y", color="#E6E8EB", linewidth=0.5)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("a  RMSE：跨尺度误差上升", loc="left", fontweight="bold")
    axes[1].axhline(0, color=DARK, linewidth=0.8, linestyle="--", zorder=1)
    axes[1].set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    axes[1].set_ylim(-6000, 1.2)
    # Explicit ASCII-minus labels avoid machine-dependent Unicode-minus glyph
    # failures in Matplotlib's symlog formatter while preserving the scale.
    axes[1].set_yticks([-1000, -100, -10, -1, 0, 1])
    axes[1].set_yticklabels(["-1000", "-100", "-10", "-1", "0", "1"])
    axes[1].set_title(r"b  $R^2$：跨尺度显著恶化", loc="left", fontweight="bold")
    axes[1].text(0.98, 0.97, "symlog；线性阈值 ±1", transform=axes[1].transAxes, ha="right", va="top", fontsize=6.4, color=GRAY)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markeredgecolor="white", markersize=5, label="领域目标（13个）"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=GRAY, markeredgecolor=DARK, markersize=6, label="13目标等权宏平均"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92 if show_title else 0.985),
        ncol=2,
        frameon=False,
    )
    if show_title:
        fig.suptitle("附件 A 的 p-only 配比模型：同尺度与跨尺度评估", fontsize=11, y=0.992)
    if show_footer:
        fig.text(
            0.5,
            0.016,
            "点为领域目标，不是独立训练重复；菱形为13个目标指标的算术平均，无置信区间或误差棒。\n"
            "这是已有评估切分而非新增盲测；跨尺度表现仅说明迁移限制，不证明 B 侧 p 效应。",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#40464D",
        )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.80 if show_title else 0.865,
        bottom=0.23 if show_footer else 0.18,
        wspace=0.30,
    )
    stem = output_dir / "result_q2_A_p_only_cross_scale"
    outputs = save_figure(fig, stem, size)
    stats = {
        "source_rows": int(len(data)),
        "filtered_rows": int(len(filtered)),
        "filter": "model=simplex_ridge; variant=closed; split in test_1m,test_60m,test_1b",
        "targets_per_split": filtered.groupby("split")["target"].nunique().astype(int).to_dict(),
        "n_per_target": expected_n,
        "macro_rmse": {key: float(macro.loc[key, "rmse"]) for key in split_order},
        "macro_r2": {key: float(macro.loc[key, "r2"]) for key in split_order},
        "overall_title_included": show_title,
        "inline_footer_included": show_footer,
    }
    return outputs, stats


def plot_p_gate_audit(
    output_dir: Path,
    *,
    show_title: bool = True,
    show_source_note: bool = True,
) -> tuple[list[str], dict]:
    summary = json.loads(P_AUDIT_JSON.read_text(encoding="utf-8"))
    mapping = pd.read_csv(P_MAPPING)
    validation = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
    protocol = pd.read_csv(LOSS_PROTOCOL)
    if int(summary["verified_p_vectors_joined_to_loss"]) != 0 or int(summary["loss_rows_with_p_joined"]) != 0:
        raise ValueError("p 审计状态已变化：不再是零个已核验且联接 Loss 的配比向量")

    expected = {
        "B1": (1176, 8),
        "B2": (1029, 7),
        "B3": (4000, 8),
        "B4": (57, 57),
        "B5": (44, 44),
    }
    observed = {}
    for dataset, (record_total, unit_rows) in expected.items():
        subset = mapping.loc[mapping["dataset"] == dataset]
        observed[dataset] = (int(subset["record_count"].sum()), int(len(subset)))
        if observed[dataset] != (record_total, unit_rows):
            raise ValueError(f"{dataset} 映射计数变化：预期 {(record_total, unit_rows)}，实际 {observed[dataset]}")
        if not subset["p_loss_join_status"].astype(str).str.startswith("NO").all():
            raise ValueError(f"{dataset} 出现新的 p-Loss 联接状态，请重新审计")
    for dataset in ("B4", "B5"):
        values = set(protocol.loc[protocol["dataset"] == dataset, "classification_vs_B1"].astype(str))
        if values != {"NOT_COMPARABLE"}:
            raise ValueError(f"{dataset} 的 Loss 可比性状态变化：{values}")
    if not {"B2", "B3", "B4", "B5"}.issubset(validation["source_comparability"]):
        raise ValueError("validation_comparability.json 缺少 B2-B5 状态")

    row_labels = ["B1", "B2", "B3", "B4", "B5"]
    columns = [
        "记录数 / 来源性质",
        "已核验\n17维 p",
        "p-Loss\n连接",
        "验证角色",
        "相对 B1 的\nLoss 可比性",
    ]
    cells = [
        ["1,176个 Pythia 检查点\n8条轨迹", "无", "0", "主参考来源", "参考组\n不是外部比较"],
        ["1,029条半合成记录\n7条轨迹", "无", "0", "半合成压力测试", "不作为真实外部验证"],
        ["4,000个 Pythia 插值点\n8条轨迹", "无", "0", "同来源插值检查", "不作为独立真实验证"],
        ["57个跨族记录\n12个模型族", "无", "0", "行级来源 / 运行键缺失", "NOT_COMPARABLE"],
        ["44个文献记录\n9个模型族", "无", "0", "多来源汇编\n缺行级运行键", "NOT_COMPARABLE"],
    ]
    row_colors = [LIGHT_BLUE, LIGHT_ORANGE, LIGHT_ORANGE, LIGHT_RED, LIGHT_RED]
    cell_colors = []
    for index in range(len(row_labels)):
        cell_colors.append([LIGHT_GRAY, LIGHT_RED, LIGHT_RED, row_colors[index], row_colors[index]])

    size = (7.2, 4.5)
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        rowLabels=row_labels,
        colLabels=columns,
        cellColours=cell_colors,
        colColours=["#DDE3E8"] * len(columns),
        rowColours=["#DDE3E8"] * len(row_labels),
        cellLoc="center",
        rowLoc="center",
        colLoc="center",
        colWidths=[0.27, 0.12, 0.11, 0.24, 0.26],
        bbox=[0.035, 0.23 if show_source_note else 0.20, 0.945, 0.61 if show_source_note else 0.68],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.0)
        if row == 0 or col == -1:
            cell.set_text_props(fontweight="bold", color=DARK)
        if row > 0 and col in (1, 2):
            cell.set_text_props(fontweight="bold", color=WARNING_RED)
        if row > 0 and col == 4 and row in (4, 5):
            cell.set_text_props(fontweight="bold", color=WARNING_RED)
    if show_title:
        fig.suptitle("B1–B5 配比数据与验证边界状态矩阵", fontsize=11, y=0.975)
    fig.text(
        0.5,
        0.89 if show_title else 0.955,
        "当前没有已核验并逐行连接到 Loss 的 17 维 p 向量，正式 B 侧 p-only 拟合尚不能启动",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=WARNING_RED,
    )
    fig.text(
        0.5,
        0.135 if show_source_note else 0.095,
        "“0”表示当前审计材料中没有已核验且连接到 Loss 的17维配比，不表示真实训练中的配比效应为零。\n"
        "B2半合成、B3插值、B4/B5 Loss口径不可比是不同限制，不能合并为同一状态；颜色仅作辅助。",
        ha="center",
        va="center",
        fontsize=6.8,
        color="#40464D",
    )
    if show_source_note:
        fig.text(
            0.5,
            0.055,
            "数据来源：p配比可行性审计、轨迹映射、验证可比性与 Loss 协议矩阵（均为已有审计结果）",
            ha="center",
            va="center",
            fontsize=6.2,
            color=GRAY,
        )
    stem = output_dir / "result_q2_p_gate_audit"
    outputs = save_figure(fig, stem, size)
    stats = {
        "mapping_rows": int(len(mapping)),
        "record_total_and_units": {key: {"record_total": value[0], "unit_rows": value[1]} for key, value in observed.items()},
        "verified_p_vectors_joined_to_loss": int(summary["verified_p_vectors_joined_to_loss"]),
        "loss_rows_with_p_joined": int(summary["loss_rows_with_p_joined"]),
        "B4_B5_classification_vs_B1": "NOT_COMPARABLE",
        "B2_label_correction": "1,029 semi-synthetic records; 7 tracks (the task sheet incorrectly called all 1,029 rows tracks)",
        "overall_title_included": show_title,
        "semantic_footer_included": True,
        "source_footer_included": show_source_note,
    }
    return outputs, stats


def write_manifest(
    output_dir: Path,
    figure_outputs: dict[str, list[str]],
    statistics: dict[str, dict],
    style_info: dict,
    *,
    titleless: bool = False,
) -> Path:
    sources = [
        FIG1_SOURCE,
        FIG2_SOURCE,
        FIG2_REPORT,
        P_AUDIT_JSON,
        P_MAPPING,
        VALIDATION_JSON,
        LOSS_PROTOCOL,
        P_AUDIT_REPORT,
        LOSS_AUDIT_REPORT,
    ]
    manifest = {
        "purpose": "Q2 supplementary figures only; no data modification and no model fitting",
        "variant": "paper-titleless" if titleless else "standard-titled",
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "Pillow": Image.__version__,
        },
        "style": style_info,
        "dpi": DPI,
        "paper_annotation_policy": (
            "remove narrative footers from trajectory/scatter figures; retain only the anti-misinterpretation footer in the status matrix"
            if titleless
            else "full standalone annotations"
        ),
        "command_from_project_root": (
            f'& "{sys.executable}" src/q2_supplementary_figures.py '
            f'--titleless --output-dir "{output_dir.relative_to(PROJECT_ROOT).as_posix()}"'
            if titleless
            else f'& "{sys.executable}" src/q2_supplementary_figures.py '
            f'--output-dir "{output_dir.relative_to(PROJECT_ROOT).as_posix()}"'
        ),
        "inputs": {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sources},
        "figures": figure_outputs,
        "statistics": statistics,
        "optional_figure_4": {
            "generated": False,
            "reason": "Within every model size, fitted loss is a monotone transform of D (Spearman=-1); residual-vs-D is therefore a horizontal-axis re-expression of the existing residual-vs-fitted diagnostic and adds no independent statistical or validation evidence.",
        },
    }
    path = output_dir / "复现清单.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 Q2 三张补充图及复现清单")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--titleless", action="store_true", help="生成去除整图标题的论文版")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if SUPPLEMENTARY_ROOT not in output_dir.parents and output_dir != SUPPLEMENTARY_ROOT:
        raise ValueError(f"输出目录必须位于 {SUPPLEMENTARY_ROOT} 下：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    style_info = configure_style()
    figures: dict[str, list[str]] = {}
    statistics: dict[str, dict] = {}
    show_title = not args.titleless
    show_narrative_footer = not args.titleless
    figures["result_q2_m0_elasticity_profile"], statistics["result_q2_m0_elasticity_profile"] = plot_m0_elasticity(
        output_dir, show_title=show_title, show_footer=show_narrative_footer
    )
    figures["result_q2_A_p_only_cross_scale"], statistics["result_q2_A_p_only_cross_scale"] = plot_a_cross_scale(
        output_dir, show_title=show_title, show_footer=show_narrative_footer
    )
    figures["result_q2_p_gate_audit"], statistics["result_q2_p_gate_audit"] = plot_p_gate_audit(
        output_dir, show_title=show_title, show_source_note=not args.titleless
    )
    manifest_path = write_manifest(output_dir, figures, statistics, style_info, titleless=args.titleless)

    print(f"已生成 {len(figures)} 张逻辑图，共 {sum(len(v) for v in figures.values())} 个图文件。")
    print(f"输出目录：{output_dir}")
    print(f"复现清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
