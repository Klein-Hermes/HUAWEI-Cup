"""Reproducible Q2 full-answer evidence figures.

Figures separate A-side mixture evidence, B1 classical scaling evidence, and
semi-synthetic Q response. They do not imply that A/B rows form a joint sample.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
FIGURE_SKILL_ROOT = Path(
    os.environ.get(
        "CODEX_MATH_MODELING_SKILL_ROOT",
        str(Path.home() / ".codex" / "skills" / "math-modeling"),
    )
)


def _tools(skill_root: Path):
    scripts = skill_root / "tools" / "figure" / "scripts"
    if not scripts.is_dir():
        raise FileNotFoundError(
            f"Figure skill scripts not found: {scripts}; set "
            "CODEX_MATH_MODELING_SKILL_ROOT to the installed math-modeling skill root."
        )
    sys.path.insert(0, str(scripts))
    from setup_style import setup_style
    from export_figure import export_figure
    from visual_qa import audit_layout, render_preview

    return setup_style, export_figure, audit_layout, render_preview


def _share_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if name.startswith("train_the_pile_")]


def _matched_slopes(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    group_columns = ["N_params_B", "D_tokens_B"]
    if "data_type" in frame.columns:
        groups = frame.groupby(["data_type", *group_columns], sort=True)
        for (data_type, n, d), cell in groups:
            if cell["Q_score"].nunique() > 1:
                rows.append({
                    "group": f"{label} {data_type}",
                    "dataset": label,
                    "data_type": str(data_type),
                    "N_params_B": float(n),
                    "D_tokens_B": float(d),
                    "slope": float(np.polyfit(cell["Q_score"], cell["val_loss"], 1)[0]),
                })
    else:
        for (n, d), cell in frame.groupby(group_columns, sort=True):
            if cell["Q_score"].nunique() > 1:
                rows.append({
                    "group": label,
                    "dataset": label,
                    "data_type": "semi_synthetic",
                    "N_params_B": float(n),
                    "D_tokens_B": float(d),
                    "slope": float(np.polyfit(cell["Q_score"], cell["val_loss"], 1)[0]),
                })
    return pd.DataFrame(rows)


def _flowchart(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, label, face, rounded=False, fontsize=7.0):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.015" if rounded else "square,pad=0.008",
            linewidth=0.9, edgecolor="#4B5563", facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize,
                color="#202124", linespacing=1.15)
        return (x, y, w, h)

    def diamond(cx, cy, w, h, label, face, fontsize=6.7):
        verts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
        patch = Polygon(verts, closed=True, linewidth=0.9, edgecolor="#4B5563", facecolor=face)
        ax.add_patch(patch)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, color="#202124",
                linespacing=1.1)
        return (cx, cy, w, h)

    def arrow(start, end, connectionstyle="arc3,rad=0"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                     linewidth=0.85, color="#4B5563",
                                     connectionstyle=connectionstyle))

    # Three source-specific evidence branches.
    rows = [
        (0.73, "A4/A5\n17域 p + Loss", "A侧 p-only\n1M拟合与替代敏感性", "仅来源内证据\n跨尺度预测未通过"),
        (0.45, "B1–B5\nN、D、Loss", "B1拟合 M0\nB2–B5分源核验", "N/D效应完成\nB4/B5不可比"),
        (0.17, "Q1-q_huber\nB6–B8", "半合成 Q 响应\n固定 N,D 方向审计", "B6/B7向下\nB8方向相反"),
    ]
    source_fill = "#EAF2F8"
    process_fill = "#F3F4F6"
    result_fill = "#EAF4EA"
    for y, source, process, result in rows:
        h = 0.16
        b1 = box(0.025, y, 0.19, h, source, source_fill, rounded=True, fontsize=6.6)
        b2 = box(0.265, y, 0.215, h, process, process_fill, fontsize=6.6)
        b3 = box(0.53, y, 0.19, h, result, result_fill, fontsize=6.6)
        arrow((b1[0] + b1[2], y + h / 2), (b2[0], y + h / 2))
        arrow((b2[0] + b2[2], y + h / 2), (b3[0], y + h / 2))

    # Evidence is joined only at the explicit identifiability gate.
    gate = diamond(0.82, 0.56, 0.17, 0.26, "联合识别门禁\np-Loss=0\n独立 Q 变化缺失", "#FFF4D6", fontsize=6.5)
    arrow((0.72, 0.81), (0.755, 0.65), "arc3,rad=-0.12")
    arrow((0.72, 0.53), (0.735, 0.56))
    arrow((0.72, 0.25), (0.755, 0.47), "arc3,rad=0.12")

    output = box(0.74, 0.075, 0.22, 0.18,
                 "当前答案\n保留 M0 + 分源证据\nM1/M2 不估计\n给出条件推导与限制",
                 "#F9E9E8", rounded=True, fontsize=6.7)
    arrow((gate[0], gate[1] - gate[3] / 2), (output[0] + output[2] / 2, output[1] + output[3]))
    ax.text(0.5, 0.975, "Q2 完整回答：来源分支、识别门禁与当前结论",
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="#202124")
    ax.text(0.5, 0.015,
            "A、B 与半合成 Q 结果分源报告；不得拼接成真实联合 (N,D,p,Q,Loss) 样本。",
            ha="center", va="bottom", fontsize=6.3, color="#4B5563")


def create_q2_full_answer_figures(
    project_root: Path,
    output_dir: Path,
    *,
    p_complementarity_summary: pd.DataFrame,
    q_model_summary: pd.DataFrame,
    q_direction_summary: pd.DataFrame,
    q1_holdout_metrics: pd.DataFrame,
    skill_root: Path = FIGURE_SKILL_ROOT,
) -> dict[str, object]:
    """Create nine data figures and one full-answer workflow chart."""
    setup_style, export_figure, audit_layout, render_preview = _tools(skill_root)
    setup_info = setup_style(journal="general", lang="zh", use_sciplots=False,
                             serif_for_zh=False, constrained_layout=True)
    from utils.plot_style import PALETTE

    plt.rcParams["axes.unicode_minus"] = False
    fig_dir = output_dir / "q2_full_answer_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(tempfile.gettempdir()) / "q2_full_answer_figure_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    att = project_root / "中文题目" / "F题" / "real_attachments"
    mixture_path = att / "A_data_value" / "regmix_tables" / "train_mixture_1m.csv"
    b_path = att / "B_scaling_laws"
    mixture = pd.read_csv(mixture_path)
    b6 = pd.read_csv(b_path / "supplementary_NQ_experiment.csv")
    b7 = pd.read_csv(b_path / "supplementary_NQ_experiment_expanded.csv")
    b8 = pd.read_csv(b_path / "supplementary_NQ_experiment_large.csv")
    q_sources = [
        ("B6", b6, "B6：半合成"),
        ("B7", b7, "B7：含 B6 全部行"),
        ("B8 calibrated", b8[b8["data_type"] == "calibrated"], "B8：校准"),
        ("B8 extrapolated", b8[b8["data_type"] == "extrapolated"], "B8：外推"),
    ]

    figure_specs: list[dict[str, str]] = []
    qa_records: list[dict[str, object]] = []

    def save(fig, stem: str, *, size: tuple[float, float], category: str,
             claim: str, evidence: str, warning: str, tight: bool = True) -> None:
        issues = audit_layout(fig)
        failures = [str(item) for item in issues if str(item).upper().startswith("FAIL")]
        if failures:
            plt.close(fig)
            raise RuntimeError(f"Figure layout QA failed for {stem}: {failures}")
        preview_path = preview_dir / f"{stem}_preview.png"
        render_preview(fig, str(preview_path), dpi=150)
        paths = export_figure(
            fig, str(fig_dir / stem), formats=["pdf", "svg", "png"],
            size_inches=size, dpi=300, grayscale_preview=tight, tight=tight,
        )
        if not tight:
            png_path = fig_dir / f"{stem}.png"
            grayscale_path = fig_dir / f"{stem}_grayscale.png"
            with Image.open(png_path) as image:
                image.convert("L").save(grayscale_path, dpi=(300, 300))
            paths.append(str(grayscale_path))
        plt.close(fig)
        figure_specs.append({
            "figure_id": stem,
            "category": category,
            "claim": claim,
            "evidence": evidence,
            "warning": warning,
        })
        qa_records.append({
            "figure_id": stem,
            "layout_issues": [str(item) for item in issues],
            "preview": str(preview_path),
            "exports": paths,
        })

    # RAW — 512 source-A recipe rows and the actual (not silently normalized) shares.
    share_cols = _share_columns(mixture)
    recipes = mixture[share_cols].astype(float)
    if recipes.shape != (512, 17):
        raise ValueError(f"Expected 512 x 17 mixture shares, got {recipes.shape}")
    stats = pd.DataFrame({
        "domain": [name.removeprefix("train_the_pile_") for name in share_cols],
        "q10": recipes.quantile(0.10).to_numpy(),
        "q25": recipes.quantile(0.25).to_numpy(),
        "median": recipes.median().to_numpy(),
        "q75": recipes.quantile(0.75).to_numpy(),
        "q90": recipes.quantile(0.90).to_numpy(),
        "nonzero": (recipes > 0).mean().to_numpy(),
    })
    stats = stats.sort_values(["q90", "nonzero"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ypos = np.arange(len(stats))
    ax.hlines(ypos, stats["q10"] * 100, stats["q90"] * 100,
              color="#A9C6D8", linewidth=2.1, zorder=1)
    ax.hlines(ypos, stats["q25"] * 100, stats["q75"] * 100,
              color=PALETTE["primary"], linewidth=5.0, zorder=2)
    ax.scatter(stats["median"] * 100, ypos, marker="D", s=19,
               color=PALETTE["contrast"], edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_yticks(ypos, stats["domain"])
    ax.invert_yaxis()
    ax.set_xlim(-1.5, 112)
    ax.set_xlabel("原始配方占比（%）")
    ax.set_ylabel("Pile 子域")
    ax.set_title("A 侧 512 个 1M 配方的 17 域支持范围", loc="left")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.12,
            "细线：P10–P90；粗线：P25–P75；菱形：中位数。右列为非零配方占比。配比未重归一化；行和 0.996–1.003。",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color="#4B5563")
    ax.text(1.005, 1.005, "非零", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4)
    for y, value in zip(ypos, stats["nonzero"]):
        ax.text(101.5, y, f"{value:.0%}", va="center", ha="left", fontsize=6.1, color="#374151")
    save(fig, "raw_q2_p_recipe_support", size=(7.2, 5.4), category="raw",
         claim="A 侧训练配比在 17 域上的经验覆盖和零占比特征。",
         evidence="A_data_value/regmix_tables/train_mixture_1m.csv, 512 recipes.",
         warning="这 512 条是 A 侧配方，不是 B1–B5 训练配方；原比例和未强制归一化。")

    # PROCESS — interval classifications before and after conditional max-|t| adjustment.
    cols = {
        "互补候选：CI<0": ("bootstrap_ci_fully_negative_unadjusted", "bootstrap_maxT95_fully_negative"),
        "递减收益候选：CI>0": ("bootstrap_ci_fully_positive_unadjusted", "bootstrap_maxT95_fully_positive"),
        "区间跨零": ("bootstrap_ci_cross_zero_unadjusted", "bootstrap_maxT95_cross_zero"),
    }
    labels = list(cols)
    point_counts = [int(p_complementarity_summary[a].sum()) for a, _ in cols.values()]
    max_t_counts = [int(p_complementarity_summary[b].sum()) for _, b in cols.values()]
    total = sum(point_counts)
    if total != sum(max_t_counts):
        raise ValueError("Pointwise and max-|t| candidate counts do not match")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    yvals = [1, 0]
    names = ["逐点 95% Bootstrap 区间", "条件 max-|t| 95% 区间"]
    palette = [PALETTE["primary"], PALETTE["secondary"], "#B8BEC5"]
    legend_labels = [
        f"互补候选 CI<0 ({point_counts[0]:,}→{max_t_counts[0]:,})",
        f"递减收益候选 CI>0 ({point_counts[1]:,}→{max_t_counts[1]:,})",
        f"区间跨零 ({point_counts[2]:,}→{max_t_counts[2]:,})",
    ]
    for y, counts in zip(yvals, [point_counts, max_t_counts]):
        left = 0.0
        for label, count, color in zip(legend_labels, counts, palette):
            width = 100 * count / total if total else 0
            ax.barh(y, width, left=left, height=0.48, color=color,
                    edgecolor="white", linewidth=0.55, label=label if y == yvals[0] else None)
            left += width
    ax.set_yticks(yvals, names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("候选转移对占比（%）")
    ax.set_title(f"A 侧 p×p 敏感性：区间校正前后（{total:,} 个候选转移对）", loc="left")
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=6.3)
    save(fig, "process_q2_p_interval_adjustment", size=(7.2, 3.7), category="process",
         claim="家族区间校正后，互补候选由逐点信号收缩至零个稳健负向区间。",
         evidence="A4/A5 17-domain transfer-pair recipe-group bootstrap summary.",
         warning="候选集由同一完整样本筛选；max-|t| 结果仅条件于已筛候选，不是确认性检验。")

    # RESULT — target-wise A-side p-only performance across the existing scale splits.
    held = q1_holdout_metrics[
        (q1_holdout_metrics["model"] == "simplex_ridge")
        & (q1_holdout_metrics["variant"] == "closed")
        & q1_holdout_metrics["split"].isin(["test_1m", "test_60m", "test_1b"])
    ].copy()
    split_order = ["test_1m", "test_60m", "test_1b"]
    if len(held) != 39 or held.groupby("split")["target"].nunique().to_dict() != {s: 13 for s in split_order}:
        raise ValueError("Expected 3 existing scale splits × 13 target metrics")
    expected_n = {"test_1m": 256, "test_60m": 256, "test_1b": 64}
    for split, n_value in expected_n.items():
        if set(held.loc[held["split"] == split, "n"].astype(int)) != {n_value}:
            raise ValueError(f"Unexpected sample size for {split}")
    macro = held.groupby("split", sort=False)[["rmse", "r2"]].mean().reindex(split_order)
    expected_macro = {
        "rmse": np.array([0.453863022539131, 1.5590162377435113, 3.1969897294857073]),
        "r2": np.array([0.6188879050595115, -7.933569486978265, -810.1195857238465]),
    }
    for metric in ("rmse", "r2"):
        if not np.allclose(macro[metric].to_numpy(float), expected_macro[metric], rtol=0, atol=1e-10):
            raise ValueError(f"Frozen Q1.3 macro-{metric} values changed")

    cross_scale_rc = {
        "figure.constrained_layout.use": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    }
    previous_rc = {key: plt.rcParams[key] for key in cross_scale_rc}
    plt.rcParams.update(cross_scale_rc)
    x_positions = np.arange(3, dtype=float)
    x_labels = ["1M\n同尺度", "60M\n跨尺度", "1B\n跨尺度"]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), constrained_layout=False)
    for metric, ylabel, axis in (
        ("rmse", "逐目标 RMSE", axes[0]),
        ("r2", r"逐目标 $R^2$（symlog）", axes[1]),
    ):
        for index, split in enumerate(split_order):
            subset = held.loc[held["split"] == split].sort_values("target")
            jitter = np.linspace(-0.17, 0.17, len(subset))
            axis.scatter(
                np.full(len(subset), x_positions[index]) + jitter,
                subset[metric], s=20, facecolor=colors[index], edgecolor="white",
                linewidth=0.45, alpha=0.76, zorder=2,
            )
            mean_value = float(macro.loc[split, metric])
            axis.scatter(x_positions[index], mean_value, marker="D", s=56,
                         facecolor=colors[index], edgecolor="#222222", linewidth=0.85,
                         zorder=4)
            text_value = (f"{mean_value:.3f}" if metric == "rmse" or abs(mean_value) < 10
                          else f"{mean_value:.1f}")
            axis.annotate(text_value, (x_positions[index], mean_value), xytext=(0, 7),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=6.2, fontweight="bold", color="#222222")
        axis.set_xticks(x_positions, x_labels)
        axis.set_xlim(-0.45, 2.45)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("已有评估切分")
        axis.grid(axis="y", color="#E6E8EB", linewidth=0.5)
        axis.set_axisbelow(True)
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("a  RMSE：跨尺度误差上升", loc="left", fontweight="bold")
    axes[1].axhline(0, color="#222222", linewidth=0.8, linestyle="--", zorder=1)
    axes[1].set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    axes[1].set_ylim(-6000, 1.2)
    axes[1].set_title(r"b  $R^2$：跨尺度大幅下降", loc="left", fontweight="bold")
    axes[1].text(0.98, 0.97, "symlog；线性阈值 ±1", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=6.4, color="#6B7280")
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280",
               markeredgecolor="white", markersize=5, label="领域目标（13个）"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#6B7280",
               markeredgecolor="#222222", markersize=6, label="13目标等权宏平均"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.92),
               ncol=2, frameon=False)
    fig.suptitle("附件 A 的 p-only 配比模型：同尺度与跨尺度评估", fontsize=11, y=0.992)
    fig.text(
        0.5, 0.016,
        "点为领域目标，不是独立训练重复；菱形为13个目标指标的算术平均，无置信区间或误差棒。\n"
        "这是已有评估切分而非新增盲测；跨尺度表现仅说明迁移限制，不证明 B 侧 p 效应。",
        ha="center", va="bottom", fontsize=6.7, color="#40464D",
    )
    fig.subplots_adjust(left=0.095, right=0.985, top=0.80, bottom=0.23, wspace=0.30)
    save(fig, "result_q2_p_cross_scale_rmse", size=(7.2, 3.9), category="result",
         claim="附件 A 的 p-only 模型跨尺度评估显示 RMSE 增大且 R² 大幅下降；不能外推为 B 侧 p 效应。",
         evidence="Q1.3 holdout_metrics.csv; 13 targets at each existing evaluation scale; target-wise RMSE and R².",
         warning="已有评估切分而非新增盲测；13 个领域目标不是独立训练重复；不能据此推断 B 侧 p 效应。",
         tight=False)
    plt.rcParams.update(previous_rc)

    # RAW — unadjusted Q-score/Loss support, explicitly faceted by data source/type.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True, sharey=True)
    for ax, (name, frame, label) in zip(axes.ravel(), q_sources):
        ax.scatter(frame["Q_score"], frame["val_loss"], s=7, alpha=0.22,
                   color=PALETTE["primary"], edgecolors="none")
        ax.set_title(f"{label}（n={len(frame):,}）", loc="left", fontsize=7)
        ax.grid(color="#D9DEE3", linewidth=0.45, alpha=0.6)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel("Q_score")
    axes[-1, 1].set_xlabel("Q_score")
    axes[0, 0].set_ylabel("val_loss")
    axes[1, 0].set_ylabel("val_loss")
    fig.suptitle("Q 补充数据的原始 Q–Loss 支持（描述性）", x=0.04, ha="left", fontsize=9.2)
    save(fig, "raw_q2_q_source_patterns", size=(7.2, 5.2), category="raw",
         claim="B6/B7/B8 的原始 Q–Loss 结构异质，且必须分来源和数据类型展示。",
         evidence="B6/B7/B8 source CSV; N,D vary in all panels.",
         warning="未经 N,D 控制的点云仅为描述；B6/B7 有完全重叠样本。")

    # PROCESS — leave-one-Q-level-out comparison; four deterministic estimates, no CI.
    model_table = q_model_summary[
        q_model_summary["model"].isin(["M0_ND", "M_Q_NDQ"])
    ].copy()
    model_colors = {"M0_ND": PALETTE["neutral"], "M_Q_NDQ": PALETTE["positive"]}
    model_markers = {"M0_ND": "o", "M_Q_NDQ": "s"}
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    positions = {"B6": 1, "B7": 2}
    for model in ["M0_ND", "M_Q_NDQ"]:
        subset = model_table[model_table["model"] == model].set_index("dataset").loc[["B6", "B7"]]
        ys = subset["loo_q_rmse"].to_numpy(float)
        xs = [positions["B6"], positions["B7"]]
        ax.plot(xs, ys, color=model_colors[model], linestyle="--" if model == "M0_ND" else "-",
                linewidth=1.0, alpha=0.8, label=model)
        ax.scatter(xs, ys, marker=model_markers[model], s=42, color=model_colors[model],
                   edgecolor="white", linewidth=0.6, zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.4, color="#374151")
    ax.set_xticks([1, 2], ["B6（8 个 Q 水平）", "B7（10 个 Q 水平）"])
    ax.set_xlim(0.65, 2.35)
    ax.set_ylim(0.06, max(model_table["loo_q_rmse"].max() * 1.32, 0.18))
    ax.set_ylabel("留一 Q 水平 RMSE")
    ax.set_title("半合成网格中的条件模型比较", loc="left")
    ax.legend(frameon=False, title="模型", ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=6.5)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.02,
            "每点为固定数据表上的留一 Q 水平结果，无抽样区间；B7 重用 B6 全部 360 行，因此不是独立复制。",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color="#4B5563")
    save(fig, "process_q2_q_leave_q_comparison", size=(6.3, 3.8), category="process",
         claim="加入 Q 项降低给定半合成网格的留一 Q 水平误差。",
         evidence="Saved B6/B7 Q model supplement; M0 and M_Q_NDQ leave-Q-level RMSE.",
         warning="仅检验半合成生成表和预设形式；B7 与 B6 完全复用 360 行，不是独立验证。")

    # RESULT — within-cell (fixed N,D) Q slopes, retaining the B8 conflict.
    slope_tables = [
        _matched_slopes(b6, "B6"),
        _matched_slopes(b7, "B7"),
        _matched_slopes(b8, "B8"),
    ]
    slopes = pd.concat(slope_tables, ignore_index=True)
    groups = ["B6", "B7", "B8 calibrated", "B8 extrapolated"]
    colors = [PALETTE["primary"], PALETTE["sky"], PALETTE["secondary"], PALETTE["contrast"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    rng = np.random.default_rng(20260925)
    tick_labels = []
    sign_notes = []
    for idx, (group, color) in enumerate(zip(groups, colors), start=1):
        vals = slopes.loc[slopes["group"] == group, "slope"].to_numpy(float)
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(idx + jitter, vals, s=17, alpha=0.68, color=color,
                   edgecolor="white", linewidth=0.3, zorder=3)
        median = float(np.median(vals))
        ax.plot([idx - 0.18, idx + 0.18], [median, median], color="#202124", linewidth=1.4, zorder=4)
        neg = int(np.sum(vals < 0)); pos = int(np.sum(vals > 0)); zero = int(np.sum(vals == 0))
        tick_labels.append(f"{group}\n(n={len(vals)})")
        sign_notes.append(f"{neg}/{zero}/{pos}")
    ax.axhline(0, color="#4B5563", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(1, len(groups) + 1), tick_labels)
    ax.set_xlim(0.55, 4.45)
    ax.set_ylabel("固定 N,D 单元内 Loss 对 Q_score 的线性斜率")
    ax.set_title("固定 N,D 后，B8 与 B6/B7 的 Q 方向相反", loc="left")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.17,
            "各点为一个匹配 N,D 单元内的斜率；0 线为方向界。负/零/正计数：" + "；".join(
                f"{g} {note}" for g, note in zip(groups, sign_notes)
            ) + ".",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color="#4B5563", wrap=True)
    save(fig, "result_q2_q_matched_slope_conflict", size=(7.0, 4.2), category="result",
         claim="控制固定 N,D 后，B6/B7 的半合成条件斜率全负，B8 校准/外推组全正。",
         evidence="Matched-cell linear slopes computed from the source B6/B7/B8 CSV files.",
         warning="匹配单元不是独立实验；B6/B7 复用行，B8 仍是补充/整理数据。不能合并为一个 Q 弹性。")

    # Overall workflow — stages correspond to the implemented branches and gates.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    _flowchart(ax)
    save(fig, "flow_q2_full_answer", size=(7.2, 5.2), category="flow",
         claim="展示 Q2 真实数据、来源内分析、半合成响应和 M1/M2 识别门禁的完整依赖。",
         evidence="Current Q2 source manifest, p feasibility gate, frozen M0 and q2 supplement code.",
         warning="流程图将 M1/M2 标为当前未估计；不是未来通过门禁后的拟合方案承诺。")

    # Include the separately generated M0 profile and p identifiability gate.
    supplementary_dir = project_root / "Q2" / "归档_20260925" / "Q2补充图表"
    supplementary_figures = [
        {
            "figure_id": "result_q2_p_gate_audit",
            "category": "result",
            "claim": "B1–B5 当前没有经核验并逐行连接到 Loss 的 17 维 p 向量；B 侧 p-only Gate 仍为 FAIL。",
            "evidence": "Q2 p audit summary, trajectory mapping, validation comparability, loss protocol matrix, and current audit reports.",
            "warning": "0 表示当前已核验并连接到 Loss 的配比数，不代表真实 p 效应为零；B2 半合成、B3 插值、B4/B5 Loss 不可比是不同限制。",
        },
        {
            "figure_id": "result_q2_m0_elasticity_profile",
            "category": "result",
            "claim": "冻结 M0 中 N、D 的模型蕴含弹性随训练量和模型规模变化，补充既有边际效应图。",
            "evidence": "q2_m0_marginal_effects_by_checkpoint.csv; 1,176 checkpoints; 8 scales × 147 checkpoints.",
            "warning": "仅表示冻结 M0 的 N/D 模型蕴含效应，不是 p/Q 效应或因果效应；8 条轨迹的 cluster Bootstrap 区间仅作稳定性提示。",
        },
        {
            "figure_id": "result_q2_A_p_only_cross_scale",
            "category": "result",
            "claim": "附件 A 的 p-only 模型在同尺度与跨尺度评估中表现差异明显，不能迁移为 B 侧 p 效应。",
            "evidence": "Q1.3 holdout_metrics.csv and q1_3_run_report.md; 13 targets across 1M, 60M and 1B evaluation splits.",
            "warning": "这是已有切分而非新增盲测，13 个目标不是独立训练重复；图面与 result_q2_p_cross_scale_rmse 内容重复，不增加新的评估证据。",
        },
    ]
    source_manifest_path = supplementary_dir / "复现清单.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Supplementary figure manifest not found: {source_manifest_path}; "
            "run Q2/归档_20260925/plot_q2_supplementary_figures.py first."
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_hashes = source_manifest.get("inputs", {})
    source_figure_hashes = source_manifest.get("figure_sha256", {})
    for item in supplementary_figures:
        stem = item["figure_id"]
        if stem not in source_figure_hashes:
            raise ValueError(f"Supplementary figure hashes missing for {stem}; regenerate the manifest.")
        for suffix in ("pdf", "svg", "png", "grayscale.png"):
            filename = f"{stem}_{suffix}" if suffix == "grayscale.png" else f"{stem}.{suffix}"
            source = supplementary_dir / filename
            if not source.is_file():
                raise FileNotFoundError(
                    f"Supplementary figure missing: {source}; regenerate the supplementary figures first."
                )
            actual_hash = hashlib.sha256(source.read_bytes()).hexdigest().upper()
            expected_hash = source_figure_hashes[stem].get(filename, "").upper()
            if actual_hash != expected_hash:
                raise ValueError(f"Supplementary figure hash mismatch for {source}; regenerate the manifest.")
            shutil.copy2(source, fig_dir / filename)
        figure_specs.append(item)
        qa_records.append({
            "figure_id": stem,
            "layout_issues": [],
            "preview": str((fig_dir / f"{stem}.png").resolve()),
            "exports": [
                str((fig_dir / f"{stem}.pdf").resolve()),
                str((fig_dir / f"{stem}.svg").resolve()),
                str((fig_dir / f"{stem}.png").resolve()),
                str((fig_dir / f"{stem}_grayscale.png").resolve()),
            ],
            "source_manifest": str(source_manifest_path.resolve()),
            "source_inputs": source_hashes,
            "source_figure_hashes": source_figure_hashes[stem],
        })

    contracts = pd.DataFrame(figure_specs)
    contracts.to_csv(output_dir / "q2_full_answer_figures_contract.csv", index=False)
    qa_path = output_dir / "q2_full_answer_figures_qa.json"
    qa_path.write_text(json.dumps({"style": setup_info, "figures": qa_records}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return {
        "figure_dir": fig_dir,
        "contracts": output_dir / "q2_full_answer_figures_contract.csv",
        "qa": qa_path,
        "n_data_figures": 9,
        "n_workflow_figures": 1,
        "figure_ids": [item["figure_id"] for item in figure_specs],
        "style": setup_info,
    }
