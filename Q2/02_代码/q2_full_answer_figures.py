"""Reproducible Q2 full-answer evidence figures.

Figures separate A-side mixture evidence, B1 classical scaling evidence, and
semi-synthetic Q response. They do not imply that A/B rows form a joint sample.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


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
    """Create six data figures and one full-answer workflow chart."""
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
             claim: str, evidence: str, warning: str) -> None:
        issues = audit_layout(fig)
        failures = [str(item) for item in issues if str(item).upper().startswith("FAIL")]
        if failures:
            plt.close(fig)
            raise RuntimeError(f"Figure layout QA failed for {stem}: {failures}")
        preview_path = preview_dir / f"{stem}_preview.png"
        render_preview(fig, str(preview_path), dpi=150)
        paths = export_figure(
            fig, str(fig_dir / stem), formats=["pdf", "svg", "png"],
            size_inches=size, dpi=300, grayscale_preview=True,
        )
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

    # RESULT — target-wise held-out RMSE for the A-side p-only model across training scales.
    held = q1_holdout_metrics[
        (q1_holdout_metrics["model"] == "simplex_ridge")
        & (q1_holdout_metrics["variant"] == "closed")
        & q1_holdout_metrics["split"].isin(["test_1m", "test_60m", "test_1b"])
    ].copy()
    split_order = ["test_1m", "test_60m", "test_1b"]
    split_labels = ["1M", "60M", "1B"]
    if held.groupby("split")["target"].nunique().to_dict() != {s: 13 for s in split_order}:
        raise ValueError("Expected 13 held-out target metrics at each scale")
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    rng = np.random.default_rng(20260924)
    means_r2 = held.groupby("split")["r2"].mean().to_dict()
    for idx, (split, label) in enumerate(zip(split_order, split_labels), start=1):
        values = held.loc[held["split"] == split, "rmse"].to_numpy(float)
        bp = ax.boxplot(values, positions=[idx], widths=0.44, patch_artist=True,
                        showfliers=False, medianprops={"color": "#202124", "linewidth": 1.1},
                        whiskerprops={"color": "#6B7280", "linewidth": 0.8},
                        capprops={"color": "#6B7280", "linewidth": 0.8})
        bp["boxes"][0].set_facecolor([PALETTE["primary"], PALETTE["secondary"], PALETTE["contrast"]][idx - 1])
        bp["boxes"][0].set_alpha(0.30)
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(idx + jitter, values, s=18, alpha=0.78,
                   color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["contrast"]][idx - 1],
                   edgecolor="white", linewidth=0.35, zorder=3)
        ax.text(idx, 4.55, f"mean R2={means_r2[split]:.3g}", ha="center", va="top", fontsize=6.4)
    ax.set_xticks([1, 2, 3], split_labels)
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 4.9)
    ax.set_ylabel("13 个 Loss 目标的目标级 RMSE")
    ax.set_xlabel("配比模型训练尺度")
    ax.set_title("A 侧 p-only 的跨尺度误差扩大", loc="left")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.text(0.01, -0.17,
            "箱线为 13 个目标的分布，点为各目标 RMSE；不是 13 次独立训练。模型在 1M 训练，60M/1B 只作跨尺度检查。",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color="#4B5563")
    save(fig, "result_q2_p_cross_scale_rmse", size=(6.3, 4.0), category="result",
         claim="A 侧 p-only 模型的目标级绝对误差随测试尺度增大，不能作为 B 侧通用 p 系数。",
         evidence="Q1.3 holdout_metrics.csv; 13 targets per held-out scale.",
         warning="RMSE 按目标汇总；跨尺度样本并非独立训练复制，不能外推到 B1。")

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

    contracts = pd.DataFrame(figure_specs)
    contracts.to_csv(output_dir / "q2_full_answer_figures_contract.csv", index=False)
    qa_path = output_dir / "q2_full_answer_figures_qa.json"
    qa_path.write_text(json.dumps({"style": setup_info, "figures": qa_records}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    return {
        "figure_dir": fig_dir,
        "contracts": output_dir / "q2_full_answer_figures_contract.csv",
        "qa": qa_path,
        "n_data_figures": 6,
        "n_workflow_figures": 1,
        "figure_ids": [item["figure_id"] for item in figure_specs],
        "style": setup_info,
    }
