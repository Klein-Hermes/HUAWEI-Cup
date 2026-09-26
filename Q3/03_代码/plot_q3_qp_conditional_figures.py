#!/usr/bin/env python3
"""Create Q3 raw/process/result figures and the modeling flowcharts.

All plots are generated from the frozen Q3 outputs plus the exact source files
used by fit_q3_qp_conditional_model.py.  SVG keeps editable text; PNG is 300 dpi.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Q3" / "04_结果" / "Qp联合模型敏感性_20260926"
FIGDIR = OUT / "figures"
RAW_DIR = ROOT / "中文题目" / "F题" / "real_attachments" / "B_scaling_laws"
Q2_GRID = ROOT / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1" / "b1_fitted_predictions.csv"
SKILL_SCRIPTS = Path(r"C:\Users\86147\.codex\skills\math-modeling\tools\figure\scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from setup_style import setup_style  # noqa: E402
from export_figure import export_figure  # noqa: E402
from visual_qa import audit_layout, render_preview  # noqa: E402


SOURCE_LABELS = {
    "B6": "B6",
    "B8_calibrated": "B8 calibrated",
    "B8_extrapolated": "B8 extrapolated",
}
SOURCE_COLORS = {
    "B6": "#0072B2",
    "B8_calibrated": "#D55E00",
    "B8_extrapolated": "#009E73",
}
G_LABELS = {"exponential": "exp", "power": "power", "logarithmic": "log"}
MODEL_LABELS = {"full": "来源全曲面", "anchored": "B1 参数锚定"}


def load_data() -> dict[str, object]:
    b6 = pd.read_csv(RAW_DIR / "supplementary_NQ_experiment.csv")
    b8 = pd.read_csv(RAW_DIR / "supplementary_NQ_experiment_large.csv")
    sources = {
        "B6": b6.copy(),
        "B8_calibrated": b8.loc[b8["data_type"].eq("calibrated")].copy(),
        "B8_extrapolated": b8.loc[b8["data_type"].eq("extrapolated")].copy(),
    }
    q2 = pd.read_csv(Q2_GRID)
    return {
        "sources": sources,
        "q2": q2,
        "fits": pd.read_csv(OUT / "q_response_full_fits.csv"),
        "anchored": pd.read_csv(OUT / "q_response_b1_anchored_fits.csv"),
        "cv": pd.read_csv(OUT / "q_response_leave_one_q_validation.csv"),
        "feas": pd.read_csv(OUT / "scenario_support_feasibility.csv"),
        "scenarios": pd.read_csv(OUT / "conditional_scenario_results.csv"),
        "transitions": pd.read_csv(OUT / "conditional_transition_pairs.csv"),
        "transboot": pd.read_csv(OUT / "conditional_transition_bootstrap_summary.csv"),
        "kboot": pd.read_csv(OUT / "q_response_anchored_kappa_bootstrap.csv"),
    }


def _save(fig: plt.Figure, name: str, issues: list[dict[str, object]]) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    base = FIGDIR / name
    fig.canvas.draw()
    # Freeze any one-time constrained-layout placement before preview/export.
    # Re-running the engine after colorbar axes are added can otherwise produce
    # a zero-size nested GridSpec on some Matplotlib versions.
    fig.set_layout_engine("none")
    layout = audit_layout(fig)
    issues.extend({"figure": name, "severity": severity, "message": message}
                  for severity, message in layout)
    # A review-sized preview is kept away from the audit directory.
    preview_dir = FIGDIR / "_preview"
    preview_dir.mkdir(exist_ok=True)
    render_preview(fig, str(preview_dir / f"{name}_preview.png"), dpi=120)
    export_figure(fig, str(base), formats=["svg", "png"], dpi=300,
                  size_inches=fig.get_size_inches(), grayscale_preview=False)
    # Grayscale copies live in a nested directory so the top-level SVG/PNG pair
    # audit sees only the deliverable pair for each logical figure.
    gray_dir = FIGDIR / "grayscale_previews"
    gray_dir.mkdir(exist_ok=True)
    with Image.open(base.with_suffix(".png")) as image:
        ImageOps.grayscale(image).save(gray_dir / f"{name}_grayscale.png", dpi=(300, 300))
    plt.close(fig)


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    if grid:
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fig_raw_loss_by_q(data, issues) -> None:
    sources = data["sources"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=False, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.26, wspace=0.34)
    for ax, (source, frame) in zip(axes, sources.items()):
        grouped = [g["val_loss"].to_numpy() for _, g in frame.groupby("Q_score", sort=True)]
        q_values = sorted(frame["Q_score"].unique())
        bp = ax.boxplot(grouped, positions=np.arange(len(q_values)), widths=0.55,
                        patch_artist=True, showfliers=False, medianprops={"color": "#222222", "linewidth": 1.2})
        for box in bp["boxes"]:
            box.set(facecolor=SOURCE_COLORS[source], alpha=0.28, edgecolor=SOURCE_COLORS[source], linewidth=0.9)
        rng = np.random.default_rng(20260926 + len(q_values))
        for i, vals in enumerate(grouped):
            if len(vals) > 500:
                idx = rng.choice(len(vals), 500, replace=False)
                vals = vals[idx]
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(i + jitter, vals, s=7, alpha=0.20, color=SOURCE_COLORS[source], linewidths=0, rasterized=False)
        ax.set_xticks(np.arange(len(q_values)))
        ax.set_xticklabels([f"{q:g}" for q in q_values], rotation=45, ha="right")
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xlabel("观测质量得分 Q")
        ax.set_ylabel("验证损失")
        style_axes(ax)
    fig.suptitle("原始数据：各观测 Q 水平下的验证损失分布", fontsize=13)
    fig.text(0.5, 0.035, "箱线图展示分布；散点为抽样显示。N/D 同时变化，因此该图不作 Q 的因果解释。",
             ha="center", fontsize=8)
    _save(fig, "raw_q3_loss_by_q_distribution", issues)


def fig_raw_q_loss_relation(data, issues) -> None:
    sources = data["sources"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.82, bottom=0.25, wspace=0.62)
    for ax, (source, frame) in zip(axes, sources.items()):
        ax.scatter(frame["Q_score"], frame["val_loss"], color=SOURCE_COLORS[source],
                   s=13, alpha=0.58, edgecolors="none")
        ax.set_title(f"{SOURCE_LABELS[source]}（n={len(frame)}）")
        ax.set_xlabel("观测质量得分 Q")
        ax.set_ylabel("验证损失")
        style_axes(ax)
    fig.suptitle("原始关系：Q 与验证损失的观测配对", fontsize=13)
    fig.text(0.5, 0.035, "颜色表示 N；D 亦为变化因素，散点关系仅描述来源内观测，不等同于单变量效应。",
             ha="center", fontsize=8)
    _save(fig, "raw_q3_q_loss_relation", issues)


def fig_raw_b1_support(data, issues) -> None:
    sources = data["sources"]
    b1 = data["q2"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.82, bottom=0.24, wspace=0.30)
    for ax, source in zip(axes, ("B6", "B8_calibrated")):
        ax.scatter(b1["N_params_B"], b1["D_tokens_B"], s=8, alpha=0.25,
                   color="#666666", marker="o", label="B1 优化网格")
        frame = sources[source]
        cells = frame[["N_params_B", "D_tokens_B"]].drop_duplicates()
        ax.scatter(cells["N_params_B"], cells["D_tokens_B"], s=25, alpha=0.75,
                   color=SOURCE_COLORS[source], marker="s", label=f"{SOURCE_LABELS[source]} 观测单元")
        nd = b1[["N_params_B", "D_tokens_B"]].drop_duplicates()
        nlo, nhi = frame["N_params_B"].min(), frame["N_params_B"].max()
        dlo, dhi = frame["D_tokens_B"].min(), frame["D_tokens_B"].max()
        intersect = nd[nd["N_params_B"].between(nlo, nhi) & nd["D_tokens_B"].between(dlo, dhi)]
        ax.scatter(intersect["N_params_B"], intersect["D_tokens_B"], s=14, facecolors="none",
                   edgecolors="#CC3311", linewidths=0.8, label="支持域交集")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{SOURCE_LABELS[source]}：来源支持与 B1 网格")
        ax.set_xlabel("参数量 N（十亿，对数刻度）")
        ax.set_ylabel("训练 token 数 D（十亿，对数刻度）")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        style_axes(ax)
    fig.suptitle("原始支持域：可用于条件迁移的 N/D 交集", fontsize=13)
    fig.text(0.5, 0.035, "B8 extrapolated 的 N=20–700B，与 B1 网格 N 范围无交集，故不纳入优化面板。",
             ha="center", fontsize=8)
    _save(fig, "raw_q3_b1_source_support", issues)


def pooled_cv(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, model), group in cv.groupby(["source", "model"], sort=False):
        n = int(group["n_test"].sum())
        sse = float(group["sse"].sum())
        sy = float(group["test_sum_y"].sum())
        sy2 = float(group["test_sum_y2"].sum())
        rows.append({"source": source, "model": model, "rmse": np.sqrt(sse / n),
                     "r2": 1.0 - sse / (sy2 - sy * sy / n), "n": n})
    return pd.DataFrame(rows)


def fig_process_fit_cv(data, issues) -> None:
    cv = pooled_cv(data["cv"])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.22, wspace=0.35)
    x = np.arange(3)
    offsets = {"full": -0.11, "anchored": 0.11}
    for ax, metric, title in zip(axes, ("rmse", "r2"), ("留一 Q 水平 RMSE（越低越好）", "留一 Q 水平 R2")):
        for model in ("full", "anchored"):
            subset = cv[cv["model"].eq(model)].set_index("source").reindex(SOURCE_LABELS)
            xs = x + offsets[model]
            for idx, source in enumerate(SOURCE_LABELS):
                ax.scatter(xs[idx], subset.loc[source, metric], s=55, marker="o" if model == "full" else "s",
                           color=SOURCE_COLORS[source], edgecolor="white", linewidth=0.7, zorder=3)
                ax.annotate(f"{subset.loc[source, metric]:.2f}", (xs[idx], subset.loc[source, metric]),
                            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABELS[s] for s in SOURCE_LABELS])
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        style_axes(ax)
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", label="来源全曲面", markersize=6),
              Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", label="B1 参数锚定", markersize=6)]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("模型过程：全曲面与 B1 锚定的留一 Q 验证", fontsize=13, y=0.99)
    fig.text(0.5, 0.025, "每次完整留出一个 Q 水平后重新拟合；RMSE/R2 按全部留出观测汇总。B1 锚定采用未验证的 Q 映射假设。",
             ha="center", fontsize=8)
    _save(fig, "process_q3_fit_cv", issues)


def fig_process_feasibility(data, issues) -> None:
    feas = data["feas"]
    budgets = sorted(feas["budget_flops"].unique())
    contexts = sorted(feas["context_length_tokens"].unique())
    sources = ("B6", "B8_calibrated")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.82, bottom=0.24, wspace=0.35)
    vmax = max(1, int(feas["feasible_at_Q0"].max()))
    for ax, source in zip(axes, sources):
        part = feas[feas["source"].eq(source)]
        mat = part.pivot(index="budget_flops", columns="context_length_tokens", values="feasible_at_Q0").reindex(index=budgets, columns=contexts)
        x_edges = np.arange(mat.shape[1] + 1) - 0.5
        y_edges = np.arange(mat.shape[0] + 1) - 0.5
        im = ax.pcolormesh(x_edges, y_edges, mat.to_numpy(), cmap="Blues", vmin=0,
                           vmax=vmax, shading="flat", edgecolors="white", linewidth=0.9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = int(mat.iloc[i, j])
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > vmax / 2 else "#20242A", fontsize=9)
        ax.set_xticks(range(len(contexts)), [f"{c:,}" for c in contexts], rotation=35, ha="right")
        ax.set_yticks(range(len(budgets)), [f"1e{len(str(int(b))) - 1}" for b in budgets])
        ax.set_xlabel("上下文长度 (token)")
        ax.set_ylabel("预算 (FLOPs)")
        ax.set_title(SOURCE_LABELS[source])
        ax.set_xticks(np.arange(-0.5, len(contexts), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(budgets), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle("模型过程：质量基线下的预算可行域", fontsize=13)
    key_vals = [0, vmax // 2, vmax]
    cmap = plt.get_cmap("Blues")
    fig.legend(handles=[Patch(facecolor=cmap(v / vmax), edgecolor="none", label=f"{v:,}") for v in key_vals],
               title="候选数（代表刻度）", loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.text(0.5, 0.025, "计数只覆盖来源支持域与 B1 实测网格的交集；0 表示该情景无候选配置。",
             ha="center", fontsize=8)
    _save(fig, "process_q3_feasibility", issues)


def fig_process_kappa(data, issues) -> None:
    fits = data["fits"].set_index("source")
    anchored = data["anchored"].set_index("source")
    boot = data["kboot"]
    rows = []
    for source in SOURCE_LABELS:
        rows.append((source, "来源全曲面", float(fits.loc[source, "kappa"]), np.nan, np.nan))
        b = boot.loc[boot["source"].eq(source), "kappa"].to_numpy(float)
        if len(b):
            lo, hi = np.quantile(b, [0.025, 0.975])
        else:
            lo = hi = np.nan
        rows.append((source, "B1 锚定", float(anchored.loc[source, "kappa_B1_anchored"]), lo, hi))
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.84, bottom=0.31)
    yloc = np.arange(len(rows))[::-1]
    marker = {"来源全曲面": "o", "B1 锚定": "s"}
    for y, (source, model, val, lo, hi) in zip(yloc, rows):
        color = SOURCE_COLORS[source]
        if np.isfinite(lo):
            ax.errorbar(val, y, xerr=[[val - lo], [hi - val]], fmt=marker[model], color=color,
                        ecolor=color, capsize=3, markersize=6, linewidth=1.1)
        else:
            ax.scatter(val, y, marker=marker[model], color=color, s=40, edgecolor="white", linewidth=0.5)
        ax.annotate(f"{val:.3f}", (val, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7)
    labels = [f"{SOURCE_LABELS[src]} · {mod}" for src, mod, *_ in rows]
    ax.set_yticks(yloc, labels)
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Q 响应系数 κ")
    ax.set_title("模型过程：Q 效应估计及方向一致性检查")
    ax.text(0.01, -0.20, "方块的横线为锚定 κ 的 95% Bootstrap 区间（1,000 次）；圆点为来源全曲面点估计。\n"
            "B8 extrapolated 不与 B1 网格重叠，因此只报告来源内曲面点估计。",
            transform=ax.transAxes, fontsize=8, va="top")
    style_axes(ax)
    _save(fig, "process_q3_kappa_estimates", issues)


def _scenario_grid(scenarios: pd.DataFrame, value: str):
    order_sources = ["B6", "B8_calibrated"]
    gs = ["exponential", "power", "logarithmic"]
    budgets = sorted(scenarios["budget_flops"].unique())
    contexts = sorted(scenarios["context_length_tokens"].unique())
    xlabels = [f"{len(str(int(b))) - 1} · {c // 1000 if c >= 10000 else c}k" if c >= 10000 else f"{len(str(int(b))) - 1} · {c}"
               for b in budgets for c in contexts]
    rowlabels = [f"{SOURCE_LABELS[s]} · {G_LABELS[g]}" for s in order_sources for g in gs]
    mat = np.full((len(rowlabels), len(xlabels)), np.nan)
    for ri, (source, g) in enumerate((s, g) for s in order_sources for g in gs):
        part = scenarios[(scenarios.source == source) & (scenarios.g_function == g)]
        for bi, budget in enumerate(budgets):
            for ci, context in enumerate(contexts):
                row = part[(part.budget_flops == budget) & (part.context_length_tokens == context)]
                if len(row) and row.iloc[0]["status"] == "feasible" and pd.notna(row.iloc[0][value]):
                    mat[ri, bi * len(contexts) + ci] = float(row.iloc[0][value])
    return mat, rowlabels, xlabels, budgets, contexts


def _plot_scenario_heatmap(ax, matrix, rowlabels, xlabels, title, cmap, vmin=None, vmax=None,
                           annot_fmt=".2f", log_label=None):
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#E7E9EC")
    x_edges = np.arange(matrix.shape[1] + 1) - 0.5
    y_edges = np.arange(matrix.shape[0] + 1) - 0.5
    im = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(matrix), cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, shading="flat",
                       edgecolors="white", linewidth=0.45)
    ax.set_aspect("auto")
    ax.set_yticks(np.arange(len(rowlabels)), rowlabels)
    ax.invert_yaxis()  # keep the first matrix row at the top, as in the source table
    ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=55, ha="right")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rowlabels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    finite = matrix[np.isfinite(matrix)]
    if len(finite):
        for r, c in zip(*np.where(np.isfinite(matrix))):
            val = matrix[r, c]
            text = format(val, annot_fmt)
            red, green, blue, _ = im.cmap(im.norm(val))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(c, r, text, ha="center", va="center", fontsize=6.3,
                    color="white" if luminance < 0.55 else "#20242A")
    ax.set_xlabel("预算量级 (10^k) · 上下文长度 (token)")
    ax.set_ylabel("条件来源 · g(Q)")
    return im


def add_vector_color_key(ax, cmap_name: str, matrix: np.ndarray, title: str, *, fixed_range=None) -> None:
    finite = matrix[np.isfinite(matrix)]
    if not len(finite):
        return
    lo, hi = fixed_range if fixed_range is not None else (float(finite.min()), float(finite.max()))
    if hi <= lo:
        hi = lo + 1.0
    cmap = plt.get_cmap(cmap_name)
    edges = np.linspace(lo, hi, 5)
    handles = []
    for i in range(4):
        color = cmap((i + 0.5) / 4)
        label = f"{edges[i]:.2f}–{edges[i + 1]:.2f}"
        handles.append(Patch(facecolor=color, edgecolor="none", label=label))
    ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.015, 1.0),
              frameon=False, fontsize=6.5, title_fontsize=7)


def fig_result_selected_nd(data, issues) -> None:
    d = data["scenarios"]
    nmat, rows, cols, *_ = _scenario_grid(d, "selected_N_B")
    dmat, _, _, *_ = _scenario_grid(d, "selected_D_B")
    # Values are shown on a log scale, but the cells are annotated in natural units.
    nlog, dlog = np.log10(nmat), np.log10(dmat)
    fig, axes = plt.subplots(2, 1, figsize=(17, 9.3), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.91, top=0.91, bottom=0.23, hspace=0.48)
    im1 = _plot_scenario_heatmap(axes[0], nlog, rows, cols, "选中参数量 N（log10 十亿参数）", "Blues", annot_fmt=".1f")
    im2 = _plot_scenario_heatmap(axes[1], dlog, rows, cols, "选中训练量 D（log10 十亿 token）", "Oranges", annot_fmt=".1f")
    add_vector_color_key(axes[0], "Blues", nlog, "log10 N")
    add_vector_color_key(axes[1], "Oranges", dlog, "log10 D")
    fig.suptitle("结果：不同预算、上下文与成本函数下的条件最优 N/D", fontsize=14)
    fig.text(0.5, 0.025, "仅展示支持域内可行解；灰色单元格为无可行配置。数值是固定 p0 与未验证 Q 映射假设下的条件结果。",
             ha="center", fontsize=8)
    _save(fig, "result_q3_selected_nd", issues)


def fig_result_selected_q_loss(data, issues) -> None:
    d = data["scenarios"]
    qmat, rows, cols, *_ = _scenario_grid(d, "selected_Q")
    lmat, _, _, *_ = _scenario_grid(d, "predicted_loss")
    fig, axes = plt.subplots(2, 1, figsize=(17, 9.3), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.91, top=0.91, bottom=0.23, hspace=0.48)
    im1 = _plot_scenario_heatmap(axes[0], qmat, rows, cols, "最优质量水平 Q", "viridis", vmin=0, vmax=1, annot_fmt=".2f")
    im2 = _plot_scenario_heatmap(axes[1], lmat, rows, cols, "条件模型预测验证损失", "magma", annot_fmt=".2f")
    add_vector_color_key(axes[0], "viridis", qmat, "Q", fixed_range=(0, 1))
    add_vector_color_key(axes[1], "magma", lmat, "损失")
    fig.suptitle("结果：条件最优质量水平及对应损失", fontsize=14)
    fig.text(0.5, 0.025, "预算由低到高、上下文由短到长；灰色单元格表示预算下无候选解。该预测未验证为 B1 真实效果。",
             ha="center", fontsize=8)
    _save(fig, "result_q3_selected_q_loss", issues)


def fig_result_transition_stability(data, issues) -> None:
    boot = data["transboot"]
    trans = data["transitions"]
    sources = ["B6", "B8_calibrated"]
    gs = ["exponential", "power", "logarithmic"]
    edges = list(dict.fromkeys(trans["edge_id"].tolist()))
    # Use the static edge classifications to keep both panels on the same edge order.
    cfg = np.full((6, len(edges)), np.nan)
    share = np.full_like(cfg, np.nan)
    rowlabels = []
    for ri, (source, g) in enumerate((s, g) for s in sources for g in gs):
        rowlabels.append(f"{SOURCE_LABELS[source]} · {G_LABELS[g]}")
        part = boot[(boot.source == source) & (boot.g_function == g)].set_index("edge_id")
        for ci, edge in enumerate(edges):
            if edge not in part.index:
                continue
            row = part.loc[edge]
            # Boundary edges are not parameter transitions and remain blank/gray.
            if int(row.get("both_feasible_replicates", 0)) > 0:
                cfg[ri, ci] = float(row["configuration_change_frequency"])
                share[ri, ci] = float(row["cost_share_change_frequency"])
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 7.5), constrained_layout=False)
    fig.subplots_adjust(left=0.15, right=0.91, top=0.89, bottom=0.22, hspace=0.46)
    im1 = _plot_scenario_heatmap(axes[0], cfg, rowlabels, edges, "N/D/Q 配置变化频率", "Blues", vmin=0, vmax=1, annot_fmt=".2f")
    im2 = _plot_scenario_heatmap(axes[1], share, rowlabels, edges, "成本份额变化频率", "Oranges", vmin=0, vmax=1, annot_fmt=".2f")
    for ax in axes:
        ax.set_xticklabels(edges, rotation=55, ha="right", fontsize=7)
        ax.set_xlabel("相邻预算/上下文边（灰格为可行域边界边）")
    add_vector_color_key(axes[0], "Blues", cfg, "变化频率", fixed_range=(0, 1))
    add_vector_color_key(axes[1], "Oranges", share, "变化频率", fixed_range=(0, 1))
    fig.suptitle("结果：相邻情景结构转移的参数稳定性（1,000 次 Bootstrap）", fontsize=13)
    fig.text(0.5, 0.025, "仅对两端均可行的边统计配置/成本份额变化；可行域进出单独标作边界事件，不赋予参数转移频率。",
             ha="center", fontsize=8)
    _save(fig, "result_q3_transition_stability", issues)


def _draw_node(ax, xy, width, height, text, kind="process", fontsize=9, face="#F4F6F8"):
    x, y = xy
    if kind in ("start", "end"):
        node = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.12",
                                      facecolor=face, edgecolor="#425466", linewidth=1.1)
    elif kind == "data":
        skew = 0.16
        node = patches.Polygon([(x+skew,y),(x+width,y),(x+width-skew,y+height),(x,y+height)],
                               closed=True, facecolor=face, edgecolor="#425466", linewidth=1.1)
    elif kind == "decision":
        node = patches.Polygon([(x+width/2,y+height),(x+width,y+height/2),(x+width/2,y),(x,y+height/2)],
                               closed=True, facecolor=face, edgecolor="#425466", linewidth=1.1)
    else:
        node = patches.Rectangle((x,y),width,height,facecolor=face,edgecolor="#425466",linewidth=1.1)
    ax.add_patch(node)
    ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x, y, width, height)


def _arrow(ax, a, b, label=None, color="#44515E", rad=0.0):
    # Inputs are node extents; terminate arrows at their box boundaries so
    # flow lines remain outside node text.
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    ca = np.array((xa + wa / 2, ya + ha / 2), dtype=float)
    cb = np.array((xb + wb / 2, yb + hb / 2), dtype=float)
    delta = cb - ca
    ratios_a = [((wa / 2) / abs(delta[0])) if abs(delta[0]) > 1e-12 else np.inf,
                ((ha / 2) / abs(delta[1])) if abs(delta[1]) > 1e-12 else np.inf]
    ratios_b = [((wb / 2) / abs(delta[0])) if abs(delta[0]) > 1e-12 else np.inf,
                ((hb / 2) / abs(delta[1])) if abs(delta[1]) > 1e-12 else np.inf]
    p1 = ca + min(ratios_a) * delta
    p2 = cb - min(ratios_b) * delta
    p1, p2 = tuple(p1), tuple(p2)
    ax.annotate("", xy=p2, xytext=p1, arrowprops={"arrowstyle":"-|>","lw":1.15,"color":color,
                 "shrinkA":0,"shrinkB":0,"connectionstyle":f"arc3,rad={rad}"})
    if label:
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.10, label,
                fontsize=7, ha="center", va="center", color=color,
                bbox={"facecolor":"white","edgecolor":"none","pad":1.0,"alpha":0.9})


def fig_flow_overall(issues) -> None:
    fig, ax = plt.subplots(figsize=(14, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.16)
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.4); ax.axis("off")
    xs = [0.25, 2.15, 4.25, 6.35, 8.45, 10.55, 12.45]
    w, h, y = 1.45, 0.82, 1.35
    labels = ["题目与数据", "Q1\n领域价值与配比", "Q2\nScaling law 估计", "Q3\n资源预算优化", "Q4\n技术前沿预测", "验证与敏感性", "结论与建议"]
    kinds = ["start", "process", "process", "process", "process", "process", "end"]
    nodes = [_draw_node(ax, (x,y), w,h,label,kind=k,fontsize=8.2) for x,label,k in zip(xs,labels,kinds)]
    for a,b in zip(nodes,nodes[1:]): _arrow(ax,a,b)
    ax.text(7, 2.75, "总体建模流程：数据输入 → 分题求解 → 验证 → 综合输出", ha="center", fontsize=13, weight="bold")
    ax.text(7, 0.48, "Q3 的结果依赖 Q1/Q2 的冻结输入；各题模型结论按可识别范围分别解释。", ha="center", fontsize=8)
    _save(fig, "flow_overall_model", issues)


def fig_flow_q3(issues) -> None:
    fig, ax = plt.subplots(figsize=(14.8, 8.0), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.89, bottom=0.065)
    ax.set_xlim(0, 14.8); ax.set_ylim(0, 8.0); ax.axis("off")
    # Three horizontal rows form a non-crossing serpentine path.
    ax.add_patch(patches.Rectangle((0.15,5.0),14.5,2.25,facecolor="#F5F7F9",edgecolor="none",zorder=-5))
    ax.add_patch(patches.Rectangle((0.15,2.55),14.5,2.1,facecolor="#F1F5F7",edgecolor="none",zorder=-5))
    ax.add_patch(patches.Rectangle((0.15,0.18),14.5,2.05,facecolor="#F5F7F9",edgecolor="none",zorder=-5))
    ax.text(7.4,7.03,"数据估计与迁移门槛",ha="center",fontsize=10,weight="bold")
    ax.text(7.4,4.43,"固定政策下的预算优化",ha="center",fontsize=10,weight="bold")
    ax.text(7.4,2.00,"转移稳定性与结论",ha="center",fontsize=10,weight="bold")
    xrow = [0.75, 4.25, 7.75, 11.25]
    w, h = 2.75, 0.85
    y1, y2, y3 = 5.65, 3.15, 0.78
    row1 = [
        _draw_node(ax,(xrow[0],y1),w,h,"读取 B6/B8、B1 网格\n与 C7 上下文","data",8.2),
        _draw_node(ax,(xrow[1],y1),w,h,"估计来源 Q 曲面\n与 B1 锚定 κ","process",8.2),
        _draw_node(ax,(xrow[2],y1),w,h,"留一 Q 验证\n审查 N/D 支持交集","process",8.0),
        _draw_node(ax,(xrow[3],y1),w,h,"B1 重叠与 Q 映射\n是否有可验证证据？","decision",7.7),
    ]
    for a,b in zip(row1,row1[1:]): _arrow(ax,a,b)
    row2 = [
        _draw_node(ax,(xrow[3],y2),w,h,"固定共同政策 p0\n冻结 M0 基线","process",8.2),
        _draw_node(ax,(xrow[2],y2),w,h,"枚举预算×上下文×g(Q)\n仅保留 N/D 支持交集","process",8.0),
        _draw_node(ax,(xrow[1],y2),w,h,"单调二分求 Q 上界\n严格校正 FLOPs","process",8.0),
        _draw_node(ax,(xrow[0],y2),w,h,"输出条件最优\nN/D/Q 与损失","end",8.0),
    ]
    _arrow(ax,row1[3],row2[0],"当前门槛未过：走 M0 支路")
    for a,b in zip(row2,row2[1:]): _arrow(ax,a,b)
    row3 = [
        _draw_node(ax,(xrow[0],y3),w,h,"相邻情景配置转移\n与成本份额变化","process",8.0),
        _draw_node(ax,(xrow[1],y3),w,h,"1,000 次条件参数\nBootstrap 稳定性","process",8.0),
        _draw_node(ax,(xrow[2],y3),w,h,"汇总来源差异\n与边界事件","process",8.0),
        _draw_node(ax,(xrow[3],y3),w,h,"报告估计结果、\n条件优化与适用边界","end",8.0),
    ]
    _arrow(ax,row2[3],row3[0])
    for a,b in zip(row3,row3[1:]): _arrow(ax,a,b)
    ax.text(7.4,7.72,"Q3 条件 Q/p 联合判定与情景优化流程",ha="center",fontsize=13,weight="bold")
    _save(fig, "flow_q3_model", issues)


def write_contracts() -> None:
    content = """# Q3 图表契约\n\n"""
    contracts = [
        ("raw_q3_loss_by_q_distribution", "展示 B6、B8 校准与外推来源的观测损失分布随离散 Q 水平的变化；箱线图保留组内分布，点为抽样观测。", "原始数据；不得解释为单独 Q 因果效应，因为 N/D 同时变化。"),
        ("raw_q3_q_loss_relation", "展示来源内 Q 与验证损失的观测配对，并用颜色编码 N。", "原始关系；识别共变因素，不作单变量效应推断。"),
        ("raw_q3_b1_source_support", "展示 B1 N/D 网格与 B6/B8 calibrated 来源支持范围及交集。", "原始支持域；决定迁移优化的可用域。"),
        ("process_q3_fit_cv", "对比来源全曲面和 B1 参数锚定模型的留一 Q 水平预测误差。", "过程验证；锚定列依赖未验证的 Q 数值桥接。"),
        ("process_q3_feasibility", "展示 Q=Q0 下预算与上下文组合的支持域可行候选数。", "过程域；不将零可行点误写成质量转移。"),
        ("process_q3_kappa_estimates", "对照来源曲面 κ 与锚定 κ，展示锚定 bootstrap 区间。", "参数诊断；外推来源仅列来源内点估计。"),
        ("result_q3_selected_nd", "展示来源、g、预算、上下文下的条件最优 N 与 D。", "结果；灰格为不可行，使用固定 p0 和未验证桥接。"),
        ("result_q3_selected_q_loss", "展示条件最优 Q 与对应预测损失。", "结果；不可解释为已验证 B1 效果。"),
        ("result_q3_transition_stability", "展示 1000 次条件参数 bootstrap 中的配置/成本份额变化频率。", "结果稳定性；边界边留空，单独视为可行域进出。"),
        ("flow_overall_model", "概括题目数据、Q1–Q4 求解、验证与综合结论的依赖链。", "建模流程图，不计入三类数据图配额。"),
        ("flow_q3_model", "展示来源估计、支持域与预算优化、转移稳定性及结论限定。", "Q3 方法流程图，节点对应当前分析脚本实际步骤。"),
    ]
    for name, claim, evidence in contracts:
        content += f"## `{name}`\n\n- 核心结论：{claim}\n- 证据与风险：{evidence}\n- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。\n\n"
    (FIGDIR / "figure_contracts.md").write_text(content, encoding="utf-8")


def main() -> int:
    setup_style(journal="general", lang="zh")
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.constrained_layout.use"] = False
    data = load_data()
    issues: list[dict[str, object]] = []
    fig_raw_loss_by_q(data, issues)
    fig_raw_q_loss_relation(data, issues)
    fig_raw_b1_support(data, issues)
    fig_process_fit_cv(data, issues)
    fig_process_feasibility(data, issues)
    fig_process_kappa(data, issues)
    fig_result_selected_nd(data, issues)
    fig_result_selected_q_loss(data, issues)
    fig_result_transition_stability(data, issues)
    fig_flow_overall(issues)
    fig_flow_q3(issues)
    write_contracts()
    issue_path = OUT / "figure_visual_qa.json"
    issue_path.write_text(pd.Series(issues).to_json(force_ascii=False, indent=2), encoding="utf-8")
    fails = [item for item in issues if item["severity"] == "FAIL"]
    warns = [item for item in issues if item["severity"] == "WARN"]
    print(f"figures={11}, layout_failures={len(fails)}, layout_warnings={len(warns)}")
    print(f"contract={FIGDIR / 'figure_contracts.md'}")
    for issue in issues:
        print(f"{issue['severity']} {issue['figure']}: {issue['message']}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
