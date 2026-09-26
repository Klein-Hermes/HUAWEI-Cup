# -*- coding: utf-8 -*-
"""
问题一 · 新增图（3 张，补论证链缺口）

  1. q1-model-gain          主模型相对训练均值基线的误差改善（按 13 个损失目标）
  2. q1-conflict-effect     冲突度与评分分歧的关系（冲突是否真的影响评分）
  3. q1-substitution-volcano 配比替代效应的方向确定性（效应量 × 方向概率）

新增理由：原 15 张图中，没有任何一张直接展示"配比模型的收益"；
冲突分析只报了冲突度本身，未展示其对评分的实际影响；
替代效应只给了区间计数，未展示方向确定性。

运行：python3 plot_q1_new.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

S.setup()

Q11 = S.res("q1_1", "v1")
Q12 = S.res("q1_2_conflict_model", "v1")
Q13 = S.res("q1_3", "v1")


# ----------------------------------------------------------------------
def fig_model_gain():
    """主模型相对训练均值基线的误差改善（按 13 个损失目标）"""
    df = pd.read_csv(Q13 / "nested_cv_metrics.csv")
    nc = df[df.split == "nested_cv_closed"]
    piv = nc.pivot_table(index="target", columns="model", values="rmse")
    piv = piv[["mean", "simplex_ridge"]].dropna()
    piv["gain"] = (piv["mean"] - piv["simplex_ridge"]) / piv["mean"] * 100
    piv = piv.sort_values("gain")

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 4.4), sharey=True,
        gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.08))
    for a in (ax, ax2):
        S.style_axes(a, grid_axis="x")
    y = np.arange(len(piv))

    # 左：两条 RMSE 的哑铃
    for i, (_, r) in enumerate(piv.iterrows()):
        ax.plot([r["simplex_ridge"], r["mean"]], [i, i],
                color=S.C["ref"], lw=1.1, zorder=1)
    ax.scatter(piv["mean"], y, s=42, color=S.C["base"], zorder=3,
               label="训练均值基线", edgecolors="white", lw=0.6)
    ax.scatter(piv["simplex_ridge"], y, s=42, color=S.C["ours"], zorder=3,
               marker="D", label="单纯形岭回归（主模型）", edgecolors="white", lw=0.6)
    ax.set_yticks(y, [S.short_domain(t) for t in piv.index], fontsize=8)
    ax.set_xlabel("嵌套交叉验证 RMSE")
    S.legend_below(ax, ncol=2, y=-0.17, fontsize=8.6)

    # 右：改善幅度
    ax2.barh(y, piv["gain"], height=0.62, color=S.C["ours"], alpha=0.9)
    gmax = piv["gain"].max()
    for i, v in enumerate(piv["gain"]):
        ax2.text(gmax * 1.06, i, f"{v:.1f}%", fontsize=8, va="center",
                 ha="left", color="#404040")
    ax2.set_xlabel("相对基线的 RMSE 改善")
    ax2.set_xlim(0, gmax * 1.32)
    ax2.axvline(piv["gain"].mean(), color=S.C["hl"], lw=1.2, ls="--",
                label=f'平均 {piv["gain"].mean():.1f}%')
    ax2.set_ylim(-0.7, len(piv) - 0.3)
    S.save(fig, "q1-model-gain")


# ----------------------------------------------------------------------
def fig_conflict_effect():
    """冲突度与评分分歧的关系（冲突是否真的影响评分）"""
    cols = ["dataset", "conflict_intensity", "delta_q_huber_minus_equal",
            "high_conflict"]
    df = pd.read_csv(Q12 / "sample_scores.csv.gz", usecols=cols)
    a1 = df[df.dataset == "a1"].dropna(subset=cols[1:3]).copy()
    a1["abs_dq"] = a1["delta_q_huber_minus_equal"].abs()

    # wspace 必须留够：左面板右侧挂着色标，右面板左侧挂着 y 轴刻度标签，
    # 两者之间的净空余不足时标签会伸进色标把刻度数字压住。
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.5),
        gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.40))

    # 左：冲突度 vs |ΔQ|（六边形密度 + 分箱中位数曲线）
    S.style_axes(ax)
    hb = ax.hexbin(a1["conflict_intensity"], a1["abs_dq"], gridsize=46,
                   cmap="viridis", mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02, shrink=0.9)
    cb.set_label("样本数", fontsize=8.4); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=8)
    qs = np.linspace(a1["conflict_intensity"].quantile(.01),
                     a1["conflict_intensity"].quantile(.99), 27)
    centers = (qs[:-1] + qs[1:]) / 2                  # 26 个箱中心
    idx = np.digitize(a1["conflict_intensity"], qs)
    med = [a1["abs_dq"][idx == k].median() for k in range(1, len(qs))]
    ax.plot(centers, med, color=S.C["base"], lw=2.0, zorder=5,
            label="分箱中位数")
    thr = a1["conflict_intensity"][a1["high_conflict"]].min() \
        if a1["high_conflict"].any() else None
    if thr is not None:
        S.ref_line(ax, thr, axis="x", label="$q_{95}$ 阈值", color=S.C["warn"])
    ax.set_xlabel("行内冲突度 $D$")
    ax.set_ylabel("|$\\Delta Q$|（两候选评分之差）")
    S.legend_below(ax, ncol=2, y=-0.24, fontsize=8.6)
    S.panel_label(ax, "a")

    # 右：高冲突组 vs 其余组的 |ΔQ| 分布对比
    S.style_axes(ax2, grid_axis="y")
    hi = a1.loc[a1["high_conflict"], "abs_dq"].values
    lo = a1.loc[~a1["high_conflict"], "abs_dq"].values
    parts = ax2.violinplot([lo, hi], orientation="horizontal", showextrema=False, widths=0.78)
    for b, c in zip(parts["bodies"], [S.C["ref"], S.C["base"]]):
        b.set_facecolor(c); b.set_alpha(0.75); b.set_edgecolor("white")
    for pos, arr, c in [(1, lo, S.C["ref"]), (2, hi, S.C["base"])]:
        m = float(np.median(arr))
        ax2.scatter(m, pos, marker="o", s=34, color="white",
                    edgecolors=c, lw=1.4, zorder=5)
        # 中位数值放在面板右侧留白里，靠右对齐。原先它挂在 y 轴刻度的第二行，
        # 两行刻度标签会向左伸进左侧面板的色标刻度区压住那些数字；
        # 改到面板内部又会落在小提琴体上，故用右侧空白。
        ax2.annotate(f"中位 {m:.4f}", xy=(0.985, pos),
                     xycoords=("axes fraction", "data"),
                     ha="right", va="center", fontsize=8.2, color=c, zorder=6)
    ax2.set_yticks([1, 2], ["其余样本", "高冲突样本"], fontsize=8.8)
    ax2.set_xlabel("|$\\Delta Q$|")
    ax2.set_ylim(0.5, 2.6)
    # 小提琴体覆盖的是数据全距而非分位数，故右边界要按实际最大值留白，
    # 否则长尾会伸到标注所在的那一列
    ax2.set_xlim(0, max(float(hi.max()), float(lo.max())) * 1.55)
    S.panel_label(ax2, "b")
    S.save(fig, "q1-conflict-effect")


# ----------------------------------------------------------------------
def fig_substitution_volcano():
    """配比替代效应的方向确定性（效应量 × 方向概率）"""
    df = pd.read_csv(Q13 / "substitution_effects_10pp_bootstrap.csv")
    df = df.dropna(subset=["effect_per_10pp_transfer",
                           "bootstrap_probability_effect_positive"])
    p = df["bootstrap_probability_effect_positive"].values
    e = df["effect_per_10pp_transfer"].values
    # 方向确定性：越靠近 0 或 1 越确定
    det = np.abs(p - 0.5) * 2
    decided = det >= 0.95

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    S.style_axes(ax)
    ax.scatter(e[~decided], p[~decided], s=11, alpha=0.45, color=S.C["ref"],
               edgecolors="none", label=f"方向未定（{int((~decided).sum())} 个区间）")
    ax.scatter(e[decided & (e > 0)], p[decided & (e > 0)], s=13, alpha=0.65,
               color=S.C["base"], edgecolors="none",
               label=f"方向为正（{int((decided & (e > 0)).sum())} 个）")
    ax.scatter(e[decided & (e <= 0)], p[decided & (e <= 0)], s=13, alpha=0.65,
               color=S.C["ours"], edgecolors="none",
               label=f"方向为负（{int((decided & (e <= 0)).sum())} 个）")
    S.ref_line(ax, 0.5, axis="y", label="方向无偏", color=S.C["ref"])
    S.ref_line(ax, 0.0, axis="x", label="零效应", color=S.C["ref"])
    ax.set_xlabel("10 个百分点配比转移的效应量")
    ax.set_ylabel("bootstrap 中效应为正的概率")
    ax.set_ylim(-0.04, 1.04)
    S.legend_below(ax, ncol=2, y=-0.22, fontsize=8.6)
    S.save(fig, "q1-substitution-volcano")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("问题一 · 新增图：")
    fig_model_gain()
    fig_conflict_effect()
    fig_substitution_volcano()
    print("完成。")
