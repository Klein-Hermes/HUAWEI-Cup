# -*- coding: utf-8 -*-
"""
问题三 · 算力约束下的资源配置（3 张图）

  1. q3-c7-lengths   C7 上下文长度的分布与临界长度 30000 的位置
  2. q3-frontier     15 个预算×上下文情景的预算利用率与成本份额
  3. q3-cost-curve   三种质量成本函数的增量算力曲线

数据源（全部为已冻结的 Q3 产物，本脚本只读取、不重算、不重拟合）：
  Q3/02_数据审计/C7_context_by_model.csv          (45 行)
  Q3/04_结果/M0_ND_reference_frontier.csv         (15 行)
  Q3/04_结果/M0_budget_context_cost_shares.csv    (15 行)
  Q3/04_结果/Q_cost_increment_envelope.csv        (36 行)

设计说明：图内不放标题（解释交给 LaTeX 题注），中文标签，Okabe-Ito 语义色，
被画的量全部直接读自 CSV，无任何硬编码数值。

运行：python3 plot_q3.py
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import style as S

S.setup()

REPO = S.DATA_ROOT.parent
Q3 = REPO / "Q3"
AUDIT = Q3 / "02_数据审计"
OUT = Q3 / "04_结果"

ETA = 2e-4
L_CRIT = 6 / ETA                      # = 30000，由题面成本公式解析推出

BUDGET_LABEL = {10**19: "$10^{19}$", 10**22: "$10^{22}$", 10**24: "$10^{24}$"}


def budget_label(b):
    """预算的 LaTeX 标签。注意 10**24 转 float 后不再精确等于 int(10**24)，
    不能直接拿 float 去查 int 键，故按数量级还原。"""
    return f"$10^{{{int(round(math.log10(b)))}}}$"


G_LABEL = {"exponential": "指数型 $\\gamma e^{\\lambda Q}$",
           "power4": "幂函数型 $\\gamma Q^{\\lambda}$",
           "logarithmic": "对数渐进型 $\\gamma\\ln(1+\\lambda Q)$"}
G_COLOR = {"exponential": S.C["ours"], "power4": S.C["base"],
           "logarithmic": S.C["alt"]}


# ----------------------------------------------------------------------
def fig_c7_lengths():
    """C7 上下文长度的分布，临界长度以参考线标出。

    高于 / 低于临界长度用语义色区分：低于为蓝（训练开销占优），高于为琥珀
    （注意力开销占优）。柱宽在对数轴上取固定比例，避免相邻柱粘连。
    """
    df = pd.read_csv(AUDIT / "C7_context_by_model.csv")
    assert len(df) == 45, f"C7 应为 45 行，实际 {len(df)}"

    g = (df.groupby("max_position_embeddings")
           .size().reset_index(name="n").sort_values("max_position_embeddings"))
    assert len(g) == 5, f"C7 应得到 5 个离散长度，实际 {len(g)}"

    x = g["max_position_embeddings"].values.astype(float)
    n = g["n"].values
    above = x > L_CRIT

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    S.style_axes(ax, grid_axis="y")

    ax.bar(x, n, width=x * 0.42, align="center",
           color=[S.C["hl"] if a else S.C["ours"] for a in above],
           edgecolor="white", lw=0.8, zorder=3)
    for xi, ni in zip(x, n):
        ax.text(xi, ni + 0.45, str(ni), ha="center", fontsize=8.6,
                color=S.C["ref"], zorder=4)

    S.ref_line(ax, L_CRIT, axis="x", color=S.C["warn"], ls="--",
               label="临界长度 30000")
    ax.set_xscale("log")
    ax.set_xlim(x.min() * 0.5, x.max() * 2.0)
    ax.set_ylim(0, n.max() * 1.28)
    ax.set_xticks(x, [f"{int(v)}" for v in x])
    ax.set_xlabel("架构支持的最大上下文长度（token，对数轴）")
    ax.set_ylabel("模型数")
    ax.legend(handles=[
        Patch(facecolor=S.C["ours"], label="低于临界（训练开销占优）"),
        Patch(facecolor=S.C["hl"], label="高于临界（注意力开销占优）"),
        Line2D([], [], color=S.C["warn"], ls="--", lw=1.0, label="临界长度 30000"),
    # 图例放右上：2048 柱高 18 且位于左侧，左上位置会把它的数值标签盖住；
    # 右侧只有 131072 一根矮柱，右上是唯一的空区。
    ], loc="upper right", fontsize=8.4)
    S.save(fig, "q3-c7-lengths")


# ----------------------------------------------------------------------
def fig_frontier():
    """15 个情景的预算利用率与注意力开销份额。"""
    fr = pd.read_csv(OUT / "M0_ND_reference_frontier.csv")
    sh = pd.read_csv(OUT / "M0_budget_context_cost_shares.csv")
    assert len(fr) == 15 and len(sh) == 15

    fr = fr.copy()
    fr["budget"] = fr["budget_flops"].astype(float)
    fr["L"] = fr["context_length_tokens"].astype(int)
    fr["util"] = fr["budget_utilization"].astype(float)
    budgets = sorted(fr["budget"].unique())
    ctxs = sorted(fr["L"].unique())
    bcol = {b: c for b, c in zip(budgets, [S.C["ours"], S.C["base"], S.C["alt"]])}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5),
                                  gridspec_kw=dict(wspace=0.26))
    S.style_axes(ax); S.style_axes(ax2)

    # (a) 利用率：同一预算下随上下文变化
    for b in budgets:
        sub = fr[fr.budget == b].sort_values("L")
        ax.plot(sub["L"], sub["util"] * 100, marker="o", ms=5.5,
                color=bcol[b], lw=1.4, label=budget_label(b), zorder=3)
    ax.axhline(100, color=S.C["ref"], ls="--", lw=1.0, zorder=1)
    ax.set_xscale("log"); ax.set_xticks(ctxs, [str(c) for c in ctxs])
    # 两端留白：最左点若贴住左脊，居中标注的左缘会压到 y 轴脊线上
    ax.set_xlim(1450, 190000)
    ax.set_xlabel("上下文长度 $L_{\\mathrm{ctx}}$")
    ax.set_ylabel("预算利用率（%）")
    ax.set_ylim(0, 112)
    ax.legend(loc="center left", fontsize=8.4, title="算力预算", title_fontsize=8.4)
    S.panel_label(ax, "a")

    # (b) 注意力开销份额：只依赖上下文长度，与预算无关
    s = (sh.assign(L=sh["context_length_tokens"].astype(int))
           .groupby("L")["attention_cost_share"].median().sort_index())
    ax2.plot(s.index, s.values * 100, marker="D", ms=5.5, color=S.C["hl"], lw=1.5,
             zorder=3)
    for k, (xi, yi) in enumerate(zip(s.index, s.values)):
        # 曲线单调上升，标记上方紧邻的就是下一段折线，故标注一律放在标记下方；
        # 唯一的例外是起点，它下方空间最窄，改放上方。
        off, va = ((0, 13), "bottom") if k == 0 else ((0, -13), "top")
        ax2.annotate(f"{yi*100:.1f}%", (xi, yi * 100),
                     textcoords="offset points", xytext=off, va=va,
                     ha="center", fontsize=8.2, color=S.C["hl"])
    S.ref_line(ax2, 50, axis="y", color=S.C["ref"], ls="--", label="份额过半")
    ax2.set_xscale("log"); ax2.set_xticks(ctxs, [str(c) for c in ctxs])
    ax2.set_xlim(1450, 190000)
    ax2.set_xlabel("上下文长度 $L_{\\mathrm{ctx}}$")
    ax2.set_ylabel("注意力开销占总额（%）")
    # 下界留出负空间：最低一点在 6.4%，标注放在标记下方，不留空间会压到底轴
    ax2.set_ylim(-11, 95)
    ax2.set_yticks([0, 20, 40, 60, 80])
    ax2.legend(loc="upper left", fontsize=8.4)
    S.panel_label(ax2, "b")
    S.save(fig, "q3-frontier")


# ----------------------------------------------------------------------
def fig_cost_curve():
    """三种 g(Q) 的增量算力随目标质量的变化（D 取 B1 中位值）。

    纵轴取对数：三种形式在同一起点相差近一个量级，线性轴会把两条曲线压扁。
    """
    df = pd.read_csv(OUT / "Q_cost_increment_envelope.csv")
    df = df[df["D_B1_median_tokens"].notna()].copy()
    df["q"] = df["Q_target"].astype(float)
    # Q=Q0 处增量恰为 0，对数轴无法表示，置为 NaN 让 matplotlib 断开该点
    df["cq"] = df["C_Q_at_D_B1_median_flops"].astype(float).replace(0.0, np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    S.style_axes(ax)

    for g, sub in df.groupby("g_function"):
        sub = sub.sort_values("q")
        ax.plot(sub["q"], sub["cq"], marker="o", ms=3.6, lw=1.4,
                color=G_COLOR[g], label=G_LABEL[g], zorder=3)

    q0 = float(df["Q0_q_huber"].iloc[0])
    S.ref_line(ax, q0, axis="x", color=S.C["ref"], ls=":",
               label=f"质量基线 $Q_0={q0:.4f}$")

    # 三档预算参考线：低于线者表示该质量水平在对应预算下不可负担。
    # 标签放在坐标区【右侧之外】——区内三条曲线会穿过全部三档预算线，
    # 区内任何位置标注都会压在数据上。
    for b in (10**19, 10**22, 10**24):
        ax.axhline(b, color=S.C["ref"], ls="--", lw=0.9, zorder=1)
        ax.annotate(budget_label(b), xy=(1.005, b), xycoords=("axes fraction", "data"),
                    xytext=(3, 0), textcoords="offset points",
                    fontsize=7.8, color=S.C["ref"], va="center", ha="left",
                    annotation_clip=False)

    ax.set_yscale("log")
    ax.set_xlim(q0 - 0.01, 1.02)
    ax.set_xlabel("目标质量 $Q$")
    ax.set_ylabel("增量算力 $C_Q$（FLOPs，对数轴）")
    S.legend_below(ax, ncol=2, y=-0.30, fontsize=8.2)
    S.save(fig, "q3-cost-curve")


# ----------------------------------------------------------------------
def fig_conditional():
    """条件 Q-only 模型：首次触发 N/D 重分配的质量响应阈值 β_Q。

    45 个情景里，10²⁴ 档的 15 个不存在重分配（所有网格点都可行、最优解钉在网格
    上界），故只画 10¹⁹ 与 10²² 两档；阈值跨两个数量级，正是"成本函数选择影响
    最优解"的定量体现。
    """
    df = pd.read_csv(OUT / "Q_conditional_Q_break_even_summary.csv")
    df = df.copy()
    df["th"] = pd.to_numeric(df["beta_Q_first_ND_reallocation"], errors="coerce")
    df["B"] = df["budget_flops"].astype(float)
    df["L"] = df["context_length_tokens"].astype(int)
    assert len(df) == 45
    n_none = int(df["th"].isna().sum())
    assert n_none == 15, f"无重分配情景应为 15 个，实际 {n_none}"

    budgets = [b for b in sorted(df["B"].unique()) if df[df.B == b]["th"].notna().any()]
    ctxs = sorted(df["L"].unique())
    bcol = {b: c for b, c in zip(budgets, [S.C["ours"], S.C["base"], S.C["alt"]])}

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    for ax, g in zip(axes, ["exponential", "power4", "logarithmic"]):
        S.style_axes(ax)
        sub = df[(df.g_function == g) & df["th"].notna()]
        for b in budgets:
            s = sub[sub.B == b].sort_values("L")
            if not len(s):
                continue
            ax.plot(s["L"], s["th"], marker="o", ms=5, lw=1.4, color=bcol[b],
                    label=budget_label(b), zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(ctxs, [str(c) for c in ctxs], fontsize=8)
        ax.set_xlabel("$L_{\\mathrm{ctx}}$")
        ax.set_title(G_LABEL[g].split("$")[0].strip(), fontsize=9)
    axes[0].set_ylabel("首次 N/D 重分配的 $\\beta_Q$ 阈值\n（对数轴）")
    axes[0].legend(loc="lower left", fontsize=8.2, title="算力预算",
                   title_fontsize=8.2)
    # 三个面板里曲线占满全高，区内没有可靠的空位，故把注记放到坐标区外
    fig.text(0.5, -0.02, f"$10^{{24}}$ 档的 {n_none} 个情景不存在 N/D 重分配，"
                         f"未在本图绘出", ha="center", va="top", fontsize=8,
             color=S.C["ref"])
    S.save(fig, "q3-conditional")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("问题三 · 算力约束与资源配置：")
    fig_c7_lengths()
    fig_frontier()
    fig_cost_curve()
    fig_conditional()
    print("完成。")
