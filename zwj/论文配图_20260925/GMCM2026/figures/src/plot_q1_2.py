# -*- coding: utf-8 -*-
"""
问题一 · 冲突消解部分（3 张图）

  1. q1-conflict-dist      各来源域的域内冲突度分布与高冲突阈值（山脊图）
  2. q1-pair-conflict      质量指标两两冲突强度的分布（Top-20 条形）
  3. q1-conflict-external  冲突指标在扩展集上的复核结果（森林图）

运行：python3 plot_q1_2.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

S.setup()

Q12 = S.res("q1_2_conflict_model", "v1")


# ----------------------------------------------------------------------
def fig_conflict_dist():
    """各来源域的域内冲突度分布与高冲突阈值（山脊图，共享横轴）"""
    cols = ["dataset", "source_domain", "conflict_intensity"]
    df = pd.read_csv(Q12 / "sample_scores.csv.gz", usecols=cols)
    a1 = df[df.dataset == "a1"]
    th = pd.read_csv(Q12 / "domain_summary.csv")
    th = th[th.dataset == "a1"].set_index("source_domain")

    doms = list(th.sort_values("median_conflict_intensity",
                               ascending=False).index)
    grid = np.linspace(0.05, 0.75, 260)

    fig, axes = plt.subplots(len(doms), 1, figsize=(6.8, 5.2), sharex=True)
    plt.subplots_adjust(hspace=-0.42)                 # 山脊图的关键：负间距
    for i, (ax, d) in enumerate(zip(axes, doms)):
        vals = a1.loc[a1.source_domain == d, "conflict_intensity"].dropna().values
        if len(vals) == 0:
            continue
        # 高斯核密度（手写，避免依赖 scipy 版本差异）
        h = 1.06 * vals.std() * len(vals) ** (-1 / 5)
        dens = np.exp(-0.5 * ((grid[:, None] - vals[None, :]) / h) ** 2).sum(1)
        dens = dens / dens.max() * 0.92
        col = S.DOMAIN_COLORS[i % len(S.DOMAIN_COLORS)]
        ax.fill_between(grid, 0, dens, color=col, alpha=0.72, lw=0, zorder=3 - i)
        ax.plot(grid, dens, color="white", lw=0.9, zorder=3 - i)
        # 该域的高冲突阈值
        q = th.loc[d, "p95_conflict_intensity"]
        ax.axvline(q, color=S.C["warn"], lw=1.1, ls="--", zorder=4,
                   label="高冲突阈值 $q_{95}$" if i == 0 else None)
        # 中位数标记
        ax.plot([th.loc[d, "median_conflict_intensity"]] * 2, [0, 0.30],
                color=S.C["ref"], lw=1.0, zorder=5,
                label="域内中位数" if i == 0 else None)
        ax.set_yticks([])
        ax.set_ylim(0, 1.25)
        ax.set_xlim(grid[0], grid[-1])
        ax.grid(False)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.062, 0.62, f"{d}", fontsize=8.8, color=col, va="center")
        ax.text(0.995, 0.62, f"中位 {th.loc[d,'median_conflict_intensity']:.3f}"
                             f"　阈值 {q:.3f}", fontsize=7.6, color=S.C["ref"],
                va="center", ha="right", transform=ax.transAxes)
    axes[-1].set_xlabel("行内平均两两绝对分差 $D$（冲突度）")
    # 参考线必须有图例：否则读者只能从题注反查这两道竖线是什么
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                   fontsize=8.4, frameon=False, handlelength=1.6)
    S.plot_heading(fig, "A1 七个来源域的冲突度分布",
                   "虚线标出各域第 95 百分位高冲突阈值，灰色实线为域内中位数。")
    S.save(fig, "q1-conflict-dist")


# ----------------------------------------------------------------------
def fig_pair_conflict():
    """质量指标两两冲突强度的分布（A1 七域宏平均 Top-20）"""
    df = pd.read_csv(Q12 / "pair_diagnostics.csv")
    a1 = df[df.dataset == "a1"].copy()
    agg = (a1.groupby(["indicator_a", "indicator_b"])
             .agg(mean_abs_gap=("mean_abs_gap", "mean"),
                  pair_n=("pair_n", "sum"),
                  macro_rank=("a1_macro_rank", "first"))
             .reset_index())
    top = agg.nsmallest(20, "macro_rank").sort_values("mean_abs_gap")

    fig, ax = plt.subplots(figsize=(7.0, 7.2))
    # 每行是两行文字，增加纵向间距，避免相邻指标组合的名称互相碰撞。
    fig.subplots_adjust(left=0.28, right=0.98, top=0.98, bottom=0.10)
    S.style_axes(ax, grid_axis="x")
    # 含 dsir × 一元熵 的组合高亮——正文据此归因
    is_dsir_ent = top.apply(
        lambda r: ("dsir" in r["indicator_a"] and "entropy" in r["indicator_b"])
                  or ("dsir" in r["indicator_b"] and "entropy" in r["indicator_a"]),
        axis=1)
    colors = [S.C["hl"] if f else S.C["ours"] for f in is_dsir_ent]
    y = np.arange(len(top))
    ax.barh(y, top["mean_abs_gap"], height=0.66, color=colors, alpha=0.92)
    for i, (v, n) in enumerate(zip(top["mean_abs_gap"], top["pair_n"])):
        ax.text(v + 0.005, i, f"{v:.4f}", fontsize=7.8, va="center",
                color="#404040")
    labs = [f'{S.short_domain(a)}\n× {S.short_domain(b)}'
            for a, b in zip(top["indicator_a"], top["indicator_b"])]
    ax.set_yticks(y, labs, fontsize=7.0)
    ax.set_xlabel("域内平均两两绝对分差（A1 七域宏平均）")
    ax.set_xlim(0, top["mean_abs_gap"].max() * 1.14)
    ax.grid(True, axis="x", **S.GRID)
    S.plot_heading(fig, "A1 中冲突最强的 20 组质量指标配对",
                   "按七域宏平均排序；橙色标出包含 DSIR 与一元熵的配对。")
    S.save(fig, "q1-pair-conflict")


# ----------------------------------------------------------------------
def fig_conflict_external():
    """冲突指标在扩展集上的复核结果（双面板森林图）。

    原先把"中位冲突度之差"与"高冲突率之差"画在同一个数值轴上——两者虽都是
    无量纲比例，却是两个不同的量，放在一根轴上会让读者误以为可以互相比较。
    这里拆成两个面板，各自给轴与量纲。
    """
    df = pd.read_csv(Q12 / "external_validation.csv")

    panels = [
        ("median", "中位冲突度之差", "冲突度之差（无量纲）", S.C["ours"],
         "median_conflict_diff", "median_diff_ci_low", "median_diff_ci_high"),
        ("rate", "高冲突率之差", "高冲突率之差（比例）", S.C["base"],
         "high_conflict_rate_diff", "rate_diff_ci_low", "rate_diff_ci_high"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5),
                             gridspec_kw=dict(wspace=0.30))
    for ax, (_, title, xlabel, col, vc, loc, hic) in zip(axes, panels):
        S.style_axes(ax, grid_axis="x")
        vals = df[vc].values
        y = np.arange(len(vals))[::-1]
        for yi, (_, r) in zip(y, df.iterrows()):
            v, lo, hi = r[vc], r[loc], r[hic]
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", ms=6,
                        color=col, ecolor=col, elinewidth=1.4, capsize=3.2,
                        capthick=1.1, zorder=3)
        ax.set_yticks(y, [f'{r["extension_dataset"].upper()}·{r["source_domain"]}'
                          for _, r in df.iterrows()], fontsize=8.6)
        ax.tick_params(axis="y", length=0)
        S.ref_line(ax, 0.0, axis="x", color=S.C["ref"], ls="--")
        # 两侧留白：数值标签要放到区间【之外】，否则会压在须线上
        m = float(np.max(np.abs(np.concatenate([vals, df[loc], df[hic]]))))
        ax.set_xlim(-m * 2.0, m * 2.0)
        for yi, (_, r) in zip(y, df.iterrows()):
            v, lo, hi = r[vc], r[loc], r[hic]
            right, left = max(hi, v), min(lo, v)
            xpos = right + m * 0.14 if v >= 0 else left - m * 0.14
            ax.text(xpos, yi, f'{v:+.5f}', fontsize=8, va="center",
                    ha="left" if v >= 0 else "right", color=col)
        ax.set_title(title, fontsize=9.2)      # 面板标签，非图标题
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.7, len(vals) - 0.3)
    S.plot_heading(fig, "冲突度指标在扩展集上的外部复核",
                   "点和区间表示差异估计及 95% 区间；区间跨零时未检出差异，左右面板分别为冲突度和高冲突率。")
    S.save(fig, "q1-conflict-external")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("问题一 · 冲突消解：")
    fig_conflict_dist()
    fig_pair_conflict()
    fig_conflict_external()
    print("完成。")
