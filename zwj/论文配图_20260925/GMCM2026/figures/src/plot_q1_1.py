# -*- coding: utf-8 -*-
"""
问题一 · 质量评分部分（5 张图）

  1. q1-indicator-dist      22 个质量指标经方向统一后的分布
  2. q1-weight-calibration  指标权重的两个构成分量：完整度与排序稳定性
  3. q1-domain-scores       各来源域的域级质量评分及估计区间
  4. q1-domain-transfer     同域扩展集相对抽样集的评分偏移
  5. q1-rank-robustness     域序对预处理场景的稳健性

运行：python3 plot_q1_1.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import style as S

S.setup()

PRE = S.res("q1_common_preprocess", "v1")
Q11 = S.res("q1_1", "v1")

SCEN_LABEL = {
    "weight_alpha_1":                 "权重指数 α=1（主口径）",
    "weight_alpha_0.5":               "权重指数 α=0.5",
    "weight_alpha_0.25":              "权重指数 α=0.25",
    "weighted_arithmetic_mean":       "加权算术平均",
    "huber_delta_0.5x":               "Huber 阈值 ×0.5",
    "huber_delta_2x":                 "Huber 阈值 ×2",
    "exclude_7_unit_ambiguous_fields":"剔除 7 个单位存疑字段",
    "unit_flagged_a1_excluded":       "剔除单位存疑的 A1 行",
    "unicode_strict_clean_a1":        "窄口径清洁 A1",
    "unicode_broad_clean_a1":         "宽口径清洁 A1",
}


# ----------------------------------------------------------------------
def fig_indicator_dist():
    """22 个质量指标经方向统一后的分布（横向箱线，按中位数排序）"""
    usecols = ["source_domain"] + [c for c in
               pd.read_csv(PRE / "a1_preprocessed.csv.gz", nrows=1).columns
               if c.startswith("norm_")]
    df = pd.read_csv(PRE / "a1_preprocessed.csv.gz", usecols=usecols)

    # 从冻结的校准文件读取方向与单位存疑字段
    with open(PRE / "calibration_a1.json", encoding="utf-8") as f:
        cal = json.load(f)
    fields = cal.get("fields", cal)
    direction = {k: v.get("direction", "?") for k, v in fields.items()
                 if isinstance(v, dict)}
    unit_suspect = set(cal.get("unit_suspect_fields", []))

    cols = sorted([c for c in df.columns if c.startswith("norm_")],
                  key=lambda c: df[c].median())
    labels = [c.replace("norm_", "") for c in cols]
    data = [df[c].dropna().values for c in cols]

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    S.style_axes(ax, grid_axis="x")
    bp = ax.boxplot(data, orientation="horizontal", widths=0.62, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(color="white", lw=1.3),
                    whiskerprops=dict(color=S.C["ref"], lw=0.9),
                    capprops=dict(color=S.C["ref"], lw=0.9))
    for i, c in enumerate(cols):
        # 校准文件按【原始字段名】索引（fineweb_edu），而 cols 是 norm_* 列名；
        # 不剥前缀就会全部落到默认值 positive，22 个箱体同色、图例形同虚设。
        d = direction.get(c[len("norm_"):], "positive")
        face = {"positive": S.C["ours"], "negative": S.C["base"],
                "central": S.C["alt"]}.get(d, S.C["ref"])
        bp["boxes"][i].set(facecolor=face, edgecolor="white", lw=0.6, alpha=0.92)

    ax.set_yticks(range(1, len(cols) + 1), labels)
    ax.tick_params(axis="y", labelsize=8)
    # 单位存疑字段用星号标出
    for i, c in enumerate(cols):
        if c.replace("norm_", "") in unit_suspect:
            ax.get_yticklabels()[i].set_color(S.C["warn"])
    S.ref_line(ax, 0.5, axis="x", color=S.C["ref"])
    ax.set_xlabel("方向统一后的无量纲得分 $u_j$")
    ax.set_xlim(-0.02, 1.02)
    # 图例置于坐标区下方——放在区内必然压住箱线
    ax.legend(handles=[
        Patch(facecolor=S.C["ours"], label="正向指标"),
        Patch(facecolor=S.C["base"], label="负向指标（已补转换）"),
        Patch(facecolor=S.C["alt"],  label="中心型指标"),
        Patch(facecolor=S.C["warn"], label="单位来源未核实（7 个字段，标签同色）"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2,
       fontsize=8.6, frameon=False, handlelength=1.4)
    S.plot_heading(fig, "22 项质量指标的方向统一分布",
                   "箱体为四分位范围，须线覆盖第 5–95 百分位；红色名称标出单位来源待核字段。")
    S.save(fig, "q1-indicator-dist")


# ----------------------------------------------------------------------
def fig_weight_calibration():
    """指标权重的两个构成分量：完整度与排序稳定性（哑铃图）"""
    df = pd.read_csv(Q11 / "indicator_weights.csv")
    df = df.sort_values("weight", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 5.6), sharey=True,
        gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.08))
    for a in (ax, ax2):
        S.style_axes(a, grid_axis="x")

    # 左：完整度与稳定性的哑铃
    for i, r in df.iterrows():
        lo, hi = sorted([r["a1_completeness"], r["domain_rank_stability"]])
        ax.plot([lo, hi], [i, i], color=S.C["ref"], lw=1.2, zorder=1)
    ax.scatter(df["a1_completeness"], y, s=34, color=S.C["ours"],
               zorder=3, label="完整度 $c_j$", edgecolors="white", lw=0.5)
    ax.scatter(df["domain_rank_stability"], y, s=34, color=S.C["base"],
               zorder=3, marker="D", label="排序稳定性 $S_j$",
               edgecolors="white", lw=0.5)
    ax.set_yticks(y, df["indicator"], fontsize=8)
    ax.set_xlabel("分量取值")
    ax.set_xlim(0.955, 1.004)
    S.ref_line(ax, 1.0, axis="x", label="分量上界", color=S.C["ref"])
    S.legend_below(ax, ncol=2, y=-0.16, fontsize=8.6)

    # 右：合成权重（窄区间——正是正文"权重密集"论点的证据）
    ax2.barh(y, df["weight"], height=0.62, color=S.C["ours"], alpha=0.9)
    ax2.set_xlabel("合成权重 $w_j$")
    lo, hi = df["weight"].min(), df["weight"].max()
    S.band(ax2, lo, hi, axis="x", color=S.C["hl"], alpha=0.16)
    ax2.set_xlim(0, hi * 1.60)
    ax2.set_ylim(-0.7, len(df) - 0.3)
    S.plot_heading(fig, "质量指标权重的构成与最终分配",
                   "左图对比完整度与排序稳定性，右图给出合成权重；黄色带标示权重取值范围。")
    S.save(fig, "q1-weight-calibration")


# ----------------------------------------------------------------------
def fig_domain_scores():
    """各来源域的域级质量评分及估计区间（误差棒 + 95% CI）"""
    sc = pd.read_csv(Q11 / "domain_summary.csv")
    ci = pd.read_csv(Q11 / "domain_confidence_intervals.csv")
    m = sc.merge(ci, on=["dataset", "source_domain", "candidate"])

    order = ["a1"] * 7 + ["a2", "a3"]
    cands = ["q_equal", "q_huber"]
    cand_lab = {"q_equal": "等权候选", "q_huber": "加权 Huber 候选"}
    cand_col = {"q_equal": S.C["ours"], "q_huber": S.C["base"]}

    # 按 A1 的等权分排序
    a1 = (m[(m.dataset == "a1") & (m.candidate == "q_equal")]
          .sort_values("domain_score", ascending=True))
    doms = list(a1["source_domain"])
    rows = [(d, c) for d in doms for c in cands]
    # 扩展集追加在末尾
    for d, c in [("arxiv", c) for c in cands] + [("github", c) for c in cands]:
        pass

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    S.style_axes(ax, grid_axis="x")

    ypos, ylab, ycol = [], [], []
    k = 0
    seen = set()          # 每种候选只给第一个点挂 label，避免图例里 9 个重复项
    for d in doms:
        for c in cands:
            r = m[(m.dataset == "a1") & (m.source_domain == d) & (m.candidate == c)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            ax.errorbar(r["domain_score"], k,
                        xerr=[[r["domain_score"] - r["ci_low"]],
                              [r["ci_high"] - r["domain_score"]]],
                        fmt="o", ms=5.5, color=cand_col[c], ecolor=cand_col[c],
                        elinewidth=1.2, capsize=2.2, capthick=1.0, zorder=3,
                        label=cand_lab[c] if c not in seen else None)
            seen.add(c)
            ypos.append(k); ylab.append(d if c == "q_equal" else ""); ycol.append(c)
            k += 1
        k += 0.55                                    # 域间留白

    a1_top = k - 0.55                       # A1 分组的顶端（用于画分组线）

    # 扩展集
    for ds, tag in [("a2", "A2"), ("a3", "A3")]:
        for c in cands:
            r = m[(m.dataset == ds) & (m.candidate == c)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            ax.errorbar(r["domain_score"], k,
                        xerr=[[r["domain_score"] - r["ci_low"]],
                              [r["ci_high"] - r["domain_score"]]],
                        fmt="s", ms=5.5, color=cand_col[c], ecolor=cand_col[c],
                        elinewidth=1.2, capsize=2.2, capthick=1.0, zorder=3,
                        label=cand_lab[c] if c not in seen else None)
            seen.add(c)
            ypos.append(k)
            ylab.append(f"{tag}·{r['source_domain']}" if c == "q_equal" else "")
            k += 1
        k += 0.55

    ax.set_yticks(ypos, ylab, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.9, k - 0.1)
    ax.set_xlabel("域级质量评分")
    # 分组分隔线与参考线：都进图例，读者不必回头查题注
    ax.axhline(a1_top + 0.28, color=S.C["ref"], lw=0.8, ls=(0, (5, 3)),
               label="A1 抽样集与扩展集分界")
    S.ref_line(ax, 0.5, axis="x", color=S.C["ref"], label="0.5 参考线")
    S.legend_below(ax, ncol=2, y=-0.17, fontsize=8.6)
    S.plot_heading(fig, "候选评分的域级结果及估计区间",
                   "点和横线表示域级评分与 95% 区间；虚线分隔 A1 抽样集和 A2/A3 扩展集。")
    S.save(fig, "q1-domain-scores")


# ----------------------------------------------------------------------
def fig_domain_transfer():
    """同域扩展集相对抽样集的评分偏移（森林图，区间跨零即无可检出偏移）"""
    df = pd.read_csv(Q11 / "same_domain_differences.csv")
    cand_lab = {"q_equal": "等权候选", "q_huber": "加权 Huber 候选"}
    cand_col = {"q_equal": S.C["ours"], "q_huber": S.C["base"]}

    rows = []
    for ds, tag, dom in [("a2", "A2", "arxiv"), ("a3", "A3", "github")]:
        for c in ["q_equal", "q_huber"]:
            r = df[(df.extension_dataset == ds) & (df.candidate == c)]
            if len(r):
                rows.append((f"{tag}·{dom} − A1·{dom}（{cand_lab[c]}）",
                             r.iloc[0], cand_col[c]))
    rows = rows[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    S.style_axes(ax, grid_axis="x")
    for i, (lab, r, col) in enumerate(rows):
        ax.errorbar(r["difference_extension_minus_a1"], i,
                    xerr=[[r["difference_extension_minus_a1"] - r["ci_low"]],
                          [r["ci_high"] - r["difference_extension_minus_a1"]]],
                    fmt="o", ms=6, color=col, ecolor=col, elinewidth=1.4,
                    capsize=3, capthick=1.1, zorder=3)
        ax.text(r["ci_high"] + 0.00022, i, f'{r["difference_extension_minus_a1"]:+.5f}',
                fontsize=8.2, va="center", color=col)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("域分差值（扩展集 − 抽样集）")
    ax.set_xlim(-0.0052, 0.0036)          # 右侧留白给数值标签
    S.band(ax, -0.0005, 0.0005, axis="x", color=S.C["ref"], alpha=0.10)
    S.ref_line(ax, 0.0, axis="x", label="无偏移（零参考线）", color=S.C["ref"])
    S.legend_below(ax, ncol=1, y=-0.30, fontsize=8.6)
    S.plot_heading(fig, "扩展集相对 A1 的域内评分偏移",
                   "横线为估计区间；区间跨零时没有检出可辨别的评分偏移。")
    S.save(fig, "q1-domain-transfer")


# ----------------------------------------------------------------------
def fig_rank_robustness():
    """域序对预处理场景的稳健性（热图：场景 × 域 的秩位移）"""
    df = pd.read_csv(Q11 / "domain_rank_sensitivity.csv")
    df = df[df.candidate == "q_equal"]
    base = df[df.scenario == "weight_alpha_1"].set_index("source_domain")
    doms = list(base.sort_values("all_a1_domain_score", ascending=False).index)

    scens = [s for s in SCEN_LABEL if s in set(df.scenario) and s != "weight_alpha_1"]
    scens = sorted(scens, key=lambda s: df[df.scenario == s]
                   ["seven_domain_rank_spearman"].iloc[0])
    M = np.full((len(scens), len(doms)), np.nan)
    for i, s in enumerate(scens):
        sub = df[df.scenario == s].set_index("source_domain")
        for j, d in enumerate(doms):
            if d in sub.index:
                M[i, j] = sub.loc[d, "rank_shift"]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    lim = np.nanmax(np.abs(M)) or 1
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(doms)), doms, rotation=32, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(scens)),
                  [f"{SCEN_LABEL[s]}  (ρ={df[df.scenario == s]['seven_domain_rank_spearman'].iloc[0]:.2f})"
                   for s in scens], fontsize=8.2)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02, shrink=0.9)
    cb.set_label("秩位移（正=排名下降）", fontsize=8.5)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=8)
    S.plot_heading(fig, "预处理场景下的域排名稳健性",
                   "热图为相对主口径的排名位移；正值表示名次下降，ρ表示七域排名相关。")
    S.save(fig, "q1-rank-robustness")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("问题一 · 质量评分：")
    fig_indicator_dist()
    fig_weight_calibration()
    fig_domain_scores()
    fig_domain_transfer()
    fig_rank_robustness()
    print("完成。")
