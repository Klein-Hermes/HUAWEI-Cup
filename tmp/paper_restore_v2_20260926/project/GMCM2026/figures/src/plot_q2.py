# -*- coding: utf-8 -*-
"""
问题二 · 经典标度律 M0 的边际效应与弹性（2 张图）

  1. q2-m0-marginal    双面板：|∂L/∂N| 随规模、|∂L/∂D| 随训练量
  2. q2-m0-elasticity  双面板：ε_N、ε_D 随模型规模（含轨迹内 p10–p90 跨度）
  3. q2-b1-residual    B1 拟合残差结构（R² 高，但必须给出拟合诊断）
  4. q2-b1-support     B1 的设计支撑：8 个 N × 共享 147 个 D 检查点
  5. q2-q-sign         B6/B7 与 B8 在固定 N、D 内的 Loss–Q 斜率符号相反

数据源（全部为已冻结的 Q2 产物，本脚本只读取、不重拟合、不重抽样）：
  Q2/03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_by_checkpoint.csv  (1176 行)
  Q2/03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_by_size.csv        (8 行)

设计说明：队员已出过同一组图（英文标签 + 图内标题），本脚本用同一份数据重绘；
原为 4 张单面板，其中两两画的是同一件事的两个缩放，已合并为 2 张双面板。
风格：中文字体、无图内标题、Okabe-Ito 语义色。
被画的量与区间全部直接读自 CSV，没有任何硬编码数值。

运行：python3 plot_q2.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style as S

S.setup()

# 结果根目录由 style.DATA_ROOT 推出，换机器时只需改 style.py 或设 Q1_DATA_ROOT
REPO = S.DATA_ROOT.parent
Q2DIR = REPO / "Q2" / "03_结果" / "经典ScalingLaw基线" / "v1"

SIZE_ORDER = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]


def _load():
    ck = pd.read_csv(Q2DIR / "q2_m0_marginal_effects_by_checkpoint.csv")
    sz = pd.read_csv(Q2DIR / "q2_m0_marginal_effects_by_size.csv")
    # 按参数规模排序，保证面板顺序与正文一致
    key = {s: i for i, s in enumerate(SIZE_ORDER)}
    ck = ck.sort_values("N_params_B").reset_index(drop=True)
    sz = sz.sort_values("N_params_B").reset_index(drop=True)
    assert len(ck) == 1176, f"检查点行数应为 1176，实际 {len(ck)}"
    assert len(sz) == 8, f"规模行数应为 8，实际 {len(sz)}"
    assert set(sz["B12_model_size"]) == set(SIZE_ORDER), "规模标签与预期不符"
    return ck, sz, key


# ----------------------------------------------------------------------
def fig_marginal(ck, sz):
    """双面板：参数量边际效应 |∂L/∂N| 与数据量边际效应 |∂L/∂D|。

    两张原图（q2-m0-effect-n / -d）画的是同一件事的两个缩放，合并后既省版面，
    也让"区间窄到看不见"这件事只需说明一次。
    """
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                  gridspec_kw=dict(wspace=0.26))
    S.style_axes(ax); S.style_axes(ax2)

    # (a) ∂L/∂N 只依赖 N，故每个规模取轨迹中位数，区间取该中位数的 bootstrap 区间
    x = sz["N_params_B"].values
    y = -sz["dLoss_dN_per_B_params_median"].values
    lo = -sz["dLoss_dN_per_B_params_median_upper_95"].values
    hi = -sz["dLoss_dN_per_B_params_median_lower_95"].values
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", ms=6, color=S.C["ours"],
                ecolor=S.C["ours"], elinewidth=1.2, capsize=2.6, capthick=1.0,
                zorder=3)
    for xi, yi, s in zip(x, y, sz["B12_model_size"]):
        ax.annotate(s, (xi, yi), textcoords="offset points", xytext=(6, 5),
                    fontsize=8, color=S.C["ours"])
    ax.set_xlabel("参数量 $N$（十亿参数，对数轴）")
    ax.set_ylabel("$|\\partial L/\\partial N|$（每十亿参数）")
    S.panel_label(ax, "a")

    # (b) ∂L/∂D 只依赖 D，八条轨迹在同一 D 上取值相同，故是一条曲线
    d = ck.groupby("D_tokens_B", as_index=False).agg(
        y=("dLoss_dD_per_B_tokens", "median"),
        lo=("dLoss_dD_per_B_tokens_lower_95", "min"),
        hi=("dLoss_dD_per_B_tokens_upper_95", "max"),
    ).sort_values("D_tokens_B")
    xd = d["D_tokens_B"].values
    yd = -d["y"].values
    lod, hid = -d["hi"].values, -d["lo"].values
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.plot(xd, yd, color=S.C["ours"], lw=1.5, zorder=3)
    # 区间宽度先量后画：窄于坐标跨度 0.4% 的带在纸上看不见，只标注不填充
    rel = float(np.max((hid - lod) / yd))
    if rel > 0.004:
        ax2.fill_between(xd, lod, hid, color=S.C["ours"], alpha=0.18, lw=0)
        note = "95% 整轨迹 bootstrap 区间"
    else:
        note = f"95% 区间半宽最大 {rel/2*100:.3f}%，窄于线宽，未单独绘出"
    ax2.set_xlabel("训练数据量 $D$（十亿 token，对数轴）")
    ax2.set_ylabel("$|\\partial L/\\partial D|$（每十亿 token）")
    ax2.text(0.03, 0.06, note, transform=ax2.transAxes, fontsize=8,
             color=S.C["ref"], va="bottom", ha="left")
    S.panel_label(ax2, "b")
    S.save(fig, "q2-m0-marginal")


# ----------------------------------------------------------------------
def fig_elasticity(ck, sz):
    """2×2：上排是全尺度的 ε_N、ε_D；下排把 95% 区间单独放大看。

    95% 整轨迹 bootstrap 区间只有纵轴跨度的 0.01%–0.14%，画在全尺度图里必然
    窄于线宽、读者根本看不见——与其在图例里挂一个看不见的区间，不如另给一排
    放大图：把每点的区间以【它自己的中位数】为零点展开，各点的区间宽度就可比了。
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2),
                             gridspec_kw=dict(hspace=0.42, wspace=0.24))
    (ax, ax2), (ax3, ax4) = axes

    spec = [("elasticity_N", "$N$ 弹性 $\\varepsilon_N$"),
            ("elasticity_D", "$D$ 弹性 $\\varepsilon_D$")]
    x = sz["N_params_B"].values

    # ---- 上排：全尺度 ----
    for a, (k, ylab) in zip((ax, ax2), spec):
        S.style_axes(a)
        y = sz[f"{k}_median"].values
        # 轨迹内 p10–p90 跨度：同一规模内不同检查点之间的离散程度
        for xi, a_, b_ in zip(x, sz[f"{k}_checkpoint_p10"].values,
                              sz[f"{k}_checkpoint_p90"].values):
            a.plot([xi, xi], [a_, b_], color=S.C["ref"], lw=3.4, alpha=0.38,
                   solid_capstyle="round", zorder=1)
        a.plot(x, y, "o", ms=6, color=S.C["ours"], zorder=3)
        a.set_xscale("log")
        a.set_xlabel("参数量 $N$（十亿参数，对数轴）")
        a.set_ylabel(ylab + "（无量纲）")
        S.ref_line(a, 0, axis="y", color=S.C["ref"])
        # 区间有多窄，就在图上如实说出来，而不是靠读者猜。
        # 放在左上角：两个弹性的点云都是从左上向右下走，左上角是空区。
        span = y.max() - y.min()
        wid = (sz[f"{k}_median_upper_95"] - sz[f"{k}_median_lower_95"]).max()
        a.text(0.03, 0.95,
               f"95% 区间宽 ≤{wid/span*100:.2f}% 纵轴跨度，窄于线宽，见下方放大图",
               transform=a.transAxes, fontsize=7.8, color=S.C["ref"],
               va="top", ha="left")

    # ---- 下排：以各自中位数为零点，把 95% 区间放大 ----
    for a, (k, ylab) in zip((ax3, ax4), spec):
        S.style_axes(a)
        med = sz[f"{k}_median"].values
        lo = (sz[f"{k}_median_lower_95"].values - med) * 1e5
        hi = (sz[f"{k}_median_upper_95"].values - med) * 1e5
        for xi, l_, h_ in zip(x, lo, hi):
            a.plot([xi, xi], [l_, h_], color=S.C["ours"], lw=2.4,
                   solid_capstyle="round", zorder=3)
        a.plot(x, np.zeros_like(x), "o", ms=5.5, color=S.C["ours"], zorder=4)
        a.set_xscale("log")
        a.set_xlabel("参数量 $N$（十亿参数，对数轴）")
        a.set_ylabel("相对自身中位数的偏差\n（$10^{-5}$，无量纲）")
        S.ref_line(a, 0, axis="y", color=S.C["ref"], ls="--")

    S.panel_label(ax, "a"); S.panel_label(ax2, "b")
    S.panel_label(ax3, "c"); S.panel_label(ax4, "d")
    ax4.legend(handles=[
        Line2D([], [], marker="o", ls="", ms=5.5, color=S.C["ours"],
               label="轨迹中位数"),
        Line2D([], [], color=S.C["ours"], lw=2.4, label="95% 整轨迹 bootstrap 区间"),
        Line2D([], [], color=S.C["ref"], lw=3.4, alpha=0.38,
               label="轨迹内 p10–p90 跨度（仅上排）"),
    ], loc="upper center", bbox_to_anchor=(1.12, -0.52), ncol=3, fontsize=8.2,
        frameon=False, handlelength=1.6, columnspacing=1.2)
    S.save(fig, "q2-m0-elasticity")


# ----------------------------------------------------------------------
def fig_residual():
    """B1 拟合残差结构：R² 高到 0.99999982，但必须让读者看到这张图。

    原图（队员）用 viridis 给点上色却没有 colorbar，颜色不可解码；这里去掉那层
    编码，只留残差本身，并把偏倚与标准差标出来。
    """
    pr = pd.read_csv(Q2DIR / "b1_fitted_predictions.csv",
                     usecols=["predicted_val_loss", "residual_pred_minus_obs"])
    fit = pr["predicted_val_loss"].values
    res = pr["residual_pred_minus_obs"].values
    assert len(res) == 1176

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    S.style_axes(ax)
    ax.scatter(fit, res, s=9, color=S.C["ours"], alpha=0.35,
               edgecolors="none", zorder=3)
    S.ref_line(ax, 0, axis="y", color=S.C["ref"], ls="--", label="零残差")
    mu, sd = float(res.mean()), float(res.std(ddof=1))
    ax.set_xlabel("拟合验证损失（B1，$n=1176$）")
    ax.set_ylabel("残差（预测 $-$ 观测）")
    ax.text(0.985, 0.94,
            f"偏倚 ${mu:+.2e}$　标准差 ${sd:.2e}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            color=S.C["ref"])
    ax.legend(loc="lower right", fontsize=8.6)
    S.save(fig, "q2-b1-residual")


# ----------------------------------------------------------------------
def fig_support():
    """B1 的设计支撑：8 个参数量 × 共享的 147 个训练检查点。

    这张图回答"拟合到底站在什么数据上"：D 网格在各 N 列之间水平对齐，说明设计是
    交叉的——这正是 α 与 β 可分别识别、条件数只有 56.378 的原因；同时它把 B10
    外推所依赖的支持边界（N ≤ 11.965825B、D ≤ 299.893B）画成了可见的范围。
    """
    pr = pd.read_csv(Q2DIR / "b1_fitted_predictions.csv",
                     usecols=["N_params_B", "D_tokens_B", "observed_val_loss"])
    assert len(pr) == 1176
    assert pr["N_params_B"].nunique() == 8 and pr["D_tokens_B"].nunique() == 147

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    S.style_axes(ax)
    sc = ax.scatter(pr["N_params_B"], pr["D_tokens_B"], c=pr["observed_val_loss"],
                    cmap="viridis", s=16, marker="s", edgecolors="none",
                    zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("参数量 $N$（十亿参数，对数轴）")
    ax.set_ylabel("训练数据量 $D$（十亿 token，对数轴）")
    cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02, shrink=0.88)
    cb.set_label("观测验证损失", fontsize=8.5)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=8)
    # 设计点计数不写在区内——网格铺满整个坐标区，区内任何位置都会压到数据上，
    # 该信息改由 LaTeX 题注承担。
    S.save(fig, "q2-b1-support")


# ----------------------------------------------------------------------
def fig_q_sign():
    """B6/B7 与 B8 在固定 N、D 单元内的 Loss–Q 斜率符号完全相反。

    这是正文只能用文字断言的那条结论的图证：两个半合成来源的 90 个匹配单元斜率
    全为负，而 B8 的 150 个（校准 + 外推）全为正。
    """
    df = pd.read_csv(REPO / "Q2" / "03_结果" / "完整回答补充" / "v1" /
                     "b6_b8_q_direction_audit.csv")
    label = {"B6": "B6（半合成）", "B7": "B7（半合成）"}
    dtype = {"calibrated": "校准", "extrapolated": "外推"}
    rows = []
    for _, r in df.iterrows():
        # 没有映射就退回原始英文类型名，会漏出 "B8（calibrated）" 这种半中半英，
        # 故对每个出现的类型都显式给中文名。
        nm = label.get(r["dataset"],
                       f"{r['dataset']}（{dtype.get(r['data_type'], r['data_type'])}）")
        rows.append((nm, int(r["matched_slope_negative"]),
                     int(r["matched_slope_positive"]),
                     float(r["median_matched_loss_slope_per_Q"])))

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    S.style_axes(ax, grid_axis="y")
    y = np.arange(len(rows))[::-1]
    neg = np.array([r[1] for r in rows], dtype=float)
    pos = np.array([r[2] for r in rows], dtype=float)
    tot = neg + pos
    ax.barh(y, -neg / tot * 100, color=S.C["ours"], height=0.55,
            edgecolor="white", lw=0.8, zorder=3, label="斜率为负（Loss 随 $Q$ 下降）")
    ax.barh(y, pos / tot * 100, color=S.C["base"], height=0.55,
            edgecolor="white", lw=0.8, zorder=3, label="斜率为正（Loss 随 $Q$ 上升）")
    for yi, (nm, n, p, med) in zip(y, rows):
        ax.text(-(n / (n + p) * 100) - 1.5, yi, f"{n}/{n+p}", ha="right",
                va="center", fontsize=8.4, color=S.C["ours"])
        ax.text(p / (n + p) * 100 + 1.5, yi, f"{p}/{n+p}", ha="left",
                va="center", fontsize=8.4, color=S.C["base"])
    ax.axvline(0, color=S.C["ref"], lw=0.9, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8.6)
    ax.set_xlim(-118, 118)
    ax.set_xticks([-100, -50, 0, 50, 100], ["100", "50", "0", "50", "100"])
    ax.set_xlabel("匹配 N–D 单元中斜率符号的占比（%）")
    S.legend_below(ax, ncol=2, y=-0.30, fontsize=8.4)
    S.save(fig, "q2-q-sign")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ck, sz, _ = _load()
    print("问题二 · M0 边际效应与弹性：")
    fig_marginal(ck, sz)
    fig_elasticity(ck, sz)
    fig_residual()
    fig_support()
    fig_q_sign()
    print("完成。")
