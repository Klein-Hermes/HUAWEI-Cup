# -*- coding: utf-8 -*-
"""
问题一 · 配比—损失建模部分（5 张图）

  1. q1-share-loss-corr   配比与损失目标之间的相关系数结构（发散色标热图）
  2. q1-model-selection   三种配比模型族的分组交叉验证误差（哑铃图）
  3. q1-holdout-1m        同尺度留出集上预测损失与观测损失的一致性（散点 + 残差）
  4. q1-cross-scale       检验尺度对预测误差与秩相关的影响（双轴）
  5. q1-substitution      配比替代效应的关联系数结构（发散色标热图）

运行：python3 plot_q1_3.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

S.setup()

Q13 = S.res("q1_3", "v1")
MIX = S.res("..")            # 占位，实际用下面的 A_ROOT
A_ROOT = ("/Users/zhaomengchen/Downloads/第二十三届中国研究生数学建模竞赛 "
          "- 中文题目/中文题目/F题/real_attachments/A_data_value")
MIXDIR = f"{A_ROOT}/regmix_tables"


# ----------------------------------------------------------------------
def fig_share_loss_corr():
    """配比与损失目标之间的相关系数结构"""
    mx = pd.read_csv(f"{MIXDIR}/train_mixture_1m.csv")
    ls = pd.read_csv(f"{MIXDIR}/train_pile_loss_1m.csv")
    mx = mx.set_index("index"); ls = ls.set_index("index")
    X = mx[[c for c in mx.columns if c.startswith("train_the_pile_")]]
    D = [c.replace("train_the_pile_", "") for c in X.columns]
    Y = ls[[c for c in ls.columns if c.startswith("metric/")]]
    T = [S.short_domain(c) for c in Y.columns]

    M = np.full((len(D), len(T)), np.nan)
    for i, dc in enumerate(X.columns):
        for j, tc in enumerate(Y.columns):
            v = X[dc].values.astype(float); w = Y[tc].values.astype(float)
            if v.std() > 0 and w.std() > 0:
                M[i, j] = np.corrcoef(v, w)[0, 1]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(T)), T, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(D)), D, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.86)
    cb.set_label("Pearson $r$", fontsize=8.5)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=8)
    # 13 个损失目标的域与配比域有 4 个不重合，标出无损失的分量
    miss = [d for d in D if d not in T]
    if miss:
        for d in miss:
            ax.get_yticklabels()[D.index(d)].set_color(S.C["warn"])
    S.save(fig, "q1-share-loss-corr")


# ----------------------------------------------------------------------
def fig_model_selection():
    """三种配比模型族的分组交叉验证误差（哑铃图，以下界为参照）"""
    df = pd.read_csv(Q13 / "full_training_model_selection.csv")
    name = {
        "mean": "训练均值空模型",
        "reference_ols": "参考分量 OLS",
        "simplex_ridge": "单纯形岭回归（主模型）",
        "simplex_huber_ridge_sensitivity": "Huber 岭回归",
        "quadratic_ridge_sensitivity": "二次面（153 项）",
    }
    df["label"] = df["candidate_model"].map(name).fillna(df["candidate_model"])
    df = df.sort_values("grouped_cv_standardized_mse", ascending=False)

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    S.style_axes(ax, grid_axis="x")
    base = df["grouped_cv_standardized_mse"].max()
    lo = df["grouped_cv_standardized_mse"].min()
    # 数值统一排成右侧独立数值列——放在标记旁必被哑铃连线穿过
    xcol = lo - (base - lo) * 0.62
    for i, (_, r) in enumerate(df.iterrows()):
        sel = bool(r["selected_on_full_training_cv"])
        col = S.C["ours"] if sel else S.C["ref"]
        # 这条连线只表示"该候选离空模型基线有多远"，不是置信区间——
        # 选型的报表里没有折间离散度，所以画成点线并进图例，避免被读成误差棒。
        ax.plot([r["grouped_cv_standardized_mse"], base], [i, i],
                color=S.C["ref"], lw=1.1, alpha=0.6, ls=(0, (1.5, 1.8)), zorder=1,
                label="至训练均值基线的距离" if i == 0 else None)
        ax.scatter(r["grouped_cv_standardized_mse"], i, s=58, color=col,
                   zorder=3, edgecolors="white", lw=0.7)
        ax.text(xcol, i, f'{r["grouped_cv_standardized_mse"]:.5f}'
                + ("　← 选中" if sel else ""),
                fontsize=8.4, va="center", ha="left", color=col)
    ax.set_yticks(range(len(df)), df["label"], fontsize=8.8)
    ax.set_xlabel("分组交叉验证的标准化 MSE")
    ax.set_xlim(xcol - (base - lo) * 0.04, base * 1.04)
    ax.legend(loc="lower right", fontsize=8.4)
    S.save(fig, "q1-model-selection")


# ----------------------------------------------------------------------
def fig_holdout_1m():
    """同尺度留出集上预测损失与观测损失的一致性（散点 + 残差）"""
    cols = ["split", "variant", "model", "target",
            "observed_or_estimated_loss", "predicted_loss",
            "error_pred_minus_observed_or_estimated"]
    pr = pd.read_csv(Q13 / "holdout_predictions.csv", usecols=cols)
    sub = pr[(pr.split == "test_1m") & (pr.variant == "closed")
             & (pr.model == "simplex_ridge")].copy()

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(width_ratios=[1.35, 1],
                                                   wspace=0.22))
    for a in (ax, ax2):
        S.style_axes(a)

    obs = sub["observed_or_estimated_loss"].values
    pred = sub["predicted_loss"].values
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    S.band(ax, lo, hi, axis="y", color=S.C["ours"], alpha=0.0)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.1, color=S.C["ref"],
            zorder=1, label="理想一致线 $y=x$")
    # 按目标着色（13 个目标）
    for k, t in enumerate(sorted(sub["target"].unique())):
        m = sub["target"] == t
        ax.scatter(obs[m.values], pred[m.values], s=13, alpha=0.55,
                   color=S.DOMAIN_COLORS[k % len(S.DOMAIN_COLORS)],
                   edgecolors="none", zorder=2)
    ax.set_xlabel("观测损失")
    ax.set_ylabel("预测损失")
    ax.set_xlim(lo - .2, hi + .2); ax.set_ylim(lo - .2, hi + .2)
    ax.legend(loc="upper left", fontsize=8.4)
    S.panel_label(ax, "a")

    err = sub["error_pred_minus_observed_or_estimated"].values
    mu, sd = err.mean(), err.std()
    ax2.hist(err, bins=46, color=S.C["ours"], alpha=0.82, edgecolor="white", lw=0.3)
    S.ref_line(ax2, 0, axis="x", color=S.C["ref"], label="零残差")
    ax2.axvline(mu + 1.96 * sd, color=S.C["warn"], lw=1.0, ls=":")
    ax2.axvline(mu - 1.96 * sd, color=S.C["warn"], lw=1.0, ls=":",
                label=r"$\pm1.96\sigma$")
    ax2.set_xlabel("预测 − 观测（残差）")
    ax2.set_ylabel("样本数")
    ax2.legend(loc="upper right", fontsize=8.2)
    S.panel_label(ax2, "b")
    ax2.set_xlabel(f"预测 − 观测（残差）\n偏倚 {mu:+.4f}　标准差 {sd:.4f}")
    S.save(fig, "q1-holdout-1m")


# ----------------------------------------------------------------------
def fig_cross_scale():
    """检验尺度对预测误差与秩相关的影响（双轴）"""
    hm = pd.read_csv(Q13 / "holdout_metrics.csv")
    m = (hm[(hm.model == "simplex_ridge") & (hm.variant == "closed")]
         .groupby("split")[["rmse", "r2", "spearman_rho"]].mean())
    order = ["test_1m", "test_60m", "test_1b"]
    lab = {"test_1m": "1M\n（同尺度）", "test_60m": "60M\n（跨尺度）",
           "test_1b": "1B\n（跨尺度）"}
    spl = [s for s in order if s in m.index]
    x = np.arange(len(spl))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    S.style_axes(ax)
    ax.bar(x - 0.17, m.loc[spl, "rmse"], width=0.32, color=S.C["base"],
           alpha=0.9, label="宏平均 RMSE")
    for i, v in enumerate(m.loc[spl, "rmse"]):
        ax.text(i - 0.17, v + 0.06, f"{v:.3f}", ha="center", fontsize=8.4,
                color=S.C["base"])
    ax.set_ylabel("宏平均 RMSE", color=S.C["base"])
    ax.tick_params(axis="y", labelcolor=S.C["base"])
    ax.set_xticks(x, [lab[s] for s in spl])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x + 0.17, m.loc[spl, "spearman_rho"], marker="D", ms=7,
             color=S.C["ours"], lw=1.8, zorder=5, label="宏平均 Spearman")
    for i, v in enumerate(m.loc[spl, "spearman_rho"]):
        ax2.text(i + 0.17, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.4,
                 color=S.C["ours"])
    ax2.set_ylabel("宏平均 Spearman $\\rho$", color=S.C["ours"])
    ax2.tick_params(axis="y", labelcolor=S.C["ours"])
    ax2.set_ylim(0.55, 0.95)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.6)
    S.save(fig, "q1-cross-scale")


# ----------------------------------------------------------------------
def fig_substitution():
    """配比替代效应的关联系数结构（发散色标，零为中心）"""
    df = pd.read_csv(Q13 / "parameters_selected_closed.csv")
    coef_cols = [c for c in df.columns if c.startswith("coef::")]
    D = [S.short_domain(c.replace("coef::train_the_pile_", "")) for c in coef_cols]
    T = [S.short_domain(t) for t in df["target"]]
    M = df[coef_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    lim = np.nanpercentile(np.abs(M), 98)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(D)), D, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(T)), T, fontsize=8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02, shrink=0.86)
    cb.set_label("替代关联系数 $\\beta_{ij}$\n（每单位配比份额的损失变化）", fontsize=8.5)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=8)
    S.save(fig, "q1-substitution")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("问题一 · 配比—损失建模：")
    fig_share_loss_corr()
    fig_model_selection()
    fig_holdout_1m()
    fig_cross_scale()
    fig_substitution()
    print("完成。")
