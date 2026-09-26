# -*- coding: utf-8 -*-
"""
问题二 · A 侧转移对的互补／替代分布（1 张图）。

原图（队友版 `process_q2_p_interval_adjustment`）把 6 个汇总数字画成两根堆叠条，
约 85% 的画布是空白，而且带着图内标题。这张改用底层数据：6551 个转移对的逐对
交互效应本身。

一张图同时传达三件事：
  1. 点估计几乎按符号均分（互补 3310 / 替代 3241）；
  2. 逐点 95% 区间下绝大多数跨零（5833/6551 = 89%）；
  3. 施加 max-|t| 家族校正后，互补方向一对都不剩（0/6551）。

数据源（只读，不重算）：
  Q2/03_结果/完整回答补充/v1/a_side_complementarity/
    a4_a5_transfer_pair_complementarity_bootstrap.csv   (6551 行，逐对交互效应)
    a4_a5_transfer_pair_complementarity_summary.csv      (13 行，按目标的分类计数)

运行：python3 plot_q2_compl.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style as S

S.setup()

REPO = S.DATA_ROOT.parent
COMPL = REPO / "Q2" / "03_结果" / "完整回答补充" / "v1" / "a_side_complementarity"


def fig_compl_dist():
    b = pd.read_csv(COMPL / "a4_a5_transfer_pair_complementarity_bootstrap.csv")
    s = pd.read_csv(COMPL / "a4_a5_transfer_pair_complementarity_summary.csv")
    x = b["interaction_delta_loss"].values.astype(float)
    assert len(x) == 6551, f"应为 6551 个转移对，实际 {len(x)}"

    # 分类计数从汇总表逐目标求和，不硬编码
    n = {
        "逐点全负": int(s.bootstrap_ci_fully_negative_unadjusted.sum()),
        "逐点全正": int(s.bootstrap_ci_fully_positive_unadjusted.sum()),
        "逐点跨零": int(s.bootstrap_ci_cross_zero_unadjusted.sum()),
        "校正后全负": int(s.bootstrap_maxT95_fully_negative.sum()),
        "校正后全正": int(s.bootstrap_maxT95_fully_positive.sum()),
        "校正后跨零": int(s.bootstrap_maxT95_cross_zero.sum()),
    }

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    S.style_axes(ax)

    # 以 0 为中心分色：负侧=互补，正侧=替代。区间取 1–99 百分位，
    # 让长尾不至于把主峰压扁；超出范围的条数在题注中说明。
    lo, hi = np.percentile(x, [1, 99])
    bins = np.linspace(lo, hi, 61)
    neg = np.histogram(x[x < 0], bins=bins)[0]
    pos = np.histogram(x[x > 0], bins=bins)[0]
    ctr = (bins[:-1] + bins[1:]) / 2
    ax.bar(ctr, neg, width=(bins[1] - bins[0]) * 0.92, color=S.C["ours"],
           label=f"互补（$\\Delta_u\\Delta_vL<0$），{int((x<0).sum())} 对")
    ax.bar(ctr, pos, width=(bins[1] - bins[0]) * 0.92, color=S.C["base"],
           label=f"替代（$\\Delta_u\\Delta_vL>0$），{int((x>0).sum())} 对")
    S.ref_line(ax, 0, axis="x", color=S.C["ref"], ls="--")

    # 把两类判据的结果写在图内空白处，指代清楚
    ax.text(0.985, 0.95,
            f"逐点 $95\\%$ 区间：全负 {n['逐点全负']}、全正 {n['逐点全正']}、"
            f"跨零 {n['逐点跨零']}\n"
            f"$\\max|t|$ 校正后：全负 {n['校正后全负']}、全正 {n['校正后全正']}、"
            f"跨零 {n['校正后跨零']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2,
            color=S.C["ref"])
    ax.set_xlabel("转移对的二阶交互效应 $\\Delta_u\\Delta_vL$（损失单位）")
    ax.set_ylabel("转移对数")
    ax.legend(loc="upper left", fontsize=8.6)
    S.save(fig, "q2-compl-dist")


if __name__ == "__main__":
    print("问题二 · 转移对互补分布：")
    fig_compl_dist()
    print("完成。")
