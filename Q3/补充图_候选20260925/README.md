# Q3 补充图候选（2026-09-25）

此目录保留 S1–S3 的候选图源文件、图表契约、灰度预览与可复现绘图脚本。它们只读取已有结果 CSV；不重拟合、不重跑 Bootstrap。当前论文图集将对应的整理版纳入 [`../04_结果/figures/supplementary_20260925/`](../04_结果/figures/supplementary_20260925/README.md)，但本候选目录本身不是冻结包交付物。图集 manifest 记录了候选目录的来源路径，因此不要移动或改名；若重生成源图，需同步刷新并复核图集 manifest。

## 运行

从项目根目录运行：

```powershell
D:\Anaconda\python.exe Q3\补充图_候选20260925\plot_q3_supplementary_candidates.py
```

脚本会先核对字段、缺失值、情景覆盖、单位口径和已有阈值标记，再输出 SVG、301 dpi PNG 与 `_qa/*_grayscale.png` 灰度预览。脚本检查目标图像是否已存在；若存在，会跳过对应图以防覆盖。

## 图表与口径

| 候选图 | 信息增量 | 图表契约 |
|---|---|---|
| Bootstrap Loss 区间 | 以中位数为零点显示每情景 2.5–97.5 百分位相对偏差，并在右列保留绝对 Loss 中位数，便于辨读窄区间。 | [bootstrap_loss_sensitivity.md](bootstrap_loss_sensitivity.md) |
| 固定 Q 成本构成 | 展示 15 个已选情景中训练与注意力的实际 FLOPs 成本份额；固定 Q 下质量成本份额为 0。 | [fixedq_cost_shares.md](fixedq_cost_shares.md) |
| 近优候选规模响应 | 在 0–10% 相对 Loss 容许差内，枚举 `final_feasible` 候选数量随阈值的阶梯变化。 | [near_optimal_count_response.md](near_optimal_count_response.md) |

## 文件名

- `result_q3_supp_bootstrap_loss_sensitivity.{svg,png}` 和 `_qa/result_q3_supp_bootstrap_loss_sensitivity_grayscale.png`
- `result_q3_supp_fixedq_cost_shares.{svg,png}` 和 `_qa/result_q3_supp_fixedq_cost_shares_grayscale.png`
- `result_q3_supp_near_optimal_count_response.{svg,png}` 和 `_qa/result_q3_supp_near_optimal_count_response_grayscale.png`

这些图属于待评审候选，不是正式冻结图，也不构成新的优化、因果或域外结论。
