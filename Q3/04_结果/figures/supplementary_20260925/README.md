# Q3 补充图集（2026-09-25）

本目录纳入 P1、P2、P3 论文版、四张新增结果图，以及 Q3 专属建模分支图；P4 按用户决定不纳入。所有图像文件均记录在 [supplementary_manifest.json](supplementary_manifest.json) 并核对 SHA-256。

## 图件

| 图件 | 内容与用途 | 数据来源 | 文件 |
|---|---|---|---|
| P1 预算—N/D参考配置 | 15 个预算×上下文离散情景下的参考配置；线段只连接给定预算档位，不代表连续预算估计。 | `Q3/04_结果/M0_ND_reference_frontier.csv` | [PNG](P1_预算_ND前沿/result_q3_p1_budget_nd_frontier.png)、[PDF](P1_预算_ND前沿/result_q3_p1_budget_nd_frontier.pdf)、[SVG](P1_预算_ND前沿/result_q3_p1_budget_nd_frontier.svg)、[灰度版](P1_预算_ND前沿/_qa/result_q3_p1_budget_nd_frontier_grayscale.png) |
| P2 Bootstrap 配置稳定性 | 15 个固定 Q 情景的 N、D、预测 Loss 区间及配置切换概率；稳定性限于 8 条 B1 轨迹、既定 M0 形式和观测支持域。 | `M0_ND_cluster_bootstrap_frontier.csv`、`M0_ND_cluster_bootstrap_selection_frequency.csv` | [PNG](P2_Bootstrap稳定性/result_q3_p2_bootstrap_selection_stability.png)、[PDF](P2_Bootstrap稳定性/result_q3_p2_bootstrap_selection_stability.pdf)、[SVG](P2_Bootstrap稳定性/result_q3_p2_bootstrap_selection_stability.svg)、[灰度版](P2_Bootstrap稳定性/_qa/result_q3_p2_bootstrap_selection_stability_grayscale.png) |
| P3 资源配置仪表盘 | 汇总 N、D、成本份额和 M0 Loss；22 对相邻情景变化是描述性比较，不是正式结构转移结论。 | `M0_ND_reference_frontier.csv`、`M0_budget_context_cost_shares.csv`、`M0_budget_context_descriptive_shifts.csv` | [PNG](P3_资源配置仪表盘/result_q3_p3_resource_shift_dashboard.png)、[PDF](P3_资源配置仪表盘/result_q3_p3_resource_shift_dashboard.pdf)、[SVG](P3_资源配置仪表盘/result_q3_p3_resource_shift_dashboard.svg)、[灰度版](P3_资源配置仪表盘/_qa/result_q3_p3_resource_shift_dashboard_grayscale.png) |
| S1 Bootstrap Loss 区间 | 单独放大各情景 Loss 的 Bootstrap 区间；与 P2 的 Loss 面板信息重叠，作为更聚焦的备选视角保留。 | `Q3/04_结果/Q3_T3_稳健性与配置切换/q3_bootstrap_summary.csv` | [PNG](S1_Bootstrap_Loss区间/result_q3_supp_bootstrap_loss_sensitivity.png)、[SVG](S1_Bootstrap_Loss区间/result_q3_supp_bootstrap_loss_sensitivity.svg)、[灰度版](S1_Bootstrap_Loss区间/_qa/result_q3_supp_bootstrap_loss_sensitivity_grayscale.png) |
| S2 固定 Q 成本占比 | 单独展示训练与注意力成本构成；与 P3 的成本构成面板重叠，保留为更聚焦的备选视角。 | `Q3/04_结果/M0_budget_context_cost_shares.csv` | [PNG](S2_固定Q成本占比/result_q3_supp_fixedq_cost_shares.png)、[SVG](S2_固定Q成本占比/result_q3_supp_fixedq_cost_shares.svg)、[灰度版](S2_固定Q成本占比/_qa/result_q3_supp_fixedq_cost_shares_grayscale.png) |
| S3 近优候选规模响应 | 显示 Loss 容许差从 0% 到 10% 时可行近优候选数量的阶梯变化；这是有限网格计数，不是概率或弹性。 | `Q3/04_结果/Q3_T2_近最优_Pareto_边际收益/q3_near_optimal.csv` | [PNG](S3_近优候选规模响应/result_q3_supp_near_optimal_count_response.png)、[SVG](S3_近优候选规模响应/result_q3_supp_near_optimal_count_response.svg)、[灰度版](S3_近优候选规模响应/_qa/result_q3_supp_near_optimal_count_response_grayscale.png) |
| S4 质量增量成本曲线 | 对比三种题面 $g(Q)$ 形式和 B1 的 D 最小/中位/最大参考值；是成本公式计算，不表示质量收益或最优 Q。 | `Q3/04_结果/Q_cost_increment_envelope.csv` | [PNG](S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost.png)、[SVG](S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost.svg)、[灰度版](S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost_grayscale.png) |

## Q3 分支流程图

[flow_q3_model.png](flow_q3_model/flow_q3_model.png)（[SVG](flow_q3_model/flow_q3_model.svg)、[灰度版](flow_q3_model/flow_q3_model_grayscale.png)）概括已计算的固定 Q₀/M0 分支、依赖未知 βQ 的条件 Q-only 分支，以及尚未识别的 Q/p 联合优化证据门槛。它不表示完整联合最优已求得。图表契约与可复现脚本见 `flow_q3_model/`。

## 复现来源

- P1–P3：绘图源已归入 [`Q3/03_代码/图表源/`](../../../03_代码/图表源/README.md)，论文版入口为 `../../../03_代码/图表源/论文版/plot_q3_supplementary_figures_paper.py`；排版决定和建议图注在本目录。
- S1–S3：`Q3/补充图_候选20260925/plot_q3_supplementary_candidates.py`；字段、统计口径和限制见同目录的三份图表契约。
- S4：`Q3/04_结果/figures/supplementary_20260925/S4_质量增量成本曲线/plot_q3_quality_incremental_cost.py`，直接读取既有成本曲线 CSV。
- Q3 分支图：`Q3/04_结果/figures/supplementary_20260925/flow_q3_model/plot_flow_q3_model.py`，依据现有模型代码与报告生成。
- 文件来源和输出哈希见本目录的 `supplementary_manifest.json`。

P1–P3 的历史源图已由 Q3 图件副本承接；清单中的 `source_path` 均指向 Q3 内留存文件。重绘脚本使用 Q1/Q3 项目内数据，并把结果写入 `_rebuild_supplementary_*` 暂存目录，不依赖 `zwj/`。

## 审查记录

S4 与分支图的生成脚本均退出码为 0，布局门禁和彩色/灰度目检通过；两图的彩色 PNG 与灰度 PNG 均为 300.99 DPI。全局 `figure_audit.py --questions q3 --strict` 返回 1，因此不记为 Q3 全局审计通过：它还报告 Q3 图类数量门槛及既有顶层图的 DPI/嵌入栅格问题。本记录只说明本补充目录图件的检查结果。

## 取用提示

P2 与 S1、P3 与 S2 分别有信息重叠；按版面需要可在正文和附录中选用不同视角。S3 扩展现有 1%、3%、5% 近优阈值对照，报告时应称“候选数量对 Loss 容许差的响应”。P4 未收入本图集：固定 Q 的 M0 中 N、D 和 Loss 重合，单图呈现的信息增量有限。
