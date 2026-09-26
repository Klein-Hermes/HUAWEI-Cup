# Q/p 条件响应下的 Q3 资源转移敏感性（按来源分支）

日期：2026-09-26。此处完成的是现有附件的条件模型拟合与迁移压力测试，不是新增真实训练实验。

## 计算口径

- 冻结 B1 M0 参数；Q1 q_huber 参考值 Q0=0.4984781048。假设 Q2 `Q_score` 与 Q1 `q_huber` 数值同尺度，此映射尚未验证。
- 来源曲面逐一拟合 L=E+A N^(-alpha)+B D^(-beta)(Q/Qref)^(-kappa)，不跨来源拼参数；报告留一 Q 水平预测。
- 转移分支冻结 B1 的 E,A,alpha,B,beta，仅在 B6 与 B8 calibrated 上拟合有符号 kappa；按各自 N/D 观测矩形与 B1 网格取交集。
- B8 extrapolated 只参与来源内曲面诊断；N 支持域与 B1 无交集，不进入预算优化。B7 未当作独立重复。
- 分别使用指数、四次幂、对数型 g(Q)；预算不足的情景明确记为 INFEASIBLE。固定 p 为 B1 共同数据政策条件，不填造 17 维配比。

## B1 锚定 Q 响应

| 来源 | κ | M0 RMSE | 锚定 RMSE | 留一 Q RMSE | cluster bootstrap 95% | 支持域内 B1 点 |
|---|---:|---:|---:|---:|---:|---:|
| B6_semi_synthetic | 0.383001 | 0.21724 | 0.16395 | 0.17306 | [0.34453, 0.42902] | 1112/1176 |
| B8_calibrated | -0.819373 | 1.1825 | 1.0942 | 1.1072 | [-0.84516, -0.79773] | 1128/1176 |

bootstrap 以来源内匹配 (N,D) 单元为重抽样单位；它刻画半合成/校准网格稳定性，不代表真实训练随机性。

## 来源曲面与留一 Q 验证

完整参数表见 `q2_source_full_curve_fits.csv`；逐 Q 留出误差见 `q2_leave_one_Q_level_predictions.csv`。B8 extrapolated 结果仅解释其外推来源内拟合，不迁移到 B1 优化。

## Q3 转移摘要

| 来源 | g(Q) | κ | 可行/15 | 不可行情景 | Q>Q0 | 相对受限 M0 重选 N/D | 可比相邻边/22 | N变边 | D变边 | Q变边 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B6_semi_synthetic | exponential | 0.383001 | 14/15 | 1 | 14 | 5 | 20/22 | 10 | 16 | 12 |
| B6_semi_synthetic | power4 | 0.383001 | 14/15 | 1 | 14 | 5 | 20/22 | 10 | 16 | 9 |
| B6_semi_synthetic | logarithmic | 0.383001 | 14/15 | 1 | 14 | 5 | 20/22 | 10 | 16 | 7 |
| B8_calibrated | exponential | -0.819373 | 14/15 | 1 | 0 | 0 | 20/22 | 11 | 15 | 0 |
| B8_calibrated | power4 | -0.819373 | 14/15 | 1 | 0 | 0 | 20/22 | 11 | 15 | 0 |
| B8_calibrated | logarithmic | -0.819373 | 14/15 | 1 | 0 | 0 | 20/22 | 11 | 15 | 0 |

## 结论与边界

1. B6 与 B8 calibrated 的锚定 κ 分别估计，不合并。若符号相反，Q 导致的结构转移没有来源稳健的唯一方向。
2. 这些曲线把半合成/校准来源的质量响应迁移到 B1 的固定 M0 结构上，只能回答条件情景；不构成 B1 的实测 Q 效应。
3. A 侧 p→Loss 的跨尺度检验没有通过，且 B1 缺 checkpoint 级 p；所以本结果按题面允许分支固定 p，不报告 p 的 B 侧边际效应或完整 p/Q 联合最优。
4. Q1 `q_huber` 与 Q2 `Q_score` 映射未经验证，B8/B6 数据为半合成/校准来源；真实独立 p/Q 识别仍需受控训练或可追溯运行日志。
5. 逐情景成本和可行性见 `q3_qp_conditional_scenarios.csv`；邻接边见 `q3_qp_adjacent_transitions.csv`；来源域交集见 `q3_source_support_intersections.csv`。

复现命令：`D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/solve_q3_qp_conditioned_transition_v2.py`。
