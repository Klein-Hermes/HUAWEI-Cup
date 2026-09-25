# Q3 的 Q0 基线敏感性：质量成本与条件重分配阈值

## 分析口径

以 Q1 A1 七域宏平均 `q_huber` 点值 0.4984781048 为主基线；同时传播其 95% 区间端点 [0.4965674432, 0.5004265176]，并将冻结的 `q_equal` 0.4962230100 作为单独接口敏感性分支。`q_equal` 是替代口径，不是 `q_huber` 的置信区间端点。

第一部分按题面三种 g(Q) 公式重算 Q 提升成本；第二部分在原有条件假设 L_cond=L_M0−β_Q(Q−Q0)、β_Q≥0 下重算 N/D 首次重分配阈值。β_Q 保持未知，Q0 敏感性不提供 Q→Loss 的经验估计。

## Q=1 时的成本敏感性（B1 最大 D 支持点）

| Q0 分支 | g(Q) | 增量成本 FLOPs |
|---|---|---:|
| q_huber_point (0.498478) | exponential | 1.15017e+21 |
| q_huber_point (0.498478) | power4 | 1.40688e+21 |
| q_huber_point (0.498478) | logarithmic | 3.65075e+20 |
| q_huber_ci95_low (0.496567) | exponential | 1.15085e+21 |
| q_huber_ci95_low (0.496567) | power4 | 1.4083e+21 |
| q_huber_ci95_low (0.496567) | logarithmic | 3.66993e+20 |
| q_huber_ci95_high (0.500427) | exponential | 1.14947e+21 |
| q_huber_ci95_high (0.500427) | power4 | 1.40543e+21 |
| q_huber_ci95_high (0.500427) | logarithmic | 3.63126e+20 |
| q_equal_sensitivity (0.496223) | exponential | 1.15097e+21 |
| q_equal_sensitivity (0.496223) | power4 | 1.40855e+21 |
| q_equal_sensitivity (0.496223) | logarithmic | 3.67339e+20 |

## Q0 变化对条件 N/D 重分配阈值的影响

下表比较 q_huber 点值、区间端点和 q_equal 分支，在本次实际计算的 1 个预算×上下文情景和所列 g(Q) 下的首次 N/D 重分配 β_Q 阈值范围；它不是对 β_Q 的估计。

| Q0 分支 | g(Q) | 首次 N/D 重分配 β_Q 最小值 | 最大值 | 无重分配情景数 |
|---|---|---:|---:|---:|
| q_huber_point | exponential | 0.453701 | 0.453701 | 0 |
| q_huber_ci95_low | exponential | 0.45164 | 0.45164 | 0 |
| q_huber_ci95_high | exponential | 0.455826 | 0.455826 | 0 |
| q_equal_sensitivity | exponential | 0.451271 | 0.451271 | 0 |

## 解释边界

`q_huber` 区间来自 Q1 A1 七域宏平均 Bootstrap；它描述该质量接口的 A1 汇总不确定性，不是 B1/Pythia 训练配方混合质量的区间。`q_equal` 作为另一冻结接口的敏感性比较，不应与抽样区间混为一谈。
所有阈值仍依赖未识别的线性 β_Q 条件模型、Q2 M0 和 B1 支持网格；不得把它们写成 Q2 估计出的质量收益或经验 Q 最优。p 未进入目标函数。

成本明细见 `Q3_Q0_baseline_sensitivity_cost.csv`；条件阈值明细见 `Q3_Q0_baseline_sensitivity_beta_thresholds.csv`。

复现命令：`python Q3/03_代码/analyze_q3_q0_baseline_sensitivity.py --smoke --output-dir "Q3/归档_20260925/tmp_q3_q0_p1"`。
