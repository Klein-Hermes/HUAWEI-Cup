# Q3 固定 Q 的 M0 离散资源转移审计

**计算范围：** P1 smoke slice: one adjacent-budget pair, 20 actual Q2 trajectory-bootstrap fits。

## 判据与方法

固定 Q=Q0，不估计 Q→Loss，也不纳入 p。对预算相邻档与上下文相邻档分别组成 1 对情景；每对使用同一个 Q2 B1 轨迹 Bootstrap 复本进行配对比较，共计 20 个复本/情景对。每个复本都在满足题面成本约束的 B1 实测 1,176 点子集中重新选择 M0 预测 Loss 最低配置。

将 N/D 网格点发生改变的 Bootstrap 频率至少 95% 定义为‘Bootstrap 稳健配置改变’。训练与注意力成本份额的总变差按精确有理数计算；至少 95% 配对复本出现非零总变差时，定义为‘Bootstrap 稳健成本构成改变’。同时报告总变差第 5 百分位数作为分布摘要，但不以它单独决定该标签。成本构成变化与 N/D 重新分配分开报告。这里不使用连续优化器的 KKT 活跃集判据，因为当前支路是有限候选网格枚举。

## 逐对结果

| 对比 | 左情景 | 右情景 | N/D配置改变频率 | Δlog N 中位数 [95%区间] | Δlog D 中位数 [95%区间] | 成本份额 TV 第5百分位数 | 判定摘要 |
|---|---|---|---:|---:|---:|---:|---|
| T01 (adjacent_budget) | 1e+19 FLOPs / 2,048 tok | 1e+22 FLOPs / 2,048 tok | 100.0% | 3.7435 [3.7435, 3.7435] | 3.2958 [3.2958, 3.2958] | 0 | Bootstrap稳健的N/D配置改变 |

## 解释边界

在本次分析的 1 对情景中，1 对达到 N/D 配置改变频率 ≥95% 的门槛，0 对达到成本份额非零变化频率 ≥95% 的门槛。配对 Bootstrap 复本数为 20；正式全量运行使用 1,000 次复本，但独立轨迹簇只有 8 条，故分位数用于稳定性提示。
该结果是给定冻结 M0 形式和 B1 离散支持域下的情景敏感性结论，不代表因果效应、连续预算阈值或完整 Q3 的结构转移。最高预算若选择 B1 支持网格上界，结果仍受网格边界截断。Q、p 效应以及支持域外行为不在本分析范围。

配对逐复本结果见 `M0_discrete_transition_bootstrap_draws.csv`；汇总及可复现哈希见 `M0_discrete_transition_pairs.csv` 与 `M0_discrete_transition_manifest.json`。

复现命令：`python Q3/03_代码/assess_q3_m0_discrete_transitions.py --smoke --output-dir "Q3/归档_20260925/tmp_q3_transition_p1"`。
