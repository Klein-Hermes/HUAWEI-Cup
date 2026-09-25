# Q3-T1 固定 Q₀ 主结果整合与资源转移分析

**生成时间（UTC）：** 2026-09-24T18:13:28+00:00  
**范围：** 固定 Q=Q₀、p 不进入 M0；在 Q2 B1 的精确观测 N/D 支持网格中复用既有 M0/M1/M3/M4 与配对 Bootstrap 结果。
**方法优化：** 本整合只做带哈希核验的数据连接和派生差值，不重新拟合 M0、不重跑 1,000 次 Bootstrap；15 个场景使用预算与上下文双键，22 个转移保留原 pair_id。

## 交付检查

- 主结果：15 个预算×上下文场景。
- 资源转移：22 对相邻预算/上下文场景。
- B1/M3 支持点：1176；M4/M1 掩码各 17,640 行（15 个场景，每场景 1,176 个候选点）。
- 独立成本审计一致：15/15 个场景。
- 15/15 个最优点均满足 M3 支持、M4 预算与 M1 最终可行标记；固定 Q₀ 下 C_Q=0，成本份额合计为 1。

## 资源转移汇总

| 比较类型 | 配对数 | Bootstrap 稳健 N/D 配置改变 | Bootstrap 稳健成本构成改变 |
|---|---:|---:|---:|
| 相邻预算 | 10 | 10/10 | 0/10 |
| 相邻上下文 | 12 | 7/12 | 12/12 |

资源转移表保留左右场景配置、Loss、成本、预算利用率、可行点数，并附原分析的 N/D 对数变化区间、成本份额总变差和稳健改变标记。配置改变与成本构成改变分开判断。

## 核心图

图左汇总 15 个情景的 N*、D*、M0 Loss* 和预算利用率；图右汇总 22 对配对 Bootstrap 的稳健变化比例。变化比例只表示离散配置或成本构成改变，不表示变化方向更优。

- `figures/result_q3_fixedq_master_and_resource_transition.svg`
- `figures/result_q3_fixedq_master_and_resource_transition.png`
- `figures/_qa/result_q3_fixedq_master_and_resource_transition_grayscale.png`
- `figures/result_q3_fixedq_master_and_resource_transition.pdf`

## 解释边界

- 本表为固定 Q₀/M0 的 B1 支持网格参考结果；不含 Q/p 响应效应，不是完整 Q3 联合最优。Loss 为 B1 同域拟合 M0 的预测值。
- 当前 M3 只开放精确观测组合；凸包与距离是诊断字段，不授权非观测插值。
- 高预算所有 B1 网格点可行，最优点可能触及支持上界；预算利用率低不代表域外资源没有收益。
- Bootstrap 使用既有 8 条独立轨迹、1,000 个配对复本；本次未重拟合或扩大不确定性解释。
- T1 的 15 行主表用于汇总，不代替候选级 M3/M4/M1 掩码；近优/Pareto 下游任务仍应读取完整候选表。

复现命令：`$env:MPLBACKEND = 'Agg'; & 'D:/Anaconda/envs/mmdet/python.exe' 'Q3/03_代码/assemble_q3_fixedq_master_results.py'`。
