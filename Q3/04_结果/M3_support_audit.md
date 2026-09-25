# Q3 支持域、预算与 M1 求优总审计

## 审计范围

本轮落实了 M3 支持域、M4 预算掩码与 M1 求优分工。数据候选域沿用 Q2 B1 的 1,176 个实际 N/D 组合；模型固定 $Q=Q_0$，$p$ 未进入目标函数，因此结论是既有 M0/B1 离散支路的参考优化，不是完整 Q/p 联合最优。

## 关键结果

| 检查项 | 结果 |
|---|---:|
| B1 实测 N/D 组合 | 1,176 |
| M3 候选中的精确观测组合 | 1,176 |
| 凸包内非观测候选 / 凸包外候选 | 0 / 0 |
| M3 主搜索许可点 | 1,176（精确观测点） |
| M4 掩码规模 | 17,640 行（1,176 点 × 15 场景） |
| M4 跨情景预算可行记录数 | 10,935（候选×场景记录求和） |
| M1 对既有 M0 的选择/可行数一致性 | 15/15 场景一致 |

M3 在标准化 $(\log N,\log D)$ 空间计算凸包和最近观测距离；这些字段用于几何诊断，不作为自动插值许可。M4 独立计算预算可行性。M1 在优化前合成 `m3_search_allowed AND budget_feasible`，再最小化冻结的 M0 预测验证 Loss。

## 解释边界

- 当前支持集是离散实测网格；未验证任何非观测插值点。若未来需要插值，应先明确候选网格、交叉验证证据和接受规则。
- $10^{24}$ FLOPs 档中 1,176 个支持点全部预算可行，M1 最优点触及 $N,D$ 支持上界；不应据此向更大规模外推。
- 131,072 Token 上下文、$10^{19}$ FLOPs 情景下最优点触及 $N_{\min}$。
- Q2 尚未满足独立 Q/p 响应面的证据门槛；因此本轮没有解除完整 Q/p 联合优化限制。

## 交付物

- `m3_observed_support.csv`、`m3_candidate_grid.csv`、`m3_q3_support_grid.csv`
- `m4_budget_feasible_mask.csv`、`m1_final_feasible_mask.csv`
- `m1_optimum_boundary_audit.csv`
- `M3_support_domain_audit.md`、`M4_budget_feasibility_audit.md`、`M1_masked_optimization_report.md`
- `figures/raw_q3_observed_nd_support.*`
- `figures/process_q3_m3_m4_mask_audit.*`
- `figures/result_q3_m1_budget_optima.*`
- `Q3_M3_M4_M1_图表QA.md`、`Q3_M3_M4_M1_图表清单.json`
- `Q3_M3_M4_M1_图表读图复核.md`
- `Q3_M3_M4_M1_任务拆分与并行计划.md`

## 复现

执行顺序、并行边界和运行时说明见 `Q3_M3_M4_M1_任务拆分与并行计划.md`。各步骤均保留独立报告和 SHA-256 manifest；图表清单记录输入输出哈希。
