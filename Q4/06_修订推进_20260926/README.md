# Q4 继续推进修订包（工作中）

**状态：** `revision_in_progress`。本目录是 Q4 历史内部冻结包之后的补充分析，不覆盖或改写已冻结基线。新增内容经复核并由队伍审阅前，不代表修订版已经冻结。

**进度：** 2026-09-26 BBH 补充分析独立复核已 PASS；Task 3 模型烟雾 P1 与正式全量结果 P2 复核均 PASS；Task 5 分轨快照 P1 复核 PASS（回执：[`q4_p1_fixed_receipt.md`](../../tmp/p1_q4_score_bridge_review_v3/q4_p1_fixed_receipt.md)）；Loss–Benchmark 证据链独立范围 P2 复核 PASS，见 [`最终证据审计`](../08_历史遗留问题收口_20260926/q4_loss_bridge_final_audit_20260926.md)。修订版仍待队伍审阅，尚未冻结。

## 本轮范围

截图所列的 C1/C2 与 C3 联用、C4 算力/数据量/Open Weights 年度字段，以及 C8 逐模型逐叶任务原始分数，已在历史包中存在：

- C1/C2 是主队列与元数据来源；C3 Open LLM Leaderboard 另作年度背景序列，见 `../01_方案说明/q4_modeling_report.md` 和 `../03_结果/results/q4_c3_year_summary.csv`。
- C4 年度表包含训练算力、训练数据量、参数量的有效数/中位数/P90、置信等级分层及 Open Weights Yes/No/unknown 计数，见 `../03_结果/results/q4_c4_year_summary.csv`。C4 与 C1 精确名称交集为 0，因此只作宏观背景，不并入个体模型回归。
- C8 已有逐模型逐叶任务表 `../03_结果/results/q4_selected_task_scores.csv` 和六族聚合表。本轮补充对 BBH 24 个叶任务执行逐项剔除影响分析，检查六族指数排序是否依赖某一个叶任务。

## 结果解释边界

逐项剔除是确定性的敏感性分析，不是抽样置信区间，也不表示 24 个任务可互换抽样。它固定历史包中的逐任务百分位和其余五族得分，仅重算 BBH 族内等权均值及六族等权指数。所有 C8 名称候选仍是 `version_unresolved`。

12/24 月无同口径年度滚动起点。Q4/07 已补入从 2025-03 起算的 C4 条件情景点值和模拟分位数带；以 Q4/07 最终报告为准，规模—时间模型一步 MAE 为 1.263，高于持平基线 0.757。Q4/06 较早运行中的 1.305 是旧一轮结果，保留在其历史报告中，不与最终 Q4/07 结果混用。年度预测技能仍 `NOT_VALIDATED`，模拟带覆盖率为 `UNKNOWN`。C4 只作宏观情景锚点，不识别个体能力—算力关系。G4 补充给出描述性贡献点份额，但 Fieller 比率无有限界，不能称为稳定识别的贡献百分比。

v0.6 的 `BLOCKED_NO_FIT` 是冻结历史状态。Q4/08 已补充 C5/C6 的分等级探索映射：高可比层仍只含 7 个 Pythia 尺寸；中可比层分别报告 C5 主表和 C6 扩展表结果，不跨等级合并。C6 Medium 的 68 行包含 C5 Medium 的全部 36 行，另增 32 行；扩展表只作敏感性。C5 留出误差未胜过均值基线；C6 扩展表仅小幅胜出且绝对误差仍大，故当前映射状态为探索性、未校准、不作 Q2 Loss 到 Benchmark 的可靠转换。桥接线独立范围 P2 已通过；这不代表整个 Q4 补充包整体通过。许可证审计也确认 201 个 Open Weights 候选均无已核实评测版本，研究/复现许可状态为 `not_assessed`。本轮完整汇总与复现入口见 [`Q4/08 当前状态`](../08_历史遗留问题收口_20260926/q4_current_status.md)；队伍审阅完成前，本修订包保持 `revision_in_progress`。

分轨历史结果见 [`q4_model_contract_v0.7_addendum.md`](q4_model_contract_v0.7_addendum.md) 与 [`results/smoke/q4_score_forecast_loss_bridge_separation_report.md`](results/smoke/q4_score_forecast_loss_bridge_separation_report.md)。新结果不会绕过各自门控。

分轨状态复核可从仓库根目录运行：`python Q4/06_修订推进_20260926/scripts/q4_score_forecast_loss_bridge_separation.py`。它只读取 Q4 既有结果和门控文件，把状态表、报告与输入/输出 SHA-256 清单写入本修订目录的 `results/smoke/`。该复核是在历史结果生成后对合同作的澄清，不把旧结果追认为按 v0.7 生成。

### Task 3：前沿预测模型 + C4 放缓情景

用户随后明确要求继续完成题目要求的 12/24 个月数值情景预测。新增 `q4_task3_forecast_amendment_v0.8.md` 有限覆盖 v0.7 的“点值留空”规则：只准许给定情景的条件数值，不代表年度预测技能已验证。实现入口为 `scripts/q4_frontier_scenario_model.py`；全量结果已写入 `results/forecast/`，含三种 C4 算力路径、候选模型一步回测、固定情景模拟分位数带、`N∝C^0.73` 敏感性和 SHA-256 清单。正式运行每种情景/期限执行 12,000 次模拟；12 行点值和模拟端点都在 0–100，输入/输出/代码/合同哈希核对通过。覆盖率仍未知，不得称为校准的 95% 预测区间。

## 文件

- `scripts/q4_bbh_task_influence.py`：完整复现入口，支持最小烟雾运行。
- `results/`：逐任务剔除指标、模型排名稳定性、逐任务难度摘要、运行报告与哈希清单。
- `q4_bbh_task_review.md`：独立 P1/P2 范围和证据回执摘要。
- `q4_revision_backlog.md`：历史证据状态、可继续推进项与数据门控项。
- `scripts/q4_loss_benchmark_exploratory_mapping.py` 与 `results/bridge_exploratory/`：Pythia 单家族探索映射复现入口、结果和哈希清单。
- [`Loss–Benchmark 最终证据审计`](../08_历史遗留问题收口_20260926/q4_loss_bridge_final_audit_20260926.md)：分层留出误差、样本重叠、来源身份限制与独立 P2 回执摘要。
