# README_Q2_FINAL：Q2 当前数据版本收口包

## 最终状态

当前数据版本记为 **FROZEN_WITH_LIMITATIONS**。经典 M0 `Loss=f(N,D)` 已冻结；M0 边际效应、弹性、Bootstrap 区间、B1–B5 分源验证边界及 p/Q 可识别性门禁已整理。B 侧 p 门禁为 `FAIL_CURRENT_DATA`，Q 独立增量门禁为 `FAIL_CURRENT_EVIDENCE`；M1/M2 均为 `NOT_ESTIMATED`。这不是 p 或 Q 无效的结论。

Q1 冻结的操作性接口为 `q_huber`，敏感性接口为 `q_equal`。这只是 Q1 的接口决定，不表示 B1–B5 已观测到与 Loss 行级匹配、且独立变化的 Q。`Q_covered(p)` 是 p 的派生特征，也不能证明独立 Q 增量。M0 本身不使用 Q/p。

## 按题目分析报告核对 Q2 要求

阶段结论与完整回答补充均按四项组织：广义标度律；边际效应和弹性；领域配比替代与互补；质量—规模等 Loss 条件及分来源验证。详见 `q2_stage_conclusion.md` 和 `Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`。该组织用于逐项对照题意，不把 Q2 重新命名为四个正式小问。

## 主要交付

- 阶段结论：`q2_stage_conclusion.md`
- A/B 证据矩阵：`q2_evidence_matrix.csv`
- p/Q 门禁：`q2_identifiability_gates.csv`、`q2_q_incremental_effect_gate.md`
- 模块状态：`q2_model_status.csv`
- M0 逐检查点效应：`q2_m0_marginal_effects_by_checkpoint.csv`（1,176 行）
- M0 轨迹摘要：`q2_m0_marginal_effects_by_size.csv`（8 条轨迹）
- M0 轨迹覆盖分布摘要：`q2_m0_effects_distribution_summary.csv`（含 N/D/Loss 与边际效应的 min/Q1/median/Q3/max，并区分 Bootstrap CI）
- 分布摘要口径说明：`q2_m0_effects_summary_readme.md`
- B2–B5 验证边界：`q2_validation_scope_closeout.md`
- M0 四类边际效应/弹性图：`Q2/04_图表/q2_m0_effects/`（PNG、SVG、PDF 和灰度预览）
- 图表契约：`q2_m0_effects_figure_contract.csv`
- 输入登记：`inputs_manifest.csv`，共 107 个输入/来源记录；解释见 `README_inputs.md`
- 运行日志：`run_log.txt`
- 输入输出哈希：`q2_final_package_manifest.json`、`q2_closeout_manifest.json`、`q2_m0_effects_validation_manifest.json`

## 完整回答、图表与来源索引

- 完整回答及按题意逐项对照：`Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`；该报告含图表索引、数据投毒边界和 M1/M2 识别条件。
- 完整回答图表：`Q2/04_图表/q2_full_answer/`；图表合同和布局 QA：`Q2/03_结果/完整回答补充/v1/q2_full_answer_figures_contract.csv`、`q2_full_answer_figures_qa.json`。
- 图表数据/来源与复现哈希：`Q2/03_结果/完整回答补充/v1/repro_manifest.json`。
- B4/B5 原文口径和来源定位：`Q2/05_审计交付/论文网址清单.md`、`Q2/03_结果/经典ScalingLaw基线/v1/loss_protocol_evidence_sources.md`、`Q2/03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md`；引用时按这些文件区分原文指标与本地 `val_loss`，不宣称其与 B1 绝对可比。
- M0 参数和效应、B2–B5 结果的依据文件见本包上列报告及 `q2_evidence_matrix.csv`；完整回答补充的原始结果副本在 `Q2/03_结果/完整回答补充/v1/reference_evidence/`。

## 唯一收口复现命令

在项目根目录运行：

```powershell
"D:\Anaconda\python.exe" src/q2_finalize_current_data_package.py
```

该命令先从已冻结的 M0 参数和已保存的 Bootstrap 样本重建效应表，再重建证据矩阵和门禁，最后出图、登记输入并生成哈希与运行日志。**不重新拟合 M0、不重新抽样 Bootstrap、不估计 M1/M2、不重复外部来源搜索。** 若要重新运行 M0 拟合，须单独执行经典基线入口并重新审核 Gate 0；本收口命令不会触发该步骤。

## 适用边界和输入完整性

- B1：1176 个检查点属于 8 条轨迹，每条 147 点；重复检查点不能按独立实验计数。
- Bootstrap：1000 次保存的完整轨迹重抽样拟合，独立轨迹 cluster 只有 8 条，区间只作稳定性提示。
- B2 半合成、B3 同来源插值；B4/B5 与 B1 的绝对 Loss 可比性未建立，不汇总单一跨源分数。
- 数据说明版本：完整回答补充核对 B6–B10 数据角色时，实际使用 `中文题目/F题/数据说明(无隐藏字段版本）.pdf` 的可见文本；该来源已单独记录在 `inputs_manifest.csv` 和 `Q2/03_结果/完整回答补充/v1/repro_manifest.json`，仅用于角色核验，不作为 M0 拟合输入。
- 当前输入与历史冻结清单的差异：missing frozen source: 中文题目/F题/数据说明.pdf；changed since frozen snapshot: 题目分析报告.md
- 历史基线清单记录的原始 `中文题目/F题/数据说明.pdf` 当前在原路径缺失；无隐藏字段版本是另一个明确登记的文件，不替代历史原件，也没有用于重跑 M0。因该原件缺失且基线记录的根目录题目分析文件版本已变化，本包**不声称从全部原始附件重跑过 M0**。原始带隐藏字段 PDF 的低可见度文字未读取，亦不作为题意、数据、方法或结论依据。

## 版本边界

这是一份基于当前可核验证据的 Q2 版本收口，不永久关闭研究问题。若新的、可追溯且与 B 侧 Loss/检查点行级匹配的训练组成或独立 Q 变化出现，应先重开对应门禁，再按 `M0 → M1 → M2` 顺序推进；不以当前 `FAIL` 推断“p/Q 无效”。
