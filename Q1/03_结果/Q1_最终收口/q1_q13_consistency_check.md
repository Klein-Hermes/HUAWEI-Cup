# Q1.3 与 Q1 最终 Q 接口一致性核查

日期：2026-09-24

## 核查结论

```text
Q1.3_RERUN_REQUIRED=false
formal_interface_generated=true
operational_q=q_huber
q_version=Q1-q_huber-v1
```

Q1.3 的现有 p+Q 扩展已覆盖 `q_equal_candidate`、`q_huber_candidate`、宽 Unicode Huber 敏感性和排除 7 个 `rps_*frac*` 字段的 Huber 敏感性。项目负责人已选择 `q_huber`，对应的 `q_huber_candidate` 已在原比较中；正式接口现已生成，因此不需要重跑该实验。

## 代码与运行证据

- 执行代码：`src/f_q1_3_pq_extension.py`，SHA-256 `1d713013e8b3378553352b02ffae61897a991f4239e9748aeec8e883429fa558`。
- p-only 基线代码：`src/f_q1_3_mixture_loss.py`，SHA-256 `43644a23171192d50fedfc761d142b080e70a2233a11cd41b55b5d7ed9c5026a`。
- 运行 manifest：`results/q1_3/pq_extension_v1/manifest.json`，状态 `PASS`；主运行模式 `full`。
- 输入：域映射表、Q1.1 域候选与敏感性表、A4/A5 训练及 A6–A11 既有评估文件。12/12 个输入哈希通过。
- 输出：`target_metrics.csv`、`model_summary.csv`、`paired_comparison.csv`、预测、lambda 选择、可识别性审计及 Q 候选输入表，7/7 个输出哈希通过。

### 实际依赖链

- [`src/f_q1_3_pq_extension.py`](../../../src/f_q1_3_pq_extension.py)：`select_q_rows`（第 116–134 行）读取 `scenario`、`candidate`、`dataset` 和 `score_col`；`load_q_scenarios`（第 138–177 行）为 `q_equal_candidate` 和 `q_huber_candidate` 均指定 `results/q1_1/v1/domain_summary.csv`、`scenario=all_a1`、对应候选名及 `score_col=domain_score`。因此负责人最终选中的 q_huber 与既有场景完全一致。
- 同一场景表也加入宽 Unicode Huber 和排除 7 个 `frac` 字段的 Huber 敏感性。`score_sensitivity_domain.csv` 与 `domain_rank_sensitivity.csv` 分别向这两个场景提供 `domain_score` / `scenario_domain_score`。
- 映射表为 `中文题目/F题/real_attachments/A_data_value/domain_mapping_guide.csv`（代码第 33 行）；输入配比与 Loss 表是 A4/A5 训练及 A6–A11 既有评估文件。映射变体为 `direct_only` 和 `direct_plus_near`，11 个 inferred 领域不填补；A12–A15 估算表未参与。所有这些文件的 manifest 哈希列于复现清单。
- `q_covered`（第 212–221 行）计算 `sum(p_j*q_j for mapped j)/sum(p_j for mapped j)`；`identifiability_audit`（第 223–267 行）将 `qcovered_is_deterministic_function_of_p=true`、`independent_quality_effect_identified=false` 写入审计输出。运行摘要再次固定了公式和解释限制（第 924–932 行）。
- 正式领域接口从同一既有 q_huber `domain_score` 与 `domain_confidence_intervals.csv` 提取，不重新拟合阈值或重新聚合记录。Q1.1 计划（`Q1/01_方案与设计/q1_1_modeling_plan.md` 第 109–111 行）预先规定 dataset×source_domain 的域内 Huber 位置、A1 域阈值校准后冻结应用于 A2/A3，并说明覆盖口径；九个输出组均为全覆盖。

## 公式与结论边界

`Q_covered(p) = sum(p_j q_j for mapped j) / sum(p_j for mapped j)`。因此该项完全由配比 p 与候选域分 q 决定；它是受限预测基函数，不是独立观测的 Q。完整的线性 `pᵀq` 处在线性 p 列空间内，不能识别独立 Q 系数。现有比较中，p+Qcovered 在 A4/A5 嵌套 OOF 和 1M 同尺度评估劣于二次 p-only，60M 局部改善，1B 仍劣于基线；不能据此宣称通用模型优势。

细节和逐目标数据见 [Q1.3 p+Q 报告](../Q1.3/pq_extension_v1/q1_3_pq_extension_report.md)、[运行 manifest](../../../results/q1_3/pq_extension_v1/manifest.json) 及 [评审后整合说明](../Q1.3/pq_extension_v1/postreview_integration_20260924.md)。

正式接口现为 `q1_final_quality.csv`（272,505 行，q_version=`Q1-q_huber-v1`），并由 `q1_final_domain_quality.csv` 提供 9 个既有域级 Huber 汇总。审计文件单列于 `q1_candidate_audit.csv`。Q2 的 M2 仍需独立 Q 变化证据；形式上有了 Q 文件并不等于统计上识别了 Q 的独立效应。
