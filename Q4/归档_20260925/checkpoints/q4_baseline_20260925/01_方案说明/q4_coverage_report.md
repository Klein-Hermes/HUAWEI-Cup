# Q4 覆盖率与有效样本报告

> **历史快照说明（截至 2026-09-25）：** 本报告的数据与 AI 字段对应当日检查点，不代表现行披露或当前复核状态。现行披露见[当前 AI 工具使用记录](../../../../05_审计交付/AI工具使用记录.md)，当前包状态见[冻结包说明](../../../../05_审计交付/q4_freeze_package.md)。

> AI 辅助说明：OpenAI Codex 桌面版 26.917.71314（build 10954，prod；2026-09-25 更新器核验为 up_to_date），开发机构 OpenAI；模型选择器显示名称：；模型版本/发布日期：。（按用户指示暂空。）数据字段、合并和结论须由队伍逐项复核。

本报告按 C8 最新可解析模型目录建立唯一名称候选，并将 C1/C2 行级配准、C3/C4 精确名称交集和任务分数覆盖分开列示。所有名称连接均是候选连接，不等于版本确认。

## 交集样本

| 队列交集 | 模型数 | 候选分母 |
| --- | --- | --- |
| 唯一名称候选 | 1819 | 1860 |
| 完整六族指数 | 789 | 1819 |
| 完整七族指数 | 0 | 1819 |
| 六族 + 正参数 + 提交日期 | 788 | 1819 |
| 六族 + 明确开放权重 | 119 | 1819 |

ARC 完整族有 83 个模型，但它们与完整六族样本的交集为 0；因此七族完整综合分没有可计算样本。主分析保留六族，ARC 单独报告覆盖与任务分数，不把空交集误写为有统计量的敏感性结果。

## 任务族覆盖

| family | version | metric | expected_tasks | models_with_selected_scores | complete_family_models |
| --- | --- | --- | --- | --- | --- |
| ARC | 1.0 | acc_norm | 1 | 83 | 83 |
| BBH | 0.0 | acc_norm | 24 | 1819 | 1819 |
| GPQA | 1.0 | acc_norm | 3 | 1815 | 1815 |
| IFEval | 2.0 | inst_level_strict_acc | 1 | 1815 | 1815 |
| MATH-Hard | 2.0 | exact_match | 7 | 789 | 789 |
| MMLU-Pro | 0.1 | acc | 1 | 1815 | 1815 |
| MuSR | 1.0 | acc_norm | 3 | 1815 | 1815 |

叶任务逐项有效模型数及平均 `n_effective` 见 `../03_结果/results/q4_coverage.csv` 的 `leaf_task` 行；模型交集与缺失分母也在同表。

## 外部来源名称交集

| source_id | row_count | exact_overlap_with_1819_unique_name_candidates | file |
| --- | --- | --- | --- |
| C3 | 4599 | 1818 | leaderboard_extended_timeseries.csv |
| C4 | 3523 | 0 | epoch_all_ai_models.csv |
| C5 | 43 | 21 | loss_benchmark_bridge.csv |
| C6 | 75 | 42 | loss_benchmark_bridge_expanded.csv |
| C7 | 45 | 17 | model_architecture_metadata.csv |

C3/C4 只作为来源覆盖审计。C3 不同来源的评测口径不合并，C4 无精确模型名交集；C5/C6/C7 的局部数据覆盖见 `q4_auxiliary_coverage_review.md` 和桥接质量表。

## 时间与规模样本

| submission_month | n_models | decomposition_supported_month_n_ge_20 |
| --- | --- | --- |
| 2024-09 | 1 | False |
| 2024-10 | 4 | False |
| 2024-11 | 106 | True |
| 2024-12 | 164 | True |
| 2025-01 | 219 | True |
| 2025-02 | 176 | True |
| 2025-03 | 118 | True |

回归只使用月样本量至少 20 的支持月份；较早的稀疏月份保留为描述记录。包含参数、提交日期及完整六族分数的模型交集用于尺度/时间可行性判断，不据缺失字段补齐样本。

C8 文件级解析计数见 `../03_结果/results/q4_c8_manifest.json`（扫描 1958 个文件，解析 1954 个）；C1/C2 完整行配准见 `../03_结果/results/q4_c1_c2_alignment_summary.csv`。

## 主要表格

- `../03_结果/results/q4_model_registry.csv`：每个 C8 最新模型目录一行，保留 C1/C2 原始候选字段、C3 匹配记录摘要和 C4 可用元数据候选。
- `../03_结果/results/q4_match_audit.csv`：C8→C1/C3/C4 名称匹配状态和版本未确认原因。
- `../03_结果/results/q4_coverage.csv`：来源、任务、族、交集和月份有效样本数。
- `../03_结果/results/q4_selected_metric_coverage.csv`：所选任务族的主指标、版本和完整队列计数。
