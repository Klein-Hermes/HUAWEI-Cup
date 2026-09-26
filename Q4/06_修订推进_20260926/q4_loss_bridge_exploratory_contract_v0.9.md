# Q4 Loss–Benchmark 单家族探索映射补充合同 v0.9

**范围：** 本补充只重审 Pythia 高可比记录的来源证据，并在原始 C1/C5/C6 实测支持内拟合一条单家族描述关系。它不改写 Q4 v0.6 冻结基线、v0.7 分轨判定或原始附件，也不把探索关系提升为跨模型校准。

## 1. 新增证据与可估范围

1. C5/C6 的 7 条高可比 Pythia 记录按 `(Model, N, D, Loss, 六项 Benchmark)` 与 B1/C1 核对；按模型名、N、D 匹配 B1 最终 `steps=143000` 行，D 均为 299.893B tokens，Loss 与 `val_loss` 一致。B1 展示的 `val_loss` 为 3–4 位小数、`ppl` 为 2 位小数；逐行比较二者舍入区间后，`exp(val_loss)` 与 `ppl` 相容。故本补充将 Loss 表述为以自然对数记的验证集 token 交叉熵（nats/token），并保留其展示精度带来的有限不确定性。
2. C5/C6 的六项分数和 `LB_Average` 与 C1 对应行逐项一致；`LB_Average` 是六项排行榜分数的算术平均。结果定义和归一化按 Open LLM Leaderboard v2 官方说明记录，目标量仍是排行榜六项归一化平均，而非准确率百分数或 Q4 六族百分位指数。
3. 数据仅有 Pythia 一个模型家族。仅因不同参数规模提供了 7 个不同 Loss 值，可估计一条低自由度的**家族内探索线性关系**：`LB_Average = intercept + slope × Val_Loss`。参数规模、训练数据量和家族身份共同变化，所以斜率只是该终点截面的描述关联，不能解释为 Loss 的因果效应。
4. Q2 B1 端点 Loss 只允许在高可比 Loss 实测范围 `[min, max]` 内作同家族条件点映射；支持范围外留空，不外推。与拟合样本重叠的映射标为样本内演示。

## 2. 审核与判定

- 原 v0.6 严格预拟合门控记录保持 `BLOCKED_NO_FIT`：当时尚未做本补充的来源级交叉核对；该历史状态及其代码结果不覆盖。
- 本补充属于新的 `ELIGIBLE_LIMITED_EXPLORATORY` 支线，不等于原严格门控曾通过。其有限许可来自上述来源交叉核对和仅限一个家族的范围约束。
- 在独立检查 C1 与评测详情 JSON 时，如发现单项分数差异，保留原 C1/C5/C6 目标值，记录指标、模型和差值；不得用详情 JSON 静默替换排行榜原值。发生差异的行仍可进入“官方 C1 表值”主探索拟合，但须在敏感性和身份限制中披露。
- 同时报告样本内拟合、按参数规模留一的误差敏感性和留一均值基线。留一规模结果仅为同家族敏感性，不是独立家族验证。
- 不报告跨家族置信区间、可迁移校准误差、校准支持结论或经过校准的预测区间；若留一 MAE 不优于均值基线，明确说明此映射不具备已证实的预测优势。
- 不把这条线接入 C1 前沿预测或 Task 3 C4 情景。Task 3 仍直接预测 C1 原始前沿，二者目标量不同。

## 3. 追溯与来源

- B1：`中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv`。
- C1/C5/C6：`leaderboard_cleaned.csv`、`loss_benchmark_bridge.csv` 和 Q4 逐行配对审计。
- 版本诊断：`C_efficiency_evolution/detailed_results/EleutherAI_pythia-*/*.json`；其 revision、task hash、评测日期用于核对，不能覆盖 C1 清单数值。
- 官方指标协议：Hugging Face [Open LLM Leaderboard v2 About](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/about) 与 [Scores Normalization](https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization)。
- Pythia 训练来源：Biderman et al., [Pythia paper](https://arxiv.org/abs/2304.01373)；EleutherAI [Pythia repository](https://github.com/EleutherAI/pythia)。

入口脚本、逐行 crosswalk、探索拟合、目标支持表、报告和 SHA-256 清单见本目录 `scripts/q4_loss_benchmark_exploratory_mapping.py` 与 `results/bridge_exploratory/`。
