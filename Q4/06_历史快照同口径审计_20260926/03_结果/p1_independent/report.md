# Q4 历史快照与同口径审计（补充件）

**审计范围：** 核对 C1/C2 月度新提交口径、C1 来源、v2 公开历史快照覆盖范围、v1/v2 可比性，以及本地 C1 Average P90 重建。**本补充不改写 2026-09-26 已冻结的 Q4 基线。**

## 结论

1. 目标量锁定为每个 C1 `Submission Date` 月份中新提交记录的 C1 `Average ⬆️` 第 90 百分位；队列仅保留 C2 `Epoch_AI_Open_Weights` 明确为 `Yes`。分位数采用线性插值。C1/C2 按全部 12 个基础字段做一对一匹配，不按行号或仅按模型名拼接。
2. 项目附件登记把 `leaderboard_cleaned.csv` 指向 v1 仓库 `open-llm-leaderboard-old/results`，该登记与文件内容不符。此文件与公开 v2 `open-llm-leaderboard/contents` 的 C1 字段行可做全量核验；来源更正记录见 `q4_source_manifest_correction.csv`。Q4 主队列先限于原流程中的 **1,819 个 C8 名称精确且 C1 唯一候选**，再依据 C2 筛选；原附件清单保持只读。
3. Hugging Face v2 `contents` 历史有 3,598 次提交，当前固定版本为 `9c09a7cae43334062a82cb164f2ef255013dafa2`（`2025-03-20T12:17:27Z`）。它支持复原至 2025-03 的历史快照，但不含 2026-03 的观测。v1 的六项任务与 v2 六个任务族不同，不能把 v1 的 Average/P90 接到 v2 时间序列。
4. C1/C2 完整字段对齐：**4,576/4,576** 一对一匹配。Remote v2 row check: not run in this output directory.
5. 重建得到 **201** 条“明确开放权重”记录，覆盖 10 个有数据月份；每月样本数为 4–38。其中 2 个月 n<10、7 个月 n<20；若采用 n≥20 才纳入动态估计，仅剩 3 个月，不能支持年度预测。重建 P90 与冻结表 `0/10` 月一致。
6. 当前 C2 的 Epoch 开放权重标签没有随 v2 月度快照版本化，因此历史月份结果只能称“用当前标签回溯定义的描述性队列”，不能充当无前视偏差的开放状态回测。C8 revision comparison was not run because a full pinned v2 source join was not available.
7. **2026-03 事后验证不可得**：所核 v2 内容库最后版本为 2025-03-20；不得用 v1、混合论文数据或新评测套件冒充同口径实际值。2027-03 仍是未来目标。继续保持 2025-03 训练截点和 12/24 月数值预测空值。

## 逐月重建

| 提交月 | 明确 Open Weights 记录数 | 线性 P90（C1 Average） | 与冻结表一致 |
|---|---:|---:|---|
| 2024-06 | 38 | 28.332922 | True |
| 2024-07 | 9 | 28.692984 | True |
| 2024-08 | 4 | 30.810184 | True |
| 2024-09 | 19 | 40.965103 | True |
| 2024-10 | 11 | 38.744236 | True |
| 2024-11 | 15 | 39.837872 | True |
| 2024-12 | 17 | 40.318708 | True |
| 2025-01 | 37 | 39.739830 | True |
| 2025-02 | 33 | 41.321819 | True |
| 2025-03 | 18 | 41.274513 | True |

P90 的单位/量尺沿用 C1 `Average ⬆️` 原值；本表不对六个指标重新加权、不使用 C8 子任务分数替代 C1 Average。

## 同口径纳入规则

- **主数据：** 固定到 Hugging Face v2 `open-llm-leaderboard/contents` commit `9c09a7cae43334062a82cb164f2ef255013dafa2`；保留 C1 原始分值、提交月和上游 `Model sha`。
- **样本：** 每个 C1 记录单独保留；只把 C2 明确 `Yes` 的行用于描述统计。C2 的外部标签不是版本化的 as-of 字段，历史回测必须额外取得带日期的开放权重快照或人工核验记录。
- **版本：** `Model sha` 与 C8 `model_sha` 精确相等可确认权重版本相同；同名不等于同版本，SHA 相同也不单独证明评测任务、指标定义、评测代码和提示设置相同。需要将逐项任务配置另列门控。
- **拒绝混池：** `open-llm-leaderboard-old/results` 为 v1；C3 `Historical (papers/reports)` 为异源结果；二者都不进入 v2 月度 P90 序列。
- **时间目标：** 2026-03 只有在同一个 v2 指标源有记录且能重建当时开放权重标签时才能事后验证；若源未延续，目标标记 `not_observable_under_protocol`。2027-03 不发布当前实际值或无证据外推。

## 来源与复现

- 本地 C1：`D:/PycharmProject/HUAWEI-Cup/中文题目/F题/real_attachments/C_efficiency_evolution/leaderboard_cleaned.csv`
- 本地 C2：`D:/PycharmProject/HUAWEI-Cup/中文题目/F题/real_attachments/C_efficiency_evolution/leaderboard_enhanced.csv`
- v2 表及版本：<https://huggingface.co/datasets/open-llm-leaderboard/contents/tree/9c09a7cae43334062a82cb164f2ef255013dafa2>
- v2 官方历史说明：<https://github.com/huggingface/leaderboards/blob/main/docs/source/en/open_llm_leaderboard/archive.md>
- v1 当前公开详情：<https://huggingface.co/datasets/open-llm-leaderboard-old/results>
- 复现：`python "Q4/06_历史快照同口径审计_20260926/02_代码/q4_historical_snapshot_audit.py" --verify-remote`

本附件只修正来源追溯并核算同源月度 P90，不改变既有模型、训练截点、预测门槛或冻结结论。
