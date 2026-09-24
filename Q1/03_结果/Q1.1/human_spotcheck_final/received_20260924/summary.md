# Q1.1 30 条双人盲评回收分析

## 数据检查

- 两份表均为两位人工评审填写（按项目负责人说明）；每份 30/30 条完整、评分均为 1–5 整数，ID 完全匹配且无重复。
- Reviewer A/B 的行序相同：`True`；A、B 均匹配 reviewer1 模板：`True`；B 匹配 reviewer2 模板：`False`。因此独立随机顺序要求未满足。
- 本次文本来自中文机译辅助评审；结论限于机译版本上的评分可靠性核查，不等同于直接核验英文原文。

## 一致性与候选比较

- 整体质量二次加权 Cohen Kappa：0.765146。
- 整体质量 Spearman：0.805995；完全同分比例：0.500000；相差不超过 1 分比例：0.966667。
- 24 条 representative：Spearman(q_equal, 共识)=0.566692；Spearman(q_huber, 共识)=0.566692；点差=0.000000。
- 配对案例 bootstrap：2000/2000 个有效重复；q_equal、q_huber、相关差的 95% 区间分别见 `paired_bootstrap.csv`。相关差区间 [-0.058987, 0.062315] 跨 0，不能区分候选。
- 这组区间按最新流程要求补算，仅作 24 条小样本的描述性辅助结果。q_equal 和 q_huber 的单独区间半宽均超过 0.15；由于抽样清单未保存设计权重，不能用本次普通配对 bootstrap 替代方案验收段要求的设计加权 bootstrap。

## 选择状态与限制

- 选择状态：保留 `q_equal` 与 `q_huber`，`candidate_selected` 留空。q_equal 与 q_huber 在 24 条代表样本上的点相关相同，配对 bootstrap 的相关差区间跨 0；因此保留两个候选，不强行选择。 两份回收表行序相同且都对应 reviewer1 模板，须在结论中记录顺序独立随机化未满足。
- 门槛核对：30 条方案存在口径冲突：分析步骤写明不增加 bootstrap/区间，验收段又要求 2,000 次分层配对 bootstrap、复用设计权重，并以 95% 区间半宽 0.15 和有效重复率 95% 作门槛。本轮按最新流程附加的普通配对案例 bootstrap 不是该设计加权方法；且抽样清单未提供纳入概率/设计权重，因此不能据它宣称预注册门槛通过。
- 本项完成的是 A1 可读长度子集 30 条文本的双人评分核查；不代表 793 条全量人工验证，也不补足 A2/A3 原文映射缺口。
- 六条 diagnostic 仅作案例表，不并入 24 条总体相关；见 `diagnostic_cases.csv`。

## 输出

- `human_consensus.csv`
- `human_check_metrics.csv`
- `candidate_human_comparison.csv`
- `paired_bootstrap.csv`
- `diagnostic_cases.csv`
- `decision_status.json`
