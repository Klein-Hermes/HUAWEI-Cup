# BBH 逐任务聚合影响分析

**修订状态：** 属于 Q4 继续推进工作包，修订版尚未封包或经队伍复审。

## 分析范围与方法

- 读取历史 Q4 的逐模型逐叶任务得分、逐族聚合分数和六族主指数；输入表没有被改写。
- 分析固定六族完整样本中的 **789** 个模型及 BBH **24** 个 `acc_norm` 叶任务。完整六族既定指数与六族得分等权平均的最大差为 **2.84e-14**。
- 对每个 BBH 叶任务依次执行一次 leave-one-task-out：保留其他 23 个任务的逐任务百分位平均，等权替换 BBH 族分，再与其他五族重新计算六族指数。百分位沿用历史 C8 候选池计算值，不在子样本中重排。
- 排名完全并列时，按模型名称升序稳定打破并列。任务难度表使用同一完整六族样本中的原始 `acc_norm` 分数中位数与四分位数。

## 结果

- 逐个剔除任务后，Spearman 名次相关的最小值为 **0.9999**（剔除 `leaderboard_bbh_tracking_shuffled_objects_three_objects`），24 项诊断的中位数为 **1.0000**。
- 基线 Top-100 在逐任务剔除后的最小保留率为 **99.0%**（剔除 `leaderboard_bbh_geometric_shapes`）。
- 六族指数点数变化的最大绝对值为 **0.5719**，对应剔除 `leaderboard_bbh_tracking_shuffled_objects_three_objects`。
- 最大单模型名次位移为 **13** 位，发生在剔除 `leaderboard_bbh_tracking_shuffled_objects_three_objects`；模型 `someon98/qwen-CoMa-0.5b`。
- 在完整六族样本中，BBH 任务原始分数中位数最低/最高的叶任务分别是 `leaderboard_bbh_tracking_shuffled_objects_seven_objects`（0.1560）和 `leaderboard_bbh_boolean_expressions`（0.8160）。此处“难/易”只表示本队列的描述性中位数，不表示题目难度的普遍属性。

## 可复核文件

- `q4_bbh_task_leave_one_out.csv`：24 次任务剔除对应的排序与指数变化诊断。
- `q4_bbh_model_rank_stability.csv`：每个纳入模型跨 24 次剔除的名次范围与最大位移。
- `q4_bbh_task_difficulty_summary.csv`：逐任务有效数及队列原始分数分布。
- `q4_bbh_task_influence_manifest.json`：输入哈希、运行环境、参数和输出哈希。

## 限制

该分析是确定性敏感性检验，不是置信区间或总体抽样不确定性估计；一次只剔除一个 BBH 叶任务，不能代替更广泛的基准有效性验证。模型名称连接仍不确认版本身份。该分析不解除 Loss–Benchmark 拟合或 12/24 月预测的数据门控，也不把描述性差异解释为因果作用。
