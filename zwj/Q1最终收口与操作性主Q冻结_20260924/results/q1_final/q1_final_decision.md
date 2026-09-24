# Q1 操作性主 Q 决策记录（待负责人填写）

## 当前阶段状态

- `status = PENDING_PROJECT_OWNER_DECISION`
- `operational_q = null`
- `formal_interface_generated = false`
- `PRESET_RULE_FOUND = true`
- 预设规则当前可执行到唯一候选：否；双评表已齐，但最小化核查未提供原方案的 bootstrap 门禁，且 24+6 分组清单仍缺
- 人工核查区分候选：false（任务说明结论；本地原始输出待补核）
- Q1.3 重跑：false

## 负责人决策栏

以下内容必须由项目负责人明确后才能进入第二阶段：

- 操作性主 Q：`q_equal` / `q_huber`
- 决策人：
- 决策日期：
- 选择依据：`project_operational_choice`，或在补齐统计证据后按 `preset_rule`
- 确认另一候选只作为敏感性对照：是 / 否
- 确认不把操作性选择描述为人工评审唯一胜出：是 / 否

## 第二阶段动作

负责人决定后，从原始 `sample_scores.csv.gz` 逐行提取被选候选，生成：

- `q1_final_quality.csv`：只含唯一正式 Q；
- `q1_candidate_audit.csv`：保留两个候选用于审计；
- 更新后的 `q1_final_freeze.json` 与 `reproduction_manifest.json`。

禁止重新评分、重估权重、改变候选分、把 Q1.2 算术加入 Q，或未经批准重跑 Q1.3。
