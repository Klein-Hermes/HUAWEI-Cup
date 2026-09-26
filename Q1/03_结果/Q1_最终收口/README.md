# Q1 最终收口交付包

本目录是项目根目录 `results/q1_final/` 的 Q1 文件夹镜像，便于集中查阅。主 Q 已按项目负责人选择冻结为 `q_huber`，`q_equal` 保留为敏感性对照；该选择是操作性决定，不是 30 条盲评选出的胜者。最终状态为 `FROZEN_WITH_LIMITATIONS`。

按题面三项任务撰写的集中答案见上级目录[《问题一答案》](../Q1答案.md)；本目录主要保存正式 Q 接口、冻结决策和复现审计材料。

## 文件

- `q1_final_quality.csv`：272,505 条逐样本正式 Q 接口。
- `q1_candidate_audit.csv`：两种候选分、覆盖和审计字段。
- `q1_final_domain_quality.csv`：9 个领域的汇总分和区间。
- `q1_final_decision.md`：收口判断、评审结果和限制。
- `q1_q13_consistency_check.md`：Q1.3 输入依赖与是否需要重跑的核验。
- `Q1交付物总清单.md`：Q1.1、Q1.2、Q1.3 及公共预处理的材料索引。
- `export_q1_final_interface.py`：可复现导出脚本；请从项目根目录运行根目录 `results/q1_final/export_q1_final_interface.py`，不要从本镜像副本运行。
- `q1_final_freeze.json`、`reproduction_manifest.json`：冻结状态和原始生成源的审计记录。
- `mirror_manifest.json`：本目录镜像文件的本地哈希。

数据污染和溯源限制仍按原报告披露；`题目分析报告.md` 未被本次修改。镜像文档中的相对链接已按本目录位置修正，根目录 `results/q1_final/` 保留原始生成版本。
