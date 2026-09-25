# Q3 本轮可识别结果包（2026-09-25）

本快照整合 T1/M3/M4、T2、T3 和 T4 的当前正式结果。结论范围是固定 Q₀ 与冻结 M0 下的 B1 观测 N/D 支持域；T4 是条件敏感性。完整经验 Q/p 联合优化仍未识别，因此本包不是完整 Q3 联合最优结果，也不是竞赛论文提交包。

## 入口

- `q3_final_report.md`：本轮结果报告。
- `q3_master_table.csv`：按预算×上下文键整合的 15 行 T1/T2/T3/T4 汇总。
- `figure_manifest.csv`、`table_manifest.csv`：包内图件、表格路径与 SHA-256。
- `reproduction_manifest.md`：源命令、环境、哈希和复现顺序。
- `frozen_numbers.json`：由源 CSV 直接汇总的锁定数字。
- `scope_decision_ledger.md`：本轮决策和适用范围。
- `QA/P2_independent_release_receipt.md`：本轮整包独立范围 P2 复核回执（PASS）。
- `QA/`：限定范围的完整性、一致性、可复现性和独立复核记录。

## 分区

- `01_fixed/`：固定 Q 主结果、成本/预算审计、支持表和 T1 图件。
- `02_robustness/T2/`、`02_robustness/T3/`：近优/Pareto/边际收益、Bootstrap 稳健性和切换结果。
- `03_conditional/`：条件 βQ 阈值、区间结果和图件。
- `code_sources/`：项目内生成/整理这些产物的脚本快照。

复现需要在项目根目录使用清单中的命令，并保留 Q1/Q2 上游输入文件。快照保存了直接用于核对和解释的结果文件，但不复制完整 Q1/Q2 原始数据集。
