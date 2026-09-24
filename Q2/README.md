# Q2 交付物索引

本目录汇总 Q2 阶段性交付物，便于集中查看。原始工作区目录 docs/、src/、results/、figures/ 保持原位；本目录中的方案、代码、结果和图表是整理副本。复现请在项目根目录使用原路径和下方命令。

## 状态

经典基线 Loss=f(N,D) 已拟合并完成受限验证；公共实验框架已用 B1/M0 演练；B2 Cerebras 半合成压力测试已生成。Q2 配比可行性审计已完成，当前 B1–B5 的 p-only 数据 Gate 为 **FAIL**：没有已核验并联接到 Loss 的数值 p 向量；Pythia 配方重建和 B4/B5 文献级追溯仍是条件性后续工作。完整广义 Q2 尚未完成，也没有 Q2 的 M1 拟合器；M2 还需冻结版本的最终质量分数 Q。

## 目录

- 01_方案说明/：Q2 工作包、公共实验框架、B2 压力测试规则及题目分析报告。
- 02_代码/：经典基线、公共实验框架、B1 演练、B2 生成器、p 可行性审计脚本和现有测试脚本。
- 03_结果/经典ScalingLaw基线/v1/：Gate 0 审计、拟合参数、B1 与 B2–B5 预测、验证指标、Bootstrap、留一规模结果、复现清单和预览图。
- 03_结果/p配比可行性审计/v1/：p Gate 审计报告、B1–B5 轨迹/记录映射、来源证据目录、审计摘要和输入/输出哈希清单。
- B4/B5 Loss 口径专审：`03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md`、`loss_protocol_matrix.csv`、`loss_protocol_evidence_sources.md`。当前所有来源组相对 B1 均为 `NOT_COMPARABLE`；没有 `ABSOLUTE_COMPARABLE`。B5 来源内趋势仍须逐点定位后才能升级；B4 Cerebras 与 GPT-3 来源只登记候选/未决，不修改原始数据。
- 对照矩阵和逐组来源待核记录：`03_结果/经典ScalingLaw基线/v1/loss_protocol_matrix.csv`、`b4_b5_provenance_review.csv`；原始论文网址清单：`03_结果/经典ScalingLaw基线/v1/论文网址清单.md`。
- 03_结果/公共实验框架/：B1 五折整轨迹演练的切分、配置、预测、指标、Bootstrap 和复现清单。
- 03_结果/B2半合成压力测试/v1/：合成轨迹、情景指标、锚点核对、运行报告和生成清单。
- 04_图表/q2_scaling_baseline/：经典基线图表，每张图均有 PNG 与 SVG。
- 05_审计交付/：B4/B5 专审交付副本、逐组 provenance 待核表和原始论文网址清单；权威版本以 03_结果 中的审计文件为准。

## 关键报告

- 经典基线运行报告：03_结果/经典ScalingLaw基线/v1/q2_scaling_baseline_report.md
- 公共框架 B1 指标：03_结果/公共实验框架/B1/metrics.json
- B2 压力测试报告：03_结果/B2半合成压力测试/v1/b2_cerebras_stress_report.md
- B4/B5 审计交付副本：05_审计交付/README.md
- 论文原始网址：05_审计交付/论文网址清单.md
- p 可行性审计：03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md

## 复现命令

从项目根目录运行：

    "D:\Anaconda\python.exe" src/f_q2_scaling_baseline.py --mode full --seed 20260923
    "D:\Anaconda\python.exe" src/q2_framework_b1_demo.py --splits 5 --bootstrap-resamples 2000
    "D:\Anaconda\python.exe" src/f_q2_b2_semisynthetic.py --seed 20260924
    python src/f_q2_p_feasibility_audit.py

B2 结果为半合成压力测试，不是 Cerebras 原始训练日志或独立外部验证。B2/B3 不作为独立真实验证；B4/B5 的评测口径可比性未证实。B1 只有 8 条独立轨迹，Bootstrap 区间仅作稳定性提示。
