# Q2 交付物索引

本目录汇总 Q2 阶段性交付物，便于集中查看。原始工作区目录 docs/、src/、results/、figures/ 保持原位；本目录中的方案、代码、结果和图表是整理副本。清单中以 `01_` 至 `05_` 开头的路径相对本 `Q2/` 目录；`docs/`、`src/`、`results/`、`中文题目/` 路径和复现命令均相对项目根目录。

## 状态

经典基线 Loss=f(N,D) 已拟合并完成受限验证；公共实验框架已用 B1/M0 演练；B2 Cerebras 半合成压力测试已生成。Q2 配比可行性审计已完成，当前 B1–B5 的 p-only 数据 Gate 为 **FAIL**：没有已核验并联接到 Loss 的数值 p 向量。Q1.3 的 A4/A5 p-only 结果已完成，但其跨尺度预测没有通过，不能替代 B 组逐行配比。Q1.1 两份 30 条评审表已完成有限分析，项目负责人已选择 `q_huber` 并冻结正式接口；人工评审本身没有区分两候选。此外 M2 仍需要独立于 p 的 Q 变化识别依据。

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
- 04_图表/q2_full_answer/：完整回答补充图组，含原始数据、分析过程、结果和总流程图；每图导出 PNG、SVG、PDF。
- 05_审计交付/：B4/B5 专审交付副本、逐组 provenance 待核表和原始论文网址清单；权威版本以 03_结果 中的审计文件为准。

## 关键报告

- 经典基线运行报告：03_结果/经典ScalingLaw基线/v1/q2_scaling_baseline_report.md
- 公共框架 B1 指标：03_结果/公共实验框架/B1/metrics.json
- B2 压力测试报告：03_结果/B2半合成压力测试/v1/b2_cerebras_stress_report.md
- B4/B5 审计交付副本：05_审计交付/README.md
- 论文原始网址：05_审计交付/论文网址清单.md
- p 可行性审计：03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md
- Q2 主模型路线复审：01_方案说明/Q2_主模型路线复审_2026-09-24.md
- Q2 方法与局限性草稿：01_方案说明/Q2_方法与局限性草稿_2026-09-24.md
- B3 逐轨迹收口：03_结果/经典ScalingLaw基线/v1/b3_validation_closeout.md
- p 来源一次性补查：05_审计交付/q2_p_provenance_one_time_supplement.md

## 复现命令

从项目根目录运行：

    "D:\Anaconda\python.exe" src/f_q2_scaling_baseline.py --mode full --seed 20260923
    "D:\Anaconda\python.exe" src/q2_framework_b1_demo.py --splits 5 --bootstrap-resamples 2000
    "D:\Anaconda\python.exe" src/f_q2_b2_semisynthetic.py --seed 20260924
    python src/f_q2_p_feasibility_audit.py

B2 结果为半合成压力测试，不是 Cerebras 原始训练日志或独立外部验证。B2/B3 不作为独立真实验证；B4/B5 的评测口径可比性未证实。B1 只有 8 条独立轨迹，Bootstrap 区间仅作稳定性提示。

## 路线复审（2026-09-24）

原题要求把附件 A 的 Q、p 信息与附件 B 的 Loss 规律通过可检验假设连接，并估计边际效应、弹性与替代关系。当前保留 B1 的 M0；A4/A5 p-only 结果作为来源内证据；B1–B5 的 p Gate 继续为 FAIL，不能声称 B 组 p 效应已识别。B3 已形成单独的逐轨迹收口报告，只支持 Pythia 同来源插值检查。

本路线复审生成时尚未收到评审表；其后两份 30 条评审表完成有限分析，项目负责人选择 `q_huber`，正式接口已生成。人工核查没有选出 q_equal/q_huber 胜者，选择记录为操作性决定。见 [Q1 最终收口审计](../results/q1_final/q1_final_decision.md)、[Q1 交付物总清单](../results/q1_final/Q1交付物总清单.md) 、[Q1 样本级 Q 接口](../results/q1_final/q1_final_quality.csv) 与 [Q1 领域级 Q](../results/q1_final/q1_final_domain_quality.csv)。M2 仍需可识别的独立 Q 变化；B 侧 p Gate 仍为 FAIL。路线、门禁和可写结论见 [Q2 主模型路线复审](01_方案说明/Q2_主模型路线复审_2026-09-24.md)。

## 方案文档副本补充

`01_方案说明/Q2_p配比可行性审计.md` 是项目 `docs/Q2_p配比可行性审计.md` 的副本；审计运行结果见 `03_结果/p配比可行性审计/v1/`。

## M0 边际效应与验证收口

- 边际效应与弹性报告：`03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_report.md`
- B1 逐检查点效应与 Bootstrap 区间：`03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_by_checkpoint.csv`
- 8 条轨迹的尺度摘要：`03_结果/经典ScalingLaw基线/v1/q2_m0_marginal_effects_by_size.csv`
- B2–B5 验证边界：`03_结果/经典ScalingLaw基线/v1/q2_validation_scope_closeout.md`
- 输入输出哈希清单：`03_结果/经典ScalingLaw基线/v1/q2_m0_effects_validation_manifest.json`
- 复现命令：`D:/Anaconda/python.exe src/q2_m0_effects_validation_closeout.py`
- Q2 代码副本：`02_代码/q2_m0_effects_validation_closeout.py`；入口源文件位于项目 `src/`。

以上结果由已冻结 M0 参数及保存的 cluster Bootstrap 样本派生，不重拟合 M0。B4/B5 不与 B1 合并评分；p-only Gate 仍为 FAIL。

## Q2 当前数据版本综合收口

- 阶段性综合结论：`03_结果/综合收口/v1/q2_stage_conclusion.md`
- 按《题目分析报告》§4.2 逐项组织的四项答复（不是四个正式小问）：见阶段结论中的“按题目分析报告 §4.2 的四项答复”，完整推导、图表索引和证据缺口见 `03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`
- A/B 证据矩阵：`03_结果/综合收口/v1/q2_evidence_matrix.csv`
- p/Q 可识别性门禁：`03_结果/综合收口/v1/q2_identifiability_gates.csv`
- Q 独立增量效应门禁说明：`03_结果/综合收口/v1/q2_q_incremental_effect_gate.md`
- 当前模型与模块状态：`03_结果/综合收口/v1/q2_model_status.csv`
- 输入输出哈希清单：`03_结果/综合收口/v1/q2_closeout_manifest.json`
- 复现命令：`D:/Anaconda/python.exe src/q2_build_evidence_closeout.py`
- Q2 代码副本：`02_代码/q2_build_evidence_closeout.py`；入口源文件位于项目 `src/`。

当前数据版本状态为 `FROZEN_WITH_LIMITATIONS`：M0 和验证/识别审计已收口；B 侧 p 门禁为 FAIL、Q 独立增量门禁为 FAIL_CURRENT_EVIDENCE，M1/M2 均为 NOT_ESTIMATED。这不是 p 或 Q 无效的结论；有新的可追溯 B 侧联合数据时可重开门禁。

## Q2 当前数据版本完整收口包

- 最终收口 README：`03_结果/综合收口/v1/README_Q2_FINAL.md`
- 输入说明与冻结输入登记：`03_结果/综合收口/v1/README_inputs.md`、`inputs_manifest.csv`
- 完整收口运行日志：`03_结果/综合收口/v1/run_log.txt`
- 最终包输入/输出哈希与限制：`03_结果/综合收口/v1/q2_final_package_manifest.json`
- M0 四类边际效应/弹性图：`04_图表/q2_m0_effects/`（PNG、SVG、PDF、灰度预览）；图表契约为 `03_结果/综合收口/v1/q2_m0_effects_figure_contract.csv`
- M0 轨迹覆盖分布摘要：`03_结果/综合收口/v1/q2_m0_effects_distribution_summary.csv`；检查点间 Q1/Q3 与 Bootstrap 中位数区间分开列示，口径见 `q2_m0_effects_summary_readme.md`
- 可复现收口命令：

      "D:\Anaconda\python.exe" src/q2_finalize_current_data_package.py

该命令从已冻结的 M0 参数和 Bootstrap 样本重新生成派生效应、验证边界、证据矩阵、门禁与四类效应图，不重拟合 M0、不重新抽样 Bootstrap、不拟合 M1/M2。完整原始输入状态以 `inputs_manifest.csv` 为准；原始数据说明 PDF 的历史路径缺失及基线记录的根目录分析报告版本变化均已明示，不宣称从当前全部原始附件重跑过基线。

## Q2 完整题意补充收口

**数据说明版本声明：**完整回答补充在核验 B6–B10 数据角色时，实际使用 `中文题目/F题/数据说明(无隐藏字段版本）.pdf` 的可见文本。该文件哈希登记在 `03_结果/综合收口/v1/inputs_manifest.csv` 和本补充的 `03_结果/完整回答补充/v1/repro_manifest.json` 中；用途仅限角色核验，不是 M0 拟合输入。本次同步将该版本纳入仓库，供核对哈希和复现角色核验。历史清单中的 `中文题目/F题/数据说明.pdf` 是另一文件，其原路径缺失仍单独记录；无隐藏字段版本不替代历史原件。原始带隐藏字段 PDF 的隐藏文本未读取。

- 证据补充报告：`03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`（同步副本：`docs/q2_full_answer_evidence_supplement.md`）
- 完整回答图表：`04_图表/q2_full_answer/`（现含原图组、M0 弹性剖面、B1–B5 p Gate 审计图及 A 侧跨尺度来源副本；该副本与已有跨尺度图内容重复；图表契约与 QA：`03_结果/完整回答补充/v1/q2_full_answer_figures_contract.csv`、`q2_full_answer_figures_qa.json`）
- B6/B7 半合成 Q 响应、多初值诊断及留一 Q 水平结果：`03_结果/完整回答补充/v1/b6_b7_q_model_supplement.csv`
- B6/B7 半合成 Q→N 等 Loss 数值示例：`03_结果/完整回答补充/v1/b6_b7_q_n_equal_loss_demo.csv`
- B6–B8 Q 方向与 B7/B6 重叠审计：`03_结果/完整回答补充/v1/b6_b8_q_direction_audit.csv`
- B9/B10 大尺度情景预测及汇总：`03_结果/完整回答补充/v1/b9_b10_m0_extrapolation.csv`、`b9_b10_extrapolation_summary.json`
- A4/A5 二次 p-only 域转移互补性探索分析：`03_结果/完整回答补充/v1/a_side_complementarity/a4_a5_transfer_pair_complementarity_report.md`、逐转移对 Bootstrap 区间和目标汇总表；结果仅限 A 侧，候选筛选未校正，不能外推为 B 侧 p 效应
- M1/M2 不可识别性的代数证明与重开数据条件：`03_结果/完整回答补充/v1/q2_m1_m2_identifiability_proof.md`（同步副本：`docs/q2_m1_m2_identifiability_proof.md`）
- 输入/输出哈希及角色边界：`03_结果/完整回答补充/v1/repro_manifest.json`
- B1 M0 参数、边际效应及 B2–B5 验证原始收口材料副本：`03_结果/完整回答补充/v1/reference_evidence/`
- Q1.3 的关键结果表与报告只读副本：`03_结果/完整回答补充/v1/q1_source_evidence/`（哈希同时记录在上级清单中）
- 复现入口：`D:\Anaconda\python.exe src/q2_full_answer_supplement.py`；代码副本：`02_代码/q2_full_answer_supplement.py`、`02_代码/q2_a_side_complementarity_sensitivity.py`、`02_代码/q2_full_answer_figures.py`（主入口会复算 500 次配方组 Bootstrap 并重新导出图表）

此补充扩展解析边际效应、A 侧配比替代统计、半合成 Q 响应与 B9/B10 外推情景；不重拟合 M0。B6/B7 只作半合成条件响应，B8 的 Q 方向相反而未合并，B10 为估算 Loss 且远超 B1 支持域。B 侧真实 p-Loss 行级连接仍为 0，因此不能声称完整联合 `Loss=f(N,D,p,Q)` 已估计或验证；当前状态见该补充包的机器清单。
