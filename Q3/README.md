# Q3 交付物索引

> Q1–Q4 历史遗留问题的统一当前状态见[总表](../docs/q1_q4_legacy_issue_register_20260926.md)。

## 当前交付状态（2026-09-26 G2 增补）

T2、预算饱和与有限网格切换，以及新增的 M0 函数形式/整轨迹留一敏感性，均已纳入本轮可识别结果报告。当前基线版本入口为 [rev2 独立复核结果包](冻结包_20260925_模型形式敏感性_独立复核/README.md)，其 [整包 P2 回执](冻结包_20260925_模型形式敏感性_独立复核/QA/P2_independent_release_receipt.md) 已通过。2026-09-26 另完成固定-Q 分支的 G2 有限网格预算限制状态判定，经过独立 M1、P1 与结果 P2；增补产物位于 `04_结果/G2_结构性转移判定_20260926/`，主结果报告已整合。rev2 冻结包作为历史基线保留，没有被覆盖；完整经验 Q/p 联合优化仍未识别，不纳入本次有限范围判定。

项目级限定范围 QA 结果和当前门禁见 [`04_结果/Q3_预冻结QA记录.md`](04_结果/Q3_预冻结QA记录.md)。rev2 的限定范围静态核对和独立 P2 均已完成；没有活动的 `rigor_profile`，所以不把包内 QA 标记为原生 submission 完整性审计或完整竞赛提交 QA 通过。

## Q3 直接答案

当前工作结果报告已把数值结论放在正文开头：[Q3 本轮可识别结果范围报告](04_结果/q3_final_report.md#q3-直接答案当前证据支持范围)。固定 $Q=Q_0$、C7 架构支持上下文长度为 2,048 tokens 时，$10^{19}$、$10^{22}$、$10^{24}$ FLOPs 对应的 $(N^*,D^*)$ 分别为 $(0.162405,8.389)$B、$(6.86104,226.492)$B、$(11.965825,299.893)$B。该架构支持长度不必然等于任务实际使用的窗口。题面代理成本的临界上下文长度为 30,000 tokens。以上是 B1 实测 N/D 网格上的参考配置；完整经验 $Q/p$ 联合最优仍未识别。rev2 冻结包保留为已复核快照，新增 G2 判定和本段直接答案以工作结果报告为入口。

## 目录与版本导航

- **唯一当前冻结入口：**[`冻结包_20260925_模型形式敏感性_独立复核/`](冻结包_20260925_模型形式敏感性_独立复核/README.md)，rev2，整包独立范围 P2 PASS。
- **历史冻结快照（保留原路径，不再作为当前结果入口）：**`冻结包_20260925/` 是早期整合版；`冻结包_20260925_有限网格预算分析/` 是预算增补版；`冻结包_20260925_有限网格预算分析_独立复核/` 是该增补版的独立复核版。各包均保留自己的报告、manifest 和回执。
- **工作目录：**`01_方案与设计/`、`02_数据审计/`、`03_代码/`、`04_结果/` 按方案、输入审计、代码和权威结果分层。`04_结果/_q3_fixedq_p1_smoke/` 与专题目录中的 `_figure_previews/` 是复现清单追踪的 QA 输入，不是论文图件。
- **补充图：**`04_结果/figures/supplementary_20260925/` 是图集入口；根目录 `补充图_候选20260925/` 保留 S1–S3 的绘图源文件，图集 manifest 仍引用该路径。
- **历史材料：**`归档_20260925/` 的内容索引见 [`归档_20260925/README.md`](归档_20260925/README.md)。本次目录审查记录见 [`整理记录_20260925.md`](整理记录_20260925.md)。版本包与来源路径由清单/历史计划引用，暂保留原位。

清单中的 `01_` 至 `04_` 路径均相对本 `Q3/` 目录；`../中文题目/` 指向项目根目录下的原始附件；复现命令从项目根目录运行。Q1 的正式操作质量接口已经冻结。Q3 已完成 C7 审计、固定 Q 的 Q2-M0 N/D 支持网格参考前沿、轨迹 Bootstrap 敏感性、质量成本曲线、情景变化描述、Q-only 条件优化、M3/M4 掩码与 M1 求优、M3/M4 独立成本与可行性审计，以及离散资源转移和 Q0 基线敏感性。上述固定 Q 审计复用已归档的 Q1/Q2 结果，不等待新增 Q2 数据；完整 Q/p 联合优化仍需 Q2 的配比与独立质量效应数据。

## 当前上游状态

- **Q1：带限制冻结完成。** 操作主 Q 为 `q_huber`，版本 `Q1-q_huber-v1`；`q_equal` 是敏感性分。负责人选择不等同于人工评分判出胜者。译文辅助评审、回收顺序、独立评分过程及 A2/A3 映射等限制见 Q1 收口审计。
- **Q3 质量成本基线：**A1 七个来源域正式 q_huber 域分的等权宏平均为 $Q_0=0.4984781048$，对应 95% 区间 [0.496567, 0.500427]。这是 Q3 成本参考基线，不代表 B1/Pythia 配方的实测混合质量。`q_equal` 对应宏平均 0.4962230100，只作敏感性对照。
- **Q2：**B1 的经典 `Loss=f(N,D)` M0 已拟合。B1 有 8 个实际参数规模与共同的 147 个训练 Token 检查点，组成 1,176 个 N/D 支持网格点。
- **Q2 的完整 Q/p 响应仍未就绪。** B1–B5 已核验并连接到 Loss 的数值配比向量为 0，p Gate 为 FAIL；当前不能对配比或质量对 Loss 的独立效应作经验优化。
- **C7：**45 条模型架构上下文记录审计通过，形成 5 种离散长度。`max_position_embeddings` 是架构支持上限，不一定等于题目实际部署窗口。

## 已推进：M0 支持网格参考前沿

固定正式 q_huber 基线 $Q=Q_0$，因此 $C_Q=0$。对每个预算与 C7 上下文长度，筛选满足 $C_{\mathrm{train}}+C_{\mathrm{attn}}\le C$ 的 B1 实测 $(N,D)$ 组合，并以冻结的 Q2 M0 预测 Loss 选取最优网格点。该分支给出 B1/M0 支持范围内的 N/D 预算参考，不估计 Q/p 效应，也不称为完整 Q3 联合最优解。

15 个预算×上下文情景全部有可行点。$10^{24}$ FLOPs 档在五种上下文下均让 1,176 个网格点可行，选中点到达 B1 网格上界，预算利用率约 2.3%–11.6%。这表明该档结果受现有 B1 支持范围限制，不能外推为更大 N/D 的结论。

已将 Q2 对八条 B1 轨迹的 1,000 次 cluster Bootstrap 参数逐次传播到上述 15 个情景，并重新选择可行网格点。当前主分析选中点在 15 个情景中均于 1,000 次重抽样下保持不变；这只表示既定 M0 函数形式和 B1 支持网格内的参数重抽样稳定性，不覆盖模型形式误差、外部迁移或 Q/p 响应。新增代码与结果已通过独立 P1 复核。

另已按题面三种 $g(Q)$ 计算从冻结 $Q_0$ 到 $Q=1$ 的纯质量成本曲线，并用现有 B1 D 支持范围标出增量成本；同时将现有 15 个情景展开为 15 行成本份额和 22 对相邻预算/上下文变化。该分支只回答算力成本和情景间配置如何变化，不估计质量收益、不作正式结构转移判断。它仅依赖既有冻结接口、题面公式、已归档 B1 网格和前沿结果，不等待新增 Q2 数据；独立 P1 复核通过。

另已构造一个 Q-only 参数化条件模型：$L_{cond}=L_{M0}(N,D)-\beta_Q(Q-Q_0)$，其中 $\beta_Q\ge0$ 是显式未知的质量响应系数；p 不进入模型，也不作为决策变量，本分支未设置配比向量。对三种 $g(Q)$、五种上下文和三档预算，报告 $\beta_Q$ 下的最优区间及 N/D 首次重分配阈值；共 45 个条件情景和 166 个下包络区间。该模型不为 $\beta_Q$ 编造估计值，也不是经数据识别的 Q 最优。新增结果的独立 P1 复核通过。

## 新增阶段推进（2026-09-24）：离散转移审计与 Q0 基线敏感性

这两项工作只使用已归档的 Q1 冻结接口、Q2 B1/M0 网格与既有轨迹 Bootstrap 参数，不等待新增 Q2 数据。它们推进的是固定 $Q=Q_0$ 的 M0 参考支路和条件敏感性，不解除 Q2 的 p/Q 识别门槛。

### 固定 Q 的 M0 离散资源转移审计

将 3 档预算与 5 档 C7 上下文形成的 15 个情景按相邻预算、相邻上下文配成 22 对；每对使用同一组 1,000 个 Q2 轨迹 Bootstrap 复本，在 B1 实测的 1,176 个 $(N,D)$ 点内重新选择配置。用“至少 95% 配对复本改变 N/D 网格点”标记配置改变；成本份额则单独按精确总变差是否非零进行判断。

- 17/22 对达到 N/D 配置稳健改变门槛；12/22 对出现稳健成本构成改变。
- 10 对相邻预算比较均改变 N/D 配置，但成本份额不变；上下文变化的成本构成均发生变化，其中 7/12 对同时改变 N/D 配置。
- 这是冻结 M0 函数和 B1 离散支持域内的情景敏感性，不是因果结论、连续预算转折点或完整 Q3 的结构转移；最高预算仍有网格上界约束。因当前是有限网格枚举，使用离散选择与成本份额判据，不套用连续求解器 KKT 判据。

### Q0 基线敏感性

对照 `q_huber` 点值、A1 七域宏平均 95% 区间两端，以及独立的 `q_equal` 敏感性分支，重算三种 $g(Q)$ 下的 Q 成本与原 Q-only 条件模型中的 N/D 重分配阈值。产出 141 条成本记录和 180 条条件阈值记录。`q_equal` 不是置信区间端点；A1 区间也不是 B1 混合质量的不确定区间。$\beta_Q$ 仍未识别，因此这些阈值只是条件敏感性，不是经验质量收益。

逐对转移结果见 `04_结果/M0_discrete_transition_report.md`；基线敏感性结果见 `04_结果/Q3_Q0_baseline_sensitivity_report.md`；本轮任务、当前可推进范围和后续门槛汇总见 `04_结果/Q3_阶段推进总结.md`。两项分析各自的复现命令和输入输出哈希记录在对应 manifest 中。

## 后续计划

1. **已完成：**冻结 Q1 `q_huber` 接口与 B1 支持网格；生成预算×上下文参考配置、摘要、报告和复现清单；将既有轨迹 Bootstrap 传播到 15 个情景，评估参考配置稳定性。M1 模型契约审查、M0 前沿计算及 Bootstrap 敏感性产物的独立 P1 复审均通过。
2. **已完成并通过独立 P1：**计算题面三种 $g(Q)$ 的增量质量成本曲线，并比较现有 15 个预算×上下文场景中的成本份额和相邻配置变化；该分析不需要新增 Q2 数据，结果仅作成本敏感性和描述性比较。
3. **已完成并通过独立 P1：**建立 p 不进入模型的 Q-only 参数化条件优化，以未知 $\beta_Q$ 做分段最优与 N/D 重分配阈值分析。它不需要新增 Q2 数据，明确作为条件情景，不替代 Q2 的实证响应面。
4. **已完成并通过独立 P1：**在 22 对相邻预算/上下文情景上做固定 Q 的 M0 离散资源转移审计。该结果仅适用于 B1 实测支持网格和冻结 M0；详见 `04_结果/M0_discrete_transition_report.md`。
5. **已完成并通过独立 P1：**传播 Q1 `q_huber` 的 A1 区间端点并单列 `q_equal`，重算 Q 成本曲线及 Q-only 条件重分配阈值；详见 `04_结果/Q3_Q0_baseline_sensitivity_report.md`。
6. **Q2 联合响应门槛：**若要解除完整 Q/p 优化限制，需要可追溯的 B 侧配比 p、Loss 行连接、配比独立变化，以及独立于 p 的 Q 变化证据；之后才能锁定 Q/p 响应面、适用域和参数不确定性。
7. **完整场景求解（门槛满足后）：**在证据支持的 N/D/Q/p 域内，交叉三档预算、五种 C7 长度和三种 $g(Q)$ 形式；`q_huber` 为正式主分，`q_equal` 作为质量成本敏感性分支。
8. **联合模型验证与归档（门槛满足后）：**做可行性预扫、收敛与边界诊断、bootstrap 稳健性和完整模型结构转移判定；交付可运行求解代码、情景表、结果报告、图表和复现清单。当前 M0 前沿不替代完整模型的结构转移判定。
9. **已完成并通过独立 M1/P1/P2：**按题目要求，在固定 $Q=Q_0$ 的 1,176 点 B1/M0 支持域内定义并判定预算限制状态转换。五个上下文均在 $10^{22}\to10^{24}$ FLOPs 间发生 1→0 转换，配对参数 Bootstrap 状态率为 1,000/1,000；$10^{19}\to10^{22}$ 仅有 N/D 配置切换，状态率为 0/1,000。该增补关闭 G2 在当前可识别固定-Q分支内的判定缺口，不替代完整 Q/p 联合模型结论。方法、脚本、结果与哈希见 [G2 判定计划](01_方案与设计/Q3_G2结构性转移判定计划_20260926.md)、[G2 判定报告](04_结果/G2_结构性转移判定_20260926/report.md) 和 [完整复算脚本](03_代码/analyze_q3_g2_structural_transition.py)。

## 现有交付物

- `01_方案与设计/题目分析报告.md`：Q3 任务、冻结 Q1/Q2 接口、M0 支持网格分支与完整联合优化门槛。
- `01_方案与设计/术语表格.md`：Q3 符号、单位和决策域。
- `02_数据审计/C7上下文长度审计报告.md`：C7 校验、长度分布、临界长度和解释限制。
- `02_数据审计/C7_context_by_model.csv`：45 个模型的上下文长度明细及相对临界长度。
- `../中文题目/F题/real_attachments/C_efficiency_evolution/model_architecture_metadata.csv`：C7 审计的原始架构元数据输入。
- `03_代码/audit_q3_c7_contexts.py`：标准库 C7 审计脚本。
- `03_代码/solve_q3_m0_reference_grid.py`：读取 Q1 冻结接口、Q2 M0 和 B1 实测网格，生成固定 Q 的预算参考配置。
- `03_代码/assess_q3_m0_cluster_bootstrap.py`：传播 Q2 既有的 1,000 次轨迹 cluster Bootstrap 拟合，重新评估 15 个参考配置的选择稳定性。
- `03_代码/analyze_q3_cost_sensitivities.py`：计算题面三种质量成本曲线、固定 Q 情景成本份额及预算/上下文相邻配置变化。
- `03_代码/solve_q3_conditional_q_sensitivity.py`：在显式线性 Q→Loss 假设下，不建模 p，并计算不同未知响应系数对应的条件最优区间。
- `03_代码/assess_q3_m0_discrete_transitions.py`：在固定 Q 的 M0/B1 离散支持网格上，对 22 对相邻预算/上下文情景做配对轨迹 Bootstrap 转移审计。
- `03_代码/analyze_q3_q0_baseline_sensitivity.py`：传播 Q0 基线端点和 `q_equal` 分支，重算质量成本及 Q-only 条件重分配阈值。
- `04_结果/C7_context_audit_summary.json`、`04_结果/复现清单.json`：C7 摘要及复现记录。
- `04_结果/M0_ND_reference_frontier.csv`：15 个情景的配置、Loss、成本分解、预算利用率和可行候选数。
- `04_结果/M0_ND_reference_frontier_report.md`：M0 支持前沿的方法、结果和解释边界。
- `04_结果/M0_ND_reference_frontier_summary.json`、`04_结果/M0_reference_reproduction_manifest.json`：机器可读结果与输入/输出哈希。
- `04_结果/M0_ND_cluster_bootstrap_report.md`：轨迹 Bootstrap 敏感性方法、15 个情景区间及解释边界。
- `04_结果/M0_ND_cluster_bootstrap_frontier.csv`、`04_结果/M0_ND_cluster_bootstrap_selection_frequency.csv`：情景级区间摘要和逐配置选择频率。
- `04_结果/M0_cluster_bootstrap_sensitivity_manifest.json`：该敏感性分析的复现信息与哈希。
- `04_结果/Q3_cost_sensitivities_report.md`：质量成本曲线与情景间配置/成本变化的解释报告。
- `04_结果/Q_cost_increment_envelope.csv`：12 个 Q 场景、三种成本函数及 B1 D 支持范围的增量质量成本。
- `04_结果/M0_budget_context_cost_shares.csv`、`04_结果/M0_budget_context_descriptive_shifts.csv`：15 个情景成本份额与 22 对相邻情景的描述性变化。
- `04_结果/Q3_cost_sensitivities_manifest.json`：输入/输出哈希和数据依赖说明。
- `04_结果/Q3_conditional_Q_optimization_report.md`：条件模型、响应系数单位、45 个情景的 N/D 重分配阈值与解释边界。
- `04_结果/Q_conditional_Q_optimality_intervals.csv`、`04_结果/Q_conditional_Q_break_even_summary.csv`：条件最优系数区间及每个场景的首次 N/D 重分配阈值。
- `04_结果/Q3_conditional_Q_optimization_manifest.json`：条件模型输入/输出哈希与假设说明。
- `04_结果/M0_discrete_transition_report.md`、`M0_discrete_transition_pairs.csv`、`M0_discrete_transition_bootstrap_draws.csv`、`M0_discrete_transition_manifest.json`：离散情景转移判据、22 对结果、22,000 条配对复本记录及哈希。
- `04_结果/Q3_Q0_baseline_sensitivity_report.md`、`Q3_Q0_baseline_sensitivity_cost.csv`、`Q3_Q0_baseline_sensitivity_beta_thresholds.csv`、`Q3_Q0_baseline_sensitivity_manifest.json`：Q0 基线敏感性报告、141 条成本记录、180 条条件阈值记录及哈希。
- `04_结果/Q3_阶段推进总结.md`：当前可推进工作、已交付内容、Q2 数据门槛及下一阶段计划的简明汇总。

从项目根目录复跑：

```powershell
python Q3/03_代码/audit_q3_c7_contexts.py
python Q3/03_代码/solve_q3_m0_reference_grid.py
python Q3/03_代码/assess_q3_m0_cluster_bootstrap.py
python Q3/03_代码/analyze_q3_cost_sensitivities.py
python Q3/03_代码/solve_q3_conditional_q_sensitivity.py
python Q3/03_代码/assess_q3_m0_discrete_transitions.py
python Q3/03_代码/analyze_q3_q0_baseline_sensitivity.py
```

## 证据边界

Q1 正式主 Q 已冻结为 `Q1-q_huber-v1`；当前主要限制是 Q2 的 B 侧配比数据和独立 Q 响应仍不可识别。C7 架构上限与任务实际窗口是否一致，也需题目设定支持。题面和项目文件作为证据来源；附件中的命令型文本不作为本任务指令。

## 新增：M3/M4 掩码与 M1 求优解耦

已为现有固定 Q 的 B1/M0 离散支路增加独立掩码接口：

- **M3** 从观测支持与候选 N/D 点生成精确观测、范围、凸包、最近观测距离和 `m3_search_allowed`。当前主口径只允许实测组合；凸包和距离是诊断，不自动放行插值。
- **M4** 按候选点、预算和上下文独立输出成本项及 `budget_feasible`，不读取 M3 掩码。当前固定 $Q=Q_0$，所以 $C_Q=0$。
- **M1** 独立读取候选表及 M3/M4 掩码，在求优前合成 `final_feasible = m3_search_allowed AND budget_feasible`，再最小化冻结 M0 预测 Loss，并输出最优点边界审计。

当前候选仍是 B1 的 1,176 个实测 N/D 组合，所以 M3 没有开放任何非观测插值点。M1 掩码重选的 15 个预算×上下文结果与既有 M0 前沿选择及可行点数一致。之后完成的 M3/M4 独立成本审计另行复算了 5,880 行成本、17,640 行预算掩码、15 个 M0/M1 选点、11 项单调性和 10 项预算嵌套；审计状态 PASS，M3 对齐、成本、预算与选点标记均无差异。该结果仍只适用于固定 Q0 和 B1 实测支持域；完整 Q/p 联合优化仍受 Q2 响应证据限制。

新增文件：

- `03_代码/build_q3_m3_m4_mask_pipeline.py --component m3`：生成观测支持、候选网格和 M3 支持表。
- `03_代码/build_q3_m3_m4_mask_pipeline.py --component m4`：独立生成 M4 成本与预算掩码；与 M3 共用冻结的候选域，可并行运行。
- `03_代码/solve_q3_m1_from_masks.py`：消费掩码运行 M1 并审计最优点边界。
- `04_结果/m3_observed_support.csv`、`m3_candidate_grid.csv`、`m3_q3_support_grid.csv`：M3 输入与支持域掩码。
- `04_结果/m4_budget_feasible_mask.csv`：M4 独立预算可行掩码。
- `04_结果/M3_support_domain_audit.md`、`M3_support_manifest.json`：M3 支持域审计及哈希记录。
- `04_结果/M4_budget_feasibility_audit.md`、`M4_budget_manifest.json`：M4 预算可行性审计及哈希记录。
- `04_结果/m1_final_feasible_mask.csv`、`m1_optimum_boundary_audit.csv`、`M1_masked_optimization_report.md`、`M1_masked_optimization_manifest.json`：M1 合并后的最终可行掩码、选择、边界审计及哈希记录。
- `04_结果/M3_support_audit.md`：汇总 M3 支持域、M4 预算和 M1 求优边界的总审计。
- `04_结果/Q3_M3_M4_M1_任务拆分与并行计划.md`：任务依赖表、并行边界和运行顺序。
- `03_代码/plot_q3_m3_m4_m1_audit.py`、`04_结果/Q3_M3_M4_M1_图表QA.md`、`Q3_M3_M4_M1_图表读图复核.md`、`Q3_M3_M4_M1_图表清单.json`：三张审计图、程序/人工 QA 和输入/输出哈希。
- `04_结果/figures/`：实测支持、M3/M4 掩码、M1 最优点图表，含 PDF、SVG、300 dpi PNG 与灰度预览。
- `01_方案与设计/M3_M4成本独立审计执行计划.md`：用户截图流程的并行性判断、优化任务表、依赖与完成状态。
- `03_代码/audit_q3_m3_m4_cost_outputs.py`：固定 Q0 的 M3/M4 成本与可行性独立复算脚本，包含 P1 小样和全量模式。
- `04_结果/M3_M4成本独立审计/`：独立审计交付目录，包含 5,880 行成本网格、17,640 行预算审计、15 个选点对照、M3 对齐表、单调性与嵌套检查、审计报告、P1 回执、复现 manifest、SVG/300 dpi PNG 和灰度预览；修改前 M1 产物保存在 `upstream_before_fix/` 子目录。

独立审计的全量复现命令（使用已验证的 `mmdet` 环境；完整 Q3/M1 主线输入保持只读）：

```powershell
$env:MPLBACKEND='Agg'
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/audit_q3_m3_m4_cost_outputs.py --smoke
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/audit_q3_m3_m4_cost_outputs.py
```

M3 和 M4 可在两个终端并行运行；两者完成后再运行 M1：

```powershell
python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m3
python Q3/03_代码/build_q3_m3_m4_mask_pipeline.py --component m4
python Q3/03_代码/solve_q3_m1_from_masks.py
```

图表单独复现（建议使用项目现有 `mmdet` 环境；脚本固定 Agg 无窗口后端）：

```powershell
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/plot_q3_m3_m4_m1_audit.py
```

## 本轮新增：固定 Q 正文草稿、P1–P4 补图与 M3/M4 独立审计

- `04_结果/Q3固定Q分支正文草稿.md`：可整合进论文的固定 Q₀/M0 分支章节，含 15 个预算—上下文情景、Bootstrap、离散变化和识别边界。Q3 目录目前没有完整论文稿，因此该文件是章节稿，不代表完整 Q3 论文已经定稿。
- `归档_20260925/Q3补充图表/figures/`：四张补充图 P1–P4；每张均有 SVG、300 DPI PNG 和灰度预览。
- `归档_20260925/Q3补充图表/图表契约.md`、`复现清单.json`：图表口径、来源、输入输出哈希、分辨率与版面核验。
- `04_结果/M3_M4成本独立审计/m4_cost_audit_report.md`：M3/M4 成本与可行性独立审计。5,880 行成本复算、17,640 行预算掩码、1,176 个 M3 键值对齐、15 个 M0/M1 选点、11 项单调性和 10 项预算嵌套检查均通过；完整输入/输出哈希见同目录 `reproduction_manifest.json`。
- 补充图由 `node Q3/归档_20260925/plot_q3_supplementary_figures.mjs` 生成，使用 Node/Sharp，不调用 Python 绘图运行时。已有 M3/M4 审计产物与哈希齐全，无需为查看结果重跑审计。

当前唯一不能进入正式联合拟合的部分仍是 Q/p→Loss：缺少可审计的 B 侧配比独立变化与独立于 p 的 Q 变化证据。正文草稿、图表和成本审计均保持这一边界。

## Q3-T1：固定 Q₀ 主结果与资源转移整合

已完成对 15 个预算×上下文情景和 22 对相邻情景的统一整合。流水线复用冻结的 M0/M1/M3/M4、成本份额和配对 Bootstrap 结果，不重新拟合 M0 或重复抽样；输入 CSV 均按生产 manifest 校验 SHA-256。主表的场景键为预算与上下文，转移表保留 pair_id 和左右场景端点。15/15 个选点与独立成本审计一致，并满足 M3/M4/M1 标记。

- 执行计划：`01_方案与设计/Q3_T1_固定Q主结果整合_优化执行计划.md`。
- 代码：`03_代码/assemble_q3_fixedq_master_results.py`；完整复现命令见结果 manifest，绘图使用项目现有 `mmdet` 环境和 Agg 后端。
- 主结果表：`04_结果/q3_fixedQ_master_results.csv`（15 个场景）。
- 资源转移表：`04_结果/q3_resource_transition.csv`（22 对配对变化）。
- 核心图：`04_结果/figures/result_q3_fixedq_master_and_resource_transition.svg`、`.png`、`.pdf`；灰度预览在 `04_结果/figures/_qa/`。
- 报告、图表契约与输入/输出 SHA-256：`04_结果/Q3_fixedQ_results_integration_report.md`、`Q3_fixedQ_master_resource_transition_图表契约.md`、`Q3_fixedQ_results_manifest.json`。
- `_q3_fixedq_p1_smoke/` 是真实数据的一行样例和 P1 回执，不是完整结果表。该 15 行情景汇总也不能替代 Q3-T2 需要的候选级 1,176 点×情景输入。

该整合仍固定 Q=Q₀、只适用于 Q2 B1 的精确实测 N/D 支持网格，不估计 Q/p 响应，也不构成完整 Q3 联合最优。相邻变化频率描述冻结 M0 与 1,000 次配对 Bootstrap 下的离散切换，不表示因果或收益方向。

## Q3-T2：近最优集、Cost–Loss Pareto 与边际收益

此任务已将截图中的三项分析收敛为一条确定性有限网格流水线。执行计划和详细步骤见 `01_方案与设计/Q3_T2_近最优_Pareto_边际收益执行计划.md`；代码为 `03_代码/analyze_q3_t2_near_optimal_pareto.py`。按每个预算×上下文分别计算 1%、3%、5% 近优集合，按上下文精确枚举成本—M0 Loss 非支配点，并比较相邻预算的点估计和配对 Bootstrap 区间。

- 结果目录：`04_结果/Q3_T2_近最优_Pareto_边际收益/`。
- 主表：`q3_near_optimal.csv`（17,640 个候选—情景行）、`q3_pareto.csv`（1,865 个非支配点）、`q3_marginal_return.csv`（10 段预算转换）。
- 报告和追溯：`report.md`、`figure_contract.md`、`p1_smoke_receipt.json`、`reproduction_manifest.json`。
- 图件：`figures/result_q3_t2_near_optimal_pareto_marginal_return.svg`、同名 `.png` 及 `_grayscale.png`。含 Pareto、近优候选计数和按预算数量级计的边际 Loss 下降与配对 Bootstrap 区间。
- P1 已通过独立复跑；全量计算复现了 15/15 个上游 Bootstrap 情景摘要。图件为 3300×2130 像素、300 DPI，严格图件检查通过。
- 独立 P2 已通过：三张 CSV 的规模/口径、嵌套关系、15 个 M0/M1 选点、成本/掩码及 10 项输出哈希均复核一致；未发现阻断问题。
- 解释限制：固定 $Q=Q_0$、排除 $p$，只覆盖 M3 允许的 B1 1,176 个观测 N/D 点及样本内 M0 Loss；它不能替代完整 Q3 Q/p 联合优化。

从项目根目录复现：

```powershell
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py --p1
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/analyze_q3_t2_near_optimal_pareto.py
```

## Q3-T3：Bootstrap 稳定性与配置切换

**阶段记录（截至 T3 标准化完成时）：**T3 的表、图和数值口径已完成标准化，独立范围 P2 已通过；当时 Q3 整包尚未冻结。之后 Q3-T5 形成 rev2 并通过整包范围 P2，当前冻结状态以上方版本导航及 Q3-T5 小节为准。既有 1,000 次完整轨迹 cluster Bootstrap 和 22 对相邻情景配对审计是数值来源；T3 标准化只整理其交付接口，不重抽样、不重拟合，也不覆盖权威来源表。任务判断、优化步骤、质检证据和适用边界见 [`Q3-T3_Bootstrap稳定性与配置切换_优化执行记录.md`](01_方案与设计/Q3-T3_Bootstrap稳定性与配置切换_优化执行记录.md)。

标准包位于 `04_结果/Q3_T3_稳健性与配置切换/`，分开提供 `q3_bootstrap_summary.csv`、`q3_configuration_switch.csv` 和 `q3_cost_structure_switch.csv`。15 个情景的基线选点重选率均为 100%；22 对中 17 对达到 N/D 配置改变频率 ≥95%，12 对达到成本结构改变频率 ≥95%。其中预算相邻对为 N/D 10/10、成本 0/10；上下文相邻对为 N/D 7/12、成本 12/12。稳定性、N/D 切换和成本结构切换分别有独立图件，字段定义与图件映射见包内 `figure_contract.md`。

该阶段的只读 P1 与独立范围 P2 已通过；内嵌回执绑定当时的标准化脚本、支持模块及 6 项来源输入。六张图件尺寸、300 DPI、PDF 页数和嵌入字体已核验，21 项输出 SHA-256 记录在 `reproduction_manifest.json`。这份 P2 只覆盖 T3 标准化子包；整包冻结状态以后续 Q3-T5 rev2 回执为准。数值结论仍限于固定 Q=Q0、冻结 M0、8 条训练轨迹与 B1 实测支持域；不代表独立外部验证、因果效应或完整 Q/p 联合优化。增加同一 Bootstrap 复本数不能替代独立轨迹或解除 Q/p 识别门槛。

```powershell
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/standardize_q3_t3_outputs.py --validate-only
D:/Anaconda/envs/mmdet/python.exe Q3/03_代码/standardize_q3_t3_outputs.py
```

## Q3-T5：最终报告与本轮结果包 QA（rev2 已完成）

本次在既有 T1/M3/M4、T2、T3、T4 和预算覆盖结果上，补入 M0 函数形式敏感性与整轨迹留一决策分析，并生成新版本快照。当前入口为 [`冻结包_20260925_模型形式敏感性_独立复核/README.md`](冻结包_20260925_模型形式敏感性_独立复核/README.md)，正文为 [`q3_final_report.md`](04_结果/q3_final_report.md)，总表为 [`q3_master_table.csv`](04_结果/q3_master_table.csv)。rev2 包含 43 张表、36 个图件文件；登记的 140 项文件哈希/字节数、7 份源 manifest 的 132 项哈希和 27 个本地报告链接均通过独立复核。P2 回执见 [`P2_independent_release_receipt.md`](冻结包_20260925_模型形式敏感性_独立复核/QA/P2_independent_release_receipt.md)。此前版本保持不变。该范围 QA 不等同于完整 Q/p 联合优化，也不等同于原生 submission/final 完整性审计。

## Q3 补充图集（2026-09-25）

按补充图评审结果，已将 P1、P2、P3 和新增结果图整理到 [`04_结果/figures/supplementary_20260925/`](04_结果/figures/supplementary_20260925/README.md)，现含七个结果图主题和一张 Q3 分支流程图；图件提供 PNG、SVG 与灰度版，P1–P3 另有 PDF，并附输出 SHA-256 清单。P1–P3 绘图源已纳入 `03_代码/图表源/`，源脚本、清单和图件均不依赖 `zwj/`。P2 与单独 Bootstrap Loss 图、P3 与单独成本占比图存在视角重叠；图集说明已标注，便于按正文/附录版面取用。P4 未纳入。

## 新增：预算饱和与有限网格切换（2026-09-25）

优化执行方案：01_方案与设计/Q3_预算饱和与有限网格切换_优化执行计划.md。

已把预算饱和阈值和离散预算切换接入固定 Q₀/M0 主线。饱和预算按每个上下文下 1,176 个 B1 实测候选点的最大成本定义；同时报告三档预算下的可行点数、占比、剩余点数和预算/C_sat。没有采用未定义的“接近饱和”比例门槛。完整成本阈值事件按有限候选网格枚举，不解释为连续 KKT 临界点。

- 执行代码：`03_代码/analyze_q3_budget_grid_transitions.py`。
- 结果表与报告：`04_结果/预算饱和与有限网格切换/`，含 5 个上下文的 C_sat、15 个正式预算覆盖情景、完整下包络转折和 10 个正式预算端点对照。
- 主图：横轴为绝对预算 C（FLOPs），纵轴为预算可行候选比例；竖线标出各上下文的 C_sat。因成本结构使按 C_sat 归一化的五条曲线完全重合，图中采用绝对预算轴。1e24 FLOPs 下五个上下文均有 1,176/1,176 点可行。
- 汇总表、图表清单、复现清单与新的不可变快照随本轮结果刷新。快照范围仍是固定 Q₀/M0 与 B1 实测支持域，不能解释为完整 Q/p 联合最优。
