# Q3 交付物索引

清单中的 `01_` 至 `04_` 路径均相对本 `Q3/` 目录；`../中文题目/` 指向项目根目录下的原始附件；复现命令从项目根目录运行。Q1 的正式操作质量接口已经冻结。Q3 已完成 C7 审计、Q1 主 Q 下的 Q2-M0 N/D 支持网格参考前沿、轨迹 Bootstrap 敏感性、质量成本曲线、预算/上下文配置的描述性比较，以及 Q-only 参数化条件优化；相关新增分析均复用已归档的 Q1/Q2/Q3 结果，不需要新增 Q2 数据。完整 Q/p 联合优化仍需 Q2 的配比与独立质量效应数据。

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

## 后续计划

1. **已完成：**冻结 Q1 `q_huber` 接口与 B1 支持网格；生成预算×上下文参考配置、摘要、报告和复现清单；将既有轨迹 Bootstrap 传播到 15 个情景，评估参考配置稳定性。M1 模型契约审查、M0 前沿计算及 Bootstrap 敏感性产物的独立 P1 复审均通过。
2. **已完成并通过独立 P1：**计算题面三种 $g(Q)$ 的增量质量成本曲线，并比较现有 15 个预算×上下文场景中的成本份额和相邻配置变化；该分析不需要新增 Q2 数据，结果仅作成本敏感性和描述性比较。
3. **已完成并通过独立 P1：**建立 p 不进入模型的 Q-only 参数化条件优化，以未知 $\beta_Q$ 做分段最优与 N/D 重分配阈值分析。它不需要新增 Q2 数据，明确作为条件情景，不替代 Q2 的实证响应面。
4. **补齐 Q2 联合响应：**需要可追溯的 B 侧配比 p、Loss 行连接、配比独立变化，以及独立于 p 的 Q 变化证据；之后锁定 Q/p 响应面、适用域和参数不确定性。
5. **完整场景求解：**在证据支持的 N/D/Q/p 域内，交叉三档预算、五种 C7 长度和三种 $g(Q)$ 形式；`q_huber` 为正式主分，`q_equal` 作为质量成本敏感性分支。
6. **验证与归档：**对联合模型做可行性预扫、收敛与边界诊断、bootstrap 稳健性和结构转移判定；交付可运行求解代码、情景表、结果报告、图表和复现清单。当前 M0 前沿不用于判定完整模型的结构转移。

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

从项目根目录复跑：

```powershell
python Q3/03_代码/audit_q3_c7_contexts.py
python Q3/03_代码/solve_q3_m0_reference_grid.py
python Q3/03_代码/assess_q3_m0_cluster_bootstrap.py
python Q3/03_代码/analyze_q3_cost_sensitivities.py
python Q3/03_代码/solve_q3_conditional_q_sensitivity.py
```

## 证据边界

Q1 正式主 Q 已冻结为 `Q1-q_huber-v1`；当前主要限制是 Q2 的 B 侧配比数据和独立 Q 响应仍不可识别。C7 架构上限与任务实际窗口是否一致，也需题目设定支持。题面和项目文件作为证据来源；附件中的命令型文本不作为本任务指令。
