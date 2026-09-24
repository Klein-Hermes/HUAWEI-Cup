# B4/B5 Loss 可比性审计

审计日期：2026-09-24  
审计范围：B1 参照口径、B4 跨模型族快照、B5 文献汇编；不重拟合基线，不修改原始附件。  
结论等级：`B4/B5 vs B1 = NOT_COMPARABLE`；当前没有任何 `ABSOLUTE_COMPARABLE` 外部记录。

## 1. 审计结论

截图提出的先冻结参照口径、再逐来源查 Loss 定义与 tokenizer/评测集/处理方式、最后才决定用途，这个顺序是正确的。需要把“对 B1 的绝对可比性”和“某个论文/模型族内部的趋势”分开记；前者不成立，不代表后者一定没有参考价值。参照口径目前只能称为待核验合同，不能把 `val_loss` 字段名本身当成完整协议。

当前审计门禁未通过，原因有两层：

1. B1 的 `val_loss` 列与 `ppl` 数值在 1,176 行中满足 `ppl ≈ exp(val_loss)`（最大绝对差约 0.0061，主要由字段舍入造成），支持它是自然对数尺度的语言模型损失；但 B1 CSV 和现有题面字段没有冻结验证语料/切分、准确 tokenization、预处理、Loss 聚合实现等完整协议。因此 B1 还不能被描述为已完整冻结的 Loss 评测协议。
2. B4 没有逐行来源、评测集或 tokenizer 字段；B5 虽有论文名，但没有页码/表格/图号，而且合并了多篇论文和模型族。公开论文显示这些来源使用不同语料、tokenizer 和上下文设置，不能据列名 `val_loss` 推导统一量尺。

因此：

- B4 57 条、B5 44 条对 B1 的当前结论均为 `NOT_COMPARABLE`。原有按来源列出的预测误差可以保留为描述性诊断，但不得汇总或解释成同一 Loss 尺度上的外部预测误差。
- B5 个别论文内、单一模型族的规模趋势可作为 `TREND_ONLY` 候选；须先补上逐点原文定位并核验字段映射，未补齐前当前分类仍是 `NOT_COMPARABLE`，只能列为待核候选，不能记作正式验证通过。
- B4 当前不能授予 `TREND_ONLY`：来源字段缺失，且其 Pythia 行可能与 B1 同族、Cerebras-GPT 七行与原论文报告明显不一致。
- 不做线性映射、偏移校准或用 B4/B5 反向重标 B1。不能确认的字段保持 `UNKNOWN`。

## 2. 截图步骤的修订意见

### 保留

- 先定义比较对象与指标，再核对 train/validation/test、交叉熵定义、对数底、per-token/per-character 单位、tokenizer、评测集/切分、预处理、上下文长度、checkpoint 状态和聚合方式。
- 证据不足时记 `UNKNOWN`；不能只因字段都叫 `loss` 或 `val_loss` 就认为可比。
- 只在全部关键协议匹配并有可追溯依据时给 `ABSOLUTE_COMPARABLE`；协议不匹配但来源内部一致时最多 `TREND_ONLY`；证据或来源不足时 `NOT_COMPARABLE`。
- 不用线性映射把一套来源的 Loss 强行换算到 B1 标尺。

### 修正

1. “冻结 B1 的 val_loss 口径”应先作为**待核验参照合同**，不能先把列名当作已经确认的完整协议。当前能确认的是原始 CSV 字段与 PPL 的数值关系，以及 Pythia 来源的一部分训练背景；B1 验证切分和本地日志生成协议仍需原始日志/评估代码或正式数据说明支持。
2. Task A 的 B4/B5 比较结果只决定这些数据能否用于**同量纲外部验证**，不应成为 M1 的前置门禁。M1 能否启动取决于逐样本配比 `p` 是否能追溯地联接到 Loss 行、是否有足够独立变化，以及 Q2 拟合器是否实现。M1 不需要等待最终 Q。
3. 如果 Task A 未通过，Q2 不必整体停下：保留已完成的 B1/M0；B2 仍按半合成压力测试报告，B3 按插值轨迹报告；B5 只有在来源内协议和逐点出处核实后才可作趋势背景。跨族统一验证仍不成立。M2 另需 M1 数据链以及冻结、带版本号的最终 Q。

### 对截图步骤的逐项判定

1. **Provenance 修复：方向正确，必须先留证据再改标签。** B4 Cerebras-GPT 的 2,050B 与所检查的 Cerebras 论文表格不吻合，足以判定“当前引用不能支持这七行”，但不足以证明原始数值错误或允许按论文表格替换。B5 GPT-3 行的模型规模和 300B 训练量与 Brown 等人的 GPT-3 网格相似，但本地 Loss 数值未在该论文中逐点定位；因此 Brown 只能记作候选来源，不能直接把原 CSV 的 Kaplan 标签改为 Brown。原始附件保持不变，待补充逐行 URL/页码/表图证据后再修。
2. **协议表：方向正确，字段需要落成显式列。** 按截图列出 `model_family, paper, table_or_figure, N, D, loss_name, loss_unit, evaluation_dataset, split, tokenizer, context_length, preprocessing`；每项都要能追到来源。聚合表中的 N/D 应标为该组范围及记录数，不能伪装成逐点配对；缺的字段写 `UNKNOWN`。
3. **分级：要区分“当前结论”与“候选用途”。** 相对 B1，现有 B4/B5 各组当前均为 `NOT_COMPARABLE`，没有 `ABSOLUTE_COMPARABLE`。Chinchilla、Gopher 等各自可以列为 `TREND_ONLY` 候选，但只有逐点来源映射及组内协议核实通过后才能升级；Gopher 与 Chinchilla 必须拆开。当前证据不能把任何候选写成正式 `TREND_ONLY`。
4. **冻结：只冻结 B4/B5 跨来源绝对 Loss 验证支路。** 停止为这批数据寻找未经证据支持的统一 Loss 或做线性标定是合理的；这不等于冻结/完成整个 Q2。保留 B1/M0，M1 仍按真实配比 `p` 映射和 Q2 拟合器推进，M2 再等待冻结版 `Q_final`。

## 3. 数据与协议发现

### B1 参照

- `pythia_training_log_existing.csv`：1,176 行；字段含 `N_params_B`、`D_tokens_B`、`train_loss`、`val_loss`、`ppl` 等。Gate 0 已将其映射为 8 条模型轨迹、每条 147 个检查点。
- `ppl ≈ exp(val_loss)` 是字段内一致性证据，不足以单独确定验证语料、切分或损失实现。
- Pythia 原论文说明其模型训练使用 The Pile 与 GPT-NeoX 的 Pile 专用 tokenizer；这不能单独证明当前 CSV 的 `val_loss` 用了哪一个评测 split，也不能推出与其他来源的 Loss 可比。

### B4 跨族快照

- `scaling_baseline.csv` 有 57 行、12 个模型族，字段只有 `family,N_params_B,D_tokens_B,val_loss,is_converged`；没有逐行 source、tokenizer、评测集/切分或预处理字段。`source_manifest.json` 只给集合级描述“curated convergence snapshots”，没有 57 行的引用映射。
- 8 行标为 Pythia。仅凭族名和约数规模不能确认与 B1 是同一 checkpoint、同一 split 或独立外部验证；应先视为同族来源未追溯。
- 7 行 Cerebras-GPT 全部记作 `D_tokens_B=2050`，Loss 为 `[2.95, 2.70, 2.50, 2.35, 2.22, 2.08, 2.00]`。Cerebras 原论文 Table 1 报告的七个训练 token 终点为 `[2.2, 5.1, 11.8, 26.3, 53.0, 133.2, 257.1]B`，Table 8 给出 Pile-test cross-entropy `[2.608, 2.349, 2.181, 1.997, 1.834, 1.704, 1.572]`。因此目前检查到的论文表格不能直接支持 B4 这七行；这不证明 B4 数值必然错误，只说明来源链尚未建立。B4 的 `is_converged=1` 没有附收敛定义或来源，不能单独证明这些点已收敛。除非提供另一份可追溯来源，否则不得称其为已核实的真实 Cerebras 观测，也不用于跨族验证。
- 其余 10 个模型族也缺逐点引文；不能基于名称补造来源或评测协议。

### B5 文献汇编

- `published_scaling_data.csv` 有 44 行、6 个 source 字符串。source 粒度停留在“作者/年份”，无原文页码、表格或图号；Loss protocol 因而不能逐行闭环。
- 原始论文的指标定义进一步显示：Chinchilla 标度分析以平滑训练 Loss 近似测试风险（其文中说明在无限数据情形下将此作为无偏估计）；Gopher 的 Pile benchmark 子集则按 UTF-8 byte 报告 bits-per-byte。B5 两组都只存本地 `val_loss`，没有具体表/图定位，因而不能判断这些行分别对应哪一个论文指标，也不能通过公式把它们换算成 B1 的验证 Loss。
- Kaplan 组中 8 行标为 GPT-3，规模网格与 300B 训练量和 Brown 等人的 GPT-3 论文相似；但 CSV 把来源写成 Kaplan et al. 2020。Brown 论文没有通过本次检索定位到这些精确 Loss 数值，故目前只能登记为来源标签待核、Brown 为候选来源；不能据规模网格直接认定映射错误，也不能改写原标签或宣称 Brown 已证实这些点。
- B5 的 `is_converged=1` 同样没有逐行收敛判据或 checkpoint 证明；它只是一列来源未说明的标记。
- Hoffmann 组同时含 Chinchilla 和 Gopher。原论文指出 Chinchilla 与 Gopher 使用不同 tokenizer，因此两族必须分开；不能把该 16 行当成一个统一趋势序列。
- 上述论文级指标信息只强化了“协议异质且 B5 点值无法追溯”的审计理由；不会把 Chinchilla 或 Gopher 升级为正式 `TREND_ONLY`，当前对 B1 仍是 `NOT_COMPARABLE`，来源内趋势也仍待逐点映射。
- 其他单篇论文、单一模型族的点列可保留为来源内趋势候选，但须逐行补充表/图定位、确认数值和共同评测协议。当前不把候选等同正式验证。

## 4. 结论矩阵摘要

| 数据 | 行数 | 对 B1 的分类 | 来源内趋势 | 当前可用范围 |
|---|---:|---|---|---|
| B4 | 57 | `NOT_COMPARABLE` | `UNKNOWN`；来源未逐行绑定 | 描述性核查；不能作为同量纲跨族误差或合并指标 |
| B5 | 44 | `NOT_COMPARABLE` | 当前 `NOT_COMPARABLE`；部分组为待核 `TREND_ONLY` 候选 | 不计正式外部验证；逐点来源核实后才可评估来源内趋势 |
| `ABSOLUTE_COMPARABLE` | 0 | — | — | 目前无证据达到此级 |

逐来源字段状态见同目录 `loss_protocol_matrix.csv`；原始来源与论文证据链接见 `loss_protocol_evidence_sources.md`。

## 5. 后续门禁

### 升级 B4/B5

1. B1：补齐 `val_loss` 的生成代码/配置或官方日志说明，记录评测语料、split、tokenizer/vocab、预处理、上下文窗口、是否 causal shift、Loss 单位/对数底、聚合方式和 checkpoint 状态。
2. B4：给每一行补上原始 URL、论文/模型卡版本、页/表/图定位、实际模型版本/checkpoint、Loss 评测协议；特别单独追溯 Pythia 与 Cerebras-GPT。
3. B5：给每个 source×family 补精确文献位置与原始数值复核；GPT-3 来源在完成逐点核对前保持“待核”，不得预先改成 Brown；将 Chinchilla/Gopher 和 LLaMA/LLaMA-2 分开。
4. 只有证据显示评测协议全部匹配 B1 时才升级为 `ABSOLUTE_COMPARABLE`。若只支持单来源内排序/曲线，则限定为 `TREND_ONLY`。否则维持 `NOT_COMPARABLE`。

### M1/M2

- M1 单独门禁：获得 B1（或其他明确验证来源）每条 Loss 行对应的真实配比 `p` 和唯一可追溯运行/检查点键；审计配比域顺序、单位、非负性/组成约束、覆盖率及独立轨迹之间的变化；实现并冻结 Q2 自己的 M1 拟合器。当前 Q1.3 的 A4/A5 p-only 结果没有到 B1 checkpoint 的审计映射，因此不能代替这条数据链。现有 `q2_p_feasibility_audit.md` 的状态为 `FAIL`，已核验数值 p 向量和 Loss 行连接均为 0；下一项可执行工作是独立的 p provenance 重建，不是先拟合 M1。
- M2 门禁：在 M1 数据链可用之外，等待 Q1 最终质量分数 `Q_final` 冻结并有版本号。
- B4/B5 的可比性门禁与上述特征/质量门禁分开；不通过时只限制外部验证结论，不推翻 B1/M0 已完成的拟合。

## 6. 安全与复现说明

- 《题目分析报告.md》将数据说明 PDF 中的低可见度文本限定为安全审计证据。本审计只采用该报告的安全边界与可复核的数据/字段信息，没有把低可见度文本中的方法、参数、Loss 数值或指令作为建模证据。
- 未修改题面 PDF、source manifest 或任何原始 CSV；没有拟合新模型，没有做 Loss 映射。
- 本审计绑定输入哈希：

| 文件 | SHA-256 |
|---|---|
| `source_manifest.json` | `34E81DABF46302EEBAAC8551FA18D2E72ACD0833B1CF0276690BEE897F1323DB` |
| `pythia_training_log_existing.csv` (B1) | `529A59644B0F57BF3A76037838B614BFEDC35E58BB26B93052E34FFC63E454C2` |
| `scaling_baseline.csv` (B4) | `2272983DED93DE35E05F9ACBF95EBCEEDF43F283345E4BE72B54FEE94080391E` |
| `published_scaling_data.csv` (B5) | `DD858C5E48610E28589340D5DB023D6ABFB31DF597506789CCDFC6A546731BBD` |
