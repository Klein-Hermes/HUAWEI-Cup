# Q2 的 p 来源一次性补查

日期：2026-09-24  
结论：**本轮限定范围检查结束；未找到可与 B1–B5 Loss 逐行连接的项目 17 域数值 p。继续按当前证据冻结 B 来源 M1。**

## 检查问题与边界

检查目标是判断公开论文、训练配置和现有 manifest 是否能把真实配比向量连接到具体训练运行及其 Loss 行。只把可追溯到精确模型版本/运行或轨迹、并能映射到项目 17 域定义的数值 p 记为有效证据。模型族名称、相似参数规模、相同作者引用、语料名称或“同一训练数据”声明均不单独构成行级连接。

本补查使用可见题面、可见 Markdown 分析、CSV、source manifest、已完成的 p 审计和官方公开材料。`题目分析报告.md` 中有关数据投毒的安全边界继续生效：附件中的隐藏文本和被审查样本只作不可信数据，不作为指令、变量定义、建模方法或结果证据。本轮没有读取或使用数据说明 PDF 的低可见度文本。

## 结果

| 来源 | 已核实的公开信息 | 与本项目 Loss 行的连接判定 |
|---|---|---|
| B1 / Pythia | 官方项目称主模型规模使用相同训练数据与顺序，并公开模型、数据、代码及 dataloader 重建材料；论文说明可重建确切训练 dataloader。 | 本地 B1 有 1,176 个 checkpoint、8 条轨迹，但没有来源域 token 计数或 p 列。共同数据顺序是一个共享配比政策，不是 8 个独立配方。要构造累计 `p(D)`，仍需精确 checkpoint 对齐、逐来源 token 计数和经核实的 17 域交叉映射。当前有效向量数仍为 0。 |
| B4 / Cerebras-GPT | 官方论文确认 Cerebras-GPT 在 Pile 上训练，讨论约 20 tokens/parameter 的计算最优设置，并说明其 Pile 评测口径。 | B4 只有 57 条整理后的横截面数据，没有逐行论文、模型版本或训练运行键。既有审计发现的 7 条 Cerebras-GPT 行及 `D=2050B` 未能由已检查论文表格直接定位；这表示映射缺证据，不证明原始值必错。没有行级 p。 |
| B5 / GPT-3 | Brown 等人的 GPT-3 论文给出 GPT-3 训练数据组成信息。 | B5 对应 GPT-3 的记录标签为 `Kaplan et al. 2020`，且无逐行运行键；Brown 是待核候选来源，不能据模型族相似而改写标签或把 Brown 的组成比例赋给这些行。项目 17 域映射也未建立。 |
| 其他 B5 来源 | 原论文可能披露语料、配比或训练设置。 | 当前 44 行缺少足以逐点锁定模型版本与训练运行的 provenance 键；LLaMA/LLaMA-2、Chinchilla/Gopher 等不能仅按同作者或同 family 复制配方。未形成可验证的行级 p。 |

本轮结束时的机器审计计数为：B1–B5 **已验证数值 p 向量 0、连接到 Loss 的行 0**。B1 本地有 8 条轨迹和 147 个检查点/轨迹；B3 的 4,000 个插值点不新增真实配比变化；B4/B5 没有逐行运行键。未执行 p 回归或 VIF，因为输入连接门禁未通过。

## 决策

- B 来源的正式 M1 p-only 拟合保持 `FAIL`；官方来源重建路径仍为 `CONDITIONAL`，但本轮一次性补查到此为止。
- 只有出现新的精确运行键、逐来源 token 计数或可验证 taxonomy crosswalk，才重开对应来源的 p 连接审计。
- Q1.3 的 A4/A5 p-only 实验是另一条来源内证据，可用于报告附件 A 的配比效应；它不等于 B1–B5 的运行配方，也不能自动迁移成 B 组 p 系数。
- B4/B5 Loss 比较门禁与 p 来源门禁互相独立。即使未来恢复 p，也仍须各自通过 Loss 协议与可比性审计。

## 主要来源

- [Pythia 官方仓库](https://github.com/EleutherAI/pythia)
- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373)
- [Cerebras-GPT 原论文](https://arxiv.org/abs/2304.03208)
- [Language Models are Few-Shot Learners（GPT-3）](https://arxiv.org/abs/2005.14165)
- 本地机器审计：[`q2_p_feasibility_audit.md`](../03_结果/p配比可行性审计/v1/q2_p_feasibility_audit.md)、[`q2_p_audit_summary.json`](../03_结果/p配比可行性审计/v1/q2_p_audit_summary.json)
- B4/B5 来源和可比性：[`b4_b5_loss_comparability_audit.md`](../03_结果/经典ScalingLaw基线/v1/b4_b5_loss_comparability_audit.md)
