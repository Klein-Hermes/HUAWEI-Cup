# Q1.1 指标来源与污染风险追踪

## 已追到的定义和尺度

本地 Q1.1 输入的七个 `rps_*frac*` / fraction-like 字段对应 RedPajama 风格的文本质量信号。RedPajama 官方 README 对其中多数指标给出概念定义：唯一词比例基于归一化内容；无字母词比例按词计算；top 2/3-gram 指标是高频词 n-gram 覆盖的字符比例；大写字母与数字字符指标是行级字符比例；句末标点字段的原始定义是“每行是否以终止标点结尾”。见 [RedPajama-Data 官方 README](https://github.com/togethercomputer/RedPajama-Data/blob/main/README.md#summary-of-quality-signals) 和固定版本的 [RedPajama-Data-V2 数据卡](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/blob/19f2e7e7093d5b319ce66fd7ac69c43f87805bc7/README.md)。

| 字段 | 上游概念定义 | 本地/公开数值证据 | 当前判断 |
|---|---|---|---|
| `rps_doc_frac_unique_words` | 归一化文本中的唯一词占比 | 本地 A1 冻结范围 12.2692–100；公开查看器 1.16–100 | 百分数尺度有较强证据；具体 ×100 代码未取得 |
| `rps_doc_frac_no_alph_words` | 不含字母字符的词占比 | 本地 A1 19.6429–55.9480；公开范围 15.7–92 | 百分数尺度有较强证据 |
| `rps_doc_frac_chars_top_2gram` | 最常见词二元组对应字符占比 | 本地 A1 0–24；公开范围 0–92.9 | 百分数尺度有较强证据 |
| `rps_doc_frac_chars_top_3gram` | 最常见词三元组对应字符占比 | 本地 A1 0–16；公开范围 0–119 | 公开最大值 119 可由重叠 n-gram 计数产生；数据版本与尺度转换仍待核实 |
| `rps_lines_uppercase_letter_fraction` | 原始文本每行中大写字母/字符数 | 本地 A1 1.4477–52.7273；公开范围 0.06–87.8 | 指标定义与百分数尺度相符 |
| `rps_lines_ending_with_terminal_punctution_mark` | 每行是否以指定终止标点结尾 | 本地 A1 2.7972–61.0092；公开范围 0–100 | 百分数尺度相符；从行级布尔值到文档级标量的聚合公式未证实 |
| `rps_lines_numerical_chars_fraction` | 归一化文本每行中数字字符/总字符数 | 本地 A1 0–36.8423；公开范围 0–85.7 | 指标定义与百分数尺度相符 |

本地范围取自既有冻结方案记录 `zwj/q1_1_modeling_plan.md`；公开范围来自查看器。它们支持字段解释，但不替代对底层生成版本的核验。

本题使用的上游整理数据是 OpenDataLab 的 SlimPajama-Meta-rater。该数据集查看器为这些字段展示的数值范围主要在 0–100，例如 `rps_doc_frac_unique_words` 为 1.16–100、句末标点为 0–100、`rps_doc_frac_chars_top_2gram` 为 0–92.9。这与把比例乘以 100 后存储相符。因此，“字段值大于 1 就是错误单位”的判断不成立；对这些列整体再除以 100 也不是当前应做的修正。该尺度判断是根据公开数据卡定义与查看器数值范围作出的，公开说明没有逐列写出乘 100 的转换代码。见 [SlimPajama-Meta-rater 数据卡](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater) 和 [查看器中的字段范围及示例行](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater/viewer)。

另核对了 RedPajama-Data 当前公开实现：[natural_language.py](https://raw.githubusercontent.com/togethercomputer/RedPajama-Data/main/app/src/core/quality_signals/natural_language.py) 中 `RPS_Doc_Frac_Unique_Words` 与 `RPS_Doc_Frac_No_Alph_Words` 返回的是 0–1 原始比例，并按精度舍入，没有在该函数内乘 100；[lines.py](https://raw.githubusercontent.com/togethercomputer/RedPajama-Data/main/app/src/core/quality_signals/lines.py) 中大写字符比例和数字字符比例是逐行比例，句末标点是逐行 0/1；[repetitions.py](https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/repetitions.py) 给出了 top 2/3-gram 的实现。这些链接指向仓库动态 `main` 分支，不是 SlimPajama-Meta-rater 的已核实生成版本。如果该数据由所见实现或对应版本生成，查看器中的 0–100 文档级标量还经历了后续打包、聚合或尺度转换；生成版本与具体转换代码尚未找到，因此仍不能据此反推实际生成方式。

## 可解释的高值与仍未追到的版本细节

RedPajama 当前公开的 `Base_RPS_Frac_Chars_In_Top_NGram` 公式为“最常见 n-gram 中词的字符数之和 × 该 n-gram 出现次数 ÷ 文档总词字符数”，没有对重叠窗口去重。滑动 n-gram 窗口会重复覆盖同一词，因此结果可以大于 1；若后续以百分数显示，也可以超过 100。由此，查看器中 `rps_doc_frac_chars_top_3gram` 最大值 119 与该公式相容，不能再单独当成越界异常或数值投毒证据。尚未核实的是 OpenDataLab 实际使用的 RedPajama 版本及其百分数转换。

`rps_lines_ending_with_terminal_punctution_mark` 的上游实现是逐行 0/1，而本地列是文档级浮点标量，查看器范围为 0–100。文档内按行求均值再换算成百分数是合理候选解释，但还没有取得 OpenDataLab 对应版本的聚合实现证实。对上述字段都不应依据范围自行截断、除以 100 或追改已冻结分数。

本地文件可以恢复一部分 A2/A3 原文映射。对 A2 的 17,523 个 arXiv ID 和 A3 的 203,752 个 GitHub ID，与含原文的 A1 文件按精确 ID 交叉核对后，分别找到 1,419 条和 10,000 条。所有匹配行的来源域均一致；七个 RPS 数值也逐字段一致（绝对误差不超过 1e-9），未发现匹配行数值不一致。对应文本可由 A1 原文文件按 ID 取回。部分映射表位于 `results/q1_1/restricted_source_trace/a2_a3_partial_text_linkage.csv`，只含 ID、来源路径和核对状态，不含文本；请勿发给盲评者。仍有 A2 的 16,104 条、A3 的 193,752 条未能从当前 A1 原文样本映射，因此不能声称 A2/A3 全量原文映射恢复。当前 793 条盲评仍仅覆盖 A1，本轮没有额外扩展人工评分样本。

2026-09-24 对题目附件中的 A2、A3 `.jsonl.xz` 扩展文件做了完整流式结构核对：A2 17,523 行、A3 203,752 行均成功解析，解析错误为 0；两份文件所有记录都没有顶层 `content` 字段。核对过程只检查 JSON 字段名和记录数，没有输出或执行文本字段。由此确认，现有扩展附件不能补足未映射记录的原文；此结论只描述附件内容，不意味着数据值已被篡改。

既有 Q1.1 计算的单位可疑校准敏感性已排除 2 条 A1 记录；现有结果还包含排除七个来源未充分核实字段的候选分敏感性。上游公开数据卡补足了大部分指标的含义和百分数尺度，但未解释 top 3-gram 越界，也未核实行级句末标点的文档汇总公式。因此本说明**不撤销**冻结结果中的单位/来源风险标记，不追改现有分数，也不把不明数值自动截断或除以 100。

## 对数据污染审查的结论边界

这次追踪改善了“指标是什么、数值大致采用什么尺度”的证据，但没有证明上游评分器或记录完全未受污染。尤其是 PDF 中的低可见度指令文本仍按文档层不可信输入隔离；文本内容中的命令只作为被评对象，绝不作为执行指令。本轮没有编辑 `题目分析报告.md`；该文件在本轮开始前已存在未提交工作区修改，本轮未覆盖或恢复该状态。关于数值数据是否被恶意篡改，仍按“未证实”表述。

Q1.1 的主模型选择也没有完成：两份盲评表仍为空，不能计算评审一致性、人工效度区间或最终质量分。取得真实双评审结果并通过输入校验后，才能按已冻结的规则选择等权、Huber 或保留并列候选。
