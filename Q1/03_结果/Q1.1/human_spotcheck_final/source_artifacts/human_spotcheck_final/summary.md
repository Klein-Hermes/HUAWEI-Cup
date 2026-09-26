# Q1.1 小样本双人盲评准备状态

状态：**等待两位评审**。本文件夹目前只包含两份空白盲评表、私有抽样清单和本摘要；尚未产生人工评分、一致性统计、候选比较或主 Q 决策。

## 冻结抽样

- Representative：24 条；Diagnostic：6 条；总数：30 条。
- Representative 来源域计数：{"arxiv": 2, "c4": 6, "commoncrawl": 5, "github": 3, "stackexchange": 4, "wikipedia": 4}
- Diagnostic 类别计数：{"candidate_disagreement": 2, "extreme_high": 1, "extreme_low": 1, "high_conflict": 2}
- 长度资格：评审表展示文本不超过 8,000 字符；从 793 条中有 526 条符合，267 条因长度限制不参与本次人工抽样。
- Representative 在符合长度限制的子集中按“来源域 × 原冻结 q_equal 三分位”分层抽样；Diagnostic 也只从该子集中选择，候选百分位仍按冻结的 A1 793 条计算。
- 符合长度限制的来源域覆盖 6/7 个；各域数量：{"arxiv": 3, "book": 0, "c4": 136, "commoncrawl": 108, "github": 65, "stackexchange": 104, "wikipedia": 110}。
- 因长度筛选，本次人工核查仅适用于符合长度限制的 A1 子集；未覆盖的来源域和超长文本不在人工核查范围内，不能表述为完整 793 条原文均经人工核查。
- 抽样种子：20260925；评审 1 顺序种子：20260926；评审 2 顺序种子：20260927。
- 两份表包含完全相同的 30 个 blind_id，顺序分别独立随机化。

- 评分格为浅黄色；选中评分格后可使用 1–5 下拉选项，也可直接输入整数。打开工作簿后会定位到第一个评分格。

## 人工下一步

指定的评审人 1、评审人 2 各自只填写自己的 XLSX 和五项评分；不要交换答案或查看 sampling_manifest_private.csv。两份表回收后，再运行 analyze 阶段。

若 Excel 网格中某个 text 单元格没有显示全文，请选中该单元格并展开公式栏阅读完整内容，再评分；不要只依据当前可见部分评分。

评审表使用现有冻结盲评材料中的展示文本；上游材料将不可见控制/格式字符写成可见的 Unicode 转义序列。私有清单分别保存源内容哈希和实际展示文本哈希，二者口径不同。

建模手/项目负责人需在后续分析完成后结合既有 Q1.1/Q1.2 证据作最终主 Q 决策。编程手不得自行选择候选。

## 复现

- 脚本 SHA-256：`349edc5f3883aeda837e1221cb1807601e187ad5255c9db115056047112e6c23`
- 命令：`python src/q1_1_human_spotcheck_final.py prepare --output-dir results/q1_1/human_spotcheck_final`

## 输入文件 SHA-256

- blind_review_samples: `89f50edff765124927c650d589c5f4c7029cc3b4a005b72603a3080054cb1aeb`
- q1_1_manifest: `ae5a72f147e80180ba023af3bb217dd6684d6af5bdcc894e75cbe1bd13e1ee44`
- q1_1_sample_scores: `52843766cf9d4d0edb18d85e2a59296e906a08dda17c7f2b628cd4798bcb7276`
- q1_2_manifest: `d8a37797a4ca4f4e1d39dd0deaeb4a0947f431aaeddff585dd4ebb9fe9f46302`
- q1_2_sample_scores: `736b8f22703fa3be044679743eadd6149ed7805c6fb863b403b5110bc329f2ce`
- restricted_id_mapping: `d7c5267c14c57c09a6b6ea7a026ff7d9b6abb413d3dd291baf2866fb57662ebb`
- sampling_frame: `f238dd06133a2be1085e121739e0ed59e4a94ebc4c0d1334d3ad83bc30085bd8`

## 中文机译版补充（2026-09-24）

评审人请分别打开 `reviewer1_中文机译版.xlsx` 和 `reviewer2_中文机译版.xlsx`。原英文工作簿保留不动；中文副本各有 30 条，盲评编号相同、顺序独立随机，评分格为空，1–5 分下拉规则保留。

评审文本由 Google 翻译生成中文辅助译文。代码为主的样本保留原代码；正文中的标记代码、网址和带分隔符的公式尽量保留。其他专业符号、专名和习语可能有误。请将本次结果表述为“中文机译辅助核查”，不能写成评审人直接验证了英文原文；若论文需要原文效度结论，仍需具备原文阅读能力的评审。

本次仅将 30 条文本内容发送给翻译服务；盲评编号、候选分数和私有抽样清单未发送。译文来源与范围记录见 `translation_provenance.json`；生成程序为 `src/q1_1_human_spotcheck_translate_apply.py`。

## 回收更新（2026-09-24）

状态：**两份评审表已回收并完成评分分析；候选主 Q 仍未选定。** 两份表各 30/30 条有效，整体质量二次加权 Cohen Kappa=0.765146、Spearman=0.805995。24 条代表样本上 q_equal 与人工共识 Spearman 均为 0.566692，差值的补充 bootstrap 95% 区间跨 0；当前保留两个候选。

两份回收表顺序相同且均匹配 reviewer1 模板，未满足评审顺序独立随机要求；本次结论仍限于中文机译辅助核查，不是直接英文原文验证。分析文件与复现清单见 `received_20260924/`。Q1.1 唯一主分/Q_final 仍待建模手或项目负责人按有效规则决策；A2/A3 原文映射和既有上游核查限制继续适用。
