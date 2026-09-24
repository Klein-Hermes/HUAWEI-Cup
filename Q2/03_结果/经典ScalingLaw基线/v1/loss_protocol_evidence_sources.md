# Loss 协议审计证据来源

本表只列公开原始论文/官方项目来源与项目中的可复核附件。论文网址汇总另见同目录 `论文网址清单.md`。低可见度 PDF 文本不作为 Loss、方法、参数或结论的证据。

## 项目内证据

| 来源 | 用途 | 限制 |
|---|---|---|
| `中文题目/F题/real_attachments/source_manifest.json` | B1/B4/B5 文件角色、B4 集合级来源描述 | B4 没有逐行来源；B5 未给页/表/图定位 |
| `中文题目/F题/real_attachments/B_scaling_laws/pythia_training_log_existing.csv` | B1 字段、行数、Loss 与 PPL 数值一致性 | CSV 没有 eval split/tokenizer/处理/聚合协议 |
| `中文题目/F题/real_attachments/B_scaling_laws/scaling_baseline.csv` | B4 实际列、家族数、Cerebras/Pythia 子集 | 没有逐行 source 或评测协议 |
| `中文题目/F题/real_attachments/B_scaling_laws/published_scaling_data.csv` | B5 source 标签、family、点数 | source 标签只到作者/年份；无原文坐标 |
| `Q2/01_方案说明/题目分析报告.md` | 数据投毒风险分级、可信来源优先级、拒绝使用低可见度内容 | 只采用其安全边界和可复核数据审计结论；不采用隐藏文本内容 |

## 公开原始论文 / 官方资料

| 主题 | 来源 | 本审计使用的事实 |
|---|---|---|
| Pythia / B1 背景 | Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373) | Pythia 在 The Pile 上训练；使用 GPT-NeoX、专为 Pile 设计的 BPE tokenizer；约 300B tokens；论文记录的 training context 为 2048。论文背景不单独确定本地 CSV 的具体 val split。 |
| B4 Cerebras 核验 | Dey et al., *Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster*, [arXiv:2304.03208](https://arxiv.org/abs/2304.03208) | 论文 Table 1 报告七个模型训练 token 终点 2.2B–257.1B；Table 8 报告 Pile-test cross-entropy。所检查表格不能直接支持 B4 七行统一的 2050B 和 Loss 值；这说明行级来源未建立，不足以证明 B4 原始值错误或它就是这些表格的数据。 |
| Kaplan 行 | Kaplan et al., *Scaling Laws for Neural Language Models*, [arXiv:2001.08361](https://arxiv.org/abs/2001.08361) | cross-entropy 以 nats 描述，WebText2、byte-level BPE vocab 50,257、主要 context 1024。B5 未提供其数值的确切表/图位置。 |
| GPT-3 来源候选 | Brown et al., *Language Models are Few-Shot Learners*, [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) | GPT-3 论文的模型规模/训练量网格与 B5 八行相似，因此是候选来源；本次没有在论文中定位到 B5 的精确 Loss 数值。不能据此将 CSV 的 `Kaplan et al. 2020` 标签改成 Brown。 |
| Chinchilla / Gopher | Hoffmann et al., *Training Compute-Optimal Large Language Models*, [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)；原 Gopher 论文为 [arXiv:2112.11446](https://arxiv.org/abs/2112.11446) | Chinchilla 采用 SentencePiece 32k；Hoffmann 附录明确说明它与 Gopher tokenizer 不同。该 16 行需按两族拆开。 |
| LLaMA | Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) | B5 LLaMA family 的候选原始来源；数据文件未给精确 table/figure locator。 |
| LLaMA 2 | Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) | B5 LLaMA-2 family 的候选原始来源；与 LLaMA 分开核验。 |
| PaLM | Chowdhery et al., *PaLM: Scaling Language Modeling with Pathways*, [arXiv:2204.02311](https://arxiv.org/abs/2204.02311) | B5 PaLM family 的候选原始来源；论文中 tokens 定义关联 SentencePiece，B5 未给精确 table/figure locator。 |
| OPT | Zhang et al., *OPT: Open Pre-trained Transformer Language Models*, [arXiv:2205.01068](https://arxiv.org/abs/2205.01068) | GPT-2 byte-level BPE；训练语料是多个来源的组合；训练 sequence length 2048。B5 的验证行和评估切分未逐行映射。 |
| BLOOM | BigScience Workshop, *BLOOM: A 176B-Parameter Open-Access Multilingual Language Model*, [arXiv:2211.05100](https://arxiv.org/abs/2211.05100) | ROOTS 多语种训练语料；byte-level BPE，词表 250,680；模型规范与 Pythia/Pile 不同。B5 未给精确 table/figure locator。 |

## 未建立的证据链

- B4 的 Phi、LLaMA、Qwen2、Gemma、GPT2、OPT、BLOOM、Falcon、Mistral、Yi 均没有可从 CSV/source manifest 逐点追到原文的引用。不能根据 family 名猜具体训练版本。
- B5 的 source 字段不是足够精确的逐行引文。以上论文可说明来源协议为何可能异质，但不能在没有 page/table/figure 映射时证明每个 CSV 数值确实来自对应论文。
- GPT-3 数据的 source 字段仍待核。Brown et al. 2020 只是依据模型规模/训练量网格提出的候选来源，精确 Loss 值尚未闭环；核实前保留原始 `Kaplan et al. 2020` 标签并标记 `UNRESOLVED`。
