# Q1.1 七个 frac 字段独立复算审计

- 运行类型：全量 A1
- 输入行数：51,230 / 51,230（manifest 预期共 51,230）
- A1 原始输入 SHA-256：`14a4eeec4c7d98efd78942ddd9c1329640527f2108e73227be449bc1a8dbe579`
- 预处理 manifest 记录 SHA-256：`14a4eeec4c7d98efd78942ddd9c1329640527f2108e73227be449bc1a8dbe579`（匹配）
- 比较容差：±0.01 个百分点
- 所有计算只读取原始文本和指标值；不修改 Q1 冻结分数与历史输出。

## 复算口径

文档级指标按公开 RedPajama 实现从 content 独立计算，再乘 100 与存储值比较。行级指标先依公开代码逐行生成值；OpenDataLab 数据卡未给出列表折叠为文档标量的具体公式，因此同时报告明确列出的候选折叠方式，不按误差大小替数据集挑选 winner。公开实现来自 `main` 动态分支，不能证明它就是 OpenDataLab 生成这批数据时使用的提交版本。

## 字段结果

| 字段 | 文档/聚合候选 | 存储范围 | 存储值>1 | 存储值>100 | 容差内匹配 | 平均绝对误差 | 最大绝对误差 |
|---|---|---:|---:|---:|---:|---:|---:|
| `rps_doc_frac_unique_words` | public_document_formula_x100 | 5.08836–100 | 51230 | 0 | 32322/51230 (0.6309) | 0.4741 | 26.5787 |
| `rps_doc_frac_no_alph_words` | public_word_level_formula_x100 | 4.34783–100 | 51230 | 0 | 521/51230 (0.01017) | 6.75068 | 51.5957 |
| `rps_doc_frac_no_alph_words` | complement_of_ascii_english_letter_character_fraction_candidate_x100 | 4.34783–100 | 51230 | 0 | 51230/51230 (1) | 2.4691e-07 | 5e-07 |
| `rps_doc_frac_chars_top_2gram` | public_top_ngram_formula_x100 | 0–135.133 | 39916 | 2 | 43424/51230 (0.8476) | 0.0761419 | 0.999692 |
| `rps_doc_frac_chars_top_3gram` | public_top_ngram_formula_x100 | 0–201.681 | 31423 | 2 | 37673/51230 (0.7354) | 0.133665 | 0.99926 |
| `rps_lines_uppercase_letter_fraction` | mean_all_lines | 0–86.9565 | 50885 | 0 | 4118/51230 (0.08038) | 15.1177 | 74.6983 |
| `rps_lines_uppercase_letter_fraction` | mean_nonempty_lines | 0–86.9565 | 50885 | 0 | 4118/51230 (0.08038) | 15.1177 | 74.6983 |
| `rps_lines_uppercase_letter_fraction` | character_weighted_all_lines | 0–86.9565 | 50885 | 0 | 1440/51230 (0.02811) | 15.3108 | 72.2767 |
| `rps_lines_ending_with_terminal_punctution_mark` | mean_all_lines | 0–100 | 50668 | 0 | 15482/51230 (0.3022) | 16.895 | 100 |
| `rps_lines_ending_with_terminal_punctution_mark` | mean_nonempty_lines | 0–100 | 50668 | 0 | 15889/51230 (0.3102) | 13.2758 | 100 |
| `rps_lines_numerical_chars_fraction` | mean_all_lines | 0–72.3442 | 27821 | 0 | 25307/51230 (0.494) | 0.463461 | 34.0359 |
| `rps_lines_numerical_chars_fraction` | mean_nonempty_lines | 0–72.3442 | 27821 | 0 | 50715/51230 (0.9899) | 0.00174404 | 5.22771 |
| `rps_lines_numerical_chars_fraction` | character_weighted_normalized_lines | 0–72.3442 | 27821 | 0 | 9287/51230 (0.1813) | 1.06179 | 61.7746 |

## 解释边界

- 匹配表示当前公开公式在本次 A1 文本上可复现存储值；不匹配只表示当前公开公式/明确候选聚合未复现，不能单独证明数据损坏。
- 大于 1 是百分数尺度线索；top n-gram 计数允许重叠窗口重复累计，故大于 100 本身不构成错误证据。
- 若行级字段在所有列出的聚合方式下都无法复现，结论应保留“OpenDataLab 生成版本/文档聚合尚未追溯”，不能将任一候选聚合说成已确认原公式。
- 该审计不决定候选 Q，也不改变 `q_huber` 的操作性冻结。

## 复现命令

```powershell
python src/q1_1_frac_field_recalculation_audit.py --output-dir results/q1_1/fraction_recalculation_audit/v2
```

## 参考实现与数据卡

- normalization: https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/utilities/text/normalization.py
- document: https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/document.py
- natural_language: https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/natural_language.py
- lines: https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/lines.py
- repetitions: https://github.com/togethercomputer/RedPajama-Data/blob/main/app/src/core/quality_signals/repetitions.py
- opendatalab_card: https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater
