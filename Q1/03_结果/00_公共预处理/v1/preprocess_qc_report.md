# Q1 A1/A2/A3 公共预处理 QC 验收报告

- 冻结版本：`q1-common-v1.1`
- 冻结状态：`FROZEN`
- 验收时间（UTC）：`2026-09-23T09:28:48Z`
- 验收范围：A1 校准、A1/A2/A3 统一预处理结果及可复现性信息

## 1. 验收结论

本报告由冻结脚本基于已生成文件重新扫描得到，不重新拟合校准参数，不修改三份预处理数据。

| 检查项 | 结果 |
|---|---|
| 指标数 | PASS：22 / 22 |
| 输出列契约 | PASS：140 列 |
| A1 校准复用到 A2/A3 | PASS |
| 所有非空 norm 值位于 [0,1] | PASS |
| 输入/输出哈希可追溯 | PASS |
| 行数与域计数保持 | PASS |

## 2. 冻结版本与校准规则

- `freeze_version`：`q1-common-v1.1`
- `schema_version`：`q1-common-preprocess-v1`
- `fit_dataset`：`A1`
- `fit_rows`：`51230`
- `quantile_points`：`2001`
- `calibration_quantile_clip`：`[0.005, 0.995]`
- `same_calibration_for`：`a1, a2, a3`

校准规则只在 A1 上拟合；A2、A3 直接应用同一套字段解析、方向处理和归一化参数。

## 3. 数据集级验收

| 数据集 | 行数 | 列数 | 域计数 | 缺失值 | 非法值 | 单位疑似值 | 异常标记总数 | 输出 SHA-256 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| A1 | 51230 | 140 | `{"arxiv": 1419, "book": 171, "c4": 10000, "commoncrawl": 9640, "github": 10000, "stackexchange": 10000, "wikipedia": 10000}` | 18 | 0 | 4 | 63708 | `38c753ab9b5fa3c2ac370f9a1c600ae761dca1154814d542a45fb98737e13590` |
| A2 | 17523 | 140 | `{"arxiv": 17523}` | 0 | 0 | 0 | 75785 | `55cd687cc60614084568ac09ca141b0a6a847528edc8edac88ea228362f055fb` |
| A3 | 203752 | 140 | `{"github": 203752}` | 1 | 0 | 16 | 385678 | `a011c3c04a603fd306ce77f40044ecb0924ab79f18acfd53195fd0bf92213d8f` |

说明：`outlier` 是相对于 A1 稳健中心和尺度的审计标记，不表示删除该行；后续综合评分应使用稳健聚合或敏感性分析。

## 4. 字段级 QC

`norm_valid` 为该字段成功生成归一化值的记录数；缺失值保留为空并由对应标记列记录。

| 数据集 | 指标 | n | missing | invalid | unit_suspect | outlier | norm_valid | norm_min | norm_max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | `fineweb_edu` | 51230 | 0 | 0 | 0 | 693 | 51230 | 0 | 1 |
| A1 | `fluency_en` | 51230 | 0 | 0 | 0 | 8321 | 51230 | 0 | 1 |
| A1 | `modernbert_cleanliness` | 51230 | 0 | 0 | 0 | 0 | 51230 | 0 | 1 |
| A1 | `modernbert_readability` | 51230 | 0 | 0 | 0 | 124 | 51230 | 0 | 1 |
| A1 | `modernbert_reasoning` | 51230 | 13 | 0 | 0 | 6129 | 51217 | 0 | 1 |
| A1 | `modernbert_professionalism` | 51230 | 5 | 0 | 0 | 0 | 51225 | 0 | 1 |
| A1 | `dsir_books` | 51230 | 0 | 0 | 0 | 4369 | 51230 | 0 | 1 |
| A1 | `dsir_wiki` | 51230 | 0 | 0 | 0 | 4379 | 51230 | 0 | 1 |
| A1 | `dsir_math` | 51230 | 0 | 0 | 0 | 4300 | 51230 | 0 | 1 |
| A1 | `qurater` | 51230 | 0 | 0 | 0 | 118 | 51230 | 0 | 1 |
| A1 | `ad_en` | 51230 | 0 | 0 | 0 | 14278 | 51230 | 0 | 1 |
| A1 | `rps_doc_word_count` | 51230 | 0 | 0 | 0 | 281 | 51230 | 0.00178083 | 1 |
| A1 | `rps_doc_num_sentences` | 51230 | 0 | 0 | 0 | 312 | 51230 | 0.00189142 | 1 |
| A1 | `rps_doc_unigram_entropy` | 51230 | 0 | 0 | 0 | 296 | 51230 | 0 | 1 |
| A1 | `rps_doc_frac_unique_words` | 51230 | 0 | 0 | 0 | 0 | 51230 | 0 | 0.995 |
| A1 | `rps_doc_frac_no_alph_words` | 51230 | 0 | 0 | 0 | 3788 | 51230 | 0.005 | 1 |
| A1 | `rps_doc_frac_chars_top_2gram` | 51230 | 0 | 0 | 2 | 3762 | 51230 | 0 | 0.917285 |
| A1 | `rps_doc_frac_chars_top_3gram` | 51230 | 0 | 0 | 2 | 2499 | 51230 | 0 | 0.774725 |
| A1 | `rps_lines_uppercase_letter_fraction` | 51230 | 0 | 0 | 0 | 1 | 51230 | 0 | 1 |
| A1 | `rps_lines_ending_with_terminal_punctution_mark` | 51230 | 0 | 0 | 0 | 0 | 51230 | 0.00896 | 0.995 |
| A1 | `rps_lines_numerical_chars_fraction` | 51230 | 0 | 0 | 0 | 4133 | 51230 | 1.77154e-19 | 0.999988 |
| A1 | `rps_doc_mean_word_length` | 51230 | 0 | 0 | 0 | 5925 | 51230 | 1.78474e-35 | 0.999997 |
| A2 | `fineweb_edu` | 17523 | 0 | 0 | 0 | 1034 | 17523 | 0 | 1 |
| A2 | `fluency_en` | 17523 | 0 | 0 | 0 | 81 | 17523 | 0 | 1 |
| A2 | `modernbert_cleanliness` | 17523 | 0 | 0 | 0 | 0 | 17523 | 0.0112605 | 1 |
| A2 | `modernbert_readability` | 17523 | 0 | 0 | 0 | 0 | 17523 | 0.0491943 | 1 |
| A2 | `modernbert_reasoning` | 17523 | 0 | 0 | 0 | 17383 | 17523 | 0.0118704 | 1 |
| A2 | `modernbert_professionalism` | 17523 | 0 | 0 | 0 | 0 | 17523 | 0.152057 | 1 |
| A2 | `dsir_books` | 17523 | 0 | 0 | 0 | 17098 | 17523 | 0 | 0.978035 |
| A2 | `dsir_wiki` | 17523 | 0 | 0 | 0 | 17053 | 17523 | 0 | 0.978166 |
| A2 | `dsir_math` | 17523 | 0 | 0 | 0 | 16755 | 17523 | 0 | 0.979781 |
| A2 | `qurater` | 17523 | 0 | 0 | 0 | 521 | 17523 | 0.0147059 | 1 |
| A2 | `ad_en` | 17523 | 0 | 0 | 0 | 1324 | 17523 | 0.018191 | 1 |
| A2 | `rps_doc_word_count` | 17523 | 0 | 0 | 0 | 702 | 17523 | 0.00384396 | 1 |
| A2 | `rps_doc_num_sentences` | 17523 | 0 | 0 | 0 | 965 | 17523 | 0.00159295 | 1 |
| A2 | `rps_doc_unigram_entropy` | 17523 | 0 | 0 | 0 | 1 | 17523 | 0.01391 | 1 |
| A2 | `rps_doc_frac_unique_words` | 17523 | 0 | 0 | 0 | 0 | 17523 | 0 | 0.995 |
| A2 | `rps_doc_frac_no_alph_words` | 17523 | 0 | 0 | 0 | 426 | 17523 | 0.021619 | 1 |
| A2 | `rps_doc_frac_chars_top_2gram` | 17523 | 0 | 0 | 0 | 25 | 17523 | 0.00829949 | 0.917285 |
| A2 | `rps_doc_frac_chars_top_3gram` | 17523 | 0 | 0 | 0 | 40 | 17523 | 0 | 0.774725 |
| A2 | `rps_lines_uppercase_letter_fraction` | 17523 | 0 | 0 | 0 | 2 | 17523 | 0 | 1 |
| A2 | `rps_lines_ending_with_terminal_punctution_mark` | 17523 | 0 | 0 | 0 | 0 | 17523 | 0.00960926 | 0.995 |
| A2 | `rps_lines_numerical_chars_fraction` | 17523 | 0 | 0 | 0 | 2263 | 17523 | 5.42044e-23 | 0.999984 |
| A2 | `rps_doc_mean_word_length` | 17523 | 0 | 0 | 0 | 112 | 17523 | 8.60381e-16 | 0.999974 |
| A3 | `fineweb_edu` | 203752 | 0 | 0 | 0 | 294 | 203752 | 0 | 1 |
| A3 | `fluency_en` | 203752 | 0 | 0 | 0 | 66151 | 203752 | 0 | 0.980398 |
| A3 | `modernbert_cleanliness` | 203752 | 0 | 0 | 0 | 0 | 203752 | 0 | 0.994021 |
| A3 | `modernbert_readability` | 203752 | 0 | 0 | 0 | 2515 | 203752 | 0 | 1 |
| A3 | `modernbert_reasoning` | 203752 | 0 | 0 | 0 | 8293 | 203752 | 0 | 0.991809 |
| A3 | `modernbert_professionalism` | 203752 | 1 | 0 | 0 | 0 | 203751 | 0 | 1 |
| A3 | `dsir_books` | 203752 | 0 | 0 | 0 | 29363 | 203752 | 0 | 1 |
| A3 | `dsir_wiki` | 203752 | 0 | 0 | 0 | 29009 | 203752 | 0 | 1 |
| A3 | `dsir_math` | 203752 | 0 | 0 | 0 | 26201 | 203752 | 0 | 1 |
| A3 | `qurater` | 203752 | 0 | 0 | 0 | 1 | 203752 | 0 | 1 |
| A3 | `ad_en` | 203752 | 0 | 0 | 0 | 32213 | 203752 | 0.00536292 | 1 |
| A3 | `rps_doc_word_count` | 203752 | 0 | 0 | 0 | 891 | 203752 | 0.00769331 | 1 |
| A3 | `rps_doc_num_sentences` | 203752 | 0 | 0 | 0 | 1059 | 203752 | 0.00178777 | 1 |
| A3 | `rps_doc_unigram_entropy` | 203752 | 0 | 0 | 0 | 5943 | 203752 | 0 | 1 |
| A3 | `rps_doc_frac_unique_words` | 203752 | 0 | 0 | 0 | 7 | 203752 | 0 | 0.995 |
| A3 | `rps_doc_frac_no_alph_words` | 203752 | 0 | 0 | 0 | 32344 | 203752 | 0.0124955 | 1 |
| A3 | `rps_doc_frac_chars_top_2gram` | 203752 | 0 | 0 | 7 | 19949 | 203752 | 0 | 0.917285 |
| A3 | `rps_doc_frac_chars_top_3gram` | 203752 | 0 | 0 | 9 | 14489 | 203752 | 0 | 0.774725 |
| A3 | `rps_lines_uppercase_letter_fraction` | 203752 | 0 | 0 | 0 | 28 | 203752 | 0 | 1 |
| A3 | `rps_lines_ending_with_terminal_punctution_mark` | 203752 | 0 | 0 | 0 | 0 | 203752 | 0.00896 | 0.995 |
| A3 | `rps_lines_numerical_chars_fraction` | 203752 | 0 | 0 | 0 | 7834 | 203752 | 5.09744e-17 | 0.999999 |
| A3 | `rps_doc_mean_word_length` | 203752 | 0 | 0 | 0 | 109094 | 203752 | 2.89868e-37 | 0.999892 |

## 5. 输入文件哈希

| 数据集 | 输入文件 | SHA-256 |
|---|---|---|
| A1 | `D:\PycharmProject\HUAWEI-Cup\中文题目\F题\real_attachments\A_data_value\slimpajama_quality_signal_sample.jsonl.xz` | `14a4eeec4c7d98efd78942ddd9c1329640527f2108e73227be449bc1a8dbe579` |
| A2 | `D:\PycharmProject\HUAWEI-Cup\中文题目\F题\real_attachments\A_data_value\slimpajama_quality_extended\arxiv_part-6777d8857c6e-000486.jsonl.xz` | `ae1e3399f84f605d994fc60b46984d742e14abb3355d8e7a91822eae1b99e16a` |
| A3 | `D:\PycharmProject\HUAWEI-Cup\中文题目\F题\real_attachments\A_data_value\slimpajama_quality_extended\github_part-6777d8857c6e-000275.jsonl.xz` | `7af069c71c6027a10f2013cc14dd9d5d17855c264734f69c955544ff6cff7382` |

## 6. 后续使用约束

1. Q1.1、Q1.2 后续计算直接使用三份结果中的 `norm_*` 列。
2. 不得在 A2、A3 内重新拟合 Min-Max、Z-score 或经验分布。
3. `missing_*`、`invalid_*`、`unit_suspect_*`、`outlier_*` 仅作为审计和稳健性分析依据。
