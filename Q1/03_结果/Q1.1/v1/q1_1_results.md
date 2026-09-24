# F 题 Q1.1 质量评分：计算结果与限制

本文件只报告 Q1.1。Q1.2 冲突分析与 Q1.3 配比—Loss 估计不在本计算范围内。

## 计算状态

- 输入：A1 51,230、A2 17,523、A3 203,752 条；22 个冻结标准化指标。
- 样本级候选分：等权平均与 A1 域排序稳定性加权 Huber；异常与审计标记不直接当作质量指标。
- A2-arxiv 对 A1-arxiv、A3-github 对 A1-github 的迁移比较已按 A1 参数冻结计算。
- A1 原文盲评包：793 条；需两名人工评审完成后才能计算评审一致性、设计加权 Spearman 和主分选择。
- 当前状态：**评分计算与稳健性候选结果已生成；人工效度验证及最终主分选择待评审分数回填。不得称为全面效度验证完成。**

## 数据集层候选综合质量分

| 数据集 | 候选评分 | 综合分 | 95%区间 | 汇总口径 |
|---|---|---:|---:|---|
| A1 | 等权 `q_equal` | 0.49622 | [0.49491, 0.49763] | 七个来源域域分等权宏平均 |
| A1 | 加权 Huber `q_huber` | 0.49848 | [0.49657, 0.50043] | 七个来源域域分等权宏平均 |
| A2 | 等权 `q_equal` | 0.53096 | [0.53047, 0.53148] | arxiv 域分；仅代表该匹配域 |
| A2 | 加权 Huber `q_huber` | 0.55997 | [0.55920, 0.56075] | arxiv 域分；仅代表该匹配域 |
| A3 | 等权 `q_equal` | 0.40394 | [0.40364, 0.40423] | github 域分；仅代表该匹配域 |
| A3 | 加权 Huber `q_huber` | 0.38887 | [0.38856, 0.38919] | github 域分；仅代表该匹配域 |

上述是已计算的**候选综合分**，不是经人工效度验证后选定的唯一最终分。A1 的等权与加权 Huber 暂并列；待双人盲评回填后按预设规则选择主分。A2/A3 只报告与 A1 对应的单一匹配域，不能解读为全域宏平均。

## A1 权重参数

权重由 A1 指标完整度与七域中位数排序 bootstrap 稳定性产生。它表示计算规则下的候选权重，不是人工重要性或效度。

| 指标 | 权重 | A1完整度 | 域排序稳定性 |
|---|---:|---:|---:|
| fineweb_edu | 0.045492 | 1.000000 | 1.000000 |
| fluency_en | 0.045492 | 1.000000 | 1.000000 |
| modernbert_cleanliness | 0.045492 | 1.000000 | 1.000000 |
| modernbert_readability | 0.045492 | 1.000000 | 1.000000 |
| dsir_books | 0.045492 | 1.000000 | 1.000000 |
| dsir_wiki | 0.045492 | 1.000000 | 1.000000 |

完整 22 项权重、样本级候选分、领域估计区间和敏感性表见同目录 CSV。

## 领域结果

主候选领域分使用 A1 同域估计并冻结的 Huber 位置阈值；区间为记录 bootstrap percentile 95% CI。

| 数据集 | 域 | 候选 | n | 域分 | 区间 |
|---|---|---|---:|---:|---:|
| a1 | arxiv | q_equal | 1419 | 0.53210 | [0.5303271234035491, 0.5337712049484253] |
| a1 | arxiv | q_huber | 1419 | 0.56133 | [0.5587501093745232, 0.5638743057847023] |
| a1 | book | q_equal | 171 | 0.43160 | [0.42312069833278654, 0.44069116860628127] |
| a1 | book | q_huber | 171 | 0.42875 | [0.4166682198643684, 0.44231095165014267] |
| a1 | c4 | q_equal | 10000 | 0.57053 | [0.5686645656824112, 0.5723332583904266] |
| a1 | c4 | q_huber | 10000 | 0.57673 | [0.5747235760092735, 0.5785818085074425] |
| a1 | commoncrawl | q_equal | 9640 | 0.53401 | [0.5323690488934517, 0.5356416761875152] |
| a1 | commoncrawl | q_huber | 9640 | 0.53597 | [0.5341207027435303, 0.5378116250038147] |
| a1 | github | q_equal | 10000 | 0.40450 | [0.4031180918216705, 0.4057907149195671] |
| a1 | github | q_huber | 10000 | 0.38948 | [0.3880361646413803, 0.3909398004412651] |
| a1 | stackexchange | q_equal | 10000 | 0.51734 | [0.5159537687897682, 0.5186818644404412] |
| a1 | stackexchange | q_huber | 10000 | 0.51837 | [0.5169330433011055, 0.519771882891655] |
| a1 | wikipedia | q_equal | 10000 | 0.48349 | [0.4819977954030037, 0.4849605768918991] |
| a1 | wikipedia | q_huber | 10000 | 0.47871 | [0.4769694611430168, 0.4804027646780014] |
| a2 | arxiv | q_equal | 17523 | 0.53096 | [0.5304746761918068, 0.5314777508378029] |
| a2 | arxiv | q_huber | 17523 | 0.55997 | [0.5591986939311028, 0.5607526257634163] |
| a3 | github | q_equal | 203752 | 0.40394 | [0.4036353811621666, 0.4042311757802963] |
| a3 | github | q_huber | 203752 | 0.38887 | [0.3885551869869232, 0.38919114619493483] |

## 同域扩展比较

| 扩展集 | 域 | 候选 | 扩展集−A1 | 95%区间 |
|---|---|---|---:|---:|
| a2 | arxiv | q_equal | -0.00114 | [-0.00288, 0.00070] |
| a2 | arxiv | q_huber | -0.00136 | [-0.00399, 0.00128] |
| a3 | github | q_equal | -0.00056 | [-0.00185, 0.00086] |
| a3 | github | q_huber | -0.00061 | [-0.00211, 0.00085] |

## 污染、单位与模型敏感性

Unicode 场景只用各自未标记 A1 记录估计权重、Huber 阈值和域级 κ，之后将参数应用于完整 A1/A2/A3；评分表仍保留审计命中记录。这不能证明上游预计算信号已经去污染。`unit_flagged_a1_excluded` 从 A1 参数校准中排除带单位可疑标记的记录；`exclude_7_unit_ambiguous_fields` 则从各样本评分中排除 7 个语义/单位未核实字段，两种敏感性含义不同。

| 场景 | 数据集 | 候选 | 样本绝对分差 P95 | 最大绝对分差 | 记录排序 Spearman |
|---|---|---|---:|---:|---:|
| unicode_strict_clean_a1 | a1 | q_huber | 0.000348 | 0.001005 | 0.999998 |
| unicode_strict_clean_a1 | a2 | q_huber | 0.000618 | 0.001106 | 0.999993 |
| unicode_strict_clean_a1 | a3 | q_huber | 0.000368 | 0.001001 | 0.999998 |
| unicode_broad_clean_a1 | a1 | q_huber | 0.000216 | 0.000652 | 0.999999 |
| unicode_broad_clean_a1 | a2 | q_huber | 0.000315 | 0.000546 | 0.999994 |
| unicode_broad_clean_a1 | a3 | q_huber | 0.000214 | 0.000653 | 0.999999 |
| unit_flagged_a1_excluded | a1 | q_huber | 0.000001 | 0.000001 | 1.000000 |
| unit_flagged_a1_excluded | a2 | q_huber | 0.000001 | 0.000001 | 1.000000 |
| unit_flagged_a1_excluded | a3 | q_huber | 0.000001 | 0.000001 | 1.000000 |
| exclude_7_unit_ambiguous_fields | a1 | q_huber | 0.113048 | 0.255681 | 0.872404 |
| exclude_7_unit_ambiguous_fields | a2 | q_huber | 0.141483 | 0.335909 | 0.729556 |
| exclude_7_unit_ambiguous_fields | a3 | q_huber | 0.100505 | 0.252882 | 0.824519 |
| weight_alpha_0.25 | a1 | q_huber | 0.000148 | 0.000512 | 1.000000 |
| weight_alpha_0.25 | a2 | q_huber | 0.000212 | 0.000448 | 0.999999 |
| weight_alpha_0.25 | a3 | q_huber | 0.000141 | 0.000448 | 1.000000 |
| weight_alpha_0.5 | a1 | q_huber | 0.000000 | 0.000000 | 1.000000 |
| weight_alpha_0.5 | a2 | q_huber | 0.000000 | 0.000000 | 1.000000 |
| weight_alpha_0.5 | a3 | q_huber | 0.000000 | 0.000000 | 1.000000 |
| weight_alpha_1 | a1 | q_huber | 0.000294 | 0.001017 | 0.999999 |
| weight_alpha_1 | a2 | q_huber | 0.000423 | 0.000890 | 0.999994 |
| weight_alpha_1 | a3 | q_huber | 0.000280 | 0.000890 | 0.999998 |
| huber_delta_0.5x | a1 | q_huber | 0.071657 | 0.145049 | 0.980320 |
| huber_delta_0.5x | a2 | q_huber | 0.094353 | 0.140092 | 0.906503 |
| huber_delta_0.5x | a3 | q_huber | 0.078722 | 0.145956 | 0.954881 |
| huber_delta_2x | a1 | q_huber | 0.036720 | 0.098812 | 0.994886 |
| huber_delta_2x | a2 | q_huber | 0.056787 | 0.076923 | 0.966761 |
| huber_delta_2x | a3 | q_huber | 0.045655 | 0.099178 | 0.974832 |

A1 七域分数排序及宏平均：

| 场景 | 候选 | 七域排序 Spearman | A1 七域宏平均变化 |
|---|---|---:|---:|
| exclude_7_unit_ambiguous_fields | q_equal | 0.678571 | +0.002973 |
| exclude_7_unit_ambiguous_fields | q_huber | 0.750000 | +0.005381 |
| huber_delta_0.5x | q_equal | 1.000000 | +0.000000 |
| huber_delta_0.5x | q_huber | 0.964286 | +0.007374 |
| huber_delta_2x | q_equal | 1.000000 | +0.000000 |
| huber_delta_2x | q_huber | 0.964286 | -0.002277 |
| unicode_broad_clean_a1 | q_equal | 1.000000 | +0.000019 |
| unicode_broad_clean_a1 | q_huber | 1.000000 | -0.000022 |
| unicode_strict_clean_a1 | q_equal | 1.000000 | +0.000019 |
| unicode_strict_clean_a1 | q_huber | 1.000000 | -0.000072 |
| unit_flagged_a1_excluded | q_equal | 1.000000 | +0.000000 |
| unit_flagged_a1_excluded | q_huber | 1.000000 | +0.000000 |
| weight_alpha_0.25 | q_equal | 1.000000 | +0.000000 |
| weight_alpha_0.25 | q_huber | 1.000000 | +0.000020 |
| weight_alpha_0.5 | q_equal | 1.000000 | +0.000000 |
| weight_alpha_0.5 | q_huber | 1.000000 | +0.000000 |
| weight_alpha_1 | q_equal | 1.000000 | +0.000000 |
| weight_alpha_1 | q_huber | 1.000000 | -0.000039 |

### 污染/单位场景下 A2/A3 同域差异区间

区间按场景的 A1 校准子集重估域 Huber 阈值，再冻结到对应 A2/A3 域；Unicode 场景只用各自未标记 A1，单位场景只用无单位可疑标记 A1。差异由两个数据集独立的记录 bootstrap 抽样逐次相减。

| 场景 | 扩展集 | 域 | 候选 | 扩展集−A1 | 95%区间 | 状态 |
|---|---|---|---|---:|---:|---|
| unicode_strict_clean_a1 | a2 | arxiv | q_equal | -0.00114 | [-0.00299, +0.00066] | computed |
| unicode_strict_clean_a1 | a2 | arxiv | q_huber | -0.00136 | [-0.00434, +0.00133] | computed |
| unicode_strict_clean_a1 | a3 | github | q_equal | -0.00056 | [-0.00195, +0.00078] | computed |
| unicode_strict_clean_a1 | a3 | github | q_huber | -0.00061 | [-0.00198, +0.00092] | computed |
| unicode_broad_clean_a1 | a2 | arxiv | q_equal | -0.00114 | [-0.00298, +0.00064] | computed |
| unicode_broad_clean_a1 | a2 | arxiv | q_huber | -0.00136 | [-0.00403, +0.00149] | computed |
| unicode_broad_clean_a1 | a3 | github | q_equal | -0.00056 | [-0.00184, +0.00075] | computed |
| unicode_broad_clean_a1 | a3 | github | q_huber | -0.00061 | [-0.00216, +0.00088] | computed |
| unit_flagged_a1_excluded | a2 | arxiv | q_equal | -0.00114 | [-0.00287, +0.00063] | computed |
| unit_flagged_a1_excluded | a2 | arxiv | q_huber | -0.00136 | [-0.00412, +0.00147] | computed |
| unit_flagged_a1_excluded | a3 | github | q_equal | -0.00056 | [-0.00190, +0.00077] | computed |
| unit_flagged_a1_excluded | a3 | github | q_huber | -0.00061 | [-0.00221, +0.00087] | computed |
| exclude_7_unit_ambiguous_fields | a2 | arxiv | q_equal | -0.00198 | [-0.00390, -0.00014] | computed |
| exclude_7_unit_ambiguous_fields | a2 | arxiv | q_huber | -0.00234 | [-0.00503, +0.00018] | computed |
| exclude_7_unit_ambiguous_fields | a3 | github | q_equal | -0.00002 | [-0.00168, +0.00163] | computed |
| exclude_7_unit_ambiguous_fields | a3 | github | q_huber | +0.00003 | [-0.00194, +0.00191] | computed |

### A1 未标记子集与全样本域分

对窄口径 Cf/Cc 与宽 Unicode 两种标记分别报告未标记记录子集和全样本域分；95%区间使用配对分层 bootstrap，未标记样本的重抽结果同时进入子集与全样本估计。

| 场景 | 域 | 候选 | 全样本分 | 未标记分 | 未标记−全样本 | 95%区间 | 七域排序相关 | 七域宏平均差 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| unicode_strict_clean_a1 | arxiv | q_equal | 0.53210 | 0.53214 | +0.00003 | [-0.00007, +0.00011] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | arxiv | q_huber | 0.56105 | 0.56112 | +0.00007 | [-0.00010, +0.00017] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | book | q_equal | 0.43172 | 0.43112 | -0.00061 | [-0.00368, +0.00236] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | book | q_huber | 0.42851 | 0.42713 | -0.00138 | [-0.00573, +0.00310] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | c4 | q_equal | 0.57052 | 0.57052 | -0.00000 | [-0.00014, +0.00013] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | c4 | q_huber | 0.57666 | 0.57667 | +0.00001 | [-0.00014, +0.00015] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | commoncrawl | q_equal | 0.53402 | 0.53453 | +0.00051 | [+0.00024, +0.00080] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | commoncrawl | q_huber | 0.53595 | 0.53647 | +0.00053 | [+0.00022, +0.00084] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | github | q_equal | 0.40450 | 0.40451 | +0.00001 | [-0.00003, +0.00005] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | github | q_huber | 0.38967 | 0.38968 | +0.00001 | [-0.00004, +0.00006] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | stackexchange | q_equal | 0.51734 | 0.51742 | +0.00008 | [+0.00001, +0.00015] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | stackexchange | q_huber | 0.51832 | 0.51841 | +0.00009 | [+0.00001, +0.00016] | 1.00000 | -0.00010 |
| unicode_strict_clean_a1 | wikipedia | q_equal | 0.48349 | 0.48350 | +0.00000 | [-0.00007, +0.00009] | 1.00000 | +0.00000 |
| unicode_strict_clean_a1 | wikipedia | q_huber | 0.47867 | 0.47866 | -0.00001 | [-0.00011, +0.00008] | 1.00000 | -0.00010 |
| unicode_broad_clean_a1 | arxiv | q_equal | 0.53210 | 0.53214 | +0.00003 | [-0.00007, +0.00011] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | arxiv | q_huber | 0.56128 | 0.56135 | +0.00007 | [-0.00010, +0.00017] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | book | q_equal | 0.43172 | 0.43146 | -0.00026 | [-0.00330, +0.00310] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | book | q_huber | 0.42867 | 0.42779 | -0.00088 | [-0.00576, +0.00389] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | c4 | q_equal | 0.57052 | 0.57053 | +0.00000 | [-0.00014, +0.00015] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | c4 | q_huber | 0.57667 | 0.57669 | +0.00001 | [-0.00015, +0.00017] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | commoncrawl | q_equal | 0.53402 | 0.53458 | +0.00056 | [+0.00027, +0.00083] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | commoncrawl | q_huber | 0.53597 | 0.53655 | +0.00058 | [+0.00028, +0.00087] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | github | q_equal | 0.40450 | 0.40452 | +0.00002 | [-0.00003, +0.00008] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | github | q_huber | 0.38956 | 0.38958 | +0.00002 | [-0.00004, +0.00008] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | stackexchange | q_equal | 0.51734 | 0.51743 | +0.00009 | [+0.00001, +0.00017] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | stackexchange | q_huber | 0.51836 | 0.51846 | +0.00010 | [+0.00002, +0.00018] | 1.00000 | -0.00002 |
| unicode_broad_clean_a1 | wikipedia | q_equal | 0.48349 | 0.48350 | +0.00000 | [-0.00008, +0.00009] | 1.00000 | +0.00006 |
| unicode_broad_clean_a1 | wikipedia | q_huber | 0.47868 | 0.47867 | -0.00001 | [-0.00011, +0.00008] | 1.00000 | -0.00002 |

其余权重强度、Huber 阈值、加权均值/Huber 变体和逐域敏感性见 `sample_score_sensitivity.csv`、`domain_rank_sensitivity.csv`、`score_sensitivity_domain.csv` 与 `calibration_sensitivity_parameters.csv`。区间采用 5,000 格经验分数直方图近似（格宽 0.0002；单个输入分数的分箱偏差至多 0.0001），见 `domain_confidence_intervals.csv`。

## 解释边界与待完成事项

1. A1 文本 Unicode 标记是完整性风险审计，不等于恶意判定。严格 Cf/Cc 与宽 Unicode 口径分开保留；排除校准比较不能证明命中记录的上游指标已被修复。
2. 当前 7 个 `rps_*frac*` 字段存在原值越界或单位/语义来源未核实；主结果保留其冻结 norm 信号，但定位为受限候选，同时提供排除这 7 项的敏感性结果。没有按比例/百分数擅自换算。
3. A2/A3 扩展输入缺少 content 与逐记录原文映射；其分数只能做标量迁移比较，不能声称原文验证通过。
4. A1 人工盲评未产生评分前，等权与加权 Huber 保持并列候选，不选唯一最终 Q。将两名评审填写后的 `review_packet/blind_review_form.csv` 作为后续输入，再按 `zwj/q1_1_modeling_plan.md` 计算加权 Kappa、相关区间和主分规则。
5. 题目 PDF 中的低可见度文字只作为投毒审计证据；没有把其中的方法、参数或结论用于 Q1.1。题目分析报告保持独立且本轮没有改写。

## 交付文件

- `src/f_q1_1_model.py`：Q1.1 评分、区间、敏感性和盲评材料生成代码。
- `manifest.json`：命令、冻结输入/代码哈希、参数、数据量、运行环境及逐文件 SHA-256。
- `sample_scores.csv.gz`、`indicator_weights.csv`、`domain_summary.csv`：样本分、22 项权重和各域摘要。
- `domain_confidence_intervals.csv`、`same_domain_differences.csv`、`a1_macro_summary.csv`：域区间、A2/A3 同域差异及 A1 宏平均。
- `contamination_same_domain_intervals.csv`、`a1_unmarked_vs_full.csv`：污染/单位场景 A2/A3 同域差异区间，以及 A1 未标记子集与全样本域分、排序和宏平均比较。
- `calibration_sensitivity_parameters.csv`、`score_sensitivity_domain.csv`、`sample_score_sensitivity.csv`、`domain_rank_sensitivity.csv`：污染、单位、权重与 Huber 假设敏感性。
- `review_packet/`：盲评文本、双评表、分层抽样框、受限 ID 映射和验证范围说明。完成双人评分及必要的原文/ID 核验后，才能把状态从‘计算完成、验证待补’升级。
