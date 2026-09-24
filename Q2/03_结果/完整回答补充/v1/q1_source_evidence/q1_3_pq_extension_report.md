# Q1.3 p+Q 受限扩展实验与正式比较

## 建模依据与解释边界

固定领域质量值下，完整 (Q_{mix}=p^Tq) 是 (p) 的线性组合；已有线性 p-only 模型再加该项时秩不增加，不能识别独立 Q 系数。映射表只为部分领域提供 Q，因此本实验的 `Q_covered(p)` 定义为：已映射份额上的质量分加权均值，即 \(\sum_{j\in M}p_jq_j/\sum_{j\in M}p_j\)。它是 p 的确定性非线性变换，可用来检验预测基函数是否有用，不能解释成独立质量信息或因果效应。

Q1.1 双评盲表仍为空，故本实验只使用等权、Huber 及预先已有的污染敏感性候选，不产生或宣称最终 Q。A1 标量质量信号的 7 个 `rps_*frac*` 字段单位/语义尚未完全核实；“排除 7 字段”和宽 Unicode 未标记样本重校准仅作为已存在的候选分敏感性。Q 点估计及映射误差没有进入本模型的预测区间。

## 比较设计

- 比较模型：原单纯形线性 p-only；既有 153 项零友好二次 p-only；线性 p + 单一 `Q_covered(p)` 特征。
- 数据划分：A4/A5 训练上的 5 折分组嵌套 CV；所有模型共用相同外层/内层种子和同一配方分组。Lambda 只在 A4/A5 内层 CV 选择。
- 冻结评估：A6/A7 同尺度、A8/A9 60M、A10/A11 1B；全部此前已接触，不作为新盲测。每种映射仅在 `Q_covered` 有定义的共同支持行上配对比较。A12–A15 估算表不进入模型选择或测试指标。
- 映射：`direct_only` 为主证据；`direct_plus_near` 是近似映射敏感性。11 个 inferred 领域始终不填值。无覆盖配方不插补、不置零。
- 指标：逐目标 MAE/RMSE/R²/Spearman、13 目标宏平均和按训练折尺度计算的标准化 RMSE。差值定义为 p+Q 减 p-only，负值表示 p+Q 误差较低；不把描述性差异解释为显著性。

## 训练集有效覆盖

- direct-only：478/512 条有 Q_covered；34 条不进入三模型共同比较。
- direct+near：507/512 条有 Q_covered；5 条不进入三模型共同比较。

## 训练 OOF 关键比较

下表中的四种 Q 列仅表示 Q1.1 候选与敏感性场景；模型没有据外部 Loss 选择其中任一个。详细逐目标结果和所有冻结集比较见 `target_metrics.csv`、`model_summary.csv` 与 `paired_comparison.csv`。

| 映射 | Q 候选场景 | 模型 | 嵌套 OOF 标准化 RMSE | 宏 RMSE | 与二次 p-only 的标准化 RMSE 差 |
|---|---|---|---:|---:|---:|
| direct_only | q_equal_candidate | p_only_linear | 0.627628 | 0.472048 | +0.032440 |
| direct_only | q_equal_candidate | p_only_quadratic | 0.595170 | 0.453641 | +0.000000 |
| direct_only | q_equal_candidate | p_plus_Qcovered | 0.753253 | 0.560450 | +0.158071 |
| direct_only | q_huber_candidate | p_only_linear | 0.627628 | 0.472048 | +0.032440 |
| direct_only | q_huber_candidate | p_only_quadratic | 0.595170 | 0.453641 | +0.000000 |
| direct_only | q_huber_candidate | p_plus_Qcovered | 0.763354 | 0.566737 | +0.168174 |
| direct_only | q_huber_exclude_7_frac_sensitivity | p_only_linear | 0.627628 | 0.472048 | +0.032440 |
| direct_only | q_huber_exclude_7_frac_sensitivity | p_only_quadratic | 0.595170 | 0.453641 | +0.000000 |
| direct_only | q_huber_exclude_7_frac_sensitivity | p_plus_Qcovered | 0.785202 | 0.583084 | +0.190023 |
| direct_only | q_huber_unicode_broad_sensitivity | p_only_linear | 0.627628 | 0.472048 | +0.032440 |
| direct_only | q_huber_unicode_broad_sensitivity | p_only_quadratic | 0.595170 | 0.453641 | +0.000000 |
| direct_only | q_huber_unicode_broad_sensitivity | p_plus_Qcovered | 0.763335 | 0.566723 | +0.168155 |
| direct_plus_near | q_equal_candidate | p_only_linear | 0.630189 | 0.490394 | +0.040722 |
| direct_plus_near | q_equal_candidate | p_only_quadratic | 0.589455 | 0.462889 | +0.000000 |
| direct_plus_near | q_equal_candidate | p_plus_Qcovered | 0.746233 | 0.573914 | +0.156831 |
| direct_plus_near | q_huber_candidate | p_only_linear | 0.630189 | 0.490394 | +0.040722 |
| direct_plus_near | q_huber_candidate | p_only_quadratic | 0.589455 | 0.462889 | +0.000000 |
| direct_plus_near | q_huber_candidate | p_plus_Qcovered | 0.751400 | 0.580039 | +0.161995 |
| direct_plus_near | q_huber_exclude_7_frac_sensitivity | p_only_linear | 0.630189 | 0.490394 | +0.040722 |
| direct_plus_near | q_huber_exclude_7_frac_sensitivity | p_only_quadratic | 0.589455 | 0.462889 | +0.000000 |
| direct_plus_near | q_huber_exclude_7_frac_sensitivity | p_plus_Qcovered | 0.773337 | 0.590777 | +0.183944 |
| direct_plus_near | q_huber_unicode_broad_sensitivity | p_only_linear | 0.630189 | 0.490394 | +0.040722 |
| direct_plus_near | q_huber_unicode_broad_sensitivity | p_only_quadratic | 0.589455 | 0.462889 | +0.000000 |
| direct_plus_near | q_huber_unicode_broad_sensitivity | p_plus_Qcovered | 0.751407 | 0.580040 | +0.162002 |

## 结果判读与模型决策

以下 Δ 为 `p_plus_Qcovered` 减二次 p-only 的宏标准化 RMSE；负值表示该场景中 p+Q 误差较低。每个范围覆盖四种 Q1.1 候选场景，不是置信区间：

- A4/A5 嵌套 OOF：direct-only +0.1581 至 +0.1900；direct+near +0.1568 至 +0.1839。
- 已接触的同尺度 A6/A7（1M）：direct-only +0.1789 至 +0.2117；direct+near +0.1675 至 +0.1960。
- 已接触的跨尺度 A8/A9（60M）：direct-only -0.0992 至 -0.0753；direct+near -0.0646 至 -0.0149。
- 已接触的跨尺度 A10/A11（1B）：direct-only +0.0353 至 +0.0617；direct+near +0.0683 至 +0.1328。

决策：`Q_covered(p)` 确实给设计矩阵增加了一个由配比确定的非线性方向，但相对二次 p-only，它在嵌套 OOF 和同尺度 1M 评估均更差，在 1B 评估也更差；只有 60M 评估更好。改进方向不稳定，当前结果不支持把 p+Q 作为通用预测模型或宣称其带来独立质量信息。就本次比较的预测误差而言，保留二次 p-only 作为首选基线；60M 的局部改善仅作为跨尺度现象记录，不据此升级模型。所有外部评估集此前已接触，且上游 Q 仍是待盲评候选，因此这是一项受限、描述性的比较，不构成新盲测或显著性结论。

## 可识别性审计

| 映射 | Q 场景 | Qmix partial 秩增量 | Qcovered 秩增量 | Qcovered 去除线性 p 后的相对残差 SD | 独立 Q 效应可识别 |
|---|---|---:|---:|---:|---|
| direct_only | q_equal_candidate | 0 | 1 | 0.632004 | 否 |
| direct_only | q_huber_candidate | 0 | 1 | 0.625166 | 否 |
| direct_only | q_huber_unicode_broad_sensitivity | 0 | 1 | 0.62517 | 否 |
| direct_only | q_huber_exclude_7_frac_sensitivity | 0 | 1 | 0.622431 | 否 |
| direct_plus_near | q_equal_candidate | 0 | 1 | 0.456066 | 否 |
| direct_plus_near | q_huber_candidate | 0 | 1 | 0.443827 | 否 |
| direct_plus_near | q_huber_unicode_broad_sensitivity | 0 | 1 | 0.443901 | 否 |
| direct_plus_near | q_huber_exclude_7_frac_sensitivity | 0 | 1 | 0.439117 | 否 |

解释：若 `Qmix partial` 秩增量为 0，说明已映射部分的未归一化加权和是 p 线性组合；这只是代数诊断，未把未知领域当成零质量。`Qcovered` 的秩增量只说明受限非线性基函数给设计矩阵增加了预测自由度。因为它完全由 p 计算，仍不能识别独立质量效应。

## 污染与数据完整性处理

- 没有解析 PDF 隐藏文字、报告正文命令或样本文本作为程序指令；`题目分析报告.md` 未被修改。
- Q1.3 数值输入只来自 A4/A5 训练；A6–A11 只在训练内选定后评估；A12–A15 未读取。A4/A5 原行不删除、不修值，沿用冻结脚本的原始行闭合比例。
- Q1.1 Q 候选来自既有结果表，不重新评分。排除 7 个字段和宽 Unicode 校准只检验上游 Q 候选变化，不表示源文本已清洗或数据已证实被投毒。

## 结论规则

本实验只回答 `Q_covered(p)` 作为确定性非线性预测基函数是否改善既有 p-only 预测。只有当改进在 A4/A5 嵌套 OOF、已接触的 1M 同尺度评估和主要 Q/映射敏感性下方向一致，且相对二次 p-only 仍有稳定增益时，才可称它为有条件的预测扩展；无论指标如何，都不得称为独立 Q 质量效应。若只胜过线性 p-only、不胜过二次 p-only，说明结果不超出已有配比非线性表达。

## 复现

```powershell
& 'C:\Users\86147\AppData\Local\Programs\Python\Python310\python.exe' -B -u src\f_q1_3_pq_extension.py --mode full
```

输入与输出 SHA-256、随机种子和超参数见 `manifest.json`。
