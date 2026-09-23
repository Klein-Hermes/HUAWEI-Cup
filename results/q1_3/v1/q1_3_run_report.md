# Q1.3 p-only 配比—Loss：本轮运行摘要

> 该结果由脚本对项目 A4/A5 及冻结评估文件实际计算。它是建模进展稿，不等于 Q1.3 全部敏感性、制图和最终验收完成。

## 数据审计

| split | role | n_recipes | n_mixture_domains | n_loss_targets | row_sum_min | row_sum_max | rows_abs_sum_deviation_gt_0_001 | zero_cells | rows_with_zero | exact_duplicate_recipes | nearest_pair_l1_after_closure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train_1m | fit | 512 | 17 | 13 | 0.996 | 1.0030000000000001 | 47 | 3928 | 511 | 0 | 0.022022022021999994 |
| test_1m | heldout_same_scale | 256 | 17 | 13 | 0.9969999999999999 | 1.0030000000000001 | 22 | 1871 | 255 | 0 | 0.16599999999999998 |
| test_60m | heldout_cross_scale | 256 | 17 | 13 | 0.9969999999999999 | 1.0030000000000001 | 22 | 1871 | 255 | 0 | 0.16599999999999998 |
| test_1b | heldout_cross_scale | 64 | 17 | 13 | 0.9979999999999999 | 1.0019999999999998 | 10 | 360 | 64 | 0 | 0.21300100100099995 |
| estimate_10b | estimate_consistency_only | 63 | 17 | 13 | 0.998 | 1.002 | 5 | 487 | 63 | 0 | 0.11611611611600003 |
| estimate_70b | estimate_consistency_only | 63 | 17 | 13 | 0.998 | 1.002 | 5 | 487 | 63 | 0 | 0.11611611611600003 |

## 训练内嵌套交叉验证

在完整 A4/A5 上按分组 CV 选出的最终模型族：`simplex_ridge`。
外层分组 CV 对‘仅在各外层训练折内重新选择模型族’的整套流程给出宏平均标准化 RMSE：`0.6330482361236224`。
候选模型单独列出的外层分数用于描述；最终家族选择不使用这些外层验证标签，而在完整 A4/A5 的内层 CV 上完成。A6–A11 和 A12–A15 未用于调参或选模。

| split | role | model | macro_mae | macro_rmse | macro_r2 | macro_spearman_rho | nested_macro_normalized_rmse | variant | worst_rmse_target | worst_rmse |
|---|---|---|---|---|---|---|---|---|---|---|
| nested_cv_closed | out_of_fold | mean | 0.6441103235431791 | 0.7543173716305791 | -0.004713706478550397 | -0.08621897885266007 | 1.002920333223083 | closed | metric/the_pile_dm_mathematics_val_loss | 1.53719661242288 |
| nested_cv_closed | out_of_fold | reference_ols | 0.4190022561901845 | 0.49877654745115446 | 0.5977319975907618 | 0.8180635004265142 | 0.634226726457778 | closed | metric/the_pile_dm_mathematics_val_loss | 1.2092532587823899 |
| nested_cv_closed | out_of_fold | simplex_ridge | 0.4197244134302251 | 0.49780276869786505 | 0.5992323888452079 | 0.8207703066830229 | 0.6330482361236224 | closed | metric/the_pile_dm_mathematics_val_loss | 1.2071686403526285 |
| nested_cv_closed | out_of_fold | nested_selected | 0.4197244134302251 | 0.49780276869786505 | 0.5992323888452079 | 0.8207703066830229 | 0.6330482361236224 | closed | metric/the_pile_dm_mathematics_val_loss | 1.2071686403526285 |
| nested_cv_raw | out_of_fold | mean | 0.6441103235431791 | 0.7543173716305791 | -0.004713706478550397 | -0.08621897885266007 | 1.002920333223083 | raw | metric/the_pile_dm_mathematics_val_loss | 1.53719661242288 |
| nested_cv_raw | out_of_fold | reference_ols | 0.41900556967127645 | 0.49877538030242086 | 0.5977368733183177 | 0.8180158188212892 | 0.6342235746886887 | raw | metric/the_pile_dm_mathematics_val_loss | 1.2091620269296406 |
| nested_cv_raw | out_of_fold | simplex_ridge | 0.41972301712812465 | 0.4977899080050427 | 0.5992452048241448 | 0.8208049829636437 | 0.6330371752644487 | raw | metric/the_pile_dm_mathematics_val_loss | 1.2070908320621487 |
| nested_cv_raw | out_of_fold | nested_selected | 0.41972301712812465 | 0.4977899080050427 | 0.5992452048241448 | 0.8208049829636437 | 0.6330371752644487 | raw | metric/the_pile_dm_mathematics_val_loss | 1.2070908320621487 |
| nested_cv_closed | out_of_fold_robustness_sensitivity | simplex_huber_ridge_sensitivity | 0.4177127296788562 | 0.4968860558740841 | 0.5999821797575641 | 0.8224461779008461 | 0.6328063605411687 | closed | metric/the_pile_dm_mathematics_val_loss | 1.2058657712568484 |
| nested_cv_closed | nonlinear_sensitivity_only | quadratic_ridge_sensitivity | 0.37506792891075325 | 0.4677903708529232 | 0.653103439958543 | 0.8309729116213435 | 0.5890145267134201 | closed | metric/the_pile_dm_mathematics_val_loss | 1.3333376451500287 |

## 冻结的真实验证集

以下 A6–A11 结果仅评估已在 A4/A5 内部选定并冻结的模型；跨尺度不作事后校准。闭合配比为主结果，原始配比为闭合敏感性。

| split | role | model | macro_mae | macro_rmse | macro_r2 | macro_spearman_rho | variant | worst_rmse_target | worst_rmse |
|---|---|---|---|---|---|---|---|---|---|
| test_1m | heldout_same_scale | simplex_ridge | 0.38761438148965993 | 0.453863022539131 | 0.6188879050595115 | 0.832661716287832 | closed | metric/the_pile_dm_mathematics_val_loss | 1.1634313403714827 |
| test_1m | heldout_same_scale | simplex_ridge | 0.3875887445252036 | 0.45385260992252713 | 0.6188871692648076 | 0.8326481812419669 | raw | metric/the_pile_dm_mathematics_val_loss | 1.1632257030963862 |
| test_1m | heldout_same_scale | simplex_huber_ridge_sensitivity | 0.38551909326919964 | 0.4520283966817628 | 0.6221707320574269 | 0.8345860586826768 | closed | metric/the_pile_dm_mathematics_val_loss | 1.1609007426381552 |
| test_1m | heldout_same_scale | quadratic_ridge_sensitivity | 0.34307378889726825 | 0.41981408935485853 | 0.6817412094797302 | 0.8441488987094389 | closed | metric/the_pile_dm_mathematics_val_loss | 1.2062186760526614 |
| test_60m | heldout_cross_scale | simplex_ridge | 1.5125359346603569 | 1.5590162377435113 | -7.933569486978265 | 0.8313861757663256 | closed | metric/the_pile_github_val_loss | 2.633286160363353 |
| test_60m | heldout_cross_scale | simplex_ridge | 1.512591828036681 | 1.5590698423940488 | -7.934153261900644 | 0.8313345115058893 | raw | metric/the_pile_github_val_loss | 2.633370284710832 |
| test_60m | heldout_cross_scale | simplex_huber_ridge_sensitivity | 1.5078909933273446 | 1.5543916910706261 | -7.8769516010230465 | 0.8332122515273693 | closed | metric/the_pile_github_val_loss | 2.6167958561464353 |
| test_60m | heldout_cross_scale | quadratic_ridge_sensitivity | 1.4718783413045768 | 1.517020651271766 | -7.48734952615411 | 0.8461882339736254 | closed | metric/the_pile_github_val_loss | 2.557574133872994 |
| test_1b | heldout_cross_scale | simplex_ridge | 3.1829562651714958 | 3.1969897294857073 | -810.1195857238465 | 0.7268455903071287 | closed | metric/the_pile_github_val_loss | 4.427974887788331 |
| test_1b | heldout_cross_scale | simplex_ridge | 3.1828455403924045 | 3.196884811915695 | -810.0814074898954 | 0.7259897154127924 | raw | metric/the_pile_github_val_loss | 4.42781104982988 |
| test_1b | heldout_cross_scale | simplex_huber_ridge_sensitivity | 3.1810319144600787 | 3.1954702024569523 | -808.6068878056595 | 0.7262890955198648 | closed | metric/the_pile_github_val_loss | 4.41466917870357 |
| test_1b | heldout_cross_scale | quadratic_ridge_sensitivity | 2.962477127048354 | 2.982685123024072 | -672.8919479553898 | 0.8002993801070724 | closed | metric/the_pile_github_val_loss | 4.047076039739268 |

## 估计表一致性检查

A12–A15 对应的 10B/70B 表为估计结果。其误差只称为相对估计表的偏差，不解释为真实测试精度。

| split | role | model | macro_mae | macro_rmse | macro_r2 | macro_spearman_rho | variant | worst_rmse_target | worst_rmse |
|---|---|---|---|---|---|---|---|---|---|
| estimate_10b | estimate_consistency_only | simplex_ridge | 3.5730412973210024 | 3.601552358100534 | -236.85817604692062 | 0.4896256082155573 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.451284967887 |
| estimate_10b | estimate_consistency_only | simplex_ridge | 3.573040906116955 | 3.6015521347197796 | -236.85737565532463 | 0.489780697102644 | raw | metric/the_pile_ubuntu_irc_val_loss | 4.451270379513749 |
| estimate_10b | estimate_consistency_only | simplex_huber_ridge_sensitivity | 3.567815976029194 | 3.5960976714580317 | -236.18882164098534 | 0.4883018162964898 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.455094712965432 |
| estimate_10b | estimate_consistency_only | quadratic_ridge_sensitivity | 3.565153585766789 | 3.597603720368892 | -235.9774207298281 | 0.5241379351117216 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.462146556485459 |
| estimate_70b | estimate_consistency_only | simplex_ridge | 3.9556156562953615 | 3.9847737743741685 | -385.31394032279627 | 0.4131620555131711 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.932324143784353 |
| estimate_70b | estimate_consistency_only | simplex_ridge | 3.955615265091313 | 3.984773390246092 | -385.31282702124656 | 0.41328575775397225 | raw | metric/the_pile_ubuntu_irc_val_loss | 4.9323095481370425 |
| estimate_70b | estimate_consistency_only | simplex_huber_ridge_sensitivity | 3.950390335003554 | 3.9793060969502703 | -384.2082899947624 | 0.4118290311260449 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.93617701838213 |
| estimate_70b | estimate_consistency_only | quadratic_ridge_sensitivity | 3.947592086701868 | 3.9817744899295273 | -383.99765853343433 | 0.4399792365645823 | closed | metric/the_pile_ubuntu_irc_val_loss | 4.944704682742526 |

## OOF 残差与替代效应不确定性

残差诊断基于外层折中、仅由各折训练数据选择模型族后产生的 OOF 预测；分目标总体及预测值五分位诊断见 `residual_diagnostics_nested_selection.csv`。
最大 OOF 逐目标 RMSE 为 `metric/the_pile_dm_mathematics_val_loss`：1.2072；最大预测五分位平均偏差出现在 `metric/the_pile_dm_mathematics_val_loss` 第 5 组，预测减观测为 -0.5174。
配比替代效应使用 500/500 次有效 pairs bootstrap；95% percentile 区间为逐点区间、未作多重比较调整，并以最终模型族和正则强度固定为条件。136 个域对中 106 个至少有一条配方可支持10个百分点替代，30 个没有该份额支持；受支持的 1378 个目标×域对区间中，214 个全为正、535 个全为负、629 个跨零。每项支持量见 `substitution_effects_10pp_bootstrap.csv`，这些区间不作显著性结论。

## 选择与限制

- 完整 A4/A5 分组CV选择模型：`simplex_ridge`；外层模型选择流程标准化 RMSE：`0.6330482361236224`。
- 闭合配比最终正则强度：`0.0001`；原始配比敏感性正则强度：`0.0001`。
- 主候选包含训练均值空模型、固定参考配比系数为 0 的无正则线性对照，以及原单纯形 ridge；Huber ridge 单列为污染敏感性，不参与主模型选族。两种单纯形回归均对每个目标约束17个配比系数和为0。线性系数按配比替代效应解释。
- 参考分量线性对照的固定参考域为 `train_the_pile_uspto_backgrounds`；训练设计矩阵条件数超过 1e+07 时标记病态。
- 算法依据为闭合单纯形的可识别参数化和 A4/A5 内部验证；隐藏文本建议既不授权也不否决任何方法。
- 超参数与候选模型选择仅基于 A4/A5；跨尺度结果不能当作训练尺度校准数据。
- Huber 稳健敏感性 nested-CV 宏平均标准化 RMSE：`0.6328063605411687`；完整训练集 CV 选得 λ=`3.1622776601683795e-05`。它降低大条件残差的影响，不证明数据被污染，也不能抵抗高杠杆配比错误；权重明细见 `huber_training_residual_weights.csv`。
- 零友好的二次配比敏感性 nested-CV 宏平均标准化 RMSE：`0.5890145267134201`；A4/A5 分组 CV 选得 λ=`0.2848035868435799`，153 个原配比/两两乘积项仅作为敏感性比较，不参与主模型选族。逐项系数见 `parameters_quadratic_sensitivity.csv`。
- 行和审计更正：分析报告早期版本记录 189 行偏差超过 0.001；已按当前 A4 原表十进制逐行重算更正为 47 行，哈希绑定脚本复算为 `47` 行。旧值留作历史差异记录。
- 当前未运行 ILR：零配比来源尚未确认。二次面在原配比空间计算，对数变换没有必要；它不区分结构零和舍入零。Q1.3 已完成 9 张图表（PDF/SVG/300 DPI PNG/灰度预览），Python 版面 QA 全部通过，三类候选图审计通过；两条 SVG 色标位图提示和 PDF Type0 字体检测误报的核查说明见 `figures/q1_3_model/图表契约.md`。独立最终 P2 复核已通过，当前无待办。
