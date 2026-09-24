# A4/A5 配比转移互补性敏感性分析

## 范围与定义

本分析只复用 A4/A5 的 512 个闭合配比和 13 个 Loss 目标，按 Q1.3 已有的零友好二次 p-only 敏感性模型及其冻结正则强度 lambda=0.284803587 估计转移交叉效应；不改写 Q1 主模型、不接入 B 侧、不解释为独立 Q 效应。512 个样本对应 512 个精确配方组。

参照配方取 512 个 A4/A5 配方的算术平均。候选方向限定为：在该参照配方上从份额至少 10pp 的供体域转出 10pp 到任一接收域，且二次敏感性模型预测该单一转移降低目标 Loss。对两个单一有益转移 u、v，计算：

`I_uv = L(p + 0.1(u+v)) - L(p + 0.1u) - L(p + 0.1v) + L(p)`。

I_uv < 0 表示两项转移同时实施时的 Loss 降幅超过各自降幅之和，称为该参照配方与方向下的互补；I_uv > 0 表示边际降幅减弱，称为递减/替代；区间跨零则不作方向判断。只保留两个转移共同实施后仍落在单纯形内的组合。

参照配方的合格供体域为：train_the_pile_arxiv, train_the_pile_pubmed_central, train_the_pile_github, train_the_pile_pile_cc。候选筛选基于全样本二次敏感性点估计；随后按精确配方组有放回 Bootstrap 500 次，固定 lambda 与候选方向，对每个组合给逐点 percentile 95% 区间。区间未作多重比较调整，且未包含模型族、lambda、参照配方和候选筛选的不确定性。

## 汇总

13 个 Loss 目标共形成 6551 个满足条件的转移对；全样本二次模型中互补方向 3310 个、递减/替代方向 3241 个。未校正的逐点 Bootstrap 95% 区间全负 273 个、全正 445 个、跨零 5833 个。对全部 6,551 个候选转移对采用居中 recipe-bootstrap max-|t| 近似单步家族校正后，全负 0 个、全正 39 个、跨零 6512 个；校正仅覆盖全样本模型筛出的候选集合，不覆盖候选筛选本身。

| Loss 目标 | 可行且单项降 Loss 的转移数 | 转移对数 | 点估计互补 | 点估计递减/替代 | 点区间全负* | 点区间全正* | maxT 全负† | maxT 全正† |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metric/the_pile_arxiv_val_loss | 24 | 167 | 81 | 86 | 1 | 13 | 0 | 2 |
| metric/the_pile_freelaw_val_loss | 34 | 421 | 220 | 201 | 27 | 18 | 0 | 2 |
| metric/the_pile_pubmed_central_val_loss | 34 | 371 | 187 | 184 | 9 | 13 | 0 | 1 |
| metric/the_pile_wikipedia_en_val_loss | 38 | 497 | 306 | 191 | 26 | 24 | 0 | 6 |
| metric/the_pile_dm_mathematics_val_loss | 51 | 969 | 475 | 494 | 15 | 26 | 0 | 0 |
| metric/the_pile_github_val_loss | 34 | 389 | 142 | 247 | 11 | 44 | 0 | 3 |
| metric/the_pile_stackexchange_val_loss | 33 | 383 | 151 | 232 | 11 | 54 | 0 | 4 |
| metric/the_pile_gutenberg_pg_19_val_loss | 46 | 752 | 367 | 385 | 26 | 58 | 0 | 5 |
| metric/the_pile_pile_cc_val_loss | 42 | 599 | 338 | 261 | 52 | 38 | 0 | 4 |
| metric/the_pile_ubuntu_irc_val_loss | 30 | 337 | 148 | 189 | 18 | 14 | 0 | 0 |
| metric/the_pile_hackernews_val_loss | 45 | 718 | 326 | 392 | 16 | 70 | 0 | 1 |
| metric/the_pile_pubmed_abstracts_val_loss | 39 | 519 | 275 | 244 | 45 | 53 | 0 | 5 |
| metric/the_pile_uspto_backgrounds_val_loss | 35 | 429 | 294 | 135 | 16 | 20 | 0 | 6 |

*点区间逐项 95% percentile 区间，未作多重比较调整。†maxT 区间在已筛选的 6,551 个候选交叉效应上作近似单步家族校正；候选转移由同一全样本模型选出，所以即使 maxT 也不校正前置筛选，也不是独立确认性检验。

## 解释边界

主 p-only 线性模型的转移交叉效应为 0 是其线性形式的结构结果。此处二次面给出的是 A4/A5 内、固定正则化形式下的探索性曲率诊断；不同目标可能呈现不同转移对关系，不能概括成普遍的域对排序。特别是该结果不能迁移成 B1–B5 的 p 边际效应、B 侧 M1 系数或联合 p/Q 标度律。跨尺度 A 侧预测限制、B 侧 p-Loss 连接为 0、B4/B5 不可比的门禁仍有效。

## 复现

`D:\Anaconda\python.exe src/q2_a_side_complementarity_sensitivity.py --seed 20260924 --bootstrap 500`
