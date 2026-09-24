# Q2 阶段性综合结论与当前数据收口

## 当前结论

截至本收口版本，Q2 的经典参照支路、M0 的 N/D 局部效应、B1–B5 证据审计及 p/Q 可识别性判断已经整理完成。现有数据**不能支持 B 侧 M1 或 M2 的正式系数估计**。因此当前数据包状态记为 `FROZEN_WITH_LIMITATIONS`：这表示基于现有证据的分析版本已收口，不表示题面要求的独立 p/Q 效应已经估计，也不把“不可识别”写成“没有作用”。

## 与题目模型层级的对应

题目分析报告的广义标度律为

\[L=E+A N^{-\alpha}+B D_{\mathrm{eff}}^{-\beta},\qquad D_{\mathrm{eff}}=D\exp\{h(Q,\boldsymbol{p})\}.\]

M0、M1、M2 是同一模型族内的嵌套限制：M0 令 `h=0`；M1 令 `h=g_p(p)`；M2 令 `h=g_p(p)+g_Q(Q)`。它们不是三套互不相关的模型。交互项只在数据支持时作为受限的额外扩展；当前没有进入该扩展的识别依据。

## 现有结果能回答什么

### 1. B1 经典 M0 与 N/D 局部效应

B1 有 1,176 个检查点，按 B12 映射为 8 条完整轨迹、每条 147 点；点数不等于独立轨迹数。冻结 M0 在 B1 上的拟合 `R²=0.999999817`。按模型轨迹中位数汇总，N 弹性范围约为 [-0.10333, -0.02408]，D 弹性范围约为 [-0.03998, -0.02995]；完整逐检查点效应和 Bootstrap 区间见 `q2_m0_marginal_effects_by_checkpoint.csv` 与 `q2_m0_marginal_effects_by_size.csv`。

这些是条件于冻结 M0 的局部模型效应，单位与支持范围以 M0 报告为准。区间复用 1,000 次完整轨迹 cluster Bootstrap，但独立 cluster 只有 8 条，因此只作稳定性提示。

### 2. A 侧 p 与 p+Q 证据

Q1.3 的 A4/A5 p-only 模型在同尺度 1M 留出集上的 13 目标宏平均 `R²=0.6189`；跨尺度 60M 和 1B 分别为 `-7.9336` 与 `-810.1196`。因此可报告 A 来源内的配比—Loss 关系，但不能将其作为跨尺度通用规律或直接替代 B 侧证据。

在 A4/A5 的 direct-only 嵌套 OOF 中，二次 p-only 的标准化 RMSE 为 `0.595170`，加入 `Q_covered(p)` 后为 `0.763354`，变化为 `+0.168184`。审计表明 `Q_covered(p)` 是 p 的确定性函数；该比较不识别独立 Q 效应。Q1 的 `q_huber` 冻结是操作性决定，有限人工核查没有证明其为唯一优胜候选。

### 3. B2–B5 验证边界

- B2：n=1029，RMSE=1.2431，R²=-5.0728；它是半合成压力测试，不是真实 Cerebras 外部验证。
- B3：n=4000，RMSE=0.003821，R²=0.999958；4,000 点来自 8 条 Pythia 插值轨迹，只支持同来源插值检查。
- B4：n=57；B5：n=44。两者相对 B1 均为 `NOT_COMPARABLE`；其已有误差只能分源作描述，不构成同量纲外部验证。

## 按《题目分析报告》§4.2 的四项答复

以下四项是对题面要求的收口组织，不代表题目正式拆分为 Q2.1–Q2.4。验证边界纳入第 4 项及各来源结果说明。完整公式、来源内结果和逐项证据缺口见 `Q2/03_结果/完整回答补充/v1/q2_full_answer_evidence_supplement.md`。

### 1. 广义标度律：经典 N、D 基线及 p/Q 纳入条件

当前可报告的经验基线是 B1 上冻结的 M0：`L=E+A N^(-alpha)+B D^(-beta)`。它描述 B1 支持域内的 N/D 关系；B1 训练内拟合不等于独立验证。B 侧可核验的 p—Loss 行级连接为 0，且没有 p 之外独立变化的 Q 证据，故 M1/M2 保持 `NOT_ESTIMATED`；当前结果不能写成广义联合标度律已被估计或验证。

### 2. 边际效应与无量纲弹性

B1 的 M0 已给出逐检查点 `dL/dN`、`dL/dD` 与对应弹性，弹性以该点的 M0 预测 Loss 为分母。A4/A5 的 p 关联限于来源内；B6/B7 的 Q 响应是半合成条件结果。它们不构成同一批真实 B 侧运行上 p、Q 的联合边际效应或弹性。

### 3. 领域配比替代与互补

配比 `p` 位于单纯形，域间转移必须保持总和为 1。A4/A5 可报告 1M 同来源的配比关联和替代敏感性；线性主模型的零二阶项是模型限制，不是“没有互补”的证据。二次探索的候选区间经条件 max-|t| 校正后没有稳健的负向互补信号，少数正向区间也受同样本候选筛选限制。该证据不能迁移为 B 侧或跨尺度普遍规律。

### 4. 质量—规模等 Loss 替代条件与分来源验证

可以报告在明确模型假设下的解析等 Loss 条件；例如 `dN/dQ|L=-L_Q/L_N`，其数值依赖可识别的 Q 系数。B6/B7 给出的是半合成曲面上的代数示例，不是实测 Q 效应或现实资源替代率。验证上，B1 用于 M0 拟合；B2 只作半合成压力测试，B3 只作 Pythia 同来源插值检查，B4/B5 对 B1 均为 `NOT_COMPARABLE`。B6–B8 是质量响应补充，其中 B8 与 B6/B7 方向冲突；B9/B10 是超出 B1 支持域的大尺度情景外推，不是独立验证。真实经验 Q—N 替代率及联合 p/Q 模型的外部验证均未完成。

## p/Q 门禁与当前状态

| 门禁                                 | 状态                  | 判定条件                                                                                                                 | 当前证据                                                                                                                                                                            | 结论                                                                                                          | 重开条件                                                                                                                                                                                            |
|:-------------------------------------|:----------------------|:-------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| p-effect identifiability             | FAIL_CURRENT_DATA     | Verified B-side 17-domain recipe or p(D), linked to Loss/checkpoints, with independent variation beyond N, D and family. | verified p vectors=0; Loss rows with p=0; numeric recipe vectors in B1=0.                                                                                                           | M1 NOT ESTIMATED; this is an identification limit, not evidence that p has no effect.                         | New traceable B-side source/token counts and checkpoint mapping that produce multiple non-confounded numeric recipes or p(D) trajectories.                                                          |
| Q incremental-effect identifiability | FAIL_CURRENT_EVIDENCE | A versioned Q value must link to B runs/checkpoints and vary independently of p after accounting for N and D.            | Q1 operational=q_huber (Q1-q_huber-v1); A-side Qcovered deterministic from p in 8/8 audit scenarios; independent quality effect identified in 0/8 scenarios; B-side p-Loss joins=0. | M2 NOT ESTIMATED; this is an identification limit, not evidence that Q has no effect and not a failure of Q1. | B-side Q-to-run/checkpoint links plus an identifiable design with Q variation not determined by p and not confounded with N/D/family; first resolve the p gate needed for the M2 nested comparison. |

当前模块状态：

| 模块                                        | 状态                                      | 依据与边界                                                                                |
|:--------------------------------------------|:------------------------------------------|:------------------------------------------------------------------------------------------|
| Q1 formal Q interface                       | FROZEN_WITH_LIMITATIONS                   | q_huber operational; q_equal sensitivity; review did not select a statistical winner      |
| M0 classical Loss=f(N,D)                    | COMPLETE                                  | B1 fit frozen; 1,176 checkpoints across 8 tracks                                          |
| M0 Bootstrap                                | COMPLETE_WITH_LIMITS                      | 1,000 complete-track resamples; only 8 independent trajectory clusters                    |
| M0 N/D marginal effects and elasticities    | COMPLETE                                  | Checkpoint output and 8-track summary generated                                           |
| B2/B3 validation checks                     | COMPLETE_WITH_LIMITS                      | B2 semi-synthetic; B3 same-source interpolation                                           |
| B4/B5 cross-source validation               | AUDIT_COMPLETE_VALIDATION_NOT_ESTABLISHED | All source groups remain NOT_COMPARABLE to B1                                             |
| A-side p-only evidence                      | COMPLETE_WITH_LIMITS                      | Same-scale signal; cross-scale generalization fails                                       |
| A-side p+Q sensitivity                      | COMPLETE_WITH_LIMITS                      | Qcovered derived from p; independent Q effect not identified                              |
| B-side p identifiability                    | FAIL_CURRENT_DATA                         | 0 verified p vectors joined to Loss                                                       |
| M1 (N,D,p)                                  | NOT_ESTIMATED                             | Do not fit before p gate passes                                                           |
| B-side Q incremental-effect identifiability | FAIL_CURRENT_EVIDENCE                     | No linked, independent B-side Q variation                                                 |
| M2 (N,D,p,Q)                                | NOT_ESTIMATED                             | Do not fit before p and independent-Q gates pass                                          |
| p×Q interaction extension                   | NOT_ESTIMATED                             | Optional only if evidence supports it; not fit under current gates                        |
| Q2 current-data package                     | FROZEN_WITH_LIMITATIONS                   | M0 and evidence audit closed; full p/Q increment remains unidentified with current B data |

## 可写结论

> 现有证据支持以 B1 建立经典 N、D 参照关系，并报告其局部边际效应和弹性；A4/A5 提供来源内配比实验结果，但跨尺度预测未通过。B1–B5 当前没有可核验并连接到 Loss 的数值 p 向量，且 B4/B5 Loss 与 B1 的绝对可比性未建立，因此无法识别 B 侧 p 的独立增量效应。Q1 已冻结操作性质量接口，但当前 B 侧没有可识别的 Q—p—N—D—Loss 联合变化，且 `Q_covered(p)` 为 p 的派生特征，因此不能识别 Q 在 p 之外的独立增量效应。M1/M2 记为 `NOT_ESTIMATED`，不是“无效”或“拟合失败”。

本结论是当前数据条件下的收口，不是对 p 或 Q 无效的证明。只有新出现可追溯、行级匹配且有独立变化的 B 侧数据时，才重开门禁，并按 M0→M1→M2 的嵌套顺序估计。

## 数据完整性与复现边界

本收口复用已生成的结构化结果、哈希清单和可见审计报告；不重跑模型、不重复一次性来源搜索、不修改原始附件。题目分析报告所记录的低可见度/隐藏文本按不可信内容处理，不作为题意、数据、变量或方法证据。证据行及其来源见 `q2_evidence_matrix.csv`；复现输入与输出哈希见 `q2_closeout_manifest.json`。
