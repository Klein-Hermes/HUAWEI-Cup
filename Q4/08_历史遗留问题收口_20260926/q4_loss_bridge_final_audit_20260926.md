# Q4 Loss–Benchmark 最终证据审计

**审计日期：** 2026-09-26  
**裁定：** `TIERED_EXPLORATORY_MAP_NOT_CALIBRATED`  
**范围：** 只裁定 Loss–Benchmark 桥接证据，不改写 Q4 v0.6 冻结包、C1 直接分数预测或其他 Q4 子项。

## 裁定摘要

当前资料足以报告按源表可比性标签分层的描述关系和留出误差；不足以把关系提升为跨家族、跨评测协议的校准器，也不足以据 Loss 可靠换算 Q1–Q3 的 Benchmark 分数。最终保留探索性结果口径。历史 v0.6 `BLOCKED_NO_FIT` 是当时严格预拟合门控的记录，继续保留；Q4/06–08 的补充拟合是另行审查的探索支线，不追溯改写历史门控。

本次没有找到足以升级证据等级的独立家族、独立数据源或独立评测协议验证。C6 扩展表新增 32 条 Medium 记录，但完整包含 C5 Medium 的 36 条和全部 7 条 High 记录；它扩大了描述样本，不构成独立数据源复验。C6 Medium 留一发布者误差略低于均值基线，但仍有约 9.5 个 Benchmark 分的 MAE，且系数相较 C5 有变化。

## 核验范围与复现

审计检查了 C5/C6 原始桥接表、Pythia Loss–Benchmark 逐行 crosswalk、C1/B1 对照、C5/C6 分层拟合和留出折表，以及两份结果 manifest。交叉核对的重点是样本重叠、来源身份、层内拟合和按模型发布者留出是否足以支持一般化结论。

从仓库根目录复现：

```powershell
python Q4/06_修订推进_20260926/scripts/q4_loss_benchmark_exploratory_mapping.py
python Q4/08_历史遗留问题收口_20260926/scripts/q4_evidence_gap_reconciliation.py
```

两条命令本次均退出码 `0`。按 manifest 重新计算输入、代码、合同与输出 SHA-256，共核对 29 项，29 项全部匹配。Q4/06 manifest 现分别记录 C5/C6，并经脚本确认 C5 的 43 行全部精确包含于 C6（C6 新增 32 行）；Q4/08 的组数列名也已修正为 `n_publisher_groups`。

独立范围 P2 复核状态为 `PASS`（P0/P1=0，修正后的 P2=0）。复核者独立重算了四组分层拟合和留出误差，所有数值与当前结果一致。该门禁仅覆盖 Loss–Benchmark 代码、结果与证据边界，不代表 Q4/08 其他主题或整个修订包已经整体验收/冻结。

## 证据审计结果

| 分层/来源 | 样本与支持 | 留出误差 vs 均值基线 | 审计判断 |
|---|---|---|---|
| C5 High（源表标签） | 7 个 Pythia 尺寸，1 个家族；Loss 2.0933–2.5978 nats/token；R²=0.158 | 留一尺寸 MAE 0.478 vs 0.376；RMSE 0.553 vs 0.424 | 仅家族内探索关系，且未胜基线；不能外推到其他家族。 |
| C5 Medium | 36 行、9 个发布者；R²=0.240 | 留一发布者 MAE 8.058 vs 7.999 | 未胜基线；只支持描述性关系。 |
| C6 Medium（扩展敏感性） | 68 行、11 个发布者；R²=0.252 | 留一发布者 MAE 9.515 vs 10.057 | MAE 约改善 5.4%，但绝对误差仍大，且不是独立外部验证。 |

High 组的身份仍有未决项：C5/C6 源表把这 7 行标为“同模型、同验证集”，但该标签不能替代 checkpoint/run 身份。逐行 crosswalk 中，只有 5/7 行的六项 Benchmark 与本地评测详情完全一致；另两行（Pythia-1b、Pythia-6.9b）存在 MATH 差异。C1 与 B1/桥接表的参数量字段 7/7 均有未解释差异，范围为约 0.29%–31.15%。主拟合继续使用 C1/C5/C6 原始表值，没有用详情 JSON 静默替换；因此精确版本连接仍未证实。

C5 与 C6 的 Medium 样本按 `(Model, N_params_B, Val_Loss, LB_Average)` 核对，C5 的 36/36 行均出现在 C6；High 的 7/7 行也重复。C6 相对 C5 新增 32 条 Medium 记录。C6 结果可作为扩展样本敏感性，不能当作一份独立复验来累计证据强度。

## 最终使用边界

- 可报告分层探索关系、样本量、留出误差和上述来源限制；Loss 单位按 Pythia 训练日志的 `val_loss` 记录为 nats/token，证据仅支持当前观察支持范围。
- 不称为已校准桥接、不报告跨家族置信区间或通用预测误差，不做 Loss 到 Benchmark 的可靠跨问换算，也不把关联斜率解释为因果效应。
- Q4 Task 3 继续直接使用 C1 `Average` 分数；不接入 Loss 桥，不将该映射用于替代 C1 观测或外推前沿。
- 若未来重新申请更高证据等级，至少须补齐可核对的 checkpoint/run ID、Loss 验证集与 Benchmark 任务/版本协议，并在多个独立家族上执行预先约定的家族级留出验证；还需显示相对基线稳定改善并报告误差不确定性。否则维持本次探索性裁定。

## 关联材料

- 分层结果与逐折误差：[`results/q4_bridge_tier_validation.csv`](results/q4_bridge_tier_validation.csv)、[`results/q4_bridge_holdout_folds.csv`](results/q4_bridge_holdout_folds.csv)。
- Pythia 来源/版本 crosswalk 与单家族探索结果：[`../06_修订推进_20260926/results/bridge_exploratory/q4_loss_benchmark_pythia_crosswalk.csv`](../06_修订推进_20260926/results/bridge_exploratory/q4_loss_benchmark_pythia_crosswalk.csv)、[`../06_修订推进_20260926/results/bridge_exploratory/q4_loss_benchmark_exploratory_report.md`](../06_修订推进_20260926/results/bridge_exploratory/q4_loss_benchmark_exploratory_report.md)。
- 权威当前汇总：[`q4_current_status.md`](q4_current_status.md)。
