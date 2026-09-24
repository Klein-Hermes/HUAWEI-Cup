# Q2 经典 Scaling Law 基线运行报告

## 状态

**基线已拟合、验证受限（B2/B3 非独立真实验证；B4/B5 Loss 可比性与逐点来源未证实）**。此产物冻结的是经典参考支路 `Loss=f(N,D)`，不代表完整 Q2 已完成。Q2 后续仍需使用正式冻结的质量分数 Q 与领域配比 p 扩展模型。

Gate 0：36/36 项通过；B1 通过 B12 映射到 8 条 Pythia 模型轨迹，每条 147 个检查点。`run_id` 每行唯一，未用作 cluster。

## 模型与拟合

拟合形式：

`L = E + A*N_params_B**(-alpha) + B*D_tokens_B**(-beta)`

只在 B1 上做有界非线性最小二乘。参数约束为 `0 <= E < min(B1 val_loss)` 且 `A, alpha, B, beta > 0`；固定多初值拟合 24 次。

- `E` = 1.689797563
- `A` = 0.3539803207
- `alpha` = 0.3399765819
- `B` = 1.240305584
- `beta` = 0.2798781285

训练指标：RMSE=0.0001464961，MAE=0.0001052319，R²=0.9999998，Spearman=0.9999996。这些是 B1 拟合内指标。

优化诊断：成功初值 24/24；近最优解数量 24；缩放 Jacobian 秩 5/5，条件数 56.37775912944804。可辨识性状态：`未触发预设警告`。详细初值结果和删项诊断分别见 `multistart_fits.csv`、`dropped_term_diagnostics.json`。

## 轨迹 Bootstrap 与留一规模分析

对 B1 的 8 个完整模型轨迹做 1000 次 cluster Bootstrap，成功 1000 次、失败 0 次。区间见 `cluster_bootstrap_intervals.json` 和 `cluster_bootstrap_estimates.csv`。**只有 8 个 cluster，区间对少量轨迹高度敏感，应视作稳定性提示而非高精度总体区间。**

留一参数规模敏感性（每次整条 147 点轨迹留出）：

| held_out_N_params_B | holdout_rmse | holdout_mae |
| --- | --- | --- |
| 0.070542 | 0.00022023 | 0.00016876 |
| 0.1624 | 0.00018182 | 0.00013714 |
| 0.40901 | 0.00013511 | 0.00010756 |
| 1.0409 | 0.0001546 | 0.00010866 |
| 1.4162 | 0.00013192 | 9.9969e-05 |
| 2.7828 | 0.00011441 | 8.1915e-05 |
| 6.861 | 0.00013504 | 8.3156e-05 |
| 11.966 | 0.00010198 | 7.5869e-05 |

## B2–B5 留出验证

基线在读取验证误差之前只由 B1 拟合。验证集没有回流拟合或调参，且不汇总成一个跨来源分数。B2 是半合成 Cerebras 轨迹；B3 是 Pythia 插值轨迹；它们不作为独立真实验证。B4 是真实跨族数据；B5 是多来源文献数据，但这两个 CSV 不提供统一 tokenizer/评测集协议，Loss 量尺可比性未被证实。

| dataset | role | n | rmse | mae | bias_pred_minus_obs | r2 | spearman | max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2 | separate: semisynthetic; same field name does not establish independent real validation | 1029 | 1.2431 | 1.1784 | -1.1784 | -5.0728 | 0.7881 | 3.5117 |
| B3 | separate: interpolation from Pythia trajectories; not independent real validation | 4000 | 0.0038213 | 0.002893 | -0.0019683 | 0.99996 | 0.99999 | 0.014734 |
| B4 | separate: curated cross-family snapshots; row-level provenance and evaluation protocol are unknown | 57 | 0.29267 | 0.22726 | -0.22101 | 0.6045 | 0.98298 | 1.0477 |
| B5 | separate: literature values from multiple sources; CSV lacks common evaluation protocol | 44 | 0.1976 | 0.17072 | -0.02057 | 0.73056 | 0.95881 | 0.45084 |

按参数规模、轨迹、模型族和 B5 文献来源的分层指标见 `validation_metrics_by_stratum.csv`。预测逐点见 `b2_predictions.csv` 至 `b5_predictions.csv`。

## 结论边界

- B6–B8 含 Q 的半合成补充数据不进入经典基线；不估计 Q、p、质量—规模替代条件或 Q3 配置。
- B9–B10 不进入拟合，只留给未来大尺度外推分析。
- B2/B3 支持合成迁移/轨迹形状检查，不支持独立真实外部验证结论。
- B4/B5 的误差分源报告；没有统一 Loss 量尺证据，不合并评分。
- B1 只有八条模型轨迹；残差行数 1,176 不能解释为 1,176 个独立训练实验。

## B4/B5 Loss 协议审计补充（2026-09-24）

已完成逐来源协议审计，详见 [`b4_b5_loss_comparability_audit.md`](b4_b5_loss_comparability_audit.md)、[`loss_protocol_matrix.csv`](loss_protocol_matrix.csv) 和 [`loss_protocol_evidence_sources.md`](loss_protocol_evidence_sources.md)。当前结论为 B4/B5 对 B1 均是 `NOT_COMPARABLE`，没有任何来源达到 `ABSOLUTE_COMPARABLE`。B5 个别论文内、单一模型族的趋势只有在补齐逐点原文定位后才可记为 `TREND_ONLY`；B4 目前也没有可审计的来源内趋势。

审计另发现 B4 的七条 Cerebras-GPT 记录均记为 2,050B tokens，所检查的 Cerebras 原论文表格不能直接支持这些行；这表明行级来源未建立，不足以断言原始值错误。B5 的 GPT-3 行保留 `Kaplan et al. 2020` 原标签；Brown et al. 2020 仅是因模型规模/训练量网格相似而提出的候选，精确 Loss 未定位，不能先行改标。原始 CSV 未被修改。

此门禁只冻结 B4/B5 对 B1 的同量纲外部验证支路，不阻断 Q2 其余工作。M1 另由逐样本、可追溯的配比 `p` 和 Q2 拟合器门禁控制，不依赖最终 Q 或 B4/B5 绝对可比性；M2 仍需 M1 数据链和冻结版本的 `Q_final`。

## 复现

唯一复现命令（使用本次验证通过的解释器）：

```powershell
"D:\Anaconda\python.exe" src/f_q2_scaling_baseline.py --mode full --seed 20260923
```

本次实际运行环境：Python 3.13.9，解释器 `D:\Anaconda\python.exe`。当前主机裸命令 `python` 会解析到独立 Python 3.10；其 SciPy 线性代数调用曾触发 Windows 异常 `0xc06d007f`。复现时请使用上面的解释器路径，或先切换到已验证可工作的 Python/SciPy 环境。输入 SHA-256、Python/依赖版本和产物清单见 `复现清单.json`。原始附件保持只读。
