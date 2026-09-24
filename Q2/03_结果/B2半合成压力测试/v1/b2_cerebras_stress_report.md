# B2 Cerebras 半合成轨迹压力测试

**性质：半合成压力测试，不是 Cerebras 原始训练日志，也不是独立外部验证。**

- B1 冻结门禁：PASS；参数文件 SHA-256：`d2bdfcf9cb8b66f32cbd633b36614a3552d290f7207cf86f4e3a02b602e4e4d8`。
- 轨迹：7 个 Cerebras-GPT 参数规模 × 6 个情景 × 147 个检查点。
- 校准：`For each Cerebras size, target_loss(D)=published_Pile_test_xent + slope_multiplier*(frozen_B1_Pythia_loss(N,D)-frozen_B1_Pythia_loss(N,D_end)) + family_offset + endpoint-centered_AR1_noise. D_end uses the published per-model token count. The terminal Cerebras anchor is never included in the interior stress metrics.`
- 正式报告分数只用非终点合成检查点；终点用于锚定，已从误差统计中排除。
- 终点位置标记不代表每个情景都等于论文锚点：族偏移情景的终点保留 ±0.15 偏移；逐点是否匹配锚点见锚点审计表。
- Cerebras 论文报告的是 Pile test xent（nats/token）；B1 的 `val_loss` 是否同一评测协议未证实，因此所有误差只作压力诊断。
- 本运行不改写题目附件中的 `cerebras_training_log.csv`，也不回写或重拟合 B1。题目目录的 `scaling_baseline.csv` 与论文逐模型终点不一致，已排除于校准；其 SHA-256 写入清单。

## 情景诊断

| scenario                   |   n_interior_points |   n_trajectory_clusters |   rmse_diagnostic_only |   mae_diagnostic_only |   bias_prediction_minus_synthetic_target |   max_abs_error_diagnostic_only | endpoint_used_for_calibration_excluded   | independent_external_validation   | comparability_caveat                                                                                  |
|:---------------------------|--------------------:|------------------------:|-----------------------:|----------------------:|-----------------------------------------:|--------------------------------:|:-----------------------------------------|:----------------------------------|:------------------------------------------------------------------------------------------------------|
| calibrated_pythia_shape    |                1022 |                       7 |               0.598479 |              0.587442 |                                 0.587442 |                        0.823887 | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |
| family_offset_minus_0p15   |                1022 |                       7 |               0.746263 |              0.737442 |                                 0.737442 |                        0.973887 | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |
| family_offset_plus_0p15    |                1022 |                       7 |               0.452155 |              0.437442 |                                 0.437442 |                        0.673887 | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |
| shallow_progress_0p75      |                1022 |                       7 |               0.659726 |              0.643959 |                                 0.643959 |                        1.04892  | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |
| steep_progress_1p25        |                1022 |                       7 |               0.540255 |              0.530924 |                                 0.530924 |                        0.822787 | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |
| ar1_noise_sd_0p02_rho_0p70 |                1022 |                       7 |               0.602043 |              0.591041 |                                 0.591041 |                        0.88432  | True                                     | False                             | B1 val_loss evaluation protocol is not established as identical to published Cerebras Pile test xent. |

## 再现

```powershell
"D:\Anaconda\python.exe" src/f_q2_b2_semisynthetic.py --seed 20260924
```

## 来源

Cerebras-GPT 原始模型规格、各模型训练 token 数与 Pile 测试损失取自 [arXiv:2304.03208v1, 2023-04-06](https://arxiv.org/html/2304.03208v1)。
