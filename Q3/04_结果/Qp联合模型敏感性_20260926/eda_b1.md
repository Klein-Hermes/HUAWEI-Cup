# Data profile: Q2/03_结果/经典ScalingLaw基线/v1/b1_fitted_predictions.csv

**Shape:** 1176 rows × 19 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `run_id` | continuous | 1176 | 0 | mean=620, sd=356, range=[8, 1.23e+03], skew=0.00 (approximately symmetric); -> log axis |
| `N_params_B` | continuous | 1176 | 0 | mean=3.09, sd=3.95, range=[0.0705, 12], skew=1.35 (highly skewed); outliers=147 (IQR); -> log axis |
| `D_tokens_B` | continuous | 1176 | 0 | mean=147, sd=88.9, range=[0.134, 300], skew=0.01 (approximately symmetric); -> log axis |
| `C_FLOPs_1e21` | continuous | 1176 | 0 | mean=2.72, sd=4.39, range=[0.0001, 21.5], skew=2.27 (highly skewed); outliers=159 (IQR); -> log axis |
| `steps` | continuous | 1176 | 0 | mean=7e+04, sd=4.24e+04, range=[64, 1.43e+05], skew=0.01 (approximately symmetric); -> log axis |
| `batch_tokens_M` | continuous | 1176 | 0 | mean=2.1, sd=0.00116, range=[2.09, 2.1], skew=-8.39 (highly skewed); outliers=16 (IQR) |
| `lr` | continuous | 1176 | 0 | mean=0.000246, sd=4.13e-05, range=[0.00019, 0.000318], skew=0.37 (approximately symmetric) |
| `wd` | continuous | 1176 | 0 | mean=0.0555, sd=0.025, range=[0.01, 0.1], skew=-0.03 (approximately symmetric) |
| `precision` | categorical | 1176 | 0 | 2 levels: bf16(797), fp16(379); min_group_n=379 |
| `gpu_days` | continuous | 1176 | 0 | mean=101, sd=163, range=[0, 799], skew=2.27 (highly skewed); outliers=159 (IQR) |
| `step_time_ms` | continuous | 1176 | 0 | mean=532, sd=398, range=[100, 1.31e+03], skew=0.79 (moderately skewed) |
| `train_loss` | continuous | 1176 | 0 | mean=2.35, sd=0.329, range=[1.94, 4.47], skew=2.07 (highly skewed); outliers=35 (IQR) |
| `val_loss` | continuous | 1176 | 0 | mean=2.48, sd=0.342, range=[2.09, 4.74], skew=2.11 (highly skewed); outliers=33 (IQR) |
| `ppl` | continuous | 1176 | 0 | mean=12.9, sd=7.59, range=[8.11, 114], skew=6.30 (highly skewed); outliers=52 (IQR) |
| `grad_norm_avg` | continuous | 1176 | 0 | mean=1.59, sd=2.45, range=[0.392, 25.8], skew=4.91 (highly skewed); outliers=80 (IQR) |
| `observed_val_loss` | continuous | 1176 | 0 | mean=2.48, sd=0.342, range=[2.09, 4.74], skew=2.11 (highly skewed); outliers=33 (IQR) |
| `predicted_val_loss` | continuous | 1176 | 0 | mean=2.48, sd=0.342, range=[2.09, 4.74], skew=2.11 (highly skewed); outliers=33 (IQR) |
| `residual_pred_minus_obs` | continuous | 1176 | 0 | mean=2.64e-14, sd=0.000147, range=[-0.000648, 0.00092], skew=0.01 (approximately symmetric); outliers=50 (IQR) |
| `abs_error` | continuous | 1176 | 0 | mean=0.000105, sd=0.000102, range=[1.1e-07, 0.00092], skew=2.32 (highly skewed); outliers=48 (IQR); -> log axis |

## Group structure
- Grouped by: `N_params_B`
- Number of groups: 8
- Group size: min=147, median=147, max=147

## Correlations (Pearson, sorted by |r|)
- `val_loss` ↔ `observed_val_loss` : r = 1.000 (very strong)
- `D_tokens_B` ↔ `steps` : r = 1.000 (very strong)
- `C_FLOPs_1e21` ↔ `gpu_days` : r = 1.000 (very strong)
- `val_loss` ↔ `predicted_val_loss` : r = 1.000 (very strong)
- `observed_val_loss` ↔ `predicted_val_loss` : r = 1.000 (very strong)
- `train_loss` ↔ `predicted_val_loss` : r = 0.991 (very strong)
- `train_loss` ↔ `val_loss` : r = 0.991 (very strong)
- `train_loss` ↔ `observed_val_loss` : r = 0.991 (very strong)
- `run_id` ↔ `lr` : r = -0.981 (very strong)
- `N_params_B` ↔ `step_time_ms` : r = 0.972 (very strong)
- ... +143 more pairs

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['run_id', 'N_params_B', 'D_tokens_B', 'C_FLOPs_1e21', 'steps']）或 pairplot 散点矩阵
- run_id 跨数个量级（8 ~ 1.23e+03）→ 用对数 y 轴
- N_params_B 跨数个量级（0.0705 ~ 12）→ 用对数 y 轴
- D_tokens_B 跨数个量级（0.134 ~ 300）→ 用对数 y 轴
- C_FLOPs_1e21 跨数个量级（0.0001 ~ 21.5）→ 用对数 y 轴
- steps 跨数个量级（64 ~ 1.43e+05）→ 用对数 y 轴
- batch_tokens_M 高度偏态（skew=-8.39）→ 考虑对数变换或小提琴图代替均值柱图
- gpu_days 高度偏态（skew=2.27）→ 考虑对数变换或小提琴图代替均值柱图
- train_loss 高度偏态（skew=2.07）→ 考虑对数变换或小提琴图代替均值柱图
- val_loss 高度偏态（skew=2.11）→ 考虑对数变换或小提琴图代替均值柱图
- ppl 高度偏态（skew=6.30）→ 考虑对数变换或小提琴图代替均值柱图
- grad_norm_avg 高度偏态（skew=4.91）→ 考虑对数变换或小提琴图代替均值柱图
- observed_val_loss 高度偏态（skew=2.11）→ 考虑对数变换或小提琴图代替均值柱图
- predicted_val_loss 高度偏态（skew=2.11）→ 考虑对数变换或小提琴图代替均值柱图
- abs_error 跨数个量级（1.1e-07 ~ 0.00092）→ 用对数 y 轴

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
