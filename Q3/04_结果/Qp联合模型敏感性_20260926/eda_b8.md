# Data profile: 中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment_large.csv

**Shape:** 1704 rows × 6 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `experiment_id` | text | 1704 | 0 |  |
| `N_params_B` | continuous | 1704 | 0 | mean=90, sd=185, range=[0.07, 700], skew=2.56 (highly skewed); outliers=240 (IQR); -> log axis |
| `D_tokens_B` | continuous | 1704 | 0 | mean=430, sd=618, range=[5, 2e+03], skew=1.68 (highly skewed); outliers=180 (IQR); -> log axis |
| `Q_score` | continuous | 1704 | 0 | mean=0.542, sd=0.321, range=[0.05, 1], skew=-0.09 (approximately symmetric) |
| `val_loss` | continuous | 1704 | 0 | mean=1.34, sd=0.698, range=[0.5, 3.37], skew=0.30 (approximately symmetric) |
| `data_type` | categorical | 1704 | 0 | 2 levels: calibrated(984), extrapolated(720); min_group_n=720 |

## Group structure
- Grouped by: `data_type`, `Q_score`
- Number of groups: 24
- Group size: min=60, median=69, max=90

## Correlations (Pearson, sorted by |r|)
- `Q_score` ↔ `val_loss` : r = 0.913 (very strong)
- `N_params_B` ↔ `val_loss` : r = -0.152 (weak)
- `D_tokens_B` ↔ `val_loss` : r = -0.088 (negligible)
- `N_params_B` ↔ `D_tokens_B` : r = -0.013 (negligible)
- `D_tokens_B` ↔ `Q_score` : r = -0.000 (negligible)
- `N_params_B` ↔ `Q_score` : r = -0.000 (negligible)

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['N_params_B', 'D_tokens_B', 'Q_score', 'val_loss']）或 pairplot 散点矩阵
- N_params_B 跨数个量级（0.07 ~ 700）→ 用对数 y 轴
- D_tokens_B 跨数个量级（5 ~ 2e+03）→ 用对数 y 轴

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
