# Data profile: 中文题目/F题/real_attachments/B_scaling_laws/supplementary_NQ_experiment.csv

**Shape:** 360 rows × 5 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `experiment_id` | text | 360 | 0 |  |
| `N_params_B` | continuous | 360 | 0 | mean=3.11, sd=3.79, range=[0.07, 12], skew=1.33 (highly skewed); outliers=40 (IQR); -> log axis |
| `D_tokens_B` | ordinal | 360 | 0 | 5 levels: 10(72), 50(72), 150(72), 300(72), 600(72); min_group_n=72 |
| `Q_score` | continuous | 360 | 0 | mean=0.537, sd=0.316, range=[0.1, 1], skew=0.11 (approximately symmetric) |
| `val_loss` | continuous | 360 | 0 | mean=2.62, sd=0.346, range=[2.02, 3.72], skew=0.68 (moderately skewed); outliers=3 (IQR) |

## Group structure
- Grouped by: `Q_score`
- Number of groups: 8
- Group size: min=45, median=45, max=45

## Correlations (Pearson, sorted by |r|)
- `N_params_B` ↔ `val_loss` : r = -0.574 (strong)
- `Q_score` ↔ `val_loss` : r = -0.332 (moderate)
- `N_params_B` ↔ `Q_score` : r = 0.000 (negligible)

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['N_params_B', 'Q_score', 'val_loss']）或 pairplot 散点矩阵
- N_params_B 跨数个量级（0.07 ~ 12）→ 用对数 y 轴

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
