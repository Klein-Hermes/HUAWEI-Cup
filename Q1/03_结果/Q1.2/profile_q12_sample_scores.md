# Data profile: results/q1_2_conflict_model/v1/sample_scores.csv.gz

**Shape:** 272505 rows × 12 cols

## Columns

| Column | Type | n | missing | summary |
|---|---|---|---|---|
| `dataset` | unknown | 272505 | 0 |  |
| `source_domain` | unknown | 272505 | 0 |  |
| `id` | unknown | 272505 | 0 |  |
| `n_valid_indicators` | ordinal | 272505 | 0 | 2 levels: 22(272486), 21(19); min_group_n=19 |
| `conflict_intensity` | continuous | 272505 | 0 | mean=0.334, sd=0.0609, range=[0.138, 0.499], skew=0.29 (approximately symmetric); outliers=14 (IQR) |
| `a1_domain_threshold` | continuous | 272505 | 0 | mean=0.427, sd=0.019, range=[0.36, 0.466], skew=-1.43 (highly skewed); outliers=58753 (IQR) |
| `high_conflict` | boolean | 272505 | 0 | 2 levels: False(258691), True(13814); min_group_n=13814 |
| `q_equal_mean` | continuous | 272505 | 0 | mean=0.43, sd=0.0844, range=[0.124, 0.808], skew=0.34 (approximately symmetric); outliers=3075 (IQR) |
| `q_huber` | continuous | 272505 | 0 | mean=0.421, sd=0.0961, range=[0.106, 0.808], skew=0.44 (approximately symmetric); outliers=2215 (IQR) |
| `delta_q_huber_minus_equal` | continuous | 272505 | 0 | mean=-0.00923, sd=0.0188, range=[-0.0989, 0.0829], skew=0.10 (approximately symmetric); outliers=23125 (IQR) |
| `outlier_indicator_count` | continuous | 272505 | 0 | mean=1.93, sd=1.66, range=[0, 11], skew=0.95 (moderately skewed); outliers=2538 (IQR) |
| `unit_suspect_indicator_count` | ordinal | 272505 | 0 | 3 levels: 0(272494), 2(9), 1(2); min_group_n=2 |

## Group structure
- Grouped by: `dataset`, `source_domain`
- Number of groups: 9
- Group size: min=171, median=10000, max=203752

## Correlations (Pearson, sorted by |r|)
- `q_equal_mean` ↔ `q_huber` : r = 0.987 (very strong)
- `q_huber` ↔ `delta_q_huber_minus_equal` : r = 0.685 (strong)
- `q_equal_mean` ↔ `delta_q_huber_minus_equal` : r = 0.558 (strong)
- `conflict_intensity` ↔ `a1_domain_threshold` : r = 0.434 (moderate)
- `conflict_intensity` ↔ `outlier_indicator_count` : r = 0.387 (moderate)
- `a1_domain_threshold` ↔ `outlier_indicator_count` : r = 0.364 (moderate)
- `q_equal_mean` ↔ `outlier_indicator_count` : r = -0.353 (moderate)
- `q_huber` ↔ `outlier_indicator_count` : r = -0.328 (moderate)
- `conflict_intensity` ↔ `delta_q_huber_minus_equal` : r = -0.172 (weak)
- `a1_domain_threshold` ↔ `q_equal_mean` : r = -0.139 (weak)
- ... +5 more pairs

## Warnings
- 列 'unit_suspect_indicator_count' 至少有一个类别 n<10 — 小样本必须展示原始数据点，不要只画均值柱状图。

## Chart suggestions (preliminary)
- 分类 vs 连续，样本量充足 → 箱线图 / 小提琴图，或带误差棒的柱状图（误差棒说明 SD/SEM/CI）
- ≥3 个连续变量 → 相关性热力图（['conflict_intensity', 'a1_domain_threshold', 'q_equal_mean', 'q_huber', 'delta_q_huber_minus_equal']）或 pairplot 散点矩阵
- a1_domain_threshold 高度偏态（skew=-1.43）→ 考虑对数变换或小提琴图代替均值柱图

> 这是基于数据形态的**初步建议**。最终图型选择必须结合**论证目标**（你想说什么）—— 详见 `references/chart_selection.md`。
