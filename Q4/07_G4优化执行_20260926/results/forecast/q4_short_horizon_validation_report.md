# G4 短期限预测能力诊断

**结论标签：** 仅为 1–4 个月短期限滚动诊断；不验证 12/24 个月预测技能。

- 原始月数：10（2024-06 至 2025-03）；
- 每月模型数至少 10 的有效月份：8；
- 有效月份中跳过的日历月：2；
- 训练最少 3 个有效月份，按精确日历月回溯预测；缺月不会压缩成相邻月份。

## 分期限回测

| 期限（月） | 方法 | 有效折数 | MAE（Average P90 点） | RMSE（点） | 目标月范围 |
|---:|---|---:|---:|---:|---|
| 1 | last_value | 5 | 0.757 | 0.924 | 2024-11–2025-03 |
| 1 | linear_time_trend | 5 | 3.596 | 3.670 | 2024-11–2025-03 |
| 2 | last_value | 4 | 1.053 | 1.209 | 2024-12–2025-03 |
| 2 | linear_time_trend | 4 | 5.612 | 5.721 | 2024-12–2025-03 |
| 3 | last_value | 3 | 1.145 | 1.170 | 2025-01–2025-03 |
| 3 | linear_time_trend | 3 | 8.221 | 8.338 | 2025-01–2025-03 |
| 4 | last_value | 2 | 2.007 | 2.087 | 2025-02–2025-03 |
| 4 | linear_time_trend | 2 | 10.782 | 10.809 | 2025-02–2025-03 |

有效折数随期限增长迅速减少；这些小样本结果只能比较短期基线，不足以证明年度预测能力或校准任何区间。主情景模型的一步 MAE 和持平基线仍以 `q4_forecast_one_step_validation.csv` 为准。

## 可复现入口

`& "C:\Users\86147\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "D:\PycharmProject\HUAWEI-Cup\Q4\07_G4优化执行_20260926\scripts\q4_short_horizon_validation.py"`

输入哈希：`54553f1909546dc9988b74dfe21d6eb9d98b182b72ec19dde5b3d4f0567b34f8`；脚本哈希：`a5e91c4aa6373c507f64dc28279d7e2d15250a654b6b192de22438e275e33bfa`。

逐折明细、汇总和环境信息分别见 `q4_short_horizon_validation_folds.csv`、`q4_short_horizon_validation_summary.csv`、`q4_short_horizon_validation_manifest.json`。
