# Q4 模型合同补充 v0.7：C1 直接分数预测与 Loss–Benchmark 桥接分轨

**适用范围：** 仅适用于 `Q4/06_修订推进_20260926/` 中的本轮修订。它补充并在预测准入问题上取代历史合同 v0.6 第 5 节，不改写已冻结的 Q4 基线。

## 1. 修订目的

历史合同把 12/24 月预测同时要求为：C1 月度表现序列、年度滚动验证、未来算力情景，以及 Loss–Benchmark 桥接或其他能力映射。若预测目标直接定义为 C1 分数，这会把直接观测分数预测与 Loss 到 Benchmark 的换算混成一条链。本补充将两项交付拆为独立路径，各自保留自己的准入条件和状态。

## 2. 路径 A：无条件直接 C1 分数趋势

- **目标量：** 当前 Q4 实现使用每月新提交、C2 明确标记 `Open Weights=Yes` 的模型中，C1 `Average` 原始分数（0–100）的第 90 百分位。它是从 C1 分数字段直接构造的月度统计量，不是个体模型分数预测，也不是由 Loss 转换得到。
- **输入与方法：** 仅从同源的 C1 分数与提交日期构造该序列。可在序列长度允许时直接比较时间序列模型和持平基线；不读取 Loss 来生成 C1 分数，也不以 Loss 门控状态作为该路径的准入条件。
- **解释边界：** 这是 C1 分数序列的无条件时间趋势，不表示“算力增长放缓”情景，也不能替代对算力变化条件下模型能力的预测。若报告该序列，须明确称为 C1 分数趋势/前沿描述。
- **准入依据：** 只根据目标分数定义、连续时间支持、对应预测期限的滚动验证起点和预测区间方法决定是否可估计。若证据不足，输出不可估计状态及原因，点预测和区间留空。
- **本轮实证状态：** `q4_frontier_monthly.csv` 只有 10 个有观测月份（2024-06 至 2025-03），跨度 9 个月；12/24 月年度滚动起点均为 0。AR(1) 扩展窗口只有 2 个一步起点，MAE 为 0.821，略高于持平基线 0.815。较宽松的一步比较有 5 个目标月，last-value MAE 为 0.757、线性趋势 MAE 为 3.596；这些一步诊断不能替代 12/24 月验证。
- **结论：** 12/24 月直接 C1 分数趋势预测均为 `not_estimable_from_supplied_history`，目标月仍为 2026-03 与 2027-03；不得报告数值点预测或区间。这个状态由 C1 历史长度和对应期限验证支持不足导致，与 Loss 桥接是否通过无关，也不构成算力放缓情景结论。

## 3. 路径 B：算力增长放缓条件下的直接 C1 分数预测

- **问题含义：** 若要回答算力增长放缓情景下的能力预测，可将 C1 `Average` 原分数作为响应量，直接建立分数与经核验算力情景的关系；这条路径不必经过 Loss–Benchmark 桥接。
- **所需证据：** 需先给出有来源、单位、时间范围和数值定义的未来算力情景；并需有可与 C1 分数对齐的算力观测，或一个事先定义、经对应期限留出验证的直接分数—算力关系。还需有与 12/24 月预测期相称的验证起点及区间方法。
- **现有数据限制：** C4 与 C1 的精确模型名称交集为 0，C4 只作为年度宏观背景，不能形成 C1 个体分数—算力配对或未来算力增长路径。Q2 M0 及 B9/B10 的 Loss 标度/curated/estimated 外推不是实测前沿算力增长率，不能替代未来算力情景。
- **本轮状态：** 12/24 月均为 `not_estimable_from_supplied_inputs`，不报告数值点预测或区间。该状态来自未来算力情景和 C1 可验证分数—算力关系缺失，并独立于 Loss 桥接。

## 4. 路径 C：Loss–Benchmark 桥接

- **目标量：** 仅当研究问题明确要求将一个有定义的 Loss 观测转换成 C1/OLLB Benchmark 分数时，才进入此路径。C8 百分位指数不等于 Benchmark 原始分数，不能作为桥接响应。
- **预拟合门控：** 沿用 `q4_model_contract_v0.6.md` 第 4 节；门控必须先核对 Loss 单位、评测数据与切分、tokenizer/预处理、Benchmark 定义、模型 revision/运行身份、去重后的独立支持，以及事先定义的目标 Loss 网格。任一必要证据缺失或目标超出实测支持，拟合必须拒绝执行。
- **本轮实证状态：** `q4_loss_benchmark_feasibility.csv` 与门控 manifest 为 `BLOCKED_NO_FIT` / `fit_executed=false`。C5、C6 各有 7 条高可比标签记录，但跨表 7 条完全重复；合并去重后仍只有 7 条同一 Pythia 来源家族提示的观测，已核验 revision/版本数为 0。Loss 单位、评测集/切分和 tokenizer/预处理仍未知，且没有获批的未来目标 Loss 网格。高可比记录的 Loss 范围为 2.0933–2.5978；Q2 16 个终点候选中有 13 个数值落入范围，但这些不是获批预测目标，单位和协议亦未核实。
- **结论：** 保持 `BLOCKED_NO_FIT`；没有执行拟合，不输出桥接系数或误差。现有公开来源只支持部分字段语义/训练 token 数，不能补足上述行级协议、版本和目标支持，也未提供适用于本目标的可核验映射误差，因此本轮不以文献替代桥接结果。

## 5. 三条路径的报告规则

1. 分别报告 `direct_C1_score_trend_status`、`compute_conditioned_C1_score_status` 与 `loss_benchmark_bridge_status`，不得合并成一个“预测完成/失败”状态。
2. 无条件 C1 分数趋势不得因 Loss 桥接受阻而被标为受阻；若 C1 时间序列支持不足，应报告其独立的 `not_estimable_from_supplied_history`。它也不得被表述为算力放缓条件预测。
3. 算力条件预测可以直接以 C1 分数为响应，不要求 Loss 桥通过；但必须有经核验的未来算力情景、可验证的 C1 分数—算力关系和匹配的预测期验证。任一项缺失时报告 `not_estimable_from_supplied_inputs`。
4. Loss 桥接不得因 C1 分数趋势或直接分数—算力模型可建模而视为通过；没有完成桥接门控时，仍报告 `BLOCKED_NO_FIT`。
5. 本轮结论为：无条件 C1 分数趋势和算力条件 C1 分数预测分别因不同证据不足而不可估计；Loss 桥接已单独审查并继续阻断。任何后续数值化都须由对应路径的新证据单独解锁。

## 6. 追溯文件

- C1 直接分数与回测：`Q4/03_结果/results/q4_frontier_monthly.csv`、`q4_forecast_feasibility.csv`、`q4_ar1_expanding_backtest.csv`、`q4_ar1_backtest_summary.csv`、`q4_forecast_backtest_1month_summary.csv`。
- 算力情景边界：`Q4/01_方案说明/q4_model_contract_v0.6.md` 第 2、5 节、`Q4/01_方案说明/q4_modeling_report.md` 的 C4 背景说明，以及 `Q4/05_审计交付/q4_next_steps_plan.md` 中对 Q2 M0/B9/B10 不可充当前沿实测算力增长率的记录。
- Loss 门控与逐行证据：`Q4/03_结果/results/q4_loss_benchmark_feasibility.csv`、`q4_loss_benchmark_feasibility_manifest.json`、`q4_loss_bridge_pair_audit.csv`、`q4_loss_bridge_target_audit.csv`、`q4_loss_benchmark_feasibility_report.md`。
- 生成链：`Q4/02_代码/scripts/q4_modeling_analysis.py` 直接从月度 C1 分数序列构造前沿与预测可行性表；它读取 Loss 门控用于汇总报告和输入追溯，不用 Loss 计算 C1 分数预测。`q4_loss_bridge_feasibility.py` 是单独的预拟合门控入口。
