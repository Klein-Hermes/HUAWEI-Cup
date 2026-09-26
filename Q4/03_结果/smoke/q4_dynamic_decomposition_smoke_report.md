# Q4 动态模型、共同支持分解与 C3/C4 年度背景

> 最小范围运行：C3 限于 2024 年 OLLB，C4 限于截止日前的 2024 年记录，分解只用预设早晚月份。此包用于 P1 代码与数据接口复核，不作为全量结果。

> AI 辅助说明：OpenAI Codex desktop 26.917.71314 (build 10954; prod; updater checked 2026-09-25, up_to_date); developer OpenAI; model selector name: ; model version/publication date: . 模型选择器名称、模型版本和发布日期按用户指示留空。

## 动态能力项

- AR(1) 状态：`estimated_descriptive`；连续月数 10；转移数 9。
- 系数为 a=13.51154200162852、rho=0.6694820585299196；HC3 区间见 `q4_ar1_dynamic_smoke.csv`。区间不校正序列相关，仅作描述。
- 扩展窗口 AR(1) 与上月持平基线各有 2 个合格的一步预测起点；误差见 `q4_ar1_backtest_summary_smoke.csv`。样本不用于年度外推。

## 规模与非规模组成分解

- 早晚端点为 2024-11 / 2025-03；共同参数区间 log10(B)=[0.0920, 1.3724]；共同支持端点样本数 92 / 110。
- 拟合记录数 202，设计矩阵秩 7/7，残差自由度 195。规模、类型、月份与总预测差由同一回归的线性对比精确加总；HC3 区间适用于这些拟合对比，并固定早晚期经验协变量分布。
- 观测变化另以共同支持两组均值差给出 HC3 区间。`arithmetic_closure_residual` 是观测变化减去模型预测变化的算术闭合检查；由于端点月份虚拟变量进入同一 OLS 且端点记录参与拟合，该值按正规方程预期为机器精度下的 0，不代表未解释因素，故只报点值并检查加总误差。贡献占比仅在观测变化、规模项和非规模总项的 95% 区间均不含 0 时显示；占比为点值比率，不另构造区间。
- 贡献项与系数见 `q4_scale_tech_contribution_smoke.csv`、`q4_scale_tech_contribution_coefficients_smoke.csv`；共同支持和占比门记录见 `q4_scale_tech_support_audit_smoke.csv`。

## C3 年度排行榜背景

- 年表按 Source 单独列示；Open LLM Leaderboard 按 (Year, Model) 去重，存在不同指标冲突的模型年组排除。2025 标为部分年度覆盖，不纳入完整年度趋势；Historical (papers/reports) 保持独立。详见 `q4_c3_year_summary_smoke.csv` 与逐行审计表。

## C4 年度参数量、算力与数据量背景

- 只汇总 Publication date 不晚于 2025-03-14 的记录；区间文本和非数值文本不转成数值，数值列只对有限标量做中位数/P90。Confidence、Numerical format、Training compute estimation method、原始值和来源说明均在行审计中保留。
- 年度 C4 只作宏观背景，不与 C1/C8 候选作个体连接，也不直接充当 12/24 月算力增长率。主年表列参数量、训练算力和数据量的标量描述及 Confident 标量计数；置信度分层表保留类别差异。
- 年表、置信度分层和截止日期行审计见 `q4_c4_year_summary_smoke.csv`、`q4_c4_confidence_summary_smoke.csv`、`q4_c4_cutoff_row_audit_smoke.csv`。
