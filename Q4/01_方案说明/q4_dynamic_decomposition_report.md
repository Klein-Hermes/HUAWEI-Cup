# Q4 动态模型、共同支持分解与 C3/C4 年度背景

> 全量描述性建模已运行；Loss–Benchmark 仍为 BLOCKED_NO_FIT，未执行桥接拟合。

> AI 辅助说明：OpenAI Codex desktop 26.917.71314 (build 10954; prod); model ID gpt-6-luna (GPT-6 Luna); developer OpenAI; official release date 2026-09-22. 其他 AI 对 Q4 交付的参与范围仍待团队核对。

## 动态能力项

- AR(1) 状态：`estimated_descriptive`；连续月数 10；转移数 9。
- 系数为 a=13.51154200162852、rho=0.6694820585299196；HC3 区间见 `q4_ar1_dynamic.csv`。区间不校正序列相关，仅作描述。
- 扩展窗口 AR(1) 与上月持平基线各有 2 个合格的一步预测起点；误差见 `q4_ar1_backtest_summary.csv`。样本不用于年度外推。

## 规模与非规模组成分解

- 端点为 2024-11 / 2025-03；共同支持区间是两端点各自线性插值 P05–P95 参数量区间的交集 log10(B)=[0.0920, 1.3724]；端点区间内样本数 92 / 110。该中心区间降低尾部观测影响，结论不外推至区间外记录。
- 精确分解使用所有 n≥20 月份中落入共同支持区间的 645 条记录重新拟合，β=40.7755，设计矩阵秩 10/10，残差自由度 635。规模、类型、月份与总预测差由同一回归的线性对比精确加总；HC3 区间适用于这些拟合对比，并固定早晚期经验协变量分布。
- 观测变化另以共同支持两组均值差给出 HC3 区间。`arithmetic_closure_residual` 是观测变化减去模型预测变化的算术闭合检查；由于端点月份虚拟变量进入同一 OLS 且端点记录参与拟合，该值按正规方程预期为机器精度下的 0，不代表未解释因素，故只报点值并检查加总误差。贡献占比仅在观测变化、规模项和非规模总项的 95% 区间均不含 0 时显示；占比为点值比率，不另构造区间。
- 贡献项与系数见 `q4_scale_tech_contribution.csv`、`q4_scale_tech_contribution_coefficients.csv`；共同支持和占比门记录见 `q4_scale_tech_support_audit.csv`。

## C3 年度排行榜背景

- 年表按 Source 单独列示；Open LLM Leaderboard 按 (Year, Model) 去重，存在不同指标冲突的模型年组排除。2025 标为部分年度覆盖，不纳入完整年度趋势；Historical (papers/reports) 保持独立。详见 `q4_c3_year_summary.csv` 与逐行审计表。

## C4 年度参数量、算力与数据量背景

- 只汇总 Publication date 不晚于 2025-03-14 的记录；区间文本和非数值文本不转成数值，数值列只对有限标量做中位数/P90。Confidence、Numerical format、Training compute estimation method、原始值和来源说明均在行审计中保留。
- 年度 C4 只作宏观背景，不与 C1/C8 候选作个体连接，也不直接充当 12/24 月算力增长率。主年表列参数量、训练算力和数据量的标量描述及 Confident 标量计数；置信度分层表保留类别差异。
- 年表、置信度分层和截止日期行审计见 `q4_c4_year_summary.csv`、`q4_c4_confidence_summary.csv`、`q4_c4_cutoff_row_audit.csv`。
