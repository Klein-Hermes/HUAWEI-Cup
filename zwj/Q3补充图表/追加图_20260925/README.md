# Q3 追加图件归档（2026-09-25）

本目录保存两张已生成 Q3 补充候选图的归档副本。图件原件仍保留在 Q3/04_结果/figures/supplementary_20260925/；本次只复制，不移动、不覆盖源图件，也未触碰 zwj/Q3补充图表 中原有 P1–P4 图件。

## S4：质量增量成本曲线

- 彩色 PNG：S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost.png
- 灰度 PNG：S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost_grayscale.png
- SVG：S4_质量增量成本曲线/result_q3_supp_quality_incremental_cost.svg
- 图表契约：S4_质量增量成本曲线/图表契约.md
- 复现入口说明：S4_质量增量成本曲线/复现入口说明.md
- 数据源：Q3/04_结果/Q_cost_increment_envelope.csv；解释参考 Q3/04_结果/Q3_cost_sensitivities_report.md。36 行、三种 g(Q)、D_B1 最小/中位/最大三面板；Q0=0.4984781048 的 C_Q=0 是精确零值，未绘入对数轴。仅表达题面代理成本，不表示 Loss 收益、最优 Q 或新优化结果。

## Q3 建模/求解分支图

- 彩色 PNG：flow_q3_model/flow_q3_model.png
- 灰度 PNG：flow_q3_model/flow_q3_model_grayscale.png
- SVG：flow_q3_model/flow_q3_model.svg
- 图表契约：flow_q3_model/图表契约.md
- 复现入口说明：flow_q3_model/复现入口说明.md
- 图示区分已计算的固定 Q0/M0 分支、βQ 未知的条件 Q-only 分支，以及完整 Q/p 联合优化尚未满足的上游证据门槛；不表示已求得完整联合最优。

## 验证边界

两图的源生成脚本已在原任务中通过运行与脚本布局门禁，彩色和灰度图已做目检；归档后逐文件比较源/目标 SHA-256，并检查 PNG 尺寸及 DPI。PNG 为 300.99 DPI，SVG 保留矢量路径和文本。Q3 全局 figure_audit.py --questions q3 --strict 实际退出码为 1，因全局目录审计的图类数量要求与本补充 result 候选范围不匹配，并报告既有顶层图问题；本归档不声称全局审计通过。

逐文件字节数和 SHA-256 见 归档清单.json。清单不列自身哈希，避免自引用递归；其余归档文件均列入。
