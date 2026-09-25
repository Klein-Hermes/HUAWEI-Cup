# Q3 M3/M4/M1 图表 QA

**生成时间（UTC）：** 2026-09-24T15:48:09+00:00  
**绘图环境：** Python 3.10.20；中文字体 `SimHei`；通用报告样式。
**数据行数：** observed=1176, candidate=1176, M3=1176, M4=17640, M1 final mask=17640, optima=15。
**解释口径：** 当前候选全部来自实测 B1 网格；图中凸包只作诊断，M1 只在 `m3_search_allowed AND budget_feasible` 中选择。

## 程序版面检查与导出

- `raw_q3_observed_nd_support`：程序布局检查 PASS；问题数 0。
- 图意契约：展示全部 B1 实测 N/D 点及标准化对数空间凸包；凸包不作为插值许可证明。
  - 输出：`Q3/04_结果/figures/raw_q3_observed_nd_support.pdf`
  - 输出：`Q3/04_结果/figures/raw_q3_observed_nd_support.svg`
  - 输出：`Q3/04_结果/figures/raw_q3_observed_nd_support.png`
  - 输出：`Q3/04_结果/figures/raw_q3_observed_nd_support_grayscale.png`
- `process_q3_m3_m4_mask_audit`：程序布局检查 PASS；问题数 0。
- 图意契约：按 N 汇总 M3 支持类别，并展示 15 个情景的 M4 预算可行点数。
  - 输出：`Q3/04_结果/figures/process_q3_m3_m4_mask_audit.pdf`
  - 输出：`Q3/04_结果/figures/process_q3_m3_m4_mask_audit.svg`
  - 输出：`Q3/04_结果/figures/process_q3_m3_m4_mask_audit.png`
  - 输出：`Q3/04_结果/figures/process_q3_m3_m4_mask_audit_grayscale.png`
- `result_q3_m1_budget_optima`：程序布局检查 PASS；问题数 0。
- 图意契约：五个上下文小面板叠加实测网格、凸包诊断边界和三档预算下的 M1 最优点。
  - 输出：`Q3/04_结果/figures/result_q3_m1_budget_optima.pdf`
  - 输出：`Q3/04_结果/figures/result_q3_m1_budget_optima.svg`
  - 输出：`Q3/04_结果/figures/result_q3_m1_budget_optima.png`
  - 输出：`Q3/04_结果/figures/result_q3_m1_budget_optima_grayscale.png`

## 数据与文件检查

- 原始/支持/预算/最终掩码均按完整输入绘制；图中没有删除候选点。
- 未画误差棒或显著性标记：图示为确定性支持域、成本掩码和离散最优配置。
- 预览位图位于 `Q3/04_结果/_figure_previews/`；正式图文件位于 `Q3/04_结果/figures/`。
- 三张图已逐图检查，未见文字遮挡或裁切；复核说明见 `Q3_M3_M4_M1_图表读图复核.md`。
- 三张彩色 PNG 已通过 `check_figure.py --strict`，分辨率均为 300 dpi。
