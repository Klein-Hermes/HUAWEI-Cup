# Q3 图表契约

## `raw_q3_loss_by_q_distribution`

- 核心结论：展示 B6、B8 校准与外推来源的观测损失分布随离散 Q 水平的变化；箱线图保留组内分布，点为抽样观测。
- 证据与风险：原始数据；不得解释为单独 Q 因果效应，因为 N/D 同时变化。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `raw_q3_q_loss_relation`

- 核心结论：展示来源内 Q 与验证损失的观测配对，并用颜色编码 N。
- 证据与风险：原始关系；识别共变因素，不作单变量效应推断。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `raw_q3_b1_source_support`

- 核心结论：展示 B1 N/D 网格与 B6/B8 calibrated 来源支持范围及交集。
- 证据与风险：原始支持域；决定迁移优化的可用域。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `process_q3_fit_cv`

- 核心结论：对比来源全曲面和 B1 参数锚定模型的留一 Q 水平预测误差。
- 证据与风险：过程验证；锚定列依赖未验证的 Q 数值桥接。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `process_q3_feasibility`

- 核心结论：展示 Q=Q0 下预算与上下文组合的支持域可行候选数。
- 证据与风险：过程域；不将零可行点误写成质量转移。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `process_q3_kappa_estimates`

- 核心结论：对照来源曲面 κ 与锚定 κ，展示锚定 bootstrap 区间。
- 证据与风险：参数诊断；外推来源仅列来源内点估计。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `result_q3_selected_nd`

- 核心结论：展示来源、g、预算、上下文下的条件最优 N 与 D。
- 证据与风险：结果；灰格为不可行，使用固定 p0 和未验证桥接。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `result_q3_selected_q_loss`

- 核心结论：展示条件最优 Q 与对应预测损失。
- 证据与风险：结果；不可解释为已验证 B1 效果。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `result_q3_transition_stability`

- 核心结论：展示 1000 次条件参数 bootstrap 中的配置/成本份额变化频率。
- 证据与风险：结果稳定性；边界边留空，单独视为可行域进出。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `flow_overall_model`

- 核心结论：概括题目数据、Q1–Q4 求解、验证与综合结论的依赖链。
- 证据与风险：建模流程图，不计入三类数据图配额。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

## `flow_q3_model`

- 核心结论：展示来源估计、支持域与预算优化、转移稳定性及结论限定。
- 证据与风险：Q3 方法流程图，节点对应当前分析脚本实际步骤。
- 后端与交付：Python/Matplotlib；SVG 可编辑文字，PNG 300 dpi，另存灰度预览。

