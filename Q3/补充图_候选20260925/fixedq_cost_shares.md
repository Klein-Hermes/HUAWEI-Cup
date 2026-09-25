# 图表契约：固定 Q 成本构成

- **核心结论：** 15 个既有固定 Q₀ 情景的总 FLOPs 由训练与注意力成本构成，质量计算成本份额为零。
- **数据源：** `Q3/04_结果/M0_budget_context_cost_shares.csv`。
- **图型与映射：** 100% 横向堆叠条形图；每行一个预算—上下文情景；横轴为占 `C_total_flops` 的百分比，训练与注意力分别使用色盲友好颜色与填充纹理。
- **统计口径：** 使用现有 `train_cost_share`、`quality_cost_share`、`attention_cost_share`，并交叉验证前两种实际成本份额分别等于 `C_train_flops / C_total_flops` 与 `C_attn_flops / C_total_flops`。所有行份额之和为 1，`C_Q_flops = 0`。
- **解释边界：** 这是描述性情景成本份额；不是新的优化结果、因果效应或结构转移结论。Q 固定为 Q₀，所以质量份额为零。
- **支持域与单位：** 预算及成本为 FLOPs；上下文长度为 tokens。数据覆盖 3 个预算 × 5 个上下文，共 15 情景。
- **导出：** SVG、301 dpi 彩色 PNG、同尺寸灰度预览；按双栏 7.2 in 宽布局。
