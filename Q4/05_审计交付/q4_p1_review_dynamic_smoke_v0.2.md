# Q4 P1 复核回执：动态分解最小实现 v0.2

## 范围与结论

只读复核 `q4_dynamic_decomposition.py`、合同 v0.5、12 个 smoke CSV、smoke 报告/manifest，以及 C3/C4 输入字段与 smoke 结果；未运行或修改文件。

**状态：PASS。** P0/P1/P2 均无待返工项。此 PASS 允许按合同开展全量描述性运行，不授权 Loss–Benchmark 拟合或年度数值外推。

## SHA-256 快照

- 动态脚本：`57E8EBE3E36E4AB249A0F7E24FFB33B5F01AF77932C5D2C0F956E8026F8CDF9C`
- 合同 v0.5：`F4ABBB43C36A211CA6EC253E7ACE3E073AC658D45CBD4B2D542A9CC302A047C9`
- 类型映射依赖 `q4_modeling_analysis.py`：`0F9F641DB14FE5A8A96E78561D705E9051484082F3D39997F1FEC7DCE8A5F957`
- smoke manifest：`57DEB98BA96DB8775504355C6B6FAB032882276221D988379303EECD399F044B`
- smoke 报告：`F22A957854F7B4434523FDB6BEBF5F391894F5017F0E6CAE3D71508105A26D12`
- 聚合表、月度前沿表、C3、C4：`E51C5516A8E1A3E7A451D726EEEB7699B324A9BD765306462C408A62E9935C18`；`0678B8F7FD781528F13C97702D06F4DB47E8F1F6D637BF21E82BF0159F86E712`；`AB2B5715B2945EBDEEABFB18E8D040F7C68D8C76BD02C01E446959218BFCD184`；`0B98A01BCB8D96958745B6FBE505416C38F719948D9029B38FF6233425C0E1C1`

Smoke manifest 中 6 项输入、12 个 CSV 与报告哈希一致。

## 关键证据

- AR(1) smoke 使用 10 个连续月份、9 次转移、7 个残差自由度；扩展窗口为每个训练折要求 8 个连续月份，持平基线取该折最后观测值。秩不足时降级为 `not_estimable`，回测秩失败折单独标记。
- HC3 sandwich 协方差及分解线性对比与公式相符。共同支持区间为 `log10(B)=[0.0920,1.3724]`，早晚端点分别 92/110 条记录；smoke 矩阵秩 7/7。
- 拟合样本含端点月份固定效应，闭合项已命名为 `arithmetic_closure_residual`，仅为算术校验。smoke 闭合值 `5.773159728050814e-15`，分量加总与观测重建误差均为 `2.220446049250313e-16`。
- C3 按来源分开、按模型年去重并排除指标冲突组。2024 smoke 行数 2,671、保留 2,535、排除 68 个冲突模型年组。
- C4 解析参数量、算力与数据量有限标量，保留原始值、Confidence、Numerical format、估算方法、日期和来源说明；2024 smoke 参数量有效数 553、中位数 `7e9`、P90 `7.256e10`。发布日期晚于 2025-03-14 的记录排除于统计。
- AR(1) smoke 只有两个合格的一步回测起点，MAE 0.82069，高于持平基线 0.81465；不能据此授权 12/24 月外推。Smoke 明确标注为 P1 接口复核，不是全量结果。

## 历史返工

首次 P1 复核的 FAIL 与返工记录见 `q4_p1_review_dynamic_smoke_v0.1.md`。本回执确认其中两项 P1 与两项 P2 均已修复。
