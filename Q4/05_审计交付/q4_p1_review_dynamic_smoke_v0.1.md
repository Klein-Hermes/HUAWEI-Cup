# Q4 P1 初次复核回执：动态分解最小实现

## 范围与结论

只读复核动态分解脚本、合同 v0.5、smoke 表与报告、smoke manifest，以及对应输入字段；未运行脚本或模型，未修改文件。

**状态：FAIL。** 主要线性对比和 HC3 计算可复核；发现两项 P1 与两项 P2。全量运行暂停，待修复并复审。

## 复核快照（SHA-256）

- 复核时脚本：`2EB679BF0B87C08616B50E1F921A0063132FBFFEC7D53030C805F439AFDC77CB`
- 合同 v0.5（修正参数量与闭合项说明前）：`06C463E40AE215DEB601F34D4CA9640AB358EDC93EB32661A159FDBC11B05E2F`
- `q4_task_aggregate.csv`：`E51C5516A8E1A3E7A451D726EEEB7699B324A9BD765306462C408A62E9935C18`
- `q4_frontier_monthly.csv`：`0678B8F7FD781528F13C97702D06F4DB47E8F1F6D637BF21E82BF0159F86E712`
- C3：`AB2B5715B2945EBDEEABFB18E8D040F7C68D8C76BD02C01E446959218BFCD184`
- C4：`0B98A01BCB8D96958745B6FBE505416C38F719948D9029B38FF6233425C0E1C1`
- Smoke manifest：`0DAA91920FE65EB5DBCB8B24F5B0126EDC55A38959704B2638021642D1EBB1AD`

## 主要证据

- AR(1) smoke 有 10 个连续月份、9 个转移、7 个残差自由度；持平基线取训练窗口最后一个已观测值。HC3 sandwich 公式和区间说明符合合同。
- 共同参数支持为 `log10(B)=[0.0920, 1.3724]`，早晚端点样本 92/110，smoke 拟合样本 202，设计矩阵秩 7/7。
- OLS 含月份固定效应且端点行都参与拟合，按正规方程端点组内均值残差为零；smoke 的观测变化与预测变化仅有 `5.77e-15` 闭合差。因此 `unexplained_residual` 命名会误导。
- 合同要求 C4 `Parameters` 按年摘要；当时脚本只解析训练算力和数据量，参数字段仅原样保留。
- C3 来源隔离及去重审计、C4 2025-03-14 发布日期截止与行级出处保留基本符合计划。

## P2 建议

- Smoke manifest 未登记导入的 `q4_modeling_analysis.py` 类型映射依赖哈希。
- AR(1) 设计矩阵秩不足时当前实现抛异常，建议按合同输出 `not_estimable`。

## 返工状态

后续版本将闭合项改为 `arithmetic_closure_residual` 并明确只是算术检查，补入 C4 参数量标量解析和年度统计、登记类型映射脚本哈希，并处理 AR(1) 秩不足的降级状态。上述修订须重新 P1 复核。
