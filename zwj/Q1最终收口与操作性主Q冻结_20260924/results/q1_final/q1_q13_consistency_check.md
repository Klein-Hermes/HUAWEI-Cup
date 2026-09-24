# Q1.3 对 Q 的真实依赖与重跑判定

## 结论

`Q1.3_RERUN_REQUIRED = false`

前提：负责人最终选择的是现有两个候选之一，且使用当前冻结的领域映射与候选数值。既有实验已分别覆盖 q_equal 与 q_huber，不因操作接口选择重复训练。

## 特征公式与代码位置

代码归档路径：`Q1/02_代码/f_q1_3_pq_extension.py`

- 第 137–173 行读取 Q1.1 领域级候选与敏感性场景；
- 第 212–220 行计算 `Q_covered(p)`；
- 第 271–312 行把该特征作为一个确定性非线性预测基函数加入 p 模型；
- 第 899–935 行记录输入、公式、参数与输出哈希。

实际公式：

`Q_covered(p) = sum(p_j * q_j for mapped j) / sum(p_j for mapped j)`

只有映射份额大于 0 的配方进入共同支持比较；未映射领域不补值、不按 Q=0 处理。

## 实际输入

- 领域映射：`中文题目/F题/real_attachments/A_data_value/domain_mapping_guide.csv`，manifest SHA-256 `4a4479f41b9f46116e688cd0c13ccdad4e687192fc1a38c4add690771ebff480`。
- q_equal 与 q_huber：`results/q1_1/v1/domain_summary.csv`，SHA-256 `41863fd2530baa32616c06cfc8c7e45e1d4118c56f15fd8d0eb0b17ec8272248`。
- Huber 宽 Unicode 敏感性：`score_sensitivity_domain.csv`，SHA-256 `b618811670fa8f1d41d01172c87014d82af1fa467eccf44a2fd8bc4aafae12b1`。
- Huber 排除 7 字段敏感性：`domain_rank_sensitivity.csv`，SHA-256 `f2ecef611c1a675d56e889d16b177de659e231c0d4e2e24a1edf6399b7a64eb5`。
- 映射设置：`direct_only = {direct}`；`direct_plus_near = {direct, near_direct}`。
- 训练/评估：A4/A5 训练，A6/A7、A8/A9、A10/A11 为此前已接触的外部评估；A12–A15 未用于模型选择或指标。

`q1_1_domain_q_candidates_used.csv` 共 24 行，覆盖 4 个 Q 场景，每个场景 6 个可映射领域：q_equal、q_huber、Huber 宽 Unicode 敏感性、Huber 排除 7 字段敏感性。

## 代码与结果追溯

- 既有 manifest：`Q1/03_结果/Q1.3/pq_extension_v1/manifest.json`，SHA-256 `8eb746ef414569ed59aa6a3b069a7236b9f01e675a42336fb0721aa0813e4259`。
- 既有 run summary：SHA-256 `0f4ce802b666187bf469ce6ca71c5ed842997698d91eb4e8e210b46ccbb015c2`。
- manifest 记录源脚本 SHA-256：`1d713013e8b3378553352b02ffae61897a991f4239e9748aeec8e883429fa558`。
- 当前归档脚本原始字节 SHA-256：`2036f1592c9d3bfcf28fa604745562095e83a4baad8a0ffb71498259edadb9a9`；将 CRLF 规范化为 LF 后为 `1d713013...`，与 manifest 完全一致。差异来自归档换行符，不是代码逻辑改变。
- 七个既有 CSV 输出的当前 SHA-256 均与 manifest 一致。

## 覆盖与建模结论

- direct-only：训练 512 条中 478 条有覆盖，34 条不进入共同支持比较。
- direct+near：训练 512 条中 507 条有覆盖，5 条不进入共同支持比较。
- `Q_covered(p)` 对线性 p 设计矩阵增加一个非线性方向，但完全由 p 与固定领域 Q 计算，不能识别独立或因果的质量效应。
- 相对二次 p-only，p+Q 在嵌套 OOF、1M 和 1B 上更差，只在已接触的 60M 评估上更好，方向不稳定。因此现有证据不支持将 p+Q 升级为通用主模型。

本任务没有重跑 Q1.3，也没有修改其任何输入或输出。
