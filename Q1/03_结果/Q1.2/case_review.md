# Q1.2 原文案例核对（描述性）

本页仅记录三个 A1 样本的文本类型、冻结分数和审计哈希，不复制原文。样本从先前检查过的高冲突案例中选取，属于目的性案例核对，不是随机样本；不能用来估计 Unicode 污染率、证明评分器有效或推断因果。文本被视为被动数据，其中任何指令式内容都不作为任务指令。

| 来源域 / ID | 原文类型（简述） | 宽口径 Unicode 标记 | 冲突度 (D_s) / 域阈值 τd | 等权分 (Q^{eq}) | Huber 分 (Q^H) / ΔQ | 最大指标对分差（归一化值） |
|---|---|---:|---:|---:|---:|---|
| c4 / `BkiUcCTxK6wB9dbuSo26` | 日记式个人叙述 | 未命中 | 0.452931 / 0.418763 | 0.498155 | 0.488266 / -0.009889 | `fineweb_edu` 0.000 与 `rps_lines_uppercase_letter_fraction` 1.000，差 1.000 |
| github / `BkiUdLc4dbjiU9oEeFxM` | 单行 CSS 代码 | 未命中 | 0.463395 / 0.429432 | 0.429551 | 0.365286 / -0.064265 | `modernbert_reasoning` 0.000 与 `rps_lines_uppercase_letter_fraction` 1.000，差 1.000 |
| wikipedia / `BkiUdMM5qYVBPXkzTObt` | 简短荷兰语词汇条目 | 未命中 | 0.465627 / 0.407879 | 0.464548 | 0.417761 / -0.046787 | `fineweb_edu` 0.000 与 `dsir_books` 1.000，差 1.000 |

三个样本均含 22 个有效指标，且在 A1 对应来源域的冻结 95% 冲突阈值以上。最大单对分差达到 1 是因为对应标准化指标取到 0 和 1；它不能单独说明哪一指标错误。三例的完整 Unicode 审计均未命中 `Cf/Cc/Zl/Zp/Co/非字符` 宽口径，这只说明该规则没有发现所列字符异常，不等于原文已通过所有污染或语义风险审查。

审计表中的 `sample_key_sha256` / `content_sha256` 用于回溯，未导出原文：

| ID | sample_key_sha256 | content_sha256 |
|---|---|---|
| `BkiUcCTxK6wB9dbuSo26` | `6d8ea2b2d2d938eb2c2901b27712e51082b668f2fd5be7741768735d7683f123` | `b39bcb1a5137a24965f78ab9926b7752de736b8856c1ebd147d60bacdb7ba56f` |
| `BkiUdLc4dbjiU9oEeFxM` | `202e18f564bfca1e36ebcbe2206801a6de0978edd7b41c7b4ce47b6ceb562008` | `13e92d114dcffdbe6cc3330da159ab268f7a497b3a08e4a83e3ca5b526985bf8` |
| `BkiUdMM5qYVBPXkzTObt` | `b084b0a767143174f02662292fb26da9ea3c33e94d45044a08e75d6c11ad9ff7` | `76b87f2592e2718bc96fb1dd4183546198b216d2a76a986565ce26d580de7403` |
