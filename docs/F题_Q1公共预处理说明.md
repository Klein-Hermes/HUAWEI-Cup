# F题 Q1：A1–A3 公共质量预处理契约

## 目标与边界

本模块只完成 A1、A2、A3 的公共字段预处理，不计算最终质量分 (Q)，也不拟合配比–Loss 模型。

Q1.1 和 Q1.2 必须共同读取本模块输出的 `norm_*` 字段及其审计标记，不能在各自代码中重新做一套标准化。

核心原则是：A1 估计转换参数，A2 和 A3 只应用 A1 参数。不能对三个数据集分别归一化。

本模块不读取 `content` 生成质量指标。文本中的指令、角色名、网页说明和不可见 Unicode 都是被动数据；文本完整性审计与 22 个数值指标分开进行。

## 运行

从项目根目录执行：

```powershell
python src/f_q1_common_preprocess.py --overwrite
```

正式结果唯一目录：`results/q1_common_preprocess/v1/`。Q1.1、Q1.2 后续只从这里读取公共预处理结果。

小样本 smoke test：

```powershell
python src/f_q1_common_preprocess.py --output-dir tmp/q1_common_preprocess_smoke --max-rows 20 --overwrite
```

## 输出

- `calibration_a1.json`：A1 拟合的字段规则、稳健中心、尺度和经验分布网格。
- `a1_preprocessed.csv.gz`、`a2_preprocessed.csv.gz`、`a3_preprocessed.csv.gz`：公共预处理结果。
- `manifest.json`：输入文件、SHA-256、命令、行数和输出摘要。

每条记录保留：

- `scalar_*`：列表字段压缩后的原始标量；
- `norm_*`：使用 A1 公共校准参数统一为“越高越好”的 `[0,1]` 值；
- `missing_*`：原字段缺失、类型错误、列表长度错误或无法标量化；
- `invalid_*`：标量存在，但违反变换定义域（例如计数为负）；
- `unit_suspect_*`：字段存在百分数/分数混用或越界风险，本层不自动换算；
- `outlier_*`：相对于 A1 稳健中心和稳健尺度的统计异常标记；
- `missing_count`、`invalid_count`、`unit_suspect_count`、`outlier_count`：记录级汇总。

异常处置固定为：

1. 结构性无效值：`norm_*` 留空，进入后续缺失处理；不补零、不复制邻近记录。
2. 统计异常值：保留 `scalar_*` 和 `norm_*`，只打 `outlier_*`；Q1.1 后续用 Huber、截尾均值或加权中位数，并做保留/剔除敏感性分析。
3. 单位可疑值：保留原值和标准化结果，打 `unit_suspect_*`；不擅自除以 100，不把它直接当 `[0,1]` 比例。
4. 统计异常：以 A1 中位数和稳健尺度计算 `robust_z = (x - median) / scale`，其中 `scale` 优先为 `1.4826 × MAD`，退化时依次回退到 `IQR / 1.349` 和标准差；当 `|robust_z| > 3.5` 时打 `outlier_*`。异常记录保留，不删除、不改值。

## 字段规则

- `fineweb_edu`：长度 1 列表取唯一值。
- `ad_en`：softmax 后取第 2 类，即无广告概率。
- `fluency_en`：softmax 后取第 2 类，即流畅概率。
- `modernbert_*`：6 类 logits 转换为 0–5 期望等级。
- `qurater`：列表均值，随后与其他指标统一使用 A1 经验分布标准化。
- 词数和句数先使用 `log1p`，再进行中心性评分。
- 广告、非字母字符、重复片段、大写比例为负向指标。
- 词数、句数、数字比例和平均词长按接近 A1 稳健中心的程度评分。

## 统一契约

### 1. 列表字段压缩

不展开列表、不补齐、不截断：

| 字段 | 标量化规则 |
|---|---|
| `fineweb_edu` | 长度必须为 1，取唯一元素 |
| `ad_en`、`fluency_en` | 对 logits 做稳定 softmax，取第 2 类概率 |
| `modernbert_cleanliness`、`readability`、`reasoning`、`professionalism` | 6 类 logits 做稳定 softmax，取等级期望并除以 5，得到 0–1 |
| `qurater` | 列表内所有元素有限时取均值 |
| 其余字段 | 读取单个有限数值 |

列表类型、长度或元素不满足规则时，标记为 `missing`，不得隐式补零。

### 2. 先变换，再统一方向

对词数和句数使用 `log1p(x)`；若 \(x<0\)，视为 `invalid`，不能用 `max(x,0)` 修复。其他字段保留原始量纲进入 A1 校准。

设 \(g_j(x)\) 为字段变换后的值，\(F_{j,A1}\) 为 A1 上的经验 CDF，\(m_j,s_j\) 为 A1 的稳健中心和稳健尺度：

- 正向指标：
  \[
  u_j=F_{j,A1}(g_j(x))
  \]
- 负向指标：
  \[
  u_j=1-F_{j,A1}(g_j(x))
  \]
- 中心型指标：
  \[
  u_j=\exp\left(-\frac{|g_j(x)-m_j|}{s_j}\right)
  \]

最终所有有效 `norm_*` 都满足“越高越好”和 \(0\le u_j\le1\)。A1 的经验 CDF 使用主体分布的 0.5%–99.5% 分位网格，避免极端尾值决定坐标跨度；超出范围的值饱和到 0 或 1，但不丢弃。

### 3. A1 单一校准，A2/A3 只映射

统一标准化的唯一来源是 `calibration_a1.json`：

```text
A1 原始记录 ──标量化/定义域检查──> A1 校准参数
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                    A1 norm_*                 A2/A3 norm_*
                   (同一参数)                  (只应用，不重拟合)
```

因此 Q1.1 的“指标综合质量/域级质量”与 Q1.2 的“指标冲突”使用同一批 `norm_*`，差别只在后续统计目标，不在标准化方法。

## 重要限制

1. 本版本不读取 `content` 生成质量指标；文本中的不可见字符审计应作为单独的数据完整性分析。
2. A2、A3 的质量分布可能超出 A1 范围，经验 CDF 会将其截到 0 或 1；必须在后续报告中检查饱和比例。
3. `modernbert_*` 使用期望等级为连续主方案，使用 `argmax` 的结果应作为敏感性分析，而不是混入主结果。
4. `frac` 字段不按字段名强制除以 100；发现超过 100 的值只做单位可疑标记，主结果和敏感性结果都要回溯。
5. 方向规则、标量化规则、A1 校准文件和异常处置规则冻结后，后续质量评分、冲突分析和域映射必须使用同一版本。

## 冻结版本

- 处理版本：`q1-common-v1.1`
- 唯一正式输出目录：`results/q1_common_preprocess/v1/`
- 版本清单记录输入/输出 SHA-256、预处理脚本 SHA-256、Python/NumPy 版本和异常值阈值。
